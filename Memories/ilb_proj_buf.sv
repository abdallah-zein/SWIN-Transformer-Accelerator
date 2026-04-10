// =============================================================================
// ilb_proj_buf.sv
//
// Intermediate Layer Buffer — Projection output + Residual store
// (MSA_Out + X residual, and FFN input / FFN residual)
//
// ── Role in the Swin Block dataflow ──────────────────────────────────────
//
//   From initial_thoughts.txt:
//     MSA_Out = linear(ATTN)  →  49×96 INT8
//     Out_MSA = MSA_Out + X   →  residual addition, stored here
//     FFN input               =  Out_MSA  (read from here)
//     FFN_Out = FFN(Out_MSA) + Out_MSA  → second residual, also stored here
//
//   This buffer therefore serves THREE roles in sequence:
//
//   Phase A — PROJ WRITE:
//     Receives W_Proj output (49×96 INT8) from MMU, column by column.
//
//   Phase B — RESIDUAL ADD:
//     Controller reads X from FIB and this buffer's current content, adds
//     them externally (via the adder in full_system_top / arch diagram),
//     writes the sum back here.  This is an in-place read-modify-write.
//     After this phase the buffer holds Out_MSA = MSA_Out + X.
//
//   Phase C — FFN INPUT READ:
//     The controller reads this buffer row-by-row to load the FFN input
//     into unified_input_buf for the FFN1 computation.
//     Simultaneously this buffer's content is the residual for FFN2.
//
//   Phase D — FFN RESIDUAL ADD + STORE:
//     FFN2 output (49×96 INT8) is added to this buffer's content
//     (Out_MSA) element-by-element.  The result (final Swin Block output)
//     is written back into this buffer, then flushed to off-chip by MWU.
//
// ── Double-banking ────────────────────────────────────────────────────────
//   Phases A→B→C→D are sequential for the SAME window.  No double-banking
//   is needed within one window.  The controller controls when each phase
//   transitions.  A single flag (phase [1:0]) is exposed as an output for
//   the controller's convenience — but the buffer itself is agnostic to phase;
//   all write and read operations go to/from the same single memory.
//
// ── Residual ADD port ─────────────────────────────────────────────────────
//   Matching the pattern from ilb_score_buf and the arch adder block:
//     1. Read current content:   rmw_rd_en, rmw_addr → rmw_rd_data (next cycle)
//     2. External adder forms:   sum = rmw_rd_data + x_word (from FIB/FFN2)
//     3. Write back:             rmw_wr_en, rmw_addr, rmw_wr_data
//
// ── Buffer sizing ─────────────────────────────────────────────────────────
//   49 × 96 = 4704 bytes (INT8 stored in 8-bit cells; reads packed 4/word).
//
// ── Read protocol ─────────────────────────────────────────────────────────
//   rd_patch [5:0], rd_col [6:0] → rd_data [31:0]  (4 bytes, 1-cycle latency)
//   Used to read FFN input rows into unified_input_buf (Phase C) and
//   also for the RMW residual reads (Phase B and D).
// =============================================================================

module ilb_proj_buf #(
    parameter int N_PATCHES = 49,
    parameter int C_BYTES   = 96,
    parameter int N_ROWS    = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // ═════════════════════════════════════════════════════════════════════
    // WRITE PORT — receives Proj or FFN2+residual output column-by-column
    //
    // wr_patch_base [5:0] : first patch of burst (0,7,...,42)
    // wr_col        [6:0] : byte column (0..95)
    // wr_en               : write strobe
    // wr_data [N_ROWS-1:0]: 7 INT8 values
    // ═════════════════════════════════════════════════════════════════════
    input  logic        wr_en,
    input  logic [5:0]  wr_patch_base,
    input  logic [6:0]  wr_col,
    input  logic [7:0]  wr_data [0:N_ROWS-1],

    // ═════════════════════════════════════════════════════════════════════
    // READ PORT — 4 bytes per cycle, for FFN input loading and MWU flush
    //
    // rd_patch [5:0] : patch index (0..48)
    // rd_col   [6:0] : word-aligned byte column (0..92, step 4)
    // rd_en          : read enable
    // rd_data [31:0] : 4 packed INT8 bytes
    // ═════════════════════════════════════════════════════════════════════
    input  logic        rd_en,
    input  logic [5:0]  rd_patch,
    input  logic [6:0]  rd_col,
    output logic [31:0] rd_data,

    // ═════════════════════════════════════════════════════════════════════
    // READ-MODIFY-WRITE PORT — residual addition  (Phase B and Phase D)
    //
    // Flat word address (4-byte granularity):
    //   rmw_addr = patch * (C_BYTES/4) + col_word
    //            = patch * 24 + col_word   (col_word = 0..23)
    // Total words = 49 × 24 = 1176.  11 bits sufficient (2^11=2048 > 1176).
    //
    // Protocol: identical to ilb_score_buf RMW (2-cycle round-trip).
    //   Cycle 0: rmw_rd_en=1, rmw_addr → rmw_rd_data valid cycle 1
    //   Cycle 1: rmw_wr_en=1, rmw_addr, rmw_wr_data = adder result
    //
    // rd_en and rmw_rd_en must not be asserted simultaneously.
    // ═════════════════════════════════════════════════════════════════════
    input  logic        rmw_rd_en,
    input  logic [10:0] rmw_addr,              // 0..1175  (49 × 24)
    output logic [31:0] rmw_rd_data,
    input  logic        rmw_wr_en,
    input  logic [31:0] rmw_wr_data            // adder result written back
);

    // ── Storage: 4704 bytes as flat byte array ────────────────────────────
    localparam int BANK_BYTES  = N_PATCHES * C_BYTES;   // 4704
    localparam int WORDS       = N_PATCHES * (C_BYTES/4); // 1176

    logic [7:0] mem [0:BANK_BYTES-1];

    initial begin
        for (int i = 0; i < BANK_BYTES; i++) mem[i] = '0;
    end

    // =========================================================================
    // Write port — Proj or FFN2+residual, 7 bytes per cycle
    // =========================================================================
    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int patch = int'(wr_patch_base) + r;
                if (patch < N_PATCHES) begin
                    automatic int addr = patch * C_BYTES + int'(wr_col);
                    mem[addr] <= wr_data[r];
                end
            end
        end
    end

    // =========================================================================
    // Read port — 4-byte word read (registered, 1-cycle latency)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else if (rd_en) begin
            automatic int base = int'(rd_patch) * C_BYTES + int'(rd_col);
            rd_data <= { mem[base + 3],
                         mem[base + 2],
                         mem[base + 1],
                         mem[base    ] };
        end
    end

    // =========================================================================
    // RMW read port — for residual addition (registered, 1-cycle latency)
    //
    // rmw_addr is a WORD address (4-byte granularity).
    // Converts: byte_base = rmw_addr * 4
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rmw_rd_data <= '0;
        end else if (rmw_rd_en) begin
            automatic int bbase = int'(rmw_addr) * 4;
            rmw_rd_data <= { mem[bbase + 3],
                             mem[bbase + 2],
                             mem[bbase + 1],
                             mem[bbase    ] };
        end
    end

    // =========================================================================
    // RMW write port — residual sum written back
    // rmw_addr is a WORD address (same mapping as RMW read).
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rmw_wr_en) begin
            automatic int bbase = int'(rmw_addr) * 4;
            mem[bbase    ] <= rmw_wr_data[ 7: 0];
            mem[bbase + 1] <= rmw_wr_data[15: 8];
            mem[bbase + 2] <= rmw_wr_data[23:16];
            mem[bbase + 3] <= rmw_wr_data[31:24];
        end
    end

endmodule
