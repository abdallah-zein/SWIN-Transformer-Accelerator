// =============================================================================
// ilb_ffn1_buf.sv
//
// Intermediate Layer Buffer — FFN1 intermediate output store
// (Z = GELU( X_ffn × W_FFN1 ))
//
// ── Role in the Swin Block dataflow ──────────────────────────────────────
//
//   From initial_thoughts.txt and accelerator paper:
//     FFN step 1:  Z_raw = X_ffn × W_FFN1   →  49 × 384, INT32 accumulators
//     GELU:        Z = GELU(Z_raw)           →  49 × 384, INT8 after GCU + quant
//     FFN step 2:  Out = Z × W_FFN2          →  49 × 96,  INT32 accumulators
//
//   This buffer stores the GELU-quantised output Z (INT8) ONLY.
//   The raw INT32 accumulators are NOT stored here — they flow directly from
//   the MMU output buffer to the GCU without buffering, because the GCU
//   processes element-by-element and its output (INT8) is what needs buffering.
//
//   The GCU output feeds this buffer column-by-column.
//   Then FFN2 reads this buffer row-by-row as its input activation.
//
// ── Why this is the largest ILB buffer ────────────────────────────────────
//   49 patches × 384 features × 1 byte (INT8) = 18,816 bytes.
//   Compare with ilb_qkv_buf / ilb_proj_buf / ilb_context_buf at 4704 bytes.
//   This buffer is 4× larger because FFN expands the feature dimension 96→384.
//
// ── Write protocol ────────────────────────────────────────────────────────
//   GCU produces 7 INT8 values per cycle (one per patch row in the burst).
//   wr_patch_base [5:0] : first patch of burst (0, 7, ..., 42)
//   wr_col        [8:0] : feature column (0..383)
//   wr_en                : write strobe
//   wr_data  [6:0][7:0] : 7 INT8 bytes from GCU
//
// ── Read protocol ─────────────────────────────────────────────────────────
//   FFN2 reads the buffer row-by-row to load Z rows into unified_input_buf.
//   Each row is 384 bytes = 96 words of 4 bytes.
//   rd_patch [5:0], rd_col_word [7:0] → rd_data [31:0]  (4 bytes, 1-cycle lat.)
//   rd_col_word = 0..95 (word-aligned column, step 4).
//
// ── No double-banking ─────────────────────────────────────────────────────
//   GCU writes are complete before FFN2 reads begin (sequential phases in FSM).
//
// ── Buffer sizing ─────────────────────────────────────────────────────────
//   49 × 384 = 18,816 bytes.
// =============================================================================

module ilb_ffn1_buf #(
    parameter int N_PATCHES = 49,
    parameter int C_BYTES   = 384,  // expanded feature width (96 × 4)
    parameter int N_ROWS    = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // ═════════════════════════════════════════════════════════════════════
    // WRITE PORT — GCU output (GELU-activated, INT8)
    //
    // wr_patch_base : first patch of 7-row burst (0, 7, 14, ..., 42)
    // wr_col        : feature column (0..383)
    // wr_en         : write strobe
    // wr_data       : 7 INT8 bytes
    // ═════════════════════════════════════════════════════════════════════
    input  logic        wr_en,
    input  logic [5:0]  wr_patch_base,
    input  logic [8:0]  wr_col,               // 0..383  (9 bits)
    input  logic [7:0]  wr_data [0:N_ROWS-1],

    // ═════════════════════════════════════════════════════════════════════
    // READ PORT — FFN2 activation input
    //
    // Reads one 4-byte word per cycle.
    // rd_patch     : patch index (0..48)
    // rd_col_word  : word-aligned column (0..95, step=4 from caller's view,
    //                but here given as the word index 0..95 directly)
    // rd_en        : read enable
    // rd_data      : 4 packed INT8 bytes → fed to unified_input_buf loader
    // ═════════════════════════════════════════════════════════════════════
    input  logic        rd_en,
    input  logic [5:0]  rd_patch,
    input  logic [7:0]  rd_col_word,          // 0..95  (C_BYTES/4 = 96 words per row)
    output logic [31:0] rd_data
);

    // ── Storage: 18,816 bytes ─────────────────────────────────────────────
    localparam int BANK_BYTES = N_PATCHES * C_BYTES;  // 49 × 384 = 18816

    logic [7:0] mem [0:BANK_BYTES-1];

    initial begin
        for (int i = 0; i < BANK_BYTES; i++) mem[i] = '0;
    end

    // =========================================================================
    // Write — 7 bytes per cycle from GCU
    // Layout: mem[patch * C_BYTES + col]
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
    // Read — 4 bytes per cycle (registered, 1-cycle latency)
    // rd_col_word is the word index (= byte_col / 4).
    // byte base = patch * C_BYTES + rd_col_word * 4
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else if (rd_en) begin
            automatic int base = int'(rd_patch) * C_BYTES
                               + int'(rd_col_word) * 4;
            rd_data <= { mem[base + 3],
                         mem[base + 2],
                         mem[base + 1],
                         mem[base    ] };
        end
    end

endmodule
