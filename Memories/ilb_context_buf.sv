// =============================================================================
// ilb_context_buf.sv
//
// Intermediate Layer Buffer — Attention Context store (A = S × V)
// Accumulates all 3 attention heads into a single 49×96 INT8 matrix
// ready for the linear projection step (W_Proj).
//
// ── Role in the Swin Block dataflow ──────────────────────────────────────
//
//   Step 5a:  ATTN = S × V
//
//   For each head h (0..2):
//     A_h = S_h × V_h   →  shape 49×32 INT8
//     Stored at columns [h*32 .. h*32+31] of this buffer's 49×96 layout.
//
//   After all 3 heads:
//     concat(A_0, A_1, A_2)  →  49×96 INT8  (already in this buffer, flat)
//   Ready to be read into unified_input_buf for the Proj step.
//
// ── Why a dedicated buffer (not reuse ilb_qkv_buf) ───────────────────────
//   ilb_qkv_buf is actively holding V (needed for S×V read) while A_h is
//   being written here.  They cannot share the same physical bank safely
//   because both are being accessed simultaneously.
//
// ── Write protocol ────────────────────────────────────────────────────────
//   The MMU produces 7 rows (patches) of one output column per compute burst.
//   wr_patch_base [5:0] : first patch of burst (0, 7, ..., 42)
//   wr_head       [1:0] : current head (0, 1, 2)
//   wr_head_col   [4:0] : column within the head (0..31)
//   wr_en                : write strobe
//   wr_data  [6:0][7:0] : 7 INT8 values from MMU (post-quantisation)
//
//   Physical column = wr_head * 32 + wr_head_col
//   Physical address = patch * 96 + physical_column
//
// ── Read protocol ─────────────────────────────────────────────────────────
//   After all 3 heads are written the buffer is read row-by-row to load the
//   concatenated 49×96 matrix into unified_input_buf for the Proj step.
//   Same as ilb_qkv_buf Port A: word-aligned 4-byte reads.
//   rd_patch [5:0], rd_col [6:0] → rd_data [31:0]  (1-cycle latency)
//
// ── No double-banking needed ──────────────────────────────────────────────
//   Write (A_h computation) and read (Proj input loading) are strictly
//   sequential in the controller FSM — all 3 heads are written THEN the
//   whole buffer is read.  A single bank suffices.
//
// ── Buffer sizing ─────────────────────────────────────────────────────────
//   49 patches × 96 bytes = 4704 bytes.
// =============================================================================

module ilb_context_buf #(
    parameter int N_PATCHES  = 49,
    parameter int C_BYTES    = 96,   // total feature bytes (3 heads × 32)
    parameter int HEAD_DIM   = 32,   // feature bytes per head
    parameter int N_HEADS    = 3,
    parameter int N_ROWS     = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── Bank swap is not needed; controller reads after all writes finish ──

    // ═════════════════════════════════════════════════════════════════════
    // WRITE PORT
    //
    // wr_head_col : column index within head h (0..31)
    // Physical column = wr_head * HEAD_DIM + wr_head_col
    // ═════════════════════════════════════════════════════════════════════
    input  logic        wr_en,
    input  logic [5:0]  wr_patch_base,  // 0, 7, 14, 21, 28, 35, 42
    input  logic [1:0]  wr_head,        // 0..2
    input  logic [4:0]  wr_head_col,    // 0..31
    input  logic [7:0]  wr_data [0:N_ROWS-1],  // 7 INT8 values

    // ═════════════════════════════════════════════════════════════════════
    // READ PORT — word-aligned, 4 bytes per cycle
    //
    // Reads 4 consecutive bytes at (rd_patch, rd_col).
    // rd_col must be word-aligned (0, 4, 8, ... 92).
    // ═════════════════════════════════════════════════════════════════════
    input  logic        rd_en,
    input  logic [5:0]  rd_patch,       // 0..48
    input  logic [6:0]  rd_col,         // 0..92 word-aligned
    output logic [31:0] rd_data         // 4 packed INT8 bytes
);

    localparam int BANK_BYTES = N_PATCHES * C_BYTES;  // 4704

    logic [7:0] mem [0:BANK_BYTES-1];

    initial begin
        for (int i = 0; i < BANK_BYTES; i++) mem[i] = '0;
    end

    // =========================================================================
    // Write — 7 bytes per cycle, one per patch row in the burst
    // Physical col = wr_head * HEAD_DIM + wr_head_col
    // =========================================================================
    always_ff @(posedge clk) begin
        if (wr_en) begin
            automatic int phys_col = int'(wr_head) * HEAD_DIM + int'(wr_head_col);
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int patch = int'(wr_patch_base) + r;
                if (patch < N_PATCHES) begin
                    automatic int addr = patch * C_BYTES + phys_col;
                    mem[addr] <= wr_data[r];
                end
            end
        end
    end

    // =========================================================================
    // Read — 4 bytes per cycle (registered, 1-cycle latency)
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

endmodule
