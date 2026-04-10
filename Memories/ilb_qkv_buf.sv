// =============================================================================
// ilb_qkv_buf.sv
//
// Intermediate Layer Buffer — QKV / Projection output store
// for one 7×7 window of the Swin Transformer Block.
//
// ── Role in the Swin Block dataflow ──────────────────────────────────────
//
//   This buffer is time-multiplexed across FOUR operations (one at a time):
//
//   Phase  Operation          Source           Shape     Usage after store
//   ─────  ─────────────────  ───────────────  ────────  ─────────────────
//   QKV_Q  X × W_Q → Q        MMU output       49 × 96   Read into ibuf for QK^T
//   QKV_K  X × W_K → K        MMU output       49 × 96   Read col-major for K^T
//   QKV_V  X × W_V → V        MMU output       49 × 96   Read col-major for S×V
//   PROJ   concat(A) × W_P    MMU output       49 × 96   + X residual → ilb_proj
//
//   All four phases produce INT8 results after upstream quantisation.
//   Each phase fills the SHADOW bank column-by-column (one 7-element burst
//   per output column per compute cycle).  The controller swaps once a full
//   49×96 matrix is written.
//
// ── Head splitting — zero-copy ────────────────────────────────────────────
//   The full 49×96 matrix is stored flat (row-major: row r, col c → word at
//   row r, byte c).  Each of the 3 heads occupies columns [h*32 .. h*32+31].
//   The controller generates strided addresses when loading head data back into
//   the unified_input_buf or unified_weight_buf — no physical rearrangement.
//
//   Head h, column c' (0..31):   byte address = patch * 96 + h*32 + c'
//
// ── K^T transpose — zero-copy ─────────────────────────────────────────────
//   To read K_h^T column j (j=0..48) into the weight buffer, the controller
//   reads bytes [j*96 + h*32 + 0 .. j*96 + h*32 + 31] — i.e., row j of
//   K_h.  This presents the jth row of K_h as the jth column of K_h^T.
//   The buffer simply responds to whatever read address it receives.
//
// ── Interface ─────────────────────────────────────────────────────────────
//   Write path:  column-by-column from the MMU output pipeline.
//     wr_patch_base [5:0]  : first patch of the 7-row burst (0,7,14,...,42)
//     wr_col        [6:0]  : output column being written (0..95)
//     wr_en                : write strobe
//     wr_data [6:0][7:0]   : 7 INT8 values from the MMU (one per patch row)
//
//   Read path A (patch-sequential, for loading Q_h / A heads into ibuf):
//     rd_patch   [5:0]     : patch index (0..48)
//     rd_col     [6:0]     : column (0..95)
//     rd_en_a              : read enable
//     rd_data_a  [31:0]    : 4-byte packed word (cols rd_col, rd_col+1, +2, +3)
//
//   Read path B (column-slice read, for loading K_h^T / V_h column into wbuf):
//     rd_col_b   [6:0]     : column (0..95)
//     rd_row_b   [5:0]     : patch (0..48) within that column
//     rd_en_b              : read enable
//     rd_data_b  [31:0]    : 4-byte word (rows rd_row_b .. rd_row_b+3)
//
//   Bank control:
//     swap                 : promotes shadow → active (one cycle pulse)
//
// ── Bank sizing ───────────────────────────────────────────────────────────
//   49 patches × 96 bytes = 4704 bytes per bank.
//   Two banks → 9408 bytes total.
//
// ── Read latency ─────────────────────────────────────────────────────────
//   Both read ports are registered: data valid 1 cycle after rd_en.
// =============================================================================

module ilb_qkv_buf #(
    parameter int N_PATCHES = 49,   // patches per 7×7 window
    parameter int C_BYTES   = 96,   // feature bytes per patch (96 channels INT8)
    parameter int N_ROWS    = 7,    // MMU output rows per burst
    parameter int N_COL_B_ROWS = 4  // bytes packed per read-B word
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── Bank swap ─────────────────────────────────────────────────────────
    input  logic        swap,

    // ═════════════════════════════════════════════════════════════════════
    // WRITE PORT — receives 7 INT8 values per cycle from MMU pipeline
    //
    //   One write per compute cycle:
    //     patches [wr_patch_base .. wr_patch_base+6]  get byte at column wr_col
    //   The controller loops wr_col over 0..95 and wr_patch_base over
    //   0,7,14,21,28,35,42 to fill all 49×96 bytes.
    // ═════════════════════════════════════════════════════════════════════
    input  logic        wr_en,
    input  logic [5:0]  wr_patch_base,  // 0, 7, 14, 21, 28, 35, 42
    input  logic [6:0]  wr_col,         // 0..95  (byte column in the 96-wide row)
    input  logic [7:0]  wr_data [0:N_ROWS-1],  // 7 INT8 bytes, one per patch row

    // ═════════════════════════════════════════════════════════════════════
    // READ PORT A — patch-sequential word read
    //
    // Reads a 4-byte packed word starting at byte (rd_patch, rd_col_a).
    // Used when loading Q_h rows or concatenated A rows into unified_input_buf.
    //   rd_col_a must be word-aligned (multiple of 4).
    //   Output packs bytes [rd_col_a, +1, +2, +3] of patch rd_patch.
    // ═════════════════════════════════════════════════════════════════════
    input  logic        rd_en_a,
    input  logic [5:0]  rd_patch_a,    // 0..48
    input  logic [6:0]  rd_col_a,      // 0..92 (word-aligned, step 4)
    output logic [31:0] rd_data_a,     // packed {byte+3, byte+2, byte+1, byte+0}

    // ═════════════════════════════════════════════════════════════════════
    // READ PORT B — column-slice word read
    //
    // Reads 4 consecutive patch rows of a single column.
    // Used to stream K_h^T columns (= K_h rows) and V_h columns into
    // unified_weight_buf.
    //   rd_row_b must be word-aligned (multiple of 4).
    //   Output packs bytes at (rd_row_b, rd_col_b), (+1, ...), (+2, ...), (+3, ...).
    // ═════════════════════════════════════════════════════════════════════
    input  logic        rd_en_b,
    input  logic [5:0]  rd_row_b,      // 0..48 (word-aligned, step 4)
    input  logic [6:0]  rd_col_b,      // 0..95
    output logic [31:0] rd_data_b      // packed {(row+3,col), (row+2,col), ...}
);

    // ── Bank storage: 2 × 4704 bytes ─────────────────────────────────────
    // Layout: bank[bank_sel][patch * C_BYTES + col_byte]
    localparam int BANK_BYTES = N_PATCHES * C_BYTES;  // 4704

    logic [7:0] bank [0:1][0:BANK_BYTES-1];
    logic       active;
    logic       shadow;
    assign shadow = ~active;

    // ── Bank swap ─────────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) active <= 1'b0;
        else if (swap) active <= shadow;

    // =========================================================================
    // Write — 7 bytes in parallel, one per patch row in the burst
    // =========================================================================
    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int patch = int'(wr_patch_base) + r;
                if (patch < N_PATCHES) begin
                    automatic int addr = patch * C_BYTES + int'(wr_col);
                    bank[shadow][addr] <= wr_data[r];
                end
            end
        end
    end

    // =========================================================================
    // Read Port A — patch-sequential (registered, 1-cycle latency)
    // Packs 4 consecutive bytes of one patch into a 32-bit word.
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data_a <= '0;
        end else if (rd_en_a) begin
            automatic int base = int'(rd_patch_a) * C_BYTES + int'(rd_col_a);
            rd_data_a <= { bank[active][base + 3],
                           bank[active][base + 2],
                           bank[active][base + 1],
                           bank[active][base    ] };
        end
    end

    // =========================================================================
    // Read Port B — column-slice (registered, 1-cycle latency)
    // Packs 4 consecutive patch rows of one column into a 32-bit word.
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data_b <= '0;
        end else if (rd_en_b) begin
            automatic int b0 = (int'(rd_row_b)    ) * C_BYTES + int'(rd_col_b);
            automatic int b1 = (int'(rd_row_b) + 1) * C_BYTES + int'(rd_col_b);
            automatic int b2 = (int'(rd_row_b) + 2) * C_BYTES + int'(rd_col_b);
            automatic int b3 = (int'(rd_row_b) + 3) * C_BYTES + int'(rd_col_b);
            rd_data_b <= { (int'(rd_row_b)+3 < N_PATCHES) ? bank[active][b3] : 8'h00,
                           (int'(rd_row_b)+2 < N_PATCHES) ? bank[active][b2] : 8'h00,
                           (int'(rd_row_b)+1 < N_PATCHES) ? bank[active][b1] : 8'h00,
                           (int'(rd_row_b)   < N_PATCHES) ? bank[active][b0] : 8'h00 };
        end
    end

endmodule
