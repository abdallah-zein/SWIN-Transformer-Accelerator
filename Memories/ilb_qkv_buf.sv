// =============================================================================
// ilb_qkv_buf.sv  (rev 2 — all 4 Swin stages supported)
//
// ── What changed from rev 1 ───────────────────────────────────────────────
//   C_BYTES: 96 → 768   (Stage 4 C=768; bank = 49×768 = 37,632 bytes/bank)
//   wr_col: 7 bits → 10 bits   (covers 0..767)
//   rd_col_a: 7 bits → 10 bits
//   rd_col_b: 7 bits → 10 bits
//   rd_row_b: 6 bits → unchanged (0..48 max, 49 patches always)
//   Added runtime parameter c_bytes [9:0] so the controller can tell the
//   buffer the active stage's C without recompilation.
//
// ── Sizing per stage ──────────────────────────────────────────────────────
//   Stage 1 (C=96):  49× 96=  4,704 bytes/bank
//   Stage 2 (C=192): 49×192=  9,408 bytes/bank
//   Stage 3 (C=384): 49×384= 18,816 bytes/bank
//   Stage 4 (C=768): 49×768= 37,632 bytes/bank  ← BANK_BYTES ceiling
//
// ── All other behaviour unchanged ────────────────────────────────────────
//   Double-banked, 1-cycle read latency, Port A (row-sequential 4-byte
//   reads), Port B (column-slice 4-byte reads for K^T / V).
// =============================================================================

module ilb_qkv_buf #(
    parameter int N_PATCHES = 49,
    parameter int C_BYTES   = 768,   // Stage 4 max (dynamic range via c_bytes port)
    parameter int N_ROWS    = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // Runtime channel width — set by controller before each round:
    //   Stage1=96, Stage2=192, Stage3=384, Stage4=768
    input  logic [9:0]  c_bytes,

    // ── Bank swap ─────────────────────────────────────────────────────────
    input  logic        swap,

    // ── Write port — 7 INT8 bytes per cycle ──────────────────────────────
    input  logic        wr_en,
    input  logic [5:0]  wr_patch_base,
    input  logic [9:0]  wr_col,          // 0..C_BYTES-1
    input  logic [7:0]  wr_data [0:N_ROWS-1],

    // ── Read Port A — patch-sequential 4-byte word ────────────────────────
    input  logic        rd_en_a,
    input  logic [5:0]  rd_patch_a,
    input  logic [9:0]  rd_col_a,        // word-aligned, 0..C_BYTES-4
    output logic [31:0] rd_data_a,

    // ── Read Port B — column-slice 4-byte word ────────────────────────────
    input  logic        rd_en_b,
    input  logic [5:0]  rd_row_b,        // word-aligned rows, 0..48
    input  logic [9:0]  rd_col_b,        // 0..C_BYTES-1
    output logic [31:0] rd_data_b
);

    localparam int BANK_BYTES = N_PATCHES * C_BYTES;  // 49 × 768 = 37,632

    logic [7:0] bank [0:1][0:BANK_BYTES-1];
    logic       active;
    logic       shadow;
    assign shadow = ~active;

    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) active <= 1'b0;
        else if (swap) active <= shadow;

    // ── Write ─────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int patch = int'(wr_patch_base) + r;
                if (patch < N_PATCHES) begin
                    automatic int addr = patch * int'(c_bytes) + int'(wr_col);
                    bank[shadow][addr] <= wr_data[r];
                end
            end
        end
    end

    // ── Read Port A ───────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_data_a <= '0;
        else if (rd_en_a) begin
            automatic int base = int'(rd_patch_a) * int'(c_bytes) + int'(rd_col_a);
            rd_data_a <= { bank[active][base+3], bank[active][base+2],
                           bank[active][base+1], bank[active][base  ] };
        end
    end

    // ── Read Port B ───────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_data_b <= '0;
        else if (rd_en_b) begin
            automatic int b0 = (int'(rd_row_b)  ) * int'(c_bytes) + int'(rd_col_b);
            automatic int b1 = (int'(rd_row_b)+1) * int'(c_bytes) + int'(rd_col_b);
            automatic int b2 = (int'(rd_row_b)+2) * int'(c_bytes) + int'(rd_col_b);
            automatic int b3 = (int'(rd_row_b)+3) * int'(c_bytes) + int'(rd_col_b);
            rd_data_b <= {
                (int'(rd_row_b)+3 < N_PATCHES) ? bank[active][b3] : 8'h00,
                (int'(rd_row_b)+2 < N_PATCHES) ? bank[active][b2] : 8'h00,
                (int'(rd_row_b)+1 < N_PATCHES) ? bank[active][b1] : 8'h00,
                (int'(rd_row_b)   < N_PATCHES) ? bank[active][b0] : 8'h00 };
        end
    end

endmodule
