// =============================================================================
// ilb_context_buf.sv  (rev 2 — all 4 Swin stages supported)
//
// ── What changed from rev 1 ───────────────────────────────────────────────
//   C_BYTES:  96 → 768   (Stage 4)
//   HEAD_DIM: 32 → 32    (UNCHANGED — d_head=32 constant all stages)
//   N_HEADS:   3 → 24    (Stage 4 max)
//   BANK_BYTES: 4704 → 37632   (49 × 768)
//   wr_head:  2 bits → 5 bits   (covers 0..23 for Stage4)
//   wr_head_col: 5 bits → 5 bits (unchanged: 0..31)
//   rd_col: 7 bits → 10 bits
//   Added runtime c_bytes [9:0] port (same pattern as ilb_qkv_buf).
//
// ── Head layout ───────────────────────────────────────────────────────────
//   Physical column = wr_head * HEAD_DIM + wr_head_col
//   HEAD_DIM=32 is constant; N_HEADS varies (3/6/12/24).
//   For any stage: C_bytes = N_HEADS * HEAD_DIM = N_HEADS * 32.
//   The controller sets c_bytes = N_HEADS * 32 for the active stage.
// =============================================================================

module ilb_context_buf #(
    parameter int N_PATCHES = 49,
    parameter int C_BYTES   = 768,   // Stage 4: 24 heads × 32 = 768
    parameter int HEAD_DIM  = 32,    // constant, d_head always 32
    parameter int N_HEADS   = 24,    // Stage 4 max (runtime: wr_head 0..n_heads-1)
    parameter int N_ROWS    = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // Runtime: actual number of channel bytes for current stage
    input  logic [9:0]  c_bytes,   // 96/192/384/768

    // ── Write port — S×V output per head ─────────────────────────────────
    input  logic        wr_en,
    input  logic [5:0]  wr_patch_base,
    input  logic [4:0]  wr_head,        // 0..N_HEADS-1 (up to 23)
    input  logic [4:0]  wr_head_col,    // 0..HEAD_DIM-1 (0..31)
    input  logic [7:0]  wr_data [0:N_ROWS-1],

    // ── Read port — load concat(A) into ibuf for Proj ─────────────────────
    input  logic        rd_en,
    input  logic [5:0]  rd_patch,
    input  logic [9:0]  rd_col,         // word-aligned, 0..C_BYTES-4
    output logic [31:0] rd_data
);

    localparam int BANK_BYTES = N_PATCHES * C_BYTES;  // 49 × 768 = 37,632

    logic [7:0] mem [0:BANK_BYTES-1];

    initial begin
        for (int i = 0; i < BANK_BYTES; i++) mem[i] = '0;
    end

    // Write: physical col = wr_head * HEAD_DIM + wr_head_col
    always_ff @(posedge clk) begin
        if (wr_en) begin
            automatic int phys_col = int'(wr_head) * HEAD_DIM + int'(wr_head_col);
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int patch = int'(wr_patch_base) + r;
                if (patch < N_PATCHES) begin
                    automatic int addr = patch * int'(c_bytes) + phys_col;
                    mem[addr] <= wr_data[r];
                end
            end
        end
    end

    // Read — 4 bytes packed, 1-cycle latency
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_data <= '0;
        else if (rd_en) begin
            automatic int base = int'(rd_patch) * int'(c_bytes) + int'(rd_col);
            rd_data <= { mem[base+3], mem[base+2], mem[base+1], mem[base] };
        end
    end

endmodule
