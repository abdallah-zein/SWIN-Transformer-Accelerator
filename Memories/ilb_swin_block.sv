// =============================================================================
// ilb_swin_block.sv  (rev 2 — all 4 Swin stages supported)
//
// ── What changed from rev 1 ───────────────────────────────────────────────
//   Parameters updated:
//     C_MSA:  96 → 768    (Stage 4 max)
//     C_FFN:  384 → 3072  (Stage 4 max)
//     N_HEADS: 3 → 24     (Stage 4 max)
//     HEAD_DIM: 32 — UNCHANGED (constant across all stages)
//
//   New runtime configuration ports (wired to all sub-buffers):
//     c_bytes   [9:0]   — active stage MSA channel bytes (96/192/384/768)
//     ffn_c_bytes [11:0]— active stage FFN channel bytes (384/768/1536/3072)
//
//   Updated port widths:
//     qkv_wr_col, qkv_rd_col_a, qkv_rd_col_b: 7→10 bits
//     ctx_wr_head: 2→5 bits
//     ctx_rd_col: 7→10 bits
//     proj_wr_col, proj_rd_col: 7→10 bits
//     proj_rmw_addr: 11→14 bits
//     ffn1_wr_col: 9→12 bits
//     ffn1_rd_col_word: 8→10 bits
//
// ── Size summary (all stages) ─────────────────────────────────────────────
//   Buffer          Stage1   Stage2   Stage3   Stage4
//   ilb_qkv_buf      4704     9408    18816    37632 B (per bank × 2)
//   ilb_score_buf    9604     9604     9604     9604 B (per head, reused)
//   ilb_context_buf  4704     9408    18816    37632 B
//   ilb_proj_buf     4704     9408    18816    37632 B
//   ilb_ffn1_buf    18816    37632    75264   150528 B
//   ─────────────────────────────────────────────────────────────────────────
//   Total Stage4: 37632×2 + 9604 + 37632 + 37632 + 150528 ≈ 310 KB
// =============================================================================

module ilb_swin_block #(
    parameter int N_PATCHES = 49,
    parameter int C_MSA     = 768,    // Stage 4 max channel depth
    parameter int C_FFN     = 3072,   // Stage 4 max FFN depth
    parameter int N_HEADS   = 24,     // Stage 4 max
    parameter int HEAD_DIM  = 32,     // constant, d_head always 32
    parameter int N_ROWS    = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── Runtime stage configuration ───────────────────────────────────────
    // Set by controller at the start of each Swin Block round.
    input  logic [9:0]  c_bytes,         // MSA channel bytes: 96/192/384/768
    input  logic [11:0] ffn_c_bytes,     // FFN channel bytes: 384/768/1536/3072

    // =========================================================================
    // ilb_qkv_buf
    // =========================================================================
    input  logic        qkv_swap,
    input  logic        qkv_wr_en,
    input  logic [5:0]  qkv_wr_patch_base,
    input  logic [9:0]  qkv_wr_col,
    input  logic [7:0]  qkv_wr_data [0:N_ROWS-1],

    input  logic        qkv_rd_en_a,
    input  logic [5:0]  qkv_rd_patch_a,
    input  logic [9:0]  qkv_rd_col_a,
    output logic [31:0] qkv_rd_data_a,

    input  logic        qkv_rd_en_b,
    input  logic [5:0]  qkv_rd_row_b,
    input  logic [9:0]  qkv_rd_col_b,
    output logic [31:0] qkv_rd_data_b,

    // =========================================================================
    // ilb_score_buf
    // =========================================================================
    input  logic        score_commit,
    input  logic        score_clear,
    output logic        score_valid,

    input  logic        score_wr_en,
    input  logic [5:0]  score_wr_row_base,
    input  logic [5:0]  score_wr_col,
    input  logic [31:0] score_wr_data [0:N_ROWS-1],

    input  logic        score_rmw_rd_en,
    input  logic [11:0] score_rmw_addr,
    output logic [31:0] score_rmw_rd_data,
    input  logic        score_rmw_wr_en,
    input  logic [31:0] score_rmw_wr_data,

    input  logic        score_rd_en,
    input  logic [11:0] score_rd_addr,
    output logic [31:0] score_rd_data,

    // =========================================================================
    // ilb_context_buf
    // =========================================================================
    input  logic        ctx_wr_en,
    input  logic [5:0]  ctx_wr_patch_base,
    input  logic [4:0]  ctx_wr_head,       // 0..N_HEADS-1
    input  logic [4:0]  ctx_wr_head_col,   // 0..HEAD_DIM-1
    input  logic [7:0]  ctx_wr_data [0:N_ROWS-1],

    input  logic        ctx_rd_en,
    input  logic [5:0]  ctx_rd_patch,
    input  logic [9:0]  ctx_rd_col,
    output logic [31:0] ctx_rd_data,

    // =========================================================================
    // ilb_proj_buf
    // =========================================================================
    input  logic        proj_wr_en,
    input  logic [5:0]  proj_wr_patch_base,
    input  logic [9:0]  proj_wr_col,
    input  logic [7:0]  proj_wr_data [0:N_ROWS-1],

    input  logic        proj_rd_en,
    input  logic [5:0]  proj_rd_patch,
    input  logic [9:0]  proj_rd_col,
    output logic [31:0] proj_rd_data,

    input  logic        proj_rmw_rd_en,
    input  logic [13:0] proj_rmw_addr,
    output logic [31:0] proj_rmw_rd_data,
    input  logic        proj_rmw_wr_en,
    input  logic [31:0] proj_rmw_wr_data,

    // =========================================================================
    // ilb_ffn1_buf
    // =========================================================================
    input  logic        ffn1_wr_en,
    input  logic [5:0]  ffn1_wr_patch_base,
    input  logic [11:0] ffn1_wr_col,
    input  logic [7:0]  ffn1_wr_data [0:N_ROWS-1],

    input  logic        ffn1_rd_en,
    input  logic [5:0]  ffn1_rd_patch,
    input  logic [9:0]  ffn1_rd_col_word,
    output logic [31:0] ffn1_rd_data
);

    ilb_qkv_buf #(.N_PATCHES(N_PATCHES), .C_BYTES(C_MSA), .N_ROWS(N_ROWS)) u_qkv (
        .clk(clk), .rst_n(rst_n),
        .c_bytes(c_bytes),
        .swap(qkv_swap),
        .wr_en(qkv_wr_en), .wr_patch_base(qkv_wr_patch_base),
        .wr_col(qkv_wr_col), .wr_data(qkv_wr_data),
        .rd_en_a(qkv_rd_en_a), .rd_patch_a(qkv_rd_patch_a),
        .rd_col_a(qkv_rd_col_a), .rd_data_a(qkv_rd_data_a),
        .rd_en_b(qkv_rd_en_b), .rd_row_b(qkv_rd_row_b),
        .rd_col_b(qkv_rd_col_b), .rd_data_b(qkv_rd_data_b)
    );

    ilb_score_buf #(.N_PATCHES(N_PATCHES), .N_ROWS(N_ROWS)) u_score (
        .clk(clk), .rst_n(rst_n),
        .score_commit(score_commit), .score_clear(score_clear), .score_valid(score_valid),
        .wr_en(score_wr_en), .wr_row_base(score_wr_row_base),
        .wr_col(score_wr_col), .wr_data(score_wr_data),
        .rmw_rd_en(score_rmw_rd_en), .rmw_addr(score_rmw_addr),
        .rmw_rd_data(score_rmw_rd_data), .rmw_wr_en(score_rmw_wr_en),
        .rmw_wr_data(score_rmw_wr_data),
        .rd_en(score_rd_en), .rd_addr(score_rd_addr), .rd_data(score_rd_data)
    );

    ilb_context_buf #(.N_PATCHES(N_PATCHES), .C_BYTES(C_MSA),
                      .HEAD_DIM(HEAD_DIM), .N_HEADS(N_HEADS), .N_ROWS(N_ROWS)) u_ctx (
        .clk(clk), .rst_n(rst_n),
        .c_bytes(c_bytes),
        .wr_en(ctx_wr_en), .wr_patch_base(ctx_wr_patch_base),
        .wr_head(ctx_wr_head), .wr_head_col(ctx_wr_head_col), .wr_data(ctx_wr_data),
        .rd_en(ctx_rd_en), .rd_patch(ctx_rd_patch),
        .rd_col(ctx_rd_col), .rd_data(ctx_rd_data)
    );

    ilb_proj_buf #(.N_PATCHES(N_PATCHES), .C_BYTES(C_MSA), .N_ROWS(N_ROWS)) u_proj (
        .clk(clk), .rst_n(rst_n),
        .c_bytes(c_bytes),
        .wr_en(proj_wr_en), .wr_patch_base(proj_wr_patch_base),
        .wr_col(proj_wr_col), .wr_data(proj_wr_data),
        .rd_en(proj_rd_en), .rd_patch(proj_rd_patch),
        .rd_col(proj_rd_col), .rd_data(proj_rd_data),
        .rmw_rd_en(proj_rmw_rd_en), .rmw_addr(proj_rmw_addr),
        .rmw_rd_data(proj_rmw_rd_data), .rmw_wr_en(proj_rmw_wr_en),
        .rmw_wr_data(proj_rmw_wr_data)
    );

    ilb_ffn1_buf #(.N_PATCHES(N_PATCHES), .C_BYTES(C_FFN), .N_ROWS(N_ROWS)) u_ffn1 (
        .clk(clk), .rst_n(rst_n),
        .ffn_c_bytes(ffn_c_bytes),
        .wr_en(ffn1_wr_en), .wr_patch_base(ffn1_wr_patch_base),
        .wr_col(ffn1_wr_col), .wr_data(ffn1_wr_data),
        .rd_en(ffn1_rd_en), .rd_patch(ffn1_rd_patch),
        .rd_col_word(ffn1_rd_col_word), .rd_data(ffn1_rd_data)
    );

endmodule
