// =============================================================================
// ilb_swin_block.sv
//
// Intermediate Layer Buffer — Swin Transformer Block
//
// This module is the ILB assigned to the Swin Block round.  It collects all
// five sub-buffers required to store intermediate matrices between the MSA
// and FFN operations within ONE complete Swin Block round (W-MSA or SW-MSA
// + FFN), without any off-chip traffic between these steps.
//
// ── Sub-buffer inventory ─────────────────────────────────────────────────
//
//   Buffer             Module           Shape       Bytes    Reuse
//   ─────────────────  ───────────────  ──────────  ───────  ──────────────
//   QKV / Proj store   ilb_qkv_buf      49 × 96     4704 B   Q, K, V, Proj out
//   Attention scores   ilb_score_buf    49 × 49×32  9604 B   per head, reused ×3
//   Context vectors    ilb_context_buf  49 × 96     4704 B   concat A_0,A_1,A_2
//   Proj + residual    ilb_proj_buf     49 × 96     4704 B   Out_MSA, FFN input
//   FFN1 intermediate  ilb_ffn1_buf     49 × 384   18816 B   Z = GELU(X×W_FFN1)
//   ─────────────────  ───────────────  ──────────  ───────
//   Total                                          42532 B  (≈ 41.5 KB)
//
// ── Swin Block dataflow and buffer usage ─────────────────────────────────
//
//   Step  Operation           Read from          Write to
//   ────  ──────────────────  ─────────────────  ──────────────────
//   1     X × W_Q → Q         FIB (X)            ilb_qkv_buf
//   2     X × W_K → K         FIB (X)            ilb_qkv_buf (swap)
//   3     X × W_V → V         FIB (X)            ilb_qkv_buf (swap)
//   4a    Q_h × K_h^T → S_h   ilb_qkv_buf (Q,K)  ilb_score_buf
//   4b    mask(S_h) [SW only]  ilb_score_buf RMW  ilb_score_buf (in-place)
//   4c    Softmax(S_h)         ilb_score_buf      ilb_score_buf (in-place)
//   5     S_h × V_h → A_h     ilb_score_buf,     ilb_context_buf
//                              ilb_qkv_buf (V)
//   6     concat(A)×W_P→M_out ilb_context_buf    ilb_qkv_buf (Proj)
//   7     M_out + X → Out_MSA ilb_qkv_buf + FIB  ilb_proj_buf  (RMW)
//   8     Out_MSA × W_FFN1    ilb_proj_buf        → GCU directly
//   9     GELU(step8)         GCU output          ilb_ffn1_buf
//   10    Z × W_FFN2 → F_out  ilb_ffn1_buf        → ilb_proj_buf (RMW)
//   11    F_out + Out_MSA     ilb_proj_buf RMW    ilb_proj_buf (in-place)
//   12    flush to off-chip   ilb_proj_buf        MWU → off-chip
//
// ── Reuse note ────────────────────────────────────────────────────────────
//   ilb_qkv_buf is reused for Q (step 1), then K (step 2), then V (step 3),
//   then Proj output (step 6).  The swap mechanism ensures prior content is
//   consumed before the bank is overwritten.
//
//   ilb_score_buf is reused once per head (head 0, 1, 2) — 3 reuses per window.
//   score_commit / score_clear control the valid flag between reuses.
//
// ── Connections to the wider system ──────────────────────────────────────
//   This module exposes all sub-buffer ports directly at the top level.
//   The controller (unified_controller) drives all addresses and enables.
//   The adder for residual / mask operations lives in full_system_top and
//   is fed by rmw_rd_data outputs and FIB / mask_buffer data externally.
//
// ── What this module does NOT contain ────────────────────────────────────
//   • Softmax (SCU) — separate module
//   • GELU (GCU) — separate module
//   • Mask buffer — mask_buffer.sv (separate, already designed)
//   • Residual adder — combinatorial, in full_system_top
//   • LayerNorm — separate (if implemented; Swin often uses post-norm)
//
// =============================================================================

module ilb_swin_block #(
    parameter int N_PATCHES  = 49,
    parameter int C_MSA      = 96,    // MSA feature depth
    parameter int C_FFN      = 384,   // FFN expanded depth
    parameter int N_HEADS    = 3,
    parameter int HEAD_DIM   = 32,    // C_MSA / N_HEADS
    parameter int N_ROWS     = 7      // MMU burst height
)(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // ilb_qkv_buf — Q / K / V / Proj output  (49 × 96 INT8)
    // =========================================================================
    // Bank swap
    input  logic        qkv_swap,

    // Write port
    input  logic        qkv_wr_en,
    input  logic [5:0]  qkv_wr_patch_base,
    input  logic [6:0]  qkv_wr_col,
    input  logic [7:0]  qkv_wr_data [0:N_ROWS-1],

    // Read Port A — patch-sequential (Q rows, Proj rows)
    input  logic        qkv_rd_en_a,
    input  logic [5:0]  qkv_rd_patch_a,
    input  logic [6:0]  qkv_rd_col_a,
    output logic [31:0] qkv_rd_data_a,

    // Read Port B — column-slice (K^T columns, V columns)
    input  logic        qkv_rd_en_b,
    input  logic [5:0]  qkv_rd_row_b,
    input  logic [6:0]  qkv_rd_col_b,
    output logic [31:0] qkv_rd_data_b,

    // =========================================================================
    // ilb_score_buf — Attention scores  (49 × 49 INT32)
    // =========================================================================
    // Valid lifecycle
    input  logic        score_commit,
    input  logic        score_clear,
    output logic        score_valid,

    // Write port — QK^T accumulation
    input  logic        score_wr_en,
    input  logic [5:0]  score_wr_row_base,
    input  logic [5:0]  score_wr_col,
    input  logic [31:0] score_wr_data [0:N_ROWS-1],

    // RMW port — mask application (SW-MSA) and Softmax write-back
    input  logic        score_rmw_rd_en,
    input  logic [11:0] score_rmw_addr,
    output logic [31:0] score_rmw_rd_data,
    input  logic        score_rmw_wr_en,
    input  logic [31:0] score_rmw_wr_data,

    // Sequential read — Softmax input / S×V operand
    input  logic        score_rd_en,
    input  logic [11:0] score_rd_addr,
    output logic [31:0] score_rd_data,

    // =========================================================================
    // ilb_context_buf — Context vectors  (49 × 96 INT8, 3 heads concat)
    // =========================================================================
    // Write port — S×V output per head
    input  logic        ctx_wr_en,
    input  logic [5:0]  ctx_wr_patch_base,
    input  logic [1:0]  ctx_wr_head,
    input  logic [4:0]  ctx_wr_head_col,
    input  logic [7:0]  ctx_wr_data [0:N_ROWS-1],

    // Read port — load concat(A) into ibuf for Proj
    input  logic        ctx_rd_en,
    input  logic [5:0]  ctx_rd_patch,
    input  logic [6:0]  ctx_rd_col,
    output logic [31:0] ctx_rd_data,

    // =========================================================================
    // ilb_proj_buf — Proj output + MSA residual + FFN residual  (49 × 96 INT8)
    // =========================================================================
    // Write port — Proj MMU output, or FFN2+residual final result
    input  logic        proj_wr_en,
    input  logic [5:0]  proj_wr_patch_base,
    input  logic [6:0]  proj_wr_col,
    input  logic [7:0]  proj_wr_data [0:N_ROWS-1],

    // Read port — FFN input loading, or MWU flush
    input  logic        proj_rd_en,
    input  logic [5:0]  proj_rd_patch,
    input  logic [6:0]  proj_rd_col,
    output logic [31:0] proj_rd_data,

    // RMW port — residual additions (MSA: step 7; FFN: step 11)
    input  logic        proj_rmw_rd_en,
    input  logic [10:0] proj_rmw_addr,
    output logic [31:0] proj_rmw_rd_data,
    input  logic        proj_rmw_wr_en,
    input  logic [31:0] proj_rmw_wr_data,

    // =========================================================================
    // ilb_ffn1_buf — FFN1 GELU output  (49 × 384 INT8)
    // =========================================================================
    // Write port — GCU (GELU) output
    input  logic        ffn1_wr_en,
    input  logic [5:0]  ffn1_wr_patch_base,
    input  logic [8:0]  ffn1_wr_col,
    input  logic [7:0]  ffn1_wr_data [0:N_ROWS-1],

    // Read port — FFN2 activation input
    input  logic        ffn1_rd_en,
    input  logic [5:0]  ffn1_rd_patch,
    input  logic [7:0]  ffn1_rd_col_word,
    output logic [31:0] ffn1_rd_data
);

    // =========================================================================
    // Sub-buffer instantiations
    // =========================================================================

    // ── QKV / Proj buffer ─────────────────────────────────────────────────
    ilb_qkv_buf #(
        .N_PATCHES (N_PATCHES),
        .C_BYTES   (C_MSA),
        .N_ROWS    (N_ROWS)
    ) u_qkv (
        .clk             (clk),
        .rst_n           (rst_n),
        .swap            (qkv_swap),
        .wr_en           (qkv_wr_en),
        .wr_patch_base   (qkv_wr_patch_base),
        .wr_col          (qkv_wr_col),
        .wr_data         (qkv_wr_data),
        .rd_en_a         (qkv_rd_en_a),
        .rd_patch_a      (qkv_rd_patch_a),
        .rd_col_a        (qkv_rd_col_a),
        .rd_data_a       (qkv_rd_data_a),
        .rd_en_b         (qkv_rd_en_b),
        .rd_row_b        (qkv_rd_row_b),
        .rd_col_b        (qkv_rd_col_b),
        .rd_data_b       (qkv_rd_data_b)
    );

    // ── Attention score buffer ─────────────────────────────────────────────
    ilb_score_buf #(
        .N_PATCHES (N_PATCHES),
        .N_ROWS    (N_ROWS)
    ) u_score (
        .clk             (clk),
        .rst_n           (rst_n),
        .score_commit    (score_commit),
        .score_clear     (score_clear),
        .score_valid     (score_valid),
        .wr_en           (score_wr_en),
        .wr_row_base     (score_wr_row_base),
        .wr_col          (score_wr_col),
        .wr_data         (score_wr_data),
        .rmw_rd_en       (score_rmw_rd_en),
        .rmw_addr        (score_rmw_addr),
        .rmw_rd_data     (score_rmw_rd_data),
        .rmw_wr_en       (score_rmw_wr_en),
        .rmw_wr_data     (score_rmw_wr_data),
        .rd_en           (score_rd_en),
        .rd_addr         (score_rd_addr),
        .rd_data         (score_rd_data)
    );

    // ── Context vector buffer ─────────────────────────────────────────────
    ilb_context_buf #(
        .N_PATCHES (N_PATCHES),
        .C_BYTES   (C_MSA),
        .HEAD_DIM  (HEAD_DIM),
        .N_HEADS   (N_HEADS),
        .N_ROWS    (N_ROWS)
    ) u_ctx (
        .clk             (clk),
        .rst_n           (rst_n),
        .wr_en           (ctx_wr_en),
        .wr_patch_base   (ctx_wr_patch_base),
        .wr_head         (ctx_wr_head),
        .wr_head_col     (ctx_wr_head_col),
        .wr_data         (ctx_wr_data),
        .rd_en           (ctx_rd_en),
        .rd_patch        (ctx_rd_patch),
        .rd_col          (ctx_rd_col),
        .rd_data         (ctx_rd_data)
    );

    // ── Proj + residual buffer ─────────────────────────────────────────────
    ilb_proj_buf #(
        .N_PATCHES (N_PATCHES),
        .C_BYTES   (C_MSA),
        .N_ROWS    (N_ROWS)
    ) u_proj (
        .clk             (clk),
        .rst_n           (rst_n),
        .wr_en           (proj_wr_en),
        .wr_patch_base   (proj_wr_patch_base),
        .wr_col          (proj_wr_col),
        .wr_data         (proj_wr_data),
        .rd_en           (proj_rd_en),
        .rd_patch        (proj_rd_patch),
        .rd_col          (proj_rd_col),
        .rd_data         (proj_rd_data),
        .rmw_rd_en       (proj_rmw_rd_en),
        .rmw_addr        (proj_rmw_addr),
        .rmw_rd_data     (proj_rmw_rd_data),
        .rmw_wr_en       (proj_rmw_wr_en),
        .rmw_wr_data     (proj_rmw_wr_data)
    );

    // ── FFN1 GELU intermediate buffer ─────────────────────────────────────
    ilb_ffn1_buf #(
        .N_PATCHES (N_PATCHES),
        .C_BYTES   (C_FFN),
        .N_ROWS    (N_ROWS)
    ) u_ffn1 (
        .clk             (clk),
        .rst_n           (rst_n),
        .wr_en           (ffn1_wr_en),
        .wr_patch_base   (ffn1_wr_patch_base),
        .wr_col          (ffn1_wr_col),
        .wr_data         (ffn1_wr_data),
        .rd_en           (ffn1_rd_en),
        .rd_patch        (ffn1_rd_patch),
        .rd_col_word     (ffn1_rd_col_word),
        .rd_data         (ffn1_rd_data)
    );

endmodule
