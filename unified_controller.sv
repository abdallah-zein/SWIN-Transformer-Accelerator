// =============================================================================
// unified_controller.sv  (rev 4 — Mask Buffer integration)
// Includes: Bias Buffer (BB) additions
//
// ── What changed from rev 3 ───────────────────────────────────────────────
//   Added mask-apply sub-FSM that runs per attention head, immediately after
//   the head's full 49×49 QK^T score matrix is stored in the ILB and before
//   the S×V step begins.
//
//   The mask_buffer module (external) holds 49 × 32-bit attention-bias values.
//   Each bias is broadcast-added to every element of the corresponding 7×7
//   sub-block of the 49×49 score matrix (49 sub-blocks × 49 elements each).
//   The addition itself is performed by an adder in full_system_top:
//       assign masked_s_word = omem_fb_rd_data + mask_data_out;
//   This controller only orchestrates the read-modify-write loop and drives
//   the two handshake signals into mask_buffer.
//
// ── New ports ─────────────────────────────────────────────────────────────
//   output qkt_store_done    : 1-cycle pulse when a head's 49×49 QK^T is done
//   output mask_next_window  : 1-cycle pulse after each 49-element window R-M-W
//   input  mask_valid        : from mask_buffer (mask pass active)
//   input  mask_window_idx   : current sub-block index from mask_buffer
//   input  mask_all_done     : from mask_buffer, last window finished
//
// ── New states (3) ────────────────────────────────────────────────────────
//   S_H_MASK_RD       = 7'd83   issue ILB fb_rd_en for one score element
//   S_H_MASK_WB       = 7'd84   fb_rd_data valid → write masked value back
//   S_H_MASK_NEXT_WIN = 7'd85   pulse mask_next_window; loop or exit
//
// ── State transition change ───────────────────────────────────────────────
//   S_H_NEXT_ATTN_ROWGRP: h_last_attn_rowgrp now → S_H_MASK_RD (was S_H_NEXT_ATTN_HD)
//   S_H_MASK_NEXT_WIN:    mask_all_done           → S_H_NEXT_ATTN_HD
//
// ── New counter ───────────────────────────────────────────────────────────
//   h_mask_elem_cnt [5:0] : element index within the current 7×7 sub-block (0..48)
//
// ── Mask pass cost per window ─────────────────────────────────────────────
//   49 sub-blocks × 49 elements × 2 cycles (RD+WB) = 4 802 cycles/head
//   × 3 heads = 14 406 cycles per 7×7 window (added after QK^T, before S×V)
// =============================================================================

module unified_controller #(
    parameter int WAW      = 16,
    parameter int FAW      = 17,
    parameter int OAW      = 19,
    parameter int W2_BASE  = 9216,

    // MHA weight offsets
    parameter int WQ_BASE    = 10240,
    parameter int WK_BASE    = 19456,
    parameter int WV_BASE    = 28672,
    parameter int WPROJ_BASE = 37888,
    parameter int WFFN1_BASE = 47104,
    parameter int WFFN2_BASE = 56320,

    // MHA ILB base addresses
    parameter int ILB_Q_BASE    = 0,
    parameter int ILB_K_BASE    = 3072,
    parameter int ILB_V_BASE    = 6144,
    parameter int ILB_S_BASE    = 9216,
    parameter int ILB_A_BASE    = 16468,
    parameter int ILB_PROJ_BASE = 19540,
    parameter int ILB_FFN1_BASE = 20588,

    // Shift buffer parameters
    parameter int SB_AW        = 14,
    parameter int SB_CONV_BASE = 0,
    parameter int SB_MLP_BASE  = 5376,
    parameter int SB_MHA_BASE  = 5824,

    // Bias buffer parameters
    parameter int BB_AW           = 12,
    parameter int BB_CONV_BASE    = 0,
    parameter int BB_MLP_L1_BASE  = 96,
    parameter int BB_MLP_L2_BASE  = 480,
    parameter int BB_MHA_QKT_BASE = 576
)(
    input  logic clk,
    input  logic rst_n,

    // ── Mode and start/done ───────────────────────────────────────────────
    input  logic [1:0] mode,
    input  logic start,
    output logic done,

    // ── Weight memory — active bank read ─────────────────────────────────
    output logic [WAW-1:0] wmem_rd_addr,
    output logic           wmem_rd_en,
    input  logic [31:0]    wmem_rd_data,

    // ── Weight memory — shadow bank write ────────────────────────────────
    output logic [WAW-1:0] wmem_shadow_wr_addr,
    output logic           wmem_shadow_wr_en,

    // ── Weight memory — bank swap ─────────────────────────────────────────
    output logic           wmem_swap,

    // ── External big memory ───────────────────────────────────────────────
    output logic [WAW-1:0] ext_weight_rd_addr,
    output logic           ext_weight_rd_en,

    // ── Input / feature memory ────────────────────────────────────────────
    output logic [OAW-1:0] imem_rd_addr,
    output logic           imem_rd_en,
    input  logic [31:0]    imem_rd_data,

    // ── Output memory write ───────────────────────────────────────────────
    output logic [OAW-1:0] omem_wr_addr,
    output logic           omem_wr_en,

    // ── Weight buffer control ─────────────────────────────────────────────
    output logic           wbuf_load_en,
    output logic [3:0]     wbuf_load_pe_idx,
    output logic [6:0]     wbuf_load_k_word,
    output logic [31:0]    wbuf_load_data,
    output logic           wbuf_bias_load_en,
    output logic [31:0]    wbuf_bias_load_data,
    output logic           wbuf_swap,

    // ── Input buffer control ──────────────────────────────────────────────
    output logic           ibuf_load_en,
    output logic [3:0]     ibuf_load_pe_idx,
    output logic [2:0]     ibuf_load_win_idx,
    output logic [2:0]     ibuf_load_row,
    output logic [6:0]     ibuf_load_k_word,
    output logic [31:0]    ibuf_load_data,
    output logic           ibuf_swap,
    output logic           ibuf_l1_capture_en,
    output logic [8:0]     ibuf_l1_col_wr,

    // ── MHA-specific input buffer control ─────────────────────────────────
    output logic           ibuf_mha_load_en,
    output logic [5:0]     ibuf_mha_load_patch,
    output logic [4:0]     ibuf_mha_load_k_word,
    output logic [31:0]    ibuf_mha_load_data,
    output logic [5:0]     ibuf_mha_capture_row,

    // ── MMU control ───────────────────────────────────────────────────────
    output logic           mmu_valid_in,
    output logic [2:0]     mmu_op_code,
    output logic [1:0]     mmu_stage,
    output logic [2:0]     mmu_sub_cycle,

    // ── Output buffer control ─────────────────────────────────────────────
    output logic           obuf_capture_en,
    output logic [2:0]     obuf_rd_idx,

    // ── Feedback path select ──────────────────────────────────────────────
    output logic           omem_fb_en_ctrl,

    // ── Shift buffer control ──────────────────────────────────────────────
    output logic               sb_op_start,
    output logic [SB_AW-1:0]   sb_op_base_addr,
    output logic               sb_advance,

    // ── Bias buffer control ───────────────────────────────────────────────
    output logic               bb_op_start,
    output logic [BB_AW-1:0]   bb_op_base_addr,
    output logic               bb_advance,

    // ── Mask buffer control (NEW) ─────────────────────────────────────────
    // qkt_store_done:   1-cycle pulse when a head's full 49×49 QK^T score
    //                   matrix has just been written to the ILB. Resets the
    //                   mask_buffer read pointer and starts the mask pass.
    // mask_next_window: 1-cycle pulse after all 49 elements of the current
    //                   7×7 sub-block have been read-modify-written.
    //                   Advances mask_buffer.rd_ptr by one.
    output logic           qkt_store_done,
    output logic           mask_next_window,

    // mask_valid:       Driven by mask_buffer; 1 during an active mask pass.
    //                   The controller may use this as a sanity check.
    // mask_window_idx:  Current sub-block index (0..48) from mask_buffer.
    //                   Used to compute ILB addresses during the mask states.
    // mask_all_done:    1-cycle pulse from mask_buffer on the last
    //                   mask_next_window (window 48). Used in
    //                   S_H_MASK_NEXT_WIN to decide whether to loop or exit.
    input  logic           mask_valid,
    input  logic [5:0]     mask_window_idx,
    input  logic           mask_all_done
);

// =============================================================================
// Parameters — Conv / MLP (unchanged)
// =============================================================================
localparam int C_N_KERNELS     = 96;
localparam int C_N_ROW_GROUPS  = 56;
localparam int C_N_CHUNKS      = 8;
localparam int C_WLOAD_WORDS   = 13;
localparam int C_ILOAD_WORDS   = 84;
localparam int C_OUT_WORDS     = 7;
localparam int C_WLOAD_CYCS    = C_WLOAD_WORDS * 2;
localparam int C_ILOAD_CYCS    = C_ILOAD_WORDS * 2;
localparam int C_IMG_W_WORDS   = 56;
localparam int C_IMG_CH_WORDS  = 56 * 224;
localparam int C_OUT_ROW_WORDS = 56;
localparam int C_OUT_K_WORDS   = 56 * 56;

localparam int M_N_ROW_GRPS = 448;
localparam int M_N_L1_COLS  = 384;
localparam int M_N_L2_COLS  = 96;
localparam int M_N_ACC_L1   = 2;
localparam int M_N_ACC_L2   = 8;
localparam int M_W1_WORDS   = 24;
localparam int M_W2_WORDS   = 96;
localparam int M_X_WORDS    = 168;
localparam int M_W1_CYCS    = M_W1_WORDS * 2;
localparam int M_W2_CYCS    = M_W2_WORDS * 2;
localparam int M_X_CYCS     = M_X_WORDS  * 2;
localparam int M_OUT_WORDS  = 7;

// =============================================================================
// Parameters — MHA (unchanged except two new mask localparams)
// =============================================================================
localparam int H_N_WINDOWS  = 64;
localparam int H_N_PATCHES  = 49;
localparam int H_N_HEADS    = 3;
localparam int H_HEAD_DIM   = 32;
localparam int H_C_IN       = 96;
localparam int H_C_FFN      = 384;

localparam int H_W_QKV_WORDS  = 24;
localparam int H_W_QKV_CYCS   = H_W_QKV_WORDS * 2;
localparam int H_WKT_WORDS    = 13;
localparam int H_WKT_CYCS     = H_WKT_WORDS * 2;
localparam int H_I_PATCH7_WORDS = H_N_PATCHES * H_W_QKV_WORDS;
localparam int H_I_LOAD7_WORDS  = 7 * H_W_QKV_WORDS;
localparam int H_I_LOAD7_CYCS   = H_I_LOAD7_WORDS * 2;
localparam int H_QH_LOAD_WORDS  = 7 * 8;
localparam int H_QH_LOAD_CYCS   = H_QH_LOAD_WORDS * 2;
localparam int H_N_ACC_QKV  = 2;
localparam int H_N_ACC_ATTN = 1;
localparam int H_N_ACC_SXV  = 2;
localparam int H_N_ACC_PROJ = 2;
localparam int H_N_ACC_FFN1 = 2;
localparam int H_N_ACC_FFN2 = 8;
localparam int H_OUT_WORDS  = 7;

// ── NEW: mask-related strides ─────────────────────────────────────────────
// H_S_ELEMS       : total elements in one head's 49×49 score matrix
// H_MASK_WIN_ELEMS: elements per 7×7 sub-block (= 49)
localparam int H_S_ELEMS        = H_N_PATCHES * H_N_PATCHES; // 2401
localparam int H_MASK_WIN_ELEMS = H_N_PATCHES;               // 49

// =============================================================================
// State encoding
// =============================================================================
typedef enum logic [6:0] {
    // ── Conv ──────────────────────────────────────────────────────────────
    S_IDLE              = 7'd0,
    S_C_INIT_PRELOAD    = 7'd1,
    S_C_INIT_WMEM_SWAP  = 7'd2,
    S_C_LOAD_W          = 7'd3,
    S_C_SWAP_W          = 7'd4,
    S_C_LOAD_IMG        = 7'd5,
    S_C_SWAP_IMG        = 7'd6,
    S_C_COMPUTE         = 7'd7,
    S_C_WAIT_OUT        = 7'd8,
    S_C_WRITEBACK       = 7'd9,
    S_C_NEXT            = 7'd10,
    // ── MLP ───────────────────────────────────────────────────────────────
    S_M_L1_PRELOAD0     = 7'd11,
    S_M_L1_WMEM_SWAP0   = 7'd12,
    S_M_L1_LOAD_X       = 7'd13,
    S_M_L1_SWAP_X       = 7'd14,
    S_M_L1_LOAD_W       = 7'd15,
    S_M_L1_SWAP_W       = 7'd16,
    S_M_L1_COMPUTE      = 7'd17,
    S_M_L1_NEXT_COL     = 7'd18,
    S_M_L2_PRELOAD0     = 7'd19,
    S_M_L2_WMEM_SWAP0   = 7'd20,
    S_M_L2_LOAD_W       = 7'd21,
    S_M_L2_SWAP_W       = 7'd22,
    S_M_L2_COMPUTE      = 7'd23,
    S_M_WRITEBACK       = 7'd24,
    S_M_L2_NEXT_COL     = 7'd25,
    S_M_NEXT_ROW        = 7'd26,
    S_DONE              = 7'd27,
    // ── MHA — QKV projection ──────────────────────────────────────────────
    S_H_LOAD_INPUT      = 7'd28,
    S_H_SWAP_INPUT      = 7'd29,
    S_H_LOAD_WQKV       = 7'd30,
    S_H_SWAP_WQKV       = 7'd31,
    S_H_COMPUTE_QKV     = 7'd32,
    S_H_WRITEBACK_QKV   = 7'd33,
    S_H_NEXT_QKV_COL    = 7'd34,
    S_H_NEXT_PATCH_GRP  = 7'd35,
    S_H_NEXT_QKV_MAT    = 7'd36,
    // ── MHA — Attention QKᵀ ───────────────────────────────────────────────
    S_H_LOAD_QH         = 7'd37,
    S_H_SWAP_QH         = 7'd38,
    S_H_LOAD_WKT        = 7'd39,
    S_H_SWAP_WKT        = 7'd40,
    S_H_COMPUTE_ATTN    = 7'd41,
    S_H_WRITEBACK_ATTN  = 7'd42,
    S_H_NEXT_ATTN_COL   = 7'd43,
    S_H_NEXT_ATTN_ROWGRP= 7'd44,
    S_H_NEXT_ATTN_HD    = 7'd45,
    // ── MHA — SxV ─────────────────────────────────────────────────────────
    S_H_LOAD_SH         = 7'd46,
    S_H_SWAP_SH         = 7'd47,
    S_H_LOAD_VH         = 7'd48,
    S_H_SWAP_VH         = 7'd49,
    S_H_COMPUTE_SXV     = 7'd50,
    S_H_WRITEBACK_SXV   = 7'd51,
    S_H_NEXT_SXV_COL    = 7'd52,
    S_H_NEXT_SXV_ROWGRP = 7'd53,
    S_H_NEXT_SXV_HD     = 7'd54,
    // ── MHA — W_proj ──────────────────────────────────────────────────────
    S_H_PROJ_LOAD_INPUT = 7'd55,
    S_H_PROJ_SWAP_INPUT = 7'd56,
    S_H_PROJ_LOAD_W     = 7'd57,
    S_H_PROJ_SWAP_W     = 7'd58,
    S_H_PROJ_COMPUTE    = 7'd59,
    S_H_PROJ_WRITEBACK  = 7'd60,
    S_H_PROJ_NEXT_COL   = 7'd61,
    S_H_PROJ_NEXT_ROWGRP= 7'd62,
    S_H_SHORTCUT1       = 7'd63,
    // ── MHA — FFN1 ────────────────────────────────────────────────────────
    S_H_FFN1_LOAD_INPUT = 7'd64,
    S_H_FFN1_SWAP_INPUT = 7'd65,
    S_H_FFN1_LOAD_W     = 7'd66,
    S_H_FFN1_SWAP_W     = 7'd67,
    S_H_FFN1_COMPUTE    = 7'd68,
    S_H_FFN1_WRITEBACK  = 7'd69,
    S_H_FFN1_NEXT_COL   = 7'd70,
    S_H_FFN1_NEXT_ROWGRP= 7'd71,
    S_H_GELU_WAIT       = 7'd72,
    // ── MHA — FFN2 ────────────────────────────────────────────────────────
    S_H_FFN2_LOAD_INPUT = 7'd73,
    S_H_FFN2_SWAP_INPUT = 7'd74,
    S_H_FFN2_LOAD_W     = 7'd75,
    S_H_FFN2_SWAP_W     = 7'd76,
    S_H_FFN2_COMPUTE    = 7'd77,
    S_H_FFN2_WRITEBACK  = 7'd78,
    S_H_FFN2_NEXT_COL   = 7'd79,
    S_H_FFN2_NEXT_ROWGRP= 7'd80,
    S_H_SHORTCUT2       = 7'd81,
    S_H_NEXT_WINDOW     = 7'd82,
    // ── MHA — Mask apply (NEW) ────────────────────────────────────────────
    // These three states execute per attention head, between QK^T completion
    // and S_H_NEXT_ATTN_HD, applying the attention bias mask in-place.
    //
    //  S_H_MASK_RD:
    //      Assert fb_rd_en + fb_rd_addr for one ILB score element.
    //      (1-cycle state; memory has 1-cycle latency so data appears next.)
    //
    //  S_H_MASK_WB:
    //      fb_rd_data is valid. Assert omem_wr_en + omem_wr_addr to write
    //      back (fb_rd_data + mask_data_out), formed in full_system_top.
    //      Increment h_mask_elem_cnt.
    //      If last element of window → S_H_MASK_NEXT_WIN, else → S_H_MASK_RD.
    //
    //  S_H_MASK_NEXT_WIN:
    //      Pulse mask_next_window for one cycle (advances mask_buffer.rd_ptr).
    //      Reset h_mask_elem_cnt.
    //      If mask_all_done → S_H_NEXT_ATTN_HD, else → S_H_MASK_RD.
    S_H_MASK_RD        = 7'd83,
    S_H_MASK_WB        = 7'd84,
    S_H_MASK_NEXT_WIN  = 7'd85
} state_t;

state_t state, next_state;

// =============================================================================
// Counters
// =============================================================================

// Conv
logic [6:0] c_kernel_idx;
logic [5:0] c_row_group_idx;
logic [2:0] c_chunk_idx;
logic [2:0] c_wb_cnt;

// MLP
logic [8:0] m_row_grp_idx;
logic [8:0] m_l1_col_idx;
logic [6:0] m_l2_col_idx;
logic [3:0] m_compute_cnt;
logic [2:0] m_wb_cnt;

// MHA
logic [5:0]  h_win_idx;
logic [1:0]  h_qkv_mat;
logic [6:0]  h_qkv_col_idx;
logic [2:0]  h_patch_grp;

logic [1:0]  h_head_idx;
logic [5:0]  h_attn_col_idx;
logic [2:0]  h_attn_rowgrp;
logic [4:0]  h_sxv_col_idx;
logic [2:0]  h_sxv_rowgrp;

logic [6:0]  h_proj_col_idx;
logic [2:0]  h_proj_rowgrp;
logic [8:0]  h_ffn1_col_idx;
logic [2:0]  h_ffn1_rowgrp;
logic [6:0]  h_ffn2_col_idx;

logic [2:0]  h_ffn2_rowgrp;
logic [2:0]  h_compute_cnt;
logic [2:0]  h_wb_cnt;

// NEW: element index within current mask sub-block (0..48)
logic [5:0]  h_mask_elem_cnt;

// Shared load cycle
logic [9:0] load_cyc;
logic       is_data_ph;
logic [8:0] load_cnt;
assign is_data_ph = load_cyc[0];
assign load_cnt   = load_cyc[9:1];

// =============================================================================
// Last-element flags — Conv / MLP
// =============================================================================
logic c_last_chunk, c_last_row_group, c_last_kernel;
assign c_last_chunk     = (c_chunk_idx     == C_N_CHUNKS      - 1);
assign c_last_row_group = (c_row_group_idx == C_N_ROW_GROUPS  - 1);
assign c_last_kernel    = (c_kernel_idx    == C_N_KERNELS      - 1);

logic m_last_row_grp, m_last_l1_col, m_last_l2_col, m_last_wb;
assign m_last_row_grp = (m_row_grp_idx == M_N_ROW_GRPS - 1);
assign m_last_l1_col  = (m_l1_col_idx  == M_N_L1_COLS  - 1);
assign m_last_l2_col  = (m_l2_col_idx  == M_N_L2_COLS  - 1);
assign m_last_wb      = (m_wb_cnt      == M_OUT_WORDS   - 1);

// =============================================================================
// Last-element flags — MHA
// =============================================================================
logic h_last_win, h_last_qkv_col, h_last_patch_grp;
logic h_last_head, h_last_attn_col, h_last_attn_rowgrp;
logic h_last_sxv_col, h_last_sxv_rowgrp;
logic h_last_proj_col, h_last_proj_rowgrp;

logic h_last_ffn1_col, h_last_ffn1_rowgrp;
logic h_last_ffn2_col, h_last_ffn2_rowgrp;
// NEW
logic h_last_mask_elem;

assign h_last_win         = (h_win_idx       == H_N_WINDOWS  - 1);
assign h_last_qkv_col     = (h_qkv_col_idx   == H_C_IN       - 1);
assign h_last_patch_grp   = (h_patch_grp     == (H_N_PATCHES / 7));
assign h_last_head        = (h_head_idx      == H_N_HEADS    - 1);
assign h_last_attn_col    = (h_attn_col_idx  == H_N_PATCHES  - 1);
assign h_last_attn_rowgrp = (h_attn_rowgrp   == (H_N_PATCHES / 7));
assign h_last_sxv_col     = (h_sxv_col_idx   == H_HEAD_DIM   - 1);
assign h_last_sxv_rowgrp  = (h_sxv_rowgrp    == (H_N_PATCHES / 7));
assign h_last_proj_col    = (h_proj_col_idx  == H_C_IN       - 1);
assign h_last_proj_rowgrp = (h_proj_rowgrp   == (H_N_PATCHES / 7));
assign h_last_ffn1_col    = (h_ffn1_col_idx  == H_C_FFN      - 1);
assign h_last_ffn1_rowgrp = (h_ffn1_rowgrp   == (H_N_PATCHES / 7));
assign h_last_ffn2_col    = (h_ffn2_col_idx  == H_C_IN       - 1);
assign h_last_ffn2_rowgrp = (h_ffn2_rowgrp   == (H_N_PATCHES / 7));

// NEW: true when h_mask_elem_cnt points at the last element of a sub-block
assign h_last_mask_elem   = (h_mask_elem_cnt == 6'(H_MASK_WIN_ELEMS - 1));

// =============================================================================
// Address generation — Conv / MLP (unchanged)
// =============================================================================
logic [WAW-1:0] c_wmem_rd_addr;
assign c_wmem_rd_addr = WAW'(c_kernel_idx) * C_WLOAD_WORDS + WAW'(load_cnt);

logic [3:0]     c_img_pe;
logic [2:0]     c_img_win;

logic [1:0]     c_img_ch;
logic [7:0]     c_img_row;
logic [OAW-1:0] c_imem_addr;

always_comb begin
    c_img_pe    = 4'(load_cnt / 7);
    c_img_win   = 3'(load_cnt % 7);
    c_img_ch    = c_img_pe[3:2];
    c_img_row   = {2'b0, c_row_group_idx} * 4 + {6'b0, c_img_pe[1:0]};
    c_imem_addr = OAW'(c_img_ch)    * C_IMG_CH_WORDS
                + OAW'(c_img_row)   * C_IMG_W_WORDS
                + OAW'(c_chunk_idx) * 7
                + OAW'(c_img_win);
end

logic [OAW-1:0] c_omem_addr;
assign c_omem_addr = OAW'(c_kernel_idx)    * C_OUT_K_WORDS
                   + OAW'(c_row_group_idx) * C_OUT_ROW_WORDS
                   + OAW'(c_chunk_idx)     * C_OUT_WORDS
                   + OAW'(c_wb_cnt);

logic [WAW-1:0] m_w1_rd_addr;
assign m_w1_rd_addr = WAW'(m_l1_col_idx) * M_W1_WORDS + WAW'(load_cnt);

logic [WAW-1:0] m_w2_rd_addr;
assign m_w2_rd_addr = WAW'(W2_BASE)
                    + WAW'(m_l2_col_idx) * M_W2_WORDS
                    + WAW'(load_cnt);

logic [7:0]     m_x_sub_row;
logic [4:0]     m_x_k_word;
logic [OAW-1:0] m_xmem_addr;

always_comb begin
    m_x_sub_row = 8'(load_cnt / 24);
    m_x_k_word  = 5'(load_cnt % 24);
    m_xmem_addr = (OAW'(m_row_grp_idx) * 7 + OAW'(m_x_sub_row)) * 24
                + OAW'(m_x_k_word);
end

logic [OAW-1:0] m_omem_addr;
assign m_omem_addr = OAW'(m_l2_col_idx) * 3136
                   + OAW'(m_row_grp_idx) * M_OUT_WORDS
                   + OAW'(m_wb_cnt);

logic [WAW-1:0] c_shadow_next_addr;
assign c_shadow_next_addr = WAW'(c_kernel_idx + 1) * C_WLOAD_WORDS + WAW'(load_cnt);

logic [WAW-1:0] m_shadow_w1_next_addr;
assign m_shadow_w1_next_addr = WAW'(m_l1_col_idx + 1) * M_W1_WORDS + WAW'(load_cnt);

logic [WAW-1:0] m_shadow_w2_next_addr;
assign m_shadow_w2_next_addr = WAW'(W2_BASE)
                             + WAW'(m_l2_col_idx + 1) * M_W2_WORDS
                             + WAW'(load_cnt);

// =============================================================================
// Address generation — MHA (unchanged except new mask block below)
// =============================================================================

function automatic logic [WAW-1:0] qkv_base(input logic [1:0] mat);
    case (mat)
        2'd0: return WAW'(WQ_BASE);
        2'd1: return WAW'(WK_BASE);
        2'd2: return WAW'(WV_BASE);
        default: return WAW'(WQ_BASE);
    endcase
endfunction

logic [WAW-1:0] h_wqkv_rd_addr;
assign h_wqkv_rd_addr = qkv_base(h_qkv_mat)
                      + WAW'(h_qkv_col_idx) * H_W_QKV_WORDS
                      + WAW'(load_cnt);

logic [FAW-1:0] h_fib_patch_addr;
always_comb begin
    automatic int global_patch = int'(h_win_idx)   * H_N_PATCHES
                               + int'(h_patch_grp) * 7
                               + int'(load_cnt) / 24;
    automatic int k_word       = int'(load_cnt) % 24;
    h_fib_patch_addr = FAW'(global_patch * 24 + k_word);
end

function automatic logic [OAW-1:0] qkv_ilb_base(input logic [1:0] mat);
    case (mat)
        2'd0: return OAW'(ILB_Q_BASE);
        2'd1: return OAW'(ILB_K_BASE);
        2'd2: return OAW'(ILB_V_BASE);
        default: return OAW'(ILB_Q_BASE);
    endcase
endfunction

logic [OAW-1:0] h_qkv_omem_addr;
assign h_qkv_omem_addr = qkv_ilb_base(h_qkv_mat)
                       + OAW'(h_qkv_col_idx) * H_N_PATCHES
                       + OAW'(h_patch_grp) * 7
                       + OAW'(h_wb_cnt);

logic [OAW-1:0] h_qh_rd_addr;
assign h_qh_rd_addr = OAW'(ILB_Q_BASE)
                    + OAW'(h_head_idx)      * (H_N_PATCHES * H_HEAD_DIM)
                    + OAW'(h_attn_col_idx)
                    + OAW'(h_attn_rowgrp)   * 7
                    + OAW'(load_cnt);

logic [OAW-1:0] h_kt_rd_addr;
assign h_kt_rd_addr = OAW'(ILB_K_BASE)
                    + OAW'(h_head_idx)      * (H_N_PATCHES * H_HEAD_DIM)
                    + OAW'(h_attn_col_idx)  * H_HEAD_DIM
                    + OAW'(load_cnt);

logic [OAW-1:0] h_attn_omem_addr;
assign h_attn_omem_addr = OAW'(ILB_S_BASE)
                        + OAW'(h_head_idx)     * (H_N_PATCHES * H_N_PATCHES)
                        + OAW'(h_attn_col_idx) * H_N_PATCHES
                        + OAW'(h_attn_rowgrp)  * 7
                        + OAW'(h_wb_cnt);

logic [OAW-1:0] h_sh_rd_addr;
assign h_sh_rd_addr = OAW'(ILB_S_BASE)
                    + OAW'(h_head_idx)    * (H_N_PATCHES * H_N_PATCHES)
                    + OAW'(h_sxv_col_idx)
                    + OAW'(h_sxv_rowgrp)  * 7
                    + OAW'(load_cnt);

logic [OAW-1:0] h_vh_rd_addr;
assign h_vh_rd_addr = OAW'(ILB_V_BASE)
                    + OAW'(h_head_idx)    * (H_N_PATCHES * H_HEAD_DIM)
                    + OAW'(h_sxv_col_idx) * H_N_PATCHES
                    + OAW'(load_cnt);

logic [OAW-1:0] h_sxv_omem_addr;
assign h_sxv_omem_addr = OAW'(ILB_A_BASE)
                       + OAW'(h_head_idx)    * (H_N_PATCHES * H_HEAD_DIM)
                       + OAW'(h_sxv_col_idx) * H_N_PATCHES
                       + OAW'(h_sxv_rowgrp)  * 7
                       + OAW'(h_wb_cnt);

logic [WAW-1:0] h_wproj_rd_addr;
assign h_wproj_rd_addr = WAW'(WPROJ_BASE)
                       + WAW'(h_proj_col_idx) * H_W_QKV_WORDS
                       + WAW'(load_cnt);

logic [WAW-1:0] h_wffn1_rd_addr;
assign h_wffn1_rd_addr = WAW'(WFFN1_BASE)
                       + WAW'(h_ffn1_col_idx) * H_W_QKV_WORDS
                       + WAW'(load_cnt);

logic [WAW-1:0] h_wffn2_rd_addr;
assign h_wffn2_rd_addr = WAW'(WFFN2_BASE)
                       + WAW'(h_ffn2_col_idx) * (H_C_FFN / 4)
                       + WAW'(load_cnt);

logic [OAW-1:0] h_proj_in_rd_addr;
assign h_proj_in_rd_addr = OAW'(ILB_A_BASE)
                         + OAW'(h_proj_rowgrp) * 7 * H_W_QKV_WORDS
                         + OAW'(load_cnt);

logic [OAW-1:0] h_proj_omem_addr;
assign h_proj_omem_addr = OAW'(ILB_PROJ_BASE)
                        + OAW'(h_proj_col_idx) * H_N_PATCHES
                        + OAW'(h_proj_rowgrp)  * 7
                        + OAW'(h_wb_cnt);

logic [OAW-1:0] h_ffn1_in_rd_addr;
assign h_ffn1_in_rd_addr = OAW'(ILB_PROJ_BASE)
                         + OAW'(h_ffn1_rowgrp) * 7 * H_W_QKV_WORDS
                         + OAW'(load_cnt);

logic [OAW-1:0] h_ffn1_omem_addr;
assign h_ffn1_omem_addr = OAW'(ILB_FFN1_BASE)
                        + OAW'(h_ffn1_col_idx) * H_N_PATCHES
                        + OAW'(h_ffn1_rowgrp)  * 7
                        + OAW'(h_wb_cnt);

logic [OAW-1:0] h_ffn2_in_rd_addr;
assign h_ffn2_in_rd_addr = OAW'(ILB_FFN1_BASE)
                         + OAW'(h_ffn2_rowgrp) * 7 * (H_C_FFN / 4)
                         + OAW'(load_cnt);

logic [OAW-1:0] h_ffn2_omem_addr;
assign h_ffn2_omem_addr = OAW'(ILB_PROJ_BASE)
                        + OAW'(h_ffn2_col_idx) * H_N_PATCHES
                        + OAW'(h_ffn2_rowgrp)  * 7
                        + OAW'(h_wb_cnt);

// =============================================================================
// Address generation — Mask apply (NEW)
// =============================================================================
// ILB score element address:
//   ILB_S_BASE
//   + head_idx  × H_S_ELEMS          (2401 words per head)
//   + window_idx × H_MASK_WIN_ELEMS   (49 words per sub-block)
//   + element_idx                     (0..48 within the sub-block)
//
// mask_window_idx comes directly from mask_buffer (combinational) and is
// stable for the entire 49-element R-M-W pass of each sub-block.
// h_mask_elem_cnt is the sequential element counter driven by the FSM.
// =============================================================================
logic [OAW-1:0] h_mask_rd_addr;

always_comb begin
    h_mask_rd_addr = OAW'(ILB_S_BASE)
                   + OAW'(h_head_idx)      * OAW'(H_S_ELEMS)
                   + OAW'(mask_window_idx) * OAW'(H_MASK_WIN_ELEMS)
                   + OAW'(h_mask_elem_cnt);
end

// Write-back address: registered copy of the read address issued one cycle
// earlier (output_memory has 1-cycle read latency; wb uses the same address).
logic [OAW-1:0] h_mask_wb_addr;
always_ff @(posedge clk) begin
    if (state == S_H_MASK_RD)
        h_mask_wb_addr <= h_mask_rd_addr;
end

// =============================================================================
// State register
// =============================================================================
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= S_IDLE;
    else        state <= next_state;
end

// =============================================================================
// Next-state logic
// =============================================================================
always_comb begin
    next_state = state;

    case (state)

        S_IDLE: begin
            if (start)
                case (mode)
                    2'b00:   next_state = S_C_INIT_PRELOAD;
                    2'b01:   next_state = S_M_L1_PRELOAD0;
                    2'b10:   next_state = S_H_LOAD_INPUT;
                    default: next_state = S_IDLE;
                endcase
        end

        // ── Conv ──────────────────────────────────────────────────────────
        S_C_INIT_PRELOAD:
            if (load_cyc == C_WLOAD_CYCS - 1) next_state = S_C_INIT_WMEM_SWAP;
        S_C_INIT_WMEM_SWAP:                    next_state = S_C_LOAD_W;

        S_C_LOAD_W:
            if (load_cyc == C_WLOAD_CYCS - 1) next_state = S_C_SWAP_W;
        S_C_SWAP_W:                            next_state = S_C_LOAD_IMG;

        S_C_LOAD_IMG:
            if (load_cyc == C_ILOAD_CYCS - 1) next_state = S_C_SWAP_IMG;
        S_C_SWAP_IMG:                          next_state = S_C_COMPUTE;

        S_C_COMPUTE:                           next_state = S_C_WAIT_OUT;

        S_C_WAIT_OUT:                          next_state = S_C_WRITEBACK;

        S_C_WRITEBACK:
            if (c_wb_cnt == C_OUT_WORDS - 1)   next_state = S_C_NEXT;

        S_C_NEXT: begin
            if      (c_last_chunk && c_last_row_group && c_last_kernel)
                next_state = S_DONE;
            else if (c_last_chunk && c_last_row_group)
                next_state = S_C_LOAD_W;
            else
                next_state = S_C_LOAD_IMG;
        end

        // ── MLP ───────────────────────────────────────────────────────────
        S_M_L1_PRELOAD0:
            if (load_cyc == M_W1_CYCS - 1)    next_state = S_M_L1_WMEM_SWAP0;
        S_M_L1_WMEM_SWAP0:                    next_state = S_M_L1_LOAD_X;

        S_M_L1_LOAD_X:
            if (load_cyc == M_X_CYCS  - 1)    next_state = S_M_L1_SWAP_X;
        S_M_L1_SWAP_X:                        next_state = S_M_L1_LOAD_W;

        S_M_L1_LOAD_W:
            if (load_cyc == M_W1_CYCS - 1)    next_state = S_M_L1_SWAP_W;
        S_M_L1_SWAP_W:                        next_state = S_M_L1_COMPUTE;

        S_M_L1_COMPUTE:
            if (m_compute_cnt == M_N_ACC_L1)   next_state = S_M_L1_NEXT_COL;

        S_M_L1_NEXT_COL: begin
            if (m_last_l1_col) next_state = S_M_L2_PRELOAD0;
            else               next_state = S_M_L1_LOAD_W;
        end
        S_M_L2_PRELOAD0:
            if (load_cyc == M_W2_CYCS - 1)    next_state = S_M_L2_WMEM_SWAP0;
        S_M_L2_WMEM_SWAP0:                    next_state = S_M_L2_LOAD_W;

        S_M_L2_LOAD_W:
            if (load_cyc == M_W2_CYCS - 1)    next_state = S_M_L2_SWAP_W;
        S_M_L2_SWAP_W:                        next_state = S_M_L2_COMPUTE;

        S_M_L2_COMPUTE:
            if (m_compute_cnt == M_N_ACC_L2)   next_state = S_M_WRITEBACK;

        S_M_WRITEBACK:
            if (m_last_wb)                     next_state = S_M_L2_NEXT_COL;

        S_M_L2_NEXT_COL: begin
            if (m_last_l2_col) next_state = S_M_NEXT_ROW;
            else               next_state = S_M_L2_LOAD_W;
        end
        S_M_NEXT_ROW: begin
            if (m_last_row_grp) next_state = S_DONE;
            else                next_state = S_M_L1_PRELOAD0;
        end

        // ── MHA — QKV ─────────────────────────────────────────────────────
        S_H_LOAD_INPUT:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_SWAP_INPUT;
        S_H_SWAP_INPUT:   next_state = S_H_LOAD_WQKV;
        S_H_LOAD_WQKV:
            if (load_cyc == H_W_QKV_CYCS - 1)   next_state = S_H_SWAP_WQKV;
        S_H_SWAP_WQKV:    next_state = S_H_COMPUTE_QKV;
        S_H_COMPUTE_QKV:
            if (h_compute_cnt == H_N_ACC_QKV)    next_state = S_H_WRITEBACK_QKV;
        S_H_WRITEBACK_QKV:
            if (h_wb_cnt == H_OUT_WORDS - 1)     next_state = S_H_NEXT_QKV_COL;

        S_H_NEXT_QKV_COL: begin
            if (h_last_qkv_col) next_state = S_H_NEXT_PATCH_GRP;
            else                next_state = S_H_LOAD_WQKV;
        end
        S_H_NEXT_PATCH_GRP: begin
            if (h_last_patch_grp) next_state = S_H_NEXT_QKV_MAT;
            else                  next_state = S_H_LOAD_INPUT;
        end
        S_H_NEXT_QKV_MAT: begin
            if (h_qkv_mat == 2'd2) next_state = S_H_LOAD_QH;
            else                   next_state = S_H_LOAD_INPUT;
        end

        // ── MHA — Attention QKᵀ ───────────────────────────────────────────
        S_H_LOAD_QH:
            if (load_cyc == H_QH_LOAD_CYCS - 1) next_state = S_H_SWAP_QH;
        S_H_SWAP_QH:   next_state = S_H_LOAD_WKT;
        S_H_LOAD_WKT:
            if (load_cyc == H_WKT_CYCS - 1)     next_state = S_H_SWAP_WKT;
        S_H_SWAP_WKT:  next_state = S_H_COMPUTE_ATTN;
        S_H_COMPUTE_ATTN:
            if (h_compute_cnt == H_N_ACC_ATTN)   next_state = S_H_WRITEBACK_ATTN;
        S_H_WRITEBACK_ATTN:
            if (h_wb_cnt == H_OUT_WORDS - 1)     next_state = S_H_NEXT_ATTN_COL;

        S_H_NEXT_ATTN_COL: begin
            if (h_last_attn_col) next_state = S_H_NEXT_ATTN_ROWGRP;
            else                 next_state = S_H_LOAD_WKT;
        end
        S_H_NEXT_ATTN_ROWGRP: begin
            // CHANGED (rev 4): when the last row-group of this head's 49×49
            // score is done, enter the mask pass instead of going directly to
            // S_H_NEXT_ATTN_HD. Masking completes before the head counter
            // advances, so every head's score matrix is masked in-place before
            // S×V reads it.
            if (h_last_attn_rowgrp) next_state = S_H_MASK_RD;   // ← changed
            else                    next_state = S_H_LOAD_QH;
        end
        S_H_NEXT_ATTN_HD: begin
            if (h_last_head) next_state = S_H_LOAD_SH;
            else             next_state = S_H_LOAD_QH;
        end

        // ── MHA — Mask apply (NEW) ─────────────────────────────────────────
        // S_H_MASK_RD: issue ILB read for element h_mask_elem_cnt of sub-block
        //              mask_window_idx. Always takes exactly 1 cycle.
        S_H_MASK_RD:
            next_state = S_H_MASK_WB;

        // S_H_MASK_WB: ILB data valid; write masked result back.
        //              Increment element counter.
        //              If last element of this window → S_H_MASK_NEXT_WIN.
        //              Else → S_H_MASK_RD for the next element.
        S_H_MASK_WB: begin
            if (h_last_mask_elem) next_state = S_H_MASK_NEXT_WIN;
            else                  next_state = S_H_MASK_RD;
        end

        // S_H_MASK_NEXT_WIN: pulse mask_next_window to advance mask_buffer.
        //                    mask_all_done (from mask_buffer) tells us whether
        //                    all 49 sub-blocks have been processed.
        //                    If done → advance head counter.
        //                    Else    → start next sub-block.
        S_H_MASK_NEXT_WIN: begin
            if (mask_all_done) next_state = S_H_NEXT_ATTN_HD;
            else               next_state = S_H_MASK_RD;
        end

        // ── MHA — SxV ─────────────────────────────────────────────────────
        S_H_LOAD_SH:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_SWAP_SH;
        S_H_SWAP_SH:   next_state = S_H_LOAD_VH;
        S_H_LOAD_VH:
            if (load_cyc == H_WKT_CYCS - 1)     next_state = S_H_SWAP_VH;
        S_H_SWAP_VH:   next_state = S_H_COMPUTE_SXV;
        S_H_COMPUTE_SXV:
            if (h_compute_cnt == H_N_ACC_SXV)    next_state = S_H_WRITEBACK_SXV;
        S_H_WRITEBACK_SXV:
            if (h_wb_cnt == H_OUT_WORDS - 1)     next_state = S_H_NEXT_SXV_COL;

        S_H_NEXT_SXV_COL: begin
            if (h_last_sxv_col) next_state = S_H_NEXT_SXV_ROWGRP;
            else                next_state = S_H_LOAD_VH;
        end
        S_H_NEXT_SXV_ROWGRP: begin
            if (h_last_sxv_rowgrp) next_state = S_H_NEXT_SXV_HD;
            else                   next_state = S_H_LOAD_SH;
        end
        S_H_NEXT_SXV_HD: begin
            if (h_last_head) next_state = S_H_PROJ_LOAD_INPUT;
            else             next_state = S_H_LOAD_SH;
        end

        // ── MHA — W_proj ──────────────────────────────────────────────────
        S_H_PROJ_LOAD_INPUT:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_PROJ_SWAP_INPUT;
        S_H_PROJ_SWAP_INPUT: next_state = S_H_PROJ_LOAD_W;
        S_H_PROJ_LOAD_W:
            if (load_cyc == H_W_QKV_CYCS - 1)   next_state = S_H_PROJ_SWAP_W;
        S_H_PROJ_SWAP_W:     next_state = S_H_PROJ_COMPUTE;
        S_H_PROJ_COMPUTE:
            if (h_compute_cnt == H_N_ACC_PROJ)   next_state = S_H_PROJ_WRITEBACK;
        S_H_PROJ_WRITEBACK:
            if (h_wb_cnt == H_OUT_WORDS - 1)     next_state = S_H_PROJ_NEXT_COL;

        S_H_PROJ_NEXT_COL: begin
            if (h_last_proj_col) next_state = S_H_PROJ_NEXT_ROWGRP;
            else                 next_state = S_H_PROJ_LOAD_W;
        end
        S_H_PROJ_NEXT_ROWGRP: begin
            if (h_last_proj_rowgrp) next_state = S_H_SHORTCUT1;
            else                    next_state = S_H_PROJ_LOAD_INPUT;
        end
        S_H_SHORTCUT1: next_state = S_H_FFN1_LOAD_INPUT;

        // ── MHA — FFN1 ────────────────────────────────────────────────────
        S_H_FFN1_LOAD_INPUT:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_FFN1_SWAP_INPUT;
        S_H_FFN1_SWAP_INPUT: next_state = S_H_FFN1_LOAD_W;
        S_H_FFN1_LOAD_W:
            if (load_cyc == H_W_QKV_CYCS - 1)   next_state = S_H_FFN1_SWAP_W;
        S_H_FFN1_SWAP_W:     next_state = S_H_FFN1_COMPUTE;
        S_H_FFN1_COMPUTE:
            if (h_compute_cnt == H_N_ACC_FFN1)   next_state = S_H_FFN1_WRITEBACK;
        S_H_FFN1_WRITEBACK:
            if (h_wb_cnt == H_OUT_WORDS - 1)     next_state = S_H_FFN1_NEXT_COL;

        S_H_FFN1_NEXT_COL: begin
            if (h_last_ffn1_col) next_state = S_H_FFN1_NEXT_ROWGRP;
            else                 next_state = S_H_FFN1_LOAD_W;
        end
        S_H_FFN1_NEXT_ROWGRP: begin
            if (h_last_ffn1_rowgrp) next_state = S_H_GELU_WAIT;
            else                    next_state = S_H_FFN1_LOAD_INPUT;
        end
        S_H_GELU_WAIT: next_state = S_H_FFN2_LOAD_INPUT;

        // ── MHA — FFN2 ────────────────────────────────────────────────────
        S_H_FFN2_LOAD_INPUT:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_FFN2_SWAP_INPUT;
        S_H_FFN2_SWAP_INPUT: next_state = S_H_FFN2_LOAD_W;
        S_H_FFN2_LOAD_W:
            if (load_cyc == (H_C_FFN/4)*2 - 1)  next_state = S_H_FFN2_SWAP_W;
        S_H_FFN2_SWAP_W:     next_state = S_H_FFN2_COMPUTE;
        S_H_FFN2_COMPUTE:
            if (h_compute_cnt == H_N_ACC_FFN2)   next_state = S_H_FFN2_WRITEBACK;
        S_H_FFN2_WRITEBACK:
            if (h_wb_cnt == H_OUT_WORDS - 1)     next_state = S_H_FFN2_NEXT_COL;

        S_H_FFN2_NEXT_COL: begin
            if (h_last_ffn2_col) next_state = S_H_FFN2_NEXT_ROWGRP;
            else                 next_state = S_H_FFN2_LOAD_W;
        end
        S_H_FFN2_NEXT_ROWGRP: begin
            if (h_last_ffn2_rowgrp) next_state = S_H_SHORTCUT2;
            else                    next_state = S_H_FFN2_LOAD_INPUT;
        end
        S_H_SHORTCUT2:  next_state = S_H_NEXT_WINDOW;

        S_H_NEXT_WINDOW: begin
            if (h_last_win) next_state = S_DONE;
            else            next_state = S_H_LOAD_INPUT;
        end

        S_DONE:  next_state = S_DONE;
        default: next_state = S_IDLE;

    endcase
end

// =============================================================================
// Counter updates
// =============================================================================
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        c_kernel_idx    <= '0;
        c_row_group_idx <= '0;
        c_chunk_idx     <= '0;
        c_wb_cnt        <= '0;
        m_row_grp_idx   <= '0;
        m_l1_col_idx    <= '0;
        m_l2_col_idx    <= '0; m_compute_cnt    <= '0;
        m_wb_cnt        <= '0;
        load_cyc        <= '0;
        h_win_idx       <= '0;
        h_qkv_mat       <= '0;
        h_qkv_col_idx   <= '0;
        h_patch_grp     <= '0;
        h_head_idx      <= '0;
        h_attn_col_idx  <= '0;
        h_attn_rowgrp   <= '0; h_sxv_col_idx    <= '0;
        h_sxv_rowgrp    <= '0; h_proj_col_idx   <= '0;
        h_proj_rowgrp   <= '0;
        h_ffn1_col_idx  <= '0;
        h_ffn1_rowgrp   <= '0; h_ffn2_col_idx   <= '0;
        h_ffn2_rowgrp   <= '0;
        h_compute_cnt   <= '0;
        h_wb_cnt        <= '0;
        h_mask_elem_cnt <= '0;
        // NEW
    end else begin
        case (state)

            // ── load_cyc resets ───────────────────────────────────────────
            S_IDLE,
            S_C_INIT_WMEM_SWAP, S_C_SWAP_W,    S_C_SWAP_IMG,   S_C_NEXT,
            S_M_L1_WMEM_SWAP0,  S_M_L1_SWAP_X, S_M_L1_SWAP_W,  S_M_L1_NEXT_COL,
            S_M_L2_WMEM_SWAP0,  S_M_L2_SWAP_W, S_M_L2_NEXT_COL, S_M_NEXT_ROW,
            S_H_SWAP_INPUT, S_H_SWAP_WQKV, S_H_SWAP_QH,  S_H_SWAP_WKT,
            S_H_SWAP_SH,    S_H_SWAP_VH,    S_H_PROJ_SWAP_INPUT, S_H_PROJ_SWAP_W,
            S_H_FFN1_SWAP_INPUT, S_H_FFN1_SWAP_W,
            S_H_FFN2_SWAP_INPUT, S_H_FFN2_SWAP_W:
                load_cyc <= '0;

            // ── Conv ──────────────────────────────────────────────────────
            S_C_INIT_PRELOAD, S_C_LOAD_W:
                load_cyc <= (load_cyc < C_WLOAD_CYCS - 1) ?
                            load_cyc + 1 : '0;
            S_C_LOAD_IMG:
                load_cyc <= (load_cyc < C_ILOAD_CYCS - 1) ?
                            load_cyc + 1 : '0;
            S_C_WRITEBACK:
                c_wb_cnt <= (c_wb_cnt < C_OUT_WORDS - 1) ?
                            c_wb_cnt + 1 : '0;
            S_C_NEXT: begin
                if (!c_last_chunk)
                    c_chunk_idx <= c_chunk_idx + 1;
                else begin
                    c_chunk_idx <= '0;
                    if (!c_last_row_group)
                        c_row_group_idx <= c_row_group_idx + 1;
                    else begin
                        c_row_group_idx <= '0;
                        if (!c_last_kernel)
                            c_kernel_idx <= c_kernel_idx + 1;
                    end
                end
            end

            // ── MLP ───────────────────────────────────────────────────────
            S_M_L1_PRELOAD0:
                load_cyc <= (load_cyc < M_W1_CYCS - 1) ?
                            load_cyc + 1 : '0;
            S_M_L1_LOAD_X:
                load_cyc <= (load_cyc < M_X_CYCS  - 1) ?
                            load_cyc + 1 : '0;
            S_M_L1_LOAD_W:
                load_cyc <= (load_cyc < M_W1_CYCS - 1) ?
                            load_cyc + 1 : '0;
            S_M_L1_COMPUTE:
                m_compute_cnt <= (m_compute_cnt < M_N_ACC_L1) ?
                                 m_compute_cnt + 1 : '0;
            S_M_L1_NEXT_COL: begin
                m_l1_col_idx  <= m_last_l1_col ?
                                 '0 : m_l1_col_idx + 1;
                m_compute_cnt <= '0;
            end
            S_M_L2_PRELOAD0:
                load_cyc <= (load_cyc < M_W2_CYCS - 1) ?
                            load_cyc + 1 : '0;
            S_M_L2_LOAD_W:
                load_cyc <= (load_cyc < M_W2_CYCS - 1) ?
                            load_cyc + 1 : '0;
            S_M_L2_COMPUTE:
                m_compute_cnt <= (m_compute_cnt < M_N_ACC_L2) ?
                                 m_compute_cnt + 1 : '0;
            S_M_WRITEBACK:
                m_wb_cnt <= (m_wb_cnt < M_OUT_WORDS - 1) ?
                            m_wb_cnt + 1 : '0;
            S_M_L2_NEXT_COL: begin
                m_l2_col_idx  <= m_last_l2_col ?
                                 '0 : m_l2_col_idx + 1;
                m_compute_cnt <= '0;
            end
            S_M_NEXT_ROW: begin
                m_row_grp_idx <= m_last_row_grp ?
                                 m_row_grp_idx : m_row_grp_idx + 1;
                m_l1_col_idx  <= '0; m_l2_col_idx <= '0; m_compute_cnt <= '0;
            end

            // ── MHA ───────────────────────────────────────────────────────
            S_H_LOAD_INPUT, S_H_LOAD_SH, S_H_PROJ_LOAD_INPUT,
            S_H_FFN1_LOAD_INPUT, S_H_FFN2_LOAD_INPUT:
                load_cyc <= (load_cyc < H_I_LOAD7_CYCS - 1) ?
                            load_cyc + 1 : '0;

            S_H_LOAD_WQKV, S_H_PROJ_LOAD_W, S_H_FFN1_LOAD_W:
                load_cyc <= (load_cyc < H_W_QKV_CYCS - 1)  ?
                            load_cyc + 1 : '0;

            S_H_LOAD_QH:
                load_cyc <= (load_cyc < H_QH_LOAD_CYCS - 1) ?
                            load_cyc + 1 : '0;

            S_H_LOAD_WKT, S_H_LOAD_VH:
                load_cyc <= (load_cyc < H_WKT_CYCS - 1)    ?
                            load_cyc + 1 : '0;

            S_H_FFN2_LOAD_W:
                load_cyc <= (load_cyc < (H_C_FFN/4)*2 - 1) ?
                            load_cyc + 1 : '0;

            // Compute counters
            S_H_COMPUTE_QKV, S_H_PROJ_COMPUTE, S_H_FFN1_COMPUTE:
                h_compute_cnt <= (h_compute_cnt < H_N_ACC_QKV) ?
                                 h_compute_cnt + 1 : '0;
            S_H_COMPUTE_ATTN:
                h_compute_cnt <= (h_compute_cnt < H_N_ACC_ATTN) ?
                                 h_compute_cnt + 1 : '0;
            S_H_COMPUTE_SXV:
                h_compute_cnt <= (h_compute_cnt < H_N_ACC_SXV) ?
                                 h_compute_cnt + 1 : '0;
            S_H_FFN2_COMPUTE:
                h_compute_cnt <= (h_compute_cnt < H_N_ACC_FFN2) ?
                                 h_compute_cnt + 1 : '0;

            // Writeback counters
            S_H_WRITEBACK_QKV, S_H_WRITEBACK_ATTN, S_H_WRITEBACK_SXV,
            S_H_PROJ_WRITEBACK, S_H_FFN1_WRITEBACK, S_H_FFN2_WRITEBACK:
                h_wb_cnt <= (h_wb_cnt < H_OUT_WORDS - 1) ?
                            h_wb_cnt + 1 : '0;

            // Column / group advance
            S_H_NEXT_QKV_COL: begin
                h_qkv_col_idx <= h_last_qkv_col ?
                                 '0 : h_qkv_col_idx + 1;
                h_compute_cnt <= '0; h_wb_cnt <= '0;
            end
            S_H_NEXT_PATCH_GRP: begin
                h_patch_grp <= h_last_patch_grp ?
                               '0 : h_patch_grp + 1;
            end
            S_H_NEXT_QKV_MAT: begin
                h_qkv_mat     <= (h_qkv_mat == 2'd2) ?
                                 '0 : h_qkv_mat + 1;
                h_patch_grp   <= '0;
                h_qkv_col_idx <= '0;
            end
            S_H_NEXT_ATTN_COL: begin
                h_attn_col_idx <= h_last_attn_col ?
                                  '0 : h_attn_col_idx + 1;
                h_compute_cnt  <= '0; h_wb_cnt <= '0;
            end
            S_H_NEXT_ATTN_ROWGRP: begin
                h_attn_rowgrp  <= h_last_attn_rowgrp ?
                                  '0 : h_attn_rowgrp + 1;
                h_attn_col_idx <= '0;
            end
            // CHANGED (rev 4): also reset h_mask_elem_cnt here for cleanliness,
            // though S_H_MASK_NEXT_WIN already resets it before we reach this state.
            S_H_NEXT_ATTN_HD: begin
                h_head_idx      <= h_last_head ?
                                   '0 : h_head_idx + 1;
                h_attn_rowgrp   <= '0; h_attn_col_idx   <= '0;
                h_mask_elem_cnt <= '0;
                // defensive reset
            end

            // ── Mask apply counters (NEW) ──────────────────────────────────
            // S_H_MASK_RD: no counter change — 1-cycle read issue state.
            S_H_MASK_RD: ; // intentionally empty

            // S_H_MASK_WB: increment element counter.
            // Wraps to 0 at the end of each 49-element sub-block window.
            S_H_MASK_WB: begin
                h_mask_elem_cnt <= h_last_mask_elem ?
                                   '0 : h_mask_elem_cnt + 1;
            end

            // S_H_MASK_NEXT_WIN: element counter already reset in S_H_MASK_WB
            // (wraps to 0 on the last element). No extra action needed here.
            S_H_MASK_NEXT_WIN: ; // intentionally empty

            // Remaining MHA advance states (unchanged)
            S_H_NEXT_SXV_COL: begin
                h_sxv_col_idx <= h_last_sxv_col ?
                                 '0 : h_sxv_col_idx + 1;
                h_compute_cnt <= '0; h_wb_cnt <= '0;
            end
            S_H_NEXT_SXV_ROWGRP: begin
                h_sxv_rowgrp  <= h_last_sxv_rowgrp ?
                                 '0 : h_sxv_rowgrp + 1;
                h_sxv_col_idx <= '0;
            end
            S_H_NEXT_SXV_HD: begin
                h_head_idx   <= h_last_head ?
                                '0 : h_head_idx + 1;
                h_sxv_rowgrp <= '0; h_sxv_col_idx <= '0;
            end
            S_H_PROJ_NEXT_COL: begin
                h_proj_col_idx <= h_last_proj_col ?
                                  '0 : h_proj_col_idx + 1;
                h_compute_cnt  <= '0; h_wb_cnt <= '0;
            end
            S_H_PROJ_NEXT_ROWGRP: begin
                h_proj_rowgrp  <= h_last_proj_rowgrp ?
                                  '0 : h_proj_rowgrp + 1;
                h_proj_col_idx <= '0;
            end
            S_H_FFN1_NEXT_COL: begin
                h_ffn1_col_idx <= h_last_ffn1_col ?
                                  '0 : h_ffn1_col_idx + 1;
                h_compute_cnt  <= '0; h_wb_cnt <= '0;
            end
            S_H_FFN1_NEXT_ROWGRP: begin
                h_ffn1_rowgrp  <= h_last_ffn1_rowgrp ?
                                  '0 : h_ffn1_rowgrp + 1;
                h_ffn1_col_idx <= '0;
            end
            S_H_FFN2_NEXT_COL: begin
                h_ffn2_col_idx <= h_last_ffn2_col ?
                                  '0 : h_ffn2_col_idx + 1;
                h_compute_cnt  <= '0; h_wb_cnt <= '0;
            end
            S_H_FFN2_NEXT_ROWGRP: begin
                h_ffn2_rowgrp  <= h_last_ffn2_rowgrp ?
                                  '0 : h_ffn2_rowgrp + 1;
                h_ffn2_col_idx <= '0;
            end
            S_H_NEXT_WINDOW: begin
                h_win_idx       <= h_last_win ?
                                   h_win_idx : h_win_idx + 1;
                h_qkv_mat       <= '0; h_qkv_col_idx   <= '0;
                h_patch_grp     <= '0; h_head_idx       <= '0;
                h_attn_col_idx  <= '0;
                h_attn_rowgrp    <= '0;
                h_sxv_col_idx   <= '0; h_sxv_rowgrp     <= '0;
                h_proj_col_idx  <= '0; h_proj_rowgrp     <= '0;
                h_ffn1_col_idx  <= '0;
                h_ffn1_rowgrp    <= '0;
                h_ffn2_col_idx  <= '0; h_ffn2_rowgrp    <= '0;
                h_compute_cnt   <= '0; h_wb_cnt          <= '0;
                h_mask_elem_cnt <= '0;
                // defensive reset for new window
            end

            default: ;
        endcase
    end
end

// =============================================================================
// done
// =============================================================================
assign done = (state == S_DONE);

// =============================================================================
// omem_fb_en_ctrl
// Asserted for all MHA states that read from the ILB rather than the FIB.
// CHANGED (rev 4): S_H_MASK_RD added — the mask R-M-W reads score elements
// from ILB, so the feedback path must be selected.
// =============================================================================
always_comb begin
    omem_fb_en_ctrl = 1'b0;
    case (state)
        S_H_LOAD_QH, S_H_LOAD_WKT, S_H_LOAD_SH, S_H_LOAD_VH,
        S_H_PROJ_LOAD_INPUT, S_H_FFN1_LOAD_INPUT, S_H_FFN2_LOAD_INPUT,
        S_H_MASK_RD:                                              // NEW
            omem_fb_en_ctrl = 1'b1;
        default: ;
    endcase
end

// =============================================================================
// Mask buffer control outputs (NEW)
// =============================================================================

// qkt_store_done: fire when the last row-group of a head's 49×49 QK^T is
// done, i.e. in S_H_NEXT_ATTN_ROWGRP while h_last_attn_rowgrp is true.
// This is the same cycle the FSM decides to go to S_H_MASK_RD, so the
// mask_buffer has its rd_ptr reset to 0 and mask_valid rises on the very
// next clock — exactly when S_H_MASK_RD is entered.
always_comb begin
    qkt_store_done = 1'b0;
    if (state == S_H_NEXT_ATTN_ROWGRP && h_last_attn_rowgrp)
        qkt_store_done = 1'b1;
end

// mask_next_window: pulse for one cycle in S_H_MASK_NEXT_WIN to advance
// mask_buffer.rd_ptr. mask_buffer registers this on the rising edge and
// presents the new mask_data_out / mask_window_idx combinationally.
always_comb begin
    mask_next_window = 1'b0;
    if (state == S_H_MASK_NEXT_WIN)
        mask_next_window = 1'b1;
end

// =============================================================================
// MMU sub-cycle (unchanged)
// =============================================================================
always_comb begin
    case (state)
        S_M_L1_COMPUTE:
            mmu_sub_cycle = (m_compute_cnt < M_N_ACC_L1) ?
                            3'(m_compute_cnt) : 3'(M_N_ACC_L1 - 1);
        S_M_L2_COMPUTE:
            mmu_sub_cycle = (m_compute_cnt < M_N_ACC_L2) ?
                            3'(m_compute_cnt) : 3'(M_N_ACC_L2 - 1);
        S_H_COMPUTE_QKV, S_H_PROJ_COMPUTE, S_H_FFN1_COMPUTE:
            mmu_sub_cycle = 3'(h_compute_cnt);
        S_H_COMPUTE_ATTN:
            mmu_sub_cycle = 3'd0;
        S_H_COMPUTE_SXV:
            mmu_sub_cycle = 3'(h_compute_cnt);
        S_H_FFN2_COMPUTE:
            mmu_sub_cycle = 3'(h_compute_cnt);
        default:
            mmu_sub_cycle = 3'd0;
    endcase
end

// =============================================================================
// Weight memory — active bank read (unchanged)
// =============================================================================
always_comb begin
    wmem_rd_addr = '0;
    wmem_rd_en   = 1'b0;
    case (state)
        S_C_LOAD_W: begin
            wmem_rd_addr = c_wmem_rd_addr;
            wmem_rd_en   = !is_data_ph;
        end
        S_M_L1_LOAD_W: begin
            wmem_rd_addr = m_w1_rd_addr;
            wmem_rd_en   = !is_data_ph;
        end
        S_M_L2_LOAD_W: begin
            wmem_rd_addr = m_w2_rd_addr;
            wmem_rd_en   = !is_data_ph;
        end
        S_H_LOAD_WQKV: begin
            wmem_rd_addr = h_wqkv_rd_addr;
            wmem_rd_en   = !is_data_ph;
        end
        S_H_PROJ_LOAD_W: begin
            wmem_rd_addr = h_wproj_rd_addr;
            wmem_rd_en   = !is_data_ph;
        end
        S_H_FFN1_LOAD_W: begin
            wmem_rd_addr = h_wffn1_rd_addr;
            wmem_rd_en   = !is_data_ph;
        end
        S_H_FFN2_LOAD_W: begin
            wmem_rd_addr = h_wffn2_rd_addr;
            wmem_rd_en   = !is_data_ph;
        end
        default: ;
    endcase
end

// =============================================================================
// Weight memory — shadow bank / external (unchanged)
// =============================================================================
always_comb begin
    ext_weight_rd_addr  = '0;
    ext_weight_rd_en    = 1'b0;
    wmem_shadow_wr_addr = '0;
    wmem_shadow_wr_en   = 1'b0;
    wmem_swap           = 1'b0;

    case (state)
        S_C_INIT_PRELOAD: begin
            ext_weight_rd_en    = !is_data_ph;
            ext_weight_rd_addr  = WAW'(load_cnt);
            wmem_shadow_wr_en   = is_data_ph;
            wmem_shadow_wr_addr = WAW'(load_cnt);
        end
        S_C_INIT_WMEM_SWAP: wmem_swap = 1'b1;
        S_C_LOAD_W: begin
            if (!c_last_kernel) begin
                ext_weight_rd_en    = !is_data_ph;
                ext_weight_rd_addr  = c_shadow_next_addr;
                wmem_shadow_wr_en   = is_data_ph;
                wmem_shadow_wr_addr = c_shadow_next_addr;
            end
        end
        S_C_SWAP_W: wmem_swap = 1'b1;
        S_M_L1_PRELOAD0: begin
            ext_weight_rd_en    = !is_data_ph;
            ext_weight_rd_addr  = WAW'(load_cnt);
            wmem_shadow_wr_en   = is_data_ph;
            wmem_shadow_wr_addr = WAW'(load_cnt);
        end
        S_M_L1_WMEM_SWAP0: wmem_swap = 1'b1;
        S_M_L1_LOAD_W: begin
            if (!m_last_l1_col) begin
                ext_weight_rd_en    = !is_data_ph;
                ext_weight_rd_addr  = m_shadow_w1_next_addr;
                wmem_shadow_wr_en   = is_data_ph;
                wmem_shadow_wr_addr = m_shadow_w1_next_addr;
            end
        end
        S_M_L1_SWAP_W: wmem_swap = 1'b1;
        S_M_L2_PRELOAD0: begin
            ext_weight_rd_en    = !is_data_ph;
            ext_weight_rd_addr  = WAW'(W2_BASE) + WAW'(load_cnt);
            wmem_shadow_wr_en   = is_data_ph;
            wmem_shadow_wr_addr = WAW'(W2_BASE) + WAW'(load_cnt);
        end
        S_M_L2_WMEM_SWAP0: wmem_swap = 1'b1;
        S_M_L2_LOAD_W: begin
            if (!m_last_l2_col) begin
                ext_weight_rd_en    = !is_data_ph;
                ext_weight_rd_addr  = m_shadow_w2_next_addr;
                wmem_shadow_wr_en   = is_data_ph;
                wmem_shadow_wr_addr = m_shadow_w2_next_addr;
            end
        end
        S_M_L2_SWAP_W: wmem_swap = 1'b1;
        default: ;
    endcase
end

// =============================================================================
// Input / feature memory (imem_rd) — unchanged except mask states add nothing
// =============================================================================
always_comb begin
    imem_rd_addr = '0;
    imem_rd_en   = 1'b0;
    case (state)
        S_C_LOAD_IMG: begin
            imem_rd_addr = c_imem_addr;
            imem_rd_en   = !is_data_ph;
        end
        S_M_L1_LOAD_X: begin
            imem_rd_addr = m_xmem_addr;
            imem_rd_en   = !is_data_ph;
        end
        S_H_LOAD_INPUT: begin
            imem_rd_addr = OAW'(h_fib_patch_addr);
            imem_rd_en   = !is_data_ph;
        end
        S_H_LOAD_QH: begin
            imem_rd_addr = h_qh_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        S_H_LOAD_SH: begin
            imem_rd_addr = h_sh_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        S_H_LOAD_WKT: begin
            imem_rd_addr = h_kt_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        S_H_LOAD_VH: begin
            imem_rd_addr = h_vh_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        S_H_PROJ_LOAD_INPUT: begin
            imem_rd_addr = h_proj_in_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        S_H_FFN1_LOAD_INPUT: begin
            imem_rd_addr = h_ffn1_in_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        S_H_FFN2_LOAD_INPUT: begin
            imem_rd_addr = h_ffn2_in_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        // ── Mask RD: drive the ILB score element address on imem_rd ───────
        // omem_fb_en_ctrl=1 causes full_system_top to route this to
        // output_memory.fb_rd_addr. The read result (fb_rd_data) arrives
        // one cycle later in S_H_MASK_WB via the feedback path.
        S_H_MASK_RD: begin
            imem_rd_addr = h_mask_rd_addr;
            imem_rd_en   = 1'b1;
        end
        default: ;
    endcase
end

// =============================================================================
// Output memory write
// CHANGED (rev 4): S_H_MASK_WB added.
// In full_system_top, ilb_wr_bypass=1 for all MHA states (mode==2'b10), so
// output_memory will store ilb_raw_wr_data = fb_rd_data + mask_data_out
// (the adder lives in full_system_top, not here).
// =============================================================================
always_comb begin
    omem_wr_addr = '0;
    omem_wr_en   = 1'b0;

    case (state)
        S_C_WRITEBACK: begin
            omem_wr_addr = c_omem_addr;
            omem_wr_en   = 1'b1;
        end
        S_M_WRITEBACK: begin
            omem_wr_addr = m_omem_addr;
            omem_wr_en   = 1'b1;
        end
        S_H_WRITEBACK_QKV: begin
            omem_wr_addr = h_qkv_omem_addr;
            omem_wr_en   = 1'b1;
        end
        S_H_WRITEBACK_ATTN: begin
            omem_wr_addr = h_attn_omem_addr;
            omem_wr_en   = 1'b1;
        end
        S_H_WRITEBACK_SXV: begin
            omem_wr_addr = h_sxv_omem_addr;
            omem_wr_en   = 1'b1;
        end
        S_H_PROJ_WRITEBACK: begin
            omem_wr_addr = h_proj_omem_addr;
            omem_wr_en   = 1'b1;
        end
        S_H_FFN1_WRITEBACK: begin
            omem_wr_addr = h_ffn1_omem_addr;
            omem_wr_en   = 1'b1;
        end
        S_H_FFN2_WRITEBACK: begin
            omem_wr_addr = h_ffn2_omem_addr;
            omem_wr_en   = 1'b1;
        end
        // NEW: write masked score element back to the same ILB location.
        // Address is the registered capture of the read address issued one
        // cycle earlier in S_H_MASK_RD.
        S_H_MASK_WB: begin
            omem_wr_addr = h_mask_wb_addr;
            omem_wr_en   = 1'b1;
        end
        default: ;
    endcase
end

// =============================================================================
// Weight buffer control (unchanged)
// =============================================================================
always_comb begin
    wbuf_load_en        = 1'b0;
    wbuf_load_pe_idx    = '0;
    wbuf_load_k_word    = '0;
    wbuf_load_data      = '0;
    wbuf_bias_load_en   = 1'b0;
    wbuf_bias_load_data = '0;
    wbuf_swap           = 1'b0;

    case (state)
        S_C_LOAD_W: begin
            wbuf_load_en        = is_data_ph && (load_cnt < 12);
            wbuf_load_pe_idx    = 4'(load_cnt);
            wbuf_load_data      = wmem_rd_data;
            wbuf_bias_load_en   = is_data_ph && (load_cnt == 12);
            wbuf_bias_load_data = wmem_rd_data;
        end
        S_C_SWAP_W: wbuf_swap = 1'b1;

        S_M_L1_LOAD_W: begin
            wbuf_load_en     = is_data_ph;
            wbuf_load_k_word = 7'(load_cnt);
            wbuf_load_data   = wmem_rd_data;
        end
        S_M_L1_SWAP_W: wbuf_swap = 1'b1;

        S_M_L2_LOAD_W: begin
            wbuf_load_en     = is_data_ph;
            wbuf_load_k_word = 7'(load_cnt);
            wbuf_load_data   = wmem_rd_data;
        end
        S_M_L2_SWAP_W: wbuf_swap = 1'b1;

        S_H_LOAD_WQKV, S_H_PROJ_LOAD_W, S_H_FFN1_LOAD_W, S_H_FFN2_LOAD_W: begin
            wbuf_load_en     = is_data_ph;
            wbuf_load_k_word = 7'(load_cnt);
            wbuf_load_data   = wmem_rd_data;
        end
        S_H_SWAP_WQKV, S_H_PROJ_SWAP_W,
        S_H_FFN1_SWAP_W, S_H_FFN2_SWAP_W: wbuf_swap = 1'b1;

        S_H_LOAD_WKT, S_H_LOAD_VH: begin
            wbuf_load_en     = is_data_ph;
            wbuf_load_k_word = 7'(load_cnt);
            wbuf_load_data   = imem_rd_data;
        end
        S_H_SWAP_WKT, S_H_SWAP_VH: wbuf_swap = 1'b1;

        default: ;
    endcase
end

// =============================================================================
// Input buffer control (unchanged)
// =============================================================================
always_comb begin
    ibuf_load_en          = 1'b0;
    ibuf_load_pe_idx      = '0;
    ibuf_load_win_idx     = '0;
    ibuf_load_row         = '0;
    ibuf_load_k_word      = '0;
    ibuf_load_data        = '0;
    ibuf_swap             = 1'b0;

    ibuf_l1_capture_en    = 1'b0;
    ibuf_l1_col_wr        = '0;

    ibuf_mha_load_en      = 1'b0;
    ibuf_mha_load_patch   = '0;
    ibuf_mha_load_k_word  = '0;
    ibuf_mha_load_data    = '0;
    ibuf_mha_capture_row  = '0;

    case (state)
        S_C_LOAD_IMG: begin
            ibuf_load_en      = is_data_ph;
            ibuf_load_pe_idx  = c_img_pe;
            ibuf_load_win_idx = c_img_win;
            ibuf_load_data    = imem_rd_data;
        end
        S_C_SWAP_IMG: ibuf_swap = 1'b1;

        S_M_L1_LOAD_X: begin
            ibuf_load_en     = is_data_ph;
            ibuf_load_row    = 3'(m_x_sub_row[2:0]);
            ibuf_load_k_word = {2'b00, m_x_k_word};
            ibuf_load_data   = imem_rd_data;
        end
        S_M_L1_SWAP_X: ibuf_swap = 1'b1;

        S_M_L1_COMPUTE: begin
            ibuf_l1_capture_en = (m_compute_cnt == M_N_ACC_L1);
            ibuf_l1_col_wr     = m_l1_col_idx;
        end
        S_M_L1_NEXT_COL:
            ibuf_swap = m_last_l1_col;

        S_H_LOAD_INPUT, S_H_LOAD_QH, S_H_LOAD_SH,
        S_H_PROJ_LOAD_INPUT, S_H_FFN1_LOAD_INPUT, S_H_FFN2_LOAD_INPUT: begin
            if (is_data_ph) begin
                ibuf_mha_load_en     = 1'b1;
                ibuf_mha_load_patch  = 6'(int'(h_patch_grp) * 7
                                        + int'(load_cnt) / (H_C_IN/4));
                ibuf_mha_load_k_word = 5'(int'(load_cnt) % (H_C_IN/4));
                ibuf_mha_load_data   = imem_rd_data;
            end
        end
        S_H_SWAP_INPUT, S_H_SWAP_QH, S_H_SWAP_SH,
        S_H_PROJ_SWAP_INPUT, S_H_FFN1_SWAP_INPUT, S_H_FFN2_SWAP_INPUT:
            ibuf_swap = 1'b1;

        S_H_WRITEBACK_QKV: begin
            ibuf_l1_capture_en   = 1'b1;
            ibuf_l1_col_wr       = 9'(h_qkv_col_idx);
            ibuf_mha_capture_row = 6'(int'(h_patch_grp) * 7);
        end
        default: ;
    endcase
end

// =============================================================================
// MMU control (unchanged)
// =============================================================================
always_comb begin
    mmu_valid_in = 1'b0;
    mmu_op_code  = 3'd0;
    mmu_stage    = 2'd0;

    case (state)
        S_C_COMPUTE, S_C_WAIT_OUT: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd0;
            mmu_stage    = 2'd0;
        end
        S_M_L1_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd1;
            mmu_stage    = 2'd0;
        end
        S_M_L2_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd1;
            mmu_stage    = 2'd2;
        end
        S_H_COMPUTE_QKV: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd2;
            mmu_stage    = 2'd0;
        end
        S_H_COMPUTE_ATTN: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd3;
            mmu_stage    = 2'd0;
        end
        S_H_COMPUTE_SXV: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd4;
            mmu_stage    = 2'd0;
        end
        S_H_PROJ_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd2;
            mmu_stage    = 2'd1;
        end
        S_H_FFN1_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd5;
            mmu_stage    = 2'd0;
        end
        S_H_FFN2_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd5;
            mmu_stage    = 2'd1;
        end
        default: ;
    endcase
end

// =============================================================================
// Output buffer control (unchanged)
// =============================================================================
always_comb begin
    obuf_capture_en = 1'b0;
    obuf_rd_idx     = '0;
    case (state)
        S_C_WAIT_OUT:       obuf_capture_en = 1'b1;
        S_C_WRITEBACK:      obuf_rd_idx = c_wb_cnt;

        S_M_L2_COMPUTE:     obuf_capture_en = (m_compute_cnt == M_N_ACC_L2);
        S_M_WRITEBACK:      obuf_rd_idx = m_wb_cnt;

        S_H_COMPUTE_QKV:    obuf_capture_en = (h_compute_cnt == H_N_ACC_QKV);
        S_H_WRITEBACK_QKV:  obuf_rd_idx = h_wb_cnt;

        S_H_COMPUTE_ATTN:   obuf_capture_en = (h_compute_cnt == H_N_ACC_ATTN);
        S_H_WRITEBACK_ATTN: obuf_rd_idx = h_wb_cnt;

        S_H_COMPUTE_SXV:    obuf_capture_en = (h_compute_cnt == H_N_ACC_SXV);
        S_H_WRITEBACK_SXV:  obuf_rd_idx = h_wb_cnt;

        S_H_PROJ_COMPUTE:   obuf_capture_en = (h_compute_cnt == H_N_ACC_PROJ);
        S_H_PROJ_WRITEBACK: obuf_rd_idx = h_wb_cnt;

        S_H_FFN1_COMPUTE:   obuf_capture_en = (h_compute_cnt == H_N_ACC_FFN1);
        S_H_FFN1_WRITEBACK: obuf_rd_idx = h_wb_cnt;

        S_H_FFN2_COMPUTE:   obuf_capture_en = (h_compute_cnt == H_N_ACC_FFN2);
        S_H_FFN2_WRITEBACK: obuf_rd_idx = h_wb_cnt;

        default: ;
    endcase
end

// =============================================================================
// Shift buffer control (unchanged)
// =============================================================================
always_comb begin : shift_buf_ctrl
    sb_op_start     = 1'b0;
    sb_op_base_addr = '0;
    sb_advance      = 1'b0;

    if (state == S_IDLE && start) begin
        sb_op_start = 1'b1;
        case (mode)
            2'b00:   sb_op_base_addr = SB_AW'(SB_CONV_BASE);
            2'b01:   sb_op_base_addr = SB_AW'(SB_MLP_BASE);
            2'b10:   sb_op_base_addr = SB_AW'(SB_MHA_BASE);
            default: sb_op_base_addr = '0;
        endcase
    end else if (state == S_M_L1_NEXT_COL && m_last_l1_col) begin
        sb_op_start     = 1'b1;
        sb_op_base_addr = SB_AW'(SB_MLP_BASE);
    end else if (state == S_H_NEXT_WINDOW && !h_last_win) begin
        sb_op_start     = 1'b1;
        sb_op_base_addr = SB_AW'(SB_MHA_BASE);
    end

    if (state == S_C_NEXT && c_last_chunk)    sb_advance = 1'b1;
    if (state == S_M_NEXT_ROW)                sb_advance = 1'b1;
    if (state == S_H_NEXT_QKV_COL)            sb_advance = 1'b1;
    if (state == S_H_NEXT_ATTN_COL)           sb_advance = 1'b1;
    if (state == S_H_NEXT_SXV_COL)            sb_advance = 1'b1;
    if (state == S_H_PROJ_NEXT_COL)           sb_advance = 1'b1;
    if (state == S_H_FFN1_NEXT_COL)           sb_advance = 1'b1;
    if (state == S_H_FFN2_NEXT_COL)           sb_advance = 1'b1;

end : shift_buf_ctrl

// =============================================================================
// Bias buffer control
// =============================================================================
always_comb begin : bias_buf_ctrl

    bb_op_start     = 1'b0;
    bb_op_base_addr = BB_AW'(BB_CONV_BASE);
    bb_advance      = 1'b0;

    // ── bb_op_start / bb_op_base_addr ─────────────────────────────────
    
    // Conv — arm at global start
    if (state == S_IDLE && start && mode == 2'b00) begin
        bb_op_start     = 1'b1;
        bb_op_base_addr = BB_AW'(BB_CONV_BASE);
    end

    // MLP L1 — arm at global start
    if (state == S_IDLE && start && mode == 2'b01) begin
        bb_op_start     = 1'b1;
        bb_op_base_addr = BB_AW'(BB_MLP_L1_BASE);
    end

    // MLP L2 — re-arm at the L1→L2 boundary
    if (state == S_M_L1_NEXT_COL && m_last_l1_col) begin
        bb_op_start     = 1'b1;
        bb_op_base_addr = BB_AW'(BB_MLP_L2_BASE);
    end

    // MHA — arm at global start; re-arm at each new 7×7 window.
    // bb_op_base_addr always points to BB_MHA_QKT_BASE: the full
    // 2401-entry QK^T bias table is replayed identically for every window
    // (position-relative attention bias is window-invariant).
    if (state == S_IDLE && start && mode == 2'b10) begin
        bb_op_start     = 1'b1;
        bb_op_base_addr = BB_AW'(BB_MHA_QKT_BASE);
    end
    if (state == S_H_NEXT_WINDOW && !h_last_win) begin
        bb_op_start     = 1'b1;
        bb_op_base_addr = BB_AW'(BB_MHA_QKT_BASE);
    end

    // ── bb_advance ────────────────────────────────────────────────────
    
    // Conv: one pulse per completed 56×56 output plane (one per kernel).
    if (state == S_C_NEXT && c_last_chunk && c_last_row_group)
        bb_advance = 1'b1;

    // MLP L1 and L2: one pulse per 7-column output group.
    if (state == S_M_NEXT_ROW)
        bb_advance = 1'b1;

    // MHA QK^T: one pulse per 7-element column group of the 49×49 matrix.
    if (state == S_H_NEXT_ATTN_COL)
        bb_advance = 1'b1;

end : bias_buf_ctrl

endmodule