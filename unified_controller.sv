// =============================================================================
// unified_controller.sv  (rev 3 — MHA / Swin Transformer Block support + Shift Buffer)
//
// ── What changed from rev 2 ───────────────────────────────────────────────
//   Added mode 2'b10 : Swin Transformer Block (MHA).
//   The MHA round executes entirely on-chip in a single pass before the MWU
//   returns results to off-chip memory.  This matches the paper description:
//
//     "The accelerator is capable of executing the entire computation of
//      the Swin Transformer Block in a single round, which includes
//      SWMSA/WMSA, Shortcut Mechanism, and FFN."
//
//   Within one MHA round (per 7×7 window, 64 windows total):
//     Step 1:  Load input patch (49×96) from FIB → ibuf, load W_Q from wmem
//              Compute Q = X × W_Q  (49×96)  → store in ILB (output_memory)
//     Step 2:  Reload input patch from FIB, load W_K
//              Compute K = X × W_K  (49×96)  → store in ILB
//     Step 3:  Reload input patch from FIB, load W_V
//              Compute V = X × W_V  (49×96)  → store in ILB
//     Step 4:  For each head h in 0..2:
//                Load Q_h (49×32) from ILB → ibuf
//                Load K_h^T (32×49) from ILB → wbuf  (K transposed on read)
//                Compute S_h = Q_h × K_h^T  (49×49)  → store in ILB
//     Step 5:  For each head h in 0..2:
//                Load S_h (49×49) from ILB → ibuf
//                Load V_h (49×32) from ILB → wbuf
//                Compute A_h = S_h × V_h  (49×32)   → store in ILB
//     Step 6:  Concatenate A_0..A_2 → 49×96, apply W_proj linear transform
//              Output of this step + Shortcut from FIB → ILB
//     Step 7:  FFN Layer 1: 49×96 → 49×384, GELU activation, stored in ILB
//     Step 8:  FFN Layer 2: 49×384 → 49×96, + Shortcut → MWU → off-chip
//
//   The controller issues mode-2 IDs for the MMU so it knows which op to run.
//   The `mode` input is now 2 bits (was 1 bit).
//
// ── Weight memory layout (MHA additions) ─────────────────────────────────
//   Existing:
//     [0     ..  9215] : Conv kernels  (96 kernels × 13 words = 1248 words,
//                          but WAW=15 → 32768 entries, so plenty of space)
//     [W2_BASE .. ...]  : MLP W2
//   New MHA weight offsets (all configurable as parameters):
//     WQ_BASE  : W_Q  96×96  = 9216  words (at 4 bytes/word = 36864 B each)
//     WK_BASE  : W_K  96×96  = 9216  words
//     WV_BASE  : W_V  96×96  = 9216  words
//     WPROJ_BASE: W_proj 96×96 = 9216 words
//     WFFN1_BASE: W_FFN1 96×384= 36864 words (96×384=36864, /4 words = 9216)
//     WFFN2_BASE: W_FFN2 384×96= 36864 words
//   Total MHA weight words: 6 × 9216 = 55,296 words → needs WAW ≥ 16 (65536)
//   → WAW bumped to 16 in parameter declaration (was 15).
//
// ── FIB / ILB memory layout (MHA) ─────────────────────────────────────────
//   FIB: holds full 56×56×96 feature map (75,264 words) — unchanged.
//   ILB (output_memory): intermediate results written/read during MHA round.
//     Q/K/V per head:  3 heads × 49×32 = 4,704 bytes each set
//     QKᵀ attention:   3 heads × 49×49 = 7,203 bytes
//     SxV (MHA out):   3 heads × 49×32 = 4,704 bytes
//     After W_proj:    49×96 = 4,704 bytes
//     FFN intermediates (GELU): 49×384 = 18,816 bytes
//   Peak ILB demand ≈ 50 KB — fits comfortably in OAW=19 (512K word space).
//
// ── State encoding (48 states, 6-bit) ────────────────────────────────────
//   [0..27]   : unchanged from rev 2 (Conv and MLP)
//   [28..47]  : new MHA states
//
//   MHA states:
//   S_H_LOAD_INPUT   = 28   load 49-patch window from FIB into ibuf
//   S_H_SWAP_INPUT   = 29
//   S_H_LOAD_WQ      = 30   load W_Q column from wmem into wbuf
//   S_H_SWAP_WQ      = 31
//   S_H_COMPUTE_Q    = 32   compute Q column (49×1 partial, 8 sub-cycles)
//   S_H_NEXT_QKV_COL = 33   advance to next output column (0..95)
//   S_H_NEXT_QKV_MAT = 34   advance matrix (Q done → K, K done → V)
//   S_H_LOAD_WKT     = 35   load K^T row (as weight column) for attention
//   S_H_SWAP_WKT     = 36
//   S_H_LOAD_QH      = 37   load Q head from ILB into ibuf
//   S_H_SWAP_QH      = 38
//   S_H_COMPUTE_ATTN = 39   compute S = QKᵀ column
//   S_H_NEXT_ATTN_COL= 40
//   S_H_NEXT_ATTN_HD = 41   advance head (0..2)
//   S_H_LOAD_SH      = 42   load S head from ILB into ibuf for SxV
//   S_H_SWAP_SH      = 43
//   S_H_LOAD_VH      = 44   load V head column into wbuf
//   S_H_SWAP_VH      = 45
//   S_H_COMPUTE_SXV  = 46   compute A = SxV column
//   S_H_NEXT_SXV_COL = 47
//   S_H_NEXT_SXV_HD  = 48
//   S_H_PROJ_LOAD_W  = 49   load W_proj column
//   S_H_PROJ_SWAP_W  = 50
//   S_H_PROJ_COMPUTE = 51   compute A_concat × W_proj column
//   S_H_PROJ_NEXT_COL= 52
//   S_H_SHORTCUT1    = 53   add shortcut (ILB + FIB) → ILB  [ctrl signal only]
//   S_H_FFN1_LOAD_W  = 54   load W_FFN1 column (96→384 expansion)
//   S_H_FFN1_SWAP_W  = 55
//   S_H_FFN1_COMPUTE = 56
//   S_H_FFN1_NEXT_COL= 57
//   S_H_GELU_WAIT    = 58   GCU pipeline drain (GELU on 49×384)
//   S_H_FFN2_LOAD_W  = 59   load W_FFN2 column (384→96 contraction)
//   S_H_FFN2_SWAP_W  = 60
//   S_H_LOAD_FFN1_OUT= 61   reload FFN1 output from ILB into ibuf
//   S_H_SWAP_FFN1_OUT= 62
//   S_H_FFN2_COMPUTE = 63
//   S_H_FFN2_NEXT_COL= 64
//   S_H_SHORTCUT2    = 65   add shortcut (FFN2 out + attn out)
//   S_H_WRITEBACK    = 66   write FFN2 result via MWU to off-chip
//   S_H_NEXT_WINDOW  = 67   advance to next of 64 windows
// =============================================================================

module unified_controller #(
    parameter int WAW      = 16,    // weight_memory address width (was 15, bumped for MHA)
    parameter int FAW      = 17,    // fib_memory address width
    parameter int OAW      = 19,    // output_memory address width
    parameter int W2_BASE  = 9216,  // MLP W2 block offset in weight_memory

    // MHA weight offsets (in 32-bit words)
    parameter int WQ_BASE   = 10240,   // 96×96   = 9216 words  (after MLP weights)
    parameter int WK_BASE   = 19456,   // WQ_BASE + 9216
    parameter int WV_BASE   = 28672,   // WK_BASE + 9216
    parameter int WPROJ_BASE= 37888,   // WV_BASE + 9216
    parameter int WFFN1_BASE= 47104,   // WPROJ_BASE + 9216  (96×384 packed into 9216 words)
    parameter int WFFN2_BASE= 56320,   // WFFN1_BASE + 9216  (384×96 packed into 9216 words)

    // MHA ILB base addresses (in output_memory word space)
    // Layout: Q[0..3071], K[3072..6143], V[6144..9215],
    //         S[9216..16467 = 3×49×49], A[16468..19539 = 3×49×32],
    //         PROJ[19540..20587 = 49×96/4], FFN1[20588..25291 = 49×384/4]
    parameter int ILB_Q_BASE    = 0,
    parameter int ILB_K_BASE    = 3072,   // 3×49×32/4
    parameter int ILB_V_BASE    = 6144,
    parameter int ILB_S_BASE    = 9216,   // attention scores 3×49×49
    parameter int ILB_A_BASE    = 16468,  // SxV output 3×49×32
    parameter int ILB_PROJ_BASE = 19540,  // W_proj output 49×96
    parameter int ILB_FFN1_BASE = 20588,  // FFN1 output 49×384

    // ── Shift buffer parameters ──────────────────────────────────────────────
    // These must match the DEPTH and layout in shift_buffer.sv / full_system_top.sv.
    // Entry counts per operation:
    //   Conv : 96 kernels × 56 output rows            = 5 376 entries  [0 .. 5375]
    //   MLP  : 448 row-groups (3136 rows / 7)         =   448 entries  [5376 .. 5823]
    //          (same 448-entry table reused by L1 and L2 within one MLP pass)
    //   MHA  : all sub-ops per 7×7 window              = 7 749 entries  [5824 .. 13572]
    //          Layout within MHA window (sequential):
    //            QKV   : 96 cols × 7 groups × 3 mats  = 2 016
    //            ATTN  : 49 cols × 7 groups × 3 heads = 1 029
    //            SxV   : 32 cols × 7 groups × 3 heads =   672
    //            PROJ  : 96 cols × 7 groups           =   672
    //            FFN1  : 384 cols × 7 groups          = 2 688
    //            FFN2  : 96 cols × 7 groups           =   672
    parameter int SB_AW         = 14,    // must match shift_buffer AW = log2(DEPTH)
    parameter int SB_CONV_BASE  = 0,     // entry base for Conv shift table
    parameter int SB_MLP_BASE   = 5376,  // entry base for MLP shift table
    parameter int SB_MHA_BASE   = 5824   // entry base for MHA shift table (per window)
)(
    input  logic clk,
    input  logic rst_n,

    // ── Mode and start/done ───────────────────────────────────────────────
    input  logic [1:0] mode,   // 2'b00=Conv, 2'b01=MLP, 2'b10=MHA
    input  logic start,
    output logic done,

    // ── Weight memory — active bank read  ────────────────────────────────
    output logic [WAW-1:0] wmem_rd_addr,
    output logic           wmem_rd_en,
    input  logic [31:0]    wmem_rd_data,

    // ── Weight memory — shadow bank write ────────────────────────────────
    output logic [WAW-1:0] wmem_shadow_wr_addr,
    output logic           wmem_shadow_wr_en,

    // ── Weight memory — bank swap ─────────────────────────────────────────
    output logic           wmem_swap,

    // ── External big memory (weight source, 1-cycle latency) ─────────────
    output logic [WAW-1:0] ext_weight_rd_addr,
    output logic           ext_weight_rd_en,

    // ── Input / feature memory (fib or omem.fb, muxed in top) ────────────
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
    output logic [3:0]     ibuf_load_pe_idx,    // conv
    output logic [2:0]     ibuf_load_win_idx,   // conv
    output logic [2:0]     ibuf_load_row,       // mlp
    output logic [6:0]     ibuf_load_k_word,    // mlp
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
    // Tells full_system_top to route imem reads from ILB (output_memory)
    // instead of FIB during MHA intermediate steps.
    output logic           omem_fb_en_ctrl,

    // ── Shift buffer control ──────────────────────────────────────────────────
    // sb_op_start     : one-cycle pulse when a new operation begins (or each new
    //                   MHA window).  Carries the entry base address that the
    //                   shift_buffer will reset its read pointer to.
    // sb_op_base_addr : entry address of the first shift value for this op.
    // sb_advance      : one-cycle pulse after every completed 7-element row-group
    //                   writeback, causing the shift_buffer to step to the next
    //                   shift value for the following row-group.
    output logic               sb_op_start,
    output logic [SB_AW-1:0]   sb_op_base_addr,
    output logic               sb_advance
);

// =============================================================================
// Parameters — Conv and MLP (unchanged)
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

localparam int M_N_ROW_GRPS    = 448;
localparam int M_N_L1_COLS     = 384;
localparam int M_N_L2_COLS     = 96;
localparam int M_N_ACC_L1      = 2;
localparam int M_N_ACC_L2      = 8;
localparam int M_W1_WORDS      = 24;
localparam int M_W2_WORDS      = 96;
localparam int M_X_WORDS       = 168;
localparam int M_W1_CYCS       = M_W1_WORDS * 2;
localparam int M_W2_CYCS       = M_W2_WORDS * 2;
localparam int M_X_CYCS        = M_X_WORDS  * 2;
localparam int M_OUT_WORDS     = 7;

// =============================================================================
// Parameters — MHA
// =============================================================================
// Each 7×7 window: 49 patches × 96 features
localparam int H_N_WINDOWS     = 64;    // total windows in 56×56 image
localparam int H_N_PATCHES     = 49;    // patches per window
localparam int H_N_HEADS       = 3;
localparam int H_HEAD_DIM      = 32;    // features per head (96/3)
localparam int H_C_IN          = 96;    // input channels
localparam int H_C_FFN         = 384;   // FFN expansion (×4)

// Words to load a 96-column weight vector (12 PEs × 4 taps = 48 features
// → 96/48 = 2 sub-cycles, each word = 4 bytes → 96/4 = 24 words/column)
localparam int H_W_QKV_WORDS   = 24;    // words per column of W_Q/K/V (96 rows / 4)
localparam int H_W_QKV_CYCS    = H_W_QKV_WORDS * 2;   // 48

// QKᵀ: K^T has 32 rows (head_dim), 12 PEs × 4 taps = 48, so ≤ 1 sub-cycle
// K column (= row of K^T) is 49 elements; each word = 4 elements → 13 words
localparam int H_WKT_WORDS     = 13;    // ceil(49/4)
localparam int H_WKT_CYCS      = H_WKT_WORDS * 2;     // 26

// Input load for QKV step: 7 patches × 24 words = 168 words
localparam int H_I_PATCH7_WORDS = H_N_PATCHES * H_W_QKV_WORDS;  // 49*24 = 1176 (full window)
// We load 7 patches at a time (one ibuf bank row group = 7)
localparam int H_I_LOAD7_WORDS  = 7 * H_W_QKV_WORDS;            // 7*24 = 168
localparam int H_I_LOAD7_CYCS   = H_I_LOAD7_WORDS * 2;          // 336

// For QKᵀ step: Q head (49×32) loaded 7 rows at a time; 32/4 = 8 words/row
localparam int H_QH_LOAD_WORDS  = 7 * 8;                         // 56
localparam int H_QH_LOAD_CYCS   = H_QH_LOAD_WORDS * 2;          // 112

// Compute sub-cycles:
// QKV: 96 input features / (12 PE × 4 taps) = 2 sub-cycles per 7-row group
localparam int H_N_ACC_QKV      = 2;
// QKᵀ: 32 features / 48 = 1 sub-cycle (only 32/4=8 PEs active)
localparam int H_N_ACC_ATTN     = 1;
// SxV: 49 features / 48 → 2 sub-cycles (need 49 partial sums)
localparam int H_N_ACC_SXV      = 2;
// Proj: same as QKV (96→96)
localparam int H_N_ACC_PROJ     = 2;
// FFN1: 96→384, weight col is 96 rows → same as QKV
localparam int H_N_ACC_FFN1     = 2;
// FFN2: 384→96, weight col is 384 rows → 384/48 = 8 sub-cycles
localparam int H_N_ACC_FFN2     = 8;

localparam int H_OUT_WORDS      = 7;

// =============================================================================
// State encoding
// =============================================================================
typedef enum logic [6:0] {
    // ── Conv (unchanged) ───────────────────────────────────────────────────
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
    // ── MLP (unchanged) ───────────────────────────────────────────────────
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
    // ── MHA (new) ─────────────────────────────────────────────────────────
    // QKV projection sub-FSM
    S_H_LOAD_INPUT      = 7'd28,  // load 7 patches from FIB → ibuf
    S_H_SWAP_INPUT      = 7'd29,
    S_H_LOAD_WQKV       = 7'd30,  // load W_Q/K/V column from wmem → wbuf
    S_H_SWAP_WQKV       = 7'd31,
    S_H_COMPUTE_QKV     = 7'd32,  // compute one output column (7 rows)
    S_H_WRITEBACK_QKV   = 7'd33,  // write 7-row result to ILB
    S_H_NEXT_QKV_COL    = 7'd34,  // advance column (0..95)
    S_H_NEXT_PATCH_GRP  = 7'd35,  // advance 7-patch group (0..6 → 0..48)
    S_H_NEXT_QKV_MAT    = 7'd36,  // advance Q→K→V
    // Attention score QKᵀ sub-FSM
    S_H_LOAD_QH         = 7'd37,  // load Q head (7 rows × 8 words) from ILB
    S_H_SWAP_QH         = 7'd38,
    S_H_LOAD_WKT        = 7'd39,  // load K^T column (=K row) from ILB → wbuf
    S_H_SWAP_WKT        = 7'd40,
    S_H_COMPUTE_ATTN    = 7'd41,
    S_H_WRITEBACK_ATTN  = 7'd42,
    S_H_NEXT_ATTN_COL   = 7'd43,
    S_H_NEXT_ATTN_ROWGRP= 7'd44,
    S_H_NEXT_ATTN_HD    = 7'd45,
    // SxV sub-FSM
    S_H_LOAD_SH         = 7'd46,  // load S head row-group from ILB
    S_H_SWAP_SH         = 7'd47,
    S_H_LOAD_VH         = 7'd48,  // load V head column from ILB → wbuf
    S_H_SWAP_VH         = 7'd49,
    S_H_COMPUTE_SXV     = 7'd50,
    S_H_WRITEBACK_SXV   = 7'd51,
    S_H_NEXT_SXV_COL    = 7'd52,
    S_H_NEXT_SXV_ROWGRP = 7'd53,
    S_H_NEXT_SXV_HD     = 7'd54,
    // W_proj linear transform
    S_H_PROJ_LOAD_INPUT = 7'd55,  // load concat A from ILB into ibuf
    S_H_PROJ_SWAP_INPUT = 7'd56,
    S_H_PROJ_LOAD_W     = 7'd57,
    S_H_PROJ_SWAP_W     = 7'd58,
    S_H_PROJ_COMPUTE    = 7'd59,
    S_H_PROJ_WRITEBACK  = 7'd60,
    S_H_PROJ_NEXT_COL   = 7'd61,
    S_H_PROJ_NEXT_ROWGRP= 7'd62,
    S_H_SHORTCUT1       = 7'd63,  // shortcut: proj_out + X_input
    // FFN Layer 1 (96→384)
    S_H_FFN1_LOAD_INPUT = 7'd64,
    S_H_FFN1_SWAP_INPUT = 7'd65,
    S_H_FFN1_LOAD_W     = 7'd66,
    S_H_FFN1_SWAP_W     = 7'd67,
    S_H_FFN1_COMPUTE    = 7'd68,
    S_H_FFN1_WRITEBACK  = 7'd69,
    S_H_FFN1_NEXT_COL   = 7'd70,
    S_H_FFN1_NEXT_ROWGRP= 7'd71,
    S_H_GELU_WAIT       = 7'd72,
    // FFN Layer 2 (384→96)
    S_H_FFN2_LOAD_INPUT = 7'd73,
    S_H_FFN2_SWAP_INPUT = 7'd74,
    S_H_FFN2_LOAD_W     = 7'd75,
    S_H_FFN2_SWAP_W     = 7'd76,
    S_H_FFN2_COMPUTE    = 7'd77,
    S_H_FFN2_WRITEBACK  = 7'd78,
    S_H_FFN2_NEXT_COL   = 7'd79,
    S_H_FFN2_NEXT_ROWGRP= 7'd80,
    S_H_SHORTCUT2       = 7'd81,  // shortcut: FFN2_out + shortcut1_out
    S_H_NEXT_WINDOW     = 7'd82   // advance window index (0..63)
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
logic [5:0]  h_win_idx;          // 0..63 windows
logic [1:0]  h_qkv_mat;          // 0=Q, 1=K, 2=V
logic [6:0]  h_qkv_col_idx;      // 0..95 output columns
logic [2:0]  h_patch_grp;        // 0..6  (7 groups × 7 patches = 49)
logic [1:0]  h_head_idx;         // 0..2
logic [5:0]  h_attn_col_idx;     // 0..48 (QKᵀ columns = 49)
logic [2:0]  h_attn_rowgrp;      // 0..6
logic [4:0]  h_sxv_col_idx;      // 0..31 (SxV output columns per head)
logic [2:0]  h_sxv_rowgrp;       // 0..6
logic [6:0]  h_proj_col_idx;     // 0..95
logic [2:0]  h_proj_rowgrp;
logic [8:0]  h_ffn1_col_idx;     // 0..383
logic [2:0]  h_ffn1_rowgrp;
logic [6:0]  h_ffn2_col_idx;     // 0..95
logic [2:0]  h_ffn2_rowgrp;
logic [2:0]  h_compute_cnt;
logic [2:0]  h_wb_cnt;

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
assign c_last_chunk     = (c_chunk_idx      == C_N_CHUNKS      - 1);
assign c_last_row_group = (c_row_group_idx == C_N_ROW_GROUPS - 1);
assign c_last_kernel    = (c_kernel_idx    == C_N_KERNELS    - 1);

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

assign h_last_win        = (h_win_idx        == H_N_WINDOWS  - 1);
assign h_last_qkv_col    = (h_qkv_col_idx    == H_C_IN       - 1);  // 95
assign h_last_patch_grp  = (h_patch_grp      == (H_N_PATCHES / 7)); // 6  (7 groups)
assign h_last_head       = (h_head_idx       == H_N_HEADS    - 1);  // 2
assign h_last_attn_col   = (h_attn_col_idx  == H_N_PATCHES  - 1);  // 48
assign h_last_attn_rowgrp= (h_attn_rowgrp   == (H_N_PATCHES / 7)); // 6
assign h_last_sxv_col    = (h_sxv_col_idx    == H_HEAD_DIM   - 1);  // 31
assign h_last_sxv_rowgrp = (h_sxv_rowgrp     == (H_N_PATCHES / 7));
assign h_last_proj_col   = (h_proj_col_idx  == H_C_IN       - 1);
assign h_last_proj_rowgrp= (h_proj_rowgrp   == (H_N_PATCHES / 7));
assign h_last_ffn1_col   = (h_ffn1_col_idx  == H_C_FFN      - 1);  // 383
assign h_last_ffn1_rowgrp= (h_ffn1_rowgrp   == (H_N_PATCHES / 7));
assign h_last_ffn2_col   = (h_ffn2_col_idx  == H_C_IN       - 1);
assign h_last_ffn2_rowgrp= (h_ffn2_rowgrp   == (H_N_PATCHES / 7));

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
// Address generation — MHA
// =============================================================================

// ── QKV weight addresses ──────────────────────────────────────────────────
// Each of W_Q, W_K, W_V is 96×96, stored column-major.
// Column h_qkv_col_idx has 96/4=24 words.
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

// ── FIB input load address for QKV (window patches) ──────────────────────
// Window h_win_idx starts at patch h_win_idx*49 in the FIB.
// Group h_patch_grp: patches [h_patch_grp*7 .. h_patch_grp*7+6]
// Word index load_cnt = patch_in_grp * 24 + k_word
logic [FAW-1:0] h_fib_patch_addr;
always_comb begin
    automatic int global_patch = int'(h_win_idx) * H_N_PATCHES
                                + int'(h_patch_grp) * 7
                                + int'(load_cnt) / 24;
    automatic int k_word       = int'(load_cnt) % 24;
    h_fib_patch_addr = FAW'(global_patch * 24 + k_word);
end

// ── ILB addresses for Q/K/V results ──────────────────────────────────────
// Q stored at ILB_Q_BASE, K at ILB_K_BASE, V at ILB_V_BASE.
// Layout: [mat_base + col * N_PATCHES_WORDS + patch_grp * 7 + wb_cnt]
// N_PATCHES_WORDS = 49 (one word per patch, one byte per patch per column)
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

// ── Attention score addresses (QKᵀ) ──────────────────────────────────────
// Read Q_head from ILB at ILB_Q_BASE + head*49*32 + col*49 + patch_grp*7
logic [OAW-1:0] h_qh_rd_addr;
assign h_qh_rd_addr = OAW'(ILB_Q_BASE)
                     + OAW'(h_head_idx) * (H_N_PATCHES * H_HEAD_DIM)
                     + OAW'(h_attn_col_idx) // reading the k-index column of Q_h as rows
                     + OAW'(h_attn_rowgrp) * 7
                     + OAW'(load_cnt);

// Read K^T: K stored as 49×32 at ILB_K_BASE + head*49*32
// K^T column j = K row j → read 32 elements starting at row j
logic [OAW-1:0] h_kt_rd_addr;
assign h_kt_rd_addr = OAW'(ILB_K_BASE)
                     + OAW'(h_head_idx) * (H_N_PATCHES * H_HEAD_DIM)
                     + OAW'(h_attn_col_idx) * H_HEAD_DIM  // col of K^T = row of K
                     + OAW'(load_cnt);

// Attention output address (ILB_S_BASE)
logic [OAW-1:0] h_attn_omem_addr;
assign h_attn_omem_addr = OAW'(ILB_S_BASE)
                         + OAW'(h_head_idx) * (H_N_PATCHES * H_N_PATCHES)
                         + OAW'(h_attn_col_idx) * H_N_PATCHES
                         + OAW'(h_attn_rowgrp) * 7
                         + OAW'(h_wb_cnt);

// ── SxV addresses ─────────────────────────────────────────────────────────
// S head from ILB_S_BASE, V head from ILB_V_BASE
logic [OAW-1:0] h_sh_rd_addr;
assign h_sh_rd_addr = OAW'(ILB_S_BASE)
                     + OAW'(h_head_idx) * (H_N_PATCHES * H_N_PATCHES)
                     + OAW'(h_sxv_col_idx) // iterating over V columns
                     + OAW'(h_sxv_rowgrp) * 7
                     + OAW'(load_cnt);

logic [OAW-1:0] h_vh_rd_addr;
assign h_vh_rd_addr = OAW'(ILB_V_BASE)
                     + OAW'(h_head_idx) * (H_N_PATCHES * H_HEAD_DIM)
                     + OAW'(h_sxv_col_idx) * H_N_PATCHES
                     + OAW'(load_cnt);

logic [OAW-1:0] h_sxv_omem_addr;
assign h_sxv_omem_addr = OAW'(ILB_A_BASE)
                        + OAW'(h_head_idx) * (H_N_PATCHES * H_HEAD_DIM)
                        + OAW'(h_sxv_col_idx) * H_N_PATCHES
                        + OAW'(h_sxv_rowgrp) * 7
                        + OAW'(h_wb_cnt);

// ── W_proj, FFN1, FFN2 addresses ─────────────────────────────────────────
logic [WAW-1:0] h_wproj_rd_addr;
assign h_wproj_rd_addr = WAW'(WPROJ_BASE)
                        + WAW'(h_proj_col_idx) * H_W_QKV_WORDS
                        + WAW'(load_cnt);

logic [WAW-1:0] h_wffn1_rd_addr;
assign h_wffn1_rd_addr = WAW'(WFFN1_BASE)
                        + WAW'(h_ffn1_col_idx) * H_W_QKV_WORDS  // 96 rows → 24 words/col
                        + WAW'(load_cnt);

// FFN2: 384 rows → 96 words/col
logic [WAW-1:0] h_wffn2_rd_addr;
assign h_wffn2_rd_addr = WAW'(WFFN2_BASE)
                        + WAW'(h_ffn2_col_idx) * (H_C_FFN / 4)  // 96 words/col
                        + WAW'(load_cnt);

// Proj input from ILB_A_BASE (concatenated 3×49×32 = 49×96)
logic [OAW-1:0] h_proj_in_rd_addr;
assign h_proj_in_rd_addr = OAW'(ILB_A_BASE)
                          + OAW'(h_proj_rowgrp) * 7 * H_W_QKV_WORDS
                          + OAW'(load_cnt);

logic [OAW-1:0] h_proj_omem_addr;
assign h_proj_omem_addr = OAW'(ILB_PROJ_BASE)
                         + OAW'(h_proj_col_idx) * H_N_PATCHES
                         + OAW'(h_proj_rowgrp) * 7
                         + OAW'(h_wb_cnt);

// FFN1 input: proj output (after shortcut) at ILB_PROJ_BASE
logic [OAW-1:0] h_ffn1_in_rd_addr;
assign h_ffn1_in_rd_addr = OAW'(ILB_PROJ_BASE)
                           + OAW'(h_ffn1_rowgrp) * 7 * H_W_QKV_WORDS
                           + OAW'(load_cnt);

logic [OAW-1:0] h_ffn1_omem_addr;
assign h_ffn1_omem_addr = OAW'(ILB_FFN1_BASE)
                         + OAW'(h_ffn1_col_idx) * H_N_PATCHES
                         + OAW'(h_ffn1_rowgrp) * 7
                         + OAW'(h_wb_cnt);

// FFN2 input: FFN1 GELU output at ILB_FFN1_BASE
logic [OAW-1:0] h_ffn2_in_rd_addr;
assign h_ffn2_in_rd_addr = OAW'(ILB_FFN1_BASE)
                           + OAW'(h_ffn2_rowgrp) * 7 * (H_C_FFN / 4)  // 96 words/row
                           + OAW'(load_cnt);

logic [OAW-1:0] h_ffn2_omem_addr;
assign h_ffn2_omem_addr = OAW'(ILB_PROJ_BASE)  // overwrite proj slot (reuse for final output)
                         + OAW'(h_ffn2_col_idx) * H_N_PATCHES
                         + OAW'(h_ffn2_rowgrp) * 7
                         + OAW'(h_wb_cnt);

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
                    2'b00: next_state = S_C_INIT_PRELOAD;
                    2'b01: next_state = S_M_L1_PRELOAD0;
                    2'b10: next_state = S_H_LOAD_INPUT;
                    default: next_state = S_IDLE;
                endcase
        end

        // ── Conv (unchanged) ──────────────────────────────────────────────
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

        // ── MLP (unchanged) ───────────────────────────────────────────────
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

        // ── MHA ───────────────────────────────────────────────────────────
        // QKV projection
        S_H_LOAD_INPUT:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_SWAP_INPUT;
        S_H_SWAP_INPUT:   next_state = S_H_LOAD_WQKV;
        S_H_LOAD_WQKV:
            if (load_cyc == H_W_QKV_CYCS - 1) next_state = S_H_SWAP_WQKV;
        S_H_SWAP_WQKV:    next_state = S_H_COMPUTE_QKV;
        S_H_COMPUTE_QKV:
            if (h_compute_cnt == H_N_ACC_QKV)  next_state = S_H_WRITEBACK_QKV;
        S_H_WRITEBACK_QKV:
            if (h_wb_cnt == H_OUT_WORDS - 1)   next_state = S_H_NEXT_QKV_COL;
        S_H_NEXT_QKV_COL: begin
            if (h_last_qkv_col) next_state = S_H_NEXT_PATCH_GRP;
            else                next_state = S_H_LOAD_WQKV;   // same patch group, new col
        end
        S_H_NEXT_PATCH_GRP: begin
            if (h_last_patch_grp) next_state = S_H_NEXT_QKV_MAT;
            else                  next_state = S_H_LOAD_INPUT;  // reload next 7 patches
        end
        S_H_NEXT_QKV_MAT: begin
            if (h_qkv_mat == 2'd2) next_state = S_H_LOAD_QH;  // all Q,K,V done → attention
            else                   next_state = S_H_LOAD_INPUT; // reload patches for K or V
        end

        // Attention QKᵀ
        S_H_LOAD_QH:
            if (load_cyc == H_QH_LOAD_CYCS - 1) next_state = S_H_SWAP_QH;
        S_H_SWAP_QH:      next_state = S_H_LOAD_WKT;
        S_H_LOAD_WKT:
            if (load_cyc == H_WKT_CYCS - 1)   next_state = S_H_SWAP_WKT;
        S_H_SWAP_WKT:     next_state = S_H_COMPUTE_ATTN;
        S_H_COMPUTE_ATTN:
            if (h_compute_cnt == H_N_ACC_ATTN) next_state = S_H_WRITEBACK_ATTN;
        S_H_WRITEBACK_ATTN:
            if (h_wb_cnt == H_OUT_WORDS - 1)   next_state = S_H_NEXT_ATTN_COL;
        S_H_NEXT_ATTN_COL: begin
            if (h_last_attn_col) next_state = S_H_NEXT_ATTN_ROWGRP;
            else                 next_state = S_H_LOAD_WKT;
        end
        S_H_NEXT_ATTN_ROWGRP: begin
            if (h_last_attn_rowgrp) next_state = S_H_NEXT_ATTN_HD;
            else                    next_state = S_H_LOAD_QH;
        end
        S_H_NEXT_ATTN_HD: begin
            if (h_last_head) next_state = S_H_LOAD_SH;  // all heads done → SxV
            else             next_state = S_H_LOAD_QH;
        end

        // SxV
        S_H_LOAD_SH:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_SWAP_SH;
        S_H_SWAP_SH:      next_state = S_H_LOAD_VH;
        S_H_LOAD_VH:
            if (load_cyc == H_WKT_CYCS - 1)   next_state = S_H_SWAP_VH;
        S_H_SWAP_VH:      next_state = S_H_COMPUTE_SXV;
        S_H_COMPUTE_SXV:
            if (h_compute_cnt == H_N_ACC_SXV)  next_state = S_H_WRITEBACK_SXV;
        S_H_WRITEBACK_SXV:
            if (h_wb_cnt == H_OUT_WORDS - 1)   next_state = S_H_NEXT_SXV_COL;
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

        // W_proj
        S_H_PROJ_LOAD_INPUT:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_PROJ_SWAP_INPUT;
        S_H_PROJ_SWAP_INPUT: next_state = S_H_PROJ_LOAD_W;
        S_H_PROJ_LOAD_W:
            if (load_cyc == H_W_QKV_CYCS - 1) next_state = S_H_PROJ_SWAP_W;
        S_H_PROJ_SWAP_W:     next_state = S_H_PROJ_COMPUTE;
        S_H_PROJ_COMPUTE:
            if (h_compute_cnt == H_N_ACC_PROJ) next_state = S_H_PROJ_WRITEBACK;
        S_H_PROJ_WRITEBACK:
            if (h_wb_cnt == H_OUT_WORDS - 1)   next_state = S_H_PROJ_NEXT_COL;
        S_H_PROJ_NEXT_COL: begin
            if (h_last_proj_col) next_state = S_H_PROJ_NEXT_ROWGRP;
            else                 next_state = S_H_PROJ_LOAD_W;
        end
        S_H_PROJ_NEXT_ROWGRP: begin
            if (h_last_proj_rowgrp) next_state = S_H_SHORTCUT1;
            else                    next_state = S_H_PROJ_LOAD_INPUT;
        end
        S_H_SHORTCUT1: next_state = S_H_FFN1_LOAD_INPUT;  // 1-cycle shortcut add

        // FFN1
        S_H_FFN1_LOAD_INPUT:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_FFN1_SWAP_INPUT;
        S_H_FFN1_SWAP_INPUT: next_state = S_H_FFN1_LOAD_W;
        S_H_FFN1_LOAD_W:
            if (load_cyc == H_W_QKV_CYCS - 1) next_state = S_H_FFN1_SWAP_W;
        S_H_FFN1_SWAP_W:     next_state = S_H_FFN1_COMPUTE;
        S_H_FFN1_COMPUTE:
            if (h_compute_cnt == H_N_ACC_FFN1) next_state = S_H_FFN1_WRITEBACK;
        S_H_FFN1_WRITEBACK:
            if (h_wb_cnt == H_OUT_WORDS - 1)   next_state = S_H_FFN1_NEXT_COL;
        S_H_FFN1_NEXT_COL: begin
            if (h_last_ffn1_col) next_state = S_H_FFN1_NEXT_ROWGRP;
            else                 next_state = S_H_FFN1_LOAD_W;
        end
        S_H_FFN1_NEXT_ROWGRP: begin
            if (h_last_ffn1_rowgrp) next_state = S_H_GELU_WAIT;
            else                    next_state = S_H_FFN1_LOAD_INPUT;
        end
        S_H_GELU_WAIT: next_state = S_H_FFN2_LOAD_INPUT;  // GCU pipeline drain

        // FFN2
        S_H_FFN2_LOAD_INPUT:
            if (load_cyc == H_I_LOAD7_CYCS - 1) next_state = S_H_FFN2_SWAP_INPUT;
        S_H_FFN2_SWAP_INPUT: next_state = S_H_FFN2_LOAD_W;
        S_H_FFN2_LOAD_W:
            if (load_cyc == (H_C_FFN/4)*2 - 1) next_state = S_H_FFN2_SWAP_W;
        S_H_FFN2_SWAP_W:     next_state = S_H_FFN2_COMPUTE;
        S_H_FFN2_COMPUTE:
            if (h_compute_cnt == H_N_ACC_FFN2) next_state = S_H_FFN2_WRITEBACK;
        S_H_FFN2_WRITEBACK:
            if (h_wb_cnt == H_OUT_WORDS - 1)   next_state = S_H_FFN2_NEXT_COL;
        S_H_FFN2_NEXT_COL: begin
            if (h_last_ffn2_col) next_state = S_H_FFN2_NEXT_ROWGRP;
            else                 next_state = S_H_FFN2_LOAD_W;
        end
        S_H_FFN2_NEXT_ROWGRP: begin
            if (h_last_ffn2_rowgrp) next_state = S_H_SHORTCUT2;
            else                    next_state = S_H_FFN2_LOAD_INPUT;
        end
        S_H_SHORTCUT2: next_state = S_H_NEXT_WINDOW;
        S_H_NEXT_WINDOW: begin
            if (h_last_win) next_state = S_DONE;
            else            next_state = S_H_LOAD_INPUT;  // next window → Q step
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
        c_kernel_idx    <= '0; c_row_group_idx  <= '0;
        c_chunk_idx     <= '0; c_wb_cnt         <= '0;
        m_row_grp_idx   <= '0; m_l1_col_idx     <= '0;
        m_l2_col_idx    <= '0; m_compute_cnt    <= '0;
        m_wb_cnt        <= '0; load_cyc         <= '0;
        h_win_idx       <= '0; h_qkv_mat        <= '0;
        h_qkv_col_idx   <= '0; h_patch_grp      <= '0;
        h_head_idx      <= '0; h_attn_col_idx   <= '0;
        h_attn_rowgrp   <= '0; h_sxv_col_idx    <= '0;
        h_sxv_rowgrp    <= '0; h_proj_col_idx   <= '0;
        h_proj_rowgrp   <= '0; h_ffn1_col_idx   <= '0;
        h_ffn1_rowgrp   <= '0; h_ffn2_col_idx   <= '0;
        h_ffn2_rowgrp   <= '0; h_compute_cnt    <= '0;
        h_wb_cnt        <= '0;
    end else begin
        case (state)

            // ── load_cyc resets ────────────────────────────────────────────
            S_IDLE,
            S_C_INIT_WMEM_SWAP, S_C_SWAP_W,   S_C_SWAP_IMG,  S_C_NEXT,
            S_M_L1_WMEM_SWAP0,  S_M_L1_SWAP_X, S_M_L1_SWAP_W, S_M_L1_NEXT_COL,
            S_M_L2_WMEM_SWAP0,  S_M_L2_SWAP_W, S_M_L2_NEXT_COL, S_M_NEXT_ROW,
            S_H_SWAP_INPUT, S_H_SWAP_WQKV, S_H_SWAP_QH, S_H_SWAP_WKT,
            S_H_SWAP_SH, S_H_SWAP_VH, S_H_PROJ_SWAP_INPUT, S_H_PROJ_SWAP_W,
            S_H_FFN1_SWAP_INPUT, S_H_FFN1_SWAP_W,
            S_H_FFN2_SWAP_INPUT, S_H_FFN2_SWAP_W:
                load_cyc <= '0;

            // ── Conv ───────────────────────────────────────────────────────
            S_C_INIT_PRELOAD, S_C_LOAD_W:
                load_cyc <= (load_cyc < C_WLOAD_CYCS - 1) ? load_cyc + 1 : '0;
            S_C_LOAD_IMG:
                load_cyc <= (load_cyc < C_ILOAD_CYCS - 1) ? load_cyc + 1 : '0;
            S_C_WRITEBACK:
                c_wb_cnt <= (c_wb_cnt < C_OUT_WORDS - 1) ? c_wb_cnt + 1 : '0;
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

            // ── MLP ────────────────────────────────────────────────────────
            S_M_L1_PRELOAD0:
                load_cyc <= (load_cyc < M_W1_CYCS - 1) ? load_cyc + 1 : '0;
            S_M_L1_LOAD_X:
                load_cyc <= (load_cyc < M_X_CYCS  - 1) ? load_cyc + 1 : '0;
            S_M_L1_LOAD_W:
                load_cyc <= (load_cyc < M_W1_CYCS - 1) ? load_cyc + 1 : '0;
            S_M_L1_COMPUTE:
                m_compute_cnt <= (m_compute_cnt < M_N_ACC_L1) ?
                                  m_compute_cnt + 1 : '0;
            S_M_L1_NEXT_COL: begin
                m_l1_col_idx  <= m_last_l1_col ? '0 : m_l1_col_idx + 1;
                m_compute_cnt <= '0;
            end
            S_M_L2_PRELOAD0:
                load_cyc <= (load_cyc < M_W2_CYCS - 1) ? load_cyc + 1 : '0;
            S_M_L2_LOAD_W:
                load_cyc <= (load_cyc < M_W2_CYCS - 1) ? load_cyc + 1 : '0;
            S_M_L2_COMPUTE:
                m_compute_cnt <= (m_compute_cnt < M_N_ACC_L2) ?
                                  m_compute_cnt + 1 : '0;
            S_M_WRITEBACK:
                m_wb_cnt <= (m_wb_cnt < M_OUT_WORDS - 1) ? m_wb_cnt + 1 : '0;
            S_M_L2_NEXT_COL: begin
                m_l2_col_idx  <= m_last_l2_col ? '0 : m_l2_col_idx + 1;
                m_compute_cnt <= '0;
            end
            S_M_NEXT_ROW: begin
                m_row_grp_idx <= m_last_row_grp ? m_row_grp_idx : m_row_grp_idx + 1;
                m_l1_col_idx  <= '0; m_l2_col_idx <= '0; m_compute_cnt <= '0;
            end

            // ── MHA ────────────────────────────────────────────────────────
            S_H_LOAD_INPUT, S_H_LOAD_SH, S_H_PROJ_LOAD_INPUT,
            S_H_FFN1_LOAD_INPUT, S_H_FFN2_LOAD_INPUT:
                load_cyc <= (load_cyc < H_I_LOAD7_CYCS - 1) ? load_cyc + 1 : '0;

            S_H_LOAD_WQKV, S_H_PROJ_LOAD_W, S_H_FFN1_LOAD_W:
                load_cyc <= (load_cyc < H_W_QKV_CYCS - 1) ? load_cyc + 1 : '0;

            S_H_LOAD_QH:
                load_cyc <= (load_cyc < H_QH_LOAD_CYCS - 1) ? load_cyc + 1 : '0;

            S_H_LOAD_WKT, S_H_LOAD_VH:
                load_cyc <= (load_cyc < H_WKT_CYCS - 1) ? load_cyc + 1 : '0;

            S_H_FFN2_LOAD_W:
                load_cyc <= (load_cyc < (H_C_FFN/4)*2 - 1) ? load_cyc + 1 : '0;

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
                h_wb_cnt <= (h_wb_cnt < H_OUT_WORDS - 1) ? h_wb_cnt + 1 : '0;

            // Column / group advance
            S_H_NEXT_QKV_COL: begin
                if (h_last_qkv_col) h_qkv_col_idx <= '0;
                else                h_qkv_col_idx <= h_qkv_col_idx + 1;
                h_compute_cnt <= '0; h_wb_cnt <= '0;
            end
            S_H_NEXT_PATCH_GRP: begin
                if (h_last_patch_grp) h_patch_grp <= '0;
                else                  h_patch_grp <= h_patch_grp + 1;
            end
            S_H_NEXT_QKV_MAT: begin
                h_qkv_mat    <= (h_qkv_mat == 2'd2) ? '0 : h_qkv_mat + 1;
                h_patch_grp  <= '0;
                h_qkv_col_idx<= '0;
            end
            S_H_NEXT_ATTN_COL: begin
                if (h_last_attn_col) h_attn_col_idx <= '0;
                else                 h_attn_col_idx <= h_attn_col_idx + 1;
                h_compute_cnt <= '0; h_wb_cnt <= '0;
            end
            S_H_NEXT_ATTN_ROWGRP: begin
                if (h_last_attn_rowgrp) h_attn_rowgrp <= '0;
                else                    h_attn_rowgrp <= h_attn_rowgrp + 1;
                h_attn_col_idx <= '0;
            end
            S_H_NEXT_ATTN_HD: begin
                if (h_last_head) h_head_idx <= '0;
                else             h_head_idx <= h_head_idx + 1;
                h_attn_rowgrp <= '0; h_attn_col_idx <= '0;
            end
            S_H_NEXT_SXV_COL: begin
                if (h_last_sxv_col) h_sxv_col_idx <= '0;
                else                h_sxv_col_idx <= h_sxv_col_idx + 1;
                h_compute_cnt <= '0; h_wb_cnt <= '0;
            end
            S_H_NEXT_SXV_ROWGRP: begin
                if (h_last_sxv_rowgrp) h_sxv_rowgrp <= '0;
                else                   h_sxv_rowgrp <= h_sxv_rowgrp + 1;
                h_sxv_col_idx <= '0;
            end
            S_H_NEXT_SXV_HD: begin
                if (h_last_head) h_head_idx <= '0;
                else             h_head_idx <= h_head_idx + 1;
                h_sxv_rowgrp <= '0; h_sxv_col_idx <= '0;
            end
            S_H_PROJ_NEXT_COL: begin
                if (h_last_proj_col) h_proj_col_idx <= '0;
                else                 h_proj_col_idx <= h_proj_col_idx + 1;
                h_compute_cnt <= '0; h_wb_cnt <= '0;
            end
            S_H_PROJ_NEXT_ROWGRP: begin
                if (h_last_proj_rowgrp) h_proj_rowgrp <= '0;
                else                    h_proj_rowgrp <= h_proj_rowgrp + 1;
                h_proj_col_idx <= '0;
            end
            S_H_FFN1_NEXT_COL: begin
                if (h_last_ffn1_col) h_ffn1_col_idx <= '0;
                else                 h_ffn1_col_idx <= h_ffn1_col_idx + 1;
                h_compute_cnt <= '0; h_wb_cnt <= '0;
            end
            S_H_FFN1_NEXT_ROWGRP: begin
                if (h_last_ffn1_rowgrp) h_ffn1_rowgrp <= '0;
                else                    h_ffn1_rowgrp <= h_ffn1_rowgrp + 1;
                h_ffn1_col_idx <= '0;
            end
            S_H_FFN2_NEXT_COL: begin
                if (h_last_ffn2_col) h_ffn2_col_idx <= '0;
                else                 h_ffn2_col_idx <= h_ffn2_col_idx + 1;
                h_compute_cnt <= '0; h_wb_cnt <= '0;
            end
            S_H_FFN2_NEXT_ROWGRP: begin
                if (h_last_ffn2_rowgrp) h_ffn2_rowgrp <= '0;
                else                    h_ffn2_rowgrp <= h_ffn2_rowgrp + 1;
                h_ffn2_col_idx <= '0;
            end
            S_H_NEXT_WINDOW: begin
                h_win_idx     <= h_last_win ? h_win_idx : h_win_idx + 1;
                h_qkv_mat     <= '0; h_qkv_col_idx  <= '0;
                h_patch_grp   <= '0; h_head_idx      <= '0;
                h_attn_col_idx<= '0; h_attn_rowgrp   <= '0;
                h_sxv_col_idx <= '0; h_sxv_rowgrp    <= '0;
                h_proj_col_idx<= '0; h_proj_rowgrp   <= '0;
                h_ffn1_col_idx<= '0; h_ffn1_rowgrp   <= '0;
                h_ffn2_col_idx<= '0; h_ffn2_rowgrp   <= '0;
                h_compute_cnt <= '0; h_wb_cnt         <= '0;
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
// Asserted for all MHA states that read from the ILB (output_memory)
// rather than the FIB.
// =============================================================================
always_comb begin
    omem_fb_en_ctrl = 1'b0;
    case (state)
        // QKV: read patches from FIB → keep 0
        // Attention: read Q/K from ILB
        S_H_LOAD_QH, S_H_LOAD_WKT, S_H_LOAD_SH, S_H_LOAD_VH,
        S_H_PROJ_LOAD_INPUT, S_H_FFN1_LOAD_INPUT, S_H_FFN2_LOAD_INPUT:
            omem_fb_en_ctrl = 1'b1;
        default: ;
    endcase
end

// =============================================================================
// MMU sub-cycle
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
// Weight memory — active bank read
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
// Weight memory — shadow bank write + external big memory read (Conv/MLP only)
// MHA loads weights directly via the active-bank read path above;
// K^T and V are loaded from ILB via imem (feedback path), not wmem shadow.
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
        S_C_SWAP_W:         wmem_swap = 1'b1;

        S_M_L1_PRELOAD0: begin
            ext_weight_rd_en    = !is_data_ph;
            ext_weight_rd_addr  = WAW'(load_cnt);
            wmem_shadow_wr_en   = is_data_ph;
            wmem_shadow_wr_addr = WAW'(load_cnt);
        end
        S_M_L1_WMEM_SWAP0:  wmem_swap = 1'b1;
        S_M_L1_LOAD_W: begin
            if (!m_last_l1_col) begin
                ext_weight_rd_en    = !is_data_ph;
                ext_weight_rd_addr  = m_shadow_w1_next_addr;
                wmem_shadow_wr_en   = is_data_ph;
                wmem_shadow_wr_addr = m_shadow_w1_next_addr;
            end
        end
        S_M_L1_SWAP_W:      wmem_swap = 1'b1;
        S_M_L2_PRELOAD0: begin
            ext_weight_rd_en    = !is_data_ph;
            ext_weight_rd_addr  = WAW'(W2_BASE) + WAW'(load_cnt);
            wmem_shadow_wr_en   = is_data_ph;
            wmem_shadow_wr_addr = WAW'(W2_BASE) + WAW'(load_cnt);
        end
        S_M_L2_WMEM_SWAP0:  wmem_swap = 1'b1;
        S_M_L2_LOAD_W: begin
            if (!m_last_l2_col) begin
                ext_weight_rd_en    = !is_data_ph;
                ext_weight_rd_addr  = m_shadow_w2_next_addr;
                wmem_shadow_wr_en   = is_data_ph;
                wmem_shadow_wr_addr = m_shadow_w2_next_addr;
            end
        end
        S_M_L2_SWAP_W:      wmem_swap = 1'b1;

        // MHA weights come from a large external weight store; for simplicity
        // in this revision the MHA weight path uses direct active-bank reads
        // (no double-buffering for MHA weights — can be added as a future
        //  optimisation identical to the Conv/MLP pattern above).
        default: ;
    endcase
end

// =============================================================================
// Input / feature memory outputs
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
        // MHA: load 7 patches from FIB (omem_fb_en=0 in top during these states)
        S_H_LOAD_INPUT: begin
            imem_rd_addr = OAW'(h_fib_patch_addr);
            imem_rd_en   = !is_data_ph;
        end
        // MHA: read Q_h row-group from ILB (omem_fb_en=1 in top)
        S_H_LOAD_QH: begin
            imem_rd_addr = h_qh_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        // MHA: read S_h row-group from ILB
        S_H_LOAD_SH: begin
            imem_rd_addr = h_sh_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        // MHA: K^T column loaded as weights via imem (feedback)
        S_H_LOAD_WKT: begin
            imem_rd_addr = h_kt_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        // MHA: V_h column loaded via imem (feedback)
        S_H_LOAD_VH: begin
            imem_rd_addr = h_vh_rd_addr;
            imem_rd_en   = !is_data_ph;
        end
        // Proj, FFN1, FFN2 input patches from ILB
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
        default: ;
    endcase
end

// =============================================================================
// Output memory write outputs
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
        default: ;
    endcase
end

// =============================================================================
// Weight buffer control outputs
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

        // MHA weight loads (W_Q/K/V/proj): reuse MLP k_word path
        S_H_LOAD_WQKV, S_H_PROJ_LOAD_W, S_H_FFN1_LOAD_W, S_H_FFN2_LOAD_W: begin
            wbuf_load_en     = is_data_ph;
            wbuf_load_k_word = 7'(load_cnt);
            wbuf_load_data   = wmem_rd_data;
        end
        S_H_SWAP_WQKV, S_H_PROJ_SWAP_W,
        S_H_FFN1_SWAP_W, S_H_FFN2_SWAP_W: wbuf_swap = 1'b1;

        // MHA: K^T column / V column loaded from ILB via imem
        // → feed into wbuf using the k_word index (same as MLP)
        S_H_LOAD_WKT, S_H_LOAD_VH: begin
            wbuf_load_en     = is_data_ph;
            wbuf_load_k_word = 7'(load_cnt);
            wbuf_load_data   = imem_rd_data;  // comes from ILB feedback path
        end
        S_H_SWAP_WKT, S_H_SWAP_VH: wbuf_swap = 1'b1;

        default: ;
    endcase
end

// =============================================================================
// Input buffer control outputs
// =============================================================================
always_comb begin
    ibuf_load_en       = 1'b0;
    ibuf_load_pe_idx   = '0;
    ibuf_load_win_idx  = '0;
    ibuf_load_row      = '0;
    ibuf_load_k_word   = '0;
    ibuf_load_data     = '0;
    ibuf_swap          = 1'b0;
    ibuf_l1_capture_en = 1'b0;
    ibuf_l1_col_wr     = '0;
    // MHA-specific
    ibuf_mha_load_en      = 1'b0;
    ibuf_mha_load_patch   = '0;
    ibuf_mha_load_k_word  = '0;
    ibuf_mha_load_data    = '0;
    ibuf_mha_capture_row  = '0;

    case (state)
        // Conv
        S_C_LOAD_IMG: begin
            ibuf_load_en      = is_data_ph;
            ibuf_load_pe_idx  = c_img_pe;
            ibuf_load_win_idx = c_img_win;
            ibuf_load_data    = imem_rd_data;
        end
        S_C_SWAP_IMG: ibuf_swap = 1'b1;

        // MLP
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

        // MHA: load 7 patches into ibuf (MHA mode)
        S_H_LOAD_INPUT, S_H_LOAD_QH, S_H_LOAD_SH,
        S_H_PROJ_LOAD_INPUT, S_H_FFN1_LOAD_INPUT, S_H_FFN2_LOAD_INPUT: begin
            if (is_data_ph) begin
                ibuf_mha_load_en     = 1'b1;
                ibuf_mha_load_patch  = 6'(int'(h_patch_grp) * 7 + int'(load_cnt) / (H_C_IN/4));
                ibuf_mha_load_k_word = 5'(int'(load_cnt) % (H_C_IN/4));
                ibuf_mha_load_data   = imem_rd_data;
            end
        end
        S_H_SWAP_INPUT, S_H_SWAP_QH, S_H_SWAP_SH,
        S_H_PROJ_SWAP_INPUT, S_H_FFN1_SWAP_INPUT, S_H_FFN2_SWAP_INPUT:
            ibuf_swap = 1'b1;

        // MHA: capture QKV output rows back into ibuf shadow bank
        S_H_WRITEBACK_QKV: begin
            ibuf_l1_capture_en   = 1'b1;
            ibuf_l1_col_wr       = 9'(h_qkv_col_idx);
            ibuf_mha_capture_row = 6'(int'(h_patch_grp) * 7);
        end

        default: ;
    endcase
end

// =============================================================================
// MMU control outputs
// =============================================================================
always_comb begin
    mmu_valid_in = 1'b0;
    mmu_op_code  = 3'd0;
    mmu_stage    = 2'd0;

    case (state)
        S_C_COMPUTE, S_C_WAIT_OUT: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd0;   // Conv
            mmu_stage    = 2'd0;
        end
        S_M_L1_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd1;   // MLP L1
            mmu_stage    = 2'd0;
        end
        S_M_L2_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd1;   // MLP L2
            mmu_stage    = 2'd2;
        end
        // MHA sub-operations mapped to op_code 3'd2
        S_H_COMPUTE_QKV: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd2;   // MHA: linear projection
            mmu_stage    = 2'd0;
        end
        S_H_COMPUTE_ATTN: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd3;   // MHA: attention QKᵀ
            mmu_stage    = 2'd0;
        end
        S_H_COMPUTE_SXV: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd4;   // MHA: SxV
            mmu_stage    = 2'd0;
        end
        S_H_PROJ_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd2;   // MHA: W_proj linear
            mmu_stage    = 2'd1;
        end
        S_H_FFN1_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd5;   // FFN1 (pre-GELU)
            mmu_stage    = 2'd0;
        end
        S_H_FFN2_COMPUTE: begin
            mmu_valid_in = 1'b1;
            mmu_op_code  = 3'd5;   // FFN2
            mmu_stage    = 2'd1;
        end
        default: ;
    endcase
end

// =============================================================================
// Output buffer control outputs
// =============================================================================
always_comb begin
    obuf_capture_en = 1'b0;
    obuf_rd_idx     = '0;

    case (state)
        S_C_WAIT_OUT:        obuf_capture_en = 1'b1;
        S_C_WRITEBACK:       obuf_rd_idx = c_wb_cnt;
        S_M_L2_COMPUTE:      obuf_capture_en = (m_compute_cnt == M_N_ACC_L2);
        S_M_WRITEBACK:       obuf_rd_idx = m_wb_cnt;

        // MHA writeback states
        S_H_COMPUTE_QKV:     obuf_capture_en = (h_compute_cnt == H_N_ACC_QKV);
        S_H_WRITEBACK_QKV:   obuf_rd_idx = h_wb_cnt;

        S_H_COMPUTE_ATTN:    obuf_capture_en = (h_compute_cnt == H_N_ACC_ATTN);
        S_H_WRITEBACK_ATTN:  obuf_rd_idx = h_wb_cnt;

        S_H_COMPUTE_SXV:     obuf_capture_en = (h_compute_cnt == H_N_ACC_SXV);
        S_H_WRITEBACK_SXV:   obuf_rd_idx = h_wb_cnt;

        S_H_PROJ_COMPUTE:    obuf_capture_en = (h_compute_cnt == H_N_ACC_PROJ);
        S_H_PROJ_WRITEBACK:  obuf_rd_idx = h_wb_cnt;

        S_H_FFN1_COMPUTE:    obuf_capture_en = (h_compute_cnt == H_N_ACC_FFN1);
        S_H_FFN1_WRITEBACK:  obuf_rd_idx = h_wb_cnt;

        S_H_FFN2_COMPUTE:    obuf_capture_en = (h_compute_cnt == H_N_ACC_FFN2);
        S_H_FFN2_WRITEBACK:  obuf_rd_idx = h_wb_cnt;

        default: ;
    endcase
end

// =============================================================================
// Shift Buffer control
// =============================================================================
// Drives three new outputs that the shift_buffer module consumes:
//
//   sb_op_start / sb_op_base_addr
//     Asserted once per new top-level operation (Conv, MLP, MHA).
//     Also asserted when MLP transitions from L1 to L2 (resets the pointer
//     to the same MLP base so L2 reuses the same 448-entry shift table).
//     Also asserted at each new MHA window boundary (same 7 749-entry table
//     is reused for every window because weights — and therefore scaling
//     factors — are constant across all 64 windows).
//
//   sb_advance
//     Pulsed once per completed 7-element row-group for every operation.
//     Causes the shift_buffer to present the next shift value to the
//     rounding_shifter in time for the following row-group writeback.
//
//   Priority rule (enforced inside shift_buffer.sv):
//     sb_op_start beats sb_advance on any cycle where both are asserted.
//     The only case where this can happen is S_M_L1_NEXT_COL with
//     m_last_l1_col (last L1 column): sb_advance fires for the last L1
//     row-group while sb_op_start simultaneously resets for L2.
//     The op_start wins — the pointer is placed at SB_MLP_BASE and the
//     last L1 advance is discarded (irrelevant since L1 is done).
// =============================================================================
always_comb begin : shift_buf_ctrl
    sb_op_start     = 1'b0;
    sb_op_base_addr = '0;
    sb_advance      = 1'b0;

    // ── sb_op_start ──────────────────────────────────────────────────────────

    if (state == S_IDLE && start) begin
        // Arm the shift buffer at the very beginning of any operation.
        sb_op_start = 1'b1;
        case (mode)
            2'b00:   sb_op_base_addr = SB_AW'(SB_CONV_BASE);
            2'b01:   sb_op_base_addr = SB_AW'(SB_MLP_BASE);
            2'b10:   sb_op_base_addr = SB_AW'(SB_MHA_BASE);
            default: sb_op_base_addr = '0;
        endcase
    end

    // MLP Layer 2: reset read pointer back to MLP base so L2 reuses the
    // same 448 shift values that were used for L1.
    // Triggered at the last column of L1 (next state → S_M_L2_PRELOAD0).
    else if (state == S_M_L1_NEXT_COL && m_last_l1_col) begin
        sb_op_start     = 1'b1;
        sb_op_base_addr = SB_AW'(SB_MLP_BASE);
    end

    // MHA new window: each of the 64 windows uses the same weight matrices
    // and therefore the same shift values.  Reset the pointer at the start
    // of every new window so the 7 749-entry table is replayed identically.
    else if (state == S_H_NEXT_WINDOW && !h_last_win) begin
        sb_op_start     = 1'b1;
        sb_op_base_addr = SB_AW'(SB_MHA_BASE);
    end

    // ── sb_advance ───────────────────────────────────────────────────────────
    // One pulse per 7-element row-group completion for each operation.
    // The shift_buffer increments its read pointer so the next shift value
    // is already at the output before the following row-group's writeback.

    // ── Convolution ──────────────────────────────────────────────────────────
    // The same (kernel, row_group) shift value spans all 8 chunks
    // (c_chunk_idx 0..7).  Advance only when the last chunk finishes,
    // i.e. when c_last_chunk is true in S_C_NEXT.
    // Total advances: 96 kernels × 56 output rows = 5 376.
    if (state == S_C_NEXT && c_last_chunk)
        sb_advance = 1'b1;

    // ── MLP ──────────────────────────────────────────────────────────────────
    // One shift value covers all output columns for one 7-row row-group.
    // Advance once per row-group transition (S_M_NEXT_ROW).
    // Total advances per layer: 448.  The pointer is reset to SB_MLP_BASE
    // for L2 via the sb_op_start path above, so L2 replays the same 448
    // entries.
    if (state == S_M_NEXT_ROW)
        sb_advance = 1'b1;

    // ── MHA — QKV projection ─────────────────────────────────────────────────
    // Each (patch_group, output_column) pair produces one 7-element vector.
    // Advance at every S_H_NEXT_QKV_COL (fires 96 times per patch-group,
    // 7 patch-groups, 3 matrices Q/K/V → 2 016 advances per window).
    if (state == S_H_NEXT_QKV_COL)
        sb_advance = 1'b1;

    // ── MHA — Attention QKᵀ ──────────────────────────────────────────────────
    // Each (row_group, attn_col) pair → one 7-element vector.
    // Advances: 49 cols × 7 groups × 3 heads = 1 029.
    if (state == S_H_NEXT_ATTN_COL)
        sb_advance = 1'b1;

    // ── MHA — SxV ────────────────────────────────────────────────────────────
    // Each (row_group, sxv_col) pair → one 7-element vector.
    // Advances: 32 cols × 7 groups × 3 heads = 672.
    if (state == S_H_NEXT_SXV_COL)
        sb_advance = 1'b1;

    // ── MHA — W_proj ─────────────────────────────────────────────────────────
    // Each (row_group, proj_col) pair → one 7-element vector.
    // Advances: 96 cols × 7 groups = 672.
    if (state == S_H_PROJ_NEXT_COL)
        sb_advance = 1'b1;

    // ── MHA — FFN1 ───────────────────────────────────────────────────────────
    // Each (row_group, ffn1_col) pair → one 7-element vector.
    // Advances: 384 cols × 7 groups = 2 688.
    if (state == S_H_FFN1_NEXT_COL)
        sb_advance = 1'b1;

    // ── MHA — FFN2 ───────────────────────────────────────────────────────────
    // Each (row_group, ffn2_col) pair → one 7-element vector.
    // Advances: 96 cols × 7 groups = 672.
    if (state == S_H_FFN2_NEXT_COL)
        sb_advance = 1'b1;

end : shift_buf_ctrl

endmodule