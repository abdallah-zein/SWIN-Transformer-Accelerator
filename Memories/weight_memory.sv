// =============================================================================
// weight_memory.sv  (rev 5 — all 4 Swin stages supported)
//
// ── What changed from rev 4 ───────────────────────────────────────────────
//   DEPTH and AW updated to cover Stage 4 (the largest case).
//   All address-map localparams parameterised by stage.
//   No interface changes — same single write port, single read port.
//
// ── Sizing summary (dominant dimension per mode) ──────────────────────────
//
//   Mode 2'b00  Patch Embedding  (only stage 1, architecture fixed)
//     96 kernels × 12 words/kernel = 1,152 words  [0..1151]
//     (bias excluded — lives in bias_buffer)
//
//   Mode 2'b01  Patch Merging  (up to PM3: W=(1536,768) & W=(768,1536))
//     PM1: W1 (384×192)  = 18,432 w  +  W2 (192×384)  = 18,432 w = 36,864
//     PM2: W1 (768×384)  = 73,728 w  +  W2 (384×768)  = 73,728 w = 147,456
//     PM3: W1 (1536×768) = 294,912 w +  W2 (768×1536) = 294,912 w = 589,824
//     Dominant: PM3 = 589,824 words
//
//   Mode 2'b10  Swin Transformer Block  (Stage 4 largest)
//     Stage 4 (C=768, FFN=3072):
//       W_Q/K/V/P each: 768×768 / 4 = 147,456 words  × 4 = 589,824
//       W_FFN1:  3072×768 / 4 = 589,824 words
//       W_FFN2:  768×3072 / 4 = 589,824 words
//       TOTAL Stage 4 = 4 × 147,456 + 2 × 589,824 = 1,769,472 words
//
//   Maximum across all modes: Stage 4 Swin = 1,769,472 words
//   AW = 21  (2^21 = 2,097,152 ≥ 1,769,472)
//
// ── Address map — Patch Merging mode (2'b01) ──────────────────────────────
//   The DMA loads the correct PM stage before inference.  All PM stages
//   reuse [0 .. PM_W1_WORDS + PM_W2_WORDS - 1]:
//
//   PM1: W1 [0..18431] (384 cols × 48 w/col)   W2 [18432..36863] (192 cols × 96 w/col)
//   PM2: W1 [0..73727]                          W2 [73728..147455]
//   PM3: W1 [0..294911]                         W2 [294912..589823]
//
//   Column word counts (INT8, 32-bit bus):
//     PM1 W1 col = 192/4 = 48 w    PM1 W2 col = 384/4 =  96 w
//     PM2 W1 col = 384/4 = 96 w    PM2 W2 col = 768/4 = 192 w
//     PM3 W1 col = 768/4 = 192 w   PM3 W2 col = 1536/4= 384 w
//
// ── Address map — Swin Block mode (2'b10) ─────────────────────────────────
//   Relative offsets within the loaded stage (same layout for all stages):
//
//   Stage s, C = channel depth:
//     col_words = C / 4
//     ffn1_col  = FFN_C / 4  (FFN_C = 4*C)
//
//   W_Q    [0                     .. C*col_words - 1]
//   W_K    [C*col_words           .. 2*C*col_words - 1]
//   W_V    [2*C*col_words         .. 3*C*col_words - 1]
//   W_Proj [3*C*col_words         .. 4*C*col_words - 1]
//   W_FFN1 [4*C*col_words         .. 4*C*col_words + FFN_C*col_words - 1]
//   W_FFN2 [4*C*col_words+FFN1_sz .. 4*C*col_words + 2*FFN_C*col_words - 1]
//
//   Concrete totals per stage:
//     Stage 1 (C=96,  FFN=384):   27,648 words
//     Stage 2 (C=192, FFN=768):  110,592 words
//     Stage 3 (C=384, FFN=1536): 442,368 words
//     Stage 4 (C=768, FFN=3072): 1,769,472 words  ← DEPTH ceiling
//
// ── Interface ─────────────────────────────────────────────────────────────
//   Single write port (DMA/CPU, before engine start).
//   Single read port  (unified_controller → weight buffer).
//   Read latency: 1 cycle.
// =============================================================================

module weight_memory #(
    parameter int DEPTH = 1769472,  // Stage 4 Swin Block (dominant)
    parameter int AW    = 21        // ceil(log2(1769472)) = 21
)(
    input  logic          clk,
    input  logic          rst_n,

    input  logic [AW-1:0] wr_addr,
    input  logic [31:0]   wr_data,
    input  logic          wr_en,

    input  logic [AW-1:0] rd_addr,
    input  logic          rd_en,
    output logic [31:0]   rd_data
);

    logic [31:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    always_ff @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)      rd_data <= '0;
        else if (rd_en)  rd_data <= mem[rd_addr];
    end

endmodule

// =============================================================================
// Weight Base-Address Constants  (localparams for unified_controller)
//
// ── Patch Embedding  (mode 2'b00) ─────────────────────────────────────────
//   CONV_W_BASE       = 0
//   CONV_W_WORDS_KERN = 12   (12 PE weights, no bias)
//
// ── Patch Merging    (mode 2'b01) — set by controller for active PM stage ─
//   PM_W1_BASE  = 0          (always starts at 0)
//   PM_W2_BASE  = PM_W1_WORDS (immediately after W1 region)
//
//   Stage-specific col words:
//     PM1: W1_col=48, W2_col=96,  W1_cols=384, W2_cols=192
//     PM2: W1_col=96, W2_col=192, W1_cols=768, W2_cols=384
//     PM3: W1_col=192,W2_col=384, W1_cols=1536,W2_cols=768
//
// ── Swin Block  (mode 2'b10) — relative to stage base (always 0) ──────────
//   For stage with channel depth C and FFN depth FFN_C = 4*C:
//     col_words  = C   / 4
//     ffn1_col   = FFN_C / 4
//
//   MSA_WQ_BASE  = 0
//   MSA_WK_BASE  = C * col_words
//   MSA_WV_BASE  = 2 * C * col_words
//   MSA_WP_BASE  = 3 * C * col_words
//   FFN_W1_BASE  = 4 * C * col_words
//   FFN_W2_BASE  = 4 * C * col_words + FFN_C * col_words
//
//   Stage 1 (C=96):  WQ/K/V/P_BASE=0/2304/4608/6912, FFN1=9216, FFN2=18432
//   Stage 2 (C=192): WQ/K/V/P_BASE=0/9216/18432/27648, FFN1=36864, FFN2=73728
//   Stage 3 (C=384): WQ/K/V/P_BASE=0/36864/73728/110592, FFN1=147456, FFN2=294912
//   Stage 4 (C=768): WQ/K/V/P_BASE=0/147456/294912/442368, FFN1=589824, FFN2=1179648
//
// NOTE: Bias values are NOT stored here — dedicated bias_buffer module.
// =============================================================================
