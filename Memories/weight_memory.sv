// =============================================================================
// weight_memory.sv  (rev 4 — bias removed from Conv kernel words)
//
// ── What changed from rev 3 ───────────────────────────────────────────────
//   • Conv kernel word count: 13 → 12 words per kernel (bias excluded).
//     96 kernels × 12 words = 1,152 words  (was 1,248).
//   • DEPTH: 27,648 → 27,636 if recalculated, BUT since SWIN Block (27,648)
//     still dominates the address space and AW=15 covers 32,768, DEPTH and AW
//     are left at their existing values for simplicity.  The Conv region now
//     uses [0..1151] and words [1152..1247] are simply unused in Conv mode.
//   • All SWIN Block and Patch Merging sections unchanged.
//   • Bias constants removed from the address-map comment block at the bottom.
//
// ── Address Map ──────────────────────────────────────────────────────────
//
//   ┌─────────────────────────────────────────────────────────────────────┐
//   │ MODE 2'b00  Patch Embedding                                         │
//   │   96 kernels × 12 words/kernel = 1,152 words  [0 .. 1151]          │
//   │   Word  k*12 + p  → PE-p weight word of kernel k  (p = 0..11)      │
//   │   Bias is stored in a SEPARATE bias buffer (not in weight_memory).  │
//   ├─────────────────────────────────────────────────────────────────────┤
//   │ MODE 2'b01  Patch Merging  (same as legacy MLP)                     │
//   │   W1: 384 cols × 24 w/col  =  9,216 words  [0    ..  9215]         │
//   │   W2:  96 cols × 96 w/col  =  9,216 words  [9216 .. 18431]         │
//   ├─────────────────────────────────────────────────────────────────────┤
//   │ MODE 2'b10  SWIN Transformer Block  (W-MSA / SW-MSA + FFN)          │
//   │                                                                     │
//   │   W_Q    (96×96,  24 w/col) : [     0 ..  2303]   2304 words       │
//   │   W_K    (96×96,  24 w/col) : [  2304 ..  4607]   2304 words       │
//   │   W_V    (96×96,  24 w/col) : [  4608 ..  6911]   2304 words       │
//   │   W_Proj (96×96,  24 w/col) : [  6912 ..  9215]   2304 words       │
//   │   W_FFN1 (384×96, 96 w/col) : [  9216 .. 18431]   9216 words       │
//   │   W_FFN2 (96×384, 24 w/col) : [ 18432 .. 27647]   9216 words       │
//   │                                                                     │
//   │   Total SWIN Block words = 27,648                                   │
//   └─────────────────────────────────────────────────────────────────────┘
//
//   Maximum depth  = 27,648  →  AW = 15  (2^15 = 32,768 ≥ 27,648)
//
// ── Interface ────────────────────────────────────────────────────────────
//   Single write port : loaded by DMA / CPU before engine start.
//   Single read port  : driven by unified_controller.
//   Read latency      : 1 cycle.
// =============================================================================

module weight_memory #(
    parameter int DEPTH = 27648,   // covers full SWIN Block round (dominant)
    parameter int AW    = 15       // ceil(log2(27648)) = 15
)(
    input  logic          clk,
    input  logic          rst_n,

    // ── Write port (DMA / CPU — loaded before engine start) ───────────────
    input  logic [AW-1:0] wr_addr,
    input  logic [31:0]   wr_data,
    input  logic          wr_en,

    // ── Read port (unified_controller → weight buffer) ────────────────────
    input  logic [AW-1:0] rd_addr,
    input  logic          rd_en,
    output logic [31:0]   rd_data
);

    logic [31:0] mem [0:DEPTH-1];

    // Initialise to zero (simulation only; synthesis infers BRAM / SRAM)
    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    // ── Write (synchronous) ───────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    // ── Read (registered, 1-cycle latency) ────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_data <= '0;
        else if (rd_en)
            rd_data <= mem[rd_addr];
    end

endmodule

// =============================================================================
// Weight Base-Address Constants  (use as localparams in unified_controller)
//
// ── Patch Embedding  (mode 2'b00) ─────────────────────────────────────────
//   CONV_W_BASE        =    0
//   CONV_W_WORDS_KERN  =   12   (12 PE weight words, NO bias)
//   Total              = 1152 words  [0..1151]
//
// ── Patch Merging    (mode 2'b01) ─────────────────────────────────────────
//   PM_W1_BASE         =    0   (384 cols × 24 w/col = 9216 words)
//   PM_W2_BASE         = 9216   ( 96 cols × 96 w/col = 9216 words)
//
// ── SWIN Transformer Block  (mode 2'b10) ──────────────────────────────────
//   MSA_WQ_BASE        =     0   (96 cols × 24 w/col)
//   MSA_WK_BASE        =  2304   (96 cols × 24 w/col)
//   MSA_WV_BASE        =  4608   (96 cols × 24 w/col)
//   MSA_WP_BASE        =  6912   (96 cols × 24 w/col)   [Projection]
//   FFN_W1_BASE        =  9216   (96 cols × 96 w/col)   [expand 96→384]
//   FFN_W2_BASE        = 18432   (96 cols × 96 w/col)   [compress 384→96]
//
//   MSA_QKV_COL_WORDS  = 24   (96 inputs  ÷ 4 B/word)
//   FFN_W1_COL_WORDS   = 96   (384 inputs ÷ 4 B/word)
//   FFN_W2_COL_WORDS   = 96   (384 inputs ÷ 4 B/word)
//
// NOTE: Bias words are NOT stored here.  All bias values (for Conv, Proj, FFN)
//       are loaded into a dedicated bias buffer before engine start.
// =============================================================================