// =============================================================================
// weight_memory.sv  (rev 3 — SWIN Block round weight storage)
//
// ── Round boundary clarification ─────────────────────────────────────────
//
//   The accelerator has three operational modes (from paper):
//
//   Mode 2'b00  PATCH EMBEDDING  — one full round by itself.
//                                  Writes result to off-chip at end.
//   Mode 2'b01  PATCH MERGING   — one full round by itself.
//                                  Writes result to off-chip at end.
//   Mode 2'b10  SWIN BLOCK      — ONE single round that executes BOTH:
//                                  1) W-MSA / SW-MSA  + shortcut
//                                  2) FFN  (W1 → GELU → W2) + shortcut
//                                  Only MWU writes to off-chip at the
//                                  END of the FFN, not between MSA and FFN.
//
//   Therefore weight_memory must hold ALL weights needed for the full SWIN
//   Block round before it starts: W_Q, W_K, W_V, W_Proj, W_FFN1, W_FFN2.
//
// ── Address Map ──────────────────────────────────────────────────────────
//
//   ┌──────────────────────────────────────────────────────────────────────┐
//   │ MODE 2'b00  Patch Embedding                                          │
//   │   96 kernels × 13 words (12 PE weights + 1 bias) = 1,248 words      │
//   │   [0 .. 1247]                                                        │
//   ├──────────────────────────────────────────────────────────────────────┤
//   │ MODE 2'b01  Patch Merging  (same layout as legacy MLP)               │
//   │   W1: 384 cols × 24 w/col  = 9,216 words  [0 ..  9215]              │
//   │   W2:  96 cols × 96 w/col  = 9,216 words  [9216 .. 18431]           │
//   ├──────────────────────────────────────────────────────────────────────┤
//   │ MODE 2'b10  SWIN Transformer Block (MSA + FFN — single round)        │
//   │                                                                      │
//   │   W_Q    (96×96, 24 w/col) : [     0 ..  2303]  2304 words          │
//   │   W_K    (96×96, 24 w/col) : [  2304 ..  4607]  2304 words          │
//   │   W_V    (96×96, 24 w/col) : [  4608 ..  6911]  2304 words          │
//   │   W_Proj (96×96, 24 w/col) : [  6912 ..  9215]  2304 words          │
//   │   W_FFN1 (384×96,96 w/col) : [  9216 .. 18431]  9216 words          │
//   │   W_FFN2 (96×384,24 w/col) : [ 18432 .. 27647]  9216 words          │
//   │                                                                      │
//   │   Total SWIN Block words = 27,648                                    │
//   └──────────────────────────────────────────────────────────────────────┘
//
//   Maximum depth = 27,648 words  →  AW = 15  (2^15 = 32768 ≥ 27648)
//   (No AW change needed from rev 2; 15 bits already sufficient.)
//
// ── W_FFN2 column layout note ─────────────────────────────────────────────
//   W_FFN2 maps 384 inputs → 96 outputs.
//   Stored col-major: 96 output columns, each column has 384/4 = 96 words.
//   Controller reads column c  →  base W_FFN2_BASE + c*96 + word_offset.
//   This is the same word-per-column count as legacy W2 in MLP mode.
//
// ── Interface ────────────────────────────────────────────────────────────
//   Single write port  : loaded by external DMA / CPU before engine start.
//   Single read port   : driven by the controller (unified_controller).
//   Read latency       : 1 cycle (rd_data valid the cycle after rd_en).
// =============================================================================

module weight_memory #(
    parameter int DEPTH = 27648,   // words — covers full SWIN Block round
    parameter int AW    = 15       // ceil(log2(27648)) = 15  (unchanged)
)(
    input  logic          clk,
    input  logic          rst_n,

    // ── Write port (CPU / DMA — loads all weights before engine start) ────
    input  logic [AW-1:0] wr_addr,
    input  logic [31:0]   wr_data,
    input  logic          wr_en,

    // ── Read port (to controller → weight buffers) ────────────────────────
    input  logic [AW-1:0] rd_addr,
    input  logic          rd_en,
    output logic [31:0]   rd_data
);

    logic [31:0] mem [0:DEPTH-1];

    // Initialise to zero (simulation only; synthesis infers SRAM)
    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    // ── Write ─────────────────────────────────────────────────────────────
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
// Weight Base-Address Constants (for use in unified_controller)
//
// ── Patch Embedding (mode 2'b00) ──────────────────────────────────────────
//   CONV_W_BASE   =     0   (96 kernels × 13 words each)
//
// ── Patch Merging (mode 2'b01) ────────────────────────────────────────────
//   PM_W1_BASE    =     0   (384 cols × 24 w/col = 9216 words)
//   PM_W2_BASE    =  9216   (96  cols × 96 w/col = 9216 words)
//
// ── SWIN Transformer Block (mode 2'b10) ──────────────────────────────────
//   MSA_WQ_BASE   =     0   (96 cols × 24 w/col)
//   MSA_WK_BASE   =  2304   (96 cols × 24 w/col)
//   MSA_WV_BASE   =  4608   (96 cols × 24 w/col)
//   MSA_WP_BASE   =  6912   (96 cols × 24 w/col)  [Proj]
//   FFN_W1_BASE   =  9216   (96 cols × 96 w/col)  [expand 96→384]
//   FFN_W2_BASE   = 18432   (384 cols × 24 w/col) [compress 384→96]
//
//   MSA_QKV_COL_WORDS = 24  (96 weights ÷ 4 B/word)
//   FFN_W1_COL_WORDS  = 96  (384 weights ÷ 4 B/word)
//   FFN_W2_COL_WORDS  = 24  (96 weights ÷ 4 B/word)  ← same as MSA
// =============================================================================
