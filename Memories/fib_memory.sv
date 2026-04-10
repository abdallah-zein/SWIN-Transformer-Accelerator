// =============================================================================
// fib_memory.sv  (rev 4 — Patch Embedding address map corrected)
//
// ── What changed from rev 3 ───────────────────────────────────────────────
//   • Conv/Patch Embedding address map in comments corrected to match the
//     actual 4×4-patch, stride-4 architecture of the Swin Transformer:
//       - Patch size:  4×4 pixels  (NOT 7×7 or any other)
//       - Stride:      4           (non-overlapping, equals patch size)
//       - Output:      56×56 patches × 96 channels
//       - Input words: 224×224×3 / 4 B per word = 37,632 words  [0..37,631]
//   • CHW layout is confirmed and documented with explicit formula derivation.
//   • Per-mode address ranges now broken out clearly.
//   • DEPTH and AW are UNCHANGED (MHA still dominates at 75,264 words).
//
// ── Address Map ───────────────────────────────────────────────────────────
//
//   ┌───────────────────────────────────────────────────────────────────────┐
//   │ MODE 0  Patch Embedding (Conv)                                        │
//   │                                                                       │
//   │   Input:  224×224×3 image, INT8, 1 byte/pixel                        │
//   │   Packing: 4 pixels per 32-bit word (4 consecutive cols, same row+ch)│
//   │   Words: 224×224×3 / 4 = 37,632  →  [0 .. 37,631]                   │
//   │                                                                       │
//   │   Layout: CHW — channel outermost, then row, then col-word           │
//   │                                                                       │
//   │     addr = ch * C_IMG_CH_WORDS                                        │
//   │           + in_row * C_IMG_W_WORDS                                    │
//   │           + col_word                                                  │
//   │                                                                       │
//   │   Where:                                                              │
//   │     C_IMG_CH_WORDS = 224 × 224 / 4 = 12,544   (words per channel)   │
//   │     C_IMG_W_WORDS  = 224 / 4       =     56   (words per image row)  │
//   │     ch      ∈ [0..2]        (R, G, B)                                │
//   │     in_row  ∈ [0..223]      (pixel row in the 224×224 image)         │
//   │     col_word∈ [0..55]       (group of 4 consecutive pixels in a row) │
//   │                                                                       │
//   │   Controller computes (for output patch at out_row, out_col_chunk,   │
//   │   window w, and PE p):                                                │
//   │     ch       = p >> 2                  (PE 0-3 → ch0, 4-7 → ch1, …) │
//   │     in_row   = out_row*4 + (p & 3)     (4 input rows per output row) │
//   │     col_word = out_col_chunk*7 + w      (7 windows × stride-4)       │
//   │                                                                       │
//   │   Why in_row = out_row*4 + sub_row:                                  │
//   │     Stride=4 means output row r reads input rows [r*4 .. r*4+3].     │
//   │     The 4 sub-rows are covered by PE[1:0] (bits 1,0 of PE index).    │
//   │                                                                       │
//   │   Why col_word = chunk*7 + win:                                       │
//   │     N_WIN=7 output columns are computed in parallel (one chunk).      │
//   │     Output col c = chunk*7+win → input cols [(chunk*7+win)*4 ..      │
//   │     (chunk*7+win)*4+3] → col_word = chunk*7+win (4 px packed).       │
//   │                                                                       │
//   │   Max address: 2*12544 + 223*56 + 55 = 25088+12488+55 = 37,631 ✓    │
//   ├───────────────────────────────────────────────────────────────────────┤
//   │ MODE 1  Patch Merging  (MLP)                                          │
//   │                                                                       │
//   │   Input:  3,136 patches × 96 features, INT8                          │
//   │   Packing: 4 bytes per word                                           │
//   │   Words: 3,136 × 96 / 4 = 3,136 × 24 = 75,264  →  [0 .. 75,263]    │
//   │                                                                       │
//   │   Layout: patch-major                                                 │
//   │     addr = patch_idx * 24 + k_word                                    │
//   │     where patch_idx ∈ [0..3135], k_word ∈ [0..23]                   │
//   ├───────────────────────────────────────────────────────────────────────┤
//   │ MODE 2/3  Swin Block  (W-MSA / SW-MSA)                                │
//   │                                                                       │
//   │   Input:  56×56×96 feature map, INT8  (Patch Embedding output)       │
//   │   Packing: 4 bytes per word                                           │
//   │   Words: 56×56×96 / 4 = 75,264  →  [0 .. 75,263]                    │
//   │                                                                       │
//   │   Layout: row-major spatial, feature-minor                            │
//   │     addr = (row*56 + col) * 24 + k_word                              │
//   │          = patch_idx * 24 + k_word                                    │
//   │     where row, col ∈ [0..55],  patch_idx = row*56+col,  k_word∈[0..23]│
//   │                                                                       │
//   │   SW-MSA uses Port B (cyclic-shifted read) — same data in memory,    │
//   │   different address mapping applied at read time (see below).         │
//   └───────────────────────────────────────────────────────────────────────┘
//
//   Maximum depth = 75,264 words  (Patch Merging / Swin Block)
//   AW = 17  (2^17 = 131,072 ≥ 75,264 ✓)
//
// ── Port summary (unchanged from rev 3) ──────────────────────────────────
//   Write port (wr_*)     : DMA / CPU loads image or feature map before start
//   Port A (rd_*)         : direct read — Conv, Patch Merging, W-MSA
//   Port B (sw_rd_*)      : cyclic-shifted read — SW-MSA only
//                           controller supplies LOGICAL (row, col, k_word);
//                           FIB adds (shift_h, shift_w) mod (FM_H, FM_W)
// =============================================================================

module fib_memory #(
    parameter int  DEPTH   = 75264,   // words — Patch Merging/Swin Block dominant
    parameter int  AW      = 17,      // ceil(log2(75264)) = 17
    // Feature-map spatial dims for SW-MSA (mode 2/3) address computation
    parameter int  FM_H    = 56,
    parameter int  FM_W    = 56,
    parameter int  FM_CH_W = 24       // channel words per patch (96 ch / 4 B)
)(
    input  logic          clk,
    input  logic          rst_n,

    // ── Write port (DMA / CPU) ────────────────────────────────────────────
    input  logic [AW-1:0] wr_addr,
    input  logic [31:0]   wr_data,
    input  logic          wr_en,

    // ════════════════════════════════════════════════════════════════════
    // PORT A — direct-address read  (Conv, Patch Merging, W-MSA)
    // ════════════════════════════════════════════════════════════════════
    input  logic [AW-1:0] rd_addr,
    input  logic          rd_en,
    output logic [31:0]   rd_data,

    // ════════════════════════════════════════════════════════════════════
    // PORT B — cyclic-shifted read  (SW-MSA only)
    //
    // Controller supplies logical (row, col, k_word) in the SHIFTED space.
    // FIB computes:
    //   phys_row  = (sw_rd_row + shift_h) % FM_H
    //   phys_col  = (sw_rd_col + shift_w) % FM_W
    //   phys_addr = (phys_row * FM_W + phys_col) * FM_CH_W + sw_rd_k_word
    //
    // Modulo is a single compare-and-subtract (sum < 2×dim, so one step suffices).
    // ════════════════════════════════════════════════════════════════════
    input  logic [$clog2(FM_H)-1:0]    sw_rd_row,
    input  logic [$clog2(FM_W)-1:0]    sw_rd_col,
    input  logic [$clog2(FM_CH_W)-1:0] sw_rd_k_word,
    input  logic                        sw_rd_en,
    output logic [31:0]                 sw_rd_data,

    // Shift amounts — set before SW-MSA round; default window_size=7 → shift=3
    input  logic [$clog2(FM_H)-1:0]    shift_h,
    input  logic [$clog2(FM_W)-1:0]    shift_w
);

    logic [31:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    // ── Write (synchronous) ───────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    // ── Port A: direct read (1-cycle latency) ─────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)      rd_data <= '0;
        else if (rd_en)  rd_data <= mem[rd_addr];
    end

    // ── Port B: cyclic-shifted read ────────────────────────────────────────
    //
    // phys_row = (sw_rd_row + shift_h) >= FM_H
    //            ? (sw_rd_row + shift_h) - FM_H
    //            : (sw_rd_row + shift_h)
    //
    // Same for phys_col.  No divider needed: shift < FM_H,W guarantees
    // the sum never exceeds 2×FM_H,W so a single subtraction suffices.
    //
    logic [$clog2(FM_H):0]   row_sum;
    logic [$clog2(FM_W):0]   col_sum;
    logic [$clog2(FM_H)-1:0] phys_row;
    logic [$clog2(FM_W)-1:0] phys_col;
    logic [AW-1:0]            sw_phys_addr;

    always_comb begin
        row_sum = {1'b0, sw_rd_row} + {1'b0, shift_h};
        col_sum = {1'b0, sw_rd_col} + {1'b0, shift_w};

        phys_row = (row_sum >= $clog2(FM_H)'(FM_H))
                   ? row_sum[$clog2(FM_H)-1:0] - $clog2(FM_H)'(FM_H)
                   : row_sum[$clog2(FM_H)-1:0];

        phys_col = (col_sum >= $clog2(FM_W)'(FM_W))
                   ? col_sum[$clog2(FM_W)-1:0] - $clog2(FM_W)'(FM_W)
                   : col_sum[$clog2(FM_W)-1:0];

        sw_phys_addr = AW'(phys_row) * AW'(FM_W * FM_CH_W)
                     + AW'(phys_col) * AW'(FM_CH_W)
                     + AW'(sw_rd_k_word);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)         sw_rd_data <= '0;
        else if (sw_rd_en)  sw_rd_data <= mem[sw_phys_addr];
    end

endmodule