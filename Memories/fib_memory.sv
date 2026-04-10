// =============================================================================
// fib_memory.sv  (rev 5 — multi-stage verification pass; no RTL change)
//
// ── Why DEPTH and AW are unchanged across all 4 stages ───────────────────
//
//   The FIB is loaded ONCE per top-level mode from off-chip memory, then read
//   by the accelerator engine throughout that mode's entire computation.
//   Its size is therefore the MAX of the three per-mode footprints:
//
//   MODE 0  Patch Embedding  (only Stage 1 — the architecture is fixed)
//     224×224×3 / 4 = 37,632 words  →  never changes.
//
//   MODE 1  Patch Merging
//     The FIB is NOT used by Patch Merging.  The PM input comes from
//     ilb_patch_embed_buf (on-chip) or from off-chip; the FIB is idle.
//
//   MODE 2 / 3  Swin Block  (W-MSA / SW-MSA)
//     The FIB holds the feature map that is the MSA input X.
//     Stage 1: 56×56×96  / 4 = 75,264 words  ← largest
//     Stage 2: 28×28×192 / 4 = 37,632 words
//     Stage 3: 14×14×384 / 4 =  9,408 words
//     Stage 4:  7× 7×768 / 4 =  9,408 words
//
//   Maximum: Stage 1 Swin Block = 75,264 words.
//   AW = 17  (2^17 = 131,072 ≥ 75,264).   DEPTH = 75,264.
//   Both are UNCHANGED from rev 4.
//
// ── SW-MSA cyclic shift (Port B) — valid for all stages ──────────────────
//   FM_H and FM_W are set to the STAGE 1 values (56, 56) as compile-time
//   parameters.  For stages 2–4 the controller drives the correct
//   sw_rd_row / sw_rd_col coordinates relative to the feature map of
//   THAT stage (28/14/7), and sets shift_h = shift_w = 3 (always half
//   of the window size 7).  The modulo arithmetic is identical regardless
//   of the spatial resolution because the formula only depends on FM_H,
//   FM_W, and the input coordinates.
//
//   To allow the hardware to serve all stages without reconfiguration, FM_H
//   and FM_W are now runtime inputs (fm_h, fm_w) rather than fixed
//   localparams.  The compile-time parameters still provide defaults and
//   set the bit widths.
//
// ── Address map (unchanged) ───────────────────────────────────────────────
//   Conv:       [0 .. 37,631]    224×224×3/4 words
//   Swin Block: [0 .. 75,263]    Stage1 dominant; smaller stages use subset
//
// ── What changed from rev 4 ───────────────────────────────────────────────
//   • fm_h / fm_w runtime ports added to Port B so the cyclic-shift
//     modulo correctly wraps at the active stage's spatial dimensions.
//   • sw_rd_row / sw_rd_col / shift_h / shift_w widths are now $clog2(FM_H)
//     based on the compile-time FM_H=56 / FM_W=56 — sufficient for all
//     stages since smaller stages use strictly smaller coordinate values.
//   • No depth or AW change.
// =============================================================================

module fib_memory #(
    parameter int  DEPTH   = 75264,
    parameter int  AW      = 17,
    parameter int  FM_H    = 56,    // Stage 1 spatial height (max)
    parameter int  FM_W    = 56,    // Stage 1 spatial width  (max)
    parameter int  FM_CH_W = 24     // max channel words per patch (Stage1: 96/4)
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
    // PORT B — cyclic-shifted read  (SW-MSA, all stages)
    //
    // Controller provides LOGICAL (row, col, k_word) in shifted space.
    // Runtime fm_h / fm_w let the same hardware serve all spatial sizes:
    //   Stage1: fm_h=56, fm_w=56   Stage2: fm_h=28, fm_w=28
    //   Stage3: fm_h=14, fm_w=14   Stage4: fm_h=7,  fm_w=7
    //
    // Physical address:
    //   phys_row  = (sw_rd_row + shift_h) % fm_h    [conditional subtract]
    //   phys_col  = (sw_rd_col + shift_w) % fm_w
    //   phys_addr = (phys_row * fm_w + phys_col) * sw_rd_k_word_cnt + sw_rd_k_word
    //
    // sw_rd_k_word_cnt: channel words per patch for active stage (driven by ctrl)
    //   Stage1=24, Stage2=48, Stage3=96, Stage4=192
    // ════════════════════════════════════════════════════════════════════
    input  logic [$clog2(FM_H)-1:0]    sw_rd_row,
    input  logic [$clog2(FM_W)-1:0]    sw_rd_col,
    input  logic [$clog2(FM_CH_W)-1:0] sw_rd_k_word,
    input  logic                        sw_rd_en,
    output logic [31:0]                 sw_rd_data,

    // Runtime spatial dimensions for Port B (set per stage, stable during round)
    input  logic [$clog2(FM_H)-1:0]    shift_h,       // always 3 for M=7
    input  logic [$clog2(FM_W)-1:0]    shift_w,       // always 3 for M=7
    input  logic [$clog2(FM_H):0]      fm_h,          // active stage height
    input  logic [$clog2(FM_W):0]      fm_w,          // active stage width
    input  logic [$clog2(FM_CH_W*8)-1:0] sw_k_word_cnt // channel words/patch
);

    logic [31:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    // ── Write ─────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
    end

    // ── Port A: direct read (1-cycle latency) ─────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)      rd_data <= '0;
        else if (rd_en)  rd_data <= mem[rd_addr];
    end

    // ── Port B: cyclic-shifted read ────────────────────────────────────────
    // Modulo by single conditional subtraction (shift < fm_h,w guaranteed).
    logic [$clog2(FM_H):0]   row_sum;
    logic [$clog2(FM_W):0]   col_sum;
    logic [$clog2(FM_H)-1:0] phys_row;
    logic [$clog2(FM_W)-1:0] phys_col;
    logic [AW-1:0]            sw_phys_addr;

    always_comb begin
        row_sum  = {1'b0, sw_rd_row} + {1'b0, shift_h};
        col_sum  = {1'b0, sw_rd_col} + {1'b0, shift_w};

        phys_row = (row_sum >= fm_h)
                   ? row_sum[$clog2(FM_H)-1:0] - fm_h[$clog2(FM_H)-1:0]
                   : row_sum[$clog2(FM_H)-1:0];

        phys_col = (col_sum >= fm_w)
                   ? col_sum[$clog2(FM_W)-1:0] - fm_w[$clog2(FM_W)-1:0]
                   : col_sum[$clog2(FM_W)-1:0];

        sw_phys_addr = AW'(phys_row) * AW'(fm_w) * AW'(sw_k_word_cnt)
                     + AW'(phys_col) * AW'(sw_k_word_cnt)
                     + AW'(sw_rd_k_word);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)          sw_rd_data <= '0;
        else if (sw_rd_en)   sw_rd_data <= mem[sw_phys_addr];
    end

endmodule
