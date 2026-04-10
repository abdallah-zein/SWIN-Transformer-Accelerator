// =============================================================================
// shift_buffer.sv  (rev 2 — all 4 Swin stages supported)
//
// ── What changed from rev 1 ───────────────────────────────────────────────
//   DEPTH: 16,384 → 131,072   (Stage 4 MHA dominant: see table below)
//   AW:        14 → 17
//   SB_MHA_BASE: 5,824 → 5,824  (unchanged — Conv + MLP slots are same)
//   MHA region updated to cover Stage 4 window = 61,992 entries
//
// ── Sizing table — all stages ─────────────────────────────────────────────
//
//   Region              Base    Size      Derivation
//   ─────────────────────────────────────────────────────────────────────────
//   Conv                   0    5,376     96 kernels × 56 output rows
//   MLP (all PM stages) 5,376     448     max row-groups = 3136/7 = 448
//                                         (same 448-entry table reused for L1+L2)
//   MHA / window         5,824  61,992    Stage 4 per-window (dominant):
//     QKV (3 mats)              16,128    768 cols × 7 groups × 3
//     Attention (24 hd)          8,232     49 cols × 7 groups × 24
//     SxV (24 hd)                5,376     32 cols × 7 groups × 24
//     W_proj                     5,376    768 cols × 7 groups
//     FFN1                      21,504   3072 cols × 7 groups
//     FFN2                       5,376    768 cols × 7 groups
//   ─────────────────────────────────────────────────────────────────────────
//   Total used: 5,376 + 448 + 61,992 = 67,816 entries
//   DEPTH = 131,072  (next power of 2 ≥ 67,816)
//   AW = 17  (2^17 = 131,072)
//
//   Note: Stages 1–3 use fewer MHA entries (smaller C / fewer heads) but
//   they fit within the same 61,992-slot region.  The controller writes only
//   the relevant portion before each round; unused slots are never addressed.
//
//   Stage-specific MHA entry counts per window:
//     Stage1 (C=96,  h=3):   QKV=2016, ATTN=1029, SxV=672, PROJ=672, FFN1=2688, FFN2=672  = 7749
//     Stage2 (C=192, h=6):   QKV=4032, ATTN=2058, SxV=1344,PROJ=1344,FFN1=5376, FFN2=1344 = 15498
//     Stage3 (C=384, h=12):  QKV=8064, ATTN=4116, SxV=2688,PROJ=2688,FFN1=10752,FFN2=2688 = 30996
//     Stage4 (C=768, h=24):  QKV=16128,ATTN=8232,SxV=5376,PROJ=5376,FFN1=21504,FFN2=5376 = 61992
//
// ── CPU write bus format (unchanged) ─────────────────────────────────────
//   32-bit word packs 4 shift values; cpu_wr_addr is a word address (entry >> 2).
//
// ── Priority (unchanged) ──────────────────────────────────────────────────
//   sb_op_start > sb_advance (op_start absorbs simultaneous advance)
// =============================================================================

module shift_buffer #(
    parameter int DEPTH = 131072,          // total 8-bit shift entries storable
    parameter int AW    = $clog2(DEPTH),   // 17 for DEPTH=131072
    parameter int DW    = 8                // shift value width
)(
    input  logic          clk,
    input  logic          rst_n,

    // ── CPU / DMA write port ─────────────────────────────────────────────
    input  logic [AW-3:0] cpu_wr_addr,    // word address = entry_addr >> 2
    input  logic [31:0]   cpu_wr_data,
    input  logic          cpu_wr_en,

    // ── Controller interface ─────────────────────────────────────────────
    input  logic           sb_op_start,
    input  logic [AW-1:0]  sb_op_base_addr,
    input  logic           sb_advance,

    // ── Output to rounding_shifter ───────────────────────────────────────
    output logic signed [DW-1:0] shift_amt
);

    // ── Internal byte-addressable RAM ─────────────────────────────────────
    // Stored as 32-bit words; each word holds 4 DW-bit shift values.
    localparam int WORD_DEPTH = DEPTH / 4;  // 32768 words

    logic [31:0] mem [0:WORD_DEPTH-1];

    initial begin
        for (int i = 0; i < WORD_DEPTH; i++) mem[i] = '0;
    end

    always_ff @(posedge clk) begin
        if (cpu_wr_en)
            mem[cpu_wr_addr] <= cpu_wr_data;
    end

    // ── Read pointer (entry address, byte-granular) ───────────────────────
    logic [AW-1:0] rd_ptr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= '0;
        end else if (sb_op_start) begin
            // op_start wins over advance
            rd_ptr <= sb_op_base_addr;
        end else if (sb_advance) begin
            rd_ptr <= rd_ptr + AW'(1);
        end
    end

    // ── Combinatorial read-out ────────────────────────────────────────────
    // Word address = rd_ptr[AW-1:2]; byte lane = rd_ptr[1:0]
    always_comb begin
        automatic logic [31:0] word = mem[rd_ptr[AW-1:2]];
        case (rd_ptr[1:0])
            2'd0: shift_amt = signed'(word[ 7: 0]);
            2'd1: shift_amt = signed'(word[15: 8]);
            2'd2: shift_amt = signed'(word[23:16]);
            2'd3: shift_amt = signed'(word[31:24]);
            default: shift_amt = '0;
        endcase
    end

endmodule
