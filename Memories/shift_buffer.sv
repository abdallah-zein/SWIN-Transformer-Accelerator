// =============================================================================
// shift_buffer.sv
//
// Per-row-group quantization shift-amount table for the rounding_shifter.
//
// ── Purpose ──────────────────────────────────────────────────────────────────
// The rounding_shifter (quantizer) needs a different shift_amt for every
// group of 7 output elements that the MMU produces.  This module holds a
// pre-loaded table of 8-bit signed shift values and presents the correct
// value to the quantizer at all times.
//
// ── How it works ─────────────────────────────────────────────────────────────
//  1. Before an operation begins, the CPU/DMA writes the required shift values
//     into the internal RAM through the cpu_wr_* port (32-bit bus → 4 shift
//     values packed per word).
//  2. The unified_controller asserts sb_op_start once at the start of each
//     operation (or sub-operation boundary) and supplies the entry base address
//     sb_op_base_addr.  The read pointer is immediately reset to that address.
//  3. After each 7-element row-group writeback, the controller asserts
//     sb_advance for one cycle.  The read pointer increments by one, selecting
//     the next 8-bit shift value for the following row-group.
//  4. shift_amt is combinatorially decoded from the current read pointer;
//     it is stable throughout the entire 7-element writeback.
//
// ── Memory layout (default DEPTH = 16 384 entries) ───────────────────────────
//   Region        Base      Size      Calculation
//   ─────────────────────────────────────────────────────────────────────────
//   Conv          0         5 376     96 kernels × 56 output rows
//   MLP           5 376       448     448 row-groups (3 136 rows / 7)
//                                     Same table reused by L1 and L2
//   MHA / window  5 824     7 749     All sub-operations per 7×7 window
//     QKV  (Q+K+V)          2 016     96 cols × 7 patch-groups × 3 matrices
//     Attention (QKᵀ)       1 029     49 cols × 7 row-groups   × 3 heads
//     SxV                     672     32 cols × 7 row-groups   × 3 heads
//     W_proj                  672     96 cols × 7 row-groups
//     FFN1                  2 688     384 cols × 7 row-groups
//     FFN2                    672     96 cols × 7 row-groups
//   ─────────────────────────────────────────────────────────────────────────
//   Total used: 13 573 entries  ≤  DEPTH = 16 384  ✓
//
//   The MHA sub-operations are laid out contiguously at SB_MHA_BASE.
//   The controller advances the pointer through all 7 749 entries in one
//   window pass, then resets back to SB_MHA_BASE for the next window
//   (same weights → same shift values for every window).
//
// ── CPU write bus format ─────────────────────────────────────────────────────
//   The 32-bit bus packs four 8-bit shift values per word:
//     bits [7:0]   → entry N+0
//     bits [15:8]  → entry N+1
//     bits [23:16] → entry N+2
//     bits [31:24] → entry N+3
//   cpu_wr_addr is a WORD address (entry_address >> 2).
//   Example: to write shift values for entries 8..11 supply cpu_wr_addr = 2.
//
// ── Priority ─────────────────────────────────────────────────────────────────
//   sb_op_start   >  sb_advance
//   (op_start resets the pointer; simultaneous advance is absorbed)
//
// =============================================================================

module shift_buffer #(
    parameter int DEPTH = 16384,           // total 8-bit shift entries storable
    parameter int AW    = $clog2(DEPTH),   // entry address width  (14 for DEPTH=16384)
    parameter int DW    = 8                // shift value width; must match rounding_shifter W_SHIFT
)(
    input  logic          clk,
    input  logic          rst_n,

    // ── CPU / DMA write port (32-bit bus, 4 entries per word) ────────────────
    // cpu_wr_addr  : word address = desired_entry_addr >> 2  (AW-2 bits wide)
    // cpu_wr_data  : four packed DW-bit shift values (LSB-first byte order)
    // cpu_wr_en    : write-enable strobe (one cycle)
    input  logic [AW-3:0] cpu_wr_addr,    // 12-bit word address for DEPTH=16384
    input  logic [31:0]   cpu_wr_data,
    input  logic          cpu_wr_en,

    // ── Controller interface ──────────────────────────────────────────────────
    // sb_op_start     : one-cycle pulse at the start of each operation or
    //                   each MHA window.  Resets the read pointer to
    //                   sb_op_base_addr on the same rising edge.
    // sb_op_base_addr : entry (byte) address of the first shift value for
    //                   this operation.  Sampled only when sb_op_start is high.
    // sb_advance      : one-cycle pulse after each 7-element row-group
    //                   writeback is complete.  Increments the read pointer
    //                   by 1 so the next shift value is ready immediately.
    input  logic           sb_op_start,
    input  logic [AW-1:0]  sb_op_base_addr,
    input  logic           sb_advance,

    // ── Output to rounding_shifter ────────────────────────────────────────────
    // Combinatorial; stable for the entire duration of a 7-element writeback.
    // Remains valid one cycle after the last sb_advance (the pointer has
    // already moved to the next position).
    output logic signed [DW-1:0] shift_amt
);

// ---------------------------------------------------------------------------
// Derived constants
// ---------------------------------------------------------------------------
localparam int RAM_DEPTH = DEPTH / 4;   // 4096 words for DEPTH=16384
localparam int RAM_AW    = AW - 2;      // 12-bit word address for DEPTH=16384

// ---------------------------------------------------------------------------
// Internal RAM  (synchronous write, asynchronous read)
// ---------------------------------------------------------------------------
(* ram_style = "auto" *)
logic [31:0] mem [0:RAM_DEPTH-1];

// Simulation initialisation — synthesis ignores initial blocks.
`ifdef SIMULATION
initial begin
    for (int i = 0; i < RAM_DEPTH; i++) mem[i] = 32'h0;
end
`endif

// ---------------------------------------------------------------------------
// Read pointer  (entry address)
// ---------------------------------------------------------------------------
logic [AW-1:0] rd_ptr;

// ---------------------------------------------------------------------------
// CPU write  (synchronous, word-granular)
// ---------------------------------------------------------------------------
always_ff @(posedge clk) begin
    if (cpu_wr_en)
        mem[cpu_wr_addr] <= cpu_wr_data;
end

// ---------------------------------------------------------------------------
// Pointer control
//   Priority:  sb_op_start  >  sb_advance
// ---------------------------------------------------------------------------
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_ptr <= '0;
    end else if (sb_op_start) begin
        // Reset to the base address of the new operation.
        // Any simultaneous sb_advance is silently absorbed.
        rd_ptr <= sb_op_base_addr;
    end else if (sb_advance) begin
        // Step to the next shift value for the next 7-element row-group.
        rd_ptr <= rd_ptr + 1'b1;
    end
end

// ---------------------------------------------------------------------------
// Output decode  (fully combinatorial — zero extra latency)
//
//   rd_ptr[AW-1 : 2]  →  selects the 32-bit RAM word
//   rd_ptr[1:0]        →  selects which 8-bit byte within that word
// ---------------------------------------------------------------------------
logic [31:0] rd_word;
logic [1:0]  byte_sel;

assign rd_word  = mem[rd_ptr[AW-1:2]];
assign byte_sel = rd_ptr[1:0];

always_comb begin
    unique case (byte_sel)
        2'b00: shift_amt = signed'(rd_word[ 7: 0]);
        2'b01: shift_amt = signed'(rd_word[15: 8]);
        2'b10: shift_amt = signed'(rd_word[23:16]);
        2'b11: shift_amt = signed'(rd_word[31:24]);
    endcase
end

endmodule
