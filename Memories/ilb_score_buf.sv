// =============================================================================
// ilb_score_buf.sv
//
// Intermediate Layer Buffer — Attention Score store (S = QK^T / √d_k)
// for ONE attention head and ONE 7×7 window.
//
// ── Role in the Swin Block dataflow ──────────────────────────────────────
//
//   Step 4  (from initial_thoughts.txt):
//     S = QK^T   →  stored here as INT32 (raw MMU accumulator, before Softmax)
//     In SW-MSA: S_masked = S + mask(q,k)  applied in-place via RMW cycle
//     S_soft = Softmax(S_masked)           done externally by SCU
//
//   The score matrix for one head is 49×49 = 2401 INT32 values = 9604 bytes.
//   Only ONE head fits here at a time; the controller processes heads
//   sequentially (h = 0, 1, 2) and reuses this buffer for each.
//
// ── Write (QK^T accumulation result) ─────────────────────────────────────
//   The MMU produces 7 rows of one score column per compute burst.
//   wr_row_base [5:0] : first patch row of this burst (0,7,14,...,42)
//   wr_col      [5:0] : key index / score column (0..48)
//   wr_en              : write strobe
//   wr_data  [6:0][31:0] : 7 INT32 accumulators from MMU
//
// ── In-place mask add (SW-MSA only) ──────────────────────────────────────
//   For SW-MSA the mask must be added to S before Softmax.
//   This is done via a read-modify-write:
//     1. Controller reads one word: rmw_rd_en, rmw_addr → rmw_rd_data (next cycle)
//     2. Adder (in full_system_top) adds mask_data_out to rmw_rd_data
//     3. Controller writes back: rmw_wr_en, rmw_addr, rmw_wr_data
//   A dedicated RMW port is provided so the main write port is not conflicted.
//   For W-MSA the mask is all-zero so the RMW can be skipped (mask=0 is a no-op).
//
// ── Read (feeding Softmax / S×V) ─────────────────────────────────────────
//   After masking, the SCU reads the score row-by-row to apply Softmax.
//   After Softmax the result (now INT8 after quantisation) is stored in
//   ilb_score_buf itself (overwriting the INT32 with INT8 in the same cell,
//   zero-extended to 32 bits — the high 24 bits are ignored by S×V).
//   rd_en, rd_addr → rd_data (1-cycle latency).
//
// ── Buffer sizing ─────────────────────────────────────────────────────────
//   49 × 49 × 4 bytes (INT32) = 9604 bytes = 2401 words of 32 bits.
//   No double-banking: the score matrix is produced and consumed within
//   the same head's processing pass before the head changes.
//   A valid flag (score_valid) tracks whether data is ready for Softmax.
//
// ── Address convention ────────────────────────────────────────────────────
//   Flat word address = query_row * 49 + key_col  (row-major, 0..2400)
// =============================================================================

module ilb_score_buf #(
    parameter int N_PATCHES = 49,    // sequence length (7×7 window)
    parameter int N_ROWS    = 7      // MMU output burst height
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── Valid flag — set when score matrix is fully written ───────────────
    // Controller asserts score_commit once all 49×49 words are stored.
    // Cleared automatically when score_clear is pulsed (before next head).
    input  logic        score_commit,  // 1-cycle pulse: matrix fully written
    input  logic        score_clear,   // 1-cycle pulse: clear before next head
    output logic        score_valid,   // high while data is ready for Softmax

    // ═════════════════════════════════════════════════════════════════════
    // WRITE PORT — QK^T accumulation
    //
    //   One burst per compute cycle: 7 query rows × 1 key column.
    //   wr_row_base steps 0,7,...,42 across 7 row-group passes.
    //   wr_col steps 0..48 for all key patches.
    // ═════════════════════════════════════════════════════════════════════
    input  logic        wr_en,
    input  logic [5:0]  wr_row_base,          // 0,7,14,21,28,35,42
    input  logic [5:0]  wr_col,               // key index 0..48
    input  logic [31:0] wr_data [0:N_ROWS-1], // 7 INT32 accumulators

    // ═════════════════════════════════════════════════════════════════════
    // READ-MODIFY-WRITE PORT — SW-MSA mask application
    //
    // Used to add the attention bias mask to each score in-place.
    // Protocol (2-cycle round-trip per word):
    //   Cycle 0: assert rmw_rd_en, provide rmw_addr
    //   Cycle 1: rmw_rd_data is valid; external adder computes sum
    //   Cycle 1: assert rmw_wr_en, provide rmw_addr and rmw_wr_data (the sum)
    //
    // For W-MSA: controller may skip the RMW entirely (mask = 0).
    // rmw_rd_en and wr_en must not be asserted simultaneously.
    // ═════════════════════════════════════════════════════════════════════
    input  logic        rmw_rd_en,
    input  logic [11:0] rmw_addr,             // 0..2400  (12 bits: 2^12=4096 ≥ 2401)
    output logic [31:0] rmw_rd_data,          // → external adder → rmw_wr_data
    input  logic        rmw_wr_en,
    input  logic [31:0] rmw_wr_data,          // masked (or Softmax-quantised) score

    // ═════════════════════════════════════════════════════════════════════
    // READ PORT — sequential read for Softmax input / S×V operand
    //
    // Used by the SCU to read score rows for Softmax, and by the S×V
    // compute step to read Softmax-quantised scores.
    // ═════════════════════════════════════════════════════════════════════
    input  logic        rd_en,
    input  logic [11:0] rd_addr,              // 0..2400
    output logic [31:0] rd_data               // 1-cycle latency
);

    // ── Storage: 2401 × 32-bit words ─────────────────────────────────────
    localparam int DEPTH = N_PATCHES * N_PATCHES;  // 49 × 49 = 2401

    logic [31:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    // ── score_valid flag ──────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            score_valid <= 1'b0;
        else if (score_clear)
            score_valid <= 1'b0;
        else if (score_commit)
            score_valid <= 1'b1;
    end

    // =========================================================================
    // Write port — QK^T burst
    // Writes 7 rows of one column simultaneously.
    // Each row r writes mem[( wr_row_base+r ) * N_PATCHES + wr_col].
    // =========================================================================
    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int q_row = int'(wr_row_base) + r;
                if (q_row < N_PATCHES) begin
                    automatic int addr = q_row * N_PATCHES + int'(wr_col);
                    mem[addr] <= wr_data[r];
                end
            end
        end
    end

    // =========================================================================
    // RMW read port (registered, 1-cycle latency)
    // Priority: rmw_wr_en overrides rd_en at the same address to avoid
    //           read-after-write hazards on the same cycle.
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rmw_rd_data <= '0;
        else if (rmw_rd_en)
            rmw_rd_data <= mem[rmw_addr];
    end

    // =========================================================================
    // RMW write port — mask-added value written back
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rmw_wr_en)
            mem[rmw_addr] <= rmw_wr_data;
    end

    // =========================================================================
    // Sequential read port — for Softmax and S×V (registered, 1-cycle latency)
    // rd_en and rmw_wr_en should not target the same address in the same cycle.
    // (Controller FSM guarantees they are in separate phases.)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_data <= '0;
        else if (rd_en)
            rd_data <= mem[rd_addr];
    end

endmodule
