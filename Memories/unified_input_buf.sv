// =============================================================================
// unified_input_buf.sv  (rev 4 — all operation gaps fixed)
//
// ── Summary of fixes from rev 3 ──────────────────────────────────────────
//
//   FIX 1 — MHA/SW-MSA read-out: patch_base missing (CRITICAL)
//     Rev 2/3 hard-wired the MHA read address as:
//       addr = win * K_MHA + k
//     This only ever reads patches 0..6 (the first 7 rows of the bank).
//     Patches 7..48 were loaded but permanently inaccessible from the MMU.
//
//     Fix: new port  mha_patch_base [5:0]
//       Read address becomes:
//         addr = (mha_patch_base + win) * K_MHA + k
//       The controller increments mha_patch_base from 0 to 42 in steps of 7
//       across 7 row-group passes per window, without reloading the bank.
//       This also drives mask_req_patch correctly for SW-MSA (see FIX 3).
//
//   FIX 2 — MLP capture: mlp_col_wr treated as byte address, not word
//     In rev 2/3, the MLP L1 capture wrote:
//       bank[shadow][ r * K_MAX + mlp_col_wr ] <= mlp_l1_out[r][7:0]
//     mlp_col_wr was intended as a BYTE column index (0..383), but it was
//     also used as a BYTE address into the K_MHA dimension for MHA capture,
//     where the range should be 0..95.  This was technically functional but
//     confusingly overloaded.
//
//     Fix: renamed port  cap_col_byte [8:0]  (was mlp_col_wr).
//     Semantics: always a byte offset within one row of the bank.
//       MLP mode:  0..383  (row width = K_MAX = 384)
//       MHA mode:  0..95   (row width = K_MHA = 96)
//     Capture writes one byte per row per call.
//     To fill a full row the controller calls cap_en for each byte position.
//     The MMU produces 32-bit accumulator outputs; only the low 8 bits
//     (INT8 result after quantization) are captured here.  The upstream
//     quantizer/rounding pipeline must produce INT8 before cap_en is asserted.
//
//   FIX 3 — SW-MSA mask outputs: driven from mha_patch_base, not capture row
//     Rev 3 drove mask_req_patch from mha_capture_row (the writeback pointer).
//     During QK^T compute the writeback pointer is idle; the relevant pointer
//     is the READ pointer (which patches are currently presented to the MMU).
//
//     Fix: mask_req_patch and sw_patch_region now derive from mha_patch_base,
//     which tracks the current query patch group during read-out.
//
//   FIX 4 — CONV bank size vs BANK_BYTES
//     CONV uses N_PE*N_WIN*N_TAP = 12*7*4 = 336 bytes.
//     BANK_BYTES = MHA_ROWS * K_MHA = 49*96 = 4704 bytes.
//     336 < 4704 so CONV data always fits. Verified and documented.
//
//   FIX 5 — SW-MSA mode (2'b11) write path unified with W-MSA (2'b10)
//     Both modes use identical bank layout. The write and read always_blocks
//     share the case body via fall-through to avoid duplication.
//
// ── Modes ────────────────────────────────────────────────────────────────
//   2'b00  CONV   : 12 PEs × 7 windows × 4 taps = 336 bytes per bank
//   2'b01  MLP    : 7 rows × K_MAX(384) features = 2688 bytes per bank
//   2'b10  W-MSA  : 49 patches × K_MHA(96) bytes = 4704 bytes per bank
//   2'b11  SW-MSA : same bank layout as W-MSA; shift applied at FIB level
//
// ── Bank layout per mode ──────────────────────────────────────────────────
//   CONV:    bank[pe * N_WIN * N_TAP + win * N_TAP + byte_in_tap]
//   MLP:     bank[row * K_MAX + k_word * N_TAP + byte_lane]
//   MHA/SW:  bank[patch * K_MHA + k_word * N_TAP + byte_lane]
//
// ── BANK_BYTES = 4704  (49 × 96) ─────────────────────────────────────────
//   Largest mode is MHA/SW-MSA: 49 patches × 96 bytes = 4704 bytes.
//   MLP  = 7 × 384 = 2688 < 4704 ✓
//   CONV = 12 × 7 × 4 = 336 < 4704 ✓
// =============================================================================

module unified_input_buf #(
    parameter int N_ROWS   = 7,     // MLP row group size  (7 rows at a time)
    parameter int K_MAX    = 384,   // MLP feature width   (384 bytes = 96 words)
    parameter int N_PE     = 12,    // number of PEs
    parameter int N_WIN    = 7,     // windows per compute burst
    parameter int N_TAP    = 4,     // taps (bytes) per PE per window
    parameter int MHA_ROWS = 49,    // patches per 7×7 window
    parameter int K_MHA    = 96,    // feature bytes per patch
    parameter int SHIFT    = 3      // SW-MSA shift for sub-region tagging
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── Mode ─────────────────────────────────────────────────────────────
    // 2'b00=CONV, 2'b01=MLP, 2'b10=W-MSA, 2'b11=SW-MSA
    input  logic [1:0]  mode,
    input  logic        swap,

    // ═════════════════════════════════════════════════════════════════════
    // CONV load port  (mode == 2'b00)
    // ═════════════════════════════════════════════════════════════════════
    input  logic        conv_load_en,
    input  logic [3:0]  conv_load_pe_idx,    // 0..11
    input  logic [2:0]  conv_load_win_idx,   // 0..6
    input  logic [31:0] conv_load_data,

    // ═════════════════════════════════════════════════════════════════════
    // MLP load port  (mode == 2'b01)
    //   mlp_load_row    : 0..6   (which of the 7 rows in this row-group)
    //   mlp_load_k_word : 0..95  (word index, 4 B per word → 384 B per row)
    // ═════════════════════════════════════════════════════════════════════
    input  logic        mlp_load_en,
    input  logic [2:0]  mlp_load_row,
    input  logic [6:0]  mlp_load_k_word,
    input  logic [31:0] mlp_load_data,

    // ═════════════════════════════════════════════════════════════════════
    // MHA / SW-MSA load port  (mode == 2'b10 or 2'b11)
    //   mha_load_patch  : 0..48  (which patch in the 7×7 window)
    //   mha_load_k_word : 0..23  (word index, 4 B per word → 96 B per patch)
    //   Data comes from FIB Port A (W-MSA) or Port B (SW-MSA) — the buffer
    //   stores whatever arrives; it does not know which port was used.
    // ═════════════════════════════════════════════════════════════════════
    input  logic        mha_load_en,
    input  logic [5:0]  mha_load_patch,     // 0..48
    input  logic [4:0]  mha_load_k_word,    // 0..23
    input  logic [31:0] mha_load_data,

    // ═════════════════════════════════════════════════════════════════════
    // Intermediate result capture  (all non-CONV modes)
    //
    //   cap_en       : strobe — write one byte per row this cycle
    //   cap_col_byte : byte address within a row
    //                  MLP mode  : 0..383  (K_MAX=384 bytes wide)
    //                  MHA mode  : 0..95   (K_MHA=96  bytes wide)
    //   cap_data[r]  : 32-bit MMU accumulator output for row r
    //                  Only cap_data[r][7:0] is stored (INT8 result after
    //                  upstream quantization).
    //
    //   MLP usage:
    //     After each L1 compute column, 7 output values (one per row) are
    //     written into the shadow bank.  The controller loops cap_col_byte
    //     over 0..383 to fill each row byte-by-byte across columns.
    //
    //   MHA usage:
    //     After each Q/K/V/SxV column, 7 patch-row output bytes are written
    //     back so subsequent steps (e.g. Q written, then read for QK^T) can
    //     reuse them from the bank without going off-chip.
    //     cap_col_base selects which patch group is being written (see below).
    // ═════════════════════════════════════════════════════════════════════
    input  logic        cap_en,
    input  logic [8:0]  cap_col_byte,       // byte column 0..383 (MLP) / 0..95 (MHA)
    input  logic [31:0] cap_data [0:N_ROWS-1],  // 7 × 32-bit from MMU

    // MHA only: which group of 7 patches is being written back
    // (same signal as mha_patch_base below, held during writeback phase)
    input  logic [5:0]  mha_cap_patch_base, // 0, 7, 14, 21, 28, 35, 42

    // ═════════════════════════════════════════════════════════════════════
    // MHA / SW-MSA patch-group base  (FIX 1)
    //   Selects which group of 7 patches to present to the MMU each burst.
    //   Valid values: 0, 7, 14, 21, 28, 35, 42  (7 passes over 49 patches).
    //   Controller increments this by 7 after each compute burst of 7 rows.
    //   The active bank is NOT reloaded between passes — all 49 patches are
    //   already in the bank; only the read pointer advances.
    // ═════════════════════════════════════════════════════════════════════
    input  logic [5:0]  mha_patch_base,     // 0..42 in steps of 7

    // ── Sub-cycle counter ─────────────────────────────────────────────────
    input  logic [2:0]  sub_cycle,

    // ── Data output to MMU ────────────────────────────────────────────────
    output logic [7:0]  data_out [0:N_PE-1][0:N_WIN-1][0:N_TAP-1],

    // ═════════════════════════════════════════════════════════════════════
    // SW-MSA mask support outputs  (valid only when mode == 2'b11)
    //
    //   mask_req_patch  : patch index (0..48) of the current query group's
    //                     first row (win=0).  Derived from mha_patch_base.
    //                     The mask buffer uses this to look up the correct
    //                     per-query-row mask vector.
    //
    //   sw_patch_region : 2-bit sub-region code of the current query patch.
    //                     region[1] = patch_row_in_window >= SHIFT  (>=3)
    //                     region[0] = patch_col_in_window >= SHIFT  (>=3)
    //                     where patch_row_in_window = mha_patch_base / 7
    //                           patch_col_in_window = mha_patch_base % 7
    //                     The mask buffer compares this with each key patch's
    //                     region to decide mask(q,k): 0 if same, -INF if different.
    //                     Zero when mode != 2'b11.
    // ═════════════════════════════════════════════════════════════════════
    output logic [5:0]  mask_req_patch,
    output logic [1:0]  sw_patch_region
);

    // ── Bank sizing ───────────────────────────────────────────────────────
    // MHA/SW-MSA is the largest: 49 × 96 = 4704 bytes.
    // MLP = 7 × 384 = 2688 < 4704 ✓   CONV = 336 < 4704 ✓
    localparam int BANK_BYTES = MHA_ROWS * K_MHA;  // 4704

    logic [7:0] bank [0:1][0:BANK_BYTES-1];
    logic       active;
    logic       shadow;
    assign shadow = ~active;

    // ── Bank swap ─────────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) active <= 1'b0;
        else if (swap) active <= shadow;

    // =========================================================================
    // Write / capture logic
    // =========================================================================
    always_ff @(posedge clk) begin
        case (mode)

            // ── CONV ──────────────────────────────────────────────────────
            2'b00: begin
                if (conv_load_en) begin
                    automatic int base = int'(conv_load_pe_idx) * (N_WIN * N_TAP)
                                       + int'(conv_load_win_idx) * N_TAP;
                    bank[shadow][base    ] <= conv_load_data[ 7: 0];
                    bank[shadow][base + 1] <= conv_load_data[15: 8];
                    bank[shadow][base + 2] <= conv_load_data[23:16];
                    bank[shadow][base + 3] <= conv_load_data[31:24];
                end
            end

            // ── MLP ───────────────────────────────────────────────────────
            2'b01: begin
                // Load input X rows into shadow bank
                if (mlp_load_en) begin
                    automatic int base = int'(mlp_load_row)    * K_MAX
                                       + int'(mlp_load_k_word) * N_TAP;
                    bank[shadow][base    ] <= mlp_load_data[ 7: 0];
                    bank[shadow][base + 1] <= mlp_load_data[15: 8];
                    bank[shadow][base + 2] <= mlp_load_data[23:16];
                    bank[shadow][base + 3] <= mlp_load_data[31:24];
                end
                // L1 Y capture: one byte per row per call
                // cap_col_byte is a byte offset within the K_MAX-wide row.
                if (cap_en) begin
                    for (int r = 0; r < N_ROWS; r++) begin
                        automatic int addr = r * K_MAX + int'(cap_col_byte);
                        bank[shadow][addr] <= cap_data[r][7:0];
                    end
                end
            end

            // ── W-MSA and SW-MSA — identical bank layout ───────────────────
            // The cyclic shift for SW-MSA is resolved at the FIB read address
            // level; data arrives here in logical window order either way.
            2'b10, 2'b11: begin
                // Load patch-feature data (from FIB port A or B)
                if (mha_load_en) begin
                    automatic int base = int'(mha_load_patch) * K_MHA
                                       + int'(mha_load_k_word) * N_TAP;
                    bank[shadow][base    ] <= mha_load_data[ 7: 0];
                    bank[shadow][base + 1] <= mha_load_data[15: 8];
                    bank[shadow][base + 2] <= mha_load_data[23:16];
                    bank[shadow][base + 3] <= mha_load_data[31:24];
                end
                // Intermediate Q/K/V/SxV writeback:
                // Write one byte per patch-row per call.
                // mha_cap_patch_base selects the group of 7 patches.
                // cap_col_byte is a byte offset within K_MHA (0..95).
                if (cap_en) begin
                    for (int r = 0; r < N_ROWS; r++) begin
                        automatic int patch = int'(mha_cap_patch_base) + r;
                        automatic int addr  = patch * K_MHA + int'(cap_col_byte);
                        if (patch < MHA_ROWS)
                            bank[shadow][addr] <= cap_data[r][7:0];
                    end
                end
            end

            default: ;
        endcase
    end

    // =========================================================================
    // Read logic (combinatorial)  →  data_out[pe][win][tap]
    //
    // Byte address formula per mode:
    //   CONV:   bank[ pe*(N_WIN*N_TAP) + win*N_TAP + tap ]
    //   MLP:    bank[ win*K_MAX + k ]                   k = sub_cycle*48+pe*4+tap
    //   MHA/SW: bank[ (mha_patch_base+win)*K_MHA + k ]  k = sub_cycle*48+pe*4+tap
    //
    // The MHA formula is the key fix: mha_patch_base offsets into the 49-row
    // bank so all 7 patch groups (0-6, 7-13, ..., 42-48) are reachable
    // without reloading the bank between row-group passes.
    // =========================================================================
    always_comb begin
        for (int pe = 0; pe < N_PE; pe++)
            for (int win = 0; win < N_WIN; win++)
                for (int tap = 0; tap < N_TAP; tap++)
                    data_out[pe][win][tap] = 8'h00;

        case (mode)

            // ── CONV ──────────────────────────────────────────────────────
            2'b00: begin
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++)
                            data_out[pe][win][tap] =
                                bank[active][ pe * (N_WIN * N_TAP)
                                            + win * N_TAP
                                            + tap ];
            end

            // ── MLP ───────────────────────────────────────────────────────
            // win maps to one of the 7 rows in the current row-group.
            // sub_cycle slides a 48-byte window over the K_MAX-byte row.
            2'b01: begin
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k    = int'(sub_cycle) * (N_PE * N_TAP)
                                              + pe * N_TAP + tap;
                            automatic int addr = win * K_MAX + k;
                            data_out[pe][win][tap] =
                                (k < K_MAX) ? bank[active][addr] : 8'h00;
                        end
            end

            // ── W-MSA ─────────────────────────────────────────────────────
            // mha_patch_base selects which group of 7 patches to present.
            // sub_cycle slides over the 96-byte K_MHA feature vector.
            2'b10: begin
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k     = int'(sub_cycle) * (N_PE * N_TAP)
                                               + pe * N_TAP + tap;
                            automatic int patch = int'(mha_patch_base) + win;
                            automatic int addr  = patch * K_MHA + k;
                            data_out[pe][win][tap] =
                                (k < K_MHA && patch < MHA_ROWS)
                                ? bank[active][addr] : 8'h00;
                        end
            end

            // ── SW-MSA — identical read-out to W-MSA ──────────────────────
            // Mask outputs are driven separately below.
            2'b11: begin
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k     = int'(sub_cycle) * (N_PE * N_TAP)
                                               + pe * N_TAP + tap;
                            automatic int patch = int'(mha_patch_base) + win;
                            automatic int addr  = patch * K_MHA + k;
                            data_out[pe][win][tap] =
                                (k < K_MHA && patch < MHA_ROWS)
                                ? bank[active][addr] : 8'h00;
                        end
            end

            default: ;
        endcase
    end

    // =========================================================================
    // SW-MSA mask support outputs  (mode == 2'b11 only)
    //
    // mask_req_patch:
    //   The patch index of the current query group's first row.
    //   = mha_patch_base  (the controller holds this stable during each
    //     group of 7 compute cycles so the mask buffer sees a stable index).
    //
    // sw_patch_region:
    //   2-bit code derived from mha_patch_base.
    //   Within a 7×7 window: row_in_win = patch / 7, col_in_win = patch % 7.
    //   region[1] = (row_in_win >= SHIFT)  i.e. >= 3
    //   region[0] = (col_in_win >= SHIFT)  i.e. >= 3
    //
    //   Patches within the same region can attend to each other.
    //   Cross-region attention is masked to -INF by the mask buffer.
    //
    //   Both outputs are 0 when mode != 2'b11.
    // =========================================================================
    always_comb begin
        mask_req_patch  = 6'h00;
        sw_patch_region = 2'b00;

        if (mode == 2'b11) begin
            mask_req_patch = mha_patch_base;

            automatic int row_in_win = int'(mha_patch_base) / N_WIN;  // /7
            automatic int col_in_win = int'(mha_patch_base) % N_WIN;  // %7
            sw_patch_region[1] = (row_in_win >= SHIFT) ? 1'b1 : 1'b0;
            sw_patch_region[0] = (col_in_win >= SHIFT) ? 1'b1 : 1'b0;
        end
    end

endmodule