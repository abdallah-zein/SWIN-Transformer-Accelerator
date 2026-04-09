// =============================================================================
// unified_input_buf.sv  (rev 2 — MHA / Swin Transformer Block support)
//
// ── Modes ────────────────────────────────────────────────────────────────
//   mode = 2'b00  (CONV)  : Patch Embedding convolutional path
//                           12 PEs × 7 windows × 4 taps
//   mode = 2'b01  (MLP)   : Patch Merging / FFN projection path
//                           7 rows × K_MAX=384 features, double-buffered X
//   mode = 2'b10  (MHA)   : Swin Transformer Block attention path
//                           Input is a flattened 7×7 window = 49 patches
//                           each with C=96 channels; stored as 49 rows ×
//                           96 features, 4 bytes/word → 24 words/row.
//
//                           Layout in shadow bank (MHA):
//                             patch 0..48, each patch occupies K_MHA=96 bytes
//                             bank[shadow][p * K_MHA + k]  (byte addressing)
//
//                           Read-out (MHA, to MMU):
//                             N_PE=12 PEs, each PE handles 4 consecutive k
//                             N_WIN=7 windows correspond to 7 consecutive
//                             patches in the window row
//                             sub_cycle selects the group-of-4-k chunk:
//                               k = sub_cycle*(N_PE*N_TAP) + pe*N_TAP + tap
//                             patch (row) index is addressed via mlp_load_row
//                             (reused port) which now indexes 0..48.
//
//                           K_MHA = 96 (< K_MAX=384 → shadow bank fits fine).
//
// ── MHA capture path ─────────────────────────────────────────────────────
//   After each Q/K/V projection column the MMU output (7 values) is
//   written back into the shadow bank at column mha_col_wr so that the
//   next sub-step can reuse it as its input (e.g. Q written, then read
//   back for QKᵀ).  The mha_capture_en + mha_col_wr ports mirror the
//   existing mlp_capture_en / mlp_col_wr ports in semantics.
//
// ── Buffer sizing ────────────────────────────────────────────────────────
//   BANK_BYTES = N_ROWS * K_MAX = 7 * 384 = 2688 bytes
//   MHA needs  49 patches × 96 bytes = 4704 bytes   ← LARGER
//   → N_ROWS_MHA * K_MHA = 49 * 96 = 4704
//   We keep BANK_BYTES = N_ROWS_MHA * K_MHA so that one bank fits all
//   MHA patch data for a single window.
//   New BANK_BYTES = max(2688, 4704) = 4704
//   Parameterised as MHA_ROWS=49, K_MHA=96.
// =============================================================================

module unified_input_buf #(
    parameter N_ROWS   = 7,
    parameter K_MAX    = 384,
    parameter N_PE     = 12,
    parameter N_WIN    = 7,
    parameter N_TAP    = 4,
    // MHA-specific dimensions (a single 7×7 window = 49 patches × 96 features)
    parameter MHA_ROWS = 49,
    parameter K_MHA    = 96
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic [1:0]  mode,   // 2'b00=CONV, 2'b01=MLP, 2'b10=MHA
    input  logic        swap,

    // ── CONV load port ────────────────────────────────────────────────────
    input  logic        conv_load_en,
    input  logic [3:0]  conv_load_pe_idx,
    input  logic [2:0]  conv_load_win_idx,
    input  logic [31:0] conv_load_data,

    // ── MLP load port ─────────────────────────────────────────────────────
    input  logic        mlp_load_en,
    input  logic [2:0]  mlp_load_row,       // 0..6  (7 rows per row-group)
    input  logic [6:0]  mlp_load_k_word,    // 0..95 (words, 4 B each → 384 B)
    input  logic [31:0] mlp_load_data,

    // ── MLP L1 Y capture (reused by MHA for intermediate result capture) ──
    input  logic        mlp_capture_en,     // also mha_capture_en
    input  logic [8:0]  mlp_col_wr,         // also mha_col_wr (0..95 for MHA)
    input  logic [31:0] mlp_l1_out [0:N_ROWS-1],  // 7 values from MMU

    // ── MHA load port ─────────────────────────────────────────────────────
    // Loads a full 49-row × 96-feature (= K_MHA) patch-feature block
    // into the shadow bank for one window.
    input  logic        mha_load_en,
    input  logic [5:0]  mha_load_patch,    // 0..48 (which patch/row)
    input  logic [4:0]  mha_load_k_word,  // 0..23 (word index, 4 B each → 96 B)
    input  logic [31:0] mha_load_data,

    // ── MHA intermediate capture ──────────────────────────────────────────
    // Stores MMU output rows (7 values) into the MHA shadow bank.
    // Used to write Q, K, V or attention outputs back for subsequent steps.
    // Shares mlp_capture_en / mlp_col_wr signals (both are semantically
    // "output column write-back").
    // mha_capture_row selects which patch row is being written back.
    input  logic [5:0]  mha_capture_row,   // 0..48

    // ── Sub-cycle (shared across modes) ───────────────────────────────────
    input  logic [2:0]  sub_cycle,

    // ── Data output to MMU ────────────────────────────────────────────────
    output logic [7:0]  data_out [0:N_PE-1][0:N_WIN-1][0:N_TAP-1]
);

    // ── Bank sizing ───────────────────────────────────────────────────────
    // Max of: CONV=2688, MLP=2688, MHA=4704 → 4704 bytes
    localparam BANK_BYTES = MHA_ROWS * K_MHA;  // 49*96 = 4704

    logic [7:0] bank [0:1][0:BANK_BYTES-1];
    logic       active;
    logic       shadow;
    assign shadow = ~active;

    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) active <= 1'b0;
        else if (swap) active <= shadow;

    // ── Write logic ───────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        case (mode)

            // ── CONV: 12 PE × 7 win × 4 tap ─────────────────────────────
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

            // ── MLP: 7 rows × 384 K ───────────────────────────────────────
            2'b01: begin
                if (mlp_load_en) begin
                    // NOTE: MLP bank uses layout [row * K_MAX + k_word*4 + byte]
                    // This fits in BANK_BYTES since N_ROWS*K_MAX = 7*384 = 2688 < 4704
                    automatic int base = int'(mlp_load_row) * K_MAX
                                       + int'(mlp_load_k_word) * N_TAP;
                    bank[shadow][base    ] <= mlp_load_data[ 7: 0];
                    bank[shadow][base + 1] <= mlp_load_data[15: 8];
                    bank[shadow][base + 2] <= mlp_load_data[23:16];
                    bank[shadow][base + 3] <= mlp_load_data[31:24];
                end
                if (mlp_capture_en) begin
                    for (int r = 0; r < N_ROWS; r++) begin
                        automatic int addr = r * K_MAX + int'(mlp_col_wr);
                        bank[shadow][addr] <= mlp_l1_out[r][7:0];
                    end
                end
            end

            // ── MHA: 49 patches × 96 features ────────────────────────────
            2'b10: begin
                // Load from FIB or ILB into shadow bank
                if (mha_load_en) begin
                    automatic int base = int'(mha_load_patch) * K_MHA
                                       + int'(mha_load_k_word) * N_TAP;
                    bank[shadow][base    ] <= mha_load_data[ 7: 0];
                    bank[shadow][base + 1] <= mha_load_data[15: 8];
                    bank[shadow][base + 2] <= mha_load_data[23:16];
                    bank[shadow][base + 3] <= mha_load_data[31:24];
                end
                // Write-back of MMU output rows (Q/K/V/attention rows)
                // mlp_capture_en is reused; mha_capture_row selects the patch
                if (mlp_capture_en) begin
                    for (int r = 0; r < N_ROWS; r++) begin
                        // Write 7 consecutive patch rows starting at mha_capture_row
                        automatic int patch = int'(mha_capture_row) + r;
                        automatic int addr  = patch * K_MHA + int'(mlp_col_wr);
                        if (patch < MHA_ROWS)
                            bank[shadow][addr] <= mlp_l1_out[r][7:0];
                    end
                end
            end

            default: ;
        endcase
    end

    // ── Read logic (combinatorial) ────────────────────────────────────────
    always_comb begin
        // Default: drive zeros
        for (int pe = 0; pe < N_PE; pe++)
            for (int win = 0; win < N_WIN; win++)
                for (int tap = 0; tap < N_TAP; tap++)
                    data_out[pe][win][tap] = 8'b0;

        case (mode)

            // ── CONV ─────────────────────────────────────────────────────
            2'b00: begin
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++)
                            data_out[pe][win][tap] =
                                bank[active][pe * (N_WIN * N_TAP) + win * N_TAP + tap];
            end

            // ── MLP ──────────────────────────────────────────────────────
            2'b01: begin
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k    = int'(sub_cycle) * (N_PE * N_TAP)
                                              + pe * N_TAP + tap;
                            automatic int addr = win * K_MAX + k;
                            data_out[pe][win][tap] =
                                (k < K_MAX) ? bank[active][addr] : 8'b0;
                        end
            end

            // ── MHA ──────────────────────────────────────────────────────
            // N_WIN=7 maps to 7 consecutive patches in a window row.
            // N_PE=12 PEs cover 12 × 4 = 48 k-features per sub_cycle group.
            // sub_cycle selects which group of 48 k-features:
            //   k_base = sub_cycle * (N_PE * N_TAP) = sub_cycle * 48
            // patch index = win (0..6 within this 7-row read burst).
            // For a 49-row output, the controller streams 7 rows at a time
            // (win dimension) with the patch_base counter in the controller.
            //
            // The active bank layout for MHA: bank[active][patch * K_MHA + k]
            // where patch = win_base + win  (win_base driven by controller
            // via the sub_cycle / patch-group mechanism).
            2'b10: begin
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            // k index within 96-feature vector
                            automatic int k    = int'(sub_cycle) * (N_PE * N_TAP)
                                              + pe * N_TAP + tap;
                            // patch index: win maps directly to patch rows
                            // (the controller ensures only 7 patches at a time
                            //  are presented; it reloads bank for each group of 7)
                            automatic int addr = win * K_MHA + k;
                            data_out[pe][win][tap] =
                                (k < K_MHA) ? bank[active][addr] : 8'b0;
                        end
            end

            default: ;
        endcase
    end

endmodule
