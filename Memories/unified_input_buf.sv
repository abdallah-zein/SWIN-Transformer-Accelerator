// =============================================================================
// unified_input_buf.sv  (rev 5 — all 4 Swin stages supported)
//
// ── What changed from rev 4 ───────────────────────────────────────────────
//   K_MAX:      384  → 3072  (Stage 4 FFN1 input row: 768 words × 4 B)
//   K_MHA:       96  →  768  (Stage 4 C = 768 bytes per patch)
//   BANK_BYTES: 4704 → 37632 (49 patches × 768 bytes, dominant over MLP 7×3072=21504)
//   mlp_load_k_word: 7 bits → 10 bits  (covers 0..767 for Stage4)
//   mha_load_k_word: 5 bits → 8 bits   (covers 0..191 for Stage4: 768/4=192 words)
//   cap_col_byte: 9 bits → 12 bits      (covers 0..3071 for Stage4 MLP)
//   mha_cap_patch_base / mha_patch_base: unchanged (49 patches, 6 bits)
//
// ── Bank sizing ───────────────────────────────────────────────────────────
//   MLP  max: 7 rows × 3072 bytes/row = 21,504 bytes
//   MHA  max: 49 patches × 768 bytes  = 37,632 bytes
//   CONV max: 12 × 7 × 4 = 336 bytes
//   BANK_BYTES = max(21504, 37632) = 37,632 bytes
//
// ── Modes (unchanged) ─────────────────────────────────────────────────────
//   2'b00  CONV   — Patch Embedding
//   2'b01  MLP    — Patch Merging (any PM stage)
//   2'b10  W-MSA  — all Swin stages
//   2'b11  SW-MSA — all Swin stages
//
// ── K_MHA runtime parameterisation ───────────────────────────────────────
//   Because the patch feature width varies per Swin stage (96/192/384/768),
//   the controller provides mha_k_bytes at runtime (set once per round):
//     Stage1=96, Stage2=192, Stage3=384, Stage4=768
//   The read-out valid guard uses mha_k_bytes instead of K_MHA.
//   Similarly, mlp_k_bytes is provided for MLP rows.
// =============================================================================

module unified_input_buf #(
    parameter int N_ROWS   = 7,
    parameter int K_MAX    = 3072,  // Stage 4 FFN1 row width in bytes
    parameter int N_PE     = 12,
    parameter int N_WIN    = 7,
    parameter int N_TAP    = 4,
    parameter int MHA_ROWS = 49,    // always 49 (7×7 window, all stages)
    parameter int K_MHA    = 768,   // Stage 4 max patch feature bytes
    parameter int SHIFT    = 3
)(
    input  logic        clk,
    input  logic        rst_n,

    // 2'b00=CONV, 2'b01=MLP, 2'b10=W-MSA, 2'b11=SW-MSA
    input  logic [1:0]  mode,
    input  logic        swap,

    // ── Runtime row-width signals (set once per round, stable) ────────────
    // mlp_k_bytes: valid byte count per MLP input row (96..3072)
    //   PM1=96, PM2=192, PM3=384, used also for FFN within Swin block
    // mha_k_bytes: valid byte count per patch (96..768)
    //   Stage1=96, Stage2=192, Stage3=384, Stage4=768
    input  logic [11:0] mlp_k_bytes,    // 96..3072
    input  logic [9:0]  mha_k_bytes,    // 96..768

    // ═════════════════════════════════════════════════════════════════════
    // CONV load port  (mode == 2'b00)
    // ═════════════════════════════════════════════════════════════════════
    input  logic        conv_load_en,
    input  logic [3:0]  conv_load_pe_idx,
    input  logic [2:0]  conv_load_win_idx,
    input  logic [31:0] conv_load_data,

    // ═════════════════════════════════════════════════════════════════════
    // MLP load port  (mode == 2'b01)
    //   mlp_load_k_word : 0..767 (Stage4 768/4=192 words per row,
    //                             but load_k_word indexes bytes/4)
    // ═════════════════════════════════════════════════════════════════════
    input  logic        mlp_load_en,
    input  logic [2:0]  mlp_load_row,
    input  logic [9:0]  mlp_load_k_word,   // 10 bits: 0..767
    input  logic [31:0] mlp_load_data,

    // ═════════════════════════════════════════════════════════════════════
    // MHA / SW-MSA load port  (mode == 2'b10 or 2'b11)
    //   mha_load_k_word : 0..191 (Stage4: 768/4=192 words)
    // ═════════════════════════════════════════════════════════════════════
    input  logic        mha_load_en,
    input  logic [5:0]  mha_load_patch,
    input  logic [7:0]  mha_load_k_word,   // 8 bits: 0..191
    input  logic [31:0] mha_load_data,

    // ═════════════════════════════════════════════════════════════════════
    // Capture port  (all non-CONV modes)
    //   cap_col_byte : byte offset within a row
    //     MLP: 0..K_MAX-1 (0..3071)  MHA: 0..K_MHA-1 (0..767)
    // ═════════════════════════════════════════════════════════════════════
    input  logic        cap_en,
    input  logic [11:0] cap_col_byte,          // 12 bits: 0..3071
    input  logic [31:0] cap_data [0:N_ROWS-1],

    input  logic [5:0]  mha_cap_patch_base,    // 0,7,...,42

    // ═════════════════════════════════════════════════════════════════════
    // MHA / SW-MSA patch-group read base
    // ═════════════════════════════════════════════════════════════════════
    input  logic [5:0]  mha_patch_base,        // 0,7,...,42

    // Sub-cycle counter
    input  logic [2:0]  sub_cycle,

    // Data output to MMU
    output logic [7:0]  data_out [0:N_PE-1][0:N_WIN-1][0:N_TAP-1],

    // SW-MSA mask outputs
    output logic [5:0]  mask_req_patch,
    output logic [1:0]  sw_patch_region
);

    // ── Bank sizing ───────────────────────────────────────────────────────
    // MHA dominant: 49 × 768 = 37,632 bytes
    localparam int BANK_BYTES = MHA_ROWS * K_MHA;  // 37,632

    logic [7:0] bank [0:1][0:BANK_BYTES-1];
    logic       active;
    logic       shadow;
    assign shadow = ~active;

    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) active <= 1'b0;
        else if (swap) active <= shadow;

    // =========================================================================
    // Write logic
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

            // ── MLP: row × mlp_k_bytes layout ────────────────────────────
            // bank addr = row * K_MAX + k_word * N_TAP
            // K_MAX is the compile-time max; actual valid range is mlp_k_bytes
            2'b01: begin
                if (mlp_load_en) begin
                    automatic int base = int'(mlp_load_row)    * K_MAX
                                       + int'(mlp_load_k_word) * N_TAP;
                    bank[shadow][base    ] <= mlp_load_data[ 7: 0];
                    bank[shadow][base + 1] <= mlp_load_data[15: 8];
                    bank[shadow][base + 2] <= mlp_load_data[23:16];
                    bank[shadow][base + 3] <= mlp_load_data[31:24];
                end
                if (cap_en) begin
                    for (int r = 0; r < N_ROWS; r++) begin
                        automatic int addr = r * K_MAX + int'(cap_col_byte);
                        bank[shadow][addr] <= cap_data[r][7:0];
                    end
                end
            end

            // ── W-MSA and SW-MSA: patch × mha_k_bytes layout ─────────────
            // bank addr = patch * K_MHA + k_word * N_TAP
            // K_MHA is compile-time max; actual valid range is mha_k_bytes
            2'b10, 2'b11: begin
                if (mha_load_en) begin
                    automatic int base = int'(mha_load_patch) * K_MHA
                                       + int'(mha_load_k_word) * N_TAP;
                    bank[shadow][base    ] <= mha_load_data[ 7: 0];
                    bank[shadow][base + 1] <= mha_load_data[15: 8];
                    bank[shadow][base + 2] <= mha_load_data[23:16];
                    bank[shadow][base + 3] <= mha_load_data[31:24];
                end
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
    // Read logic (combinatorial)
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
                                bank[active][ pe*(N_WIN*N_TAP) + win*N_TAP + tap ];
            end

            // ── MLP ───────────────────────────────────────────────────────
            // Valid guard: k < mlp_k_bytes (runtime row width)
            2'b01: begin
                automatic int vk_mlp = int'(mlp_k_bytes);
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k    = int'(sub_cycle)*(N_PE*N_TAP) + pe*N_TAP + tap;
                            automatic int addr = win * K_MAX + k;
                            data_out[pe][win][tap] = (k < vk_mlp) ? bank[active][addr] : 8'h00;
                        end
            end

            // ── W-MSA ─────────────────────────────────────────────────────
            // mha_patch_base offsets into the 49-row bank.
            // Valid guard: k < mha_k_bytes (runtime patch feature width)
            2'b10: begin
                automatic int vk_mha = int'(mha_k_bytes);
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k     = int'(sub_cycle)*(N_PE*N_TAP) + pe*N_TAP + tap;
                            automatic int patch = int'(mha_patch_base) + win;
                            automatic int addr  = patch * K_MHA + k;
                            data_out[pe][win][tap] =
                                (k < vk_mha && patch < MHA_ROWS)
                                ? bank[active][addr] : 8'h00;
                        end
            end

            // ── SW-MSA — identical read-out to W-MSA ──────────────────────
            2'b11: begin
                automatic int vk_sw = int'(mha_k_bytes);
                for (int pe = 0; pe < N_PE; pe++)
                    for (int win = 0; win < N_WIN; win++)
                        for (int tap = 0; tap < N_TAP; tap++) begin
                            automatic int k     = int'(sub_cycle)*(N_PE*N_TAP) + pe*N_TAP + tap;
                            automatic int patch = int'(mha_patch_base) + win;
                            automatic int addr  = patch * K_MHA + k;
                            data_out[pe][win][tap] =
                                (k < vk_sw && patch < MHA_ROWS)
                                ? bank[active][addr] : 8'h00;
                        end
            end

            default: ;
        endcase
    end

    // =========================================================================
    // SW-MSA mask outputs (mode 2'b11 only)
    // =========================================================================
    always_comb begin
        mask_req_patch  = 6'h00;
        sw_patch_region = 2'b00;
        if (mode == 2'b11) begin
            mask_req_patch = mha_patch_base;
            automatic int row_in_win = int'(mha_patch_base) / N_WIN;
            automatic int col_in_win = int'(mha_patch_base) % N_WIN;
            sw_patch_region[1] = (row_in_win >= SHIFT) ? 1'b1 : 1'b0;
            sw_patch_region[0] = (col_in_win >= SHIFT) ? 1'b1 : 1'b0;
        end
    end

endmodule
