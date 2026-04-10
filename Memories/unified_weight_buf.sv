// =============================================================================
// unified_weight_buf.sv  (rev 7 — all 4 Swin stages supported)
//
// ── What changed from rev 6 ───────────────────────────────────────────────
//   MAX_BYTES: 384 → 3072   (Stage 4 FFN1 col: 768 words × 4 B = 3072 B)
//   msa_load_word: 7 bits → 10 bits  (covers 0..767 for Stage 4)
//   mlp_load_k_word: 7 bits → 10 bits (covers 0..767 for Stage 4 PM W2 col)
//   New valid-byte constants added for all stages.
//   mlp_sub_mode meaning clarified: 0=W1, 1=W2 (W2 is always 4×C input col).
//
// ── Valid-byte table — all stages ─────────────────────────────────────────
//
//   Stage │ C   │ FFN_C │ QKV/Proj col │ QK^T col │ SxV col │ FFN1 col │ FFN2 col
//   ──────┼─────┼───────┼──────────────┼──────────┼─────────┼──────────┼─────────
//     1   │  96 │   384 │   96 B  (2)  │  32 B (1)│ 52 B (2)│  384 B(8)│  96 B(2)
//     2   │ 192 │   768 │  192 B  (4)  │  32 B (1)│ 52 B (2)│  768 B(16)│ 192 B(4)
//     3   │ 384 │  1536 │  384 B  (8)  │  32 B (1)│ 52 B (2)│ 1536 B(32)│ 384 B(8)
//     4   │ 768 │  3072 │  768 B (16)  │  32 B (1)│ 52 B (2)│ 3072 B(64)│ 768 B(16)
//
//   Numbers in parentheses = sub-cycles needed (each sub-cycle = 48 bytes).
//   d_head = 32 is constant across all stages → QK^T col always 32 B, 1 sub-cycle.
//   SxV col always 49 weights = 52 B padded → always 2 sub-cycles.
//
// ── MLP (Patch Merging) valid bytes ───────────────────────────────────────
//   mlp_sub_mode = 1'b0 → W1 col (input = C_in/4 words):
//     PM1=96B, PM2=192B, PM3=384B
//   mlp_sub_mode = 1'b1 → W2 col (input = FFN_out/4 words, but for PM it's
//     the expanded input: PM1=384B, PM2=768B, PM3=1536B)
//   The controller drives the correct mlp_valid_bytes via mlp_col_bytes port.
//
// ── New port: mlp_col_bytes [12:0] ───────────────────────────────────────
//   Because MLP W1 and W2 column widths vary across PM stages, the
//   controller now explicitly provides the valid byte count for the current
//   MLP column.  This replaces the 1-bit mlp_sub_mode (which only handled
//   the fixed Stage-1 W1=96B / W2=384B split).
//   Range: 96..3072, step 48 (always a multiple of N_PE*N_TAP=48).
//   The controller sets this once per op and holds it stable.
//
// ── shadow_clr remains ────────────────────────────────────────────────────
//   Pulse once before each shadow fill to zero stale bytes.
//   Critical for SxV (49→52 B pad) and QK^T (32 B, upper 16 B stale).
//
// ── MAX_BYTES = 3072 ──────────────────────────────────────────────────────
//   Stage 4 FFN1 column: 768 words × 4 B = 3072 B (largest single column).
// =============================================================================

module unified_weight_buf #(
    parameter int MAX_BYTES = 3072,  // Stage 4 FFN1 column — largest operand
    parameter int N_PE      = 12,
    parameter int N_TAP     = 4
)(
    input  logic        clk,
    input  logic        rst_n,

    // 2'b00=CONV, 2'b01=MLP (Patch Merging), 2'b10=SWIN Block
    input  logic [1:0]  mode,

    // Promotes shadow → active
    input  logic        swap,

    // Pulse once before each shadow fill — clears stale bytes
    input  logic        shadow_clr,

    // ── CONV load  (mode 2'b00) ───────────────────────────────────────────
    // conv_load_pe_idx = 0..11; 1 word per PE; no bias
    input  logic        conv_load_en,
    input  logic [3:0]  conv_load_pe_idx,
    input  logic [31:0] conv_load_data,

    // ── MLP load  (mode 2'b01 — Patch Merging) ────────────────────────────
    // mlp_load_k_word  : word index within column (0..767 max for Stage4 W2)
    // mlp_col_bytes    : valid byte count for current column, set by controller
    //                    PM1: W1=96, W2=384   PM2: W1=192, W2=768
    //                    PM3: W1=384, W2=1536
    input  logic        mlp_load_en,
    input  logic [9:0]  mlp_load_k_word,    // 10 bits: 0..767
    input  logic [31:0] mlp_load_data,
    input  logic [11:0] mlp_col_bytes,       // 96..3072 (set stable per col)

    // ── SWIN Block load  (mode 2'b10) ─────────────────────────────────────
    // msa_load_word range per sub-mode:
    //   QKV/Proj/FFN2 : 0..C/4-1      (e.g. 0..23 Stage1, 0..191 Stage4)
    //   QK^T          : 0..7           (d_head=32 / 4 = 8, constant)
    //   SxV           : 0..11          (load only 12 words; byte 48 zero via shadow_clr)
    //   FFN1          : 0..FFN_C/4-1  (0..767 Stage4)
    input  logic        msa_load_en,
    input  logic [9:0]  msa_load_word,       // 10 bits: covers 0..767 (Stage4 FFN1)
    input  logic [31:0] msa_load_data,

    // SWIN sub-mode — selects valid-byte window for read-out
    // 2'b00 = QKV / Proj / FFN2   (C bytes)
    // 2'b01 = QK^T                (32 B — constant, d_head always 32)
    // 2'b10 = SxV                 (52 B — 49 padded to 52, constant)
    // 2'b11 = FFN1                (FFN_C bytes)
    input  logic [1:0]  msa_sub_mode,

    // For SWIN mode, controller provides exact valid byte count per sub-mode
    // to handle all stages without a lookup table in this module:
    //   msa_col_bytes = C   for sub-mode 2'b00 (QKV/Proj/FFN2)
    //                 = 32  for sub-mode 2'b01 (QK^T)
    //                 = 52  for sub-mode 2'b10 (SxV)
    //                 = FFN_C for sub-mode 2'b11 (FFN1)
    input  logic [11:0] msa_col_bytes,        // 32..3072 (set stable per col)

    // Sub-cycle counter — selects 48-byte window into active bank
    input  logic [6:0]  sub_cycle,            // 7 bits: up to 64 sub-cycles (Stage4 FFN1)

    // Weight output to MMU (bias_out removed — dedicated bias_buffer)
    output logic [7:0]  w_out [0:N_PE-1][0:N_TAP-1]
);

// =============================================================================
// Fixed valid-byte constants (only those that don't vary with stage)
// =============================================================================
localparam int BYTES_CONV = N_PE * N_TAP;  // 48 B — 1 sub-cycle, constant
localparam int BYTES_QKT  = 32;            // 32 B — d_head=32, constant all stages
localparam int BYTES_SV   = 52;            // 52 B — 49 padded, constant all stages

// Stage-varying widths come in via mlp_col_bytes / msa_col_bytes at runtime.

// =============================================================================
// Double-banked storage
// =============================================================================
logic [7:0]  bank [0:1][0:MAX_BYTES-1];
logic        active;
logic        shadow;
assign shadow = ~active;

always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) active <= 1'b0;
    else if (swap) active <= shadow;

// =============================================================================
// Shadow bank write  (shadow_clr highest priority)
// =============================================================================
always_ff @(posedge clk) begin
    if (shadow_clr) begin
        for (int i = 0; i < MAX_BYTES; i++)
            bank[shadow][i] <= 8'h00;
    end else begin
        case (mode)

            // ── CONV: 48 bytes, no bias ───────────────────────────────────
            2'b00: begin
                if (conv_load_en) begin
                    bank[shadow][conv_load_pe_idx * N_TAP    ] <= conv_load_data[ 7: 0];
                    bank[shadow][conv_load_pe_idx * N_TAP + 1] <= conv_load_data[15: 8];
                    bank[shadow][conv_load_pe_idx * N_TAP + 2] <= conv_load_data[23:16];
                    bank[shadow][conv_load_pe_idx * N_TAP + 3] <= conv_load_data[31:24];
                end
            end

            // ── MLP (Patch Merging) ───────────────────────────────────────
            2'b01: begin
                if (mlp_load_en) begin
                    bank[shadow][mlp_load_k_word * N_TAP    ] <= mlp_load_data[ 7: 0];
                    bank[shadow][mlp_load_k_word * N_TAP + 1] <= mlp_load_data[15: 8];
                    bank[shadow][mlp_load_k_word * N_TAP + 2] <= mlp_load_data[23:16];
                    bank[shadow][mlp_load_k_word * N_TAP + 3] <= mlp_load_data[31:24];
                end
            end

            // ── SWIN Block (all sub-modes share byte-level write) ─────────
            2'b10: begin
                if (msa_load_en) begin
                    bank[shadow][msa_load_word * N_TAP    ] <= msa_load_data[ 7: 0];
                    bank[shadow][msa_load_word * N_TAP + 1] <= msa_load_data[15: 8];
                    bank[shadow][msa_load_word * N_TAP + 2] <= msa_load_data[23:16];
                    bank[shadow][msa_load_word * N_TAP + 3] <= msa_load_data[31:24];
                end
            end

            default: ;
        endcase
    end
end

// =============================================================================
// Active-bank read-out → w_out[pe][tap]
//
// k = sub_cycle * (N_PE * N_TAP) + pe * N_TAP + tap
// w_out[pe][tap] = (k < valid_bytes) ? bank[active][k] : 8'h00
//
// valid_bytes is provided at runtime by the controller:
//   CONV  → BYTES_CONV (48, fixed)
//   MLP   → mlp_col_bytes (controller-driven, varies per PM stage)
//   SWIN  → msa_col_bytes for sub-modes 00/11;
//            BYTES_QKT(32) for sub-mode 01;
//            BYTES_SV(52)  for sub-mode 10
// =============================================================================
always_comb begin
    for (int pe = 0; pe < N_PE; pe++)
        for (int tap = 0; tap < N_TAP; tap++)
            w_out[pe][tap] = 8'h00;

    case (mode)

        // ── CONV: 48 bytes, 1 sub-cycle ──────────────────────────────────
        2'b00: begin
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP) + pe * N_TAP + tap;
                    w_out[pe][tap] = (k < BYTES_CONV) ? bank[active][k] : 8'h00;
                end
        end

        // ── MLP: column width driven by mlp_col_bytes ────────────────────
        2'b01: begin
            automatic int vb_mlp = int'(mlp_col_bytes);
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP) + pe * N_TAP + tap;
                    w_out[pe][tap] = (k < vb_mlp) ? bank[active][k] : 8'h00;
                end
        end

        // ── SWIN Block ────────────────────────────────────────────────────
        2'b10: begin
            // QK^T and SxV have fixed sizes (d_head=32, seq_len=49 constant).
            // QKV/Proj/FFN1/FFN2 sizes come from msa_col_bytes.
            automatic int vb_msa;
            case (msa_sub_mode)
                2'b00:   vb_msa = int'(msa_col_bytes);  // QKV / Proj / FFN2
                2'b01:   vb_msa = BYTES_QKT;             // 32 B, constant
                2'b10:   vb_msa = BYTES_SV;              // 52 B, constant
                2'b11:   vb_msa = int'(msa_col_bytes);  // FFN1
                default: vb_msa = 0;
            endcase
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP) + pe * N_TAP + tap;
                    w_out[pe][tap] = (k < vb_msa) ? bank[active][k] : 8'h00;
                end
        end

        default: ;
    endcase
end

// =============================================================================
// Simulation assertion — Conv must always use sub_cycle == 0
// =============================================================================
// synthesis translate_off
always_ff @(posedge clk) begin
    if (mode == 2'b00 && sub_cycle != 7'd0)
        $error("[unified_weight_buf] CONV mode: sub_cycle=%0d but must be 0", sub_cycle);
end
// synthesis translate_on

endmodule
