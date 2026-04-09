// =============================================================================
// unified_weight_buf.sv  (rev 3 — FFN sub-mode added to SWIN Block round)
//
// ── Round boundary change ─────────────────────────────────────────────────
//   The SWIN Transformer Block is ONE round (from paper).  MSA and FFN are
//   executed back-to-back before MWU writes to off-chip memory.  Therefore
//   the weight buffer must serve BOTH the MSA and FFN weight columns within
//   mode 2'b10, distinguished by msa_sub_mode:
//
//   msa_sub_mode encoding (mode == 2'b10):
//     2'b00  QKV / Proj  : 96-weight column  (24 words, 2 sub-cycles)
//     2'b01  QK^T        : 32-weight column  ( 8 words, 1 sub-cycle)
//     2'b10  SxV         : 49-weight column  (13 words, 2 sub-cycles)
//     2'b11  FFN         : up to 384-weight col  (96 words, 8 sub-cycles)
//                          Used for W_FFN1 (96 w/col = 384÷4)
//                          AND W_FFN2 (24 w/col = 96÷4, sub-cycles 0..5 only)
//                          Controller sets correct word count via msa_sub_mode
//                          and drives sub_cycle 0..N-1 accordingly.
//
// ── MAX_BYTES change ──────────────────────────────────────────────────────
//   Old MAX_BYTES = 384  (MLP W2: 96 words × 4 = 384 B)
//   New MAX_BYTES = 384  (FFN W1 col: 96 words × 4 = 384 B) — same size!
//   No change needed; 384 B already covers the largest column (W_FFN1).
//
// ── Sub-cycle counts per sub-mode ────────────────────────────────────────
//   2'b00  QKV/Proj  : 96  B → 2 sub-cycles  (2 × 48 B = 96 B)
//   2'b01  QK^T      : 32  B → 1 sub-cycle   (1 × 48 B, 32 used)
//   2'b10  SxV       : 52  B → 2 sub-cycles  (1st=48B full, 2nd=4B partial)
//   2'b11  FFN_W1    : 384 B → 8 sub-cycles  (8 × 48 B = 384 B)
//           FFN_W2   :  96 B → 2 sub-cycles  (same as QKV/Proj)
//   The controller drives the correct number of sub-cycles; the buffer
//   zero-fills any PE taps beyond the valid byte range automatically.
//
// ── All other behaviour unchanged ────────────────────────────────────────
// =============================================================================

module unified_weight_buf #(
    parameter int MAX_BYTES = 384,   // largest column: FFN_W1 = 96 words × 4 B
    parameter int N_PE      = 12,
    parameter int N_TAP     = 4
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── Mode select ───────────────────────────────────────────────────────
    // 2'b00 = Conv (Patch Embedding / Patch Merging conv kernel)
    // 2'b01 = MLP  (Patch Merging linear — legacy, kept for compatibility)
    // 2'b10 = SWIN Block (W-MSA QKV/QK^T/SxV + FFN W1/W2, single round)
    input  logic [1:0]  mode,

    // ── Bank swap (from controller) ───────────────────────────────────────
    input  logic        swap,

    // ── Conv load (mode == 2'b00) ─────────────────────────────────────────
    input  logic        conv_load_en,
    input  logic [3:0]  conv_load_pe_idx,
    input  logic [31:0] conv_load_data,

    input  logic        conv_bias_load_en,
    input  logic [31:0] conv_bias_load_data,

    // ── MLP load (mode == 2'b01, Patch Merging legacy path) ──────────────
    input  logic        mlp_load_en,
    input  logic [6:0]  mlp_load_k_word,   // 0..95 (W2 col), 0..23 (W1 col)
    input  logic [31:0] mlp_load_data,

    // ── SWIN Block load (mode == 2'b10 — MSA and FFN share this path) ─────
    // msa_load_word range:
    //   QKV/Proj  : 0..23  (24 words per column)
    //   QK^T      : 0..7   ( 8 words per column)
    //   SxV       : 0..12  (13 words, last zero-padded)
    //   FFN W1    : 0..95  (96 words per column)
    //   FFN W2    : 0..23  (24 words per column, same width as QKV)
    input  logic        msa_load_en,
    input  logic [6:0]  msa_load_word,     // expanded to 7 bits (was 5) for FFN W1 (0..95)
    input  logic [31:0] msa_load_data,

    // ── Sub-mode (MSA phase or FFN phase within SWIN Block round) ─────────
    // 2'b00 = QKV/Proj   (96 B, 2 sub-cycles)
    // 2'b01 = QK^T       (32 B, 1 sub-cycle)
    // 2'b10 = SxV        (52 B, 2 sub-cycles)
    // 2'b11 = FFN        (up to 384 B, up to 8 sub-cycles for W1,
    //                     or 96 B, 2 sub-cycles for W2 — controller controls)
    input  logic [1:0]  msa_sub_mode,

    // ── Shared sub-cycle counter (from controller) ────────────────────────
    input  logic [2:0]  sub_cycle,

    // ── Outputs to MMU ────────────────────────────────────────────────────
    output logic [7:0]  w_out   [0:N_PE-1][0:N_TAP-1],
    output logic [31:0] bias_out
);

// =============================================================================
// Valid byte counts per sub-mode (SWIN Block mode)
// =============================================================================
localparam int MSA_QKV_COL_BYTES = 96;    // 24 words × 4 B
localparam int MSA_QKT_COL_BYTES = 32;    //  8 words × 4 B
localparam int MSA_SV_COL_BYTES  = 52;    // 13 words × 4 B (49 B padded to 52)
localparam int FFN_W1_COL_BYTES  = 384;   // 96 words × 4 B  (384 inputs)
// FFN_W2 col = 96 B = same as MSA_QKV_COL_BYTES (96 inputs)
// Controller distinguishes W1 vs W2 by driving the correct msa_sub_mode:
//   W1 → msa_sub_mode = 2'b11, valid_bytes = 384
//   W2 → msa_sub_mode = 2'b00, valid_bytes =  96  (reuse QKV sub-mode)

// =============================================================================
// Internal storage: double-banked byte array + bias register
// =============================================================================
logic [7:0]  bank  [0:1][0:MAX_BYTES-1];
logic [31:0] bias  [0:1];
logic        active;
logic        shadow;
assign shadow = ~active;

// ── Bank swap ─────────────────────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) active <= 1'b0;
    else if (swap) active <= shadow;

// =============================================================================
// Shadow-bank write  (only one mode active at a time)
// =============================================================================
always_ff @(posedge clk) begin
    case (mode)

        // ── Conv (Patch Embedding / Patch Merging conv kernel) ────────────
        2'b00: begin
            if (conv_load_en) begin
                bank[shadow][conv_load_pe_idx * N_TAP    ] <= conv_load_data[ 7: 0];
                bank[shadow][conv_load_pe_idx * N_TAP + 1] <= conv_load_data[15: 8];
                bank[shadow][conv_load_pe_idx * N_TAP + 2] <= conv_load_data[23:16];
                bank[shadow][conv_load_pe_idx * N_TAP + 3] <= conv_load_data[31:24];
            end
            if (conv_bias_load_en)
                bias[shadow] <= conv_bias_load_data;
        end

        // ── MLP (Patch Merging legacy linear path) ────────────────────────
        2'b01: begin
            if (mlp_load_en) begin
                bank[shadow][mlp_load_k_word * N_TAP    ] <= mlp_load_data[ 7: 0];
                bank[shadow][mlp_load_k_word * N_TAP + 1] <= mlp_load_data[15: 8];
                bank[shadow][mlp_load_k_word * N_TAP + 2] <= mlp_load_data[23:16];
                bank[shadow][mlp_load_k_word * N_TAP + 3] <= mlp_load_data[31:24];
            end
        end

        // ── SWIN Block (MSA + FFN — single round) ─────────────────────────
        // All sub-modes (QKV/QK^T/SxV/FFN) share the same byte-level write.
        // msa_load_word is 7 bits to cover 0..95 for FFN_W1.
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

// =============================================================================
// Active-bank read-out → w_out[pe][tap]
//
// sub_cycle selects a 48-byte window (12 PE × 4 tap) within the active bank.
// For modes where the column fits in 1 sub-cycle, controller drives sub_cycle=0.
// =============================================================================
always_comb begin
    // Default: drive zeros
    for (int pe = 0; pe < N_PE; pe++)
        for (int tap = 0; tap < N_TAP; tap++)
            w_out[pe][tap] = 8'b0;
    bias_out = 32'd0;

    case (mode)

        // ── Conv ──────────────────────────────────────────────────────────
        2'b00: begin
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++)
                    w_out[pe][tap] = bank[active][pe * N_TAP + tap];
            bias_out = bias[active];
        end

        // ── MLP (Patch Merging legacy) ────────────────────────────────────
        2'b01: begin
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP)
                                      + pe * N_TAP + tap;
                    w_out[pe][tap] = (k < MAX_BYTES) ? bank[active][k] : 8'b0;
                end
        end

        // ── SWIN Block (MSA + FFN) ────────────────────────────────────────
        // sub_cycle selects the 48-byte slice of the currently-loaded column.
        //
        //   2'b00 QKV/Proj : 96 B → sub_cycle 0..1
        //   2'b01 QK^T     : 32 B → sub_cycle 0  (only, 32 valid, 16 zero)
        //   2'b10 SxV      : 52 B → sub_cycle 0 (48 B), 1 (4 B valid)
        //   2'b11 FFN      : W_FFN1: 384 B → sub_cycle 0..7 (full)
        //                    W_FFN2: 96 B  → sub_cycle 0..1 (controller
        //                            switches msa_sub_mode to 2'b00 for W2,
        //                            so W2 reuses the QKV/Proj path above)
        2'b10: begin
            automatic int valid_bytes;
            case (msa_sub_mode)
                2'b00:   valid_bytes = MSA_QKV_COL_BYTES;  // QKV, Proj, FFN W2
                2'b01:   valid_bytes = MSA_QKT_COL_BYTES;  // QK^T
                2'b10:   valid_bytes = MSA_SV_COL_BYTES;   // SxV
                2'b11:   valid_bytes = FFN_W1_COL_BYTES;   // FFN W1 (384 B)
                default: valid_bytes = 0;
            endcase

            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP)
                                      + pe * N_TAP + tap;
                    w_out[pe][tap] = (k < valid_bytes) ? bank[active][k] : 8'b0;
                end
        end

        default: ;
    endcase
end

endmodule
