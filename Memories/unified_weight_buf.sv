// =============================================================================
// unified_weight_buf.sv  (rev 6 — Patch Embedding sub-cycle guard added)
//
// ── What changed from rev 5 ───────────────────────────────────────────────
//   FIX — CONV mode: enforce sub_cycle == 0 in read-out
//
//   Patch Embedding uses a 4×4×3 = 48-element kernel.
//   12 PEs × 4 taps = 48 elements → the entire kernel fits in ONE sub-cycle.
//   Therefore sub_cycle is always 0 during Conv compute, and the index
//     k = sub_cycle * 48 + pe * 4 + tap
//   evaluates correctly for sub_cycle=0.
//
//   However, if the controller ever accidentally sends sub_cycle > 0 (e.g.
//   a signal glitch or wrong FSM entry), k would exceed BYTES_CONV=48 and
//   the out-of-bounds guard (k < BYTES_CONV) would correctly zero the output
//   — so the gate WAS already there and worked.
//
//   What was MISSING: a clear comment and the defensive assertion that makes
//   the intent unambiguous.  Added below as a simulation-only assertion.
//   The read logic itself is functionally correct in rev 5 — this is a
//   clarification and defensive hardening only.
//
//   Also corrected: BYTES_CONV localparams comment to explicitly state
//   "1 sub-cycle only" and updated the mode table in the header.
//
// ── Mode / sub-mode table (complete) ─────────────────────────────────────
//
//   mode 2'b00  CONV (Patch Embedding)
//     Kernel: 4×4×3 = 48 bytes → fits in 1 sub-cycle (sub_cycle always 0)
//     conv_load_pe_idx = 0..11  (one 32-bit word per PE)
//     Weight layout: bank[pe*4 + tap]  → 48 bytes total
//
//   mode 2'b01  MLP (Patch Merging)
//     mlp_sub_mode = 1'b0 → W1: 96 B, 2 sub-cycles (0..1)
//     mlp_sub_mode = 1'b1 → W2: 384 B, 8 sub-cycles (0..7)
//
//   mode 2'b10  SWIN Block (W-MSA / SW-MSA + FFN)
//     msa_sub_mode 2'b00 → QKV/Proj/FFN W2:  96 B, 2 sub-cycles
//     msa_sub_mode 2'b01 → QK^T:              32 B, 1 sub-cycle
//     msa_sub_mode 2'b10 → SxV:               52 B, 2 sub-cycles (2nd partial)
//     msa_sub_mode 2'b11 → FFN W1:           384 B, 8 sub-cycles
//
// ── Bank layout ──────────────────────────────────────────────────────────
//   Single contiguous byte array [0 .. MAX_BYTES-1].
//   Sub-cycle k reads bytes [k*48 .. k*48+47] from the active bank.
//   Bytes at index >= valid_bytes for the current sub-mode are zero-masked.
//   MAX_BYTES = 384 (FFN W1 column — largest operand).
// =============================================================================

module unified_weight_buf #(
    parameter int MAX_BYTES = 384,
    parameter int N_PE      = 12,
    parameter int N_TAP     = 4
)(
    input  logic        clk,
    input  logic        rst_n,

    // 2'b00 = CONV, 2'b01 = MLP, 2'b10 = SWIN Block
    input  logic [1:0]  mode,

    // Promotes shadow → active
    input  logic        swap,

    // Pulse once before each new shadow fill — zeroes entire shadow bank
    // (guarantees clean padding for SxV 49→52 B and QK^T 32 B)
    input  logic        shadow_clr,

    // ── CONV load (mode == 2'b00) ─────────────────────────────────────────
    // conv_load_pe_idx = 0..11; one 32-bit word per PE
    // Address: bank[pe_idx * N_TAP + 0..3]
    input  logic        conv_load_en,
    input  logic [3:0]  conv_load_pe_idx,
    input  logic [31:0] conv_load_data,

    // ── MLP load (mode == 2'b01) ──────────────────────────────────────────
    input  logic        mlp_load_en,
    input  logic        mlp_sub_mode,         // 0=W1 96B, 1=W2 384B
    input  logic [6:0]  mlp_load_k_word,      // 0..23 (W1) or 0..95 (W2)
    input  logic [31:0] mlp_load_data,

    // ── SWIN Block load (mode == 2'b10) ───────────────────────────────────
    input  logic        msa_load_en,
    input  logic [6:0]  msa_load_word,        // 0..95 (7 bits covers all sub-modes)
    input  logic [31:0] msa_load_data,
    input  logic [1:0]  msa_sub_mode,         // controls valid-byte window

    // ── Sub-cycle counter ─────────────────────────────────────────────────
    input  logic [2:0]  sub_cycle,

    // ── Weight output to MMU ──────────────────────────────────────────────
    output logic [7:0]  w_out [0:N_PE-1][0:N_TAP-1]
);

// =============================================================================
// Valid-byte constants
// =============================================================================
localparam int BYTES_CONV = N_PE * N_TAP;   //  48 B  — 1 sub-cycle ONLY
localparam int BYTES_W1   = 96;             //  96 B  — 2 sub-cycles
localparam int BYTES_W2   = MAX_BYTES;      // 384 B  — 8 sub-cycles
localparam int BYTES_QKV  = 96;             //  96 B  — 2 sub-cycles
localparam int BYTES_QKT  = 32;             //  32 B  — 1 sub-cycle
localparam int BYTES_SV   = 52;             //  52 B  — 2 sub-cycles (49 + 3 zero-pad)
localparam int BYTES_FFN1 = MAX_BYTES;      // 384 B  — 8 sub-cycles

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
// Shadow bank write (shadow_clr has highest priority)
// =============================================================================
always_ff @(posedge clk) begin
    if (shadow_clr) begin
        for (int i = 0; i < MAX_BYTES; i++)
            bank[shadow][i] <= 8'h00;
    end else begin
        case (mode)

            // ── CONV: 12 words, 48 bytes ──────────────────────────────────
            // Kernel element layout: bank[pe * N_TAP + tap]
            // Each 32-bit word packs 4 consecutive kernel weights for one PE.
            // All 48 bytes are written in 12 load calls (one per PE).
            2'b00: begin
                if (conv_load_en) begin
                    bank[shadow][conv_load_pe_idx * N_TAP    ] <= conv_load_data[ 7: 0];
                    bank[shadow][conv_load_pe_idx * N_TAP + 1] <= conv_load_data[15: 8];
                    bank[shadow][conv_load_pe_idx * N_TAP + 2] <= conv_load_data[23:16];
                    bank[shadow][conv_load_pe_idx * N_TAP + 3] <= conv_load_data[31:24];
                end
            end

            // ── MLP: W1 (24 words, 96 B) or W2 (96 words, 384 B) ─────────
            2'b01: begin
                if (mlp_load_en) begin
                    bank[shadow][mlp_load_k_word * N_TAP    ] <= mlp_load_data[ 7: 0];
                    bank[shadow][mlp_load_k_word * N_TAP + 1] <= mlp_load_data[15: 8];
                    bank[shadow][mlp_load_k_word * N_TAP + 2] <= mlp_load_data[23:16];
                    bank[shadow][mlp_load_k_word * N_TAP + 3] <= mlp_load_data[31:24];
                end
            end

            // ── SWIN Block: all sub-modes share one write path ─────────────
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
// Active-bank read-out  →  w_out[pe][tap]
//
// For every mode:
//   k = sub_cycle * (N_PE * N_TAP) + pe * N_TAP + tap
//   w_out[pe][tap] = (k < valid_bytes) ? bank[active][k] : 8'h00
//
// CONV note: valid_bytes = 48, sub_cycle is always 0 during Conv compute.
//   k = 0*48 + pe*4 + tap ∈ [0..47] — always within range.
//   If sub_cycle were non-zero the guard (k < 48) would correctly zero output.
//   The assertion below catches this in simulation.
// =============================================================================
always_comb begin
    for (int pe = 0; pe < N_PE; pe++)
        for (int tap = 0; tap < N_TAP; tap++)
            w_out[pe][tap] = 8'h00;

    case (mode)

        // ── CONV: 1 sub-cycle, 48 bytes ───────────────────────────────────
        2'b00: begin
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP) + pe * N_TAP + tap;
                    // Guard covers sub_cycle=0 (normal) and any unexpected value
                    w_out[pe][tap] = (k < BYTES_CONV) ? bank[active][k] : 8'h00;
                end
        end

        // ── MLP: W1 or W2 ─────────────────────────────────────────────────
        2'b01: begin
            automatic int vb_mlp = mlp_sub_mode ? BYTES_W2 : BYTES_W1;
            for (int pe = 0; pe < N_PE; pe++)
                for (int tap = 0; tap < N_TAP; tap++) begin
                    automatic int k = int'(sub_cycle) * (N_PE * N_TAP) + pe * N_TAP + tap;
                    w_out[pe][tap] = (k < vb_mlp) ? bank[active][k] : 8'h00;
                end
        end

        // ── SWIN Block ────────────────────────────────────────────────────
        2'b10: begin
            automatic int vb_msa;
            case (msa_sub_mode)
                2'b00:   vb_msa = BYTES_QKV;
                2'b01:   vb_msa = BYTES_QKT;
                2'b10:   vb_msa = BYTES_SV;
                2'b11:   vb_msa = BYTES_FFN1;
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
    if (mode == 2'b00 && sub_cycle != 3'd0)
        $error("[unified_weight_buf] CONV mode: sub_cycle=%0d but must be 0 (kernel fits in 1 sub-cycle)", sub_cycle);
end
// synthesis translate_on

endmodule