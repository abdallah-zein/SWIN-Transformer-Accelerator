// =============================================================================
// ilb_ffn1_buf.sv  (rev 2 — all 4 Swin stages supported)
//
// ── What changed from rev 1 ───────────────────────────────────────────────
//   C_BYTES:    384 →  3072   (Stage 4: FFN_C = 4×768 = 3072)
//   BANK_BYTES: 18816 → 150528  (49 × 3072)
//   wr_col: 9 bits → 12 bits   (covers 0..3071)
//   rd_col_word: 8 bits → 10 bits  (covers 0..767: 3072/4=768 words)
//   Added ffn_c_bytes [11:0] runtime port.
//
// ── Sizing per stage ──────────────────────────────────────────────────────
//   Stage1 (FFN=384):   49× 384=  18,816 bytes
//   Stage2 (FFN=768):   49× 768=  37,632 bytes
//   Stage3 (FFN=1536):  49×1536=  75,264 bytes
//   Stage4 (FFN=3072):  49×3072= 150,528 bytes  ← BANK_BYTES ceiling
// =============================================================================

module ilb_ffn1_buf #(
    parameter int N_PATCHES = 49,
    parameter int C_BYTES   = 3072,   // Stage 4 max
    parameter int N_ROWS    = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // Runtime: actual FFN expanded width in bytes
    input  logic [11:0] ffn_c_bytes,  // 384/768/1536/3072

    // ── Write port — GCU (GELU) output ───────────────────────────────────
    input  logic        wr_en,
    input  logic [5:0]  wr_patch_base,
    input  logic [11:0] wr_col,        // 0..C_BYTES-1
    input  logic [7:0]  wr_data [0:N_ROWS-1],

    // ── Read port — FFN2 activation input ────────────────────────────────
    // rd_col_word = word index = byte_col / 4  (0..767 Stage4)
    input  logic        rd_en,
    input  logic [5:0]  rd_patch,
    input  logic [9:0]  rd_col_word,   // 10 bits: 0..767
    output logic [31:0] rd_data
);

    localparam int BANK_BYTES = N_PATCHES * C_BYTES;  // 49 × 3072 = 150,528

    logic [7:0] mem [0:BANK_BYTES-1];

    initial begin
        for (int i = 0; i < BANK_BYTES; i++) mem[i] = '0;
    end

    // Write — 7 bytes per cycle from GCU
    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int patch = int'(wr_patch_base) + r;
                if (patch < N_PATCHES) begin
                    automatic int addr = patch * int'(ffn_c_bytes) + int'(wr_col);
                    mem[addr] <= wr_data[r];
                end
            end
        end
    end

    // Read — 4 bytes per cycle (1-cycle latency)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_data <= '0;
        else if (rd_en) begin
            automatic int base = int'(rd_patch) * int'(ffn_c_bytes)
                               + int'(rd_col_word) * 4;
            rd_data <= { mem[base+3], mem[base+2], mem[base+1], mem[base] };
        end
    end

endmodule
