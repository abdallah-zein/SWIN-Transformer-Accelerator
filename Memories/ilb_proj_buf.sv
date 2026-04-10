// =============================================================================
// ilb_proj_buf.sv  (rev 2 — all 4 Swin stages supported)
//
// ── What changed from rev 1 ───────────────────────────────────────────────
//   C_BYTES:     96 → 768  (Stage 4)
//   BANK_BYTES: 4704 → 37632  (49 × 768)
//   wr_col: 7 bits → 10 bits
//   rd_col: 7 bits → 10 bits
//   rmw_addr: 11 bits → 14 bits  (49 × (768/4) = 49×192 = 9408 words; 2^14=16384 ≥ 9408)
//   Added c_bytes [9:0] runtime port.
//
// ── rmw_addr sizing ───────────────────────────────────────────────────────
//   rmw_addr is a WORD address (4-byte granularity):
//     Stage4: 49 × (768/4) = 9,408 words → 14 bits (2^14=16384 ≥ 9408)
// =============================================================================

module ilb_proj_buf #(
    parameter int N_PATCHES = 49,
    parameter int C_BYTES   = 768,  // Stage 4
    parameter int N_ROWS    = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // Runtime: actual channel bytes for current stage
    input  logic [9:0]  c_bytes,  // 96/192/384/768

    // ── Write port ────────────────────────────────────────────────────────
    input  logic        wr_en,
    input  logic [5:0]  wr_patch_base,
    input  logic [9:0]  wr_col,
    input  logic [7:0]  wr_data [0:N_ROWS-1],

    // ── Read port ─────────────────────────────────────────────────────────
    input  logic        rd_en,
    input  logic [5:0]  rd_patch,
    input  logic [9:0]  rd_col,
    output logic [31:0] rd_data,

    // ── RMW port — residual additions ────────────────────────────────────
    // rmw_addr = patch * (c_bytes/4) + col_word
    // Stage4 max: 49 × 192 = 9,408 → 14 bits
    input  logic        rmw_rd_en,
    input  logic [13:0] rmw_addr,
    output logic [31:0] rmw_rd_data,
    input  logic        rmw_wr_en,
    input  logic [31:0] rmw_wr_data
);

    localparam int BANK_BYTES = N_PATCHES * C_BYTES;  // 37,632

    logic [7:0] mem [0:BANK_BYTES-1];

    initial begin
        for (int i = 0; i < BANK_BYTES; i++) mem[i] = '0;
    end

    // Write
    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int patch = int'(wr_patch_base) + r;
                if (patch < N_PATCHES) begin
                    automatic int addr = patch * int'(c_bytes) + int'(wr_col);
                    mem[addr] <= wr_data[r];
                end
            end
        end
    end

    // Read
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_data <= '0;
        else if (rd_en) begin
            automatic int base = int'(rd_patch) * int'(c_bytes) + int'(rd_col);
            rd_data <= { mem[base+3], mem[base+2], mem[base+1], mem[base] };
        end
    end

    // RMW read
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rmw_rd_data <= '0;
        else if (rmw_rd_en) begin
            automatic int bbase = int'(rmw_addr) * 4;
            rmw_rd_data <= { mem[bbase+3], mem[bbase+2], mem[bbase+1], mem[bbase] };
        end
    end

    // RMW write
    always_ff @(posedge clk) begin
        if (rmw_wr_en) begin
            automatic int bbase = int'(rmw_addr) * 4;
            mem[bbase  ] <= rmw_wr_data[ 7: 0];
            mem[bbase+1] <= rmw_wr_data[15: 8];
            mem[bbase+2] <= rmw_wr_data[23:16];
            mem[bbase+3] <= rmw_wr_data[31:24];
        end
    end

endmodule
