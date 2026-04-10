// =============================================================================
// ilb_score_buf.sv  (rev 2 — all 4 Swin stages; depth unchanged)
//
// ── What changed from rev 1 ───────────────────────────────────────────────
//   DEPTH unchanged: 49×49 = 2401 INT32 words.
//   d_head = 32 is CONSTANT across all Swin stages (heads vary, d_head fixed).
//   The window size is ALWAYS 7×7 = 49 patches across all stages.
//   Therefore the per-head, per-window score matrix is ALWAYS 49×49.
//   This buffer holds ONE head at a time and is reused per head.
//   Number of heads varies (3/6/12/24) but the buffer is the same size.
//
//   The ONLY change: rmw_addr width expanded from 12 to 12 bits (unchanged:
//   2^12=4096 ≥ 2401 ✓).  No RTL change needed.
//
// ── No changes to RTL ─────────────────────────────────────────────────────
//   This file is reproduced here for completeness with consistent rev history.
// =============================================================================

module ilb_score_buf #(
    parameter int N_PATCHES = 49,
    parameter int N_ROWS    = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    input  logic        score_commit,
    input  logic        score_clear,
    output logic        score_valid,

    // Write port — QK^T burst
    input  logic        wr_en,
    input  logic [5:0]  wr_row_base,
    input  logic [5:0]  wr_col,
    input  logic [31:0] wr_data [0:N_ROWS-1],

    // RMW port — mask / Softmax write-back
    input  logic        rmw_rd_en,
    input  logic [11:0] rmw_addr,
    output logic [31:0] rmw_rd_data,
    input  logic        rmw_wr_en,
    input  logic [31:0] rmw_wr_data,

    // Sequential read port
    input  logic        rd_en,
    input  logic [11:0] rd_addr,
    output logic [31:0] rd_data
);

    localparam int DEPTH = N_PATCHES * N_PATCHES;  // 2401

    logic [31:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)            score_valid <= 1'b0;
        else if (score_clear)  score_valid <= 1'b0;
        else if (score_commit) score_valid <= 1'b1;
    end

    // Write — QK^T burst (7 query rows × 1 key column)
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

    // RMW read (1-cycle latency)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)         rmw_rd_data <= '0;
        else if (rmw_rd_en) rmw_rd_data <= mem[rmw_addr];
    end

    // RMW write
    always_ff @(posedge clk) begin
        if (rmw_wr_en) mem[rmw_addr] <= rmw_wr_data;
    end

    // Sequential read (1-cycle latency)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)      rd_data <= '0;
        else if (rd_en)  rd_data <= mem[rd_addr];
    end

endmodule
