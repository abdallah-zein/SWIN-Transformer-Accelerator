// =============================================================================
// ilb_patch_embed_buf.sv  (rev 2 — multi-stage verification pass)
//
// ── Why this file is unchanged ────────────────────────────────────────────
//   Patch Embedding occurs exactly ONCE in the Swin-T pipeline (before
//   Stage 1) and is a fixed 4×4-stride convolution of the 224×224×3 input
//   producing a (56×56, 96) INT8 feature map.  Its dimensions are
//   INDEPENDENT of the Swin stage count; there is no Stage 2/3/4 version
//   of Patch Embedding.
//
//   All sizing parameters (H=56, W=56, C=96) therefore remain correct for
//   all-stage support without modification.
//
//   The existing spa_wr_addr bug (18 bits, max addr 301055 > 2^18=262144)
//   noted in the sister module (ilb_patch_merge_buf) does NOT apply here
//   because ilb_patch_embed_buf uses a different write interface:
//     wr_ch [6:0], wr_row [5:0], wr_col_grp [2:0]
//   rather than a flat byte address.  The backing store address is computed
//   internally, so there is no address-width bug to fix.
//
// ── No RTL changes ────────────────────────────────────────────────────────
// =============================================================================

module ilb_patch_embed_buf #(
    parameter int H      = 56,
    parameter int W      = 56,
    parameter int C      = 96,
    parameter int N_ROWS = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── Write port ────────────────────────────────────────────────────────
    input  logic        wr_en,
    input  logic [6:0]  wr_ch,
    input  logic [5:0]  wr_row,
    input  logic [2:0]  wr_col_grp,
    input  logic [7:0]  wr_data [0:N_ROWS-1],

    // ── Read port — 32-bit word, 1-cycle latency ──────────────────────────
    input  logic        rd_en,
    input  logic [16:0] rd_addr,     // word addr: 0 .. TOTAL_WORDS-1 (75263)
    output logic [31:0] rd_data,

    // ── Status ────────────────────────────────────────────────────────────
    output logic        embed_done,
    input  logic        flush_req,
    output logic        flush_done
);

    localparam int PIXELS_PER_CH = H * W;             // 3136
    localparam int TOTAL_BYTES   = C * PIXELS_PER_CH; // 301056
    localparam int TOTAL_WORDS   = TOTAL_BYTES / 4;   // 75264
    localparam int COL_GROUPS    = W / N_ROWS;         // 8

    logic [7:0] mem [0:TOTAL_BYTES-1];

    initial begin
        for (int i = 0; i < TOTAL_BYTES; i++) mem[i] = '0;
    end

    // ── Write: 7 bytes per cycle ──────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int col   = int'(wr_col_grp) * N_ROWS + r;
                automatic int baddr = int'(wr_ch)  * PIXELS_PER_CH
                                    + int'(wr_row) * W
                                    + col;
                if (col < W)
                    mem[baddr] <= wr_data[r];
            end
        end
    end

    // ── Read: 4 packed bytes, 1-cycle latency ────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else if (rd_en) begin
            automatic int bbase = int'(rd_addr) * 4;
            rd_data <= { mem[bbase+3], mem[bbase+2], mem[bbase+1], mem[bbase] };
        end
    end

    // ── embed_done ────────────────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            embed_done <= 1'b0;
        else
            embed_done <= wr_en
                          && (wr_ch      == 7'(C - 1))
                          && (wr_row     == 6'(H - 1))
                          && (wr_col_grp == 3'(COL_GROUPS - 1));
    end

    // ── Off-chip flush sequencer ──────────────────────────────────────────
    logic [16:0] flush_addr_r;
    logic        flushing_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            flushing_r   <= 1'b0;
            flush_addr_r <= '0;
            flush_done   <= 1'b0;
        end else begin
            flush_done <= 1'b0;
            if (flush_req && !flushing_r) begin
                flushing_r   <= 1'b1;
                flush_addr_r <= '0;
            end else if (flushing_r) begin
                if (flush_addr_r == 17'(TOTAL_WORDS - 1)) begin
                    flushing_r   <= 1'b0;
                    flush_addr_r <= '0;
                    flush_done   <= 1'b1;
                end else begin
                    flush_addr_r <= flush_addr_r + 1'b1;
                end
            end
        end
    end

endmodule
