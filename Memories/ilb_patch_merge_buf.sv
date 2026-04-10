// =============================================================================
// ilb_patch_merge_buf.sv  (rev 2 — all 3 Patch Merging stages supported)
//
// ── What changed from rev 1 ───────────────────────────────────────────────
//
//   BUG FIX — spa_wr_addr was 18 bits but PM1 needs 19 bits
//     PM1 SPA_BYTES = 28×28×384 = 301,056 bytes → max addr = 301,055
//     2^18 = 262,144 < 301,055  →  address OVERFLOW in rev 1
//     Fixed: spa_wr_addr [18:0]  (19 bits; 2^19 = 524,288 ≥ 301,055 ✓)
//
//   PORT WIDTH UPDATES for PM2 and PM3:
//     fc_wr_col / pb_rd_col : 8 bits → 10 bits  (PM3 C_FC=768, max=767)
//     fc_wr_row_base        : 10 bits unchanged  (PM1 max=777, dominant)
//     pb_rd_patch           : 10 bits unchanged  (PM1 max=783, dominant)
//
//   BACKING STORE unchanged at 301,056 bytes (PM1 dominant, all later
//   PM stages fit within this footprint since their SPA_BYTES are smaller).
//
//   PARAMETERS updated and runtime configuration port added:
//     New port: pm_cfg_* — runtime sizing for the active PM stage.
//     This avoids needing separate module instances per PM stage.
//
// ── Patch Merging parameters per stage ────────────────────────────────────
//
//   Stage │ H_IN │ W_IN │ C_IN │ H_OUT │ W_OUT │ C_SPA │ C_FC  │ SPA_BYTES │ FC_BYTES
//   ──────┼──────┼──────┼──────┼───────┼───────┼───────┼───────┼───────────┼──────────
//    PM1  │  56  │  56  │  96  │  28   │  28   │  384  │  192  │  301,056  │  150,528
//    PM2  │  28  │  28  │  192 │  14   │  14   │  768  │  384  │   75,264  │   37,632
//    PM3  │  14  │  14  │  384 │   7   │   7   │ 1,536 │  768  │   18,816  │    9,408
//
//   All fit within the MEM_BYTES = 301,056 ceiling.
//   The controller loads pm_cfg_* before asserting the first write enable.
//
// ── Runtime configuration ports (pm_cfg_*) ────────────────────────────────
//   pm_cfg_h_out     [5:0]  : output spatial height (28/14/7)
//   pm_cfg_w_out     [5:0]  : output spatial width  (28/14/7)
//   pm_cfg_c_spa    [10:0]  : spatial-merge channel depth (384/768/1536)
//   pm_cfg_c_fc     [10:0]  : FC-layer output channels    (192/384/768)
//   pm_cfg_spa_bytes[18:0]  : total Phase-A bytes          (SPA_BYTES above)
//   pm_cfg_spa_words[16:0]  : total Phase-A words          (SPA_BYTES / 4)
//   pm_cfg_fc_patches[9:0]  : total Phase-B patches         (H_OUT × W_OUT)
//
//   These are stable throughout each PM pass and drive the done/flush logic.
//
// =============================================================================

module ilb_patch_merge_buf #(
    // Compile-time maxima (PM1 dominant)
    parameter int H_IN   = 56,
    parameter int W_IN   = 56,
    parameter int C_IN   = 96,
    parameter int H_OUT  = 28,
    parameter int W_OUT  = 28,
    parameter int C_SPA  = 384,   // max spatial-merge channels (PM1)
    parameter int C_FC   = 192,   // max FC output channels — COMPILE-TIME DEFAULT
    parameter int N_ROWS = 7
)(
    input  logic        clk,
    input  logic        rst_n,

    // ── Runtime PM stage configuration ────────────────────────────────────
    // Set by the controller before each PM round and held stable.
    input  logic [5:0]   pm_cfg_h_out,       // 28 / 14 / 7
    input  logic [5:0]   pm_cfg_w_out,       // 28 / 14 / 7
    input  logic [10:0]  pm_cfg_c_spa,       // 384 / 768 / 1536
    input  logic [10:0]  pm_cfg_c_fc,        // 192 / 384 / 768
    input  logic [18:0]  pm_cfg_spa_bytes,   // 301056 / 75264 / 18816
    input  logic [16:0]  pm_cfg_spa_words,   //  75264 / 18816 /  4704
    input  logic [9:0]   pm_cfg_fc_patches,  //    784 /   196 /    49

    // =========================================================================
    // PHASE-A WRITE PORT — Spatial-merge byte stream
    //
    // The controller provides a flat byte address for each byte it transfers
    // from the input buffer.  For PM stage s:
    //
    //   spa_wr_addr = pr × (W_OUT_s × C_SPA_s) + pc × C_SPA_s + g × C_IN_s + ch
    //
    // Address range: [0 .. pm_cfg_spa_bytes - 1]
    // 19 bits: covers PM1 max = 301,055 (2^19 = 524,288 ≥ 301,055 ✓)
    // =========================================================================
    input  logic        spa_wr_en,
    input  logic [18:0] spa_wr_addr,   // 19 bits — BUG FIX from 18 bits
    input  logic [7:0]  spa_wr_data,

    // =========================================================================
    // PHASE-A READ PORT — 32-bit word, 1-cycle latency
    // pa_rd_addr [16:0] : word address 0 .. pm_cfg_spa_words-1
    // =========================================================================
    input  logic        pa_rd_en,
    input  logic [16:0] pa_rd_addr,
    output logic [31:0] pa_rd_data,

    // =========================================================================
    // PHASE-B WRITE PORT — FC-layer result (7 INT8 bytes/cycle)
    //
    // fc_wr_row_base [9:0] : first patch of burst (0, 7, 14, …)
    //                        Range: 0 .. pm_cfg_fc_patches - N_ROWS
    // fc_wr_col      [9:0] : byte column 0 .. pm_cfg_c_fc - 1
    //                        10 bits: covers PM3 max = 767 (PM1=191, PM2=383)
    // =========================================================================
    input  logic        fc_wr_en,
    input  logic [9:0]  fc_wr_row_base,
    input  logic [9:0]  fc_wr_col,     // 10 bits: 0..767 (PM3 C_FC=768)
    input  logic [7:0]  fc_wr_data [0:N_ROWS-1],

    // =========================================================================
    // PHASE-B READ PORT — 4 packed INT8 bytes, 1-cycle latency
    //
    // pb_rd_patch [9:0] : patch index 0 .. pm_cfg_fc_patches-1
    // pb_rd_col   [9:0] : word-aligned byte column 0 .. pm_cfg_c_fc-4
    //                     10 bits: covers PM3 max = 764
    // =========================================================================
    input  logic        pb_rd_en,
    input  logic [9:0]  pb_rd_patch,
    input  logic [9:0]  pb_rd_col,     // 10 bits: word-aligned
    output logic [31:0] pb_rd_data,

    // =========================================================================
    // STATUS
    // =========================================================================
    output logic        spa_done,   // 1-cycle pulse: Phase-A fully written
    output logic        fc_done,    // 1-cycle pulse: Phase-B fully written
    input  logic        flush_req,
    output logic        flush_done
);

    // ── Backing store: PM1 SPA_BYTES = 301,056 bytes (largest case) ───────
    localparam int MEM_BYTES = H_OUT * W_OUT * C_SPA * 4;  // 28*28*384 = 301056
    // (compile-time params give PM1 ceiling; runtime pm_cfg_* limit usage)

    logic [7:0] mem [0:MEM_BYTES-1];

    initial begin
        for (int i = 0; i < MEM_BYTES; i++) mem[i] = '0;
    end

    // =========================================================================
    // Phase-A Write — 1 byte per cycle, flat byte address
    // =========================================================================
    always_ff @(posedge clk) begin
        if (spa_wr_en)
            mem[spa_wr_addr] <= spa_wr_data;
    end

    // =========================================================================
    // Phase-A Read — 4 bytes per word, 1-cycle latency
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pa_rd_data <= '0;
        else if (pa_rd_en) begin
            automatic int bbase = int'(pa_rd_addr) * 4;
            pa_rd_data <= { mem[bbase+3], mem[bbase+2], mem[bbase+1], mem[bbase] };
        end
    end

    // =========================================================================
    // Phase-B Write — 7 INT8 bytes per cycle
    // byte addr = patch * pm_cfg_c_fc + fc_wr_col
    // =========================================================================
    always_ff @(posedge clk) begin
        if (fc_wr_en) begin
            for (int r = 0; r < N_ROWS; r++) begin
                automatic int patch = int'(fc_wr_row_base) + r;
                if (patch < int'(pm_cfg_fc_patches)) begin
                    automatic int baddr = patch * int'(pm_cfg_c_fc) + int'(fc_wr_col);
                    mem[baddr] <= fc_wr_data[r];
                end
            end
        end
    end

    // =========================================================================
    // Phase-B Read — 4 bytes per word, 1-cycle latency
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pb_rd_data <= '0;
        else if (pb_rd_en) begin
            automatic int bbase = int'(pb_rd_patch) * int'(pm_cfg_c_fc)
                                + int'(pb_rd_col);
            pb_rd_data <= { mem[bbase+3], mem[bbase+2], mem[bbase+1], mem[bbase] };
        end
    end

    // =========================================================================
    // spa_done — last Phase-A byte written
    // Condition: spa_wr_en && spa_wr_addr == pm_cfg_spa_bytes - 1
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) spa_done <= 1'b0;
        else spa_done <= spa_wr_en &&
                         (spa_wr_addr == (pm_cfg_spa_bytes - 19'd1));
    end

    // =========================================================================
    // fc_done — last Phase-B burst write
    // Last group: fc_wr_row_base = pm_cfg_fc_patches - N_ROWS
    //             fc_wr_col = pm_cfg_c_fc - 1
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fc_done <= 1'b0;
        else fc_done <= fc_wr_en
                        && (fc_wr_row_base == (pm_cfg_fc_patches - 10'(N_ROWS)))
                        && (fc_wr_col      == (pm_cfg_c_fc       - 10'd1));
    end

    // =========================================================================
    // Off-chip flush sequencer — Phase-A DMA
    // =========================================================================
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
                if (flush_addr_r == (pm_cfg_spa_words - 17'd1)) begin
                    flushing_r   <= 1'b0;
                    flush_addr_r <= '0;
                    flush_done   <= 1'b1;
                end else begin
                    flush_addr_r <= flush_addr_r + 17'd1;
                end
            end
        end
    end

endmodule
