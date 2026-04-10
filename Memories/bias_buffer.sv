// =============================================================================
// bias_buffer.sv  (rev 3 — all 4 Swin stages supported)
//
// ── What changed from rev 2 ───────────────────────────────────────────────
//   AW: 12 → 16   (65,536 entries; Stage 4 QK^T = 24 heads × 2401 = 57,624)
//   DEPTH: 4096 → 65536
//   BB_MLP_L1_BASE: 96  → 96   (unchanged — Conv biases still 96)
//   BB_MLP_L2_BASE: 480 → 3168 (96 + 3072; Stage4 FFN1 = 3072 biases)
//   BB_MHA_QKT_BASE: 576 → 3936 (96 + 3072 + 768)
//   BB_PM_BASE: 2977 → removed — PM biases now use the MLP L1/L2 region
//   Added BB_PM_FC_BASE = 3936 (same slot as MHA QKT; PM and MHA never overlap)
//
// ── Revised Memory Map ────────────────────────────────────────────────────
//
//   Address range        Entries  Content
//   ─────────────────────────────────────────────────────────────────────────
//      0 ..     95          96    Conv  (96 output channels)
//     96 ..   3167        3072    MLP/FFN L1  (Stage4 FFN1: 3072 output cols)
//   3168 ..   3935         768    MLP/FFN L2  (Stage4 C=768 output cols)
//   3936 ..  61559       57624    MHA QK^T    (Stage4: 24 heads × 49×49 = 57,624)
//                                 Note: Stage1=2401, Stage2=14406, Stage3=28812
//                                 Stage-specific portions are written before each round
//  61560 ..  65535        3976    Reserved
//   ─────────────────────────────────────────────────────────────────────────
//   Total used: 61,560  ≤  DEPTH = 65,536  ✓
//
//   PM (Patch Merging) biases share the MLP L1/L2 region:
//     PM FC1 bias → BB_MLP_L1_BASE (192/384/768 entries for PM1/2/3)
//     PM FC2 bias → BB_MLP_L2_BASE (192/384/768 entries for PM1/2/3)
//   This is safe because PM and Swin Block never run simultaneously.
//
// ── QK^T bias layout (row-major) ─────────────────────────────────────────
//   bias[head h, query q, key k] → address BB_MHA_QKT_BASE + h*(49*49) + q*49 + k
//   The controller computes the base address for the current (head, row-group)
//   before asserting bb_op_start.
//
// ── Per-operation behaviour (unchanged from rev 2) ────────────────────────
//   Conv / PM (mode=2'b00 or 2'b11): broadcast bias_reg[0] to all 7 outputs
//   MLP / FFN  (mode=2'b01):         bias_out[k] = bias_reg[k], k=0..6
//   MHA QK^T   (mode=2'b10, op=3):   bias_out[k] = bias_reg[k], k=0..6
//   MHA other  (mode=2'b10, op≠3):   bias_out[k] = 0
//
// ── Interface changes from rev 2 ─────────────────────────────────────────
//   cpu_wr_addr: 12 bits → 16 bits
//   bb_op_base_addr: 12 bits → 16 bits
//   No other interface changes.
// =============================================================================

module bias_buffer #(
    parameter int AW              = 16,    // 65,536 entries
    parameter int DW              = 32,
    parameter int BB_CONV_BASE    = 0,
    parameter int BB_MLP_L1_BASE  = 96,    // also PM FC1
    parameter int BB_MLP_L2_BASE  = 3168,  // also PM FC2  (96 + 3072)
    parameter int BB_MHA_QKT_BASE = 3936   // (3168 + 768)
)(
    input  logic           clk,
    input  logic           rst_n,

    // ── CPU / DMA preload ─────────────────────────────────────────────────
    input  logic [AW-1:0]  cpu_wr_addr,
    input  logic [DW-1:0]  cpu_wr_data,
    input  logic           cpu_wr_en,

    // ── Controller interface ───────────────────────────────────────────────
    // mode: 2'b00=Conv  2'b01=MLP/FFN  2'b10=MHA  2'b11=PM
    input  logic [1:0]     mode,
    input  logic [2:0]     mmu_op_code,

    input  logic           bb_op_start,
    input  logic [AW-1:0]  bb_op_base_addr,
    input  logic           bb_advance,

    // ── Status ────────────────────────────────────────────────────────────
    output logic           bias_ready,

    // ── MMU bias output (7 values → mmu_bias[0:6]) ───────────────────────
    output logic [DW-1:0]  bias_out [0:6]
);

    // ── Internal RAM: 65536 × 32-bit ─────────────────────────────────────
    localparam int DEPTH = 1 << AW;  // 65536

    logic [DW-1:0] bias_mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) bias_mem[i] = '0;
    end

    always_ff @(posedge clk) begin
        if (cpu_wr_en)
            bias_mem[cpu_wr_addr] <= cpu_wr_data;
    end

    // ── 7-element register bank ───────────────────────────────────────────
    logic [DW-1:0] bias_reg [0:6];

    // ── FSM ───────────────────────────────────────────────────────────────
    typedef enum logic [1:0] {
        S_IDLE    = 2'd0,
        S_LOADING = 2'd1,
        S_READY   = 2'd2
    } fsm_t;

    fsm_t           fsm;
    logic [AW-1:0]  load_base;
    logic [2:0]     load_cnt_rd;
    logic [2:0]     load_cnt_wr;
    logic [AW-1:0]  ram_rd_addr;

    // ── Registered RAM read address → 1-cycle latency ────────────────────
    logic [DW-1:0]  ram_rd_data;
    always_ff @(posedge clk) begin
        ram_rd_data <= bias_mem[ram_rd_addr];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm         <= S_IDLE;
            load_base   <= '0;
            load_cnt_rd <= '0;
            load_cnt_wr <= '0;
            bias_ready  <= 1'b0;
            for (int i = 0; i < 7; i++) bias_reg[i] <= '0;
        end else begin
            case (fsm)
                S_IDLE: begin
                    bias_ready <= 1'b0;
                    if (bb_op_start) begin
                        load_base   <= bb_op_base_addr;
                        load_cnt_rd <= 3'd0;
                        load_cnt_wr <= 3'd0;
                        fsm         <= S_LOADING;
                    end
                end

                S_LOADING: begin
                    // Override: restart if bb_op_start during load
                    if (bb_op_start) begin
                        load_base   <= bb_op_base_addr;
                        load_cnt_rd <= 3'd0;
                        load_cnt_wr <= 3'd0;
                    end else begin
                        // Advance read pointer (issues reads ahead)
                        if (load_cnt_rd < 3'd6)
                            load_cnt_rd <= load_cnt_rd + 3'd1;

                        // Capture arriving data (1 cycle after read)
                        load_cnt_wr <= load_cnt_rd;
                        if (load_cnt_wr <= 3'd6)
                            bias_reg[load_cnt_wr] <= ram_rd_data;

                        // Done after write-counter hits 6
                        if (load_cnt_wr == 3'd6) begin
                            bias_ready <= 1'b1;
                            fsm        <= S_READY;
                        end
                    end
                end

                S_READY: begin
                    if (bb_op_start) begin
                        // Re-arm immediately
                        load_base   <= bb_op_base_addr;
                        load_cnt_rd <= 3'd0;
                        load_cnt_wr <= 3'd0;
                        bias_ready  <= 1'b0;
                        fsm         <= S_LOADING;
                    end else if (bb_advance) begin
                        // Advance to next group of 7 entries
                        load_base   <= load_base + AW'(7);
                        load_cnt_rd <= 3'd0;
                        load_cnt_wr <= 3'd0;
                        bias_ready  <= 1'b0;
                        fsm         <= S_LOADING;
                    end
                end

                default: fsm <= S_IDLE;
            endcase
        end
    end

    // ── RAM read address mux ──────────────────────────────────────────────
    assign ram_rd_addr = load_base + AW'(load_cnt_rd);

    // ── Output mux ────────────────────────────────────────────────────────
    // Conv / PM (mode 00 / 11): broadcast bias_reg[0]
    // MLP / FFN (mode 01):      bias_out[k] = bias_reg[k]
    // MHA QK^T  (mode 10, op=3):bias_out[k] = bias_reg[k]
    // MHA other (mode 10, op≠3):bias_out[k] = 0
    always_comb begin
        for (int k = 0; k < 7; k++) begin
            case (mode)
                2'b00, 2'b11: bias_out[k] = bias_reg[0];
                2'b01:        bias_out[k] = bias_reg[k];
                2'b10:        bias_out[k] = (mmu_op_code == 3'd3) ? bias_reg[k] : '0;
                default:      bias_out[k] = '0;
            endcase
        end
    end

endmodule
