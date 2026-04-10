// =============================================================================
// full_system_top.sv  (rev 9 — Bias Buffer integration)
//
// ── What changed from rev 8 ───────────────────────────────────────────────
//
//   1. New top-level CPU/DMA write ports for the bias_buffer:
//        cpu_bbuf_wr_addr [11:0]  — entry address   (0 .. 4095)
//        cpu_bbuf_wr_data [31:0]  — 32-bit bias value
//        cpu_bbuf_wr_en           — write-enable strobe
//      The CPU must preload all required bias values for the intended
//      operation BEFORE asserting start.
//
//   2. New controller wires (driven by unified_controller):
//        ctrl_bb_op_start         — arm/re-arm pulse for bias_buffer
//        ctrl_bb_op_base_addr     — base entry address for current op
//        ctrl_bb_advance          — step to next 7-element bias group
//
//   3. mmu_bias_bus width changed: [0:11] → [0:6]
//      This matches the updated mmu.sv (rev 2) and mmu_top.sv (rev 2).
//
//   4. mmu_bias_bus wiring changed:
//        OLD: driven by unified_weight_buf.bias_out (scalar broadcast)
//        NEW: driven by bias_buffer.bias_out[0:6] (per-column)
//      The old mmu_bias_bus assignment that set [p]==0 for p>0 is removed.
//
//   5. unified_weight_buf.bias_out (ubuf_bias_out) retained as a wire but
//      no longer connected to mmu_bias_bus.  It can be removed in a future
//      revision once unified_weight_buf is updated to strip the bias path.
//
//   6. New instance: bias_buffer u_bbuf
//        Parameters: AW=12, DW=32 with the standard memory-map constants.
//
//   7. unified_controller instantiation gains three new outputs:
//        .bb_op_start      (ctrl_bb_op_start)
//        .bb_op_base_addr  (ctrl_bb_op_base_addr)
//        .bb_advance       (ctrl_bb_advance)
//
//   8. bias_ready from u_bbuf is an internal wire.  The controller must
//      check this before asserting mmu_valid_in (implementation detail
//      inside unified_controller; not exposed as a top-level port).
//
// ── CPU preload sequence (bias) ───────────────────────────────────────────
//   (a) Choose the intended operation (mode).
//   (b) Write bias values via cpu_bbuf_wr_addr / cpu_bbuf_wr_data / cpu_bbuf_wr_en.
//       Memory map:
//         Conv       entries [    0 ..   95]   96 × 32-bit (one per output channel)
//         MLP L1     entries [   96 ..  479]  384 × 32-bit (one per output column)
//         MLP L2     entries [  480 ..  575]   96 × 32-bit (one per output column)
//         MHA QK^T   entries [  576 .. 2976] 2401 × 32-bit (row-major, 49×49)
//   (c) Assert start.  The controller fires bb_op_start on the same cycle,
//       arming the bias_buffer at the correct base address.
//
// =============================================================================

module full_system_top (
    input  logic clk,
    input  logic rst_n,

    // ── Mode and control ───────────────────────────────────────────────────
    input  logic [1:0] mode,   // 2'b00=Conv, 2'b01=MLP, 2'b10=MHA
    input  logic start,
    output logic done,

    // ── Post-processing controls ──────────────────────────────────────────
    input  logic               relu_en,

    // ── Feedback control ──────────────────────────────────────────────────
    input  logic               omem_fb_en,

    // ── CPU/DMA: weight_memory write ──────────────────────────────────────
    input  logic [15:0] cpu_wmem_wr_addr,
    input  logic [31:0] cpu_wmem_wr_data,
    input  logic        cpu_wmem_wr_en,

    // ── CPU/DMA: fib_memory write ─────────────────────────────────────────
    input  logic [16:0] cpu_fib_wr_addr,
    input  logic [31:0] cpu_fib_wr_data,
    input  logic        cpu_fib_wr_en,

    // ── CPU/DMA: output_memory read ───────────────────────────────────────
    input  logic [18:0] cpu_omem_rd_addr,
    input  logic        cpu_omem_rd_en,
    output logic [31:0] cpu_omem_rd_data,

    // ── CPU/DMA: shift_buffer write ───────────────────────────────────────
    input  logic [11:0] cpu_sbuf_wr_addr,
    input  logic [31:0] cpu_sbuf_wr_data,
    input  logic        cpu_sbuf_wr_en,

    // ── CPU/DMA: bias_buffer write (NEW rev 9) ────────────────────────────
    // 32-bit bus; one entry per write.  Entry address = entry index directly.
    input  logic [11:0] cpu_bbuf_wr_addr,  // 12-bit address → 4096 entries
    input  logic [31:0] cpu_bbuf_wr_data,
    input  logic        cpu_bbuf_wr_en,

    // ── MWU trigger (MHA only) ────────────────────────────────────────────
    output logic        mha_window_done,

    // ── GCU GELU control (MHA FFN1 output) ───────────────────────────────
    output logic        gcu_start,
    input  logic        gcu_done
);

// =============================================================================
// Localparams
// =============================================================================
localparam int WAW    = 16;
localparam int FAW    = 17;
localparam int OAW    = 19;
localparam int SB_AW  = 14;    // shift_buffer entry address width
localparam int BB_AW  = 12;    // bias_buffer  entry address width

// =============================================================================
// Unified controller output wires (unchanged from rev 8 except BB additions)
// =============================================================================

logic [WAW-1:0] ctrl_wmem_rd_addr;
logic           ctrl_wmem_rd_en;
logic [31:0]    ctrl_wmem_rd_data;

logic [OAW-1:0] ctrl_imem_rd_addr;
logic           ctrl_imem_rd_en;
logic [31:0]    ctrl_imem_rd_data;

logic [OAW-1:0] ctrl_omem_wr_addr;
logic           ctrl_omem_wr_en;

logic           ctrl_wbuf_load_en;
logic [3:0]     ctrl_wbuf_load_pe_idx;
logic [6:0]     ctrl_wbuf_load_k_word;
logic [31:0]    ctrl_wbuf_load_data;
logic           ctrl_wbuf_bias_load_en;
logic [31:0]    ctrl_wbuf_bias_load_data;
logic           ctrl_wbuf_swap;

logic           ctrl_ibuf_load_en;
logic [3:0]     ctrl_ibuf_load_pe_idx;
logic [2:0]     ctrl_ibuf_load_win_idx;
logic [2:0]     ctrl_ibuf_load_row;
logic [6:0]     ctrl_ibuf_load_k_word;
logic [31:0]    ctrl_ibuf_load_data;
logic           ctrl_ibuf_swap;
logic           ctrl_ibuf_l1_capture_en;
logic [8:0]     ctrl_ibuf_l1_col_wr;

logic           ctrl_ibuf_mha_load_en;
logic [5:0]     ctrl_ibuf_mha_load_patch;
logic [4:0]     ctrl_ibuf_mha_load_k_word;
logic [31:0]    ctrl_ibuf_mha_load_data;
logic [5:0]     ctrl_ibuf_mha_capture_row;

logic           ctrl_mmu_valid_in;
logic [2:0]     ctrl_mmu_op_code;
logic [1:0]     ctrl_mmu_stage;
logic [2:0]     ctrl_mmu_sub_cycle;

logic           ctrl_obuf_capture_en;
logic [2:0]     ctrl_obuf_rd_idx;

logic           ctrl_omem_fb_en;

// ── Shift buffer controller wires ────────────────────────────────────────
logic               ctrl_sb_op_start;
logic [SB_AW-1:0]   ctrl_sb_op_base_addr;
logic               ctrl_sb_advance;

// ── Bias buffer controller wires (NEW rev 9) ─────────────────────────────
logic               ctrl_bb_op_start;
logic [BB_AW-1:0]   ctrl_bb_op_base_addr;
logic               ctrl_bb_advance;

// =============================================================================
// Physical memory wires
// =============================================================================
logic [31:0]    wmem_rd_data_phys;

logic [FAW-1:0] fib_rd_addr;
logic           fib_rd_en;
logic [31:0]    fib_rd_data;

logic [OAW-1:0] omem_fb_rd_addr;
logic           omem_fb_rd_en;
logic [31:0]    omem_fb_rd_data;

// =============================================================================
// Feedback path mux (unchanged)
// =============================================================================
logic omem_fb_sel;
assign omem_fb_sel = omem_fb_en | ctrl_omem_fb_en;

logic omem_fb_sel_d;
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) omem_fb_sel_d <= 1'b0;
    else        omem_fb_sel_d <= omem_fb_sel;
end

always_comb begin
    fib_rd_addr     = '0; fib_rd_en     = 1'b0;
    omem_fb_rd_addr = '0; omem_fb_rd_en = 1'b0;
    if (!omem_fb_sel) begin
        fib_rd_addr = ctrl_imem_rd_addr[FAW-1:0];
        fib_rd_en   = ctrl_imem_rd_en;
    end else begin
        omem_fb_rd_addr = ctrl_imem_rd_addr;
        omem_fb_rd_en   = ctrl_imem_rd_en;
    end
end

assign ctrl_imem_rd_data = omem_fb_sel_d ? omem_fb_rd_data : fib_rd_data;

// =============================================================================
// MHA window-done strobe (unchanged)
// =============================================================================
logic omem_wr_en_d;
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) omem_wr_en_d <= 1'b0;
    else        omem_wr_en_d <= ctrl_omem_wr_en;
end
assign mha_window_done = (mode == 2'b10) && omem_wr_en_d && !ctrl_omem_wr_en;

// =============================================================================
// GCU start strobe (unchanged)
// =============================================================================
logic mmu_valid_in_d;
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) mmu_valid_in_d <= 1'b0;
    else        mmu_valid_in_d <= ctrl_mmu_valid_in;
end
assign gcu_start = (mode == 2'b10)
                 && (ctrl_mmu_op_code == 3'd5) && (ctrl_mmu_stage == 2'd0)
                 && mmu_valid_in_d && !ctrl_mmu_valid_in;

// =============================================================================
// Unified buffers → MMU bus wires
// =============================================================================
logic [7:0]  ubuf_w_out    [0:11][0:3];
logic [31:0] ubuf_bias_out;                 // retained but no longer drives MMU
logic [7:0]  ubuf_in_out   [0:11][0:6][0:3];

// bias_buffer output (NEW rev 9)
logic [31:0] bbuf_bias_out [0:6];           // 7 per-column biases → MMU
logic        bbuf_bias_ready;               // status: bias_reg valid

// =============================================================================
// MMU wires
// =============================================================================
logic        mmu_valid_out;
logic [7:0]  mmu_in_bus   [0:11][0:6][0:3];
logic [7:0]  mmu_w_bus    [0:11][0:3];
logic [31:0] mmu_bias_bus [0:6];            // ← [0:6] now (was [0:11])
logic [31:0] mmu_out      [0:6];

// =============================================================================
// Output buffer / post-processing wires (unchanged)
// =============================================================================
logic [31:0]        obuf_rd_data;
logic [31:0]        obuf_raw_rd_data;
logic signed [31:0] quant_data;
logic signed [31:0] relu_data;
logic signed [31:0] post_proc_data;

logic ctrl_ilb_wr_bypass;
assign ctrl_ilb_wr_bypass = (mode == 2'b10);

logic signed [7:0]  sbuf_shift_amt;

// =============================================================================
// MMU bus wiring
// ── Rev 9: mmu_bias_bus[0:6] driven by bias_buffer (bbuf_bias_out) ────────
// ── Rev 8: mmu_bias_bus was driven by ubuf_bias_out (scalar broadcast)
// =============================================================================
always_comb begin
    for (int p = 0; p < 12; p++) begin
        for (int t = 0; t < 4; t++)
            mmu_w_bus[p][t] = ubuf_w_out[p][t];
        for (int w = 0; w < 7; w++)
            for (int t = 0; t < 4; t++)
                mmu_in_bus[p][w][t] = ubuf_in_out[p][w][t];
    end
    // Per-column bias from bias_buffer
    for (int k = 0; k < 7; k++)
        mmu_bias_bus[k] = bbuf_bias_out[k];
end

// =============================================================================
// Instance: unified_controller  (rev 9 additions: BB ports)
// =============================================================================
unified_controller #(
    .WAW      (WAW),
    .FAW      (FAW),
    .OAW      (OAW),
    .W2_BASE  (9216),
    // MHA weight offsets
    .WQ_BASE   (10240),
    .WK_BASE   (19456),
    .WV_BASE   (28672),
    .WPROJ_BASE(37888),
    .WFFN1_BASE(47104),
    .WFFN2_BASE(56320),
    // MHA ILB base addresses
    .ILB_Q_BASE   (0),
    .ILB_K_BASE   (3072),
    .ILB_V_BASE   (6144),
    .ILB_S_BASE   (9216),
    .ILB_A_BASE   (16468),
    .ILB_PROJ_BASE(19540),
    .ILB_FFN1_BASE(20588),
    // Shift buffer parameters
    .SB_AW        (SB_AW),
    .SB_CONV_BASE (0),
    .SB_MLP_BASE  (5376),
    .SB_MHA_BASE  (5824),
    // ── Bias buffer parameters (NEW rev 9) ──────────────────────────────
    .BB_AW           (BB_AW),
    .BB_CONV_BASE    (0),
    .BB_MLP_L1_BASE  (96),
    .BB_MLP_L2_BASE  (480),
    .BB_MHA_QKT_BASE (576)
) u_ctrl (
    .clk                  (clk),
    .rst_n                (rst_n),
    .mode                 (mode),
    .start                (start),
    .done                 (done),

    .wmem_rd_addr         (ctrl_wmem_rd_addr),
    .wmem_rd_en           (ctrl_wmem_rd_en),
    .wmem_rd_data         (wmem_rd_data_phys),

    .wmem_shadow_wr_addr  (),
    .wmem_shadow_wr_en    (),
    .wmem_swap            (),

    .ext_weight_rd_addr   (),
    .ext_weight_rd_en     (),

    .imem_rd_addr         (ctrl_imem_rd_addr),
    .imem_rd_en           (ctrl_imem_rd_en),
    .imem_rd_data         (ctrl_imem_rd_data),

    .omem_wr_addr         (ctrl_omem_wr_addr),
    .omem_wr_en           (ctrl_omem_wr_en),

    .wbuf_load_en         (ctrl_wbuf_load_en),
    .wbuf_load_pe_idx     (ctrl_wbuf_load_pe_idx),
    .wbuf_load_k_word     (ctrl_wbuf_load_k_word),
    .wbuf_load_data       (ctrl_wbuf_load_data),
    .wbuf_bias_load_en    (ctrl_wbuf_bias_load_en),   // retained (see note D)
    .wbuf_bias_load_data  (ctrl_wbuf_bias_load_data), // retained (see note D)
    .wbuf_swap            (ctrl_wbuf_swap),

    .ibuf_load_en         (ctrl_ibuf_load_en),
    .ibuf_load_pe_idx     (ctrl_ibuf_load_pe_idx),
    .ibuf_load_win_idx    (ctrl_ibuf_load_win_idx),
    .ibuf_load_row        (ctrl_ibuf_load_row),
    .ibuf_load_k_word     (ctrl_ibuf_load_k_word),
    .ibuf_load_data       (ctrl_ibuf_load_data),
    .ibuf_swap            (ctrl_ibuf_swap),
    .ibuf_l1_capture_en   (ctrl_ibuf_l1_capture_en),
    .ibuf_l1_col_wr       (ctrl_ibuf_l1_col_wr),

    .ibuf_mha_load_en     (ctrl_ibuf_mha_load_en),
    .ibuf_mha_load_patch  (ctrl_ibuf_mha_load_patch),
    .ibuf_mha_load_k_word (ctrl_ibuf_mha_load_k_word),
    .ibuf_mha_load_data   (ctrl_ibuf_mha_load_data),
    .ibuf_mha_capture_row (ctrl_ibuf_mha_capture_row),

    .mmu_valid_in         (ctrl_mmu_valid_in),
    .mmu_op_code          (ctrl_mmu_op_code),
    .mmu_stage            (ctrl_mmu_stage),
    .mmu_sub_cycle        (ctrl_mmu_sub_cycle),

    .obuf_capture_en      (ctrl_obuf_capture_en),
    .obuf_rd_idx          (ctrl_obuf_rd_idx),

    .omem_fb_en_ctrl      (ctrl_omem_fb_en),

    // Shift buffer outputs
    .sb_op_start          (ctrl_sb_op_start),
    .sb_op_base_addr      (ctrl_sb_op_base_addr),
    .sb_advance           (ctrl_sb_advance),

    // ── Bias buffer outputs (NEW rev 9) ──────────────────────────────────
    .bb_op_start          (ctrl_bb_op_start),
    .bb_op_base_addr      (ctrl_bb_op_base_addr),
    .bb_advance           (ctrl_bb_advance),

    // Mask buffer ports (unchanged from rev 4)
    .qkt_store_done       (),
    .mask_next_window     (),
    .mask_valid           (1'b0),
    .mask_window_idx      ('0),
    .mask_all_done        (1'b0)
);

// =============================================================================
// Instance: weight_memory (unchanged)
// =============================================================================
weight_memory #(.AW(WAW)) u_wmem (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_addr (cpu_wmem_wr_addr),
    .wr_data (cpu_wmem_wr_data),
    .wr_en   (cpu_wmem_wr_en),
    .rd_addr (ctrl_wmem_rd_addr),
    .rd_en   (ctrl_wmem_rd_en),
    .rd_data (wmem_rd_data_phys)
);

// =============================================================================
// Instance: fib_memory (unchanged)
// =============================================================================
fib_memory u_fib (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_addr (cpu_fib_wr_addr),
    .wr_data (cpu_fib_wr_data),
    .wr_en   (cpu_fib_wr_en),
    .rd_addr (fib_rd_addr),
    .rd_en   (fib_rd_en),
    .rd_data (fib_rd_data)
);

// =============================================================================
// Instance: output_memory (unchanged from rev 8)
// =============================================================================
output_memory u_omem (
    .clk             (clk),
    .rst_n           (rst_n),
    .wr_addr         (ctrl_omem_wr_addr),
    .wr_data         (post_proc_data),
    .ilb_raw_wr_data (obuf_raw_rd_data),
    .ilb_wr_bypass   (ctrl_ilb_wr_bypass),
    .wr_en           (ctrl_omem_wr_en),
    .cpu_rd_addr     (cpu_omem_rd_addr),
    .cpu_rd_en       (cpu_omem_rd_en),
    .cpu_rd_data     (cpu_omem_rd_data),
    .fb_rd_addr      (omem_fb_rd_addr),
    .fb_rd_en        (omem_fb_rd_en),
    .fb_rd_data      (omem_fb_rd_data)
);

// =============================================================================
// Instance: unified_weight_buf (unchanged)
// =============================================================================
unified_weight_buf u_wbuf (
    .clk                 (clk),
    .rst_n               (rst_n),
    .mode                (mode[0]),
    .swap                (ctrl_wbuf_swap),

    .conv_load_en        ((mode == 2'b00) & ctrl_wbuf_load_en),
    .conv_load_pe_idx    (ctrl_wbuf_load_pe_idx),
    .conv_load_data      (ctrl_wbuf_load_data),
    .conv_bias_load_en   ((mode == 2'b00) & ctrl_wbuf_bias_load_en),
    .conv_bias_load_data (ctrl_wbuf_bias_load_data),

    .mlp_load_en         ((mode != 2'b00) & ctrl_wbuf_load_en),
    .mlp_load_k_word     (ctrl_wbuf_load_k_word),
    .mlp_load_data       (ctrl_wbuf_load_data),

    .sub_cycle           (ctrl_mmu_sub_cycle),
    .w_out               (ubuf_w_out),
    .bias_out            (ubuf_bias_out)    // no longer drives MMU (see note 5)
);

// =============================================================================
// Instance: unified_input_buf (unchanged)
// =============================================================================
unified_input_buf u_ibuf (
    .clk                 (clk),
    .rst_n               (rst_n),
    .mode                (mode),
    .swap                (ctrl_ibuf_swap),

    .conv_load_en        ((mode == 2'b00) & ctrl_ibuf_load_en),
    .conv_load_pe_idx    (ctrl_ibuf_load_pe_idx),
    .conv_load_win_idx   (ctrl_ibuf_load_win_idx),
    .conv_load_data      (ctrl_ibuf_load_data),

    .mlp_load_en         ((mode == 2'b01) & ctrl_ibuf_load_en),
    .mlp_load_row        (ctrl_ibuf_load_row),
    .mlp_load_k_word     (ctrl_ibuf_load_k_word),
    .mlp_load_data       (ctrl_ibuf_load_data),

    .mlp_capture_en      (ctrl_ibuf_l1_capture_en),
    .mlp_col_wr          (ctrl_ibuf_l1_col_wr),
    .mlp_l1_out          (mmu_out),

    .mha_load_en         (ctrl_ibuf_mha_load_en),
    .mha_load_patch      (ctrl_ibuf_mha_load_patch),
    .mha_load_k_word     (ctrl_ibuf_mha_load_k_word),
    .mha_load_data       (ctrl_ibuf_mha_load_data),
    .mha_capture_row     (ctrl_ibuf_mha_capture_row),

    .sub_cycle           (ctrl_mmu_sub_cycle),
    .data_out            (ubuf_in_out)
);

// =============================================================================
// Instance: mmu_top  (rev 2 — 7-bias interface)
// =============================================================================
mmu_top u_mmu (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in  (ctrl_mmu_valid_in),
    .op_code   (ctrl_mmu_op_code),
    .stage     (ctrl_mmu_stage),
    .valid_out (mmu_valid_out),
    .mmu_in    (mmu_in_bus),
    .mmu_w     (mmu_w_bus),
    .mmu_bias  (mmu_bias_bus),     // [0:6] — driven by bias_buffer
    .mmu_out   (mmu_out)
);

// =============================================================================
// Instance: output_buffer (unchanged from rev 8)
// =============================================================================
output_buffer u_obuf (
    .clk         (clk),
    .rst_n       (rst_n),
    .capture_en  (ctrl_obuf_capture_en),
    .mmu_out     (mmu_out),
    .rd_idx      (ctrl_obuf_rd_idx),
    .rd_data     (obuf_rd_data),
    .raw_rd_data (obuf_raw_rd_data)
);

// =============================================================================
// Instance: shift_buffer (unchanged from rev 7)
// =============================================================================
shift_buffer #(
    .DEPTH (16384),
    .DW    (8)
) u_sbuf (
    .clk             (clk),
    .rst_n           (rst_n),
    .cpu_wr_addr     (cpu_sbuf_wr_addr),
    .cpu_wr_data     (cpu_sbuf_wr_data),
    .cpu_wr_en       (cpu_sbuf_wr_en),
    .sb_op_start     (ctrl_sb_op_start),
    .sb_op_base_addr (ctrl_sb_op_base_addr),
    .sb_advance      (ctrl_sb_advance),
    .shift_amt       (sbuf_shift_amt)
);

// =============================================================================
// Instance: bias_buffer  (NEW rev 9)
// =============================================================================
bias_buffer #(
    .AW             (BB_AW),         // 12-bit address → 4096 entries
    .DW             (32),
    .BB_CONV_BASE   (0),
    .BB_MLP_L1_BASE (96),
    .BB_MLP_L2_BASE (480),
    .BB_MHA_QKT_BASE(576)
) u_bbuf (
    .clk              (clk),
    .rst_n            (rst_n),
    // CPU/DMA preload
    .cpu_wr_addr      (cpu_bbuf_wr_addr),
    .cpu_wr_data      (cpu_bbuf_wr_data),
    .cpu_wr_en        (cpu_bbuf_wr_en),
    // Controller
    .mode             (mode),
    .mmu_op_code      (ctrl_mmu_op_code),
    .bb_op_start      (ctrl_bb_op_start),
    .bb_op_base_addr  (ctrl_bb_op_base_addr),
    .bb_advance       (ctrl_bb_advance),
    // Status
    .bias_ready       (bbuf_bias_ready),   // internal; used by controller
    // Output
    .bias_out         (bbuf_bias_out)      // → mmu_bias_bus[0:6]
);

// =============================================================================
// Post-processing pipeline (unchanged from rev 8)
// =============================================================================
rounding_shifter #(.W_INPUT(32), .W_SHIFT(8)) u_quantizer (
    .in_value  ($signed(obuf_rd_data)),
    .shift_amt (sbuf_shift_amt),
    .out_value (quant_data)
);

relu #(.W(32)) u_relu (
    .in_value  (quant_data),
    .out_value (relu_data)
);

post_proc_mux #(.W(32)) u_mux (
    .relu_in  (relu_data),
    .quant_in (quant_data),
    .relu_en  (relu_en),
    .data_out (post_proc_data)
);

endmodule
// =============================================================================
// End of full_system_top.sv  (rev 9)
// =============================================================================