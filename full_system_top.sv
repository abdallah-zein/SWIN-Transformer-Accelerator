// =============================================================================
// full_system_top.sv  (rev 7 — shift_buffer added)
//
// ── What changed from rev 6 ───────────────────────────────────────────────
//   1. Removed:  input logic signed [7:0] quant_shift_amt
//      The per-element quantization shift amount is now driven entirely by the
//      new shift_buffer (u_sbuf) rather than a single static value from the
//      CPU.  The rounding_shifter receives shift_amt from u_sbuf.
//
//   2. Added three new top-level CPU/DMA write ports for the shift_buffer:
//        cpu_sbuf_wr_addr [11:0]  — word address  (= entry_addr >> 2)
//        cpu_sbuf_wr_data [31:0]  — four packed 8-bit shift values
//        cpu_sbuf_wr_en           — write-enable strobe
//      The CPU must preload the shift table corresponding to the intended
//      operation BEFORE asserting start.
//
//   3. The unified_controller gains three new outputs (see
//      unified_controller_sb_additions.sv for the patch):
//        sb_op_start      — resets shift_buffer read pointer to sb_op_base_addr
//        sb_op_base_addr  — 14-bit entry base address for the current operation
//        sb_advance       — step to next shift value after each 7-row group
//
//   4. shift_buffer u_sbuf is instantiated between the controller and the
//      rounding_shifter.  DEPTH = 16 384, DW = 8, RAM_AW = 12.
//
// ── Shift buffer memory map (for CPU preload) ─────────────────────────────
//   All addresses below are 8-bit entry addresses.
//   Divide by 4 to get the cpu_sbuf_wr_addr word address.
//
//   Conv         entries [    0 ..  5375]  96 kernels × 56 output rows
//   MLP          entries [ 5376 ..  5823]  448 row-groups (shared L1 & L2)
//   MHA/window   entries [ 5824 .. 13572]  7 749 per window:
//     QKV                                    2 016  (96 cols × 7 grp × 3 mat)
//     Attention (QKᵀ)                        1 029  (49 cols × 7 grp × 3 head)
//     SxV                                      672  (32 cols × 7 grp × 3 head)
//     W_proj                                   672  (96 cols × 7 grp)
//     FFN1                                   2 688  (384 cols × 7 grp)
//     FFN2                                     672  (96 cols × 7 grp)
//
// ── CPU preload sequence ──────────────────────────────────────────────────
//   (a) Set mode to the intended operation.
//   (b) Write shift values to cpu_sbuf_wr_addr/data/en.
//       Pack four consecutive 8-bit shift values into each 32-bit word.
//       Entry N occupies bits [(N%4)*8 +: 8] of word N/4.
//   (c) Assert start.  The controller fires sb_op_start on the same cycle,
//       arming the shift_buffer at the correct base address.
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
    // quant_shift_amt REMOVED in rev 7 — now driven by shift_buffer.
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
    // Preload the shift table before asserting start.
    // cpu_sbuf_wr_addr is a WORD address (= desired_entry_index >> 2).
    // Each 32-bit word packs four consecutive 8-bit signed shift values:
    //   bits [7:0]   → entry (addr×4 + 0)
    //   bits [15:8]  → entry (addr×4 + 1)
    //   bits [23:16] → entry (addr×4 + 2)
    //   bits [31:24] → entry (addr×4 + 3)
    input  logic [11:0] cpu_sbuf_wr_addr,  // 12-bit word addr → 4096 words → 16384 entries
    input  logic [31:0] cpu_sbuf_wr_data,
    input  logic        cpu_sbuf_wr_en,

    // ── MWU trigger (MHA only) ────────────────────────────────────────────
    output logic        mha_window_done,

    // ── GCU GELU control (MHA FFN1 output) ───────────────────────────────
    output logic        gcu_start,
    input  logic        gcu_done
);

// =============================================================================
// Localparams
// =============================================================================
localparam int WAW = 16;
localparam int FAW = 17;
localparam int OAW = 19;
localparam int SB_AW = 14;   // shift_buffer entry address width

// =============================================================================
// Unified controller output wires
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

// MHA ibuf ports
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

// ── Shift buffer controller wires (new in rev 7) ──────────────────────────
logic               ctrl_sb_op_start;
logic [SB_AW-1:0]   ctrl_sb_op_base_addr;
logic               ctrl_sb_advance;

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
// Feedback path mux
// =============================================================================
logic omem_fb_sel;
assign omem_fb_sel = omem_fb_en | ctrl_omem_fb_en;

logic omem_fb_sel_d;
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) omem_fb_sel_d <= 1'b0;
    else        omem_fb_sel_d <= omem_fb_sel;
end

always_comb begin
    fib_rd_addr    = '0; fib_rd_en    = 1'b0;
    omem_fb_rd_addr= '0; omem_fb_rd_en= 1'b0;
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
// MHA window-done strobe
// =============================================================================
logic omem_wr_en_d;
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) omem_wr_en_d <= 1'b0;
    else        omem_wr_en_d <= ctrl_omem_wr_en;
end
assign mha_window_done = (mode == 2'b10) && omem_wr_en_d && !ctrl_omem_wr_en;

// =============================================================================
// GCU start strobe
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
logic [7:0]  ubuf_w_out   [0:11][0:3];
logic [31:0] ubuf_bias_out;
logic [7:0]  ubuf_in_out  [0:11][0:6][0:3];

// =============================================================================
// MMU wires
// =============================================================================
logic        mmu_valid_out;
logic [7:0]  mmu_in_bus   [0:11][0:6][0:3];
logic [7:0]  mmu_w_bus    [0:11][0:3];
logic [31:0] mmu_bias_bus [0:11];
logic [31:0] mmu_out      [0:6];

// =============================================================================
// Output buffer / post-processing wires
// =============================================================================
logic [31:0]        obuf_rd_data;
logic signed [31:0] quant_data;
logic signed [31:0] relu_data;
logic signed [31:0] post_proc_data;

// ── Shift buffer → quantizer ──────────────────────────────────────────────
logic signed [7:0]  sbuf_shift_amt;   // drives rounding_shifter.shift_amt

// =============================================================================
// MMU bus wiring
// =============================================================================
always_comb begin
    for (int p = 0; p < 12; p++) begin
        for (int t = 0; t < 4; t++)
            mmu_w_bus[p][t] = ubuf_w_out[p][t];
        mmu_bias_bus[p] = (p == 0) ? ubuf_bias_out : 32'd0;
        for (int w = 0; w < 7; w++)
            for (int t = 0; t < 4; t++)
                mmu_in_bus[p][w][t] = ubuf_in_out[p][w][t];
    end
end

// =============================================================================
// Instance: unified_controller
// — Three new SB parameters and three new SB output ports added in rev 7.
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
    // ── Shift buffer parameters (NEW rev 7) ──────────────────────────────
    .SB_AW        (SB_AW),    // 14 — matches shift_buffer DEPTH=16384
    .SB_CONV_BASE (0),        // Conv shift table base entry
    .SB_MLP_BASE  (5376),     // MLP shift table base entry
    .SB_MHA_BASE  (5824)      // MHA per-window shift table base entry
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
    .wbuf_bias_load_en    (ctrl_wbuf_bias_load_en),
    .wbuf_bias_load_data  (ctrl_wbuf_bias_load_data),
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

    // ── Shift buffer outputs (NEW rev 7) ─────────────────────────────────
    .sb_op_start          (ctrl_sb_op_start),
    .sb_op_base_addr      (ctrl_sb_op_base_addr),
    .sb_advance           (ctrl_sb_advance)
);

// =============================================================================
// Instance: weight_memory
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
// Instance: fib_memory
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
// Instance: output_memory (also ILB for MHA)
// =============================================================================
output_memory u_omem (
    .clk         (clk),
    .rst_n       (rst_n),
    .wr_addr     (ctrl_omem_wr_addr),
    .wr_data     (post_proc_data),
    .wr_en       (ctrl_omem_wr_en),
    .cpu_rd_addr (cpu_omem_rd_addr),
    .cpu_rd_en   (cpu_omem_rd_en),
    .cpu_rd_data (cpu_omem_rd_data),
    .fb_rd_addr  (omem_fb_rd_addr),
    .fb_rd_en    (omem_fb_rd_en),
    .fb_rd_data  (omem_fb_rd_data)
);

// =============================================================================
// Instance: unified_weight_buf
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
    .bias_out            (ubuf_bias_out)
);

// =============================================================================
// Instance: unified_input_buf
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
// Instance: mmu_top  (UNTOUCHED — no changes in rev 7)
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
    .mmu_bias  (mmu_bias_bus),
    .mmu_out   (mmu_out)
);

// =============================================================================
// Instance: output_buffer
// =============================================================================
output_buffer u_obuf (
    .clk        (clk),
    .rst_n      (rst_n),
    .capture_en (ctrl_obuf_capture_en),
    .mmu_out    (mmu_out),
    .rd_idx     (ctrl_obuf_rd_idx),
    .rd_data    (obuf_rd_data)
);

// =============================================================================
// Instance: shift_buffer  (NEW in rev 7)
//
// Holds up to 16 384 pre-loaded 8-bit shift values (4 packed per 32-bit word).
// The controller drives sb_op_start / sb_op_base_addr to arm the buffer at the
// correct table base whenever a new operation or MHA window begins.
// sb_advance steps the read pointer by one after each 7-element row-group
// writeback so shift_amt is always valid for the upcoming row-group.
// =============================================================================
shift_buffer #(
    .DEPTH (16384),    // 16 384 entries → 4 096 × 32-bit words → 12-bit word addr
    .DW    (8)         // must match rounding_shifter W_SHIFT
) u_sbuf (
    .clk             (clk),
    .rst_n           (rst_n),
    // CPU preload port
    .cpu_wr_addr     (cpu_sbuf_wr_addr),   // 12-bit word address
    .cpu_wr_data     (cpu_sbuf_wr_data),
    .cpu_wr_en       (cpu_sbuf_wr_en),
    // Controller
    .sb_op_start     (ctrl_sb_op_start),
    .sb_op_base_addr (ctrl_sb_op_base_addr),
    .sb_advance      (ctrl_sb_advance),
    // Output
    .shift_amt       (sbuf_shift_amt)
);

// =============================================================================
// Post-processing pipeline
//
// rounding_shifter.shift_amt now comes from u_sbuf rather than the old
// top-level quant_shift_amt port (removed in rev 7).
// =============================================================================
rounding_shifter #(.W_INPUT(32), .W_SHIFT(8)) u_quantizer (
    .in_value  ($signed(obuf_rd_data)),
    .shift_amt (sbuf_shift_amt),          // ← driven by shift_buffer (was quant_shift_amt)
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
