// =============================================================================
// bias_buffer.sv  (rev 1)
//
// Dedicated bias storage and delivery unit for the MMU's 7 bias inputs.
// Loaded from CPU/DMA before operation start, then autonomously streams
// the correct 7-wide bias vector to the MMU during computation.
//
// ── Memory Map (32-bit entry addresses) ───────────────────────────────────
//   Conv        [      0 ..     95]    96 entries   1 bias per output channel
//   MLP  L1     [     96 ..    479]   384 entries   1 bias per output column
//   MLP  L2     [    480 ..    575]    96 entries   1 bias per output column
//   MHA  QK^T   [    576 ..   2976]  2401 entries   1 bias per 49×49 element
//   Reserved    [   2977 ..   4095]   for future use
//
// ── Per-operation behaviour ────────────────────────────────────────────────
//
//   Conv (mode=2'b00):
//     • One 32-bit bias per output channel (96 channels total).
//     • All 7 bias_out[k] slots carry the SAME value (spatial broadcast):
//       the 7 MMU outputs are 7 spatial positions of the SAME channel.
//     • rd_ptr advances by 1 on each bb_advance pulse.
//     • bb_advance should be pulsed by the controller at the
//       "next kernel" state transition (once per completed 56×56 plane).
//
//   MLP (mode=2'b01):
//     • One bias per output column; 384 for L1, 96 for L2.
//     • bias_out[k] = bias_mem[rd_ptr + k], k = 0..6.
//     • rd_ptr advances by 7 on each bb_advance pulse.
//     • bb_advance fires at S_M_NEXT_ROW (same timing as sb_advance).
//     • The controller arms L1 base (96) at operation start and L2 base
//       (480) at the L1→L2 boundary using bb_op_start + bb_op_base_addr.
//
//   MHA (mode=2'b10):
//     • Bias is applied ONLY during the QK^T sub-operation
//       (mmu_op_code == 3'd3, the 49×49 attention score computation).
//       bias_out is driven to zero for QKV, S×V, PROJ, FFN1, FFN2.
//     • 2401 biases stored row-major: bias[r][c] at addr = 576 + r*49 + c.
//     • bias_out[k] = bias_mem[rd_ptr + k], k = 0..6.
//     • rd_ptr advances by 7 on each bb_advance pulse.
//     • bb_advance fires at S_H_NEXT_ATTN_COL (each attention column group).
//     • bb_op_start re-arms to BB_MHA_QKT_BASE at the beginning of each
//       7×7 window's QK^T phase so the 2401 biases are replayed per window.
//
// ── Load sequence ──────────────────────────────────────────────────────────
//   1. CPU/DMA writes all biases via cpu_wr_addr / cpu_wr_data / cpu_wr_en
//      before asserting the global start.  No operation ordering constraint;
//      the bus is 32-bit (one entry per write).
//   2. At operation start the controller asserts bb_op_start and presents the
//      matching bb_op_base_addr.  The buffer resets rd_ptr and transitions to
//      S_LOADING, filling bias_reg[0..6] over 7 consecutive clock cycles.
//   3. bias_ready is asserted when the 7-entry register bank is valid
//      (S_READY).  The controller must hold mmu_valid_in LOW until
//      bias_ready is seen (identical guard to the weight and shift buffers).
//   4. Each bb_advance pulse triggers a new 7-cycle reload of the register
//      bank.  bias_ready de-asserts during the reload and re-asserts when
//      the bank is refilled.  The controller weight-load phase provides
//      adequate slack for the 7-cycle reload.
//   5. A fresh bb_op_start (e.g. MLP L1→L2 transition, or MHA new window)
//      overrides any in-progress load and restarts from the new base.
//
// ── Ports ──────────────────────────────────────────────────────────────────
//   CPU/DMA      cpu_wr_addr, cpu_wr_data, cpu_wr_en
//   Controller   mode, mmu_op_code, bb_op_start, bb_op_base_addr, bb_advance
//   Status       bias_ready
//   MMU          bias_out[0:6]
//
// =============================================================================

module bias_buffer #(
    parameter int AW             = 12,   // address width → 4 096 entries max
    parameter int DW             = 32,   // data width in bits
    parameter int BB_CONV_BASE   = 0,    // start of Conv bias region
    parameter int BB_MLP_L1_BASE = 96,   // start of MLP L1 bias region
    parameter int BB_MLP_L2_BASE = 480,  // start of MLP L2 bias region
    parameter int BB_MHA_QKT_BASE= 576   // start of MHA QK^T bias region
)(
    input  logic           clk,
    input  logic           rst_n,

    // ── CPU / DMA preload port ─────────────────────────────────────────────
    // 32-bit bus; one entry per write cycle.
    input  logic [AW-1:0]  cpu_wr_addr,   // entry address (0 .. 4095)
    input  logic [DW-1:0]  cpu_wr_data,   // 32-bit bias value
    input  logic           cpu_wr_en,     // write strobe (active high)

    // ── Controller interface ───────────────────────────────────────────────
    input  logic [1:0]     mode,          // 2'b00=Conv 2'b01=MLP 2'b10=MHA
    input  logic [2:0]     mmu_op_code,   // forwarded from ctrl_mmu_op_code

    // bb_op_start: 1-cycle pulse to arm the buffer.
    //   • resets rd_ptr to bb_op_base_addr
    //   • clears the register bank and begins a fresh 7-cycle load
    //   • overrides any in-progress load
    input  logic           bb_op_start,
    input  logic [AW-1:0]  bb_op_base_addr, // entry base for the current op

    // bb_advance: 1-cycle pulse to step to the next group of bias values.
    //   • Conv     → called at "next kernel" controller state; step = 1
    //   • MLP/MHA  → called at column-group advance state;    step = 7
    //   Ignored while a load is already in progress (S_LOADING).
    input  logic           bb_advance,

    // ── Status ────────────────────────────────────────────────────────────
    output logic           bias_ready,    // 1 = bias_out is valid for MMU

    // ── MMU bias output ───────────────────────────────────────────────────
    output logic [DW-1:0]  bias_out [0:6] // directly fed to mmu_bias[0:6]
);

// =============================================================================
// Derived parameters
// =============================================================================
localparam int DEPTH = 1 << AW;   // 4096 32-bit entries
localparam int NOUT  = 7;         // MMU bias slots
localparam int LCNT_W = 3;        // ceil(log2(NOUT)) = 3

// =============================================================================
// Internal bias RAM
// =============================================================================
// Simple single-port SRAM model:
//   • CPU write (synchronous, priority over read on same address)
//   • Sequential read by internal load FSM
// The read port is registered to match typical SRAM behaviour.
// =============================================================================
logic [DW-1:0] bias_mem [0:DEPTH-1];
logic [DW-1:0] ram_rd_data;

always_ff @(posedge clk) begin
    if (cpu_wr_en)
        bias_mem[cpu_wr_addr] <= cpu_wr_data;
end

// Registered read — data valid one cycle after rd_addr is presented
logic [AW-1:0] ram_rd_addr;
always_ff @(posedge clk) begin
    ram_rd_data <= bias_mem[ram_rd_addr];
end

// =============================================================================
// Output register bank  (7 × 32-bit)
// Holds the currently active group of bias values for the MMU.
// =============================================================================
logic [DW-1:0] bias_reg [0:NOUT-1];

// =============================================================================
// Load base register
// Tracks the first entry address of the current (or next) bias group.
// =============================================================================
logic [AW-1:0] load_base;

// =============================================================================
// Read pointer
// Increments sequentially during S_LOADING.
// Reset to load_base at the start of every load phase.
// =============================================================================
logic [AW-1:0] rd_ptr;

// =============================================================================
// Load counter
// Counts 0..6 during the LOADING phase.
// Used to:
//   (a) select which bias_reg slot to write on the cycle rd_data is valid
//   (b) determine end-of-load (load_cnt_wr == NOUT-1 with registered read)
// Because the RAM has a 1-cycle read latency, we maintain two counters:
//   load_cnt_rd : drives ram_rd_addr (issues the read)
//   load_cnt_wr : drives bias_reg write-enable (captures the data)
// load_cnt_wr = load_cnt_rd delayed by 1 cycle.
// =============================================================================
logic [LCNT_W-1:0] load_cnt_rd;   // issues RAM reads  (0..6)
logic [LCNT_W-1:0] load_cnt_wr;   // captures RAM data (1..7, 7 = done)
logic              load_cnt_wr_vld;// load_cnt_wr in range 0..6 = bank write valid

// =============================================================================
// Advance step: how many entries rd_ptr jumps per bb_advance
//   Conv  → 1  (single broadcast bias per channel)
//   other → 7  (full column group of 7 biases)
// =============================================================================
logic [AW-1:0] adv_step;
assign adv_step = (mode == 2'b00) ? AW'(1) : AW'(NOUT);

// =============================================================================
// FSM
// =============================================================================
typedef enum logic [1:0] {
    S_IDLE    = 2'b00,   // waiting for first bb_op_start
    S_LOADING = 2'b01,   // 7-cycle sequential RAM read into bias_reg
    S_READY   = 2'b10    // bias_reg is valid; waiting for next bb_advance
} bb_state_t;

bb_state_t state;

// ── State transitions ─────────────────────────────────────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
    end else begin
        case (state)
            S_IDLE: begin
                if (bb_op_start)
                    state <= S_LOADING;
            end
            S_LOADING: begin
                // bb_op_start overrides in-progress load (re-arm)
                if (bb_op_start)
                    state <= S_LOADING;           // restart load
                else if (load_cnt_wr == LCNT_W'(NOUT-1) && load_cnt_wr_vld)
                    state <= S_READY;             // last write to bias_reg
            end
            S_READY: begin
                if (bb_op_start || bb_advance)
                    state <= S_LOADING;
            end
            default: state <= S_IDLE;
        endcase
    end
end

// ── load_base: first entry address of the current group ──────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        load_base <= AW'(BB_CONV_BASE);
    end else if (bb_op_start) begin
        load_base <= bb_op_base_addr;
    end else if (state == S_READY && bb_advance) begin
        // Advance base by step; the new load will start from here
        load_base <= load_base + adv_step;
    end
    // During S_LOADING load_base is frozen (new base is staged in rd_ptr)
end

// ── rd_ptr and load_cnt_rd ────────────────────────────────────────────────
// Manages the sequential RAM read address during S_LOADING.
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_ptr      <= AW'(BB_CONV_BASE);
        load_cnt_rd <= '0;
    end else if (bb_op_start) begin
        // (Re-)arm: set rd_ptr to the NEW base.
        // bb_op_base_addr takes priority over load_base (not yet updated).
        rd_ptr      <= bb_op_base_addr;
        load_cnt_rd <= '0;
    end else if (state == S_READY && bb_advance) begin
        // Begin next group from load_base + adv_step.
        // load_base hasn't updated yet (FF above is concurrent), so
        // compute next_base combinatorially here.
        rd_ptr      <= load_base + adv_step;
        load_cnt_rd <= '0;
    end else if (state == S_LOADING && load_cnt_rd != LCNT_W'(NOUT-1)) begin
        rd_ptr      <= rd_ptr + AW'(1);
        load_cnt_rd <= load_cnt_rd + LCNT_W'(1);
    end
    // Hold rd_ptr and load_cnt_rd after NOUT reads are issued.
end

// ── ram_rd_addr: combinatorial — registered inside bias_mem ──────────────
always_comb begin
    if (state == S_LOADING || (bb_op_start) ||
        (state == S_READY && bb_advance))
        ram_rd_addr = rd_ptr;
    else
        ram_rd_addr = rd_ptr;   // hold (harmless extra read)
end

// ── load_cnt_wr: delayed by one cycle relative to load_cnt_rd ────────────
// load_cnt_wr[k] valid when the RAM data for slot k has arrived.
logic [LCNT_W-1:0] load_cnt_rd_d;
logic              loading_d;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        load_cnt_rd_d <= '0;
        loading_d     <= 1'b0;
    end else begin
        load_cnt_rd_d <= load_cnt_rd;
        // loading_d is high one cycle after we enter/stay in S_LOADING
        loading_d     <= (state == S_LOADING) && !bb_op_start;
    end
end

assign load_cnt_wr     = load_cnt_rd_d;
assign load_cnt_wr_vld = loading_d;

// ── bias_reg: capture RAM data into the output bank ───────────────────────
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (int k = 0; k < NOUT; k++)
            bias_reg[k] <= '0;
    end else if (load_cnt_wr_vld) begin
        // Slot 0..6 filled as data arrives
        bias_reg[load_cnt_wr] <= ram_rd_data;
    end
end

// =============================================================================
// bias_ready
// =============================================================================
assign bias_ready = (state == S_READY);

// =============================================================================
// Output mux
//
// Conv (mode=2'b00)  : broadcast bias_reg[0] to all 7 outputs.
//                      (All 7 spatial positions of a channel share one bias.)
//
// MLP  (mode=2'b01)  : bias_out[k] = bias_reg[k], k=0..6.
//                      (Each of the 7 output columns gets its own bias.)
//
// MHA  (mode=2'b10)  : bias_out[k] = bias_reg[k] when mmu_op_code==3'd3
//                      (QK^T sub-operation only); zero for all other
//                      MHA sub-operations (QKV, S×V, PROJ, FFN1, FFN2).
// =============================================================================
always_comb begin
    for (int k = 0; k < NOUT; k++) begin
        unique case (mode)
            2'b00:   bias_out[k] = bias_reg[0];                      // Conv: broadcast
            2'b01:   bias_out[k] = bias_reg[k];                      // MLP: per-column
            2'b10:   bias_out[k] = (mmu_op_code == 3'd3)             // MHA: QK^T only
                                    ? bias_reg[k] : DW'(0);
            default: bias_out[k] = DW'(0);
        endcase
    end
end

endmodule
// =============================================================================
// End of bias_buffer.sv
// =============================================================================
