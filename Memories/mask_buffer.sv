// =============================================================================
// mask_buffer.sv
//
// Stores 49 × 32-bit attention-bias mask values for the Swin Transformer
// shifted-window masked attention (SWMSA / WMSA) operation.
//
// ── Overview ─────────────────────────────────────────────────────────────────
// The QK^T attention score matrix for one attention head is 49 × 49.
// It is conceptually tiled into a 7 × 7 grid of non-overlapping 7 × 7
// sub-blocks (windows), giving 49 sub-blocks in total.  One 32-bit scalar
// mask value is broadcast-added to every element of each sub-block before
// the softmax and S×V steps.  This module tracks which sub-block is being
// processed and presents the correct mask word on the bus.
//
// ── Operational phases ───────────────────────────────────────────────────────
//
//  1. LOAD  (one-time initialisation, before inference starts)
//     The CPU / DMA writes all 49 mask entries one word at a time via the
//     mb_wr_en / mb_wr_addr / mb_wr_data port.  Content is constant for the
//     entire inference run — it depends only on the shift configuration of
//     the Swin window, not on the image data.
//
//  2. IDLE  — mask_valid = 0.
//
//  3. MASK PASS  (one pass per attention head, per 7 × 7 window)
//     Triggered by a one-cycle pulse on qkt_store_done.  The controller
//     asserts this signal in state S_H_NEXT_ATTN_ROWGRP the cycle the last
//     word of a head's 49 × 49 score matrix is committed to the ILB.
//
//     Behaviour after qkt_store_done:
//       • rd_ptr resets to 0.
//       • mask_valid rises to 1.
//       • mask_data_out / mask_window_idx are stable for the entire duration
//         of processing sub-block window 0 (49 elements).
//       • When the controller finishes the read-modify-write of those 49
//         elements it pulses mask_next_window.  rd_ptr increments to 1,
//         mask_data_out updates to mask_mem[1], etc.
//       • After the pulse for the last window (rd_ptr == 48):
//           – mask_all_done fires for one cycle.
//           – active clears; mask_valid returns to 0.
//           – The controller may then proceed to S_H_NEXT_ATTN_HD.
//
// ── Signal summary ────────────────────────────────────────────────────────────
//
//  INPUTS — load port
//    mb_wr_en        write enable (1 → load one mask word)
//    mb_wr_addr[5:0] destination address (0 .. 48)
//    mb_wr_data[31:0]mask value (32-bit signed bias, e.g. 0 or -1e9)
//
//  INPUTS — control (from unified_controller)
//    qkt_store_done  1-cycle pulse: head's 49×49 QK^T fully stored in ILB
//    mask_next_window1-cycle pulse: current sub-block's 49 elements processed
//
//  OUTPUTS — to controller + datapath
//    mask_data_out[31:0]  bias for the current 7×7 sub-block (broadcast)
//    mask_window_idx[5:0] current sub-block index (0 .. 48)
//    mask_valid           1 while a mask pass is active
//    mask_all_done        1-cycle pulse after the 49th mask_next_window
//
// ── Timing diagram (abbreviated) ─────────────────────────────────────────────
//
//   clk         _/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_ ...
//   qkt_store_done  ‾‾‾|___________________________
//   mask_valid          __|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|__
//   mask_window_idx      [0    ][1    ][2    ]...[48]
//   mask_next_window         __|    |__|    |...  |__
//   mask_all_done                                |__
//
// ── Connections in full_system_top (new) ─────────────────────────────────────
//   mask_buffer u_mask_buffer (
//       .clk             (clk),
//       .rst_n           (rst_n),
//       .mb_wr_en        (mb_wr_en),
//       .mb_wr_addr      (mb_wr_addr),
//       .mb_wr_data      (mb_wr_data),
//       .qkt_store_done  (ctrl_qkt_store_done),     // new ctrl output
//       .mask_next_window(ctrl_mask_next_window),   // new ctrl output
//       .mask_data_out   (mask_data_out),
//       .mask_window_idx (mask_window_idx),
//       .mask_valid      (mask_valid),              // new ctrl input
//       .mask_all_done   (mask_all_done)            // new ctrl input
//   );
//
//   The 32-bit sum (QK^T score + mask) is formed OUTSIDE this module by an
//   adder placed between output_memory.fb_rd_data and output_memory.ilb_raw_wr_data:
//
//       assign masked_s_word = fb_rd_data + mask_data_out;
//
//   During states S_H_MASK_RD / S_H_MASK_WB the controller sets
//   ilb_wr_bypass = 1 so output_memory writes masked_s_word, not wr_data.
//
// =============================================================================

module mask_buffer (
    input  logic        clk,
    input  logic        rst_n,

    // ── Load port  ────────────────────────────────────────────────────────────
    input  logic        mb_wr_en,
    input  logic [5:0]  mb_wr_addr,          // 0 .. 48
    input  logic [31:0] mb_wr_data,

    // ── Control (from unified_controller) ─────────────────────────────────────
    // qkt_store_done:  one-cycle pulse asserted in S_H_NEXT_ATTN_ROWGRP when
    //                  h_last_attn_rowgrp is true (last row-group of a head's
    //                  49×49 QK^T matrix has just been written to the ILB).
    // mask_next_window:one-cycle pulse asserted in S_H_MASK_NEXT_WIN after the
    //                  controller finishes the 49-element R-M-W for sub-block w.
    input  logic        qkt_store_done,
    input  logic        mask_next_window,

    // ── Mask output ───────────────────────────────────────────────────────────
    output logic [31:0] mask_data_out,      // mask bias for current sub-block
    output logic [5:0]  mask_window_idx,    // sub-block index (0 .. 48)
    output logic        mask_valid,         // 1 during active mask pass
    output logic        mask_all_done       // 1-cycle pulse after window 48 done
);

    // =========================================================================
    // 49 × 32-bit mask register file
    // =========================================================================
    // Synthesis note: 49 words × 32 bits = 1 568 bits → mapped to registers
    // (not block RAM) because it is small and needs single-cycle random read.
    // =========================================================================
    logic [31:0] mask_mem [0:48];

    // ── Load (write) port ────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (mb_wr_en)
            mask_mem[mb_wr_addr] <= mb_wr_data;
    end

    // =========================================================================
    // State: active flag + read pointer
    // =========================================================================
    logic       active;
    logic [5:0] rd_ptr;    // 0 .. 48

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            rd_ptr <= 6'd0;
        end else begin
            // ── New mask pass ────────────────────────────────────────────────
            // qkt_store_done has priority: it can restart a pass even if one
            // is somehow still in progress (defensive).
            if (qkt_store_done) begin
                active <= 1'b1;
                rd_ptr <= 6'd0;
            end
            // ── Advance window ───────────────────────────────────────────────
            // Only act on mask_next_window when actually active.
            // Ignore stray pulses during IDLE.
            else if (active && mask_next_window) begin
                if (rd_ptr == 6'd48) begin
                    // All 49 sub-blocks processed — end of mask pass
                    active <= 1'b0;
                    rd_ptr <= 6'd0;
                end else begin
                    rd_ptr <= rd_ptr + 6'd1;
                end
            end
        end
    end

    // =========================================================================
    // Combinational outputs
    // =========================================================================

    // mask_valid: high throughout the mask pass (windows 0 .. 48)
    assign mask_valid = active;

    // mask_window_idx: sub-block currently being written back
    assign mask_window_idx = rd_ptr;

    // mask_data_out: combinational read — always presents the current mask
    // word.  Glitch-free because rd_ptr is registered and mask_mem is stable.
    assign mask_data_out = mask_mem[rd_ptr];

    // mask_all_done: fires exactly one cycle — the same cycle as the
    // mask_next_window pulse for the very last sub-block (rd_ptr == 48).
    // The controller uses this to leave S_H_MASK_NEXT_WIN and enter
    // S_H_NEXT_ATTN_HD.
    assign mask_all_done = active && (rd_ptr == 6'd48) && mask_next_window;

endmodule
