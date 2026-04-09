// =============================================================================
// fib_memory.sv  (rev 2 — MHA / Swin Transformer Block support)
//
// ── Address Map (time-shared, only one mode active at a time) ─────────────
//   Conv mode : CHW image, 224×224×3 pixels, 4 pixels/word
//               3 channels × (224×224/4) = 3 × 12,544 = 37,632 words
//               [0 .. 37,631]
//               Layout: ch*12544 + row*56 + word_in_row
//
//   MLP  mode : X input matrix, 3136 rows × 96 features, 4 bytes/word
//               3136 × 24 words/row = 75,264 words
//               [0 .. 75,263]
//               Layout: row*24 + k_word
//
//   MHA  mode : Feature map from Patch Embedding output, 56×56×96
//               Stored as 3136 patches × 96 features = 301,056 bytes
//               Packed 4 bytes/word → 75,264 words  (same footprint as MLP)
//               [0 .. 75,263]
//               Layout: patch_idx*24 + k_word
//               where patch_idx = win_row*7 + win_col  (within a 7×7 window),
//               and the FIB holds all 64 windows sequentially:
//               global_patch = win_idx*49 + patch_idx
//               addr = global_patch * 24 + k_word
//
//               NOTE: For MHA the FIB is READ ONLY during a transformer block
//               round (loaded once from external memory via DMA before start).
//               Intermediate QKV / attention / FFN results live in the ILB
//               (output_memory in this implementation).
//
//   Maximum depth = 75,264 words → AW = 17  (ceil log2 = 16.2, use 17)
//
// ── Why no resize for MHA ────────────────────────────────────────────────
//   MHA input (56×56 feature map, 96 channels, 4 B/elem) = 301,056 B
//   = 75,264 × 32-bit words, identical to the MLP footprint.  No change
//   to DEPTH or AW is required.
//
// ── Interface ────────────────────────────────────────────────────────────
//   Single write port  : loaded by DMA / CPU before engine start.
//   Single read port   : driven by the DSU on behalf of the active controller.
//   Read latency       : 1 cycle.
// =============================================================================

module fib_memory #(
    parameter DEPTH = 75264,    // words (MLP/MHA X is the larger case)
    parameter AW    = 17        // ceil(log2(75264)) = 17
)(
    input  logic          clk,
    input  logic          rst_n,

    // ── Write port (CPU / DMA) ────────────────────────────────────────────
    input  logic [AW-1:0] wr_addr,
    input  logic [31:0]   wr_data,
    input  logic          wr_en,

    // ── Read port (to DSU → controllers → input buffers) ──────────────────
    input  logic [AW-1:0] rd_addr,
    input  logic          rd_en,
    output logic [31:0]   rd_data
);

    logic [31:0] mem [0:DEPTH-1];

    // Initialise to zero (simulation only)
    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    // ── Write ─────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    // ── Read (registered, 1-cycle latency) ────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_data <= '0;
        else if (rd_en)
            rd_data <= mem[rd_addr];
    end

endmodule
