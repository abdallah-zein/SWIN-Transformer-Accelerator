# Bias Buffer — Design Document  (rev 1)

## 1. Overview

The Bias Buffer (`bias_buffer.sv`) is a dedicated on-chip memory unit that stores
all bias parameters for the three supported operations and delivers the correct
7-element bias vector to the MMU on every compute cycle.

It replaces the scalar bias path that previously lived inside `unified_weight_buf`
and expands it to support independent per-column biases — a requirement for MLP
and MHA operations.

---

## 2. Bias Requirements per Operation

| Operation    | # Biases | Bits Total | Distribution Rule                         |
|--------------|----------|------------|-------------------------------------------|
| Conv         | 96       | 3 072 b    | 1 bias per output channel; broadcast to all 7 MMU outputs |
| MLP Layer 1  | 384      | 12 288 b   | 1 bias per output column; 7 different values per MMU cycle |
| MLP Layer 2  | 96       | 3 072 b    | 1 bias per output column; 7 different values per MMU cycle |
| MHA QK^T     | 2 401    | 76 832 b   | 1 bias per element of 49×49 attention matrix (row-major)   |
| MHA others   | 0        | —          | Biases zeroed for QKV, S×V, PROJ, FFN1, FFN2              |

---

## 3. Internal Memory Map

Stored in a 4096-entry × 32-bit RAM (12-bit address, `AW=12`):

```
Address  0 ..   95  →  Conv  (96 entries)
Address  96 ..  479  →  MLP Layer 1  (384 entries)
Address 480 ..  575  →  MLP Layer 2   (96 entries)
Address 576 .. 2976  →  MHA QK^T  (2 401 entries, row-major: bias[r][c] @ 576+r*49+c)
Address 2977 .. 4095 →  Reserved
```

Total active entries: **2 977**.  RAM depth (4 096) leaves 1 119 reserved words
for future expansion.

---

## 4. Module Interface

```
bias_buffer #(
    .AW              = 12,    // address width
    .DW              = 32,    // data width
    .BB_CONV_BASE    = 0,
    .BB_MLP_L1_BASE  = 96,
    .BB_MLP_L2_BASE  = 480,
    .BB_MHA_QKT_BASE = 576
)
```

### CPU / DMA preload port
| Signal            | Dir | Width | Description                              |
|-------------------|-----|-------|------------------------------------------|
| `cpu_wr_addr`     | In  | 12    | Entry address (0–4095)                   |
| `cpu_wr_data`     | In  | 32    | Bias value to write                      |
| `cpu_wr_en`       | In  | 1     | Write strobe (active-high, synchronous)  |

### Controller interface
| Signal             | Dir | Width | Description                              |
|--------------------|-----|-------|------------------------------------------|
| `mode`             | In  | 2     | 00=Conv, 01=MLP, 10=MHA                  |
| `mmu_op_code`      | In  | 3     | Forwarded from `ctrl_mmu_op_code`        |
| `bb_op_start`      | In  | 1     | 1-cycle arm pulse; resets rd_ptr and begins 7-cycle load |
| `bb_op_base_addr`  | In  | 12    | Entry base address for current op segment |
| `bb_advance`       | In  | 1     | 1-cycle pulse; step to next bias group   |

### Status
| Signal       | Dir | Width | Description                              |
|--------------|-----|-------|------------------------------------------|
| `bias_ready` | Out | 1     | 1 = output register bank is valid        |

### MMU output
| Signal         | Dir | Width     | Description                          |
|----------------|-----|-----------|--------------------------------------|
| `bias_out[0:6]`| Out | 7 × 32   | Per-column biases → `mmu_bias[0:6]` |

---

## 5. Datapath Architecture

```
 CPU/DMA ──[cpu_wr_addr/data/en]──► bias_mem [4096×32]
                                        │
                              [sequential read, 1-cycle latency]
                                        │
                                   bias_reg[7×32]  ◄── loaded during S_LOADING
                                        │
                                   Output Mux
                                   ┌────┴─────────────────────────────────┐
                                   │ mode=Conv  → broadcast bias_reg[0]   │
                                   │ mode=MLP   → bias_reg[k], k=0..6     │
                                   │ mode=MHA   → bias_reg[k] if QK^T     │
                                   │              else 0                   │
                                   └────┬─────────────────────────────────┘
                                        │
                                   bias_out[0:6] ──► mmu_bias[0:6]
```

---

## 6. FSM

Three states:

```
          bb_op_start
S_IDLE ──────────────► S_LOADING ──(7 cycles)──► S_READY
   ▲                       │  ▲                      │
   │                       │  │ bb_op_start (re-arm)  │ bb_advance
   │                       └──┘                      │ or bb_op_start
   └─────────────────────────────────────────────────┘
```

- **S_IDLE**: Waiting for first `bb_op_start`.
- **S_LOADING**: Reads 7 consecutive entries from `bias_mem` into `bias_reg[0..6]`
  over 7 clock cycles (accounts for 1-cycle RAM read latency).
  `bias_ready = 0` during this state.
- **S_READY**: `bias_reg` valid; `bias_out` driven to MMU.
  `bias_ready = 1`.

`bb_op_start` in S_LOADING triggers an immediate restart (override in-progress load).

---

## 7. Load Mechanism & Timing

The internal RAM has a **1-cycle registered read latency**.  Two counters are used:

- `load_cnt_rd` — drives `ram_rd_addr` (issues reads ahead)
- `load_cnt_wr` — `load_cnt_rd` delayed by 1 cycle (captures arriving data into `bias_reg`)

Timeline for a 7-cycle load starting at base address `B`:

```
Cycle │  load_cnt_rd │ ram_rd_addr │ load_cnt_wr │  bias_reg written
──────┼──────────────┼─────────────┼─────────────┼───────────────────
  0   │      0       │      B      │    (—)      │  —
  1   │      1       │     B+1     │     0       │  bias_reg[0] ← mem[B]
  2   │      2       │     B+2     │     1       │  bias_reg[1] ← mem[B+1]
  3   │      3       │     B+3     │     2       │  bias_reg[2] ← mem[B+2]
  4   │      4       │     B+4     │     3       │  bias_reg[3] ← mem[B+3]
  5   │      5       │     B+5     │     4       │  bias_reg[4] ← mem[B+4]
  6   │      6       │     B+6     │     5       │  bias_reg[5] ← mem[B+5]
  7   │  (hold)      │    (hold)   │     6       │  bias_reg[6] ← mem[B+6]
      │              │             │  → S_READY  │  bias_ready asserts
```

Total latency: **8 cycles** from `bb_op_start` to `bias_ready`.

---

## 8. Advance Step per Operation Mode

| Mode      | `adv_step` | Trigger state in controller    |
|-----------|------------|-------------------------------|
| Conv      | **1**      | "next kernel" state (once per 56×56 output plane) |
| MLP L1/L2 | **7**      | `S_M_NEXT_ROW` (same as `sb_advance`) |
| MHA QK^T  | **7**      | `S_H_NEXT_ATTN_COL` (same as `sb_advance` for attention) |

Conv uses step=1 because only one channel advances at a time.  The 7-entry
register bank still loads 7 consecutive entries (harmless reads of future channels),
but only `bias_reg[0]` is broadcast to the MMU.

---

## 9. Output Mux Logic

```systemverilog
// Conv:  all 7 slots carry same bias (broadcast)
// MLP:   each slot carries its own column bias
// MHA:   per-column bias only during QK^T (op_code==3); zero otherwise
for (int k = 0; k < 7; k++) begin
    unique case (mode)
        2'b00: bias_out[k] = bias_reg[0];
        2'b01: bias_out[k] = bias_reg[k];
        2'b10: bias_out[k] = (mmu_op_code == 3'd3) ? bias_reg[k] : 32'h0;
    endcase
end
```

---

## 10. Changes to Existing Modules

### mmu.sv → rev 2
- `mmu_bias [0:11]` changed to `mmu_bias [0:6]`
- All PE_block instances receive `bias = 32'd0`
- Bias added **after** adder_tree: `mmu_out[i] = tree_sum[i] + mmu_bias[i]`
- New wire `tree_sum [0:6]` added

### mmu_top.sv → rev 2
- `mmu_bias` port width: `[0:11]` → `[0:6]`
- No other changes

### full_system_top.sv → rev 9
- New top-level ports: `cpu_bbuf_wr_addr [11:0]`, `cpu_bbuf_wr_data [31:0]`, `cpu_bbuf_wr_en`
- `mmu_bias_bus` width: `[0:11]` → `[0:6]`
- `mmu_bias_bus` now driven by `bbuf_bias_out[0:6]` (was `ubuf_bias_out` scalar)
- New wires: `ctrl_bb_op_start`, `ctrl_bb_op_base_addr [11:0]`, `ctrl_bb_advance`
- New instance: `bias_buffer u_bbuf`
- `unified_controller` gains three new output connections

### unified_controller.sv → rev 5 (patch)
New parameters: `BB_AW`, `BB_CONV_BASE`, `BB_MLP_L1_BASE`, `BB_MLP_L2_BASE`, `BB_MHA_QKT_BASE`

New output ports: `bb_op_start`, `bb_op_base_addr [BB_AW-1:0]`, `bb_advance`

New `always_comb` block `bias_buf_ctrl`:

| Condition                                           | Action                                               |
|-----------------------------------------------------|------------------------------------------------------|
| `S_IDLE && start && mode==2'b00`                    | `bb_op_start=1`, `bb_op_base_addr=BB_CONV_BASE`      |
| `S_IDLE && start && mode==2'b01`                    | `bb_op_start=1`, `bb_op_base_addr=BB_MLP_L1_BASE`    |
| `S_M_L1_NEXT_COL && m_last_l1_col`                  | `bb_op_start=1`, `bb_op_base_addr=BB_MLP_L2_BASE`    |
| `S_IDLE && start && mode==2'b10`                    | `bb_op_start=1`, `bb_op_base_addr=BB_MHA_QKT_BASE`   |
| `S_H_NEXT_WINDOW && !h_last_win`                    | `bb_op_start=1`, `bb_op_base_addr=BB_MHA_QKT_BASE`   |
| Conv "next kernel" state                            | `bb_advance=1`                                       |
| `S_M_NEXT_ROW`                                      | `bb_advance=1`                                       |
| `S_H_NEXT_ATTN_COL`                                 | `bb_advance=1`                                       |

> **Note on Conv bb_advance**: The exact state depends on the Conv FSM.
> Use `state == S_C_NEXT_KERNEL` if that state exists, otherwise use
> `state == S_C_NEXT && c_last_chunk && (c_row_grp == C_N_ROW_GROUPS-1)`.

---

## 11. CPU Preload Protocol

```
(a) Set mode to the intended operation.
(b) Write bias values to cpu_bbuf_wr_addr/data/en:
      Conv  : write 96 words  to addresses   0..95
      MLP   : write 480 words to addresses  96..575  (L1 then L2 contiguous)
      MHA   : write 2401 words to addresses 576..2976 (row-major 49×49)
(c) Assert start.  The controller fires bb_op_start on the same cycle,
    arming the bias_buffer at the correct base address.
(d) The controller must not assert mmu_valid_in until bias_ready is HIGH
    (8-cycle latency from bb_op_start).
```

---

## 12. Area & Timing Estimates

| Resource      | Estimate         | Notes                                 |
|---------------|------------------|---------------------------------------|
| SRAM          | 4096 × 32 bits   | ≈ 16 KB; one single-port SRAM macro   |
| Register bank | 7 × 32 bits      | 224 flip-flops                        |
| Counters      | 3 × 12 bits      | rd_ptr, load_base, load_cnt (36 FFs)  |
| FSM           | 2 bits           | 3-state                               |
| Output mux    | 7 × 2:1 mux      | trivially small                       |
| Critical path | RAM read + adder | bias_mem read → adder in mmu.sv       |

The 8-cycle preload latency is well within the weight-load slack of every
operation (Conv: ~200 cycles, MLP: ~50 cycles, MHA: ~40 cycles per group).
