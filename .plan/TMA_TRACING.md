# TMA Tracing — Discovery and Next Steps

## Context

This document records the findings from inspecting a real FlashAttention-3 (sm_90, H100) hardware trace on a Blackwell system. It covers the TMA instructions found in the trace, the current tracing infrastructure's limitations, and the options for capturing TMA descriptor values.

Trace examined:
```
flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24
backend: 12.8
device: sm_90 (H100)
```

---

## What TMA Is

**TMA** (Tensor Memory Accelerator) is a hardware unit introduced in NVIDIA H100 (sm_90) that performs asynchronous, coalesced memory transactions for multi-dimensional tensor data. It is the hardware primitive that FlashAttention-3 uses to load Q/K/V tiles without blocking the warp scheduler.

Key properties:
- TMA transactions are **asynchronous** — the warp fires the TMA and continues executing while the transaction proceeds in the memory system
- TMA uses a **256-bit descriptor** that encodes the full 4D tensor address, dimensions, strides, and cache behavior
- TMA issues **separate arrive-barrier transactions** so individual tile loads can be independently synchronized
- TMA bypasses the standard L1/L2 miss path and uses a dedicated TLB and transaction pipeline

TMA is central to FlashAttention-3's performance because it overlaps Q/K/V tile loading with computation. Without modeling TMA correctly, the simulator's cycle estimate for attention kernels will systematically mispredict memory latency hiding.

---

## TMA Instructions Found in the Trace

Both FlashAttention kernels (forward and backward) contain TMA instructions:

### FlashAttention Forward (Kernel 5)

| Opcode | Count | Purpose |
|---|---|---|
| `UTMALDG.4D` | 18 | Async 4D tensor load via TMA |
| `UTMACCTL.PF` | 8 | TMA control / prefetch |
| `UTMASTG.5D` | 2 | Async 5D tensor store |
| `UTMACMDFLUSH` | 2 | TMA cache flush |
| `UTMAPF.L2.4D` | 2 | TMA L2 prefetch |
| `USETMAXREG.TRY_ALLOC.CTAPOOL` | 1 | CTA pool register allocation |
| `USETMAXREG.DEALLOC.CTAPOOL` | 1 | CTA pool register deallocation |

### FlashAttention Backward (Kernel 10)

| Opcode | Count | Purpose |
|---|---|---|
| `UTMALDG.4D` | 20 | Async 4D tensor load |
| `UTMACMDFLUSH` | 14 | TMA cache flush |
| `UTMACCTL.PF` | 12 | TMA control / prefetch |
| `UTMASTG.4D` | 4 | Async 4D tensor store |
| `USETMAXREG.TRY_ALLOC.CTAPOOL` | 1 | CTA pool register allocation |
| `USETMAXREG.DEALLOC.CTAPOOL` | 1 | CTA pool register deallocation |

---

## UTMALDG.4D — Operand and Control Bit Details

The `UTMALDG.4D` instruction is the primary async load primitive. Example from CTA (0,0,0), warp 0, PC=0x94e0:

### Operand Format

```
UTMALDG.4D  [URdest, ...], [URsrc, ...], desc[URdesc]
```

| Field | Value | Meaning |
|---|---|---|
| Destination regs | `UR16`, `UR8` | Registers receiving the loaded 4D tile |
| Source/addr reg | `[UR8]`, `[UR16]` | Register holding tensor address/coords |
| Descriptor reg | `desc[UR18]` | 256-bit TMA descriptor (in descriptor cache) |

### Control Bits

```
stall_count:           1-2     Warp stall cycles before TMA issue
is_new_read_barrier:   True    Fires an arrive-barrier after issue
is_new_write_barrier:  False
id_new_read_barrier:   1/2/3   Which BARRIER.arrive this TMA fires
id_new_write_barrier:  7       Fixed write barrier
```

The `id_new_read_barrier` field is critical: each TMA load fires a **separate arrive transaction** with a distinct barrier ID (1, 2, 3...). This is how FlashAttention pipelines multiple Q/K/V tile loads simultaneously — different warps wait on different barrier IDs to synchronize on individual tile completions rather than a single global barrier.

### Register Usage

| Register class | Count | Purpose |
|---|---|---|
| Regular (GR) | 14–21 | Working registers for tensor data |
| Uniform (UR) | 18–20 | Addressing, descriptor, metadata |
| Predicate | 3–4 | Lane masking for the 4D tile |

The high uniform register count reflects the complex 4D address calculation. TMA addresses are not simple pointer+offset — they require mapping 4D tensor coordinates to a flat memory address, all computed in uniform registers before the TMA transaction is initiated.

---

## Current Tracing Infrastructure

### What the Tracer Captures

The NVBit-based tracer (`tracer_nvbit/tracer_tool/`) captures per-instruction:

```
threadblock_id (x, y, z)
warp_id
PC (virtual offset)
active_mask / predicate_mask
unique_function_id   ← maps to kernel variant in enhanced_execution_info.json
memory addresses     ← compressed: base+stride OR base+delta OR list-all
memory width
opcode               ← derived from PC + unique_function_id
```

The `inst_trace_t` struct (in `common.h`) records:
```c
typedef struct {
    int cta_id_x, cta_id_y, cta_id_z;
    int warpid_tb;
    uint64_t addrs_or_reg_val_0[32];  // memory addresses, NOT reg values
    uint32_t reg_id;                   // register NUMBER, not value
    // ...
    uint32_t per_operand_type;          // TRACED_REG_TYPE enum
} inst_trace_t;
```

The `DESC` operand type is recognized in the enhanced tracer's JSON output (`operand_type: 'DESC'`), but the tracer only records **which register** holds the descriptor (`desc[UR18]`), not **the contents** of that register.

### Where the Gap Is

The TMA descriptor is a 256-bit value stored in the GPU's **hardware descriptor cache**. The tracer cannot read it because:

1. NVBit instruments SASS instructions at runtime but has no hook to read GPU hardware state (descriptor cache contents)
2. The descriptor is set up by `DMNP`/`TMMA` instructions that write to a special hardware structure, not to memory
3. No current field in `inst_trace_t` or the protobuf schema carries the 256-bit descriptor value
4. The `reg_id` field records only the register number, not the register's value at trace time

---

## Three Options to Capture TMA Descriptors

### Option 1 — Decode TMA Descriptors from CUBIN Binary (Recommended)

**How it works:** The TMA descriptor is a 256-bit immediate value encoded directly in `DMNP` (define null pointer + descriptor) instructions in the CUBIN. The SASS disassembly (`cuobjdump -sass`) contains this value as an operand.

**Steps:**
1. Disassemble each kernel's CUBIN:
   ```bash
   cuobjdump -sass kernel.sm_90.cubin > kernel.sm_90.sass
   ```
2. Find all `DMNP` instructions — they encode the full 256-bit TMA descriptor as immediate operands
3. Parse the descriptor fields (see below)
4. Add a `tma_descriptors` section to `enhanced_execution_info.json`, keyed by kernel `unique_function_id`
5. Extend the protobuf schema to carry TMA descriptor metadata

**Descriptor 256-bit format** (from NVIDIA ISA reference for sm_90):

| Bits | Field | Description |
|---|---|---|
| 255:192 | `ADDR[63:0]` | 64-bit byte address |
| 191:176 | `DIM[47:32]` | 3rd dimension |
| 175:160 | `DIM[31:16]` | 2nd dimension |
| 159:144 | `DIM[15:0]` | 1st dimension |
| 143:128 | `STRIDE[47:32]` | 3rd stride |
| 127:112 | `STRIDE[31:16]` | 2nd stride |
| 111:96 | `STRIDE[15:0]` | 1st stride |
| 95:88 | `MISC` | Element size, swizzle, etc. |
| 87:80 | `BOX_DIM` | 4D/3D box dimensions |
| 79:0 | `RESERVED` | |

**Limitation:** Only captures statically initialized descriptors. Any descriptor constructed dynamically at runtime would be missed.

### Option 2 — Instrument Descriptor-Setting Instructions in NVBit

**How it works:** Extend the tracer to capture register values written by `DMNP`/`TMMA` instructions at runtime.

**Changes required:**

In `tracer_tool.cu` → `instrument_function_if_needed()`:
```c
if (std::string(instr->getOpcode()).find("DMNP") != std::string::npos ||
    std::string(instr->getOpcode()).find("TMMA") != std::string::npos) {
    // Capture the full 256-bit immediate operand value
    // Add field to inst_trace_t: uint64_t tma_desc[4];
}
```

In `common.h` → extend `inst_trace_t`:
```c
typedef struct {
    // ... existing fields ...
    uint64_t tma_desc[4];    // 256-bit TMA descriptor
    bool has_tma_desc;
} inst_trace_t;
```

In `threadblock.proto` → extend `instruction` message:
```protobuf
message TmaDescriptor {
    uint64 addr = 1;
    repeated uint64 dims = 2;
    repeated uint64 strides = 3;
    uint32 element_size = 4;
    // ...
}

message Instruction {
    // ... existing fields ...
    TmaDescriptor tma_desc = 6;
}
```

**Limitation:** Requires rebuilding `tracer_tool.so`. Requires identifying all descriptor-construction instructions. NVBit can read register values at instrument time, but only for registers that are operands of the instrumented instruction.

### Option 3 — Use NVIDIA Nsight Compute for TMA Metrics

**How it works:** Run the kernel under `ncu` to get aggregate TMA statistics.

```bash
ncu --metrics \
    sm__tma_transactions.sum, \
    sm__tma_transactions_per_request.average, \
    sm__tma_hit_rate, \
    sm__tma_non_posted_transactions, \
    sm__pipeline_utilization.tma \
    --kernel-name "flash.*fwd" \
    ./flash_attention_kernel
```

This gives aggregate per-kernel TMA metrics:
- Total TMA transaction count and bytes
- Average transactions per TMA request
- TMA cache hit rate
- TMA pipeline utilization

**Limitation:** Aggregate per-kernel statistics only. No per-CTA, per-warp, or per-instruction descriptor data. Useful for validation but not for cycle-accurate simulation.

---

## Why TMA Descriptors Matter for CTA Sampling

The current CTA sampling system assumes per-SM issue throughput is constant or scales with a simple log function as CTAs/SM increases. TMA-based memory access breaks this assumption in two ways:

### 1. TMA Has Different Latency Hiding Behavior

TMA transactions are initiated by one warp and complete asynchronously while other warps execute. This means:
- TMA latency is **not hidden by additional CTAs** on the same SM — it is hidden by other warps within the same CTA
- Adding more CTAs to fill idle SMs does **not** improve TMA transaction throughput — the bottleneck is warp-level parallelism within each CTA, not CTA-level concurrency
- The log-fit model `T(N) = a + b*log(N+1)` was calibrated on workloads where memory latency is hidden by additional concurrent CTAs — this mechanism does not apply to TMA

### 2. TMA Descriptor Cache Pressure

Each CTA using TMA holds descriptors in the per-SM TMA descriptor cache. If multiple CTAs share an SM, they compete for the descriptor cache. At high CTA/SM density, descriptor cache misses may cause TMA to fall back to a slower path, creating a nonlinear throughput cliff.

---

## CTA Sampling Implications

The presence of TMA in the FlashAttention trace has direct implications for the CTA sampling system on `cta-sampling`:

| Issue | Impact |
|---|---|
| Log-fit model wrong for TMA kernels | TMA latency hiding is intra-CTA (warp-level), not inter-CTA. The log-fit calibrated on concurrent CTAs does not apply. |
| Memory pressure signals are biased | TMA bypasses the standard L1/L2 miss path. `dram_bytes`, `achieved_bw_ratio`, and `mem_stall_frac` from standard stall counters do not capture TMA transaction behavior. |
| K-rep collapse is worse for TMA kernels | FlashAttention grids are often 1D or 2D. The corner+midpoint heuristic collapses to very few unique CTAs — and if those CTAs happen to map to the same TMA tile group, the "full kernel" pilot expansion just replicates the same TMA access pattern. |
| Pilot loop doubling doesn't reach TMA saturation | Pilot doubles CTA count but TMA throughput is not a function of CTA count. The force-expand condition (`full_ctas_per_sm > 4 * sampled_ctas_per_sm`) would fire but for the wrong reason — the TMA system is not bottlenecked by CTA concurrency. |

---

## Recommended Next Steps

1. **Parse `DMNP` instructions from existing SASS files.** The `extra_info/sass/` directory already contains SASS disassembly for all kernels. Write a parser to extract TMA descriptors from `DMNP` immediate operands and add them to `enhanced_execution_info.json`.

2. **Extend the protobuf** to carry a `tma_descriptor` field on the `instruction` message, populated for `UTMALDG.*` / `UTMASTG.*` instructions.

3. **Add TMA-specific pressure signals** to `pressure_signals_t`: TMA transaction count, bytes transferred, and TMA hit rate, sourced from the remodeled SM's TMA unit model.

4. **Add a TMA-aware classifier flag** — if a kernel uses TMA, bypass the log-fit model and use a separate TMA-specific cycle model (e.g., warp-occupancy-based rather than CTA-concurrency-based).

---

## Files Referenced

| File | Role |
|---|---|
| `util/tracer_nvbit/tracer_tool/tracer_tool.cu` | NVBit tracer implementation |
| `util/tracer_nvbit/tracer_tool/common.h` | `inst_trace_t` struct definition |
| `util/traces_enhanced/pb_trace/instruction.proto` | Protobuf for per-instruction trace data |
| `util/traces_enhanced/pb_trace/threadblock.proto` | Protobuf for per-CTA data |
| `traces/extra_info/sass/*.sass` | SASS disassembly (contains `DMNP` encoding) |
| `traces/extra_info/enhanced_execution_info.json` | Kernel instruction metadata |
| `traces/threadblocks/device_0/.../*.pb` | Per-CTA trace data |
| `traces/dynamic_trace.pb` | Top-level trace metadata |

---

## Trace Examined

Path:
```
simulator-remodeled/hw_run/traces/device-0/12.8/
  flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/
  _home_jihyun_project_accorde_scripts_flashattn_runner_py___variant_fa3___
   dtype_bf16___direction_bwd___causal_true___batch_1___seq_len_2048___
   head_dim_64___heads_24___warmup_0___iters_1/traces/
```

Key stats:
- Hardware: sm_90 (H100)
- Total kernels: 11
- Total CTAs traced: 10,911 (across all kernels)
- FlashAttention Fwd: 132 CTAs × 512 threads
- FlashAttention Bwd: 384 CTAs × 384 threads
- Dominant instruction types: control-flow/sync (>85% of instructions), TMA memory ops (~2% but ~100% of memory traffic)
