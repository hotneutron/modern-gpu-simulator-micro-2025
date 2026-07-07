# TMA ISA Support Status (FA3 Trace)

## Scope

This note is for **FlashAttention-3 (FA3)** trace investigation on Hopper.

Trace path checked:

`/home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/hw_run/traces/device-0/12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/traces/dynamic_trace.pb`

Related files checked:

- `.../traces/extra_info/tma_discovery.json`
- `.../traces/extra_info/nvdisasm/*.nvdisasm`
- `.../traces/extra_info/enhanced_execution_info.json` (exists, very large)

## Operand Semantics — What We Actually Recovered (base / dest / size)

> This section supersedes the earlier "operand 1/2/3 are tuple-like" guesses.
> Meanings below are **verified against real FA3 runtime operands**
> (`tma_runtime_operand_debug.jsonl`), host `cuTensorMapEncodeTiled` dumps
> (`tensor_map_encode_dump.csv`), and SASS def-chains. Full derivation and the
> exact-base recovery pipeline live in
> [.plan/TMA_BASE_ADDR.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.plan/TMA_BASE_ADDR.md)
> and
> [.plan/TMA_exact_base_mapping_integration.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.plan/TMA_exact_base_mapping_integration.md).

**Two operand worlds (crucial):**
- The **descriptor operand** `desc[URx]` is NOT the address. Its register pair is
  `{lo=0, hi=desc_value_hi}` where `desc_value_hi` is a compile-time immediate
  (`0x14f00000` / `0x12f00000` / `0x10000000`) — a coarse **tensor-map slot tag**,
  only 3 values for 7 tensors, never appears in the host encode blob. It cannot
  identify a base by itself.
- The **runtime GMEM address is never fully in the operand**: `first_lane_addr`
  keeps only the low 32 bits (`value_hi=0x85` does not merge in). So no op's
  operand value is directly usable as the real GMEM dest.

**Two base kinds (what "synthetic base" means):**
- **real base** — the tensor's actual GMEM start address, read from the 128B
  descriptor's `qword0`. Repeated accesses to the same tensor resolve to the same
  address, so L2 reuse matches HW.
- **synthetic base** — a **fabricated placeholder address, NOT the real GMEM
  address**. When no descriptor is available the mover uses
  `(transfer_uid << 20) + agu_index*128`, where `transfer_uid` is a per-transfer
  serial number. It has no relation to where the tensor really lives; every
  transfer lands at a different address so **L2 never hits on it**. Its only
  purpose is to give the memory hierarchy *something* to time (latency/bandwidth);
  it deliberately does not model address locality. Note the **transfer size stays
  exact** (from operand covered_bytes) even when the base is synthetic — only the
  *where* is fake, not the *how much*.

**How the real base is actually recovered** (per `(uid,pc)`, done in
`build_tma_pc_base_map.py`): the 128B `CUtensorMap` is passed **by value** as a
kernel argument, so it physically sits in the param block. We match the encoded
128B blob into the param struct byte-for-byte, keyed by the **SMEM-window offset
(low16 = tensor ID)** for load/store, or the **prologue `param_base+off`
(`off−0x30` = struct slot)** for the prefetch-control chain. `qword0` of that
matched 128B = the tensor's real GMEM base. Result on FA3: **23/23 descriptor
sites exact, 0 pool, 0 unresolved.**

| opcode | desc ref? | dest (GMEM base) source | size source | mapped to real base? |
|---|---|---|---|---|
| **UTMALDG.4D** | yes (`desc[URx]`, SMEM window) | operand ci=0 SMEM-window low16 = tensor ID → base map → **descriptor qword0** | descriptor `Πbox_dim·element_size` | **yes (exact)** |
| **UTMASTG.4D/.5D** | yes (SMEM window) | same as UTMALDG (ci=0 SMEM window → base map) | descriptor box | **yes (exact)** |
| **UTMAPF.L2.4D** | no (`desc_valid=false`) | prologue `param_base+off`, `off−0x30` = struct slot → **descriptor qword0** (prefetch target) | descriptor box | **yes (exact)** — fwd only |
| **UTMACCTL.PF** | no (`desc_valid=false`) | prologue `param_base+off` (off∈{0xb0,0x170,0x230,0x2f0,0x4f0,0x5b0}) → struct slot → descriptor qword0 | descriptor box (prefetch control) | **yes (exact)** |
| **UBLKRED.G.S.ADD.F32.RN** | yes but **bare handle** (`desc_value_hi=0x14f00000`, NOT a tensormap) | dest = raw dQaccum GMEM ptr (`c[0x0][0x280]/0x2a0`) + dynamic tile offset — NOT in a tensormap | operand ci=2 `span·16B` (covered_bytes, e.g. `0x400·16=16384B`) | **no** — base stays synthetic (M2.5 raw-ptr deref); size exact |
| **UBLKCP.S.G** | no (`desc_valid=false`) | dst=SMEM (ci=0 `0xff..`), src=GMEM (ci=1, low32) — not a tensormap | operand ci=2 `span·16B` (e.g. `0x20·16=512B`) | **no** — synthetic base; size exact |
| **UTMACMDFLUSH** | no | none (control only) | none (no data request) | n/a |

**Key correction (was wrong before):** UBLKRED was first read as "operand ci=1 is
the GMEM address." Re-checked on data + SASS: its `desc[URx]` is a **bare handle**
(`UMOV UR16,0x0` + `UMOV UR17,0x14f00000`), not a CUtensorMap; the real destination
is a **raw GMEM pointer** from constant bank (`c[0x0][0x280]/0x2a0`) plus a dynamic
tile offset. So UBLKRED/UBLKCP get **synthetic base (correct size)** in M1; exact
base needs raw-pointer dereference (M2.5), which static tensormap mapping cannot do.

**Simulator integration:** the base is looked up in `build_tma_command`
(the only place `(uid,pc)` is still live), stored in `TMACommand.global_base`, and
carried to `mover_issue_requests`. Gated by `-tma_real_base_addr_enable`. All 11
descriptor fields (`tensor_rank, tensor_data_type, global_dim[5],
global_strides[5], box_dim[5], element_strides[5], interleave, swizzle,
l2_promotion, oob_fill, element_size`) now come from the exact matched 128B, so
the old `handle_hi → config_id` heuristic is superseded.

## Ops Seen In This FA3 Trace

From `tma_discovery.json` (`tma_family_opcode_counts`):

- `UTMALDG.4D` = 4068
- `UTMACCTL.PF` = 777
- `UBLKCP.S.G` = 387
- `UTMACMDFLUSH` = 372
- `UBLKRED.G.S.ADD.F32.RN` = 336
- `UTMASTG.5D` = 102
- `UTMAPF.L2.4D` = 72
- `UTMASTG.4D` = 24

Additional Hopper ops observed in FA3 `nvdisasm`:

- `USETMAXREG.TRY_ALLOC.CTAPOOL`, `USETMAXREG.DEALLOC.CTAPOOL`
- `ACQBULK`
- `FENCE.VIEW.ASYNC.S`
- `HGMMA.*`

## Current Simulator Handling Status

> **Status note (updated):** the "Gap / blocker — no real TMA semantics" rows below
> are **outdated**. FA3 traces now run: TMA ops are decoded, issued to a dedicated
> `tma_unit_sm`, and generate real 128B-sector memory requests through the
> interconnect/L2. The table's original "no runtime handling in `trace_driven.cc`"
> wording described the earliest state; the runtime path now lives in
> `remodeling/tma_unit_sm.cc` (`build_tma_command` → `mover_issue_requests`). See
> the per-op corrections in "Operation Explanations" below.

Trace-driven decode/execution path reference:

- `gpu-simulator/trace-driven/trace_driven.cc`
- Hopper map: `gpu-simulator/ISA_Def/hopper_opcode.h`
- **TMA runtime unit: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc`**

| Opcode family in FA3 trace | Decode category now         | Runtime handling status in remodeled simulator                                                                                                |
| -------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `UTMALDG.*`                | `LOAD_OP`                   | **Modeled**. Issued to `tma_unit_sm`; real base from `tma_pc_base_map.json` (exact), size from descriptor box, 128B-sector requests to L2.     |
| `UTMAPF.*`                 | `LOAD_OP`                   | **Modeled (prefetch)**. Exact base (prologue chain), descriptor box size; warms L2 (fwd only in FA3).                                          |
| `UTMASTG.*`                | `STORE_OP`                  | **Modeled**. Issued to `tma_unit_sm`; exact base (SMEM window), descriptor box size, write-side requests.                                      |
| `UBLKCP.*`                 | `LOAD_OP`                   | **Modeled (synthetic base)**. Bulk copy, not a tensormap; size = operand covered_bytes (exact), base synthetic.                                |
| `UBLKRED.*`                | `STORE_OP`                  | **Modeled (synthetic base)**. Bulk reduction; desc is a bare handle, dest is a raw GMEM ptr; size = covered_bytes (exact), base synthetic.     |
| `UTMACCTL.*`               | `MEMORY_MISCELLANEOUS_OP`   | **Modeled (control)**. Prefetch-control; exact descriptor recoverable via prologue chain (used to link the prefetch target tensor).           |
| `UTMACMDFLUSH`             | `MEMORY_MISCELLANEOUS_OP`   | **Stubbed**. Control-only; no data request (correct — nothing to model beyond timing).                                                          |
| `USETMAXREG.*`             | `UNIFORM_OP`                | **Stubbed**. Decoded and issued as generic uniform op; no CTA-pool/register-budget semantics modeled.                                         |
| `ACQBULK`                  | `MISCELLANEOUS_NO_QUEUE_OP` | **Stubbed**. Decoded and issued as generic misc op; no explicit bulk-acquire semantics modeled.                                               |
| `FENCE.*`                  | `MEMORY_BARRIER_OP`         | **Implemented (basic)**. Uses existing memory-barrier path (membar wait behavior), not TMA-specific logic.                                    |
| `HGMMA.*`                  | `TENSOR_CORE_OP`            | **Implemented (basic)**. Uses tensor-core op path; not a TMA op.                                                                              |

## Code Links

- **`UTMALDG.*`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [missing special-case switch](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L375-L525), [crash path](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L621-L662)
- **`UTMAPF.*`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [missing special-case switch](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L375-L525), [crash path](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L621-L662)
- **`UTMASTG.*`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [missing special-case switch](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L375-L525), [crash path](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L621-L662)
- **`UBLKCP.*`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [missing special-case switch](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L375-L525), [crash path](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L621-L662)
- **`UBLKRED.*`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [missing special-case switch](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L375-L525), [crash path](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L621-L662)
- **`UTMACCTL.*`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [memory-unit routing](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L917-L925), [generic mem-op latency](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L950-L964)
- **`UTMACMDFLUSH`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [memory-unit routing](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L917-L925), [generic mem-op latency](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L950-L964)
- **`USETMAXREG.*`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [uniform routing](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L893-L895)
- **`ACQBULK`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [misc routing](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L911-L915)
- **`FENCE.*`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [memory-unit routing](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L917-L925), [membar wait](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1208-L1228)
- **`HGMMA.*`**: [decode](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L290-L306), [tensor routing](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L896-L898)

## Operation Explanations

### `UTMALDG.4D`

Example seen in checked FA3 files:

- `UTMALDG.4D [UR16], [UR8], desc[UR10]`
- `UTMALDG.4D [UR16], [UR6], desc[UR14]`

Operand shape:

- 3 operands
- operand 1: `[URdst]` — **SMEM destination tuple** (where the tile lands in shared memory)
- operand 2: `[URsrc]` — **coordinate/address tuple** (runtime value is a SMEM-window pointer; low16 = tensor ID used to look up the base)
- operand 3: `desc[URdesc]` — **descriptor handle** `{lo=0, hi=desc_value_hi}`; a coarse slot tag, NOT the base

Simple meaning:

- Hopper TMA async tensor load (GMEM → SMEM)
- the HW reads the 128B descriptor to get `base + Σ coord·stride`; the base is
  recovered offline via the by-value descriptor in the param block (see the
  Operand Semantics section)

Current simulator handling:

- decoded as `LOAD_OP`, issued to `tma_unit_sm`
- **real base** from `tma_pc_base_map.json` (exact, gated by `-tma_real_base_addr_enable`)
- **size** = descriptor `Πbox_dim·element_size`; emits `⌈box·elem/128⌉` 128B-sector requests
- M2 visit-counter tile spread scatters transfers across the tensor's tiles

### `UTMAPF.L2.4D`

Example seen in checked FA3 files:

- `UTMAPF.L2.4D [UR28], [UR8]`

Operand shape:

- 2 operands
- operand 1: `[URa]` — SMEM/control tuple
- operand 2: `[URb]` — address tuple (SMEM-window pointer)
- **no `desc[URx]` operand** (`desc_valid=false`); the descriptor is reached via
  the prologue `param_base+off` chain, not an operand

Simple meaning:

- Hopper TMA prefetch toward L2 (`.L2`), tensor-region style (`.4D`)
- warms L2 with the tile before the matching `UTMALDG` consumes it
- **present in fwd (uid3) only**; the bwd kernel has no `UTMAPF`

Current simulator handling:

- decoded as `LOAD_OP`, issued to `tma_unit_sm` as a prefetch
- **real base** recovered via the prologue chain (`off−0x30` = struct slot →
  descriptor qword0), so the prefetch target tensor is known exactly
- size = descriptor box; warms L2 like a load

### `UTMASTG.4D` / `UTMASTG.5D`

Examples seen in checked FA3 files:

- `UTMASTG.4D [UR8], [UR6]`
- `UTMASTG.5D [UR8], [UR16]`

Operand shape:

- 2 operands
- operand 1: `[URa]` — GMEM-side tuple (SMEM-window pointer; low16 = tensor ID → base map)
- operand 2: `[URb]` — SMEM source tuple
- `desc[URx]` present and valid (`desc_valid=true`), SMEM-window handle like UTMALDG

Simple meaning:

- Hopper TMA async tensor store (SMEM → GMEM)
- `.4D` vs `.5D` = tensor-rank/layout variant (FA3 uses both)

Current simulator handling:

- decoded as `STORE_OP`, issued to `tma_unit_sm`
- **real base** from `tma_pc_base_map.json` (exact, SMEM-window path)
- size = descriptor box; emits write-side 128B-sector requests

### `UBLKCP.S.G`

Example seen in checked FA3 files:

- `UBLKCP.S.G [UR28], [UR6], UR9`

Operand shape:

- 3 operands
- operand 1: `[URa]` — **SMEM destination** (ci=0, `0xff..` window)
- operand 2: `[URb]` — **GMEM source** (ci=1, low-32 bits only)
- operand 3: `URc` — **covered_bytes / size control** (ci=2 span; e.g. `0x20·16=512B`)
- **no tensormap** (`desc_valid=false`)

Simple meaning:

- Hopper bulk copy (`.S.G` = SMEM ← GMEM direction variant)
- plain address/size operands, NOT a CUtensorMap-described transfer

Current simulator handling:

- decoded as `LOAD_OP`, issued to `tma_unit_sm`
- **synthetic base** — a fabricated placeholder address (NOT the real GMEM
  address; see "Two base kinds"), because there is no tensormap to recover a base
  from; **size = operand covered_bytes (exact)**
- `⌈covered_bytes/128⌉` requests

### `UBLKRED.G.S.ADD.F32.RN`

Example seen in checked FA3 files:

- `UBLKRED.G.S.ADD.F32.RN [UR10], [UR4], UR12, desc[UR16]`

Operand shape:

- 4 operands
- operand 1: `[URa]` — GMEM destination tuple (raw pointer, low-32 only at runtime)
- operand 2: `[URb]` — SMEM source tuple
- operand 3: `URc` — **covered_bytes / size** (ci=2 span; e.g. `0x400·16=16384B`)
- operand 4: `desc[URd]` — **bare handle, NOT a CUtensorMap** (`desc_value_hi=0x14f00000`, `lo=0`)

Simple meaning:

- Hopper bulk reduction (`ADD.F32.RN`): SMEM tile is reduced into GMEM
- **Correction:** the `desc[URd]` here is a compile-time bare handle
  (`UMOV UR16,0x0` + `UMOV UR17,0x14f00000`), **not** a tensormap. The real
  destination is a **raw GMEM pointer** from the constant bank
  (`c[0x0][0x280]/0x2a0` = dQaccum) plus a dynamic tile offset — so there is no
  128B descriptor to read a base from.

Current simulator handling:

- decoded as `STORE_OP`, issued to `tma_unit_sm`
- **synthetic base** in M1 — a fabricated placeholder address (NOT the real GMEM
  address; see "Two base kinds"). Static tensormap mapping is impossible here
  (proven); exact base needs raw-pointer deref + tile coords (M2.5, separate task)
- **size = operand covered_bytes (exact)**; `⌈covered_bytes/128⌉` requests
- excluded from the base-map strict gate (correctly "not tensormap-addressed":
  the 1:1 diagnostic reports 0/9)

### `UTMACCTL.PF`

Example seen in checked FA3 files:

- `UTMACCTL.PF [UR6]`

Operand shape:

- 1 operand
- operand 1: `[URx]` — control pointer (SMEM-window); **no `desc` operand** (`desc_valid=false`)

Simple meaning:

- Hopper TMA prefetch-control (`.PF`): sets up / issues descriptor prefetch state
- in FA3 it copies the 128B descriptor from the param struct into a SMEM window
  that later `UTMALDG` sites read (the "chain" source in base recovery)

Current simulator handling:

- decoded as `MEMORY_MISCELLANEOUS_OP`, routed through the memory unit
- **descriptor recoverable exactly** via the prologue chain: `param_base+off`
  (off∈{0xb0,0x170,0x230,0x2f0,0x4f0,0x5b0}), `off−0x30` = struct slot →
  descriptor qword0. Used to (a) link chain-mapped `UTMALDG` sites and (b) emit
  the prefetch target tensor per pc (6→6 clean, single-source, no ambiguity)

### `UTMACMDFLUSH`

Example seen in checked FA3 files:

- `UTMACMDFLUSH`

Operand shape:

- 0 operands

Simple meaning:

- Hopper TMA metadata/cache flush control op

Current simulator handling:

- decoded as `MEMORY_MISCELLANEOUS_OP`
- routed and timed as a generic memory-misc op
- no explicit TMA flush behavior is modeled
- runnable, but effectively stubbed

### `USETMAXREG.TRY_ALLOC.CTAPOOL` / `USETMAXREG.DEALLOC.CTAPOOL`

Examples seen in checked FA3 files:

- `USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xf0`
- `USETMAXREG.DEALLOC.CTAPOOL 0x18`

Operand shape:

- `TRY_ALLOC` form: 2 operands
- `DEALLOC` form: 1 operand

Simple meaning:

- Hopper CTA-pool register allocation / deallocation control
- these ops change register-budget state rather than perform normal ALU or memory work

Current simulator handling:

- base mnemonic decodes as `UNIFORM_OP`
- issued as a generic uniform datapath instruction
- no CTA-pool or register-budget semantics are modeled
- runnable, but effectively stubbed

### `ACQBULK`

Example seen in checked FA3 files:

- `ACQBULK`

Operand shape:

- 0 operands

Simple meaning:

- Hopper bulk acquire / control-style operation

Current simulator handling:

- decoded as `MISCELLANEOUS_NO_QUEUE_OP`
- issued through generic miscellaneous path
- no explicit bulk-acquire semantics are modeled
- runnable, but effectively stubbed

### `FENCE.VIEW.ASYNC.S`

Example seen in checked FA3 files:

- `FENCE.VIEW.ASYNC.S`

Operand shape:

- 0 operands in the observed form

Simple meaning:

- Hopper memory barrier / ordering instruction
- not a TMA op, but relevant because it appears in FA3 kernels

Current simulator handling:

- decoded as `MEMORY_BARRIER_OP`
- uses existing membar wait path in the simulator
- has basic barrier behavior rather than being unsupported

### `HGMMA.*`

Examples seen in checked FA3 files:

- `HGMMA.64x96x16.F32.BF16 ...`
- `HGMMA.64x256x16.F32.BF16 ...`

Operand shape:

- variable rich tensor-core operand form
- examples include accumulator register, descriptor-like operand, source operand, and optional modifiers

Simple meaning:

- Hopper warpgroup/tensor-core matrix multiply-accumulate family
- not a TMA op, but present in the same FA3 kernels

Current simulator handling:

- decoded as `TENSOR_CORE_OP`
- sent through the tensor-core execution path
- basic execution path exists, even if it is not a full Hopper-specific model

## Practical Summary For FA3 Runnability

**Updated:** the five ops below are now **modeled and runnable** in trace-driven
mode via `tma_unit_sm` (this section originally listed them as unsupported
blockers — that is no longer true):

1. `UTMALDG` — modeled; **exact real base** + descriptor-box size
2. `UTMAPF` — modeled prefetch; exact base (prologue chain), fwd only
3. `UTMASTG` — modeled; exact real base + descriptor-box size
4. `UBLKCP` — modeled; **synthetic base**, exact covered_bytes size
5. `UBLKRED` — modeled; **synthetic base** (raw-ptr dest, not a tensormap), exact covered_bytes size

Remaining work (address realism, not runnability):
- **M2 coords** — spread each tensor's transfers across its tiles so the L2
  hit-rate stops over-counting (base-only collapses all tiles to one address).
  Implemented as visit-counter tile spread; verify hit-rate direction toward HW.
- **M2.5 UBLKRED/UBLKCP exact base** — dereference the raw GMEM pointer
  (`c[0x0][0x280]/0x2a0`) + dynamic tile coords; static tensormap mapping cannot
  do this.
- **M3** — remove the legacy `handle_hi → config_id` heuristic entirely (base +
  all 11 descriptor fields + size now come from the exact base map).
