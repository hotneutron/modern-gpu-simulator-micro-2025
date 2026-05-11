# TMA ISA Support Status (FA3 Trace)

## Scope

This note is for **FlashAttention-3 (FA3)** trace investigation on Hopper.

Trace path checked:

`/home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/hw_run/traces/device-0/12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/traces/dynamic_trace.pb`

Related files checked:

- `.../traces/extra_info/tma_discovery.json`
- `.../traces/extra_info/nvdisasm/*.nvdisasm`
- `.../traces/extra_info/enhanced_execution_info.json` (exists, very large)

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

Trace-driven decode/execution path reference:

- `gpu-simulator/trace-driven/trace_driven.cc`
- Hopper map: `gpu-simulator/ISA_Def/hopper_opcode.h`

| Opcode family in FA3 trace | Decode category now         | Runtime handling status in remodeled simulator                                                                                                |
| -------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `UTMALDG.*`                | `LOAD_OP`                   | **Gap / blocker**. No explicit `OP_UTMALDG` memory-space setup in `trace_driven.cc` switch. Acts as load category without real TMA semantics. |
| `UTMAPF.*`                 | `LOAD_OP`                   | **Gap / blocker**. No explicit `OP_UTMAPF` handling in `trace_driven.cc` switch.                                                              |
| `UTMASTG.*`                | `STORE_OP`                  | **Gap / blocker**. No explicit `OP_UTMASTG` handling in `trace_driven.cc` switch.                                                             |
| `UBLKCP.*`                 | `LOAD_OP`                   | **Gap / blocker**. No explicit `OP_UBLKCP` handling in `trace_driven.cc` switch.                                                              |
| `UBLKRED.*`                | `STORE_OP`                  | **Gap / blocker**. No explicit `OP_UBLKRED` handling in `trace_driven.cc` switch.                                                             |
| `UTMACCTL.*`               | `MEMORY_MISCELLANEOUS_OP`   | **Stubbed**. Routed/timed as generic memory-misc op; no TMA-control semantics modeled.                                                        |
| `UTMACMDFLUSH`             | `MEMORY_MISCELLANEOUS_OP`   | **Stubbed**. Routed/timed as generic memory-misc op; no explicit TMA metadata/cache flush semantics modeled.                                  |
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
- operand 1: `[URdst]`
- operand 2: `[URsrc]`
- operand 3: `desc[URdesc]`

Simple meaning:

- Hopper TMA async tensor load
- first bracketed operand is the destination/control tuple
- second bracketed operand is the source/address or coordinate tuple
- descriptor operand selects the TMA descriptor state

Current simulator handling:

- decoded as `LOAD_OP`
- no explicit `OP_UTMALDG` setup in `trace_driven.cc`
- no real TMA request / descriptor semantics are modeled
- currently a blocker for FA3 trace-driven runnability

### `UTMAPF.L2.4D`

Example seen in checked FA3 files:

- `UTMAPF.L2.4D [UR28], [UR8]`

Operand shape:

- 2 operands
- operand 1: `[URa]`
- operand 2: `[URb]`

Simple meaning:

- Hopper TMA prefetch operation
- `.L2` suggests prefetch intent toward L2
- `.4D` indicates tensor-region style operation

Current simulator handling:

- decoded as `LOAD_OP`
- no explicit `OP_UTMAPF` handling in `trace_driven.cc`
- no real TMA prefetch behavior is modeled
- currently a blocker for FA3 trace-driven runnability

### `UTMASTG.4D` / `UTMASTG.5D`

Examples seen in checked FA3 files:

- `UTMASTG.4D [UR8], [UR6]`
- `UTMASTG.5D [UR8], [UR16]`

Operand shape:

- 2 operands
- operand 1: `[URa]`
- operand 2: `[URb]`

Simple meaning:

- Hopper TMA async tensor store
- one bracketed operand carries one side of the store tuple
- the other bracketed operand carries the other side
- `.4D` vs `.5D` indicates tensor-rank/layout variant

Current simulator handling:

- decoded as `STORE_OP`
- no explicit `OP_UTMASTG` setup in `trace_driven.cc`
- no real async TMA store semantics are modeled
- currently a blocker for FA3 trace-driven runnability

### `UBLKCP.S.G`

Example seen in checked FA3 files:

- `UBLKCP.S.G [UR28], [UR6], UR9`

Operand shape:

- 3 operands
- operand 1: `[URa]`
- operand 2: `[URb]`
- operand 3: `URc`

Simple meaning:

- Hopper bulk copy operation
- suffix suggests space direction / variant
- first two operands are tuple-like address/control operands
- third uniform register is extra control/size/metadata input

Current simulator handling:

- decoded as `LOAD_OP`
- no explicit `OP_UBLKCP` setup in `trace_driven.cc`
- no real bulk-copy memory behavior is modeled
- currently a blocker for FA3 trace-driven runnability

### `UBLKRED.G.S.ADD.F32.RN`

Example seen in checked FA3 files:

- `UBLKRED.G.S.ADD.F32.RN [UR10], [UR4], UR12, desc[UR16]`

Operand shape:

- 4 operands
- operand 1: `[URa]`
- operand 2: `[URb]`
- operand 3: `URc`
- operand 4: `desc[URd]`

Simple meaning:

- Hopper bulk reduction operation
- mnemonic encodes the reduction itself: `ADD.F32.RN`
- descriptor operand indicates that descriptor state participates in the operation

Current simulator handling:

- decoded as `STORE_OP`
- no explicit `OP_UBLKRED` setup in `trace_driven.cc`
- no bulk reduction semantics are modeled
- currently a blocker for FA3 trace-driven runnability

### `UTMACCTL.PF`

Example seen in checked FA3 files:

- `UTMACCTL.PF [UR6]`

Operand shape:

- 1 operand
- operand 1: `[URx]`

Simple meaning:

- Hopper TMA control operation
- `.PF` strongly suggests prefetch/control-state management

Current simulator handling:

- decoded as `MEMORY_MISCELLANEOUS_OP`
- routed and timed as a generic memory-misc op
- no explicit TMA control semantics are modeled
- runnable, but effectively stubbed

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

Highest-priority unsupported operation groups to make FA3 trace runnable in trace-driven mode:

1. `UTMALDG`
2. `UTMAPF`
3. `UTMASTG`
4. `UBLKCP`
5. `UBLKRED`

These are present in the checked FA3 trace and currently lack explicit runtime handling in `trace_driven.cc` opcode special-case setup.
