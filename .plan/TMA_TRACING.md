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

## New Extraction Findings

### `UBLKRED` uses explicit `desc[URx]`

The backward trace shows `UBLKRED.G.S.ADD.F32.RN` as a descriptor-consuming TMA op with the same explicit `desc[URx]` form used by `UTMALDG.*`.

Example:

```text
UBLKRED.G.S.ADD.F32.RN [UR28], [UR18], UR11, desc[UR30]
```

Practical implication:

- `UBLKRED` can use the same descriptor extraction path as `UTMALDG`
- if the opcode text does not expose rank directly, the resolver can infer rank from same-function reuse of the same handle family or descriptor register pair

### `UBLKCP` uses a uniform span operand encoded in 16-byte units

The FlashAttention-3 backward trace shows multiple `UBLKCP.S.G` sites with the form:

```text
UBLKCP.S.G [URdst], [URsrc], URspan
```

Example FA3 sites:

```text
/*a260*/ UBLKCP.S.G [UR28], [UR6], UR9
/*a620*/ UBLKCP.S.G [UR6], [UR8], UR10
/*a800*/ UBLKCP.S.G [UR14], [UR22], UR10
```

The important finding is that callback 2 is the span-like uniform operand and it decodes the same way as the validated `UBLKCP` microbench:

- raw callback value
  - `32`
- decoded covered bytes
  - `512`
- decode rule
  - `covered_bytes = operand_3 * 16`

Concrete FA3 evidence:

- `pc 0xa260`
  - raw `UR9 = 32`, decoded `512` bytes
- `pc 0xa620`
  - raw `UR10 = 32`, decoded `512` bytes
- `pc 0xa800`
  - raw `UR10 = 32`, decoded `512` bytes
- `pc 0x4f0`
  - raw `UR10 = 2048`, decoded `32768` bytes

Practical implication:

- bulk `UBLKCP` should use the same resolver style as bulk `UBLKRED`
- operand 3 should be recorded as:
  - top-level operand label: `covered_bytes_or_encoded_span`
  - semantic decode: `kind = covered_bytes`, `encoding = 16B_units`, `scale_bytes = 16`
- the simulator-facing byte count should come from `decoded_byte_samples`, not from the raw callback value alone

### `UTMASTG` uses a desc-like first operand pair

The trace shows `UTMASTG.*` in the form:

```text
UTMASTG.4D [UR8], [UR6]
UTMASTG.5D [UR8], [UR16]
```

The first bracketed uniform-register pair behaves like the descriptor selector / handle source:

- `UTMASTG.4D [UR8], [UR6]`
  - desc-like pair: `UR8/UR9`
  - support pair: `UR6/UR7`
- `UTMASTG.5D [UR8], [UR16]`
  - desc-like pair: `UR8/UR9`
  - support pair: `UR16/UR17`

This interpretation is supported by nearby producer instructions and later `UTMALDG.* ... desc[UR8]` reuse in the same function.

Practical implication:

- static extraction should classify the first `UTMASTG` operand pair as:
  - `desc_refs = [8]`
  - `desc_regs = [8, 9]`
- runtime capture should treat that first operand pair as the desc-like source for `UTMASTG`

### `support_regs` meaning

In `tma_discovery.json`, `support_regs` means:

- uniform-register operands used by the instruction that are **not** currently classified as the descriptor pair

So after the `UTMASTG` update:

- `UTMASTG.4D [UR8], [UR6]`
  - `desc_regs = [8, 9]`
  - `support_regs = [6, 7]`

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

For non-descriptor bulk `UBLKCP`, this is not a blocker because the simulator-relevant span comes from the uniform operand callback rather than from descriptor state.

### Where the Gap Is

The TMA descriptor is a 256-bit value stored in the GPU's **hardware descriptor cache**. The tracer cannot read it because:

1. NVBit instruments SASS instructions at runtime but has no hook to read GPU hardware state (descriptor cache contents)
2. The descriptor is set up by `DMNP`/`TMMA` instructions that write to a special hardware structure, not to memory
3. No current field in `inst_trace_t` or the protobuf schema carries the 256-bit descriptor value
4. The `reg_id` field records only the register number, not the register's value at trace time

---

## Three Options to Capture TMA Descriptors

### Option 1 — Decode TMA Descriptors from CUBIN Binary (Not Viable)

**How it works:** Attempt to recover the TMA descriptor by decoding preserved CUBIN / SASS artifacts and matching descriptor-definition instructions to later `UTMALDG.*` / `UTMASTG.*` consumers.

**Steps:**
1. Disassemble each kernel's CUBIN:
   ```bash
   cuobjdump -sass kernel.sm_90.cubin > kernel.sm_90.sass
   ```
2. Find the descriptor-definition instruction sequence in the preserved disassembly
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

**Why Option 1 is not viable as the main path:**

1. It only has access to preserved static artifacts, so it cannot recover descriptor state that is assembled, patched, or specialized dynamically at runtime.
2. The simulator needs the descriptor semantics that are actually visible at the `UTMALDG.*` / `UTMASTG.*` issue point, including per-CTA or per-launch variation; static binary inspection cannot prove those final runtime values.
3. The descriptor construction flow may involve multiple producer instructions and runtime inputs, so a pure CUBIN/SASS decode is not a reliable source of truth for the resolved 256-bit descriptor used by the consumer.
4. Because of those gaps, Option 1 is still useful as a cross-check or static reference, but it is not sufficient to drive simulator-visible TMA behavior.

### Option 2 — Reconstruct Runtime TMA Descriptors from Producer Instructions

**How it works:** Instead of trying to read the descriptor value at the `UTMALDG.*` / `UTMASTG.*` consumer PC, extend the tracer to capture the runtime inputs of the instructions that build or update descriptor state, reconstruct the 256-bit descriptor on the host, and then attach that reconstructed descriptor to later TMA consumers.

This is the most promising direction if the simulator needs the actual descriptor semantics at runtime:
- tensor rank and box dimensions
- tile dimensions and strides
- swizzle / layout controls
- element size and cache policy
- base address and per-CTA variations

**Why not capture the value directly at the consumer PC?**

Reading a single value from `desc[URx]` at `UTMALDG.*` / `UTMASTG.*` is not promising for exact reconstruction because:

1. The consumer instruction exposes only the descriptor operand reference such as `desc[UR18]`, not the full 256-bit descriptor payload.
2. The current tracer path can capture at most one uniform-register value for a memory reference, which is far smaller than the full descriptor.
3. The descriptor itself is held in special descriptor state / cache, so the operand-side `URx` is typically a selector, handle, or partial input rather than the final expanded descriptor image.
4. Different producer instructions may contribute different fields at different points before the TMA issue, so a snapshot taken only at the consumer PC can miss earlier dynamic updates.
5. The current TMA-specific runtime UR capture path already shows instability on FlashAttention runs, so increasing dependence on consumer-side reads is high risk.

That concern applies mainly to descriptor reconstruction. For bulk `UBLKCP`, the currently needed runtime fact is much narrower:

- the destination / source runtime references from the memory-ref callbacks
- the operand-3 span value from the uniform callback

The FA3 trace already provides that information sufficiently to validate `covered_bytes = operand_3 * 16` for the checked `UBLKCP` sites.

**Revised changes required:**

In `tracer_tool.cu` → `instrument_function_if_needed()`:
```c
if (is_tma_descriptor_producer(instr)) {
    // Capture the producer PC, opcode, descriptor selector/slot operands,
    // and the raw register/immediate inputs used to build descriptor state.
}

if (is_tma_consumer(instr)) {
    // Capture the descriptor operand reference such as desc[URx]
    // so the host can link the runtime consumer back to the latest
    // reconstructed descriptor snapshot in the same kernel / CTA / warp scope.
}
```

In `common.h` → extend `inst_trace_t` with a dedicated descriptor-event payload:
```c
typedef struct {
    // ... existing fields ...
    uint32_t tma_event_kind;      // none / producer / consumer
    uint32_t tma_desc_ref;        // desc[URx] selector or equivalent operand id
    uint64_t tma_raw_words[4];    // raw producer-side words when directly available
    uint32_t tma_aux_words[8];    // extra runtime fields / immediates / controls
    bool has_tma_descriptor_event;
} inst_trace_t;
```

In the CPU-side trace processing path:
```c
// Maintain runtime descriptor state keyed by
// (unique_function_id, cta_id, warp_id, descriptor reference).
// Each producer event updates the partial / full 256-bit descriptor.
// Each consumer event resolves to the latest reconstructed snapshot.
```

In the sideband trace artifact or protobuf path:
```protobuf
message TmaDescriptorSnapshot {
    uint32 function_unique_id = 1;
    uint32 pc = 2;
    uint32 desc_ref = 3;
    repeated uint64 raw_words = 4;
    uint64 base_address = 5;
    repeated uint32 dims = 6;
    repeated uint64 strides = 7;
    uint32 element_size = 8;
    uint32 swizzle = 9;
    uint32 box_dim = 10;
    uint32 interleave = 11;
}
```

**Implementation notes:**
- Do not assume in advance that the full descriptor comes from one mnemonic or one instruction.
- First identify the real descriptor producer sequence from preserved disassembly plus NVBit operand metadata.
- Keep producer capture and consumer capture separate from the existing generic memory-reference injection path to reduce callback argument fragility.
- Prefer writing reconstructed descriptor snapshots to a sideband debug artifact first, and only move them into the main trace schema after validation.

**Limitation:** Requires rebuilding `tracer_tool.so`, identifying the actual descriptor producer sequence, and implementing a host-side descriptor state tracker. This is more work than Option 1, but it is the only viable path for recovering true runtime descriptor semantics when the descriptor is assembled dynamically.

### Current practical trace-backed rule

For the current FlashAttention trace, exact reconstructed 256-bit descriptor payloads are still not available at all consumer sites. The current extraction pipeline therefore uses the following practical rules:

- `UTMALDG.*`
  - consume explicit `desc[URx]`
  - usually expose a stable nonzero `handle_hi` family signal
- `UBLKRED.*`
  - consume explicit `desc[URx]`
  - may require rank inference from same-function related uses when the opcode text does not include `4D` or `5D`
- `UTMASTG.*`
  - use the first bracketed `UR` pair as desc-like
  - may not expose a useful nonzero `handle_hi`
  - should therefore be resolved from rank plus same-function descriptor-register reuse

---

## Finalized Option 2 implementation design

The final implementation path does **not** try to reconstruct the descriptor by decoding arbitrary producer instructions in SASS. Instead, it uses the CUDA driver API call that already assembles the tensor map descriptor on the host:

- `cuTensorMapEncodeTiled`

That API is the most reliable source of truth because it exposes:
- the semantic tensor-map inputs:
  - rank — how many logical tensor dimensions the descriptor uses
  - data type — the element type encoded into the tensor map
  - global dimensions — the full logical shape of the tensor being described
  - global strides — the outer-dimension address jumps used to walk that tensor in memory
  - box dimensions — the tile shape that each TMA transaction reads or writes
  - element strides — per-dimension stepping in element units inside the tensor map
  - interleave — whether adjacent tensor data is interleaved in a special hardware-friendly layout
  - swizzle — which address swizzle pattern the tensor map applies to improve memory behavior
  - cache policy — how the TMA request should interact with cache promotion / residency controls
  - OOB mode — what the hardware should do when the tensor access goes out of bounds
- the encoded output object:
  - `CUtensorMap`

The tracer now dumps both:
- a human-readable CSV row in `tensor_map_encode_dump.csv`
- the raw encoded `CUtensorMap` bytes in `tensor_map_encode_blobs/<dump_id>.bin`

This makes the host-side encoded tensor map the descriptor source of truth for simulator modeling.

### Terminology used in this section

The word **family** in the remaining discussion is a tracing term, not an official NVIDIA architectural field.

- **Tensor-map dump**
  - one observed `cuTensorMapEncodeTiled` output row plus its raw `CUtensorMap` blob
- **Tensor-map family**
  - a normalized group of tensor-map dumps that share the same simulator-relevant semantic fields:
    - `tensor_rank`
    - `tensor_data_type`
    - `global_dim`
    - `global_strides`
    - `box_dim`
    - `element_strides`
    - `interleave`
    - `swizzle`
    - `l2_promotion`
    - `oob_fill`
- **Runtime handle family**
  - a group of dynamic `UTMALDG.*` observations that share the same stable `desc_value_hi` for the same `(unique_function_id, pc_hex)` site
- **Resolved descriptor config**
  - the normalized tensor-map family that the simulator should use for a given runtime TMA consumer site

The important point is that the current workflow does **not** claim that `desc_value_hi` is itself a decoded box dimension or a complete descriptor. It is only a stable runtime tag that can be correlated with the host-side tensor-map encode outputs.

### Final data flow

The finalized path has three primary data sources and two downstream processing steps:

1. **Static TMA discovery**
   - scan enhanced trace / disassembly and identify TMA consumer PCs like `UTMALDG.4D ... desc[URx]`
   - also backtrack the likely producer chain that writes the uniform register pair consumed by `desc[URx]`
   - produce `tma_discovery.json`
   - main value:
     - identifies the static TMA site by `(unique_function_id, pc_hex)`
     - supplies the opcode needed to infer tensor rank from the static instruction form
     - provides producer-side evidence for debugging, but is not the primary resolver key

2. **Runtime consumer capture**
   - at each `UTMALDG.*` issue, capture:
     - `unique_function_id` — identifies which static kernel/function variant this dynamic consumer belongs to
     - `pc_hex` — identifies the exact TMA consumer instruction site inside that function
     - `desc_reg_id` — records which uniform descriptor register pair the consumer reads from
     - `desc_value_lo` — captures the low 32-bit runtime descriptor handle value seen by the consumer
     - `desc_value_hi` — captures the high 32-bit runtime descriptor handle value used as the stable runtime tag
     - `first_lane_addr` — records one sampled effective memory address from the first active thread slot in the warp, used only as a lightweight correlation hint
   - write `tma_desc_runtime_debug.csv`

3. **Producer-side handle capture**
   - find the producer instruction chain that sets the uniform register pair used by `desc[URx]`
   - write `tma_desc_producer_debug.csv`
   - main value:
     - verify that the consumer-side `desc_reg_id` is fed by the expected producer chain
     - verify that the observed runtime handle family is not an artifact of one instruction site
     - provide debugging evidence when multiple candidate tensor-map families exist

4. **Tensor-map encode capture**
   - hook `cuTensorMapEncodeTiled`
   - on callback exit, dump:
     - semantic descriptor fields to `tensor_map_encode_dump.csv`
     - raw `CUtensorMap` bytes to `tensor_map_encode_blobs/*.bin`
   - this is the source of truth for simulator-visible tensor-map semantics

5. **Normalization for simulation**
   - parse `tensor_map_encode_dump.csv`
   - deduplicate semantically equivalent tensor maps into normalized config families
   - build:
     - `tma_descriptor_configs.json`
     - `tma_descriptor_resolver.json`

The simulator should use the normalized config/resolver JSON files, not the raw blob files.

### Why this final design is preferred

This design was chosen after several failed and successful iterations:

- **Normal HtoD copy capture was too weak**
  - hooking `cuMemcpyHtoD*` could not reliably observe descriptor creation
- **Generic callback logging was only a discovery tool**
  - useful to identify the real API path
  - not needed in steady-state tracing
- **`cuTensorMapEncodeTiled` is the right source**
  - it already produces the encoded descriptor object
  - it exposes the semantic fields the simulator actually needs

### Algorithm 1 — Capture descriptor candidates

For each CUDA callback:

1. If callback name is `cuTensorMapEncodeTiled`
2. Wait until callback exit
3. Read the `cuTensorMapEncodeTiled_params`
4. Write one row:
   - `dump_id`
   - tensor-map semantic fields
   - blob path
5. Write the raw `CUtensorMap` bytes to `tensor_map_encode_blobs/<dump_id>.bin`

This produces the full set of candidate descriptor tables created during the traced run.

### Algorithm 2 — Map a static TMA site to a normalized tensor-map config

This is the main resolver algorithm. Its job is to connect:

- the **static site identity** discovered from the TMA consumer or its producer chain
- the **runtime handle family** observed when that consumer executes
- the **host-side tensor-map encode output** captured from `cuTensorMapEncodeTiled`

#### Inputs

1. **Static discovery input**
   - `tma_discovery.json`
   - provides:
     - `unique_function_id`
     - `pc_hex`
     - `opcode`
     - descriptor register references and producer-search evidence

2. **Runtime consumer input**
   - `tma_desc_runtime_debug.csv`
   - provides:
     - `unique_function_id`
     - `pc_hex`
     - `desc_reg_id`
     - `desc_value_lo`
     - `desc_value_hi`

3. **Tensor-map encode input**
   - `tensor_map_encode_dump.csv`
   - provides:
     - semantic tensor-map fields
     - one row per `cuTensorMapEncodeTiled` call

#### Output

- one resolver entry that maps:
  - `(unique_function_id, pc_hex, handle_hi_hex)`
  - to a normalized `config_id`

#### Procedure

1. **Normalize tensor-map dumps into config families**
   - read all rows from `tensor_map_encode_dump.csv`
   - group rows by semantic fields only:
     - `tensor_rank`
     - `tensor_data_type`
     - `global_dim`
     - `global_strides`
     - `box_dim`
     - `element_strides`
     - `interleave`
     - `swizzle`
     - `l2_promotion`
     - `oob_fill`
   - assign one normalized `config_id` to each unique semantic group
   - ignore differences that are not needed for this simulation phase, such as backing buffer address

2. **Identify the static TMA consumer site**
   - from `tma_discovery.json`, select the site by:
     - `unique_function_id`
     - `pc_hex`
   - read the site opcode such as `UTMALDG.4D`
   - infer tensor rank from the static opcode suffix

3. **Collect runtime observations for that exact static site**
   - filter `tma_desc_runtime_debug.csv` by the same:
     - `unique_function_id`
     - `pc_hex`
   - group the remaining rows by `desc_value_hi`
   - each unique `desc_value_hi` becomes one runtime handle family for that static site

4. **Use producer information only as supporting evidence**
   - if producer-side discovery exists, verify that:
     - the producer chain writes the uniform register pair consumed by the static site
     - the same producer/consumer path consistently leads to the same runtime handle family
   - producer information is useful for confidence and debugging, but it is not the main resolver key in the current implementation

5. **Restrict candidate tensor-map families**
   - consider only tensor-map config families produced in the same launch window
   - prefer config families whose `tensor_rank` matches the consumer opcode rank

6. **Correlate runtime handle families with normalized tensor-map families**
   - for each static site, compare the set of observed runtime handle families against the set of normalized tensor-map families in the same launch window
   - in the current FlashAttention-3 trace, the two normalized 4D config families are distinguished primarily by `box_dim`
   - because the runtime samples also split into two stable `desc_value_hi` values, the mapping is:
     - one stable handle family ↔ one normalized tensor-map family

7. **Emit the resolver entry**
   - write the final mapping as:
     - `(unique_function_id, pc_hex, handle_hi_hex)` → `config_id`
   - if the correlation is not unique, emit candidate configs instead of a single resolved config

#### Important clarification

The current observed mapping:

- `0x14f00000` → `box_dim = 64 192 1 1`
- `0x12f00000` → `box_dim = 64 128 1 1`

does **not** mean that `desc_value_hi` is a decoded `box_dim` field.

It means only that, in the currently observed FlashAttention-3 run:

- `desc_value_hi` is a stable runtime tag
- the normalized tensor-map families in the same launch window are mainly distinguished by `box_dim`
- therefore the current correlation between runtime handle family and tensor-map family is effectively resolved by that distinction

If later traces contain multiple normalized tensor-map families with the same rank and the same `box_dim`, then this heuristic will no longer be sufficient by itself and more evidence will be required.

### Algorithm 3 — Worked example: why `0x94e0` maps to `dump_id 0`

For the first important consumer:

- PC `0x94e0`
- runtime handle high word `0x14f00000`

The mapping process is:

1. Query `tma_desc_runtime_debug.csv` for `pc_hex = 0x94e0`
2. Observe stable runtime tuple:
   - `desc_value_lo = 0`
   - `desc_value_hi = 0x14f00000`
3. Query tensor-map dumps from the same launch window
4. Compare descriptor families:
   - `dump_id 0` has:
     - `tensor_rank = 4`
     - `box_dim = 64 192 1 1`
   - `dump_id 1/2/...` have:
     - `tensor_rank = 4`
     - `box_dim = 64 128 1 1`
5. Match the unique `0x14f00000` runtime handle family to the unique normalized tensor-map family with `box_dim = 64 192 1 1`

So the current best mapping is:

- `pc 0x94e0` → runtime handle family `0x14f00000` → `dump_id 0` family → config family `tm_4d_dt9_box_64x192x1x1`

### Algorithm 4 — Worked example: why `0x96b0` maps to the `dump_id 1/2` family

For the second consumer:

- PC `0x96b0`
- runtime handle high word `0x12f00000`

The mapping process is:

1. Query `tma_desc_runtime_debug.csv` for `pc_hex = 0x96b0`
2. Observe stable runtime tuple:
   - `desc_value_lo = 0`
   - `desc_value_hi = 0x12f00000`
3. Query tensor-map dumps from the same launch window
4. Compare descriptor families:
   - `dump_id 1` and `dump_id 2` share identical semantic tensor-map fields
   - they differ only in `global_address_hex`
5. Since the simulator does not require the exact backing buffer base address for this phase, merge those rows into one normalized tensor-map family

So the current best mapping is:

- `pc 0x96b0` → runtime handle family `0x12f00000` → `dump_id 1 or 2` family → config family `tm_4d_dt9_box_64x128x1x1`

### Algorithm 5 — Build simulator-facing configs

Input:
- `tensor_map_encode_dump.csv`
- `tma_desc_runtime_debug.csv`
- `tma_discovery.json`

Steps:

1. Read all tensor-map rows
2. Deduplicate rows by semantic fields:
   - rank, type, dims, strides, box dims, element strides, interleave, swizzle, l2 promotion, oob fill
3. Assign a normalized `config_id` to each unique family
4. Read all runtime consumer rows
5. Group runtime rows by:
   - `unique_function_id`
   - `pc_hex`
   - `handle_hi_hex`
6. Read `tma_discovery.json` and build the `(unique_function_id, pc_hex) -> opcode` map
7. Use the static opcode rank plus the observed runtime `handle_hi_hex` family to resolve each grouped runtime consumer to a normalized `config_id`
8. Write:
   - `tma_descriptor_configs.json`
   - `tma_descriptor_resolver.json`

### Final resolver key

The simulator should **not** resolve by `desc_reg_id` alone.

The current best lookup key is:

- `(unique_function_id, pc_hex, handle_hi_hex)`

because:
- the same `desc[URx]` register pair can carry different descriptor families at different PCs
- `handle_hi_hex` is the stable runtime family signal observed so far
- the static site identity from `(unique_function_id, pc_hex)` is required to connect the runtime observation back to the correct consumer instruction

### Current scope

The current implementation is validated for the first analyzed `UTMALDG` family in the FlashAttention-3 trace:

- `0x94e0`
- `0x96b0`
- `0x9a80`

The same config/resolver framework is intended to be extended later to:
- additional `UTMALDG.*` sites
- `UTMASTG.*`
- other TMA ops that also use `desc[URx]`

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

1. **Do not pursue Option 1 as the implementation path.** Keep preserved cubin/SASS artifacts only as a static reference or cross-check mechanism because static decoding is not viable for recovering simulator-visible runtime descriptor semantics.

2. **Move to revised Option 2 as the main runtime path.** Instrument descriptor producer instructions, reconstruct descriptor state on the host, and attach the resolved descriptor snapshot to each `UTMALDG.*` / `UTMASTG.*` consumer.

3. **Persist decoded descriptor semantics** needed by the future simulator: base address, dimensions, strides, element size, swizzle, box dimensions, and any other cache/layout controls that affect TMA behavior.

4. **Add TMA-specific pressure signals** to `pressure_signals_t`: TMA transaction count, bytes transferred, and TMA hit rate, sourced from the remodeled SM's TMA unit model.

5. **Add a TMA-aware classifier flag** — if a kernel uses TMA, bypass the log-fit model and use a separate TMA-specific cycle model (e.g., warp-occupancy-based rather than CTA-concurrency-based).

### Current Status of Phase 1 Artifacts

The artifact-preservation part of Phase 1 appears to already be working for the examined FlashAttention trace:

```text
hw_run/traces/device-0/12.8/
  flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/
  flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/
  traces/extra_info/
```

Observed preserved artifacts under that path:
- `extra_info/cubin/`
- `extra_info/sass/`
- `extra_info/nvdisasm/`
- `extra_info/register_usage/`
- `extra_info/enhanced_execution_info.json`

This means revised Option 2 should treat artifact preservation as mostly complete for this workload and focus first on:
- identifying the real descriptor producer sequence in the preserved disassembly
- mapping producer PCs to `UTMALDG.*` / `UTMASTG.*` consumer PCs
- validating whether the preserved disassembly is sufficient to recover descriptor-construction structure before adding new runtime events

### Code Change Plan for Revised Option 2

Implement revised Option 2 in four patches so each stage can be validated independently.

#### Patch 1 — TMA discovery and producer/consumer mapping

**Goal:** use the already-preserved artifacts to identify the real descriptor producer sequence before adding new runtime payloads.

Files:
- `util/tracer_nvbit/tracer_tool/tracer_tool.cu`
- `traces/extra_info/sass/*.sass`
- `traces/extra_info/nvdisasm/*.nvdisasm`

Changes:
1. Add a TMA debug/discovery mode that emits a lightweight `extra_info/tma_consumers.json` artifact.
2. Record, for each TMA consumer instruction:
   - `unique_function_id`
   - PC
   - opcode
   - descriptor operand reference such as `desc[URx]`
3. Add helper logic in the tracer-side disassembly parsing path to:
   - find TMA consumer instructions in preserved SASS / nvdisasm
   - search backward for candidate descriptor producer instructions
   - generate a provisional producer-to-consumer mapping table
4. Validate the mapping specifically on FlashAttention function 5 and function 10.

Success criteria:
- the trace emits a stable consumer list
- at least one plausible producer sequence is identified for the main `UTMALDG.*` sites in the FlashAttention kernels

#### Patch 2 — Dedicated TMA runtime event channel

**Goal:** add a clean runtime event path for descriptor work without extending the fragile generic memory-reference callback.

Files:
- `util/tracer_nvbit/tracer_tool/common.h`
- `util/tracer_nvbit/tracer_tool/inject_funcs.cu`
- `util/tracer_nvbit/tracer_tool/tracer_tool.cu`

Changes:
1. Extend `inst_trace_t` with a dedicated TMA event payload:
   - event kind
   - descriptor reference / slot
   - producer opcode id or tag
   - raw words / aux fields for producer operands
2. Add a separate device callback for TMA descriptor events instead of reusing the existing `instrument_inst(...)` memory-reference path.
3. In `instrument_function_if_needed()`, create two separate instrumentation flows:
   - descriptor producer events
   - TMA consumer events
4. Keep this logic separate from the existing generic injection path so callback argument ordering stays simple and debuggable.

Success criteria:
- no illegal memory access during trace collection
- producer and consumer events appear in a dedicated debug artifact

#### Patch 3 — Host-side descriptor reconstruction

**Goal:** reconstruct the actual runtime descriptor state from producer events and resolve it at consumer PCs.

Files:
- `util/tracer_nvbit/tracer_tool/tracer_tool.cu`
- optional helper under `util/tracer_nvbit/tracer_tool/` for descriptor reconstruction state

Changes:
1. Add a host-side descriptor-state tracker keyed by:
   - `unique_function_id`
   - CTA id
   - warp id
   - descriptor reference / slot
2. Update descriptor state as producer events arrive.
3. Resolve the latest visible descriptor snapshot when a `UTMALDG.*` / `UTMASTG.*` consumer event arrives.
4. Decode the resolved snapshot into simulator-relevant fields:
   - base address
   - dimensions
   - strides
   - element size
   - swizzle
   - box dimension / rank
   - any additional layout or cache controls that can be recovered
5. Write the result into a debug-first artifact such as `extra_info/tma_descriptors_runtime.json`.

Success criteria:
- reconstructed descriptor snapshots are stable across repeated runs when descriptor inputs are invariant
- decoded output contains meaningful tensor-shape and layout metadata instead of only a scalar descriptor token

#### Patch 4 — Simulator-visible metadata attachment

**Goal:** attach validated runtime descriptor metadata to instruction metadata so later simulator work can consume it directly.

Files:
- `util/traces_enhanced/src/traced_instruction.h`
- `util/traces_enhanced/src/traced_instruction.cc`
- `gpu-simulator/trace-driven/trace_driven.cc`
- loader path for `enhanced_execution_info.json` or the new sideband runtime metadata artifact

Changes:
1. Add a `tma_descriptor_info` payload to traced static instruction metadata.
2. Store:
   - whether the instruction uses TMA
   - descriptor operand reference
   - raw descriptor words
   - decoded runtime fields
3. Attach the reconstructed descriptor metadata to TMA consumer instructions using function id and PC.
4. Reuse the existing `traced_instruction` attachment path so the simulator can query TMA metadata without re-decoding the descriptor per dynamic instruction.

Success criteria:
- a `UTMALDG.*` or `UTMASTG.*` instruction can access decoded descriptor metadata through its associated traced instruction object
- no simulator timing-model changes are required in the first integration patch

### Execution Plan for Revised Option 2

The runtime path should be implemented in stages so we can first identify the true producer sequence, then validate descriptor reconstruction before changing simulator semantics.

#### Pass 1 — Identify the real descriptor producer sequence

1. Start from the already-preserved `extra_info/cubin/`, `extra_info/sass/`, and `extra_info/nvdisasm/` artifacts for the FlashAttention trace.
2. Inspect FlashAttention function 5 and function 10 to find the instructions that build or update the descriptor state consumed by `UTMALDG.*` / `UTMASTG.*`.
3. Record the producer-to-consumer mapping as:
   - producer PC
   - producer opcode
   - descriptor operand reference such as `desc[URx]`
   - consumer PC
4. Confirm whether the descriptor is built in one instruction or through a short sequence of updates.
5. Only if the existing artifacts are insufficient, extend the tracer to emit additional debug mapping artifacts.

#### Pass 2 — Add dedicated runtime descriptor events

1. Extend `util/tracer_nvbit/tracer_tool/common.h` with a dedicated TMA descriptor event payload rather than reusing the existing single-field `ureg_desc_value` path.
2. Add a separate instrumentation path in `util/tracer_nvbit/tracer_tool/tracer_tool.cu` for:
   - descriptor producer events
   - TMA consumer events
3. Keep this new path logically separate from the generic memory-reference instrumentation so callback argument order remains easy to reason about and debug.
4. Rebuild the tracer and validate first on a minimal TMA microbenchmark before returning to FlashAttention.

#### Pass 3 — Reconstruct descriptor state on the host

1. In the CPU-side trace handling path, maintain descriptor state keyed by:
   - `unique_function_id`
   - CTA id
   - warp id
   - descriptor reference or slot
2. For each producer event, update the partial or complete 256-bit descriptor state.
3. For each `UTMALDG.*` / `UTMASTG.*` consumer event, resolve the latest descriptor snapshot visible in that scope.
4. Emit both the raw 256-bit words and decoded fields needed by the simulator.

#### Pass 4 — Persist a debug-first descriptor artifact

1. Before changing the main protobuf schema, write reconstructed descriptor snapshots into a sideband artifact under `extra_info/`.
2. For each snapshot, store:
   - function id
   - producer PC
   - consumer PC
   - descriptor reference
   - raw 256-bit words
   - decoded address / dims / strides / swizzle / box-dim metadata
3. Re-run the same workload twice and check that the reconstructed snapshots are stable where the descriptor should be invariant and vary only where runtime inputs truly differ.

#### Pass 5 — Promote validated fields into simulator-visible metadata

1. After the debug artifact is validated, extend the trace metadata path so each TMA consumer can access:
   - base address
   - tensor dimensions
   - strides
   - element size
   - swizzle
   - box dimension / rank
   - relevant cache or interleave controls
2. Attach that metadata either to the dynamic trace instruction or to a side lookup keyed by function id and PC.
3. Reuse the existing `traced_instruction` attachment path so the simulator can access TMA metadata without re-decoding the descriptor on every dynamic instruction.

#### Scope limits for the first Option 2 patch

- Do not modify the TMA simulator model yet.
- Do not rely on consumer-PC `desc[URx]` reads as the authoritative descriptor value.

---

## Bulk `UBLKRED` Follow-up Findings

Additional Hopper microbench work established that there are at least two practically different `UBLKRED` forms:

- **Descriptor-backed form**
  - example from FlashAttention:
    - `UBLKRED.G.S.ADD.F32.RN [UR28], [UR18], UR11, desc[UR30]`
  - this form should continue to use the descriptor extraction / resolver path
- **Bulk non-descriptor form**
  - reproduced with a standalone microbench:
    - `UBLKRED.G.S.ADD.F32.RN [UR8], [UR4], UR6`
  - this form has no `desc[URx]` operand
  - increasing source / destination backing buffer sizes does **not** cause a descriptor operand to appear

### Bulk-form operand roles

For the validated bulk microbench form:

```text
UBLKRED.G.S.ADD.F32.RN [UR8], [UR4], UR6
```

the runtime and source-level probes support the following interpretation:

- operand 1
  - destination base / destination memory reference
- operand 2
  - source base / source shared-memory reference
- operand 3
  - covered span control for the current bulk reduce-store operation

The key point is that operand 3 is **not** a many-to-one reduction fan-in count.
Instead, the bulk form behaves like an elementwise reduce-store across a covered region.

### Operand 3 encoding

Runtime operand capture plus size-scaling microbenches showed:

- `op_bytes = 16`   → operand 3 runtime value `1`
- `op_bytes = 32`   → operand 3 runtime value `2`
- `op_bytes = 64`   → operand 3 runtime value `4`
- `op_bytes = 256`  → operand 3 runtime value `16`
- `op_bytes = 1024` → operand 3 runtime value `64`

Therefore, for this bulk `UBLKRED` form:

```text
covered_bytes = operand_3 * 16
covered_elements_f32 = operand_3 * 4
```

and the execution model for F32 is:

```cpp
for (int i = 0; i < operand_3 * 4; ++i) {
    dst[dst_base + i] += src[src_base + i];
}
```

### Practical tracing implication

Static CUBIN / SASS inspection is useful as a decode aid for operand preparation, but it is not sufficient by itself to recover the actual runtime operand value used in one traced launch.

The current practical split is:

- `tma_descriptor_configs.json`
  - descriptor semantics only
- `tma_descriptor_resolver.json`
  - descriptor-backed consumer → config mapping
- `tma_operand_resolver.json`
  - runtime operand semantics and static decode formulas for TMA-family ops, including bulk `UBLKRED`

For simulator work, bulk `UBLKRED` should therefore use the operand resolver rather than the descriptor resolver as the primary source of operand-3 semantics.

### Descriptor-backed `UBLKRED` findings from FlashAttention FA3

The FlashAttention-3 backward trace shows multiple descriptor-backed `UBLKRED` sites, for example:

- `/*90a0*/ UBLKRED.G.S.ADD.F32.RN [UR10], [UR4], UR12, desc[UR16]`
- `/*94e0*/ UBLKRED.G.S.ADD.F32.RN [UR32], [UR21], UR22, desc[UR30]`

Current evidence suggests:

- operand 3 remains an independent size/span-style control operand
  - the raw runtime value is stable at `1024` across multiple descriptor-backed `UBLKRED` PCs in the analyzed FA3 run
- operand 1 is **not** obviously a full linear destination address by itself
  - the textual register changes across PCs
  - but the observed runtime payload can remain a small stable value such as `33792`
- the descriptor carries the tensor-map layout information that operand 1 / operand 2 / operand 3 do not encode
  - shape
  - strides
  - box dimensions
  - related tensor-map interpretation state

This supports the working model:

```text
operand 1 = destination-side runtime state
operand 2 = source-side runtime state
operand 3 = covered span / encoded size control
desc      = tensor-layout / address-resolution context
```

This is the most likely explanation for why `desc` is still required even when operand 3 already provides size-like semantics.

### `UTMAREDG` modeling note

For simulator design, `UTMAREDG` should not be reduced to only:

- amount of data
- reduction op kind

unless the intended model is a deliberately coarse throughput-only abstraction.

At minimum, there are two simulator levels:

- **coarse performance model**
  - amount of data moved / reduced
  - reduction op kind
  - issue / completion cost
  - this may be enough for first-order bandwidth or latency estimation
- **layout-aware / address-aware model**
  - descriptor-carried tensor-map layout
  - destination mapping behavior
  - possible effects on locality, addressing, or correctness-sensitive placement

So the short answer is:

- **for a minimal coarse model**: yes, amount + op kind may be sufficient
- **for a faithful model**: no, descriptor/layout semantics still matter
- Do not require the first patch to support every possible descriptor-producing opcode family; proving correct reconstruction on FlashAttention is enough.
- Keep Option 1 only as a non-authoritative static reference and cross-check path.

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
