# TMA Architecture Reflection — 2026-05-18

## Purpose

This document records the architectural reasoning behind the TMA implementation plan,
including decisions made during design review that changed or sharpened the original
staged approach. It is intended to be a durable record of *why* the implementation
is shaped the way it is, so that future contributors can make good trade-off calls
without re-deriving the same conclusions.

---

## Fundamental Architectural Invariants

These three decisions cannot be cheaply reversed and must be preserved across all
implementation phases. Violating any one of them requires a significant rework.

### 1. TMA never routes through `ldst_unit_sm`

`LDGSTS` is an LD/ST-managed async copy that transitions through `LOAD_STAGE ->
STORE_STAGE` inside the normal per-warp memory pipeline. TMA is architecturally
different: one thread issues a command, a hardware engine performs address generation
and data movement independently, and completion is tracked against async transaction
progress rather than instruction retirement. Conflating the two makes it impossible
to model compute-TMA overlap and produces wrong scoreboarding semantics.

Every TMA opcode family (`UTMALDG`, `UTMASTG`, `UTMAPF`, `UTMACCTL`, `UTMACMDFLUSH`,
`UBLKCP`, `UBLKRED`, `UBLKPF`) must route through `m_tma_pipeline` → `tma_unit_sm`.
`ldst_unit_sm` must not be touched.

### 2. The issuing warp releases after enqueue, not after transfer completion

The subcore-side TMA pipeline (`m_tma_pipeline`) consumes one issue slot, packages a
`TMACommand`, enqueues it into `tma_unit_sm`, and releases the warp. Transfer progress
lives entirely inside `tma_unit_sm` from that point. This is the structural source
of compute-TMA overlap. If the subcore holds the warp for the transfer lifetime, the
overlap never emerges and the model degrades to a synchronous load.

### 3. `TMACompletionObject` is separate from `DEPBAR`/`LDGDEPBAR` tracking

Existing barrier mechanisms track per-instruction retirement. TMA completion is tied
to async transaction progress: bytes returned from memory, not instruction retirement.
These are different completion domains. Conflating them now makes Hopper-style
`mbarrier` (parity, expected-tx-bytes, phase) a full rewrite rather than an extension
of an already-correct structure.

---

## AGU Timing: Why `total_bytes` Is Wrong

### The failure mode

The TMA AGU pipeline produces one request per clock at peak. Its throughput is
determined by the number of cache-line-aligned requests it must generate, not the
total byte count. The correct quantity is:

```
row_bytes        = box_dim[0] * element_size
requests_per_row = ceil(row_bytes / 128)
outer_iters      = box_dim[1] * box_dim[2] * ... * box_dim[rank-1]
requests_total   = requests_per_row * outer_iters
```

`requests_total == total_bytes / 128` only when `box_dim[0] * element_size` is exactly
a multiple of 128. This is true for FA3's `box_dim[0]=64` with bf16 (`64*2=128`), which
creates an accidental pass on the primary test workload. It fails for any other geometry.

### Concrete example where `total_bytes` misleads

| Config | box_dim | total_bytes | requests_total |
|---|---|---|---|
| A | `[64, 8, 1, 1]` bf16 | 1,024 B | 8 |
| B | `[4, 128, 1, 1]` bf16 | 1,024 B | 128 |

Configs A and B are identical under a `total_bytes`-derived timing model. The AGU
hardware issues 16× more requests for B. A simulator using `total_bytes` would predict
identical TMA latency; the correct model predicts B takes 16× longer in the AGU phase.

### Why doing it right from the start saves work

If `total_bytes` is embedded in `TMATransferEntry` and the `IN_FLIGHT` state
transition, then:

- FA3 calibration looks correct (the accidental geometry match hides the bug)
- A new workload with different `box_dim` aspect ratio reveals wrong timing
- The fix requires changing `TMATransferEntry`, the `IN_FLIGHT` update logic,
  and re-baselining every test that used the old model

The correct quantity (`requests_total`) is four arithmetic operations on fields already
present in the descriptor config. There is no discovery or tracing work required.
Computing it correctly from day one costs essentially nothing and avoids a structural
rework later.

### What can still be simplified in early phases

The following simplifications are all additive — they do not require changing
`requests_total` or the core AGU structure when replaced with higher-fidelity versions:

- **Issue rate**: start by issuing all `requests_total` requests in one cycle;
  Phase 3 adds an `agu_issue_rate_per_cycle` admission parameter
- **Memory latency**: start with a fixed `cycles_per_request` latency;
  Phase 3 hooks the existing L2/DRAM latency model
- **Boundary handling**: start by assuming all requests are valid;
  Phase 7 adds per-row OOB checks
- **Swizzle effects on landing**: purely additive to the SMEM write path;
  never touches `requests_total`

---

## Data Structures: Design Final Fields Now

The single biggest source of wasted work in staged implementations is redesigning
core data structures mid-stream because early stubs used placeholder fields. The
structures below should be written with their final fields from day one, even when
most fields are unused in early phases.

### `TMACommand`

Produced by `m_tma_pipeline` at enqueue time. This is the input to `tma_unit_sm`.

```cpp
struct TMACommand {
    // Identity
    uint32_t warp_id;
    uint32_t cta_id;
    uint32_t sm_id;
    uint32_t subcore_id;

    // Instruction
    TMAOpcodeFamily opcode_family;   // UTMALDG, UTMASTG, UTMAPF, etc.
    TMADirection direction;          // GMEM_TO_SMEM, SMEM_TO_GMEM
    TMATransferType transfer_type;   // LOAD, STORE, PREFETCH, REDUCTION, CONTROL

    // Descriptor resolution
    std::string config_id;           // from tma_descriptor_resolver
    TMAMetadataSource meta_source;   // DESCRIPTOR, OPERAND, MIXED
    std::string mapping_method;      // for debugging resolver confidence
    float resolver_confidence;

    // Transfer geometry (from descriptor config + runtime coords)
    int rank;
    std::array<uint32_t, 5> box_dim;
    std::array<uint32_t, 5> coords;
    uint32_t element_size;           // bytes
    uint64_t smem_ptr;               // SMEM source or destination pointer
    uint32_t requests_total;         // = ceil(box_dim[0]*elem_sz/128) * prod(box_dim[1:])
    uint32_t total_bytes;            // = product(box_dim) * element_size (kept for reference)
    uint32_t covered_bytes;          // for validated bulk operand-driven forms
    uint32_t operand3_raw;           // raw runtime value for descriptor-backed size/span forms

    // Layout
    uint32_t swizzle;
    uint32_t interleave;
    uint32_t oob_fill;
    uint32_t l2_promotion;

    // Operand classification
    TMAOperandForm operand_form;     // EXPLICIT_DESC, DESC_LIKE_PAIR, BULK_OPERAND

    // Completion
    uint32_t completion_id;          // index into SM-level completion table
};
```

### `TMATransferEntry`

Lives inside `tma_unit_sm`. One entry per in-flight TMA operation.

```cpp
struct TMATransferEntry {
    TMACommand cmd;

    // State machine
    enum class State {
        ISSUED,          // warp has issued the TMA instruction
        ENQUEUED,        // accepted by tma_unit_sm; subcore ownership ends here
        AGU_READY,       // descriptor/operand metadata resolved; ready to issue requests
        IN_FLIGHT,       // requests are being issued and/or completed
        COMPLETED,       // all requests done; completion object is ready
        WAIT_SATISFIED   // consumer wait has observed the ready state
    };
    State state;

    // AGU / transfer progress
    uint32_t requests_issued;     // incremented at agu_issue_rate requests/cycle
    uint32_t requests_completed;  // incremented as memory-system completions arrive
    int cycle_enqueued;
    int cycle_agu_ready;
    int cycle_first_request;
    int cycle_last_completion;

    // Completion linkage
    uint32_t completion_id;       // same as cmd.completion_id
};
```

### `TMACompletionObject`

Kept at SM scope, one per in-flight TMA operation (or per mbarrier slot in Phase 5).
Consumer warps check this; `tma_unit_sm` writes it.

```cpp
struct TMACompletionObject {
    // Phase 5 will use these for full mbarrier semantics
    uint32_t expected_tx_bytes;   // set at issue time from descriptor config
    uint32_t completed_tx_bytes;  // incremented as requests_completed grows
    uint32_t phase;               // parity bit for Hopper-style mbarrier
    bool ready;                   // true when completed_tx_bytes >= expected_tx_bytes

    // Linkage
    uint32_t warp_id;
    uint32_t cta_id;
    int cycle_ready;
};
```

In Phase 1-4: `ready` is set when `requests_completed == requests_total`. `expected_tx_bytes`,
`completed_tx_bytes`, and `phase` are populated but the consumer wait only checks `ready`.
Phase 5 replaces the `ready` check with the full parity/expected-bytes protocol without
changing the structure layout.

---

## Descriptor Metadata Interface

The simulator must never interpret raw `desc[URx]` handle values or parse `CUtensorMap`
blobs directly. All TMA metadata is consumed through three pre-resolved JSON artifacts
under `traces/extra_info/`:

- `tma_descriptor_configs.json` — normalized tensor-map semantic families
- `tma_descriptor_resolver.json` — maps `(unique_function_id, pc_hex, handle_hi_hex)` → `config_id`
- `tma_operand_resolver.json` — operand roles and runtime-observed values for
  bulk forms and descriptor-linked forms like `UTMAPF`

The simulator-facing interface is a single lookup:

```cpp
const TmaConfig* resolve(uint32_t func_id, uint32_t pc_hex, uint32_t handle_hi);
```

This interface is stable regardless of how the JSON generation pipeline evolves.
The JSON complexity stays behind this wall.

**Resolver fallback chain** (from `TMA_ARCH.md`, preserved here for reference):

1. Exact `(unique_function_id, pc_hex, handle_hi_hex)` hit
2. Same-function `desc_reg` config reuse
3. Single-rank-candidate config selection

UTMASTG uses the first bracketed uniform-register pair as the desc-like handle source
(not an explicit `desc[URx]`), so its resolution may land at fallback 2 or 3.

---

## TMA Form Classification

Three operand forms require different resolver paths:

| Form | Example | Primary resolver |
|---|---|---|
| Explicit descriptor | `UTMALDG.4D [UR16], [UR8], desc[UR10]` | `tma_descriptor_resolver` + `tma_descriptor_configs` |
| Desc-like pair | `UTMASTG.4D [UR8], [UR6]` (first pair is desc-like) | same, with fallback resolution |
| Bulk operand-driven | `UBLKCP.S.G [UR28], [UR6], UR9` | `tma_operand_resolver` only |

Descriptor-backed `UBLKRED` (`UBLKRED ... desc[URx]`) uses **both** resolvers:
- descriptor config for tensor-layout semantics
- operand resolver for operand-3 size/span value (conservatively; do not apply the bulk
  `operand_3 * 16` formula to descriptor-backed forms until directly validated)

Bulk non-descriptor `UBLKRED` and `UBLKCP` use only the operand resolver:
- `covered_bytes = operand_3 * 16` (validated against microbench data)

---

## Per-Cycle `tma_unit_sm` Logic

At each SM cycle the TMA engine performs these steps in order:

1. **Accept** newly issued commands from subcores (up to `tma_cmd_accept_rate` per cycle);
   transition accepted entries from `ISSUED` → `ENQUEUED`
2. **Advance AGU** for `ENQUEUED` entries: resolve descriptor → compute `requests_total` →
   transition to `AGU_READY` (first pass: same cycle; later: model AGU lookup latency)
3. **Issue requests** for `AGU_READY` / `IN_FLIGHT` entries: increment `requests_issued`
   by up to `agu_issue_rate` per cycle; transition `AGU_READY` → `IN_FLIGHT` on first issue
4. **Consume completions**: process returning memory events; increment `requests_completed`
5. **Signal completion**: when `requests_completed == requests_total`, transition to
   `COMPLETED`, set `completion_object.ready = true`, record `cycle_last_completion`
6. **Release wait**: when a consumer warp observes the ready state, transition to
   `WAIT_SATISFIED` and free the entry

`tma_unit_sm` must advance every SM cycle regardless of tensor-core, CUDA-core, or
ordinary LD/ST occupancy. This is the structural guarantee of compute-TMA overlap.

---

## Synchronization and Consumer Wait

### First-pass (Phases 1–4)

When a warp reaches `ACQBULK` or `FENCE.VIEW.ASYNC.S`:

1. Look up the TMA completion object(s) for the issuing CTA
2. If any referenced object has `ready == false`, stall the warp
3. When `tma_unit_sm` sets `ready = true` in a subsequent cycle, the stall releases

This is sufficient to produce the correct producer-consumer overlap behavior without
full `mbarrier` fidelity.

### Phase 5 upgrade path

`TMACompletionObject` already has `expected_tx_bytes`, `completed_tx_bytes`, and
`phase` fields. Phase 5 replaces the `ready` check with:

```
ready = (completed_tx_bytes >= expected_tx_bytes) && (phase == expected_phase)
```

No structural change to `TMATransferEntry` or `tma_unit_sm` is required. Phase 5 is
an incremental elaboration of what is already correct in structure.

### Existing barriers remain unchanged

`BAR`, `MEMBAR`, `DEPBAR`, `LDGDEPBAR` continue to use their existing scoreboard paths.
TMA completion objects are a separate tracking domain. The two must not share state.

---

## What `UTMACCTL` and `UTMACMDFLUSH` Need

Both are control-only ops with no data-movement semantics:

- `UTMACCTL.PF`: prefetch control/setup state; one operand is a state token, not a
  byte-count field
- `UTMACMDFLUSH`: command/control flush; no operands carry data-movement meaning

For simulator purposes both should be routed through `m_tma_pipeline` → `tma_unit_sm`
as control commands, consuming issue bandwidth but generating zero transfer requests.
They complete in a small fixed number of cycles (1–2) and fire no completion object.
This preserves their architectural presence in the TMA command stream without adding
incorrect transfer accounting.

---

## Phasing Summary

The original seven-phase roadmap in `TMA_ARCH.md` remains correct in structure.
The reflections above sharpen two things about how early phases should be executed:

1. **Use `requests_total` from day one**, not `total_bytes`. The formula is cheap.
   The cost of correcting a `total_bytes`-based model after calibration breaks it on
   non-FA3 geometry is disproportionately high.

2. **Write `TMATransferEntry` and `TMACompletionObject` with all final fields from
   day one**, even if most fields are unused in Phases 1–2. This converts later phases
   from structural rewrites into behavioral extensions, which is the definition of an
   evolvable implementation.

Everything else in the phasing plan stands:

- Phase 1: routing skeleton (`TMA_OP`, `m_tma_pipeline`, `tma_unit_sm` shell)
- Phase 2: descriptor and command formation (JSON loader, resolver interface)
- Phase 3: GMEM → SMEM transfers (real `requests_total`-based progress)
- Phase 4: SMEM → GMEM transfers
- Phase 5: completion / `mbarrier` model (already-present fields become functional)
- Phase 6: proxy-fence ordering
- Phase 7: fidelity refinements (swizzle, boundary, multicast, calibration)
