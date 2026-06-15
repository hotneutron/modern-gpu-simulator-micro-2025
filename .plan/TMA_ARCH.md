# TMA Architectural Plan for the Remodeled Simulator

## Goal

Add a **new TMA architectural component** to the remodeled simulator so that Hopper/Blackwell-style TMA instructions use a **dedicated datapath and execution domain**, while non-TMA instructions continue to use the existing remodeled LD/ST path unchanged.

This is not a one-shot implementation plan. It is a staged architecture plan intended to guide incremental development.

## Core Design Decision

TMA should **not** be modeled as an extension of `LDGSTS` or as a specialized case inside the normal `ldst_unit_sm`.

Instead:

- **`LDGSTS` remains on the current LD/ST path**
- **TMA gets its own opcode families and a new SM-shared engine**
- **TMA performs descriptor-driven address generation internally**
- **TMA completion is tracked through a TMA-specific completion model**
- **TMA never reuses `ldst_unit_sm` ownership or `DEPBAR`/`LDGDEPBAR` completion tracking**

This follows the architectural distinction:

- `LDGSTS` = async bulk copy managed by the normal LD/ST machinery
- `TMA` = command-triggered engine that performs address generation, transfer, and completion tracking in hardware after issue

## Architectural Intent

The remodeled simulator should represent TMA as a separate execution domain with these responsibilities:

1. **Command frontend**
   - Accepts issued TMA instructions from the subcore pipeline
   - Decodes transfer direction and transfer mode
   - Allocates an in-flight TMA entry

2. **Descriptor / AGU block**
   - Consumes the tensor descriptor, coordinates, and shared-memory pointer
   - Computes multidimensional tile addresses and bounds
   - Produces transfer geometry and request segmentation

3. **Data mover**
   - Issues bulk requests between global memory and shared memory
   - Handles GMEM -> SMEM and SMEM -> GMEM paths
   - Remains distinct from ordinary per-warp load/store coalescing

4. **Layout / landing path**
   - Represents descriptor-driven layout placement into shared memory
   - Later can model swizzle and bank-aware layout effects

5. **Completion / synchronization block**
   - Tracks transaction progress and readiness
   - Eventually models Hopper-style `mbarrier` semantics
   - Supports consumer/producer overlap with compute

### Independence From Compute Pipelines

The TMA model should explicitly preserve the architectural fact that TMA progress is independent from ordinary compute execution.

- TMA command issue may originate from a warp running on the normal subcore frontend
- but transfer progress must not be tied to tensor-core occupancy
- and it must not be tied to CUDA-core/SP/INT/DP pipeline occupancy
- after a TMA command is accepted, the SM-shared TMA engine should continue advancing it even while unrelated compute instructions execute

This means the simulator should model:

- **issue-side ownership**
  - the subcore only spends issue bandwidth to submit the command
- **execution-side ownership**
  - `tma_unit_sm` owns AGU work, request issuance, transfer progress, and completion updates
- **overlap**
  - tensor-core work, CUDA-core work, and TMA transfer progress can coexist in the same SM cycle model

## What Stays Unchanged

The following should continue to use the current remodeled path:

- `LDG`, `STG`
- `LDS`, `STS`, `LDSM`, `STSM`
- `LDGSTS`
- existing tensor-core execution
- existing ordinary memory barriers and wait-barrier scoreboarding

This preserves the current remodeled behavior for non-TMA instructions and isolates TMA-specific changes.

## Current Simulator Situation

### Existing Strengths

The remodeled simulator already has:

- a clean **subcore -> SM-shared** pipeline split
- an existing **shared LD/ST structure** (`ldst_unit_sm`)
- async-copy-like behavior for `LDGSTS`
- generic barrier and wait-barrier support
- trace-side TMA discovery artifacts already generated in `extra_info`

### Existing Gaps

The remodeled simulator does **not** currently have:

- an explicit TMA engine
- Hopper `mbarrier` objects
- transaction-byte tracking for async completion
- parity-based wait/test semantics
- `fence.proxy.async` semantics
- simulator-side consumption of TMA descriptor sidecar metadata
- descriptor-driven address generation in the execution model

These gaps mean that faithful TMA support requires more than adding decode cases.

## Why TMA Must Be Separate From LDGSTS

`LDGSTS` in the current simulator is a two-phase memory instruction:

- it is tagged at decode
- it transitions `LOAD_STAGE -> STORE_STAGE`
- it reuses the normal LD/ST pending-request machinery

That is a reasonable model for Ampere-style async copy, but not for Hopper TMA.

TMA differs because:

- one thread or warp leader issues the command
- the engine performs address generation in hardware
- multidimensional layouts are encoded by a descriptor
- data movement proceeds independently of ordinary thread-issued load/store operations
- synchronization is tied to async transaction completion rather than only instruction retirement

Therefore, TMA should submit a **command**, not mutate into a normal load/store pipeline state machine.

## Proposed Simulator Structures

### 1. New Opcode Class

Add dedicated TMA architectural operation types:

- `TMA_LOAD_OP`
- `TMA_STORE_OP`
- `TMA_MISCELLANEOUS_OP`

This should be added in the op-type definitions and assigned to Hopper TMA opcode families.

Candidate families:

- **`TMA_LOAD_OP`**
  - `UTMALDG`
  - `UTMAPF`
  - `UBLKCP`
  - `UBLKPF`
- **`TMA_STORE_OP`**
  - `UTMASTG`
  - `UTMAREDG`
  - `UBLKRED`
- **`TMA_MISCELLANEOUS_OP`**
  - `UTMACCTL`
  - `UTMACMDFLUSH`

Do not add `TMA_RED_OP` or `TMA_CTRL_OP` at this stage. Reduction vs. non-reduction and control vs. prefetch distinctions should remain command-level fields, not top-level execution-domain splits.

#### Per-Opcode Implementation Phase Map

This table records, for each TMA opcode family, its op-class (verified against `ISA_Def/hopper_opcode.h`), which implementation phase owns its behavior, and the current status. This is the authoritative cross-reference so work can resume after any context loss.

| Opcode | Op class | Direction / role | Owning phase | Status |
|---|---|---|---|---|
| `UTMALDG` | `TMA_LOAD_OP` | GMEM→SMEM load | **Phase 3** | ✅ Implemented + validated (real 32B-sector mf issue, L1 bypass, mbarrier completion) |
| `UBLKCP` | `TMA_LOAD_OP` | GMEM→SMEM bulk copy | **Phase 3** | ✅ Implemented + validated (same load mover; observed `family=4`, `bytes=512` complete) |
| `UTMAPF` | `TMA_LOAD_OP` | GMEM→L2 prefetch (no SMEM landing) | **Phase 4.5** | ⬜ Not implemented. Currently passthrough. Not runtime-observed in the FA3 backward trace, but must be modeled separately (L2 prefetch timing, fire-and-forget). |
| `UBLKPF` | `TMA_LOAD_OP` | bulk prefetch | **Phase 4.5** | ⬜ Not implemented (same prefetch family as `UTMAPF`). |
| `UTMASTG` | `TMA_STORE_OP` | SMEM→GMEM store | **Phase 4** | ✅ Implemented (Step 1, validation pending). Routed through the store mover (`GLOBAL_ACC_W` per sector). FA3 backward writes dQ/dK/dV via UBLKRED; pure UTMASTG not observed in that trace. |
| `UTMAREDG` | `TMA_STORE_OP` | SMEM→GMEM reduction store | **Phase 4** | ✅ Implemented (Step 1, validation pending). Reduce-store RMW (read+write per sector, non-atomic). |
| `UBLKRED` | `TMA_STORE_OP` | bulk reduce-store | **Phase 4** (both descriptor-backed and bulk non-descriptor) | ✅ Implemented (Step 1). Reduce-store RMW via covered-span size source. ⚠️ Only the descriptor-backed form is runtime-observed in FA3 backward; the bulk non-descriptor form (`covered_bytes = operand_3 * 16`) is implemented but **NOT validated** (no FA3 coverage). |
| `UTMACCTL` | `TMA_MISCELLANEOUS_OP` | control / prefetch state setup | **Phase 4.5** | ⬜ Currently passthrough. `UTMACCTL.PF` sets up state later consumed by `UTMAPF`, so the simulator must **record** that prefetch control state (not just drop it), even though it moves no bytes. |
| `UTMACMDFLUSH` | `TMA_MISCELLANEOUS_OP` | store commit-group flush | **Phase 4** | ✅ Implemented as warp-local drain-all wait (Step 2). SASS form of the store-side commit-group / wait-group completion; wait point for `UTMASTG`/`UTMAREDG`/`UBLKRED`. |

Notes:

- **`UTMALDG.MULTICAST` is explicitly out of scope** for now and is hard-blocked by an `assert` in `tma_unit_sm` (it must not be silently downgraded to a single-CTA load). Revisit only when an FA2-oriented multicast path is needed.
- **Phase 4.5** is a new sub-phase (not in the original roadmap) covering the prefetch family: `UTMAPF` / `UBLKPF` data movement *and* the `UTMACCTL.PF` state recording that feeds it. It runs **after** Phase 4 (store) completes, per the decision that store is the higher-priority gap for FA3 backward.
- `UTMACMDFLUSH` is logically part of the Phase 4 store completion mechanism (commit-group / wait-group), distinct from the mbarrier path used by loads (see "TMA load vs store use different completion mechanisms").
- **Phase 5 (mbarrier completion) was done as a separate SYNC work stream**, not as part of this TMA effort. See the reference documents linked in the Phase 5 section: [SYNC_ISA.md](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/.plan/SYNC_ISA.md) and [Full_sync_impl.md](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/.plan/Full_sync_impl.md).

### 2. New Subcore-Side Pipeline

Add a new subcore-side pipeline, e.g.:

- `m_tma_pipeline`

Responsibilities:

- accept `TMA_LOAD_OP`, `TMA_STORE_OP`, and `TMA_MISCELLANEOUS_OP`
- perform lightweight issue-side timing
- forward commands to the SM-shared TMA engine

This should be queue-based, similar in spirit to the memory subcore unit.

The important modeling rule is that `m_tma_pipeline` should represent only **command issue**, not the full transfer lifetime.

- once a command is accepted into the TMA path, the issuing warp should not remain busy just because the transfer is still in flight
- the subcore-side pipeline should therefore retire ownership quickly after enqueue, subject only to issue-side structural constraints
- long-lived transfer state belongs in the SM-shared TMA engine, not in the subcore execution latch

### 3. New SM-Shared TMA Engine

Add a new SM-shared unit, e.g.:

- `tma_unit_sm`

Responsibilities:

- receive issued TMA commands
- hold in-flight TMA transfers
- run descriptor lookup / AGU behavior
- issue bulk transfer work
- handle completion and readiness state

Architecturally it should sit alongside:

- `ldst_unit_sm`
- the shared DP structure

not inside them.

It should also advance independently each SM cycle, regardless of whether tensor-core or CUDA-core pipelines are busy in that same cycle.

Suggested per-cycle responsibilities:

- accept newly issued TMA commands from subcores
- advance descriptor/AGU work for pending entries
- issue bulk request work subject to TMA-engine bandwidth/admission limits
- consume returning progress or completion events
- update completion objects and release finished entries

### 4. TMA In-Flight Table

Add a TMA-specific in-flight structure, conceptually similar to a PRT but with different semantics, e.g.:

- `TMATransferTable`
- `TMATransferEntry`

Responsibilities:

- admission control
- command ownership
- tracking progress of a bulk tensor transfer
- tracking bytes issued / bytes completed
- connecting completion to synchronization state

This should not be a simple reuse of the LD/ST Pending Request Table.

### 5. TMA Completion Object

Add a simple TMA completion model that can later evolve into Hopper-like `mbarrier` behavior.

Possible first-pass structure:

- `TMACompletionObject`

Suggested state:

- expected bytes
- completed bytes
- phase/parity
- ready bit
- associated warp / CTA / stage

First implementation does not need to fully match Hopper encoding, but it should preserve the architectural idea that TMA completion is tied to async transaction progress rather than only to the issuing instruction’s ordinary retirement.

**Implementation note (Phase 1.5 → Phase 5):** The current skeleton uses a `std::vector<TMACompletionObject>` that grows unbounded — each new TMA command appends an entry and none are ever freed. This is acceptable for Phase 1.5 validation but must be replaced with a fixed-size pool or ring before Phase 5. The `completion_id` is currently the raw vector index; a pool design would allocate from a free-list and return the slot on WAIT_SATISFIED.

### 6. Wait-Side Stall Rule

The simulator should preserve TMA's async nature by placing the main stall point on the **consumer wait side**, not on the producer issue side.

That means:

- issuing a TMA command should usually be a short issue-side event
- the issuing warp may continue executing later independent instructions
- a consumer should stall only when it reaches a wait/test/readiness point that depends on incomplete TMA progress
- readiness must therefore be derived from the TMA completion object rather than from simple instruction retirement

This is the key rule that allows TMA transfer overlap with compute to emerge in the timing model.

## Descriptor-Driven Execution Model

TMA execution should be based on trace-side metadata rather than ordinary per-lane memory addresses.

For most Hopper-style TMA forms, the execution model should consume:

- tensor descriptor config
- tile coordinates
- source/destination memory role
- tensor rank and strides
- box/tile dimensions
- boundary policy
- optional swizzle mode

The simulator already has useful simulator-facing sidecar artifacts for this, including:

- `tma_descriptor_configs.json`
- `tma_descriptor_resolver.json`
- `tma_operand_resolver.json`

These should be loaded into the simulator’s trace metadata path and consulted during TMA decode or command creation.

The simulator should treat these JSON artifacts as the primary modeling inputs.
Raw tracer debug artifacts such as:

- `tensor_map_encode_dump.csv`
- `tensor_map_encode_blobs/*.bin`
- `tma_desc_runtime_debug.csv`
- `tma_desc_producer_debug.csv`
- `tma_runtime_operand_debug.jsonl`

are important generation/debug evidence, but they are not the main simulator-facing interface.

### Resolver Split

The updated tracing work implies that the simulator should not assume one single metadata path for all TMA-family instructions.

Instead, Phase 2+ should use a split resolver model:

- **Descriptor config table**
  - `tma_descriptor_configs.json`
  - normalized tensor-map semantics
- **Descriptor-backed consumer resolver**
  - `tma_descriptor_resolver.json`
  - maps a TMA consumer site to one normalized descriptor config
  - primary key is `(unique_function_id, pc_hex, handle_hi_hex)` when available
  - fallback evidence can include:
    - desc-like register-pair reuse
    - same-function config reuse
    - single-rank-candidate selection
- **Operand semantics resolver**
  - `tma_operand_resolver.json`
  - maps operand roles, runtime-observed values, and static decode formulas for TMA-family instructions whose behavior depends on operands beyond descriptor resolution alone

### Updated TMA Form Classification

The traced Hopper forms now split into three practical categories:

1. **Explicit descriptor-backed forms**
   - examples:
     - `UTMALDG.* ... desc[URx]`
     - descriptor-backed `UBLKRED.* ... desc[URx]`
   - these should primarily use:
     - `tma_descriptor_resolver.json`
     - `tma_descriptor_configs.json`

2. **Desc-like operand-pair forms**
   - current important case:
     - `UTMASTG.* [URa], [URb]`
  - tracing now treats the first bracketed uniform-register pair as the desc-like selector / handle source
  - for example, `UTMASTG.4D [UR8], [UR6]` uses:
    - desc-like pair: `UR8/UR9`
    - support pair: `UR6/UR7`
   - these should still resolve through the descriptor path, but with decode logic that understands the first operand pair is descriptor-like even when the opcode text does not contain `desc[URx]`

3. **Operand-driven bulk forms**
   - current important case:
     - bulk non-descriptor `UBLKRED.G.S.ADD.F32.RN [URdst], [URsrc], URspan`
   - this form should not require descriptor resolution as the primary semantic source
   - instead it should use:
     - `tma_operand_resolver.json`
   - for the validated F32 bulk form, operand 3 currently represents covered span in units of 16 bytes
   - this form is **not part of the first test suite** and can be deferred until after the descriptor-backed path is stable

### UBLKRED-Specific Rule

`UBLKRED` should no longer be treated as a single architectural form in the simulator plan.

It has at least two practically different forms:

- **Descriptor-backed `UBLKRED`**
  - example:
    - `UBLKRED.G.S.ADD.F32.RN [UR28], [UR18], UR11, desc[UR30]`
  - should use descriptor resolution plus extra operand semantics
  - operand 3 is a runtime size/span-style control operand
  - however, its exact decode rule should remain conservative until directly validated for the descriptor-backed form
- **Bulk non-descriptor `UBLKRED`**
  - example:
    - `UBLKRED.G.S.ADD.F32.RN [UR8], [UR4], UR6`
  - should use operand-resolver semantics as the primary execution source
  - this path can be deferred from the first implementation and first test suite

For the currently validated non-descriptor bulk `UBLKRED` operand interpretation:

- operand 1 = destination base
- operand 2 = source base
- operand 3 = encoded covered span / actual data size consumed by the operation

Current decode rule:

- `covered_bytes = operand_3 * 16`
- `covered_elements_f32 = operand_3 * 4`

The TMA engine should therefore allow `UBLKRED` to select between:

- descriptor-backed reduce-store mode
- operand-driven bulk reduce-store mode

For the first implementation and first test suite, prioritize the **descriptor-backed `UBLKRED`** path.
In that path:

- operand 3 must still be preserved as the runtime size/span control operand
- but the simulator should **not** hard-code the bulk non-descriptor `operand_3 * 16` rule unless and until that exact descriptor-backed form is directly validated

## Minimal Internal TMA Transfer Representation

Introduce a TMA transfer record carried from decode into the TMA engine.

Suggested contents:

- opcode family
- transfer type: load / store / prefetch / reduction / control
- direction: GMEM -> SMEM or SMEM -> GMEM
- issuing warp ID / subcore ID / SM ID
- descriptor config ID
- metadata source: descriptor resolver / operand resolver / mixed
- resolver confidence / mapping method
- rank
- coordinates
- shared-memory source/destination pointer
- requests total
- total bytes
- covered bytes for validated bulk operand-driven forms
- raw operand-3 runtime value for descriptor-backed size/span-controlled forms
- bytes completed
- swizzle/layout mode
- operand form: explicit descriptor / desc-like pair / bulk
- completion object identifier
- engine state

This becomes the primary abstraction for TMA execution.

### Canonical Struct Layouts

The plan should target compact canonical layouts from the beginning so later phases extend behavior without forcing structural rewrites.

```cpp
enum class TMAOpcodeFamily {
    UTMALDG,
    UTMAPF,
    UTMASTG,
    UTMAREDG,
    UBLKCP,
    UBLKPF,
    UBLKRED,
    UTMACCTL,
    UTMACMDFLUSH
};

enum class TMADirection {
    GMEM_TO_SMEM,
    SMEM_TO_GMEM,
    NONE
};

enum class TMATransferType {
    LOAD,
    STORE,
    PREFETCH,
    REDUCTION,
    CONTROL
};

enum class TMAMetadataSource {
    DESCRIPTOR,
    OPERAND,
    MIXED,
    NONE
};

enum class TMAOperandForm {
    EXPLICIT_DESC,
    DESC_LIKE_PAIR,
    BULK_OPERAND,
    GENERIC
};

struct TMACommand {
    uint32_t warp_id;
    uint32_t cta_id;
    uint32_t sm_id;
    uint32_t subcore_id;
    TMAOpcodeFamily opcode_family;
    TMADirection direction;
    TMATransferType transfer_type;
    std::string config_id;
    TMAMetadataSource meta_source;
    std::string mapping_method;
    float resolver_confidence;
    uint32_t rank;
    std::array<uint32_t, 5> box_dim;
    std::array<uint32_t, 5> coords;
    uint32_t element_size;
    uint64_t smem_ptr;
    uint32_t requests_total;
    uint32_t total_bytes;
    uint32_t covered_bytes;
    uint32_t operand3_raw;
    uint32_t swizzle;
    uint32_t interleave;
    uint32_t oob_fill;
    uint32_t l2_promotion;
    TMAOperandForm operand_form;
    uint32_t completion_id;
};

struct TMATransferEntry {
    enum class State {
        ISSUED,
        ENQUEUED,
        AGU_READY,
        IN_FLIGHT,
        COMPLETED,
        WAIT_SATISFIED
    };

    TMACommand cmd;
    State state;
    uint32_t requests_issued;
    uint32_t requests_completed;
    int cycle_enqueued;
    int cycle_agu_ready;
    int cycle_first_request;
    int cycle_last_completion;
    uint32_t completion_id;
};

struct TMACompletionObject {
    uint32_t expected_tx_bytes;
    uint32_t completed_tx_bytes;
    uint32_t phase;
    bool ready;
    uint32_t warp_id;
    uint32_t cta_id;
    int cycle_ready;
    // Phase 5: add expected_arrival_count and completed_arrival_count.
    // Hopper mbarrier has two independent counters (arrival count + tx-count);
    // a phase completes only when BOTH reach zero.
};
```

Field usage by phase:

- **Phase 1**
  - identity, opcode family, direction, transfer type, operand form
- **Phase 2**
  - config ID, metadata source, mapping method, rank, descriptor-derived geometry
- **Phase 3**
  - `requests_total`, `requests_issued`, `requests_completed`, `completion_id`
- **Phase 4**
  - store-side `smem_ptr`, reduction/control transfer type usage
- **Phase 5**
  - `expected_tx_bytes`, `completed_tx_bytes`, `phase`, `ready`

Even if some fields are initially populated with conservative defaults, the structure shape should match this target early so later phases remain additive.

### AGU Timing Basis

The TMA AGU timing model should be based on the number of aligned transfer requests, not on `total_bytes`.

Required quantity:

- `row_bytes = box_dim[0] * element_size`
- `requests_per_row = ceil(row_bytes / 128)`
- `outer_iters = box_dim[1] * box_dim[2] * ... * box_dim[rank - 1]`
- `requests_total = requests_per_row * outer_iters`

`total_bytes / 128` is only correct when `row_bytes` is already an exact multiple of 128. That happens to hold for some FA3 geometries, but it is not a general rule. Therefore:

- `total_bytes` should remain in the command for reference and completion-byte accounting
- but AGU / transfer progress must be driven by `requests_total`
- early phases may simplify request issue rate or per-request latency
- early phases must not simplify `requests_total` into `total_bytes / 128`

The minimum acceptable Phase 3 transfer entry should therefore carry both:

- `requests_total`
- `total_bytes`

and treat them as different quantities with different uses.

### Minimal Async State Machine

As a first implementation, each TMA transfer entry should follow a short explicit state machine so issue, transfer progress, and wait satisfaction remain clearly separated.

- **`ISSUED`**
  - the warp has issued a TMA instruction into the subcore-side TMA path
  - issue-side timing is consumed here
- **`ENQUEUED`**
  - the command has been accepted by `tma_unit_sm`
  - subcore ownership ends here, even though transfer work is not yet complete
- **`AGU_READY`**
  - descriptor/operand metadata has been resolved enough to form transfer geometry
  - the entry is ready to begin bulk request issuance
- **`IN_FLIGHT`**
  - the TMA engine is actively issuing or completing bulk transfer work
  - bytes issued and bytes completed may continue changing over multiple cycles
- **`COMPLETED`**
  - transfer-side work is done and the completion object has reached its ready condition
  - the issuing command is no longer waiting on transfer progress
- **`WAIT_SATISFIED`**
  - any dependent consumer wait/test/use point has observed the ready completion state
  - the dependent warp or stage may now proceed without further TMA-specific blocking

The key modeling distinction is:

- `ISSUED` and `ENQUEUED` are command-submission states
- `AGU_READY` and `IN_FLIGHT` are engine-progress states
- `COMPLETED` and `WAIT_SATISFIED` are readiness / consumer-observation states

This keeps the simulator from collapsing producer issue, engine progress, and consumer release into one event.

## Synchronization Plan

### Current Reusable Pieces

The simulator already has:

- `BAR` / `MEMBAR`
- wait-barrier scoreboarding
- `DEPBAR`
- `LDGDEPBAR`

These are useful as scaffolding, but they are **not equivalent** to Hopper TMA synchronization.

TMA completion must remain a separate tracking domain from:

- `DEPBAR`
- `LDGDEPBAR`
- ordinary wait-barrier / scoreboard retirement tracking

### Required New Semantics

Longer term, the simulator needs explicit support for:

- `MBAR.INIT`
- `MBAR.ARV.EXPECT_TX`
- `MBAR.TEST_WAIT.PARITY`
- `FENCE.PROXY.ASYNC`

The right plan is:

1. introduce a simplified TMA completion object first
2. connect consumer readiness to that object
3. later refine it into a more Hopper-like `mbarrier` model
4. finally add simplified proxy-fence visibility rules for TMA stores

#### Hardware gap: mbarrier has two independent completion counters

Hopper `mbarrier` tracks two independent counters, not one:

- **arrival count**: decremented by each thread arriving at the barrier (`MBAR.ARV`)
- **tx-count**: decremented by each byte arriving from async transfers (`MBAR.ARRIVE_DROP_EXPECT_TX` / `complete_tx`)

A barrier phase completes only when **both** counts reach zero. Current `TMACompletionObject` only tracks tx-count (bytes). Phase 5 must add `expected_arrival_count` and `completed_arrival_count` to model the full readiness condition.

#### Hardware gap: TMA load vs store use different completion mechanisms

These are two distinct hardware mechanisms, not variants of the same path:

- **TMA loads** (`UTMALDG`, `UTMAPF`, `UBLKCP`, `UBLKPF`): completion is signaled via mbarrier `complete_tx`. The mbarrier is the synchronization object; `MBAR.TEST_WAIT.PARITY` is the consumer wait point.
- **TMA stores** (`UTMASTG`, `UTMAREDG`, `UBLKRED`): completion is signaled via commit-group / wait-group, not via mbarrier. In SASS this corresponds to `UTMACMDFLUSH`. The consumer wait pattern is `cp.async.bulk.wait_group 0`, not mbarrier phase wait.

Phase 4 and Phase 5 must model these as separate synchronization domains, not as one unified completion model.

#### Hardware gap: fence.proxy.async is required in two contexts

`FENCE.PROXY.ASYNC` is not only required before TMA stores. It is also required **after `mbarrier.init`** to ensure the mbarrier object is visible across the async proxy before any thread attempts to use it. Phase 6 must account for both:

1. after `mbarrier.init` — ensures async proxy sees the initialized mbarrier object
2. before TMA stores — ensures prior generic-path writes to shared memory are visible to the TMA store engine

### First Async Execution Rule

Before full Hopper `mbarrier` fidelity exists, the simulator should still model a minimal async rule:

- TMA issue and TMA completion are separate events
- warp issue/retirement and transfer completion are separate events
- transfer progress is updated by `tma_unit_sm` over time
- waits are released only when the completion object reaches the required ready state

This minimal rule is sufficient to preserve the intended architectural distinction from `LDGSTS` and from ordinary load/store scoreboarding.

## Dataflow Model

### GMEM -> SMEM

For `UTMALDG` / `UTMAPF` / `UBLKCP` / `UBLKPF`:

1. warp issues TMA command
2. subcore forwards command to `tma_unit_sm`
3. issuing ownership ends after command enqueue, so the warp can continue with later independent work
4. TMA AGU resolves addresses using descriptor + coordinates
5. TMA engine emits bulk requests toward memory hierarchy
   - **Hardware note: TMA loads bypass L1 data cache.** Requests go directly to L2/DRAM, not through the per-SM L1 cache. Phase 3 memory hierarchy routing must not send TMA traffic through the L1 data cache path. TMA traffic should be counted separately from L1-cached loads.
6. returning data is placed into shared memory through TMA landing path
7. completion object is updated as bytes arrive (via mbarrier `complete_tx`)
8. consumers proceed once mbarrier tx-count AND arrival count both reach zero

### SMEM -> GMEM

For descriptor-backed store forms such as `UTMASTG` / `UTMAREDG` and descriptor-backed `UBLKRED`:

1. warp issues TMA store command
2. issuing ownership ends after enqueue, while the store transfer continues under `tma_unit_sm`
3. generic-path writes to shared memory must already be visible; `FENCE.PROXY.ASYNC` is required to ensure proxy-domain visibility
4. TMA engine reads from shared memory source region
5. TMA engine writes to global memory using descriptor-generated addresses and descriptor-carried tensor-layout context
6. completion is signaled via **commit-group / wait-group** (`UTMACMDFLUSH` in SASS), **not via mbarrier**. This is a separate completion mechanism from TMA load completion.

### Reduction Stores

`UTMAREDG` and descriptor-backed `UBLKRED` should initially be modeled as:

- store-like bulk traffic plus a reduction mode tag
- with descriptor/layout semantics preserved for any model that aims to be more than throughput-only

The detailed arithmetic semantics can be refined later if needed.

### Bulk Non-Descriptor `UBLKRED`

For the validated bulk non-descriptor form:

- `UBLKRED.G.S.ADD.F32.RN [URdst], [URsrc], URspan`

the dataflow should be modeled separately from descriptor-backed TMA stores:

1. warp issues a TMA bulk reduce-store command
2. operand resolver identifies:
   - destination base
   - source base
   - covered span encoding
3. TMA engine decodes covered bytes from operand 3
4. TMA engine performs a bulk shared-to-global reduce-store over that covered region
5. completion is tracked like other TMA stores if later code depends on it

This path should remain under the TMA engine, but it should not be forced through descriptor-config lookup as its primary semantic source.

This path is currently a **follow-on extension**, not a first-suite requirement.

## Descriptor-Resolver Validity Gate

The simulator-facing descriptor interface is still correct:

- `tma_descriptor_configs.json`
- `tma_descriptor_resolver.json`
- `tma_operand_resolver.json`

However, the current resolver-generation path still depends on a workload-specific handle-family heuristic. Previous qword-comparison attempts did not yield a useful or stable direct descriptor lookup path, so Phase 2 should treat the validated handle-family heuristic as the intended generator-side binding method for the current target traces.

Phase-2 gate:

1. verify that the handle-family heuristic remains stable on the first target traces
2. confirm FA3 / FA2 keep the expected `config_id` assignments after hard-coded literal removal
3. keep `UTMAPF` descriptor linkage one-to-one on the targeted traces
4. keep `UTMASTG` desc-like first-operand-pair handling valid on the targeted traces
5. keep descriptor-backed `UBLKRED` conservative unless its operand-3 rule is directly validated

This gate does not change the simulator lookup API, but it does determine whether the current heuristic-backed descriptor JSONs are trustworthy enough to begin Phase 2 metadata binding for the first target traces.

## Proposed File-Level Direction

### Existing files likely to change

- `gpu-simulator/gpgpu-sim/src/operation_type.h`
- `gpu-simulator/ISA_Def/hopper_opcode.h`
- `gpu-simulator/trace-driven/trace_driven.cc`
- `gpu-simulator/trace-parser/trace_parser.cc`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`
- `util/traces_enhanced/src/traced_execution.h`
- loader/support code for descriptor and operand resolver metadata

### New files likely to be added

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.h`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc`
- optionally:
  - `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_barrier.h`
  - `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_barrier.cc`
  - or `tma_completion.h/.cc`

## Staged Implementation Roadmap

### Phase 1: Architectural Skeleton

Goal:

- establish a distinct TMA execution domain with minimal behavior

Tasks:

- add `TMA_LOAD_OP`, `TMA_STORE_OP`, and `TMA_MISCELLANEOUS_OP`
- map Hopper TMA families to those TMA-specific op classes
- add `m_tma_pipeline`
- add `tma_unit_sm`
- route only TMA ops to the new path
- keep non-TMA ops unchanged

Success criterion:

- TMA instructions are no longer routed through ordinary LD/ST execution ownership
- TMA architectural ownership is separated from tensor-core, CUDA-core, and ordinary LD/ST execution paths even before full transfer semantics are implemented

Test plan:

- add a trace-decode unit check that Hopper TMA mnemonics map to `TMA_LOAD_OP`, `TMA_STORE_OP`, or `TMA_MISCELLANEOUS_OP` while `LDGSTS`, `LDG`, `STG`, `LDS`, and `STS` keep their previous routing
- add a routing-level check that `Subcore::get_fu()` sends only the three TMA op families to `m_tma_pipeline` and leaves non-TMA memory ops on `m_memory_unit_subcore`
- add a construction/initialization check that `SM` instantiates the new TMA reception latch and `tma_unit_sm` without changing LD/ST initialization
- add a structural check that the new TMA path is modeled as a separate engine path rather than as a tensor-core or CUDA-core sub-variant
- run a non-TMA regression workload and verify there is no behavioral change in issue counts, memory-unit traffic, or cycle count beyond harmless noise
- run a TMA-containing trace and verify it no longer falls into the old LD/ST ownership path or crashes due to missing TMA execution routing

### Phase 2 Gate: Descriptor Mapping Validation

Goal:

- ensure descriptor-backed TMA metadata is not silently tied to one FA3-specific handle heuristic

Tasks:

- validate the runtime-observed `handle_hi_hex` family mapping path after hard-coded literal removal
- confirm the resulting `config_id` assignments remain stable on the targeted FA3 / FA2 traces
- keep unresolved or weakly resolved descriptor-backed sites out of Phase 2 binding

Success criterion:

- descriptor-backed `config_id` resolution is justified by a validated heuristic handle-mapping path on the first target traces, with known limits documented explicitly

Test plan:

- confirm the resulting resolver JSON remains stable when the same descriptor-backed site is encountered across repeated targeted traces
- confirm FA3 / FA2 preserve the expected `config_id` assignments after heuristic cleanup
- document the supported heuristic cases and unresolved cases before enabling Phase 2 metadata binding

### Phase 2: Descriptor and Command Formation

Goal:

- make TMA commands descriptor-aware

Tasks:

- load TMA sidecar descriptor metadata
- resolve descriptor-backed forms primarily by `(unique_function_id, pc_hex, handle_hi_hex)`
- load TMA operand-resolver metadata
- classify TMA forms as descriptor-backed, desc-like-pair-backed, or operand-driven bulk
- support `UTMASTG` first-operand-pair descriptor-like resolution
- support descriptor-backed `UBLKRED` resolution plus conservative operand-3 size/span metadata attachment
- attach TMA command information to decoded instructions
- ensure the internal TMA command object represents a submitted async transfer request rather than a long-lived subcore execution record

**UTMAPF descriptor_link handling:** `UTMAPF` has no explicit `desc[URx]` operand.
Its descriptor association is stored in `tma_operand_resolver.json` under `descriptor_link.matched_descriptor.config_ids`, pointing to a later `UTMALDG` at the same `unique_function_id`.
The Phase 2 JSON loader should **pre-resolve** this link at load time and store the resolved `config_id` directly in the `TMACommand`.
There is no `descriptor_link` field in `TMACommand` and none is needed — the resolver is resolved to a `config_id` string before the command is formed.

**UBLKRED operand_form at Phase 2:** `classify_tma_operand_form` currently returns `BULK_OPERAND` for all UBLKRED sites as a static default.
The Phase 2 metadata loader must override this per-site: if `tma_descriptor_resolver.json` has an entry for the site, set `operand_form = EXPLICIT_DESC`; if only `tma_operand_resolver.json` covers the site with `operand_form = "bulk"`, keep `BULK_OPERAND`.
No change to `TMACommand` struct is needed; `operand_form` is already present.

Success criterion:

- the TMA engine receives structured transfer commands rather than guessed ordinary memory accesses, and descriptor-backed `UBLKRED` carries both descriptor semantics and conservative operand-3 size/span metadata
- command formation is cleanly separated from later transfer progress and completion accounting
- descriptor-backed lookup relies on the validated mapping path established by the Phase 2 gate rather than an undocumented FA3-only assumption

Test plan:

- add a parser test that the simulator can load `tma_descriptor_configs.json`, `tma_descriptor_resolver.json`, and `tma_operand_resolver.json` when present
- add a lookup test for descriptor-backed resolution covering:
  - exact `(unique_function_id, pc_hex, handle_hi_hex)` hit
  - same-function desc-reg reuse fallback
  - single-rank-candidate fallback
- add a decode-time test that a TMA instruction produces a populated internal TMA command object with direction, metadata source, config ID when applicable, rank, mapping method, and coordinates metadata
- add a decode/issue ownership test confirming that TMA command formation finishes at enqueue time and does not keep the issuing warp bound to full transfer lifetime
- add a `UTMASTG` test confirming the first bracketed operand pair is treated as desc-like for resolution and the second pair is preserved as support state
- add a descriptor-backed `UBLKRED ... desc[URx]` test confirming:
  - descriptor config resolution works
  - operand 3 raw runtime value is propagated as size/span metadata
  - no unsupported bulk-only decode rule is silently applied
- add negative tests confirming that ordinary load/store instructions do not try to consume TMA descriptor metadata
- run a TMA trace with debug counters/logging and verify the number of formed TMA commands matches the number of decoded TMA instructions

### Shared: TMA Transfer Size Computation (used by Phase 3, 4, 4.5)

How `total_bytes` / `requests_total` / `covered_bytes` are derived for every TMA data-movement op. This is computed once at command-formation time (Phase 2 binding, in `tma_unit_sm.cc`) and reused unchanged by the load (Phase 3), store/reduce (Phase 4), and prefetch (Phase 4.5) movers. All inputs are trace-derived metadata; asserts guarantee nonzero values for executed ops.

There are **two size sources**, selected by op family:

**A. Descriptor-geometry source — `UTMALDG`, `UTMASTG`, `UTMAPF`, `UTMAREDG`** (tensor-map / box-shaped ops)

From `infer_descriptor_total_bytes` and `infer_descriptor_request_total` (`tma_unit_sm.cc`):

```text
total_bytes    = (product of all nonzero box_dim[i]) * element_size
requests_per_row = ceil( box_dim[0] * element_size / 128 )   // 128B = MAX_MEMORY_ACCESS_SIZE
outer_iters    = product of box_dim[1..]                      // rows beyond the innermost
requests_total = requests_per_row * outer_iters              // count of 128B AGU line requests
```

- `box_dim`, `element_size` come from the resolved descriptor config (tensor map).
- `box_dim[0]` is the innermost (row) extent; the rest are outer iterations. This is why two transfers with equal `total_bytes` but different `box_dim[0]` aspect ratios produce **different** `requests_total` (AGU throughput depends on row shape, not just byte count).
- Validated example (FA3): `box_dim=[64,128]`, `element_size=2` (bf16) → `total_bytes = 64*128*2 = 16384`; the FA3 `UTMALDG` site observed `bytes=24576`, `requests=768` (after the 32B-sector expansion, see below).

**B. Covered-span source — `UBLKCP`, `UBLKPF`, `UBLKRED`** (1D bulk ops, no box geometry)

From the operand resolver (`covered_bytes`, decoded from operand 3) and `infer_request_total_from_covered_bytes`:

```text
total_bytes    = covered_bytes
requests_total = ceil( covered_bytes / 128 )
```

- `covered_bytes` is taken from `metadata.covered_bytes` (operand-3 decode), **not** recomputed from box geometry. For the non-descriptor bulk form the decode is `covered_bytes = operand_3 * 16` (16B units); for descriptor-backed `UBLKRED` the raw operand-3 value is preserved and `covered_bytes` is supplied by the resolver (the bulk `operand_3 * 16` formula is **not** assumed for the descriptor-backed form). See [TMA_TRACING.md](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/.plan/TMA_TRACING.md) "Bulk `UBLKRED` Follow-up Findings".
- Asserts enforce `total_bytes == covered_bytes` and `requests_total == ceil(covered_bytes/128)` for `UBLKCP`/`UBLKPF` and descriptor-backed `UBLKRED`.
- Validated example (FA3): `UBLKCP` site observed `bytes=512`, `requests=16`.

**Wire-level expansion (issue granularity).** `requests_total` counts **128B** AGU line requests (the address-generation throughput unit). At issue, the mover expands each 128B request into `SECTOR_CHUNCK_SIZE` (= 4) separate **32B** sector `mem_fetch`es, so the count of issued sector mfs is `requests_total * 4`. Byte accounting uses 32B per sector mf. AGU timing is still modeled in 128B units; only the wire granularity is 32B (L2-friendly, see Phase 3).

**Read vs write byte volume per op.** The size computation above gives the *moved region* size. How it maps to read/write traffic depends on direction:

| op | reads | writes | notes |
|---|---|---|---|
| `UTMALDG` / `UBLKCP` (load) | `total_bytes` from GMEM | — (lands in SMEM model) | Phase 3 |
| `UTMASTG` (pure store) | — | `total_bytes` to GMEM | Phase 4 |
| `UBLKRED` / `UTMAREDG` (reduce-store) | `covered_bytes` from GMEM (read dst) | `covered_bytes` to GMEM (write dst) | Phase 4; RMW, see Phase 4 reduction model |
| `UTMAPF` / `UBLKPF` (prefetch) | `total_bytes` into L2 only | — | Phase 4.5, fire-and-forget |

For reduce-store the moved region is read **and** written (`dst[i] += src[i]` over the covered region), so read bytes = write bytes = `covered_bytes`. This is an elementwise reduce-store, **not** a many-to-one atomic (see Phase 4 reduction model).

### Phase 3: GMEM -> SMEM TMA Data Movement  ✅ COMPLETED + VALIDATED

**Status:** Implemented and validated on the FA3 backward trace. `UTMALDG` and `UBLKCP` issue real 32B-sector `mem_fetch`es through the shared interconnect to L2 (L1 bypassed), and completion drives the Phase 5 mbarrier path. Validation evidence: thousands of `complete uid=...` events with exact byte accounting (`bytes=24576` for the 192-request UTMALDG, `bytes=512` for the 4-request UBLKCP) and zero `CACHE-FILL-MISS` / assert / signal events. `UTMAPF`/`UBLKPF` are **not** covered here — they were moved to Phase 4.5 (see phase map).

**Key implementation detail (L2-friendly issue):** the TMA mover emits each 128B AGU request as `SECTOR_CHUNCK_SIZE` (4) separate 32B sector mem_fetches, each carrying a single-bit `sector_mask` and matching `byte_mask`, exactly like a coalesced ldst request. This keeps the L2 from re-splitting the request (`breakdown_request_to_sector_requests` passes `data_size==32 && sector_mask.count()==1` unchanged), so one issued mf maps to exactly one response. AGU throughput remains modeled in 128B cache-line units (`requests_total`); only the wire-level mf granularity is 32B.

**Transfer-size derivation** for the Phase 3 loads (`total_bytes` / `requests_total`) is documented in the **Shared: TMA Transfer Size Computation** section above (`UTMALDG` uses the descriptor-geometry source, `UBLKCP` uses the covered-span source).

Goal:

- implement first real TMA transfers

Tasks:

- support `UTMALDG`, `UBLKCP` (load-side data movement). `UTMAPF`/`UBLKPF` are handled in Phase 4.5, not here.
- add AGU-based address generation
- emit bulk transfer work
- land into shared memory model
- update completion state
- advance transfer progress inside `tma_unit_sm` independently of tensor-core and CUDA-core occupancy
- drive AGU-side progress using `requests_total`, not `total_bytes / 128`
- **route TMA load traffic to bypass L1 data cache** — TMA loads go directly to L2; they must not increment L1 hit/miss counters or pass through the L1 hit/miss path

**UTMALDG.MULTICAST decision point:** `UTMALDG` has a `.MULTICAST` variant that takes an extra 16-bit `ctaMask` operand and delivers one global fetch to multiple CTAs' SMEM simultaneously. This variant is already observed in FA2 traces, but FA2 is not the first implementation target. Phase 3 may defer MULTICAST fidelity until after the first FA3-oriented path is stable. When it is addressed, the model must decide whether to represent MULTICAST as:
  - a no-op (treat as single-CTA load, suppress extra copies) — acceptable as a conservative first model
  - a bandwidth optimization (reduce redundant GMEM fetches for overlapping CTA tiles) — higher fidelity but requires cross-CTA SMEM write visibility

The choice must be documented and made explicit; silently routing MULTICAST as a plain UTMALDG without noting the suppressed ctaMask is an untracked accuracy gap.

Success criterion:

- TMA loads/prefetches execute through the new engine and become asynchronously ready
- after enqueue, the issuing warp is free to execute later independent work while the TMA engine keeps making progress

Test plan:

- create a focused micro-trace or synthetic test with `UTMALDG`/`UTMAPF`/`UBLKCP` and verify the command enters `tma_unit_sm`, allocates an in-flight entry, and emits bulk transfer work
- add AGU tests for 1D/2D/4D or 5D descriptor cases to verify base+stride address generation is stable and deterministic
- add an AGU-timing contrast test with equal `total_bytes` but different `box_dim[0]` aspect ratios and verify the modeled request count differs
- add boundary-handling tests where tile coordinates exceed tensor bounds and confirm the selected fallback behavior is applied consistently
- verify GMEM -> SMEM TMA traffic increments new TMA counters rather than LDGSTS counters or ordinary LD/ST counters
- add an overlap test where unrelated compute continues while a TMA transfer remains in flight, and verify the transfer still progresses each SM cycle
- run a regression where `LDGSTS` is still present and confirm its old two-phase path remains unchanged while TMA uses the new engine
- compare a known FA3 or TMA-bearing trace before and after the phase and verify TMA loads reach a ready/completed state without using ordinary per-lane memory-access generation

### Phase 4: SMEM -> GMEM TMA Data Movement

Goal:

- support reverse-direction (store / reduce-store) TMA data movement using the same dedicated `tma_unit_sm` engine as Phase 3, and model the store-side completion-wait correctly.

#### Implementation status (two-step)

- **Step 1 — store/reduce data movement: ✅ IMPLEMENTED (validation pending).** Store-class transfers (`UTMASTG` / `UTMAREDG` / `UBLKRED`) are routed through the same mover as loads; `mover_issue_requests` selects read / write / read+write-RMW per 32B sector (reduce = 2× sector mfs), and only `GMEM_TO_SMEM` loads credit the mbarrier `complete_tx`. Validation blocked only on a new-binary run reaching the kernel epilogue (stores first appear ~73% through the FA3 backward kernel-10 trace).
- **Step 2 — `UTMACMDFLUSH` warp-local drain-all wait: ✅ IMPLEMENTED.** `tma_unit_sm` keeps a per-warp count of outstanding store-class transfers (incremented at enqueue when `direction == SMEM_TO_GMEM`, decremented at transfer completion in `mover_on_response`; counted per **transfer/command**, not per sector mf). `SM::warp_waiting_at_tma_flush(warp, pI)` returns true iff `pI` is a `UTMACMDFLUSH` and that warp's count is > 0; the subcore issue-eligibility gate (`is_not_warp_waiting_tma_flush`) stalls only that warp until the count drains to zero. Loads are excluded (their count is never incremented). Fire-and-forget store issue is preserved — only `UTMACMDFLUSH` may stall.

#### Trace evidence (FA3 backward, verified)

Confirmed against the real backward SASS (`flash_bwd_hdim64_bf16_softcapall_sm90.sm_90a.sass`):

- The high-level PTX `cp.async.bulk.commit_group` and `cp.async.bulk.wait_group N` do **not** appear as separate SASS opcodes. The compiler folds the group commit + group wait into a single **`UTMACMDFLUSH`** instruction emitted right after a run of store/reduce ops. Example (`...:2981-2987`):

  ```
  UTMASTG.4D [UR8], [UR6]    // store 1 issued
  UTMASTG.4D [UR8], [UR14]   // store 2 issued
  UTMACMDFLUSH               // commit_group + wait_group folded here
  ```

- `UBLKRED.G.S.ADD.F32.RN ... , desc[URx]` (reduce-store) shows the same pattern: each reduce-store run is terminated by a `UTMACMDFLUSH`. This recurs at every reduction-loop tail and just before `BSYNC`/`EXIT`, i.e. `UTMACMDFLUSH` sits at natural *drain points* (right before something depends on the store result), not after every individual store.
- `UTMACMDFLUSH` is operand-less / control-only in this trace, so the `wait_group` count `N` is **not** recoverable from the trace.

#### Async semantics (NVIDIA docs) — store stays asynchronous

Per the CUDA Programming Guide completion-mechanism table and the Colfax TMA tutorial, TMA store has a **different** completion mechanism from TMA load and must remain fire-and-forget at issue:

- TMA **load** (GMEM→SMEM): completion via shared-memory barrier (`mbarrier` `complete_tx`) — already handled by the separate SYNC work (Phase 5).
- TMA **store** (SMEM→GMEM): completion via a **bulk async-group** mechanism; only the initiating thread can wait.
  - `commit_group` (`tma_store_arrive`): seals the issued stores into a group. **No stall.**
  - `wait_group N` (`tma_store_wait<N>`): stalls only until at most `N` committed groups remain pending. `N=0` = drain all. This is what allows an N-deep store pipeline and preserves latency hiding.

Therefore issuing a store must **not** stall the warp; only a flush/wait point may stall, and the async idea is preserved.

#### Modeling decision (confirmed with user)

Transfer sizes for all store/reduce ops come from the **Shared: TMA Transfer Size Computation** section above; Phase 4 does not recompute them.

- **`UTMASTG` (pure store)**: issue `GLOBAL_ACC_W` (`wr=true`) data-movement requests through the same 32B-sector, L2-friendly issue path built in Phase 3 (so L2 does not re-split), covering `total_bytes`. Then let the warp **proceed immediately** (fire-and-forget; no warp serialization at issue). Store back-pressure uses the write side of the interconnect (`m_icnt->full(size, /*write=*/true)`).
- **`UBLKRED` / `UTMAREDG` (reduce-store)**: this is an **elementwise read-modify-write** over the covered region (`dst[i] = dst[i] + src[i]`), **not** a many-to-one atomic. So:
  - `UBLKRED` vs `UTMAREDG` differ **only in data layout / size source**, not in reduce behavior: `UBLKRED` is a 1D bulk reduce-store sized from the covered-span source (operand-3 `covered_bytes`; the reduce counterpart of `UBLKCP`), while `UTMAREDG` is a tensor-map (box-shaped) reduce-store sized from the descriptor-geometry source (the reduce counterpart of `UTMASTG`, always descriptor-backed). In the FA3 backward trace only `UBLKRED` is runtime-observed; `UTMAREDG` is not present but shares the same reduce path.
  - **Both `UBLKRED` forms are implemented in Phase 4:** the descriptor-backed form (FA3-observed) and the bulk non-descriptor form (`covered_bytes = operand_3 * 16`). The non-descriptor form reuses the same covered-span size source and RMW issue path, so it is nearly free to support — but it is **NOT validated**, because it does not appear in the FA3 backward trace. Treat its byte/request accounting as unverified until a trace that exercises it is available.
  - It is modeled as a true RMW: for each 32B sector of the covered region, issue **one read** (`GLOBAL_ACC_R`, `wr=false`) to fetch the destination **and one write** (`GLOBAL_ACC_W`, `wr=true`) to store the reduced value. Read bytes = write bytes = `covered_bytes`.
  - It is **not** flagged `isatomic`. The simulator's `isatomic` path models lane-level many-to-one atomics with `do_atomic()` serialization, which is the wrong semantics here. TMA reduce-store has no cross-lane contention; it is a region-wide elementwise update. Using the atomic flag would inject incorrect contention/serialization.
  - The actual arithmetic (`+`) is **not** performed (timing-only simulator, no real SMEM payload); only the reduction tag (`ADD.F32.RN` etc.) is preserved in the transfer record, and the read+write memory traffic is modeled for bandwidth/latency.
  - `requests_completed` accounting must expect **2×** the sector count of a pure store (read mf + write mf per sector). Same fire-and-forget issue rule as `UTMASTG`.
- **`UTMACMDFLUSH`**: model as a **warp-local drain-all** wait. Because the trace does not expose `N`, treat each flush as `wait_group 0`: stall the issuing warp until **all** outstanding store/reduce transfers it has issued have reached GMEM completion, then release. This matches the observed placement of `UTMACMDFLUSH` at dependency/exit drain points and does not penalize the issue or overlap windows.
  - **Scope: applies to ALL store-class TMA, not just reduce.** The drain-all target is every outstanding transfer that warp issued with `direction == SMEM_TO_GMEM` — i.e. pure stores (`UTMASTG`) AND reduce-stores (`UBLKRED` / `UTMAREDG`) alike. This follows the hardware: `commit_group` seals the whole bulk async-store group (stores + reduce-stores together), and `wait_group` waits on that combined group. The trace confirms `UTMASTG ... UTMACMDFLUSH` appears after plain stores, not only after reductions.
  - **Loads are excluded.** Transfers with `direction == GMEM_TO_SMEM` (`UTMALDG` / `UBLKCP` / prefetch) must NOT be waited on by `UTMACMDFLUSH`; their completion is the mbarrier `complete_tx` domain. The flush only drains the SMEM->GMEM set.
- The store/reduce completion path is tracked **inside `tma_unit_sm`** (its own outstanding-transfer accounting from the `fill` callback). It does **not** use the mbarrier `complete_tx` path (that is load-only) and does **not** call `notify_tma_completion` (that drives the load-side mbarrier).

#### Out of scope / already handled

- `BSYNC` / `BSSY` (seen immediately after `UTMACMDFLUSH`) are warp **convergence/reconvergence barriers**, mapped to `BRANCH_OP` in `hopper_opcode.h` and already handled by the existing control-flow path. They are unrelated to TMA store completion and require **no** new work in this phase.

Tasks:

- support `UTMASTG`, `UTMAREDG`, `UBLKRED` as store/reduce-direction transfers in `tma_unit_sm`, reusing the Shared size computation (no recompute)
- replace the Phase-3 store-passthrough stub in `advance_in_flight_transfers` with a real store/reduce mover
- extend `mover_issue_requests` to support three issue shapes over the shared 32B-sector + sector_mask logic: load (`GLOBAL_ACC_R` only), pure store (`GLOBAL_ACC_W` only), reduce-store (`GLOBAL_ACC_R` **and** `GLOBAL_ACC_W` per sector — RMW, not atomic)
- adjust `requests_completed` / sector-goal accounting so reduce-store expects 2× the sector mf count (read + write) and store/load expect 1×
- support descriptor-backed `UBLKRED` execution setup with operand-3 size/span control preserved conservatively, and preserve the reduction tag (`ADD.F32.RN` etc.) in the transfer record even if the arithmetic is not actually performed (timing-only)
- do **not** set `isatomic` on reduce-store mfs
- implement `UTMACMDFLUSH` as a warp-local drain-all wait consumer over the store/reduce-side outstanding-transfer set
- keep the async ownership split: store/reduce issue is fire-and-forget, only `UTMACMDFLUSH` may stall

Success criterion:

- TMA stores/reductions use the dedicated TMA engine rather than the normal store datapath, including descriptor-backed `UBLKRED` with descriptor layout plus operand-3 size/span control
- issuing a TMA store does **not** serialize the warp; the warp continues to the next instruction immediately
- `UTMACMDFLUSH` stalls the issuing warp until all of that warp's outstanding TMA stores have completed, then releases
- ordinary `STG` / `RED*` instructions are unaffected and still use the normal store datapath and counters

Test plan:

- create a focused micro-trace or synthetic test with `UTMASTG` and verify the source is the TMA-side shared-memory model and the transfer is issued by `tma_unit_sm` as `GLOBAL_ACC_W`
- add a reduction-tag propagation test for `UTMAREDG` / `UBLKRED` to ensure the reduction mode is preserved in the transfer record
- add a reduce-store RMW-traffic test: confirm `UBLKRED`/`UTMAREDG` issues both `GLOBAL_ACC_R` and `GLOBAL_ACC_W` sector mfs (read bytes == write bytes == `covered_bytes`, ~2× the sector mf count of a same-size pure store) and that the mfs are **not** flagged `isatomic`
- add a descriptor-backed `UBLKRED` test that goes through descriptor resolution when `desc[URx]` is present and preserves operand-3 size/span runtime state
- add a fire-and-forget store test (store issued, no following flush) and confirm the warp does not stall at issue
- add a flush test (`UTMASTG ... UTMACMDFLUSH`) and confirm the warp stalls until store completion is reported by `fill`, then resumes
- add an overlap test where unrelated compute continues while the store-side TMA transfer is in flight, up to the flush point
- verify ordinary `STG` and `RED*` instructions still use the normal store datapath and counters
- regression: confirm the FA3 backward trace's `UTMASTG`/`UBLKRED`/`UTMACMDFLUSH` sequences are routed to the TMA engine and the run completes without asserts

### Phase 4.5: Prefetch Family (UTMAPF / UBLKPF)

**Runs after Phase 4 (store).** Split out because the prefetch family is a distinct memory behavior (GMEM→L2, no SMEM landing, no completion consumer).

> **Status (this session):** Task 1 (mis-credit fix) and Task 2 (prefetch issue classification/log) are **✅ IMPLEMENTED** (build pending). **Task 3 (UTMACCTL.PF state recording) is DROPPED** — see "UTMACCTL.PF: control-only, no state recording needed" below.

#### Trace evidence (FA3 backward, b1-s2048-hd64-nh24)

Static decode (`traces/.../extra_info/tma_discovery.json`) does contain prefetch:

- `UTMAPF.L2.4D` : 72 static occurrences, tagged `"role": "prefetch"`, form `@!UP2 UTMAPF.L2.4D [URx], [URy]` (predicate-guarded).
- `UTMACCTL.PF` : 777 static occurrences (prefetch state-setup control).

Whether `UTMAPF` *executes at runtime* is **NOT yet confirmed**. The dynamic trace (`instruction.proto`) stores only `pc` + `predicate_mask`, not the opcode string, so a binary grep cannot decide it (verified: grepping any opcode string in the `.pb` returns 0). The kernel-10 validation run observed **0** runtime `UTMAPF` in its first ~3h window, but that does NOT prove the remaining window is prefetch-free — the `@!UP2` predicate may enable it later. Treat prefetch as **possible at runtime** until a proper proto decode (pc 0x91e0 → UTMAPF) over the full kernel-10 trace says otherwise.

#### Current behavior is a latent correctness bug (must fix)

`UTMAPF` and `UBLKPF` are both mapped to `TMA_LOAD_OP` ([hopper_opcode.h:199,204](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/ISA_Def/hopper_opcode.h#L199-L204)). Consequences in the current TMA unit:

- `classify_tma_direction` → `GMEM_TO_SMEM` (same as a load); `classify_tma_transfer_type` → `PREFETCH` ([tma_unit_sm.cc:15-46](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L15-L46)).
- Subcore routes all TMA op-classes to `m_tma_pipeline` ([subcore.cc:957-961](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L957-L961)), so prefetch enters the TMA unit.
- In `advance_in_flight_transfers` the passthrough path only fires for CONTROL or `direction==NONE`; PREFETCH is neither, so it falls through to the mover and issues read traffic like a load.
- **The bug (now fixed — Task 1):** `mover_on_response` originally keyed its completion path on `direction==GMEM_TO_SMEM` and called `notify_tma_completion(warp_id, total_bytes)`. A prefetch would therefore **credit the mbarrier `complete_tx`**, corrupting the completion accounting of unrelated `UTMALDG` loads. This was harmless only while `UTMAPF` never executes; if it executed in the run window, the whole run would be invalidated. The gate is now `transfer_type==LOAD` with a separate PREFETCH branch ([tma_unit_sm.cc:734-751](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L734-L751)).

Size binding already works for prefetch (no work needed): Phase 2 treats `UTMAPF` as descriptor-backed (`config_id` + `total_bytes` + `requests_total`, [tma_unit_sm.cc:426-436](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L426-L436)) and `UBLKPF` as covered-span ([tma_unit_sm.cc:447-456](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L447-L456)).

Goal:

- model TMA L2 prefetch as fire-and-forget GMEM→L2 traffic with **no** mbarrier completion.

Tasks:

1. **Fix the completion mis-credit (highest priority).** ✅ DONE. In `mover_on_response`, `notify_tma_completion` is now gated on `transfer_type==LOAD` (not direction), and PREFETCH has its own completion branch that retires the transfer silently with a `prefetch-complete` log ([tma_unit_sm.cc:734-751](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L734-L751)). PREFETCH never credits an mbarrier.
2. **Issue prefetch traffic toward L2 only.** ✅ DONE (no datapath change needed). Prefetch reuses the Phase 3 32B-sector read issue path (`GLOBAL_ACC_R`): it has `is_store=false`/`is_reduction=false`, so it emits the same read shape as a load. It is not counted as an outstanding store (the store counter keys on `direction==SMEM_TO_GMEM`). A `prefetch-issue` log marks its issue distinctly; `enqueue`/`first-request` logs now carry `ttype` so prefetch (`ttype=2`) is distinguishable from a load in the trace.
3. ~~Record `UTMACCTL.PF` control state~~ **DROPPED.** See below.
4. Preserve `descriptor_link` resolution (unchanged): `UTMAPF` carries no explicit `desc[URx]`; its descriptor association is pre-resolved at JSON load time (Phase 2 `UTMAPF descriptor_link handling`). Size binding already works for prefetch.
5. Debug logging: ✅ `prefetch-issue` (uid/warp/family/total_bytes) and `prefetch-complete` (uid/warp/family/ttype/bytes/`mbarrier_credited=0`/cycle) added; `enqueue`/`first-request` carry `ttype`. One-event-per-episode discipline preserved.

#### UTMACCTL.PF: control-only, no state recording needed (Task 3 dropped)

`UTMACCTL.PF` is a 1-operand control op (`UTMACCTL.PF [UR6]`) whose single operand is a **state token** (`prefetch_control_state`), not a byte count — its value matches the `operand_1` of nearby `UTMAPF`/`UTMALDG` (see [TMA_TRACING.md:136-183](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/.plan/TMA_TRACING.md#L136-L183)). The original motivation for "recording its state" was that `UTMAPF` consumes that token. **But TMA_TRACING.md already concludes** the data-size path comes entirely from `UTMAPF → descriptor_link → UTMALDG → descriptor config`, so:

- `UTMACCTL.PF` needs **no** byte-movement accounting and can stay a control/passthrough op.
- Recording the token would additionally require touching the decode stage — the `.PF` suffix is collapsed into `OP_UTMACCTL` in [trace_driven.cc:314-323](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L314-L323) — for no timing-model benefit.

Therefore Task 3 is dropped; `UTMACCTL.PF` keeps its existing control-passthrough handling.

Success criterion:

- `UTMAPF`/`UBLKPF` generate prefetch traffic toward L2, distinct from ordinary loads, and **never** call `notify_tma_completion`.
- No mbarrier `complete_tx` is credited by any prefetch, even if `UTMAPF` executes at runtime.

Test plan:

- micro-trace `UTMAPF` + `UBLKPF`: verify prefetch traffic issues toward L2 (`prefetch-issue`/`prefetch-complete` logs), no mbarrier completion, no consumer stall.
- regression: re-run FA3 backward and confirm load/store/reduce/flush logs are unchanged AND that any runtime `UTMAPF` (if it now appears) produces `prefetch-*` logs with zero mbarrier credit — i.e. confirm the latent bug above is closed.
- **Before trusting Phase 4.5 validation on FA3:** decode the kernel-10 dynamic trace (pc 0x91e0) to establish whether/where `UTMAPF` actually executes; if it never executes, FA3 can only show the regression-unchanged case and a dedicated micro-trace is required for positive validation.

**Dedicated prefetch micro-trace (positive path) — harness prepared:**

The `utmapf_probe` microbench now emits **both** prefetch opcodes in every variant, so a single kernel trace exercises both Phase 4.5 size sources:
- `UTMAPF.L2.4D` (descriptor-backed tensor prefetch) — pre-existing `cp.async.bulk.prefetch.tensor.4d.L2.global.tile`.
- `UBLKPF.L2` (covered-span bulk prefetch) — new `cp.async.bulk.prefetch.L2.global [src], bytes` path added to `utmapf_probe.cu` (`issue_ublkpf`, gated by `--issue-bulk-prefetch`, default on; size via `--bulk-prefetch-bytes`, multiple of 16, clamped to the global input buffer). Verified in SASS: `@UPx UTMAPF.L2.4D` and `@UP4 UBLKPF.L2` both present alongside `UTMALDG.4D` consumers (nvcc 12.8, `arch=sm_90a`).

New job suite entry `utmapf-prefetch-both` added to [define-utmapf.yml](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/util/job_launching/apps/define-utmapf.yml) (`--issue-prefetch 1 --issue-bulk-prefetch 1 --bulk-prefetch-bytes 256 --use-load 1`).

Run procedure (requires a Hopper GPU + NVBit; the current dev box has nvcc 12.8 but no NVIDIA driver, so trace generation must run on the Hopper trace host):
1. Build: `cd gpu-app-collection/src/cuda/GPU_Microbenchmark/utmapf_probe && make` (or build the whole collection so it lands in `$GPUAPPS_ROOT/bin/$CUDA_VERSION/release/`).
2. Trace: `cd util/tracer_nvbit && ./run_hw_trace.py -B utmapf_micro -D 0` (auto-runs the TMA post-passes; `discover_tma_producers.py` already classifies both `UTMAPF`/`UBLKPF` as `"prefetch"`).
3. Simulate (run by user) with the H100 config pair `SM90_H100_L2_50MB_80GB`, then check for: `prefetch-issue`/`prefetch-complete` events with `mbarrier_credited=0`, `ttype=2` on `enqueue`/`first-request`, no consumer stall, clean exit (no deadlock).

**Validation status — FA3 bwd full run (.e304/.o304, kernel-10, sim 31 h):** ✅ **Regression PASSED, prefetch positive NOT covered.**

- Simulation terminated cleanly (`GPGPU-Sim: *** exit detected ***`) with `-gpgpu_deadlock_detect 1` active → **no deadlock**. `gpu_tot_sim_cycle=376735`, `gpu_sim_insn=629197320`, `gpu_ipc=1670.13`. Zero FATAL/assert/segfault.
- Store/reduce accounting: `store-outstanding++ == store-outstanding-- == 7296` (no leak). Flush stall: `flush-wait-enter == flush-wait-release == 1987` (every UTMACMDFLUSH stall released; no hang).
- Prefetch: `prefetch-issue == prefetch-complete == 0` — **`UTMAPF` never executed at runtime in this whole trace.** The mis-credit fix (Task 1) is therefore a latent-bug closure with no effect on FA3 bwd accuracy; its positive path (prefetch issuing real L2 traffic with zero mbarrier credit) is **still unverified** and requires the dedicated prefetch micro-trace above.

**Validation status — prefetch micro-trace (`utmapf_micro`):** ❌ **NOT VALIDATED.** Prefetch (`UTMAPF` *and* `UBLKPF`) is **not verified** in the simulator. A dedicated `utmapf_probe` micro-trace was attempted but abandoned — the verification cost was judged too high relative to its priority. The harness and findings below are kept for whoever resumes this.

**What was tried and why it stalled (descriptor dependency is the crux):**
- `UTMAPF` (`cp.async.bulk.prefetch.tensor.4d`) needs the **descriptor (tensor map)** to compute `box_dim × element_size`. In the micro-trace the TMA post-pass left the descriptor binding unresolved (`desc_refs:[]` at the top level, despite `descriptor_link.status:"matched"`). The release build has `assert` enabled, so a descriptor-required site with empty `config_id` / zero `total_bytes` does **not** become a no-op — it **aborts** (`abort`, signal 6) inside [`build_tma_command`](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L426-L436). This is the simulator-side confirmation that **descriptor-less tensor prefetch fails outright** — it cannot be positively validated from this trace.
- The same abort path also hits descriptor-required `UTMALDG` in the probe (`use-load 1`), so any kernel emitting them on this micro-trace crashes before reaching prefetch issue (observed: `prefetch-issue == 0`, sim aborted at `thread block = 0,0,0`).
- `UBLKPF` (`cp.async.bulk.prefetch.L2.global [src], bytes`) is descriptor-independent (size = explicit covered-bytes operand; the trace did capture `covered_bytes=16`), so it *could* be validated in isolation by disabling all descriptor-required sites (`--issue-prefetch 0 --use-load 0 --use-expect-tx 0 --issue-bulk-prefetch 1`). That isolated run was **not** carried out.

**To resume prefetch validation later, two independent gaps must be closed:**
1. `UBLKPF` positive path: trace + sim a UBLKPF-only kernel (descriptor sites disabled) and confirm `prefetch-issue`/`prefetch-complete` with `total_bytes>0`, `mbarrier_credited=0`, no consumer stall, clean exit.
2. `UTMAPF` positive path: root-fix the descriptor-binding post-pass so `desc_refs` is populated for generic-operand (`[URx]`) TMA sites; only then can `UTMAPF` (and descriptor-backed `UTMALDG` on the micro-trace) avoid the abort.

### Phase 5: TMA Completion / Barrier Model  ✅ COMPLETED (separate SYNC work)

**Status:** Implemented and validated as a **separate, independent work stream** (the Hopper `SYNCS` mbarrier / synchronization model), not as part of this TMA architecture effort. It was completed and full-trace validated before Phase 3 of this plan. The TMA load completion implemented in Phase 3 simply *plugs into* the mbarrier `complete_tx` path that the SYNC work already provides.

What the SYNC work delivered: `MBARRIER_OP`, address-keyed barrier objects keyed on `(trace_kernel_id, cta_id, barrier_addr)`, pending-wait tracking, classification by operand 2 (`semantic_raw`) rather than opcode suffix, bit31 phase-parity decode, the TMA-instruction→barrier-register binding rules (`dst_reg+1` for `UTMALDG`/`UTMAPF`, explicit 3rd operand for `UBLKCP`, none for stores), and the TMA-completion→barrier-progress connection.

The store-side commit-group / wait-group completion (Phase 4, `UTMACMDFLUSH`) is a **separate** mechanism and is *not* covered by this mbarrier model.

**Reference documents (SYNC work, maintained separately from this plan):**

- [SYNC_ISA.md](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/.plan/SYNC_ISA.md) — Hopper `SYNCS` mbarrier SASS-level ISA notes (operand layout, `semantic_raw` semantics, classification rules), derived from real traces incl. CUTLASS FA3 backward.
- [Full_sync_impl.md](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/.plan/Full_sync_impl.md) — the deterministic hardware-rule mbarrier implementation (TMA register binding, deadlock-free async-TMA synchronization, removal of the old heuristic Python/JSON inference).
- [FA3-enablement.md](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/.plan/FA3-enablement.md) — FA3 backward enablement notes (referenced by the SYNC work).

Goal:

- make TMA synchronization architecturally meaningful

Tasks:

- add a first-pass `mbarrier`-like object
- support expected bytes and completed bytes
- support phase/parity readiness
- connect consumer waits to the completion object
- make the main blocking point appear at consumer wait/test time rather than at producer issue time

Success criterion:

- readiness depends on transaction completion rather than simple instruction lifetime
- producer issue and transfer completion remain distinct events in the timing model

Test plan:

- add unit tests for the new TMA completion object covering initialize, expect-bytes, partial progress, complete, phase/parity flip, and reset/reuse
- add producer-consumer tests where compute is allowed to overlap with an in-flight TMA transfer and consumers only proceed after the completion state becomes ready
- verify that readiness is driven by completed bytes rather than by command issue or ordinary instruction retirement
- add a wait-side stall test confirming that a warp is blocked only when it reaches a readiness-dependent wait/use point, not simply because it previously issued a TMA command
- add regression tests confirming existing `BAR`, `MEMBAR`, `DEPBAR`, and `LDGDEPBAR` behavior is unchanged
- add a mixed-workload test containing both LDGSTS and TMA to confirm the two completion mechanisms stay independent

### Phase 6: Proxy-Fence / Ordering Model — ✅ ANALYZED, NO IMPLEMENTATION NEEDED

**Conclusion: the simulator already handles this correctly; no code change required.**

> **Trace evidence (FA3 bwd, `flash_bwd_hdim64_bf16_softcapall_sm90` SASS):**
> The exact spelling `FENCE.PROXY.ASYNC` does **not** appear. The async-proxy
> visibility fence in this kernel is **`FENCE.VIEW.ASYNC.S` (399 occurrences)**.
> Plain CTA/GPU barriers are `MEMBAR.ALL.CTA` (372) and `MEMBAR.ALL.GPU` (120).

**What `FENCE.VIEW.ASYNC.S` does (from the SASS context):** it appears almost
exclusively immediately **before an mbarrier-init store** (`@UP0 SYNCS.EXCH.64 [smem], ...`)
or just before a `BAR.ARV`. Its purpose is the generic↔async proxy split on
Hopper: a generic-proxy write (the mbarrier init) must be made visible to the
**async proxy** (the TMA hardware) before TMA can observe the initialized
mbarrier. It corresponds to PTX `fence.proxy.async`.

**Why no modeling is needed — the proxy split does not exist in this simulator:**

- The mbarrier is a single object keyed by `(kernel, cta, addr)`. `SYNCS.EXCH`
  (init) writes its fields **immediately, in the same cycle** ([sm.cc:1580-1598](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1580-L1598)). The generic-proxy writer (`EXCH`) and the async-proxy consumer (TMA `complete_tx`) read/write the **same object** — there is no separate proxy view to make consistent, so the proxy-visibility delay the fence guards against is **already zero** in the model.
- Therefore neither an "ordering gate before TMA store" nor an "mbarrier-init visibility edge" would release any latency that exists in the model; they would only add unnecessary stalls and deadlock risk against FA3's already-dense mbarrier dependencies.

**Current handling is already correct and conservative:**

- `FENCE*` decodes to `OP_FENCE`/`MEMORY_BARRIER_OP` ([hopper_opcode.h:118](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/ISA_Def/hopper_opcode.h#L118)).
- `FENCE.VIEW.ASYNC.S` is classified lightweight ([sm.cc:155-162](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L155-L162)): it **still receives the default fixed stall** of `num_cycles_to_stall_SM_at_gpu_memory_barrier` (186 cycles in the H100 config) ([sm.cc:598-604](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L598-L604)), sets the membar flag, and skips only the CTA-barrier reach logic (correct — it is not a CTA join).
- `MEMBAR.ALL.GPU` (186) / `MEMBAR.ALL.CTA` (53) get their own fixed stalls via the system/CTA flags.
- The warp is released by `warp_waiting_at_mem_barrier` once its scoreboard pending mem ops drain ([sm.cc:1763-1784](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1763-L1784)).

So the desired behavior ("a fence applies a default-cycle stall like other fences, with no false ordering hazard") is **exactly what already happens**. Validated implicitly by the FA3 bwd full run (.e304/.o304): 399 such fences executed with the default stall and the run terminated cleanly, deadlock-free.

**Phase 6 is closed as analyzed-only.** Revisit only if a future model introduces a real generic/async proxy split (separate physical views), at which point the fence would have a latency to release.

### Phase 7: Fidelity Refinements

Potential later additions:

- swizzle-aware landing effects
- bank-conflict-sensitive shared-memory placement
- cluster multicast / DSM interactions
- Blackwell-specific extensions
- bandwidth/latency calibration

Test plan:

- add calibration tests that compare simulator counters against known TMA-bearing microbenchmarks or trace-derived expectations for transfer size scaling and overlap behavior
- add swizzle/layout sensitivity tests to measure whether modeled SMEM placement changes bank-conflict or landing behavior in the intended direction
- add multicast/cluster tests, when implemented, to confirm reduced redundant global traffic compared with independent copies
- add throughput-scaling tests that increase data in flight and verify the modeled TMA engine scales differently from LDGSTS
- maintain a standing non-TMA regression suite to ensure fidelity upgrades do not perturb unrelated workloads

## Shared Test Infrastructure Plan

Because the TMA changes are large and cross-cut decode, routing, memory movement, and synchronization, every phase should reuse a common testing scaffold rather than inventing ad hoc checks.

### Recommended Microbenchmarks

The testing stack should include both **focused microbenchmarks** and **end-to-end application traces**.

Recommended benchmark categories:

- **FlashAttention traces and runs**
  - Use FlashAttention as the primary end-to-end target because it is the motivating workload and exercises real Hopper TMA patterns.
  - Keep at least one stable FA3 trace as the main regression target, especially the existing traced workload already discussed in this investigation.
  - Prefer one “small/fast” FlashAttention case for frequent iteration and one “realistic/heavier” case for milestone validation.
- **TMA load-only microbenchmarks**
  - Minimal kernels dominated by `UTMALDG` / `UTMAPF` behavior
  - Useful for validating descriptor lookup, AGU address generation, and GMEM -> SMEM completion timing without epilogue complications
- **TMA store-only microbenchmarks**
  - Minimal kernels dominated by `UTMASTG` behavior
  - Useful for validating SMEM visibility, source consumption, and completion handling for SMEM -> GMEM
- **TMA reduction-store microbenchmarks**
  - Minimal kernels that exercise `UTMAREDG` and descriptor-backed `UBLKRED`
  - Useful for validating reduction-mode propagation and operand-3-covered-size handling in the first test suite
- **LDGSTS comparison microbenchmarks**
  - Kernels that use `LDGSTS` instead of TMA
  - Useful as a control group to ensure the legacy async-copy path is not accidentally disturbed by TMA changes
- **Non-TMA tensor-core regressions**
  - Existing GEMM or tensor-core traces without TMA
  - Useful to prove the new TMA architecture is not perturbing unrelated paths

Practical benchmark grouping:

- **Tier A: developer-fast**
  - tiny synthetic TMA kernels
  - small FlashAttention trace or reduced problem-size trace
- **Tier B: milestone validation**
  - representative FlashAttention FA3 trace
  - one store-oriented TMA trace if available
  - one non-TMA tensor-core regression
- **Tier C: calibration**
  - throughput- and overlap-focused runs
  - larger FlashAttention and later Blackwell/Hopper TMA stress cases

### Trace Fixtures

The test infrastructure should maintain a stable set of reusable trace fixtures.

Recommended fixture classes:

- **Fixture 1: TMA decode fixture**
  - a tiny trace containing a few TMA opcodes across load, store, prefetch, and control families
  - used to validate opcode classification and command formation
- **Fixture 2: descriptor-resolution fixture**
  - a trace with known `tma_descriptor_configs.json` and `tma_descriptor_resolver.json` entries
  - used to validate `(unique_function_id, pc, handle_hi)` lookup and explicit zero-handle cases
- **Fixture 3: GMEM -> SMEM fixture**
  - a trace dominated by `UTMALDG` / `UTMAPF`
  - used to validate TMA load flow and completion
- **Fixture 4: SMEM -> GMEM fixture**
  - a trace dominated by `UTMASTG`
  - used to validate TMA store flow and ordering
- **Fixture 5: reduction fixture**
  - a trace containing `UTMAREDG` or descriptor-backed `UBLKRED`
  - used to validate descriptor-backed reduction-mode propagation and store-side control flow
- **Fixture 6: future bulk `UBLKRED` fixture**
  - a trace or synthetic microbench containing non-descriptor `UBLKRED [URdst], [URsrc], URspan`
  - reserved for later extension after the first suite is stable
- **Fixture 7: LDGSTS control fixture**
  - a trace that uses `LDGSTS` but no TMA
  - used to confirm the legacy async-copy path remains unchanged
- **Fixture 8: non-TMA regression fixture**
  - a tensor-core-heavy trace without TMA
  - used as a guardrail for unrelated regressions
- **Fixture 9: FlashAttention integration fixture**
  - the main FA3 trace used for end-to-end architectural validation
  - should become the primary acceptance fixture for milestone checks
- **Fixture 10: mixed TMA-form fixture**
  - a trace containing both descriptor-backed and operand-driven TMA forms when available
  - used to validate that one kernel can exercise multiple metadata paths without ambiguity

Fixture policy:

- prefer fixtures that are small enough to run frequently
- store expected metadata alongside them when possible
- keep a documented “golden” fixture list so phase-to-phase comparisons remain stable

### Debug Counters To Add

The TMA path should be instrumented from the start. Debug counters are necessary both for functional validation and for avoiding silent routing mistakes.

Recommended counter groups:

- **Decode / classification counters**
  - number of decoded TMA ops by family
  - number of non-TMA ops incorrectly classified as TMA
  - number of TMA ops falling back to generic handling
- **Descriptor-resolution counters**
  - resolver hits
  - resolver misses
  - multi-candidate fallbacks
  - missing-config fallbacks
- **Operand-resolution counters**
  - operand-resolver hits
  - operand-resolver misses
  - static-formula decode applications
  - `UBLKRED` covered-span decodes from operand 3
- **Command-formation counters**
  - TMA commands formed
  - TMA commands rejected or dropped
  - commands by direction: GMEM -> SMEM, SMEM -> GMEM
  - commands by subtype: load, prefetch, store, reduction, control
  - commands by metadata source: descriptor, operand, mixed
- **TMA-engine admission counters**
  - commands accepted by `tma_unit_sm`
  - queue full / backpressure events
  - in-flight transfer count high-water mark
- **AGU / geometry counters**
  - descriptor rank usage
  - tile bytes requested
  - boundary-handling or zero-fill events
  - swizzle mode usage
  - operand-driven covered bytes for bulk `UBLKRED`
- **Transfer-progress counters**
  - bytes issued
  - bytes completed
  - transactions issued
  - transfers completed
  - transfers canceled or failed
- **Completion / synchronization counters**
  - completion object allocations
  - completion-ready events
  - waits satisfied
  - waits blocked on incomplete bytes
  - phase/parity flips
- **Ordering counters**
  - proxy-fence-required cases
  - proxy-fence missing or hazard-detected cases
  - TMA stores delayed due to visibility/order rules
- **Path separation counters**
  - `LDGSTS` uses legacy path count
  - TMA uses dedicated path count
  - unexpected cross-path fallbacks
- **Regression guard counters**
  - ordinary LD/ST issue counts
  - ordinary LDGSTS counters
  - non-TMA tensor-core issue counts
  - cycle counts for golden traces

These counters should be exposed early even before full fidelity exists, because they provide the quickest signal that the architecture split is behaving as intended.

### Pass / Fail Criteria By Phase

Each phase should have explicit acceptance conditions beyond “it runs.”

#### Phase 1 Pass / Fail

Pass if:

- all targeted TMA opcodes decode as `TMA_LOAD_OP`, `TMA_STORE_OP`, or `TMA_MISCELLANEOUS_OP`
- only TMA ops route to the new TMA path
- non-TMA memory ops still route exactly as before
- no existing non-TMA regression trace shows unintended routing changes

Fail if:

- any TMA opcode still executes under ordinary LD/ST ownership unintentionally
- any non-TMA opcode is rerouted to the TMA path
- the simulator crashes because TMA routing is incomplete

#### Phase 2 Pass / Fail

Pass if:

- descriptor configs and resolver metadata load successfully
- known `(unique_function_id, pc, handle_hi)` tuples resolve to expected configs
- TMA commands contain the expected structured metadata
- fallback cases are explicit and counted

Fail if:

- command formation depends on guessed ordinary memrefs rather than descriptor metadata
- resolution failures are silent
- non-TMA instructions attempt to consume TMA metadata

#### Phase 3 Pass / Fail

Pass if:

- GMEM -> SMEM TMA commands allocate in-flight state and complete through `tma_unit_sm`
- consumers observe readiness only after transfer completion state is satisfied
- `LDGSTS` behavior remains unchanged on control fixtures
- FlashAttention or another TMA-bearing trace shows TMA load progress without reverting to the LDST path

Fail if:

- TMA load flow still depends on ordinary per-lane access generation
- completion is tied only to issue/retirement rather than transfer progress
- LDGSTS regressions appear

#### Phase 4 Pass / Fail

Pass if:

- SMEM -> GMEM TMA store commands use the TMA engine
- reduction-mode tags propagate correctly for reduction stores
- optional completion tracking works when enabled
- ordinary `STG` / `RED*` behavior is unchanged

Fail if:

- TMA stores silently fall back to normal store datapaths
- store completion cannot be observed when required
- non-TMA stores regress

#### Phase 5 Pass / Fail

Pass if:

- the new completion object tracks expected/completed bytes correctly
- consumer waits are released by completion state rather than instruction lifetime
- mixed LDGSTS + TMA tests show the two mechanisms remain independent

Fail if:

- completion state becomes ready before the transfer is actually complete
- existing barrier mechanisms regress
- TMA and LDGSTS completion accounting interfere with each other

#### Phase 6 Pass / Fail

Pass if:

- store-side ordering rules are enforced for TMA stores
- proxy-fence-sensitive tests distinguish ordered vs unordered cases
- non-TMA store behavior remains unchanged

Fail if:

- TMA stores can observe stale shared-memory contents under cases meant to be ordered
- ordering logic leaks into ordinary store paths

#### Phase 7 Pass / Fail

Pass if:

- fidelity features improve modeled behavior in the intended direction
- FlashAttention and calibration traces remain stable or improve in architectural plausibility
- non-TMA regression suite remains stable

Fail if:

- fidelity changes perturb unrelated workloads
- scaling or overlap behavior becomes less plausible than the simpler baseline
- debug counters no longer support root-causing regressions quickly

### FlashAttention-Specific Testing Guidance

Because FlashAttention is the main real workload target, it should be treated as a first-class validation track rather than an optional integration check.

Recommended FlashAttention policy:

- keep at least one **fast-turnaround** FA trace for frequent validation
- keep one **primary acceptance** FA3 trace for milestone signoff
- compare the following at each major phase:
  - whether the trace runs to completion
  - whether TMA op counts match expectation
  - whether TMA commands are routed to `tma_unit_sm`
  - whether completion/wait behavior looks architecturally plausible
  - whether non-TMA tensor and memory behavior remains stable

FlashAttention should not be the only test, but it should be the **main end-to-end acceptance workload**.

## Minimal Credible First Model

The first usable TMA model does **not** need to implement all Hopper features.

Recommended minimum:

- separate TMA opcode class
- separate subcore pipeline
- separate SM-shared TMA engine
- descriptor-driven command formation
- AGU-based bulk transfer model
- TMA-specific completion object
- non-TMA path unchanged

This is enough to establish the intended architecture correctly.

## Full-Fidelity Later Model

Later fidelity work may include:

- true `mbarrier` semantics
- parity waits and transaction-byte accounting
- proxy-fence ordering
- swizzle-driven shared-memory placement effects
- cluster multicast and DSM
- Blackwell throughput/coalescer features

## Open Questions

These do not block the first architectural implementation:

1. Exact TMA barrier semantics for all instruction variants
2. Exact simulator representation of proxy-fence ordering
3. Whether `ACQBULK` should be part of the TMA engine or remain a control-side instruction with TMA side effects
4. How much swizzle detail is needed in the first implementation
5. Whether cluster multicast should be ignored initially or represented as a bandwidth optimization mode

## Practical Recommendation

Implement TMA in the remodeled simulator in this order:

1. **separate architecture**
2. **descriptor-aware command formation**
3. **GMEM -> SMEM transfers**
4. **SMEM -> GMEM transfers**
5. **completion / barrier modeling**
6. **ordering and fidelity refinements**

That preserves the correct architectural split:

- **LDGSTS stays as an LD/ST-managed async copy**
- **TMA becomes a dedicated engine with its own datapath and completion model**
