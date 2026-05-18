# TMA Architectural Plan for the Remodeled Simulator

## Goal

Add a **new TMA architectural component** to the remodeled simulator so that Hopper/Blackwell-style TMA instructions use a **dedicated datapath and execution domain**, while non-TMA instructions continue to use the existing remodeled LD/ST path unchanged.

This is not a one-shot implementation plan. It is a staged architecture plan intended to guide incremental development.

## Core Design Decision

TMA should **not** be modeled as an extension of `LDGSTS` or as a specialized case inside the normal `ldst_unit_sm`.

Instead:

- **`LDGSTS` remains on the current LD/ST path**
- **TMA gets a new opcode class and a new SM-shared engine**
- **TMA performs descriptor-driven address generation internally**
- **TMA completion is tracked through a TMA-specific completion model**

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

Add a new architectural operation type, e.g.:

- `TMA_OP`

This should be added in the op-type definitions and assigned to Hopper TMA opcode families.

Candidate families:

- `UTMALDG`
- `UTMAPF`
- `UTMASTG`
- `UTMAREDG`
- `UBLKCP`
- `UBLKPF`
- `UBLKRED`
- likely also TMA-control families such as `UTMACCTL`, `UTMACMDFLUSH`

### 2. New Subcore-Side Pipeline

Add a new subcore-side pipeline, e.g.:

- `m_tma_pipeline`

Responsibilities:

- accept `TMA_OP`
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
- total bytes
- covered bytes for validated bulk operand-driven forms
- raw operand-3 runtime value for descriptor-backed size/span-controlled forms
- bytes completed
- swizzle/layout mode
- operand form: explicit descriptor / desc-like pair / bulk
- completion object identifier
- engine state

This becomes the primary abstraction for TMA execution.

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
6. returning data is placed into shared memory through TMA landing path
7. completion object is updated as bytes arrive
8. consumers proceed once completion state is satisfied

### SMEM -> GMEM

For descriptor-backed store forms such as `UTMASTG` / `UTMAREDG` and descriptor-backed `UBLKRED`:

1. warp issues TMA store command
2. issuing ownership ends after enqueue, while the store transfer continues under `tma_unit_sm`
3. generic-path writes to shared memory must already be visible
4. TMA engine reads from shared memory source region
5. TMA engine writes to global memory using descriptor-generated addresses and descriptor-carried tensor-layout context
6. completion object is optionally tracked if later code depends on completion

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

- add `TMA_OP`
- map Hopper TMA families to `TMA_OP`
- add `m_tma_pipeline`
- add `tma_unit_sm`
- route only TMA ops to the new path
- keep non-TMA ops unchanged

Success criterion:

- TMA instructions are no longer routed through ordinary LD/ST execution ownership
- TMA architectural ownership is separated from tensor-core, CUDA-core, and ordinary LD/ST execution paths even before full transfer semantics are implemented

Test plan:

- add a trace-decode unit check that Hopper TMA mnemonics map to `TMA_OP` while `LDGSTS`, `LDG`, `STG`, `LDS`, and `STS` keep their previous routing
- add a routing-level check that `Subcore::get_fu()` sends only `TMA_OP` to `m_tma_pipeline` and leaves non-TMA memory ops on `m_memory_unit_subcore`
- add a construction/initialization check that `SM` instantiates the new TMA reception latch and `tma_unit_sm` without changing LD/ST initialization
- add a structural check that the new TMA path is modeled as a separate engine path rather than as a tensor-core or CUDA-core sub-variant
- run a non-TMA regression workload and verify there is no behavioral change in issue counts, memory-unit traffic, or cycle count beyond harmless noise
- run a TMA-containing trace and verify it no longer falls into the old LD/ST ownership path or crashes due to missing TMA execution routing

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

Success criterion:

- the TMA engine receives structured transfer commands rather than guessed ordinary memory accesses, and descriptor-backed `UBLKRED` carries both descriptor semantics and conservative operand-3 size/span metadata
- command formation is cleanly separated from later transfer progress and completion accounting

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

### Phase 3: GMEM -> SMEM TMA Data Movement

Goal:

- implement first real TMA transfers

Tasks:

- support `UTMALDG`, `UTMAPF`, `UBLKCP`, `UBLKPF`
- add AGU-based address generation
- emit bulk transfer work
- land into shared memory model
- update completion state
- advance transfer progress inside `tma_unit_sm` independently of tensor-core and CUDA-core occupancy

Success criterion:

- TMA loads/prefetches execute through the new engine and become asynchronously ready
- after enqueue, the issuing warp is free to execute later independent work while the TMA engine keeps making progress

Test plan:

- create a focused micro-trace or synthetic test with `UTMALDG`/`UTMAPF`/`UBLKCP` and verify the command enters `tma_unit_sm`, allocates an in-flight entry, and emits bulk transfer work
- add AGU tests for 1D/2D/4D or 5D descriptor cases to verify base+stride address generation is stable and deterministic
- add boundary-handling tests where tile coordinates exceed tensor bounds and confirm the selected fallback behavior is applied consistently
- verify GMEM -> SMEM TMA traffic increments new TMA counters rather than LDGSTS counters or ordinary LD/ST counters
- add an overlap test where unrelated compute continues while a TMA transfer remains in flight, and verify the transfer still progresses each SM cycle
- run a regression where `LDGSTS` is still present and confirm its old two-phase path remains unchanged while TMA uses the new engine
- compare a known FA3 or TMA-bearing trace before and after the phase and verify TMA loads reach a ready/completed state without using ordinary per-lane memory-access generation

### Phase 4: SMEM -> GMEM TMA Data Movement

Goal:

- support reverse-direction TMA behavior

Tasks:

- support `UTMASTG`, `UTMAREDG`, `UBLKRED`
- add source-side shared-memory handling
- support descriptor-backed `UBLKRED` execution setup with operand-3 size/span control preserved conservatively
- track optional completion dependencies
- preserve the same async ownership split used by load-side TMA so store transfer lifetime remains inside `tma_unit_sm`

Success criterion:

- TMA stores/reductions use the dedicated TMA engine rather than the normal store datapath, including descriptor-backed `UBLKRED` with descriptor layout plus operand-3 size/span control
- issuing a TMA store does not serialize the warp for the full transfer lifetime unless a later wait/completion dependency requires it

Test plan:

- create a focused micro-trace or synthetic test with `UTMASTG` and verify the source is taken from the TMA-side shared-memory model and the transfer is issued by `tma_unit_sm`
- add a reduction-tag propagation test for `UTMAREDG` / `UBLKRED` to ensure the reduction mode is preserved in the transfer record even if arithmetic semantics remain simplified
- add a descriptor-backed `UBLKRED` test that verifies it goes through descriptor resolution when `desc[URx]` is present and preserves operand-3 size/span runtime state without assuming the bulk-only formula
- verify ordinary `STG` and `RED*` instructions still use the normal store datapath and counters
- add a completion-dependency test where later code waits for store completion and confirm the TMA completion object changes state correctly
- add a fire-and-forget store test where no later wait occurs and confirm the simulator does not introduce unnecessary stalls
- add an overlap test where unrelated compute or memory-independent work continues while the store-side TMA transfer is still in flight

### Phase 5: TMA Completion / Barrier Model

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

### Phase 6: Proxy-Fence / Ordering Model

Goal:

- capture the separation between generic execution and async proxy

Tasks:

- add a simplified `FENCE.PROXY.ASYNC` visibility model
- model the ordering needed before TMA stores source from shared memory

Success criterion:

- TMA stores have a distinct ordering requirement from ordinary stores

Test plan:

- create an ordering-focused test where generic-path writes populate shared memory before a TMA store and verify the modeled proxy-fence path is required for correct visibility
- create a negative test that omits the proxy-fence condition and verify the simulator records an ordering hazard, delayed visibility, or the intended fallback behavior
- verify ordinary non-TMA stores are unaffected by the new proxy-fence logic
- add a regression test for TMA loads to ensure introducing store-side ordering rules does not disturb GMEM -> SMEM behavior
- compare store-heavy traces before and after the phase to ensure only TMA-store synchronization behavior changes

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
  - used to validate `(unique_function_id, pc)` lookup and fallback cases
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

- all targeted TMA opcodes decode as `TMA_OP`
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
- known `(unique_function_id, pc)` pairs resolve to expected configs
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
