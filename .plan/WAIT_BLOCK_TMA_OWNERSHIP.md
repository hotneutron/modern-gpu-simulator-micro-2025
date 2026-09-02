# WaitBlock TMA/Barrier Completion Ownership (Consolidated Batch B)

**Work type:** Exact TMA-command-to-mbarrier completion ownership design
(Batch B: B0-A through B0-D, plus B1 release ordering, metrics, and lifecycle).

**Parent index:** [WAIT_BLOCK.md](WAIT_BLOCK.md). The scope/status table for every
Batch B item (B0-A … B7, D0 … D2) and the current B0-E deadlock analysis live in
the parent index. Shared terminology and hard invariants are defined once in the
parent index. Detailed server-run evidence (`.e84`, `.e85`, `.e59`, `.e61`,
`.e62`) is preserved in [WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md).

This document absorbs the complete TMA/barrier-timing work that was previously
kept in `.trae/documents/SPIN_WAIT_FOLD_ROOT_CAUSE_PLAN.md`. It is
authoritative for all TMA-ownership work after `.e83`. Where an older SYNC, TMA,
spin-fold, or Batch B document conflicts with this document, this document wins.

The implementation objective is not to make the watchdog disappear. It is to
restore exact ownership between every asynchronous TMA LOAD and the mbarrier
generation it completes, then make WaitBlock observe that corrected state
without introducing scheduler-order bias.

No item here may weaken the existing WaitBlock generation, cursor/replay, or
semantic-work invariants. Feature-off identity is evaluated within one TMA
ownership baseline; B0-D intentionally retires the legacy-FIFO baseline for both
WaitBlock-on and WaitBlock-off execution.

Terms used throughout this document:

- **descriptor**: the 128-byte `CUtensorMap` metadata that defines a tensor's
  base, shape, strides, and element type;
- **runtime TMA-state offset**: the low bits extracted from the executed
  descriptor-family instruction's runtime state operand;
- **struct descriptor offset**: the byte position at which an exact 128-byte
  encoded descriptor blob occurs in the captured kernel-parameter struct;
- **direct mapping**: the old shortcut that equates a runtime TMA-state offset
  with a numerically equal struct descriptor offset, without a data-flow edge;
- **chain mapping**: a mapping that follows a captured `UTMACCTL.PF` source and
  destination relation from one exact struct descriptor to a runtime TMA state;
- **transaction contract**: one active `ARRIVE_EXPECT_TX` byte declaration plus
  the completion-bearing LOAD commands that name the same runtime barrier.

## 1. Unified state model

The model must distinguish three objects that the current FIFO conflates:

1. **Expected-tx announcement**
   - Created by an active `SYNCS.ARRIVE_EXPECT_TX`.
   - Identified by barrier key and current generation.
   - Adds captured `semantic_raw` bytes to that barrier generation's
     `expected_tx_bytes`.
2. **TMA transfer**
   - Created when a real LOAD instruction issues.
   - Owns memory traffic geometry (`memory_bytes`, request count, source/dest).
   - Owns one exact completion target.
3. **Completion credit**
   - Applied when that exact transfer completes.
   - Carries a separately validated `credit_bytes` value.
   - Updates only the transfer's bound barrier generation.

Minimum target representation:

```cpp
struct TmaCompletionTarget {
  bool valid = false;
  bool completed = false;
  HopperMBarrierKey key;
  uint64_t barrier_generation = 0;
  uint64_t kernel_launch_uid = 0;
  uint32_t global_cta_id = 0;
  uint32_t owner_warp_id = 0;
  uint32_t source_barrier_ureg_id = 0;
  uint64_t contract_id = 0;
  uint64_t binding_id = 0;
  uint32_t credit_bytes = 0;
};

struct TMACommand {
  // Existing command fields.
  uint32_t trace_kernel_id = 0;
  uint32_t unique_function_id = 0;
  uint64_t tma_pc = 0;
  TmaCompletionTarget completion;
};
```

The existing `TMACommand::total_bytes` remains the memory-transfer size; do
not add a second mutable copy named `memory_bytes`. Completion code receives
both `total_bytes` and `completion.credit_bytes` and asserts equality for the
current FA3 subset.

`ARRIVE_EXPECT_TX` assignment identity lives in the exact barrier object:

```cpp
// Additional fields in HopperMBarrierObject, reset only on phase flip.
uint64_t tx_contract_id = 0;
bool assignment_open = false;
uint32_t assignment_owner_warp_id = ~0u;
uint64_t assignment_generation = ~0ull;
uint64_t assignment_kernel_launch_uid = 0;
uint32_t assignment_global_cta_id = ~0u;
uint32_t assigned_tx_bytes = 0;
uint32_t outstanding_tma_bindings = 0;
```

Different barrier keys owned by one warp may have open contracts
simultaneously. This is required by K10: the warp can dispatch the LOADs for
`0x31010`, then execute `ARRIVE_EXPECT_TX` for `0x31000` before the older LOADs
reach the shared TMA unit's claim point. The LOAD payload selects the exact
barrier-local contract; neither assignment nor completion uses a FIFO.

Two fixed per-warp counters support lifecycle checks without becoming routing
authority:

```cpp
open_contracts_per_warp[warp]
outstanding_bindings_per_warp[warp]
```

The open count increments at `ARRIVE_EXPECT_TX` and decrements only when that
barrier's assigned bytes equal expected bytes. The outstanding count remains
live after assignment closes and decrements at command completion. Warp-slot
reset requires both to be zero.

Launch identity is read from the command owner warp's stamped kernel UID, not
from the SM's mutable current-kernel pointer. The current launch-scoped
diagnostic accumulator supports one TMA-bearing launch per SM at a time;
a second concurrent TMA-bearing launch fails closed until counters are split
per launch. This does not affect the current FA3 configuration, where
`gpgpu_concurrent_kernel_sm=0`.

The scalar `TmaCompletionTarget` is intentional for the currently supported
non-multicast subset. It must not be treated as a permanent one-target ISA
restriction. A future multicast implementation replaces it with a bounded
inline target array plus a target count; it must not introduce per-cycle heap
allocation. Until the destination set and credit partition are traced,
multicast remains fail-closed.

`memory_bytes` and `credit_bytes` are intentionally separate. They are equal by
default, but a difference is legal only when static ISA evidence plus dynamic
trace evidence explicitly proves that distinction. For K5 `0x96b0`, the
post-review evidence instead proves that the descriptor mapping is wrong:
after correction both values are 24,576.

The completion target key must include enough identity to reject stale reuse:

```text
trace kernel ID
runtime kernel launch UID
CTA instance/slot identity
global CTA ID
barrier address
barrier generation
contract ID
```

`HopperMBarrierKey` remains `(trace_kernel_id, local CTA slot, barrier
address)` so existing WaitBlock lookup does not change. The completion target
adds runtime launch UID and global CTA ID as reuse guards. Monotonic contract
and binding IDs are allocated per SM and are not reset on CTA-slot reuse.
Assert-before-clear makes this sufficient for current FA3's one-EXCH
contract. A future repeated-EXCH design must add an explicit initialization
epoch before relaxing it.

## 2. Barrier identity source

Completion routing must come from the TMA instruction, not from completion
order and not from the most recently observed SYNCS.

Required source rules:

- `UTMALDG`: the first destination-state operand `[URn]` is a register pair;
  capture `URn+1`, the implicit completion-barrier address.
- `UBLKCP`: the observed `UBLKCP.S.G [URn], [URm], URk` form also carries the
  completion barrier in the high half `URn+1` of its first destination-state
  pair. It does not have a separate textual barrier operand.
- `UTMAPF`/`UBLKPF`: prefetch only; no mbarrier completion target.
- `UTMASTG`, `UTMAREDG`, and `UBLKRED`: store/reduce completion uses the
  asynchronous store-group/`UTMACMDFLUSH` domain, not this load mbarrier path.
- multicast or any instruction with more than one completion destination is
  unsupported until its exact destination set and credit semantics are traced;
  fail closed.

Existing documentation claimed the runtime already implemented these rules.
It does not. The current `TMACommand` has no completion-barrier field, and
`notify_tma_completion()` receives only `(warp_id, bytes)`.

Trace-data gate history and final result:

1. The pre-`9e92fe9` protobuf had `sync` and `bar_runtime` payloads but no TMA
   completion payload. Generic register values and
   `tma_runtime_operand_debug.jsonl` were not a trustworthy replacement.
2. The static SASS proved the relationship. For K5:

   ```text
   R4  = R16 + 0x16800
   UR8 = R3                    // destination
   UR9 = R4                    // completion barrier
   SYNCS ... [R16+0x16800]
   UTMALDG.4D [UR8], ...
   ```

   K10 similarly uses the high half of each `UTMALDG`/`UBLKCP` destination
   state pair for the same contract barrier.
3. Commit `9e92fe9` added a dedicated payload with a valid bit, the 64-bit
   runtime barrier address, and the captured high-UReg ID. The field is plumbed
   through `inst_trace_t` and `trace_warp_inst_t`; `TMACommand` ownership is
   intentionally deferred to B0-D.
4. The regenerated trace manifest reports `git_sha=9e92fe9` and
   `compact_wire,prune_operands`. All 132 K5 CTAs and 384 K10 CTAs parse.
5. Every active completion-bearing LOAD contains the payload:

   ```text
   K5:  5,112 / 5,112, missing=0
   K10: 13,824 / 13,824, missing=0
   LOAD payload barrier vs active expect-tx barrier mismatches=0
   LOAD before an active contract=0
   ```

6. Do not reuse a descriptor handle, memory address, `sync_barrier_addr` from an
   unrelated SYNCS occurrence, or PC-order inference as the completion target.

The regenerated trace is sufficient for B0-A through B0-D. No additional
hardware retrace is required unless a later schema change is separately
approved.

## 3. Expected-tx and transfer-size decision

The `.e83` contract conflict is resolved:

```text
SYNCS at 0x9680:
    barrier       = 0x16c00
    expected_tx   = 24576

UTMALDG at 0x96b0:
    current descriptor mapping = box_64x128xbf16
    current modeled bytes       = 16384
    architectural transaction   = 24576
```

Evidence:

- The trace kernel type is `FlashAttnFwdSm90` with
  `TileShape_MNK=(192,128,64)` and BF16 elements.
- FA3 defines
  `TmaTransactionBytesQ = size(SmemLayoutQ) * sizeof(Element)`, where
  `SmemLayoutQ` is `(M,K)`. This gives `192 * 64 * 2 = 24576`.
- The producer calls `barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ)`
  and then issues exactly one `tma_load_Q` using that barrier.
- Across all 132 K5 CTAs, the full trace contains 5,112 completion-bearing
  LOAD occurrences. Every contract is ordered and non-overlapping.
- The `0x9680 -> 0x96b0` pair occurs 264 times. Every occurrence is
  `expected=24576` followed by exactly one `UTMALDG`; no second 8-KiB producer,
  unmatched LOAD, overwritten contract, or residual contract exists.
- Every other K5 contract sums exactly to 16,384 bytes under current metadata.
- The current `0x96b0` descriptor result is not an exact descriptor identity.
  It came from the old direct mapping defined below.
- K10 provides the required multi-producer control case. All 13,824 LOAD
  occurrences close their contracts exactly:
  - `16896 = UTMALDG(16384) + UBLKCP(512)`
  - `32768 = UTMALDG(16384) + UTMALDG(16384)`
- K5 `ufid=3` contains no static or runtime-observed multicast TMA.

Conclusion:

```text
0x96b0 is the Q transaction.
the 64x128 classification is wrong.
the repaired kernel-aware chain must either resolve it unambiguously to encode
0, the rank-4 BF16 64x192 descriptor, or abort; expected-tx independently
validates encode 0 as the only byte-compatible candidate.
memory_bytes = credit_bytes = 24576.
```

The descriptor identity failure is now understood more precisely. The old
postprocessor uses these two concepts:

```text
direct:
    runtime TMA-state offset numerically equals struct descriptor offset
    no instruction-level copy relation proves the identity

chain:
    struct descriptor -> UTMACCTL.PF source/destination
                      -> runtime TMA state -> descriptor-family command
```

For K5 `0x96b0`, runtime state offset `0x400` produces conflicting candidates:

```text
old direct shortcut:
    struct 0x400 -> encode 8 -> 64x128 BF16 -> 16,384 bytes

UTMACCTL relation:
    struct 0x1c0 -> runtime state 0x400
    encode 0 -> 64x192 BF16 -> 24,576 bytes
```

The pre-B0-B `build_tma_pc_base_map.py` checked direct before chain, so it
silently selected encode 8. The old `23/23 exact` claim measured deterministic
coverage, not semantic descriptor identity. It did not compare both candidate
sources.

Aggregated over the regenerated K5/K10 trace, the direct-vs-chain divergence is
not isolated to `0x96b0`:

```text
descriptor-family sites examined = 23
at least one chain candidate      = 23
direct candidate present         = 10
direct-only sites                = 0
selected direct/chain agreement  = 0
selected direct/chain conflicts  = 10
pre-overwrite candidates/state   = 2 or 3
```

Every site the direct shortcut could reach also has a chain candidate, no site
depends on direct alone, and on every one of the 10 overlaps the two sources
disagree. This is why direct cannot be retained even as a tie-breaker. The
matching per-site derivation is in `TMA_BASE_ADDR.md` §2.29 and
`TMA_exact_base_mapping_integration.md` M0.

The pre-B0-B chain output was not yet an authority either. Its keys omitted
`trace_kernel_id`; repeated `(unique_function_id, pc)` records from different
trace-kernel variants were merged, and a single `state offset -> descriptor`
entry overwrote earlier candidates. Before that overwrite, each observed
aggregate state key had two or three descriptor candidates. B0-B at
`0f7cbc2` therefore:

1. key static and runtime TMA metadata by
   `(trace_kernel_id, unique_function_id, pc)`;
2. preserve every `UTMACCTL.PF` source/destination edge until the dynamic kernel
   variant is selected;
3. require one exact descriptor for every executed descriptor-family site;
4. remove direct numeric equality and pool assignment as routing authorities;
5. abort on missing, duplicate, or conflicting chain identity (the **ambiguity
   gate**: a site resolves only if exactly one chain candidate survives kernel
   variant selection, otherwise sidecar generation fails closed);
6. use the independently captured transaction contract only to validate
   `descriptor bytes == expected_tx`, never as the primary identity selector;
7. correct descriptor identity, base, geometry, `memory_bytes`, and
   `credit_bytes` together.

The ambiguity gate is absolute: if two chain candidates survive
kernel-variant selection, `expected_tx` must not select the candidate whose
size happens to match. Sidecar generation aborts even when exactly one
candidate is byte-compatible.

No kernel ID, PC, barrier offset, descriptor handle ordering, or 8-KiB
correction may be hardcoded.

There was a separate B0-A sidecar regression in the regenerated trace. The
pre-B0-A `build_tma_pc_base_map.py` treated the maximum value among all
`UNIFORM` callbacks as the UBLKCP span. The new hidden completion callback was
also `UNIFORM`, so a barrier address was multiplied by 16:

```text
architectural operand-3 callback: raw=32 -> 512 bytes
completion callback:              raw=0x31010
incorrect base-map tile bytes:    0x31010 * 16 = 3,211,520
```

The simulator's command-size path remained correct because
`tma_operand_resolver.json` identified operand 3 and supplied 512 bytes, but
the corrupted base-map value changed raw-pointer mock tiling. B0-A at
`0f7cbc2` now selects the statically identified operand-3 callback, asserts
`covered_bytes == operand3_raw * 16`, and emits consistent regenerated
sidecars. It did not require retracing.

## 4. Issue-time binding algorithm

`ARRIVE_EXPECT_TX` creates one barrier-local assignment contract. Trace
instructions are in program order, but sync execution and shared-TMA-unit
claim are different pipeline points. A later ARRIVE may therefore execute
while earlier LOAD instructions are still in the TMA pipeline. The runtime
barrier identity comes from each LOAD's traced completion payload and selects
the matching barrier-local contract directly.

This algorithm belongs to B0-D. B0-C intentionally retains the current FIFO to
isolate the causal effect of restoring the missing K5 bytes.

Contract rules:

- `LOAD before ARRIVE_EXPECT_TX` aborts.
- A new expect-tx for the same barrier generation while that barrier has
  residual contract state aborts.
- A new expect-tx for a different barrier key is allowed even when the same
  warp owns another open contract.
- zero-active or predicated-off instructions create neither a contract nor a
  command target;
- the current supported subset requires the LOAD issuer warp to equal the
  `ARRIVE_EXPECT_TX` owner warp; traced cross-warp assignment is unsupported
  and aborts rather than searching another warp's contract;
- One contract may own multiple commands.
- Each command owns exactly one barrier key/generation.
- One command may not span contracts or barrier keys.
- The sum of assigned command credits must equal the contract's
  `expected_tx_bytes`.
- K5 exercises one-command contracts; K10 exercises both
  `UTMALDG + UBLKCP` and `UTMALDG + UTMALDG` contracts.

At each active `ARRIVE_EXPECT_TX`:

1. Build the existing `HopperMBarrierKey` from trace kernel, local CTA slot,
   and the SYNCS runtime barrier address.
2. Capture the current barrier generation, runtime kernel launch UID, and
   global CTA ID.
3. Require only this exact barrier generation to have no tx contract or residual
   expected/assigned/completed/outstanding state.
4. Allocate a monotonically increasing `contract_id`, record owner warp,
   generation, launch UID, global CTA ID, and `expected_tx_bytes` in the
   barrier object.
5. Increment `open_contracts_per_warp[owner]`.

At each active LOAD issue, inside `tma_unit_sm::issue()` and before the command
is pushed into `m_command_queue`:

1. Build the normal `TMACommand`, including trace kernel, function ID, TMA PC,
   and existing `total_bytes`.
2. Require `tma_completion_runtime_valid=true`, a nonzero aligned barrier
   address, and a valid captured high-UReg ID.
3. Build the completion key from the LOAD payload's barrier address plus the
   command's trace-kernel and current CTA identity.
4. Find that exact barrier without creating it; require its assignment to be
   open and its owner warp, generation, launch UID, and global CTA ID to match.
5. Set `credit_bytes=total_bytes`; any future difference requires a separately
   approved traced rule.
6. Require `assigned_tx_bytes + credit_bytes <= expected_tx_bytes`.
7. Allocate a monotonically increasing `binding_id`, copy the complete target
   into `TMACommand`, and increment assigned/outstanding accounting.
8. Close only that barrier-local assignment when assigned bytes equal expected
   bytes and decrement its owner's open-contract count. Otherwise leave that
   barrier open for the next LOAD in the same contract.
9. Push the command only after all assertions and accounting succeed.

This placement is required. Binding at `enqueue_issued_commands()` is too late
to use `warp_inst_t` directly, while binding at completion has already lost
instruction identity. `build_tma_command()` may remain metadata-only; a
separate mutating SM claim method should fill the target before queueing.

Required issue-time assertions:

```text
LOAD                         -> completion.valid
PREFETCH / STORE / CONTROL   -> !completion.valid
LOAD payload address         == active contract barrier address
completion.key.cta           == command CTA
completion.global_cta        == current global CTA
completion.kernel_launch_uid == current launch UID
completion.generation        == live barrier generation
completion.contract_id       == barrier contract ID
credit_bytes                 > 0
credit_bytes                 == command.total_bytes  // current FA3 subset
assigned_credit_bytes        <= expected_tx_bytes
one command                  -> one barrier key
```

Multiple LOAD commands may target one barrier generation only when every
instruction independently resolves to that same key. A single non-multicast
LOAD must never consume ranges belonging to different barriers.

Old traces that execute a completion-bearing LOAD without the dedicated
payload fail closed. They must not silently select the legacy FIFO. Prefetch,
store, reduction, and control commands must carry an invalid completion target.

Implementation ownership is deliberately narrow:

```text
remodeling/tma_types.h
    fixed-size target, assignment-contract, and barrier accounting fields

remodeling/sm.h / sm.cc
    contract creation, issue-time target claim, exact completion application,
    lifecycle assertions, and TMABIND diagnostics

remodeling/tma_unit_sm.cc / tma_unit_sm.h
    claim before command queue insertion, carry target through transfer state,
    and invoke exact completion with transfer UID
```

No tracer, protobuf, descriptor sidecar, or WaitBlock release-phase change is
part of B0-D. The existing `9e92fe9` payload and schema-v2 validated sidecars
are sufficient; no retrace or sidecar regeneration is required.

## 5. Completion algorithm (B0-D only)

This section is not part of the B0-C size-only diagnostic. B0-C retains
`notify_tma_completion(warp_id, completed_tx_bytes)` and the legacy FIFO
unchanged.

Replace:

```text
notify_tma_completion(warp_id, completed_tx_bytes)
```

with a command-owned operation equivalent to:

```text
complete_tma_load(command.completion, command.total_bytes, transfer_uid)
```

On completion:

1. Require a valid, not-yet-completed binding.
2. Find the exact barrier object named by the command; completion must never
   create a missing barrier object.
3. Require stored launch UID and global CTA ID to match the live SM/CTA
   incarnation.
4. Require the stored generation to equal the barrier's current generation.
   A lower value is a stale completion; a higher value is impossible.
5. Require the target contract ID to equal the barrier generation's contract
   ID and `command.total_bytes == credit_bytes`.
6. Require `completed_tx_bytes + credit_bytes <= assigned_tx_bytes` and
   `assigned_tx_bytes <= expected_tx_bytes`.
7. Mark the command target completed, decrement outstanding bindings, and add
   exactly `credit_bytes` to that object.
8. Recompute readiness and phase flip for only that barrier. Before a tx-driven
   flip, require exact equality:
   `completed == assigned == expected` and `outstanding == 0`.

There is no loop over another barrier's pending credit and no leftover-byte
spill. Completion order is therefore irrelevant.

## 6. TMA ownership and lifecycle

The following invariants hold at all times:

```text
bindings_created == bindings_completed + bindings_in_flight
contracts_opened - contracts_fully_assigned == count(barrier.assignment_open)
sum(open_contracts_per_warp) == count(barrier.assignment_open)
sum(outstanding_bindings_per_warp) == sum(barrier.outstanding_tma_bindings)
sum(assigned_credit for key/gen) <= expected_tx for key/gen
sum(completed_credit for key/gen) <= assigned_credit for key/gen
one binding_id is claimed once and completed once
one non-multicast binding names one barrier key/generation
binding launch/global-CTA identity matches the live CTA incarnation
no completion updates a different generation after a phase flip
```

At a wait that is about to hold, detect an impossible contract early:

```text
expected_tx > completed_tx
AND (
  assigned_tx < expected_tx AND no open assignment contract for key/generation
  OR
  assigned_tx == expected_tx AND outstanding_tma_bindings == 0
)
    -> [TMABIND][abort] reason=unfulfillable-tx-contract
```

An open assignment contract is not itself an error: an earlier LOAD may still
be traveling through the TMA pipeline and claim it later. The abort only fires
when accounting proves that no future supported command can satisfy the
current generation.

At CTA exit or warp-slot reuse, assert before clearing:

```text
no active WaitBlock hold
no armed partial transition
no deferred replay/progress owner
no barrier-local open assignment contract for the CTA
open_contracts_per_warp[reused warp] == 0
no outstanding command-owned binding for a reused warp slot
for every CTA barrier: assignment_open == false
for every CTA barrier: expected_tx == assigned_tx == completed_tx
for every CTA barrier: outstanding_tma_bindings == 0
no completed-but-unapplied credit
```

The same assertions apply in `clear_sync_barrier_state_for_cta()`, whole-SM
`reinit(reset_not_completed)`, and warp-slot reset. The existing code that
silently filters/clears `m_pending_tma_barrier_binds_per_warp` is deleted with
the FIFO. Cleanup may erase exact-contract state only after these assertions
pass; it must never hide an orphan binding or stale completion.

Per-SM, launch-scoped diagnostics must report:

```text
contracts_opened
contracts_fully_assigned
max_open_contracts_per_warp
bindings_created
bindings_completed
assigned_credit_bytes
completed_credit_bytes
cross_barrier_applications
stale_generation
launch_or_cta_mismatch
duplicate_completion
orphan_cleanup
```

Each SM emits one launch-scoped final line after its last kernel-owned state is
drained:

```text
[TMABIND][FINAL] status=PASS sm=N ...
```

For a clean run, every participating SM must report PASS and an offline
reduction checks the aggregate. A fatal WaitBlock path must print an unbudgeted
`[TMABIND][snapshot]` for the aborting SM before termination, because process
abort cannot guarantee normal final-summary emission. Counters reset at kernel
binding only after ownership state is empty. Binding and completion add O(1)
scalar work plus the barrier-map lookup already required by the current path.
Do not add per-cycle scans, heap allocations, atomics on every event, or a
global lock.

## 7. Deterministic WaitBlock release phase (B1)

The previous implementation called `service_wait_block_release()` from each
`Subcore::modify_warp_state()`. This let subcore traversal order determine
whether a same-cycle barrier change was observed immediately or one cycle
later.

Target SM-cycle order:

```text
1. advance shared pipelines, including TMA response/completion
2. service WaitBlock release once for every hardware warp
3. run all subcore issue phases
4. arrivals issued by a subcore become visible to all releases next cycle
```

Required properties:

- every warp is serviced exactly once per SM cycle;
- all subcores observe the same cycle-start barrier snapshot;
- a TMA completion visible at cycle start may release every matching warp that
  cycle;
- a SYNCS arrival issued during subcore issue cannot release a later-iterated
  subcore in the same cycle;
- moving the service point does not change trace cursors except through an
  already-approved WaitBlock release action.

B1 validation checks release-phase accounting and event ordering directly. A
completion present in the cycle-start snapshot may release in that cycle; an
arrival issued during subcore issue may release no warp until the next cycle.
Release fingerprint count must equal the released-hold count, and every
hardware warp must receive exactly one service visit per active SM cycle.

Implementation checkpoint (2026-09-01): the release service has moved to SM
scope immediately after shared-pipeline/TMA advancement. Release and
scheduler-side WaitBlock queries use the same immutable cycle-start generation
and phase snapshot. Runtime accounting asserts one visit per hardware warp per
active SM cycle and emits canonical release fingerprints. K5 `.e91` and K10
`.e67` pass with exact release-service visit accounting and release counts.
K5's 6,466 releases all occur at or after their target generation flip, with
zero phase/service guard aborts.

The `.e92` experiment that reversed the complete subcore issue traversal is
rejected as a B1 validation method. It changed L2/TMA arbitration and all CTA
timing, so its 85,900-cycle result cannot isolate release ordering. The option
was removed. Detailed evidence is in [WAIT_BLOCK.md](WAIT_BLOCK.md).

## 8. Startup, payload, and provenance gates

When `-wait_block_enable=1`, retain or enforce:

- exact supported sidecar schema/version;
- `validation_complete=true`, `failed=0`, non-empty runtime action maps;
- unique `(trace_kernel_id, unique_function_id, action_pc)` keys;
- complete static wait condition, effect, mask, join, and cursor proof;
- sidecar CTA coverage equal to the selected trace grid;
- fold and observe modes are mutually exclusive;
- required enhanced trace, sync resolver, TMA metadata, and completion-target
  payloads are present before kernel execution;
- exact-mode TMA metadata uses
  `(trace_kernel_id, unique_function_id, pc)` keys;
- every executed descriptor-family site has one unambiguous chain-derived
  descriptor and no direct/pool routing dependency;
- every UBLKCP command has one statically identified operand-3 callback and its
  base-map tile size equals its operand-resolver command size;
- active action instructions with missing runtime payload abort unbudgeted;
- zero-effective instructions are architectural no-ops and do not mutate
  barrier, WaitBlock, cursor, or accounting state.

The current v5 WaitBlock loader and zero-active path already implement most of
this section. The new TMA completion-target payload must join the same
fail-closed startup contract.

The integrated loader's malformed-input rejection cannot be runtime-proven by
the local Python/source-contract tests alone. The B2 startup negative matrix
would require launching the simulator binary with malformed artifacts; the B3
lifecycle negative matrix additionally requires a runtime fault-injection hook
or deliberately retained state. Both server-only negative matrices were
waived/deferred for Opt 12 closure on 2026-09-02. Existing K5/K10 runs validate
clean positive teardown, not those negative paths.

## 9. Watchdog and kernel scope

Replace hardcoded WaitBlock timing constants with explicit configuration:

```text
-wait_block_hold_watchdog_cycles 32000
-wait_block_heartbeat_cycles      4096
```

Requirements:

- print effective values in the startup banner;
- use `{active, start_cycle}` rather than cycle zero as an inactive sentinel;
- scope/reset WaitBlock totals, pace records, work counters, TMA binding
  counters, and debug budgets per runtime kernel launch;
- never inherit pending generations or exhausted log budgets across launches;
- a provably unfulfillable TMA contract aborts immediately and is not delayed
  until the WaitBlock watchdog;
- increasing the watchdog threshold is not a correctness fix.

**Implementation checkpoint.** The watchdog, heartbeat, and WaitBlock event
budget are now validated `uint64` config options. Kernel UID transitions first
assert that all WaitBlock/TMA state is drained and then reset WaitBlock totals,
release-service accounting, pace maps, semantic anchors, sync work counters,
and all sync/TMA/WaitBlock log budgets. Pending objects use their `valid` flag
as the active sentinel, so a start cycle of zero is not ambiguous. Local
source-contract tests pass; the back-to-back runtime launch test remains
pending on the simulator server.

The generic repeated-EXCH initialization epoch remains deferred because current
FA3 proves one EXCH per barrier. Existing generation identity is nevertheless
mandatory for every TMA completion binding.

## 10. Pure hold query and scheduler accounting

`peek_wait_block_hold_mode(warp_id)` is the authoritative non-mutating query for
WaitBlock state. It inspects only `m_pending_sync_waits` and must never:

- call a release service;
- evaluate or mutate named BAR state;
- call `waiting()`, `tma_flush`, cache access, or another stateful readiness
  helper;
- alter generation, counters, cursors, or scheduler order.

The SR-M0A population rule is authoritative:

```text
selected
+ eligible_not_selected
+ wait_block_hold
+ blocked_with_valid_head
+ no_valid_instruction
+ drain
+ inactive
== all hardware warp slots
```

For active warps:

```text
selected
+ eligible_not_selected
+ wait_block_hold
+ blocked_with_valid_head
+ no_valid_instruction
+ drain
== active_warps
```

A WaitBlock hold is terminal for classification. Do not inspect a future
IBuffer head, FU availability, or generic programmer-barrier state afterward.
Legacy `mbarrier_spin` metrics that call a generic barrier predicate must be
migrated to the pure query or explicitly marked non-authoritative.

**Implementation checkpoint.** The authoritative issue predicate and legacy
mbarrier-stall metric now delegate WaitBlock classification exclusively to
`peek_wait_block_hold_mode()`. The legacy alias no longer includes named-BAR
state, eliminating cross-domain double classification. SR-M0A retains the
enum-based exclusive population and per-cycle conservation checks.

## 11. Metric contracts

Keep these domains separate:

1. **Issued wait outcomes**
   - active wait instruction reports ready or not-ready once;
   - skipped residual polls are not reconstructed as issued events.
2. **Logical WaitBlock episodes**
   - immediate fold;
   - hold;
   - release;
   - skipped/folded occurrences.
3. **TMA completion ownership**
   - bindings created/assigned/completed/in-flight;
   - memory bytes and completion-credit bytes;
   - completion UID reorder observations;
   - stale generation, missing identity, over-credit, under-coverage, and
     cross-key application aborts.
4. **Exclusive scheduler population**
   - SR-M0A selected/eligible/blocked/no-valid/drain/WaitBlock classes.

Required conservation equations:

```text
captured_poll_holds == captured_poll_releases
partial_join_holds == partial_join_releases
partial_deferred_occurrences
    == partial_replayed_occurrences + partial_folded_occurrences
TMA bindings created
    == TMA bindings completed + TMA bindings in flight
cross_barrier_completion_applications == 0
```

Instruction reconciliation uses matching units:

```text
fold-off retired warp instructions
- fold-on retired warp instructions
== WaitBlock skipped warp occurrences
```

Do not compare raw skipped warp occurrences directly with active-lane
`gpu_sim_insn`.

WaitBlock summaries now expose `retired_warp_instructions`, allowing the
single-run trace contract:

```text
sum(TRACECHECK.warp_insts)
    == sum(retired_warp_instructions)
     + sum(captured_poll_advanced_occurrences)
```

Final log analysis checks this equation and the WaitBlock, TMABIND, named-BAR,
release-fingerprint, SR-M0A, CTA-count, and clean-exit invariants together.
Named-BAR teardown explicitly aborts on residual participant or deferred-credit
state independent of build assertions. BAR teardown diagnostics assemble each
summary/leak/abort record before emission and use one `fprintf(stderr, ...)`
call per completed record/block, preventing concurrent OpenMP SM diagnostics
from splitting a record across lines.

Final `7ac4d23` regression results:

```text
K5 .e93/.o93:
  14,448,016 retired + 371,587 skipped = 14,819,603 traced
  6,481 releases = 6,481 release fingerprints

K10 .e69/.o69:
  19,912,509 retired + 1,084,920 skipped = 20,997,429 traced
  9,061 releases = 9,061 release fingerprints
```

Both kernels complete all CTAs with exact release-service visit counts,
TMABIND command/byte conservation, zero named-BAR leaks, zero active final
state, and `STALLALIGN FINAL status=PASS`. One K10 BAR summary is split across
two stderr lines; the continuation preserves all four zero leak counters.

## 12. Diagnostics

Every TMA binding event must be sufficient to reconstruct the causal chain:

```text
[TMABIND][assign]
kernel_launch_uid trace_kernel global_cta cta_slot sm warp
tma_pc opcode barrier generation contract_id binding_id
memory_bytes credit_bytes expected assigned outstanding

[TMABIND][enqueue]
transfer_uid contract_id binding_id barrier generation

[TMABIND][complete]
transfer_uid and the same target identity fields
completion_cycle completed_before completed_after expected

[TMABIND][abort]
reason identity fields
pending contracts and in-flight bindings for the affected warp/CTA
```

Required abort reasons include:

```text
missing-completion-target
ambiguous-completion-target
tx-contract-unresolved
tx-contract-overassigned
stale-barrier-generation
duplicate-binding-completion
cross-barrier-credit
cta-cleanup-with-unassigned-credit
cta-cleanup-with-inflight-completion
```

Heartbeat and abort output must remain bounded during normal execution, but
fatal diagnostics are unbudgeted. Clean exit prints one grep-able aggregate:

```text
[TMABIND][FINAL] status=PASS
bindings_created=N bindings_completed=N in_flight=0
memory_bytes=X credit_bytes=Y
uid_reorders=R cross_barrier_applications=0
contract_mismatches=0 stale_generation=0
```

All accounting is event-driven. Do not add per-cycle maps, vectors, queues, or
heap allocations. Per-command fields are fixed-size; per-barrier/warp
containers are allocated with existing SM state.

## 13. Implementation order

1. **Evidence and retrace gates: complete**
   - completion payload capture is implemented at `9e92fe9`;
   - K5/K10 payload coverage is 100%, with zero barrier mismatch;
   - K5 `0x96b0` is a 24-KiB Q transaction mapped to the wrong descriptor;
   - K5 has no multicast TMA.
2. **B0-A UBLKCP sidecar repair: complete at `0f7cbc2`**
   - select operand 3 by its static callback role, not by maximum uniform value;
   - regenerate sidecars from the existing raw trace;
   - require all 8 K10 UBLKCP static sites and all 6,528 K10 dynamic
     occurrences to report operand-3 command bytes and tile bytes of 512;
   - require the separate kernel-11 UBLKCP site and its 384 occurrences to
     report 32,768 bytes;
   - require zero selections of the hidden completion callback as a size source.
3. **B0-B descriptor identity and byte restoration: complete at `0f7cbc2`**
   - introduce kernel-aware TMA metadata keys;
   - preserve and validate UTMACCTL source/destination edges;
   - prohibit direct/pool routing authority;
   - regenerate consistent base-map and operand-resolver metadata;
   - print `sites_total`, `chain_resolved`, `ambiguous`, `unresolved`,
     `direct_authority`, and `pool_authority`; require
     `chain_resolved == sites_total` and every failure count to be zero;
   - record the historical aggregate baseline of 23 sites, 10 direct overlaps,
     and 10 selected conflicts as diagnostic drift data, not as a success gate;
   - require K5 `0x96b0` to form a 24-KiB command with 192 128-byte AGU
     requests and 768 32-byte sector requests.
4. **B0-C size-only byte-conservation diagnostic: complete**
   - retain the current per-warp FIFO, completion API, and release timing;
   - make command formation, request count, and expected-minus-produced byte
     conservation the primary acceptance criteria;
   - record clean exit, watchdog location, and cycle count as diagnostic
     outcomes only because changed transfer latency may expose FIFO misrouting;
   - do not interpret either a clean exit or a new failure as proof that the
     legacy FIFO is correct or that byte restoration is insufficient.
   - observed result: K10 exits cleanly, while K5 restores the missing bytes but
     still exposes both one FIFO cross-binding event and a distinct all-live
     WaitBlock dependency cycle.
5. **B0-D exact ownership**
   - add fixed-size completion target fields to `TMACommand`;
   - create the assignment contract at active `ARRIVE_EXPECT_TX`;
   - claim the exact target in `tma_unit_sm::issue()` before queue insertion;
   - update only the bound barrier at completion;
   - remove `m_pending_tma_barrier_binds_per_warp`, anonymous FIFO
     consumption, and cross-barrier spill;
   - keep existing `TMACommand::total_bytes` as memory bytes and require
     `completion.credit_bytes == total_bytes` for current FA3.
6. **B0-D lifecycle**
   - keep assignment identity in each barrier object so one warp may own
     multiple open contracts for different exact keys;
   - use fixed per-warp open-contract and outstanding-binding counts only for
     lifecycle/accounting checks, never for target selection;
   - guard local CTA-slot reuse with runtime launch UID, global CTA ID,
     generation, and monotonic contract/binding IDs;
   - assert no orphan state before CTA cleanup, whole-SM reset, and warp reuse.
7. **B0-D validation and separate commit**
   - run focused tests, K5, and K10 without moving the WaitBlock release phase;
   - retain the current B1 timing so a regression is attributable to B0.
   - require K10 to observe `max_open_contracts_per_warp >= 2`, proving the
     `0x31010`/`0x31000` overlap was exercised rather than optimized away;
   - do not assume B0-D alone must make K5 exit: the `.e84` terminal SM has
     clean, in-order TMA completion accounting and a separate all-live
     WaitBlock cycle.
8. **B0-E K5 all-live WaitBlock cycle**
   - post-B0-D K5 `.e85` confirms that the 13-warp cycle remains with a clean
     TMA ownership snapshot;
   - treat the missing proof as CTA-global park safety, not local wait-loop
     safety: holding a locally valid consumer must not suppress an ARRIVE or
     other semantic event required to release another held warp;
   - retain the observed wait-state register provenance, but do not derive an
     exact generation ticket from its one-bit phase value;
   - compare with the historical `.e76` shared-barrier/compound-wait evidence,
     but do not assume the root cause is identical: `.e76` proved stale
     consumed-generation tracking, while `.e85` observes a global cycle with
     no runnable participating warp and `current_gen < expected_gen`;
   - design and validate the correction in a separate commit without force
     release, watchdog inflation, generation guessing, or combining it with
     B1. The current authoritative B0-E analysis and gates live in
     [WAIT_BLOCK.md](WAIT_BLOCK.md).
9. **B1 release ordering in a separate commit**
   - move release service to one SM-level cycle phase after TMA completion and
     before all subcore issue.
   - rerun normal and reversed subcore traversal, focused timing checks, K5,
     and K10 independently.
10. **Scope and metrics**
   - add configurable watchdog/heartbeat values;
   - reset launch-scoped counters/budgets;
   - complete pure-query migration and metric reconciliation.

Do not combine B0 and B1 in one commit or one causal validation step. Do not
increase the watchdog, force a barrier phase, drop a WaitBlock hold, or consume
the next contract to manufacture progress.

## 14. Verification matrix

Focused tests:

- UBLKCP callback-role selection ignores the hidden completion callback and
  yields `operand3_raw=32`, `covered_bytes=tile_bytes=512`;
- K10 reports 8/8 UBLKCP static sites and 6,528/6,528 dynamic occurrences with
  512-byte command/tile sizes;
- kernel 11 reports 1/1 UBLKCP site and 384/384 occurrences with 32,768-byte
  command/tile sizes;
- a missing or ambiguous kernel-aware UTMACCTL chain fails sidecar generation;
- `chain_resolved == sites_total`, `ambiguous=0`, `unresolved=0`,
  `direct_authority=0`, and `pool_authority=0`;
- direct and chain candidates are reported before selection, and direct/pool
  candidates cannot become exact-mode routing authorities;
- K5 size-only mode forms `0x96b0` as 24,576 bytes, 192 AGU requests, and 768
  sector requests while retaining the legacy completion FIFO;
- one 16-KiB LOAD bound to one 16-KiB barrier;
- one corrected 24-KiB Q LOAD bound to one 24-KiB barrier;
- `UTMALDG(16384) + UBLKCP(512) = 16896` bound to one K10 barrier;
- `UTMALDG(16384) + UTMALDG(16384) = 32768` bound to one K10 barrier;
- two barriers with same-warp LOAD completions returned in reverse UID order;
- a deliberately ambiguous descriptor candidate set fails sidecar generation;
- a LOAD before expect-tx, a new contract before exact assignment equality, and
  an under/over-assigned contract all abort;
- prefetch completion produces no mbarrier credit;
- store/reduce completion affects only the `UTMACMDFLUSH` domain;
- stale completion after phase advance aborts;
- CTA cleanup with unassigned or in-flight ownership aborts;
- `wait_block_enable=0` still uses exact TMA ownership and preserves non-TMA
  behavior; it is not required to match the known-wrong legacy FIFO when LOAD
  completions reorder.

Size-only B0-C primary acceptance:

```text
K5 0x96b0 commands=264
K5 0x96b0 memory_bytes=completion_input_bytes=24576
K5 global expected-minus-produced byte deficit=0
K10 UBLKCP bytes=512
K10 16896=16384+512 contracts PASS
K10 32768=16384+16384 contracts PASS
```

B0-C also records whether K5 passes the historical `.e82` failure checkpoint
at cycle 50,173 and the historical pre-size-fix `.e83` terminal checkpoint at
cycle 68,271, plus its actual final cycle and watchdog signature. The `.e84`
run passes 50,173 but aborts at 68,176, 95 cycles before the old `.e83`
checkpoint. These cycle positions are diagnostic observations, not B0-C
pass/fail criteria. The additional 8 KiB and any corrected base change transfer
latency and may expose the independent completion-order bug. Therefore a clean
run supports, but does not alone prove, the missing-byte causal hypothesis; a
changed failure does not invalidate a byte-conservation PASS.

The detailed B0-C, B0-D, and B0-E server-run evidence
(`K5 .e84/.e85`, `K10 .e59/.e61/.e62`) is preserved in
[WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md). The current authoritative B0-E
terminal analysis and required implementation gates live in
[WAIT_BLOCK.md](WAIT_BLOCK.md).

End-to-end K5 acceptance remains a later combined gate:

```text
passes cycle 50173
partial replay synthetic/reservation aborts = 0
partial replay accounting/watchdog aborts = 0
released capsules == replay drains == replay closes
WaitBlock generation overshoots/watchdog aborts = 0
clean kernel exit
[WAITBLOCK][summary] status=PASS for every completed CTA
[TMABIND][FINAL] status=PASS
```

B0-D validation intentionally establishes a new canonical TMA ownership
baseline for both `wait_block_enable=0` and `wait_block_enable=1`. Feature-off
bit identity after this point means that disabling WaitBlock adds no WaitBlock
state, cursor, scheduling, or metric effects relative to that same post-B0-D
TMA baseline. It does not mean reproducing the retired legacy FIFO.

B1-specific acceptance:

```text
feature-off run == B0-D baseline
every hardware warp serviced exactly once per SM cycle
cycle-start completion may release in the current cycle
issue-phase arrival releases no warp before the next cycle
release fingerprint count == released-hold count
trace cursor, semantic work, and conservation summaries unchanged
```

Whole-subcore issue-order reversal is explicitly not an acceptance test because
it perturbs shared-pipeline, cache, and TMA scheduling beyond B1.

Record the total-cycle delta from the B0-D baseline, but do not impose an
unsupported fixed percentage or direction. A performance threshold may be
added only after repeated unchanged-binary runs establish a measurement
envelope.

## 15. Explicit deferrals and future extension rules

The following work is not required for the current K5/K10 fix, but its boundary
is part of the design.

**Generic effect framework**

The current sidecar's exact PC/effect allow sets remain authoritative. A future
ISA-wide detector may classify wait observation, delay-only backoff,
register/predicate def-use, memory space, async launch/completion, sync
mutation, fence, control flow, and nondeterministic reads. Until that framework
is separately reviewed, unknown instructions and effects remain unsafe and
must fail closed.

**Repeated EXCH initialization**

Current FA3 validation assumes exactly one EXCH initialization per barrier. A
future implementation may support repeated initialization only with:

```text
init_epoch++
reset phase, generation, flip history, and per-phase counters
remove consumed-generation state from the old epoch
include init_epoch in pending waits and TMA completion targets
```

Reinitialization must abort when the old epoch has a pending wait, unassigned
tx contract, in-flight TMA completion, or any state that cannot be separated
from the new epoch. A completion from the old epoch must never update the new
barrier object.

**Multicast TMA completion**

Current FA3 K5/K10 contains no supported completion-bearing multicast command,
so B0-D deliberately uses one inline completion target per command and rejects
multicast. Future support must change the carrier to a bounded inline
target-array representation, with one explicit barrier key/generation and
credit share per destination. It must preserve total-credit conservation and
must not allocate dynamically in the cycle path.

**Trace regeneration**

The pre-`9e92fe9` trace did not contain a distinct runtime
completion-barrier value. The regenerated trace now contains that payload with
100% K5/K10 coverage and is the production evidence source. B0-A and B0-B
require offline sidecar regeneration from this raw trace, not another hardware
trace run.
