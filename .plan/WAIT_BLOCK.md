# WaitBlock: Design Index and Live B0-E Work

**Status (2026-08-31):** Post-processing v5 and the partial-replay simulator are
implemented and full-trace validated. B0-A/B/C are complete. B0-D (exact TMA
completion ownership) is **complete and validated** at `39abefb`: K5 `.e85`
validates ownership-clean out-of-order completion and K10 `.e62` completes
384/384 CTAs with 132/132 clean SM-level `TMABIND` final records. The only open
correctness item is **B0-E**: a K5 all-live 13-warp cross-warp WaitBlock cycle
on SM124 (`0xb370`/`0x16c60`). Three attempts to infer an absolute wait
generation from the old trace (ordered generation ticket, ordered candidate-set,
runtime UR recompute) remain **rejected**. Generic and PC-hardcoded runtime
release experiments also remain rejected/reverted: they moved the hang to later
generations and new wait sites, and the generic release regressed K10.

A new, narrower direction is now the live B0-E candidate: capture the
**barrier-local opened-contract ordinal at each wait** for TMA-backed barriers
whose trace proves one initialization, decoded arrival count one, one
`ARRIVE_EXPECT_TX` arrival per generation, and exact B0-D byte ownership. This
is not a revival of the rejected offline-order guess: it adds the missing
cross-warp state to the trace. An audit of the existing full FA3 trace
(`git_sha=9e92fe9`) found that K5 has 2,112 ordinal-unique TMA wait streams and
3,072 ambiguous sparse-role streams, while all 11,904 K10 TMA wait streams are
ordinal-unique under a strict increasing-generation assumption. The existing
trace therefore proves both the applicability of the restricted contract model
and the need for one new captured field. **No simulator behavior change is
authorized until a regenerated FA3 trace resolves every eligible occurrence
with zero ambiguity.** The option-gated Stage E0 implementation is now present
locally; CUDA/NVBit build and the regenerated-trace acceptance gate remain
pending. B1 and later items remain pending.

This document is the authoritative index and the live execution record for
B0-E. The detailed per-work-type designs and the historical evidence live in the
linked companion files below; this file is self-contained for deciding *what to
do next* and *what must not be re-proposed*.

## Document map

| Document | Work type | Authority |
|---|---|---|
| **WAIT_BLOCK.md** (this file) | Index, shared invariants, Batch B scope table, live B0-E analysis and gates | Authoritative for status and next action |
| [WAIT_BLOCK_POSTPROC.md](WAIT_BLOCK_POSTPROC.md) | Offline trace post-processing / sidecar classification | Authoritative for the two-strategy classifier |
| [WAIT_BLOCK_SIMULATOR.md](WAIT_BLOCK_SIMULATOR.md) | Simulator runtime paths + partial-lane v2 replay | Authoritative for runtime hold/release/replay |
| [WAIT_BLOCK_TMA_OWNERSHIP.md](WAIT_BLOCK_TMA_OWNERSHIP.md) | Exact TMA-command-to-barrier completion ownership (Batch B) | Authoritative for B0-A … B0-D, B1, metrics |
| [WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md) | Chronological server runs; rejected ticket design; reverted experiment | Historical only; not implementation authority |

Supporting documents (`TMA_BASE_ADDR.md`, `TMA_exact_base_mapping_integration.md`,
`TMA_ISA.md`, `TMA_ARCH.md`, `TMA_LATENCY_INJECTION_H100.md`,
`SPIN_WAIT_SIM_MODELING.md`, `SPIN_WAIT_POSTPROCESS.md`) are historical/supporting
material. Where they conflict with the companion designs above, the companion
designs win.

**Debug records:** [`../debug-k10-wait-fold-deadlock.md`](../debug-k10-wait-fold-deadlock.md),
[`../debug-k5-waitblock-cycle.md`](../debug-k5-waitblock-cycle.md).
**Next-session entry point:** [B0E_HANDOVER.md](B0E_HANDOVER.md).

## Current B0-E authority (intentionally narrow)

1. B0-D at `39abefb` plus K5 `.e85` and K10 `.e62` is the fixed correctness
   baseline.
2. The post-`951b058` ordered-candidate experiment and the ordered-ticket design
   are historical evidence only (see [WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md)).
3. The next planned work is a **capture-only feasibility stage** for
   barrier-local contract ordinals. It must not infer a generation from current
   parity alone, reconstruct a CTA-global total order, or change simulator
   wait/release behavior.
4. A simulator event-ID implementation may be reviewed only after the new trace
   proves one release generation for every eligible K5/K10 wait occurrence.
   Any missing, duplicate, or ambiguous mapping rejects the design.
5. First-divergence timing analysis and global park-safety remain fallback and
   diagnostic work. They must not be mixed into the capture-only experiment.

## New-session handover snapshot (2026-08-31)

### Repository state

- Branch: `tma-impl`.
- The local cleanup commit with subject
  `chore(waitblock): retire probe and update B0-E plan` has not been pushed at
  the time of this handover.
- Relative to `5c575d5`, the intended net changes before Stage E0 are only:

```text
.plan/WAIT_BLOCK.md
simulator-remodeled/util/tracer_nvbit/run_hw_trace.py
```

- `run_hw_trace.py` retains the generic `-f/--force-overwrite` option.
- The `mbarrier_release_probe` source/build/YAML integration, its plain
  `NANOSLEEP` sidecar extension, and the unconfirmed inactive-lane tracer
  experiment from `7240bee` were reverted.
- The temporary K5 PC-hardcoded release code remains reverted.
- Existing unrelated untracked debug files must not be added, modified, or
  deleted as part of Stage E0.

### Fixed source-trace baseline

Use this existing full trace for all before/after comparisons:

```text
simulator-remodeled/hw_run/traces/device-0/12.8/
  flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/
  flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/
```

It is 2.0 GiB, contains K1-K11, and was generated with:

```text
tracer git SHA                 = 9e92fe9
LIGHTER_TRACER canonical mode  = compact_wire,prune_operands
K5 CTA files                   = 132
K10 CTA files                  = 384
```

Required artifacts are present: `dynamic_trace.pb`, all CTA protobufs,
`enhanced_execution_info.json`, SASS, `sync_operand_resolver.json`,
`spin_wait_sidecar.json`, `tma_operand_resolver.json`,
`tma_pc_base_map.json`, and `tma_completion_contracts.json`.

Do **not** use `-f` on this path before preserving it under a distinct baseline
directory name. Stage E0 needs both the `9e92fe9` trace and the newly captured
trace for structural comparison.

### Immediate task boundary

The next session starts at **Stage E0 only**:

```text
implement barrier-local contract-ordinal capture
implement offline uniqueness/reject analysis
regenerate one FA3 trace
compare it against the fixed source trace
```

Do not modify `sm.cc`, WaitBlock release behavior, generation comparison,
cursor/replay behavior, scheduler timing, TMA completion timing, or B1 during
Stage E0. Do not restore the deleted microbenchmark unless a separately
identified tracer primitive needs an isolated test.

## The two runtime strategies

WaitBlock modeling reduces to two runtime strategies that solve two different
trace shapes and must never share an action site:

```text
INJECT_MISSING_WAIT   ready-only captured gate; the simulator observes not-ready
                      at the entry probe, holds the warp until the modeled
                      generation arrives, then resumes the captured ready arm
                      without moving the trace cursor.

FOLD_CAPTURED_POLL    captured polling loop; the simulator leaves the entry
                      probe unchanged, applies the existing fold behavior at the
                      poll probe, and consumes only residual poll-loop
                      occurrences after the modeled release.
```

`INJECT_MISSING_WAIT` is mask-aware at runtime:

```text
modeled ready:
    normal replay, regardless of entry mask

modeled not-ready and effective entry mask covers all live lanes:
    inject a whole-warp wait hold

modeled not-ready and effective entry mask is partial:
    v1: print UNSUPPORTED_PARTIAL_MISSING_WAIT and abort
    v2: defer the blocked suffix, issue the complementary join, then hold
```

The classifier is in [WAIT_BLOCK_POSTPROC.md](WAIT_BLOCK_POSTPROC.md); the
runtime paths and the partial-lane v2 policy are in
[WAIT_BLOCK_SIMULATOR.md](WAIT_BLOCK_SIMULATOR.md).

The original failed implementation made both shapes use an entry hold and then
advanced toward semantic convergence. That caused K10 `0xa9c0` (a captured-poll
record wrongly held at the entry instead of folding at `0xb980`) and K5 `0x9e80`
(release advancing toward convergence `0x9ea0` instead of stopping before the
poll-loop exit `0xb210`). The unified WaitBlock database is still useful for
sidecar generation; unified metadata does not require unified runtime behavior.

## Terminology

- **entry probe:** first `TRYWAIT` that tests the wait condition;
- **captured ready arm:** path recorded when the entry probe was ready;
- **captured poll loop:** recorded repeated probe/backoff/backedge sequence;
- **poll probe:** probe whose result controls the poll-loop backedge;
- **poll-loop exit PC:** first instruction after the poll loop; it must execute;
- **wait hold:** remove the warp from scheduler eligibility after an issued probe
  until the modeled event is ready;
- **descriptor / runtime TMA-state offset / struct descriptor offset / direct
  mapping / chain mapping / transaction contract:** defined in
  [WAIT_BLOCK_TMA_OWNERSHIP.md](WAIT_BLOCK_TMA_OWNERSHIP.md).

## Hard invariants (apply across all companion documents)

- Every selected probe executes once before a wait hold can begin.
- The all-live `INJECT_MISSING_WAIT` path never advances either trace cursor.
- The approved partial extension never holds the whole warp at the partial
  entry. It may extract blocked-lane occurrences into an owned replay queue, but
  it never discards or semantically consumes them.
- A partial whole-warp hold begins only after the complementary join occurrence
  has actually issued.
- `FOLD_CAPTURED_POLL` never acts at the entry probe.
- Poll folding stops before `poll_loop_exit_pc`.
- Mainline, complementary-lane, reconvergence, and semantic instructions are
  never consumed by poll folding.
- Under the historical v1 implementation, a ready-only partial-mask modeled miss
  never silently continues and never becomes a whole-warp wait hold; it aborts
  with a complete diagnostic. Under the approved v2 design it may hold only after
  the complementary join occurrence has issued and all extracted blocked
  occurrences are owned by the replay queue.
- Unticketed generation overshoot and a 32,000-cycle wait-hold timeout abort.
  A future exact-ticket runtime may replace overshoot abort with
  `current_generation >= target_release_generation`, but only after the
  capture-only gate below passes and the runtime change is separately reviewed.
- With a fixed TMA implementation baseline, `wait_block_enable=0` is
  bit-identical to that baseline. A separately approved cross-subsystem
  correctness repair such as B0-D establishes a new baseline; disabling WaitBlock
  must not restore the known-wrong TMA FIFO.

## Batch B scope and current status

| ID | Work item | Status after `.e85`/`.e62` final |
|---|---|---|
| B0-A | Repair UBLKCP sidecar callback-role selection | **IMPLEMENTED / VALIDATED** at `0f7cbc2`; K10 is 8 sites/6,528 occurrences at 512 bytes, and kernel 11 is 1 site/384 occurrences at 32,768 bytes |
| B0-B | Kernel-aware descriptor identity and K5 tx-size restoration | **IMPLEMENTED / VALIDATED** at `0f7cbc2`; chain 23/23, ambiguity 0, K5 `0x96b0` = 24,576 bytes, 192 128-byte AGU requests, and 768 32-byte sector requests |
| B0-C | Size-only K5/K10 byte-conservation diagnostic with existing FIFO/release order | **COMPLETE**; byte contract PASS, K10 clean exit at 138,456 cycles, K5 runtime size PASS but all-live WaitBlock abort at cycle 68,176 |
| B0-D | Exact TMA-command-to-barrier completion ownership | **COMPLETE / VALIDATED** at `39abefb`; `.e85` validates ownership-clean out-of-order completion, and K10 `.e62` completes 384/384 CTAs with 132/132 clean SM-level TMABIND final records after exercising two simultaneous open contracts per warp |
| B0-E | K5 all-live cross-warp WaitBlock cycle (`0xb370`/`0x16c60`) | **OPEN / CAPTURE FEASIBILITY NEXT**. Old offline-order ticket designs and runtime releases remain rejected. Existing FA3 trace audit: K5 2,112 unique and 3,072 ambiguous TMA wait streams; K10 11,904 unique. Candidate v1 records a barrier-local opened-contract interval at each wait, only for barriers that pass the arrival-count-one and exact B0-D contract gates. Simulator behavior remains unchanged until regenerated K5/K10 traces resolve all eligible waits uniquely |
| B1 | One SM-level WaitBlock release phase per cycle | **REQUIRED / NOT IMPLEMENTED** |
| B2 | WaitBlock startup and provenance fail-fast checks | **MOSTLY IMPLEMENTED** |
| B3 | Assert-before-clear lifecycle checks | **IMPLEMENTED for WaitBlock and B0-D TMA ownership**; broader non-ownership cleanup audit remains partial |
| B4 | Zero-active no-op and active required-payload checks | **IMPLEMENTED** |
| B5 | Configurable watchdog/heartbeat and kernel-scoped counters | **REQUIRED / NOT IMPLEMENTED** |
| B6 | WaitBlock-only non-mutating hold query | **PARTIAL**: SR-M0A path implemented; legacy metrics still need audit |
| B7 | Fold/WaitBlock metric integrity and final reconciliation | **PARTIAL** |
| D0 | Generic ISA-wide effect framework | **DEFERRED**; exact sidecar allow/proof sets remain authoritative |
| D1 | Generic repeated-EXCH initialization epochs | **DEFERRED** for current FA3 one-EXCH-per-barrier contract |
| D2 | Full trace regeneration | **COMPLETE** at tracer commit `9e92fe9`; K5/K10 payload coverage validated |

No item may weaken the existing WaitBlock generation, cursor/replay, or
semantic-work invariants. Feature-off identity is evaluated within one TMA
ownership baseline; B0-D intentionally retires the legacy-FIFO baseline for both
WaitBlock-on and WaitBlock-off execution.

## Live B0-E analysis

B0-D is complete (see [WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md) sections 5-6
for the `.e84`/`.e85`/`.e62` evidence). B0-E work uses `39abefb` plus the `.e62`
result as its fixed exact-ownership baseline. What remains is a WaitBlock
liveness defect, not a TMA-ownership defect.

### Minimal SM124 causal cycle (`.e85`)

The terminal 13-warp pending set is:

```text
warp 0:
  FOLD_CAPTURED_POLL at 0xb370
  waits for barrier 0x16c60 generation 4
  observed state: current generation 3, arrive 1/3

warps 4-6:
  INJECT_MISSING_WAIT at 0x5a50
  representative dependency: warp 4 waits for 0x16c38 generation 5

warp 7:
  INJECT_MISSING_WAIT at 0x5a50
  waits for 0x16c30 generation 6

warps 8-15:
  INJECT_MISSING_WAIT at 0x74e0
  representative dependency: warp 12 waits for 0x16c50 generation 5
```

The minimum cycle needs only warps 0, 4, and 12, plus the already executed
arrival from warp 8:

```text
warp 0 waits for 0x16c60 generation 4
  -> 0x16c60 needs three active arrivals from warps 4, 8, and 12
  -> warp 8 has supplied the one observed arrival
  -> warp 4 and warp 12 have not reached their future 0x16c60 arrivals
  -> warp 4 is held on 0x16c38 generation 5
  -> warp 12 is held on 0x16c50 generation 5
  -> those later consumer generations require future producer work from warp 0
  -> warp 0 cannot issue that work until 0x16c60 reaches generation 4
```

This is not missing TMA completion credit. The simulator converted ready-only
hardware observations at `0x5a50` and `0x74e0` into blocking holds when its
modeled barriers were not ready. Those holds prevented the future ARRIVEs that
the captured poll at `0xb370` needed, while the poll hold prevented warp 0 from
issuing the work needed by the consumers.

The source-trace classification remains internally consistent:

```text
K5 ready-only records: 12
K5 ready-only visits: 24,564
all-live visits: 23,616
partial visits: 948

0x5a50: 7,680 ready-arm visits, all live
0x74e0: 3,168 ready-arm visits, all live
0xb370: 132 captured poll-loop visits
```

These counts prove the captured path and masks. They do not provide a
cross-warp hardware execution order for the semantic events after each gate.

### Proof gap: local safety versus global park safety

The current sidecar proves local safety:

- the fallback arm or folded loop uses the same barrier and wait-state
  expression;
- the folded body contains only wait observation, delay, and loop control;
- the selected cursor movement and local register/predicate use are valid.

It does not prove global park safety:

- whether the held warp has a future ARRIVE, TMA issue, or other semantic event
  needed by another held warp;
- whether parking several individually safe warps closes a CTA-wide dependency
  cycle;
- which held warp, if any, may advance to a later safe boundary without
  violating modeled wait semantics.

Therefore `local proof == PASS` is not sufficient authority to park an all-live
warp. B0-E must add a CTA-global progress argument or obtain stronger trace
evidence before changing hold placement.

### Candidate direction: barrier-local contract identity

The new direction separates a wait's **logical producer dependency** from the
cycle at which the simulator completes that dependency:

```text
trace / sidecar:
    wait occurrence W -> exact barrier release generation G

simulator timing model:
    computes the cycle at which generation G completes
```

This is neither unconditional trace replay nor hardware-cycle replay. If W
reaches the simulator before G, it waits. If it reaches after G, the dependency
is already complete and W continues. Simulator timing still determines the
wait duration, but timing drift cannot rebind W to a later generation with the
same one-bit parity.

#### Existing FA3 trace audit

The full trace at
`hw_run/traces/device-0/12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/`
`flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1`
was generated by tracer `9e92fe9`. It contains all CTA protobufs,
`sync_operand_resolver.json`, schema-v5 `spin_wait_sidecar.json`, and validated
B0-D TMA metadata.

The barrier-lifecycle audit over all K5/K10 CTA files produced:

| property | K5 | K10 |
|---|---:|---:|
| CTAs inspected | 132 | 384 |
| TMA barrier instances | 660 | 1,872 |
| active `EXCH` records per TMA barrier | exactly 1 | exactly 1 |
| `EXCH` semantic value | `0x1ffffe` for all | `0x1ffffe` for all |
| decoded expected arrival count | 1 for all | 1 for all |
| TMA contracts | 5,112 | 6,912 |
| completion-bearing LOADs | 5,112 | 13,824 |
| expected bytes == assigned bytes | yes | yes |
| initialization/address/byte errors | 0 | 0 |

K10 commonly has two TMA LOAD commands in one contract. That is still one
barrier generation: the two commands jointly satisfy one expected-byte
contract.

`contract` and `generation` are distinct concepts:

```text
contract:
    one ARRIVE_EXPECT_TX occurrence plus the completion-bearing TMA command(s)
    that satisfy its expected-byte value

generation:
    one barrier phase interval, ending when required arrivals and transaction
    bytes are both complete
```

They are one-to-one for the eligible FA3 TMA barriers because every such
barrier is initialized with expected arrival count one. The one active
`ARRIVE_EXPECT_TX` supplies that sole arrival, and its TMA command(s) supply the
transaction bytes. The phase can flip only after both are complete. K10's two
LOADs per contract therefore remain one generation, not two generations.

For a general barrier initialized with arrival count greater than one, several
`ARRIVE_EXPECT_TX` or plain `ARRIVE` occurrences may contribute to one
generation. In that case contract ordinal is not generation identity and this
v1 must reject it.

The existing trace records barrier address, wait parity/result path, and
per-warp order, but no cross-warp sequence or barrier-local contract count at a
wait. Under the strongest admissible trace-only assumption -- successive waits
by one warp target strictly increasing generations -- the TMA wait streams
resolve as follows:

```text
K5:
    unique streams       = 2,112  (consumer warps 4-7)
    ambiguous streams    = 3,072  (sparse-role warps 8-15)

K10:
    unique streams       = 11,904
    ambiguous streams    = 0
```

Representative K5 CTA 0 ambiguity:

```text
barrier 0x16c30 contracts/releases = 9
warp 8 wait parity sequence        = [0, 0]

first wait candidates  = {1, 3, 5, 7}
second wait candidates = {3, 5, 7, 9}
strictly increasing assignments = 10
```

This proves that the existing compact trace cannot provide exact event IDs for
K5. It also identifies the missing datum more narrowly than the rejected
CTA-global ordered trace.

The evidence status is:

| condition | status | evidence / limitation |
|---|---|---|
| one initialization per eligible TMA barrier | **PASS** | every K5/K10 TMA barrier instance has exactly one active `EXCH` |
| expected arrival count is one | **PASS** | all eligible `EXCH` values are `0x1ffffe`, decoded as one |
| contract byte ownership is exact | **PASS** | B0-D has zero missing, residual, or cross-barrier byte assignments |
| one contract corresponds to one generation | **PASS under the validated FA3 lifecycle** | arrival count one plus one active `ARRIVE_EXPECT_TX`; not valid for general multi-arrival barriers |
| actual TMA phase-flip cycle is recorded | **NO** | TMA completion is asynchronous and has no SASS callback point |
| wait ready/miss result is known | **PASS** | ready-arm path or final captured-poll exit; never inferred from parity |
| opened contract count at each wait is known | **NO -- Stage E0 target** | current protobuf has no cross-warp sequence or barrier-local snapshot |
| exact K5 wait-to-generation mapping from current trace | **FAIL** | 3,072 sparse-role streams remain ambiguous |

The 64-bit mbarrier object lives at a shared-memory address, but the ISA does
not expose a portable monotonic generation field. A raw shared-memory value or
the wait parity reveals at most current state; after two flips the parity repeats
and the lost generation count cannot be recovered.

#### Restricted applicability gate

Contract ordinal can stand in for generation only when every barrier instance
passes all of these checks:

1. exactly one active `EXCH` initialization in the supported epoch;
2. decoded expected arrival count is one;
3. no plain `ARRIVE` contributes to that TMA-backed barrier;
4. one active `ARRIVE_EXPECT_TX` opens the generation;
5. B0-D assigns every expected byte to the exact completion-bearing TMA
   command(s), with no residual or cross-barrier ownership;
6. the same-barrier lifecycle forbids opening the next contract before the
   previous generation has completed;
7. the wait result is proven from the captured branch path or final poll-loop
   exit, never inferred from parity.

This is a capability predicate, not an FA3 name/PC whitelist. A barrier that
fails any check receives no ticket and the capture stage fails closed.

This v1 does **not** cover:

- arrival count greater than one;
- multiple `ARRIVE_EXPECT_TX` arrivals in one generation;
- mixed plain-ARRIVE and TMA-arrival generations;
- count-only barriers;
- repeated `EXCH` epochs.

Those cases need arrival-group reconstruction or an explicit monotonic
generation source. They must not silently use contract count as generation.

#### Capture-only implementation plan

Add barrier-local trace state keyed by:

```text
(kernel launch, CTA xyz, barrier shared-memory offset, initialization epoch)
```

Record:

```text
EXCH:
    initialize/reset the epoch and opened-contract counter

ARRIVE_EXPECT_TX:
    allocate the next contract ordinal

TRYWAIT / PHASECHK:
    record opened-contract count at the existing pre-instruction callback
```

The schema retains before/after bounds so the resolver always consumes an
interval. V1 used dedicated callbacks on both sides of the instruction. V2
removes those callbacks and emits `[before,before]` from the existing compact
sync callback. Full-trace reanalysis showed all 81,561 eligible occurrences
remain unique under this degenerate interval, including the 137 v1 occurrences
whose bounds differed. The following branch path still proves READY versus MISS;
an occurrence is rejected if that result and the single count do not select one
generation.

The tracer must not claim that it observed the asynchronous TMA phase flip.
It observes contract-open instruction callbacks and wait instruction
boundaries only. The resolver uses the validated same-barrier lifecycle to
infer the adjacent completed-generation candidates. This distinction must
remain explicit in schema names, logs, and documentation.

For a supported TMA barrier with `N` contracts opened, same-barrier lifecycle
restricts the current release generation to adjacent candidates:

```text
{N - 1, N}
```

The captured ready/miss result and parity select between them. For a polling
episode, use the final observation that exits the loop. The output record is:

```text
barrier initialization epoch
contract ordinal interval observed at the wait
target release generation
proof inputs and reject reason
```

The trace record must not contain a hardware or instrumented cycle value used
by simulator timing. It carries dependency identity only.

#### Why this is not the rejected ordered-ticket design

The rejected design imposed a CTA-global order over all synchronization events
and attempted to infer generations from overlapping begin/end intervals. It
changed K5 control-path counts and still left candidate sets such as `{1, 3}`.

The new proposal:

- orders only events for the same barrier instance;
- captures the contract count at the wait instead of reconstructing it from all
  cross-warp events;
- uses the already-validated B0-D contract lifecycle;
- emits nothing when the barrier is outside the restricted applicability gate;
- changes no simulator behavior during the feasibility stage.

It still adds state work inside an existing callback and can perturb execution.
Therefore the new trace is rejected if WaitBlock action structure or ready-only
entry/mask counts, TMA contracts, semantic instruction counts, or CTA completion
differ from the `9e92fe9` source trace. Semantic-free FOLD retry and episode
counts are diagnostic because runtime folding does not consume them as timing
inputs.

### Why generation removal is not the fix

**The primary work is (1) global park safety and (2) producer/consumer timing
realignment, not deletion of the WaitBlock generation counter.**

An earlier draft claimed that retiring the WaitBlock generation audit would make
the deadlock "abort go away." That is misleading. Deleting the audit removes the
overshoot ABORT MESSAGE, not the deadlock. Generation is a DETECTOR of
divergence, not its root cause:

```text
K10 .e51:86459  SM24/warp0 entry 0xa9c0 barrier 0x31020 gen=0 for 32,001 cycles
```

This hang sits at `gen=0`; it is a watchdog stall, not a generation overshoot.
Its cause (`debug-k10-wait-fold-deadlock.md` L59-L75) is a whole-warp park of a
partial-mask (`0x1`) entry that blocks lane 1-31 `WARPSYNC.ALL 0xb2c0`, which
prevents the producer ARRIVE that would flip `0x31020` gen0->gen1. Generation
arithmetic is nowhere in that cycle.

Release readiness never reads `barrier.gen`:

```text
is_sync_wait_satisfied == (barrier.phase != wait_parity)   (sm.cc L4026-L4037)
```

If every `barrier.gen` were deleted, a wait with `phase == wait_parity` still
holds. The `.e85` 13-warp cycle is built from exactly these phase-holds blocking
one another, so it re-forms identically without the counter. The only change is
that the current overshoot abort would surface as a watchdog hang.

`barrier.gen` roles for reference:

```text
Role 1 (readiness):   phase only, already hardware-like. Not a fix target.
Role 2a (WaitBlock generation audit):  expected_gen_for_wait + cur_gen compare
        ladder + overshoot/undershoot abort (sm.cc L3419-L3494, L4045-L4074).
        This is a DETECTOR only. Deleting it hides the symptom.
Role 2b (TMA completion identity):     assignment_generation /
        target.barrier_generation exact routing (sm.cc L4962, L5098-L5133,
        L5252-L5263). MUST be preserved; B0-D depends on it.
```

Overshoot is also not a data-correctness hazard in this simulator: the barrier
model stores only counters and ownership metadata, never shared-memory tensor
data, so a `current_gen > target_gen` release cannot corrupt values here. The
overshoot abort is a model-divergence auditor.

Actual deadlock root causes, neither of which is the generation counter:

1. **Whole-warp park that stalls lane-group / producer progress** (the direct
   trigger of K10 `.e51` and K5 `.e78`). A hold must not stop another warp's or
   lane group's required barrier progress.
2. **A missed captured-ready phase window** that leaves the consumer waiting on a
   parity that already passed (the `.e85` terminal cycle). In `.e85` the parity-0
   `0x16c50` window opens at ~30,135 and closes at ~34,575, but warp 12's
   `0x74e0` arrives at ~36,624 — a ~2K-cycle skew where warp 0 reissues the next
   `0x16c50` contract before the consumer reaches `0x74e0`.

Direction:

- The immediate B0-E axis is the capture-only barrier-local contract-identity
  experiment above. It attempts to prevent simulator timing drift from changing
  a wait's logical target.
- Producer/consumer timing analysis remains necessary for cycle accuracy, but
  timing calibration is not accepted as a liveness mechanism.
- Global park safety remains a fail-fast guard for unsupported or unticketed
  holds; it must not release a warp.
- Retiring the Role 2a audit without exact identity only changes an abort into a
  hang and remains forbidden.
- Role 2b (TMA completion identity) stays intact regardless.
- The ordered-ticket design and the post-`951b058` experiment are recorded as
  rejected/closed alternatives in [WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md).
  Do not reintroduce CTA-global ordering, forced phase flips, guessed
  generations, or unproven whole-warp releases.

## Required B0-E implementation gates

### Stage E0: capture feasibility only

1. Treat the full `9e92fe9` FA3 trace and K10 `.e62` as the fixed source and
   simulator baselines.
2. Extend the protobuf/wire record with initialization epoch, contract ordinal,
   and a wait-side contract-count interval. V2 emits a degenerate
   `[before,before]` interval. Do not record or consume hardware cycles.
3. Maintain state per CTA and barrier, not in a CTA-global sequence. Reset only
   on the exact `EXCH` event and reject storage overflow, duplicate initialization,
   unsupported active-arrival shape, or lifecycle inconsistency.
4. Extend post-processing with the restricted applicability predicate and a
   closed reject-reason taxonomy.
5. Generate one new full FA3 trace. Compare it with `9e92fe9` and require:

```text
same K5/K10 CTA counts
same semantic instruction and TMA command counts
same B0-D contract counts and byte totals
same WaitBlock action structure and ready-only entry/mask counts
report FOLD retry/episode counts without requiring equality
zero missing/duplicate contract snapshots
zero unsupported snapshots on eligible barriers
zero inconsistent generation assignments
zero ambiguous generation assignments
K5's previously ambiguous 3,072 streams become unique
K10's 11,904 unique streams retain the same ordinal mapping
```

6. If any condition fails, stop. Do not tune the resolver and do not modify the
   simulator.

Implementation order inside E0:

```text
E0.1  Define protobuf/wire fields and closed reject reasons.
E0.2  Design bounded per-launch (CTA, barrier) state and lifecycle cleanup.
E0.3  Capture EXCH epoch, ARRIVE_EXPECT_TX ordinal, and wait-before snapshots.
E0.4  Unit-test serialization, overflow, duplicate-init, unsupported-shape,
      missing-snapshot, interval ambiguity, and many-waits-to-one-event cases.
E0.5  Run post-processing on synthetic fixtures before generating FA3.
E0.6  Preserve/rename the 9e92fe9 baseline directory.
E0.7  Generate exactly one new full FA3 trace and run the acceptance comparison.
```

#### Stage E0 local implementation checkpoint (2026-09-01)

E0.1-E0.5 are implemented locally. E0.6-E0.7 remain user-owned hardware
validation work.

- `--mbarrier-contract-capture` enables the tracer environment option
  `MBARRIER_CONTRACT_CAPTURE=1`. The default is zero.
- Enabled runs use the separate `-mbarrier-contract-v2` output-directory suffix,
  so `-f` cannot delete the unsuffixed `9e92fe9` baseline.
- The v2 capture path adds no instruction callbacks and no channel pushes. The
  existing compact barrier-address callback updates bounded `(CTA, barrier)`
  state and appends a 72-byte contract payload to that callback's existing wire
  record. There is no second `ChannelDev`, device event buffer, record-reservation
  atomic, warp scratch, or `IPOINT_AFTER` callback.
- The existing compact callback executes its original warp collectives before
  selecting the publisher lane. The semantic register is shuffled from the first
  predicated-active lane before the publisher writes the inline payload.
- Only existing barrier-address records for `SYNCS.EXCH*`,
  `SYNCS.ARRIVE.TRANS64*`, and `SYNCS.PHASECHK*` can carry the inline extension.
  Blocking `BAR.SYNC`, `WARPSYNC`, and `WARPGROUP*` records are unchanged.
- Device state is partitioned by launch and linear CTA. Each CTA owns only a
  bounded open-addressed barrier table. Table exhaustion, invalid launch
  identity, malformed inline payloads, and incomplete protobuf instructions
  abort.
- The host attaches the inline payload while processing the same existing sync
  record and writes `mbarrier_contract_runtime_info`; no post-kernel occurrence
  join and no hardware cycle are used.
- Strict post-processing writes
  `extra_info/mbarrier_contract_ordinal_feasibility.json`, applies the
  arrival-count-one/TMA-ownership gate, enumerates every generation allowed by
  each contract-count interval, and rejects empty or multi-generation results.
  Reusing a barrier for contract `j+1` additionally requires a captured READY
  wait at the exact opened count `j`; absence of that witness rejects the
  barrier as `CONTRACT_SERIAL_LIFECYCLE_UNPROVEN`.
- Strict enabled runs also compare the new and unsuffixed baseline traces over
  semantic `(kernel, function, PC, opcode, active_mask, predicate_mask)`
  histograms, WaitBlock action structure/ready-only counts, and B0-D
  contract/load/byte totals. FOLD dynamics are emitted separately.

Local validation:

```text
33 Python unit tests PASS
Python syntax compilation PASS
capture ABI size assertions compile with host C++17 PASS
git diff --check PASS
existing full-trace capacity audit:
  K5  max barriers/CTA = 22
  K10 max barriers/CTA = 9
  default table bound  = 64 barriers/CTA
```

The local macOS environment has neither `nvcc` nor `protoc`; generated protobuf
compilation, NVBit linking, callback execution, and the full FA3 acceptance
comparison remain pending on the CUDA server. This checkpoint does not authorize
Stage E1 or any simulator behavior change.

#### Stage E0 v1 full-trace result (2026-09-01)

The first rebuilt full trace completed without a kernel hang, buffer overflow,
callback mismatch, or protobuf-join failure:

```text
K5  220,017 capture records attached
K10 496,301 capture records attached
K11   3,840 capture records attached
missing snapshots = 0
capture mismatches = 0
```

Two post-processing defects in `527af10` were found by inspecting the copied
trace and fixed locally:

1. Proto3 omits zero-valued scalar fields. The custom parser incorrectly decoded
   omitted epoch/count fields as `None` instead of zero.
2. Fold lifecycle validation inspected only `poll_probe_pc`. Three K5 CTA
   instances completed at the earlier TRYWAIT in `poll_wait_pcs`, so the final
   PHASECHK was predicated off. The lifecycle witness now uses every statically
   proven wait site in the fold loop.

Reanalysis of the existing trace, without retracing:

```text
eligible TMA barriers           = 2,916
eligible action occurrences     = 81,561
unique generation assignments  = 81,561
ambiguous assignments           = 0
inconsistent assignments        = 0
unique streams                  = 21,045
invalid streams                 = 0
semantic instructions K5/K10   = 5,755,752 baseline / 5,755,752 capture
semantic mask-histogram sites  = 648 baseline / 648 capture
semantic histogram SHA-256     = b49bed1e... / b49bed1e...
B0-D contracts/loads/bytes      = unchanged
```

Therefore the barrier-local ordinal capture resolves every eligible occurrence
uniquely. The trace still fails the independent perturbation gate: the added
callbacks reduce repeated polling/backoff iterations while preserving semantic
work. This remains an E0 stop condition under lines 645-661; Stage E1 and
simulator behavior changes are not authorized.

#### Stage E0 v2 lightweight capture (2026-09-01)

The v2 implementation replaces both dedicated callbacks and the launch event
buffer with an inline extension of the existing compact sync-address record:

```text
additional NVBit callbacks per sync occurrence = 0
additional ChannelDev pushes per occurrence    = 0
IPOINT_AFTER callbacks                          = 0
device event-buffer bytes                       = 0
per-event record-reservation atomics            = 0
managed state                                   = CTA-local barrier table only
```

Raw retry counts inside a proven semantic-free FOLD episode are not simulator
timing inputs and are not required to be identical. The perturbation comparison
must preserve semantic PC counts, B0-D ownership totals, INJECT entry
occurrences/masks, and static FOLD proof/action structure. Dynamic FOLD retry and
episode counts remain reported diagnostics but are not hard acceptance inputs.

Hardware build and trace execution remain user-owned. The expected trace command
after E0 implementation is:

```bash
nohup ./util/tracer_nvbit/run_hw_trace.py \
  --mbarrier-contract-capture \
  -f \
  -B flashattn:flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24 \
  -D 0 \
  > /tmp/flashattn.log 2>&1 &
```

This command is valid only after the old run directory has been preserved under
a different name; `-f` intentionally deletes the selected run directory.

Expected Stage E0 files:

```text
util/traces_enhanced/dynamic_trace/instruction.proto
util/tracer_nvbit/tracer_tool/common.h
util/tracer_nvbit/tracer_tool/inject_funcs.cu
util/tracer_nvbit/tracer_tool/tracer_tool.cu
util/tracer_nvbit/build_spin_wait_sidecar.py
util/tracer_nvbit/tests/test_build_spin_wait_sidecar.py
util/tracer_nvbit/run_hw_trace.py
```

The exact device-state allocation must be designed before editing: it needs a
bounded per-launch `(CTA, barrier)` index, explicit reset/cleanup, and
overflow/unknown-key rejection. It must not use application names, kernel IDs,
PC lists, fixed barrier addresses, or expected occurrence counts.

### Stage E1: separately reviewed simulator use

Only after Stage E0 passes:

1. Add exact `target_release_generation` to eligible
   `INJECT_MISSING_WAIT` records.
2. At runtime, bind the wait to that target instead of deriving a target from
   the simulator's current phase.
3. Use:

```text
current_generation < target  -> hold
current_generation >= target -> dependency already complete; continue
```

4. Count `current_generation > target` as a late-arrival timing divergence. It
   is not a reason to bind the wait to a later same-parity generation.
5. Leave unsupported TMA barriers, count-only barriers,
   `FOLD_CAPTURED_POLL`, partial replay ownership, B0-D TMA completion identity,
   and cursor movement unchanged.
6. Validate K5 and K10 separately from B1. Require K5 clean exit without forced
   releases, K10 384/384 completion, all WaitBlock/TMABIND summaries PASS, and
   unchanged semantic TMA/BAR/memory work. Report total-cycle and producer/
   consumer-tail deltas rather than accepting completion alone.

Expected Stage E1 files are the existing sidecar loader/runtime state in
`shader.h` and `sm.cc`, plus parser/schema integration where the current v5
records are loaded. File scope must be finalized from the code at implementation
time; no B1 scheduler changes belong in E1.

B0-E acceptance must eliminate the 13-warp all-live cycle without dropping a
dependency, forcing a phase, guessing a generation, advancing a cursor outside
an approved release, or changing semantic TMA/BAR/memory work.

## Temporary K5 hardcoded deadlock breaker (measurement only)

**This is not a fix and must not ship.** After the generic runtime
deadlock-release attempt was reverted (see below), a deliberately
application-specific probe was added purely to let K5 run to completion so its
cycle count is observable.

- **Flag:** `-wait_block_k5_hardcoded_deadlock_release` (`OPT_BOOL`, default `0`).
  Default-off keeps every run byte-identical to the `326ef0b` baseline.
- **Behavior:** in `SM::service_wait_block_release`, when a hold is stuck
  (`current_gen < expected_gen`) with mode `MISSING_WAIT_HOLD` and its
  `action_pc` is one of the K5 producer/consumer wait PCs
  `{0x13b0, 0x5a50, 0x5bc0, 0x74e0}`, force-release it (clear the pending wait,
  same as the normal missing-wait release: no cursor move, no phase force, no
  generation guess) instead of holding it to the watchdog. Emits
  `[WAITBLOCK][k5-hardcoded-deadlock-release]` per event.
- **PC set provenance:** the four PCs are exactly the `INJECT_MISSING_WAIT`
  `action_pc` values observed parked in the pristine K5 deadlock (`.e85`).
- **Scope:** K5-specific, PC-hardcoded, generation-agnostic. It exists only to
  measure `gpu_tot_sim_cycle` once K5 completes; it does not generalize and no
  companion document treats it as authority.
- **Files:** flag in `shader.h`, registration in `gpu-sim.cc`, hook in
  `SM::service_wait_block_release` (`sm.cc`), enabled in the H100 config.
  **These four code files were reverted after the run below** (restored to the
  pre-probe `fa9e95e` state); this section is retained as findings/handover only,
  so re-applying the probe means re-adding the hook, not just flipping a flag.

### First hardcoded-probe run (`.e89`/`.o89` K5, `.e65`/`.o65` K10)

Committed at `9e16c6f`. Result: **K5 still does not complete** — the probe cleared
the original deadlock but exposed deeper hang layers.

- **K10 (`.e65`): clean, no regression.** 384/384 `[WAITBLOCK][summary]
  status=PASS`, zero aborts, and the hardcoded release fired **0 times** (the four
  PCs are K5-only). Confirms the probe is inert for K10.
- **K5 (`.e89`): progressed much further, then aborted again.** The four-PC
  release fired **7,636 times** and pushed the original `0x16c50/38/30/58`
  consumer cycle from generations 4-6 out to generations 8-11; many CTAs reached
  `[CTAFIN]` and 52 SMs emitted `status=PASS`. Final abort at **cyc 89,126**
  (vs. the pristine `.e85` watchdog at 68,240), reason `hold-watchdog`,
  `strategy=FOLD_CAPTURED_POLL`, warp 0 (producer), `action_pc=0xb0b0`,
  `barrier=0x16c20`, gen 2->3, held 32,001.

**Key finding — the K5 deadlock is multi-layer, and the parked-PC set grows as
the run advances.** Breaking the four consumer PCs let the pipeline reach later
generations, where *new* wait sites enter the terminal cycle. Full census of
what was still parked in the last window (cyc 85,000-89,126) of `.e89`:

| strategy | action_pc | barrier | state | in current whitelist? |
|---|---|---|---|---|
| INJECT_MISSING_WAIT | `0x35e0` | 0x16c30 / 0x16c38 | gen 8-11 stuck | no |
| INJECT_MISSING_WAIT | `0x3750` | 0x16c50 / 0x16c58 | gen 9-10 stuck | no |
| FOLD_CAPTURED_POLL | `0xb0b0` | 0x16c20 | gen 2->3, max_held 28,672 (the abort) | no |
| DEFER_PARTIAL_MISSING_WAIT | `0xab40` | 0x16c40 / 0x16c48 | gen 8-9, max_held 24,576 | no |

Method note for the census: `[WAITBLOCK][heartbeat]` repeats every 4,096 cycles,
so `max_held` distinguishes a true hang from normal park. FOLD sites `0xb1f0`,
`0xb2b0`, `0xb370` show many heartbeats but `max_held=4096` (many warps parking
once each, then releasing) — **not** hangs. Only `0xb0b0` (FOLD) and `0xab40`
(DEFER_PARTIAL) reach ~24-28k held cycles, i.e. real hangs, alongside the
still-stuck INJECT sites `0x35e0`/`0x3750`.

### Historical handover — superseded 2026-08-31

The old A-vs-B choice between extending the PC whitelist and adding a generic
held-cycle threshold is closed. Both are measurement-only release mechanisms,
both contaminate cycle timing, and neither supplies the missing dependency
identity. The hardcoded probe has been reverted.

**Current next step:** review and approve Stage E0 above. If approved, implement
only barrier-local contract-ordinal capture and the offline uniqueness proof,
then regenerate FA3 once. Do not re-add either release probe and do not modify
simulator wait behavior during E0.

**Evidence pointers:** K5 `.e89`/`.o89` and K10 `.e65`/`.o65` under
`sim_run_12.8/.../H100_80GB-OnlyKernel5` and `...-OnlyKernel10`. Pristine
(pre-probe) deadlock is `.e85`.

### Rejected/closed generic attempts (do not re-propose as-is)

1. **Offline pairing (3 variants).** Ordered generation ticket, ordered
   candidate-set, and runtime UR recompute all fail for the same reason: the
   compact trace preserves per-warp program order but drops cross-warp execution
   order, and the SASS/HW state is a 1-bit parity plus per-warp *modulo* counters
   (UR4/UR5/UR8 wrap), so a wait's absolute release generation is unrecoverable.
2. **Generic runtime single-consumer deadlock-release** (implemented at
   `65bbf44`/`2608f40`, reverted at `6c335bf`/`fa9e95e`). On a proven CTA-wide
   deadlock it released one `INJECT_MISSING_WAIT` consumer per detection, guarded
   by a trace-cursor progress check and a sustained-stuck (`>=128` cycle,
   `>=2` live warp) gate. Result: K5 advanced generation 4 → 9 but still tripped
   the hold watchdog (~90k cycles), and K10 regressed (release fired on
   transients whose episode span reached ~1,659 cycles, beyond the 128 gate).
   The single-release cadence accumulates latency each generation, and no fixed
   cycle threshold separates a real deadlock from a long K10 transient.

### B1-specific acceptance (separate commit)

The deterministic SM-level release-phase design is in
[WAIT_BLOCK_TMA_OWNERSHIP.md](WAIT_BLOCK_TMA_OWNERSHIP.md) section 7. Its
acceptance is:

```text
feature-off run == B0-D baseline
normal traversal release fingerprint == reversed traversal release fingerprint
normal traversal final cycle == reversed traversal final cycle
every hardware warp serviced exactly once per SM cycle
cycle-start completion may release in the current cycle
issue-phase arrival releases no warp before the next cycle
trace cursor, semantic work, and conservation summaries unchanged
```

Record the total-cycle delta from the B0-D baseline, but do not impose an
unsupported fixed percentage or direction. A performance threshold may be added
only after repeated unchanged-binary runs establish a measurement envelope.

Do not combine B0 and B1 in one commit or one causal validation step. Only after
both B0-E and B1 pass may the `.e83` debugging session be closed and the older
Batch B document be treated as historical rather than authoritative.

## Document ownership

| Document | Responsibility |
|---|---|
| `WAIT_BLOCK.md` | Authoritative index, shared invariants, Batch B scope, live B0-E analysis and gates |
| `WAIT_BLOCK_POSTPROC.md` | Offline sidecar classification design |
| `WAIT_BLOCK_SIMULATOR.md` | Simulator runtime paths and partial-lane v2 replay design |
| `WAIT_BLOCK_TMA_OWNERSHIP.md` | Exact TMA completion ownership (B0-A … B0-D), B1 release phase, metrics, deferrals |
| `WAIT_BLOCK_HISTORY.md` | Chronological server runs; rejected ticket design; reverted ordered-candidate experiment |
| `TMA_BASE_ADDR.md` | Historical derivation plus the corrected descriptor-identity evidence |
| `TMA_exact_base_mapping_integration.md` | Detailed sidecar/schema and simulator metadata integration |
| `TMA_ISA.md` | Stable per-opcode operand and byte-source semantics |
| `TMA_ARCH.md` | Stable TMA command, mover, and completion architecture |
| `TMA_LATENCY_INJECTION_H100.md` | Historical performance evidence; K5 timing is provisional |

Supporting documents must link to the relevant companion design rather than
duplicate it. If a supporting document conflicts with a companion design, the
companion design wins.
