# WaitBlock: Design Index and Current Status

**Status (2026-09-02):** B0-A through B0-E are complete. B0-D exact TMA
completion ownership remains validated at `39abefb`. B0-E exact-generation
capture and consumption is implemented at `087b53f` and validated by K5 `.e90`
and K10 `.e66`. K5 completes 132/132 CTAs at 85,172 cycles without a forced
release; K10 completes 384/384 CTAs at 137,348 cycles. Every emitted WaitBlock,
TMABIND, and named-barrier final summary passes, and all pending holds, TMA
contracts, bindings, and byte credits drain to zero.

The ABI-correct v2 trace assigns an exact generation to 90,868/90,868 eligible
WaitBlock waits: 75,840 INJECT entries and 15,028 FOLD waits. The runtime
validates the intended dependency identity and eliminates the K5 all-live
13-warp cycle. It does not establish cycle accuracy by itself. K5 reports 9,542
late-arrival divergences where `current_generation > target_generation` when an
INJECT action is reached; K10 reports zero. The K5 events are concentrated at
`0x5a50` and `0x5bc0`, so producer/consumer timing analysis remains open even
though dependency correctness and completion now pass. That timing analysis is
a separate cycle-accuracy optimization, not an Opt 12 completion gate. B1
release-phase accounting and cycle-order checks pass K5/K10. The rejected
reverse-subcore validation experiment is recorded below.

The final Opt 12 functional regression at `7ac4d23` passes: K5
`.e93`/`.o93` completes 132/132 CTAs at 84,906 cycles and K10
`.e69`/`.o69` completes 384/384 CTAs at 140,003 cycles. Both runs preserve
their prior CTA-level TRACECHECK records exactly and pass WaitBlock, TMABIND,
named-BAR, SR-M0A, release-service, and warp-instruction conservation with zero
active final state or aborts.

**Opt 12 closure (2026-09-02): COMPLETE.** B2 and B3 are accepted through
implementation review and local source-contract validation; their server-only
negative runtime matrices are waived/deferred. B5 is likewise accepted through
implementation and local source-contract validation, with its server-only
back-to-back launch test deferred. These deferred tests are not Opt 12 closure
gates. K5 cycle-accuracy analysis remains a separate optimization.

## Separate cycle-accuracy follow-up (outside Opt 12)

### K5 first late-arrival root cause

**Problem.** K5 completes, but 9,542 exact INJECT actions execute with
`current_generation > target_generation`. K10 has zero. A late observation does
not violate dependency safety because the target is complete, but it proves
that simulator producer/consumer timing has advanced past the hardware-mapped
generation before the consumer reaches the action PC.

**State key.**

```text
(trace_kernel_id, CTA x/y/z, CTA-local warp_id,
 unique_function_id, action_pc, source_warp_instruction_index,
 barrier_addr, target_generation)
```

**Required analysis.**

1. Select the first late-arrival event in each affected CTA before later
   generations inherit the timing skew.
2. Record the simulator cycles for target generation completion, next-contract
   open, next-generation completion, and consumer action-PC issue.
3. Correlate those transitions with the last producer and consumer semantic
   anchors before the divergence. Split the delay into producer progress,
   consumer scheduling, FOLD hold/release, partial replay, and memory/TMA
   intervals.
4. Aggregate by action PC and generation distance. Current evidence is:
   `0x5a50=4,301`, `0x5bc0=4,413`, all other parseable INJECT PCs=827;
   generation
   distance `+1=6,846`, `+2=2,667`, `+3=28`, plus one stderr-interleaved record
   accounted for by the authoritative per-SM totals.
5. Do not infer a hardware completion cycle from the ticket. The trace proves
   generation identity, not asynchronous TMA completion time. If simulator
   evidence cannot distinguish an early producer from a late consumer, obtain
   an additional hardware timing observation instead of choosing one.

**Instrumentation checkpoint.** The simulator now has the default-off
`-wait_block_late_arrival_instrument_enable` option. When enabled, it emits one
`[WAITBLOCK][late-timing-first]` record for the first exact INJECT late arrival
in each CTA. The record preserves the full trace action identity and reports the
target/next generation's contract-open, first/last TMA-issue, and completion
cycles together with the preceding producer and consumer semantic anchors. The
tested H100 configuration enabled this probe for K5 `.e91` and K10 `.e67`.
This is observe-only and does not infer hardware completion cycles.

**First-late result (`.e91`/`.o91`, 2026-09-01).** K5 emitted exactly 132
records, one for every CTA. The first divergence is at `0x5a50` for 126 CTAs and
`0x5bc0` for 6 CTAs; 131 are one generation late and one CTA is two generations
late. Every target/next contract-open, TMA-issue, completion, producer-anchor,
and consumer-anchor field is present.

For the dominant `0x5a50` population, median intervals are:

```text
target completion -> next contract open       10,926 cycles
next contract open -> next completion            357 cycles
next completion -> consumer action PC           1,797 cycles
last producer semantic anchor -> action PC         919 cycles
last consumer semantic anchor -> action PC         107 cycles
```

The first divergence therefore already has a fully completed extra generation;
it is not a late release-service observation or a long TMA-completion interval.
The consumer is still issuing semantic work near the action PC, while producer
warp 0 has opened and completed the next contract. This localizes the defect to
relative producer/consumer progress before the final consumer anchor, not to
the final wait issue itself. The trace still does not determine whether hardware
would classify the simulator difference as an early producer, a late consumer,
or both; that requires a cross-warp hardware timing observation.

**Acceptance.** A proposed timing correction must preserve the exact ticket
mapping, semantic TMA/BAR/memory histograms, TMA command/request/byte
conservation, and K10's zero late-arrival result. It must reduce K5's cycle
error relative to the 67,696-cycle HW target and improve the
Producer-minus-Consumer tail without a forced release, phase force, generation
rewrite, or PC whitelist. A lower divergence count is diagnostic evidence, not
a standalone acceptance condition.

## Opt 12 completion work

### B1 deterministic SM-cycle release phase

**Problem addressed.** `service_wait_block_release()` previously ran from each
`Subcore::modify_warp_state()`. Subcore traversal order could therefore
determine whether a barrier transition was observed in the current or next
simulator cycle.

**Required implementation.**

```text
cycle start
  -> advance shared pipelines and TMA completions
  -> snapshot release-visible barrier generations
  -> service every hardware warp exactly once
  -> run all subcore issue phases
  -> expose issue-phase arrivals to releases on the next cycle
```

The release pass may clear an approved pending WaitBlock state and perform the
already-defined primary FOLD cursor transition. It must not execute an
instruction, mutate named-BAR state, or make an issue-phase arrival visible to a
later-iterated subcore in the same cycle.

**Acceptance.** Require exactly one release-service visit per hardware warp per
active SM cycle, no release before its target generation flip, no issue-phase
arrival visible to release service in the same cycle, release fingerprints equal
to released-hold totals, unchanged semantic/TMA work, and zero active state at
kernel completion.

**Implementation checkpoint.** The release service now runs once at SM scope
after shared LD/ST and TMA pipeline advancement and before any subcore issue.
The release service and scheduler-side hold queries share an immutable
cycle-start `(generation, phase)` snapshot, so an issue-phase ARRIVE cannot
release a later subcore in the same cycle. The runtime checks one service visit
per hardware warp and emits canonical per-release fingerprints.

**Normal-traversal result (`.e91`/`.o91` K5, `.e67`/`.o67` K10).**

```text
K5:  132/132 CTAs, 84,358 cycles, 132/132 WaitBlock PASS,
     132/132 TMABIND PASS, 10,064 immediate late arrivals
K10: 384/384 CTAs, 137,756 cycles, 384/384 WaitBlock PASS,
     132/132 TMABIND PASS, zero late arrivals
```

Both kernels have zero watchdog/generation aborts, zero active final state, and
exact `release_service_visits == release_service_expected_visits`. Canonical
release-fingerprint counts equal the released-hold totals: K5 6,466 and K10
9,104. One K10 fingerprint was split by a concurrent MEMBAR stderr line; the
logger now formats the complete record before one `fprintf` call.

Relative to `.o90`/`.o66`, K5 is 814 cycles faster and K10 is 408 cycles slower.
K5's consumer-minus-producer drain tail improves in mean (2,075.7 -> 1,966.5),
95th percentile (4,424 -> 3,511), and maximum (5,399 -> 5,257), although its
median regresses (1,467 -> 1,665). These cycle deltas are diagnostic, not clean
B1 causal measurements: the current v2 trace and the earlier run trace have
different dynamic FOLD retry populations. The strict trace comparison still
passes semantic fingerprint, WaitBlock structure/counts, and TMA contracts.

**Direct release-order audit.** K5 normal services 648,760,320/648,760,320
expected warp visits; K10 normal services
1,096,073,024/1,096,073,024. K5 has 6,466 fingerprints for exactly 6,466
released holds, and K10 has 9,104 for exactly 9,104. For all 6,466 K5 releases,
the matching `(SM, barrier, target_generation)` flip exists and no release
precedes it. In 3,614 same-cycle cases the phase-flip log precedes the release;
the remaining 2,852 releases occur 1-6 cycles later. No duplicate/incomplete
service, invalid snapshot, or generation-ready-state guard fires.

**Rejected validation perturbation (`.e92`/`.o92`).** Reversing the complete
subcore issue traversal is not an isolated release-order test. It changes shared
pipeline arbitration, L2/TMA request order, and subsequent barrier timing. K5
still completes cleanly with identical TRACECHECK work and TMA bytes, but moves
from 84,358 to 85,900 cycles, changes L2 hits/misses, and produces 6,422 rather
than 6,466 releases. Therefore normal/reversed final-cycle or fingerprint
equality is not a valid B1 criterion. The validation option was removed; normal
subcore issue order remains canonical.

### B2 startup and provenance closure

Fail-closed loading checks are implemented for every enabled runtime path
before kernel execution.

**Required checks.**

- sidecar and ticket schema/version, `validation_complete`, and non-empty action
  maps;
- unique action keys and exact CTA-grid coverage;
- presence of enhanced trace, sync resolver, TMA metadata, eligible-barrier
  manifest, and exact ticket artifact;
- complete static wait/effect/mask/join/cursor proofs;
- unambiguous descriptor and UBLKCP operand provenance;
- active action instructions always carry the required runtime payload;
- zero-effective instructions remain no-ops;
- fold and observe modes cannot be enabled together.

**Acceptance.** Add or confirm one negative test per rejection class. Each
malformed input must abort before kernel execution with the violated identity
field in the diagnostic. A valid K5/K10 sidecar must still load without
weakening the current exact/fallback barrier-manifest checks.

**Implementation checkpoint.** Startup now rejects an empty runtime-action
set, duplicate `(trace_kernel_id, unique_function_id)` records, conflicting
runtime action keys with their kernel/function/PC identity, simultaneous
WaitBlock/FOLD-observe modes, empty sync metadata, and incomplete exact-mode
TMA metadata. Existing exact-target/fallback summary checks remain
authoritative. Local source-contract tests pass; executing the malformed-input
matrix against a simulator binary remains a server validation item.

**Validation scope decision (2026-09-02).** The runtime negative matrix is
waived for Opt 12 closure and remains deferred evidence, not an executed test.
Local Python/source-contract tests can prove that the guards are present, but
cannot prove that the integrated C++ loader aborts before cycle 0. That proof
requires launching the simulator binary with malformed artifacts on the
server; it would be a short startup-abort test rather than a full K5/K10 run.

### B3 assert-before-clear lifecycle audit

WaitBlock and B0-D TMA ownership assert their primary conservation rules across
every cleanup path that can erase state: CTA finish, warp-slot reuse, kernel
finish, and simulator teardown.

Before clearing state, assert:

```text
no active exact/count-only holds
no partial arm, capsule, replay queue, or replay progress
no open TMA contract or in-flight binding
no outstanding barrier/warp completion credit
no leaked named-BAR participant or deferred-sync credit
```

Acceptance requires clean K5/K10 teardown and a targeted negative test showing
that each residual-state class aborts rather than being silently discarded.

**Implementation checkpoint.** CTA finish, warp-slot reuse, kernel switch, and
simulator teardown now all reject active WaitBlock holds, partial arms/replay
state/capsules, open TMA ownership, and unbalanced hold/release counters before
state can be cleared. Terminal conservation now includes auxiliary FOLD holds.
Named-BAR deallocation already asserts participant and deferred-credit state.
Local source-contract tests pass; runtime residual-state fault injection remains
a server validation item.

**Validation scope decision (2026-09-02).** Runtime residual-state fault
injection is waived for Opt 12 closure and remains deferred evidence. There is
no isolated local SM/CTA lifecycle harness in this repository; proving the
negative path requires a simulator build plus a test hook or deliberately
injected residual state on the server. The clean K5/K10 regressions validate
the non-faulted teardown path only.

### B5 configurable watchdogs and launch-scoped diagnostics

Replace hardcoded timing constants with:

```text
-wait_block_hold_watchdog_cycles 32000
-wait_block_heartbeat_cycles      4096
```

Use `{active, start_cycle}` instead of cycle zero as the inactive sentinel.
Reset WaitBlock totals, pace records, semantic-work counters, TMA binding
counters, and log budgets at each runtime kernel launch. A provably
unfulfillable TMA contract must abort immediately rather than waiting for the
watchdog.

Acceptance requires startup printing of effective values and a back-to-back
multi-kernel test proving that the second launch inherits no pending state,
counter value, start cycle, or exhausted debug budget from the first.

**Implementation checkpoint.** The hold watchdog, heartbeat period, and
per-launch debug budget are now validated `uint64` config options and are
printed with the loaded WaitBlock manifest. Pending/arm/replay `valid` fields
are the explicit active sentinel, so cycle zero is a valid start cycle. Runtime
kernel UID changes assert that the prior launch drained, then reset WaitBlock
totals, release-service accounting, pace maps, semantic anchors, late-report
state, sync work counters, and sync/TMA/WaitBlock log budgets. The existing
unfulfillable-contract gate aborts before the watchdog. Local source-contract
tests pass. The back-to-back runtime kernel test is deferred server evidence
and is not an Opt 12 closure gate.

### B6 pure hold query and scheduler accounting

`peek_wait_block_hold_mode(warp_id)` must remain the only scheduler-side
WaitBlock query. Complete the audit of legacy `mbarrier_spin` and stall metrics
so no read-only classification path calls a release service, named-BAR
`waiting()`, `tma_flush`, cache access, or another state-mutating readiness
helper.

Acceptance requires repeated queries to leave generations, pending waits,
counters, cursors, and scheduler order unchanged. The SR-M0A population must
partition every active hardware warp exactly once into selected,
eligible-not-selected, WaitBlock-held, blocked-with-valid-head,
no-valid-instruction, drain, or inactive.

**Implementation checkpoint.** Both the authoritative issue predicate and the
legacy mbarrier-stall metric now obtain WaitBlock state only through
`peek_wait_block_hold_mode()`. The legacy alias previously included named-BAR
state and could misclassify it as a WaitBlock hold; it is now WaitBlock-only.
The existing SR-M0A enum-based partition and per-cycle conservation remain the
acceptance mechanism. Local source-contract tests pass.

### B7 metric reconciliation

Keep issued wait outcomes, logical WaitBlock episodes, TMA completion
ownership, and exclusive scheduler population as separate metric domains.
Close the final equations:

```text
captured_poll_holds == captured_poll_releases
captured_poll_aux_holds == captured_poll_aux_releases
partial_join_holds == partial_join_releases
partial_deferred == partial_replayed + partial_folded
TMA bindings created == completed + in_flight
cross_barrier_completion_applications == 0
fold-off retired warp instructions - fold-on retired warp instructions
    == WaitBlock skipped warp occurrences
```

Do not compare skipped warp occurrences directly with active-lane
`gpu_sim_insn`. Acceptance requires all equations at kernel scope and zero
active state at finalization for both K5 and K10.

**Implementation checkpoint.** WaitBlock summaries now expose retired warp
instructions in the same unit as skipped occurrences. For a single trace run,
the equivalent invariant is:

```text
sum(TRACECHECK.warp_insts)
    == sum(retired_warp_instructions)
     + sum(captured_poll_advanced_occurrences)
```

Final log analysis must close this equation together with WaitBlock
hold/release, partial replay, release-fingerprint, TMABIND byte/ownership,
named-BAR, SR-M0A, CTA-count, and clean-exit checks. Named-BAR teardown now
explicitly aborts on residual participant or credit state even when C/C++
assertions are disabled.

**Final regression.**

```text
K5:  retired 14,448,016 + skipped 371,587 = traced 14,819,603
     releases 6,481 = fingerprints 6,481
K10: retired 19,912,509 + skipped 1,084,920 = traced 20,997,429
     releases 9,061 = fingerprints 9,061
```

Both runs have exact release-service visit counts, complete TMA binding/byte
conservation, zero named-BAR leaks, and `STALLALIGN FINAL status=PASS`. One K10
BAR summary was split by concurrent stderr output; its continuation contains
all four leak counters at zero. The split was caused by chained `std::cerr`
insertions from concurrent OpenMP SM threads. BAR teardown summary, leak, and
abort records now assemble off-stream and use one `fprintf(stderr, ...)` call
per completed record/block; a local source-contract test prevents reintroducing
chained teardown output. This diagnostics-only fix does not alter simulated
state or timing.

### Deferred generalization

D0 generic ISA-wide effects and D1 repeated-EXCH initialization epochs remain
deferred. They are not blockers for the validated FA3 lifecycle and must not be
mixed into the K5 timing correction or B1 validation.

This document is the authoritative index and the live execution record for
B0-E. The detailed per-work-type designs and the historical evidence live in the
linked companion files below; this file is self-contained for deciding *what to
do next* and *what must not be re-proposed*.

## Document map

| Document | Work type | Authority |
|---|---|---|
| **WAIT_BLOCK.md** (this file) | Index, shared invariants, Batch B scope table, B0-E result and remaining work | Authoritative for status and next action |
| [WAIT_BLOCK_POSTPROC.md](WAIT_BLOCK_POSTPROC.md) | Offline trace post-processing / sidecar classification | Authoritative for the two-strategy classifier |
| [WAIT_BLOCK_SIMULATOR.md](WAIT_BLOCK_SIMULATOR.md) | Simulator runtime paths + partial-lane v2 replay | Authoritative for runtime hold/release/replay |
| [WAIT_BLOCK_TMA_OWNERSHIP.md](WAIT_BLOCK_TMA_OWNERSHIP.md) | Exact TMA-command-to-barrier completion ownership (Batch B) | Authoritative for B0-A … B0-D, B1, metrics |
| [WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md) | Chronological server runs; rejected ticket design; reverted experiment | Historical only; not implementation authority |

Supporting documents (`TMA_BASE_ADDR.md`, `TMA_exact_base_mapping_integration.md`,
`TMA_ISA.md`, `TMA_ARCH.md`, `TMA_LATENCY_INJECTION_H100.md`,
`SPIN_WAIT_SIM_MODELING.md`, `SPIN_WAIT_POSTPROCESS.md`) are historical/supporting
material. Where they conflict with the companion designs above, the companion
designs win.

**Debug records:** the K10 `.e51` whole-warp park deadlock evidence is preserved
in [WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md) section 6A.
The K5 all-live WaitBlock-cycle evidence lives in
[WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md) sections 2-6 plus the
"Historical B0-E root-cause analysis" section below.
**Historical pre-E0 handover:** inlined in the "Historical pre-E0 handover
snapshot" section below.

## Current B0-E authority (intentionally narrow)

1. B0-D at `39abefb` is the fixed exact TMA-ownership baseline.
2. B0-E at `087b53f`, K5 `.e90`/`.o90`, and K10 `.e66`/`.o66` is the
   exact-generation WaitBlock correctness result.
3. The ABI-correct v2 trace and compact ticket artifact are authoritative for
   the 90,868 eligible exact-generation actions. The ordered-candidate and
   parity-only designs remain rejected historical evidence.
4. B1, B6, and B7 are runtime validated. B2, B3, and B5 implementation and
   local source-contract tests are complete; their server-only negative and
   multi-kernel matrices are explicitly waived/deferred. Opt 12 is complete
   under this validation scope.
5. Separate cycle-accuracy work starts from the first K5
   `current_generation > target_generation` occurrence. Completion alone must
   not be used to close the 9,542 late-arrival timing divergences; this is not
   an Opt 12 completion gate.

## Historical pre-E0 handover snapshot (2026-08-31)

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

### Historical Stage E0 task boundary

The handover constrained the next session to **Stage E0 only**:

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

Stage E0 and the separately reviewed Stage E1 are now complete. This section is
retained only to preserve the original implementation boundary.

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
- Count-only fallback generation overshoot and a 32,000-cycle wait-hold timeout
  abort. Exact tickets use
  `current_generation >= target_release_generation`; an exact
  `current_generation > target_release_generation` continues and increments the
  late-arrival divergence counter.
- With a fixed TMA implementation baseline, `wait_block_enable=0` is
  bit-identical to that baseline. A separately approved cross-subsystem
  correctness repair such as B0-D establishes a new baseline; disabling WaitBlock
  must not restore the known-wrong TMA FIFO.

## Batch B scope and current status

| ID | Work item | Current status |
|---|---|---|
| B0-A | Repair UBLKCP sidecar callback-role selection | **IMPLEMENTED / VALIDATED** at `0f7cbc2`; K10 is 8 sites/6,528 occurrences at 512 bytes, and kernel 11 is 1 site/384 occurrences at 32,768 bytes |
| B0-B | Kernel-aware descriptor identity and K5 tx-size restoration | **IMPLEMENTED / VALIDATED** at `0f7cbc2`; chain 23/23, ambiguity 0, K5 `0x96b0` = 24,576 bytes, 192 128-byte AGU requests, and 768 32-byte sector requests |
| B0-C | Size-only K5/K10 byte-conservation diagnostic with existing FIFO/release order | **COMPLETE**; byte contract PASS, K10 clean exit at 138,456 cycles, K5 runtime size PASS but all-live WaitBlock abort at cycle 68,176 |
| B0-D | Exact TMA-command-to-barrier completion ownership | **COMPLETE / VALIDATED** at `39abefb`; `.e85` validates ownership-clean out-of-order completion, and K10 `.e62` completes 384/384 CTAs with 132/132 clean SM-level TMABIND final records after exercising two simultaneous open contracts per warp |
| B0-E | K5 all-live cross-warp WaitBlock cycle (`0xb370`/`0x16c60`) | **COMPLETE / VALIDATED** at `087b53f`; K5 `.e90` completes 132/132 CTAs without forced release and K10 `.e66` completes 384/384. All WaitBlock/TMABIND summaries pass. K5's 9,542 late-arrival divergences remain a cycle-accuracy diagnostic |
| B1 | One SM-level WaitBlock release phase per cycle | **IMPLEMENTED / LOG-VALIDATED**; K5 `.e91` and K10 `.e67` have exact release-service accounting, and the invalid whole-subcore reversal experiment is retired |
| B2 | WaitBlock startup and provenance fail-fast checks | **IMPLEMENTED / LOCAL CONTRACT VALIDATED / SERVER NEGATIVE TEST WAIVED** |
| B3 | Assert-before-clear lifecycle checks | **IMPLEMENTED / LOCAL CONTRACT VALIDATED / SERVER NEGATIVE TEST WAIVED** |
| B4 | Zero-active no-op and active required-payload checks | **IMPLEMENTED** |
| B5 | Configurable watchdog/heartbeat and kernel-scoped counters | **IMPLEMENTED / LOCAL CONTRACT VALIDATED / SERVER MULTI-KERNEL TEST DEFERRED** |
| B6 | WaitBlock-only non-mutating hold query | **IMPLEMENTED / VALIDATED** by K5 `.o93` and K10 `.o69` SR-M0A PASS |
| B7 | Fold/WaitBlock metric integrity and final reconciliation | **IMPLEMENTED / VALIDATED** by K5 `.e93`/`.o93` and K10 `.e69`/`.o69` |
| D0 | Generic ISA-wide effect framework | **DEFERRED**; exact sidecar allow/proof sets remain authoritative |
| D1 | Generic repeated-EXCH initialization epochs | **DEFERRED** for current FA3 one-EXCH-per-barrier contract |
| D2 | Full trace regeneration | **COMPLETE**; the `9e92fe9` trace is the comparison baseline and the ABI-correct v2 capture after `9bb9630` supplies the accepted contract ordinals and exact tickets |

No item may weaken the existing WaitBlock generation, cursor/replay, or
semantic-work invariants. Feature-off identity is evaluated within one TMA
ownership baseline; B0-D intentionally retires the legacy-FIFO baseline for both
WaitBlock-on and WaitBlock-off execution.

## Historical B0-E root-cause analysis

B0-D is complete (see [WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md) sections 5-6
for the `.e84`/`.e85`/`.e62` evidence). B0-E work uses `39abefb` plus the `.e62`
result as its fixed exact-ownership baseline. The analysis below describes the
WaitBlock liveness defect that Stage E0/E1 subsequently resolved.

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

The CTA124 protobuf establishes the exact provider chain and proves that bypassing
a hold is not locally semantics-preserving:

- warp 12's held `0x74e0` occurrence is 72 source instructions before its
  `0x7980` ARRIVE to `0x16c60`;
- warp 4's held `0x5a50` occurrence is 199 source instructions before its
  `0x6700` ARRIVE to `0x16c60`;
- warp 0's held `0xb370` poll precedes the `0xa8a0` ARRIVE_EXPECT_TX and `0xa8b0`
  TMA LOAD that feed `0x16c50`, then the corresponding `0x16c38` producer work.

Those consumer prefixes contain HGMMA/MUFU and further synchronization, so
advancing a held warp directly to its provider ARRIVE would execute or skip
semantic work. Global park safety can reject or diagnose this state, but cannot
by itself manufacture a safe release.

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
Its cause ([WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md) section 6A) is a
whole-warp park of a
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

At this checkpoint, the local macOS environment had neither `nvcc` nor
`protoc`; generated protobuf compilation, NVBit linking, callback execution,
and the full FA3 acceptance comparison were still pending on the CUDA server.
Those steps were subsequently completed before Stage E1 was accepted.

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
work. This was the v1 result; v2 replaces that capture path and is evaluated
below.

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

The first server build of `f9484f3` loaded v2 successfully but faulted in K5
before channel flush. The error surfaced at the post-kernel
`cudaDeviceSynchronize`; the launch state allocation had already passed. The
v2 callback had extended the established NVBit callback ABI from 16 to 19
arguments. The follow-up fix restores the exact 16-argument ABI: three fixed
event-kind symbols select EXCH/ARRIVE/WAIT, while ARRIVE semantic and the
registry pointer low/high words occupy the existing unused operand-value slots.
No callback or channel-push count changes.

The ABI-correct v2 full FA3 run completed all instrumented kernels:

```text
K5 contract records             = 75,015
K10 contract records            = 266,705
K11 contract records            = 3,840
active sync occurrences                = 923,835
eligible TMA barriers                  = 2,916
primary-only WaitBlock actions         = 81,999
primary-only unique assignments        = 81,999
missing/mismatch/ambiguous      = 0/0/0
inconsistent assignments        = 0
semantic histogram              = baseline-identical
WaitBlock action/count/mask     = baseline-identical
B0-D TMA contracts/bytes        = baseline-identical
```

The first strict post-processing attempt reported
`wait_block_structure=false` because the comparator still treated INJECT
`observed_barrier_addrs` and `observed_wait_states` frequency histograms as
structural invariants. Eight K5 records changed only those two histograms.
Those distributions, like FOLD retry counts, depend on hardware scheduling and
are now diagnostic-only. Entry visit counts, path counts, effective/live masks,
partial replay proof, static FOLD structure, semantic work, and TMA ownership
remain hard equality gates. Reprocessing the same v2 trace after this correction
passes all strict gates and publishes the schema-v5 sidecar.

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

The consumer is implemented and the tested H100 config enables it after Stage
E0 passed. The following rules are authoritative:

1. Add exact `target_release_generation` to every eligible selected WaitBlock
   handshake occurrence: `INJECT_MISSING_WAIT` entry probes and every proven
   `FOLD_CAPTURED_POLL.poll_wait_pcs` probe.
2. At runtime, bind the wait to that target instead of deriving a target from
   the simulator's current phase.
3. Use:

```text
current_generation < target  -> hold
current_generation >= target -> dependency already complete; continue
```

4. Count `current_generation > target` as a late-arrival timing divergence. It
   is not a reason to bind the wait to a later same-parity generation.
5. Count-only barriers retain the existing phase-derived fallback. Auxiliary
   FOLD fallback probes remain nonblocking. Auxiliary exact probes may hold and
   release without cursor movement; only the primary FOLD probe may advance the
   cursor. Partial replay ownership and B0-D TMA completion identity remain
   unchanged.
6. Validate K5 and K10 separately from B1. Require K5 clean exit without forced
   releases, K10 384/384 completion, all WaitBlock/TMABIND summaries PASS, and
   unchanged semantic TMA/BAR/memory work. Report total-cycle and producer/
   consumer-tail deltas rather than accepting completion alone.

Stage E1 file scope is:

```text
util/tracer_nvbit/build_spin_wait_sidecar.py
util/tracer_nvbit/tests/test_build_spin_wait_sidecar.py
gpu-simulator/trace-parser/trace_parser.h
gpu-simulator/trace-parser/trace_parser.cc
gpu-simulator/trace-driven/trace_driven.cc
gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h
gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h
gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc
gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h
gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h
gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h
gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc
gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config
```

No B1 scheduler changes belong in E1.

The implementation binds a ticket before PC-indexed trace storage using:

```text
(trace_kernel_id, CTA x/y/z, CTA-local warp_id,
 unique_function_id, action_pc, source_warp_instruction_index)
```

Schema v2 stores 816,411 occurrence entries in 5,172 CTA-warp streams in a
separate compact artifact. The main sidecar plus ticket artifact is 32,038,200
bytes, compared with 271,695,970 bytes for the prior inline sidecar. The
artifact also carries all 2,916 eligible `(kernel, CTA, barrier)` keys.

The loader rejects a missing/duplicate ticket, action mismatch, mask mismatch,
barrier-address mismatch, wait-state mismatch, malformed stream ordering,
summary disagreement, or target-kind/barrier-eligibility mismatch. Exact
tickets use the comparison above in both immediate and parked paths. Fallback
tickets are accepted only for barriers absent from the eligible manifest and
use the previous phase-derived target.

The final full-trace check is:

```text
eligible barriers              = 2,916
eligible wait occurrences      = 90,868
unique generation assignments  = 90,868
INJECT exact targets            = 75,840
FOLD exact targets              = 15,028
fallback occurrences           = 725,543
missing/mismatch/ambiguous      = 0/0/0
inconsistent/rejected TMA       = 0/0
baseline conservation          = PASS
```

The scheduler's non-mutating hold predicate uses the same exact
`current_generation >= target` test as the release service; it does not fall
back to one-bit parity for an exact pending hold. The tested H100 config sets
`-wait_block_exact_generation_enable 1`.

B0-E acceptance must eliminate the 13-warp all-live cycle without dropping a
dependency, forcing a phase, guessing a generation, advancing a cursor outside
an approved release, or changing semantic TMA/BAR/memory work.

### Stage E1 simulator acceptance (`.e90` K5, `.e66` K10)

Both runs use `-wait_block_enable 1` and
`-wait_block_exact_generation_enable 1`.

K5:

```text
CTA completion                 = 132/132
gpu_tot_sim_cycle              = 85,172
WaitBlock summaries            = 132/132 PASS
TMABIND summaries              = 132/132 PASS
missing holds/releases         = 2,404/2,404
primary FOLD holds/releases    = 2,500/2,500
auxiliary FOLD holds/releases  = 1,553/1,553
partial deferred/replayed      = 3,726/3,726
partial join holds/releases    = 343/343
active holds/watchdog aborts   = 0/0
exact actions observed         = 26,561
late-arrival divergences       = 9,542
generation overshoots          = 0
```

All K5 late-arrival records are immediate-path observations. Of the 9,541
records whose fields survived concurrent stderr output, 4,301 are at `0x5a50`
and 4,413 are at `0x5bc0`; the remaining 827 are split across `0x13b0`,
`0x35e0`, `0x3750`, and `0x74e0`. The generation distance is +1 for 6,846,
+2 for 2,667, and +3 for 28 records. One record's fields were interleaved with
a TMABIND line, while the per-SM summary counters retain the authoritative
9,542 total.

K10:

```text
CTA completion                 = 384/384
gpu_tot_sim_cycle              = 137,348
WaitBlock summaries            = 384/384 PASS
TMABIND summaries              = 132/132 PASS
missing holds/releases         = 2,888/2,888
primary FOLD holds/releases    = 3,263/3,263
auxiliary FOLD holds/releases  = 2,998/2,998
active holds/watchdog aborts   = 0/0
exact actions observed         = 57,777
late-arrival divergences       = 0
generation overshoots          = 0
```

The runtime `exact actions observed` counters include only ticketed
instructions that reach the action handler. They are lower than the artifact's
per-kernel exact-ticket totals (K5 31,123; K10 59,745) because an approved
primary FOLD advances past the remaining captured retry records after release.
Those skipped ticket slots are validated at load time but are not runtime
actions.

Relative to the B0-D K10 `.o62` result at 136,931 cycles, exact-generation mode
adds 417 cycles (+0.30%). Semantic work is conserved: both runs issue 30,336
TMA commands, issue and complete 349,372,416 TMA bytes, execute the same 103,296
named-barrier operations, and report identical global read/write byte totals.

K5 internally conserves 6,432 TMA commands, 3,090,432 issued/completed requests,
and 98,893,824 issued/completed bytes. Its 132 named-barrier summaries contain
no leaked barrier IDs, participants, arrivals, or deferred-sync credits.

Therefore B0-E dependency correctness passes. The next cycle-accuracy target is
the K5 timing skew exposed by the 9,542 late-arrival divergences. Exact
generation identity prevents parity aliasing but cannot rewind a barrier that
has already advanced beyond the hardware-observed target.

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

**Superseded next step:** Stage E0 capture, offline uniqueness proof, and the
separately reviewed Stage E1 simulator consumer are complete. Do not re-add
either historical release probe.

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
every hardware warp serviced exactly once per SM cycle
cycle-start completion may release in the current cycle
issue-phase arrival releases no warp before the next cycle
release fingerprint count == released-hold count
trace cursor, semantic work, and conservation summaries unchanged
```

The former whole-subcore traversal reversal is rejected as a validation method:
it changes shared-pipeline arbitration and memory/TMA scheduling rather than
isolating release-service order. K5 `.e92` is the recorded negative result.

Record the total-cycle delta from the B0-D baseline, but do not impose an
unsupported fixed percentage or direction. A performance threshold may be added
only after repeated unchanged-binary runs establish a measurement envelope.

Do not combine B0 and B1 in one commit or one causal validation step. Only after
both B0-E and B1 pass may the `.e83` debugging session be closed and the older
Batch B document be treated as historical rather than authoritative.

## Code anchors

Verified simulator locations for the WaitBlock and TMA-generation logic. Role 2b
TMA generation equality is a B0-D invariant and must remain exact under any B0-E
or B1 change.

- Pure WaitBlock hold query:
  [sm.cc:L2747-L2755](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2747-L2755)
- Hold release, generation audit, watchdog, and replay:
  [sm.cc:L3352-L3802](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L3352-L3802)
- Phase flip and per-generation reset:
  [sm.cc:L3945-L4023](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L3945-L4023)
- Phase-only readiness:
  [sm.cc:L4026-L4037](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L4026-L4037)
- Wait generation auditor:
  [sm.cc:L4045-L4074](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L4045-L4074)
- Poll/missing-wait admission:
  [sm.cc:L4262-L4554](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L4262-L4554)
- Exact TMA completion generation (Role 2b, preserve):
  [sm.cc:L5098-L5133](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L5098-L5133)
  and
  [sm.cc:L5252-L5263](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L5252-L5263)
- Scheduler-facing non-mutating hold classification:
  [sm.cc:L5454-L5465](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L5454-L5465)

Line numbers are approximate anchors from the B0-E analysis session and should be
re-confirmed against the current tree before editing.

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
