# WaitBlock: Design Index and Live B0-E Work

**Status (2026-08-30):** Post-processing v5 and the partial-replay simulator are
implemented and full-trace validated. B0-A/B/C are complete. B0-D (exact TMA
completion ownership) is **complete and validated** at `39abefb`: K5 `.e85`
validates ownership-clean out-of-order completion and K10 `.e62` completes
384/384 CTAs with 132/132 clean SM-level `TMABIND` final records. The only open
correctness item is **B0-E**: a K5 all-live 13-warp cross-warp WaitBlock cycle
on SM124 (`0xb370`/`0x16c60`). Three offline pairing designs (ordered generation
ticket, ordered candidate-set, runtime UR recompute) were explored and
**rejected**, and a generic runtime deadlock-release attempt
(single-consumer release under a proven CTA-wide deadlock) was **implemented,
tested, and reverted** — it advanced K5 from generation 4 to generation 9 but
still tripped the hold watchdog, and it perturbed K10 timing (regression). The
tree is back at the `326ef0b` baseline. A **temporary, measurement-only,
PC-hardcoded K5 deadlock breaker** was added at `9e16c6f` solely to observe the
K5 completion cycle count; it is explicitly **not a fix**. Its first run
confirmed K10 stays clean but K5 **still does not complete** — the four consumer
PCs cleared let K5 reach later generations where new hang sites (`0x35e0`,
`0x3750`, producer FOLD `0xb0b0`, partial `0xab40`) surface, i.e. the K5 deadlock
is multi-layer. **The probe code has since been reverted** (only the four code
files: `shader.h`, `gpu-sim.cc`, `sm.cc`, config); this document's probe findings
and handover are intentionally kept for the next session. See "Temporary K5
hardcoded deadlock breaker" → "HANDOVER — next session" for the A/B decision that
is the immediate next step. B1 and later items remain pending.

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
3. The next authorized work is the first-divergence and global park-safety
   analysis under the gates in this document. No new trace schema or simulator
   release heuristic is authorized.

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
- Generation overshoot and a 32,000-cycle wait-hold timeout abort.
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
| B0-E | K5 all-live cross-warp WaitBlock cycle (`0xb370`/`0x16c60`) | **OPEN / ROOT-CAUSE UNDERSTOOD / NO GENERIC FIX YET**. Rejected: three offline pairing designs (ordered generation ticket, ordered candidate-set, runtime UR recompute) — all blocked by missing cross-warp order; a 1-bit parity plus per-warp modulo state cannot name a wait's absolute release generation. Implemented then **reverted**: generic runtime single-consumer deadlock-release (advanced K5 gen4→gen9 but still watchdog-aborted; regressed K10 timing). Tree restored to `326ef0b`. A **temporary PC-hardcoded, measurement-only** deadlock breaker (flag `-wait_block_k5_hardcoded_deadlock_release`, default off) is in place only to observe the K5 completion cycle count — not a fix |
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

- The primary axis is (1) global park safety and (2) producer/consumer timing
  realignment.
- Retiring the Role 2a audit is optional cleanup that may follow, and only
  changes an abort into a hang; it must not be presented as resolving the cycle.
- Role 2b (TMA completion identity) stays intact regardless.
- The ordered-ticket design and the post-`951b058` experiment are recorded as
  the rejected/closed alternatives in
  [WAIT_BLOCK_HISTORY.md](WAIT_BLOCK_HISTORY.md). Do not re-propose ordered
  generation tickets, forced phase flips, guessed generations, or unproven
  whole-warp releases.

## Required B0-E implementation gates

1. Treat K10 `.e62` as the completed B0-D baseline: 384/384 CTAs complete,
   132/132 TMABIND final records pass, and every WaitBlock summary passes.
2. Extract the chronological `.e85` simulator-log timeline for CTA124 warps 0,
   4, and 12 around the `0x16c50`/`0x16c60` flips and the `0x74e0`/`0xb370`
   holds. Identify the first simulator interval that diverges from the intended
   producer-to-consumer order. This is root-cause evidence, not an input to a
   ticket resolver.
3. For every hold in the terminal cycle, determine whether the held warp still
   has an ARRIVE or other semantic event ahead of the hold that another warp
   requires. Compare completing SM119 against hanging SM124 at the same
   occurrence.
4. Use the evidence from gates 2-3 to select a pure timing correction, a
   park-safety correction, or both. Do not reintroduce ordered generation
   tickets, forced phase flips, guessed generations, or unproven releases.
5. Review the correction separately from B1 and preserve every semantic
   TMA/BAR/memory occurrence, cursor ownership rule, and B0-D byte/identity
   invariant.
6. Require focused cycle tests plus K5 clean exit with every
   `[WAITBLOCK][summary] status=PASS` and every `[TMABIND][FINAL] status=PASS`.

B0-E acceptance must eliminate the 13-warp all-live cycle without dropping a
hold, forcing a phase, guessing a generation, advancing a cursor outside an
approved release, or changing semantic TMA/BAR/memory work.

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

### HANDOVER — next session

**State:** tree at `9e16c6f` (three reverts + the temporary probe on top of
`326ef0b`). The generic B0-E fix remains unsolved; the probe is measurement-only.
Goal of the probe track is still just to observe K5's completion cycle count.

**Decision pending (asked, not yet answered):** how to get K5 to actually finish.
Two options were put to the user:

- **Option A — extend the PC whitelist.** Add `0x35e0`, `0x3750`, `0xb0b0`, and
  `0xab40`. Caveats: (1) `0xab40` is `DEFER_PARTIAL_MISSING_WAIT`, whose release
  path differs from `MISSING_WAIT_HOLD`, so the hook needs a separate branch for
  it (the current hook only matches `MISSING_WAIT_HOLD`); (2) `0xb0b0` is a
  producer `FOLD_CAPTURED_POLL`, so releasing it lets the producer advance past a
  poll the HW actually spun on — acceptable for measurement only; (3) advancing
  further may surface yet more PCs (the set has grown every time the run gets
  further), so this may need several build/run round-trips.
- **Option B (assistant's recommendation) — strategy-agnostic held-cycles
  threshold.** Replace the PC whitelist with "if any hold's `held_cycles` exceeds
  a high threshold (e.g. 28,000, below the 32,000 watchdog), force-release it,
  regardless of strategy or PC." Rationale: covers all layers at once (INJECT,
  FOLD, DEFER_PARTIAL) so K5 likely completes in one run; K10 is safe because its
  longest observed transient park is ~1,659 cycles, far under 28,000. This drops
  PC-hardcoding entirely in favor of a single dwell-time gate, which matches the
  "just let it finish so we can read the cycle count" intent. Still measurement-
  only, still default-off, still not a shippable fix.

**Immediate next step:** get the user's A-vs-B decision, then implement, rebuild,
rerun K5 (`gpu_tot_sim_cycle` is the target number) and K10 (must stay 384/384
PASS). If Option A and a later run surfaces still more PCs, that is itself
evidence that the deadlock is unbounded-layer and Option B is the pragmatic path.

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
