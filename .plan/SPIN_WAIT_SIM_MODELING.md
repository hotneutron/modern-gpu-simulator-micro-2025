# Producer Spin-Wait Collapse — Simulator Modeling Design

**Role.** This is the **simulator-modeling** design doc for spin-wait collapse: how
the simulator *consumes* the spin-wait sidecar and models the producer/consumer
mbarrier handshake at replay time. It is one of three linked docs in the Opt 12
"lightweight-trace → post-process → sim-model" chain:

- **tracer** (Opt 11): [`Lighter_NVbit_Tracing.md`](Lighter_NVbit_Tracing.md) —
  the compact full-capture trace this all runs on.
- **post-processing**: [`SPIN_WAIT_POSTPROCESS.md`](SPIN_WAIT_POSTPROCESS.md) —
  the static spin-wait sidecar (region detection, `collapsible` gate). Read-only
  wrt the trace.
- **simulator modeling** (this doc): the collapse block/skip mechanics, stall
  taxonomy, and the runtime generation-counter **audit**.

> **STATUS (2026-08-12): IMPLEMENTED, then redirected.** §1–§12 are the original
> pre-implementation design review (kept as the record of *why* each mechanic is
> shaped the way it is). Collapse was then implemented and run; **K5 deadlocked on a
> low-flip cross-warp barrier double-flip.** The root cause, and the resulting
> **change of direction (collapse = accuracy audit, generation = assert-only, never
> force-release)**, are in **§13**. §13 SUPERSEDES the earlier decisions in **§6.1**
> (“keep 1-bit parity, no epoch counter”), **§6** (watchdog framing), and **§12.8**
> (“circular wait precluded”) wherever they conflict. Read §13 first; treat §1–§12
> as background.

## 1. Goal restated

At a detected producer spin region, stop replaying the capture-time poll count.
Instead: **block the warp at the region's poll until the modeled mbarrier is
ready, then leave the loop and continue into the semantic block that follows.**
The number of polls becomes a simulator output, not a trace input.

Constraints (from prior analysis):
- Semantic work (HGMMA/TMA/MUFU/BAR/memory/arrive, TMA cmd/byte) must stay
  bit-identical (Gate 1). Only spin-poll icount may change.
- Feature-off must be bit-identical to today's baseline.
- Must not silently hang (runtime watchdog mandatory).

## 2. The two counters that make this non-trivial

`trace_shd_warp_t` (trace_driven.cc) advances TWO cursors as it feeds instructions:

1. `used_insts` — the warp's linear cursor into `traced_pcs` (whole-stream index).
2. `map_warp_traces[pc].num_used_instructions[0]` — a PER-PC occurrence cursor,
   used to bind the correct dynamic payload (addresses, sync runtime) to each
   dynamic instance of that PC.

Any "skip the remaining N poll iterations" scheme must keep BOTH consistent, or
`trace_done()` (`used_insts == traced_pcs.size()`) and per-PC payload binding
break. This is the core risk in a cursor-jump approach.

## 3. Two candidate mechanisms

### Option A — "block only", let the trace replay the spin (NO cursor jump)

Revive the currently-dead `m_pending_sync_waits[warp_id].valid` path so that
`SM::warp_waiting_at_barrier()` returns true while the region's barrier is
not-ready. The warp is parked; when the barrier flips, `warp_waiting_at_barrier`
returns false and the warp resumes.

- **Pro:** trivial, touches no cursor; feature-off bit-identity is easy; Gate 1
  (semantic work) is automatically preserved because nothing is skipped.
- **Con (fatal for timing):** we already proved this cannot fix timing. After the
  barrier flips, the warp resumes AT the poll and `next_traced_pc` still points
  back into the loop (HW spun more there), so it replays all remaining recorded
  poll iterations. icount is preserved but the warp still burns the recorded
  loop-body latency after readiness — the exact "ready but still spinning" error
  the user rejected. Timing stays wrong.

Conclusion: Option A alone is insufficient. Keep the block, but it must be paired
with skipping the residual polls.

### Option B — "block + cursor jump to exit" (RECOMMENDED)

At the region's primary_wait poll:
- **not ready:** park the warp (same block mechanism as A).
- **ready:** advance the warp's trace cursor past the remaining recorded loop
  iterations to `exit_pc`, so the residual polls are consumed WITHOUT being
  issued, then fall into the semantic block.

The jump is the hard part; §4 specifies it precisely.

## 4. Where and how to jump (Option B mechanics)

Key realization: the simulator ALREADY has a "redirect to a non-fall-through PC"
mechanism — `IBuffer_Remodeled::issued()` compares `next_not_taken_pc` with the
captured `next_traced_pc`; when they differ it `flush()`es the fetch buffer and
sets `m_next_pc_to_fetch_request = next_traced_pc`. Fetch then re-primes from the
new PC and the per-warp `get_pc()`/`used_insts` follow because fetch pulls via
`get_next_trace_inst(pc)`.

So the cleanest jump is NOT to hand-edit `used_insts`, but to make the warp's
control flow arrive at `exit_pc` through the existing fetch/flush path. Two
sub-options:

- **B1 — redirect at the poll's issue.** When the poll is allowed to issue (i.e.
  we decided "ready, leave the loop"), override the effective `next_traced_pc`
  for that dynamic instruction to `exit_pc`. `issued()` then flushes and re-fetches
  from `exit_pc`. The intervening recorded poll occurrences for this region are
  never fetched, so they are never issued.
  - Risk: the per-PC occurrence cursor (`num_used_instructions`) for the skipped
    poll PCs is NOT advanced (they were never fetched). That is FINE as long as
    those PCs are never needed again — but if the same wait_pc recurs in a LATER
    phase of the same warp, its occurrence cursor is now "behind", and the payload
    bound to a future poll would be the skipped one's. Must verify poll PCs carry
    no per-occurrence payload that matters (mbarrier polls read no addresses; the
    barrier addr is recomputed from operands each time — likely safe, must confirm).

- **B2 — explicit dual-cursor skip.** Add a `trace_shd_warp_t` method that, given
  the region's `body_pcs` and a target `exit_pc`, advances `used_insts` and each
  body PC's `num_used_instructions` by exactly the counts consumed, until the next
  instruction is `exit_pc`. Fully explicit; keeps both cursors consistent by
  construction.
  - Risk: must replicate the exact per-PC accounting `get_next_trace_inst` does;
    off-by-one desyncs `trace_done()`.

Recommendation: prototype **B1** first (reuses the proven flush path, less new
state), with an assert that the skipped body PCs have no bound payload. Fall back
to **B2** if B1's occurrence-cursor drift proves unsafe.

### RESOLVED: use B2 (explicit dual-cursor skip)

Two investigations settled this:

**Investigation 1 — poll payload per occurrence (trace_parser.cc:316-321).** Each
dynamic poll occurrence carries its own `sync_barrier_addr` and `sync_semantic_raw`
parsed from protobuf and stored in `map_warp_traces[pc].instructions[occurrence]`.
Measured on K5 CTA0 collapsible wait_pcs:
- `barrier_addr` is CONSTANT per wait_pc (good — B1 could not mis-bind the barrier).
- but `semantic_raw` takes TWO values on some regions (`0xb1f0`, `0xb2b0`: `0x0`
  and `0x80000000`) — bit31 is the phase-parity of the wait. So consecutive
  occurrences of the SAME poll pc differ in their captured wait_state.

Consequence: B1 (redirect via `next_traced_pc` override, leaving the skipped
occurrences' per-pc `num_used_instructions` un-advanced) would leave the per-pc
occurrence cursor behind, so a LATER poll of the same pc would bind a stale
phase-parity `semantic_raw`. Poll PCs are not semantic work, so semantic binding
is unaffected, but the mis-bound phase parity could corrupt a subsequent
`is_sync_wait_satisfied` on that barrier. B1 is therefore unsafe here.

**Investigation 2 — exit_pc uniformity (K10 producer regions).** For every
collapsible region the loopback BRA's successor is ONLY {loop_head, sidecar
exit_pc} — no third target. Verified on K10 `0xb960`/`0xb820`/`0xbb40` (551/469/100
loop-backs vs 7/6/1 exits, exit pc matches the sidecar exactly). The
first-iteration-different regions (leading WARPSYNC.ALL / STL-LDL) exit to the
same pc because the first-iteration prologue is consumed before the primary_wait
is reached. So the sidecar `exit_pc` is a trustworthy jump target.

**Decision: B2.** Add a `trace_shd_warp_t` method that, on a ready collapse, walks
the linear cursor forward from the current position to the next `exit_pc`,
advancing BOTH `used_insts` and each traversed body PC's `num_used_instructions`
exactly as `get_next_trace_inst` would, so payload cursors stay consistent and
`trace_done()` is unaffected. Investigation 2 guarantees the walk terminates at a
well-defined exit_pc; Investigation 1 is why we must advance the per-pc cursors
rather than jump over them (B1). The only residual risk is off-by-one in
mirroring `get_next_trace_inst`'s increment — covered by a feature-off bit-identity
test and an assert that the post-skip head pc equals `exit_pc`.

### 4a. Relationship to fetch prefetch (why collapse doesn't fight it)

The remodeled IBuffer prefetches on a FALL-THROUGH assumption. `get_next_pc_to_
fetch_request()` (ibuffer_remodeled.cc:81-96) pushes empty entries for
`res, res+16, res+32, ...` and sets `m_next_pc_to_fetch_request = res + 16*width`
— i.e. it guesses the next PCs are sequential. The real payload (and the per-pc
occurrence-cursor increment) is bound later, at fetch/decode, by
`get_next_trace_inst(pc)` (subcore.cc:1656).

When a branch is actually taken, `issued()` (ibuffer_remodeled.cc:119-131) detects
`next_traced_pc != pc+isize`, calls `flush()`, and re-points fetch at
`next_traced_pc`. Crucially `flush()` calls `decrease_num_used_inst` for the
discarded prefetched entries (line 145), rolling back the occurrence cursor they
advanced. A producer spin loop's `BRA -> loop_head` is a backward taken branch, so
**every spin iteration already triggers a flush + cursor rollback today** — the
loop is not a quiet fall-through, it is a fetch that mis-predicts and is corrected
each turn.

Implication for collapse: at the moment we decide to leave the loop, the IBuffer
may hold mis-prefetched loop entries. We do NOT need new machinery to clean them —
the existing `flush()` already discards them and rolls their cursor back. Collapse
then advances the (now-consistent) linear+per-pc cursors to `exit_pc` and lets the
next fetch start there. So collapse composes with prefetch by reusing the exact
flush/rollback the loop already uses every iteration.

### 4b. Reused vs new code (explicit)

- **Reused (no change):** `IBuffer_Remodeled::flush(false)` — empties the IBuffer
  and rolls back occurrence cursors of prefetched-but-unissued entries; and the
  normal fetch path, which will re-prime from `get_pc()` == `exit_pc` after the
  skip.
- **New (one method):** `trace_shd_warp_t::skip_spin_to_exit(exit_pc)` — the
  FORWARD analogue of `get_next_trace_inst`'s cursor bookkeeping:
  ```
  while (used_insts < traced_pcs.size() && traced_pcs[used_insts] != exit_pc) {
      pc = traced_pcs[used_insts];
      map_warp_traces[pc].num_used_instructions[0]++;   // per-pc occurrence cursor
      used_insts++;                                       // linear cursor
  }
  assert(used_insts < traced_pcs.size() && traced_pcs[used_insts] == exit_pc);
  ```
  It does NOT call `flush`'s redirect (that only moves the fetch PC, not the
  cursors — the B1 hazard). The bound `used_insts < traced_pcs.size()` protects
  the `get_pc()` assert at trace_driven.cc:172.

  **CRITICAL — stop on `exit_pc`, not on "leaving body_pcs".** `body_pcs` is
  `loop_head .. up to but EXCLUDING loopback_bra_pc`, so the loopback BRA
  (`loopback_bra_pc`, e.g. 0xacf0) is NOT in `body_pcs`. The set of PCs the walk
  actually consumes each iteration is `body_pcs ∪ {loopback_bra_pc}` =
  {0xacb0,0xacc0,0xacd0,0xace0,0xacf0}. If the walk instead stopped as soon as it
  saw a PC not in `body_pcs`, it would halt at the FIRST loopback BRA and land the
  cursor on 0xacf0 instead of the exit — corrupting the collapse. The condition
  MUST be the positive `traced_pcs[used_insts] != exit_pc`, advancing over
  whatever PCs appear (verified contiguous, no foreign interleave), which
  naturally consumes the loopback BRA occurrences too.

  **Landing: stop AT exit_pc, then ISSUE it normally — do NOT jump to its target.**
  exit_pc (e.g. 0xad00) is itself a BRA to the semantic block (0x1380). Leave
  `used_insts` pointing AT the exit BRA occurrence and let it issue: its captured
  `next_traced_pc` (=0x1380) differs from `pc+isize`, so the existing
  `IBuffer::issued()` redirect flushes and re-fetches at 0x1380 — the standard
  taken-branch path, no special fetch surgery. Landing directly on 0x1380 instead
  would leave the exit BRA's occurrence cursor unincremented and force a manual
  fetch redirect; do not do that.
- **Not reused (rejected):** the `m_next_pc_to_fetch_request = next_traced_pc`
  redirect alone (B1) — leaves per-pc cursors behind, re-binds stale phase-parity
  payload (Investigation 1).

Sequence at a ready collapse: (1) issue/observe the primary_wait poll, decide
ready; (2) `flush(false)` the warp's IBuffer; (3) `skip_spin_to_exit(exit_pc)`;
(4) next fetch naturally starts at `exit_pc`.

### 4c. Metrics — no special handling needed, and why

All headline metrics are incremented only when an instruction actually
executes/completes, in `SM::warp_inst_complete` (sm.cc:3073-3112):
`gpu_sim_insn` (active-lane count), `m_num_sim_insn` / `m_num_sim_winsn` (warp
icount), and the SP/SFU/MEM committed counters. Opcode-family work (HGMMA / MUFU /
TMA / BAR) is likewise counted on execution, not on fetch.

Collapse skips instructions by advancing cursors WITHOUT issuing/completing them,
so:
- Skipped spin polls (PHASECHK / NANOSLEEP / loopback BRA) never reach
  `warp_inst_complete` → they drop out of `gpu_sim_insn` and icount. This is the
  intended Gate-1-allowed reduction (spin poll count is a model output now).
- Semantic ops are never in a `semantic_free_body` region (verified by the
  sidecar), so HGMMA/TMA/MUFU/BAR/memory/arrive counts and TMA cmd/byte totals are
  untouched → Gate 1 (semantic-work invariant) holds automatically.

Two metrics to keep in mind (not blockers):
- `[TRIMCHK]` issue counters (trimchk_spin_phasechk_issued, etc.) count at issue,
  so they fall under collapse. That is the desired signal (fewer spins issued),
  not an error.
- `[TRACECHECK]` expected warp/active-lane instruction contract is computed at
  trace LOAD time from `traced_pcs.size()`, so `expected` is unchanged while
  `actual` (executed) drops. Under collapse-on, expected != actual is EXPECTED and
  consistent with the relaxed Gate 1. Confirmed `[TRACECHECK]` is a print, not an
  assert (trace_parser.cc:621); the only hard invariant on the cursor is
  `get_pc()`'s `assert(used_insts < traced_pcs.size())` (trace_driven.cc:172),
  which `skip_spin_to_exit`'s bound respects.

Feature-off keeps every path identical (no skip, no block), so all metrics are
bit-identical to today.

## 5. Blocking mechanism (shared by A and B)

- On a not-ready primary_wait poll of a collapsible region, set
  `m_pending_sync_waits[warp_id] = {valid=true, is_trywait, key, wait_state_raw}`
  (currently this path is dead because handle_sync_instruction always resets it).
- `SM::warp_waiting_at_barrier()` already consults `m_pending_sync_waits` and
  re-tests `is_sync_wait_satisfied` each call, clearing it when satisfied. This is
  wired into subcore eligibility via `c_warp->waiting()`, so parking one warp does
  NOT stall the subcore — other warps still issue (verified: the scheduler loop
  `continue`s past an ineligible warp). This is what lets the consumer run and
  flip the barrier while the producer is parked.
- Only arm this for polls that match a sidecar region with
  `semantic_free_body && is_spin_loop`. All other PHASECHK/TRYWAIT stay nonblocking
  (unchanged).

### 5.1 Exact park point and pre-poll semantics (resolved)

The park is armed at the region's `primary_wait_pc` (the last PHASECHK, e.g.
0xace0), NOT at the earlier TRYWAIT (0xacc0). Evidence: the TRYWAIT and the
PHASECHK have byte-identical operands — same barrier address `[R7+URZ+0x16800]`,
same phase-parity reg R15 — so they test the SAME barrier at the SAME phase; the
PHASECHK's `!P0` result is what gates the loopback BRA, and the sidecar binds the
resolved runtime `barrier_addr` to `primary_wait_pc`.

An iteration issues IMAD(0xacb0)+TRYWAIT(0xacc0)+NANOSLEEP(0xacd0) BEFORE reaching
the parking poll. Issuing those once is harmless and in fact scoreboard-faithful:
- IMAD writes only R7, a scratch base recomputed each iteration and redefined at
  0x13a0 before its next read on the exit path (dead across exit).
- TRYWAIT writes only P0 (scratch, overwritten by every poll); NANOSLEEP has no
  destination register and a dedicated latency knob.
- The RD1/WR2 scoreboards armed by this iteration's TRYWAIT/PHASECHK are exactly
  those a real (untrimmed) exit iteration arms; WR2's only consumer is the
  skipped loopback BRA, RD1's later consumer (0x13a0) sees the identical lifecycle
  — matching `barrier_self_tied=true`.

Implementation rule: issue the pre-poll trio ONCE when first reaching the parking
iteration; on a not-ready poll at primary_wait_pc, hold the warp ineligible (do
NOT re-issue the trio each retry); on ready, run `skip_spin_to_exit`.

## 6. Runtime watchdog (mandatory)

Reuse the observe threshold idea in the live path: if a warp stays parked on the
same barrier for > threshold cycles, do NOT hang — emit a fatal diagnostic
(sm/cta/warp/wait_pc/barrier/arrive/expected/tx/phase) and abort (or, behind a
sub-flag, release the park and fall back to trace replay for that warp so the run
completes). This covers the two observe blind spots (mutual-collapse interaction,
2nd-order scheduling) called out in §9 of SPIN_WAIT_POSTPROCESS.md.

### 6.1 Phase-parity double-flip — keep the 1-bit model (reviewed, not a hazard)

> ⚠️ **SUPERSEDED by §13.2/§13.3 (2026-08-12).** This section concluded a
> double-flip cannot strand a parked warp. That is false for a **low-flip
> cross-warp barrier** (`0x16c20`, 2 flips/kernel): it flipped twice before the
> collapsed producer even reached its poll, on 21/132 CTAs. The 1-bit model is
> kept for the non-collapse issue path, but the collapse path now needs the §13.4
> generation **audit** to detect this. Read the reasoning below as the (refuted)
> original argument.

`is_sync_wait_satisfied` releases when `barrier.phase != wait_parity` (1-bit
parity, bit 31 of wait_state_raw). A theoretical concern: if the barrier flips
TWICE while a warp is parked (P -> !P -> P), the level compare could miss the
wakeup. Reviewed and rejected as a hazard, keeping the existing 1-bit model:

- The 1-bit parity is NOT a simulator convenience — it is the exact Hopper
  `mbarrier.try_wait.parity` semantic (SYNC_ISA.md §; new_sync_plan.md
  "Two-phase parity model is enough"). Replacing it with an epoch counter would
  DIVERGE from the validated HW model that the whole sync implementation rests on.
- Missed-wakeup-by-double-flip is precluded structurally AND empirically:
  (a) the simulator re-tests every scheduler cycle with no latch (sm.cc
  2242-2264), worst-case re-eval gap is a few cycles; (b) the mbarrier protocol
  forbids a second flip before the waiter consumes its phase (the buffer cannot be
  reused until the consumer arrives), so a double-flip cannot happen "over" a
  still-waiting warp; (c) measured minimum flip interval on the SAME barrier is
  1738 cyc (K5) / 2435 cyc (K10), with zero sub-8-cycle gaps — three orders of
  magnitude above the re-eval gap.
- The §6 watchdog is the backstop for any unforeseen never-release.

Decision: keep the 1-bit phase compare unchanged (matches HW and the existing
sync model); do NOT add an epoch counter for v1.

## 6.5 Stall-reason taxonomy must be corrected for the new park (REQUIRED)

Collapse introduces a NEW reason a warp is not eligible: it is parked on an
mbarrier via `m_pending_sync_waits`. Today that path is dead, so no mbarrier stall
is ever attributed. The subcore's per-cycle stall classifier currently folds our
park into the WRONG bucket, which would corrupt exactly the analysis this whole
effort depends on.

The bug: the collapse park routes through `c_warp->waiting()` →
`SM::warp_waiting_at_barrier()` → `m_pending_sync_waits`. But `waiting()` also
covers NAMED barriers (`m_barriers`), and the classifier at subcore.cc:989 does:
```
if(!is_not_warp_waiting_in_programmer_barrier || !is_not_warp_waiting_ldgdepbar)
    is_any_waiting_in_inst_barrier = true;   // = named barrier / ldgdepbar
```
with `is_not_warp_waiting_in_programmer_barrier = !c_warp->waiting()`. So a warp
parked on an mbarrier would be counted as `inst_barrier` (named-barrier) stall,
NOT as the mbarrier bucket `is_any_waiting_in_wait_barrier`.

Why it matters (subcore.cc:534-539 taxonomy):
- `is_any_waiting_in_wait_barrier` = mbarrier (NCU barrier/long_scoreboard), TMA axis
- `is_any_waiting_in_inst_barrier` = named barrier + ldgdepbar, TMA axis
Both sum into the TMA axis so the AXIS total stays right, but the sub-reason is
wrong. The producer-parked-on-mbarrier cycles are precisely what Gate 3 needs to
read ("did collapse convert producer race-ahead into mbarrier-wait pacing?"). If
they hide inside named-barrier stalls, the timing story is unreadable. Note also
that `wait_barrier` today means the SASS control-bit WGMMA/DEPBAR wait
(`are_wait_barriers_ready`), a DIFFERENT mechanism — so we should not blindly
reuse it either.

Required change (gated, so feature-off stays bit-identical):
1. Add a dedicated query, e.g. `SM::warp_parked_on_mbarrier_spin(warp_id)`, that
   returns true only when the warp is parked via `m_pending_sync_waits` (the
   collapse path), distinct from named-barrier `m_barriers` waiting.
2. Add a dedicated stall flag/counter, e.g. `is_any_waiting_in_mbarrier_spin`
   (+ `total_num_warpcyc_stalled_mbarrier_spin` in the per-warp-cycle block near
   subcore.cc:835), and attribute the collapse park to it instead of
   `inst_barrier`.
3. Keep it inside the TMA axis grouping so axis totals are unchanged, but make the
   mbarrier-spin sub-reason separately visible.
4. Feature-off (`-spin_wait_collapse_enable 0`): `m_pending_sync_waits` stays
   always-invalid as today, so the new flag is never set and every existing stall
   counter is bit-identical.

This is the only stall-metric change collapse needs; the execution-count metrics
(§4c) require none.

### 6.5.1 The park query used off the issue path MUST be non-mutating (REQUIRED)

`SM::warp_waiting_at_barrier` has a SIDE EFFECT: when the mbarrier wait is
satisfied it clears `m_pending_sync_waits[warp_id]` and bumps a debug counter
(sm.cc:2253-2260). Today that clear never fires for an mbarrier (the slot is
always invalid), but under collapse it becomes a real state mutation — and
`shd_warp_t::waiting()` (which routes to it) is called from a `std::sort`
comparator `sort_warps_by_highest_id_dynamic_id` (subcore.cc:1472-1485, invoked
from BOTH `issue()` and `fetch()` every cycle).

Assessment (reviewed): this is NOT undefined behavior — within one sort the
barrier's satisfied-status is constant, so the comparator stays a valid strict-
weak-ordering, and the clear mutates SM state, not the sorted `shd_warp_t`
elements (same benign shape as the existing membar clear inside `waiting()` at
sm.cc:2709). BUT it IS a real attribution hazard: the authoritative "wait
released" (park clear + counter + log, and the §6.7 park/release balance the
debug logs rely on) would fire nondeterministically as a side effect of a
scheduling sort in fetch/issue, not from the deliberate issue-eligibility
decision.

Required rule: the clearing `warp_waiting_at_barrier` is called from EXACTLY ONE
authoritative site — the issue-eligibility check (subcore.cc:747,
`!c_warp->waiting()` on the non-tail path). Everywhere else that inspects park
state (the sort comparator, any read-only stall scan, the §6.5 stall
attribution) MUST use a NON-mutating peek:
- `SM::warp_parked_on_mbarrier_spin(warp_id) const` — returns
  `m_barriers.warp_waiting_at_barrier(warp_id) || (pending.valid &&
  !is_sync_wait_satisfied(pending))` with NO clear, NO counter bump.
- a `shd_warp_t::waiting_peek() const` (or `waiting(bool no_side_effects)`) that
  routes the barrier check through the peek; use it at subcore.cc:1475/1477.
This makes the comparator a pure function of immutable-during-sort state (removes
even the theoretical UB), and makes release deterministic and correctly
attributed. Only the remodeled SM path needs this; the baseline
`shader_core_ctx::warp_waiting_at_barrier` never consults `m_pending_sync_waits`.

## 6.6 Full metric audit (all instrument flags) and required fixes

A sweep of every `*_instrument_enable` / debug flag classifies each counter by how
collapse's two behaviors touch it: PARK (a warp is now ineligible on an mbarrier
via `m_pending_sync_waits`) and SKIP (spin polls are no longer issued/executed).
Verdicts:

| Instrument | Verdict | Root cause (file:line) |
|---|---|---|
| headofline_instrument_enable | **MUST-FIX (park)** | read-only re-scan `scan_head_of_line_when_blocked` (subcore.cc:424-502) does NOT consult `waiting()`, only `are_wait_barriers_ready` (SB dep-state). A warp parked on an mbarrier passes `warp_side_ready` and is counted in `n_ready` → reports a parked (non-issuable) warp as recoverable head-of-line. |
| overlap_instrument_enable | **MUST-FIX (park)** | pipe-overlap part UNAFFECTED (spin ops touch no tensor/SFU pipe), but the warpcyc sub-probe increments `total_num_warpcyc_stalled_any` (subcore.cc:871) for a parked warp while matching NONE of the reason checks → inflates the denominator with unattributed cycles. |
| mufu_lockstep_instrument_enable | **MUST-FIX (park)** | a parked PHASECHK stays a valid head (op=MBARRIER_OP) counted in `mufu_ls_valid_head`/`_other` (subcore.cc:813-825) and turns issue cycles into `still_idle` cycles that pass the finalize gate (subcore.cc:1077-1086). |
| producer_tail_instrument_enable | **MUST-FIX (park) + skip** | park bucketed `stall_inst_barrier` not `stall_wait_barrier` (sm.cc:1080 via subcore.cc:989); plus `issued_*`/`phasechk_*` tail loses collapsed polls. |
| cta_stall_breakdown_instrument_enable | SKIP (info-loss) | `m_wait_released_by_cta_slot`/`m_wait_pending_by_cta_slot` are per-poll (sm.cc:2377,2416); collapse removes intermediate/closing polls → undercount. |
| spin_instrument_enable | SKIP (by design) | counts issued spin winners (subcore.cc:1053); spins no longer issued → collapses toward 0. |
| producer_spin_ready_instrument_enable | SKIP (by design) | per-poll counters (sm.cc:824,857); `ready_run_retries` self-nullifies (it measures exactly the redundant re-polls collapse removes). |
| sync_wait_hist_instrument_enable | SKIP (array park-safe) | dedicated first-miss arrays are NOT touched by park (intended), but the closing ready poll is skipped so the episode never closes → long-wait buckets dropped (sm.cc:2384-2396). |

Two classes, two treatments:

- **MUST-FIX (park mis-attribution)** — headofline, overlap-warpcyc, mufu_lockstep,
  producer_tail stall bucket. These make a PARKED warp look READY or drop its
  cycles into an unlabeled/wrong bucket, which corrupts the very timing analysis
  collapse exists to produce. Fix pattern: add a read-only park predicate
  `SM::warp_parked_on_mbarrier_spin(warp_id)` (non-clearing peek) and (a) exclude
  parked warps from any "ready/valid-head/eligible" population, (b) attribute
  parked warp-cycles to the new `mbarrier_spin` bucket (§6.5) instead of `_any` /
  `inst_barrier`. All gated: with collapse off the predicate is always false and
  every counter is bit-identical.

- **SKIP (info-loss, mostly intended)** — cta_stall_breakdown, spin, producer_spin_
  ready, sync_wait_hist, producer_tail opcode tail. These lose the spin polls that
  collapse deliberately removes; that is the point, not a bug. Treatment: do NOT
  try to reconstruct the old per-poll counts. Instead, (a) document them as
  "meaningful with collapse OFF", and (b) where a duration signal is still wanted
  (sync_wait_hist), bucket `now - first_miss_cycle` at the PARK-RELEASE site
  (`warp_waiting_at_barrier` when satisfied) so wait duration survives even though
  the closing poll is skipped. The park itself is the more faithful signal than the
  old issued-poll count anyway.

## 6.7 Debugging logs to add (verify collapse actually works)

Collapse is a control-flow change on a 12h run, so it needs first-class,
low-volume, cyc-stamped logs (like the observe stage) to prove it behaves. Add
under a `-spin_wait_collapse_debug` sub-flag (or reuse `sync_debug_enable` budget),
default off, and keep them per-event with a budget so they cannot explode:

1. `[COLLAPSE][park]` — when a producer/consumer poll parks: `cyc sm cta warp
   wait_pc barrier arrive/expected tx/expected phase`. Confirms the block fires on
   the intended regions only.
2. `[COLLAPSE][release]` — when the barrier flips and the park clears: `cyc sm cta
   warp wait_pc barrier parked_cycles`. `parked_cycles` here is the REAL collapse
   park (contrast with the observe would-be park).
3. `[COLLAPSE][skip]` — when the cursor jumps to exit: `cyc sm warp wait_pc
   loop_head exit_pc skipped_insts used_insts_before/after`. The single most
   important correctness log: `skipped_insts` = poll iterations removed, and the
   post-jump head PC MUST equal `exit_pc` (assert + log).
4. `[COLLAPSE][summary]` per SM at kernel end: regions collapsed, total parks,
   total skipped polls, min/avg/max real park, and a reconciliation line
   `gpu_sim_insn_delta` = (baseline icount − collapsed icount) attributable to
   skips, so a run can be checked against Gate 1 in one grep.
5. Watchdog `[COLLAPSE][DEADLOCK]` (from §6) already covers the pathological case.

Cross-checks these logs enable on the same run:
- park count (log 1) vs release count (log 2) must balance (every park releases,
  else it is a §6 watchdog case).
- `skipped_insts` summed (log 3) vs the `gpu_sim_insn` drop vs baseline — must
  match, proving no semantic op was skipped.
- post-jump head == exit_pc on every skip — proves the B2 cursor walk is correct.
- new `mbarrier_spin` stall bucket (§6.5) should now be non-zero and the
  `inst_barrier` bucket should be ~unchanged from a named-barrier-only baseline.

## 7. Gating and bit-identity

- New flag `-spin_wait_collapse_enable` (default 0), separate from
  `-spin_wait_observe_enable`.
- Feature-off: `handle_sync_instruction` takes exactly today's path
  (nonblocking), `issued()` unchanged, no cursor edits → bit-identical.
- Feature-on but a region not collapsible → also unchanged.

## 8. Open questions for review — status

1. **B1 vs B2** — RESOLVED (§4): B2 (explicit dual-cursor skip), because poll
   occurrences carry a per-occurrence phase-parity `semantic_raw` that B1 would
   desync.
2. **exit_pc trust** — RESOLVED (§12.5): every collapsible exit_pc is a mid-stream
   unconditional BRA; never terminal, never at stream end (K5 8 / K10 6 checked).
3. **Multi-region producer** — RESOLVED: no collapsible region's exit_pc lands on
   another region's loop_head or inside another region's body (checked on the
   generated sidecar for K5 and K10). Each region exits to its own unique
   post-loop PC.
4. **Watchdog abort vs fallback** — DECIDED (§6, §12.5): hard abort for bring-up
   (surface model bug loudly); soft fallback only later, behind a sub-flag, and
   never masking a failed assert in v1.
5. **Consumer-side collapse** — RESOLVED (§9): collapse BOTH producer and consumer
   waits (FA3 handshake is symmetric); the runtime watchdog is the mutual-deadlock
   net. Gate is `collapsible` (role-agnostic).
6. **Skip-walk stop condition / landing** — RESOLVED (§4b): stop on `exit_pc`
   (not on "left body_pcs"), consume `body_pcs ∪ {loopback_bra_pc}`, land AT
   exit_pc and issue it so the existing taken-branch redirect reaches the semantic
   block.
7. **Park point** — RESOLVED (§5.1): arm at `primary_wait_pc`; issue the pre-poll
   trio once, hold ineligible on retry.

No open blockers remain for a v1 implementation. Remaining watch-items are
bring-up validations (feature-off bit-identity, park==release balance,
skipped-insts == gpu_sim_insn delta, post-jump head == exit_pc) covered by the
§6.7 debug logs.

## 9. Producer vs consumer waits — decision (resolved from observe logs)

FA3's handshake is bidirectional, so BOTH roles spin, and the sidecar detector
(pattern = PHASECHK/TRYWAIT + backward BRA) flagged BOTH without distinguishing
them. Confirmed on K5 SM27:

| Role | waits for | example | evidence |
|---|---|---|---|
| Producer (warp 0) | buffer **empty** (consumers finished reading) | barrier `0x16c40`, `expected_arrive=3`, warp 0 polls 473x `arrive=0/3` | only warp 0 does `arrive_expect_tx` (36x) |
| Consumer (warps 4-15) | buffer **full** (TMA landed) | barrier `0x16c00`, warps 4-15 poll | `0xace0` region, the 1371 false-open waits |

**Decision: first version collapses PRODUCER waits only.** Rationale:

- The symptom being fixed is "Producer ends ~61K cycles BEFORE Consumer" — i.e.
  the producer races ahead because its buffer-empty wait is under-enforced in the
  sim. Blocking the producer's wait directly targets that direction.
- Leaving consumer waits nonblocking keeps them exactly as today, so collapse
  cannot introduce a NEW deadlock from the consumer side — it shrinks the
  mutual-collapse blind spot from §9 of SPIN_WAIT_POSTPROCESS.md to zero for v1.
- Consumer-side collapse would matter only if the consumer raced ahead, which is
  the opposite of the current symptom.

**UPDATE (decided with the user): collapse BOTH producer and consumer waits.**
FA3 is a symmetric bidirectional handshake — both roles genuinely spin (verified:
K5 consumers warps 4-15 spin on `0x16c00`, producer warp 0 spins on `0x16c40`).
Collapsing only one side would model half the handshake and is architecturally
inconsistent. The mutual-deadlock risk this reintroduces is exactly what the §6
runtime watchdog covers (it turns any unbounded mutual wait into a surfaced model
bug instead of a silent hang). So v1 collapses any region with
`semantic_free_body && is_spin_loop`, regardless of role, and relies on the
watchdog as the safety net. Role tagging is therefore NOT required for v1; it is
kept only as optional diagnostic metadata.

**Required change:** none for role gating (both roles collapse). (Earlier this
section required producer/consumer classification for a producer-only v1; that is
no longer needed. If a future version wants role-selective collapse, classify via
the existing `mark_producer_warp` runtime mark.)

## 10. Scheduler guarantee — a parked warp does NOT stall the core (verified)

Collapse depends on: while warp A is parked at its wait, other warps on the same
subcore must keep issuing so the barrier A waits on can be flipped by them. This
is confirmed both statically and dynamically.

Static (subcore.cc issue scan): the per-cycle scan iterates
`priority_ordered_for_issue`; an ineligible warp is skipped and the scan tries the
next warp. The eligibility predicate `are_switch_warp_conditions_ready` includes
`is_not_warp_waiting_in_programmer_barrier = !c_warp->waiting()`, and
`waiting()` routes to `warp_waiting_at_barrier()` → `m_pending_sync_waits` — the
exact path collapse arms. The in-code comment is explicit: "the warp is simply
skipped this cycle and the scheduler tries another warp ... never the whole core."
All former `break`s on the scan were converted to `tail_readonly=true; continue`,
so a parked warp never halts the scan.

Dynamic (observe .e70, SM27): in the window cyc 5544-5586, warps
5,6,7,8,13,14,15,12,9,10,11,4 each issue on consecutive cycles; producer warp 0
and consumer warps 4-15 have fully overlapping active-cycle ranges (~2.5K-36K+).
So multiple warps interleave freely while others wait.

Caveat — subcore locality: the skip-and-try-next is WITHIN a subcore. A parked
producer is only rescued by warps in the SAME subcore. FA3's warp specialization
places producer + consumers together (SM27 hosts warp 0 producer and 4-15
consumers), so this holds here, but the collapse bring-up must re-confirm the
subcore assignment of the specialized warps (a producer alone in a subcore with
all-parked peers would idle that subcore — correct behavior, but worth logging).

## 12. Deeper risk re-review (non-obvious failure modes)

A second, adversarial pass looked for problems beyond the scheduler/cursor/metric
ones already covered. Findings, each checked against the actual trace/source:

### 12.1 Spin body arms SASS scoreboard dependency barriers — SAFE (self-tied)

The spin body is NOT pure poll: e.g. K5 `0xacc0 TRYWAIT` sets new_read_barrier id1
+ new_write_barrier id2, `0xace0 PHASECHK` sets new_write_barrier id2, `0xacb0
IMAD` waits mask=2 (id1) and writes R7. The sim arms these at issue
(subcore.cc:378-383) and releases at completion/writeback (sm.cc:1164-1170,
functional_unit.cc read-stage). Skipping means neither arm nor release runs.

Why it is safe: `m_wait_barriers` is one counter per id (READ/WRITE share it), and
every wait-barrier is armed AND released by the SAME instruction (write barrier at
its own WB, read barrier at its own read-stage). So skipping an instruction drops
its +1 arm and its -1 release together = net-zero on every id. Verified all K5/K10
hot regions are self-contained (armed_ids subset of waited_ids in the body) and
contain no async producer (TMA/WGMMA/LDGSTS/DEPBAR) whose release could straddle
the loop boundary — which is exactly what the `semantic_free_body` gate
guarantees. Post-exit blocks that appear to wait on a body id actually re-arm that
id themselves before waiting (WAR guard on the poll addr/reg), so they behave
identically with or without the skip.

### 12.2 Do not cancel the already-issued primary_wait poll — CRITICAL

The one poll that DID issue (the one that observed ready) must self-release its
own read/write barrier normally. If collapse tried to retroactively cancel it, the
pending decrement would either underflow (`assert(m_counter>0)`,
warp_dependency_state.cc:57) or leave an id permanently armed. Rule: collapse only
skips NOT-YET-ISSUED trace entries via the cursor; it never un-issues the poll in
flight. A cursor jump satisfies this by construction.

### 12.3 Register live-out of the poll — N/A in this trace-driven model

K10 poll writes P1 which is read post-exit by an IADD3 carry-in. In a functional
sim this would be a live-out hazard, but K5/K10 are `is_captured_from_binary=true`,
so (a) the register scoreboard (checkCollision) is NOT consulted at issue
(subcore.cc:456-471) and (b) register values are trace-replayed, never recomputed.
The consuming IADD3 has wait_mask=0. So skipping the poll cannot change any
modeled value or timing here. (Would need revisiting only if the model became
functional.)

### 12.4 Parked warp vs CTA-exit detection — SAFE

`check_if_warp_has_finished_executing_and_can_be_reclaim` gates on
`hardware_done() = functional_done() && stores_done() && !inst_in_pipeline()`, and
`functional_done()` needs `trace_done() = (used_insts == traced_pcs.size())`
(sm.cc:1413-1416, trace_driven.cc). A PARKED warp has `used_insts` stuck at the
poll, so `trace_done()=false` -> `hardware_done()=false` -> the CTA cannot exit
while a producer/consumer is parked. Correct. It also gates on
`!are_pending_dependencies()`, consistent with 12.1 (no dangling barrier).

### 12.5 Cursor over-run at skip — bounded; exit_pc is never terminal (RESOLVED)

If the skip walk failed to find `exit_pc` and ran to the stream end, `used_insts`
would reach `traced_pcs.size()` (silent early `trace_done`) and the next
`get_pc()` would trip `assert(used_insts < traced_pcs.size())`
(trace_driven.cc:172).

Checked statically: across ALL collapsible regions (K5 8, K10 6) `exit_pc` is
never terminal — every one is a mid-stream UNCONDITIONAL `BRA`, none is EXIT/RET,
none is the kernel's max pc, none is BRA-to-self; the next static instructions are
always real work or the next wait. So "the spin loop is the warp's last op" does
not occur here.

Therefore the bound is a pure safety net, and an **assert is the right choice for
bring-up** (`used_insts < traced_pcs.size()` during the walk, and post-jump head
== `exit_pc`): a violation means a mis-detected region, which we want to surface
loudly rather than hide. A soft fallback (skip the jump, replay normally) can be
added later once stable, but should not mask a failed assert in v1.

### 12.6 Subcore locality and slot reuse — both SAFE (RESOLVED)

- **Subcore locality:** warp→subcore assignment is round-robin `i % num_subcores`
  (sm.cc:1721; `num_subcores=4`), so subcore 0 = {warp0,4,8,12}, etc.
  `translate_warp_id_of_sm_to_subcore` (`warp_id / num_subcores`) is only the
  in-subcore local index. K5 SM27 has producer=warp0 and consumers=warp4..15, so
  each subcore holds warp0's-role + consumer roles together — a parked producer is
  rescued by consumers on the SAME subcore. No producer-only-subcore idle in this
  mapping. (Still worth a bring-up log if a future kernel packs a subcore with
  only producer warps.)
- **Slot reuse across CTA waves:** `m_pending_sync_waits` is cleared TWICE — at CTA
  exit `clear_sync_barrier_state_for_cta` (sm.cc:2586, clears entries whose
  `key.cta_id` matches) and at CTA (re)schedule `reinit` (sm.cc:1854, resets the
  warp slot). A stale `valid=true` cannot survive a wave boundary. Safe.

### 12.7 Non-self-tied barrier — not present now, but MUST be gated (design rule)

12.1 showed collapse is safe only because every wait-barrier armed in the body is
self-tied (armed and released by the same instruction) and no async producer
(GMMA/TMA/LDGSTS/DEPBAR) sits in the body. Audited across all 14 collapsible
regions (K5+K10): `waited_ids == armed_ids` in every body (ids {1,2}, or {2,3} for
one K10 region), and the only opcodes are PHASECHK/TRYWAIT/NANOSLEEP/IMAD/BRA — so
zero non-self-tied cases TODAY.

But a different kernel (or a future FA variant) COULD put a non-self-tied barrier
or an async producer in a spin body. If we blindly whole-skip such a body, the
scoreboard would be corrupted (an arm inside the body whose release is an
out-of-body async completion would be dropped, permanently blocking or wrongly
releasing a downstream consumer). This must not be left to chance.

**Design rule (enforce, do not assume):** collapse eligibility is
`is_spin_loop && semantic_free_body && barrier_self_tied`, where
`barrier_self_tied` is a NEW sidecar flag computed offline in
`build_spin_wait_sidecar.py`:
- FALSE if any body instruction has `wait_barrier_bits` on an id NOT armed inside
  the body (waited-on-externally-armed), OR
- FALSE if any body op is an async producer (`*GMMA`, `UTMA*`, `UBLK*`, `LDGSTS`,
  `LDG/STG/LDS/STS`, `ATOM/RED`, `DEPBAR/LDGDEPBAR`).
A region failing this is emitted with `collapsible=false` and the simulator leaves
it fully nonblocking (today's replay). The simulator ALSO asserts the flag at the
skip site as defense-in-depth, so a mislabeled region fails loudly instead of
silently corrupting the scoreboard. This turns "we happened to be safe" into "we
are safe by construction, and unsafe regions are auto-excluded."

### 12.8 Producer+consumer both-collapsed circular wait — precluded (RESOLVED)

> ⚠️ **PARTIALLY SUPERSEDED by §13.2/§13.3 (2026-08-12).** The "issue-independent
> TMA completion precludes a circular wait" argument holds **only for TMA-credited
> barriers** (`0x16c00`/`0x16c30`). It does **not** cover the **warp-arrived**
> barrier `0x16c20`, which has no TMA credit and flips only twice per kernel — that
> is exactly where K5 deadlocked. The depth-4 reasoning below is still correct for
> the buffer barriers; it simply did not apply to the low-flip cross-warp handshake.

The biggest un-verified risk of the §9 "collapse both roles" decision: could
parking BOTH the producer's buffer-empty wait AND the consumers' buffer-full wait
create a circular wait that did not exist under nonblocking replay? Analyzed
against the K5 SM27 cta0 SYNCDBG timeline; answer is NO, precluded by three
independent mechanisms:

Dependency structure (confirmed textbook warp-specialization): producer (warp 0)
only ARRIVEs on full barriers (fills via TMA) and only WAITs on empty barriers;
consumers (warpgroup leaders, expected_arrive=3) only WAIT on full and only ARRIVE
on empty. Full/empty barriers cycle 4-deep (full 0x16c30/38/50/58 ↔ empty
0x16c40/48/60/68).

Why no cycle:
1. TMA completion is issue-independent — `m_tma_unit_shared_of_sm->cycle()`
   (sm.cc:558) runs BEFORE `subcore->cycle()` (sm.cc:566) every cycle, so a
   producer's already-issued in-flight TMA keeps completing and flipping full
   barriers even while the producer warp is parked (observed: uid30@24652,
   uid31@24780 complete during the producer's 24499-27770 park).
2. Multi-buffer decoupling — the producer's reuse-wait frontier sits several
   buffers BEHIND its fill frontier, so the empty(m) it waits on is freed by
   consumers draining ALREADY-LANDED earlier buffers, never depending on the
   parked producer's next fill.
3. Collapse preserves trace ORDER (only timing changes) — the parked producer
   resumes at the same wait instruction and issues the same next UTMALDG once its
   empty frees.

The deadlock pattern {producer waits empty(k) ∧ every consumer waits full(k+1) ∧
producer stuck before k+1's TMA} never occurs in the timeline: during the whole
producer park no consumer is blocked on a full barrier whose TMA the producer has
not yet issued (they are computing on already-landed data). Log-wide: K5/K10 both
clean-exit, SPINWAITDEADLOCK=0, watchdog_hits=0, max would-be park 3,495 (K5) /
18,561 (K10) cyc — two orders under the 400K watchdog threshold.

Residual: only a degenerate depth-1 (effectively single-buffer) scheme could close
the cycle; FA3 is depth-4 so it cannot arise. Any future pathological trace/config
that did would be caught by the §6 runtime watchdog ([SPINWAITDEADLOCK] mid-run,
[SPINWAITOPEN] at exit), never a silent hang.

---

## 13. First COLLAPSE run results + K5 deadlock root cause (2026-08-12)

The first collapse-ON experiment (`-spin_wait_collapse_enable 1`,
`-spin_wait_collapse_debug 1`) was run on K5 and K10 simultaneously. **K10 passed;
K5 deadlocked.** This section is the self-contained handoff for the next session:
the log paths, the confirmed root cause, and the redefined direction. Everything
below is re-derived from the final logs (not the earlier live-streaming reads,
which were wrong — see §13.6).

> **Direction was redefined on 2026-08-12 (§13.4), and it SUPERSEDES the earlier
> design decisions in this doc (§6.1 "keep 1-bit parity", §6 watchdog framing,
> §12.8 "circular wait precluded").** The earlier proposal (generation-counter-as-
> release + watchdog force-release, so K5 would "complete" by skipping the missed
> flip) has been **removed**. Reasoning: a producer that misses a consumer phase
> edge is a **simulator-accuracy bug**, not something to paper over with a skip. If
> collapse were done correctly the producer/consumer handshake would line up
> exactly as it does on real (tracer-free) hardware, so a missed edge must be
> **surfaced as a model defect (assert), never force-released**. The generation
> counter is retained but repurposed as a **consistency auditor**, not a release
> mechanism. See §13.4.

### 13.1 Run outcome

| Kernel | stdout / stderr | Result | Final cycle | vs target |
|---|---|---|---:|---|
| **K10** bwd | `.o45` / `.e45` | ✅ clean `exit detected` | **135,352** | target 132,901 → **+1.84%** |
| **K5** fwd | `.o71` / `.e71` | ❌ `deadlock detected @ cyc 85333` | — (hang) | — |

- K10 collapse-ON 135,352 vs collapse-OFF (Opt 11-A) 136,898 = **−1,546 cyc
  (−1.13%)** → collapse moved K10 **toward** the HW target. Direction is correct.
- Full log dir base:
  `simulator-remodeled/sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/`
  - K5: `H100_80GB-OnlyKernel5/...-63a73d452237.{o71,e71}`
  - K10: `H100_80GB-OnlyKernel10/...-63a73d452237.{o45,e45}`
  - `[COLLAPSESUM]` (per-SM summary) is on **stdout** (`.o*`); per-event
    `[COLLAPSE][SMx] ... park/release+skip` is on **stderr** (`.e*`, K5 has 7,259).

### 13.2 Confirmed root cause — realized phase double-flip on a low-flip cross-warp barrier

The deadlock is **not** in the collapse bookkeeping (that is internally
consistent). It is a single poison park site, proven by set-equality:

- **The 21 SMs that parked at `wait_pc=0xb0b0` are EXACTLY the 21 deadlocked SMs**
  (`diff` of the two sets is empty). The GPGPU-Sim deadlock dump's stuck-core list
  (`6 8 9 16 23 24 25 27 ...`) equals the set of SMs that never printed a
  `[COLLAPSESUM]`.
- Park/release balance **by barrier** (all SMs, from `.e71`): every barrier is
  perfectly balanced **except one**:

  | barrier | parks | releases | dangling | how it is advanced (on SM6) |
  |---|---:|---:|---:|---|
  | 0x16c00 | 1370 | 1371 | 0 | **TMA-credited** (`tma_complete`, issue-independent) |
  | 0x16c30 | — | — | 0 | **TMA-credited** |
  | 0x16c40 | 979 | 979 | 0 | consumer `arrive` (warp-arrived) |
  | 0x16c48 | 907 | 907 | 0 | consumer `arrive` |
  | 0x16c60 | 228 | 228 | 0 | consumer `arrive` |
  | 0x16c68 | 126 | 126 | 0 | consumer `arrive` |
  | **0x16c20** | **29** | **8** | **21** | **warp15 `arrive` ONLY** (`tma_complete`=0) |

- **Two distinct wait sites share barrier `0x16c20`** (both `collapsible=True` in
  the sidecar):
  - dangling: `wait_pc=0xb0b0 exit_pc=0xb0d0` → **0/21 released** (poison)
  - successful: `wait_pc=0xb3d0 exit_pc=0xb3f0` → **8/8 released**
- The mechanism (`.e71`, barrier `0x16c20` only flips **twice** the whole kernel —
  warp15 arrives exactly 2×):
  - **Success (SM26):** producer reaches `0xb0b0` **while phase is still 1** (its
    prior generation) → **immediate-hit, never parks** → sails to `0xb3d0`, parks,
    and warp15's 2nd arrive (`@60782`) releases it. Clean 1:1.
  - **Dangling (SM6):** warp15 has **already done both arrives** (flip `0→1 @52212`,
    `1→0 @58477`) before the producer even polls `0xb0b0` (`@59786`). Phase is back
    to 0; the producer waits for `0→1`, but that flip **already passed** and no 3rd
    arrive exists → parks `@59821` **forever**. warp15 then blocks on downstream
    data the parked producer owes → circular wait closes → whole SM hangs.
  - A 1-bit parity cannot tell "my flip already passed" from "my flip has not
    happened yet", so the producer parks on an edge that will never recur.

### 13.2.1 Root cause quantified — this run only (`.e71`), poison vs success margin

**This is derived entirely from the collapse-ON run `.e71`.** The earlier
observe/collapse-OFF logs (`.o70`/`.e70`) are **not** a valid reference here: that
pass is observe-only, so the mbarrier does not drive timing at all and every wait
is walked past by construction — it is the first time mbarrier is actually in the
timing model that matters, and that is exactly this `.e71` run. The root cause must
be (and is) established inside `.e71`.

Splitting all 132 SMs by outcome (parked at `0xb0b0` = poison, immediate-hit at
`0xb0b0` = success) and measuring where the producer reaches `0xb0b0` relative to
the two flips of its own barrier:

| quantity (per SM, `.e71`) | POISON (21 SMs) | SUCCESS (111 SMs) |
|---|---|---|
| producer arrival − **gen2** flip | **+1 … +5,648** (always AFTER) | **−711 … −20,445** (always BEFORE) |
| gen1 flip cycle | med **50,697** (late) | med **45,779** (early) |
| gen2 − gen1 gap (the gen1 window) | med **7,085** (narrow) | med **13,474** (wide) |
| producer total collapse-park | med **40,055** | med **47,653** |

Reading:

- **Clean separator:** success iff the producer reaches `0xb0b0` **before gen2**;
  poison iff **after gen2**. SM16 fails by **+1 cycle** — this is a *timing-margin*
  fault, not a structural one.
- **It is NOT "poison producers park more".** The opposite is true (poison med
  40,055 < success med 47,653 collapse-park cycles). So the fault is not the
  absolute amount of collapse-injected park; it is the **relative** placement of
  the producer against the consumer-driven flip timeline.
- **The discriminating variable is the gen1 window width.** Poison SMs have gen1
  arriving *late* AND a *narrow* gen1→gen2 gap (~7K); the two consumer flips are
  bunched, so the window in which the producer's `0xb0b0` (a gen-1 wait) can land
  is short and the producer — paced by its own collapse parks on the empty barriers
  `0x16c40/48/68` — misses it. Success SMs open gen1 early and hold a wide (~13K)
  window.
- **Where the gen1/gen2 flip timing comes from:** `0x16c20` is flipped only by
  warp15's two arrives, and warp15 is itself gated between them by TMA/consumer
  barriers (`0x16c00/30/58`) whose timing is set by the modeled consumer-compute
  and TMA-fill cadence. So the accuracy fault that produces a too-narrow gen1
  window is in **that consumer/TMA pacing**, not in the collapse bookkeeping.

### 13.3 Why the design review missed it (corrections to earlier sections of this doc)

- **§12.8 "circular wait precluded (TMA completion is issue-independent)"** — true
  ONLY for TMA-credited barriers (0x16c00/0x16c30). `0x16c20` is **warp15-arrived**,
  so this safeguard does not apply to it.
- **§6.1 "double-flip precluded"** — assumed the parked barrier flips at most once
  between polls. A **cross-warp, low-flip barrier** (only 2 flips per kernel) can
  flip twice before the producer even reaches its poll — and §13.2.1 shows this is
  a *timing-margin* miss (poison SMs have a narrow ~7K gen1→gen2 window vs ~13K on
  success SMs), not a rare corner case: it happens on 21 of 132 CTAs.
- **Detector `collapsible` gate** (`is_spin_loop && semantic_free_body &&
  barrier_self_tied`) marked `0xb0b0` collapsible. `barrier_self_tied` is a
  **scoreboard** property; it does not capture **who advances the mbarrier** or
  **how few times it flips**. (The gate is produced by the post-processing detector;
  see SPIN_WAIT_POSTPROCESS.md.)
- **Watchdog never fired** (`COLLAPSEDEADLOCK=0`): threshold 400K ≫ K5's ~85K
  cycles, and GPGPU-Sim's global detector (114K window) fired first. Max
  *legitimate* park across all SMs is only **8,104 cyc** (from `[COLLAPSESUM]
  max_park`). The watchdog currently only **logs then `return`**
  ([sm.cc:2294-2319](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2294-L2319)).
  Note (per the redefined direction §13.4): the watchdog must **assert/abort**, not
  force-release — an unbounded park is an accuracy-bug signal to surface, not to
  paper over.

### 13.4 Redefined direction — collapse is an accuracy audit, generation is the auditor (2026-08-12)

**Framing (decided with the user).** The reason collapse exists is to remove the
*tracer-induced* spin count so the sim paces the producer/consumer handshake the
way tracer-free **hardware** does. It follows that **if collapse were done
correctly, the producer would reach each wait while its matching consumer edge is
still the current phase — exactly as on HW — and every wait would line up 1:1 with
a flip.** A producer that arrives *after* its edge already passed (the SM6 poison)
is therefore **evidence of a simulator-accuracy defect somewhere in the timing
model**, not a situation collapse should "recover" from by skipping.

This inverts the earlier proposal. The generation counter is **kept**, but its job
changes from *release mechanism* to *consistency auditor*:

- **Collapse honors the mbarrier for real.** At a collapsible `primary_wait_pc`:
  not-ready → park; ready → `skip_spin_to_exit`. (Unchanged from the current
  implementation; this is the correct, HW-faithful behavior.)
- **Generation counter audits the 1:1 pairing and ASSERTS on violation — it never
  force-releases.**
  - Add a monotonically-increasing `gen` per barrier: `gen++` on every phase flip
    in `recompute_sync_barrier_ready_and_maybe_flip_phase`
    ([sm.cc:2391-2421](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2391-L2421)),
    **never reset**. Keep the existing 1-bit `phase` untouched → Gate 0
    (feature-off bit-identity) preserved.
  - Each collapsible wait carries the **absolute generation it is supposed to pair
    with** (`expected_gen`; see the formula caveat below). At wait entry, compare
    `barrier.gen` to `expected_gen`:
    - `barrier.gen < expected_gen` → the edge has not happened yet → **park**
      (normal, correct wait).
    - `barrier.gen == expected_gen` → the edge just landed → **release + skip**
      (normal, correct release).
    - `barrier.gen > expected_gen` → **the producer's edge already passed** → this
      is the poison. **ASSERT/abort with a full diagnostic** (sm/cta/warp/wait_pc/
      barrier/`expected_gen`/`barrier.gen`/first-flip-cyc/now). Do **NOT** skip,
      do **NOT** park-forever. The run stops loudly at the exact accuracy fault.
- **Watchdog = assert, not force-release.** The `service_collapse_release`
  over-threshold path
  ([sm.cc:2294-2319](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2294-L2319))
  becomes a hard assert/abort as well (threshold right-sized to ~16–32K, above the
  8.1K legitimate max, below the 114K global detector). An unbounded park is, by
  the same logic, an accuracy defect to surface — never something to `skip` past.

**Why not just skip the missed edge (the rejected earlier plan).** Skipping makes
K5 "complete", but it launders an accuracy bug into a green run: the producer would
be advanced past a consumer edge it never actually observed, so the reported cycle
count would be a fiction. The user's explicit priority is correctness over a
passing run — "한번의 run으로 맞지 않더라도 반드시 풀어야 한다." The assert makes
the defect impossible to ignore and pins it to a coordinate.

**`expected_gen` formula — must be parity-aware, NOT `waits_completed + 1`.** The
earlier `target_gen = waits_completed + 1` is off-by-one for the producer's first
wait. Confirmed on the logs: the producer's very first wait on `0x16c20`
(`@7780`, parity=1) is an **immediate-hit at gen 0** (no flip yet), i.e. wait #1 ↔
gen 0, not flip #1. Because the barrier starts at `phase=0, gen=0` and keeps
`phase == gen & 1`, a parity-`p` wait is satisfied at the smallest `gen` with
`(gen & 1) != p`. So:

```
expected_gen(p) = smallest g >= (last consumed gen for this warp+barrier)+1
                  such that (g & 1) != p        # p = bit31 of wait_state_raw
# with the very first wait allowed to match g = 0 when phase already differs
```

This must be **validated offline** (no GPU) against the captured traces before any
coding: confirm, per barrier, that each warp's successive collapsible waits map to
strictly increasing generations and that the parity alternation holds. The CTA
structure is already confirmed identical (see §13.4.1), which is what makes an
absolute `expected_gen` well-defined.

**What the assert will most likely reveal (the real root cause to chase).** Per
§13.2.1 (this run, `.e71`), the discriminator is **not** the absolute amount of
producer collapse-park (poison SMs actually park *less*), but the **relative**
placement: poison SMs open gen1 late and keep a **narrow gen1→gen2 window (~7K
cyc)**, so the producer's `0xb0b0` gen-1 wait misses it. `0x16c20` is flipped only
by warp15's two arrives, and warp15 is gated between them by TMA/consumer barriers
(`0x16c00/30/58`) whose timing is set by modeled consumer-compute (WGMMA/MUFU) and
TMA-fill cadence. So the accuracy fault the assert pins is almost certainly **"the
modeled consumer/TMA pacing that sets the gen1→gen2 window is off relative to HW,"**
not the collapse bookkeeping. The assert coordinate (which barrier, `expected_gen`
vs `barrier.gen`, the gen1/gen2 flip cycles, first-not-ready cycle) is the lead for
that investigation.

### 13.4.1 CTA-consistency of the handshake (validated, all 132 CTAs)

The `0x16c20` handshake is structurally identical across every CTA, which is what
licenses an absolute-generation audit:

| property | value | scope |
|---|---|---|
| `EXCH` expected_arrive | `= 1` | 132/132 CTAs |
| flips per CTA on `0x16c20` | **exactly 2** | 132/132 SMs |
| arrive owner | **warp15 only** | 264/264 arrives |
| `wait_pc=0xb0b0` waits on | **parity 0** (→ 1st flip `0→1`) | 21/21 parks |
| `wait_pc=0xb3d0` waits on | **parity 1** (→ 2nd flip `1→0`) | 8/8 parks |

So `0xb0b0 ↔ generation 1` and `0xb3d0 ↔ generation 2` is a stable, per-CTA
invariant. The poison is purely a *timing* misorder (producer arrives at
generation 2 for a generation-1 wait), never a structural ambiguity. (The wait_pc↔
region mapping is emitted by the post-processing detector; see
SPIN_WAIT_POSTPROCESS.md.)

### 13.4.2 Root-cause investigation — the gen1→gen2 window is set by producer TMA-fill spacing (`.e71`, 2026-08-12)

Following the §13.4 lead, the accuracy fault was traced to a concrete
causal chain, all measured inside the collapse-ON run `.e71`.

**The failure is a race between two quantities, decomposed over all 132 SMs:**

| quantity (median) | POISON (21) | SUCCESS (111) |
|---|---|---|
| producer `0xb0b0` arrival − gen1 open | **+8,398** | (arrives inside window) |
| gen1→gen2 window width | **7,085** | **13,474** |
| producer arrival − gen2 (the verdict) | **+1,104 (miss)** | **−9,113 (hit)** |

So poison happens when **producer needs ~8.4K after gen1 to reach `0xb0b0`, but the
window is only ~7.1K** → it overshoots by ~1.3K. Both blades of the scissors move
the wrong way on poison SMs: producer arrives later (+8,443 vs success) **and** the
window is narrower (7,085 vs 13,474).

**What sets the window width — the key finding.** `0x16c20` is flipped only by
warp15's two arrives, and each warp15 arrive is gated by the **TMA-credited barrier
`0x16c58`** (`tx=16384/16384`, flips 8×/CTA — one per tile). The interval between
the two `0x16c58` flips that release warp15's gen1 and gen2 arrives tracks the
window almost exactly:

| | two-`0x16c58`-flip interval | resulting gen1→gen2 window |
|---|---|---|
| POISON | med **7,553** | med **7,085** |
| SUCCESS | med **12,140** | med **13,474** |

And each `0x16c58` flip is the TMA completion of a **producer `arrive_expect_tx`**
(confirmed on SM6: producer `arrive_expect_tx@56237` → TMA fill → `0x16c58`
flip@58132 → warp15 gen2 arrive@58477). Therefore:

> **The gen1→gen2 window width is the spacing between the two TMA fills the producer
> issues into `0x16c58`. That spacing is the producer's own execution pacing —
> which collapse changed by converting the producer's empty-barrier spins
> (`0x16c40/48/68`) into real parks.**

**Why this is a self-consistent (not paradoxical) loop.** The producer both (a)
feeds `0x16c58`, whose two fills define the window, and (b) must itself land inside
that window at `0xb0b0`. When collapse re-paces the producer, it perturbs BOTH the
window it creates AND its own arrival into that window. On the 21 poison SMs the two
perturbations land on the wrong side of the ~1.3K margin. This is exactly the class
of second-order scheduling interaction §9 warned observe could not rule out.

**What this means for the fix direction.** The audit (§13.4) is still correct: the
missed edge is a real accuracy defect and must assert, not skip. The *underlying*
inaccuracy to chase is **the producer's inter-fill pacing on `0x16c58` under
collapse** — i.e. whether the modeled latencies that space the producer's two TMA
fills (its parks on `0x16c40/48/68`, and the WGMMA/MUFU/TMA costs between them)
match HW. Open item: obtain the HW reference gen1→gen2 spacing (NCU / trace
timestamps) to decide whether the true window is ~7K (then the producer path is too
slow) or ~13K+ (then the window-narrowing pacing is the bug). The assert's
coordinate (gen1/gen2 flip cycles + first-not-ready) is exactly the per-SM data this
comparison needs.

### 13.5 Where the fix touches code (for next session)

Implementation is **audit-only** additions plus the assert; the collapse
park/skip logic itself is already correct and stays as-is.

- `tma_types.h` `HopperMBarrierObject` (~line 272): add `uint64_t gen = 0;`.
- `sm.cc::recompute_sync_barrier_ready_and_maybe_flip_phase` (2391-2421):
  `gen++` alongside `phase ^= 1` (never reset).
- `HopperMBarrierPendingWait` (tma_types.h): carry `expected_gen`.
- park-set branch in `handle_sync_instruction` (~2466-2534): compute
  `expected_gen` from a new per-`(warp,barrier)` generation-progress map (parity-
  aware formula, §13.4); then **audit**: `gen > expected_gen` → assert/abort with
  the diagnostic; `gen == expected_gen` → release+skip; `gen < expected_gen` →
  park. Do **not** replace the parity path used by the non-collapse issue logic
  (Gate 0).
- `sm.cc::service_collapse_release` (2285-2355): the over-threshold `else`/`return`
  (2294-2319) becomes an **assert/abort** (not `skip_spin_to_exit`), threshold
  lowered to ~16–32K.
- Add a `[COLLAPSEAUDIT]` diagnostic line (cyc-stamped) at the assert site so a
  future run's fault is greppable on the same timeline as `[SYNCDBG]`/`[TMADBG]`.

### 13.6 Corrections to the earlier (live-streaming) analysis — do NOT trust these

The mid-run notes were from partial logs and are **refuted** by the final logs:

- ❌ "20 unreleased parks" → ✅ actually **21**, all at `wait_pc=0xb0b0` on
  barrier `0x16c20`.
- ❌ "all collapse activity is warp=0 only; consumers never park" → ✅ consumers
  **do** park (park events show warp=12/13/14 on 0x16c00; 0x16c40/48/60/68 are
  consumer barriers with thousands of balanced parks). The dangling park happens
  to be producer warp0, but that is site-specific, not warp-role-specific.
- ❌ "watchdog too small" confusion → ✅ watchdog was too **large** (400K vs ~85K
  kernel) AND only logs, so it never fired.
- ❌ earlier proposed fix "exclude warp-arrived barriers" → ✅ wrong; see §13.4.
- ❌ earlier proposed fix "generation counter as edge-triggered **release** +
  watchdog **force-release**" (so K5 completes by skipping the missed edge) →
  ✅ **removed** on 2026-08-12. A missed edge is an accuracy defect to **assert**
  on, not to skip. The generation counter is kept only as the audit that raises
  that assert. See §13.4.

### 13.7 Quick re-verification commands (next session)

```bash
DIR=simulator-remodeled/sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1
K5E=$DIR/H100_80GB-OnlyKernel5/*-63a73d452237.e71   # collapse-ON run — the ONLY valid reference
# park/release balance per barrier (expect only 0x16c20 imbalanced):
grep -aoE "cyc=[0-9]+ park .*barrier=0x[0-9a-f]+ "    $K5E | grep -oE "barrier=0x[0-9a-f]+" | sort | uniq -c
grep -aoE "cyc=[0-9]+ release.*barrier=0x[0-9a-f]+ "  $K5E | grep -oE "barrier=0x[0-9a-f]+" | sort | uniq -c
# the two wait sites sharing 0x16c20 (0xb0b0 poison vs 0xb3d0 ok):
grep -aoE "park .*barrier=0x16c20 wait_pc=0x[0-9a-f]+ exit_pc=0x[0-9a-f]+" $K5E | sed -E 's/warp=[0-9]+ cta=[0-9]+ //' | sort | uniq -c
# ROOT CAUSE (§13.2.1) — poison=producer reaches 0xb0b0 AFTER gen2; success=BEFORE gen2.
# On a poison SM (e.g. SM6): the two 0x16c20 flips (gen1,gen2) both precede the 0xb0b0 park:
grep -aE "\[SM6\].*phase flip barrier=0x16c20" $K5E          # gen1 @52212 (0->1), gen2 @58477 (1->0)
grep -aE "\[SM6\] cyc=[0-9]+ park warp=0 cta=0 barrier=0x16c20 wait_pc=0xb0b0" $K5E   # @59821 (> gen2)
# On a success SM (e.g. SM26): producer immediate-hits 0xb0b0 BEFORE gen2:
grep -aE "\[SM26\].*wait immediate-hit warp=0 cta=0 barrier=0x16c20 .*parity=0" $K5E | head -1  # < gen2
# NOTE: do NOT use .o70/.e70 (observe/collapse-OFF) as a baseline for this wait —
# that pass does not put mbarrier in the timing model, so walking past the wait there
# is by construction, not evidence about the collapse-ON handshake.
```

Sidecar of record (both wait sites are `collapsible=True`, `barrier_self_tied=True`;
produced by the post-processing detector, see SPIN_WAIT_POSTPROCESS.md):
`simulator-remodeled/hw_run/traces/device-0/12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1_lighter-compact_wire+prune_operands-69a20aed/traces/extra_info/spin_wait_sidecar.json`
(schema v2, K5 = kernel_id 5, funcid 3, 21 spin_regions; the poison region is
`region_id=k_fid3_h0xb090`, `primary_wait_pc=0xb0b0`, only **2** dynamic flips of
its barrier per kernel).

### 13.8 HANDOVER — NCU cross-check on another server (self-contained)

This is the next action item and can be done independently on a GPU/NCU server.
Everything needed to run it and to interpret it is here; no prior context required.

**The one question to answer.** On real hardware, what is the elapsed time between
the two arrives on the K5 coarse barrier `0x16c20` (equivalently: the spacing of the
two producer TMA fills into `0x16c58` that gate them)? Call it the **HW gen1→gen2
window**. The simulator (`.e71`, collapse-ON) produces:

- window on the 111 **success** CTAs: median **~13,474 cyc** (range 6,849–22,508)
- window on the 21 **poison** CTAs: median **~7,085 cyc** (range 5,578–9,042)

**Decision rule (what the HW number tells us):**

| HW gen1→gen2 window | Verdict | Meaning / next fix |
|---|---|---|
| **≈ 13K+** (matches success) | window-narrowing is the bug | The sim is *shrinking* the window on 21 CTAs. Chase why collapse re-pacing bunches the producer's two `0x16c58` TMA fills too close — the parks on empty barriers `0x16c40/48/68` and/or the TMA-fill latency model. Fix the pacing so the window stays ~13K; the producer then lands inside it and no edge is missed. |
| **≈ 7K** (matches poison) | producer path is too slow | The true window really is narrow; then the accuracy fault is that the **producer takes ~8.4K after gen1 to reach `0xb0b0`** when HW takes < window. Chase the producer's inter-fill latency (WGMMA/MUFU/TMA between the two fills) being too high under collapse. |
| **bimodal / tile-dependent** | both regimes are real | Some tiles legitimately have a short window (e.g. causal-mask tail tiles do less work). Then the audit assert must be reconciled with which tile-index each CTA is on; record tile index alongside the assert coordinate. |

**Why NCU is needed (sim cannot self-answer this).** The sim's own window is the
very quantity under suspicion (collapse perturbs it), so it cannot be its own
reference. Only HW/NCU gives the ground-truth spacing.

**What to pull from NCU / HW trace (concrete):**

- Kernel: **K5** = the FA3-bwd **forward-of-backward** kernel, `kernel_id 5`,
  `unique_function_id 3` (BF16, bwd, causal, `b1-s2048-hd64-nh24`). Same launch as
  the trace dir named below.
- Barrier addresses are shared-memory offsets; on HW you cannot grep `0x16c20`
  directly, so anchor on the **SASS PCs** instead (stable across HW/sim):
  - producer wait `wait_pc=0xb0b0` (region `k_fid3_h0xb090`, operand
    `[R16+URZ+0x16820]`) ↔ generation 1 wait.
  - producer wait `wait_pc=0xb3d0` (region `k_fid3_h0xb3b0`, operand
    `[R15+URZ+0x16820]`) ↔ generation 2 wait. (Both wait operands resolve at
    runtime to barrier `0x16c20`; the `+0x16820` is the static base+offset text.)
  - the two producer `arrive_expect_tx` that fill `0x16c58` are the events that
    bound the window. Get their PCs from the sim log directly — grep
    `arrive_expect_tx warp=0 .*barrier=0x16c58` in `.e71` and read back the
    `wait_pc`/site, or cross-map via `sync_operand_resolver.json` (funcid 3) by the
    resolved runtime `barrier_addr=0x16c58`, not by a static offset guess.
- Preferred HW signal, in order of usefulness:
  1. **NCU `SourceCounters` / warp-state sampling** on those PCs to get cycles-between
     the two producer TMA-fill arrives (the window) per scheduler.
  2. If per-arrive timing is not resolvable, use **`smsp__average_warps_issue_stalled_barrier`**
     and the **`long_scoreboard`/`tma`** stall spans around the producer warp to bound
     the inter-fill interval.
  3. Coarse fallback: the NCU `NANOSLEEP` totals already on record
     (K5 HW `NANOSLEEP` = **8,461**) confirm the producer *does* spin-wait on HW,
     i.e. the window is not zero; but this alone does not give its width.
- Also record **whether the window depends on tile index** (causal → later tiles do
  less work). If yes, the two regimes above are both physical and the sim must not
  assume a single window.

**Sim-side numbers to compare against (already measured, `.e71`):**

- producer `0xb0b0` arrival − gen1 open: poison med **+8,398** (this is the producer
  path latency after its window opens — compare to HW producer path).
- two-`0x16c58`-flip interval: poison med **7,553** / success med **12,140** (this is
  the sim window generator — compare to HW TMA-fill spacing).
- 21 poison CTAs miss by only ~1.3K median → a **small** pacing correction may
  suffice; this is not a gross error.

**How to regenerate the sim evidence (CPU-only, if the `.e71` logs are gone).**
Re-run K5 with `-spin_wait_collapse_enable 1 -spin_wait_collapse_debug 1
-sync_debug_enable 1` (K5 hangs ~85K cyc, so the global deadlock detector ends it;
that is expected and fine for harvesting the pre-hang timeline). Then the §13.7
commands reproduce every number above from the new `.e71`. The trace dir is:
`simulator-remodeled/hw_run/traces/device-0/12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1_lighter-compact_wire+prune_operands-69a20aed/`.

**Definition of done for this handover step.** A single line recorded back into this
doc: "HW gen1→gen2 window = N cyc (method M)", plus which row of the decision table
it selects. That determines whether the follow-up fix targets the **window
generator** (TMA-fill spacing) or the **producer path latency** — and only then does
the §13.5 audit implementation get paired with the correct pacing fix.

### 13.8.1 NCU cross-check attempt — RESULT: NOT MEASURABLE FROM NCU (2026-08-12)

**HW gen1→gen2 window = UNKNOWN (method: not resolvable from NCU on this GPU/CUDA
setup).** The §13.8 handover **cannot be closed with NCU**. No row of the decision
table is selected. Two NCU experiments were run/inspected on an H100 (device 2,
CUDA 12.8, `ncu`), FA3 K5 = `FlashAttnFwdSm90`:

**Experiment 1 — existing reports** (`nv_reports/h100/flashattn-..._full_rpt.ncu-rep`,
`fa3_pipe_rawcounts.ncu-rep`). The two producer wait PCs were positively located by
operand in the K5 SASS source page:
- gen1 wait `[R16+URZ+0x16820]` (parity P4) → `@!P4 SYNCS.PHASECHK.TRANS64 P4,[R16+URZ+0x16820]`
- gen2 wait `[R15+URZ+0x16820]` (parity P0) → `@!P0 SYNCS.PHASECHK.TRANS64 P0,[R15+URZ+0x16820]`
Their warp-stall samples were ≈0; kernel-wide only **3,942** PC samples. No signal.

**Experiment 2 — targeted high-density warp sampling** (K5 only:
`--launch-skip 4 --launch-count 1`, `--section WarpStateStats --section
SourceCounters`, `--warp-sampling-interval 0 --warp-sampling-max-passes 30`; report
`nv_reports/h100/fa3_k5_warpsampling.ncu-rep`, script
`simulator-remodeled/fa3_k5_warpsampling_window.sh`). Sample count rose 8× to
**31,495** (buffer_overflow=0, dropped=0, kernel 68,364 cyc @ 1.98 GHz). Even so:
- The producer spin's PHASECHK PCs still carry ~0 samples; the warp actually stalls
  in the following **NANOSLEEP** (`0x989680` = 10,000,000 ns backoff), and those
  NANOSLEEP samples classify mostly as **`long_scoreboard`**, not `barrier`.
- Whole-kernel NANOSLEEP (21 PCs) = **88 samples (0.3%)**; PHASECHK = 335 (1.1%).
  The two target wait loops carry only tens of samples.

**Why NCU fundamentally cannot answer this (confirmed, not a density problem).**
§13.8 asks for the *elapsed cycles between two specific events* (the two
`0x16c58`-gating arrives on `0x16c20`, which flips exactly 2×/CTA). NCU standard
profiling stores only **per-kernel aggregates** and **cumulative PC-stall sample
counts** — there is **no timestamp/trace axis**, so an instant-to-instant interval
of a **low-frequency event (2 flips/CTA)** cannot be recovered. The 8× density
experiment proves the limit is the *data type* (statistical, no time axis), not the
sample rate. The §13.8 fallback (bound the interval via `stall_barrier` span) also
fails: the producer's stall is `long_scoreboard`/NANOSLEEP-backoff, not `barrier`,
and is still per-PC-cumulative rather than per-episode.

**Consequence for the plan.**
- The §13.4/§13.5 **audit (generation counter + assert)** does NOT depend on the HW
  window — implement it as specified; it pins the poison deterministically.
- The *pairing* pacing fix (§13.8 decision table: window-generator vs producer-path)
  stays **blocked on a timestamped HW measurement**. NCU is ruled out. The only
  remaining ways to get a per-CTA window are: (a) **in-kernel `clock64()`
  instrumentation** of the two `0x16c58` producer arrives — but flash-attn here is a
  **pip-installed prebuilt `flash_attn_3` `.so`** (no kernel source in the repo, no
  recompile path), so this needs an external flash_attn_3 source rebuild; or
  (b) **Nsight Systems / CUPTI SASS-level trace** (unproven for an event this
  low-frequency). Sim-side reference for the eventual compare (from `.e71`): success
  window med ~13,474 cyc, poison med ~7,085 cyc, poison miss margin ~1.3K.