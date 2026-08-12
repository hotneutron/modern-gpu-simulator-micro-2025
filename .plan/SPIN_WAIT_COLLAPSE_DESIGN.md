# Producer Spin-Wait Collapse — Implementation Design (for review)

Status: DESIGN REVIEW. Observe stage returned GO (see SPIN_WAIT_SIDECAR.md §11).
This document is for design review BEFORE writing the collapse feature. Nothing
here is implemented yet.

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
2nd-order scheduling) called out in §9 of SPIN_WAIT_SIDECAR.md.

### 6.1 Phase-parity double-flip — keep the 1-bit model (reviewed, not a hazard)

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
  mutual-collapse blind spot from §9 of SPIN_WAIT_SIDECAR.md to zero for v1.
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
