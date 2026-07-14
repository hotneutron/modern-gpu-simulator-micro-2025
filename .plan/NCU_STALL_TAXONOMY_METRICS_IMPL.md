# Implementation Plan — NCU-aligned stall-taxonomy metrics (FA3 fwd 2× diagnosis)

> Concrete build plan for the [METRICS_Add_Fix.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.plan/METRICS_Add_Fix.md) §2 worklist (NCU stall-taxonomy alignment).
> Upstream prerequisite framing: [CTA_SAMPLING.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.plan/CTA_SAMPLING.md) §7.1 (P11/R12). Diagnosis tracked in
> [FA3_progress.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.result/FA3_progress.md) Ongoing item 3.

## Context

FA3 fwd is stuck at **2.02× HW** (137,053 vs 67,696 cycles). The diagnostic reduced this to a single
fact: the sim issues warp-instructions at **half** HW's rate (0.79 vs 1.63 issued warp-inst/cycle/SM =
2.07× = the whole cycle gap) with the **same warp count** — so HW overlaps work the sim serializes.
Two structural suspects (tracked in `.result/FA3_progress.md` Ongoing item 3):

- **Suspect #1 — WGMMA modeled synchronously**, not async (`TENSOR_CORE_OP` uses the fixed-latency FU
  reserve path, [subcore.cc:288-327](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L288-L327); contrast TMA which IS async, [sm.cc:1519](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1519)).
- **Suspect #2 — single-issue per scheduler** (loop `break`s after one issue, [subcore.cc:635-638](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L635-L638)).

**The reason we cannot size these today is that the sim's stall taxonomy has no counter for either** —
NCU exposes `warpgroup_arrive` (→ #1) and `dispatch_stall`/`not_selected` (→ #2), but the sim folds
them into other buckets and never classifies stalls on cycles that *did* issue. This plan builds the
NCU-aligned counters (worklist: [METRICS_Add_Fix.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.plan/METRICS_Add_Fix.md) §2) so we can **measure** which suspect
dominates before touching any timing model.

**Intended outcome:** an NCU-shaped per-reason stall breakdown emitted on the existing fwd `.o37` / bwd
`.o20` runs, decomposing the 2× issue-rate deficit into `warpgroup_arrive` (suspect #1) vs
`not_selected`+`dispatch_stall` (suspect #2), with a hard bit-identity guarantee.

**Decisions (user):** implement **both phases together**; counters are **always-on** (no config gate).
Since always-on removes any off-switch, **side-effect-freedom of the every-cycle pass is the paramount
constraint** — the sim's `gpu_sim_cycle` MUST stay bit-identical.

## The invariant: every counter touches 3 sites

Confirmed by source exploration — there is **no lazy auto-create**; a missing map key default-constructs
a null `shared_ptr` and the `->increment_with_integer` dereference **crashes**. So each new counter
requires all three:

1. **Register** in `gpgpu_sim::create_gpu_per_sm_stats()` — [gpu-sim.cc:2639-2804](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L2639) — via
   `m_gpu_per_sm_stats.add_unsigned_long_long_stat("key", ...)`. This seeds every per-SM `m_sm_stats`
   copy ([sm.cc:463-480](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L463-L480)).
2. **Increment** in `Subcore::issue()` — [subcore.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc) (sites per-counter below).
3. **Print + fold** in `shader_core_stats::print_remodeling_stats()` — [shader.cc:1230-1582](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L1230) (explicit
   per-key `get_value()` + `fprintf`; grouped folds at [shader.cc:1523-1546](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L1523-L1546); percentage denom =
   `total_num_cycles_issue_stage_evaluated`).

All counters are observe-only (read existing per-cycle predicates), so the whole change is
**timing-neutral / bit-identical** by construction.

---

## Phase 1 — counters that need NO second pass (cheap, low-risk)

Each row = register + increment + print. No change to the issue loop control flow.

| Counter | Increment site + predicate | NCU reason |
|---|---|---|
| **`selected`** | Emit `total_num_cycles_issue_stage_selected` from the existing `is_issued_inst` branch ([subcore.cc:715-716](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L715-L716)). Document it as the per-issue denominator. | `selected` (13.9%) |
| **`dispatch_stall`** | Re-derive from the existing `issue_port_busy` path ([subcore.cc:719-720](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L719-L720), set by `set_num_pending_cycles_with_issue_port_busy` at [:629](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L629)) **plus** the `is_write_available_result_queue_for_fixed_latency_available == false` case. Emit `total_num_cycles_issue_stage_stall_dispatch`. | `dispatch_stall` (11.0%) |
| **`warpgroup_arrive`** ⭐ (sizes #1) | At the `!are_wait_barriers_ready` stall site ([subcore.cc:685](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L685)), split the WGMMA-group wait from a plain mbarrier wait by string-matching the head opcode: `pI->get_extra_trace_instruction_info().get_op_code()` contains `"WARPGROUP"` (reuse the existing classifier `is_warpgroup_tensor_opcode()`, [traced_instruction.cc:54-61](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/traces_enhanced/src/traced_instruction.cc#L54-L61), and add a `.DEPBAR`/`.ARRIVE` match). Emit `total_num_cycles_issue_stage_stall_warpgroup_arrive`. | `warpgroup_arrive` (5.7% bwd) |
| **`long_scoreboard` fold** | Grouped print = `wait_barrier + tma_flush + global-load RAW`. Verify global-load (non-const) RAW is counted (`waiting_scoreboard` today is traditional RAW only, [subcore.cc:562-563](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L562-L563)); if missing, note as a gap, do not fabricate. Add to the fold block [shader.cc:1523-1546](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L1523-L1546). | `long_scoreboard` |
| **`short_scoreboard` fold** | Grouped print = `waiting_scoreboard + waiting_l1c`. | `short_scoreboard` |
| **`mio_throttle` map** | Map `waiting_result_queue_full` → NCU `mio_throttle` (rename/alias in print only). | `mio_throttle` |
| **`sleeping`** | Cannot be distinguished from `yield` in trace mode (NANOSLEEP = `MISCELLANEOUS_NO_QUEUE_OP`, no duration in trace; yield path is control-bit-driven, [warp_dependency_state.cc:95-116](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/warp_dependency_state.cc#L95-L116)). **Decision: fold into the yield bucket** and document; do NOT invent a sleep timer (would break "no fabricated timing"). | `sleeping` (residual) |

**NCU scheduler scalars** (Phase 1, print-only, from counters we now have):
`issued_warp_per_scheduler` = `selected`/evaluated; `issue_slots_busy_pct`; `no_eligible_pct` /
`one_or_more_eligible_pct` (the last two need the eligible count from Phase 2 — emit them there).

---

## Phase 1b — FULL 20-reason coverage (goal: every NCU stall reason has a sim counterpart)

> **User goal (2026-07-14):** know **every** NCU stall reason in the sim and compare 1:1 with HW, not
> just the newly-added ones. Phase 1/2 add the *missing* counters; this table closes the taxonomy so a
> complete stall stack is emitted. Shares are kernel-10 (bwd) NCU aggregate (CTA_SAMPLING §7.1 spike);
> fwd shares differ but the reason set is the same.

Three classes: **NEW** (Phase 1/2 above), **MAP** (existing counter, just needs an NCU-named print line
+ documented fold), **RESIDUAL/NEGLIGIBLE** (≈0% in FA3, bucket to keep the sum honest).

| NCU reason | k10 % | class | sim source → action |
|---|---:|---|---|
| `long_scoreboard` | 21.9% | fold | `wait_barrier + tma_flush + global-load RAW` (Phase 1). Verify global-load RAW counted. |
| `barrier` | 17.2% | **MAP (add print)** | existing `waiting_inst_barrier` (named BAR/LDGDEPBAR). Emit NCU-named line; disambiguate vs `warpgroup_arrive`. |
| `selected` | 13.1% | NEW | Phase 1 denominator. |
| `wait` | 10.4% | **MAP (add print)** | existing `waiting_stall_count` (fixed-latency dep). Direct 1:1 print. |
| `short_scoreboard` | 8.6% | fold | `waiting_scoreboard + waiting_l1c` (Phase 1). |
| `mio_throttle` | 5.9% | map | `waiting_result_queue_full` (Phase 1). |
| `warpgroup_arrive` | 5.7% | NEW | Phase 1 (opcode-string split). |
| `not_selected` | 5.4% | NEW | Phase 2 (tail read-only). |
| `dispatch_stall` | 4.5% | NEW | Phase 1 (issue-port + RF-queue). |
| `sleeping` | 3.1% | NEW/fold | Phase 1 (fold into yield; documented trace-mode limit). |
| `math_pipe_throttle` | 1.3% | **MAP (add print)** | existing `with_fu_occupied_sfu` + `..._sp_int_dp` (non-tensor math pipe busy). Emit NCU-named line. |
| `no_instructions` | 1.2% | **MAP (fold+print)** | existing `no_valid_instruction_*` frontend sub-tree (L0I/ibuffer/decode family, [subcore.cc:721-735](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L721-L735)). Fold whole sub-tree → one NCU-named line. |
| `imc_miss` | 0.9% | **MAP (add print)** | existing `waiting_l1c` (const-cache miss). Resolve overlap w/ `short_scoreboard`: put const *miss* here, const *latency* in short_sb. |
| `branch_resolving` | 0.8% | **RESIDUAL** | no branch unit in trace-driven sim → **explicit `branch_resolving` line = 0** + a note it maps to residual (do NOT fabricate). |
| `mma` (build-dependent) | — | map | `with_fu_occupied_tensor` (tensor-pipe busy). Emit if present. |
| `drain` | 0.0% | negligible | partial `tma_flush`/`yield`; bucket into residual. |
| `lg_throttle` | 0.0% | negligible | partial `result_queue_full`; residual. |
| `membar` | 0.0% | negligible | scope-only membar handled in `sm.cc`; print 0 + note. |
| `misc` | 0.0% | residual | catch-all line. |
| `tex_throttle` | 0.0% | n/a | no texture in FA3; print 0 + note. |

**Deliverable:** `print_remodeling_stats()` emits a **complete NCU-named stall stack** — every row above
as `ncu_stall_<reason> = <count>` + `pct`, so the sim's stack lines up 1:1 with the NCU stall page for
both fwd and bwd. MAP rows are print-only (no new increment; the underlying counter already exists), so
they add **zero** timing risk. The RESIDUAL/negligible rows are emitted as explicit 0-lines with a note
(not silently omitted) so a reader can confirm the sum is complete.

**Fold-accounting rule (avoid double counting).** Because the per-reason `is_any_waiting_*` flags are a
per-cycle boolean-OR (≥1 warp), reasons overlap and can sum to more than 100% ([subcore.cc:738-741](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L738-L741)).
Emit the NCU-named stack **both** ways: (a) the raw per-reason lines (overlapping, matches how NCU's
own reasons overlap), and (b) note the overlap in the header so the comparison is shape-vs-shape, not a
forced 100% partition (R5 coarse-axis fallback if a strict partition is later required).

---

## Phase 2 — tail-only read-only completion of the SAME loop (`not_selected` + `eligible_warps_per_scheduler`, sizes #2)

> **User decision (2026-07-14): NOT a separate every-cycle full-warp scan** — that re-evaluates warps
> the primary loop already covered and is wasteful on the 12h run. Instead, **continue the existing
> primary loop past the winner in read-only mode** (do not add a second full pass). We DO need the exact
> `not_selected` count (goal = match NCU), so we keep counting exactly, just without the redundant scan.

**Why the loop must be *extended*, not *re-run*.** The primary loop has **TWO** early exits, not one:
- the `break` at [subcore.cc:638](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L638) — the winner issued;
- the `break` at [subcore.cc:703](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L703) — the **greedy** warp is blocked *only* on the const-cache
  (`is_l1c_ready` false) and is NOT allowed to yield to a lower-priority warp, so the loop stops with
  **no issue** (`is_issued_inst == false`).

Split the priority-ordered warp list (`order_greedy_then_highest_id`, [subcore.cc:439](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L439)) into three
zones relative to **wherever the loop stopped** (either break, OR natural end):

| zone (priority order) | already evaluated today? | `not_selected` contribution |
|---|---|---|
| **before the stop** | YES — loop didn't stop on them ⇒ they were NOT eligible | 0 (by definition not eligible) |
| **the stopping warp** | YES — either issued (`:638` winner → `selected`) or const-cache-blocked (`:703` greedy → NOT eligible, belongs to `short_scoreboard`/`imc_miss`, NOT `not_selected`) | 0 |
| **after the stop** | **NO — either `break` skips them** | **exactly this zone = `not_selected`** |

So the missing information is the **tail after whichever exit fired**. Key correction from the first
draft: the tail trigger is **"the loop exited early"** (both breaks), NOT "a warp issued". On a
**natural-end** cycle (no break — every warp examined and none issued) the loop already walked all warps,
so `not_selected`/`eligible` there is free. Only when a break fires do we need the read-only tail.

**The hard constraint (always-on ⇒ must be truly side-effect-free).** The primary eligibility predicate
calls 3 **mutating** functions that must NOT be re-invoked:

| mutator in predicate | why it mutates | read-only substitute |
|---|---|---|
| `c_warp->waiting()` ([shader.cc:4853](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L4853)) | clears membar state, erases watchdog maps, may invalidate L1 | `SM::warp_waiting_at_barrier(warp_id)` (const, [sm.cc:1547](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1547)) + read-only membar-pending-stores query (`warp_has_pending_fence_stores` / const `get_membar()`) |
| `are_l1c_operands_ready()` ([subcore.cc:953](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L953)) | issues a real L0-const-cache `access()` | **omit** from the eligibility mirror (const-cache readiness is a greedy-switch tolerance nuance, [subcore.cc:613-622](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L613-L622), not part of `are_switch_warp_conditions_ready`). Document that `not_selected` excludes it. |
| `warp_waiting_at_tma_flush()` ([sm.cc:2011](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2011)) | insert/erases a log-tracking set (not timing, but non-const) | `warp_has_outstanding_stores()` (read-only; the value it derives from) |

All other predicate calls are read-only (scoreboard `checkCollision_remodeling` const, `is_stall_counter_0`,
`is_wait_barriers_ready_entry_point`, `is_yield_ready`, `is_waiting_ldgdepbar`, `can_issue`,
`is_fixed_latency_unit`, RF-queue-space, `IBuffer::is_next_valid`/`next_inst` — no pop).

**Design — replace BOTH breaks with a read-only tail continuation (single pass, exact count):**

Do NOT add a second loop. Keep the primary loop as-is up to and including the stopping warp (full
predicate, side effects intact — unchanged behavior). At **each** point where it would `break`, instead
set `tail_readonly = true` and `continue`; the SAME loop then finishes the remaining warps with the
read-only mirror only.

```
// subcore.cc — read-only eligibility mirror (no mutating calls).
// Mirrors are_switch_warp_conditions_ready (subcore.cc:606-609) using ONLY read-only
// calls + the 3 substitutes above; excludes is_l1c_ready by design (documented).
bool Subcore::is_warp_eligible_readonly(SM* sm, shd_warp_t* w,
                                        warp_inst_t* pI, unsigned subcore_wid) const;

// Inside the EXISTING primary loop (order_greedy_then_highest_id):
//
// exit 1 — winner issued (subcore.cc:638):
//   if (is_inst_ready_to_issue) {
//       issue_warp(...); is_issued_inst = true; m_greedy_pointer_issue = subcore_warp_id;
//       tail_readonly = true; continue;          // was: break;
//   }
//
// exit 2 — greedy warp const-cache-blocked (subcore.cc:703):
//   } else {                                     // greedy warp, !is_l1c_ready, cannot yield
//       tail_readonly = true; continue;          // was: break;  (is_issued_inst stays false)
//   }
//
// At loop TOP, when tail_readonly is set, skip the full predicate and use the mirror:
//   if (tail_readonly) {
//       if (!c_warp || c_warp->done_exit() || !ibuffer_is_next_valid) continue;
//       warp_inst_t* pI = next_inst();           // read-only, no pop
//       if (is_warp_eligible_readonly(shared_sm, c_warp, pI, subcore_warp_id)) {
//           ++n_eligible_tail;
//           not_selected++;                       // every eligible tail warp is by definition not-selected
//       }
//       continue;                                 // NEVER call issue_warp / mutating predicate again
//   }
```

**Correctness notes (address the re-review findings):**
- **Both exits handled.** `tail_readonly` is the single trigger for "loop exited early", covering the
  `:638` (issued) and `:703` (greedy const-cache) breaks. The natural-end case never sets it and is
  counted inline as today.
- **The `:703` greedy warp is NOT counted as `not_selected`.** It is the *stopping* warp, evaluated by
  the full path, and it is blocked on `is_l1c_ready` — which the mirror deliberately excludes. NCU
  classifies this as `short_scoreboard`/`imc_miss`, not `not_selected`. Because the tail only counts
  warps *after* the stop, the greedy warp is correctly excluded. (The mirror is used ONLY on the tail,
  never re-applied to the stopping warp.)
- **Winner exclusion.** On exit 1 we `continue` *past* the winner before flipping to read-only, so the
  winner is never re-counted; it is `selected`.

- **Natural-end cycles (no break):** `tail_readonly` never set → loop runs exactly as today over all
  warps; count `eligible`/`not_selected` inline from values it already computes. **Zero added scan.**
- **Early-exit cycles (either break):** heavy path runs only up to the stopping warp (as today); the tail
  runs the cheap read-only mirror. Cost = O(#warps after the stop), not O(all warps). FA3 fwd has ~3–4
  active warps/subcore, so the tail is on average under half.
- **Exact count preserved (NCU-matching):** every eligible warp after the stop is, by the zone argument
  above, exactly a `not_selected` warp — so the sum is the true NCU `not_selected`, not an
  approximation. `eligible_warps_per_scheduler` accumulates (eligible warps examined by the heavy path,
  incl. the winner) + `n_eligible_tail` per cycle.
- **Bit-identity (the paramount always-on constraint):** the heavy path up to the stop is byte-for-byte
  unchanged (`issue_warp` called at most once, same const-cache probes at [subcore.cc:503](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L503) / `are_l1c_operands_ready` fire on exactly the same warps as today, because those warps are still
  visited by the heavy path before either break). The tail warps were **never** visited today (the break
  skipped them), so running them read-only adds **no** new mutating call — provided the mirror truly
  avoids the 3 mutators (§ table above) AND the const-cache probes at [:503](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L503)/[:480](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L480). Verify via the cycle-equality gate below.

- **Maintenance coupling (documented risk):** the mirror duplicates the predicate. If the primary
  predicate changes, the mirror must change too. Accepted because touching the hot-path predicate risks
  bit-identity; a future cleanup can extract a shared read-only core. Add a code comment linking the two
  sites ([subcore.cc:606-609](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L606-L609) ↔ the mirror).

**Perf note:** this is NOT a per-cycle full re-scan. Natural-end cycles add nothing (the loop already
visits all warps); early-exit cycles add only the read-only tail after the stop. Net overhead is far
below the rejected full-scan design and is safe to ship always-on.

---

## Files to modify

| File | Change |
|---|---|
| [gpu-sim.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) `create_gpu_per_sm_stats()` (~2639-2804) | Register all new keys: `..._selected`, `..._stall_dispatch`, `..._stall_warpgroup_arrive`, `..._not_selected`, `total_num_warps_eligible_accumulator`. |
| [subcore.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc) | (a) emit `selected` in the issued branch; (b) `dispatch` split; (c) `warpgroup_arrive` opcode-string split at the wait_barrier stall; (d) new `is_warp_eligible_readonly()` helper; (e) replace **both** breaks ([:638](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L638) winner, [:703](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L703) greedy-l1c) with a `tail_readonly` continuation so the SAME loop finishes the post-stop tail read-only (no second full-warp scan). |
| [subcore.h](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h) | declare `is_warp_eligible_readonly()`. |
| [shader.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) `print_remodeling_stats()` (~1230-1582) | Emit the **full 20-reason NCU-named stall stack** (Phase 1b table) as `ncu_stall_<reason>` + pct: new counters, MAP print-lines for existing counters (`barrier`/`wait`/`math_pipe_throttle`/`no_instructions`/`imc_miss`), the `long_scoreboard`/`short_scoreboard` folds, and explicit 0-lines for residual/negligible reasons. Plus NCU scalars (`issued_warp_per_scheduler`, `eligible_warps_per_scheduler`, `issue_slots_busy_pct`, `no_eligible_pct`). |

No config, no tracer, no header (`mem_fetch.h`/`shader.h`) struct-size change beyond the `subcore.h`
method decl. Build required. **Because `subcore.h` is edited, run `make clean` before building** (project
rule for header edits).

## Verification

1. **Bit-identity (the gate).** Build, run fwd K5 (`OnlyKernel5`) → confirm `gpu_sim_cycle == 137,053`
   (unchanged vs `.o37`). Any movement = a counter mutated timing (the read-only tail leaked a side
   effect) → reject and fix. Same check on bwd K10 vs `.o20` (234,665).
2. **Counter sanity.** `selected + Σ(stall reasons)` reconstructs the eval-cycle population (with the
   documented per-warp overlap, [subcore.cc:738-741](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L738-L741)). `eligible_warps_per_scheduler`, `issued_warp_per_scheduler`,
   `issue_slots_busy_pct` should land near NCU fwd anchors (0.83 / 0.46 / 45.03%).
3. **Shape match to NCU** (CTA_SAMPLING §11.2 pre-experiment): the new reason *shares* should line up in
   shape with NCU `smsp__pcsamp_*` (fwd: `wait`/`long_scoreboard`/`barrier` dominant). Pull HW anchors
   from `nv_reports/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.csv` (fwd = grid (132,1,1))
   and the `.ncu-rep` raw stall page.
4. **Full-coverage check (Phase 1b).** Confirm the emitted `ncu_stall_*` stack has **all 20 NCU reasons**
   present (including explicit 0-lines for `branch_resolving`/`membar`/`tex_throttle`) — no NCU reason
   should be missing from the sim output. This is the "compare every stall reason with HW" deliverable.
5. **The payoff read.** On the new `.o` files, compare `warpgroup_arrive` + tensor-coupled `wait_barrier`
   (→ suspect #1) vs `not_selected` + `dispatch_stall` (→ suspect #2). The larger bucket names the next
   timing lever (async-WGMMA model vs dual-issue), now with an NCU-anchored before/after target.

## Risks

- **R1 — read-only tail leaks a side effect** (breaks bit-identity; always-on has no off-switch). The
  post-winner tail must call ONLY the read-only mirror + the 3 substitutes — never `waiting()`,
  `are_l1c_operands_ready()`, `warp_waiting_at_tma_flush()`, or `issue_warp()`. Mitigated by the
  substitute table above + verification step 1 (cycle equality). This is the single highest-risk item.
- **R2 — predicate duplication drift** (mirror vs primary). Mitigated by a cross-referencing comment;
  future cleanup extracts a shared read-only core.
- **R3 — `long_scoreboard` global-load RAW may be uncounted** today. If so, report the gap; do not
  fabricate the fold. Decide whether to add global-load RAW tracking as a follow-up.
- **R4 — null-key crash** if a counter is incremented but not registered in `gpu-sim.cc`. Mitigated by
  the 3-site checklist; grep every new key string appears in all three files before building.
