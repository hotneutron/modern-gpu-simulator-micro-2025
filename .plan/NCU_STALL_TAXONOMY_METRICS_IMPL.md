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

## Phase 2 — restructure the SAME loop to count `not_selected` + `eligible_warps_per_scheduler` (sizes #2)

> **User decisions (2026-07-14):** (1) NOT a separate every-cycle full-warp scan (wasteful); (2) exact
> `not_selected` count (goal = match NCU); (3) **restructure the primary loop rather than clone a mirror
> predicate** — reuse the original eligibility logic verbatim so the count cannot drift, gating only the
> side-effecting calls on the post-stop tail.

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

**The hard constraint (always-on ⇒ must be truly side-effect-free).** The per-warp evaluation block
([subcore.cc:549-707](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L549-L707)) calls 3 side-effecting functions. On the tail (warps after the stop, which today
are never reached) NONE must run. **Final decision (verified against source): the tail skips all three
and computes `not_selected` from only the read-only sub-conditions** — which is exactly correct because
each skipped condition maps to a *different* NCU reason, not `not_selected`:

| function skipped on tail | why it mutates | why skipping is correct for `not_selected` |
|---|---|---|
| `are_l1c_operands_ready()` ([subcore.cc:584](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L584), def [:953-975](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L953-L975)) | real L0-const-cache `access()` | `is_l1c_ready` is NOT in `are_switch_warp_conditions_ready` ([:606-609](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L606-L609)); const-cache wait = NCU `short_scoreboard`/`imc_miss`. |
| `c_warp->waiting()` ([subcore.cc:573](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L573)) | its `warp_waiting_at_mem_barrier` sub-check drains membar state | a warp parked here is waiting on a CTA-barrier / grid-barrier / membar = NCU `barrier`/`membar`, **not** `not_selected`. In FA3 the consumer/producer sync is the mbarrier `wait_barrier` (kept read-only), and `barrier`/`membar` are separate reasons, so excluding it does not steal from `not_selected`. |
| `warp_waiting_at_tma_flush()` ([sm.cc:2011-2015](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2011-L2015)) | inserts a debug-log set (only when opcode==UTMACMDFLUSH; else early-returns false with NO side effect) | UTMACMDFLUSH store-drain = NCU `tma_flush` (drain axis), not `not_selected`. fwd is load-only (tma_flush≈0). |

So the tail's eligibility test uses ONLY the read-only conditions already computed in the block:
`is_not_yield && is_stall_counter_0 && are_wait_barriers_ready && is_fu_available &&
is_not_warp_waiting_ldgdepbar && are_traditional_scoreaboards_ready &&
is_write_available_result_queue_for_fixed_latency_available`. Every one of these is a read-only call
(scoreboard `checkCollision_remodeling` const, `is_stall_counter_0`, `is_wait_barriers_ready_entry_point`,
`is_yield_ready`, `is_waiting_ldgdepbar`, `can_issue`, RF-queue-space, `next_inst` — no pop). **No header
change, no `make clean` needed.**

> **Accuracy note (R2, refined):** the tail eligibility drops `!c_warp->waiting()` from the predicate,
> so it does **not** exclude warps parked at a CTA/grid barrier or membar. In FA3 those are ~0 on the
> consumer path (sync is via mbarrier `wait_barrier`, which IS checked), so `not_selected` is exact for
> FA3. For a kernel with heavy `__syncthreads`, tail `not_selected` could be slightly over-counted; a
> read-only 3-of-4 `waiting()` (SM getters) would close it but needs a header helper — deferred.

**Design — restructure the SAME loop (no mirror function, no second scan): skip only the mutators +
`issue_warp` on the tail.**

Per the user decision (accuracy + no drift), do NOT clone the predicate into a mirror. Instead keep the
one evaluation block and gate the 3 side-effecting calls behind a `tail_readonly` flag. This reuses the
original eligibility logic verbatim, so `not_selected` cannot drift from the real predicate.

```
// Add two locals at the top of Subcore::issue():
bool tail_readonly = false;      // set once the loop would have broken (either exit)
unsigned n_not_selected = 0;
unsigned n_eligible = 0;

// Inside the block (subcore.cc:549-707), gate the 2 mutators on !tail_readonly:
//   is_l1c_ready            = tail_readonly ? true /*unused*/ : are_l1c_operands_ready(sm, pI);
//   is_not_warp_in_prog_bar = tail_readonly
//                               ? !(functional_done() || warp_waiting_at_barrier()
//                                                     || warp_waiting_grid_barrier())   // read-only 3-of-4
//                               : !c_warp->waiting();
//   is_not_warp_tma_flush   = tail_readonly ? !warp_has_outstanding_stores(sm_warp_id)
//                                           : !warp_waiting_at_tma_flush(sm_warp_id, pI);
//
// Then, at each former break, DO NOT break — flip to read-only and keep looping:
//
//   exit 1 (subcore.cc:625-638, winner): after issue_warp(...) / is_issued_inst=true /
//           m_greedy_pointer_issue=subcore_warp_id  ->  tail_readonly = true; continue;
//
//   exit 2 (subcore.cc:702-704, greedy const-cache): the `else{break;}` becomes
//           tail_readonly = true; continue;   // is_issued_inst stays false
//
// Count not_selected ONLY while tail_readonly is already set (i.e. warps strictly AFTER the stop):
//   if (tail_readonly && are_switch_warp_conditions_ready) {   // l1c excluded by definition
//       ++n_not_selected; ++n_eligible;
//   }
// (eligible warps encountered BEFORE the stop are counted inline in the normal path.)
```

**Why this is exact and matches NCU:**
- `not_selected` = `are_switch_warp_conditions_ready` true on a warp strictly after the stop. That
  predicate excludes `is_l1c_ready` — matching NCU, which files const-cache waits under
  `short_scoreboard`/`imc_miss`, not `not_selected`. **No approximation for FA3.**
- The winner is excluded because we set `tail_readonly` and `continue` *past* it before counting.
- The `:703` greedy warp is excluded because it is the *stopping* warp (evaluated by the full path), and
  the tail counts only warps after it.

**Bit-identity (the paramount always-on constraint):**
- Warps **up to and including the stop** run the byte-for-byte original path (`tail_readonly` still
  false): same `are_l1c_operands_ready` / `waiting()` / `warp_waiting_at_tma_flush` calls on exactly the
  same warps as today, `issue_warp` called at most once.
- Warps **after the stop** were **never visited today** (the break skipped them). Running them with the
  3 mutators gated off adds **zero** new mutating call. Verify via the cycle-equality gate below.

- **Maintenance note:** no predicate duplication (the whole point of restructuring vs a mirror). The only
  divergence is the 3 gated calls, each documented inline with a comment pointing at this plan.

**Perf note:** NOT a per-cycle full re-scan. Natural-end cycles are unchanged (loop already visits all
warps). Early-exit cycles add only the read-only tail after the stop (FA3 fwd ~3–4 active warps/subcore,
tail is on average under half). Safe to ship always-on.

---

## Files to modify

| File | Change |
|---|---|
| [gpu-sim.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) `create_gpu_per_sm_stats()` (~2639-2804) | Register all new keys: `..._selected`, `..._stall_dispatch`, `..._stall_warpgroup_arrive`, `..._not_selected`, `total_num_warps_eligible_accumulator`. |
| [subcore.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc) `Subcore::issue()` | (a) emit `selected` in the issued branch; (b) `dispatch` split; (c) `warpgroup_arrive` opcode-string split at the wait_barrier stall; (d) **restructure the loop**: add `tail_readonly` local, gate the 2 mutators (`are_l1c_operands_ready` skip, `waiting()`→read-only 3-of-4) + `warp_waiting_at_tma_flush`→`warp_has_outstanding_stores` on the tail, replace **both** breaks ([:638](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L638) winner, [:703](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L703) greedy-l1c) with `tail_readonly=true; continue;`, count `not_selected`/`eligible` (no mirror function). |
| [subcore.h](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h) | **No change needed** for `not_selected` (all logic is inside `issue()`, no new method). Edit only if a small read-only helper for the tma-flush/waiting read-only checks is factored out; otherwise leave untouched (then no `make clean` required). |
| [shader.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc) `print_remodeling_stats()` (~1230-1582) | Emit the **full 20-reason NCU-named stall stack** (Phase 1b table) as `ncu_stall_<reason>` + pct: new counters, MAP print-lines for existing counters (`barrier`/`wait`/`math_pipe_throttle`/`no_instructions`/`imc_miss`), the `long_scoreboard`/`short_scoreboard` folds, and explicit 0-lines for residual/negligible reasons. Plus NCU scalars (`issued_warp_per_scheduler`, `eligible_warps_per_scheduler`, `issue_slots_busy_pct`, `no_eligible_pct`). |

No config, no tracer change. If `subcore.h` is **not** edited (default — all logic in `issue()`), only
`.cc` files change and a normal incremental build suffices. **If any header (`subcore.h`) is touched,
run `make clean` first** (project rule for header edits). Build required either way.

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

- **R1 — tail leaks a side effect** (breaks bit-identity; always-on has no off-switch). When
  `tail_readonly` is set, the block must NOT call `issue_warp()`, `are_l1c_operands_ready()`,
  `c_warp->waiting()` (use the read-only 3-of-4), or `warp_waiting_at_tma_flush()` (use
  `warp_has_outstanding_stores`). Mitigated by the mutator table above + verification step 1 (cycle
  equality). Single highest-risk item.
- **R2 — membar-warp misclassification** (accuracy edge, not bit-identity). The tail drops
  `warp_waiting_at_mem_barrier` from `waiting()`, so a warp blocked *only* on a memory fence would be
  counted as `not_selected` instead of NCU `membar`. In FA3 `membar` ≈ 0% (scope-only), so the error is
  negligible; documented so it is revisited if a future kernel has non-trivial membar.
- **R3 — `long_scoreboard` global-load RAW may be uncounted** today. If so, report the gap; do not
  fabricate the fold. Decide whether to add global-load RAW tracking as a follow-up.
- **R4 — null-key crash** if a counter is incremented but not registered in `gpu-sim.cc`. Mitigated by
  the 3-site checklist; grep every new key string appears in all three files before building.

---

## Post-implementation — measured findings + mapping-bug fixes (2026-07-15)

First measured run (FWD K5 `.o39`, on top of Opt9) confirmed the build and **passed the bit-identity
gate** (`.o38` == `.o39` == `gpu_sim_cycle 135,999`). Results (scalars + stack) are recorded in
[FA3_progress.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.result/FA3_progress.md) "Measured result — FWD K5". The run also exposed **two mapping bugs** in the
first-cut counters. Both are **metrics-only** (no timing-model change; bit-identity preserved).

### Bug 1 — `warpgroup_arrive` was in the wrong bucket AND matched the wrong string

**Symptom:** `.o39` reported `warpgroup_arrive = 0.00%`.

**Root cause (two errors):**
1. Wrong bucket: the split was placed inside the **`wait_barrier`** (mbarrier) stall. But WGMMA is a
   tensor-**FU** op — a consumer waiting on a WGMMA result stalls on a **scoreboard RAW**, not an
   mbarrier wait. So the split could never fire from `wait_barrier`.
2. Wrong string: it matched the literal `"WARPGROUP"` in the head opcode, whereas the tracer names
   tensor ops `WGMMA`/`HGMMA`/… (`is_warpgroup_tensor_opcode`, [traced_instruction.cc:54-61](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/traces_enhanced/src/traced_instruction.cc#L54-L61)).

**Fix (scoreboard-based — the correct design).** Track which pending destination registers belong to an
in-flight tensor op, then attribute a scoreboard stall to `warpgroup_arrive` when the stalling warp's
head instruction has a RAW against one of those registers:

| site | change |
|---|---|
| [scoreboard.h](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/scoreboard.h) | add `std::vector<std::set<unsigned>> tensoropregs;` (parallel to `longopregs`) + declare `checkTensorCollision_remodeling()`. **Header edit ⇒ `make clean` required.** |
| [scoreboard.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/scoreboard.cc) | `tensoropregs.resize(n_warps)` in ctor; on reserve, `if (inst->is_tensor_core_op()) tensoropregs[wid].insert(reg)` ([:180-182](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/scoreboard.cc#L180-L182)); erase symmetrically in both `releaseRegisters_remodeling` and `releaseRegisters`; new `checkTensorCollision_remodeling()` = same operand-gather as `checkCollision_remodeling` but intersect against `tensoropregs` ([:294-314](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/scoreboard.cc#L294-L314)). |
| [subcore.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc) | move the split into the `!are_traditional_scoreaboards_ready` branch; set `warpgroup_arrive` when `get_scoreboard()->checkTensorCollision_remodeling(sm_warp_id, pI)` ([:728-737](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L728-L737)). Remove the old `"WARPGROUP"` string match. |

Detection uses `inst->is_tensor_core_op()` (== `op == TENSOR_CORE_OP`, [abstract_hardware_model.h:1193](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L1193)),
which the parser already sets — more robust than opcode-string matching. `warpgroup_arrive` is now a
real suspect-#1 signal, **orthogonal to `mma`** (`fu_occupied_tensor`, FU-busy): `mma` = tensor pipe is
busy; `warpgroup_arrive` = a consumer is RAW-blocked on a not-yet-retired WGMMA result. Together they
bound the async-WGMMA opportunity.

**Bit-identity note.** `tensoropregs` is written only in reserve/release and read only by the new
observe-only `checkTensorCollision_remodeling` (never gates issue). So it is timing-neutral — but the
next post-`make clean` run must re-confirm FWD = 135,999 (and bwd = its Opt9 value) since a header
changed.

### Bug 2 — `dispatch_stall` duplicated `mio_throttle`

**Symptom:** `.o39` had `dispatch_stall == mio_throttle == 211,152` (identical).

**Root cause:** both were mapped to `waiting_result_queue_full`. The real issue-port backpressure signal
`stall_issue_port_busy` (472,562) was left unused.

**Fix:** in [shader.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L1598) map `dispatch_stall` ← `total_num_cycles_issue_stage_stall_issue_port_busy`;
`mio_throttle` keeps `result_queue_full`. The per-warp `..._stall_dispatch` counter (result-queue axis)
is retained but no longer aliased to NCU `dispatch_stall`.

### Resolved — `math_pipe_throttle` mapping is correct as-is (`sfu + sp_int_dp`)

Re-examined after `.o39` showed `sfu = 0`, `sp_int_dp = 11.05%`. **Decision: keep the mapping
(`math_pipe_throttle` ← `fu_occupied_sfu + fu_occupied_sp_int_dp`); no code change.** Rationale:

- NCU `math_pipe_throttle` = "a **math execution pipe** (FMA / ALU / FP64 / XU-transcendental) was busy
  so the warp could not issue." It is deliberately distinct from `mma` (tensor pipe) and `tex_throttle`.
- The sim's pipe classification ([subcore.cc:685-690](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L685-L690)) is exactly this partition:
  `sp_int_dp` = `SP_OP/HALF_OP/INTP_OP/DP_OP/UNIFORM_OP` (the FMA/ALU/FP64 math pipes),
  `sfu` = `SFU_OP` (the XU/MUFU transcendental pipe). Their sum **is** NCU's math-pipe set.
  `tensor` → `mma` (excluded, correct) and `other` (LDST etc., 0.86%) → not a math pipe (excluded,
  correct). So `sfu + sp_int_dp` is the accurate mapping; the earlier "generic ALU may not belong"
  worry was wrong — FMA/ALU *is* the math pipe NCU counts.
- **Why `sfu = 0` (not a mapping bug):** transcendental cost is not being routed to the SFU pipe because
  the config sets `-ptx_opcode_latency_sfu` but not `-trace_opcode_latency_initiation_sfu`, so in trace
  mode `exp`/MUFU falls back to the FP-add pipe and lands in `sp_int_dp`. That is the separate **TODO-2**
  (SFU under-cost) tracked in `.result/FA3_progress.md`; it is a *timing* fix (would break bit-identity)
  and is out of scope for the observe-only taxonomy. The current mapping still totals the math-pipe
  stall correctly (it is just all attributed to `sp_int_dp` until TODO-2 is done, at which point part of
  it shifts into `sfu` — the **sum stays the NCU `math_pipe_throttle` value** either way).

Net: the NCU-named line `math_pipe_throttle = sfu + sp_int_dp` is correct and stays. No `make clean`.

### Follow-up — HW per-reason export

The NCU CSV export (`nv_reports/…full_rpt.csv`) contains Scheduler/Warp-State **scalars** (Issue Slots
Busy, Eligible/Issued/No-Eligible Warps Per Scheduler, Warp Cyc/Issued Inst) but **not** the per-reason
Warp-State stall breakdown. To fill the HW per-reason column in the FA3_progress tables, re-export the
`.ncu-rep` with the Warp State Statistics detail (the `smsp__average_warps_issue_stalled_*` group).
