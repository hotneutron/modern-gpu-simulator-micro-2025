# Plan: Fast FA3 Simulation via CTA Subsetting, Validated by Per-PC Stall Fit

**Date:** 2026-06-29, redesigned 2026-07-07, risk-hardened 2026-07-08. **Branch:** `accorde_npu` (~/accorde). **Status:** plan + verification spikes; no simulator code yet.
_(Renamed from `phased_ncu_gcom_compare_early_term.md` → `CTA_SAMPLING.md` on 2026-07-08: intra-kernel phased run was ruled infeasible (P9); CTA subsetting is the mechanism.)_

## Progress summary (2026-07-08)

**Objective.** The simulator (GCOM) diverges from H100 and is being debugged; a full FA3 run (~12 h) makes
each debug iteration too slow. Goal: **cut wall-clock per experiment** by simulating a **CTA subset**, and
validate model fidelity by comparing **per-PC stall distributions** against NCU — *without* requiring a full
run per iteration.

**Two-stage structure (why).** Stage 1 = pick smallest faithful K by CTA subsetting (speed); Stage 2 =
per-PC stall reason-share vs full-HW (fidelity). Kept separate so subset error and model error don't
confound (P1).

**Done / verified:**
- **Gating 1(a) — PASS:** NCU `.ncu-rep --page source` gives per-SASS-PC stall taxonomy + `All`/`Not-issued`
  twins + auxiliary exact counters (exec/memory). No re-profiling needed. Kernel 10 (bwd) is memory/sync-bound.
- **Gating 1(b) — PASS:** PC spaces align — `sim trace-PC == sass offset == (NCU_VA − function_base_VA)`,
  instruction-for-instruction. (§11.1b)
- **Resolved by code/data:** R4 (PC align, P6), R6 (head-PC attribution), R9 (per-CTA HW asymmetry, P8),
  R11 (phased run infeasible, P9).
- **Design done:** R13/P12 warp-role region labeling (measured SASS: `HGMMA*`/`WARPGROUP*`→consumer,
  `UTMALDG*`→producer; CFG-propagate from `SR_TID.X` split) — §9.
- **Kernel identity:** launch-order **kernel 5 = FA3 fwd**, **kernel 10 = FA3 bwd**.
- **NCU↔GCOM reason mapping table + counter-redesign worklist** built from measured data — §7.1.

**Decisions:**
- Phased (intra-kernel mid-phase) run is **off the table** (P9): no microarch-state checkpoint; mid-phase
  FA3 deadlocks. CTA (spatial) subsetting is the only working time-cut axis.
- Full-sim baseline is **optional, not routine** (P10): cheap K-vs-K+ΔK convergence every iteration + an
  occasional full-sim anchor on demand.
- **PREREQUISITE (user):** the GCOM stall-metric redesign (§7.1) is done **first**, before the rest.

**Open / next (need simulator runs, deferred):**
- **Step 0 — GCOM metric redesign (HIGH, R12/P11):** add `not_selected`, `warpgroup_arrive`, `dispatch_stall`,
  `sleeping`, decide `branch_resolving`; define `selected` as denominator; fix many→one folds. §7.1.
- **R2 pre-experiment (HIGH):** kernel-level GCOM no-issue-share vs NCU all-cycle-share → size Gap C.
- **Gating 1(c) CTA-completion**, **1(d) speedup + K-convergence** (need sim runs).
- Stage 1 K-sweep → faithful K; Stage 2 Change B′ code; pipeline scripts (`define_regions.py`, compare).

---


> **2026-07-08 code-verification pass.** The plan's claims were checked against the actual
> tree. Confirmed correct: the stall taxonomy exists and matches §7 verbatim
> ([subcore.cc:409-431](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L409-L431),
> [736-757](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L736-L757));
> the Gap-C diagnosis is exact — attribution lives in the no-issue `else` and the issue loop
> `break`s on first issue ([subcore.cc:638](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L638));
> an existing `wgmma_step0_instrument_enable` flag gives the exact pattern for the Change-B′ flag.
> **New defect found (see §2 and R8):** `-gpgpu_max_cta` terminates on *issued*, not *completed*,
> CTAs — it does **not** run K CTAs to completion as the plan assumed.
>
> **2026-07-08 spike 1(a) — PASS.** On the existing `.ncu-rep` (H100, driver 595.71.05, NCU 2025.2.1),
> `--page source` yields per-SASS-PC rows with the full stall taxonomy **and** auxiliary exact counters
> (execution + memory), plus separate `All`/`Not-issued` sample columns — directly exercising P2/Gap C. FA3
> mainloop kernels are memory/sync-bound (`long_scoreboard`+`barrier`+`warpgroup_arrive` dominant), matching
> P4. No re-profiling needed; the HW side of Stage 2 is already in hand. `not_selected` is a real 5–11% share
> ⇒ Gap C is non-trivial. **Also verified:** NCU exposes **no per-CTA-id breakdown** (only launch props /
> grid sums) — see §6/P8. Remaining spike gates: 1(c) CTA-completion, 1(d) speedup-vs-K (1(a), 1(b) PASS).

## 0. What "phased run" turned out to mean, and why the axis is CTA not phase (P9, P10)

**The real objective** (clarified 2026-07-08): the simulator diverges from H100 and is being **debugged**; a
full FA3 run (~12 h) makes each debug iteration too slow. The goal is to **cut wall-clock per experiment** so
the sim↔HW gap can be chased. Two consequences the earlier draft got wrong:

- **P9 — "phase" = arbitrary mid-kernel sub-phase, and it is NOT feasible.** The intent was to split one
  kernel into ~10 sub-phases and run only phase *k..k+1* in the middle. **Verified impossible** on this tree:
  - The only checkpoint infra is `checkpoint::{load,store}_global_mem`
    ([abstract_hardware_model.h:1774-1782](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L1774-L1782))
    — it saves **register + local/shared mem + SIMT stack only**, keyed on **CTA-id ranges**
    ([cuda-sim.cc:2418-2450](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc#L2418-L2450),
    resume at [sm.cc:2175-2200](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2175-L2200)).
    There is **no save/restore of microarch state** — mbarrier phase-parity, scoreboard, TMA in-flight,
    result queue, FU pipeline. Those *are* the FA3 stall sources.
  - Starting an FA3 mainloop mid-phase without that state deadlocks immediately — producer/consumer are
    mbarrier phase-locked; the missing prior-phase arrivals hang the consumers
    ([debug-fa3-deadlock.md](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/debug-fa3-deadlock.md) is a live instance). Same reason §6 says HW can't be truncated mid-flight.
  - A mid-phase's correct start state depends on all prior iterations (shared-mem tiles, accumulators,
    mbarrier phase); reconstructing it means simulating the prefix anyway — no time saved. Functional
    fast-forward can't rebuild microarch state.
  - **The only time-cutting axis that works is CTA (spatial), not phase (temporal):** CTAs are independent
    and each runs a complete mbarrier lifecycle, so K of them can be simulated to completion with no
    deadlock. This is exactly why the plan uses CTA subsetting.
- **P10 — full-sim is optional, not the routine baseline.** The whole point is that the full run is too slow
  to iterate on, so Stage 1 must **not require** a full-sim per experiment. But full-sim comparison is not
  forbidden — it is the *strongest* subset-error check and may be run **occasionally** (e.g. once to anchor
  the convergence tolerance, or when the cheap check looks suspicious). **Two-track:** the cheap **K vs K+ΔK
  convergence** check runs every iteration; the expensive **K-sim vs full-sim** check runs on demand, rarely.
  Neither the routine loop nor the debug objective depends on doing a full-sim every time (see §1, §4).

**Net:** phased (intra-kernel) run is off the table; **CTA sampling is the mechanism**, and its trust is
established primarily **without** a full-sim reference — with an **optional occasional full-sim anchor**.

## 1. Goal and two-stage structure (re-scoped: cheap default + optional full-sim anchor)

Run GCOM **faster** on FA3 by simulating a **CTA subset** and prove the subset is trustworthy — **without
requiring** a full-sim run per experiment (P10), while still **allowing** an occasional one. The trap (P1) is
that comparing a K-CTA sim directly to full HW confounds two errors — **subset error** (fewer CTAs) and
**model error** (sim ≠ HW). They are separated as follows:

- **Stage 1 — pick K (speed), two-track subset-error check.**
  - **Cheap, routine — K vs K+ΔK convergence.** Sweep K upward and stop when the *normalized* aggregate
    stall/IPC counters stop changing (successive differences < tolerance). The converged K is the fast proxy;
    the full grid is never needed. Strengthen with **per-CTA-id agreement** (§3 tier-3): a CTA appearing in
    both a small-K and larger-K run should give the same per-CTA stall profile.
  - **Expensive, occasional — K-sim vs full-sim.** The direct subset-error measurement, run **on demand**
    (once to anchor the convergence tolerance, or to spot-check a suspicious result), reusing an existing
    full-sim run if one is available. Not part of the per-iteration loop.
  Both measure subset error; the cheap check is the default, the full-sim check is the strongest anchor.
- **Stage 2 — model fidelity, vs HW directly at the chosen K.** Compare the **K-sim per-PC stall
  distribution** to **full-HW** per-PC (§5, §10). Because HW gives only kernel-summed per-PC (§6, P8) and the
  comparison is a *reason-share distribution* (not absolute counts), a representative K-CTA sample and the
  full HW launch are commensurable on shape. This is the harder per-PC track (Change B′). If a full-sim
  anchor was run, Stage 2 may substitute full-sim for K-sim to get the cleanest model-error signal.

Stage 1 establishes *how small K can be* using cheap sim-internal convergence by default, with an optional
full-sim anchor; Stage 2 uses that K to chase the sim↔HW gap. The routine loop requires no 12 h full-sim run.

> **Note on P1 vs P10 tension.** The original two-stage split used full-sim to *isolate* subset error from
> model error. Making full-sim optional (P10) means the **routine** Stage 2 compares K-sim (not full-sim) to
> HW, so residual subset error can leak into the model-error signal. Mitigations, in order of strength:
> (1) Stage 1 K-convergence bounds residual subset error below the model-error signal; (2) when a clean
> separation is needed, run the occasional full-sim anchor and compare full-sim↔HW. Weaker-by-default than
> always-full-sim, but recoverable on demand (see R1, R10).

---

# Stage 1 — Pick K by CTA subsetting (speed)

## 2. Cut axis = CTA subset, not instruction window

- **Instruction window (`-gpgpu_max_warp_insn`) — do NOT use.** It sees every CTA's prologue + early
  mainloop only, never epilogue/steady-state ⇒ cold-start biased; can't represent the full kernel.
- **CTA subset — correct axis, but `-gpgpu_max_cta` alone is WRONG (P7, new).** The intent is "run K CTAs
  to completion so each covers its full prologue→mainloop→epilogue lifetime." But **`-gpgpu_max_cta` counts
  *issued* CTAs, not *completed* ones**: `active()` returns false the instant `gpu_tot_issued_cta >= K`
  ([gpu-sim.cc:3138-3140](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3138-L3140);
  same predicate in `cycle_insn_cta_max_hit` [gpu-sim.h:794-795](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h#L794-L795)).
  So the last wave of in-flight CTAs is **truncated mid-lifetime** — reintroducing on the CTA axis the exact
  epilogue/steady-state loss the plan rejected the instruction window for.
- **Fix: use `-gpgpu_max_completed_cta K` instead** (or in addition). It keys on `gpu_completed_cta`
  ([gpu-sim.cc:3141-3143](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3141-L3143),
  option at [gpu-sim.cc:2251](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L2251)),
  which increments only when a CTA finishes, so **≥K CTAs are guaranteed to complete** before termination.
  Note this lets *more than K* CTAs launch (any concurrently-resident ones finish too) — acceptable and even
  desirable, but it means K is a completed-CTA floor, not an exact launch count.
- **Normalization guard.** Regardless of which stop flag is used, any CTA still in-flight at termination must
  be **excluded** from the per-CTA / per-cycle normalization (§4.2), or its missing epilogue skews the stall
  shares. Prefer completed-CTA-only accounting. CTAs run their full lifetime; CTAs are independent ⇒ **no
  deadlock** (avoids the FA3 truncation problem in §6). Selection precedent:
  [gpu-sim.cc:3964](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3964),
  [hit_max_cta_count:2402-2408](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L2402-L2408).
- Speedup roughly tracks active-SM count: FA3 bwd is 384 CTA / 132 SM ≈ 2.9 waves; icnt/L2/DRAM fixed cost
  is the floor — measure.

## 3. Causal FA3 caveat — representative sample, not first-K (P3)

Causal masking makes CTAs **non-uniform**: mainloop trip count grows with query-block position. First-K (and
CTA-0 specifically) is the *shortest, least-contended* case — unrepresentative. Need a **representative CTA
sample** spanning the causal distribution. CTA-0-only is a **bring-up** convenience only, never the
comparison config.

**CTA-sampling implementation path (three tiers, resolves worry P8b "how do we sample CTAs").** The
simulator already tracks CTA-id end to end, so this is mostly config + a small gate extension, not new
subsystems:

1. **Count-only (zero code, already implemented).** `-gpgpu_max_completed_cta K` stops after K CTAs
   *complete* ([gpu-sim.cc:3050-3051](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3050-L3051),
   option [gpu-sim.cc:2160](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L2160)).
   Runs the *first* K in launch order — fine for bring-up and for the speedup-vs-K curve, but launch order
   is causally biased (see above), so **not** the final comparison config.
2. **Representative subset = CTA-id allow-list (small extension).** Gate CTA launch on membership in a chosen
   id set instead of a prefix. Precedent is the checkpoint/resume **CTA-range gate** already in
   `issue_block2core` — it launches/loads a CTA only when `ctaid >= resume_CTA && ctaid < checkpoint_CTA_t`
   ([sm.cc:2102 issue_block2core](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2102),
   and the launch-side gate at [gpu-sim.cc:3138-3140](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3138-L3140)).
   Generalize that range test to an allow-set (`-gpgpu_cta_allow_list "id,id,…"`); `kernel.get_next_cta_id_single()`
   already yields the global id to test. accorde has CTA-sampling infra to *choose* the set
   (`cluster_simpoint.py`, `checkpointing_and_sampling.md`). Because CTAs are independent, skipping ids
   cannot deadlock (§2).
3. **Per-CTA stat dump (for the Stage-1 subset-fidelity check).** Key the existing issue-stall counters by
   `m_local_to_global_cta_id[hw_slot]`
   ([sm.h:159](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h#L159)),
   flushing on CTA completion ([register_cta_thread_exit](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1305),
   which already bumps `ctas_completed`). This lets a CTA's stall profile be checked **per matched CTA-id
   across two K values** (K vs K+ΔK, §4), the strongest form of the subset-error test — and it needs no HW
   counterpart (§6, P8).

## 4. The K-selection loop (cheap K vs K+ΔK convergence; optional full-sim anchor)

1. Start at a small representative K (§3). Run K-sim; record the **existing aggregate counters** (the
   `m_sm_stats` issue-stall totals at
   [subcore.cc:742-757](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L742-L757),
   plus L2/DRAM/IPC), all normalized per-CTA or per-cycle. **In-flight-CTA exclusion guard (P7):** only CTAs
   that reached completion contribute to the normalized stats; any CTA truncated at termination is dropped,
   so a partial final wave cannot skew the stall shares. This is why §2 uses `-gpgpu_max_completed_cta`.
2. Increase K by ΔK and re-run. Compare the **new** normalized counters to the **previous** K's (successive
   differences), **not** to a full-sim. Optionally cross-check per-CTA-id agreement (§3 tier-3) for CTAs
   present in both runs.
3. **Convergence stop:** when successive normalized differences fall below tolerance, the current K is the
   fast proxy — its per-CTA behavior is no longer changing as more CTAs are added, i.e. subset error is
   bounded. **No full grid, no 12 h baseline.** No per-PC code needed for this stage.
4. **Optional full-sim anchor (on demand, not per-iteration).** Once — or whenever the cheap check is
   suspect — run one **K-sim vs full-sim** comparison (reuse an existing full-sim if available) to (a)
   calibrate the step-3 tolerance against the true subset error, and (b) validate that convergence implies
   fidelity, not just stability. This is the expensive track from §1; keep it rare so the routine loop stays
   cheap.

**Speedup is contention-bounded (P4).** Intra-CTA stalls (scoreboard, tensor-pipe, mbarrier) survive small
K; inter-CTA/memory stalls (`long_scoreboard` from L2/DRAM-contended TMA arrival) only exist at large K.
FA3's interesting stalls *are* the memory/mbarrier ones, so K may be pushed high and the speedup modest.
**Risk:** if the counters only converge near full K, the convergence test itself becomes expensive — bound
this by capping the sweep and, if it hasn't converged by the cap, treating the largest affordable K as the
proxy and folding the residual into Stage-2 error budget. **Quantify in spike 1(d)** before Stage 2.

---

# Stage 2 — Model fidelity by per-PC stall fit (K-sim at converged K vs full-HW)

## 5. Comparable signal = per-PC warp-issue-stall distribution (+ auxiliary per-PC signals)

HW attributes stalls by **PC** (spatial). An instruction-window phase is temporal and has no HW counterpart
(FA3's mainloop revisits the same PCs; PC sampling can't separate visits). So the comparison unit is a
**PC region**, and GCOM must also bucket stalls by PC. Aggregate PMCs (L2/DRAM/tensor) stay kernel-level.

**Verified (spike 1(a), 2026-07-08).** NCU's `--page source` on the existing `.ncu-rep` gives, **per SASS
PC (`Address` + disassembled `Source`)**, not just the full stall taxonomy but a set of **auxiliary per-PC
signals** — all confirmed populated on kernels 5 & 10 of the FA3 bwd report:

| axis | per-PC NCU columns | nature | use |
|---|---|---|---|
| **stall (primary)** | `stall_long_sb / short_sb / barrier / wait / mio / math / not_selected / wgmma…` and each `(Not Issued)` twin | statistical PC-sample counts | the reason-share comparison (§10) |
| **execution** | `Instructions Executed`, `Thread Instructions Executed`, `Predicated-On …`, `Avg. Threads Executed`, `Divergent Branches` | **exact** HW counters | (i) cross-check GCOM stall classification per PC; (ii) **PC-alignment aid for 1(b)** — match sim per-PC exec counts to NCU's |
| **memory** | `Address Space`, `Access Operation`, `Access Size`, `L1 Tag Requests Global`, `L1 Conflicts Shared`, `L2 Theoretical Sectors Global (+Excessive/Ideal)`, L2 evict-policy breakdown | **exact** HW counters | validate that a PC GCOM marks `short/long_scoreboard` is really a memory op of the expected size/space |

**Two axes, not one comparison.** The stall columns are *statistical samples*; the execution/memory columns
are *exact counters*. Keep them separate per §10's rule (never compare a sampled count to an exact count).
The auxiliary signals are for **cross-validation and PC-alignment**, not for the reason-share divergence
number. All of these are still **kernel-aggregate per PC** — see §6 on the per-CTA limit.

## 6. Why HW can only give PC sampling (not sub-kernel counters)

- **NCU aggregates per launch and replays the whole kernel** — `Elapsed Cycles`, `sm__/lts__/dram__` are
  launch properties; an intra-launch nvtx range is not separately counted on H100.
- **FA3 cannot be truncated mid-flight** — warp-specialized producer/consumer with TMA `expect-tx` + mbarrier
  phase-parity across 384 CTAs ([FA3-enablement.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.plan/FA3-enablement.md)).
  Early-return ⇒ consumers hang; predicate-off ⇒ warps spin to the end. No deadlock-free prefix on HW.
- **Consequence:** HW's only sub-kernel signal is **PC sampling** — one full normal run recording
  `(warp PC, stall reason)`.
- **No per-CTA breakdown on HW (verified, P8).** Every CTA/block metric NCU exposes is either a *launch
  property* (`Block Size`, `launch__*`) or a *whole-grid sum* (`sm__ctas_launched.sum`); there is **no
  per-CTA-id decomposition** of stalls or execution. per-PC is available, per-(PC×CTA) is not.
- **Why the asymmetry is harmless (resolves worry P8).** Per-CTA is only ever used **sim↔sim**; the sim↔HW
  comparison never needs it:
  - Stage 1 (K vs K+ΔK convergence): both operands are the simulator, which *does* track CTA-id fully
    ([issue_block2core sets `m_local_to_global_cta_id`](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2102),
    [`get_global_cta_id`](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h#L159),
    per-CTA completion in [`register_cta_thread_exit`](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1305)),
    so per-CTA accounting exists on **both** sides — no HW needed.
  - Stage 2 (K-sim at converged K vs full-HW): compared on **per-PC (CTA-agnostic, kernel-summed)** as a
    reason-*share*, exactly the form NCU gives (§10). Per-CTA is never on this axis.
  So HW's missing per-CTA dimension is only ever needed where **both operands are the simulator**; it never
  blocks a sim↔HW comparison.

## 7. GCOM already computes an NCU-aligned stall taxonomy — but on the wrong cycles (P2)

[subcore.cc:409-431](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L409-L431)
classifies why each non-issuing warp is blocked, with comments mapping to NCU
`smsp__average_warps_issue_stalled_<reason>`. Taxonomy:

| GCOM counter (`...stall_at_least_one_warp_...`) | meaning | NCU reason (`smsp__pcsamp_warps_issue_stalled_*`) |
|---|---|---|
| `waiting_wait_barrier` | mbarrier (TMA data arrival) | `long_scoreboard` (+`barrier`) |
| `waiting_inst_barrier` | named barrier / LDGDEPBAR | `barrier` / `long_scoreboard` |
| `waiting_tma_flush` | `cp.async.bulk.wait_group` drain | `long_scoreboard` / `drain` |
| `waiting_scoreboard` | traditional RAW/WAR | `short_scoreboard` |
| `waiting_l1c` | constant-cache miss | `short_scoreboard` / `imc_miss` |
| `waiting_stall_count` | fixed-latency dep | `wait` |
| `with_fu_occupied` (+`_tensor`/`_sfu`/`_sp_int_dp`) | pipe busy | `mma` / `math_pipe_throttle` |
| `waiting_result_queue_full` | RF/result-queue backpressure | `mio_throttle` / `lg_throttle` |
| `waiting_yield` | YIELD | `drain` / `not_selected` |
| `..._no_valid_instruction_*` (frontend/L0I/ibuffer) | fetch/decode not ready | `no_instruction` |
| *(missing)* | eligible warp not picked this cycle | **`not_selected`** |

**Three gaps, not one:**
- **Gap A — aggregate, not per-PC.** Flat per-SM `m_sm_stats.m_stats_map[...]`
  ([new_stats.h](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/new_stats.h#L172-L197)), no PC key.
- **Gap B — per-cycle boolean OR, not per-warp count.** Increments `is_any_waiting_X` ("≥1 warp"), NCU
  counts *warps*.
- **Gap C (the P2 crux) — recorded only on no-issue cycles.** The attribution block is inside the
  `else` at [subcore.cc:736-757](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L736-L757),
  reached only when nothing issued, and the issue loop `break`s on first issue
  ([subcore.cc:638](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L638)).
  NCU PC-sampling samples **all** cycles/warps, including eligible-but-`not_selected` on issue cycles.
  Differing populations ⇒ **normalizing to shares does NOT fix it**.

## 7.1 NCU↔GCOM reason mapping table + counter-redesign worklist (P11 / R12 prerequisite)

**This section is a PREREQUISITE: complete the GCOM stall-metric redesign below *before* the rest of the
project proceeds** (user decision). Built from the measured kernel-10 (FA3 bwd) NCU reasons and the actual
GCOM counters at
[subcore.cc:742-757](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L742-L757).
NCU shares are kernel-10 aggregate (from spike 1(a)).

**Full NCU reason list (kernel 10, BWD), with the GCOM counter that should correspond and the action needed:**

| NCU reason | k10 share | closest existing GCOM counter | status → action |
|---|---:|---|---|
| `long_scoreboard` | 21.9% | `waiting_wait_barrier` (mbarrier/TMA arrival) + `waiting_tma_flush` | **split ambiguity** — NCU lumps global-mem dep + TMA arrival here; GCOM splits across 2 counters. Need a defined many→one fold, and confirm GCOM's global-load RAW (non-const) is counted (currently `waiting_scoreboard` is traditional RAW only). |
| `barrier` | 17.2% | `waiting_inst_barrier` (named BAR/LDGDEPBAR) | map; but disambiguate vs `warpgroup_arrive` (below). |
| `selected` | 13.1% | *(none — this is the issued winner)* | **exclude from stall denominator.** Not a stall. GCOM's `is_issued_inst` cycle ≈ this. Define as denominator, not a reason. |
| `wait` | 10.4% | `waiting_stall_count` (fixed-latency dep) | direct map. |
| `short_scoreboard` | 8.6% | `waiting_scoreboard` (RAW/WAR) + `waiting_l1c` (const-cache) | fold two GCOM counters → one; verify const-cache belongs here (NCU puts const miss under `imc_miss`, see below). |
| `mio_throttle` | 5.9% | `waiting_result_queue_full` (RF/result-queue backpressure) | map; verify GCOM models MIO-queue pressure, not just RF. |
| `warpgroup_arrive` | 5.7% | *(none clean)* — partially in `waiting_wait_barrier`/`waiting_inst_barrier` | **NEW COUNTER NEEDED** — WGMMA warpgroup arrival/wait (SASS `WARPGROUP.ARRIVE` / `WARPGROUP.DEPBAR.LE`). Currently blended into barrier counters; must be separated to match NCU. |
| `not_selected` | 5.4% | *(missing)* | **NEW COUNTER NEEDED** (P2/Gap C) — eligible-but-not-picked. Added by Change B′ §8.1. |
| `dispatch_stall` | 4.5% | *(none)* | **NEW COUNTER NEEDED / redesign** — dispatch-port / dual-issue contention. GCOM has `issue_port_busy` cycles ([subcore.cc:719](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L719)) but it's outside the per-warp taxonomy; needs re-derivation into the per-PC per-warp axis. |
| `sleeping` | 3.1% | *(none)* — closest `waiting_yield` (YIELD) | **NEW COUNTER NEEDED** — NANOSLEEP / warp sleep. GCOM `waiting_yield` is YIELD only; sleeping is distinct. |
| `math_pipe_throttle` | 1.3% | `with_fu_occupied_sfu` / `..._sp_int_dp` (non-tensor pipe busy) | map (fu-occupied on math pipes). |
| `no_instructions` | 1.2% | `..._no_valid_instruction_*` (frontend/L0I/ibuffer family) | fold the whole frontend sub-tree → one. |
| `imc_miss` | 0.9% | `waiting_l1c` (const-cache miss) | map — but resolve overlap with `short_scoreboard` decision above (const miss = `imc_miss`, const *latency* = short_sb). |
| `branch_resolving` | 0.8% | *(none)* | **NEW COUNTER NEEDED** — branch not yet resolved. GCOM has no branch-resolve stall (trace-driven, no real branch unit); decide whether to model or map to a residual bucket. |
| `drain` | 0.0% | `waiting_tma_flush` (partial) / `waiting_yield` | negligible in FA3 bwd; map loosely or bucket. |
| `misc` | 0.0% | residual | catch-all. |
| `lg_throttle` | 0.0% | `waiting_result_queue_full` (partial) | negligible here. |
| `membar` | 0.0% | (MEMBAR handling in `sm.cc`) | negligible here (scope-only membar, §sm.cc). |
| `tex_throttle` | 0.0% | *(n/a — no tex in FA3)* | ignore. |
| `mma` *(NCU also has this on some builds)* | — | `with_fu_occupied_tensor` | map tensor-pipe busy. |

**Counter-redesign worklist (the actual P11 deliverable — do these first):**
1. **Add `not_selected`** (eligible-but-not-picked) — Change B′ §8.1. *(required)*
2. **Add `warpgroup_arrive`** — separate WGMMA warpgroup arrive/wait out of the barrier counters. *(required — 5.7%)*
3. **Add `dispatch_stall`** — re-derive dispatch/issue-port contention into the per-warp taxonomy. *(required — 4.5%)*
4. **Add `sleeping`** — NANOSLEEP/warp-sleep distinct from YIELD. *(3.1%)*
5. **Add `branch_resolving`** — model or explicitly bucket as residual; decide feasibility in trace-driven mode. *(0.8%)*
6. **Define `selected` as the denominator**, not a stall reason — exclude from share vectors.
7. **Resolve the many→one folds** with a fixed rule: `long_scoreboard` ⊇ {wait_barrier, tma_flush, global-RAW}; `short_scoreboard` ⊇ {scoreboard, l1c-latency}; `no_instructions` ⊇ {all frontend/L0I/ibuffer sub-counters}; `imc_miss` = const-cache *miss* only.
8. **Validate**: after adding counters, the GCOM per-PC reason-share on a kernel-level run must line up in *shape* with NCU kernel-10 (the §11.2 pre-experiment is the measurement that both justifies and checks this).

**Fallback:** any NCU reason that proves impossible to derive in trace-driven GCOM (e.g. `branch_resolving`) is folded into a coarse grouped axis (memory-dep / pipe-busy / sync-barrier / frontend / compute-dep / not-selected / dispatch), and the comparison is done on the coarse axis (R5 fallback).

## 8. Change B′: per-PC, per-warp, every-cycle stall pass

Behind `-gpgpu_perpc_stall_instrument_enable` (default 0 ⇒ baseline unchanged):

1. **Separate stall-state pass (fixes Gap C):** every cycle, classify **every** resident warp's head-PC
   stall state — decoupled from the issue loop's early `break`. Add a `not_selected` reason for warps that
   were eligible but not the issued winner. This is a **new per-warp pass**, more than "add a counter to the
   existing loop."
2. **Per-PC, per-warp counts (Gaps A/B):** `stall_by_pc[pI->pc][reason] += 1` once per warp per cycle,
   per-SM `unordered_map<pc, array<u64, NUM_REASONS>>`. Also `issued_by_pc[pc]` for the denominator.
3. Dump one row per PC to `gcom_stall_by_pc.csv` at kernel end / `timeout`. **No aggregation in C++** —
   PC→region bucketing is offline.
4. Reuse the Stage-1 representative-CTA sample so this runs at the chosen K, not full (still deterministic).

## 9. Regions must be per warp-role (P5)

**Warp-specialization primer.** A CTA (== thread block) holds many warps. In an ordinary SPMD kernel every
warp in the CTA runs the *same* code (only the data differs). FA3 is **warp-specialized**: inside one CTA
the warps branch on their warp-id (`SR_TID.X`) into **different code paths by role** —
- **producer warps**: TMA loads (`UTMALDG`) global→shared + mbarrier arrive ("data ready"),
- **consumer warps**: tensor-core matmul (`WGMMA`) reading shared + mbarrier wait,
so within one CTA warp A is a "data mover" and warp B is a "compute" unit, overlapping load and math. Across
CTAs the *structure* is identical (every CTA has the same producer/consumer split); only the data tile and,
under causal masking, the *iteration count* differ (P3). So: **heterogeneous within a CTA, homogeneous in
structure across CTAs.**

**Consequence for regions.** Producer and consumer PCs are interleaved in the address space, so a linear
`[lo,hi]` "mainloop" range would blend producer's TMA-arrival stalls with consumer's WGMMA-wait stalls —
two different stall profiles averaged into noise. `define_regions.py` therefore emits **PC sets per role**
(`{role, region, pc_set}`), not linear ranges.

**Role-labeling algorithm (P12/R13) — grounded in the actual FA3 bwd SASS (2026-07-08).** Analyzed the
`FlashAttnBwdSm90` mainloop function in `extra_info/sass/flash_bwd_hdim64_bf16_softcapall_sm90…`:
- Warp-role split is a `S2R Rx, SR_TID.X` (read lane/warp id) → `ISETP` (compare warp-id) → `@P BRA` chain
  at function entry (7 `SR_TID.X` reads in the function; branch chain at PCs `0x20→0x70→0x80`, `0x270→0x2d0`,
  `0x560→0x5a0`, …).
- **Role-characteristic ops (measured tags):**
  - **consumer (WGMMA warpgroup):** `HGMMA.64x128x16.F32.BF16` (BF16 tensor-core; 32 in fn), plus
    `WARPGROUP.ARRIVE` / `WARPGROUP.DEPBAR.LE` (5 each). *(Note: SASS mnemonic is `HGMMA`, not `WGMMA`.)*
  - **producer (TMA):** `UTMALDG` (global→shared load, 10 in fn), `UTMACCTL`, `UTMASTG`.
- **Algorithm:**
  1. Parse the function's SASS into a per-PC list (offsets, already aligned to sim/NCU per §11.1(b)).
  2. Build the intra-function control-flow graph from `BRA`/`@P BRA`/`BSSY`/`BSYNC` targets.
  3. Seed role labels: any PC with `HGMMA*`/`WARPGROUP*` → **consumer**; any PC with `UTMALDG`/`UTMACCTL`/
     `UTMASTG` → **producer**.
  4. Propagate labels along the CFG within the warp-id-guarded region each seed sits in (a basic block
     reached only under the consumer-side branch is consumer, etc.). Blocks reachable from **both** roles
     (shared prologue/epilogue, setup) get a **`common`** label — do not force them into one role.
  5. Emit `{role, region, pc_set}`; `region` (prologue/mainloop/epilogue) is a secondary cut by loop
     back-edge detection, applied within each role.
- **Fallback / validation:** cross-check role assignment against NCU per-PC — consumer PCs should carry the
  tensor/`math_pipe_throttle`/`warpgroup_arrive`-heavy shares, producer PCs the `long_scoreboard`/TMA-arrival
  shares; a mislabeled PC shows up as a role-inconsistent stall profile. Reuse `discover_tma_producers.py`
  for the producer seed set (it already finds TMA-op PCs).

## 10. Comparison statement (apples-to-apples)

Per (role, region): HW `smsp__pcsamp_*` sample counts and GCOM `stall_by_pc` cycle counts each summed into
the **same** regions, **normalized to a reason-share vector**; compare vectors + one divergence number
(cosine/L1). **Do not compare absolute cycles vs samples** — HW statistical, sim exact; only the reason
*distribution* is commensurable. L2/DRAM/tensor aggregates stay kernel-level.

---

## 11. Execution order

Each gating check below is a **concrete experiment with a pass/fail predicate and a named fallback**, not a
prose assertion. Nothing in Stage 2 is built until spike checks (a)–(d) pass.

0. **PREREQUISITE — GCOM stall-metric redesign (P11 / §7.1).** By user decision this is done **first**, before
   the rest of the project. Deliverable = the §7.1 counter-redesign worklist: add `not_selected`,
   `warpgroup_arrive`, `dispatch_stall`, `sleeping` (and decide `branch_resolving`), define `selected` as
   denominator, and fix the many→one folds, so GCOM's taxonomy lines up with NCU's reason set. Validated by
   the §11.2 pre-experiment (shape match to NCU kernel-10). Only after this does the gating spike / Stage-1
   loop proceed.

1. **Gating spike (~1–2 days) — four checks, all must pass:**

   **(a) PC sampling populates. [PASS — verified 2026-07-08]** The existing `.ncu-rep` `--page source`
   yields per-SASS-PC rows with the full stall taxonomy (`stall_long_sb/short_sb/barrier/wait/mio/not_selected/
   wgmma…`) plus `(All)`/`(Not-issued)` twins. Confirmed non-zero on kernels 5 & 10 (e.g. kernel 10:
   `long_scoreboard` 21.9%, `barrier` 17.2%, `not_selected` 5.4%). No re-profiling needed. **FAIL fallback
   (not needed):** CUPTI PC Sampling API, else `clock64()` per-region CPI.

   **(b) PC-space alignment (P6). [PASS — verified 2026-07-08]** Three PC spaces reconcile to a single
   **function-relative offset**:
   - Sim trace-PC = function-relative offset (`enhanced_execution_info.json` shows `pc_num_dec:0` for the
     first instr, with a separate `function_address` base; `trace.m_pc` at
     [trace_driven.cc:258](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L258)).
   - `extra_info/sass` disassembly = function-relative offset (`/*0000*/ LDC R1, c[0x0][0x28]`, 16 B steps).
   - NCU `Address` = runtime VA. Verified: `VA − min(VA)` gives `0x0,0x10,0x20,…` matching the sass offsets
     **instruction-for-instruction** (offset 0 = `LDC R1, c[0x0][0x28]` on both). Mapping:
     `sim_offset == sass_offset == (NCU_VA − function_base_VA)`. Sim↔sass need no transform; NCU needs only
     a per-function base subtraction. Trace kernel dirs are `kernel_1..kernel_11`, matching NCU launch
     indices (so trace `kernel_10` ↔ NCU kernel 10). **FAIL fallback (not needed):** match opcodes at region
     boundaries.

   **(c) CTA-completion sanity (P7, new).** Run FA3 with `-gpgpu_max_completed_cta K` for a small K and
   confirm the log prints `gpu_completed_cta >= K` and that each counted CTA reached its epilogue PCs (not
   truncated). **PASS:** completed-CTA count ≥ K and no in-flight CTA is folded into the normalized stats
   (§4.2 guard active). **FAIL:** fix the stop flag / normalization before any K sweep.

   **(d) Speedup-vs-K + convergence (Stage 1, P4/P10).** Sweep K on the representative sample; plot wall-clock
   speedup and the **K vs K+ΔK** successive normalized divergence (§4), **not** a full-sim divergence.
   **PASS:** there exists a K where successive divergence ≤ tolerance AND speedup ≥ target. **FAIL:** FA3 is
   too contention-bound to subset (converges only near full K); record the achievable speedup and cap K,
   folding residual subset error into the Stage-2 budget.

2. **R2 pre-experiment (cheap, before writing Change B′).** Prove the every-cycle pass is *necessary*, not
   assumed: at kernel level (no regioning, no new code) compare the **existing no-issue-only** GCOM stall
   shares (§7 counters) against NCU's all-cycle `smsp__pcsamp_*` shares. **If they already match within
   tolerance, Change B′ (Gap C) may be unnecessary** — skip or downscope §8.1. **If they diverge, quantify
   the gap** — this both justifies R2's HIGH rating and gives the target the every-cycle pass must close.

3. **Stage 1:** representative CTA-sample selection (allow-list extension if needed), using
   `-gpgpu_max_completed_cta` per §2 → pick smallest faithful K on existing counters. **Delivers the speed goal.**
4. **Stage 2 code (only if step 2 shows a real gap):** Change B′ (Gap-C every-cycle pass + per-PC counts +
   `not_selected`) behind the flag; verify flag-off ⇒ byte-identical baseline stats.
5. `stall_taxonomy.json` + `define_regions.py` (per-role) + `run_ncu_pcsamp.sh`.
6. Parse + compare: kernel-level taxonomy join first (no regioning), then per-(role,region) **K-sim (at the
   converged K) vs full-HW**.

## 12. Risks

- **R1 (HIGH→partially mitigated) — confounded fit:** the two-stage split still separates subset error
  (Stage 1) from model error (Stage 2), but **without full-sim (P10)** the separation is weaker — Stage 2
  compares K-sim, not full-sim, to HW. Residual subset error is bounded first by Stage 1 K-convergence (§1
  note); accepted trade. See R10.
- **R2 (HIGH) — Gap C population mismatch:** the every-cycle per-warp pass (§8.1) is required; without it,
  GCOM and NCU stall populations differ and shares are not comparable. **De-risk:** the §11.2 pre-experiment
  measures the gap at kernel level *before* writing any C++ — if the no-issue-only shares already match NCU,
  Change B′ is downscoped or skipped.
- **R3 (MED) — speedup may be small (P4):** FA3 is memory/mbarrier-bound; faithful K may be near-full.
  Quantify in spike 1(d) before building Stage 2.
- **R4 (resolved, P6) — PC alignment:** **verified in spike 1(b)** — sim trace-PC and sass are the same
  function-relative offset; NCU VA maps by a per-function base subtraction (instruction-for-instruction
  match confirmed). No open risk.
- **R5 (MED→escalated, needs GCOM metric changes) — GCOM↔NCU reason mismatch:** the mapping is not just
  many-to-many; spike 1(a) shows NCU exposes reasons GCOM has **no clean counterpart for**
  (`dispatch_stall` 4–11%, `sleeping`, `branch_resolving`) and a non-stall bucket (`selected` 13–15%,
  = successfully-issued, must be excluded from the stall denominator). **Decision (user):** rather than only
  coarse-grouping, **fix/redesign the GCOM stall counters** so its taxonomy lines up with NCU's — i.e. add
  `not_selected` (P2/Gap C) *and* re-derive counters for the currently-unmatched NCU reasons. This makes R5
  part of the Change-B′ metric redesign (§7/§8), not a post-hoc grouping. A coarse grouped axis (memory-dep /
  pipe-busy / sync-barrier / frontend / compute-dep / not-selected) remains the fallback if some reasons
  stay fundamentally unmappable.
- **R6 (resolved) — head-PC vs blocking-PC:** **verified** — the issue pass evaluates each warp's head
  instruction `pI = get_IBuffer_remodeled()->next_inst()`
  ([subcore.cc:554](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L554))
  and all stall conditions are tested on that head `pI`, so Change B′ keying on `pI->pc` matches NCU's
  head-PC attribution. No open risk.
- **R7 (LOW) — same PC, different dynamic visit:** PC-region bucketing merges all loop visits on both sides
  ⇒ consistent (region = steady-state, not one iteration).
- **R8 (HIGH→mitigated, new P7) — CTA truncation:** `-gpgpu_max_cta` stops on *issued* not *completed* CTAs
  ([gpu-sim.cc:3138-3140](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3138-L3140)),
  truncating the last wave and reintroducing the epilogue-loss the plan rejected instruction-windows for.
  **Mitigated** by switching to `-gpgpu_max_completed_cta` + the §4.2 in-flight-CTA exclusion guard;
  **verified** in spike 1(c).
- **R9 (LOW→resolved, new P8) — no per-CTA HW metric:** NCU gives per-PC but not per-(PC×CTA)
  (verified: only launch props / grid sums). **Resolved by structure** (§6): the sim↔HW comparison (Stage 2)
  is per-PC kernel-summed, which NCU provides; per-CTA is only used sim↔sim (Stage 1), where the simulator
  supplies it on both sides. No open risk.
- **R10 (MED, new P10) — routine loop lacks a full-sim baseline:** by default Stage 1 proves subset error is
  *converged* (K vs K+ΔK stable) but not *zero*, and routine Stage 2 compares K-sim to HW. Mitigation:
  bound residual subset error via K-convergence, and run the **optional full-sim anchor** (§4 step 4) when a
  clean subset/model separation is needed. If FA3 only converges near full K (R3), the speed goal itself is
  limited. Measured in spike 1(d).
- **R11 (resolved, new P9) — intra-kernel phased run infeasible:** verified no microarch-state
  checkpoint/restore exists and mid-phase start deadlocks. **Resolved** by dropping phased run entirely and
  using CTA subsetting (spatial) as the sole time-cut axis (§0). No open risk.
- **R12 (HIGH, new P11) — GCOM taxonomy is incomplete vs NCU, requires metric redesign:** spike 1(a) exposed
  NCU reasons with no GCOM counterpart (`dispatch_stall`, `sleeping`, `branch_resolving`) and the `selected`
  non-stall bucket. **Decision (user):** the fix is to **redesign the GCOM stall metrics** so the simulator's
  own taxonomy is re-derived to match NCU's reason set — done together with the Gap-C every-cycle pass and
  `not_selected` addition in Change B′ (§8). Scope: for each unmatched NCU reason, decide whether GCOM can
  produce an equivalent signal from existing microarch state, add the counter, and validate its share
  against NCU per-PC. This is larger than "add `not_selected`"; it is a taxonomy alignment pass. Gated/sized
  by the §11.2 pre-experiment. Overlaps R5.
- **R13 (MED→specified, new P12) — warp-role region identification:** FA3 is warp-specialized, so "mainloop"
  is disjoint producer/consumer PC sets, not a linear `[lo,hi]`. **Design done (§9, 2026-07-08):** seed roles
  from measured SASS tags (`HGMMA*`/`WARPGROUP*`→consumer, `UTMALDG`/`UTMACCTL`/`UTMASTG`→producer),
  propagate along the intra-function CFG from the `SR_TID.X`→`ISETP`→`@P BRA` split, `common` for
  both-reachable blocks. Remaining: implement `define_regions.py` + validate role labels against NCU per-PC.
  Kernel-level join (§11.6) works without it.

## 13. P1–P12 resolution map

Origin of the two-stage redesign: the six problems found while stress-testing the prior single-fit plan,
plus P7 (CTA-completion) and P8 (per-CTA HW asymmetry) from the code-verification passes, P9 (phased-run
infeasibility) + P10 (no full-sim baseline) from the 2026-07-08 objective clarification, and P11 (GCOM
taxonomy incomplete) + P12 (warp-role region identification) from the spike-1(a) reason analysis.

| # | Problem | Resolved in | How | Residual risk |
|---|---|---|---|---|
| **P1** | K-sim-vs-full-HW fit confounds *subset error* + *model error* | Structure (§1) → Stage 1 & 2 split | Stage 1 = K vs K+ΔK convergence (subset error, **no full-sim** per P10); Stage 2 = K-sim@converged-K vs full-HW (model error) | R1 (partial — P10 weakens it) |
| **P2** | GCOM logs stalls only on **no-issue** cycles; NCU samples **all** cycles/warps (+ missing `not_selected`) | Stage 2 (§7 Gap C, §8.1); §11.2 pre-experiment | New **every-cycle per-warp stall pass**, decoupled from issue-loop `break`; adds `not_selected`. Necessity first proven by §11.2 kernel-level gap measurement | R2 (HIGH — measured before build) |
| **P3** | CTA-0-only default, but CTA-0 is causally biased | Stage 1 (§3) | CTA-0-only demoted to **bring-up only**; comparison uses a **representative CTA sample** | — |
| **P4** | Interesting FA3 stalls are contention-bound → forces large K → small speedup | Stage 1 (§4) + spike 1(d) | States speedup is contention-bounded; **quantify in spike before Stage 2** | R3 (MED — measure) |
| **P5** | Linear prologue/mainloop/epilogue regions don't fit warp-specialized FA3 | Stage 2 (§9) | `define_regions.py` emits **PC sets per warp-role** (producer/consumer), not `[lo,hi]` | — |
| **P6** | NCU SASS-PC ↔ sim trace-PC alignment assumed | Gating spike 1(b) **[PASS]** | Verified: sim trace-PC == sass offset == (NCU_VA − function_base); instruction-for-instruction match | R4 (resolved) |
| **P7** | `-gpgpu_max_cta` stops on *issued* CTAs ⇒ last wave truncated mid-lifetime (CTA-axis epilogue loss) | §2 fix + spike 1(c) | Use `-gpgpu_max_completed_cta` (keys on `gpu_completed_cta`) + exclude in-flight CTAs from normalization | R8 (mitigated — verify) |
| **P8** | Can CTA-sampled sim be compared to HW, given NCU has no per-CTA metric? + how to actually sample CTAs | §6 (asymmetry) + §3 (impl path) | HW compared only per-PC to sim; CTA sampling only sim↔sim. Sampling = `-gpgpu_max_completed_cta` → CTA-id allow-list (checkpoint-gate precedent) → per-CTA dump | R9 (resolved) |
| **P9** | "phased run" = arbitrary mid-kernel sub-phase — assumed feasible | §0 | **Verified infeasible**: no microarch-state checkpoint; mid-phase FA3 deadlocks. CTA (spatial) is the only working time-cut axis | R11 (resolved) |
| **P10** | Stage 1 calibrated against a 12 h full-sim — contradicts the "iterate faster" objective | §0, §1, §4 | Full-sim **optional not routine**: cheap **K vs K+ΔK convergence** every iteration + **occasional full-sim anchor** on demand | R10 (MED — measure) |
| **P11** | GCOM stall taxonomy incomplete vs NCU (`dispatch_stall`/`sleeping`/`branch_resolving` unmatched; `selected` non-stall) | Change B′ metric redesign (§7/§8) | Redesign GCOM counters to match NCU's reason set (not just group); validate per-PC share; sized by §11.2 | R12 (HIGH — open) |
| **P12** | Warp-role regions (§9) need automatic role labeling per PC | §9 (design done) + `define_regions.py` | Seed from measured SASS tags (`HGMMA*`/`WARPGROUP*`→consumer, `UTMALDG*`→producer), propagate along CFG from `SR_TID.X` branch; `common` for shared blocks | R13 (specified — implement) |

**Kernel identity (confirmed):** in launch order, **kernel 5 = FA3 forward** (`FlashAttnFwdSm90`, sass
`flash_fwd_hdim64_256_bf16_sm90`), **kernel 10 = FA3 backward** (`FlashAttnBwdSm90`, sass
`flash_bwd_hdim64_bf16_softcapall_sm90`). Trace dir `kernel_10` ↔ NCU kernel 10.

**Stage-level view:**
- **Objective (§0):** cut wall-clock to debug the sim↔HW gap. Phased (intra-kernel) run infeasible (P9);
  CTA subsetting is the mechanism; full-sim baseline optional, not routine (P10).
- **Gating spike (§11.1)** clears feasibility: 1(a) PC-sampling exists **[PASS]**, 1(b) P6 alignment
  **[PASS]**, 1(c) P7 CTA-completion, 1(d) P4/P10 speedup + K-convergence. Remaining **two** (1c, 1d) must
  pass before Stage 2 code.
- **§11.2 pre-experiment** measures the R2 gap at kernel level before any C++ — gates whether Change B′ is
  built at all, and sizes the P11 taxonomy redesign.
- **Stage 1** (speed, no C++ for tier-1; small gate for tier-2) resolves P3, P7, P8, P10; quantifies P4.
- **Stage 2** (the `subcore.cc` change) resolves P2, P5, **P11**; per-(role,region) needs **P12**; justified
  only after spike 1(d) and §11.2.
- **P1** is resolved structurally — it is why the two stages exist.
- **Resolved by verification:** P6 (R4), P8 (R9), P9 (R11), and R6 (head-PC) — no open risk.
