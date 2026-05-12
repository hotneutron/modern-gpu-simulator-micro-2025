# CTA Sampling — History 1: Foundation, Phase A/B, and Accuracy Push

**Date snapshot:** 2026-04-30
**Branch state at end of stage:** 29 commits ahead of `main`.
**Outcome:** Cycle-error targets met on the original 6-workload sweep
(p50 = 12.3%, p90 = 17.3%). A subsequent wider 10-workload sweep
revealed three new failure modes that became the input to History 2.

This file is the historical record of the first half of the
`cta-sampling` branch. It is derived from the 2026-04-30 walkthrough
(`20260430_CTA_SAMPLING_WALKTHROUGH.md`) with supplemental context
folded in from `CTA_SAMPLING_PHASE_AB_DONE.md`,
`CTA_SAMPLING_PLAN.md`, and `CTA_SAMPLING_STATUS.md` where it makes
the narrative readable on its own.

---

## 1. The problem

On `main`, the simulator replays **every CTA of every kernel** in a
trace. A kernel with 938 CTAs runs all 938 sequentially in our
trace-driven front-end, even though most CTAs do redundant work. Big
traces take hours.

**Goal:** sample a representative subset of CTAs, simulate just those,
and project a whole-kernel cycle count from the sampled wave —
within ±15% / ±25% (p50 / p90) of a full-grid baseline.

Two questions to answer:
1. **Which CTAs to sample?** A bad sample biases everything downstream.
2. **How to project to the full kernel?** Sampled cycles aren't full
   cycles; the formula matters.

### Design-space context (folded in from `CTA_SAMPLING_PLAN.md`)

Two original approaches were considered:

- **Coordinate-based heuristic (Approach 1, chosen).** For regular
  compute kernels (GEMM, convolution, stencil), CTAs doing identical
  work are identifiable by grid position; the dominant CTA
  heterogeneity is boundary effects. Sampling rule: corners + edge
  midpoints + interior → K≈9 regardless of grid size.
- **Two-pass clustering (Approach 2, deferred).** Pass 1 collects
  per-CTA feature vectors via Nsight Compute counters and clusters
  with K-Means; pass 2 NVBit-traces only the K representatives. Not
  implemented — would require significant new infrastructure.

A third evaluated direction — a **single-pass conservative SM-sizing
variant** (`compute→35%, mixed→55%, memory→75%`) — was set aside in
favor of the adaptive pilot loop described below.

The work in this file uses Approach 1 plus an adaptive pilot loop
that addresses the **occupancy-collapse problem** (a naive K-rep
simulation on 9 CTAs leaves most SMs idle, degrading latency hiding
and missing inter-CTA contention).

---

## 2. Foundation (first 11 commits)

The first chunk of work built the basic sampling infrastructure on
top of `main`.

| Step | What it added |
|---|---|
| Initial sampling | `-cta_sampling_mode 1` knob; runtime selects a small set of CTAs and scales stats by `total/sampled`. |
| Stat scaling fix | Leave cycles unscaled; only scale insn / CTA-count by sampling weight. |
| Pressure signals | Per-kernel snapshot of `gpu_sim_insn`, `dram_bytes`, `l2_misses`, `achieved_bw_ratio`, `dram_queue_occupancy`. |
| 3-way roofline classifier | `compute / memory / mixed` from `kernel_ai` + memory pressure. Knob-tunable thresholds. |
| `N_sat_est` + initial target | Estimates the concurrency (CTAs/SM) needed to saturate the GPU for each class. Uses this to set the initial `sim_ctas` target: `target ≈ min(total_ctas, total_sms × N_sat_est)`. Ensures the sample is large enough to exercise realistic SM occupancy. |
| Stratified-shuffle replication | When K-rep collapses to fewer unique CTAs than the target (e.g., grid corners on a 1D grid), duplicates the representative CTAs and shuffles them across SMs to fill the target occupancy while spreading similar CTAs rather than clumping them. Trades diversity for SM occupancy. |
| Adaptive pilot loop | Re-launches the kernel with doubled `sim_ctas` if pressure signals haven't stabilized. Snapshot/restore so rejected iters don't pollute totals. |
| Validation harness | `validate.py` runs each workload across 4 sampling modes (`baseline`, `K-rep`, `expanded40`, `pilot`) and reports cycle/insn/IPC error vs. baseline. |

**End-of-foundation state:** sampling mechanically works, but cycle
accuracy hadn't been pushed. Hotspot showed **+43% cycle error** vs.
baseline because the cycle estimator used a `ceil`-based per-CTA
formula that over-counted partial waves. Several memory-bound kernels
(bfs, srad_v2, lud) were misclassified as compute because
`kernel_ai = gpu_sim_insn / dram_bytes` counts memory and control-flow
ops as compute.

---

## 3. Phase A — refine the classifier (5 commits)

Tightened what the classifier sees so it stops mis-routing
memory-bound kernels.

| # | Hash      | Subject |
|---|-----------|---------|
| A1a | `e8f9895` | per-class instruction counters in pressure signals |
| A1b | `18581f9` | refined `kernel_ai` with AI weight knobs |
| A2  | `852d765` | memory-stall fraction in pressure signals |
| A3  | `ad055b0` | classifier consumes `mem_stall_frac` |
| A4  | `c8a9175` | validation harness consumes refined classifier signals |

What changed:

- **Per-class instruction counters** in pressure signals: FP, INT,
  DP, TC, SFU, loads, stores. Previously only `gpu_sim_insn`. The
  first five sum across SMs from the legacy `shader_core_stats` POD
  arrays; load/store come from each SM's `m_sm_stats` via a new
  `read_sm_stat_value` virtual on `shader_core_ctx_wrapper` and a
  `sum_sm_stat_value` helper on the cluster + gpgpu_sim. This avoids
  the gather-side double-counting that affects stats registered with
  `is_erase_after_gather_in_sm=false`.
- **Refined `kernel_ai`**:
  ```
  compute_ops = FP + INT + W_dp·DP + W_tc·TC + W_sfu·SFU
  kernel_ai   = compute_ops / dram_bytes
  ```
  Knob-tunable weights: `-cta_sampling_ai_w_dp` (2.0),
  `-cta_sampling_ai_w_tc` (8.0), `-cta_sampling_ai_w_sfu` (4.0).
  Replaces the `gpu_sim_insn / dram_bytes` proxy that included memory
  and control-flow ops in the FLOP count.
- **Memory-stall fraction**: `mem_stall_frac = stall_l1c_cycles /
  issue_eval_cycles`. Required re-enabling
  `is_any_waiting_l1c` and adding the counter increment in
  `subcore.cc` (it had been commented out previously).
- **Classifier consumes `mem_stall_frac`** as a third memory-pressure
  signal alongside `achieved_bw_ratio` and `dram_queue_occupancy`.
  Memory pressure now fires on **any of**: `achieved_bw_ratio >=
  pressure_bw`, `dram_queue_occupancy >= pressure_queue`, OR
  `mem_stall_frac >= pressure_mstall`. New knob
  `-cta_sampling_pressure_mstall` (default 0.4).
- **Validation harness** gains a `pilot+refined` mode and 3 new
  workloads (bfs, srad_v2, lud) that the old AI proxy misclassified.

**Result:** bfs gets `kernel_ai=0.19`, `mem_stall_frac=0.36`,
classifies as MEMORY (was COMPUTE), and the pilot expands it
correctly. `insn_err%` doesn't regress on previously-correct
workloads.

---

## 4. Phase B — whole-kernel cycle estimator scaffold (3 commits)

| # | Hash      | Subject |
|---|-----------|---------|
| B1  | `f27982c` | plumb wave-info struct for whole-kernel cycle estimation |
| B2  | `9aa26fa` | whole-kernel cycle estimator + new stat lines |
| B3  | `c99bbe7` | validation harness reports estimated cycles |

- **Plumb a `last_kernel_wave_info_t` struct** from the pilot's accept
  path through `gpgpu_sim` so `print_stats()` can see what the pilot
  decided. New `set_last_kernel_wave_info` setter on `gpgpu_sim`.
- **Add the estimator + new stats** (existing `gpu_tot_sim_cycle`
  deliberately stays as the sampled-wave wall-clock):
  ```
  gpu_tot_sim_cycle_estimated         = ...   whole-grid projection
  gpu_tot_sim_cycle_estimation_mode   = ...   which model fired
  gpu_tot_ipc_estimated               = ...
  ```
- **Two original estimation modes**, auto-selected by class:
  - `per_cta` for COMPUTE: `cyc_per_cta = sampled_cycles /
    rounds_per_sm_sampled`, projected by `ceil(total_ctas /
    total_sms)`. (The plan's literal formula `sampled_cycles /
    sampled_ctas` is wrong for parallel CTAs — corrected to use
    rounds-per-SM, which matches the actual sampled-wave wall-clock.)
  - `steady_state` for MEMORY/MIXED: `sampled_cycles * N_waves_full /
    N_waves_sample`.
- **Validation harness** captures the new stats and prints both raw
  and estimated cycles side by side, plus `ESTIMATION_MODE_PAT`.

The first formula was structurally correct but **hotspot still showed
+43%** because `ceil` rounds partial waves up to a full round. That
gap is what the accuracy push closes.

---

## 5. The accuracy push (4 commits)

This is where most of the interesting work happened.

| Hash      | Subject |
|-----------|---------|
| `8af3c98` | throughput-conservation cycle formula |
| `c3204ec` | force-expand when K-rep undersamples + mem-stall fires |
| `5ea3114` | log-fit concurrency-throughput model from pilot history |
| `d27f968` | extend force-expand to high projection-ratio kernels |

### 5a. Throughput-conservation formula

The old per-CTA model assumed each round of the full grid takes the
same wall-clock as the sampled wave. Partial waves run faster: CTAs
finish at staggered times, freeing SMs early. Replaced both branches
with a single throughput-conservation formula:

```
sampled_active = min(sampled_ctas, total_sms)
full_active    = min(total_ctas,   total_sms)
scale          = (total_ctas / full_active)
               / (sampled_ctas / sampled_active)
est_cycles     = sampled_cycles × scale
```

`active_sms` is capped at `total_sms`, *not*
`total_sms × max_cta_per_core` — multiple resident CTAs per SM share
the SM's issue bandwidth, so they don't add parallelism.

| Workload | Before | After |
|---|---|---|
| hotspot | +43.4 | +14.7 |
| srad_v2 | +37.0 | +17.3 |
| backprop | +108 | +90.2 (still bad) |

p50 met, p90 still off — backprop remains the outlier.

### 5b. Force-expansion when K-rep undersamples

backprop's grid is 256×1×1. The corner+midpoint heuristic *collapses*
on 1D grids — only 3 unique CTAs. Pilot accepted the 3-CTA sample at
iter 0 because it classified COMPUTE; the throughput formula
extrapolated 3 CTAs to 256.

Fix: refuse to accept iter 0 when the sample is small *and* the
kernel shows a memory-stall signal (`undersized && mem_stall_frac >
0.10`). Pilot then expands at least once, capturing more CTAs/SM
concurrency.

```
backprop  +90.2 → +30.5
```

p90 still over (30 vs 25), but 3× better.

### 5c. Log-fit concurrency-throughput model

The remaining residual comes from physics, not the formula. Per-SM
throughput grows nonlinearly with concurrent CTAs/SM —
memory-latency hiding scales sub-linearly. The throughput formula
assumed per-SM throughput is constant; it's not.

The pilot already produces multiple iterations at different
CTAs-per-SM densities. Just have to use them:

```
T(N) = a + b × log(N + 1)        where N = CTAs per SM
```

- Each pilot iteration's `(sampled_ctas, sampled_cycles, active_sms)`
  is recorded in `pilot_state_t.history`.
- At accept time, fit `(a, b)` by least squares over
  per-density-aggregated points.
- `print_stats()` extrapolates `T(full_ctas_per_sm)` and computes
  `est_cycles = total_ctas / (full_active × T_full)`.
- Skips the fit cleanly when there are <2 distinct densities or the
  slope is non-positive — falls back to constant-throughput.

| Workload | Estimation mode | err% |
|---|---|---|
| hotspot | throughput_compute (1 iter, no fit) | +14.7 |
| backprop | log_fit_compute (4 distinct densities) | **−15.4** |
| pathfinder | throughput_memory | −2.6 |
| bfs | throughput_mixed | −9.9 |
| srad_v2 | throughput_mixed (sampled==total) | +17.3 |
| lud | throughput_compute (small grid) | −0.1 |

**p50 = 12.3% (target <15%, met). p90 = 17.3% (target <25%, met.)**

### 5d. High-projection-ratio force-expand

Discovered during wider validation. The mem-stall heuristic missed nn
— its kernels classify COMPUTE with `mem_stall_frac=0` but still see
massive concurrency-throughput swings (1 CTA/SM sample → 23 CTAs/SM
full). Latency hiding matters even when the latency isn't memory.

Generalized the force-expand condition to also fire when
`full_ctas_per_sm > 4 × sampled_ctas_per_sm`, regardless of stall
signal. Spares hotspot (1.6×) and pathfinder (1×); catches backprop
(6.4×) and nn (23×).

---

## 6. Wider validation — 4 more workloads

After the original 6 hit both targets, added the rest of
rodinia2/Turing — heartwall, nn, nw, streamcluster — to test
generalization.

```
              | original 6 (met)  | wider 10 (3 fail)
              |                   |
hotspot       | +14.7             | +14.7
backprop      | -15.4             | -15.4
pathfinder    |  -2.6             |  -2.6
bfs           |  -9.9             |  -9.9
srad_v2       | +17.3             | +17.3
lud           |  -0.1             |  -0.1
heartwall     |   --              | -28.7  ← K-rep replication
nn            |   --              | +103.5 ← log-fit under-extrapolates
nw            |   --              | +56.8  ← pilot overhead, tiny grids
streamcluster |   --              |  -1.5
              |                   |
p50           | 12.3% (met)       | 15.0% (just over)
p90           | 17.3% (met)       | 56.8% (over)
```

**Three distinct failure modes**, each genuinely different:

1. **heartwall (−29%)** — K-rep collapses to 3 corner CTAs on a
   51×1×1 grid. Pilot expands by *replicating* those corners, so
   "full grid sample" runs 51 instances of 3 corner CTAs instead of
   51 unique CTAs. Sampling-quality issue; no formula tweak fixes it.
2. **nn (+103%)** — 938 CTAs/kernel; pilot reaches 80 sampled
   (2 CTAs/SM) but full grid is at 23 CTAs/SM. The log-fit
   extrapolation from N=2 to N=23 is too far past the sampled range
   and under-shoots actual throughput growth ~2×.
3. **nw (+57%)** — 1–8 CTA kernels. Pilot's iter-0-reject +
   iter-1-accept doubles wall-time, plus cache-state pollution from
   iter 0 makes iter 1 itself slower than a clean baseline. **Negative
   speedup** regime — pilot shouldn't run at all on grids this small.

The honest scientific finding: the model **generalizes within the
6-workload regime** (medium grids, well-behaved K-rep, reasonable
log-fit extrapolation distance) and **breaks** when those assumptions
don't hold.

Two longer-tail concerns flagged but not yet addressed:
- Shared memory contention.
- I-cache misses.

---

## 7. Files touched in this stage

| Path | Touched in |
|---|---|
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h` | A1a, A1b, A2, B1, B2, log-fit |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc` | A1a, A1b, A2, B2, throughput-fix, log-fit |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h` | A1a |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader_core_wrapper.h` | A1a |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h` | A1a |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc` | A2 |
| `simulator-remodeled/gpu-simulator/trace-driven/trace_driven.h` | A1b, A3 |
| `simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc` | A1b, A3 |
| `simulator-remodeled/gpu-simulator/main.cc` | A1b, A3, B1, force-expand, log-fit, ratio-expand |
| `simulator-remodeled/util/cta_sampling/validate.py` | A4, B3, wider-validation |

---

## 8. Open follow-ups at end of this stage (priority-ranked)

These are the items that fed into History 2.

1. **Deeper pilot expansion for high-projection kernels** (nn). Two
   options: (a) adaptively bump `pilot_max_doublings` when the
   force-expand condition fires (sample at 4+ CTAs/SM gives the log
   fit a wider range); (b) try a different functional form (e.g.,
   power law `T(N) = a*N^b` or saturating `T(N) = T_max*(1-exp(-N/τ))`)
   though both need ≥3 distinct densities to fit reliably.
2. **K-rep clustering replacement** (heartwall). Replace the
   corner+midpoint heuristic with per-CTA feature clustering so the
   sampled set covers the actual work distribution rather than just
   grid corners.
3. **Skip-sampling threshold for tiny kernels** (nw). When
   `total_ctas < some_threshold` (e.g., `< 2 × total_sms`) the pilot
   overhead is larger than any sampling speedup; cache-state
   pollution between rejected and accepted iters makes the accepted
   sample slower than a clean baseline run.
4. **AI weight calibration** (open decision D2): fit `W_dp`, `W_tc`,
   `W_sfu` against a known-FLOPs kernel (a tiled GEMM is ideal)
   instead of the 2/8/4 guesses. Tangential to cycle accuracy now
   that the formula is class-independent.
5. **Pilot-rejected iteration cleanup**: `pilot_restore` undoes
   `gpu_tot_sim_*` but does not roll back per-SM aggregates
   (`m_gpu_per_sm_stats`, legacy `shader_core_stats` POD arrays,
   per-SM `m_sm_stats` for stall counters). Pollutes
   `gpu_tot_ipc_estimated` numerator but **not** cycle estimate.
6. **TMA modeling**. FlashAttention-3 uses H100 TMA
   (`UTMALDG.4D` / `UTMASTG.*`) which bypasses standard memory
   pipelines. Memory pressure signals don't capture TMA transaction
   behavior. Discovered post-implementation; see `TMA_TRACING.md`.

---

## 9. Acknowledged limitations at end of this stage

1. **AI proxy is crude** — `kernel_ai = compute_ops / dram_bytes`
   still over-counts. Fine-grained ALU / load-store / tensor counter
   separation is a future improvement.
2. **Pilot loop requires `window_size == 1`** — auto-disables for
   concurrent kernels.
3. **K-rep collapses on 1D grids** — corner+midpoint heuristic gives
   only 2–3 unique CTAs on 256×1×1 grids. Force-expand mitigates by
   adding pilot iterations but sampling quality stays unrepresentative
   for some kernels.
4. **The log-fit form has no physical basis.** `T(N) = a + b·log(N+1)`
   is empirical with no hardware-limit asymptote. It works inside the
   sampled range but extrapolates unboundedly past it — which is what
   bit nn. This limitation is what triggers the debate in
   `CTA_SAMPLING_Debate_Log_Fit.md` and the subsequent log-fit
   replacement work.

---

## 10. One-slide summary

> **CTA sampling on `cta-sampling`: from "every CTA simulated" to
> "sampled wave + ±17% cycle projection" on 6 representative
> workloads. Three accuracy mechanisms (throughput conservation,
> force-expand on K-rep undersampling, log-fit concurrency model)
> generalize cleanly when the sample is representative. Wider
> 10-workload sweep identifies 3 failure modes (heartwall, nn, nw),
> each with a named scoped follow-up — addressed in History 2.**
