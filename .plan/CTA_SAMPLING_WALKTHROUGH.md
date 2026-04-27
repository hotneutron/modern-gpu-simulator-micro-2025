# CTA Sampling — Branch Walkthrough

A concise tour of the work on the `cta-sampling` branch, starting from
`main`. Pairs with `CTA_SAMPLING_PHASE_AB_DONE.md` for the technical
details; this doc is the story.

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
| `N_sat_est` + initial target | Picks a starting `sim_ctas` count based on classification + saturation estimate. |
| Stratified-shuffle replication | When the K-rep heuristic returns fewer reps than the target, replicate them with stratified shuffle to fill more SMs. |
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

- **Per-class instruction counters** in pressure signals: FP, INT,
  DP, TC, SFU, loads, stores. Previously only `gpu_sim_insn`.
- **Refined `kernel_ai`**:
  ```
  compute_ops = FP + INT + W_dp·DP + W_tc·TC + W_sfu·SFU
  kernel_ai   = compute_ops / dram_bytes
  ```
  Knob-tunable weights (`-cta_sampling_ai_w_dp/tc/sfu`).
- **Memory-stall fraction**: `mem_stall_frac = stall_l1c_cycles /
  issue_eval_cycles`. Required re-enabling a counter increment that
  was commented out in `subcore.cc`.
- **Classifier consumes `mem_stall_frac`** as a third memory-pressure
  signal alongside `achieved_bw_ratio` and `dram_queue_occupancy`.
- **Validation harness** gains a `pilot+refined` mode and 3 new
  workloads (bfs, srad_v2, lud) that the old AI proxy misclassified.

**Result:** bfs gets `kernel_ai=0.19`, `mem_stall_frac=0.36`,
classifies as MEMORY (was COMPUTE), and the pilot expands it
correctly. `insn_err%` doesn't regress on previously-correct
workloads.

---

## 4. Phase B — whole-kernel cycle estimator scaffold (3 commits)

- **Plumb a `last_kernel_wave_info_t` struct** from the pilot's accept
  path through `gpgpu_sim` so `print_stats()` can see what the pilot
  decided.
- **Add the estimator + new stats** (existing `gpu_tot_sim_cycle`
  deliberately stays as the sampled-wave wall-clock):
  ```
  gpu_tot_sim_cycle_estimated         = ...   whole-grid projection
  gpu_tot_sim_cycle_estimation_mode   = ...   which model fired
  gpu_tot_ipc_estimated               = ...
  ```
- **Validation harness** captures the new stats and prints both raw
  and estimated cycles side by side.

The first formula was the textbook one — `cyc_per_round × ceil(total_ctas / total_sms)` for compute, steady-state wave count for memory. Structurally correct but **hotspot still showed +43%** because `ceil` rounds partial waves up to a full round.

---

## 5. The accuracy push (4 commits)

This is where most of the interesting work happened.

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
kernel shows a memory-stall signal. Pilot then expands at least once,
capturing more CTAs/SM concurrency.

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
CTAs-per-SM densities. We just have to use them:

```
T(N) = a + b × log(N+1)        where N = CTAs per SM
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
do.

---

## 7. Where we stand

**Shipped on `cta-sampling`** (29 commits ahead of `main`):

| Theme | Commits |
|---|---|
| Foundation (sampling, classifier, pilot, harness) | 11 |
| Phase A — refine classifier signals | 5 |
| Phase B — cycle estimator scaffold | 3 |
| Accuracy push (4 fixes + 4 doc updates) | 8 |
| Wider validation + cleanup | 2 |

**Targets met on the original 6-workload set: p50 < 15%, p90 < 25%.**

**Wider 10-workload set has 3 documented failure modes** with named
follow-ups in `CTA_SAMPLING_PHASE_AB_DONE.md`:

1. Deeper pilot expansion for high-projection kernels (nn).
2. K-rep clustering replacement (heartwall, generalizes).
3. Skip-sampling threshold for tiny kernels (nw).
4. AI weight calibration against a known-FLOPs kernel (open D2).
5. Pilot-rejected-iteration cleanup of per-SM aggregates (only
   matters if IPC numbers become a primary signal).

---

## 8. One-slide summary

> **CTA sampling on `cta-sampling`: from "every CTA simulated" to
> "sampled wave + ±17% cycle projection" on 6 representative
> workloads. Three accuracy mechanisms (throughput conservation,
> force-expand on K-rep undersampling, log-fit concurrency model)
> generalize cleanly when the sample is representative. Wider
> 10-workload sweep identifies 3 failure modes, each with a named
> scoped follow-up.**
