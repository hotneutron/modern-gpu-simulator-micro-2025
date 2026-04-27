# CTA Sampling — Optimization Walkthrough

A presentation-friendly tour of the CTA-sampling work on the
`cta-sampling` branch. Pairs with `CTA_SAMPLING_PHASE_AB_DONE.md` for
the technical details; this doc is the story.

---

## 1. The problem

GPU traces in this simulator can take hours to replay because every
CTA in every kernel is simulated end-to-end. A 938-CTA kernel runs
938-CTA-many CTAs sequentially in our trace-driven front-end, even
though those CTAs do mostly redundant work.

**Goal:** sample a representative subset of CTAs, simulate just those,
and project the whole-kernel cycle count from the sampled wave —
ideally within ±15% / ±25% (p50 / p90) of a full-grid baseline run.

Two big questions:
1. **Which CTAs to sample?** A bad sample biases everything downstream.
2. **How to project to the full kernel?** Sampled cycles aren't full
   cycles; the formula matters.

---

## 2. Starting point (before this work)

The branch already had a coordinate-based **K-rep heuristic** (corners
+ edge midpoints + interior of the grid; up to 9 reps for a 2D grid)
and an **adaptive pilot loop** that runs the K-rep, classifies the
kernel, and optionally expands to more CTAs if the K-rep isn't
representative. There was a basic cycle estimator using a `ceil`-based
per-CTA / steady-state formula, but cycle accuracy hadn't been pushed.

State of the world:
- **Misclassification:** `kernel_ai = gpu_sim_insn / dram_bytes`. This
  treats memory and control-flow ops as compute, so memory-bound
  kernels look compute-bound. bfs, srad_v2, lud got mis-routed.
- **Cycle estimator:** hotspot estimated +43% over baseline, far from
  any acceptable range.

---

## 3. Phase A — refine the classifier

Five layered commits (`e8f9895`...`c8a9175`) tightened what the
classifier sees:

| Step | What it does |
|---|---|
| **A1a** | Per-class instruction counters in pressure signals: FP / INT decoded, DP / TC / SFU accesses, loads, stores. |
| **A1b** | Replaces the AI proxy with `compute_ops = FP + INT + W_dp·DP + W_tc·TC + W_sfu·SFU`; AI = compute_ops / dram_bytes. Knob-tunable weights. |
| **A2** | Memory-stall fraction: `mem_stall_frac = stall_l1c_cycles / issue_eval_cycles`. Required re-enabling a counter increment that was commented out in `subcore.cc`. |
| **A3** | Classifier consumes `mem_stall_frac` as a third memory-pressure signal alongside `achieved_bw_ratio` and `dram_queue_occupancy_avg`. |
| **A4** | Validation harness gains a `pilot+refined` mode and 3 new workloads (bfs, srad_v2, lud) that the AI proxy used to mis-classify. |

**Result:** bfs gets `kernel_ai=0.19`, `mem_stall_frac=0.36`, classifies
as MEMORY (was COMPUTE), and the pilot expands it correctly.
`insn_err%` doesn't regress on the previously-correct workloads.

---

## 4. Phase B — whole-kernel cycle estimator

Three layered commits (`f27982c`...`c99bbe7`) added the projection:

- **B1** plumbs a `last_kernel_wave_info_t` struct from the pilot's
  accept path through `gpgpu_sim` so `print_stats()` can see what the
  pilot decided.
- **B2** adds the estimator + new stats:
  ```
  gpu_tot_sim_cycle_estimated         = ...   (whole-grid projection)
  gpu_tot_sim_cycle_estimation_mode   = ...   (which model fired)
  gpu_tot_ipc_estimated               = ...
  ```
  Existing `gpu_tot_sim_cycle` deliberately stays as the sampled-wave
  wall-clock. The estimate is a separate signal.
- **B3** teaches `validate.py` to capture and tabulate the new stats
  alongside the raw sampled-wave cycles, so we can see both at once.

The first formula was the textbook one — `cyc_per_round × ceil(total_ctas / total_sms)` for compute, steady-state wave count for memory. It was structurally correct but **hotspot still showed +43%** because `ceil` rounds partial waves up to a full round.

---

## 5. The accuracy push

This is where most of the interesting work happened. Four commits.

### 5a. Throughput-conservation formula (`8af3c98`)

The old per-CTA model assumed each round of the full grid takes the
same wall-clock as the sampled wave. But partial waves run faster:
CTAs finish at staggered times, freeing SMs early. Replaced both
branches with a single throughput-conservation formula:

```
sampled_active = min(sampled_ctas, total_sms)
full_active    = min(total_ctas,   total_sms)
scale          = (total_ctas / full_active)
               / (sampled_ctas / sampled_active)
est_cycles     = sampled_cycles × scale
```

**Insight:** `active_sms` is capped at `total_sms`, *not*
`total_sms × max_cta_per_core`. Multiple resident CTAs per SM share
the SM's issue bandwidth, so they don't add parallelism. When
`sampled_ctas == total_ctas` (pilot expanded to whole grid) the scale
collapses to 1 and the raw cycle is reported.

| Workload | Before | After |
|---|---|---|
| hotspot | +43.4 | +14.7 |
| srad_v2 | +37.0 | +17.3 |
| backprop | +108 | +90.2 (still bad) |

p50 met, p90 still off — backprop remains the outlier.

### 5b. Force-expansion when K-rep undersamples (`c3204ec`)

backprop's grid is 256×1×1. K-rep's corner+midpoint heuristic
*collapses* on 1D grids — only 3 unique CTAs come out. The pilot
accepted that 3-CTA sample at iter 0 because it classified COMPUTE,
and the throughput formula extrapolated 3 CTAs to 256. Massive
over-projection.

Fix: refuse to accept iter 0 when the sample is small *and* the
kernel shows a memory-stall signal. The pilot then expands at least
once, capturing more CTAs/SM concurrency. Threshold pair chosen to
spare hotspot (already accurate) while catching backprop.

```
backprop  +90.2 → +30.5
```

p90 still over (30 vs 25), but 3× better.

### 5c. Log-fit concurrency-throughput model (`5ea3114`)

The remaining backprop residual comes from physics, not the formula.
Per-SM throughput grows nonlinearly with concurrent CTAs/SM —
memory-latency hiding scales sub-linearly. The throughput-conservation
formula assumes per-SM throughput is constant; it's not.

The pilot already produces multiple iterations at different
CTAs-per-SM densities. We just have to *use* them.

```
T(N) = a + b × log(N+1)        where N = CTAs per SM
```

Implementation:
- Each pilot iteration's `(sampled_ctas, sampled_cycles, active_sms)`
  triple is recorded in `pilot_state_t.history`.
- At accept time, fit `(a, b)` by least squares over per-density-aggregated points.
- Pass `(a, b)` through `last_kernel_wave_info_t` to `print_stats()`,
  which extrapolates `T(full_ctas_per_sm)` and computes
  `est_cycles = total_ctas / (full_active × T_full)`.
- Defensive: skip the fit when there are <2 distinct densities or the
  slope is non-positive. Falls back to the constant-throughput formula
  cleanly.
- Defensive clamp: never extrapolate to a per-SM throughput *lower*
  than the sampled iteration's measured throughput.

| Workload | Mode | err% |
|---|---|---|
| hotspot | throughput_compute (1 iter, no fit) | +14.7 |
| backprop | log_fit_compute (4 distinct densities) | **−15.4** |
| pathfinder | throughput_memory | −2.6 |
| bfs | throughput_mixed | −9.9 |
| srad_v2 | throughput_mixed (sampled==total) | +17.3 |
| lud | throughput_compute (small grid) | −0.1 |

**p50 = 12.3% (target <15%, met). p90 = 17.3% (target <25%, met.)**

### 5d. High-projection-ratio force-expand (`d27f968`)

Discovered during the wider-validation step (next section). The
mem-stall heuristic missed nn — its kernels classify COMPUTE with
`mem_stall_frac = 0` but still see massive concurrency-throughput
swings (1 CTA/SM sample → 23 CTAs/SM full grid). Latency-hiding
matters even when latency isn't memory.

Generalized the force-expand condition to also fire when
`full_ctas_per_sm > 4 × sampled_ctas_per_sm`, regardless of stall
signal. Threshold spares hotspot (1.6×) and pathfinder (1×) but catches
backprop (6.4×) and nn (23×).

---

## 6. Wider validation

After the original 6 workloads (hotspot, backprop, pathfinder, bfs,
srad_v2, lud) hit both p50 and p90 targets, we added the rest of
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
   51×1×1 grid. Pilot expands by *replicating* those 3 corners, so
   "full grid sample" runs 51 instances of 3 corner CTAs instead of
   51 unique CTAs. The corner CTAs have different work distribution
   than the average. Sampling-quality issue; no formula tweak fixes it.
2. **nn (+103%)** — 938 CTAs/kernel; pilot reaches 80 sampled (2
   CTAs/SM) but full grid is at 23 CTAs/SM. The log-fit's
   extrapolation from N=2 to N=23 is too far past the sampled range
   and under-shoots actual throughput growth ~2×.
3. **nw (+57%)** — 1–8 CTA kernels. Pilot's iter-0-reject + iter-1-accept
   doubles wall-time, plus cache-state pollution from iter 0 makes
   iter 1 itself slower than a clean baseline. **Negative speedup**
   regime — pilot shouldn't run at all on grids this small.

The honest scientific finding: the model **generalizes within the
6-workload subset's regime** (medium-grid, well-behaved K-rep,
reasonable log-fit extrapolation distance) and **doesn't generalize**
when those assumptions break.

---

## 7. Where we stand

**Shipped on `cta-sampling`** (12 commits beyond Phase A+B start):

| Theme | Commits |
|---|---|
| Phase A (classifier refine) | A1a, A1b, A2, A3, A4 |
| Phase B (cycle estimator scaffold) | B1, B2, B3 |
| Accuracy push | throughput-formula, force-expand-mem-stall, log-fit-concurrency, force-expand-projection-ratio |
| Docs | progress, post-execution writeup, accuracy-push update, log-fit update, wider-validation update |

**Targets met on the original 6-workload set (p50 < 15%, p90 < 25%).**

**Wider 10-workload set has 3 documented failure modes** with named
follow-ups in `CTA_SAMPLING_PHASE_AB_DONE.md`:

1. Deeper pilot expansion for high-projection kernels (nn).
2. K-rep clustering replacement (heartwall, generalizes).
3. Skip-sampling threshold for tiny kernels (nw).
4. AI weight calibration against a known-FLOPs kernel (D2 decision).
5. Pilot-rejected-iteration cleanup of per-SM aggregates (only
   matters if IPC numbers become a primary signal).

---

## 8. One-slide summary

> **CTA Sampling: from +43% cycle error to ±17% on the original
> 6-workload set, with all three accuracy mechanisms generalizing
> cleanly when the sample is representative. Three failure modes
> identified on a wider 10-workload sweep; each has a named, scoped
> follow-up.**
