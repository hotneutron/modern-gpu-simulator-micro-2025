# CTA Sampling — Handoff

**As of 2026-05-12.**
**Branch:** `cta-sampling` (33 commits ahead of `main`).
**Status:** Cycle-accuracy targets met on the wider 9-workload set
(p50 = 9.5%, p90 = 23.4%, max = 30.0%). Cumulative wall-time speedup
crossed unity on the rodinia2 toy traces (1.05× vs. 0.92× at the prior
checkpoint). C3a removed; the new default concurrency-throughput model
is `sat_exp`, a 2-parameter saturating exponential.

This file is the self-contained current-state pickup point. An agent
should be able to read just this file and act, without consulting
`history/`. History files are pointers to *how* we got here, not
load-bearing context for next steps.

---

## 1. What's currently in the code

The `cta-sampling` branch ships an adaptive sampling pipeline:

- **K-rep CTA selection** (corners + edge midpoints + interior),
  K ≈ 9 regardless of grid size, in `main.cc::compute_sampled_ctas()`.
- **3-way classifier** (compute / memory / mixed) driven by refined
  `kernel_ai = (FP + INT + W_dp·DP + W_tc·TC + W_sfu·SFU) /
  dram_bytes`, plus three memory-pressure signals
  (`achieved_bw_ratio`, `dram_queue_occupancy`, `mem_stall_frac`).
  Knobs: `-cta_sampling_ai_w_dp/tc/sfu` (2.0/8.0/4.0),
  `-cta_sampling_pressure_mstall` (0.4).
- **Adaptive pilot loop** that doubles `sim_ctas` per iteration until
  pressure signals stabilize, max doublings hit, or full grid reached.
  Force-expand fires on either
  `undersized && mem_stall_frac > 0.10` or
  `full_per_sm > 4 × sampled_per_sm` (back to the pre-C3a, plain
  `return false` behavior).
- **Whole-kernel cycle estimator** with four selectable concurrency-
  throughput models, plus a constant-throughput fallback when no fit
  is produced:
  - `sat_exp` (default) — `T(N) = T_max · (1 − exp(−k·N))`. Two-param
    fit via log-spaced grid search over `k`, analytical T_max per k.
    Built-in hardware-physical asymptote.
  - `logfit` (legacy) — `T(N) = a + b·log(N+1)`. Linear regression on
    `(log(N+1), T)` pairs.
  - `roofline_clamp` — log-fit clamped by `min(T_fit, T_roofline)`.
  - `roofline_exp` — `T(N) = T_roofline · (1 − exp(−k·N))`, one-param
    fit, asymptote tied to the kernel's measured AI.
- **Stat scaling**: `gpu_tot_sim_cycle` stays raw (sampled wave); new
  `gpu_tot_sim_cycle_estimated`, `gpu_tot_sim_cycle_estimation_mode`,
  `gpu_tot_ipc_estimated` carry the projection. The estimation_mode
  tag combines the model and the kernel class
  (e.g. `sat_exp_compute`, `roofline_exp_memory`,
  `throughput_compute` when no fit).
- **C1 — tiny-grid skip + full-grid auto-accept** (`4fe81c6`):
  pilot disabled when `total_ctas < total_sms`; non-COMPUTE classes
  short-circuit to accept when iter 0 already covers the full grid.
- **C2 — K-rep replication fix** (`31a4dd2`): when target ≥
  total_ctas, sample the full unique grid (no replication); when K <
  target < total_ctas, supplement K-rep with evenly-spaced fresh
  unique CTAs before falling back to duplication.
- **C3a is removed** (was `2af2c4c`, reverted as part of `96ffb68`).
  The new concurrency model has a built-in asymptote, so the per-
  kernel adaptive doublings cap that was compensating for the
  log-fit's unbounded extrapolation is no longer needed.

### Code locations to know

| Area | File |
|---|---|
| Sampling, pilot loop, classifier, CTA selection/replication, model dispatch at accept | `simulator-remodeled/gpu-simulator/main.cc` |
| Per-iter obs (cycles, ctas, active_sms, AI, compute_ops, peak_*), fit functions | `simulator-remodeled/gpu-simulator/main.cc` (`pilot_iter_obs_t`, `pilot_history_points`, `pilot_roofline_T`, `pilot_fit_log_throughput`, `pilot_fit_sat_exp`, `pilot_fit_roofline_exp`) |
| Cycle estimator switch (`logfit / sat_exp / roofline_clamp / roofline_exp`) | `.../gpgpu-sim/src/gpgpu-sim/gpu-sim.cc::gpu_print_stat` |
| Wave-info fit params struct | `.../gpgpu-sim/src/gpgpu-sim/gpu-sim.h::last_kernel_wave_info_t` |
| Concurrency-model knob | `.../trace-driven/trace_driven.{cc,h}` (`-cta_sampling_concurrency_model`, default `sat_exp`) |
| Refined `kernel_ai` + mem_stall_frac plumbing | `.../trace-driven/trace_driven.{cc,h}`, `.../remodeling/subcore.cc` |
| Per-class instruction counters | `.../shader.h`, `.../shader_core_wrapper.h`, `.../remodeling/sm.h` |
| Accuracy sweep harness (4 models × workloads) | `simulator-remodeled/util/cta_sampling/validate_models.py` |
| Speedup measurement harness (3 trials × workloads) | `simulator-remodeled/util/cta_sampling/measure_speedup.py` |
| Classifier validation harness | `simulator-remodeled/util/cta_sampling/validate.py` |

### Runtime knobs

| Knob | Default | Notes |
|---|---|---|
| `-cta_sampling_mode` | `0` | `1` enables sampling. |
| `-cta_sampling_pilot_max_doublings` | `0` (off) | `>0` enables the adaptive pilot loop. Use `2` for the validated configuration. |
| `-cta_sampling_concurrency_model` | `sat_exp` | One of `logfit / sat_exp / roofline_clamp / roofline_exp`. |

See the `build_environment.md` memory note for the conda env required
to build this branch.

---

## 2. Measured results (current state)

### Cycle accuracy — wider 9-workload set (rodinia2 / Turing)

Sweep harness: `util/cta_sampling/validate_models.py`. Pilot mode with
`-cta_sampling_pilot_max_doublings 2 -cta_sampling_concurrency_model
sat_exp` (the default).

```
hotspot     +14.7
backprop     -9.5
pathfinder   -0.3
bfs          +3.3
srad_v2     +18.1
lud           0.0
heartwall   -23.4
nn          +30.0
nw           +5.0

p50   9.5%    ✓ target <15%
p90  23.4%    ✓ target <25%
max  30.0%    (was 128.4% pure log-fit, 23.4% log-fit+C3a)
```

Full 4-model table is in `history/2026-05-12_log_fit_replacement.md`
§4 if a comparison is needed.

### Wall-time speedup — same 9-workload set, 3 trials each

Harness: `util/cta_sampling/measure_speedup.py`.
Raw CSV: `history/speedup_results_2026-05-12.csv`.

```
hotspot      1.53x   (9.72s  -> 6.33s)
backprop     0.87x
pathfinder   1.07x
bfs          1.19x
srad_v2      0.71x   ← regression
lud          1.00x
heartwall    0.53x   ← regression
nn           1.34x   ← biggest improvement (was 0.50x with C3a)
nw           1.00x

Cumulative   1.05x   (49.03s -> 46.73s)
```

Compared to the History 2 checkpoint (cumulative 0.92×), the new state
moves four workloads in the right direction and only `hotspot`
backward (by ~0.6×, within trial-to-trial variance for that workload).
The main driver of the cumulative move across unity is `nn`: 0.50× →
1.34× once C3a's extra pilot iterations go away.

### Why some workloads still regress (crossover model, unchanged)

`Pilot wins when N > sampled + a/c` where N = total CTAs, c = per-CTA
sim cost, a ≈ 4 s fixed pilot overhead. Crossovers:

| Per-CTA cost                | Crossover N |
|---|---|
| 1.3 ms/CTA (nn-like)        | ~3300 CTAs  |
| 33 ms/CTA (hotspot-like)    | ~210 CTAs   |
| 344 ms/CTA (lud-like)       | ~90 CTAs    |

Production-scale DL / HPC traces (10K+ CTAs/kernel) are projected to
land at 100×+ and have not been measured. The cumulative-1.05× toy-
trace number is a softer floor than the prior 0.92×, but doesn't change
the fundamental shape of the cost model — small grids still lose to
pilot overhead.

---

## 3. Open issues

### 3.1 Pilot wall-time budget + abort-to-baseline

Still the top blocker for default-on. Cumulative-1.05× softens the
"safe to default-on?" question on toy mixes, but a single low-CTA
kernel can still regress by 2×. Two options for the budget (carried
verbatim from History 2):

- **(a) CTA-count budget** (recommended first): track cumulative
  `sampled_ctas` across `pst.history`; abort when ≥ `pst.total_ctas`.
  No wall-clock infrastructure needed; conservative upper bound.
- **(b) Wall-time budget**: capture `chrono::steady_clock` per iter,
  abort when `pilot_elapsed > k · baseline_wall_estimate` for
  `k ∈ [1.0, 1.5]`.

Abort path: restore via `pilot_restore`, set `cta_sampling_mode = 0`
for this kernel only (use existing `update_sampling_on_trace_info`
with `target = total_ctas`), relaunch, finalize via normal accept
path. Emit `pilot_aborted_reason=budget_exceeded` in the
`CTA_PRESSURE_SIGNALS:` log line for the validation harness to count.
Expose `-cta_sampling_pilot_max_wall_ratio` (default 1.5; 0 disables
abort for production-scale users).

### 3.2 Speedup measurement on production-scale traces

NVBit-trace a transformer-layer-scale kernel (~100 K–1 M CTAs/kernel)
on real hardware and re-run the speedup comparison. Closes the "is
this useful?" question quantitatively. Toy-trace cumulative is now
1.05×; the production projection is unchanged at 100×+ but unverified.

### 3.3 Roofline-Exp resurrection via better AI source

`backprop` cycle error on `roofline_exp` is **−0.1%** (essentially
exact), but `nn` is **+675%** because the cold-sample AI for `nn`
yields a `T_roofline` below the measured `T_sample`. Two follow-ups
could unlock a model with a better physical asymptote than `sat_exp`:

- **D2 — AI weight calibration** against a known-FLOPs kernel
  (calibrates `W_dp / W_tc / W_sfu`).
- **Use the deepest accepted pilot iter's AI** (not just the deepest
  iter regardless of acceptance) — cache state is more representative
  at higher densities.

Speculative until measured. `sat_exp` is fine for now.

### 3.4 Lower-priority follow-ups (carried from history)

- **heartwall cache-state pollution** (−23.4%): would need
  hardware-state reset between rejected and accepted pilot iters.
- **K-rep clustering replacement**: full k-means over per-CTA
  features instead of C2's evenly-spaced supplement.
- **Pilot-rejected iter cleanup**: roll back per-SM aggregates.
  Affects IPC numerator only, not cycle estimate.
- **TMA modeling**: see `TMA_TRACING.md`.

---

## 4. Pointers

| File | Purpose |
|---|---|
| `history/2026-05-12_log_fit_replacement.md` | History 3 — this session's work: C3a revert, concurrency-model knob, 3 candidate models, comparison, sat_exp winner |
| `history/2026-04-30_foundation_and_accuracy_push.md` | History 1 — foundation, Phase A/B, accuracy push, wider-validation failure modes |
| `history/2026-05-07_C1_C2_C3a_and_speedup.md` | History 2 — C1, C2, C3a implementations + speedup measurement + crossover model |
| `history/speedup_results.csv` | History 2 speedup baseline |
| `history/speedup_results_2026-05-12.csv` | Current (post-C3a-revert, sat_exp) speedup data |
| `history/speedup_chart.png`, `history/speedup_projection.png` | History 2 chart artifacts (not regenerated yet for the new state) |
| `CLAUDE.md` | Project-level guide for the simulator codebase |
| `KNOWN_BUILD_FAILURES.md` | Unrelated — 3 benchmarks that fail to build with `sm_86` |


## Next Steps

Let's implement the pilot wall-time budget and abort-to-baseline path (issue 3.1) to make this safe to default-on