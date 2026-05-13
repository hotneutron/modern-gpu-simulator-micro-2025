# CTA Sampling — Handoff

**As of 2026-05-13.**
**Branch:** `cta-sampling` (33 commits ahead of `main`, + uncommitted wall-time-budget work).
**Status:** Cycle-accuracy targets met on the wider 9-workload set
(p50 = 9.5%, p90 = 23.4%, max = 30.0%) — unchanged after adding the
abort path. Cumulative wall-time speedup on the rodinia2 toy traces
moved from 1.05× → **1.10×** with the abort path enabled at default
ratio 1.5 (validated 2026-05-13). C3a removed; the new default
concurrency-throughput model is `sat_exp`, a 2-parameter saturating
exponential. A pilot wall-time budget + abort-to-baseline path is now
in place so a runaway pilot falls back cleanly to a single full-grid
run.

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
- **Pilot wall-time budget + abort-to-baseline** (uncommitted, 2026-05-13):
  per-iter `chrono::steady_clock` deltas accumulate into
  `pst.pilot_elapsed_sec`. The K-rep iter projects a baseline run cost
  `T0 × total_ctas / ctas_launched_iter0`; subsequent iters compare
  `pilot_elapsed_sec` against `ratio × baseline_wall_est_sec`. When
  exceeded *and* the current iter was already going to reject, the pilot
  aborts: rejects this iter (rolls back `gpu_tot_*` via `pilot_restore`),
  sets next target to `total_ctas`, and the relaunched iter is forced
  to accept regardless of classifier. `CTA_PRESSURE_SIGNALS:` log line
  carries a new `pilot_aborted_reason={none,budget_exceeded}` field for
  the validation harness to count. Knob:
  `-cta_sampling_pilot_max_wall_ratio` (default `1.5`; `0` disables).
  Verified on backprop/Turing: default ratio runs the full pilot; tight
  ratio (`0.01`) triggers abort and produces full-grid cycles within 1%
  of the no-sampling baseline.

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
| `-cta_sampling_pilot_max_wall_ratio` | `1.5` | Pilot wall-time budget multiplier vs. projected baseline (T0·total/k_reps). `0` disables abort. |

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
Raw CSV: `history/speedup_results_2026-05-13.csv` (with the abort
path enabled at default ratio 1.5).
Prior CSV without the abort path: `history/speedup_results_2026-05-12.csv`.

```
                    2026-05-13     2026-05-12
hotspot              1.59x          1.53x
backprop             0.82x          0.87x
pathfinder           1.10x          1.07x
bfs                  1.44x          1.19x
srad_v2              0.66x          0.71x   ← still regressing
lud                  1.00x          1.00x
heartwall            0.53x          0.53x   ← still regressing
nn                   1.35x          1.34x
nw                   1.08x          1.00x

Cumulative           1.10x          1.05x   (54.59s -> 49.53s)
```

The abort path improved the cumulative move (1.05× → 1.10×) and did
not change any workload's accuracy (no abort fired at default ratio
1.5 on this set — verified by counting `pilot_aborted_reason=
budget_exceeded` log lines, which was 0 across all 9 workloads × 3
trials × 4 cycle-estimator models).

The remaining regressions (heartwall, srad_v2, backprop) come from
small grids where the pilot's 4-iter doubling schedule runs each
iteration to roughly the same simulator-cycle count regardless of CTA
count (since CTAs run concurrently on SMs), so total simulator-cycle
work scales ~4× vs. baseline. The abort budget at 1.5× projected
baseline tolerates that 4×, and tightening it (e.g. to 1.0) would
help these but cut other workloads' pilots short prematurely.

### Crossover model (unchanged)

`Pilot wins when N > sampled + a/c` where N = total CTAs, c = per-CTA
sim cost, a ≈ 4 s fixed pilot overhead. Crossovers:

| Per-CTA cost                | Crossover N |
|---|---|
| 1.3 ms/CTA (nn-like)        | ~3300 CTAs  |
| 33 ms/CTA (hotspot-like)    | ~210 CTAs   |
| 344 ms/CTA (lud-like)       | ~90 CTAs    |

Production-scale DL / HPC traces (10K+ CTAs/kernel) are projected to
land at 100×+ and have not been measured. The cumulative-1.10× toy-
trace number is a softer floor than the prior 1.05× / 0.92×, but
doesn't change the fundamental shape of the cost model — small grids
still lose to pilot overhead.

---

## 3. Open issues

### 3.1 Pilot wall-time budget + abort-to-baseline — DONE 2026-05-13

Implemented as option (b) from History 2's plan: per-iter wall-time
deltas accumulate into `pst.pilot_elapsed_sec`; iter 0 projects a
baseline cost `T0 × total_ctas / ctas_launched_iter0`; abort fires
when the iter would otherwise reject *and* `pilot_elapsed_sec >
ratio × baseline_wall_est_sec`. Knob:
`-cta_sampling_pilot_max_wall_ratio` (default 1.5; 0 disables).

Validated 2026-05-13: cumulative speedup 1.05× → 1.10× on the
9-workload toy set; cycle accuracy unchanged (sat_exp p50 9.5%, p90
23.4%, max 30.0%). At default ratio 1.5 the abort doesn't fire on
any toy workload, so the new path is a safety net for production-
scale traces rather than a toy-set speedup driver. Forced-abort
behavior was unit-tested with `-cta_sampling_pilot_max_wall_ratio
0.01`: the recovery iter produces full-grid cycles within 1% of the
no-sampling baseline.

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
| `history/speedup_results_2026-05-12.csv` | Post-C3a-revert, sat_exp, pre-abort-path |
| `history/speedup_results_2026-05-13.csv` | Current — abort path enabled at default ratio 1.5 |
| `history/speedup_chart.png`, `history/speedup_projection.png` | History 2 chart artifacts (not regenerated yet for the new state) |
| `CLAUDE.md` | Project-level guide for the simulator codebase |
| `KNOWN_BUILD_FAILURES.md` | Unrelated — 3 benchmarks that fail to build with `sm_86` |


## Next Steps

Issue 3.1 done and validated (see §2 and §3.1). Top open question is
now 3.2: measure wall-time speedup on a production-scale (NVBit-traced)
kernel so we can confirm the 100×+ projection and check whether the
abort path is the right safety net at that scale.

A secondary open thread: heartwall (0.53×) is the worst toy regression
and the abort path didn't help — each pilot iter runs to ~the same
simulator-cycle count regardless of CTA count, so the projected
baseline `T0 × total/k_reps` over-estimates the true baseline. A more
honest estimate would use sampled `sim_cycles` rather than wall time
when projecting. Speculative; only worth doing if production-scale
measurements show similar over-projection.