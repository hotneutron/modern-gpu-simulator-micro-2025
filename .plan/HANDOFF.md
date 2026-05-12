# CTA Sampling — Handoff

**As of 2026-05-12.**
**Branch:** `cta-sampling` (32 commits ahead of `main`).
**Status:** Cycle-accuracy targets met on the wider 9-workload set
(p50 = 10.1%, p90 = 23.4%). Wall-time speedup workload-dependent
(0.92× cumulative on rodinia2 toy traces; projected 100×+ on
production-scale).

This file is the self-contained current-state pickup point. An agent
should be able to read just this file and act, without consulting
`history/`. History files are pointers to *how* we got here, not
load-bearing context for next steps.

---

## 1. What's currently in the code

The `cta-sampling` branch ships an adaptive sampling pipeline:

- **K-rep CTA selection** (corners + edge midpoints + interior),
  K≈9 regardless of grid size, in `main.cc::compute_sampled_ctas()`.
- **3-way classifier** (compute / memory / mixed) driven by refined
  `kernel_ai = (FP + INT + W_dp·DP + W_tc·TC + W_sfu·SFU) /
  dram_bytes`, plus three memory-pressure signals
  (`achieved_bw_ratio`, `dram_queue_occupancy`, `mem_stall_frac`).
  Knobs: `-cta_sampling_ai_w_dp/tc/sfu` (2.0/8.0/4.0),
  `-cta_sampling_pressure_mstall` (0.4).
- **Adaptive pilot loop** that doubles `sim_ctas` per iteration until
  pressure signals stabilize, max doublings hit, or full grid
  reached. Force-expand fires on either
  `undersized && mem_stall_frac > 0.10` or
  `full_per_sm > 4 × sampled_per_sm`.
- **Whole-kernel cycle estimator** with two modes:
  - **Throughput-conservation formula** as default:
    `est_cycles = sampled_cycles × (total_ctas/full_active) /
    (sampled_ctas/sampled_active)`.
  - **Log-fit concurrency model** `T(N) = a + b·log(N+1)` (N =
    CTAs/SM) when the pilot history has ≥ 2 distinct densities and a
    positive slope; extrapolates to full-grid CTAs/SM density.
- **Stat scaling**: `gpu_tot_sim_cycle` stays raw (sampled wave); new
  `gpu_tot_sim_cycle_estimated`, `gpu_tot_sim_cycle_estimation_mode`,
  `gpu_tot_ipc_estimated` carry the projection.
- **C1 — tiny-grid skip + full-grid auto-accept** (`4fe81c6`):
  pilot disabled when `total_ctas < total_sms`; non-COMPUTE classes
  short-circuit to accept when iter 0 already covers the full grid.
- **C2 — K-rep replication fix** (`31a4dd2`): when target ≥
  total_ctas, sample the full unique grid (no replication); when K <
  target < total_ctas, supplement K-rep with evenly-spaced fresh
  unique CTAs before falling back to duplication.
- **C3a — adaptive `pilot_max_doublings`** (`471ccd5`): when
  force-expand fires for a high projection ratio, bump the per-state
  doublings cap to `max(global, ceil(log2(projection_ratio)))`,
  capped at `PILOT_MAX_DOUBLINGS_CEILING = 5`. Kernels that don't
  trigger force-expand are unaffected.

### Code locations to know

| Area | File |
|---|---|
| Sampling, pilot loop, classifier, CTA selection/replication | `simulator-remodeled/gpu-simulator/main.cc` |
| Cycle estimator (`throughput_*`, `log_fit_*` modes) + new stats | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc`, `.../gpu-sim.h` |
| Refined `kernel_ai` + mem_stall_frac plumbing | `.../trace-driven/trace_driven.{cc,h}`, `.../remodeling/subcore.cc` |
| Per-class instruction counters | `.../shader.h`, `.../shader_core_wrapper.h`, `.../remodeling/sm.h` |
| Validation harness | `simulator-remodeled/util/cta_sampling/validate.py` |

### Runtime knob

`-cta_sampling_mode 1` enables sampling; 0 disables. See the
`build_environment.md` memory note for the conda env required to
build this branch.

---

## 2. Measured results (current state)

### Cycle accuracy — wider 9-workload set (rodinia2 / Turing)

```
hotspot     +14.7
backprop    -13.9
pathfinder   -0.3
bfs          +3.3
srad_v2     +18.1
lud           0.0
heartwall   -23.4
nn          +10.1
nw           +5.0

p50  10.1%    ✓ target <15%
p90  23.4%    ✓ target <25%
```

### Wall-time speedup — same 9-workload set, 3 trials each

```
hotspot      2.12x   (10.58s -> 4.98s)    ← biggest gain
pathfinder   1.05x
bfs          1.04x
lud          1.01x
nw           0.96x
srad_v2      0.77x   ← regression
backprop     0.70x   ← regression
nn           0.50x   ← regression
heartwall    0.46x   ← regression

Cumulative   0.92x   (45.96s -> 49.88s)
```

Charts and raw data: `history/speedup_chart.png`,
`history/speedup_projection.png`, `history/speedup_results.csv`.

### Why some workloads regress (crossover model)

`Pilot wins when N > sampled + a/c` where N = total CTAs, c = per-CTA
sim cost, a ≈ 4.3 s fixed pilot overhead. Crossovers:

| Per-CTA cost                | Crossover N |
|---|---|
| 1.3 ms/CTA (nn-like)        | ~3300 CTAs  |
| 33 ms/CTA (hotspot-like)    | ~210 CTAs   |
| 344 ms/CTA (lud-like)       | ~90 CTAs    |

Every measured workload matches: hotspot 320 > 210 → 2.12× ✓; nn
938 ≪ 3300 → 0.50× ✗; lud 24 ≈ 90 → 1.01× tied ✓. Production-scale
DL / HPC traces (10K+ CTAs) are projected to land 100×+ but have
not been measured.

---

## 3. Top open issues

### 3.1 Log-fit replacement (active task — see §4)

The C3a fix mitigates nn by bringing the pilot's deepest sampled
density close to the full-grid density, so the log-fit barely
extrapolates. It does **not** address the underlying critique:
`T(N) = a + b·log(N+1)` has no physical basis, no hardware-limit
asymptote, and no microarchitectural grounding. The full debate of
candidate replacements (Saturating Exponential, Roofline-Bounded,
Roofline-Tied Exponential hybrid) is in
**[`CTA_SAMPLING_Debate_Log_Fit.md`](CTA_SAMPLING_Debate_Log_Fit.md)**.

When the log-fit replacement lands, **C3a should be reverted** — it
exists to compensate for a weakness of the log-fit, and once the
form has a proper asymptote that compensation costs accuracy and
wall-time for no benefit.

### 3.2 Pilot wall-time budget + abort-to-baseline (independent follow-up)

Without an upper bound on pilot wall time, the optimization is unsafe
to default-on for workload mixes that include any kernel below the
crossover threshold — the user could be served better by simply not
enabling sampling. Two options for the budget:

- **(a) CTA-count budget** (recommended first): track cumulative
  `sampled_ctas` across `pst.history`; abort when ≥ `pst.total_ctas`.
  No wall-clock infrastructure needed; conservative upper bound.
- **(b) Wall-time budget**: capture `chrono::steady_clock` per iter,
  abort when `pilot_elapsed > k × baseline_wall_estimate` for
  `k ∈ [1.0, 1.5]`.

Abort path: restore via `pilot_restore`, set `cta_sampling_mode = 0`
for this kernel only (use existing `update_sampling_on_trace_info`
with `target = total_ctas`), relaunch, finalize via normal accept
path. Emit `pilot_aborted_reason=budget_exceeded` in the
`CTA_PRESSURE_SIGNALS:` log line for the validation harness to count.
Expose `-cta_sampling_pilot_max_wall_ratio` (default 1.5; 0 disables
abort for production-scale users).

### 3.3 Speedup measurement on production-scale traces

NVBit-trace a transformer-layer-scale kernel (~100 K–1 M CTAs/kernel)
on real hardware and re-run the speedup comparison. Closes the "is
this useful?" question quantitatively.

### 3.4 Lower-priority follow-ups (carried from history)

- **heartwall cache-state pollution** (−23.4%): would need
  hardware-state reset between rejected and accepted pilot iters.
- **AI weight calibration** (D2): fit `W_dp/W_tc/W_sfu` against a
  known-FLOPs kernel.
- **K-rep clustering replacement**: full k-means over per-CTA
  features instead of C2's evenly-spaced supplement.
- **Pilot-rejected iter cleanup**: roll back per-SM aggregates.
  Affects IPC numerator only, not cycle estimate.
- **TMA modeling**: see `TMA_TRACING.md`.

---

## 4. Next steps

Let's start by reverting C3a, then handling 3.1 (log-fit replacement). The document describes 3 different candidates. Let's try each of them and report the accuracy results of each. We can take a decision based on those results.


---

## 5. Pointers

| File | Purpose |
|---|---|
| `CTA_SAMPLING_Debate_Log_Fit.md` | Active design debate — log-fit replacement candidates |
| `history/2026-04-30_foundation_and_accuracy_push.md` | History 1 — foundation, Phase A/B, accuracy push, identification of wider-validation failure modes |
| `history/2026-05-07_C1_C2_C3a_and_speedup.md` | History 2 — C1, C2, C3a implementations + speedup measurement + crossover model |
| `history/speedup_chart.png`, `history/speedup_projection.png`, `history/speedup_results.csv` | Wall-time measurement artifacts (per-workload, mean of 3 trials, 9 workloads) |
| `CLAUDE.md` | Project-level guide for the simulator codebase |
| `KNOWN_BUILD_FAILURES.md` | Unrelated — 3 benchmarks that fail to build with `sm_86` |
