# CTA Sampling — Planned vs. Implemented

This document records the delta between what was planned across all three planning documents and what was actually implemented on the `cta-sampling` branch. It is a snapshot of the gap between intention and execution.

---

## 1. What Was Planned

### CTA_SAMPLING.md (Original Full Design)

**Two approaches:**
1. **Coordinate heuristic** — K-rep corners + midpoints + interior, K≈9, weight by position category
2. **Two-pass clustering** — Nsight Compute profiling → K-Means → selective NVBit re-trace of representatives

**Then revised** to add Approach 1 Revised: Roofline-guided adaptive occupancy filling with 3-way classifier and pilot loop.

### CTA_SAMPLING_REVISED_PLAN.md

- 3-way classifier (`compute` / `memory` / `mixed`) using `ridge_ratio` and pressure signals
- Pilot adaptation loop: start at class-dependent target, double until stable
- Stratified shuffle CTA replication
- Stat aggregation rules (scale additive counts, recompute ratios)
- Inter-CTA contention guardrails

### CTA_SAMPLING_SINGLE_PASS_PLAN.md

- Single-pass only — no pilot loop
- Conservative one-shot SM sizing: `compute→35%, mixed→55%, memory→75%`
- Goal: avoid multi-pass complexity

### CTA_SAMPLING_NEXT_STEPS.md (Phase A + B)

**Phase A — refine classifier inputs:**
- Per-class instruction counters (FP, INT, DP, TC, SFU, loads, stores) in `pressure_signals_t`
- Refined `kernel_ai = (FP + INT + W_dp·DP + W_tc·TC + W_sfu·SFU) / dram_bytes` with tunable weights
- Memory-stall fraction from `stall_l1c_cycles / issue_eval_cycles`
- 3-input classifier: BW ratio OR queue occupancy OR mem_stall → memory

**Phase B — cycle estimation:**
- Wave-info struct plumbing from main.cc to print_stats()
- `per_cta` mode for compute: `cyc_per_cta × ceil(total_ctas/total_sms)`
- `steady_state` mode for memory: `sampled_cycles × N_waves_full / N_waves_sample`
- New stats: `gpu_tot_sim_cycle_estimated`, `gpu_tot_sim_cycle_estimation_mode`, `gpu_tot_ipc_estimated`

---

## 2. What Was Implemented

### 2.1 CTA Selection

| Planned | Implemented |
|---|---|
| Coordinate heuristic (corners + edges + interior) | ✅ Exactly as planned — `compute_sampled_ctas()` in `main.cc` |

### 2.2 CTA Replication

| Planned | Implemented |
|---|---|
| Stratified shuffle (Fisher-Yates) to avoid periodic L2/DRAM artifacts | ✅ `expand_sampled_ctas()` in `main.cc` |

### 2.3 Pressure Signal Extraction

| Planned | Implemented |
|---|---|
| `gpu_tot_sim_cycle`, `gpu_tot_sim_insn`, `dram_bytes`, `l2_misses`, `achieved_bw_ratio`, `dram_queue_occupancy` | ✅ `compute_kernel_pressure_signals()` in `gpu-sim.cc` |

### 2.4 Per-Class Instruction Counters

| Planned | Implemented |
|---|---|
| FP, INT, DP, TC, SFU, loads, stores counters in `pressure_signals_t` | ✅ All implemented. Weights: `W_dp=2.0, W_tc=8.0, W_sfu=4.0` as planned (not calibrated) |

### 2.5 Memory-Stall Fraction

| Planned | Implemented |
|---|---|
| `mem_stall_frac = stall_l1c_cycles / issue_eval_cycles` from `subcore.cc` counter | ✅ Implemented. Re-enabled previously-commented-out increment in `subcore.cc` |

### 2.6 3-Way Classifier

| Planned | Implemented |
|---|---|
| `ridge_ratio >= T_high` → compute; `<= T_low` OR high pressure → memory; else mixed | ✅ Exactly as planned in `classify_kernel()` |
| Thresholds: `T_low=0.9, T_high=1.3` | ✅ As planned |
| Memory pressure: `achieved_bw_ratio >= 0.6` OR `dram_queue_occupancy >= threshold` OR `mem_stall_frac >= threshold` | ✅ All three as planned |
| New knob `-cta_sampling_pressure_mstall` | ✅ Implemented |

### 2.7 Initial Target Selection

| Planned | Implemented |
|---|---|
| `compute → max(k_reps, total_sms/2)` | ✅ `compute_initial_sim_ctas()` |
| `memory → max(k_reps, N_sat_est)` | ✅ |
| `mixed → max(k_reps, ceil(1.5 × N_sat_est))` | ✅ |

### 2.8 N_sat_est

| Planned | Implemented |
|---|---|
| `N_sat_est = ceil(peak_dram_bw / per_sm_bw_est)` | ✅ `compute_n_sat_est()` in `main.cc` |

### 2.9 Pilot Loop Stop Conditions

| Planned | Implemented |
|---|---|
| `sim_ctas == total_ctas` → stop | ✅ |
| `iter > max_doublings` → stop | ✅ |
| `achieved_bw_ratio >= 0.8` (primary stop target) | ✅ |
| BW delta < 5% AND IPC delta < 5% (stability check) | ✅ |

### 2.10 Force-Expansion Triggers (iter-0 rejection)

| Planned | Implemented |
|---|---|
| Not planned — added post-initial-implementation | ✅ `undersized && mem_stall_frac > 0.10` |
| Not planned — added after wider validation | ✅ `full_ctas_per_sm > 4 × sampled_ctas_per_sm` |

### 2.11 Cycle Estimation — Formula

| Planned | Implemented |
|---|---|
| `per_cta` mode: `cyc_per_cta × ceil(total_ctas/total_sms)` for compute | ✅ Replaced by throughput-conservation formula |
| `steady_state` mode: `sampled_cycles × N_waves_full / N_waves_sample` for memory | ✅ Replaced by throughput-conservation formula |
| Throughput-conservation formula (not in original plan, added in accuracy push) | ✅ `scale = (total_ctas / full_active) / (sampled_ctas / sampled_active)` |
| Log-fit concurrency model (not in original plan) | ✅ `T(N) = a + b × log(N+1)` from pilot iteration history |

### 2.12 Wave-Info Plumbing

| Planned | Implemented |
|---|---|
| `last_kernel_wave_info_t` struct from main.cc to print_stats() | ✅ |
| `gpu_tot_sim_cycle_estimated` | ✅ |
| `gpu_tot_sim_cycle_estimation_mode` | ✅ (`throughput_compute`, `throughput_memory`, `throughput_mixed`, `log_fit_compute`, `log_fit_memory`, `log_fit_mixed`) |
| `gpu_tot_ipc_estimated` | ✅ |

### 2.13 Validation Harness

| Planned | Implemented |
|---|---|
| `validate.py` with 4 modes: baseline, K-rep, expanded40, pilot | ✅ |
| Cycle error % vs baseline | ✅ |
| 6 workloads: hotspot, backprop, pathfinder, bfs, srad_v2, lud | ✅ |
| Extended to 10 workloads (heartwall, nn, nw, streamcluster) | ✅ Beyond plan |

---

## 3. What Was NOT Implemented

### 3.1 Two-Pass Clustering (Approach 2)

**Planned in:** CTA_SAMPLING.md

**Status:** Not implemented. Explicitly deferred as a future redesign. Requires:
- Nsight Compute integration for per-CTA hardware counter profiling
- K-Means clustering pipeline
- Selective NVBit re-trace of representative CTAs

### 3.2 Single-Pass Variant

**Planned in:** CTA_SAMPLING_SINGLE_PASS_PLAN.md

**Status:** Not implemented. The conservative 35%/55%/75% SM tiering was never coded. The adaptive pilot loop was chosen instead.

### 3.3 Contention Guardrails (full set)

**Planned in:** CTA_SAMPLING_REVISED_PLAN.md

Partial implementation — some guardrails are implicitly handled by the pilot loop's stop conditions, but the full explicit checklist was not implemented as a pre-acceptance gate:
- L2 slice/set contention drift ✅ (implicit in pilot)
- DRAM partition camping indicators ❌ (not explicit)
- MSHR/miss-queue pressure saturation ❌ (not explicit)
- NoC/crossbar pressure proxies ❌ (not explicit)
- Atomic hotspot amplification ❌ (not explicit)
- Warp scheduler pressure (`eligible warps per cycle`) ❌ (not explicit)

### 3.4 AI Weight Calibration (D2)

**Planned in:** CTA_SAMPLING_NEXT_STEPS.md

**Status:** Not implemented. Weights `W_dp=2.0, W_tc=8.0, W_sfu=4.0` remain as first-order guesses. Calibration against a known-FLOPs kernel (e.g., tiled GEMM) is documented as a follow-up.

### 3.5 Multi-Wave Cycle Model (Phase B, literal version)

**Planned in:** CTA_SAMPLING_NEXT_STEPS.md

**Status:** Partially addressed — the `per_cta` and `steady_state` formulas were replaced by the throughput-conservation formula + log-fit. The original wave-count model was not implemented as specified. The goal (whole-kernel cycle estimate) was achieved by different means.

### 3.6 TMA Modeling

**Not planned** (TMA discovered post-implementation during trace analysis).

**Status:** Not modeled. FlashAttention-3 uses `UTMALDG.4D` / `UTMASTG.*` TMA instructions which bypass standard memory pipelines. Memory pressure signals do not capture TMA transaction behavior. See `.plan/TMA_TRACING.md`.

---

## 4. Major Deviations from Plan

### 4.1 Cycle Estimator Was Completely Replaced

The original Phase B plan specified `per_cta` and `steady_state` formulas. In practice, the `ceil`-based per-CTA formula produced +43% error on hotspot because it over-counted partial last waves. The entire cycle estimator was replaced with a unified throughput-conservation formula that was not in any plan doc.

### 4.2 Log-Fit Model Added Post-Hoc

The `T(N) = a + b × log(N+1)` log-fit concurrency model was added after the throughput-conservation formula still left backprop at +30.5% error. This was an unplanned addition driven by empirical results on the pilot iteration history.

### 4.3 Two Force-Expansion Heuristics Added Post-Hoc

Not in any plan document:
1. `undersized && mem_stall_frac > 0.10` — discovered during initial accuracy testing
2. `full_ctas_per_sm > 4 × sampled_ctas_per_sm` — discovered during wider 10-workload validation

### 4.4 Wider Validation Extended Beyond 6 Kernels

The original plan specified 5 kernels. Validation was extended to 10 workloads (adding heartwall, nn, nw, streamcluster), which revealed the three distinct failure modes documented in the follow-ups.

---

## 5. Validation Results

### Original 6-Workload Sweep (Targets Met)

| Workload | cycle_err% | estimation_mode |
|---|---|---|
| hotspot | +14.7 | throughput_compute |
| backprop | -15.4 | log_fit_compute |
| pathfinder | -2.6 | throughput_memory |
| bfs | -9.9 | throughput_mixed |
| srad_v2 | +17.3 | throughput_mixed |
| lud | -0.1 | throughput_compute |

**p50 = 12.3% (< 15%). p90 = 17.3% (< 25%).**

### Wider 10-Workload Sweep (3 Failures)

| Workload | cycle_err% | Failure Mode |
|---|---|---|
| hotspot | +14.7 | — |
| backprop | -15.4 | — |
| pathfinder | -2.6 | — |
| bfs | -9.9 | — |
| srad_v2 | +17.3 | — |
| lud | -0.1 | — |
| heartwall | -28.7 | K-rep collapse on 1D grid |
| nn | +103.5 | Log-fit under-extrapolation |
| nw | +56.8 | Pilot overhead on tiny kernels |
| streamcluster | -1.5 | — |

**p50 = 15.0%. p90 = 56.8%.**

---

## 6. Follow-Up Items

Ranked by priority (from CTA_SAMPLING_PHASE_AB_DONE.md):

1. **Deeper pilot expansion for high-projection kernels** — nn-style failure: log-fit extrapolates from 2 CTAs/SM to 23 CTAs/SM and underestimates throughput growth. Fix: adaptively increase `pilot_max_doublings` when force-expand fires.
2. **K-rep clustering replacement** — heartwall-style failure: corner+midpoint heuristic collapses on 1D grids, samples unrepresentative CTAs. Fix: replace with per-CTA feature clustering.
3. **Skip-sampling threshold for tiny kernels** — nw-style failure: pilot overhead exceeds speedup on grids < 2× total_sms. Fix: short-circuit to baseline for tiny grids.
4. **AI weight calibration** — fit `W_dp`, `W_tc`, `W_sfu` against a known-FLOPs kernel.
5. **Pilot rejected iteration cleanup** — `pilot_restore` doesn't roll back per-SM aggregates.
6. **TMA modeling** — model TMA transaction behavior separately from standard memory (see TMA_TRACING.md).
