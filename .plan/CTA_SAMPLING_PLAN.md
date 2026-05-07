# CTA Sampling — Coherent Planning Document

This document is a merged reconstruction of all planning docs that were written during the design of the CTA sampling feature. It captures the full design space explored, the rationale for decisions made, and the three planning directions that were evaluated before the implementation was committed.

---

## 1. Problem Statement

GCL-Sampler clusters kernel *invocations* across a multi-kernel workload. When the workload is a **single kernel**, there is nothing to cluster at the kernel level. Yet individual kernels can be extremely expensive:

- **Trace generation** (NVBit): instrumenting every CTA × warp × instruction can take hours for large kernels
- **Simulation** (Accel-Sim): simulation time scales roughly linearly with CTA count

**Goal:** sample a representative subset of CTAs, simulate just those, and project whole-kernel stats from the sampled wave — within ±15% / ±25% (p50 / p90) cycle accuracy against a full-grid baseline.

---

## 2. The Two Fundamental Questions

Every sampling system must answer:

1. **Which CTAs to sample?** A bad sample biases everything downstream.
2. **How to project to the full kernel?** Sampled cycles aren't full cycles; the formula matters.

---

## 3. Original Design: Two Approaches

### Approach 1: Coordinate-Based Heuristic (Fast Baseline)

For **regular compute kernels** (GEMM, convolution, stencil), CTAs doing identical work are identifiable by grid position. The dominant CTA heterogeneity is **boundary effects** — edge/corner CTAs handle partial tiles.

**Sampling rule:**
- Select corners, edge midpoints (X/Y/Z faces), and one interior representative → K≈9 regardless of grid size
- Weight results by how many CTAs fall into each position category

**When it works:** GEMM, convolution, pooling, elementwise kernels, stencil codes.
**When it fails:** Sparse kernels, graph algorithms, reductions with irregular access — CTA behavior is determined by data, not grid position.

### Approach 2: Two-Pass Clustering (Full Redesign)

1. **Pass 1:** Collect per-CTA feature vectors via Nsight Compute hardware counters (L1 hit rate, L2 hit rate, active warps, memory transactions, branch divergence, etc.) — 10–50× faster than full SASS trace
2. **Cluster** CTAs by feature similarity (K-Means)
3. **Pass 2:** NVBit traces only the K representative CTAs
4. **Simulate** representatives, weight by cluster size

**Status:** Deferred. Requires significant new infrastructure (NVBit counter pass + clustering pipeline). The coordinate heuristic was pursued first as the tractable path.

---

## 4. The Occupancy Collapse Problem

Naive K-rep simulation on K≈9 CTAs leaves most SMs idle, causing two classes of error:

1. **Occupancy collapse:** per-SM warp count drops, latency hiding degrades, IPC underestimated even for compute-bound kernels
2. **Inter-CTA contention absence:** L2 set conflicts, DRAM bank camping, MSHR pressure, NoC contention, atomic hotspot amplification — all absent or underrepresented

---

## 5. Three Planning Directions

Three planning documents were written to address these problems:

### Direction A — Revised Plan (Chosen): Roofline-Guided Adaptive Pilot Loop

**CTA_SAMPLING_REVISED_PLAN.md** + **CTA_SAMPLING_NEXT_STEPS.md** (Phase A + B)

Use a 3-way classifier (`compute` / `memory` / `mixed`) based on Roofline-derived Arithmetic Intensity plus memory pressure signals. Then use an **adaptive pilot loop** that starts at a conservative CTA count and doubles until pressure metrics stabilize or DRAM is saturated.

Key design decisions:
- `kernel_ai = (FP + INT + W_dp·DP + W_tc·TC + W_sfu·SFU) / dram_bytes`
- `ridge_ratio = kernel_ai / ridge_point`
- Memory pressure: `achieved_bw_ratio`, `dram_queue_occupancy`, `mem_stall_frac`
- Initial targets: `compute → sms/2`, `memory → N_sat_est`, `mixed → 1.5× N_sat_est`
- Pilot stops when: BW ratio ≥ 0.8, OR deltas < tolerance, OR max doublings hit, OR sim_ctas == total_ctas
- CTA replication: stratified shuffle to avoid L2/DRAM striping periodicity

**Phase A** (classifier refinement):
- Per-class instruction counters (FP, INT, DP, TC, SFU, loads, stores) in `pressure_signals_t`
- Refined `kernel_ai` with knob-tunable weights `W_dp=2.0, W_tc=8.0, W_sfu=4.0`
- Memory-stall fraction from `stall_l1c_cycles / issue_eval_cycles`
- 3-input classifier: any of (BW ratio ≥ 0.6, queue occupancy high, mem_stall_frac ≥ threshold) → memory

**Phase B** (cycle estimation):
- Wave-info plumbing: `last_kernel_wave_info_t` struct carried from `main.cc` to `print_stats()`
- Two estimation modes: `per_cta` (compute) and `steady_state` (memory)
- New stats: `gpu_tot_sim_cycle_estimated`, `gpu_tot_sim_cycle_estimation_mode`, `gpu_tot_ipc_estimated`
- Backward-compatible: raw `gpu_tot_sim_cycle` unchanged

### Direction B — Single-Pass Variant: Conservative One-Shot SM Sizing

**CTA_SAMPLING_SINGLE_PASS_PLAN.md**

Skip the pilot loop entirely. Use a single one-shot mapping from kernel class to fixed SM fraction:
- `compute → 35% of SMs`
- `mixed → 55% of SMs`
- `memory → 75% of SMs`

**Rationale:** Avoid multi-pass complexity. Conservative SM sizing addresses the occupancy collapse problem without adaptive loops.

**Status:** Not implemented. Set aside in favor of the adaptive pilot loop which was judged more accurate for memory-bound kernels.

### Direction C — Two-Pass Clustering (Deferred)

From the original **CTA_SAMPLING.md** (Approach 2).

**Status:** Not implemented. Explicitly deferred as a larger future redesign to replace the coordinate heuristic.

---

## 6. Core Design Decisions

### 6.1 CTA Selection

**Implemented:** Coordinate-based K-rep heuristic — corners + edge midpoints + interior. K≈9. Implemented in `main.cc::compute_sampled_ctas()`.

**Deferred:** Clustering-based selection. Requires two-pass infrastructure (Nsight Compute profiling + K-Means) not yet built.

### 6.2 CTA Replication

**Implemented:** Stratified shuffle replication via `expand_sampled_ctas()` in `main.cc`. Uses Fisher-Yates shuffle to break L2/DRAM striping periodicity when replicating reps to fill target SM count.

**Planned:** Same — no change from revised plan.

### 6.3 Kernel Classification

**Implemented:** 3-way classifier (`compute` / `memory` / `mixed`) driven by:
- `kernel_ai = (FP + INT + W_dp·DP + W_tc·TC + W_sfu·SFU) / dram_bytes`
- `ridge_ratio = kernel_ai / ridge_point`
- Memory pressure via `achieved_bw_ratio`, `dram_queue_occupancy`, `mem_stall_frac`
- Knobs: `T_low=0.9`, `T_high=1.3`, `pressure_bw=0.6`, `pressure_queue=0.5`, `pressure_mstall=0.4`

### 6.4 Initial Target Selection

| Class | Initial sim_ctas |
|---|---|
| COMPUTE | `max(k_reps, total_sms / 2)` |
| MEMORY | `max(k_reps, N_sat_est)` |
| MIXED | `max(k_reps, ceil(1.5 × N_sat_est))` |

Where `N_sat_est = ceil(peak_dram_bw / per_sm_bw_est)` from the K-rep run.

### 6.5 Pilot Loop

**Stop conditions** (any of):
- `sim_ctas == total_ctas` — full grid reached
- `iter > max_doublings` — cap hit
- `achieved_bw_ratio >= stop_bw_target` (default 0.8)
- BW delta < 5% AND IPC delta < 5% — stable

**Force-expansion triggers** (force iter-0 rejection so pilot expands):
- `full_ctas_per_sm > 4 × sampled_ctas_per_sm` — high projection ratio
- `undersized && mem_stall_frac > 0.10` — small sample + memory stalls

### 6.6 Cycle Estimation — Evolution

**Planned (Phase B):** `per_cta` mode for compute, `steady_state` wave-count mode for memory.

**Replaced by (accuracy push):** Unified throughput-conservation formula:
```
sampled_active = min(sampled_ctas, total_sms)
full_active    = min(total_ctas,   total_sms)
scale          = (total_ctas / full_active) / (sampled_ctas / sampled_active)
est_cycles     = sampled_cycles × scale
```
Active SMs capped at `total_sms` (NOT `total_sms × max_cta_per_core`).

**Further replaced by:** Log-fit concurrency-throughput model:
```
T(N) = a + b × log(N + 1)     (N = CTAs per SM)
```
Pilot iteration history provides `(N, T)` points; least-squares fit gives `(a, b)`. Extrapolated to full-grid CTAs-per-SM density. Falls back to throughput-conservation formula when <2 distinct densities.

### 6.7 Validation Targets

| Metric | Target |
|---|---|
| p50 cycle error | < 15% |
| p90 cycle error | < 25% |
| Instruction count error | < 5% (weight scaling) |

---

## 7. Stat Aggregation Rules

| Stat | Rule |
|---|---|
| `gpu_tot_sim_cycle` | Raw sampled-wave wall-clock (unscaled) |
| `gpu_tot_sim_cycle_estimated` | Model-projected whole-kernel cycle |
| `gpu_tot_sim_insn` | × weight (total_ctas / sim_ctas) |
| `gpu_tot_issued_cta` | × weight |
| IPC | `weighted_insns / weighted_cycles` (recomputed from weighted numerators) |
| Hit rates | `weighted_hits / weighted_accesses` |

Ratio metrics are never averaged directly — always recomputed from weighted numerators/denominators.

---

## 8. Implementation Task Map

| Task | Status | Files |
|---|---|---|
| K-rep CTA selection | Done | `main.cc` |
| Stat scaling (insn, CTA count) | Done | `gpu-sim.cc`, `gpu-sim.h` |
| Pressure signal extraction | Done | `gpu-sim.cc`, `gpu-sim.h` |
| Per-class instruction counters | Done | `gpu-sim.h`, `gpu-sim.cc` |
| Refined `kernel_ai` with weights | Done | `trace_driven.cc` |
| Memory-stall fraction | Done | `gpu-sim.cc`, `subcore.cc` |
| 3-way classifier | Done | `main.cc` |
| N_sat_est + initial sim_ctas | Done | `main.cc` |
| Stratified shuffle replication | Done | `main.cc` |
| Adaptive pilot loop | Done | `main.cc` |
| Pilot snapshots (rollback) | Done | `gpu-sim.cc`, `gpu-sim.h` |
| Throughput-conservation cycle formula | Done | `gpu-sim.cc` |
| Log-fit concurrency model | Done | `main.cc` |
| Force-expand on projection ratio | Done | `main.cc` |
| Validation harness | Done | `util/cta_sampling/validate.py` |
| Two-pass clustering (Approach 2) | Deferred | — |
| Whole-kernel cycle model (multi-wave) | Deferred (absorbed into log-fit) | — |

---

## 9. Open Decisions (pre-implementation)

| # | Decision | Resolution |
|---|---|---|
| D1 | Phase A first, or A+B parallel? | A first, then B |
| D2 | AI weights calibrated or guessed? | Guessed (2/8/4); calibration is follow-up |
| D3 | Add `gpu_tot_sim_cycle_estimated` or replace raw? | Add (backward-compatible) |
| D4 | Extra validation workloads? | rodinia2/Turing: bfs, srad_v2, lud added |
| D5 | mem_stall threshold default? | 0.4 |

---

## 10. Acknowledged Limitations

1. **Arithmetic-intensity proxy is crude** — `kernel_ai = compute_ops / dram_bytes` still over-counts. Fine-grained ALU / load-store / tensor counter separation is a future improvement.
2. **Whole-kernel cycle estimation is deferred** — `gpu_tot_sim_cycle` is sampled-wave wall-clock; a proper multi-wave model was deferred.
3. **Pilot loop requires `window_size==1`** — auto-disables for concurrent kernels.
4. **K-rep collapses on 1D grids** — corner+midpoint heuristic gives only 2–3 unique CTAs on 256×1×1 grids. Fixed by force-expand heuristic, but sampling quality remains unrepresentative for some kernels.
5. **TMA kernels not modeled** — FlashAttention-3 uses H100 TMA (UTMALDG.4D) which bypasses standard memory pipelines. Memory pressure signals (dram_bytes, achieved_bw_ratio) do not capture TMA transaction behavior. See `.plan/TMA_TRACING.md`.
