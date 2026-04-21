# CTA Sampling Revised Plan

## Goals

- Keep the current CTA representative sampling flow.
- Fix bound classification (compute vs memory) so it is robust near ridge-point cases.
- Add adaptive SM filling that preserves inter-CTA contention signals.
- Maintain high speedup while reducing risk of systematic IPC/cycle bias.

## Core Principles

- Treat `ridge_point = peak_flops / peak_dram_bw` as a prior, not a hard decision boundary.
- Use a 3-way classifier (`compute`, `memory`, `mixed`) instead of binary classification.
- Use `num_SMs_to_saturate` as an initial estimate only; validate it with pilot runs.
- Preserve contention by scaling active SMs/CTAs until pressure metrics stabilize.

## Inputs and Metrics

Collect from sampled simulation runs:

- `gpu_tot_sim_cycle`
- `gpu_tot_sim_insn`
- Memory bytes / memory transactions
- L2 miss traffic
- DRAM average queue occupancy
- DRAM achieved bandwidth ratio (`achieved_bw / peak_bw`)
- Memory stall fraction

Derived:

- `kernel_ai = effective_flops_or_alu_ops / bytes_to_dram`
- `ridge_ratio = kernel_ai / ridge_point`

Notes:

- Prefer bytes-to-DRAM over bytes-requested if available.
- If tensor/SFU ops dominate, include them in effective FLOP/ops estimation.

## Decision Algorithm

### Step 1: Baseline K-rep run

- Run K representative CTAs (existing coordinate heuristic).
- Compute `ridge_ratio` and contention indicators.

### Step 2: Initial class

- `compute` if `ridge_ratio >= T_high` and memory pressure is low.
- `memory` if `ridge_ratio <= T_low` or memory pressure is high.
- `mixed` otherwise.

Initial thresholds (to calibrate per architecture):

- `T_low = 0.9`
- `T_high = 1.3`
- memory pressure high if any of:
  - achieved BW ratio `>= 0.6`
  - DRAM queue occupancy above configured threshold
  - memory stall fraction above configured threshold

### Step 3: Choose initial active SMs

- `compute`: `sim_sms = min(total_sms, max(K, sms_floor_compute))`
- `memory`: `sim_sms = min(total_sms, max(K, N_sat_est))`
- `mixed`: `sim_sms = min(total_sms, max(K, ceil(1.5 * N_sat_est)))`

Where:

- `N_sat_est = ceil(peak_dram_bw / per_sm_bw_est)`
- `per_sm_bw_est` comes from Step-1 run and is clamped to sane bounds.

### Step 4: Pilot adaptation loop

- Run pilot at `sim_sms` and at `sim_sms2 = min(total_sms, 2 * sim_sms)`.
- Compare key pressure deltas:
  - achieved BW ratio
  - DRAM queue occupancy
  - L2 miss traffic
  - IPC
- If deltas exceed tolerance, increase `sim_sms` and repeat.
- Stop when stabilized or `sim_sms == total_sms`.

Suggested stop tolerances:

- BW delta < 5%
- Queue occupancy delta < 10%
- IPC delta < 5%

### Step 5: CTA replication/filling

- Expand representative CTA list to match target active CTAs.
- Use stratified round-robin or shuffled assignment across SMs to avoid periodic artifacts.
- Preserve per-cluster weights for final stat scaling.

## Stat Aggregation Rules

- Scale additive counts by weight:
  - instructions
  - issued CTAs
  - memory transactions/bytes
- Do not directly average ratio metrics.
- Recompute ratios from weighted numerators/denominators:
  - IPC = weighted_instructions / weighted_cycles (with explicit cycle semantics)
  - Hit rate = weighted_hits / weighted_accesses

Cycle semantics must be explicit:

- If reporting one-wave simulated time, keep cycles unscaled.
- If estimating whole-kernel time, apply a documented wave model.

## Inter-CTA Contention Guardrails

Track and gate on these effects before accepting sampled result:

- L2 slice/set contention drift
- DRAM partition camping indicators
- MSHR/miss-queue pressure saturation
- NoC/crossbar pressure proxies
- Atomic hotspot amplification
- Warp scheduler pressure (`eligible warps per cycle`)

If guardrails fail, escalate to more SMs or more representatives.

## Validation Plan

Test at least 5 kernels:

- compute-regular (e.g., GEMM)
- memory-regular (stencil)
- reduction/scan
- sparse irregular
- graph irregular

For each kernel, compare against full simulation:

- cycle error
- IPC error
- DRAM bytes error
- L2 hit-rate error
- speedup

Acceptance targets (initial):

- median cycle error < 8%
- p90 cycle error < 15%
- speedup > 20x on large kernels

## Implementation Tasks

1. Add metric extraction helpers for AI and memory-pressure signals.
2. Implement 3-way classifier and configurable thresholds.
3. Implement `N_sat_est` plus pilot adaptation loop.
4. Update CTA replication policy to stratified/shuffled fill.
5. Fix/verify stat aggregation path for additive vs ratio metrics.
6. Add validation harness and per-kernel report table.

## Deliverables

- Config knobs for thresholds and adaptation tolerances.
- Reproducible benchmark script for sampled vs full runs.
- Report with accuracy/speedup and failure-case analysis.
