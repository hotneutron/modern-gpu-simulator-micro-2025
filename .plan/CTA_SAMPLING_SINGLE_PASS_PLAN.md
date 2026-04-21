# CTA Sampling Single-Pass Plan

## Goal

- Keep CTA sampling as a single simulation pass (no pilot reruns, no multi-pass adaptation).
- Improve accuracy by conservative one-shot sizing of active SMs/CTAs.
- Preserve speedup while reducing under-modeled inter-CTA contention.

## Constraints

- One sampled run only.
- No iterative tuning loops.
- Minimal new complexity in runtime flow.

## One-Shot Strategy

### 1) Select CTA representatives (existing)

- Use coordinate-based representatives (corners, edges, interior).
- Keep K small for trace/simulation speed.

### 2) One-shot kernel characterization

From the same sampled run, compute:

- `kernel_ai = effective_ops / dram_bytes`
- `ridge_point = peak_flops / peak_dram_bw`
- `ridge_ratio = kernel_ai / ridge_point`

And read pressure proxies from that run:

- memory-stall fraction
- achieved DRAM BW ratio
- L2 miss traffic

### 3) Single decision rule for active SMs

Use a conservative one-shot mapping:

- If clearly compute-heavy (`ridge_ratio >= 1.5` and low memory pressure):
  - `sim_sms = max(K, ceil(0.35 * total_sms))`
- If clearly memory-heavy (`ridge_ratio <= 0.85` or high memory pressure):
  - `sim_sms = max(K, ceil(0.75 * total_sms))`
- Otherwise (mixed/uncertain):
  - `sim_sms = max(K, ceil(0.55 * total_sms))`

Then clamp:

- `sim_sms = min(sim_sms, total_sms)`

Rationale:

- This avoids brittle `num_SMs_to_saturate` estimation.
- It intentionally over-provisions SM concurrency for memory/mixed cases to preserve contention.
- It remains single-pass and predictable.

### 4) Fill CTA slots in one shot

- Replicate representative CTAs to populate `sim_sms` active slots.
- Use shuffled/stratified assignment across SMs (not naive periodic replication) to reduce mapping artifacts.

### 5) Stat scaling semantics

- Scale additive counts (instructions, CTAs, bytes, transactions) by weight.
- Recompute ratios from weighted numerators/denominators; do not directly average ratios.
- Keep cycle semantics explicit in docs:
  - If cycles are one-wave sampled time, leave unscaled.
  - If whole-kernel estimate is needed, apply a documented wave model.

## Inter-CTA Contention Guardrails (single-pass compatible)

Use static guardrails to decide conservative class in Step 3:

- High L2 miss + high BW ratio => treat as memory-heavy.
- High atomics intensity => treat as memory-heavy.
- Near ridge (`0.85 < ridge_ratio < 1.5`) => treat as mixed, not compute.

If any guardrail triggers uncertainty, bias upward to the next more conservative SM tier.

## Why this is better than adaptive multi-pass

- No extra runs or control-loop complexity.
- No sensitivity to noisy pilot deltas.
- Easy to reason about and implement in current flow.
- Still addresses the major failure mode of K-only under-filling.

## Validation Targets

Evaluate against full simulation on representative kernels:

- compute-regular
- memory-regular
- reduction
- irregular sparse/graph

Track:

- cycle error
- IPC error
- DRAM bytes error
- speedup

Initial targets:

- median cycle error < 10%
- p90 cycle error < 18%
- speedup > 15x on large kernels

## Implementation Tasks

1. Add one-shot classifier using `ridge_ratio` + pressure proxies.
2. Add tiered `sim_sms` policy (`35% / 55% / 75%` of total SMs).
3. Replace naive replication with shuffled/stratified fill.
4. Audit stat aggregation for additive vs ratio correctness.
5. Add evaluation script and summary table.
