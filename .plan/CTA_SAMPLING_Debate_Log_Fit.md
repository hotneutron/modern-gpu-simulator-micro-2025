# CTA Sampling — Debate: Replacing the Log-Fit Concurrency Model

## Context

The `cta-sampling` branch on `modern-gpu-simulator-micro-2025` implements a GPU microarchitecture simulator that samples a representative subset of CTAs (cooperative thread arrays) from a kernel and projects whole-kernel cycle counts from the sampled wave. The projection is done via a concurrency-throughput model that maps CTAs-per-SM density to per-SM issue throughput.

The current implementation uses a **log-fit model**:

```
T(N) = a + b × log(N + 1)
```

where `N` is CTAs per SM and `T(N)` is the per-SM throughput (in cycles per CTA wave). The pilot loop records `(sampled_ctas, sampled_cycles, active_sms)` at each accepted iteration. At accept time, a least-squares fit over per-density-aggregated points produces `(a, b)`. `est_cycles` is then computed by extrapolating `T(N)` at the full-grid CTAs-per-SM density.

This debate addresses a legitimate critique: the `log(N+1)` functional form lacks fundamental physical support. It is an empirical curve-fit with no microarchitectural grounding, no saturation mechanism, and no connection to hardware limits. When asked to extrapolate far beyond the sampled range (e.g., fitting on N=1 and N=2 to predict N=23, as with the `nn` kernel at 938 CTAs), the fit produces physically impossible results (throughput exceeding hardware limits, or systematically under-predicting by ~2× as observed in the `nn` failure mode at +103.5% error).

Two candidate replacements are evaluated: a **Saturating Exponential** model and a **Roofline-Bounded** model.

---

## Candidate 1: Saturating Exponential

### Functional Form

```
T(N) = T_max × (1 − exp(−k × N))
```

Where:
- `T_max` is the hardware saturation throughput (issue bandwidth ceiling per SM)
- `k` is the "characteristic CTA density" (units: inverse CTA/SM) — how quickly the system approaches saturation
- `N` is CTAs per SM

### Physical Basis

This form originates from queueing theory — specifically the cumulative distribution function of an exponential distribution, or the blocked-state probability in an M/M/1 queue. In GPU microarchitectural terms, it models the probability that a warp scheduler has a ready warp to issue at any given cycle as a function of concurrent CTA density. As CTA density increases, the probability that all CTAs are simultaneously stalled approaches zero, and issue bandwidth approaches its structural maximum.

The key physical claim is: **throughput is concave and saturates at a hardware-defined maximum**. This matches the observed diminishing marginal returns of adding more concurrent CTAs per SM due to latency hiding. The `log(N+1)` form also exhibits concavity but lacks an asymptote — it grows without bound, which is physically impossible.

### Strengths

1. **Built-in saturation.** Unlike the log-fit, the exponential naturally asymptotes to `T_max`. Extrapolating to N=23 cannot produce a throughput exceeding the hardware ceiling — the math enforces physical limits.
2. **Only 2 parameters.** `T_max` and `k` are solvable with as few as 2 data points via nonlinear least squares (Levenberg-Marquardt or simple grid search over `k`). Tractable for the pilot loop's 2–4 data points.
3. **Physically interpretable parameters.** `k` has units of inverse CTA/SM — it represents the "CTA density at which the SM is ~63% saturated." A compute-bound kernel with short-issue latencies would have a high `k` (saturates quickly with few CTAs); a memory-bound kernel with long DRAM latency would have a low `k` (requires many CTAs to hide stalls). This gives the parameter direct microarchitectural meaning, unlike the dimensionless coefficients `(a, b)` in the log-fit.
4. **Smooth interpolation.** Between sampled points, the exponential is smooth and concave, matching the latency-hiding physics.
5. **Handles far extrapolation better than log-fit.** Because of the explicit asymptote, the exponential cannot produce physically absurd throughput values even when extrapolating 10× past the sampled range. The log-fit, which has no ceiling, is unbounded upward and produced a +103.5% error on `nn` exactly because it extrapolated unboundedly.

### Weaknesses

1. **`T_max` is not directly known a priori.** The `log(N+1)` fit implicitly embeds saturation by bending downward, but all of its "saturation" behavior is baked into the fitted `(a, b)` — it is not connected to any hardware limit. The exponential requires `T_max` to be specified. Options:
   - (a) Estimate `T_max` as a third free parameter in the fit — this becomes a 2-parameter nonlinear fit with 2–4 data points, which is tight but potentially solvable.
   - (b) Derive `T_max` from static analysis (kernel instruction mix + hardware peak throughput). Ahmed already has per-class instruction counters from Phase A (FP, INT, DP, TC, SFU, loads, stores). If we have hardware peak rates for each class, we can compute a `Peak_Issue_Rate` per SM. This is the physically correct approach but requires non-trivial additional instrumentation and knob-setting.
2. **The functional form assumes a single latency bucket.** GPU workloads have heterogeneous latencies (L1 hit ~10 cycles, L2 hit ~200 cycles, DRAM ~400+ cycles). A single exponential with one `k` fits a single "average" latency rather than modeling the multi-modal stall distribution explicitly. In practice, this is also true of the log-fit, but the exponential makes the assumption more transparent.
3. **Numerical instability at small N ranges.** If the pilot loop produces data points that are too close together in CTA/SM density (e.g., N=1 and N=1.2), the exponential fit is ill-conditioned — the optimizer can drive `T_max` to infinity and `k` toward zero and still fit the near-linear region. The pilot loop's doubling strategy (1, 2, 4, 8 CTAs/SM) is specifically designed to avoid this, but it must be respected.
4. **Implementation complexity.** Unlike the linear least-squares used by the log-fit, exponential fitting requires nonlinear optimization. Levenberg-Marquardt is the standard approach, but a simple grid-search over `k` (computing `T_max` analytically for each `k`) is simpler to implement and numerically stable for this use case.

---

## Candidate 2: Roofline-Bounded Extrapolation

### Functional Form

Keep the existing log-fit (or exponential) as the primary empirical model, but **clamp** its output to never exceed a Roofline-computed upper bound:

```
T_est(N) = min(T_fit(N), T_roofline(N))
```

The Roofline bound is computed from the kernel's Arithmetic Intensity (AI) and hardware peak rates:

```
kernel_ai = compute_ops / dram_bytes
roofline_throughput = min(Peak_FLOPS, Peak_BW × kernel_ai)
T_roofline = total_work / roofline_throughput  [converted to cycles per SM]
```

Where `compute_ops = FP + INT + W_dp·DP + W_tc·TC + W_sfu·SFU` (the refined AI from Phase A).

### Physical Basis

The Roofline model (Williams et al., 2009) provides a hard upper bound on achievable performance for a given kernel AI and hardware peak rates. It is derived from conservation laws (bytes moved, FLOPs executed) and structural hardware limits (peak DRAM bandwidth, peak compute throughput). It is not an empirical fit — it is a theoretical ceiling based on arithmetic.

### Strengths

1. **Physically rigorous upper bound.** The Roofline bound cannot be exceeded by any real execution of the kernel on the given hardware. Adding it as a clamp provides a provably correct ceiling on the estimated throughput.
2. **Uses existing instrumentation.** Ahmed already computes `kernel_ai` with per-class weights in Phase A, has `peak_dram_bw` from config, and has `achieved_bw_ratio`. The Roofline bound can be computed at accept time without new simulator instrumentation.
3. **Compositional and low-risk.** You do not replace the existing log-fit — you add a single `min()` clamp. This is a minimal code change that preserves all existing behavior while preventing catastrophic extrapolation. If the fit is good within the sampled range, the Roofline bound is never active. If the fit extrapolates badly (like on `nn`), the bound activates and caps the error.
4. **No additional free parameters.** Unlike the exponential, which requires fitting `T_max` and `k`, the Roofline bound is computed deterministically from existing signals and hardware constants.

### Weaknesses

1. **The Roofline is a per-SM ceiling, not a direct throughput predictor.** The Roofline gives maximum achievable *FLOPS* or *GB/s* for a given AI, but translating that to "cycles for this kernel" requires an extra unit conversion. Specifically: `roofline_throughput` is in FLOPs/cycle or Bytes/cycle per SM. To get total cycles, we need `total_work / roofline_throughput`, where `total_work` must be expressed in the same units (FLOPs or Bytes). Ahmed's cycle estimator works in cycles directly, not FLOPs or bytes. Bridging this requires careful unit analysis and may introduce new sources of error.
2. **The kernel's AI is measured from the *sampled* wave at low CTA/SM.** At low concurrency, the kernel's achieved bandwidth and compute rates may not reflect full-grid behavior due to cold caches, missing bank-level parallelism, and partial utilization of DRAM ports. The AI measured at N=2 CTAs/SM may differ meaningfully from the AI at N=23 CTAs/SM. If the sampled AI underestimates the full-grid AI (e.g., because L2 is cold during the sample), the Roofline bound will be too conservative and the extrapolation will be unnecessarily capped.
3. **Clamping is not fitting.** Taking `min(fit_output, roofline_bound)` is a heuristic, not a model. If the fit systematically under-predicts by 20% (as the log-fit does on `nn`), the Roofline clamp only prevents catastrophic overruns — it does not fix systematic bias. The fundamental problem of the log-fit lacking physical support is only partially addressed.
4. **For kernels that are not Roofline-limited**, the bound may be loose. For compute-bound kernels, `Peak_FLOPS` is the ceiling, but the actual achieved throughput may be far below peak due to pipeline hazards, branch divergence, or instruction mix. The Roofline bound provides no guidance here.

---

## Core Tension

The fundamental tension between these two approaches is **where to place the physical prior:**

| | Saturating Exponential | Roofline-Bounded Fit |
|---|---|---|
| **Physical prior location** | In the functional form (`T_max` asymptote) | In the upper ceiling (Roofline clamp) |
| **Assumption** | Throughput saturates exponentially toward a maximum | Throughput cannot exceed Roofline-computed bound |
| **What is unknown** | `T_max` must be estimated or derived | Bound may be weak if AI is poorly measured from sample |
| **Handles far extrapolation** | Better (explicit asymptote) | Only prevents catastrophe, does not fix bias |
| **Handles near interpolation** | Good (convexity matches latency hiding) | Good (clamp is inactive in-range) |
| **Free parameters** | 1–2 (depending on whether `T_max` is derived or fitted) | 0 (deterministic) |
| **Implementation complexity** | Medium (nonlinear least squares) | Low (deterministic formula + `min()`) |

The exponential model is **stronger in principle for far-extrapolation**, because it has a built-in saturation mechanism. But it requires knowing or estimating `T_max` — and if `T_max` is derived from the Roofline, you've reinvented Option 2 in a more complicated way.

The Roofline-bound approach is **stronger as a safety net than as a primary model.** Using it to clamp the existing log-fit is a low-risk improvement. But if the goal is to replace the log-fit with something that has fundamental physical support, simply adding a clamp on top of an empirical fit still leaves the core problem unsolved.

---

## Hybrid Proposal: Roofline-Tied Exponential

The most defensible approach combines the strengths of both:

```
T(N) = T_roofline × (1 − exp(−k × N))
```

Here:
- `T_roofline` is computed once from the kernel's AI (measured from the sample) and hardware peak rates — this is the physical ceiling, derived from conservation laws.
- `k` is the only free parameter — a 1-parameter fit solvable with as few as 2 data points (e.g., N=1, N=2 from the pilot history).
- `N` is CTAs per SM at the target density.

This gives:
- **Correct asymptotic behavior** (hardware-physical ceiling tied to Roofline)
- **Correct concave shape** (diminishing returns from latency hiding)
- **Minimal free parameters** (1 parameter, 2 data points needed)
- **No catastrophic extrapolation** even if N is far outside the sampled range

### Open Question

The main unresolved issue is whether the kernel's AI measured at low CTA/SM density (sample) is a sufficiently accurate proxy for the full-grid AI. If the sample's L2 miss rate and DRAM access pattern differ significantly from the full grid (e.g., due to cold-cache effects at low concurrency), the Roofline `T_roofline` could be systematically off. This is the same concern that applies to the pure Roofline-bound approach, but here it only affects `T_roofline` (the asymptote), not the shape of the curve.

If the AI measured at low density is a poor proxy, an alternative is to fit `T_max` as a free parameter (making this a 2-parameter exponential) and use the Roofline only as a sanity check — but then we are back to the original 2-parameter exponential problem.

---

## Summary

| Criterion | Log-Fit (current) | Saturating Exponential | Roofline-Bounded | Hybrid (Roofline-Tied Exponential) |
|---|---|---|---|---|
| Physical basis | None (empirical) | Queueing theory (single latency bucket) | Conservation laws + hardware peaks | Queueing theory + Roofline ceiling |
| Saturation mechanism | None (unbounded) | Yes (`T_max`) | Yes (Roofline clamp) | Yes (`T_roofline`) |
| Free parameters | 2 (`a`, `b`) | 1–2 (`k`, optionally `T_max`) | 0 | 1 (`k`) |
| Far extrapolation behavior | Unbounded (catastrophic) | Bounded by `T_max` | Bounded by Roofline | Bounded by `T_roofline` |
| Near interpolation | Good | Good | Good | Good |
| AI measured from sample | N/A | Used to derive `T_max` (optional) | Directly | Directly |
| Implementation complexity | Low | Medium | Low | Medium |
| Handles `nn`-style failure | Poorly | Well | Partially (clamps only) | Well |
| Failure modes | Extrapolates unboundedly | `T_max` unknown; needs derivation | Bound may be weak if AI is poor | AI proxy concern remains |

The log-fit's fundamental flaw is not its shape — the `log(N+1)` concave form is reasonable for latency-hiding physics — but its **lack of an asymptote and lack of connection to hardware limits**. Any replacement should address both. The **Roofline-Tied Exponential** is the most physically grounded option among the candidates, trading only implementation complexity for a model that is both theoretically justified and practically tractable with the pilot loop's data constraints.
