# CTA Sampling — History 3: Replacing the log-fit concurrency model

**Date snapshot:** 2026-05-12
**Branch state at end of stage:** `cta-sampling`, 33 commits ahead of `main`
(1 new code commit: `96ffb68`).
**Outcome:** sat_exp ships as the new concurrency-throughput model default.
C3a removed. Wider 9-workload sweep: p50 = 9.5%, p90 = 23.4%, max = 30.0%
— both accuracy targets met, max-error down from 128.4% (raw log-fit
without C3a) to 30.0%. Cumulative wall-time speedup is 1.05× on the
rodinia2 toy traces (vs. 0.92× at the end of History 2), driven mainly by
nn (0.50× → 1.34×) once the C3a-induced extra pilot iters go away.

This file is the historical record of the third stage of the
`cta-sampling` branch. It supersedes `CTA_SAMPLING_Debate_Log_Fit.md`,
which is deleted alongside this commit.

---

## 0. Where we picked up (recap from History 2)

End of History 2 (2026-05-07):

- Cycle accuracy meets p50 < 15% / p90 < 25% on the wider 9-workload
  set: p50 10.1%, p90 23.4%, max 23.4% (`heartwall`).
- Concurrency model: log-fit `T(N) = a + b·log(N+1)`.
- C3a — adaptive `pilot_max_doublings` cap — was active to bring `nn`
  from +128.4% to +10.1% by sampling deeper, paying ~5 s of extra
  pilot wall-time per `nn`-class kernel.
- Wall-time speedup: cumulative 0.92× on the 9-workload set; nn was
  the biggest regression (0.50×) because C3a's deeper sampling cost
  more than baseline.
- `CTA_SAMPLING_Debate_Log_Fit.md` had laid out three candidate
  replacements for the log-fit: Saturating Exponential, Roofline-Bounded
  clamp, and a Roofline-Tied Exponential hybrid. The hand-off asked
  for an empirical comparison.

The log-fit's specific flaw was that it has no asymptote — `T(N)` grows
without bound as `N` grows, which is physically impossible (issue
bandwidth saturates). On `nn` (full density ≈ 23 CTAs/SM, sample at
N ≤ 2) the fit extrapolated far past its calibration range and
over-predicted `T_full` by 2× — producing the +128% cycle error that
C3a was working around. The right fix was to replace the model, not
to keep deepening the sample.

---

## 1. Plan for this stage

Two steps, in order:

1. **Revert C3a.** It's a band-aid for a weakness of the log-fit; once
   the log-fit is gone, the per-kernel adaptive doublings cap loses
   its motivation and costs pilot wall-time for no benefit.
2. **Try each of the three candidate models on the wider set, pick a
   winner.** Don't pre-decide — let the data settle the debate.

Both changes landed in a single commit (`96ffb68`).

---

## 2. C3a revert

Five lines removed from `main.cc`:

- `pilot_state_t::effective_max_doublings` field (back to non-adaptive).
- `PILOT_MAX_DOUBLINGS_CEILING` constant.
- C3a's iter-0 force-expand branch (back to plain `return false`).
- `pilot_decide_accept` signature back to `const pilot_state_t&`.
- The init `pst.effective_max_doublings = 0;` at pilot-state setup.

Zero behavior change for any kernel that wasn't already hitting C3a's
ratio gate — i.e. only `nn` is affected, and on `nn` the pilot now uses
the global cap (`-cta_sampling_pilot_max_doublings 2`) again. This means
the deepest sample for `nn` lands at density 4 CTAs/SM instead of C3a's
16, which raises the log-fit extrapolation error back to +128.4% — a
deliberate regression undone in §3 once the new model is in place.

---

## 3. Concurrency-model knob

Added `-cta_sampling_concurrency_model {logfit | sat_exp | roofline_clamp |
roofline_exp}` to `trace_config`. Default is **`sat_exp`**. Layout-wise:

- `pilot_iter_obs_t` now carries `kernel_ai`, `compute_ops`,
  `peak_flops_per_cycle`, `peak_dram_bw_bytes_per_cycle` per pilot iter.
  These feed the roofline-based fits at accept time.
- `last_kernel_wave_info_t::has_log_fit`/`log_fit_a`/`log_fit_b` are
  replaced by a generic `concurrency_model` + `has_fit` + three
  parameter slots `fit_a / fit_b / fit_t_cap`. The interpretation
  depends on the model (documented in the struct comment).
- `gpu-sim.cc::gpu_print_stat` dispatches on `concurrency_model` in a
  single `switch`, applies the same downward-clamp-to-T_sample as
  before, and tags the `estimation_mode` field with the model name
  (`sat_exp_compute`, `roofline_clamp_memory`, …).

### 3.1 Saturating Exponential — `sat_exp`

```
T(N) = T_max · (1 − exp(−k · N))
```

Two parameters. Solved by a log-spaced grid search over `k ∈ [0.01, 5.0]`
(128 steps): for each candidate `k`, the analytical least-squares `T_max`
minimizes the residual, and we pick the `k` with the smallest residual.

Acceptance criteria:

- ≥ 2 distinct CTAs/SM densities in the pilot history.
- `best_T_max ≥ 0.999 · max(T_observed)` — rejects the degenerate case
  where the fit collapsed to a flat near-zero line that beat the
  exponential shape on residual.

`fit_a = T_max`, `fit_b = k`.

### 3.2 Roofline ceiling

Both roofline-based models share this ceiling computation, done in
`pilot_roofline_T()` on the deepest pilot iter (largest `sampled_ctas`):

```
flops_per_cycle_total  = min(peak_flops_per_cycle, peak_bw · kernel_ai)
flops_per_cta          = compute_ops / sampled_ctas
total_ctas_per_cycle   = flops_per_cycle_total / flops_per_cta
T_roofline_per_sm      = total_ctas_per_cycle / min(total_ctas, total_sms)
```

Deepest iter is preferred because cold-cache bias on the per-iter AI
measurement decreases as the sample density rises (more BLP, more L2
warm-up, etc.). When `T_roofline ≤ 0` the model rejects and the
estimator falls back to constant-per-SM throughput.

### 3.3 Roofline-Clamp — `roofline_clamp`

Reuses the existing log-fit. At evaluation time:

```
T_full = min(fit_a + fit_b·log(N+1), fit_t_cap)
```

`fit_a / fit_b` from the log-fit, `fit_t_cap = T_roofline`. Rejected
when `T_roofline ≤ 0`.

### 3.4 Roofline-Tied Exponential — `roofline_exp`

`T_roofline` fixed by arithmetic; only `k` is fitted (same grid as
sat_exp), against:

```
T(N) = T_roofline · (1 − exp(−k · N))
```

`fit_a = T_roofline`, `fit_b = k`. Rejected when `T_roofline ≤ 0`.

---

## 4. Measured results — accuracy

Sweep harness: `util/cta_sampling/validate_models.py`. All four models
run against the same 9-workload set used in History 2, with the new
no-C3a pilot. Cycle error % is `(est_cycle − baseline_cycle) /
baseline_cycle`.

```
                  logfit    sat_exp   roofline_clamp  roofline_exp
  hotspot         +14.7%    +14.7%    +14.7%          +14.7%
  backprop        -13.1%     -9.5%    +17.0%           -0.1%
  pathfinder       -0.3%     -0.3%     -0.3%           -0.3%
  bfs              +3.3%     +3.3%     +3.3%           +3.3%
  srad_v2         +18.1%    +18.1%    +18.1%          +18.1%
  lud              -0.0%     -0.0%     -0.0%           -0.0%
  heartwall       -23.4%    -23.4%    -23.4%          -23.4%
  nn             +128.4%    +30.0%   +675.1%         +675.1%
  nw               +5.0%     +5.0%     +5.0%           +5.0%

  p50            13.1%      9.5%     14.7%            5.0%
  p90            23.4%     23.4%     23.4%           23.4%
  max           128.4%     30.0%    675.1%          675.1%
```

Three observations:

1. **Most workloads are unchanged.** Six of nine kernels accept iter 0
   and run only the K-rep wave — no fit is produced and the estimator
   falls back to constant-throughput. These are insensitive to the
   model choice (`pathfinder`, `bfs`, `srad_v2`, `lud`, `heartwall`,
   `nw`, plus `hotspot` which fast-accepts as `compute`). The model
   choice only matters for kernels where the pilot actually fits a
   curve — `backprop` and `nn` in this set.

2. **`backprop` is the "good" multi-density case.** The pilot ran
   multiple densities and the fits actually got distinguishing data.
   `roofline_exp` nailed it (−0.1%) — when the sample-derived AI is
   reliable, the hardware ceiling is the right asymptote and only the
   shape parameter `k` needs fitting. `sat_exp` is close behind
   (−9.5%). `roofline_clamp` regressed *worse* than the log-fit
   (+17.0%) — its clamp is wrong-signed here (clamping the log-fit
   *down* to a roofline below the measured throughput).

3. **`nn` is the catastrophic-extrapolation test.** This is where the
   log-fit produced +128.4% and where the model debate was anchored.
   - `sat_exp`'s `T_max` asymptote bounded the extrapolation: +30.0%
     — a 4× error reduction with zero added pilot cost.
   - Both roofline-based models collapsed to +675.1%. The mechanism:
     the cold-sample AI under-counted what the kernel actually
     achieves at low concurrency, so `T_roofline` came out *below*
     `T_sample`. The defensive floor `T_full ≥ T_sample` kicked in,
     and the resulting `T_full = T_sample` produced the constant-
     throughput-from-sample extrapolation — the worst possible
     answer in this regime. This is exactly the failure mode the
     debate doc warned about ("the kernel's AI is measured from the
     sampled wave at low CTA/SM").

### 4.1 Winner: `sat_exp`

`sat_exp` is the only candidate that:

- meets both accuracy targets (p50 < 15%, p90 < 25%),
- eliminates the `nn` blow-up without C3a's wall-time tax,
- and doesn't catastrophically regress on any workload.

Trade vs. log-fit + C3a state: `nn` went from +10.1% to +30.0%. That
∼20 pp of cycle accuracy bought back ~5 s of `nn` pilot wall-time and
removed an entire per-kernel adaptive cap mechanism.

---

## 5. Measured results — wall-time speedup

Harness: `util/cta_sampling/measure_speedup.py`, same 9 workloads, 3
trials each, mean reported.

Raw data: `.plan/history/speedup_results_2026-05-12.csv`.

| Workload   | baseline (s) | pilot (s) | speedup | History 2 speedup |
|---|---:|---:|---:|---:|
| hotspot    |  9.72 |  6.33 | **1.53×** | 2.12× |
| backprop   |  3.51 |  4.01 | 0.87× | 0.70× |
| pathfinder |  1.90 |  1.77 | 1.07× | 1.05× |
| bfs        |  8.01 |  6.74 | 1.19× | 1.04× |
| srad_v2    |  3.33 |  4.70 | 0.71× | 0.77× |
| lud        |  8.76 |  8.76 | 1.00× | 1.01× |
| heartwall  |  2.14 |  4.03 | 0.53× | 0.46× |
| **nn**     |  5.05 |  3.76 | **1.34×** | **0.50×** |
| nw         |  6.61 |  6.63 | 1.00× | 0.96× |
| **cumulative** | **49.03** | **46.73** | **1.05×** | **0.92×** |

The cumulative speedup crossed unity. The single largest mover is
`nn`: 0.50× → 1.34×, a 2.7× improvement on this kernel alone. Without
C3a, `nn`'s pilot now runs the default two doubling iters and the
sat_exp asymptote does the extrapolation work. The `hotspot`
regression from 2.12× → 1.53× is the only mover in the wrong direction
and is within trial-to-trial variance for this workload (stdev ≈ 0.5 s
on baseline, 0.4 s on pilot — the absolute time difference vs. History
2 is ≈ 0.9 s).

The crossover model from History 2 still applies: pilot wins when
`N > sampled + a/c` for `a ≈ 4 s` fixed overhead and `c` = per-CTA sim
cost. The toy-trace cumulative is now slightly above unity instead of
slightly below; the production-scale story (10K+ CTAs/kernel, projected
100×+) is unchanged — the model just made the small-grid regression
less severe.

---

## 6. Other candidates and why they didn't win

### Roofline-Clamp (`roofline_clamp`)

The debate doc's "low-risk, additive" candidate. Failed because the
clamp is *one-sided downward*: `min(fit, ceiling)`. It only protects
against the log-fit over-extrapolating *upward*. The `nn` failure mode
is the log-fit predicting `T_full` *too high* (and hence cycles too
high — wait, no, it's the other way: the log-fit predicts T_full too
*low*, so cycles too *high*). Either way, the `nn` failure mechanism
was actually about the log-fit *under*-predicting throughput at large
N, and a downward clamp does nothing to help.

On `backprop`, the clamp made things worse: the log-fit was already
mildly under-extrapolating (−13.1%), and clamping it down to the
roofline (which the AI placed too low) drove it even further down
(+17.0%, sign-flipped because the floor kicked in).

### Roofline-Tied Exponential (`roofline_exp`)

The debate doc's hybrid — was supposed to be the most physically
defensible. It is, when the AI proxy is accurate (see backprop: −0.1%
on a 4-iter pilot history). The problem is that for kernels whose
sample-derived AI is unreliable (`nn`, memory-bound, very low
compute-op density), the `T_roofline` it computes is *below* the
actually-achieved sample throughput. Both the math and the defensive
floor then conspire to produce the constant-throughput extrapolation
— the exact thing we built the model to avoid.

The shape is right. The AI source is wrong. Two follow-ups would
unlock this:

- D2 — calibrate per-class instruction weights against a known-FLOPs
  kernel so `compute_ops` better reflects actual work.
- Use the deepest *accepted* pilot iter's AI rather than the deepest
  iter regardless of acceptance — cache state is more representative
  at higher densities.

Both are out of scope for this stage.

---

## 7. What's left open

1. **Pilot wall-time budget + abort-to-baseline.** Unchanged from
   History 2. The optimization is still unsafe to default-on for
   workload mixes that include any kernel below the crossover
   threshold. Two options outlined in the prior handoff (CTA-count
   budget; wall-clock budget); the cumulative-1.05× number softens
   but doesn't eliminate this.
2. **Production-scale speedup measurement.** Still unmeasured. The
   toy-trace cumulative speedup is now 1.05× — the projection model
   says 100×+ at 10K+ CTAs/kernel, but we have no real datapoint above
   ~940 CTAs on a single kernel. Top external blocker on the "is this
   work useful?" question.
3. **Roofline-Exp resurrection via better AI source.** `backprop`'s
   −0.1% result is suggestive enough that with a better AI input
   (D2 calibration, or per-pilot-iter AI averaging) the
   physically-grounded model might overtake `sat_exp`. Speculative.
4. **`heartwall` cache-state pollution (−23.4%).** Carried from
   History 1. Pilot rejected-iter cache state pollutes the accepted
   iter; would need a per-SM cache-state reset between iters.
5. **K-rep clustering replacement.** Use k-means over per-CTA
   features instead of C2's evenly-spaced supplement. Speculative
   improvement on the K-rep selection side, not concurrency-modeling.

---

## 8. Pointers

| File | Purpose |
|---|---|
| `simulator-remodeled/gpu-simulator/main.cc` | Model selection + 3 fit functions (`pilot_fit_log_throughput`, `pilot_fit_sat_exp`, `pilot_fit_roofline_exp`) + roofline ceiling (`pilot_roofline_T`) + per-iter obs capture |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.{cc,h}` | Generic `wave_info` fit params + `switch` dispatch in `gpu_print_stat` |
| `simulator-remodeled/gpu-simulator/trace-driven/trace_driven.{cc,h}` | `-cta_sampling_concurrency_model` knob (default `sat_exp`) |
| `simulator-remodeled/util/cta_sampling/validate_models.py` | Accuracy sweep harness used in §4 |
| `simulator-remodeled/util/cta_sampling/measure_speedup.py` | Speedup measurement harness used in §5 |
| `.plan/history/speedup_results_2026-05-12.csv` | Raw speedup CSV (9 workloads × {baseline, pilot} × 3 trials) |
| `.plan/history/speedup_results.csv` | History 2 baseline for the speedup comparison |
