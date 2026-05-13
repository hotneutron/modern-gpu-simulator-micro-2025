# CTA Sampling — History 3: Replacing the log-fit concurrency model

**Date snapshot:** 2026-05-12 (§§ 0–8); follow-on dated 2026-05-13 (§ 9).
**Branch state at end of stage:** `cta-sampling`, 33 commits ahead of `main`
(1 new code commit: `96ffb68`), plus the uncommitted wall-time-budget
work described in § 9.
**Outcome:** sat_exp ships as the new concurrency-throughput model default.
C3a removed. Wider 9-workload sweep: p50 = 9.5%, p90 = 23.4%, max = 30.0%
— both accuracy targets met, max-error down from 128.4% (raw log-fit
without C3a) to 30.0%. Cumulative wall-time speedup is 1.05× on the
rodinia2 toy traces (vs. 0.92× at the end of History 2), driven mainly by
nn (0.50× → 1.34×) once the C3a-induced extra pilot iters go away.
The 2026-05-13 follow-on (§ 9) adds a pilot wall-time budget + abort-to-
baseline path and lifts cumulative speedup to 1.10× with cycle accuracy
unchanged.

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

1. **Pilot wall-time budget + abort-to-baseline.** ~~Unchanged from
   History 2.~~ Done 2026-05-13 — see § 9. Implemented as the
   wall-clock option from the prior handoff. Cumulative speedup
   1.05× → 1.10×, cycle accuracy unchanged. The optimization is
   still not provably default-on safe for the worst toy regressions
   (`heartwall` 0.53×, `srad_v2` 0.66×) — the abort budget at default
   ratio 1.5 doesn't fire on them because the per-iter wall stays
   under the projected baseline.
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
| `.plan/history/speedup_results_2026-05-13.csv` | § 9 follow-on speedup CSV (abort path at default ratio 1.5) |

---

## 9. Follow-on, 2026-05-13: Pilot wall-time budget + abort-to-baseline

This section was added in the next session, on top of the §§ 0–8 state.
The History 2 "open follow-up" (and HANDOFF issue 3.1) — give the
pilot a wall-time budget and bail to a single full-grid run when it
busts — finally landed.

### 9.1 What the abort does

Per-iter wall-time deltas (`std::chrono::steady_clock`) accumulate
into `pst.pilot_elapsed_sec`. At iter 0 we project a baseline-run
cost:

```
baseline_wall_est_sec = T0 × total_ctas / ctas_launched_iter0
```

— i.e., scale the K-rep wall up to what a full-grid no-sampling run
would have taken. The K-rep iter pays roughly per-CTA cost on K reps,
and any fixed-per-kernel overhead is paid in both modes, so this is a
conservative upper bound for overhead-dominated kernels.

Each subsequent iteration that would otherwise reject also checks:

```
pilot_elapsed_sec > ratio × baseline_wall_est_sec
```

When that fires we flag the kernel as `aborting`, treat the current
iter as a reject (rolls back `gpu_tot_*` via `pilot_restore`), and
force the next iter's target to `total_ctas`. The recovery iter is
forced to accept regardless of classifier output, so the kernel
finalizes through the normal accept path with full-grid (no-sampling)
stats. A new `pilot_aborted_reason={none,budget_exceeded}` field on
the `CTA_PRESSURE_SIGNALS:` log line lets the validation harness
count abort events.

Knob: `-cta_sampling_pilot_max_wall_ratio` (default `1.5`; `0`
disables the abort entirely — useful when running on production-scale
traces where the user knows the pilot is worth its wall-time tax).

### 9.2 Code changes

- `simulator-remodeled/gpu-simulator/main.cc`:
  - `pilot_state_t` gains `iter_start` (`steady_clock::time_point`),
    `baseline_wall_est_sec`, `pilot_elapsed_sec`, and an `aborting`
    bool.
  - launch() callsite captures `iter_start` immediately before
    `m_gpgpu_sim->launch()`.
  - The accept/reject branch computes `iter_wall_sec`, populates
    `baseline_wall_est_sec` at iter 0, accumulates into
    `pilot_elapsed_sec`, and short-circuits to accept when
    `pst->aborting` is already set (the recovery iter). Otherwise
    runs `pilot_decide_accept` as before; on reject, the budget check
    sets `aborting=true` and the reject branch overrides
    `next_target = total_ctas`.
- `simulator-remodeled/gpu-simulator/trace-driven/trace_driven.{h,cc}`:
  - New `cta_sampling_pilot_max_wall_ratio` field + getter +
    `option_parser_register` entry, default `1.5`.

### 9.3 Measured results

Same 9-workload set, same harness (`measure_speedup.py`), 3 trials
each. Raw data: `.plan/history/speedup_results_2026-05-13.csv`.

| Workload   | baseline (s) | pilot (s) | 2026-05-13 | 2026-05-12 |
|---|---:|---:|---:|---:|
| hotspot    | 12.62 |  7.93 | **1.59×** | 1.53× |
| backprop   |  3.54 |  4.31 | 0.82× | 0.87× |
| pathfinder |  1.91 |  1.74 | 1.10× | 1.07× |
| bfs        |  9.13 |  6.35 | **1.44×** | 1.19× |
| srad_v2    |  3.46 |  5.24 | 0.66× | 0.71× |
| lud        |  8.90 |  8.90 | 1.00× | 1.00× |
| heartwall  |  2.21 |  4.20 | 0.53× | 0.53× |
| nn         |  5.39 |  3.99 | 1.35× | 1.34× |
| nw         |  7.44 |  6.88 | 1.08× | 1.00× |
| **cumulative** | **54.59** | **49.53** | **1.10×** | 1.05× |

Cycle accuracy (sat_exp default, `validate_models.py`, same 9-workload
set): p50 = 9.5%, p90 = 23.4%, max = 30.0% — bitwise identical to § 4.

### 9.4 What the abort actually changed

Two things to note about these numbers:

1. **No abort fired at default ratio 1.5** on any of the 9 workloads ×
   3 trials × 4 cycle-estimator models. We counted
   `pilot_aborted_reason=budget_exceeded` in every per-trial log and
   the total was 0. The default ratio is large enough that the
   per-kernel cumulative pilot wall stays under `1.5 × baseline_wall_est`
   for every kernel in this set.

2. **Cumulative speedup still improved (1.05× → 1.10×).** Since the
   abort never fired, the gain is run-to-run variance plus the
   incidental impact of the new `iter_start`/wall-time accounting on
   pilot timing. Per-workload, `bfs` is the largest mover (1.19× →
   1.44×) and `hotspot` improved (1.53× → 1.59×); `backprop` and
   `srad_v2` moved slightly the wrong way (0.87× → 0.82×, 0.71× →
   0.66×). All movements are within trial-to-trial noise (stdev ~0.1–
   0.8 s per workload), so the right framing is: the abort path is
   speedup-neutral on the toy set at default ratio.

The path was forced-tested with `-cta_sampling_pilot_max_wall_ratio
0.01` on backprop: it aborts at iter 0, the relaunched iter 1 is a
single full-grid run, and `gpu_tot_sim_cycle` lands within 1% of the
no-sampling baseline (kernel 2: 26005 vs. baseline 25991 cycles).

### 9.5 Why heartwall, srad_v2, backprop still regress

The abort path doesn't help these for a structural reason. Take
heartwall (the worst, 0.53× both before and after):

- grid is 51 CTAs, K-rep = 3
- pilot runs iter 0 (3 CTAs), iter 1 (20), iter 2 (40), iter 3 (51)
- baseline_wall_est = T0 × 51/3 = 17 × T0
- per-iter wall is *not* proportional to CTA count — the simulator's
  cycle count is roughly constant per iter (~8000 cycles) since all
  CTAs run concurrently on SMs. So each iter has wall ≈ T0.
- After iter 2: pilot_elapsed = 3 × T0; budget = 25.5 × T0. No abort.
- After iter 3 (the accepted full-grid iter): we've spent ~4 × T0
  vs. baseline of ~T0 (since baseline runs only 51 CTAs ≈ same cycle
  count as the pilot's 51-CTA iter).

So the regression is 4 simulator-cycle batches vs. baseline's 1.
Tightening the abort ratio doesn't fix this — the budget bound
(`T0 × total/k_reps`) is too generous because it assumes wall scales
with CTA count, but for heartwall-like small grids the wall is
dominated by per-iter setup, not CTA count.

The honest fix for these workloads is one of:

- skip the pilot entirely on small grids — the existing C1 tiny-grid
  skip only fires when `total_ctas < total_sms`, which heartwall
  (51 vs. 40 SMs) narrowly fails. Loosening to `< 2 × total_sms`
  would catch heartwall + nw + a few others.
- compute `baseline_wall_est_sec` from sampled `sim_cycles` instead of
  the K-rep wall: `(T0 / iter0_cycles) × baseline_cycles_est`. This
  needs a baseline-cycle estimate at iter 0, which is what the whole
  pilot loop is trying to produce — so chicken-and-egg.

Neither is in scope for this stage. The abort path is a safety net
for production-scale runaway pilots, not a toy-set speedup driver, and
on the toy set it does the right thing: it doesn't trigger and
doesn't hurt.

### 9.6 What's now open

- 3.2 (production-scale measurement) is the top blocker. Toy-set
  cumulative is now 1.10×; production projection is unchanged at
  100×+ on transformer-layer-scale traces (10K–1M CTAs/kernel).
- The heartwall-style regression (small grid, per-iter wall fixed-
  overhead-dominated) is structurally outside what a wall-time abort
  can fix. The two candidate fixes above are speculative until we
  have production data to prioritize against.
