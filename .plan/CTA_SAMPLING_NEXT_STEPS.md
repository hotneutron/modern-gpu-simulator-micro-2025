# CTA Sampling — Next Steps Plan

Addresses three of the four limitations called out in `.plan/PROGRESS.md`:

- **Limitation #1** — crude arithmetic-intensity proxy (Phase A)
- **Limitation #4** — no memory-stall fraction (Phase A)
- **Limitation #2** — whole-kernel cycle estimation deferred (Phase B)

Limitation #3 (`window_size > 1` / concurrent kernels) is **out of scope** for this plan.

The two phases are independent — A can ship without B, and vice versa. Recommended order: A first (low risk, high leverage; directly fixes the misclassification cases that drove the limitation in the first place), B second.

---

## Phase A — refine classifier inputs (limitations #1 + #4)

### Goal

Replace the over-counting `kernel_ai = gpu_sim_insn / dram_bytes` proxy with a real arithmetic-intensity estimate, and add a memory-stall-fraction signal so the classifier sees the SM's actual stall behavior, not just DRAM-side proxies.

The misclassification today: memory-heavy kernels show high `gpu_sim_insn` (because every load/store/index-arith op counts as a "FLOP" in the proxy), `kernel_ai` looks high, the classifier returns COMPUTE, the pilot loop accepts on iter 0 and never expands. Fixing the AI proxy makes those kernels classify MEMORY/MIXED, which routes them through the pilot expansion as designed.

### A1. Plumb per-class instruction counters into `pressure_signals_t`

The simulator already aggregates per-SM counters into `m_gpu_per_sm_stats`. Reuse these, exposed via the existing `gather_gpu_per_sm_single_stat()` pattern that `compute_kernel_pressure_signals` already uses for `gpu_sim_insn`.

Counters to plumb (all live in `shader.h:2350-2410` / `shader_core_stats_pod`):

| Counter | Role in new AI formula |
|---|---|
| `m_num_FPdecoded_insn` | compute-FP |
| `m_num_INTdecoded_insn` | compute-int |
| `m_num_dp_acesses` | weighted heavy compute (×W_dp) |
| `m_num_tensor_core_acesses` | weighted heavy compute (×W_tc) |
| `m_num_sfu_acesses` | weighted heavy compute (×W_sfu) |
| `gpgpu_n_load_insn` | memory ops (numerator excludes them; tracked for AI denominator sanity) |
| `gpgpu_n_store_insn` | memory ops |

**Code touchpoints:**

- `gpu-sim.h:755` — extend `pressure_signals_t` with `compute_ops`, `mem_ops`, plus the raw per-class fields for logging.
- `gpu-sim.h:780` — extend `pressure_snapshot_t` with launch-time baselines for these new counters so per-kernel deltas work.
- `gpu-sim.cc:2040` — extend `snapshot_pressure_signals_kernel_start()` to capture the new baselines.
- `gpu-sim.cc:2066` — in `compute_kernel_pressure_signals()`:
  - `gather_gpu_per_sm_single_stat()` for each new counter.
  - Subtract the launch snapshot for each.
  - Compute `compute_ops = FP_decoded + INT_decoded + W_dp*dp + W_tc*tc + W_sfu*sfu`.
  - Replace `kernel_ai = sim_insns / dram_bytes` with `kernel_ai = compute_ops / dram_bytes`.
- New knobs in `trace_driven.cc` (alongside the existing `-cta_sampling_*` knobs):
  - `-cta_sampling_ai_w_dp` (default 2.0)
  - `-cta_sampling_ai_w_tc` (default 8.0)
  - `-cta_sampling_ai_w_sfu` (default 4.0)

The defaults are first-order guesses based on relative ops/cycle — see open decision **D2** for the calibration question.

### A2. Memory-stall fraction

The issue-stage stall counters already exist (`shader.h:2465-2476`) and are aggregated into `m_gpu_per_sm_stats` (see `shader.cc:1204-1207` for the read pattern). Plumbing pattern is identical to A1.

Add to `pressure_signals_t`:

```cpp
double mem_stall_frac;        // l1c-wait / issue_eval
double total_stall_frac;      // (l1c + stall_count + barriers + ...) / issue_eval
```

Calculation in `compute_kernel_pressure_signals`:

```cpp
double mem_stall_frac = (issue_eval > 0)
    ? (double)stall_l1c / (double)issue_eval : 0.0;
```

The launch snapshot needs the same baseline-subtract pattern; these counters are cumulative.

### A3. Update `classify_kernel`

In `main.cc:88-98`, add memory-stall as a third pressure proxy:

```cpp
bool memory_pressure_high =
    (s.achieved_bw_ratio        >= tc.get_cta_sampling_pressure_bw())     ||
    (s.dram_queue_occupancy_avg >= tc.get_cta_sampling_pressure_queue())  ||
    (s.mem_stall_frac           >= tc.get_cta_sampling_pressure_mstall());
```

New knob `-cta_sampling_pressure_mstall` (default `0.4` — pending decision **D5**).

### A4. Validation

Extend `simulator-remodeled/util/cta_sampling/validate.py:43`:

- Add a new mode `pilot+refined` that exercises the new classifier (same flags as `pilot`, just runs against the refined build).
- Extend `STAT_PATTERNS` to capture the new pressure-signal fields from the `CTA_PRESSURE_SIGNALS:` log line so the table shows class transitions before/after the refinement.
- Add 2-3 misclassification-prone kernels (see decision **D4** for the list).

Acceptance:

- Memory-heavy kernels that previously classified COMPUTE on iter 0 now classify MEMORY/MIXED and the pilot loop expands them.
- Accepted-iteration `insn_err%` and `cycle_err%` for those kernels improve materially.
- Existing 3-kernel results (hotspot, backprop, pathfinder) do not regress.

### A5. Per-commit shape

Each commit independently buildable, layered:

| # | Commit | Files |
|---|---|---|
| A1a | Per-class instruction counters in pressure signals | `gpu-sim.h`, `gpu-sim.cc` |
| A1b | New AI weight knobs + refined `kernel_ai` formula | `trace_driven.h`, `trace_driven.cc`, `gpu-sim.cc` |
| A2  | Memory-stall fraction in pressure signals | `gpu-sim.h`, `gpu-sim.cc`, `trace_driven.{h,cc}` |
| A3  | Classifier consumes `mem_stall_frac` | `main.cc`, `trace_driven.{h,cc}` |
| A4  | Validation harness extended | `util/cta_sampling/validate.py` |

---

## Phase B — whole-kernel cycle estimation (limitation #2)

### Goal

Report a credible whole-kernel cycle number, not just the sampled-wave wall-clock, **without** breaking the existing semantics. Current `gpu_tot_sim_cycle` is the wall-clock of the sampled wave (deliberate — see `gpu-sim.cc:2170`); after Phase B it remains so, and a *new* stat carries the model-projected whole-kernel cycle count.

### B1. Wave-count derivation

```
N_per_wave_full   = total_sms × max_cta_per_core(kernel)
N_per_wave_sample = active_sms_in_sample × max_cta_per_core(kernel)
                    where active_sms_in_sample = min(target_sim_ctas, total_sms)
N_waves_full      = ceil(total_ctas / N_per_wave_full)
N_waves_sample    = ceil(sampled_ctas / N_per_wave_sample)
```

`max_cta_per_core(kernel)` is already exposed via `gpgpu_sim::max_cta_per_core()` at `gpu-sim.cc:1922`.

### B2. Two estimation modes (auto-selected by kernel class)

**Per-CTA mode** — good for COMPUTE-classified kernels where per-SM throughput is largely independent of grid size:

```
cycles_per_cta_avg = sampled_cycles / sampled_ctas
total_cycles_est   = cycles_per_cta_avg × ceil(total_ctas / total_sms)
```

**Steady-state mode** — good for MEMORY/MIXED kernels *after* pilot expansion has saturated DRAM (i.e. the sampled wave is representative of a real wave):

```
total_cycles_est = sampled_cycles × N_waves_full / N_waves_sample
```

Both are first-order. They are auto-selected based on `kernel_class`:

- COMPUTE → per-CTA
- MEMORY, MIXED → steady-state (only valid post-pilot-expansion; if pilot disabled, fall back to per-CTA with a warning)

### B3. Reporting

In `print_stats()` at `gpu-sim.cc:2391-2400`, add new lines:

```
gpu_tot_sim_cycle_estimated = ...
gpu_tot_sim_cycle_estimation_mode = per_cta | steady_state | none
gpu_tot_ipc_estimated = gpu_tot_sim_insn / gpu_tot_sim_cycle_estimated
```

Existing `gpu_tot_sim_cycle` and `gpu_tot_ipc` lines unchanged. This is backward-compatible — see open decision **D3** if we want a more disruptive UX instead.

### B4. Plumbing

The estimator needs:

- `kernel_class` — currently computed in `main.cc:471`. Plumb into the gpgpu_sim instance (e.g. `set_last_kernel_class()` called from main.cc) so `print_stats()` can read it.
- `target_sim_ctas` and `sampled_ctas` — `target_sim_ctas` is the pilot's accepted target; `sampled_ctas` is `m_total_cta_launched` at finalization.
- `total_ctas` (full grid) — `kernel_trace_t::grid_dim_*` product, plumbed in via the same setter.

A small struct, e.g. `last_kernel_wave_info_t { kernel_class kc; unsigned target_sim_ctas; unsigned sampled_ctas; unsigned total_ctas; }`, set from `main.cc` after pilot acceptance and read by `print_stats()`.

### B5. Validation

In `validate.py`, capture both `gpu_tot_sim_cycle` (raw) and `gpu_tot_sim_cycle_estimated` (new). Compute `cycle_err%` against baseline using the *estimated* number. Targets:

- p50 `cycle_err%` < 15%
- p90 `cycle_err%` < 25%

Across the 5+ kernels in the extended workload set. The revised plan's stretch goal of <8% / <15% is unrealistic for first-order wave models; reset expectations explicitly.

### B6. Per-commit shape

| # | Commit | Files |
|---|---|---|
| B1 | Plumb wave-info struct (kernel class + sample sizes) | `gpu-sim.h`, `gpu-sim.cc`, `main.cc` |
| B2 | Multi-wave estimator + new stat lines | `gpu-sim.cc` |
| B3 | Validation harness reports estimated cycles | `util/cta_sampling/validate.py` |

---

## Open decisions (need answers before implementation)

| # | Decision | Default if no answer |
|---|---|---|
| D1 | Phase A only, A+B, or A+B in parallel? | A first, then B |
| D2 | AI weights (`W_dp`, `W_tc`, `W_sfu`) — calibrate empirically (fit to a known-FLOPs kernel) or ship guess defaults? | Ship guesses (2/8/4); calibration is a follow-up |
| D3 | For Phase B, keep `gpu_tot_sim_cycle` raw and add `*_estimated` (backward-compatible) or replace `gpu_tot_sim_cycle` with the estimate (cleaner)? | Keep raw + add `_estimated` |
| D4 | Which extra workloads for the validation harness? Need at least: a GEMM-like, a sparse/graph kernel, and a larger memory-bound kernel. Pull from `exampleTraces/rodinia2{Turing,Ampere,Blackwell}.tar.gz` — which? | Pick from rodinia2-Turing: gaussian (GEMM-like dense), bfs (graph/irregular), srad (memory-bound stencil) |
| D5 | Memory-stall threshold knob name + default — `-cta_sampling_pressure_mstall = 0.4`? | Use this |

---

## Acceptance criteria (rolled up)

- **Phase A:** misclassified kernels (memory-heavy ones currently routed COMPUTE) now route through pilot expansion; existing 3-kernel `insn_err%` does not regress; new `mem_stall_frac` and per-class instruction counts visible in `CTA_PRESSURE_SIGNALS:` log line.
- **Phase B:** `gpu_tot_sim_cycle_estimated` printed; estimated cycle p50 error < 15%, p90 < 25% across the extended validation set; raw `gpu_tot_sim_cycle` semantics unchanged.

## Out of scope (explicit)

- Concurrent-kernel pilot loop (limitation #3).
- Per-architecture threshold calibration (depends on D2 outcome; could become a follow-up plan).
- Replacing the K-rep coordinate heuristic with clustering (separate, larger redesign).
