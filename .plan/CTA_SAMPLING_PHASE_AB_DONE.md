# CTA sampling — Phase A + Phase B status

This is the post-execution writeup for `.plan/CTA_SAMPLING_NEXT_STEPS.md`.
Both phases shipped to the `cta-sampling` branch as 8 layered commits
matching the per-commit shape in A5/B6 of that plan.

## Commits (8, all on `cta-sampling`)

| # | Hash      | Subject |
|---|-----------|---------|
| A1a | `e8f9895` | per-class instruction counters in pressure signals |
| A1b | `18581f9` | refined `kernel_ai` with AI weight knobs |
| A2  | `852d765` | memory-stall fraction in pressure signals |
| A3  | `ad055b0` | classifier consumes `mem_stall_frac` |
| A4  | `c8a9175` | validation harness consumes refined classifier signals |
| B1  | `f27982c` | plumb wave-info struct for whole-kernel cycle estimation |
| B2  | `9aa26fa` | whole-kernel cycle estimator + new stat lines |
| B3  | `c99bbe7` | validation harness reports estimated cycles |

Each commit is independently buildable; A1a–A4 and B1–B3 each cleanly
recompile and pass the hotspot smoke test in isolation.

## What changed (rolled up)

### Phase A — refined classifier inputs

- **Per-class instruction counters** in `pressure_signals_t`/
  `pressure_snapshot_t`: `n_fp_decoded`, `n_int_decoded`, `n_dp_acc`,
  `n_tc_acc`, `n_sfu_acc`, `n_load_insn`, `n_store_insn`. The first
  five sum across SMs from the legacy `shader_core_stats` POD arrays;
  load/store come from each SM's `m_sm_stats` via a new
  `read_sm_stat_value` virtual on `shader_core_ctx_wrapper` and a
  `sum_sm_stat_value` helper on the cluster + gpgpu_sim. This avoids
  the gather-side double-counting that affects stats registered with
  `is_erase_after_gather_in_sm=false`.
- **Refined kernel_ai**:
  ```
  compute_ops = FP_decoded + INT_decoded
              + W_dp  * m_num_dp_acesses
              + W_tc  * m_num_tensor_core_acesses
              + W_sfu * m_num_sfu_acesses
  kernel_ai   = compute_ops / dram_bytes
  ```
  Knob-tunable: `-cta_sampling_ai_w_dp` (2.0), `-cta_sampling_ai_w_tc`
  (8.0), `-cta_sampling_ai_w_sfu` (4.0). Replaces the
  `gpu_sim_insn / dram_bytes` proxy that included memory and
  control-flow ops in the FLOP count.
- **Memory-stall fraction** sourced from `total_num_cycles_issue_stage_
  stall_at_least_one_warp_waiting_l1c`. The counter previously existed
  on `shader_core_stats_pod` but its increment in `subcore.cc` was
  commented out — re-enabled `is_any_waiting_l1c` and added the
  increment guarded on a non-issuing cycle. Registered the counter in
  `m_gpu_per_sm_stats`. New fields on `pressure_signals_t`:
  `issue_eval_cycles`, `stall_l1c_cycles`, `stall_total_cycles`,
  `mem_stall_frac` (= stall_l1c / issue_eval), `total_stall_frac`.
- **Three-input classifier**: memory pressure now fires on any of
  `achieved_bw_ratio >= pressure_bw`, `dram_queue_occupancy_avg >=
  pressure_queue`, OR `mem_stall_frac >= pressure_mstall`. New knob
  `-cta_sampling_pressure_mstall` (default 0.4).
- **Validation harness**: added a `pilot+refined` mode (pilot mechanics
  with explicit AI weights and the mstall threshold), three new
  workloads (`bfs`, `srad_v2`, `lud` — kernels the original AI proxy
  mis-routed as COMPUTE), captured the new pressure-signal fields from
  the per-kernel log line, and printed a per-mode class/AI/stall table
  so classification flips are visible alongside the cycle/insn
  comparison.

### Phase B — whole-kernel cycle estimation

- **Wave-info plumbing**: `last_kernel_wave_info_t` carries the
  classifier's pilot-final state from `main.cc` to `print_stats()`
  via a `set_last_kernel_wave_info` setter on `gpgpu_sim`.
- **Two estimation modes**, auto-selected by class:
  - `per_cta` for COMPUTE: `cyc_per_cta = sampled_cycles /
    rounds_per_sm_sampled`, then projected by `ceil(total_ctas /
    total_sms)`. (The plan's literal formula `sampled_cycles /
    sampled_ctas` is wrong for parallel CTAs — corrected to use
    rounds-per-SM, which matches the actual sampled-wave wall-clock.)
  - `steady_state` for MEMORY/MIXED: `sampled_cycles * N_waves_full /
    N_waves_sample` with `per_wave_full = total_sms *
    max_cta_per_core`.
- **New cumulative stats** in `print_stats`, distinct from the existing
  `gpu_tot_sim_cycle` (which deliberately stays as the sampled-wave
  wall-clock):
  ```
  gpu_tot_sim_cycle_estimated         = ...
  gpu_tot_sim_cycle_estimation_mode   = per_cta | steady_state | none
  gpu_tot_ipc_estimated               = ...
  ```
- **Validation harness**: new `STAT_PATTERNS` entries plus an
  `ESTIMATION_MODE_PAT`. The comparison table now has columns for
  `cycles_est` and `cycles_raw` and reports both `cycle_err%` (against
  the baseline's raw cycles using the projection) and
  `cycle_err_raw%` (the un-projected sampled-wave error).

## Acceptance criteria

- **Phase A** (misclassification): met. With the refined classifier,
  bfs has `kernel_ai = 0.19`, `mem_stall_frac = 0.36`, classifies as
  MEMORY, and the pilot loop expands it to iter 1 with
  `pilot_accepted=1`. Hotspot/backprop/pathfinder retain their
  pre-existing behavior; `insn_err%` does not regress.
- **Phase B** (cycle projection): structurally complete. Hotspot
  estimates 159k cycles vs baseline 111k (+43% over) — the per-CTA
  model assumes a perfectly compute-bound full-grid wall-clock that
  scales linearly with rounds-per-SM, which over-counts when partial
  waves run faster than full ones. Pathfinder is within 3%. The plan
  reset acceptance to p50 < 15% / p90 < 25% across the extended
  validation set; running the full sweep is the first follow-up below.

## Files changed

| Path | Touched in commits |
|---|---|
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h` | A1a, A1b, A2, B1, B2 |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc` | A1a, A1b, A2, B2 |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h` | A1a |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader_core_wrapper.h` | A1a |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h` | A1a |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc` | A2 |
| `simulator-remodeled/gpu-simulator/trace-driven/trace_driven.h` | A1b, A3 |
| `simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc` | A1b, A3 |
| `simulator-remodeled/gpu-simulator/main.cc` | A1b, A3, B1 |
| `simulator-remodeled/util/cta_sampling/validate.py` | A4, B3 |

## Follow-ups (not done — listed by priority)

1. **Run the full validation sweep** across hotspot, backprop,
   pathfinder, bfs, srad_v2, lud and check whether
   `gpu_tot_sim_cycle_estimated` meets p50 < 15% / p90 < 25%. Hotspot
   already shows +43% (per-CTA model overshoots), so the per-CTA
   formula likely needs an empirical tweak (or a class-aware blend
   between per-CTA and steady-state).
2. **Calibrate AI weights** (open decision **D2**): fit `W_dp`,
   `W_tc`, `W_sfu` against a known-FLOPs kernel (a tiled GEMM is
   ideal) instead of shipping the 2/8/4 guesses.
3. **Concurrent-kernel pilot** (limitation #3, explicitly out of
   scope here): the pilot loop assumes `window_size == 1`. Extending
   it to a window of running kernels needs careful pressure-signal
   composition since kernel deltas would overlap.
4. **Pilot-rejected iteration cleanup**: `pilot_restore` undoes
   `gpu_tot_sim_*` but does not roll back the per-SM aggregates
   (`m_gpu_per_sm_stats`, the legacy `shader_core_stats` POD arrays,
   the per-SM `m_sm_stats` for stall counters). For long pilot
   sweeps with many rejected iterations this can pollute the cycle
   estimate's denominator. Pre-existing behavior, but worth a
   targeted fix once cycle accuracy is being chased.
5. **Drop the K-rep coordinate heuristic** in favor of clustering
   (the larger redesign noted in the plan's "out of scope" section).
   Only worth doing once Phase B's cycle-error targets are firm.

## Build / run notes

The `make clean && make` flow is sensitive to the user's NFS quota:
`make clean` deletes `build/release/accelsim_version.h` and
`gpgpu-sim/build/.../detailed_version`, both of which the Makefile
re-creates with `$(shell echo '...' > $file)`. If that write fails
silently (quota exceeded), the next compile fails with
`'g_gpgpusim_build_string' was not declared` and `'g_accelsim_version'
was not declared`. The fix is just to free disk space, not anything
in this branch's source.

The build environment is documented in
`~/.claude/projects/-home-afa55-Projects-modern-gpu-simulator-micro-2025/memory/build_environment.md`.
