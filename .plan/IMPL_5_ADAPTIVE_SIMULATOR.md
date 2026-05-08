# Implementation Plan: [5] Adaptive CTA Simulator

**Files:** `simulator-remodeled/gpu-simulator/main.cc`,
           `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc`
**Depends on:** `cta_weights.json` sidecar from [4] (or coordinate heuristic fallback)
**Status:** Partially implemented on `cta-sampling` branch. See `CTA_SAMPLING_STATUS.md`.

---

## Remaining Work (beyond cta-sampling branch)

### A. Load external weights from `cta_weights.json`

Currently the simulator computes `weight = total_ctas / K` uniformly. With cluster-based
representatives, each CTA has a different weight. The `cta_weights.json` sidecar from [4]
provides per-CTA weights.

**Change in `main.cc`:**

```cpp
// After loading kernel_trace_info, check for sidecar
void load_cta_weights_sidecar(kernel_trace_t *kti, const std::string &trace_dir) {
    std::string sidecar = trace_dir + "/cta_weights.json";
    if (!std::filesystem::exists(sidecar)) return;

    // Parse JSON: {"kernel_3": [{"x":0,"y":0,"z":0,"weight":0.25}, ...]}
    // Match by kernel name from kti->kernel_name
    // Populate kti->sampled_ctas with {x,y,z} and per-CTA weights
    // Store per-CTA weights in kti->sampled_cta_weights vector
}
```

**New field in `kernel_trace_t` (`trace_parser.h`):**
```cpp
std::vector<float> sampled_cta_weights;  // parallel to sampled_ctas; 1.0/K if uniform
```

**Stat scaling in `gpu-sim.cc`:** replace `m_cta_sampling_weight` (single float) with
per-representative weights accumulated per-CTA as they complete.

### B. Fix failure mode 1: nn-style (high-projection-ratio, log-fit under-extrapolation)

**Problem:** Force-expand fires (`full_ctas_per_sm > 4 × sampled_ctas_per_sm`) but
`pilot_max_doublings` cap prevents enough expansion. Log-fit extrapolates from 2 CTAs/SM
to 23 CTAs/SM and underestimates throughput growth.

**Fix in `main.cc`:**
```cpp
// When force-expand fires, adaptively raise the doublings cap
if (force_expand_triggered) {
    int projection_ratio = full_ctas_per_sm / sampled_ctas_per_sm;
    // Allow log2(projection_ratio) + 2 additional doublings
    pilot_max_doublings = std::max(pilot_max_doublings,
                                   (int)std::ceil(std::log2(projection_ratio)) + 2);
}
```

### C. Fix failure mode 2: heartwall-style (K-rep collapse on 1D grids)

**Problem:** `compute_sampled_ctas()` for a 1D grid (gz=gy=1) collapses to only
2 unique CTAs (first and last), which may be unrepresentative.

**Fix in `main.cc`:**
```cpp
static std::vector<std::tuple<unsigned,unsigned,unsigned>>
compute_sampled_ctas(unsigned gx, unsigned gy, unsigned gz) {
    // ... existing corner/midpoint logic ...

    // NEW: for 1D grids, add more interior samples
    if (gy == 1 && gz == 1 && gx > 8) {
        // Add quartile samples: 25%, 50%, 75%
        add(gx / 4,     0, 0);
        add(gx / 2,     0, 0);
        add(3 * gx / 4, 0, 0);
    }
    return ...;
}
```

### D. Fix failure mode 3: nw-style (pilot overhead on tiny kernels)

**Problem:** Total grid < 2× num_SMs. Pilot loop overhead exceeds any speedup.

**Fix in `main.cc` in `create_kernel_info()`:**
```cpp
// Short-circuit: if total_ctas <= num_sms, simulate all CTAs directly
if (total_ctas <= (unsigned)config->get_num_shader()) {
    // No sampling: weight = 1.0, gridDim unchanged
    kernel_trace_info->cta_sampling_weight = 1.0f;
    std::cout << "CTA sampling: kernel " << kernel_trace_info->kernel_name
              << " grid too small (" << total_ctas << " <= " << config->get_num_shader()
              << " SMs), simulating all CTAs\n";
    return create_normal_kernel_info(...);
}
```

### E. Per-kernel accuracy report (validation mode)

When `-cta_sampling_validate 1` is set, emit a per-kernel JSON report:

```cpp
// gpu-sim.cc: in print_stats(), if validation mode enabled
void gpgpu_sim::print_cta_sampling_report(const char *kernel_name) {
    printf("cta_sampling_report: kernel=%s sim_ctas=%u total_ctas=%llu "
           "weight=%.4f class=%s sim_cycle=%llu estimated_cycle=%llu "
           "sim_insn=%llu scaled_insn=%llu ipc=%.4f estimated_ipc=%.4f\n",
           kernel_name, m_sim_ctas, m_total_ctas_full,
           m_cta_sampling_weight, m_kernel_class_str,
           gpu_sim_cycle, m_estimated_cycle,
           gpu_sim_insn, (unsigned long long)(gpu_sim_insn * m_cta_sampling_weight),
           (float)gpu_sim_insn / gpu_sim_cycle,
           m_estimated_ipc);
}
```

---

## Config Knobs (add to `trace_driven.cc` `reg_options()`)

```cpp
option_parser_register(opp, "-cta_sampling_mode", OPT_INT32, &cta_sampling_mode,
    "0=disabled, 1=coordinate-heuristic, 2=external-weights", "0");

option_parser_register(opp, "-cta_weights_file", OPT_CSTR, &cta_weights_file,
    "Path to cta_weights.json sidecar from selective tracer", "");

option_parser_register(opp, "-cta_sampling_validate", OPT_INT32,
    &cta_sampling_validate,
    "1=emit per-kernel sampling accuracy report", "0");

option_parser_register(opp, "-cta_pilot_max_doublings", OPT_INT32,
    &cta_pilot_max_doublings,
    "Max adaptive pilot doublings (default 3)", "3");
```

---

## Stat Scaling Summary (current + changes)

| Stat | Current | After Per-CTA Weights |
|---|---|---|
| `gpu_tot_sim_cycle` | unscaled | unscaled (no change) |
| `gpu_tot_sim_insn` | × uniform weight | × Σ(per-rep weight × rep_insns) |
| `gpu_tot_issued_cta` | × uniform weight | × Σ(per-rep weight × 1) = total_ctas |
| `gpu_tot_ipc` | scaled_insns / cycles | scaled_insns / cycles (no change) |
| `gpu_tot_sim_cycle_estimated` | throughput/log-fit | throughput/log-fit (no change) |

---

## Testing

```bash
# Test failure mode fixes
./validate.py --mode pilot --workloads heartwall,nn,nw \
  --baseline-dir /tmp/sim_full/ --sampled-dir /tmp/sim_sampled/

# Test external weights
./validate.py --mode pilot --weights ./representatives.json \
  --workloads hotspot,sgemm
```

Expected after fixes:
- heartwall: cycle_err < 15% (was -28.7%)
- nn: cycle_err < 20% (was +103.5%)
- nw: skip-sampling fires, cycle_err ≈ 0%
