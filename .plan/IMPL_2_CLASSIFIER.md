# Implementation Plan: [2] Hotspot Analyzer + Classifier

**File:** `tools/kernel_classifier.py`
**Depends on:** `kernel_report.csv` from [1], GPU config JSON
**Produces:** `kernel_classifier_report.csv`

---

## GPU Config JSON Format

```json
{
  "name": "SM86_RTX3080",
  "num_sms": 68,
  "peak_flops_fp32_tflops": 29.7,
  "peak_dram_bw_gbps": 760.0,
  "peak_l2_bw_gbps": 3584.0,
  "warp_size": 32,
  "max_warps_per_sm": 48,
  "max_ctas_per_sm": 16,

  "classifier": {
    "T_low": 0.9,
    "T_high": 1.3,
    "W_tc": 8.0,
    "W_sfu": 4.0,
    "W_fp64": 2.0,
    "mem_stall_threshold_memory": 0.5,
    "mem_stall_threshold_compute": 0.2,
    "achieved_bw_threshold": 0.6
  }
}
```

Pre-built configs in `tools/configs/`: `sm75_turing.json`, `sm86_rtx3080.json`,
`sm80_a100.json`, `sm90_h100.json`, `sm100_b200.json`.

---

## Classification Logic

```python
def classify_kernel(r: KernelRecord, cfg: GPUConfig) -> KernelClass:
    """
    Three-way roofline classification.
    All signals from kernel_report.csv (kernel-level, not per-CTA).
    """
    ridge_point = cfg.peak_flops / cfg.peak_dram_bw   # FLOP/byte

    effective_ops = (r.fp32_insns
                     + r.int_insns
                     + cfg.W_fp64 * r.fp64_insns
                     + cfg.W_tc  * r.tc_insns
                     + cfg.W_sfu * r.sfu_insns)
    dram_bytes = r.dram_bytes_read + r.dram_bytes_write

    if dram_bytes == 0:
        kernel_ai = float('inf')   # purely compute, no DRAM traffic
    else:
        kernel_ai = effective_ops / dram_bytes

    ridge_ratio = kernel_ai / ridge_point

    # Pressure signals (biased low — see circularity note in CTA_SAMPLING.md)
    high_mem_pressure = (
        r.mem_stall_pct > cfg.mem_stall_threshold_memory * 100
    )
    low_mem_pressure  = r.mem_stall_pct < cfg.mem_stall_threshold_compute * 100

    if ridge_ratio >= cfg.T_high and low_mem_pressure:
        return KernelClass.COMPUTE
    elif ridge_ratio <= cfg.T_low or high_mem_pressure:
        return KernelClass.MEMORY
    else:
        return KernelClass.MIXED
```

---

## sim_ctas Recommendation

```python
def recommend_sim_ctas(r: KernelRecord, klass: KernelClass,
                       k_reps: int, cfg: GPUConfig) -> int:
    total_ctas = r.grid_x * r.grid_y * r.grid_z

    if klass == KernelClass.COMPUTE:
        return min(total_ctas, max(k_reps, cfg.num_sms // 2))

    # Estimate per-SM memory BW demand from instruction mix
    mem_fraction = r.mem_insns / max(r.fp32_insns + r.int_insns + r.mem_insns, 1)
    bytes_per_mem_insn = 32   # conservative: assume 32B avg transaction
    sm_issue_rate = cfg.num_sms * 4  # ~4 mem insns/cycle/SM (rough)
    per_sm_bw_est = mem_fraction * bytes_per_mem_insn * sm_issue_rate  # bytes/cycle

    peak_bw_bytes_per_cycle = (cfg.peak_dram_bw_gbps * 1e9) / (cfg.peak_flops_fp32_tflops * 1e12 / 64)
    n_sat_est = math.ceil(peak_bw_bytes_per_cycle / max(per_sm_bw_est, 1e-9))

    if klass == KernelClass.MEMORY:
        sim_ctas = min(total_ctas, max(k_reps, n_sat_est))
    else:  # MIXED
        sim_ctas = min(total_ctas, max(k_reps, math.ceil(1.5 * n_sat_est)))

    return sim_ctas
```

---

## CTA Heterogeneity Flag

Determines whether Pass B (per-CTA profiling) and [3] clustering are needed,
or whether the coordinate heuristic suffices:

```python
def needs_clustering(r: KernelRecord) -> bool:
    """
    Heuristic: irregular kernels need clustering.
    Regular kernels (GEMM, stencil, conv) have predictable CTA behavior.
    """
    # Flag irregular if instruction mix suggests data-dependent behavior
    branch_rate = r.branch_insns / max(r.fp32_insns + r.int_insns, 1)
    if branch_rate > 0.05:       # >5% branch instructions → irregular
        return True
    # Flag if grid is 1D and large (heartwall-style collapse risk)
    total_ctas = r.grid_x * r.grid_y * r.grid_z
    if r.grid_y == 1 and r.grid_z == 1 and total_ctas > 64:
        return True
    # Otherwise coordinate heuristic is sufficient
    return False
```

This flag is refined by the actual silhouette score from [3] once Pass B is run.

---

## Output Format

**`kernel_classifier_report.csv`:**
```
kernel_id,kernel_name,total_ctas,hotspot_score,class,ridge_ratio,
mem_stall_pct,kernel_ai,ridge_point,recommended_sim_ctas,
estimated_speedup,needs_clustering,notes
```

**`hotspot_priority_report.txt`** — human-readable simulation priority list:
```
Priority  KernelID  Name              Class    RidgeRatio  SimCTAs  Speedup  Cluster?
       1  kernel_3  volta_sgemm...    compute  4.21        9        114x     no
       2  kernel_7  calculate_temp    memory   0.31        24       2.7x     no
```

---

## CLI Interface

```
python tools/kernel_classifier.py \
  --kernel-report ./profiler_output/kernel_report.csv \
  --gpu-config tools/configs/sm86_rtx3080.json \
  --output-dir ./classifier_output/ \
  [--k-reps 9]             # default coordinate heuristic K
  [--top-k 10]             # how many kernels to include in priority list
```

---

## Notes

- `ridge_ratio` from kernel-level metrics is less biased than from a K-rep sim run
  (uses hardware counters on full-occupancy execution). This makes the classifier
  more reliable here than in the simulator's internal classifier.
- `needs_clustering` is a soft recommendation; always overridden if the user
  provides `representatives.json` from [3].
- Speedup estimate: `total_ctas / recommended_sim_ctas`. Conservative — actual
  speedup depends on pilot loop convergence.
