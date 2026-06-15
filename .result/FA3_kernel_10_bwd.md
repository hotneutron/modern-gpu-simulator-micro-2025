# FA3 BWD Kernel 10 — Simulator vs. Real H100 Comparison

## Target Information

- **Workload**: `flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24`
- **Kernel**: trace_kernel_id = 10, `FlashAttnBwdSm90` (causal backward, bf16, tile 128×128×64)
- **Grid / Block**: (384, 1, 1) / (384, 1, 1)
- **Registers/Thread**: 168

### Source Files
- Measured (H100, NCU): `/home/jihyun/project/accorde/ncu_report/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.csv`
- Simulator output: `sim_run_12.8/.../H100_80GB-OnlyKernel10/...warmup_-82f2f3a37882.o304`
  - (OnlyKernel10 configuration, simulation wall time ~1 day 7 hours)

---

## Key Metric Comparison

| Metric | Real H100 (NCU) | Simulator | Sim / HW | Note |
|---|---|---|---|---|
| **Elapsed Cycles** | 132,901 | 376,735 | **2.83×** | Primary cycle comparison |
| SM Active Cycles | 118,089 | 376,735 (gpu_sim_cycle) | 3.19× | sim has no separate split |
| Duration | 92.22 µs | — | — | sim is cycle-based |
| Achieved Occupancy | 15.02 % | 14.49 % | 0.96× | **Very close** |
| Active Warps / SM | 9.61 | — | — | Consistent via occupancy |
| L2 Hit Rate | 82.26 % | ~98.7 % (miss 1.27%) | — | sim overestimates |
| L1/TEX Hit Rate | 80.87 % | ~62.2 % (miss 37.77%) | — | sim L1 access very low (9,912) |
| SM Count | 132 (SXM) | 132 (n_clusters) | 1.0× | Match |
| Core Clock | 1.43 GHz | 1.80 GHz | — | sim configured value |

### Accuracy (based on Elapsed Cycles)
- Ratio: 376,735 / 132,901 = **~2.83×**
- Error: **+183.5%** (simulator overestimates)
- Based on SM Active Cycles: ~**3.19×** (+219%)

---

## Simulator Detailed Statistics

| Item | Value |
|---|---|
| gpu_sim_cycle | 376,735 |
| gpu_sim_insn | 629,197,320 |
| gpu_occupancy | 14.49 % |
| L1D total accesses | 9,912 |
| L1D miss rate | 0.3777 |
| L2 total accesses | 6,907,840 |
| L2 miss rate | 0.0127 |
| total dram reads | 71,520 |
| total dram writes | 0 |
| DRAM bw_util (per-channel avg) | ~0.005 |
| DRAM efficiency (dram_eff) | ~0.27–0.29 |
| gpu_stall_dramfull | 33,899 |

> Memory access breakdown (Total_core_cache_stats_breakdown):
> - GLOBAL_ACC_R: L1 HIT 0 (all go through L2) / L2 HIT 5,802,795, MISS 17,356
> - LOCAL_ACC_R: L1 HIT 5,976, SECTOR_MISS 1,848
> - CONST_ACC_R: HIT 14,786, MISS 4,368, MSHR_HIT 2,916
> - gpgpu_n_mem_read_global = 5,916,672 / gpgpu_n_mem_write_global = 798,720

---

## NCU (Measured) Detailed Metrics

| Metric | Unit | Value |
|---|---|---|
| Elapsed Cycles | cycle | 132,901 |
| SM Active Cycles | cycle | 118,089 |
| Duration | µs | 92.22 |
| SM Frequency | GHz | 1.43 |
| DRAM Frequency | GHz | 2.62 |
| Compute (SM) Throughput | % | 48.45 |
| Memory Throughput | % | 56.58 |
| Memory Throughput | GB/s | 497.76 |
| DRAM Throughput | % | 14.85 |
| L1/TEX Hit Rate | % | 80.87 |
| L2 Hit Rate | % | 82.26 |
| Achieved Occupancy | % | 15.02 |
| Active Warps / SM | warp | 9.61 |
| Executed IPC (active) | inst/cycle | 1.28 |
| Issued IPC (active) | inst/cycle | 1.28 |
| Issue Slots Busy | % | 32.07 |
| Registers / Thread | reg | 168 |
| Waves / SM | — | 2.91 |

---

## Summary

1. **Cycle accuracy**: The simulator overestimates cycles by **~2.8–3.2×** relative to the real H100 (+183% on Elapsed basis). Significant room for accuracy improvement.
2. **Occupancy matches closely** (14.49% vs 15.02%) — resource modeling is relatively accurate.
3. **L2 hit rate**: sim estimates 98.7% vs measured 82.3% → the L2 model is optimistic.
4. **L1 hit rate**: the sim L1 access count (9,912) is very low, indicating that FA3's TMA / shared-memory path is modeled as largely bypassing L1. Hard to map directly to the measured L1/TEX hit rate (80.87%).
5. **DRAM bandwidth utilization is very low in sim** (bw_util ~0.005, dram_eff ~0.28), suggesting the cycle overestimation stems more from the compute/scheduling pipeline model than from memory.

> Note: NCU reports `Invocations=1` (single-call measurement), and the simulator likewise runs a single launch_uid=1 / trace_kernel_id=10, so the 1:1 comparison is appropriate. IPC is excluded from direct comparison because the sim's gpu_ipc (=1670, aggregated across all cores) and NCU's per-SM IPC (=1.28) use different definitions.
