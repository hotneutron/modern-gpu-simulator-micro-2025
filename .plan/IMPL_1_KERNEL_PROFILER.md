# Implementation Plan: [1] Kernel Profiler

**File:** `tools/kernel_profiler.py`
**Depends on:** `ncu` (Nsight Compute CLI) installed and on PATH
**Produces:** `kernel_report.csv`, `cta_features_{kernel_id}.csv`, `hotspot_ranking.txt`

---

## ncu Metric Lists

### Pass A — kernel-level metrics

```
smsp__cycles_elapsed.avg                      # execution time proxy
smsp__cycles_elapsed.avg.per_second           # for wall-clock time
sm__cycles_elapsed.max                        # total cycles

# Instruction mix
smsp__sass_thread_inst_executed_op_fp32_pred_on.sum
smsp__sass_thread_inst_executed_op_fp64_pred_on.sum
smsp__sass_thread_inst_executed_op_integer_pred_on.sum
smsp__sass_thread_inst_executed_op_tensor_pred_on.sum
smsp__sass_thread_inst_executed_op_sfu_pred_on.sum
smsp__sass_thread_inst_executed_op_memory_pred_on.sum
smsp__sass_thread_inst_executed_op_branch_pred_on.sum

# Memory
l1tex__t_bytes.sum                            # L1 bytes requested
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  # global load sectors
lts__t_bytes.sum                              # L2 bytes
lts__t_sectors_op_read.sum
lts__t_sectors_op_write.sum
dram__bytes_read.sum
dram__bytes_write.sum

# Hit rates
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct

# Occupancy + stalls
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct   # mem stall
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct  # compute stall
smsp__warp_issue_stalled_barrier_per_warp_active.pct
smsp__warp_issue_stalled_membar_per_warp_active.pct

# Coalescing
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct
```

### Pass B — per-CTA metrics (subset, cheaper)

Nsight Compute does not natively expose per-CTA counters. Two options:

**Option A (preferred):** Use `--replay-mode kernel` with `--target-processes all`
and filter post-hoc by CTA ID using the `--print-details` output. Nsight Compute
can serialize per-CTA data in CSV format with `--csv` and `--units base`.

**Option B (fallback):** NVBit counter instrumentation — insert counters per-CTA
using a lightweight NVBit tool (separate from the SASS tracer). This avoids ncu
overhead for Pass B.

Pass B metrics (7D feature vector for clustering):
```
sm__warps_active.avg                          # active warps (occupancy)
l1tex__t_sector_hit_rate.pct                  # L1 hit rate
lts__t_sector_hit_rate.pct                    # L2 hit rate
smsp__sass_thread_inst_executed.sum           # issued instructions
smsp__sass_thread_inst_executed_op_memory_pred_on.sum  # memory transactions
smsp__sass_thread_inst_executed_op_branch_pred_on.sum  # branches (divergence proxy)
smsp__inst_executed_pipe_lsu.sum              # shared mem accesses (proxy)
```

---

## Class Design

```python
# tools/kernel_profiler.py

@dataclass
class KernelRecord:
    kernel_id: int
    kernel_name: str
    invocation: int
    grid_x: int; grid_y: int; grid_z: int
    block_x: int; block_y: int; block_z: int
    cycles: int
    wall_time_us: float
    fp32_insns: int; fp64_insns: int; int_insns: int
    tc_insns: int; sfu_insns: int; mem_insns: int; branch_insns: int
    l1_hit_rate: float; l2_hit_rate: float
    dram_bytes_read: int; dram_bytes_write: int
    mem_stall_pct: float
    occupancy_pct: float
    ld_coalescing_pct: float; st_coalescing_pct: float

@dataclass
class CTARecord:
    kernel_id: int
    cta_x: int; cta_y: int; cta_z: int
    active_warps: float
    l1_hit_rate: float; l2_hit_rate: float
    issued_insns: int; mem_transactions: int
    branch_insns: int; shmem_accesses: int


class KernelProfiler:
    def __init__(self, ncu_path: str = "ncu"):
        ...

    def run_pass_a(self, workload_cmd: list[str], output_dir: Path) -> list[KernelRecord]:
        """Run ncu Pass A on full workload. Returns per-invocation records."""

    def run_pass_b(self, workload_cmd: list[str], kernel_ids: list[int],
                   output_dir: Path) -> dict[int, list[CTARecord]]:
        """Run ncu Pass B on hot kernels only. Returns per-kernel CTA records."""

    def compute_hotspot_scores(self, records: list[KernelRecord]) -> pd.DataFrame:
        """hotspot_score = mean_exec_time × invocation_count, ranked descending."""

    def write_outputs(self, records, cta_records, output_dir: Path):
        """Write kernel_report.csv, cta_features_*.csv, hotspot_ranking.txt"""
```

---

## ncu Invocation

```python
def _run_ncu(self, workload_cmd, metrics, output_csv, kernel_id_filter=None):
    cmd = [
        self.ncu_path,
        "--csv",
        "--units", "base",
        "--metrics", ",".join(metrics),
    ]
    if kernel_id_filter is not None:
        cmd += ["--kernel-id", f"::{kernel_id_filter}:"]
    cmd += ["--"] + workload_cmd

    result = subprocess.run(cmd, capture_output=True, text=True)
    # parse CSV from result.stdout
    return self._parse_ncu_csv(result.stdout)
```

---

## Output Formats

**`kernel_report.csv`** — one row per kernel invocation:
```
kernel_id,kernel_name,invocation,grid_x,grid_y,grid_z,block_x,block_y,block_z,
cycles,wall_time_us,fp32_insns,fp64_insns,int_insns,tc_insns,sfu_insns,
mem_insns,branch_insns,l1_hit_rate,l2_hit_rate,dram_bytes_read,dram_bytes_write,
mem_stall_pct,occupancy_pct,ld_coalescing_pct,st_coalescing_pct
```

**`cta_features_{kernel_id}.csv`** — one row per CTA (Pass B):
```
kernel_id,cta_x,cta_y,cta_z,active_warps,l1_hit_rate,l2_hit_rate,
issued_insns,mem_transactions,branch_insns,shmem_accesses
```

**`hotspot_ranking.txt`** — human-readable sorted summary:
```
Rank  KernelID  Name                    Invocations  MeanTime(us)  HotspotScore
   1  kernel_3  volta_sgemm_128x64_nn   47           18432         866304
   2  kernel_7  _Z14calculate_temp...   7            12006          84042
```

---

## CLI Interface

```
python tools/kernel_profiler.py \
  --workload "./app arg1 arg2" \
  --output-dir ./profiler_output/ \
  --hotspot-top-k 5 \
  [--pass-b-only]          # skip Pass A, re-use existing kernel_report.csv
  [--kernel-ids 3,7,12]    # override: run Pass B on specific kernels
  [--ncu-path /usr/local/cuda/bin/ncu]
```

---

## Edge Cases

- **Kernel deduplication**: multiple invocations of the same kernel name get
  the same `kernel_id`; `invocation` counter distinguishes them. For Pass B,
  profile only the first (or median) invocation.
- **Short kernels**: skip Pass B for kernels with `wall_time_us < 500` (not
  worth the profiling overhead).
- **ncu timeout**: wrap subprocess call with a configurable timeout; log and
  continue on failure.
- **Missing metrics**: some metrics unavailable on older GPUs (e.g., TC metrics
  on SM75). Gracefully substitute 0 and warn.
