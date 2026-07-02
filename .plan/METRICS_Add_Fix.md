# Metrics Add / Fix Plan

Goal: add missing simulator metrics and fix misleading existing metrics so FA3 and future H100
validation can compare simulator output against NCU without guessing. The first priority is
bandwidth, because current logs already expose the problem: `DRAM_BW_total`, `L2_BW_total`, and
`L1D_BW_total` exist, but their meanings are mixed and they do not map cleanly to NCU summary
metrics.

## 1. Cache / Memory Metrics

### 1.1 NV - Simulator metrics mapping

Architecture / measurement context:

- **NV hardware architecture**: NVIDIA Hopper H100 / SM90-SM90a class.
- **Simulator target config**: `SM90_H100_L2_50MB_80GB`.
- **Primary workload**: `flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24`, kernels 5 and 10.
- **NCU source**: `nv_reports/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.csv`
  plus raw `.ncu-rep` when deeper metrics are needed.
- **Important rule**: do not compare unlike metrics. NCU `Memory Throughput` is not automatically
  equivalent to simulator `DRAM_BW_total`; use DRAM-specific NCU metrics for DRAM BW, and use
  L1/L2/TMA-side simulator metrics for NCU memory/L1/L2-side throughput.
- **Exact current CSV naming**: `Section Name / Metric Name`.
- **Raw metric naming**: exact NCU hardware metric identifiers to request explicitly when the
  current section CSV is too coarse.

#### Memory throughput, byte/sec

- **Exact NCU metric name(s)**: CSV: `Memory Workload Analysis / Memory Throughput`; raw
  candidates: `dram__bytes.sum.per_second`, `dram__bytes_read.sum.per_second`,
  `dram__bytes_write.sum.per_second`, `lts__t_bytes.sum.per_second`,
  `l1tex__t_bytes.sum.per_second`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `SIM_MEM_BW_total_GBps` (planned aggregate from L1/L2/TMA/DRAM
  submetrics).
- **Current status**: missing.
- **Include in simulator summary?** Yes.
- **Config needed?** No for summary; raw NCU metrics need explicit collection.
- **Notes**: broad NCU memory BW. Do not map directly to `DRAM_BW_total`.

#### Memory throughput, percent of peak

- **Exact NCU metric name(s)**: CSV: `GPU Speed Of Light Throughput / Memory Throughput`;
  raw/derived: `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `SIM_MEM_THROUGHPUT_pct_of_peak` (planned).
- **Current status**: missing.
- **Include in simulator summary?** Yes.
- **Config needed?** Needs peak-BW config.
- **Notes**: requires configured peak or calibrated H100 peak. Lower priority than byte/sec.

#### DRAM throughput, percent of peak

- **Exact NCU metric name(s)**: CSV: `GPU Speed Of Light Throughput / DRAM Throughput`;
  raw/derived: `dram__throughput.avg.pct_of_peak_sustained_elapsed`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `DRAM_BW_total_GBps` plus `% peak` variant.
- **Current status**: partial.
- **Include in simulator summary?** Yes.
- **Config needed?** `% peak` needs peak DRAM BW config.
- **Notes**: existing `DRAM_BW_total` is useful, but add read/write/type/TMA splits.

#### DRAM bytes/sec

- **Exact NCU metric name(s)**: raw: `dram__bytes.sum.per_second`,
  `dram__bytes_read.sum.per_second`, `dram__bytes_write.sum.per_second`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `DRAM_BW_read_GBps`, `DRAM_BW_write_GBps`, `DRAM_BW_total_GBps`.
- **Current status**: missing split.
- **Include in simulator summary?** Yes.
- **Config needed?** NCU raw metric collection needed.
- **Notes**: best apples-to-apples target for simulator DRAM BW.

#### L1/TEX throughput, percent of peak

- **Exact NCU metric name(s)**: CSV: `GPU Speed Of Light Throughput / L1/TEX Cache Throughput`;
  raw/derived: `l1tex__throughput.avg.pct_of_peak_sustained_elapsed`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `L1TEX_BW_total_GBps`, split into `L1D_BW`, `L1C_BW`, `L1T_BW`.
- **Current status**: missing / misleading.
- **Include in simulator summary?** Yes.
- **Config needed?** No.
- **Notes**: H100 NCU combines L1/TEX-side units; simulator has separate L1D/L1C/L1T and
  instruction caches.

#### L1/TEX hit rate

- **Exact NCU metric name(s)**: CSV: `Memory Workload Analysis / L1/TEX Hit Rate`;
  raw/derived: `l1tex__t_sector_hit_rate.pct`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: existing `L1D_total_cache_miss_rate`, plus the implemented per-cache
  granular breakdown `<cache>_total_cache_{hits,mshr_hits,full_misses,sector_misses,true_hit_rate}`
  for L0I/L1I/L1D/L1C/L1T (see §6).
- **Current status**: partial (granular per-status breakdown + `true_hit_rate` implemented for all
  core caches; byte/throughput-side L1/TEX metrics still missing).
- **Include in simulator summary?** Yes.
- **Config needed?** No.
- **Notes**: current FA3 TMA bypass means L1D accesses are tiny; must print this caveat in
  analysis. The granular fields fix a prior bug where `MSHR_HIT` was dropped and `MISS`/`SECTOR_MISS`
  were merged (§6.1), which distorted any per-cache hit-rate derivation.

#### L2 throughput, percent of peak

- **Exact NCU metric name(s)**: CSV: `GPU Speed Of Light Throughput / L2 Cache Throughput`;
  raw/derived: `lts__throughput.avg.pct_of_peak_sustained_elapsed`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `L2_BW_total_GBps`, `L2_BW_read_GBps`, `L2_BW_write_GBps`,
  `L2_TMA_BW_GBps`.
- **Current status**: partial / misleading.
- **Include in simulator summary?** Yes.
- **Config needed?** No.
- **Notes**: existing `L2_BW_total` is reply-ICNT based, not cache-access BW. Add real L2
  cache-byte counters.

#### L2 hit rate

- **Exact NCU metric name(s)**: CSV: `Memory Workload Analysis / L2 Hit Rate`;
  raw/derived: `lts__t_sector_hit_rate.pct`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: existing `L2_total_cache_miss_rate`, plus TMA/non-TMA hit-rate split.
- **Current status**: partial.
- **Include in simulator summary?** Yes.
- **Config needed?** No.
- **Notes**: TMA split is required for Opt 6 validation.

#### L2 bytes/sec

- **Exact NCU metric name(s)**: raw: `lts__t_bytes.sum.per_second`, plus read/write/sector
  variants if available from `ncu --query-metrics`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `L2_cache_BW_total_GBps`, `L2_cache_BW_read_GBps`,
  `L2_cache_BW_write_GBps`.
- **Current status**: missing.
- **Include in simulator summary?** Yes.
- **Config needed?** Raw NCU metrics need explicit collection.
- **Notes**: needed to validate L2-side traffic independently from DRAM.

#### L1/TEX bytes/sec

- **Exact NCU metric name(s)**: raw: `l1tex__t_bytes.sum.per_second`, plus pipe/op-specific
  variants if available from `ncu --query-metrics`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `L1TEX_BW_total_GBps`, `L1D_BW_*`, `L1C_BW_*`, `L1T_BW_*`.
- **Current status**: missing.
- **Include in simulator summary?** Yes.
- **Config needed?** Raw NCU metrics need explicit collection.
- **Notes**: needed because NCU summary gives percent/GBps only for broad memory workload.

#### Memory pipe busy

- **Exact NCU metric name(s)**: CSV: `Memory Workload Analysis / Mem Pipes Busy`; raw/derived
  candidates: `l1tex__throughput.avg.pct_of_peak_sustained_active`,
  `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` family as needed.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `L1D/L1C/L1T data_port_util`, `fill_port_util`, planned LSU/TMA pipe BW.
- **Current status**: partial.
- **Include in simulator summary?** Yes.
- **Config needed?** No for simulator; raw NCU metrics need explicit collection.
- **Notes**: existing cache port util is printed for some caches; extend consistently.

#### TMA pipe active

- **Exact NCU metric name(s)**: raw: `sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active`;
  related exact metric in notes: `sm__inst_executed_pipe_tma`.
- **NV architecture version**: H100 / SM90a.
- **Simulator metric**: `TMA_BW_issued_GBps`, `TMA_BW_completed_GBps`,
  `TMA_requests_issued_per_cycle`.
- **Current status**: partial.
- **Include in simulator summary?** Yes.
- **Config needed?** `-tma_debug_enable` only for event logs; summary should not require it.
- **Notes**: TMA already has private bytes issued/completed; convert to summary BW and split by
  load/store/reduce.

#### Core-to-memory interconnect traffic

- **Exact NCU metric name(s)**: simulator-only; no direct NCU metric.
- **NV architecture version**: H100 / SM90a proxy only.
- **Simulator metric**: existing `traffic_breakdown_coretomem[...]`, planned
  `ICNT_core_to_mem_BW_*`.
- **Current status**: partial.
- **Include in simulator summary?** Optional summary, full details raw.
- **Config needed?** No.
- **Notes**: useful to diagnose injection pressure.

#### Memory-to-core interconnect traffic

- **Exact NCU metric name(s)**: simulator-only; no direct NCU metric.
- **NV architecture version**: H100 / SM90a proxy only.
- **Simulator metric**: existing `traffic_breakdown_memtocore[...]`, planned
  `ICNT_mem_to_core_BW_*`.
- **Current status**: partial.
- **Include in simulator summary?** Optional summary, full details raw.
- **Config needed?** No.
- **Notes**: useful for reply bandwidth and L2-to-core pressure.

Summary policy:

- Always include top-level `DRAM_BW_total/read/write`, `L2_BW_total/read/write`, `L2_TMA_BW`,
  `TMA_BW_issued/completed`, and corrected `L1D/L1C/L1T/L1I` BW totals.
- Print detailed per-access-type and per-status tables in the full stats section, not necessarily
  in the compact summary.
- Add a clear label to old metrics that are not direct cache BW, or replace them with correctly
  named metrics.

### 1.2 Sim metrics calculation logic

Use one consistent bandwidth formula:

```text
BW_GBps = bytes / (cycles * clock_period_seconds) / 1e9
```

Use the clock domain for the point being measured:

| Simulator level | Clock period | Byte source | Main output |
|---|---|---|---|
| L0I / L1I / L1D / L1C / L1T | `core_period` | cache access byte counters | `Lx_BW_*_GBps` |
| L2 cache | `l2_period` | L2 cache access byte counters | `L2_BW_*_GBps` |
| Core-to-memory ICNT | `icnt_period` | existing `traffic_breakdown_coretomem` bytes | `ICNT_core_to_mem_BW_*_GBps` |
| Memory-to-core ICNT | `icnt_period` | existing `traffic_breakdown_memtocore` bytes | `ICNT_mem_to_core_BW_*_GBps` |
| DRAM | `dram_period` | DRAM atom bytes | `DRAM_BW_*_GBps` |
| TMA unit | core cycle or L2/ICNT cycle, explicitly labeled | TMA issued/completed bytes | `TMA_BW_issued_*`, `TMA_BW_completed_*` |

Implementation details:

1. **Extend `cache_stats` with byte counters.**
   - Current `cache_stats` records counts only:
     `m_stats[mem_access_type][cache_request_status]`.
   - Add `m_bytes[mem_access_type][cache_request_status]`.
   - Add `inc_stats_bytes(access_type, status, bytes)` or extend `inc_stats`.
   - Use `mf->get_data_size()` or `mf->get_access_size()` consistently. Prefer the mem_fetch data
     size at the cache access point because sector splitting may already have changed size.
   - Preserve current count counters for compatibility.

2. **Add read/write/type/status split helpers.**
   - Read bytes: non-write mem_fetches.
   - Write bytes: write mem_fetches.
   - Type split: `GLOBAL_ACC_R`, `GLOBAL_ACC_W`, `LOCAL_ACC_R`, `LOCAL_ACC_W`, `CONST_ACC_R`,
     `INST_ACC_R`, `TEXTURE_ACC_R`, `L1_WRBK_ACC`, `L2_WRBK_ACC`, `L1_WR_ALLOC_R`,
     `L2_WR_ALLOC_R`, `TLB_MISS_ACC_DATA`, `TLB_MISS_ACC_INST`.
   - Status split: `HIT`, `MISS`, `SECTOR_MISS`, `HIT_RESERVED`, `RESERVATION_FAIL`.
   - For bandwidth summary, exclude `RESERVATION_FAIL` by default; optionally print a separate
     `*_replay_or_fail_bytes` if the failed request consumed a modeled port.

3. **Fix L1 metrics.**
   - Replace the current `L1D_BW_total` calculation. It currently derives from aggregate
     `GLOBAL_ACC_R/W` access counts only and multiplies by 32, which is not a true L1D BW metric.
   - Print individual cache BW:
     - `L0I_BW_total_GBps`
     - `L1I_BW_total_GBps`
     - `L1D_BW_total_GBps`, `L1D_BW_read_GBps`, `L1D_BW_write_GBps`
     - `L1C_BW_total_GBps`
     - `L1T_BW_total_GBps`
   - Keep cache hit/miss/access counters unchanged.

4. **Fix L2 metrics.**
   - Keep old `L2_BW_total` only if renamed to something like
     `ICNT_mem_to_core_reply_BW_total_GBps`; it is currently based on replies, not full L2 traffic.
   - Add real L2 cache byte counters from `cache_stats`.
   - Add TMA split by checking `mf->is_tma()` at L2 access time:
     - `L2_TMA_BW_read_GBps`
     - `L2_TMA_BW_write_GBps`
     - `L2_nonTMA_BW_read_GBps`
     - `L2_nonTMA_BW_write_GBps`
   - Add TMA/non-TMA L2 hit-rate split if feasible using the same byte/count classification.

5. **Extend DRAM BW.**
   - Existing `DRAM_BW_total` is based on DRAM atom counts and `dram_period`.
   - Add:
     - `DRAM_BW_read_GBps`
     - `DRAM_BW_write_GBps`
     - `DRAM_BW_by_access_type[...]`
     - `DRAM_TMA_BW_GBps`, `DRAM_nonTMA_BW_GBps`
   - `mem_fetch::is_tma()` is inherited by L2 sector-split children, so this split should work for
     both current 32B TMA sectors and future 128B parent/child accounting.

6. **Add TMA-private BW summary.**
   - Existing TMA counters: `m_stat_bytes_issued`, `m_stat_bytes_completed`, requests issued /
     completed.
   - Add load/store/reduce byte counters.
   - Print:
     - `TMA_BW_issued_total_GBps`
     - `TMA_BW_completed_total_GBps`
     - `TMA_BW_issued_load/store/reduce_GBps`
     - `TMA_requests_issued_per_cycle`
     - `TMA_icnt_backpressure_events` if not already summarized.
   - This should be printed in summary without requiring `-tma_debug_enable`; debug flag should only
     control verbose per-transfer logs.

7. **Normalize interconnect traffic breakdowns.**
   - Existing `traffic_breakdown_coretomem[...]` and `traffic_breakdown_memtocore[...]` already
     print bytes by type.
   - Add normalized GB/s lines using `icnt_period`.
   - Keep the raw histograms because packet sizes distinguish control packets from data packets.

8. **Naming convention.**
   - Use `*_BW_*_GBps` for byte/sec bandwidth.
   - Use `*_util` only for port utilization / cycles busy.
   - Use `ICNT_*` for interconnect packet traffic.
   - Use `L2_cache_*` for true L2 cache accesses.
   - Do not use `L2_BW_total` for reply-ICNT traffic without an `ICNT` label.

9. **Validation checks.**
   - For FA3 fwd/bwd, compare:
     - NCU DRAM bytes/sec vs `DRAM_BW_*`
     - NCU L2 throughput/hit rate vs `L2_cache_BW_*` and `L2_total_cache_miss_rate`
     - NCU L1/TEX throughput/hit rate vs `L1D/L1C/L1T` combined view
     - NCU TMA pipe activity vs `TMA_BW_*` and TMA request rate
   - Confirm bandwidth changes are explainable after Opt 6A:
     - `TMA_BW_issued` should change if 32B sectors become 128B line mfs.
     - `DRAM_BW_*` should not collapse to zero.
     - L2 hit rate must not become less realistic unless the change is intentionally an address
       model experiment.

## 2. Scheduler / Warp-Issue Metrics

Planned follow-up. Add NCU-style scheduler metrics:

- `selected`
- `not_selected`
- `dispatch_stall`
- `eligible_warps_per_scheduler`
- `active_warps_per_scheduler`
- per-SMSP issue-slot utilization

Current issue-stage counters are useful but not NCU-shaped enough for direct comparison.

## 3. Pipe Utilization Metrics

Planned follow-up. Add pipe utilization split:

- tensor / GMMA / HMMA
- XU / SFU
- LSU / MIO
- ALU / FMA
- TMA pipe

This is needed to compare against NCU pipe utilization and to avoid over-prioritizing WGMMA or
frontend fixes when the SM-level recoverable budget is small.

## 4. TMA Latency Metrics

Partially exists via debug logs. Promote selected fields into summary:

- `TMA_lat_total_mean/median/p90/max`
- `TMA_lat_mem_mean/median/p90/max`
- `TMA_lat_issue_mean/median/p90/max`
- `TMA_lat_to_first_request_mean/median/p90/max`
- `TMA_lat_emit_mean/median/p90/max`
- `TMA_lat_drain_mean/median/p90/max`
- `TMA_issue_active_cycles_mean`
- `TMA_icnt_full_cycles_mean`
- `TMA_requests_per_issue_active_cycle`
- `TMA_icnt_backpressure_events`
- load/store/reduce split

Do not require per-event logging for these summary metrics.

## 5. Metric Correctness Fixes

Known metrics to rename or fix:

| Current metric | Problem | Fix |
|---|---|---|
| `L1D_BW_total` | Derived from aggregate core cache `GLOBAL_ACC_R/W` counts and fixed 32B size; not true L1D BW. | Replace with byte counters from L1D cache accesses. |
| `L2_BW_total` | Actually memory-to-core reply traffic from `partiton_replys_in_parallel`, not total L2 cache BW. | Rename to `ICNT_mem_to_core_reply_BW_total_GBps`; add true `L2_cache_BW_total_GBps`. |
| `DRAM_BW_total` | Useful, but only total. No read/write/type/TMA split. | Add split metrics. |
| `traffic_breakdown_*` | Raw byte totals only, no normalized BW. | Add GB/s normalized lines. |
| NCU `Memory Throughput` comparison | Was sometimes compared directly to `DRAM_BW_total`. | Compare DRAM-to-DRAM and L1/L2/TMA-to-memory-side metrics separately. |
| `<cache>_total_cache_misses` / `_miss_rate` (all caches via `cache_sub_stats`) | `cache_stats::get_sub_stats()` merged `MISS`+`SECTOR_MISS` into one `misses` field, counted only `HIT_RESERVED` as `pending_hits`, and dropped `MSHR_HIT` from every field (not in `accesses` or `misses`). Made it impossible to separate full vs sector misses, and silently lost MSHR-merged hits, distorting hit-rate math for L1/L2/L0I. | Added granular `hits`/`mshr_hits`/`full_misses`/`sector_misses` fields + `true_hit_rate`, populated at the single `get_sub_stats()` aggregation point and printed for every cache via `print_hit_breakdown()`. Legacy fields left untouched so existing numbers do not move (§6.1). |

## 6. Implemented per-status cache metrics (done)

These are already in the code tree (build/run validation still pending on a CUDA host). Both are
timing-neutral observers — they only read existing per-status counters or tag classifications and
do not change any scheduling/latency behavior.

### 6.1 Generic per-status breakdown for every cache

- **Problem fixed**: `cache_sub_stats` (the struct every cache aggregates through) collapsed the
  `cache_request_status` enum: `misses = MISS + SECTOR_MISS`, `pending_hits = HIT_RESERVED` only,
  and `MSHR_HIT` was counted **nowhere** (absent from `accesses`, `misses`, and `pending_hits`).
- **New fields** (in [gpu-cache.h](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h#L1149-L1152) `struct cache_sub_stats`):
  - `hits` ← `HIT`
  - `mshr_hits` ← `MSHR_HIT` (previously dropped)
  - `full_misses` ← `MISS`
  - `sector_misses` ← `SECTOR_MISS`
- **Populated** at the single aggregation point [cache_stats::get_sub_stats()](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1035-L1042); legacy
  `accesses`/`misses`/`pending_hits`/`res_fails` are deliberately **unchanged** so historical
  miss-rate numbers and the L2 gate do not silently shift.
- **Printed** via `cache_sub_stats::print_hit_breakdown(fout, prefix)` for all core caches in
  [shader.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L3527) (L0I, L1I, L1D, L1C, L1T) and for L2 in
  [gpu-sim.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3454).
- **Output lines** (per cache `X` ∈ {L0I, L1I, L1D, L1C, L1T, L2}):
  - `X_total_cache_hits`
  - `X_total_cache_mshr_hits`
  - `X_total_cache_full_misses`
  - `X_total_cache_sector_misses`
  - `X_total_cache_true_hit_rate` = `hits / accesses` (denominator = legacy `accesses` =
    `HIT+MISS+SECTOR_MISS+HIT_RESERVED`, so it lines up with the existing `_miss_rate`).
- **Caveat**: `mshr_hits` is outside the legacy `accesses` base, so it is reported raw (not folded
  into `true_hit_rate`). If a future NCU correlation needs MSHR hits counted as hits, define a
  separate rate rather than mutating `accesses`.
- **Not touched**: the AerialVision per-window path `get_sub_stats_pw()` (time-series visualizer,
  separate from summary hit-rate); left as-is by design.

### 6.2 TMA-only L2 admission counters (Opt 6 Part-0)

- **Why**: the generic `L2_total_cache_*` mixes TMA with normal LDG/STG, so it cannot attribute the
  ADDR_MERGE synthetic-address RESERVATION_FAIL re-probe storm to TMA. These isolate `is_tma()`
  requests at the L2 admission probe.
- **Counted** per-sub-partition in [memory_sub_partition::cache_cycle()](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L530-L572) (admission probe + the two backpressure skip-causes); members + getters in
  [l2cache.h](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h#L212-L301); summed and printed in [gpu-sim.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3476-L3496).
- **Output lines**:
  - `L2_TMA_hits` / `L2_TMA_true_hit_rate` — true HIT (genuine locality)
  - `L2_TMA_pending_hits` / `L2_TMA_pending_hit_rate` — `HIT_RESERVED`/`MSHR_HIT`, i.e. merged onto
    an in-flight miss (the cross-SM single-base collision fingerprint; **not** a free hit)
  - `L2_TMA_misses` — `MISS`/`SECTOR_MISS` → DRAM
  - `L2_TMA_reservation_fails` / `L2_TMA_res_fail_per_probe` — re-probe **cycles** (head-of-line
    blocking from the synthetic-address hotspot), not distinct failing requests
  - `L2_TMA_output_full_cycles` — head mf is TMA but the L2→ICNT reply queue (`m_L2_icnt_queue`) is
    full, so `access()` is skipped this cycle (downstream reply backpressure, **not** a res_fail)
  - `L2_TMA_port_busy_cycles` — head mf is TMA, reply queue not full, but the L2 data port is busy,
    so `access()` is skipped (port backpressure, **not** a res_fail)
- **Why the two backpressure counters matter (avoids a second run)**: the hit/miss/res_fail counters
  only advance on cycles where `access()` actually runs, i.e. the admission gate
  `!output_full && port_free` is open. A TMA head can be stuck for **four** distinct reasons; without
  separating them a *low* `res_fail_per_probe` could be misread as "admission is not the limiter"
  when the head is really jammed downstream. The four head-of-line causes are now fully separable
  from a single run:
  1. DRAM queue full → existing global `gpu_stall_dramfull`
  2. reply queue full → `L2_TMA_output_full_cycles`
  3. data port busy → `L2_TMA_port_busy_cycles`
  4. L2 set/MSHR locked (the hotspot) → `L2_TMA_reservation_fails`
- **Decision gate** (vs HW L2 hit rate fwd 69.58% / bwd 82.26%): high `res_fail_per_probe` + hit
  rate above HW ⇒ synthetic-address hotspot ⇒ **6B (address), not 6A**; low `res_fail_per_probe`
  **with low `output_full`/`port_busy`** ⇒ `lat_drain` is genuine memory latency, neither 6A nor 6B
  helps; low `res_fail_per_probe` **with high `output_full`/`port_busy`** ⇒ the limiter is
  downstream reply/port backpressure (a different fix axis, not the address hotspot). Cross-referenced
  in `.plan/TMA_LATENCY_INJECTION_H100.md` §2-C / §4.
- **Measured (fwd `.o24` / bwd `.o6`, 2026-07-01)**: `res_fail_per_probe = 0`, `port_busy = 0`, but
  `output_full_cycles = 180,998 / 314,724` and `gpu_stall_icnt2sh = 258,818 / 464,997`. → the gate
  landed on the **third branch: L2→core reply-path backpressure**, not the hotspot and not injection.
  The two backpressure counters were decisive here — without them a `res_fail = 0` would have been
  misread as "no admission pressure".
- **Follow-up A/B (fwd k5, 4 runs `.o24/.o25/.o7/.o8`, 2026-07-02)**: raising reply-queue depth
  (64→256) and/or drain rate (1→4) moved `output_full` a lot but **cycles stayed within ±0.8%**;
  drain=4 just relocated the stall to `gpu_stall_icnt2sh` (259K→722K). So `output_full` is a
  **symptom, not the lever**, and — critically — all runs sat on fake ~98% L2 hit, so the reply
  path can't be judged until TMA addresses are realistic. Next step is 6B (address realism), then
  re-run this A/B. Full result in `.plan/TMA_LATENCY_INJECTION_H100.md` §4.5.
