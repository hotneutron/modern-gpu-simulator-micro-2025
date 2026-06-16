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

## Accuracy Strategy — Compare Warp-Issue Stall Breakdown (TMA vs non-TMA)

Plain cycle/occupancy/hit-rate numbers only show the *symptom*, not *where* the 2.83×
overestimation comes from. For a warp-specialized FA3 bwd kernel, almost all time is spent
in **warps that cannot issue (stall)**, so the right accuracy metric is the **per-reason
warp-issue stall distribution**, split into a **TMA (producer) axis** and a
**non-TMA (consumer / compute / scheduling) axis**.

### HW says: NOT memory-bandwidth bound — it is latency / pipeline-dependency bound
- `Compute (SM) Throughput 48.45%`, `Memory Throughput 56.58%`, but `DRAM Throughput 14.85%` and `Mem Pipes Busy 29.54%` → DRAM is not the bottleneck.
- `Issue Slots Busy 32.07%` / Scheduler `No Eligible 67.17%` → 2 of every 3 cycles issue nothing.
- `Warp Cycles Per Issued Instruction = 7.53` → each issued instruction is paid for with 7.53 warp-cycles of stall. **Decomposing this 7.53 is the accuracy target.**

### Ground truth: HW warp-issue stall decomposition (kernel 10, FlashAttnBwdSm90)

Source metrics: `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` from the
`.ncu-rep` raw page. The reasons sum to **7.539**, matching `Warp Cycles Per Issued Instruction = 7.53` exactly.

| Stall reason (NCU) | Value | Share | Axis |
|---|---|---|---|
| long_scoreboard | 1.491 | **19.8%** | **TMA** — global-memory / TMA data-arrival latency |
| barrier | 1.313 | **17.4%** | **TMA** — mbarrier / named-barrier (producer↔consumer sync) |
| selected (actually issued) | 1.000 | 13.3% | (issued, not a stall) |
| wait | 0.784 | 10.4% | non-TMA — fixed-latency dependency |
| short_scoreboard | 0.643 | 8.5% | non-TMA — short-latency (shared/MUFU) wait |
| mio_throttle | 0.474 | 6.3% | non-TMA — shared/LSU (MIO) pipe throttle |
| not_selected | 0.405 | 5.4% | non-TMA — scheduler contention |
| gmma | 0.398 | **5.3%** | **non-TMA** — WGMMA (tensor) pipe wait |
| dispatch_stall | 0.338 | 4.5% | non-TMA — dispatch |
| sleeping | 0.328 | 4.4% | TMA — warp-specialization producer idle |
| no_instruction | 0.115 | 1.5% | frontend — I-cache / fetch |
| imc_miss | 0.094 | 1.3% | frontend — constant/immediate cache |
| math_pipe_throttle | 0.092 | 1.2% | non-TMA — math pipe throttle |
| branch_resolving | 0.060 | 0.8% | non-TMA — branch |
| misc / drain | ~0.003 | 0.0% | misc |
| membar / lg_throttle / tex_throttle | 0.000 | 0.0% | n/a |
| **TOTAL** | **7.539** | 100% | |

**TMA axis (long_scoreboard + barrier + sleeping) ≈ 41.6%** of warp-cycles;
**non-TMA axis (wait + short_scoreboard + mio + gmma + not_selected + dispatch + math + branch) ≈ 42.4%**;
issued 13.3%; frontend ~2.8%. → TMA pipeline and consumer compute/scheduling are roughly equal contributors, so both must be modeled accurately.

### HW pipe utilization (anchors the non-TMA axis)

| Pipe | % of peak (active) | Note |
|---|---|---|
| sm__inst_executed_pipe_xu (MUFU/transcendental) | 21.44% | exp/log in softmax bwd |
| sm__inst_executed_pipe_lsu | 16.28% | shared-memory traffic |
| sm__inst_executed_pipe_alu | 15.34% | |
| sm__pipe_tensor_cycles_active | 15.33% | WGMMA+HMMA combined |
| sm__inst_executed_pipe_fma | 9.45% | |
| sm__inst_executed_pipe_tensor_op_gmma | 5.36% | WGMMA |
| sm__inst_executed_pipe_tensor_op_hmma | 2.68% | HMMA |
| sm__inst_executed_pipe_tma | 0.19% | TMA insts are rare, but cause 37%+ of stalls |

Shared memory: `lsu_wavefronts_mem_shared 56.6% of peak`, with **store bank conflicts = 35,493**
(`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st`) — a real non-TMA cost the sim must capture.

---

## Sim Comparison Metrics — map to existing (mostly dormant) counters

The simulator already classifies the issue stage every cycle in
`subcore.cc::cycle()` (the `total_num_cycles_issue_stage_*` family). The coarse buckets are
emitted today; the **per-reason `no_warps_ready` breakdown that maps to the HW stall reasons
is defined but commented out** at both the increment site and the print site.

| HW stall reason | Sim counter (flag in subcore.cc) | Status |
|---|---|---|
| barrier / long_scoreboard (TMA) | `is_any_waiting_in_wait_barrier` (mbarrier) + `is_not_warp_waiting_tma_flush` | flag exists, **aggregation commented out** |
| long_scoreboard (async load) | `num_pending_ldgsts` / `is_waiting_ldgdepbar` | tracked, not bucketed |
| gmma / math_pipe_throttle (non-TMA) | `is_any_waiting_in_fu_occupied` (`!is_fu_available`) | flag exists, **commented out** |
| wait (non-TMA) | `is_any_waiting_in_stall_count` (`!is_stall_counter_0`) | flag exists, **commented out** |
| short_scoreboard / mio_throttle (non-TMA) | `is_any_waiting_l1c` (`!is_l1c_ready`) | flag exists, **commented out** |
| not_selected (non-TMA) | greedy-scheduler skip (warp ready but not picked) | not yet a counter |
| no_instruction (frontend) | `total_num_cycles_issue_stage_stall_no_valid_instruction_*` | **already emitted** |
| (issued) | `total_num_cycles_issue_stage_issuing` | **already emitted** |

### Action items to enable the comparison
1. **DONE — Re-enabled the dormant `no_warps_ready` sub-breakdown** in
   [subcore.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc) (the `is_any_waiting_in_*` flags), each wired to an `m_sm_stats`
   counter: `..._waiting_wait_barrier`, `..._waiting_inst_barrier`, `..._waiting_tma_flush`,
   `..._with_fu_occupied`, `..._waiting_stall_count`, `..._waiting_l1c`, `..._waiting_scoreboard`,
   `..._waiting_result_queue_full`, `..._waiting_yield`.
2. **DONE — Registered the counters** in [gpu-sim.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) and **re-enabled + extended the prints** in
   [shader.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc): per-reason counts, per-reason percentages, plus the grouped
   `total_percentage_cycles_issue_stage_stall_tma_axis` and `..._non_tma_axis`.
3. **DONE — A dedicated TMA-flush counter** (`..._waiting_tma_flush`) isolates the
   `cp.async.bulk.wait_group` drain from the mbarrier (`wait_barrier`) wait, so the **TMA axis**
   is fully separated from the non-TMA scoreboard waits.
4. All sim buckets are normalized by `total_num_cycles_issue_stage_evaluated`; compare the
   distribution shape against the HW table above (not just absolute cycles).

### Sim ↔ HW axis mapping (emitted percentages)

| Sim emitted metric | HW reason it tracks | Axis |
|---|---|---|
| `..._stall_tma_axis` (= wait_barrier + inst_barrier + tma_flush) | long_scoreboard + barrier (≈37%) | **TMA** |
| `..._waiting_wait_barrier` | barrier / long_scoreboard (mbarrier) | TMA |
| `..._waiting_inst_barrier` | barrier (named) / ldgdepbar | TMA |
| `..._waiting_tma_flush` | drain / long_scoreboard (bulk store) | TMA |
| `..._stall_non_tma_axis` (= fu_occupied + stall_count + l1c + scoreboard + result_queue + yield) | gmma + wait + short_scoreboard + mio (≈42%) | **non-TMA** |
| `..._with_fu_occupied` | gmma / math_pipe_throttle | non-TMA |
| `..._waiting_stall_count` | wait | non-TMA |
| `..._waiting_l1c` | short_scoreboard | non-TMA |
| `..._waiting_scoreboard` | short/long_scoreboard (traditional) | non-TMA |

### Extra logging/counter options in the GPU config (gpgpusim.config)

The H100 config already exposes several verbosity knobs that complement the new stall counters
([SM90_H100 gpgpusim.config](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config)):

| Option | Effect | Use for |
|---|---|---|
| `-sync_debug_enable 1` | Per-SM **`[SYNCDBG]`** (mbarrier) + **`[TMADBG]`** (TMA transfer) event logs to stderr | Trace exactly when the **TMA axis** stalls (producer/consumer mbarrier handshake, bulk-copy issue/complete) |
| `-sync_debug_print_budget N` | Caps `[SYNCDBG]`/`[TMADBG]` lines per SM (currently 40,000,000) | Bound log size on long runs |
| `-sync_debug_skip_runtime_budget N` | Caps "skip sync" log lines per SM | Reduce noise |
| `-gpgpu_runtime_stat 500` | Periodic runtime stat dump every 500 cycles | Time-series of stalls, not just end-of-kernel totals |
| `-gpgpu_memlatency_stat 14` | Detailed memory-latency histograms | Cross-check `long_scoreboard` (TMA/global latency) modeling |
| `-enable_ptx_file_line_stats 1` | Per-source-line stat attribution | Attribute stalls to specific FA3 code regions |
| `-measure_coalescing_potential_stats 1` | Intra/inter-warp coalescing distance stats (more RAM) | Memory-access modeling (off by default) |

### Per-cycle behavior trace (`-trace_enabled`) — not just stats

Separate from the stat counters above, GPGPU-Sim has a **per-cycle behavior trace** that prints
the actual operation of each component to stdout in the form
`GPGPU-Sim Cycle <N>: <COMPONENT> - ...`. It is commented out in the config by default
([gpgpusim.config](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config#L203-L206)):

```
-trace_enabled 1
-trace_components WARP_SCHEDULER,SCOREBOARD,MEMORY_PARTITION_UNIT,MEMORY_SUBPARTITION_UNIT,INTERCONNECT,LIVENESS,TMA
-trace_sampling_core 0
```

| Component | Behavior printed |
|---|---|
| `WARP_SCHEDULER` | per-cycle warp selection / issue decisions |
| `SCOREBOARD` | scoreboard reserve / release / collision |
| `MEMORY_PARTITION_UNIT` / `MEMORY_SUBPARTITION_UNIT` | L2 / DRAM request handling |
| `INTERCONNECT` | packet movement |
| `LIVENESS` | per-core progress heartbeat |
| **`TMA`** (**new — added for this work**) | **per-event TMA unit operation**: command enqueue, AGU-ready, in-flight, first request (L1-bypass / shared icnt), store-write / reduce-read issue, completion + mbarrier credit, prefetch fire-and-forget, store outstanding++/-- |

**`-trace_sampling_core 0`** restricts the trace to a single core (here core 0) so the log does
not explode; leave it unset to trace all cores.

#### How the TMA trace component was added (new architecture component)
1. Registered a `TMA` stream in [trace_streams.tup](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/trace_streams.tup#L35) (enum + `trace_streams_str` are generated from this `.tup`).
2. Routed the existing TMA event sink [SM::debug_log_tma_event](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L365-L376) into the trace system via `DPRINTF(TMA, ...)`, honoring `-trace_sampling_core`. The `[TMADBG]` stderr path (gated by `-sync_debug_enable`) is kept as a second, independent sink.
3. All TMA events already emitted by [tma_unit_sm.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc) (enqueue / in-flight / first-request / complete / prefetch / store-outstanding) now appear in the `-trace_enabled` stream automatically.

> Why this matters for accuracy: the **TMA axis** (`..._stall_tma_axis`) tells you *how much*
> time is lost to TMA waits; the `TMA` trace tells you *exactly which* transfer/mbarrier event
> caused each stall on a chosen core, so the modeled TMA latency / pipeline depth can be tuned
> against the real `long_scoreboard` + `barrier` behavior cycle by cycle.

> Recommended workflow: keep `-sync_debug_enable 1` while iterating on TMA latency / mbarrier
> depth so the `[TMADBG]`/`[SYNCDBG]` event stream can be correlated with the new
> `..._stall_tma_axis` counter; use `-gpgpu_runtime_stat` to see how the TMA vs non-TMA split
> evolves over the kernel rather than only at the end.

### Cycle-gap diagnostic logs (added — one run captures everything)

A full run takes ~40 h, so the logs below were chosen to make **re-runs unnecessary**: a single
run yields the stall distribution, the per-transfer TMA latency breakdown, and the complete
timing-model parameter set. No extra per-cycle cost.

**1. Per-transfer TMA latency breakdown** — appended to the `complete` TMA event
([tma_unit_sm.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L767-L779)), reusing the already-tracked
`cycle_enqueued / cycle_agu_ready / cycle_first_request / cycle_last_completion` fields:

```
complete uid=.. ... lat_total=.. lat_queue=.. lat_issue=.. lat_mem=.. cycle=..
```

| Field | Span | Diagnoses |
|---|---|---|
| `lat_queue` | enqueue → agu_ready | descriptor / AGU wait |
| `lat_issue` | agu_ready → first_request | request-issue serialization |
| `lat_mem` | first_request → complete | interconnect + L2 + DRAM roundtrip |
| `lat_total` | enqueue → complete | full modeled transfer latency (compare vs HW) |

This is the **primary signal for the cycle gap**: it pinpoints which stage of a TMA transfer the
model over-estimates, per transfer.

**2. One-time timing-model latency dump** (`[LATCFG]`) — printed once at init in
[shader_core_config::set_pipeline_latency()](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L3937-L3965).
Dumps **all** latency/initiation knobs (not just WGMMA) so the over-estimated parameter is
identifiable without a second run, each tagged with the NCU stall axis it feeds:

| `[LATCFG]` group | Key values (H100 config) | NCU axis |
|---|---|---|
| compute / tensor | `tensor`, `tensor_init`, `tensor_rate_per_cycle`, sfu, fp/int | gmma / math_pipe_throttle (non-TMA) |
| memSM | `shared_min=7`, `l1d_min=13`, `ldgsts=19`, `const_l1c=11` | short_scoreboard / mio (non-TMA) |
| regfile | `rf_fixed_latency=24` | wait (non-TMA) |
| (note line) | `l2_rop=211`, `dram=243`, `kernel_launch=1500` (read from config) | long_scoreboard (TMA + global) |

> Logs deliberately **not** added (40 h run = log-explosion risk, low marginal value): per-cycle
> mover progress, per-mbarrier-arrive events, every sector request (only the first is logged).

### Recommended single-run configuration (max info per 40 h run)
```
-trace_enabled 1
-trace_components TMA
-trace_sampling_core 0     # latency-breakdown sample from core 0 (avoids log explosion)
-sync_debug_enable 1       # all-SM [TMADBG]/[SYNCDBG] events too
```
End of run yields: new TMA/non-TMA stall counters + `[LATCFG]` dump + core-0 TMA latency
breakdown + all-SM TMA events — in a single pass.

> Note: the sim breakdown is **per-cycle, core-level** ("at least one warp waiting for X"),
> while NCU's `per_issue_active` is **per-issued-instruction, warp-averaged**. The reasons are
> not mutually exclusive across warps in a cycle, so sub-counters can sum to more than
> `no_warps_ready`. Compare the *relative shape / ranking* of the buckets rather than identical
> absolute ratios.

---

## Summary

1. **Cycle accuracy**: the simulator overestimates cycles by **~2.8–3.2×** vs real H100 (+183% on Elapsed basis). Significant room for improvement.
2. **Root cause is latency/pipeline, not bandwidth**: HW shows DRAM only 14.85% and `No Eligible 67%`. The accuracy gap should be debugged via the **warp-issue stall decomposition**, not DRAM/L2 numbers.
3. **Compare metric = stall distribution split TMA vs non-TMA**: TMA axis (long_scoreboard+barrier+sleeping ≈ 42%) and non-TMA axis (gmma/wait/short_scoreboard/mio/not_selected ≈ 42%) are co-dominant; the sum (7.539) matches `Warp Cycles Per Issued Instruction`.
4. **Occupancy matches closely** (14.49% vs 15.02%) — resource modeling is relatively accurate, so the gap is in timing/latency models, not occupancy.
5. **L2/L1 hit-rate divergence** (sim L2 98.7% vs 82.3%; sim L1 access only 9,912) is a secondary effect of the TMA path bypassing L1 in the model; it is not the primary cycle driver.
6. **Instrumentation already exists**: the per-reason stall counters needed for this comparison are present in `subcore.cc` but commented out — re-enabling them is the fastest path to a side-by-side TMA/non-TMA accuracy comparison.

> Note: NCU reports `Invocations=1` (single-call measurement), and the simulator likewise runs a single launch_uid=1 / trace_kernel_id=10, so the 1:1 comparison is appropriate. IPC is excluded from direct comparison because the sim's gpu_ipc (=1670, aggregated across all cores) and NCU's per-SM IPC (=1.28) use different definitions.
