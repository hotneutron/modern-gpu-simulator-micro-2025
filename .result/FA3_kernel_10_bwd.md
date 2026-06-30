# FA3 BWD Kernel 10 — Simulator vs. Real H100 Comparison

## UPDATE — Opt 5 (L1I eager-promote) simulator result (2026-06)

The detailed sections below were written at the **Opt 1 (rop=100)** stage (sim = 361,760 cycles).
The simulator has since improved through Opt 2 (BAR engine fix), Opt 3 (MEMBAR scope fix),
Opt 4 (deeper L1I stream buffer, sb=4), and Opt 5 (L1I eager-promote). **Real-HW (NCU) numbers
are unchanged**; only the simulator side is updated here.

### Simulator cycle progression (FA3 bwd, trace kernel 10, `FlashAttnBwdSm90`) — HW Elapsed = 132,901

| Stage | sim cycles | Sim / HW | Δ vs prev |
|---|---|---|---|
| Init (rop=211) | 376,735 | 2.83× | — |
| Opt 1 (`rop=100`) | 361,760 | 2.72× | −4.0% |
| Opt 2 (BAR engine fix) | 328,643 | 2.47× | −9.1% |
| Opt 3 (MEMBAR scope fix) | 259,456 | 1.95× | −21.1% |
| Opt 4 (prefetch, sb=4) | 241,528 | 1.82× | −6.9% |
| **Opt 5 (L1I eager-promote)** | **241,425** | **1.82×** | **−0.04%** |

- Opt 5 run: `.../H100_80GB-OnlyKernel10/...warmup_-63a73d452237.o3` (clean exit; Step-0
  instrumentation counters are timing-neutral, so this is a valid Opt-5 baseline; supersedes the
  earlier `.o320` 242,270 that hit a teardown SIGSEGV — now fixed).
- eager-promote works (`eager_promote_to_cache=994,032`, `demand_hit_later=366,329`,
  `demand_miss_after_promote=0`, L1I miss rate 0.1977) but gives **essentially no bwd cycle
  benefit** — the bwd frontend stall is not on the critical path.

### Opt 5 issue-stage breakdown (top-level, mutually exclusive)

| Class | Opt 5 % | Note |
|---|---|---|
| no_warps_ready | 36.45% | dominant |
| issuing | 25.70% | |
| no_valid_instruction (frontend) | 18.63% | |
| next_stage_not_available | 18.52% | downstream pipe back-pressure |
| issue_port_busy | 0.70% | |

Inside `no_warps_ready` (overlapping `..._at_least_one_warp_*`, % of all eval cycles):
`non_tma_axis 26.66%`, `tma_axis 22.17%`, `fu_occupied 17.75%`, `wait_barrier 14.71%`,
`stall_count 7.40%`, `tma_flush 6.20%`, `inst_barrier 1.26%`.

### Opt 5 / Step-0 SM-idle decomposition (true SM-level, not per-subcore)

A Step-0 instrumentation run decomposed `sm_all_subcores_idle ≈ 18.45%` (cycles where **no**
subcore on the SM issued) by the dominant blocking reason. This corrects per-subcore over-counts:

| SM-idle reason | Opt 5 % | note |
|---|---|---|
| **no_valid_other** (ibuffer empty / decode / not stream-buffer) | **10.55%** | coarse bucket; follow-up split shows this is almost entirely `nv_ibuffer_empty` = tail-drain / winding-down warp imbalance |
| **wait_barrier** (mbarrier / DEPBAR) | **11.09%** | #2 |
| no_valid_frontend (incl. `sbwait` 5.31%) | 5.49% | the L1I frontend send-bandwidth idea was deferred / parked (only ~5% recoverable) |
| stall_count | 3.82% | |
| tma_flush | 4.76% | bwd-only |
| fu_occupied (tensor 1.64%) | 2.84% | WGMMA fix deferred (≤1.6% recoverable) |
| next_stage | 2.12% | |

> Follow-up split instrumentation resolved the old `no_valid_other` bucket: almost all of it is
> `nv_ibuffer_empty`, while `nv_ibuf_fetch_inflight = 0` and `nv_ibuf_fetch_not_issued ~= 0`, so the
> dominant residual is best interpreted as **tail-drain / winding-down warp imbalance**, not an
> actionable frontend fetch bottleneck. The WGMMA idea and the L1I frontend send-bandwidth idea
> therefore remain **deferred / parked**.

---

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

## Pre-run cycle-reduction analysis (no experiment, code + NCU only)

Before spending another ~40 h run, the existing data was used to locate the
over-estimation **quantitatively**. The NCU raw report contains 11 kernel
invocations; the target (FlashAttnBwdSm90, k10) is the row with
`smsp__average_warp_latency_per_inst_issued.ratio = 7.531`
(`inst_executed=19,952,816`, `tma_ld=13,824`, `tma_st=7,296`,
`shared_gmma=835,584`, `sm__pipe_tensor_cycles_active=53.6%`, `gpc_cycles_elapsed=131,632`).

### The decisive contradiction
- Sim **L2 hit rate = 98.7%** is *higher* than HW **82.3%**, yet sim cycles are **2.86×** HW.
- A higher hit rate should make sim *faster*. It is slower → the inflation is **per-access modeled latency / non-overlap**, **not** bandwidth or miss count. DRAM is only 14.85% busy on HW and L2 miss is 1.3% in sim, so DRAM/L2-miss tuning cannot explain the gap.

### Ranked over-estimation candidates

| # | Candidate | Evidence | Verdict |
|---|---|---|---|
| **1** | **L2 ROP delay `rop_latency=211` on every global/TMA access** | [l2cache.cc:811-818](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L811-L818): `r.ready_cycle = cycle + rop_latency` is added to **all non-texture reqs before L2 lookup**, so even the 98.7% L2 hits pay +211 one-way (~422 RT). HW Hopper L2-hit **round-trip** ≈ 150–220 cyc. Directly inflates HW `long_scoreboard` (20.2%). | **REDUCE → done (211→100)** |
| 2 | `dram_latency=243` stacked on rop for L2 miss (rop+dram+rop ≈ 665) | Sim L2 miss only 1.3% in this kernel → small effect here | keep (not the driver) |
| 3 | Clock 1800 MHz vs NCU-measured 1430 MHz (`-gpgpu_clock_domains 1800:...`) | cycle-vs-cycle comparison is clock-independent for fixed-cycle latencies | **not touched this run** (per decision) |
| 4 | WGMMA/tensor latency | [abstract_hardware_model.cc:427-438](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L427-L443) `M·N·K·bits/rate` → II=16–64 cyc; serialized ≈ 25k–100k cyc, consistent with HW tensor-pipe 53.6% busy; HW `gmma` stall only 5.4% | **DO NOT touch** (`tensor_latency=32`) |
| 5 | TMA issue serialization (`kMaxRequestsPerCycle` in [tma_unit_sm.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc)) | could inflate `barrier` (17.8%) | confirm via new `lat_issue` log post-run |

### Additional findings (beyond the ranked latency knobs)

**[B] `barrier` stall (HW 17.8%, the 2nd-largest reason) is a *downstream* effect of the TMA memory latency, not an independent knob.**
- In the model, the consumer's mbarrier credit is granted **only when the TMA transfer completes** ([tma_unit_sm.cc:789-797](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L789-L797)). So the time a consumer warp spends blocked on the mbarrier == the TMA transfer's `rop + L2/DRAM` round-trip.
- Therefore lowering `rop_latency` (#1) shrinks **both** `long_scoreboard` (the producer's data-arrival wait, 20.2%) **and** `barrier` (the consumer's mbarrier wait, 17.8%) at once. Together these are the two largest stalls (≈38%), which is why #1 is the single highest-leverage change.
- Action: none separate from #1; verify after the run that the emitted `..._stall_tma_axis` percentage (which lumps both) drops proportionally to the rop reduction.

**[C] TMA AGU request throttle `kMaxRequestsPerCycle = 2`** ([tma_unit_sm.h:47](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.h#L47)) **may over-serialize TMA transfer *issue*.**
- A transfer emits its data as 32B sector mem-fetches; the loop caps emission at **2 sector mfs per cycle** ([tma_unit_sm.cc:614-615](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L613-L666)). A 16 KB tile = 512 sectors ⇒ **~256 cycles just to *emit* one transfer** before any memory latency.
- HW says the TMA unit is essentially idle: `sm__pipe_tma_cycles_active = 0.33%`. So this emission throttle could be a **structural over-serialization absent on HW**, inflating `barrier`/`long_scoreboard`.
- **Uncertainty**: emission overlaps with in-flight memory latency, so the net impact may be partly hidden. **Do not change blindly.** The new per-transfer `lat_issue` field (`agu_ready → first_request`, in the `complete` TMA log) will quantify it; if `lat_issue` is large, raise `kMaxRequestsPerCycle` in the next iteration.

> Note: TMA bypassing L1 (GMEM↔SMEM direct, going through L2) is the **correct** Hopper behavior ([tma_unit_sm.cc:628](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L628), [tma_unit_sm.h:83](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.h#L83)) and is **not** a problem — the low sim L1 access count is expected.

### Change applied for this run
- `gpgpusim.config`: **`-gpgpu_l2_rop_latency 211 → 100`** ([config](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config#L169-L176)). Runtime config → no rebuild needed.
- Rationale: rop is the **only** latency paid by 100% of global/TMA accesses (incl. L2 hits), so it is the single highest-leverage knob for the `long_scoreboard` axis and the overall 2.86× gap; via [B] it also shrinks `barrier`.
- Everything else (tensor, dram, clock, `kMaxRequestsPerCycle`) left unchanged so this run isolates the rop effect, while the new `[LATCFG]` dump + per-transfer TMA `lat_*` breakdown still capture candidates #2/#5/[C] for the next iteration.

---

## Post-run results (rop=100, full trace) — 2026-06-16

Run output: `sim_run_12.8/.../H100_80GB-OnlyKernel10/...warmup_-fe2c19726f6a.o307` (rop=100, `-trace_enabled 1`, full `trace_components`, `-sync_debug_enable 0`).

### rop 211→100 effect: almost none

| Metric | baseline (rop=211) | this run (rop=100) | Δ |
|---|---|---|---|
| `gpu_tot_sim_cycle` | 376,735 | **361,760** | **−14,975 (−4.0%)** |
| `gpu_sim_insn` | 629,197,320 | 629,197,320 | identical (sanity OK) |
| `gpu_ipc` | 1670.13 | 1739.27 | +4.1% |

Cutting rop by more than half (−111 cyc/access) moved total cycles by only **4%**. This **falsifies the #1 hypothesis** that per-access L2 latency (`long_scoreboard`) was the dominant inflation source. The memory latency is almost entirely overlapped/hidden behind another stall.

### Where the cycles actually go — top-level issue-stage breakdown (mutually exclusive, sums to 100%)

Denominator = `total_num_cycles_issue_stage_evaluated` = 178,008,343 (per-subcore issue cycles, summed).

| Class | % | Note |
|---|---|---|
| **no_warps_ready** | **66.64%** | no eligible warp to issue — the bottleneck |
| issuing | 12.71% | actually issued |
| next_stage_not_available | 10.69% | downstream pipe back-pressure |
| no_valid_instruction | 8.96% | (frontend / L0I miss 8.95%) |
| issue_port_busy | 1.01% | |
| **sum** | **100.0%** | ✓ verified exclusive |

### Inside `no_warps_ready` — why warps aren't ready (overlapping counters, NOT exclusive)

These `..._at_least_one_warp_waiting_*` counters answer "in this stalled cycle, was there ≥1 warp blocked on reason X?", so they double-count and do **not** sum to 100. Read as relative intensity. (% normalized to the 118,616,507 no_warps_ready cycles.)

| Wait reason | % of all eval cycles | % of no_warps_ready | maps to (this workload) |
|---|---|---|---|
| **inst_barrier** | 58.47% | **87.7%** | **BAR.SYNC only** (`__syncthreads`) |
| fu_occupied (tensor/WGMMA) | 11.55% | 17.3% | function-unit busy |
| wait_barrier | 7.98% | 12.0% | **DEPBAR** (SB phase wait = TMA mbarrier) |
| stall_count | 4.11% | 6.2% | explicit stall cycles |
| tma_flush | 0.83% | 1.2% | UTMACMDFLUSH |
| yield | 0.68% | 1.0% | YIELD |
| result_queue_full | 0.03% | — | fixed-latency result queue |
| l1c | 0.03% | — | L1 constant |
| **scoreboard (memory)** | **0.00%** | **0.0%** | traditional scoreboard (unused here) |

#### What `inst_barrier` and `wait_barrier` actually count (verified against the SASS)

The increment site ([subcore.cc:591-593](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L591-L593)) sets `inst_barrier` when a warp is blocked on **`programmer_barrier` OR `ldgdepbar`**:
- `programmer_barrier` = `c_warp->waiting()` → **BAR.SYNC / `__syncthreads`** (CTA-wide barrier).
- `ldgdepbar` = `pI->op == LDGDEPBAR_OP && are_ldgsts_pending()` ([subcore.cc:741-742](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L741-L742)) → cp.async (LDGSTS) dependency barrier.

**This is a pure-TMA workload, so the `ldgdepbar` term is always false.** Verified by counting opcodes in the kernel-10 backward SASS (`extra_info/sass/flash_bwd_hdim64_bf16_softcapall_sm90.sm_90a.sass`):

| opcode | count | note |
|---|---|---|
| **LDGDEPBAR** | **0** | ⇒ `inst_barrier`'s ldgdepbar branch never fires |
| LDGSTS | 0 | no cp.async at all |
| BAR.SYNC | 678 | `__syncthreads` |
| DEPBAR | 828 | SB scoreboard phase wait |
| UTMALDG | 480 | TMA load |
| UTMACMDFLUSH | 348 | = `tma_flush` |
| WARPGROUP | 960 | WGMMA |

Therefore:
- **`inst_barrier` (58.47%) ≡ BAR.SYNC (`__syncthreads`) alone** — *not* TMA-related, *not* cp.async.
- **TMA mbarrier waits land in `wait_barrier` (7.98%)** via the **DEPBAR** path ([subcore.cc:754](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L754), `inst->op == DEPBAR_OP` + wait-barrier mask), which is a **separate, correctly-isolated** counter.
- The earlier framing ("inst_barrier ≈ waiting on TMA completion") was **wrong**. The dominant stall is **CTA-barrier serialization**, while TMA-completion waits are only ~8%.

> Caveat: the `tma_axis` grouping ([shader.cc:1448-1450](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L1448-L1450)) lumps `wait_barrier + inst_barrier + tma_flush` together and so is **mislabeled** for this workload — most of its 67% is actually BAR.SYNC, not TMA.

> Counter mechanics: top-level classes are exclusive ([shader.cc:1486-1504](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L1486-L1504)); the `at_least_one_warp_waiting_*` and `tma_axis`/`non_tma_axis` are overlap counts ([shader.cc:1437-1467](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L1437-L1467)) — that is why they exceed 100%.

### Revised conclusions

1. **Memory latency (`scoreboard`) is 0%** of the stall — confirming the rop change was nearly inert. Per-access latency (rop/dram) is **not** the lever; do not pursue it further.
2. **The single dominant bottleneck is `BAR.SYNC` (`__syncthreads`)**: in **87.7%** of no_warps_ready cycles at least one warp is blocked on a CTA-wide instruction barrier. Verified that `inst_barrier` is BAR.SYNC-only here (LDGDEPBAR=0, no cp.async). The warp-specialized pipeline is serializing at `__syncthreads`, leaving the GPU stalled ~2/3 of the time.
3. **TMA-completion waits are NOT the dominant term**: the TMA mbarrier wait shows up as `wait_barrier` (DEPBAR) at only **~8%**. So the gap is driven by how the model serializes the CTA barrier, not by TMA latency or TMA-credit timing. ([B]/[C] are secondary.)

### Next direction (data-driven)
- **Drop** per-access memory-latency tuning (rop/dram) — proven near-zero leverage.
- **Primary target = the `BAR.SYNC` / `__syncthreads` model** in [sm.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc): how long a warp sits at `c_warp->waiting()`, whether all warps in the CTA are forced to rendezvous more strictly/slowly than on HW (e.g. barrier release timing, arrival accounting). This is the 58%/87.7% term.
- **Secondary**: the `wait_barrier`/DEPBAR (TMA mbarrier, ~8%) and `fu_occupied` (WGMMA, ~17%) paths.
- Use the captured trace (WARP_SCHEDULER / SCOREBOARD / TMA) to locate which BAR.SYNC PCs dominate and how long warps sit there.

---

## Summary

1. **Cycle accuracy**: the simulator overestimates cycles by **~2.8–3.2×** vs real H100 (+183% on Elapsed basis). Significant room for improvement.
2. **Root cause is latency/pipeline, not bandwidth**: HW shows DRAM only 14.85% and `No Eligible 67%`. The accuracy gap should be debugged via the **warp-issue stall decomposition**, not DRAM/L2 numbers.
3. **Compare metric = stall distribution split TMA vs non-TMA**: TMA axis (long_scoreboard+barrier+sleeping ≈ 42%) and non-TMA axis (gmma/wait/short_scoreboard/mio/not_selected ≈ 42%) are co-dominant; the sum (7.539) matches `Warp Cycles Per Issued Instruction`.
4. **Occupancy matches closely** (14.49% vs 15.02%) — resource modeling is relatively accurate, so the gap is in timing/latency models, not occupancy.
5. **L2/L1 hit-rate divergence** (sim L2 98.7% vs 82.3%; sim L1 access only 9,912) is a secondary effect of the TMA path bypassing L1 in the model; it is not the primary cycle driver.
6. **Instrumentation already exists**: the per-reason stall counters needed for this comparison are present in `subcore.cc` but commented out — re-enabling them is the fastest path to a side-by-side TMA/non-TMA accuracy comparison.

> Note: NCU reports `Invocations=1` (single-call measurement), and the simulator likewise runs a single launch_uid=1 / trace_kernel_id=10, so the 1:1 comparison is appropriate. IPC is excluded from direct comparison because the sim's gpu_ipc (=1670, aggregated across all cores) and NCU's per-SM IPC (=1.28) use different definitions.
