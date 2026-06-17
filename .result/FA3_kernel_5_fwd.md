# FA3 Kernel 5 (FWD) — Simulator vs. Real H100 Comparison

## Target Information

- **Workload**: `flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24`
- **Trace kernel**: `kernel-5.trace` = `FlashAttnFwdSm90` (causal **forward**, bf16, tile 192×128×64)
  - NOTE: although the workload name says `bwd`, **the 5th kernel in the trace is the FA3 forward (FwdSm90) kernel**, not a backward kernel. (Backward = kernels 9/10/11.)
- **Grid / Block**: (132, 1, 1) / (512, 1, 1)
- **Registers/Thread**: 128
- **launch_uid**: 1 (single launch)

### Source Files
- Measured (H100, NCU): `/home/jihyun/project/accorde/ncu_report/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.csv` (+ `.ncu-rep` for the raw stall page)
- Simulator output: `sim_run_12.8/.../H100_80GB-OnlyKernel5/...warmup_-63a73d452237.o3` (stats) + `.e3` (SYNCDBG/TMA event summary)
  - OnlyKernel5 configuration. **The actual run used `-gpgpu_l2_rop_latency 100`** (the reduced value, confirmed by the echoed config at `.o3` line 319 and the `[LATCFG]` dump). The stale `gpgpusim.config` copy in the run directory shows `211`, but the running simulator parsed `100`.

---

## Key Metric Comparison

| Metric | Real H100 (NCU) | Simulator | Sim / HW | Note |
|---|---|---|---|---|
| **Elapsed Cycles** | 67,696 | 220,024 | **3.25×** | Primary cycle comparison |
| SM Active Cycles | 61,147 | 220,024 (gpu_sim_cycle) | 3.60× | sim has no separate split |
| Duration | 47.55 µs | — | — | sim is cycle-based |
| Achieved Occupancy | 20.14 % | 19.85 % | 0.99× | **Very close** |
| Achieved Active Warps / SM | 12.89 | — | — | Consistent via occupancy |
| Theoretical Occupancy | 25.00 % | — | — | reg-limited (1 block/SM) |
| L2 Hit Rate | 69.58 % | ~99.0 % (miss 1.00%) | — | sim overestimates |
| L1/TEX Hit Rate | 55.60 % | — (L1 bypassed by TMA, 6,144 acc all miss) | — | sim L1 access very low |
| SM Count | 132 | 132 (n_clusters) | 1.0× | Match |
| Core Clock | 1.42 GHz | 1.80 GHz | — | sim configured value |
| Waves / SM | 1.00 | — | — | one block per SM |

### Accuracy (based on Elapsed Cycles)
- Ratio: 220,024 / 67,696 = **~3.25×**
- Error: **+225.0%** (simulator overestimates)
- Based on SM Active Cycles: ~**3.60×**

---

## Simulator Detailed Statistics

| Item | Value |
|---|---|
| gpu_sim_cycle | 220,024 |
| gpu_sim_insn | 455,639,416 |
| gpu_ipc (aggregated, all cores) | 2,070.86 |
| gpu_occupancy | 19.85 % |
| gpu_tot_sms_occupancy | 95.71 % |
| L1D total accesses | 6,144 (miss rate 1.000) |
| L1C (const) accesses | 52,956 (miss rate 0.1122) |
| L2 total accesses | 4,355,547 |
| L2 miss rate | 0.0100 |
| total dram reads | 36,229 |
| total dram writes | 0 |
| DRAM bw_util (per-channel avg) | ~0.0046–0.0048 |
| DRAM efficiency (dram_eff) | ~0.27–0.29 |
| gpu_stall_dramfull | 8,096 |
| shared-mem bank conflicts (gpgpu_n_shmem_bkconflict) | 38,016 |
| averagemflatency / maxmflatency | 1,248 / 7,028 |

> Memory access breakdown (Total_core_cache_stats_breakdown):
> - GLOBAL_ACC_R: handled via L2 (L1 bypassed); L2 GLOBAL_ACC_R HIT 3,914,001 / MISS 8,705 / HIT_RESERVED 45,046
> - GLOBAL_ACC_W: MISS 1,536 / SECTOR_MISS 4,608
> - CONST_ACC_R: HIT 47,412 / MISS 6,864 / MSHR_HIT 4,884
> - gpgpu_n_mem_read_global = 3,993,864 / gpgpu_n_mem_write_global = 184,753

### TMA activity (from `.e3` summary)
- Per SM ~46–54 TMA commands, ~39–47 transfers completed, ~29k–35k sector requests, ~0.94–1.14 MB transferred per SM.
- Phase2 site classification (all SMs): 16 unique TMA sites; UTMALDG 9 sites / 5,112 commands, UTMAPF 1 site / 264, UTMASTG 1 site / 264.

---

## NCU (Measured) Detailed Metrics

| Metric | Unit | Value |
|---|---|---|
| Elapsed Cycles | cycle | 67,696 |
| SM Active Cycles | cycle | 61,147 |
| Duration | µs | 47.55 |
| SM Frequency | GHz | 1.42 |
| DRAM Frequency | GHz | 2.62 |
| Compute (SM) Throughput | % | 43.04 |
| Memory Throughput | % | 28.84 |
| Memory Throughput | GB/s | 404.88 |
| DRAM Throughput | % | 12.09 |
| L1/TEX Hit Rate | % | 55.60 |
| L2 Hit Rate | % | 69.58 |
| Achieved Occupancy | % | 20.14 |
| Theoretical Occupancy | % | 25.00 |
| Achieved Active Warps / SM | warp | 12.89 |
| Executed IPC (active) | inst/cycle | 1.79 |
| Issued IPC (active) | inst/cycle | 1.80 |
| Issue Slots Busy | % | 45.03 |
| Registers / Thread | reg | 128 |
| Waves / SM | — | 1.00 |
| Executed Instructions | inst | 14,482,551 |

---

## Accuracy Strategy — Warp-Issue Stall Breakdown (TMA vs non-TMA)

As with the kernel-10 (bwd) analysis, the right accuracy metric for this warp-specialized FA3
kernel is the **per-reason warp-issue stall distribution**, split into a **TMA (producer) axis**
and a **non-TMA (consumer / compute / scheduling) axis** — not raw cycle/hit-rate numbers.

### HW says: NOT memory-bandwidth bound — latency / pipeline-dependency bound
- `Compute (SM) Throughput 43.04%`, `Memory Throughput 28.84%`, `DRAM Throughput 12.09%`, `Mem Pipes Busy 22.68%` → DRAM is **not** the bottleneck.
- `Issue Slots Busy 45.03%` / Scheduler `No Eligible 54.26%` → about half of all cycles issue nothing.
- `Warp Cycles Per Issued Instruction = 7.16` → each issued instruction is paid for with 7.16 warp-cycles of stall. **Decomposing this 7.16 is the accuracy target.**

### Ground truth: HW warp-issue stall decomposition (kernel 5, FlashAttnFwdSm90)

Source metrics: `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` from the
`.ncu-rep` raw page. The reasons sum to **7.177**, matching `Warp Cycles Per Issued Instruction = 7.16`.

| Stall reason (NCU) | Value | Share | Axis |
|---|---|---|---|
| wait | 1.363 | **19.0%** | non-TMA — fixed-latency dependency |
| selected (actually issued) | 1.000 | 13.9% | (issued, not a stall) |
| not_selected | 0.820 | **11.4%** | non-TMA — scheduler contention |
| dispatch_stall | 0.787 | **11.0%** | non-TMA — dispatch |
| barrier | 0.782 | **10.9%** | **TMA** — mbarrier / named-barrier (producer↔consumer sync) |
| long_scoreboard | 0.700 | **9.8%** | **TMA** — global-memory / TMA data-arrival latency |
| mio_throttle | 0.457 | 6.4% | non-TMA — shared/LSU (MIO) pipe throttle |
| short_scoreboard | 0.330 | 4.6% | non-TMA — short-latency (shared/MUFU) wait |
| math_pipe_throttle | 0.229 | 3.2% | non-TMA — math pipe throttle |
| sleeping | 0.223 | 3.1% | TMA — warp-specialization producer idle |
| no_instruction | 0.169 | 2.4% | frontend — I-cache / fetch |
| imc_miss | 0.156 | 2.2% | frontend — constant/immediate cache |
| gmma | 0.097 | 1.4% | non-TMA — WGMMA (tensor) pipe wait |
| branch_resolving | 0.061 | 0.9% | non-TMA — branch |
| misc / drain | ~0.003 | 0.0% | misc |
| membar / lg_throttle / tex_throttle | 0.000 | 0.0% | n/a |
| **TOTAL** | **7.177** | 100% | |

**TMA axis (long_scoreboard + barrier + sleeping) ≈ 23.8%** of warp-cycles;
**non-TMA axis (wait + not_selected + dispatch + mio + short_scoreboard + math + gmma + branch) ≈ 57.8%**;
issued 13.9%; frontend (no_instruction + imc_miss) ≈ 4.5%.

> Contrast with the kernel-10 (bwd) kernel, where the TMA axis (~42%) and non-TMA axis (~42%)
> were co-dominant. For this **forward** kernel the **non-TMA axis dominates** (~58%): the
> three largest single reasons are `wait` (19.0%), `not_selected` (11.4%) and `dispatch_stall`
> (11.0%) — all scheduler / fixed-latency-dependency effects rather than memory/TMA latency.
> `barrier` + `long_scoreboard` together are only ~20.7%.

### HW pipe utilization (anchors the non-TMA axis)

| Pipe | % of peak (active) | Note |
|---|---|---|
| sm__inst_executed_pipe_xu (MUFU/transcendental) | 47.75% | exp in softmax fwd — heaviest pipe |
| sm__pipe_tensor_cycles_active | 46.13% | WGMMA+HMMA combined — heavy |
| sm__inst_executed_pipe_alu | 27.28% | |
| sm__inst_executed_pipe_fma | 16.88% | |
| sm__inst_executed_pipe_tensor_op_gmma | 4.32% | WGMMA |
| sm__inst_executed_pipe_lsu | 2.31% | very low (TMA does the bulk loads) |
| sm__inst_executed_pipe_tensor_op_hmma | 2.16% | HMMA |
| sm__inst_executed_pipe_tma | 0.08% | TMA insts are rare |
| sm__pipe_tma_cycles_active | 0.15% | TMA unit essentially idle on HW |

Shared-memory store bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st`) = **281** on HW
(vs sim `gpgpu_n_shmem_bkconflict = 38,016` — sim massively over-counts shared bank conflicts here).

> The fwd kernel is **MUFU/transcendental + tensor heavy** (xu 47.75%, tensor 46.13%), unlike the bwd
> kernel where xu was only 21%. This explains why the HW non-TMA stall axis (math/dispatch/wait/not_selected)
> dominates for the forward kernel.

---

## Sim ↔ HW: emitted stall percentages (kernel 5)

The re-enabled `no_warps_ready` sub-breakdown in `subcore.cc` produced the following
(normalized by `total_num_cycles_issue_stage_evaluated = 110,380,490`):

| Sim emitted metric | Value | HW reason it tracks | Axis |
|---|---|---|---|
| `..._stall_tma_axis` (wait_barrier + inst_barrier + tma_flush) | **62.73 %** | long_scoreboard + barrier (HW ≈ 20.7%) | **TMA** |
| `..._waiting_wait_barrier` (mbarrier) | 6.64 % | barrier (mbarrier) | TMA |
| `..._waiting_inst_barrier` | **56.09 %** | named barrier / ldgdepbar | TMA |
| `..._waiting_tma_flush` | 0.00 % | bulk-store drain | TMA |
| `..._stall_non_tma_axis` (fu_occupied + stall_count + l1c + scoreboard + result_queue + yield) | **17.80 %** | gmma + wait + short_scoreboard + mio (HW ≈ 58%) | **non-TMA** |
| `..._with_fu_occupied` | 11.83 % | gmma / math_pipe_throttle | non-TMA |
| `..._waiting_stall_count` | 5.00 % | wait | non-TMA |
| `..._waiting_yield` | 0.92 % | scheduler yield | non-TMA |
| `..._waiting_result_queue_full` | 0.05 % | result-queue backpressure | non-TMA |
| `..._waiting_l1c` | 0.0005 % | short_scoreboard | non-TMA |
| `..._waiting_scoreboard` | 0.00 % | short/long_scoreboard | non-TMA |
| `..._issuing` | 14.56 % | selected (HW 13.9%) | (issued) |
| `..._no_warps_ready` (total) | 64.02 % | (all stall reasons aggregated) | — |
| `..._stall_no_valid_instruction` | 9.52 % | no_instruction + imc_miss (frontend, HW ≈ 4.5%) | frontend |
| `..._stall_next_stage_not_available` | 11.40 % | (pipeline backpressure) | — |

### Diagnosis — the sim's TMA / non-TMA split is **inverted** vs HW

| Axis | HW share (of 7.177) | Sim emitted % | Direction |
|---|---|---|---|
| TMA (barrier + long_scoreboard + sleeping) | **~23.8 %** | **62.73 %** | sim **massively over-attributes to TMA** |
| non-TMA (wait/not_selected/dispatch/mio/...) | **~57.8 %** | **17.80 %** | sim **under-attributes to non-TMA** |
| issued | 13.9 % | 14.56 % | **matches well** |
| frontend | ~4.5 % | 9.52 % | sim ~2× high |

- The sim's biggest bucket by far is **`inst_barrier` = 56.09 %**, i.e. warps blocked at a named
  barrier / `ldgdepbar`. On HW the entire TMA axis (barrier + long_scoreboard + sleeping) is only
  ~24%, and `barrier` alone is ~10.9%. **The model spends ~3× too long blocked on barriers.**
- Because the consumer's mbarrier credit is only granted when the TMA transfer completes
  (see kernel-10 note [B]), the inflated barrier wait is a **downstream effect of over-modeled
  global/TMA round-trip latency** plus per-transfer TMA emission/serialization. Note this run
  **already used the reduced `rop_latency=100`** (not 211), yet `inst_barrier` is still 56% and the
  kernel is 3.25× over — so for the forward kernel the rop reduction alone is **insufficient**, and
  the residual TMA-axis inflation points to the **TMA emission serialization** (`kMaxRequestsPerCycle`)
  and/or the mbarrier credit timing rather than the ROP delay.
- The forward kernel's *true* bottleneck is compute (xu/tensor), not memory, so any remaining
  memory-latency / barrier over-estimation shows up as a large relative inflation.
- The sim's `with_fu_occupied = 11.83%` is the closest non-TMA analog to HW's tensor/math waits,
  but HW's dominant non-TMA reasons (`wait` 19%, `not_selected` 11.4%, `dispatch_stall` 11%) are
  scheduler/fixed-latency effects that the sim does **not** reproduce in proportion.

---

## Pre-run cycle-reduction analysis (code + NCU only)

### The decisive contradiction (same shape as bwd kernel 10)
- Sim **L2 hit rate = 99.0%** is *higher* than HW **69.58%**, yet sim cycles are **3.25×** HW.
- A higher hit rate should make sim *faster*; it is slower → the inflation is **per-access modeled
  latency / non-overlap**, **not** bandwidth or miss count. HW DRAM is only 12.09% busy and sim L2
  miss is 1.0%, so DRAM/L2-miss tuning cannot explain the gap.
- **This run already used the reduced `rop_latency=100`**, so the residual 3.25× gap is **not**
  attributable to the ROP delay — the over-modeling is elsewhere (TMA emission serialization /
  mbarrier credit timing / fixed-latency dependencies).

### Ranked over-estimation candidates

| # | Candidate | Evidence | Verdict |
|---|---|---|---|
| 1 | L2 ROP delay `rop_latency` | **Already reduced to 100 for this run** (see `[LATCFG]` / `.o3` L319). Gap persists ⇒ no longer the driver for this kernel. | **already applied; not the residual driver** |
| **2** | **TMA emission serialization (`kMaxRequestsPerCycle=2`)** | ~29k–35k sector requests/SM emitted ≤2/cycle ⇒ structural serialization HW does not have (HW `sm__pipe_tma_cycles_active=0.15%`). Most likely cause of the residual `inst_barrier=56%` inflation. | **investigate / raise (highest leverage now)** |
| **3** | **mbarrier credit timing** (credit granted only at TMA completion) | inflates consumer `barrier` waits as a downstream effect; `inst_barrier`=56% vs HW barrier ~10.9% | investigate via per-transfer `lat_*` log |
| 4 | Clock 1800 MHz vs NCU 1420 MHz | cycle-vs-cycle comparison is clock-independent for fixed-cycle latencies | not the driver |
| 5 | WGMMA/tensor latency (`tensor_latency=32`) | HW `gmma` stall only 1.4%, tensor pipe 46% busy is *expected* for fwd | **DO NOT touch** |
| 6 | shared bank-conflict model | sim 38,016 vs HW 281 — sim grossly over-counts; feeds `mio_throttle`/`short_scoreboard` non-TMA cost | investigate (small absolute vs barrier inflation) |

### Highest-leverage change
- Since `rop_latency=100` is already in effect and the gap persists, the next target is the
  **TMA emission serialization (`kMaxRequestsPerCycle`)** and the **mbarrier-credit timing**, which
  together drive the dominant `inst_barrier`/`wait_barrier` buckets (the 62.7% TMA axis). Use the
  per-transfer TMA `lat_issue` / `lat_mem` log to decide whether emission serialization or memory
  round-trip is the larger contributor before changing a knob.
- Unlike memory-bound kernels, the forward kernel's true bottleneck is the **xu (MUFU) + tensor**
  compute pipes; once the spurious barrier waits are removed, accuracy will hinge on the
  fixed-latency / scheduler (`wait`, `not_selected`, `dispatch`) modeling that the sim currently
  under-represents.

---

## Summary

1. **Identity**: the "5th kernel" is the FA3 **forward** kernel `FlashAttnFwdSm90` (grid 132, block 512,
   128 reg/thread), even though the workload is labeled `bwd`. Backward kernels are 9/10/11.
2. **Cycle accuracy**: the simulator overestimates cycles by **~3.25×** vs real H100 (+225% on the
   Elapsed basis; ~3.60× on SM-active). Larger error than the bwd kernel-10 (~2.83×).
3. **Root cause is latency/pipeline, not bandwidth**: HW DRAM only 12.09% busy, `No Eligible 54.26%`,
   L2 hit *lower* on HW than sim. The gap is modeled per-access latency / non-overlap.
4. **Occupancy matches closely** (19.85% vs 20.14%) — resource/occupancy modeling is accurate; the
   gap is purely in the timing/latency models.
5. **Stall-axis split is inverted**: HW is non-TMA-dominated (~58% non-TMA vs ~24% TMA, driven by
   `wait`/`not_selected`/`dispatch` + heavy xu/tensor pipes), but the sim attributes **62.7% to the
   TMA axis** (almost all `inst_barrier`=56%) and only 17.8% non-TMA. The model spends ~3× too long
   blocked on barriers. **This run already used the reduced `rop_latency=100`**, so the residual
   inflation is driven by TMA emission serialization / mbarrier-credit timing, not the ROP delay.
6. **Secondary modeling errors**: sim L1 is bypassed by TMA (expected Hopper behavior), but sim
   shared-mem bank conflicts (38,016) vastly exceed HW (281); the frontend (`no_valid_instruction`
   9.5%) is ~2× the HW frontend share (~4.5%).

> Note: NCU reports a single FwdSm90 invocation and the sim runs a single launch_uid=1, so the 1:1
> comparison is appropriate. IPC is excluded from direct comparison: sim `gpu_ipc=2070.86` is
> aggregated across all cores while NCU `Executed IPC (active)=1.79` is per-SM.
