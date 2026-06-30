# FA3 Kernel 5 (FWD) — Simulator vs. Real H100 Comparison

---

## UPDATE — Opt 5 (L1I eager-promote) simulator result (2026-06)

Sections below were written through the **Opt 2 (BAR fix)** stage. The simulator has since improved
through Opt 3 (MEMBAR scope fix), Opt 4 (deeper L1I stream buffer, sb=4), and Opt 5 (L1I
eager-promote). **Real-HW (NCU) numbers are unchanged**; only the simulator side is updated here.

### Simulator cycle progression (FA3 fwd, trace kernel 5, `FlashAttnFwdSm90`) — HW Elapsed = 67,696

| Stage | sim cycles | Sim / HW | Δ vs prev |
|---|---|---|---|
| Init (rop=211) | 220,024 | 3.25× | — |
| Opt 2 (BAR engine fix) | 162,582 | 2.40× | −26% (vs init) |
| Opt 3 (MEMBAR scope fix) | 158,990 | 2.35× | −2.2% |
| Opt 4 (prefetch, sb=4) | 155,765 | 2.30× | −2.0% |
| **Opt 5 (L1I eager-promote)** | **149,727** | **2.21×** | **−3.4%** |

- Opt 5 run: `.../H100_80GB-OnlyKernel5/...warmup_-63a73d452237.o20` (clean exit; Step-0
  instrumentation counters are timing-neutral, so this is a valid Opt-5 baseline; supersedes the
  earlier `.o18` 150,755).
- eager-promote works (`eager_promote_to_cache=662,658`, `demand_hit_later=252,212`,
  `demand_miss_after_promote=0`, L1I miss rate 0.3574); modest fwd gain (−3.4%).

### Opt 5 issue-stage breakdown (top-level, mutually exclusive)

| Class | Opt 5 % | Note |
|---|---|---|
| issuing | 31.19% | |
| no_warps_ready | 27.32% | dominant stall class |
| next_stage_not_available | 22.46% | downstream pipe back-pressure |
| no_valid_instruction (frontend) | 18.11% | |
| issue_port_busy | 0.92% | |

Inside `no_warps_ready` (overlapping `..._at_least_one_warp_*`, % of all eval cycles):
`non_tma_axis 24.07%`, `fu_occupied 13.50%`, `tma_axis 12.65%`, `wait_barrier 12.58%`,
`stall_count 8.48%`, `yield 1.69%`, `inst_barrier 0.07%`.

### Opt 5 / Step-0 SM-idle decomposition (true SM-level, not per-subcore)

A Step-0 instrumentation run decomposed `sm_all_subcores_idle ≈ 18.33%` (cycles where **no**
subcore on the SM issued) by the dominant blocking reason:

| SM-idle reason | Opt 5 % | note |
|---|---|---|
| **no_valid_other** (ibuffer empty / decode / not stream-buffer) | **11.90%** | coarse bucket; follow-up split shows this is almost entirely `nv_ibuffer_empty` = tail-drain / winding-down warp imbalance |
| **wait_barrier** (mbarrier / DEPBAR) | **9.71%** | #2 |
| no_valid_frontend (incl. `sbwait` 3.99%) | 4.23% | L1I frontend send-bandwidth idea deferred / parked (only ~4% recoverable at true SM level) |
| stall_count | 3.76% | |
| fu_occupied (tensor 0.67%) | 2.10% | WGMMA fix deferred (≤0.7% recoverable) |
| next_stage | 1.78% | |

> Follow-up split instrumentation resolved the old `no_valid_other` bucket: almost all of it is
> `nv_ibuffer_empty`, while `nv_ibuf_fetch_inflight = 0` and `nv_ibuf_fetch_not_issued ~= 0`, so the
> dominant residual is best interpreted as **tail-drain / winding-down warp imbalance**, not an
> actionable frontend fetch bottleneck. Both the WGMMA idea and the L1I frontend send-bandwidth idea
> therefore remain **deferred / parked**.

---

## UPDATE — after the OP_BAR named/counted-barrier fix (2026-06)

The numbers in the original sections below ("Simulator" = **220,024 cycles**) were produced
**before** the named/counted-barrier engine fix. The barrier model was over-serializing the
warp-specialized FA3 pipeline (`inst_barrier` ≈ 56% of issue-stage stall). After fixing the
`OP_BAR` decode + the barrier engine (see `.plan/BAR_OP_H100.md`), the same OnlyKernel5 run
completes cleanly (`*** exit detected ***`, no assert / deadlock) and the accuracy improves
substantially.

### Before vs. After (FA3 fwd, trace kernel 5, `FlashAttnFwdSm90`)

| Metric | Real H100 (NCU) | Before (sim) | After fix (sim) | Note |
|---|---|---|---|---|
| **Elapsed / sim cycles** | 67,696 | **220,024** (3.25×) | **162,582** (2.40×) | **−26% cycles; 3.25× → 2.40×** |
| gpu_sim_insn | — | 455,639,416 | 455,648,438 | ≈ identical (decode unchanged) |
| gpu_occupancy | 20.14% | 19.85% | 19.30% | unchanged (resource model intact) |
| `inst_barrier` stall share | barrier ≈ 10.9% | **56.09%** | **9.09%** | now matches HW barrier share |
| `wait_barrier` (mbarrier) share | — | 6.64% | 8.07% | similar |
| **TMA-axis stall share** | ≈ 23.8% | **62.73%** | **17.16%** | inverted attribution removed |
| non-TMA-axis stall share | ≈ 57.8% | 17.80% | 17.34% | still under-attributed |
| `issuing` share | 13.9% | 14.56% | 21.17% | closer to HW |
| `no_valid_instruction` (frontend) | ≈ 4.5% | 9.52% | 39.12% | **new dominant bucket at this stage**; later Step-0 work showed only a small true SM-level frontend share |
| shared-mem bank conflicts | 281 | 38,016 | 38,016 | untouched (separate model bug) |
| run termination | — | abort (deadlock / teardown assert) | `*** exit detected ***` (12h38m) | **fixed** |

> Caveat on the percentage rows: the issue-stage normalization denominator
> (`total_num_cycles_issue_stage_evaluated`) changed between runs (110.4M → 75.9M), so the
> percentages are **not** on an identical base — read them as a *trend* (barrier share
> collapsing), not an exact like-for-like delta. The cycle counts (220,024 → 162,582) and the
> termination status are directly comparable.

### What the fix changed (engine, not decode)

- The `OP_BAR` decode and the blocking-vs-arrive rule were already correct (see
  `.plan/BAR_OP_H100.md` FINAL rule). The residual failure was a **CTA-teardown leak**:
  in a warp-specialized kernel the producer/consumer warpgroups exit at different times, and
  a counted/named barrier whose closing credit would have come from an already-exited warp
  was never released, tripping `shader.cc:4252 deallocate_barrier` assert.
- `barrier_set_t::warp_exit` now removes the exiting warp from every per-id participant /
  arrive-credit / sync-credit set, and a new helper `release_satisfiable_barriers()`
  generalizes the legacy full-CTA release (`at_barrier == active`) to counted/named barriers
  so they drain once the remaining active participants have all arrived.
- BARDBG verification on this run: all 40 CTA teardowns report `leaked_ids=0` (was 6),
  8 `[BARDBG][exit-release]` events (ids 1,4,5,8,9,10,11), and `0` `exit-clear was_parked=1`
  (no warp wrongly parked at a blocking SYNC ⇒ decode classification confirmed sound).

### Residual gap (unchanged conclusion, smaller magnitude)

With the spurious barrier serialization removed, the remaining 2.40× over-estimation is no
longer barrier-dominated. The new largest bucket is the **frontend**
(`no_valid_instruction = 39.1%`, almost entirely `head_invalid_waiting_frontend` = I-cache
prefetch / stream-buffer wait), followed by the still-under-modeled non-TMA scheduler axis.
The TMA-emission-serialization / mbarrier-credit-timing candidates from the analysis below
remain the next levers; the ROP-latency conclusion is unchanged.

---

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

## NCU ↔ GCOM-sim Metric Mapping

How each measured NCU metric maps to a GPGPU-Sim (GCOM, this remodeled sim) counter. "Direct" =
an emitted sim counter; "Derived" = computed from sim counters (formula given); "No direct sim
stat" = NCU-only (sim is cycle-based, no NCU-style throughput-% / wall-clock). Sim counter names
are the exact strings from the Opt 5 run `.o20`.

| NCU metric | GCOM-sim counter / derivation | Mapping note |
|---|---|---|
| Elapsed Cycles | `gpu_tot_sim_cycle` (= `gpu_sim_cycle`) | Direct; the primary cycle comparison. |
| SM Active Cycles | — (no separate active/elapsed split) | Sim emits a single cycle count; compare both NCU rows vs `gpu_tot_sim_cycle`. |
| Duration (µs) | — (no direct sim stat) | Sim is cycle-based, no wall clock. |
| SM Frequency (GHz) | config `-gpgpu_clock_domains` core clock (1.80) | Configured, not measured; cycle-vs-cycle comparison is clock-independent. |
| DRAM Frequency (GHz) | config DRAM clock | Configured value. |
| Compute (SM) Throughput % | — (no direct sim stat) | No NCU-style %-of-peak in sim; closest proxy = issue-stage `issuing` share. |
| Memory Throughput % / GB/s | — (no direct sim stat) | Proxy = `dram bw_util` / `dram_eff` (both very low in sim). |
| DRAM Throughput % | Derived proxy: `dram bw_util` (~0.005) | No %-of-peak; DRAM is near-idle in sim too. |
| L1/TEX Hit Rate % | `1 − L1D_total_cache_miss_rate` | Sim: TMA bypasses L1D, so L1D = 6,144 acc / 100% miss → not comparable. |
| L2 Hit Rate % | `1 − L2_total_cache_miss_rate` | Direct-derived; sim overestimates (98.96% vs 69.58%). |
| Achieved Occupancy % | `gpu_occupancy` | Direct (per-SM achieved). |
| Theoretical Occupancy % | reg/CTA-limited max (1 block/SM ⇒ 25%) | Static; sim matches (CTA=132, SM=132). |
| Achieved Active Warps / SM | Derived: `gpu_occupancy × 64` (H100 max warps/SM) | No direct counter. |
| Executed IPC (active) | Derived per-SM: `gpu_sim_insn /(gpu_tot_sim_cycle × n_SM)` | `gpu_ipc` (3043) is **aggregated over all cores** — different definition, not 1:1. |
| Issued IPC (active) | same as above | sim has no separate executed/issued IPC. |
| Issue Slots Busy % | issue-stage `issuing` share (Opt 5 = 31.19%) | Closest analog ("fraction of issue cycles that issued"). |
| Registers / Thread | config / launch reg count (128) | Static; matches. |
| Waves / SM | `gpu_tot_issued_cta / n_SM` (132/132 = 1.00) | Direct-derived; matches. |
| Executed Instructions | `gpu_sim_insn` (455,648,438) | **Different granularity**: NCU counts SASS/warp-insts; sim counts thread-insts → not 1:1. |

## NCU (Measured) Detailed Metrics — vs Opt 5 simulator (side by side)

> The "Real H100 (NCU)" column is **unchanged**. The "GCOM Sim (Opt 5)" column is from the
> clean-exit, timing-neutral run `.../H100_80GB-OnlyKernel5/...warmup_-63a73d452237.o20`
> (`gpu_tot_sim_cycle = 149,727`). See the mapping table above for how each sim value is obtained;
> `— (no direct sim stat)` marks NCU-only metrics.

| Metric | Unit | Real H100 (NCU) | GCOM Sim (Opt 5, `.o20`) | Sim/HW or note |
|---|---|---|---|---|
| Elapsed Cycles | cycle | 67,696 | 149,727 | **2.21×** (primary) |
| SM Active Cycles | cycle | 61,147 | — (no split) | compare vs 149,727 → 2.45× |
| Duration | µs | 47.55 | — | sim is cycle-based |
| SM Frequency | GHz | 1.42 | 1.80 (config) | configured, not measured |
| DRAM Frequency | GHz | 2.62 | (config) | configured |
| Compute (SM) Throughput | % | 43.04 | — (no direct sim stat) | proxy: `issuing` 31.19% |
| Memory Throughput | % | 28.84 | — (no direct sim stat) | proxy: dram bw_util ~0.005 |
| Memory Throughput | GB/s | 404.88 | — (no direct sim stat) | — |
| DRAM Throughput | % | 12.09 | — (no direct sim stat) | dram near-idle in sim too |
| L1/TEX Hit Rate | % | 55.60 | L1 bypassed (6,144 acc, 100% miss) | not comparable (TMA→L2 direct) |
| L2 Hit Rate | % | 69.58 | 98.96 (1 − miss 0.0104) | sim **overestimates** |
| Achieved Occupancy | % | 20.14 | 12.04 (`gpu_occupancy`) | see note ‡ |
| Theoretical Occupancy | % | 25.00 | 25.00 (reg-limited, 1 blk/SM) | match |
| Achieved Active Warps / SM | warp | 12.89 | ~7.7 (= 12.04% × 64) | derived; tracks occupancy |
| Executed IPC (active) | inst/cycle | 1.79 | not 1:1 (see note †) | `gpu_ipc` aggregated = 3043 |
| Issued IPC (active) | inst/cycle | 1.80 | not 1:1 (see note †) | — |
| Issue Slots Busy | % | 45.03 | 31.19 (`issuing` share) | sim lower |
| Registers / Thread | reg | 128 | 128 (config) | match |
| Waves / SM | — | 1.00 | 1.00 (132 CTA / 132 SM) | match |
| Executed Instructions | inst | 14,482,551 | `gpu_sim_insn` = 455,648,438 | different granularity (note §) |

> ‡ The `.o20` run reports `gpu_occupancy = 12.04%`. An earlier (Opt 1/2) run reported 19.85%;
> the difference is a per-run occupancy-accounting artifact, not a model change — the latest Opt 5
> value (12.04%) is shown here.
> † NCU IPC is **per-SM** (inst/cycle); sim `gpu_ipc` (3043) is **aggregated over all cores**, a
> different definition. The comparable per-SM figure must be derived as
> `gpu_sim_insn /(gpu_tot_sim_cycle × 132)`, but note `gpu_sim_insn` is a thread-instruction count
> (see §), so even that is not a clean 1:1.
> § NCU "Executed Instructions" counts SASS/warp-level instructions; sim `gpu_sim_insn` counts
> thread-level instructions, so the two are on different granularities and are not directly equal.

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

### Highest-leverage change at that stage
- At the time of this section, with `rop_latency=100` already in effect and the gap still large,
  the leading hypothesis was **TMA emission serialization (`kMaxRequestsPerCycle`)** plus
  **mbarrier-credit timing**, because they appeared to drive the dominant
  `inst_barrier`/`wait_barrier` buckets (the 62.7% TMA axis). Later work superseded this specific
  prioritization: the barrier and MEMBAR fixes landed first, and the eventually investigated L1I
  frontend path was also parked after Step-0 showed it was not a major true-SM bottleneck.
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
