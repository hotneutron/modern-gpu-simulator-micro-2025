# FA3 — HW (NCU) vs Simulator: SFU / critical-path / SM-idle analysis (HW_VS_SIM_SFU.md, 2026-07-27)

Goal: reduce the FA3 sim-vs-HW cycle gap (fwd **1.56×**, bwd **1.35×** after Opt 10). This doc is the
**fresh, definition-checked baseline** for that work. It supersedes the tangle of findings that had
accreted onto `.plan/WARP_STAGGER_LOCKSTEP.md` (now CLOSED): the warp-stagger / lockstep / trace-`stall_count`
leads were all either refuted or shown to be symptoms, and the residual is now positively localized to
**issue-density / pipe-overlap** — with every per-pipe COST confirmed HW-faithful-or-lighter.

Target HW: H100. Kernels: FA3 fwd = kernel 5 (`FlashAttnFwdSm90`), bwd = kernel 10 (`FlashAttnBwdSm90`).
NCU reports: `nv_reports/h100/...full_rpt.ncu-rep` (ratios only) and
`nv_reports/h100/fa3_pipe_rawcounts.ncu-rep` (raw `.sum` + `.peak_sustained`, produced by
`simulator-remodeled/rerun_ncu_pipe_rawcounts.sh`).

## Executive Summary (state as of 2026-07-27, read this first)

**Problem confirmed.** fwd is 1.56× HW (105k vs 68k). Decomposition path: `cycle = work(1.10×) ×
issue-density(1.41×)`. The gap is issue-density, i.e. the sim does not keep its warps issuing as densely
as HW.

**What has been RULED OUT (all HW-faithful or refuted):**
- per-pipe COSTS: tensor busy-cyc/op sim 29.5 vs HW 42.7 (sim UNDER-models, `peak_sustained=4` confirmed);
  SFU II=8 and total latency 16 both HW-faithful; op-counts (tensor/MUFU) bit-match HW. §1c, §6, §7.
- issue-path serialization (1-deep ISSUE_CONTROL latch): already fixed by Opt 10, head-of-line ~0.06%. §4.
- trace `stall_count`: `>>=1` decay UNDER-applies vs HW → cannot inflate cycles. Arch TODO-2.
- warp-stagger via launch-offset (E1): mbarrier re-synchronizes each tile → ~0. (WARP_STAGGER closed.)
- "sim 64% vs HW 90% SM-active": apples-vs-oranges (issue-rate vs resident-rate). §0.4.

**Where the gap actually is (critical-path / SM-idle).** Per tile the compute WORK is identical sim/HW
(~730 cyc if overlapped) but sim wall-clock/tile ≈ 3300 vs HW ≈ 1700 — the whole gap is **wait/idle**.
sim SM is **fully-idle 31.7%** (HW ~10%). Each warp's own critical-path wait is SHORT (sim ≤ HW), but the
3 in-phase consumer warps/SMSP hit those waits SIMULTANEOUSLY → SM stalls as a whole = **failed
latency-hiding**. §8. This unifies the "SFU contention" and "de-phasing" framings: the lever is filling the
31.7% SM-idle by de-phasing warps so one runs during another's dependency wait — NOT emptying an SFU queue
(SFU is only 28%-floored).

Per-tile wall-clock decomposition (fwd, per SMSP) — the clearest picture of the gap:

| per tile (fwd, SMSP) | sim | HW |
|---|---|---|
| tensor work (16.5 op; sim 29.5 / HW 42.7 busy-cyc/op) | 487 | 705 |
| MUFU work (91.2 op; II 8) | 730 | 730† |
| **compute work (if overlapped = max of the two)** | **~730** | **~730** |
| **wall-clock / tile** | **~3300** | **~1700** |
| **pure wait/idle / tile** | **~2570** | **~970** |

Note the tensor work is NOT equal (sim 487 < HW 705 — HW holds each WGMMA in the pipe longer, §6); it's the
MUFU term (730, II=8 HW-faithful) that DOMINATES, so `compute = max(tensor, MUFU) ≈ 730` on BOTH — equal by
coincidence, not because tensor matches. The key point stands and is stronger: **HW runs MORE tensor work
per op yet finishes each tile in ~1700, while sim runs LESS and takes ~3300** ⇒ the ~1600 gap is entirely
**wait/idle** (sim 2570 vs HW 970), never compute.

† **MUFU work "730" is an II=8 THROUGHPUT-FLOOR, not an HW measurement.** op-count is identical sim=HW
(work-invariant); II=8 is HW-faithful (peak_sustained 0.5/cyc/SM = 8 cyc/MUFU/SMSP). But unlike tensor
(which has an HW busy-cycle counter → 42.7/op measured), the **SFU has NO HW pipe-busy counter** (§1c), so
HW's real per-op SFU occupancy cannot be measured — only the II floor. HW's actual MUFU issue rate is 48%
of peak, i.e. SFU is far from saturated. So "730=730" means "same op-count × same II floor", NOT a measured
equality.

**How much can be recovered (the number the user wants).** SM-idle 34.5% splits into:
| bucket | % of elapsed | recoverable by de-phasing? |
|---|---|---|
| rest (clearly recoverable) | **~9.9%** | ✅ yes (this is the "10%") |
| wait_barrier_only | 19.1% | ⏳ PENDING probe: WGMMA=recoverable by de-phasing / TMA=NOT by de-phasing (producer/mem axis) |
| ibuffer_empty + drained | ~24.7% | ❌ trace-drain straggler tail (floor) |

So the total recoverable = **rest 9.9% + (wait_barrier 19.1% × WGMMA-fraction)**. Pending probe scenarios:
WGMMA 0% → recover 9.9% (1.57×→1.41×); 50% → 19.5% (→1.26×); 100% → 29% (→1.11×). Upper-bound (idle-fill
isn't 100% cycle-for-cycle).

**Next measurement (probe implemented, awaiting 12h run):** `-overlap_instrument_enable` now also emits
`warpcyc_wb_wait_{tensor,tma,other}` — splits the 19.1% wait_barrier into WGMMA-result (recoverable) vs
TMA (producer floor). That decides the WGMMA-fraction above and thus the final recoverable cycle count.

**Why WGMMA is recoverable but TMA is not (by de-phasing):** the distinction is WHO produces the awaited
data. A `wgmma.wait_group` wait is for the warp's OWN tensor result — produced INSIDE the SM; while warp A
waits, a de-phased warp B can run its independent work, so the wait is hidden ⇒ recoverable. A TMA-tile
wait is for the producer to land the next tile from GMEM→SMEM — produced OUTSIDE the SM; if the tile isn't
there yet, NO warp on the SM can proceed on that data, so re-phasing SM-internal warps cannot help ⇒ NOT
recoverable by de-phasing. **Caveat:** TMA is only a hard floor if it's DRAM-bandwidth-bound; here DRAM
util is low (~12% fwd), so a large TMA-wait would instead point at a producer scheduling/issue problem
(a DIFFERENT, still-open lever) — not necessarily an immovable floor. So read "TMA" as "not fixable by
de-phasing" rather than "unfixable".


---

## 0. Methodology — how to compare an NCU metric against the sim WITHOUT a definition trap

Repeatedly in this project a "gap" turned out to be an apples-vs-oranges metric mismatch. Rules that
must be applied before any HW-vs-sim number is trusted:

1. **Know the NCU denominator.** A `*.pct_of_peak_sustained_active` divides by (SM-**active** cycles ×
   `peak_sustained`). A `*.pct_of_peak_sustained_elapsed` divides by (**elapsed** cycles × peak). The sim
   usually normalizes by **elapsed**, so convert HW to the elapsed basis (× active/elapsed) before comparing.
2. **Know `peak_sustained` (the pipe count).** For `sm__pipe_*_cycles_active`, `peak_sustained` is the
   number of that pipe per SM (e.g. tensor = **4** = one per SMSP). The sim's per-SM counter sums all 4
   subcore pipes, so it must be divided by the SAME 4. Getting this wrong flips over-vs-under (it did once).
   → get `peak_sustained` from a raw-count profile (`--metrics ...avg.peak_sustained`), do NOT guess.
3. **cycles_active (BUSY) vs inst_executed (ISSUE).** `sm__pipe_tensor_cycles_active` (46%) counts cycles
   the pipe is busy; `sm__inst_executed_pipe_tensor_op_hmma` (2%) counts issue slots. WGMMA issues rarely
   but runs long → these differ ~20×. Compare sim busy-cycles against the BUSY metric only.
4. **"SM active" ≠ "issuing".** NCU `sm__cycles_active` = a warp is **resident** (90%). The sim's
   "SM-active"/`issuing` proxy = a subcore **issued** that cycle ([sm.cc:612](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L612)).
   HW's issue-basis metric is `Issue Slots Busy` (45%), not `sm__cycles_active`.
5. **Trace work-invariance.** The sim replays the HW trace, so op-counts should match HW `inst_executed`
   exactly — verify (they do: tensor 349,056; a mismatch means a counting bug, not a real difference).

---

## 1. Confirmed HW-vs-sim numbers (fwd kernel 5 unless noted)

### 1a. Cycles / occupancy (context)
| quantity | sim | HW | note |
|---|---|---|---|
| kernel cycles (fwd) | 105,245 | 67,696 | **1.56×** |
| kernel cycles (bwd) | 178,856 | 132,901 | 1.35× |
| CTA/SM | 1 (reg-limited) | 1 | same |
| warps/scheduler resident | 4 fwd / 3 bwd | 4 / 3 (theoretical) | occupancy matches; NOT the gap |
| active/elapsed | ~100% (1 CTA/SM) | 90.1% | different DEFINITION — see §0.4 |

### 1b. Issue rate / density (the real gap)
| quantity | sim | HW | ratio |
|---|---|---|---|
| issue rate (sim `issuing` vs HW `Issue Slots Busy`) | 38% | 45% | 1.18× |
| **issue-density** (warp-insts ÷ 528 sched ÷ elapsed cyc) | **0.289** | **0.407** | **1.41×** |
| HW `No Eligible` (cycles with 0 eligible warp) | — | **54.26%** | HW itself waits most cycles |
| HW eligible warps/scheduler | — | 0.83 fwd / 0.46 bwd | HW pool is also thin |

`cycle = work(1.10×) × density(1.41×) = 1.55×`. The gap IS issue-density. But HW's own No-Eligible 54%
means the de-phase ceiling is tightly bounded — HW is itself mostly-waiting on this workload.

### 1c. Per-pipe COST — all HW-faithful or LIGHTER in sim (raw-count confirmed, `peak_sustained` known)
| pipe | basis | sim | HW | sim/HW | verdict |
|---|---|---|---|---|---|
| **tensor (WGMMA)** | busy cyc / op (4-pipe) | **29.5** | **42.7** | **0.69×** | sim UNDER-models (not the cause) |
| tensor | busy % (÷ n_sm×elapsed×4) | 18.6% | 41.0% | 0.45× | |
| tensor op-count | total | 349,008 | 349,056 | 1.00× | per-warp convention identical |
| **SFU (MUFU)** | II (cyc/MUFU/SMSP) | 8 | 8 (`peak_sustained` 0.5/cyc/SM ⇒ 8/SMSP) | 1.00× | HW-faithful (Opt 10) |
| SFU op-count | — | =HW (work-inv) | 1,926,936 | 1.00× | identical |

**Key reframe**: sim runs each WGMMA *faster* than HW (29.5 vs 42.7 cyc/op) yet is 1.56× slower overall.
So the gap is categorically **not** a functional-unit cost error. It is **overlap/scheduling**: HW keeps
the tensor pipe busy 42.7 cyc/op AND concurrently runs MUFU (xu 47.7%) + memory waits, filling the SM
(density 0.407); the sim's in-phase consumer warps cannot interleave those pipes (density 0.289).

### 1d. HW stall breakdown (per issue_active) — the overlap map
| stall reason | HW fwd | HW bwd |
|---|---|---|
| `wait` (mbarrier / async / WGMMA.wait) | **1.363** | 0.784 |
| `long_scoreboard` (global/L2 latency) | 0.700 | **1.491** |
| `barrier` (named/CTA) | 0.782 | **1.313** |
| `not_selected` | 0.820 | 0.405 |
| `dispatch_stall` | 0.787 | 0.338 |
| `mio_throttle` / `short_scoreboard` (MIO/SFU) | 0.457 / 0.330 | 0.474 / 0.643 |
| pipe active tensor / xu / alu / fma | 46.1 / 47.7 / 27.3 / 16.9 | 53.6 / 21.4 / 15.3 / 9.5 |

fwd's dominant HW stall is `wait` (producer→consumer data dep) — a genuine floor that the sim's
`wait_barrier` (25% of `no_warps_ready`) mirrors. bwd's is `long_scoreboard` (memory latency).

---

## 2. What is REFUTED / closed (do not re-open without new evidence)
- **tensor over-model** — refuted; sim is 0.69× (under). The "1.78×" was a ÷1-vs-÷4 `peak_sustained` error.
- **SFU throughput / II** — HW-faithful (II=8 == HW 8/SMSP). Not a lever.
- **trace `stall_count` latency** — refuted as inflator: `>>=1` decay UNDER-applies the stall
  (FA3_progress Arch TODO-2), so it cannot make the sim slower.
- **warp-stagger / launch-offset (E1)** — measured ~0 (mbarrier re-synchronizes each tile). Reverted `0b863b0`.
- **occupancy floor** — sim resident warps == HW theoretical.

## 3. The open axis — issue-density / pipe-overlap
The residual is: **the sim does not overlap tensor + MUFU + memory-wait the way HW does.** HW hides each
pipe's latency behind another pipe (tensor 46% ∥ xu 48%), reaching density 0.407; the sim's consumer
warps move in phase so the pipes run more serially (density 0.289). This is a scheduling/concurrency
property of the model, not a per-op cost. Next investigations (choose):
- (a) **pipe concurrency in the sim**: can two FUs (tensor + SFU) in the same subcore be busy in the same
  cycle, and across the 4 subcores? Where does the model serialize what HW overlaps?
- (b) map HW `wait` / `long_scoreboard` magnitudes to the sim's `wait_barrier` / memory-wait to see if any
  sim wait is *longer* than HW's (a real lever) vs equal (a floor).
- (c) whether the 1-CTA/SM + in-phase mbarrier structurally prevents overlap that HW gets from jitter.

## 4. Investigation (a) DONE — pipe concurrency is NOT the bottleneck (2026-07-27h)

Traced the issue→FU path to see if the sim serializes what HW overlaps. Result: the FU **execution** is
already parallel, and the issue-path serialization is already resolved by Opt 10.

### Structure (source-confirmed)
- `Subcore::execute()` calls `fu->cycle()` for **every** FU each cycle ([subcore.cc:264-268](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L264)) ⇒
  once ops are IN their FUs, tensor + SFU + others progress **in parallel** (HW-faithful). Pipe
  concurrency at execute is NOT the problem.
- The only shared serial resource all FUs pass through is the 1-deep `m_ISSUE_CONTROL_latch` (1 issue /
  subcore / cycle). But that is **HW-faithful** — an SMSP scheduler also issues ≤1/cycle (no dual-issue
  post-Volta). Fixed-latency ops additionally pass `m_CONTROL_ALLOCATE_latch` (1-deep) + a 6-deep read
  pipeline; SFU bypasses both (goes straight to its dispatch reg).

### Measured — the latch head-of-line is already ~0 after Opt 10 (`.o59`)
| counter | value | % of evaluated (42.2M) |
|---|---|---|
| `next_stage_not_available` (latch full ⇒ scan skipped) | 25,472 | **0.06%** |
| `blocked_by_tensor` / `blocked_by_sfu` / `blocked_by_mem` | 2 / 0 / 2,640 | ~0 |
| `hol_reason_control_allocate_full` | 15,673 | 0.04% |
| `hol_reason_read_stage_full` | 0 | 0 |

⇒ Opt 10 (SFU issue-gate) already collapsed the issue-stage head-of-line to noise. The
CONTROL_ALLOCATE/read-stage serialization for tensor is also negligible (0.04%). **There is no remaining
issue-path serialization lever.**

### Where the density gap actually is (re-localized)
The 1.41× issue-density deficit is NOT latch serialization. It is that the warps are **not eligible** in
the first place: `no_warps_ready` = 51% of cycles, composed of `stall_count` 30% + `wait_barrier` 25% +
`fu_occupied` ~20% (overlapping). i.e. the sim's ~3 warps/scheduler are all simultaneously waiting on a
dependency/pipe, so there is nothing to issue — whereas HW, at the same moment, has a warp at a different
pipe stage to issue (density 0.407). This is the **warp-phasing/overlap** effect again.

### The honest strategic position (2026-07-27h)
Every mechanistic lead is now closed: per-pipe costs HW-faithful-or-lighter (§1c), issue-path
serialization resolved (Opt 10, this section), stagger refuted (E1), stall_count refuted (decay under-
applies). The residual density gap traces to **in-phase consumer warps all waiting together**, and the
one measured attempt to de-phase them (E1) was annihilated by the per-tile mbarrier. HW avoids this via
per-warp arrival jitter (real DRAM/bank/latency variance spread across a larger effective pipeline), which
the deterministic trace-replay model does not reproduce between the mbarrier re-sync points.

**Candidate levers that remain (all require care / a build; none is a clear win):**
- **(b) wait-magnitude audit**: compare sim `wait_barrier`/memory-wait DURATIONS against HW `wait`
  (1.363) / `long_scoreboard` per-occurrence. If any sim wait is *longer* than HW's, that is a real
  (faithful) lever; if equal, it is a floor. This is the last un-audited quantitative comparison and is
  the recommended next step (read-only-ish, needs per-wait duration instrumentation).
- **(c) structural floor acceptance**: if (b) shows sim waits ≈ HW waits, then the residual is the
  deterministic-lockstep floor bounded by HW's own 54% No-Eligible — accept, no cycle claim.


## 5. Investigation (b) — wait / throughput-floor audit: the gap is TENSOR↔MUFU OVERLAP (2026-07-27i)

Decomposed HW's warp-cycle budget and the sim's stall budget on a definition-matched basis, then computed
the hard throughput floors. This localizes the 1.56× to a **pipe-overlap** deficit with a concrete size.

### HW warp-cycle budget (fwd, `Warp Cycles Per Issued Inst` = 7.12, sum of states = 7.17 ✓)
| state | warp-cyc/inst | % |
|---|---|---|
| **wait** (mbarrier/async/WGMMA.wait) | 1.363 | 19.0% |
| selected (issued) | 1.000 | 13.9% |
| not_selected | 0.820 | 11.4% |
| dispatch_stall | 0.787 | 11.0% |
| barrier | 0.782 | 10.9% |
| long_scoreboard (mem) | 0.700 | 9.8% |
| mio_throttle | 0.457 | 6.4% |
| short_scoreboard | 0.330 | 4.6% |
| math_pipe_throttle | 0.229 | 3.2% |
| sleeping / no_instruction / imc / gmma / branch | 0.71 | 9.9% |

### sim subcore-cycle budget (fwd `.o59`, % of all evaluated)
`issuing 38.1%` | `no_warps_ready 51.3%` | `no_valid_instruction 9.5%` | `issue_port_busy 1.1%`.
`no_warps_ready` overlap sub-reasons: **fu_occupied 42.9%** (sfu 31.0% + sp_int_dp 13.8% + tensor 8.9%),
stall_count 15.6%, wait_barrier 13.0%, yield 4.7%.

### ⭐ The decisive throughput floors (per SMSP, over the run, same op-count as HW)
| pipe | ops/SMSP | II | floor cyc | % of HW elapsed (67,696) |
|---|---|---|---|---|
| **MUFU (SFU)** | 3,650 | 8 | **29,196** | **43.1%** |
| tensor (WGMMA) | 661 | 32 | 21,155 | 31.3% |

- **HW is itself ~43% MUFU-throughput-bound** (xu 47.7% ≈ this 43% floor). HW reaches elapsed 67,696 ≈
  the MUFU floor 29,196 × ~2.3 (with tensor OVERLAPPED underneath, not added).
- **sim SFU is UNDER-utilized per-cycle (27.7% vs HW 47.7%) yet blocks issue 31% of cycles** — because the
  3 in-phase consumer warps hit the SFU in the SAME bursts then idle together, instead of HW's smooth
  8-cyc-spaced stream. Same MUFU work (op-count identical), same II=8 (HW-faithful), but bursty vs smooth.
- The kernel-time model: HW ≈ **max(tensor 31%, MUFU 43%) + overhead**; sim ≈ **tensor + MUFU partially
  SERIALIZED + idle** = 105,245. The 1.56× is the **failure to overlap the tensor and MUFU pipes across
  the warpgroup**, NOT any per-op cost (all confirmed faithful).

### Why the pipes don't overlap in the sim (mechanism, partly confirmed)
- FU `execute()` is parallel (§4), so two FUs CAN run same-cycle. The block is upstream: the consumer
  warpgroup's dependent chain WGMMA→softmax(MUFU)→WGMMA is serial *within* a warp, and the 3 warps/SMSP
  are **in phase** (deterministic prior region + per-tile mbarrier re-sync), so when they're all in the
  softmax stage the tensor pipe is idle and vice-versa. HW's per-warp jitter keeps some warp in each stage.
- This is the SAME phasing root as the (closed) stagger axis — but now sized as a **43% MUFU floor that HW
  overlaps and sim serializes**, which is a much more concrete target than "lockstep".

### The remaining lever candidates (NOT a floor — still to test)
1. **wait_barrier / DEPBAR duration**: is the sim's WGMMA→MUFU (and MUFU→WGMMA) dependency wait LONGER
   than HW's? If the sim over-holds a warp at `wait_barrier`/`stall_count` between the tensor and SFU
   stages, that *lengthens* the serialization and is a real, faithful lever. **← next probe: measure
   per-occurrence wait_barrier duration in sim vs HW `wait` 1.363 / `short_scoreboard`.**
2. **cross-warp de-phasing via a REAL mechanism** (not launch-offset E1): if warps could enter the
   softmax stage at staggered cycles and STAY staggered (survive the mbarrier), tensor+MUFU would overlap.
   Needs a faithful jitter source (bank/RF-port cross-warp contention — the candidate B from the closed doc).
3. **stall_count magnitude** (Arch TODO-2): even though `>>=1` under-applies, the `num_stall_cycles_wait_
   after_bits_stall_0_and_yield=46` path fires on every yield-with-stall0; quantify how many sim
   warp-cycles that specific rule adds vs HW `sleeping` (3.1%) — if sim >> HW it is an over-model.

⚠️ The MUFU floor (43% of HW) means the ABSOLUTE ceiling for closing the gap by overlap alone is bounded:
even perfect tensor∥MUFU overlap cannot beat max(31%,43%)=43% of HW elapsed as the compute floor. So the
achievable sim improvement is from 105,245 toward ~HW's overlap efficiency, not below it.


## 6. RESULT — overlap+warp-cyc probe (2026-07-27j, fwd `.o61` / bwd `.o36`)

The 12h instrumentation run (gate `-overlap_instrument_enable 1`). `gpu_sim_insn` bit-identical
(fwd 455,565,060 / bwd 629,211,348 = work invariant); `gpu_sim_cycle` fwd 106,045 (+0.76% vs `.o59`
105,245) / bwd 179,175 — the per-warp map-lookup counters cost a hair of timing but do not affect the
composition below.

### Finding 1 — tensor↔SFU OVERLAP is actually GOOD (the earlier "no overlap" worry is refuted)
Per-SM-cycle classification (÷ n_sm×cyc):
| | both tensor&SFU | tensor-only | sfu-only | neither |
|---|---|---|---|---|
| FWD | 38.6% | 2.5% | 19.2% | 31.7% |
| BWD | 40.5% | 3.5% | 8.1% | 40.4% |

Of cycles where ≥1 of {tensor,SFU} is busy, **BOTH = 64% fwd / 78% bwd**. tensor-only is only 2.5/3.5%.
⇒ when the tensor pipe runs, the SFU almost always runs too. **The pipes DO overlap**; serialization of
tensor-vs-SFU is NOT the main lever. (This corrects §5's overlap-deficit hypothesis.)

### Finding 2 — the real concentration is SFU CONTENTION (the dominant stall)
Per-warp warp-cycles stalled, per issued warp-inst (HW `warp cycles per issued inst` basis):
| reason | FWD | BWD | HW-ish ref (fwd) |
|---|---|---|---|
| **fu_sfu** | **2.211** | **1.313** | (math_pipe 0.229 + mio 0.457 + short_sb 0.330 = ~1.0) |
| fu_other (sp/int/uniform) | 1.007 | 0.788 | alu/fma stalls |
| stall_count | 0.622 | 0.464 | (folded into HW pipe latencies) |
| wait_barrier | 0.552 | 0.626 | HW `wait` 1.363 |
| fu_tensor | 0.482 | 0.745 | HW `gmma` 0.097 |
| yield | 0.180 | 0.094 | HW `sleeping` 0.223 |
| **any (sum, non-excl)** | 4.561 | 3.882 | |

- **`fu_sfu` is the #1 sim stall — 48.5% of all stall warp-cycles (fwd)** — versus HW's SFU-ish share
  (~16.6% of stall). **sim is ~3× over-concentrated on SFU contention.**
- Each MUFU issue is associated with **~18 warp-cycles** of OTHER warps blocked on the SFU (fwd), i.e.
  ~2.3 warps queued on the one SFU for its full II=8 window every MUFU. Heavy same-cycle SFU contention.
- Note **`wait_barrier` is sim 0.552 vs HW `wait` 1.363** — the sim actually waits LESS on mbarrier/DEPBAR
  than HW. So the producer→consumer data-dep is NOT over-modeled; it is a HW-faithful/under floor. The
  excess is squarely SFU contention, not the barrier wait.


### fwd vs bwd — is the SFU over-concentration the same in both? (2026-07-27k)
Both kernels have SFU as the #1 stall, and the PER-MUFU contention is nearly identical; but the SHARE
differs because bwd has more tensor/barrier stall diluting it.
| metric | FWD (k5, .o61) | BWD (k10, .o36) |
|---|---|---|
| fu_sfu warp-cyc/inst | **2.211** | **1.313** |
| fu_sfu share of tracked stall | **43.7%** | **32.6%** |
| fu_sfu warp-cyc PER MUFU op | 18.4 | 17.8 |
| avg warps blocked over the II=8 window | **2.30** | **2.22** |
| fu_tensor warp-cyc/inst | 0.482 | 0.745 |
| wait_barrier warp-cyc/inst | 0.552 | 0.626 |

- **The per-MUFU SFU contention is the SAME in both** (~18 warp-cyc/op, ~2.3 warps queued over II=8) —
  it is the same in-phase consumer-warp structure.
- **But SFU DOMINATES fwd (43.7%) while in bwd it is #1-but-shared (32.6%)**: bwd runs fewer MUFU
  (HW xu 21.4% vs fwd 47.7%) and more tensor (53.6% vs 46%), so bwd's fu_tensor (0.745) and wait_barrier
  (0.626) rise to SFU's level and spread the stall out.
- So the earlier "~3× over-concentrated on SFU" statement is a **fwd** figure. bwd is SFU-led but
  multi-factor. The de-phasing lever helps both (identical per-MUFU contention) but its ceiling is larger
  for fwd.

### Interpretation — same root (phasing), now precisely localized to SFU contention
II=8 is HW-faithful (confirmed by `peak_sustained` 0.5/cyc/SM = 8/SMSP). The sim's 3 consumer warps/SMSP
hit the SFU **in phase** (all in the softmax MUFU stage at once), so they queue on the single SFU;
HW's 12 warps are phase-dispersed so the same MUFU work spreads out (xu 47.7% steady, low per-warp SFU
stall). So the gap is the same lockstep/phasing root, but its DOMINANT symptom is SFU issue contention
(fu_sfu 2.211), not tensor cost (faithful), not overlap (good), not wait_barrier (under HW).

### Consequence for a lever
- The only faithful way to reduce fu_sfu contention is to **de-phase the consumer warps** so their MUFU
  streams interleave (HW's behavior) — the same candidate B (real cross-warp jitter) from the closed
  stagger doc, now with a concrete target (cut fu_sfu 2.211 → toward HW ~1.0 ⇒ recovers a large share of
  the 48.5% SFU stall). Launch-offset (E1) is dead (mbarrier re-sync); the jitter must live INSIDE the
  per-tile compute region and survive the bell.
- ⚠️ Bounded: HW itself is ~43% MUFU-throughput-floored, so even perfect de-phasing cannot go below that.
  But sim's fu_sfu 2.211 vs HW ~1.0 is headroom well above the floor — this IS a real, sized lever.


## 7. SFU latency/II — clarified (2026-07-27k): total is ALREADY 16 (HW-faithful); config was just confusing

Investigating whether SFU result latency is over-modeled (suspected 21 vs HW ~16) resolved to: **it is
already 16, and correct.** An earlier read in this session that "SFU op latency = 21" was WRONG — it
confused the FU pipeline-array depth with the op's latency field. Corrected trace:

### The mechanism (why II and latency are two numbers)
The SFU is modeled as a **pipelined** unit. An op:
1. sits in `m_dispatch_reg` for **II** cycles (`dispatch_delay` counts `cycles=initiation_interval` down,
   [functional_unit.cc:306-318](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L306)) — during this window `can_issue()=false` so it blocks OTHER warps' SFU issue (this is the throughput/contention term);
2. then enters `m_pipeline_reg[latency_field-1]` and takes **latency_field** more cycles to produce its
   result (the result-latency term).
So **total result latency = II + latency_field**, and because II<total, up to `total/II ≈ 2` ops are
in-flight concurrently (real pipelining — the "op1 in the back half, op2 in the front half" the user asked
about). II and latency are two numbers precisely because a pipelined unit has independent throughput vs depth.

### The THREE confusing SFU knobs (only one drives FA3)
| knob | value | used for FA3 (trace mode)? |
|---|---|---|
| **`-trace_opcode_latency_initiation_sfu 8,8`** | lat=8, II=8 | ✅ **THE real value** ([trace_driven.cc:811,849-852](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L811)) ⇒ total = 8+8 = **16** |
| `-ptx_opcode_latency_sfu 21` / `_initiation 8` | 21 / 8 | ❌ PTX functional-sim only ([cuda-sim.cc:1009](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc#L1009)); ignored in trace |
| `-sfu_latency 21` (default) | 21 | ❌ only sizes the FU pipeline-array depth `m_pipeline_depth` ([subcore.cc:1772](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1772)); op enters stage 7, so it's just array capacity, NOT timing. This is why `[LATCFG]` prints sfu=21 while the real op latency is 8. |

### Verdict
- **SFU total result latency = 16 cyc = HW-faithful** (Ampere/Hopper MUFU microbench ~16). Confirmed by the
  tensor precedent: `generate_tensor_core_latencies` also splits one op's total as `II=nc/2; latency=nc-II`
  (II+latency=total). So there is **nothing to lower** — the earlier "reduce 21→16" plan is moot.
- Config action taken: added a disambiguation comment block at the `-ptx_opcode_*_sfu` lines so the three
  knobs are no longer confusing (timing unchanged / bit-identical).
- ⇒ SFU is HW-faithful in BOTH II (8, throughput) AND total latency (16). The `fu_sfu` contention
  (§6, 2.211 warp-cyc/inst) is therefore NOT a latency over-model — it is purely the II=8 throughput
  meeting the in-phase consumer warps (the phasing root). No SFU-cost lever exists; the lever remains
  de-phasing the warps so their MUFU streams interleave.


## 8. CRITICAL-PATH static analysis (2026-07-27l) — the gap is SM-idle from failed latency-hiding, NOT SFU cost

User pushback (correct): "if the sim is fully deterministic and warps just pile onto the SFU at the same
cycle, the total cycle should be the same — de-phasing only reorders, it can't cut cycles unless the SFU is
a real throughput bottleneck." This forced a critical-path re-analysis. The user is right, and it resolves
the paradox (per-op latencies are ≤ HW yet total is 1.56×).

### SFU is NOT the sim bottleneck (user confirmed)
- SFU throughput floor = MUFU/SMSP(3650) × II(8) = **29,196 cyc = only 28% of sim elapsed** (106,045).
  The sim runs 3.6× longer than its own SFU floor ⇒ SFU throughput is not what sets the cycle count.
- `fu_sfu` 2.211 warp-cyc/inst is an issue-stage OBSERVATION (non-exclusive overlap counter); much of it
  is warps that were ALSO waiting on their own data dep, double-attributed to "SFU busy". Not a lever by itself.

### The real decomposition — per tile (fwd, SMSP)
| component (per tile, fwd, SMSP) | sim | HW |
|---|---|---|
| tensor work (16.5 op; busy-cyc/op sim 29.5 / HW 42.7) | 487 | 705 |
| MUFU work (91.2 op; II 8) | 730 | 730† |
| **compute work (if overlapped = max)** | **730** | **730** |
| **wall-clock / tile** | **~3300** | **~1700** |
| **pure wait/idle / tile** | **~2570** | **~970** |

(Tensor work differs — HW is LONGER per op, §6 — but MUFU 730 dominates so compute≈730 on both by
coincidence. HW does more tensor work yet finishes faster ⇒ the gap is wait/idle, not compute.
† MUFU 730 is an II=8 throughput-floor, not an HW measurement: op-count is identical and II=8 is
HW-faithful, but the SFU has no HW busy-cycle counter (§1c) so HW's real per-op SFU occupancy is
unmeasurable — HW's actual MUFU issue is only 48% of peak.)

The compute work (730) is identical sim/HW; the **entire gap is wait/idle** — sim wastes ~2570 cyc/tile vs
HW's ~970.

### Where the wait shows up: SM FULLY-IDLE = 31.7% (fwd)
- `total_num_cycles_sm_all_subcores_idle` = 4,442,929 ⇒ **31.7% of elapsed** (= /(132×106,045)).
  ⚠️ the run's printed `percentage_..._sm_all_subcores_idle = 10.50%` is WRONG — it divides the SM-cycle
  numerator by the subcore-cycle denominator (evaluated 42.3M = 132×4×cyc), a 4× definition mismatch. The
  correct SM-idle is 31.7%. (HW SM-idle is ~10%.)
- Per-warp waits are SHORT: sim `wait_barrier` 0.552 + `stall_count` 0.622 = 1.17 warp-cyc/inst, actually
  LESS than HW `wait` 1.363. So each warp's own critical-path wait is HW-faithful/shorter.

### ⭐ Resolution — critical-path and de-phasing are the SAME problem
1. Each warp's critical-path wait (WGMMA→MUFU→WGMMA data deps) is NOT over-modeled (sim ≤ HW per warp).
2. BUT the 3 in-phase warps/SMSP hit those waits **simultaneously**, so the whole SM stalls together →
   **SM fully-idle 31.7%** = failed latency-hiding. HW overlaps: while warp A waits on a WGMMA/MUFU result,
   warp B issues → SM-idle ~10%.
- So the earlier "SFU-contention lever" framing was a symptom; the ROOT is **latency-hiding failure from
  warp lockstep**. de-phasing's payoff is NOT emptying an SFU queue (SFU is 28%-floored) — it is
  **filling the 31.7% SM-idle** by letting a non-waiting warp run during another's dependency wait.
- ⇒ **This RE-VALIDATES de-phasing as the lever** (user: "keep de-phasing in mind"), now with the correct
  mechanism (idle-fill, not queue-drain) and a sized target: cut SM-idle 31.7% → ~10% (HW) ≈ the whole gap.

### Caveat / next check
The 2570-cyc/tile wait must be decomposed into: (a) intra-warp dependency stall that COULD be hidden by
another warp (recoverable by de-phasing) vs (b) genuine all-warps-blocked (e.g. all waiting on the same
TMA tile via mbarrier — a producer floor). The overlap `neither` bucket (tensor&SFU both idle) + the
per-CTA `sm_idle_wait_barrier_only` (CTAFIN) can size (b). If (b) dominates, de-phasing is capped by the
producer; if (a) dominates, de-phasing is the real lever. This is the next static/probe step.


## 9. (a) recoverable vs (b) floor decomposition (2026-07-27m) — SM-idle is MOSTLY wait_barrier; need to split it

Decomposed the 34.5% SM-fully-idle (CTAFIN, fwd `.o61`) to see if de-phasing is a real lever (a) or capped
by a producer/mbarrier floor (b).

### SM-idle breakdown (avg/SM, % of elapsed 97,609)
| component | % of elapsed | % of SM-idle | interpretation |
|---|---|---|---|
| **wait_barrier_only** | 19.1% | **55.4%** | SM-idle whose sole block is `wait_barrier` (mbarrier/DEPBAR) |
| ibuffer_empty | 19.3% | 56.0% | trace-drained head (tail / winding down) |
| drained | 5.4% | 15.7% | trace exhausted (straggler tail) |
| tensor-block | 3.5% | — | tensor FU busy |
| **rest (not wb-only, not drained)** | ~9.9% | **~29%** | the clearest recoverable-by-scheduling bucket |
| **total SM fully-idle** | **34.5%** | 100% | |

### The blocker to a clean (a)/(b) verdict: wait_barrier is TWO things merged
`wait_barrier_only` (55% of idle) is the dominant idle reason, BUT the sim's single `are_wait_barriers_ready`
check does NOT distinguish:
- **(b) TMA-tile mbarrier** — consumer waiting for the producer to land the next tile ⇒ NOT recoverable by
  de-phasing (the tile is produced outside the SM). NOTE this is a producer/mem-axis wait, not necessarily
  an immovable floor: with DRAM util ~12% it may be a producer scheduling/prefetch gap (separate lever).
- **(a) WGMMA-result DEPBAR** (`wgmma.wait_group`) — consumer waiting on its OWN prior WGMMA ⇒ if warps were
  de-phased, another warp could run during this wait ⇒ recoverable.
So the 55% cannot yet be assigned to (a) or (b).

### Producer/consumer timing (CTAFIN) — a hint, not a verdict
- `producer_last_drain` = 99,103 ≈ `finish` 99,110 → the producer (TMA) keeps landing tiles until ~kernel
  end.
- `consumer_last_drain` = 77,848 = **78% of finish** → the consumer does its last work at 78%, then the last
  ~22% is drain/idle (straggler tail).
- ⚠️ This looks like "producer busy to the end, consumer finishes early" — the OPPOSITE of the old
  CONSUMER_COMPUTE_BOUND "producer waits on consumer" reading. But `producer_last_drain≈finish` may just be
  the producer warp's teardown, not late tile supply, so treat as a hint only.

### Verdict so far
- **~29% of SM-idle is clearly recoverable** (rest bucket), **~16% is drain tail (floor)**, and the big
  **55% wait_barrier chunk is UNRESOLVED** (split needed).
- If most of the 55% is WGMMA-DEPBAR (a), de-phasing is a large lever (up to ~55%+29% ≈ 84% of the 34.5%
  idle ≈ recovering ~29% of elapsed → could bring sim near HW's 10% idle). If most is TMA-mbarrier (b),
  de-phasing won't recover it and the investigation reroutes to the producer path (bandwidth vs prefetch
  scheduling) — a true floor only if the producer is genuinely bandwidth-bound (DRAM ~12% suggests not).

### Next step (needs a build) — split wait_barrier by kind
Add a gated read-only counter that, on a `wait_barrier`-blocked warp, classifies the barrier it waits on:
TMA-tile mbarrier (producer floor) vs WGMMA/DEPBAR (recoverable). The sync-barrier addr / kind is already
known at the wait site (`build_sync_barrier_key`, HopperMBarrier vs the SASS wait_barrier bits). This is
THE measurement that decides whether de-phasing is the lever or a producer floor caps it. Fold into the
next 12h run alongside the existing overlap probe.

## 10. wait_barrier-kind probe — IMPLEMENTED (2026-07-27n), awaiting 12h run

To split the unresolved 19.1% `wait_barrier` (§9) into recoverable (WGMMA-result) vs floor (TMA), a gated
read-only probe was added under the existing `-overlap_instrument_enable`.

### Mechanism (why it can separate them)
- Each warp has 6 scoreboard (SB) wait-barriers. An async op that will later signal SB#k arms it at
  control_stage (`add_pending_wait_barrier_increment`, sm.cc). We tag that SB with the ARMING op's class:
  `WB_ARM_TENSOR` (WGMMA), `WB_ARM_TMA` (UTMALDG/cp.async.bulk), or `WB_ARM_OTHER`.
- When a warp is blocked at issue on `!are_wait_barriers_ready`, we look up which of its checked SBs are
  NOT ready and return the dominant arming class (precedence TENSOR>TMA>OTHER), accumulating per-SM into
  `warpcyc_wb_wait_{tensor,tma,other,unknown}`.
- Files: `warp_dependency_state.{h,cc}` (enum + Wait_Barrier tag + `blocking_wait_barrier_armed_op`),
  `sm.cc` (tag at arm time), `subcore.cc` (classify at stall), `gpu-sim.cc`/`shader.cc` (4 stats+prints).
  Read-only; arm-time tag is a plain enum push (timing-neutral). Headers changed → `make clean`.
- ⚠️ Caveat from the search audit: in this trace model, WGMMA RESULT waits may surface as
  `stall_count`/traditional-scoreboard rather than SB `wait_barrier`, and TMA-tile arrival mostly goes
  through the mbarrier path (`waiting()` = inst_barrier, which is ~0.03% here). So the SB `wait_barrier`
  bucket is expected to be dominated by DEPBAR / `wgmma.wait_group` (WGMMA-group) — the probe will confirm
  or refute this. If `wb_wait_tensor` dominates ⇒ the 19.1% is largely recoverable.

### Analysis plan for the next run (combine with the "10%")
1. Read `warpcyc_wb_wait_tensor / tma / other` (fwd `.o*`, bwd `.o*`).
2. WGMMA-fraction = `wb_wait_tensor / (tensor+tma+other)`.
3. Total recoverable-by-de-phasing = **rest 9.9% + 19.1% × WGMMA-fraction** (of elapsed).
4. Upper-bound new cycle = `sim_cyc × (1 − recoverable%)`; report the new sim×HW multiplier.
   (Reminder: this is an UPPER bound — idle-fill via de-phasing is not 100% cycle-for-cycle because the
   per-warp critical paths only partly overlap; the honest deliverable is the sized ceiling, then a
   faithful de-phasing mechanism (candidate B: dynamic cross-warp bank/RF-port contention) would be built
   only if the ceiling justifies it.)

### If the probe says WGMMA dominates (recoverable is large)
Then de-phasing is the real lever and the next build is the faithful jitter mechanism (candidate B from the
CLOSED WARP_STAGGER doc — cross-warp bank/RF-port contention that survives the mbarrier). If TMA dominates,
de-phasing SM-internal warps will NOT recover it (the awaited tile is produced outside the SM) — but that
does NOT automatically make it an immovable floor: with DRAM util only ~12% (fwd) the tiles are not
bandwidth-starved, so a large TMA-wait would indicate a PRODUCER scheduling/prefetch problem (a separate,
still-open lever), whereas a true floor only holds if the producer is genuinely bandwidth-bound. So a
TMA-dominated result reroutes the investigation to the producer path, it does not simply cap the gain at
rest-9.9%.


## 11. Solution direction — scheduler policy (GTO → LRR/GTLRR) to break the lockstep (2026-07-27o)

The "how", designed while the wait_barrier-kind run is in flight. Root cause (mechanism): the sim's
**GTO** scheduler actively CREATES the lockstep — it greedily pushes one warp until it stalls, then the
next warp runs the SAME deterministic code the same distance and stalls at the SAME point, so the 3
consumer warps/SMSP converge to one phase and hit dependency waits together (SM-idle 31.7%).

### What NVIDIA HW actually does (web audit)
- H100 SM = 4 warp schedulers (1/SMSP), 1 issue/cycle; eligible = operands ready + target FU free; zero-
  overhead switch to a ready warp when one stalls. Matches the sim's per-subcore 1-issue model.
- **The exact warp-selection policy is NOT public (black-box).** NVIDIA docs and academic microbench
  papers (WPI thread-block scheduler; "Dissecting Blackwell" arXiv:2507.10789) only say "pick a ready
  warp"; they cannot pin GTO vs LRR vs age. GPGPU-Sim lore: real HW is closer to an age+round-robin mix
  than to pure GTO.

### Sim status (code)
- The remodeling `subcore.cc` HARD-CODES GTO via `order_greedy_then_highest_id` ([subcore.cc:589](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L589));
  there is no policy switch on this path. Stock GPGPU-Sim's LRR / GTO / GTLRR / two-level (shader.h:529-543)
  live only in the original `scheduler_unit`, which the remodeling issue path does NOT use.

### The plan: add an LRR / GTLRR ordering to the remodeling subcore, gated, and A/B it
- **GTO** (today): greedy → aligns warps → lockstep.
- **LRR** (loose round-robin): rotate warps every cycle → warps naturally spread across different
  instruction stages → de-phase → a warp is likely in a non-waiting stage while another waits ⇒ fills
  SM-idle. 
- **GTLRR** (greedy-then-LRR) is the sweet spot: keep greedy locality (good for i-cache/RF), but when the
  greedy warp stalls, fall to round-robin instead of oldest — so the fallback doesn't re-pick a warp that
  will stall at the same point. This directly targets the lockstep without throwing away GTO's locality.
- Why this is defensible (not an arbitrary knob like physical jitter): LRR/GTLRR are established GPGPU-Sim
  policies with published fidelity data; HW's true policy is unknown and believed to be a round-robin-ish
  mix, so LRR/GTLRR is a legitimate hypothesis, not a fudge.

### Risk / open question (same failure mode as E1)
GTO→LRR reorders WHO issues, but if every warp still runs the identical deterministic path, LRR may only
shift the phase without WIDENING it — the warps could re-converge (this is exactly why launch-offset E1
died on the per-tile mbarrier). LRR's advantage over E1 is that it re-applies the rotation EVERY cycle
(not once at launch), so a warp pulled ahead stays ahead until the next mbarrier — but whether that
survives the mbarrier re-sync is the empirical question. **Validate cheaply first:** add the ordering as a
gated option, A/B on FA3, and read SM-idle 31.7% + the overlap `neither`/`sfu_only` buckets. If SM-idle
drops, it's real; if it re-converges, we learn LRR alone is insufficient and the jitter must come from
somewhere the mbarrier can't erase (→ candidate B).

### Guardrail
Any new ordering is gated + default-off (GTO stays the default, bit-identical when off). Report the new
cycle count AND confirm `gpu_sim_insn` stays bit-identical (work-invariant) so we know only scheduling,
not work, changed.


⚠️ Guardrails: keep every II/latency HW-faithful (confirmed above); no pipe may exceed its NCU occupancy;
any prototype gated + default-off + bit-identical when off; if the honest answer is "structural floor
bounded by HW's own 54% No-Eligible", accept it and make no cycle claim.
