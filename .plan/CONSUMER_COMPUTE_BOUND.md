# FA3 fwd — Consumer-Compute-Bound Analysis (2026-07-20)

## ⚠️ UPDATE (2026-07-20b) — the "consumer compute is 1.9× too slow" conclusion below is CORRECTED

A follow-up pipe-level measurement (HW NCU pipe-active vs sim per-pipe fu_occupied) **overturns** the
"consumer per-tile *compute* is 1.9× too slow" reading in the original TL;DR. The truth is the opposite:
**sim's consumer compute is if anything FASTER than HW; the wall is that sim cannot keep the compute
pipes busy, whereas HW packs tensor (46%) + MUFU/softmax (48%) nearly back-to-back.** See the
"Pipe-level breakdown" section at the bottom — that is the current conclusion. The original text is kept
below as the reasoning trail.

---

## TL;DR (original — see correction above)

The fwd 2.0× sim-vs-HW gap is **not** a memory, producer, frontend, or occupancy problem. A per-CTA
TMA-timeline analysis of the SM0 log (`OnlyKernel5/.o45`) shows the kernel is **consumer-bound**:
the producer fetches ~2 tiles then waits ~6,645 cyc on average for the consumer warpgroup to compute
and free the shared-memory buffer. **93% of the TMA span is producer-waiting-on-consumer.** The wall
time is set by the consumer's per-tile progress (~3,300 cyc/tile sim vs ~1,700 HW) — but the per-tile
breakdown (below) shows the difference is **stall/overlap, not raw compute cost**.

This localizes the residual to the **tensor/math re-issue interval** (stall-depth 1.29× factor,
`math_pipe_throttle` sim 11% vs HW 3.2%, `mma` sim 5.65% vs HW 1.4%) — the same axis the async-WGMMA
investigation pointed at but deferred. It re-opens that axis with a concrete, quantified target.

## How we got here (this session)

Starting from the "drain-idle 1.39× is the largest factor, not HW-faithful" finding, we chased the
producer/consumer warp structure and ruled out several candidates before landing on consumer compute:

1. **HW trace warp structure (kernel_5 CTA0, 16 warps).** Parsed the HW dynamic trace
   (`hw_run/.../traces/threadblocks/.../kernel_5/d_0_s_0_k_5_0,0,0.pb`) with a minimal protobuf reader:
   - **warp 0 = producer** (1 warp): 14,082 insts, of which **~66% is a spin loop**
     (`PHASECHK.TRYWAIT` + `NANOSLEEP.SYNCS` + `PHASECHK` + `BRA`), only 44 real `UTMALDG`/`UTMAPF`.
   - **warps 4-15 = consumer** (12 warps): ~8,600 insts each, **216 HGMMA each**, plenty of `MUFU.EX2`
     (softmax exp), almost no spin.
   - warps 1-3 = setup (55 insts).
   - Producer having MORE insts than consumer is **trace-inherent**; work is spread uniformly across the
     trace (per-decile spin ~69% and HGMMA ~24/decile, no back-loading).

2. **NANOSLEEP spin lever — REFUTED (separate experiment).** Widening NANOSLEEP latency 1→64 (dedicated
   knob) did not raise eligible-warp and slightly raised cycles (+0.7-0.9%). At 1-CTA/SM occupancy the
   freed issue slots have no other warp to fill. Producer spin is a *symptom* (waiting on consumer), not
   the driver. See FA3_progress.md. Config restored to baseline.

3. **3-factor identity re-confirmed (`.o45`).** `cycle 2.03× = work 1.11× × issue-throughput 1.85×`,
   where issue-throughput = `active-frac 1.39× × issue-depth 1.31×`. This matches the pre-existing
   documented decomposition (`2.01× = 1.10 × 1.29 × 1.39`); nothing new, just re-derived. The two big
   factors (active-frac, issue-depth) are systematic (spread across all insts), which is why per-axis
   scans each look "small".

## The decisive measurement — TMA timeline (SM0, `.o45`)

Extracted every `TMA - SM 0 - complete uid=... dir=0` (load) event = the 40 tiles that feed the consumer
mbarriers, with their completion cycle, `lat_total`, and inferred issue cycle.

| quantity | value | read |
|---|---|---|
| load-TMA completes (tiles) | 40 | one per consumer mbarrier fill |
| TMA span (first→last complete) | 7,988 → 115,031 = 107,043 cyc | |
| tail after last TMA → kernel end | 115,031 → 137,207 = 22,176 cyc | consumer finishing last tiles + epilogue |
| **per-tile `lat_total` (arrival latency)** | **mean 402 cyc** | memory is NOT the wall |
| inter-completion gap | mean 2,745, p50 484, **max 16,290** | |
| **large gaps (>2000 cyc)** | **15 windows, Σ = 99,687 cyc = 93% of TMA span** | producer waiting on consumer to free buffer |
| avg large-gap length | **6,645 cyc** | ≈ time for consumer to process ~2 tiles (double-buffer) |
| producer issues tile[i] before tile[i-1] completes (pipelined) | 17/39 | ~2-deep double buffer |

**Pattern:** tiles arrive in **pairs** (gap ~100-1000 cyc), then a **7,000-16,000 cyc void** with no new
tile, then another pair. The void is the consumer computing the pair; the producer sits in its spin loop.
So the software pipeline is ~2-stage (double buffer), and the **consumer's per-tile compute (~3,300
cyc/tile) dominates the wall time**, not the ~402-cyc memory latency.

## Why this is consumer *compute*, not consumer *idle*

Top-level breakdown (`.o45`):
- SM-all-idle = **17.2%** (no subcore issues at all)
- issuing = **34.8%**
- ≥1 warp fu_occupied stall = **15.2%**

So during the "producer-waiting" voids the SM is **not** idle — the consumer's 12 warps are busy
(only 17% of the whole kernel is fully idle). The 93%-producer-wait is producer-relative; SM-wide it is
consumer compute. This reconciles with the earlier "tensor SM-wide-blocked only 1.75%": that counts
cycles where *all* subcores are blocked by tensor, but the consumer compute shows up as **stall-depth**
(warps present but not issuing: `math_pipe`/`mma` stalls), not as SM-idle.

## The quantified target

- sim: consumer per-tile compute ≈ 99,687 cyc void ÷ ~30 tiles-in-voids ≈ (≈ 6,645 cyc / 2 tiles) ≈
  **~3,300 cyc/tile** (double-buffered pairs).
- HW: elapsed 67,696 / 40 tiles, double-buffered ≈ **~1,700 cyc/tile**.
- ratio ≈ **1.9×** — essentially the whole 2.0× gap.

This lines up with the warp-state taxonomy (per issue_active, `.o45` vs NCU kernel-5):
`math_pipe_throttle` sim 11.0% vs HW 3.2% (**3.4×**), `mma` sim 5.65% vs HW 1.4% (**4×**). The sim is
spending too many cycles with the consumer's WGMMA/math head blocked on the tensor/math FU re-issue
interval — i.e. the **per-tile tensor+softmax compute is modeled ~1.9× too slow**.

## Relation to the (deferred) async-WGMMA axis

FA3_progress.md "Deferred Opts → async-WGMMA" closed the tensor axis on the grounds that (a) the sim is
already effectively async (producer re-issues after the small II; consumer blocks on the real DEPBAR data
dep, as HW does at `wgmma.wait_group`), and (b) the HW back-calc suggested sim II=32 is *smaller* than HW
II≈72, so "II too big" was falsified. **This analysis does not contradict that**, but it re-frames the
target: the wall is the consumer's *aggregate* per-tile compute (WGMMA + `MUFU.EX2` softmax + the math
pipe between them), which is ~1.9× HW. The open question is **which part of the per-tile consumer compute
is over-costed** — WGMMA issue serialization, MUFU/SFU latency (TODO-2: SFU modeled at FP-add cost, HW
xu-pipe is fwd's busiest at 47.75%), or the math pipe between them.

⚠️ **Direction caveat:** TODO-2 (SFU/MUFU latency) is currently *under*-modeled (4-cyc, FP-add cost).
Fixing it makes fwd *slower*, not faster — it is a compensating error. The net effect of correcting the
consumer-compute model on the 2× ratio is therefore not obvious and must be measured, not assumed.

## Pipe-level breakdown (2026-07-20b) — the CORRECTED conclusion

Measured HW pipe-active (NCU kernel 5, `sm__pipe_*` / `sm__inst_executed_pipe_*` .avg.pct_of_peak_
sustained_active) vs sim per-pipe fu_occupied (`.o45`, step0 counters). This is the decisive split.

| pipe | HW (% of SM-active) | HW cyc (×61,147) | HW cyc/tile (÷40) | sim | note |
|---|---:|---:|---:|---|---|
| **tensor (WGMMA)** | **46.1%** | 28,189 | 705 | fu_occupied_tensor **17,041/CTA = 13.4%**, ~426/tile | sim tensor busy is **LESS** than HW |
| **xu (MUFU = softmax `EX2`)** | **47.75%** (HW's #1 pipe) | 29,167 | 729 | **fu_occupied_sfu = 0.00%** | sim SFU at 4-cyc (TODO-2) → ~free |
| fma | 16.9% | 10,334 | 258 | folded in sp_int_dp 11.0% | |
| alu | 27.3% | 16,693 | 417 | | |
| **SM-active fraction** | **90.1%** (61,147/67,838) | | | sim **64%** | HW packs pipes; sim idles |

**The correction (this overturns the "1.9× too-slow compute" reading):**
- **sim's consumer compute is NOT too slow — it is if anything too FAST.** sim tensor busy 13.4% of
  elapsed (~426 cyc/tile) vs HW tensor 46.1% (~705 cyc/tile); sim MUFU/softmax ≈ **0** (SFU modeled at
  4-cyc FP-add cost) vs HW MUFU **47.75%** — HW's single busiest pipe.
- **HW is compute-DENSE:** tensor 46% + MUFU 48% run nearly back-to-back, so HW's SM is **90% active**.
  The per-tile ~1,700 cyc is ~1,434 cyc of actual tensor+MUFU work (84% busy) — HW is genuinely
  compute-bound and packs the two pipes tightly.
- **sim is compute-SPARSE:** per-tile ~3,300 cyc but only ~426 cyc of tensor work and ~0 MUFU; the other
  ~87% is stall/idle (`math_pipe`/`mma`/`wait_barrier` between ops). sim runs the compute *faster* but
  **cannot overlap/pack it**, so the consumer's wall-clock per tile is longer despite less busy work.

**Reframed root cause:** the fwd 2× is **not** over-costed compute. It is that (a) sim under-models the
softmax `MUFU.EX2` pipe (SFU@4-cyc → ~0 occupancy vs HW 47.75%, TODO-2), so the sim consumer has almost
no MUFU work to interleave with WGMMA, and (b) sim fails to keep tensor+math densely overlapped the way
HW does (HW 90% SM-active vs sim 64%). The gap is an **overlap/packing + missing-MUFU-cost** problem, not
a per-op latency-too-large problem. This is why async-WGMMA ("II too big") was correctly refuted, and why
NANOSLEEP (spin) was a non-lever.

**Sharp implication for TODO-2 (SFU/MUFU):** giving MUFU a realistic latency is not just a fidelity fix —
it is the **only way to reproduce HW's #1 pipe (xu 47.75%)**. Whether it moves the *cycle ratio* toward
HW depends on whether the added MUFU work **overlaps** with WGMMA (like HW) or **serializes** (making sim
slower). That overlap behavior is now the key open question — and it cannot be answered by adding SFU
latency alone; it needs the SFU cost to sit on a pipe that runs concurrently with the tensor pipe.

## Next steps (proposed, not yet run)

1. ✅ **DONE — HW per-tile compute measured.** HW ~1,700 cyc/tile = tensor 705 + MUFU 729 + overlap;
   SM 90% active. Confirms HW is compute-dense.
2. ✅ **DONE — sim per-tile split.** tensor ~426/tile, MUFU ~0 (SFU@4-cyc), rest is stall. sim compute
   is sparse, not slow.
3. **NEW key question:** does the sim model MUFU (SFU pipe) as a **separate concurrent pipe** from tensor?
   If yes, applying TODO-2 (realistic SFU latency) could raise sim SM-active toward HW 90% by giving the
   consumer MUFU work to overlap with WGMMA — potentially *reducing* the ratio, not increasing it (unlike
   the earlier caveat which assumed serialization). Audit the SFU FU concurrency + measure with TODO-2 on.
4. **Then** decide if TODO-2 (SFU latency) is a cycle lever (overlap) or a fidelity-only slowdown (serial).

## Root cause of the "87% wait" — issue-pipeline head-of-line blocking (2026-07-20c)

Decomposed the sim per-SMSP cycle budget (`.o45`, % of `evaluated`): issuing 34.8%,
**`next_stage_not_available` 24.96%** (11.5M cyc — the 2nd-largest bucket), fu_occupied 15.2%,
wait_barrier 12.9%, stall_count 9.4%. The big under-examined term is `next_stage_not_available`.

**What it is (source-confirmed):** the fixed-latency issue pipeline is
`issue → ISSUE_CONTROL(1) → CONTROL_ALLOCATE(1) → read_stage(6-deep for WGMMA) → TENSOR FU(lat 32)`
([subcore.h:189-190](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h#L189) — each latch is **1-deep**;
WGMMA read latency = `MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST` = 3×2 = 6, tensor_latency = 32). In
`Subcore::issue()` the entire warp-scan loop runs **only if `m_ISSUE_CONTROL_latch.has_free()`**
([subcore.cc:458](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L458)); otherwise it falls straight to
`else { is_next_stage_availabe = false; }` ([subcore.cc:822-824](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L822)) and
`n_eligible_this_cycle` stays 0. So when the 1-deep ISSUE_CONTROL latch is occupied (its instruction
can't advance because the tensor FU latency-bitset slot is reserved, or CONTROL_ALLOCATE/read_stage is
full), **the subcore does not even look at any other warp that cycle — the whole subcore is stalled.**
This is head-of-line blocking at the issue stage. (Note: tensor-II lockout is NOT the driver —
`tensor_add_extra_cycle_initiation_interval` = 14,763, i.e. 0.1% of the 11.5M; the backpressure is the
latch-pipeline structure itself.)

**HW does NOT do this — it warp-switches (NCU kernel-5, per issue_active cycle):**
- `smsp__warps_active = 3.276`, `smsp__average_warps_issue_stalled_not_selected = 0.820` — HW has, on
  average, **0.82 *other* warps eligible-but-not-picked every issue-active cycle.** It always holds spare
  eligible warps and picks a different one when the front warp is stuck.
- HW `dispatch_stall = 0.787` is a **per-warp** stall (that one warp waits for its dispatch port); the
  scheduler still issues a different warp, so the SMSP stays busy → **SM-active 90%**.
- sim's `next_stage` is a **per-subcore** stall (25% of cycles the whole subcore issues nothing, no
  warp-switch attempted) → **SM-active 64%**.

**Conclusion — the mechanical root of "HW compute-dense (90%) vs sim compute-sparse (64%)":** it is not
compute cost and not memory. It is that sim's 1-deep `ISSUE_CONTROL → CONTROL_ALLOCATE` issue pipeline
propagates FU/read backpressure into a **full-subcore stall** and cannot switch to another ready warp
the way HW's scheduler does. The consumer's WGMMA pipeline (6-deep read + lat-32 tensor FU behind 1-deep
latches) is the main source that keeps ISSUE_CONTROL occupied.

**⚠️ Still unconfirmed (needs a run):** whether, in those 11.5M `next_stage` cycles, a *different* warp
was actually eligible (true recoverable head-of-line) vs. the subcore having no other ready warp anyway
(then it is not recoverable). The stat can't tell us because the loop is skipped. **Decisive test:** in
the `else` branch, do a read-only scan of the other warps and count how many were eligible while
ISSUE_CONTROL was full (a new gated counter, timing-neutral) — this sizes the recoverable fraction.
A second test: widen `ISSUE_CONTROL_latch` / `CONTROL_ALLOCATE_latch` to 2-deep and measure (⚠️ changes
timing; may or may not help if the real limit is the read_stage/FU latency behind them).

## Head-of-line instrumentation — IMPLEMENTED (2026-07-20c, gated, run pending)

The decisive test is now implemented (gate `-headofline_instrument_enable`, default 0, timing-neutral,
read-only). On every `next_stage_not_available` cycle, `Subcore::issue()` calls
`scan_head_of_line_when_blocked()` ([subcore.cc:392](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L392), invoked at the next_stage
branch [subcore.cc:906-910](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L904)). It re-scans the subcore's warps
read-only using the same warp-side eligibility as the issue loop (scoreboard / stall_count / wait_barrier
/ yield / ldgdepbar), **excluding FU-side conditions** (a full latch blocks FU entry regardless) and
**never calling** the side-effecting `waiting()` / `warp_waiting_at_tma_flush()` (bit-identity safe).
Files: `shader.h` (gate), `gpu-sim.cc` (option + 6 stat registers), `subcore.{h,cc}`. **Headers changed
→ `make clean`.** Counters (per-SM):
- `total_num_cycles_next_stage_scanned` — next_stage cycles scanned (denominator; ≈ next_stage 11.5M).
- `total_num_cycles_next_stage_with_ready_warp` — of those, cycles with ≥1 warp-side-ready warp =
  **true recoverable head-of-line**.
- `total_num_ready_warps_during_next_stage` — Σ ready warps (avg recoverable warps per HoL cycle).
- `total_num_next_stage_blocked_by_{tensor,mem,other}` — FU class of the instruction holding the latch.

**Read of results (after run):**
- `with_ready_warp / scanned` **high** ⇒ next_stage is largely **recoverable head-of-line**: the 1-deep
  issue latch throws away cycles where another warp was ready → deepen latches / decouple warp-scan =
  real cycle lever (up to ~25% of evaluated cycles). Confirm with the 2-deep latch experiment.
- `with_ready_warp / scanned` **low** ⇒ **do NOT conclude "floor" — ask WHY all warps wait.** This is the
  interesting case: the whole consumer warpgroup is blocked at the same time. The `notready_*` counters
  (below) say on what. HW runs the *same* trace/dependency chain at 90% SM-active, so if sim's warpgroup
  is stuck lockstep, the limit is **over-serialization / lack of warp-stagger in sim**, not an inherent
  dependency floor. FA3 softmax is a serial chain per tile (`S=QK^T` WGMMA → rowmax → `P=exp` MUFU →
  rowsum → `O=PV` WGMMA); HW hides it by staggering the 12 consumer warps across different tiles/stages
  so that while warp A waits on WGMMA, warp B runs MUFU. If sim's 12 warps march in lockstep (all hit the
  same wait_barrier/scoreboard at once), that stagger never forms → the real lever is *why sim doesn't
  stagger* (e.g. GTO always re-picking the same warp, or a barrier that releases all warps together).
- `blocked_by_tensor` dominant ⇒ WGMMA clogs the head of line (6-deep read + lat-32 tensor FU);
  `blocked_by_mem` dominant ⇒ TMA/LDST FU-queue backpressure.

**`notready_*` counters (why the OTHER warps weren't ready; summed over warps × next_stage cycles):**
`notready_{wait_barrier, scoreboard, stall_count, yield, ldgdepbar}` + `valid_head_warps` (denominator =
warps with a valid head seen across all scanned cycles). Read:
- one reason ≫ others AND ≈ `valid_head_warps` ⇒ **lockstep**: nearly every warp blocked on the SAME
  thing every cycle → over-serialization. If it's `wait_barrier` → they all wait on the same mbarrier/
  WGMMA-group; if `scoreboard` → RAW on the same register/result. Either way, fixing warp-stagger (or the
  specific over-conservative dependency) is the lever, NOT latch depth.
- reasons spread out / rotating ⇒ genuine diverse waits, closer to a real dependency floor.

## Other debug counters live in this build (12h-run harvest)

Gates on in the H100 config for this run (all timing-neutral): `-headofline_instrument_enable`,
`-spin_instrument_enable`, `-wgmma_step0_instrument_enable`, `-cta_stall_breakdown_instrument_enable`,
`-sync_wait_hist_instrument_enable`, `-l1i_frontend_step0_instrument_enable`. This one run yields: the
head-of-line split (above), per-CTA `[CTAFIN]` (drain/role/fu_occupied_tensor/wait_pending), the full
issue-stage stall taxonomy, the mbarrier wait-duration histogram, and the spin counters — enough to size
every open axis without a second 12h run. ⚠️ NANOSLEEP knob is back at baseline `1,1` (spin experiment
closed), so cycles are comparable to the tracked baseline modulo the ~0.4% instrument shift.

## Provenance

- sim: `OnlyKernel5/...o45` (fwd, gpu_sim_cycle 137,207; NANOSLEEP-experiment build, spin/step0 gates on).
  The TMA-timeline pattern is a property of the trace + pipeline, not of the (reverted) NANOSLEEP knob.
- HW: `nv_reports/h100/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.ncu-rep`, kernel 5.
- HW trace: `hw_run/traces/device-0/12.8/.../kernel_5/`.
