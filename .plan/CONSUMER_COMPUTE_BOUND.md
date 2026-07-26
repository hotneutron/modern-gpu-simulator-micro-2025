# FA3 fwd — Consumer-Compute-Bound Analysis (2026-07-20)

## ⚠️ UPDATE (2026-07-20b) — the "consumer compute is 1.9× too slow" conclusion below is CORRECTED

A follow-up pipe-level measurement (HW NCU pipe-active vs sim per-pipe fu_occupied) **overturns** the
"consumer per-tile *compute* is 1.9× too slow" reading in the original TL;DR. The truth is the opposite:
**sim's consumer compute is if anything FASTER than HW; the wall is that sim cannot keep the compute
pipes busy, whereas HW packs tensor (46%) + MUFU/softmax (48%) nearly back-to-back.** See the
"Pipe-level breakdown" section at the bottom — that is the current conclusion. The original text is kept
below as the reasoning trail.

---

## Background — SM / SMSP (subcore) hierarchy + warp scheduling (GTO)

This section grounds the terminology used throughout (subcore = SMSP, warp-switch, greedy pointer)
so the head-of-line root cause below is unambiguous. It reflects H100 (SM90) hardware and how this
simulator maps to it (verified in `subcore.cc`).

### Hardware hierarchy (big → small)

```
GPU (H100)
 └─ SM (Streaming Multiprocessor) × 132
     └─ SMSP (SM Sub-Partition) × 4          ← the simulator calls this a "subcore"
         └─ 1 warp scheduler (picks ≤1 warp to issue per cycle)
         └─ functional units: FP / INT / TENSOR / SFU / ...
         └─ resident warps (H100: up to 16 / SMSP)
             └─ warp = 32 threads that always execute in lockstep (SIMT)
```

- **SMSP == subcore.** "SMSP" (SM Sub-Partition) is NVIDIA's term; "subcore" is this simulator's name
  for the same unit. An SM has **4** of them, each an independent scheduler + its own FUs.
- **Warp = 32 threads**, the atomic scheduling unit (not a thread). The scheduler reasons about warps.

### How work maps onto the hardware

- The grid's CTAs are distributed to SMs by the GigaThread engine. **A CTA's warps are then split
  across the SM's 4 SMSPs and stay pinned there for the CTA's lifetime — no warp migration between
  SMSPs, and none between SMs.** A warp's location is fixed at launch.
- **Resident vs executing (the key distinction):**
  - *Resident cap:* 132 SM × 4 SMSP × 16 warp = **8,448 warps** can be *loaded* (≈270K threads;
    equivalently 64 warps/SM). Context-switch cost is **zero** — every resident warp's registers stay
    live in the register file, so the scheduler can swap warps cycle-to-cycle for free.
  - *Executing:* each SMSP issues **≤1 warp/cycle**, so at most 132×4 = **528 warps** issue in any
    single cycle. The other resident warps are **not idle waste — they are the pool the scheduler
    switches to when the front warp stalls** (this is latency hiding, the whole point of high
    occupancy).
- **Waves.** When there are more CTAs than fit at once (e.g. FA3 is 1 CTA/SM due to shared-mem), the
  extras wait in the launch queue and load onto an SM only after a resident CTA finishes. fwd =
  `Waves Per SM 1.00` (132 CTAs, one wave); bwd = `2.91` (384 CTAs over several waves → the causal
  tail imbalance).

### Warp scheduling policy — GTO (Greedy-Then-Oldest), what this simulator uses

Each SMSP scheduler must pick, every cycle, one **eligible** warp among its residents. This simulator
implements **GTO** (verified: [order_greedy_then_highest_id()](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1288)):

1. **Greedy:** keep issuing the *same* warp that issued last (`m_greedy_pointer_issue` is pushed to
   the front of the priority list, [subcore.cc:1291](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1291); refreshed on a successful issue,
   [subcore.cc:839](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L839)) — stick with it until it stalls. Good for **cache/data
   locality** (the hot warp keeps reusing what it just loaded).
2. **Then-Oldest:** once the greedy warp stalls, fall back to the **oldest** warp — the rest are
   sorted by `dynamic_warp_id` ascending (smallest = launched earliest = oldest,
   [subcore.cc:1310](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1310)); `done_exit()`/`waiting()` warps are pushed to the back
   ([subcore.cc:1305-1308](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1305)). Retiring the oldest first frees its
   registers/shared-mem soonest (lets new CTAs in) and prevents starvation. `dynamic_warp_id` (not
   static `warp_id`) is used so warp recycling across waves still reflects true launch order.

GTO is the de-facto GPGPU-Sim standard and matches real NVIDIA scheduler behavior well.

### Two levels of "switch" — where the bug is NOT vs where it IS

| level | what | HW? | this simulator? |
|---|---|---|---|
| **Inter-SMSP** | one SMSP blocked → the other 3 SMSPs still issue | ✅ | ✅ modeled correctly (4 independent `Subcore`s) |
| **Intra-SMSP warp-switch** | one scheduler picks another ready warp when the front warp's FU is busy | ✅ (NCU `not_selected` 0.82, SM-active 90%) | ❌ **the defect** |

**Crucially, the GTO policy itself is correct and HW-faithful — the bug is upstream of it.** The
warp-scan that applies GTO runs **only if** the 1-deep ISSUE_CONTROL latch is free
(`else if(m_ISSUE_CONTROL_latch.has_free())`, [subcore.cc:566](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L566)). When a MUFU is stuck in
that latch (SFU busy for its HW-faithful 8-cyc II), the whole scan is skipped, so GTO **never gets to
pick** the ~2.77 other ready warps. HW would greedy→oldest onto a free pipe (TENSOR/FMA/ALU); the
simulator stalls the entire SMSP instead. The sections below quantify this (25% of fwd subcore-cycles,
99.7% SFU-held, 98.3% recoverable).

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

## ⭐⭐⭐ RESULT (2026-07-20d, bwd `OnlyKernel10/.o30`) — head-of-line is REAL and RECOVERABLE (99.2%)

First run with the counters actually emitted (the `.o48/.o29` run had the counters registered but not
printed — fixed in shader.cc). bwd (gpu_sim_cycle 215,934):

```
next_stage_scanned          = 16,365,885   (== next_stage_not_available exactly ✓ instrumentation sane)
next_stage_with_ready_warp  = 16,229,962   → 99.2% of next_stage cycles HAD a ready warp
ready_warps_during_next_stage = 33,730,794 → ~2.06 ready warps per next_stage cycle on average
valid_head_warps            = 36,065,505
next_stage / evaluated      = 16.37M / 76.67M = 21.3%
```

**Verdict: next_stage is NOT a dependency floor — it is recoverable head-of-line blocking.** On 99.2% of
the cycles where the subcore issued nothing (because the 1-deep ISSUE_CONTROL latch was full), there were
on average ~2 OTHER warps that WOULD have been able to issue if the latch were free. The sim throws these
away because `Subcore::issue()` skips the whole warp-scan when the latch is occupied. This is ~21% of all
evaluated subcore-cycles — a large recoverable lever, not a floor.

**What holds the head of line (`blocked_by_*`):**
```
blocked_by_tensor = 3          (~0 — WGMMA is NOT the clog, contrary to the earlier guess)
blocked_by_mem    = 484,495    (3.0%)
blocked_by_other  = 15,881,387 (97.0%)   <-- fixed-latency non-tensor ops (SP/INT/SFU/UNIFORM/BRANCH)
```
So the instruction stuck in ISSUE_CONTROL is almost always a **plain fixed-latency op**, whose downstream
(CONTROL_ALLOCATE → read_stage(≤3) → FU latency-bitset) is momentarily full, which then blocks the entire
subcore for that cycle even though ~2 other warps were ready. (⚠️ `other` is a coarse bucket — SP/INT/SFU/
UNIFORM/BRANCH combined; splitting it is a cheap next counter but not needed for the verdict.)

**not-ready reasons for the few blocked warps** (only the ~0.8% residual): stall_count 1.58M >
wait_barrier 785K > yield 423K; scoreboard/ldgdepbar 0. Minor — 99.2% were ready.

**Conclusion:** the fwd/bwd 2× "compute-sparse / SM-active 64% vs HW 90%" is driven by the **1-deep
ISSUE_CONTROL→CONTROL_ALLOCATE issue pipeline serializing warp issue**: when the head instruction can't
advance, the subcore stalls instead of issuing a different ready warp (HW warp-switches; NCU
`not_selected`=0.82). Fixing this is a genuine cycle lever (~21% of evaluated cycles in bwd). fwd (`.o48`
re-run) pending but expected to show the same shape.

## ⭐⭐⭐⭐ ROOT CAUSE PINNED (2026-07-20e, fwd `OnlyKernel5/.o56`) — SFU initiation interval = 8

With the root-cause counters emitted, fwd (gpu_sim_cycle 136,069) nails it:

```
next_stage_scanned          = 11,533,614   (== next_stage_not_available ✓)
next_stage_with_ready_warp  = 11,339,290   → 98.3%  (recoverable head-of-line, ~2.77 ready warps/cyc)
next_stage / evaluated      = 11.53M / 46.02M = 25.1%

-- what holds the ISSUE_CONTROL latch (FU class) --
blocked_by_sfu        = 11,498,648  → 99.7% !!!
blocked_by_sp_int_dp  = 22,806      (0.2%)
blocked_by_tensor/mem = 4 / 2,640   (~0)

-- why the latch cannot drain (root reason) --
hol_reason_fu_cannot_issue     = 11,508,436  → 99.8% !!!
hol_reason_control_allocate_full = 25,350    (0.2%)
hol_reason_read_stage_full     = 0           <-- NOT a latch/read-stage depth problem
hol_reason_rf_conflict         = 33,064
hol_reason_fu_latency_full     = 35
```

**Verdict: fwd's 25%-of-evaluated head-of-line is 99.7% the SFU functional unit refusing new ops.**
`functional_unit_sfu::can_issue()` = `m_dispatch_reg->empty()` ([functional_unit.cc:523](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L522)); the SFU
dispatch reg is held for the initiation interval, and the config is **`-sfu_initiation 8` /
`-trace_opcode_latency_initiation_sfu 8,8`** → the SFU accepts only **1 MUFU every 8 cycles**. FA3 softmax
issues a stream of `MUFU.EX2` (HW's busiest pipe, xu 47.75%); in sim they serialize on this narrow SFU,
so 7 of every 8 cycles the head MUFU sits in the 1-deep ISSUE_CONTROL latch with `can_issue=false`, which
stalls the ENTIRE subcore even though ~2.77 other warps were ready. `read_stage_full=0` and
`control_allocate_full≈0` prove it is **NOT** a latch/pipeline-depth problem — it is pure **SFU
throughput (initiation interval)**.

**This reconciles the earlier pipe paradox.** HW xu(MUFU) pipe is 47.75% active (busiest); sim's SFU is
so throughput-starved (II=8) that MUFU can't flow, so sim's SFU fu_occupied read as ~0 while the MUFU
stream instead piles up as head-of-line `next_stage` stalls. The MUFU work didn't vanish — it turned into
issue-stall.

**Fix direction (next: implement, not measure):** lower the SFU initiation interval to match H100's real
MUFU throughput (II=8 is far too coarse for a pipe that HW keeps ~48% active). Candidate:
`-sfu_initiation` / `-trace_opcode_latency_initiation_sfu` init from 8 → 1–2. Expected effect: MUFU
stream flows → SFU stops clogging ISSUE_CONTROL → the ~2.77 ready warps issue → SM-active rises toward
HW 90% → **cycles drop**. ⚠️ This is a *throughput* knob (initiation), the opposite axis from TODO-2
(SFU *latency*, which would slow sim); lowering II should reduce the ratio. Validate with an A/B run and
re-check the head-of-line counters collapse.

⚠️ Physical sanity before committing to II=1: check H100's actual MUFU/SFU throughput per SM (transcendental
issue rate). If HW SFU is e.g. 1 op / 4 threads-worth per cycle, pick the II that matches, don't just
minimize. The point is HW-faithful throughput, not a free speedup.

## ⛔⛔ MAJOR CORRECTION (2026-07-20f) — SFU II=8 is HW-FAITHFUL; the bug is head-of-line, NOT the II value

Checked H100 SFU throughput against hardware (web sources):
- **H100 has 2,112 SFUs / 132 SMs = 16 SFU/SM = 4 SFU per subcore (SMSP).** (NVIDIA specs; SM90.)
- An SFU is scalar; a 32-lane warp transcendental takes `warp_size / SFU_per_subcore = 32 / 4 =` **8
  cycles** to issue one MUFU warp-inst. This is the classic SFU throughput (Fermi onward: 4 SFU ⇒ 32/4).
- So **sim's `-trace_opcode_latency_initiation_sfu` II = 8 is exactly the correct HW throughput** (1 MUFU
  warp / 8 cyc per subcore). It is NOT too big. **Lowering it would be an anti-HW fake win — do NOT do it.**

**This overturns the "lower SFU II" fix proposed above.** The MUFU throughput model is right. The real
bug is what the head-of-line counters actually showed: while the SFU is (correctly) locked out for its
8-cyc throughput window, **~2 other warps are ready but the sim cannot issue them**, because the 1-deep
ISSUE_CONTROL latch holds the stuck MUFU and `Subcore::issue()` skips the whole warp-scan
([subcore.cc:458](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L458)). HW hides the same 8-cyc SFU throughput by
warp-switching (NCU `not_selected`=0.82, SM-active 90%); sim can't, so SFU throughput turns into a
full-subcore stall (SM-active 64%).

**Corrected root cause:** not "SFU II too large" but **"the sim issue stage cannot warp-switch around a
correctly-throttled FU."** The SFU II=8 is the *trigger* (FA3 is MUFU-heavy so it's the FU that most often
occupies the latch), but the *defect* is the 1-deep ISSUE_CONTROL latch serializing issue.

**Corrected fix direction (structural, harder — verify carefully):** make `Subcore::issue()` able to
issue a different ready warp when the head warp's target FU can't accept it, instead of skipping the whole
warp-scan on `!ISSUE_CONTROL_latch.has_free()`. Options: (a) decouple the warp-scan from latch-full and
let a ready warp whose FU *can* accept it issue; (b) widen/parallelize the ISSUE_CONTROL path per-FU. This
is exactly HW's per-SMSP scheduler behavior. ⚠️ NOT a config knob — a scheduler-model change; must keep II
(FU throughput) and latency intact and validate the head-of-line counters collapse without inflating any
FU beyond HW. Also re-confirm this is the same structural issue the async-WGMMA axis hit (WGMMA II is also
HW-faithful; tensor just rarely holds the latch, so it wasn't the visible trigger).



## Mechanism (same II path as WGMMA) — how the latch gets stuck

Traced the SFU issue path end-to-end. It is the **same II mechanism** the async-WGMMA work analyzed:
- `warp_inst_t::issue()` sets **`cycles = initiation_interval`** ([abstract_hardware_model.cc:97](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L97)).
- SFU uses the **base** `functional_unit::cycle()` (functional_unit_sfu overrides only `can_issue`, not
  `cycle`): [functional_unit.cc:306-319](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L306) — `if(!m_dispatch_reg->dispatch_delay())` counts
  `cycles` (=II) down before the op leaves `m_dispatch_reg` into `m_pipeline_reg[latency-1]`.
- `functional_unit_sfu::can_issue()` = `m_dispatch_reg->empty()` ([functional_unit.cc:522](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L522)).
So a MUFU **holds the SFU dispatch reg for `initiation_interval` cycles** → `can_issue=false` for those
cycles → the head MUFU sits in the 1-deep ISSUE_CONTROL latch → whole-subcore head-of-line. This is
exactly WGMMA's `tensor_add_extra_cycle_initiation_interval` re-issue lockout, on the SFU pipe.

**This RESOLVES `WGMMA_FU_OCCUPIED_H100.md` Caveat #1** (which flagged that `fu_occupied` is not
tensor-only and that the tensor-vs-SFU split of the 13.6%/17.75% was an unmeasured hypothesis, esp.
"fwd has 47% MUFU busy on HW"). Measured answer: the fwd/bwd `fu_occupied`/head-of-line clog is **99.7% /
93.8% SFU, tensor ~0**. So the "over-estimated fu_occupied" lever that async-WGMMA chased was actually the
**SFU pipe**, not WGMMA — which is why async-WGMMA (tensor II) was correctly refuted (tensor II wasn't the
clog) while the real clog (SFU II) went unnoticed until this per-FU split.

**Fix safety CONFIRMED (resolves Caveat #3 too).** The concern "does lowering II keep the result/
dependency wait?" is answered by the code: the op is placed at `m_pipeline_reg[latency-1]`
([functional_unit.cc:308](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L308)), so the **result latency is driven by `latency`, independent of the
initiation interval**. Therefore setting `-trace_opcode_latency_initiation_sfu <latency>,1` (keep the
latency the run used — 8 or the HW-anchored 21 — and set **II=1**) preserves the exp result-dependency
wait while pipelining the throughput — a clean, HW-faithful fix, config-only (no rebuild).

⚠️ **Config provenance — RESOLVED (2026-07-20e).** TODO-2 in FA3_progress.md claimed trace-mode SFU
defaults to `4,1` (II=1, not a bottleneck) "because the config never sets
`-trace_opcode_latency_initiation_sfu`". **That is wrong.** The tracked baseline run itself
(`OnlyKernel5/.o37`, gpu_sim_cycle **137,053** — the canonical fwd baseline) dumps
**`-trace_opcode_latency_initiation_sfu 8,8`** (+ `-sfu_latency 21`, `-sfu_initiation 8`). So the real
baseline is **II=8, not 4,1** — SFU is genuinely the clog, and the head-of-line result (`.o56`, 136,069,
same `8,8`) applies directly to the tracked baseline. The primary config file on disk currently shows no
explicit `-trace_opcode_latency_initiation_sfu` line, so the actual value the runs use is injected
elsewhere (job_launching config copy / a merged config) — the on-disk file that TODO-2 inspected was NOT
the config the runs executed with. **Net: the "SFU II=8 is the FA3 root cause" conclusion is VALID; TODO-2
needs correcting (its `4,1`/`fu_occupied_sfu=0` reasoning was based on the wrong config).**


### bwd confirms the same (2026-07-20e, `OnlyKernel10/.o31`, gpu_sim_cycle 216,174)

```
next_stage / evaluated      = 16.36M / 76.84M = 21.3% ;  with_ready_warp = 99.1% (~2.06 ready/cyc)
blocked_by_sfu        = 15,351,266  → 93.8%   (branch_other 3.0%, mem 3.0%, sp_int_dp 0.1%, tensor ~0)
hol_reason_fu_cannot_issue = 16,334,823 → 99.8% ; read_stage_full 0 ; control_allocate_full 0.1%
```
Same verdict as fwd: SFU `can_issue=false` (II=8) is the dominant head-of-line clog (93.8%), not
latch depth. bwd has a little more branch/mem diversity (pre/post-processing) but SFU still dominates.
Both kernels: fix = HW-faithful SFU initiation interval.


## WGMMA vs SFU — SAME structural defect, DIFFERENT magnitude (2026-07-23)

Q (raised this session): the deferred WGMMA axis (`WGMMA_FU_OCCUPIED_H100.md`, FA3_progress.md
"Deferred Opts → async-WGMMA") had the *identical* II/latency/`fu_occupied` story. Is the SFU
head-of-line the same bug? Does HW really schedule around a throttled FU this way? Answer below.

**1. Same code mechanism — confirmed.** Both TENSOR (WGMMA) and SFU (MUFU) ride the exact same path:
- issue sets `cycles = initiation_interval` ([abstract_hardware_model.cc:97](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L97));
- `can_issue()` returns false while the dispatch reg / reserved-cycles hold (SFU:
  [functional_unit.cc:522](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L522); tensor: `m_dispatch_pending_reserved_cycles`);
- result latency is driven separately by `latency` (`m_pipeline_reg[latency-1]`,
  [functional_unit.cc:308](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L308)), so shrinking II preserves the dependency wait on **both** pipes.
So they are two instances of one mechanism, not two different bugs. This is why the SFU finding is
annotated "same II path as WGMMA" in the Mechanism section above, and why it **RESOLVES
`WGMMA_FU_OCCUPIED_H100.md` Caveat #1** (which explicitly flagged that `fu_occupied` was not
tensor-only and that "fwd has 47% MUFU busy on HW" was an unmeasured hypothesis — the SFU split is
that missing measurement).

**2. Two DISTINCT stall sites (this is the whole subtlety).** `Subcore::issue()` has two places a
warp can fail to issue, and WGMMA vs SFU hit them with very different frequency:

| site | code | behavior |
|---|---|---|
| **(A) latch-entry gate** | `else if(m_ISSUE_CONTROL_latch.has_free())` ([subcore.cc:566](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L566)) | latch full ⇒ **whole warp-scan skipped** = per-subcore head-of-line |
| **(B) in-loop FU check** | `!fu->can_issue()` ⇒ bump `fu_occupied`, `continue` | only **that** warp skipped; scan keeps looking |

`WGMMA_FU_OCCUPIED_H100.md` studied (B) (`fu_occupied`) and explicitly noted the loop "skips a
blocked warp rather than stalling the whole SM" (§2). The SFU analysis here found the damage is at
(A): the MUFU stuck in the 1-deep ISSUE_CONTROL latch skips the *entire* scan.

**3. Why WGMMA was a small lever but SFU is a big one — the trigger frequency differs.** Same defect,
opposite magnitude, and the measured counters pin exactly why:

| | WGMMA (tensor) | SFU (MUFU) |
|---|---|---|
| holds the ISSUE_CONTROL latch (`blocked_by_*`, `.o56`) | **~0** (`blocked_by_tensor` = 3–4) | **99.7%** (`blocked_by_sfu` = 11.5M) |
| dynamic II-lockout extension (`add_extra_cycle`, Step-0) | 0.028% fwd / 0.024% bwd (negligible) | — |
| per-subcore `..._reissue_lockout_only` | 4.69% fwd / 10.58% bwd | (folds into 25% next_stage) |
| **SM-level** recoverable (`sm_idle_all_blocked_by_tensor`) | **0.65% fwd / 1.59% bwd** | 25.1% fwd / 21.3% bwd next_stage |

The decisive number is the **SM-level** one. `WGMMA_FU_OCCUPIED_H100.md` (V) found that when one
subcore is locked on the tensor II, **another subcore is almost always issuing**, so the per-subcore
4.69% collapses to 0.65% SM-wide (~7× overcount) → "at most ~0.65–1.59% recoverable" → correctly
**Deferred**. WGMMA simply does not occupy the latch often enough (a warpgroup issues an HGMMA then
moves on; the tensor II rarely sits at the head). **SFU is the opposite:** FA3 softmax emits a dense
`MUFU.EX2` stream (HW's busiest pipe, xu 47.75%), so a MUFU sits at the head of the 1-deep latch
~7 of every 8 cycles → the latch is clogged 99.7% of head-of-line cycles → 25% of evaluated.

**⇒ Reconciliation:** WGMMA and SFU are the SAME structural defect (1-deep ISSUE_CONTROL latch cannot
warp-switch around a correctly-throttled FU). The async-WGMMA / `WGMMA_FU_OCCUPIED` deferral was
**correct** — tensor rarely triggers it, so its recoverable ceiling really is ~1%. The defect only
became a large lever once the *frequently-latch-occupying* pipe (SFU) was measured. WGMMA didn't
"miss" it; its Caveat #1 named the exact unmeasured suspect (MUFU), which this analysis then filled in.

**4. Does HW actually schedule this way? — YES, this is HW's normal latency-hiding, not a sim hack.**
The warp-switch does **not** re-send anything to the (correctly) busy SFU. It issues a *different*
ready warp to a *different, idle* FU (TENSOR / FMA / ALU). Evidence is direct HW measurement (NCU
kernel-5), not a sim assumption:
- `smsp__average_warps_issue_stalled_not_selected` = **0.82** — every issue-active cycle HW has, on
  average, 0.82 *other* eligible-but-not-picked warps it can switch to.
- `dispatch_stall` = 0.787 is a **per-warp** stall (that one warp waits for its port); the scheduler
  issues a different warp meanwhile → SMSP stays busy → **SM-active 90%**.
- Physical basis: H100 SMSP has 4 SFU ⇒ a 32-lane MUFU legitimately ties up the SFU for 8 cyc (II=8
  is HW-faithful, do NOT lower it). During those 8 cyc the TENSOR/FMA/ALU pipes are free, and a
  consumer warp on a different tile/stage runs there. sim's 1-deep latch forbids this (warp-scan
  skipped) → the 8-cyc SFU throughput turns into a full-subcore stall (SM-active 64%).

**5. Open nuance (noted, not yet separately measured).** The SM-level dilution that shrank WGMMA to
0.65% relies on *another subcore* issuing while one is blocked. For SFU that dilution is evidently
NOT saving us (fwd next_stage is 25% SM-wide, per-SM counters). The likely reason: all 4 subcores of
an FA3 fwd SM run the same MUFU-heavy consumer warpgroup, so they tend to hit SFU head-of-line
**in lockstep** (no subcore free to cover). The head-of-line counters are per-SM (already SM-wide),
so the 25% is real SM-level; but the direct "how often are all 4 subcores SFU-blocked simultaneously"
split — the SFU analogue of `WGMMA_FU_OCCUPIED_H100.md` (V) `sm_idle_all_blocked_by_tensor` — has NOT
been isolated yet. If/when instrumentation resumes, that is the one counter that would convert the 25%
head-of-line into an exact recoverable-cycle bound for the scheduler-model fix.

**Bottom line:** same bug, and the WGMMA deferral stands as correct for the tensor pipe. The
scheduler-model fix (let `Subcore::issue()` pick another ready warp when the head warp's FU is busy)
is the shared cure for both pipes; it restores what HW already does (`not_selected` 0.82), keeps every
FU's II/latency HW-faithful, and its payoff is dominated by the SFU trigger, not WGMMA.


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


## Fix design — deterministic-II FUs should be issue-gated (2026-07-24)

Code review of the issue pipeline (`subcore.cc`, `functional_unit.{h,cc}`, `abstract_hardware_model.h`)
pinned the mechanism one level deeper than "1-deep latch": it is that **the FU-availability check is
skipped at issue time for non-fixed-latency FUs**, so an op whose FU is busy still enters the latch and
clogs it. The fix follows directly.

### The two-category split and why SFU is misfiled

`is_fixed_latency_unit()` = `!m_has_queue && !m_is_sfu` ([functional_unit.cc:109](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L109)). In
`Subcore::issue()` the FU is checked **only if fixed-latency** ([subcore.cc:724-727](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L724)):

```cpp
bool is_fixed_latency_inst = fu->is_fixed_latency_unit();
if(is_fixed_latency_inst) { is_fu_available = fu->can_issue(pI); }   // non-fixed: stays true
```

- **Fixed-latency (SP/INT/TENSOR/BRANCH/UNIFORM/MISC_NO_QUEUE):** `can_issue` checked at issue → a busy
  FU makes the warp non-eligible, it is **filtered before the latch**, the scan continues to other
  warps, and the latch stays free. This is why `blocked_by_tensor ≈ 0` — WGMMA never clogs the latch.
- **Queue-based (MEM/TMA/MISC_QUEUE):** `can_issue = m_dispatch_reg->empty()` and
  `functional_unit_with_queue::cycle()` drains `m_dispatch_reg` into the queue every cycle, gated by
  **non-deterministic** RF/memory readiness ([functional_unit.cc:473-504](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L473)). You **cannot know at issue
  time** when it will free, and the queue absorbs bursts — so deferring the check is **correct** for
  these. (User's insight, confirmed in code: variable-latency + burst-absorbing ⇒ can't pre-decide
  block.)
- **SFU is misfiled between the two.** `functional_unit_sfu` sets `m_is_sfu=true`
  ([functional_unit.cc:519](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L519)) so `is_fixed_latency_unit()` is false ⇒ it gets the
  **deferred (queue-style) treatment** — but SFU has **no queue** (`m_has_queue=false`). It uses the
  base `functional_unit::cycle()`, holding `m_dispatch_reg` for a **deterministic** II via
  `dispatch_delay()` (`cycles=initiation_interval` → count down, [abstract_hardware_model.h:1654](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L1654),
  [abstract_hardware_model.cc:97](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L97)). So SFU knows **exactly** when it frees (8 cyc), just like a
  fixed-latency unit — it should be issue-gated, but isn't. This one misfiling is the whole FA3 clog.

### The fix (narrowed by the deterministic-II criterion)

Gate `Subcore::issue()` to check `can_issue` for **deterministic-II FUs**, i.e. `!m_has_queue` (which
now includes SFU) instead of `is_fixed_latency_unit()`. Concretely, at [subcore.cc:725](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L725), under a new
default-off gate, evaluate `is_fu_available = fu->can_issue(pI)` when the FU has no queue (SFU + all
current fixed-latency units), leaving queue-based FUs (MEM/TMA/MISC_QUEUE) on the deferred path
untouched. Effect: a busy-SFU MUFU becomes non-eligible at issue → filtered before the latch → GTO
picks the next ready warp (WGMMA/FMA/ALU) → latch no longer clogs → intra-SMSP warp-switch restored.
II and latency values are **unchanged** (SFU II=8 stays HW-faithful).

> Alternative considered — **reclassify SFU as fixed-latency** (drop the `m_is_sfu` exception): rejected
> as riskier, because `is_fixed_latency_unit()` also routes SFU through CONTROL_ALLOCATE + the structured
> RF-read/latency-bitset path ([subcore.cc:288-359](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L288),
> [functional_unit.cc:306-319](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L306)), a much larger behavior change. Gating only the
> issue-time `can_issue` check is the minimal, targeted change.

### Side-effect audit (issue-gating SFU) — SAFE

Traced every side effect on the issue path (`issue_warp()` → `SM::issue_warp()`) to confirm filtering an
SFU op *before* the latch does not change ordering/correctness. All side effects fire **only on a
successful issue**, so deferring them to the real issue cycle is neutral-or-more-correct:

| side effect | site | impact of issue-gating SFU |
|---|---|---|
| `IBuffer->issued()` (ibuffer pop) | [sm.cc:799](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L799) | more correct — no pop when it didn't actually issue (matches fixed-latency) |
| `func_exec_inst` (functional exec) | [sm.cc:811](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L811) | only delayed to real issue; MUFU still executes → same result |
| `set_num_pending_cycles_with_issue_port_busy` | [subcore.cc:826](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L826) | called only on issue-success → already not called when blocked |
| `reserve_unit` (holds dispatch_reg for II) | [subcore.cc:1283-1284](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1283) | guarded by `is_fixed_latency_inst` → **not called for SFU today anyway**; II is set via `warp_inst.cycles` on `issue()`, so the II mechanism is untouched |
| wait-barrier increment (READ/WRITE) | [subcore.cc:366-375](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L366) | runs in `control_stage()` on the latch occupant; SFU (MUFU.EX2) rarely carries read/write barriers, and if it does the set is merely deferred to real issue (op hasn't executed yet → logically correct) |
| `MBARRIER_OP`/`BARRIER_OP` special handling | [sm.cc:827-849](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L827) | routed to MISC_NO_QUEUE (fixed-latency), **not SFU** → unaffected |

**Verdict: safe for SFU.** The only thing to watch in the A/B is the wait-barrier timing shift (expected
zero-impact since SFU seldom sets barriers) — confirm the mbarrier wait-duration histogram and scoreboard
outcomes are unchanged. Gate default-off must reproduce baseline **bit-identically**.

### Validation plan (unchanged principles: gated, default-off, timing-neutral off)

1. New gate `-intra_smsp_warpswitch_enable` (default 0). Off ⇒ bit-identical to baseline.
2. A/B on fwd (vs `.o56`) + bwd: head-of-line counters must collapse
   (`next_stage_with_ready_warp`↓ toward 0), SM-active 64%→toward HW 90%, cycles drop.
3. HW-fidelity guards: no pipe's `fu_occupied`/tensor-active may exceed HW (NCU); `gpu_sim_insn`
   unchanged (same work); II/latency values untouched.
4. Regression: since only SFU newly gates (queue FUs excluded), memory-bound kernels should be
   unaffected — spot-check one if available.
5. wait-barrier timing: confirm mbarrier histogram / scoreboard results unchanged (side-effect audit).

## ⭐⭐⭐⭐⭐ RESULT (2026-07-26) — FIX WORKS: fwd 2.01×→1.56×, bwd 1.62×→1.35×

A/B run, same build, gate off vs on. Baseline = fwd `OnlyKernel5/.o57` / bwd `OnlyKernel10/.o32`
(`-intra_smsp_warpswitch_enable 0`); fix = fwd `.o58` / bwd `.o33` (gate `1`). All other instrument
gates identical on both sides.

### Cycles — large drop, and it is a REAL (work-invariant) speedup

| kernel | baseline (gate 0) | **fix (gate 1)** | Δcyc | HW | mult before→after |
|---|---:|---:|---:|---:|---|
| **fwd (K5)** | 135,833 | **105,464** | **−22.4%** | 67,696 | **2.01× → 1.56×** |
| **bwd (K10)** | 214,826 | **178,856** | **−16.7%** | 132,901 | **1.62× → 1.35×** |

**Work-invariant proof (not a fake win):** `gpu_sim_insn` **bit-identical** (fwd 455,565,060 both;
bwd 629,211,348 both) and `issue_stage_issuing` **bit-identical** (fwd 16,064,281 both; bwd
22,626,216 both). Same instructions issued, same work — only the issue-*stall* was removed.

### Head-of-line collapse (the mechanism, confirmed)

| metric (fwd) | baseline | fix |
|---|---:|---:|
| `next_stage_not_available` / evaluated | 11,514,572 = **25.1%** | 25,423 = **0.06%** |
| `next_stage_blocked_by_sfu` | 11,480,175 | **0** |
| `issue_stage_issuing` / evaluated (≈ SM-active proxy) | 35.0% | **38.0%** |

bwd same shape: next_stage 21.3% → **1.6%**, `blocked_by_sfu` 15,361,151 → **0**. The SFU clog is
eliminated on both kernels; the ~25% (fwd) / 21% (bwd) of subcore-cycles previously thrown away at the
1-deep latch are recovered.

### Warp-switch effect counters (direct causal evidence)

| counter | fwd | bwd | read |
|---|---:|---:|---|
| `sfu_filtered` (fix fired) | 19,148,406 | 21,412,122 | busy-SFU heads filtered at issue |
| **`other_warp_issued`** (recovered slot) | **6,066,539 (31.7%)** | **5,634,782 (26.3%)** | another warp issued into the freed slot = the warp-switch working |
| `still_idle` | 13,081,867 | 15,777,340 | SFU filtered but no other warp ready that cycle |

So ~27–32% of fix-fired cycles directly recovered an issue slot via intra-SMSP warp-switch.

⚠️ **Note — recovery rate (31.7%) is LOWER than the baseline head-of-line "98.4% had a ready warp".**
Not a contradiction: the baseline "98.4% recoverable / ~2.73 ready warps" was a snapshot of the
*un-fixed* pipeline. Once the fix changes timing, the warp-state distribution shifts — many cycles
that used to show "another warp ready" now have that warp already issued elsewhere (its slot was
consumed earlier), so the residual `still_idle` is genuine low-occupancy idle (1 CTA/SM), not a
recoverable stall. This is consistent with the fix capturing the bulk of the recoverable cycles
(−22% fwd) while the structural 1-CTA/SM idle floor remains. `still_idle` did NOT re-stall the whole
subcore the way the baseline latch-clog did — it just means no work existed that cycle.

### Verdict

The intra-SMSP warp-switch (SFU issue-gating) is a **confirmed cycle lever**: fwd −22.4% / bwd −16.7%,
work-invariant, head-of-line collapsed to ~0, SFU clog eliminated, warp-switch fired and recovered
slots as designed. The residual gap (fwd 1.56×) is now the structural 1-CTA/SM low-occupancy floor
(no eligible warp exists on `still_idle` cycles), not the issue-pipeline defect. Remaining fidelity
check (deferred, not blocking): confirm mbarrier histogram / scoreboard unchanged (SFU rarely sets
barriers, expected zero-impact).
