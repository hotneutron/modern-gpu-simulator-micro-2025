> ⛔ **CLOSED / SUPERSEDED (2026-07-27g).** This doc was the "lever A = warp-stagger / MUFU-lockstep"
> investigation. That hypothesis is **dead** and the framing it was built on has been overturned:
> - warp-stagger (E1 launch-offset) recovers ~0 (mbarrier re-synchronizes each tile; reverted `0b863b0`);
> - the "trace `stall_count` latency" reframe is refuted (`>>=1` decay UNDER-applies the stall — Arch TODO-2);
> - the "sim 64% vs HW 90% SM-active" premise was a definition mismatch (issue-rate vs resident-rate);
> - raw-count re-profile (`peak_sustained=4`) shows the tensor pipe is UNDER-modeled 0.69×, not over — no
>   per-pipe cost is the cause. The residual is **issue-density / pipe-overlap**.
>
> **Active work moved to [`HW_VS_SIM_SFU.md`](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/HW_VS_SIM_SFU.md)**
> (the clean, definition-checked baseline). This file is retained ONLY as the historical record of the
> stagger/lockstep dead-ends + the E1 reverted-commit trail. Do NOT add new findings here.

---

# FA3 fwd — Warp-Stagger / MUFU-Lockstep Axis (lever A) (2026-07-27)

Owner axis for the DOMINANT part of the post-Opt-10 SFU-throughput residual: **MUFU-lockstep**
(fwd 74.6% / bwd 70.2% of `still_idle` cycles). Split out of `.plan/CONSUMER_COMPUTE_BOUND.md` because,
unlike Opt 10 (a local issue-gate fix), this is a scheduling/occupancy-model question with no obvious
low-cost fix and real HW-fidelity trade-offs. **No cycle claim until a verified improvement lands.**

Starting point: post-Opt-10 sim is fwd **1.55×** / bwd **1.34×** vs HW (fwd 105,245 vs 67,696;
bwd 178,389 vs 132,901). This doc is the investigation record for what remains.

## The problem in one paragraph

FA3's consumer warpgroup runs 12 warps of the *same* code: `WGMMA (S=QK^T) → softmax (MUFU.EX2) →
WGMMA (O=PV) → …`. The MUFU-lockstep probe (`.o59`/`.o34`) showed that on ~3/4 of `still_idle` cycles
*every* valid-head warp is on a `MUFU` at once, so all 12 want the one HW-faithful SFU (II=8) and nobody
can issue — and there is no free-pipe (WGMMA/FMA) warp to switch to. HW hides the same 8-cycle SFU
throughput because its warps are spread across different pipe stages (NCU `not_selected=0.82`,
SM-active 90%); the simulator's warps march in phase, so SFU throughput becomes whole-subcore idle.

## Evidence (from the MUFU-lockstep probe, fwd `.o59` / bwd `.o34`)

Still-idle valid-head composition

| still_idle valid-head | FWD | BWD |
|---|---|---|
| head_mufu | 74.6% | 70.2% |
| free-pipe (tensor+other) | 25.4% | 29.8% |
| avg valid-head warps / cycle | 2.716 | 2.155 |

Two facts that shape the axis:
1. **avg valid-head ≈ 2.7 / 2.2**, not ~12. At the stalled moment most of the 12 warps are *not even
   valid-head* — they are themselves parked (their own `waiting()`/mbarrier/stall_count). So lockstep is
   not "12 warps racing on MUFU"; it is "the few warps that ARE eligible are all on MUFU."
2. The lockstep is **re-imposed every tile by the mbarrier**: consumer warps rendezvous at the
   TMA-arrival / WGMMA-wait barrier, get released together, and re-enter the WGMMA→softmax sequence in
   phase. This is why it is structural, not a scheduler-order artifact.

## Step 1 — occupancy floor RULED OUT; the residual is a PHASING problem (2026-07-27, static)

This is the gate the plan required before any code. The verdict is decisive: **it is phasing
(mechanisms 1–2), NOT the occupancy floor (mechanism 3).** Proof below.

### 1a. Resident occupancy is essentially IDENTICAL sim-vs-HW — not a floor

The floor hypothesis (mechanism 3) requires "HW has more resident warps to hide the SFU, which FA3's
SMEM forbids in sim." Measured, that premise is false — both are register-limited to **1 CTA/SM** with
the **same** warps/CTA, so the resident pool per scheduler is the same.

Resident occupancy (how many warps are loaded on the SM) — sim ≈ HW

| quantity | FWD K5 | BWD K10 | source |
|---|---|---|---|
| CTA / SM (limited by) | 1 (regs) | 1 (regs) | sim boot-log `CTA/core = 1, limited by: regs`; NCU `Block Limit Registers = 1` |
| block size (threads) | 512 | 384 | NCU Block Size |
| warps / CTA (block ÷ 32) | 16 | 12 | derived |
| resident warps / scheduler (÷ 4 SMSP) | 4 | 3 | sim `-gpgpu_num_sched_per_core 4` |
| HW theoretical active warps / SM | 16.0 | 12.0 | NCU Theoretical Active Warps per SM |
| HW theoretical occupancy | 25.0% | 18.75% | NCU |
| HW achieved occupancy | 20.14% | 15.02% | NCU |
| HW achieved active warps / SM | 12.89 | 9.61 | NCU |
| HW active warps / scheduler | 3.28 | 2.47 | NCU Active Warps Per Scheduler |
| sim avg valid-head warps / still_idle cyc | 2.72 | 2.16 | probe `.o59`/`.o34` |

Reads:
- sim's resident pool (4 / 3 warps per scheduler) **matches HW's theoretical** (16÷4 / 12÷4). sim
  achieved occupancy is if anything **slightly higher** than HW (sim ~25% vs HW achieved 20.14% fwd),
  because HW loses a little more to its own launch/tail imbalance.
- sim's avg valid-head at the stalled moment (2.72) is close to HW's whole-run avg active-warps/scheduler
  (3.28). The sim is **not** warp-starved relative to HW at these cycles.
- ⇒ Plan Step-1 criterion "sim ≥ HW's eligible pool ⇒ phasing" is **met**. **Occupancy floor is ruled
  out.** The deepest cause is NOT "too few resident warps"; it is that the warps sim *does* have all sit
  at the same pipe stage.

### 1b. The real gap is issue-DENSITY, not occupancy

Same warp pool, but HW issues far more often from it. Decompose the cycle ratio by the standard identity
`cycle = work × (1 / issue-density)`, where issue-density = warp-insts issued per scheduler per cycle
(528 schedulers = 132 SM × 4).

Issue-density decomposition (elapsed-cycle basis) — this is where the gap lives

| quantity | FWD K5 | BWD K10 |
|---|---|---|
| sim warp-insts issued | 16,064,281 | 22,626,216 |
| HW warp-insts issued | 14,539,714 | 19,993,658 |
| sim cycles | 105,245 | 178,389 |
| HW cycles | 67,696 | 132,901 |
| **sim issue-density** (insts ÷ 528 ÷ cyc) | **0.2891** | **0.2402** |
| **HW issue-density** | **0.4068** | **0.2849** |
| issue-density ratio (HW / sim) | **1.407×** | **1.186×** |
| work ratio (sim / HW insts) | 1.105× | 1.132× |
| cycle ratio (sim / HW) | 1.555× | 1.342× |
| identity check: work × density-ratio | 1.105 × 1.407 = **1.555** ✓ | 1.132 × 1.186 = **1.342** ✓ |

Reads:
- The fwd 1.55× cycle gap is **almost entirely** the 1.407× issue-density deficit (work is only 1.10×,
  a trace-vs-HW inst-count definitional diff, already closed as "no lever").
- Restated in words: HW keeps its ~3 warps/scheduler issuing **40.7%** of cycles; sim keeps the *same*
  ~3 warps issuing only **28.9%**. The wall is "how densely the same pool is kept issuing," which is
  exactly the lockstep/phasing effect — not the pool size.

## Why lockstep persists — the two re-synchronization points (source-confirmed)

The plan flagged "the per-tile mbarrier likely re-synchronizes them" as a caveat. Confirmed in code —
both the launch and the per-tile release are **atomic same-cycle** events, and the compute region
between them is fully deterministic, so nothing ever de-phases the warps.

- **Launch is synchronized.** [sm.cc:1100](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1100):
  all warps of a CTA are `init()`'d in one loop, same cycle, **identical `start_pc`**. There is NO
  per-warp launch-latency / start-cycle offset / stagger mechanism anywhere in the remodeling tree
  (searched: no `TB_launch_latency`, `warp_launch`, `launch_delay`, `stagger`, per-warp `start_cycle`).
- **Per-tile mbarrier release is synchronized.** The FA3-relevant async mbarrier flips its phase once in
  [sm.cc:1799](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1799);
  every consumer warp polling that barrier passes on the same cycle
  ([sm.cc:1749](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1749)).
  The classic named/counted barrier does the same via a single cohort mask-clear
  ([shader.cc:4641](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L4641)).
- **Consequence — naive launch-phase stagger (mechanism 2) will be WASHED OUT.** Even if the 12 warps
  start at different offsets, the first per-tile mbarrier re-aligns them within a tile or two. This is
  now a **code-level prediction, not a guess** — so a plain launch-stagger prototype is expected to be
  ~null and should NOT consume a build/run before this is addressed. (Mechanism 2's own caveat, now
  confirmed.)

## "But HW has the same mbarrier" — the mbarrier is NOT where lockstep is created (2026-07-27, user Q + source)

Key user question: HW runs the *same* FA3 kernel, so it has the *same* mbarrier — why isn't HW locked
too? The answer separates the barrier into **arrive** (when each warp reaches the bell) vs **release**
(when waiters are let through), and it decides where a fix may and may NOT go.

| | arrive (per-warp reach time) | release (let-through) | result |
|---|---|---|---|
| HW | **dispersed** (jitter: each warp's prior-tile work ends at a different cycle) | as-arrived / on condition | warps pass one-by-one → stay dispersed |
| sim | **all same cycle** (deterministic prior region) | single shared phase-flip → all pass together | same arrive ⇒ same release = lockstep |

**The mbarrier RELEASE logic is identical in spirit sim-vs-HW and is NOT the culprit** (source-confirmed):
- The FA3 consumer wait is a **per-warp independent pull-poll**, not a cohort wait —
  [sm.cc:1742](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1742),
  [sm.cc:1812](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1812)
  (each warp compares its captured parity against the one shared `barrier.phase`).
- The tile-ready signal is a **single discrete event** (the TMA LOAD completion callback
  [tma_unit_sm.cc:1031](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L1031)
  → `notify_tma_completion` → phase flip [sm.cc:1799](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1799)).
  There is **no loop clearing waiters**; the release check has **zero per-warp latency or stagger**.
- So on the flip cycle, every consumer that is *already polling* releases together — but **only because
  they were all already polling** (i.e. they all arrived together). The release path merely *reveals*
  the lockstep; it does not create it.

**Where lockstep is actually created: arrive-time determinism, upstream of the bell.** The region
between two bells (WGMMA→softmax→WGMMA) runs with zero per-warp variance (see the jitter-source table
below), so 12 warps that leave one bell together reach the next bell together. HW's jitter makes them
reach it apart.

**Consequence — do NOT put the fix in the mbarrier (this rules out mechanism C):**
- Perturbing `recompute_sync_barrier_ready_and_maybe_flip_phase` / the release poll to hand-stagger
  warps = adding behavior HW does not have = a tuning knob (Trap 1). Rejected.
- The correct fix injects **real jitter upstream of arrive** (in the compute or memory region) so warps
  *naturally* reach the bell on different cycles; then the existing per-warp pull-poll release
  ([sm.cc:1742](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1742))
  will let them through dispersed **with no change to the barrier code at all**. This is exactly how HW
  behaves.

## Why "release them 1 cycle apart" does NOT help — and what actually would (2026-07-27, user Q)

A natural objection: "if I just release warp B one cycle late, B still has to wait for the same busy
SFU — no cycle is saved." **Correct for the naive version.** The gain does NOT come from any MUFU
finishing sooner (MUFU throughput is fixed at the HW-faithful II=8 and must stay so). It comes from
**pipe overlap**:

- HW runs tensor pipe **46.1%** + xu/MUFU pipe **47.75%** of SM-active — nearly equal load on two
  *different* pipes.
- **Lockstep (current sim):** all eligible warps are in the SAME stage. During softmax the SFU runs and
  the tensor pipe is IDLE; during WGMMA the tensor pipe runs and the SFU is IDLE. The two loads execute
  **serially** ⇒ time ∝ 46 + 48 = 94.
- **Staggered (HW):** while warp A is on the SFU, warp C is on the tensor pipe ⇒ the two loads run
  **concurrently** ⇒ time ∝ max(46, 48) = 48. The saving is A's 8-cycle SFU shadow being **filled by
  C's tensor op**, not by A going faster. Ceiling ≈ 94/48 ≈ **1.9×** (matches the doc's per-tile 1.9×).

**What the objection correctly implies (the necessary conditions for ANY real gain):**
- A 1-cycle offset is useless — C would still be at nearly the same PC (also softmax), so there is no
  tensor op to overlap. The warps must be offset by roughly a **whole softmax region (tens of cycles)**
  so that when B is on MUFU, C is in the WGMMA region.
- That large offset must **survive the per-tile mbarrier re-alignment** (see the section above), else it
  resets every tile.
- The pool is thin (~3 warps/scheduler): with one warp on the SFU, the other ~2 must be on tensor/other.
  Feasible but fragile — the actual recovered fraction is very likely **well below the 1.9× ceiling**,
  and that "how much below" is precisely what a measurement must decide.

## Feasibility scan of the jitter candidates (2026-07-27, code-only, no build)

Before spending a build, each candidate was checked in source for two things: **(i) does the variance
source exist / how much does it produce**, and **(ii) does it survive the mbarrier** (reach the
consumers as *relative* dispersion). Verdicts below.

### Candidate A — memory-arrival jitter: variance EXISTS but is collapsed at the bell

- **Variance source is already real (no new memory model needed).** TMA LOADs allocate real `mem_fetch`
  and traverse the shared SM→L2→DRAM path
  ([tma_unit_sm.cc:872](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L872));
  completion cycle = when the last sector response lands
  ([tma_unit_sm.cc:968](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L968)),
  not a constant. DRAM is a full DDR model with row-hit/miss (`tRCD`/`tRP`/`tRAS`) and FR-FCFS reorder
  ([dram.cc:662](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/dram.cc#L662),
  [dram_sched.cc:143](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/dram_sched.cc#L143));
  L2/ICNT add queue-occupancy-dependent contention. Per-transfer latency genuinely varies.
- **But it is washed out at the mbarrier — two compounding reasons:**
  1. the phase flips only when `completed_tx_bytes >= expected_tx_bytes`
     ([sm.cc:1799](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1799)),
     i.e. release is gated on the **slowest** transfer of the phase — the spread across tiles collapses
     to one edge;
  2. all consumers read the **one shared `barrier.phase`**
     ([sm.cc:1812](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1812)),
     so the flip carries **zero per-consumer arrival timestamp** — memory jitter moves *when the group
     releases*, but produces **zero relative dispersion among consumers**.
- **Verdict: A alone does NOT de-phase consumers.** Adding more raw memory variance changes nothing; the
  block is the shared-phase release semantics, not a lack of variance. A would only work if individual
  consumers observed *their own* tile's arrival cycle (a per-warp arrival timestamp) — a barrier-model
  change that borders on mechanism C (knob risk).

### Candidate B — bank-conflict / RF-port jitter: models are LIVE but warp-symmetric ⇒ non-dispersing

- **Both models already exist and are active** (not deferred):
  - shared-mem bank conflict sets a variable initiation interval `cycles = total_accesses`
    ([abstract_hardware_model.cc:787](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L787),
    consumed at [ldst_unit_sm.cc:1867](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/ldst_unit_sm.cc#L1867));
  - RF read/write bank-port contention can extend effective II via `add_extra_cycle_initiation_interval`
    ([register_file.cc:185](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/register_file.cc#L185),
    [subcore.cc:319](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L319)).
- **But both are a deterministic function of the per-warp (register / shared-address) access pattern**,
  and FA3's consumer warps run identical SASS with identical layout ⇒ identical conflict count ⇒
  **identical delay for every symmetric warp** ⇒ phase is preserved, not dispersed. The models add
  variance *between different instructions*, never *between symmetric warps*.
- **Dispatch-port / FU arbitration adds no drift either:** a loser on a busy SFU just fails `can_issue`
  and retries; the wait is exactly the remaining fixed II (phase-*preserving*), not a contention-length
  penalty ([functional_unit.cc:125](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L125)).
  WGMMA/MUFU latency+II are pure per-opcode constants
  ([cuda-sim.cc:1009](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc#L1009));
  no RNG exists anywhere in the compute path.
- **Verdict: the compute region between bells is effectively fully deterministic per-warp.** B requires
  a **NEW** variance mechanism — specifically one where the conflict/stall depends on **dynamic
  cross-warp contention** (two warps hitting the same bank/port in the same cycle, one delayed by an
  amount that depends on the *other* warp), not on each warp's own static pattern. Only that produces
  real inter-warp dispersion, and it must also survive the bell (it does, since it lives inside the
  compute region — B's one structural advantage over A).

### Candidate C — inject per-warp spread at mbarrier release: RULED OUT

- Adds behavior HW does not have, directly in the release path — a tuning knob by construction (Trap 1),
  and contradicts the arrive-vs-release finding (the release path is faithful; lockstep is upstream).
  **Rejected.**

### Scan summary

| candidate | variance source today | survives mbarrier? | needs new code? | verdict |
|---|---|---|---|---|
| A memory-arrival | **exists** (real DRAM/L2/ICNT) | **NO** — collapsed by shared-phase + slowest-transfer gate | yes (per-warp arrival observation ≈ barrier change) | weak / borders on C |
| B bank / RF-port | **exists but warp-symmetric** ⇒ 0 inter-warp dispersion | **YES** (lives in compute region) | yes (dynamic cross-warp contention model) | **most faithful + wash-out-resistant, but real build** |
| C mbarrier-release spread | none (would be injected) | n/a | yes | **rejected (knob)** |

**Overall:** no candidate is a "flip an existing knob" win. A's variance exists but can't reach
consumers without a barrier change; B's models exist but are warp-symmetric so add zero dispersion; C is
a knob. The only faithful, wash-out-resistant path is **B extended to model dynamic cross-warp
bank/port contention** — a genuine new mechanism, sizeable build, with the ceiling still bounded by HW's
own 54% No-Eligible. This sharpens the plan's "most invasive / most fidelity-sensitive" label with the
exact reason.

## Candidate mechanisms (unproven — this is the open design space)

Ordered by how faithful-to-HW they are, not by ease:

1. **Execution-time jitter / non-deterministic latency.** HW warps de-phase because per-warp memory
   arrival, bank conflicts and FU latencies vary slightly; a big resident-warp pool absorbs it so some
   warp is always at a non-MUFU stage. The simulator is largely deterministic and FA3 runs 1 CTA/SM, so
   there is little pool and no jitter to de-phase. Modeling this is the most HW-honest but the most
   invasive and the most fidelity-sensitive. **Now the leading candidate** (mechanisms 2/3 closed
   below); design constraints in the next section.
2. **Launch-phase stagger.** Start the 12 consumer warps at slightly different offsets. **Downgraded to
   ~null by the source finding above** — the per-tile mbarrier re-synchronizes them, so this is expected
   to wash out within a tile. Not worth a standalone build unless combined with a mechanism that resists
   re-alignment.
3. **Occupancy floor.** **RULED OUT** (Step 1, §1a): sim and HW have the same 1 CTA/SM and the same
   warps/CTA, so sim is not warp-starved relative to HW. The residual is phasing, not too-few-warps.

## What "execution jitter" is, precisely — and the two traps (2026-07-27, user Q)

Definition: **jitter = the same instruction taking a slightly different number of cycles per warp, per
occurrence, so warps that started together drift out of phase over time.** Analogy: 12 workers leave on
one bell; in a real factory each walks at a microscopically different pace (a part arrives 0.3 cyc late,
a machine seat is contended) so after a few laps they naturally spread across stations; in sim every
worker is a robot at identical pace, so they stay in lockstep forever.

Physical sources of HW jitter (NOT hand-wave), and how sim models them today (confirmed by the scan above)

| HW source | why it differs per warp | sim today |
|---|---|---|
| memory arrival latency spread | DRAM row-buffer hit/miss (~2×), bank/channel queueing, L2-slice contention, ICNT arbitration — tens of cycles of per-warp variance | variance is REAL (full DDR + FR-FCFS + ICNT), but collapsed at the mbarrier (shared phase + slowest-transfer gate) ⇒ 0 relative dispersion |
| shared-memory bank conflicts | conflict count depends on the per-warp access pattern | model is LIVE (`cycles=total_accesses`) but a deterministic fn of the pattern ⇒ identical for symmetric warps ⇒ non-dispersing |
| FU + dispatch-port arbitration | two warps wanting the SFU: one wins, one waits; the wait depends on timing | fixed II countdown; loser waits exactly remaining II ⇒ phase-preserving, zero drift |
| register-file bank conflicts | read/write port contention is variable | model is LIVE (extra-II) but deterministic per pattern ⇒ identical for symmetric warps ⇒ non-dispersing |

Why sim is lockstep, restated at the mechanism level:
1. the compute pipes are **fully fixed-latency** (SFU lat=21/II=8, tensor lat=32 — always identical), so
   the region between two bells has zero variance;
2. the mbarrier **atomically re-aligns** every tile ([sm.cc:1799](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1799)),
   so even small upstream memory jitter is washed out.

The two traps that make this the "most fidelity-sensitive" mechanism:
- **Trap 1 — jitter becomes a tuning knob.** Injecting `+rand()%N` per warp lets one dial N to hit the HW
  cycle count = curve-fitting = a fake win. The guardrail forbids this. To be HW-honest the jitter must
  **emerge as a consequence** of an independently-correct physical model (real DRAM row-buffer variance,
  real bank-conflict counts), NOT be a value the modeler injects.
- **Trap 2 — the mbarrier washes it out.** Jitter applied only inside the compute region resets at each
  bell (same failure as launch-stagger). To persist, the jitter must either accumulate faster than the
  bell re-aligns, or enter the mbarrier release itself as a per-warp arrival spread — which again
  requires a real mechanism behind that spread, not an injected offset.

Design decision this forces (owner's call, not the modeler's): **which physical mechanism sources the
jitter.** The three candidates (A memory-arrival / B bank-conflict / C mbarrier-release) were scanned in
source — see "Feasibility scan of the jitter candidates" above for the per-candidate verdicts. Net: A's
variance exists but is collapsed at the bell, B's models exist but are warp-symmetric (need a new
dynamic cross-warp contention model), C is a rejected knob.

## What HW actually does — reference facts (do NOT hand-wave)

- NCU kernel-5: `smsp__warps_active ≈ 3.28`, `not_selected = 0.82`, SM-active ~90%, xu(MUFU) pipe 47.75%,
  tensor pipe 46.1%. `One or More Eligible = 45.74%` / `No Eligible = 54.26%`.
  ⇒ HW is *itself* partly SFU/dependency-bound — on 54% of active cycles HW ALSO has no eligible warp.
  So a meaningful part of sim's lockstep idle is a **faithful** reflection of the workload, and the
  achievable ceiling from de-phasing is bounded (≈1.9× per-tile at most, less in practice), not open.
- The SFU II=8 is HW-faithful (4 SFU/subcore) and must NOT be lowered (see Opt 10). This axis is about
  *overlap*, never about making the SFU wider.

## Plan / next steps

1. ✅ **DONE — occupancy floor ruled OUT (Step 1).** sim resident warps/SMSP (4 fwd / 3 bwd) match HW
   theoretical; sim achieved occupancy ≥ HW achieved. The gap is issue-density (1.41× fwd), i.e. phasing,
   not pool size. See §1a/§1b.
2. ⚠️ **Launch-phase stagger (mechanism 2) downgraded — do not spend a build on the naive version.**
   Source shows launch + per-tile mbarrier are atomic same-cycle, so a plain start-offset washes out.
   Confirmed prediction, not a guess.
3. ✅ **DONE — jitter feasibility scan (code-only, no build).** See "Feasibility scan" + arrive-vs-release
   sections. Findings: the mbarrier release path is faithful (lockstep is upstream arrive-determinism, not
   release) ⇒ **C rejected**; memory variance is real but collapsed at the bell ⇒ **A can't reach
   consumers without a barrier change**; the bank/RF models are live but warp-symmetric ⇒ **B needs a new
   dynamic cross-warp contention model**. No knob-flip win exists.
4. **Decision pending (owner): whether to build candidate B.** It is the only faithful,
   wash-out-resistant path, but it is a genuine new mechanism (model bank/port conflict as *dynamic
   cross-warp* contention so two warps colliding in the same cycle drift apart), a sizeable build, with
   the payoff still bounded by HW's own 54% No-Eligible (ceiling ≈1.9× per-tile, likely much less with a
   ~3-warp pool). Recommended pre-build step: a gated, read-only probe that measures the *potential*
   dispersion (how often ≥2 symmetric consumer warps hit the same shared-mem bank / RF port in the same
   cycle) to size B's ceiling before committing to the timing change.
5. **If B's probe shows the collision rate is low (little dispersion to be had):** stop and write it up as
   a structural floor in FA3_progress.md — occupancy floor is out, but "deterministic-model lockstep,
   bounded by HW's own 54% No-Eligible" is itself a defensible floor — make no cycle claim, and consider
   the compute-path axis closed.

## E1 — oracle de-phase probe: RUN DONE → NEGATIVE (mbarrier washes it out) (2026-07-27)

> ⛔ **RESULT: de-phasing recovers ~0 cycles; the per-tile mbarrier re-synchronizes the warps exactly as
> predicted. Lever A is a structural floor. The E1 oracle code was REVERTED per the up-front disposition
> (kept only in git history + this doc).** Details below; the design/run-plan text is retained as the record.

### Measured result (fwd `.o60` / bwd `.o35`, `-oracle_dephase_enable 1`, stride=16)

| metric | FWD K5 baseline (`.o58`) | FWD E1 (`.o60`) | BWD K10 baseline (`.o33`) | BWD E1 (`.o35`) |
|---|---:|---:|---:|---:|
| `gpu_sim_cycle` | 105,464 | **105,862 (+0.38%)** | 178,856 | **178,724 (−0.07%)** |
| `gpu_sim_insn` (work) | 455,565,060 | 455,565,060 (bit-identical) | 629,211,348 | 629,211,348 (bit-identical) |
| `head_mufu / valid_head` | 0.746 | **0.745 (unchanged)** | 0.702 | **0.703 (unchanged)** |

**Verdict — decisive NEGATIVE, both diagnostics agree:**
1. **Cycles did not improve** — fwd +0.38% (slightly *worse*), bwd −0.07% (noise). Against the ≈1.9×
   per-tile overlap ceiling, the recovered fraction is **zero**. `gpu_sim_insn` bit-identical confirms the
   oracle ran correctly (work invariant), so the null is real, not a measurement error.
2. **Lockstep re-formed** — `head_mufu/valid_head` moved 0.746→0.745 (fwd) / 0.702→0.703 (bwd), i.e. it
   **did not budge**. The launch-phase offset was fully erased by the first tile's mbarrier: the warps
   arrived staggered exactly once, then the atomic phase-flip release
   ([sm.cc:1799](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1799))
   re-collapsed them into phase. This is the **direct empirical confirmation** of the code-level
   "mbarrier washes out any launch stagger" prediction (arrive-vs-release section).

**Why a stride sweep (8/32) was NOT run:** the failure is not a wrong stride — `head_mufu` not moving at
all means the offset is annihilated at tile 1 regardless of its size. No stride survives one mbarrier, so
sweeping cannot change the verdict.

### Disposition executed: E1 code REVERTED

Per the up-front decision (below), the oracle produced ~null / washed-out ⇒ the probe did its job
(proved the floor) and the knob must not linger. The 4 source+config edits from commit `0b863b0`
(`shader.h`, `gpu-sim.cc`, `subcore.cc`, `gpgpusim.config`) were removed — those files are now
byte-identical to the pre-E1 state (`e9a64e8`). Only this analysis doc is kept. `0b863b0` remains in
history as the reproducible source state that produced `.o60`/`.o35`.

### Conclusion for the STAGGER sub-lever: dead. (The gap itself is NOT closed — see reframe below.)

Combined with Step 1 (occupancy floor ruled out) and the feasibility scan:
- **not occupancy** (sim resident warps == HW theoretical);
- **is phasing** (issue-density 1.41× fwd), but **phasing is not recoverable by staggering** because the
  per-tile mbarrier deterministically re-synchronizes the warps (E1 proved it empirically);
- so warp-stagger / launch-offset / jitter-to-de-phase (mechanisms 1–3) are all **dead ends** for this
  workload — the mbarrier annihilates any stagger.

⇒ **The warp-stagger *sub-lever* is closed.** But this does NOT make the residual a floor: the
issue-density gap (1.41×) is real and still unexplained by throughput. The very next step
(2026-07-27b) re-examined *why* the warps stall at all and found it is the trace `stall_count`
latency, not SFU throughput — a different, still-open axis. **Do not read this section as "axis closed."**
See "Reframe" below.

## Reframe (2026-07-27b) — the residual is NOT SFU-throughput; it is trace `stall_count` latency

The user pushed back on closing as a floor ("issue is far too low vs HW — think differently"). Re-examining
the numbers with a fresh lens overturns the "SFU-throughput bound" label that Opt 10 left in place.

### SFU is NOT saturated (demand/capacity ≈ 28%)

- SFU capacity over the run = `eval_subcore_cyc / II` = (105,862 × 132 × 4) / 8 = **6.99M** MUFU slots.
- HW MUFU demand (from NCU xu 47.75% over SM-active) ≈ **1.93M** MUFU warp-inst; sim runs the same trace
  (work-invariant) so its MUFU count is the same order.
- ⇒ **demand/capacity ≈ 27.6%.** A throughput-bound pipe sits near 100%. **SFU is nowhere near saturated**,
  so "SFU-throughput bound" (the Opt-10 residual label) is **wrong** — the warps are not queued on SFU
  *capacity*. They are stalled for some *latency* while the SFU pipe sits mostly idle.

### Source: the post-MUFU stall is the trace-encoded `stall_count`, not the SFU FU latency

Traced the dependency model for FA3 (trace mode, `is_captured_from_binary=1`,
`-is_remodeling_scoreboarding_enabled 0`):
- The register scoreboard is **OFF** for these traces
  ([subcore.cc:711](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L711)),
  so a dependent op's readiness is gated by the **per-warp `stall_count`**, not by a scoreboard bit tied
  to SFU pipeline retirement.
- `stall_count` is set from the issuing instruction's SASS control word
  (`get_stall_count()`, [sm.cc:918](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L918))
  and decayed **by a right-shift** `m_stall_counter >>= 1` each cycle
  ([warp_dependency_state.cc:92](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/warp_dependency_state.cc#L92)),
  checked via `is_stall_counter_0()` in the issue predicate.
- **Consequence:** changing `-sfu_latency` / the SFU FU `latency` field does **NOT** move this stall
  (scoreboard off ⇒ FU-pipeline retirement does not gate the dependent). II (=8) is only a MUFU→MUFU
  throughput lever, and we just showed throughput isn't the wall. **The load-bearing quantity is the
  trace `stall_count` and how the sim decays it.**

### Why this is the real axis (and why stagger was a red herring)

- The `stall_count` is HW's compiler-computed static-scheduling delay (≈ the result latency the SASS
  scheduler budgeted after each op). On HW those cycles are hidden by *other* warps; in the sim's lockstep
  all eligible warps serve the *same* `stall_count` window at once ⇒ subcore idle. This is why still_idle
  co-occurs with "all heads MUFU" (they just issued MUFU and are all serving its post-stall) — but the
  cause is the **stall, not SFU capacity**.
- Stagger (E1) can't help because `stall_count` rides *every* instruction regardless of warp phase, and
  the mbarrier re-aligns phase anyway. So E1's null is consistent — it just wasn't the right lever.

### `>>=1` decay AUDIT — DONE (2026-07-27c): the decay UNDER-applies the stall ⇒ stall_count is NOT the inflator

Static audit of the decay rule (no build). Findings:
- `stall_count` is a 4-bit trace field (0..15), decoded `control_bits_num & 0x0000f`
  ([control_bits.cc:34](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/util/traces_enhanced/src/control_bits.cc#L34)).
  HW semantics: value V ⇒ wait exactly V cycles (linear) before the same warp issues next.
- sim decays it with a **right-shift** `m_stall_counter >>= 1` per cycle
  ([warp_dependency_state.cc:92](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/warp_dependency_state.cc#L92)),
  gated by `is_stall_counter_0()`. A shift blocks the warp for only `floor(log2 V)+1` cycles:

  | V | `>>=1` blocked cyc | HW linear |
  |---|---|---|
  | 1 | 1 | 1 |
  | 2 | 2 | 2 |
  | 4 | 3 | 4 |
  | 8 | 4 | 8 |
  | 15 | 4 | 15 |

- **Verdict — this REFUTES the "trace stall_count latency" reframe as the inflator.** The shift makes the
  sim hold a stalling warp for FEWER cycles than HW (V=8 → 4, V=15 → 4). If `stall_count` were the source
  of the excess cycles, the sim would be *faster* than HW, not 1.56× slower. A too-short stall cannot
  inflate cycles.
- **What the `stall_count`-dominated `still_idle` share actually is.** `.o59`:
  `at_least_one_warp_waiting_stall_count` = 6.58M ≈ 30% of `no_warps_ready` (21.62M) — the largest single
  wait reason, but it co-occurs with `wait_barrier` 5.49M (25%), non-SFU fu_occupied (~20%), yield 1.98M
  (9%): still_idle is **multi-factor, not a single stall_count axis**. Since the stall itself is short,
  the 30% share is a **lockstep symptom** (all consumer warps serve their short post-MUFU stall window at
  once ⇒ the whole subcore idles that window), not a stall-magnitude problem.
- **Disposition**: recorded as a known modeling gap in FA3_progress.md **Arch TODO-2**, and parked —
  making the decay linear (`--`) would *lengthen* stalls and make the sim slower (wrong direction), so it
  is not a cycle lever. The residual axis is the consumer-warp lockstep / pipe-overlap absence, NOT the
  stall latency. (This also downgrades the "two open sub-questions" below: sub-question 1 is answered —
  the decay is not HW-faithful but that under-applies the stall, so it is not the lever; sub-question 2,
  GTO re-picking a stalling warp, remains open but is a lockstep/scheduling property, not a stall axis.)

### Two open sub-questions (un-probed — this is the new axis)

1. **Is the `>>=1` decay HW-faithful?** A shift makes a stall field V last only `floor(log2 V)+1` cycles
   (V=8 ⇒ 4 cycles), i.e. it *under*-applies vs a linear countdown. If the sim is nonetheless slower than
   HW, the `stall_count` *values* in the trace (or their interaction with lockstep) must be the inflator —
   needs a per-still_idle-cycle measurement of how many warps are blocked *specifically* on `stall_count`
   vs everything else.
2. **Does GTO make it worse by re-picking a stalling warp?** If the greedy pointer keeps selecting a warp
   that is serving its `stall_count` (instead of another warp whose stall already decayed), the sim wastes
   issue slots HW would fill. This is a *scheduler* question distinct from occupancy/stagger.

### Next step — measure, don't assume (gated read-only probe, needs a build)

The existing `mufu_lockstep_nonsfu_stall_count` counter only tallies non-SFU heads. Add a per-still_idle
decomposition: on each still_idle cycle, of the valid-head warps, how many are blocked **only** by
`stall_count` (would issue if stall were 0) vs by wait_barrier vs by a genuinely busy non-SFU FU. If
`stall_count`-only dominates, the lever is the stall model / its decay (sub-question 1) — a compute-latency
fidelity axis, testable by A/B on the decay rule — NOT a floor. Only if the residual is genuinely
wait_barrier (producer→consumer data dep) does the floor conclusion hold.

**Until this probe runs, do NOT record lever A as a closed floor in FA3_progress.md** — downgrade it to
"reframed: trace-stall_count latency axis, probe pending".

---

## E1 design + run plan (retained as the record)

**Purpose.** The one big unknown is the *upper bound*: if the lockstep consumer warps WERE de-phased,
how many cycles would pipe-overlap actually recover — and does the per-tile mbarrier wash the stagger
out? Static analysis can only bound this to "≈0 … 29%", too wide to decide whether to build the
expensive faithful jitter model (candidate B). E1 measures it directly with an **oracle** (deliberately
NOT HW-faithful — a measurement instrument, not a fix): force a one-time per-warp launch-phase offset
and read the cycle delta + whether lockstep re-forms.

**Mechanism (gated `-oracle_dephase_enable`, default 0 = bit-identical off).** The first cycle a warp is
otherwise fully eligible to issue, arm its release cycle = `now + (cta_local_warp_index * stride)`; hold
the warp out of issue until then. Arming is **one-time per warp** (never re-armed) so it is a pure
*launch* offset — this is exactly what tests whether the per-tile mbarrier re-synchronizes the warps
(if it does, the stagger vanishes after tile 1–2 and cycles barely move; if it survives, cycles drop).
`gpu_sim_insn` stays bit-identical (same work, only issue timing shifts).

- Files: `shader.h` (2 config fields + per-warp state + `arm_oracle_dephase` / `is_oracle_dephase_ready`
  helpers, cleared in `reset()`), `gpu-sim.cc` (2 options), `subcore.cc` (arm + gate in the issue-scan,
  right after `are_switch_warp_conditions_ready`), `gpgpusim.config` (`-oracle_dephase_enable 0`,
  `-oracle_dephase_stride 16`). **Headers changed → `make clean` before rebuild.**
- Offset uses CTA-local warp index (`warp_id % warps_per_cta`) × stride, so warp 0 has 0 offset (issues
  immediately) and later warps stagger by `stride` cycles each.

**How to run (A/B, needs a build — user builds/runs).**
1. Rebuild after `make clean` (headers changed).
2. Baseline A = current tracked config (`-oracle_dephase_enable 0`) — must reproduce `.o58`/`.o33`
   bit-identically (sanity that the gate is truly off).
3. Fix B = set `-oracle_dephase_enable 1`, sweep `-oracle_dephase_stride` over **8 / 16 / 32** (roughly
   fractions of the softmax-region length). Run fwd (K5) + bwd (K10).

**Read of results.**
- `gpu_sim_cycle` B vs A: the actual recovered fraction (the number the whole axis hinges on). Compare
  against the ≈1.9× per-tile ceiling — expect **much less** if the mbarrier re-synchronizes.
- `gpu_sim_insn` must stay **bit-identical** A vs B (else the oracle changed work — a bug, not a result).
- `mufu_lockstep` probe counters (already on): if `head_mufu / valid_head` **drops** under B, the stagger
  took hold; if it stays ~0.75, the mbarrier washed it out (decisive for the "washes out" prediction).
- `intra_warpswitch_other_warp_issued / sfu_filtered` (recovery rate): should **rise** if de-phasing
  gives free-pipe warps to switch to.

**Decision rule.**
- B shows a **sizeable, stride-robust** cycle drop AND `head_mufu/valid_head` falls ⇒ phasing is a real
  lever; candidate B (faithful dynamic cross-warp contention jitter) is worth building.
- B shows **~null** cycle drop or the gain evaporates as the mbarrier re-aligns (head_mufu stays high)
  ⇒ the residual is a structural deterministic-lockstep floor; write it up in FA3_progress.md, no cycle
  claim, close the compute-path axis. Either outcome is decisive — that is why E1 is worth the one build.

**Disposition of the E1 code after the run (decided up front, 2026-07-27).** E1 is a throwaway
measurement instrument, not a keeper. It is committed now only so the A/B build is reproducible and the
result is traceable to an exact source state. After the run:
- **If B is a strong, faithful-lookalike win** (sizeable + stride-robust + head_mufu drops): keep the
  commit as the reference point, then build the *faithful* candidate B on top; E1 itself still does not
  ship (it is an injected oracle, not emergent physics).
- **If B is ~null, OR it "works" only by making the sim much slower / needs an implausibly large stride,
  OR gpu_sim_insn is not bit-identical:** the probe has done its job (proved the floor / no faithful
  lever) — **revert the E1 commit** so the oracle knob does not linger in the tree, record the negative
  result + the reverted-commit hash in FA3_progress.md, and close the axis. Do NOT keep a de-phase knob
  that only helps by being unfaithful (that is exactly Trap 1 / a tuning knob).

⚠️ E1 is an **oracle measurement**, not a shippable optimization: the launch offset is injected, not
emergent, so even a positive result does NOT authorize a cycle claim — it only authorizes building the
*faithful* mechanism (candidate B) that would produce the same de-phasing from real physics.

## ⭐ Active-definition MISMATCH — the "sim 64% vs HW 90% SM-active" framing is apples-vs-oranges (2026-07-27d)

The whole "sim can't fill the SM (64%) the way HW does (90%) ⇒ lockstep" narrative rested on comparing
two metrics that **measure different things**. Verified against both the sim source and the H100 NCU report
(kernel 5, `nv_reports/h100/...ncu-rep`, read in-container via `ncu -i`).

### The two definitions
- **sim "SM-active" (≈64% fwd) = ISSUE rate.** [sm.cc:612](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L612):
  `if (!any_subcore_issued) sm_all_subcores_idle++;` — a cycle is "idle" unless ≥1 of the 4 subcores
  **actually issued** an instruction. The tracked `issuing` proxy (fwd 38%) is likewise an issue count.
- **HW `sm__cycles_active` (90%) = RESIDENT rate.** NCU "SM Active Cycles" 61,147 / elapsed 67,838 =
  90.1% counts every cycle a warp is **resident** on the SM, whether it issues or stalls. The SAME NCU
  report states **`No Eligible = 54.26%`** ("every cycle with no eligible warp results in no instruction
  being issued") and **`Issue Slots Busy = 45.03%`**, `Issued Warp/Scheduler = 0.46`.

### The correct correspondence
| basis | sim | HW | ratio |
|---|---|---|---|
| **issue rate** (apples-to-apples) | `issuing` ~38% | **`Issue Slots Busy` 45%** | ~1.18× |
| resident rate | ~100% (1 CTA/SM, minus drain tail) | `sm__cycles_active` 90% | ~1× |
| issue-DENSITY (insts÷sched÷cyc, def-consistent) | 0.289 | 0.407 | **1.41×** |

So "64% vs 90%" was never a valid gap — it compared sim's *issue* rate against HW's *resident* rate. The
only def-consistent gap is **issue-density 1.41×** (which the doc computed correctly elsewhere).

### Consequence for the lockstep framing
- HW spends **54% of its "active" cycles with No-Eligible** (eligible warps/scheduler = **0.83** fwd /
  **0.46** bwd). HW is **itself** mostly-waiting on this workload — the same phenomenon as sim's lockstep,
  not a sim-only pathology. HW just books those waiting cycles as "active" (warp resident) while sim books
  them as "idle" (no issue).
- ⇒ the achievable ceiling from de-phasing is even more tightly bounded than the doc's "≈1.9× per-tile":
  HW's own No-Eligible 54% is the wall. Lockstep is a **real but small-ceiling** effect, not the 1.4×
  driver by itself.
- The def-consistent residual is **issue-density 1.41×**: sim issues 1.41× less often per elapsed cycle.
  Since occupancy matches and de-phase is bounded, the live question is now **WHERE the missing issues
  go** — cross the sim `no_warps_ready` sub-reasons against HW's stall breakdown (below), which is a
  latency/stall axis, not purely lockstep.

### HW stall breakdown (per issue_active, NCU raw) — the map for the next step
| stall reason | HW fwd | HW bwd |
|---|---|---|
| `wait` (mbarrier / async / WGMMA.wait) | **1.363** | 0.784 |
| `long_scoreboard` (global/L2 latency) | 0.700 | **1.491** |
| `barrier` (named/CTA barrier) | 0.782 | **1.313** |
| `not_selected` | 0.820 | 0.405 |
| `dispatch_stall` | 0.787 | 0.338 |
| `mio_throttle` (MIO/shared/SFU queue) | 0.457 | 0.474 |
| `short_scoreboard` (MIO/SFU latency) | 0.330 | 0.643 |
| `math_pipe_throttle` | 0.229 | 0.092 |
| pipe active: tensor / xu(MUFU) / alu / fma | 46.1 / 47.7 / 27.3 / 16.9 | 53.6 / 21.4 / 15.3 / 9.5 |

Reads: fwd's #1 HW stall is `wait` (mbarrier/async data dep) — the producer→consumer dependency, which is
a genuine floor and exactly what sim's `wait_barrier` (25% of `no_warps_ready`) mirrors. bwd's #1 is
`long_scoreboard` (memory latency). Neither is "warps in lockstep on the SFU". This is the concrete
evidence that the residual is a **stall/latency-overlap** axis (how densely each pipe's waits are hidden),
consistent with — but broader than — the lockstep sub-lever.

## Pipe-utilization sim-vs-HW — DEFINITION-MATCHED comparison (2026-07-27e)

Comparing sim per-FU occupancy against NCU pipe metrics, being careful that (as with the active-def
mismatch above) sim and NCU count different things. Verified in code + NCU raw (fwd kernel5).

### The three DIFFERENT tensor metrics (do not confuse)
1. **NCU `sm__pipe_tensor_cycles_active`** = cycles the tensor pipe is **busy processing** (a WGMMA
   occupies it for many cycles) ÷ SM-**active** cycles = **46.1%** fwd. On an **elapsed** basis
   (× active/elapsed 90.1%) = **41.6%**.
2. **NCU `sm__inst_executed_pipe_tensor_op_hmma`** = HGMMA **issue** rate ÷ peak = **2.16%**. This is
   per-op issue, NOT busy — WGMMA issues rarely but runs long. (Comparing sim busy against THIS would be
   the mistake.)
3. **sim `total_num_cycles_tensor_pipe_active`** ([functional_unit.cc:321-337](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L321):
   `m_active_insts_in_pipeline>0 || !dispatch_reg.empty() || reserved_cycles>0`) = the TRUE pipe-busy
   counter, reported as `Throughput_ComputeTensor_pct` ÷ (n_sm × elapsed) = **74.18%** (`.o59`).
   Same definition as (1), same (elapsed) denominator — so (3) vs (1-elapsed) is apples-to-apples.

### ⭐ Result — sim OVER-models the tensor pipe ~1.78×
| tensor pipe-busy (elapsed basis) | sim | HW | ratio |
|---|---|---|---|
| % of (n_sm × elapsed) | **74.18%** | **41.58%** | **1.78×** |
| implied cycles / tensor-op | **29.5** | **10.7** | **2.76×** |

- sim holds the tensor pipe busy **29.5 cyc/op** vs HW's **10.7** — the tensor pipe occupancy per WGMMA
  is ~2.76× too long in the sim (the per-op latency/II from `generate_tensor_core_latencies()` —
  `M*N*K*operand_bit / tensor_rate_per_cycle=32768`, then split II=`nc/2`, latency=`nc-II`).
- ⚠️ **Caveat before acting**: WGMMA is a warpgroup op — 4 warps each issue the full-tile HGMMA in the
  sim (the deferred "4× over-execution" axis, FA3_progress.md Deferred Opts). If the sim counts 4× the
  ops AND each at the whole-SM `tensor_rate`, the 2.76×/1.78× could be that same 4×-vs-rate interaction
  rather than a raw latency error. The `29.5 cyc/op` here is computed against sim's own `tensor_ops`
  count, so whether HW's `gmma` count uses the same per-warp convention must be re-checked before
  calling this a lever. This measurement REFRAMES the tensor axis from "faithful" (the old
  ≈48% vs 46% claim in CONSUMER_COMPUTE_BOUND, which used the wrong II=16 hand-calc) to "possibly
  over-modeled ~1.8× on a busy-cycle basis" — needs the op-count convention pinned.
- **Note the earlier hand-calc was WRONG**: II=16 (config `[LATCFG] tensor_init=16`) gives 43.5%, but the
  config value is NOT used in trace mode — the effective per-op occupancy is dynamic (~29.5 cyc). Always
  read `Throughput_ComputeTensor_pct`, never the config II.

### SFU(MUFU): no sim pipe-busy counter exists
- NCU `sm__inst_executed_pipe_xu` = **47.7%** fwd / 21.4% bwd (MUFU issue-rate ÷ peak). This is the busiest
  HW pipe in fwd.
- sim has **no** SFU pipe-busy counter (only tensor has `total_num_cycles_tensor_pipe_active`). To compare
  SFU on the same basis needs either a new gated counter mirroring the tensor one, or an issue-count
  derivation. Deferred to a follow-up (the tensor over-model is the sharper lead).

### Why this matters for the issue-density gap
If the tensor pipe is genuinely ~1.8× over-busy, it directly throttles issue-density: a WGMMA sitting in
the pipe ~2.76× too long keeps the consumer warp's dependent softmax (MUFU) waiting longer, which is
exactly the `stall_count`/`wait_barrier` window that shows up as `still_idle`. This is a more concrete,
measurable lead than "lockstep" — and unlike lockstep it is a per-op MODEL quantity, not a phasing
property. **Next: pin the WGMMA op-count convention (sim 4-warp vs HW gmma) to decide if 1.78× is a real
over-model or a counting artifact.**

## Pipe-utilization comparison — CORRECTION (2026-07-27f): op-count matches, but the pipe-count normalization is the confound

The prior section's "sim tensor 1.78× over-model" claim was PREMATURE. Follow-up pinned two things: the
op-count convention (settled) and a pipe-count normalization mismatch that makes the 74.18%-vs-41.6%
comparison invalid as stated.

### Settled — op-count convention is IDENTICAL (no 4× artifact)
- sim `tensor_ops` total (Σ CTAFIN) = **349,008** (fwd) vs HW `sass_inst_executed_op_shared_gmma.sum` =
  **349,056** — a **0.01% match**. bwd likewise (sim 835,584-order == HW 835,584). Both count WGMMA
  **per-warp** (all 4 warps of a warpgroup each counted). So the residual is purely **cycles-per-op**, not
  a count artifact. (Confirms FA3_progress Deferred "count is faithful".)
- sim per-op tensor-pipe occupancy from the shape calc ([abstract_hardware_model.cc:434-439](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L434)):
  `[WGMMADBG-MNK] m64n128k16 -> number_of_cycles=64, II=32, latency=32` (and m64n64k16 -> 32/16/16). One
  tensor pipe is "busy" (in-flight OR dispatch-reg OR reserved) ~II+latency-overlapped ≈ **29.5 cyc/op**
  measured. Self-consistent with the shape calc.

### The confound — sim's tensor-pipe-active counter is a 4-subcore SUM, but the reported % divides by n_sm×1
- `total_num_cycles_tensor_pipe_active` is incremented **independently by each subcore's TENSOR FU**
  ([functional_unit.cc:328-336](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L328))
  into the shared per-SM stat → up to **+4 per cycle** (4 subcores). But `Throughput_ComputeTensor_pct`
  divides by `n_sm × run_cycles` ([gpu-sim.cc:4062](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4062)) = **1 pipe/SM peak**. Numerator counts 4 pipes, denominator assumes 1 ⇒ the 74.18% is inflated by up to 4×.
- Normalizing to the same basis gives two bracketing readings, and NEITHER can be confirmed without the
  NCU raw `peak_sustained` count (not stored in this report):
  - sim ÷ (n_sm × **1** × cyc) = **74.18%** (numerator 4-pipe sum, wrong denom)
  - sim ÷ (n_sm × **4** × cyc) = **18.5%** (one-pipe basis) vs HW **41.6%** (elapsed) ⇒ sim ~0.44× (UNDER)
- HW `sm__pipe_tensor_cycles_active` is a **per-SM** metric normalized to the SM's tensor peak; whether
  that peak = 1 SM-tensor-unit or 4-SMSP-units decides which sim normalization matches. Cannot resolve
  from the stored report (raw peak_sustained count absent, driver unavailable in-container).

### Honest verdict (this axis is UNRESOLVED, not a proven lever)
- **CONFIRMED**: op-count identical; sim one-pipe occupancy ≈ 29.5 cyc/op, HW-implied ≈ 10.7 cyc/op on a
  1-pipe basis — a **2.76× per-op** gap IF the 1-pipe basis is the right one for both.
- **UNCONFIRMED**: whether HW's per-SM tensor metric should be compared to sim's 1-pipe (÷4) or 4-pipe-sum
  number. The two give opposite conclusions (sim under vs over). So "tensor over-model 1.78×" is
  **retracted** as a claim; it is at most a *candidate* pending a definition-exact normalization.
- To resolve without guessing: either (a) obtain NCU raw `sm__pipe_tensor_cycles_active` absolute count +
  its `.peak_sustained` to fix HW's per-cycle pipe basis, or (b) add a sim SM-level OR counter ("≥1 subcore
  tensor-busy this cycle") to match HW's per-SM active-cycle semantics directly (small gated counter,
  mirrors the existing `any_subcore_*` pattern at [sm.cc:577-601](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L577)). (b) is the clean apples-to-apples fix.

⇒ Supersedes the "sim OVER-models tensor ~1.78×" result in the section above — that number used a
numerator/denominator pipe-count mismatch. The tensor axis is a **definition-pending candidate**, not a
confirmed lever.

## ⭐⭐ RESOLVED via raw-count re-profile (2026-07-27g): peak_sustained=4 ⇒ sim tensor is UNDER-modeled, not over

The pipe-count confound is now SETTLED with a fresh NCU run that captured the raw `.sum` + `.peak_sustained`
the original `--set full` report omitted (`nv_reports/h100/fa3_pipe_rawcounts.ncu-rep`, produced by
[rerun_ncu_pipe_rawcounts.sh](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/rerun_ncu_pipe_rawcounts.sh)).

### The decisive value: `sm__pipe_tensor_cycles_active.avg.peak_sustained = 4`
HW normalizes the per-SM tensor-busy metric by **4** (one tensor pipe per SMSP × 4 SMSP). Verified:
`pct_active = tensor_sum / (SM_active × 4)` reproduces the report exactly (fwd 14,893,056/(8,075,759×… ) →
46.10% == report). ⇒ HW's per-SM tensor number is a **4-pipe sum**, EXACTLY the same basis as the sim's
`total_num_cycles_tensor_pipe_active` (which sums all 4 subcore TENSOR FUs). So they ARE directly
comparable — the earlier `÷1` was the bug, `÷4` is correct.

### Definition-exact tensor comparison (fwd, same op-count 349,056)
| tensor (4-pipe basis) | sim | HW | sim/HW |
|---|---|---|---|
| busy cycles / op | **29.5** | **42.7** | **0.69×** |
| busy % (÷ n_sm×elapsed×4) | 18.6% | 41.0% | 0.45× |
| bwd busy cyc/op | — | 40.0 | — |

⇒ **sim UNDER-models the tensor pipe (~0.69× cyc/op), it does NOT over-model it.** The prior
"1.78× over" was purely the ÷1-vs-÷4 normalization error and is now fully **retracted**.

### What this means for the residual (important reframe)
- sim runs each WGMMA in **fewer** cycles than HW (29.5 vs 42.7) yet the whole kernel is **1.56× slower**.
  So the tensor pipe's *execution time* is categorically **not** the source of the gap — if anything sim
  finishes tensor work faster.
- The gap must therefore be in **overlap / scheduling / dependency waiting**: HW keeps the tensor pipe
  busy 42.7 cyc/op AND simultaneously runs the MUFU pipe (xu 47.7%) and others, filling the SM
  (issue-density 0.407); sim cannot overlap those pipes (issue-density 0.289), so its shorter tensor ops
  still leave the SM idle waiting between dependent stages. This is the SAME issue-density/overlap axis,
  now positively confirmed to be NOT a per-pipe cost error.

### SFU / MUFU also definition-confirmed HW-faithful
- `sm__inst_executed_pipe_xu.avg.peak_sustained = 0.5` inst/cyc/SM = 0.125/SMSP = **1 MUFU / 8 cyc/SMSP**
  — exactly the sim's SFU II=8 (4 SFU/SMSP, 32-lane warp ⇒ 32/4=8). HW MUFU issue = 47.7% of peak (fwd) /
  21.1% (bwd). sim MUFU op-count is work-invariant (same trace). ⇒ the SFU throughput model is HW-faithful
  (confirms Opt 10), and MUFU is NOT over-modeled either.

### Net conclusion of the whole pipe-utilization investigation
Every per-pipe COST is HW-faithful or lighter in the sim (tensor 0.69×, SFU II exact, op-counts exact).
The 1.41× issue-density gap is therefore **not** a functional-unit cost-model error — it is an
**overlap/scheduling** property (how densely HW interleaves tensor + MUFU + memory waits vs the sim's
in-phase consumer warps). This closes the "tensor over-model" lead and points the residual squarely back
at issue-density/overlap — with the reassurance that no pipe is being over-charged.

## Guardrails

- Observation/measurement only until a mechanism is proven; any prototype is gated + default-off +
  bit-identical when off (project rule).
- II / latency stay HW-faithful. No pipe may exceed its NCU occupancy.
- Jitter must **emerge from an independently-correct physical model**, never be an injected offset tuned
  to the HW cycle count (Trap 1). If it can only be made to work as a knob, it is not faithful — reject.
- If the honest conclusion is "structural floor," accept it — do not manufacture a speedup.
