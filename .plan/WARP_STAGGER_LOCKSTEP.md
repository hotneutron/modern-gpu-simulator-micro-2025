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

### Conclusion for lever A: STRUCTURAL FLOOR — axis closed

Combined with Step 1 (occupancy floor ruled out) and the feasibility scan, the chain is now complete:
- **not occupancy** (sim resident warps == HW theoretical);
- **is phasing** (issue-density 1.41× fwd), but **phasing is not recoverable by staggering** because the
  per-tile mbarrier deterministically re-synchronizes the warps (E1 proved it empirically);
- the only faithful path left (candidate B: dynamic cross-warp bank/port contention jitter) would have to
  overcome that same re-synchronization *and* stay bounded by HW's own 54% No-Eligible — E1 shows even a
  perfect launch stagger yields 0, so B's realistic payoff is almost certainly negligible.

⇒ **Lever A / MUFU-lockstep is a structural deterministic-model floor. No cycle claim. Compute-path axis
closed.** Recorded in FA3_progress.md.

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

## Guardrails

- Observation/measurement only until a mechanism is proven; any prototype is gated + default-off +
  bit-identical when off (project rule).
- II / latency stay HW-faithful. No pipe may exceed its NCU occupancy.
- Jitter must **emerge from an independently-correct physical model**, never be an injected offset tuned
  to the HW cycle count (Trap 1). If it can only be made to work as a knob, it is not faithful — reject.
- If the honest conclusion is "structural floor," accept it — do not manufacture a speedup.
