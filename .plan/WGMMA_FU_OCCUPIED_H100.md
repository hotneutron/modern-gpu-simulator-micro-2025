# Opt 6 — WGMMA / Tensor-Pipe Issue-Serialization Fix (H100 / FA3)

> Target: FA3 fwd (kernel 5) and FA3 bwd (kernel 10), on top of Opt 5 (eager-promote).
> This replaces the deferred shared-bank-conflict idea (see `SHMEM_BANK_CONFLICT_H100.md`,
> parked). Chosen because a full HW-vs-sim stall comparison shows the **tensor/WGMMA
> `fu_occupied` bucket is the only large bucket the simulator OVER-estimates** — i.e. the only
> high-leverage lever that actually shrinks the sim-is-slower cycle gap.

## 1. Why this optimization — full stall comparison (data-driven)

HW warp-issue stall decomposition (NCU raw `smsp__average_warps_issue_stalled_*_per_issue_active`,
normalized to % of total warp-cycles-per-issued) vs the sim issue-stage inner breakdown
(`..._at_least_one_warp_*`, per-cycle "≥1 warp waiting"). Compare **ranking / over-vs-under**,
not identical absolutes.

### bwd (k10) — sim 242,270 (Opt 5 `.o320`) / HW 132,901 = **1.82×**

| HW stall reason | HW % | sim bucket | sim % | verdict |
|---|---|---|---|---|
| long_scoreboard (TMA/global arrival) | 19.8% | `wait_barrier` (DEPBAR mbarrier) | 14.8% | ~ok (under) |
| barrier (mbarrier sync) | 17.4% | `inst_barrier` | 1.3% | sim **under** |
| wait (fixed-latency dep) | 10.4% | `stall_count` | 7.4% | ~ok |
| short_scoreboard | 8.5% | `l1c`/— | ~0% | sim **under** |
| mio_throttle (shared/LSU) | 6.3% | (shared; Opt-deferred) | — | n/a |
| not_selected (scheduler) | 5.4% | — | none | sim missing |
| **gmma (WGMMA)** | **5.3%** | **`fu_occupied`** | **17.75%** | **sim OVER (see §1 caveats)** |
| dispatch_stall | 4.5% | — | none | sim missing |
| sleeping | 4.4% | — | none | sim missing |

### fwd (k5) — sim 150,755 / HW 67,696 = **2.23×**

| HW stall reason | HW % | sim bucket | sim % | verdict |
|---|---|---|---|---|
| wait | 19.0% | `stall_count` | 8.5% | sim **under** |
| not_selected | 11.4% | — | none | sim missing |
| dispatch_stall | 11.0% | — | none | sim missing |
| barrier | 10.9% | `inst_barrier`/`wait_barrier` | 0.07% / 13% | mixed |
| long_scoreboard | 9.8% | `tma_axis` | ~13% | ~ok |
| mio_throttle | 6.4% | — | — | shared |
| **gmma (WGMMA)** | **1.4%** | **`fu_occupied`** | **13.6%** | **sim OVER (see §1 caveats)** |

### The decisive observation

The sim is **1.8–2.2× slower** than HW, yet the stall decomposition shows the sim **over-states
exactly one** large bucket: **`fu_occupied`** — bwd 17.75% vs HW `gmma` 5.3%, fwd 13.6%
vs HW `gmma` 1.4%. Every *other* bucket (`wait`, `not_selected`, `dispatch`, `barrier`) the sim
*under*-states or doesn't model — fixing those would make the sim *slower* (widen the gap).
**So `fu_occupied` is essentially the only lever that closes the cycle gap.**

> **CAVEATS on the comparison above (must resolve before committing to the fix).**
> 1. **`fu_occupied` is NOT tensor-only.** It is set whenever *any* fixed-latency pipe's
>    `can_issue()` fails ([subcore.cc:586-590](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L586-L590)),
>    i.e. SP / INT / DP / TENSOR / UNIFORM / BRANCH / MISC all share the one flag. We do **not**
>    yet know what fraction of the 18% / 13.6% is the tensor pipe vs e.g. SFU/MUFU (fwd has 47%
>    MUFU busy on HW). The "fu_occupied == WGMMA" identification is a hypothesis until measured
>    (Step 0).
> 2. **Different units, so the "3.4× / 10×" is not a literal ratio.** Sim `fu_occupied` is a
>    per-cycle, core-level "≥1 warp blocked" overlap counter (can exceed 100% when summed); HW
>    `gmma` is a per-issued-instruction, warp-averaged ratio. Compare *direction/shape*, not the
>    exact multiple.
> 3. **"latency preserved ⇒ WGMMA result-wait preserved" is unverified.** The fix keeps `latency`
>    and shrinks only `initiation_interval`; whether the WGMMA result/scoreboard release for a
>    fixed-latency unit is actually driven by `latency` (so keeping it preserves the dependency
>    stall) must be confirmed in code before trusting the safety argument.

## 2. Root cause (verified in code + SASS)

WGMMA on Hopper is **asynchronous**: a warpgroup issues a stream of `HGMMA` ops back-to-back
(pipelined), commits a group, and only later `wait`s; meanwhile other warps issue freely. The
tensor pipe sustains roughly one HGMMA per its throughput interval, and stalls only when a `wait`
actually blocks on an unfinished group (HW `gmma` stall = 1.4% fwd / 5.3% bwd).

The simulator instead **serializes WGMMA issue** through a single per-subcore tensor pipe slot:

1. **`initiation_interval` is large.** [generate_tensor_core_latencies()](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L427-L443):
   `number_of_cycles = M·N·K·operand_bits / tensor_rate_per_cycle`,
   `initiation_interval = number_of_cycles/2`.
   - SASS (verified): bwd uses `HGMMA.64x128x16.F32.BF16` and `HGMMA.64x64x16` (M=64, N=128/64,
     K=16, bf16 → operand_bits=16). Config `tensor_rate_per_cycle = 32768`
     ([gpgpusim.config:245](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config#L245)).
   - m64n128k16 → number_of_cycles = 64·128·16·16/32768 = **64** → `initiation_interval = 32`.
     m64n64k16 → **32** → `initiation_interval = 16`.
2. **The interval blocks the next WGMMA issue.** On issue,
   [Subcore::issue_warp](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L891-L892)
   calls `fu->reserve_unit()`, which sets
   `m_dispatch_pending_reserved_cycles = initiation_interval`
   ([functional_unit.cc:125-131](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L125-L131)).
   `can_issue()` then returns false until it decrements to 0
   ([functional_unit.cc:137-139](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L137-L139)).
   So a warp whose head is the **next** WGMMA cannot issue for ~32 cycles
   ([subcore.cc:525-530](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L525-L530)).
3. **That stall is exactly what `fu_occupied` counts.**
   [subcore.cc:586-590](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L586-L590)
   sets `is_any_waiting_in_fu_occupied` when a warp can't issue because `!is_fu_available`.
   FA3 warpgroups issue many consecutive HGMMAs in one subcore, so the 32-cycle re-issue lockout
   repeats and inflates `fu_occupied` far above HW's async behavior.

**Granularity is NOT the main cause** (ruled out): the tensor pipe is **per-subcore** (4 per SM,
[subcore.cc:1264-1265](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1264-L1265),
[subcore.cc:950-952](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L950-L952)),
and the issue loop skips a blocked warp rather than stalling the whole SM. The serialization is
**within a warpgroup's own back-to-back WGMMA stream**, driven by (1)+(2). WGMMA result waits are
handled separately (`WGMMA.WAIT`→BARRIER_OP, `WGMMA.COMMIT`→MISC), so reducing the issue lockout
does **not** remove real dependency stalls.

## 3. Step 0 — one-run instrumentation (MANDATORY before the fix; a full run is ~12 h)

Because a run costs ~12 h, add **all** of the following logs/counters in a single instrumentation
pass so one run resolves every open question (Caveats 1-3 in §1) and quantifies the expected gain.
These are **observe-only** (no timing change), so this run also stays a valid baseline.

**(I) Split `fu_occupied` by pipe (resolves Caveat 1 — what fraction is really the tensor pipe).**
At [subcore.cc:586-590](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L586-L590),
when `!is_fu_available`, classify by `pI->op` and bump per-pipe counters in addition to the
existing aggregate:
- `..._with_fu_occupied_tensor` (`pI->op == TENSOR_CORE_OP`)
- `..._with_fu_occupied_sfu` (`SFU_OP`)
- `..._with_fu_occupied_sp_int_dp` (SP/INT/DP/UNIFORM)
- `..._with_fu_occupied_other` (rest)
Register in `gpu-sim.cc`, print in `shader.cc` next to the existing `..._with_fu_occupied`.
This directly tells us the tensor share of the 18% / 13.6%.

**(II) WGMMA `initiation_interval` / `latency` histogram (confirms the 32-cycle assumption and
the m64n128 vs m64n64 mix).** In
[generate_tensor_core_latencies()](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L427-L443),
accumulate per-(M,N,K) buckets: count of WGMMA decoded, and the resulting `initiation_interval`
and `latency`. Emit once at end of run (e.g. `[WGMMADBG] mnk=64x128x16 count=.. II=32 lat=32`).

**(III) Tensor-pipe re-issue-lockout lost cycles (quantifies the *upper bound* of the gain).**
Add a counter that, per subcore, counts cycles where the tensor pipe is the *sole* reason the
greedy/eligible warp could not issue — i.e. `pI->op == TENSOR_CORE_OP && !fu->can_issue(pI)` while
all of that warp's *other* issue conditions (`is_stall_counter_0`, `are_wait_barriers_ready`,
`is_not_warp_waiting_*`, scoreboard, result-queue) are **true**. This isolates "blocked only by
the tensor II lockout" from "would have stalled on a real dependency anyway", giving the realistic
ceiling on how many cycles the fix can recover. Name e.g.
`total_num_cycles_tensor_reissue_lockout_only`.

**(IV) Verify the latency/scoreboard release path (resolves Caveat 3).** Statically confirm (read
+ a one-time `[WGMMADBG]` print at issue) where a fixed-latency TENSOR_CORE_OP's destination
scoreboard/result is released — i.e. that it is driven by `latency`
([functional_unit.cc:295-307](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L295-L307),
`start_stage = latency-1`) and **not** by `initiation_interval`. If release is latency-driven,
shrinking only `initiation_interval` provably preserves the result-wait. Record the finding before
the run so the fix in §4 is justified.

**Decision gate after Step 0:** proceed with the §4 fix only if (I) shows the tensor pipe is a
**majority** of `fu_occupied`, (III) shows a large `tensor_reissue_lockout_only`, and (IV)
confirms latency-driven release. Otherwise re-target (e.g. SFU/MUFU if that dominates fwd).

**(V) SM-level idle counter (REQUIRED — corrects per-subcore overcount of III).** `fu_occupied`
and (III) are counted **per-subcore** (`Subcore::issue` runs once per scheduler per cycle;
SM has 4 subcores, [sm.cc:547-548](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L547-L548),
[gpgpusim.config -gpgpu_num_sched_per_core 4]). A subcore blocked on the tensor pipe is only a
**real cycle loss if the whole SM issued nothing that cycle**. No such SM-level stat exists today
(the nearest, `SM::is_any_subcore_problems_of_fordward_progress()`
[sm.cc:1380-1386](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1380-L1386), only tracks
downstream back-pressure). Add an SM-level counter: cycles where **all 4 subcores issued nothing**,
and a sub-variant where **all non-idle subcores were blocked specifically by the tensor pipe**.
This is the true denominator for the recoverable-cycle bound — without it, (III) overstates the
gain. Aggregate at the end of the per-SM subcore loop in `SM::cycle()`.

**(VI) `fu_occupied` ↔ `wait_barrier` coupling counter (REQUIRED — bounds net gain).** Lowering
the tensor re-issue interval can merely **shift** a stall from `fu_occupied` into `wait_barrier`
(the WGMMA.WAIT / DEPBAR result-dependency bucket) with little net cycle gain. The buckets are
already counted independently (`wait_barrier` set at
[subcore.cc:603-604](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L603-L604), `fu_occupied` at
[:588-590](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L586-L590)), but their *coupling* is not. Add a counter for cycles
where a warp's head is TENSOR_CORE_OP, blocked only by `!is_fu_available`, **and** that same warp
also already has `!are_wait_barriers_ready` (its next WGMMA.WAIT would block anyway). A large value
means the issue-interval fix mostly moves stalls into `wait_barrier` ⇒ small net gain; a small
value means the recovered cycles are genuine. Both predicates are already computed in the same
block ([subcore.cc:514-515,526-530](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L514-L530)).

**(VII) Dynamic re-issue-lockout extensions + RF-port conflicts (REQUIRED — the real lockout is
NOT just the static II).** The tensor pipe's actual re-issue lockout can grow **beyond** the
static `initiation_interval` (=32): `functional_unit::add_extra_cycle_initiation_interval()` is
called when the RF read cannot be served or the CONTROL→ALLOCATE latch is full
([subcore.cc:329,333](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L329-L335),
[subcore.cc:366-368](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L366-L368);
[functional_unit.cc:133-135](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L133-L135)). WGMMA reads
many registers (`is_tensor_core_op_with_4_registers_per_op`,
[abstract_hardware_model.h:1196](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L1196)), so RF-port conflicts may be
a real second contributor. Implications:
- (II)'s static II/latency histogram is **not** the full picture — also log the **tensor-pipe
  `add_extra_cycle_initiation_interval` call count** (or the realized `m_dispatch_pending_reserved_cycles`
  distribution at issue time) so we see the *effective* lockout, not just the formula value.
- The existing `total_num_evals_rf_with_conflict`
  ([subcore.cc:330,334](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L330)) is SM-wide; add a
  **tensor-only** RF-conflict counter so we know how much of the lockout is RF-port vs the static
  II. This decides whether shrinking II alone is enough, or whether RF read-port modeling for
  WGMMA must also be revisited.

**(IV) result — already confirmed in code (no run needed):** issue sets
`target_latency = ... + latency + initiation_interval` for the release/result path
([subcore.cc:300,326](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L300)) and the pipeline entry stage is
`latency-1` ([functional_unit.cc:295-307](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L295-L307)), while
`initiation_interval` only drives `m_dispatch_pending_reserved_cycles` (the re-issue lockout,
[functional_unit.cc:128-129](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L128-L129)). So shrinking only
`initiation_interval` provably leaves WGMMA result latency unchanged. Keep a one-time
`[WGMMADBG]` print to re-confirm at runtime, but this is settled.

> **Per-kernel separation requirement (so ONE run covers both fwd k5 and bwd k10).** Counters are
> printed and **reset per kernel** (`print_remodeling_stats` then `reset_gpu_per_sm_stats`,
> [gpu-sim.cc:3410-3411](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3410-L3411)). **All new counters (I)/(III)/(V)/(VI) MUST be
> registered in `m_gpu_per_sm_stats`** (same path as the existing `..._with_fu_occupied`), not in a
> static/global accumulator — otherwise k5 and k10 values get mixed. With that, a single run with
> `-filter_first_kernel_id`/`-filter_last_kernel_id` spanning 5..10 emits k5 and k10 separately.
> The target re-issue-interval *value* itself is NOT derivable from sim/trace (no real WGMMA
> throughput is encoded; `tensor_latency=32` is unused by `generate_tensor_core_latencies`); it
> must come from NCU `sm__inst_executed_pipe_tensor_op_gmma` throughput. (II)'s histogram is the
> sim-side input to set it.

## 4. How to implement the fix

> All changes behind a config flag, default off, for clean A/B vs Opt 5.

The fix is to make the modeled tensor-pipe **issue throughput** match real back-to-back WGMMA,
i.e. reduce the per-WGMMA re-issue lockout so consecutive WGMMAs pipeline instead of serializing
at 32-cycle gaps — **without** touching the WGMMA result `latency` (the dependency/`wait` path,
which HW does pay and which the sim currently under-states, must not shrink).

Two candidate levers (A/B both, keep whichever matches HW `gmma` 1.4%/5.3% and the cycle trend):

- **Lever 1 — separate "issue throughput interval" from "compute latency".** Today
  `initiation_interval = number_of_cycles/2` couples them. Introduce a config
  `-tensor_issue_interval_cycles` (or a divisor) that sets the tensor pipe's re-issue lockout
  (`reserve_unit`'s `m_dispatch_pending_reserved_cycles`) to the real WGMMA throughput (a few
  cycles), while leaving `latency` (result availability) unchanged at `number_of_cycles - II`.
  This directly lowers the false `fu_occupied` while preserving the WGMMA completion time.
- **Lever 2 — config-only first probe:** raise `-tensor_rate_per_cycle` so `number_of_cycles`
  (hence `initiation_interval`) drops. Cheapest A/B (no rebuild), but it **also** shortens the
  result `latency` — which may be wrong if the WGMMA compute latency is currently accurate. Use
  this only as a quick sensitivity check, not the final fix; Lever 1 is the principled one.

Implementation points:
- Gate the new interval in [generate_tensor_core_latencies()](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L427-L443)
  and/or in [reserve_unit()](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L125-L131)
  for `TENSOR_CORE_OP` only.
- Register the flag in `gpu-sim.cc`; expose in `gpgpusim.config`.
- Do **not** alter the `WGMMA.WAIT`/scoreboard path — only the tensor pipe's issue lockout.

*Files expected to change:* `abstract_hardware_model.cc` (tensor II vs latency split),
`functional_unit.cc` (tensor re-issue lockout), `gpu-sim.cc` (flag), `gpgpusim.config` (flag).

## 4. Verification

- **Primary:** sim `fu_occupied` (`..._with_fu_occupied`) drops toward HW `gmma` share
  (bwd 17.75% → ~5%, fwd 13.6% → ~1–2%), and total sim cycles drop toward HW (bwd 242,270 → lower,
  fwd 150,755 → lower).
- **Must NOT regress:** WGMMA result-dependency stalls (`wait_barrier`/scoreboard) should not
  collapse — the compute latency is unchanged. If `wait`-type stalls vanish too, the latency was
  wrongly shortened (Lever-2 risk).
- **Safety:** flag-off run reproduces Opt 5 / Opt 4 cycles bit-for-bit.
- **Sanity:** `gpu_sim_insn` unchanged.

## 5. Result

— (pending implementation + run) —
