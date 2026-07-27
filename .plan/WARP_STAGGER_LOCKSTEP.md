# FA3 fwd — Warp-Stagger / MUFU-Lockstep Axis (lever A) (2026-07-27)

Owner axis for the DOMINANT part of the post-Opt-10 SFU-throughput residual: **MUFU-lockstep**
(fwd 74.6% / bwd 70.2% of `still_idle` cycles). Split out of `.plan/CONSUMER_COMPUTE_BOUND.md` because,
unlike Opt 10 (a local issue-gate fix), this is a scheduling/occupancy-model question with no obvious
low-cost fix and real HW-fidelity trade-offs. **No cycle claim until a verified improvement lands.**

## The problem in one paragraph

FA3's consumer warpgroup runs 12 warps of the *same* code: `WGMMA (S=QK^T) → softmax (MUFU.EX2) →
WGMMA (O=PV) → …`. The MUFU-lockstep probe (`.o59`/`.o34`) showed that on ~3/4 of `still_idle` cycles
*every* valid-head warp is on a `MUFU` at once, so all 12 want the one HW-faithful SFU (II=8) and nobody
can issue — and there is no free-pipe (WGMMA/FMA) warp to switch to. HW hides the same 8-cycle SFU
throughput because its warps are spread across different pipe stages (NCU `not_selected=0.82`,
SM-active 90%); the simulator's warps march in phase, so SFU throughput becomes whole-subcore idle.

## Evidence (from the MUFU-lockstep probe, fwd `.o59` / bwd `.o34`)

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

## Why the "easy" fixes DON'T work (already reasoned out)

- **GTO → LRR / weaken greedy: NO.** The scheduler only *reorders eligible warps*. On a lockstep cycle
  every eligible warp is on MUFU (SFU-busy ⇒ not eligible), so any order picks a blocked warp. Opt 10
  already makes the scan continue past a blocked head; the problem is the pool it scans is all-MUFU, not
  that it stops scanning. (User-confirmed.)
- **lever B (tensor/fma issue-gate extension): does not exist.** tensor/fma are fixed-latency and already
  `can_issue()`-checked at issue; they never clog the latch. See CONSUMER_COMPUTE_BOUND.md "lever B …
  DOES NOT EXIST".

## Candidate mechanisms (unproven — this is the open design space)

Ordered by how faithful-to-HW they are, not by ease:

1. **Execution-time jitter / non-deterministic latency.** HW warps de-phase because per-warp memory
   arrival, bank conflicts and FU latencies vary slightly; a big resident-warp pool (up to 16/SMSP)
   absorbs it so some warp is always at a non-MUFU stage. The simulator is largely deterministic and
   FA3 runs 1 CTA/SM, so there is little pool and no jitter to de-phase. Modeling this is the most
   HW-honest but the most invasive and the most fidelity-sensitive (must not become a tuning knob).
2. **Launch-phase stagger.** Start the 12 consumer warps at slightly different offsets so they enter the
   MUFU region out of phase. Cheap, but the per-tile mbarrier likely re-synchronizes them within a tile
   or two ⇒ expected small, must be measured before trusting.
3. **Occupancy floor (root, maybe not fixable).** The deepest cause may simply be 1 CTA/SM: HW's 90% is
   partly "more resident warps to hide the SFU," which FA3's SMEM footprint forbids. If so, the lockstep
   residual is a faithful reflection of low occupancy, not a model defect — and the correct outcome is to
   *document it as a floor*, not to "fix" it. **This must be ruled in/out before any code change.**

## What HW actually does — reference facts (do NOT hand-wave)

- NCU kernel-5: `smsp__warps_active ≈ 3.28`, `not_selected = 0.82`, SM-active ~90%, xu(MUFU) pipe 47.75%.
  ⇒ HW is *itself* partly SFU-bound; the achievable ceiling from de-phasing is bounded, not 2×.
- The SFU II=8 is HW-faithful (4 SFU/subcore) and must NOT be lowered (see Opt 10). This axis is about
  *overlap*, never about making the SFU wider.

## Plan / next steps (investigation-first, no code yet)

1. **Rule the occupancy floor in or out.** Quantify how much of the 74.6% would survive if resident warp
   count matched HW. Static first: compare sim resident warps/SMSP (1 CTA/SM ⇒ ?) vs NCU
   `warps_active 3.28`. If sim already has ≥ HW's eligible pool at these moments, lockstep is a phasing
   problem (mechanisms 1–2); if sim has far fewer, it is the occupancy floor (mechanism 3).
2. **If phasing:** prototype launch-phase stagger (mechanism 2) behind a gate, measure whether the
   per-tile mbarrier washes it out (probe: does `head_mufu/valid_head` drop?). Cheap A/B.
3. **If occupancy floor:** stop — write it up as a structural floor in FA3_progress.md, make no cycle
   claim, and consider the compute-path axis closed.
4. Only escalate to jitter modeling (mechanism 1) if 1–3 show a real, faithful, sizeable lever.

## Guardrails

- Observation/measurement only until a mechanism is proven; any prototype is gated + default-off +
  bit-identical when off (project rule).
- II / latency stay HW-faithful. No pipe may exceed its NCU occupancy.
- If the honest conclusion is "occupancy floor," accept it — do not manufacture a speedup.
