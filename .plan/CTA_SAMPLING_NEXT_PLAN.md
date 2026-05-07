# CTA Sampling — Plan to close the wider-validation gap

The wider 10-workload sweep on `cta-sampling` exposed three distinct
failure modes (full diagnosis in `CTA_SAMPLING_PHASE_AB_DONE.md` §5):

```
heartwall  -28.7%   K-rep replicates 3 corner CTAs to fill target
nn        +103.5%   log-fit extrapolates from N=2 to N=23 CTAs/SM
nw         +56.8%   pilot overhead + state pollution on tiny grids
```

This doc is the plan to close those gaps. Three layered changes,
ordered cheap → expensive. Each is independently shippable.

---

## Goals

- Hit p50 < 15%, p90 < 25% on the **wider 10-workload set** (not just
  the original 6).
- No regression on the workloads already at target.
- No more than ~2× sim-time cost on the workloads that need extra
  pilot work — sampling is supposed to *speed up* simulation.

---

## Change 1 — Skip pilot for tiny grids + auto-accept full-grid samples

**Targets:** nw (+56.8%). Likely also helps lud/pathfinder by
avoiding unnecessary iter-1 re-runs.

**Two small fixes in `pilot_decide_accept` and pilot enable logic:**

**(a) Auto-accept iter 0 when sampled_ctas covers the full grid,
regardless of class.** Today `pilot_decide_accept` returns `false` for
non-COMPUTE classes at iter 0 *before* checking the all-CTAs-covered
short-circuit. nw kernel 1 has 1 CTA total, K-rep returns 1, classifier
says memory → rejected → pilot re-runs the same 1-CTA kernel at iter 1
for nothing. Fix: hoist the `ps.ctas_launched >= pst.total_ctas` check
above the class check.

**(b) Skip the pilot loop entirely when `total_ctas < total_sms`.**
For these tiny grids, the kernel runs in ~1 wave anyway; the speedup
upper bound is small, and the pilot's iter-0-reject + iter-1-accept
sequence often *exceeds* baseline wall time (cache-state pollution
from rejected iters slows the accepted iter). Just disable
`pilot_loop_enabled` for the kernel when `total_ctas < total_sms` and
fall through to the K-rep run — which will cover most/all of the
grid anyway.

**Code touches:** `main.cc` only. ~30 LOC.
**Expected outcome:** nw moves from +56.8% to within ±5%. lud /
pathfinder stay where they are (they already work; this just shaves
unused iter time off the wall clock).

**Risk:** very low. The threshold is conservative (`total_ctas <
total_sms`); no kernel that currently meets target relies on the
tiny-grid pilot path.

---

## Change 2 — Fix K-rep replication: never duplicate before exhausting unique CTAs

**Targets:** heartwall (−28.7%). Also marginal improvement for
backprop's K1 (3 reps replicated to 80).

**Today** `expand_sampled_ctas` always duplicates the K-rep set to
fill the target:

```cpp
unsigned per_rep = target_sim_ctas / K;
unsigned remainder = target_sim_ctas % K;
for (unsigned i = 0; i < K; ++i)
  for (unsigned c = 0; c < per_rep + (i<rem?1:0); ++c)
    out.push_back(reps[i]);                     // ← only ever duplicates
```

For heartwall's 51×1×1 grid: K-rep returns 3 (corners + midpoint),
target=51 → out is 51 copies of 3 corners. The "full grid sample"
is biased toward boundary work.

**Fix in two layers:**

**(a) When `target >= total_ctas`, return all unique CTAs.** If the
pilot expanded enough that we'd cover the whole grid anyway, just
generate `(0..gx-1, 0..gy-1, 0..gz-1)` and skip the K-rep
replication entirely. heartwall iter 3 with target=51, total=51 →
sample all 51 unique CTAs.

**(b) When `K < target < total_ctas`, supplement K-rep with
evenly-spaced fresh CTAs before replicating.** If we need
`target − K` more CTAs and `total_ctas − K` unique ones are
available, pick `min(target − K, total_ctas − K)` unique CTAs by
linearly walking the grid (skipping coordinates already in the
K-rep set). Only after that pool is exhausted do we fall back to
duplication. For backprop K1 (256 CTAs, K=3, target=80): pick 77
unique additional CTAs, no duplication.

**Code touches:** `main.cc` `expand_sampled_ctas` only. ~50 LOC.
**Expected outcome:** heartwall from −28.7% toward ±10%. backprop
slight tightening (already at −15.4%).

**Risk:** low. The replication path was the only way to fill
`target_sim_ctas` slots above K; switching to unique CTAs is strictly
more representative. Stat-scaling math (`weight = total / sampled`)
doesn't care whether sampled CTAs are duplicates or unique.

---

## Change 3 — Bound log-fit extrapolation distance + adaptive deeper expansion

**Targets:** nn (+103.5%). Most expensive of the three; also the
most uncertain.

**Diagnosis recap:** nn has 938 CTAs; full grid runs 23 CTAs/SM. With
`pilot_max_doublings=2`, the deepest pilot iteration reaches 80 CTAs
sampled = 2 CTAs/SM. `T(N) = a + b*log(N+1)` is fit on densities ∈
{1, 2}. Extrapolating from N=2 to N=23 is past the model's
calibration range, and the actual per-SM throughput growth is
super-log past N=2 (still in the "lots of latency to hide" regime).
Result: `T_full` is under-predicted ~2×, cycle estimate ~2× over.

**Two complementary fixes:**

**(a) Adaptive `pilot_max_doublings` for high-projection kernels.**
When the force-expand condition fires (`full_per_sm > 4 ×
sampled_per_sm`), bump the cap from 2 doublings to log2(projection_ratio)
doublings. For nn (ratio 23×): allow up to 5 doublings, reaching 80
→ 160 → 320 CTAs sampled = 8 CTAs/SM. The log-fit then has
densities ∈ {1, 2, 4, 8} and extrapolates only N=8 → N=23
(2.9× past the max sampled, much closer to calibration).

The adaptive cap only fires for kernels that already triggered
force-expand, so hotspot/srad_v2/lud (no force-expand) keep their
fast-accept paths.

**(b) Clamp log-fit extrapolation distance.** Even with deeper
expansion, defensively clamp `T_full` so we never extrapolate more
than 4× past the maximum sampled N. If `full_per_sm > 4 × max_sampled_per_sm`,
predict `T(4 × max_sampled_per_sm)` and stop there — i.e., assume the
remaining concurrency growth saturates. Conservative; biases toward
*under*-projection (smaller cycle estimate) when the fit can't be
trusted, which is the right direction since over-projection is what
hurt nn.

**Code touches:** `main.cc` (adaptive cap in pilot_state init or
pilot_next_target), `gpu-sim.cc` (clamp in the log_fit branch of the
estimator). ~80 LOC.
**Expected outcome:** nn from +103.5% toward ±25%. May or may not
clear p90 target depending on how saturating the actual T(N) curve
is past N=8.

**Risk:** medium.
- Sim-time cost on nn: 4 kernels × 2 extra iters ≈ 2× simulation
  time for those kernels. Worst-case per nn run: ~1.5× current pilot
  wall, still faster than baseline (which simulates all 938 CTAs).
  Acceptable trade since accuracy was the goal.
- Could over-clamp on workloads with genuine super-log throughput
  growth. Mitigate by making the 4× clamp factor a knob
  (`-cta_sampling_log_fit_extrapolation_max`, default 4) so it can be
  tuned without a recompile.

---

## Validation plan

After **each** change, run the wider 10-workload sweep (
`./util/cta_sampling/validate.py --out-dir /tmp/sweep_changeN`) and
compare cycle_err% to the current baseline:

```
                  | now    | after C1 | after C1+C2 | after C1+C2+C3
hotspot           | +14.7  | +14.7    | +14.7       | +14.7
backprop          | -15.4  | -15.4    | ≈-12        | ≈-12
pathfinder        |  -2.6  |  -2.6    |  -2.6       |  -2.6
bfs               |  -9.9  |  -9.9    |  -9.9       |  -9.9
srad_v2           | +17.3  | +17.3    | +17.3       | +17.3
lud               |  -0.1  |  -0.1    |  -0.1       |  -0.1
heartwall         | -28.7  | -28.7    | ≈±10        | ≈±10
nn                | +103.5 | +103.5   | +103.5      | ≈±25
nw                | +56.8  |  ≈±5     |  ≈±5        |  ≈±5
streamcluster     |  -1.5  |  -1.5    |  -1.5       |  -1.5
                  |        |          |             |
p50               | 15.0%  | ~12%     | ~10%        | ~12%
p90               | 56.8%  | ~28%     | ~17%        | ~25%
```

**Acceptance:** p50 < 15%, p90 < 25% on the 10-workload set. If C3
doesn't clear nn's p90 contribution, document and fall back to a
conservative estimator (e.g., return the highest measured iteration's
cycle count as an upper bound for the projection) — better to
under-project than triple-over-project.

---

## Order of execution

1. **C1** (tiny-grid + auto-accept) — ship first, lowest risk, fixes
   nw immediately and removes pilot overhead noise from the data.
2. **C2** (K-rep replication fix) — fixes heartwall, also tightens
   backprop. Independent of C1.
3. **C3** (log-fit clamp + adaptive expansion) — only if C1+C2 leave
   nn outside target. The clamp half is cheap and worth shipping
   even if the adaptive-expansion half is deferred.

Estimated total effort: 3 days end-to-end including validation
sweeps. C1 and C2 each ~half a day; C3 ~1.5 days because of the
calibration sweep.

---

## Out of scope (deferred follow-ups)

These remain on the follow-up list in
`CTA_SAMPLING_PHASE_AB_DONE.md` but are not part of this plan:

- **AI weight calibration** (open D2). Tangential to cycle accuracy
  now that the formula is class-independent.
- **Pilot-rejected iteration cleanup of per-SM aggregates.** Affects
  IPC numerator only, not cycle estimate.
- **K-rep clustering replacement** (full k-means redesign). C2's
  evenly-spaced supplement is the cheap version of this. Full
  clustering only worth doing if C2 doesn't close heartwall.
