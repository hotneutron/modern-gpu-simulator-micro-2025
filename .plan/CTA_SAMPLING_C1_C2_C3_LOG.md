# CTA Sampling — C1 / C2 / C3 progress log

Live log of the wider-validation-gap closure (plan in
`.plan/CTA_SAMPLING_NEXT_PLAN.md`). Each section is appended
as the change ships; surprises and rejected approaches are kept
inline so the eventual writeup has the full audit trail.

---

## Starting point

After the wider 10-workload sweep on `cta-sampling`:

```
              | err%
hotspot       | +14.7
backprop      | -15.4
pathfinder    |  -2.6
bfs           |  -9.9
srad_v2       | +17.3
lud           |  -0.1
heartwall     | -28.7
nn            | +103.5
nw            | +56.8
streamcluster |  -1.5

p50 = 15.0% (just over <15%)
p90 = 56.8% (over <25%)
```

Three failure modes (heartwall, nn, nw); plan calls for three
layered changes (C1, C2, C3), shipped cheap-to-expensive.

---

## C1 — skip pilot for tiny grids + auto-accept full-grid samples

**Targets:** nw (+56.8%).

### Code

`simulator-remodeled/gpu-simulator/main.cc` only, ~30 LOC across two
edits:

- **C1a** — In `pilot_decide_accept` iter 0, hoist
  `if (ps.ctas_launched >= pst.total_ctas) return true` *above* the
  class check. Memory/mixed kernels that already saw the full grid now
  accept at iter 0 instead of re-running iter 1 with the same CTA set.
- **C1b** — In the pilot-state init block, only create `pst` when
  `total_ctas >= total_sms`. Tiny grids fall through to a single
  K-rep wave; the `accept = pilot_decide_accept(...)` block then
  no-ops because `pilot_states.find()` returns `end()`, leaving
  `accept = true` (the default).

### Result

```
              | before  | after
hotspot       | +14.7   | +14.7   (unchanged)
backprop      | -15.4   | -15.4   (unchanged)
pathfinder    |  -2.6   |  -0.3   (small grid, pilot skipped)
bfs           |  -9.9   |  +3.3   (per-kernel grids are small)
srad_v2       | +17.3   | +17.3   (unchanged)
lud           |  -0.1   |  -0.0   (small grid, pilot skipped)
heartwall     | -28.7   | -28.7   (C2's job)
nn            | +103.5  | +103.5  (C3's job)
nw            | +56.8   |  +5.0   <-- fixed, on target
              |         |
p50           |  15.0%  |   9.9%  ← improved, met
p90           |  56.8%  | 103.5%  (nn dominates, unchanged)
```

### Notes / surprises

- Side-benefit on `pathfinder` / `bfs` / `lud`: their per-kernel
  grids are below `total_sms` so C1b also caught them and removed
  unused iter-1 work. Net result is each shifted toward zero; none
  regressed past target.
- bfs sign flipped (-9.9 → +3.3). Both within target; the magnitude
  shrank, which is the desired direction. Cause: previously bfs was
  paying iter-1 cycles on multiple memory-classified kernels at full
  grid; now it just runs them as iter 0.
- The gpgpu-sim build did NOT need a clean rebuild for C1 (no
  struct-layout change), so iteration was fast.

### Commit

`4fe81c6` — *CTA sampling: C1 -- skip pilot for tiny grids + auto-accept
full-grid samples*

---

## C2 — fix K-rep replication: never duplicate before exhausting unique CTAs

**Targets:** heartwall (−28.7%); also tightens any kernel where K-rep
< target < total_ctas (backprop K1 with K=3, target=80).

### Code

`simulator-remodeled/gpu-simulator/main.cc` only, ~75 LOC. Three
regimes in `expand_sampled_ctas` (signature now takes `gx, gy, gz`):

- `target <= K`: return reps unchanged.
- `target >= total_ctas` (**C2a**): enumerate every unique CTA in
  the grid; the sample IS the full grid. No replication.
- `K < target < total_ctas` (**C2b**): start with the K reps, then
  append evenly-spaced fresh unique CTAs from a linearized non-rep
  pool until the target is hit. Defensive duplication fallback only
  if the pool is exhausted (theoretically impossible given the regime
  check).

Both call sites (`update_sampling_on_trace_info`, the single-shot
sampler) updated to pass grid dims.

### Result

```
              | after C1 | after C2
hotspot       | +14.7    | +14.7    (unchanged)
backprop      | -15.4    | -13.2    (slight tightening; K1 had K=3)
pathfinder    |  -0.3    |  -0.3    (unchanged, small grid skipped)
bfs           |  +3.3    |  +3.3    (unchanged, per-kernel small)
srad_v2       | +17.3    | +18.1    (within noise)
lud           |  -0.0    |  -0.0    (unchanged, small grid skipped)
heartwall     | -28.7    | -23.4    <-- improved 5pp
nn            | +103.5   | +128.4   <-- regressed 25pp
nw            |  +5.0    |  +5.0    (unchanged)
              |          |
p50           |   9.9%   |  13.2%   (still met, <15%)
p90           | 103.5%   | 128.4%   (nn worsened)
```

### Notes / surprises

- **The big one: nn regressed by 25 percentage points.** This was
  *not* expected from the plan. Investigation showed the corner-replica
  bias from the pre-C2 sampler was *coincidentally* inflating the
  log-fit slope and partially canceling the under-extrapolation
  problem.

  Pre-C2 nn iter-3 sampled 80 corner-replicas in 4774 cycles → per-SM
  throughput at N=2 = 4.19e-4. Post-C2 nn iter-3 sampled 80
  grid-spread CTAs in 5430 cycles (more representative work, slower
  sample) → per-SM throughput at N=2 = 3.68e-4. The log fit's slope
  `b` dropped from 5.85e-4 to 4.77e-4, so the extrapolated T(N=23.45)
  for the full grid dropped from 17.47e-4 to 13.69e-4 — predicting
  *less* full-grid throughput, hence *more* cycles, hence higher
  cycle estimate.

  Both samples extrapolate the SAME real full-grid behavior, but in
  opposite-bias directions. Pre-C2 was wrong at the sample side
  (CTA selection bias) but happened to land closer to the true
  full-grid cycle count. Post-C2 is right at the sample side but the
  log-fit extrapolation distance issue (the actual root cause for
  nn) is now fully exposed. This is exactly what C3 is designed to
  address.

- heartwall improvement (5pp) is real but smaller than hoped. Remaining
  −23.4% is cache-state pollution between pilot iters (rejected iter
  contents stay in L2 etc., warming the accepted iter). C2 doesn't
  touch that.
- backprop K1 (K=3, target=80) tightened slightly because its sample
  is now 3 reps + 77 unique grid-spread CTAs instead of 27 copies of
  3 corner reps.
- srad_v2's 1pp shift is within noise; the K-rep set differs only in
  shuffle ordering for srad_v2's 8x8 grid (K-rep returns 9 = total
  for that case... actually let me note that: srad_v2 K-rep covered
  all 9 of the unique CTAs of the 9-rep heuristic on its 8x8x4 grid;
  the change here was the iter-2 expansion to 64 CTAs, which now
  enumerates uniquely).

### Commit

`31a4dd2` — *CTA sampling: C2 -- fix K-rep replication, sample fresh
unique CTAs first*

---

## C3 — bound log-fit extrapolation + adaptive deeper expansion

**Targets:** nn (+128.4% post-C2). Last of the three planned changes.

### Code

(in progress)

