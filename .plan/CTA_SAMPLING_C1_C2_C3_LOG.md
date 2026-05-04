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

(in progress — appended as the change is implemented)

