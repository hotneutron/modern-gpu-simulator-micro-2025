# Plan: Phased NCU↔GCOM Comparison with Early-Terminated Simulation (Phase 0, 30-min cap)

**Date:** 2026-06-29
**Branch:** `accorde_npu` (~/accorde)
**Status:** Plan only — no code yet.
**Related docs:** `~/accorde/.plan/20260512-0054-pc-sampling-debate.md` (§7.4 is exactly this idea),
`~/accorde/.plan/20260512-0111-simpoint-cta.md` (Variant B = contiguous phase segments),
`~/accorde/.plan/20260512-0154-reflection-simpoint-cta.md`, `~/accorde/docs/checkpointing_and_sampling.md`.

---

## 1. Summary

Build a **phased validation harness** that compares per-phase HW metrics (NCU) against per-phase
GCOM simulator stats, where a **phase = a fixed number of warp/SASS-level instructions**. To bound
cost, GCOM is **terminated early after the first phase** (or a 30-minute wall-clock cap, whichever
comes first). The deliverable for this iteration is: *run phase 0 only, compare NCU(phase 0) vs
GCOM(phase 0), within ≤30 min of sim time per kernel.*

This is the BBV/SimPoint idea reduced to its smallest validating slice: SimPoint/Variant-B says
"decompose a kernel into contiguous instruction-window phases and label each phase's CPI." We are
not yet clustering or projecting — we are validating that **one phase can be measured on both HW
and sim and that they agree**, while proving the **early-termination plumbing** that the full
pipeline will later depend on.

### BBV / SimPoint context (so the harness is built the right shape)
- **BBV (Basic Block Vector):** per-interval opcode/basic-block frequency vector. SimPoint clusters
  BBVs (Variant A) or change-point-segments them (Variant B) to find representative phases.
- **Variant B (contiguous phase segments)** is the relevant model here: phases are
  **non-overlapping, contiguous instruction windows**; no skipping ⇒ no warm-up/cold-state problem.
  Our "fixed N instructions per phase" is the simplest Variant-B segmentation (uniform K).
- The eventual value is per-phase CPI labels; this harness produces the **first** such labeled phase
  and the HW-vs-sim error for it.

### Two confirmed design decisions (from clarification)
1. **Per-phase HW ground truth = NCU range-replay** (nvtx ranges / `--replay-mode application` with
   range markers), not kernel-aggregate. NCU aggregates per launch, so phases must be exposed as
   profilable ranges.
2. **Phase boundary unit = warp/SASS-level instructions**, aligned on both sides. GCOM counts
   thread-level (`gpu_sim_insn += inst.active_count()`, shader.cc:2634), so the GCOM-side
   early-termination threshold must be converted from warp-level N to thread-level, OR a warp-level
   counter must be used (see §4.3).

---

## 2. Current State Analysis (grounded)

### 2.1 GCOM (simulator) — what exists
Simulator tree: `/home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled` (the accorde repo
refers to it as `3rdparty/modern-gpu-simulator-micro-2025`, see
`~/accorde/docs/checkpointing_and_sampling.md:328`).

- **Early-termination knobs** (registered in `gpu-sim.cc::reg_options`, enforced in
  `gpgpu_sim::active()` at `gpu-sim.cc:2970-2992`):
  - `-gpgpu_max_insn` (gpu-sim.cc:2086; enforced 2974-2975) — **cumulative thread-level** insn stop.
  - `-gpgpu_max_cycle` (gpu-sim.cc:2084; enforced 2971-2972).
  - `-gpgpu_max_cta` / `-gpgpu_max_completed_cta` (gpu-sim.cc:2088-2092).
  - These are **global/cumulative**, not per-kernel or per-segment.
- **Instruction counting is thread-level** (`shader.cc:2634`: `gpu_sim_insn += inst.active_count()`).
  So N warp-insns ≠ N `gpu_sim_insn` (factor = mean active lanes, 1–32).
- **No per-instruction-window stat dump** exists. The only periodic hook is
  `-gpgpu_runtime_stat <freq>:<flag>` (gpu-sim.cc:2093-2096; fires on `gpu_sim_cycle % freq` at
  gpu-sim.cc:4093-4139) — **cycle-cadence, and it does not emit cumulative cycle/insn per window.**
- **Kernel filtering**: `-filter_first_kernel_id` / `-filter_last_kernel_id` (gpu-sim.cc:2008-2013;
  applied in trace_parser.cc:421). Lets us isolate the single target kernel (e.g. FA fwd k5).
- **No trace-layer partial-kernel ("first K insn of this kernel") knob** — only the global
  `active()` stops. For a single-kernel trace, `-gpgpu_max_insn` effectively becomes a
  first-K-instructions stop.
- **No wall-clock limit inside the simulator** — accorde already wraps the binary with shell
  `timeout` (`~/accorde/scripts/02_run_simulation.sh:317`, `CASS_SIM_TIMEOUT`, default 120s).

### 2.2 accorde (harness) — what exists
- `~/accorde/scripts/02_run_simulation.sh` already: runs `timeout $CASS_SIM_TIMEOUT $SIMULATOR
  -trace … -config …`, scrapes `gpu_sim_cycle` / `gpu_tot_sim_insn` / `gpu_ipc` from the log
  (lines 317-351), emits per-config JSON. This is the reuse point for sim execution + early stop.
- `~/accorde/scripts/intra_cta/` already has the BBV/segmentation Python:
  `segment_changepoint.py`, `cluster_simpoint.py`, `bbv_cpi_variance.py`, `collect_corpus.py`,
  `validate_p1.py`, plus `run_*_simpoint*.sh`.
- `~/accorde/docs/checkpointing_and_sampling.md` documents the kernel/CTA-granularity sampling and
  explicitly states **no arbitrary instruction-level resume / no cache-state restore** (lines
  379-384). M2 milestone proposes PKP-style internal early-stop on IPC convergence.
- The fwd reference numbers (Opt 5) live in
  `/home/jihyun/modern-gpu-simulator-micro-2025/.result/FA3_kernel_5_fwd.md` (kernel-level NCU↔sim
  table just extended).

### 2.3 The core tension this plan must resolve
NCU is **per-launch aggregate**; GCOM is **thread-level cumulative**; phases are **warp-level**.
The harness therefore needs (a) a way to make NCU emit one number set *per phase* (range-replay),
(b) a warp-level instruction boundary that both NCU ranges and GCOM stop at, (c) a GCOM stat path
that reports phase-0 cycles before being killed.

---

## 3. Proposed Changes

> Scope of THIS iteration: **phase 0 only**, one kernel (FA fwd, trace kernel 5), 30-min cap.
> Everything is additive; no existing behavior changed. Files below are grouped by side.

### Change A — Define the phase grid (offline, Python; accorde)
- **New:** `~/accorde/scripts/phased_compare/define_phases.py`
- **What:** given a kernel's total warp-level instruction count (from NVBit `opcode_hist`/instr
  count already collected, or `nvbit instr_count`), compute phase boundaries as fixed warp-insn
  windows `N` (default `N` chosen so phase 0 ≈ the first contiguous Variant-B segment; start with a
  round number, e.g. 1,000,000 warp-insns, configurable). Emit `phases.json`:
  `{kernel_id, phase_size_warp_insns, phases:[{id, start_warp_insn, end_warp_insn}], total}`.
- **Why:** single source of truth for the boundary, consumed by both the NCU-range step and the
  GCOM early-stop threshold so they bucket identically.

### Change B — Per-phase HW via NCU range-replay (accorde + tiny NVBit marker)
- **New NVBit tool (or reuse pc_interval_sampler backbone):**
  `~/accorde/.../tools/range_marker/` — instruments per-warp instruction count and calls
  `nvtxRangePush("phase_k")` / `nvtxRangePop()` at the warp-level boundaries from `phases.json`.
  (NVBit instruction callback + a host-side nvtx push/pop driven by a global instruction-count
  trigger; per the pc-sampling debate §5 backbone.)
- **New:** `~/accorde/scripts/phased_compare/run_ncu_ranges.sh` — invoke
  `ncu --nvtx --nvtx-include "phase_0/" --replay-mode application --metrics <set> -o ncu_phase`
  so NCU reports the metric set **for the phase-0 nvtx range only**. Capture the same metric family
  used in the kernel-level table (cycles, issue-slots-busy, L2 hit, occupancy, stall reasons).
- **Why:** NCU cannot window a launch by itself; the nvtx range is the only way to get true per-phase
  HW counters. **This is the highest-risk component (see §5 holes H1/H2).**

### Change C — GCOM early termination at phase-0 boundary (config-only first; code only if needed)
- **First attempt (no code):** isolate the kernel with `-filter_first_kernel_id 5
  -filter_last_kernel_id 5`, and stop at phase 0 with `-gpgpu_max_insn <T>` where
  `T = phase_size_warp_insns × mean_active_lanes`. Wrap with `timeout 1800` (30 min) like
  `02_run_simulation.sh:317`.
- **Decision point (warp-level alignment, per clarification):** because the boundary unit must be
  warp/SASS-level but `-gpgpu_max_insn` is thread-level, the thread-level threshold `T` is only
  approximate (depends on mean active lanes). If the resulting phase-0 instruction window does not
  match the NCU nvtx range within tolerance, add a **warp-level stop knob**:
  - **New (code, small):** `-gpgpu_max_warp_insn` in `gpu-sim.cc::reg_options` (mirror
    `-gpgpu_max_insn` at gpu-sim.cc:2086) + a warp-level counter incremented once per issued warp
    instruction (alongside shader.cc:2634) + an `active()` check (mirror gpu-sim.cc:2974-2975). This
    makes GCOM stop at exactly the same warp-insn count NCU's range used.
- **Why:** gives a phase-0-only simulation that ends in minutes, bounded by `timeout 1800`.

### Change D — GCOM phase-0 stat capture (reuse existing print path)
- **What:** GCOM already prints `gpu_sim_cycle`, `gpu_tot_sim_insn`, `gpu_ipc`, and the rich
  per-SM/issue-stage stats at the end of a (terminated) run. Because phase 0 is the *only* thing
  simulated, the end-of-run dump **is** the phase-0 stat. No new per-segment partitioning needed for
  phase 0.
- **New:** `~/accorde/scripts/phased_compare/parse_gcom_phase.py` — scrape the phase-0 stats from
  the GCOM log into `gcom_phase0.json` (cycles, ipc, issue-stage breakdown, L2/L1 rates), reusing
  the grep patterns from `02_run_simulation.sh:323-325`.
- **Why:** avoids the ~200-LoC `print_stats()` segment-partitioning (Variant B §3.5.2) until we
  actually need phase 1+ in the same run.

### Change E — The comparison + report (accorde)
- **New:** `~/accorde/scripts/phased_compare/compare_phase.py` — join `ncu_phase0` and
  `gcom_phase0` by the metric mapping already established in
  `/home/jihyun/modern-gpu-simulator-micro-2025/.result/FA3_kernel_5_fwd.md` ("NCU ↔ GCOM-sim Metric
  Mapping" section), emit a side-by-side `phase0_compare.csv/.md` with `Sim/HW` per metric and the
  cycle/CPI error. Reuse the unit caveats from that doc (IPC aggregated vs per-SM; L1 bypass; insn
  granularity).
- **New:** `~/accorde/scripts/phased_compare/run_phase0.sh` — orchestrates A→B→C→D→E for one kernel
  with a single `--kernel`, `--phase-size`, `--sim-timeout 1800` interface.
- **Why:** one command produces the phase-0 NCU-vs-GCOM comparison.

---

## 4. Assumptions & Decisions

- **Confirmed:** per-phase HW = **NCU range-replay** (Change B); phase unit = **warp/SASS-level**,
  aligned on both sides (Change C decision point / `-gpgpu_max_warp_insn`).
- **Target kernel:** FA fwd (trace kernel 5), to line up with the existing kernel-level table; the
  harness is kernel-agnostic via `phases.json`.
- **"First phase" + 30 min:** phase 0 only; `timeout 1800` is the hard cap (reuse
  `02_run_simulation.sh` pattern). If 30 min is hit before phase-0 boundary, record a **partial**
  result (cycles so far) and flag it — do not discard.
- **No warm-up / no skipping:** phase 0 starts at kernel entry, so there is **zero cold-state /
  state-inheritance error** (this is the Variant-B guarantee). This is *only* true for phase 0; any
  later phase would need warm-up and is explicitly out of scope here.
- **Reuse over rebuild:** prefer config-only GCOM stop (`-gpgpu_max_insn` + `timeout`) and only add
  `-gpgpu_max_warp_insn` if the unit mismatch breaks alignment (§5 H4).
- **Metric set:** reuse the kernel-level mapping doc; do not invent new metrics this iteration.

---

## 5. Poke holes in the plan (risks & failure modes) — user explicitly requested

**H1 — NCU range-replay may not bucket *a single kernel launch* by instruction window the way we
want (HIGH).** `--replay-mode range` / nvtx ranges profile a *range of API/kernel activity*, but a
range *inside one kernel launch* requires device-side nvtx push/pop, which NCU may treat as part of
the same replayed launch rather than a separately-counted range. NCU may also **serialize/replay the
whole kernel multiple times**, so "phase 0 only" gives no HW speedup and the per-phase counters may
include spill from neighboring instructions. *Mitigation / pre-check:* a 1-day spike on a tiny kernel
to confirm NCU emits distinct counters for an intra-kernel nvtx range; if it can't, fall back to the
reflection doc's recommendation — **NVBit `clock64()` per-interval CPI** as the HW phase label (the
option I did not pick, kept as fallback).

**H2 — nvtx push/pop at a warp-level instruction boundary is ill-defined across warps (HIGH).** A
"phase boundary at warp-insn N" is per-warp; warps reach N at different real times, so a single
host-visible nvtx Push/Pop cannot cleanly bracket "phase 0 across all warps." Phase 0 on HW is
therefore fuzzy at its trailing edge. *Mitigation:* define phase 0 by a **grid-wide instruction
count** (sum across warps) or by **CTA-0 only** (as P1 in the SimPoint plan does — one CTA per
kernel), which makes the boundary single-streamed. Recommend **CTA-0-only for phase 0** to make the
boundary well-defined; note this changes the comparison to "phase 0 of CTA 0," not the whole grid.

**H3 — NCU is per-launch aggregate; "phase 0" HW cycles may be unmeasurable in isolation (HIGH).**
Even with ranges, NCU's `Elapsed Cycles` is a launch property. Per-range cycle attribution inside one
launch is not a first-class NCU metric on all GPUs/driver versions. *Mitigation:* validate which
metrics NCU actually supports per nvtx range on this H100 + driver before committing; if cycles can't
be ranged, use issue/stall *ratios* (which are smsp per-active and may range) and treat absolute
cycles as kernel-level only.

**H4 — Warp-level vs thread-level instruction unit mismatch makes the two phase-0 windows different
sets of instructions (MEDIUM).** GCOM's `-gpgpu_max_insn` is thread-level; converting via "mean
active lanes" is approximate and **varies within the kernel** (divergent regions have fewer active
lanes), so a fixed multiplier mis-aligns the trailing edge. *Mitigation:* implement
`-gpgpu_max_warp_insn` (Change C decision point) so both sides stop at the same warp-insn count;
accept the small code addition rather than rely on a lane-factor guess.

**H5 — Early termination changes the simulated dynamics, biasing phase-0 stats (MEDIUM).** Killing
the sim at phase 0 means caches, TMA pipelines, and mbarrier credits are still "warming up";
steady-state effects that NCU's full-launch (even ranged) sees may differ. Phase 0 is *inherently*
the cold-start phase, so GCOM-phase0 vs NCU-phase0 might both be cold — but if NCU's range can't
isolate cold-start, we compare cold (sim) vs warm-ish (HW). *Mitigation:* explicitly label phase 0
as "cold-start phase"; do not generalize its error to later phases.

**H6 — 30-min cap may not even reach phase 0 on large kernels (MEDIUM).** FA fwd phase 0 (1M
warp-insns) at GCOM's ~a few-thousand-cycle/sec rate (`gpgpu_simulation_rate ≈ 3 cycle/sec` seen in
`.o20`) could exceed 30 min if phase 0 is cycle-heavy. *Mitigation:* size phase 0 from a *time
budget* backward (pick `N` so projected sim time < 25 min using the known sim rate), not a fixed
1M; record partial-on-timeout (Change C).

**H7 — GCOM thread-level `gpu_sim_insn` is cumulative across kernels (LOW, but a footgun).** With
`-filter_first/last_kernel_id` isolating k5, the counter still starts at 0 for the launch, so
`-gpgpu_max_insn` works — *but* if the trace has preceding kernels not filtered out, the threshold is
off. *Mitigation:* always pair `-gpgpu_max_insn`/`-gpgpu_max_warp_insn` with the kernel filter and
verify `gpu_tot_sim_insn` at stop ≈ threshold.

**H8 — The comparison is only as meaningful as the metric mapping (LOW).** The kernel-level mapping
doc already flags IPC definition mismatch, L1-bypass, and insn-granularity. Per-phase makes these
*worse* (e.g. phase-0 occupancy is ramp-up). *Mitigation:* carry the same caveats; report ratios and
trends, not just absolute deltas.

**H9 — "Phase = fixed instruction count" is not a real BBV/SimPoint phase (LOW/conceptual).** Uniform
windows ignore the actual phase structure that change-point detection (Variant B) would find; phase
0's boundary may fall mid-phase. *Mitigation:* acceptable for this validating slice; note that the
real pipeline uses `segment_changepoint.py`, and a fixed window is the deliberate simplification.

**H10 — Two NVBit/NCU passes + GCOM is a 3-tool pipeline; reproducibility/version drift (LOW).**
*Mitigation:* pin driver/NCU/NVBit versions in `run_phase0.sh`; record them in the output JSON like
`02_run_simulation.sh` records config.

---

## 6. Verification

1. **Boundary alignment:** GCOM stop point (`gpu_tot_sim_insn` or warp-insn counter at termination)
   equals the phase-0 boundary in `phases.json` (within 1 warp-insn for `-gpgpu_max_warp_insn`, or
   document the lane-factor error for `-gpgpu_max_insn`).
2. **Time bound:** GCOM phase-0 run wall-time < 30 min (or partial-flag set), proven by the
   `timeout 1800` wrapper exit code (124 = timeout, per `02_run_simulation.sh:355`).
3. **NCU range sanity (the H1/H3 gate):** confirm NCU emits distinct phase-0 metrics for the nvtx
   range on a toy kernel *before* trusting FA numbers; if not, switch to clock64() fallback.
4. **Comparison output:** `phase0_compare.md` lists each mapped metric with NCU(phase0),
   GCOM(phase0), Sim/HW, reusing the FA3_kernel_5_fwd.md mapping; cycle/CPI error computed.
5. **No regression:** running the harness does not modify the simulator default behavior
   (flag-off / no new flags set ⇒ identical to today); `git diff` on the sim tree is empty unless
   `-gpgpu_max_warp_insn` was added, in which case flag-default-off reproduces baseline.

---

## 7. Suggested execution order (smallest-risk-first)

1. **Spike H1/H3 first** (1 day): does NCU range-replay give per-nvtx-range metrics inside one launch
   on this H100? This gates the whole approach. If no → fallback to clock64() per-phase.
2. Change A (`define_phases.py`) — cheap, unblocks everything.
3. Change C config-only path (`-filter_*` + `-gpgpu_max_insn` + `timeout 1800`) on FA k5; measure
   phase-0 sim time; size `N` from the time budget (H6).
4. Change D + E (parse + compare) against **kernel-level** NCU first (free, no range risk) to prove
   the plumbing, then swap in NCU-range phase-0 numbers once the spike passes.
5. Add `-gpgpu_max_warp_insn` only if H4 alignment is out of tolerance.
