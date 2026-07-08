# Plan: Fast FA3 Simulation via CTA Subsetting, Validated by Per-PC Stall Fit

**Date:** 2026-06-29, redesigned 2026-07-07. **Branch:** `accorde_npu` (~/accorde). **Status:** plan only, no code.

## 1. Goal and two-stage structure

Run GCOM **faster** on FA3 (full runs exceed the budget, ~12 h) by simulating a **CTA subset**, and prove the
subset is trustworthy. The trap (P1) is that comparing a K-CTA sim directly to full HW confounds two errors —
**subset error** (fewer CTAs) and **model error** (sim ≠ HW). They must be separated:

- **Stage 1 — pick K (speed).** K-sim vs **full-sim**: identical model, so any difference is *pure subset
  error*. Cheap, config-only, uses existing aggregate counters. This alone answers "how small can K be."
- **Stage 2 — model fidelity.** full-sim vs **full-HW**, per-PC stall distribution: identical scale, so any
  difference is *pure model error*. This is the harder per-PC track (Change B′).

Stage 1 unblocks the speed goal without any C++; Stage 2 is a separable accuracy track. Do **not** fuse them
into one K-sim-vs-full-HW fit.

---

# Stage 1 — Pick K by CTA subsetting (speed)

## 2. Cut axis = CTA subset, not instruction window

- **Instruction window (`-gpgpu_max_warp_insn`) — do NOT use.** It sees every CTA's prologue + early
  mainloop only, never epilogue/steady-state ⇒ cold-start biased; can't represent the full kernel.
- **CTA subset (`-gpgpu_max_cta K`, run K CTAs to completion) — correct.** Each CTA runs its full
  prologue→mainloop→epilogue lifetime; CTAs are independent ⇒ **no deadlock** (avoids the FA3 truncation
  problem in §6). Config-only for the basic case: `-gpgpu_max_cta`
  ([gpu-sim.cc:2227](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L2227),
  [hit_max_cta_count:2380](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L2380)).
- Speedup roughly tracks active-SM count: FA3 bwd is 384 CTA / 132 SM ≈ 2.9 waves; icnt/L2/DRAM fixed cost
  is the floor — measure.

## 3. Causal FA3 caveat — representative sample, not first-K (P3)

Causal masking makes CTAs **non-uniform**: mainloop trip count grows with query-block position. First-K (and
CTA-0 specifically) is the *shortest, least-contended* case — unrepresentative. Need a **representative CTA
sample** spanning the causal distribution. `-gpgpu_max_cta` only limits count in launch order, so
arbitrary-subset selection is a small **extension** (CTA-id allow-list); precedent in the checkpoint/resume
CTA-range gate
([gpu-sim.cc:3964](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3964)); accorde has CTA-sampling infra (`cluster_simpoint.py`, `checkpointing_and_sampling.md`).
CTA-0-only is a **bring-up** convenience only, never the comparison config.

## 4. The K-selection loop (K-sim vs full-sim)

1. Run full-sim **once** as calibration (reuse an existing 12 h run; memory shows `gpu_sim_cycle=336,579`).
   Amortized over all later fast runs.
2. Run K-sim for increasing K (representative sample) → compare to full-sim on **existing aggregate
   counters** (the `m_sm_stats` issue-stall totals at
   [subcore.cc:742-757](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L742-L757),
   plus L2/DRAM/IPC), all normalized per-CTA or per-cycle.
3. **Smallest K whose normalized stats match full-sim within tolerance = the fast proxy.** No per-PC code
   needed for this stage.

**Speedup is contention-bounded (P4).** Intra-CTA stalls (scoreboard, tensor-pipe, mbarrier) survive small
K; inter-CTA/memory stalls (`long_scoreboard` from L2/DRAM-contended TMA arrival) only exist at large K.
FA3's interesting stalls *are* the memory/mbarrier ones, so K may be pushed high and the speedup modest.
**Quantify this first** (§9) before investing in Stage 2 — if K must be near-full, the speed goal is
limited and Stage 2's value shifts to pure fidelity study.

---

# Stage 2 — Model fidelity by per-PC stall fit (full-sim vs full-HW)

## 5. Comparable signal = per-PC warp-issue-stall distribution

HW attributes stalls by **PC** (spatial). An instruction-window phase is temporal and has no HW counterpart
(FA3's mainloop revisits the same PCs; PC sampling can't separate visits). So the comparison unit is a
**PC region**, and GCOM must also bucket stalls by PC. Aggregate PMCs (L2/DRAM/tensor) stay kernel-level.

## 6. Why HW can only give PC sampling (not sub-kernel counters)

- **NCU aggregates per launch and replays the whole kernel** — `Elapsed Cycles`, `sm__/lts__/dram__` are
  launch properties; an intra-launch nvtx range is not separately counted on H100.
- **FA3 cannot be truncated mid-flight** — warp-specialized producer/consumer with TMA `expect-tx` + mbarrier
  phase-parity across 384 CTAs ([FA3-enablement.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.plan/FA3-enablement.md)).
  Early-return ⇒ consumers hang; predicate-off ⇒ warps spin to the end. No deadlock-free prefix on HW.
- **Consequence:** HW's only sub-kernel signal is **PC sampling** — one full normal run recording
  `(warp PC, stall reason)`.

## 7. GCOM already computes an NCU-aligned stall taxonomy — but on the wrong cycles (P2)

[subcore.cc:409-431](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L409-L431)
classifies why each non-issuing warp is blocked, with comments mapping to NCU
`smsp__average_warps_issue_stalled_<reason>`. Taxonomy:

| GCOM counter (`...stall_at_least_one_warp_...`) | meaning | NCU reason (`smsp__pcsamp_warps_issue_stalled_*`) |
|---|---|---|
| `waiting_wait_barrier` | mbarrier (TMA data arrival) | `long_scoreboard` (+`barrier`) |
| `waiting_inst_barrier` | named barrier / LDGDEPBAR | `barrier` / `long_scoreboard` |
| `waiting_tma_flush` | `cp.async.bulk.wait_group` drain | `long_scoreboard` / `drain` |
| `waiting_scoreboard` | traditional RAW/WAR | `short_scoreboard` |
| `waiting_l1c` | constant-cache miss | `short_scoreboard` / `imc_miss` |
| `waiting_stall_count` | fixed-latency dep | `wait` |
| `with_fu_occupied` (+`_tensor`/`_sfu`/`_sp_int_dp`) | pipe busy | `mma` / `math_pipe_throttle` |
| `waiting_result_queue_full` | RF/result-queue backpressure | `mio_throttle` / `lg_throttle` |
| `waiting_yield` | YIELD | `drain` / `not_selected` |
| `..._no_valid_instruction_*` (frontend/L0I/ibuffer) | fetch/decode not ready | `no_instruction` |
| *(missing)* | eligible warp not picked this cycle | **`not_selected`** |

**Three gaps, not one:**
- **Gap A — aggregate, not per-PC.** Flat per-SM `m_sm_stats.m_stats_map[...]`
  ([new_stats.h](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/new_stats.h#L172-L197)), no PC key.
- **Gap B — per-cycle boolean OR, not per-warp count.** Increments `is_any_waiting_X` ("≥1 warp"), NCU
  counts *warps*.
- **Gap C (the P2 crux) — recorded only on no-issue cycles.** The attribution block is inside the
  `else` at [subcore.cc:736-757](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L736-L757),
  reached only when nothing issued, and the issue loop `break`s on first issue
  ([subcore.cc:638](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L638)).
  NCU PC-sampling samples **all** cycles/warps, including eligible-but-`not_selected` on issue cycles.
  Differing populations ⇒ **normalizing to shares does NOT fix it**.

## 8. Change B′: per-PC, per-warp, every-cycle stall pass

Behind `-gpgpu_perpc_stall_instrument_enable` (default 0 ⇒ baseline unchanged):

1. **Separate stall-state pass (fixes Gap C):** every cycle, classify **every** resident warp's head-PC
   stall state — decoupled from the issue loop's early `break`. Add a `not_selected` reason for warps that
   were eligible but not the issued winner. This is a **new per-warp pass**, more than "add a counter to the
   existing loop."
2. **Per-PC, per-warp counts (Gaps A/B):** `stall_by_pc[pI->pc][reason] += 1` once per warp per cycle,
   per-SM `unordered_map<pc, array<u64, NUM_REASONS>>`. Also `issued_by_pc[pc]` for the denominator.
3. Dump one row per PC to `gcom_stall_by_pc.csv` at kernel end / `timeout`. **No aggregation in C++** —
   PC→region bucketing is offline.
4. Reuse the Stage-1 representative-CTA sample so this runs at the chosen K, not full (still deterministic).

## 9. Regions must be per warp-role (P5)

FA3 is warp-specialized: producer (TMA) and consumer (WGMMA) warps run **disjoint instruction streams**
behind a warp-id branch. There is no single linear mainloop PC range. `define_regions.py` emits **PC sets
per role** (`{role, region, pc_set}`), not linear `[lo,hi]` ranges, or the "mainloop" blends two different
stall profiles. Seed from `extra_info/sass` (same disassembly as `extract_tma_descriptor_offsets.py`).

## 10. Comparison statement (apples-to-apples)

Per (role, region): HW `smsp__pcsamp_*` sample counts and GCOM `stall_by_pc` cycle counts each summed into
the **same** regions, **normalized to a reason-share vector**; compare vectors + one divergence number
(cosine/L1). **Do not compare absolute cycles vs samples** — HW statistical, sim exact; only the reason
*distribution* is commensurable. L2/DRAM/tensor aggregates stay kernel-level.

---

## 11. Execution order

1. **Spike (gating, ~1 day) — three checks, all must pass:**
   (a) does `smsp__pcsamp_warps_issue_stalled_*` populate per-PC on this H100 + driver?
   (b) **does NCU's SASS-PC space align with the sim trace-PC space** (P6) — same cubin/function base, not
   just "PC exists"? If not, regioning silently misbuckets.
   (c) **speedup vs K curve + K-sim-vs-full-sim aggregate match** (Stage 1, P4) — is there a small faithful
   K at all, or is FA3 too contention-bound to subset? This gates whether Stage 2 is even worth it.
   Fallbacks if (a)/(b) fail: CUPTI PC Sampling API, or `clock64()` per-region CPI (coarser).
2. **Stage 1:** representative CTA-sample selection (allow-list extension if needed) → pick smallest faithful
   K on existing counters. **Delivers the speed goal.**
3. **Stage 2 code:** Change B′ (Gap-C every-cycle pass + per-PC counts + `not_selected`) behind the flag;
   verify flag-off ⇒ identical baseline.
4. `stall_taxonomy.json` + `define_regions.py` (per-role) + `run_ncu_pcsamp.sh`.
5. Parse + compare: kernel-level taxonomy join first (no regioning), then per-(role,region) full-sim vs
   full-HW.

## 12. Risks

- **R1 (HIGH→mitigated) — confounded fit:** dissolved by the two-stage split (subset error vs model error
  measured separately).
- **R2 (HIGH) — Gap C population mismatch:** the every-cycle per-warp pass (§8.1) is required; without it,
  GCOM and NCU stall populations differ and shares are not comparable.
- **R3 (MED) — speedup may be small (P4):** FA3 is memory/mbarrier-bound; faithful K may be near-full.
  Quantify in spike 1(c) before building Stage 2.
- **R4 (MED) — PC alignment (P6):** verify in spike 1(b), not assumed.
- **R5 (MED) — GCOM↔NCU reason many-to-many:** may need a coarser grouped axis (memory-dep / pipe-busy /
  sync-barrier / frontend / compute-dep / not-selected) before comparing.
- **R6 (LOW) — head-PC vs blocking-PC:** confirm the pass keys on the stalling head instruction.
- **R7 (LOW) — same PC, different dynamic visit:** PC-region bucketing merges all loop visits on both sides
  ⇒ consistent (region = steady-state, not one iteration).

## 13. P1–P6 resolution map

Origin of the two-stage redesign: the six problems found while stress-testing the prior single-fit plan.

| # | Problem | Resolved in | How | Residual risk |
|---|---|---|---|---|
| **P1** | K-sim-vs-full-HW fit confounds *subset error* + *model error* | Structure (§1) → Stage 1 & 2 split | Stage 1 = K-sim vs **full-sim** (subset error only); Stage 2 = **full-sim** vs full-HW (model error only); full-sim calibrated once (§4.1) | R1 (mitigated) |
| **P2** | GCOM logs stalls only on **no-issue** cycles; NCU samples **all** cycles/warps (+ missing `not_selected`) | Stage 2 (§7 Gap C, §8.1) | New **every-cycle per-warp stall pass**, decoupled from issue-loop `break`; adds `not_selected` | R2 (HIGH — open) |
| **P3** | CTA-0-only default, but CTA-0 is causally biased | Stage 1 (§3) | CTA-0-only demoted to **bring-up only**; comparison uses a **representative CTA sample** | — |
| **P4** | Interesting FA3 stalls are contention-bound → forces large K → small speedup | Stage 1 (§4) + spike 1(c) | States speedup is contention-bounded; **quantify in spike before Stage 2** | R3 (MED — measure) |
| **P5** | Linear prologue/mainloop/epilogue regions don't fit warp-specialized FA3 | Stage 2 (§9) | `define_regions.py` emits **PC sets per warp-role** (producer/consumer), not `[lo,hi]` | — |
| **P6** | NCU SASS-PC ↔ sim trace-PC alignment assumed | Gating spike 1(b) | Alignment is an explicit **gating check** (same cubin/function base), not assumed | R4 (MED — verify) |

**Stage-level view:**
- **Gating spike (§11.1)** clears feasibility: 1(a) PC-sampling exists, 1(b) P6 alignment, 1(c) P4
  speedup-vs-K. All must pass before Stage 2 code.
- **Stage 1** (speed, no C++) resolves P3, quantifies P4.
- **Stage 2** (the `subcore.cc` change) resolves P2, P5; justified only after spike 1(c).
- **P1** is resolved structurally — it is why the two stages exist.
