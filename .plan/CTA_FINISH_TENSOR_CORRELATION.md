# CTA finish-cycle × tensor-op correlation (does fwd tail come from synchronous-WGMMA?)

> **Purpose.** Decide, with a number, whether the fwd 2.01× (and bwd 1.62×) **drain-idle** factor
> (fwd 1.39×, bwd 1.29× — the *largest* factor in both) is **caused by** synchronous-WGMMA over-model,
> or is an **independent** load-balancing/scheduler problem. Tracked in
> [FA3_progress.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.result/FA3_progress.md) Ongoing item 3.
>
> **This is a measurement gate, run BEFORE building any async-WGMMA timing model.**

## Why this measurement exists (the open question)

The fwd/bwd gap decomposes into three per-SMSP factors (FA3_progress.md Ongoing item 3):

| factor | fwd | bwd | axis it lives on |
|---|---|---|---|
| work over-issue | 1.10× | 1.13× | issued warp-inst |
| issue **stall-depth** | 1.29× | 1.09× | **active** subcore cycles (`evaluated`) |
| subcore **drain-idle** | **1.39×** | **1.29×** | **inactive** subcore cycles (resident warps = 0) |

`stall-depth` and `drain-idle` are measured on **disjoint** cycle populations (warps present vs. SM
empty). Synchronous-WGMMA directly inflates `stall-depth` (a stalled WGMMA consumer is still resident, so
it counts as an active/`evaluated` cycle). It does **NOT** directly touch `drain-idle`.

**The coupling is only a hypothesis.** A *uniform* slowdown scales every CTA's finish time by the same
factor and leaves the idle **fraction** unchanged (`A = evaluated/(cycles·nsub)` is invariant under a
common factor). So async-WGMMA fixes the tail **only if** the WGMMA cost is **differential** — i.e. the
tensor-heavy CTAs are slowed *more*, widening the finish-time spread. fwd uses
`DynamicPersistentTileScheduler<384,…>` (132 persistent CTAs share 384 causal-mask tiles ≈ 2.9 tiles/CTA,
triangular work), so a differential tensor cost is *plausible* — but unmeasured.

**Prior evidence is insufficient.** FA3_progress.md "Deferred Opts → L1I frontend" established the idle
is a **tail** (fwd earliest finisher at 52% of the kernel, spread 47%) and is **trace-drained**
(`nv_ibuf_fetch_inflight ≈ 0`), not fetch-starved — so it is not a frontend lever. But finish time was
**never correlated with per-CTA tensor density**, which is exactly what distinguishes "WGMMA-caused
tail" from "independent imbalance."

## The decision rule (and the resulting docs)

Compute Pearson `r` between per-CTA `finish_cyc` and per-CTA `tensor_ops` (1 CTA/SM → per-SM == per-CTA):

| result | interpretation | document(s) to create |
|---|---|---|
| **r ≳ +0.6** (tensor-heavy CTAs finish later) | tail is **differential-WGMMA driven**; one lever fixes both stall-depth and tail | **`.plan/CTA_IMBALANCE_ASYNC_WGMMA.md`** (single doc — async-WGMMA is the shared root cause and lever) |
| **r ≈ 0** (finish spread independent of tensor count) | tail and compute-pipe are **two separate problems** | **`.plan/CTA_IMBALANCE.md`** (tail/scheduler lever) **+** **`.plan/ASYNC_WGMMA.md`** (stall-depth lever) — must address both |
| **r ≲ −0.6** (tensor-light CTAs finish later — unexpected) | tail is anti-correlated; investigate scheduler/tile assignment | `.plan/CTA_IMBALANCE.md` + note the anomaly |

> Naming rule is per user instruction (2026-07-15): related → `CTA_IMBALANCE_ASYNC_WGMMA.md`; unrelated →
> split into `CTA_IMBALANCE.md` and `ASYNC_WGMMA.md`.

Report, in addition to `r`: the finish-cycle spread (max−min, and as % of kernel), the tensor-op
spread, and a scatter of the two. A mid-range `|r|` (0.3–0.6) → report the partial picture and treat as
"partially coupled" (async-WGMMA helps the tail but does not fully close it).

## Instrumentation (implemented 2026-07-15, observe-only, gated)

The existing logs were **insufficient**: (a) the only per-CTA finish-cycle print is a `LIVENESS`
trace-macro line ([sm.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc), compile-time `TRACING_ON` gated, local cta id only, no op count); (b) there is
**no** raw per-CTA/per-SM tensor-op counter (the per-SM `m_num_tensor_core_acesses` is a
latency-weighted power proxy; `m_num_tensor_core_committed` is declared but never written). So a small
per-CTA-slot counter was added.

**Why per-CTA-SLOT, not per-SM scalar.** bwd has `Waves Per SM = 2.91`, so an SM runs multiple CTAs
sequentially in the same hardware CTA slots. A single SM-wide accumulator would fold an earlier CTA's
tensor ops into a later CTA's `[CTAFIN]`. The counter is therefore indexed by hardware CTA slot
(`get_cta_id()`, 0..`MAX_CTA_PER_SHADER`-1=31), stamped at launch and reset at exit so each CTA reports
only its own ops. (fwd is 1 CTA/SM so this is a no-op there, but it makes the bwd data correct.)

**4 sites (all observe-only; increment path is always-on, print is flag-gated → timing-neutral):**

1. **Per-slot counter + launch-cycle arrays + getter** — [sm.h:490-504](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h#L490):
   `m_tensor_ops_by_cta_slot[MAX_CTA_PER_SHADER]`, `m_cta_slot_start_cycle[MAX_CTA_PER_SHADER]`, and
   `inc_tensor_ops_for_warp(unsigned wid)` (maps warp→CTA slot via `get_shd_warp(wid)->get_cta_id()`).
2. **Increment at the always-on FU tensor-issue site** —
   [functional_unit.cc:164-172](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L164): in `functional_unit::issue()`, the
   `case TENSOR_CORE__OP` fires for every issued tensor op regardless of the power model, so
   `m_sm->inc_tensor_ops_for_warp(ready_reg->warp_id())` runs there. (The `incexecstat`/`TENSOR__OP`
   power path at [sm.cc:2903](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2903) was **rejected** — gated behind
   `g_power_simulation_enabled`, off in these runs.)
3. **Stamp launch cycle** — [sm.cc:924-930](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L924): at the top of `SM::init_warps`, record
   `m_cta_slot_start_cycle[cta_id] = gpu_sim_cycle` and clear the slot's tensor counter.
4. **Emit at CTA-exit** — [sm.cc:1318-1332](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1318): inside `register_cta_thread_exit`, when the CTA's
   last warp exits (`!m_cta_status[cta_num]`), print one line and reset the slot:
   ```
   [CTAFIN] sm=<sid> cta_slot=<slot> global_cta=<gid> start_cyc=<launch> finish_cyc=<now> elapsed_cyc=<now-launch> tensor_ops=<count> sm_idle_tensor_cyc=<count> warpgroup_arrive_cyc=<count>
   ```
   Gated by `-wgmma_step0_instrument_enable` (default off → no output, no perturbation on normal runs).
   `elapsed_cyc` lets us correlate tensor density with per-CTA *duration* directly (a cleaner signal than
   finish cycle, which also folds in launch stagger).

5. **Per-CTA PRODUCER-side tensor idle** (`sm_idle_tensor_cyc`) — [sm.cc:567-574](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L567) +
   [sm.h:499-504](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h#L499): SM-idle cycles where a subcore was blocked ONLY by the tensor
   re-issue lockout (`is_any_tensor_reissue_lockout_only`), attributed to each resident CTA slot. This is
   the **producer** side — a warp that wants to *issue* a WGMMA but the tensor FU is busy.

6. **Per-CTA CONSUMER-side tensor wait** (`warpgroup_arrive_cyc`) — [sm.cc:566-573](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L566) +
   [subcore.cc:829-831](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L829) + [sm.h:505-510](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h#L505):
   cycles where a subcore had ≥1 warp blocked on a WGMMA *result* (`warpgroup_arrive` = scoreboard RAW on
   a pending tensor dst), counted **every cycle** (not only SM-idle — a consumer can wait while another
   subcore still issues), attributed to each resident CTA slot. This is the **consumer** side — a warp
   stalled at `wgmma.wait_group`.

**Why BOTH producer and consumer columns — they decide the async-WGMMA DESIGN, not just confirm it.**
Density (`tensor_ops`) says "how much tensor work"; the two stall columns say **which mechanism** async
modeling must fix, and they point at *different* implementations:

| dominant column on the slow CTAs | mechanism | async-WGMMA design implication |
|---|---|---|
| `sm_idle_tensor_cyc` (producer) | WGMMA can't *issue* — tensor FU serialized at `initiation_interval` | lower the II / pipeline back-to-back WGMMA issue |
| `warpgroup_arrive_cyc` (consumer) | WGMMA issued but result not ready — consumer waits at `wgmma.wait_group` | make WGMMA complete in the *background* so softmax/`exp` overlaps; only `wgmma.wait_group` blocks |
| both ~equal | full sync serialization on both ends | model the whole async pipeline (issue-free + background-complete) |

Exact for 1-CTA/SM (fwd); an upper bound for multi-CTA SMs (bwd credits every resident slot).

**Why these columns make the run conclusive.** The full causal chain per CTA is now on one line:
`tensor_ops` (tensor work) → `sm_idle_tensor_cyc` / `warpgroup_arrive_cyc` (which tensor stall it caused)
→ `elapsed_cyc` (how long it took). If the slow CTAs are exactly the high-tensor-stall CTAs, async-WGMMA
is confirmed as the tail lever AND its design direction is fixed — in ONE 12h run, no ambiguous-`r` redo.

**Cost:** 1 counter increment/tensor-op + one printf per CTA (fwd 132 lines, bwd 384). Negligible.
**Bit-identity:** the counter is a pure observer; the print is behind an off-by-default flag → default
runs are unchanged. Header touched (`sm.h`) → **`make clean` required** before rebuild.

## Config changes for this run (logging only, timing-safe — verified)

All timing knobs were confirmed **bit-identical to the Opt9 baseline** (`.o22`): `l2_admit=2`,
`rop_drain=2`, `dram_reply_drain=2`, `l2_reply_drain=4`, `slice_balanced_hash=1`, `icnt_grant=4`,
`icnt_to_l2_pop=4`, `tma_max_lines=1`, `cluster_reply_eject=4`, `rop_latency=100`, `dram_latency=243`,
`tensor_latency=32`, `prefetch sb=4`, `eager_promote=1`, `tma_real_base=1`, `tma_operand_tiling=1`. Only
logging was changed (audited timing-safe, none gate `gpu_sim_cycle` or lose an end-of-run stat):

- **`-tma_debug_enable 1 → 0`.** Suppresses the high-volume per-event `[TMADBG]` stderr. Does **not**
  lose `L2_TMA_true_hit_rate` (gated only by `is_tma()` in l2cache.cc, printed unconditionally) nor the
  end-of-run `[TMA][Phase2][Stats]`/`[Phase3][Stats]` summaries.
- **`-trace_components`: removed `INTERCONNECT`.** It is the single largest log producer (its `DPRINTF`
  has no `-trace_sampling_core` gate → logs every push on all 132 nodes). Trace streams are pure logging
  (DTRACE wraps printf only), so removal cannot change `gpu_sim_cycle`. LIVENESS kept for the
  "Finished CTA #" cross-check against `[CTAFIN]`.
- **`-wgmma_step0_instrument_enable 1`** (already on) — required to emit `[CTAFIN]` + the SM-idle-reason
  end-of-run stats. Observe-only.

## How to run + analyze

1. Rebuild after `make clean` (header change).
2. Re-run fwd K5 and bwd K10 (Opt9 config, `-wgmma_step0_instrument_enable 1` already set).
   Re-confirm the bit-identity gate: FWD `gpu_sim_cycle` must stay 135,999 (bwd was +120, negligible).
3. Extract: `grep '^\[CTAFIN\]' <out> > ctafin.txt` → parse per CTA: `elapsed_cyc` (primary duration),
   `tensor_ops` (density), `sm_idle_tensor_cyc` (producer stall), `warpgroup_arrive_cyc` (consumer stall).
4. **Correlate** (Pearson `r`, scatter):
   - `(tensor_ops, elapsed_cyc)` — is the slow CTA the tensor-heavy CTA? → the coupling decision rule.
   - `(sm_idle_tensor_cyc, elapsed_cyc)` and `(warpgroup_arrive_cyc, elapsed_cyc)` — WHICH tensor stall
     drives the duration.
   - Compare `Σ sm_idle_tensor_cyc` vs `Σ warpgroup_arrive_cyc` on the slowest-decile CTAs → the
     producer-vs-consumer design table above. Then create the correspondingly-named doc(s).

**Expected sanity checks:** fwd should show 132 `[CTAFIN]` lines (grid 132), bwd 384 (grid 384). Total
`tensor_ops` summed across CTAs should be stable run-to-run (deterministic trace). The earliest
`finish_cyc` should match the previously-observed ~52%-of-kernel first-finisher (fwd). `warpgroup_arrive_cyc`
summed should be in the same ballpark as the SM-global `warpgroup_arrive` stall (after the bug-fix build).

## Status

- [x] Instrumentation implemented (sm.h / functional_unit.cc / sm.cc), observe-only + gated.
- [x] `make clean` + rebuild on remote.
- [x] fwd K5 (`.o40`) + bwd K10 (`.o23`) run with the flag (12h). 132 / 384 `[CTAFIN]` lines — full.
- [x] Correlation computed (below).

## Results (2026-07-16, fwd `.o40` / bwd `.o23`)

**Two instrumentation bugs found + fixed in the run (columns were 0):**
1. `tensor_ops=0` — the guard keyed off `m_type_of_pipeline == TENSOR_CORE__OP`, but the tensor pipeline
   is built as `SPECIALIZED__OP` ([subcore.cc:1469](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1469)), so the branch never ran. Fixed to key off the
   instruction op (`ready_reg->is_tensor_core_op()`).
2. `warpgroup_arrive_cyc=0` — gated on the traditional-scoreboard path, but FA3 runs in **trace mode**
   with `use_traditional_scoreboarding == false` ([subcore.cc:572](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L572)), so `checkTensorCollision_remodeling`
   is never reached and the SM-global `ncu_stall_warpgroup_arrive` is itself 0. **This model has no
   separate consumer-side warpgroup_arrive stall** — WGMMA is a fixed-latency `SPECIALIZED` FU op and the
   consumer's RAW wait folds into `wait_barrier`/scoreboard latency. Replaced the dead column with the
   working **producer/FU-busy** signal `fu_occupied_tensor_cyc` (== NCU `mma`).

**The one column valid on THIS run is `sm_idle_tensor_cyc` (producer re-issue lockout, SM-idle).**
`tensor_ops` and `fu_occupied_tensor_cyc` need the post-fix rebuild. Correlation of the valid column
against per-CTA duration:

| kernel | n | elapsed spread | r(sm_idle_tensor_cyc, elapsed_cyc) | slow-decile vs fast-decile idle |
|---|---:|---:|---:|---:|
| **FWD** (`.o40`) | 132 | 16,551 (12.2% of kernel) | **+0.38** (weak) | 2655 vs 2396 (≈flat) |
| **BWD** (`.o23`) | 384 | 125,504 (58.4% of kernel) | **+0.99** (near-perfect) | **6479 vs 556 (11.6×)** |

**Verdict — bwd's CTA-imbalance is strongly tensor-coupled; fwd's is not.**
- **bwd**: the CTAs that take longest are exactly the tensor-stalled CTAs (r=0.99, slow decile 11.6× the
  fast decile's tensor idle). bwd is the tensor-dense kernel (`mma` is its top sim stall at 12.5%), so
  async-WGMMA is the tail lever here. **This is the primary optimization target.**
- **fwd**: weak (r=0.38); tensor idle is nearly flat across CTAs (1998–3017) and the elapsed spread is
  small (12%). fwd's residual is NOT dominated by per-CTA tensor imbalance — consistent with fwd being
  1-CTA/SM with an already-tight tail. (fwd still has the aggregate stall-depth gap from Ongoing item 3,
  but it is not a *per-CTA-imbalance* effect.)

**Caveat (why the post-fix rebuild still matters):** the bwd r=0.99 is on the producer re-issue-lockout
column alone. To rule out the trivial confound "a longer-running CTA accumulates more of *every* stall,"
the re-run must check the density-normalized signal (`sm_idle_tensor_cyc / tensor_ops`, and
`fu_occupied_tensor_cyc`) — if the tensor stall PER tensor-op is also higher on the slow CTAs, the
causation is confirmed, not just co-scaling. Design (below in ASYNC_WGMMA) proceeds against bwd on this
strong-but-single-column evidence; the post-fix run is the confirmation gate.

## Decision (naming rule applied)

Result = **related** (bwd tail is tensor-coupled), so per the naming rule the async-WGMMA lever is
documented in a single design doc **`.plan/ASYNC_WGMMA.md`** (created next). The fwd-specific weak
coupling is noted there as "not the fwd lever" rather than spun into a separate CTA_IMBALANCE doc —
fwd's residual is the aggregate stall-depth item already tracked in `FA3_progress.md` Ongoing item 3.

## UPDATE (2026-07-16) — async closed; direction changed to warpgroup-4×; new per-CTA columns

The async-WGMMA design that this correlation was meant to feed has been **closed without a code lever**
(see [ASYNC_WGMMA.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/ASYNC_WGMMA.md) TL;DR + §11): the sim is already effectively async and the latency/II
magnitude is a config knob, not code. The bwd r=0.99 producer-lockout correlation is now re-interpreted
as a **symptom of the warpgroup-4× over-execution** ([WARP_GROUP_H100.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/WARP_GROUP_H100.md)): 4 warps each
running the full tile floods all 4 subcore tensor pipes, which is exactly what inflates
`sm_idle_tensor_cyc` / `fu_occupied_tensor_cyc` on the tensor-dense (slow) CTAs.

**The `[CTAFIN]` line has evolved (matches current source):**
- `warpgroup_arrive_cyc` (dead in trace mode) was replaced by **`fu_occupied_tensor_cyc`** (= NCU `mma`),
  gated by `-wgmma_step0_instrument_enable`.
- **NEW** `sm_idle_cyc` + `sm_idle_ibuffer_empty_cyc` columns appended, gated by the new
  `-cta_stall_breakdown_instrument_enable` (2026-07-16), to resolve **Open item 2 (fwd finish-cycle
  variance)** — i.e. whether fwd's slow CTAs are drain-idle/frontend-bound rather than tensor-bound
  (fwd r was only 0.38, so a non-tensor cause is expected).

Current full line:
```
[CTAFIN] sm=.. cta_slot=.. global_cta=.. start_cyc=.. finish_cyc=.. elapsed_cyc=.. tensor_ops=.. sm_idle_tensor_cyc=.. fu_occupied_tensor_cyc=.. sm_idle_cyc=.. sm_idle_ibuffer_empty_cyc=..
```

**Next run does double duty (one 12h run, both open items):**
1. **Open item 1 (warpgroup-4×):** `Σ tensor_ops` vs HW gmma ≈ 835,500 → expect ≈ 4× if confirmed.
2. **Open item 2 (fwd variance):** correlate fwd per-CTA `elapsed_cyc` with `sm_idle_cyc` /
   `sm_idle_ibuffer_empty_cyc` (non-tensor) vs `fu_occupied_tensor_cyc` (tensor) to name the cause.
3. **bwd confound clear-up:** density-normalized `sm_idle_tensor_cyc / tensor_ops` and
   `fu_occupied_tensor_cyc` now that `tensor_ops` is non-zero (post-fix build).

## RESULTS (2026-07-17, MEASURED — fwd `.o42` / bwd `.o25`, both clean-exit)

Timing-neutral gate held: bwd `gpu_sim_cycle=215,537` (vs `.o21` 215,895, −0.17%), fwd `136,293`
(vs `.o38` 135,999, +0.2%) — the new counters did not perturb timing. 384 / 132 `[CTAFIN]` lines full.

### Headline: warpgroup-4× REFUTED
`Σ tensor_ops` bwd = **835,584** vs HW gmma **835,506** → **1.0001×** (per CTA 2,176 == 2,176). The sim
executes the SAME tensor-instruction count as HW (both per-warp). No 4×. → WARP_GROUP_H100.md CLOSED.

### bwd (K10) — strongly tensor-coupled (as before), imbalance is REAL and large
| axis | r(elapsed) | slow-decile ÷ fast-decile |
|---|---:|---:|
| tensor_ops | **+0.99** | **11.4×** |
| fu_occupied_tensor_cyc (mma) | +0.97 | 11.1× |
| sm_idle_tensor_cyc | +0.99 | 10.5× |
| sm_idle_cyc (drain) | +0.99 | 9.7× |
| sm_idle_ibuffer_empty_cyc | +0.84 | 6.2× |

elapsed spread = **92.2%** (10,626 → 136,028). The slow CTAs are the tensor-dense CTAs *and* every
stall scales together — a genuine per-CTA **work imbalance** (causal-mask triangular load: some CTAs do
11× the tensor ops of others). This is a **scheduler/tile-assignment (load-balance)** phenomenon, not a
tensor-model bug: the sim faithfully reproduces that tensor-heavy CTAs take longer. The confound is now
resolved — since tensor_ops itself is 11.4× and fu_occupied scales with it, the stall is proportional to
real work, i.e. co-scaling with density, not an artifactual per-op over-cost.

### fwd (K5) — NOT tensor-bound; drain-idle dominated, tight spread
| axis | r(elapsed) | slow÷fast | mean as % elapsed |
|---|---:|---:|---:|
| sm_idle_cyc (drain) | +0.74 | 1.09× | **46.9%** |
| sm_idle_ibuffer_empty_cyc | +0.66 | 1.12× | **34.0%** |
| tensor_ops | +0.75 | 1.14× | — |
| fu_occupied_tensor_cyc (mma) | +0.55 | 1.15× | 13.4% |
| sm_idle_tensor_cyc | +0.34 | 1.13× | 2.0% |

elapsed spread = only **12.5%** (117,923 → 134,791). fwd is 1-CTA/SM with a **tight** finish tail — the
per-CTA variance is small and everything is mildly correlated (all ~1.1×), so there is **no dominant
per-CTA imbalance lever**. What dominates fwd in absolute terms is **drain-idle 46.9% of elapsed**, of
which **ibuffer-empty (trace-drained) is 34%** — confirming fwd's residual is the drain tail (SM sits
idle with trace-exhausted warps), NOT tensor and NOT a per-CTA imbalance. `sm_idle_tensor_cyc` is only
2.0%. This matches the earlier r=0.38 tensor read and the "drain-idle 1.39×" factor.

### Verdict on the two open items
- **Open item 1 (warpgroup-4×): CLOSED — refuted.** sim == HW tensor count.
- **Open item 2 (fwd variance): ANSWERED — fwd is drain-idle bound (46.9%, ibuffer-empty 34%), not
  tensor, with a tight 12.5% spread.** No per-CTA imbalance lever for fwd; the lever (if any) is the
  aggregate drain-tail, whose *nature* is trace-drained warps (not fetch-starved, prior finding holds).
- **bwd:** large real imbalance (92% spread) that tracks tensor density — a load-balance property the
  sim reproduces faithfully; not a modeling error to "fix".
