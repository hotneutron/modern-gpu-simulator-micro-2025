# Opt 6 — L1I Frontend `stream_buffer_wait` (prefetch send-bandwidth) — Step-0 measure first

> **STATUS: DEFERRED — frontend GATE FAILED.** Step-0 SM-idle decomposition (fwd `.o20`, bwd
> `.o3`) shows the L1I frontend (`sm_idle_blocked_by_frontend_sbwait`) is only **fwd 3.99% /
> bwd 5.31%** of evaluated cycles — the per-subcore `stream_buffer_wait` (14.8/15.6%) shrank ~3x at
> SM level, exactly the WGMMA per-subcore mirage. So widening the L0→L1 send bandwidth can recover
> at most ~4–5%; it is not the dominant SM-idle cause. **Parked.** The §4 fix design is kept for
> reference. See §3 for the full SM-idle decomposition and the next target.

> Target: FA3 fwd (k5) / bwd (k10), on top of Opt 5. Supersedes the earlier
> `L1I_PREFETCH_LOOKAHEAD_H100.md` draft, whose diagnosis ("lookahead = 1 line") was **wrong**
> (the stream buffer already prefetches ~4 lines ahead via `do_prefetch`). This plan records the
> corrected root cause and gates the fix on an SM-level measurement, to avoid the per-subcore
> over-estimate trap that sank the WGMMA idea.

## 1. Symptom

`no_valid_instruction` (frontend) is large: fwd 18.7% / bwd 17.7%, almost all
`stream_buffer_wait` (fwd 14.8% / bwd 15.6%). HW frontend stall (NCU `no_instruction`+`imc_miss`)
is ~4.5% / ~2.8% → sim over-states ~4–5x. These are **per-subcore** issue-stage percentages.

## 2. Root cause (code + measured, corrected twice)

- **Not lookahead.** `single_stream_buffer::do_prefetch()` fills sequentially
  ([stream_buffer.cc:392-422](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/stream_buffer.cc#L392-L422))
  up to `prefetch_per_stream_buffer_size = 4`, so lookahead is already ~4 lines.
- **Not promote.** L1I_prefetch_redesign.md fixed the promote↔response-slot coupling; Opt 5
  eager-promote **works** (bwd `.o2`: `eager_promote_to_cache = 1,007,270`,
  `demand_hit_later = 367,005`, `demand_miss_after_promote = 0`).
- **Yet `head_demand_arrived_after_ready = 0`** despite prefetch being issued on average
  **521 cycles** before demand (`prefetch_issue_to_first_demand` 42,450,755 / 81,496). The line is
  still not READY when demand arrives → `prefetch_issued_not_ready = 13,387,337` (99% of
  stream_buffer_wait).
- **Real cause = prefetch SEND bandwidth.** `prefetch_blocked_memport_full = 1,917,581` >
  `prefetch_issued = 1,138,222`. `m_memport` is a **single per-SM `L0_icnt`** request port
  ([sm.cc:1171-1175](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1171-L1175))
  with capacity `max_request_allowed_to_L1I × latency_L0_to_L1 = 1 × 2`, **shared by all 4
  subcores' demand misses + prefetch + const requests**. prefetch
  ([stream_buffer.cc:421](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/stream_buffer.cc#L421)) and
  demand ([gpu-cache.cc:1110-1117](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1110-L1117))
  contend on the same `full()`/`push()`. With only **1 request/cycle** out of L0→L1 (plus
  1/cycle reply, `max_reply_allowed_from_L1I = 1`), prefetches can't be sent fast enough to be
  ready before the demand. Config: `-max_request_allowed_to_L1I 1`, `-max_reply_allowed_from_L1I 1`,
  `-latency_L0_to_L1 2`, `-num_instruction_prefetches_per_cycle 1`
  ([gpgpusim.config:226-229,317](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config#L226)).

## 3. STEP-0 GATE (mandatory before any fix) — is the 15% real at SM level?

The 14.8%/15.6% is **per-subcore**. The WGMMA Step-0 showed a per-subcore 18% bucket collapsed to
1.6% at SM level (another subcore was almost always issuing). The frontend may do the same.

**Instrumentation added (behind `-wgmma_step0_instrument_enable`, observe-only):**
- `total_num_cycles_sm_idle_blocked_by_frontend_sbwait` — cycles where **no subcore on the SM
  issued** AND ≥1 subcore had a warp blocked on the L1I stream-buffer frontend. This is the true
  recoverable ceiling, analogous to `sm_idle_all_blocked_by_tensor`.
- **FULL SM-idle decomposition (one run attributes ALL of `sm_all_subcores_idle`):** each subcore
  exports a per-cycle non-issue reason bitmask (`step0_reason_mask_this_cycle`, enum
  `Subcore::STEP0_R_*`); on every true SM-idle cycle, SM::cycle() ORs the 4 subcores' masks and
  bumps a per-reason counter `total_num_cycles_sm_idle_reason_*` for: `next_stage`,
  `issue_port_busy`, `no_valid_frontend`, `no_valid_sbwait`, `no_valid_other`, `fu_occupied`,
  `fu_occupied_tensor`, `inst_barrier`, `wait_barrier`, `tma_flush`, `stall_count`, `scoreboard`,
  `l1c`, `result_queue_full`, `yield`, and `none` (unattributed). Reasons overlap (sum can exceed
  `sm_all_subcores_idle`); read as relative intensity. **This is the key output**: it tells us not
  only the frontend's true SM-level share but also what the *other* ~16% of SM-idle actually is,
  in a single 12 h run — so the next target is data-driven regardless of the frontend gate.
- Existing rich frontend counters (`prefetch_issued`, `prefetch_blocked_memport_full`,
  `prefetch_issued_not_ready`, `head_demand_arrived_before/after_ready`,
  `demand_wait_for_ready`, `eager_promote_*`) already quantify the prefetch-send saturation; no
  extra baseline-cache (shared-code) counters are added to avoid regression risk on the demand
  path. The send-bandwidth A/B in §4 will confirm via `prefetch_blocked_memport_full`.

(subcore exports the mask + frontend bit; SM aggregates in
[sm.cc:551-598](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L551-L598);
printed via the Step-0 dump in shader.cc.)

**Decision gate:** run once with the flag on; read `sm_idle_blocked_by_frontend_sbwait` /
`sm_all_subcores_idle`.
- If frontend is a **majority** of true SM-idle → proceed to §4 fix.
- If it collapses to a few % (like WGMMA) → frontend is NOT on the critical path; do not fix it,
  re-target the rest of `sm_all_subcores_idle`.

### Step-0 RESULT (fwd `.o20` 149,727 cyc / bwd `.o3` 241,425 cyc) — GATE FAILED

`sm_all_subcores_idle ≈ 18.3% (fwd) / 18.5% (bwd)`. SM-idle decomposed by reason (≥1 subcore
blocked on that reason during a true SM-idle cycle; reasons overlap, read as relative intensity):

| reason | fwd | bwd | note |
|---|---|---|---|
| **no_valid_other** (ibuffer empty / decode pending) | **11.90%** | **10.55%** | **#1 — never analyzed; NOT stream_buffer** |
| **wait_barrier** (mbarrier / DEPBAR, TMA data arrival) | **9.71%** | **11.09%** | **#2 — HW long_scoreboard+barrier analog** |
| no_valid_frontend | 4.23% | 5.49% | |
| &nbsp;&nbsp;— `sm_idle_blocked_by_frontend_sbwait` (this Opt) | **3.99%** | **5.31%** | GATE metric — too small |
| stall_count (fixed-latency dep) | 3.76% | 3.82% | |
| fu_occupied (all pipes) | 2.10% | 2.84% | |
| &nbsp;&nbsp;— tensor | 0.67% | 1.64% | matches WGMMA Step-0 |
| next_stage | 1.78% | 2.12% | 1-deep latch backpressure |
| tma_flush | 0.00% | 4.76% | bwd-only (UTMACMDFLUSH) |
| inst_barrier / l1c / scoreboard / yield / result_queue | ≤0.3% | ≤0.4% | negligible |
| none (unattributed) | 0.00% | 0.00% | full coverage ✓ |

**Verdict:** frontend `sbwait` is only ~4–5% of SM-idle (vs the 14.8/15.6% per-subcore figure) →
GATE fails, fix parked. **The real SM-idle drivers are `no_valid_other` (~11%) and `wait_barrier`
(~10%).** `none = 0` confirms the decomposition accounts for all SM-idle.

> Next target: **`no_valid_other`** — SM-idle where the head warp has no valid instruction but it
> is NOT a stream-buffer wait (ibuffer empty / decode pending). This is a *different* frontend
> stage (fetch→decode→ibuffer) than the prefetch send path, and has never been analyzed.

> **Follow-up instrumentation added (for the NEXT run):** the current `no_valid_other` bit is too
> coarse — it lumps ibuffer-empty / decode-pending / l0i-response-ready / unknown. Note the apparent
> paradox: per-subcore counters say `head_invalid_waiting_frontend = 18.2%` with
> `ibuffer_empty = 0.03%`, yet the SM-idle `no_valid_other` is ~10–11%. This is because on a true
> SM-idle cycle different subcores can be in different no_valid sub-states and the SM-level OR
> catches the non-frontend ones. To resolve it, the reason bitmask now splits `no_valid_other` into
> `nv_ibuffer_empty`, `nv_decode_pending`, `nv_l0i_resp_ready`, `nv_unknown`
> (`Subcore::STEP0_R_NV_*`, emitted as `sm_idle_reason_nv_*`). Run again (Step-0 flags on) to learn
> which one actually dominates the #1 SM-idle bucket before designing a fix.

### Step-0 FOLLOW-UP RESULT (`no_valid_other` split, fwd `.o23` / bwd `.o5`)

The split run answers the open question from above: **`no_valid_other` is almost entirely
`nv_ibuffer_empty`**, not decode / L0I-response latency.

| reason | fwd `.o23` | bwd `.o5` | note |
|---|---|---|---|
| `sm_all_subcores_idle` | 18.6768% | 18.1723% | true SM-idle ceiling |
| `sm_idle_reason_no_valid_other` | 12.2080% | 10.0735% | original coarse bucket |
| `sm_idle_reason_nv_ibuffer_empty` | 12.2099% | 10.0777% | essentially all of `no_valid_other` |
| `sm_idle_reason_nv_decode_pending` | 0.5831% | 0.5597% | small |
| `sm_idle_reason_nv_l0i_resp_ready` | 0.0279% | 0.0237% | negligible |
| `sm_idle_reason_nv_unknown` | 0.0000% | 0.0000% | full attribution |
| `sm_idle_reason_nv_ibuf_fetch_inflight` | 0.0000% | 0.0000% | not a fetch-latency problem |
| `sm_idle_reason_nv_ibuf_fetch_not_issued` | 0.0049% | 0.0043% | effectively zero |
| `sm_idle_reason_wait_barrier` | 9.9623% | 10.9032% | still the other major bucket |

**Interpretation.**

- If `nv_ibuffer_empty` were a real fetch/decode bottleneck, we would expect a meaningful share of
  `fetch_inflight` or `fetch_not_issued`. Instead both are ~0%.
- Therefore the dominant `nv_ibuffer_empty` cycles are **not** "frontend cannot fetch the next
  instruction fast enough". They are overwhelmingly **trace-exhausted / winding-down warps**:
  a warp has no next instruction left to place into the ibuffer, while sibling warps / sibling CTAs
  on the same SM are still alive, so the SM can still show up as idle on that cycle.
- In other words, the apparent #1 SM-idle bucket is largely a **tail-drain / imbalance effect**, not
  an actionable L1I fetch-path defect. The true frontend-fix leverage is therefore even smaller than
  the already-failed `sm_idle_blocked_by_frontend_sbwait` gate suggested.

**Updated verdict.** The Opt-6 frontend idea remains **deferred**. After the split, the remaining
high-value targets are:

- `wait_barrier` (~10-11%): real synchronization / arrival-side wait.
- tail-drain / winding-down imbalance hidden inside `nv_ibuffer_empty` (~10-12%): likely compare
  against HW imbalance/tail metrics before attempting any simulator-side fix.

> **This run must be PURE instrumentation (no fix in the same run).** All Step-0 counters are
> observe-only (timing unchanged), so this run must reproduce the Opt-5 baseline cycle exactly —
> which makes it the clean baseline for the §4 A/B. Do **not** bump any bandwidth knob in this run:
> (1) the GATE is unresolved (frontend may be a per-subcore mirage like WGMMA), so a fix here can't
> be cleanly attributed; (2) mixing a fix destroys the baseline; (3) if the GATE fails the whole
> fix run is wasted. The §4 bandwidth A/B runs separately, after the GATE passes.
> *Optional time-saver:* if you are confident the GATE will pass and have a spare run slot, the §4a
> config-only run may be launched **in parallel as an independent run** (separate output) — that is
> two independent runs, not a fix mixed into the measurement run.

## 4. Fix (only if Step-0 gate passes) — widen the L0→L1 prefetch/request bandwidth

The L0→L1 send path is a multi-stage pipeline in `L0_icnt`, and **every stage is currently 1-deep**.
Relieving only one stage just moves the serialization downstream, so all three must widen together.
Concrete change points (verified):

1. **Request ports / send queue** — `m_max_num_L1_request_ports_allowed` = `-max_request_allowed_to_L1I`
   (config:226, consumed at [l0_icnt.cc:360,370-373](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/l0_icnt.cc#L356-L373)).
   This sizes `m_icnt_to_L1_queue[ports][latency]` and is what `full()`
   ([:390-399](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/l0_icnt.cc#L390-L399)) /
   `push()` test — i.e. the direct source of `prefetch_blocked_memport_full`. Raise 1 → N
   (config-only, no rebuild).
2. **TLB→cache mid-queue** — `m_max_size_icnt_L1_TLB_to_cache = 1` is **hardcoded**
   ([l0_icnt.cc:379](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/l0_icnt.cc#L379)), and the TLB→cache loop is
   also capped at `m_max_num_L1_request_ports_allowed` per cycle
   ([l0_icnt.cc:530](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/l0_icnt.cc#L530)). Make this size a config
   knob (e.g. `-l1i_tlb_to_cache_queue_size`, default 1) and set it to N so the widened requests
   are not re-serialized here. (Requires `l0_icnt.{h,cc}` edit + rebuild.)
3. **Reply ports** — `m_max_num_L1_reply_ports_allowed` = `-max_reply_allowed_from_L1I` (config:227),
   sizes `m_L1_to_icnt_queue` ([:374-377](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/l0_icnt.cc#L374-L377)).
   Raise 1 → N so the ready line can return without 1/cycle throttling (config-only).
- `-num_instruction_prefetches_per_cycle` (config:317): raise only **after** the above; alone it
  just calls `do_prefetch` more and still hits the same `full()`.

**Two-step rollout (each its own A/B run vs the Step-0 baseline):**
- **4a (config-only, no rebuild):** bump `max_request_allowed_to_L1I` and `max_reply_allowed_from_L1I`
  1 → e.g. 4. Confirms the diagnosis fast: `prefetch_blocked_memport_full` and
  `prefetch_issued_not_ready` should drop and `head_demand_arrived_after_ready` should go > 0.
  (The TLB→cache stage stays 1-deep, so 4a may only partially relieve — that itself tells us
  whether stage 2 matters.)
- **4b (code, if 4a is partial):** make `m_max_size_icnt_L1_TLB_to_cache` a config knob and widen
  it with the loop; re-run.
- **HW-justification check before committing N:** real Hopper L0→L1 I-fetch is wider than 1
  req/cycle; pick N from NCU I-fetch behavior / literature, not by maximizing blindly. Each knob
  flag-off (=1) must reproduce the Opt-5/Step-0 baseline bit-for-bit.

*Files:* `gpgpusim.config` (knobs), `l0_icnt.{h,cc}` (only for 4b: the hardcoded mid-queue size +
its loop), `gpu-sim.cc` (register the new mid-queue knob).

## 5. Verification

- `prefetch_blocked_memport_full` drops; `prefetch_issued_not_ready` drops;
  `head_demand_arrived_after_ready` > 0; `no_valid_instruction` → toward HW ~3–5%.
- **Gate metric**: `sm_idle_blocked_by_frontend_sbwait` was a majority of `sm_all_subcores_idle`
  (else the fix can't move total cycles regardless).
- Flag-off (and knobs at 1) reproduces Opt 5 cycles bit-for-bit; `gpu_sim_insn` unchanged.

## 6. Result

- **Frontend GATE failed.** The true SM-level frontend share is only `sm_idle_blocked_by_frontend_sbwait
  = 3.99%` (fwd `.o20`) / `5.31%` (bwd `.o3`), far below the per-subcore 14.8/15.6% figure.
- The follow-up split shows the apparent #1 bucket `no_valid_other` is almost entirely
  `nv_ibuffer_empty` (`12.21%` fwd / `10.08%` bwd), while its fetch sub-causes are ~0
  (`fetch_inflight = 0`, `fetch_not_issued ~ 0.005%`).
- So the dominant residual is **not** an L1I send / fetch / decode throughput problem. It is best
  interpreted as **tail-drain / winding-down warp imbalance** during trace execution.
- Practical conclusion: keep the send-bandwidth redesign parked for now; prioritize
  `wait_barrier`-side analysis and compare the sim tail-drain interpretation against HW imbalance
  metrics before reopening frontend work.
