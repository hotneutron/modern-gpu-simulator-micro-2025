# Opt 6 — L1I Frontend `stream_buffer_wait` (prefetch send-bandwidth) — Step-0 measure first

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

— (Step-0 frontend-idle measurement pending; gate decision after the run) —
