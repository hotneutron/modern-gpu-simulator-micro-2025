# Opt candidate — ISSUE_CONTROL latch depth (non-TMA)

> **Status: investigation only (static analysis done, NOT yet gated by measurement).**
> This is a **non-TMA** cycle-reduction candidate found while the M2/M2.5 TMA address runs
> were in flight. It is recorded here so it is not re-investigated blindly. **Do not implement
> before the SM-level gate below passes** — it may be the same per-subcore over-count mirage that
> parked the WGMMA `fu_occupied` opt (see FA3_progress Deferred Opts).

Target: FA3 fwd (k5) / bwd (k10), on top of the TMA M2/M2.5 baseline.

## 1. Symptom — `next_stage_not_available` is a large issue-stage stall

The issue-stage stall class `next_stage_not_available` is ~17–22% (fwd) / ~15–19% (bwd) of
issue-stage stalls in FA3 (grows in later opts). It means the issue stage could not push an
instruction because the **next pipeline latch was still occupied** this cycle.

- Counter increment: [subcore.cc:718](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L718)
  (`total_num_cycles_issue_stage_stall_next_stage_not_available`), set when `!is_issued_inst &&
  !is_next_stage_availabe` ([subcore.cc:710-718](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L710-L718)).
- The exact downstream buffer is the **ISSUE→CONTROL pipeline latch** `m_ISSUE_CONTROL_latch`;
  `is_next_stage_availabe` is cleared iff `m_ISSUE_CONTROL_latch.has_free() == false`
  ([subcore.cc:436](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L436)).
  Definition/comment: [subcore.h:96](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h#L96)
  ("ISSUE_CONTROL latch full").
- Note: when `m_num_pending_cycles_with_issue_port_busy > 0` (e.g. IMAD.WIDE port occupancy),
  the cycle is classified as `issue_port_busy`, not this class
  ([subcore.cc:434,719-720](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L434)).

## 2. Root cause (static): depth-1 latch + tensor II lockout

The issue→execute path is a chain of **depth-1** latches, hardcoded in source (NOT config):
- `m_ISSUE_CONTROL_latch  = register_set_uniptr(1, ...)` — [subcore.h:161](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h#L161)
- `m_CONTROL_ALLOCATE_latch = register_set_uniptr(1, ...)` — [subcore.h:162](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h#L162)

Stage order within one cycle is drain-first (`writeback → execute → read_rf → allocate →
control_stage → issue → decode → fetch`, [subcore.cc:103-115](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L103-L115)),
so `control_stage` tries to drain `m_ISSUE_CONTROL_latch` before `issue` refills it.

`control_stage` ([subcore.cc:348-384](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L348-L384))
only drains the latch when:
- fixed-latency inst: `m_CONTROL_ALLOCATE_latch.has_free()` — else it calls
  `fu->add_extra_cycle_initiation_interval()` ([subcore.cc:376](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L376)) and the latch stays full.
- non-fixed (queue-type MEM/TMA/etc): `fu->can_issue()`.

FA3 is WGMMA/tensor-heavy. A tensor issue calls `fu->reserve_unit()` →
`m_dispatch_pending_reserved_cycles = initiation_interval` (~32) ([functional_unit.cc:125-131](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L125-L131)),
and `can_issue()` requires `m_dispatch_pending_reserved_cycles == 0 && m_dispatch_reg->empty()`
([functional_unit.cc:137-141](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L137-L141)).
So the tensor pipe is locked for ~II cycles; during that window the depth-1 ISSUE_CONTROL latch
that holds a tensor inst cannot drain → every subsequent issue cycle records
`next_stage_not_available`. RF-read conflicts / CONTROL_ALLOCATE-full extend the lockout via
`add_extra_cycle_initiation_interval` ([subcore.cc:329,338,376](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L329)),
matching the observed growth in later opts.

## 3. The gate (do this FIRST — is it real SM-level loss or a per-subcore mirage?)

`next_stage_not_available` is a **per-subcore** counter. The WGMMA `fu_occupied` opt was parked
because its per-subcore share (13–18%) over-counted the true SM-level loss ~7x (another subcore is
almost always issuing); the real `sm_idle_all_blocked_by_tensor` was only 0.65% (fwd) / 1.59%
(bwd). This candidate is at risk of the **same mirage**.

**Gate metric (read from the M2/M2.5 run stdout, no new run needed):**
- `total_num_cycles_sm_idle_all_blocked_by_tensor` / total cycles
  ([sm.cc:566](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L566)) — the true SM-level loss where **all** subcores are blocked by the tensor pipe.
- Related instrumentation already present: `is_any_tensor_reissue_lockout_only`
  ([subcore.cc:662-664](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L662-L664)),
  `total_num_tensor_add_extra_cycle_initiation_interval` ([subcore.cc:342,379](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L342)).

Decision rule:
- If `sm_idle_all_blocked_by_tensor` is small (< ~3%): the depth-1 latch is **not** the SM-level
  bottleneck → **park** this candidate (same as WGMMA fu_occupied).
- If it is meaningful (>= ~3–5%): the latch-depth fix below is worth implementing.

## 4. Implementation risk (if the gate passes) — infra supports depth>1, but 2 hardcodes break it

`register_set_uniptr` already supports multi-slot correctly:
- `has_free()`/`has_ready()` scan all slots ([abstract_hardware_model.h:2067-2085](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L2067-L2085)).
- `get_ready()` returns the **smallest-uid (oldest, program order)** entry
  ([abstract_hardware_model.h:2092-2102](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L2092-L2102)).
- `move_in()` fills the free slot via `get_index_of_free()` ([:2129-2137](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L2129-L2137)).

Producer side (issue_warp → shared_sm->issue_warp → move_in), `fu->issue()`
([functional_unit.cc:150-151](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L150-L151), uses `get_ready()`),
and `reserve_unit()` (uses `get_ready()`) are all get_ready/get_free based → **safe at depth>1**.

**Two hardcoded spots must be fixed before increasing the depth:**

1. **[subcore.cc:367](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L367)** —
   `move_warp_between_reg_sets(m_CONTROL_ALLOCATE_latch, 0, m_ISSUE_CONTROL_latch, 0)` hardcodes
   `src_idx=0` ([move_warp_between_reg_sets](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L2209-L2215) asserts src slot non-empty and moves slot 0).
   But `control_stage` picked the inst via `get_ready()` (oldest, could be slot 1). At depth 2, if
   the oldest is in slot 1 (and slot 0 is empty or holds a different inst), this moves the wrong /
   an empty slot → mismatch/assert. Fix: move the **exact slot `get_ready()` chose** (use an
   index-returning ready API, or `move_out_to`), not a hardcoded 0.

2. **[subcore.cc:1024-1025](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1024-L1025)** —
   right after `move_in`, `reserve_unit(dispatch_latch)` calls `get_ready()` (oldest). At depth 1
   the just-inserted inst is the only one, so this reserves *its* `initiation_interval`. At depth 2
   it may reserve the **pre-existing oldest inst's** II instead → wrong tensor-II accounting. Fix:
   reserve against the just-inserted inst specifically.

No config knob exists for the latch depth — the `register_set_uniptr(1, ...)` constant is hardcoded
([subcore.h:161-162](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h#L161-L162)).
A clean implementation would add a config knob (default 1 = bit-identical) and fix the two spots
above so depth>1 is honored.

## 5. Related knobs (context, not the primary lever)

From [SM90_H100_L2_50MB_80GB/gpgpusim.config](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config):
- `-tensor_latency 32` (:255), `-tensor_rate_per_cycle 32768` (:256) — tensor II is computed at
  runtime from GMMA shape ([abstract_hardware_model.cc:429-440](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L429-L440)); do NOT touch (that is the tensor-pipe model, a separate axis).
- `-max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions 1` (:401) —
  writeback drain = 1/cycle; a secondary backpressure source, only if the gate points there.
- `-gpgpu_pipeline_widths` / operand-collector knobs (:73, :98-101) are **baseline GPGPU-Sim only**
  and NOT used by the remodeled subcore issue path (width is hardwired to dispatch_reg=1 per pipe).

## 6. Verification plan (when the gate passes)

1. Add `-issue_control_latch_depth` (default 1). Depth 1 must be bit-identical to today.
2. Fix subcore.cc:367 (slot-accurate drain) and subcore.cc:1024-1025 (reserve just-inserted inst).
3. A/B: depth 1 vs 2 on FA3 fwd K5 + bwd K10 (on the M2/M2.5 baseline).
   - Success: `next_stage_not_available` drops AND `gpu_sim_cycle` drops.
   - Mirage confirmation: `next_stage_not_available` drops but cycle ~flat → the depth-1 latch was
     a per-subcore artifact, not the SM-level limiter → park (record the delta).
4. Sanity: clean exit, no assert, `sm_idle_all_blocked_by_tensor` unchanged or lower.
