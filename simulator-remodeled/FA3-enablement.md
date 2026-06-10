# FA3 Enablement

## Scope
- Goal: unblock `flashattn-fa3` execution on Hopper/H100 far enough to expose the real `SYNC + TMA` correctness issues.
- Current status: the main FA3 enablement blockers are documented below; the original `SYNC + TMA` problem is still open.

## Addressed Issues

### Tensor Core Latency Overflow For Hopper `WGMMA`
- **Symptom / Error**
  - Run `...-d545de0f6b9a.e290` aborted with:
    - `subcore.cc:315: Assertion 'target_latency_execution < 512' failed`
    - `[BITSETDBG][Subcore] overflow ... latency=512 initiation=512 target=1027`
  - The crash occurs at `Subcore::allocate` when computing the latency reservation slot for a `TENSOR_CORE_OP` at `pc=0x16b0`.
- **Root Cause**
  - `generate_tensor_core_latencies()` computes:
    - `number_of_cycles = m × n × k × operand_bit_size / tensor_rate_per_cycle`
  - FA3 backward uses a `WGMMA.64×128×16 BF16` shape → `131072 × 16 / 2048 = 1024` cycles → `latency=512, initiation=512`.
  - `target = lat_read(3) + 512 + 512 = 1027` overflows the 512-slot `std::bitset<512>` in `functional_unit`.
  - `tensor_rate_per_cycle=2048` was calibrated for Ampere HMMA (16×8×8) shapes and was never updated for Hopper.
  - H100 tensor throughput per SM is ~16× higher than RTX3080, so the correct value is `2048 × 16 = 32768`.
- **Fix**
  - Changed `-tensor_rate_per_cycle 2048` → `-tensor_rate_per_cycle 32768` in the H100 canonical config.
  - With 32768: `WGMMA.64×128×16 BF16` → 64 cycles → `latency=32, initiation=32, target=67` — within the 512 limit.
- **Status**
  - Implemented.
  - Pending rerun verification.

### `STSM` Shared-Memory Store Support
- **Symptom / Error**
  - Run `...-d9a5d39d651b.e292` showed:
    - `[DBG][unsupported-space] ufid=8 warp=7 pc=0x3c20 op=11 memory_op=0 space=0 opcode=STSM.16.MT88.4`
  - Immediate abort happened in `warp_inst_t::generate_mem_accesses()` because `space=undefined_space`.
- **Root Cause**
  - `STSM` was mapped as `STORE_OP`, but trace decode never assigned:
    - `memory_op = memory_store`
    - `space = shared_space`
  - As a result, `STSM` entered the normal memory-access generation path as a store-like instruction with `undefined_space`.
- **Fix Applied**
  - Added explicit `OP_STSM` handling in `trace-driven/trace_driven.cc`.
  - `STSM` now sets `memory_op = memory_store` and `space = shared_space`.
  - Extended opcode-aware matrix width parsing in `trace-parser/trace_parser.cc` so `STSM` follows the same multiplicity-aware path as `LDSM`.
  - `LDSM` already had special handling for multi-matrix forms like `.2` / `.4`, but `STSM` did not.
  - Added `memory_shared_memory_extra_latency_stsm_multiple_matrix`.
  - Registered `-memory_shared_memory_extra_latency_stsm_multiple_matrix`.
  - Added `STSM` opcode-aware latency handling in `abstract_hardware_model.cc`.
  - Reordered the logic to check opcode class (`LDSM` / `STSM`) first, then the `.2/.4` suffix.
  - `LDSM` and `STSM` originally shared one admission queue and one backend shared-memory pipeline.
  - That was too coarse for FA3 enablement because load-side and store-side matrix shared-memory traffic were being treated identically at admission time.
  - Kept one backend shared-memory pipeline.
  - Split admission into:
    - shared-memory load queue
    - shared-memory store queue
  - Added round-robin preference in shared-memory dispatch.
- **Status**
  - Implemented.
  - Base H100 config now includes:
    - `-memory_shared_memory_extra_latency_stsm_multiple_matrix 2`
  - Post-fix verification from ongoing run `...-3a4ac87c71bd.{o293,e293}` shows the simulator progressed well past the old `STSM` crash point and reached later FA3/TMA regions (`0xadc0`, `0xaf30`, `0xb270`, `0x90a0`, `0x97d0`) without reproducing the previous `unsupported-space` assert.
  - This is strong evidence that the old `STSM` blocker has been cleared, although the run is still ongoing.

## Open Issues

### `BAR.SYNC.DEFER_BLOCKING` Modeling
- **Symptom**
  - Trace shows real operands for `BAR.SYNC.DEFER_BLOCKING`, but current decode routes it through generic `OP_BAR`.
- **Root Cause**
  - Current handling hardcodes `bar_id = 0`, `bar_count = -1`, and `bar_type = SYNC`.
  - This discards the actual barrier operands in the trace.
- **Findings From Old Branch**
  - Historical commit `aad6763487cfcfbfc5a6a182d262c47c0179f71e` on branch `tma-impl-sync-temp` did not implement operand-aware `BAR.SYNC.DEFER_BLOCKING` semantics.
  - What that branch primarily fixed was separation of Hopper `SYNCS` from the legacy CTA barrier path:
    - `SYNCS` mapped to `MBARRIER_OP`
    - extra Hopper sync metadata/debug plumbing
    - assertions/logs when Hopper sync leaks into legacy barrier handling
  - The current branch already contains the main architectural separation (`SYNCS -> MBARRIER_OP` and skip of legacy CTA barrier handling for `MBARRIER_OP`), so the remaining `BAR` problem still needs a fresh implementation here.
- **Fix Direction**
  - Decode and preserve operand-derived barrier identity/count.
  - Distinguish `BAR.SYNC`, `BAR.ARV`, and `BAR.SYNC.DEFER_BLOCKING` semantics.
  - Add stronger assertion/logging to catch unexpected leakage of Hopper sync / mbarrier instructions into legacy CTA barrier handling.
- **Planned Implementation**
  - In `trace-driven/trace_driven.cc`, parse static `OP_BAR` operands from traced instruction metadata instead of hardcoding `bar_id=0` and `bar_count=-1`.
  - Map `BAR.ARV` to legacy barrier type `ARRIVE`.
  - Map `BAR.RED*` to `RED` if encountered.
  - Keep `BAR.SYNC` and `BAR.SYNC.DEFER_BLOCKING` on legacy barrier type `SYNC`, but with real operand-derived `bar_id` / `bar_count`.
  - In legacy CTA barrier handling, add assertion/logging so Hopper `SYNCS` / `MBARRIER_OP` cannot silently leak into the CTA `BAR` path.
  - Rebuild and run a fresh post-BAR experiment using the existing `SYNCDBG` logs to verify:
    - `BAR.SYNC.DEFER_BLOCKING` no longer uses fake operands
    - `BAR.ARV` is decoded as arrival-only
    - the remaining failure, if any, is the deeper `SYNC + TMA` issue
- **Status**
  - Not fixed yet.
  - Known FA3 correctness gap, but not the direct cause of the `STSM` crash.
  - Implementation intentionally deferred until the current simulator run finishes and the current FA3-enablement fixes are committed.

### Original `SYNC + TMA` Problem
- **Symptom**
  - Arrival-only microbench passes, but FA3-style `SYNC + TMA` does not.
- **Current Understanding**
  - The direct `STSM` crash was a secondary enablement blocker, not the underlying `SYNC + TMA` failure.
  - TMA ops in the examined failing run stayed on the TMA path; they were not the direct source of the `generate_mem_accesses()` abort.
  - Finished post-`STSM` run `...-17cf425686f3.{o295,e295}` ran the FA3-bwd kernel (`kernel-10`) to completion: all 384 CTAs were launched (`thread block = 0,0,0` → `383,0,0`) and the simulator reported `gpu_tot_issued_cta = 384` / `gpu_completed_cta = 384`, ending with `GPGPU-Sim: *** exit detected ***`, with no simulator-declared deadlock or assert.
  - The final `SYNCDBG` summaries show:
    - many `wait_released`
    - some `EXCH`
    - nonzero `tma_completions`
    - but zero `arrive`, zero `arrive_expect_tx`, and zero `phase_flip`
  - This strengthens the view that the remaining issue is in deeper sync / barrier / TMA binding semantics rather than the old `STSM` path.
- **Root Cause Of Zero `arrive` (now identified)**
  - The zero-`arrive` counters were caused by the tracer not recognizing the FA3 `SYNCS.ARRIVE.TRANS64` variants:
    - the static resolver (`build_sync_operand_mapping.py`) and the runtime capture (`tracer_tool.cu`) matched only nonexistent opcode spellings (e.g. `.RED.A0TR`), so every FA3 arrive site got `sync_site_valid=0` / `kind=NONE` / `sync_runtime_valid=0` and was dropped at the `handle_sync_instruction` early-return gate.
  - After fixing both opcode matchers, a fresh FA3 trace now emits `sync_operand_resolver.json` with 187 sync sites (EXCH 32, ARRIVE 30, PHASECHK 39, TRYWAIT 86) and the protobuf `instruction.sync` payload is populated.
- **Validated Arrive / EXCH Semantics (from new FA3 trace + microbench ground truth)**
  - Verified by decoding the new per-CTA `.pb` joined with the resolver across many `kernel_10` CTAs. The FA3-bwd barriers split cleanly into exactly two roles:

    | barrier group | EXCH `expected = 0x200000 - raw` | arrive variant | how it closes |
    |---|---|---|---|
    | `0x31000/10/18/30/38` | 2 | `SYNCS.ARRIVE.TRANS64` with **register** semantic operand (tx bytes `0x4200`/`0x8000`) | by accumulated **tx bytes** |
    | `0x31020/28/40/48` | 512 | `SYNCS.ARRIVE.TRANS64.RED.A1T0` with **`RZ`** semantic operand (no tx) | by accumulated **arrive count** |

  - Key confirmed facts:
    - `.RED.A1T0` (semantic operand = `RZ`) is a **plain / counted arrive, NOT expect-tx**. It only targets the count-closed barriers and never carries tx bytes. This confirms the standing hypothesis that `.RED.A1T0` is effectively a "tx value = 0" arrive rather than an expect-tx arrive.
    - The expect-tx arrive is the variant whose **semantic operand is a register** (plain `SYNCS.ARRIVE.TRANS64` here; `.RED.A0TR` in the nvcc microbenches). The `.RED`/`AxTy` suffix spelling differs between the CUTLASS build (FA3) and the nvcc build (microbench), so the reliable discriminator is the **semantic operand**, not the suffix string: `RZ` → plain arrive, register → expect-tx.
    - EXCH decode is `expected_arrive_count = (0x200000 - raw) / 2`. Microbench ground truth (`init_arrivals` knob): `3 → 6`, `4 → 8`, `8 → 16`, i.e. `0x200000 - raw = 2 × init_arrivals`. FA3 count barriers give `0x200000 - raw = 512 → logical expected = 256`.
  - Implication for the simulator: treat arrive sites by semantic value, `semantic_raw == 0` (operand `RZ`) → plain arrive, `semantic_raw != 0` → expect-tx (also accumulates `expected_tx_bytes += semantic_raw`); **both paths increment `arrive_count` by the active thread count** (mbarrier counts arrivals in thread units); decode EXCH with the `/2` factor.
- **Implemented: runtime-`semantic_raw` arrive classification + unverified-variant guard**
  - `remodeling/sm.cc` `handle_sync_instruction` now branches the `ARRIVE_EXPECT_TX` case on the captured `semantic_raw`: `== 0` → plain arrive (`m_sync_debug_arrive`), `!= 0` → expect-tx (`m_sync_debug_arrive_expect_tx`, tx bytes accumulated + TMA completion binding). Both paths now do `arrive_count += active_threads` (= `inst.active_count()`).
  - Both the resolver build (`build_sync_operand_mapping.py`, `VALIDATED_ARRIVE_OPCODES`) and the simulator (`is_validated_arrive_opcode`) now reject any `SYNCS.ARRIVE.TRANS64*` variant that executes but has not been runtime-validated (`SystemExit` / `abort()` with diagnostics). See `.plan/SYNC_ISA.md`.
  - No re-tracing required: trace format/content is unchanged. Validation done: FA3 resolver still yields 187 sites (arrive = 2 validated variants), all 5 nvcc microbenches pass, an injected `.RED.A0T1` is correctly rejected, and the simulator builds clean (`accel-sim.out`).
- **Likely Remaining Areas**
  - TMA completion to mbarrier identity/binding
  - byte-count semantics
  - `BAR.SYNC.DEFER_BLOCKING` correctness
- **Resolved**
  - **Phase/wait parity semantics fixed (was an FA3-bwd deadlock).** Diagnosed from run
    `...-53cb9e043fde.{o296,e296}`: after barriers flipped to `phase=1`, consumer
    `SYNCS.PHASECHK`/`TRYWAIT` waits missed forever (~52k `wait miss`, all on
    `phase=1`; `phase=0` had zero misses), so tx barrier `0x31000` deadlocked once its
    TMA bytes completed and the phase flipped. Root cause: `decode_sync_wait_phase` in
    `remodeling/sm.cc` read the parity from **bit 0** (`raw & 0x1`) and compared with
    `==`, but the FA3 wait-state raw only ever takes `0x0` / `0x80000000` (differing
    solely in **bit 31**), so the bit-0 decode discarded the parity entirely and hit/miss
    depended only on `barrier.phase`. Fix (2 lines): decode parity from bit 31
    (`(raw >> 31) & 1`) and flip the completion comparison to `!=` (Hopper
    `try_wait.parity` completes when input parity differs from the barrier's current
    phase parity — NVIDIA CUDA Programming Guide §4.9). Validated by the trace truth
    table and consistent with `.plan/SYNC_ISA.md` (which already documented bit 31 as
    the parity bit). Rebuild + rerun to confirm `wait miss` drops and phases cycle
    both directions is the next step.
  - The "arrive += 1" question is settled: arrive increments `arrive_count` by the **active thread count** (`inst.active_count()`), not a fixed 1. Confirmed by FA3 active_mask (count barrier `0xffffffff`→+32, tx barrier `0x1`→+1) matching the logical `expected_arrive_count` (count 256, tx 1). Two bugs fixed together: (1) EXCH decode was missing the `/2` factor (`decode_sync_exch_expected_arrive_count`); (2) arrive was hard-coded to `+= 1`. The expect-tx path does **not** need an extra arrive on TMA completion — the expect-tx instruction's own arrive already satisfies the logical count of 1.
  - Phase-flip reset bug fixed: phase flip now preserves `expected_arrive_count` and resets only the per-phase accumulators (`arrive_count`, `completed_tx_bytes`, `expected_tx_bytes`). `SYNCS.EXCH` (= `mbarrier.init`) runs exactly once per barrier (validated: 1 EXCH vs many phases), so the hardware reuses the expected arrive count every phase. Previously clearing it on flip would let the next phase become ready after a single arrive (`arrive_count > 0 >= 0`), causing premature flips/races.
- **Status**
  - Arrive/EXCH semantics fixed (thread-unit arrivals + EXCH `/2`) and the simulator builds clean (`accel-sim.out`). Re-running the FA3 sim to confirm barriers actually flip phases (nonzero `phase_flip`, no deadlock) is the next step.

## Relevant Runs
- Original failing stderr:
  - `sim_run_12.8/.../flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/...-03ff5a5bcdfe.e291`
- Instrumented run exposing `STSM` crash:
  - `sim_run_12.8/.../flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/...-d9a5d39d651b.e292`
- Post-`STSM` verification run:
  - `/tmp/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24-flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_-3a4ac87c71bd.e293`
  - `/tmp/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24-flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_-3a4ac87c71bd.o293`
- Finished post-`STSM` progress run:
  - `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/H100_80GB-OnlyKernel10/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24-flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_-17cf425686f3.e295`
  - `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/H100_80GB-OnlyKernel10/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24-flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_-17cf425686f3.o295`
- Tensor-core overflow run:
  - `sim_run_12.8/.../flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/...-d545de0f6b9a.e290`

## Relevant Config
- Base H100 config:
  - `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config`
- Added for `STSM`:
  - `-memory_shared_memory_extra_latency_stsm_multiple_matrix 2`
- Updated for Hopper WGMMA tensor core latency:
  - `-tensor_rate_per_cycle 32768` (was `2048`)
