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
  - Pending rerun verification.

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
- **Likely Remaining Areas**
  - TMA completion to mbarrier identity/binding
  - byte-count / phase / wait semantics
  - `BAR.SYNC.DEFER_BLOCKING` correctness
- **Status**
  - Open.

## Relevant Runs
- Original failing stderr:
  - `sim_run_12.8/.../flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/...-03ff5a5bcdfe.e291`
- Instrumented run exposing `STSM` crash:
  - `sim_run_12.8/.../flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/...-d9a5d39d651b.e292`
- Tensor-core overflow run:
  - `sim_run_12.8/.../flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/...-d545de0f6b9a.e290`

## Relevant Config
- Base H100 config:
  - `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config`
- Added for `STSM`:
  - `-memory_shared_memory_extra_latency_stsm_multiple_matrix 2`
- Updated for Hopper WGMMA tensor core latency:
  - `-tensor_rate_per_cycle 32768` (was `2048`)
