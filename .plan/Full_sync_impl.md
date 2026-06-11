# 1. Overview
The simulator's synchronization model for Hopper architecture has been completely revamped. The previous heuristic-based Python scripts and JSON parsing/inference logic have been entirely removed. The new implementation strictly follows deterministic hardware rules to manage the `mbarrier` protocol, enabling accurate and deadlock-free simulation of asynchronous TMA (Tensor Memory Accelerator) operations.

# 2. Deterministic Hardware Rules for MBarrier

## 2.1 TMA Instruction Register Binding
Unlike previous generations where synchronization points had to be guessed via PC-based heuristics, Hopper TMA instructions use strict hardware rules to bind memory transactions to barrier registers.

1. **UTMALDG / UTMAPF (Shared memory loads):**
   - The barrier register is **implicitly** defined as the next register after the destination register (`dst_reg + 1`).
   - The destination register (`dst_reg`) is always the **1st operand** in the SASS instruction.
   - *Example:* If the instruction is `UTMALDG.64 UR16, ...`, the 1st operand is `UR16`, so the hardware implicitly uses `UR17` as the mbarrier register.
   - *Tracer Update:* The NVBit tracer captures this `N+1` register via the `value_hi` slot.
2. **UBLKCP:**
   - The barrier register is **explicitly** defined as the 3rd operand in the SASS instruction.
3. **UTMASTG / UBLKRED (Global memory stores/reductions):**
   - No barrier is required as data flows outwards to global memory.

```text
 +-----------------+
 | TMA Instruction |
 +--------+--------+
          |
          v
 +-----------------+
 |   Opcode Type   |
 +-----------------+
   |      |      |
   |      |      +--------------------------------+
   |      |                                       |
   |      +---------------------+                 |
   v                            v                 v
+-------------+         +-------------+   +-------------------+
| UTMALDG /   |         |   UBLKCP    |   | UTMASTG / UBLKRED |
| UTMAPF      |         |             |   |                   |
+-------------+         +-------------+   +-------------------+
| Implicit:   |         | Explicit:   |   | No Barrier Needed |
| dst_reg + 1 |         | 3rd Operand |   | (Data flows out)  |
| (e.g. UR16  |         |             |   |                   |
|  -> UR17)   |         |             |   |                   |
+-------------+         +-------------+   +-------------------+
```

## 2.2 SASS-Suffix Based SYNCS Decoding
The `SYNCS` instruction explicitly defines its role in the mbarrier protocol through its SASS suffix. The simulator deterministically decodes these suffixes to update the `HopperMBarrierObject` state.

> **CORRECTION (validated, see `.plan/SYNC_ISA.md`).** The arrive suffix is *not* a
> reliable decoder across toolchains: nvcc and CUTLASS use disjoint suffix spellings
> for the same operation (e.g. expect-tx is `.RED.A0TR` under nvcc but the suffix-less
> `SYNCS.ARRIVE.TRANS64` under CUTLASS/FA3). The reliable discriminator is the
> runtime **semantic operand value** (`semantic_raw`):
> `semantic_raw == 0` → plain arrive; `semantic_raw != 0` → expect-tx.
> The table below is kept for the nvcc microbench encoding only.

| SYNCS Suffix | Function | Operand Action | State Updated |
|---|---|---|---|
| `SYNCS.EXCH` | Init barrier | URsrc contains `0x200000 - 2*expected_arrive_count` | `expected_arrive_count` |
| `SYNCS.ARRIVE` | Thread arrival | None (increments count by the **active thread count**, `inst.active_count()`, per execution) | `arrive_count` |
| `SYNCS.ARRIVE.TRANS64.RED.A0TR` | Expect Tx / Reduce | URsrc contains expected Tx bytes | `expected_tx_bytes` |
| `WAIT` / `TRYWAIT` | Wait completion | URphase carries input parity in **bit 31**; proceed when `barrier.phase != ((raw>>31)&1)` | Evaluates readiness |

## 3.3 State Initialization & Arrive Count Decoding
Through microbenchmark trace analysis, it was discovered that the `SYNCS.EXCH` instruction does not receive the raw `arrive_count`. To utilize hardware counter overflow for phase flipping, the SASS operand (`URsrc`) encodes the arrive count by subtracting it from a large constant.

> **CORRECTION (validated).** The constant is `0x200000` (not `0x2000000`), and the
> raw encodes **2× the logical count**. Microbench ground truth (`init_arrivals`):
> `1→0x1ffffe`, `2→0x1ffffc`, `4→0x1ffff8`, `8→0x1ffff0`, i.e.
> `0x200000 - raw == 2 * init_arrivals`. So the decode is:
> `expected_arrive_count = (0x200000 - value_operand) / 2`.

Therefore, the simulator must decode the actual `expected_arrive_count` from the `value_operand`:
```cpp
// On SYNCS.EXCH (validated decode)
if (sync_info.has_value_operand) {
    uint32_t ENCODED_BASE = 0x200000;
    barrier_object.expected_arrive_count =
        (ENCODED_BASE - sync_info.value_operand) / 2;
}
```
*Note: base `0x200000` and the `/2` scale are validated against microbench `init_arrivals` knobs and FA3 count barriers (`0x200000 - raw = 512 → logical 256`).*

Similarly, `expected_tx_bytes` is dynamically accumulated by an expect-tx arrive
(`.RED.A0TR` under nvcc, suffix-less `SYNCS.ARRIVE.TRANS64` under FA3), which adds its
`value_operand`/`semantic_raw` directly to the expected transaction size without any
subtraction or decoding. The simulator selects this path at runtime when
`semantic_raw != 0`.

## 3.4 Math Calculation Logic (Barrier Readiness)
The actual condition for a barrier to be marked as `ready` and flip its phase is strictly evaluated by the `recompute_mbarrier_ready_and_maybe_flip_phase` function. A barrier is considered ready **only if both** the arrival count and transaction bytes conditions are satisfied:

```cpp
bool arrive_ready = (arrive_count >= expected_arrive_count);
bool tx_ready = (completed_tx_bytes >= expected_tx_bytes);

bool is_ready = arrive_ready && tx_ready;
```
When `is_ready` evaluates to true, the phase is flipped, allowing threads waiting on `SYNCS.PHASECHK` or `SYNCS.TRYWAIT` to proceed.

> **Wait-completion polarity (validated, was a bug).** A `SYNCS.PHASECHK` /
> `.TRYWAIT` consumer carries an input phase parity in **bit 31** of its wait-state
> raw (`(raw >> 31) & 1`), and the wait proceeds when that parity **DIFFERS** from
> the barrier's current phase parity (`barrier.phase != input_parity`) — the phase
> the consumer was waiting on has flipped away. The simulator originally decoded the
> parity from bit 0 (`raw & 0x1`) and compared with `==`. Because the FA3 wait-state
> only ever takes `0x0` / `0x80000000` (differing only in bit 31), the bit-0 decode
> collapsed both to `0`, making hit/miss depend solely on `barrier.phase`: every wait
> passed at `phase=0` and missed forever at `phase=1`. This caused the FA3-bwd
> deadlock where tx barrier `0x31000` flipped to `phase=1` (TMA bytes done) and then
> ~52k consumer waits spun without ever completing. Fix: decode parity from bit 31
> and use `!=` (`recompute`/`is_sync_wait_satisfied` in `remodeling/sm.cc`,
> `decode_sync_wait_phase`). See `.plan/SYNC_ISA.md`.

On phase flip, only the per-phase accumulators are reset (`arrive_count`, `completed_tx_bytes`, `expected_tx_bytes` → 0); **`expected_arrive_count` is preserved**. `SYNCS.EXCH` (= `mbarrier.init`) executes exactly once per barrier (validated in FA3 traces: one EXCH vs. many `PHASECHK`/arrive phases) and the hardware reuses that expected arrive count on every phase. Clearing it on flip would make the next phase ready after a single arrive (`arrive_count > 0 >= 0`), causing premature flips/races. `expected_tx_bytes` is reset because the expect-tx arrive re-sets it on each phase.

# 3. MBarrier State Management & Lifecycle

The `HopperMBarrierObject` maintains the exact architectural state required for deterministic synchronization, eliminating speculative attributes like `pending_value`, `wait_mode`, or `contract`.

## 3.1 Core Counters
- `expected_tx_bytes`: Total bytes expected to be written to shared memory.
- `completed_tx_bytes`: Total bytes that have successfully finished transferring.
- `expected_arrive_count`: Number of **threads** expected to arrive (logical, after EXCH `/2`).
- `arrive_count`: Current number of arrived **threads** (each `SYNCS.ARRIVE` adds `inst.active_count()`).
- `bound_pending_tx_bytes`: Transaction size currently in flight by the TMA engine.

## 3.2 Synchronization Lifecycle Diagram

```text
  Thread (Warp)                     MBarrier (UR17)                      TMA Engine
       |                                   |                                 |
       |--- SYNCS.EXCH (Init & Expect) --->|                                 |
       |                                   |                                 |
       |--- UTMALDG.64 UR16 (Issue) ---------------------------------------->|
       |                                   |                                 |
       |                                   |<--- Bind Tx Size (Pending) -----|
       |                                   |                                 |
       |--- SYNCS.ARRIVE (Inc arrive) ---->|                                 |
       |                                   |                                 |
       |  (Continues other work...)        |                                 |
       |                                   |                                 |
       |                                   |<--- Async Transfer Completes ---|
       |                                   |     (Add to completed_tx_bytes) |
       |                                   |                                 |
       |--- WAIT / TRYWAIT --------------->|                                 |
       |                                   | (Checks completed_tx >=         |
       |                                   |  expected_tx & arrives match)   |
       |<--- Phase Flip (Ready, Resume) ---|                                 |
       |                                   |                                 |
```

# 4. Deadlock Prevention (The Simulation Gap Fix)
**The Problem:** The old simulator model bound TMA completions via a single active slot per warp (`SM::m_active_mbarrier_addr_by_warp`), which was overwritten by every `SYNCS` instruction. FlashAttention 3 (FA3) rotates multiple barrier slots, meaning TMA completions would land on the wrong barrier object, causing deadlocks.

**The Solution:** 
When a TMA instruction executes, it no longer heuristically increments `expected_tx_bytes`. Instead, it safely binds its transaction size to `bound_pending_tx_bytes` directly on the address-exact barrier object looked up from the instruction. 
The `recompute_mbarrier_ready_and_maybe_flip_phase` function now strictly evaluates the exact hardware counters (completed vs. expected bytes and arrives) to determine barrier readiness and flip the phase.

# 5. Legacy Code Removal
All speculative and heuristic-based implementations have been purged:
- Removed `build_barrier_phase_resolver.py` and other PC-based heuristic scripts.
- Removed legacy JSON parsing logic (`load_barrier_wait_site_modes`).
- Removed `arm_mbarrier_wait_mode` and related wait-mode inference functions in `subcore.cc`.
- Deleted `syncs_operand_site_records` from the trace parser, as external lookups are no longer required.