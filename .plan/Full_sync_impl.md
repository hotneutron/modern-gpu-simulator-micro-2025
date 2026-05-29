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

| SYNCS Suffix | Function | Operand Action | State Updated |
|---|---|---|---|
| `SYNCS.EXCH` | Init barrier | URsrc contains `0x2000000 - expected_arrive_count` | `expected_arrive_count` |
| `SYNCS.ARRIVE` | Thread arrival | None (Increments count by 1 per execution) | `arrive_count` |
| `SYNCS.ARRIVE.TRANS64.RED.A0TR` | Expect Tx / Reduce | URsrc contains expected Tx bytes | `expected_tx_bytes` |
| `WAIT` / `TRYWAIT` | Wait completion | URphase contains parity | Evaluates readiness |

## 3.3 State Initialization & Arrive Count Decoding
Through microbenchmark trace analysis, it was discovered that the `SYNCS.EXCH` instruction does not receive the raw `arrive_count`. To utilize hardware counter overflow for phase flipping, the SASS operand (`URsrc`) encodes the arrive count by subtracting it from a large constant (`0x2000000`).

Therefore, the simulator must decode the actual `expected_arrive_count` from the `value_operand`:
```cpp
// On SYNCS.EXCH
if (sync_info.has_value_operand) {
    // Decode the actual arrive count from the encoded hardware operand
    uint32_t ENCODED_CONSTANT = 0x2000000; 
    barrier_object.expected_arrive_count = ENCODED_CONSTANT - sync_info.value_operand;
}
```
*Note: `0x2000000` is derived from trace trends and represents the hardware's internal overflow threshold.*

Similarly, `expected_tx_bytes` is dynamically accumulated via the `SYNCS.ARRIVE.TRANS64.RED.A0TR` instruction, which adds its `value_operand` directly to the expected transaction size without any subtraction or decoding.

## 3.4 Math Calculation Logic (Barrier Readiness)
The actual condition for a barrier to be marked as `ready` and flip its phase is strictly evaluated by the `recompute_mbarrier_ready_and_maybe_flip_phase` function. A barrier is considered ready **only if both** the arrival count and transaction bytes conditions are satisfied:

```cpp
bool arrive_ready = (arrive_count >= expected_arrive_count);
bool tx_ready = (completed_tx_bytes >= expected_tx_bytes);

bool is_ready = arrive_ready && tx_ready;
```
When `is_ready` evaluates to true, the phase is flipped, allowing threads waiting on `SYNCS.PHASECHK` or `SYNCS.TRYWAIT` to proceed.

# 3. MBarrier State Management & Lifecycle

The `HopperMBarrierObject` maintains the exact architectural state required for deterministic synchronization, eliminating speculative attributes like `pending_value`, `wait_mode`, or `contract`.

## 3.1 Core Counters
- `expected_tx_bytes`: Total bytes expected to be written to shared memory.
- `completed_tx_bytes`: Total bytes that have successfully finished transferring.
- `expected_arrive_count`: Number of threads/warps expected to arrive.
- `arrive_count`: Current number of arrived threads/warps.
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