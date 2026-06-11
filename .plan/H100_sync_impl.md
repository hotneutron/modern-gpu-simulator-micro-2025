# H100 SYNCS / mbarrier Implementation Plan

## Goal

Implement a real FA3-first Hopper `SYNCS` / mbarrier model so the simulator can make forward progress on H100 TMA kernels without relying on the temporary `SYNCS` bypass.

The implementation should:

- keep classic CTA barriers unchanged
- treat Hopper `SYNCS*` as address-based barrier-object operations
- make `SYNCS.PHASECHK` / `TRYWAIT` depend on real barrier-object readiness
- wire TMA completion bytes into barrier readiness
- remove the temporary `SYNCS` bypass after the new model is active

### Why This Goal Exists

#### 1) Difference Between Legacy Barriers and Hopper `mbarrier` / `SYNCS`

- Legacy `BAR.SYNC`-style barriers are **barrier-id-based collective synchronization**.
- Their core meaning is: “all participating warps in the CTA must arrive at the same barrier point before anyone is released.”
- Hopper `mbarrier` / `SYNCS`, in contrast, behaves as an **address-based barrier-object state machine**.
- Its core meaning is: “producer warps, consumer warps, and TMA all update and observe the state of a barrier object stored at a specific memory address.”

In short:

- `BAR.SYNC` is a **rendezvous where threads/warps meet**
- `mbarrier` / `SYNCS` is **asynchronous completion synchronization mediated through a shared memory-resident state object**

#### 2) Legacy Barrier Picture

```text
Warp 0  ----\
Warp 1  -----+----> [ BAR.SYNC id=0 ] ----> all released together
Warp 2  -----+
Warp 3  ----/
```

- All participating warps must arrive at the same barrier id.
- The barrier releases only when the final required participant arrives.
- The important state is:
  - barrier id
  - arriving warp set
  - arrival count

#### 3) Hopper `mbarrier` / `SYNCS` Picture

```text
                +----------------------+
                | mbarrier object @ A  |
                |----------------------|
Producer warp ->| arrive / exch        |
TMA engine ---->| complete bytes       |
Consumer warp ->| trywait / phasechk   |
                +----------------------+
```

- Producer-side warps prepare or signal the barrier object using `SYNCS.EXCH` / `SYNCS.ARRIVE`.
- TMA updates the same barrier object when asynchronous data movement completes.
- Consumer-side warps use `TRYWAIT` / `PHASECHK` to observe whether the object is ready.

So in Hopper, it is much more important to know:

- which **barrier address** is being referenced
- what the object’s **phase / ready / expected / completed** state is

than simply how many warps have “met” at a barrier.

#### 4) Why TMA Must Be Modeled Together With `SYNCS`

In Hopper kernels such as FA3, the following appear together:

- `SYNCS.EXCH`
- `SYNCS.ARRIVE.TRANS64`
- `SYNCS.PHASECHK.TRANS64.TRYWAIT`
- TMA-family instructions such as `UTMALDG`, `UTMASTG`, and `UBLKRED`

Conceptually, the intended flow is:

```text
(1) EXCH / ARRIVE prepares the barrier object
(2) TMA performs asynchronous transfer
(3) TMA completion updates the barrier object's ready state
(4) TRYWAIT / PHASECHK observes ready and proceeds
```

Therefore, if `SYNCS` is treated as a legacy CTA barrier, or if TMA completion is not connected to the barrier object:

- the waiter may be waiting on barrier A
- while TMA completion updates barrier B
- or the expected / completed comparison may never match
- and the kernel can deadlock

#### 5) What This Goal Really Means

The purpose of this document is not “fully implement every Hopper synchronization feature.”

Instead, the goal is to:

- capture the key semantics of Hopper `SYNCS` / `mbarrier` that FA3 actually depends on
- connect TMA completion to wait semantics
- make the kernel achieve at least **forward progress**

So the near-term goal is:

- leave legacy `BAR.SYNC` behavior untouched
- route Hopper `SYNCS` through a separate `MBARRIER_OP` path
- create address-based barrier objects
- make `TRYWAIT` / `PHASECHK` observe those objects
- make TMA completion transition those objects to ready

#### 6) Important Findings From Debugging So Far

- FA3 SASS uses **multiple distinct barrier addresses** within the same CTA.
- Therefore, the assumption “one active barrier per CTA” is too weak.
- The current implementation should prefer active barrier association by `(cta, warp)`, with CTA-wide fallback only when the active barrier address is unambiguous.
- Also, the observed `SYNCS.ARRIVE*` operand values such as `RZ`, `R3`, `R5`, and `R9` do not look trustworthy as transfer-byte counts.
- For the FA3-first model, expected bytes should therefore come from **TMA issue-time `cmd.total_bytes`**, not from decoded `ARRIVE` operand values.

## Design Summary

- Do not route Hopper `SYNCS` through legacy CTA `bar_id` barrier logic.
- Add a dedicated simulator-side mbarrier object model.
- Store mbarrier objects in SM/CTA shared-state ownership, not in `barrier_set_t`.
- Decode `SYNCS` forms from traced opcode + operands into structured sync semantics.
- Use TMA completion progress to update barrier-object readiness.

## File-by-File Plan

### 1. Add New Sync Types

**File**

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h`

**Add**

- `enum class HopperSyncOpKind`
  - `NONE`
  - `SYNCS_EXCH`
  - `SYNCS_ARRIVE`
  - `SYNCS_ARRIVE_RED`
  - `SYNCS_PHASECHK`
  - `SYNCS_TRYWAIT`
- `struct HopperMBarrierKey`
  - `unsigned cta_id`
  - `new_addr_type barrier_addr`
- `struct HopperMBarrierObject`
  - `new_addr_type barrier_addr`
  - `unsigned cta_id`
  - `uint64_t phase`
  - `uint64_t expected_tx_bytes`
  - `uint64_t completed_tx_bytes`
  - `uint32_t arrive_count`
  - `bool initialized`
  - `bool ready`
- `struct HopperSyncInstructionInfo`
  - `HopperSyncOpKind kind`
  - `new_addr_type barrier_addr`
  - `uint64_t value_operand`
  - `bool has_value_operand`
  - `bool is_reduction_form`
  - `bool valid`

**Also extend**

- `TMACommand`
  - `new_addr_type completion_barrier_addr`
  - `bool has_completion_barrier_addr`

## 2. Add a Dedicated Sync Opcode Category

**Files**

- `gpu-simulator/gpgpu-sim/src/operation_type.h`
- `gpu-simulator/ISA_Def/hopper_opcode.h`

**Change**

- Add `MBARRIER_OP` to `operation_type`.
- Remap Hopper `SYNCS` from `BARRIER_OP` to `MBARRIER_OP`.
- Keep classic `BAR` and `MEMBAR` unchanged.

**Reason**

- Hopper `SYNCS` is not a classic CTA barrier and should not enter `barrier_set_t`.

## 3. Extend warp_inst_t With Decoded Hopper Sync Info

**File**

- `gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h`

**Add**

- `HopperSyncInstructionInfo m_hopper_sync_info`
- `bool has_hopper_sync_info() const`
- `const HopperSyncInstructionInfo &get_hopper_sync_info() const`
- `void set_hopper_sync_info(const HopperSyncInstructionInfo &)`

**Reason**

- Avoid reparsing raw opcode strings at every execution point.
- Make `SYNCS` semantics explicit on the live instruction object.

## 4. Decode Each SYNCS Form During Trace-Driven Instruction Build

**File**

- `gpu-simulator/trace-driven/trace_driven.cc`

**Add helper functions**

- `bool is_syncs_opcode(const traced_instruction &)`
- `HopperSyncInstructionInfo decode_hopper_syncs_info(const traced_instruction &)`
- `new_addr_type decode_syncs_barrier_addr(...)`

**Populate during**

- `parse_from_trace_struct()`

**Decode rules**

- `SYNCS.EXCH.64`
  - `kind = SYNCS_EXCH`
  - barrier address = memory-ref operand
  - optional value operand = trailing register/immediate
  - semantic role = initialize or exchange barrier-object state
- `SYNCS.ARRIVE.TRANS64`
  - `kind = SYNCS_ARRIVE`
  - barrier address = memory-ref operand
  - optional value operand = bytes or contribution operand
  - semantic role = arrival on that barrier object
- `SYNCS.ARRIVE.TRANS64.RED.A1T0`
  - `kind = SYNCS_ARRIVE_RED`
  - same address decode
  - first implementation can treat reduction as normal arrival unless control flow requires more
- `SYNCS.PHASECHK.TRANS64`
  - `kind = SYNCS_PHASECHK`
  - same address decode
  - semantic role = evaluate barrier-object readiness or phase match
- `SYNCS.PHASECHK.TRANS64.TRYWAIT`
  - `kind = SYNCS_TRYWAIT`
  - same address decode
  - semantic role = non-blocking readiness check that keeps warp waiting until satisfied

## 5. Store mbarrier Objects in SM Shared State

**Files**

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`

**Add to `SM`**

- `std::map<HopperMBarrierKey, HopperMBarrierObject> m_mbarriers`
- `std::map<std::pair<unsigned, unsigned>, uint64_t> m_active_mbarrier_addr_by_warp`

**Add helpers**

- `HopperMBarrierObject &get_or_create_mbarrier(unsigned cta_id, new_addr_type addr)`
- `const HopperMBarrierObject *find_mbarrier(unsigned cta_id, new_addr_type addr) const`
- `void set_active_mbarrier_addr(unsigned cta_id, unsigned warp_id, new_addr_type addr)`
- `bool get_active_mbarrier_addr(unsigned cta_id, unsigned warp_id, new_addr_type &addr) const`
- `void reset_mbarriers_for_cta(unsigned cta_id)`

**Why SM**

- Barrier objects are address-based synchronization state tied to CTA progress.
- `barrier_set_t` is still correct for classic `BAR`, but not for `SYNCS`.

## 6. Add Real Issue / Wait Semantics for SYNCS

**Files**

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`

**Subcore gating**

- Add a predicate like:
  - `bool is_mbarrier_wait_satisfied(const warp_inst_t *inst) const`
- Integrate it into the issue gating path near current DEPBAR checks.

**Execution semantics**

- `SYNCS_EXCH`
  - create or find barrier object
  - mark initialized
  - update phase or base state
  - record the active barrier address for the issuing warp
- `SYNCS_ARRIVE`
  - increment `arrive_count`
  - record the active barrier address for the issuing warp
  - do not derive `expected_tx_bytes` from the operand value in the FA3-first model
- `SYNCS_ARRIVE_RED`
  - treat as `ARRIVE` for FA3-first implementation
- `SYNCS_PHASECHK`
  - evaluate readiness or phase
- `SYNCS_TRYWAIT`
  - if not ready, keep warp waiting
  - if ready, release warp

## 7. Replace the Temporary SYNCS Bypass

**Files**

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`

**Current temporary behavior**

- `SYNCS` is bypassed to avoid the legacy CTA barrier assert.

**After new model is active**

- Remove the temporary bypass logic.
- `SYNCS` should no longer enter `warp_reaches_barrier(...)`.
- Only classic CTA barriers should continue using `barrier_set_t`.

## 8. Wire TMA Completion Into Barrier Readiness

**File**

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc`

**At command build time**

- bind TMA completion to the active mbarrier address for the same `(cta, warp)` when available
- only use CTA-wide fallback if the CTA has a single unambiguous active barrier address
- populate:
  - `cmd.completion_barrier_addr`
  - `cmd.has_completion_barrier_addr`

**At TMA completion time**

- find matching `HopperMBarrierObject`
- update:
  - `completed_tx_bytes += transfer_bytes`
- if:
  - `completed_tx_bytes >= expected_tx_bytes`
  then:
  - `ready = true`
  - optionally advance `phase`

**FA3-first rule**

- do not use `SYNCS.ARRIVE*` operand values as expected transfer bytes
- `expected_tx_bytes` comes from TMA issue-time `cmd.total_bytes`
- `completed_tx_bytes` comes from TMA completion progress
- `TRYWAIT` and `PHASECHK` become satisfied when completed bytes meet expectation

## 9. Add Trace Operand Helpers for SYNCS Decoding

**Files**

- `util/traces_enhanced/src/traced_instruction.h`
- `util/traces_enhanced/src/traced_instruction.cc`
- if needed: `util/traces_enhanced/src/traced_operand.*`

**Add helpers**

- `bool traced_instruction::is_syncs_opcode() const`
- `std::optional<new_addr_type> traced_instruction::get_syncs_barrier_addr() const`
- `std::optional<uint64_t> traced_instruction::get_syncs_value_operand() const`

**Reason**

- Centralize operand extraction and avoid ad hoc string matching outside trace helpers.

## 10. Add Limited Debug Visibility

**Files**

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc`

**Add one-time debug prints**

- first few `SYNCS_EXCH`
- first few `SYNCS_ARRIVE`
- first few `TRYWAIT blocked`
- first few `TRYWAIT released`
- first few `TMA completion -> mbarrier ready`
- first few raw decoded `SYNCS` opcode and operand strings
- first few active-barrier overwrites
- first few `TMA bind-missing` events

**Add deadlock summaries**

- wait hotspot summary keyed by `(cta, pc, barrier_addr, sync_kind)`
- TMA↔mbarrier correlation summary keyed by `(cta, barrier_addr)`
- per-SM deadlock-time mbarrier object dump

**Purpose**

- confirm the new progress model is active
- quickly diagnose deadlocks if readiness never flips

## Implementation Order

### Step 1

- add new sync types in `tma_types.h`
- add `MBARRIER_OP` in `operation_type.h`
- remap Hopper `SYNCS` in `hopper_opcode.h`

### Step 2

- extend `warp_inst_t` in `abstract_hardware_model.h`
- decode `SYNCS` forms in `trace_driven.cc`

### Step 3

- add mbarrier storage and helpers in `sm.h` and `sm.cc`
- use warp-aware active barrier tracking instead of a single active barrier per CTA

### Step 4

- add wait and progress logic in `subcore.cc`

### Step 5

- wire TMA completion to barrier readiness in `tma_unit_sm.cc`
- set expected bytes from TMA issue-time transfer size rather than `SYNCS.ARRIVE` operand values

### Step 6

- remove the temporary `SYNCS` bypass from `shader.cc` and `sm.cc`

### Step 7

- rerun FA3 kernel 10
- then rerun the wider H100 config

## Current FA3-First Notes

- The single active-CTA mbarrier heuristic was too weak for FA3 because the kernel uses multiple live barrier addresses in the same CTA.
- The current implementation should prefer active barrier association by `(cta, warp)` and only use CTA-wide fallback when the active address is unambiguous.
- Static inspection of the FA3 SASS suggests `SYNCS.ARRIVE*` operands such as `RZ`, `R3`, `R5`, and `R9` are unlikely to represent transfer-byte counts.
- For the FA3-first model, expected bytes should therefore be driven by TMA command size, not by decoded `ARRIVE` operand values.
- The next likely semantic bug, if deadlock persists after these fixes, is that `SYNCS.EXCH` may still be resetting barrier state too aggressively.

## Explicit Non-Goals For First Pass

The first implementation does not need full Hopper synchronization completeness.

It can defer:

- `UCGABAR_ARV`
- `UCGABAR_WAIT`
- full cluster-scope synchronization
- exact reduction semantics for all `RED.*` forms
- full warpgroup synchronization completeness
- perfect parity or state-token fidelity if FA3 does not require it immediately

## Recommended Scope

Implement this as:

- **FA3-first H100 mbarrier support**

not as:

- **complete Hopper sync support in one step**

The temporary `SYNCS` bypass should remain only until the new `MBARRIER_OP` path is working end-to-end.
