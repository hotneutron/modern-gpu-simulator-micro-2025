# New Sync Plan

## Goal

Implement Hopper-style sync support with a clean separation between:

- static sync semantics from trace-side metadata
- dynamic sync runtime values from the execution trace
- simulator-side `mbarrier` execution state

This plan is intentionally **sync-first**. TMA completion integration is deferred until the standalone sync path is stable.

## Core Design

The clean design is:

1. **static sync semantics** from a sidecar, like TMA
2. **dynamic sync runtime values** in protobuf, unlike TMA descriptor configs
3. **simulator barrier state** keyed by trace-stable kernel identity plus CTA plus barrier address

The most important rule is:

- `barrier_addr` selects **which barrier object**
- `semantic_raw` tells the simulator **what to update/check on that barrier**

## Identity

Barrier object identity must be stable across filtered simulation and independent from simulator-local kernel renumbering.

Use:

- `trace_kernel_id`
- `cta_id`
- `barrier_addr`

Recommended key:

```cpp
struct HopperMBarrierKey {
  uint32_t trace_kernel_id;
  uint32_t cta_id;
  uint64_t barrier_addr;
};
```

Do not use:

- simulator-local kernel order
- `unique_function_id`
- wait token / semantic operand

Why:

- `trace_kernel_id` scopes one traced kernel launch instance
- `cta_id` scopes one CTA shared-memory namespace
- `barrier_addr` identifies the concrete mbarrier object within that CTA

## Tracer Design

Reuse the TMA-style sidecar idea only for **static site meaning**.

Do **not** put dynamic per-execution barrier address or semantic operand values into JSON sidecars.

Split tracer output into:

- `sync_operand_resolver.json`
- protobuf `instruction.sync`

### Static Sidecar

New file under `traces/extra_info/`:

- `sync_operand_resolver.json`

Role:

- one record per `(unique_function_id, pc)`
- says what a `SYNCS*` site means

Suggested schema:

```json
{
  "version": 1,
  "resolver": [
    {
      "unique_function_id": 1,
      "pc_hex": "0x0610",
      "opcode": "SYNCS.EXCH.64",
      "sync_kind": "EXCH",
      "barrier_operand_index": 1,
      "semantic_operand_index": 2,
      "semantic_operand_role": "EXCH_ARRIVE_COUNT_ENCODED",
      "barrier_operand_text": "[UR17]",
      "semantic_operand_text": "UR10"
    }
  ]
}
```

Core fields:

- `sync_kind`
- `barrier_operand_index`
- `semantic_operand_index`
- `semantic_operand_role`

### Dynamic Protobuf

Add a nested sync payload to `instruction.proto`.

```proto
message sync_runtime_info {
  bool valid = 1;
  uint64 barrier_addr = 2;
  bool has_semantic_raw = 3;
  uint64 semantic_raw = 4;
}

message instruction {
  uint32 pc = 1;
  uint32 active_mask = 2;
  uint32 predicate_mask = 3;
  int32 function_unique_id = 4;
  repeated address addresses = 5;
  sync_runtime_info sync = 6;
}
```

Keep `trace_kernel_id` outside per-instruction protobuf if it is already available from the enclosing kernel context. If the existing simulator loader cannot inherit it cleanly, duplicating it later is acceptable, but not required for the first tracer capture patch.

### Runtime Capture Rules

The runtime capture implementation must follow these rules:

- do not hardcode callback positions like "barrier is callback 1" and "semantic is callback 2"
- build a tracer-side per-`(unique_function_id, pc)` sync capture lookup during instrumentation
- map actual callback indices for:
  - barrier operand callback
  - semantic operand callback
- at runtime, use that lookup instead of assuming fixed callback order

Predicate handling is also strict:

- if `predicate_mask == 0`, leave `instruction.sync` unset
- do not fabricate `barrier_addr` or `semantic_raw` from `active_mask`
- simulator ingestion must also ignore sync runtime payload if `predicate_mask == 0` or `sync.valid == false`

This is required because predicated-off `SYNCS` instructions may still appear as dynamic instruction records in the trace stream, but they must not contribute sync state.

## Confirmed Operand Mapping

From confirmed SASS for current microbench traces, the useful pattern is stable:

- operand 2 = barrier memory reference
- operand 3 = semantic raw input/token

Confirmed examples:

- `SYNCS.EXCH.64 URZ, [UR14], UR4`
- `SYNCS.ARRIVE.TRANS64.RED.A0TR RZ, [UR17], R5`
- `SYNCS.PHASECHK.TRANS64 P0, [R6+URZ], R11`
- `SYNCS.PHASECHK.TRANS64 P0, [UR17], R5`

### Sync Kind Table

| `sync_kind` | barrier operand | semantic operand | simulator normalized field | simulator meaning |
|---|---|---|---|---|
| `EXCH` | operand 2 | operand 3 | `exch_arrive_count_encoded_raw` | initialize expected arrivals |
| `ARRIVE_EXPECT_TX` | operand 2 | operand 3 | `expect_tx_bytes_raw` | accumulate expected tx bytes |
| `ARRIVE_COUNTED` | operand 2 | operand 3 | `arrive_count_raw` | accumulate arrive count |
| `PHASECHK` | operand 2 | operand 3 | `wait_state_raw` | wait token/state |
| `TRYWAIT` | operand 2 | operand 3 | `wait_state_raw` | wait token/state |

## Simulator Design

Add a real Hopper `mbarrier` object, likely owned by `SM`.

Suggested state:

```cpp
struct HopperMBarrierObject {
  uint32_t expected_arrive_count = 0;
  uint32_t arrive_count = 0;
  uint32_t expected_tx_bytes = 0;
  uint32_t completed_tx_bytes = 0;
  uint32_t bound_pending_tx_bytes = 0;
  uint32_t phase = 0;  // parity bit: 0/1
  bool ready = false;
};
```

Two-phase parity model is enough for sync MVP:

- `0 -> 1 -> 0 -> 1 -> ...`

### Execution Semantics

#### `EXCH`

- key = `(trace_kernel_id, cta_id, barrier_addr)`
- create or reset the barrier object
- decode:

```cpp
expected_arrive_count = 0x200000 - exch_arrive_count_encoded_raw;
```

- clear phase-local counters

#### `ARRIVE_COUNTED` or implicit arrive

- increment `arrive_count`
- optionally preserve returned wait token for debug

#### `ARRIVE_EXPECT_TX`

- `expected_tx_bytes += expect_tx_bytes_raw`

#### `PHASECHK` / `TRYWAIT`

- use `barrier_addr` to find the barrier object
- use `wait_state_raw` only to test state against that barrier
- do **not** use `wait_state_raw` to find the barrier

#### Ready Condition

The barrier is ready only when:

```cpp
arrive_count >= expected_arrive_count &&
completed_tx_bytes >= expected_tx_bytes
```

On ready:

- flip phase parity
- clear or re-arm the phase according to chosen lifecycle rules

## PHASECHK / TRYWAIT Meaning

Important mapping:

- operand 2 picks **which barrier**
- operand 3 carries **which phase/state transition is being waited on**

The wait token is produced by arrive-side behavior, not by `EXCH`.

The microbench flow is:

```cpp
state = mbarrier_arrive(bar);
// or
state = mbarrier_arrive_count(bar, count);

mbarrier_wait(bar, state);
```

So:

- `barrier_addr` selects the barrier object
- `wait_state_raw` is compared against that barrier's current phase/state

## Validated Findings

The following findings are now validated by generated microbench traces:

- `SYNCS.EXCH` raw values scale consistently with `init-arrivals = 1/2/4/8`
- observed raw values:
  - `1 -> 0x1ffffe`
  - `2 -> 0x1ffffc`
  - `4 -> 0x1ffff8`
  - `8 -> 0x1ffff0`
- these values match:

```cpp
expected_arrive_count = 0x200000 - raw;
```

- therefore, the current validated decode base is `0x200000`, not `0x2000000`
- relevant producer/consumer pairing must be checked by barrier address, not by assuming the first `EXCH` site in the kernel is the producer for a later `PHASECHK`
- a later `PHASECHK` may legitimately pair with a different `EXCH` site on the same barrier address, for example `EXCH [UR17]` with later `PHASECHK [UR17]`

## Where Fields Live

### `sync_operand_resolver.json`

Static semantics by `(unique_function_id, pc)`.

### protobuf `instruction.sync`

- `barrier_addr`
- `semantic_raw`

### simulator `warp_inst_t` normalized sync payload

- `sync_kind`
- `barrier_addr`
- `exch_arrive_count_encoded_raw`
- `expect_tx_bytes_raw`
- `arrive_count_raw`
- `wait_state_raw`

### simulator barrier table

Keyed by:

- `(trace_kernel_id, cta_id, barrier_addr)`

## File-by-File Plan

### Tracer / static side

- `util/traces_enhanced/src/traced_instruction.h`
- `util/traces_enhanced/src/traced_instruction.cc`

Add sync-kind and operand-role classification helpers.

### Tracer / runtime side

- `util/tracer_nvbit/tracer_tool/common.h`
- `util/tracer_nvbit/tracer_tool/inject_funcs.cu`
- `util/tracer_nvbit/tracer_tool/tracer_tool.cu`

Capture:

- barrier operand effective address
- semantic operand raw value

### Protobuf

- `util/traces_enhanced/dynamic_trace/instruction.proto`

Regenerate protobuf bindings as needed by the build.

### Sidecar generator

New helper script, similar in spirit to TMA mapping:

- `util/tracer_nvbit/build_sync_operand_mapping.py`

### Simulator loader

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc`

Load `sync_operand_resolver.json`.

### Trace materialization

- `gpu-simulator/trace-driven/trace_driven.cc`

Attach normalized sync payload to `warp_inst_t`.

### Execution

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`

Optionally new files:

- `hopper_mbarrier.h`
- `hopper_mbarrier.cc`

### Wait integration

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc`

Make `PHASECHK/TRYWAIT` consult the Hopper barrier table.

## Bring-Up Order

### Phase 1

- static sidecar classification only
- no simulator behavior change

### Phase 2

- protobuf runtime capture for `barrier_addr` + `semantic_raw`
- add debug dump

### Phase 3

- simulator loader + normalized `warp_inst_t` sync payload

### Phase 4

- simulator `EXCH`

### Phase 5

- simulator counted/implicit arrive

### Phase 6

- simulator `PHASECHK/TRYWAIT`

### Phase 7

- simulator `ARRIVE_EXPECT_TX`

### Phase 8

- later, TMA completion bridge into `completed_tx_bytes`

## Validation Order

Start with microbenches that avoid TMA data movement:

- `utmapf_exch_arrive_micro`
- `utmapf_exch_arrive_fine_micro`
- `utmapf_exch_arrive_split_micro`
- `utmapf_exch_phase_micro`
- `utmapf_exch_multiphase_micro`

Then enable expect-tx path:

- `utmapf_exch_tx_micro`

Then move to FA3.

## Current Status

The sync-first bring-up has now reached a stable checkpoint.

Validated status:

- Hopper `SYNCS` is classified as dedicated `MBARRIER_OP`, not legacy `BARRIER_OP`
- legacy CTA barrier behavior remains on `BARRIER_OP`
- `FENCE` remains `MEMORY_BARRIER_OP`
- simulator-side Hopper sync state in `SM` is active and exercised
- pure sync-only microbench runs complete successfully:
  - `sync_arrive_micro:sync-arrive-1`
  - `sync_arrive_micro:sync-arrive-phase-4`
- existing `utmapf` sync-only-by-runtime configs now also complete successfully:
  - `utmapf_exch_arrive_micro:utmapf-exch-arrive-1`
  - `utmapf_exch_multiphase_micro:utmapf-exch-multiphase-1x1x4`

This means:

- the current Hopper sync logic works on both:
  - a truly TMA-free benchmark
  - the original `utmapf_probe` path when dead TMA execution is filtered correctly

## Fixes Applied So Far

### Simulator-side sync classification and execution

- added dedicated `MBARRIER_OP`
- mapped Hopper `SYNCS*` to `MBARRIER_OP`
- routed `MBARRIER_OP` through Hopper sync execution in remodeled `SM`
- kept legacy CTA barrier path unchanged for `BARRIER_OP`
- kept `FENCE` / `MEMBAR` on `MEMORY_BARRIER_OP`
- restored `SM::issue_warp()` split so `FENCE` does not incorrectly enter legacy CTA barrier handling

### Hopper sync state model

- barrier identity uses `(trace_kernel_id, cta_id, barrier_addr)`
- Hopper barrier state now lives in `SM`
- wait handling uses pending wait state keyed by barrier object, not by semantic token alone
- TMA completion hook back into sync state exists, but full TMA execution modeling is still not the current checkpoint goal

### Trace / build fixes needed during bring-up

- `util/traces_enhanced/Makefile` now tracks header dependencies, so opcode-map header changes rebuild correctly
- this fixed the stale-object case where source mapped `SYNCS -> MBARRIER_OP` but the runtime binary still used the old classification

### Filtering fixes for sync-only `utmapf` runs

Two separate filtering fixes were required:

1. tracer-side TMA runtime callback filtering

- predicated-off TMA callbacks are no longer written into `tma_runtime_operand_debug.jsonl`
- TMA runtime operand values are now sampled from the first predicated lane, not blindly from lane 0
- this keeps dead TMA sites from becoming `runtime_observed=true`

2. simulator-side zero-mask TMA skip

- zero-active traced TMA instructions are now treated as TMA-engine no-ops in `tma_unit_sm::issue()`
- this avoids Phase 2 operand-metadata asserts for instructions that remain in the main trace stream but have no active lanes

### Practical interpretation

The remaining work is no longer "basic sync bring-up".

The current checkpoint is:

- sync logic itself is working
- the original predicated-off filtering bug has been fixed
- the `utmapf` runtime path no longer fails on fake TMA execution for sync-only configurations
- next work can move forward from this stable baseline toward the next real TMA target

## First Implementation Slice

The first implementation target should be **tracer-side only**:

1. `sync_operand_resolver.json`
2. `instruction.proto` sync payload
3. tracer capture for:
   - `SYNCS.EXCH`
   - `SYNCS.ARRIVE.TRANS64.RED.A0TR`
   - `SYNCS.PHASECHK` / `SYNCS.TRYWAIT`

No simulator execution changes in the first patch.
