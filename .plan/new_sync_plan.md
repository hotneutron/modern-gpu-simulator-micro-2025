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
expected_arrive_count = 0x2000000 - exch_arrive_count_encoded_raw;
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

## First Implementation Slice

The first implementation target should be **tracer-side only**:

1. `sync_operand_resolver.json`
2. `instruction.proto` sync payload
3. tracer capture for:
   - `SYNCS.EXCH`
   - `SYNCS.ARRIVE.TRANS64.RED.A0TR`
   - `SYNCS.PHASECHK` / `SYNCS.TRYWAIT`

No simulator execution changes in the first patch.
