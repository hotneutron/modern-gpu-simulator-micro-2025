# Hopper Sync / TMA Current Algorithm

## Scope

This document describes the current implemented Hopper sync / TMA algorithm in
the repository.

It is not the ideal end-state design. It records what is actually implemented
now:

- tracer-side raw event capture
- offline barrier-phase resolver generation
- simulator-side sidecar loading
- runtime wait lookup and TMA binding flow
- remaining legacy heuristic behavior that still exists in runtime

## Current Pipeline

The current implementation has three stages:

1. tracer emits raw sync / wait / TMA execution events
2. post-processing builds `barrier_phase_resolver.json`
3. simulator loads the resolver sidecar and consults it at runtime

## Tracer Output

The current tracer pipeline emits these sync-related artifacts under
`extra_info/`:

- `sync_producer_events.jsonl`
- `sync_wait_events.jsonl`
- `tma_producer_events.jsonl`
- `barrier_phase_resolver.json`

The tracer also preserves the existing operand / descriptor helper artifacts:

- `syncs_operand_resolver.json`
- `tma_operand_resolver.json`
- `tma_descriptor_resolver.json`
- `tma_descriptor_configs.json`

## Raw Event Semantics

### Sync producer events

The tracer records executed sync producer instances such as:

- `SYNCS.EXCH`
- `SYNCS.ARRIVE`
- `SYNCS.ARRIVE_RED`

Important fields currently carried:

- `kernel_id`
- `unique_function_id`
- `pc_hex`
- CTA coordinates
- `warp_id_tb`
- `issue_seqno_tb`
- `opcode`
- `barrier_addr`
- `has_value_operand`
- `value_operand`

### Sync wait events

The tracer records executed wait instances such as:

- `SYNCS.PHASECHK`
- `SYNCS.TRYWAIT`

Important fields currently carried:

- `kernel_id`
- `unique_function_id`
- `pc_hex`
- CTA coordinates
- `warp_id_tb`
- `issue_seqno_tb`
- `barrier_addr`
- `has_wait_raw_operand`
- `wait_raw_operand`
- `wait_phase_parity_hint`

### TMA producer events

The tracer records executed TMA-family instances for:

- load-side TMA ops
- store-side TMA ops
- control-family ops that belong to the TMA path

Important fields currently carried depend on the opcode family, but the
intended common keys are:

- `kernel_id`
- `unique_function_id`
- `pc_hex`
- CTA coordinates
- `warp_id_tb`
- `issue_seqno_tb`
- `opcode`
- `opcode_family`
- `sync_domain`
- `descriptor_handle_hi`
- `total_bytes`

## Offline Resolver

`build_barrier_phase_resolver.py` builds `barrier_phase_resolver.json`.

The current resolver:

- groups executed sync and wait instances into per-barrier phases
- builds `mbarrier_phases`
- builds `wait_group_sequences`
- tries to attach load-side TMA launches to resolved barrier phases
- emits phase-level semantic summary fields

## Current Resolver Output

The runtime-relevant parts of `barrier_phase_resolver.json` currently include:

- `mbarrier_phases`
- `wait_group_sequences`
- `resolved_mbarrier_tma_bindings`

Each resolved `mbarrier` phase can currently carry:

- `phase_instance_id`
- `kernel_id`
- `cta`
- `barrier_addr`
- `phase_epoch`
- `require_arrive`
- `require_tx`
- `semantic_contract`
- `semantic_contract_confidence`
- `exch_value_semantics`
- `arrive_red_semantics`
- `wait_semantics`
- `raw_sync_events`
- `wait_events`
- `resolved_tma_launches`

The current resolver contract labels seen in FA3 include:

- `arrive_only`
- `arrive_and_tx`

The current `EXCH` semantic summary may include labels such as:

- `arrive_phase_token`

This is only a resolver-side diagnostic / semantic label. It is not yet fully
replayed by runtime logic.

## Simulator Sidecar Loading

The simulator currently loads:

- descriptor sidecars
- operand sidecars
- wait-site sidecar
- barrier phase resolver sidecar

The barrier-phase resolver metadata is stored in
`TMASidecarMetadataDB::barrier_phase_resolver_db`.

Current runtime indices include:

- exact wait-site lookup keyed by
  `(kernel_id, cta_x, cta_y, cta_z, barrier_addr, unique_function_id, pc)`
- fallback wait-site lookup without CTA
- exact TMA binding lookup keyed by
  `(kernel_id, cta_x, cta_y, cta_z, warp_id_tb, unique_function_id, pc)`
- fallback TMA binding lookup without CTA

## Current Runtime Kernel ID Rule

Runtime resolver lookup now uses the trace-side `kernel_id`, not simulator
launch uid.

This matters when the simulation runs only a filtered kernel such as
`OnlyKernel10`:

- runtime launch uid may be `1`
- trace / JSON `kernel_id` may still be `10`

Current wait lookup and TMA binding lookup use the trace kernel id from the
trace-driven kernel metadata.

## Current Runtime Barrier Object Model

The runtime still uses the existing `HopperMBarrierObject` model.

Important state fields that are still active in the current algorithm:

- `pending_value`
- `expected_arrive_count`
- `arrive_count`
- `expected_tx_bytes`
- `completed_tx_bytes`
- `bound_pending_tx_bytes`
- `ready`
- `phase`
- `contract`
- `wait_mode`

So current runtime is still a hybrid:

- resolver metadata is consulted first
- but runtime still performs local contract arming and pending-value commitment

## Current Wait Algorithm

At a `TRYWAIT` / `PHASECHK` site, runtime currently does:

1. decode runtime barrier address from the instruction
2. fetch the barrier object by `(cta_id, barrier_addr)`
3. if resolver metadata is available and local wait mode is unknown:
   - lookup resolved phase by
     `(trace kernel_id, CTA, barrier_addr, ufid, pc)`
   - map resolved phase to a wait classification
   - synthesize a `BarrierWaitSiteRecord`
   - arm the barrier wait mode from that record
4. if resolver lookup misses, fall back to `barrier_wait_site_modes.json`
5. check readiness using the barrier object's current counters

## Current Wait Classification Mapping

The runtime currently maps resolved phase metadata to wait classification using
the resolved phase contract:

- `arrive_only` -> `ArriveOnly`
- `arrive_and_tx` -> `ArriveAndTx`
- tx-only-like phase -> `TxOnly`
- otherwise -> `Unknown`

## Current Wait Arming Behavior

The current runtime still commits `pending_value` at wait-arm time.

Current behavior:

- `ArriveOnly`
  - `contract = ARRIVE_COUNT`
  - `pending_value -> expected_arrive_count`
- `TxOnly`
  - `contract = TX_BYTES`
  - `pending_value -> expected_tx_bytes`
- `ArriveAndTx`
  - `contract = TX_BYTES`
  - `pending_value -> expected_tx_bytes`

This is the most important remaining legacy heuristic in the current runtime.

## Current Producer Runtime Rules

### `SYNCS.EXCH`

Current runtime behavior:

- fetch / create the barrier object for `(cta_id, barrier_addr)`
- increment `arrive_count`
- if `has_value_operand`, add the raw value into `pending_value`
- recompute readiness immediately

Current runtime does **not** yet use resolved `exch_value_semantics` to decide
how the raw value should be interpreted.

### `SYNCS.ARRIVE` / `SYNCS.ARRIVE_RED`

Current runtime behavior:

- fetch / create the barrier object
- increment arrival-side state
- try to bind pending TMA completions through the existing runtime path
- recompute readiness

The old ARRIVE-side late binding path still exists for unresolved / deferred
cases.

## Current TMA Binding Algorithm

### At issue time

For a load-side TMA instruction, runtime currently tries:

1. lookup resolved TMA binding by
   `(trace kernel_id, CTA, warp_id_tb, ufid, pc)`
2. if found:
   - lookup the resolved phase record
   - set the command's completion barrier address from that phase
   - bind the command to the barrier object immediately
3. if not found:
   - fall back to the old deferred binding path

### At completion time

When the TMA command completes:

- `completed_tx_bytes` is credited to the bound barrier object
- readiness is recomputed

## Current Store-Side Status

Store-side wait-group metadata is generated in the resolver output, but the
runtime path is not yet fully migrated to a final producer-driven wait-group
model.

So current implementation status is:

- metadata side exists
- full runtime completion-domain separation is not finished

## Current Algorithm Summary

The current implemented algorithm is:

1. trace raw executed sync / wait / TMA instances
2. build offline barrier-phase metadata
3. load the metadata into the simulator
4. use trace `kernel_id` for runtime resolver lookup
5. resolve wait sites from phase metadata first
6. resolve TMA completion barrier at issue time when the binding exists
7. keep legacy wait-side `pending_value` commitment and legacy deferred binding
   as fallback behavior

## Current Known Mismatch With Intended Final Design

The following behaviors are still present in the current implementation:

- wait-side contract arming is still active
- `pending_value` is still reinterpreted at wait time
- runtime does not yet replay resolved `exch_value_semantics`
- deferred late binding still exists as a fallback
- `barrier_wait_site_modes.json` is still present as a fallback path

These are current implementation facts, not target behavior.

## Current FA3-Relevant Failure Mode

In the current FA3 debug state:

- wait at `pc=0x9ef0` resolves successfully to
  `k10.cta0,0,0.bar0x31020.phase0`
- that phase is labeled `arrive_only`
- runtime still commits raw `EXCH` value `2096640` into
  `expected_arrive_count`
- the barrier never becomes ready

So the main currently observed failure is not resolver lookup anymore. It is
the interaction between:

- resolved `ArriveOnly` wait classification
- legacy runtime `pending_value -> expected_arrive_count` arming

## Files Involved

- tracer:
  - `/home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu`
- resolver:
  - `/home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/build_barrier_phase_resolver.py`
- sidecar loading:
  - `/home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc`
- wait runtime:
  - `/home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc`
- sync runtime:
  - `/home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`
- TMA runtime:
  - `/home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc`
