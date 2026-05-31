# TX-256B Revert And Reimplementation Plan

## Goal

Preserve a working path for the `utmapf-exch-tx-256b` microbench without
breaking FA3 trace generation and descriptor post-processing.

The immediate priority is:

- restore FA3-compatible trace generation behavior
- keep a record of the TX-256B root cause and successful experiments
- re-implement the TX-256B fix in a narrow, microbench-scoped way later

This document is intentionally biased toward short-term safety, not a full
generalized TMA descriptor architecture.

## What Happened

### TX-256B Root Cause

For `utmapf-exch-tx-256b`, the executed descriptor-backed TMA consumers did not
see the same identity as the object recorded at `cuTensorMapEncodeTiled`.

Observed behavior:

- producer-side encode dump recorded a host-side `CUtensorMap` object pointer
- runtime `UTMAPF` / `UTMALDG` observed a different descriptor pointer
- the microbench passes `CUtensorMap` by value into the kernel
- launch-time materialization changes the consumer-visible descriptor identity

So the old direct producer-pointer assumption was insufficient for this
microbench.

### Why The Broad Fix Worked For TX-256B

The broad fix introduced a launch-aware path:

- dump by-value `CUtensorMap` kernel args at `cuLaunchKernel`
- connect TMA producer chains back to formal KPARAM layout
- match launch arg blobs to encoded tensor-map blobs
- resolve runtime sites through launch argument identity

This successfully fixed `utmapf-exch-tx-256b`.

### Why That Regressed FA3

FA3 does not follow a single descriptor-identity pattern.

Per `TMA_TRACING.md`:

- `UTMALDG` is an explicit descriptor consumer (`desc[URx]`)
- `UTMAPF` is not supposed to be resolved as a direct descriptor carrier; it is
  linked through a later `UTMALDG` consumer

The broadened builder behavior effectively pushed too many opcode families into
the same strict direct descriptor-resolution model.

That caused FA3 trace post-processing to fail even though older, looser logic
could produce usable sidecars.

## Fresh Evidence Collected

### TX-256B

Validated:

- descriptor resolver succeeds
- operand resolver succeeds
- simulator no longer aborts in TMA Phase 2
- mocked TMA completions fire
- simulator run reaches normal exit

But TX-256B still has later sync-side work remaining; this document only covers
the descriptor mapping scope.

### FA3

Fresh FA3 trace with the current tracer shows mixed descriptor forms:

- some sites, especially `UTMAPF`, look pointer-like and plausible
- some `UTMALDG` sites are materialized from uniform immediates, not from the
  launch-arg/pointer model used by TX-256B

This means the current broad builder path is too narrow semantically for FA3.

## Recommendation

Do not throw away the TX-256B findings.

Instead:

1. Restore FA3-compatible descriptor-builder behavior.
2. Keep documentation of the TX-256B root cause and successful launch-aware
   mapping path.
3. Re-implement TX-256B support narrowly, scoped to the microbench or its exact
   kernel/descriptor pattern.

## What To Keep

These changes are still generally useful and should be considered for keeping
unless they are proven to regress FA3 by themselves:

1. Tracer-side consumer descriptor UREG extraction fixes in
   `util/tracer_nvbit/tracer_tool/tracer_tool.cu`
   - explicit-desc consumer detection improvements
   - trailing plain `[URx]` extraction support
   - fail-fast when an explicit-desc consumer has no descriptor UREG

2. Simulator-side descriptor pointer plumbing
   - `abstract_hardware_model.h`
   - `trace_driven.cc`
   - `gpu-sim.h`
   - `gpu-sim.cc`
   - `tma_types.h`
   - `tma_unit_sm.cc`

These are infrastructure improvements and do not necessarily force the broad
builder semantics by themselves.

## What To Roll Back

Roll back the broad descriptor-builder semantics that were made global for
executed descriptor-backed TMA sites:

1. `util/tracer_nvbit/build_tma_descriptor_mapping.py`
   - exact direct resolution requirements applied broadly to all descriptor-like
     families
   - launch-arg/KPARAM mapping as a generic primary rule
   - fail-fast conditions that block FA3 trace post-processing under the current
     opcode mix

2. `util/tracer_nvbit/tracer_tool/tracer_tool.cu`
   - the generic launch-arg dump can remain only if it is treated as a
     diagnostic artifact
   - do not rely on it as the generic resolver model for all workloads

The rollback target is not “everything new”, but “everything that made
TX-256B-style direct descriptor resolution a global assumption”.

## Narrow Reimplementation Plan

After rollback to FA3-compatible behavior:

1. Keep FA3 trace generation usable again.
2. Add a narrow TX-256B-only resolver path based on:
   - benchmark family or exact kernel name
   - by-value `CUtensorMap` launch arguments
   - launch-arg blob -> encoded tensor-map blob match
   - KPARAM-backed producer chain reconstruction

That narrow path should only activate where its assumptions are known to hold.

## Why This Is Preferred

This trades generality for safety:

- FA3 is the real target workload
- TX-256B still serves as a valuable regression case
- a generalized multi-opcode resolver is still the right long-term direction,
  but it should be implemented later, intentionally, not as an accidental side
  effect of fixing one microbench

## Suggested Workflow

1. Make a temporary checkpoint commit of the current state.
2. Roll back the broad builder behavior.
3. Confirm FA3 trace generation works again.
4. Re-implement the TX-256B descriptor fix in a narrow scope.
5. Validate:
   - FA3 trace generation remains usable
   - TX-256B trace and simulator matching still work

## Follow-Up Work

When revisiting the generalized implementation later, treat descriptor
resolution as opcode-aware:

- `UTMALDG`: explicit desc consumer
- `UTMASTG`: desc-like first operand consumer
- `UTMAPF`: descriptor link to later consumer, not direct descriptor config

That future work should start from `TMA_TRACING.md`, not from the narrower
TX-256B model.
