# TX-256B Revert And Reimplementation Plan

## Goal

Preserve a working path for the `utmapf-exch-tx-256b` microbench without
breaking FA3 trace generation, descriptor post-processing, or existing valid
descriptor binds.

The immediate priority is:

- restore FA3-compatible trace generation behavior
- keep a record of the TX-256B root cause and successful experiments
- re-implement the TX-256B fix in a narrow, behavior-derived way later
- make executed descriptor-involved TMA sites fail clearly if they cannot be
  uniquely bound

This document remains intentionally biased toward short-term safety, but the
reimplementation path below is now stricter about final binding correctness and
FA3 regression checking.

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
- `UTMASTG` uses a desc-like first operand pair, not the same exact model as
  `UTMALDG`
- descriptor-backed `UBLKRED` exists in FA3 and carries layout information in
  the descriptor in addition to span-like operand metadata
- `UTMAREDG` should also be treated as descriptor-backed when present

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

This means the current broad builder path was too narrow semantically for FA3
because it pushed multiple opcode families into one direct descriptor-resolution
model.

## Recommendation

Do not throw away the TX-256B findings.

Instead:

1. Restore FA3-compatible descriptor-builder behavior.
2. Keep documentation of the TX-256B root cause and successful launch-aware
   mapping path.
3. Re-implement TX-256B support narrowly, but activate it from traced behavior
   and evidence, not from benchmark or application naming.
4. Require every executed descriptor-involved TMA site to end with exactly one
   final bind, or fail clearly.

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
   - any launch-arg dump can remain only if it is treated as generic evidence
     and not as the universal resolver model for all workloads
   - do not rely on launch-arg identity as the generic primary rule for all
     descriptor-involved opcode families

The rollback target is not “everything new”, but “everything that made
TX-256B-style direct descriptor resolution a global assumption”.

## Updated Binding Policy

The new reimplementation should distinguish between:

- validated resolver heuristics
- rescue fallbacks

Validated resolver heuristics are allowed if they:

- are derived from traced runtime behavior plus opcode semantics
- are deterministic
- end in one unique final binding

Rescue fallbacks are not allowed as final authorities for executed
descriptor-involved sites if they:

- only guess based on weak similarity
- leave multiple candidates alive
- promote one candidate without strong evidence

The final rule is:

- every executed descriptor-involved TMA site must bind to exactly one
  descriptor config if that site's semantics require descriptor binding
- otherwise the generator or simulator must fail clearly

Important scope:

- this rule applies to executed sites, not to all statically discovered TMA
  sites in disassembly

## Descriptor-Involved Opcode Coverage

The strict binding work must cover more than `UTMALDG` and `UTMAPF`.

### Direct descriptor-backed families

- `UTMALDG`
- `UTMAREDG`
- descriptor-backed `UBLKRED`

These require a unique final `config_id` for every executed site.

### Indirect or desc-like descriptor families

- `UTMAPF`
- `UTMASTG`

`UTMAPF` must resolve through a validated descriptor link to a later descriptor
consumer.

`UTMASTG` must resolve through its validated desc-like first operand pair rule,
not by pretending it is identical to `UTMALDG`.

### Operand-sensitive or control-only families

- `UBLKCP`
- `UBLKPF`
- `UTMACCTL.PF`
- `UTMACMDFLUSH`

These do not necessarily require a descriptor config bind at every site, but
they still require correct operand/control metadata for executed sites.

## Narrow Reimplementation Plan

After rollback to FA3-compatible behavior:

1. Keep FA3 trace generation usable again.
2. Add a generic launch-evidence capture path that records by-value kernel
   argument materialization without relying on benchmark name.
3. Add a narrow TX-256B-compatible resolver path based on:
   - by-value `CUtensorMap` launch arguments
   - launch-arg blob -> encoded tensor-map blob exact match
   - KPARAM-backed producer chain reconstruction where needed
   - runtime consumer identity at the actual executed site
4. Only activate that path when its evidence pattern is observed in the trace.

That narrow path should be behavior-derived, not benchmark-aware.

## File-Level Implementation Direction

### 1. Tracer

`util/tracer_nvbit/tracer_tool/tracer_tool.cu`

- keep `cuTensorMapEncodeTiled` capture
- add generic launch-time kernel-argument evidence capture
- write a new launch artifact under `extra_info/`
- do not encode application-specific policy in the tracer

### 2. Orchestrator

`util/tracer_nvbit/run_hw_trace.py`

- keep it generic and artifact-driven
- allow it to run the new post-pass only when the necessary artifacts exist
- do not teach it benchmark or application identity

### 3. Descriptor resolver

`util/tracer_nvbit/build_tma_descriptor_mapping.py`

- promote validated FA3 heuristics into explicit resolver methods
- keep behavior-derived exact or validated-heuristic binding rules
- remove rescue fallback rules as final authorities for executed
  descriptor-involved sites
- add a final verification pass:
  - executed descriptor-involved site -> exactly one bind
  - otherwise fail with a precise message

### 4. Operand resolver

`util/tracer_nvbit/build_tma_operand_mapping.py`

- keep `UTMAPF` indirect descriptor-link logic
- keep `UTMASTG` desc-like first-pair logic
- keep per-site classification for descriptor-backed vs non-descriptor
  `UBLKRED`
- make `UTMAREDG` handling explicit as a descriptor-backed family when present

### 5. Simulator loader and execution

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h`
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc`

Changes:

- reject malformed or ambiguous descriptor binds at load time
- preserve per-site distinction for descriptor-backed `UBLKRED`
- improve runtime assert diagnostics with:
  - `unique_function_id`
  - `pc`
  - `handle_hi`
  - opcode family
  - failure reason

## FA3 Comparison Gate

Before enabling the new strict bind policy broadly, compare new FA3 binding
outputs against the saved baseline trace:

- `~/project/modern-gpu-simulator-micro-2025/simulator-remodeled/hw_run/traces/device-0/12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1.bak/`

Use that saved FA3 trace as the baseline oracle for current binding behavior.

### Comparison targets

Compare:

- `tma_descriptor_resolver.json`
- `tma_descriptor_configs.json`
- `tma_operand_resolver.json`

For executed descriptor-involved sites, compare by:

- `unique_function_id`
- `pc_hex`
- `handle_hi_hex` where applicable

### Families included in FA3 binding comparison

- `UTMALDG`
- `UTMAPF`
- `UTMASTG`
- descriptor-backed `UBLKRED`
- `UTMAREDG` if present in the trace

For operand/control families, compare metadata stability separately:

- `UBLKCP`
- `UBLKPF`
- `UTMACCTL.PF`
- `UTMACMDFLUSH`

### Acceptance criteria

- previously valid FA3 binds must not silently change
- changed `config_id` for an already-bound executed FA3 site requires manual
  review
- stronger provenance with the same `config_id` is acceptable
- previously unresolved sites may become uniquely bound
- newly ambiguous or newly unbound executed descriptor-involved sites are
  failures

### Suggested tool

Add a comparison helper such as:

- `util/tracer_nvbit/compare_tma_binding_outputs.py`

It should report:

- unchanged binds
- newly resolved binds
- changed binds
- ambiguous or missing binds
- baseline-only or new-only executed sites

## Why This Is Preferred

This trades generality for safety:

- FA3 is the real target workload
- TX-256B still serves as a valuable regression case
- a generalized multi-opcode resolver is still the right long-term direction,
  but it should be implemented later, intentionally, from opcode-aware rules
  rather than as an accidental side effect of fixing one microbench

## Suggested Workflow

1. Make a temporary checkpoint commit of the current state.
2. Roll back the broad builder behavior.
3. Confirm FA3 trace generation works again.
4. Re-implement the TX-256B descriptor fix in a narrow, behavior-derived scope.
5. Generate new FA3 sidecars and compare them against the saved FA3 baseline
   trace bindings.
6. Validate:
   - FA3 trace generation remains usable
   - FA3 binding outputs remain stable or improve in a reviewable way
   - TX-256B trace and simulator matching still work
   - executed descriptor-involved sites now fail clearly if unique binding is
     impossible

## Follow-Up Work

When revisiting the generalized implementation later, treat descriptor
resolution as opcode-aware:

- `UTMALDG`: explicit desc consumer
- `UTMASTG`: desc-like first operand consumer
- `UTMAPF`: descriptor link to later consumer, not direct descriptor config
- `UTMAREDG`: descriptor-backed reduction family
- `UBLKRED`: per-site distinction between descriptor-backed and non-descriptor
  forms
- `UTMACCTL.PF` / `UTMACMDFLUSH`: control-state families, not descriptor-config
  carriers

That future work should start from `TMA_TRACING.md`, not from the narrower
TX-256B model.
