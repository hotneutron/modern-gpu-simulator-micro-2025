# Lighter NVBit Tracing - compact full-capture transport

This document replaces the original spin-only optimization plan. It first
summarizes that experiment, then defines the adopted full-capture transport
change and the decision gate for a later two-run trace reorganization.

The root problem is capture clock skew:

- Full NVBit instrumentation slows Consumer warps on every dynamic instruction.
- The Producer's real-time `NANOSLEEP` does not dilate by the same factor.
- The instrumented H100 execution therefore performs far more mbarrier polls
  than an uninstrumented execution.
- That timing-dependent control flow is recorded in the dynamic trace and then
  replayed by the simulator.

Hardware reference:

| Kernel | H100 `NANOSLEEP` executions | Heavy-trace producer polls | Inflation |
|---|---:|---:|---:|
| FWD K5 | 8,461 | 407,234 | 48.1x |
| BWD K10 | 7,860 | 678,768 | 86.4x |

## 1. Previous implementation - spin-only gather reduction

The first implementation added `TRACER_LIGHT_SPIN`.

- `NANOSLEEP`, `BRA`, and `SYNCS.PHASECHK/TRYWAIT` were routed through a
  first-lane scalar gather instead of the full 32-lane gather.
- The existing `inst_trace_t` layout and channel push size were unchanged.
- Consumer-heavy instructions such as HGMMA, MUFU, LDGSTS, and memory
  operations still used the original path.

Result:

- The reported producer-spin metric did not improve (`821K -> 822K`).
- The experiment removed only a minority gather cost.
- Every callback still pushed the fixed approximately 712B `inst_trace_t`
  through `ChannelDev`.
- The Consumer path that creates the Producer/Consumer clock skew remained
  heavy.

Conclusion:

- The spin-only path was safe but ineffective.
- The old spin-only behavior is removed.
- The existing `TRACER_LIGHT_SPIN` configuration name is reused as the
  full-capture transport selector described below.

## 2. Adopted implementation - compact full capture

### Details

The new implementation reduces the GPU-to-host transport record for every
dynamic callback while preserving the final protobuf contract.

The old `inst_trace_t` contained four 32-lane arrays:

```text
addrs_or_reg_val_0[32]  = 256B
reg_val_1[32]           = 128B
reg_val_2[32]           = 128B
reg_val_3[32]           = 128B
metadata                =  72B
total                   = 712B
```

Most callbacks do not need those lane arrays after host post-processing.
`inst_trace_t` is therefore now a versioned variable-size wire record with a
compact scalar prefix and an optional lane payload.

| Configuration | Device callback | Wire format |
|---|---|---|
| `TRACER_LIGHT_SPIN=0` | `instrument_inst` | Original fixed 712B legacy record |
| `TRACER_LIGHT_SPIN=1` | `instrument_inst_compact` | New 104B/360B compact record |

Current Option 1 configuration:

```text
TRACER_LIGHT_SPIN=1
TRIM_SPIN=0
```

| Wire record | Size | Payload |
|---|---:|---|
| Compact non-memory | 104B | Identity, PC, masks, callback metadata, first predicated-active lane scalars |
| Memory-like | 360B | Compact prefix plus 32 lane values and TMA descriptor fields |
| Legacy fixed record | 712B | Four unconditional 32-lane arrays |

Per-callback transport reduction:

| Callback class | Reduction |
|---|---:|
| Non-memory | 85.4% |
| Memory-like | 49.4% |

### Goal

Produce one full, untrimmed trace whose Producer/Consumer timing distortion is
small enough that the recorded spin count approaches uninstrumented H100,
without deleting spin instructions offline.

### Why this is the next decision point

The failed spin-only experiment did not reduce the dominant fixed-size channel
copy or Consumer instrumentation cost. Compact full capture attacks both:

- non-memory Consumer callbacks send 104B instead of 712B;
- non-memory register values use one predicated-active lane instead of a
  32-lane gather;
- memory/TMA callbacks retain every lane address needed by the simulator;
- channel flush frequency and host drain pressure should decrease with total
  wire bytes.

This experiment intentionally does not yet:

- merge multiple operand callbacks into one callback;
- replace `ChannelDev` with per-warp device buffers;
- change the final protobuf schema;
- alter simulator mbarrier semantics;
- trim or reorganize any spin loop.

## 3. Data preservation boundary

The reduction follows the actual host and simulator consumers rather than
dropping fields by opcode name alone.

| Dynamic data | Required by | New capture |
|---|---|---|
| CTA/warp identity | Host trace organization | Preserved |
| PC/function ID | Static SASS lookup and replay | Preserved |
| Active/predicate masks | Dynamic execution mask | Preserved |
| 32-lane memory addresses | Memory/TMA replay | Preserved in memory record |
| TMA descriptor ID/value | TMA post-processing | Preserved |
| Sync barrier address | Hopper mbarrier key | Preserved |
| Sync semantic/phase | PHASECHK/TRYWAIT semantics | Preserved |
| BAR ID/count | Named barrier replay | Preserved |
| Non-memory register arrays | Not serialized into replay trace | Replaced by first-lane scalars |
| Static control bits | Enhanced SASS sidecar | Unchanged |

For scalar runtime operands, the device records the first lane in
`active_mask & predicate_mask`. This matches the host's existing BAR and sync
selection rule. If every lane is predicated off, the host does not create a
runtime sync/BAR payload.

Absent TMA debug operands are represented explicitly:

- scalar value slots without a captured value are `0`;
- callbacks without a register operand emit `operand_reg_id=-1`;
- `SECRET_UREG_DESC_NOT_USED=133` remains restricted to descriptor fields.

## 4. Implementation flow

### Device instrumentation

The instrumentation site selects one device callback for the whole capture:

1. `TRACER_LIGHT_SPIN=0` inserts the original `instrument_inst` callback;
2. `TRACER_LIGHT_SPIN=1` inserts `instrument_inst_compact`;
3. the compact callback collects `active_mask` and `predicate_mask`;
4. it records first-predicated-lane values for non-memory callbacks;
5. it appends the 32-lane payload only for `STANDARD_MEM`, `CONSTANT_MEM`, and
   `CALL_OR_RET`;
6. it calls `ChannelDev::push(&record, record.record_size)`.

The dynamic callback count is unchanged. This isolates record-size and gather
reduction from a later callback-merging experiment.

`run_hw_trace.py` persists the mode in generated artifacts:

- generated `run.sh` explicitly exports the selected
  `TRACER_LIGHT_SPIN=0/1` value;
- compact capture intentionally uses the original full-trace run directory so
  the simulator consumes the same path;
- the user backs up the previous trace externally before capture, and the new
  compact trace overwrites the working trace location.

### Host receiver

`recv_thread_fun` selects the parser using the same configuration.

- Legacy records are normalized into the common host representation before
  protobuf generation.
- Compact records carry `record_size`, `record_version`, and `record_flags`.
- The receiver rejects:
  - truncated records;
  - unknown versions or flags;
  - invalid compact/memory sizes;
  - `mem_type` and lane-payload mismatches.
- Compact scalar values and memory lane values are routed into the same
  protobuf fields used before this change.
- The channel flush token follows the selected legacy or compact format.

Channel transport statistics are emitted as:

```text
[TRACE-WIRE] mode=legacy|compact
             legacy_records=...
             compact_records=...
             memory_records=...
             wire_bytes=...
             legacy_bytes=...
             reduction=...%
```

### Files

- `simulator-remodeled/util/tracer_nvbit/tracer_tool/common.h`
- `simulator-remodeled/util/tracer_nvbit/tracer_tool/inject_funcs.cu`
- `simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu`

## 5. Validation

The user performs the build and GPU runs.

### Guard 0 - wire integrity

- No `[TRACE-WIRE]` version, size, flags, truncation, or payload/type failure.
- Record counts and byte reduction are printed at shutdown.
- Expected reduction is between the two per-record bounds, depending on the
  memory callback fraction.

### Guard 1 - protobuf contract

Compare two captures from the same binary:

- baseline: `TRACER_LIGHT_SPIN=0`;
- candidate: `TRACER_LIGHT_SPIN=1`.

Compare all fields not expected to change due to capture timing:

- non-spin PC sequence;
- function ID;
- active/predicate masks;
- every memory address;
- sync barrier address and phase;
- BAR ID/count;
- TMA descriptor/runtime values;
- HGMMA, MUFU, and TMA counts.

Any missing memory, sync, BAR, or TMA payload is a stop condition.

Spin iteration counts are allowed to change because reducing capture dilation
is the purpose of this experiment.

### Guard 2 - hardware timing fidelity

Measure:

- FWD K5 per-PC `NANOSLEEP`, target total `8,461`;
- BWD K10 per-PC `NANOSLEEP`, target total `7,860`;
- K5 Producer-only tail, target mean approximately `7,950.9 cycles`;
- capture duration and run-to-run stability;
- per-CTA HGMMA/TMA work distribution.

### Guard 3 - simulator

Only after Guards 0-2 pass:

- run the untrimmed compact trace;
- verify clean completion;
- verify memory/TMA/sync work counts;
- compare K5/K10 cycles and Producer/Consumer lifetime against HW.

## 6. Decision after the run

- If spin count and role lifetime move close to HW:
  - adopt the compact full trace;
  - retire offline NCU-proportional trim.
- If wire bytes fall substantially but clock skew remains:
  - next reduce callback count and channel synchronization through callback
    merging and/or per-warp device buffering.
- If a practical single-run tracer still cannot approach HW:
  - move to Option 2;
  - disable compact capture (`TRACER_LIGHT_SPIN=0`);
  - reuse `TRIM_SPIN=1` after replacing its current aggregate NCU trim with
    the per-burst reorganization path;
  - capture one heavy full run and one Producer-only ultra-light run;
  - match spin bursts by `(kernel, CTA, warp, loop PC, occurrence index)`;
  - reorganize the full trace with measured per-burst counts/timing;
  - abort on burst count/order mismatch.

Do not repeat the current aggregate NCU-proportional trim. It corrects total
spin count without restoring per-burst Producer/Consumer pacing.

`TRACER_LIGHT_SPIN=1` and `TRIM_SPIN=1` are mutually exclusive and
`run_hw_trace.py` rejects that combination.

## 7. Status

- Compact wire layout implemented.
- `TRACER_LIGHT_SPIN=0/1` selects legacy/compact full capture.
- Compact runs reuse the original trace directory and self-contained `run.sh`.
- Device non-memory scalar path implemented.
- Host mixed-record parser and validation implemented.
- TMA absent-value sentinel filtering implemented.
- `[TRACE-WIRE]` accounting implemented.
- `TRIM_SPIN=1` mutual-exclusion guard implemented.
- Static source and diff checks passed.
- Build, GPU capture, protobuf parity, and HW spin measurements are pending.
