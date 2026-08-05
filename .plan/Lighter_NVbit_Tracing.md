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

Option 1 has two stages:

- **Option 1A** reduces the GPU-to-host record size for every callback.
- **Option 1B** removes operand callbacks whose values have no simulator or
  post-processing consumer.

Both stages preserve the final protobuf contract.

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

Current Option 1B configuration:

```text
TRACER_LIGHT_SPIN=1
GATHER_REGISTERS=0
TRIM_SPIN=0
```

`GATHER_REGISTERS=1` disables pruning and restores Option 1A's all-operand
compact callback stream for debugging and A/B comparison.

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

The first Option 1A capture intentionally did not:

- merge multiple operand callbacks into one callback;
- replace `ChannelDev` with per-warp device buffers;
- change the final protobuf schema;
- alter simulator mbarrier semantics;
- trim or reorganize any spin loop.

Its measured 84.81% byte reduction lowered spin by only 42-43%. Static code
inspection then found that ordinary HGMMA, MUFU, and ALU instructions still
issued one callback for every REG/UREG/PRED operand even though those values
never enter the replay protobuf. Option 1B removes those callbacks before
moving to the two-run fallback.

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
| TMA-family runtime operands | TMA operand/base-map post-processing | All callbacks preserved |
| Generic compute register values | No replay or post-processing consumer | Operand callbacks pruned |
| Static control bits | Enhanced SASS sidecar | Unchanged |

For retained scalar runtime operands, the device records the first lane in
`active_mask & predicate_mask`. This matches the host's existing BAR and sync
selection rule. If every lane is predicated off, the host does not create a
runtime sync/BAR payload.

The pruning allowlist is semantic:

- every memory reference, including dynamic constant-memory and call/return
  addresses;
- the runtime sync semantic/phase operand;
- register-form BAR ID/count;
- every operand of TMA-family opcodes consumed by
  `tma_runtime_operand_debug.jsonl` and strict TMA sidecar generation;
- one metadata-only callback when an instruction has no retained operand.

`tma_desc_producer_debug.csv` may lose generic UREG producer candidates under
pruning. Repository-wide consumer search found no simulator or post-processing
reader for that diagnostic CSV. Descriptor configs, exact PC base maps, the
TMA operand resolver, and dynamic TMA descriptor/address payloads do not depend
on it and remain preserved.

### Consumer audit

The allowlist was derived from concrete consumers:

| Consumer | File | Data retained |
|---|---|---|
| Dynamic replay protobuf schema | `util/traces_enhanced/dynamic_trace/instruction.proto` | PC, function, masks, addresses, sync, BAR |
| Simulator trace decoder | `gpu-simulator/trace-parser/trace_parser.cc` | Active/predicate masks, memory/TMA payload, sync semantic, BAR runtime |
| TMA base-map pipeline | `util/tracer_nvbit/build_tma_pc_base_map.py` | Every executed TMA-family runtime operand |
| TMA operand resolver | `util/tracer_nvbit/build_tma_operand_mapping.py` | TMA callback index and runtime values |
| Sync metadata pipeline | `util/tracer_nvbit/build_sync_operand_mapping.py` | Static operand roles plus dynamic sync protobuf values |

Repository search found no simulator read of generic dynamic REG/UREG/PRED
values and no reader of `tma_desc_producer_debug.csv`. The protobuf instruction
message has no generic register-value field. Therefore pruning those callbacks
does not remove replay state.

Absent TMA debug operands are represented explicitly:

- scalar value slots without a captured value are `0`;
- callbacks without a register operand emit `operand_reg_id=-1`;
- `SECRET_UREG_DESC_NOT_USED=133` remains restricted to descriptor fields.

## 4. Implementation flow

### Device instrumentation

The instrumentation site first classifies the original operands, then selects
the compact callback set:

1. `TRACER_LIGHT_SPIN=0` inserts the original `instrument_inst` callback;
2. `TRACER_LIGHT_SPIN=1` inserts `instrument_inst_compact`;
3. legacy mode and `GATHER_REGISTERS=1` retain the original operand list;
4. compact mode with `GATHER_REGISTERS=0` keeps only the semantic allowlist
   above and remaps sync/BAR callback indices to the retained list;
5. if the allowlist is empty, one `NO_REGS` callback preserves PC/function and
   active/predicate masks;
6. the compact callback records first-predicated-lane values for retained
   scalar operands;
7. it appends the 32-lane payload only for `STANDARD_MEM`, `CONSTANT_MEM`, and
   `CALL_OR_RET`;
8. it calls `ChannelDev::push(&record, record.record_size)`.

The host still receives `num_of_injects`, but that count now means retained
callbacks for the instruction. Existing instruction grouping therefore
continues to produce exactly one protobuf instruction.

`run_hw_trace.py` persists the mode in generated artifacts:

- generated `run.sh` explicitly exports the selected
  `TRACER_LIGHT_SPIN=0/1` value;
- generated `run.sh` explicitly exports `GATHER_REGISTERS=0/1`;
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

For Option 1B, compare two captures from the pruning-capable binary:

- all-operand compact baseline:
  `TRACER_LIGHT_SPIN=1 GATHER_REGISTERS=1`;
- pruned compact candidate:
  `TRACER_LIGHT_SPIN=1 GATHER_REGISTERS=0`.

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

The first compact trace measured:

| Kernel | Option 1A callbacks | Dynamic warp instructions | Callbacks/instruction |
|---|---:|---:|---:|
| K5 | 45,993,757 | 15,359,561 | 2.99 |
| K10 | 59,730,650 | 21,474,079 | 2.78 |

Option 1B should move callback count toward one per dynamic instruction, with
small excesses for multi-memory-reference, sync/BAR, and TMA instructions.
Failure to reduce K5/K10 callbacks by roughly 60% is an implementation stop
condition.

`stats.csv` retains its historical `total_insts` column name, but that value is
the number of device callback records, not the number of protobuf warp
instructions. Use it to measure pruning reduction against the Option 1A
baseline above.

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

Functional-gate logging uses:

```text
-trimchk_percta_log_enable 1
-sync_debug_enable 0
-tma_debug_enable 0
-bar_debug_enable 1
-bar_debug_issue_budget 0
```

`TRIMCHK/TRACECHECK` provide per-CTA load and protobuf-contract evidence.
SYNCDBG and TMA Phase2 summaries are emitted without enabling their per-event
logs. BARDBG keeps release/CTA summary/leak validation while budget 0
suppresses BAR issue and MEMBAR detail lines.

The sum of all `[TRACECHECK]` CTA rows must equal:

| Contract metric | K5 expected | K10 expected |
|---|---:|---:|
| CTA rows | 132 | 384 |
| Warp instructions | 14,847,194 | 21,007,397 |
| Active-lane instructions | 450,627,804 | 624,799,542 |
| Memory-reference records | 342,688 | 1,396,238 |
| Memory-reference lane uses | 3,180,354 | 25,289,459 |
| Sync runtime payloads | 258,342 | 682,925 |
| Sync semantic payloads | 258,342 | 682,925 |
| BAR runtime payloads | 51,840 | 52,992 |
| BAR runtime IDs | 51,840 | 52,992 |
| BAR runtime counts | 0 | 0 |
| Zero-mask instructions | 250,964 | 93,738 |
| HGMMA instructions | 349,056 | 835,584 |
| MUFU instructions | 1,926,144 | 1,671,168 |
| TMA-family instructions | 6,432 | 30,336 |

`BAR runtime counts == 0` is expected: this workload captures register-form
BAR IDs dynamically while BAR counts are immediate and decoded statically.

Additional runtime gates:

- `gpgpu_n_tot_w_icount` equals the expected warp-instruction total;
- `gpu_sim_insn` equals the expected active-lane-instruction total;
- sidecar bootstrap reports `descriptor_configs=3`, `operand_sites=77`,
  `base_records=48`, and `[SYNC] loaded operand_sites=187`;
- K5 TMA Phase2 reports `commands total=6,432`, `unresolved=0`,
  `real_base_commands=6,168`, `synthetic_commands=264`;
- K10 TMA Phase2 reports `commands total=30,336`, `unresolved=0`,
  `real_base_commands=23,424`, `synthetic_commands=6,912`;
- no deadlock, assertion, missing trace file, or protobuf parse failure;
- any `[BARDBG][leak]` must be compared with the prior valid trace before the
  run is accepted.

Do not require `SYNCDBG missing_runtime=0`; the existing model intentionally
skips unsupported static sync sites. The gate is preservation of the aggregate
runtime/semantic payload counts above and clean completion.

## 6. Compact full-capture result

The first compact trace was captured from the same FA3 workload as the saved
legacy full run:

```text
candidate: .../flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1
baseline:  .../flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1.fullrun
```

The candidate `run.sh` records `TRACER_LIGHT_SPIN=1`. It predates the Option
1B pruning implementation, so it is the all-operand Option 1A baseline even
though the old binary accepted the default `GATHER_REGISTERS=0`. Both runs
contain the same 11 kernels, CTA counts, and 10,982 threadblock protobuf files.

### Spin result

The comparison uses the strict contiguous four-instruction loop body and sums
all warps, not only the warp with the largest spin count. This is required
because K5 `0xacc0` and K10 `0xb4f0` execute outside the single selected
Producer warp.

| Kernel | HW target | Legacy full | Compact full | Compact change | Compact / HW |
|---|---:|---:|---:|---:|---:|
| FWD K5 | 8,461 | 406,994 | 230,433 | -43.4% | 27.2x |
| BWD K10 | 7,860 | 678,048 | 393,110 | -42.0% | 50.0x |

The compact transport removes a substantial part of the capture clock skew,
but it does not make the trace representative of uninstrumented H100.

K5 detail:

| TRYWAIT PC | HW | Legacy full | Compact full | Compact / HW |
|---|---:|---:|---:|---:|
| `0xac40` | 132 | 132 | 132 | 1.0x |
| `0xacc0` | 3,148 | 1,499 | 2,418 | 0.8x |
| `0xb090` | 662 | 9,235 | 6,228 | 9.4x |
| `0xb1d0` | 955 | 136,311 | 74,974 | 78.5x |
| `0xb230` | 96 | 9,909 | 5,344 | 55.7x |
| `0xb290` | 1,051 | 157,298 | 85,952 | 81.8x |
| `0xb2f0` | 100 | 10,198 | 5,144 | 51.4x |
| `0xb350` | 484 | 39,536 | 21,721 | 44.9x |
| `0xb3b0` | 1,833 | 42,876 | 28,520 | 15.6x |

K10 detail:

| TRYWAIT PC | HW | Legacy full | Compact full | Compact / HW |
|---|---:|---:|---:|---:|
| `0xb480` | 384 | 384 | 384 | 1.0x |
| `0xb4f0` | 777 | 21,548 | 10,349 | 13.3x |
| `0xb820` | 2,017 | 200,707 | 111,057 | 55.1x |
| `0xb960` | 2,804 | 274,126 | 152,160 | 54.3x |
| `0xba20` | 390 | 32,267 | 18,090 | 46.4x |
| `0xbae0` | 720 | 65,742 | 37,854 | 52.6x |
| `0xbb40` | 768 | 83,274 | 63,216 | 82.3x |

### Protobuf preservation result

The dynamic protobuf was decoded independently of the simulator and compared
by static opcode class.

| Preserved dynamic data | Legacy K5 | Compact K5 | Legacy K10 | Compact K10 |
|---|---:|---:|---:|---:|
| HGMMA | 349,056 | 349,056 | 835,584 | 835,584 |
| MUFU | 1,926,144 | 1,926,144 | 1,671,168 | 1,671,168 |
| TMA | 6,432 | 6,432 | 17,280 | 17,280 |
| BAR runtime payload | 51,840 | 51,840 | 52,992 | 52,992 |
| Decoded address lanes | 435,840 | 435,840 | 13,713,408 | 13,713,408 |
| Zero decoded addresses | 0 | 0 | 0 | 0 |

Additional preservation checks:

- normalized enhanced SASS is identical; raw files differ only in relocated
  function addresses;
- `sync_operand_resolver.json` is byte-identical;
- TMA resolver entry count and the `(function, PC, opcode, role)` projection
  are identical;
- TMA descriptor configs and descriptor offsets are byte-identical;
- TMA PC base-map structure is identical after removing process-specific GPU
  allocation base addresses;
- the compact resolver removes the legacy absent-operand sentinel pollution
  without changing its static operand semantics.

The full protobufs are intentionally not byte-identical. The lighter capture
changes timing-dependent execution:

- K5 dynamic warp instructions: `16,064,281 -> 15,359,561`;
- K10 dynamic warp instructions: `22,626,216 -> 21,474,079`;
- mbarrier spin accounts for nearly all of this reduction;
- USETMAXREG allocation polling and a small number of other synchronization
  paths also change;
- active/predicate masks at those timing-dependent branches therefore differ.

Those differences are execution changes caused by lower capture overhead, not
missing compact-record fields. Cross-run raw addresses also cannot be compared
byte-for-byte because the GPU allocations use different virtual bases; payload
presence, lane count, and TMA descriptor structure were preserved.

The GPU-server shutdown record confirms the actual transport reduction:

| Wire metric | Compact capture |
|---|---:|
| Compact non-memory records | 188,663,153 |
| Memory records | 3,123,267 |
| Total callback records | 191,786,420 |
| Actual wire bytes | 20,745,344,032 |
| Equivalent legacy bytes | 136,551,931,040 |
| Byte reduction | 84.81% |
| Legacy/compact traffic ratio | 6.58x |

Non-memory records are 98.37% of all callbacks, so the observed reduction is
close to the 85.4% non-memory record bound. The implementation therefore
removed the intended fixed-record payload cost. However, reducing transport
from 136.55GB to 20.75GB reduced spin by only 42-43%, leaving 27-50x HW
inflation. Remaining skew is dominated by the approximately 191.8M injected
callbacks, per-callback channel synchronization, and Consumer instrumentation,
not by additional removable bytes in the compact record.

### Option 1B callback-pruning result

The pruned trace was captured with:

```text
TRACER_LIGHT_SPIN=1
GATHER_REGISTERS=0
TRIM_SPIN=0
```

The saved Option 1A trace is the all-operand compact baseline.

| Callback metric | Option 1A | Option 1B | Change |
|---|---:|---:|---:|
| Whole workload | 191,786,420 | 64,775,615 | -66.2% |
| K5 | 45,993,757 | 15,102,306 | -67.2% |
| K10 | 59,730,650 | 21,666,115 | -63.7% |
| K5 callbacks / warp instruction | 2.99 | 1.017 | -66.0% |
| K10 callbacks / warp instruction | 2.78 | 1.031 | -62.9% |

The callback-count target passes. K5/K10 are now close to one callback per
dynamic warp instruction; the remaining excess is the expected retained
memory, sync/BAR, and TMA operand payload.

Option 1B wire result:

| Wire metric | Option 1A | Option 1B | Change |
|---|---:|---:|---:|
| Compact non-memory records | 188,663,153 | 62,139,404 | -67.1% |
| Memory records | 3,123,267 | 2,636,211 | -15.6% |
| Total callback records | 191,786,420 | 64,775,615 | -66.2% |
| Actual wire bytes | 20,745,344,032 | 7,411,533,976 | -64.3% |
| Equivalent legacy bytes | 136,551,931,040 | 46,120,237,880 | -66.2% |
| Per-record byte reduction | 84.81% | 83.93% | -0.88pp |

Option 1B sends 7.41GB, 18.4x less than the original 136.55GB
all-operand/fixed-record baseline. This is a 94.6% end-to-end traffic
reduction after composing callback pruning with compact records.

Spin result:

| Kernel | HW target | Option 1A | Option 1B | Change vs 1A | Option 1B / HW |
|---|---:|---:|---:|---:|---:|
| FWD K5 | 8,461 | 230,433 | 102,296 | -55.6% | 12.1x |
| BWD K10 | 7,860 | 393,110 | 278,107 | -29.3% | 35.4x |

K5 per-loop:

| TRYWAIT PC | HW | Option 1A | Option 1B | Option 1B / HW |
|---|---:|---:|---:|---:|
| `0xac40` | 132 | 132 | 132 | 1.0x |
| `0xacc0` | 3,148 | 2,418 | 3,928 | 1.2x |
| `0xb090` | 662 | 6,228 | 3,371 | 5.1x |
| `0xb1d0` | 955 | 74,974 | 30,722 | 32.2x |
| `0xb230` | 96 | 5,344 | 1,715 | 17.9x |
| `0xb290` | 1,051 | 85,952 | 35,673 | 33.9x |
| `0xb2f0` | 100 | 5,144 | 1,742 | 17.4x |
| `0xb350` | 484 | 21,721 | 9,410 | 19.4x |
| `0xb3b0` | 1,833 | 28,520 | 15,603 | 8.5x |

K10 per-loop:

| TRYWAIT PC | HW | Option 1A | Option 1B | Option 1B / HW |
|---|---:|---:|---:|---:|
| `0xb480` | 384 | 384 | 384 | 1.0x |
| `0xb4f0` | 777 | 10,349 | 3,804 | 4.9x |
| `0xb820` | 2,017 | 111,057 | 83,667 | 41.5x |
| `0xb960` | 2,804 | 152,160 | 112,285 | 40.0x |
| `0xba20` | 390 | 18,090 | 13,482 | 34.6x |
| `0xbae0` | 720 | 37,854 | 26,102 | 36.3x |
| `0xbb40` | 768 | 63,216 | 38,383 | 50.0x |

Replay and post-processing preservation:

| Preserved data | Option 1A K5 | Option 1B K5 | Option 1A K10 | Option 1B K10 |
|---|---:|---:|---:|---:|
| HGMMA | 349,056 | 349,056 | 835,584 | 835,584 |
| MUFU | 1,926,144 | 1,926,144 | 1,671,168 | 1,671,168 |
| TMA | 6,432 | 6,432 | 17,280 | 17,280 |
| BAR runtime payload | 51,840 | 51,840 | 52,992 | 52,992 |
| Decoded memory-address lanes | 435,840 | 435,840 | 13,713,408 | 13,713,408 |
| Zero decoded addresses | 0 | 0 | 0 | 0 |

Additional gates:

- both traces contain the same 11 kernels and 10,982 threadblock protobufs;
- normalized enhanced SASS is identical;
- `sync_operand_resolver.json`, TMA descriptor configs, and descriptor offsets
  are byte-identical;
- TMA runtime operand rows are identical at 77,736;
- the TMA resolver static projection and normalized PC base-map structure are
  identical;
- `tma_desc_producer_debug.csv` is header-only under pruning, as expected; no
  simulator or post-processing consumer reads it.

Option 1B therefore passes callback reduction and data-preservation gates, but
still fails the HW spin-fidelity gate.

## 7. Decision after the run

Option 1A is a partial success:

- the wire representation preserves simulator-visible data;
- K5/K10 spin falls by approximately 42-43%;
- the remaining 27-50x HW inflation fails the timing-fidelity gate.

Option 1B passes the callback and payload gates but still leaves K5/K10 spin at
12.1x/35.4x HW. The next sequence is:

1. run the Option 1B compact trace through the simulator functional gate;
2. verify clean completion and replay work counts;
3. proceed to Option 2 for the remaining timing-dependent spin distortion.

Option 2 remains:

1. use the compact full trace as the complete-data source because its dynamic
   payload is preserved and its timing distortion is smaller;
2. keep the saved legacy `.fullrun` only as the parity and rollback baseline;
3. implement a Producer-only ultra-light run with no per-instruction host
   channel traffic;
4. record each spin burst by
   `(kernel, CTA, warp, loop PC, occurrence index)`;
5. replace the current aggregate NCU-proportional behavior behind
   `TRIM_SPIN=1` with strict per-burst reorganization;
6. copy all non-spin instructions and dynamic payloads from the compact full
   run;
7. use the ultra-light run only to select spin count and timing per burst;
8. abort on CTA/warp/burst count, order, barrier address, or phase mismatch.

Do not repeat the current aggregate NCU-proportional trim. It corrects total
spin count without restoring per-burst Producer/Consumer pacing.

`TRACER_LIGHT_SPIN=1` and `TRIM_SPIN=1` remain mutually exclusive within one
capture invocation. Option 2 first captures the compact full trace, then runs
the reorganizer as a separate offline stage using the ultra-light sidecar.

## 8. Status

- Compact wire layout implemented.
- `TRACER_LIGHT_SPIN=0/1` selects legacy/compact full capture.
- Compact runs reuse the original trace directory and self-contained `run.sh`.
- Device non-memory scalar path implemented.
- Host mixed-record parser and validation implemented.
- TMA absent-value sentinel filtering implemented.
- Semantic-role callback pruning implemented for compact
  `GATHER_REGISTERS=0`.
- Memory, sync semantic, BAR runtime, and all TMA-family operand callbacks are
  explicitly retained.
- `GATHER_REGISTERS=1` restores the all-operand compact fallback.
- Generated `run.sh` persists the pruning selector.
- `[TRACE-WIRE]` accounting implemented.
- `TRIM_SPIN=1` mutual-exclusion guard implemented.
- Static source and diff checks passed.
- Build and compact GPU capture completed.
- Simulator-visible protobuf payload preservation passed.
- `[TRACE-WIRE]` measured 20.75GB compact traffic versus 136.55GB legacy
  equivalent, an 84.81% reduction.
- Compact spin reduction measured at 43.4% for K5 and 42.0% for K10.
- HW timing-fidelity gate failed at 27.2x K5 and 50.0x K10 residual spin.
- Capture wall time remains to be collected from the GPU-server log.
- Option 1B build and GPU capture completed.
- Option 1B callback reduction passed at 66.2% workload-wide.
- Option 1B wire traffic is 7.41GB, 64.3% below Option 1A and 94.6% below
  the original all-operand/fixed-record equivalent.
- Option 1B replay/TMA/sync payload parity passed.
- Option 1B spin is 12.1x HW for K5 and 35.4x HW for K10.
- Option 1B simulator functional run is pending.
- Option 2 per-burst ultra-light capture and reorganization are not yet
  implemented.
