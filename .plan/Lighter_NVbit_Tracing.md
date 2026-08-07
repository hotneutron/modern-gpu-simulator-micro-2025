# Lighter NVBit Tracing - compact full-capture transport

This document replaces the original spin-only optimization plan. It first
summarizes that experiment, then defines the adopted full-capture transport
change, the next single-run tracer stages, and the parked two-run fallback.

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
- `TRACER_LIGHT_SPIN` was reused for the first compact implementation and is
  now retained only as a deprecated compatibility input for
  `LIGHTER_TRACER`.

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
| `LIGHTER_TRACER=off` | `instrument_inst` | Original fixed 712B legacy record |
| `LIGHTER_TRACER=compact_wire` | `instrument_inst_compact` | New 104B/360B compact record |

Current Option 1B configuration:

```text
LIGHTER_TRACER=compact_wire,prune_operands
TRIM_SPIN=0
```

Omitting `prune_operands` restores Option 1A's all-operand compact callback
stream for debugging and A/B comparison.

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

### Why this remains the primary path

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

1. `LIGHTER_TRACER=off` inserts the original `instrument_inst` callback;
2. `compact_wire` inserts `instrument_inst_compact`;
3. legacy mode and compact mode without `prune_operands` retain the original
   operand list;
4. `compact_wire,prune_operands` keeps only the semantic allowlist
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

- generated `run.sh` unsets deprecated selectors and exports the canonical
  `LIGHTER_TRACER` string;
- `lighter_tracer_config.json` records the requested/canonical configuration,
  feature list, configuration hash, artifact suffix, and git SHA;
- lighter captures append a readable configuration suffix plus an
  eight-character hash to the original argument-folder name and refuse to
  overwrite an existing run directory;
- the selected `<benchmark>/<argument-folder>_<config>` run directory can be
  passed directly to `run_simulations.py -T`;
- `LIGHTER_TRACER=off` retains the legacy unsuffixed argument-folder behavior.

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
- `simulator-remodeled/util/tracer_nvbit/run_hw_trace.py`
- `simulator-remodeled/util/job_launching/run_simulations.py`

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
  `LIGHTER_TRACER=compact_wire`;
- pruned compact candidate:
  `LIGHTER_TRACER=compact_wire,prune_operands`.

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

## 7. Decision after the simulator functional run

Option 1B completed the simulator functional gate:

| Result | K5 fwd `.o68/.e68` | K10 bwd `.o42/.e42` |
|---|---:|---:|
| Cycles | 83,308 | 136,898 |
| Error versus HW | 23.1% | 3.0% |
| Change versus Opt 10 | -21.0% | -23.5% |
| Warp instructions | 14,847,194 | 21,007,397 |
| Active-lane instructions | 450,627,804 | 624,799,542 |
| TMA-family instructions | 6,432 | 30,336 |
| TMA unresolved / BAR leak / fatal | 0 / 0 / 0 | 0 / 0 / 0 |

### Payload/functional gate - PASS

- All 132 K5 and 384 K10 `[TRACECHECK]` rows sum exactly to the expected
  protobuf contract.
- `gpgpu_n_tot_w_icount` and `gpu_sim_insn` exactly match the loaded Option 1B
  trace.
- Memory-reference records/lane uses, sync runtime/semantic payloads, BAR IDs,
  HGMMA, MUFU, and TMA-family work all match their independently decoded
  totals.
- Sidecars load as `descriptor_configs=3`, `operand_sites=77`,
  `base_records=48`, and sync sites `187`.
- TMA Phase2 reports `unresolved=0`; BAR reports no leak; both kernels exit
  cleanly.

### Timing-fidelity gate - FAIL

| Role lifetime | K5 | K10 |
|---|---:|---:|
| Producer mean completion | 39,084.7 | 97,190.7 |
| Consumer mean completion | 77,017.8 | 109,147.8 |
| Producer - Consumer | **-37,933.1** | **-11,957.0** |

Real K5 HW has an approximately `+7,950.9-cycle` Producer tail, not a negative
tail. Runtime NANOSLEEP also remains 102,535 K5 / 278,827 K10 versus HW
8,461 / 7,860, or 12.1x / 35.5x.

The K10 3.0% aggregate cycle error is therefore not full timing validation.
Excess captured spin and premature simulated Producer completion can partially
cancel.

### Revised direction

The previous aggregate count-trim and the proposed dual-run merge are not the
primary next step.

1. Continue reducing single-run tracer observer effect until the untrimmed full
   trace approaches HW spin counts. The next concrete levers are removal of
   per-callback device statistic atomics, a smaller metadata-only callback, and
   per-warp device buffering that avoids one `ChannelDev::push` per dynamic
   instruction.
2. Investigate simulated Producer/Consumer lifetime as an independent timing
   model problem. A dynamic full trace preserves instruction order and payload,
   but it does not contain the original HW inter-warp timestamps. The simulator
   must reconstruct timing through scheduling, PHASECHK/mbarrier handling, and
   TMA completion.
3. The fact that the simulator finishes the Producer early despite replaying
   12.1x/35.5x excess NANOSLEEP is strong evidence that role pacing is wrong in
   the simulator independently of the remaining capture pollution.
4. Do not add modeled-ready blocking without first measuring per-burst
   PHASECHK readiness, Consumer ARRIVE/TMA completion, Producer TMA issue, and
   role-exit ordering. That path still carries the deadlock risk identified
   earlier.
5. Keep Option 2 as a fallback if further single-run tracer reduction cannot
   approach HW and the two-run scheduler/mapping gates can be satisfied.

The current compact `LIGHTER_TRACER` / `TRIM_SPIN=1` mutual-exclusion guard
remains in place. Do not repeat the current aggregate NCU-proportional trim: it
corrects total spin count without preserving Producer/Consumer timing.

### Next-session implementation handoff

This section is the executable handoff for the next tracer iteration. The work
must remain a single-run, untrimmed full capture. Implement and measure one
stage at a time so a spin reduction can be attributed to a specific source of
observer effect.

#### Starting point

Code baseline:

```text
ed6e519  compact 104B/360B full-trace transport
c797d49  unused operand callback pruning
d06df70  simulator compact-trace functional contract logging
```

Current accepted capture mode:

```text
LIGHTER_TRACER=compact_wire,prune_operands
TRIM_SPIN=0
```

Reference artifacts:

```text
Option 1B:
simulator-remodeled/hw_run/traces/device-0/12.8/
  flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/
  flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1.opt1b

Option 1A:
same path with suffix .opt1a

Legacy full capture:
same path with suffix .fullrun

Option 1B simulator:
H100_80GB-OnlyKernel5/*.o68 and *.e68
H100_80GB-OnlyKernel10/*.o42 and *.e42
```

The user performs the NVBit build, GPU capture, and long simulator runs. Do not
launch those experiments locally.

#### Non-negotiable contract

- Keep at least one callback-derived record for every dynamic instruction,
  including instructions whose retained operand list is empty.
- Preserve CTA/warp identity, PC/function ID, active/predicate masks, all
  memory lane addresses, sync barrier/semantic values, BAR ID/count, and all
  TMA-family runtime operands.
- Keep the final protobuf schema unchanged.
- Keep `TRIM_SPIN=0`; do not delete, clone, or reorganize dynamic instructions.
- A new wire representation must be versioned and strictly validated before
  protobuf construction.
- Any buffer overflow, malformed record, missing callback, or incomplete flush
  is fatal.
- Simulator `gpu_sim_insn` and `gpgpu_n_tot_w_icount` must equal the newly
  loaded trace exactly. They are not required to equal Option 1B because a
  lighter capture is expected to change timing-dependent spin control flow.

#### Composable A/B feature configuration

The new public configuration is one comma-separated feature set:

```text
LIGHTER_TRACER="compact_wire,prune_operands"
```

Do not expose implementation history such as `opt1a` and `opt1b` as composable
tokens. Option 1B contains Option 1A, so `"opt1a,opt1b"` would be redundant and
would not identify which behavior is independently enabled. Use feature names
whose effects are orthogonal:

| Feature token | Behavior | Required features |
|---|---|---|
| `compact_wire` | Use versioned 104B/360B compact wire records | None |
| `prune_operands` | Remove generic REG/UREG/PRED callbacks with no consumer | `compact_wire` |
| `host_count` | Count received records on the host and remove active-capture device statistic atomics | `compact_wire` |
| `metadata_record` | Use the dedicated approximately 48B metadata-only record | `compact_wire` |
| `sparse_reconstruct` | Capture only dynamic anchors/payloads and reconstruct safe straight-line instructions from static SASS/CFG | `compact_wire,prune_operands` |
| `warp_batch_1k` | Batch serialized records in a 1KB per-warp device buffer | `compact_wire` |
| `warp_batch_2k` | Batch serialized records in a 2KB per-warp device buffer | `compact_wire` |
| `warp_batch_4k` | Batch serialized records in a 4KB per-warp device buffer | `compact_wire` |
| `spin_fusion` | Emit the separately validated combined spin-iteration event | `compact_wire,prune_operands,sparse_reconstruct` |

`warp_batch_1k`, `warp_batch_2k`, and `warp_batch_4k` are mutually exclusive.
The parser trims surrounding whitespace, rejects empty/unknown/duplicate
tokens, validates dependencies, and canonicalizes the set in the table order.
Input order never changes behavior. Missing dependencies are fatal; the parser
does not silently add them.

Configuration equivalents and isolated A/B pairs:

| Purpose | Baseline | Candidate |
|---|---|---|
| Historical Option 1A | `off` | `compact_wire` |
| Historical Option 1B | `compact_wire` | `compact_wire,prune_operands` |
| Stage 1C-A host counting | `compact_wire,prune_operands` | `compact_wire,prune_operands,host_count` |
| Stage 1C-B metadata record | `compact_wire,prune_operands` | `compact_wire,prune_operands,metadata_record` |
| Stage 1C-S sparse reconstruction | `compact_wire,prune_operands` | `compact_wire,prune_operands,sparse_reconstruct` |
| Stage 1C-C 2KB batching example | `compact_wire,prune_operands` | `compact_wire,prune_operands,warp_batch_2k` |
| Stage 1C-F spin fusion | `compact_wire,prune_operands,sparse_reconstruct` | `compact_wire,prune_operands,sparse_reconstruct,spin_fusion` |

Every candidate is compared with its listed baseline captured from the same
binary, git SHA, benchmark arguments, and GPU. A source change requires a new
baseline capture even when an older equivalent artifact exists.

`LIGHTER_TRACER` unset or `LIGHTER_TRACER=off` selects the legacy tracer.
`LIGHTER_TRACER=all` resolves to a separate production feature set containing
only independently validated features. The current production set is
`compact_wire,prune_operands`. Implementing a new feature does not add it to
`all`; promotion occurs only after its A/B and payload/timing gates pass. The
resolved canonical set is persisted, so the meaning of an old artifact does
not depend on a later source version.

`all` is the production publication mode:

- explicit feature lists keep their configuration suffixes for A/B isolation;
- `LIGHTER_TRACER=all` writes the validated result to the original unsuffixed
  `<CUDA>/<benchmark>/<argument-folder>/traces` location;
- publication still refuses to overwrite an existing unsuffixed run, so the
  prior default artifact must be archived or removed explicitly;
- simulation uses the standard CUDA-version trace root, for example
  `-T ./hw_run/traces/device-0/12.8/`;
- `-T` remains required because it selects trace-driven simulation mode.
  Omitting `-T` selects application/binary simulation and does not auto-discover
  the default HW trace directory.

`TRACER_LIGHT_SPIN` and `GATHER_REGISTERS` become deprecated compatibility
inputs:

- if `LIGHTER_TRACER` is set, setting either legacy selector is fatal;
- legacy `TRACER_LIGHT_SPIN=0` maps to `off`;
- legacy `TRACER_LIGHT_SPIN=1,GATHER_REGISTERS=1` maps to `compact_wire`;
- legacy `TRACER_LIGHT_SPIN=1,GATHER_REGISTERS=0` maps to
  `compact_wire,prune_operands`;
- newly generated `run.sh` files use only `LIGHTER_TRACER`.

Any non-`off` `LIGHTER_TRACER` configuration requires `TRIM_SPIN=0`; a trim
request is fatal rather than combined with lighter capture.

`run_hw_trace.py` derives a readable suffix from the canonical feature set and
short configuration hash and appends it to the original argument-folder name.
Generated `run.sh` and the capture manifest persist the requested string,
canonical set, config hash, git SHA, and artifact suffix. Startup emits:

```text
[LIGHTER-TRACER] requested=...
                 canonical=...
                 config_hash=...
                 artifact_suffix=...
```

First Stage 1C-A capture after the user builds the tracer:

```bash
nohup env \
  LIGHTER_TRACER="compact_wire,prune_operands,host_count" \
  ./util/tracer_nvbit/run_hw_trace.py \
  -B flashattn:flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24 \
  -D 0 \
  > /tmp/flashattn.log 2>&1 &
```

Required capture-log extraction:

```bash
grep -a -n -F '[TRACE-COUNT]' /tmp/flashattn.log
grep -a -n -F '[TRACE-WIRE]' /tmp/flashattn.log
grep -a -n -F '[TRACE-BATCH]' /tmp/flashattn.log
```

`[TRACE-BATCH]` is expected only after Stage 1C-C is enabled.

#### Stage 1C-A - remove active-capture statistic atomics

This is the first implementation because it changes no trace payload and no
wire bytes.

Current overhead:

- `common.h` defines managed `total_dynamic_instr_counter` and
  `reported_dynamic_instr_counter`.
- `instrument_inst_compact` performs two globally contended `atomicAdd`
  operations after every successful `ChannelDev::push`.
- The same function performs one `atomicAdd` when `stop_report` suppresses a
  callback.
- `tracer_tool.cu` reads cumulative deltas into the historical
  `stats.csv total_insts,total_reported_insts` columns.

Implementation:

1. Leave legacy `instrument_inst` unchanged.
2. With compact capture and `TRACER_DEVICE_COUNTERS=0`, remove the two atomics
   after a successful push.
3. Count received records per kernel in `recv_thread_fun`, before operand
   callbacks are grouped into one protobuf instruction.
4. Publish the count when the end-of-kernel channel token is consumed. The
   launch callback already waits for `recv_thread_receiving[ctx]` to clear, so
   it can write the published count to `stats.csv`.
5. For an actively captured compact kernel, set both historical columns to the
   received callback-record count. Do not relabel this as a warp-instruction
   count.
6. Preserve or separately account for the suppressed-kernel
   `stop_report=true` behavior; the receiver cannot count records that were
   never pushed. Do not silently change those rows to the active-capture
   meaning.
7. Add a per-kernel line:

   ```text
   [TRACE-COUNT] kernel=... received_records=... stats_total=...
                 stats_reported=... stop_report=0|1
   ```

Acceptance:

- direct-push record count and wire bytes are unchanged except for legitimate
  timing-dependent spin changes;
- for every captured kernel, `received_records == stats_total ==
  stats_reported`;
- no record or protobuf field changes are attributable to the counter path;
- K5/K10 spin and capture wall time are measured before proceeding.

#### Stage 1C-S - sparse semantic capture and static reconstruction

Option 1B still injects one `NO_REGS` callback for ordinary dynamic
instructions that need no runtime payload. The host can reconstruct a safe
subset from the enhanced static SASS and CFG while preserving the final
per-instruction protobuf contract.

The initial direct-capture set is conservative:

- every instruction with a memory address, sync, BAR, or TMA runtime payload;
- every predicated instruction;
- every CFG control-transfer, call, return, reconvergence, and exit site;
- a function/warp entry anchor when needed to make the first span explicit;
- every instruction in a degenerate or ambiguous CFG region.

Only unpredicated, payload-free, straight-line instructions may be omitted from
device instrumentation. For each reconstructed instruction the host emits the
static function/PC and sets `predicate_mask == active_mask` from the enclosing
dynamic span. The implementation classifier must use NVBit CFG/instruction
metadata plus the existing semantic allowlist, not an opcode-prefix heuristic.

Read-only analysis of the saved Option 1B protobuf gives the initial ceiling:

| Kernel | Dynamic instructions | Reconstructable | Fraction |
|---|---:|---:|---:|
| K5 FWD | 14,847,194 | 13,789,067 | 92.87% |
| K10 BWD | 21,007,397 | 18,675,774 | 88.90% |

Across those reconstructable instructions,
`active_mask != predicate_mask` occurs zero times. This is a feasibility
measurement, not yet an acceptance result.

Before a GPU capture, perform an offline proof using the `.opt1b` artifact:

1. Remove exactly the records the proposed classifier would omit.
2. Reconstruct them from the saved static SASS/CFG and retained dynamic
   anchors.
3. Require exact per-warp PC order, function ID, active/predicate masks,
   address/sync/BAR payloads, and serialized protobuf hash.
4. Require exact HGMMA, MUFU, TMA-family, memory, sync, and BAR totals.
5. Abort on an ambiguous path; do not guess or choose a shortest CFG path.

The GPU implementation is attempted only after this offline round trip is
exact. The sparse capture still produces a complete full trace after host
reconstruction; it does not trim, clone, or reorganize dynamic instructions.

#### Stage 1C-B - dedicated metadata-only record

Option 1B is already near one callback per dynamic instruction, but most
ordinary instructions still send the 104B compact prefix even though they need
only execution identity. Use a separate record only when:

```text
per_operand_type == NO_REGS
num_of_injects == 1
mem_type == NONE
```

Define a versioned approximately 48B record containing:

```text
record size/version/flags
CTA x/y/z
CTA-local warp, SM-local warp, SM ID
PC and function ID
active mask and predicate mask
```

The host normalizes it to:

```text
num_of_injects = 1
per_operand_type = NO_REGS
mem_type = NONE
```

No scalar value, descriptor field, width, or lane array is needed for this
record. Retained sync/BAR/TMA scalar callbacks continue to use 104B records;
memory-like callbacks continue to use 360B records.

Required changes:

- add a dedicated flag/record type and size assertion in `common.h`;
- add the `NO_REGS` fast path in `instrument_inst_compact`;
- make `validated_inst_trace_record_size` accept exactly the metadata, compact,
  or memory size allowed by its flags;
- normalize the metadata record before the existing host protobuf path;
- keep the flush token in an existing validated format;
- add `metadata_records` to `[TRACE-WIRE]`.

The absolute lower bound, if all 62.14M Option 1B non-memory records were
metadata-only, would be approximately 3.93GB:

```text
62,139,404 * 48B + 2,636,211 * 360B = 3,931,727,352B
```

The real result will be larger because retained scalar callbacks remain 104B.
Measure the record-class distribution instead of treating 3.93GB as a target.

Acceptance:

- total record count is unchanged except for legitimate spin changes;
- every metadata record becomes exactly one protobuf instruction;
- compact scalar and memory record counts remain internally consistent;
- protobuf/TMA/sync/BAR preservation gates remain clean.

#### Stage 1C-C - per-warp ChannelDev batching

This is the highest-risk transport change. Start only after Stage 1C-A has a
separate measurement and Stage 1C-S has passed its offline round-trip proof.
Stage 1C-B is an independent byte-size experiment, not a batching prerequisite.
First test one `warp_batch_*` feature alone on top of
`compact_wire,prune_operands`; test sparse plus batching only after both
individual features pass.

Design:

1. Allocate the staging area in device memory, not managed memory and not
   shared memory. Managed writes can fault/migrate; shared-memory reservation
   would change occupancy.
2. Give each active warp a byte buffer and cursor. Only the first active lane
   writes the already serialized wire record, so no per-record atomic is
   required within a proven-unique warp slot.
3. Never split a variable-size record across batches. If the next record does
   not fit, push the current contiguous batch once, reset the cursor, then
   append the record.
4. After the application kernel completes, launch a flush kernel that pushes
   every non-empty partial buffer before the existing end-of-kernel token.
5. Preserve record order within each warp. Cross-warp arrival order is not a
   protobuf contract because the receiver stores separate CTA/warp streams.
6. Report:

   ```text
   [TRACE-BATCH] records=... push_calls=... mean_records_per_push=...
                 partial_flushes=... oversized=... overflow=...
   ```

Slot identity is a prerequisite, not an assumption. Prefer physical
`(sm_id, warpid_sm)` only after proving that it uniquely identifies concurrently
resident warps and that slot reuse cannot race with a prior flush. Otherwise
use logical `(blockIdx.x,y,z, warpid_tb)` with an explicit per-launch capacity
bound. Abort if the grid exceeds that bound.

Start with direct push (`0`) versus 1KB, 2KB, and 4KB batches. Record total
device-memory footprint and reject a size that changes application occupancy
or causes allocation pressure. A batch implementation that silently falls
back, drops a partial buffer, or uses a managed high-frequency ring is invalid.

Acceptance:

- `push_calls` falls by the measured batching factor;
- concatenating all batches yields the same validated record stream per warp;
- `records`, wire bytes, protobuf payload, and sidecars are unchanged except
  for legitimate timing-dependent spin;
- overflow and oversized-record counts are zero;
- K5/K10 spin and capture wall time improve or the stage is reverted.

#### Deferred tracer ideas

- Do not start with operand-callback fusion within one instruction. Option 1B
  already measures only 1.017 K5 and 1.031 K10 callbacks per warp instruction,
  so that fusion has a small theoretical reduction while sync/TMA grouping
  risk is high. This does not apply to Stage 1C-S, which removes callbacks for
  reconstructable dynamic instructions, or to the separately gated
  `spin_fusion` feature.
- Do not compress memory lane addresses first. Memory records account for only
  about 0.95GB of the 7.41GB Option 1B wire traffic, and compression work in the
  injected callback can increase observer effect.
- Do not combine any of these stages in one first measurement. Keep each stage
  independently disableable until its capture result passes.

#### Capture acceptance matrix

| Stage | Record count | Wire bytes | `ChannelDev::push` count | Primary evidence |
|---|---|---|---|---|
| 1C-A atomics | Same class mix | Same bytes | Same | Lower contention / wall time / spin |
| 1C-S sparse | Much lower device records; same reconstructed instructions | Much lower | Much lower | Offline exact round trip, reconstructed protobuf parity |
| 1C-B metadata | Same callbacks | Lower | Same | `metadata_records`, lower bytes |
| 1C-C batching | Same callbacks | Same serialized bytes | Much lower | `mean_records_per_push` |

For every stage:

- inspect `[TRACE-WIRE]`, `[TRACE-COUNT]`, and when applicable
  `[TRACE-BATCH]`;
- compare per-PC K5/K10 NANOSLEEP with HW targets 8,461 / 7,860;
- preserve HGMMA, MUFU, TMA-family, memory, BAR, and non-spin control-flow
  work;
- require TMA resolver/base-map generation with no missing operands;
- run the simulator functional gate only after the capture gates pass;
- in that simulator run, compare the simulator against the new trace contract,
  not against Option 1B's timing-dependent instruction total.

Suggested first next-session scope:

```text
implement Stage 1C-A only
static/source review
user builds once
capture "compact_wire,prune_operands" baseline
capture "compact_wire,prune_operands,host_count" candidate with the same binary
evaluate TRACE-COUNT + TRACE-WIRE + spin
then perform the offline Stage 1C-S round-trip proof
```

#### Independent simulator pacing track

Do not wait for perfect capture before collecting observe-only simulator
evidence, but do not alter scheduling or barrier eligibility yet.

Existing implementation anchors:

- PHASECHK/TRYWAIT execution and mbarrier state:
  `remodeling/sm.cc` around `2142-2432`;
- Producer ready/retry probe:
  `remodeling/sm.cc` around `847-880`;
- TMA completion credit:
  `SM::notify_tma_completion` and
  `remodeling/tma_unit_sm.cc` around `1024-1041`;
- warp completion:
  `remodeling/sm.cc` around `1282-1303`.

Add one bounded per-CTA/per-barrier-phase lifecycle summary containing:

```text
first and last Producer PHASECHK cycle
first modeled-ready cycle and ready cause
Consumer ARRIVE/EXCH cycle
Producer TMA issue cycle
TMA completion-credit cycle
Producer EXIT and maximum Consumer EXIT cycle
```

The probe must only observe existing state. Probe on/off must produce
bit-identical `gpu_sim_insn`, `gpgpu_n_tot_w_icount`, and total cycles. Use the
result to decide whether early Producer completion originates in scheduler
pacing, mbarrier phase/arrival timing, TMA completion timing, or EXIT handling.
Do not add modeled-ready blocking until this causal ordering is known.

## 8. Fallback Option 2 - dual-capture sync-timeline reorganization

### Details

Option 2 is parked as a fallback rather than the immediate next implementation.
If activated, it uses the Option 1B trace for complete replay data and a
second, ultra-light run for HW synchronization timing. It does not use modeled
mbarrier-ready as a release condition.

#### Directory contract

One `run_hw_trace.py` invocation creates all three directories:

```text
<run_dir>/
  traces_fullrun/
    dynamic_trace.pb
    threadblocks/
    extra_info/
    stats.csv
    capture_manifest.json

  traces_ultralight/
    sync_oracle_events.bin
    sync_oracle_summary.csv
    oracle_manifest.json

  traces/
    dynamic_trace.pb
    threadblocks/
    extra_info/
      spin_timing_oracle.json
      spin_reorg_manifest.json
    stats.csv
```

Directory ownership:

- `traces_fullrun` is an immutable Option 1B complete-data source.
- `traces_ultralight` is an immutable timing-oracle bundle, not a simulator
  dynamic trace.
- `traces` is the reorganized simulator input and may be regenerated from the
  two source directories.
- Merge writes to `traces.merge.tmp`, validates the complete result, and only
  then publishes it as `traces`. If `traces` already exists, the publisher
  first renames it to `traces.previous`, renames the validated temporary
  directory to `traces`, and removes `traces.previous` only after publication
  succeeds.
- Merge never modifies either source directory.

Each manifest records:

```text
git SHA
benchmark arguments
GPU/CUDA version
kernel launch sequence
grid/block dimensions
CUBIN hash
per-function SASS hash
tracer mode and environment
event-buffer capacity/overflow
```

The full child run uses:

```text
LIGHTER_TRACER=compact_wire,prune_operands
TRIM_SPIN=0
```

The ultra-light child run uses a new dedicated oracle mode. It disables normal
dynamic protobuf capture, TMA descriptor dumping, and `ChannelDev`.

#### Ultra-light event contract

The oracle instruments six SASS semantic families plus one CTA timeline probe.
These are semantic families, not six literal opcode strings.

| Event family | Actual coverage | Purpose |
|---|---|---|
| `CTA_BEGIN` | One first-event timestamp per CTA | CTA-relative HW timeline |
| Barrier arrival | `SYNCS.ARRIVE.TRANS64*`, `SYNCS.EXCH.64` | Actual barrier-ready cause |
| Wait check | `SYNCS.PHASECHK.TRANS64*` | Barrier key/phase and burst start |
| Sleep | `NANOSLEEP.SYNCS` | Executed repeat count |
| Back edge | Only validated spin-loop `BRA` sites | Repeat versus final exit |
| Async issue anchor | `UTMALDG`, `UTMASTG`, `UTMAPF`, `UTMACCTL`, `UTMACMDFLUSH`, `UBLKCP`, `UBLKRED`, `UTMAREDG` | Producer pacing and logical-work signature |
| Warp exit | Predicated SASS `EXIT` | Producer/Consumer lifetime |

`CTA_BEGIN` is inserted at the kernel entry site. Multiple warps may execute
the probe, so the per-CTA slot publishes the minimum `globaltimer` timestamp
with a one-time atomic update. It does not instrument the rest of the entry
basic block.

No HGMMA, MUFU, ALU, LDGSTS, or generic memory instruction is instrumented.
Those instructions and their payloads come from `traces_fullrun`.

The ultra-light device record contains:

```text
kernel launch ordinal
stable function/SASS hash
logical block_id (blockIdx.x,y,z) and CTA-local warp
event kind and relative PC
normalized barrier offset
semantic/phase raw value
guard-predicate outcome
globaltimer timestamp
TMA site and normalized logical-address signature
```

The oracle also emits one summary per spin burst:

```text
TRYWAIT loop ID
occurrence index
burst start timestamp
burst final-exit timestamp
executed NANOSLEEP count
barrier offset/phase
static loop-descriptor ID and expected fallthrough
next TMA-family anchor
```

Records are written to a preallocated per-CTA/per-warp device buffer. There is
no per-event host channel push. The host copies the buffer once after kernel
completion. Any buffer overflow is fatal.

Use `globaltimer`, not SM-local `clock64`, for cross-SM timestamps. Both absolute
timestamps and CTA-relative offsets are stored.

#### Static loop descriptor

Loop sites are discovered from the uninstrumented CUBIN/SASS rather than a
hardcoded PC table:

```text
function SASS hash
TRYWAIT PC
NANOSLEEP PC
PHASECHK PC
back-edge BRA PC and target
fallthrough PC
optional preheader PC
barrier operand index
semantic/phase operand index
```

The current strict four-consecutive-PC matcher is insufficient:

- K5 `0xacc0` and K10 `0xb4f0` can include a preheader between iterations.
- Some calibrated loops execute in warps other than the single selected
  Producer warp.
- Option 2 therefore scans every warp and uses a control-flow state machine,
  not `TRYWAIT,+0x10,+0x20,+0x30` alone.

### Mapping

Mapping has four stages.

#### Stage 1 - launch and static identity

The two runs must have identical:

```text
kernel launch count/order
kernel name and grid/block dimensions
CUBIN hash
function SASS hashes
static loop descriptors
```

Capture-local `unique_function_id` is not a stable cross-run key. The mapper
uses function SASS hash plus relative PC and records the full-to-ultra function
ID translation in the merge manifest.

#### Stage 2 - full-trace burst extraction

For every CTA and every warp, the mapper walks the dynamic PC stream and
extracts maximal spin bursts. A burst begins at a validated TRYWAIT site and
ends at the final back-edge BRA whose predicate does not repeat the loop.

Each full burst records:

```text
kernel/CTA/warp
loop descriptor ID
occurrence ordinal
repeat-iteration ranges
final-exit iteration range
barrier offset/phase from dynamic sync payload
previous/fallthrough anchor PCs
next TMA-family site and normalized address signature
```

#### Stage 3 - oracle burst extraction

The device-side oracle finalizes a burst when the validated back-edge BRA does
not repeat. It assigns the occurrence ordinal within
`(kernel, CTA, warp, loop descriptor)`.

#### Stage 4 - strict join

Primary join key:

```text
(kernel launch ordinal,
 function SASS hash,
 logical block_id (blockIdx.x, blockIdx.y, blockIdx.z),   # NOT a physical CTA/SM slot
 warp ID,
 TRYWAIT relative PC,
 occurrence ordinal)
```

The join key is the logical `blockIdx`, not a physical CTA/SM slot. NVBit records
each threadblock protobuf under its logical `blockIdx`, and the simulator replays
one protobuf per logical `blockIdx` (`trace_parser.cc:509-514`). Under a dynamic
tile scheduler the same logical `blockIdx` may still change which tile it
processes across two runs; see the empirical result below for why this key holds
for K6 but not for K5.

Required equality checks:

```text
normalized barrier offset
phase/semantic raw
next TMA-family site
logical TMA address signature, when available
```

The full trace's previous/fallthrough anchors are checked against the static
loop descriptor. They are not compared with a dynamic ultra-light event:
generic non-spin instructions are deliberately not instrumented in the
ultra-light run.

There is no fuzzy match in the first implementation. Missing, duplicated, or
reordered bursts abort the merge.

Dynamic tile scheduling is the largest mapping risk, and it is confirmed present
in this workload (see "Empirical scheduler determinism" below). The relevant
question is not whether the same *physical* CTA gets a different logical tile
(the trace is keyed by logical `blockIdx`, so physical placement is irrelevant),
but whether the same *logical* `blockIdx` processes the same tile across two
runs. Measurement shows this holds for K6 but fails for K5. The first
implementation therefore joins on logical `blockIdx`, cross-checks each burst
with the TMA site plus normalized logical-address signature, and aborts on any
mismatch rather than silently remapping. Logical-tile remapping is a separate
follow-up only if the mismatch is proven deterministic and one-to-one.

#### Empirical scheduler determinism (measured)

The two existing compact captures (`.opt1a` all-operand vs Option 1B pruned)
are physically different runs with different capture overhead and therefore
different spin counts. Decoding every threadblock protobuf and comparing, per
logical `blockIdx` and warp, the instruction stream with differing-PC (spin/poll)
instructions removed gives:

| Kernel | Scheduler (from mangled name) | block_ids | spin-removed stream mismatches | real-mem (addresses>0 & sync==0) grand total | block_ids with differing real-mem | Verdict |
|---|---|---:|---:|---|---:|---|
| K6 (FWD) | `DynamicPersistentTileScheduler` | 384 | 32 / 6144 | 111,997 == 111,997 (conserved) | 2 / 384 (one tile pair swapped: `(0,265,0)`↔`(0,270,0)`) | logical keying holds for 382/384 |
| K5 (BWD) | `SingleTileBwdLPTScheduler` (LPT work-stealing) | 132 | 533 / 2112 | 79,150 → 78,706 | 131 / 132 (symmetric ±156 shifts, e.g. block 7↔47) | logical keying FAILS |

Consequences for Option 2:

- K6 tiles are almost perfectly pinned to logical `blockIdx`; the strict join
  succeeds for 382/384 CTAs and only two swapped tiles need handling.
- K5 uses a longest-processing-time work-stealing scheduler: nearly every
  logical `blockIdx` processes a different tile set across runs, so the strict
  join aborts for almost every K5 CTA even with logical-`blockIdx` keying.
- The `traces_fullrun` (Option 1B) and `traces_ultralight` runs are two more
  physically different runs and will exhibit the same K5 non-determinism.
  Option 2 as specified therefore cannot merge K5 without a scheduler-aware
  step. This is an open problem, tracked in Section 8.x below, and must be
  resolved before implementing the K5 merge path.

### Reorganization

Each full-trace burst is split into:

```text
repeat iteration 0
repeat iteration 1
...
repeat iteration N-1
final exit iteration
```

The final iteration is identified by the back-edge BRA predicate outcome, not
only by PC order.

For an oracle target of `P` executed repeat iterations:

1. Keep the first `P` repeat iterations from the Option 1B full trace.
2. Keep the original final-exit iteration.
3. Delete only excess repeat iterations between them.
4. Preserve every retained instruction message byte-for-byte.
5. Preserve every non-spin instruction and every other warp byte-for-byte.
6. Attach the matching ultra-light event timestamp to every retained
   TRYWAIT/NANOSLEEP/PHASECHK/BRA instruction.
7. Attach the matching timestamp to the next validated TMA-family anchor.
8. Write all changed/deleted indices, timing joins, and hashes to
   `spin_reorg_manifest.json`.

`P` is the number of predicated-executed `NANOSLEEP.SYNCS` repeat iterations.
The retained dynamic loop therefore contains `P` repeat bodies plus one final
exit body. If `P == 0`, only the original final-exit body remains.

If `P` is greater than the repeat count available in the full trace, merge
fails. The initial implementation does not clone or synthesize a missing
iteration because predicate masks, barrier payloads, and phase may be wrong.

NCU per-PC NANOSLEEP counts remain an independent aggregate gate. If the
ultra-light per-PC total differs materially from NCU, the oracle run is
rejected. A later calibrated mode may allocate the exact NCU total among
bursts using ultra-light counts and largest-remainder rounding, but this is not
the initial path.

The merged `dynamic_trace.pb` and all non-spin `extra_info` come from
`traces_fullrun`. The merged `stats.csv` must not reuse callback counts as
instruction counts; reorganization emits separate capture and final-trace
statistics.

#### Instruction-count invariants (two distinct counts)

Reorganization and pacing touch two different counts that must not be conflated
with the project's bit-identical `gpu_sim_insn` rule.

- Trace-level count (changes by construction): the merged trace intentionally
  contains fewer spin records than `traces_fullrun`, because excess repeat
  iterations are deleted. `gpu_sim_insn` and `gpgpu_n_tot_w_icount` are
  accumulated one-to-one from retired records (`sm.cc:2942`, `sm.cc:2949`), so
  they drop by exactly the deleted spin records. The Section 5 fixed totals
  (14,847,194 warp-insts, 450,627,804 active-lane insts, ...) apply to the
  Option 1B full trace and are NOT the expected totals for the merged Option 2
  trace. The expected merged totals must instead be predicted from the manifest:

  ```text
  gpu_sim_insn(merged) ==
      gpu_sim_insn(fullrun) - sum(active lanes of every deleted spin record)
  gpgpu_n_tot_w_icount(merged) ==
      gpgpu_n_tot_w_icount(fullrun) - (number of deleted spin records)
  ```

  The `spin_reorg_manifest.json` deleted-index list must make both right-hand
  sides exactly computable, and the merged run is checked against them.

- Sidecar-level count (must be invariant): given one fixed merged trace, turning
  oracle pacing on or off must not change `gpu_sim_insn` or
  `gpgpu_n_tot_w_icount`. Pacing only delays eligibility; it never deletes,
  clones, or ghost-retires a record and it consumes the real issue slot. This
  pacing-on-equals-pacing-off equality is the constraint-compliant analogue of
  the project's bit-identical rule and is a required simulator gate (added
  below).

#### Simulator timing semantics

The ultra-light run records a CTA start timestamp and CTA-relative timestamps
for every retained spin event and TMA anchor. The simulator converts each HW
event timestamp to an earliest issue cycle:

```text
HW_CTA_relative_event_ns  = event_globaltimer_ns - CTA_begin_globaltimer_ns
HW_CTA_relative_event_cycle = round(HW_CTA_relative_event_ns * core_clock_GHz)

target_issue_cycle(event) =
    simulated_CTA_start_cycle + HW_CTA_relative_event_cycle
```

The two operands are in different clock domains and must be reconciled
explicitly:

- `globaltimer` is a nanosecond wall clock (fixed ~1 GHz), not core cycles;
- `simulated_CTA_start_cycle` is an SM core cycle (H100 core ~1.755 GHz);
- `core_clock_GHz` is the modeled core frequency, recorded in the merge manifest
  and the sidecar, not silently hardcoded;
- the conversion factor is a modeling assumption, so the raw `globaltimer` ns
  and the factor are both stored, and CTA/warp lifetime is validated against HW
  after conversion (see the timestamp-conversion risk row).

For each timing-annotated instruction:

- if current cycle is earlier than its target, the warp is ineligible until
  that target;
- if current cycle is already later, the instruction issues normally without
  an additional stall;
- the instruction executes through the normal issue/FU/scoreboard path and
  consumes its real issue slot;
- modeled mbarrier-ready is sampled but never gates oracle release.

This distributes the HW wait across the actual retained polls instead of
placing one large stall at burst entry. It preserves poll issue-slot
contention, warp/thread instruction statistics, normal sync reads, and the
final-exit instruction. There is no zero-cycle ghost-retirement path.

Required runtime log:

```text
[SPINORACLE]
kernel/CTA/warp/loop/occurrence
target_release_cycle
actual_release_cycle
model_ready_at_release
ready_delta
retained_repeat_iterations
paced_instruction_count
late_arrival_cycles
```

The oracle can delay an early simulated Producer but cannot accelerate a warp
that reaches the burst after the HW target. `late_arrival_cycles` quantifies
that limitation. A large late-arrival population means the residual is outside
the spin-pacing model.

#### Implementation order

0. Per-kernel scheduler determinism gate FIRST. Using the two existing captures
   (or any two physically different runs), verify per logical `blockIdx` that the
   spin-removed instruction stream matches. This is already measured: K6 passes
   (382/384), K5 fails (131/132 differ). Only kernels that pass this gate enter
   the strict-merge path; failing kernels are routed to the open-problem path and
   are not merged by the initial implementation.
1. Implement directory orchestration and raw ultra-light capture only.
2. Capture twice and validate oracle overhead/counts against NCU.
3. Implement the all-warp burst extractor and strict mapping dry run.
4. Proceed only if logical-`blockIdx` mapping coverage is 100% for that kernel.
5. Implement reorganization and byte/hash validation without simulator timing.
6. Add sidecar loading and per-event earliest-issue pacing last.

This order prevents simulator work from being built on an oracle or mapping
scheme that does not survive real FA3 dynamic tile scheduling.

### Goal

Preserve the Option 1B complete-data contract while replacing capture-induced
spin count and Producer pacing with a low-perturbation HW synchronization
timeline.

### Why this fallback remains

Option 1B proves that a lightweight full trace can preserve the complete replay
payload, but it does not yet recover HW spin or Producer/Consumer lifetime.
Further single-run tracer reduction and simulator pacing diagnosis remain the
primary path. If that path reaches an observer-effect floor, Option 2 remains a
fallback that must preserve both poll work and CTA-relative release timing,
unlike the earlier aggregate count-only trim.

### Measurement

Ultra-light capture gates:

- per-PC NANOSLEEP totals agree with NCU;
- all expected CTA/warp EXIT records are present;
- no oracle-buffer overflow;
- Producer/Consumer lifetime agrees with HW;
- event count and capture time are stable across repeated runs.

Mapping/reorganization gates:

- 100% burst-key coverage;
- zero duplicate or ambiguous joins;
- exact barrier offset/phase and anchor agreement;
- non-spin serialized hash is identical before/after;
- HGMMA, MUFU, TMA, memory, sync, and BAR work is unchanged;
- only validated excess spin-repeat records are deleted;
- every retained spin event and TMA anchor has exactly one timing join.

Simulator gates:

- clean exit with no deadlock;
- merged-trace `gpu_sim_insn` and `gpgpu_n_tot_w_icount` equal the manifest
  prediction (fullrun total minus deleted spin records);
- pacing-on and pacing-off produce identical `gpu_sim_insn` and
  `gpgpu_n_tot_w_icount` on the same merged trace (sidecar is timing-only);
- TMA Phase2 coverage remains unchanged;
- `[SPINORACLE]` has zero missing records;
- Producer-only tail and total cycles move toward HW;
- modeled-ready mismatch and late-arrival distributions are reported.

### Risks and mitigations

| Risk | Failure mode | Mitigation |
|---|---|---|
| Ultra-light observer effect | Sync callbacks change poll count/release timestamp | No ChannelDev; device slots; compare per-PC counts with NCU; repeat run |
| Dynamic tile non-determinism (CONFIRMED) | Same logical `blockIdx` processes a different tile across runs; K5 fails, K6 mostly holds (measured) | Join on logical `blockIdx`; TMA logical signature; strict abort; per-kernel go/no-go; K5 needs a scheduler-aware step (open problem) |
| Non-canonical loop shape | Four-PC matcher misses preheader/interleaving | Static CFG descriptor plus all-warp state machine |
| Full trace under-spins | Oracle requires iterations absent from full | Fatal merge; no synthetic cloning in initial version |
| Timestamp conversion | `globaltimer` ns vs core-cycle domain mismatch or wrong `core_clock_GHz` | Store raw ns and `core_clock_GHz`; convert explicitly; validate CTA/warp lifetime |
| Timestamp resolution vs per-poll pacing | `globaltimer` tick coarser than spin period collapses adjacent polls to one timestamp, degrading per-poll pacing to per-burst | Measure `globaltimer` tick on target; if coarser than burst period use `clock64` for intra-CTA relative timing and `globaltimer` only for cross-SM `CTA_BEGIN` anchor |
| Oracle device-buffer memory kind | High-frequency stores to managed memory fault/migrate and distort timing | Device-resident buffer (`cudaMalloc`), never `cuMemAllocManaged`; single post-kernel copy |
| Device-buffer overflow / sizing | Fixed capacity sized to HW spin (~8k) overflows under still-inflated bring-up spin | Size to observed Option 1B max burst length plus headroom; overflow counter is fatal, not silent truncation |
| Simulator arrives late | Oracle can delay but cannot accelerate | Report `late_arrival_cycles`; reject if dominant |
| Modeled mbarrier divergence | Ready state differs at oracle release | Diagnostic only; never block release; report ready delta |
| Event publication race | Burst/TMA patch records are partially visible | Per-warp sequence numbers; publish-valid last; kernel sync before host copy |
| NCU count semantics | Callback count and predicated-executed NANOSLEEP differ | Count only nonzero guard masks; use identical per-PC aggregation |
| TMA signature overhead | Capturing logical coordinates perturbs Producer timing | First predicated lane/uniform operands only; compare oracle with/without signature |
| Poll pacing distortion | One bulk wait removes HW poll issue-slot contention | Earliest-issue timestamp per retained poll; normal pipeline execution |
| Cross-run launch drift | Kernel ordinal/function identity mismatch | Launch/CUBIN/SASS manifest equality required |
| Allocation-address drift | Raw pointer comparison fails across runs | Compare normalized shared offsets and logical TMA signatures |
| Partial output publication | Failed merge leaves simulator-visible trace | Temporary output plus atomic rename after all gates |

### Decision after the run

- If ultra-light counts disagree with NCU, reduce oracle instrumentation before
  implementing merge.
- If strict logical-`blockIdx` mapping fails for a kernel (measured: K5),
  diagnose tile assignment before considering logical remapping; do not merge
  that kernel in the initial implementation.
- If mapping and work gates pass but simulated Producers are mostly late, do
  not tune spin timing further; the bottleneck is elsewhere in the simulator.
- If oracle pacing restores role lifetime and lowers cycle error, adopt
  `traces` as the simulator input and retain both source directories for audit.

### Open problems (resolve one at a time, in order)

These are the unresolved items surfaced by review and the empirical scheduler
measurement. Each is independently actionable and gated before the dependent
implementation step.

1. K5 dynamic-tile join (blocks Stage 4 for K5). LPT work-stealing makes logical
   `blockIdx` unstable across runs (131/132 blocks differ). Decide one of:
   (a) merge only K6 in the initial version and treat K5 as out of scope;
   (b) build a tile-identity join (match on TMA logical-address signature and
   per-tile work fingerprint instead of `blockIdx`) and prove it is
   deterministic one-to-one before use. Do not implement K5 pacing until this
   is decided.
2. `globaltimer` resolution (blocks per-poll pacing design). Measure the tick on
   the target H100. If it is coarser than the spin-burst period, adopt the
   `clock64` intra-CTA + `globaltimer` cross-SM hybrid; otherwise keep
   `globaltimer` everywhere. Record the measured tick in the manifest.
3. Clock-domain conversion (blocks simulator pacing). Fix `core_clock_GHz`,
   store it and raw ns in the sidecar, and validate converted CTA/warp lifetime
   against HW before enabling eligibility gating.
4. Oracle buffer memory kind and sizing (blocks ultra-light capture). Choose a
   device-resident (`cudaMalloc`) per-CTA/per-warp buffer, size it from the
   observed Option 1B max burst length plus headroom, and make overflow fatal.
   The existing `warp_lifetime_capture_t` CTA×warp indexing is the template, but
   it is managed memory and must not be copied verbatim for high-frequency
   stores.
5. Two-count bookkeeping (blocks acceptance gates). Emit the manifest prediction
   for merged `gpu_sim_insn`/`gpgpu_n_tot_w_icount` and add the pacing-on ==
   pacing-off equality check to the simulator gates.

### Expected deliverable

```text
per-kernel scheduler-determinism gate + report
ultra-light device capture implementation
raw oracle event format and converter
static loop-descriptor generator
strict burst matcher and dry-run report
timing-aware trace reorganizer
spin timing sidecar loader
simulator per-event earliest-issue pacing path
end-to-end manifest and validation report
```

## 9. Status

- Compact wire layout implemented.
- `LIGHTER_TRACER=off|<comma-separated features>` parsing, canonicalization,
  dependency validation, legacy-input mapping, and unimplemented-feature
  rejection are implemented in both `run_hw_trace.py` and the NVBit tool.
- Configuration parser validation passed with Python syntax/config checks and a
  host-only C++17 extraction/smoke test of the NVBit-side parser.
- The renamed configuration has not yet been built with CUDA/NVBit or exercised
  in a GPU capture; those remain user-run validation gates.
- `compact_wire` selects compact full capture; `prune_operands` selects
  semantic callback pruning.
- Lighter runs use configuration-suffixed argument folders containing a
  `lighter_tracer_config.json` manifest; existing lighter run directories are
  not overwritten.
- `run_simulations.py -T <selected-run-directory>` accepts an exact
  benchmark/argument run directory after validating its benchmark and argument
  prefix, while preserving legacy trace-root discovery.
- `LIGHTER_TRACER=all` currently resolves to the validated
  `compact_wire,prune_operands` set and publishes to the unsuffixed default
  argument directory for use with the standard CUDA-version `-T` root.
- Device non-memory scalar path implemented.
- Host mixed-record parser and validation implemented.
- TMA absent-value sentinel filtering implemented.
- Semantic-role callback pruning implemented for `prune_operands`.
- Memory, sync semantic, BAR runtime, and all TMA-family operand callbacks are
  explicitly retained.
- Omitting `prune_operands` restores the all-operand compact fallback.
- Generated `run.sh` persists only the canonical `LIGHTER_TRACER` selector.
- `[TRACE-WIRE]` accounting implemented.
- `TRIM_SPIN=1` mutual-exclusion guard implemented.
- Static source and diff checks passed.
- Build and compact GPU capture completed.
- Simulator-visible protobuf payload preservation passed.
- `[TRACE-WIRE]` measured 20.75GB compact traffic versus 136.55GB legacy
  equivalent, an 84.81% reduction.
- Compact spin reduction measured at 43.4% for K5 and 42.0% for K10.
- Option 1A HW timing-fidelity gate failed at 27.2x K5 and 50.0x K10 residual
  spin.
- Capture wall time remains to be collected from the GPU-server log.
- Option 1B build and GPU capture completed.
- Option 1B callback reduction passed at 66.2% workload-wide.
- Option 1B wire traffic is 7.41GB, 64.3% below Option 1A and 94.6% below
  the original all-operand/fixed-record equivalent.
- Option 1B replay/TMA/sync payload parity passed.
- Option 1B captured-trace spin is 12.1x HW for K5 and 35.4x HW for K10;
  simulator runtime NANOSLEEP is 12.1x and 35.5x, respectively.
- Option 1B simulator functional run completed: K5 83,308 cycles (23.1% error)
  and K10 136,898 cycles (3.0% error).
- Option 1B payload/functional gate passed: exact warp/active-lane instruction
  totals, exact memory/sync/BAR/HGMMA/MUFU/TMA contract totals, TMA
  `unresolved=0`, BAR leak=0, fatal=0, and clean exit.
- Option 1B timing-fidelity gate failed: Producer completes 37.9K cycles before
  Consumer in K5 and 12.0K before Consumer in K10; K5 HW instead has an
  approximately +8.0K Producer tail.
- Primary next direction is further single-run tracer lightening plus an
  independent simulator Producer/Consumer pacing investigation.
- The next-session handoff now defines the composable `LIGHTER_TRACER` feature
  set, strict dependency/conflict validation, isolated A/B configurations,
  staged Option 1C-A/S/B/C implementations, the reserved `spin_fusion` and
  `all` configurations, artifact-suffix isolation, acceptance gates, and the
  separate observe-only simulator lifecycle probe. The configuration layer is
  implemented; the new Option 1C optimization features are not implemented yet.
- Option 2 per-burst ultra-light capture and reorganization remain unimplemented
  and are parked as a fallback.
- FA3 tile scheduler identified from mangled kernel names and SASS: K6 (FWD) is
  `DynamicPersistentTileScheduler` (grid 384) with a `VOTEU/ATOMG.E.ADD/SHFL`
  work-counter; K5 (BWD) is `SingleTileBwdLPTScheduler` (grid 132 = SM count).
  Both are dynamic, not static.
- Scheduler-determinism measured on the two existing captures (`.opt1a` vs
  Option 1B): K6 logical-`blockIdx` keying holds (382/384; real-mem grand total
  conserved 111,997; one tile pair swapped), K5 fails (131/132 differ; symmetric
  work redistribution). Option 2 join key changed to logical `blockIdx`; K5
  merge is an open problem.
