# WaitBlock Post-Processing: Classify One of Two Runtime Strategies

**Work type:** Offline trace post-processing (static SASS + compact dynamic
trace classification). This document is the authoritative design for the
sidecar generator only.

**Parent index:** [WAIT_BLOCK.md](WAIT_BLOCK.md). Runtime consumption of the
classification is designed in
[WAIT_BLOCK_SIMULATOR.md](WAIT_BLOCK_SIMULATOR.md). Shared terminology and hard
invariants are defined once in the parent index and are not repeated here.

Post-processing reads static SASS metadata and compact dynamic traces. It does
not mutate the trace. It classifies each WaitBlock as one of exactly two runtime
strategies (`INJECT_MISSING_WAIT`, `FOLD_CAPTURED_POLL`) or leaves it as inert
metadata.

## 1. Purpose

Post-processing reads static SASS metadata and compact dynamic traces. It does
not mutate the trace.

Its runtime-facing job is limited to:

1. Detect a valid missing-wait gate and emit `INJECT_MISSING_WAIT`.
2. Detect a valid captured polling loop and emit `FOLD_CAPTURED_POLL`.
3. Report which ready-only gates have observed partial effective masks so the
   runtime guard is visible before simulation.
4. Keep other discovered WaitBlocks as metadata without enabling a runtime
   action.

Admission as a WaitBlock is not itself a runtime strategy.

## 2. Input and identity

Required inputs:

- `enhanced_execution_info.json`: static PC, opcode, predicate, operands,
  direct branch target, and dependency metadata;
- `sync_operand_resolver.json`: normalized sync kind and barrier operand;
- compact `.pb` traces: dynamic PC order, active mask, predicate mask, runtime
  barrier address, and runtime wait state;
- existing spin-region records: proven captured poll loops and their local exit
  PCs.

Every runtime-enabled record is keyed by:

```text
(trace_kernel_id, unique_function_id, action_pc)
```

`action_pc` is:

- `entry_probe_pc` for `INJECT_MISSING_WAIT`;
- `poll_probe_pc` for `FOLD_CAPTURED_POLL`.

Records from different kernel launches must not overwrite each other even when
they share a function ID and PC.

## 3. Classification order

Classification is deliberately ordered.

```text
if the WaitBlock owns a captured, foldable poll loop:
    strategy = FOLD_CAPTURED_POLL

else if every validated visit is ready-only
     and the runtime wait target/state are decodable
     and the gate condition is proven:
    strategy = INJECT_MISSING_WAIT

else:
    no runtime strategy
```

The first rule prevents a Type-2/captured-poll block from being converted into
an entry hold.

Current intended action sets:

```text
22 ready-only records     -> INJECT_MISSING_WAIT
14 captured-poll records  -> FOLD_CAPTURED_POLL
```

The earlier "8 missing-wait targets" count referred only to the initially
identified compound Type-1 subset. It is not the total ready-only count. Full
v4 validation finds 22 ready-only records. The additional K10 `0xa810` gate was
excluded by v3 because its uncaptured wait arm contains local memory; v4
correctly admits the entry because `INJECT_MISSING_WAIT` neither executes nor
skips that wait arm.

The v4 generator remains authoritative for the final strategy and visit counts
and must print:

```text
ready_only_records
captured_poll_records
ready_only_all_live_visits
ready_only_partial_visits
ready_only_records_with_partial_visits
```

## 4. Strategy A: `INJECT_MISSING_WAIT`

### 4.1 Required trace shape

The validated dynamic trace contains only the ready arm:

```text
entry probe
-> controlling branch
-> captured ready continuation
```

The not-ready behavior is not dynamically present. This is the missing wait the
simulator must inject when modeled readiness disagrees with the capture.

### 4.2 Required static proof

The detector proves:

- `entry_probe_pc` is a wait-class probe;
- a direct predicate-controlled branch consumes its result;
- the captured path is the ready arm;
- all validated visits take the ready arm;
- the barrier-address and wait-state operands are resolvable;
- any second static probe associated with the same gate tests an equivalent
  condition;
- no unresolved indirect control is involved in arm selection.

The detector does not need to make the uncaptured wait arm cursor-consumable.
Runtime never enters or advances through that arm.

### 4.3 Required dynamic proof

For every validated visit:

```text
effective_mask = active_mask & predicate_mask
```

Post-processing records the effective entry mask and the corresponding live
warp mask for every visit. A partial mask does not reject the ready-only record:
when the simulator is also ready, no wait action is required and normal replay
is correct.

A partial mask becomes unsupported only when runtime observes modeled not-ready.
The simulator scheduler can wait-hold a warp, not an independent lane group, so
that dynamic combination must abort instead of silently over-holding the warp.

The dynamic barrier address and normalized wait state must be stable for the
gate identity used by runtime.

### 4.4 Runtime metadata

```text
runtime_strategy = INJECT_MISSING_WAIT
entry_probe_pc
entry_branch_pc
captured_ready_pc
canonical_wait_condition
observed_effective_masks
observed_live_masks
partial_modeled_miss_action = ABORT
```

`captured_ready_pc` is diagnostic metadata. Runtime does not jump to it.

## 5. Strategy B: `FOLD_CAPTURED_POLL`

### 5.1 Required trace shape

The trace contains a repeated polling loop:

```text
poll_loop_head
-> wait/backoff instructions
-> poll_probe
-> loopback or poll_loop_exit
```

This is the behavior already modeled successfully by the original spin-fold
implementation.

### 5.2 Required static proof

Reuse the existing spin-region requirements:

- the loopback is a direct branch to `poll_loop_head_pc`;
- `poll_probe_pc` defines the exact predicate consumed by the loopback;
- the loopback uses the not-ready polarity, targets `poll_loop_head_pc`, and
  falls through to `poll_loop_exit_pc`;
- every wait in the skipped loop uses the same normalized barrier operand,
  wait-state operand, and destination predicate;
- the body contains only reviewed wait/control/backoff/setup instructions;
- the body contains no semantic, memory, reconvergence, dependency, call,
  return, or unknown operation;
- an `IMAD.U32` or `IMAD.MOV.U32` setup is allowed only when it is an
  idempotent zero-product register copy;
- dependency barriers are self-tied across the complete skip set, including the
  loopback branch;
- `poll_loop_exit_pc` is the first instruction after the loopback and exists in
  the trace.

The exact runtime allow set is:

```text
poll_loop_body_pcs U {poll_loopback_pc}
```

It does not contain `poll_loop_exit_pc` or semantic convergence.

### 5.3 Required dynamic proof

The dynamic trace must show repeated loop visits and a reachable local exit.

Post-processing verifies:

- every active wait occurrence has a runtime payload;
- active waits in one loop iteration agree on runtime barrier address and wait
  state;
- every loop occurrence belongs to the expected loop body or loopback;
- the next local exit occurrence is `poll_loop_exit_pc`;
- no complementary-lane instruction occurs inside a cursor interval that is
  declared consumable.

Disjoint-lane instructions before the poll probe remain in the trace and execute
normally.

### 5.4 Runtime metadata

```text
runtime_strategy = FOLD_CAPTURED_POLL
fold_proof_version = 1
poll_loop_head_pc
poll_loop_body_pcs
poll_wait_pcs
poll_probe_pc
poll_loopback_pc
poll_loop_exit_pc
poll_advance_allowed_pcs
control_safe = true
control_proof
wait_condition_safe = true
wait_condition_proof
effect_safe = true
effect_classes
effect_proof
barrier_self_tied = true
scoreboard_proof
```

There is no entry action for this strategy.

The v4 loader independently rechecks the wait operands, predicate-owned
loopback, idempotent-copy shape, and scoreboard arm/wait closure against
`enhanced_execution_info.json`. Required proof objects are evidence, not trusted
substitutes for loader validation.

## 6. Simplified CFG design

The post-processor does not need one universal WaitBlock CFG algorithm.
It needs two bounded recognizers.

**Missing-wait recognizer**

```text
entry probe -> predicate branch -> captured ready continuation
```

It proves the gate identity, all-live-lane mask, and ready-only capture.

**Captured-poll recognizer**

```text
loop head -> poll body -> poll probe -> loopback / local exit
```

It reuses the existing spin-region detector and strengthens its opcode,
dependency, and local-exit guards.

Basic CFG requirements still apply:

- branch target and fallthrough are explicit;
- branch predicate ownership is proven;
- a backward branch is not accepted merely because its target PC is lower;
- unresolved control rejects the runtime strategy.

No semantic convergence PC is needed for cursor movement.

## 7. PC and cursor contract

`INJECT_MISSING_WAIT` uses PCs only to identify the issued entry probe.

```text
entry_probe_pc
entry_branch_pc
captured_ready_pc
```

Release resumes from the trace's existing next dynamic occurrence. It performs
no cursor edit.

`FOLD_CAPTURED_POLL` uses:

```text
poll_loop_head_pc
poll_probe_pc
poll_loopback_pc
poll_loop_exit_pc
```

The trace has two authoritative cursors:

```text
used_insts
map_warp_traces[pc].num_used_instructions[0]
```

Poll folding advances both cursors over residual allowed loop occurrences and
stops when the current head equals `poll_loop_exit_pc`. It does not consume the
exit instruction.

K5 example:

```text
0xb1d0 TRYWAIT              poll-loop head
0xb1e0 NANOSLEEP
0xb1f0 PHASECHK             poll probe
0xb200 BRA -> 0xb1d0        loopback
0xb210 BRA -> 0x9ea0        poll-loop exit PC; execute normally
```

The cursor stops before `0xb210`. Normal replay then executes its branch to
`0x9ea0`.

## 8. Canonical wait-condition checks

Both strategies require:

- normalized barrier-address expression;
- normalized wait-state/parity expression;
- runtime barrier-address equality at the action PC;
- runtime wait-state compatibility;
- static opcode and function identity.

For `INJECT_MISSING_WAIT`, equivalent static probes belonging to the same gate
must have equivalent reaching definitions. Equal immediate offsets alone are not
sufficient.

For `FOLD_CAPTURED_POLL`, the runtime payload at an active `poll_probe_pc`
occurrence remains the authority used by the already-successful fold path.
A predicated-off occurrence has `effective_mask == 0`, is an architectural
no-op, and may legitimately omit the runtime payload. Strict post-processing
must prove that every active poll occurrence has a valid payload while counting
predicated-off occurrences separately.

## 9. Sidecar schema

The redesigned runtime requires a schema bump because existing version-3 records
implicitly enable entry holds.

```json
{
  "version": 4,
  "schema": "wait_block_sidecar",
  "wait_block_report": {
    "discovered": 43,
    "passed": 36,
    "failed": 0,
    "skipped": 7,
    "inject_missing_wait": 22,
    "fold_captured_poll": 14,
    "metadata_only": 0,
    "ready_only_all_live_visits": 76224,
    "ready_only_partial_visits": 4596,
    "ready_only_records_with_partial_visits": 11
  },
  "kernels": [
    {
      "trace_kernel_id": 5,
      "unique_function_id": 3,
      "wait_blocks": [
        {
          "block_id": "wb_k5_fid3_e0x74e0",
          "runtime_strategy": "INJECT_MISSING_WAIT",
          "entry_probe_pc": 29920,
          "observed_effective_masks": ["0xffffffff"],
          "observed_live_masks": ["0xffffffff"],
          "partial_modeled_miss_action": "ABORT",
          "canonical_wait_condition": {}
        },
        {
          "block_id": "wb_k5_fid3_e0x9e80",
          "runtime_strategy": "FOLD_CAPTURED_POLL",
          "fold_proof_version": 1,
          "poll_loop_head_pc": 45520,
          "poll_wait_pcs": [45520, 45552],
          "poll_probe_pc": 45552,
          "poll_loopback_pc": 45568,
          "poll_loop_exit_pc": 45584,
          "poll_advance_allowed_pcs": [
            45520,
            45536,
            45552,
            45568
          ],
          "control_safe": true,
          "wait_condition_safe": true,
          "effect_safe": true,
          "barrier_self_tied": true
        }
      ]
    }
  ]
}
```

The `22/14` record split and visit-level mask counts are measured from all 516
K5/K10 CTAs in the current compact trace. Generated output remains the
authority.

Metadata-only WaitBlocks omit `runtime_strategy`. They remain available for
reporting and future work but create no runtime lookup.

> The `partial_modeled_miss_action` field is `ABORT` under the v1 runtime. The
> approved v2 partial policy bumps this field to
> `DEFER_BLOCKED_SUFFIX_THEN_HOLD` and requires sidecar schema v5. See the v2
> partial policy section in [WAIT_BLOCK_SIMULATOR.md](WAIT_BLOCK_SIMULATOR.md).

## 10. Reporting and fail-closed rules

The report must satisfy:

```text
discovered = passed + failed + skipped
passed = inject_missing_wait + fold_captured_poll + metadata_only
```

Closed strategy failure reasons:

```text
ENTRY_GATE_UNRESOLVED
ENTRY_NOT_READY_ONLY
ENTRY_WAIT_CONDITION_MISMATCH
ENTRY_WAIT_TARGET_UNDECODABLE
POLL_LOOP_UNRESOLVED
POLL_EFFECT_UNSAFE
POLL_DEPENDENCY_UNSAFE
POLL_LINEAR_CURSOR_UNSAFE
POLL_EXIT_UNREACHABLE
TRACE_IDENTITY_MISMATCH
```

`--strict` fails when an intended runtime target is not emitted with its expected
strategy. `wait_block_report.failed` is the single report-level authority:

```text
failed == 0  -> no strategy-proof strict failure
failed != 0  -> strict exits nonzero before publishing output
```

A semantic-bearing, local-memory-bearing, single-shot, or dependency-unsafe
region that is not an intended runtime target remains non-foldable/skipped. Its
existence alone does not make strict generation fail.

Runtime rejects:

- missing or malformed sidecar;
- any schema other than v4 when WaitBlock modeling is enabled;
- duplicate runtime keys;
- nonzero `failed`;
- missing required strategy fields.

Partial ready-only entries are not generation failures. They are emitted with
`INJECT_MISSING_WAIT` plus `partial_modeled_miss_action=ABORT`, and their visit
counts are mandatory report fields.

## 11. Current K5/K10 anchors

Required `INJECT_MISSING_WAIT` anchor:

- K5 `0x74e0`: ready-only capture, all-live-lane entry probe, release resumes
  normal trace replay with no cursor advance.

Required `FOLD_CAPTURED_POLL` anchors:

- K5 `0x9e80`: no entry action; poll probe `0xb1f0`; local exit `0xb210`;
  `BSYNC 0xa150` is never in the advance set.
- K10 `0xa9c0`: no entry action; poll probe `0xb980`; local exit `0xb9a0`;
  `WARPSYNC 0xb2c0` is never in the advance set.

The current K10 `0xa9c0` record has observed path counts `A/B/C=0/0/1344`.
It is therefore unambiguously `FOLD_CAPTURED_POLL`, not
`INJECT_MISSING_WAIT`.

K5 `0x9e80` has both ready and poll-loop visits. Because it owns a captured
foldable poll loop, v1 assigns only `FOLD_CAPTURED_POLL`. Entry-level modeled
misses on its captured ready visits are outside this version.

## 12. Measured ready-only mask and convergence findings

The current trace was scanned across all K5/K10 CTAs. Live masks were initialized
from block dimensions and updated by dynamic `EXIT` occurrences before each
entry. Every measured entry count matches the v3 sidecar's
`active_entry_visits`.

```text
K5:  12 ready-only records, 24,564 visits
     23,616 all-live, 948 partial

K10: 10 ready-only records, 56,256 visits
      52,608 all-live, 3,648 partial

Total: 80,820 visits
       76,224 all-live
        4,596 partial
```

Eleven ready-only records contain partial visits:

| Kernel | Entry PC | Partial visits | Effective mask |
|---|---:|---:|---:|
| K5 | `0x93b0` | 264 | `0x1` |
| K5 | `0x9930` | 144 | `0x1` |
| K5 | `0x9aa0` | 144 | `0x1` |
| K5 | `0xab40` | 132 | `0x1` |
| K5 | `0xabc0` | 132 | `0x1` |
| K5 | `0xac00` | 132 | `0x1` |
| K10 | `0x9ef0` | 384 | `0x1` |
| K10 | `0xa460` | 1,344 | `0x1` |
| K10 | `0xa810` | 1,344 | `0x1` |
| K10 | `0xac10` | 192 | `0x1` |
| K10 | `0xb000` | 384 | `0x1` |

Mask value alone is not the criterion. K10 `0xb430` executes with
`effective_mask=0x1` after lanes 1-31 have exited, so its live mask is also
`0x1` and all 384 visits are all-live.

The same full-trace scan tested whether both all-live and partial entries could
delay their wait hold to a later all-live convergence:

- For all 4,596 partial visits, the sidecar `resume_pc` occurrence remains
  partial (`effective_mask=0x1`, `live_mask=0xffffffff`). It is a lane-0 path
  resume, not warp reconvergence.
- A later all-live occurrence exists before the next same entry for only 2,580
  partial visits.
- K10 `0xa460` and `0xa810` each have no such occurrence for 1,008 of their
  1,344 partial visits, for 2,016 missing-boundary visits total.
- Every one of the 2,580 visits with a later all-live occurrence executes
  waiting-lane work before that boundary.

Representative waiting-lane work before the later all-live occurrence:

```text
K5 0x93b0 -> IMAD, R2UR, ...
K5 0x9930 -> IMAD, ISETP, UIADD3, R2UR, ...
K10 0xa460 -> LEA, LDC, ...
K10 0xa810 -> LDL, R2UR, UMOV, IADD3, ...
K10 0xac10 -> LDL, LDC, IMAD, R2UR, ...
K10 0xb000 -> LDL, R2UR, LDC, ...
```

Therefore "always wait-hold at the next all-live convergence" is rejected. It
would allow the waiting lane to execute work that the modeled miss is supposed
to block, and some visits have no usable all-live boundary at all.

## 13. Explicit v1 non-goals

This version does not implement:

- lane-group-only scheduler holds;
- per-dynamic-visit switching between the two strategies;
- entry wait injection for mixed ready/poll records;
- convergence-PC monitoring;
- a universal ready/wait-arm state machine.

These can be added only if current K5/K10 evidence shows they are required.
They are not prerequisites for the two requested functions. A ready-only
partial-mask modeled miss is still detected at runtime and aborts; it is not
silently treated as supported.

## 14. Server full-trace acceptance

The v4 post-processor was rerun on the server against the production compact
trace, without a kernel filter:

```text
python3 simulator-remodeled/util/tracer_nvbit/build_spin_wait_sidecar.py \
    --traces <TRACE_ROOT>/traces \
    --validate-ctas all \
    --strict
```

For every normal (non-warp-lifetime-only) trace, `run_hw_trace.py` requires
`enhanced_execution_info.json` and `sync_operand_resolver.json`. Missing
metadata aborts trace post-processing instead of silently omitting the sidecar.
It passes `--validate-ctas all --strict`; `subprocess.run(check=True)` makes any
proof or completeness failure fail the trace-generation workflow instead of
publishing a new sidecar.

The automatic order is:

```text
NVBit trace
-> TMA/sync operand mapping
-> optional TRIM_SPIN rewrite
-> strict WaitBlock v4 generation on the final simulator-facing trace
```

`TRIM_SPIN` copies `extra_info` while rewriting dynamic protobuf streams.
Generating v4 only after that rewrite prevents full-trace counts and proofs
from being copied unchanged into a different trimmed trace.

The strict run completed successfully and published a schema-v4 sidecar with:

```text
discovered=43 passed=36 failed=0 skipped=7
INJECT_MISSING_WAIT=22
FOLD_CAPTURED_POLL=14
metadata_only=0
ready-only all-live visits=76,224
ready-only partial visits=4,596
ready-only records with partial visits=11
```

After publishing the sidecar, the generator repeats the complete acceptance
result as its final log line:

```text
[wait-block] FINAL status=PASS validation_complete=true discovered=43 passed=36 failed=0 skipped=7 inject=22 fold=14 metadata=0 all_live=76224 partial=4596 records_with_partial=11 output=<SIDECAR_PATH>
```

Consequently `tail -n 1` is sufficient to inspect a successful run; no `grep`
is required. A strict proof failure exits before publishing a sidecar and never
prints `status=PASS`.

Kernel-level acceptance:

```text
K5:  INJECT_MISSING_WAIT=12, FOLD_CAPTURED_POLL=8
K10: INJECT_MISSING_WAIT=10, FOLD_CAPTURED_POLL=6
```

All skips are expected and fail-closed:

- K5 `0x96c0` is `CAPTURED_POLL_ENTRY_NOT_REQUIRED`; its runtime action is the
  independently proven captured poll at `0xb0b0`.
- K5 `0xac40` and `0xac60`, K10 `0xb480` and `0xb4a0`, and K11 `0x5a0` and
  `0x5c0` are unresolved single-shot/internal shapes and create no runtime
  action.

The report invariants hold:

```text
43 = 36 passed + 0 failed + 7 skipped
36 = 22 inject + 14 fold + 0 metadata
```

The captured-poll proof hardening was revalidated against the same complete
trace. All 14 fold records passed control ownership, static and dynamic wait
condition, idempotent-copy effect, and complete skip-set scoreboard checks.
Across those records, 754,859 active wait occurrences had zero payload omissions
and zero per-iteration condition mismatches. The runtime action projection
(strategy key, body, loopback, and exit) is unchanged from the previous v4
sidecar.

Because the command omitted `--out`, it wrote
`extra_info/spin_wait_sidecar.json`. That server artifact is schema v4 and must
not be consumed by the reverted baseline simulator. It is a post-processing
acceptance artifact until the v4 loader and runtime are implemented.

This acceptance proves classification, payload extraction, mask census,
strategy ownership, and strict report completeness. It does not prove simulator
hold/release or cursor behavior.
