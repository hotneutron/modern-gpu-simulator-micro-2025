# WaitBlock: Two-Strategy Post-Processing and Simulator Design

**Status:** POST-PROCESSING V4 IMPLEMENTED AND FULL-TRACE VALIDATED;
SIMULATOR V1 IMPLEMENTED; SERVER BUILD AND K5/K10 RUNTIME VALIDATION PENDING.

**Extends:** [`SPIN_WAIT_SIM_MODELING.md`](SPIN_WAIT_SIM_MODELING.md) and
[`SPIN_WAIT_POSTPROCESS.md`](SPIN_WAIT_POSTPROCESS.md).

**Debug record:** [`../debug-k10-wait-fold-deadlock.md`](../debug-k10-wait-fold-deadlock.md).

This document intentionally reduces WaitBlock modeling to two runtime
strategies:

```text
INJECT_MISSING_WAIT
FOLD_CAPTURED_POLL
```

The strategies solve two different trace shapes and must not share an action
site.

`INJECT_MISSING_WAIT` handles a ready-only captured gate. The trace contains no
dynamic wait arm, but the simulator observes not-ready at the entry probe. The
simulator issues the entry probe, wait-holds the warp until the expected modeled
generation arrives, and then resumes the captured ready arm without moving the
trace cursor.

The handler is mask-aware at runtime:

```text
modeled ready:
    normal replay, regardless of entry mask

modeled not-ready and effective entry mask covers all live lanes:
    inject a whole-warp wait hold

modeled not-ready and effective entry mask is partial:
    print UNSUPPORTED_PARTIAL_MISSING_WAIT and abort
```

The last branch is a guard, not a third runtime strategy. It detects when the
current warp-level scheduler cannot faithfully represent the required
lane-group wait.

`FOLD_CAPTURED_POLL` handles a captured polling loop. The simulator leaves the
entry probe unchanged, applies the existing successful fold behavior at the
poll probe, and consumes only residual captured poll-loop occurrences after the
modeled release.

The original failed implementation made both shapes use an entry hold and then
advanced toward semantic convergence. That caused:

- K10 `0xa9c0`: a captured-poll record was incorrectly treated as a missing-wait
  record and held at the entry instead of folding at `0xb980`;
- K5 `0x9e80`: release advanced toward convergence `0x9ea0` instead of stopping
  before the local poll-loop exit instruction `0xb210`.

The unified WaitBlock database remains useful for Stage 1.9. Unified metadata
does not require unified runtime behavior.

Terminology:

- **entry probe:** first `TRYWAIT` that tests the wait condition;
- **captured ready arm:** path recorded when the entry probe was ready;
- **captured poll loop:** recorded repeated probe/backoff/backedge sequence;
- **poll probe:** probe whose result controls the poll-loop backedge;
- **poll-loop exit PC:** first instruction after the poll loop; it must execute;
- **wait hold:** remove the warp from scheduler eligibility after an issued
  probe until the modeled event is ready.

Hard invariants:

- Every selected probe executes once before a wait hold can begin.
- `INJECT_MISSING_WAIT` never advances either trace cursor.
- `FOLD_CAPTURED_POLL` never acts at the entry probe.
- Poll folding stops before `poll_loop_exit_pc`.
- Mainline, complementary-lane, reconvergence, and semantic instructions are
  never consumed by poll folding.
- A ready-only partial-mask modeled miss never silently continues and never
  becomes a whole-warp wait hold; it aborts with a complete diagnostic.
- Generation overshoot and a 32,000-cycle wait-hold timeout abort.
- Feature-off behavior is bit-identical.

## 1. Trace Post-Processing: Classify One of Two Runtime Strategies

### 1.1 Purpose

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

### 1.2 Input and identity

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

### 1.3 Classification order

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

### 1.4 Strategy A: `INJECT_MISSING_WAIT`

#### Required trace shape

The validated dynamic trace contains only the ready arm:

```text
entry probe
-> controlling branch
-> captured ready continuation
```

The not-ready behavior is not dynamically present. This is the missing wait the
simulator must inject when modeled readiness disagrees with the capture.

#### Required static proof

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

#### Required dynamic proof

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

#### Runtime metadata

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

### 1.5 Strategy B: `FOLD_CAPTURED_POLL`

#### Required trace shape

The trace contains a repeated polling loop:

```text
poll_loop_head
-> wait/backoff instructions
-> poll_probe
-> loopback or poll_loop_exit
```

This is the behavior already modeled successfully by the original spin-fold
implementation.

#### Required static proof

Reuse the existing spin-region requirements:

- the loopback is a direct branch to `poll_loop_head_pc`;
- `poll_probe_pc` controls the loopback;
- the body contains only reviewed wait/control/backoff/setup instructions;
- the body contains no semantic, memory, reconvergence, dependency, call,
  return, or unknown operation;
- setup definitions do not escape the poll loop;
- dependency barriers are self-tied;
- `poll_loop_exit_pc` is the first instruction after the loopback and exists in
  the trace.

The exact runtime allow set is:

```text
poll_loop_body_pcs U {poll_loopback_pc}
```

It does not contain `poll_loop_exit_pc` or semantic convergence.

#### Required dynamic proof

The dynamic trace must show repeated loop visits and a reachable local exit.

Post-processing verifies:

- the poll probe's runtime barrier address and wait state;
- every loop occurrence belongs to the expected loop body or loopback;
- the next local exit occurrence is `poll_loop_exit_pc`;
- no complementary-lane instruction occurs inside a cursor interval that is
  declared consumable.

Disjoint-lane instructions before the poll probe remain in the trace and execute
normally.

#### Runtime metadata

```text
runtime_strategy = FOLD_CAPTURED_POLL
poll_loop_head_pc
poll_loop_body_pcs
poll_probe_pc
poll_loopback_pc
poll_loop_exit_pc
poll_advance_allowed_pcs
effect_safe = true
effect_classes
canonical_wait_condition
```

There is no entry action for this strategy.

### 1.6 Simplified CFG design

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

### 1.7 PC and cursor contract

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

### 1.8 Canonical wait-condition checks

Both strategies require:

- normalized barrier-address expression;
- normalized wait-state/parity expression;
- runtime barrier-address equality at the action PC;
- runtime wait-state compatibility;
- static opcode and function identity.

For `INJECT_MISSING_WAIT`, equivalent static probes belonging to the same gate
must have equivalent reaching definitions. Equal immediate offsets alone are not
sufficient.

For `FOLD_CAPTURED_POLL`, the runtime payload at `poll_probe_pc` remains the
authority used by the already-successful fold path.

### 1.9 Sidecar schema

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
          "poll_loop_head_pc": 45520,
          "poll_probe_pc": 45552,
          "poll_loopback_pc": 45568,
          "poll_loop_exit_pc": 45584,
          "poll_advance_allowed_pcs": [
            45520,
            45536,
            45552,
            45568
          ],
          "effect_safe": true,
          "canonical_wait_condition": {}
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

### 1.10 Reporting and fail-closed rules

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

### 1.11 Current K5/K10 anchors

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

### 1.12 Measured ready-only mask and convergence findings

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

### 1.13 Explicit v1 non-goals

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

### 1.14 Server full-trace acceptance

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

Because the command omitted `--out`, it wrote
`extra_info/spin_wait_sidecar.json`. That server artifact is schema v4 and must
not be consumed by the reverted baseline simulator. It is a post-processing
acceptance artifact until the v4 loader and runtime are implemented.

This acceptance proves classification, payload extraction, mask census,
strategy ownership, and strict report completeness. It does not prove simulator
hold/release or cursor behavior.

## 2. Simulator Redesign: Two Independent Runtime Paths

### 2.1 Runtime maps

Load two disjoint maps:

```text
missing_wait_by_entry:
    (trace_kernel_id, unique_function_id, entry_probe_pc)
    -> INJECT_MISSING_WAIT record

captured_poll_by_probe:
    (trace_kernel_id, unique_function_id, poll_probe_pc)
    -> FOLD_CAPTURED_POLL record
```

A record cannot appear in both maps for the same runtime action.

The old captured spin-region runtime owner is replaced only after its
`captured_poll_by_probe` entry is installed. This prevents a captured loop from
losing its fold owner.

### 2.2 Minimal pending state

Only two pending hold modes are required:

```text
NONE
MISSING_WAIT_HOLD
CAPTURED_POLL_HOLD
```

Common state:

```text
strategy and block identity
runtime barrier key
normalized wait state
expected generation
action PC
hold start cycle
```

`CAPTURED_POLL_HOLD` additionally stores:

```text
poll_loop_exit_pc
poll_advance_allowed_pcs
```

There is no deferred-observation state.

### 2.3 Runtime path A: `INJECT_MISSING_WAIT`

Lookup occurs only at `entry_probe_pc`.

The entry probe executes normally. Runtime then evaluates the modeled wait.

```text
if modeled ready:
    continue normal trace replay

if modeled not-ready and effective_mask == live_mask:
    pin expected generation
    enter MISSING_WAIT_HOLD

if modeled not-ready and effective_mask != live_mask:
    print UNSUPPORTED_PARTIAL_MISSING_WAIT diagnostic
    abort
```

While held, the warp is ineligible and other warps/async engines continue.

On release:

```text
assert current generation == expected generation
clear MISSING_WAIT_HOLD
continue from the existing trace head
```

No IBuffer flush is needed solely for release. No linear or per-PC cursor is
advanced. The captured ready branch executes normally after release.

The partial-mask abort is explicit (`fprintf` plus `abort()` or the repository's
equivalent fatal helper), not a C/C++ `assert()` that can disappear under
`NDEBUG`.

Required partial-miss diagnostic:

```text
cycle
trace kernel and unique function ID
SM, CTA, warp, and entry_probe_pc
active_mask, predicate_mask, effective_mask, and live_mask
barrier address and normalized wait state
current and expected generation
captured ready/poll visit counts
```

This is the entire missing-wait feature and its unsupported-case guard.

### 2.4 Runtime path B: `FOLD_CAPTURED_POLL`

There is no lookup or action at the WaitBlock entry.

Lookup occurs only when the captured trace reaches `poll_probe_pc`.

The poll probe executes normally.

```text
if modeled ready:
    fold residual captured poll-loop occurrences immediately

if modeled not-ready:
    pin expected generation
    enter CAPTURED_POLL_HOLD
```

On release:

```text
assert current generation == expected generation
flush IBuffer and roll back prefetched occurrences
advance residual allowed poll-loop occurrences
assert trace head == poll_loop_exit_pc
clear CAPTURED_POLL_HOLD
resume normal replay at poll_loop_exit_pc
```

This reuses the previously successful captured-spin fold path.

### 2.5 Exact cursor advance

Cursor advance is used only by `FOLD_CAPTURED_POLL`.

```text
while trace_head_pc != poll_loop_exit_pc:
    assert trace is not exhausted
    assert trace_head_pc in poll_advance_allowed_pcs
    assert current opcode/effect is reviewed
    advance map_warp_traces[pc].num_used_instructions[0]
    advance used_insts

assert trace_head_pc == poll_loop_exit_pc
```

The operation stops before the exit instruction.

It aborts before consuming:

- `BSYNC`, `BSSY`, or `WARPSYNC`;
- memory, TMA, HGMMA, MUFU, BAR, arrive, or exchange;
- dependency/fence operations;
- indirect control, call, return, exit, or unknown opcode;
- any PC not present in the exact allow set.

The runtime guard is an exact allow check, not a broad denylist.

### 2.6 Generation and watchdog

Both strategies use the existing generation auditor.

```text
current_gen < expected_gen:
    remain held

current_gen == expected_gen and modeled condition ready:
    release

current_gen > expected_gen:
    abort generation overshoot
```

Generation never force-releases a wait.

The shared wait-hold watchdog threshold is 32,000 cycles. At 32,001 cycles,
emit a full diagnostic and abort.

For long runs, print a low-rate heartbeat every 4,096 held cycles:

```text
cycle, SM, CTA, warp, strategy, block, action PC,
barrier, expected/current generation, held cycles
```

### 2.7 Scheduler and lifecycle

Wait hold means whole-warp scheduler ineligibility.

`INJECT_MISSING_WAIT` creates a hold only when the current entry occurrence
covers all live lanes. A partial occurrence may replay only when modeled ready;
a partial modeled miss aborts before scheduler state changes.

`FOLD_CAPTURED_POLL` retains the existing poll-site behavior that already
completed K5/K10 before the entry-hold redesign.

Release is owned by one authoritative service. Scheduler sorting,
instrumentation, and stall queries use a non-mutating held-state query.

Before CTA cleanup or warp-slot reuse:

```text
assert pending mode == NONE
assert hold start cycle is clear
assert holds == releases
```

Assertions execute before cleanup resets any state.

### 2.8 Metrics and debug logs

Keep strategy counters separate.

```text
missing_wait_ready
missing_wait_holds
missing_wait_releases
partial_missing_wait_model_misses

captured_poll_immediate_folds
captured_poll_holds
captured_poll_releases
captured_poll_advanced_occurrences

generation_overshoots
watchdog_aborts
```

A ready `INJECT_MISSING_WAIT` entry is a passthrough, not a fold.
Generation overshoot is not a watchdog event.

Required events:

```text
[WAITBLOCK][missing-wait-hold]
[WAITBLOCK][missing-wait-release]
[WAITBLOCK][UNSUPPORTED_PARTIAL_MISSING_WAIT]
[WAITBLOCK][poll-hold]
[WAITBLOCK][poll-release]
[WAITBLOCK][poll-advance]
[WAITBLOCK][heartbeat]
[WAITBLOCK][abort]
[WAITBLOCK][summary]
```

Poll-advance logs include:

```text
used_insts before/after
per-PC occurrences consumed
skipped opcode counts
expected exit PC
actual landing PC
```

### 2.9 Feature gating

The v4 runtime has one canonical feature gate and one debug gate:

```text
-wait_block_enable
-wait_block_debug
```

The former `-spin_wait_fold_enable` and `-spin_wait_fold_debug` options are
removed. `FOLD_CAPTURED_POLL` is a strategy selected by v4 sidecar metadata, not
a separately enabled runtime mode. `-spin_wait_observe_enable` remains an
independent read-only measurement option and never creates a hold or advances a
cursor.

With `-wait_block_enable 0`:

- do not require sidecar v4;
- do not build either strategy map;
- do not create pending wait state;
- do not change scheduler eligibility;
- do not flush or advance cursors;
- retain current trace replay and metrics exactly.

With the feature enabled, schema, identity, generation, cursor, or lifecycle
violations abort. Runtime does not silently revert after a hold or cursor
mutation. The partial missing-wait guard is feature-on only; feature-off remains
bit-identical.

### 2.10 Partial-lane v1 policy and future design boundary

Partial ready-only visits are present in the current trace, but modeled-miss
occurrence cannot be established offline. V1 therefore detects the unsupported
dynamic combination exactly:

```text
captured shape is ready-only
AND effective_mask != live_mask
AND modeled_ready == false
```

V1 behavior is a mandatory diagnostic followed by `abort()`. It must not:

- continue the captured ready path;
- convert the event into a whole-warp wait hold;
- defer blindly to the next all-live occurrence;
- force-release or consume trace occurrences.

This abort is the temporary correctness boundary while partial-lane execution is
designed. It is not the final support mechanism.

Full support requires a separate design for one of:

1. lane-group trace cursors; or
2. a deferred-occurrence queue that preserves blocked-lane instructions and
   their per-PC payloads while allowing disjoint-lane occurrences to issue.

That design must preserve:

- per-lane instruction order;
- the linear and per-PC trace occurrence cursors;
- IBuffer prefetch and rollback;
- branch/reconvergence ordering;
- scoreboard/dependency state;
- execution counters and semantic-work invariants;
- deterministic merge of deferred occurrences after modeled release.

No lane-group implementation is part of this v1 document. The
`UNSUPPORTED_PARTIAL_MISSING_WAIT` abort remains enabled until a separately
reviewed design replaces it.

### 2.11 Focused tests

Post-processing:

- missing `enhanced_execution_info.json` aborts automatic generation;
- missing `sync_operand_resolver.json` aborts automatic generation;
- an empty resolver with no static wait opcode emits a valid empty v4 sidecar;
- an empty resolver with static wait opcodes aborts;
- `failed=0` cannot produce a strategy-proof strict failure;
- `failed>0` exits before output publication;
- a published sidecar prints the complete `FINAL` summary as the last log line;
- `TRIM_SPIN` runs before strict sidecar generation;
- ready-only/all-live gate emits `INJECT_MISSING_WAIT`;
- ready-only/partial-mask gate emits `INJECT_MISSING_WAIT`, records its masks,
  and declares `partial_modeled_miss_action=ABORT`;
- partial visit and partial-record report counts reconcile with raw trace visits;
- any record owning a foldable captured loop emits
  `FOLD_CAPTURED_POLL`, never `INJECT_MISSING_WAIT`;
- unsafe loop effects or dependency barriers reject poll folding;
- K5/K10 action counts and PC anchors match the reviewed sidecar;
- duplicate kernel/function/action-PC identities are rejected.

Runtime path A:

- ready entry performs no hold and no cursor change;
- ready partial entry performs no hold and replays normally;
- not-ready all-live entry holds and releases at the expected generation;
- not-ready partial entry prints the complete diagnostic and aborts before
  scheduler or cursor mutation;
- release preserves both trace cursors;
- the guard remains disabled when `-wait_block_enable 0`.

Runtime path B:

- ready poll folds immediately to the local exit;
- not-ready poll holds, releases, and folds to the local exit;
- both cursors advance consistently;
- the exit instruction is not consumed;
- an unexpected complementary-lane or semantic occurrence aborts before it is
  consumed.

K5:

- `0x74e0` uses `INJECT_MISSING_WAIT`;
- `0x9e80` has no entry action;
- `0xb1f0` uses `FOLD_CAPTURED_POLL`;
- landing PC is `0xb210`;
- `BSYNC 0xa150` executes normally.

K10:

- `0xa9c0` has no entry action;
- `0xb980` uses `FOLD_CAPTURED_POLL`;
- landing PC is `0xb9a0`;
- `WARPSYNC 0xb2c0` executes normally;
- consumer warps reach their `0x31020` arrives.

Cross-kernel:

- feature-off output is bit-identical;
- every hold has one release;
- every poll advance has a successful landing audit;
- semantic instruction and TMA command/byte counts are unchanged;
- with zero partial modeled misses, K5 and K10 exit without generation
  overshoot or watchdog abort;
- if a partial modeled miss occurs, the run stops at the first
  `UNSUPPORTED_PARTIAL_MISSING_WAIT` diagnostic before scheduler or cursor
  mutation. It must not hang or continue with approximate behavior.

### 2.12 Implementation order

1. [done] Approve this simplified design.
2. [done] Revert the failed universal WaitBlock implementation.
3. [done] Change post-processing classification and emit sidecar v4.
4. [done] Generate a full-CTA candidate sidecar with:

   ```text
   build_spin_wait_sidecar.py --traces <TRACE_DIR> --kernels <IDS>
       --validate-ctas all --strict
       --out <TRACE_DIR>/extra_info/spin_wait_sidecar.v4.candidate.json
   ```

5. [done] Review the strategy counts, missing-wait mask report, and captured
   poll records (`discovered=43`, `passed=36`, `failed=0`, `skipped=7`;
   `inject=22`, `fold=14`).
6. [done] Repeat full-CTA strict generation on the server without a kernel
   filter and verify the same report and strategy counts.
7. [done] Integrate full-CTA strict sidecar generation into `run_hw_trace.py`
   after sync operand mapping, with mandatory metadata and post-trim ordering.
8. [done] Restore captured-poll ownership at poll probes in the v4
   simulator.
9. [done] Implement `INJECT_MISSING_WAIT` and its partial-miss diagnostic
   abort as a
   separate entry-only path.
10. [done] Add strict loader, identity, generation, landing, and lifecycle
   guards.
11. [in progress] Static sidecar-accounting, stale-field, whitespace,
    hardcoding, metadata-only serialization, and closed cursor-opcode audits
    pass. Full-CTA strict regeneration on the compact trace passes with the
    reviewed `43/36/0/7`, `22/14`, and `76,224/4,596` results. A local C++ build
    is unavailable because this macOS environment has no CUDA headers; compile
    validation remains a server step.
12. [not started] The user builds and runs K5/K10.

The design is complete when each runtime action is explained by exactly one of
the two strategies. No entry action is permitted for a captured-poll record, and
no cursor action is permitted for a missing-wait record.
