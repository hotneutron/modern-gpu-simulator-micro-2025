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

### Occurrence-specific partial joins

The HD128 capture demonstrates that a single ready-only entry PC can rejoin at
different synchronization sites on different dynamic visits. Action selection
keeps the existing key, but partial hold proof version 3 adds exact
`(trace_kernel_id, CTA coordinates, warp_id, source instruction index)` bindings
for the entry and its sibling, blocked, and post-join occurrences. It is separate
from the mbarrier generation-ticket stream.

Each visit must pass either the legacy direct-branch proof (variant 1), or the
already-arrived `WARPSYNC.ALL` proof (variant 2). Variant 2 requires the last
sibling-owned occurrence before the wait to be an active full-mask WARPSYNC
for that sibling group, no later sibling work until reconvergence, and a
matching blocked-group join followed by the live-warp continuation. Earlier
sibling work is neither moved nor replayed. Repeated PCs are matched by dynamic
source indices. Different visits may select different variants; a failed visit
still rejects the record.

Legacy single-join records remain proof version 1. Version 3 emits a variant
table plus occurrence bindings, and requires the matching C++ loader. The
loader checks static anchors, masks, occurrence ordering, uniqueness, and
coverage. Runtime checks entry identity and requires an already-arrived
sibling join to have actually issued before holding the warp. Join bookkeeping
is occurrence-specific to avoid matching an unrelated visit to the same PC.

### Independent lifecycle witnesses

Lifecycle witness discovery is independent of INJECT/FOLD selection. Static
analysis connects each supported wait predicate to a direct branch without
intervening predicate references or unknown instructions. Dynamic analysis
requires the same lane group to reach the recorded branch and matching edge.
Predicated-off branches are evaluated using their raw active mask; an absent
branch is not a ready result. Ambiguous or incomplete evidence stays unknown.

These witness-only waits do not become runtime actions. Their ready outcomes
feed the existing barrier epoch, ordinal, single-arrival, completion ownership,
and generation uniqueness checks. A conflict with an existing runtime-action
outcome fails validation.

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

#### Two-gate reload setup

The Type-1 detector also accepts a bounded sequence of unconditional
`IMAD.U32` / `IMAD.MOV.U32` zero-product register copies between the first
ready branch and the second probe. This covers HD128 consumer gates such as
`0x1640` and `0x1a40`; these PCs are examples, not discovery filters.

- Both probes must have identical normalized predicate, barrier, and state
  operands. Their direct branches must consume the same predicate namespace,
  have opposite ready/miss polarity, and share one continuation.
- A copied wait-state register must already contain the same source value at
  the first probe. A bounded backwards proof finds its prior copy and rejects
  clobbers, unknown instructions, control flow, and potentially overlapping
  wide destinations. The dynamic trace must execute that exact prefix with
  all entering lanes. Setup must not change the barrier address.
- Existing first-gate setup stays on the mainline. Second-gate setup is
  pass-through only on the captured miss path; it is not added to skippable
  PCs. A ready-only capture executes neither that setup nor the fallback.
- Each visit must reach a predicate-consistent branch edge. A dynamic proof
  failure for a recognized reload gate is a strict failure, not a silent
  skip. Mixed ready/second-gate/fallback captures retain the existing strategy
  classification; they are not promoted to unconditional INJECT records.

The emitted record still uses schema-v5 `INJECT_MISSING_WAIT` and schema-v2
generation tickets. No new simulator mode or C++ hold logic is required.
Regenerate both artifacts together from the actual simulation trace. A
lifecycle-witness PASS alone does not guarantee runtime wait coverage.

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

## 15. HD128 two-gate validation (2026-09-10)

The post-processing fix admits the register-reload Type-1 gates without changing
simulator production C++, schema-v5 records, or schema-v2 ticket format.

| Check | Result |
|---|---|
| CPU tests, including compiled production loader | 120 PASS in the isolated staged snapshot (102 existing + 18 new) |
| HD128 all-CTA strict generation with contract capture | PASS; INJECT 20, FOLD 11, metadata 3, failed 0 |
| K10 `0x1640` / `0x1a40` | 52224 ready-only all-live visits each; no missing sync payload |
| HD128 exact target slots | 36210 -> 140658; exactly 104448 new slots |
| Existing HD128 tickets / eligible barriers | Unchanged |
| HD128 K5 records / K10 FOLD records | Unchanged |
| HD64 contract-v2 baseline | All kernel records, report counts, and entire ticket document unchanged |
| Original `.pb` files and `stats.csv` | Before/after aggregate SHA-256 unchanged for both inputs |

Validation used separate output directories without overwriting the source
trace or its sidecars. The final dynamic guards additionally passed all 384
K10 CTAs and produced the same records. Temporary output directories, logs,
and validation helpers were removed after recording these results; the
permanent regression tests remain under `util/tracer_nvbit/tests/`.

The payload check hashed each sorted relative `.pb` path and its bytes,
followed by `stats.csv` and its bytes. Before/after values were identical:

| Input | Files | Aggregate SHA-256 |
|---|---:|---|
| HD128 device-2 capture | 18040 | `6cd57f6eb8fd2e279ffcbf1f67c66e63a29249c7e8f70a00654b7ff4ba8d53e9` |
| HD64 device-0 contract-v2 baseline | 10984 | `0b0c0410791b9f2c4cedc62df5b832868b3fe6b303168400075e8cbe9e5a7fe9` |

Local HD128 raw instruction counts and sidecar counts differ from `.e74/.o74`.
These checks qualify the local capture, not a reproduction of that exact run.
No new `gpu_sim_cycle` or cross-run `gpu_sim_insn` claim is made.

### Server rerun

Use the actual `traces` directory consumed by the failed K10 run. Preserve its
existing `extra_info/spin_wait_sidecar.json`,
`wait_block_exact_generation_tickets.json`, and
`mbarrier_contract_ordinal_feasibility.json` before regeneration. From
`simulator-remodeled`, with `T` set to that directory:

```bash
python3 util/tracer_nvbit/build_spin_wait_sidecar.py \
  --traces "$T" --validate-ctas all --strict --require-contract-capture
```

Regenerate all three artifacts together; `--out` redirects only the sidecar,
not the two contract/ticket artifacts. Do not use the local tickets with a
different capture. No GPU retracing is required for this fix.

Rerun K10 using the same trace, binary, and timing config with only these
post-processing artifacts changed. A simulator already supporting `f1e7acc`
does not need a rebuild for this detector-only fix. Required acceptance:
new consumer holds resolve at their exact generations, the TMA reuse abort is
gone, all lifecycle/conservation audits pass, and original instruction
accounting is preserved. Then rerun K5/HD64 cycle regression. Full Linux/CUDA
simulation remains pending; the local machine has no `nvcc`.

## Noncausal Interleaved Poll Proof

`partial_hold_proof_version=4` extends the v5 sidecar with
`lane_group_spans` and `lane_group_occurrences`. It retains the variant table
for ordinary join-only visits of the same entry PC. Older simulators reject
version 4; rebuild the simulator before consuming these artifacts.

- First try the existing direct/arrived join proofs, including a statically and
  dynamically verified full-member `WARPSYNC.COLLECTIVE` with linear SHFL body.
- Only a visit rejected by these proofs can cause discovery of a lane-group
  span. Thus accepted causal traces do not acquire new replay paths.
- Prove the sibling's four-instruction TRYWAIT/NANOSLEEP/PHASECHK/BRA loop,
  lane-private operands, wait condition, predicates, and scoreboard closure.
- Permit captured-ready TRYWAIT tails with predicated-off instructions, retaining
  raw lane ownership. They are not missing occurrences.
- Bind each span to exact kernel, function, CTA, warp, start/poll-start/merge
  source indices, masks, poll condition, and occurrence counts.
- A main-lane poll prefix may move the activation earlier, before interleaving
  begins. Prove this prefix's poll shape separately. Ordinary sibling prefix
  and tail instructions remain ordered and replayed, never folded as polling.
- Accept a full-mask wait boundary or a proven collective continuation. Do not
  label a full-mask PHASECHK as WARPSYNC. Reject calls, returns, exits, indirect
  control, mask escapes, ambiguous boundaries, and overlapping spans.
- Keep strict validation and exact-generation capture requirements unchanged.

`--out` redirects only `spin_wait_sidecar.json`; contract and generation-ticket
artifacts are still generated under the input's `extra_info`. Regenerate the
three matching artifacts together for a server run. Raw protobufs and `stats.csv`
are not rewritten.

Local regression: causal HD64 contract-v2 and causal HD128 pass all-kernel,
all-CTA strict validation (11 kernels each), with 36 and 34 passed records
respectively and zero failures. Both emit zero proof-v4 records. HD64's complete
kernel document and report equal its stored sidecar. HD128 K5 matches its stored
sidecar; K10 equals a fresh run of pre-extension commit `ba556d6`, including
that separate commit's two newly recognized two-gate waits. Its older stored
sidecar predates that detector fix.

Four additional HD64 captures (base, sync-order-v1, lighter-compact_wire, and
lighter-compact_wire+prune_operands-69a20aed) pass before/after partial-proof
comparison on all K5/K10 CTAs: 132/384 CTAs and 948/3,648 partial visits per
capture. All proof results equal `ba556d6`, with no lane-group spans added.
This comparison targets the changed partial-proof path; it is not an all-path
simulator run or a claim that older v4 sidecars can be loaded unchanged.

Noncausal HD64 all-kernel/all-CTA strict validation passes: 32 records, zero
failures, 6 skips; contract capture has 1,795,127 active sync events with zero
missing/mismatched snapshots and 205,202 uniquely assigned eligible actions.
The 132 replay spans cover 1,188 previously rejected partial waits; 125 end at
a common poll and 7 at a collective continuation. C++ loader fixtures accept
these actual span/wait records. C++ extraction fixtures replay all 552,610
extracted occurrences exactly once using actual PC/mask/sync/source fields.
Memory payload preservation uses synthetic payload fixtures; full simulator
execution and cycle accuracy remain unverified locally.

The local non-FA3 `mbarrier-release-count-many` capture still fails strict
validation with `POLL_EFFECT_UNSAFE / DEFUSE_UNDECODED`. The complete output
and rejection match `ba556d6`; this is an existing unsupported shape, not an
application-name exception or a regression. No sidecar is emitted for it.
Consequently neither all applications nor every local experimental capture is
claimed supported. The admissible lane-group pattern is application-independent.

Reusable tests cover relocated PCs, function IDs, registers/predicates, barrier
addresses, phases, and disjoint masks including non-lane-0 partitions. All 142
tests pass after removal of the one-off capture harnesses and HTTP debug
instrumentation. The replay queue reuse test runs with address/undefined
sanitizers. Trace-specific temporary scripts and their generated fixtures were
deleted; generic tests under `util/tracer_nvbit/tests` remain.
