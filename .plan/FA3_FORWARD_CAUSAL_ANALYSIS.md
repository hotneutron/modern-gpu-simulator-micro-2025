# FA3 Multi-Shape: Causal Analysis and Failure History

Date: 2026-09-11
Last updated: 2026-09-14

Scope: explain the completed forward runs using raw NCU reports, function-
matched SASS, current simulator code, and simulator runtime logs. Historical
analysis documents are hypotheses, not evidence or exclusion rules.

The results-only report is `.result/FA3_multi.md`. This document consolidates
the independent forward investigation and the historical failed-run notes.
Temporary analysis scripts, the compiled probe, and generated JSON were
removed after verification; the findings and verification record remain here.

## Evidence status

The offline investigation identifies a hardware-inconsistent SFU assignment
for BF16 packing, a missing warpgroup completion-dependency model, and an
extra SFU issue-stage bubble in a source-derived recurrence. These findings
are more specific than "consumer lockstep." The first server experiment now
shows substantially lower HD64 cycle error and SFU contention with unchanged
INT `2,2` configuration. However, the historical and new logs consumed
different trace-load populations and tickets. Strict same-input instruction
conservation and patch-isolated cycle attribution remain unverified.
The user accepted the F2FP-only scope as completed Opt 13 on 2026-09-11.
Its implementation/experiment status is Done; the preceding measurement
limitations remain part of the record, not a claim of a failed implementation.
Additional opcode candidates are deferred in `.result/FA3_progress.md`
Arch TODO-3. The SFU acceptance correction (problem 3) is now implemented
and committed as `10505f2`. The 2026-09-14 baseline server-log comparison
preserves instruction counts and passes in-run audits, but shows no
kernel-cycle accuracy improvement: K5 changes by +63 cycles and K10 by -485.
SFU-blocked warp-cycles decrease substantially. Exact binary provenance and
per-instruction acceptance timestamps remain unavailable. On 2026-09-14,
the user decided to retain the SFU correction and close this investigation
without further tuning or instrumentation. It is not a new optimization
entry: no Opt 14 is added to `.result/FA3_progress.md`. WGMMA group completion
(problem 2) remains an unimplemented proposal, not an automatic next action.
Sections 1-4 and the routing audit preserve the pre-patch evidence;
section 3 also records the SFU implementation and its validation limits.
Section 6.8 records the F2FP implementation; section 6.9 records its experiment.

Primary evidence already re-read:

- `nv_reports/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.ncu-rep`,
  action 4, whose mangled kernel name exactly matches the HD64 forward runs.
- Forward `.o90/.e90` (baseline), `.o97/.e97` (s1024),
  `.o98/.e98` (nh8), and `.o99/.e99` (HD128).
- Server-experiment forward `.o102/.e102`, `.o103/.e103`, `.o104/.e104`,
  and baseline backward `.o77/.e77` against historical `.o66/.e66`.
- Exact-function SASS in the baseline `mbarrier-contract-v2` trace.
- Current opcode classification, control-bit and operand parsers, scheduler
  dependency checks, and functional-unit execution paths.

## Corrections to earlier reasoning

1. PCs `0x5a50` and `0x5bc0` are `SYNCS.PHASECHK.TRANS64.TRYWAIT` in the
   executed causal HD64 function, not MUFU. Other functions in the same SASS
   file contain MUFU at those offsets. A PC-only search is insufficient.
2. NCU reports the MUFU contention samples in `mio_throttle`, not
   `math_pipe_throttle`, for this kernel. Counter-name analogy is not a
   definition-matched comparison.
3. The local binary `.ncu-rep` contains per-PC samples and opcode counts.
   The report can be opened on this Mac with the installed Nsight Python API;
   new hardware collection is not needed for this baseline evidence.
4. Correct total HGMMA counts do not establish correct warpgroup synchronization
   or completion dependencies. Those properties need independent validation.
5. `sole_fu_sfu`, `head_is_sfu`, and `total_num_warpcyc_stalled_fu_sfu`
   classify `SFU_OP`, not the MUFU mnemonic. F2FP is also classified as
   `SFU_OP`. Dividing these counters by MUFU count does not measure
   MUFU-only waiting or a MUFU-only queue length.

## Competing hypotheses

| Hypothesis | Discriminating evidence |
|---|---|
| H1: excess semantic work | NCU opcode counts versus trace counts for the same function |
| H2: SFU resource assignment/service cost is wrong | Opcode-to-pipe mapping, actual dispatch cadence, and hardware opcode/pipe samples |
| H3: warpgroup dependency reconstruction is incomplete | GSB operand, WARPGROUP decode, group state, generic scoreboard fallback, and HW group-wait samples |
| H4: trace-fixed persistent work distribution creates a long tail | Per-CTA equal-work cohorts and last-CTA timing |
| H5: local late-window SFU contention explains the whole residual | Critical-path attribution rather than selected first-late windows |

## 1. Strongest over-model: F2FP shares the MUFU SFU

### Production path

The actual Hopper map contains:

```cpp
{"F2FP", OpcodeChar(OP_F2FP, SFU_OP)}
{"MUFU", OpcodeChar(OP_MUFU, SFU_OP)}
```

`trace_config::set_latency()` gives both the SFU configuration `8,8`;
the opcode-aware wrapper only overrides NANOSLEEP, not F2FP.
`Subcore::get_fu()` sends both to the same `m_sfu_pipeline`.
Consequently, `F2FP.BF16.F32.PACK_AB` both takes the expensive MUFU service
time and competes with MUFU for the same single acceptance slot.

This is not a proposed explanation inferred from an old document. A temporary
compiled probe executed the unmodified production opcode map; the
resource/latency paths are in:

- `simulator-remodeled/gpu-simulator/ISA_Def/hopper_opcode.h:92`
- `simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc:1387-1389`
- `simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc:1435-1442`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc:2131-2133`

### Independent hardware evidence

NCU's executed-opcode totals and XU utilization give:

| Baseline kernel | MUFU warp instructions | F2FP packing warp instructions | Reconstructed HW XU warp instructions | Opcode-map SFU demand from MUFU + F2FP |
|---|---:|---:|---:|---:|
| Forward, NCU action 4 | 1926144 | 981504 | 1926936 | 2907648 |
| Backward, NCU action 9 | 1671168 | 1769472 | 1671168 | 3440640 |

The XU column is reconstructed, not a directly exported raw count:

```text
XU instructions =
  sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active / 100
  * sm__cycles_active.sum
  * 0.5 warp instructions / SM / cycle
```

The last factor is the H100 XU peak assumed for this reconstruction.
For forward the result differs from MUFU count by 792 instructions (0.041%);
for backward it equals MUFU exactly. It does not include the approximately
0.98M / 1.77M F2FP instructions. Thus the two-kernel hardware evidence is
incompatible with charging every BF16 pack to the same XU resource as MUFU.

PC sampling provides a separate check, using each metric's own correlation
IDs and the exact function's SASS:

| NCU sample reason | MUFU sites | F2FP sites |
|---|---:|---:|
| `mio_throttle` | 220 | 0 |
| `math_pipe_throttle` | 0 | 15 |
| `wait` | 266 | 3 |
| `selected` | 89 | 43 |
| All sampled reasons | 788 | 91 |

These finite samples are not exact stall-cycle counts or a service-rate
measurement. They nevertheless corroborate different execution paths.
The Nsight profiling guide describes MIO as including special math; its ALU
description explicitly identifies fast FP32-to-FP16 conversion on Ampere.
That description alone does not prove the exact Hopper BF16 destination
port or latency. Characterize that destination before implementing a remap;
do not globally move every conversion opcode or simply lower SFU II.

### Scale on the actual critical CTA

Baseline `.o90` ends on CTA43 at cycle 85171. Its protobuf contains, on
each of the four SMSPs:

```text
MUFU instructions: 3972
F2FP instructions: 2016
F2FP service incorrectly assigned to SFU: 2016 * 8 = 16128 cycles
Observed kernel excess: 85172 - 67838 = 17334 cycles
```

At the kernel mean, the corresponding F2FP service demand is
`981504 * 8 / (132 * 4) = 14871.27` cycles per SMSP. The simulator's mean
CTA lifetime is 76612.92 cycles versus HW `sm__cycles_active.avg=61147.42`,
a 15465.49-cycle gap.

The magnitude makes this a high-priority causal correction candidate.
It is not a prediction of a 16K-cycle speedup: SFU service overlaps other
work, the replacement execution resource is not free, and the critical CTA
can change. The small backward/HD128 aggregate error does not refute the
assignment defect; it does refute claiming this defect alone predicts every
shape's signed error. Other bottlenecks or compensating model errors must
be measured, not assumed.

## 2. Confirmed missing GSB completion dependency

The actual binary-captured HD64 function uses group markers and waits:

```text
0x5b70 HGMMA.64x128x16 ... R24, gsb0
0x5e70 HGMMA.64x64x16  ... R88, gsb0
0x5e90 WARPGROUP.DEPBAR.LE gsb0, 0x1
0x5ea0 FMNMX.FTZ R10, R24, R15, !PT
```

The production-parser executable reports:

```text
WARPGROUP category=BRANCH_OP explicit_depbar_path=false
gsb0 type=GENERIC has_reg=false; SB0 type=SB has_reg=true
0x5b70 wait_mask=0 new_read=0 new_write=0 stall=1
0x5e70 wait_mask=0 new_read=0 new_write=0 stall=1
0x5e90 wait_mask=0 new_read=0 new_write=0 stall=2
0x5ea0 wait_mask=0 new_read=0 new_write=0 stall=4
```

The complete source path explains why generic scoreboard logic does not
rescue this dependency:

1. `hopper_opcode.h:194` maps WARPGROUP to `BRANCH_OP`, not `DEPBAR_OP`.
2. `traced_operand.cc:133-135` treats `gsb0` as GENERIC, not a tracked SB.
3. `trace_driven.cc:1214-1245` has no OP_WARPGROUP control-flow case.
4. `Subcore::is_wait_barriers_ready_entry_point()` checks explicit dependency
   operands only when `inst->op == DEPBAR_OP`; generic control-bit wait masks
   at these sites are zero.
5. The trace metadata has `is_captured_from_binary=true`, while the run
   configuration has `is_remodeling_scoreboarding_enabled=0`. In
   `subcore.cc:738-756`, this selects compiler hints instead of register RAW/WAR
   scoreboarding.
6. The tensor pipe is selected per subcore. There is no modeled group-commit
   queue or four-warp completion identity on the inspected path.

The hardware behavior is not hypothetical. NCU records 58176 dynamic
`WARPGROUP.DEPBAR.LE` instructions.
At these sites there are 133 barrier samples, including 73 samples when the
scheduler issued nothing. HGMMA sites also have 65 warpgroup-arrive samples.
All inspected simulator forward runs print `ncu_stall_warpgroup_arrive=0`;
that counter alone would not prove the omission, but the decoded path does.

This invalidates the earlier claim that an ordinary `gsb0`/DEPBAR dependency
already makes the tensor model effectively correct. Matching HGMMA counts
or average pipe utilization cannot validate group ordering.

Qualification: this is an independently established fidelity defect, not
proof that adding group waits will reduce kernel cycles. Missing waits can
make local paths too fast while changing downstream competition. The HD128
trace also has six static WARPGROUP.DEPBAR sites, so its small cycle error
does not establish correct group semantics. Do not repair this with mbarrier
generation forcing, artificial producer delays, or a PC whitelist.

## 3. SFU issue cadence has an extra stage bubble

The pre-patch path was rechecked on 2026-09-11 after committing/pushing the
Opt 13 records as `3d1e0c8`. That baseline imposes an `II + 1` acceptance
interval under a saturated, independent instruction stream with no downstream
stalls. For the configured II=8, this is 9 cycles. This is a source-derived
conditional result; the copied FA3 logs do not contain per-instruction SFU
acceptance timestamps.

The user subsequently authorized the correction below. The source patch is
implemented, but no build or simulator run was performed. No configuration
was changed. The user approved committing and pushing this patch on 2026-09-11.

### Actual path and cause

The evidence table refers to the pre-patch source at `3d1e0c8`.
Paths are relative to
`simulator-remodeled/gpu-simulator/gpgpu-sim/src/`:

| Evidence | Code location | Consequence |
|---|---|---|
| Phase order | `gpgpu-sim/remodeling/subcore.cc:107-116` | `execute()` precedes `control_stage()`, which precedes scheduler `issue()` |
| Dispatch countdown initialization | `abstract_hardware_model.cc:88-100` | `warp_inst_t::issue()` sets `cycles = initiation_interval` at selection |
| Countdown semantics | `abstract_hardware_model.h:1672-1675` | Decrement, then test remaining cycles; 1 -> 0 permits movement in that same call |
| SFU is non-fixed despite `has_queue=false` | `gpgpu-sim/remodeling/functional_unit.cc:109-110,543-557` | `m_is_sfu=true`; no fixed-latency allocation/early reservation path |
| Scheduler gate added by Opt 10 | `gpgpu-sim/remodeling/subcore.cc:769-784` | With warp-switch enabled, no-queue SFU candidates must pass current `can_issue()` |
| Actual control-stage transfer | `gpgpu-sim/remodeling/subcore.cc:373-397` | The selected SFU instruction reaches dispatch one cycle later; `can_issue()` is checked again |
| SFU readiness override | `gpgpu-sim/remodeling/functional_unit.cc:556-558` | Tests only `m_dispatch_reg->empty()`, not future capacity |
| Execution entry | `gpgpu-sim/remodeling/functional_unit.cc:289-337` | After countdown, dispatch moves to `m_pipeline_reg[latency-1]` only if that stage is empty |
| Fixed-FU reservation boundary | `gpgpu-sim/remodeling/subcore.cc:2019-2035` | `reserve_unit()` is called only for fixed-latency FUs, not SFU |

The baseline `.o102` enables `intra_smsp_warpswitch_enable=1` and sets
trace SFU latency/II to `8,8`. Its separate `sfu_latency=21` is the allocated
FU pipeline depth, not this decoded instruction's latency or initiation
interval. The current SFU uses the dispatch-held instruction's `cycles`;
the generic `m_dispatch_pending_reserved_cycles` is not its active II gate.
Neither that generic reservation counter nor `stall_count >>= 1` explains
this particular extra cycle.

Numbering the first selection as cycle 0:

```text
cycle 0: scheduler selects A into ISSUE_CONTROL; A.cycles = 8
cycle 1: execute has already run; control transfers A to SFU dispatch
cycle 2: execute decrements A.cycles 8 -> 7
...
cycle 8: execute decrements A.cycles 2 -> 1
         scheduler rejects B: A is still in dispatch now
cycle 9: execute decrements A.cycles 1 -> 0 and moves A to pipeline
         control has no B to transfer
         scheduler finally selects B into ISSUE_CONTROL
cycle 10: control transfers B to SFU dispatch
```

Countdown takes exactly eight execute visits, not nine. The wasted slot is
cycle 9's control-stage refill: B could have been selected at cycle 8 for
arrival at cycle 9. The scheduler incorrectly applies the control stage's
"can receive now" predicate to an instruction that arrives one cycle later.
Opt 10 correctly prevented early blocked SFU instructions from occupying the
shared issue latch, but this phase-insensitive guard also rejects the last
safe lookahead cycle.

`next_stage_blocked_by_sfu=0` cannot detect this loss: B has not entered the
issue latch yet. It is rejected as an FU-unavailable warp instead. This also
does not mean every interval in FA3 is nine cycles; dependency, instruction
fetch, other winners, and downstream stalls can make intervals longer.

### Implemented correction

The patch preserves phase order, II, instruction latency, and FU routing,
and separates the two meanings of readiness:

1. `can_issue()` remains a strict current-dispatch-empty check, used by the
   actual control transfer.
2. `can_issue_at_next_control()` is a new read-only scheduler query. The
   base implementation delegates to `can_issue()`; only SFU overrides it.
   The scheduler uses it under the existing
   `intra_smsp_warpswitch_enable && !fu->get_has_queue()` non-fixed-FU gate.
   Fixed-latency and queue-based paths retain their previous checks.
3. If dispatch is empty now, the existing rule suffices. If occupied,
   the predictor requires `get_num_cycles() <= 1` and proof that the current instruction
   can leave dispatch in the next `execute()` call. A remaining value of
   zero is possible when a prior pipeline-entry attempt was blocked.
4. The predictor uses the dispatch-held instruction's own `latency-1`
   destination, rejecting zero latency or latency beyond allocated depth.
   It assumes no output retirement/predicate-stage drain. Under the actual
   ascending shift, the destination is guaranteed empty exactly when there
   is an existing vacancy at or below it and either it is the last stage or
   the immediately higher stage is empty. That second condition prevents a
   higher-stage instruction from refilling the destination. The code checks
   occupancy directly without allocating or mutating a simulated pipeline.
5. If the guarantee fails, the scheduler skips that warp and can select
   another ready warp. Uncertain SFU work does not enter the shared latch.

Implementation locations under `gpgpu-sim/remodeling/`:

| File / function | Change |
|---|---|
| `functional_unit.h` | Declare the virtual next-control query and SFU override |
| `functional_unit.cc`: `functional_unit::can_issue_at_next_control()` | Default to existing readiness for other FU types |
| `functional_unit.cc`: `functional_unit_sfu::can_issue_at_next_control()` | Check countdown and conservative next-execute entry capacity |
| `subcore.cc`: `Subcore::issue()`, `capture_pre_issue_snapshot()` | Use the same gated next-control eligibility query |
| `subcore.cc`: `Subcore::control_stage()` | Retain strict current readiness and assert the promised no-queue transfer succeeds when warp-switch is enabled |

The conservative assumption avoids predicting shared result-port/RF progress.
SFU writes to the one-entry `m_EX_WB_sm_variable_latency_latch`, shared with
`MISC_QUEUE` (`subcore.cc:2449-2453`, `subcore.h:301`). A full result latch
can eventually fill the pipeline and prevent dispatch from leaving even
after countdown reaches zero. SFU has no fixed-latency result-queue credit
reservation. This is why `remaining <= 1` alone is not a sufficient guarantee.

The current single-issuer design already gives the winning instruction an
exclusive reservation: it occupies the one-entry `ISSUE_CONTROL` latch,
and the next control stage services it before any new selection. SFU is
not written by `read_rf()`, because that path is fixed-latency-only.
A separate SFU queue or reservation countdown is unnecessary for this
one-cycle lookahead. No new mutable reservation state or winner tag was
added. The control-stage assertion checks the existing latch's guarantee.
Even with assertions disabled, the actual transfer still requires strict
`can_issue()` success, so the patch does not permit an occupied dispatch
register to be overwritten.

Both `capture_pre_issue_snapshot()` and the authoritative issue scan,
including its read-only tail, now use the same scheduler-facing query.
This keeps STALLALIGN eligibility and actual selection on the same rule;
runtime winner/population conservation still requires the server run.

| Event | Pre-patch saturated II=8 | Implemented phase-aware II=8 recurrence |
|---|---|---|
| Scheduler selection | 0, 9, 18, 27 | 0, 8, 16, 24 |
| SFU dispatch acceptance | 1, 10, 19, 28 | 1, 9, 17, 25 |
| Execution-pipeline entry (latency=8) | 9, 18, 27, 36 | 9, 17, 25, 33 |
| FU result-latch arrival, unblocked | 17, 26, 35, 44 | 17, 25, 33, 41 |
| RF writeback/retirement, unblocked | 18, 27, 36, 45 | 18, 26, 34, 42 |

Each instruction still spends eight execute calls in dispatch and eight
cycles in the modeled execution pipeline. An isolated instruction follows
the same schedule in both versions. The table is a mechanism recurrence,
not a claim that hardware MUFU dependent latency is 18 cycles: RF conflicts
and end-of-SM-cycle write-barrier updates remain separate in the real model.

Rejected shortcuts:

- `II=7` or decrementing the instruction earlier also changes its own
  dispatch/read-barrier/result timing; it is not just removing a refill gap.
- Disabling Opt 10 restores early latch occupancy and the head-of-line block.
- Globally making `can_issue()` return true with a nearly finished dispatch
  instruction also affects the actual control transfer, which would overwrite
  occupied dispatch when downstream progress fails.
- Reordering the whole subcore/SM tick changes unrelated FUs, writeback,
  dependencies, and TMA timing. No such reordering is needed.

### Source-delivery verification and remaining checks

This subsection records verification at source delivery on 2026-09-11.
The subsequent server-log comparison follows below.

A file-free Node check extracted the actual SFU predictor body and evaluated
it as JavaScript after limited C++ type/member-access syntax substitutions.
Separate recurrence checks followed the production phase order, dispatch
countdown, 21 allocated stages, one-entry output latch, and variable output
readiness. Neither check executes compiled C++ or the full scheduler,
trace, dependency, or WaitBlock system.

- Pre-patch II=8 produced selection `0,9,18,...` and acceptance `1,10,19,...`.
  The implemented conservative next-control query produced `0,8,16,...` and
  `1,9,17,...`, without changing the 18-cycle isolated select-to-retire
  recurrence under unblocked output.
- Twenty combinations of II `{1,2,4,8,16}` and decoded latency `{1,2,8,21}`,
  plus four output-stall/intermittent-demand cases, preserved all 32 input
  IDs per run in order without duplicate retirements or broken lookahead
  guarantees.
- For the tested II values greater than one under free output, the candidate
  reaches the configured interval. At II=1, the conservative no-drain assumption
  sometimes leaves two-cycle gaps; this is a known conservative limitation,
  not evidence for generalizing an II=8 fix to every possible configuration.
- Enumerated all pipeline occupancy masks and target stages at depths 1-12
  with countdowns `{0,1,2,8}`: 360,456 states matched a no-drain reference
  shift. An additional 90,114 implication checks found that permitting an
  output drain never invalidated a no-drain acceptance guarantee.
- Checked empty dispatch, invalid latency/depth, zero-countdown backpressure,
  and higher-stage refill boundaries. In particular, an empty target with an
  occupied immediately higher stage does not by itself permit lookahead.
- Source guards confirmed unchanged `functional_unit_sfu::can_issue()`,
  `functional_unit::cycle()/issue()/reserve_unit()`, and
  `Subcore::cycle()/allocate()/read_rf()/issue_warp()/get_fu()`. New declarations
  match definitions, and scheduler/snapshot use the same gated predictor.
- An initial blanket expectation that the conservative query always reaches
  II, including II=1, failed. It was narrowed as above rather than hiding
  the downstream-capacity restriction.

No test scripts, probe executables, or generated data files were added.
Local IDE diagnostics are blocked by missing CUDA `vector_types.h`; source
checks do not establish C++ compilation success. The user will rebuild the
simulator on the server and run FA3 directly. Trace regeneration, new
configuration flags, and tracer rebuild are not required by this patch.

The server comparison must freeze trace/sidecar contents and all configuration
values against a control that already includes Opt 13. Check final
`gpu_sim_insn` bit identity, existing WaitBlock/TMA/STALLALIGN conservation,
normal completion, and signed cycle error, including backward regressions.
The patch applies with `intra_smsp_warpswitch_enable=1`; comparing against
that flag disabled would also remove Opt 10 and would not isolate this change.

Per-instruction selection/acceptance/entry/result/writeback timestamps remain
unmeasured in production. A separately scoped observe-only FA3 probe can
quantify the interval histogram if needed. No such instrumentation was added,
and no runtime preservation or cycle-accuracy improvement is claimed yet.

An II=8 stream limited only by this mechanism loses 1/9 (11.11%) of ideal
acceptance throughput; restoring the interval raises that stream's rate by
12.5%. Neither figure is a predicted FA3 kernel speedup. The critical-path
fraction and actual H100 cadence still need evidence, especially because
baseline backward already slightly underpredicts hardware cycles.

### First SFU server experiment: reduced contention, negligible cycle change

Checked on 2026-09-14. The latest local baseline pairs are K5 `.e106/.o106`
and K10 `.e80/.o80`, superseding the user's suggested `.e102/.o102` and
`.e77/.o77`. Compare against those immediately preceding post-F2FP runs,
not the pre-F2FP `.o90/.o66` controls.

All four runs completed normally. Hardware references remain the
`FA3_multi.md` elapsed-cycle references, K5 67,838 and K10 130,641.

| Kernel | Previous -> latest | Cycles before -> after | Change | Signed HW error before -> after |
|---|---|---|---|---|
| K5 forward | `.o102 -> .o106` | 71800 -> 71863 | +63 (+0.0877%) | +5.84% -> +5.93% |
| K10 backward | `.o77 -> .o80` | 128702 -> 128217 | -485 (-0.3768%) | -1.48% -> -1.86% |

Pair MAPE moves from 3.66% to 3.89%. Backward becomes slightly faster but
less accurate because it was already below the hardware reference.
This is not a measured FA3 speedup/accuracy win.

#### Work and audit checks

- `gpu_sim_insn` is bit-identical within each pair: K5 447,604,920 and
  K10 623,703,264. Retired warp instructions also match: 14,447,843 and
  19,912,645 respectively.
- All sorted per-CTA `[TRACECHECK]` records match exactly: 132 records for
  K5 and 384 for K10. This includes instruction, active-lane, memory-record,
  synchronization, HGMMA, MUFU, and TMA counts, not just the aggregate
  warp-instruction total.
- All 427 printed configuration key/value entries match within each pair,
  including `intra_smsp_warpswitch_enable=1`, INT `2,2`, and SFU `8,8`.
  TMA/synchronization/WaitBlock loader summaries also match, including
  5,172 exact-ticket streams and 817,199 ticket slots.
- WaitBlock summaries are all PASS: 132 for each K5 run and 384 for each
  K10 run. The last cumulative snapshot per SM gives balanced hold/release
  and replay accounting, zero residual active state, and zero watchdog
  aborts or generation overshoots.
- Each run has 132 PASS TMA final audits, with matching created/completed
  bindings and assigned/completed byte credits. Error and residual-state
  fields are zero. `uid_reorders` is an informational out-of-order completion
  counter, not a zero-required failure field in the production audit.
- STALLALIGN FINAL is PASS with zero population, selected, sampled, and
  winner mismatches. Occurrence identities are unchanged:
  K5 `14818470 = 14447843 + 370627`;
  K10 `20997218 = 19912645 + 1084573`.
- Neither log pair contains a reported assertion failure, dispatch
  overwrite, fatal/abort marker, or segmentation fault.

These checks pass the cross-run instruction-count gate for this experiment.
The earlier pre-F2FP input mismatch does not apply to these two pairs.
Nevertheless, equal input summaries are not cryptographic identity of all
trace/sidecar bytes. Both binaries print only `*_modified_0.0`, not a commit
SHA, so the logs alone cannot prove an SFU-only binary diff. Local history
also contains the later WaitBlock watchdog change `369693c`.

#### Resource and critical-path observations

| Metric | K5 before -> after | K10 before -> after |
|---|---|---|
| SFU-blocked warp-cycles | 12452184 -> 8784917 (-29.45%) | 2997352 -> 1954253 (-34.80%) |
| Scheduler cycles with an SFU-blocked candidate and no issue | 5155852 -> 3757159 (-27.13%) | 1668043 -> 1168652 (-29.94%) |
| SFU-active subcore-cycles | 20035269 -> 19087946 (-4.73%) | 19734837 -> 19515756 (-1.11%) |
| SFU holding the shared issue latch | 0 -> 0 | 0 -> 0 |

The signature is consistent with the new eligibility rule: fewer candidates
are classified as SFU-blocked, without reintroducing SFU head-of-line latch
stalls. These counters do not directly measure an 8-cycle acceptance
interval or independently establish the hardware cadence.

For K5, reduced SFU blocking coexists with more wait-barrier warp-cycles
`2830815 -> 3694558` (+30.51%), tensor-FU blocking
`7489591 -> 8007531` (+6.92%), and other-FU blocking
`27886681 -> 29092368` (+4.32%). At the issue-stage level,
`no_warps_ready` decreases by 394,362 cycles, while `no_valid_instruction`
increases by 398,816 cycles, nearly offsetting that reduction. Warp-cycle
blocker counters can overlap; do not add them as exclusive kernel time.

K5's previously last CTA 2 finishes 568 cycles earlier, but CTA 39 finishes
750 cycles later and becomes the last CTA. Mean CTA finish improves by only
88.4 cycles, with 75 CTAs earlier and 57 later; the kernel ends 63 cycles
later. K10's last CTA changes from 284 to 305. Schedule and tail changes
therefore matter more than the aggregate SFU-stall reduction suggests.
This is an observed redistribution, not a complete causal attribution of
the remaining forward error.

Conclusion: instruction-count preservation and existing runtime audits pass;
SFU contention metrics improve, but no meaningful kernel-cycle or accuracy
gain is established. Do not claim the II+1 bubble explains the remaining
forward residual, retune II/latency from these results, or mark a new
optimization Done solely because the runs complete. Direct cadence
validation would require separately scoped timestamp evidence; it is not
required to close this investigation under the user's decision below.

Source locations use the baseline run directory from `FA3_multi.md`:
K5 `.o102/.o106:4175-4179` (final audit/cycles/instructions),
`:4693-4709` (warp blockers/frontend), `.o106:4141` (last CTA);
K10 `.o77:8021-8025` and `.o80:8020-8024`,
`.o77:8541-8557` / `.o80:8540-8556`, and `.o80:7986` (last CTA).
Loader summaries are `.e102/.e106/.e77/.e80:7-9`.
No simulator code, configuration, results table, or progress status was
changed during this analysis; no temporary files were created.

#### Final disposition: retain correction and close investigation

User decision on 2026-09-14:

- Retain the SFU next-control acceptance correction in `10505f2`, with the
  existing II/latency and configuration unchanged. Record it as a retained
  timing-model correction, not a demonstrated FA3 speedup or accuracy gain.
- Close this SFU investigation. Do not pursue additional SFU tuning,
  timestamp instrumentation, microbenchmarks, or reruns for this scope.
- Do not add Opt 14 or otherwise change `.result/FA3_progress.md` for this
  correction. The accepted Opt 13 entry remains unchanged.
- Preserve the validation limits: no direct production measurement of
  8-cycle acceptance, no exact logged binary SHA, and no full trace/sidecar
  byte-identity proof. Closure is a scope decision, not a claim that those
  measurements were performed.
- Keep measured results in `.result/FA3_multi.md`; keep the cause,
  implementation, experiment interpretation, and retention decision in
  this document. No separate SFU report is needed.

Further work requires a new request. The deferred opcode candidates and
WGMMA proposal remain documented, but this closeout starts neither.

## 4. Work and critical-path controls

The baseline is function-matched independently:

- NCU mangled name exactly equals the simulator name after removing `___0`.
- 943/943 NCU sampled function-relative PCs match the trace opcode, including
  modifiers. NCU PC base is `0x7f76309bac00`.
- NCU and trace have identical MUFU (1926144) and HGMMA (349056) counts.
- Parsing all 132 baseline CTA protobufs also gives exactly 981504
  `F2FP.BF16.F32.PACK_AB`, 58176 `WARPGROUP.ARRIVE`, and 58176
  `WARPGROUP.DEPBAR.LE`, matching NCU. Every compute opcode/modifier total
  checked by the evidence script matches.
- Total simulated retired warp instructions are 14448163 versus NCU's
  opcode sum 14473188 (-0.173%); thread instructions are 447612504 versus
  448134746 (-0.117%). Aggregate work inflation is not the 25.55% gap.

The latest critical CTAs also show why selected late windows are insufficient:

| Run | Last CTA | Final CTA finish | First-late action cycle on that CTA | Post-TMA first-late window | Whole-kernel excess |
|---|---:|---:|---:|---:|---:|
| s1024 `.o97` | 37 | 44441 | 29889 | 356 | 6289 |
| nh8 `.o98` | 9 | 70486 | 36155 | 2544 | 12898 |
| HD128 `.o99` | 109 | 90734 | None | None | 3823 |

Even on the last CTA, the sampled first-late event is well before completion.
Its stall fraction cannot be multiplied by total kernel cycles.

The baseline's shortest CTA is 71444 cycles, longer than HW's maximum
SM-active lifetime 63556. Mean lifetimes are 76612.92 versus 61147.42
(+25.29%). Although lifetime definitions differ at boundaries, this is not
just one anomalous final CTA or a launch-overhead artifact.

Trace-fixed work remains a confounder: nh8 has 44 zero-tensor CTAs, and
its 2160-HGMMA cohort averages 67836.63 cycles while the larger 2304-HGMMA
cohort averages only 64253.00. Tensor count alone does not identify identical
work, pipeline phase, epilogue, or initialization cost.

## 5. Decision and next experiment

### Agreed order and authorization

User-selected investigation/implementation order on 2026-09-10:

```text
problem 1: F2FP execution resource and cost
  -> problem 3: SFU issue-stage acceptance bubble
  -> problem 2: WGMMA group completion dependencies
```

The user first authorized the F2FP source change, explicitly without a build.
That source change is implemented as recorded in section 6.8.
After the server experiment, the user closed that scope as Opt 13 and
requested its full progress entry on 2026-09-11. Other operator changes
remain deferred in the existing progress document's Arch TODO-3.
The user subsequently authorized the SFU next-control acceptance correction
and ran the server experiment. That source change was committed/pushed as
`10505f2` after approval on 2026-09-11. Section 3 records the implementation
and the 2026-09-14 log comparison: instruction-count/audit checks pass,
kernel accuracy does not improve, and exact cadence remains unmeasured.
The user then closed the SFU scope with the correction retained, no further
tuning/instrumentation, and no Opt 14 entry. WGMMA dependency changes remain
proposals and are not authorized or started by this closeout.

This order separates wrongly assigned SFU demand from the SFU's own service
cadence, before changing cross-warp dependency timing. Stage A must leave the
SFU handshake and WGMMA behavior unchanged; Stage B must leave the accepted
F2FP mapping and WGMMA behavior unchanged. Stage C is the larger dependency
model change and is evaluated only after those two effects are isolated.

### Stage A / problem 1: characterize and remap BF16 packing

Objective: execute `F2FP.BF16.F32.PACK_AB` on its measured H100 resource with
the measured service interval and result availability, without changing
the program's instruction stream.

The code-level destination candidate is now explicit: H100 ALU capacity,
represented by `INTP_OP -> Subcore::get_fu() -> m_int_pipeline` under the
observed configuration. Section 6 records the broader opcode audit and the
evidence/limitations behind this choice. The first candidate should inherit
the existing INT `2,2` timing, with no new F2FP-specific latency knob.
Measurement decides whether that reuse is valid; it does not automatically
require a new timing value. No family-wide remap is included.

Latest user decision: skip standalone simulator microbenchmarks for this
first patch. After inexpensive classification/source checks and a successful
server build, the user will run FA3 directly. Keep the existing INT `2,2`
setting. The hardware characterization below is retained as a deferred
diagnostic procedure, not a prerequisite for that first FA3 comparison.

Deferred hardware characterization:

1. Verify that a small conversion kernel actually emits the same SASS
   mnemonic/modifiers and active-lane pattern as FA3. Prevent dead-code
   elimination and inspect the generated SASS; a source-level cast alone
   is not enough to identify the instruction being timed.
2. Use many independent conversions to measure throughput, and collect raw
   ALU/FMA/XU instruction counts plus their peak definitions where available.
   Use the existing XU reconstruction only as supporting evidence.
3. Measure a real producer-consumer dependency chain separately for latency.
   F2FP consumes FP32 and produces packed BF16; any unpack/reconversion or
   timing scaffold in a loop needs a matched control. Do not divide a
   round-trip latency and call it the F2FP latency without isolating the
   other instructions.
4. Compare F2FP+MUFU, F2FP+ALU, and F2FP+FMA mixes against the isolated cases.
   This checks shared-resource contention, not just a pipe label. Amortize
   loop/timing overhead, repeat measurements, record clocks, and avoid
   full-NVBit callback timing as the hardware reference.

First-candidate implementation boundary:

- Recognize the architecture and exact conversion form; do not change the
  base `F2FP` entry for all formats, rounding modes, architectures, or other
  conversion opcodes without evidence covering them.
- Keep semantic opcode, operands, masks, PCs, and instruction accounting
  unchanged. Resolve execution resource and timing consistently with both
  static instruction metadata and dynamic decode. The existing opcode-aware
  latency wrapper receives only the base opcode ID, so exact-form selection
  needs the existing modifier information or a validated derived tag.
- Reuse an existing FU when its measured sharing/ports match the hardware.
  An NCU "ALU" label is not by itself a reason to assign `INTP_OP`: inspect
  `Subcore::get_fu()` and the configured SP/INT sharing rules. Do not invent
  a free dedicated conversion pipe merely to remove contention.
- Translate hardware issue-to-result latency into the simulator's existing
  stage conventions. Its `latency` field is not necessarily the complete
  observed latency. Account for operand-read, dispatch, and result/writeback
  stages once, and preserve any ordinary SB dependencies.
- Leave MUFU II/result latency, SFU acceptance, tensor timing, and WaitBlock
  semantics unchanged during this stage.

Before the first full FA3 run: check target/non-target classification and
static/dynamic agreement, then build on the server. No separate F2FP/INT
simulator microbenchmark is required.

First-run acceptance checks: BF16-pack instructions use INT rather than SFU,
instruction work/counts and WaitBlock/TMA conservation are preserved, and
FA3 completes without new failures. Record signed cycle errors against the
same NCU reference and frozen control inputs. A smaller aggregate error is
useful evidence, but does not independently certify F2FP latency/throughput
or every resource-sharing detail. Reopen the deferred measurements if needed;
do not fit a new latency to the FA3 result.

### Stage B / problem 3: fix a demonstrated acceptance bubble

Objective: preserve the configured SFU service time while removing a
simulator-only idle slot at the issue/control boundary.

Status: retained and investigation closed by the user on 2026-09-14; not
registered as Opt 14. The conservative next-control predictor and
scheduler/snapshot integration remain implemented. Section 3 records the
implementation, verification, experiment, and final disposition. No
latency/II setting, F2FP routing, WGMMA dependency, WaitBlock behavior, queue,
or phase ordering was changed.

Server experiment: section 3 compares `.o102 -> .o106` and `.o77 -> .o80`.
Both retain `intra_smsp_warpswitch_enable=1` and pass count/audit checks,
but kernel cycles change negligibly and absolute error increases slightly.
No standalone simulator microbenchmark or new timestamp instrumentation
is included in this implementation.

```text
pre-patch: select A at 0 -> accept A at 1 -> select B at 9 -> accept B at 10
patched:   select A at 0 -> accept A at 1 -> select B at 8 -> accept B at 9
```

This is a saturated II=8 source-model recurrence with downstream capacity,
not a hardcoded FA3 schedule or measured hardware cadence.

The observed runs satisfy instruction-count, conservation, normal-exit, and
zero-SFU-latch-blockage checks. This does not individually prove dependent
result timing or every backpressure boundary in compiled code. Exact binary
provenance and timestamped acceptance remain unverified. These limits remain
in the record, but no additional SFU validation or tuning is scheduled.

### Stage C / problem 2: model group identity and completion

Status: deferred proposal. Retaining and closing Stage B does not authorize
or start this dependency-model change.

Objective: let unrelated work overlap asynchronous tensor execution, but
prevent dependent progress until the specific committed WGMMA groups have
completed. This stage is about fidelity, not guaranteed cycle reduction.

First establish the compiler/ISA contract with small kernels using
`wait_group<0>` and `wait_group<1>`, several operations per committed group,
and multiple groups in flight. Match those kernels to SASS:

- `WARPGROUP.ARRIVE` ordering/fence behavior;
- HGMMA group-end/commit markers such as `gsb0`;
- `WARPGROUP.DEPBAR.LE` and its outstanding-group limit.

Do not equate these with ordinary per-warp SB counters or TMA mbarriers.
Confirm that the existing trace contains the required identities and
modifiers. If it does not, fail explicitly and obtain authoritative metadata;
do not synthesize missing relationships from a kernel/PC whitelist.

Proposed logical model:

```text
identity = (launch, CTA, warpgroup, GSB identifier, dynamic group sequence)

issued HGMMA contributions
  -> assembled logical operation / committed group
  -> completion of all required contributions
  -> retirement of the completed group
  -> release of the waits whose committed-group frontier is satisfied
```

Key design points:

- Preserve all four per-warp HGMMA trace instructions and their counts.
  Associate matching contributions with one collective operation; do not
  solve the problem by discarding three instructions or multiplying tensor
  throughput. Existing FU completions can feed the logical completion record
  before any separate tensor-rate recalibration.
- Track the committed-group frontier at each wait. A limit of 0 drains all
  preceding committed groups; a limit of 1 permits only the architecturally
  allowed newest group to remain pending. Future commits must not retarget
  an already-issued wait.
- A fast peer must not make the collective group complete while another
  required warp contribution is absent or incomplete. However, do not force
  all four warps to issue in the same cycle or add barriers around unrelated
  softmax work unless the ISA requires it.
- Ordinary mbarrier/TMA completion identity and release order stay intact.
  GSB readiness is a separate condition, not a replacement for those checks.

Implement and test group bookkeeping separately from timing enforcement
when this stage is authorized. A shadow tracker may first validate dynamic
group identities without modifying scheduler decisions; it is not itself
the completed dependency fix.

Tests: empty group/wait, limits 0 and 1, multiple operations/groups, a delayed
peer warp, completion reordering, multiple independent warpgroups/CTAs,
launch reuse, and zero outstanding group state at retirement. Dynamic group
membership and wait release must be asserted directly, not inferred from a
nonzero diagnostic stall counter.

Pass criterion: waits release exactly for completed dependencies and useful
asynchronous overlap remains. Cycles may rise when omitted waits are restored.
Do not compensate by forcing generations, inflating/reducing TMA latency, or
loosening a lifecycle check.

### Comparison protocol and stopping rules

Use a reproducible same-input series, not the historical `.o90` as an
uncontrolled patch baseline. The B0/B1 F2FP isolation below remains a
validation follow-up after the user-accepted Opt 13 completion; future
SFU/WGMMA pairs must freeze a fresh control that already includes F2FP:

| Snapshot | Change from predecessor | Purpose |
|---|---|---|
| B0 | Frozen current build/config/trace/sidecars | Reproducible control |
| B1 | Accepted Stage A only | Isolate F2FP assignment/cost |
| B13 | B1 plus accepted Stage B | Isolate SFU handshake timing |
| B132 | B13 plus Stage C | Isolate warpgroup dependency fidelity |

Record binary/source identity, configs, trace/sidecar hashes, and repeated-run
variation. Keep unrelated ongoing partial-join work fixed in each pair.
Use observe-only instrumentation separately from behavioral commits.

For the first F2FP patch: classification/source checks and a server build,
then user-run FA3 directly, without a standalone simulator microbenchmark.
Use completed baseline K5/K10 and the s1024/nh8/HD128-forward controls.
Validation for later SFU/GSB stages is considered separately.
Failed HD128-backward and unqualified
noncausal cases remain excluded from accuracy averages; any new failure in
a previously completed control is investigated rather than silently excluded.

Mandatory gates are baseline-identical `gpu_sim_insn`, identical semantic
instruction streams/counts, balanced WaitBlock/TMA events, no residual state,
and no forced releases. A changed count or newly failed proof invalidates the
timing comparison until explained and resolved; do not normalize it away.

Report per-opcode FU assignments, acceptance intervals, dependency latency,
critical CTA/role timing, and signed cycle errors. Aggregate error can worsen
when an over-model and an under-model previously canceled each other. In that
case retain the hardware evidence, flag the regression, and investigate;
do not call the result accurate or tune another mechanism to cancel it.

Full-simulator builds/experiments require the CUDA server. None were started
for the authorized source change. No new scripts or auxiliary documents were
created. Keep this Markdown as the record and remove future temporary
non-Markdown analysis artifacts after use.

## 6. Stage A detail and complete Hopper routing audit

### Scope and evidence levels

This is a source-level audit of all 212 unique entries in
`gpu-simulator/ISA_Def/hopper_opcode.h`, across 27 uarch categories, and their
decode, FU selection, and timing paths in the remodeled simulator.
All paths below are relative to `simulator-remodeled/` unless stated otherwise.

The official Hopper instruction table has 204 entries; all are present in
the map. The map additionally has `CSMTEST`, `F2FP`, `LDCU`, `R2B`, `RED`,
`RTT`, `VOTE_VTG`, and `WGMMA`. An extra entry is not automatically wrong:
F2FP is directly present in the measured SASS despite its omission from that
official table.

The two existing local NCU reports, the baseline `*_full_rpt.ncu-rep` and
`nv_reports/out_test.ncu-rep`, contain 22 kernel records in total. Their
positive executed-opcode counters cover 82 distinct base opcodes and 170
opcode/modifier forms. All 82 base opcodes resolve in the current map.
These are two reports of the same workload family, not 22 independent
architectural microbenchmarks. The remaining 130 mapped base opcodes have
no dynamic observation in this corpus.

Evidence labels used below:

- HW: observed hardware counters/decoded SASS contradict the present route
  for the specific measured form.
- Source: the code assigns a different operation class or loses a modifier;
  no hardware cycle contribution is established.
- Candidate: documentation or cross-kernel accounting suggests another
  route, but an isolated hardware experiment is still required.
- Open: inspected route, insufficient evidence to declare correct or wrong.

Source coverage is complete for this map. Hardware validation of every
instruction form, architecture, latency, and contention rule is not complete.
No correction was applied during the audit itself. The later first F2FP
implementation is recorded separately in section 6.8.

### 6.1 F2FP destination in the actual code

Recommended target to validate first:

```text
F2FP.BF16.F32.PACK_AB on H100
  -> ALU execution resource
  -> uarch dispatch category INTP_OP
  -> Subcore::get_fu()
  -> m_int_pipeline
```

The last step is specific to the actual configuration:

```text
is_fp32_and_int_unified_pipeline = 0
is_fp32ops_allowed_in_int_pipeline = 1
trace_opcode_latency_initiation_int = 2,2
trace_opcode_latency_initiation_sp = 2,2
trace_opcode_latency_initiation_sfu = 8,8
```

`INTP_OP` is an execution classification, not a request to replace the
floating-point conversion with integer arithmetic. The raw mnemonic,
`OP_F2FP` identity, FP32 inputs, packed BF16 output, masks, and dynamic
instruction occurrences must remain unchanged.

`Subcore::get_fu()` currently implements:

```cpp
case PREDICATE_OP:
case INTP_OP:
  if (m_config->is_fp32_and_int_unified_pipeline)
    fu = m_sp_pipeline;
  else
    fu = m_int_pipeline;
  break;
```

Do not substitute `ALU_OP` merely because the hardware calls the resource
ALU: the remodeled `get_fu()` switch has no `ALU_OP` case. `SP_OP` is also
not equivalent to an ALU-only route. With the active configuration it first
selects INT when available and the opcode is not IMAD, otherwise SP.
The same fallback also applies to `HALF_OP` because it falls through to the
SP case. A resource must not gain an unmeasured alternative execution pipe.

The first validation candidate should reuse the existing `2,2` INT setting
unchanged. `trace_config::set_latency()` already assigns `int_latency` and
`int_init` for `INTP_OP`; no separate F2FP timing parameter is needed if this
complete execution path matches hardware.

The precise insertion point is normal instruction dispatch, not an arbitrary
internal FU stage:

```text
decode exact F2FP form as the validated ALU execution class
  -> normal scheduler issue and INT resource reservation
  -> ISSUE_CONTROL
  -> CONTROL_ALLOCATE
  -> fixed-latency register-read stages
  -> existing INT FU dispatch register
  -> existing INT execution stages
  -> regular fixed-latency result queue and writeback
```

No new pipe or bypass is proposed. F2FP shares the existing INT capacity with
other users under the current H100 configuration.

The first `2` is the instruction's FU latency field; the second is its
initiation interval. They are not the whole issue-to-dependent-use delay.
`sm.h` defines one control stage, one allocation stage, and three regular
register-read stages. `Subcore::allocate()` also uses:

```text
target_latency_execution =
    latency_read_fixed_latency_inst + inst.latency + inst.initiation_interval
```

For this class the expression is `3 + 2 + 2 = 7`. It is a reservation
calculation inside allocation, not a measured seven-cycle end-to-end
latency. Dispatch/phase boundaries and result availability must be compared
at matching observation points. Do not assign a measured hardware total
directly to the FU latency field and charge the preceding stages again.

#### Concrete measurement procedure

These are deferred diagnostic measurements, not benchmarks run during this
discussion or prerequisites for the user-selected direct FA3 validation.

| Test | Construction | Measurement / interpretation |
|---|---|---|
| Instruction identity | Build a tiny SM90a kernel emitting the same BF16 pack form; inspect its SASS and prevent dead-code elimination | Validate exact mnemonic, two FP32 inputs, one packed output, active mask, modifiers, and surrounding instructions |
| Resource destination | Repeat known counts of F2FP with matched scaffolding controls; collect raw ALU/FMA/XU counters and their supported metric definitions | Attribute incremental pipe work to F2FP without relying only on utilization percentages |
| Independent throughput | Use multiple independent register chains and vary unrolled work and active warps until the rate stabilizes | Compare cycles per executed warp instruction at matched SM/SMSP scope; distinguish issue throughput from dependent latency |
| Dependent latency | Feed each F2FP result into the next F2FP through a genuine register RAW dependency | Estimate the slope of elapsed cycles versus chain length, after checking SASS dependencies and timing overhead |
| Resource sharing | Compare F2FP-only, MUFU/ALU/FMA-only, and the corresponding mixed instruction streams | Check whether the proposed destination models the actual competition; shared scheduler limits must not be mistaken for shared FU limits |

For a same-op RAW chain, the packed 32-bit result can be bit-reinterpreted
as a subsequent FP32 input purely for the microbenchmark. This is not a
numerical BF16-to-FP32 conversion and does not represent FA3's algorithm.
Require the disassembly to show the real dependency, for example:

```text
F2FP.BF16.F32.PACK_AB R8, R8, R9
F2FP.BF16.F32.PACK_AB R8, R8, R9
```

Use stable finite input bit patterns and verify that reinterpretation did
not insert another instruction or destroy the RAW chain. If the compiler
cannot emit this exact form, characterize an explicit conversion/consumer
chain with its auxiliary operations; do not label an unisolated round trip
as pure F2FP latency.

Time uninstrumented kernels using bounded device-cycle measurements; do not
use per-instruction NVBit timestamp callbacks. Use several chain lengths,
warm-up/repeats, and matched timing/loop controls. Verify clock-read ordering
and a real final consumer, or account for the terminal drain in the fixed
overhead. Report launch-inclusive CUDA-event timing only as a separate
throughput sanity check. NCU profiling is a separate counter run, not the
dependency-latency timer. Reject regimes dominated by instruction-cache
capacity, changing clocks, spills, or timing scaffolding.

If isolated simulator diagnostics are later requested, use the same accepted
instruction sequence/control bits with only the F2FP route changed and INT
timing still at `2,2`. Compare benchmark observables, not a bare configuration
number. This is not a gate before the first direct FA3 run.

| Outcome | Timing decision |
|---|---|
| Destination, throughput, latency, and mixed-resource behavior agree | Keep existing INT `2,2`; do not add an F2FP-specific knob |
| Both F2FP and established INT control instructions disagree similarly | Investigate shared pipeline stages, compiler stall-count handling, or dependency/result timing before inventing an F2FP exception |
| Only the exact F2FP form has a reproducible distinct hardware cost | Consider opcode-form-specific timing backed by the isolated evidence |
| Hardware resource sharing contradicts the INT route | Reconsider destination/sharing; do not force a fit with latency |

The first experiment remains minimal: reuse the established INT pipeline
and its default timing, then compare FA3 directly. Hardware characterization
is a later fidelity check when needed, not a mandate to manufacture another
parameter.

Stage A code ownership:

| Location / function | Proposed Stage A responsibility |
|---|---|
| `util/traces_enhanced/src/traced_instruction.cc`, `set_simulation_opcode()` | Resolve an architecture-qualified full instruction form consistently, rather than applying the base F2FP mapping to every modifier |
| `gpu-simulator/trace-driven/trace_driven.cc`, `parse_from_trace_struct()` | Preserve `m_opcode=OP_F2FP`; select the validated execution category using the same rule as static metadata |
| `trace_config::set_latency()` | Initially reuse existing INT timing; introduce opcode-form-specific timing only if isolated measurements require it. Its current opcode-aware overload only receives the base ID and specially handles NANOSLEEP |
| `remodeling/subcore.cc`, `get_fu()` | Reuse the existing INT/ALU path for the validated route; do not alter the global SP/INT-sharing policy |
| `Subcore::single_decode()` / allocation / writeback | Check the changed fixed-latency operand-read and result-queue path; preserve SB release timing and register widths |

A shared, narrow architecture/form classification rule avoids different
static and dynamic categories. It should not become a kernel-name, PC, or
CTA special case. Preserve semantic/power attribution separately if the
execution category changes `get_oprnd_type()` or decode-stat buckets.

Acceptance is threefold: correct execution resource, correct issue/latency
cost, and unchanged program work. A faster result with an uncharged result
queue, missing operand read, or changed instruction count is invalid.

### 6.2 New hardware cross-checks

The backward postprocess kernel, NCU action 10, is a particularly useful
control: it executes 49152 `F2FP.BF16.F32.PACK_AB` instructions and has
zero measured XU activity. This independently rejects treating that
conversion as a normal MUFU/XU operation; it does not rely on subtracting
the large forward MUFU population.

The initialization kernels expose a second misclassified conversion:

| Per initialization kernel, NCU actions 0/1/2 | Warp instructions |
|---|---:|
| MUFU | 160512 |
| `I2F.U32.RP` | 8448 |
| `F2I.FTZ.U32.TRUNC.NTZ` | 8448 |
| Sum of the three above | 177408 |
| Reconstructed XU count | 177408 |
| `I2FP.F32.U32` | 101376 |
| `I2FP.F32.S32` | 50688 |
| `F2FP.BF16.F32.PACK_AB` | 98304 |

The equality repeats in all three initialization records in both reports.
The XU reconstruction uses the previously stated peak of 0.5 warp
instructions/SM/cycle. The extra I2FP and F2FP instructions are excluded
from that resource count. The ordinary I2F and F2I forms are included:
similar names do not imply identical pipelines.

An ALU route for the fast packing/conversion forms is supported by the
Nsight pipeline descriptions and these resource balances. There is also a
cross-generation NVIDIA forum report identifying I2FP with ALU and F2I with
XU on RTX 3080; that is corroboration, not an H100 calibration source.

Cross-kernel ALU/FMA accounting further narrows candidates. Assuming peaks
of 2 ALU and 4 FMA warp instructions/SM/cycle:

- Initialization action 0 reconstructs 3094272 FMA instructions, exactly
  `FADD + FFMA + FMUL + IMAD + VIADD`.
- The same record reconstructs 2668800 ALU instructions, matching the
  candidate set `F2FP + I2FP + FSEL + FSETP + IADD3 + ISETP + LEA + LOP3
  + MOV + SHF` in that kernel.
- Postprocess action 10 reconstructs 310656 FMA instructions, exactly
  `FMUL + IMAD + VIADD`; its XU count is zero.
- Baseline K5/K10 FMA residuals for the same candidate assignment are only
  -4996/-1137 instructions. ALU residuals for the broader candidate set are
  +6553/+12540. These are aggregate cross-pass comparisons, not exact
  per-opcode hardware routing measurements.

These equalities do not uniquely solve every opcode's destination. They
make `VIADD -> FMA` and `FMNMX/FSEL/FSETP -> ALU` useful isolated-test
candidates; they do not justify changing all those routes together.

One modifier-specific contradiction is more direct: initialization contains
59136 `HFMA2.MMA` instructions, no HMMA/HGMMA base instructions, nonzero
`sm__inst_executed_pipe_tensor_op_hmma`, and zero
`sm__inst_executed_pipe_fma_type_fp16`. NCU explicitly describes its HMMA
counter as including `HFMA2.MMA`. The simulator instead strips `.MMA`,
maps HFMA2 to HALF, and applies the ordinary half-op timing rule. This
specific form needs tensor-path treatment, not the generic HALF route.
Its exact tensor service/operand model must be characterized; simply giving
it a default full-matrix shape would be another error.

### 6.3 Conversion-family disposition

All ten SFU-map entries and the separate F2F entry were reviewed:

| Opcode/form | Current route | Disposition / destination candidate | K5 / K10 occurrences in baseline NCU |
|---|---|---|---:|
| `MUFU.*` | SFU | Retain XU/SFU for measured forms; per-function timing still needs calibration | 1926144 / 1671168 |
| `F2FP.BF16.F32.PACK_AB` | SFU | HW contradiction; validate ALU / `m_int_pipeline` first | 981504 / 1769472 |
| Other F2FP forms | SFU | Open; no family-wide remap authorized | 0 / 0 |
| `I2FP.F32.U32`, `I2FP.F32.S32` | SFU | HW contradiction; ALU candidate, separately test; observed in initialization only | 0 / 0 |
| `I2F.U32.RP` | SFU | XU accounting agrees; retain for this form | 0 / 0 |
| `F2I.FTZ.U32.TRUNC.NTZ` | SFU | XU accounting agrees; retain for this form | 0 / 0 |
| `F2F.*` | Shared DP | Candidate for width-specific split; no F2F sample here proves which forms belong to DP versus faster conversion paths | 0 / 0 |
| `I2I`, `I2IP` | SFU | Candidate ALU/packing paths, but no measured instance; do not infer latency or remap from the name | 0 / 0 |
| `F2IP` | SFU | Candidate fast conversion path; no measured instance; I2FP evidence does not prove this reverse conversion | 0 / 0 |
| `FRND` | SFU | Open; round-to-integer-valued floating point is not the same operation as integer conversion | 0 / 0 |
| `UF2FP` | Vector SFU | Candidate uniform conversion path; official ISA identifies a uniform operation, but no measured instruction here establishes the physical route/cost | 0 / 0 |

This is why the first correction should be exact-form F2FP, not a global
"all conversions go to INT" change. I2FP is a real additional audit finding,
but is absent from the main K5/K10 kernels and must not be blamed for their
current cycle error.

### 6.4 Other routing and classification findings

| Opcode/form | Current route | Evidence and disposition |
|---|---|---|
| `HFMA2.MMA` | HALF -> INT/SP | HW contradiction from initialization; tensor-path form. Ordinary HFMA2 must remain separately classified |
| `FMNMX`, `FSEL`, `FSETP` | SP -> INT when free, SP otherwise | ALU-only candidates from counter balances/sample distribution; current code permits an extra SP resource. Not isolated-HW proof |
| `VIADD` | INT | FMA candidate from the exact balances above; test separately |
| `IMUL`, `IMUL32I`, `IDP`, `IDP4A` | INT | Candidate/source-versus-documentation mismatch: Nsight describes integer multiply and dot-product work on FMA. No instances in this corpus |
| `DMMA` | Shared scalar DP | Source operation-class mismatch: ISA defines matrix MMA and NCU has a tensor-DMMA counter, but current dispatch bypasses tensor shape/service logic. No DMMA timing experiment here |
| `MOVM` | TENSOR | Source operation-class mismatch: matrix move/transpose is charged through the generic MMA shape/operand path. Correct movement resource/cost remains unmeasured |
| `WARPGROUP`, `WARPGROUPSET` | BRANCH / MISC_NO_QUEUE | Group synchronization/counter semantics are not established by a generic pipe assignment; retain the separate Stage C investigation |
| `UIADD3.64` | Nominal UNIFORM entry | Source lookup defect: tokenization splits on '.', so the dedicated key/ID is unreachable and the UIADD3 base entry wins. Same FU today; width-specific identity/timing is not obtained from this entry |
| `WGMMA.WAIT*`, `WGMMA.COMMIT*` aliases | Static metadata overrides; dynamic lookup uses base WGMMA -> TENSOR | Source classification inconsistency if these aliases appear; actual measured Hopper SASS uses WARPGROUP instead |
| `HMUL2` versus `HMUL2_32I` | HALF | Timing inconsistency to test: the explicit half-II division includes the immediate form but omits HMUL2. No HMUL2 sample here; do not fix by blindly halving all HALF forms |
| `CGAERRBAR` | MISC_QUEUE | Source/comment inconsistency: decoder comment calls it a memory-fence path, but it does not change the opcode category to MEMORY_BARRIER_OP. Semantics need separate characterization |
| `REDUX`, `R2UR`, `S2UR` | MISC_NO_QUEUE / INT / MISC_QUEUE | Open vector-to-uniform transfer/reduction paths; a uniform destination alone is insufficient to choose UNIFORM execution |
| `USETMAXREG`, `UCGABAR_*`, `ACQBULK` | UNIFORM / BARRIER / MISC_NO_QUEUE | Open special resource/synchronization semantics; a routing label alone does not establish support |
| Texture/surface families | Common MEM front end | Open TEX-path sharing/backpressure audit; sharing a front end with loads is not by itself a wrong-pipeline proof |

Baseline impact of the additional high-volume candidates:
`FMNMX=1971648/0`, `FSEL=404928/1671168`, `FSETP=18432/0`, and
`VIADD=183024/232488` for K5/K10. These remain candidates, not extra fixes
silently added to Stage A.

False-positive checks:

- IMAD already targets SP and is explicitly excluded from the opportunistic
  INT fallback. Its integer operands do not make that FMA-like route wrong.
- The official ISA calls VHMNMX a SIMD FP16 min/max operation; do not
  relabel it integer solely because it sits near integer DPX entries.
- SHFL/S2R going through MEM is consistent with the documented LSU family;
  it is not evidence that they are mistakenly treated as DRAM loads.
- PREDICATE_OP explicitly selects INT; special predicate latency is applied
  later. The category name is not a separate predicate execution pipe.
- Memory/TMA instructions have staged service and side effects beyond the
  initial latency pair; a default pair alone does not describe their cost.

### 6.5 Complete map inventory

Every map entry appears once below. INT/SP means the configured dynamic
selection described in section 6.1, not two instructions being executed.
The table records pre-patch base-map routing, not a hardware-correctness
certificate. The base map is retained; section 6.8 adds only the exact-form
F2FP execution override.

| Uarch category | Count | Effective FU | Complete opcode list |
|---|---:|---|---|
| SP_OP | 13 | INT/SP; IMAD uses SP | FADD, FADD32I, FCHK, FFMA32I, FFMA, FMNMX, FMUL, FMUL32I, FSEL, FSET, FSETP, FSWZADD, IMAD |
| SFU_OP | 10 | SFU | MUFU, F2FP, F2I, I2F, I2I, I2IP, I2FP, F2IP, FRND, UF2FP |
| HALF_OP | 10 | INT/SP | HADD2, HADD2_32I, HFMA2, HFMA2_32I, HMUL2, HMUL2_32I, HSET2, HSETP2, HMNMX2, VHMNMX |
| TENSOR_CORE_OP | 9 | TENSOR | HMMA, BMMA, IMMA, MOVM, WGMMA, BGMMA, HGMMA, IGMMA, QGMMA |
| DP_OP | 6 | DP_SUBCORE_UNIT -> shared DP | DMMA, DADD, DFMA, DMUL, DSETP, F2F |
| INTP_OP | 36 | INT | BMSK, BREV, FLO, IABS, IADD, IADD3, IADD32I, IDP, IDP4A, IMNMX, IMUL, IMUL32I, ISCADD, ISCADD32I, ISETP, LEA, LOP, LOP3, LOP32I, POPC, SHF, SHL, SHR, VABSDIFF, VABSDIFF4, VIADD, VIADDMNMX, VIMNMX, VIMNMX3, MOV, MOV32I, PRMT, SEL, SGXT, R2UR, CS2R |
| MEMORY_MISCELLANEOUS_OP | 7 | MEM_SUBCORE_UNIT | SHFL, MATCH, QSPC, CCTL, CCTLL, CCTLT, S2R |
| PREDICATE_OP | 4 | INT | PLOP3, PSETP, P2R, R2P |
| MEMORY_BARRIER_OP | 2 | MEM_SUBCORE_UNIT + fence handling | FENCE, MEMBAR |
| LOAD_OP | 9 | MEM_SUBCORE_UNIT | LD, LDC, LDCU, LDG, LDGMC, LDL, LDS, LDSM, LDGSTS |
| STORE_OP | 12 | MEM_SUBCORE_UNIT | STSM, ST, STG, STL, STS, STAS, ATOM, ATOMS, ATOMG, RED, REDAS, REDG |
| MBARRIER_OP | 1 | MISC_NO_QUEUE + sync handling | SYNCS |
| MISCELLANEOUS_QUEUE_OP | 3 | MISC_QUEUE | ERRBAR, S2UR, CGAERRBAR |
| LDGDEPBAR_OP | 1 | MISC_NO_QUEUE + dependency handling | LDGDEPBAR |
| UNIFORM_OP | 28 | UNIFORM | UBMSK, UBREV, UCLEA, UFLO, UIADD3, UIMAD, UISETP, ULDC, ULEA, ULOP, ULOP3, ULOP32I, UMOV, UP2UR, UPLOP3, UPOPC, UPRMT, UPSETP, UR2UP, USEL, USGXT, USHF, USHL, USHR, VOTEU, ULEPC, USETMAXREG, UIADD3.64 |
| BARRIER_OP | 3 | MISC_NO_QUEUE + BAR handling | UCGABAR_ARV, UCGABAR_WAIT, BAR |
| BRANCH_OP | 19 | BRANCH | WARPGROUP, BMOV, BPT, BRA, BREAK, BRX, BRXU, BSSY, BSYNC, ELECT, ENDCOLLECTIVE, JMP, JMX, JMXU, PREEXIT, RPCMOV, RTT, WARPSYNC, YIELD |
| MISCELLANEOUS_NO_QUEUE_OP | 16 | MISC_NO_QUEUE | WARPGROUPSET, ACQBULK, KILL, NANOSLEEP, REDUX, B2R, CSMTEST, GETLMEMBASE, LEPC, NOP, PMTRIG, R2B, SETCTAID, SETLMEMBASE, VOTE, VOTE_VTG |
| TMA_LOAD_OP | 4 | TMA_SUBCORE_UNIT -> TMA shared | UBLKCP, UBLKPF, UTMALDG, UTMAPF |
| TMA_STORE_OP | 3 | TMA_SUBCORE_UNIT -> TMA shared | UBLKRED, UTMAREDG, UTMASTG |
| TMA_MISCELLANEOUS_OP | 2 | TMA_SUBCORE_UNIT -> TMA shared | UTMACCTL, UTMACMDFLUSH |
| TEXTURE_OP | 6 | MEM_SUBCORE_UNIT | TEX, TLD, TLD4, TMML, TXD, TXQ |
| SURFACE_OP | 4 | MEM_SUBCORE_UNIT | SUATOM, SULD, SURED, SUST |
| CALL_OPS | 1 | BRANCH | CALL |
| EXIT_OPS | 1 | BRANCH | EXIT |
| RET_OPS | 1 | BRANCH | RET |
| DEPBAR_OP | 1 | MISC_NO_QUEUE + explicit SB check | DEPBAR |

Latency routing was also checked: INT/SP inherit `2,2`, SFU `8,8`,
HALF `3,2` before selected II overrides, UNIFORM `2,2`, BRANCH `2,1`,
DP `24,16` before staged DP service, MISC_QUEUE `2,2`, and MISC_NO_QUEUE
`1,1`. Tensor shape and predicate/SETP handling override generic timing.
These are recorded configurations/paths, not hardware validation of the
numbers. All 27 mapped categories have a `get_fu()` route.

### 6.6 Stage A execution boundary after this audit

Do not turn this inventory into a batch remap. The agreed 1 -> 3 -> 2
sequence remains in force.

Stage A first qualifies and changes only the measured F2FP form. Use a
shared full-form classification decision, existing INT timing as the first
candidate, unchanged
raw instruction identity, and existing physical resource sharing. Verify
the newly used fixed-latency allocation/result path as well as the pipe
label. In particular, moving only `get_fu()` while leaving SFU category
timing and stall attribution unchanged would not be a complete correction.

I2FP and HFMA2.MMA are additional well-supported findings with useful
initialization-kernel controls. They get independent follow-up experiments;
they are not explanations for K5/K10 when their dynamic counts are zero.
The ALU/FMA placement candidates need isolated tests before any correction.

A final implementation checklist for the first F2FP patch:

1. Exact architecture/form match and identical static/dynamic classification.
2. `OP_F2FP`, operands, masks, PCs, trace order, and `gpu_sim_insn` preserved.
3. Existing INT resource sharing and `2,2` timing reused without free
   additional capacity; isolated cost/sharing characterization is deferred
   and must not be claimed from the aggregate FA3 result.
4. Operand width, SB read/write release, result queue, and semantic/power
   counters consistent after changing the execution category.
5. Non-target F2FP formats, I2F/F2I, FP64 conversions, and other architectures
   unchanged.
6. Classification/source checks and a server build precede direct user-run
   FA3; standalone simulator microbenchmarks are skipped. SFU handshake and
   GSB behavior remain unchanged for this patch.

Audit sources: the production map/dispatch/latency code, both retained NCU
reports, the [official Hopper opcode table](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#hopper-instruction-set),
and [Nsight pipeline definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-pipelines).
The [I2FP/I2F forum discussion](https://forums.developer.nvidia.com/t/whats-difference-between-i2f-and-i2fp-instruction/257786)
is explicitly secondary, non-H100 evidence. No third-party simulator map
was treated as a hardware oracle. All extraction commands ran inline;
no temporary scripts, JSON bundles, or executables were created.

### 6.7 First implementation package and deferred opcodes

Default timing reuse is the agreed policy. The initial implementation
candidate is deliberately smaller than the audit's complete findings list.
This section records its design boundaries; the authorized source
implementation and unperformed build/runtime checks are recorded in 6.8.

#### Exact first-patch scope

```text
architecture: H100 / SM90
instruction form: F2FP.BF16.F32.PACK_AB
raw opcode identity: OP_F2FP, unchanged
execution category: SFU_OP -> INTP_OP
functional unit: existing m_int_pipeline under the active H100 configuration
timing: existing int_latency / int_init, currently 2,2
```

No additional opcode is enabled merely because it appears in the audit.
No new latency parameter, execution pipe, shared-resource capacity, trace
schema, or hardware trace recapture is required for this first candidate:
the existing trace already carries the complete F2FP opcode and operands.
Matching input-artifact provenance remains mandatory.

#### Proposed code structure

1. Add one narrow architecture/full-form execution-category resolver in the
   existing shared instruction/ISA layer. It receives the architecture,
   complete opcode or parsed tokens, and existing base category. Only the
   validated SM90 BF16-pack form overrides the category; all other inputs
   return the existing classification.
2. Call the same resolver from static metadata classification
   (`traced_instruction::set_simulation_opcode()`) and dynamic instruction
   decode (`trace_warp_inst_t::parse_from_trace_struct()`). If needed,
   `traced_kernel` passes its existing architecture value to static
   classification; do not infer architecture from pointer identity.
3. Keep the raw opcode, operand widths/types, masks, PC, and trace cursor
   untouched. Preserve semantic/power attribution explicitly: changing the
   dispatch category must not accidentally turn an FP conversion into an
   integer arithmetic event in unrelated counters.
4. Resolve execution category before the existing `set_latency()` call.
   The unmodified INTP case then supplies `int_latency/int_init`. The normal
   `Subcore::get_fu()` branch supplies INT; it needs no F2FP-specific case.
5. Add focused classification/regression tests and verify the already-existing
   fixed-latency operand-read/result path. Change that path only if a separate
   demonstrated defect requires a separately evaluated patch.

The base F2FP map entry need not be globally changed when only one modifier
form is covered. Reusing a shared exact-form decision avoids two divergent
local exceptions while preserving every unvalidated form.

Expected first-patch code ownership:

| Area | First patch |
|---|---|
| Shared instruction classification, static call site, dynamic decode | Change only as required for the single architecture/form rule |
| Existing tracer classification call site | Pass the architecture to the shared API; no instrumentation/schema change |
| `trace_config::set_latency()` and INT latency config | Reuse unchanged |
| `Subcore::get_fu()` and INT/SFU FU implementations | Reuse unchanged |
| SFU acceptance, group synchronization, WaitBlock, TMA | Unchanged |
| Input traces and opcode occurrence counts | Unchanged |

#### Disposition of the rest of the audit

| Finding | Included in first F2FP patch? | Follow-up decision |
|---|---|---|
| `I2FP.F32.U32/S32` | No | Hardware-backed non-XU finding; qualify ALU timing/sharing and test initialization kernels in a separate patch |
| `HFMA2.MMA` | No | Hardware-backed tensor-path finding; needs form-specific operand/operation/service modeling, not a blind TENSOR_CORE_OP reassignment |
| `FMNMX`, `FSEL`, `FSETP`, `VIADD` | No | Isolated resource measurements before any routing change; aggregate balances are not sufficient |
| `IMUL*`, `IDP*` | No | Documentation-backed candidates without measured instances here |
| `F2F`, `I2I`, `I2IP`, `F2IP`, `UF2FP`, `FRND` | No | Width/form-specific evidence needed; no family-wide conversion remap |
| `DMMA`, `MOVM` | No | Separate tensor/movement model work, including operand widths and service cost |
| `UIADD3.64`, WGMMA aliases, HMUL2 timing, special control operations | No | Independent decode/semantic/timing fixes after their own reproduction |
| `MUFU`, measured ordinary I2F/F2I forms | No | Retain the current resource assignment |

The agreed mainline remains F2FP (problem 1), then SFU acceptance (problem 3),
then WGMMA dependencies (problem 2). Additional opcode findings stay in this
document as separately reviewed work; they do not silently expand Stage A
or change that order.

#### Validation and delivery boundaries

First, check target/non-target forms across architecture boundaries and
static/dynamic classification, and build the simulator on the server.
The user then runs FA3 directly, using the frozen baseline protocol from
section 5. Do not require or launch a standalone F2FP/INT simulator
microbenchmark before that comparison.

Keep `gpu_sim_insn` and semantic instruction work identical. Verify that the
change in per-FU counts is the intended reassignment, not skipped work or
misleading opcode counters. Track fixed-latency result-queue pressure and
ordinary SB dependencies, not just SFU idle cycles.

Retain INT `2,2` when the observables agree with hardware. A mismatch is a
diagnostic result, not permission to tune a new F2FP delay to the FA3 cycle
label. Separate behavioral fixes from bounded instrumentation and commit
each independently when implementation is authorized.

### 6.8 Implemented source change

Status at implementation (2026-09-10): source complete; no local build,
standalone simulator test, or FA3 simulation was run. The user subsequently
provided server logs, analyzed separately in section 6.9.

`resolve_trace_execution_category()` returns `INTP_OP` only when all three
conditions hold:

```text
binary_version == HOPPER_BINART_VERSION (90)
base_category == SFU_OP
opcode == "F2FP.BF16.F32.PACK_AB"
```

Every other category, architecture, and opcode/modifier form returns its
existing base category. No case folding, prefix matching, PC whitelist,
latency override, or new configuration option was added.

Files changed by this implementation, relative to `simulator-remodeled/`:

| File | Change |
|---|---|
| `util/traces_enhanced/src/traced_instruction.h` | Declare the shared resolver; require architecture for static classification |
| `util/traces_enhanced/src/traced_instruction.cc` | Implement the pure exact-form rule and use it in static classification |
| `util/traces_enhanced/src/traced_kernel.cc` | Pass existing metadata architecture into the classifier |
| `gpu-simulator/trace-driven/trace_driven.cc` | Apply the same execution override after raw operand/power classification and before the unchanged latency lookup |
| `util/tracer_nvbit/tracer_tool/tracer_tool.cu` | Update the existing classifier call with `binary_version` for API compatibility |

The dynamic decoder retains `m_opcode=OP_F2FP`, the existing `sp_op`, and
the original floating-point `oprnd_type`; only `op` changes for timing/FU
selection. The unchanged INTP timing case supplies current INT `2,2`, and
the unchanged `get_fu()` selects INT under the observed H100 configuration.
Ordinary SFU/other-FU counters follow the new execution category; semantic
opcode counts are not intentionally changed.

Verification performed without building:

- Source-guard matrix: 1813 cases covering all 212 base map entries,
  target/non-target modifier forms, eight architecture values, and
  non-SFU base categories. Evaluated the small guard as an equivalent
  expression; this is not a compiled C++ unit test.
- Confirmed both simulator classifiers call the same resolver and both
  metadata-loader/tracer call sites pass architecture explicitly.
- Confirmed dynamic raw operand/power classification precedes the override,
  while the existing latency lookup follows it.
- Checked the patch for whitespace errors and inspected the unchanged
  opcode-map/FU/configuration boundaries.

Existing lane-group replay and other worktree changes were preserved.
No temporary scripts, binaries, JSON, or additional Markdown files were
created. At source delivery, C++/CUDA compilation, runtime `gpu_sim_insn`
bit identity, per-FU counts, WaitBlock/TMA conservation, and cycle accuracy
were pending server-side verification. The following section distinguishes
the later runtime findings from the remaining validation gaps.

### 6.9 First server experiment: improvement with changed inputs

Analyzed 2026-09-11. The user identified baseline K5 `.e102/.o102` and K10
`.e77/.o77` as the completed F2FP experiment. The same copied corpus also
contains new s1024 K5 `.e103/.o103` and nh8 K5 `.e104/.o104`.
The result tables and exact log locations are in `.result/FA3_multi.md`.
No build, simulator execution, or source modification was performed during
this log analysis. Read-only inline analysis created no temporary files.

#### Measured outcome

| Case / kernel | Historical -> new cycles | HW cycles | Signed error before -> after | Cycle reduction |
|---|---:|---:|---:|---:|
| baseline K5 `.o90 -> .o102` | 85172 -> 71800 | 67838 | +25.55% -> +5.84% | 13372 (15.70%) |
| baseline K10 `.o66 -> .o77` | 137348 -> 128702 | 130641 | +5.13% -> -1.48% | 8646 (6.29%) |
| s1024 K5 `.o97 -> .o103` | 44442 -> 39460 | 38153 | +16.48% -> +3.43% | 4982 (11.21%) |
| nh8 K5 `.o98 -> .o104` | 70487 -> 60967 | 57589 | +22.40% -> +5.87% | 9520 (13.51%) |

Baseline forward absolute excess falls from 17334 to 3962 cycles, a 77.14%
reduction. Descriptive MAPE across the three HD64 forward cases falls from
21.48% to 5.04%. These are improvements in the observed historical comparison,
not measured isolated F2FP contributions. Backward crosses slightly below HW;
further cycle reduction is not automatically a cycle-accuracy improvement.

#### Resource signature

Both configurations retain INT latency/II `2,2` and SFU `8,8`.

| Case / kernel | SFU-blocked warp-cycles before -> after | Reduction | Other-FU-blocked warp-cycles before -> after |
|---|---:|---:|---:|
| baseline K5 | 35372161 -> 12452184 | 64.80% | 17569946 -> 27886681 |
| baseline K10 | 31312865 -> 2997352 | 90.43% | 18017466 -> 30904658 |
| s1024 K5 | 11406906 -> 4321073 | 62.12% | 7558874 -> 10417476 |
| nh8 K5 | 11905209 -> 4452329 | 62.60% | 6331079 -> 9143741 |

These are `total_num_warpcyc_stalled_fu_sfu` and
`total_num_warpcyc_stalled_fu_other`. The latter includes more than INT.
Moving F2FP changes the numerator's opcode population as well as contention;
the SFU reduction must not be described as a MUFU-only stall reduction.
Concurrent warp waits are not additive kernel cycles.

SFU pipe-active subcore-cycles fall from 27835090 to 20035269 in baseline
K5, and from 33432805 to 19734837 in K10. Baseline K5's result-queue-full
issue-stage counter rises from 141327 to 199612; K10's falls from 101823
to 91259. The work is using other resources, not simply becoming free.
All four new runs retain `total_num_next_stage_blocked_by_sfu=0`.

This is the expected qualitative signature of removing BF16 packing from
the SFU and sharing the existing fixed-latency INT path. It strengthens the
resource-assignment hypothesis without calibrating a new latency parameter.
These logs do not export a complete opcode-by-FU retirement matrix, so they
do not independently prove the exact number of F2FP instructions rerouted.

#### Input mismatch is independently visible before execution

| Comparison | Input warp instructions before -> after | Input active-lane instructions before -> after | `gpu_sim_insn` delta |
|---|---:|---:|---:|
| baseline K5 | 14818156 -> 14818470 | 450396424 -> 450422688 | -7584 |
| baseline K10 | 20995973 -> 20997218 | 624946858 -> 624958026 | +3072 |
| s1024 K5 | 5279446 -> 5278073 | 160741746 -> 160727012 | -2400 |
| nh8 K5 | 5073262 -> 5073019 | 154134952 -> 154120112 | -1568 |

`trace_parser.cc:583-687` builds these `[TRACECHECK]` fields directly from
the loaded CTA protobuf: `warp.instructions_size()`, masks, memory records,
and raw opcode strings. This precedes replay and retirement; changing
`SFU_OP` to `INTP_OP` does not change those input counts. Therefore this is
not merely an observed difference in runtime poll folding.

The baseline K5 input changed HGMMA/MUFU allocation in 49 of 132 CTAs.
For example, CTA1 changes from 2592 HGMMA / 14304 MUFU to 2736 / 15096.
The last-finishing CTA changes from CTA43 (2880 HGMMA, finish 85171) to
CTA2 (3024 HGMMA, finish 71799). A trace-fixed persistent work distribution
is consequently an additional variable in the kernel-tail comparison.
The other three comparisons preserve per-CTA HGMMA/MUFU counts but still
change poll/mask/memory-record populations.

Aggregate input compute/transfer counts are unchanged:

| Case / kernel | HGMMA | MUFU | TMA-family | TMA completion records |
|---|---:|---:|---:|---:|
| baseline K5 | 349056 | 1926144 | 6432 | 5112 |
| baseline K10 | 835584 | 1671168 | 30336 | 13824 |
| s1024 K5 | 110592 | 611712 | 2640 | 1680 |
| nh8 K5 | 116352 | 642048 | 2496 | 1704 |

The loaded ticket populations also differ:

| Case | `exact_ticket_slots` before -> after |
|---|---:|
| baseline (shared K5/K10 artifacts) | 816411 -> 817199 |
| s1024 | 253101 -> 252256 |
| nh8 | 231176 -> 230723 |

The files may have been recaptured or replaced; the logs do not establish
which operation changed them. Identical relative `./traces/...` paths are
not evidence of identical content. A new F2FP routing rule itself does not
require trace regeneration. No cause of the input replacement is assumed.

The printed s1024/nh8 configuration entries match exactly. Baseline logs
have four newly printed WaitBlock entries (late-arrival instrumentation,
watchdog cycles, heartbeat cycles, and debug print budget), with no changes
to the shared printed timing entries. Every binary still identifies itself
only as `*_modified_0.0`; the exact binary SHA and intervening WaitBlock
changes cannot be isolated from these logs. Checking recent historical
baseline `.o90-.o94` and `.o66-.o69` found no matching input-statistics
fingerprint for `.o102/.o77`.

#### Correctness gates and interpretation

All four improved runs exit normally. Every WaitBlock summary is PASS:
132 for each forward and 384 for baseline backward. Using the last
cumulative summary per SM gives zero live state and balanced hold/release
and deferred/replayed occurrences. All TMA final audits pass with zero
in-flight work, outstanding barriers, mismatches, duplicate completions,
and orphan cleanup. Assigned/completed bytes match the historical controls:
85917696 (baseline K5), 122880000 (K10), 28704768 (s1024 K5), and 28639232
(nh8 K5). STALLALIGN final checks pass.

The input-warp = retired-warp + folded-occurrence identities also hold,
as recorded in the results report. These within-run checks do not prove
cross-run lane-mask/semantic preservation. Because the inputs differ,
the `gpu_sim_insn` deltas neither satisfy the bit-identical A/B gate nor
by themselves prove a regression introduced by the F2FP patch. Do not
waive the invariant or label the delta harmless solely because it is small.

The current conclusion is: the accuracy improvement and resource signature
strongly support the proposed correction; its isolated magnitude and
instruction conservation remain unverified. The user has accepted Opt 13
as Done and can proceed to the separate SFU-acceptance investigation.
Do not add an F2FP latency override or infer an SFU/WGMMA fix from the
remaining cycle error alone; those mechanisms require their own evidence.

The clean confirmation is to freeze the current trace, static metadata,
sidecars, configuration, and all unrelated source changes, then compare two
server builds differing only in the shared exact-form F2FP resolver
(`SFU_OP` control versus `INTP_OP` candidate). Record binary/trace/sidecar
hashes. Require matching per-CTA input statistics and `gpu_sim_insn` before
accepting the measured cycle delta. This calls for a same-input control
simulation, not a trace recollection or an additional microbenchmark.
It is retained as validation follow-up rather than a blocker to recording
Opt 13 as complete. New SFU experiments must independently preserve identical
inputs and `gpu_sim_insn` within their own control/candidate pair.

Other opcode dispositions remain in sections 6.3-6.4; their actionable backlog
is Arch TODO-3 in `.result/FA3_progress.md`. `I2FP` and `HFMA2.MMA` require
initialization-kernel coverage; main-kernel `FMNMX/FSEL/FSETP/VIADD` candidates
need isolated resource evidence; `DMMA/MOVM` need service/operand-model work.
None is silently added to Opt 13 or the next SFU patch.

#### Remaining copied runs

The ten-pair copy also includes unchanged s1024 K10 `.o72`, HD128 K5 `.o100`,
and HD128 K10 `.o75`. They are not evidence of new F2FP improvements.
Noncausal K9 `.o76` completes at 184505 cycles (-15.88%), but has no
previous completed control. nh8 K10 `.o78` terminates with signals 15/11;
noncausal K4 `.o101` aborts at the tensor pipeline `start_stage` capacity
assertion. Both lack final kernel cycles and are excluded from improvements.
The latter assertion is addressed separately by commit `322e7cf`; its
existence in the local tree does not mean `.o101` ran that corrected binary.

## Verification record

The temporary tools were removed after the following checks. Original NCU
reports, simulator logs, trace protobufs, and production sources are retained.

1. Opened the binary NCU report with the installed x86_64 Nsight Python API,
   running `/usr/bin/python3` through `arch -x86_64`. The module directory was
   `/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python`; the
   library directory passed to `ncu_report._ncu_report.load_report()` was
   `/Applications/NVIDIA Nsight Compute.app/Contents/MacOS`.
2. Selected the action by its complete mangled name, after removing the
   simulator's `___0` suffix. Matched each sampled PC through the metric's
   own correlation IDs and `action.sass_by_pc()`. All 943 sampled opcode
   identities, including modifiers, matched the static trace.
3. Parsed all 132 baseline CTA protobufs with the existing
   `build_spin_wait_sidecar.vpb_parse_full()` parser. Joined each instruction
   by `(function_unique_id, pc)` to `enhanced_execution_info.json`; all
   MUFU, F2FP, HGMMA, and WARPGROUP opcode/modifier totals matched NCU.
4. Joined `[CTAFIN]` and first-late records by global CTA ID, and checked
   zero population deltas in the bounded stall windows.
5. Compiled the opcode/operand/control-bit probe with `clang++ -std=c++17`,
   linking the unmodified `traced_operand.cc`, `control_bits.cc`,
   `string_utilities.cc`, and `JSONBase.cc` sources. The assertions passed;
   decoded values are recorded in sections 1 and 2.
6. Checked the source-derived SFU recurrence against production phase
   ordering. All saturated issue gaps were 9 cycles. This was not a
   full-simulator execution.

Cleanup removed the two temporary script/probe source files, the probe
executable, and the generated evidence JSON. No production code, input
trace, NCU report, or original simulator log was removed.

Primary report SHA256:
`367a2c21e612f430bdf59877d2c56642f72bfd95695309a9208fdac546015fc0`.

Trace static-metadata SHA256:
`ceecccd30cb661ea1e5e155750c51c62f0f340e18d9c38ed7ad51938292cab54`.

Architecture references:

- [Nsight Compute profiling guide: pipelines and stall reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference).
- [NVIDIA WGMMA programming guide: collective scope and commit/wait ordering](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wgmma_programming.html).

## Prior multi-shape diagnostics

These values were moved from the results report. They are supporting
measurements only, not a root-cause proof. In particular, the "per MUFU"
normalization includes F2FP waiting in its SFU numerator and must not be
interpreted as the delay of a MUFU instruction or as a MUFU-only queue.

| Forward case / run | CTA warps | Consumer warpgroups | SFU-blocked warp-cycles / MUFU | SM tensor + SFU overlap % | Exact-generation late events |
|---|---:|---:|---:|---:|---:|
| baseline `.o90` | 16 | 3 | 18.36 | 51.81 | 9542 |
| s1024 `.o97` | 16 | 3 | 18.65 | 24.23 | 755 |
| nh8 `.o98` | 16 | 3 | 18.54 | 32.08 | 2560 |
| HD128 `.o99` | 12 | 2 | 10.26 | 57.45 | 3 |

The SFU numerator counts blocked warp-cycles; simultaneous waits are not
additive kernel cycles. Overlap ORs pipe-busy state across subcores and
divides the both-busy count by the sum of all four overlap buckets; it is
not a per-SMSP hardware utilization measurement. Late-event populations
depend on work and ticket coverage. The baseline is not a same-build control.

For first-late action `0x5a50`, the bounded post-TMA-completion windows are:

| Metric | s1024 `.e97` | nh8 `.e98` |
|---|---:|---:|
| CTAs in this first-late cohort | 68 | 54 |
| Median next-open -> completion cycles | 375 | 299 |
| Median completion -> consumer action cycles | 417.5 | 1274 |
| Sole-SFU blocker / consumer active warp-cycles | 72.80% | 74.69% |
| Eligible, not selected / consumer active warp-cycles | 9.73% | 11.52% |
| No valid head / consumer active warp-cycles | 0% | 0% |
| WaitBlock hold / producer active warp-cycles | 90.65% | 91.70% |

This describes upstream work while approaching a TRYWAIT, not an SFU-backed
TRYWAIT instruction. It is selected-window evidence, not the kernel's
critical-path decomposition. All three new forward runs report
`total_num_next_stage_blocked_by_sfu=0`.

HD128 changes the tile from `(192,128,64)` to `(128,128,128)` and the consumer
population from 384 to 256 threads. Its lower per-MUFU wait is therefore not
an isolated same-work experiment. Its mean producer-minus-consumer drain gap
is -618 cycles, versus -1663 for s1024 and -1027 for nh8. No corresponding
hardware role-tail measurement for these new shapes is available here.

The nh8 trace has 44 zero-tensor CTAs among 132 launches. Work distribution
must be controlled before assigning causal meaning to aggregate idle shares.

## Historical failure analysis

These notes were moved from `.result/FA3_multi.md` on 2026-09-10 and
consolidated here. They preserve the prior failed-run investigation and are
not new evidence for the forward cycle-accuracy findings above.

### HD128 K10: unfinished TMA generation reused

The first explicit failure is `.e74:203529-203530`:

```text
[TMABIND][abort][SM66] cyc=24320 reason=contract-generation-not-empty warp=0 trace_kernel=10 cta=0 barrier=0x30c30 generation=1 owner_warp=0 expected=16640 assigned=16640 completed=256 outstanding=2
```

This is a simulator runtime failure, not the former UTMAPF mapping failure.
Both TMA and WaitBlock sidecars loaded successfully (`.e74:7-9`). The first
stack enters `SM::open_tma_assignment_contract` from
`SM::handle_sync_instruction` (`.o74:1117-1118`).

| Cycle | SM66 / global CTA66 evidence | `.e74` lines |
|---:|---|---|
| 21106-22014 | Consumer warps 4-11 observe `0x30c30` not ready; no consumer WaitBlock hold is logged | 131110-148898 |
| 22723 | Producer warp 0 opens generation 1, contract 7, at PC `0x72e0`, expecting 16640 bytes | 165917-165918 |
| 22876-22880 | Assignments at `0x7410/0x7420/0x7430`: 8192 + 8192 + 256 bytes | 170313-170453 |
| 23607 | Only the 256-byte transfer completes; two 8192-byte loads remain outstanding | 187958-187959 |
| 23752 | Consumer arrivals nevertheless complete reuse barrier `0x30c40`, generation 1 -> 2 | 191473-191474 |
| 24261 | Producer wait at `0x7200` sees `0x30c40` ready | 202347 |
| 24320 | Opening another contract on still-live `0x30c30` generation 1 aborts | 203529-203530 |

Confirmed mechanism in the prior analysis: consumer progress allows producer
buffer reuse before the associated TMA data generation has completed. The
contract guard prevents overwriting that in-flight generation; 24320 is the
failure cycle, not a usable `gpu_sim_cycle`.

The leading source-level explanation was incomplete consumer wait coverage.
HD128 K10 has an early-ready two-probe path (`0x1a40` -> ready branch, then
`0x1a70` -> fallback); the local sidecar has no runtime action at `0x1a40`,
marks `0x1a70` metadata-only, and leaves consumer poll `0x86a0` non-foldable.
`SM::handle_sync_instruction` creates WaitBlock holds only for selected
INJECT/FOLD actions; other modeled misses can continue along the captured
instruction path. Thus a lifecycle ready witness is not a runtime wait.

Qualification: the local HD128 sidecar is not identical to the run's input.
It reports 11 FOLD records / 36210 exact targets, while `.e74:9` reports
12 captured polls / 36339 exact targets. The timeline above is from the
actual failed run; exact source/sidecar attribution needs its matching inputs.
Next debugging target is consumer wait/hold coverage and the resulting
data-ready -> consumer-release ordering, not relaxing the contract guard or
tuning TMA latency.

The subsequent `signal 11` and `FSEL undefined instruction` occur after the
first abort (`.e74:203531-203543`, `.o74:1131-1132`). Current
`ISA_Def/hopper_opcode.h:25` already maps FSEL. These may be secondary
multithreaded shutdown errors; they are not evidence that missing FSEL
support caused the first failure.

### Relation to recent commits

- `979b4ed`: resolves HD128 UTMAPF using validated kernel-aware descriptor
  chains. K5 `.e99` resolves both prefetch sites and all 768 UTMAPF
  commands with zero unresolved mappings.
- `f1e7acc`: adds lifecycle witness discovery and occurrence-specific partial
  joins. Witness-only waits intentionally do not become runtime hold actions;
  its commit explicitly leaves full simulator/cycle regression pending.
- The failing contract guard predates these changes (`4de5c80`, extended by
  `39abefb`). It permits overlapping assignments to different barriers, not
  reuse of an unfinished generation on the same barrier.
- `943bbb6` pins NVBit 1.7.6 after the noncausal 1.7.5 instrumentation failure;
  `2b79220` keeps SDK installation explicit and separate from compilation.
  These do not by themselves establish noncausal simulator readiness.

### Noncausal: no new simulator result

The old NVBit `illegal instruction` at `tracer_tool.cu:3474` is historical.
A newer device-4 capture at revision `f1e7acc` is present locally, with K4
forward and K9 backward. Its TMA contract validation has zero errors and its
generation-ticket report is PASS, but no `spin_wait_sidecar.json` was present
when these notes were written. The outstanding qualification was WaitBlock
partial-join support; the earlier investigation found independently waiting
lane groups that did not meet the sibling-quiescent proof. No noncausal
simulator pair was included in the six-pair update.
