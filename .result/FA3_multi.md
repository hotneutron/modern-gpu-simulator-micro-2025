# FA3 Multi-Shape Validation: NCU vs GCoM Cycle Accuracy

Cycle-accuracy comparison for FlashAttention-3 (bf16) across shape variations
of the baseline `flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24`.

Goal: measure how well GCoM reproduces real-H100 per-kernel cycles as one
axis of the shape is varied at a time (seq-len, heads, head-dim, causal mode).

Updated 2026-09-14 from the ten latest local simulator log pairs. All ten
completed and pass their in-run WaitBlock, TMA, STALLALIGN, and occurrence
audits. The set mixes newly downloaded and unchanged runs; HW references are
unchanged. Five HD64 historical comparisons and HD128 K5 have different input
or `gpu_sim_insn` counts from their controls, so those cycle deltas remain
provisional rather than strict A/B PASS.

## Conventions

- HW reference metric: `sm__cycles_elapsed.avg` (median across NCU repeats),
  the primary hardware cycle label. `sm__cycles_active.avg` retained as a
  diagnostic.
- Simulator metric: final GCoM per-kernel `gpu_sim_cycle` from a completed
  run. A cycle printed at an abort is not a kernel cycle result.
- Signed error: `(sim_cycles - hw_cycles) / hw_cycles * 100`.
- One application (bwd spec) launches both a forward kernel and a backward
  kernel. They are reported as separate rows.
- Causal trace kernel ids (for `-C H100_80GB-OnlyKernelN`): forward = 5,
  backward main = 10; kernel 11 is backward postprocess-convert-dQ, not a
  target. The newer noncausal device-4 capture uses forward = 4 and backward
  main = 9, verified by kernel names in its `stats.csv`.
- HW cycles below are `sm__cycles_elapsed.avg` from the `_full_rpt.ncu-rep`
  reports under `accorde/ncu_report/`.

## Case Inventory

| Case | Varied axis vs baseline | Trace / sidecar status | Latest sim (fwd / bwd) | NCU status |
|---|---|---|---|---|
| `s2048-hd64-nh24` (baseline) | - | Loaders accepted; input differs from historical control | COMPLETED (INPUT/INSN DIFFER) / COMPLETED (INPUT/INSN DIFFER) | done |
| `s1024-hd64-nh24` | seq-len 2048 -> 1024 | Loaders accepted; K5 input differs from historical control | COMPLETED (INPUT/INSN DIFFER) / PASS | done |
| `s2048-hd64-nh8` | heads 24 -> 8 | Loaders accepted; K5/K10 input differs from historical controls | COMPLETED (INPUT/INSN DIFFER) / COMPLETED (INPUT/INSN DIFFER) | done |
| `s2048-hd128-nh24` | head-dim 64 -> 128 | Repaired; TMA and WaitBlock loaders accepted | COMPLETED (INSN MISMATCH) / PASS | done |
| `noncausal-s2048-hd64-nh24` | causal -> noncausal | TMA and WaitBlock loaders accepted by K4/K9 | PASS (first complete) / PASS (first complete) | done |

## Forward Kernel (`FlashAttnFwdSm90`, causal K5 / noncausal K4)

| Case | HW cycles (elapsed.avg) | HW cycles (active.avg) | GCoM cycles | Signed err % | Abs err % |
|---|---:|---:|---:|---:|---:|
| `s2048-hd64-nh24` (baseline) | 67838 | 61147 | 71863 | +5.93 | 5.93 |
| `s1024-hd64-nh24` | 38153 | 27665 | 39460 | +3.43 | 3.43 |
| `s2048-hd64-nh8` | 57589 | 23617 | 60967 | +5.87 | 5.87 |
| `s2048-hd128-nh24` | 86912 | 78421 | 90794 | +4.47 | 4.47 |
| `noncausal-s2048-hd64-nh24` | 101012 | 94637 | 104808 | +3.76 | 3.76 |

## Backward Kernel (`FlashAttnBwdSm90`, causal K10 / noncausal K9)

| Case | HW cycles (elapsed.avg) | HW cycles (active.avg) | GCoM cycles | Signed err % | Abs err % |
|---|---:|---:|---:|---:|---:|
| `s2048-hd64-nh24` (baseline) | 130641 | 118089 | 128217 | -1.86 | 1.86 |
| `s1024-hd64-nh24` | 48523 | 41242 | 57471 | +18.44 | 18.44 |
| `s2048-hd64-nh8` | 80191 | 43073 | 74915 | -6.58 | 6.58 |
| `s2048-hd128-nh24` | 256243 | 203220 | 199796 | -22.03 | 22.03 |
| `noncausal-s2048-hd64-nh24` | 219335 | 205731 | 184505 | -15.88 | 15.88 |

The baseline `.o106/.o80` pair supersedes `.o102/.o77`. The nh8 backward
`.o79` run supersedes failed `.o78`, and noncausal forward `.o107` supersedes
the `.o101/.o105` failures. The s1024 backward and HD128 rows are unchanged
copied results.

## Historical Control Comparison

| Case / kernel | Previous log -> latest | Cycles before -> after | Cycle reduction % | Signed error % before -> after |
|---|---|---|---:|---|
| baseline K5 | `.o90 -> .o106` | 85172 -> 71863 | 15.63 | +25.55 -> +5.93 |
| baseline K10 | `.o66 -> .o80` | 137348 -> 128217 | 6.65 | +5.13 -> -1.86 |
| s1024 K5 | `.o97 -> .o103` | 44442 -> 39460 | 11.21 | +16.48 -> +3.43 |
| nh8 K5 | `.o98 -> .o104` | 70487 -> 60967 | 13.51 | +22.40 -> +5.87 |
| nh8 K10 | `.o73 -> .o79` | 83018 -> 74915 | 9.76 | +3.53 -> -6.58 |

Descriptive MAPE for the baseline pair is 15.34% -> 3.89%; for the three
HD64 forward cases it is 21.48% -> 5.08%; across all five historical
comparisons it is 14.62% -> 4.73%. These are not same-input patch-isolated
validation. In particular, nh8 K10 reduces cycles but increases absolute
cycle error from 3.53% to 6.58%.

## Backward Auxiliary Kernels (HW reference only)

Preprocess and postprocess-convert-dQ kernels bracket the backward main kernel.
Recorded for completeness; not primary GCoM targets.

| Case | BwdPreprocess elapsed.avg | BwdPostprocessConvertdQ elapsed.avg |
|---|---:|---:|
| `s2048-hd64-nh24` (baseline) | 15969 | 14272 |
| `s1024-hd64-nh24` | 13796 | 11849 |
| `s2048-hd64-nh8` | 10533 | 9887 |
| `s2048-hd128-nh24` | 32250 | 26870 |
| `noncausal-s2048-hd64-nh24` | 19173 | 17191 |

## Latest Simulator Runs

| Case | Kernel | Log pair | Status | `gpu_sim_insn` | Cycles: previous -> latest (delta) |
|---|---|---|---|---:|---|
| `s2048-hd64-nh24` | K5 | `.e106/.o106` | COMPLETED; INPUT/INSN DIFFER | 447604920 | 85172 -> 71863 (-13309) |
| `s2048-hd64-nh24` | K10 | `.e80/.o80` | COMPLETED; INPUT/INSN DIFFER | 623703264 | 137348 -> 128217 (-9131) |
| `s1024-hd64-nh24` | K5 | `.e103/.o103` | COMPLETED; INPUT/INSN DIFFER | 158640628 | 44442 -> 39460 (-4982) |
| `s1024-hd64-nh24` | K10 | `.e72/.o72` | PASS; unchanged copied result | 179487936 | 57471 retained |
| `s2048-hd64-nh8` | K5 | `.e104/.o104` | COMPLETED; INPUT/INSN DIFFER | 152748888 | 70487 -> 60967 (-9520) |
| `s2048-hd64-nh8` | K10 | `.e79/.o79` | COMPLETED; INPUT/INSN DIFFER | 207892832 | 83018 -> 74915 (-8103) |
| `s2048-hd128-nh24` | K5 | `.e100/.o100` | COMPLETED; INSN MISMATCH | 434069888 | 90735 -> 90794 (+59) |
| `s2048-hd128-nh24` | K10 | `.e75/.o75` | PASS | 861518928 | FAILED -> 199796 (first complete) |
| `noncausal-s2048-hd64-nh24` | K4 | `.e107/.o107` | PASS; no previous completed control | 731518352 | 104808 (first complete) |
| `noncausal-s2048-hd64-nh24` | K9 | `.e76/.o76` | PASS; no previous completed control | 1145693312 | 184505 (first complete) |

- PASS means final kernel statistics, normal simulator exit, and only PASS
  WaitBlock summaries / TMA final audits. All ten latest logs have
  balanced hold/release and partial-replay counters, zero active WaitBlock
  state, and zero watchdog aborts / generation overshoots in their summaries.
- PASS in this table is an in-run audit result, not proof of cross-run
  instruction conservation. The five HD64 comparisons below have different
  trace-load counts and ticket populations as well as different retired
  lane-instruction counts. Bit-identical-input validation is unavailable.
- Baseline `.o106/.o80` reproduce the `gpu_sim_insn` and input warp-instruction
  counts of `.o102/.o77` exactly. Their cycles move by +63 / -485 respectively,
  but both still differ from the historical `.o90/.o66` controls.
- HD128 K5 `.o100` exercised 701 partial modeled misses, all as already-issued
  joins, with zero deferred capsules. K10 exercised no partial modeled misses.
- HD128 K5 is not a validation PASS: `gpu_sim_insn` changed from 434070656 in
  `.o99` to 434069888 in `.o100` (-768). This violates the bit-identical
  instruction-count invariant, so its 90,794-cycle result is provisional.
  K10 has no prior completed HD128 run for a cross-run instruction comparison.
- Older multi-shape comparisons at `538be99` used `.o95/.o70` and `.o96/.o71`.
  Those four files remain absent locally. The five HD64 control logs in
  the comparison above (`.o90/.o66/.o97/.o98/.o73`) are available and were
  reread.
- The new logs identify both builds only as `*_modified_0.0`, not an exact
  git SHA. INT latency/II remains `2,2` and SFU remains `8,8` in all five
  HD64 comparisons. The s1024/nh8 printed configuration entries match;
  baseline logs differ by four added WaitBlock instrumentation/watchdog
  option entries. Binary and full artifact equivalence is not established.
- Across all ten latest completed kernels, descriptive MAPE is 8.82%:
  forward 4.69% and backward 12.96%. This includes provisional input-different
  rows and first-complete noncausal rows, so it is not a strict validation
  aggregate.
- All ten completed runs have `STALLALIGN FINAL status=PASS` and
  `sum(TRACECHECK.warp_insts) = retired_warp_instructions +
  captured_poll_advanced_occurrences`. For multi-CTA SMs, use the last
  cumulative WaitBlock snapshot per SM, not the sum of every CTA snapshot.
  These within-run identities do not resolve cross-run `gpu_sim_insn`
  mismatches.

| Comparison | `gpu_sim_insn` before -> after (delta) | Input `TRACECHECK.warp_insts` before -> after |
|---|---|---|
| baseline K5 | 447612504 -> 447604920 (-7584) | 14818156 -> 14818470 |
| baseline K10 | 623700192 -> 623703264 (+3072) | 20995973 -> 20997218 |
| s1024 K5 | 158643028 -> 158640628 (-2400) | 5279446 -> 5278073 |
| nh8 K5 | 152750456 -> 152748888 (-1568) | 5073262 -> 5073019 |
| nh8 K10 | 207887392 -> 207892832 (+5440) | 6911117 -> 6910425 |

Latest-run occurrence conservation:

| Run | Input warp instructions | Retired warp instructions | Captured-poll advanced occurrences | Residual |
|---|---:|---:|---:|---:|
| baseline K5 `.o106` | 14818470 | 14447843 | 370627 | 0 |
| baseline K10 `.o80` | 20997218 | 19912645 | 1084573 | 0 |
| s1024 K5 `.o103` | 5278073 | 5128950 | 149123 | 0 |
| s1024 K10 `.o72` | 6052209 | 5750828 | 301381 | 0 |
| nh8 K5 `.o104` | 5073019 | 4945133 | 127886 | 0 |
| nh8 K10 `.o79` | 6910425 | 6637594 | 272831 | 0 |
| HD128 K5 `.o100` | 14656804 | 14095568 | 561236 | 0 |
| HD128 K10 `.o75` | 29015897 | 27722674 | 1293223 | 0 |
| noncausal K4 `.o107` | 24167901 | 23394795 | 773106 | 0 |
| noncausal K9 `.o76` | 38634511 | 36410344 | 2224167 | 0 |

## HW and Baseline Notes

- NCU references were collected for all five cases regardless of trace status,
  using `accorde/ncu_report/fa3_ncu_pipe_rawcounts.sh` (`--set full` unioned
  with raw pipe count/peak metrics).
- Historical baseline controls are from `.result/FA3_progress.md` Opt 12
  (producer-consumer mbarrier WaitBlock/exact-generation): fwd 85,172
  (`.o90`), bwd 137,348 (`.o66`). Current baseline-case rows use `.o106/.o80`;
  the pre-experiment results snapshot is commit `009b419`.
  `FA3_progress.md` quotes the historical controls against its own
  earlier HW reference (fwd 67,696 / bwd 132,901, giving +25.82% / +3.35%). The
  error percentages here are recomputed against this campaign's freshly measured
  NCU `sm__cycles_elapsed.avg` (fwd 67,838 / bwd 130,641), so they differ
  slightly (+25.55% / +5.13%). The two HW references should be reconciled before
  cross-campaign comparison.

## Data Provenance

Where every value in the tables above came from. Paths under `sim_run_12.8/`
are relative to `modern-gpu-simulator-micro-2025/simulator-remodeled/`; NCU
reports are under `accorde/ncu_report/`.

### HW cycles (NCU)

Metric `sm__cycles_elapsed.avg` / `sm__cycles_active.avg`, read from the
`_full_rpt.ncu-rep` reports (page `raw`), all produced by
`accorde/ncu_report/fa3_ncu_pipe_rawcounts.sh` (`--set full` + raw pipe metrics):

| Case | NCU report file |
|---|---|
| baseline s2048-hd64-nh24 | `accorde/ncu_report/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.ncu-rep` |
| s1024-hd64-nh24 | `accorde/ncu_report/flashattn-fa3-bf16-bwd-causal-b1-s1024-hd64-nh24_full_rpt.ncu-rep` |
| s2048-hd64-nh8 | `accorde/ncu_report/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh8_full_rpt.ncu-rep` |
| s2048-hd128-nh24 | `accorde/ncu_report/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd128-nh24_full_rpt.ncu-rep` |
| noncausal-s2048-hd64-nh24 | `accorde/ncu_report/flashattn-fa3-bf16-bwd-noncausal-b1-s2048-hd64-nh24_full_rpt.ncu-rep` |

### GCoM cycles (simulator `gpu_sim_cycle`)

Read from each simulator stats output file (`.oNN`). fwd = `OnlyKernel5`,
bwd = `OnlyKernel10` for causal cases; noncausal uses `OnlyKernel4/9`.
The paired `.eNN` file is in the same directory with the same filename stem.

| Case | Kernel | GCoM cycle | Source stats file |
|---|---|---:|---|
| baseline s2048-hd64-nh24 | fwd | 71863 | `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/.../H100_80GB-OnlyKernel5/...___warmup_-63a73d452237.o106:4178` |
| baseline s2048-hd64-nh24 | bwd | 128217 | `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/.../H100_80GB-OnlyKernel10/...___warmup_-63a73d452237.o80:8023` |
| s1024-hd64-nh24 | fwd | 39460 | `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s1024-hd64-nh24/.../H100_80GB-OnlyKernel5/...___warmup_-fd9021e586ff.o103:3666` |
| s1024-hd64-nh24 | bwd | 57471 | `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s1024-hd64-nh24/.../H100_80GB-OnlyKernel10/...___warmup_-fd9021e586ff.o72:4082` |
| s2048-hd64-nh8 | fwd | 60967 | `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh8/.../H100_80GB-OnlyKernel5/...___warmup_0_-ed3ccf38e6b5.o104:3059` |
| s2048-hd64-nh8 | bwd | 74915 | `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh8/.../H100_80GB-OnlyKernel10/...___warmup_0_-ed3ccf38e6b5.o79:3005` |
| s2048-hd128-nh24 | fwd | 90794 | `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd128-nh24/.../H100_80GB-OnlyKernel5/...___warmu-9e387f39a890.o100:3764` |
| s2048-hd128-nh24 | bwd | 199796 | `sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd128-nh24/.../H100_80GB-OnlyKernel10/...___warmu-9e387f39a890.o75:5869` |
| noncausal-s2048-hd64-nh24 | fwd | 104808 | `sim_run_12.8/flashattn-fa3-bf16-bwd-noncausal-b1-s2048-hd64-nh24/.../H100_80GB-OnlyKernel4/...___w-5d6a7701635f.o107:4204` |
| noncausal-s2048-hd64-nh24 | bwd | 184505 | `sim_run_12.8/flashattn-fa3-bf16-bwd-noncausal-b1-s2048-hd64-nh24/.../H100_80GB-OnlyKernel9/...___w-5d6a7701635f.o76:9005` |

### Trace / sidecar status

Original collection logs are historical; newer loader output and copied
trace artifacts supersede the HD128/noncausal failure labels:

| Case | Evidence | Result |
|---|---|---|
| baseline s2048-hd64-nh24 | `.e106/.e80:7-9`; final WaitBlock/TMABIND audits | TMA contracts validated; WaitBlock loaded; K5/K10 complete |
| s1024-hd64-nh24 | Prior `/tmp/flashattn_s1024.log`; `.e103/.e72:7-9` | TMA contracts validated; WaitBlock loaded; K5/K10 complete |
| s2048-hd64-nh8 | Prior `/tmp/flashattn_nh8.log`; `.e104/.e79:7-9`; failed `.e78` | K5/K10 complete; `.e79` supersedes the signals 15/11 failure |
| s2048-hd128-nh24 | `.e100/.e75:7-9`; final WaitBlock/TMABIND audits; device-2 trace artifacts | 75 operand sites / 69 base records loaded; UTMAPF resolved; K5/K10 complete |
| noncausal-s2048-hd64-nh24 | `.e107/.e76:7-9`; final WaitBlock/TMABIND audits; device-4 capture metadata | K4/K9 complete; `.e107` supersedes the `.e101` pipeline assertion and `.e105` generation-watchdog failure |

Copied HD128 and noncausal artifacts are under
`simulator-remodeled/hw_run/traces/device-<D>/12.8/<case>/<run>/`, with D=2
and D=4 respectively. The run name is the case with hyphens replaced by
underscores, followed by `___warmup_0___iters_1`. Local HD128 proof details
are in `.dbg/flashattn-hd128-validation.md`; the latest HD128 runs consume the
repaired run-specific sidecar represented by `.e100/.e75`.

Note: the `/tmp/flashattn_*.log` files are transient (overwritten on re-run and
not persisted across reboots). The durable per-run copies live in each trace
directory as `.../traces/collection.stdout.log` / `collection.stderr.log` and
`.../traces/extra_info/*.status.json` when retained/copied.
