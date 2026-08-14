# FA3 bwd — Stall-Reason Alignment (Sim vs HW NCU)

Goal: the FA3 bwd kernel now matches HW on **total cycles** (HW 132,901 vs sim
134,817, +1.4%), but this document audits whether the **stall-reason
distribution** also matches HW. It separates two distinct questions:

1. **Metric (labeling) problem** — does the sim classify a warp's stall into the
   same NCU bucket HW would?
2. **Implementation problem** — after the labels are made HW-consistent, what
   residual distribution difference remains? That residual is a genuine
   simulation-model error, not a reporting artifact.

Bottom line: **the labeling is essentially correct; the residual is an
implementation problem** — the sim holds warps on functional-unit (FU)
throughput **6.6× more** than HW.

---

## 1. Kernel identity (verified)

The progress notes call this "kernel 10 bwd"; the NCU report calls it **ID 9**.
They are the **same kernel** — the off-by-one is only because NCU numbers all
kernels of the process from 0 (9 earlier rng/fill/reduce/preprocess kernels
precede it).

Evidence:
- Sim launch name (`.o48` line 527) and NCU ID 9 name are **identical incl.
  template args**:
  `FlashAttnBwdSm90<CollectiveMainloopBwdSm90<2,2,2,...,C<128>,C<128>,C<64>,bfloat16_t,float,Sm90,1,0,0,0,0,1,0,0,2,1,2,2,0>, CollectiveEpilogueBwd<...,256,0,0,1>, SingleTileBwdLPTScheduler<0,128,0>>`
- Sim: `-filter_first/last_kernel_id = 10`, `trace_kernel_id: 10`, `CTA/core=1
  limited by shmem regs`, 12 warps/CTA.
- NCU ID 9: `gpc__cycles_elapsed.max = 132,901`, grid (384,1,1), block (384,1,1)
  = 12 warps/CTA.
- (For reference NCU ID 4 = 67,696 cyc = the fwd "kernel 5".)

Run under audit: sim `.o48` (Opt-12 spin-wait-fold, `gpu_sim_cycle = 134,817`).
NCU report: `/modern-gpu-simulator-micro-2025/nv_reports/h100/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.ncu-rep`
(inside `gpu-sim` docker; `ncu` at `/usr/local/cuda-12.8/bin/ncu`).

---

## 2. How each side is obtained

- **Sim**: the sim already emits an NCU-aligned stall stack in `shader.cc`
  ([shader.cc:1653-1725](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L1653-L1725)),
  `ncu_stall_<reason>` + `_pct`, one line per NCU reason. `.o48` lines 7968+.
- **HW**: `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`,
  extracted from NCU ID 9 via
  `ncu -i ... --page raw --csv --launch-skip 9 --launch-count 1 --metrics ...`.

### 2a. The `mma` vs `gmma` naming (corrected)

HW **does** have a tensor-core stall reason; on Hopper it is named **`gmma`**
(`smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio`), not `mma`.
An earlier claim that "HW has no mma stall" was wrong — it was queried under the
wrong name. So the sim's `mma` line (tensor-pipe busy) maps to HW `gmma`.

---

## 3. Distribution comparison (each side normalized to 100%)

Note on normalization: HW ratios are warp-cycles-per-issued-inst (sum = CPI).
The sim's `ncu_stall_*_pct` are % of total scheduler cycles and its per-reason
flags are a per-cycle boolean-OR (reasons overlap, sum ≈ 152.8%). So the table
below normalizes **each side to 100%** and compares **shape vs shape**, not
absolute %.

| reason | HW % | Sim % | verdict |
|---|---:|---:|---|
| barrier | 17.4 | 12.0 | close |
| long_scoreboard | **19.8** | 7.9 | **sim under** |
| short_scoreboard | 8.5 | 0.1 | **sim ≈ 0** |
| wait | 10.4 | 5.4 | sim under |
| gmma (mma) | 5.3 | 10.5 | sim ~2× over |
| **math_pipe_throttle** | **1.2** | **25.3** | **sim massively over** |
| mio_throttle | 6.3 | 0.1 | sim under |
| no_instructions | 1.5 | 6.7 | sim over (L1I frontend) |
| sleeping | 4.4 | 1.7 | sim under |
| dispatch_stall | 4.5 | 0.6 | sim under |
| not_selected | 5.4 | 9.4 | close |
| selected (issued) | 13.3 | 20.3 | sim over |
| imc_miss | 1.3 | 0.1 | sim under |
| branch_resolving | 0.8 | 0.0 | (no branch unit in trace sim) |
| drain/membar/lg/tex/misc | ~0 | 0 | match |

The single dominant discrepancy is the **compute/pipe axis**: sim
`math_pipe_throttle`(25.3) + `mma`(10.5) = **35.8%** vs HW
`gmma`(5.3) + `math_pipe`(1.2) = **6.5%**.

---

## 4. Where the sim decides the label (code)

Mapping in `shader.cc`
([shader.cc:1682-1704](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L1682-L1704)):

```
math_pipe_throttle <- fu_occ_sfu + fu_occ_sp_int_dp   // SFU + SP/INT/DP pipe busy
mma                <- fu_occ_tensor                    // tensor pipe busy
long_scoreboard    <- wait_barrier + tma_flush
short_scoreboard   <- scoreboard + l1c
wait               <- stall_count
mio_throttle       <- result_queue_full
barrier            <- inst_barrier + mbarrier_spin
```

The `fu_occ_*` flags are set in the issue scan
([subcore.cc:960-993](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L960-L993)),
gated on `!is_fu_available` where
([subcore.cc:758-773](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L758-L773)):

```
is_fu_available = fu->can_issue(pI);   // false iff the FU is still counting
                                       // down a prior op's initiation_interval (II)
```

So `fu_occupied` is **purely a pipe-throughput (II) wait** — which is exactly
what HW's `gmma` / `math_pipe_throttle` mean. **The label is conceptually
correct.** (It is NOT a data-dependency/scoreboard wait, so the earlier idea of
re-routing it to `short_scoreboard` was wrong and is retracted.)

Stall flags are non-exclusive (a warp missing several conditions increments
several buckets), matching how NCU's own `smsp__pcsamp` reasons overlap.

---

## 5. Metric vs implementation — the decisive measurement

The sim already carries a **definition-matched** probe,
`-overlap_instrument_enable`
([subcore.cc:838-877](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L838-L877)):
per-warp warp-cycle counters on the **same basis** as HW's
`per_issue_active` (each stalled valid-head warp contributes 1 warp-cycle to its
reason; denominator = issued warp-insts `gpgpu_n_tot_w_icount`). Values from
`.o48` (denominator `gpgpu_n_tot_w_icount = 19,911,524`):

| sim warp-cyc counter | raw | /issued (HW-basis) |
|---|---:|---:|
| `fu_tensor` | 17,294,182 | **0.869** |
| `fu_sfu` | 30,161,607 | **1.515** |
| `fu_other` | 17,447,082 | **0.876** |
| `fu_any` (sum) | 64,902,871 | **3.260** |
| `wait_barrier` | 5,045,807 | 0.253 |
| `stall_count` | 7,153,010 | 0.359 |
| `yield` | 2,124,036 | 0.107 |
| `scoreboard` | 0 | 0.000 |

Directly comparable to HW ID 9:

| axis | HW ratio | Sim ratio | sim/HW |
|---|---:|---:|---:|
| **FU-throughput total** (HW `gmma`+`math_pipe` vs sim `fu_any`) | **0.490** | **3.260** | **6.6×** |
| ├ tensor (HW `gmma` vs sim `fu_tensor`) | 0.398 | 0.869 | 2.2× |
| ├ SFU (in HW `math_pipe`) vs sim `fu_sfu` | — | 1.515 | (dominant) |
| └ SP/INT/DP (in HW `math_pipe`) vs sim `fu_other` | — | 0.876 | — |
| long_scoreboard | 1.491 | 0.000 (trace mode) | sim ~0 |
| mio_throttle | 0.474 | (folded, ~0) | sim ~0 |

This is on **identical basis**, so it is not a normalization artifact.

---

## 6. Conclusions

### 6a. Metric (labeling) problem — essentially NONE
- `fu_occupied → math_pipe_throttle / mma` is the correct NCU bucket
  (FU II wait == HW `gmma`/`math_pipe_throttle`). No re-routing to
  scoreboard/wait is warranted.
- Minor precision nit only: the sim could split `fu_tensor` into its own HW-named
  `gmma` line (already effectively done via the `mma` line) and keep
  `fu_sfu + fu_other` as `math_pipe_throttle`. This refines the two labels but
  **does not change the distribution conclusion** — the totals are already what
  they are.
- The one genuine label limitation is `scoreboard = 0` and `mio_throttle ≈ 0`:
  in trace mode the traditional scoreboard is off
  ([subcore.cc:736-744](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L736-L744)),
  so the sim never *produces* a `long_scoreboard`/`short_scoreboard`/`mio_throttle`
  event to label. That is not a mislabel — it is reason 6b.

### 6b. Implementation problem — the real cause
- Sim holds warps on FU throughput **6.6× more** than HW (`fu_any` 3.26 vs 0.49),
  dominated by **SFU** (`fu_sfu` 1.515).
- Meanwhile HW spends its stalls on **data-wait axes the sim barely produces**:
  `long_scoreboard` 1.49 (global-load result wait) and `mio_throttle` 0.47
  (shared/MIO queue) — sim ≈ 0 on both.
- Interpretation: HW hides FU-throughput latency by **staggering its 12 consumer
  warps across pipe stages**, so a warp's stall surfaces as "waiting for its
  load/shared result" (scoreboard/mio). The sim runs 1 CTA/SM with the 12
  consumer warps in **lockstep**, so they all present their head to the same
  SFU/tensor pipe at once and pile up as "FU busy" (II countdown). Same total
  wait, wrong axis.
- This is the **exact axis already identified and deferred** in
  `FA3_progress.md` as *"consumer-warp lockstep / pipe-overlap absence"* and in
  [WARP_STAGGER_LOCKSTEP.md](WARP_STAGGER_LOCKSTEP.md) /
  [CONSUMER_COMPUTE_BOUND.md](CONSUMER_COMPUTE_BOUND.md). Total cycles matched
  only because other corrections (spin-fold etc.) compensated; the stall *shape*
  still reflects the lockstep.

### 6c. Do we need a new metric / re-run?
- **Not required to reach the above conclusion** — the `overlap_instrument`
  probe already gives the HW-basis split, and `.o48` contains it.
- Optional (labeling-only) follow-up: emit the HW-basis `fu_tensor/fu_sfu/fu_other`
  ratios directly in the `ncu_stall_*` block so the reported stack lines up 1:1
  with HW without hand-computation. Timing-neutral. Would NOT close 6b.

---

## 7. Open next steps (implementation axis, if pursued)

1. Confirm the lockstep mechanism quantitatively: run with
   `-mufu_lockstep_instrument_enable` and read `head_mufu / valid_head`
   (≈1 ⇒ lockstep) plus `nonsfu_{wait_barrier,stall_count,scoreboard}` from the
   `total_num_mufu_lockstep_*` counters.
2. Cross-check whether the 6.6× is II magnitude vs pile-up: compare the
   `[LATCFG]` II values (`sfu`, `tensor`, `tensor_init`) against HW instruction
   throughput (NCU `sm__pipe_*_cycles_active` / inst-executed). Prior notes hold
   SFU II=8 and the tensor rate as HW-faithful, which would make the 6.6× a
   pile-up (lockstep) effect, not a per-op-cost error.

## 8. Raw HW numbers (NCU ID 9, per_issue_active.ratio)

```
barrier            1.313106      long_scoreboard    1.491451
branch_resolving   0.060231      math_pipe_throttle 0.091678
dispatch_stall     0.337559      mio_throttle       0.474407
drain              0.001395      not_selected       0.405119
gmma               0.398055      selected           0.999982
imc_miss           0.094410      short_scoreboard   0.642884
lg_throttle        0.000000      sleeping           0.328385
membar             0.000000      wait               0.783902
misc               0.001876      tex_throttle       0.000000
no_instruction     0.115050
```
