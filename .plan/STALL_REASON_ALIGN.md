# FA3 bwd - Stall-Reason Alignment (Sim vs HW NCU)

## 0. Corrected conclusion

The current FA3 bwd WaitBlock-v5 run remains close to HW in total cycles, but
the stall-reason distribution is **not yet definition-matched**:

- HW: `132,901` cycles
- Sim `.o55`: `136,997` cycles (`+3.08%`)
- Previous sim `.o48`: `134,817` cycles (`+1.44%`); `.o55` is
  `+2,180` cycles (`+1.62%`) slower than `.o48`

The current verdict therefore has two separate parts:

- Macro timing and traced work are close: elapsed cycles differ by `3.08%`
  and issued warp instructions by `-0.41%`.
- HW stall-reason alignment is unproven. `.o55` does not contain the SR-M0
  complete, mutually exclusive active-warp accounting required by Section 6.

The previous version of this document concluded that stall labeling was
essentially correct and that the simulator had a proven `6.6x` FU-throughput
implementation error. **That conclusion is retracted.**

The current `.o55` measurements still compare two different state definitions:

- HW `smsp__average_warps_issue_stalled_*_per_issue_active` assigns one
  highest-priority state to each active warp each cycle. The states are
  mutually exclusive and include `selected` and `not_selected`.
- The sim `ncu_stall_*` stack is a set of overlapping per-scheduler-cycle
  boolean-OR conditions.
- The sim `overlap_instrument` probe is per valid-head warp-cycle, but its
  reasons are also overlapping and it does not cover every active-warp state.

Consequently, current sim `fu_any = 3.350` and HW
`gmma + math_pipe_throttle = 0.490` are **not definition-matched values**.
Neither their raw `6.84x` ratio nor the previous `.o48` `6.6x` ratio
proves an FU timing-model error.

The first problem to solve is the stall metric itself: construct a complete,
mutually exclusive per-active-warp state accounting and preserve the
provenance of trace-mode dependencies. Only the residual after that correction
can be called an implementation error.

The shared-memory `gpgpu_n_shmem_bkconflict` mismatch is also a metric-semantics
problem, but a different one. Its `37-135x` raw ratio to HW must not be read as
a `37-135x` timing error. Section 7 defines a timing-neutral metric fix.

---

## 1. Kernel identity

The progress notes call this kernel 10; NCU calls it ID 9. They are the same
kernel. NCU numbers all launches in the process from zero, while the sim filter
uses the trace kernel ID.

Evidence:

- Sim and NCU names have identical template arguments.
- Sim uses `-filter_first_kernel_id 10` / `-filter_last_kernel_id 10`.
- Both launches have grid `(384,1,1)`, block `(384,1,1)`, and 12 warps/CTA.
- Both are limited to one CTA/SM by registers/shared memory.

Run under audit:

- Sim:
  `simulator-remodeled/sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/.../H100_80GB-OnlyKernel10/*.o55`
- HW:
  `nv_reports/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.ncu-rep`

Run-integrity evidence for `.o55`:

- all `384` CTAs completed;
- all `384` `[WAITBLOCK][summary]` records report `status=PASS`;
- aggregate WaitBlock watchdog and residual active-state counters are zero;
- `.e55` has no fatal, abort, assertion-failure, or deadlock marker;
- the log ends with `GPGPU-Sim: *** exit detected ***`.

The printed configuration replaces the old `spin_wait_fold_enable/debug`
controls with `wait_block_enable/debug`. Therefore `.o55` is the current
WaitBlock-v5 timing result, not a bit-identical rerun of `.o48`.

---

## 2. Macro timing and issue density remain close

The cycle comparison is supported by matched work and issue density, not just
by the final elapsed-cycle scalar.

| quantity | HW | Sim `.o55` | sim/HW |
|---|---:|---:|---:|
| elapsed cycles | 132,901 | 136,997 | 1.0308 |
| issued warp instructions | 19,993,658 | 19,911,524 | 0.9959 |
| issued inst / scheduler / elapsed cycle | 0.284925 | 0.275270 | 0.9661 |
| issue slots busy | 32.07% | 29.97% | contextual; denominator audit still required |
| eligible warps / scheduler | 0.46 | 0.4409 | contextual; sim tail scan is approximate |

The matched-basis identity is:

```text
cycle ratio
= work ratio x inverse issue-density ratio
= 0.995892 x 1.035072
= 1.030820
```

This does not prove that every internal timing mechanism is correct. Different
modeling errors can compensate. It does show that the aggregate scheduler
readiness behavior is already close, so a claimed large internal error requires
stronger evidence than the current overlapping reason counters.

The sim's printed `ncu_no_eligible_pct = 70.03%` is intentionally not included
as a matched metric. It is currently computed as
`100% - ncu_issue_slots_busy_pct`, so it also includes non-issue causes other
than "zero eligible warps." SR-M0 must replace this proxy with a direct
per-scheduler eligible-warp population.

---

## 3. Why the old distribution comparison was invalid

### 3.1 HW warp states are mutually exclusive

On each cycle, each active HW warp reports one highest-priority state. A warp
that issues reports `selected`; another eligible warp reports `not_selected`;
an ineligible warp reports one stall reason.

For NCU ID 9, the raw states in Section 9 sum to:

```text
sum(HW states) = 7.539490 warp-cycles / issued instruction
```

NCU reports `Warp Cycles Per Issued Instruction = 7.53`. The small difference
is rounding. This conservation is direct evidence that these HW values form an
exclusive state accounting.

Reference:
<https://forums.developer.nvidia.com/t/stall-reasons-summation-is-not-100/168096/7>

### 3.2 Sim `ncu_stall_*` is not the same axis

The `shader.cc` block emits:

```text
ncu_stall_<reason> = cycles where at least one warp met that condition
```

Reasons can overlap in one scheduler cycle. The denominator is evaluated
scheduler cycles, not issued instructions or active warp-cycles. Normalizing
this stack to 100% does not make it equivalent to the HW state vector.

Therefore the old shape table, which normalized the HW exclusive vector and
the sim overlapping vector independently, has been removed.

### 3.3 Sim `overlap_instrument` is closer, but still not matched

The probe in
`subcore.cc` counts every valid-head, non-issuing warp and independently tests:

```text
wait_barrier
stall_count
yield
fu_tensor / fu_sfu / fu_other
traditional scoreboard
```

A warp can increment several reasons in one cycle. In particular,
`!is_fu_available` is recorded even when another condition already prevents
the warp from issuing.

The probe also omits active warps without a valid head from its reason vector.
The available partial-population cross-check is:

| sim state population | warp-cycles / issued inst |
|---|---:|
| selected | 1.000000 |
| not_selected | 0.471181 |
| valid-head non-issuing (`stalled_any`) | 4.934007 |
| **observed partial total** | **6.405187** |
| HW active-warp state reference | **7.539490** |

The numeric difference to HW is `1.134303` warp-cycles/inst, but this must
**not** be called a measured sim coverage gap. The simulator does not report
its own total active-warp population, and real HW/sim occupancy can differ.
The table only shows that the available sim counters cannot establish their
own conservation or reconstruct the HW vector. No per-reason comparison is
definition-complete until SR-M0 adds the missing sim-side total and categories.

---

## 4. Reinterpretation of the FU counters

The raw `.o55` probe values remain useful as **condition-incidence**
diagnostics:

| sim condition | raw warp-cycles | / issued |
|---|---:|---:|
| `fu_tensor` | 17,660,053 | 0.887 |
| `fu_sfu` | 31,081,995 | 1.561 |
| `fu_other` | 17,955,055 | 0.902 |
| `fu_any` | 66,697,103 | 3.350 |
| `wait_barrier` | 5,121,159 | 0.257 |
| `stall_count` | 7,146,796 | 0.359 |
| `yield` | 2,123,549 | 0.107 |

They do **not** yet answer how often FU availability was the warp's exclusive
or highest-priority blocker. The required split is:

```text
FU-only
FU + dependency
FU + barrier/park
FU + frontend/other
```

Only `FU-only` is priority-independent evidence that the target pipe itself
prevented issue. Multi-blocker states require an NCU-consistent priority rule
or must remain explicitly ambiguous.

### 4.1 `fu_occupied -> math_pipe/gmma` is conditional, not globally proven

When FU unavailability is the primary reason, the target-pipe mapping is
conceptually reasonable:

```text
tensor target      -> gmma
SFU/SP/INT/DP target -> math_pipe_throttle
```

The error was not the target-pipe name. The error was treating every
simultaneous `!is_fu_available` observation as if HW would necessarily choose
that FU state over all other unmet conditions.

### 4.2 `.o55` does not confirm the old lockstep claim

On still-idle cycles with at least one SFU-head warp:

| valid-head class | raw | share |
|---|---:|---:|
| MUFU/SFU | 25,985,715 | 69.30% |
| tensor | 3,232,340 | 8.62% |
| other | 8,281,914 | 22.09% |

`head_mufu / valid_head` is `0.693`, not approximately one. This shows an
SFU-heavy head population, but it does not show that every available warp was
at the same MUFU stage.

The SM-level execution-overlap probe also shows substantial simultaneous
tensor/SFU activity:

| pipe state | cycles | share of classified cycles |
|---|---:|---:|
| tensor and SFU | 8,886,447 | 52.55% |
| tensor only | 1,030,405 | 6.09% |
| SFU only | 2,364,731 | 13.98% |
| neither | 4,628,146 | 27.37% |

Consumer phasing can remain a hypothesis, but `.o55` does not establish it as
the explanation for a `6.6x` FU error. Earlier launch-stagger and GTLRR tests
also produced only small effects and must not be cited as positive proof.

---

## 5. Mapping audit: known unsupported folds

| emitted sim label | current source | corrected status |
|---|---|---|
| `gmma` (`mma`) | tensor FU unavailable | plausible only for exclusive/primary FU blocker |
| `math_pipe_throttle` | SFU + SP/INT/DP FU unavailable | plausible only for exclusive/primary FU blocker |
| `long_scoreboard` | `wait_barrier + tma_flush` | unsupported many-to-one fold |
| `short_scoreboard` | traditional scoreboard + L1C | incomplete in trace mode |
| `mio_throttle` | result queue full | not an NCU MIO-input-queue model |
| `barrier` | instruction barrier + folded mbarrier park | fold changes the represented HW state sequence |
| `wait` | trace `stall_count` | approximate; compiler-control timing is not an HW reason identity |
| `no_instruction` | per-scheduler no-valid cycle | not yet a per-active-warp state |

### 5.1 `wait_barrier` is not synonymous with long scoreboard

The provenance probe splits `.o55` `wait_barrier` warp-cycles as:

| arming class | warp-cycles | share |
|---|---:|---:|
| tensor | 0 | 0.00% |
| TMA | 107,140 | 2.09% |
| other | 5,014,019 | 97.91% |
| total | 5,121,159 | 100% |

Folding all of this into `long_scoreboard` is not justified. `other` is too
coarse: it can contain arithmetic, conversion, control, shared-memory, or
other SASS scoreboard dependencies. The arming provenance must be extended
before assigning an NCU reason.

### 5.2 Trace mode removes dynamic scoreboard provenance

With `-is_remodeling_scoreboarding_enabled 0`, readiness is reconstructed from
SASS control fields (`wait_barrier`, `stall_count`, yield) plus modeled
functional-unit and barrier state. This can reproduce aggregate issue timing
without preserving the physical HW source that NCU uses to name the state.

The correct response is not to add arbitrary scoreboard latency on top of the
matched cycle count. First recover provenance for the waits already present.

### 5.3 Spin folding changes stall-state shape

HW executes a polling sequence such as `PHASECHK/TRYWAIT`, `NANOSLEEP`, and
`BRA`. During that interval its warp can report selected, wait, sleeping,
scoreboard, or branch-related states.

The fold model parks the warp until modeled mbarrier readiness and records that
interval primarily as `mbarrier_spin -> barrier`. This is a valid timing
abstraction, but not a state-by-state reproduction of the executed polling
loop. Folded park cycles must be reported separately until a validated shadow
state model exists.

---

## 6. Correct stall-metric implementation plan

No timing-model parameter should be changed from the current stall stack.

### Stage SR-M0A - complete active-warp population (timing-neutral)

For every active warp every cycle, record exactly one population entry:

```text
selected
eligible_not_selected
no_valid_instruction
drain
fold_parked
blocked_with_valid_head(mask)
```

Do not extend the existing `tail_readonly` counter path. It evaluates tail
warps after the winner has issued, so the winner can already have changed FU
availability and result-queue state. It also forces the side-effecting
programmer-barrier and TMA-flush checks to ready. Those observations cannot
form an exact HW-style state vector.

Capture a cycle-start readiness snapshot after `modify_warp_state()` has
applied dependency decay and serviced fold releases, but before the issue loop
can select a winner:

```cpp
void Subcore::issue(SM *shared_sm) {
  modify_warp_state();

  StallCycleSnapshot snapshot;
  if (m_config->stall_reason_align_instrument_enable) {
    snapshot = capture_pre_issue_snapshot(shared_sm);
  }

  // Existing scheduling and issue path remains unchanged.
  run_existing_issue_path_and_record_winner();

  if (m_config->stall_reason_align_instrument_enable) {
    finalize_stall_snapshot(snapshot, issued_warp_id);
  }
}
```

An active warp is `warp != nullptr && !warp->done_exit()`. A warp whose trace
has drained but whose architectural resources have not retired belongs to
`drain`, not frontend `no_instruction`.

The snapshot needs pure, non-mutating SM queries:

```text
peek_named_barrier_wait(warp_id)
peek_fold_park(warp_id)
peek_membar_wait(warp_id)
peek_grid_barrier_wait(warp_id)
peek_tma_flush_wait(warp_id, inst)
```

`peek_membar_wait()` may inspect pending stores but must never clear the
membar, invalidate a cache, update watchdog maps, or emit an episode log.
`peek_tma_flush_wait()` may inspect outstanding store-class TMA transfers but
must not mutate `m_warps_waiting_tma_flush`.

L1C readiness currently uses the mutating `read_only_cache::access()` path. Add
a pure tag/MSHR/resource probe if it can reproduce the same ready predicate. If
that cannot be proven, record `l1c_unknown` instead of treating a tail warp as
ready and inflating `not_selected`.

Required per-cycle invariants:

```text
selected
+ eligible_not_selected
+ no_valid_instruction
+ drain
+ fold_parked
+ blocked_with_valid_head
== total active warp-cycles

selected <= 1
selected + eligible_not_selected == eligible_warps
```

After division by issued instructions, the population sum must reconstruct the
sim's own `warp_cycles_per_issued_instruction`. This is the mandatory gate
before comparing reason distributions.

Any disagreement between the pre-issue snapshot and the actual winner is a
metric implementation bug. Emit the warp/PC/mask and hard-abort rather than
continuing with corrupted statistics.

### Stage SR-M0B - blocker-mask accounting (timing-neutral)

For every non-parked `blocked_with_valid_head` warp, record a blocker bit mask
from the same pre-issue snapshot:

```text
yield
stall_count
SASS wait_barrier
named barrier
membar
grid barrier
TMA flush
FU tensor
FU SFU
FU SP/INT/DP
traditional scoreboard
L1C
result queue
issue port busy
ISSUE_CONTROL full
frontend
```

Fold parking is a terminal population state in SR-M0: do not inspect the
future head's FU and double-count a parked warp as FU-blocked.

Classify masks without inventing an HW priority:

```text
popcount(mask) == 1 -> sole blocker
popcount(mask) > 1  -> explicit multi-blocker mask
mask == 0           -> eligible_not_selected, unless it is the winner
```

`FU-only` by target pipe is the first priority-independent quantity that can
be compared with HW `gmma`/`math_pipe_throttle`.

The hot path must not perform string-keyed `m_stats_map` lookups per warp.
Represent conditions with a numeric enum and accumulate into a fixed
GPU-global mask histogram. A 16-bit mask needs 65,536 `uint64_t` counters
(512 KiB) globally; it must not be replicated per warp. Convert mask IDs to
text only at final print time.

Keep PC/opcode attribution out of SR-M0B's every-cycle hot path. Add it later
behind a separate gate, limited to dominant masks or selected SMs/CTAs.
Aggregate role (`producer/consumer/other`) with fixed arrays.

All new scalar counters and histograms must reset at each kernel launch.

Implementation sites:

- `gpu-sim.h` / `gpu-sim.cc`: config gate and scalar-stat registration;
- `subcore.h` / `subcore.cc`: snapshot structs, masks, capture, and finalize;
- `sm.h` / `sm.cc`: pure wait-state peek helpers;
- `shader.cc`: conservation report and mask histogram output.

Header edits require a clean build.

### Stage SR-M1 - priority-independent splits and dependency provenance

First report sole blockers, which do not require guessing HW priority:

```text
FU-only by target pipe
wait_barrier-only
stall_count-only
barrier-only
frontend-only
multi-blocker masks
```

The current `Wait_Barrier` stores only the most recent `armed_op`. That is
lossy: several operations can arm the same SB, and a later arm overwrites the
provenance of older pending credits. Replace the single enum with per-class
pending counts:

```text
tensor
global-memory
TMA
shared-memory/MIO
SFU
SP/INT/DP
control/synchronization
other
```

Both increment and completion/decrement actions must carry the instruction's
class. Out-of-order completion then decrements only the matching class. Assert
on every change:

```text
sum(pending_by_class) == wait_barrier.counter
no pending_by_class entry underflows
```

When a head waits on an SB, return a class mask of all pending sources that can
block the requested threshold, not one arbitrarily ranked enum. If several
classes remain, retain the state as ambiguous.

Apply the same provenance rule to trace `stall_count`: record the class and PC
of the instruction that set the current countdown, and clear that provenance
when the counter reaches zero.

Do not force multi-blocker masks into one NCU bucket until the HW
highest-priority rule has been validated, preferably with focused
microbenchmarks that create known blocker combinations.

### Stage SR-M2 - exclusive NCU projection

After SR-M0A/SR-M0B/SR-M1 validation:

1. Project sole blockers directly.
2. Apply only validated priority rules to multi-blocker masks.
3. Keep unresolved masks in an explicit `ambiguous_multi` bucket.
4. Require:

```text
sum(exclusive sim states) == sim warp-cycles per issued instruction
```

Decision gate:

- If exclusive `FU-only` remains materially above HW
  `gmma + math_pipe_throttle`, investigate FU topology, routing, II, and
  per-op active cycles.
- If raw `fu_any` is large but exclusive `FU-only` is close to HW, the old
  discrepancy was a metric-overlap artifact.
- If memory/MIO provenance is absent rather than merely mislabeled, that
  residual is an implementation-model gap.

---

## 7. Shared-memory metric correction

### 7.1 What the current numbers do and do not mean

Observed values:

| kernel | sim `gpgpu_n_shmem_bkconflict` | HW store bank conflicts | raw ratio |
|---|---:|---:|---:|
| FA3 fwd | 38,016 | 281 | 135x |
| FA3 bwd | 1,327,104 | 35,493 | 37x |

These values are not definition-matched.

The sim counter increments in `ldst_unit_sm.cc` on each PRT replay cycle while
the shared instruction's dispatch delay remains nonzero. It is therefore
closer to a dispatch-delay replay-cycle counter.

HW `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` counts extra
conflict wavefronts/events, not the base service cycles of the instruction.

The actual sim dispatch delay per shared instruction is only:

| kernel | `total_accesses_per_shared_instruction` |
|---|---:|
| fwd | 3.3607 |
| bwd | 3.0377 |

Wide shared instructions also have an inherent width cost. A 16-byte access
requires multiple ideal phases/wavefronts even when every phase is
conflict-free. HW does not count those base wavefronts as bank conflicts.
Thus the `37-135x` raw ratio is not a `37-135x` latency error. As documented in
`FA3_progress.md`, the current sim may even understate the width-based base
serialization while over-reporting the value named "bank conflict."

This metric must not be used to explain the current `mio_throttle` or
`short_scoreboard` gap.

### 7.2 Stage SM-M0 - add NCU-aligned observe-only counters

Do not change `cycles = total_accesses` in this stage.

Keep the legacy counter for compatibility, but print an explicit alias/name
that describes its unit:

```text
shmem_dispatch_delay_replay_cycles
```

Do not increment the new metric in the PRT replay loop at
`ldst_unit_sm.cc:1867-1879`; that is exactly where the legacy counter changes
units from instructions/wavefronts to replay cycles. Analyze and commit the
metric once when a shared-memory stage is admitted, after traced lane
addresses and `data_size` are available.

Add a pure helper:

```cpp
struct ShmemWavefrontStats {
  uint32_t ideal;
  uint32_t actual;
  uint32_t extra;
  uint32_t width_bytes;
  bool is_load;
  bool is_store;
  bool classified;
};

ShmemWavefrontStats analyze_shmem_wavefronts(const warp_inst_t &inst);
```

Store the result on the instruction or pass it immediately to an SM-level
numeric accumulator. Add an explicit per-shared-stage recorded guard because
`LDGSTS` regenerates memory accesses when it changes from its global-load
stage to its shared-store stage. The global stage must not suppress the later
shared stage, while the shared stage must contribute exactly once.

At shared-memory access generation, compute a separate wavefront analysis once
per instruction. Do not reuse the old proposal that applied `addr + 4*p` to all
32 lanes: for a 16-byte-per-lane request, HW first partitions the warp request
into base wavefront-sized lane groups. Applying word phases across the full
warp can classify a conflict-free vector request as conflicting again.

```text
bank_word_bytes      = WORD_SIZE
base_wavefront_bytes = num_shmem_banks * bank_word_bytes
words_per_lane       = ceil(data_size / bank_word_bytes)
lanes_per_wavefront  = min(subwarp_size,
                           max(1, floor(base_wavefront_bytes / data_size)))
```

For ordinary vector `LDS/STS`, partition lanes by their architectural lane
index into contiguous base-wavefront groups. For every non-empty group:

1. Expand each active lane's traced start address into all touched bank words.
2. Apply `shmem_bank_func()` to every expanded word.
3. Apply the existing broadcast rule for identical words.
4. Set the group's service depth to the maximum number of distinct words
   targeting any one bank.

Then accumulate:

```text
ideal_wavefronts  = number of non-empty base-wavefront groups
actual_wavefronts = sum(group_service_depth)
extra_conflict_wavefronts = actual_wavefronts - ideal_wavefronts
```

For example, a full-warp 16-byte request contains 512 bytes and therefore four
ideal 128-byte base wavefronts. If every 128-byte group reaches all 32 banks
without duplication, `ideal=4`, `actual=4`, and `extra=0`. The four base
wavefronts are width cost, not bank conflicts.

The `num_shmem_banks * WORD_SIZE = 128B` grouping is the H100 metric-model
hypothesis to validate, not a value to tune against FA3 cycles. SM-M0 must first
confirm it with raw NCU wavefront counts and focused 4B/8B/16B shared-memory
microbenchmarks.

`LDSM/STSM` must not be assumed to use the generic contiguous-lane grouping
without validation. Their decoded matrix shape and trace-address convention
must either produce a verified instruction-family grouping or remain in an
explicit `shmem_wavefront_unclassified_matrix` bucket. Before enabling the
counter for those instructions, dump representative lane addresses and verify
the reconstructed ideal/actual wavefronts by hand and with a focused
microbenchmark.

Emit load/store counters separately:

```text
shmem_instructions_{ld,st}
shmem_ideal_wavefronts_{ld,st}
shmem_actual_wavefronts_{ld,st}
shmem_bank_conflict_extra_wavefronts_{ld,st}
shmem_access_width_{4,8,16,...}_bytes
```

Do not reuse `gpgpu_n_shmem_insn` as the instruction denominator: that existing
counter increments by active lanes. Add explicit warp-instruction counters for
the NCU comparison.

Required invariants:

```text
actual_wavefronts >= ideal_wavefronts
extra_conflict_wavefronts
  == actual_wavefronts - ideal_wavefronts
each shared instruction contributes exactly once
```

Validation against raw NCU metrics:

- `extra_conflict_wavefronts_st` vs
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`
- load equivalent if available;
- `actual_wavefronts_{ld,st}` vs raw shared LSU wavefront counts;
- width histogram vs the traced `LDS/STS/LDSM/STSM` instruction mix.

Timing-neutral gate:

```text
gpu_sim_cycle, gpgpu_n_tot_w_icount, HGMMA, MUFU, TMA
must remain bit-identical
```

Implementation sites:

- `abstract_hardware_model.h/.cc`: analysis result and address/width expansion;
- `ldst_unit_sm.cc` or the shared-stage admission owner: once-only commit;
- `gpu-sim.cc`: gate and numeric counter registration;
- `shader.cc`: legacy alias, new counters, ratios, and invariants.

Avoid string map lookups inside per-lane/per-word loops. Build the local
`ShmemWavefrontStats` with stack arrays, then update fixed numeric counters
once per instruction.

### 7.3 Stage SM-M1 - timing model only after metric validation

Only after SM-M0 matches the HW metric definitions should a gated timing
experiment consider:

```text
cycles = actual_wavefronts
```

This can increase the base cost of wide conflict-free instructions even while
reducing extra conflict cost. It must not be presented as a guaranteed
speedup. The current `FA3_progress.md` conclusion remains in force: this is a
metric/architecture-fidelity item, not an established cycle lever.

### 7.4 Separate TMA swizzle limitation

The TMA descriptor carries a shared-memory swizzle mode, but the TMA store path
does not currently model SMEM-bank placement. That is a separate architecture
TODO. It must not be conflated with:

- the legacy `gpgpu_n_shmem_bkconflict` counter;
- L2 slice hashing;
- the SR-M0A/SR-M0B stall-state accounting.

---

## 8. Current decision

1. **Retracted:** the old normalized shape table.
2. **Retracted:** `fu_any 3.260 / HW FU 0.490 = 6.6x` as a
   definition-matched implementation result.
3. **Not supported:** `.o55` proves consumer MUFU lockstep.
4. **Retained as a candidate:** FU incidence is high and may expose a real
   residual after exclusive accounting.
5. **Required next:** SR-M0A, SR-M0B, then SR-M1, with no timing change.
6. **Independent metric fix:** SM-M0 for NCU-aligned shared-memory wavefront and
   extra-conflict counters.
7. **Current `.o55` verdict:** macro timing is close at `+3.08%`, but the
   stall taxonomy cannot pass alignment until SR-M0 demonstrates sim-side
   active-warp conservation and reports sole/multi-blocker masks.

Implementation should proceed one stage at a time with review after each stage.

---

## 9. Raw HW warp states (NCU ID 9)

Units: warp-cycles per issued instruction.

```text
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

```text
sum = 7.539490
NCU Warp Cycles Per Issued Instruction = 7.53
```
