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

#### SR-M0A.1 Scope and non-goals

SR-M0A is an accounting change, not an NCU-priority model and not a timing
change. For every active warp on every Subcore issue cycle, it must record
exactly one population entry. It must not:

- change scheduler order, eligibility, issue selection, or WaitBlock behavior;
- call a mutating readiness function from the observation path;
- retain per-cycle history or add a runtime queue;
- project blocked warps into final NCU reasons.

The implementation is gated by
`-stall_reason_align_instrument_enable`, default off. With the gate off, the
binary must follow the original path without an additional scan or branch in
the per-warp loop beyond the top-level disabled gate.

#### SR-M0A.2 Why the existing tail scan cannot be reused

Do not extend the existing `tail_readonly` counter path in
`remodeling/subcore.cc`. It can start after either:

- a winner has issued; or
- a greedy warp's L1C miss prevents switching to another warp.

In the first case, the winner may already have changed FU and result-queue
availability. In both cases, the tail path forces the side-effecting
programmer-barrier and TMA-flush checks to ready and forces L1C ready:

```cpp
tail_readonly ? true : !c_warp->waiting()
tail_readonly ? true : !shared_sm->warp_waiting_at_tma_flush(...)
tail_readonly ? true : are_l1c_operands_ready(...)
```

The `overlap_instrument` path is also unsuitable: it observes only valid-head
non-issuing warps and independently increments every true reason. It is useful
as condition-incidence telemetry but cannot produce a complete, exclusive
population.

SR-M0A therefore needs a separate cycle-start snapshot. Keep the old counters
unchanged for compatibility until the new accounting passes its invariants.

#### SR-M0A.3 Capture and finalize points

`Subcore::cycle()` executes `control_stage()` before `issue()`, so the
`ISSUE_CONTROL` latch already reflects this cycle's upstream pipeline state.
Inside `issue()`, capture immediately after `modify_warp_state()`:

```text
writeback -> execute -> read_rf -> allocate -> control_stage
                                              |
                                              v
issue:
  modify_warp_state()       dependency decay + WaitBlock release
  capture snapshot          read-only, cycle-start issue state
  existing issue path       remains authoritative
  finalize snapshot         use actual winner ID
decode -> fetch
```

Capture must occur before
`m_num_pending_cycles_with_issue_port_busy` is decremented. A positive value
blocks the current cycle; observing the post-decrement value would shift the
metric by one cycle.

```cpp
void Subcore::issue(SM *shared_sm) {
  modify_warp_state();

  if (m_config->stall_reason_align_instrument_enable) {
    // m_stall_cycle_snapshot.samples was sized once in the constructor.
    capture_pre_issue_snapshot(shared_sm, m_stall_cycle_snapshot);
  }

  int issued_subcore_warp_id = -1;
  int issued_sm_warp_id = -1;
  address_type issued_pc = 0;

  // Existing scheduling and issue logic is unchanged. Immediately before its
  // one issue_warp() call, save sm_warp_id and pI->pc.

  if (m_config->stall_reason_align_instrument_enable) {
    finalize_stall_snapshot(m_stall_cycle_snapshot,
                            issued_subcore_warp_id,
                            issued_sm_warp_id,
                            issued_pc);
  }
}
```

The snapshot never chooses a winner. The existing issue path remains the sole
authority for scheduling and all state transitions.

#### SR-M0A.4 Population states and classification precedence

Use one enum-valued state per warp rather than independent population bitsets.
This makes mutual exclusion structural: a warp cannot accidentally be counted
as both `drain` and `no_valid_instruction`.

The pre-issue state is:

```cpp
enum class StallPreIssueClass : uint8_t {
  INACTIVE,
  WAIT_BLOCK_HOLD,
  NON_FRONTEND_HOLD_NO_HEAD,
  DRAIN,
  NO_VALID,
  BLOCKED_VALID,
  ELIGIBLE_CANDIDATE,
  UNKNOWN_VALID,
};
```

Final output counters use a separate enum:

```cpp
enum class StallPopulation : uint8_t {
  SELECTED,
  ELIGIBLE_NOT_SELECTED,
  NO_VALID_INSTRUCTION,
  DRAIN,
  WAIT_BLOCK_HOLD,
  BLOCKED_WITH_VALID_HEAD,
  NON_FRONTEND_HOLD_WITHOUT_VALID_HEAD,
  UNKNOWN_VALID_HEAD,
  COUNT,
};
```

`unknown_valid_head` is a fail-closed instrumentation state, not an NCU state.
It keeps conservation exact when a readiness predicate cannot be observed
without mutation. A run is not accepted as SR-M0A-complete unless this counter
is zero.

An active warp is:

```cpp
warp != nullptr && !warp->done_exit()
```

Classify each active warp in this order:

1. Read WaitBlock, architectural-wait, IBuffer, and trace state.
2. If an unresolved WaitBlock pending wait exists, classify
   `wait_block_hold`. Do not inspect a future head's FU state.
3. If there is no valid head but a named barrier, MEMBAR, grid barrier,
   nonzero `stall_count`, or unresolved yield hold is blocking the warp,
   classify `non_frontend_hold_without_valid_head`. These conditions are
   observable without a head instruction and must not inflate
   `no_valid_instruction`.
4. If there is no valid head, the IBuffer is empty, all canonical and partial
   replay sources are exhausted, and there is no matching decode or fetch
   request that can still produce a head, classify `drain`.
5. Classify every remaining no-valid-head case as `no_valid_instruction`;
   this includes decode/fetch work that is still able to produce a head.
6. If a valid head has one or more known blockers, classify
   `blocked_with_valid_head`.
   A definite blocker wins even when another predicate, such as L1C, is
   unknown.
7. If a valid head has no known blocker but at least one unknown readiness
   result, classify `unknown_valid_head`.
8. Otherwise mark it as an internal `eligible_candidate`; finalize converts
   that candidate to `selected` or `eligible_not_selected`.

`trace_done()` alone is not sufficient for `drain`. In trace mode the canonical
cursor advances when an instruction is decoded, so the final decoded
instruction can still be a valid IBuffer head after `trace_done()` becomes
true. A dedicated pure `peek_trace_drained()` predicate must therefore require
an empty IBuffer, exhausted canonical and partial-replay sources, and no
decode/L0I state capable of supplying another instruction. The `done_exit()`
test remains the outer active-warp boundary.

The following combinations indicate a state-machine error and must abort:

```text
wait_block_hold && done_exit
wait_block_hold && non_frontend_hold_without_valid_head
selected && pre_issue_state != eligible_candidate
selected && snapshot_head_pc != issued_pc
```

#### SR-M0A.5 Eligibility inputs

Eligibility is the conjunction of global Subcore gates and warp-local gates.
The cycle-start snapshot must include both global gates:

```text
issue_port_ready   := m_num_pending_cycles_with_issue_port_busy == 0
issue_control_free := m_ISSUE_CONTROL_latch.has_free()
```

For a valid head, capture these warp-local predicates:

```text
SASS wait-barrier ready
stall_count == 0
yield ready
traditional RAW/WAR scoreboard ready
named barrier not waiting
MEMBAR not waiting
grid barrier not waiting
LDGDEPBAR not waiting
TMA flush not waiting
target FU available
fixed-latency result queue available
L1C operands ready
```

The existing scheduler-specific greedy L1C hold controls selection, not the
definition of a warp-local ready state. It is therefore legal for a cycle to
contain `eligible_not_selected > 0` and `selected == 0`; report such cycles as
`eligible_without_issue_cycles` instead of forcing a synthetic winner.

The per-warp readiness facts may be stored internally as a numeric bit mask so
SR-M0B can reuse the representation. SR-M0A uses only `mask == 0` versus
`mask != 0`; it does not emit the mask histogram or map masks to NCU reasons.

#### SR-M0A.6 Non-mutating query contract

Add the following explicit SM queries:

```cpp
bool peek_named_barrier_wait(unsigned warp_id) const;
WaitBlockPendingMode peek_wait_block_hold_mode(unsigned warp_id) const;
bool peek_membar_wait(unsigned warp_id) const;
bool peek_grid_barrier_wait(unsigned warp_id) const;
bool peek_tma_flush_wait(unsigned warp_id,
                         const warp_inst_t *inst) const;
bool peek_trace_drained(unsigned subcore_warp_id) const;
```

Do not reuse `warp_held_on_mbarrier_wait()` for
`peek_wait_block_hold_mode()`. The current implementation delegates to
`warp_waiting_at_barrier()`, which first checks the programmer/named-barrier
set and then `m_pending_sync_waits`; despite its name, it combines both states.
The new WaitBlock-only query must inspect only the pending WaitBlock state and
return `NONE` when no unresolved hold exists. Otherwise it returns one of the
three pending modes:

```text
MISSING_WAIT_HOLD
PARTIAL_MISSING_WAIT_HOLD
CAPTURED_POLL_HOLD
```

`peek_membar_wait()` may inspect the membar flag, scoreboard state, scope, and
pending store counters. It must never clear the membar, invalidate a cache,
update watchdog maps, or emit a release/stuck log.

`peek_tma_flush_wait()` may inspect the opcode and outstanding store-class TMA
transfers. It must never insert into or erase from
`m_warps_waiting_tma_flush`, and it must not emit an episode log.

`peek_named_barrier_wait()` must inspect only `m_barriers`.
`peek_grid_barrier_wait()` must read only the warp's grid-barrier flag.
`peek_trace_drained()` must require no valid head, an empty IBuffer, exhausted
canonical and partial-replay sources, and no matching decode/L0I state capable
of producing a head.
Existing dependency-state queries, scoreboard collision checks,
`functional_unit::can_issue()`, and result-queue capacity checks are already
read-only and may be reused.

The current wait-barrier wrappers construct
`std::vector<Wait_Barrier_Checking>` objects. The snapshot hot path must instead
use a fixed-capacity array plus count, with a non-allocating dependency-state
overload:

```cpp
bool are_wait_barriers_ready(
    const Wait_Barrier_Checking *checks,
    unsigned int num_checks) const;
```

The existing vector-based issue-path API may delegate to this overload. This
keeps the readiness predicate shared without adding per-warp heap allocation.

#### SR-M0A.7 L1C readiness

`are_l1c_operands_ready()` cannot be called by the snapshot. It allocates a
`mem_fetch`, invokes `read_only_cache::access()`, updates cache/LRU/MSHR state,
and may enqueue a miss.

Add a tri-state pure query:

```cpp
enum class PeekReady { READY, BLOCKED, UNKNOWN };

PeekReady peek_l1c_operands_ready(const warp_inst_t *inst) const;
```

For perfect constant-cache configurations or instructions without generated
constant accesses, return `READY`. Otherwise use the const
`tag_array::probe()` path for every constant access without allocating a
`mem_fetch` or updating cache state. Since the authoritative issue path accepts
only `HIT` as ready, any observed non-`HIT` status is `BLOCKED`.

Return `UNKNOWN` when the access mask or address cannot be reconstructed
exactly. If another predicate is definitely blocked, classify the warp as
`blocked_with_valid_head`; the known blocker is sufficient regardless of L1C.
Only a warp with no known blocker and at least one unknown predicate contributes
to `unknown_valid_head`. It must never be counted as
`eligible_not_selected`. An actual winner whose pre-issue L1C state was unknown
is not classifiable; emit the mismatch context and hard-abort before committing
population counters.

For every pre-winner warp on which the authoritative path later calls
`are_l1c_operands_ready()`, compare the pure result with the authoritative
result. Emit cycle/SM/Subcore/warp/PC/status and abort on disagreement. This
detects an incorrect pure probe or an in-cycle cache-ordering dependency rather
than silently corrupting the population.

#### SR-M0A.8 Fixed-storage implementation

SR-M0A must not allocate a queue or retain one record per cycle. Use fixed
totals plus a contiguous scratch vector allocated once in the `Subcore`
constructor and reused on every cycle:

```cpp
struct StallWarpSample {
  StallPreIssueClass state = StallPreIssueClass::INACTIVE;
  unsigned int sm_warp_id = 0;
  address_type head_pc = 0;
  uint32_t blocker_mask = 0;
};

struct StallCycleSnapshot {
  // resize(m_num_warps_per_subcore) exactly once during Subcore construction
  std::vector<StallWarpSample> samples;
  std::array<uint32_t,
             static_cast<size_t>(StallPopulation::COUNT)> population{};
  uint32_t active_count = 0;
  uint32_t eligible_count = 0;
};

struct StallPopulationTotals {
  std::array<uint64_t,
             static_cast<size_t>(StallPopulation::COUNT)> population{};
  uint64_t active_warp_cycles = 0;
  uint64_t eligible_warp_cycles = 0;
  uint64_t sampled_subcore_cycles = 0;
  uint64_t eligible_without_issue_cycles = 0;
  uint64_t winner_mismatch = 0;
  unsigned int measured_kernel_uid = 0;
  bool measured_kernel_uid_valid = false;
};
```

Do not construct or resize `samples` inside `issue()`. Its size is the existing
`m_num_warps_per_subcore`, so the implementation adds no kernel-specific warp
count and does not clear unused SM-wide slots. Every active entry is
overwritten once in the bounded capture pass; only the small scalar counts and
population array are zeroed per cycle.

`StallWarpSample::state` is the sole ownership of the warp's pre-issue
population. Do not mirror `blocked`, `no_valid`, `drain`, and
`wait_block_hold` in independent bitsets. `blocker_mask` is retained only for
winner diagnostics and SR-M0B reuse; SR-M0A does not allocate or update a mask
histogram.

Accumulate by numeric enum index. Do not perform per-warp string-keyed
`m_stats_map` lookups, build `std::vector<Wait_Barrier_Checking>` objects
solely for statistics, or emit per-cycle logs.

Only the fixed totals survive the cycle. Text names are resolved at final
print time. The 65,536-entry blocker-mask histogram belongs to SR-M0B and is
not allocated by SR-M0A.

#### SR-M0A.9 Finalization and winner validation

Record the actual Subcore-local warp index, SM warp ID, and PC immediately
before the existing `issue_warp()` call. Keeping the local index avoids a
second search or ID translation during finalization.

```cpp
const bool issued = issued_subcore_warp_id >= 0;
if (issued) {
  const StallWarpSample &winner =
      snapshot.samples[issued_subcore_warp_id];
  if (winner.sm_warp_id != issued_sm_warp_id ||
      winner.state != StallPreIssueClass::ELIGIBLE_CANDIDATE ||
      winner.head_pc != issued_pc) {
    print_stall_align_mismatch_and_abort(...);
  }
}

const uint32_t selected = issued ? 1 : 0;
assert(snapshot.eligible_count >= selected);
const uint32_t eligible_not_selected =
    snapshot.eligible_count - selected;

snapshot.population[
    static_cast<size_t>(StallPopulation::SELECTED)] += selected;
snapshot.population[
    static_cast<size_t>(StallPopulation::ELIGIBLE_NOT_SELECTED)] +=
        eligible_not_selected;
```

Do not require the snapshot to select or predict the exact winner. It validates
that the actual winner belonged to the cycle-start eligible set. Scheduler
ordering remains owned by the existing issue path.

`selected <= 1` is a per-Subcore/SMSP-cycle invariant. It is not an SM-wide
invariant because up to four Subcores may issue in the same SM cycle.

#### SR-M0A.10 Conservation and fail-closed checks

Required per-Subcore-cycle invariant:

```text
selected
+ eligible_not_selected
+ no_valid_instruction
+ drain
+ wait_block_hold
+ blocked_with_valid_head
+ non_frontend_hold_without_valid_head
+ unknown_valid_head
== active_warps

selected <= 1
selected + eligible_not_selected == cycle_start_eligible_warps
active_warps == independently_counted_active_warps
```

Required per-kernel invariants:

```text
sum(population) == total_active_warp_cycles
selected == total_num_cycles_issue_stage_selected
sampled_subcore_cycles == total_num_cycles_issue_stage_evaluated
winner_mismatch == 0
unknown_valid_head == 0
selected > 0 before printing any "/ selected" ratio
```

The selected equality is against issue-active warp cycles, not blindly against
an instruction counter in configurations that permit dual issue. For the
current remodeling Subcore, at most one `issue_warp()` call occurs per cycle,
so the two are expected to coincide; print both values and their delta.

At kernel end, emit raw totals and `/ selected` values, then a grep-friendly
summary:

```text
[STALLALIGN] FINAL status=PASS active=... population_sum=... delta=0
selected=... issue_selected=... eligible=... unknown=0 sampled=...
issue_evaluated=... kernel_uid=... winner_mismatch=0
```

Any per-cycle conservation failure, winner mismatch, counter underflow, or
impossible WaitBlock state prints cycle/SM/Subcore/warp/PC/state/mask and
hard-aborts immediately. A nonzero unknown count makes the final status FAIL
even though arithmetic conservation still holds.

At capture, independently recount active entries and require that count to
equal `m_num_active_warps_subcore`. Also verify each non-inactive sample's
Subcore-local index translates back to its stored SM warp ID. These checks catch
slot-lifecycle and indexing errors before they become plausible-looking global
totals.

#### SR-M0A.11 Boundary of the NCU comparison

SR-M0A provides the denominator and the scheduler population envelope. After
its invariants pass, normalize by `selected` and compare these first-order
quantities:

| SR-M0A output | HW NCU reference |
|---|---:|
| `total_active_warp_cycles / selected` | warp cycles per issued instruction: `7.539490` |
| `selected / selected` | `selected`: `0.999982` |
| `eligible_not_selected / selected` | `not_selected`: `0.405119` |
| `no_valid_instruction / selected` | `no_instruction`: `0.115050` |
| `drain / selected` | `drain`: `0.001395` |

These comparisons validate population accounting, not the detailed reason
distribution. `wait_block_hold` remains a simulator-only timing abstraction:
real HW executes the represented polling instructions and can report several
states during that interval. Do not map the whole hold directly to NCU
`barrier`.

SR-M0A is intentionally insufficient for these NCU states:

```text
barrier
long_scoreboard
short_scoreboard
wait
gmma
math_pipe_throttle
mio_throttle
dispatch_stall
sleeping
branch_resolving
imc_miss
misc
```

The remaining stages have separate responsibilities:

```text
SR-M0B: retain the complete blocker mask; split sole and multi blockers
SR-M1:  recover wait-barrier and stall-count producer provenance
SR-M2:  apply only validated HW priority rules to form one NCU state per warp
```

Only SR-M2 can claim detailed stall-reason alignment. SR-M0A can reject a bad
population basis before that more expensive work is attempted.

#### SR-M0A.12 Reset, output, and implementation ownership

- `shader.h`: add the `shader_core_config` gate.
- `gpu-sim.cc`: register the option and scalar output counters; explicitly
  gather the new direct totals before printing and reset them from the existing
  `gpgpu_sim::reset_gpu_per_sm_stats()` kernel boundary.
- `subcore.h` / `subcore.cc`: define fixed snapshot/totals storage, capture
  cycle-start state, record the actual winner, and finalize.
- `sm.h` / `sm.cc`: add the separated pure wait-state queries.
- `gpu-cache.h` / `gpu-cache.cc`: add the pure L1C probe.
- `shader.cc`: print totals, normalized values, deltas, and final status only.

`reset_gpu_per_sm_stats()` currently resets registered `m_stats_map` entries,
not direct `StallPopulationTotals` members. Add an explicit lifecycle:

```text
Subcore::gather_stall_reason_align_stats(dst)
SM::gather_stall_reason_align_stats(dst)       # folds all Subcores
gpgpu_sim::gather_gpu_per_sm_stats()           # gathers before report
shader.cc                                      # prints the gathered snapshot
gpgpu_sim::reset_gpu_per_sm_stats()
  -> SM::reset_stall_reason_align_stats()
     -> Subcore::reset_stall_reason_align_stats()
```

Do not reset local totals from `SM::reinit()` because an SM can execute several
CTAs from the same kernel in successive waves. Reset once per kernel only after
the per-SM values have been gathered and printed.

The first implementation supports exactly one measured kernel launch at a
time. Stamp the first active sample's `kernel_info_t::get_uid()` into
`measured_kernel_uid` and hard-abort if another UID contributes before reset.
This prevents silent cross-kernel mixing without adding a per-warp map lookup.
If concurrent-kernel measurement is later required, replace this guard with
per-kernel accumulator slots rather than weakening the invariant.

Also correct the existing `overlap_instrument` comments that call its
non-exclusive values "directly comparable" or "definition-matched"; retaining
those comments after SR-M0A would contradict this audit.

#### SR-M0A.13 Performance and acceptance gates

The expected runtime cost is one bounded read-only pass over the Subcore's
resident warps plus fixed-array increments. For Kernel 10, 12 warps/CTA across
four Subcores means roughly three resident warps per Subcore. The implementation
must still satisfy:

```text
gate OFF: bit-identical baseline
gate ON:  identical gpu_sim_cycle and issued work
gate ON:  no state/hash/log differences except SR-M0A counters
wall-clock overhead target: <= 5%
```

The first implementation deliberately keeps the authoritative issue path
independent from the snapshot, so some pure readiness predicates are evaluated
twice for pre-winner warps. Do not feed cached snapshot answers back into
scheduling merely to reduce observer overhead; that would turn the
instrumentation into a timing change. First make the pure helpers
allocation-free and measure their cost.

Validation order:

1. Clean build because public headers change.
2. Focused traces covering selected, not-selected, frontend, drain, each
   WaitBlock hold mode, and an L1C miss.
3. Gate-off versus gate-on bit-identity comparison on a short trace.
4. Run a bounded Kernel 10 prefix with the gate off and on; compare wall-clock
   time over repeated runs. If median overhead exceeds 5%, profile and optimize
   the pure helpers before starting the full run.
5. Full Kernel 10 run with all conservation and winner checks passing.
6. Only after SR-M0A passes, enable SR-M0B mask histogram accounting.

Kernel 5 is not the first acceptance target because its existing cross-warp
deadlock can terminate independently of the observer. It becomes a
non-perturbation and diagnostic test after Kernel 10 passes.

### Stage SR-M0B - blocker-mask accounting (timing-neutral)

For every non-held `blocked_with_valid_head` warp, record a blocker bit mask
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

A WaitBlock hold is a terminal population state in SR-M0: do not inspect the
future head's FU and double-count a held warp as FU-blocked.

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

- `shader.h` / `gpu-sim.cc`: config gate and scalar-stat registration;
- `subcore.h` / `subcore.cc`: blocker-mask capture and histogram commit;
- `sm.h` / `sm.cc`: pure wait-state peek helpers;
- `shader.cc`: mask histogram and sole/multi-blocker output.

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
