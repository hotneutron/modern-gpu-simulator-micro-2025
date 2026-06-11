# Sync Implementation: Critique of the Actionable Plan — 2026-05-20

## Critique of `Full_sync_impl.md`

### What the plan gets right

The phased dependency chain (A→F) is correct and strictly superior to the
step ordering in the original `H100_sync_impl.md`. Each phase's output feeds the
next, and the verification gates between phases are a sound engineering practice.
The FA3-first rule, the `MBARRIER_OP` / `CLUSTER_BARRIER_OP` split, and the
mbarrier object model are well-designed. The gap analysis of the original plan
is thorough — particularly the UCGABAR misclassification and the
`expected_tx_bytes` initialization gap.

### Gaps and problems

#### 1. Non-TMA ARRIVE logic has a TOCTOU race with TMA binding

Phase C2's ARRIVE handler:

```cpp
if (!obj.tma_bound && obj.expected_tx_bytes == 0) {
    obj.ready = true;
}
```

This is correct **only if ARRIVE always executes after TMA binding**. In FA3's
specific SASS sequence (EXCH → TMA issue → ARRIVE), this holds. But the plan
itself acknowledges (Gap #4) that warp-leader arrive patterns and epilogue
arrives exist. Consider this sequence:

```
1. EXCH        → tma_bound=false, expected_tx_bytes=0, ready=false
2. ARRIVE      → !tma_bound && expected_tx_bytes==0 → ready=true  ✅ premature
3. TMA issue   → sets expected_tx_bytes=N, tma_bound=true
4. TMA complete → completed_tx_bytes >= expected_tx_bytes → ready=true, phase flips
```

The barrier was marked ready at step 2 before TMA completed at step 4. A
consumer that saw `ready=true` at step 2 would proceed before data arrived.

The plan acknowledges this is "Minimum tier / FA3 only" but does not flag the
ordering assumption explicitly or add a guard. Fix: in the TMA-binding step
(E1), when `tma_bound` transitions from `false` to `true`, reset `ready = false`
if `completed_tx_bytes < expected_tx_bytes`. This prevents a premature ARRIVE
from causing a consumer to observe readiness before the transfer completes.

#### 2. Bypass exists in two files — plan only addresses one

The string-prefix bypass hack exists in **both**:

- `sm.cc:453` (remodeling path)
- `shader.cc:2116` (legacy path)

The plan only mentions removing the bypass from `sm.cc`. If the legacy
`shader.cc` path is still active for any configuration or test, the bypass there
will cause SYNCS to silently no-op even after the new `MBARRIER_OP` path is
wired. Either the plan should remove both, or explicitly document that
`shader.cc` is dead code for Hopper traces.

#### 3. `find_mbarrier_mutable` is referenced but never declared

Phase E1 calls `m_sm->find_mbarrier_mutable(cmd.cta_id, barrier_addr)`, but
Phase C1 only declares `find_mbarrier` (const return). The mutable overload is
missing from the interface spec. This will cause a compile error if followed
literally.

Add to Phase C1:

```cpp
HopperMBarrierObject *find_mbarrier_mutable(unsigned cta_id, uint64_t addr);
```

#### 4. `reset_mbarriers_for_cta` is declared but never called

Phase C1 declares this function, but no phase specifies **where** it should be
called. The natural call site is CTA completion / deallocation. Without this,
mbarrier objects accumulate in `m_mbarriers` across kernel launches, leaking
memory and potentially causing stale-state bugs on barrier address reuse.

The plan should add a call site — likely in the CTA completion handler, e.g.:

```cpp
// in CTA deallocation path
reset_mbarriers_for_cta(cta_id);
```

#### 5. Phase C2 and Phase F redundantly claim bypass removal

Phase C2 says: *"Remove the string-prefix bypass. SYNCS is now MBARRIER_OP and
never reaches the BARRIER_OP branch."*

Phase F says: *"Remove the `bypass_syncs_barrier` string-prefix hack from
sm.cc."*

These describe the same action. If C2 already removes it, F is a no-op
verification step. If C2 does not remove it (because the bypass is in the
`BARRIER_OP` branch which SYNCS no longer reaches), then F is the actual
removal but C2's wording is misleading.

Clarification: Phase C2 **replaces** the bypass with real `MBARRIER_OP`
handling (making the bypass dead code). Phase F **deletes** the dead bypass
code and validates. The plan should state this distinction explicitly.

#### 6. `std::map` for mbarrier storage may be a performance concern

The plan uses `std::map<std::pair<unsigned, uint64_t>, HopperMBarrierObject>`
keyed by `(cta_id, barrier_addr)`. For a simulator processing billions of
instructions, `std::map`'s O(log n) per lookup with heap allocation per node is
a potential bottleneck. An `std::unordered_map` or a flat vector indexed by CTA
slot would be more appropriate. This is not a correctness issue, but it is
worth noting since the plan explicitly calls out "no behavioral changes" for
Phase A — performance regressions from container choice in the hot path would
violate that spirit.

#### 7. Verification gates are manual, not automated

Every verification gate says "add debug prints" and "confirm" — these are
manual checks. For a plan this detailed, each gate should specify an automated
assertion or test script command. For example, Phase E's gate could be: *"Run
`./run_fa3_kernel10.sh --check-mbarrier` and assert exit code 0."* Without
automation, the gates are easy to skip or misinterpret.

#### 8. `SYNCS_PHASECHK` handler is a no-op but may be needed sooner

Phase C2's PHASECHK handler does nothing (`// evaluates readiness; no state
change needed at Minimum tier`). But PHASECHK is the predicate check that
precedes TRYWAIT in many SASS sequences. If the simulator's trace replay
expects PHASECHK to have some observable effect (e.g., setting a predicate
register), the no-op stub could cause downstream issues. The plan should at
least note this risk.

#### 9. Blackwell opcode routing guesses are unvalidated commitments

Phase A4 assigns specific `op_type` values to Blackwell opcodes (e.g.,
`WARPGROUP → BRANCH_OP`, `ACQBULK → MISCELLANEOUS_NO_QUEUE_OP`). These are
reasonable guesses but are completely unvalidated against Blackwell SASS
behavior. The plan acknowledges "Blackwell semantic divergence is Tier 4" but
the routing choices themselves are semantic commitments. A safer approach: route
all unknown Blackwell opcodes to `MISCELLANEOUS_NO_QUEUE_OP` as a universal
stub, and refine routing in Tier 4 when traces are available.

#### 10. Missing `NONE`/default case in the switch

Phase C2's switch statement handles `SYNCS_EXCH`, `SYNCS_ARRIVE`,
`SYNCS_ARRIVE_RED`, `SYNCS_PHASECHK`, `SYNCS_TRYWAIT` — but not `NONE`. If
`has_hopper_sync_info()` returns true but `kind == NONE` (e.g., due to a decode
bug), the switch falls through silently. A default case would catch this:

```cpp
default:
    assert(!"unhandled HopperSyncOpKind");
    break;
```

---

## Summary

| # | Severity | Issue |
|---|----------|-------|
| 1 | **Critical** | ARRIVE→TMA binding TOCTOU: premature `ready=true` can let consumer proceed before data arrives |
| 2 | Moderate | Bypass in `shader.cc` not addressed |
| 3 | Moderate | `find_mbarrier_mutable` undeclared |
| 4 | Moderate | `reset_mbarriers_for_cta` has no call site |
| 5 | Moderate | C2/F bypass removal redundancy is confusing |
| 6 | Minor | `std::map` performance in hot path |
| 7 | Minor | Verification gates are manual-only |
| 8 | Minor | PHASECHK no-op may be insufficient |
| 9 | Nit | Blackwell routing guesses are unvalidated commitments |
| 10 | Nit | Missing `NONE`/default case in switch |

The plan's core architecture — the dependency-ordered phases, the FA3-first
rule, the mbarrier object model — is sound. Fixing issues #1–#5 would make it
ready for execution.
