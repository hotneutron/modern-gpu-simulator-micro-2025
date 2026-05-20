# Full Sync Implementation Tiers

## Purpose

This document defines four implementation tiers for Hopper / Blackwell synchronization
support:

- **minimum**
- **good**
- **full**
- **blackwell-support**

It is intended to answer two questions clearly:

1. What is the difference between a workload-unblocking sync model and an architecturally credible one?
2. Where does the current `H100_sync_impl.md` effort actually sit on that ladder?

This document incorporates the current H100 FA3-first design and the architectural
coverage review in `.plan/20260520-1232-reflection-sync-coverage.md`.

## Current Status

### Current design status

- The current plan in [H100_sync_impl.md](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/.plan/H100_sync_impl.md) is a **minimum-tier plan**.
- Its intent is to reach a **minimum passable Hopper sync / mbarrier implementation** for FA3-style kernels.

### Current implementation status

- The current codebase is **not yet at the minimum tier**.
- It is still **below minimum passable** because the implementation does not yet produce stable forward progress on the target FA3 kernel.
- The current blockers are still in bring-up/debug territory, including:
  - TMA operand sidecar lookup mismatches
  - incomplete or heuristic TMA-to-barrier association
  - incomplete phase semantics
  - unresolved early-path behavior before the intended steady-state `SYNCS` / TMA handshake is reached

In other words:

- `H100_sync_impl.md` describes a **minimum-tier target**
- the current implementation is still **working toward that target**

## Architectural Boundaries That Must Hold At Every Tier

The following are cross-tier invariants and should not be violated:

- Hopper `SYNCS` must not be treated as legacy CTA `BAR.SYNC`
- TMA must not route through `ldst_unit_sm`
- the issuing warp must release after command enqueue, not after transfer completion
- TMA completion must remain separate from `DEPBAR` / `LDGDEPBAR`
- address-based barrier objects must live outside `barrier_set_t`
- transfer progress should be tied to `requests_total`-style accounting rather than a naive `total_bytes` timing model

## Tier 1: Minimum

### Goal

Allow FA3-style Hopper kernels to achieve forward progress without relying on a temporary
`SYNCS` bypass.

### What minimum must include

- A dedicated `MBARRIER_OP` path for Hopper `SYNCS`
- Address-based barrier objects keyed by barrier address rather than CTA barrier id
- Decoded support for the main Hopper `SYNCS` forms used by FA3:
  - `SYNCS.EXCH`
  - `SYNCS.ARRIVE`
  - `SYNCS.ARRIVE.RED`
  - `SYNCS.PHASECHK`
  - `SYNCS.TRYWAIT`
- TMA completion wired into barrier-object readiness
- Consumer wait instructions observing barrier-object state rather than legacy CTA arrival counts
- Enough TMA/sidecar metadata resolution to support the executed FA3 command stream
- An explicit binding rule for `expected_tx_bytes`:
  - at TMA command build time, bind the issuing warp's active mbarrier address
  - write `cmd.total_bytes` into the bound barrier object's `expected_tx_bytes`
  - store the bound barrier address on the TMA command so completion updates the same object
- A defined non-TMA arrival fallback:
  - if a barrier has no bound TMA transfer for the current round, pure warp-arrive progress must still be able to satisfy the waiter
  - minimum may simplify the exact rule, but it must not leave non-TMA `ARRIVE` / `TRYWAIT` pairs permanently unready

### What minimum may still simplify

- Barrier-object association may still use a guarded heuristic
- Wait satisfaction may still rely on a simplified ready condition
- `phase` may exist without full parity/phase-token semantics
- `ARRIVE` value operands may still be ignored if their exact architectural meaning is unresolved
- Partial completion and full per-request protocol may still be simplified

### Explicit approximations allowed at minimum

The sync coverage reflection note identifies several areas that can remain approximated
at minimum, but they should be documented explicitly rather than silently ignored:

- `WARPSYNC`
- `BSYNC`
- `WARPGROUP`
- `WARPGROUPSET`

These can remain branch/misc approximations at minimum as long as the plan states that
they are deferred rather than covered.

The current FA3 backward target trace does not show `UCGABAR_*`, so cluster-barrier
routing is not a minimum blocker for the present FA3-first bring-up target. It still
remains a Hopper-general safety hazard once the goal expands beyond this trace.

### What minimum does not require yet

- Full phase/parity protocol
- Full cluster barrier semantics
- Full proxy-fence model
- Full Blackwell sync coverage
- Full Gen5 tensor-core sync coverage

### Minimum validation gates

Minimum should be checked in two stages rather than one vague "forward progress" label:

- **Bring-up gate**
  - the first target FA3 backward kernel executes past its first blocking `SYNCS.TRYWAIT`
  - debug visibility shows at least one end-to-end path:
    - mbarrier object created
    - TMA bound to that barrier
    - TMA completion updates that barrier
    - `TRYWAIT` is released
- **Minimum achieved**
  - the target FA3 backward kernel completes end-to-end with the temporary `SYNCS` bypass removed
  - Hopper `SYNCS` no longer enter legacy `barrier_set_t`
  - non-Hopper regressions remain unchanged

## Tier 2: Good

### Goal

Move from workload-driven forward progress to a semantically credible Hopper sync model
that is stable across multiple kernels and not narrowly tuned to FA3.

### A good implementation should add

- Phase-aware barrier-object semantics
- Less heuristic TMA-to-barrier association
- Cleaner separation between:
  - object initialization
  - producer arrival
  - transfer progress
  - consumer wait satisfaction
- Better-defined `EXCH` semantics instead of using it as a broad reset hammer
- Better-defined `ARRIVE` / `ARRIVE.RED` semantics
- Completion accounting driven by transfer progress in a way that scales beyond the FA3 geometry accident

### A good implementation should also clean up coverage hazards

The reflection note makes several important points that belong here:

- `UCGABAR_ARV` and `UCGABAR_WAIT` must not continue entering `barrier_set_t` through `BARRIER_OP`
- they need their own stub or dedicated routing before the Hopper sync plan can be considered architecturally safe
- `ACQBULK` must be verified against FA3 SASS so we know whether a no-op stub is harmless or whether it silently breaks consumer ordering

For clarity:

- `UCGABAR_*` remains a **Good-tier Hopper-general safety fix**, not a blocker for the current FA3 backward minimum target, because the present FA3 target trace does not execute those opcodes
- if the scope changes from "FA3-first runnable" to "safe Hopper sync support beyond FA3", this item should be pulled earlier

### Good means stable beyond one kernel

A good implementation should survive:

- multiple live barrier addresses in the same CTA
- multiple command geometries
- kernels where `total_bytes` is not a safe proxy for progress
- kernels that do not happen to align with the current FA3-first assumptions

## Tier 3: Full

### Goal

Implement Hopper synchronization as an architectural protocol, not just a workload
approximation.

### Full Hopper support means

- Proper phase/parity-aware mbarrier behavior
- Barrier-object semantics defined around:
  - address
  - phase/parity
  - expected transaction bytes
  - completed transaction bytes
  - arrival state
  - wait token satisfaction
- Correct behavior across barrier reuse in loops and repeated producer/consumer rounds
- Correct treatment of multiple producers and consumers on the same barrier object
- Full treatment of `SYNCS.ARRIVE.RED`
- Correct coordination between TMA progress and barrier satisfaction

### Full Hopper also requires broader sync coverage

The sync coverage reflection note highlights items that are not optional once the goal
moves beyond minimum/good bring-up:

- proper non-CTA routing for `UCGABAR_ARV` / `UCGABAR_WAIT`
- `FENCE` scope and proxy-kind decode, especially before any Phase-6-style proxy ordering work
- explicit handling of `ACQBULK` once its role in consumer ordering is confirmed
- explicit documentation of deferred warpgroup-level behavior until it is actually modeled

### What full does not mean

It does not mean “every possible future architecture is already covered.”

It means Hopper is implemented in a way that no longer depends on:

- FA3-specific heuristics
- silent sync opcode omissions
- accidental geometry matches
- debug-only reasoning for core state transitions

## Tier 4: Blackwell-Support

### Goal

Extend the Hopper sync model so Blackwell kernels can run without silent opcode gaps,
wrong routing, or architectural category mistakes.

### Blackwell-support must include

At minimum, Blackwell opcode coverage needs to stop being incomplete in the critical
sync/TMA-related areas identified by the reflection note:

- `SYNCS`
- `FENCE`
- `UCGABAR_ARV`
- `UCGABAR_WAIT`
- `ACQBULK`
- `WARPGROUP`
- `WARPGROUPSET`
- `ELECT`
- `ENDCOLLECTIVE`

### Infrastructure that may land before Tier 4

Some Blackwell work is cheaper than full semantic support and may be pulled earlier if
it reduces obvious crashes:

- adding missing sync-related opcode names to `blackwell_opcode.h`
- mapping them to Hopper-equivalent safe baseline routing

This does not mean Blackwell synchronization itself is solved. It only means opcode
recognition and safe initial routing can be decoupled from full Blackwell semantic work.

### Why this is its own tier

The reflection note makes the key point clearly:

- missing `SYNCS` in `blackwell_opcode.h` is a hard blocker for Blackwell TMA kernels
- missing `FENCE` blocks proxy-fence / ordering work
- missing cluster and warpgroup-related opcodes leave large correctness holes

So “Blackwell-support” is not just “full Hopper plus a little extra.”
It remains a distinct semantic coverage milestone:

- Blackwell-specific semantic validation
- correct routing categories
- prevention of silent fallback into wrong legacy paths

### Blackwell-support should start by preserving Hopper routing

The safest baseline is:

- copy Hopper sync-related opcode coverage into Blackwell with equivalent routing
- then refine only where Blackwell is actually different

This is better than leaving opcodes missing and discovering gaps through crashes.

### Known Blackwell unknowns

The reflection note also calls out a deferred but important item:

- possible `TCGEN05.*` synchronization / wait / commit instructions for Gen5 tensor-core flows

These should remain on the known-unknowns list until real Blackwell traces are available.

## Tier Mapping Summary

### Minimum

- FA3-focused forward progress target
- current `H100_sync_impl.md` is aimed here
- current code has **not yet reached this tier**

### Good

- architecturally credible Hopper implementation for more than one workload
- fewer heuristics
- explicit handling of current Hopper coverage hazards

### Full

- full Hopper sync semantics and coverage for the relevant synchronization domains
- no longer FA3-driven in its core design

### Blackwell-support

- Blackwell opcode coverage and routing parity for the Hopper-derived sync/TMA architecture
- enough support to avoid immediate hard blockers on Blackwell TMA kernels

## What To Fix Next Relative To These Tiers

### Dependency ordering note

The implementation dependency chain should be treated as:

- opcode/types infrastructure
- `warp_inst_t` fields and `SYNCS` decode
- SM-side mbarrier storage and helpers
- subcore-side wait gating
- TMA-to-mbarrier binding and completion updates
- bypass removal
- validation

Trace operand helpers are useful cleanup, but they are not a hard prerequisite if the
minimum decode path is implemented directly in `trace_driven.cc`.

### To reach minimum

- fix the current operand sidecar lookup mismatch / runtime key mismatch
- get the intended kernel-10 TMA path to resolve metadata reliably
- make the Hopper `SYNCS` / TMA handshake reach a stable executed path
- specify and implement the `expected_tx_bytes` binding path from TMA issue to mbarrier state
- specify and implement the non-TMA arrival fallback so non-transfer waits cannot deadlock by construction
- remove the temporary `SYNCS` bypass only after the new path passes the bring-up gate

### To move from minimum to good

- reduce barrier-association heuristics
- refine `EXCH`, `ARRIVE`, and wait semantics
- address `UCGABAR_*` routing hazard
- verify the role of `ACQBULK`
- decide whether early Blackwell opcode-map parity should be landed as low-risk infrastructure before full Tier 4 work

### To move from good to full

- add proper phase/parity protocol
- add full `FENCE` decode for scope/proxy kinds
- broaden sync coverage beyond the FA3 subset

### To move from full to blackwell-support

- add the missing Blackwell sync opcodes and route them safely
- verify Blackwell-specific sync families once traces exist
