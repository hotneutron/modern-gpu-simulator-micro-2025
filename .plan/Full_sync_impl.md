# Purpose

This document defines four implementation tiers for Hopper / Blackwell synchronization support:

- **minimum**
- **good**
- **full**
- **blackwell-support**

It is intended to answer two questions clearly:

1. What is the difference between a workload-unblocking sync model and an architecturally credible one?
2. Where does the current `H100_sync_impl.md` effort actually sit on that ladder?

This document incorporates the current H100 FA3-first design and the architectural coverage review in `.plan/20260520-1232-reflection-sync-coverage.md`.

# Current Status

## Tier definitions (authoritative)

- **minimum**: run FA3 (current goal).
- **good**: reasonable Hopper support beyond FA3 (multiple workloads, fewer heuristics).
- **full**: all Hopper sync scenarios (architecturally complete Hopper sync model).
- **blackwell-support**: Blackwell coverage on top of full Hopper.

## Current design status

- The current plan in [H100_sync_impl.md](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/.plan/H100_sync_impl.md) is a **minimum-tier plan**.
- Its intent is to reach a **minimum passable Hopper sync / mbarrier implementation** that runs FA3 end-to-end.

## Current implementation status

The earlier minimum blockers have largely been resolved. Concretely:

- TMA operand sidecar lookup is stable on the `(unique_function_id, pc, handle_hi)` resolver key.
- Kernel-10 TMA metadata resolves reliably; descriptor / operand resolvers are merged at command build time.
- The Hopper `SYNCS` / TMA handshake reaches the steady-state path: mbarrier objects are created, TMA completion binds to a barrier, completion updates that barrier, and `TRYWAIT` / `PHASECHK` observe its state.
- A new state-modeling problem was identified along the way: FA3 rotates through multiple mbarrier slots (e.g. `0x30c00`, `0x30c10`, `0x30c20`, ...) within one CTA / warp flow. The earlier "single active barrier per `(cta, warp)`, last-`SYNCS.EXCH` wins" model conflated those slots and could deadlock on slots like `0x31020`.
- That problem is addressed by the current implementation through a **deferred-contract mbarrier model** (see Tier 1 below): `SYNCS.EXCH` does not commit a contract, the first downstream observed event (TMA bind or wait poll) decides whether the barrier expects transaction bytes or an arrive count, and a late TMA bind can promote an arrive-count contract back to a transaction-bytes one. This replaces the earlier "active-slot wins" `expected_tx_bytes` binding rule.

What is still in progress:

- **minimum-achieved validation**: confirm FA3 backward runs end-to-end on the target kernel under the deferred-contract mbarrier model (currently being measured on a long simulator run).
- **bypass removal**: the temporary `SYNCS` bypass is removed only after the deferred-contract model passes the minimum-achieved gate on the FA3 target.

In other words, `H100_sync_impl.md` describes a **minimum-tier target**, and the current implementation has cleared the bring-up gate and is now in the **minimum-achieved validation** phase.

# Architectural Boundaries That Must Hold At Every Tier

The following are cross-tier invariants and should not be violated:

- Hopper `SYNCS` must not be treated as legacy CTA `BAR.SYNC`.
- TMA must not route through `ldst_unit_sm`.
- The issuing warp must release after command enqueue, not after transfer completion.
- TMA completion must remain separate from `DEPBAR` / `LDGDEPBAR`.
- Address-based barrier objects must live outside `barrier_set_t`.
- Transfer progress should be tied to `requests_total`-style accounting rather than a naive `total_bytes` timing model.

# How Hopper mbarrier handshake actually works

This section gives the visual mental model of the protocol, so the rest of the document is easier to read. The diagrams use ASCII so they survive in plain markdown.

## Actors and barrier-object fields

There are four actors, and they should always be read as separate lanes in the diagrams. The producer warp arms the barrier slot and (optionally) issues a TMA. The TMA engine is a separate hardware actor: it accepts TMA commands, runs them in the background, and reports completion bytes back to the slot it was bound to. The barrier object is the shared-memory state. The consumer warp polls the barrier slot.

The barrier object exposes two completion contracts: a transaction-bytes budget (`expected_tx_bytes`) and an arrive count (`expected_arrive_count`). When the active contract is satisfied, the barrier flips to a new phase and waiters are released. The fields in the diagrams have the following meanings:

- `contract`: which contract is currently active for this round — `UNSET`, `TX_BYTES`, or `ARRIVE_COUNT`.
- `pending_value`: a scratch accumulator; every `SYNCS.EXCH` adds its `URsrc` (NVBit-captured arrive contribution) here while the contract is still `UNSET`.
- `expected_tx_bytes`: the total transaction-bytes budget the bound TMA must deliver before the slot is ready (set when contract becomes `TX_BYTES`).
- `completed_tx_bytes`: how many transaction bytes the TMA engine has already delivered to this slot (incremented by TMA completion events).
- `expected_arrive_count`: how many `SYNCS.EXCH` arrivals the slot is waiting for (set when contract becomes `ARRIVE_COUNT`).
- `arrive_count`: how many arrivals have actually been observed (incremented on each `SYNCS.EXCH`).
- `phase` / `ready`: the slot becomes `ready=true` and flips its phase when the active contract is satisfied (`completed_tx_bytes >= expected_tx_bytes` or `arrive_count >= expected_arrive_count`).

## The big picture: producer / TMA / consumer around one barrier slot

```
                    ┌─────────────────────────────────────────────┐
                    │  mbarrier object @addr                      │
                    │   keyed by (cta_id, barrier_addr)           │
                    │                                             │
                    │   contract                                  │
                    │   pending_value                             │
                    │   expected_tx_bytes / completed_tx_bytes    │
                    │   expected_arrive_count / arrive_count      │
                    │   phase / ready                             │
                    └─────▲────────────▲────────────────▲─────────┘
                          │            │                │
              SYNCS.EXCH @addr         │                │ PHASECHK / TRYWAIT
              (arms slot, adds         │                │ (consumer reads
               URsrc to pending,       │                │  ready / phase)
               arrive_count++)         │                │
                          │            │                │
                  ┌───────┴──────┐     │     ┌──────────┴────────┐
                  │ producer warp│     │     │ consumer warp(s)  │
                  └───────┬──────┘     │     └───────────────────┘
                          │            │
                  issue TMA cmd        │ TMA completion event
                  (bind to most        │ (delivers bytes to the slot
                   recent unbound      │  the TMA was bound to:
                   slot of this warp)  │  completed_tx_bytes += N)
                          │            │
                  ┌───────▼────────────┴──────┐
                  │ TMA engine (UTMALDG/...)  │
                  │  in-flight transfer       │
                  └───────────────────────────┘
```

Two things to keep in mind while reading the rest:

- The barrier object is keyed by `(cta_id, barrier_addr)`. FA3 uses many slots (`0x30c00`, `0x30c10`, ...) at once, so a single "active barrier" per warp is not enough.
- The producer-side instruction (`SYNCS.EXCH`) and the bytes-delivering side (the TMA engine) are decoupled in time. `SYNCS.EXCH` arms the slot now, the TMA engine may finish delivering bytes much later.

## A typical FA3 round (TMA-driven contract)

This is the common path on the FA3 target: a producer warp arms a barrier and kicks off a TMA, the TMA engine delivers bytes in the background, and consumer warps wait until enough bytes have landed. Read the diagram left-to-right; each lane is one actor.

```
time ──────────────────────────────────────────────────────────────────────────►

 producer  │ SYNCS.EXCH(v) ──► issue TMA cmd ─────────────────────────────────────
  warp     │     │                  │
           │     │                  │ (bind: associate this TMA's bytes
           │     │                  │  with producer's most recent unbound slot)
           │     ▼                  │
 barrier   │ contract=UNSET    contract=TX_BYTES                          ready=true
 @addr     │ pending=v         expected_tx_bytes = N        completed_tx  phase flips
           │ arrive_count++    pending=0                       ↑↑↑       contract=UNSET
           │                                                  ▲
           │                                                  │ TMA reports
           │                                                  │ N bytes delivered
           │                       ┌────────────────────────┐ │
 TMA       │                       │ in-flight transfer ... │─┘
 engine    │                       └────────────────────────┘
           │                                                                │
 consumer  │ ............ PHASECHK / TRYWAIT @addr (spin) ..................▼ passes
  warp     │                                          (because completed_tx ≥ expected_tx)
```

`completed_tx_bytes` is the slot's running total of bytes the TMA engine has actually delivered. The slot becomes ready when `completed_tx_bytes >= expected_tx_bytes`.

## A non-TMA round (arrive-count contract)

If a barrier has no bound TMA for this round, pure thread arrivals must still satisfy the waiter. Two things are worth spelling out before reading the diagram, because they are the parts that the model treats differently from a textbook mbarrier.

**How is "non-TMA round" decided?** It is *not* decided ahead of time. The deferred-contract model never commits to "this round will be a TMA round" or "this round will be an arrive-count round" up front. Instead, the contract is committed by whichever observable event happens first on the slot:

- If the TMA engine binds to the slot first → the round is a `TX_BYTES` round.
- If a consumer wait poll (`PHASECHK` / `TRYWAIT`) hits the slot first while it is still `UNSET` → the round is treated as an `ARRIVE_COUNT` round, because no producer has attached a transaction-bytes budget yet.
- If the TMA engine binds *after* the round was already classified as `ARRIVE_COUNT`, the safety promotion in the next subsection re-classifies it to `TX_BYTES`.

So "non-TMA round" really means "no TMA bound by the time the consumer starts polling". The classification is a property of the timeline, not of the program text.

**Where does `expected_arrive_count` come from?** It is not a constant pulled out of thin air. The PTX-level `mbarrier.arrive(addr, count)` lowers to `SYNCS.EXCH @addr, URsrc`, where `URsrc` is the run-time arrive-count contribution of that single arrival. The NVBit `SYNCS` operand capture path records `URsrc` per `(unique_function_id, pc)` and feeds it back into the simulator's executed `SYNCS` command. On every `SYNCS.EXCH`, the simulator does `pending_value += URsrc` and `arrive_count++`. When the wait side commits the contract, the accumulated `pending_value` becomes `expected_arrive_count` and `pending_value` is cleared.

```
time ──────────────────────────────────────────────────────────────────────────►

 producer  │ SYNCS.EXCH(v1) ──► SYNCS.EXCH(v2) ────────────────────────────────────
  warp     │     │                  │       (vN = NVBit-captured URsrc;
           │     │                  │        each EXCH contributes one arrival)
           │     ▼                  ▼
 barrier   │ contract=UNSET    pending += v1+v2                              ready=true
 @addr     │ pending=0         arrive_count++                                phase flips
           │ (round type                                                     contract=UNSET
           │  undecided —                       commit on first poll:        pending=0
           │  no TMA yet)                       (still no TMA bound,
           │                                     so arrive-count round)
           │                                    contract=ARRIVE_COUNT
           │                                    expected_arrive_count = pending
           │                                    pending=0
           │                                            ▲                       │
 TMA       │                       (no TMA for this round)                      │
 engine    │                                                                    │
           │                                            │                       │
 consumer  │ .............. PHASECHK / TRYWAIT @addr (spin) ....................▼ passes
  warp     │                                          (because arrive_count ≥ expected_arrive_count)
```

## Bind-after-arrive: the safety promotion

The dangerous case is when consumers start polling first (which would commit `ARRIVE_COUNT`) but the TMA engine actually does bind to the slot later. Without a promotion, that TMA's bytes would never be required, and the waiter could pass before the data is in shared memory. The deferred-contract model closes this race by **promoting** the contract back to `TX_BYTES` when the late TMA binds.

```
time ──────────────────────────────────────────────────────────────────────────►

 producer  │ SYNCS.EXCH(v) ─────────────────► (late) issue TMA cmd ────────────────
  warp     │     │                                  │
           │     │                                  │ (late bind to this slot)
           │     ▼                                  │
 barrier   │ contract=UNSET   contract=ARRIVE_COUNT │  PROMOTE:                   ready=true
 @addr     │ pending=v        expected_arrive=v     │  contract=TX_BYTES          phase flips
           │                       ▲                │  expected_tx += folded      contract=UNSET
           │                       │ commit         │  expected_arrive_count=0
           │                       │ on first poll  │
           │                                        │                completed_tx
           │                                        │                  ↑↑↑
           │                                        │                  ▲
           │                                        │                  │ TMA reports
           │                                        │                  │ bytes delivered
           │                       ┌────────────────▼────────────────┐ │
 TMA       │                       │ in-flight transfer ...          │─┘
 engine    │                       └─────────────────────────────────┘
           │                       │                                                │
 consumer  │ .......... PHASECHK / TRYWAIT @addr (still spinning) ..................▼ passes
  warp     │                                          (only after completed_tx ≥ expected_tx)
```

## Where the current simulator implementation lives

The diagrams above map directly onto the implementation:

- `SYNCS.EXCH` handler in [sm.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc) only stashes `pending_value`, increments `arrive_count`, and leaves `contract = UNSET`.
- TMA bind in [tma_unit_sm.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc) commits or promotes to `TX_BYTES`, folding `pending_value` into `expected_tx_bytes`.
- TMA completion in the same file delivers bytes back to the bound slot by incrementing `completed_tx_bytes`.
- Wait polling in [subcore.cc](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc) commits `ARRIVE_COUNT` on the first poll of an `UNSET` barrier.
- `recompute_mbarrier_ready_and_maybe_flip_phase` (also in `sm.cc`) checks `tma_ready || arrive_ready`, flips the phase, and resets `contract` / `pending_value` for the next round.

# Tier 1: Minimum

## Goal

Allow FA3-style Hopper kernels to achieve forward progress without relying on a temporary `SYNCS` bypass.

## What minimum must include

- A dedicated `MBARRIER_OP` path for Hopper `SYNCS`.
- Address-based barrier objects keyed by barrier address rather than CTA barrier id.
- Decoded support for the main Hopper `SYNCS` forms used by FA3:
  - `SYNCS.EXCH`
  - `SYNCS.ARRIVE`
  - `SYNCS.ARRIVE.RED`
  - `SYNCS.PHASECHK`
  - `SYNCS.TRYWAIT`
- TMA completion wired into barrier-object readiness.
- Consumer wait instructions observing barrier-object state rather than legacy CTA arrival counts.
- Enough TMA / sidecar metadata resolution to support the executed FA3 command stream.
- An **NVBit-side `SYNCS` operand capture path** that records the dynamic value operand (`URsrc`) of `SYNCS.EXCH` / `SYNCS.ARRIVE*` per `(unique_function_id, pc)`, aggregates it into a `syncs_operand_resolver.json` sidecar, and is loaded by the simulator and merged into the executed `SYNCS` command. Without this, the contract decision in the deferred-contract mbarrier model has no input.
- A **per-slot mbarrier object model** that survives multiple live barrier addresses in the same `(cta, warp)`. FA3 rotates through several barrier addresses (e.g. `0x30c00`, `0x30c10`, `0x30c20`, ...), so a single mutable "active barrier" per `(cta, warp)` is not sufficient. Barrier objects must be addressable per `(cta_id, barrier_addr)`, and each `SYNCS.EXCH` / `SYNCS.ARRIVE*` / `TRYWAIT` / `PHASECHK` / TMA bind must operate on its own slot.
- An explicit binding rule for `expected_tx_bytes` and `expected_arrive_count` via the **deferred-contract mbarrier model** (deferred contract decision + late-bind contract promotion):
  - `SYNCS.EXCH` does **not** decide the contract. It only records that the barrier is initialized, increments `arrive_count`, and accumulates the value operand (`URsrc`) into `pending_value`. Contract stays `UNSET`.
  - The first downstream observed event commits the contract:
    - On TMA bind to this barrier slot, contract becomes `TX_BYTES`. Any previously stashed `pending_value` is committed into `expected_tx_bytes`.
    - On the first wait-side polling (`PHASECHK` / `TRYWAIT`) of a still-`UNSET` barrier, contract becomes `ARRIVE_COUNT`. `pending_value` is moved into `expected_arrive_count`.
  - If a barrier's contract was provisionally `ARRIVE_COUNT` but a later TMA binds to it, the contract is **promoted to `TX_BYTES`**: the previously committed `expected_arrive_count` is re-interpreted as a transaction-bytes contribution and folded into `expected_tx_bytes`. This is the bind-after-arrive safety promotion.
  - On phase flip (barrier becomes ready), contract resets to `UNSET` and `pending_value` is cleared so the next phase can be decided freshly.
- A defined non-TMA arrival fallback. If a barrier has no bound TMA transfer for the current round, pure warp-arrive progress must still be able to satisfy the waiter. Under the deferred-contract model, this is realized by the `ARRIVE_COUNT` contract path: `pending_value` accumulated by `SYNCS.EXCH` becomes `expected_arrive_count` at the first wait poll. The bind-after-arrive race is closed by the `ARRIVE_COUNT` → `TX_BYTES` promotion described above, so a barrier provisionally marked ready as an arrive-count contract cannot let consumers pass before a real later TMA transfer completes.

## What minimum may still simplify

- Barrier-object association may still use a guarded heuristic (e.g. binding TMA completion to the issuing warp's most recent unbound barrier slot).
- Wait satisfaction may still rely on a simplified ready condition (`tma_ready || arrive_ready`) instead of full phase-token semantics.
- `phase` may exist without full parity / phase-token semantics (a single phase flip on ready is sufficient for FA3).
- Partial completion and full per-request protocol may still be simplified.
- The deferred-contract model is allowed to commit a contract eagerly on the first observed event per phase rather than tracking both contracts simultaneously.

## Explicit approximations allowed at minimum

The sync coverage reflection note identifies several areas that can remain approximated at minimum, but they should be documented explicitly rather than silently ignored:

- `WARPSYNC`
- `BSYNC`
- `WARPGROUP`
- `WARPGROUPSET`

These can remain branch / misc approximations at minimum as long as the plan states that they are deferred rather than covered.

The current FA3 backward target trace does not show `UCGABAR_*`, so cluster-barrier routing is not a minimum blocker for the present FA3-first bring-up target. It still remains a Hopper-general safety hazard once the goal expands beyond this trace.

## What minimum does not require yet

- Full phase / parity protocol.
- Full cluster barrier semantics.
- Full proxy-fence model.
- Full Blackwell sync coverage.
- Full Gen5 tensor-core sync coverage.

## Minimum validation gates

Minimum should be checked in two stages rather than one vague "forward progress" label:

- **Bring-up gate**:
  - The first target FA3 backward kernel executes past its first blocking `SYNCS.TRYWAIT`.
  - Debug visibility shows at least one end-to-end path: mbarrier object created, TMA bound to that barrier, TMA completion updates that barrier, `TRYWAIT` is released.
- **Minimum achieved**:
  - The target FA3 backward kernel completes end-to-end with the temporary `SYNCS` bypass removed.
  - Hopper `SYNCS` no longer enter legacy `barrier_set_t`.
  - Non-Hopper regressions remain unchanged.

These gates should be written so they can be checked automatically where practical, not only by manual log reading. Debug prints are useful during bring-up, but the intended end state is a scriptable pass / fail check for the FA3 target kernel.

# Tier 2: Good

## Goal

Move from workload-driven forward progress to a semantically credible Hopper sync model that is stable across multiple kernels and not narrowly tuned to FA3.

## A good implementation should add

- Phase-aware barrier-object semantics.
- Less heuristic TMA-to-barrier association.
- Cleaner separation between object initialization, producer arrival, transfer progress, and consumer wait satisfaction.
- Better-defined `EXCH` semantics instead of using it as a broad reset hammer.
- Better-defined `ARRIVE` / `ARRIVE.RED` semantics.
- Completion accounting driven by transfer progress in a way that scales beyond the FA3 geometry accident.

## Limitations of the deferred-contract mbarrier model

The deferred-contract mbarrier model is sufficient for FA3 at the minimum tier, but it is intentionally lossy and good-tier work must lift its limitations:

- It commits a single contract per phase based on the first observed event. It cannot represent a barrier that legitimately combines a transaction-bytes budget **and** an arrive count in the same phase (e.g. CUTLASS-style split-arrive patterns where threads call `mbarrier.arrive` while a TMA also contributes `expect_tx`).
- The `ARRIVE_COUNT` → `TX_BYTES` promotion is a safety net, not an architectural model. It folds an already-committed arrive count into transaction bytes when a TMA arrives late, which preserves forward progress on FA3 but is not a faithful mbarrier semantics.
- The model resets contract / `pending_value` on phase flip, so it does not retain cross-phase parity. Workloads that rely on observing the previous phase token before the next round are not modeled.

Good-tier work should track `expected_tx_bytes` and `expected_arrive_count` as **independent** state on the same barrier object, so both contracts can coexist within one phase. It should also replace "first observed event commits the contract" with explicit decode-driven attribution, where each `SYNCS.EXCH` / `SYNCS.ARRIVE*` is classified by its own semantics (e.g. via per-PC operand context, value operand range, or an enriched tracer signal that disambiguates `mbarrier.arrive` vs `mbarrier.arrive.expect_tx`). The late-bind promotion should remain only as a fallback, not as the primary mechanism.

## A good implementation should also clean up coverage hazards

The reflection note makes several important points that belong here:

- `UCGABAR_ARV` and `UCGABAR_WAIT` must not continue entering `barrier_set_t` through `BARRIER_OP`. They need their own stub or dedicated routing before the Hopper sync plan can be considered architecturally safe.
- `ACQBULK` must be verified against FA3 SASS so we know whether a no-op stub is harmless or whether it silently breaks consumer ordering.
- `PHASECHK` must be verified as more than a harmless no-op if the traced SASS relies on predicate-producing behavior before `TRYWAIT`.

For clarity, `UCGABAR_*` remains a **Good-tier Hopper-general safety fix**, not a blocker for the current FA3 backward minimum target, because the present FA3 target trace does not execute those opcodes. If the scope changes from "FA3-first runnable" to "safe Hopper sync support beyond FA3", this item should be pulled earlier.

## Good means stable beyond one kernel

A good implementation should survive multiple live barrier addresses in the same CTA, multiple command geometries, kernels where `total_bytes` is not a safe proxy for progress, and kernels that do not happen to align with the current FA3-first assumptions.

# Tier 3: Full

## Goal

Implement Hopper synchronization as an architectural protocol, not just a workload approximation.

## Full Hopper support means

- Proper phase / parity-aware mbarrier behavior.
- Barrier-object semantics defined around address, phase / parity, expected transaction bytes, completed transaction bytes, arrival state, and wait token satisfaction.
- Correct behavior across barrier reuse in loops and repeated producer / consumer rounds.
- Correct treatment of multiple producers and consumers on the same barrier object.
- Full treatment of `SYNCS.ARRIVE.RED`.
- Correct coordination between TMA progress and barrier satisfaction.
- **Replacement of the deferred-contract model** with a decode-driven contract model: each `SYNCS.EXCH` / `SYNCS.ARRIVE*` is attributed directly to either an `expect_tx` contribution, an arrive-count contribution, or both, without relying on later observation to retroactively decide. This removes the deferred-contract + late-bind promotion path as a primary mechanism.

## Full Hopper also requires broader sync coverage

The sync coverage reflection note highlights items that are not optional once the goal moves beyond minimum / good bring-up:

- Proper non-CTA routing for `UCGABAR_ARV` / `UCGABAR_WAIT`.
- `FENCE` scope and proxy-kind decode, especially before any Phase-6-style proxy ordering work.
- Explicit handling of `ACQBULK` once its role in consumer ordering is confirmed.
- Explicit documentation of deferred warpgroup-level behavior until it is actually modeled.

## What full does not mean

Full does not mean "every possible future architecture is already covered." It means Hopper is implemented in a way that no longer depends on FA3-specific heuristics, silent sync opcode omissions, accidental geometry matches, or debug-only reasoning for core state transitions.

# Tier 4: Blackwell-Support

## Goal

Extend the Hopper sync model so Blackwell kernels can run without silent opcode gaps, wrong routing, or architectural category mistakes.

## Blackwell-support must include

At minimum, Blackwell opcode coverage needs to stop being incomplete in the critical sync / TMA-related areas identified by the reflection note:

- `SYNCS`
- `FENCE`
- `UCGABAR_ARV`
- `UCGABAR_WAIT`
- `ACQBULK`
- `WARPGROUP`
- `WARPGROUPSET`
- `ELECT`
- `ENDCOLLECTIVE`

## Infrastructure that may land before Tier 4

Some Blackwell work is cheaper than full semantic support and may be pulled earlier if it reduces obvious crashes:

- Adding missing sync-related opcode names to `blackwell_opcode.h`.
- Mapping them to Hopper-equivalent safe baseline routing.

This does not mean Blackwell synchronization itself is solved. It only means opcode recognition and safe initial routing can be decoupled from full Blackwell semantic work. When the correct Blackwell category is uncertain, conservative stub routing is safer than making a stronger semantic commitment too early.

## Why this is its own tier

The reflection note makes the key point clearly. Missing `SYNCS` in `blackwell_opcode.h` is a hard blocker for Blackwell TMA kernels, missing `FENCE` blocks proxy-fence / ordering work, and missing cluster and warpgroup-related opcodes leave large correctness holes. So "Blackwell-support" is not just "full Hopper plus a little extra." It remains a distinct semantic coverage milestone covering Blackwell-specific semantic validation, correct routing categories, and prevention of silent fallback into wrong legacy paths.

## Blackwell-support should start by preserving Hopper routing

The safest baseline is to copy Hopper sync-related opcode coverage into Blackwell with equivalent routing, then refine only where Blackwell is actually different. This is better than leaving opcodes missing and discovering gaps through crashes.

## Known Blackwell unknowns

The reflection note also calls out a deferred but important item: possible `TCGEN05.*` synchronization / wait / commit instructions for Gen5 tensor-core flows. These should remain on the known-unknowns list until real Blackwell traces are available.

# Tier Mapping Summary

## Minimum

- FA3-focused forward progress target.
- Current `H100_sync_impl.md` is aimed here.
- Includes the deferred-contract mbarrier model (deferred contract decision + late-bind promotion), per-slot mbarrier object model, and NVBit-side `SYNCS` operand capture pipeline.
- Current code has cleared the bring-up gate and is in **minimum-achieved validation**.

## Good

- Architecturally credible Hopper implementation for more than one workload.
- Fewer heuristics.
- Deferred-contract limitations lifted: independent tx-bytes / arrive-count tracking, decode-driven attribution.
- Explicit handling of current Hopper coverage hazards.

## Full

- Full Hopper sync semantics and coverage for the relevant synchronization domains.
- No longer FA3-driven in its core design.
- Deferred-contract model replaced entirely by a decode-driven contract model.

## Blackwell-support

- Blackwell opcode coverage and routing parity for the Hopper-derived sync / TMA architecture.
- Enough support to avoid immediate hard blockers on Blackwell TMA kernels.

# What To Fix Next Relative To These Tiers

## Dependency ordering note

The implementation dependency chain should be treated as opcode / types infrastructure → `warp_inst_t` fields and `SYNCS` decode → SM-side mbarrier storage and helpers → subcore-side wait gating → TMA-to-mbarrier binding and completion updates → bypass removal → validation.

Trace operand helpers are useful cleanup, but they are not a hard prerequisite if the minimum decode path is implemented directly in `trace_driven.cc`. Also, "bypass removal" should be read in two parts: first, the real `MBARRIER_OP` path replaces the old bypass in live execution; later, the dead legacy bypass code is deleted once the new path has passed validation.

## To reach minimum

Done:

- ✅ Operand sidecar lookup stable on `(unique_function_id, pc, handle_hi)`.
- ✅ Kernel-10 TMA metadata resolves reliably; descriptor / operand resolvers merged at command build time.
- ✅ Hopper `SYNCS` / TMA handshake reaches the steady-state executed path.
- ✅ NVBit-side `SYNCS` operand capture (`URsrc` per `(unique_function_id, pc)`), aggregator, and simulator loader landed; sidecar is merged into executed `SYNCS` commands.
- ✅ Per-slot mbarrier object model `(cta_id, barrier_addr)` (no longer a single mutable "active barrier" per `(cta, warp)`).
- ✅ Deferred-contract mbarrier model implemented end-to-end:
  - `SYNCS.EXCH` defers contract, accumulates `pending_value`, increments `arrive_count`.
  - TMA bind commits or promotes to `TX_BYTES` and folds `pending_value` into `expected_tx_bytes`.
  - First wait poll on a still-`UNSET` barrier commits `ARRIVE_COUNT`.
  - Phase flip resets contract / `pending_value`.
- ✅ Non-TMA arrival fallback (covered by `ARRIVE_COUNT` contract path).
- ✅ Bind-after-arrive race closed (by `ARRIVE_COUNT` → `TX_BYTES` promotion).

Still to do:

- 🔄 Minimum-achieved validation: confirm FA3 backward target kernel runs end-to-end under the deferred-contract mbarrier model.
- ⏳ Remove the temporary `SYNCS` bypass only after the FA3 target passes the minimum-achieved gate.
- ⏳ Scriptable pass / fail check for the FA3 target kernel (instead of relying on manual log inspection).

## To move from minimum to good

- Reduce barrier-association heuristics.
- Refine `EXCH`, `ARRIVE`, and wait semantics.
- Lift deferred-contract limitations: track `expected_tx_bytes` and `expected_arrive_count` independently within a phase rather than committing one contract; replace observed-event-driven contract commit with explicit decode-driven attribution per `SYNCS` form; keep the `ARRIVE_COUNT` → `TX_BYTES` promotion only as a fallback, not as the primary mechanism.
- Harden coverage for non-FA3 workloads (multiple geometries, multiple live barriers per CTA, mixed arrive + expect_tx patterns).
- Address `UCGABAR_*` routing hazard.
- Verify the role of `ACQBULK`.
- Verify whether `PHASECHK` needs explicit predicate-visible behavior beyond shared wait-state observation.
- Decide whether early Blackwell opcode-map parity should be landed as low-risk infrastructure before full Tier 4 work.

## To move from good to full

- Add proper phase / parity protocol.
- Add full `FENCE` decode for scope / proxy kinds.
- Broaden sync coverage beyond the FA3 subset.
- Replace the deferred-contract mbarrier model entirely with a decode-driven contract model that does not rely on later observation to retroactively decide barrier semantics.

## To move from full to blackwell-support

- Add the missing Blackwell sync opcodes and route them safely.
- Verify Blackwell-specific sync families once traces exist.
