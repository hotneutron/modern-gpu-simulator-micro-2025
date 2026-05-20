# Sync Coverage Reflection — 2026-05-20

## Purpose

Cross-check the H100 sync / mbarrier implementation plan (`H100_sync_impl.md`) against
the full set of synchronization instructions present in the Hopper and Blackwell opcode
maps, across all barrier scopes: CTA, SM/cluster, grid, and chip. Also covers
warpgroup-level coordination and Blackwell-specific gaps.

---

## CTA-Level Sync

### Covered

- `BAR.*` (`BARRIER_OP`) — classic CTA rendezvous; plan leaves it unchanged. Correct.
- `SYNCS.*` — plan re-routes from `BARRIER_OP` to `MBARRIER_OP` and implements
  `EXCH`, `ARRIVE`, `ARRIVE_RED`, `PHASECHK`, `TRYWAIT`. This is the main body of the
  H100 sync plan and is architecturally correct.

### Not Covered: `WARPSYNC`, `BSYNC`

Both are mapped to `BRANCH_OP` with no barrier semantics modeled. The plan does not
mention them. For a timing simulator this is currently harmless — they participate in
warp reconvergence and branch-stack control, not in memory ordering or data-movement
completion. Treating them as branches is an acceptable approximation but should be
documented as an explicit simplification, not left as an accidental omission.

---

## SM / Cluster-Level Sync

### Routing hazard introduced by the plan

`UCGABAR_ARV` and `UCGABAR_WAIT` are Hopper cluster-scope gather/arrive barriers.
They currently map to `BARRIER_OP`, the same type as `BAR`. The plan lists them as
non-goals and leaves their routing unchanged.

**This becomes a problem once the plan lands.** After `SYNCS` moves to `MBARRIER_OP`,
`BAR` becomes the only opcode still using `BARRIER_OP`. But `UCGABAR_ARV` and
`UCGABAR_WAIT` remain as `BARRIER_OP` and will enter `barrier_set_t` — the legacy
CTA-rendezvous code path — with wrong participant counts and wrong semantics. Any
Hopper cluster kernel (including cluster-multicast TMA variants) that issues these will
hit incorrect barrier behavior or assertion failures.

**Required fix before the plan lands:** route `UCGABAR_ARV` and `UCGABAR_WAIT` to a
dedicated stub type — not `BARRIER_OP`, not `MBARRIER_OP`. A new `CLUSTER_BARRIER_OP`
or explicit no-op routing is sufficient. The point is that they must not enter
`barrier_set_t`. Full cluster semantics can remain a non-goal; correct non-entry into
the CTA path is not optional.

---

## Memory Ordering: `FENCE` and `MEMBAR`

Both opcodes collapse all forms to `MEMORY_BARRIER_OP` without scope decode.

### `MEMBAR` — acceptable approximation for now

| Form | Scope | Treatment |
|---|---|---|
| `MEMBAR.CTA` | CTA | `MEMORY_BARRIER_OP` |
| `MEMBAR.SM` | SM | `MEMORY_BARRIER_OP` |
| `MEMBAR.GL` | GPU-global | `MEMORY_BARRIER_OP` |
| `MEMBAR.SYS` | system | `MEMBAR_OP` |

Treating all scopes identically is conservative and correct for forward progress.
Scope-aware modeling can be deferred.

### `FENCE` — scope decode is a Phase 6 prerequisite

| Form | Semantic role |
|---|---|
| `FENCE.VIEW.ASYNC.S` | async-proxy visibility ordering |
| `FENCE.PROXY.ASYNC` | TMA store proxy ordering (Phase 6 of TMA plan) |
| Other `FENCE.*` | various ordering forms |

The TMA architecture plan's Phase 6 (proxy-fence / ordering model) depends on
`FENCE.PROXY.ASYNC` behaving differently from other FENCE forms. Currently all FENCE
forms collapse into one `MEMORY_BARRIER_OP` with no distinguishing state. A scope /
proxy-kind field must be decoded at instruction build time before Phase 6 is
implementable. This decode is not in the H100 sync plan and should be added as an
explicit prerequisite for Phase 6.

---

## Grid-Level Sync

`GRID_BARRIER_OP` exists in `operation_type.h` and has live handlers in `sm.cc`,
`ldst_unit_sm.cc`, and `gpu-sim.cc`, but no Hopper or Blackwell SASS opcode maps to
it. Grid sync via `cooperative_groups::grid_group::sync()` compiles to `MEMBAR.GL`
plus atomics in Hopper SASS — not a dedicated single opcode. `GRID_BARRIER_OP` is
therefore a vestigial enum value in the trace-driven path. It is not a gap, but it
should be noted so no one tries to add a SASS opcode → `GRID_BARRIER_OP` mapping.

---

## `ACQBULK`: Ordering Role Unaddressed

`ACQBULK` is the bulk-acquire fence the compiler inserts to enforce visibility of TMA
returned data before consumer code accesses shared memory. Currently mapped to
`MISCELLANEOUS_NO_QUEUE_OP` — a no-op stub. The H100 sync plan does not mention it.

The unresolved question is whether FA3's consumer-side ordering is fully covered by
`SYNCS.TRYWAIT` (which the plan does model) or whether some paths rely on `ACQBULK`
as an additional or alternative ordering barrier.

- If FA3 uses only `SYNCS.TRYWAIT` for consumer ordering → `ACQBULK` stub is harmless.
- If FA3 uses `ACQBULK` to acquire TMA completion before reading → the stub silently
  breaks ordering and any resulting reads may observe stale data with no simulator
  indication of the problem.

This must be verified against the FA3 SASS before the H100 sync plan is considered
complete. At minimum, the plan should state explicitly which path FA3 uses.

---

## Warpgroup-Level Sync

`WARPGROUP` maps to `BRANCH_OP` and `WARPGROUPSET` maps to `MISCELLANEOUS_NO_QUEUE_OP`.
Neither has an execution model. The H100 plan defers "warpgroup synchronization
completeness" as a non-goal, which is correct. But the deferred item needs to be named:

- `WARPGROUP` controls which warps participate as a unit in warpgroup MMA (`HGMMA` /
  `WGMMA`). Without modeling group membership, the simulator cannot accurately account
  for warpgroup-level register-file handoffs or the occupancy effects of warpgroup
  scheduling.
- `WARPGROUPSET` is the corresponding teardown instruction.

These are low-priority for FA3 forward-progress but should appear on the deferred list
explicitly, not fall through silently as branch/misc ops.

---

## Blackwell Opcode Map: Hard Gaps

The Blackwell opcode map is missing every sync opcode introduced in Hopper. These are
not optional omissions — several are hard blockers for any Blackwell TMA kernel.

| Opcode | Hopper mapping | Blackwell status | Impact |
|---|---|---|---|
| `SYNCS` | `BARRIER_OP` | **missing** | Hard blocker for Blackwell TMA |
| `FENCE` | `MEMORY_BARRIER_OP` | **missing** | Hard blocker for proxy-fence / TMA store ordering |
| `UCGABAR_ARV` | `BARRIER_OP` | **missing** | Needed for cluster kernels |
| `UCGABAR_WAIT` | `BARRIER_OP` | **missing** | Same |
| `ACQBULK` | `MISCELLANEOUS_NO_QUEUE_OP` | **missing** | Any kernel using TMA bulk acquire |
| `WARPGROUP` | `BRANCH_OP` | **missing** | Warpgroup MMA |
| `WARPGROUPSET` | `MISCELLANEOUS_NO_QUEUE_OP` | **missing** | Same |
| `ELECT` | `BRANCH_OP` | **missing** | Election primitives |
| `ENDCOLLECTIVE` | `BRANCH_OP` | **missing** | Collective operation cleanup |

`SYNCS` missing from `blackwell_opcode.h` is the most critical gap. Blackwell uses
TMA with mbarrier identically to Hopper. Any Blackwell TMA kernel will produce
unrecognized-opcode crashes or silent misroutes on every `SYNCS` instruction. This is
an independent fix required in `blackwell_opcode.h` regardless of when the H100 sync
plan is implemented.

The simplest fix is to copy all Hopper sync opcodes into `blackwell_opcode.h` with
the same routing values as a baseline, then override specific entries as Blackwell
semantics diverge.

---

## Gen5 TC (Blackwell Tensor Core) Sync

The Blackwell opcode map includes `BGMMA` and `QGMMA` (both `TENSOR_CORE_OP`). It
does not include any `TCGEN05.*` family, which public Blackwell documentation indicates
is the Gen5 tensor memory / tensor core instruction family for matrix results and
commit operations.

If Blackwell gen5 TC kernels emit `TCGEN05.WAIT` or related commit/fence instructions,
those opcodes will be unrecognized. This is a gap to verify against actual Blackwell
traces once available. No action can be taken until traces are in hand, but the gap
should be on the known-unknowns list.

---

## Action Items Summary

| Item | Urgency | Owner scope |
|---|---|---|
| Route `UCGABAR_ARV/WAIT` away from `BARRIER_OP` before H100 plan lands | **Must fix** | `hopper_opcode.h`, `operation_type.h` |
| Add all missing Hopper sync ops to `blackwell_opcode.h` | **Must fix** | `blackwell_opcode.h` |
| Verify whether FA3 uses `ACQBULK` for consumer ordering | **Verify before closing H100 plan** | FA3 SASS inspection |
| Add `FENCE` scope / proxy-kind decode | **Required before TMA Phase 6** | `trace_driven.cc`, `tma_types.h` |
| Document `WARPSYNC`, `BSYNC`, `WARPGROUP`, `WARPGROUPSET` as explicit approximations | Low | plan docs |
| Investigate Blackwell `TCGEN05.*` once traces available | Deferred | `blackwell_opcode.h` |
