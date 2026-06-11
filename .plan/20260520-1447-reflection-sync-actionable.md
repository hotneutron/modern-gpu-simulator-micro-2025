# Sync Implementation: Critique and Actionable Plan — 2026-05-20

## Critique of `Full_sync_impl.md`

### What the plan gets right

The four-tier framing is correct: minimum / good / full / blackwell-support is a
meaningful ladder. The cross-tier invariants are sound. The FA3-first rule
(expected bytes from TMA issue, not ARRIVE operand) is the right call given the
evidence.

### Gaps and problems

#### 1. Step ordering in `H100_sync_impl.md` is inverted

The plan orders steps 1→2→3→4→5→6→7→8→9. The actual dependency chain is:

```
A: types + opcodes (steps 1–2)
  ↓
B: warp_inst_t + SYNCS decode (steps 3–4 and 9)
  ↓
C: mbarrier object storage in SM (step 5)
  ↓
D: consumer wait gating in subcore (step 6)
  ↓
E: TMA→mbarrier binding in tma_unit_sm (step 8)
  ↓
F: remove bypass (step 7)
```

Step 9 ("trace operand helpers") is listed last but is a prerequisite of step 4
("decode SYNCS in trace_driven.cc"). Removing the bypass (step 7) is listed sixth but
must be last — removing it before the new path is complete is a guaranteed crash.

#### 2. `UCGABAR` routing hazard is misclassified as "Good" — it is Minimum

The reflection note shows that `UCGABAR_ARV` and `UCGABAR_WAIT` both map to
`BARRIER_OP`, same as `BAR`. After `SYNCS` moves to `MBARRIER_OP`, they remain
`BARRIER_OP` and will enter `barrier_set_t` with wrong participant counts. Any
Hopper cluster kernel will hit incorrect behavior or assertion failures.

This is a mandatory fix before the bypass is removed, not a Good-tier cleanup.
A `CLUSTER_BARRIER_OP` stub is needed in the Minimum tier.

#### 3. `expected_tx_bytes` initialization path is unspecified

The plan says expected bytes come from TMA issue-time `cmd.total_bytes`. But the
plan does not connect the two objects:

- `SYNCS.EXCH` creates and initializes a `HopperMBarrierObject` (no TMA yet)
- TMA issue creates a `TMACompletionObject` (knows total_bytes, no mbarrier yet)

Who writes `expected_tx_bytes` into the mbarrier object, and when? The plan names
the rule ("use TMA total_bytes") but omits the binding mechanism. The answer is:
at TMA command build time, look up the active mbarrier address for the issuing warp
and write `cmd.total_bytes` into the matching `HopperMBarrierObject.expected_tx_bytes`.
`TMACommand` also needs `completion_barrier_addr` to record which mbarrier to update
at completion time. Neither of these is in the current `TMACommand` struct.

#### 4. Non-TMA arrive leaves the barrier permanently unready

The plan says ARRIVE operand values are unreliable for FA3 and should be ignored for
expected bytes. Fine. But if a warp issues `SYNCS.ARRIVE` without a corresponding
in-flight TMA (warp-leader arrive pattern, epilogue, etc.), the mbarrier never
accumulates any `completed_tx_bytes` and `TRYWAIT` stalls forever.

The minimum tier must specify the fallback: if no TMA is bound to the mbarrier and
`completed_tx_bytes == 0`, then a non-TMA ARRIVE should set `ready = true` directly
(the no-transfer-arrive case). This rule is absent from the plan.

#### 5. Blackwell opcode gaps are trivially fixable and should not be Tier 4

The missing opcodes in `blackwell_opcode.h` (SYNCS, FENCE, UCGABAR_ARV/WAIT,
ACQBULK, WARPGROUP, WARPGROUPSET, ELECT, ENDCOLLECTIVE) are one-liner additions
with identical routing to Hopper. Deferring them to Tier 4 means any Blackwell run
crashes on SYNCS. The correct approach: add them as part of the opcode infrastructure
step at the very start. Blackwell semantics refinement is Tier 4; opcode map parity
is Tier 1.

#### 6. No concrete pass/fail criterion for "minimum achieved"

The tier document says minimum = FA3 forward progress. Forward progress needs a
concrete definition for it to be testable: **FA3 kernel 10 (H100 bwd) executes past
the first `SYNCS.TRYWAIT` without hitting a deadlock timeout, and the mbarrier debug
log shows at least one `TRYWAIT released` event.**

---

## Current Implementation State

From code inspection:

| Component | Status |
|---|---|
| `operation_type.h` — `MBARRIER_OP`, `CLUSTER_BARRIER_OP` | **Missing** |
| `tma_types.h` — Hopper sync structs | **Missing** |
| `tma_types.h` — `completion_barrier_addr` on `TMACommand` | **Missing** |
| `hopper_opcode.h` — `SYNCS` → `MBARRIER_OP` | **Missing** (still `BARRIER_OP`) |
| `hopper_opcode.h` — `UCGABAR` → `CLUSTER_BARRIER_OP` | **Missing** (still `BARRIER_OP`) |
| `blackwell_opcode.h` — SYNCS + FENCE + UCGABAR + others | **Missing** |
| `abstract_hardware_model.h` — `m_hopper_sync_info` on warp_inst_t | **Missing** |
| `trace_driven.cc` — SYNCS decode | **Missing** |
| `sm.h/.cc` — mbarrier object storage + helpers | **Missing** |
| `subcore.cc` — mbarrier wait gating | **Missing** |
| `tma_unit_sm.cc` — TMA→mbarrier binding | **Missing** |
| `sm.cc` — bypass | **Active** (string-prefix hack) |
| `tma_types.h` — TMA transfer structs | ✓ Complete |
| `tma_unit_sm.cc` — transfer state machine, completion objects | ✓ Complete |

---

## Actionable Plan

Tasks are in dependency order. Each phase's output is the input to the next.

---

### Phase A — Opcode and Type Infrastructure

No behavioral changes. Pure plumbing. All of these can land in one commit.

#### A1. `operation_type.h` — add two new op types

```cpp
MBARRIER_OP,       // Hopper address-based mbarrier (SYNCS.*)
CLUSTER_BARRIER_OP,  // Hopper cluster-scope barriers (UCGABAR_*)
```

#### A2. `tma_types.h` — add Hopper sync structs and extend `TMACommand`

Add to `tma_types.h`:

```cpp
enum class HopperSyncOpKind {
  NONE,
  SYNCS_EXCH,
  SYNCS_ARRIVE,
  SYNCS_ARRIVE_RED,
  SYNCS_PHASECHK,
  SYNCS_TRYWAIT,
};

struct HopperSyncInstructionInfo {
  HopperSyncOpKind kind = HopperSyncOpKind::NONE;
  uint64_t barrier_addr = 0;
  bool has_barrier_addr = false;
  uint64_t value_operand = 0;
  bool has_value_operand = false;
  bool valid = false;
};

struct HopperMBarrierObject {
  uint64_t barrier_addr = 0;
  unsigned cta_id = 0;
  uint64_t expected_tx_bytes = 0;
  uint64_t completed_tx_bytes = 0;
  uint32_t phase = 0;
  uint32_t arrive_count = 0;
  bool initialized = false;
  bool ready = false;
  bool tma_bound = false;   // true once a TMA command has been associated
};
```

Extend `TMACommand` with:

```cpp
uint64_t completion_barrier_addr = 0;
bool has_completion_barrier_addr = false;
```

#### A3. `hopper_opcode.h` — reroute two opcode families

```cpp
// change:
{"SYNCS",        OpcodeChar(OP_SYNCS, BARRIER_OP)},
{"UCGABAR_ARV",  OpcodeChar(OP_UCGABAR_ARV, BARRIER_OP)},
{"UCGABAR_WAIT", OpcodeChar(OP_UCGABAR_WAIT, BARRIER_OP)},

// to:
{"SYNCS",        OpcodeChar(OP_SYNCS, MBARRIER_OP)},
{"UCGABAR_ARV",  OpcodeChar(OP_UCGABAR_ARV, CLUSTER_BARRIER_OP)},
{"UCGABAR_WAIT", OpcodeChar(OP_UCGABAR_WAIT, CLUSTER_BARRIER_OP)},
```

#### A4. `blackwell_opcode.h` — add all missing Hopper sync opcodes

Add with Hopper-equivalent routing as the baseline. Blackwell semantic divergence
is a Tier 4 concern; opcode recognition is Tier 1.

```cpp
{"SYNCS",         OpcodeChar(OP_SYNCS, MBARRIER_OP)},
{"FENCE",         OpcodeChar(OP_FENCE, MEMORY_BARRIER_OP)},
{"UCGABAR_ARV",   OpcodeChar(OP_UCGABAR_ARV, CLUSTER_BARRIER_OP)},
{"UCGABAR_WAIT",  OpcodeChar(OP_UCGABAR_WAIT, CLUSTER_BARRIER_OP)},
{"ACQBULK",       OpcodeChar(OP_ACQBULK, MISCELLANEOUS_NO_QUEUE_OP)},
{"WARPGROUP",     OpcodeChar(OP_WARPGROUP, BRANCH_OP)},
{"WARPGROUPSET",  OpcodeChar(OP_WARPGROUPSET, MISCELLANEOUS_NO_QUEUE_OP)},
{"ELECT",         OpcodeChar(OP_ELECT, BRANCH_OP)},
{"ENDCOLLECTIVE", OpcodeChar(OP_ENDCOLLECTIVE, BRANCH_OP)},
```

#### A5. `subcore.cc` — add routing cases for the two new op types

In `get_fu()`:

```cpp
case MBARRIER_OP:
  fu = m_miscellaneous_no_queue_pipeline;  // will be refined in Phase D
  break;
case CLUSTER_BARRIER_OP:
  fu = m_miscellaneous_no_queue_pipeline;  // stub: no-op, never enters barrier_set_t
  break;
```

**Verification gate for Phase A**: build cleanly; no existing regression trace changes
behavior (SYNCS is still bypassed in sm.cc, so routing change has no live effect yet).

---

### Phase B — SYNCS Decode in warp_inst_t and trace_driven.cc

#### B1. `abstract_hardware_model.h` — add Hopper sync info to `warp_inst_t`

Add member and accessors:

```cpp
HopperSyncInstructionInfo m_hopper_sync_info;

bool has_hopper_sync_info() const { return m_hopper_sync_info.valid; }
const HopperSyncInstructionInfo &get_hopper_sync_info() const {
  return m_hopper_sync_info;
}
void set_hopper_sync_info(const HopperSyncInstructionInfo &info) {
  m_hopper_sync_info = info;
}
```

Initialize `m_hopper_sync_info.valid = false` in `warp_inst_t` default constructor.

#### B2. `trace_driven.cc` — decode SYNCS forms

Add `decode_hopper_syncs_info()` called from `parse_from_trace_struct()` when
`op == MBARRIER_OP`. Decode rules:

| Opcode prefix | `kind` | Barrier addr source | Notes |
|---|---|---|---|
| `SYNCS.EXCH` | `SYNCS_EXCH` | memory-ref operand 1 | value operand = operand 2 if present |
| `SYNCS.ARRIVE.TRANS64` | `SYNCS_ARRIVE` | memory-ref operand 1 | value operand ignored in FA3-first |
| `SYNCS.ARRIVE.TRANS64.RED` | `SYNCS_ARRIVE_RED` | memory-ref operand 1 | treat as ARRIVE for now |
| `SYNCS.PHASECHK.TRANS64` | `SYNCS_PHASECHK` | memory-ref operand 1 | — |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | `SYNCS_TRYWAIT` | memory-ref operand 1 | consumer wait |

The barrier address comes from the traced memory reference of the first operand
(already captured in `addrs_or_reg_val_0[first_active_lane]`).

**Verification gate for Phase B**: add a decode-time log that prints the first few
decoded `HopperSyncInstructionInfo` records from the FA3 trace. Confirm barrier
addresses are nonzero and stable across calls from the same CTA.

---

### Phase C — Mbarrier Object Storage in SM

#### C1. `sm.h` — add mbarrier state

```cpp
#include "tma_types.h"

// keyed by (cta_id, barrier_addr)
std::map<std::pair<unsigned, uint64_t>, HopperMBarrierObject> m_mbarriers;

// active barrier address per (cta_id, warp_id)
std::map<std::pair<unsigned, unsigned>, uint64_t> m_active_mbarrier_addr_by_warp;

HopperMBarrierObject &get_or_create_mbarrier(unsigned cta_id, uint64_t addr);
const HopperMBarrierObject *find_mbarrier(unsigned cta_id, uint64_t addr) const;
void set_active_mbarrier_addr(unsigned cta_id, unsigned warp_id, uint64_t addr);
bool get_active_mbarrier_addr(unsigned cta_id, unsigned warp_id, uint64_t &addr_out) const;
void reset_mbarriers_for_cta(unsigned cta_id);
void notify_mbarrier_tma_complete(unsigned cta_id, uint64_t barrier_addr,
                                  uint64_t bytes_completed);
```

#### C2. `sm.cc` — implement mbarrier helpers and MBARRIER_OP execution

**In the warp execution path** (where the bypass currently lives), replace the bypass
with real MBARRIER_OP handling:

```cpp
if (pipe_reg->op == MBARRIER_OP && pipe_reg->has_hopper_sync_info()) {
  const HopperSyncInstructionInfo &si = pipe_reg->get_hopper_sync_info();
  HopperMBarrierObject &obj =
      get_or_create_mbarrier(cta_id, si.barrier_addr);

  switch (si.kind) {
    case HopperSyncOpKind::SYNCS_EXCH:
      obj.barrier_addr = si.barrier_addr;
      obj.cta_id = cta_id;
      obj.expected_tx_bytes = 0;   // set later at TMA issue
      obj.completed_tx_bytes = 0;
      obj.phase = 0;
      obj.arrive_count = 0;
      obj.initialized = true;
      obj.ready = false;
      obj.tma_bound = false;
      set_active_mbarrier_addr(cta_id, warp_id, si.barrier_addr);
      break;

    case HopperSyncOpKind::SYNCS_ARRIVE:
    case HopperSyncOpKind::SYNCS_ARRIVE_RED:
      obj.arrive_count++;
      set_active_mbarrier_addr(cta_id, warp_id, si.barrier_addr);
      // FA3-first: if no TMA is bound and no expected bytes set,
      // a pure warp arrive completes the barrier directly
      if (!obj.tma_bound && obj.expected_tx_bytes == 0) {
        obj.ready = true;
      }
      break;

    case HopperSyncOpKind::SYNCS_PHASECHK:
      // evaluates readiness; no state change needed at Minimum tier
      break;

    case HopperSyncOpKind::SYNCS_TRYWAIT:
      // wait gating handled in subcore; execution here just records intent
      break;
  }
}

if (pipe_reg->op == CLUSTER_BARRIER_OP) {
  // no-op stub: must NOT enter barrier_set_t or warp_reaches_barrier
  // cluster semantics deferred to Tier 3
}
```

Remove the string-prefix bypass. `SYNCS` is now `MBARRIER_OP` and never reaches
the `BARRIER_OP` branch.

**Implement `notify_mbarrier_tma_complete`:**

```cpp
void SM::notify_mbarrier_tma_complete(unsigned cta_id, uint64_t barrier_addr,
                                      uint64_t bytes_completed) {
  auto it = m_mbarriers.find({cta_id, barrier_addr});
  if (it == m_mbarriers.end()) return;
  HopperMBarrierObject &obj = it->second;
  obj.completed_tx_bytes += bytes_completed;
  if (obj.completed_tx_bytes >= obj.expected_tx_bytes && obj.expected_tx_bytes > 0) {
    obj.ready = true;
    obj.phase ^= 1;
  }
}
```

**Verification gate for Phase C**: add one-time debug prints for first EXCH, first
ARRIVE, first TRYWAIT. Run FA3 kernel 10; confirm mbarrier objects are created at
barrier addresses seen in the SASS and that arrive_count increments.

---

### Phase D — Consumer Wait Gating in Subcore

#### D1. `subcore.cc` — add mbarrier predicate and integrate into issue gating

Add near the DEPBAR check in the warp issue path:

```cpp
bool SM::is_mbarrier_wait_satisfied(const warp_inst_t *inst) const {
  if (inst->op != MBARRIER_OP || !inst->has_hopper_sync_info()) return true;
  const HopperSyncInstructionInfo &si = inst->get_hopper_sync_info();
  if (si.kind != HopperSyncOpKind::SYNCS_TRYWAIT &&
      si.kind != HopperSyncOpKind::SYNCS_PHASECHK) return true;
  const HopperMBarrierObject *obj = find_mbarrier(
      m_physical_warp[inst->warp_id()]->get_cta_id(), si.barrier_addr);
  if (obj == nullptr) return false;  // object not yet initialized → stall
  return obj->ready;
}
```

Integrate in the subcore issue predicate: if `!sm->is_mbarrier_wait_satisfied(inst)`,
do not issue — re-add the instruction back to the issue slot or leave the warp
stalled.

**Verification gate for Phase D**: run FA3 kernel 10; confirm TRYWAIT warps stall
(debug print "TRYWAIT blocked") rather than crashing or silently executing.

---

### Phase E — TMA→Mbarrier Binding

#### E1. `tma_unit_sm.cc` — bind mbarrier at command build time and update at completion

**In `build_tma_command()`**, after setting `cmd.warp_id` and `cmd.cta_id`, look up
the issuing warp's active barrier address and bind it:

```cpp
uint64_t barrier_addr = 0;
if (m_sm->get_active_mbarrier_addr(cmd.cta_id, cmd.warp_id, barrier_addr)) {
  cmd.completion_barrier_addr = barrier_addr;
  cmd.has_completion_barrier_addr = true;
  // set expected_tx_bytes in the mbarrier object now that we know the size
  HopperMBarrierObject *obj = m_sm->find_mbarrier_mutable(cmd.cta_id, barrier_addr);
  if (obj != nullptr && obj->initialized) {
    obj->expected_tx_bytes = cmd.total_bytes;
    obj->tma_bound = true;
  }
}
```

**In `advance_in_flight_transfers()`**, when transitioning to `COMPLETED`:

```cpp
if (entry.cmd.has_completion_barrier_addr) {
  m_sm->notify_mbarrier_tma_complete(
      entry.cmd.cta_id,
      entry.cmd.completion_barrier_addr,
      entry.cmd.total_bytes);
}
```

**Verification gate for Phase E**: run FA3 kernel 10; confirm debug log shows at least
one `TMA completion → mbarrier ready` event and at least one `TRYWAIT released` event.

---

### Phase F — Remove Bypass and Validate

Remove the `bypass_syncs_barrier` string-prefix hack from `sm.cc`. `SYNCS` now routes
through `MBARRIER_OP` and never reaches the `BARRIER_OP` / `barrier_set_t` branch.

**Pass criterion for Minimum tier:**
- FA3 kernel 10 executes past the first `SYNCS.TRYWAIT` without deadlock timeout
- Debug log shows: mbarrier created, TMA bound, TMA completed, TRYWAIT released
- Non-TMA regression traces (rodinia, existing test suite) show no behavioral change

---

## Tier 2 (Good) — Concrete Tasks After Minimum

Tier 2 begins once FA3 kernel 10 runs end-to-end.

| Task | Description |
|---|---|
| **G1** | Replace warp-active-barrier heuristic with explicit EXCH→TMA-issue→TRYWAIT chain tracking: record the mbarrier address set at EXCH time, assert it matches at TMA issue |
| **G2** | Refine `SYNCS.EXCH` semantics: distinguish init-from-scratch from phase-reset-in-loop; handle repeated EXCH on the same address correctly |
| **G3** | Verify `ACQBULK` role in FA3 SASS: if any consumer path uses ACQBULK as the ordering fence rather than SYNCS.TRYWAIT, add ordering check to `ACQBULK` execution |
| **G4** | Add `FENCE` scope decode: distinguish `FENCE.PROXY.ASYNC` from `FENCE.VIEW.ASYNC.S`; store scope kind on `warp_inst_t` as prerequisite for TMA Phase 6 |
| **G5** | Run at minimum two non-FA3 Hopper TMA kernels; confirm mbarrier model generalizes beyond FA3's specific geometry and barrier-address pattern |

## Tier 3 (Full) — Concrete Tasks

| Task | Description |
|---|---|
| **F1** | Phase/parity protocol: TRYWAIT checks `phase == expected_phase`; EXCH resets phase; each completion round flips phase |
| **F2** | Full `ARRIVE.RED` reduction semantics: decode the reduction mode, model the atomic contribute-and-wait protocol |
| **F3** | `UCGABAR_ARV / UCGABAR_WAIT` real semantics: cluster-scope gather/arrive model separate from CTA mbarrier |
| **F4** | `FENCE.PROXY.ASYNC` enforcement: TMA store ordering requires proxy fence before consumer access; wire scope decode from G4 into enforcement |
| **F5** | Multiple producers / multiple consumers on one mbarrier object: arrive_count semantics must gate readiness correctly when more than one producer warp arrives |

## Tier 4 (Blackwell) — Concrete Tasks

| Task | Description |
|---|---|
| **B1** | Verify Blackwell SYNCS semantics match Hopper or diverge; update Blackwell routing if divergent |
| **B2** | Verify Blackwell FENCE forms; update scope decode for Blackwell-specific proxy variants if needed |
| **B3** | Investigate `TCGEN05.*` Gen5 TC sync once Blackwell traces are available; add opcode entries and stub routing |
| **B4** | Run at least one Blackwell TMA kernel; confirm no unrecognized-opcode crashes on SYNCS, FENCE, ACQBULK |
