# Hopper `SYNCS` mbarrier ISA Notes (SM90a)

This document captures the observed SASS-level ISA for Hopper `SYNCS.*` mbarrier
instructions, as used by this simulator's trace pipeline. Everything here is derived
from real traces (nvcc microbenches and the CUTLASS FlashAttention-3 backward kernel),
cross-checked against the runtime values captured by the tracer in
`instruction.sync` (protobuf) and `sync_operand_resolver.json` (sidecar).

## Conventions

All `SYNCS.*` instructions use the same 3-operand textual layout:

```
SYNCS.<op> <operand0>, <operand1>, <operand2>
            ^dst/pred  ^barrier    ^semantic
```

- **operand 0** — destination / predicate. `RZ`/`URZ`/`Pn`. Often `RZ` (discarded).
- **operand 1** — barrier memory reference, e.g. `[UR6+0x30c00]`, `[R7+URZ]`, `[UR17]`.
  This selects **which mbarrier object** (its shared-memory address).
- **operand 2** — semantic operand (the "raw" value). A register holds a real value;
  `RZ` means the value is zero / not supplied.

Simulator identity for a barrier object is `(trace_kernel_id, cta_id, barrier_addr)`,
where `barrier_addr` is the effective address of operand 1.

### What operand 2 (`semantic_raw`) means per instruction

operand 2 is always the semantic/raw value, but its meaning depends on `sync_kind`:

| instruction | operand 2 (`semantic_raw`) meaning |
|---|---|
| `EXCH` | encoded init value → `expected_arrive_count = (0x200000 - raw) / 2` |
| `ARRIVE` (plain) | `RZ` → `semantic_raw == 0` → plain arrive, no tx bytes |
| `ARRIVE` (expect-tx) | register → `semantic_raw != 0` → the value is the expected tx bytes |
| `PHASECHK` / `TRYWAIT` | wait-state raw → input phase parity in **bit 31** |

### Critical rule: classify by operand 2, not by the opcode suffix

The `SYNCS.ARRIVE.TRANS64` suffix spelling is **toolchain-dependent** and must not be
used to decide the operation's meaning. The two builds emit disjoint suffix sets for
the same logical operations (see the ARRIVE section). The reliable signal is the
**semantic operand (operand 2)**:

- operand 2 = `RZ` (runtime `semantic_raw == 0`) → **plain arrive** (no tx bytes)
- operand 2 = register (runtime `semantic_raw != 0`) → **arrive-expect-tx** (tx bytes)

#### About the `AxTy` suffix digits

The `A` and `T` digits are a **3-state field per slot**, not booleans:

| digit | meaning |
|---|---|
| `0` | value is 0 (slot unused) |
| `1` | value is immediate `1` |
| `R` | value comes from the register operand |

So `A1T0` = arrive-count immediate 1, tx 0; `ART0` = arrive-count from register, tx 0;
`A0TR` = arrive-count 0, tx from register. This is why the earlier
"`A1`=arrive-on / `T0`=tx-off" boolean reading was wrong (it cannot represent the `R`
state seen in `.ART0` and `.A0TR`).

---

## Opcode summary

| opcode | role | `sync_kind` | barrier (op1) | semantic (op2) |
|---|---|---|---|---|
| `SYNCS.EXCH.64` | init / reset mbarrier, set expected arrive count | `EXCH` | mbarrier addr | encoded expected-count (register) |
| `SYNCS.ARRIVE.TRANS64*` (op2=reg) | arrive + expect transaction bytes | `ARRIVE_EXPECT_TX` | mbarrier addr | tx byte count (register) |
| `SYNCS.ARRIVE.TRANS64*` (op2=`RZ`) | plain / counted arrive | `ARRIVE` (plain) | mbarrier addr | `RZ` (0) |
| `SYNCS.PHASECHK.TRANS64` | test phase / wait predicate | `PHASECHK` | mbarrier addr | wait-state (phase parity) |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | try-wait on phase | `TRYWAIT` | mbarrier addr | wait-state (phase parity) |

---

## `SYNCS.EXCH.64` — initialize mbarrier / set expected arrive count

- **Meaning**: create/reset an mbarrier object at operand 1 and program its expected
  arrive count from the encoded value in operand 2.
- **Operands**: `op0 = URZ` (discarded), `op1 = [barrier]`, `op2 = encoded count (UReg)`.
- **Decode** (validated, see below):

  ```cpp
  expected_arrive_count = (0x200000 - exch_raw) / 2;   // raw encodes 2x logical count
  ```

### Examples

nvcc (microbench):
```
SYNCS.EXCH.64 URZ, [UR14], UR4
```

CUTLASS (FA3):
```
SYNCS.EXCH.64 URZ, [UR6+0x30c00], UR4
```

### Validated raw values

Microbench ground truth (`init_arrivals` knob → captured `semantic_raw`):

| `init_arrivals` | `exch_raw` | `0x200000 - raw` | `/2` (logical) |
|---|---|---|---|
| 1 | `0x1ffffe` | 2 | 1 |
| 2 | `0x1ffffc` | 4 | 2 |
| 3 | `0x1ffffa` | 6 | 3 |
| 4 | `0x1ffff8` | 8 | 4 |
| 8 | `0x1ffff0` | 16 | 8 |

So `0x200000 - raw == 2 * init_arrivals`. The decode base is `0x200000` (not
`0x2000000`), and the raw is scaled by 2.

FA3 observed: `0x1ffffe` (→1), `0x1ffe00` (→256, the count-closed barriers).

---

## `SYNCS.ARRIVE.TRANS64*` — arrive (plain or expect-tx)

One opcode family with several suffix spellings. **Use operand 2 to classify.**

- **operand 2 = register** → expect-tx arrive: `arrive_count += active_threads`
  **and** `expected_tx_bytes += semantic_raw`. The expect-tx arrive *also* counts
  as an arrival in hardware (mbarrier.arrive.expect_tx arrives once per thread).
- **operand 2 = `RZ`** → plain arrive: `arrive_count += active_threads` (no tx bytes).

**Arrive increment unit = active thread count, not 1 per warp.** mbarrier tracks
pending arrivals in thread units; a `SYNCS.ARRIVE` instruction arrives once per
active thread. Validated against FA3 traces: the count barrier executes the plain
arrive with `active_mask = 0xffffffff` (+32 per warp), the tx barrier executes the
expect-tx arrive with `active_mask = 0x1` (+1, leader thread only). With the
logical `expected_arrive_count` (EXCH `/2`), the per-phase arrivals match exactly:
count barrier `8 warps × 32 = 256`, tx barrier `1`.

`sync_kind` in the current resolver labels all of them `ARRIVE_EXPECT_TX`; the
runtime `semantic_raw` (or operand-2 text `RZ`) is what actually separates the two
behaviors. The semantic operand role is `EXPECT_TX_BYTES` when present.

### nvcc (microbench) variants

| SASS | op0 | op2 | classification |
|---|---|---|---|
| `SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR17], RZ` | `RZ` | `RZ` | plain arrive (count) |
| `SYNCS.ARRIVE.TRANS64.ART0 RZ, [UR17], R6` | `RZ` | register | counted arrive (count from reg) |
| `SYNCS.ARRIVE.TRANS64.RED.A0TR RZ, [UR17], R5` | `RZ` | register | expect-tx (tx bytes from reg) |

Notes:
- `.A1T0` → op2 always `RZ` (plain arrive, implicit count 1).
- `.ART0` → op2 always a register (arrive count supplied by register).
- `.RED.A0TR` → op2 always a register carrying transaction bytes (e.g. `256`).

### CUTLASS (FA3) variants

Only two arrive variants are actually **executed** by the FA3-bwd trace
(`b1-s2048-hd64`, causal); the other two exist in the SASS binary but were never
run for this input, so they appear in neither `enhanced_execution_info.json` nor
`sync_operand_resolver.json` and are **unverified at runtime**.

| SASS | op0 | op2 | runtime | classification |
|---|---|---|---|---|
| `SYNCS.ARRIVE.TRANS64 RZ, [R18+URZ+0x30c10], R3` | `RZ` | register | executed (19 sites) | expect-tx (tx bytes from reg) |
| `SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [UR9], RZ` | `RZ` | `RZ` | executed (11 sites) | plain arrive |
| `SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR11+0x22830], RZ` | `RZ` | `RZ` | **not executed** | (static only, unverified) |
| `SYNCS.ARRIVE.TRANS64.RED.A0T1 RZ, [UR11+0x22830], RZ` | `RZ` | `RZ` | **not executed** | (static only, unverified) |

Notes:
- Only the **suffix-less** `SYNCS.ARRIVE.TRANS64` carries a register op2 (expect-tx);
  `.RED.A1T0` has op2 = `RZ` (plain arrive). These are the only two that run.
- `.A1T0` and `.RED.A0T1` are present statically with op2 = `RZ`, but never executed
  for this input. Their static op2 = `RZ` *suggests* plain arrive, but this is not
  runtime-confirmed; the toolchain guards (below) reject them if they ever execute.
- The FA3 and nvcc suffix sets do **not** overlap, so suffix spelling cannot be used
  to decide the kind across builds.

### Runtime tx-byte values (FA3)

| arrive variant | observed `semantic_raw` |
|---|---|
| suffix-less `SYNCS.ARRIVE.TRANS64` (expect-tx) | `0x8000` (32768), `0x4200` (16896) |
| `.RED.A1T0` (plain) | always `0` |

---

## `SYNCS.PHASECHK.TRANS64` / `SYNCS.PHASECHK.TRANS64.TRYWAIT` — wait / try-wait

- **Meaning**: test the mbarrier (operand 1) against an expected phase parity carried
  in operand 2; produce a predicate (operand 0 = `Pn`).
- `PHASECHK` is the blocking-style phase check; `.TRYWAIT` is the non-blocking try
  variant. Both consult the same barrier and wait-state semantics.
- `sync_kind`: `PHASECHK` / `TRYWAIT`. Semantic role: `WAIT_STATE`.
- The wait-state raw is a **phase parity bit** (bit 31), not a barrier selector:
  operand 1 selects the barrier; operand 2 only says which phase you are waiting on.

### Examples

nvcc (microbench):
```
SYNCS.PHASECHK.TRANS64 P0, [UR17], R5
SYNCS.PHASECHK.TRANS64 P0, [R6+URZ], R11
```

CUTLASS (FA3):
```
SYNCS.PHASECHK.TRANS64        P0, [R7+URZ], R3
SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR5+0x30c00], RZ
```

### Validated wait-state raw values (FA3)

| `semantic_raw` | meaning |
|---|---|
| `0x0` | input phase parity 0 |
| `0x80000000` | input phase parity 1 (bit 31 set) |

Across the entire FA3-bwd trace the wait-state raw takes **only** these two values,
differing solely in bit 31. The parity must therefore be decoded from bit 31
(`(raw >> 31) & 1`), not bit 0.

### Wait-completion polarity (validated, was a bug)

Hopper `mbarrier.try_wait.parity` / `test_wait.parity` completes when the consumer's
input phase parity **DIFFERS** from the barrier's current phase parity — i.e. the
phase the consumer was waiting on has flipped away. Equality means the barrier is
still in that phase, so the wait keeps spinning. This is the inverse of the naive
"parity must match" intuition and is stated explicitly in the NVIDIA CUDA
Programming Guide (§4.9 Asynchronous Barriers) and the H100 async-barrier course
notes:

```cpp
// satisfied (proceed) when:
barrier.phase != decode_sync_wait_phase(wait_state_raw)   // bit 31
```

> **Bug found and fixed (FA3-bwd deadlock).** `remodeling/sm.cc` previously
> decoded the parity from **bit 0** (`raw & 0x1`) and compared with `==`. Because
> the FA3 wait-state only ever differs in bit 31, the bit-0 decode collapsed both
> values to `0`, so hit/miss depended solely on `barrier.phase`. Observed truth
> table from `...-53cb9e043fde.e296` (per-phase `wait miss` counts): `phase=0` →
> 0 misses (always hit), `phase=1` → 52,292 misses (always miss). Concretely, tx
> barrier `0x31000` flipped to `phase=1` after its TMA bytes completed but every
> subsequent consumer wait (parity 0) then missed forever (~52k `wait miss`,
> effectively a deadlock), while count barriers spuriously passed at `phase=0`.
> The fix is two lines: decode parity from bit 31 **and** flip the comparison to
> `!=`. With `decode(bit31) + !=`, `phase=1, parity=0` → `1 != 0` → proceed,
> exactly releasing the stuck `0x31000` waiters.

---

## FA3-bwd barrier roles (validated end-to-end)

Decoding the new FA3 per-CTA `.pb` joined with `sync_operand_resolver.json` across many
`kernel_10` CTAs shows the backward kernel uses exactly two barrier roles:

| barrier group | EXCH `0x200000-raw` | arrive variant | closes by |
|---|---|---|---|
| `0x31000/10/18/30/38` | 2 | suffix-less `ARRIVE` (op2 = register, tx `0x4200`/`0x8000`) | accumulated **tx bytes** |
| `0x31020/28/40/48` | 512 (logical 256) | `.RED.A1T0` (op2 = `RZ`) | accumulated **arrive count** |

This confirms: `.RED.A1T0` is a plain/counted arrive (not expect-tx), and the
expect-tx arrive is the register-op2 variant.

---

## Simulator mapping cheat-sheet

| observed | normalized action |
|---|---|
| `EXCH`, raw `r` | `expected_arrive_count = (0x200000 - r) / 2`; reset phase counters |
| arrive, `semantic_raw == 0` | plain arrive: `arrive_count += active_threads` |
| arrive, `semantic_raw != 0` | expect-tx: `arrive_count += active_threads` and `expected_tx_bytes += semantic_raw` |
| `PHASECHK` / `TRYWAIT`, raw `w` | proceed when `barrier.phase != ((w >> 31) & 1)` (parity in bit 31, completion polarity is `!=`) |
| ready condition | `arrive_count >= expected_arrive_count && completed_tx_bytes >= expected_tx_bytes` → flip phase |
| on phase flip | reset per-phase accumulators (`arrive_count`, `completed_tx_bytes`, `expected_tx_bytes`) to 0, flip the phase parity; **keep `expected_arrive_count`** |

### Phase-flip resets only the per-phase accumulators

`SYNCS.EXCH.64` (= `mbarrier.init`) runs **exactly once per barrier** (validated in
FA3 traces: each barrier shows a single EXCH versus many `PHASECHK`/arrive phases).
The hardware programs `expected_arrive_count` once and **reuses it on every phase**.
So when a phase flips, the simulator must reset only the per-phase accumulators
(`arrive_count`, `completed_tx_bytes`, `expected_tx_bytes`) and **preserve
`expected_arrive_count`**. Clearing `expected_arrive_count` on flip would make the
next phase ready after a single arrive (`arrive_count > 0 >= 0`), causing premature
phase flips / races. `expected_tx_bytes` *is* reset because the expect-tx arrive
re-sets it on each phase.

## Validated arrive variants and the unverified-variant guard

The resolver labels every `SYNCS.ARRIVE.TRANS64*` site `ARRIVE_EXPECT_TX`; the real
arrive-vs-expect-tx decision is made **at runtime from `semantic_raw`** (see the
cheat-sheet). Because suffix spelling is not a reliable cross-build signal, only the
arrive variants whose operand/semantics have been observed executing are trusted:

| variant | build | runtime op2 | behavior |
|---|---|---|---|
| `SYNCS.ARRIVE.TRANS64` | FA3 | register (tx bytes) | expect-tx |
| `SYNCS.ARRIVE.TRANS64.RED.A1T0` | FA3 | `RZ` (0) | plain arrive |
| `SYNCS.ARRIVE.TRANS64.RED.A0TR` | nvcc | register (tx bytes) | expect-tx |
| `SYNCS.ARRIVE.TRANS64.ART0` | nvcc | register, runtime 0 | plain arrive |
| `SYNCS.ARRIVE.TRANS64.A1T0` | nvcc | `RZ` (0) | plain arrive |

Any other `SYNCS.ARRIVE.TRANS64*` variant that actually executes is rejected loudly,
so an unverified encoding can never be silently mislabelled:

- `build_sync_operand_mapping.py` (`VALIDATED_ARRIVE_OPCODES`) raises `SystemExit`
  during resolver build.
- `remodeling/sm.cc` (`is_validated_arrive_opcode`) `abort()`s in
  `handle_sync_instruction` with a diagnostic (pc / cta / barrier).

If a new variant appears, validate its operand layout and runtime `semantic_raw`
(re-trace + decode the `.pb`) before adding it to both lists.

## Provenance

- nvcc SASS: `hw_run/.../utmapf-exch-arrive-4/.../traces/extra_info/sass/*.sass`,
  `.../sync-arrive-phase-4/...`, `.../utmapf-exch-tx-256b/...`
- CUTLASS SASS: `hw_run/.../flashattn-fa3-bf16-bwd-causal-.../traces/extra_info/sass/*.sass`
- Runtime values: per-CTA `traces/threadblocks/.../*.pb` (`instruction.sync`) joined with
  `traces/extra_info/sync_operand_resolver.json`
- EXCH knob ground truth: microbench `sync_producer_events.jsonl` (`EXCH_RAW.value_operand`)
