# Fix OP_BAR named/partial/arrive barrier decoding (FA3 bwd kernel 10 CTA serialization)

## Summary

The simulator overestimates FA3 bwd (trace kernel 10 = `FlashAttnBwdSm90`) cycles by ~2.8×.
The dominant stall is `inst_barrier` (BAR.SYNC) at 58% of issue cycles / 87.7% of
`no_warps_ready`. Root cause: the trace decoder collapses **every** `BAR.*` instruction
into a single full-CTA blocking `bar.sync 0`, discarding the real barrier **id**, **thread
count**, and the **arrive-vs-sync** distinction that the FA3 warp-specialized pipeline
depends on.

This plan corrects the `OP_BAR` decode in `trace_driven.cc` to honor the operands that are
already present in the trace, so producer/consumer warpgroups stop being artificially
serialized.

---

## Answers to the three questions (grounded in this trace)

**Q1 — Why is trace kernel id 10 = function id 8? Are they the same kernel?**
Yes, same kernel; the two numbers are *different ID spaces* and are not expected to match.
- `trace_kernel_id` = launch order (1-based), used for the `kernel_<N>` directory.
- `unique_function_id` = static-function identity (assigned in discovery order).
- The mapping is stored explicitly per launch in `dynamic_trace.pb`
  (`kernel.function_unique_id`, resolved in `trace_parser.cc:436-457`).
- Decoded directly from `dynamic_trace.pb`:
  `trace_kernel_id=10 → func_unique_id=8`, block `(384,1,1)`,
  name `_ZN7cutlass13device_kernelIN5flash20enable_sm90_or_laterINS1_16FlashAttnBwdSm90...`.
  So k10 is `FlashAttnBwdSm90`, and func_unique_id 8 is the correct static record for it.

**Q2 — OP_BAR was never implemented and not touched this time; why does FA3 specifically
blow up?**
Because FA3 bwd is *warp-specialized* and leans heavily on **named, partial-count, and
arrive-only** barriers. Verified opcode mix for func_unique_id 8 (23 `BAR.*` total):
- 13 `BAR.SYNC.DEFER_BLOCKING` with explicit ids `0x0,0x1,0x9,0xd,0xe,...` and counts
  `0xa0`(160), `0x100`(256), `0x120`(288) — i.e. *subsets* of the 384-thread CTA.
- 10 `BAR.ARV` (arrive-only, non-blocking) with ids `0xa,0xb,0x4,0x8,...`.
The placeholder decode (`bar_id=0; bar_count=-1; bar_type=SYNC`) turns all 23 into one
full-CTA blocking barrier. Ordinary (non-warp-specialized) kernels use a single
`__syncthreads`-style full-CTA `BAR.SYNC 0x0`, for which the placeholder *happens* to be
correct — that is why this bug was invisible until FA3.

**Q3 — TMA runs async with the cuda cores; OP_BAR is a cuda-core op. Why does it affect the
whole run?**
The barrier never stalls the TMA hardware unit directly. It stalls the **warp issue**:
- `warp_reaches_barrier` sets `m_warp_at_barrier` for `bar_type==SYNC`
  (`shader.cc:4231-4232`), which makes `shd_warp_t::waiting()` true
  (`shader.cc:4487`), which clears `is_not_warp_waiting_in_programmer_barrier`
  and removes the warp from issue (`subcore.cc:520,555`).
- The barrier only releases when `at_barrier == active` (all 12 warps), `shader.cc:4238`.
- Effect on the async pipeline: the **producer warps** that *issue* the `UTMALDG` /
  mbarrier-arrive instructions are themselves warps. If a producer warp executes a
  `BAR.ARV` (which should let it continue immediately) but the sim blocks it as a full-CTA
  `BAR.SYNC`, the producer stops issuing → it stops launching new TMA loads and stops
  arriving on consumer mbarriers. So the *cuda-core barrier bug indirectly throttles the
  TMA producer*, stalling the whole decoupled pipeline. The async TMA unit is idle not
  because TMA is slow, but because the warp that feeds it is wrongly parked at a barrier.

---

## Current State Analysis

### The defect
`gpu-simulator/trace-driven/trace_driven.cc:540-552`
```cpp
case OP_CGAERRBAR:
case OP_MEMBAR:
case OP_BAR:
  // TO DO: fill this correctly
  bar_id = 0;
  bar_count = (unsigned)-1;
  bar_type = SYNC;
  break;
```
`bar_id`, `bar_count`, `bar_type` are members of the inst
(`abstract_hardware_model.h:1234-1238`, setters `set_bar_id/set_bar_count` at 1209-1210).

### How barriers are consumed downstream (must stay compatible)
- Arrival/release: `barrier_set_t::warp_reaches_barrier` (`shader.cc:4212-4260`).
  - `m_bar_id_to_warps[bar_id].set(warp_id)` — uses `bar_id`.
  - sets `m_warp_at_barrier` only for `bar_type == SYNC || RED` (`shader.cc:4231`).
  - `bar_count == -1` → wait for all active warps; else release when
    `at_barrier.count()*warp_size == bar_count` (`shader.cc:4248-4250`).
- Issue gating: `subcore.cc:520` `c_warp->waiting()`; `subcore.cc:555`,
  `subcore.cc:591-593` (the `inst_barrier` stat increment).
- Dispatch into the barrier set: `sm.cc:610-615` (`op == BARRIER_OP` →
  `warp_reaches_barrier`).
- `m_max_barriers_per_cta` controls the size of `m_bar_id_to_warps` — must be large enough
  for the ids actually used (kernel 10 uses ids up to `0xe`=14).

### Operand availability (verified for func_unique_id 8, 23 barriers)
Per-instruction static operands are reachable at decode time via
`static_trace_info.get_kernel_by_unique_function_id(unique_function_id).get_instruction(pc)`
(pattern already used at `trace_driven.cc:373-374`), and `trace.opcode` / `opcode_tokens`
carry the `.ARV` / `.SYNC` modifier.

| group | count | id | count operand |
|---|---|---|---|
| `BAR.ARV` | 10 | 7 imm / 3 reg | all imm |
| `BAR.SYNC` | 13 | 11 imm / 2 reg | 10 imm, 1 none (bare `0x0`), 2 imm |

- **18 / 23** have an immediate id; **all but the bare `0x0`** have an immediate count.
- **5 / 23** use a register-form id (`R7,R12,R13,R14,R23`) whose runtime value is **not**
  recorded anywhere in the trace (`instruction.proto` only has `sync_runtime_info` for
  `SYNCS.*` mbarriers, not for `BAR.*` register operands). These cannot be resolved exactly
  from this trace.

---

## Proposed Changes

### Change 1 — Decode real barrier semantics in `trace_driven.cc` (`OP_BAR` case)
File: `gpu-simulator/trace-driven/trace_driven.cc` (the `case OP_BAR:` block ~line 542).

Split `OP_BAR` away from `OP_MEMBAR`/`OP_CGAERRBAR` (they keep the existing placeholder),
and for `OP_BAR` decode from the static operands + opcode modifier:

- **bar_type**: if `opcode_tokens` contains `"ARV"` → `bar_type = ARRIVE`; else
  `bar_type = SYNC`. (Keep `RED` out of scope — no `BAR.RED` in this kernel.)
- **bar_id**: from `get_instruction(pc).get_operand(0)`, resolve via a single PRIORITY
  CHAIN: (1) if the operand is an immediate → use it; (2) else if a runtime register value
  is present in the trace (Change 4 field) → use it; (3) else (register form, no runtime
  field = old trace) → fall back to `bar_id = 0`.
- **bar_count**: from operand 1, same priority chain: (1) immediate → use it (thread
  count); (2) register + runtime field present → use it; (3) no second operand (bare
  `BAR.SYNC 0x0`) → `(unsigned)-1` (full CTA, correct); (4) register form, no runtime field
  → fall back to `(unsigned)-1`.

Implement this as ONE decision function so Change 4 only needs to *populate* the runtime
field — the decoder logic is written once here and does not change later.

Use the existing accessor pattern; do not introduce new parsing. Add a concise comment
explaining the priority chain and the register-form fallback.

Rationale: 18/23 barriers get exact ids and 22/23 get exact counts; the
arrive-vs-sync split is exact for all 23. This removes the artificial all-CTA blocking for
the immediate-form majority and stops `BAR.ARV` from blocking at all.

### Change 2 — Ensure `ARRIVE` barriers do not block the issuing warp
File: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc:4231` (and verify `sm.cc:610-615`).

`warp_reaches_barrier` already sets `m_warp_at_barrier` only for `SYNC || RED`, so an
`ARRIVE` will register the warp's arrival in `m_bar_id_to_warps[bar_id]` (so a later
matching `BAR.SYNC` on the same id can release) **without** parking the arriving warp.
Action: confirm no other path forces `ARRIVE` to block; if `sm.cc` unconditionally treats
`BARRIER_OP` as blocking, gate the `store_info_of_last_inst_at_barrier`/wait behavior on
`bar_type != ARRIVE`. (Likely no code change needed beyond Change 1; this is a
verification + minimal guard step.)

### Change 3 — Make `m_max_barriers_per_cta` cover the ids actually used
File: barrier-set construction (`shader.cc:4175 allocate_barrier`, and where
`m_bar_id_to_warps` is sized / `m_max_barriers_per_cta` is set).

Kernel 10 uses ids up to `0xe` (14). Verify `m_max_barriers_per_cta >= 16` (Hopper has 16
hardware named barriers). If it is currently smaller (legacy default), raise it to 16 so
`m_bar_id_to_warps[bar_id]` indexing is safe. Add an assert/clamp so an out-of-range id
fails loudly rather than corrupting memory.

### Change 4 (register-form ids) — capture runtime register values by RE-TRACING
The 5 register-form barriers (`BAR.SYNC R7/R13/R14`, `BAR.ARV R12/R23`) have NO runtime
value in the current trace. We extend the NVBit tracer to record them, reusing the EXACT
mechanism the tracer already uses for `SYNCS.*` mbarrier operands
(`sync_runtime_capture_sites_by_pc`). No new device-side machinery is needed.

**How the tracer pipeline works (verified, `tracer_tool.cu`):**
- Instrumentation (per static instruction, lines 1532-1620): each operand is walked. Only
  `REG`/`UREG`/`PRED` operands emit a device callback and bump `num_of_injects`; **IMM
  operands emit NO callback**. So `BAR.SYNC R7, 0x100` produces exactly ONE callback (for
  `R7`); `BAR.ARV 0xa, 0xa0` (both IMM) produces ZERO → a `NO_REGS` dummy (line 1622).
- For `REGULAR` operands the device already ships the live register value via
  `nvbit_add_call_arg_reg_val(...)` (line 1688); it lands in
  `ma->addrs_or_reg_val_0[lane]` on the host.
- The `SYNCS.*` path tags, at instrumentation time, WHICH callback_index carries the
  barrier/semantic value (`build_sync_runtime_capture_site_info` → stored in
  `sync_runtime_capture_sites_by_pc`, line 1618), then at host post-processing
  (lines 2300-2331) matches `callback_index` and writes `addrs_or_reg_val_0[first_lane]`
  into the proto `sync` field. **We mirror this 1:1 for `BAR.*`.**

**Sub-steps:**
1. **Proto** (`util/traces_enhanced/dynamic_trace/instruction.proto`): add
   `optional uint32 bar_id_runtime` and `optional uint32 bar_count_runtime` (or one small
   `bar_runtime_info{bool valid; uint32 id; bool has_id; uint32 count; bool has_count;}`
   message — preferred, mirrors `sync_runtime_info` and keeps "which fields are present"
   explicit so the immediate-form path is untouched).
2. **Capture-site tagging** (new function next to `build_sync_runtime_capture_site_info`,
   ~line 351): for opcodes starting with `BAR`, classify each operand by POSITION:
   operand 0 = barrier id, operand 1 = count. If that operand is `REG` (not IMM), record
   its `callback_index` as `bar_id_callback_index` / `bar_count_callback_index` into a new
   `bar_runtime_capture_sites_by_pc[func_id][vpc]` map (mirror of the sync map, populated at
   line 1617-1620). IMM operands need no capture (decoder reads them statically).
3. **Host write** (`tracer_tool.cu` ~lines 2300-2331, beside the sync block): if the PC is
   in `bar_runtime_capture_sites_by_pc` and the current `callback_index` matches
   `bar_id_callback_index` (resp. count), set `inst->mutable_bar_runtime()` fields from
   `ma->addrs_or_reg_val_0[first_lane]` (use `get_first_predicated_lane`, same as sync).
   Keep `mem_type==NONE`; do NOT touch `add_addresses()`.
4. **Parser read** (`gpu-simulator/trace-parser/trace_parser.cc` ~lines 300-339): read the
   new field(s) into `inst_trace_t`, mirroring how `pb_inst.sync()`/`udesc_value` are read.
5. **Decoder use**: NONE — Change 1's priority chain already prefers the runtime field over
   the (0,-1) fallback. Change 4 only makes the field present.
6. **Re-generate** the FA3 bwd trace with the rebuilt tracer into a new dir, re-run k10.

**Why operand-position classification is safe:** SASS `BAR.SYNC bar, count` / `BAR.ARV
bar, count` always order id first, count second; verified against the 23 barriers of func
8 (immediate forms show `0x9,0x100`; register forms show `R7,0x100` / `R13,0xa0`). The
count is consistently the 2nd operand and an IMM in all observed cases, so in practice only
the id is ever register-form for SYNC; ARRIVE has a few register ids too. The plan still
captures both positions generically in case a future kernel uses a register count.

**Backward compatibility**: the new proto fields are `optional`; old traces without them
decode via the existing (0,-1) fallback, so this does not break other runs.

**Scope note / decision (D6 — REVISED execution order)**: Implement and validate the
TRACER side (Change 4) FIRST, before any simulator change. Rationale: the user wants to
confirm the new trace actually captures `bar_id`/`bar_count` correctly (including the 5
register-form barriers) before building simulator logic that consumes it. This avoids
debugging decode + engine + capture all at once. Only after the re-generated trace is
verified do we touch the decoder (Change 1-3) and the barrier engine (B1-B4).

---

## Assumptions & Decisions

- **A1**: Block dim is 384 threads / 12 warps / CTA (verified from `dynamic_trace.pb`).
- **A2**: Barrier counts in the trace are in **threads** (e.g. `0xa0`=160 = 5 warps × 32),
  matching the `at_barrier.count()*warp_size` test. The engine assumes each participating
  warp contributes a full 32 threads, so counts MUST be multiples of 32; this is enforced
  by the explicit guard B5 (`bar_count % warp_size == 0`), not just assumed.
- **A3 (CORRECTED after the deadlock investigation)**: `BAR.ARV` = arrive-only,
  non-blocking. **`BAR.SYNC.DEFER_BLOCKING` is NOT a plain blocking `bar.sync`.** It
  performs a named-barrier arrive whose actual *wait* is split off and expressed by the
  instruction's `control_bits.wait_barrier_bits` (a scoreboard wait), NOT by a CTA-wide
  rendezvous block at the BAR itself. Treating it as an immediate blocking SYNC was wrong
  and produced a real deadlock (see "Deadlock root cause" below). The correct blocking-vs-
  non-blocking signal is `wait_barrier_bits`, not the opcode:
  - `wait_barrier_bits != 0` → the wait is handled by the scoreboard model
    (`Subcore::is_wait_barriers_ready_entry_point`, already implemented). The BAR should
    be modeled as a non-blocking ARRIVE; blocking it causes producer/consumer cycles.
  - `wait_barrier_bits == 0` → no scoreboard wait, so the BAR itself must block as a real
    CTA-wide `__syncthreads` (e.g. reduce_kernel id=0).

### Deadlock root cause (FA3 bwd full run, gpu_sim_cycle≈128751) and the BAR.SYNC.DEFER_BLOCKING semantics

The first full OnlyKernel10 run terminated via `-gpgpu_deadlock_detect`, not `exit
detected`. Investigation (BARDBG logs + per-CTA `.pb` walk + SASS `control_bits`):

- The named-barrier groups form a producer/consumer handshake, e.g. id=13 needs
  `SYNC(warp1) + ARV(warp4..7)` and id=10 needs `SYNC(warp4..7) + ARV(warp1)`. Modeling
  every `BAR.SYNC.DEFER_BLOCKING` as an immediate CTA block makes warp1 park at id=13
  before it can issue its `BAR.ARV id=10`, while warp4 parks at id=10 before issuing its
  `BAR.ARV id=13` → cyclic deadlock. This is independent of in-order issue (HW is also
  in-order and does NOT deadlock).
- SASS `control_bits` for the *executed* `BAR.SYNC.DEFER_BLOCKING` in func 8 show
  `is_new_*_barrier=False` and `wait_barrier_bits=22 (0b010110 = SB 1,2,4)`. Tracing the
  setters: SB 1/2/4 are written by the immediately-preceding `SYNCS.EXCH.64` (mbarrier
  init) at pc 0x510/0x530/0x540. So the BAR's wait is waiting on the warp's own
  `SYNCS.*` completion — NOT on a cross-warp named-barrier rendezvous. The named-barrier
  arrive is just a count signal; the real synchronization is scoreboard + mbarrier
  (`SYNC_ISA.md`), both already modeled.

### kernel 5 (reduce_kernel, ufid 4) cross-check — different scenario
`reduce_kernel` uses 9x `BAR.SYNC.DEFER_BLOCKING 0x0` (full-CTA, no count) with
**`wait_barrier_bits=0`** on every one. With no scoreboard wait, these ARE real CTA-wide
`__syncthreads` and MUST block. This is why the blocking decision must key off
`wait_barrier_bits`, not the `DEFER_BLOCKING` suffix: the same opcode is a non-blocking
arrive in FA3 (wait offloaded to scoreboard) but a blocking sync in reduce_kernel (no
scoreboard wait. The decoder/engine fix must preserve blocking for the
`wait_barrier_bits==0` case so reduce_kernel does not lose synchronization.

### All-kernel census (9 traced kernels) and the FINAL blocking rule
Every `BAR*` instruction in `enhanced_execution_info.json` was categorized by
(opcode, id-kind, count-kind, wait_barrier_bits, predicate). Result:

| kernel(s) | category | classification |
|---|---|---|
| distribution(1), reduce(4), BwdPostprocess(9) | `SYNC.DEFER` id=0 full-CTA **wait==0** | **SYNC (blocking __syncthreads)** |
| FA3 fwd(3), FA3 bwd(8) | `SYNC.DEFER` id=0 full-CTA **wait!=0** | ARRIVE (wait is a scoreboard wait on the warp's own preceding `SYNCS.EXCH`) |
| FA3 fwd(3), FA3 bwd(8) | `SYNC.DEFER` named(id!=0) and/or partial count | ARRIVE (named handshake; any `wait!=0` waits on a preceding `FENCE.VIEW.ASYNC.S`, traced) |
| FA3 fwd(3), FA3 bwd(8) | `BAR.ARV` (any id/count, wait 0 or !=0) | ARRIVE |

Only two opcode forms appear across all kernels: `BAR.ARV` and `BAR.SYNC.DEFER_BLOCKING`
(no plain `BAR.SYNC`, no `BAR.RED`, no `BAR.ALL.*` as `OP_BAR`). The `wait_barrier_bits`
of every non-blocking case was traced to a scoreboard set by a preceding `SYNCS.EXCH.64`
(mbarrier init) or `FENCE.VIEW.ASYNC.S` — i.e. the warp's own async ordering, never a
cross-warp named-barrier rendezvous. So those waits are already modeled by the scoreboard
path (`Subcore::is_wait_barriers_ready_entry_point`) and the BAR itself must NOT block.

**FINAL rule (implemented in `trace_driven.cc` OP_BAR):**
```
is_plain_full_cta_syncthreads = SYNC.DEFER && id==0 && count==full(-1) && wait_barrier_bits==0
bar_type = is_plain_full_cta_syncthreads ? SYNC : ARRIVE
```
- `BAR.ARV` and `BAR.SYNC.DEFER_BLOCKING` are the only accepted opcode forms; **any other
  BAR form reaching `OP_BAR` aborts** (`assert(is_arv || is_sync_defer)`) so an
  uncharacterized form can never be silently mis-modeled.
- B4 (`bar_id < MAX_BARRIERS_PER_CTA`) and B5 (`bar_count % warp_size == 0`) asserts remain.

### Deadlock confirmed on FA3 fwd too (kernel 5 / ufid 3)
The OnlyKernel5 run (FA3 fwd) hit the same `-gpgpu_deadlock_detect` abort
(`gpu_sim_cycle 101251`), with the same named/partial handshake pattern (id 4/8/9 SYNC at
count 416/256 + matching ARV). The FINAL rule reclassifies all of these as ARRIVE and so
resolves the fwd deadlock identically to bwd.

- **D1**: Register-form id/count → fall back to (0, -1). Not exact, but safe and bounded
  (only 5/23 barriers, mostly the dynamically-scheduled paths).
- **D2**: Scope is the `OP_BAR` decode + barrier-set plumbing only. No changes to TMA,
  mbarrier (`SYNCS.*`), DEPBAR, or memory-latency models. `OP_MEMBAR`/`OP_CGAERRBAR` keep
  current behavior.
- **D3**: This is the OnlyKernel10 run; correctness is evaluated against trace kernel 10
  (`FlashAttnBwdSm90`) only.
- **D7 (BAR ≠ SYNCS — keep the two mechanisms separate)**: `BAR.SYNC`/`BAR.ARV` are CTA
  **named barriers** (PTX `bar.sync`/`bar.arrive`): warp↔warp rendezvous, identified by
  `bar_id` (0-15), handled by `BARRIER_OP`→`warp_reaches_barrier`. They are NOT the
  TMA/async mechanism. The TMA↔compute async handshake is `SYNCS.*` **mbarrier**
  (`MBARRIER_OP`→`handle_sync_instruction`, tx-bytes + phase parity), already designed and
  validated in `SYNC_ISA.md`. This work touches ONLY the `BAR.*` path; it must not modify
  or depend on the `SYNCS.*` logic. The `BAR` bug affects the run *indirectly*: blocking a
  warp that should only `arrive` delays every later instruction that warp would issue
  (including its subsequent `SYNCS.ARRIVE`), so the mbarrier/TMA pipeline stalls as a
  second-order effect — not because `BAR` gates TMA directly.

### Relation to TMA_ARCH.md Phase 6 (Proxy-Fence / Ordering Model)

Phase 6 (`TMA_ARCH.md:1173-1204`) is *related context* but a *separate mechanism*; it
does not overlap with or block this work.

- **Same handshake, different opcode class.** Phase 6's `FENCE.VIEW.ASYNC.S` (399x) appears
  almost exclusively right before mbarrier-init or before a **`BAR.ARV`** — i.e. it guards
  the very producer↔consumer handshake whose `BAR.ARV` we are fixing. But `FENCE*` →
  `OP_FENCE`/`MEMORY_BARRIER_OP` (`sm.cc:598-604`), whereas our bug is in `BAR*` →
  `OP_BAR`/`BARRIER_OP` (`trace_driven.cc:540-552`). Different op classes, no shared code;
  Phase 6's "no change needed" conclusion does **not** apply to `OP_BAR`.
- **D4 (leverage Phase 6 finding)**: the simulator has **no generic/async proxy split** —
  the mbarrier is one `(kernel,cta,addr)` object written in-cycle by `SYNCS.EXCH`
  (`sm.cc:1580-1598`). Therefore, once we make `BAR.ARV` non-blocking, we do **not** need
  to add any proxy-fence ordering/visibility edge around the producer/consumer handshake;
  the visibility delay it would guard against is already zero. This keeps our fix minimal.
- **D5 (deadlock caution, aligned with Phase 6)**: Phase 6 warns against adding stalls
  into FA3's already-dense mbarrier dependencies. Our Change 3 (bar_id range assert), D1
  (safe register-form fallback), and Verification step 6 must confirm the run terminates
  cleanly under `-gpgpu_deadlock_detect 1`, exactly as Phase 6 was validated
  (`TMA_ARCH.md:1202`).

---

## Barrier engine — concrete implementation plan

Decoding correct `(bar_type,bar_id,bar_count)` is not enough; the engine in
`barrier_set_t` must execute partial/named/arrive barriers correctly. Below is the exact
current behavior (read from source) and the precise change for each case.

### How the engine works today (verified)
- State per barrier set (`shader.h` / `shader.cc:4143-4172`):
  - `m_warp_at_barrier` — one bit per warp; "this warp is blocked at a barrier".
    `warp_waiting_at_barrier(warp_id)` returns exactly this bit (`shader.cc:4287-4289`),
    and `shd_warp_t::waiting()` ORs it into issue-gating (`subcore.cc:520`).
  - `m_bar_id_to_warps[bar_id]` — one warp_set_t per id; "which warps have arrived at id".
- Arrival (`warp_reaches_barrier`, `shader.cc:4212-4260`):
  1. `assert(bar_id != -1)`; `m_bar_id_to_warps[bar_id].set(warp_id)`.
  2. if `SYNC||RED` → `m_warp_at_barrier.set(warp_id)` (block). ARRIVE does NOT block. ✅
  3. release test: if `bar_count==-1` → release when `at_barrier == active` (whole CTA);
     else release when `at_barrier.count()*warp_size == bar_count`.
  4. on release: clear those warps from `m_bar_id_to_warps[bar_id]` and `m_warp_at_barrier`.
- Dispatch (`sm.cc:610-615`): EVERY `BARRIER_OP` calls
  `store_info_of_last_inst_at_barrier(...)` then `warp_reaches_barrier(...)` — including
  ARRIVE.
- CTA alloc/dealloc (`shader.cc:4175-4209`): `allocate_barrier` clears the CTA's warps from
  all ids; `deallocate_barrier` asserts no warp is still parked. ids are independent. ✅

### Structural limitation to respect (+ explicit guard)
`m_bar_id_to_warps` tracks arrival at **warp granularity** and the count path computes
arrived threads as `count()*warp_size`, i.e. it ASSUMES every participating warp arrives
with a full 32 active threads. It does NOT sum per-warp active-thread counts, so a barrier
whose `bar_count` is not a whole-warp multiple of 32 (a sub-warp / divergent participation)
cannot be represented. Verified true for kernel 10: `0xa0`=160=5w, `0x100`=256=8w,
`0x120`=288=9w — all multiples of 32. No sub-warp partial-arrival modeling is added.

**Explicit guard (B5)**: at the `OP_BAR` decode AND at `warp_reaches_barrier` entry, when
`bar_count != (unsigned)-1`, assert `bar_count % warp_size == 0`. If a future kernel emits
a non-32-multiple count, the run aborts loudly (with pc/bar_id/bar_count) instead of
silently mis-counting and either hanging or releasing early. This converts the unmodeled
case into a clear, diagnosable failure rather than a wrong result.

### B1 — exact-count release (`==`) → make it `>=`
`shader.cc:4250`: `(at_barrier.count()*warp_size) == bar_count`. With named barriers an
id can legitimately accumulate arrivals across ARRIVE+SYNC and momentarily overshoot the
exact equality window between cycles, never firing. **Change**: use `>=`:
`(at_barrier.count()*warp_size) >= bar_count`. On release, clear ONLY the participating
warps (current `&= ~at_barrier` already clears all arrived-at-this-id warps, which is
correct since the whole id cohort is releasing together).

### B2 — ARRIVE must not park the warp, and must not be treated as "last inst at barrier"
- Blocking is already skipped for ARRIVE (`shader.cc:4231`). ✅ (no change)
- But `sm.cc:613` calls `store_info_of_last_inst_at_barrier` unconditionally. For a
  non-blocking ARRIVE this can leave stale "waiting" bookkeeping
  (`check_if_non_released_reduction_barrier`/`waiting()` paths consult last-inst-at-barrier
  state). **Change**: in `sm.cc`, only call `store_info_of_last_inst_at_barrier` when
  `bar_type != ARRIVE` (i.e. for SYNC/RED). Still call `warp_reaches_barrier` for ARRIVE so
  the arrival is recorded on the id.
- **ARRIVE semantics decided**: an ARRIVE warp is added to `m_bar_id_to_warps[bar_id]` and
  counts toward that id's release total, exactly like a SYNC arrival, but its own
  `m_warp_at_barrier` bit is never set, so it keeps issuing. When a later SYNC on the same
  id pushes the total to `>= bar_count`, the whole cohort (including the earlier ARRIVE
  warps) is cleared from the id. This matches HW arrive/wait split.

### B3 — ARRIVE re-entry / producer loop correctness
FA3 producers execute the same `BAR.ARV` repeatedly across the K-loop. Because B2 clears
the id only on release, an ARRIVE warp that arrives again before release would call
`.set(warp_id)` on an already-set bit (idempotent — fine), but two arrivals from the SAME
warp must not be double-counted. Since `m_bar_id_to_warps` is a bitset keyed by warp_id,
re-arrival is naturally idempotent (count unchanged). **Action**: confirm by assertion
that a warp re-arriving at an id it is already recorded on does not inflate the count;
add a one-line guard/comment. No structural change expected.

### B4 — id range safety
`m_max_barriers_per_cta` comes from `-gpgpu_num_cta_barriers`; hard cap
`MAX_BARRIERS_PER_CTA=16` (`shader.cc:4160`). Kernel 10 uses ids ≤ 14. **Change**: ensure
the config value is 16 (Hopper) and add `assert(bar_id < m_max_barriers_per_cta)` at decode
AND at `warp_reaches_barrier` entry so an unexpected id fails loudly.

### B5 — count must be a whole-warp multiple of 32 (explicit guard)
The release math `at_barrier.count()*warp_size` assumes full-32-thread warps (see
"Structural limitation"). **Change**: when `bar_count != -1`, assert
`bar_count % warp_size == 0` at decode AND at `warp_reaches_barrier` entry, with a
diagnostic (pc / bar_id / bar_count). This makes any unmodeled sub-warp/divergent barrier
abort loudly instead of silently hanging or releasing early.

### Decision order (these interlock)
B2 (ARRIVE counts toward the id, never blocks) defines the accounting; B1 (`>=`) makes the
SYNC release robust given that accounting; B3 guarantees idempotent re-arrival; B4 (id
range) and B5 (count%32) are independent safety asserts. Implement B4+B5 → B2 → B1 → B3,
then wire decode (Change 1-3).

### Files touched by the engine work
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc` — `warp_reaches_barrier` (`==`→`>=`,
  re-arrival guard, id assert B4, count%32 assert B5).
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc` — gate
  `store_info_of_last_inst_at_barrier` on `bar_type != ARRIVE`.
- `gpu-simulator/trace-driven/trace_driven.cc` — decode-time B4/B5 asserts.
- (config) `gpgpusim.config` `-gpgpu_num_cta_barriers` = 16 if not already.

These are the "does it run correctly" guarantees; they gate the plan as much as the
decode.

---

## Execution order & Verification (two phases)

### Phase A — Tracer first, verify the NEW trace in isolation (NO simulator change yet)
Goal: prove the regenerated trace carries correct `bar_id`/`bar_count` for all 23 barriers
of func 8 — especially the 5 register-form ones — before writing any consumer code.

A1. **Implement Change 4 only** (proto + capture-site tagging + host write + parser-side
    proto field; the parser read into `inst_trace_t` can be a no-op stub for now). Rebuild
    the NVBit tracer (`util/tracer_nvbit/`).
A2. **Re-generate the trace** for the FA3 bwd run on the H100 host into a NEW trace dir
    (do not overwrite the existing one). This requires running on real hardware with the
    rebuilt `tracer_tool.so`.
A3. **Inspect the new `.pb` directly** (small python protobuf walk, like the one already
    used to read `dynamic_trace.pb`): for kernel_10's threadblock files, dump every `BAR*`
    instruction's `bar_runtime` field and cross-check against the static operands in
    `enhanced_execution_info.json`:
    - immediate-form barriers: `bar_runtime` absent/`valid=false` (decoder will use IMM).
    - register-form barriers (`BAR.SYNC R7,0x100`, `BAR.ARV R12,0xa0`, ...): `bar_runtime`
      present, `id` a plausible small value (0-15), and (if a reg count ever appears) a
      multiple of 32.
A4. **Sanity vs HW named-barrier count**: the distinct runtime ids captured should fall in
    [0,15] and match the named-barrier ids the kernel is expected to use (0,1,5,8,9,a,b,d,e
    + the resolved register ones). If a register id resolves to something ≥16 or wildly
    varying per-CTA, STOP and re-examine the operand-position assumption before proceeding.
A5. **Old-trace compatibility**: confirm the simulator (unchanged) still parses the new
    `.pb` without error (new field is `optional`), and that an OLD trace still parses too.

Only when A3/A4 pass do we move to Phase B.

### Phase B — Simulator: decode + engine, then measure
B1. **Decoder (Change 1-3)** + wire the parser read of `bar_runtime` into `inst_trace_t`
    and into the priority chain.
B2. **Barrier engine (B1-B4 in the engine section)**: `>=` release, ARRIVE non-block +
    no `store_info_of_last_inst_at_barrier`, re-arrival idempotence, id-range assert.
B3. **Build** the simulator.
B4. **Static decode sanity via the built-in barrier debug print**: run kernel 10 with
    `-sync_debug_enable 1`. This activates `debug_print_sm_barrier_issue`
    (`remodeling/sm.cc:177-200`), which emits one stderr line per BARRIER_OP issue:
    `[SMDBG][barrier-issue] sm=.. warp=.. pc=0x.. op=.. trace_opcode=.. bar_id=.. bar_count=..`.
    Use it to MEASURE (not assume) the decode + per-warp arrival:
    - confirm `BAR.ARV` lines show `bar_type`/non-blocking handling and the expected id;
    - confirm `BAR.SYNC 0x9,0x100` → `bar_id=9 bar_count=256`; bare `0x0` →
      `bar_id=0 bar_count=-1`;
    - register-form lines show ids matching the Phase-A `bar_runtime` capture;
    - group the printed lines by `warp=` to verify that each named barrier is actually
      issued by the expected NUMBER of warps (e.g. a `count=256` barrier is reached by 8
      distinct warps), directly answering "do all participating warps carry the BAR?".
    NOTE: the printer has a hard `budget = 64` lines (`sm.cc:184`); for a fuller per-warp
    census, temporarily raise that budget (or filter to a single CTA) when running this
    check. This is a debug-only measurement, reverted before final runs.
B5. **No deadlock**: run kernel 10 with `-gpgpu_deadlock_detect 1`; must terminate.
B6. **Stall distribution**: `inst_barrier` share drops sharply from ~58% (and
    `no_warps_ready` from 66.6%) toward the HW `barrier` share (~17%).
B7. **Cycle accuracy**: `gpu_tot_sim_cycle` falls from 361,760 toward HW 132,901; compare
    to `.result/FA3_kernel_10_bwd.md`.
B8. **Regression guard**: a plain full-CTA `__syncthreads` kernel (e.g. `reduce_kernel`,
    single `BAR.SYNC 0x0`) is unchanged (id=0, count=-1, SYNC).

---

## Files to touch
- `gpu-simulator/trace-driven/trace_driven.cc` — `OP_BAR` decode (Change 1).
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc` — verify/guard ARRIVE non-block
  (Change 2), `m_max_barriers_per_cta` sizing + bounds (Change 3).
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc` — verify `BARRIER_OP` dispatch
  respects `bar_type == ARRIVE` (Change 2).

(Read-only references used for grounding: `trace_parser.cc`, `abstract_hardware_model.h`,
`subcore.cc`, `enhanced_execution_info.json`, `dynamic_trace.pb`, `instruction.proto`.)
