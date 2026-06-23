# BAR_OP on H100: named/partial/arrive barrier decoding + warp-exit drain (FA3 warp-specialized kernels)

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
- **A4 (REVISED after the teardown-assert investigation)**: a named barrier's counted
  release is NOT always representable as "one credit per unique warp id". The current
  engine stores only `m_bar_id_to_warps[bar_id]` (a warp bitset), so a warp that performs
  `BAR.ARV` and later a matching `BAR.SYNC.DEFER_BLOCKING` on the SAME id contributes only
  one bit. The new logs from FA3 show barriers whose required `bar_count` is satisfied only
  if that later SYNC contributes an additional arrival credit. Therefore the current B3
  assumption ("same-warp re-arrival is always idempotent") is too strong for these named
  handshake barriers.
- **A5 (NEW after the first accounting patch failed)**: preserving only `bar_type` is NOT
  sufficient. `BAR.ARV` and `BAR.SYNC.DEFER_BLOCKING` can both be non-blocking from the
  scheduler's point of view (`bar_type = ARRIVE`), but they are NOT the same for counted
  release accounting. The decode must preserve a BAR-internal subtype (`bar_subop`) so the
  engine can distinguish:
  - `BAR.ARV` = base arrive credit
  - `BAR.SYNC.DEFER_BLOCKING` = non-blocking execution, but potentially an additional
    deferred-sync credit if the same warp already executed `BAR.ARV` for that id/epoch.

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

**FINAL blocking rule (implemented in `trace_driven.cc` OP_BAR):**
```
is_plain_full_cta_syncthreads = SYNC.DEFER && id==0 && count==full(-1) && wait_barrier_bits==0
bar_type = is_plain_full_cta_syncthreads ? SYNC : ARRIVE
```
- `BAR.ARV` and `BAR.SYNC.DEFER_BLOCKING` are the only accepted opcode forms; **any other
  BAR form reaching `OP_BAR` aborts** (`assert(is_arv || is_sync_defer)`) so an
  uncharacterized form can never be silently mis-modeled.
- B4 (`bar_id < MAX_BARRIERS_PER_CTA`) and B5 (`bar_count % warp_size == 0`) asserts remain.

**NEW subtype rule (must be preserved alongside `bar_type`):**
```
if opcode == BAR.ARV:
    bar_subop = BAR_SUBOP_ARV
elif is_plain_full_cta_syncthreads:
    bar_subop = BAR_SUBOP_SYNC_PLAIN
elif opcode == BAR.SYNC.DEFER_BLOCKING:
    bar_subop = BAR_SUBOP_SYNC_DEFER_BLOCKING
```

This is the missing piece from the first fix: deadlock avoidance requires
`BAR.SYNC.DEFER_BLOCKING` to stay non-blocking in many FA3 cases, but counted-release
accounting still needs to know that it is a deferred-sync form, not the same event as
`BAR.ARV`.

### Deadlock confirmed on FA3 fwd too (kernel 5 / ufid 3)
The OnlyKernel5 run (FA3 fwd) hit the same `-gpgpu_deadlock_detect` abort
(`gpu_sim_cycle 101251`), with the same named/partial handshake pattern (id 4/8/9 SYNC at
count 416/256 + matching ARV). The FINAL rule reclassifies all of these as ARRIVE and so
resolves the fwd deadlock identically to bwd.

### Post-deadlock follow-up: teardown assert shows undercounted named-barrier credits
After the FINAL blocking rule was implemented, the early producer/consumer deadlock went
away, but both kernels later failed at CTA teardown:

- `shader.cc:4223` `deallocate_barrier()` asserts that no per-id barrier state remains.
- OnlyKernel10 and OnlyKernel5 both terminate on that assert, which means a named-barrier
  id cohort never reached its release threshold and remained recorded until CTA exit.

The new BARDBG logs show a very specific pattern:

- **OnlyKernel10**: `bar_id=1, bar_count=288` never releases. The observed participants are
  `warp 4..11` doing `BAR.ARV` at `pc=0x7df0` (8 warps total), and then `warp 11` doing a
  later `BAR.SYNC.DEFER_BLOCKING` on the SAME id/count at `pc=0x7e10`. `288 / 32 = 9`, so
  the logs match **8 ARRIVE credits + 1 later SYNC credit**, not 9 distinct warp ids.
- **OnlyKernel5**: `bar_id=1, bar_count=416` shows the analogous pattern. `warp 4..15` do
  `BAR.ARV` at `pc=0x80f0` (12 warps), and then `warp 15` later executes
  `BAR.SYNC.DEFER_BLOCKING` on the SAME id/count at `pc=0x8170`. `416 / 32 = 13`, so this
  likewise matches **12 ARRIVE credits + 1 later SYNC credit**.
- In contrast, barriers that release successfully under the current engine are exactly the
  ones whose `bar_count/32` already matches the number of UNIQUE participating warps
  (e.g. OnlyKernel10 `bar_id=9,count=256`, OnlyKernel5 `bar_id=1,count=384`).

This strongly suggests the new failure mode is NOT a generic decode bug that adds `+32`
threads to every barrier count. Instead, it is an **engine accounting bug**: some named
handshake barriers require one additional arrival credit from a warp that first arrives
non-blockingly and later performs the matching SYNC, but the current bitset-based state
(`m_bar_id_to_warps`) collapses both events into one bit.

### Follow-up after the first accounting patch: why the extra-credit fix did not fire
The first accounting patch added separate "ARRIVE credited" and "SYNC credited" state, but
it keyed the extra-credit path on `bar_type == SYNC`. That patch still failed on
OnlyKernel10, and the new BARDBG summary/leak logs explained why:

- the teardown assert remained (`shader.cc:4223`);
- `BARDBG[summary]` showed `parked=0` but `leaked_ids>0`, so this is a participant-only
  leak, not a warp-still-blocked case;
- `BARDBG[leak]` for the failing CTA showed `bar_id=1` with `arv_seen=8` and
  `defer_sync_extra_credits=0`;
- all `BARDBG[release]` lines still had `released_credits == released_warps`, i.e. the
  expected extra credit was never granted.

Root cause: `BAR.SYNC.DEFER_BLOCKING` had already been intentionally reclassified to
`bar_type = ARRIVE` to avoid the original deadlock. That is still correct for blocking
semantics, but it means the engine cannot use `bar_type` alone to tell `BAR.ARV` apart from
`BAR.SYNC.DEFER_BLOCKING`. So the extra-credit patch never triggered.

This proves the next fix must preserve **two orthogonal dimensions**:

- `bar_type`: scheduler / parking semantics (`SYNC` vs non-blocking `ARRIVE`);
- `bar_subop`: BAR-internal semantic subtype (`ARV`, `SYNC_DEFER_BLOCKING`, plain
  full-CTA `SYNC`, `RED`).

The secondary `undefined instruction` asserts seen in stdout/stderr are treated as
post-abort noise, not the root cause:

- the barrier teardown assert fires first;
- only afterwards does another thread trip `trace_driven.cc:341` while running the
  instruction-region prewarm path (`get_instruction_regions_to_prewarm` /
  `enqueue_instruction_region_prewarm`).

So the next fix target is the barrier engine's counted-release accounting, not opcode-map
coverage.

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
    This can represent only UNIQUE participating warps. It cannot represent multiple
    arrival credits from the same warp in one barrier epoch.
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
`shader.cc:4250` currently uses the number of UNIQUE participating warps:
`at_barrier.count()*warp_size`. That is insufficient for the new failure mode: some named
barriers require `ARRIVE + later SYNC` from the SAME warp to contribute **two** credits in
one epoch. **Change**:

- keep `m_bar_id_to_warps[bar_id]` as the participating-warp set to clear on release;
- add a separate per-id `arrival_credit_count`;
- counted release becomes `arrival_credit_count * warp_size >= bar_count`.

Use `>=`, not `==`, for the same reason as before: once credit accounting is fixed,
ARRIVE+SYNC sequences can legally overshoot the equality boundary before the release test
fires.

### B2 — ARRIVE must not park the warp, and must not be treated as "last inst at barrier"
- Blocking is already skipped for ARRIVE (`shader.cc:4231`). ✅ (no change)
- But `sm.cc:613` calls `store_info_of_last_inst_at_barrier` unconditionally. For a
  non-blocking ARRIVE this can leave stale "waiting" bookkeeping
  (`check_if_non_released_reduction_barrier`/`waiting()` paths consult last-inst-at-barrier
  state). **Change**: in `sm.cc`, only call `store_info_of_last_inst_at_barrier` when
  `bar_type != ARRIVE` (i.e. for SYNC/RED). Still call `warp_reaches_barrier` for ARRIVE so
  the arrival is recorded on the id.
- **ARRIVE semantics decided**: an ARRIVE warp is added to `m_bar_id_to_warps[bar_id]` and
  contributes one arrival credit, but its own `m_warp_at_barrier` bit is never set, so it
  keeps issuing. A later SYNC on the same id may contribute an ADDITIONAL arrival credit
  before parking, depending on whether that warp has already consumed its per-epoch SYNC
  credit for this id. Release still clears the whole participating cohort from the id.

### B3 — replace "idempotent re-arrival" with split accounting AND subtype preservation
The earlier B3 assumption is now contradicted by the logs. "Same-warp re-arrival is
idempotent" is true only for repeated ARRIVE of the SAME role. It is NOT generally true
for the mixed named-handshake pattern:

- `BAR.ARV (warp X, id=k)` may contribute one arrival credit without blocking.
- the SAME `warp X` may later execute `BAR.SYNC.DEFER_BLOCKING (id=k)` and contribute one
  additional arrival credit before parking.

Therefore the engine needs split state, and decode must preserve which BAR subtype actually
arrived:

- `participant_warps[bar_id]` — which warps are recorded on this id and must be cleared on
  release;
- `arv_seen_warps[bar_id]` — which warps executed `BAR.ARV` on this id in the current
  epoch;
- `deferred_sync_extra_credit_warps[bar_id]` — which warps have already consumed their
  per-epoch extra credit via `BAR.SYNC.DEFER_BLOCKING` on this id.

Minimal policy:

- `BAR.ARV`:
  - add the warp to `participant_warps`;
  - mark it in `arv_seen_warps`;
  - contributes the base credit already represented by `participant_warps.count()`.
- `BAR.SYNC.DEFER_BLOCKING`:
  - add the warp to `participant_warps`;
  - execution stays non-blocking (`bar_type = ARRIVE`);
  - if this warp is already in `arv_seen_warps` and not yet in
    `deferred_sync_extra_credit_warps`, grant one extra credit and mark it there.
- plain blocking `BAR.SYNC` / `BAR.RED`:
  - add the warp to `participant_warps`;
  - park it according to `bar_type`;
  - no deferred extra credit unless a future kernel proves otherwise.
- total counted-release credit:
  - `participant_warps.count() + deferred_sync_extra_credit_warps.count()`
- Release on id `k`: clear `participant_warps[k]`, `arv_seen_warps[k]`,
  `deferred_sync_extra_credit_warps[k]`, and the parked-warp bits that were waiting on that
  id.

This is the smallest change that explains BOTH problematic cases:

- OnlyKernel10 `bar_id=1,count=288` = `8 ARV + 1 later SYNC`.
- OnlyKernel5 `bar_id=1,count=416` = `12 ARV + 1 later SYNC`.

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

### Debug signals required for the next validation run
The next validation run must print enough information to prove the subtype-aware credit path
is or is not firing:

- `BARDBG[issue]` must include both `bar_type` and `bar_subop`;
- `BARDBG[deferred-sync-credit]` should print the first successful extra-credit grant for a
  `(bar_id, bar_subop, bar_count)` tuple;
- `BARDBG[deferred-sync-miss]` should print the first miss, with
  `had_prior_arv` / `already_extra_credited`;
- `BARDBG[release]` must include `released_warps` and `released_credits`;
- `BARDBG[summary]` / `BARDBG[leak]` must remain enabled so participant-only leaks are still
  visible at CTA teardown.

### Decision order (these interlock)
B2 (ARRIVE non-blocking) defines blocking behavior; the new subtype rule preserves
`BAR.ARV` vs `BAR.SYNC.DEFER_BLOCKING`; B3 (split arrival-credit accounting keyed by
subtype) fixes the undercounted named-handshake barriers; B1 (`>=` on the new credit
count) makes release robust once that accounting is correct; B4 (id range) and B5
(count%32) remain independent safety asserts. Implement B4+B5 → B2 → subtype preservation
→ B3 → B1.

### Files touched by the engine work
- `gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h` — add `bar_subop` so `BAR.ARV`
  and `BAR.SYNC.DEFER_BLOCKING` remain distinguishable after decode.
- `gpu-simulator/trace-driven/trace_driven.cc` — preserve both `bar_type` and `bar_subop`
  in `OP_BAR` decode.
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h` — add per-id participant / ARV-seen /
  deferred-sync-extra-credit state.
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc` — reset the new state at allocate /
  release / deallocate, and change `warp_reaches_barrier` from UNIQUE-warp counting to
  subtype-aware split participant-set + deferred-sync-credit accounting.
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc` — gate
  `store_info_of_last_inst_at_barrier` on `bar_type != ARRIVE`, and print `bar_subop` in
  `BARDBG[issue]`.
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
  (Change 2), `m_max_barriers_per_cta` sizing + bounds (Change 3), **and the warp-exit
  drain (B6, see below)**.
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h` — declaration of the new
  `release_satisfiable_barriers()` helper (B6).
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc` — verify `BARRIER_OP` dispatch
  respects `bar_type == ARRIVE` (Change 2).

(Read-only references used for grounding: `trace_parser.cc`, `abstract_hardware_model.h`,
`subcore.cc`, `enhanced_execution_info.json`, `dynamic_trace.pb`, `instruction.proto`.)

---

## B6 — The decisive fix: drain named/counted barriers at warp-specialized exit (IMPLEMENTED)

### Correction to the earlier teardown-leak diagnosis
The "Post-deadlock follow-up" sections above concluded the teardown leak
(`shader.cc:4252 deallocate_barrier(): Assertion at_a_specific_barrier.any() == false`) was
purely a **counted-release accounting bug** solvable by adding an ARRIVE+later-SYNC extra
credit (B1/B3). That accounting fix (`m_bar_id_to_arrive_credited_warps` /
`m_bar_id_to_sync_credited_warps`, `>=` release) was implemented and is correct **for a
single barrier epoch**, but it was **not sufficient** and the earlier diagnosis was
**incomplete**. The actual residual root cause is an **exit-timing** problem, not a
per-epoch credit problem:

- FA3 is warp-specialized: producer and consumer warpgroups reach `exit` at **different**
  times. After a counted/named barrier releases once and is **re-armed for a later epoch**,
  the warp that would contribute the *closing* credit (e.g. the 13th credit of
  `id=1 count=416`, or the full-CTA quorum of `id=0 count=-1`) may have **already exited**
  the kernel. That credit can therefore never arrive, the release threshold is never met,
  and the participant bits stay recorded until CTA teardown → the assert fires.
- Evidence (BARDBG on the failing run): `[BARDBG][summary]` showed `parked=0` (no warp
  blocked) but `leaked_ids>0` (participant-only leak), and the leaked ids were exactly the
  re-armed named/counted barriers (`id=0,1,8,9,10,11`) — i.e. a release-quorum problem, not
  a still-blocked-warp problem.
- Why B1/B3 alone did not fix it: the extra-credit accounting only helps **while all
  participants are still alive**. Once a participant exits, no amount of credit re-counting
  on the *remaining* warps can reach a threshold that assumed the exited warp would arrive.

The decode + blocking rule (FINAL rule, Change 1) and B1–B5 remain correct and necessary;
B6 is the missing piece that makes the engine robust to warpgroup-staggered exit.

### The change (`barrier_set_t::warp_exit` + new helper)
`gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc`,
`gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h`:

- **`warp_exit(warp_id)`** now, in addition to `m_warp_active.reset(warp_id)`:
  - removes the exiting warp from **every** per-id set
    (`m_bar_id_to_warps`, `m_bar_id_to_arrive_credited_warps`,
    `m_bar_id_to_sync_credited_warps`) and from `m_warp_at_barrier` — an exited warp can
    never arrive at a future epoch, so leaving its bit set is what caused the leak. This
    also preserves the invariant *participant ⊆ active*.
  - calls the new helper to re-evaluate releases against the shrunken active set.
- **`release_satisfiable_barriers(cta_id)`** (new private helper): for each barrier id,
  release the whole arrived cohort when `at_a_specific_barrier.any() && at_a_specific_barrier
  == active`. This **generalizes the legacy full-CTA release** (which only the old
  `bar_count==-1` path performed) to **counted and named** barriers, so any id whose
  *remaining active* participants have all arrived drains immediately once the other
  participants exit — instead of dangling until teardown.

Semantics note: this is the same "all still-alive participants have arrived → release"
rule the legacy full-CTA `warp_exit` already used; B6 only extends it to the counted/named
ids and adds the per-id state cleanup. It does **not** weaken the normal in-flight release
paths in `warp_reaches_barrier` (B1/B3), which still fire first during normal execution.

### B6 debug signals (gated by `-bar_debug_enable`, first-occurrence rate-limited)
- `[BARDBG][exit-clear]`: which `bar_id` an exiting warp was still recorded on, plus
  `was_parked` — surfaces a genuine mis-decode (a warp exiting while parked at a *real*
  blocking SYNC) instead of silently swallowing it.
- `[BARDBG][exit-release]`: first exit-triggered release per `bar_id` — confirms the new
  path actually fires and measures how much the kernel relies on it.

### B6 verification (OnlyKernel5 = FA3 fwd, `FlashAttnFwdSm90`)
- Terminates cleanly: `*** exit detected ***`, **no** assert / signal / `-gpgpu_deadlock_detect`
  abort (12h38m wall).
- **All 40 CTA teardowns report `leaked_ids=0`** (was `leaked_ids=6`).
- **8 `[BARDBG][exit-release]`** events (ids 1,4,5,8,9,10,11); **0 `[BARDBG][exit-clear]
  was_parked=1`** ⇒ no warp wrongly parked at a blocking SYNC, confirming the FINAL decode
  rule is sound.
- Cycle accuracy vs real H100: **3.25× → 2.40×** (220,024 → 162,582 cycles);
  `inst_barrier` issue-stage stall **56.09% → 9.09%** (HW barrier ≈ 10.9%); TMA-axis stall
  **62.73% → 17.16%**, removing the inverted barrier-stall attribution documented in
  `.result/FA3_kernel_5_fwd.md`.
- (Note: this validation is on **kernel 5 (fwd)**, which hit the same teardown leak; D3's
  "kernel 10 only" scope is superseded — both fwd and bwd share the warp-specialized exit
  pattern and are fixed by B6.)

### Remaining known limitations (unchanged by B6, currently inactive)
- Register-form `(0,-1)` fallback (D1) can collide on `id=0`; not hit on the current trace
  (runtime capture present).
- `>=` release does not explicitly track epoch boundaries; safe under the in-order trace but
  could over/early-release if a future kernel reuses the same id within a single warp's
  in-flight window.
- `BAR.RED` + ARRIVE path is dead code (decode never produces `RED`).
