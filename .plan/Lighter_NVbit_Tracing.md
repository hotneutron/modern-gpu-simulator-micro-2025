# Lighter NVBit Tracing — reduce capture clock-skew so producer spin ≈ real H100 (Lighter_NVbit_Tracing.md, 2026-08-03)

Follow-up to `.plan/HW_VS_SIM_SFU.md` §26. That doc root-caused the FA3 fwd 1.56× / bwd 1.35× sim-vs-HW
cycle gap to a **trace-capture artifact**: during NVBit capture every instruction pays a heavy per-op cost,
which dilates the CONSUMER warps' wall-clock while the producer's fixed real-time `NANOSLEEP` does not
dilate → the producer over-polls the mbarrier and that inflated spin is baked into the trace the simulator
replays. §26 chose the direction: **make the capture itself lighter (attack the skew at the source)**,
sub-option **(a) per-opcode-class record** — skip the expensive per-lane gather for control/sync ops that
don't need it.

This doc is the concrete, code-grounded plan for (a), plus the correctness/HW guardrails required before
trusting it.

## 0. Root cause (code-confirmed, from HW_VS_SIM_SFU §20/§24/§26)
- Every instrumented instruction runs the heavy gather in `instrument_inst`
  (util/tracer_nvbit/tracer_tool/inject_funcs.cu L74-140): up to four
  `for(int tid=0;tid<32;tid++) __shfl_sync(...)` loops (addresses / reg_val_1/2/3) plus a
  several-hundred-byte `inst_trace_t` channel push (inject_funcs.cu L158-163).
- The producer's `NANOSLEEP 0x989680` is a **real-time** HW sleep; instrumentation does NOT dilate it, but
  it DOES dilate consumers → producer over-polls **48× (fwd) / 86× (bwd)** vs HW (§24). HW real spin
  (Σ NANOSLEEP-exec): **fwd 8,461 / bwd 7,860**; trace replayed poll: **fwd 407,234 / bwd 678,768**.
- Goal: cut the per-op cost on the consumer-heavy path so the consumer/producer wall-clock ratio moves
  toward HW and the recorded producer spin approaches 8,461 / 7,860 — WITHOUT any post-hoc trimming and
  WITHOUT changing the trace schema or anything downstream sees for non-spin ops.

## 1. Two independent audits done (read-only) — both must be respected

### 1a. Post-processing (host channel → protobuf trace) — what each inst_trace_t field feeds
`inst_trace_t` defined common.h L40-63; host drain in `recv_thread_fun` (tracer_tool.cu L2738-3018).
- (A) always needed: cta_id_x/y/z, warpid_tb, vpc, unique_function_id, active_mask, predicate_mask,
  num_of_injects, per_operand_type, mem_type.
- (B) memory/TMA only (mem_type != NONE): ureg_desc_id/value/value_hi, width, addrs_or_reg_val_0[] address
  payload. Consumed by TMA debug + address serialization (tracer_tool.cu L2918-2978).
- (C) sync/bar only: the single value addrs_or_reg_val_0[first_lane] — for SYNCS.EXCH/ARRIVE.TRANS64/PHASECHK
  it is barrier_addr (op-1 MREF, mem_type!=NONE) or semantic_raw (op-2, mem_type==NONE) (tracer_tool.cu
  L2821-2853); for BAR.* it is id/count. Only [first_lane] is read, never the other 31 lanes.
- (D) effectively unused in the protobuf trace (only optional TMA debug dumps): reg_val_2, reg_val_3,
  reg_id, warpid_sm, sm_id; reg_val_1 only for a TMA producer-candidate branch.

### 1b. Simulator trace reader (protobuf → sim warp_inst_t) — what the SIM consumes
Two stages: parse_from_pb (trace_parser.cc L311) → parse_from_trace_struct (trace_driven.cc L236).
Sim MANDATORY dynamic-trace fields per spin opcode:
- pc, function_unique_id, active&predicate mask (AND-ed at trace_parser.cc L314): required for all three.
- addresses[] (32-lane): NOT used by NANOSLEEP / BRA / PHASECHK.
- sync{valid, barrier_addr, semantic_raw}: REQUIRED for PHASECHK/TRYWAIT (OP_SYNCS, MBARRIER_OP).
  valid=true else poll skipped (sm.cc L2169); barrier_addr = mbarrier map key (sm.cc L2113);
  semantic_raw = phaseParity bit31 (sm.cc L122-127). n/a for NANOSLEEP/BRA.
- bar_runtime{id,count}: not used by any of the three.

Critical facts that make (a) SAFE:
- Control bits (stall/yield/read_bar/write_bar/wait_mask) are NOT in the dynamic trace — they come from the
  STATIC enhanced SASS keyed by (function_id, pc), attached via set_extra_trace_instruction_info
  (trace_driven.cc L125; decoder control_bits.cc L35-41). So the §19 wait_barrier and NANOSLEEP latency
  (config knob) are UNAFFECTED by the gather.
- sync_kind (PHASECHK vs TRYWAIT vs ARRIVE) is resolved statically (sidecar DB lookup_sync_site_metadata),
  not from the dynamic trace.
- The sync operand values (barrier_addr, semantic_raw) are per-instruction scalars captured from ONE lane —
  exactly what a light path can still produce; the 31 other lanes are never read.

⇒ Dropping the 32-lane gather is sufficient for the sim to replay NANOSLEEP/BRA/PHASECHK correctly, provided
(i) the effective active&predicate mask stays truthful, and (ii) for PHASECHK/TRYWAIT the sync block
(valid + barrier_addr + semantic_raw) is still emitted (the single first-lane value per inject).

## 2. Design principle
Do NOT change the trace schema or what post-processing / the sim sees for any op. Only make the DEVICE-SIDE
production of the identical record cheaper for opcodes whose consumers don't need the full 32-lane payload:
- NANOSLEEP, BRA: need NO operand payload → skip all gather loops, leave addrs_or_reg_val_0[]/reg_val_*
  unwritten.
- SYNCS.PHASECHK / TRYWAIT: need only addrs_or_reg_val_0[first_lane] → gather ONLY that one lane's value with
  a single __shfl_sync(mask, val, first_lane) (still collective/legal, O(1) not O(32)) and write it at
  [first_lane]. Category-(A) fields filled identically. Keep BOTH the barrier-addr inject (mem_type!=NONE)
  and the semantic inject (mem_type==NONE) so the host builds the full sync block.

## 3. Implementation

### 3a. New device fn instrument_inst_light (additive) — inject_funcs.cu
- Keep the collective __ballot_sync for active_mask/predicate_mask (one op, cheap, correct).
- Keep the pstop_report early-exit and BOTH atomic counters (inject_funcs.cu L59-64,161-162) IDENTICALLY
  (so total_dynamic_instr / work-invariance is unchanged).
- Fill all category-(A) fields exactly as instrument_inst.
- Sync single-operand case: compute first_laneid, then v = __shfl_sync(active_mask, operand_val, first_laneid)
  and store ma.addrs_or_reg_val_0[first_laneid]=v; set mem_type / per_operand_type exactly as the heavy path
  would for that inject (so the host's callback_index/mem_type branch picks the right barrier-vs-semantic slot).
- Push the same sizeof(inst_trace_t) record (schema-identical). Existing instrument_inst left untouched.

### 3b. Host routing — tracer_tool.cu instrumentation loop (~L1957-1959)
- New default-off env flag TRACER_LIGHT_SPIN (plumbed like warp_lifetime_only / enable_tma_desc).
- Conservative allowlist ONLY: NANOSLEEP*, BRA*, and SYNCS.PHASECHK* / SYNCS only when the site's injects
  reduce to the sync scalar case (no MREF address payload beyond the single barrier operand, not a
  TMA-consumer opcode). Everything else → existing full instrument_inst path, UNCHANGED.

### 3c. Explicit non-goals / never routed to light path
Memory ops, TMA-consumer opcodes (UTMALDG/UTMASTG/UBLKCP…), capture_tma_param_base, factcheck_tma_descriptor,
BAR.*, tensor ops, any reg/address-carrying op → full path. So TMA post-processing, sync barrier_addr, bar
id/count, and address serialization are byte-for-byte unaffected.

## 4. Guardrails — run BEFORE trusting any cycle number

### Guard 0 (from the sim-reader audit): schema/field parity self-check
On a tiny kernel, assert the light path emits, for PHASECHK/TRYWAIT, a valid sync block with non-zero
barrier_addr and correct semantic_raw (phaseParity bit) — the sim hard-requires these (sm.cc L2169). If the
light path ever emits sync.valid=false or zeroed semantic_raw for a real poll, the sim silently mis-models
the mbarrier → must fail here.

### Guard 1: correctness vs original trace (offline protobuf diff)
Capture the same kernel twice (flag off = baseline, flag on = light). Assert per (cta,warp):
- identical instruction PC sequence + active_mask + predicate_mask + function_unique_id;
- HGMMA / MUFU / TMA / all memory ops byte-identical (addresses, sync.barrier_addr, bar id/count);
- PHASECHK/TRYWAIT sync.semantic_raw / barrier_addr identical;
- ONLY the spin-iteration COUNT may differ (the intended effect).
Any mismatch outside spin-op count ⇒ stop. Reuse the §23/§24 pure-python protobuf parser.

### Guard 2: closer to HW (over-poll reduction)
Re-run the §24 per-loop NANOSLEEP/TRYWAIT counting on the light trace; compare per loop-PC to HW targets
(fwd Σ 8,461 / bwd Σ 7,860). Success = producer spin moved substantially down from 407K / 679K toward HW.
Emit the per-loop table.

### Guard 3: simulate (only after 0/1/2 pass)
Run sim OnlyKernel5 / OnlyKernel10 on the light trace; compare cycles vs HW (fwd 67,696 / bwd 132,901).

## 5. Open risks
- Residual dilation: a cheaper record still costs something; light capture may only PARTIALLY close the
  48×/86× gap. Guard 2 quantifies it. Fallbacks: §26(b) bigger-channel/batched push, or the parked §25
  offline trim.
- Re-capture required: fresh NVBit run (12h-class). De-risk on a small kernel with Guards 0/1/2 first.
- fwd loop 0xacc0 under-spin (0.48×, §24): lighter capture reduces over-spin; it will NOT fix an under-spun
  loop. Separate under-model item.
  
## 5b. Cost-model reality check (2026-08-03) — WHY (a) is expected to give LIMITED gain
1. The channel push memcpy is the fixed dominant cost and (a) does NOT reduce it. ChannelDev::push
   (channel.hpp L60-94) does atomicAdd(head)+memcpy(nbytes)+atomicAdd(tail); flush() busy-waits on a host
   doorbell round-trip (L96-120). Light path keeps the record schema-identical (~700B), so memcpy +
   atomic contention + flush backpressure are UNCHANGED. (a) only removes the shfl gather (minority term).
2. NANOSLEEP and BRA do NOT run the 32-lane gather today. Gather loops fire only for REGULAR/UNIFORM/
   STANDARD_MEM injects (inject_funcs.cu L74-140); @!P0 NANOSLEEP is PREDICATE, BRA is NO_REGS → both skip
   it. So (a) meaningfully lightens ONLY SYNCS.PHASECHK.
3. The dilation is on the CONSUMER path (HGMMA/LDGSTS/MUFU), which (a) intentionally does NOT touch (they
   need full payload). (a) cannot lighten the ops whose slowdown causes the skew (§20).
=> (a) is a CHEAP EXPERIMENT that likely moves SYNCS.PHASECHK cost only; NOT expected to close 48×/86× on
its own. Decision: try (a) once and MEASURE (Guard 2). If recorded producer spin does not move toward HW
(8,461 / 7,860), fall back to the offline full-trace + per-loop trim (§25 tool) and harden §26 fragilities.
Bigger levers if pursued: §26(b) shrink per-op record (cuts memcpy) and/or enlarge channel/batch push.

## 6. Status
- Plan + both audits + cost-model reality check complete.
- Decision: implement (a) as a measurement experiment (Guards 0/1/2), then decide trim-fallback by Guard 2.
- BLOCKED on implementation: auto_review_reviewer service rejecting all Write/Edit (infra config error).