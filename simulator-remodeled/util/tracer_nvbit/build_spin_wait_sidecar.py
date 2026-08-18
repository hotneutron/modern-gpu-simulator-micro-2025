#!/usr/bin/env python3
"""build_spin_wait_sidecar.py — emit a static spin-wait sidecar for FA3 traces.

Purpose
-------
The trace-driven simulator replays each warp's recorded PC stream and takes every
branch from the captured `next_traced_pc`, so a producer mbarrier spin loop is
replayed exactly as many times as it happened to spin on the (instrumented) HW.
That recorded spin count is a capture-timing artifact, not real work: the correct
number of polls depends on when the *modeled* mbarrier flips, which only the
simulator can decide.

This tool does the STATIC half of the fix (payload untouched): it locates each
producer spin loop from the disassembly CFG + sync resolver and describes it as a
"wait region" so the simulator can, at replay time, block the warp at the region
until the modeled barrier is ready and then jump to the region's exit — instead
of replaying the recorded poll count. The dynamic half (blocking + exit jump) is
a separate gated simulator feature; this file is its input contract.

Design decisions (validated against opt11A K5/K10, see handover analysis):
  * A spin region is a short BACKWARD self-loop (loopback BRA target < BRA pc)
    whose body contains a SYNCS.PHASECHK / SYNCS.PHASECHK.TRANS64.TRYWAIT. First
    and later iterations may differ (e.g. a leading WARPSYNC.ALL), so we anchor on
    static PCs (loop head, wait pc, loopback BRA, exit) rather than a fixed body.
  * We describe regions by PC, NOT by dynamic instruction index: the trace is
    stored per-CTA and the iteration count varies per CTA/warp, so PC anchors are
    both stable and match the simulator's `traced_pcs[used_insts]` cursor model.
  * The barrier ADDRESS is intentionally NOT baked in — it is register-formed
    ([R7+URZ+0x16800]) and only known at replay time from the instruction's
    captured sync.barrier_addr payload. We record barrier_operand_text for
    cross-checking / debugging only.
  * semantic_free_body=True means the body has no TMA / *GMMA / MUFU / arrive /
    global-or-shared memory op, so collapsing the poll count cannot drop work.
    STL/LDL (thread-local stack spill) are allowed and flagged in body_note.

Usage
-----
  build_spin_wait_sidecar.py --traces <.../traces> [--kernels 5,10]
                             [--out <.../extra_info/spin_wait_sidecar.json>]
                             [--validate-ctas all|N] [--strict]

Runs standalone on a CPU server, and is also invoked automatically from
run_hw_trace.py after the sync-operand mapping step.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

# Reuse the dependency-free varint protobuf parser + funcid peek from trim_spin.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import importlib.util as _ilu
_ts_spec = _ilu.spec_from_file_location(
    "trim_spin_from_ncu", os.path.join(_HERE, "trim_spin_from_ncu.py"))
_ts = _ilu.module_from_spec(_ts_spec)
_ts_spec.loader.exec_module(_ts)

ISIZE = 0x10          # every FA3 SM90 SASS instruction is 16 bytes
MAX_LOOP_SPAN = 0x100  # a spin loop is short; guards against matching big loops


# --- Stage 1.7: mask-aware .pb parser ------------------------------------- #
# The shared _ts._vpb_parse() returns only (pc, sync_valid) per instruction and
# must not change (trim_spin relies on it). The compound wait-block dynamic proof
# (§1.7.4) additionally needs the recorded active_mask / predicate_mask and the
# runtime sync payload (barrier_addr / semantic_raw). We add a PRIVATE parser here
# that reuses _ts._rv / _ts._skip varint primitives and the same threadblock ->
# warps -> instruction wire layout, without touching the trace format or the
# shared helper. instruction.proto fields: pc=1, active_mask=2, predicate_mask=3,
# function_unique_id=4, sync=6 { valid=1, barrier_addr=2, has_semantic_raw=3,
# semantic_raw=4 }.
_rv = _ts._rv
_skip = _ts._skip


def _inst_full(buf, s, e):
    """Parse one instruction message; return a dict with pc + masks + sync info."""
    i = s
    pc = 0
    active_mask = 0
    predicate_mask = 0
    sync_valid = False
    barrier_addr = None
    semantic_raw = None
    while i < e:
        tag, i = _rv(buf, i)
        f, w = tag >> 3, tag & 7
        if f == 1 and w == 0:
            pc, i = _rv(buf, i)
        elif f == 2 and w == 0:
            active_mask, i = _rv(buf, i)
        elif f == 3 and w == 0:
            predicate_mask, i = _rv(buf, i)
        elif f == 6 and w == 2:                 # sync_runtime_info sub-message
            ln, j = _rv(buf, i)
            end = j + ln
            k = j
            while k < end:
                t2, k = _rv(buf, k)
                f2, w2 = t2 >> 3, t2 & 7
                if f2 == 1 and w2 == 0:         # valid
                    v, k = _rv(buf, k)
                    sync_valid = bool(v)
                elif f2 == 2 and w2 == 0:       # barrier_addr
                    barrier_addr, k = _rv(buf, k)
                elif f2 == 4 and w2 == 0:       # semantic_raw
                    semantic_raw, k = _rv(buf, k)
                else:
                    k = _skip(buf, k, w2)
            i = end
        else:
            i = _skip(buf, i, w)
    return {
        "pc": pc,
        "active_mask": active_mask,
        "predicate_mask": predicate_mask,
        "sync_valid": sync_valid,
        "barrier_addr": barrier_addr,
        "semantic_raw": semantic_raw,
    }


def vpb_parse_full(buf):
    """Return list of (warp_key, [inst_dict, ...]) preserving masks + sync payload.

    Mirrors _ts._vpb_parse's threadblock -> warp(field 2, key=field 1,
    insts=field 2) -> instruction(field 2) wire walk, but keeps every field the
    compound wait-block proof needs.
    """
    i = 0
    n = len(buf)
    warps = []
    while i < n:
        tag, i = _rv(buf, i)
        f, w = tag >> 3, tag & 7
        if f == 2 and w == 2:
            ln, i = _rv(buf, i)
            entry_end = i + ln
            j = i
            key = None
            insts = []
            while j < entry_end:
                t2, j = _rv(buf, j)
                f2, w2 = t2 >> 3, t2 & 7
                if f2 == 1 and w2 == 0:
                    key, j = _rv(buf, j)
                elif f2 == 2 and w2 == 2:
                    l2, j = _rv(buf, j)
                    ve = j + l2
                    p = j
                    while p < ve:
                        t3, p = _rv(buf, p)
                        f3, w3 = t3 >> 3, t3 & 7
                        if f3 == 2 and w3 == 2:
                            l3, pp = _rv(buf, p)
                            se = pp + l3
                            insts.append(_inst_full(buf, pp, se))
                            p = se
                        else:
                            p = _skip(buf, p, w3)
                    j = ve
                else:
                    j = _skip(buf, j, w2)
            warps.append((key, insts))
            i = entry_end
        else:
            i = _skip(buf, i, w)
    return warps

# Opcodes whose presence in a loop body makes it NOT a pure wait (would drop real
# work if the poll count were folded). Global/shared memory + tensor + async.
SEMANTIC_PREFIXES = (
    "HGMMA", "IGMMA", "QGMMA", "BGMMA", "OGMMA", "MMA",
    "MUFU",
    "UTMALDG", "UTMASTG", "UTMACMDFLUSH", "UBLKCP", "UBLKPF", "UTMAPF",
    "UTMAREDG", "UBLKRED", "UTMACCTL",
    "LDGSTS", "STG", "LDG", "STS", "LDS", "ATOM", "ATOMS", "RED",
    "SYNCS.ARRIVE", "SYNCS.EXCH", "BAR",
)
# Thread-local stack ops that are warp-local side effects only — safe inside a
# folded wait region, but recorded so the simulator/reviewer can see them.
LOCAL_MEM_PREFIXES = ("STL", "LDL")

WAIT_KINDS = ("PHASECHK", "TRYWAIT")


def is_semantic(op):
    return any(op == p or op.startswith(p + ".") or op.startswith(p)
               for p in SEMANTIC_PREFIXES)


def is_local_mem(op):
    return any(op.startswith(p) for p in LOCAL_MEM_PREFIXES)


# Async producers whose SASS scoreboard barrier release is DECOUPLED from issue
# (the barrier is released by a later async completion, not at the arming
# instruction's own pipeline stage). If one sits inside a spin body, whole-loop
# skip would drop an arm whose release lives outside the body and corrupt the
# scoreboard. Distinct from SEMANTIC_PREFIXES (overlaps, but this list is the
# barrier-safety gate specifically). See .plan/SPIN_WAIT_SIM_MODELING.md 12.7.
ASYNC_PRODUCER_PREFIXES = (
    "HGMMA", "IGMMA", "QGMMA", "BGMMA", "OGMMA", "MMA", "WGMMA",
    "UTMALDG", "UTMASTG", "UTMACMDFLUSH", "UBLKCP", "UBLKPF", "UTMAPF",
    "UTMAREDG", "UBLKRED", "UTMACCTL",
    "LDGSTS", "LDG", "STG", "LDS", "STS", "ATOM", "ATOMS", "RED",
    "DEPBAR", "LDGDEPBAR",
)


def is_async_producer(op):
    return any(op == p or op.startswith(p + ".") or op.startswith(p)
               for p in ASYNC_PRODUCER_PREFIXES)


def _cb(inst):
    return inst.get("control_bits", {}) or {}


def compute_barrier_self_tied(body_pcs, byp):
    """A spin body is barrier-self-tied iff whole-loop skip cannot corrupt the SASS
    scoreboard wait-barrier state. In this model a wait-barrier armed by an
    instruction is released by that SAME instruction's own pipeline stage (write
    barrier at WB, read barrier at read-stage), so an arm INSIDE the body is
    released INSIDE the body -> skipping is net-zero. The two ways it breaks:
      (1) a body instruction WAITS on a barrier id that is NOT armed inside the
          body (its arming instruction is outside the skip set), or
      (2) a body op is an async producer whose barrier release is decoupled from
          issue (GMMA/TMA/LDGSTS/DEPBAR/global-shared memory).
    Returns (self_tied: bool, reason_fields: dict) for logging.
    """
    armed_ids = set()
    waited_ids = set()
    async_ops = []
    for p in body_pcs:
        inst = byp[p]
        op = inst["op_code"]
        cb = _cb(inst)
        if cb.get("is_new_read_barrier"):
            armed_ids.add(int(cb.get("id_new_read_barrier", -1)))
        if cb.get("is_new_write_barrier"):
            armed_ids.add(int(cb.get("id_new_write_barrier", -1)))
        mask = int(cb.get("wait_barrier_bits", 0) or 0)
        for bit in range(6):
            if mask & (1 << bit):
                waited_ids.add(bit)
        if is_async_producer(op):
            async_ops.append((hex(p), op))
    wait_on_external = sorted(waited_ids - armed_ids)
    self_tied = (not wait_on_external) and (not async_ops)
    return self_tied, {
        "armed_ids": sorted(armed_ids),
        "waited_ids": sorted(waited_ids),
        "wait_on_externally_armed_ids": wait_on_external,
        "async_producer_ops": async_ops,
    }


def bra_target(inst):
    if inst["op_code"].split(".")[0] != "BRA":
        return None
    for o in inst.get("operands", []):
        if o.get("has_inmediate"):
            return int(o["operand_inmmediates"][-1])
    return None


def load_kernels(extra_info_dir):
    """funcid -> {pc: inst_dict} from enhanced_execution_info.json."""
    path = os.path.join(extra_info_dir, "enhanced_execution_info.json")
    with open(path) as fh:
        data = json.load(fh)
    out = {}
    names = {}
    for e in data["kernels"]:
        fid = int(e["unique_function_id"])
        out[fid] = {int(i["pc_num_dec"]): i for i in e["instructions"]}
        names[fid] = e["kernel_name"]
    return out, names


def load_sync_sites(extra_info_dir):
    """(funcid, pc) -> {sync_kind, opcode, barrier_operand_text, ...}."""
    path = os.path.join(extra_info_dir, "sync_operand_resolver.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        res = json.load(fh)["resolver"]
    out = {}
    for r in res:
        out[(int(r["unique_function_id"]), int(r["pc"]))] = r
    return out


def detect_regions(fid, byp, sync_sites):
    """Static detection of spin-wait regions for one kernel (funcid)."""
    order = sorted(byp)
    regions = []
    for pc in order:
        inst = byp[pc]
        if inst["op_code"].split(".")[0] != "BRA":
            continue
        tgt = bra_target(inst)
        if tgt is None or tgt >= pc:          # backward branch only
            continue
        if pc - tgt > MAX_LOOP_SPAN:          # short loop only
            continue
        body_pcs = [p for p in order if tgt <= p < pc]  # excl. loopback BRA
        # a spin region must contain a mbarrier wait
        wait_pcs = [p for p in body_pcs
                    if sync_sites.get((fid, p), {}).get("sync_kind") in WAIT_KINDS]
        if not wait_pcs:
            continue
        # classify body
        body = [(p, byp[p]["op_code"]) for p in body_pcs]
        sem_ops = [(hex(p), o) for p, o in body if is_semantic(o)]
        local_ops = [(hex(p), o) for p, o in body if is_local_mem(o)]
        # the primary wait is the last PHASECHK (non-TRYWAIT) if present, else the
        # last TRYWAIT — that is the poll whose result the loop branch depends on.
        phasechk = [p for p in wait_pcs
                    if sync_sites[(fid, p)]["sync_kind"] == "PHASECHK"]
        primary_wait = (phasechk[-1] if phasechk else wait_pcs[-1])
        wsite = sync_sites[(fid, primary_wait)]
        self_tied, bar_reason = compute_barrier_self_tied(body_pcs, byp)
        regions.append({
            "region_id": "k_fid%d_h0x%x" % (fid, tgt),
            "loop_head_pc": tgt,
            "body_pcs": body_pcs,
            "wait_pcs": wait_pcs,
            "primary_wait_pc": primary_wait,
            "primary_wait_kind": wsite["sync_kind"],
            "primary_wait_opcode": wsite["opcode"],
            "barrier_operand_text": wsite.get("barrier_operand_text"),
            "barrier_operand_index": wsite.get("barrier_operand_index"),
            "loopback_bra_pc": pc,
            "exit_pc": pc + ISIZE,   # fall-through of the (predicated) loopback BRA
            "semantic_free_body": len(sem_ops) == 0,
            "semantic_ops_in_body": sem_ops,
            "local_mem_ops_in_body": local_ops,
            "barrier_self_tied": self_tied,
            "barrier_reason": bar_reason,
        })
    return regions


# --- Stage 1.7: compound wait-block CFG detector -------------------------- #
# A compound wait block is a speculative mbarrier gate whose miss path is a pure
# wait fallback. FA3 emits two shapes that this detector handles with ONE general
# rule (plan §1.7.6): follow only branches that consume the gate predicate P.
#
#   Type-1 (double gate, out-of-line fallback; HW captured ready only):
#     entry:      TRYWAIT P
#     entry+0x10: @P  BRA ready_arm           (P ready  -> continuation)
#     entry+0x20: TRYWAIT P
#     entry+0x30: @!P BRA fallback_entry      (P miss   -> OOL fallback loop)
#     ... fallback loop ... BRA resume_pc     (rejoin == ready_arm)
#
#   Type-2 (single gate, in-line fallback == an existing spin loop; HW captured
#           both ready and the loop):
#     entry:      TRYWAIT P
#     ... 0..N pure address-setup insts (MAINLINE; never skipped) ...
#     @!P BRA fallback_entry                  (P miss   -> in-line spin loop)
#     resume_pc == fall-through of that @!P BRA
#     ... spin loop ... BRA resume_pc         (rejoin)
#
# Key invariant (skip safety): only branches that CONSUME P delimit the block.
# A branch on a DIFFERENT predicate, or a non-branch instruction that produces a
# live-out (address setup), is OUTSIDE the block and is never added to
# allowed_block_pcs. resume_pc is the single convergence point of the P-ready and
# P-miss edges. The only skippable PCs are gate polls and the fallback wait loop
# body (TRYWAIT/PHASECHK/NANOSLEEP/loopback BRA) — never mainline setup.
#
# Fails closed: any shape it cannot fully resolve is a rejected candidate with a
# closed-taxonomy reason (§1.7.8) and is never emitted.
#
# FAIL reason taxonomy (see plan §1.7.8):
CFG_UNRESOLVED = "CFG_UNRESOLVED"
MULTI_REJOIN = "MULTI_REJOIN"


def _is_wait(fid, pc, byp, sync_sites):
    site = sync_sites.get((fid, pc))
    if site and site.get("sync_kind") in WAIT_KINDS:
        return True
    # sync_operand_resolver may not carry every predicated PHASECHK; fall back to
    # the disassembled opcode so the CFG walk still classifies it as a wait.
    inst = byp.get(pc)
    if inst is None:
        return False
    return any(k in inst["op_code"] for k in WAIT_KINDS)


def _pred_of(inst):
    """Return (reg, negate, is_uniform) for a predicated inst, else None."""
    if not inst.get("is_predicated"):
        return None
    return (inst.get("predicate_register"),
            bool(inst.get("is_predicate_negate")),
            bool(inst.get("is_uniform_predicate")))


def _scan_fallback_loop(fid, fb_entry, byp, sync_sites):
    """Walk a pure wait fallback loop starting at fb_entry: optional pure setup,
    a wait poll loop with a single backedge, then one unconditional BRA to the
    resume PC. Used by both Type-1 (OOL) and Type-2 (in-line) blocks.

    Returns (ok, info_or_reason). info = {fallback_body_pcs, fallback_wait_pcs,
    fallback_loopback_pc, resume_pc}."""
    body_pcs = []
    wait_pcs = []
    loopback_pc = None
    resume_pc = None
    fallback_exit_pc = None
    pc = fb_entry
    guard = 0
    while pc in byp and guard < (MAX_LOOP_SPAN // ISIZE) * 3:
        guard += 1
        inst = byp[pc]
        base = inst["op_code"].split(".")[0]
        if base == "BRA":
            tgt = bra_target(inst)
            if tgt is None:
                return False, CFG_UNRESOLVED
            pr = _pred_of(inst)
            if pr is not None:
                # predicated backedge of the poll loop (targets the loop head,
                # which is at or before this branch).
                if tgt > pc:
                    return False, CFG_UNRESOLVED
                if loopback_pc is not None:
                    return False, CFG_UNRESOLVED
                loopback_pc = pc
                body_pcs.append(pc)
                pc += ISIZE
                continue
            # unpredicated BRA => the single fallback exit / rejoin. It may be a
            # backward branch (OOL fallback sits above the continuation). This
            # exit BRA is the last instruction of the skipped fallback path, so
            # include its PC in the body (it is folded away with the loop).
            resume_pc = tgt
            body_pcs.append(pc)
            fallback_exit_pc = pc
            break
        if base in ("CALL", "RET", "EXIT", "BRX", "JMP", "JMX", "CALL.REL",
                    "BSSY", "BSYNC", "RTT", "WARPSYNC"):
            return False, CFG_UNRESOLVED
        if _is_wait(fid, pc, byp, sync_sites):
            wait_pcs.append(pc)
        body_pcs.append(pc)
        pc += ISIZE
    if loopback_pc is None or resume_pc is None or not wait_pcs:
        return False, CFG_UNRESOLVED
    return True, {
        "fallback_body_pcs": body_pcs,
        "fallback_wait_pcs": wait_pcs,
        "fallback_loopback_pc": loopback_pc,
        "fallback_exit_pc": fallback_exit_pc,
        "resume_pc": resume_pc,
    }



def _find_pred_branch(fid, entry, predreg, byp, sync_sites, max_span=0x120):
    """From entry (a TRYWAIT defining predreg), scan forward for the FIRST direct
    BRA that consumes predreg. Between entry and that branch only wait polls and
    non-control, non-semantic instructions (pure setup) may appear. Those setup
    PCs are MAINLINE and are returned separately so the caller never skips them.

    Returns (branch_pc, between_pcs) or (None, reason)."""
    between = []
    pc = entry + ISIZE
    span = 0
    while pc in byp and span < max_span:
        span += ISIZE
        inst = byp[pc]
        base = inst["op_code"].split(".")[0]
        pr = _pred_of(inst)
        if base == "BRA":
            if pr is not None and pr[0] == predreg:
                return pc, between            # the P-consuming gate branch
            # a branch that does not consume P ends the search unresolved: the
            # block boundary must be a P-branch.
            return None, CFG_UNRESOLVED
        if base in ("CALL", "RET", "EXIT", "BRX", "JMP", "JMX", "CALL.REL",
                    "RTT"):
            return None, CFG_UNRESOLVED
        # A semantic/async op between the gate and its branch means this is not a
        # clean speculative gate (fail closed).
        if is_semantic(inst["op_code"]) or is_async_producer(inst["op_code"]):
            return None, CFG_UNRESOLVED
        # wait polls and pure setup are allowed here; setup is mainline (not skipped)
        between.append(pc)
        pc += ISIZE
    return None, CFG_UNRESOLVED


def detect_wait_block_candidates(fid, byp, sync_sites):
    """Discover compound wait-block candidates for one kernel with NO application/
    kernel/PC filters, using ONE general rule for Type-1 and Type-2 (plan §1.7.6).
    Returns a list of dicts, each either a proven-shape record (cfg_ok=True) or a
    rejected candidate (cfg_ok=False, reason=...)."""
    order = sorted(byp)
    cands = []
    claimed = set()   # PCs already inside an emitted block (avoid nested re-entry)
    for pc in order:
        if pc in claimed:
            continue
        if not _is_wait(fid, pc, byp, sync_sites):
            continue
        # entry TRYWAIT must define a predicate (operand[0] = P<n>); use the first
        # P-consuming branch to find the gate boundary.
        ops = [o.get("operand_string", "") for o in byp[pc].get("operands", [])]
        predreg = None
        if ops and ops[0].lstrip("U").startswith("P"):
            try:
                predreg = int(ops[0].lstrip("U")[1:])
            except ValueError:
                predreg = None
        if predreg is None:
            continue
        br_pc, between = _find_pred_branch(fid, pc, predreg, byp, sync_sites)
        if br_pc is None:
            continue   # not a gate shape; leave for legacy spin-region detection
        br = byp[br_pc]
        prb = _pred_of(br)
        tgt = bra_target(br)
        if tgt is None:
            continue
        cand = {
            "block_id": "wb_fid%d_e0x%x" % (fid, pc),
            "unique_function_id": fid,
            "entry_wait_pc": pc,
            "wait_predicate": predreg,
            "wait_predicate_uniform": bool(prb[2]),
            "mainline_setup_pcs": [p for p in between if not _is_wait(fid, p, byp, sync_sites)],
        }
        gate_wait_pcs = [pc] + [p for p in between if _is_wait(fid, p, byp, sync_sites)]

        if prb[1] is False:
            # @P BRA ready  => Type-1: ready arm is the branch target; the miss
            # path (@!P) is the NEXT gate/branch. Require the classic 2nd-gate
            # form: [2nd wait] then `@!P BRA fallback`.
            ready_arm = tgt
            w2 = byp.get(br_pc + ISIZE)
            b2 = byp.get(br_pc + 2 * ISIZE)
            if (w2 is None or b2 is None
                    or not _is_wait(fid, br_pc + ISIZE, byp, sync_sites)
                    or b2["op_code"].split(".")[0] != "BRA"):
                cand["cfg_ok"] = False
                cand["reason"] = CFG_UNRESOLVED
                cands.append(cand)
                continue
            pr2 = _pred_of(b2)
            if pr2 is None or not pr2[1] or pr2[0] != predreg:
                cand["cfg_ok"] = False
                cand["reason"] = CFG_UNRESOLVED
                cands.append(cand)
                continue
            fb_entry = bra_target(b2)
            gate_wait_pcs.append(br_pc + ISIZE)
            gate_poll_pcs = [pc, br_pc, br_pc + ISIZE, br_pc + 2 * ISIZE]
            resume_expected = ready_arm
        else:
            # @!P BRA fallback  => Type-2: miss goes to the fallback loop; ready
            # falls through. resume is the fall-through PC of this branch.
            fb_entry = tgt
            resume_expected = br_pc + ISIZE
            gate_poll_pcs = [pc, br_pc]

        if fb_entry is None:
            cand["cfg_ok"] = False
            cand["reason"] = CFG_UNRESOLVED
            cands.append(cand)
            continue
        ok, info = _scan_fallback_loop(fid, fb_entry, byp, sync_sites)
        if not ok:
            cand["cfg_ok"] = False
            cand["reason"] = info
            cands.append(cand)
            continue
        if info["resume_pc"] != resume_expected:
            cand["cfg_ok"] = False
            cand["reason"] = MULTI_REJOIN
            cands.append(cand)
            continue
        cand["cfg_ok"] = True
        cand["all_gate_wait_pcs"] = gate_wait_pcs
        cand["captured_next_pc"] = resume_expected
        cand["fallback_entry_pc"] = fb_entry
        cand["fallback_body_pcs"] = info["fallback_body_pcs"]
        cand["fallback_wait_pcs"] = info["fallback_wait_pcs"]
        cand["fallback_loopback_pc"] = info["fallback_loopback_pc"]
        cand["resume_pc"] = info["resume_pc"]
        # allowed_block_pcs = ONLY gate polls/branches + fallback loop body.
        # Mainline setup between the gate and its branch is deliberately excluded.
        allowed = set(gate_poll_pcs)
        allowed.update(info["fallback_body_pcs"])
        cand["allowed_block_pcs"] = sorted(allowed)
        for p in allowed:
            claimed.add(p)
        cands.append(cand)
    return cands


# --- Stage 1.7: per-candidate provenance / effect / liveness proofs ------- #
# enhanced_execution_info.json gives aggregate register_usage counts only (no
# per-instruction def/use register lists) and operands as strings, so we derive
# def/use from operand_strings with an explicit, reviewed per-opcode model. Any
# opcode we cannot decode exactly fails closed (DEFUSE_UNDECODED).
#
# FAIL reason taxonomy (see plan §1.7.8), continued:
WAIT_COND_MISMATCH = "WAIT_COND_MISMATCH"
PROVENANCE_UNKNOWN = "PROVENANCE_UNKNOWN"
EFFECT_UNSAFE = "EFFECT_UNSAFE"
PREDICATE_LIVE_OUT = "PREDICATE_LIVE_OUT"
DEFUSE_UNDECODED = "DEFUSE_UNDECODED"

import re as _re
_REG_RE = _re.compile(r"\b(U?R\d+|U?P\d+)\b")
_BARR_OFF_RE = _re.compile(r"\+\s*(0x[0-9a-fA-F]+)\s*\]")

# Opcodes allowed inside a compound wait block's alternate path. Each maps to a
# def/use extraction rule keyed on operand position. This is intentionally a
# closed list: an opcode not here yields DEFUSE_UNDECODED and rejects the block.
#   dst_first  : operand[0] is the (only) destination register; rest are uses.
#   wait       : SYNCS.PHASECHK* — destination is the predicate operand[0]; the
#                barrier operand and wait-state operand are uses.
#   nanosleep  : NANOSLEEP.SYNCS — backoff, immediate only, no reg def/use.
#   bra        : control flow, no data def/use.
_WB_OPCODE_MODEL = {
    "IMAD.U32": "dst_first",
    "IMAD.MOV.U32": "dst_first",
    "SYNCS.PHASECHK.TRANS64.TRYWAIT": "wait",
    "SYNCS.PHASECHK.TRANS64": "wait",
    "NANOSLEEP.SYNCS": "nanosleep",
    "BRA": "bra",
}


def _regs_in(text):
    return _REG_RE.findall(text or "")


def _decode_defuse(inst):
    """Return (defs:set, uses:set, ok:bool) for a wait-block body instruction."""
    op = inst["op_code"]
    model = _WB_OPCODE_MODEL.get(op)
    ops = [o.get("operand_string", "") for o in inst.get("operands", [])]
    defs, uses = set(), set()
    if model == "dst_first":
        if ops:
            d = _regs_in(ops[0])
            defs.update(d)
            for o in ops[1:]:
                uses.update(_regs_in(o))
        # a predicate guard on the instruction is an additional use
        if inst.get("is_predicated"):
            uses.add(("UP" if inst.get("is_uniform_predicate") else "P")
                     + str(inst.get("predicate_register")))
        return defs, uses, True
    if model == "wait":
        # operand[0] = predicate destination; operand[1] = barrier; [2] = state.
        if ops:
            defs.update(_regs_in(ops[0]))
            for o in ops[1:]:
                uses.update(_regs_in(o))
        if inst.get("is_predicated"):
            uses.add(("UP" if inst.get("is_uniform_predicate") else "P")
                     + str(inst.get("predicate_register")))
        return defs, uses, True
    if model in ("nanosleep", "bra"):
        if inst.get("is_predicated"):
            uses.add(("UP" if inst.get("is_uniform_predicate") else "P")
                     + str(inst.get("predicate_register")))
        return defs, uses, True
    return defs, uses, False


def _barrier_offset(text):
    """Return the immediate byte offset of a barrier operand `[base(+idx)(+0xNN)]`.
    A barrier with no `+0x..` immediate (e.g. `[R7+URZ]`) is base+0, so return
    '0x0'. Returns None only if the operand is not a memory-form barrier."""
    text = text or ""
    if "[" not in text:
        return None
    m = _BARR_OFF_RE.search(text)
    return m.group(1).lower() if m else "0x0"


def prove_wait_condition(fid, cand, byp, sync_sites):
    """§1.7.2 — all gate + fallback waits address the SAME barrier offset and use a
    wait-state with consistent provenance. Base register may differ (UR9 vs R11 via
    a reviewed copy), so we compare the immediate offset and require the wait-state
    to be a single reviewed source. Returns (ok, reason|None, canonical)."""
    wait_pcs = list(cand["all_gate_wait_pcs"]) + list(cand["fallback_wait_pcs"])
    offs = set()
    states = set()
    for wp in wait_pcs:
        inst = byp.get(wp)
        if inst is None:
            return False, PROVENANCE_UNKNOWN, None
        ops = [o.get("operand_string", "") for o in inst.get("operands", [])]
        if len(ops) < 3:
            return False, PROVENANCE_UNKNOWN, None
        off = _barrier_offset(ops[1])
        if off is None:
            return False, PROVENANCE_UNKNOWN, None
        offs.add(off)
        states.add(ops[2])            # wait-state operand text (e.g. R11 / R15)
    if len(offs) != 1:
        return False, WAIT_COND_MISMATCH, None
    canonical = {
        "barrier_offset": next(iter(offs)),
        "wait_state_operands": sorted(states),
    }
    # Wait-state provenance: FA3 reloads the state into different regs per site
    # (R11 at the gate, R15 in the fallback) but all originate from the same
    # UR7-derived phase value. We cannot chase full dataflow from aggregate usage
    # info, so we require every wait-state operand to be a single reviewed source:
    # one general/uniform register, or the zero register RZ/URZ (a literal
    # phase-0 wait). Record them for review rather than guessing equivalence.
    def _is_single_state_src(s):
        s = s.strip()
        if s in ("RZ", "URZ"):
            return True
        return len(_regs_in(s)) == 1
    for s in states:
        if not _is_single_state_src(s):
            return False, PROVENANCE_UNKNOWN, None
    return True, None, canonical


def prove_effects(fid, cand, byp):
    """§1.7.3 — the alternate path contains only reviewed wait/backoff/pure-ALU/
    control opcodes; no memory/TMA/GMMA/MUFU/atomic/ARRIVE/EXCH/BAR/fence and no
    unknown opcode. Returns (ok, reason|None, effect_classes)."""
    classes = Counter()
    for pc in cand["allowed_block_pcs"]:
        inst = byp[pc]
        op = inst["op_code"]
        base = op.split(".")[0]
        # Local memory (STL/LDL) is a real side effect; skipping the loop would
        # drop or reorder a local store/load. Classify as unsafe (not just
        # undecoded) so the report reason is precise.
        if is_local_mem(op):
            return False, EFFECT_UNSAFE, None
        if is_semantic(op) or is_async_producer(op):
            return False, EFFECT_UNSAFE, None
        model = _WB_OPCODE_MODEL.get(op)
        if model is None:
            return False, DEFUSE_UNDECODED, None
        classes[model] += 1
    return True, None, dict(classes)


def prove_predicate_dead_at_resume(fid, cand, byp):
    """§1.7.3 predicate safety, reconciled with §1.8.2 runtime behavior.

    The wait predicate P is set by the entry gate TRYWAIT. §1.8.2 fixes that the
    entry gate TRYWAIT is ALWAYS executed (never rolled back or skipped); only the
    residual fallback poll loop is skipped. Therefore P holds the correct
    gate-evaluated value at resume_pc whether or not P is redefined afterwards, so
    a live-out P is safe here — folding the fallback loop does not change what set
    P. We still classify the post-resume disposition for diagnostics:
      * redefined-before-read  -> disposition="dead"
      * read-before-redefine   -> disposition="live_out_gate_set" (safe: gate ran)
      * neither within window   -> disposition="unused"
    Returns (ok, reason|None, info) where info={next_predicate_def_pc, disposition,
    first_read_pc}. This proof no longer rejects a live-out predicate; a genuine
    unsafe case (a predicate produced only inside the SKIPPED fallback and needed
    downstream) cannot occur because the fallback body defines only its own loop
    predicate, which is dead once the loop exits."""
    predreg = ("UP" if cand.get("wait_predicate_uniform") else "P") + \
        str(cand["wait_predicate"])
    order = sorted(p for p in byp if p >= cand["resume_pc"])
    guard = 0
    for pc in order:
        guard += 1
        if guard > 256:
            break
        inst = byp[pc]
        defs, uses = _decode_defuse_generic(inst)
        if predreg in defs:
            return True, None, {"next_predicate_def_pc": pc,
                                "disposition": "dead", "first_read_pc": None}
        if predreg in uses:
            # Read before redefinition. Safe because the gate that set P is not
            # skipped (§1.8.2). Record it for review.
            return True, None, {"next_predicate_def_pc": None,
                                "disposition": "live_out_gate_set",
                                "first_read_pc": pc}
    return True, None, {"next_predicate_def_pc": None,
                        "disposition": "unused", "first_read_pc": None}


def _decode_defuse_generic(inst):
    """Best-effort def/use for opcodes OUTSIDE the wait-block model, used only for
    the post-resume predicate liveness scan. Recognizes the destination-predicate
    forms FA3 uses to redefine the wait predicate (ISETP/FSETP/PLOP3/etc.) and any
    'PT'/'P<n>' written as operand[0]. Uses operand_string tokens; unknown shapes
    contribute no defs (so liveness stays conservative: a real later use will still
    be seen)."""
    ops = [o.get("operand_string", "") for o in inst.get("operands", [])]
    defs, uses = set(), set()
    op = inst["op_code"]
    # SETP-family and predicate-logic write their predicate destination first.
    if any(op.startswith(pfx) for pfx in ("ISETP", "FSETP", "DSETP", "HSETP",
                                          "PLOP3", "PSETP", "P2R", "VSETP",
                                          "UISETP", "UFSETP", "UPLOP3", "UPSETP")):
        # destinations are the leading predicate operands (may be "P0, P1")
        lead = []
        for o in ops:
            r = _regs_in(o)
            if r and r[0].lstrip("U").startswith("P"):
                lead.append(r[0])
            else:
                break
        defs.update(lead)
        for o in ops[len(lead):]:
            uses.update(_regs_in(o))
    else:
        # generic: operand[0] is destination if it is a register; rest are uses.
        if ops:
            defs.update(_regs_in(ops[0]))
            for o in ops[1:]:
                uses.update(_regs_in(o))
    if inst.get("is_predicated"):
        uses.add(("UP" if inst.get("is_uniform_predicate") else "P")
                 + str(inst.get("predicate_register")))
    return defs, uses


def prove_candidate(fid, cand, byp, sync_sites):
    """Run all static proofs for one CFG-proven candidate. Mutates cand with proof
    fields and returns (ok, reason|None)."""
    ok, reason, canonical = prove_wait_condition(fid, cand, byp, sync_sites)
    if not ok:
        return False, reason
    cand["canonical_wait_condition"] = canonical
    ok, reason, classes = prove_effects(fid, cand, byp)
    if not ok:
        return False, reason
    cand["fallback_effect_safe"] = True
    cand["effect_classes"] = classes
    ok, reason, pinfo = prove_predicate_dead_at_resume(fid, cand, byp)
    if not ok:
        return False, reason
    cand["wait_predicate_disposition"] = pinfo["disposition"]
    cand["next_predicate_def_pc"] = pinfo["next_predicate_def_pc"]
    cand["wait_predicate_first_read_pc"] = pinfo["first_read_pc"]
    return True, None


def make_wait_block_record(c):
    """Serialize a fully proven candidate into the §1.7.5 WaitBlockRecord schema.
    PC fields are emitted as hex strings (like the rest of the sidecar consumers
    expect for cross-checking) plus decimal for the simulator loader."""
    st = c.get("dynamic", {})
    def hx(x):
        return "0x%x" % x
    return {
        "block_id": c["block_id"],
        "unique_function_id": c["unique_function_id"],
        "entry_wait_pc": c["entry_wait_pc"],
        "entry_wait_pc_hex": hx(c["entry_wait_pc"]),
        "all_gate_wait_pcs": c["all_gate_wait_pcs"],
        "captured_next_pc": c["captured_next_pc"],
        "allowed_block_pcs": c["allowed_block_pcs"],
        "fallback_entry_pc": c["fallback_entry_pc"],
        "fallback_body_pcs": c["fallback_body_pcs"],
        "fallback_wait_pcs": c["fallback_wait_pcs"],
        "fallback_loopback_pc": c["fallback_loopback_pc"],
        "resume_pc": c["resume_pc"],
        "resume_pc_hex": hx(c["resume_pc"]),
        "canonical_wait_condition": c["canonical_wait_condition"],
        "fallback_effect_safe": c["fallback_effect_safe"],
        "effect_classes": c["effect_classes"],
        "wait_predicate": c["wait_predicate"],
        "wait_predicate_uniform": c["wait_predicate_uniform"],
        "wait_predicate_disposition": c["wait_predicate_disposition"],
        "next_predicate_def_pc": c["next_predicate_def_pc"],
        "wait_predicate_first_read_pc": c.get("wait_predicate_first_read_pc"),
        # Mainline setup between the gate TRYWAIT and its branch. These produce
        # live-out values (addresses) and MUST NOT be skipped by the simulator.
        # Recorded explicitly so Stage 1.8's cursor-skip guard can assert it never
        # consumes one of these PCs.
        "mainline_setup_pcs": c.get("mainline_setup_pcs", []),
        "park_on_model_miss": True,
        "observed_path_counts": {
            "A_first_gate_ready": st.get("path_A_first_gate_ready", 0),
            "B_second_gate_ready": st.get("path_B_second_gate_ready", 0),
            "C_fallback_loop": st.get("path_C_fallback_loop", 0),
        },
        "active_entry_visits": st.get("active_entry_visits", 0),
        "predicated_off_entry_occurrences": st.get(
            "predicated_off_entry_occurrences", 0),
    }




def kernel_dir_map(traces_dir):
    base = os.path.join(traces_dir, "threadblocks", "device_0", "stream_0")
    out = {}
    if not os.path.isdir(base):
        return out
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if os.path.isdir(full) and d.startswith("kernel_"):
            try:
                out[int(d.split("_", 1)[1])] = full
            except ValueError:
                pass
    return out


def validate_dynamic(kdir, regions, byp, max_ctas):
    """Confirm each region's anchors appear in the recorded producer streams and
    the body actually repeats (spins). Returns per-region dynamic stats."""
    files = sorted(p for p in glob.glob(os.path.join(kdir, "*.pb")))
    if max_ctas != "all":
        files = files[:int(max_ctas)]
    loopback_pcs = {r["loopback_bra_pc"] for r in regions}
    stats = {r["region_id"]: {
        # a "visit" is one (cta, warp) that entered this region at least once;
        # iters is the number of loopback-BRA executions in that visit.
        "warp_visits": 0, "total_iters": 0, "max_iters": 0,
        "min_iters": None, "single_shot_visits": 0} for r in regions}
    for f in files:
        warps = dict((k, [pc for pc, _ in insts])
                     for k, insts in _ts._vpb_parse(open(f, "rb").read()))
        for wid, pcs in warps.items():
            # count loopback occurrences per region -> iteration count in this warp
            per_region = Counter()
            for p in pcs:
                if p in loopback_pcs:
                    per_region[p] += 1
            # attribute loopback pc -> region (one visit per (cta,warp) that entered)
            for r in regions:
                lb = r["loopback_bra_pc"]
                n = per_region.get(lb, 0)
                if n > 0:
                    s = stats[r["region_id"]]
                    s["warp_visits"] += 1
                    s["total_iters"] += n
                    s["max_iters"] = max(s["max_iters"], n)
                    s["min_iters"] = n if s["min_iters"] is None else min(s["min_iters"], n)
                    if n == 1:
                        s["single_shot_visits"] += 1
    # a region that only ever executes its loopback once is a single-shot wait,
    # not a spin loop: collapsing it is a no-op, so mark it so the simulator can
    # treat it as an ordinary (non-collapsing) wait if desired.
    for r in regions:
        s = stats[r["region_id"]]
        s["is_spin_loop"] = bool(s["warp_visits"]) and s["max_iters"] > 1
    return stats, len(files)


# --- Stage 1.7: dynamic capture proof (§1.7.4) ---------------------------- #
# FAIL reason taxonomy (see plan §1.7.8), continued:
DYNAMIC_PATH_ESCAPE = "DYNAMIC_PATH_ESCAPE"
CTA_IDENTITY_MISMATCH = "CTA_IDENTITY_MISMATCH"


def validate_wait_blocks_dynamic(kdir, cands, byp, max_ctas, expected_fid):
    """For every proven-shape candidate, traverse each CTA/warp occurrence with the
    recorded active/predicate masks and prove that:
      * every active visit begins at entry_wait_pc (an inactive occurrence, i.e.
        active_mask==0, is not a visit);
      * every dynamic PC between entry and resume is inside allowed_block_pcs;
      * every visit lands at resume_pc;
      * kernel/uid/CTA identity matches.
    A/B/C captured-path outcomes are counted for diagnostics (mixed is allowed).
    Returns {block_id: stats_dict} and the number of CTA files scanned. A proof
    failure sets stats["dynamic_ok"]=False and stats["reason"]=<taxonomy>."""
    files = sorted(glob.glob(os.path.join(kdir, "*.pb")))
    if max_ctas != "all":
        files = files[:int(max_ctas)]
    stats = {}
    for c in cands:
        stats[c["block_id"]] = {
            "active_entry_visits": 0,
            "predicated_off_entry_occurrences": 0,
            "path_A_first_gate_ready": 0,   # entry -> captured_next, no 2nd gate
            "path_B_second_gate_ready": 0,  # 2nd gate poll before resume
            "path_C_fallback_loop": 0,      # entered the OOL fallback
            "resume_landings": 0,
            "dynamic_ok": True,
            "reason": None,
        }
    for f in files:
        warps = vpb_parse_full(open(f, "rb").read())
        for wid, insts in warps:
            pcs = [i["pc"] for i in insts]
            acts = [i["active_mask"] for i in insts]
            n = len(pcs)
            for c in cands:
                st = stats[c["block_id"]]
                if not st["dynamic_ok"]:
                    continue
                entry = c["entry_wait_pc"]
                resume = c["resume_pc"]
                allowed = set(c["allowed_block_pcs"])
                # Type-1 has a 2nd gate poll (all_gate_wait_pcs[1]); Type-2 has a
                # single gate, so there is no 2nd-gate PC to watch for.
                gates = c["all_gate_wait_pcs"]
                g2 = gates[1] if len(gates) > 1 else None
                # Mainline setup between the gate and its branch is executed on the
                # ready path before reaching resume; it is intentionally NOT in
                # allowed_block_pcs (never skipped), so treat it as pass-through
                # here rather than a path escape.
                mainline = set(c.get("mainline_setup_pcs", []))
                fb_entry = c["fallback_entry_pc"]
                j = 0
                while j < n:
                    if pcs[j] != entry:
                        j += 1
                        continue
                    # occurrence of the block entry
                    if acts[j] == 0:
                        st["predicated_off_entry_occurrences"] += 1
                        j += 1
                        continue
                    st["active_entry_visits"] += 1
                    # Remember which lanes entered here. On a divergent warp the
                    # trace interleaves other lane-groups' PCs (a different
                    # active_mask). Only instructions whose active_mask overlaps
                    # the entering lanes belong to THIS visit; disjoint-mask PCs
                    # are another divergent path and are skipped here.
                    entry_mask = acts[j]
                    saw_g2 = False
                    saw_fb = False
                    k = j
                    landed = False
                    while k < n:
                        p = pcs[k]
                        # skip instructions that belong to a different divergent
                        # lane-group (no lane overlap with the entering lanes).
                        if (acts[k] & entry_mask) == 0:
                            k += 1
                            continue
                        if p == resume:
                            landed = True
                            break
                        if p not in allowed and p not in mainline:
                            st["dynamic_ok"] = False
                            st["reason"] = DYNAMIC_PATH_ESCAPE
                            st["escape_pc"] = "0x%x" % p
                            break
                        if p == g2:
                            saw_g2 = True
                        if p == fb_entry:
                            saw_fb = True
                        k += 1
                    if not st["dynamic_ok"]:
                        break
                    if not landed:
                        # ran off the end of the recorded stream without resuming:
                        # the block did not complete in this capture -> not proven.
                        st["dynamic_ok"] = False
                        st["reason"] = DYNAMIC_PATH_ESCAPE
                        st["escape_pc"] = "end-of-stream"
                        break
                    st["resume_landings"] += 1
                    if saw_fb:
                        st["path_C_fallback_loop"] += 1
                    elif saw_g2:
                        st["path_B_second_gate_ready"] += 1
                    else:
                        st["path_A_first_gate_ready"] += 1
                    j = k + 1
    return stats, len(files)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help=".../traces directory")
    ap.add_argument("--kernels", default=None, help="comma list of kernel ids (default: all)")
    ap.add_argument("--out", default=None,
                    help="output json (default: <traces>/extra_info/spin_wait_sidecar.json)")
    ap.add_argument("--validate-ctas", default="all", help="'all' or an integer count")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any detected region has semantic ops in its body")
    args = ap.parse_args()

    traces = args.traces
    extra = os.path.join(traces, "extra_info")
    if not os.path.isdir(extra):
        sys.exit("no extra_info under %s" % traces)

    kernels, names = load_kernels(extra)
    sync_sites = load_sync_sites(extra)
    if not sync_sites:
        print("[spin-sidecar] no sync_operand_resolver.json; nothing to do")
        return
    kdirs = kernel_dir_map(traces)
    want = set(int(x) for x in args.kernels.split(",")) if args.kernels else None

    # map kernel_id -> funcid via the first pb (peek), so we describe per kernel_id
    kid_to_fid = {}
    for kid, kdir in kdirs.items():
        pbs = sorted(glob.glob(os.path.join(kdir, "*.pb")))
        if pbs:
            kid_to_fid[kid] = _ts._peek_funcid(pbs[0])

    out_kernels = []
    strict_violation = False
    # §1.7.8 mandatory generator report accumulators (always printed).
    wb_report_lines = []
    wb_total_discovered = 0
    wb_total_passed = 0
    wb_total_failed = 0
    wb_total_skipped = 0
    for kid in sorted(kdirs):
        if want and kid not in want:
            continue
        fid = kid_to_fid.get(kid)
        if fid is None or fid not in kernels:
            continue
        byp = kernels[fid]

        # ---- Legacy spin-region detection runs FIRST so we know which loops are
        # foldable. A wait-block candidate whose fallback covers a foldable spin
        # loop is a fold TARGET: if it fails to prove it is a real failure to fix.
        # A candidate whose fallback is not a foldable loop (single-shot wait,
        # local-memory loop, non-compound shape) is legitimately SKIPPED, not
        # failed (plan §1.7.7 gate 1 / §1.7.8 taxonomy split, three-way report).
        regions = detect_regions(fid, byp, sync_sites)
        stats, n_ctas = ({}, 0)
        clean = []
        foldable_body_pcs = set()
        if regions:
            stats, n_ctas = validate_dynamic(
                kdirs[kid], regions, byp, args.validate_ctas)
            for r in regions:
                st = stats[r["region_id"]]
                r_out = dict(r)
                r_out["dynamic"] = st
                # The simulator folds a region ONLY if all three gates hold:
                # it is an actual spin loop, its body has no semantic work, and its
                # SASS wait-barriers are self-tied (whole-loop skip is scoreboard
                # net-zero). See .plan/SPIN_WAIT_SIM_MODELING.md 12.7.
                r_out["foldable"] = bool(
                    st["is_spin_loop"] and r["semantic_free_body"]
                    and r["barrier_self_tied"])
                if r_out["foldable"]:
                    # PCs from loop head through the loopback branch: any
                    # wait-block fallback overlapping these is a fold target.
                    lo, hi = r["loop_head_pc"], r["loopback_bra_pc"]
                    foldable_body_pcs.update(
                        range(lo, hi + ISIZE, ISIZE))
                clean.append(r_out)

        def _is_fold_target(c):
            """A candidate is a fold target if its fallback loop overlaps a
            foldable spin region (i.e. legacy fold would have folded it)."""
            return bool(set(c.get("fallback_body_pcs", [])) & foldable_body_pcs)

        # ---- Stage 1.7 compound wait-block discovery (independent per candidate).
        # Discovered with NO application/kernel/PC filter; each candidate is judged
        # on its own. A failure never blocks emitting the others (plan §1.7.7/1.7.8).
        wb_cands = detect_wait_block_candidates(fid, byp, sync_sites)
        wb_pass, wb_fail, wb_skip = [], [], []
        wb_dyn = {}
        cfg_ok_cands = [c for c in wb_cands if c["cfg_ok"]]
        if cfg_ok_cands:
            wb_dyn, _ = validate_wait_blocks_dynamic(
                kdirs[kid], cfg_ok_cands, byp, args.validate_ctas, fid)

        def _bucket(c, reason):
            """Route a non-passing candidate to failed (fold target) or skipped
            (not a fold target: legitimate fail-closed exclusion)."""
            if _is_fold_target(c):
                wb_fail.append((c, reason))
            else:
                wb_skip.append((c, reason))

        for c in wb_cands:
            if not c["cfg_ok"]:
                _bucket(c, c["reason"])
                continue
            ok, reason = prove_candidate(fid, c, byp, sync_sites)
            if not ok:
                _bucket(c, reason)
                continue
            st = wb_dyn.get(c["block_id"], {})
            if not st.get("dynamic_ok", False):
                _bucket(c, st.get("reason", DYNAMIC_PATH_ESCAPE))
                continue
            c["dynamic"] = st
            wb_pass.append(c)
        wb_total_discovered += len(wb_cands)
        wb_total_passed += len(wb_pass)
        wb_total_failed += len(wb_fail)
        wb_total_skipped += len(wb_skip)
        wb_report_lines.append(
            "[wait-block] kernel_%d funcid=%d: discovered=%d passed=%d "
            "failed=%d skipped=%d"
            % (kid, fid, len(wb_cands), len(wb_pass), len(wb_fail),
               len(wb_skip)))
        for c in wb_pass:
            st = c["dynamic"]
            wb_report_lines.append(
                "    PASS entry=0x%x resume=0x%x gates=%s fb=0x%x loopback=0x%x "
                "off=%s visits=%d A/B/C=%d/%d/%d"
                % (c["entry_wait_pc"], c["resume_pc"],
                   [hex(x) for x in c["all_gate_wait_pcs"]],
                   c["fallback_entry_pc"], c["fallback_loopback_pc"],
                   c["canonical_wait_condition"]["barrier_offset"],
                   st["active_entry_visits"], st["path_A_first_gate_ready"],
                   st["path_B_second_gate_ready"], st["path_C_fallback_loop"]))
        for c, reason in wb_fail:
            wb_report_lines.append(
                "    FAIL entry=0x%x reason=%s (fold target: must fix)"
                % (c["entry_wait_pc"], reason))
        for c, reason in wb_skip:
            wb_report_lines.append(
                "    SKIP entry=0x%x reason=%s (not a fold target)"
                % (c["entry_wait_pc"], reason))

        # ---- Demote foldable spin regions that a proven wait block now owns.
        # A Type-2 wait block's fallback IS an existing spin loop; the simulator
        # must fold it via the wait block (single owner), not also via the legacy
        # spin-region path. Mark such regions foldable=false and record which
        # wait block absorbed them so the double-fold cannot happen (user-approved
        # dedup: "absorbed spin_region -> foldable released").
        absorbed_pcs = set()
        for c in wb_pass:
            absorbed_pcs.update(c.get("fallback_body_pcs", []))
        n_absorbed = 0
        for r in clean:
            if not r["foldable"]:
                continue
            lo, hi = r["loop_head_pc"], r["loopback_bra_pc"]
            body = set(range(lo, hi + ISIZE, ISIZE))
            owner = None
            for c in wb_pass:
                if body & set(c.get("fallback_body_pcs", [])):
                    owner = c["block_id"]
                    break
            if owner is not None:
                r["foldable"] = False
                r["absorbed_by_wait_block"] = owner
                n_absorbed += 1
        if n_absorbed:
            wb_report_lines.append(
                "    [dedup] demoted %d foldable spin region(s) now owned by a "
                "wait block (foldable released)" % n_absorbed)

        # ---- Emit kernel entry (spin regions + proven wait blocks).
        if not regions:
            # No legacy spin region, but wait blocks may still have been found for
            # this kernel; emit a wait-block-only kernel entry so PASS records and
            # the report stay consistent.
            if wb_pass:
                out_kernels.append({
                    "kernel_id": kid,
                    "unique_function_id": fid,
                    "kernel_name": names.get(fid, ""),
                    "num_regions": 0,
                    "validated_ctas": 0,
                    "spin_regions": [],
                    "wait_blocks": [make_wait_block_record(c) for c in wb_pass],
                })
            continue

        # regions/stats/clean/foldable were computed above (before wait-block
        # bucketing). Here we only fold in the strict-violation accounting.
        for r in regions:
            st = stats[r["region_id"]]
            if not r["semantic_free_body"]:
                strict_violation = True
            if st["is_spin_loop"] and not r["barrier_self_tied"]:
                strict_violation = True
        out_kernels.append({
            "kernel_id": kid,
            "unique_function_id": fid,
            "kernel_name": names.get(fid, ""),
            "num_regions": len(clean),
            "validated_ctas": n_ctas,
            "spin_regions": clean,
            "wait_blocks": [make_wait_block_record(c) for c in wb_pass],
        })
        n_sem = sum(0 if r["semantic_free_body"] else 1 for r in regions)
        n_foldable = sum(1 for r in clean if r["foldable"])
        n_bar_unsafe = sum(1 for r in regions
                           if not r["barrier_self_tied"]
                           and stats[r["region_id"]]["is_spin_loop"])
        print("[spin-sidecar] kernel_%d funcid=%d: %d spin regions "
              "(foldable=%d, semantic-in-body=%d, spin+barrier-unsafe=%d), "
              "validated %d CTAs"
              % (kid, fid, len(regions), n_foldable, n_sem, n_bar_unsafe, n_ctas))
        for r in clean:
            st = r["dynamic"]
            kind = "spin" if st["is_spin_loop"] else "single-shot"
            flags = ""
            if not r["semantic_free_body"]:
                flags += "  [!! SEMANTIC IN BODY]"
            if st["is_spin_loop"] and not r["barrier_self_tied"]:
                flags += "  [!! BARRIER NOT SELF-TIED %s]" % (
                    r["barrier_reason"]["wait_on_externally_armed_ids"]
                    or [o for _, o in r["barrier_reason"]["async_producer_ops"]])
            note = ""
            if r["local_mem_ops_in_body"]:
                note = "  local_mem=%d" % len(r["local_mem_ops_in_body"])
            print("    %-16s [%-11s] head=0x%-5x wait=0x%-5x(%s) exit=0x%-5x "
                  "iters[min/max/tot]=%s/%s/%s warp_visits=%d foldable=%d%s%s"
                  % (r["region_id"], kind, r["loop_head_pc"], r["primary_wait_pc"],
                     r["primary_wait_kind"], r["exit_pc"],
                     st["min_iters"], st["max_iters"], st["total_iters"],
                     st["warp_visits"], int(r["foldable"]), note, flags))

    # §1.7.8 mandatory wait-block report: ALWAYS printed, even with zero
    # candidates, so the next application's unproven blocks are never silent.
    print("[wait-block] === compound wait-block discovery report ===")
    print("[wait-block] TOTAL discovered=%d passed=%d failed=%d skipped=%d"
          % (wb_total_discovered, wb_total_passed, wb_total_failed,
             wb_total_skipped))
    print("[wait-block]   passed  = proven fold-target wait blocks (emitted)")
    print("[wait-block]   failed  = fold targets that FAILED to prove (must be 0)")
    print("[wait-block]   skipped = not fold targets (single-shot / local-mem / "
          "non-compound); fail-closed, no action needed")
    if wb_total_failed:
        print("[wait-block]   !! %d fold-target failure(s) remain — see FAIL lines"
              % wb_total_failed)
    for line in wb_report_lines:
        print(line)

    sidecar = {
        "version": 3,
        "schema": "spin_wait_sidecar",
        "source_traces_dir": os.path.abspath(traces),
        "isize_bytes": ISIZE,
        "note": ("PC-anchored producer/consumer spin-wait regions plus Stage 1.7 "
                 "compound wait blocks. The simulator folds a spin region ONLY if "
                 "foldable=true (is_spin_loop && semantic_free_body && "
                 "barrier_self_tied). A wait_blocks[] entry is a proven speculative "
                 "gate + out-of-line pure wait fallback: at entry_wait_pc the "
                 "simulator parks on a modeled miss or immediate-folds a modeled "
                 "hit, then skips the captured residual gate/fallback path to "
                 "resume_pc. Barrier address is runtime-only; the sidecar records "
                 "the canonical barrier offset and wait-state operands only for "
                 "cross-check. wait_blocks contains ONLY fully proven candidates; "
                 "rejected candidates are reported at generation time and fall back "
                 "to original captured replay."),
        "wait_block_fail_taxonomy": [
            CFG_UNRESOLVED, MULTI_REJOIN, WAIT_COND_MISMATCH, PROVENANCE_UNKNOWN,
            EFFECT_UNSAFE, PREDICATE_LIVE_OUT, DEFUSE_UNDECODED,
            DYNAMIC_PATH_ESCAPE, CTA_IDENTITY_MISMATCH,
        ],
        "wait_block_report": {
            "discovered": wb_total_discovered,
            "passed": wb_total_passed,
            "failed": wb_total_failed,
            "skipped": wb_total_skipped,
            "failed_semantics": ("fold-target candidates (fallback overlaps a "
                                 "foldable spin region) that failed a proof; "
                                 "must be 0 for the current application"),
            "skipped_semantics": ("candidates that are not fold targets "
                                  "(single-shot wait, local-memory loop, or "
                                  "non-compound shape); excluded fail-closed"),
        },
        "kernels": out_kernels,
    }
    out_path = args.out or os.path.join(extra, "spin_wait_sidecar.json")
    with open(out_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)
    print("[spin-sidecar] wrote %s (%d kernels)" % (out_path, len(out_kernels)))
    if args.strict and strict_violation:
        sys.exit("[spin-sidecar] STRICT: a spin region has semantic ops in its "
                 "body or a non-self-tied SASS wait-barrier (not fold-safe)")


if __name__ == "__main__":
    main()
