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
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict, deque

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
MAX_WAIT_BLOCK_NODES = 64
MAX_WAIT_BLOCK_BRANCH_EDGES = 16
MAX_PROVENANCE_DEPTH = 32
MAX_LIVENESS_STATES = 128
MAX_DYNAMIC_WAIT_BLOCK_SCAN = 100000
TARGET_K5_WAIT_BLOCK_ENTRY = 0x74E0

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


class WaitBlockProofError(RuntimeError):
    pass


def _inst_runtime_meta(buf, start, end):
    """Decode only fields needed by sidecar validation from one instruction."""
    i = start
    pc = 0
    active_mask = None
    predicate_mask = None
    sync_valid = False
    while i < end:
        tag, i = _ts._rv(buf, i)
        field, wire = tag >> 3, tag & 7
        if field == 1 and wire == 0:
            pc, i = _ts._rv(buf, i)
        elif field == 2 and wire == 0:
            active_mask, i = _ts._rv(buf, i)
        elif field == 3 and wire == 0:
            predicate_mask, i = _ts._rv(buf, i)
        elif field == 6 and wire == 2:
            length, payload = _ts._rv(buf, i)
            sync_end = payload + length
            j = payload
            while j < sync_end:
                sync_tag, j = _ts._rv(buf, j)
                sync_field, sync_wire = sync_tag >> 3, sync_tag & 7
                if sync_field == 1 and sync_wire == 0:
                    value, j = _ts._rv(buf, j)
                    sync_valid = bool(value)
                else:
                    j = _ts._skip(buf, j, sync_wire)
            i = sync_end
        else:
            i = _ts._skip(buf, i, wire)
    return {
        "pc": pc,
        "active_mask": active_mask,
        "predicate_mask": predicate_mask,
        "sync_valid": sync_valid,
    }


def _vpb_parse_with_masks(buf):
    """Return `(warp_key, instruction-metadata)` without changing trim helpers."""
    i = 0
    warps = []
    while i < len(buf):
        tag, i = _ts._rv(buf, i)
        field, wire = tag >> 3, tag & 7
        if field != 2 or wire != 2:
            i = _ts._skip(buf, i, wire)
            continue
        length, entry_start = _ts._rv(buf, i)
        entry_end = entry_start + length
        j = entry_start
        key = None
        insts = []
        while j < entry_end:
            entry_tag, j = _ts._rv(buf, j)
            entry_field, entry_wire = entry_tag >> 3, entry_tag & 7
            if entry_field == 1 and entry_wire == 0:
                key, j = _ts._rv(buf, j)
            elif entry_field == 2 and entry_wire == 2:
                warp_length, warp_start = _ts._rv(buf, j)
                warp_end = warp_start + warp_length
                p = warp_start
                while p < warp_end:
                    warp_tag, p = _ts._rv(buf, p)
                    warp_field, warp_wire = warp_tag >> 3, warp_tag & 7
                    if warp_field == 2 and warp_wire == 2:
                        inst_length, inst_start = _ts._rv(buf, p)
                        inst_end = inst_start + inst_length
                        insts.append(_inst_runtime_meta(buf, inst_start, inst_end))
                        p = inst_end
                    else:
                        p = _ts._skip(buf, p, warp_wire)
                j = warp_end
            else:
                j = _ts._skip(buf, j, entry_wire)
        warps.append((key, insts))
        i = entry_end
    return warps


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


def _operand_name(operand):
    text = operand.get("operand_string", "")
    return text.replace(".reuse", "").lstrip("-")


def _is_register_name(name):
    return bool(re.match(r"^(?:U?R|U?P)\d+$", name))


def _defined_names(inst):
    operands = inst.get("operands", [])
    op = inst["op_code"]
    if op.startswith("SHFL."):
        return [_operand_name(o) for o in operands[:2]]
    if "SETP" in op:
        return [_operand_name(o) for o in operands[:2]]
    count = int(inst.get("num_destination_registers", 0) or 0)
    return [_operand_name(o) for o in operands[:count]]


def _wait_predicate(inst):
    operands = inst.get("operands", [])
    if not operands or operands[0].get("operand_type") != "PRED":
        raise WaitBlockProofError("wait does not define a regular predicate")
    return _operand_name(operands[0])


def _successors(pc, byp):
    inst = byp[pc]
    root = inst["op_code"].split(".")[0]
    if root == "BRA":
        target = bra_target(inst)
        if target is None:
            raise WaitBlockProofError("BRA at 0x%x has no direct target" % pc)
        result = [target]
        if inst.get("is_predicated"):
            fallthrough = pc + ISIZE
            if fallthrough not in byp:
                raise WaitBlockProofError(
                    "predicated BRA at 0x%x has no fallthrough" % pc)
            result.append(fallthrough)
        return list(dict.fromkeys(result))
    if root in ("BRX", "BRXU", "CALL", "CAL", "RET", "EXIT", "JMP", "JMX",
                "JMXU"):
        raise WaitBlockProofError(
            "unsupported control flow %s at 0x%x" % (inst["op_code"], pc))
    fallthrough = pc + ISIZE
    return [fallthrough] if fallthrough in byp else []


def _branch_arms(branch_pc, predicate, byp):
    inst = byp.get(branch_pc)
    if not inst or inst["op_code"].split(".")[0] != "BRA":
        return None
    if not inst.get("is_predicated"):
        return None
    if int(inst.get("predicate_register", -1)) != int(predicate[1:]):
        return None
    target = bra_target(inst)
    fallthrough = branch_pc + ISIZE
    if target is None or fallthrough not in byp:
        return None
    if inst.get("is_predicate_negate"):
        return fallthrough, target
    return target, fallthrough


def _collect_subgraph(start_pc, resume_pc, byp):
    nodes = set()
    edges = {}
    stack = [start_pc]
    branch_edges = 0
    while stack:
        pc = stack.pop()
        if pc == resume_pc or pc in nodes:
            continue
        if pc not in byp:
            raise WaitBlockProofError(
                "fallback reaches unknown PC 0x%x" % pc)
        nodes.add(pc)
        if len(nodes) > MAX_WAIT_BLOCK_NODES:
            raise WaitBlockProofError("wait-block CFG node bound exceeded")
        successors = _successors(pc, byp)
        if not successors:
            raise WaitBlockProofError(
                "fallback terminates before resume at 0x%x" % pc)
        if len(successors) > 1:
            branch_edges += len(successors)
            if branch_edges > MAX_WAIT_BLOCK_BRANCH_EDGES:
                raise WaitBlockProofError("wait-block branch bound exceeded")
        edges[pc] = successors
        for successor in successors:
            if successor != resume_pc:
                stack.append(successor)
    return nodes, edges


def _strongly_connected_components(nodes, edges):
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    lowlinks = {}
    components = []

    def visit(node):
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for successor in edges.get(node, []):
            if successor not in nodes:
                continue
            if successor not in indices:
                visit(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])
        if lowlinks[node] == indices[node]:
            component = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            components.append(sorted(component))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return components


def _postdominators(nodes, edges, resume_pc):
    all_nodes = set(nodes) | {resume_pc}
    postdom = {node: set(all_nodes) for node in all_nodes}
    postdom[resume_pc] = {resume_pc}
    changed = True
    while changed:
        changed = False
        for node in sorted(all_nodes - {resume_pc}):
            successors = edges.get(node, [])
            if not successors or any(s not in all_nodes for s in successors):
                raise WaitBlockProofError(
                    "CFG has a non-resume side exit at 0x%x" % node)
            intersection = set(all_nodes)
            for successor in successors:
                intersection &= postdom[successor]
            updated = {node} | intersection
            if updated != postdom[node]:
                postdom[node] = updated
                changed = True
    return postdom


def _reviewed_copy_source(inst, destination):
    operands = inst.get("operands", [])
    if not inst["op_code"].startswith("IMAD") or len(operands) != 4:
        return None
    if _operand_name(operands[0]) != destination:
        return None
    if _operand_name(operands[1]) not in ("RZ", "URZ"):
        return None
    if _operand_name(operands[2]) not in ("RZ", "URZ"):
        return None
    source = _operand_name(operands[3])
    return source if _is_register_name(source) else None


def _find_prior_definition(register, pc, byp):
    checked = 0
    for candidate in reversed([p for p in sorted(byp) if p < pc]):
        checked += 1
        if checked > MAX_PROVENANCE_DEPTH:
            break
        if register in _defined_names(byp[candidate]):
            return candidate, byp[candidate]
    raise WaitBlockProofError(
        "no bounded reaching definition for %s at 0x%x" % (register, pc))


def _resolve_register_root(register, pc, byp, depth=0):
    if depth >= MAX_PROVENANCE_DEPTH:
        raise WaitBlockProofError("register provenance depth exceeded")
    if register.startswith("UR") or register in ("RZ", "URZ"):
        return register, []
    if not register.startswith("R"):
        raise WaitBlockProofError("unsupported provenance root %s" % register)
    def_pc, definition = _find_prior_definition(register, pc, byp)
    source = _reviewed_copy_source(definition, register)
    if source is None:
        raise WaitBlockProofError(
            "unsupported definition of %s at 0x%x (%s)"
            % (register, def_pc, definition["op_code"]))
    root, chain = _resolve_register_root(source, def_pc, byp, depth + 1)
    step = {
        "register": register,
        "definition_pc": def_pc,
        "opcode": definition["op_code"],
        "source": source,
    }
    return root, [step] + chain


def _parse_barrier_expression(text, pc, byp):
    if not text or not text.startswith("[") or not text.endswith("]"):
        raise WaitBlockProofError(
            "unsupported barrier expression at 0x%x: %s" % (pc, text))
    inner = text[1:-1]
    registers = re.findall(r"UR\d+|R\d+", inner)
    if len(registers) != 1:
        raise WaitBlockProofError(
            "barrier expression needs one base register at 0x%x: %s"
            % (pc, text))
    base, chain = _resolve_register_root(registers[0], pc, byp)
    constants = [
        term for term in (part.strip() for part in inner.split("+"))
        if re.match(r"^(?:0x[0-9a-fA-F]+|\d+)$", term)
    ]
    offset = sum(int(value, 0) for value in constants)
    return {
        "canonical": "%s+0x%x" % (base, offset),
        "base": base,
        "offset": offset,
        "provenance": chain,
    }


def _parse_wait_state(text, pc, byp):
    register = text.replace(".reuse", "") if text else ""
    if not _is_register_name(register):
        raise WaitBlockProofError(
            "unsupported wait-state expression at 0x%x: %s" % (pc, text))
    root, chain = _resolve_register_root(register, pc, byp)
    return {
        "canonical": root,
        "source": register,
        "provenance": chain,
    }


def _canonical_wait_condition(fid, wait_pcs, byp, sync_sites):
    site_proofs = []
    expected = None
    for pc in wait_pcs:
        site = sync_sites.get((fid, pc))
        if not site or site.get("sync_kind") not in WAIT_KINDS:
            raise WaitBlockProofError("missing sync metadata at 0x%x" % pc)
        barrier = _parse_barrier_expression(
            site.get("barrier_operand_text"), pc, byp)
        state = _parse_wait_state(site.get("semantic_operand_text"), pc, byp)
        canonical = (barrier["canonical"], state["canonical"])
        if expected is None:
            expected = canonical
        elif canonical != expected:
            raise WaitBlockProofError(
                "wait condition differs at 0x%x: %s != %s"
                % (pc, canonical, expected))
        site_proofs.append({
            "pc": pc,
            "sync_kind": site["sync_kind"],
            "barrier": barrier,
            "wait_state": state,
        })
    return {
        "barrier": expected[0],
        "wait_state": expected[1],
        "sites": site_proofs,
    }


def _effect_class(fid, pc, byp, sync_sites):
    inst = byp[pc]
    site = sync_sites.get((fid, pc))
    if site and site.get("sync_kind") in WAIT_KINDS:
        return "WAIT_OBSERVATION"
    root = inst["op_code"].split(".")[0]
    if root == "BRA":
        return "DIRECT_CONTROL"
    if inst["op_code"].startswith("NANOSLEEP"):
        return "DELAY_ONLY"
    destinations = _defined_names(inst)
    if len(destinations) == 1 and _reviewed_copy_source(
            inst, destinations[0]) is not None:
        return "PURE_COPY"
    raise WaitBlockProofError(
        "unsupported effect %s at 0x%x" % (inst["op_code"], pc))


def _scoreboard_proof(entry_pc, residual_pcs, byp):
    armed_ids = set()
    waited_ids = set()
    async_ops = []
    for pc in [entry_pc] + sorted(residual_pcs):
        inst = byp[pc]
        cb = _cb(inst)
        if cb.get("is_new_read_barrier"):
            armed_ids.add(int(cb.get("id_new_read_barrier", -1)))
        if cb.get("is_new_write_barrier"):
            armed_ids.add(int(cb.get("id_new_write_barrier", -1)))
        mask = int(cb.get("wait_barrier_bits", 0) or 0)
        for bit in range(6):
            if mask & (1 << bit):
                waited_ids.add(bit)
        if is_async_producer(inst["op_code"]):
            async_ops.append({"pc": pc, "opcode": inst["op_code"]})
    external = sorted(waited_ids - armed_ids)
    if external or async_ops:
        raise WaitBlockProofError(
            "scoreboard proof failed: external=%s async=%s"
            % (external, async_ops))
    entry_cb = _cb(byp[entry_pc])
    return {
        "safe": True,
        "retained_entry_pc": entry_pc,
        "retained_entry_write_barrier": (
            int(entry_cb.get("id_new_write_barrier"))
            if entry_cb.get("is_new_write_barrier") else None),
        "armed_ids": sorted(armed_ids),
        "waited_ids": sorted(waited_ids),
        "wait_on_externally_armed_ids": external,
        "async_producer_ops": async_ops,
        "entry_release": "instruction_retirement",
    }


def _tracked_accesses(inst, tracked):
    operands = inst.get("operands", [])
    op = inst["op_code"]
    root = op.split(".")[0]
    names = [_operand_name(operand) for operand in operands]
    guard_uses = set()
    if inst.get("is_predicated"):
        guard_uses.add("P%d" % int(inst.get("predicate_register", -1)))

    if op.startswith("SHFL."):
        defs = set(names[:2])
        uses = set(names[2:])
    elif "SETP" in op:
        defs = set(names[:2])
        uses = set(names[2:])
    elif op.startswith("SYNCS.PHASECHK"):
        defs = set(names[:1])
        uses = set(names[1:])
    elif root in ("FADD", "IMAD"):
        defs = set(names[:1])
        uses = set(names[1:])
    else:
        mentioned = (set(names) | guard_uses) & tracked
        if mentioned:
            raise WaitBlockProofError(
                "unknown def/use layout for %s at 0x%x"
                % (op, int(inst["pc_num_dec"])))
        return set(), set()
    return defs & tracked, (uses | guard_uses) & tracked


def _prove_dead_at_resume(resume_pc, tracked, byp):
    queue = deque([(resume_pc, frozenset(tracked))])
    visited = set()
    witnesses = defaultdict(set)
    states = 0
    while queue:
        pc, remaining_frozen = queue.popleft()
        state = (pc, remaining_frozen)
        if state in visited:
            continue
        visited.add(state)
        states += 1
        if states > MAX_LIVENESS_STATES:
            raise WaitBlockProofError("liveness state bound exceeded")
        remaining = set(remaining_frozen)
        if not remaining:
            continue
        if pc not in byp:
            raise WaitBlockProofError(
                "liveness reaches unknown PC 0x%x" % pc)
        inst = byp[pc]
        defs, uses = _tracked_accesses(inst, remaining)
        if uses:
            raise WaitBlockProofError(
                "live-out %s read at 0x%x before overwrite"
                % (sorted(uses), pc))
        if defs and inst.get("is_predicated"):
            raise WaitBlockProofError(
                "tracked value conditionally overwritten at 0x%x" % pc)
        for name in defs:
            witnesses[name].add(pc)
        remaining -= defs
        if not remaining:
            continue
        successors = _successors(pc, byp)
        if not successors:
            raise WaitBlockProofError(
                "live-out reaches function exit: %s" % sorted(remaining))
        for successor in successors:
            queue.append((successor, frozenset(remaining)))
    missing = set(tracked) - set(witnesses)
    if missing:
        raise WaitBlockProofError(
            "no kill-before-read witness for %s" % sorted(missing))
    return {name: sorted(pcs) for name, pcs in sorted(witnesses.items())}


def _function_identity_sha256(byp):
    payload = []
    for pc in sorted(byp):
        inst = byp[pc]
        payload.append({
            "pc": pc,
            "opcode": inst["op_code"],
            "predicated": bool(inst.get("is_predicated")),
            "predicate_negate": bool(inst.get("is_predicate_negate")),
            "predicate_register": int(inst.get("predicate_register", 0)),
            "operands": [
                operand.get("operand_string", "")
                for operand in inst.get("operands", [])
            ],
            "control_bits": _cb(inst),
        })
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _build_wait_block(fid, entry_pc, byp, sync_sites, spin_regions):
    entry = byp[entry_pc]
    predicate = _wait_predicate(entry)
    first_branch_pc = entry_pc + ISIZE
    first_arms = _branch_arms(first_branch_pc, predicate, byp)
    if first_arms is None:
        raise WaitBlockProofError(
            "entry TRYWAIT at 0x%x has no predicate-controlled branch" % entry_pc)
    resume_pc, second_wait_pc = first_arms

    second_site = sync_sites.get((fid, second_wait_pc))
    if not second_site or second_site.get("sync_kind") != "TRYWAIT":
        raise WaitBlockProofError(
            "miss arm at 0x%x is not a second TRYWAIT" % second_wait_pc)
    if _wait_predicate(byp[second_wait_pc]) != predicate:
        raise WaitBlockProofError("gate TRYWAIT predicates differ")

    second_branch_pc = second_wait_pc + ISIZE
    second_arms = _branch_arms(second_branch_pc, predicate, byp)
    if second_arms is None:
        raise WaitBlockProofError(
            "second TRYWAIT at 0x%x has no predicate-controlled branch"
            % second_wait_pc)
    second_resume_pc, fallback_entry_pc = second_arms
    if second_resume_pc != resume_pc:
        raise WaitBlockProofError(
            "gate ready arms rejoin at different PCs: 0x%x vs 0x%x"
            % (resume_pc, second_resume_pc))

    fallback_nodes, fallback_edges = _collect_subgraph(
        fallback_entry_pc, resume_pc, byp)
    fallback_wait_pcs = sorted(
        pc for pc in fallback_nodes
        if sync_sites.get((fid, pc), {}).get("sync_kind") in WAIT_KINDS)
    if not fallback_wait_pcs:
        raise WaitBlockProofError("fallback contains no wait observation")

    residual_pcs = (
        {first_branch_pc, second_wait_pc, second_branch_pc} | fallback_nodes)
    all_nodes = {entry_pc, resume_pc} | residual_pcs
    if len(all_nodes) > MAX_WAIT_BLOCK_NODES:
        raise WaitBlockProofError("combined wait-block node bound exceeded")

    edges = {}
    for pc in all_nodes - {resume_pc}:
        successors = _successors(pc, byp)
        if not successors or any(successor not in all_nodes
                                 for successor in successors):
            raise WaitBlockProofError(
                "wait block has an unreviewed side exit at 0x%x" % pc)
        edges[pc] = successors

    components = _strongly_connected_components(all_nodes - {resume_pc}, edges)
    cyclic_components = []
    for component in components:
        if len(component) > 1 or (
                len(component) == 1
                and component[0] in edges.get(component[0], [])):
            cyclic_components.append(component)
    if len(cyclic_components) != 1:
        raise WaitBlockProofError(
            "expected one wait-loop SCC, found %d" % len(cyclic_components))
    wait_scc = set(cyclic_components[0])
    if not wait_scc.intersection(fallback_wait_pcs):
        raise WaitBlockProofError("cyclic SCC contains no wait")

    loopbacks = []
    for pc in wait_scc:
        if byp[pc]["op_code"].split(".")[0] != "BRA":
            continue
        target = bra_target(byp[pc])
        if target is not None and target < pc and target in wait_scc:
            loopbacks.append(pc)
    if len(loopbacks) != 1:
        raise WaitBlockProofError(
            "expected one fallback loopback, found %s" % loopbacks)
    fallback_loopback_pc = loopbacks[0]

    postdom = _postdominators(all_nodes - {resume_pc}, edges, resume_pc)
    first_common = postdom[resume_pc] & postdom[second_wait_pc]
    second_common = postdom[resume_pc] & postdom[fallback_entry_pc]
    if first_common != {resume_pc} or second_common != {resume_pc}:
        raise WaitBlockProofError(
            "ready/miss arms do not have one unique resume")

    effect_proof = []
    tracked_liveouts = {predicate}
    for pc in [entry_pc] + sorted(residual_pcs):
        effect = _effect_class(fid, pc, byp, sync_sites)
        effect_proof.append({
            "pc": pc,
            "opcode": byp[pc]["op_code"],
            "effect": effect,
            "retained_entry": pc == entry_pc,
        })
        if effect == "PURE_COPY":
            tracked_liveouts.update(_defined_names(byp[pc]))
        if effect == "WAIT_OBSERVATION":
            if _wait_predicate(byp[pc]) != predicate:
                raise WaitBlockProofError(
                    "wait predicate differs at 0x%x" % pc)

    all_wait_pcs = [entry_pc, second_wait_pc] + fallback_wait_pcs
    condition = _canonical_wait_condition(
        fid, all_wait_pcs, byp, sync_sites)
    scoreboard = _scoreboard_proof(entry_pc, residual_pcs, byp)
    liveness = _prove_dead_at_resume(
        resume_pc, tracked_liveouts, byp)

    fallback_region_id = None
    for region in spin_regions:
        if (region["loopback_bra_pc"] == fallback_loopback_pc
                and region["loop_head_pc"] in wait_scc):
            fallback_region_id = region["region_id"]
            break

    cfg_edges = []
    for pc in sorted(edges):
        for successor in edges[pc]:
            cfg_edges.append([pc, successor])
    record = {
        "block_id": "k_fid%d_wb0x%x" % (fid, entry_pc),
        "block_kind": "COMPOUND_FALLBACK",
        "unique_function_id": fid,
        "entry_wait_pc": entry_pc,
        "all_gate_wait_pcs": [entry_pc, second_wait_pc],
        "captured_next_pc": first_branch_pc,
        "allowed_block_pcs": sorted(residual_pcs),
        "fallback_entry_pc": fallback_entry_pc,
        "fallback_body_pcs": sorted(fallback_nodes),
        "fallback_wait_pcs": fallback_wait_pcs,
        "fallback_loopback_pc": fallback_loopback_pc,
        "fallback_region_id": fallback_region_id,
        "resume_pc": resume_pc,
        "canonical_wait_condition": condition,
        "fallback_effect_safe": True,
        "wait_predicate": predicate,
        "wait_predicate_dead_at_resume": True,
        "next_predicate_def_pc": liveness[predicate][0],
        "park_on_model_miss": True,
        "observed_path_counts": {"A": 0, "B": 0, "C": 0},
        "active_entry_visits": 0,
        "predicated_off_entry_occurrences": 0,
        "static_proof": {
            "cfg_nodes": sorted(all_nodes),
            "cfg_edges": cfg_edges,
            "sccs": components,
            "cyclic_sccs": cyclic_components,
            "unique_common_postdominator": resume_pc,
            "effect_classes": effect_proof,
            "dead_liveouts": liveness,
            "scoreboard": scoreboard,
            "sass_identity_sha256": _function_identity_sha256(byp),
        },
    }
    return record


def detect_wait_blocks(fid, byp, sync_sites, spin_regions, candidate_entries):
    """Prove only explicitly approved two-gate WaitBlock entry PCs."""
    blocks = []
    rejected = []
    for entry_pc in sorted(candidate_entries):
        site = sync_sites.get((fid, entry_pc))
        if not site or site.get("sync_kind") != "TRYWAIT":
            rejected.append({
                "entry_wait_pc": entry_pc,
                "reason": "approved entry is not a TRYWAIT",
            })
            continue
        try:
            predicate = _wait_predicate(byp[entry_pc])
        except WaitBlockProofError as error:
            rejected.append({
                "entry_wait_pc": entry_pc,
                "reason": str(error),
            })
            continue
        first_arms = _branch_arms(entry_pc + ISIZE, predicate, byp)
        if first_arms is None:
            rejected.append({
                "entry_wait_pc": entry_pc,
                "reason": "approved entry has no predicate-controlled branch",
            })
            continue
        _, possible_second_wait = first_arms
        second_site = sync_sites.get((fid, possible_second_wait))
        if not second_site or second_site.get("sync_kind") != "TRYWAIT":
            rejected.append({
                "entry_wait_pc": entry_pc,
                "reason": "approved entry has no second TRYWAIT",
            })
            continue
        try:
            block = _build_wait_block(
                fid, entry_pc, byp, sync_sites, spin_regions)
            blocks.append(block)
        except WaitBlockProofError as error:
            rejected.append({
                "entry_wait_pc": entry_pc,
                "reason": str(error),
            })

    entries = [block["entry_wait_pc"] for block in blocks]
    if len(entries) != len(set(entries)):
        raise WaitBlockProofError("duplicate WaitBlock entry PC")
    for index, block in enumerate(blocks):
        body = set(block["allowed_block_pcs"])
        for other in blocks[index + 1:]:
            if body.intersection(other["allowed_block_pcs"]):
                raise WaitBlockProofError(
                    "WaitBlock bodies overlap: %s and %s"
                    % (block["block_id"], other["block_id"]))
    return blocks, rejected


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


def validate_wait_blocks_dynamic(kdir, blocks, max_ctas):
    """Validate active A/B/C visits against each statically proven WaitBlock."""
    files = sorted(p for p in glob.glob(os.path.join(kdir, "*.pb")))
    if max_ctas != "all":
        files = files[:int(max_ctas)]
    stats = {
        block["block_id"]: {
            "active_entry_visits": 0,
            "predicated_off_entry_occurrences": 0,
            "observed_path_counts": {"A": 0, "B": 0, "C": 0},
            "max_residual_path_insts": 0,
            "observed_residual_pcs": set(),
        }
        for block in blocks
    }
    for path in files:
        with open(path, "rb") as trace_file:
            warps = _vpb_parse_with_masks(trace_file.read())
        for _, instructions in warps:
            for block in blocks:
                entry_pc = block["entry_wait_pc"]
                resume_pc = block["resume_pc"]
                allowed = set(block["allowed_block_pcs"])
                second_wait_pc = block["all_gate_wait_pcs"][1]
                fallback_entry_pc = block["fallback_entry_pc"]
                stat = stats[block["block_id"]]
                for index, metadata in enumerate(instructions):
                    if metadata["pc"] != entry_pc:
                        continue
                    active_mask = metadata["active_mask"]
                    predicate_mask = metadata["predicate_mask"]
                    if active_mask is None or predicate_mask is None:
                        raise WaitBlockProofError(
                            "dynamic entry 0x%x lacks active/predicate masks"
                            % entry_pc)
                    effective_mask = active_mask & predicate_mask
                    if effective_mask == 0:
                        stat["predicated_off_entry_occurrences"] += 1
                        continue
                    if not metadata["sync_valid"]:
                        raise WaitBlockProofError(
                            "active entry 0x%x lacks sync runtime payload"
                            % entry_pc)
                    stat["active_entry_visits"] += 1
                    residual = []
                    cursor = index + 1
                    while cursor < len(instructions):
                        if len(residual) > MAX_DYNAMIC_WAIT_BLOCK_SCAN:
                            raise WaitBlockProofError(
                                "dynamic wait-block scan bound exceeded")
                        pc = instructions[cursor]["pc"]
                        if pc == resume_pc:
                            break
                        if pc == entry_pc:
                            raise WaitBlockProofError(
                                "nested wait-block entry before resume")
                        if pc not in allowed:
                            raise WaitBlockProofError(
                                "dynamic path contains PC 0x%x outside block"
                                % pc)
                        residual.append(pc)
                        cursor += 1
                    if cursor >= len(instructions):
                        raise WaitBlockProofError(
                            "dynamic wait-block visit has no resume PC")
                    if fallback_entry_pc in residual:
                        path_class = "C"
                    elif second_wait_pc in residual:
                        path_class = "B"
                    else:
                        path_class = "A"
                    stat["observed_path_counts"][path_class] += 1
                    stat["max_residual_path_insts"] = max(
                        stat["max_residual_path_insts"], len(residual))
                    stat["observed_residual_pcs"].update(residual)
    for stat in stats.values():
        stat["observed_residual_pcs"] = sorted(stat["observed_residual_pcs"])
    return stats, len(files)


def _is_target_k5_wait_block(kernel_id, fid, kernel_name, byp):
    return (
        kernel_id == 5
        and fid == 3
        and TARGET_K5_WAIT_BLOCK_ENTRY in byp
        and "FlashAttnFwdSm90" in kernel_name
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help=".../traces directory")
    ap.add_argument("--kernels", default=None, help="comma list of kernel ids (default: all)")
    ap.add_argument("--out", default=None,
                    help="output json (default: <traces>/extra_info/spin_wait_sidecar.json)")
    ap.add_argument("--validate-ctas", default="all", help="'all' or an integer count")
    ap.add_argument("--strict", action="store_true",
                    help="fail on any unsafe spin region or rejected WaitBlock")
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
    for kid in sorted(kdirs):
        if want and kid not in want:
            continue
        fid = kid_to_fid.get(kid)
        if fid is None or fid not in kernels:
            continue
        byp = kernels[fid]
        regions = detect_regions(fid, byp, sync_sites)
        kernel_name = names.get(fid, "")
        target_k5 = _is_target_k5_wait_block(
            kid, fid, kernel_name, byp)
        candidate_entries = (
            [TARGET_K5_WAIT_BLOCK_ENTRY] if target_k5 else [])
        wait_blocks, wait_block_rejections = detect_wait_blocks(
            fid, byp, sync_sites, regions, candidate_entries)
        target_blocks = [
            block for block in wait_blocks
            if block["entry_wait_pc"] == TARGET_K5_WAIT_BLOCK_ENTRY
        ]
        if target_k5 and len(target_blocks) != 1:
            reasons = [
                rejection["reason"] for rejection in wait_block_rejections
                if rejection["entry_wait_pc"] == TARGET_K5_WAIT_BLOCK_ENTRY
            ]
            sys.exit(
                "[spin-sidecar] K5 WaitBlock 0x%x proof failed: %s"
                % (TARGET_K5_WAIT_BLOCK_ENTRY,
                   reasons or ["candidate not detected"]))
        if not regions and not wait_blocks:
            continue
        stats, region_ctas = validate_dynamic(
            kdirs[kid], regions, byp, args.validate_ctas)
        if wait_blocks:
            block_stats, block_ctas = validate_wait_blocks_dynamic(
                kdirs[kid], wait_blocks, args.validate_ctas)
        else:
            block_stats, block_ctas = {}, region_ctas
        if region_ctas != block_ctas:
            raise WaitBlockProofError(
                "spin/wait-block CTA validation counts differ")
        n_ctas = region_ctas
        clean = []
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
            clean.append(r_out)
            if not r["semantic_free_body"]:
                strict_violation = True
            if st["is_spin_loop"] and not r["barrier_self_tied"]:
                strict_violation = True
        clean_wait_blocks = []
        for block in wait_blocks:
            block_out = dict(block)
            block_out["trace_kernel_id"] = kid
            dynamic = block_stats[block["block_id"]]
            block_out["observed_path_counts"] = dynamic["observed_path_counts"]
            block_out["active_entry_visits"] = dynamic["active_entry_visits"]
            block_out["predicated_off_entry_occurrences"] = (
                dynamic["predicated_off_entry_occurrences"])
            block_out["dynamic"] = dynamic
            clean_wait_blocks.append(block_out)
        if wait_block_rejections:
            strict_violation = True
        out_kernels.append({
            "kernel_id": kid,
            "unique_function_id": fid,
            "kernel_name": kernel_name,
            "num_regions": len(clean),
            "num_wait_blocks": len(clean_wait_blocks),
            "validated_ctas": n_ctas,
            "spin_regions": clean,
            "wait_blocks": clean_wait_blocks,
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
        for block in clean_wait_blocks:
            paths = block["observed_path_counts"]
            print("    %-16s [wait-block ] entry=0x%-5x resume=0x%-5x "
                  "active=%d off=%d paths[A/B/C]=%d/%d/%d proof=%s"
                  % (block["block_id"], block["entry_wait_pc"],
                     block["resume_pc"], block["active_entry_visits"],
                     block["predicated_off_entry_occurrences"],
                     paths["A"], paths["B"], paths["C"],
                     block["static_proof"]["sass_identity_sha256"][:12]))
        for rejection in wait_block_rejections:
            print("    [wait-block rejected] entry=0x%x reason=%s"
                  % (rejection["entry_wait_pc"], rejection["reason"]))

    sidecar = {
        "version": 3,
        "schema": "spin_wait_sidecar",
        "source_traces_dir": os.path.abspath(traces),
        "isize_bytes": ISIZE,
        "note": ("PC-anchored spin regions plus statically proven compound "
                 "WaitBlocks. Existing spin_regions retain their v2 records. "
                 "A WaitBlock is emitted only when bounded CFG, wait-condition "
                 "provenance, effect, scoreboard, liveness, dynamic active-mask, "
                 "and same-resume proofs all pass. Barrier addresses remain "
                 "runtime-only."),
        "kernels": out_kernels,
    }
    out_path = args.out or os.path.join(extra, "spin_wait_sidecar.json")
    with open(out_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)
    print("[spin-sidecar] wrote %s (%d kernels)" % (out_path, len(out_kernels)))
    if args.strict and strict_violation:
        sys.exit("[spin-sidecar] STRICT: unsafe spin region or rejected "
                 "WaitBlock candidate")


if __name__ == "__main__":
    main()
