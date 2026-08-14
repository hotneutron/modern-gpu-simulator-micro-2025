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
  * Foldable bodies contain no TMA / *GMMA / MUFU / arrive / memory operation.
    STL/LDL are local-memory side effects and therefore fail the fold gate.

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
# Thread-local stack ops are still stateful memory operations and cannot be
# skipped by the runtime cursor.
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
    for kid in sorted(kdirs):
        if want and kid not in want:
            continue
        fid = kid_to_fid.get(kid)
        if fid is None or fid not in kernels:
            continue
        byp = kernels[fid]
        regions = detect_regions(fid, byp, sync_sites)
        if not regions:
            continue
        stats, n_ctas = validate_dynamic(kdirs[kid], regions, byp, args.validate_ctas)
        clean = []
        for r in regions:
            st = stats[r["region_id"]]
            r_out = dict(r)
            r_out["dynamic"] = st
            # Fold only actual spin loops with no semantic or local-memory work
            # and self-tied SASS wait barriers.
            r_out["foldable"] = bool(
                st["is_spin_loop"] and r["semantic_free_body"]
                and not r["local_mem_ops_in_body"]
                and r["barrier_self_tied"])
            clean.append(r_out)
            if not r["semantic_free_body"]:
                strict_violation = True
            if r["local_mem_ops_in_body"]:
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

    sidecar = {
        "version": 2,
        "schema": "spin_wait_sidecar",
        "source_traces_dir": os.path.abspath(traces),
        "isize_bytes": ISIZE,
        "note": ("PC-anchored producer/consumer spin-wait regions. The simulator "
                 "folds a region ONLY if foldable=true (is_spin_loop && "
                 "semantic_free_body && no local memory && "
                 "barrier_self_tied): block the warp until "
                 "the modeled mbarrier at the runtime barrier_addr of "
                 "primary_wait_pc is ready, then advance the trace cursor to the "
                 "next exit_pc instead of replaying the recorded poll count. "
                 "barrier_self_tied guarantees whole-loop skip is scoreboard "
                 "net-zero. Barrier address is runtime-only."),
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
