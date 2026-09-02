#!/usr/bin/env python3
"""Build a kernel-aware real-GMEM-base map for TMA operations.

Descriptor identity follows the UTMACCTL.PF data-flow edge that copies an exact
128-byte descriptor from the launch-parameter struct into runtime TMA state.
Numeric equality between offsets from those two address spaces is not identity.

Real base assignment, per runtime kernel:
  1. discover struct descriptor slots + qword0 base by sliding each encode blob over the
     launch-arg params-struct blob (exact 128B match).
  2. select the exact static kernel variant for each runtime trace kernel.
  3. resolve each runtime TMA-state offset through the UTMACCTL.PF
     source/destination chain. Missing or ambiguous chains fail closed.
  4. derive UBLK covered bytes from the statically identified operand-3 callback;
     hidden instrumentation callbacks never participate in size selection.

Output: extra_info/tma_pc_base_map.json
        {"kernel_id:uid:pc": {base_hex, smem_offset_hex, source, ...}}

Usage: python3 build_tma_pc_base_map.py --traces <dir> [--struct-desc-delta 0x30]
"""
import argparse, csv, json, struct
from collections import defaultdict
from pathlib import Path
import re

DESC_OPS = ("UTMALDG", "UTMASTG", "UTMAPF", "UTMAREDG")  # ops we map (UTMACCTL is the linker)


def parse_int(x):
    x = (str(x) if x is not None else "").strip()
    return int(x, 0) if x else 0


def parse_int_list(x):
    """CSV list fields are space-separated (e.g. box_dim='64 192 1 1')."""
    x = (str(x) if x is not None else "").strip()
    return [int(t, 0) for t in x.split()] if x else []


# Mirror of gpu-sim.cc parse_tma_tensor_data_type_size() so the base map is
# self-describing (C++ still derives element_size independently; we emit it too).
DTYPE_SIZE = {0: 1, 1: 2, 2: 4, 3: 4, 7: 4, 4: 8, 5: 8, 8: 8, 6: 2, 9: 2}

# Descriptor fields carried per (uid,pc) so the simulator gets base + size (and M2
# strides) WITHOUT the handle_hi/config_id resolver. Order = emit order.
DESC_FIELD_KEYS = (
    "tensor_rank", "tensor_data_type", "element_size",
    "global_dim", "global_strides", "box_dim", "element_strides",
    "interleave", "swizzle", "l2_promotion", "oob_fill",
)


def descriptor_fields_from_row(r):
    dt = parse_int(r.get("tensor_data_type"))
    return {
        "tensor_rank": parse_int(r.get("tensor_rank")),
        "tensor_data_type": dt,
        "element_size": DTYPE_SIZE.get(dt, 0),
        "global_dim": parse_int_list(r.get("global_dim")),
        "global_strides": parse_int_list(r.get("global_strides")),
        "box_dim": parse_int_list(r.get("box_dim")),
        "element_strides": parse_int_list(r.get("element_strides")),
        "interleave": parse_int(r.get("interleave")),
        "swizzle": parse_int(r.get("swizzle")),
        "l2_promotion": parse_int(r.get("l2_promotion")),
        "oob_fill": parse_int(r.get("oob_fill")),
    }


def is_smem_descriptor_addr(fla):
    """True if first_lane_addr is a generic-SMEM descriptor window address (carries the
    tensor's SMEM offset in its low bits). Two observed encodings across traces:
      - full generic-neg form 0xffffffffXXXXX... (top 32 bits set)
      - low-32 form 0x0072aXXXX (SMEM base in bits >=16, small tensor offset in low bits)
    A raw staging cursor (e.g. 0x2c0/0x400) has nothing above bit 16, so >>16==0."""
    if not isinstance(fla, int):
        return False
    if fla > 0xFFFFFFFF00000000:      # full generic-neg encoding
        return True
    return (fla >> 16) != 0           # low-32 SMEM-window encoding (base in high bits)


def smem_offset_of(fla):
    return fla & 0xFFFF


def normalize_runtime_kernel_name(name):
    return re.sub(r"___[0-9]+$", "", name or "")


def load_runtime_rows(extra):
    path = extra / "tma_runtime_operand_debug.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_runtime_kernel_names(extra, runtime_rows):
    """Map each observed (trace kernel id, function id) to its exact static name."""
    info_path = extra / "enhanced_execution_info.json"
    if not info_path.exists():
        raise SystemExit(f"missing enhanced execution info: {info_path}")
    data = json.loads(info_path.read_text())
    names_by_uid = {}
    for kernel in data.get("kernels", []):
        uid = str(kernel.get("unique_function_id"))
        name = normalize_runtime_kernel_name(kernel.get("kernel_name"))
        old = names_by_uid.get(uid)
        if old is not None and old != name:
            raise SystemExit(
                f"ambiguous enhanced kernel name for uid={uid}: {old} vs {name}"
            )
        names_by_uid[uid] = name

    result = {}
    for row in runtime_rows:
        if row.get("kernel_id") is None or row.get("unique_function_id") is None:
            continue
        key = (int(row["kernel_id"]), str(row["unique_function_id"]))
        name = names_by_uid.get(key[1])
        if not name:
            raise SystemExit(
                f"missing enhanced kernel name for trace_kernel={key[0]} uid={key[1]}"
            )
        result[key] = name
    return result


def map_key(trace_kernel_id, uid, pc):
    return f"{trace_kernel_id}:{uid}:{pc}"


def load_encode_blobs(extra):
    out = []
    p = extra / "tensor_map_encode_dump.csv"
    if not p.exists():
        return out
    with p.open(newline="") as fh:
        for r in csv.DictReader(fh):
            base = parse_int(r.get("global_address_hex") or r.get("qword0_hex"))
            cand = extra / "tensor_map_encode_blobs" / (str(r.get("dump_id")) + ".bin")
            data = cand.read_bytes() if cand.exists() else b""
            out.append((r.get("dump_id"), base, data, descriptor_fields_from_row(r)))
    return out


def struct_blob_by_kernel(extra):
    """Largest launch arg per (trace kernel, uid) = the by-value params struct.

    The CSV blob_path is an
    absolute path from the tracer host, so rebuild it relative to THIS extra_info dir
    (launch_param_blobs/<basename>) for portability across machines."""
    best = {}
    p = extra / "tma_launch_param_dump.csv"
    if not p.exists():
        return best
    with p.open(newline="") as fh:
        for r in csv.DictReader(fh):
            kernel_id = int(r.get("kernel_id"))
            uid = str(r.get("unique_function_id"))
            sz = parse_int(r.get("arg_size"))
            raw = r.get("blob_path") or ""
            local = extra / "launch_param_blobs" / Path(raw).name if raw else None
            key = (kernel_id, uid)
            if key not in best or sz > best[key][0]:
                best[key] = (
                    sz,
                    str(local) if local else None,
                    parse_int(r.get("param_offset_hex")),
                )
    return best


def discover_struct_descriptors(struct_bytes, encode_blobs):
    """inner struct offset -> {base, encode_id, fields...}, for every exact 128B
    encode-blob placement. Each matched slot carries the full descriptor (base +
    box_dim/element_size/strides) read straight from the matched encode row, so the
    (uid,pc) entry needs no handle_hi/config_id resolver."""
    slots = {}
    for did, base, data, fields in encode_blobs:
        if len(data) < 128:
            continue
        needle = data[:128]
        start = 0
        while True:
            idx = struct_bytes.find(needle, start)
            if idx < 0:
                break
            slots[idx] = {"base": base, "encode_id": did, "fields": fields}
            start = idx + 8
    return slots


def runtime_smem_offsets(runtime_rows, opcodes):
    """(trace kernel, uid, pc) -> one runtime TMA-state offset."""
    out = {}
    seen_multi = defaultdict(set)
    for r in runtime_rows:
        op = str(r.get("opcode", "")).split(".")[0]
        if op not in opcodes or r.get("callback_index") != 0:
            continue
        fla = r.get("first_lane_addr")
        if not is_smem_descriptor_addr(fla):
            continue
        key = (
            int(r.get("kernel_id")),
            str(r.get("unique_function_id")),
            r.get("pc_hex"),
        )
        off = smem_offset_of(fla)
        seen_multi[key].add(off)
        out[key] = off
    return out, seen_multi


def utmacctl_chain(extra, delta, struct_slots_by_kernel, runtime_rows,
                   runtime_kernel_names):
    """Build exact runtime-state -> descriptor edges for each trace kernel."""
    destinations = defaultdict(set)
    for row in runtime_rows:
        if (not str(row.get("opcode", "")).startswith("UTMACCTL") or
                row.get("callback_index") != 0):
            continue
        fla = row.get("first_lane_addr")
        if not is_smem_descriptor_addr(fla):
            continue
        key = (
            int(row.get("kernel_id")),
            str(row.get("unique_function_id")),
            row.get("pc_hex"),
        )
        destinations[key].add(smem_offset_of(fla))

    off_json = extra / "tma_descriptor_offsets.json"
    sources = defaultdict(set)
    if off_json.exists():
        data = json.loads(off_json.read_text())
        for entry in data.get("prefetch_sites", []):
            if entry.get("unique_function_id") is None:
                continue
            uid = str(entry.get("unique_function_id"))
            function_name = entry.get("function_name")
            offset = entry.get("tensormap_offset_hex")
            if offset is None:
                continue
            for (kernel_id, active_uid), active_name in runtime_kernel_names.items():
                if active_uid == uid and function_name == active_name:
                    sources[(kernel_id, uid, entry.get("pc_hex"))].add(
                        parse_int(offset)
                    )

    state_candidates = defaultdict(dict)
    prefetch_by_pc = {}
    problems = []
    for site, dest_values in sorted(destinations.items()):
        kernel_id, uid, pc = site
        source_values = sources.get(site, set())
        if len(dest_values) != 1 or len(source_values) != 1:
            problems.append(
                f"trace_kernel={kernel_id} uid={uid} pc={pc} "
                f"destinations={sorted(dest_values)} sources={sorted(source_values)}"
            )
            continue
        sm = next(iter(dest_values))
        struct_offset = next(iter(source_values)) - delta
        slot = struct_slots_by_kernel.get((kernel_id, uid), {}).get(struct_offset)
        if slot is None:
            problems.append(
                f"trace_kernel={kernel_id} uid={uid} pc={pc} "
                f"struct_offset=0x{struct_offset:x} has no exact descriptor"
            )
            continue
        identity = (str(slot["encode_id"]), slot["base"])
        state_candidates[(kernel_id, uid, sm)][identity] = slot
        prefetch_by_pc[site] = slot

    chain = {}
    for state_key, candidates in state_candidates.items():
        if len(candidates) != 1:
            problems.append(
                f"trace_kernel={state_key[0]} uid={state_key[1]} "
                f"state=0x{state_key[2]:x} descriptor_candidates="
                f"{sorted(candidates)}"
            )
            continue
        chain[state_key] = next(iter(candidates.values()))
    return chain, prefetch_by_pc, problems


def entry_from_slot(slot, sm, source):
    """Build a (uid,pc) map entry from a matched struct slot dict. Carries the full
    descriptor (base + all fields) so the simulator needs no handle_hi/config_id."""
    e = {"base_hex": f"0x{slot['base']:x}",
         "source": source,
         "operand_addressed": False,
         "encode_id": slot["encode_id"]}
    if sm is not None:
        e["smem_offset_hex"] = f"0x{sm:x}"
    e.update(slot["fields"])
    return e


# ---------------------------------------------------------------------------
# M2.5 — UBLKRED/UBLKCP real base + mock tiling (TMA_exact_base_mapping §M2.5)
#
# These ops are NOT tensormap-addressed (desc is a UMOV bare handle). The real
# GMEM base is a raw kernel-argument pointer sitting by value in the params
# struct (launch_param_blobs), and the per-transfer GMEM offset is in the
# runtime operand. We recover, per (uid,pc):
#   base       = the params-struct GMEM pointer that CONTAINS this pc's runtime
#                GMEM operand addresses (matched by low-32 containment).
#   tile_bytes = covered_bytes = UNIFORM span operand * 16.
#   num_tiles  = region_bytes / tile_bytes, where region_bytes is the address
#                extent of every UBLKRED/UBLKCP transfer sharing that base.
# The simulator then applies the same visit-counter mock tiling as the
# descriptor family (base + tile_idx*tile_bytes + agu_index*128), so repeated
# tile visits hit while distinct tiles miss — anchored in the real 64-bit
# address space so cross-op L2 residency is decided by the L2 model, not assumed.
# ---------------------------------------------------------------------------
UBLK_OPS = ("UBLKRED", "UBLKCP")
GMEM_VA_MIN = 0x7E00000000     # typical CUDA device VA floor
GMEM_VA_MAX = 0x800000000000   # device VA ceiling


def load_ublk_span_callbacks(extra, runtime_kernel_names, runtime_rows):
    """Return the architectural operand-3 callback for each executed UBLK site.

    The observed UBLK forms instrument their three architectural operands as
    callbacks 0, 1, and 2. A completion-barrier capture, when present, is an
    auxiliary callback appended after them and must not change this mapping.
    """
    path = extra / "tma_discovery.json"
    if not path.exists():
        raise SystemExit(f"missing TMA discovery metadata: {path}")
    discovery = json.loads(path.read_text())
    callbacks = {}
    for function in discovery.get("functions", []):
        static_uid = function.get("unique_function_id")
        function_name = function.get("function_name")
        for (kernel_id, active_uid), active_name in runtime_kernel_names.items():
            if function_name != active_name:
                continue
            if static_uid is not None and active_uid != str(static_uid):
                continue
            for instruction in function.get("tma_family_ops", []):
                opcode = str(instruction.get("opcode", "")).split(".")[0]
                if opcode not in UBLK_OPS:
                    continue
                text = instruction.get("text", "")
                suffix = text.split(instruction.get("opcode", ""), 1)[-1]
                operands = [
                    token.strip()
                    for token in suffix.split(";", 1)[0].split(",")
                    if token.strip()
                ]
                if len(operands) < 3 or not operands[2].startswith("UR"):
                    raise SystemExit(
                        f"unsupported {opcode} operand-3 form: "
                        f"trace_kernel={kernel_id} uid={active_uid} "
                        f"pc={instruction.get('pc_hex')} text={text}"
                    )
                key = (kernel_id, active_uid, instruction.get("pc_hex"))
                callback_index = 2
                old = callbacks.get(key)
                if old is not None and old != callback_index:
                    raise SystemExit(
                        f"ambiguous {opcode} span callback: key={key} "
                        f"old={old} new={callback_index}"
                    )
                callbacks[key] = callback_index

    runtime_sites = set()
    for row in runtime_rows:
        opcode = str(row.get("opcode", "")).split(".")[0]
        if opcode not in UBLK_OPS:
            continue
        runtime_sites.add((
            int(row.get("kernel_id")),
            str(row.get("unique_function_id")),
            row.get("pc_hex"),
        ))
    missing = sorted(runtime_sites - callbacks.keys())
    if missing:
        raise SystemExit(
            "missing static operand-3 callback roles for executed UBLK sites: "
            f"{missing[:8]}"
        )
    return callbacks


def gmem_pointers_from_struct(struct_bytes):
    """Every plausible device GMEM pointer (0x7e..0x80 range) in the by-value
    params struct, unique and sorted by low-32 (the coordinate the runtime
    operand exposes)."""
    ptrs = set()
    for off in range(0, len(struct_bytes) - 7, 8):
        v = struct.unpack_from("<Q", struct_bytes, off)[0]
        if GMEM_VA_MIN <= v < GMEM_VA_MAX:
            ptrs.add(v)
    return sorted(ptrs, key=lambda p: (p & 0xFFFFFFFF, p))


def runtime_ublk_operands(runtime_rows, opcodes, span_callbacks):
    """Kernel-aware per-callback operand summary for executed UBLK operations.

    The architectural span comes only from the statically identified operand-3
    callback. Hidden instrumentation callbacks are retained for diagnostics but
    can never become a byte source.
    {
       opcode, cb_addrs:{cb->set(low32)}, cb_memtype:{cb->str},
       uniform_vals:{cb->set(int)}, callback_events:{cb->count}
    }
    The runtime jsonl only holds executed ops, so every entry is an executed site."""
    per = {}
    for r in runtime_rows:
        op = str(r.get("opcode", "")).split(".")[0]
        if op not in opcodes:
            continue
        key = (
            int(r.get("kernel_id")),
            str(r.get("unique_function_id")),
            r.get("pc_hex"),
        )
        d = per.get(key)
        if d is None:
            d = per[key] = {
                "opcode": op,
                "cb_addrs": defaultdict(set),
                "cb_memtype": {},
                "uniform_vals": defaultdict(set),
                "callback_events": defaultdict(int),
                "span_callback": span_callbacks.get(key),
            }
        cb = r.get("callback_index")
        fla = r.get("first_lane_addr")
        d["cb_memtype"][cb] = r.get("mem_type")
        d["callback_events"][cb] += 1
        if r.get("operand_type") == "UNIFORM":
            if isinstance(fla, int):
                d["uniform_vals"][cb].add(fla)
        elif isinstance(fla, int):
            d["cb_addrs"][cb].add(fla & 0xFFFFFFFF)
    return per


def resolve_ublk_base(min_addr, struct_ptrs):
    """The struct GMEM pointer with the largest low-32 <= min_addr — i.e. the
    region whose start contains this pc's runtime GMEM operand addresses."""
    best = None
    for full in struct_ptrs:
        lo = full & 0xFFFFFFFF
        if lo <= min_addr and (best is None or lo > (best & 0xFFFFFFFF)):
            best = full
    return best


def build_ublk_entries(per_ublk, struct_ptrs_by_kernel):
    """Resolve base/tile_bytes/num_tiles per kernel-aware UBLK site.

    Returns (entries, stats, unmatched)."""
    # pass 1: per site pick the GMEM callback (STANDARD_MEM with the most
    # distinct addresses; the SMEM staging cursor is ~1 distinct) + covered_bytes.
    resolved = {}
    unmatched = []
    for (kernel_id, uid, pc), d in per_ublk.items():
        site_text = f"{kernel_id}:{uid}:{pc}"
        gmem_cb, gmem_addrs = None, set()
        for cb, addrs in d["cb_addrs"].items():
            if d["cb_memtype"].get(cb) == "STANDARD_MEM" and len(addrs) > len(gmem_addrs):
                gmem_cb, gmem_addrs = cb, addrs
        if not gmem_addrs:
            unmatched.append(f"{site_text}(no-gmem-operand)")
            continue
        span_cb = d["span_callback"]
        span_values = d["uniform_vals"].get(span_cb, set())
        if span_cb is None or len(span_values) != 1:
            unmatched.append(
                f"{site_text}(span-callback={span_cb},values={sorted(span_values)})"
            )
            continue
        span = next(iter(span_values))
        covered = span * 16
        base = resolve_ublk_base(
            min(gmem_addrs), struct_ptrs_by_kernel.get((kernel_id, uid), [])
        )
        if base is None:
            unmatched.append(
                f"{site_text}(no-base<=0x{min(gmem_addrs):x})"
            )
            continue
        resolved[(kernel_id, uid, pc)] = {
            "opcode": d["opcode"],
            "gmem_cb": gmem_cb,
            "addrs": gmem_addrs,
            "covered": covered,
            "base": base,
            "span_callback": span_cb,
            "span_events": d["callback_events"][span_cb],
        }

    # pass 2: region extent per (uid, base) — the full tensor span every sharing
    # transfer reaches (so a pc that only samples part still gets the true count).
    region_end = defaultdict(int)
    region_addrs = defaultdict(set)
    for (kernel_id, uid, pc), e in resolved.items():
        tb = e["covered"] or 128
        rk = (kernel_id, uid, e["base"])
        region_end[rk] = max(region_end[rk], max(e["addrs"]) + tb)
        region_addrs[rk] |= e["addrs"]

    # emit
    entries, stats = {}, defaultdict(lambda: {"UBLKRED": 0, "UBLKCP": 0})
    for (kernel_id, uid, pc), e in sorted(resolved.items()):
        base = e["base"]; base_lo = base & 0xFFFFFFFF
        tb = e["covered"] or 128
        rk = (kernel_id, uid, base)
        region_bytes = region_end[rk] - base_lo
        num_tiles = max(1, (region_bytes + tb - 1) // tb)
        entries[map_key(kernel_id, uid, pc)] = {
            "trace_kernel_id": kernel_id,
            "unique_function_id": int(uid),
            "pc_hex": pc,
            "base_hex": f"0x{base:x}",
            "source": "operand_raw_pointer",
            "operand_addressed": True,
            "raw_pointer_addressed": True,
            "has_static_base": False,          # not a tensormap descriptor base
            "opcode": e["opcode"],
            "gmem_cb_index": e["gmem_cb"],
            "span_callback_index": e["span_callback"],
            "span_event_count": e["span_events"],
            "covered_bytes": e["covered"],
            "tile_bytes": tb,
            "num_tiles": num_tiles,
            "region_bytes": region_bytes,
            "distinct_addrs_pc": len(e["addrs"]),
            "distinct_addrs_region": len(region_addrs[rk]),
        }
        stats[(kernel_id, uid)][e["opcode"]] += 1
    return entries, stats, unmatched


def count_executed_ublkred_resolved(extra):
    """UBLKRED 1:1-absence proof: how many EXECUTED UBLKRED sites the def-chain still
    resolves to an in-struct param offset. Expected 0 (desc is a UMOV bare handle, dst is
    a raw GMEM pointer) — a nonzero count would mean a UBLKRED is tensormap-addressed
    after all and the M1 'synthetic base for UBLKRED' decision must be revisited."""
    off_json = extra / "tma_descriptor_offsets.json"
    if not off_json.exists():
        return None
    data = json.loads(off_json.read_text())
    ex_total = 0
    ex_resolved = 0
    for s in data.get("sites", []):
        if not str(s.get("opcode", "")).startswith("UBLKRED"):
            continue
        if not s.get("executed"):
            continue
        ex_total += 1
        if s.get("resolved"):
            ex_resolved += 1
    return {"executed": ex_total, "resolved_to_struct": ex_resolved}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--struct-desc-delta", default="0x30",
                    help="param_source - delta == struct descriptor offset (§2.27, =0x30)")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero unless every kernel-aware descriptor site "
                         "has one unambiguous UTMACCTL chain")
    ap.add_argument("--no-ublk", action="store_true",
                    help="skip UBLKRED/UBLKCP raw-pointer base+mock-tiling emit (M2.5); "
                         "leaves them synthetic in the simulator")
    a = ap.parse_args()
    delta = parse_int(a.struct_desc_delta)
    extra = Path(a.traces).resolve() / "extra_info"

    runtime_rows = load_runtime_rows(extra)
    runtime_kernel_names = load_runtime_kernel_names(extra, runtime_rows)
    encode_blobs = load_encode_blobs(extra)
    structs = struct_blob_by_kernel(extra)
    struct_slots_by_kernel = {}
    struct_ptrs_by_kernel = {}
    for launch_key, (sz, blob, argoff) in structs.items():
        bp = Path(blob) if blob else None
        if bp and bp.exists():
            sbytes = bp.read_bytes()
            struct_slots_by_kernel[launch_key] = discover_struct_descriptors(
                sbytes, encode_blobs
            )
            struct_ptrs_by_kernel[launch_key] = gmem_pointers_from_struct(sbytes)
    chain, prefetch_by_pc, chain_problems = utmacctl_chain(
        extra,
        delta,
        struct_slots_by_kernel,
        runtime_rows,
        runtime_kernel_names,
    )
    smem_by_pc, multi = runtime_smem_offsets(runtime_rows, DESC_OPS)
    runtime_opcode_by_site = {
        (
            int(row.get("kernel_id")),
            str(row.get("unique_function_id")),
            row.get("pc_hex"),
        ): str(row.get("opcode", "")).split(".")[0]
        for row in runtime_rows
    }

    result = {}
    stats = defaultdict(lambda: {"chain": 0, "none": 0, "prefetch": 0})
    unresolved_pcs = []
    direct_candidate_sites = 0
    direct_candidate_conflicts = 0
    # Descriptor operations are keyed only through their kernel-aware UTMACCTL
    # source/destination chain. Numeric direct matches are diagnostic.
    for (kernel_id, uid, pc), sm in sorted(smem_by_pc.items()):
        site = (kernel_id, uid, pc)
        if len(multi[site]) != 1:
            stats[(kernel_id, uid)]["none"] += 1
            unresolved_pcs.append(
                f"{map_key(kernel_id, uid, pc)}"
                f"(runtime-state-values={sorted(multi[site])})"
            )
            continue
        slot = chain.get((kernel_id, uid, sm))
        if slot is None:
            stats[(kernel_id, uid)]["none"] += 1
            unresolved_pcs.append(
                f"{map_key(kernel_id, uid, pc)}(state=0x{sm:x})"
            )
            continue
        direct_slot = struct_slots_by_kernel.get((kernel_id, uid), {}).get(sm)
        if direct_slot is not None:
            direct_candidate_sites += 1
            if (str(direct_slot["encode_id"]), direct_slot["base"]) != (
                str(slot["encode_id"]),
                slot["base"],
            ):
                direct_candidate_conflicts += 1
        stats[(kernel_id, uid)]["chain"] += 1
        entry = entry_from_slot(slot, sm, "utmacctl_chain")
        entry.update({
            "trace_kernel_id": kernel_id,
            "unique_function_id": int(uid),
            "pc_hex": pc,
            "opcode": runtime_opcode_by_site[site],
        })
        result[map_key(kernel_id, uid, pc)] = entry

    # UTMACCTL.PF sites are emitted separately for prefetch modeling.
    for (kernel_id, uid, pc), slot in sorted(prefetch_by_pc.items()):
        key = map_key(kernel_id, uid, pc)
        if key in result:
            chain_problems.append(f"duplicate descriptor site key={key}")
            continue
        stats[(kernel_id, uid)]["prefetch"] += 1
        entry = entry_from_slot(slot, None, "utmacctl_prefetch_edge")
        entry.update({
            "trace_kernel_id": kernel_id,
            "unique_function_id": int(uid),
            "pc_hex": pc,
            "opcode": "UTMACCTL",
        })
        result[key] = entry

    # UBLKRED/UBLKCP raw-pointer ops (M2.5): real base from the by-value params
    # struct, keyed on the runtime GMEM operand rather than a descriptor.
    ublk_entries, ublk_stats, ublk_unmatched = {}, {}, []
    if not a.no_ublk:
        span_callbacks = load_ublk_span_callbacks(
            extra, runtime_kernel_names, runtime_rows
        )
        per_ublk = runtime_ublk_operands(
            runtime_rows, UBLK_OPS, span_callbacks
        )
        ublk_entries, ublk_stats, ublk_unmatched = build_ublk_entries(
            per_ublk, struct_ptrs_by_kernel
        )
        for key, entry in ublk_entries.items():
            if key in result:
                chain_problems.append(f"duplicate descriptor/UBLK site key={key}")
                continue
            result[key] = entry

    out = extra / "tma_pc_base_map.json"
    ublkred_diag = count_executed_ublkred_resolved(extra)
    descriptor_sites_total = len(smem_by_pc)
    chain_resolved = sum(s["chain"] for s in stats.values())
    ublkcp_entries = [
        entry for entry in ublk_entries.values()
        if entry["opcode"] == "UBLKCP"
    ]
    ublkcp_by_kernel = {}
    for entry in ublkcp_entries:
        kernel_key = str(entry["trace_kernel_id"])
        summary = ublkcp_by_kernel.setdefault(
            kernel_key,
            {"sites": 0, "dynamic_occurrences": 0, "covered_bytes": set()},
        )
        summary["sites"] += 1
        summary["dynamic_occurrences"] += entry["span_event_count"]
        summary["covered_bytes"].add(entry["covered_bytes"])
    for summary in ublkcp_by_kernel.values():
        summary["covered_bytes"] = sorted(summary["covered_bytes"])
    validation = {
        "sites_total": descriptor_sites_total,
        "chain_resolved": chain_resolved,
        "ambiguous": len(chain_problems),
        "unresolved": len(unresolved_pcs),
        "direct_authority": 0,
        "pool_authority": 0,
        "legacy_direct_candidates": direct_candidate_sites,
        "legacy_direct_conflicts": direct_candidate_conflicts,
        "ublkcp_sites": len(ublkcp_entries),
        "ublkcp_dynamic_occurrences": sum(
            entry["span_event_count"] for entry in ublkcp_entries
        ),
        "ublkcp_by_kernel": ublkcp_by_kernel,
    }
    out.write_text(json.dumps({
        "delta_hex": f"0x{delta:x}",
        "version": 2,
        "key_fields": [
            "trace_kernel_id", "unique_function_id", "pc_hex"
        ],
        "field_order": list(DESC_FIELD_KEYS),
        "ublkred_diagnostic": ublkred_diag,
        "ublk_raw_pointer_fields": [
            "base_hex", "covered_bytes", "tile_bytes", "num_tiles",
            "region_bytes", "gmem_cb_index", "raw_pointer_addressed",
        ],
        "validation_complete": not (
            chain_problems or unresolved_pcs or ublk_unmatched
        ),
        "validation": validation,
        "map": result,
    }, indent=2) + "\n")

    print("struct descriptors discovered per trace kernel:")
    for (kernel_id, uid), slots in sorted(struct_slots_by_kernel.items()):
        bases = {s["base"] for s in slots.values()}
        print(f"  trace_kernel={kernel_id} uid={uid}: "
              f"{len(slots)} slots, {len(bases)} distinct bases")
    print(f"\nruntime TMA-state sites: {len(smem_by_pc)}")
    print(f"kernel-aware mappings: {len(result)} -> {out.name}")
    for (kernel_id, uid) in sorted(stats):
        s = stats[(kernel_id, uid)]
        print(f"  trace_kernel={kernel_id} uid={uid}: "
              f"chain={s['chain']} prefetch={s['prefetch']} "
              f"unresolved={s['none']}")
    print(
        "  descriptor-validation "
        f"sites_total={validation['sites_total']} "
        f"chain_resolved={validation['chain_resolved']} "
        f"ambiguous={validation['ambiguous']} "
        f"unresolved={validation['unresolved']} "
        f"direct_authority={validation['direct_authority']} "
        f"pool_authority={validation['pool_authority']} "
        f"legacy_direct_candidates={validation['legacy_direct_candidates']} "
        f"legacy_direct_conflicts={validation['legacy_direct_conflicts']}"
    )
    if ublkred_diag is not None:
        print(f"\nUBLKRED 1:1 diagnostic (executed sites reaching an in-struct slot): "
              f"{ublkred_diag['resolved_to_struct']}/{ublkred_diag['executed']} "
              f"(expected 0 -> UBLKRED is NOT tensormap-addressed; base stays synthetic)")

    # (3) UBLKRED/UBLKCP raw-pointer base + mock tiling (M2.5) report
    if not a.no_ublk:
        print(f"\nUBLKRED/UBLKCP raw-pointer base+tiling (M2.5): "
              f"{len(ublk_entries)} (uid,pc) mapped")
        for kernel_uid in sorted(ublk_stats):
            s = ublk_stats[kernel_uid]
            print(f"  trace_kernel={kernel_uid[0]} uid={kernel_uid[1]}: "
                  f"UBLKRED={s['UBLKRED']} UBLKCP={s['UBLKCP']}")
        # per (uid,base) region summary — one line per distinct tensor region
        regions = {}
        for key, e in ublk_entries.items():
            region_key = (
                e["trace_kernel_id"],
                e["unique_function_id"],
                e["base_hex"],
            )
            regions.setdefault(region_key, e)
        for (kernel_id, uid, bh), e in sorted(regions.items()):
            print(f"  trace_kernel={kernel_id} uid={uid} base={bh} "
                  f"tile_bytes={e['tile_bytes']} "
                  f"num_tiles={e['num_tiles']} region_bytes={e['region_bytes']} "
                  f"(0x{e['region_bytes']:x}) distinct_addrs={e['distinct_addrs_region']}")
        if ublk_unmatched:
            print(f"  UNMATCHED ({len(ublk_unmatched)}): {ublk_unmatched[:8]}")

    if a.strict:
        problems = []
        problems.extend(chain_problems)
        if unresolved_pcs:
            problems.append(f"{len(unresolved_pcs)} unresolved: {unresolved_pcs[:8]}")
        if ublk_unmatched:
            problems.append(
                f"{len(ublk_unmatched)} UBLK operand failures: "
                f"{ublk_unmatched[:8]}"
            )
        if descriptor_sites_total == 0 or chain_resolved != descriptor_sites_total:
            problems.append(
                f"descriptor chain coverage {chain_resolved}/{descriptor_sites_total}"
            )
        if problems:
            raise SystemExit("STRICT GATE FAILED: " + "; ".join(problems))
        print("\nSTRICT GATE: PASS")
        print(
            "FINAL status=PASS "
            f"sites_total={descriptor_sites_total} "
            f"chain_resolved={chain_resolved} ambiguous=0 unresolved=0 "
            "direct_authority=0 pool_authority=0 "
            f"ublkcp_sites={validation['ublkcp_sites']} "
            f"ublkcp_dynamic_occurrences="
            f"{validation['ublkcp_dynamic_occurrences']} "
            f"ublkcp_by_kernel={json.dumps(ublkcp_by_kernel, sort_keys=True)}"
        )


if __name__ == "__main__":
    main()
