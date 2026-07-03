#!/usr/bin/env python3
"""Build (uid, pc) -> real GMEM base for TMA descriptor ops (TMA_BASE_ADDR.md §2.27).

Unifying key = the SMEM descriptor offset each descriptor-based TMA op carries at runtime
(ci=0 operand, a generic-SMEM address 0xffffff...; low bits = a per-kernel-stable tensor
ID). Same SMEM offset => same tensor => same base, which is what L2 locality needs.

Real base assignment (best-effort, deterministic), per uid:
  1. discover struct descriptor slots + qword0 base by sliding each encode blob over the
     launch-arg params-struct blob (exact 128B match).
  2. resolve each SMEM offset to a base:
       (a) DIRECT: SMEM offset == a struct descriptor slot -> that slot's base.
       (b) PREFETCH CHAIN: UTMACCTL.PF maps (param source offset -> SMEM dest offset);
           param_source - 0x30 == struct slot -> base. Ties param<->SMEM coordinate
           systems (§2.27).
       (c) POOL: unresolved SMEM offsets get a deterministic base from the kernel's base
           pool (stable, so same SMEM offset always yields the same base -> locality kept).
Output: extra_info/tma_pc_base_map.json  { "uid:pc": {base_hex, smem_offset_hex, source} }

Usage: python3 build_tma_pc_base_map.py --traces <dir> [--struct-desc-delta 0x30]
"""
import argparse, csv, json, struct
from collections import defaultdict
from pathlib import Path

DESC_OPS = ("UTMALDG", "UTMASTG", "UTMAPF", "UTMAREDG")  # ops we map (UTMACCTL is the linker)


def parse_int(x):
    x = (str(x) if x is not None else "").strip()
    return int(x, 0) if x else 0


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
            out.append((r.get("dump_id"), base, data))
    return out


def struct_blob_by_uid(extra):
    """largest launch arg per uid = the by-value params struct."""
    best = {}
    p = extra / "tma_launch_param_dump.csv"
    if not p.exists():
        return best
    with p.open(newline="") as fh:
        for r in csv.DictReader(fh):
            uid = str(r.get("unique_function_id"))
            sz = parse_int(r.get("arg_size"))
            if uid not in best or sz > best[uid][0]:
                best[uid] = (sz, r.get("blob_path"), parse_int(r.get("param_offset_hex")))
    return best


def discover_struct_descriptors(struct_bytes, encode_blobs):
    """inner struct offset -> base, for every exact 128B encode-blob placement."""
    slots = {}
    for did, base, data in encode_blobs:
        if len(data) < 128:
            continue
        needle = data[:128]
        start = 0
        while True:
            idx = struct_bytes.find(needle, start)
            if idx < 0:
                break
            slots[idx] = base
            start = idx + 8
    return slots


def runtime_smem_offsets(extra, opcodes):
    """(uid, pc) -> the SMEM descriptor offset (low bits of ci=0 0xffffff.. operand)."""
    out = {}
    seen_multi = defaultdict(set)
    p = extra / "tma_runtime_operand_debug.jsonl"
    if not p.exists():
        return out, seen_multi
    with p.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            op = str(r.get("opcode", "")).split(".")[0]
            if op not in opcodes:
                continue
            if r.get("callback_index") != 0:
                continue
            fla = r.get("first_lane_addr")
            if not (isinstance(fla, int) and fla > 0xFFFFFFFF00000000):
                continue
            uid = str(r.get("unique_function_id"))
            pc = r.get("pc_hex")
            off = fla & 0xFFFF
            seen_multi[(uid, pc)].add(off)
            out[(uid, pc)] = off  # deterministic per pc (verified); last wins
    return out, seen_multi


def utmacctl_chain(extra, delta, struct_slots_by_uid):
    """uid -> {SMEM dest offset -> base}, via UTMACCTL param_source-delta == struct slot.
    Uses runtime SMEM dest per UTMACCTL pc + def-chain param sources for that pc."""
    # runtime SMEM dest per (uid, pc)
    dest = {}
    p = extra / "tma_runtime_operand_debug.jsonl"
    if p.exists():
        with p.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not str(r.get("opcode", "")).startswith("UTMACCTL"):
                    continue
                if r.get("callback_index") != 0:
                    continue
                fla = r.get("first_lane_addr")
                if isinstance(fla, int) and fla > 0xFFFFFFFF00000000:
                    dest[(str(r.get("unique_function_id")), r.get("pc_hex"))] = fla & 0xFFFF
    # param sources per (uid, pc) from prefetch_sites
    off_json = extra / "tma_descriptor_offsets.json"
    src = defaultdict(set)
    if off_json.exists():
        data = json.loads(off_json.read_text())
        for s in data.get("prefetch_sites", []):
            uid = str(s.get("unique_function_id"))
            pc = s.get("pc_hex")
            o = s.get("tensormap_offset_hex")
            if o is not None:
                src[(uid, pc)].add(parse_int(o))
    smem_to_base = defaultdict(dict)
    for (uid, pc), sm in dest.items():
        slots = struct_slots_by_uid.get(uid, {})
        for psrc in src.get((uid, pc), ()):
            ss = psrc - delta
            if ss in slots:
                smem_to_base[uid][sm] = slots[ss]
                break
    return smem_to_base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--struct-desc-delta", default="0x30",
                    help="param_source - delta == struct descriptor offset (§2.27, =0x30)")
    a = ap.parse_args()
    delta = parse_int(a.struct_desc_delta)
    extra = Path(a.traces).resolve() / "extra_info"

    encode_blobs = load_encode_blobs(extra)
    structs = struct_blob_by_uid(extra)
    struct_slots_by_uid = {}
    for uid, (sz, blob, argoff) in structs.items():
        bp = Path(blob) if blob else None
        if bp and bp.exists():
            struct_slots_by_uid[uid] = discover_struct_descriptors(bp.read_bytes(),
                                                                   encode_blobs)
    chain = utmacctl_chain(extra, delta, struct_slots_by_uid)
    smem_by_pc, multi = runtime_smem_offsets(extra, DESC_OPS)

    # per-uid base pool for fallback (deterministic order)
    pool = {uid: sorted(set(slots.values())) for uid, slots in struct_slots_by_uid.items()}

    result = {}
    stats = defaultdict(lambda: {"direct": 0, "chain": 0, "pool": 0, "none": 0})
    # stable pool assignment: map each unresolved SMEM offset -> pool[idx] round-robin
    pool_assign = defaultdict(dict)
    for (uid, pc), sm in sorted(smem_by_pc.items()):
        base, source = None, None
        slots = struct_slots_by_uid.get(uid, {})
        if sm in slots:                       # (a) direct
            base, source = slots[sm], "direct"
        elif sm in chain.get(uid, {}):        # (b) prefetch chain
            base, source = chain[uid][sm], "chain"
        elif pool.get(uid):                   # (c) deterministic pool
            pa = pool_assign[uid]
            if sm not in pa:
                pa[sm] = pool[uid][len(pa) % len(pool[uid])]
            base, source = pa[sm], "pool"
        if base is None:
            stats[uid]["none"] += 1
            continue
        stats[uid][source] += 1
        result[f"{uid}:{pc}"] = {"base_hex": f"0x{base:x}",
                                 "smem_offset_hex": f"0x{sm:x}",
                                 "source": source}

    out = extra / "tma_pc_base_map.json"
    out.write_text(json.dumps({"delta_hex": f"0x{delta:x}", "map": result}, indent=2) + "\n")

    print(f"struct descriptors discovered per uid:")
    for uid, slots in sorted(struct_slots_by_uid.items()):
        print(f"  uid{uid}: {len(slots)} slots, {len(set(slots.values()))} distinct bases")
    print(f"\n(uid,pc) mapped: {len(result)}  -> {out.name}")
    for uid in sorted(stats):
        s = stats[uid]
        print(f"  uid{uid}: direct={s['direct']} chain={s['chain']} pool={s['pool']} "
              f"unresolved={s['none']}")
    # locality integrity: each SMEM offset must map to exactly one base
    bad = defaultdict(set)
    for k, v in result.items():
        uid = k.split(":")[0]
        bad[(uid, v["smem_offset_hex"])].add(v["base_hex"])
    conflicts = {k: v for k, v in bad.items() if len(v) > 1}
    print(f"\nlocality check (one base per SMEM offset): "
          f"{'OK' if not conflicts else 'CONFLICTS: %s' % conflicts}")


if __name__ == "__main__":
    main()
