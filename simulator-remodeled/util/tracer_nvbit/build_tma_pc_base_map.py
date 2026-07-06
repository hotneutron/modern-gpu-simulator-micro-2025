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


def struct_blob_by_uid(extra):
    """largest launch arg per uid = the by-value params struct. The CSV blob_path is an
    absolute path from the tracer host, so rebuild it relative to THIS extra_info dir
    (launch_param_blobs/<basename>) for portability across machines."""
    best = {}
    p = extra / "tma_launch_param_dump.csv"
    if not p.exists():
        return best
    with p.open(newline="") as fh:
        for r in csv.DictReader(fh):
            uid = str(r.get("unique_function_id"))
            sz = parse_int(r.get("arg_size"))
            raw = r.get("blob_path") or ""
            local = extra / "launch_param_blobs" / Path(raw).name if raw else None
            if uid not in best or sz > best[uid][0]:
                best[uid] = (sz, str(local) if local else None,
                             parse_int(r.get("param_offset_hex")))
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
            if not is_smem_descriptor_addr(fla):
                continue
            uid = str(r.get("unique_function_id"))
            pc = r.get("pc_hex")
            off = smem_offset_of(fla)
            seen_multi[(uid, pc)].add(off)
            out[(uid, pc)] = off  # deterministic per pc (verified); last wins
    return out, seen_multi


def utmacctl_chain(extra, delta, struct_slots_by_uid):
    """Resolve UTMACCTL.PF prefetch sites two ways from one pass:
      - smem_to_slot[uid][SMEM dest offset] = slot dict  (feeds the LDG chain path)
      - prefetch_by_pc[(uid,pc)]            = slot dict  (per-pc prefetch emit)
    via UTMACCTL param_source-delta == struct slot (§2.27). Only RUNTIME-executed
    UTMACCTL pcs are used (static-only pcs are dead sites), so every emitted prefetch
    entry corresponds to an instruction the simulator will actually issue."""
    # runtime SMEM dest per (uid, pc) for executed UTMACCTL only
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
                if is_smem_descriptor_addr(fla):
                    dest[(str(r.get("unique_function_id")), r.get("pc_hex"))] = smem_offset_of(fla)
    # param sources per (uid, pc) from prefetch_sites (static def-chain)
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
    smem_to_slot = defaultdict(dict)
    prefetch_by_pc = {}
    for (uid, pc), sm in dest.items():
        slots = struct_slots_by_uid.get(uid, {})
        for psrc in src.get((uid, pc), ()):
            ss = psrc - delta
            if ss in slots:
                smem_to_slot[uid][sm] = slots[ss]
                prefetch_by_pc[(uid, pc)] = slots[ss]
                break
    return smem_to_slot, prefetch_by_pc


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
                    help="exit nonzero if any descriptor-op (uid,pc) is unresolved or "
                         "falls back to the pool (enforces 100%% exact mapping)")
    ap.add_argument("--allow-pool", action="store_true",
                    help="permit deterministic pool fallback (default: pool counts as a "
                         "failure under --strict; kept only as a locality safety net)")
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
    chain, prefetch_by_pc = utmacctl_chain(extra, delta, struct_slots_by_uid)
    smem_by_pc, multi = runtime_smem_offsets(extra, DESC_OPS)

    # per-uid base pool for fallback: deterministic list of slot dicts sorted by base.
    pool = {}
    for uid, slots in struct_slots_by_uid.items():
        by_base = {}
        for slot in slots.values():
            by_base.setdefault(slot["base"], slot)
        pool[uid] = [by_base[b] for b in sorted(by_base)]

    result = {}
    stats = defaultdict(lambda: {"direct": 0, "chain": 0, "pool": 0, "none": 0,
                                 "prefetch": 0})
    unresolved_pcs = []
    pool_pcs = []
    # stable pool assignment: map each unresolved SMEM offset -> pool[idx] round-robin
    pool_assign = defaultdict(dict)
    # (1) descriptor LOAD/STORE ops keyed on runtime SMEM descriptor offset.
    for (uid, pc), sm in sorted(smem_by_pc.items()):
        slot, source = None, None
        slots = struct_slots_by_uid.get(uid, {})
        if sm in slots:                       # (a) direct
            slot, source = slots[sm], "direct"
        elif sm in chain.get(uid, {}):        # (b) prefetch chain
            slot, source = chain[uid][sm], "chain"
        elif pool.get(uid):                   # (c) deterministic pool
            pa = pool_assign[uid]
            if sm not in pa:
                pa[sm] = pool[uid][len(pa) % len(pool[uid])]
            slot, source = pa[sm], "pool"
        if slot is None:
            stats[uid]["none"] += 1
            unresolved_pcs.append(f"{uid}:{pc}(smem=0x{sm:x})")
            continue
        stats[uid][source] += 1
        if source == "pool":
            pool_pcs.append(f"{uid}:{pc}(smem=0x{sm:x})")
        result[f"{uid}:{pc}"] = entry_from_slot(slot, sm, source)

    # (2) UTMACCTL.PF prefetch ops keyed per-pc (so the simulator knows which tensor each
    # prefetch warms). Not part of smem_by_pc (different opcode); emit separately.
    for (uid, pc), slot in sorted(prefetch_by_pc.items()):
        key = f"{uid}:{pc}"
        if key in result:
            continue  # never overwrite a LDG/STG mapping
        stats[uid]["prefetch"] += 1
        result[key] = entry_from_slot(slot, None, "prefetch")

    out = extra / "tma_pc_base_map.json"
    ublkred_diag = count_executed_ublkred_resolved(extra)
    out.write_text(json.dumps({
        "delta_hex": f"0x{delta:x}",
        "field_order": list(DESC_FIELD_KEYS),
        "ublkred_diagnostic": ublkred_diag,
        "map": result,
    }, indent=2) + "\n")

    print(f"struct descriptors discovered per uid:")
    for uid, slots in sorted(struct_slots_by_uid.items()):
        bases = {s["base"] for s in slots.values()}
        print(f"  uid{uid}: {len(slots)} slots, {len(bases)} distinct bases")
    print(f"\nruntime SMEM offsets: {len(smem_by_pc)} (uid,pc) across "
          f"{len(set(u for u, _ in smem_by_pc))} uids")
    # per-uid: how many descriptor-op pcs have a SMEM offset, and do their offsets fall in
    # the discovered struct slots / chain / neither?
    by_uid_pc = defaultdict(list)
    for (uid, pc), sm in smem_by_pc.items():
        by_uid_pc[uid].append(sm)
    for uid in sorted(by_uid_pc, key=lambda x: int(x)):
        offs = sorted(set(by_uid_pc[uid]))
        slots = struct_slots_by_uid.get(uid, {})
        ch = chain.get(uid, {})
        d = [o for o in offs if o in slots]
        c = [o for o in offs if o not in slots and o in ch]
        n = [o for o in offs if o not in slots and o not in ch]
        print(f"  uid{uid}: smem_offsets={[hex(o) for o in offs]}")
        print(f"         struct_slots={[hex(o) for o in sorted(slots)]} chain_keys={[hex(o) for o in sorted(ch)]}")
        print(f"         direct={[hex(o) for o in d]} chain={[hex(o) for o in c]} pool/none={[hex(o) for o in n]}")
    print(f"\n(uid,pc) mapped: {len(result)}  -> {out.name}")
    for uid in sorted(stats):
        s = stats[uid]
        print(f"  uid{uid}: direct={s['direct']} chain={s['chain']} "
              f"prefetch={s['prefetch']} pool={s['pool']} unresolved={s['none']}")
    if ublkred_diag is not None:
        print(f"\nUBLKRED 1:1 diagnostic (executed sites reaching an in-struct slot): "
              f"{ublkred_diag['resolved_to_struct']}/{ublkred_diag['executed']} "
              f"(expected 0 -> UBLKRED is NOT tensormap-addressed; base stays synthetic)")
    # locality integrity: each SMEM offset must map to exactly one base
    bad = defaultdict(set)
    for k, v in result.items():
        if "smem_offset_hex" not in v:
            continue  # prefetch entries have no smem offset
        uid = k.split(":")[0]
        bad[(uid, v["smem_offset_hex"])].add(v["base_hex"])
    conflicts = {k: v for k, v in bad.items() if len(v) > 1}
    print(f"\nlocality check (one base per SMEM offset): "
          f"{'OK' if not conflicts else 'CONFLICTS: %s' % conflicts}")

    # STRICT GATE: enforce that every descriptor LOAD/STORE site resolved exactly (no
    # pool, no unresolved). UTMACCTL prefetch + UBLKRED are not gated here (prefetch is
    # emitted only when resolved; UBLKRED is intentionally synthetic in M1).
    if a.strict:
        problems = []
        if unresolved_pcs:
            problems.append(f"{len(unresolved_pcs)} unresolved: {unresolved_pcs[:8]}")
        if pool_pcs and not a.allow_pool:
            problems.append(f"{len(pool_pcs)} pool-fallback: {pool_pcs[:8]}")
        if conflicts:
            problems.append(f"{len(conflicts)} locality conflicts")
        if problems:
            raise SystemExit("STRICT GATE FAILED: " + "; ".join(problems))
        print("\nSTRICT GATE: PASS (all descriptor LDG/STG sites exact, no pool)")


if __name__ == "__main__":
    main()
