#!/usr/bin/env python3
"""Diagnostic: where does the base sit relative to the SASS tensormap_offset?

For each resolved (uid,pc) with a static tensormap_offset, look INSIDE the launch-arg
struct blob at that offset and report the 16 qwords there, flagging which qword equals a
real encode base. This tells us the exact base-field offset inside the descriptor struct
(qword0? +0x10?) per uid, instead of guessing a single constant shift.

Usage:
  python3 diag_tma_desc_layout.py --traces <dir> [--uid 8] [--limit 40]
"""
import argparse, csv, json, struct
from collections import defaultdict
from pathlib import Path


def parse_int(x):
    x = (str(x) if x is not None else "").strip()
    return int(x, 0) if x else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--uid")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--window", type=int, default=32, help="qwords to show at the offset")
    a = ap.parse_args()
    extra = Path(a.traces).resolve() / "extra_info"

    bases = set()
    with (extra / "tensor_map_encode_dump.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            b = parse_int(r.get("global_address_hex") or r.get("qword0_hex"))
            if b:
                bases.add(b)

    # one blob per (uid): the big by-value params struct is arg0 for persistent kernels
    blob_by_uid = {}
    with (extra / "tma_launch_param_dump.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            uid = r.get("unique_function_id")
            sz = parse_int(r.get("arg_size"))
            # pick the largest arg per uid (the params struct)
            prev = blob_by_uid.get(uid)
            if prev is None or sz > prev[0]:
                blob_by_uid[uid] = (sz, r.get("blob_path"),
                                    parse_int(r.get("param_offset_hex")))

    data = json.loads((extra / "tma_descriptor_offsets.json").read_text())
    sites = [s for s in data["sites"]
             if s.get("resolved") and s.get("tensormap_offset_hex")]
    if a.uid:
        sites = [s for s in sites if str(s["unique_function_id"]) == a.uid]

    # unique (uid, tensormap_offset)
    seen = set()
    uniq = []
    for s in sites:
        k = (str(s["unique_function_id"]), s["tensormap_offset_hex"])
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    uniq.sort(key=lambda s: (int(s["unique_function_id"]),
                             int(s["tensormap_offset_hex"], 16)))

    print(f"encode bases: {len(bases)}; sites(uniq offsets): {len(uniq)}")
    hist = defaultdict(int)
    shown = 0
    for s in uniq:
        uid = str(s["unique_function_id"])
        toff = int(s["tensormap_offset_hex"], 16)
        info = blob_by_uid.get(uid)
        if not info:
            continue
        _, blob_path, arg_off = info
        p = Path(blob_path)
        if not p.exists():
            continue
        buf = p.read_bytes()
        # struct-internal offset of the descriptor = tensormap_offset - arg_param_offset
        rel = toff - arg_off
        # scan a window and find where a base is relative to rel
        found = []
        for q in range(0, a.window):
            off = rel + q * 8
            if 0 <= off and off + 8 <= len(buf):
                (v,) = struct.unpack_from("<Q", buf, off)
                if v in bases:
                    found.append((q * 8, v))
        for delta, b in found:
            hist[delta] += 1
        if shown < a.limit:
            fs = ", ".join(f"+0x{d:x}=0x{b:x}" for d, b in found) or "(no base in window)"
            print(f"  uid{uid} pc={s['pc_hex']} toff=0x{toff:x} rel=0x{rel:x}: {fs}")
            shown += 1
    print("\nbase-relative-to-tensormap_offset histogram (delta -> count):")
    for d in sorted(hist):
        print(f"    +0x{d:x}: {hist[d]}")


if __name__ == "__main__":
    main()
