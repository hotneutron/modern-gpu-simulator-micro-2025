#!/usr/bin/env python3
"""Diagnostic: is a full 128B CUtensorMap sitting at param_base + tensormap_offset?

The SASS def-chain proves each UTMALDG computes operand `param_base + tensormap_offset`
and hands it to the TMA engine AS THE DESCRIPTOR ADDRESS. So the launch-arg params struct
should contain, at that same offset, the identical 128B CUtensorMap the host encoded.

Rather than scan for a base value (noisy — bases repeat every 0xc0 in the struct), this
compares the full 128B at each (uid, tensormap_offset) against every encoded descriptor
blob (tensor_map_encode_blobs/*.bin). An exact 128B match is unambiguous: that site's
tensor == that encode dump, whose qword0 is the real base. Falls back to reporting the
qword0 at the offset if no encode blob dir exists.

Usage:
  python3 diag_tma_desc_layout.py --traces <dir> [--uid 8] [--limit 40]
"""
import argparse, csv, json, struct
from collections import defaultdict
from pathlib import Path


def parse_int(x):
    x = (str(x) if x is not None else "").strip()
    return int(x, 0) if x else 0


def load_encode_blobs(extra):
    """Return list of (dump_id, base, box, 128B-bytes) from the encode dump + blobs."""
    out = []
    csv_path = extra / "tensor_map_encode_dump.csv"
    if not csv_path.exists():
        return out
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            base = parse_int(r.get("global_address_hex") or r.get("qword0_hex"))
            blob_rel = r.get("blob_path") or ""
            # blob_path is like "traces/extra_info/tensor_map_encode_blobs/0.bin"
            cand = extra / "tensor_map_encode_blobs" / (str(r.get("dump_id")) + ".bin")
            data = cand.read_bytes() if cand.exists() else b""
            out.append((r.get("dump_id"), base, r.get("box_dim"), data))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--uid")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--window", type=int, default=16, help="qwords to show at the offset (16=128B, one CUtensorMap)")
    a = ap.parse_args()
    extra = Path(a.traces).resolve() / "extra_info"

    bases = set()
    with (extra / "tensor_map_encode_dump.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            b = parse_int(r.get("global_address_hex") or r.get("qword0_hex"))
            if b:
                bases.add(b)
    encode_blobs = load_encode_blobs(extra)
    have_encode_blobs = any(len(d) >= 128 for _, _, _, d in encode_blobs)

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

    print(f"encode bases: {len(bases)}; sites(uniq offsets): {len(uniq)}; "
          f"encode blobs available: {have_encode_blobs}")
    exact = 0
    q0_is_base = 0
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
        rel = toff - arg_off  # struct-internal byte offset of the descriptor
        if rel < 0 or rel + 128 > len(buf):
            if shown < a.limit:
                print(f"  uid{uid} pc={s['pc_hex']} toff=0x{toff:x}: offset outside "
                      f"struct (size 0x{len(buf):x})")
                shown += 1
            continue
        desc = buf[rel:rel + 128]
        (q0,) = struct.unpack_from("<Q", desc, 0)
        # exact 128B match against an encode blob?
        match_id, match_base = None, None
        if have_encode_blobs:
            for did, base, box, data in encode_blobs:
                if len(data) >= 128 and data[:128] == desc:
                    match_id, match_base = did, base
                    break
        if match_id is not None:
            exact += 1
        if q0 in bases:
            q0_is_base += 1
        if shown < a.limit:
            tag = (f"EXACT encode#{match_id} base=0x{match_base:x}"
                   if match_id is not None else
                   ("q0=0x%x %s" % (q0, "(is a base)" if q0 in bases else "(NOT a base)")))
            print(f"  uid{uid} pc={s['pc_hex']} toff=0x{toff:x} rel=0x{rel:x}: {tag}")
            shown += 1
    print(f"\nsummary: exact 128B descriptor matches = {exact}/{len(uniq)}; "
          f"qword0==base = {q0_is_base}/{len(uniq)}")
    if exact:
        print("=> the params struct holds the by-value CUtensorMap at "
              "param_base+tensormap_offset; qword0 there is the real base. "
              "Join is exact — build the mover on it.")
    elif q0_is_base:
        print("=> qword0 at the offset is a real base (encode blobs absent for full "
              "compare). Use base=qword0 at param_base+tensormap_offset.")
    else:
        print("=> neither exact match nor qword0-base at the offset. The descriptor is "
              "NOT inlined at param_base+tensormap_offset; re-examine the SASS operand "
              "(the offset may point at a pointer to the descriptor, not the descriptor).")


if __name__ == "__main__":
    main()
