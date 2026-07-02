#!/usr/bin/env python3
"""Path A Gate verifier (TMA_BASE_ADDR.md §2.23 / §2.24).

The tracer, run with ENABLE_TMA_DESC=1, dumps each cuLaunchKernel argument's full bytes
to extra_info/launch_param_blobs/*.bin and metadata to tma_launch_param_dump.csv
(device_id,kernel_id,unique_function_id,arg_index,param_offset_hex,arg_size,arg_ptr_hex,
blob_path,qword0..3). Persistent FA3 kernels pass ONE big by-value params struct whose
tensor bases sit at various inner offsets, so we scan every 8-aligned qword of each blob
for a value that equals a real base from tensor_map_encode_dump.csv.

Outputs:
  - prints, per (uid, arg), every inner byte offset whose qword == a real base
  - param_block_offset = param_offset(arg) + inner_offset; joins that to the SASS static
    tensormap_offset (tma_descriptor_offsets.json) to map (uid,pc) -> base
  - writes tma_launch_param_join.json

Usage:
  python3 verify_tma_launch_param.py --traces <dir>
"""

import argparse
import csv
import json
import struct
from collections import defaultdict
from pathlib import Path


def parse_int(x):
    if x is None:
        return 0
    x = str(x).strip()
    if not x:
        return 0
    return int(x, 0)


def load_encode_bases(extra):
    path = extra / "tensor_map_encode_dump.csv"
    bases = {}
    if not path.exists():
        return bases, path
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            b = parse_int(row.get("global_address_hex") or row.get("qword0_hex"))
            if b:
                bases.setdefault(b, []).append(
                    (row.get("dump_id"), row.get("box_dim")))
    return bases, path


def load_launch_dump(extra):
    path = extra / "tma_launch_param_dump.csv"
    rows = []
    if not path.exists():
        return rows, path
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows, path


def load_offsets(extra):
    """Return {(uid, pc_hex): tensormap_offset_int} and offset->[(uid,pc)]."""
    path = extra / "tma_descriptor_offsets.json"
    site_offsets = {}
    offset_to_sites = defaultdict(list)
    if not path.exists():
        return site_offsets, offset_to_sites, path
    data = json.loads(path.read_text())
    for s in data.get("sites", []):
        if not s.get("resolved"):
            continue
        off_hex = s.get("tensormap_offset_hex")
        if off_hex is None:
            continue
        uid = str(s.get("unique_function_id"))
        pc = s.get("pc_hex")
        off = parse_int(off_hex)
        site_offsets[(uid, pc)] = off
        offset_to_sites[off].append((uid, pc))
    return site_offsets, offset_to_sites, path


def scan_blob_for_bases(blob_path, bases):
    """Return list of (inner_offset, base) for every 8-aligned qword equal to a base."""
    hits = []
    p = Path(blob_path)
    if not p.exists():
        return hits
    data = p.read_bytes()
    n = len(data) - (len(data) % 8)
    for off in range(0, n, 8):
        (v,) = struct.unpack_from("<Q", data, off)
        if v in bases:
            hits.append((off, v))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--show", type=int, default=60)
    args = ap.parse_args()

    extra = Path(args.traces).resolve() / "extra_info"
    dump_rows, dump_path = load_launch_dump(extra)
    if not dump_rows:
        raise SystemExit(
            f"missing/empty {dump_path}\nRun the tracer with ENABLE_TMA_DESC=1 first.")

    bases, enc_path = load_encode_bases(extra)
    site_offsets, offset_to_sites, off_path = load_offsets(extra)

    print(f"encode bases ({enc_path.name}): {len(bases)} distinct")
    for b, rows in sorted(bases.items()):
        print(f"    0x{b:x}  x{len(rows)}  box={rows[0][1]}")
    print(f"\nlaunch-arg rows ({dump_path.name}): {len(dump_rows)}")
    print(f"resolved (uid,pc) offsets ({off_path.name}): {len(site_offsets)}")

    # For each arg blob, find every inner offset that holds a real base.
    # base_locs[uid] = list of (param_block_offset, base, arg_index, inner_off)
    base_locs = defaultdict(list)
    print("\nbase locations inside launch-arg blobs (uid, arg, inner_off -> base):")
    for r in dump_rows:
        uid = r.get("unique_function_id")
        ai = parse_int(r.get("arg_index"))
        arg_off = parse_int(r.get("param_offset_hex"))
        blob = r.get("blob_path")
        if not blob:
            continue
        hits = scan_blob_for_bases(blob, bases)
        for inner, b in hits:
            pbo = arg_off + inner
            base_locs[uid].append((pbo, b, ai, inner))
    shown = 0
    for uid in sorted(base_locs, key=lambda x: int(x)):
        for pbo, b, ai, inner in base_locs[uid]:
            if shown < args.show:
                print(f"    uid={uid} arg={ai} inner=0x{inner:x} "
                      f"param_block_off=0x{pbo:x} -> base=0x{b:x}")
                shown += 1
    total_locs = sum(len(v) for v in base_locs.values())
    print(f"  total base locations found: {total_locs}")

    # Join: param_block_offset (arg_off + inner) == SASS tensormap_offset, same uid.
    join = {}
    for uid, locs in base_locs.items():
        for pbo, b, ai, inner in locs:
            for (suid, pc) in offset_to_sites.get(pbo, []):
                if suid == str(uid):
                    join[(suid, pc)] = {
                        "base_hex": f"0x{b:x}",
                        "param_block_offset_hex": f"0x{pbo:x}",
                        "arg_index": ai,
                        "inner_offset_hex": f"0x{inner:x}",
                    }

    print(f"\njoined (uid,pc) -> base via param_block_offset==tensormap_offset: "
          f"{len(join)}")
    for (uid, pc), v in sorted(join.items())[:args.show]:
        print(f"    uid={uid} pc={pc} -> base={v['base_hex']} "
              f"(param_block_off {v['param_block_offset_hex']}, arg {v['arg_index']})")

    out = extra / "tma_launch_param_join.json"
    out.write_text(json.dumps(
        {"join": {f"{u}:{pc}": v for (u, pc), v in join.items()}}, indent=2) + "\n")
    print(f"\nwrote {out}")

    # Diagnostics to help align coordinate systems if join is empty.
    if not join and total_locs:
        sass_offsets = sorted(set(offset_to_sites.keys()))
        print("\nno offset match. Compare the two offset sets:")
        print("  base param_block_offsets found:",
              sorted({f"0x{pbo:x}" for locs in base_locs.values()
                      for pbo, _, _, _ in locs})[:20])
        print("  SASS tensormap_offsets (sample):",
              [f"0x{o:x}" for o in sass_offsets[:20]])

    print()
    if join:
        print("=> PASS: recovered (uid,pc) -> real base from host launch args. "
              "Build the real-address mover on tma_launch_param_join.json.")
    elif total_locs:
        print("=> PARTIAL: bases ARE present inside the by-value param struct(s), but "
              "param_block_offset != any tensormap_offset. The arg-pack offset model or "
              "the SASS offset base differs; align them (see the two offset lists above).")
    else:
        print("=> INVESTIGATE: no base value found inside any launch-arg blob. Args may "
              "be by-pointer; dump/deref the pointer targets offline.")


if __name__ == "__main__":
    main()
