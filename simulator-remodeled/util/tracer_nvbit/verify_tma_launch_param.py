#!/usr/bin/env python3
"""Path A Gate verifier (TMA_BASE_ADDR.md §2.23).

The tracer, run with ENABLE_TMA_DESC=1, dumps the host launch-argument buffer to
extra_info/tma_launch_param_dump.csv: for each kernel argument, its packed param-block
offset and the first 16 qwords read from the *host* argument pointer (crash-free —
no device load). This is Path A: recover the real base from the host by-value
CUtensorMap arguments instead of the (impossible) device read.

This script answers two questions:
  1. BASE PRESENT?  Does any argument's qword0 equal one of the true bases from
     tensor_map_encode_dump.csv (global_address_hex)? If so, that argument IS a
     by-value CUtensorMap and we have the base on the host.
  2. JOIN?  Does an argument's param_offset match a (uid,pc)'s static
     tensormap_offset from tma_descriptor_offsets.json? If so, we can map each
     executed UTMALDG pc -> that argument -> its qword0 base.

Emits tma_launch_param_join.json: (uid, pc_hex) -> {base_hex, param_offset_hex}.

Usage:
  python3 verify_tma_launch_param.py --traces <dir>
"""

import argparse
import csv
import json
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
    """Return {(uid, pc_hex): tensormap_offset_int} for resolved descriptor sites."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    extra = Path(args.traces).resolve() / "extra_info"
    dump_rows, dump_path = load_launch_dump(extra)
    if not dump_rows:
        raise SystemExit(
            f"missing/empty {dump_path}\n"
            "Run the tracer with ENABLE_TMA_DESC=1 first.")

    bases, enc_path = load_encode_bases(extra)
    site_offsets, offset_to_sites, off_path = load_offsets(extra)

    print(f"encode bases ({enc_path.name}): {len(bases)} distinct")
    for b, rows in sorted(bases.items()):
        print(f"    0x{b:x}  x{len(rows)}  box={rows[0][1]}")
    print(f"\nlaunch-arg rows ({dump_path.name}): {len(dump_rows)}")
    print(f"resolved (uid,pc) offsets ({off_path.name}): {len(site_offsets)}")

    # 1) which args carry a real base in qword0?
    base_args = []  # (uid, arg_index, param_offset, base)
    for r in dump_rows:
        q0 = parse_int(r.get("qword0_hex"))
        if q0 in bases:
            base_args.append((
                r.get("unique_function_id"),
                parse_int(r.get("arg_index")),
                parse_int(r.get("param_offset_hex")),
                q0,
            ))
    print(f"\nargs whose qword0 == a real base: {len(base_args)}")
    for uid, ai, off, b in base_args[:args.show]:
        print(f"    uid={uid} arg={ai} param_offset=0x{off:x} base=0x{b:x}")

    if not base_args:
        # Show what qword0 looks like so we can tell by-value vs pointer.
        print("\nno arg matched a base. First qwords of each arg (to classify):")
        for r in dump_rows[:args.show]:
            print(f"    uid={r.get('unique_function_id')} "
                  f"arg={r.get('arg_index')} "
                  f"off={r.get('param_offset_hex')} size={r.get('arg_size')} "
                  f"q0={r.get('qword0_hex')} q1={r.get('qword1_hex')} "
                  f"q2={r.get('qword2_hex')}")
        print("\n=> If q0 looks like a device pointer (0x7f...), args are BY-POINTER: "
              "follow that pointer with a host cuMemcpyDtoH offline. If q0 is a small/"
              "zero value, the offset packing is off — compare param_offset to the "
              "tensormap_offset set below.")

    # 2) join param_offset (per uid) to the static tensormap_offset (uid,pc)
    join = {}
    for uid, ai, off, b in base_args:
        for (suid, pc) in offset_to_sites.get(off, []):
            if suid == str(uid):
                join[(suid, pc)] = {"base_hex": f"0x{b:x}",
                                    "param_offset_hex": f"0x{off:x}",
                                    "arg_index": ai}

    print(f"\njoined (uid,pc) -> base via matching param_offset==tensormap_offset: "
          f"{len(join)}")
    for (uid, pc), v in sorted(join.items())[:args.show]:
        print(f"    uid={uid} pc={pc} -> base={v['base_hex']} "
              f"(offset {v['param_offset_hex']}, arg {v['arg_index']})")

    out = extra / "tma_launch_param_join.json"
    out.write_text(json.dumps(
        {"join": {f"{u}:{pc}": v for (u, pc), v in join.items()}},
        indent=2) + "\n")
    print(f"\nwrote {out}")

    print()
    if join:
        print("=> PASS: recovered (uid,pc) -> real base from the host launch args. "
              "Build the real-address mover on tma_launch_param_join.json.")
    elif base_args:
        print("=> PARTIAL: bases are present in the launch args, but param_offset did "
              "not line up with any tensormap_offset. Check the offset packing (arg "
              "alignment) vs the SASS c[0x0][K]+UIADD3 offsets.")
    else:
        print("=> INVESTIGATE: no arg exposed a base in qword0 (see the qword preview "
              "above) — args are likely by-pointer; add an offline host deref of the "
              "pointer value.")


if __name__ == "__main__":
    main()
