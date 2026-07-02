#!/usr/bin/env python3
"""SPIKE 6 device fact-check verifier (TMA_BASE_ADDR.md §2.18).

The tracer, run with ENABLE_TMA_DESC=1 TMA_DESC_FACTCHECK=1, device-reads 128B at each
executed UTMALDG/UTMASTG descriptor VA (the generic-SMEM tensormap address) and writes
qword0/qword1 samples to extra_info/tma_desc_factcheck.csv.

This script answers the one open question from §2.17/§2.18:
    Does the SMEM-staged descriptor the HW actually reads carry the real base?
i.e. is each sampled qword0 equal to one of the true bases (global_address_hex /
qword0_hex) captured on the host in tensor_map_encode_dump.csv?

PASS  => the device SMEM read yields the real base. Path D1 (read the descriptor at
         issue) is confirmed; build the real-address mover on it.
FAIL  => qword0 is not a known base (e.g. it is coords-modified, swizzled, or the VA
         we read is wrong). Report what qword0 looks like so we pick D2 (global
         original) or fix the operand choice.

Usage:
  python3 verify_tma_desc_factcheck.py --traces <dir>
"""

import argparse
import csv
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
    """Return dict base_int -> list of (dump_id, box_dim) from the host encode dump."""
    path = extra / "tensor_map_encode_dump.csv"
    bases = defaultdict(list)
    if not path.exists():
        return bases, path
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            b = parse_int(row.get("global_address_hex") or row.get("qword0_hex"))
            if b:
                bases[b].append((row.get("dump_id"), row.get("box_dim")))
    return bases, path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--show", type=int, default=20, help="rows to print per bucket")
    args = ap.parse_args()

    extra = Path(args.traces).resolve() / "extra_info"
    fc_path = extra / "tma_desc_factcheck.csv"
    if not fc_path.exists():
        raise SystemExit(
            f"missing {fc_path}\n"
            "Run the tracer with ENABLE_TMA_DESC=1 TMA_DESC_FACTCHECK=1 first.")

    bases, enc_path = load_encode_bases(extra)
    print(f"encode bases ({enc_path.name}): {len(bases)} distinct")
    for b, rows in sorted(bases.items()):
        print(f"    0x{b:x}  x{len(rows)}  box={rows[0][1]}")

    SPACE = {"0": "UNKNOWN/generic", "1": "GLOBAL", "2": "SHARED",
             "3": "CONSTANT", "4": "LOCAL"}

    # Read samples, dedup by (uid,pc,desc_va,qword0)
    seen = set()
    rows = []
    with fc_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["unique_function_id"], r["pc_hex"], r["desc_va_hex"],
                   r["qword0_hex"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    # Address-space histogram (present since the crash-fix; older CSVs lack it).
    have_space = rows and "space" in rows[0]
    if have_space:
        hist = defaultdict(int)
        readable = 0
        for r in rows:
            hist[SPACE.get(r.get("space", "0"), r.get("space"))] += 1
            if r.get("read_ok") == "1":
                readable += 1
        print("\ndesc_va address space (distinct samples):")
        for sp, c in sorted(hist.items()):
            print(f"    {sp:16s} x{c}")
        print(f"    -> read_ok (byte read performed): {readable}/{len(rows)}")

    # Only samples whose bytes were actually read can be base-checked.
    def was_read(r):
        return (not have_space) or r.get("read_ok") == "1"

    matched, unmatched = [], []
    for r in rows:
        if not was_read(r):
            continue
        q0 = parse_int(r["qword0_hex"])
        (matched if q0 in bases else unmatched).append(r)

    print(f"\ndistinct fact-check samples: {len(rows)}")
    print(f"  bytes read (base-checkable) : {len(matched) + len(unmatched)}")
    print(f"  qword0 == a real encode base : {len(matched)}")
    print(f"  qword0 NOT a known base      : {len(unmatched)}")

    def va_space(va):
        if va >= 0xFFFFFFFF00000000:
            return "SMEM/generic-neg"
        if 0x7E00000000 <= va < 0x800000000000:
            return "GMEM"
        if va < 0x100000:
            return "small"
        return "other"

    if matched:
        print("\n=== MATCHED (qword0 is the real base) ===")
        for r in matched[:args.show]:
            va = parse_int(r["desc_va_hex"])
            print(f"  uid={r['unique_function_id']} pc={r['pc_hex']} "
                  f"desc_va={r['desc_va_hex']}[{va_space(va)}] "
                  f"qword0={r['qword0_hex']} -> {[hex(b) for b in bases if b==parse_int(r['qword0_hex'])]}")

    if unmatched:
        print("\n=== UNMATCHED (investigate: coords-modified? wrong operand? swizzle?) ===")
        for r in unmatched[:args.show]:
            va = parse_int(r["desc_va_hex"])
            q0 = parse_int(r["qword0_hex"])
            # is qword0 near a base (base + small offset)?
            near = None
            for b in bases:
                d = q0 - b
                if 0 <= d < (1 << 24):
                    near = (b, d)
                    break
            note = f" near 0x{near[0]:x}+0x{near[1]:x}" if near else ""
            print(f"  uid={r['unique_function_id']} pc={r['pc_hex']} "
                  f"desc_va={r['desc_va_hex']}[{va_space(va)}] "
                  f"qword0={r['qword0_hex']}{note}")

    print()
    checkable = len(matched) + len(unmatched)
    if checkable == 0:
        print("=> INCONCLUSIVE: no descriptor VA was in a readable space (all SHARED/"
              "UNKNOWN-generic), so no bytes were read. See the address-space histogram: "
              "if desc_va is SHARED, the descriptor is SMEM-staged and D1 must read it "
              "as shared at issue (a generic load faults); if it is UNKNOWN/generic, the "
              "operand is a raw cursor (e.g. 0xe800) and D2 (global original) is needed.")
    elif matched and not unmatched:
        print("=> PASS: every read descriptor's qword0 is a real base. Path D1 "
              "(device-read the descriptor at issue) is confirmed.")
    elif matched:
        print("=> PARTIAL: some samples match. If unmatched are 'near base+offset' "
              "they are coords-adjusted copies (still usable: subtract the coord term); "
              "otherwise re-examine the operand choice.")
    else:
        print("=> FAIL: no read sample matched a known base. Either the descriptor VA "
              "is wrong (try the other operand / UTMACCTL target) or the staged bytes "
              "are transformed; consider Path D2 (global original before staging).")


if __name__ == "__main__":
    main()
