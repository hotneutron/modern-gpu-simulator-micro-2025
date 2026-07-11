#!/usr/bin/env python3
"""Diagnostic — classify the RUNTIME operand values of executed TMA descriptor sites.

Decides the bwd strategy question (TMA_BASE_ADDR.md, post-§2.13): does the descriptor
operand hold a GMEM address at runtime (=> device-side operand capture recovers the
descriptor VA directly, bypassing static param_base+offset reconstruction), or is it
SMEM / something else (=> must reconstruct statically, which caps bwd at ~53%)?

Reads:
  - tma_runtime_operand_debug.jsonl  (per executed TMA op, per operand, runtime values)
  - tma_descriptor_offsets.json      (Phase 0a; to join the static resolution reason)

For each executed descriptor site (uid,pc) it reports, per operand index, the distinct
runtime addresses seen and buckets each:
  SMEM   : < 0x100000            (shared-memory window / small staging cursor)
  PARAM  : 0x100000 .. 0x7e00000000  (unlikely; kept for visibility)
  GMEM   : >= 0x7e00000000       (typical CUDA device global VA, 0x7f....)
  ZERO   : 0
so you can see at a glance whether an operand is the GMEM descriptor VA.

Usage:
  python3 analyze_tma_runtime_operands.py --traces <dir> [--only-uid 8] [--families UTMALDG,UBLKRED]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

DESC_FAMILIES = ("UTMALDG", "UTMAREDG", "UTMASTG", "UBLKRED")

# Address buckets. CUDA device global VAs on H100 are ~0x7f........; shared memory
# offsets are small (a few KB..MB). Thresholds are deliberately loose/observational.
GMEM_MIN = 0x7E00000000
SMEM_MAX = 0x100000


def bucket(addr):
    if addr == 0:
        return "ZERO"
    if addr < SMEM_MAX:
        return "SMEM"
    if addr >= GMEM_MIN:
        return "GMEM"
    return "MID"


def load_reasons(extra_info_dir):
    path = extra_info_dir / "tma_descriptor_offsets.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for s in data.get("sites", []):
        out[(s.get("unique_function_id"), s.get("pc_hex"))] = {
            "reason": ("resolved" if s.get("resolved") else s.get("reason")),
            "descriptor_operand": s.get("descriptor_operand"),
            "tensormap_offset_hex": s.get("tensormap_offset_hex"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--only-uid")
    ap.add_argument("--families", default=",".join(DESC_FAMILIES))
    args = ap.parse_args()

    extra = Path(args.traces).resolve() / "extra_info"
    jsonl = extra / "tma_runtime_operand_debug.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}")
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    reasons = load_reasons(extra)

    # (uid, pc, opcode) -> operand_index -> {bucket: count, samples:set}
    sites = defaultdict(lambda: defaultdict(lambda: {"buckets": defaultdict(int),
                                                     "samples": set()}))
    site_opcode = {}
    site_desc = defaultdict(set)  # (uid,pc) -> {(desc_lo, desc_hi)}
    with jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            op = r.get("opcode", "")
            if not op.startswith(families):
                continue
            uid = r.get("unique_function_id")
            if args.only_uid and str(uid) != args.only_uid:
                continue
            pc = r.get("pc_hex")
            key = (uid, pc)
            site_opcode[key] = op
            ci = r.get("callback_index")
            # the runtime address for a MEMORY_REF operand is first_lane_addr;
            # value_lo/value_hi are the desc/other reg values.
            addr = r.get("first_lane_addr", 0) or 0
            slot = sites[key][ci]
            slot["buckets"][bucket(addr)] += 1
            slot["mem_type"] = r.get("mem_type")
            if len(slot["samples"]) < 4:
                slot["samples"].add(addr)
            # desc[URx] handle (constant {lo,hi}); §2.8 showed lo=0, hi=handle_hi.
            dhi = r.get("desc_value_hi")
            dlo = r.get("desc_value_lo")
            if dhi is not None:
                site_desc[key].add((dlo, dhi))

    if not sites:
        print("no executed descriptor sites matched (check --only-uid / --families)")
        return

    fam_summary = defaultdict(lambda: defaultdict(int))
    print(f"executed descriptor sites: {len(sites)}")
    for key in sorted(sites, key=lambda k: (str(k[0]), k[1])):
        uid, pc = key
        op = site_opcode[key]
        meta = reasons.get(key, {})
        fam = op.split(".")[0]
        # does ANY operand look like GMEM? that's the device-capture viability signal
        any_gmem = any(
            "GMEM" in slot["buckets"] for slot in sites[key].values()
        )
        fam_summary[fam]["sites"] += 1
        fam_summary[fam]["gmem_operand" if any_gmem else "no_gmem_operand"] += 1
        print(f"\nuid={uid} pc={pc} {op}")
        print(f"  static: reason={meta.get('reason')} "
              f"operand={meta.get('descriptor_operand')} "
              f"off={meta.get('tensormap_offset_hex')}")
        descs = sorted(site_desc.get(key, set()))
        if descs:
            desc_txt = ", ".join(f"{{lo=0x{lo:x},hi=0x{hi:x}}}" for lo, hi in descs)
            print(f"  desc[URx] handle(s): {desc_txt}")
        for ci in sorted(sites[key]):
            slot = sites[key][ci]
            buckets = dict(slot["buckets"])
            samples = ", ".join(f"0x{a:x}" for a in sorted(slot["samples"]))
            print(f"  operand ci={ci} mem_type={slot.get('mem_type')}: "
                  f"{buckets}  samples=[{samples}]")

    print("\n=== by family: does an executed site expose a GMEM operand? ===")
    for fam, d in sorted(fam_summary.items()):
        print(f"  {fam}: {dict(d)}")
    print("\nreading: if UTMALDG sites show a GMEM operand, device-side capture of that "
          "operand recovers the descriptor VA directly (bypasses static param_base+offset).")


if __name__ == "__main__":
    main()
