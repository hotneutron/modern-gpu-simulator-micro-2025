#!/usr/bin/env python3
"""Phase 0c — Gate 0 host-deref verification harness (TMA_BASE_ADDR.md §2.12).

Proves the single load-bearing assumption of Path B: that reading 128B at
`descriptor_VA = param_base + tensormap_offset` returns the *same* descriptor the
host encoded, for EVERY executed descriptor offset — not just 1/7 (§2.10c).

Inputs (all under <traces>/extra_info, produced by earlier phases):
  - tma_descriptor_offsets.json      (Phase 0a: (uid,pc) -> tensormap_offset)
  - tma_descriptor_deref.csv         (Phase 0b: tracer cuMemcpyDtoH of 128B @ VA)
  - tensor_map_encode_dump.csv       (host cuTensorMapEncodeTiled truth)
  - tensor_map_encode_blobs/*.bin    (host 128B blobs; optional, byte-exact check)

Deref dump CSV contract (Phase 0b writes this; columns, header required):
  device_id,unique_function_id,pc_hex,tensormap_offset_hex,descriptor_va_hex,
  param_base_hex,qword0_hex,...,qword7_hex[,blob_path]

Verification (all must hold; --strict makes any failure exit nonzero):
  1. arithmetic: descriptor_va == param_base + tensormap_offset
  2. match: the dumped 128B equals exactly ONE encode row
     (byte-exact if blobs present, else all 8 qwords equal)
  3. base: dumped qword0 == that row's global_address_hex   (the real base)
  4. consistency: one tensormap_offset never maps to two different encode rows
  5. coverage: every *executed* descriptor offset (0a, restricted to runtime-seen
     pcs if a runtime file is given) has a deref record

Output: tma_descriptor_offset_binding.json — the offset -> encode-row binding table
(the deliverable that later replaces the handle_hi resolver). Carries global_base.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_int(raw):
    if raw is None:
        return 0
    raw = str(raw).strip()
    if raw == "":
        return 0
    return int(raw, 0)


def load_offsets(extra_info_dir: Path):
    path = extra_info_dir / "tma_descriptor_offsets.json"
    data = json.loads(path.read_text())
    # (uid, pc_hex) -> tensormap_offset(int); keep only resolved descriptor sites.
    by_site = {}
    offsets_by_uid = defaultdict(set)
    for site in data.get("sites", []):
        if not site.get("resolved"):
            continue
        uid = site.get("unique_function_id")
        pc = site.get("pc_hex")
        off = parse_int(site.get("tensormap_offset_hex"))
        by_site[(uid, pc)] = off
        offsets_by_uid[uid].add(off)
    return data, by_site, offsets_by_uid


def load_executed_sites(extra_info_dir: Path):
    """Executed (uid, pc_hex) set from the runtime operand debug (descriptor-carrying
    families only). Used to restrict the coverage check to offsets that actually ran —
    static offsets like 0x5c8/0x688 (§2.11) are never executed and must not fail
    coverage. Returns None if the runtime file is absent (coverage then falls back to
    all static offsets)."""
    path = extra_info_dir / "tma_runtime_operand_debug.jsonl"
    if not path.exists():
        return None
    executed = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            opcode = rec.get("opcode", "")
            if not (
                opcode.startswith("UTMALDG")
                or opcode.startswith("UTMAREDG")
                or opcode.startswith("UTMASTG")
                or opcode.startswith("UBLKRED")
            ):
                continue
            uid = rec.get("unique_function_id")
            executed.add((uid, rec.get("pc_hex")))
    return executed


def load_encode_rows(extra_info_dir: Path):
    path = extra_info_dir / "tensor_map_encode_dump.csv"
    rows = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            qwords = [parse_int(row.get(f"qword{i}_hex")) for i in range(8)]
            blob_path = row.get("blob_path") or ""
            blob_bytes = None
            if blob_path and Path(blob_path).exists():
                blob_bytes = Path(blob_path).read_bytes()
            rows.append(
                {
                    "dump_id": parse_int(row.get("dump_id")),
                    "global_address": parse_int(row.get("global_address_hex")),
                    "qwords": qwords,
                    "blob_bytes": blob_bytes,
                    "tensor_rank": parse_int(row.get("tensor_rank")),
                    "box_dim": row.get("box_dim", ""),
                    "global_strides": row.get("global_strides", ""),
                    "swizzle": parse_int(row.get("swizzle")),
                }
            )
    return rows


def load_deref(extra_info_dir: Path, by_site):
    """Build deref records by slicing the param-region blob (Phase 0b tracer output,
    tma_param_base_deref.csv) at each static tensormap_offset (Phase 0a, by_site).

    The tracer dumps ONE raw param-region blob per launch (param_base + region bytes),
    not per-offset 128B — so slicing/matching is done here offline. Each (uid, offset)
    yields one deref record with the 128B at region[offset:offset+128]."""
    csv_path = extra_info_dir / "tma_param_base_deref.csv"
    if not csv_path.exists():
        return None
    # uid -> (param_base, region_bytes) from the most recent launch of that uid.
    regions = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if parse_int(row.get("memcpy_ok")) != 1:
                continue
            uid = (
                None
                if row.get("unique_function_id") in (None, "", "None")
                else parse_int(row.get("unique_function_id"))
            )
            blob_path = row.get("region_blob_path") or ""
            if not blob_path or not Path(blob_path).exists():
                continue
            regions[uid] = {
                "param_base": parse_int(row.get("param_base_hex")),
                "region": Path(blob_path).read_bytes(),
            }

    # offsets to slice, grouped by uid, from the 0a static site table
    offsets_by_uid = defaultdict(set)
    pc_by_uid_offset = defaultdict(list)
    for (uid, pc), off in by_site.items():
        offsets_by_uid[uid].add(off)
        pc_by_uid_offset[(uid, off)].append(pc)

    records = []
    for uid, offsets in offsets_by_uid.items():
        region_info = regions.get(uid)
        if region_info is None:
            # Fall back: some kernels share a param_base captured under a different
            # uid attribution; try the single-region case.
            if len(regions) == 1:
                region_info = next(iter(regions.values()))
            else:
                continue
        region = region_info["region"]
        param_base = region_info["param_base"]
        for off in sorted(offsets):
            if off + 128 > len(region):
                records.append(
                    {
                        "uid": uid,
                        "pc_hex": (pc_by_uid_offset[(uid, off)] or [None])[0],
                        "tensormap_offset": off,
                        "descriptor_va": param_base + off,
                        "param_base": param_base,
                        "qwords": [0] * 8,
                        "blob_bytes": None,
                        "slice_out_of_range": True,
                    }
                )
                continue
            chunk = region[off : off + 128]
            qwords = [
                int.from_bytes(chunk[i * 8 : i * 8 + 8], "little") for i in range(8)
            ]
            records.append(
                {
                    "uid": uid,
                    "pc_hex": (pc_by_uid_offset[(uid, off)] or [None])[0],
                    "tensormap_offset": off,
                    "descriptor_va": param_base + off,
                    "param_base": param_base,
                    "qwords": qwords,
                    "blob_bytes": chunk,
                    "slice_out_of_range": False,
                }
            )
    return records


def qwords_to_bytes(qwords):
    out = b""
    for q in qwords:
        out += int(q).to_bytes(8, "little")
    return out


def match_encode_row(deref, encode_rows):
    """Return (matches, method). Prefer byte-exact; fall back to all-8-qword equal."""
    if deref["blob_bytes"] is not None and any(
        r["blob_bytes"] is not None for r in encode_rows
    ):
        target = deref["blob_bytes"][:128]
        matches = [
            r
            for r in encode_rows
            if r["blob_bytes"] is not None and r["blob_bytes"][:128] == target
        ]
        return matches, "byte_exact"
    dq = deref["qwords"]
    matches = [r for r in encode_rows if r["qwords"] == dq]
    return matches, "qword_equal"


def verify(extra_info_dir: Path, strict: bool):
    _, by_site, offsets_by_uid = load_offsets(extra_info_dir)
    encode_rows = load_encode_rows(extra_info_dir)
    deref = load_deref(extra_info_dir, by_site)
    executed_sites = load_executed_sites(extra_info_dir)

    # Expected coverage set: offsets that ACTUALLY executed. If the runtime file is
    # present, intersect the 0a static sites with executed (uid,pc); else fall back to
    # all static offsets (over-strict, but flagged in output).
    if executed_sites is not None:
        expected_offsets_by_uid = defaultdict(set)
        for (uid, pc), off in by_site.items():
            if (uid, pc) in executed_sites:
                expected_offsets_by_uid[uid].add(off)
        coverage_basis = "executed_runtime_sites"
    else:
        expected_offsets_by_uid = offsets_by_uid
        coverage_basis = "all_static_sites(no_runtime_file)"

    failures = []
    binding = {}  # (uid, offset) -> encode-row summary (the deliverable)
    offset_to_row = {}  # (uid, offset) -> dump_id, for consistency check

    if deref is None:
        failures.append(
            "MISSING tma_param_base_deref.csv — run Phase 0b (tracer, ENABLE_TMA_DESC=1) "
            "first. This harness slices the param-region blob at the 0a offsets; it does "
            "not do the cuMemcpyDtoH itself."
        )
        return {"pass": False, "failures": failures, "binding": {}}

    # Set of executed (uid, offset) — only these MUST match an encode row. Non-executed
    # static offsets (e.g. 0x5c8/0x688, §2.11) are sliced for info but never hard-fail.
    executed_offsets = set()
    for uid, offs in expected_offsets_by_uid.items():
        for off in offs:
            executed_offsets.add((uid, off))

    seen_offsets = defaultdict(set)  # uid -> {offset}
    for rec in deref:
        uid = rec["uid"]
        off = rec["tensormap_offset"]
        is_executed = (uid, off) in executed_offsets
        if is_executed:
            seen_offsets[uid].add(off)
        tag = f"uid={uid} pc={rec['pc_hex']} tm_off=0x{off:x} va=0x{rec['descriptor_va']:x}"

        # (1) arithmetic (applies to every sliced offset; cheap invariant)
        if rec["param_base"] and rec["descriptor_va"] != rec["param_base"] + off:
            failures.append(
                f"[arith] {tag}: descriptor_va != param_base(0x{rec['param_base']:x}) "
                f"+ tensormap_offset(0x{off:x})"
            )

        # (2) match to exactly one encode row
        matches, method = match_encode_row(rec, encode_rows)
        if len(matches) != 1:
            if is_executed:
                failures.append(
                    f"[match:{method}] {tag}: expected exactly 1 encode row, got "
                    f"{len(matches)} (dump_ids={[m['dump_id'] for m in matches]})"
                )
            # non-executed offset with no unique match: informational, not a failure
            continue
        row = matches[0]

        # (3) base == qword0
        if rec["qwords"][0] != row["global_address"]:
            if is_executed:
                failures.append(
                    f"[base] {tag}: dumped qword0=0x{rec['qwords'][0]:x} != "
                    f"global_address_hex=0x{row['global_address']:x} (dump_id={row['dump_id']})"
                )
            continue

        # (4) offset consistency
        key = (uid, off)
        if key in offset_to_row and offset_to_row[key] != row["dump_id"]:
            if is_executed:
                failures.append(
                    f"[consistency] {tag}: tensormap_offset maps to dump_id "
                    f"{offset_to_row[key]} AND {row['dump_id']}"
                )
            continue
        offset_to_row[key] = row["dump_id"]
        binding[f"{uid}:0x{off:x}"] = {
            "unique_function_id": uid,
            "tensormap_offset_hex": f"0x{off:x}",
            "executed": is_executed,
            "encode_dump_id": row["dump_id"],
            "global_base_hex": f"0x{row['global_address']:x}",
            "tensor_rank": row["tensor_rank"],
            "box_dim": row["box_dim"],
            "global_strides": row["global_strides"],
            "swizzle": row["swizzle"],
            "match_method": method,
        }

    # (5) coverage: every executed offset must have a deref record
    for uid, offs in expected_offsets_by_uid.items():
        missing = sorted(offs - seen_offsets.get(uid, set()))
        for off in missing:
            failures.append(
                f"[coverage] uid={uid} tensormap_offset=0x{off:x} executed but "
                f"no deref record (basis={coverage_basis})"
            )

    result = {
        "pass": len(failures) == 0,
        "coverage_basis": coverage_basis,
        "encode_rows": len(encode_rows),
        "deref_records": len(deref),
        "bound_offsets": len(binding),
        "failures": failures,
        "binding": binding,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, help="kernel traces dir (parent of extra_info)")
    parser.add_argument("--binding-out")
    parser.add_argument("--strict", action="store_true", help="exit nonzero on any failure")
    args = parser.parse_args()

    extra_info_dir = Path(args.traces).resolve() / "extra_info"
    if not extra_info_dir.exists():
        raise SystemExit(f"missing: {extra_info_dir}")

    result = verify(extra_info_dir, args.strict)

    binding_out = (
        Path(args.binding_out).resolve()
        if args.binding_out
        else extra_info_dir / "tma_descriptor_offset_binding.json"
    )
    binding_out.write_text(
        json.dumps(
            {"pass": result["pass"], "binding": result["binding"]}, indent=2
        )
        + "\n"
    )

    print(f"Gate 0: {'PASS' if result['pass'] else 'FAIL'}")
    print(
        f"  encode_rows={result.get('encode_rows')} "
        f"deref_records={result.get('deref_records')} "
        f"bound_offsets={result.get('bound_offsets')}"
    )
    print(f"  wrote {binding_out}")
    for line in result["failures"]:
        print(f"  FAIL {line}")
    if not result["pass"] and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
