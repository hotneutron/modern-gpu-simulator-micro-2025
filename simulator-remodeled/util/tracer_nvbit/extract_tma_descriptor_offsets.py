#!/usr/bin/env python3
"""Phase 0a — TMA descriptor static-offset extractor (def-chain).

Formalizes the throwaway spike from TMA_BASE_ADDR.md §2.11 into a reusable tool.

For every descriptor-carrying TMA op it recovers, per (unique_function_id, pc), the
descriptor's origin in the kernel param block:

    descriptor_VA = param_base + tensormap_offset
    param_base    = runtime value loaded by  ULDC.64 URn, c[0x0][cbank_index]
    tensormap_offset = sum of UIADD3 immediates added on the def-chain of the
                       descriptor operand register (static, unique per pc, §2.11)

Descriptor operand per family (TMA_BASE_ADDR.md §2.6):
  - UTMALDG / UTMAREDG : operand-2 (the 2nd `[URx]` bracket = the descriptor VA ptr)
  - UTMASTG            : operand-1 (the 1st `[URx]` bracket pair)
  - UBLKRED (desc)     : `desc[URx]`

IMPORTANT: this is a STATIC analysis. The executed operand's *runtime* value is SMEM
(§2.8) and must never be used as the VA — only this static offset is the binding key.

Output: extra_info/tma_descriptor_offsets.json — consumed by the Gate 0 harness
(verify_tma_descriptor_deref.py) and later by build_tma_descriptor_mapping.py.

Runs offline; only needs the disassembly under <traces>/extra_info/{sass,nvdisasm}.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the existing, proven SASS parser + kernel/uid matching so this tool stays
# consistent with discover_tma_producers.py (same function->uid mapping).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from discover_tma_producers import (  # noqa: E402
    get_desc_like_ureg_base,
    is_tma_family_opcode,
    load_kernel_tma_info,
    match_function_to_kernel,
    parse_disassembly_file,
    split_operands,
)

DEFAULT_MAX_DEPTH = 12

ULDC_RE = re.compile(r"UR(\d+),\s*c\[0x0\]\[(0x[0-9a-fA-F]+)\]")
# UIADD3[.X] URd, [UPn,] (URs|URZ), (imm|URs|URZ), ...
UIADD3_RE = re.compile(
    r"UR(\d+),\s*(?:UP\d+,\s*)?(-?UR(\d+)|URZ),\s*(0x[0-9a-fA-F]+|-?UR\d+|URZ)"
)
UREG_IN_BRACKET_RE = re.compile(r"UR(\d+)")


def bracket_operands(operands):
    return [op for op in operands if op.strip().startswith("[")]


def first_ureg(text):
    match = UREG_IN_BRACKET_RE.search(text or "")
    return int(match.group(1)) if match else None


def select_descriptor_reg(opcode, operand_text):
    """Return (reg, role) for the operand that carries the descriptor VA, per §2.6."""
    operands = split_operands(operand_text)
    if opcode.startswith("UTMALDG") or opcode.startswith("UTMAREDG"):
        brs = bracket_operands(operands)
        if len(brs) >= 2:
            return first_ureg(brs[1]), "operand2_ptr"
        return None, None
    if opcode.startswith("UTMASTG"):
        brs = bracket_operands(operands)
        if brs:
            return first_ureg(brs[0]), "operand1_ptr"
        return None, None
    if opcode.startswith("UBLKRED"):
        match = re.search(r"desc\[UR(\d+)\]", operand_text)
        if match:
            return int(match.group(1)), "desc_reg"
        return None, None
    return None, None


def is_descriptor_offset_opcode(opcode):
    return (
        opcode.startswith("UTMALDG")
        or opcode.startswith("UTMAREDG")
        or opcode.startswith("UTMASTG")
        or opcode.startswith("UBLKRED")
    )


def build_defs(instructions):
    """Record every ULDC / UIADD3 def as (index, dst_reg, kind, info)."""
    defs = []
    for idx, ins in enumerate(instructions):
        opcode = ins["opcode"]
        operand_text = ins.get("operand_text", "")
        if opcode.startswith("ULDC"):
            match = ULDC_RE.match(operand_text)
            if match:
                defs.append((idx, int(match.group(1)), "uldc", int(match.group(2), 16)))
                continue
        if opcode.startswith("UIADD3"):
            match = UIADD3_RE.match(operand_text)
            if match:
                dst = int(match.group(1))
                src = int(match.group(3)) if match.group(3) else None
                imm_tok = match.group(4)
                imm = int(imm_tok, 16) if imm_tok and imm_tok.startswith("0x") else 0
                defs.append((idx, dst, "uiadd", (src, imm)))
    return defs


def last_def(defs, reg, before_idx):
    best = None
    for (idx, dst, kind, info) in defs:
        if idx < before_idx and dst == reg and (best is None or idx > best[0]):
            best = (idx, kind, info)
    return best


def trace_offset(defs, reg, before_idx, depth=0, max_depth=DEFAULT_MAX_DEPTH):
    """Return (cbank_index, tensormap_offset) or None if the chain does not reach a
    ULDC c[0x0][...] param-base load."""
    if depth > max_depth:
        return None
    found = last_def(defs, reg, before_idx)
    if not found:
        return None
    idx, kind, info = found
    if kind == "uldc":
        return (info, 0)
    if kind == "uiadd":
        src, imm = info
        if src is None:
            return None
        sub = trace_offset(defs, src, idx, depth + 1, max_depth)
        if sub is None:
            return None
        return (sub[0], sub[1] + imm)
    return None


def build_consumers_for_uid_match(function):
    """Minimal consumer shape reused by discover's match_function_to_kernel."""
    consumers = []
    for ins in function["instructions"]:
        if not is_tma_family_opcode(ins["opcode"]):
            continue
        operands = split_operands(ins.get("operand_text", ""))
        desc_ref = get_desc_like_ureg_base(ins["opcode"], operands)
        consumers.append(
            {"pc": ins["pc"], "opcode": ins["opcode"], "desc_ref": desc_ref}
        )
    return consumers


def extract_offsets(extra_info_dir: Path, max_depth: int):
    nvdisasm_dir = extra_info_dir / "nvdisasm"
    sass_dir = extra_info_dir / "sass"
    functions_by_name = {}
    if nvdisasm_dir.exists():
        for path in sorted(nvdisasm_dir.glob("*.nvdisasm")):
            for function in parse_disassembly_file(path):
                functions_by_name.setdefault(function["function_name"], function)
    if sass_dir.exists():
        for path in sorted(sass_dir.glob("*.sass")):
            for function in parse_disassembly_file(path):
                functions_by_name.setdefault(function["function_name"], function)

    kernels = load_kernel_tma_info(extra_info_dir)

    sites = []
    prefetch_sites = []
    cbank_index_counts = defaultdict(int)
    offsets_by_function = defaultdict(set)

    for function in functions_by_name.values():
        instructions = function["instructions"]
        defs = build_defs(instructions)

        matched_kernel, _ = (
            match_function_to_kernel(
                {"consumers": build_consumers_for_uid_match(function)}, kernels
            )
            if kernels
            else (None, 0)
        )
        uid = matched_kernel["unique_function_id"] if matched_kernel else None
        fname = function["function_name"]

        for idx, ins in enumerate(instructions):
            opcode = ins["opcode"]
            operand_text = ins.get("operand_text", "")

            # UTMACCTL.PF [URx] enumerates ALL descriptor offsets in the kernel
            # (incl. ones no UTMALDG executes) — an independent cross-check list.
            if opcode.startswith("UTMACCTL"):
                brs = bracket_operands(split_operands(operand_text))
                if brs:
                    reg = first_ureg(brs[0])
                    if reg is not None:
                        traced = trace_offset(defs, reg, idx, max_depth=max_depth)
                        if traced is not None:
                            cbank_index, tensormap_offset = traced
                            prefetch_sites.append(
                                {
                                    "unique_function_id": uid,
                                    "function_name": fname,
                                    "pc_hex": ins["pc_hex"],
                                    "opcode": opcode,
                                    "descriptor_reg": reg,
                                    "cbank_index_hex": f"0x{cbank_index:x}",
                                    "tensormap_offset_hex": f"0x{tensormap_offset:x}",
                                    "key_hex": f"0x{cbank_index + tensormap_offset:x}",
                                }
                            )
                continue

            if not is_descriptor_offset_opcode(opcode):
                continue

            reg, role = select_descriptor_reg(opcode, operand_text)
            record = {
                "unique_function_id": uid,
                "function_name": fname,
                "pc_hex": ins["pc_hex"],
                "opcode": opcode,
                "descriptor_operand": role,
                "descriptor_reg": reg,
                "resolved": False,
                "cbank_index_hex": None,
                "tensormap_offset_hex": None,
                "key_hex": None,
                "reason": None,
            }
            if reg is None:
                record["reason"] = "no_descriptor_operand_register_found"
                sites.append(record)
                continue
            traced = trace_offset(defs, reg, idx, max_depth=max_depth)
            if traced is None:
                # e.g. UTMALDG desc[URx] would resolve to a UMOV constant, not a
                # ULDC param-base — a self-check that operand selection is correct.
                record["reason"] = "def_chain_did_not_reach_param_base_ULDC"
                sites.append(record)
                continue
            cbank_index, tensormap_offset = traced
            record["resolved"] = True
            record["cbank_index_hex"] = f"0x{cbank_index:x}"
            record["tensormap_offset_hex"] = f"0x{tensormap_offset:x}"
            record["key_hex"] = f"0x{cbank_index + tensormap_offset:x}"
            cbank_index_counts[cbank_index] += 1
            offsets_by_function[str(uid) if uid is not None else fname].add(
                tensormap_offset
            )
            sites.append(record)

    dominant_cbank = None
    if cbank_index_counts:
        dominant_cbank = max(cbank_index_counts.items(), key=lambda kv: kv[1])[0]

    return {
        "extra_info_dir": str(extra_info_dir),
        "cbank_param_base_index_hex": (
            f"0x{dominant_cbank:x}" if dominant_cbank is not None else None
        ),
        "cbank_index_counts": {
            f"0x{idx:x}": count for idx, count in sorted(cbank_index_counts.items())
        },
        "site_count": len(sites),
        "resolved_site_count": sum(1 for s in sites if s["resolved"]),
        "sites": sites,
        "prefetch_sites": prefetch_sites,
        "distinct_tensormap_offsets": {
            key: sorted(f"0x{off:x}" for off in offs)
            for key, offs in sorted(offsets_by_function.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, help="kernel traces dir (parent of extra_info)")
    parser.add_argument("--output")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="exit nonzero if any descriptor-carrying site did not resolve to an offset",
    )
    args = parser.parse_args()

    traces_dir = Path(args.traces).resolve()
    extra_info_dir = traces_dir / "extra_info"
    if not extra_info_dir.exists():
        raise SystemExit(f"missing: {extra_info_dir}")

    result = extract_offsets(extra_info_dir, args.max_depth)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else extra_info_dir / "tma_descriptor_offsets.json"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"wrote {output_path}")
    print(f"cbank_param_base_index={result['cbank_param_base_index_hex']}")
    print(f"sites={result['site_count']} resolved={result['resolved_site_count']}")
    for key, offs in result["distinct_tensormap_offsets"].items():
        print(f"  fn {key}: {len(offs)} distinct offsets {offs}")
    unresolved = [s for s in result["sites"] if not s["resolved"]]
    if unresolved:
        print(f"UNRESOLVED sites={len(unresolved)} (first 8):")
        for site in unresolved[:8]:
            print(
                f"  uid={site['unique_function_id']} pc={site['pc_hex']} "
                f"{site['opcode']} reason={site['reason']}"
            )
        if args.fail_on_unresolved:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
