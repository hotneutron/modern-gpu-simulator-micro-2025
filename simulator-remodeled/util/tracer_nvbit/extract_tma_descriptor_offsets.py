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

# Sentinel returned by trace_offset when the def-chain forks on 2+ live register
# sources (a UIADD3 the single-source walk cannot resolve). Distinct from None
# (clean dead end) so callers never mistake a fork for a resolved/absent offset.
AMBIGUOUS_FORK = object()

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


def enumerate_operand_regs(operand_text):
    """Return [(operand_index, role_hint, reg)] for EVERY operand that carries a UR
    register — brackets, desc[URx], and bare UR. Used to DISCOVER (not assume) which
    operand reaches param_base, since §2.6's per-family guess is wrong for bwd
    UTMASTG/UBLKRED (see tmp.out: operand-1 goes to SMEM)."""
    out = []
    operands = split_operands(operand_text)
    for i, op in enumerate(operands):
        op = op.strip()
        desc_m = re.search(r"desc\[UR(\d+)\]", op)
        if desc_m:
            out.append((i, "desc", int(desc_m.group(1))))
            continue
        if op.startswith("["):
            reg = first_ureg(op)
            if reg is not None:
                out.append((i, "bracket", reg))
            continue
        # bare UR operand (some UTMASTG/UBLKRED forms pass the descriptor as a plain UR)
        m = re.match(r"UR(\d+)$", op)
        if m:
            out.append((i, "bare", int(m.group(1))))
    return out


def is_descriptor_offset_opcode(opcode):
    return (
        opcode.startswith("UTMALDG")
        or opcode.startswith("UTMAREDG")
        or opcode.startswith("UTMASTG")
        or opcode.startswith("UBLKRED")
    )


def build_defs(instructions):
    """Record every ULDC / UIADD3 def as (index, dst_reg, kind, info).

    UIADD3 is `Rd = src0 + src1 + src2` (any of which may be a UR, URZ, or an
    immediate). The linear tracer can only follow ONE register source; if TWO or more
    sources are live registers (a fork), the single-source walk is wrong and would
    fabricate an offset. So we parse ALL three sources and record:
      info = (single_src_reg_or_None, imm_sum, extra_reg_source_count)
    trace_offset treats extra_reg_source_count>0 as an ambiguous fork (returns a
    sentinel) instead of silently dropping the second register."""
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
            operands = split_operands(operand_text)
            if not operands:
                continue
            dst = first_ureg(operands[0])
            if dst is None:
                continue
            # sources = everything after dst, skipping a leading predicate (UPn)
            srcs = operands[1:]
            reg_srcs = []
            imm_sum = 0
            for s in srcs:
                s = s.strip()
                if re.match(r"UP\d+$", s):
                    continue  # carry predicate, not a value source
                if s in ("URZ", "-URZ", "RZ"):
                    continue
                mreg = re.match(r"-?UR(\d+)$", s)
                if mreg:
                    reg_srcs.append(int(mreg.group(1)))
                    continue
                mimm = re.match(r"-?0x[0-9a-fA-F]+$", s)
                if mimm:
                    imm_sum += int(s, 16)
                    continue
                # unknown source form (e.g. a shifted reg) → treat as extra reg fork
                reg_srcs.append(-1)
            single_src = reg_srcs[0] if reg_srcs else None
            extra_reg_count = max(0, len(reg_srcs) - 1)
            defs.append((idx, dst, "uiadd", (single_src, imm_sum, extra_reg_count)))
    return defs


def last_def(defs, reg, before_idx):
    best = None
    for (idx, dst, kind, info) in defs:
        if idx < before_idx and dst == reg and (best is None or idx > best[0]):
            best = (idx, kind, info)
    return best


def trace_offset(defs, reg, before_idx, depth=0, max_depth=DEFAULT_MAX_DEPTH, chain=None):
    """Return (cbank_index, tensormap_offset), None (dead end), or the sentinel
    AMBIGUOUS_FORK if the chain hits a UIADD3 with 2+ live register sources (which the
    single-source walk cannot resolve). If `chain` is given, append a step per hop."""
    if depth > max_depth:
        if chain is not None:
            chain.append(f"UR{reg}: max_depth")
        return None
    found = last_def(defs, reg, before_idx)
    if not found:
        if chain is not None:
            chain.append(f"UR{reg}: no_def")
        return None
    idx, kind, info = found
    if kind == "uldc":
        if chain is not None:
            chain.append(f"UR{reg} <- ULDC c[0x0][0x{info:x}]")
        return (info, 0)
    if kind == "uiadd":
        src, imm, extra_reg_count = info
        if chain is not None:
            src_txt = f"UR{src}" if (src is not None and src >= 0) else (
                "UR?" if src == -1 else "URZ/none")
            fork = f" +{extra_reg_count}reg_fork" if extra_reg_count else ""
            chain.append(f"UR{reg} <- UIADD3 {src_txt} + 0x{imm:x}{fork}")
        if extra_reg_count > 0 or src == -1:
            return AMBIGUOUS_FORK
        if src is None:
            return None
        sub = trace_offset(defs, src, idx, depth + 1, max_depth, chain)
        if sub is None or sub is AMBIGUOUS_FORK:
            return sub
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


def uid_key(uid, fname):
    return str(uid) if uid is not None else fname


def load_executed_sites(extra_info_dir: Path):
    """Executed (uid, pc_hex) set from tma_runtime_operand_debug.jsonl, restricted to
    descriptor-carrying families. Returns None if absent. The REAL decision metric is
    'what % of EXECUTED loads reach param_base', not the static site count."""
    path = extra_info_dir / "tma_runtime_operand_debug.jsonl"
    if not path.exists():
        return None
    executed = set()
    try:
        import json as _json
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except ValueError:
                    continue
                op = rec.get("opcode", "")
                if not is_descriptor_offset_opcode(op):
                    continue
                uid = rec.get("unique_function_id")
                executed.add((uid, rec.get("pc_hex")))
    except OSError:
        return None
    return executed


def extract_offsets(extra_info_dir: Path, max_depth: int,
                    param_base_cbank_index_override=None):
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

    # Pre-pass: determine the param-base cbank index. Trace every descriptor-carrying
    # site's operand to its terminal ULDC c[0x0][K]; the dominant K over UTMALDG/
    # UTMAREDG (the load family, which §2.11 verified goes through param base) is the
    # param-base index. An explicit override wins.
    prepass_cbank_counts = defaultdict(int)
    for function in functions_by_name.values():
        instructions = function["instructions"]
        defs = build_defs(instructions)
        for idx, ins in enumerate(instructions):
            opcode = ins["opcode"]
            if not (opcode.startswith("UTMALDG") or opcode.startswith("UTMAREDG")):
                continue
            reg, _ = select_descriptor_reg(opcode, ins.get("operand_text", ""))
            if reg is None:
                continue
            traced = trace_offset(defs, reg, idx, max_depth=max_depth)
            if traced is not None and traced is not AMBIGUOUS_FORK:
                prepass_cbank_counts[traced[0]] += 1
    if param_base_cbank_index_override is not None:
        param_base_cbank_index = param_base_cbank_index_override
    elif prepass_cbank_counts:
        param_base_cbank_index = max(
            prepass_cbank_counts.items(), key=lambda kv: kv[1]
        )[0]
    else:
        param_base_cbank_index = 0x198

    sites = []
    prefetch_sites = []
    cbank_index_counts = defaultdict(int)
    offsets_by_function = defaultdict(set)
    executed_sites = load_executed_sites(extra_info_dir)

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
                        if traced is not None and traced is not AMBIGUOUS_FORK:
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

            # DISCOVER which operand actually reaches param_base (don't trust the
            # §2.6 guess; bwd UTMASTG/UBLKRED put SMEM in operand-1). Trace every
            # UR-bearing operand and record where each terminates.
            operand_probe = []
            param_base_operands = []
            for op_idx, hint, op_reg in enumerate_operand_regs(operand_text):
                t = trace_offset(defs, op_reg, idx, max_depth=max_depth)
                is_tuple = t is not None and t is not AMBIGUOUS_FORK
                entry = {
                    "operand_index": op_idx,
                    "role_hint": hint,
                    "reg": op_reg,
                    "cbank_index_hex": f"0x{t[0]:x}" if is_tuple else None,
                    "tensormap_offset_hex": f"0x{t[1]:x}" if is_tuple else None,
                    "ambiguous_fork": (t is AMBIGUOUS_FORK),
                }
                operand_probe.append(entry)
                if is_tuple and t[0] == param_base_cbank_index:
                    param_base_operands.append((op_idx, hint, op_reg, t[1]))

            record = {
                "unique_function_id": uid,
                "function_name": fname,
                "pc_hex": ins["pc_hex"],
                "opcode": opcode,
                "descriptor_operand": role,
                "descriptor_reg": reg,
                "operand_probe": operand_probe,
                "param_base_operand_count": len(param_base_operands),
                "executed": (
                    None if executed_sites is None
                    else ((uid, ins["pc_hex"]) in executed_sites)
                ),
                "resolved": False,
                "cbank_index_hex": None,
                "tensormap_offset_hex": None,
                "key_hex": None,
                "chain": None,
                "reason": None,
            }

            # Prefer the data-discovered param_base operand over the §2.6 guess when
            # they disagree and exactly one operand reaches param_base.
            if len(param_base_operands) == 1:
                d_idx, d_hint, d_reg, d_off = param_base_operands[0]
                if d_reg != reg:
                    record["descriptor_operand"] = f"discovered_idx{d_idx}_{d_hint}"
                    record["descriptor_reg"] = d_reg
                    reg, role = d_reg, record["descriptor_operand"]

            if reg is None:
                record["reason"] = "no_descriptor_operand_register_found"
                sites.append(record)
                continue
            chain = []
            traced = trace_offset(defs, reg, idx, max_depth=max_depth, chain=chain)
            record["chain"] = chain
            if traced is AMBIGUOUS_FORK:
                # UIADD3 forked on 2+ live register sources; the single-source walk
                # cannot resolve a real offset. Do NOT fabricate one.
                record["reason"] = "ambiguous_multisource_uiadd3"
                sites.append(record)
                continue
            if traced is None:
                # Chain did not terminate at ANY ULDC c[0x0][...] (e.g. a UMOV
                # constant, a global-memory load, or a register we do not model).
                record["reason"] = "def_chain_did_not_reach_any_cbank_ULDC"
                sites.append(record)
                continue
            cbank_index, tensormap_offset = traced
            record["cbank_index_hex"] = f"0x{cbank_index:x}"
            record["tensormap_offset_hex"] = f"0x{tensormap_offset:x}"
            record["key_hex"] = f"0x{cbank_index + tensormap_offset:x}"
            cbank_index_counts[cbank_index] += 1
            # A site is "resolved" for our purposes ONLY if it reached the param-base
            # cbank index. Chains that terminate at a DIFFERENT cbank slot (e.g.
            # c[0x0][0x0]) are real but are not tensormap-in-param descriptors, so
            # they must not pollute the descriptor offset set.
            if cbank_index == param_base_cbank_index:
                record["resolved"] = True
                offsets_by_function[uid_key(uid, fname)].add(tensormap_offset)
            else:
                record["reason"] = "reached_non_param_base_cbank"
            sites.append(record)

    dominant_cbank = None
    if cbank_index_counts:
        dominant_cbank = max(cbank_index_counts.items(), key=lambda kv: kv[1])[0]

    # Executed-site resolution: the real decision metric. Of the descriptor sites
    # that actually ran, how many reach param_base vs escape to GMEM/other cbank?
    executed_stats = None
    if executed_sites is not None:
        ex = [s for s in sites if s.get("executed")]
        by_family = defaultdict(lambda: {"executed": 0, "resolved": 0,
                                         "non_param_cbank": 0, "no_cbank": 0,
                                         "ambiguous": 0})
        # Per-uid resolution: this trace has BOTH fwd and bwd kernels, so split by uid
        # to see which executed kernel is 100% clean (fwd) vs mixed (bwd).
        by_uid = defaultdict(lambda: {"executed": 0, "resolved": 0})
        for s in ex:
            fam = s["opcode"].split(".")[0]
            b = by_family[fam]
            b["executed"] += 1
            u = by_uid[str(s["unique_function_id"])]
            u["executed"] += 1
            if s["resolved"]:
                b["resolved"] += 1
                u["resolved"] += 1
            elif s["reason"] == "reached_non_param_base_cbank":
                b["non_param_cbank"] += 1
            elif s["reason"] == "def_chain_did_not_reach_any_cbank_ULDC":
                b["no_cbank"] += 1
            elif s["reason"] == "ambiguous_multisource_uiadd3":
                b["ambiguous"] += 1
        for u in by_uid.values():
            u["pct"] = round(100.0 * u["resolved"] / max(1, u["executed"]), 1)
        executed_stats = {
            "executed_site_count": len(ex),
            "executed_resolved_count": sum(1 for s in ex if s["resolved"]),
            "by_opcode_family": {k: dict(v) for k, v in sorted(by_family.items())},
            "by_uid": {k: dict(v) for k, v in sorted(by_uid.items())},
        }

    return {
        "extra_info_dir": str(extra_info_dir),
        "param_base_cbank_index_hex": f"0x{param_base_cbank_index:x}",
        "dominant_cbank_index_hex": (
            f"0x{dominant_cbank:x}" if dominant_cbank is not None else None
        ),
        "cbank_index_counts": {
            f"0x{idx:x}": count for idx, count in sorted(cbank_index_counts.items())
        },
        "site_count": len(sites),
        "resolved_site_count": sum(1 for s in sites if s["resolved"]),
        "executed_stats": executed_stats,
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
        "--param-base-cbank",
        help="override param-base c[0x0][K] index (hex ok, e.g. 0x198); "
        "default = auto-detect dominant over UTMALDG/UTMAREDG",
    )
    parser.add_argument(
        "--only-uid",
        help="restrict console breakdown to this unique_function_id (e.g. 8 for bwd)",
    )
    parser.add_argument(
        "--dump-chains",
        type=int,
        default=0,
        help="print the full def-chain for the first N unresolved sites (diagnosis)",
    )
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

    override = None
    if args.param_base_cbank:
        override = int(args.param_base_cbank, 0)

    result = extract_offsets(extra_info_dir, args.max_depth, override)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else extra_info_dir / "tma_descriptor_offsets.json"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"wrote {output_path}")
    print(f"param_base_cbank_index={result['param_base_cbank_index_hex']} "
          f"(dominant={result['dominant_cbank_index_hex']})")
    print(f"cbank_index_counts={result['cbank_index_counts']}")
    print(f"sites={result['site_count']} resolved={result['resolved_site_count']}")

    ex = result.get("executed_stats")
    if ex is not None:
        print(f"EXECUTED sites={ex['executed_site_count']} "
              f"resolved={ex['executed_resolved_count']} "
              f"({100.0*ex['executed_resolved_count']/max(1,ex['executed_site_count']):.1f}% reach param_base)")
        for fam, b in ex["by_opcode_family"].items():
            print(f"    {fam}: {b}")
        # per-uid: this trace has BOTH fwd and bwd; 100% = clean fwd, <100% = bwd-like
        print("  by uid (100% = clean by-value fwd; <100% = by-pointer/bwd):")
        for u, b in ex.get("by_uid", {}).items():
            print(f"    uid {u}: executed={b['executed']} resolved={b['resolved']} "
                  f"({b['pct']}%)")
    else:
        print("EXECUTED stats: (no tma_runtime_operand_debug.jsonl — static-only view)")

    # reason breakdown (why sites did not resolve to a param-base offset)
    reason_counts = defaultdict(int)
    for site in result["sites"]:
        if not site["resolved"]:
            reason_counts[site["reason"]] += 1
    if reason_counts:
        print("unresolved reasons:", dict(reason_counts))

    # distinct param-base offsets per function (only the ones that matter)
    for key, offs in result["distinct_tensormap_offsets"].items():
        if args.only_uid and key != args.only_uid:
            continue
        print(f"  fn {key}: {len(offs)} param-base offsets {offs}")

    # per-opcode-family view for the target uid, so bwd structure is visible
    if args.only_uid:
        fam = defaultdict(lambda: defaultdict(int))
        for site in result["sites"]:
            if str(site["unique_function_id"]) != args.only_uid:
                continue
            status = "resolved" if site["resolved"] else site["reason"]
            fam[site["opcode"].split(".")[0]][status] += 1
        print(f"  uid {args.only_uid} by opcode family:")
        for op, statuses in sorted(fam.items()):
            print(f"    {op}: {dict(statuses)}")

    if args.dump_chains > 0:
        shown = 0
        print(f"def-chains for first {args.dump_chains} unresolved sites:")
        for site in result["sites"]:
            if site["resolved"]:
                continue
            if args.only_uid and str(site["unique_function_id"]) != args.only_uid:
                continue
            print(f"  uid={site['unique_function_id']} pc={site['pc_hex']} "
                  f"{site['opcode']} reason={site['reason']} "
                  f"reg=UR{site['descriptor_reg']} cbank={site['cbank_index_hex']} "
                  f"off={site['tensormap_offset_hex']}")
            for step in (site.get("chain") or []):
                print(f"      {step}")
            shown += 1
            if shown >= args.dump_chains:
                break

    if args.fail_on_unresolved:
        unresolved = [s for s in result["sites"] if not s["resolved"]]
        if unresolved:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
