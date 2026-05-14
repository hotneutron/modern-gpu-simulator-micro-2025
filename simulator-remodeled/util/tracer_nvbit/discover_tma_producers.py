#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


TMA_FAMILY_PREFIXES = (
    "UBLKCP",
    "UBLKPF",
    "UBLKRED",
    "UTMACCTL",
    "UTMACMDFLUSH",
    "UTMALDG",
    "UTMAPF",
    "UTMAREDG",
    "UTMASTG",
)
TMA_DISCOVERY_PREFIXES = ("UTMALDG", "UTMASTG", "UBLKRED", "UTMAREDG")
NV_SECTION_RE = re.compile(r"^//-+\s+\.text\.(.+?)\s+-+$")
SASS_FUNCTION_RE = re.compile(r"^\s*Function\s*:\s*(.+)$")
INST_RE = re.compile(r"^\s*/\*([0-9A-Fa-f]+)\*/\s+(?:@[^ ]+\s+)?([A-Z][A-Z0-9_.]*)(?:\s+(.*?))?\s*;")
DESC_REF_RE = re.compile(r"(?<!g)desc\[UR(\d+)\]")
UREG_RE = re.compile(r"\bUR(\d+)\b")
WIDE_UREG_PREFIXES = ("ULDC.64", "ULDC.128", "UIMAD.WIDE", "ULEA")


def split_operands(operand_text: str):
    parts = []
    current = []
    depth = 0
    for ch in operand_text:
        if ch == "[":
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def get_ureg_write_width(opcode: str):
    if opcode.startswith("ULDC.128"):
        return 4
    if opcode.startswith("ULDC.64"):
        return 2
    if opcode.startswith("UIMAD.WIDE") or opcode.startswith("ULEA"):
        return 2
    return 1


def get_dest_uregs(opcode: str, operand_text: str):
    operands = split_operands(operand_text)
    if not operands:
        return []
    first_operand = operands[0]
    match = UREG_RE.search(first_operand)
    if not match:
        return []
    base = int(match.group(1))
    width = get_ureg_write_width(opcode)
    return list(range(base, base + width))


def get_desc_like_ureg_base(opcode: str, operands):
    for operand in operands:
        desc_match = DESC_REF_RE.search(operand)
        if desc_match:
            return int(desc_match.group(1))
    if opcode.startswith("UTMASTG") and operands:
        first_operand = operands[0]
        if first_operand.startswith("[") and first_operand.endswith("]"):
            match = UREG_RE.search(first_operand)
            if match:
                return int(match.group(1))
    return None


def is_tma_family_opcode(opcode: str):
    return opcode.startswith(TMA_FAMILY_PREFIXES)


def is_tma_discovery_opcode(opcode: str):
    return opcode.startswith(TMA_DISCOVERY_PREFIXES)


def classify_tma_role(opcode: str, has_desc: bool):
    if opcode.startswith("UTMALDG"):
        return "descriptor_load_consumer" if has_desc else "load_consumer"
    if opcode.startswith("UBLKRED") or opcode.startswith("UTMAREDG"):
        return "descriptor_store_reduce_consumer" if has_desc else "store_reduce"
    if opcode.startswith("UTMASTG"):
        return "descriptor_store_consumer" if has_desc else "store"
    if opcode.startswith("UTMAPF") or opcode.startswith("UBLKPF"):
        return "prefetch"
    if opcode.startswith("UTMACCTL") or opcode.startswith("UTMACMDFLUSH"):
        return "control"
    if opcode.startswith("UBLKCP"):
        return "copy"
    return "other"


def collect_consumer_registers(opcode: str, operand_text: str):
    operands = split_operands(operand_text)
    desc_regs = set()
    support_regs = set()
    raw_desc_refs = []
    desc_like_base = get_desc_like_ureg_base(opcode, operands)
    if desc_like_base is not None:
        raw_desc_refs.append(desc_like_base)
        desc_regs.update({desc_like_base, desc_like_base + 1})
    for operand in operands:
        desc_match = DESC_REF_RE.search(operand)
        if desc_match:
            continue
        uregs = [int(value) for value in UREG_RE.findall(operand)]
        if opcode.startswith("UTMASTG") and operands and operand == operands[0]:
            continue
        if "[" in operand and "]" in operand:
            for reg in uregs:
                support_regs.update({reg, reg + 1})
        else:
            support_regs.update(uregs)
    support_regs.difference_update(desc_regs)
    return raw_desc_refs, sorted(desc_regs), sorted(support_regs)


def parse_instruction_line(line: str):
    match = INST_RE.match(line)
    if not match:
        return None
    pc = int(match.group(1), 16)
    opcode = match.group(2)
    operand_text = (match.group(3) or "").strip()
    desc_refs, desc_regs, support_regs = collect_consumer_registers(opcode, operand_text)
    return {
        "pc": pc,
        "pc_hex": f"0x{pc:x}",
        "opcode": opcode,
        "operand_text": operand_text,
        "text": line.strip(),
        "desc_refs": desc_refs,
        "desc_regs": desc_regs,
        "support_regs": support_regs,
        "dest_uregs": get_dest_uregs(opcode, operand_text),
    }


def parse_disassembly_file(path: Path):
    functions = []
    current = None
    for raw_line in path.read_text(errors="ignore").splitlines():
        section_match = NV_SECTION_RE.match(raw_line.strip())
        if section_match:
            if current and current["instructions"]:
                functions.append(current)
            current = {
                "function_name": section_match.group(1),
                "source_file": path.name,
                "source_kind": path.suffix.lstrip("."),
                "instructions": [],
            }
            continue
        function_match = SASS_FUNCTION_RE.match(raw_line)
        if function_match:
            if current and current["instructions"]:
                functions.append(current)
            current = {
                "function_name": function_match.group(1).strip(),
                "source_file": path.name,
                "source_kind": path.suffix.lstrip("."),
                "instructions": [],
            }
            continue
        if current is None:
            continue
        instruction = parse_instruction_line(raw_line)
        if instruction is not None:
            current["instructions"].append(instruction)
    if current and current["instructions"]:
        functions.append(current)
    return functions


def load_kernel_tma_info(extra_info_dir: Path):
    info_path = extra_info_dir / "enhanced_execution_info.json"
    if not info_path.exists():
        return []
    data = json.loads(info_path.read_text())
    kernels = []
    for kernel in data.get("kernels", []):
        consumers = []
        for instruction in kernel.get("instructions", []):
            opcode = instruction.get("op_code", "")
            if not is_tma_family_opcode(opcode):
                continue
            operand_strings = [operand.get("operand_string", "") for operand in instruction.get("operands", [])]
            desc_ref = get_desc_like_ureg_base(opcode, operand_strings)
            consumers.append({
                "pc": int(instruction.get("pc_num_dec", 0)),
                "opcode": opcode,
                "desc_ref": desc_ref,
            })
        if consumers:
            kernels.append({
                "kernel_name": kernel.get("kernel_name"),
                "unique_function_id": kernel.get("unique_function_id"),
                "consumers": consumers,
            })
    return kernels


def match_function_to_kernel(function_info, kernels):
    consumer_key_set = {
        (consumer["pc"], consumer["opcode"], consumer["desc_ref"])
        for consumer in function_info["consumers"]
    }
    best_kernel = None
    best_score = 0
    best_size_gap = None
    for kernel in kernels:
        kernel_key_set = {
            (consumer["pc"], consumer["opcode"], consumer["desc_ref"])
            for consumer in kernel["consumers"]
        }
        score = len(consumer_key_set & kernel_key_set)
        if score == 0:
            continue
        size_gap = abs(len(consumer_key_set) - len(kernel_key_set))
        if best_kernel is None or score > best_score or (score == best_score and size_gap < best_size_gap):
            best_kernel = kernel
            best_score = score
            best_size_gap = size_gap
    return best_kernel, best_score


def classify_producer_role(overlap_desc, overlap_support):
    if overlap_desc:
        return "desc_producer", 3
    if overlap_support:
        return "consumer_input_setup", 2
    return "context", 1


def collect_producer_candidates(instructions, consumer_index, desc_regs, support_regs, lookback):
    tracked_desc_regs = set(desc_regs)
    tracked_support_regs = set(support_regs)
    matches = []
    matched_desc_regs = set()
    matched_support_regs = set()
    start = max(0, consumer_index - lookback)
    for index in range(consumer_index - 1, start - 1, -1):
        instruction = instructions[index]
        dest_regs = set(instruction["dest_uregs"])
        overlap_desc = sorted(tracked_desc_regs & dest_regs)
        overlap_support = sorted(tracked_support_regs & dest_regs)
        if not overlap_desc and not overlap_support:
            continue
        role, score = classify_producer_role(overlap_desc, overlap_support)
        matches.append({
            "pc": instruction["pc"],
            "pc_hex": instruction["pc_hex"],
            "opcode": instruction["opcode"],
            "text": instruction["text"],
            "role": role,
            "score": score,
            "matched_desc_regs": overlap_desc,
            "matched_support_regs": overlap_support,
        })
        matched_desc_regs.update(overlap_desc)
        matched_support_regs.update(overlap_support)
        if matched_desc_regs == tracked_desc_regs and matched_support_regs == tracked_support_regs:
            break
    matches.reverse()
    direct_desc_producers = [match for match in matches if match["role"] == "desc_producer"]
    support_producers = [match for match in matches if match["role"] == "consumer_input_setup"]
    return {
        "tracked_desc_regs": sorted(tracked_desc_regs),
        "tracked_support_regs": sorted(tracked_support_regs),
        "resolved_desc_regs": sorted(matched_desc_regs),
        "resolved_support_regs": sorted(matched_support_regs),
        "direct_desc_producers": direct_desc_producers,
        "support_producers": support_producers,
        "ranked_sequence": matches,
    }


def build_tma_discovery(extra_info_dir: Path, lookback: int):
    kernels = load_kernel_tma_info(extra_info_dir)
    nvdisasm_dir = extra_info_dir / "nvdisasm"
    sass_dir = extra_info_dir / "sass"
    functions_by_name = {}
    if nvdisasm_dir.exists():
        for path in sorted(nvdisasm_dir.glob("*.nvdisasm")):
            for function in parse_disassembly_file(path):
                functions_by_name[function["function_name"]] = function
    if sass_dir.exists():
        for path in sorted(sass_dir.glob("*.sass")):
            for function in parse_disassembly_file(path):
                if function["function_name"] in functions_by_name:
                    continue
                functions_by_name[function["function_name"]] = function
    functions = list(functions_by_name.values())
    discoveries = []
    tma_family_opcode_counts = {}
    tma_family_desc_opcode_counts = {}
    for function in functions:
        family_ops = []
        consumers = []
        for index, instruction in enumerate(function["instructions"]):
            if not is_tma_family_opcode(instruction["opcode"]):
                continue
            opcode = instruction["opcode"]
            tma_family_opcode_counts[opcode] = tma_family_opcode_counts.get(opcode, 0) + 1
            has_desc = bool(instruction["desc_refs"])
            role = classify_tma_role(opcode, has_desc)
            family_entry = {
                "pc": instruction["pc"],
                "pc_hex": instruction["pc_hex"],
                "opcode": opcode,
                "role": role,
                "desc_refs": instruction["desc_refs"],
                "desc_regs": instruction["desc_regs"],
                "support_regs": instruction["support_regs"],
                "text": instruction["text"],
            }
            family_ops.append(family_entry)
            if instruction["desc_refs"]:
                tma_family_desc_opcode_counts[opcode] = tma_family_desc_opcode_counts.get(opcode, 0) + 1
            if not is_tma_discovery_opcode(opcode):
                continue
            if not instruction["desc_refs"]:
                continue
            desc_ref = instruction["desc_refs"][0]
            producer_info = collect_producer_candidates(
                function["instructions"],
                index,
                instruction["desc_regs"],
                instruction["support_regs"],
                lookback,
            )
            consumer_entry = dict(family_entry)
            consumer_entry.update({
                "desc_ref": desc_ref,
                "producer_search": producer_info,
            })
            consumers.append(consumer_entry)
        if not family_ops:
            continue
        function_info = {
            "function_name": function["function_name"],
            "source_file": function["source_file"],
            "source_kind": function["source_kind"],
            "tma_family_ops": family_ops,
            "consumers": consumers,
        }
        matched_kernel, match_score = match_function_to_kernel(function_info, kernels)
        if matched_kernel is not None:
            function_info["matched_kernel_name"] = matched_kernel["kernel_name"]
            function_info["unique_function_id"] = matched_kernel["unique_function_id"]
            function_info["matched_consumer_count"] = match_score
        else:
            function_info["matched_kernel_name"] = None
            function_info["unique_function_id"] = None
            function_info["matched_consumer_count"] = 0
        role_counts = {}
        desc_role_counts = {}
        for op in family_ops:
            role_counts[op["role"]] = role_counts.get(op["role"], 0) + 1
            if op["desc_refs"]:
                desc_role_counts[op["role"]] = desc_role_counts.get(op["role"], 0) + 1
        function_info["tma_role_counts"] = dict(sorted(role_counts.items()))
        function_info["tma_desc_role_counts"] = dict(sorted(desc_role_counts.items()))
        function_info["tma_flow_summary"] = [
            {
                "pc_hex": op["pc_hex"],
                "opcode": op["opcode"],
                "role": op["role"],
                "desc_ref": op["desc_refs"][0] if op["desc_refs"] else None,
            }
            for op in family_ops
        ]
        discoveries.append(function_info)
    discoveries.sort(key=lambda item: (
        item["unique_function_id"] is None,
        item["unique_function_id"] or 0,
        item["source_file"],
        item["function_name"],
    ))
    return {
        "extra_info_dir": str(extra_info_dir),
        "lookback": lookback,
        "function_count": len(discoveries),
        "tma_family_opcode_counts": dict(sorted(tma_family_opcode_counts.items())),
        "tma_family_desc_opcode_counts": dict(sorted(tma_family_desc_opcode_counts.items())),
        "functions": discoveries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output")
    parser.add_argument("--lookback", type=int, default=48)
    args = parser.parse_args()

    traces_dir = Path(args.traces).resolve()
    extra_info_dir = traces_dir / "extra_info"
    if not extra_info_dir.exists():
        raise SystemExit(f"missing: {extra_info_dir}")

    result = build_tma_discovery(extra_info_dir, args.lookback)
    output_path = Path(args.output).resolve() if args.output else extra_info_dir / "tma_discovery.json"
    output_path.write_text(json.dumps(result, indent=2))

    print(f"wrote {output_path}")
    print(f"functions_with_tma={result['function_count']}")
    print("tma_family_opcode_counts")
    for opcode, count in result["tma_family_opcode_counts"].items():
        desc_count = result["tma_family_desc_opcode_counts"].get(opcode, 0)
        print(f"  {opcode}: total={count} with_desc={desc_count}")
    for function in result["functions"][:4]:
        print(
            "function",
            function["unique_function_id"],
            function["source_file"],
            function["function_name"],
            f"family_ops={len(function['tma_family_ops'])}",
            f"consumers={len(function['consumers'])}",
        )
        print("  roles", ",".join(f"{role}={count}" for role, count in function["tma_role_counts"].items()))
        for op in function["tma_flow_summary"][:6]:
            desc_part = f" desc[UR{op['desc_ref']}]" if op["desc_ref"] is not None else ""
            print("  flow", op["pc_hex"], op["opcode"], op["role"] + desc_part)
        for consumer in function["consumers"][:3]:
            matched = ",".join(str(reg) for reg in consumer["producer_search"]["resolved_desc_regs"])
            support = ",".join(str(reg) for reg in consumer["producer_search"]["resolved_support_regs"])
            print(
                "  consumer",
                consumer["pc_hex"],
                consumer["opcode"],
                f"desc[UR{consumer['desc_ref']}]",
                f"desc_regs={matched or 'none'}",
                f"support_regs={support or 'none'}",
            )


if __name__ == "__main__":
    main()
