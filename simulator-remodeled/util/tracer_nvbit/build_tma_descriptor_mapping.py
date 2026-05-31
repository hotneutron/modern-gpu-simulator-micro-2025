#!/usr/bin/env python3

import argparse
import collections
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path


def parse_int_list(raw: str):
    raw = (raw or "").strip().strip('"')
    if not raw:
        return []
    return [int(value) for value in raw.split()]


def parse_int(raw: str):
    return int((raw or "0").strip(), 0)


def normalize_hex(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return ""
    return f"0x{int(raw, 0):x}"


def build_config_id(config):
    box = "x".join(str(value) for value in config["box_dim"])
    rank = config["tensor_rank"]
    data_type = config["tensor_data_type"]
    return f"tm_r{rank}_dt{data_type}_box_{box}"


def box_volume(config):
    volume = 1
    for value in config["box_dim"]:
        volume *= value
    return volume


def load_tensor_map_configs(extra_info_dir: Path):
    csv_path = extra_info_dir / "tensor_map_encode_dump.csv"
    rows = []
    merged = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            config = {
                "tensor_rank": parse_int(row["tensor_rank"]),
                "tensor_data_type": parse_int(row["tensor_data_type"]),
                "global_dim": parse_int_list(row["global_dim"]),
                "global_strides": parse_int_list(row["global_strides"]),
                "box_dim": parse_int_list(row["box_dim"]),
                "element_strides": parse_int_list(row["element_strides"]),
                "interleave": parse_int(row["interleave"]),
                "swizzle": parse_int(row["swizzle"]),
                "l2_promotion": parse_int(row["l2_promotion"]),
                "oob_fill": parse_int(row["oob_fill"]),
            }
            key = (
                config["tensor_rank"],
                config["tensor_data_type"],
                tuple(config["global_dim"]),
                tuple(config["global_strides"]),
                tuple(config["box_dim"]),
                tuple(config["element_strides"]),
                config["interleave"],
                config["swizzle"],
                config["l2_promotion"],
                config["oob_fill"],
            )
            if key not in merged:
                config_id = build_config_id(config)
                merged[key] = {
                    "config_id": config_id,
                    **config,
                    "source_dump_ids": [parse_int(row["dump_id"])],
                }
            else:
                merged[key]["source_dump_ids"].append(parse_int(row["dump_id"]))
            rows.append(
                {
                    "tensor_map_ptr_hex": normalize_hex(row.get("tensor_map_ptr_hex")),
                    "blob_path": row.get("blob_path", ""),
                    "config_id": merged[key]["config_id"],
                }
            )
    configs = sorted(merged.values(), key=lambda item: item["config_id"])
    return rows, configs


def is_descriptor_backed_tma_consumer_opcode(opcode):
    return opcode and opcode.startswith(("UTMALDG", "UTMASTG", "UTMAPF", "UTMAREDG"))


def build_config_ids_by_descriptor_ptr(tensor_rows):
    mapping = collections.defaultdict(set)
    for row in tensor_rows:
        descriptor_ptr_hex = row.get("tensor_map_ptr_hex")
        if not descriptor_ptr_hex:
            continue
        mapping[descriptor_ptr_hex].add(row["config_id"])
    return {key: sorted(values) for key, values in mapping.items()}


def sha256_file(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_launch_arg_config_ids(extra_info_dir: Path, tensor_rows):
    encode_hash_to_config_ids = collections.defaultdict(set)
    for row in tensor_rows:
        blob_path_raw = row.get("blob_path")
        if not blob_path_raw:
            continue
        blob_path = (extra_info_dir / Path(blob_path_raw).relative_to("traces/extra_info")
                     if blob_path_raw.startswith("traces/extra_info/")
                     else Path(blob_path_raw))
        if not blob_path.exists():
            continue
        encode_hash_to_config_ids[sha256_file(blob_path)].add(row["config_id"])

    csv_path = extra_info_dir / "kernel_launch_arg_dump.csv"
    if not csv_path.exists():
        return {}
    launch_arg_config_ids = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            blob_path = Path(row["blob_path"])
            if not blob_path.exists():
                continue
            arg_key = (
                parse_int(row["kernel_id"]),
                parse_int(row["unique_function_id"]),
                parse_int(row["arg_index"]),
            )
            blob_hash = sha256_file(blob_path)
            candidate_config_ids = sorted(encode_hash_to_config_ids.get(blob_hash, []))
            launch_arg_config_ids[arg_key] = candidate_config_ids
    return launch_arg_config_ids


def load_runtime_groups(extra_info_dir: Path, pc_opcode_map, pc_only_opcode_map):
    csv_path = extra_info_dir / "tma_desc_runtime_debug.csv"
    if not csv_path.exists():
        csv_path = extra_info_dir / "utmaldg_runtime_debug.csv"
    grouped = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            unique_function_id = parse_int(row["unique_function_id"])
            pc_hex = normalize_hex(row["pc_hex"])
            opcode = resolve_runtime_opcode(
                pc_opcode_map, pc_only_opcode_map, unique_function_id, pc_hex
            )
            if not is_descriptor_backed_tma_consumer_opcode(opcode):
                continue
            desc_value_lo = parse_int(row["desc_value_lo"])
            desc_value_hi = parse_int(row["desc_value_hi"])
            descriptor_ptr = ((desc_value_hi & 0xFFFFFFFF) << 32) | (desc_value_lo & 0xFFFFFFFF)
            key = (
                unique_function_id,
                pc_hex,
                f"0x{descriptor_ptr:016x}",
            )
            if key not in grouped:
                grouped[key] = {
                    "unique_function_id": key[0],
                    "pc_hex": key[1],
                    "descriptor_ptr_hex": key[2],
                    "handle_hi_hex": f"0x{desc_value_hi & 0xFFFFFFFF:08x}",
                    "desc_reg_ids": set(),
                    "sample_count": 0,
                    "opcode": opcode,
                    "kernel_ids": set(),
                }
            grouped[key]["desc_reg_ids"].add(parse_int(row["desc_reg_id"]))
            grouped[key]["sample_count"] += 1
            grouped[key]["kernel_ids"].add(parse_int(row["kernel_id"]))
    entries = []
    for item in grouped.values():
        item["desc_reg_ids"] = sorted(item["desc_reg_ids"])
        item["kernel_ids"] = sorted(item["kernel_ids"])
        entries.append(item)
    entries.sort(key=lambda item: (item["unique_function_id"], item["pc_hex"], item["descriptor_ptr_hex"]))
    return entries


def resolve_runtime_opcode(pc_opcode_map, pc_only_opcode_map, unique_function_id, pc_hex):
    opcode = pc_opcode_map.get((unique_function_id, pc_hex))
    if opcode is not None:
        return opcode
    return pc_only_opcode_map.get(pc_hex)


def collect_pc_opcode_maps(
    node,
    current_unique_function_id=None,
    mapping=None,
    pc_only_candidates=None,
):
    if mapping is None:
        mapping = {}
    if pc_only_candidates is None:
        pc_only_candidates = collections.defaultdict(set)
    if isinstance(node, dict):
        next_unique_function_id = current_unique_function_id
        if "unique_function_id" in node and node["unique_function_id"] is not None:
            next_unique_function_id = node["unique_function_id"]
        if "pc_hex" in node and "opcode" in node:
            candidate_opcode = node["opcode"]
            pc_hex = node["pc_hex"]
            pc_only_candidates[pc_hex].add(candidate_opcode)
            if next_unique_function_id is not None:
                key = (next_unique_function_id, pc_hex)
                existing_opcode = mapping.get(key)
                if existing_opcode is None:
                    mapping[key] = candidate_opcode
                elif existing_opcode == candidate_opcode:
                    pass
                elif extract_rank_from_opcode(candidate_opcode) is not None and extract_rank_from_opcode(existing_opcode) is None:
                    mapping[key] = candidate_opcode
        for value in node.values():
            collect_pc_opcode_maps(
                value, next_unique_function_id, mapping, pc_only_candidates
            )
    elif isinstance(node, list):
        for value in node:
            collect_pc_opcode_maps(
                value, current_unique_function_id, mapping, pc_only_candidates
            )
    return mapping, pc_only_candidates


def load_pc_opcode_map(extra_info_dir: Path):
    json_path = extra_info_dir / "tma_discovery.json"
    if not json_path.exists():
        return {}, {}
    data = json.loads(json_path.read_text())
    mapping, pc_only_candidates = collect_pc_opcode_maps(data)
    pc_only_opcode_map = {}
    for pc_hex, candidate_opcodes in pc_only_candidates.items():
        if len(candidate_opcodes) == 1:
            pc_only_opcode_map[pc_hex] = next(iter(candidate_opcodes))
    return mapping, pc_only_opcode_map


def extract_rank_from_opcode(opcode):
    if not opcode or "." not in opcode:
        return None
    suffix = opcode.split(".")[-1]
    if suffix.endswith("D") and suffix[:-1].isdigit():
        return int(suffix[:-1])
    return None


def parse_instruction_operands(inst):
    return [operand.get("operand_string", "") for operand in inst.get("operands", [])]


def parse_ureg_pair_from_operand(operand):
    match = re.fullmatch(r"\[UR(\d+)\]", operand.strip())
    if not match:
        return None
    return int(match.group(1))


def parse_uadd_base_pair(inst, desc_reg_id):
    operands = parse_instruction_operands(inst)
    if inst.get("op_code") != "UIADD3" or len(operands) < 3:
        return None
    if operands[0] != f"UR{desc_reg_id}":
        return None
    base_match = re.fullmatch(r"UR(\d+)", operands[2].strip())
    if not base_match:
        return None
    immediate_raw = operands[3].strip() if len(operands) >= 4 else ""
    immediate = int(immediate_raw, 0) if immediate_raw.startswith("0x") else None
    if immediate != 0x30:
        return None
    return int(base_match.group(1))


def parse_uaddx_base_pair(inst, desc_reg_id_plus_one):
    operands = parse_instruction_operands(inst)
    if inst.get("op_code") != "UIADD3.X" or len(operands) < 3:
        return None
    if operands[0] != f"UR{desc_reg_id_plus_one}":
        return None
    base_match = re.fullmatch(r"UR(\d+)", operands[2].strip())
    if not base_match:
        return None
    return int(base_match.group(1)) - 1


def parse_uldc64_slot_offset(inst, base_reg_id):
    operands = parse_instruction_operands(inst)
    if inst.get("op_code") != "ULDC.64" or len(operands) < 2:
        return None
    if operands[0] != f"UR{base_reg_id}":
        return None
    match = re.fullmatch(r"c\[0x0\]\[(0x[0-9a-fA-F]+)\]", operands[1].strip())
    if not match:
        return None
    return int(match.group(1), 0)


def load_kernel_instructions_by_ufid(extra_info_dir: Path):
    json_path = extra_info_dir / "enhanced_execution_info.json"
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text())
    mapping = {}
    for kernel in data.get("kernels", []):
        ufid = kernel.get("unique_function_id")
        if ufid is None:
            continue
        instructions = kernel.get("instructions", [])
        pc_to_index = {
            normalize_hex(hex(inst.get("pc_num_dec", 0))): idx
            for idx, inst in enumerate(instructions)
        }
        mapping[int(ufid)] = {
            "kernel_name": kernel.get("kernel_name", ""),
            "instructions": instructions,
            "pc_to_index": pc_to_index,
        }
    return mapping


def canonicalize_kernel_name(kernel_name):
    return re.sub(r"___\d+$", "", kernel_name or "")


def load_cubin_kparam_info(extra_info_dir: Path):
    cubin_paths = sorted((extra_info_dir / "cubin").glob("**/*.cubin"))
    kparam_info_by_kernel = collections.defaultdict(dict)
    section_kernel_name = None
    pending_kparam_kernel_name = None
    for cubin_path in cubin_paths:
        result = subprocess.run(
            ["cuobjdump", "--dump-elf", str(cubin_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        for raw_line in result.stdout.splitlines():
            line = raw_line.rstrip()
            if line.startswith(".nv.info."):
                section_kernel_name = canonicalize_kernel_name(
                    line[len(".nv.info.") :]
                )
                pending_kparam_kernel_name = None
                continue
            if "Attribute:\tEIATTR_KPARAM_INFO" in line:
                pending_kparam_kernel_name = section_kernel_name
                continue
            if pending_kparam_kernel_name is None:
                continue
            if "Value:\tIndex :" not in line:
                continue
            match = re.search(
                r"Ordinal : 0x([0-9a-fA-F]+)\s+Offset  : 0x([0-9a-fA-F]+)\s+Size    : 0x([0-9a-fA-F]+)",
                line,
            )
            if match is None:
                continue
            ordinal = int(match.group(1), 16)
            offset = int(match.group(2), 16)
            size = int(match.group(3), 16)
            kparam_info_by_kernel[pending_kparam_kernel_name][offset] = {
                "ordinal": ordinal,
                "size": size,
            }
            pending_kparam_kernel_name = None
    return kparam_info_by_kernel


def resolve_runtime_launch_arg_bindings(
    runtime_groups, kernel_instructions_by_ufid, cubin_kparam_info
):
    runtime_launch_arg_bindings = {}
    for entry in runtime_groups:
        ufid = entry["unique_function_id"]
        function_info = kernel_instructions_by_ufid.get(ufid)
        if function_info is None:
            continue
        inst_index = function_info["pc_to_index"].get(entry["pc_hex"])
        if inst_index is None:
            continue
        instructions = function_info["instructions"]
        desc_reg_id = entry["desc_reg_ids"][0] if entry["desc_reg_ids"] else None
        if desc_reg_id is None:
            continue
        consumer_operands = parse_instruction_operands(instructions[inst_index])
        if len(consumer_operands) < 2:
            continue
        consumer_desc_reg_id = parse_ureg_pair_from_operand(consumer_operands[1])
        if consumer_desc_reg_id is None:
            continue
        base_reg_id = None
        descriptor_kparam_offset = None
        for idx in range(inst_index - 1, -1, -1):
            candidate_operands = parse_instruction_operands(instructions[idx])
            if instructions[idx].get("op_code") != "UIADD3" or len(candidate_operands) < 4:
                continue
            if candidate_operands[0] != f"UR{consumer_desc_reg_id}":
                continue
            base_match = re.fullmatch(r"UR(\d+)", candidate_operands[2].strip())
            if base_match is None:
                continue
            immediate_raw = candidate_operands[3].strip()
            if not immediate_raw.startswith("0x"):
                continue
            descriptor_kparam_offset = int(immediate_raw, 0)
            base_reg_id = int(base_match.group(1))
            break
        if base_reg_id is None or descriptor_kparam_offset is None:
            continue
        base_slot_offset = None
        for idx in range(inst_index - 1, -1, -1):
            candidate = parse_uldc64_slot_offset(instructions[idx], base_reg_id)
            if candidate is not None:
                base_slot_offset = candidate
                break
        kernel_name = canonicalize_kernel_name(function_info["kernel_name"])
        kparam_entry = cubin_kparam_info.get(kernel_name, {}).get(
            descriptor_kparam_offset
        )
        if kparam_entry is None:
            continue
        runtime_launch_arg_bindings[
            (ufid, entry["pc_hex"], entry["descriptor_ptr_hex"])
        ] = {
            "arg_index": kparam_entry["ordinal"],
            "kparam_offset_hex": f"0x{descriptor_kparam_offset:x}",
            "kparam_size": kparam_entry["size"],
            "base_slot_offset_hex": (
                f"0x{base_slot_offset:x}" if base_slot_offset is not None else None
            ),
        }
    return runtime_launch_arg_bindings


def build_resolver_entries(
    runtime_groups,
    config_ids_by_descriptor_ptr,
    runtime_launch_arg_bindings,
    launch_arg_config_ids,
):
    resolver = []
    for entry in runtime_groups:
        candidate_config_ids = config_ids_by_descriptor_ptr.get(
            entry["descriptor_ptr_hex"], []
        )
        mapping_method = "exact_descriptor_ptr_match"
        launch_arg_binding = runtime_launch_arg_bindings.get(
            (entry["unique_function_id"], entry["pc_hex"], entry["descriptor_ptr_hex"])
        )
        if not candidate_config_ids and launch_arg_binding is not None:
            launch_candidate_config_ids = set()
            for kernel_id in entry.get("kernel_ids", []):
                launch_candidate_config_ids.update(
                    launch_arg_config_ids.get(
                        (kernel_id, entry["unique_function_id"],
                         launch_arg_binding["arg_index"]),
                        [],
                    )
                )
            candidate_config_ids = sorted(launch_candidate_config_ids)
            mapping_method = "launch_arg_tensor_map_match"
        if len(candidate_config_ids) > 1:
            mapping_method = (
                "ambiguous_launch_arg_tensor_map_match"
                if mapping_method == "launch_arg_tensor_map_match"
                else "ambiguous_exact_descriptor_ptr_match"
            )
        elif not candidate_config_ids:
            mapping_method = (
                "missing_launch_arg_tensor_map_match"
                if launch_arg_binding is not None
                else "missing_exact_descriptor_ptr_match"
            )
        config_id = candidate_config_ids[0] if len(candidate_config_ids) == 1 else None
        confidence = "high" if config_id is not None else "low"
        resolver_entry = {
            "unique_function_id": entry["unique_function_id"],
            "pc_hex": entry["pc_hex"],
            "descriptor_ptr_hex": entry["descriptor_ptr_hex"],
            "handle_hi_hex": entry["handle_hi_hex"],
            "desc_reg_ids": entry["desc_reg_ids"],
            "sample_count": entry["sample_count"],
            "opcode": entry["opcode"],
            "confidence": confidence,
            "mapping_method": mapping_method,
        }
        if launch_arg_binding is not None:
            resolver_entry["launch_arg_index"] = launch_arg_binding["arg_index"]
            resolver_entry["launch_arg_kparam_offset_hex"] = launch_arg_binding[
                "kparam_offset_hex"
            ]
            resolver_entry["launch_arg_kparam_size"] = launch_arg_binding[
                "kparam_size"
            ]
            if launch_arg_binding.get("base_slot_offset_hex") is not None:
                resolver_entry["launch_arg_base_slot_offset_hex"] = launch_arg_binding[
                    "base_slot_offset_hex"
                ]
        if config_id:
            resolver_entry["config_id"] = config_id
        else:
            resolver_entry["candidate_config_ids"] = candidate_config_ids
        resolver.append(resolver_entry)
    return resolver


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("extra_info_dir", type=Path)
    parser.add_argument("--configs-out", type=Path)
    parser.add_argument("--resolver-out", type=Path)
    args = parser.parse_args()

    extra_info_dir = args.extra_info_dir
    pc_opcode_map, pc_only_opcode_map = load_pc_opcode_map(extra_info_dir)
    tensor_rows, configs = load_tensor_map_configs(extra_info_dir)
    config_ids_by_descriptor_ptr = build_config_ids_by_descriptor_ptr(tensor_rows)
    launch_arg_config_ids = load_launch_arg_config_ids(extra_info_dir, tensor_rows)
    runtime_groups = load_runtime_groups(
        extra_info_dir, pc_opcode_map, pc_only_opcode_map
    )
    kernel_instructions_by_ufid = load_kernel_instructions_by_ufid(extra_info_dir)
    cubin_kparam_info = load_cubin_kparam_info(extra_info_dir)
    runtime_launch_arg_bindings = resolve_runtime_launch_arg_bindings(
        runtime_groups, kernel_instructions_by_ufid, cubin_kparam_info
    )
    resolver = build_resolver_entries(
        runtime_groups,
        config_ids_by_descriptor_ptr,
        runtime_launch_arg_bindings,
        launch_arg_config_ids,
    )
    unresolved = [
        entry for entry in resolver
        if entry.get("config_id") is None
    ]
    if unresolved:
        examples = [
            f"ufid={entry['unique_function_id']} pc={entry['pc_hex']} opcode={entry.get('opcode')} "
            f"descriptor_ptr={entry.get('descriptor_ptr_hex')}"
            for entry in unresolved[:8]
        ]
        raise SystemExit(
            "Failed to resolve exact descriptor pointer for executed descriptor-backed TMA sites:\n"
            + "\n".join(examples)
        )

    configs_out = args.configs_out or (extra_info_dir / "tma_descriptor_configs.json")
    resolver_out = args.resolver_out or (extra_info_dir / "tma_descriptor_resolver.json")

    write_json(configs_out, {
        "version": 1,
        "source": {
            "tensor_map_rows": len(tensor_rows),
            "normalized_config_count": len(configs),
        },
        "configs": configs,
    })
    write_json(resolver_out, {
        "version": 1,
        "source": {
            "runtime_group_count": len(runtime_groups),
            "mapping_method": "runtime_descriptor_ptr_or_launch_arg_tensor_map_match",
        },
        "resolver": resolver,
    })


if __name__ == "__main__":
    main()
