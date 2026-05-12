#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def parse_int_list(raw: str):
    raw = (raw or "").strip().strip('"')
    if not raw:
        return []
    return [int(value) for value in raw.split()]


def parse_int(raw: str):
    return int((raw or "0").strip(), 0)


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
            rows.append(row)
    configs = sorted(merged.values(), key=lambda item: item["config_id"])
    return rows, configs


def load_runtime_groups(extra_info_dir: Path):
    csv_path = extra_info_dir / "utmaldg_runtime_debug.csv"
    grouped = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (
                parse_int(row["unique_function_id"]),
                row["pc_hex"],
                f"0x{parse_int(row['desc_value_hi']):08x}",
            )
            if key not in grouped:
                grouped[key] = {
                    "unique_function_id": key[0],
                    "pc_hex": key[1],
                    "handle_hi_hex": key[2],
                    "desc_reg_ids": set(),
                    "sample_count": 0,
                }
            grouped[key]["desc_reg_ids"].add(parse_int(row["desc_reg_id"]))
            grouped[key]["sample_count"] += 1
    entries = []
    for item in grouped.values():
        item["desc_reg_ids"] = sorted(item["desc_reg_ids"])
        entries.append(item)
    entries.sort(key=lambda item: (item["unique_function_id"], item["pc_hex"], item["handle_hi_hex"]))
    return entries


def collect_pc_opcode_map(node, current_unique_function_id=None, mapping=None):
    if mapping is None:
        mapping = {}
    if isinstance(node, dict):
        next_unique_function_id = current_unique_function_id
        if "unique_function_id" in node and node["unique_function_id"] is not None:
            next_unique_function_id = node["unique_function_id"]
        if next_unique_function_id is not None and "pc_hex" in node and "opcode" in node:
            mapping[(next_unique_function_id, node["pc_hex"])] = node["opcode"]
        for value in node.values():
            collect_pc_opcode_map(value, next_unique_function_id, mapping)
    elif isinstance(node, list):
        for value in node:
            collect_pc_opcode_map(value, current_unique_function_id, mapping)
    return mapping


def load_pc_opcode_map(extra_info_dir: Path):
    json_path = extra_info_dir / "tma_discovery.json"
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text())
    return collect_pc_opcode_map(data)


def extract_rank_from_opcode(opcode):
    if not opcode or "." not in opcode:
        return None
    suffix = opcode.split(".")[-1]
    if suffix.endswith("D") and suffix[:-1].isdigit():
        return int(suffix[:-1])
    return None


def derive_handle_family_map_by_rank(configs):
    grouped = {}
    for config in configs:
        grouped.setdefault(config["tensor_rank"], []).append(config)
    mapping = {}
    for rank, rank_configs in grouped.items():
        sorted_configs = sorted(rank_configs, key=box_volume, reverse=True)
        rank_map = {}
        if len(sorted_configs) >= 1:
            rank_map["0x14f00000"] = sorted_configs[0]["config_id"]
        if len(sorted_configs) >= 2:
            rank_map["0x12f00000"] = sorted_configs[-1]["config_id"]
        mapping[rank] = rank_map
    return mapping


def build_resolver_entries(runtime_groups, handle_map_by_rank, configs, pc_opcode_map):
    all_config_ids = [config["config_id"] for config in configs]
    configs_by_rank = {}
    for config in configs:
        configs_by_rank.setdefault(config["tensor_rank"], []).append(config["config_id"])
    resolver = []
    for entry in runtime_groups:
        opcode = pc_opcode_map.get((entry["unique_function_id"], entry["pc_hex"]))
        rank = extract_rank_from_opcode(opcode)
        rank_handle_map = handle_map_by_rank.get(rank, {})
        config_id = rank_handle_map.get(entry["handle_hi_hex"])
        resolver_entry = {
            "unique_function_id": entry["unique_function_id"],
            "pc_hex": entry["pc_hex"],
            "handle_hi_hex": entry["handle_hi_hex"],
            "desc_reg_ids": entry["desc_reg_ids"],
            "sample_count": entry["sample_count"],
            "opcode": opcode,
            "confidence": "medium" if config_id else "low",
            "mapping_method": "handle_hi_to_box_dim_family_with_opcode_rank",
        }
        if config_id:
            resolver_entry["config_id"] = config_id
        else:
            resolver_entry["candidate_config_ids"] = configs_by_rank.get(rank, all_config_ids)
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
    tensor_rows, configs = load_tensor_map_configs(extra_info_dir)
    runtime_groups = load_runtime_groups(extra_info_dir)
    pc_opcode_map = load_pc_opcode_map(extra_info_dir)
    handle_map_by_rank = derive_handle_family_map_by_rank(configs)
    resolver = build_resolver_entries(runtime_groups, handle_map_by_rank, configs, pc_opcode_map)

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
            "mapping_method": "handle_hi_to_box_dim_family_with_opcode_rank",
        },
        "resolver": resolver,
    })


if __name__ == "__main__":
    main()
