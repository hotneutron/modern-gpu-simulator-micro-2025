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
    csv_path = extra_info_dir / "tma_desc_runtime_debug.csv"
    if not csv_path.exists():
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
            key = (next_unique_function_id, node["pc_hex"])
            existing_opcode = mapping.get(key)
            candidate_opcode = node["opcode"]
            if existing_opcode is None:
                mapping[key] = candidate_opcode
            elif existing_opcode == candidate_opcode:
                pass
            elif extract_rank_from_opcode(candidate_opcode) is not None and extract_rank_from_opcode(existing_opcode) is None:
                mapping[key] = candidate_opcode
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


def infer_rank_from_related_usage(entry, runtime_groups, pc_opcode_map):
    preferred_ranks = set()
    fallback_ranks = set()
    entry_desc_regs = set(entry["desc_reg_ids"])
    for other in runtime_groups:
        if other is entry:
            continue
        if other["unique_function_id"] != entry["unique_function_id"]:
            continue
        if other["handle_hi_hex"] != entry["handle_hi_hex"]:
            continue
        other_opcode = pc_opcode_map.get((other["unique_function_id"], other["pc_hex"]))
        other_rank = extract_rank_from_opcode(other_opcode)
        if other_rank is None:
            continue
        if entry_desc_regs.intersection(other["desc_reg_ids"]):
            preferred_ranks.add(other_rank)
        else:
            fallback_ranks.add(other_rank)
    if len(preferred_ranks) == 1:
        return preferred_ranks.pop(), "same_function_handle_and_desc_reg_rank_inference"
    if len(fallback_ranks) == 1:
        return fallback_ranks.pop(), "same_function_handle_rank_inference"
    return None, None


def derive_handle_family_map_by_rank(configs, runtime_groups, pc_opcode_map):
    grouped = {}
    for config in configs:
        grouped.setdefault(config["tensor_rank"], []).append(config)
    observed_handles_by_rank = {}
    for entry in runtime_groups:
        handle_hi_hex = entry["handle_hi_hex"]
        if handle_hi_hex == "0x00000000":
            continue
        opcode = pc_opcode_map.get((entry["unique_function_id"], entry["pc_hex"]))
        rank = extract_rank_from_opcode(opcode)
        if rank is None:
            rank, _ = infer_rank_from_related_usage(entry, runtime_groups, pc_opcode_map)
        if rank is None:
            continue
        observed_handles_by_rank.setdefault(rank, set()).add(handle_hi_hex)
    mapping = {}
    for rank, rank_configs in grouped.items():
        sorted_configs = sorted(rank_configs, key=box_volume, reverse=True)
        observed_handles = sorted(
            observed_handles_by_rank.get(rank, []),
            key=lambda value: parse_int(value),
            reverse=True,
        )
        rank_map = {}
        for handle_hi_hex, config in zip(observed_handles, sorted_configs):
            rank_map[handle_hi_hex] = config["config_id"]
        mapping[rank] = rank_map
    return mapping


def finalize_unresolved_entries(resolver):
    resolved_by_function_and_desc_reg = {}
    for entry in resolver:
        config_id = entry.get("config_id")
        if not config_id:
            continue
        key = (entry["unique_function_id"], tuple(sorted(entry["desc_reg_ids"])))
        resolved_by_function_and_desc_reg.setdefault(key, set()).add(config_id)

    for entry in resolver:
        if entry.get("config_id"):
            continue
        candidate_config_ids = entry.get("candidate_config_ids", [])
        if len(candidate_config_ids) == 1:
            entry["config_id"] = candidate_config_ids[0]
            entry["confidence"] = "medium"
            entry["mapping_method"] = "single_rank_candidate_config"
            del entry["candidate_config_ids"]
            continue
        key = (entry["unique_function_id"], tuple(sorted(entry["desc_reg_ids"])))
        reused_config_ids = resolved_by_function_and_desc_reg.get(key, set())
        if len(reused_config_ids) == 1:
            entry["config_id"] = next(iter(reused_config_ids))
            entry["confidence"] = "medium"
            entry["mapping_method"] = "same_function_desc_reg_config_reuse"
            del entry["candidate_config_ids"]
    for entry in resolver:
        entry["binding_status"] = "resolved" if entry.get("config_id") else "unresolved"


def build_resolver_entries(runtime_groups, handle_map_by_rank, configs, pc_opcode_map):
    all_config_ids = [config["config_id"] for config in configs]
    configs_by_rank = {}
    for config in configs:
        configs_by_rank.setdefault(config["tensor_rank"], []).append(config["config_id"])
    resolver = []
    for entry in runtime_groups:
        opcode = pc_opcode_map.get((entry["unique_function_id"], entry["pc_hex"]))
        rank = extract_rank_from_opcode(opcode)
        rank_source = "opcode_rank" if rank is not None else None
        if rank is None:
            rank, rank_source = infer_rank_from_related_usage(entry, runtime_groups, pc_opcode_map)
        rank_handle_map = handle_map_by_rank.get(rank, {})
        config_id = rank_handle_map.get(entry["handle_hi_hex"])
        confidence = "medium" if config_id else "low"
        mapping_method = "handle_hi_to_box_dim_family_with_opcode_rank"
        if rank_source == "same_function_handle_and_desc_reg_rank_inference":
            mapping_method = "handle_hi_to_box_dim_family_with_inferred_rank_from_desc_reg_reuse"
        elif rank_source == "same_function_handle_rank_inference":
            mapping_method = "handle_hi_to_box_dim_family_with_inferred_rank_from_handle_reuse"
        resolver_entry = {
            "unique_function_id": entry["unique_function_id"],
            "pc_hex": entry["pc_hex"],
            "handle_hi_hex": entry["handle_hi_hex"],
            "desc_reg_ids": entry["desc_reg_ids"],
            "sample_count": entry["sample_count"],
            "opcode": opcode,
            "inferred_rank": rank,
            "confidence": confidence,
            "mapping_method": mapping_method,
        }
        if config_id:
            resolver_entry["config_id"] = config_id
        else:
            resolver_entry["candidate_config_ids"] = configs_by_rank.get(rank, all_config_ids)
        resolver.append(resolver_entry)
    finalize_unresolved_entries(resolver)
    return resolver


def is_descriptor_involved_opcode(opcode):
    if not opcode:
        return False
    opcode_family = opcode.split()[0]
    descriptor_prefixes = (
        "UTMALDG",
        "UTMAREDG",
        "UTMASTG",
        "UBLKRED",
    )
    return opcode_family.startswith(descriptor_prefixes)


def should_verify_binding(entry):
    opcode = entry.get("opcode")
    return is_descriptor_involved_opcode(opcode)


def format_binding_failure(entry):
    candidate_config_ids = entry.get("candidate_config_ids", [])
    candidate_text = ",".join(candidate_config_ids) if candidate_config_ids else "<none>"
    return (
        "[TMA][DescriptorMapping][Error] executed descriptor-involved site has no "
        "unique final binding: "
        f"ufid={entry['unique_function_id']} "
        f"pc={entry['pc_hex']} "
        f"opcode={entry.get('opcode')} "
        f"handle_hi={entry['handle_hi_hex']} "
        f"desc_regs={entry['desc_reg_ids']} "
        f"mapping_method={entry.get('mapping_method')} "
        f"candidate_configs={candidate_text}"
    )


def verify_executed_bindings_or_fail(resolver):
    failures = [
        entry
        for entry in resolver
        if should_verify_binding(entry) and not entry.get("config_id")
    ]
    if not failures:
        return
    failure_lines = [format_binding_failure(entry) for entry in failures]
    raise SystemExit("\n".join(failure_lines))


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("extra_info_dir", type=Path)
    parser.add_argument("--configs-out", type=Path)
    parser.add_argument("--resolver-out", type=Path)
    parser.add_argument(
        "--fail-on-missing-binding",
        action="store_true",
        help="Fail if any executed descriptor runtime site lacks a unique config_id",
    )
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="Emit only tma_descriptor_configs.json (box_dim/strides/swizzle, still "
             "needed by the simulator) and SKIP the handle_hi->config_id resolver. Use "
             "once the exact (uid,pc) base map supersedes the handle_hi heuristic.",
    )
    args = parser.parse_args()

    extra_info_dir = args.extra_info_dir
    tensor_rows, configs = load_tensor_map_configs(extra_info_dir)

    configs_out = args.configs_out or (extra_info_dir / "tma_descriptor_configs.json")
    write_json(configs_out, {
        "version": 1,
        "source": {
            "tensor_map_rows": len(tensor_rows),
            "normalized_config_count": len(configs),
        },
        "configs": configs,
    })

    if args.configs_only:
        # The handle_hi heuristic resolver is intentionally NOT built; the exact
        # (uid,pc) base map (build_tma_pc_base_map.py) provides base + all descriptor
        # fields per site instead.
        return

    runtime_groups = load_runtime_groups(extra_info_dir)
    pc_opcode_map = load_pc_opcode_map(extra_info_dir)
    handle_map_by_rank = derive_handle_family_map_by_rank(configs, runtime_groups, pc_opcode_map)
    resolver = build_resolver_entries(runtime_groups, handle_map_by_rank, configs, pc_opcode_map)
    if args.fail_on_missing_binding:
        verify_executed_bindings_or_fail(resolver)

    resolver_out = args.resolver_out or (extra_info_dir / "tma_descriptor_resolver.json")
    write_json(resolver_out, {
        "version": 1,
        "source": {
            "runtime_group_count": len(runtime_groups),
            "mapping_method": "runtime_observed_handle_hi_to_box_dim_family_with_opcode_rank",
        },
        "resolver": resolver,
    })


if __name__ == "__main__":
    main()
