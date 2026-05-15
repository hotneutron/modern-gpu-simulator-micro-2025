#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def parse_jsonl(path: Path):
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_tma_entries(node, current_unique_function_id=None, entries=None):
    if entries is None:
        entries = []
    if isinstance(node, dict):
        next_unique_function_id = current_unique_function_id
        if "unique_function_id" in node and node["unique_function_id"] is not None:
            next_unique_function_id = node["unique_function_id"]
        if "pc_hex" in node and "opcode" in node and "text" in node:
            entries.append({
                "unique_function_id": next_unique_function_id,
                "pc_hex": node["pc_hex"],
                "opcode": node["opcode"],
                "text": node["text"],
                "role": node.get("role"),
                "desc_refs": node.get("desc_refs", []),
                "desc_regs": node.get("desc_regs", []),
                "support_regs": node.get("support_regs", []),
            })
        for value in node.values():
            collect_tma_entries(value, next_unique_function_id, entries)
    elif isinstance(node, list):
        for value in node:
            collect_tma_entries(value, current_unique_function_id, entries)
    return entries


def parse_operands_from_text(opcode: str, text: str):
    if opcode not in text:
        return []
    suffix = text.split(opcode, 1)[1]
    suffix = suffix.split(";", 1)[0]
    raw_operands = [token.strip() for token in suffix.split(",") if token.strip()]
    parsed = []
    for index, operand_text in enumerate(raw_operands, start=1):
        reg_ids = [int(value) for value in re.findall(r"UR(\d+)", operand_text)]
        parsed.append({
            "position": index,
            "text": operand_text,
            "reg_ids": reg_ids,
        })
    return parsed


def load_descriptor_refs(extra_info_dir: Path):
    path = extra_info_dir / "tma_descriptor_resolver.json"
    if not path.exists():
      return {}
    data = json.loads(path.read_text())
    mapping = {}
    for entry in data.get("resolver", []):
        key = (entry.get("unique_function_id"), entry.get("pc_hex"))
        mapping.setdefault(key, {
            "config_ids": set(),
            "desc_reg_ids": set(),
        })
        config_id = entry.get("config_id")
        if config_id:
            mapping[key]["config_ids"].add(config_id)
        for reg_id in entry.get("desc_reg_ids", []):
            mapping[key]["desc_reg_ids"].add(reg_id)
    normalized = {}
    for key, value in mapping.items():
        normalized[key] = {
            "config_ids": sorted(value["config_ids"]),
            "desc_reg_ids": sorted(value["desc_reg_ids"]),
        }
    return normalized


def build_runtime_groups(runtime_rows):
    grouped = {}
    for row in runtime_rows:
        key = (row.get("unique_function_id"), row.get("pc_hex"), row.get("callback_index"))
        if key not in grouped:
            grouped[key] = {
                "callback_index": row.get("callback_index"),
                "operand_type": row.get("operand_type"),
                "mem_type": row.get("mem_type"),
                "operand_reg_ids": set(),
                "value_lo_samples": [],
                "value_hi_samples": [],
                "value_2_samples": [],
                "value_3_samples": [],
                "sample_count": 0,
            }
        group = grouped[key]
        operand_reg_id = row.get("operand_reg_id")
        if operand_reg_id is not None and operand_reg_id >= 0:
            group["operand_reg_ids"].add(operand_reg_id)
        value_lo = row.get("value_lo")
        value_hi = row.get("value_hi")
        value_2 = row.get("value_2")
        value_3 = row.get("value_3")
        if value_lo not in group["value_lo_samples"] and len(group["value_lo_samples"]) < 16:
            group["value_lo_samples"].append(value_lo)
        if value_hi not in group["value_hi_samples"] and len(group["value_hi_samples"]) < 16:
            group["value_hi_samples"].append(value_hi)
        if value_2 not in group["value_2_samples"] and len(group["value_2_samples"]) < 16:
            group["value_2_samples"].append(value_2)
        if value_3 not in group["value_3_samples"] and len(group["value_3_samples"]) < 16:
            group["value_3_samples"].append(value_3)
        group["sample_count"] += 1
    normalized = {}
    for key, value in grouped.items():
        value["operand_reg_ids"] = sorted(value["operand_reg_ids"])
        normalized[key] = value
    return normalized


def keep_nontrivial_sample_field(samples):
    if not samples:
        return False
    return any(sample not in (0, None) for sample in samples)


def find_matching_runtime_keys(runtime_groups, unique_function_id, pc_hex):
    direct = sorted(
        key for key in runtime_groups
        if key[0] == unique_function_id and key[1] == pc_hex
    )
    if direct:
        return direct
    if unique_function_id is None:
        by_pc = sorted(
            key for key in runtime_groups
            if key[1] == pc_hex
        )
        return by_pc
    return []


def infer_operand_kinds(entry):
    opcode = entry["opcode"] or ""
    operands = entry["operands"]
    if opcode.startswith("UBLKCP") and len(operands) == 3:
        return {
            "operand_1": "dst_smem_base_or_cursor",
            "operand_2": "src_gmem_base_or_cursor",
            "operand_3": "covered_bytes_or_encoded_span",
        }
    if opcode.startswith("UBLKRED") and len(operands) == 3 and not entry.get("descriptor_ref"):
        return {
            "operand_1": "dst_base",
            "operand_2": "src_base",
            "operand_3": "covered_bytes_or_encoded_span",
        }
    if opcode.startswith("UBLKRED") and len(operands) >= 3 and entry.get("descriptor_ref"):
        return {
            "operand_1": "dst_or_coord_state",
            "operand_2": "src_state",
            "operand_3": "covered_bytes_or_encoded_span",
            "descriptor": "tensor_map",
        }
    descriptor_ref = entry.get("descriptor_ref")
    if opcode.startswith("UTMALDG") and descriptor_ref:
        inferred = {
            "descriptor": "tensor_map_shape",
        }
        if len(operands) >= 1:
            inferred["operand_1"] = "load_dst_state"
        if len(operands) >= 2:
            inferred["operand_2"] = "load_coord_or_state"
        if len(operands) >= 3:
            inferred["operand_3"] = "tensor_map_descriptor"
        return inferred
    if opcode.startswith("UTMASTG") and descriptor_ref:
        inferred = {
            "descriptor": "tensor_map_shape",
        }
        if len(operands) >= 1:
            inferred["operand_1"] = "store_dst_state"
        if len(operands) >= 2:
            inferred["operand_2"] = "store_src_or_state"
        return inferred
    return {}


def infer_operand_form(opcode, descriptor_ref):
    if opcode.startswith("UBLKCP"):
        return "bulk"
    if opcode.startswith("UBLKRED") and not descriptor_ref:
        return "bulk"
    if opcode.startswith("UBLKRED") and descriptor_ref:
        return "descriptor_backed"
    if (opcode.startswith("UTMALDG") or opcode.startswith("UTMASTG")) and descriptor_ref:
        return "descriptor_shape_driven"
    return "generic"


def infer_ublkred_element_size_bytes(opcode):
    if ".F16" in opcode or ".BF16" in opcode:
        return 2
    if ".F32" in opcode or ".S32" in opcode or ".U32" in opcode:
        return 4
    return None


def infer_runtime_semantics(opcode, operands, callback_groups, descriptor_ref):
    semantics = {}
    if opcode.startswith("UBLKCP") and len(operands) == 3:
        for callback in callback_groups:
            position = callback.get("operand_position")
            if position == 1:
                semantics["operand_1"] = {
                    "kind": "dst_smem_base_or_cursor",
                    "runtime_source": "memory_ref",
                }
            elif position == 2:
                semantics["operand_2"] = {
                    "kind": "src_gmem_base_or_cursor",
                    "runtime_source": "memory_ref",
                }
            elif position == 3:
                raw_samples = callback.get("value_lo_samples", [])
                semantics["operand_3"] = {
                    "kind": "covered_bytes",
                    "runtime_source": "uniform_reg",
                    "encoding": "16B_units",
                    "scale_bytes": 16,
                    "raw_value_samples": raw_samples,
                    "decoded_byte_samples": [value * 16 for value in raw_samples],
                }
    elif opcode.startswith("UBLKRED") and len(operands) == 3 and not descriptor_ref:
        element_size_bytes = infer_ublkred_element_size_bytes(opcode)
        for callback in callback_groups:
            position = callback.get("operand_position")
            if position == 1:
                semantics["operand_1"] = {
                    "kind": "dst_base",
                    "runtime_source": "memory_ref",
                }
            elif position == 2:
                semantics["operand_2"] = {
                    "kind": "src_base",
                    "runtime_source": "memory_ref",
                }
            elif position == 3:
                raw_samples = callback.get("value_lo_samples", [])
                semantics["operand_3"] = {
                    "kind": "covered_bytes",
                    "runtime_source": "uniform_reg",
                    "encoding": "16B_units",
                    "scale_bytes": 16,
                    "raw_value_samples": raw_samples,
                    "decoded_byte_samples": [value * 16 for value in raw_samples],
                }
                if element_size_bytes:
                    semantics["operand_3"]["element_size_bytes"] = element_size_bytes
                    semantics["operand_3"]["decoded_element_samples"] = [
                        (value * 16) // element_size_bytes for value in raw_samples
                    ]
    elif opcode.startswith("UBLKRED") and len(operands) >= 3 and descriptor_ref:
        semantics["descriptor"] = {
            "kind": "tensor_map",
        }
    elif opcode.startswith("UTMALDG") and descriptor_ref:
        semantics["descriptor"] = {
            "kind": "tensor_map",
            "shape_role": "descriptor_shape_driven",
        }
        semantics["operand_model"] = {
            "kind": "operand_state_driven",
        }
        semantics["operand_1"] = {
            "kind": "load_dst_state",
            "runtime_source": "memory_ref",
        }
        semantics["operand_2"] = {
            "kind": "load_coord_or_state",
            "runtime_source": "memory_ref",
        }
    elif opcode.startswith("UTMASTG") and descriptor_ref:
        semantics["descriptor"] = {
            "kind": "tensor_map",
            "shape_role": "descriptor_shape_driven",
        }
        semantics["operand_model"] = {
            "kind": "operand_state_driven",
        }
        semantics["operand_1"] = {
            "kind": "store_dst_state",
            "runtime_source": "memory_ref",
        }
        semantics["operand_2"] = {
            "kind": "store_src_or_state",
            "runtime_source": "memory_ref",
        }
    return semantics


def build_operand_entries(operands, inferred_operand_kinds):
    entries = {}
    for operand in operands:
        key = f"operand_{operand['position']}"
        entries[key] = {
            "position": operand["position"],
            "text": operand["text"],
            "reg_ids": operand["reg_ids"],
            "kind": inferred_operand_kinds.get(key, "unknown"),
        }
    return entries


def build_runtime_observed_values(opcode, callback_groups, descriptor_ref):
    observed = {}
    for callback in callback_groups:
        position = callback.get("operand_position")
        if position is None:
            continue
        key = f"operand_{position}"
        entry = {
            "callback_index": callback.get("callback_index"),
            "operand_type": callback.get("operand_type"),
            "mem_type": callback.get("mem_type"),
            "operand_reg_ids": callback.get("operand_reg_ids", []),
            "raw_value_lo_samples": callback.get("value_lo_samples", []),
            "sample_count": callback.get("sample_count", 0),
        }
        if callback.get("operand_type") in ("REGULAR_2_REGS", "UNIFORM_2_REGS") and keep_nontrivial_sample_field(callback.get("value_hi_samples", [])):
            entry["raw_value_hi_samples"] = callback.get("value_hi_samples", [])
        if keep_nontrivial_sample_field(callback.get("value_2_samples", [])):
            entry["raw_value_2_samples"] = callback.get("value_2_samples", [])
        if keep_nontrivial_sample_field(callback.get("value_3_samples", [])):
            entry["raw_value_3_samples"] = callback.get("value_3_samples", [])
        observed[key] = entry
    if opcode.startswith("UBLKCP") and "operand_3" in observed:
        raw_samples = observed["operand_3"]["raw_value_lo_samples"]
        observed["operand_3"]["decoded_byte_samples"] = [value * 16 for value in raw_samples]
    if opcode.startswith("UBLKRED") and "operand_3" in observed and not descriptor_ref:
        raw_samples = observed["operand_3"]["raw_value_lo_samples"]
        element_size_bytes = infer_ublkred_element_size_bytes(opcode)
        observed["operand_3"]["decoded_byte_samples"] = [value * 16 for value in raw_samples]
        if element_size_bytes:
            observed["operand_3"]["element_size_bytes"] = element_size_bytes
            observed["operand_3"]["decoded_element_samples"] = [
                (value * 16) // element_size_bytes for value in raw_samples
            ]
    return observed


def build_static_decode_formula(opcode, operands, descriptor_ref):
    if opcode.startswith("UBLKCP") and len(operands) == 3:
        return {
            "kind": "sass_validated_formula",
            "applies_to": "bulk_ublkcp",
            "operands": {
                "operand_1": {
                    "kind": "dst_smem_base_or_cursor",
                },
                "operand_2": {
                    "kind": "src_gmem_base_or_cursor",
                },
                "operand_3": {
                    "kind": "covered_bytes",
                    "encoding": "16B_units",
                    "formula": {
                        "covered_bytes": "operand_3 * 16",
                    },
                },
            },
            "notes": [
                "A dedicated UBLKCP microbench lowered cp.async.bulk.shared::cta.global to UBLKCP.S.G with the same operand ordering.",
                "Callback 0 is the shared-memory destination reference.",
                "Callback 1 is the global-memory source reference.",
                "Callback 2 is the scalar uniform operand.",
                "The microbench validates operand_3 as a 16-byte-unit span field."
            ],
        }
    if opcode.startswith("UBLKRED") and len(operands) == 3 and not descriptor_ref:
        element_size_bytes = infer_ublkred_element_size_bytes(opcode)
        operand_3_entry = {
            "kind": "covered_bytes",
            "encoding": "16B_units",
            "formula": {
                "covered_bytes": "operand_3 * 16",
            },
        }
        if element_size_bytes:
            operand_3_entry["element_size_bytes"] = element_size_bytes
            operand_3_entry["formula"]["covered_elements"] = f"(operand_3 * 16) / {element_size_bytes}"
        return {
            "kind": "sass_validated_formula",
            "applies_to": "bulk_ublkred",
            "operands": {
                "operand_1": {
                    "kind": "dst_base",
                },
                "operand_2": {
                    "kind": "src_base",
                },
                "operand_3": operand_3_entry,
            },
            "notes": [
                "Callback 0 is the destination memory reference.",
                "Callback 1 is the source memory reference.",
                "Callback 2 is the scalar uniform operand.",
            ],
        }
    if opcode.startswith("UBLKRED") and descriptor_ref:
        return {
            "kind": "descriptor_backed_formula_pending",
            "notes": [
                "Descriptor-backed UBLKRED needs additional validation before assigning a concrete operand-3 decode formula."
            ],
        }
    if opcode.startswith("UTMALDG") and descriptor_ref:
        return {
            "kind": "descriptor_shape_driven_formula_pending",
            "notes": [
                "UTMALDG logical transfer shape is primarily descriptor-driven.",
                "Runtime operands behave like load destination / coordinate state rather than a direct byte-count field."
            ],
        }
    if opcode.startswith("UTMASTG") and descriptor_ref:
        return {
            "kind": "descriptor_shape_driven_formula_pending",
            "notes": [
                "UTMASTG logical transfer shape is primarily descriptor-driven.",
                "Runtime operands behave like store destination / source state rather than a direct byte-count field."
            ],
        }
    return None


def build_resolver(extra_info_dir: Path):
    discovery_path = extra_info_dir / "tma_discovery.json"
    runtime_path = extra_info_dir / "tma_runtime_operand_debug.jsonl"
    if not discovery_path.exists() or not runtime_path.exists():
        return None

    discovery_data = json.loads(discovery_path.read_text())
    discovery_entries = collect_tma_entries(discovery_data)
    descriptor_refs = load_descriptor_refs(extra_info_dir)
    runtime_rows = parse_jsonl(runtime_path)
    runtime_groups = build_runtime_groups(runtime_rows)

    resolver = []
    for entry in discovery_entries:
        key = (entry["unique_function_id"], entry["pc_hex"])
        operands = parse_operands_from_text(entry["opcode"], entry["text"])
        callback_groups = []
        matched_runtime_keys = find_matching_runtime_keys(
            runtime_groups, entry["unique_function_id"], entry["pc_hex"]
        )
        for matched_key in matched_runtime_keys:
            callback_group = dict(runtime_groups[matched_key])
            callback_index = matched_key[2]
            if callback_index < len(operands):
                callback_group["operand_position"] = callback_index + 1
                callback_group["operand_text"] = operands[callback_index]["text"]
            callback_groups.append(callback_group)
        resolved_unique_function_id = entry["unique_function_id"]
        if resolved_unique_function_id is None and matched_runtime_keys:
            unique_function_ids = sorted({matched_key[0] for matched_key in matched_runtime_keys if matched_key[0] is not None})
            if len(unique_function_ids) == 1:
                resolved_unique_function_id = unique_function_ids[0]
        descriptor_ref = descriptor_refs.get(key)
        if descriptor_ref is None and entry["unique_function_id"] is None:
            descriptor_ref = next(
                (value for (ufid, pc), value in descriptor_refs.items() if pc == entry["pc_hex"]),
                None,
            )
        inferred_operand_kinds = infer_operand_kinds({
            "opcode": entry["opcode"],
            "operands": operands,
            "descriptor_ref": descriptor_ref,
        })
        resolver_entry = {
            "unique_function_id": resolved_unique_function_id,
            "pc_hex": entry["pc_hex"],
            "opcode": entry["opcode"],
            "text": entry["text"],
            "role": entry["role"],
            "operand_form": infer_operand_form(entry["opcode"], descriptor_ref),
            "operands": build_operand_entries(operands, inferred_operand_kinds),
            "support_regs": entry.get("support_regs", []),
            "desc_refs": entry.get("desc_refs", []),
            "desc_regs": entry.get("desc_regs", []),
            "debug": {
                "raw_runtime_callbacks": callback_groups,
            },
            "inferred_runtime_semantics": infer_runtime_semantics(
                entry["opcode"], operands, callback_groups, descriptor_ref
            ),
            "runtime_observed_values": build_runtime_observed_values(
                entry["opcode"], callback_groups, descriptor_ref
            ),
            "static_decode_formula": build_static_decode_formula(
                entry["opcode"], operands, descriptor_ref
            ),
        }
        if descriptor_ref:
            resolver_entry["descriptor_ref"] = descriptor_ref
        resolver.append(resolver_entry)
    return {
        "version": 1,
        "source": {
            "discovery_entries": len(discovery_entries),
            "runtime_events": len(runtime_rows),
        },
        "resolver": resolver,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("extra_info_dir", type=Path)
    parser.add_argument("--resolver-out", type=Path)
    args = parser.parse_args()

    extra_info_dir = args.extra_info_dir
    payload = build_resolver(extra_info_dir)
    if payload is None:
        return
    resolver_out = args.resolver_out or (extra_info_dir / "tma_operand_resolver.json")
    resolver_out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
