#!/usr/bin/env python3
"""Validate TMA LOAD bytes against exact runtime mbarrier contracts."""

import argparse
import glob
import importlib.util
import json
from collections import Counter
from pathlib import Path


def load_vpb_parser(script_dir):
    parser_path = script_dir / "build_spin_wait_sidecar.py"
    spec = importlib.util.spec_from_file_location("spin_wait_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.vpb_parse_full


def parse_kernel_aware_map(path):
    data = json.loads(path.read_text())
    if data.get("version") != 2 or not data.get("validation_complete"):
        raise SystemExit(
            f"TMA base map is not validated schema v2: {path}"
        )
    if data.get("key_fields") != [
            "trace_kernel_id", "unique_function_id", "pc_hex"]:
        raise SystemExit(f"invalid TMA base-map key schema: {path}")
    validation = data.get("validation", {})
    if (
        int(validation.get("sites_total", 0)) <= 0
        or int(validation.get("chain_resolved", 0))
        != int(validation.get("sites_total", 0))
        or int(validation.get("ambiguous", 1)) != 0
        or int(validation.get("unresolved", 1)) != 0
        or int(validation.get("direct_authority", 1)) != 0
        or int(validation.get("pool_authority", 1)) != 0
    ):
        raise SystemExit(f"invalid TMA base-map validation summary: {path}")
    result = {}
    for key, entry in data.get("map", {}).items():
        parts = key.split(":")
        if len(parts) != 3:
            raise SystemExit(f"invalid kernel-aware TMA map key: {key}")
        kernel_id, uid, pc = int(parts[0]), int(parts[1]), int(parts[2], 0)
        identity = (kernel_id, uid, pc)
        if (
            int(entry.get("trace_kernel_id", -1)) != kernel_id
            or int(entry.get("unique_function_id", -1)) != uid
            or int(entry.get("pc_hex", "-1"), 0) != pc
        ):
            raise SystemExit(
                f"TMA base-map key/value identity mismatch: {key}"
            )
        valid_source = (
            entry.get("source") == "operand_raw_pointer"
            if entry.get("operand_addressed")
            else entry.get("source") in (
                "utmacctl_chain", "utmacctl_prefetch_edge"
            )
        )
        if not valid_source:
            raise SystemExit(
                f"non-authoritative TMA base-map source for {key}: "
                f"{entry.get('source')}"
            )
        result[identity] = entry
    if not result:
        raise SystemExit(f"empty TMA base map: {path}")
    return data, result


def command_bytes(entry):
    if entry.get("operand_addressed"):
        return int(entry.get("covered_bytes", 0))
    total = int(entry.get("element_size", 0))
    for dim in entry.get("box_dim", []):
        dim = int(dim)
        if dim:
            total *= dim
    return total


def load_arrive_sites(extra):
    path = extra / "sync_operand_resolver.json"
    if not path.exists():
        raise SystemExit(f"missing sync operand resolver: {path}")
    data = json.loads(path.read_text())
    return {
        (int(entry["unique_function_id"]), int(entry["pc"]))
        for entry in data.get("resolver", [])
        if entry.get("sync_kind") == "ARRIVE_EXPECT_TX"
    }


def load_operand_map(extra):
    path = extra / "tma_operand_resolver.json"
    data = json.loads(path.read_text())
    if data.get("version") != 2 or not data.get("validation_complete"):
        raise SystemExit(
            f"TMA operand map is not validated schema v2: {path}"
        )
    if data.get("key_fields") != [
            "trace_kernel_id", "unique_function_id", "pc_hex"]:
        raise SystemExit(f"invalid TMA operand-map key schema: {path}")
    result = {}
    for entry in data.get("resolver", []):
        key = (
            int(entry["trace_kernel_id"]),
            int(entry["unique_function_id"]),
            int(entry["pc_hex"], 0),
        )
        if key in result:
            raise SystemExit(f"duplicate kernel-aware operand-map key: {key}")
        result[key] = entry
    validation = data.get("validation", {})
    if (
        int(validation.get("runtime_sites", -1)) != len(result)
        or int(validation.get("resolver_records", -1)) != len(result)
        or int(validation.get("duplicate_keys", -1)) != 0
    ):
        raise SystemExit(f"invalid TMA operand-map validation summary: {path}")
    completion_site_count = sum(
        str(entry.get("opcode", "")).startswith(("UTMALDG", "UBLKCP"))
        for entry in result.values()
    )
    source = data.get("source", {})
    if (
        completion_site_count == 0
        or int(source.get("hidden_completion_sites", -1))
        != completion_site_count
        or int(source.get("hidden_completion_events_ignored", 0)) <= 0
    ):
        raise SystemExit(
            f"invalid hidden-completion callback coverage: {path}"
        )
    return result


def validate_metadata_consistency(base_map, operand_map):
    errors = []
    checked = 0
    for key, base_entry in base_map.items():
        operand_entry = operand_map.get(key)
        if operand_entry is None:
            errors.append(f"missing operand-map record for key={key}")
            continue
        checked += 1
        if base_entry.get("operand_addressed"):
            samples = (
                operand_entry.get("runtime_observed_values", {})
                .get("operand_3", {})
                .get("decoded_byte_samples", [])
            )
            if samples != [int(base_entry.get("covered_bytes", 0))]:
                errors.append(
                    f"operand/base-map byte mismatch key={key} "
                    f"operand={samples} base={base_entry.get('covered_bytes')}"
                )
            continue
        expected_config = (
            f"tm_r{base_entry.get('tensor_rank')}"
            f"_dt{base_entry.get('tensor_data_type')}_box_"
            + "x".join(str(v) for v in base_entry.get("box_dim", []))
        )
        configs = operand_entry.get("descriptor_ref", {}).get(
            "config_ids", []
        )
        if configs != [expected_config]:
            errors.append(
                f"descriptor config mismatch key={key} "
                f"operand={configs} base={expected_config}"
            )
    return checked, errors


def finalize_contract(contract, patterns, errors, location):
    if contract is None:
        return
    pattern = (
        contract["sync_pc"],
        contract["expected"],
        contract["assigned"],
        tuple(contract["commands"]),
    )
    patterns[pattern] += 1
    if contract["assigned"] != contract["expected"]:
        errors.append(
            f"{location}: incomplete contract sync=0x{contract['sync_pc']:x} "
            f"barrier=0x{contract['barrier']:x} "
            f"expected={contract['expected']} assigned={contract['assigned']}"
        )


def validate_kernel(
        kernel_id, trace_dir, base_map, arrive_sites, completion_sites,
        parse_pb):
    pattern_counts = Counter()
    errors = []
    totals = Counter()
    pb_paths = sorted(glob.glob(str(
        trace_dir / "threadblocks" / "device_*" / "stream_*" /
        f"kernel_{kernel_id}" / "*.pb"
    )))
    if not pb_paths:
        errors.append(f"kernel={kernel_id}: no threadblock protobuf files")
    for pb_path in pb_paths:
        with open(pb_path, "rb") as fh:
            warps = parse_pb(fh.read())
        for warp_id, instructions in warps:
            pending = None
            for instruction in instructions:
                site = (
                    kernel_id,
                    int(instruction["function_unique_id"]),
                    int(instruction["pc"]),
                )
                effective_mask = (
                    int(instruction["active_mask"])
                    & int(instruction["predicate_mask"])
                )
                if effective_mask == 0:
                    continue
                if site[1:] in arrive_sites and instruction["semantic_raw"]:
                    finalize_contract(
                        pending, pattern_counts, errors,
                        f"kernel={kernel_id} file={Path(pb_path).name} "
                        f"warp={warp_id}",
                    )
                    pending = {
                        "sync_pc": site[2],
                        "barrier": int(instruction["barrier_addr"]),
                        "expected": int(instruction["semantic_raw"]),
                        "assigned": 0,
                        "commands": [],
                    }
                    totals["contracts"] += 1
                    totals["expected_bytes"] += pending["expected"]
                    continue

                entry = base_map.get(site)
                completion_bearing = site in completion_sites
                if completion_bearing and not instruction["tma_completion_valid"]:
                    errors.append(
                        f"kernel={kernel_id} file={Path(pb_path).name} "
                        f"warp={warp_id}: missing completion payload "
                        f"uid={site[1]} pc=0x{site[2]:x}"
                    )
                    continue
                if not instruction["tma_completion_valid"]:
                    continue

                totals["completion_loads"] += 1
                if not completion_bearing:
                    errors.append(
                        f"kernel={kernel_id} file={Path(pb_path).name} "
                        f"warp={warp_id}: unexpected completion payload "
                        f"uid={site[1]} pc=0x{site[2]:x}"
                    )
                    continue
                if entry is None:
                    errors.append(
                        f"kernel={kernel_id} file={Path(pb_path).name} "
                        f"warp={warp_id}: missing TMA map for "
                        f"uid={site[1]} pc=0x{site[2]:x}"
                    )
                    continue
                size = command_bytes(entry)
                if size <= 0:
                    errors.append(
                        f"kernel={kernel_id} uid={site[1]} pc=0x{site[2]:x}: "
                        f"invalid command bytes={size}"
                    )
                    continue
                if pending is None:
                    errors.append(
                        f"kernel={kernel_id} file={Path(pb_path).name} "
                        f"warp={warp_id}: completion-bearing LOAD before "
                        f"expect-tx uid={site[1]} pc=0x{site[2]:x}"
                    )
                    continue
                target = int(instruction["tma_completion_barrier_addr"])
                if target != pending["barrier"]:
                    errors.append(
                        f"kernel={kernel_id} file={Path(pb_path).name} "
                        f"warp={warp_id}: barrier mismatch "
                        f"sync=0x{pending['barrier']:x} load=0x{target:x}"
                    )
                pending["assigned"] += size
                pending["commands"].append(
                    (site[2], entry.get("opcode", ""), size)
                )
                totals["assigned_bytes"] += size
                if pending["assigned"] == pending["expected"]:
                    finalize_contract(
                        pending, pattern_counts, errors,
                        f"kernel={kernel_id} file={Path(pb_path).name} "
                        f"warp={warp_id}",
                    )
                    pending = None
                elif pending["assigned"] > pending["expected"]:
                    finalize_contract(
                        pending, pattern_counts, errors,
                        f"kernel={kernel_id} file={Path(pb_path).name} "
                        f"warp={warp_id}",
                    )
                    pending = None
            finalize_contract(
                pending, pattern_counts, errors,
                f"kernel={kernel_id} file={Path(pb_path).name} warp={warp_id}",
            )

    patterns = []
    for (sync_pc, expected, assigned, commands), count in sorted(
            pattern_counts.items()):
        patterns.append({
            "sync_pc_hex": f"0x{sync_pc:x}",
            "expected_bytes": expected,
            "assigned_bytes": assigned,
            "count": count,
            "commands": [
                {
                    "pc_hex": f"0x{pc:x}",
                    "opcode": opcode,
                    "bytes": size,
                }
                for pc, opcode, size in commands
            ],
        })
    return {
        "trace_kernel_id": kernel_id,
        "threadblocks": len(pb_paths),
        "contracts": totals["contracts"],
        "completion_loads": totals["completion_loads"],
        "expected_bytes": totals["expected_bytes"],
        "assigned_bytes": totals["assigned_bytes"],
        "patterns": patterns,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    trace_dir = args.traces.resolve()
    extra = trace_dir / "extra_info"
    base_document, base_map = parse_kernel_aware_map(
        extra / "tma_pc_base_map.json"
    )
    operand_map = load_operand_map(extra)
    metadata_checked, metadata_errors = validate_metadata_consistency(
        base_map, operand_map
    )
    arrive_sites = load_arrive_sites(extra)
    completion_sites = {
        key
        for key, entry in operand_map.items()
        if str(entry.get("opcode", "")).startswith(("UTMALDG", "UBLKCP"))
    }
    parse_pb = load_vpb_parser(Path(__file__).resolve().parent)
    kernel_ids = sorted({kernel_id for kernel_id, _, _ in completion_sites})
    kernels = [
        validate_kernel(
            kernel_id, trace_dir, base_map, arrive_sites, completion_sites,
            parse_pb
        )
        for kernel_id in kernel_ids
    ]
    errors = metadata_errors + [
        error for kernel in kernels for error in kernel["errors"]
    ]
    totals = {
        "contracts": sum(k["contracts"] for k in kernels),
        "completion_loads": sum(k["completion_loads"] for k in kernels),
        "expected_bytes": sum(k["expected_bytes"] for k in kernels),
        "assigned_bytes": sum(k["assigned_bytes"] for k in kernels),
    }
    if not completion_sites:
        errors.append("no completion-bearing TMA sites in operand map")
    if not kernels:
        errors.append("no completion-bearing TMA kernels")
    if totals["contracts"] == 0:
        errors.append("no ARRIVE_EXPECT_TX contracts observed")
    if totals["completion_loads"] == 0:
        errors.append("no completion-bearing TMA loads observed")
    if totals["expected_bytes"] != totals["assigned_bytes"]:
        errors.append(
            "global byte mismatch: "
            f"expected={totals['expected_bytes']} "
            f"assigned={totals['assigned_bytes']}"
        )
    payload = {
        "version": 2,
        "source_base_map_version": base_document["version"],
        "source_operand_map_version": 2,
        "validation_complete": not errors,
        "kernels": kernels,
        "summary": {
            "kernels": len(kernels),
            **totals,
            "metadata_records_checked": metadata_checked,
            "errors": len(errors),
        },
    }
    output = args.out or (extra / "tma_completion_contracts.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for kernel in kernels:
        print(
            "[TMA-CONTRACT] "
            f"kernel={kernel['trace_kernel_id']} "
            f"threadblocks={kernel['threadblocks']} "
            f"contracts={kernel['contracts']} "
            f"loads={kernel['completion_loads']} "
            f"expected={kernel['expected_bytes']} "
            f"assigned={kernel['assigned_bytes']} "
            f"errors={len(kernel['errors'])}"
        )
    status = "PASS" if not errors else "FAIL"
    print(
        f"FINAL status={status} "
        f"kernels={len(kernels)} contracts={payload['summary']['contracts']} "
        f"loads={payload['summary']['completion_loads']} "
        f"expected={payload['summary']['expected_bytes']} "
        f"assigned={payload['summary']['assigned_bytes']} "
        f"metadata_checked={metadata_checked} errors={len(errors)}"
    )
    if args.strict and errors:
        raise SystemExit("\n".join(errors[:32]))


if __name__ == "__main__":
    main()
