#!/usr/bin/env python3
"""Validate and join CTA-local sync-order records to protobuf occurrences."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


SCHEMA = "sync_event_order_v1"
RAW_RECORD_KIND = "SYNC_EVENT_ORDER"
RAW_RECORD_VERSION = 1
THREADBLOCK_RE = re.compile(
    r"^d_(\d+)_s_(\d+)_k_(\d+)_"
    r"(\d+),(\d+),(\d+)\.pb$"
)


def load_protobuf_parser():
    module_path = Path(__file__).with_name("build_spin_wait_sidecar.py")
    spec = importlib.util.spec_from_file_location(
        "build_spin_wait_sidecar_for_order", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.vpb_parse_full


def event_kind_for_opcode(opcode):
    if opcode.startswith("SYNCS.EXCH"):
        return "SYNCS_EXCH"
    if opcode.startswith("SYNCS.ARRIVE"):
        return "SYNCS_ARRIVE"
    if opcode.startswith("SYNCS.PHASECHK"):
        return "SYNCS_PHASECHK"
    if opcode.startswith("BAR.SYNC"):
        return "BAR_SYNC"
    if opcode.startswith("BAR.ARV"):
        return "BAR_ARRIVE"
    if opcode.startswith("WARPSYNC"):
        return "WARP_SYNC"
    if opcode.startswith("WARPGROUP"):
        return "WARPGROUP_SYNC"
    if opcode.startswith("UTMALDG") or opcode.startswith("UBLKCP"):
        return "TMA_COMPLETION"
    return None


def load_static_events(extra_info_dir):
    path = extra_info_dir / "enhanced_execution_info.json"
    data = json.loads(path.read_text())
    result = {}
    for kernel in data.get("kernels", []):
        function_id = int(kernel["unique_function_id"])
        for inst in kernel.get("instructions", []):
            pc = int(inst["pc_num_dec"])
            opcode = inst.get("op_code", "")
            kind = event_kind_for_opcode(opcode)
            if kind is not None:
                result[(function_id, pc)] = {
                    "event_kind": kind,
                    "opcode": opcode,
                }
    return result


def parse_threadblock_identity(path):
    match = THREADBLOCK_RE.match(path.name)
    if match is None:
        raise ValueError(
            "unsupported threadblock filename: {0}".format(path)
        )
    return tuple(int(value) for value in match.groups())


def iter_threadblock_paths(traces_dir):
    base = traces_dir / "threadblocks"
    if not base.is_dir():
        raise ValueError("missing threadblocks directory: {0}".format(base))
    paths = sorted(base.glob("device_*/stream_*/kernel_*/*.pb"))
    if not paths:
        raise ValueError("no protobuf threadblocks under {0}".format(base))
    return paths


def load_raw_records(path):
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "{0}:{1}: invalid JSON: {2}".format(
                        path, line_number, error
                    )
                )
            if record.get("record_kind") != RAW_RECORD_KIND:
                raise ValueError(
                    "{0}:{1}: unexpected record_kind {2!r}".format(
                        path, line_number, record.get("record_kind")
                    )
                )
            if int(record.get("record_version", -1)) != RAW_RECORD_VERSION:
                raise ValueError(
                    "{0}:{1}: unsupported record_version {2!r}".format(
                        path, line_number, record.get("record_version")
                    )
                )
            record["_raw_line"] = line_number
            records.append(record)
    return records


def _int_field(record, field):
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "raw line {0}: {1} must be an integer".format(
                record["_raw_line"], field
            )
        )
    return value


def validate_raw_records(records, static_events):
    endpoint_owner = {}
    launch_identity = {}
    linear_to_xyz = {}
    xyz_to_linear = {}
    by_warp = defaultdict(list)
    kinds = Counter()

    required_fields = (
        "launch_uid", "device_id", "stream_id", "trace_kernel_id",
        "cta_x", "cta_y", "cta_z", "linear_cta_id", "warp_id",
        "sm_id", "unique_function_id", "pc", "active_mask",
        "predicate_mask", "begin_seq", "end_seq",
    )
    for record in records:
        for field in required_fields:
            _int_field(record, field)
        launch_uid = record["launch_uid"]
        if launch_uid <= 0:
            raise ValueError(
                "raw line {0}: launch_uid must be positive".format(
                    record["_raw_line"]
                )
            )
        launch_key = (
            record["device_id"],
            record["stream_id"],
            record["trace_kernel_id"],
        )
        prior_launch = launch_identity.setdefault(launch_uid, launch_key)
        if prior_launch != launch_key:
            raise ValueError(
                "launch_uid {0} maps to multiple launches".format(
                    launch_uid
                )
            )

        static_key = (
            record["unique_function_id"],
            record["pc"],
        )
        static = static_events.get(static_key)
        if static is None:
            raise ValueError(
                "raw line {0}: event site absent from static metadata: "
                "function={1} pc=0x{2:x}".format(
                    record["_raw_line"], static_key[0], static_key[1]
                )
            )
        if record.get("event_kind") != static["event_kind"]:
            raise ValueError(
                "raw line {0}: event kind mismatch: raw={1} static={2} "
                "opcode={3}".format(
                    record["_raw_line"], record.get("event_kind"),
                    static["event_kind"], static["opcode"]
                )
            )
        begin = record["begin_seq"]
        end = record["end_seq"]
        if begin <= 0 or end <= begin:
            raise ValueError(
                "raw line {0}: invalid interval [{1},{2}]".format(
                    record["_raw_line"], begin, end
                )
            )
        if record["predicate_mask"] & ~record["active_mask"]:
            raise ValueError(
                "raw line {0}: predicate mask is not a subset of "
                "active mask".format(record["_raw_line"])
            )
        cta_key = (
            launch_uid,
            record["linear_cta_id"],
        )
        xyz = (
            record["cta_x"],
            record["cta_y"],
            record["cta_z"],
        )
        prior_xyz = linear_to_xyz.setdefault(cta_key, xyz)
        if prior_xyz != xyz:
            raise ValueError(
                "launch {0} linear CTA {1} maps to multiple coordinates".
                format(launch_uid, record["linear_cta_id"])
            )
        xyz_key = (launch_uid,) + xyz
        prior_linear = xyz_to_linear.setdefault(
            xyz_key, record["linear_cta_id"]
        )
        if prior_linear != record["linear_cta_id"]:
            raise ValueError(
                "launch {0} CTA {1} maps to multiple linear IDs".format(
                    launch_uid, xyz
                )
            )
        for endpoint_name, endpoint in (("begin", begin), ("end", end)):
            endpoint_key = cta_key + (endpoint,)
            if endpoint_key in endpoint_owner:
                raise ValueError(
                    "duplicate CTA-local endpoint launch={0} cta={1} "
                    "seq={2}: lines {3} and {4}".format(
                        launch_uid, record["linear_cta_id"], endpoint,
                        endpoint_owner[endpoint_key],
                        record["_raw_line"],
                    )
                )
            endpoint_owner[endpoint_key] = record["_raw_line"]
        warp_key = (
            launch_uid,
            record["linear_cta_id"],
            record["warp_id"],
        )
        by_warp[warp_key].append(record)
        kinds[record["event_kind"]] += 1

    for warp_key, warp_records in by_warp.items():
        previous = None
        for record in sorted(
                warp_records, key=lambda item: item["begin_seq"]):
            if previous is not None and (
                    previous["end_seq"] >= record["begin_seq"]):
                raise ValueError(
                    "overlapping same-warp intervals for {0}: "
                    "lines {1} and {2}".format(
                        warp_key, previous["_raw_line"],
                        record["_raw_line"],
                    )
                )
            previous = record
    return {
        "records": len(records),
        "launches": len(launch_identity),
        "ctas": len(linear_to_xyz),
        "warps": len(by_warp),
        "by_event_kind": dict(sorted(kinds.items())),
    }


def protobuf_occurrences(traces_dir, static_events, parse_pb):
    occurrences = []
    all_instruction_count = 0
    threadblock_count = 0
    for path in iter_threadblock_paths(traces_dir):
        identity = parse_threadblock_identity(path)
        device_id, stream_id, kernel_id, cta_x, cta_y, cta_z = identity
        threadblock_count += 1
        for warp_id, instructions in parse_pb(path.read_bytes()):
            for source_index, inst in enumerate(instructions):
                all_instruction_count += 1
                static_key = (
                    int(inst["function_unique_id"]),
                    int(inst["pc"]),
                )
                static = static_events.get(static_key)
                if static is None:
                    continue
                occurrences.append({
                    "device_id": device_id,
                    "stream_id": stream_id,
                    "trace_kernel_id": kernel_id,
                    "cta_x": cta_x,
                    "cta_y": cta_y,
                    "cta_z": cta_z,
                    "warp_id": int(warp_id),
                    "unique_function_id": static_key[0],
                    "pc": static_key[1],
                    "event_kind": static["event_kind"],
                    "opcode": static["opcode"],
                    "source_warp_instruction_index": source_index,
                    "active_mask": int(inst["active_mask"]),
                    "predicate_mask": int(inst["predicate_mask"]),
                    "sync_valid": bool(inst["sync_valid"]),
                    "barrier_addr": inst["barrier_addr"],
                    "semantic_raw": inst["semantic_raw"],
                    "tma_completion_valid":
                        bool(inst["tma_completion_valid"]),
                    "tma_completion_barrier_addr":
                        inst["tma_completion_barrier_addr"],
                    "tma_completion_barrier_ureg_id":
                        inst["tma_completion_barrier_ureg_id"],
                })
    return occurrences, {
        "threadblocks": threadblock_count,
        "instructions": all_instruction_count,
        "semantic_events": len(occurrences),
    }


def occurrence_key(record):
    return (
        int(record["device_id"]),
        int(record["stream_id"]),
        int(record["trace_kernel_id"]),
        int(record["cta_x"]),
        int(record["cta_y"]),
        int(record["cta_z"]),
        int(record["warp_id"]),
        int(record["unique_function_id"]),
        int(record["pc"]),
    )


def join_records(raw_records, pb_occurrences):
    raw_by_key = defaultdict(list)
    pb_by_key = defaultdict(list)
    for record in raw_records:
        raw_by_key[occurrence_key(record)].append(record)
    for occurrence in pb_occurrences:
        pb_by_key[occurrence_key(occurrence)].append(occurrence)

    all_keys = set(raw_by_key) | set(pb_by_key)
    mismatches = []
    for key in sorted(all_keys):
        raw_count = len(raw_by_key.get(key, ()))
        pb_count = len(pb_by_key.get(key, ()))
        if raw_count != pb_count:
            mismatches.append({
                "identity": list(key),
                "raw_count": raw_count,
                "protobuf_count": pb_count,
            })
    if mismatches:
        raise ValueError(
            "raw/protobuf occurrence cardinality mismatch: {0}".format(
                json.dumps(mismatches[:16], sort_keys=True)
            )
        )

    joined = []
    for key in sorted(all_keys):
        raw_group = sorted(
            raw_by_key[key], key=lambda item: item["begin_seq"]
        )
        pb_group = sorted(
            pb_by_key[key],
            key=lambda item: item["source_warp_instruction_index"],
        )
        for ordinal, (raw, occurrence) in enumerate(
                zip(raw_group, pb_group)):
            if raw["event_kind"] != occurrence["event_kind"]:
                raise ValueError(
                    "event kind mismatch for {0} ordinal {1}".format(
                        key, ordinal
                    )
                )
            for field in ("active_mask", "predicate_mask"):
                if int(raw[field]) != int(occurrence[field]):
                    raise ValueError(
                        "{0} mismatch for {1} ordinal {2}: raw=0x{3:x} "
                        "protobuf=0x{4:x}".format(
                            field, key, ordinal, int(raw[field]),
                            int(occurrence[field]),
                        )
                    )
            item = {
                key_name: value
                for key_name, value in raw.items()
                if key_name != "_raw_line"
            }
            item.update(occurrence)
            item["per_pc_occurrence_ordinal"] = ordinal
            joined.append(item)

    joined.sort(key=lambda item: (
        item["device_id"], item["stream_id"],
        item["trace_kernel_id"], item["cta_z"], item["cta_y"],
        item["cta_x"], item["warp_id"],
        item["source_warp_instruction_index"],
    ))
    for trace_event_id, item in enumerate(joined, 1):
        item["trace_event_id"] = trace_event_id
        item["canonical_occurrence_index"] = \
            item["source_warp_instruction_index"]
    return joined


def protobuf_tree_fingerprint(traces_dir):
    digest = hashlib.sha256()
    paths = iter_threadblock_paths(traces_dir)
    for path in paths:
        relative = path.relative_to(traces_dir).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return {
        "algorithm": "sha256",
        "threadblocks": len(paths),
        "digest": digest.hexdigest(),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compare_baseline(ordered, baseline):
    ordered_fingerprint = protobuf_tree_fingerprint(ordered)
    if baseline is None:
        return {
            "status": "NOT_REQUESTED",
            "ordered": ordered_fingerprint,
        }
    baseline_fingerprint = protobuf_tree_fingerprint(baseline)
    status = (
        "PASS"
        if ordered_fingerprint == baseline_fingerprint
        else "FAIL"
    )
    return {
        "status": status,
        "ordered": ordered_fingerprint,
        "baseline": baseline_fingerprint,
        "baseline_traces": str(baseline.resolve()),
    }


def atomic_write_jsonl(path, records):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--baseline-traces", type=Path)
    args = parser.parse_args()

    traces_dir = args.traces.resolve()
    extra_info_dir = traces_dir / "extra_info"
    raw_path = args.raw or (
        extra_info_dir / "sync_event_order.raw.jsonl"
    )
    output_path = args.out or (
        extra_info_dir / "sync_event_order.jsonl"
    )
    summary_path = args.summary or (
        extra_info_dir / "sync_event_order_summary.json"
    )
    if raw_path.resolve() in (
            output_path.resolve(), summary_path.resolve()):
        raise SystemExit(
            "[sync-order] raw input must not alias an output path"
        )
    for stale_output in (output_path, summary_path):
        if stale_output.exists():
            stale_output.unlink()
    if not raw_path.is_file():
        raise SystemExit(
            "[sync-order] missing raw order artifact: {0}".format(
                raw_path
            )
        )

    static_events = load_static_events(extra_info_dir)
    raw_records = load_raw_records(raw_path)
    raw_summary = validate_raw_records(raw_records, static_events)
    parse_pb = load_protobuf_parser()
    pb_occurrences, pb_summary = protobuf_occurrences(
        traces_dir, static_events, parse_pb
    )
    joined = join_records(raw_records, pb_occurrences)
    baseline = compare_baseline(
        traces_dir,
        args.baseline_traces.resolve()
        if args.baseline_traces is not None else None,
    )
    summary = {
        "schema": SCHEMA,
        "status": (
            "PASS"
            if baseline["status"] in ("PASS", "NOT_REQUESTED")
            else "FAIL"
        ),
        "raw_artifact": str(raw_path.resolve()),
        "joined_artifact": str(output_path.resolve()),
        "raw_sha256": file_sha256(raw_path),
        "static_event_sites": len(static_events),
        "raw": raw_summary,
        "protobuf": pb_summary,
        "joined_records": len(joined),
        "join_bijective": (
            len(joined) == len(raw_records) == len(pb_occurrences)
        ),
        "baseline_comparison": baseline,
    }
    if not summary["join_bijective"]:
        raise RuntimeError("internal error: non-bijective joined output")
    if baseline["status"] == "FAIL":
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(summary_path, summary)
        raise SystemExit(
            "[sync-order] ordered protobuf differs from baseline; "
            "see {0}".format(summary_path)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_path, joined)
    summary["joined_sha256"] = file_sha256(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)
    print(
        "[sync-order] FINAL status={0} raw={1} protobuf={2} joined={3} "
        "baseline={4} output={5}".format(
            summary["status"], len(raw_records), len(pb_occurrences),
            len(joined), baseline["status"], output_path,
        )
    )


if __name__ == "__main__":
    main()
