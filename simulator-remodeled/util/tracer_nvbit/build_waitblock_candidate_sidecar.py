#!/usr/bin/env python3
"""Build fail-closed WaitBlock generation candidates from CTA event order."""

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import os
from pathlib import Path

import build_sync_event_order as order_join
import build_spin_wait_sidecar as spin_sidecar
import validate_tma_completion_contracts as tma_contracts


SCHEMA = "wait_block_candidate_sidecar"
VERSION = 6
EXCH_BASE = 0x200000


def popcount(value):
    return bin(int(value)).count("1")


def load_json(path):
    return json.loads(path.read_text())


def load_sync_metadata(extra):
    data = load_json(extra / "sync_operand_resolver.json")
    return {
        (int(item["unique_function_id"]), int(item["pc"])): item
        for item in data.get("resolver", [])
    }


def load_order_events(path):
    events = []
    seen_ids = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            trace_event_id = int(event.get("trace_event_id", 0))
            if trace_event_id <= 0 or trace_event_id in seen_ids:
                raise ValueError(
                    "{0}:{1}: invalid/duplicate trace_event_id {2}".
                    format(path, line_number, trace_event_id)
                )
            seen_ids.add(trace_event_id)
            if int(event["end_seq"]) <= int(event["begin_seq"]):
                raise ValueError(
                    "{0}:{1}: invalid event interval".format(
                        path, line_number
                    )
                )
            events.append(event)
    return events


def action_metadata(sidecar, kernel_id):
    kernel = next(
        (
            item for item in sidecar.get("kernels", [])
            if int(item.get("trace_kernel_id", -1)) == kernel_id
        ),
        None,
    )
    if kernel is None:
        raise ValueError(
            "kernel {0} absent from WaitBlock sidecar".format(kernel_id)
        )
    regions = {
        (
            int(kernel["unique_function_id"]),
            int(region["primary_wait_pc"]),
        ): region
        for region in kernel.get("spin_regions", [])
    }
    actions = {}
    for block in kernel.get("wait_blocks", []):
        strategy = block["runtime_strategy"]
        function_id = int(
            block.get(
                "unique_function_id",
                kernel["unique_function_id"],
            )
        )
        if strategy == "INJECT_MISSING_WAIT":
            pc = int(block["entry_probe_pc"])
            action = {
                "block_id": block["block_id"],
                "strategy": strategy,
                "unique_function_id": function_id,
                "region_pcs": None,
                "loop_head_pc": None,
                "loopback_pc": None,
            }
        elif strategy == "FOLD_CAPTURED_POLL":
            pc = int(block["poll_probe_pc"])
            region = regions.get((function_id, pc))
            if region is None:
                raise ValueError(
                    "missing spin region for kernel={0} function={1} "
                    "poll=0x{2:x}".format(kernel_id, function_id, pc)
                )
            action = {
                "block_id": block["block_id"],
                "strategy": strategy,
                "unique_function_id": function_id,
                "region_pcs": set(
                    int(value) for value in
                    region["body_pcs"] + [region["loopback_bra_pc"]]
                ),
                "loop_head_pc": int(region["loop_head_pc"]),
                "loopback_pc": int(region["loopback_bra_pc"]),
            }
        else:
            continue
        key = (function_id, pc)
        if key in actions:
            raise ValueError(
                "duplicate actionable WaitBlock key: {0}".format(key)
            )
        actions[key] = action
    return kernel, actions


def iter_fold_episodes(instructions, action):
    function_id = action["unique_function_id"]
    starts = [
        index for index, inst in enumerate(instructions)
        if int(inst["function_unique_id"]) == function_id
        and int(inst["pc"]) == action["loop_head_pc"]
    ]
    if not starts:
        return
    loopback_seen = any(
        int(inst["function_unique_id"]) == function_id
        and int(inst["pc"]) == action["loopback_pc"]
        for inst in instructions
    )
    if not loopback_seen:
        return
    start = starts[0]
    observations = [
        (index, inst)
        for index, inst in enumerate(instructions[start:], start)
        if int(inst["function_unique_id"]) == function_id
        and int(inst["pc"]) in action["region_pcs"]
        and inst["sync_valid"]
    ]
    if observations:
        yield start, observations


def expected_occurrence_count(kernel):
    regions = {
        int(region["primary_wait_pc"]): region
        for region in kernel.get("spin_regions", [])
    }
    total = 0
    for block in kernel.get("wait_blocks", []):
        strategy = block["runtime_strategy"]
        if strategy == "INJECT_MISSING_WAIT":
            total += int(block["active_entry_visits"])
        elif strategy == "FOLD_CAPTURED_POLL":
            total += int(
                regions[int(block["poll_probe_pc"])]
                ["dynamic"]["warp_visits"]
            )
    return total


def event_before(left, right):
    return int(left["end_seq"]) < int(right["begin_seq"])


def event_after(left, right):
    return event_before(right, left)


def candidate_generations(max_generation, parity):
    if max_generation is None or max_generation < 0:
        return None
    return [
        generation for generation in range(max_generation + 1)
        if (generation & 1) != parity
    ]


def load_tma_metadata(extra, has_tma_events):
    if not has_tma_events:
        return {}, set()
    required = (
        extra / "tma_pc_base_map.json",
        extra / "tma_operand_resolver.json",
        extra / "tma_completion_contracts.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "ordered TMA events require validated B0-D metadata: {0}".
            format(", ".join(missing))
        )
    contract_summary = load_json(extra / "tma_completion_contracts.json")
    if not contract_summary.get("validation_complete"):
        raise ValueError("tma_completion_contracts.json is not validated")
    _, base_map = tma_contracts.parse_kernel_aware_map(required[0])
    operand_map = tma_contracts.load_operand_map(extra)
    completion_sites = {
        key for key, entry in operand_map.items()
        if str(entry.get("opcode", "")).startswith(
            ("UTMALDG", "UBLKCP")
        )
    }
    return base_map, completion_sites


def make_event_lookup(events):
    by_cta = defaultdict(list)
    by_source = {}
    launch_by_kernel = {}
    for event in events:
        cta_key = (
            int(event["device_id"]),
            int(event["stream_id"]),
            int(event["trace_kernel_id"]),
            int(event["cta_x"]),
            int(event["cta_y"]),
            int(event["cta_z"]),
        )
        by_cta[cta_key].append(event)
        source_key = cta_key + (
            int(event["warp_id"]),
            int(event["source_warp_instruction_index"]),
        )
        if source_key in by_source:
            raise ValueError(
                "duplicate joined source identity: {0}".format(source_key)
            )
        by_source[source_key] = event
        kernel_key = cta_key[:3]
        launch_uid = int(event["launch_uid"])
        prior = launch_by_kernel.setdefault(kernel_key, launch_uid)
        if prior != launch_uid:
            raise ValueError(
                "kernel identity maps to multiple launch UIDs: {0}".format(
                    kernel_key
                )
            )
    return by_cta, by_source


def order_for_source(by_source, cta_key, warp_id, source_index):
    key = cta_key + (int(warp_id), int(source_index))
    event = by_source.get(key)
    if event is None:
        raise ValueError(
            "missing joined order event for source occurrence {0}".
            format(key)
        )
    return event


def build_tma_generation_providers(
        kernel_id, cta_key, warps, by_source, sync_metadata,
        base_map, completion_sites):
    providers = defaultdict(list)
    for warp_id, instructions in warps:
        pending = None
        for source_index, inst in enumerate(instructions):
            function_id = int(inst["function_unique_id"])
            pc = int(inst["pc"])
            site = (function_id, pc)
            kernel_site = (kernel_id, function_id, pc)
            effective = (
                int(inst["active_mask"]) &
                int(inst["predicate_mask"])
            )
            sync = sync_metadata.get(site)
            if (
                effective and sync is not None
                and sync.get("sync_kind") == "ARRIVE_EXPECT_TX"
                and inst["sync_valid"]
                and int(inst["semantic_raw"] or 0) > 0
            ):
                if pending is not None:
                    raise ValueError(
                        "new TMA contract before previous completion: "
                        "kernel={0} cta={1} warp={2} index={3}".
                        format(kernel_id, cta_key[3:], warp_id,
                               source_index)
                    )
                pending = {
                    "barrier_addr": int(inst["barrier_addr"]),
                    "expected_bytes": int(inst["semantic_raw"]),
                    "assigned_bytes": 0,
                    "open_event": order_for_source(
                        by_source, cta_key, warp_id, source_index
                    ),
                }
                continue

            if kernel_site not in completion_sites:
                continue
            if effective == 0:
                continue
            if not inst["tma_completion_valid"]:
                raise ValueError(
                    "completion-bearing TMA occurrence lacks barrier "
                    "identity: kernel={0} cta={1} warp={2} index={3}".
                    format(kernel_id, cta_key[3:], warp_id,
                           source_index)
                )
            if pending is None:
                raise ValueError(
                    "completion-bearing TMA occurrence has no open "
                    "contract: kernel={0} cta={1} warp={2} index={3}".
                    format(kernel_id, cta_key[3:], warp_id,
                           source_index)
                )
            target = int(inst["tma_completion_barrier_addr"])
            if target != pending["barrier_addr"]:
                raise ValueError(
                    "TMA contract barrier mismatch: open=0x{0:x} "
                    "completion=0x{1:x}".format(
                        pending["barrier_addr"], target
                    )
                )
            metadata = base_map.get(kernel_site)
            if metadata is None:
                raise ValueError(
                    "missing TMA command metadata for {0}".format(
                        kernel_site
                    )
                )
            command_bytes = tma_contracts.command_bytes(metadata)
            if command_bytes <= 0:
                raise ValueError(
                    "invalid TMA command size for {0}: {1}".format(
                        kernel_site, command_bytes
                    )
                )
            pending["assigned_bytes"] += command_bytes
            completion_event = order_for_source(
                by_source, cta_key, warp_id, source_index
            )
            if pending["assigned_bytes"] > pending["expected_bytes"]:
                raise ValueError(
                    "TMA contract byte overflow: expected={0} "
                    "assigned={1}".format(
                        pending["expected_bytes"],
                        pending["assigned_bytes"],
                    )
                )
            if pending["assigned_bytes"] == pending["expected_bytes"]:
                pending["completion_event"] = completion_event
                providers[target].append(pending)
                pending = None
        if pending is not None:
            raise ValueError(
                "incomplete TMA contract at warp end: kernel={0} "
                "cta={1} warp={2} expected={3} assigned={4}".
                format(
                    kernel_id, cta_key[3:], warp_id,
                    pending["expected_bytes"],
                    pending["assigned_bytes"],
                )
            )
    return providers


def build_count_barrier_state(events, sync_metadata, tma_barriers):
    expected = {}
    arrivals = defaultdict(list)
    for event in events:
        if not event["sync_valid"]:
            continue
        barrier = event["barrier_addr"]
        if barrier is None:
            continue
        barrier = int(barrier)
        metadata = sync_metadata.get((
            int(event["unique_function_id"]), int(event["pc"])
        ))
        kind = metadata.get("sync_kind") if metadata else None
        raw = int(event["semantic_raw"] or 0)
        effective = (
            int(event["active_mask"]) &
            int(event["predicate_mask"])
        )
        if kind == "EXCH":
            value = (EXCH_BASE - raw) // 2
            if value <= 0:
                continue
            prior = expected.setdefault(barrier, value)
            if prior != value:
                raise ValueError(
                    "inconsistent EXCH expected count for barrier "
                    "0x{0:x}: {1} vs {2}".format(
                        barrier, prior, value
                    )
                )
        elif kind == "ARRIVE_EXPECT_TX" and raw == 0 and effective:
            arrivals[barrier].append({
                "event": event,
                "units": popcount(effective),
            })
        elif kind == "ARRIVE" and effective:
            arrivals[barrier].append({
                "event": event,
                "units": popcount(effective),
            })
        elif kind == "ARRIVE_COUNTED" and effective:
            arrivals[barrier].append({
                "event": event,
                "units": raw,
            })

    state = {}
    for barrier in set(expected) | set(arrivals):
        if barrier in tma_barriers:
            continue
        count = expected.get(barrier)
        state[barrier] = {
            "expected_arrive": count,
            "arrivals": arrivals[barrier],
        }
    return state


def resolve_candidates(
        barrier, parity, wait_event, tma_providers, count_state):
    if barrier in tma_providers:
        all_providers = [
            contract["completion_event"]
            for contract in tma_providers[barrier]
        ]
        possible = [
            event for event in all_providers
            if not event_after(event, wait_event)
        ]
        definitely_before = [
            event for event in all_providers
            if event_before(event, wait_event)
        ]
        max_generation = len(possible)
        proof = {
            "barrier_kind": "TMA",
            "total_contracts": len(all_providers),
            "definitely_completed_before_wait":
                len(definitely_before),
            "possibly_completed_before_wait": len(possible),
            "overlapping_completion_intervals":
                len(possible) - len(definitely_before),
        }
    else:
        state = count_state.get(barrier)
        if state is None or not state["expected_arrive"]:
            raise ValueError(
                "unbounded count barrier candidate set for 0x{0:x}".
                format(barrier)
            )
        expected = int(state["expected_arrive"])
        possible_units = sum(
            arrival["units"] for arrival in state["arrivals"]
            if not event_after(arrival["event"], wait_event)
        )
        definite_units = sum(
            arrival["units"] for arrival in state["arrivals"]
            if event_before(arrival["event"], wait_event)
        )
        max_generation = possible_units // expected
        proof = {
            "barrier_kind": "COUNT",
            "expected_arrive": expected,
            "total_arrival_units": sum(
                arrival["units"] for arrival in state["arrivals"]
            ),
            "definite_arrival_units_before_wait": definite_units,
            "possible_arrival_units_before_wait": possible_units,
            "overlapping_arrival_units":
                possible_units - definite_units,
        }

    candidates = candidate_generations(max_generation, parity)
    if not candidates:
        raise ValueError(
            "empty candidate set for barrier=0x{0:x} parity={1} "
            "max_generation={2}".format(
                barrier, parity, max_generation
            )
        )
    proof["max_possible_generation"] = max_generation
    return candidates, proof


def make_ticket(
        kernel_id, cta_key, warp_id, action, canonical_index,
        ready_index, wait_inst, wait_event, tma_providers,
        count_state):
    barrier = wait_inst["barrier_addr"]
    if not wait_inst["sync_valid"] or barrier is None:
        raise ValueError(
            "WaitBlock occurrence lacks runtime barrier payload: "
            "kernel={0} cta={1} warp={2} index={3}".format(
                kernel_id, cta_key[3:], warp_id, ready_index
            )
        )
    wait_state = int(wait_inst["semantic_raw"] or 0)
    parity = (wait_state >> 31) & 1
    candidates, proof_detail = resolve_candidates(
        int(barrier), parity, wait_event, tma_providers, count_state
    )
    return {
        "trace_kernel_id": kernel_id,
        "device_id": cta_key[0],
        "stream_id": cta_key[1],
        "launch_uid": int(wait_event["launch_uid"]),
        "global_cta_id": int(wait_event["linear_cta_id"]),
        "cta": {
            "x": cta_key[3],
            "y": cta_key[4],
            "z": cta_key[5],
        },
        "warp_id": int(warp_id),
        "unique_function_id": int(
            wait_inst["function_unique_id"]
        ),
        "block_id": action["block_id"],
        "runtime_strategy": action["strategy"],
        "trace_event_id": int(wait_event["trace_event_id"]),
        "source_warp_instruction_index": int(ready_index),
        "canonical_occurrence_index": int(canonical_index),
        "pc": int(wait_inst["pc"]),
        "barrier_addr": int(barrier),
        "wait_parity": parity,
        "captured_result": "READY",
        "candidate_generations": candidates,
        "min_candidate": min(candidates),
        "max_candidate": max(candidates),
        "max_reachable_without_wait": True,
        "proof": "CTA_EVENT_ORDER_CANDIDATE_SET_V2",
        "proof_detail": proof_detail,
    }


def analyze_threadblock(
        path, sidecar, events, by_source, sync_metadata,
        base_map, completion_sites, parse_pb):
    cta_key = order_join.parse_threadblock_identity(path)
    kernel_id = cta_key[2]
    kernel, actions = action_metadata(sidecar, kernel_id)
    warps = parse_pb(path.read_bytes())
    tma_providers = build_tma_generation_providers(
        kernel_id, cta_key, warps, by_source, sync_metadata,
        base_map, completion_sites,
    )
    count_state = build_count_barrier_state(
        events, sync_metadata, set(tma_providers)
    )
    tickets = []

    for warp_id, instructions in warps:
        fold_starts = defaultdict(list)
        for action in actions.values():
            if action["strategy"] != "FOLD_CAPTURED_POLL":
                continue
            for start, observations in iter_fold_episodes(
                    instructions, action):
                fold_starts[start].append((action, observations))

        for index, inst in enumerate(instructions):
            key = (
                int(inst["function_unique_id"]),
                int(inst["pc"]),
            )
            action = actions.get(key)
            effective = (
                int(inst["active_mask"]) &
                int(inst["predicate_mask"])
            )
            if (
                action is not None
                and action["strategy"] == "INJECT_MISSING_WAIT"
                and effective
            ):
                wait_event = order_for_source(
                    by_source, cta_key, warp_id, index
                )
                tickets.append(make_ticket(
                    kernel_id, cta_key, warp_id, action, index,
                    index, inst, wait_event, tma_providers,
                    count_state,
                ))

            for fold_action, observations in fold_starts.get(index, []):
                ready_index, ready_inst = observations[-1]
                wait_event = order_for_source(
                    by_source, cta_key, warp_id, ready_index
                )
                tickets.append(make_ticket(
                    kernel_id, cta_key, warp_id, fold_action, index,
                    ready_index, ready_inst, wait_event,
                    tma_providers, count_state,
                ))
    return kernel, tickets


def summarize(tickets):
    by_kernel = Counter()
    by_strategy = Counter()
    by_kind = Counter()
    multi_candidate = 0
    for ticket in tickets:
        by_kernel[str(ticket["trace_kernel_id"])] += 1
        by_strategy[ticket["runtime_strategy"]] += 1
        by_kind[ticket["proof_detail"]["barrier_kind"]] += 1
        multi_candidate += len(ticket["candidate_generations"]) > 1
    return {
        "tickets": len(tickets),
        "by_kernel": dict(sorted(by_kernel.items())),
        "by_strategy": dict(sorted(by_strategy.items())),
        "by_barrier_kind": dict(sorted(by_kind.items())),
        "multi_candidate_tickets": multi_candidate,
        "empty_candidate_tickets": 0,
        "unbounded_candidate_tickets": 0,
        "unsafe_max_tickets": 0,
    }


def atomic_write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--kernels")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    traces = args.traces.resolve()
    extra = traces / "extra_info"
    sidecar_path = extra / "spin_wait_sidecar.json"
    order_path = extra / "sync_event_order.jsonl"
    order_summary_path = extra / "sync_event_order_summary.json"
    output = args.out or (
        extra / "waitblock_candidate_sidecar.json"
    )
    if output.resolve() in tuple(path.resolve() for path in (
            sidecar_path, order_path, order_summary_path)):
        raise SystemExit(
            "[wait-candidates] output must not alias an input path"
        )
    if output.exists():
        output.unlink()
    required = (sidecar_path, order_path, order_summary_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "[wait-candidates] missing inputs: {0}".format(
                ", ".join(missing)
            )
        )

    sidecar = load_json(sidecar_path)
    if (
        int(sidecar.get("version", -1)) != 5
        or not sidecar.get("validation_complete")
    ):
        raise SystemExit(
            "[wait-candidates] WaitBlock v5 sidecar is not complete"
        )
    order_summary = load_json(order_summary_path)
    if (
        order_summary.get("status") != "PASS"
        or not order_summary.get("join_bijective")
    ):
        raise SystemExit(
            "[wait-candidates] sync order join has not passed"
        )
    joined_digest = order_join.file_sha256(order_path)
    if joined_digest != order_summary.get("joined_sha256"):
        raise SystemExit(
            "[wait-candidates] joined order artifact digest mismatch"
        )

    selected = (
        {int(value) for value in args.kernels.split(",")}
        if args.kernels else {
            int(kernel["trace_kernel_id"])
            for kernel in sidecar.get("kernels", [])
            if kernel.get("wait_blocks")
        }
    )
    known = {
        int(kernel["trace_kernel_id"])
        for kernel in sidecar.get("kernels", [])
    }
    unknown = sorted(selected - known)
    if unknown:
        raise SystemExit(
            "[wait-candidates] kernels absent from sidecar: {0}".
            format(unknown)
        )

    events = load_order_events(order_path)
    by_cta, by_source = make_event_lookup(events)
    sync_metadata = load_sync_metadata(extra)
    has_tma_events = any(
        event["event_kind"] == "TMA_COMPLETION"
        and int(event["trace_kernel_id"]) in selected
        for event in events
    )
    base_map, completion_sites = load_tma_metadata(
        extra, has_tma_events
    )
    parse_pb = spin_sidecar.vpb_parse_full

    tickets = []
    expected_by_kernel = {}
    actual_by_kernel = Counter()
    analyzed_ctas = Counter()
    kernel_metadata = {}
    for path in order_join.iter_threadblock_paths(traces):
        cta_key = order_join.parse_threadblock_identity(path)
        kernel_id = cta_key[2]
        if kernel_id not in selected:
            continue
        cta_events = by_cta.get(cta_key)
        if cta_events is None:
            raise ValueError(
                "selected threadblock has no order events: {0}".format(
                    cta_key
                )
            )
        kernel, cta_tickets = analyze_threadblock(
            path, sidecar, cta_events, by_source, sync_metadata,
            base_map, completion_sites, parse_pb,
        )
        kernel_metadata[kernel_id] = kernel
        tickets.extend(cta_tickets)
        actual_by_kernel[str(kernel_id)] += len(cta_tickets)
        analyzed_ctas[str(kernel_id)] += 1

    for kernel_id in sorted(selected):
        kernel = kernel_metadata.get(kernel_id)
        if kernel is None:
            raise ValueError(
                "selected kernel has no threadblock protobuf: {0}".
                format(kernel_id)
            )
        expected = expected_occurrence_count(kernel)
        actual = actual_by_kernel[str(kernel_id)]
        expected_by_kernel[str(kernel_id)] = expected
        if actual != expected:
            raise ValueError(
                "WaitBlock ticket coverage mismatch for kernel {0}: "
                "expected={1} actual={2}".format(
                    kernel_id, expected, actual
                )
            )

    identity_keys = [
        (
            ticket["trace_kernel_id"], ticket["device_id"],
            ticket["stream_id"], ticket["global_cta_id"],
            ticket["warp_id"], ticket["block_id"],
            ticket["canonical_occurrence_index"],
        )
        for ticket in tickets
    ]
    if len(identity_keys) != len(set(identity_keys)):
        raise ValueError("duplicate candidate ticket identity")

    tickets.sort(key=lambda ticket: (
        ticket["device_id"], ticket["stream_id"],
        ticket["trace_kernel_id"], ticket["global_cta_id"],
        ticket["warp_id"], ticket["canonical_occurrence_index"],
        ticket["block_id"],
    ))
    payload = {
        "version": VERSION,
        "schema": SCHEMA,
        "validation_complete": True,
        "uses_simulation_logs": False,
        "source_traces_dir": str(traces),
        "source_waitblock_sidecar": str(sidecar_path),
        "source_sync_event_order": str(order_path),
        "source_waitblock_sidecar_sha256":
            order_join.file_sha256(sidecar_path),
        "source_sync_event_order_sha256": joined_digest,
        "candidate_rule": (
            "All parity-compatible generations up to the maximum whose "
            "required ordered provider is not definitely after the wait. "
            "Overlapping intervals preserve both legal orders."
        ),
        "expected_tickets_by_kernel": expected_by_kernel,
        "validated_ctas_by_kernel": dict(sorted(analyzed_ctas.items())),
        "summary": summarize(tickets),
        "tickets": tickets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    print(
        "[wait-candidates] FINAL status=PASS tickets={0} kernels={1} "
        "multi={2} output={3}".format(
            len(tickets), len(selected),
            payload["summary"]["multi_candidate_tickets"], output,
        )
    )


if __name__ == "__main__":
    main()
