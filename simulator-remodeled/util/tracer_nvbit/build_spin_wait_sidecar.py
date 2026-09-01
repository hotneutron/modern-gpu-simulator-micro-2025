#!/usr/bin/env python3
"""Build a generic, two-strategy WaitBlock sidecar from compact GPU traces.

Purpose
-------
The trace-driven simulator replays each warp's recorded PC stream and takes every
branch from the captured stream. Post-processing identifies two independent
runtime contracts without modifying the payload:

  * INJECT_MISSING_WAIT: a gate is ready-only in the complete capture. A future
    simulator may wait-hold an all-live modeled miss at the entry. A partial
    modeled miss may use the v5 deferred-suffix contract only after static CFG
    and complete dynamic-trace proofs establish a join-only complementary path.
  * FOLD_CAPTURED_POLL: an existing foldable polling loop. A future simulator
    may use the already-proven poll-site hold and dual-cursor advance to the local
    loop exit.

Discovery is application-independent. It uses static CFG/opcode semantics and
all selected CTA/warp traces; it has no kernel-name, kernel-id, function-id, PC,
CTA, warp, SM, or expected-count filters. K5/K10 values are validation results,
not production rules.

The existing spin_regions records are preserved unchanged. Simulator support for
schema v5 is a separate stage.

Usage
-----
  build_spin_wait_sidecar.py --traces <.../traces> [--kernels 5,10]
                             [--out <.../extra_info/spin_wait_sidecar.json>]
                             [--validate-ctas all|N] [--strict]

Production strategy classification requires --validate-ctas all. A smaller
value is review-only and is rejected by --strict because a gate can be ready-only
in one CTA and captured-poll in another.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

# Reuse the dependency-free varint protobuf parser + funcid peek from trim_spin.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import importlib.util as _ilu
_ts_spec = _ilu.spec_from_file_location(
    "trim_spin_from_ncu", os.path.join(_HERE, "trim_spin_from_ncu.py"))
_ts = _ilu.module_from_spec(_ts_spec)
_ts_spec.loader.exec_module(_ts)

ISIZE = 0x10          # SM90 SASS instructions in this trace format are 16 bytes
MAX_LOOP_SPAN = 0x100  # a spin loop is short; guards against matching big loops


# --- Stage 1.7: mask-aware .pb parser ------------------------------------- #
# The shared _ts._vpb_parse() returns only (pc, sync_valid) per instruction and
# must not change (trim_spin relies on it). The compound wait-block dynamic proof
# (§1.7.4) additionally needs the recorded active_mask / predicate_mask and the
# runtime sync payload (barrier_addr / semantic_raw). We add a PRIVATE parser here
# that reuses _ts._rv / _ts._skip varint primitives and the same threadblock ->
# warps -> instruction wire layout, without touching the trace format or the
# shared helper. instruction.proto fields: pc=1, active_mask=2, predicate_mask=3,
# function_unique_id=4, sync=6 { valid=1, barrier_addr=2, has_semantic_raw=3,
# semantic_raw=4 }, tma_completion=8 { valid=1, barrier_addr=2,
# barrier_ureg_id=3 }, mbarrier_contract=9 { valid=1, event_kind=2,
# barrier_addr=3, epoch_before=4, epoch_after=5, opened_before=6,
# opened_after=7, has_ordinal=8, ordinal=9, capture_flags=10,
# barrier_error_flags=11, capture_launch_uid=12, has_semantic_raw=13,
# semantic_raw=14 }.
_rv = _ts._rv
_skip = _ts._skip


def _inst_full(buf, s, e):
    """Parse one instruction message; return a dict with pc + masks + sync info."""
    i = s
    pc = 0
    active_mask = 0
    predicate_mask = 0
    function_unique_id = None
    sync_valid = False
    barrier_addr = None
    semantic_raw = None
    tma_completion_valid = False
    tma_completion_barrier_addr = None
    tma_completion_barrier_ureg_id = None
    contract_capture_valid = False
    contract_event_kind = None
    contract_barrier_addr = None
    contract_epoch_before = None
    contract_epoch_after = None
    contract_opened_before = None
    contract_opened_after = None
    contract_has_ordinal = False
    contract_ordinal = None
    contract_capture_flags = 0
    contract_barrier_error_flags = 0
    contract_capture_launch_uid = None
    contract_has_semantic_raw = False
    contract_semantic_raw = None
    while i < e:
        tag, i = _rv(buf, i)
        f, w = tag >> 3, tag & 7
        if f == 1 and w == 0:
            pc, i = _rv(buf, i)
        elif f == 2 and w == 0:
            active_mask, i = _rv(buf, i)
        elif f == 3 and w == 0:
            predicate_mask, i = _rv(buf, i)
        elif f == 4 and w == 0:
            function_unique_id, i = _rv(buf, i)
        elif f == 6 and w == 2:                 # sync_runtime_info sub-message
            ln, j = _rv(buf, i)
            end = j + ln
            k = j
            while k < end:
                t2, k = _rv(buf, k)
                f2, w2 = t2 >> 3, t2 & 7
                if f2 == 1 and w2 == 0:         # valid
                    v, k = _rv(buf, k)
                    sync_valid = bool(v)
                elif f2 == 2 and w2 == 0:       # barrier_addr
                    barrier_addr, k = _rv(buf, k)
                elif f2 == 4 and w2 == 0:       # semantic_raw
                    semantic_raw, k = _rv(buf, k)
                else:
                    k = _skip(buf, k, w2)
            i = end
        elif f == 8 and w == 2:                 # tma_completion_runtime_info
            ln, j = _rv(buf, i)
            end = j + ln
            k = j
            while k < end:
                t2, k = _rv(buf, k)
                f2, w2 = t2 >> 3, t2 & 7
                if f2 == 1 and w2 == 0:         # valid
                    v, k = _rv(buf, k)
                    tma_completion_valid = bool(v)
                elif f2 == 2 and w2 == 0:       # barrier_addr
                    tma_completion_barrier_addr, k = _rv(buf, k)
                elif f2 == 3 and w2 == 0:       # barrier_ureg_id
                    tma_completion_barrier_ureg_id, k = _rv(buf, k)
                else:
                    k = _skip(buf, k, w2)
            i = end
        elif f == 9 and w == 2:                 # mbarrier_contract_runtime_info
            ln, j = _rv(buf, i)
            end = j + ln
            k = j
            while k < end:
                t2, k = _rv(buf, k)
                f2, w2 = t2 >> 3, t2 & 7
                if w2 != 0:
                    k = _skip(buf, k, w2)
                    continue
                value, k = _rv(buf, k)
                if f2 == 1:
                    contract_capture_valid = bool(value)
                elif f2 == 2:
                    contract_event_kind = value
                elif f2 == 3:
                    contract_barrier_addr = value
                elif f2 == 4:
                    contract_epoch_before = value
                elif f2 == 5:
                    contract_epoch_after = value
                elif f2 == 6:
                    contract_opened_before = value
                elif f2 == 7:
                    contract_opened_after = value
                elif f2 == 8:
                    contract_has_ordinal = bool(value)
                elif f2 == 9:
                    contract_ordinal = value
                elif f2 == 10:
                    contract_capture_flags = value
                elif f2 == 11:
                    contract_barrier_error_flags = value
                elif f2 == 12:
                    contract_capture_launch_uid = value
                elif f2 == 13:
                    contract_has_semantic_raw = bool(value)
                elif f2 == 14:
                    contract_semantic_raw = value
            i = end
            if contract_capture_valid:
                if contract_epoch_before is None:
                    contract_epoch_before = 0
                if contract_epoch_after is None:
                    contract_epoch_after = 0
                if contract_opened_before is None:
                    contract_opened_before = 0
                if contract_opened_after is None:
                    contract_opened_after = 0
            if (contract_has_semantic_raw
                    and contract_semantic_raw is None):
                contract_semantic_raw = 0
        else:
            i = _skip(buf, i, w)
    return {
        "pc": pc,
        "active_mask": active_mask,
        "predicate_mask": predicate_mask,
        "function_unique_id": function_unique_id,
        "sync_valid": sync_valid,
        "barrier_addr": barrier_addr,
        "semantic_raw": semantic_raw,
        "tma_completion_valid": tma_completion_valid,
        "tma_completion_barrier_addr": tma_completion_barrier_addr,
        "tma_completion_barrier_ureg_id":
            tma_completion_barrier_ureg_id,
        "contract_capture_valid": contract_capture_valid,
        "contract_event_kind": contract_event_kind,
        "contract_barrier_addr": contract_barrier_addr,
        "contract_epoch_before": contract_epoch_before,
        "contract_epoch_after": contract_epoch_after,
        "contract_opened_before": contract_opened_before,
        "contract_opened_after": contract_opened_after,
        "contract_has_ordinal": contract_has_ordinal,
        "contract_ordinal": contract_ordinal,
        "contract_capture_flags": contract_capture_flags,
        "contract_barrier_error_flags": contract_barrier_error_flags,
        "contract_capture_launch_uid": contract_capture_launch_uid,
        "contract_has_semantic_raw": contract_has_semantic_raw,
        "contract_semantic_raw": contract_semantic_raw,
    }


def vpb_parse_full(buf):
    """Return list of (warp_key, [inst_dict, ...]) preserving masks + sync payload.

    Mirrors _ts._vpb_parse's threadblock -> warp(field 2, key=field 1,
    insts=field 2) -> instruction(field 2) wire walk, but keeps every field the
    compound wait-block proof needs.
    """
    i = 0
    n = len(buf)
    warps = []
    while i < n:
        tag, i = _rv(buf, i)
        f, w = tag >> 3, tag & 7
        if f == 2 and w == 2:
            ln, i = _rv(buf, i)
            entry_end = i + ln
            j = i
            key = None
            insts = []
            while j < entry_end:
                t2, j = _rv(buf, j)
                f2, w2 = t2 >> 3, t2 & 7
                if f2 == 1 and w2 == 0:
                    key, j = _rv(buf, j)
                elif f2 == 2 and w2 == 2:
                    l2, j = _rv(buf, j)
                    ve = j + l2
                    p = j
                    while p < ve:
                        t3, p = _rv(buf, p)
                        f3, w3 = t3 >> 3, t3 & 7
                        if f3 == 2 and w3 == 2:
                            l3, pp = _rv(buf, p)
                            se = pp + l3
                            insts.append(_inst_full(buf, pp, se))
                            p = se
                        else:
                            p = _skip(buf, p, w3)
                    j = ve
                else:
                    j = _skip(buf, j, w2)
            warps.append((key, insts))
            i = entry_end
        else:
            i = _skip(buf, i, w)
    return warps


CONTRACT_EVENT_EXCH = 1
CONTRACT_EVENT_ARRIVE = 2
CONTRACT_EVENT_WAIT = 3
CONTRACT_CAPTURE_PLAIN_ARRIVE = 1 << 0
CONTRACT_CAPTURE_ARRIVE_BEFORE_INIT = 1 << 1
CONTRACT_CAPTURE_DUPLICATE_INIT = 1 << 2
CONTRACT_MAX_INTERVAL_WIDTH = 4096

CONTRACT_CAPTURE_MISSING = "CONTRACT_CAPTURE_MISSING"
CONTRACT_CAPTURE_EVENT_MISMATCH = "CONTRACT_CAPTURE_EVENT_MISMATCH"
CONTRACT_CAPTURE_ADDRESS_MISMATCH = "CONTRACT_CAPTURE_ADDRESS_MISMATCH"
CONTRACT_CAPTURE_SEMANTIC_MISMATCH = "CONTRACT_CAPTURE_SEMANTIC_MISMATCH"
CONTRACT_LAUNCH_IDENTITY_MISMATCH = \
    "CONTRACT_LAUNCH_IDENTITY_MISMATCH"
CONTRACT_CAPTURE_DEVICE_ERROR = "CONTRACT_CAPTURE_DEVICE_ERROR"
CONTRACT_INITIALIZATION_MISSING = "CONTRACT_INITIALIZATION_MISSING"
CONTRACT_INITIALIZATION_DUPLICATE = "CONTRACT_INITIALIZATION_DUPLICATE"
CONTRACT_INITIALIZATION_EPOCH_UNSUPPORTED = \
    "CONTRACT_INITIALIZATION_EPOCH_UNSUPPORTED"
CONTRACT_ARRIVAL_COUNT_UNSUPPORTED = "CONTRACT_ARRIVAL_COUNT_UNSUPPORTED"
CONTRACT_PLAIN_ARRIVE_PRESENT = "CONTRACT_PLAIN_ARRIVE_PRESENT"
CONTRACT_EXPECT_TX_MISSING = "CONTRACT_EXPECT_TX_MISSING"
CONTRACT_ORDINAL_MISSING = "CONTRACT_ORDINAL_MISSING"
CONTRACT_ORDINAL_DUPLICATE = "CONTRACT_ORDINAL_DUPLICATE"
CONTRACT_ORDINAL_NONCONTIGUOUS = "CONTRACT_ORDINAL_NONCONTIGUOUS"
CONTRACT_ORDINAL_INTERVAL_MISMATCH = \
    "CONTRACT_ORDINAL_INTERVAL_MISMATCH"
CONTRACT_COUNT_REGRESSION = "CONTRACT_COUNT_REGRESSION"
CONTRACT_INTERVAL_TOO_WIDE = "CONTRACT_INTERVAL_TOO_WIDE"
CONTRACT_WAIT_RESULT_UNPROVEN = "CONTRACT_WAIT_RESULT_UNPROVEN"
CONTRACT_GENERATION_INCONSISTENT = "CONTRACT_GENERATION_INCONSISTENT"
CONTRACT_GENERATION_AMBIGUOUS = "CONTRACT_GENERATION_AMBIGUOUS"
CONTRACT_TMA_OWNERSHIP_UNPROVEN = "CONTRACT_TMA_OWNERSHIP_UNPROVEN"
CONTRACT_SERIAL_LIFECYCLE_UNPROVEN = \
    "CONTRACT_SERIAL_LIFECYCLE_UNPROVEN"

CONTRACT_REJECT_REASONS = (
    CONTRACT_CAPTURE_MISSING,
    CONTRACT_CAPTURE_EVENT_MISMATCH,
    CONTRACT_CAPTURE_ADDRESS_MISMATCH,
    CONTRACT_CAPTURE_SEMANTIC_MISMATCH,
    CONTRACT_LAUNCH_IDENTITY_MISMATCH,
    CONTRACT_CAPTURE_DEVICE_ERROR,
    CONTRACT_INITIALIZATION_MISSING,
    CONTRACT_INITIALIZATION_DUPLICATE,
    CONTRACT_INITIALIZATION_EPOCH_UNSUPPORTED,
    CONTRACT_ARRIVAL_COUNT_UNSUPPORTED,
    CONTRACT_PLAIN_ARRIVE_PRESENT,
    CONTRACT_EXPECT_TX_MISSING,
    CONTRACT_ORDINAL_MISSING,
    CONTRACT_ORDINAL_DUPLICATE,
    CONTRACT_ORDINAL_NONCONTIGUOUS,
    CONTRACT_ORDINAL_INTERVAL_MISMATCH,
    CONTRACT_COUNT_REGRESSION,
    CONTRACT_INTERVAL_TOO_WIDE,
    CONTRACT_WAIT_RESULT_UNPROVEN,
    CONTRACT_GENERATION_INCONSISTENT,
    CONTRACT_GENERATION_AMBIGUOUS,
    CONTRACT_TMA_OWNERSHIP_UNPROVEN,
    CONTRACT_SERIAL_LIFECYCLE_UNPROVEN,
)


def decode_exch_arrival_count(raw):
    raw = int(raw)
    delta = 0x200000 - raw
    if raw < 0 or delta <= 0 or delta % 2:
        return None
    return delta // 2


def decode_wait_parity(raw):
    return (int(raw) >> 31) & 1


def resolve_contract_generation(opened_before, opened_after, wait_parity,
                                captured_ready):
    """Return every target generation allowed by one wait snapshot interval.

    With one arrival and one contract per generation, N opened contracts allow
    only N-1 or N completed generations. A miss targets the next generation;
    a ready observation targets the generation that already satisfied it.
    """
    lo = int(opened_before)
    hi = int(opened_after)
    if hi < lo:
        return [], CONTRACT_COUNT_REGRESSION
    if hi - lo > CONTRACT_MAX_INTERVAL_WIDTH:
        return [], CONTRACT_INTERVAL_TOO_WIDE
    targets = set()
    for opened in range(lo, hi + 1):
        completed_candidates = [opened]
        if opened > 0:
            completed_candidates.append(opened - 1)
        for completed in completed_candidates:
            ready = (completed & 1) != int(wait_parity)
            if ready != bool(captured_ready):
                continue
            targets.add(completed if ready else completed + 1)
    return sorted(targets), None


def validate_contract_barrier(events, tma_ownership_valid,
                              captured_wait_results=()):
    reasons = []
    exch = [event for event in events
            if event["contract_event_kind"] == CONTRACT_EVENT_EXCH]
    arrives = [event for event in events
               if event["contract_event_kind"] == CONTRACT_EVENT_ARRIVE]
    if not tma_ownership_valid:
        reasons.append(CONTRACT_TMA_OWNERSHIP_UNPROVEN)
    if not exch:
        reasons.append(CONTRACT_INITIALIZATION_MISSING)
    elif len(exch) != 1:
        reasons.append(CONTRACT_INITIALIZATION_DUPLICATE)
    else:
        expected = decode_exch_arrival_count(
            0 if exch[0]["semantic_raw"] is None
            else exch[0]["semantic_raw"])
        if expected != 1:
            reasons.append(CONTRACT_ARRIVAL_COUNT_UNSUPPORTED)
    if any(event["contract_capture_flags"] &
           CONTRACT_CAPTURE_DUPLICATE_INIT for event in events):
        reasons.append(CONTRACT_INITIALIZATION_DUPLICATE)
    if any(event["contract_capture_flags"] &
           CONTRACT_CAPTURE_ARRIVE_BEFORE_INIT for event in events):
        reasons.append(CONTRACT_INITIALIZATION_EPOCH_UNSUPPORTED)
    if any(event["contract_barrier_error_flags"] for event in events):
        reasons.append(CONTRACT_CAPTURE_DEVICE_ERROR)
    if any(event["contract_epoch_before"] != event["contract_epoch_after"]
           for event in events
           if event["contract_event_kind"] != CONTRACT_EVENT_EXCH):
        reasons.append(CONTRACT_INITIALIZATION_EPOCH_UNSUPPORTED)
    if any(event["contract_epoch_after"] != 1 for event in events):
        reasons.append(CONTRACT_INITIALIZATION_EPOCH_UNSUPPORTED)

    plain_arrives = [
        event for event in arrives
        if not int(event["semantic_raw"] or 0)
    ]
    expect_arrives = [
        event for event in arrives
        if int(event["semantic_raw"] or 0)
    ]
    if plain_arrives:
        reasons.append(CONTRACT_PLAIN_ARRIVE_PRESENT)
    if not expect_arrives:
        reasons.append(CONTRACT_EXPECT_TX_MISSING)
    ordinals = []
    for event in expect_arrives:
        if (not event["contract_has_ordinal"]
                or event["contract_ordinal"] is None
                or int(event["contract_ordinal"]) <= 0):
            reasons.append(CONTRACT_ORDINAL_MISSING)
            continue
        ordinal = int(event["contract_ordinal"])
        ordinals.append(ordinal)
        before = event["contract_opened_before"]
        after = event["contract_opened_after"]
        if (before is None or after is None
                or ordinal <= int(before)
                or ordinal > int(after)):
            reasons.append(CONTRACT_ORDINAL_INTERVAL_MISMATCH)
    if len(ordinals) != len(set(ordinals)):
        reasons.append(CONTRACT_ORDINAL_DUPLICATE)
    if sorted(set(ordinals)) != list(range(1, len(ordinals) + 1)):
        reasons.append(CONTRACT_ORDINAL_NONCONTIGUOUS)
    for event in events:
        before = event["contract_opened_before"]
        after = event["contract_opened_after"]
        if before is None or after is None:
            reasons.append(CONTRACT_CAPTURE_MISSING)
        elif int(after) < int(before):
            reasons.append(CONTRACT_COUNT_REGRESSION)
        elif int(after) - int(before) > CONTRACT_MAX_INTERVAL_WIDTH:
            reasons.append(CONTRACT_INTERVAL_TOO_WIDE)
    if not reasons and len(expect_arrives) > 1:
        ready_generations = set()
        for wait, captured_ready in captured_wait_results:
            if not captured_ready:
                continue
            before = wait["contract_opened_before"]
            after = wait["contract_opened_after"]
            if before is None or after is None or int(before) != int(after):
                continue
            parity = decode_wait_parity(
                0 if wait["semantic_raw"] is None
                else wait["semantic_raw"])
            targets, error = resolve_contract_generation(
                before, after, parity, True)
            if error is None and len(targets) == 1:
                ready_generations.add(targets[0])
        required = set(range(1, len(expect_arrives)))
        if not required.issubset(ready_generations):
            reasons.append(CONTRACT_SERIAL_LIFECYCLE_UNPROVEN)
    reasons = sorted(set(reasons))
    return {
        "eligible": not reasons,
        "reject_reasons": reasons,
        "contracts": len(expect_arrives),
        "initializations": len(exch),
        "plain_arrives": len(plain_arrives),
    }


def _contract_expected_event_kind(sync_kind):
    if sync_kind == "EXCH":
        return CONTRACT_EVENT_EXCH
    if sync_kind == "ARRIVE_EXPECT_TX":
        return CONTRACT_EVENT_ARRIVE
    if sync_kind in WAIT_KINDS:
        return CONTRACT_EVENT_WAIT
    return None


def _fold_wait_result(insts, index, spec):
    return _fold_wait_site_result(insts, index, spec)


def _fold_wait_site_result(insts, index, spec):
    """Recover one wait result from the proven fold-loop control path.

    A wait before poll_probe_pc is a miss when the active path reaches the
    later poll or loopback. It is ready when both are predicated off and the
    first matching active path reaches poll_loop_exit_pc.
    """
    owner_mask = (
        int(insts[index]["active_mask"])
        & int(insts[index]["predicate_mask"])
    )
    current_pc = int(insts[index]["pc"])
    for following in insts[index + 1:]:
        if int(following["function_unique_id"]) != int(
                insts[index]["function_unique_id"]):
            continue
        effective = (
            int(following["active_mask"])
            & int(following["predicate_mask"])
        )
        if effective & owner_mask == 0:
            continue
        pc = int(following["pc"])
        if (pc == int(spec["poll_probe_pc"])
                and current_pc != int(spec["poll_probe_pc"])):
            return False
        if pc == int(spec["poll_loopback_pc"]):
            return False
        if pc == int(spec["poll_loop_exit_pc"]):
            return True
    return None


def _load_tma_ownership_status(extra):
    path = os.path.join(extra, "tma_completion_contracts.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        document = json.load(handle)
    if not document.get("validation_complete"):
        return {}
    result = {}
    for kernel in document.get("kernels", []):
        kernel_id = int(kernel["trace_kernel_id"])
        result[kernel_id] = bool(
            not kernel.get("errors")
            and int(kernel.get("contracts", 0)) > 0
            and int(kernel.get("expected_bytes", -1))
            == int(kernel.get("assigned_bytes", -2))
        )
    return result


def analyze_mbarrier_contract_capture(traces, kdirs, kid_to_fid,
                                      sync_sites, sidecar_kernels,
                                      max_ctas):
    action_specs = {}
    fold_wait_specs = {}
    for kernel in sidecar_kernels:
        kid = int(kernel["trace_kernel_id"])
        for record in kernel.get("wait_blocks", []):
            strategy = record.get("runtime_strategy")
            if strategy == "INJECT_MISSING_WAIT":
                pc = int(record["entry_probe_pc"])
            elif strategy == "FOLD_CAPTURED_POLL":
                pc = int(record["poll_probe_pc"])
                for wait_pc in record["poll_wait_pcs"]:
                    fold_wait_specs[
                        (kid, int(record["unique_function_id"]),
                         int(wait_pc))] = record
            else:
                continue
            action_specs[(kid, int(record["unique_function_id"]), pc)] = record

    ownership = _load_tma_ownership_status(
        os.path.join(traces, "extra_info"))
    report = Counter()
    for field in (
            "active_sync_occurrences", "missing_snapshots",
            "duplicate_snapshots", "capture_mismatches", "barriers",
            "eligible_barriers", "unsupported_barriers",
            "unsupported_snapshots_on_eligible_barriers",
            "eligible_action_occurrences",
            "unsupported_action_occurrences",
            "rejected_tma_action_occurrences", "unique_assignments",
            "ambiguous_assignments", "inconsistent_assignments",
            "unique_streams", "ambiguous_or_invalid_streams"):
        report[field] = 0
    tickets = []
    rejects = []
    barrier_rejections = []
    stream_results = defaultdict(list)

    for kid in sorted(kdirs):
        fid = kid_to_fid.get(kid)
        if fid is None:
            continue
        paths = sorted(glob.glob(os.path.join(kdirs[kid], "*.pb")))
        if max_ctas != "all":
            paths = paths[:int(max_ctas)]
        for path in paths:
            cta_name = os.path.basename(path)
            with open(path, "rb") as handle:
                warps = vpb_parse_full(handle.read())
            barrier_events = defaultdict(list)
            action_waits = []
            captured_wait_results = defaultdict(list)
            uncaptured_waits = defaultdict(list)
            capture_launch_uids = set()
            for warp_id, insts in warps:
                for index, inst in enumerate(insts):
                    effective = (
                        int(inst["active_mask"])
                        & int(inst["predicate_mask"])
                    )
                    site = (
                        int(inst["function_unique_id"]), int(inst["pc"]))
                    sync_site = sync_sites.get(site)
                    expected_kind = (
                        _contract_expected_event_kind(
                            sync_site.get("sync_kind"))
                        if sync_site else None
                    )
                    spec = action_specs.get((kid,) + site)
                    if effective and spec is not None:
                        result = None
                        if (spec["runtime_strategy"]
                                == "INJECT_MISSING_WAIT"):
                            result = True
                        elif (spec["runtime_strategy"]
                              == "FOLD_CAPTURED_POLL"):
                            result = _fold_wait_result(
                                insts, index, spec)
                        action_waits.append({
                            "warp_id": warp_id,
                            "source_warp_instruction_index": index,
                            "instruction": inst,
                            "spec": spec,
                            "captured_ready": result,
                        })
                    if effective and expected_kind is not None:
                        report["active_sync_occurrences"] += 1
                        if not inst["contract_capture_valid"]:
                            if (expected_kind == CONTRACT_EVENT_WAIT
                                    and inst["barrier_addr"] is not None):
                                uncaptured_waits[
                                    int(inst["barrier_addr"])].append({
                                        "warp_id": warp_id,
                                        "pc": inst["pc"],
                                    })
                            else:
                                report["missing_snapshots"] += 1
                                rejects.append({
                                    "kernel_id": kid,
                                    "cta": cta_name,
                                    "warp_id": warp_id,
                                    "pc": inst["pc"],
                                    "reason": CONTRACT_CAPTURE_MISSING,
                                })
                            continue
                        if inst["contract_event_kind"] != expected_kind:
                            report["capture_mismatches"] += 1
                            rejects.append({
                                "kernel_id": kid,
                                "cta": cta_name,
                                "warp_id": warp_id,
                                "pc": inst["pc"],
                                "reason": CONTRACT_CAPTURE_EVENT_MISMATCH,
                            })
                            continue
                        if (inst["contract_barrier_addr"]
                                != inst["barrier_addr"]):
                            report["capture_mismatches"] += 1
                            rejects.append({
                                "kernel_id": kid,
                                "cta": cta_name,
                                "warp_id": warp_id,
                                "pc": inst["pc"],
                                "reason": CONTRACT_CAPTURE_ADDRESS_MISMATCH,
                            })
                            continue
                        semantic_raw = (
                            0 if inst["semantic_raw"] is None
                            else int(inst["semantic_raw"]))
                        if (not inst["contract_has_semantic_raw"]
                                or inst["contract_semantic_raw"]
                                != semantic_raw):
                            report["capture_mismatches"] += 1
                            rejects.append({
                                "kernel_id": kid,
                                "cta": cta_name,
                                "warp_id": warp_id,
                                "pc": inst["pc"],
                                "reason":
                                    CONTRACT_CAPTURE_SEMANTIC_MISMATCH,
                            })
                            continue
                        launch_uid = inst["contract_capture_launch_uid"]
                        if launch_uid is None or int(launch_uid) <= 0:
                            report["capture_mismatches"] += 1
                            rejects.append({
                                "kernel_id": kid,
                                "cta": cta_name,
                                "warp_id": warp_id,
                                "pc": inst["pc"],
                                "reason":
                                    CONTRACT_LAUNCH_IDENTITY_MISMATCH,
                            })
                            continue
                        capture_launch_uids.add(int(launch_uid))
                        barrier_events[
                            int(inst["contract_barrier_addr"])].append(inst)
                        if expected_kind == CONTRACT_EVENT_WAIT:
                            result = None
                            direct_spec = action_specs.get((kid,) + site)
                            if direct_spec is not None:
                                if (direct_spec["runtime_strategy"]
                                        == "INJECT_MISSING_WAIT"):
                                    result = True
                                else:
                                    result = _fold_wait_site_result(
                                        insts, index, direct_spec)
                            else:
                                fold_spec = fold_wait_specs.get(
                                    (kid, site[0], int(inst["pc"])))
                                if fold_spec is not None:
                                    result = _fold_wait_site_result(
                                        insts, index, fold_spec)
                            if result is not None:
                                captured_wait_results[
                                    int(inst[
                                        "contract_barrier_addr"])].append(
                                            (inst, result))

            if len(capture_launch_uids) > 1:
                report["capture_mismatches"] += 1
                rejects.append({
                    "kernel_id": kid,
                    "cta": cta_name,
                    "reason": CONTRACT_LAUNCH_IDENTITY_MISMATCH,
                    "capture_launch_uids":
                        sorted(capture_launch_uids),
                })
            barrier_status = {
                barrier: validate_contract_barrier(
                    events, ownership.get(kid, False),
                    captured_wait_results.get(barrier, ()))
                for barrier, events in barrier_events.items()
            }
            for barrier, occurrences in uncaptured_waits.items():
                status = barrier_status.get(barrier)
                if status is None or status["contracts"] == 0:
                    continue
                report["missing_snapshots"] += len(occurrences)
                report[
                    "unsupported_snapshots_on_eligible_barriers"
                ] += len(occurrences)
                status["eligible"] = False
                status["reject_reasons"] = sorted(set(
                    status["reject_reasons"]
                    + [CONTRACT_CAPTURE_MISSING]))
                for occurrence in occurrences:
                    rejects.append({
                        "kernel_id": kid,
                        "cta": cta_name,
                        "warp_id": occurrence["warp_id"],
                        "pc": occurrence["pc"],
                        "barrier_addr": barrier,
                        "reason": CONTRACT_CAPTURE_MISSING,
                    })
            report["barriers"] += len(barrier_status)
            report["eligible_barriers"] += sum(
                status["eligible"]
                for status in barrier_status.values())
            report["unsupported_barriers"] += sum(
                not status["eligible"]
                for status in barrier_status.values())
            for barrier, status in barrier_status.items():
                if status["eligible"]:
                    continue
                barrier_rejections.append({
                    "trace_kernel_id": kid,
                    "capture_launch_uid":
                        (next(iter(capture_launch_uids))
                         if len(capture_launch_uids) == 1 else None),
                    "cta_file": cta_name,
                    "barrier_addr": barrier,
                    "reject_reasons": status["reject_reasons"],
                    "contracts": status["contracts"],
                    "initializations": status["initializations"],
                    "plain_arrives": status["plain_arrives"],
                })
            for wait in action_waits:
                inst = wait["instruction"]
                barrier = (
                    inst["contract_barrier_addr"]
                    if inst["contract_barrier_addr"] is not None
                    else inst["barrier_addr"])
                status = barrier_status.get(barrier)
                if status is None or not status["eligible"]:
                    report["unsupported_action_occurrences"] += 1
                    if status is not None and status["contracts"] > 0:
                        report["rejected_tma_action_occurrences"] += 1
                        rejects.append({
                            "kernel_id": kid,
                            "cta": cta_name,
                            "warp_id": wait["warp_id"],
                            "pc": inst["pc"],
                            "barrier_addr": barrier,
                            "reason": status["reject_reasons"],
                        })
                    continue
                report["eligible_action_occurrences"] += 1
                if wait["captured_ready"] is None:
                    report["inconsistent_assignments"] += 1
                    rejects.append({
                        "kernel_id": kid,
                        "cta": cta_name,
                        "warp_id": wait["warp_id"],
                        "pc": inst["pc"],
                        "reason": CONTRACT_WAIT_RESULT_UNPROVEN,
                    })
                    continue
                wait_parity = decode_wait_parity(
                    0 if inst["semantic_raw"] is None
                    else inst["semantic_raw"])
                targets, interval_error = resolve_contract_generation(
                    inst["contract_opened_before"],
                    inst["contract_opened_after"],
                    wait_parity,
                    wait["captured_ready"],
                )
                if interval_error is not None:
                    report["inconsistent_assignments"] += 1
                    reason = interval_error
                elif not targets:
                    report["inconsistent_assignments"] += 1
                    reason = CONTRACT_GENERATION_INCONSISTENT
                elif len(targets) != 1:
                    report["ambiguous_assignments"] += 1
                    reason = CONTRACT_GENERATION_AMBIGUOUS
                else:
                    report["unique_assignments"] += 1
                    reason = None
                ticket = {
                    "trace_kernel_id": kid,
                    "capture_launch_uid":
                        int(inst["contract_capture_launch_uid"]),
                    "cta_file": cta_name,
                    "warp_id": wait["warp_id"],
                    "unique_function_id":
                        int(inst["function_unique_id"]),
                    "pc": int(inst["pc"]),
                    "source_warp_instruction_index":
                        wait["source_warp_instruction_index"],
                    "runtime_strategy":
                        wait["spec"]["runtime_strategy"],
                    "barrier_addr": int(barrier),
                    "initialization_epoch":
                        int(inst["contract_epoch_after"]),
                    "opened_contract_interval": [
                        int(inst["contract_opened_before"]),
                        int(inst["contract_opened_after"]),
                    ],
                    "wait_parity": wait_parity,
                    "captured_result":
                        "READY" if wait["captured_ready"] else "MISS",
                    "candidate_target_generations": targets,
                    "target_release_generation":
                        targets[0] if len(targets) == 1 else None,
                    "proof_inputs": {
                        "initialization_count":
                            status["initializations"],
                        "decoded_expected_arrival_count": 1,
                        "plain_arrive_count":
                            status["plain_arrives"],
                        "expect_tx_contract_count":
                            status["contracts"],
                        "exact_tma_ownership": True,
                        "serial_same_barrier_lifecycle": True,
                    },
                    "status": "UNIQUE" if reason is None else "REJECT",
                    "reject_reason": reason,
                }
                tickets.append(ticket)
                stream_key = (
                    kid, cta_name, wait["warp_id"], int(barrier),
                    wait["spec"]["runtime_strategy"],
                )
                stream_results[stream_key].append(ticket)
                if reason is not None:
                    rejects.append(ticket)

    for stream in stream_results.values():
        if all(ticket["status"] == "UNIQUE" for ticket in stream):
            report["unique_streams"] += 1
        else:
            report["ambiguous_or_invalid_streams"] += 1

    required_zero = (
        "missing_snapshots",
        "capture_mismatches",
        "rejected_tma_action_occurrences",
        "inconsistent_assignments",
        "ambiguous_assignments",
        "ambiguous_or_invalid_streams",
    )
    status = (
        "PASS"
        if report["eligible_action_occurrences"] > 0
        and all(report[name] == 0 for name in required_zero)
        else "FAIL"
    )
    return {
        "version": 1,
        "schema": "mbarrier_contract_ordinal_feasibility",
        "status": status,
        "model": "ONE_ARRIVAL_ONE_EXPECT_TX_CONTRACT_PER_GENERATION",
        "reject_reason_taxonomy": list(CONTRACT_REJECT_REASONS),
        "summary": dict(report),
        "tickets": tickets,
        "barrier_rejections": barrier_rejections,
        "rejects": rejects,
    }


def semantic_trace_fingerprint(traces, kernel_ids):
    counts = Counter()
    site_counts = Counter()
    kdirs = kernel_dir_map(traces)
    static_instructions, _ = load_kernels(
        os.path.join(traces, "extra_info"))
    for kernel_id in sorted(kernel_ids):
        kdir = kdirs.get(kernel_id)
        if kdir is None:
            continue
        paths = sorted(glob.glob(os.path.join(kdir, "*.pb")))
        counts["kernel_%d_ctas" % kernel_id] = len(paths)
        for path in paths:
            with open(path, "rb") as handle:
                warps = vpb_parse_full(handle.read())
            for _, instructions in warps:
                for inst in instructions:
                    effective_mask = (
                        int(inst["active_mask"])
                        & int(inst["predicate_mask"])
                    )
                    function_id = int(inst["function_unique_id"])
                    pc = int(inst["pc"])
                    opcode = static_instructions.get(
                        function_id, {}).get(pc, {}).get("op_code", "")
                    if not effective_mask or not is_semantic(opcode):
                        continue
                    fields = (
                        kernel_id,
                        function_id,
                        pc,
                        opcode,
                        int(inst["active_mask"]),
                        int(inst["predicate_mask"]),
                    )
                    site_counts[fields] += 1
                    counts["semantic_instructions"] += 1
                    if inst["tma_completion_valid"]:
                        counts["completion_bearing_tma"] += 1
    digest = hashlib.sha256()
    for fields, count in sorted(site_counts.items()):
        digest.update(
            ("%s|%d\n"
             % ("|".join(str(value) for value in fields), count))
            .encode("ascii"))
    counts["semantic_sites"] = len(site_counts)
    return {
        "sha256": digest.hexdigest(),
        "counts": dict(sorted(counts.items())),
    }


def compare_contract_capture_baseline(current_traces, baseline_traces,
                                      current_sidecar_kernels,
                                      current_wait_report):
    kernel_ids = sorted(
        int(kernel["trace_kernel_id"])
        for kernel in current_sidecar_kernels
        if any(
            record.get("runtime_strategy")
            in ("INJECT_MISSING_WAIT", "FOLD_CAPTURED_POLL")
            for record in kernel.get("wait_blocks", [])
        )
    )
    if not kernel_ids:
        return {
            "status": "FAIL",
            "reason": "WAIT_BLOCK_KERNELS_NOT_PRESENT",
        }
    baseline_extra = os.path.join(baseline_traces, "extra_info")
    baseline_sidecar_path = os.path.join(
        baseline_extra, "spin_wait_sidecar.json")
    baseline_contract_path = os.path.join(
        baseline_extra, "tma_completion_contracts.json")
    current_contract_path = os.path.join(
        current_traces, "extra_info", "tma_completion_contracts.json")
    missing = [
        path for path in (
            baseline_sidecar_path,
            baseline_contract_path,
            current_contract_path,
        )
        if not os.path.exists(path)
    ]
    if missing:
        return {
            "status": "FAIL",
            "reason": "BASELINE_ARTIFACT_MISSING",
            "missing": missing,
        }

    current_fingerprint = semantic_trace_fingerprint(
        current_traces, kernel_ids)
    baseline_fingerprint = semantic_trace_fingerprint(
        baseline_traces, kernel_ids)
    with open(baseline_sidecar_path) as handle:
        baseline_sidecar = json.load(handle)
    with open(baseline_contract_path) as handle:
        baseline_contracts = json.load(handle)
    with open(current_contract_path) as handle:
        current_contracts = json.load(handle)

    def wait_blocks_by_kernel(kernels, strip_fold_dynamic=False):
        result = {}
        for kernel in kernels:
            kernel_id = int(kernel["trace_kernel_id"])
            if kernel_id not in kernel_ids:
                continue
            records = []
            for original in kernel.get("wait_blocks", []):
                record = dict(original)
                if (strip_fold_dynamic
                        and record.get("runtime_strategy")
                        == "FOLD_CAPTURED_POLL"):
                    record.pop("dynamic", None)
                records.append(record)
            result[kernel_id] = records
        return result

    def fold_dynamic_by_block(kernels):
        return {
            "%d:%s" % (
                int(kernel["trace_kernel_id"]),
                record["block_id"]):
                record.get("dynamic", {})
            for kernel in kernels
            if int(kernel["trace_kernel_id"]) in kernel_ids
            for record in kernel.get("wait_blocks", [])
            if record.get("runtime_strategy")
            == "FOLD_CAPTURED_POLL"
        }

    def contract_kernel_summary(document):
        return {
            int(kernel["trace_kernel_id"]): {
                "threadblocks": int(kernel.get("threadblocks", 0)),
                "contracts": int(kernel.get("contracts", 0)),
                "completion_loads": int(
                    kernel.get("completion_loads", 0)),
                "expected_bytes": int(
                    kernel.get("expected_bytes", 0)),
                "assigned_bytes": int(
                    kernel.get("assigned_bytes", 0)),
                "errors": kernel.get("errors", []),
            }
            for kernel in document.get("kernels", [])
            if int(kernel["trace_kernel_id"]) in kernel_ids
        }

    current_wait_blocks = wait_blocks_by_kernel(
        current_sidecar_kernels, strip_fold_dynamic=True)
    baseline_wait_blocks = wait_blocks_by_kernel(
        baseline_sidecar.get("kernels", []),
        strip_fold_dynamic=True)
    current_fold_dynamic = fold_dynamic_by_block(
        current_sidecar_kernels)
    baseline_fold_dynamic = fold_dynamic_by_block(
        baseline_sidecar.get("kernels", []))
    wait_report_keys = (
        "inject_missing_wait",
        "fold_captured_poll",
        "ready_only_all_live_visits",
        "ready_only_partial_visits",
        "partial_defer_records",
        "partial_join_pending_visits",
        "partial_join_already_issued_visits",
    )
    current_wait_counts = {
        key: int(current_wait_report.get(key, 0))
        for key in wait_report_keys
    }
    baseline_wait_counts = {
        key: int(
            baseline_sidecar.get("wait_block_report", {}).get(key, 0))
        for key in wait_report_keys
    }
    comparisons = {
        "semantic_fingerprint":
            current_fingerprint == baseline_fingerprint,
        "wait_block_structure":
            current_wait_blocks == baseline_wait_blocks,
        "wait_block_counts":
            current_wait_counts == baseline_wait_counts,
        "tma_contracts":
            contract_kernel_summary(current_contracts)
            == contract_kernel_summary(baseline_contracts),
    }
    return {
        "status": (
            "PASS" if all(comparisons.values()) else "FAIL"),
        "current_traces": os.path.abspath(current_traces),
        "baseline_traces": os.path.abspath(baseline_traces),
        "kernel_ids": kernel_ids,
        "comparisons": comparisons,
        "current_semantic_trace": current_fingerprint,
        "baseline_semantic_trace": baseline_fingerprint,
        "current_wait_counts": current_wait_counts,
        "baseline_wait_counts": baseline_wait_counts,
        "fold_dynamic_equal":
            current_fold_dynamic == baseline_fold_dynamic,
        "current_fold_dynamic": current_fold_dynamic,
        "baseline_fold_dynamic": baseline_fold_dynamic,
        "current_tma_contracts":
            contract_kernel_summary(current_contracts),
        "baseline_tma_contracts":
            contract_kernel_summary(baseline_contracts),
    }

# Opcodes whose presence in a loop body makes it NOT a pure wait (would drop real
# work if the poll count were folded). Global/shared memory + tensor + async.
SEMANTIC_PREFIXES = (
    "HGMMA", "IGMMA", "QGMMA", "BGMMA", "OGMMA", "MMA",
    "MUFU",
    "UTMALDG", "UTMASTG", "UTMACMDFLUSH", "UBLKCP", "UBLKPF", "UTMAPF",
    "UTMAREDG", "UBLKRED", "UTMACCTL",
    "LDGSTS", "STG", "LDG", "STS", "LDS", "ATOM", "ATOMS", "RED",
    "SYNCS.ARRIVE", "SYNCS.EXCH", "BAR",
)
# Thread-local stack ops that are warp-local side effects only — safe inside a
# folded wait region, but recorded so the simulator/reviewer can see them.
LOCAL_MEM_PREFIXES = ("STL", "LDL")

WAIT_KINDS = ("PHASECHK", "TRYWAIT")


def is_semantic(op):
    return any(op == p or op.startswith(p + ".") or op.startswith(p)
               for p in SEMANTIC_PREFIXES)


def is_local_mem(op):
    return any(op.startswith(p) for p in LOCAL_MEM_PREFIXES)


# Async producers whose SASS scoreboard barrier release is DECOUPLED from issue
# (the barrier is released by a later async completion, not at the arming
# instruction's own pipeline stage). If one sits inside a spin body, whole-loop
# skip would drop an arm whose release lives outside the body and corrupt the
# scoreboard. Distinct from SEMANTIC_PREFIXES (overlaps, but this list is the
# barrier-safety gate specifically). See .plan/SPIN_WAIT_SIM_MODELING.md 12.7.
ASYNC_PRODUCER_PREFIXES = (
    "HGMMA", "IGMMA", "QGMMA", "BGMMA", "OGMMA", "MMA", "WGMMA",
    "UTMALDG", "UTMASTG", "UTMACMDFLUSH", "UBLKCP", "UBLKPF", "UTMAPF",
    "UTMAREDG", "UBLKRED", "UTMACCTL",
    "LDGSTS", "LDG", "STG", "LDS", "STS", "ATOM", "ATOMS", "RED",
    "DEPBAR", "LDGDEPBAR",
)


def is_async_producer(op):
    return any(op == p or op.startswith(p + ".") or op.startswith(p)
               for p in ASYNC_PRODUCER_PREFIXES)


def _cb(inst):
    return inst.get("control_bits", {}) or {}


def _normalized_operand(text):
    return (text or "").replace(".reuse", "").replace(" ", "")


def _wait_destination_predicate(inst):
    operands = inst.get("operands", [])
    if not operands:
        return None
    destination = _normalized_operand(operands[0].get("operand_string", ""))
    prefix = "UP" if destination.startswith("UP") else "P"
    digits = destination[len(prefix):]
    if not digits.isdigit():
        return None
    return {
        "name": prefix + digits,
        "register": int(digits),
        "uniform": prefix == "UP",
    }


def prove_poll_loop_control(region, byp):
    """Prove that the selected poll predicate owns the loopback decision."""
    primary = byp.get(region["primary_wait_pc"])
    branch = byp.get(region["loopback_bra_pc"])
    if primary is None or branch is None:
        return False, "POLL_CONTROL_METADATA_MISSING", None
    predicate = _wait_destination_predicate(primary)
    if predicate is None:
        return False, "POLL_PREDICATE_UNDECODABLE", None
    for wait_pc in region["wait_pcs"]:
        wait_predicate = _wait_destination_predicate(byp.get(wait_pc, {}))
        if wait_predicate != predicate:
            return False, "POLL_PREDICATE_MISMATCH", None
    target = bra_target(branch)
    branch_predicate = {
        "name": ("UP" if branch.get("is_uniform_predicate") else "P") +
                str(branch.get("predicate_register")),
        "register": int(branch.get("predicate_register", -1)),
        "uniform": bool(branch.get("is_uniform_predicate")),
    }
    if (branch["op_code"] != "BRA" or
            not branch.get("is_predicated") or
            not branch.get("is_predicate_negate") or
            branch_predicate != predicate or
            target != region["loop_head_pc"] or
            region["exit_pc"] != region["loopback_bra_pc"] + ISIZE):
        return False, "POLL_LOOPBACK_NOT_OWNED_BY_PRIMARY_WAIT", None
    return True, None, {
        "primary_predicate": predicate["name"],
        "all_wait_predicates_match": True,
        "loopback_predicate": branch_predicate["name"],
        "loopback_predicate_negated": True,
        "loopback_target_pc": target,
        "loopback_fallthrough_pc": region["exit_pc"],
    }


def prove_poll_wait_condition(region, byp):
    """Require every skipped wait to use the same static barrier and wait state."""
    sites = []
    expected = None
    for wait_pc in region["wait_pcs"]:
        inst = byp.get(wait_pc)
        operands = inst.get("operands", []) if inst else []
        if len(operands) < 3:
            return False, "POLL_WAIT_CONDITION_UNDECODABLE", None
        barrier = _normalized_operand(operands[1].get("operand_string", ""))
        wait_state = _normalized_operand(operands[2].get("operand_string", ""))
        if not barrier or not wait_state:
            return False, "POLL_WAIT_CONDITION_UNDECODABLE", None
        condition = (barrier, wait_state)
        if expected is None:
            expected = condition
        elif condition != expected:
            return False, "POLL_WAIT_CONDITION_MISMATCH", None
        sites.append({
            "pc": wait_pc,
            "barrier_operand": barrier,
            "wait_state_operand": wait_state,
        })
    return True, None, {
        "barrier_operand": expected[0],
        "wait_state_operand": expected[1],
        "sites": sites,
    }


def compute_barrier_self_tied(skip_pcs, byp):
    """A skip set is barrier-self-tied iff folding cannot corrupt the SASS
    scoreboard wait-barrier state. In this model a wait-barrier armed by an
    instruction is released by that SAME instruction's own pipeline stage (write
    barrier at WB, read barrier at read-stage), so an arm INSIDE the body is
    released INSIDE the body -> skipping is net-zero. The two ways it breaks:
      (1) a body instruction WAITS on a barrier id that is NOT armed inside the
          body (its arming instruction is outside the skip set), or
      (2) a body op is an async producer whose barrier release is decoupled from
          issue (GMMA/TMA/LDGSTS/DEPBAR/global-shared memory).
    Returns (self_tied: bool, reason_fields: dict) for logging.
    """
    armed_ids = set()
    waited_ids = set()
    async_ops = []
    per_pc = []
    for p in skip_pcs:
        inst = byp[p]
        op = inst["op_code"]
        cb = _cb(inst)
        pc_armed = []
        if cb.get("is_new_read_barrier"):
            barrier_id = int(cb.get("id_new_read_barrier", -1))
            armed_ids.add(barrier_id)
            pc_armed.append(barrier_id)
        if cb.get("is_new_write_barrier"):
            barrier_id = int(cb.get("id_new_write_barrier", -1))
            armed_ids.add(barrier_id)
            pc_armed.append(barrier_id)
        mask = int(cb.get("wait_barrier_bits", 0) or 0)
        pc_waited = []
        for bit in range(6):
            if mask & (1 << bit):
                waited_ids.add(bit)
                pc_waited.append(bit)
        if is_async_producer(op):
            async_ops.append((hex(p), op))
        per_pc.append({
            "pc": p,
            "opcode": op,
            "armed_ids": sorted(pc_armed),
            "waited_ids": sorted(pc_waited),
        })
    wait_on_external = sorted(waited_ids - armed_ids)
    self_tied = (not wait_on_external) and (not async_ops)
    return self_tied, {
        "checked_pcs": list(skip_pcs),
        "per_pc": per_pc,
        "armed_ids": sorted(armed_ids),
        "waited_ids": sorted(waited_ids),
        "wait_on_externally_armed_ids": wait_on_external,
        "async_producer_ops": async_ops,
    }


def bra_target(inst):
    if inst["op_code"].split(".")[0] != "BRA":
        return None
    for o in inst.get("operands", []):
        if o.get("has_inmediate"):
            return int(o["operand_inmmediates"][-1])
    return None


def load_kernels(extra_info_dir):
    """funcid -> {pc: inst_dict} from enhanced_execution_info.json."""
    path = os.path.join(extra_info_dir, "enhanced_execution_info.json")
    with open(path) as fh:
        data = json.load(fh)
    out = {}
    names = {}
    for e in data["kernels"]:
        fid = int(e["unique_function_id"])
        out[fid] = {int(i["pc_num_dec"]): i for i in e["instructions"]}
        names[fid] = e["kernel_name"]
    return out, names


def load_sync_sites(extra_info_dir):
    """(funcid, pc) -> {sync_kind, opcode, barrier_operand_text, ...}."""
    path = os.path.join(extra_info_dir, "sync_operand_resolver.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        res = json.load(fh)["resolver"]
    out = {}
    for r in res:
        out[(int(r["unique_function_id"]), int(r["pc"]))] = r
    return out


def detect_regions(fid, byp, sync_sites):
    """Static detection of spin-wait regions for one kernel (funcid)."""
    order = sorted(byp)
    regions = []
    for pc in order:
        inst = byp[pc]
        if inst["op_code"].split(".")[0] != "BRA":
            continue
        tgt = bra_target(inst)
        if tgt is None or tgt >= pc:          # backward branch only
            continue
        if pc - tgt > MAX_LOOP_SPAN:          # short loop only
            continue
        body_pcs = [p for p in order if tgt <= p < pc]  # excl. loopback BRA
        # a spin region must contain a mbarrier wait
        wait_pcs = [p for p in body_pcs
                    if sync_sites.get((fid, p), {}).get("sync_kind") in WAIT_KINDS]
        if not wait_pcs:
            continue
        # classify body
        body = [(p, byp[p]["op_code"]) for p in body_pcs]
        sem_ops = [(hex(p), o) for p, o in body if is_semantic(o)]
        local_ops = [(hex(p), o) for p, o in body if is_local_mem(o)]
        # the primary wait is the last PHASECHK (non-TRYWAIT) if present, else the
        # last TRYWAIT — that is the poll whose result the loop branch depends on.
        phasechk = [p for p in wait_pcs
                    if sync_sites[(fid, p)]["sync_kind"] == "PHASECHK"]
        primary_wait = (phasechk[-1] if phasechk else wait_pcs[-1])
        wsite = sync_sites[(fid, primary_wait)]
        skip_pcs = body_pcs + [pc]
        self_tied, bar_reason = compute_barrier_self_tied(skip_pcs, byp)
        region = {
            "region_id": "k_fid%d_h0x%x" % (fid, tgt),
            "loop_head_pc": tgt,
            "body_pcs": body_pcs,
            "wait_pcs": wait_pcs,
            "primary_wait_pc": primary_wait,
            "primary_wait_kind": wsite["sync_kind"],
            "primary_wait_opcode": wsite["opcode"],
            "barrier_operand_text": wsite.get("barrier_operand_text"),
            "barrier_operand_index": wsite.get("barrier_operand_index"),
            "loopback_bra_pc": pc,
            "exit_pc": pc + ISIZE,   # fall-through of the (predicated) loopback BRA
            "semantic_free_body": len(sem_ops) == 0,
            "semantic_ops_in_body": sem_ops,
            "local_mem_ops_in_body": local_ops,
            "barrier_self_tied": self_tied,
            "barrier_reason": bar_reason,
        }
        control_safe, control_reason, control_proof = (
            prove_poll_loop_control(region, byp))
        wait_safe, wait_reason, wait_proof = prove_poll_wait_condition(
            region, byp)
        region["control_safe"] = control_safe
        region["control_reason"] = control_reason
        region["control_proof"] = control_proof
        region["wait_condition_safe"] = wait_safe
        region["wait_condition_reason"] = wait_reason
        region["wait_condition_proof"] = wait_proof
        regions.append(region)
    return regions


# --- Stage 1.7: compound wait-block CFG detector -------------------------- #
# A compound wait block is a speculative mbarrier gate whose miss path is a pure
# wait arm. The detector handles two local CFG shapes with one general
# rule (plan §1.7.6): follow only branches that consume the gate predicate P.
#
#   Type-1 (double gate, out-of-line fallback; HW captured ready only):
#     entry:      TRYWAIT P
#     entry+0x10: @P  BRA ready_arm           (P ready  -> continuation)
#     entry+0x20: TRYWAIT P
#     entry+0x30: @!P BRA fallback_entry      (P miss   -> OOL fallback loop)
#     ... fallback loop ... BRA resume_pc     (rejoin == ready_arm)
#
#   Type-2 (single gate, in-line fallback == an existing spin loop; HW captured
#           both ready and the loop):
#     entry:      TRYWAIT P
#     ... 0..N pure address-setup insts (MAINLINE; never skipped) ...
#     @!P BRA fallback_entry                  (P miss   -> in-line spin loop)
#     resume_pc == fall-through of that @!P BRA
#     ... spin loop ... BRA resume_pc         (rejoin)
#
# Key invariant (skip safety): only branches that CONSUME P delimit the block.
# A branch on a DIFFERENT predicate, or a non-branch instruction that produces a
# live-out (address setup), is OUTSIDE the block and is never added to
# allowed_block_pcs. resume_pc is the single convergence point of the P-ready and
# P-miss edges. The only skippable PCs are gate polls and the fallback wait loop
# body (TRYWAIT/PHASECHK/NANOSLEEP/loopback BRA) — never mainline setup.
#
# Fails closed: any shape it cannot fully resolve is a rejected candidate with a
# closed-taxonomy reason (§1.7.8) and is never emitted.
#
# FAIL reason taxonomy (see plan §1.7.8):
CFG_UNRESOLVED = "CFG_UNRESOLVED"
MULTI_REJOIN = "MULTI_REJOIN"


def _is_wait(fid, pc, byp, sync_sites):
    site = sync_sites.get((fid, pc))
    if site and site.get("sync_kind") in WAIT_KINDS:
        return True
    # sync_operand_resolver may not carry every predicated PHASECHK; fall back to
    # the disassembled opcode so the CFG walk still classifies it as a wait.
    inst = byp.get(pc)
    if inst is None:
        return False
    return any(k in inst["op_code"] for k in WAIT_KINDS)


def _pred_of(inst):
    """Return (reg, negate, is_uniform) for a predicated inst, else None."""
    if not inst.get("is_predicated"):
        return None
    return (inst.get("predicate_register"),
            bool(inst.get("is_predicate_negate")),
            bool(inst.get("is_uniform_predicate")))


def _scan_fallback_loop(fid, fb_entry, byp, sync_sites):
    """Walk a pure wait fallback loop starting at fb_entry: optional pure setup,
    a wait poll loop with a single backedge, then one unconditional BRA to the
    resume PC. Used by both Type-1 (OOL) and Type-2 (in-line) blocks.

    Returns (ok, info_or_reason). info = {fallback_body_pcs, fallback_wait_pcs,
    fallback_loopback_pc, resume_pc}."""
    body_pcs = []
    wait_pcs = []
    loopback_pc = None
    resume_pc = None
    fallback_exit_pc = None
    pc = fb_entry
    guard = 0
    while pc in byp and guard < (MAX_LOOP_SPAN // ISIZE) * 3:
        guard += 1
        inst = byp[pc]
        base = inst["op_code"].split(".")[0]
        if base == "BRA":
            tgt = bra_target(inst)
            if tgt is None:
                return False, CFG_UNRESOLVED
            pr = _pred_of(inst)
            if pr is not None:
                # predicated backedge of the poll loop (targets the loop head,
                # which is at or before this branch).
                if tgt > pc:
                    return False, CFG_UNRESOLVED
                if loopback_pc is not None:
                    return False, CFG_UNRESOLVED
                loopback_pc = pc
                body_pcs.append(pc)
                pc += ISIZE
                continue
            # unpredicated BRA => the single fallback exit / rejoin. It may be a
            # backward branch (OOL fallback sits above the continuation). This
            # exit BRA is the last instruction of the skipped fallback path, so
            # include its PC in the body (it is folded away with the loop).
            resume_pc = tgt
            body_pcs.append(pc)
            fallback_exit_pc = pc
            break
        if base in ("CALL", "RET", "EXIT", "BRX", "JMP", "JMX", "CALL.REL",
                    "BSSY", "BSYNC", "RTT", "WARPSYNC"):
            return False, CFG_UNRESOLVED
        if _is_wait(fid, pc, byp, sync_sites):
            wait_pcs.append(pc)
        body_pcs.append(pc)
        pc += ISIZE
    if loopback_pc is None or resume_pc is None or not wait_pcs:
        return False, CFG_UNRESOLVED
    return True, {
        "fallback_body_pcs": body_pcs,
        "fallback_wait_pcs": wait_pcs,
        "fallback_loopback_pc": loopback_pc,
        "fallback_exit_pc": fallback_exit_pc,
        "resume_pc": resume_pc,
    }



def _find_pred_branch(fid, entry, predreg, byp, sync_sites, max_span=0x120):
    """From entry (a TRYWAIT defining predreg), scan forward for the FIRST direct
    BRA that consumes predreg. Between entry and that branch only wait polls and
    non-control, non-semantic instructions (pure setup) may appear. Those setup
    PCs are MAINLINE and are returned separately so the caller never skips them.

    Returns (branch_pc, between_pcs) or (None, reason)."""
    between = []
    pc = entry + ISIZE
    span = 0
    while pc in byp and span < max_span:
        span += ISIZE
        inst = byp[pc]
        base = inst["op_code"].split(".")[0]
        pr = _pred_of(inst)
        if base == "BRA":
            if pr is not None and pr[0] == predreg:
                return pc, between            # the P-consuming gate branch
            # a branch that does not consume P ends the search unresolved: the
            # block boundary must be a P-branch.
            return None, CFG_UNRESOLVED
        if base in ("CALL", "RET", "EXIT", "BRX", "JMP", "JMX", "CALL.REL",
                    "RTT"):
            return None, CFG_UNRESOLVED
        # A semantic/async op between the gate and its branch means this is not a
        # clean speculative gate (fail closed).
        if is_semantic(inst["op_code"]) or is_async_producer(inst["op_code"]):
            return None, CFG_UNRESOLVED
        # wait polls and pure setup are allowed here; setup is mainline (not skipped)
        between.append(pc)
        pc += ISIZE
    return None, CFG_UNRESOLVED


def detect_wait_block_candidates(fid, byp, sync_sites):
    """Discover compound wait-block candidates for one kernel with NO application/
    kernel/PC filters, using ONE general rule for Type-1 and Type-2 (plan §1.7.6).
    Returns a list of dicts, each either a proven-shape record (cfg_ok=True) or a
    rejected candidate (cfg_ok=False, reason=...)."""
    order = sorted(byp)
    cands = []
    claimed = set()   # PCs already inside an emitted block (avoid nested re-entry)
    for pc in order:
        if pc in claimed:
            continue
        if not _is_wait(fid, pc, byp, sync_sites):
            continue
        # entry TRYWAIT must define a predicate (operand[0] = P<n>); use the first
        # P-consuming branch to find the gate boundary.
        ops = [o.get("operand_string", "") for o in byp[pc].get("operands", [])]
        predreg = None
        if ops and ops[0].lstrip("U").startswith("P"):
            try:
                predreg = int(ops[0].lstrip("U")[1:])
            except ValueError:
                predreg = None
        if predreg is None:
            continue
        br_pc, between = _find_pred_branch(fid, pc, predreg, byp, sync_sites)
        if br_pc is None:
            continue   # not a gate shape; leave for legacy spin-region detection
        br = byp[br_pc]
        prb = _pred_of(br)
        tgt = bra_target(br)
        if tgt is None:
            continue
        cand = {
            "block_id": "wb_fid%d_e0x%x" % (fid, pc),
            "unique_function_id": fid,
            "entry_wait_pc": pc,
            "entry_branch_pc": br_pc,
            "wait_predicate": predreg,
            "wait_predicate_uniform": bool(prb[2]),
            "mainline_setup_pcs": [p for p in between if not _is_wait(fid, p, byp, sync_sites)],
        }
        gate_wait_pcs = [pc] + [p for p in between if _is_wait(fid, p, byp, sync_sites)]

        if prb[1] is False:
            # @P BRA ready  => Type-1: ready arm is the branch target; the miss
            # path (@!P) is the NEXT gate/branch. Require the classic 2nd-gate
            # form: [2nd wait] then `@!P BRA fallback`.
            ready_arm = tgt
            w2 = byp.get(br_pc + ISIZE)
            b2 = byp.get(br_pc + 2 * ISIZE)
            if (w2 is None or b2 is None
                    or not _is_wait(fid, br_pc + ISIZE, byp, sync_sites)
                    or b2["op_code"].split(".")[0] != "BRA"):
                cand["cfg_ok"] = False
                cand["reason"] = CFG_UNRESOLVED
                cands.append(cand)
                continue
            pr2 = _pred_of(b2)
            if pr2 is None or not pr2[1] or pr2[0] != predreg:
                cand["cfg_ok"] = False
                cand["reason"] = CFG_UNRESOLVED
                cands.append(cand)
                continue
            fb_entry = bra_target(b2)
            gate_wait_pcs.append(br_pc + ISIZE)
            gate_poll_pcs = [pc, br_pc, br_pc + ISIZE, br_pc + 2 * ISIZE]
            not_ready_branch_pc = br_pc + 2 * ISIZE
            resume_expected = ready_arm
        else:
            # @!P BRA fallback  => Type-2: miss goes to the fallback loop; ready
            # falls through. resume is the fall-through PC of this branch.
            fb_entry = tgt
            resume_expected = br_pc + ISIZE
            gate_poll_pcs = [pc, br_pc]
            not_ready_branch_pc = br_pc

        if fb_entry is None:
            cand["cfg_ok"] = False
            cand["reason"] = CFG_UNRESOLVED
            cands.append(cand)
            continue
        ok, info = _scan_fallback_loop(fid, fb_entry, byp, sync_sites)
        if not ok:
            cand["cfg_ok"] = False
            cand["reason"] = info
            cands.append(cand)
            continue
        if info["resume_pc"] != resume_expected:
            cand["cfg_ok"] = False
            cand["reason"] = MULTI_REJOIN
            cands.append(cand)
            continue
        cand["cfg_ok"] = True
        cand["all_gate_wait_pcs"] = gate_wait_pcs
        cand["not_ready_branch_pc"] = not_ready_branch_pc
        cand["entry_arming_pcs"] = sorted(
            (set(gate_poll_pcs) | set(cand["mainline_setup_pcs"])) - {pc}
        )
        cand["captured_next_pc"] = resume_expected
        cand["fallback_entry_pc"] = fb_entry
        cand["fallback_body_pcs"] = info["fallback_body_pcs"]
        cand["fallback_wait_pcs"] = info["fallback_wait_pcs"]
        cand["fallback_loopback_pc"] = info["fallback_loopback_pc"]
        cand["fallback_exit_pc"] = info["fallback_exit_pc"]
        cand["resume_pc"] = info["resume_pc"]
        # allowed_block_pcs = ONLY gate polls/branches + fallback loop body.
        # Mainline setup between the gate and its branch is deliberately excluded.
        allowed = set(gate_poll_pcs)
        allowed.update(info["fallback_body_pcs"])
        cand["allowed_block_pcs"] = sorted(allowed)
        for p in allowed:
            claimed.add(p)
        cands.append(cand)
    return cands


# --- Stage 1.7: per-candidate provenance / effect / liveness proofs ------- #
# enhanced_execution_info.json gives aggregate register_usage counts only (no
# per-instruction def/use register lists) and operands as strings, so we derive
# def/use from operand_strings with an explicit, reviewed per-opcode model. Any
# opcode we cannot decode exactly fails closed (DEFUSE_UNDECODED).
#
# FAIL reason taxonomy (see plan §1.7.8), continued:
WAIT_COND_MISMATCH = "WAIT_COND_MISMATCH"
PROVENANCE_UNKNOWN = "PROVENANCE_UNKNOWN"
EFFECT_UNSAFE = "EFFECT_UNSAFE"
PREDICATE_LIVE_OUT = "PREDICATE_LIVE_OUT"
DEFUSE_UNDECODED = "DEFUSE_UNDECODED"

import re as _re
_REG_RE = _re.compile(r"\b(U?R\d+|U?P\d+)\b")
_BARR_OFF_RE = _re.compile(r"\+\s*(0x[0-9a-fA-F]+)\s*\]")

# Opcodes allowed inside a compound wait block's alternate path. Each maps to a
# def/use extraction rule keyed on operand position. This is intentionally a
# closed list: an opcode not here yields DEFUSE_UNDECODED and rejects the block.
#   dst_first  : operand[0] is the (only) destination register; rest are uses.
#   wait       : SYNCS.PHASECHK* — destination is the predicate operand[0]; the
#                barrier operand and wait-state operand are uses.
#   nanosleep  : NANOSLEEP.SYNCS — backoff, immediate only, no reg def/use.
#   bra        : control flow, no data def/use.
_WB_OPCODE_MODEL = {
    "IMAD.U32": "dst_first",
    "IMAD.MOV.U32": "dst_first",
    "SYNCS.PHASECHK.TRANS64.TRYWAIT": "wait",
    "SYNCS.PHASECHK.TRANS64": "wait",
    "NANOSLEEP.SYNCS": "nanosleep",
    "BRA": "bra",
}


def _regs_in(text):
    return _REG_RE.findall(text or "")


def _decode_defuse(inst):
    """Return (defs:set, uses:set, ok:bool) for a wait-block body instruction."""
    op = inst["op_code"]
    model = _WB_OPCODE_MODEL.get(op)
    ops = [o.get("operand_string", "") for o in inst.get("operands", [])]
    defs, uses = set(), set()
    if model == "dst_first":
        if ops:
            d = _regs_in(ops[0])
            defs.update(d)
            for o in ops[1:]:
                uses.update(_regs_in(o))
        # a predicate guard on the instruction is an additional use
        if inst.get("is_predicated"):
            uses.add(("UP" if inst.get("is_uniform_predicate") else "P")
                     + str(inst.get("predicate_register")))
        return defs, uses, True
    if model == "wait":
        # operand[0] = predicate destination; operand[1] = barrier; [2] = state.
        if ops:
            defs.update(_regs_in(ops[0]))
            for o in ops[1:]:
                uses.update(_regs_in(o))
        if inst.get("is_predicated"):
            uses.add(("UP" if inst.get("is_uniform_predicate") else "P")
                     + str(inst.get("predicate_register")))
        return defs, uses, True
    if model in ("nanosleep", "bra"):
        if inst.get("is_predicated"):
            uses.add(("UP" if inst.get("is_uniform_predicate") else "P")
                     + str(inst.get("predicate_register")))
        return defs, uses, True
    return defs, uses, False


def _reviewed_idempotent_copy(inst):
    """Return proof detail only for a loop-invariant IMAD copy."""
    if inst.get("op_code") not in ("IMAD.U32", "IMAD.MOV.U32"):
        return None
    operands = [
        _normalized_operand(o.get("operand_string", ""))
        for o in inst.get("operands", [])
    ]
    if len(operands) != 4:
        return None
    destination, multiplicand_a, multiplicand_b, source = operands
    if (multiplicand_a not in ("RZ", "URZ") or
            multiplicand_b not in ("RZ", "URZ")):
        return None
    if len(_regs_in(destination)) != 1 or len(_regs_in(source)) != 1:
        return None
    return {
        "destination": destination,
        "source": source,
        "reason": "zero-product register copy is idempotent",
    }


def _barrier_offset(text):
    """Return the immediate byte offset of a barrier operand `[base(+idx)(+0xNN)]`.
    A barrier with no `+0x..` immediate (e.g. `[R7+URZ]`) is base+0, so return
    '0x0'. Returns None only if the operand is not a memory-form barrier."""
    text = text or ""
    if "[" not in text:
        return None
    m = _BARR_OFF_RE.search(text)
    return m.group(1).lower() if m else "0x0"


def prove_wait_condition(fid, cand, byp, sync_sites):
    """§1.7.2 — all gate + fallback waits address the SAME barrier offset and use a
    wait-state with consistent provenance. Base register may differ (UR9 vs R11 via
    a reviewed copy), so we compare the immediate offset and require the wait-state
    to be a single reviewed source. Returns (ok, reason|None, canonical)."""
    wait_pcs = list(cand["all_gate_wait_pcs"]) + list(cand["fallback_wait_pcs"])
    offs = set()
    states = set()
    for wp in wait_pcs:
        inst = byp.get(wp)
        if inst is None:
            return False, PROVENANCE_UNKNOWN, None
        ops = [o.get("operand_string", "") for o in inst.get("operands", [])]
        if len(ops) < 3:
            return False, PROVENANCE_UNKNOWN, None
        off = _barrier_offset(ops[1])
        if off is None:
            return False, PROVENANCE_UNKNOWN, None
        offs.add(off)
        states.add(ops[2])            # wait-state operand text (e.g. R11 / R15)
    if len(offs) != 1:
        return False, WAIT_COND_MISMATCH, None
    canonical = {
        "barrier_offset": next(iter(offs)),
        "wait_state_operands": sorted(states),
    }
    # A kernel may reload the state into different registers per site
    # (R11 at the gate, R15 in the fallback) but all originate from the same
    # UR7-derived phase value. We cannot chase full dataflow from aggregate usage
    # info, so we require every wait-state operand to be a single reviewed source:
    # one general/uniform register, or the zero register RZ/URZ (a literal
    # phase-0 wait). Record them for review rather than guessing equivalence.
    def _is_single_state_src(s):
        s = s.strip()
        if s in ("RZ", "URZ"):
            return True
        return len(_regs_in(s)) == 1
    for s in states:
        if not _is_single_state_src(s):
            return False, PROVENANCE_UNKNOWN, None
    return True, None, canonical


def prove_effects(fid, cand, byp):
    """§1.7.3 — the alternate path contains only reviewed wait/backoff/pure-ALU/
    control opcodes; no memory/TMA/GMMA/MUFU/atomic/ARRIVE/EXCH/BAR/fence and no
    unknown opcode. Returns (ok, reason|None, effect_classes)."""
    classes = Counter()
    per_pc = []
    loopback_pc = cand.get("loopback_bra_pc")
    for pc in cand["allowed_block_pcs"]:
        inst = byp[pc]
        op = inst["op_code"]
        # Local memory (STL/LDL) is a real side effect; skipping the loop would
        # drop or reorder a local store/load. Classify as unsafe (not just
        # undecoded) so the report reason is precise.
        if is_local_mem(op):
            return False, EFFECT_UNSAFE, None
        if is_semantic(op) or is_async_producer(op):
            return False, EFFECT_UNSAFE, None
        model = _WB_OPCODE_MODEL.get(op)
        if model is None:
            return False, DEFUSE_UNDECODED, None
        detail = None
        if model == "dst_first":
            detail = _reviewed_idempotent_copy(inst)
            if detail is None:
                return False, EFFECT_UNSAFE, None
            effect = "IDEMPOTENT_COPY"
        elif model == "wait":
            effect = "WAIT_OBSERVATION"
        elif model == "nanosleep":
            effect = "DELAY_ONLY"
        else:
            if loopback_pc is not None and pc != loopback_pc:
                return False, EFFECT_UNSAFE, None
            effect = "LOOPBACK_CONTROL"
        classes[model] += 1
        per_pc.append({
            "pc": pc,
            "opcode": op,
            "effect": effect,
            "detail": detail,
        })
    cand["effect_proof"] = per_pc
    return True, None, dict(classes)


def prove_predicate_dead_at_resume(fid, cand, byp):
    """§1.7.3 predicate safety, reconciled with §1.8.2 runtime behavior.

    The wait predicate P is set by the entry gate TRYWAIT. §1.8.2 fixes that the
    entry gate TRYWAIT is ALWAYS executed (never rolled back or skipped); only the
    residual fallback poll loop is skipped. Therefore P holds the correct
    gate-evaluated value at resume_pc whether or not P is redefined afterwards, so
    a live-out P is safe here — folding the fallback loop does not change what set
    P. We still classify the post-resume disposition for diagnostics:
      * redefined-before-read  -> disposition="dead"
      * read-before-redefine   -> disposition="live_out_gate_set" (safe: gate ran)
      * neither within window   -> disposition="unused"
    Returns (ok, reason|None, info) where info={next_predicate_def_pc, disposition,
    first_read_pc}. This proof no longer rejects a live-out predicate; a genuine
    unsafe case (a predicate produced only inside the SKIPPED fallback and needed
    downstream) cannot occur because the fallback body defines only its own loop
    predicate, which is dead once the loop exits."""
    predreg = ("UP" if cand.get("wait_predicate_uniform") else "P") + \
        str(cand["wait_predicate"])
    order = sorted(p for p in byp if p >= cand["resume_pc"])
    guard = 0
    for pc in order:
        guard += 1
        if guard > 256:
            break
        inst = byp[pc]
        defs, uses = _decode_defuse_generic(inst)
        if predreg in defs:
            return True, None, {"next_predicate_def_pc": pc,
                                "disposition": "dead", "first_read_pc": None}
        if predreg in uses:
            # Read before redefinition. Safe because the gate that set P is not
            # skipped (§1.8.2). Record it for review.
            return True, None, {"next_predicate_def_pc": None,
                                "disposition": "live_out_gate_set",
                                "first_read_pc": pc}
    return True, None, {"next_predicate_def_pc": None,
                        "disposition": "unused", "first_read_pc": None}


def _decode_defuse_generic(inst):
    """Best-effort def/use for opcodes OUTSIDE the wait-block model, used only for
    the post-resume predicate liveness scan. Recognizes the destination-predicate
    common forms used to redefine wait predicates (ISETP/FSETP/PLOP3/etc.) and any
    'PT'/'P<n>' written as operand[0]. Uses operand_string tokens; unknown shapes
    contribute no defs (so liveness stays conservative: a real later use will still
    be seen)."""
    ops = [o.get("operand_string", "") for o in inst.get("operands", [])]
    defs, uses = set(), set()
    op = inst["op_code"]
    # SETP-family and predicate-logic write their predicate destination first.
    if any(op.startswith(pfx) for pfx in ("ISETP", "FSETP", "DSETP", "HSETP",
                                          "PLOP3", "PSETP", "P2R", "VSETP",
                                          "UISETP", "UFSETP", "UPLOP3", "UPSETP")):
        # destinations are the leading predicate operands (may be "P0, P1")
        lead = []
        for o in ops:
            r = _regs_in(o)
            if r and r[0].lstrip("U").startswith("P"):
                lead.append(r[0])
            else:
                break
        defs.update(lead)
        for o in ops[len(lead):]:
            uses.update(_regs_in(o))
    else:
        # generic: operand[0] is destination if it is a register; rest are uses.
        if ops:
            defs.update(_regs_in(ops[0]))
            for o in ops[1:]:
                uses.update(_regs_in(o))
    if inst.get("is_predicated"):
        uses.add(("UP" if inst.get("is_uniform_predicate") else "P")
                 + str(inst.get("predicate_register")))
    return defs, uses


def prove_candidate(fid, cand, byp, sync_sites):
    """Run all static proofs for one CFG-proven candidate. Mutates cand with proof
    fields and returns (ok, reason|None)."""
    ok, reason, canonical = prove_wait_condition(fid, cand, byp, sync_sites)
    if not ok:
        return False, reason
    cand["canonical_wait_condition"] = canonical
    ok, reason, classes = prove_effects(fid, cand, byp)
    if not ok:
        return False, reason
    cand["fallback_effect_safe"] = True
    cand["effect_classes"] = classes
    ok, reason, pinfo = prove_predicate_dead_at_resume(fid, cand, byp)
    if not ok:
        return False, reason
    cand["wait_predicate_disposition"] = pinfo["disposition"]
    cand["next_predicate_def_pc"] = pinfo["next_predicate_def_pc"]
    cand["wait_predicate_first_read_pc"] = pinfo["first_read_pc"]
    return True, None


def make_inject_missing_wait_record(kid, c, sync_sites, byp):
    """Serialize a ready-only gate for the entry-only missing-wait strategy."""
    st = c.get("dynamic", {})
    partial = c.get("partial_join_proof")
    site = sync_sites.get((c["unique_function_id"], c["entry_wait_pc"]), {})
    operands = [
        o.get("operand_string", "")
        for o in byp[c["entry_wait_pc"]].get("operands", [])
    ]
    record = {
        "block_id": c["block_id"],
        "runtime_strategy": "INJECT_MISSING_WAIT",
        "unique_function_id": c["unique_function_id"],
        "entry_probe_pc": c["entry_wait_pc"],
        "entry_probe_pc_hex": "0x%x" % c["entry_wait_pc"],
        "entry_first_branch_pc": c["entry_branch_pc"],
        "entry_branch_pc": c["not_ready_branch_pc"],
        "captured_ready_pc": c["resume_pc"],
        "entry_setup_pcs": c.get("mainline_setup_pcs", []),
        "entry_arming_pcs": c.get("entry_arming_pcs", []),
        "entry_branch_not_ready_polarity": "PREDICATE_FALSE",
        "miss_arm_entry_pc": c["fallback_entry_pc"],
        "miss_arm_wait_pcs": c["fallback_wait_pcs"],
        "miss_arm_loopback_pc": c["fallback_loopback_pc"],
        "miss_arm_exit_pc": c["fallback_exit_pc"],
        "entry_probe_kind": site.get("sync_kind"),
        "entry_probe_opcode": site.get("opcode"),
        "barrier_operand_text": site.get("barrier_operand_text"),
        "wait_state_operand_text": operands[2] if len(operands) > 2 else None,
        "canonical_wait_condition": c.get("canonical_wait_condition"),
        "partial_modeled_miss_action": (
            "DEFER_BLOCKED_SUFFIX_THEN_HOLD"
            if partial is not None else "ABORT"
        ),
        "observed_path_counts": {
            "ready_arm": st.get("path_A_first_gate_ready", 0),
            "second_gate_arm": st.get("path_B_second_gate_ready", 0),
            "captured_poll": st.get("path_C_fallback_loop", 0),
        },
        "active_entry_visits": st.get("active_entry_visits", 0),
        "predicated_off_entry_occurrences": st.get(
            "predicated_off_entry_occurrences", 0),
        "all_live_entry_visits": st.get("all_live_entry_visits", 0),
        "partial_entry_visits": st.get("partial_entry_visits", 0),
        "observed_effective_masks": st.get("entry_effective_masks", {}),
        "observed_live_masks": st.get("entry_live_masks", {}),
        "observed_barrier_addrs": st.get("entry_barrier_addrs", {}),
        "observed_wait_states": st.get("entry_wait_states", {}),
        "sync_payload_missing": st.get("entry_sync_payload_missing", 0),
    }
    if partial is not None:
        region = partial["region"]
        record["partial_hold_proof_version"] = 1
        record["partial_region_id"] = "pr_k%d_fid%d_b0x%x_j0x%x" % (
            kid,
            c["unique_function_id"],
            region["sibling_branch_pc"],
            region["join_pc"],
        )
        record["sibling_branch_pc"] = region["sibling_branch_pc"]
        record["sibling_branch_target_pc"] = \
            region["sibling_branch_target_pc"]
        record["join_pc"] = region["join_pc"]
        record["post_join_pc"] = region["post_join_pc"]
        record["join_kind"] = region["join_kind"]
        record["join_bssy_pc"] = region["bssy_pc"]
        record["join_token"] = region["join_token"]
        record["observed_sibling_masks"] = partial["sibling_masks"]
        record["partial_join_pending_visits"] = \
            partial["join_pending_visits"]
        record["partial_join_already_issued_visits"] = \
            partial["join_already_issued_visits"]
        record["complementary_masks_complete"] = True
        record["sibling_join_only"] = True
        record["blocked_path_reaches_same_join"] = True
        record["linear_capsule_safe"] = True
    return record


def make_metadata_only_record(c):
    """Serialize a proven WaitBlock that owns no v5 runtime action."""
    st = c.get("dynamic", {})
    return {
        "block_id": c["block_id"],
        "record_class": "METADATA_ONLY",
        "unique_function_id": c["unique_function_id"],
        "entry_probe_pc": c["entry_wait_pc"],
        "entry_probe_pc_hex": "0x%x" % c["entry_wait_pc"],
        "entry_branch_pc": c["entry_branch_pc"],
        "captured_ready_pc": c["resume_pc"],
        "observed_path_counts": {
            "ready_arm": st.get("path_A_first_gate_ready", 0),
            "second_gate_arm": st.get("path_B_second_gate_ready", 0),
            "captured_poll": st.get("path_C_fallback_loop", 0),
        },
        "active_entry_visits": st.get("active_entry_visits", 0),
        "all_live_entry_visits": st.get("all_live_entry_visits", 0),
        "partial_entry_visits": st.get("partial_entry_visits", 0),
    }


def make_fold_captured_poll_record(kid, fid, region):
    """Serialize an existing foldable spin region without changing its owner."""
    return {
        "block_id": "wb_k%d_fid%d_p0x%x" % (
            kid, fid, region["primary_wait_pc"]),
        "runtime_strategy": "FOLD_CAPTURED_POLL",
        "fold_proof_version": 1,
        "unique_function_id": fid,
        "poll_loop_head_pc": region["loop_head_pc"],
        "poll_loop_body_pcs": region["body_pcs"],
        "poll_wait_pcs": region["wait_pcs"],
        "poll_probe_pc": region["primary_wait_pc"],
        "poll_probe_kind": region["primary_wait_kind"],
        "poll_probe_opcode": region["primary_wait_opcode"],
        "barrier_operand_text": region.get("barrier_operand_text"),
        "poll_loopback_pc": region["loopback_bra_pc"],
        "poll_loop_exit_pc": region["exit_pc"],
        "poll_advance_allowed_pcs": sorted(
            set(region["body_pcs"]) | {region["loopback_bra_pc"]}),
        "control_safe": region["control_safe"],
        "control_proof": region["control_proof"],
        "wait_condition_safe": region["wait_condition_safe"],
        "wait_condition_proof": region["wait_condition_proof"],
        "semantic_free_body": region["semantic_free_body"],
        "effect_safe": region["effect_safe"],
        "effect_classes": region["effect_classes"],
        "effect_proof": region["effect_proof"],
        "barrier_self_tied": region["barrier_self_tied"],
        "barrier_reason": region["barrier_reason"],
        "scoreboard_proof": region["barrier_reason"],
        "dynamic": region["dynamic"],
    }




def kernel_dir_map(traces_dir):
    base = os.path.join(traces_dir, "threadblocks", "device_0", "stream_0")
    out = {}
    if not os.path.isdir(base):
        return out
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if os.path.isdir(full) and d.startswith("kernel_"):
            try:
                out[int(d.split("_", 1)[1])] = full
            except ValueError:
                pass
    return out


def validate_dynamic(kdir, fid, regions, max_ctas):
    """Confirm each region's anchors appear in the recorded producer streams and
    the body actually repeats (spins). Active poll probes must carry a runtime
    sync payload; predicated-off probes are architectural no-ops and may omit it.
    Returns per-region dynamic stats."""
    files = sorted(p for p in glob.glob(os.path.join(kdir, "*.pb")))
    if max_ctas != "all":
        files = files[:int(max_ctas)]
    loopback_pcs = {r["loopback_bra_pc"] for r in regions}
    regions_by_poll = {}
    regions_by_loop_head = defaultdict(list)
    regions_by_static_pc = defaultdict(list)
    for r in regions:
        regions_by_poll.setdefault(r["primary_wait_pc"], []).append(r)
        regions_by_loop_head[r["loop_head_pc"]].append(r)
        for pc in r["body_pcs"] + [r["loopback_bra_pc"]]:
            regions_by_static_pc[pc].append(r)
    stats = {r["region_id"]: {
        # a "visit" is one (cta, warp) that entered this region at least once;
        # iters is the number of loopback-BRA executions in that visit.
        "warp_visits": 0, "total_iters": 0, "max_iters": 0,
        "min_iters": None, "single_shot_visits": 0,
        "active_poll_visits": 0,
        "predicated_off_poll_occurrences": 0,
        "poll_sync_payload_missing": 0,
        "active_wait_occurrences": 0,
        "predicated_off_wait_occurrences": 0,
        "wait_sync_payload_missing": 0,
        "wait_condition_mismatches": 0,
        "wait_iterations_checked": 0,
        "active_wait_pcs": Counter()} for r in regions}
    for f in files:
        warps = vpb_parse_full(open(f, "rb").read())
        for wid, insts in warps:
            # count loopback occurrences per region -> iteration count in this warp
            per_region = Counter()
            open_iterations = {}
            for inst in insts:
                if inst["function_unique_id"] != fid:
                    continue
                p = inst["pc"]
                if p in loopback_pcs:
                    per_region[p] += 1
                for r in regions_by_loop_head.get(p, []):
                    open_iterations[r["region_id"]] = set()
                for r in regions_by_static_pc.get(p, []):
                    rid = r["region_id"]
                    if rid not in open_iterations:
                        continue
                    if p in r["wait_pcs"]:
                        s = stats[rid]
                        effective = (
                            int(inst["active_mask"]) &
                            int(inst["predicate_mask"])
                        )
                        if effective == 0:
                            s["predicated_off_wait_occurrences"] += 1
                        else:
                            s["active_wait_occurrences"] += 1
                            s["active_wait_pcs"]["0x%x" % p] += 1
                            if (not inst["sync_valid"]
                                    or inst["barrier_addr"] is None):
                                s["wait_sync_payload_missing"] += 1
                            else:
                                wait_state = (
                                    0 if inst["semantic_raw"] is None
                                    else int(inst["semantic_raw"]))
                                open_iterations[rid].add(
                                    (int(inst["barrier_addr"]), wait_state))
                    if p == r["loopback_bra_pc"]:
                        s = stats[rid]
                        s["wait_iterations_checked"] += 1
                        if len(open_iterations[rid]) > 1:
                            s["wait_condition_mismatches"] += 1
                        del open_iterations[rid]
                for r in regions_by_poll.get(p, []):
                    s = stats[r["region_id"]]
                    effective = (
                        int(inst["active_mask"]) &
                        int(inst["predicate_mask"])
                    )
                    if effective == 0:
                        s["predicated_off_poll_occurrences"] += 1
                    else:
                        s["active_poll_visits"] += 1
                        if (not inst["sync_valid"]
                                or inst["barrier_addr"] is None):
                            s["poll_sync_payload_missing"] += 1
            # attribute loopback pc -> region (one visit per (cta,warp) that entered)
            for r in regions:
                lb = r["loopback_bra_pc"]
                n = per_region.get(lb, 0)
                if n > 0:
                    s = stats[r["region_id"]]
                    s["warp_visits"] += 1
                    s["total_iters"] += n
                    s["max_iters"] = max(s["max_iters"], n)
                    s["min_iters"] = n if s["min_iters"] is None else min(s["min_iters"], n)
                    if n == 1:
                        s["single_shot_visits"] += 1
    # a region that only ever executes its loopback once is a single-shot wait,
    # not a spin loop: collapsing it is a no-op, so mark it so the simulator can
    # treat it as an ordinary (non-collapsing) wait if desired.
    for r in regions:
        s = stats[r["region_id"]]
        s["is_spin_loop"] = bool(s["warp_visits"]) and s["max_iters"] > 1
        s["active_poll_payload_complete"] = (
            s["active_poll_visits"] > 0 and
            s["poll_sync_payload_missing"] == 0
        )
        s["active_wait_payload_complete"] = (
            s["active_wait_occurrences"] > 0 and
            s["wait_sync_payload_missing"] == 0 and
            s["wait_condition_mismatches"] == 0
        )
        s["active_wait_pcs"] = dict(sorted(s["active_wait_pcs"].items()))
    return stats, len(files)


# --- Stage 1.7: dynamic capture proof (§1.7.4) ---------------------------- #
# FAIL reason taxonomy (see plan §1.7.8), continued:
DYNAMIC_PATH_ESCAPE = "DYNAMIC_PATH_ESCAPE"
CTA_IDENTITY_MISMATCH = "CTA_IDENTITY_MISMATCH"
ENTRY_WAIT_TARGET_UNDECODABLE = "ENTRY_WAIT_TARGET_UNDECODABLE"
POLL_WAIT_TARGET_UNDECODABLE = "POLL_WAIT_TARGET_UNDECODABLE"
POLL_CONTROL_UNPROVEN = "POLL_CONTROL_UNPROVEN"
POLL_WAIT_CONDITION_MISMATCH = "POLL_WAIT_CONDITION_MISMATCH"
POLL_EFFECT_UNSAFE = "POLL_EFFECT_UNSAFE"
POLL_SCOREBOARD_UNSAFE = "POLL_SCOREBOARD_UNSAFE"
CAPTURED_POLL_ENTRY_NOT_REQUIRED = "CAPTURED_POLL_ENTRY_NOT_REQUIRED"
PARTIAL_MASK_PARTITION_INVALID = "PARTIAL_MASK_PARTITION_INVALID"
PARTIAL_SIBLING_BRANCH_UNRESOLVED = "PARTIAL_SIBLING_BRANCH_UNRESOLVED"
PARTIAL_ENTRY_BRANCH_OCCURRENCE_MISSING = \
    "PARTIAL_ENTRY_BRANCH_OCCURRENCE_MISSING"
PARTIAL_JOIN_STATIC_UNSAFE = "PARTIAL_JOIN_STATIC_UNSAFE"
PARTIAL_JOIN_OCCURRENCE_MISSING = "PARTIAL_JOIN_OCCURRENCE_MISSING"
PARTIAL_SIBLING_NOT_QUIESCENT = "PARTIAL_SIBLING_NOT_QUIESCENT"
PARTIAL_CAPSULE_MASK_ESCAPE = "PARTIAL_CAPSULE_MASK_ESCAPE"
PARTIAL_CAPSULE_CONTROL_UNSAFE = "PARTIAL_CAPSULE_CONTROL_UNSAFE"
PARTIAL_POST_JOIN_UNRESOLVED = "PARTIAL_POST_JOIN_UNRESOLVED"
PARTIAL_REGION_INCONSISTENT = "PARTIAL_REGION_INCONSISTENT"


def _opcode_base(inst):
    return inst["op_code"].split(".")[0]


def _direct_target(inst):
    for operand in inst.get("operands", []):
        if operand.get("has_inmediate"):
            return int(operand["operand_inmmediates"][-1])
    return None


def _operand_texts(inst):
    return [
        _normalized_operand(operand.get("operand_string", ""))
        for operand in inst.get("operands", [])
    ]


def prove_partial_join_static(byp, sibling_branch_pc, join_pc, post_join_pc):
    """Prove the static branch/join anchors for one partial-lane region.

    WARPSYNC is self-describing. BSYNC additionally requires a preceding BSSY
    with the same token and an exact post-join target.
    """
    branch = byp.get(sibling_branch_pc)
    join = byp.get(join_pc)
    post_join = byp.get(post_join_pc)
    if (branch is None or _opcode_base(branch) != "BRA"
            or bra_target(branch) != join_pc
            or join is None or post_join is None):
        return False, PARTIAL_JOIN_STATIC_UNSAFE, None

    join_kind = _opcode_base(join)
    if (join_kind not in ("WARPSYNC", "BSYNC")
            or join.get("is_predicated")):
        return False, PARTIAL_JOIN_STATIC_UNSAFE, None

    proof = {
        "sibling_branch_pc": sibling_branch_pc,
        "sibling_branch_target_pc": join_pc,
        "join_pc": join_pc,
        "post_join_pc": post_join_pc,
        "join_kind": join_kind,
        "bssy_pc": None,
        "join_token": None,
    }
    if join_kind == "WARPSYNC":
        return True, None, proof

    join_operands = _operand_texts(join)
    if len(join_operands) != 1:
        return False, PARTIAL_JOIN_STATIC_UNSAFE, None
    token = join_operands[0]
    proof["join_token"] = token
    for pc in sorted((p for p in byp if p < sibling_branch_pc), reverse=True):
        inst = byp[pc]
        if _opcode_base(inst) != "BSSY":
            continue
        operands = _operand_texts(inst)
        if not operands or operands[0] != token:
            continue
        if _direct_target(inst) != post_join_pc:
            return False, PARTIAL_JOIN_STATIC_UNSAFE, None
        proof["bssy_pc"] = pc
        return True, None, proof
    return False, PARTIAL_JOIN_STATIC_UNSAFE, None


def prove_partial_join_visit(cand, insts, byp, expected_fid, entry_index,
                             live_before):
    """Prove one dynamic partial entry belongs to a join-only divergence region.

    The proof uses the raw active mask for lane-group ownership. Predicate masks
    still determine the partial entry itself, but a predicated-off instruction
    remains owned by the lane group that issued it.
    """
    n = len(insts)
    entry_inst = insts[entry_index]
    entry_mask = (
        int(entry_inst["active_mask"]) &
        int(entry_inst["predicate_mask"])
    )
    live_mask = int(live_before[entry_index])
    sibling_mask = live_mask & ~entry_mask
    if (entry_mask == 0 or sibling_mask == 0
            or entry_mask & sibling_mask
            or (entry_mask | sibling_mask) != live_mask):
        return False, PARTIAL_MASK_PARTITION_INVALID, None

    effective = [
        int(inst["active_mask"]) & int(inst["predicate_mask"])
        for inst in insts
    ]

    sibling_branch_index = None
    static_proof = None
    for idx in range(entry_index - 1, -1, -1):
        if insts[idx]["function_unique_id"] != expected_fid:
            continue
        if effective[idx] == live_mask:
            break
        if effective[idx] != sibling_mask:
            continue
        static_inst = byp.get(insts[idx]["pc"])
        if static_inst is None or _opcode_base(static_inst) != "BRA":
            continue
        target = bra_target(static_inst)
        target_inst = byp.get(target)
        if target is None or target_inst is None:
            continue
        if _opcode_base(target_inst) not in ("WARPSYNC", "BSYNC"):
            continue
        post_join_pc = target + ISIZE
        ok, _, proof = prove_partial_join_static(
            byp, insts[idx]["pc"], target, post_join_pc)
        if ok:
            sibling_branch_index = idx
            static_proof = proof
            break
    if sibling_branch_index is None:
        return False, PARTIAL_SIBLING_BRANCH_UNRESOLVED, None

    join_pc = static_proof["join_pc"]
    post_join_pc = static_proof["post_join_pc"]
    sibling_join_index = None
    blocked_join_index = None
    for idx in range(sibling_branch_index + 1, n):
        if insts[idx]["function_unique_id"] != expected_fid:
            continue
        if insts[idx]["pc"] != join_pc:
            continue
        owner = int(insts[idx]["active_mask"]) & live_mask
        if sibling_join_index is None and owner == sibling_mask:
            sibling_join_index = idx
        if idx >= entry_index and blocked_join_index is None and owner == entry_mask:
            blocked_join_index = idx
        if sibling_join_index is not None and blocked_join_index is not None:
            break
    if sibling_join_index is None or blocked_join_index is None:
        return False, PARTIAL_JOIN_OCCURRENCE_MISSING, None

    resume_index = None
    for idx in range(entry_index + 1, blocked_join_index + 1):
        if (insts[idx]["function_unique_id"] == expected_fid
                and insts[idx]["pc"] == cand["resume_pc"]
                and (int(insts[idx]["active_mask"]) & entry_mask)):
            resume_index = idx
            break
    if resume_index is None:
        return False, PARTIAL_JOIN_OCCURRENCE_MISSING, None
    entry_branch_index = None
    for idx in range(entry_index + 1, resume_index):
        if (insts[idx]["function_unique_id"] == expected_fid
                and insts[idx]["pc"] == cand["not_ready_branch_pc"]
                and (int(insts[idx]["active_mask"]) & live_mask) ==
                    entry_mask):
            entry_branch_index = idx
            break
    if entry_branch_index is None:
        return False, PARTIAL_ENTRY_BRANCH_OCCURRENCE_MISSING, None

    last_join_index = max(sibling_join_index, blocked_join_index)
    post_join_index = None
    for idx in range(last_join_index + 1, n):
        if insts[idx]["function_unique_id"] != expected_fid:
            continue
        if (insts[idx]["pc"] == post_join_pc
                and effective[idx] == live_mask):
            post_join_index = idx
            break
        if effective[idx] == live_mask:
            return False, PARTIAL_POST_JOIN_UNRESOLVED, None
    if post_join_index is None:
        return False, PARTIAL_POST_JOIN_UNRESOLVED, None

    for idx in range(sibling_branch_index + 1, post_join_index):
        inst = insts[idx]
        if inst["function_unique_id"] != expected_fid:
            return False, CTA_IDENTITY_MISMATCH, None
        static_inst = byp.get(inst["pc"])
        if static_inst is None:
            return False, PARTIAL_CAPSULE_CONTROL_UNSAFE, None
        if inst["pc"] == post_join_pc:
            return False, PARTIAL_POST_JOIN_UNRESOLVED, {
                "escape_pc": inst["pc"],
                "escape_index": idx,
            }
        if _opcode_base(static_inst) in (
                "CALL", "CALL.REL", "RET", "EXIT", "BRX", "JMP", "JMX",
                "RTT"):
            return False, PARTIAL_CAPSULE_CONTROL_UNSAFE, {
                "escape_pc": inst["pc"],
                "escape_index": idx,
            }
        owner = int(inst["active_mask"]) & live_mask
        if owner == 0:
            continue
        if idx == sibling_join_index:
            if owner != sibling_mask or inst["pc"] != join_pc:
                return False, PARTIAL_JOIN_OCCURRENCE_MISSING, None
            continue
        if owner & sibling_mask:
            return False, PARTIAL_SIBLING_NOT_QUIESCENT, {
                "escape_pc": inst["pc"],
                "escape_index": idx,
                "escape_mask": owner,
            }
        if owner & ~entry_mask:
            return False, PARTIAL_CAPSULE_MASK_ESCAPE, {
                "escape_pc": inst["pc"],
                "escape_index": idx,
                "escape_mask": owner,
            }

    return True, None, {
        "entry_mask": entry_mask,
        "sibling_mask": sibling_mask,
        "live_mask": live_mask,
        "sibling_branch_index": sibling_branch_index,
        "sibling_join_index": sibling_join_index,
        "blocked_join_index": blocked_join_index,
        "entry_branch_index": entry_branch_index,
        "resume_index": resume_index,
        "post_join_index": post_join_index,
        "join_pending_at_entry": sibling_join_index > entry_index,
        "static": static_proof,
    }


def validate_partial_join_dynamic(kdir, cands, byp, max_ctas, expected_fid):
    """Validate the partial deferred-suffix contract over every selected CTA."""
    files = sorted(glob.glob(os.path.join(kdir, "*.pb")))
    if max_ctas != "all":
        files = files[:int(max_ctas)]
    stats = {}
    for cand in cands:
        stats[cand["block_id"]] = {
            "dynamic_ok": True,
            "reason": None,
            "partial_visits": 0,
            "join_pending_visits": 0,
            "join_already_issued_visits": 0,
            "entry_masks": Counter(),
            "sibling_masks": Counter(),
            "live_masks": Counter(),
            "regions": {},
        }

    for path in files:
        warps = vpb_parse_full(open(path, "rb").read())
        for _, insts in warps:
            effective = [
                int(inst["active_mask"]) & int(inst["predicate_mask"])
                for inst in insts
            ]
            live = 0
            for inst in insts:
                live |= int(inst["active_mask"])
            live_before = []
            for idx, inst in enumerate(insts):
                live_before.append(live)
                static_inst = byp.get(inst["pc"])
                if (static_inst is not None
                        and _opcode_base(static_inst) == "EXIT"
                        and effective[idx]):
                    live &= ~effective[idx]

            for cand in cands:
                st = stats[cand["block_id"]]
                if not st["dynamic_ok"]:
                    continue
                for idx, inst in enumerate(insts):
                    if (inst["function_unique_id"] != expected_fid
                            or inst["pc"] != cand["entry_wait_pc"]
                            or effective[idx] == 0
                            or effective[idx] == live_before[idx]):
                        continue
                    ok, reason, proof = prove_partial_join_visit(
                        cand, insts, byp, expected_fid, idx, live_before)
                    if not ok:
                        st["dynamic_ok"] = False
                        st["reason"] = reason
                        if proof:
                            st["failure_detail"] = {
                                key: ("0x%x" % value if key.endswith("_pc")
                                      or key.endswith("_mask") else value)
                                for key, value in proof.items()
                            }
                        break
                    st["partial_visits"] += 1
                    if proof["join_pending_at_entry"]:
                        st["join_pending_visits"] += 1
                    else:
                        st["join_already_issued_visits"] += 1
                    for key, value in (
                            ("entry_masks", proof["entry_mask"]),
                            ("sibling_masks", proof["sibling_mask"]),
                            ("live_masks", proof["live_mask"])):
                        st[key]["0x%x" % value] += 1
                    static = proof["static"]
                    region_key = (
                        static["sibling_branch_pc"],
                        static["join_pc"],
                        static["post_join_pc"],
                        static["join_kind"],
                        static["bssy_pc"],
                        static["join_token"],
                    )
                    st["regions"].setdefault(region_key, 0)
                    st["regions"][region_key] += 1

    for cand in cands:
        st = stats[cand["block_id"]]
        expected_visits = cand["dynamic"]["partial_entry_visits"]
        if st["dynamic_ok"] and st["partial_visits"] != expected_visits:
            st["dynamic_ok"] = False
            st["reason"] = PARTIAL_REGION_INCONSISTENT
        if st["dynamic_ok"] and len(st["regions"]) != 1:
            st["dynamic_ok"] = False
            st["reason"] = PARTIAL_REGION_INCONSISTENT
        if st["dynamic_ok"]:
            region_key, region_visits = next(iter(st["regions"].items()))
            st["region"] = {
                "sibling_branch_pc": region_key[0],
                "sibling_branch_target_pc": region_key[1],
                "join_pc": region_key[1],
                "post_join_pc": region_key[2],
                "join_kind": region_key[3],
                "bssy_pc": region_key[4],
                "join_token": region_key[5],
                "validated_visits": region_visits,
            }
        del st["regions"]
        for key in ("entry_masks", "sibling_masks", "live_masks"):
            st[key] = dict(sorted(st[key].items()))
    return stats, len(files)


def validate_wait_blocks_dynamic(kdir, cands, byp, all_byp, max_ctas,
                                 expected_fid):
    """For every proven-shape candidate, traverse each CTA/warp occurrence with the
    recorded active/predicate masks and prove that:
      * every active visit begins at entry_wait_pc (an inactive occurrence, i.e.
        active_mask==0, is not a visit);
      * every dynamic PC between entry and resume is inside allowed_block_pcs;
      * every visit lands at resume_pc;
      * kernel/uid/CTA identity matches.
    A/B/C captured-path outcomes are counted for diagnostics (mixed is allowed).
    Returns {block_id: stats_dict} and the number of CTA files scanned. A proof
    failure sets stats["dynamic_ok"]=False and stats["reason"]=<taxonomy>."""
    files = sorted(glob.glob(os.path.join(kdir, "*.pb")))
    if max_ctas != "all":
        files = files[:int(max_ctas)]
    stats = {}
    for c in cands:
        stats[c["block_id"]] = {
            "active_entry_visits": 0,
            "predicated_off_entry_occurrences": 0,
            "all_live_entry_visits": 0,
            "partial_entry_visits": 0,
            "entry_effective_masks": Counter(),
            "entry_live_masks": Counter(),
            "entry_barrier_addrs": Counter(),
            "entry_wait_states": Counter(),
            "entry_sync_payload_missing": 0,
            "path_A_first_gate_ready": 0,   # entry -> captured_next, no 2nd gate
            "path_B_second_gate_ready": 0,  # 2nd gate poll before resume
            "path_C_fallback_loop": 0,      # entered the OOL fallback
            "resume_landings": 0,
            "dynamic_ok": True,
            "reason": None,
        }
    for f in files:
        warps = vpb_parse_full(open(f, "rb").read())
        for wid, insts in warps:
            pcs = [i["pc"] for i in insts]
            fids = [i["function_unique_id"] for i in insts]
            effective = [
                int(i["active_mask"]) & int(i["predicate_mask"])
                for i in insts
            ]
            # Derive the initial live mask from the observed warp itself. This
            # handles arbitrary block sizes without application or kernel-shape
            # constants. Dynamic EXIT occurrences remove lanes for later visits.
            live = 0
            for inst in insts:
                live |= int(inst["active_mask"])
            live_before = []
            for idx, inst in enumerate(insts):
                live_before.append(live)
                inst_fid = fids[idx]
                static_inst = all_byp.get(inst_fid, {}).get(inst["pc"])
                if (static_inst is not None
                        and static_inst["op_code"].split(".")[0] == "EXIT"
                        and effective[idx]):
                    live &= ~effective[idx]
            n = len(pcs)
            for c in cands:
                st = stats[c["block_id"]]
                if not st["dynamic_ok"]:
                    continue
                entry = c["entry_wait_pc"]
                resume = c["resume_pc"]
                allowed = set(c["allowed_block_pcs"])
                # Type-1 has a 2nd gate poll (all_gate_wait_pcs[1]); Type-2 has a
                # single gate, so there is no 2nd-gate PC to watch for.
                gates = c["all_gate_wait_pcs"]
                g2 = gates[1] if len(gates) > 1 else None
                # Mainline setup between the gate and its branch is executed on the
                # ready path before reaching resume; it is intentionally NOT in
                # allowed_block_pcs (never skipped), so treat it as pass-through
                # here rather than a path escape.
                mainline = set(c.get("mainline_setup_pcs", []))
                fb_entry = c["fallback_entry_pc"]
                j = 0
                while j < n:
                    if pcs[j] != entry or fids[j] != expected_fid:
                        j += 1
                        continue
                    # occurrence of the block entry
                    if effective[j] == 0:
                        st["predicated_off_entry_occurrences"] += 1
                        j += 1
                        continue
                    st["active_entry_visits"] += 1
                    entry_mask = effective[j]
                    entry_live = live_before[j]
                    st["entry_effective_masks"]["0x%x" % entry_mask] += 1
                    st["entry_live_masks"]["0x%x" % entry_live] += 1
                    if entry_mask == entry_live:
                        st["all_live_entry_visits"] += 1
                    else:
                        st["partial_entry_visits"] += 1
                    entry_inst = insts[j]
                    if (not entry_inst["sync_valid"]
                            or entry_inst["barrier_addr"] is None):
                        st["entry_sync_payload_missing"] += 1
                    else:
                        st["entry_barrier_addrs"][
                            "0x%x" % int(entry_inst["barrier_addr"])] += 1
                        # Proto3 omits scalar zero on the wire. Absence of field
                        # 4 in a valid sync payload therefore means wait-state 0.
                        wait_state = entry_inst["semantic_raw"]
                        wait_state = 0 if wait_state is None else int(wait_state)
                        st["entry_wait_states"]["0x%x" % wait_state] += 1
                    # Remember which lanes entered here. On a divergent warp the
                    # trace interleaves other lane-groups' PCs (a different
                    # active_mask). Only instructions whose active_mask overlaps
                    # the entering lanes belong to THIS visit; disjoint-mask PCs
                    # are another divergent path and are skipped here.
                    saw_g2 = False
                    saw_fb = False
                    k = j
                    landed = False
                    while k < n:
                        p = pcs[k]
                        # skip instructions that belong to a different divergent
                        # lane-group (no lane overlap with the entering lanes).
                        if (effective[k] & entry_mask) == 0:
                            k += 1
                            continue
                        if fids[k] != expected_fid:
                            st["dynamic_ok"] = False
                            st["reason"] = CTA_IDENTITY_MISMATCH
                            st["escape_pc"] = "0x%x" % p
                            break
                        if p == resume:
                            landed = True
                            break
                        if p not in allowed and p not in mainline:
                            st["dynamic_ok"] = False
                            st["reason"] = DYNAMIC_PATH_ESCAPE
                            st["escape_pc"] = "0x%x" % p
                            break
                        if p == g2:
                            saw_g2 = True
                        if p == fb_entry:
                            saw_fb = True
                        k += 1
                    if not st["dynamic_ok"]:
                        break
                    if not landed:
                        # ran off the end of the recorded stream without resuming:
                        # the block did not complete in this capture -> not proven.
                        st["dynamic_ok"] = False
                        st["reason"] = DYNAMIC_PATH_ESCAPE
                        st["escape_pc"] = "end-of-stream"
                        break
                    st["resume_landings"] += 1
                    if saw_fb:
                        st["path_C_fallback_loop"] += 1
                    elif saw_g2:
                        st["path_B_second_gate_ready"] += 1
                    else:
                        st["path_A_first_gate_ready"] += 1
                    j = k + 1
    for st in stats.values():
        for key in ("entry_effective_masks", "entry_live_masks",
                    "entry_barrier_addrs", "entry_wait_states"):
            st[key] = dict(sorted(st[key].items()))
    return stats, len(files)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help=".../traces directory")
    ap.add_argument("--kernels", default=None, help="comma list of kernel ids (default: all)")
    ap.add_argument("--out", default=None,
                    help="output json (default: <traces>/extra_info/spin_wait_sidecar.json)")
    ap.add_argument("--validate-ctas", default="all", help="'all' or an integer count")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any intended runtime record fails its proof")
    ap.add_argument(
        "--require-contract-capture", action="store_true",
        help="require barrier-local contract snapshots and publish the "
             "capture-only feasibility report")
    ap.add_argument(
        "--contract-baseline-traces", default=None,
        help="accepted baseline traces directory used for the mandatory "
             "semantic, WaitBlock, and TMA comparison")
    args = ap.parse_args()

    traces = args.traces
    extra = os.path.join(traces, "extra_info")
    if not os.path.isdir(extra):
        sys.exit("no extra_info under %s" % traces)

    kernels, names = load_kernels(extra)
    sync_path = os.path.join(extra, "sync_operand_resolver.json")
    if not os.path.exists(sync_path):
        if args.strict:
            sys.exit("[spin-sidecar] STRICT: missing %s" % sync_path)
        print("[spin-sidecar] no sync_operand_resolver.json; nothing to do")
        return
    sync_sites = load_sync_sites(extra)
    if not sync_sites:
        static_waits = [
            (fid, pc)
            for fid, byp in kernels.items()
            for pc, inst in byp.items()
            if any(kind in inst.get("op_code", "") for kind in WAIT_KINDS)
        ]
        if static_waits:
            sys.exit(
                "[spin-sidecar] sync_operand_resolver.json contains no wait "
                "sites, but static wait opcodes exist (first funcid=%d pc=0x%x)"
                % static_waits[0]
            )
        print("[spin-sidecar] no wait instructions; emitting empty v5 sidecar")
    kdirs = kernel_dir_map(traces)
    want = set(int(x) for x in args.kernels.split(",")) if args.kernels else None

    # map kernel_id -> funcid via the first pb (peek), so we describe per kernel_id
    kid_to_fid = {}
    for kid, kdir in kdirs.items():
        pbs = sorted(glob.glob(os.path.join(kdir, "*.pb")))
        if pbs:
            kid_to_fid[kid] = _ts._peek_funcid(pbs[0])

    out_kernels = []
    report = Counter()
    for field in (
            "discovered", "passed", "failed", "skipped",
            "inject_missing_wait", "fold_captured_poll", "metadata_only",
            "ready_only_all_live_visits", "ready_only_partial_visits",
            "ready_only_records_with_partial_visits",
            "partial_defer_records", "partial_join_pending_visits",
            "partial_join_already_issued_visits"):
        report[field] = 0
    report_lines = []
    for kid in sorted(kdirs):
        if want and kid not in want:
            continue
        fid = kid_to_fid.get(kid)
        if fid is None or fid not in kernels:
            continue
        byp = kernels[fid]

        # Strategy B is exactly the known-good legacy foldable spin-region set.
        # Keep the legacy records unchanged; the future simulator switches owner
        # only after it can consume sidecar v5.
        regions = detect_regions(fid, byp, sync_sites)
        stats, n_ctas = ({}, 0)
        clean = []
        captured_poll_body_pcs = set()
        fold_failures = []
        if regions:
            stats, n_ctas = validate_dynamic(
                kdirs[kid], fid, regions, args.validate_ctas)
            for r in regions:
                st = stats[r["region_id"]]
                r_out = dict(r)
                r_out["dynamic"] = st
                effect_input = {
                    "allowed_block_pcs":
                        list(r["body_pcs"]) + [r["loopback_bra_pc"]],
                    "loopback_bra_pc": r["loopback_bra_pc"],
                }
                effect_safe, effect_reason, effect_classes = prove_effects(
                    fid, effect_input, byp)
                r_out["effect_safe"] = effect_safe
                r_out["effect_reason"] = effect_reason
                r_out["effect_classes"] = effect_classes
                r_out["effect_proof"] = effect_input.get("effect_proof")
                # An actual semantic-free spin is an intended fold target. Every
                # control, wait-condition, effect, scoreboard, and dynamic
                # payload proof must pass; otherwise strict generation fails.
                fold_candidate = bool(
                    st["is_spin_loop"] and r["semantic_free_body"]
                    and not r["local_mem_ops_in_body"])
                proof_failure = None
                if fold_candidate:
                    lo, hi = r["loop_head_pc"], r["loopback_bra_pc"]
                    captured_poll_body_pcs.update(
                        range(lo, hi + ISIZE, ISIZE))
                    if not r["control_safe"]:
                        proof_failure = POLL_CONTROL_UNPROVEN
                    elif not r["wait_condition_safe"]:
                        proof_failure = POLL_WAIT_CONDITION_MISMATCH
                    elif not effect_safe:
                        proof_failure = POLL_EFFECT_UNSAFE
                    elif not r["barrier_self_tied"]:
                        proof_failure = POLL_SCOREBOARD_UNSAFE
                    elif (not st["active_poll_payload_complete"] or
                          not st["active_wait_payload_complete"]):
                        proof_failure = POLL_WAIT_TARGET_UNDECODABLE
                r_out["foldable"] = bool(
                    fold_candidate and proof_failure is None)
                if proof_failure is not None:
                    fold_failures.append((r_out, proof_failure))
                clean.append(r_out)

        def _is_fold_target(c):
            """Whether a candidate's captured poll arm has a legacy fold owner."""
            return bool(
                set(c.get("fallback_body_pcs", [])) &
                captured_poll_body_pcs
            )

        # Strategy A discovery has no application, kernel, function, name, PC,
        # CTA, warp, or SM filter. The same candidate proof is run independently
        # for every function found in the selected trace kernels.
        wb_cands = detect_wait_block_candidates(fid, byp, sync_sites)
        wb_pass, wb_fail, wb_skip = [], [], []
        wb_dyn = {}
        cfg_ok_cands = [c for c in wb_cands if c["cfg_ok"]]
        if cfg_ok_cands:
            wb_dyn, _ = validate_wait_blocks_dynamic(
                kdirs[kid], cfg_ok_cands, byp, kernels,
                args.validate_ctas, fid)

        def _bucket(c, reason):
            """A captured-poll candidate is already covered by the legacy fold
            strategy; a non-target shape is reported as skipped fail-closed."""
            if _is_fold_target(c):
                c["ignored_entry_reason"] = reason
                wb_skip.append((c, CAPTURED_POLL_ENTRY_NOT_REQUIRED))
            else:
                wb_skip.append((c, reason))

        for c in wb_cands:
            if not c["cfg_ok"]:
                _bucket(c, c["reason"])
                continue
            wait_ok, wait_reason, canonical = prove_wait_condition(
                fid, c, byp, sync_sites)
            if not wait_ok:
                _bucket(c, wait_reason)
                continue
            c["canonical_wait_condition"] = canonical
            st = wb_dyn.get(c["block_id"], {})
            if not st.get("dynamic_ok", False):
                _bucket(c, st.get("reason", DYNAMIC_PATH_ESCAPE))
                continue
            c["dynamic"] = st
            wb_pass.append(c)

        inject_candidates = []
        ready_only_candidates = []
        metadata_candidates = []
        strategy_failures = []
        for c in wb_pass:
            st = c["dynamic"]
            path_a = st["path_A_first_gate_ready"]
            path_b = st["path_B_second_gate_ready"]
            path_c = st["path_C_fallback_loop"]
            if path_a > 0 and path_b == 0 and path_c == 0:
                if st["entry_sync_payload_missing"] == 0:
                    ready_only_candidates.append(c)
                else:
                    strategy_failures.append(
                        (c, ENTRY_WAIT_TARGET_UNDECODABLE))
            elif _is_fold_target(c):
                # The runtime-facing record is emitted from the legacy spin
                # region, not from this entry candidate.
                pass
            else:
                metadata_candidates.append(c)

        partial_candidates = [
            c for c in ready_only_candidates
            if c["dynamic"]["partial_entry_visits"] > 0
        ]
        partial_proofs = {}
        if partial_candidates:
            partial_proofs, _ = validate_partial_join_dynamic(
                kdirs[kid], partial_candidates, byp,
                args.validate_ctas, fid)
        for c in ready_only_candidates:
            if c["dynamic"]["partial_entry_visits"] == 0:
                inject_candidates.append(c)
                continue
            proof = partial_proofs.get(c["block_id"], {})
            if not proof.get("dynamic_ok", False):
                strategy_failures.append(
                    (c, proof.get("reason", PARTIAL_REGION_INCONSISTENT)))
                c["partial_join_failure"] = proof
                continue
            c["partial_join_proof"] = proof
            inject_candidates.append(c)
        wb_fail.extend(strategy_failures)

        inject_records = [
            make_inject_missing_wait_record(kid, c, sync_sites, byp)
            for c in inject_candidates
        ]
        metadata_records = [
            make_metadata_only_record(c)
            for c in metadata_candidates
        ]
        fold_records = [
            make_fold_captured_poll_record(kid, fid, r)
            for r in clean if r["foldable"]
        ]
        strategy_records = inject_records + fold_records
        wait_block_records = strategy_records + metadata_records

        action_keys = set()
        for rec in strategy_records:
            action_pc = (rec["entry_probe_pc"]
                         if rec["runtime_strategy"] == "INJECT_MISSING_WAIT"
                         else rec["poll_probe_pc"])
            key = (kid, fid, action_pc)
            if key in action_keys:
                raise RuntimeError(
                    "duplicate wait-block runtime key %r" % (key,))
            action_keys.add(key)

        kernel_failed = len(wb_fail) + len(fold_failures)
        kernel_skipped = len(wb_skip)
        kernel_metadata = len(metadata_candidates)
        kernel_passed = len(strategy_records) + kernel_metadata
        kernel_discovered = kernel_passed + kernel_failed + kernel_skipped

        report["discovered"] += kernel_discovered
        report["passed"] += kernel_passed
        report["failed"] += kernel_failed
        report["skipped"] += kernel_skipped
        report["inject_missing_wait"] += len(inject_records)
        report["fold_captured_poll"] += len(fold_records)
        report["metadata_only"] += kernel_metadata
        report["ready_only_records_with_partial_visits"] += sum(
            1 for c in inject_candidates
            if c["dynamic"]["partial_entry_visits"] > 0)
        report["partial_defer_records"] += sum(
            1 for c in inject_candidates
            if c.get("partial_join_proof") is not None)
        for c in inject_candidates:
            st = c["dynamic"]
            report["ready_only_all_live_visits"] += \
                st["all_live_entry_visits"]
            report["ready_only_partial_visits"] += \
                st["partial_entry_visits"]
            proof = c.get("partial_join_proof")
            if proof is not None:
                report["partial_join_pending_visits"] += \
                    proof["join_pending_visits"]
                report["partial_join_already_issued_visits"] += \
                    proof["join_already_issued_visits"]

        report_lines.append(
            "[wait-block] kernel_%d funcid=%d: discovered=%d passed=%d "
            "failed=%d skipped=%d inject=%d fold=%d metadata=%d"
            % (kid, fid, kernel_discovered, kernel_passed, kernel_failed,
               kernel_skipped, len(inject_records), len(fold_records),
               kernel_metadata))
        for c in inject_candidates:
            st = c["dynamic"]
            proof = c.get("partial_join_proof")
            report_lines.append(
                "    INJECT entry=0x%x visits=%d all_live=%d partial=%d "
                "masks=%s barriers=%s wait_states=%s%s"
                % (c["entry_wait_pc"], st["active_entry_visits"],
                   st["all_live_entry_visits"], st["partial_entry_visits"],
                   st["entry_effective_masks"], st["entry_barrier_addrs"],
                   st["entry_wait_states"],
                   (" partial_action=DEFER_BLOCKED_SUFFIX_THEN_HOLD "
                    "join=0x%x pending=%d already_issued=%d"
                    % (proof["region"]["join_pc"],
                       proof["join_pending_visits"],
                       proof["join_already_issued_visits"]))
                   if proof is not None else ""))
        for r in clean:
            if r["foldable"]:
                report_lines.append(
                    "    FOLD poll=0x%x head=0x%x loopback=0x%x exit=0x%x "
                    "visits=%d iters=%d active_polls=%d predicated_off=%d"
                    % (r["primary_wait_pc"], r["loop_head_pc"],
                       r["loopback_bra_pc"], r["exit_pc"],
                       r["dynamic"]["warp_visits"],
                       r["dynamic"]["total_iters"],
                       r["dynamic"]["active_poll_visits"],
                       r["dynamic"]["predicated_off_poll_occurrences"]))
        for r, reason in fold_failures:
            report_lines.append(
                "    FAIL poll=0x%x reason=%s control_reason=%s "
                "wait_condition_reason=%s effect_reason=%s "
                "scoreboard_external=%s active_polls=%d "
                "payload_missing=%d all_wait_payload_missing=%d "
                "dynamic_wait_mismatches=%d"
                % (r["primary_wait_pc"], reason,
                   r.get("control_reason"),
                   r.get("wait_condition_reason"),
                   r.get("effect_reason"),
                   r["barrier_reason"]["wait_on_externally_armed_ids"],
                   r["dynamic"]["active_poll_visits"],
                   r["dynamic"]["poll_sync_payload_missing"],
                   r["dynamic"]["wait_sync_payload_missing"],
                   r["dynamic"]["wait_condition_mismatches"]))
        for c, reason in wb_fail:
            detail = c.get("partial_join_failure", {})
            report_lines.append(
                "    FAIL entry=0x%x reason=%s detail=%s"
                % (c["entry_wait_pc"], reason,
                   detail.get("failure_detail")))
        for c, reason in wb_skip:
            report_lines.append(
                "    SKIP entry=0x%x reason=%s%s"
                % (c["entry_wait_pc"], reason,
                   (" (entry proof not used by captured-poll strategy; "
                    "observed=%s)" % c.get("ignored_entry_reason"))
                   if reason == CAPTURED_POLL_ENTRY_NOT_REQUIRED else ""))
        if metadata_candidates:
            for c in metadata_candidates:
                st = c["dynamic"]
                report_lines.append(
                    "    METADATA entry=0x%x A/B/C=%d/%d/%d"
                    % (c["entry_wait_pc"],
                       st["path_A_first_gate_ready"],
                       st["path_B_second_gate_ready"],
                       st["path_C_fallback_loop"]))

        out_kernels.append({
            "kernel_id": kid,
            "trace_kernel_id": kid,
            "unique_function_id": fid,
            "kernel_name": names.get(fid, ""),
            "num_regions": len(clean),
            "validated_ctas": (n_ctas if regions else
                                (len(glob.glob(os.path.join(
                                    kdirs[kid], "*.pb")))
                                 if args.validate_ctas == "all"
                                 else int(args.validate_ctas))),
            "spin_regions": clean,
            "wait_blocks": wait_block_records,
        })

        n_sem = sum(0 if r["semantic_free_body"] else 1 for r in regions)
        n_foldable = sum(1 for r in clean if r["foldable"])
        n_bar_unsafe = sum(1 for r in regions
                           if not r["barrier_self_tied"]
                           and stats[r["region_id"]]["is_spin_loop"])
        print("[spin-sidecar] kernel_%d funcid=%d: %d spin regions "
              "(foldable=%d, semantic-in-body=%d, spin+barrier-unsafe=%d), "
              "validated %d CTAs"
              % (kid, fid, len(regions), n_foldable, n_sem, n_bar_unsafe, n_ctas))
        for r in clean:
            st = r["dynamic"]
            kind = "spin" if st["is_spin_loop"] else "single-shot"
            flags = ""
            if not r["semantic_free_body"]:
                flags += "  [!! SEMANTIC IN BODY]"
            if st["is_spin_loop"] and not r["barrier_self_tied"]:
                flags += "  [!! BARRIER NOT SELF-TIED %s]" % (
                    r["barrier_reason"]["wait_on_externally_armed_ids"]
                    or [o for _, o in r["barrier_reason"]["async_producer_ops"]])
            note = ""
            if r["local_mem_ops_in_body"]:
                note = "  local_mem=%d" % len(r["local_mem_ops_in_body"])
            print("    %-16s [%-11s] head=0x%-5x wait=0x%-5x(%s) exit=0x%-5x "
                  "iters[min/max/tot]=%s/%s/%s warp_visits=%d foldable=%d%s%s"
                  % (r["region_id"], kind, r["loop_head_pc"], r["primary_wait_pc"],
                     r["primary_wait_kind"], r["exit_pc"],
                     st["min_iters"], st["max_iters"], st["total_iters"],
                     st["warp_visits"], int(r["foldable"]), note, flags))

    print("[wait-block] === two-strategy sidecar report ===")
    print("[wait-block] TOTAL discovered=%d passed=%d failed=%d skipped=%d "
          "inject=%d fold=%d metadata=%d"
          % (report["discovered"], report["passed"], report["failed"],
             report["skipped"], report["inject_missing_wait"],
             report["fold_captured_poll"], report["metadata_only"]))
    print("[wait-block] READY visits all_live=%d partial=%d "
          "records_with_partial=%d"
          % (report["ready_only_all_live_visits"],
             report["ready_only_partial_visits"],
             report["ready_only_records_with_partial_visits"]))
    print("[wait-block] PARTIAL defer_records=%d join_pending=%d "
          "join_already_issued=%d"
          % (report["partial_defer_records"],
             report["partial_join_pending_visits"],
             report["partial_join_already_issued_visits"]))
    for line in report_lines:
        print(line)

    sidecar = {
        "version": 5,
        "schema": "wait_block_sidecar",
        "validation_complete": args.validate_ctas == "all",
        "source_traces_dir": os.path.abspath(traces),
        "isize_bytes": ISIZE,
        "note": (
            "Two independent runtime strategies. INJECT_MISSING_WAIT is emitted "
            "only for a dynamically ready-only gate; modeled-ready replays "
            "normally, an all-live modeled miss holds at the entry, and a "
            "partial modeled miss uses DEFER_BLOCKED_SUFFIX_THEN_HOLD only after "
            "static CFG and complete dynamic join/capsule proofs. "
            "FOLD_CAPTURED_POLL is emitted "
            "directly from an existing foldable spin region only after proving "
            "primary-wait control ownership, one static/dynamic wait condition, "
            "idempotent skipped ALU effects, and scoreboard safety over the full "
            "body plus loopback branch. It advances only to its local "
            "poll_loop_exit_pc."),
        "wait_block_fail_taxonomy": [
            CFG_UNRESOLVED, MULTI_REJOIN,
            DYNAMIC_PATH_ESCAPE, CTA_IDENTITY_MISMATCH,
            ENTRY_WAIT_TARGET_UNDECODABLE, POLL_WAIT_TARGET_UNDECODABLE,
            POLL_CONTROL_UNPROVEN, POLL_WAIT_CONDITION_MISMATCH,
            POLL_EFFECT_UNSAFE, POLL_SCOREBOARD_UNSAFE,
            CAPTURED_POLL_ENTRY_NOT_REQUIRED,
            PARTIAL_MASK_PARTITION_INVALID,
            PARTIAL_SIBLING_BRANCH_UNRESOLVED,
            PARTIAL_ENTRY_BRANCH_OCCURRENCE_MISSING,
            PARTIAL_JOIN_STATIC_UNSAFE,
            PARTIAL_JOIN_OCCURRENCE_MISSING,
            PARTIAL_SIBLING_NOT_QUIESCENT,
            PARTIAL_CAPSULE_MASK_ESCAPE,
            PARTIAL_CAPSULE_CONTROL_UNSAFE,
            PARTIAL_POST_JOIN_UNRESOLVED,
            PARTIAL_REGION_INCONSISTENT,
        ],
        "wait_block_report": dict(report),
        "kernels": out_kernels,
    }
    contract_report = None
    if args.require_contract_capture:
        contract_report = analyze_mbarrier_contract_capture(
            traces, kdirs, kid_to_fid, sync_sites, out_kernels,
            args.validate_ctas)
        if args.contract_baseline_traces:
            baseline = os.path.abspath(args.contract_baseline_traces)
            contract_report["baseline_comparison"] = \
                compare_contract_capture_baseline(
                    traces, baseline, out_kernels, report)
            if contract_report["baseline_comparison"]["status"] != "PASS":
                contract_report["status"] = "FAIL"
        else:
            contract_report["baseline_comparison"] = {
                "status": "NOT_REQUESTED",
            }
            if args.strict:
                contract_report["status"] = "FAIL"
        contract_path = os.path.join(
            extra, "mbarrier_contract_ordinal_feasibility.json")
        contract_tmp = contract_path + ".tmp"
        with open(contract_tmp, "w") as fh:
            json.dump(contract_report, fh, indent=2)
        os.replace(contract_tmp, contract_path)
        sidecar["mbarrier_contract_capture"] = {
            "version": contract_report["version"],
            "schema": contract_report["schema"],
            "status": contract_report["status"],
            "artifact": os.path.basename(contract_path),
            "summary": contract_report["summary"],
        }
        summary = contract_report["summary"]
        print(
            "[mbarrier-contract] FINAL status=%s "
            "active_sync=%d missing=%d mismatches=%d barriers=%d "
            "eligible_barriers=%d eligible_actions=%d "
            "rejected_tma_actions=%d unique=%d "
            "ambiguous=%d inconsistent=%d unique_streams=%d "
            "invalid_streams=%d output=%s"
            % (
                contract_report["status"],
                summary.get("active_sync_occurrences", 0),
                summary.get("missing_snapshots", 0),
                summary.get("capture_mismatches", 0),
                summary.get("barriers", 0),
                summary.get("eligible_barriers", 0),
                summary.get("eligible_action_occurrences", 0),
                summary.get("rejected_tma_action_occurrences", 0),
                summary.get("unique_assignments", 0),
                summary.get("ambiguous_assignments", 0),
                summary.get("inconsistent_assignments", 0),
                summary.get("unique_streams", 0),
                summary.get("ambiguous_or_invalid_streams", 0),
                contract_path,
            )
        )
    if args.strict and args.validate_ctas != "all":
        sys.exit("[spin-sidecar] STRICT: runtime strategy emission requires "
                 "--validate-ctas all")
    if args.strict and report["failed"]:
        sys.exit("[spin-sidecar] STRICT: at least one intended runtime record "
                 "failed its proof; inspect FAIL lines")
    if (args.strict and args.require_contract_capture
            and contract_report["status"] != "PASS"):
        sys.exit("[mbarrier-contract] STRICT: contract-ordinal feasibility "
                 "failed; inspect mbarrier_contract_ordinal_feasibility.json")

    out_path = args.out or os.path.join(extra, "spin_wait_sidecar.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)
    os.replace(tmp_path, out_path)
    print("[spin-sidecar] wrote %s (%d kernels)" % (out_path, len(out_kernels)))
    final_status = (
        "FAILED" if report["failed"]
        else ("PASS" if sidecar["validation_complete"] else "INCOMPLETE")
    )
    print(
        "[wait-block] FINAL status=%s validation_complete=%s "
        "discovered=%d passed=%d failed=%d skipped=%d "
        "inject=%d fold=%d metadata=%d all_live=%d partial=%d "
        "records_with_partial=%d partial_defer_records=%d "
        "partial_join_pending=%d partial_join_already_issued=%d output=%s"
        % (
            final_status,
            str(sidecar["validation_complete"]).lower(),
            report["discovered"],
            report["passed"],
            report["failed"],
            report["skipped"],
            report["inject_missing_wait"],
            report["fold_captured_poll"],
            report["metadata_only"],
            report["ready_only_all_live_visits"],
            report["ready_only_partial_visits"],
            report["ready_only_records_with_partial_visits"],
            report["partial_defer_records"],
            report["partial_join_pending_visits"],
            report["partial_join_already_issued_visits"],
            out_path,
        )
    )


if __name__ == "__main__":
    main()
