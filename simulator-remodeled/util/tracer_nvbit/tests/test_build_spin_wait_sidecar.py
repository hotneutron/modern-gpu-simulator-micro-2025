import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_spin_wait_sidecar.py"
SPEC = importlib.util.spec_from_file_location("build_spin_wait_sidecar", MODULE_PATH)
SIDECAR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIDECAR)


def operand(text, immediate=None):
    result = {"operand_string": text}
    if immediate is not None:
        result["has_inmediate"] = True
        result["operand_inmmediates"] = [immediate]
    return result


def instruction(opcode, operands, *, predicate=None, negate=False,
                control_bits=None):
    return {
        "op_code": opcode,
        "operands": operands,
        "is_predicated": predicate is not None,
        "is_uniform_predicate": False,
        "is_predicate_negate": negate,
        "predicate_register": 0 if predicate is None else predicate,
        "control_bits": control_bits or {},
    }


def wait(pc, predicate=0, barrier="[R1]", state="R2", trywait=False):
    opcode = (
        "SYNCS.PHASECHK.TRANS64.TRYWAIT"
        if trywait else "SYNCS.PHASECHK.TRANS64"
    )
    return instruction(
        opcode,
        [operand("P%d" % predicate), operand(barrier), operand(state)],
    )


def region():
    return {
        "primary_wait_pc": 0x120,
        "wait_pcs": [0x100, 0x120],
        "loop_head_pc": 0x100,
        "loopback_bra_pc": 0x130,
        "exit_pc": 0x140,
        "body_pcs": [0x100, 0x110, 0x120],
    }


def dynamic_instruction(pc, active, predicate=None, fid=1):
    return {
        "pc": pc,
        "active_mask": active,
        "predicate_mask": active if predicate is None else predicate,
        "function_unique_id": fid,
        "sync_valid": False,
        "barrier_addr": None,
        "semantic_raw": None,
        "contract_capture_valid": False,
        "contract_event_kind": None,
        "contract_barrier_addr": None,
        "contract_epoch_before": None,
        "contract_epoch_after": None,
        "contract_opened_before": None,
        "contract_opened_after": None,
        "contract_has_ordinal": False,
        "contract_ordinal": None,
        "contract_capture_flags": 0,
        "contract_barrier_error_flags": 0,
        "contract_capture_launch_uid": None,
        "contract_has_semantic_raw": False,
        "contract_semantic_raw": None,
    }


def encode_varint(value):
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7f) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def varint_field(number, value):
    return encode_varint(number << 3) + encode_varint(value)


def message_field(number, payload):
    return encode_varint((number << 3) | 2) + encode_varint(len(payload)) + payload


def captured_instruction_wire(pc, event_kind, semantic_raw, opened_before,
                              opened_after, ordinal=0, launch_uid=7,
                              barrier=0x1000):
    sync = b"".join([
        varint_field(1, 1),
        varint_field(2, barrier),
        varint_field(3, 1),
        varint_field(4, semantic_raw),
    ])
    contract_fields = [
        varint_field(1, 1),
        varint_field(2, event_kind),
        varint_field(3, barrier),
        varint_field(4, 0 if event_kind == SIDECAR.CONTRACT_EVENT_EXCH else 1),
        varint_field(5, 1),
        varint_field(6, opened_before),
        varint_field(7, opened_after),
        varint_field(10, 0),
        varint_field(11, 0),
        varint_field(12, launch_uid),
        varint_field(13, 1),
        varint_field(14, semantic_raw),
    ]
    if ordinal:
        contract_fields.extend([
            varint_field(8, 1),
            varint_field(9, ordinal),
        ])
    return b"".join([
        varint_field(1, pc),
        varint_field(2, 1),
        varint_field(3, 1),
        varint_field(4, 3),
        message_field(6, sync),
        message_field(9, b"".join(contract_fields)),
    ])


def sync_only_instruction_wire(pc, semantic_raw, barrier=0x1000):
    sync = b"".join([
        varint_field(1, 1),
        varint_field(2, barrier),
        varint_field(3, 1),
        varint_field(4, semantic_raw),
    ])
    return b"".join([
        varint_field(1, pc),
        varint_field(2, 1),
        varint_field(3, 1),
        varint_field(4, 3),
        message_field(6, sync),
    ])


def write_single_warp_threadblock(path, instructions):
    warp = varint_field(1, 0) + b"".join(
        message_field(2, instruction)
        for instruction in instructions
    )
    map_entry = varint_field(1, 0) + message_field(2, warp)
    path.write_bytes(message_field(2, map_entry))


class ContractOrdinalCaptureTest(unittest.TestCase):
    def test_ready_snapshot_selects_one_completed_generation(self):
        targets, reason = SIDECAR.resolve_contract_generation(
            4, 4, 0, True)

        self.assertIsNone(reason)
        self.assertEqual(targets, [3])

    def test_ready_snapshot_may_target_initial_generation_zero(self):
        targets, reason = SIDECAR.resolve_contract_generation(
            0, 0, 1, True)

        self.assertIsNone(reason)
        self.assertEqual(targets, [0])

    def test_miss_snapshot_targets_next_generation(self):
        targets, reason = SIDECAR.resolve_contract_generation(
            4, 4, 0, False)

        self.assertIsNone(reason)
        self.assertEqual(targets, [5])

    def test_overlapping_two_parity_equivalent_contracts_is_ambiguous(self):
        targets, reason = SIDECAR.resolve_contract_generation(
            4, 6, 0, True)

        self.assertIsNone(reason)
        self.assertEqual(targets, [3, 5])

    def test_many_waits_can_name_the_same_release(self):
        resolutions = [
            SIDECAR.resolve_contract_generation(4, 4, 0, False)[0]
            for _ in range(3)
        ]

        self.assertEqual(resolutions, [[5], [5], [5]])

    def test_count_regression_is_rejected(self):
        targets, reason = SIDECAR.resolve_contract_generation(
            5, 4, 0, True)

        self.assertEqual(targets, [])
        self.assertEqual(reason, SIDECAR.CONTRACT_COUNT_REGRESSION)

    def test_barrier_requires_single_arrival_contract_shape(self):
        exch = dynamic_instruction(0x100, 1)
        exch.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_EXCH,
            "semantic_raw": 0x1ffffe,
            "contract_epoch_before": 0,
            "contract_epoch_after": 1,
            "contract_opened_before": 0,
            "contract_opened_after": 0,
        })
        arrive = dynamic_instruction(0x110, 1)
        arrive.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_ARRIVE,
            "semantic_raw": 256,
            "contract_epoch_before": 1,
            "contract_epoch_after": 1,
            "contract_opened_before": 0,
            "contract_opened_after": 1,
            "contract_has_ordinal": True,
            "contract_ordinal": 1,
        })

        status = SIDECAR.validate_contract_barrier(
            [exch, arrive], True)

        self.assertTrue(status["eligible"])
        self.assertEqual(status["contracts"], 1)

    def test_reused_barrier_requires_ready_witness_before_next_contract(self):
        exch = dynamic_instruction(0x100, 1)
        exch.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_EXCH,
            "semantic_raw": 0x1ffffe,
            "contract_epoch_before": 0,
            "contract_epoch_after": 1,
            "contract_opened_before": 0,
            "contract_opened_after": 0,
        })
        arrivals = []
        for ordinal in (1, 2):
            arrive = dynamic_instruction(0x110, 1)
            arrive.update({
                "contract_event_kind": SIDECAR.CONTRACT_EVENT_ARRIVE,
                "semantic_raw": 256,
                "contract_epoch_before": 1,
                "contract_epoch_after": 1,
                "contract_opened_before": ordinal - 1,
                "contract_opened_after": ordinal,
                "contract_has_ordinal": True,
                "contract_ordinal": ordinal,
            })
            arrivals.append(arrive)
        ready_wait = dynamic_instruction(0x120, 1)
        ready_wait.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_WAIT,
            "semantic_raw": 0,
            "contract_epoch_before": 1,
            "contract_epoch_after": 1,
            "contract_opened_before": 1,
            "contract_opened_after": 1,
        })

        rejected = SIDECAR.validate_contract_barrier(
            [exch] + arrivals, True)
        accepted = SIDECAR.validate_contract_barrier(
            [exch] + arrivals + [ready_wait], True,
            [(ready_wait, True)])

        self.assertIn(
            SIDECAR.CONTRACT_SERIAL_LIFECYCLE_UNPROVEN,
            rejected["reject_reasons"])
        self.assertTrue(accepted["eligible"])

    def test_duplicate_initialization_is_rejected(self):
        events = []
        for epoch in (1, 2):
            exch = dynamic_instruction(0x100, 1)
            exch.update({
                "contract_event_kind": SIDECAR.CONTRACT_EVENT_EXCH,
                "semantic_raw": 0x1ffffe,
                "contract_epoch_before": epoch - 1,
                "contract_epoch_after": epoch,
                "contract_opened_before": 0,
                "contract_opened_after": 0,
            })
            events.append(exch)

        status = SIDECAR.validate_contract_barrier(events, True)

        self.assertFalse(status["eligible"])
        self.assertIn(
            SIDECAR.CONTRACT_INITIALIZATION_DUPLICATE,
            status["reject_reasons"])

    def test_plain_arrive_is_rejected(self):
        exch = dynamic_instruction(0x100, 1)
        exch.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_EXCH,
            "semantic_raw": 0x1ffffe,
            "contract_epoch_before": 0,
            "contract_epoch_after": 1,
            "contract_opened_before": 0,
            "contract_opened_after": 0,
        })
        arrive = dynamic_instruction(0x110, 1)
        arrive.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_ARRIVE,
            "semantic_raw": 0,
            "contract_epoch_before": 1,
            "contract_epoch_after": 1,
            "contract_opened_before": 0,
            "contract_opened_after": 0,
            "contract_capture_flags":
                SIDECAR.CONTRACT_CAPTURE_PLAIN_ARRIVE,
        })

        status = SIDECAR.validate_contract_barrier(
            [exch, arrive], True)

        self.assertFalse(status["eligible"])
        self.assertIn(
            SIDECAR.CONTRACT_PLAIN_ARRIVE_PRESENT,
            status["reject_reasons"])

    def test_missing_tma_ownership_is_rejected(self):
        exch = dynamic_instruction(0x100, 1)
        exch.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_EXCH,
            "semantic_raw": 0x1ffffe,
            "contract_epoch_before": 0,
            "contract_epoch_after": 1,
            "contract_opened_before": 0,
            "contract_opened_after": 0,
        })
        arrive = dynamic_instruction(0x110, 1)
        arrive.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_ARRIVE,
            "semantic_raw": 256,
            "contract_epoch_before": 1,
            "contract_epoch_after": 1,
            "contract_opened_before": 0,
            "contract_opened_after": 1,
            "contract_has_ordinal": True,
            "contract_ordinal": 1,
        })

        status = SIDECAR.validate_contract_barrier(
            [exch, arrive], False)

        self.assertFalse(status["eligible"])
        self.assertIn(
            SIDECAR.CONTRACT_TMA_OWNERSHIP_UNPROVEN,
            status["reject_reasons"])

    def test_arrive_ordinal_must_be_inside_snapshot_interval(self):
        exch = dynamic_instruction(0x100, 1)
        exch.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_EXCH,
            "semantic_raw": 0x1ffffe,
            "contract_epoch_before": 0,
            "contract_epoch_after": 1,
            "contract_opened_before": 0,
            "contract_opened_after": 0,
        })
        arrive = dynamic_instruction(0x110, 1)
        arrive.update({
            "contract_event_kind": SIDECAR.CONTRACT_EVENT_ARRIVE,
            "semantic_raw": 256,
            "contract_epoch_before": 1,
            "contract_epoch_after": 1,
            "contract_opened_before": 0,
            "contract_opened_after": 1,
            "contract_has_ordinal": True,
            "contract_ordinal": 2,
        })

        status = SIDECAR.validate_contract_barrier(
            [exch, arrive], True)

        self.assertFalse(status["eligible"])
        self.assertIn(
            SIDECAR.CONTRACT_ORDINAL_INTERVAL_MISMATCH,
            status["reject_reasons"])

    def test_fold_result_uses_loopback_and_exit_boundaries(self):
        spec = {
            "poll_probe_pc": 0x120,
            "poll_loopback_pc": 0x130,
            "poll_loop_exit_pc": 0x140,
        }
        miss_trace = [
            dynamic_instruction(0x120, 0xf),
            dynamic_instruction(0x130, 0xf),
            dynamic_instruction(0x120, 0xf),
            dynamic_instruction(0x140, 0xf),
        ]

        self.assertFalse(
            SIDECAR._fold_wait_result(miss_trace, 0, spec))
        self.assertTrue(
            SIDECAR._fold_wait_result(miss_trace, 2, spec))

    def test_preceding_trywait_uses_predicated_poll_path(self):
        spec = {
            "poll_probe_pc": 0x120,
            "poll_loopback_pc": 0x130,
            "poll_loop_exit_pc": 0x140,
        }
        miss_trace = [
            dynamic_instruction(0x110, 0xf),
            dynamic_instruction(0x120, 0xf),
            dynamic_instruction(0x130, 0xf),
        ]
        ready_trace = [
            dynamic_instruction(0x110, 0xf),
            dynamic_instruction(0x120, 0xf, predicate=0),
            dynamic_instruction(0x130, 0xf, predicate=0),
            dynamic_instruction(0x140, 0xf),
        ]

        self.assertFalse(
            SIDECAR._fold_wait_site_result(miss_trace, 0, spec))
        self.assertTrue(
            SIDECAR._fold_wait_site_result(ready_trace, 0, spec))

    def test_contract_capture_protobuf_fields_are_parsed(self):
        contract = b"".join([
            varint_field(1, 1),
            varint_field(2, SIDECAR.CONTRACT_EVENT_WAIT),
            varint_field(3, 0x16c50),
            varint_field(4, 1),
            varint_field(5, 1),
            varint_field(6, 4),
            varint_field(7, 5),
            varint_field(8, 0),
            varint_field(10, 0),
            varint_field(11, 0),
            varint_field(12, 7),
            varint_field(13, 1),
        ])
        instruction_wire = b"".join([
            varint_field(1, 0x74e0),
            varint_field(2, 0xffffffff),
            varint_field(3, 0xffffffff),
            varint_field(4, 3),
            message_field(9, contract),
        ])

        parsed = SIDECAR._inst_full(
            instruction_wire, 0, len(instruction_wire))

        self.assertTrue(parsed["contract_capture_valid"])
        self.assertEqual(
            parsed["contract_event_kind"],
            SIDECAR.CONTRACT_EVENT_WAIT)
        self.assertEqual(parsed["contract_barrier_addr"], 0x16c50)
        self.assertEqual(parsed["contract_opened_before"], 4)
        self.assertEqual(parsed["contract_opened_after"], 5)
        self.assertEqual(parsed["contract_capture_launch_uid"], 7)
        self.assertTrue(parsed["contract_has_semantic_raw"])
        self.assertEqual(parsed["contract_semantic_raw"], 0)

    def test_omitted_proto3_zero_capture_fields_decode_as_zero(self):
        contract = b"".join([
            varint_field(1, 1),
            varint_field(2, SIDECAR.CONTRACT_EVENT_EXCH),
            varint_field(3, 0x1000),
            varint_field(5, 1),
            varint_field(12, 7),
            varint_field(13, 1),
            varint_field(14, 0x1ffffe),
        ])
        instruction_wire = b"".join([
            varint_field(1, 0x100),
            varint_field(2, 1),
            varint_field(3, 1),
            varint_field(4, 3),
            message_field(9, contract),
        ])

        parsed = SIDECAR._inst_full(
            instruction_wire, 0, len(instruction_wire))

        self.assertEqual(parsed["contract_epoch_before"], 0)
        self.assertEqual(parsed["contract_opened_before"], 0)
        self.assertEqual(parsed["contract_opened_after"], 0)

    def test_baseline_comparison_requires_semantic_and_metadata_equalities(self):
        current_kernel = {
            "trace_kernel_id": 5,
            "wait_blocks": [{
                "block_id": "fold",
                "runtime_strategy": "FOLD_CAPTURED_POLL",
                "poll_probe_pc": 0x120,
                "dynamic": {"total_iters": 10},
            }],
        }
        baseline_kernel = json.loads(json.dumps(current_kernel))
        baseline_kernel["wait_blocks"][0]["dynamic"]["total_iters"] = 100
        wait_report = {
            "inject_missing_wait": 0,
            "fold_captured_poll": 1,
            "ready_only_all_live_visits": 0,
            "ready_only_partial_visits": 0,
            "partial_defer_records": 0,
            "partial_join_pending_visits": 0,
            "partial_join_already_issued_visits": 0,
        }
        contracts = {
            "validation_complete": True,
            "kernels": [{
                "trace_kernel_id": 5,
                "threadblocks": 1,
                "contracts": 1,
                "completion_loads": 1,
                "expected_bytes": 256,
                "assigned_bytes": 256,
                "errors": [],
            }],
        }
        with tempfile.TemporaryDirectory() as current_dir, \
                tempfile.TemporaryDirectory() as baseline_dir:
            current_extra = Path(current_dir) / "extra_info"
            baseline_extra = Path(baseline_dir) / "extra_info"
            current_extra.mkdir()
            baseline_extra.mkdir()
            (baseline_extra / "spin_wait_sidecar.json").write_text(
                json.dumps({
                    "wait_block_report": wait_report,
                    "kernels": [baseline_kernel],
                }))
            for extra in (current_extra, baseline_extra):
                (extra / "tma_completion_contracts.json").write_text(
                    json.dumps(contracts))
            fingerprint = {
                "sha256": "same",
                "counts": {"instructions": 10},
            }
            with mock.patch.object(
                    SIDECAR, "semantic_trace_fingerprint",
                    return_value=fingerprint):
                result = SIDECAR.compare_contract_capture_baseline(
                    current_dir, baseline_dir,
                    [current_kernel], wait_report)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["comparisons"].values()))
        self.assertFalse(result["fold_dynamic_equal"])

    def test_baseline_comparison_ignores_inject_runtime_identity_histograms(self):
        current_kernel = {
            "trace_kernel_id": 5,
            "wait_blocks": [{
                "block_id": "inject",
                "runtime_strategy": "INJECT_MISSING_WAIT",
                "entry_probe_pc": 0x120,
                "active_entry_visits": 8,
                "observed_effective_masks": {"0xffffffff": 8},
                "observed_live_masks": {"0xffffffff": 8},
                "observed_barrier_addrs": {
                    "0x1000": 3,
                    "0x1008": 5,
                },
                "observed_wait_states": {
                    "0x0": 3,
                    "0x80000000": 5,
                },
            }],
        }
        baseline_kernel = json.loads(json.dumps(current_kernel))
        baseline_record = baseline_kernel["wait_blocks"][0]
        baseline_record["observed_barrier_addrs"] = {
            "0x1000": 4,
            "0x1008": 4,
        }
        baseline_record["observed_wait_states"] = {
            "0x0": 4,
            "0x80000000": 4,
        }
        wait_report = {
            "inject_missing_wait": 1,
            "fold_captured_poll": 0,
            "ready_only_all_live_visits": 8,
            "ready_only_partial_visits": 0,
            "partial_defer_records": 0,
            "partial_join_pending_visits": 0,
            "partial_join_already_issued_visits": 0,
        }
        contracts = {
            "kernels": [{
                "trace_kernel_id": 5,
                "threadblocks": 1,
                "contracts": 1,
                "completion_loads": 1,
                "expected_bytes": 256,
                "assigned_bytes": 256,
                "errors": [],
            }],
        }
        with tempfile.TemporaryDirectory() as current_dir, \
                tempfile.TemporaryDirectory() as baseline_dir:
            current_extra = Path(current_dir) / "extra_info"
            baseline_extra = Path(baseline_dir) / "extra_info"
            current_extra.mkdir()
            baseline_extra.mkdir()
            (baseline_extra / "spin_wait_sidecar.json").write_text(
                json.dumps({
                    "wait_block_report": wait_report,
                    "kernels": [baseline_kernel],
                }))
            for extra in (current_extra, baseline_extra):
                (extra / "tma_completion_contracts.json").write_text(
                    json.dumps(contracts))
            fingerprint = {
                "sha256": "same",
                "counts": {"semantic_instructions": 8},
            }
            with mock.patch.object(
                    SIDECAR, "semantic_trace_fingerprint",
                    return_value=fingerprint):
                result = SIDECAR.compare_contract_capture_baseline(
                    current_dir, baseline_dir,
                    [current_kernel], wait_report)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["comparisons"].values()))
        self.assertFalse(result["inject_runtime_identity_equal"])

    def test_baseline_comparison_keeps_inject_mask_histograms_strict(self):
        current_kernel = {
            "trace_kernel_id": 5,
            "wait_blocks": [{
                "block_id": "inject",
                "runtime_strategy": "INJECT_MISSING_WAIT",
                "entry_probe_pc": 0x120,
                "active_entry_visits": 8,
                "observed_effective_masks": {"0xffffffff": 8},
                "observed_live_masks": {"0xffffffff": 8},
                "observed_barrier_addrs": {"0x1000": 8},
                "observed_wait_states": {"0x0": 8},
            }],
        }
        baseline_kernel = json.loads(json.dumps(current_kernel))
        baseline_kernel["wait_blocks"][0][
            "observed_effective_masks"] = {"0x1": 8}
        wait_report = {
            "inject_missing_wait": 1,
            "fold_captured_poll": 0,
            "ready_only_all_live_visits": 8,
            "ready_only_partial_visits": 0,
            "partial_defer_records": 0,
            "partial_join_pending_visits": 0,
            "partial_join_already_issued_visits": 0,
        }
        contracts = {
            "kernels": [{
                "trace_kernel_id": 5,
                "threadblocks": 1,
                "contracts": 1,
                "completion_loads": 1,
                "expected_bytes": 256,
                "assigned_bytes": 256,
                "errors": [],
            }],
        }
        with tempfile.TemporaryDirectory() as current_dir, \
                tempfile.TemporaryDirectory() as baseline_dir:
            current_extra = Path(current_dir) / "extra_info"
            baseline_extra = Path(baseline_dir) / "extra_info"
            current_extra.mkdir()
            baseline_extra.mkdir()
            (baseline_extra / "spin_wait_sidecar.json").write_text(
                json.dumps({
                    "wait_block_report": wait_report,
                    "kernels": [baseline_kernel],
                }))
            for extra in (current_extra, baseline_extra):
                (extra / "tma_completion_contracts.json").write_text(
                    json.dumps(contracts))
            fingerprint = {
                "sha256": "same",
                "counts": {"semantic_instructions": 8},
            }
            with mock.patch.object(
                    SIDECAR, "semantic_trace_fingerprint",
                    return_value=fingerprint):
                result = SIDECAR.compare_contract_capture_baseline(
                    current_dir, baseline_dir,
                    [current_kernel], wait_report)

        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(
            result["comparisons"]["wait_block_structure"])

    def test_semantic_fingerprint_uses_pc_and_mask_histogram(self):
        instruction = captured_instruction_wire(
            0x100, SIDECAR.CONTRACT_EVENT_EXCH,
            0x1ffffe, 0, 0)
        with tempfile.TemporaryDirectory() as root:
            traces = Path(root)
            kdir = (
                traces / "threadblocks/device_0/stream_0/kernel_5")
            kdir.mkdir(parents=True)
            write_single_warp_threadblock(
                kdir / "d_0_s_0_k_5_0,0,0.pb",
                [instruction, instruction])
            extra = traces / "extra_info"
            extra.mkdir()
            (extra / "enhanced_execution_info.json").write_text(
                json.dumps({
                    "kernels": [{
                        "unique_function_id": 3,
                        "kernel_name": "synthetic",
                        "instructions": [{
                            "pc_num_dec": 0x100,
                            "op_code": "SYNCS.EXCH.64",
                        }],
                    }],
                }))

            fingerprint = SIDECAR.semantic_trace_fingerprint(
                str(traces), [5])

        self.assertEqual(
            fingerprint["counts"]["semantic_instructions"], 2)
        self.assertEqual(
            fingerprint["counts"]["semantic_sites"], 1)
        self.assertEqual(fingerprint["counts"]["kernel_5_ctas"], 1)

    def test_end_to_end_capture_analysis_emits_unique_ticket(self):
        instructions = [
            captured_instruction_wire(
                0x100, SIDECAR.CONTRACT_EVENT_EXCH,
                0x1ffffe, 0, 0),
            captured_instruction_wire(
                0x110, SIDECAR.CONTRACT_EVENT_ARRIVE,
                256, 0, 1, ordinal=1),
            captured_instruction_wire(
                0x120, SIDECAR.CONTRACT_EVENT_WAIT,
                0, 1, 1),
        ]
        with tempfile.TemporaryDirectory() as root:
            traces = Path(root)
            kdir = (
                traces / "threadblocks/device_0/stream_0/kernel_5")
            kdir.mkdir(parents=True)
            write_single_warp_threadblock(
                kdir / "d_0_s_0_k_5_0,0,0.pb",
                instructions)
            extra = traces / "extra_info"
            extra.mkdir()
            (extra / "tma_completion_contracts.json").write_text(
                json.dumps({
                    "validation_complete": True,
                    "kernels": [{
                        "trace_kernel_id": 5,
                        "contracts": 1,
                        "expected_bytes": 256,
                        "assigned_bytes": 256,
                        "errors": [],
                    }],
                }))
            sync_sites = {
                (3, 0x100): {"sync_kind": "EXCH"},
                (3, 0x110): {"sync_kind": "ARRIVE_EXPECT_TX"},
                (3, 0x120): {"sync_kind": "TRYWAIT"},
            }
            sidecar_kernels = [{
                "trace_kernel_id": 5,
                "wait_blocks": [{
                    "runtime_strategy": "INJECT_MISSING_WAIT",
                    "unique_function_id": 3,
                    "entry_probe_pc": 0x120,
                }],
            }]

            report = SIDECAR.analyze_mbarrier_contract_capture(
                str(traces), {5: str(kdir)}, {5: 3},
                sync_sites, sidecar_kernels, "all")

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["summary"]["unique_assignments"], 1)
        self.assertEqual(
            report["tickets"][0]["target_release_generation"], 1)
        self.assertEqual(
            report["runtime_ticket_summary"],
            {"slots": 1, "exact_targets": 1, "fallback_slots": 0})
        self.assertEqual(
            report["runtime_tickets"][0][
                "target_release_generation"], 1)
        self.assertEqual(
            report["runtime_tickets"][0][
                "source_warp_instruction_index"], 2)
        self.assertEqual(
            report["runtime_tickets"][0]["runtime_strategy"],
            "INJECT_MISSING_WAIT")

    def test_fold_action_emits_exact_runtime_ticket(self):
        instructions = [
            captured_instruction_wire(
                0x100, SIDECAR.CONTRACT_EVENT_EXCH,
                0x1ffffe, 0, 0),
            captured_instruction_wire(
                0x110, SIDECAR.CONTRACT_EVENT_ARRIVE,
                256, 0, 1, ordinal=1),
            captured_instruction_wire(
                0x120, SIDECAR.CONTRACT_EVENT_WAIT,
                0, 1, 1),
            sync_only_instruction_wire(0x140, 0),
        ]
        with tempfile.TemporaryDirectory() as root:
            traces = Path(root)
            kdir = (
                traces / "threadblocks/device_0/stream_0/kernel_5")
            kdir.mkdir(parents=True)
            write_single_warp_threadblock(
                kdir / "d_0_s_0_k_5_0,0,0.pb",
                instructions)
            extra = traces / "extra_info"
            extra.mkdir()
            (extra / "tma_completion_contracts.json").write_text(
                json.dumps({
                    "validation_complete": True,
                    "kernels": [{
                        "trace_kernel_id": 5,
                        "contracts": 1,
                        "expected_bytes": 256,
                        "assigned_bytes": 256,
                        "errors": [],
                    }],
                }))
            sync_sites = {
                (3, 0x100): {"sync_kind": "EXCH"},
                (3, 0x110): {"sync_kind": "ARRIVE_EXPECT_TX"},
                (3, 0x120): {"sync_kind": "TRYWAIT"},
            }
            sidecar_kernels = [{
                "trace_kernel_id": 5,
                "wait_blocks": [{
                    "runtime_strategy": "FOLD_CAPTURED_POLL",
                    "unique_function_id": 3,
                    "poll_probe_pc": 0x120,
                    "poll_wait_pcs": [0x120],
                    "poll_loopback_pc": 0x130,
                    "poll_loop_exit_pc": 0x140,
                }],
            }]

            report = SIDECAR.analyze_mbarrier_contract_capture(
                str(traces), {5: str(kdir)}, {5: 3},
                sync_sites, sidecar_kernels, "all")

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["runtime_ticket_summary"],
            {"slots": 1, "exact_targets": 1, "fallback_slots": 0})
        ticket = report["runtime_tickets"][0]
        self.assertEqual(ticket["runtime_strategy"],
                         "FOLD_CAPTURED_POLL")
        self.assertEqual(ticket["target_release_generation"], 1)

    def test_count_only_wait_may_omit_contract_payload(self):
        instructions = [
            captured_instruction_wire(
                0x100, SIDECAR.CONTRACT_EVENT_EXCH,
                0x1ffffe, 0, 0),
            captured_instruction_wire(
                0x110, SIDECAR.CONTRACT_EVENT_ARRIVE,
                0, 0, 0),
            sync_only_instruction_wire(0x120, 0),
        ]
        with tempfile.TemporaryDirectory() as root:
            traces = Path(root)
            kdir = (
                traces / "threadblocks/device_0/stream_0/kernel_5")
            kdir.mkdir(parents=True)
            write_single_warp_threadblock(
                kdir / "d_0_s_0_k_5_0,0,0.pb",
                instructions)
            extra = traces / "extra_info"
            extra.mkdir()
            (extra / "tma_completion_contracts.json").write_text(
                json.dumps({
                    "validation_complete": True,
                    "kernels": [{
                        "trace_kernel_id": 5,
                        "contracts": 1,
                        "expected_bytes": 1,
                        "assigned_bytes": 1,
                        "errors": [],
                    }],
                }))
            sync_sites = {
                (3, 0x100): {"sync_kind": "EXCH"},
                (3, 0x110): {"sync_kind": "ARRIVE_EXPECT_TX"},
                (3, 0x120): {"sync_kind": "TRYWAIT"},
            }

            report = SIDECAR.analyze_mbarrier_contract_capture(
                str(traces), {5: str(kdir)}, {5: 3},
                sync_sites, [], "all")

        self.assertEqual(report["summary"]["missing_snapshots"], 0)
        self.assertEqual(report["summary"]["eligible_barriers"], 0)

    def test_tma_wait_without_contract_payload_fails_closed(self):
        instructions = [
            captured_instruction_wire(
                0x100, SIDECAR.CONTRACT_EVENT_EXCH,
                0x1ffffe, 0, 0),
            captured_instruction_wire(
                0x110, SIDECAR.CONTRACT_EVENT_ARRIVE,
                256, 0, 1, ordinal=1),
            sync_only_instruction_wire(0x120, 0),
        ]
        with tempfile.TemporaryDirectory() as root:
            traces = Path(root)
            kdir = (
                traces / "threadblocks/device_0/stream_0/kernel_5")
            kdir.mkdir(parents=True)
            write_single_warp_threadblock(
                kdir / "d_0_s_0_k_5_0,0,0.pb",
                instructions)
            extra = traces / "extra_info"
            extra.mkdir()
            (extra / "tma_completion_contracts.json").write_text(
                json.dumps({
                    "validation_complete": True,
                    "kernels": [{
                        "trace_kernel_id": 5,
                        "contracts": 1,
                        "expected_bytes": 256,
                        "assigned_bytes": 256,
                        "errors": [],
                    }],
                }))
            sync_sites = {
                (3, 0x100): {"sync_kind": "EXCH"},
                (3, 0x110): {"sync_kind": "ARRIVE_EXPECT_TX"},
                (3, 0x120): {"sync_kind": "TRYWAIT"},
            }
            sidecar_kernels = [{
                "trace_kernel_id": 5,
                "wait_blocks": [{
                    "runtime_strategy": "INJECT_MISSING_WAIT",
                    "unique_function_id": 3,
                    "entry_probe_pc": 0x120,
                }],
            }]

            report = SIDECAR.analyze_mbarrier_contract_capture(
                str(traces), {5: str(kdir)}, {5: 3},
                sync_sites, sidecar_kernels, "all")

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["summary"]["missing_snapshots"], 1)
        self.assertEqual(
            report["summary"]["rejected_tma_action_occurrences"], 1)
        self.assertEqual(
            report["runtime_ticket_summary"],
            {"slots": 1, "exact_targets": 0, "fallback_slots": 1})

    def test_device_capture_avoids_prior_deadlock_patterns(self):
        tracer_dir = MODULE_PATH.parent / "tracer_tool"
        inject_source = (tracer_dir / "inject_funcs.cu").read_text()
        begin = inject_source.index(
            "instrument_inst_compact_mbarrier")
        capture_end = inject_source.index(
            "void record_warp_exit", begin)
        capture_body = inject_source[begin:capture_end]

        self.assertNotIn(
            "record_mbarrier_contract_begin", inject_source)
        self.assertNotIn(
            "record_mbarrier_contract_end", inject_source)
        self.assertNotIn("cta_record_count_ptr", inject_source)
        self.assertIn(
            "INST_TRACE_HAS_MBARRIER_CONTRACT", capture_body)
        self.assertIn("channel_dev->push(&record", capture_body)
        self.assertIn("captured_semantic_raw", capture_body)
        self.assertIn(
            "instrument_inst_compact_mbarrier_exch", inject_source)
        self.assertIn(
            "instrument_inst_compact_mbarrier_arrive", inject_source)
        self.assertIn(
            "instrument_inst_compact_mbarrier_wait", inject_source)
        self.assertIn(
            "(static_cast<uint64_t>(op_reg_val_3) << 32)",
            capture_body)

        tracer_source = (tracer_dir / "tracer_tool.cu").read_text()
        self.assertIn(
            '"MBARRIER_CONTRACT_CAPTURE", 0, 0, 1',
            tracer_source)
        self.assertIn(
            "transport=existing_sync_record", tracer_source)
        self.assertIn("extra_callbacks=0", tracer_source)
        self.assertNotIn("mbarrier_contract_channel", tracer_source)
        self.assertNotIn(
            '"record_mbarrier_contract_begin"', tracer_source)
        self.assertNotIn(
            '"record_mbarrier_contract_end"', tracer_source)
        self.assertIn(
            "static_cast<uint32_t>(registry_addr >> 32)",
            tracer_source)
        self.assertIn(
            "MBARRIER_CONTRACT_ERROR_BARRIER_TABLE_OVERFLOW",
            inject_source)
        runner_source = (
            MODULE_PATH.parent / "run_hw_trace.py").read_text()
        self.assertIn(
            '"--mbarrier-contract-capture"', runner_source)
        self.assertIn(
            'if mbarrier_contract_capture:', runner_source)
        self.assertIn(
            '"-mbarrier-contract-v2"', runner_source)

    def test_simulator_exact_generation_covers_inject_and_fold(self):
        simulator_root = MODULE_PATH.parents[2] / "gpu-simulator"
        sm_source = (
            simulator_root /
            "gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc"
        ).read_text()
        loader_source = (
            simulator_root /
            "gpgpu-sim/src/gpgpu-sim/gpu-sim.cc"
        ).read_text()

        self.assertIn(
            "(has_missing_action || has_poll_action)", sm_source)
        self.assertIn(
            "wait_ready = cur_gen >= action_expected_gen", sm_source)
        self.assertIn(
            "held.expected_gen = action_expected_gen", sm_source)
        self.assertIn(
            "held.uses_exact_generation_ticket =",
            sm_source)
        self.assertIn(
            "!pending_wait.uses_exact_generation_ticket &&",
            sm_source)
        self.assertIn(
            'strategy != "INJECT_MISSING_WAIT" &&',
            loader_source)
        self.assertIn(
            'strategy != "FOLD_CAPTURED_POLL"', loader_source)


class WaitBlockFoldProofTest(unittest.TestCase):
    def test_control_requires_primary_wait_predicate(self):
        candidate = region()
        by_pc = {
            0x100: wait(0x100, trywait=True),
            0x110: instruction("NANOSLEEP.SYNCS", [operand("1")]),
            0x120: wait(0x120),
            0x130: instruction(
                "BRA", [operand("0x100", 0x100)],
                predicate=3, negate=True),
        }

        safe, reason, _ = SIDECAR.prove_poll_loop_control(candidate, by_pc)

        self.assertFalse(safe)
        self.assertEqual(reason, "POLL_LOOPBACK_NOT_OWNED_BY_PRIMARY_WAIT")

    def test_wait_condition_requires_exact_barrier_and_state(self):
        candidate = region()
        by_pc = {
            0x100: wait(0x100, barrier="[R1+0x10]", state="R2", trywait=True),
            0x120: wait(0x120, barrier="[R9+0x20]", state="R3"),
        }

        safe, reason, _ = SIDECAR.prove_poll_wait_condition(candidate, by_pc)

        self.assertFalse(safe)
        self.assertEqual(reason, "POLL_WAIT_CONDITION_MISMATCH")

    def test_effect_rejects_non_idempotent_imad(self):
        by_pc = {
            0x100: instruction(
                "IMAD.U32",
                [operand("R7"), operand("R7"), operand("0x2"), operand("R7")],
            )
        }
        candidate = {
            "allowed_block_pcs": [0x100],
            "loopback_bra_pc": 0x110,
        }

        safe, reason, _ = SIDECAR.prove_effects(1, candidate, by_pc)

        self.assertFalse(safe)
        self.assertEqual(reason, SIDECAR.EFFECT_UNSAFE)

    def test_scoreboard_proof_includes_loopback_branch(self):
        by_pc = {
            0x100: instruction(
                "SYNCS.PHASECHK.TRANS64",
                [operand("P0"), operand("[R1]"), operand("R2")],
                control_bits={
                    "is_new_read_barrier": True,
                    "id_new_read_barrier": 2,
                    "wait_barrier_bits": 1 << 2,
                },
            ),
            0x110: instruction(
                "BRA", [operand("0x100", 0x100)],
                predicate=0, negate=True,
                control_bits={"wait_barrier_bits": 1 << 5},
            ),
        }

        safe, proof = SIDECAR.compute_barrier_self_tied(
            [0x100, 0x110], by_pc)

        self.assertFalse(safe)
        self.assertEqual(proof["wait_on_externally_armed_ids"], [5])

    def test_reviewed_fold_shape_passes(self):
        candidate = region()
        by_pc = {
            0x100: wait(0x100, trywait=True),
            0x110: instruction("NANOSLEEP.SYNCS", [operand("1")]),
            0x120: wait(0x120),
            0x130: instruction(
                "BRA", [operand("0x100", 0x100)],
                predicate=0, negate=True),
        }
        effect_input = {
            "allowed_block_pcs": candidate["body_pcs"] + [0x130],
            "loopback_bra_pc": 0x130,
        }

        self.assertTrue(
            SIDECAR.prove_poll_loop_control(candidate, by_pc)[0])
        self.assertTrue(
            SIDECAR.prove_poll_wait_condition(candidate, by_pc)[0])
        self.assertTrue(SIDECAR.prove_effects(1, effect_input, by_pc)[0])


class PartialJoinProofTest(unittest.TestCase):
    ENTRY = 0x100
    RESUME = 0x130
    JOIN = 0x200
    POST_JOIN = 0x210
    FULL = 0xF
    BLOCKED = 0x1
    SIBLING = 0xE

    def setUp(self):
        self.candidate = {
            "resume_pc": self.RESUME,
            "not_ready_branch_pc": 0x120,
        }
        self.by_pc = {
            0x80: instruction(
                "BRA", [operand("0x200", self.JOIN)],
                predicate=1, negate=True),
            self.ENTRY: wait(self.ENTRY),
            0x110: instruction(
                "IMAD.MOV.U32",
                [operand("R1"), operand("RZ"), operand("RZ"), operand("R2")]),
            0x120: instruction(
                "BRA", [operand("0x300", 0x300)],
                predicate=0, negate=True),
            self.RESUME: instruction(
                "MOV", [operand("R3"), operand("R4")]),
            0x140: instruction(
                "UTMALDG.4D", [operand("R5"), operand("R6")]),
            self.JOIN: instruction("WARPSYNC.ALL", []),
            self.POST_JOIN: instruction(
                "MOV", [operand("R7"), operand("R8")]),
        }

    def _pending_trace(self):
        return [
            dynamic_instruction(0x80, self.FULL, self.SIBLING),
            dynamic_instruction(self.ENTRY, self.BLOCKED),
            dynamic_instruction(0x110, self.BLOCKED),
            dynamic_instruction(0x120, self.BLOCKED, 0),
            dynamic_instruction(self.RESUME, self.BLOCKED),
            dynamic_instruction(0x140, self.BLOCKED),
            dynamic_instruction(self.JOIN, self.BLOCKED),
            dynamic_instruction(self.JOIN, self.SIBLING),
            dynamic_instruction(self.POST_JOIN, self.FULL),
        ]

    def test_warpsync_join_pending_visit_passes(self):
        insts = self._pending_trace()
        ok, reason, proof = SIDECAR.prove_partial_join_visit(
            self.candidate, insts, self.by_pc, 1, 1,
            [self.FULL] * len(insts))

        self.assertTrue(ok, reason)
        self.assertTrue(proof["join_pending_at_entry"])
        self.assertEqual(proof["static"]["join_pc"], self.JOIN)
        self.assertEqual(proof["static"]["join_kind"], "WARPSYNC")

    def test_join_already_issued_visit_passes(self):
        insts = [
            dynamic_instruction(0x80, self.FULL, self.SIBLING),
            dynamic_instruction(self.JOIN, self.SIBLING),
            dynamic_instruction(self.ENTRY, self.BLOCKED),
            dynamic_instruction(0x120, self.BLOCKED, 0),
            dynamic_instruction(self.RESUME, self.BLOCKED),
            dynamic_instruction(self.JOIN, self.BLOCKED),
            dynamic_instruction(self.POST_JOIN, self.FULL),
        ]

        ok, reason, proof = SIDECAR.prove_partial_join_visit(
            self.candidate, insts, self.by_pc, 1, 2,
            [self.FULL] * len(insts))

        self.assertTrue(ok, reason)
        self.assertFalse(proof["join_pending_at_entry"])

    def test_sibling_semantic_work_before_join_is_rejected(self):
        self.by_pc[0x90] = instruction(
            "UTMALDG.4D", [operand("R5"), operand("R6")])
        insts = self._pending_trace()
        insts.insert(1, dynamic_instruction(0x90, self.SIBLING))

        ok, reason, _ = SIDECAR.prove_partial_join_visit(
            self.candidate, insts, self.by_pc, 1, 2,
            [self.FULL] * len(insts))

        self.assertFalse(ok)
        self.assertEqual(reason, SIDECAR.PARTIAL_SIBLING_NOT_QUIESCENT)

    def test_missing_not_ready_branch_occurrence_is_rejected(self):
        insts = [
            inst for inst in self._pending_trace()
            if inst["pc"] != 0x120
        ]

        ok, reason, _ = SIDECAR.prove_partial_join_visit(
            self.candidate, insts, self.by_pc, 1, 1,
            [self.FULL] * len(insts))

        self.assertFalse(ok)
        self.assertEqual(
            reason, SIDECAR.PARTIAL_ENTRY_BRANCH_OCCURRENCE_MISSING)

    def test_call_inside_capsule_is_rejected(self):
        self.by_pc[0x140] = instruction(
            "CALL", [operand("0x400", 0x400)])
        insts = self._pending_trace()

        ok, reason, _ = SIDECAR.prove_partial_join_visit(
            self.candidate, insts, self.by_pc, 1, 1,
            [self.FULL] * len(insts))

        self.assertFalse(ok)
        self.assertEqual(reason, SIDECAR.PARTIAL_CAPSULE_CONTROL_UNSAFE)

    def test_bsync_requires_matching_bssy_target(self):
        self.by_pc[0x70] = instruction(
            "BSSY",
            [operand("B0"), operand("0x210", self.POST_JOIN)])
        self.by_pc[self.JOIN] = instruction("BSYNC", [operand("B0")])

        ok, reason, proof = SIDECAR.prove_partial_join_static(
            self.by_pc, 0x80, self.JOIN, self.POST_JOIN)

        self.assertTrue(ok, reason)
        self.assertEqual(proof["bssy_pc"], 0x70)
        self.assertEqual(proof["join_token"], "B0")

        self.by_pc[0x70] = instruction(
            "BSSY", [operand("B0"), operand("0x220", 0x220)])
        ok, reason, _ = SIDECAR.prove_partial_join_static(
            self.by_pc, 0x80, self.JOIN, self.POST_JOIN)
        self.assertFalse(ok)
        self.assertEqual(reason, SIDECAR.PARTIAL_JOIN_STATIC_UNSAFE)

    def test_partial_record_serializes_deferred_contract(self):
        candidate = {
            "block_id": "wb_fid1_e0x100",
            "unique_function_id": 1,
            "entry_wait_pc": self.ENTRY,
            "entry_branch_pc": 0x120,
            "not_ready_branch_pc": 0x120,
            "resume_pc": self.RESUME,
            "mainline_setup_pcs": [0x110],
            "entry_arming_pcs": [0x110, 0x120],
            "fallback_entry_pc": 0x300,
            "fallback_wait_pcs": [0x320],
            "fallback_loopback_pc": 0x330,
            "fallback_exit_pc": 0x340,
            "canonical_wait_condition": {
                "barrier_offset": "0x0",
                "wait_state_operands": ["R2"],
            },
            "dynamic": {
                "path_A_first_gate_ready": 1,
                "path_B_second_gate_ready": 0,
                "path_C_fallback_loop": 0,
                "active_entry_visits": 1,
                "predicated_off_entry_occurrences": 0,
                "all_live_entry_visits": 0,
                "partial_entry_visits": 1,
                "entry_effective_masks": {"0x1": 1},
                "entry_live_masks": {"0xf": 1},
                "entry_barrier_addrs": {"0x1000": 1},
                "entry_wait_states": {"0x0": 1},
                "entry_sync_payload_missing": 0,
            },
            "partial_join_proof": {
                "region": {
                    "sibling_branch_pc": 0x80,
                    "sibling_branch_target_pc": self.JOIN,
                    "join_pc": self.JOIN,
                    "post_join_pc": self.POST_JOIN,
                    "join_kind": "WARPSYNC",
                    "bssy_pc": None,
                    "join_token": None,
                },
                "sibling_masks": {"0xe": 1},
                "join_pending_visits": 1,
                "join_already_issued_visits": 0,
            },
        }
        sync_sites = {
            (1, self.ENTRY): {
                "sync_kind": "TRYWAIT",
                "opcode": "SYNCS.PHASECHK.TRANS64.TRYWAIT",
                "barrier_operand_text": "[R1]",
            }
        }

        record = SIDECAR.make_inject_missing_wait_record(
            7, candidate, sync_sites, self.by_pc)

        self.assertEqual(
            record["partial_modeled_miss_action"],
            "DEFER_BLOCKED_SUFFIX_THEN_HOLD")
        self.assertEqual(record["partial_hold_proof_version"], 1)
        self.assertEqual(
            record["partial_region_id"], "pr_k7_fid1_b0x80_j0x200")
        self.assertEqual(record["entry_branch_pc"], 0x120)
        self.assertEqual(record["join_pc"], self.JOIN)
        self.assertTrue(record["linear_capsule_safe"])


if __name__ == "__main__":
    unittest.main()
