import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_build_spin_wait_sidecar import (
    SIDECAR, dynamic_instruction, instruction, operand, wait,
)


class WaitOutcomeWitnessTest(unittest.TestCase):
    def setUp(self):
        self.byp = {
            0x100: wait(0x100, trywait=True),
            0x110: instruction("R2UR", [operand("UR4"), operand("R2")]),
            0x120: instruction(
                "BRA", [operand("0x100", 0x100)], predicate=0, negate=True),
            0x130: instruction("NOP", []),
        }
        self.sites = {(1, 0x100): {"sync_kind": "TRYWAIT"}}
        self.insts = [
            dynamic_instruction(0x100, 0xf),
            dynamic_instruction(0x110, 0xf),
            dynamic_instruction(0x120, 0xf, 0),
            dynamic_instruction(0x130, 0xf),
        ]

    def specs(self):
        return SIDECAR.build_wait_outcome_specs({1: self.byp}, self.sites)

    def outcome(self, insts=None):
        return SIDECAR.captured_wait_outcome(
            self.insts if insts is None else insts, 0, self.specs()[(1, 0x100)])

    def test_ready_fallthrough_and_not_ready_backedge(self):
        self.assertTrue(self.outcome())
        self.insts[2]["predicate_mask"] = 0xf
        self.insts[3]["pc"] = 0x100
        self.assertFalse(self.outcome())

    def test_positive_predicate_branch_and_uniform_predicate(self):
        self.byp[0x100]["operands"][0] = operand("UP2")
        branch = self.byp[0x120]
        branch.update(predicate_register=2, is_uniform_predicate=True,
                      is_predicate_negate=False)
        self.assertFalse(self.outcome())
        branch["is_uniform_predicate"] = False
        self.assertEqual(self.specs(), {})

    def test_predicate_write_in_any_operand_stops_the_proof(self):
        for op, operands in (
            ("ISETP.NE.AND", ["P0", "PT", "R1", "R2", "PT"]),
            ("IADD3", ["R1", "P0", "R2", "R3", "RZ"]),
            ("LEA", ["R1", "P0", "R2", "R3", "0x1"]),
        ):
            with self.subTest(opcode=op):
                self.byp[0x110] = instruction(op, list(map(operand, operands)))
                self.assertEqual(self.specs(), {})

    def test_unknown_opcode_or_other_control_stops_the_proof(self):
        for opcode in ("UNKNOWN", "CALL", "BSYNC"):
            with self.subTest(opcode=opcode):
                self.byp[0x110] = instruction(opcode, [])
                self.assertEqual(self.specs(), {})

    def test_absent_branch_wrong_edge_and_predicate_mask_are_not_ready(self):
        self.assertIsNone(self.outcome(self.insts[:2] + self.insts[3:]))
        self.assertIsNone(self.outcome(self.insts[:3]))
        self.insts[3]["pc"] = 0x100
        self.assertIsNone(self.outcome())
        self.insts[3]["pc"] = 0x130
        self.insts[2]["predicate_mask"] = 0x1
        self.assertIsNone(self.outcome())

    def test_unexecuted_wait_and_partial_branch_coverage_are_unknown(self):
        self.insts[0]["predicate_mask"] = 0
        self.assertIsNone(self.outcome())
        self.insts[0]["predicate_mask"] = 0xf
        self.insts[2]["active_mask"] = 0x1
        self.assertIsNone(self.outcome())

    def test_disjoint_lane_work_does_not_supply_the_branch_result(self):
        for inst in self.insts:
            inst["active_mask"] = 0x1
            inst["predicate_mask"] &= 0x1
        self.insts.insert(1, dynamic_instruction(0x120, 0xe))
        self.assertTrue(self.outcome())
        self.insts[-1]["function_unique_id"] = 2
        self.assertIsNone(self.outcome())

    def test_lifecycle_uses_a_ready_witness_without_creating_a_runtime_action(self):
        def event(pc, kind, before, after, semantic, ordinal=None):
            inst = dynamic_instruction(pc, 0xf)
            inst.update(
                sync_valid=True, barrier_addr=0x1000, semantic_raw=semantic,
                contract_capture_valid=True, contract_event_kind=kind,
                contract_barrier_addr=0x1000,
                contract_epoch_before=0 if kind == SIDECAR.CONTRACT_EVENT_EXCH else 1,
                contract_epoch_after=1, contract_opened_before=before,
                contract_opened_after=after, contract_has_ordinal=ordinal is not None,
                contract_ordinal=ordinal, contract_capture_launch_uid=1,
                contract_has_semantic_raw=True, contract_semantic_raw=semantic,
            )
            return inst

        insts = [
            event(0x80, SIDECAR.CONTRACT_EVENT_EXCH, 0, 0, 0x1ffffe),
            event(0x90, SIDECAR.CONTRACT_EVENT_ARRIVE, 0, 1, 256, 1),
            event(0x100, SIDECAR.CONTRACT_EVENT_WAIT, 1, 1, 0),
            *self.insts[1:],
            event(0x90, SIDECAR.CONTRACT_EVENT_ARRIVE, 1, 2, 256, 2),
            event(0x200, SIDECAR.CONTRACT_EVENT_WAIT, 2, 2, 0x80000000),
        ]
        sites = {
            **self.sites,
            (1, 0x80): {"sync_kind": "EXCH"},
            (1, 0x90): {"sync_kind": "ARRIVE_EXPECT_TX"},
            (1, 0x200): {"sync_kind": "TRYWAIT"},
        }
        selected = [{
            "trace_kernel_id": 5,
            "wait_blocks": [{
                "runtime_strategy": "INJECT_MISSING_WAIT",
                "unique_function_id": 1,
                "entry_probe_pc": 0x200,
            }],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d_0_s_0_k_5_0,0,0.pb"
            path.touch()
            with mock.patch.object(SIDECAR, "vpb_parse_full", return_value=[(0, insts)]), \
                    mock.patch.object(SIDECAR, "_load_tma_ownership_status",
                                      return_value={5: True}):
                report = SIDECAR.analyze_mbarrier_contract_capture(
                    directory, {5: directory}, {5: 1}, sites, selected, "all",
                    static_kernels={1: self.byp})
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["summary"]["independent_ready_witnesses"], 1)
                self.assertEqual(len(report["runtime_tickets"]), 1)
                self.assertEqual(report["runtime_tickets"][0]["action_pc"], 0x200)
                self.assertEqual(
                    report["runtime_tickets"][0]["target_release_generation"], 2)

                insts[2]["contract_epoch_after"] = 2
                rejected = SIDECAR.analyze_mbarrier_contract_capture(
                    directory, {5: directory}, {5: 1}, sites, selected, "all",
                    static_kernels={1: self.byp})
                self.assertEqual(rejected["status"], "FAIL")


class ArrivedWarpSyncProofTest(unittest.TestCase):
    def setUp(self):
        self.byp = {
            0x70: instruction("BSSY", [operand("B0"), operand("0x190", 0x190)]),
            0x80: instruction("BRA", [operand("0x180", 0x180)], predicate=1),
            0x100: wait(0x100, trywait=True),
            0x120: instruction("BRA", [operand("0x400", 0x400)],
                               predicate=0, negate=True),
            0x130: instruction("UTMALDG.4D", [operand("[UR4]")]),
            0x180: instruction("BSYNC", [operand("B0")]),
            0x190: instruction("IADD3", [operand("R1"), operand("R2")]),
            0x300: instruction("WARPSYNC.ALL", []),
            0x310: instruction("NOP", []),
        }
        self.cand = {"resume_pc": 0x130, "not_ready_branch_pc": 0x120}
        self.insts = [
            dynamic_instruction(0x80, 0xf, 0xe),
            dynamic_instruction(0x180, 0xe),
            dynamic_instruction(0x190, 0xe),
            dynamic_instruction(0x300, 0xe),
            dynamic_instruction(0x100, 0x1),
            dynamic_instruction(0x120, 0x1, 0),
            dynamic_instruction(0x130, 0x1),
            dynamic_instruction(0x180, 0x1),
            dynamic_instruction(0x190, 0x1),
            dynamic_instruction(0x300, 0x1),
            dynamic_instruction(0x310, 0xf),
        ]

    def prove(self, entry_index=4):
        return SIDECAR.prove_partial_join_visit(
            self.cand, self.insts, self.byp, 1, entry_index,
            [0xf] * len(self.insts))

    def test_outer_warpsync_proves_already_arrived_sibling(self):
        ok, reason, proof = self.prove()
        self.assertTrue(ok, reason)
        self.assertEqual(proof["static"]["proof_version"], 2)
        self.assertEqual(proof["static"]["join_pc"], 0x300)
        self.assertEqual(proof["sibling_join_index"], 3)
        self.assertEqual(proof["post_join_index"], 10)
        self.assertFalse(proof["join_pending_at_entry"])
        self.assertIsNone(proof["static"]["sibling_branch_pc"])

    def test_semantic_work_before_arrival_is_not_replayed_or_changed(self):
        self.insts.insert(3, dynamic_instruction(0x130, 0xe))
        original = [dict(inst) for inst in self.insts]
        ok, reason, proof = self.prove(5)
        self.assertTrue(ok, reason)
        self.assertEqual(proof["sibling_join_index"], 4)
        self.assertEqual(self.insts, original)

    def test_sibling_work_after_arrival_is_rejected_even_if_predicated_off(self):
        for predicate in (0, 0xe):
            with self.subTest(predicate=predicate):
                added = dynamic_instruction(0x190, 0xe, predicate)
                self.insts.insert(6, added)
                self.assertFalse(self.prove()[0])
                self.insts.pop(6)

    def test_intermediate_bsync_is_not_an_arrived_full_warp_join(self):
        self.byp[0x300] = instruction("BSYNC", [operand("B0")])
        self.assertFalse(self.prove()[0])

    def test_missing_or_duplicate_sibling_join_is_rejected(self):
        sibling = self.insts.pop(3)
        self.assertFalse(self.prove(3)[0])
        self.insts.insert(3, sibling)
        self.insts.insert(3, dict(sibling))
        self.assertFalse(self.prove(5)[0])

    def test_completed_prior_iteration_is_not_reused(self):
        self.insts.insert(4, dynamic_instruction(0x310, 0xf))
        self.assertFalse(self.prove(5)[0])

    def test_repeated_pcs_pair_with_the_current_dynamic_join(self):
        self.insts = self.insts + self.insts
        ok, reason, proof = self.prove(15)
        self.assertTrue(ok, reason)
        self.assertEqual(proof["sibling_join_index"], 14)
        self.assertEqual(proof["blocked_join_index"], 20)
        self.assertEqual(proof["post_join_index"], 21)

    def test_wrong_post_join_mask_or_pc_is_rejected(self):
        for key, value in (("active_mask", 1), ("pc", 0x190)):
            with self.subTest(key=key):
                previous = self.insts[-1][key]
                self.insts[-1][key] = value
                self.assertFalse(self.prove()[0])
                self.insts[-1][key] = previous

    def test_unsafe_blocked_path_or_foreign_function_is_rejected(self):
        self.byp[0x130] = instruction("CALL.REL", [])
        self.assertFalse(self.prove()[0])
        self.byp[0x130] = instruction("NOP", [])
        self.insts[6]["function_unique_id"] = 2
        self.assertFalse(self.prove()[0])

    def test_branch_omission_and_inactive_join_are_rejected(self):
        branch = self.insts.pop(5)
        self.assertFalse(self.prove()[0])
        self.insts.insert(5, branch)
        self.insts[3]["predicate_mask"] = 0
        self.assertFalse(self.prove()[0])

    def test_one_site_can_use_distinct_joins_in_distinct_occurrences(self):
        direct = [
            dynamic_instruction(0x80, 0xf, 0xe),
            dynamic_instruction(0x100, 1),
            dynamic_instruction(0x120, 1, 0),
            dynamic_instruction(0x130, 1),
            dynamic_instruction(0x180, 1),
            dynamic_instruction(0x180, 0xe),
            dynamic_instruction(0x190, 0xf),
        ]
        original = self.insts
        self.insts = original + direct
        candidate = {
            **self.cand, "block_id": "mixed", "entry_wait_pc": 0x100,
            "dynamic": {"partial_entry_visits": 2},
        }
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "d_0_s_0_k_5_0,0,0.pb").touch()
            with mock.patch.object(
                    SIDECAR, "vpb_parse_full", return_value=[(0, self.insts)]):
                stats, _ = SIDECAR.validate_partial_join_dynamic(
                    directory, [candidate], self.byp, "all", 1)
        proof = stats["mixed"]
        self.assertTrue(proof["dynamic_ok"], proof["reason"])
        self.assertEqual(
            {r["join_pc"] for r in proof["occurrence_regions"]}, {0x180, 0x300})
        self.assertEqual(proof["join_pending_visits"], 1)
        self.assertEqual(proof["join_already_issued_visits"], 1)
        self.assertEqual([o["source_index"] for o in proof["occurrences"]], [4, 12])
        self.assertEqual(
            [o["sibling_join_index"] for o in proof["occurrences"]], [3, 16])

    def test_failed_partial_join_reports_exact_dynamic_occurrence(self):
        self.insts.pop(9)
        self.insts.pop(7)
        candidate = {
            **self.cand, "block_id": "missing-join", "entry_wait_pc": 0x100,
            "dynamic": {"partial_entry_visits": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d_0_s_0_k_5_0,0,0.pb"
            path.touch()
            with mock.patch.object(
                    SIDECAR, "vpb_parse_full", return_value=[(0, self.insts)]):
                stats, _ = SIDECAR.validate_partial_join_dynamic(
                    directory, [candidate], self.byp, "all", 1)
        failure = stats["missing-join"]
        self.assertFalse(failure["dynamic_ok"])
        self.assertEqual(failure["reason"],
                         SIDECAR.PARTIAL_JOIN_OCCURRENCE_MISSING)
        self.assertEqual(failure["failure_detail"], {
            "cta": [0, 0, 0],
            "warp_id": 0,
            "source_index": 4,
            "entry_pc": "0x100",
            "active_mask": "0x1",
            "predicate_mask": "0x1",
            "effective_mask": "0x1",
            "live_mask": "0xf",
            "barrier_addr": None,
            "wait_state_raw": None,
        })

    def test_simulator_rejects_missing_required_join_before_extraction(self):
        root = Path(__file__).resolve().parents[3]
        source = (root / "gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc").read_text()
        handler = source.split("void SM::handle_partial_wait_instruction_issue", 1)[1]
        guard = handler.index("partial-already-arrived-join-not-issued")
        extraction = handler.index("tw->extract_partial_wait_capsule")
        self.assertLess(guard, extraction)
        loader = (root / "gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc").read_text()
        self.assertIn("region.proof_version == 2 && sibling_index >= entry_index", loader)
        self.assertIn("invalid partial occurrence masks or ordering", loader)
        self.assertIn("existing.proof_version != region.proof_version", loader)


class CollectiveWarpSyncProofTest(unittest.TestCase):
    def setUp(self):
        self.fixture = ArrivedWarpSyncProofTest()
        self.fixture.setUp()
        self.byp = self.fixture.byp
        self.insts = self.fixture.insts
        self.byp[0x2f0] = instruction(
            "IMAD.MOV.U32", [
                operand("R8"), operand("RZ"), operand("RZ"),
                operand("-0x1", float(2**64)),
            ])
        self.byp[0x300] = instruction(
            "WARPSYNC.COLLECTIVE", [operand("R8"), operand("0x330", 0x330)])
        self.byp[0x310] = instruction("SHFL.IDX", [operand("R1"), operand("R2")])
        self.byp[0x320] = instruction("ENDCOLLECTIVE", [])
        self.byp[0x330] = instruction("BSYNC", [operand("B0")])
        self.insts.insert(3, dynamic_instruction(0x2f0, 0xe))
        self.insts.insert(10, dynamic_instruction(0x2f0, 1))
        self.insts.extend([
            dynamic_instruction(0x320, 0xf),
            dynamic_instruction(0x330, 1),
        ])

    def prove(self):
        return self.fixture.prove(5)

    def test_full_mask_collective_preserves_shuffle_and_scope_exit(self):
        before = copy.deepcopy(self.insts)
        ok, reason, proof = self.prove()
        self.assertTrue(ok, reason)
        self.assertEqual(proof["static"]["proof_version"], 2)
        self.assertEqual(proof["post_join_index"], 12)
        self.assertEqual(proof["static"]["post_join_pc"], 0x310)
        self.assertEqual(self.insts, before)
        self.assertEqual(self.insts[-1]["active_mask"], 1)

    def test_unknown_or_partial_member_mask_is_rejected(self):
        for value in ("R9", "0x1", "0xfffffffe", "0x100000000"):
            with self.subTest(mask=value):
                self.byp[0x2f0]["operands"][3] = operand(value)
                self.assertFalse(self.prove()[0])

    def test_same_constant_must_define_the_consumed_mask_register(self):
        self.byp[0x2f0]["operands"][0] = operand("R9")
        self.assertFalse(self.prove()[0])
        self.byp[0x2f0]["operands"][0] = operand("R8")
        self.byp[0x2f0]["is_predicated"] = True
        self.assertFalse(self.prove()[0])

    def test_each_lane_group_must_actually_execute_the_mask_definition(self):
        for index in (3, 10):
            with self.subTest(index=index):
                saved = self.insts[index]["predicate_mask"]
                self.insts[index]["predicate_mask"] = 0
                self.assertFalse(self.prove()[0])
                self.insts[index]["predicate_mask"] = saved

    def test_wrong_collective_target_or_missing_end_is_rejected(self):
        self.byp[0x300]["operands"][1] = operand("0x310", 0x310)
        self.assertFalse(self.prove()[0])
        self.byp[0x300]["operands"][1] = operand("0x330", 0x330)
        self.byp[0x320]["op_code"] = "NOP"
        self.assertFalse(self.prove()[0])

    def test_unproven_collective_body_is_rejected(self):
        for opcode in ("BRA", "STG.E", "WARPSYNC.ALL"):
            with self.subTest(opcode=opcode):
                self.byp[0x310]["op_code"] = opcode
                self.assertFalse(self.prove()[0])

    def test_dynamic_end_scope_and_full_participation_are_required(self):
        saved = self.insts[13]
        self.insts[13] = dynamic_instruction(0x320, 1)
        self.assertFalse(self.prove()[0])
        self.insts[13] = saved
        self.insts.pop(13)
        self.assertFalse(self.prove()[0])

    def test_collective_join_does_not_bypass_legacy_direct_join_proof(self):
        self.byp[0x80]["operands"][0] = operand("0x300", 0x300)
        ok, reason, _ = SIDECAR.prove_partial_join_static(
            self.byp, 0x80, 0x300, 0x310)
        self.assertFalse(ok)
        self.assertEqual(reason, SIDECAR.PARTIAL_JOIN_STATIC_UNSAFE)


class PartialBindingConsistencyTest(unittest.TestCase):
    def setUp(self):
        block = {
            "partial_hold_proof_version": 3,
            "partial_join_variants": [{
                "proof_version": 2, "sibling_branch_pc": None,
                "join_pc": 0x300, "post_join_pc": 0x310,
                "join_kind": "WARPSYNC", "bssy_pc": None, "join_token": None,
            }],
            "partial_join_occurrences": [{
                "variant": 0, "cta": [0, 0, 0], "warp_id": 0,
                "source_index": 10, "sibling_join_index": 5,
                "blocked_join_index": 30, "post_join_index": 31,
                "active_mask": 1, "predicate_mask": 1,
                "sibling_mask": 14, "live_mask": 15,
            }],
        }
        second = copy.deepcopy(block)
        second["partial_join_occurrences"][0]["source_index"] = 20
        self.kernels = [{
            "trace_kernel_id": 5, "unique_function_id": 3,
            "wait_blocks": [block, second],
        }]

    def test_two_waits_may_share_the_same_join_occurrences(self):
        SIDECAR.validate_partial_occurrence_bindings(self.kernels)

    def test_conflicting_sibling_mask_is_rejected(self):
        self.kernels[0]["wait_blocks"][1]["partial_join_occurrences"][0]["sibling_mask"] = 12
        with self.assertRaisesRegex(ValueError, "conflicting partial join"):
            SIDECAR.validate_partial_occurrence_bindings(self.kernels)

    def test_legacy_and_occurrence_join_collision_is_rejected(self):
        self.kernels[0]["wait_blocks"].append({
            "partial_hold_proof_version": 1,
            "join_pc": 0x300, "post_join_pc": 0x310,
        })
        with self.assertRaisesRegex(ValueError, "mixes legacy"):
            SIDECAR.validate_partial_occurrence_bindings(self.kernels)


if __name__ == "__main__":
    unittest.main()


# Revision History
# 2026-09-24: require failed partial-join proofs to identify the exact dynamic
# CTA, warp, source index, masks, and wait payload occurrence.
