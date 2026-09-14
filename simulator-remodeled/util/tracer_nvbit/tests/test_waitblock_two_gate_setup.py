import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_build_spin_wait_sidecar import (
    SIDECAR, dynamic_instruction, instruction, operand, wait,
)


def register_copy(destination="R2", source="UR2"):
    return instruction(
        "IMAD.U32", list(map(operand, [destination, "RZ", "RZ", source])))


class TwoGateSetupTest(unittest.TestCase):
    def setUp(self):
        self.byp = {
            0xe0: register_copy(),
            0xf0: instruction("NOP", []),
            0x100: wait(0x100, trywait=True),
            0x110: instruction("BRA", [operand("0x150", 0x150)], predicate=0),
            0x120: register_copy(),
            0x130: wait(0x130, trywait=True),
            0x140: instruction(
                "BRA", [operand("0x200", 0x200)], predicate=0, negate=True),
            0x150: instruction("NOP", []),
            0x200: wait(0x200, trywait=True),
            0x210: instruction("NANOSLEEP.SYNCS", [operand("0x1", 1)]),
            0x220: wait(0x220),
            0x230: instruction(
                "BRA", [operand("0x200", 0x200)], predicate=0, negate=True),
            0x240: instruction("BRA", [operand("0x150", 0x150)]),
        }
        self.sites = {
            (1, pc): {
                "sync_kind": "TRYWAIT" if "TRYWAIT" in inst["op_code"] else "PHASECHK",
                "opcode": inst["op_code"],
                "barrier_operand_text": "[R1]",
            }
            for pc, inst in self.byp.items()
            if inst["op_code"].startswith("SYNCS.")
        }
        self.insts = [
            dynamic_instruction(pc, 0xffffffff)
            for pc in (0xe0, 0xf0, 0x100, 0x110, 0x150)
        ]
        self.insts[2].update(
            sync_valid=True, sync_has_semantic_raw=True,
            barrier_addr=0x1000, semantic_raw=0)

    def candidates(self):
        return SIDECAR.detect_wait_block_candidates(1, self.byp, self.sites)

    def candidate(self):
        return next(c for c in self.candidates() if c["entry_wait_pc"] == 0x100)

    def validate(self, candidate=None):
        candidate = self.candidate() if candidate is None else candidate
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "d_0_s_0_k_5_0,0,0.pb").touch()
            with mock.patch.object(
                    SIDECAR, "vpb_parse_full", return_value=[(0, self.insts)]):
                stats, _ = SIDECAR.validate_wait_blocks_dynamic(
                    directory, [candidate], self.byp, {1: self.byp}, "all", 1)
        return stats[candidate["block_id"]]

    def test_reload_between_gates_is_proven_without_mutating_instructions(self):
        before = copy.deepcopy(self.byp)
        candidate = self.candidate()
        self.assertTrue(candidate["cfg_ok"], candidate.get("reason"))
        self.assertEqual(candidate["all_gate_wait_pcs"], [0x100, 0x130])
        self.assertEqual(candidate["gate_setup_pcs"], [0x120])
        self.assertEqual(candidate["not_ready_branch_pc"], 0x140)
        self.assertEqual(candidate["resume_pc"], 0x150)
        self.assertNotIn(0x120, candidate["allowed_block_pcs"])
        self.assertNotIn(0x130, [
            c["entry_wait_pc"] for c in self.candidates() if c["cfg_ok"]])
        self.assertEqual(self.byp, before)

    def test_ready_only_full_warp_emits_existing_inject_contract(self):
        candidate = self.candidate()
        self.assertTrue(candidate["cfg_ok"])
        stats = self.validate(candidate)
        self.assertTrue(stats["dynamic_ok"], stats["reason"])
        self.assertEqual(stats["path_A_first_gate_ready"], 1)
        self.assertEqual(stats["path_B_second_gate_ready"], 0)
        self.assertEqual(stats["path_C_fallback_loop"], 0)
        self.assertEqual(stats["all_live_entry_visits"], 1)
        candidate["dynamic"] = stats
        ok, reason, canonical = SIDECAR.prove_wait_condition(
            1, candidate, self.byp, self.sites)
        self.assertTrue(ok, reason)
        candidate["canonical_wait_condition"] = canonical
        record = SIDECAR.make_inject_missing_wait_record(
            5, candidate, self.sites, self.byp)
        self.assertEqual(record["runtime_strategy"], "INJECT_MISSING_WAIT")
        self.assertEqual(record["entry_probe_pc"], 0x100)
        self.assertEqual(record["entry_branch_pc"], 0x140)
        self.assertEqual(record["partial_modeled_miss_action"], "ABORT")
        self.assertEqual(record["observed_effective_masks"], {"0xffffffff": 1})

    def test_second_gate_ready_path_preserves_setup_and_is_not_ready_only(self):
        candidate = self.candidate()
        self.assertTrue(candidate["cfg_ok"])
        self.insts[3]["predicate_mask"] = 0
        self.insts[4:4] = [
            dynamic_instruction(0x120, 0xffffffff),
            dynamic_instruction(0x130, 0xffffffff),
            dynamic_instruction(0x140, 0xffffffff, 0),
        ]
        stats = self.validate(candidate)
        self.assertTrue(stats["dynamic_ok"], stats["reason"])
        self.assertEqual(stats["path_A_first_gate_ready"], 0)
        self.assertEqual(stats["path_B_second_gate_ready"], 1)

    def test_second_gate_or_setup_cannot_be_omitted(self):
        candidate = self.candidate()
        self.insts[3]["predicate_mask"] = 0
        self.insts[4:4] = [dynamic_instruction(0x120, 0xffffffff)]
        self.assertFalse(self.validate(candidate)["dynamic_ok"])
        self.insts[4:4] = [
            dynamic_instruction(0x130, 0xffffffff),
            dynamic_instruction(0x140, 0xffffffff, 0),
        ]
        self.assertFalse(self.validate(candidate)["dynamic_ok"])

    def test_interleaved_sibling_cannot_clobber_a_uniform_reload_source(self):
        for inst in self.insts:
            inst["active_mask"] = inst["predicate_mask"] = 1
        self.insts.insert(1, dynamic_instruction(0xf0, 0xfffffffe))
        self.assertFalse(self.validate()["dynamic_ok"])

    def test_missing_reload_definition_or_changed_source_is_rejected(self):
        for change in ("missing", "different-source", "clobbered-source",
                       "clobbered-state", "unknown-op", "control"):
            with self.subTest(change=change):
                self.setUp()
                if change == "missing":
                    self.byp[0xe0] = instruction("NOP", [])
                elif change == "different-source":
                    self.byp[0x120] = register_copy(source="UR3")
                elif change == "clobbered-source":
                    self.byp[0xf0] = instruction(
                        "UMOV", [operand("UR2"), operand("0x1", 1)])
                elif change == "clobbered-state":
                    self.byp[0xf0] = register_copy(source="UR3")
                elif change == "unknown-op":
                    self.byp[0xf0] = instruction("UNKNOWN", [])
                else:
                    self.byp[0xf0] = instruction(
                        "BRA", [operand("0x100", 0x100)])
                self.assertFalse(self.candidate()["cfg_ok"])

    def test_implicit_wide_destination_clobber_is_rejected(self):
        for opcode, destination in (("LDS.64", "R1"), ("HGMMA.64x64x16.F32.BF16", "R0")):
            with self.subTest(opcode=opcode):
                self.setUp()
                self.byp[0xf0] = instruction(
                    opcode, [operand(destination), operand("[R8]")])
                self.assertFalse(self.candidate()["cfg_ok"])

    def test_nonoverlapping_wide_operations_before_gate_are_preserved(self):
        for opcode in ("LDS.64", "HGMMA.64x64x16.F32.BF16"):
            with self.subTest(opcode=opcode):
                self.byp[0xf0] = instruction(
                    opcode, [operand("R8"), operand("[R10]")])
                self.assertTrue(self.candidate()["cfg_ok"])

    def test_predicated_or_effectful_setup_is_rejected(self):
        for replacement in (
            instruction("STG.E", [operand("[R8]"), operand("R2")]),
            instruction("SYNCS.ARRIVE.TRANS64", []),
            instruction("UNKNOWN", []),
            instruction("IMAD.U32", [
                operand("R2"), operand("R1"), operand("R1"), operand("UR2")]),
            instruction("IMAD.U32", [
                operand("P0"), operand("RZ"), operand("RZ"), operand("UR2")]),
        ):
            with self.subTest(opcode=replacement["op_code"]):
                self.byp[0x120] = replacement
                self.assertFalse(self.candidate()["cfg_ok"])
        self.byp[0x120] = register_copy()
        self.byp[0x120]["is_predicated"] = True
        self.assertFalse(self.candidate()["cfg_ok"])

    def test_different_wait_condition_or_predicate_is_rejected(self):
        for field, value in ((1, "[R3]"), (2, "R3"), (0, "P1"), (0, "UP0")):
            with self.subTest(operand=field, value=value):
                self.setUp()
                self.byp[0x130]["operands"][field] = operand(value)
                self.assertFalse(self.candidate()["cfg_ok"])
        self.setUp()
        self.byp[0x140]["is_uniform_predicate"] = True
        self.assertFalse(self.candidate()["cfg_ok"])

    def test_setup_must_not_change_barrier_address(self):
        self.byp[0x120] = register_copy(destination="R1", source="UR1")
        self.assertFalse(self.candidate()["cfg_ok"])

    def test_first_gate_predicate_clobber_or_namespace_mismatch_is_rejected(self):
        self.byp[0x110] = instruction(
            "ISETP.NE.AND", list(map(operand, ["P0", "PT", "R1", "R2", "PT"])))
        self.byp[0x120] = instruction(
            "BRA", [operand("0x160", 0x160)], predicate=0)
        self.byp[0x130] = register_copy()
        self.byp[0x140] = wait(0x140, trywait=True)
        self.byp[0x150] = instruction(
            "BRA", [operand("0x200", 0x200)], predicate=0, negate=True)
        self.byp[0x160] = instruction("NOP", [])
        self.byp[0x240] = instruction("BRA", [operand("0x160", 0x160)])
        self.assertFalse(self.candidate()["cfg_ok"])
        self.setUp()
        self.byp[0x110]["is_uniform_predicate"] = True
        self.assertFalse(self.candidate()["cfg_ok"])

    def test_reload_origin_must_be_executed_by_the_same_lanes(self):
        candidate = self.candidate()
        self.assertTrue(candidate["cfg_ok"])
        for field, value in (("predicate_mask", 0), ("active_mask", 1),
                             ("function_unique_id", 2), ("pc", 0x70)):
            with self.subTest(field=field):
                saved = self.insts[0][field]
                self.insts[0][field] = value
                self.assertFalse(self.validate(candidate)["dynamic_ok"])
                self.insts[0][field] = saved

    def test_recorded_ready_edge_must_match_predicate_and_masks(self):
        candidate = self.candidate()
        self.assertTrue(candidate["cfg_ok"])
        for field, value in (("predicate_mask", 0), ("active_mask", 1)):
            with self.subTest(field=field):
                saved = self.insts[3][field]
                self.insts[3][field] = value
                self.assertFalse(self.validate(candidate)["dynamic_ok"])
                self.insts[3][field] = saved

    def test_legacy_adjacent_gate_has_no_new_proof_fields(self):
        self.byp[0x120] = wait(0x120, trywait=True)
        self.byp[0x130] = instruction(
            "BRA", [operand("0x200", 0x200)], predicate=0, negate=True)
        self.byp[0x110]["operands"] = [operand("0x140", 0x140)]
        self.byp[0x240]["operands"] = [operand("0x140", 0x140)]
        candidate = self.candidate()
        self.assertTrue(candidate["cfg_ok"])
        self.assertEqual(candidate["all_gate_wait_pcs"], [0x100, 0x120])
        self.assertNotIn("gate_setup_pcs", candidate)
        self.assertEqual(candidate["entry_arming_pcs"], [0x110, 0x120, 0x130])

    def test_production_strict_emission_and_invalid_path_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extra = root / "extra_info"
            extra.mkdir()
            (extra / "sync_operand_resolver.json").touch()
            (root / "d_0_s_0_k_5_0,0,0.pb").touch()
            output = extra / "spin_wait_sidecar.json"
            argv = ["sidecar", "--traces", directory, "--strict"]
            with mock.patch.object(SIDECAR.sys, "argv", argv), \
                    mock.patch.object(SIDECAR, "load_kernels",
                                      return_value=({1: self.byp}, {1: "fixture"})), \
                    mock.patch.object(SIDECAR, "load_sync_sites", return_value=self.sites), \
                    mock.patch.object(SIDECAR, "kernel_dir_map",
                                      return_value={5: directory}), \
                    mock.patch.object(SIDECAR._ts, "_peek_funcid", return_value=1), \
                    mock.patch.object(SIDECAR, "detect_regions", return_value=[]), \
                    mock.patch.object(SIDECAR, "vpb_parse_full",
                                      return_value=[(0, self.insts)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                SIDECAR.main()
                document = json.loads(output.read_text())
                records = document["kernels"][0]["wait_blocks"]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["entry_probe_pc"], 0x100)
                self.assertEqual(records[0]["runtime_strategy"], "INJECT_MISSING_WAIT")
                output.unlink()
                self.insts[3]["predicate_mask"] = 0
                with self.assertRaisesRegex(SystemExit, "failed its proof"):
                    SIDECAR.main()
                self.assertFalse(output.exists())

    def test_new_inject_receives_exact_generation_ticket(self):
        candidate = self.candidate()
        candidate["dynamic"] = self.validate(candidate)
        _, _, canonical = SIDECAR.prove_wait_condition(
            1, candidate, self.byp, self.sites)
        candidate["canonical_wait_condition"] = canonical
        record = SIDECAR.make_inject_missing_wait_record(
            5, candidate, self.sites, self.byp)

        def event(pc, kind, before, after, semantic, ordinal=None):
            inst = dynamic_instruction(pc, 0xffffffff)
            inst.update(
                sync_valid=True, sync_has_semantic_raw=True,
                barrier_addr=0x1000, semantic_raw=semantic,
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
            event(0x50, SIDECAR.CONTRACT_EVENT_EXCH, 0, 0, 0x1ffffe),
            event(0x60, SIDECAR.CONTRACT_EVENT_ARRIVE, 0, 1, 256, 1),
            *self.insts,
        ]
        insts[4] = event(0x100, SIDECAR.CONTRACT_EVENT_WAIT, 1, 1, 0)
        sites = {
            **self.sites,
            (1, 0x50): {"sync_kind": "EXCH"},
            (1, 0x60): {"sync_kind": "ARRIVE_EXPECT_TX"},
        }
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "d_0_s_0_k_5_0,0,0.pb").touch()
            with mock.patch.object(SIDECAR, "vpb_parse_full", return_value=[(0, insts)]), \
                    mock.patch.object(SIDECAR, "_load_tma_ownership_status",
                                      return_value={5: True}):
                report = SIDECAR.analyze_mbarrier_contract_capture(
                    directory, {5: directory}, {5: 1}, sites,
                    [{"trace_kernel_id": 5, "wait_blocks": [record]}], "all",
                    static_kernels={1: self.byp})
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["runtime_tickets"]), 1)
        ticket = report["runtime_tickets"][0]
        self.assertEqual(ticket["action_pc"], 0x100)
        self.assertEqual(ticket["source_warp_instruction_index"], 4)
        self.assertEqual(ticket["target_release_generation"], 1)


if __name__ == "__main__":
    unittest.main()
