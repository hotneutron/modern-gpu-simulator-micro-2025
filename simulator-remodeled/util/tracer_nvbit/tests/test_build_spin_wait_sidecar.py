import importlib.util
import unittest
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
    }


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

    def test_plain_nanosleep_is_delay_only(self):
        by_pc = {
            0x100: instruction("NANOSLEEP", [operand("0x40")]),
        }
        candidate = {
            "allowed_block_pcs": [0x100],
            "loopback_bra_pc": 0x110,
        }

        safe, reason, classes = SIDECAR.prove_effects(
            1, candidate, by_pc)

        self.assertTrue(safe)
        self.assertIsNone(reason)
        self.assertEqual(classes, {"nanosleep": 1})

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
