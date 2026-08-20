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


if __name__ == "__main__":
    unittest.main()
