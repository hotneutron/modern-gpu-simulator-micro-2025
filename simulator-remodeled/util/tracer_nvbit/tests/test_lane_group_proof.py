import copy
import unittest

from test_build_spin_wait_sidecar import (
    SIDECAR, dynamic_instruction, instruction, operand, wait,
)


class LaneGroupProofTest(unittest.TestCase):
    def setUp(self):
        self.byp = {
            0x100: wait(0x100, trywait=True),
            0x110: instruction("NANOSLEEP.SYNCS", [], predicate=0, negate=True),
            0x120: wait(0x120),
            0x130: instruction("BRA", [operand("0x100", 0x100)],
                               predicate=0, negate=True),
            0x140: instruction("NOP", []),
            0x200: instruction("UTMALDG.4D", [operand("[UR4]")]),
        }
        self.byp[0x120].update(is_predicated=True, is_predicate_negate=True)
        self.xs = []
        for pc, mask in [(0x100, 14), (0x200, 1), (0x110, 14),
                         (0x120, 14), (0x130, 14), (0x100, 14),
                         (0x100, 1), (0x110, 1), (0x110, 14),
                         (0x120, 15), (0x130, 15)]:
            inst = dynamic_instruction(pc, mask)
            if pc in (0x100, 0x120):
                inst.update(sync_valid=True, barrier_addr=0x800, semantic_raw=0)
            self.xs.append(inst)

    def prove(self):
        return SIDECAR.prove_lane_group_poll_span(self.xs, self.byp, 1, 0, 15)

    def test_exact_partition_does_not_include_full_mask_boundary(self):
        ok, why, proof = self.prove()
        self.assertTrue(ok, why)
        self.assertEqual(proof["merge_index"], 9)
        self.assertEqual(sorted(proof["main_indices"] + proof["polling_indices"]),
                         list(range(9)))
        self.assertEqual(proof["main_indices"], [1, 6, 7])
        self.assertEqual(proof["polling_indices"][0], 0)
        self.assertEqual(len(SIDECAR.find_lane_group_poll_spans(
            self.xs, self.byp, 1, [15] * len(self.xs))), 1)

    def test_predicated_off_semantic_occurrence_keeps_raw_owner(self):
        self.xs[1]["predicate_mask"] = 0
        self.assertTrue(self.prove()[0])
        self.assertIn(1, self.prove()[2]["main_indices"])

    def test_relocated_pcs_function_masks_registers_and_barriers(self):
        for shift, fid, main, polling, barrier, state, pred in [
                (0x4000, 73, 0xff00, 0x00ff, 0x54320, 0x80000000, 3),
                (0x180, 19, 0x55555555, 0xaaaaaaaa, 0x32140, 0, 6),
                (0x800, 4, 0x00f00000, 0x700, 0x880, 0x80000000, 1)]:
            with self.subTest(fid=fid, main=main):
                self.setUp()
                byp = {pc + shift: copy.deepcopy(inst) for pc, inst in self.byp.items()}
                for pc in (0x100, 0x120):
                    byp[pc + shift]["operands"] = [
                        operand("P%d" % pred), operand("[R17+URZ+0x80]"), operand("R21")]
                for pc in (0x110, 0x120, 0x130):
                    byp[pc + shift]["predicate_register"] = pred
                byp[0x130 + shift]["operands"] = [operand(hex(0x100 + shift), 0x100 + shift)]
                masks = {1: main, 14: polling, 15: main | polling}
                insts = copy.deepcopy(self.xs)
                for inst in insts:
                    inst["pc"] += shift
                    inst["function_unique_id"] = fid
                    inst["active_mask"] = masks[inst["active_mask"]]
                    inst["predicate_mask"] = masks[inst["predicate_mask"]]
                    if inst["sync_valid"]:
                        inst.update(barrier_addr=barrier, semantic_raw=state)
                before = copy.deepcopy(insts)
                ok, reason, proof = SIDECAR.prove_lane_group_poll_span(
                    insts, byp, fid, 0, main | polling)
                self.assertTrue(ok, reason)
                self.assertEqual(proof["main_mask"], main)
                self.assertEqual(proof["polling_mask"], polling)
                self.assertEqual(proof["merge_index"], 9)
                self.assertEqual(insts, before)

    def test_rejects_dynamic_escape_and_missing_payload(self):
        original = copy.deepcopy(self.xs)
        for index, field, value in [
            (2, "pc", 0x200), (2, "active_mask", 3),
            (3, "sync_valid", False), (3, "barrier_addr", 0x900),
            (3, "semantic_raw", 1), (2, "predicate_mask", 0),
            (9, "sync_valid", False), (9, "pc", 0x140),
            (9, "predicate_mask", 1), (6, "sync_valid", False),
            (6, "barrier_addr", 0x900), (1, "function_unique_id", 2),
        ]:
            with self.subTest(index=index, field=field):
                self.xs = copy.deepcopy(original)
                self.xs[index][field] = value
                self.assertFalse(self.prove()[0])

    def test_rejects_shared_registers_and_unknown_control(self):
        self.byp[0x100]["operands"][2] = operand("UR4")
        self.byp[0x120]["operands"][2] = operand("UR4")
        self.assertFalse(self.prove()[0])
        self.setUp()
        self.byp[0x200]["op_code"] = "CALL"
        self.assertFalse(self.prove()[0])

    def test_rejects_wrong_loop_edge_or_predicate(self):
        self.byp[0x130]["operands"][0] = operand("0x140", 0x140)
        self.assertFalse(self.prove()[0])
        self.setUp()
        self.byp[0x110]["is_predicate_negate"] = False
        self.assertFalse(self.prove()[0])

    def test_missing_merge_and_invalid_start(self):
        self.xs = self.xs[:9]
        self.assertFalse(self.prove()[0])
        self.assertFalse(SIDECAR.prove_lane_group_poll_span(
            self.xs, self.byp, 1, -1, 15)[0])

    def test_ready_trywait_preserves_predicated_off_tail_and_collective(self):
        self.byp.update({
            0x300: instruction("IMAD.MOV.U32", list(map(
                operand, ["R8", "RZ", "RZ", "-1"]))),
            0x310: instruction("WARPSYNC.COLLECTIVE",
                               [operand("R8"), operand("0x340", 0x340)]),
            0x320: instruction("SHFL.IDX", []),
            0x330: instruction("ENDCOLLECTIVE", []),
            0x340: instruction("NOP", []),
        })
        self.xs = self.xs[:4]
        self.xs[2]["predicate_mask"] = 0
        self.xs[3].update(predicate_mask=0, sync_valid=False)
        for pc, mask, pred in [
                (0x130, 14, 0), (0x140, 14, 14), (0x300, 14, 14),
                (0x310, 14, 14), (0x300, 1, 1), (0x310, 1, 1),
                (0x320, 15, 15), (0x330, 15, 15)]:
            self.xs.append(dynamic_instruction(pc, mask, pred))
        ok, why, proof = self.prove()
        self.assertTrue(ok, why)
        self.assertEqual(proof["merge_kind"], "COLLECTIVE")
        self.assertEqual(proof["reorderable_indices"], [0, 2, 3, 4])
        self.assertEqual(proof["polling_indices"], [0, 2, 3, 4, 5, 6, 7])
        self.xs[4]["predicate_mask"] = 14
        self.assertFalse(self.prove()[0])


if __name__ == "__main__":
    unittest.main()
