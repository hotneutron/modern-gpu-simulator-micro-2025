import sys
import tempfile
import unittest
from pathlib import Path


TRACER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACER_DIR))

import build_sync_event_order as ORDER
import build_waitblock_candidate_sidecar as CANDIDATES


def raw_event(begin, end, *, pc=0x100, ordinal=0):
    return {
        "record_version": 1,
        "record_kind": "SYNC_EVENT_ORDER",
        "launch_uid": 1,
        "device_id": 0,
        "stream_id": 0,
        "trace_kernel_id": 7,
        "cta_x": 2,
        "cta_y": 0,
        "cta_z": 0,
        "linear_cta_id": 2,
        "warp_id": 3,
        "sm_id": 4,
        "unique_function_id": 5,
        "pc": pc,
        "event_kind": "SYNCS_PHASECHK",
        "active_mask": 0xF,
        "predicate_mask": 0xF,
        "begin_seq": begin,
        "end_seq": end,
        "_raw_line": ordinal + 1,
    }


def protobuf_event(source_index, *, pc=0x100):
    return {
        "device_id": 0,
        "stream_id": 0,
        "trace_kernel_id": 7,
        "cta_x": 2,
        "cta_y": 0,
        "cta_z": 0,
        "warp_id": 3,
        "unique_function_id": 5,
        "pc": pc,
        "event_kind": "SYNCS_PHASECHK",
        "opcode": "SYNCS.PHASECHK.TRANS64",
        "source_warp_instruction_index": source_index,
        "active_mask": 0xF,
        "predicate_mask": 0xF,
        "sync_valid": True,
        "barrier_addr": 0x2000,
        "semantic_raw": 0,
        "tma_completion_valid": False,
        "tma_completion_barrier_addr": None,
        "tma_completion_barrier_ureg_id": None,
    }


class SyncOrderJoinTest(unittest.TestCase):
    def test_join_uses_per_pc_execution_order(self):
        raw = [
            raw_event(5, 6, ordinal=1),
            raw_event(1, 2, ordinal=0),
        ]
        static = {
            (5, 0x100): {
                "event_kind": "SYNCS_PHASECHK",
                "opcode": "SYNCS.PHASECHK.TRANS64",
            }
        }
        summary = ORDER.validate_raw_records(raw, static)
        joined = ORDER.join_records(
            raw, [protobuf_event(10), protobuf_event(20)]
        )

        self.assertEqual(summary["records"], 2)
        self.assertEqual(
            [event["begin_seq"] for event in joined], [1, 5]
        )
        self.assertEqual(
            [event["source_warp_instruction_index"] for event in joined],
            [10, 20],
        )
        self.assertEqual(
            [event["per_pc_occurrence_ordinal"] for event in joined],
            [0, 1],
        )

    def test_duplicate_cta_endpoint_is_rejected(self):
        first = raw_event(1, 2, pc=0x100, ordinal=0)
        second = raw_event(2, 3, pc=0x110, ordinal=1)
        second["event_kind"] = "BAR_SYNC"
        static = {
            (5, 0x100): {
                "event_kind": "SYNCS_PHASECHK",
                "opcode": "SYNCS.PHASECHK.TRANS64",
            },
            (5, 0x110): {
                "event_kind": "BAR_SYNC",
                "opcode": "BAR.SYNC",
            },
        }

        with self.assertRaisesRegex(ValueError, "duplicate CTA-local"):
            ORDER.validate_raw_records([first, second], static)

    def test_predicate_mask_must_be_subset_of_active_mask(self):
        record = raw_event(1, 2)
        record["active_mask"] = 0x1
        record["predicate_mask"] = 0x2
        static = {
            (5, 0x100): {
                "event_kind": "SYNCS_PHASECHK",
                "opcode": "SYNCS.PHASECHK.TRANS64",
            }
        }

        with self.assertRaisesRegex(ValueError, "not a subset"):
            ORDER.validate_raw_records([record], static)

    def test_protobuf_fingerprint_is_path_and_byte_stable(self):
        with tempfile.TemporaryDirectory() as left_dir, \
                tempfile.TemporaryDirectory() as right_dir:
            for root in (Path(left_dir), Path(right_dir)):
                path = (
                    root / "threadblocks/device_0/stream_0/kernel_1"
                    / "d_0_s_0_k_1_0,0,0.pb"
                )
                path.parent.mkdir(parents=True)
                path.write_bytes(b"same protobuf")

            left = ORDER.protobuf_tree_fingerprint(Path(left_dir))
            right = ORDER.protobuf_tree_fingerprint(Path(right_dir))
            self.assertEqual(left, right)


def provider(begin, end):
    return {"begin_seq": begin, "end_seq": end}


class CandidateIntervalTest(unittest.TestCase):
    def test_tma_candidate_max_keeps_overlap_and_prunes_after(self):
        wait = provider(10, 20)
        contracts = {
            0x2000: [
                {"completion_event": provider(1, 2)},
                {"completion_event": provider(3, 4)},
                {"completion_event": provider(5, 6)},
                {"completion_event": provider(15, 16)},
                {"completion_event": provider(21, 22)},
            ]
        }

        candidates, proof = CANDIDATES.resolve_candidates(
            0x2000, 0, wait, contracts, {}
        )

        self.assertEqual(candidates, [1, 3])
        self.assertEqual(proof["definitely_completed_before_wait"], 3)
        self.assertEqual(proof["possibly_completed_before_wait"], 4)
        self.assertEqual(proof["overlapping_completion_intervals"], 1)

    def test_count_candidate_uses_possible_arrival_units(self):
        wait = provider(10, 20)
        count_state = {
            0x3000: {
                "expected_arrive": 4,
                "arrivals": [
                    {"event": provider(1, 2), "units": 4},
                    {"event": provider(15, 16), "units": 4},
                    {"event": provider(21, 22), "units": 4},
                ],
            }
        }

        candidates, proof = CANDIDATES.resolve_candidates(
            0x3000, 0, wait, {}, count_state
        )

        self.assertEqual(candidates, [1])
        self.assertEqual(proof["max_possible_generation"], 2)
        self.assertEqual(proof["overlapping_arrival_units"], 4)

    def test_empty_candidate_set_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "empty candidate set"):
            CANDIDATES.resolve_candidates(
                0x4000,
                0,
                provider(10, 20),
                {0x4000: []},
                {},
            )

    def test_tma_provider_is_final_byte_reconciling_load(self):
        cta_key = (0, 0, 7, 2, 0, 0)
        warp_id = 3
        instructions = [
            {
                "function_unique_id": 5,
                "pc": 0x80,
                "active_mask": 0xF,
                "predicate_mask": 0xF,
                "sync_valid": True,
                "barrier_addr": 0x2000,
                "semantic_raw": 256,
                "tma_completion_valid": False,
                "tma_completion_barrier_addr": None,
            },
            {
                "function_unique_id": 5,
                "pc": 0x90,
                "active_mask": 0xF,
                "predicate_mask": 0xF,
                "sync_valid": False,
                "barrier_addr": None,
                "semantic_raw": None,
                "tma_completion_valid": True,
                "tma_completion_barrier_addr": 0x2000,
            },
            {
                "function_unique_id": 5,
                "pc": 0xA0,
                "active_mask": 0xF,
                "predicate_mask": 0xF,
                "sync_valid": False,
                "barrier_addr": None,
                "semantic_raw": None,
                "tma_completion_valid": True,
                "tma_completion_barrier_addr": 0x2000,
            },
        ]
        by_source = {}
        for source_index, interval in enumerate(((1, 2), (3, 4), (5, 6))):
            by_source[cta_key + (warp_id, source_index)] = {
                "begin_seq": interval[0],
                "end_seq": interval[1],
            }
        sync_metadata = {
            (5, 0x80): {"sync_kind": "ARRIVE_EXPECT_TX"}
        }
        base_map = {
            (7, 5, 0x90): {
                "operand_addressed": True,
                "covered_bytes": 128,
            },
            (7, 5, 0xA0): {
                "operand_addressed": True,
                "covered_bytes": 128,
            },
        }

        providers = CANDIDATES.build_tma_generation_providers(
            7,
            cta_key,
            [(warp_id, instructions)],
            by_source,
            sync_metadata,
            base_map,
            {(7, 5, 0x90), (7, 5, 0xA0)},
        )

        self.assertEqual(len(providers[0x2000]), 1)
        self.assertEqual(
            providers[0x2000][0]["completion_event"]["end_seq"], 6
        )
        self.assertEqual(
            providers[0x2000][0]["assigned_bytes"], 256
        )


if __name__ == "__main__":
    unittest.main()
