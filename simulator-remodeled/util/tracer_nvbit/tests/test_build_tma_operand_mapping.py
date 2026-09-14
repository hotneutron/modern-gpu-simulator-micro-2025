import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_tma_operand_mapping.py"
SPEC = importlib.util.spec_from_file_location("build_tma_operand_mapping", MODULE_PATH)
MAPPING = importlib.util.module_from_spec(SPEC)
with mock.patch.object(sys, "path", [str(MODULE_PATH.parent), *sys.path]):
    SPEC.loader.exec_module(MAPPING)

CONFIG_ID = "tm_r4_dt9_box_64x128x1x1"
KEY_FIELDS = ["trace_kernel_id", "unique_function_id", "pc_hex"]


def base_map_entry(pc, opcode, kernel_id=5, uid=3):
    return {
        "trace_kernel_id": kernel_id,
        "unique_function_id": uid,
        "pc_hex": pc,
        "opcode": opcode,
        "source": "utmacctl_chain",
        "operand_addressed": False,
        "base_hex": "0x7f0000000000",
        "encode_id": "0",
        "smem_offset_hex": "0x400",
        "tensor_rank": 4,
        "tensor_data_type": 9,
        "element_size": 2,
        "box_dim": [64, 128, 1, 1],
    }


def base_map_document(*entries):
    return {
        "version": 2,
        "key_fields": KEY_FIELDS,
        "validation_complete": True,
        "validation": {
            "sites_total": len(entries),
            "chain_resolved": len(entries),
            "ambiguous": 0,
            "unresolved": 0,
            "direct_authority": 0,
            "pool_authority": 0,
        },
        "map": {
            f"{entry['trace_kernel_id']}:{entry['unique_function_id']}:{entry['pc_hex']}":
                entry
            for entry in entries
        },
    }


def runtime_site(pc, opcode, descriptor_ref=None, kernel_id=5, uid=3):
    result = {
        "trace_kernel_id": kernel_id,
        "unique_function_id": uid,
        "pc_hex": pc,
        "opcode": opcode,
        "runtime_observed": True,
        "runtime_observed_values": {
            "operand_1": {"raw_value_lo_samples": [0x400]},
        },
    }
    if descriptor_ref is not None:
        result["descriptor_ref"] = descriptor_ref
    return result


class UtmapfDescriptorLinkTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.extra = Path(self.temp_dir.name)
        self.base_path = self.extra / "tma_pc_base_map.json"

    def load_refs(self, document):
        self.base_path.write_text(json.dumps(document))
        return MAPPING.load_descriptor_refs(self.extra)

    def prefetch_from_map(self, document=None):
        document = document or base_map_document(
            base_map_entry("0xa270", "UTMAPF")
        )
        refs = self.load_refs(document)
        entry = runtime_site("0xa270", "UTMAPF.L2.4D")
        entry["descriptor_ref"] = MAPPING.lookup_descriptor_ref(
            refs, 5, 3, entry
        )
        return entry

    def link_and_validate(self, prefetch, consumers):
        link = MAPPING.build_utmapf_descriptor_link(prefetch, consumers, {})
        prefetch["descriptor_link"] = link
        MAPPING.verify_descriptor_required_sites_or_fail(
            {"resolver": [prefetch]}
        )
        return link

    def test_chain_binding_does_not_require_a_unique_later_consumer(self):
        prefetch = self.prefetch_from_map()
        consumers = [
            runtime_site(pc, "UTMALDG.4D", {"config_ids": [CONFIG_ID]})
            for pc in ("0xa600", "0xa640")
        ]
        link = self.link_and_validate(prefetch, consumers)

        self.assertEqual(link["match_rule"], "kernel_aware_utmacctl_chain")
        self.assertEqual(link["matched_descriptor"]["config_ids"], [CONFIG_ID])
        self.assertEqual(
            link["matched_descriptor"]["base_hex"], "0x7f0000000000"
        )
        self.assertNotIn("matched_consumer", link)

    def test_chain_binding_does_not_require_any_later_consumer(self):
        link = self.link_and_validate(self.prefetch_from_map(), [])
        self.assertEqual(link["status"], "matched")

    def test_chain_binding_is_not_overridden_by_consumer_sample_match(self):
        prefetch = self.prefetch_from_map()
        consumer = runtime_site(
            "0xa600", "UTMALDG.4D", {"config_ids": ["different_shape"]}
        )
        link = self.link_and_validate(prefetch, [consumer])
        self.assertEqual(link["matched_descriptor"]["config_ids"], [CONFIG_ID])

    def test_invalid_v2_base_maps_are_rejected_before_linking(self):
        document = base_map_document(base_map_entry("0xa270", "UTMAPF"))
        invalid_documents = []
        incomplete = copy.deepcopy(document)
        incomplete["validation_complete"] = False
        invalid_documents.append(incomplete)
        for field in ("ambiguous", "unresolved", "direct_authority", "pool_authority"):
            invalid = copy.deepcopy(document)
            invalid["validation"][field] = 1
            invalid_documents.append(invalid)
        mismatch = copy.deepcopy(document)
        mismatch["map"]["5:3:0xa270"]["trace_kernel_id"] = 6
        invalid_documents.append(mismatch)
        for source in ("direct", "pool"):
            invalid = copy.deepcopy(document)
            invalid["map"]["5:3:0xa270"]["source"] = source
            invalid_documents.append(invalid)

        for invalid in invalid_documents:
            with self.subTest(document=invalid):
                with self.assertRaises(SystemExit):
                    self.load_refs(invalid)

    def test_chain_proof_cannot_be_reused_at_a_different_site(self):
        for field, value in (
            ("trace_kernel_id", 6),
            ("unique_function_id", 4),
            ("pc_hex", "0xa280"),
        ):
            with self.subTest(field=field):
                prefetch = self.prefetch_from_map()
                prefetch[field] = value
                with self.assertRaises(SystemExit):
                    self.link_and_validate(prefetch, [])

    def test_chain_binding_still_requires_one_descriptor_config(self):
        for config_ids in ([], [CONFIG_ID, "another_shape"]):
            with self.subTest(config_ids=config_ids):
                prefetch = self.prefetch_from_map()
                prefetch["descriptor_ref"]["config_ids"] = config_ids
                with self.assertRaises(SystemExit):
                    self.link_and_validate(prefetch, [])

    def test_legacy_base_map_cannot_claim_chain_authority(self):
        document = {
            "version": 1,
            "validation_complete": True,
            "map": {"3:0xa270": base_map_entry("0xa270", "UTMAPF")},
        }
        prefetch = self.prefetch_from_map(document)
        with self.assertRaisesRegex(SystemExit, "no_exact_later_utmaldg_match"):
            self.link_and_validate(prefetch, [])

    def test_legacy_unique_consumer_still_resolves(self):
        prefetch = runtime_site("0xa270", "UTMAPF.L2.4D")
        consumer = runtime_site(
            "0xa600", "UTMALDG.4D", {"config_ids": [CONFIG_ID]}
        )
        link = self.link_and_validate(prefetch, [consumer])
        self.assertEqual(link["match_rule"], "exact_forward_utmaldg_operand_1")
        self.assertEqual(link["matched_consumer"]["pc_hex"], "0xa600")

    def test_legacy_multiple_consumers_still_fail_closed(self):
        prefetch = runtime_site("0xa270", "UTMAPF.L2.4D")
        consumers = [
            runtime_site(pc, "UTMALDG.4D", {"config_ids": [CONFIG_ID]})
            for pc in ("0xa600", "0xa640")
        ]
        with self.assertRaisesRegex(SystemExit, "multiple_exact_later_utmaldg_matches"):
            self.link_and_validate(prefetch, consumers)

    def test_other_kernel_binding_is_not_used(self):
        refs = self.load_refs(base_map_document(
            base_map_entry("0xa270", "UTMAPF", kernel_id=6)
        ))
        prefetch = runtime_site("0xa270", "UTMAPF.L2.4D")
        self.assertIsNone(MAPPING.lookup_descriptor_ref(refs, 5, 3, prefetch))

    def test_build_resolver_handles_two_prefetches_and_multiple_consumers(self):
        sites = [
            ("0xa270", "UTMAPF.L2.4D", "[UR4], [UR8]"),
            ("0xa280", "UTMAPF.L2.4D", "[UR4], [UR8]"),
            ("0xa600", "UTMALDG.4D", "[UR4], [UR8], desc[UR12]"),
            ("0xa640", "UTMALDG.4D", "[UR4], [UR8], desc[UR12]"),
        ]
        self.load_refs(base_map_document(*(
            base_map_entry(pc, opcode.split(".")[0]) for pc, opcode, _ in sites
        )))
        discovery_entries = []
        runtime_rows = []
        for pc, opcode, operands in sites:
            is_load = opcode.startswith("UTMALDG")
            discovery_entries.append({
                "pc": int(pc, 16),
                "pc_hex": pc,
                "opcode": opcode,
                "text": f"{opcode} {operands};",
                "role": "descriptor_load_consumer" if is_load else "prefetch",
                "desc_refs": ["UR12"] if is_load else [],
                "desc_regs": [12, 13] if is_load else [],
                "support_regs": [4, 8],
            })
            registers = [4, 8, 12, 5] if is_load else [4, 8]
            for callback_index, reg_id in enumerate(registers):
                runtime_rows.append({
                    "kernel_id": 5,
                    "unique_function_id": 3,
                    "pc_hex": pc,
                    "opcode": opcode,
                    "callback_index": callback_index,
                    "operand_type": "UNIFORM",
                    "operand_reg_id": reg_id,
                    "value_lo": 0x400,
                    "value_hi": 0,
                    "value_2": 0,
                    "value_3": 0,
                })
        (self.extra / "tma_discovery.json").write_text(json.dumps({
            "unique_function_id": 3,
            "function_name": "synthetic_fwd",
            "instructions": discovery_entries,
        }))
        (self.extra / "enhanced_execution_info.json").write_text(json.dumps({
            "kernels": [{
                "unique_function_id": 3,
                "kernel_name": "synthetic_fwd___0",
            }],
        }))
        (self.extra / "tma_runtime_operand_debug.jsonl").write_text(
            "\n".join(json.dumps(row) for row in runtime_rows) + "\n"
        )

        payload = MAPPING.build_resolver(self.extra)
        MAPPING.verify_descriptor_required_sites_or_fail(payload)

        self.assertTrue(payload["validation_complete"])
        self.assertEqual(payload["validation"]["runtime_sites"], 4)
        self.assertEqual(payload["source"]["hidden_completion_sites"], 2)
        prefetches = [
            entry for entry in payload["resolver"]
            if entry["opcode"].startswith("UTMAPF")
        ]
        self.assertEqual(len(prefetches), 2)
        for entry in prefetches:
            self.assertEqual(
                entry["descriptor_link"]["match_rule"],
                "kernel_aware_utmacctl_chain",
            )


if __name__ == "__main__":
    unittest.main()
