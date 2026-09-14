import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CPP = ROOT / "gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc"


class PartialOccurrenceLoaderTest(unittest.TestCase):
    """Compile the production JSON loader and lookup without a CUDA runtime."""

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("C++ compiler not installed")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.binary = Path(cls.temp.name) / "partial-loader-test"
        source = CPP.read_text()

        def section(start, end):
            return source[source.index(start):source.index(end)]

        harness = r'''
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include "gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h"
#include "util/traces_enhanced/src/rapidjson/document.h"
using address_type = uint64_t;
[[noreturn]] void wait_block_sidecar_fatal(
    const std::filesystem::path &, const std::string &error) {
  throw std::runtime_error(error);
}
uint64_t parse_tma_json_uint(const rapidjson::Value &v) { return v.GetUint64(); }
enum class TraceEnhancedOperandType { PRED, UPRED, REG, OTHER };
struct Operand {
  std::string text;
  long reg = 0;
  TraceEnhancedOperandType type = TraceEnhancedOperandType::OTHER;
  std::string get_operand_string() { return text; }
  long get_operand_reg_number() { return reg; }
  TraceEnhancedOperandType get_operand_type() { return type; }
};
struct traced_instruction {
  std::string op;
  std::vector<Operand> operands;
  bool predicated = false, negate = false, uniform = false;
  long predicate = 0;
  uint64_t target = 0;
  std::string get_op_code() { return op; }
  bool get_is_predicated() { return predicated; }
  bool get_is_predicate_negate() { return negate; }
  bool get_is_uniform_predicate() { return uniform; }
  long get_predicate_register() { return predicate; }
  size_t get_num_operands() { return operands.size(); }
  Operand &get_operand(size_t i) { return operands.at(i); }
};
struct traced_execution {
  std::map<uint64_t, std::shared_ptr<traced_instruction>> instructions;
  bool has_kernel_with_unique_function_id(uint32_t fid) { return fid == 1; }
  traced_execution &get_kernel_by_unique_function_id(uint32_t) { return *this; }
  std::shared_ptr<traced_instruction> get_instruction_ptr(uint32_t pc) {
    auto it = instructions.find(pc);
    return it == instructions.end() ? nullptr : it->second;
  }
};
std::shared_ptr<traced_instruction> wait_block_require_static_instruction(
    traced_execution &trace, uint32_t, uint64_t pc, const std::filesystem::path &) {
  return trace.instructions.at(pc);
}
bool wait_block_is_wait_opcode(const std::string &op) {
  return op.rfind("SYNCS.PHASECHK", 0) == 0;
}
std::string wait_block_normalize_operand(std::string text) { return text; }
uint64_t wait_block_static_direct_target(
    const std::shared_ptr<traced_instruction> &inst,
    const std::filesystem::path &, const std::string &) { return inst->target; }
struct warp_inst_t {
  uint32_t trace_kernel_id = 5, trace_block_idx_x = 0, trace_block_idx_y = 0;
  uint32_t trace_block_idx_z = 0, trace_cta_warp_id = 0, unique_function_id = 1;
  uint64_t pc = 0x100, trace_instruction_index = 10;
  bool trace_masks_valid = true;
  uint32_t trace_active_mask = 1, trace_predicate_mask = 1;
  uint64_t sync_barrier_addr = 0x1000, sync_semantic_raw = 0;
  bool sync_runtime_valid = true, sync_has_semantic_raw = true;
};
class gpgpu_sim {
 public:
  WaitBlockSidecarDB m_wait_block_sidecar_db;
  void resolve_partial_wait_occurrence(const warp_inst_t &, MissingWaitRecord &) const;
  bool lookup_lane_group_start(const warp_inst_t &, LaneGroupSpanRecord &) const;
  void validate_lane_group_wait(const warp_inst_t &) const;
};
rapidjson::Document *input_document = nullptr;
bool load_tma_json_document(const std::filesystem::path &, rapidjson::Document &doc) {
  doc.CopyFrom(*input_document, doc.GetAllocator());
  return true;
}
void wait_block_validate_fold_static_proof(traced_execution &, uint32_t,
    const SpinWaitRegionRecord &, const std::vector<uint64_t> &, uint64_t,
    const std::filesystem::path &) {}
'''
        harness += section("std::string wait_block_hex_u64(",
                           "const rapidjson::Value &wait_block_require_member(")
        harness += section("const rapidjson::Value &wait_block_require_member(",
                           "bool wait_block_parse_barrier_offset(")
        harness += section("bool wait_block_parse_barrier_offset(",
                           "uint64_t wait_block_static_direct_target(")
        harness += section("SyncInstructionKind parse_sync_instruction_kind_string(",
                           "SyncSemanticOperandRole parse_sync_semantic_operand_role_string(")
        harness += section("void wait_block_validate_action_opcode(",
                           "std::shared_ptr<traced_instruction> wait_block_require_static_instruction(")
        harness += section("void wait_block_validate_partial_static_proof(",
                           "void load_wait_block_sidecar_v5(")
        harness += section("void load_wait_block_sidecar_v5(",
                           "\n}\n\n// MOD. Begin. Improved tracer")
        harness += section("static WaitBlockGenerationTicketKey partial_occurrence_key(",
                           "bool gpgpu_sim::lookup_wait_block_generation_ticket(")
        harness += r'''
int main(int argc, char **argv) {
  try {
    std::string input((std::istreambuf_iterator<char>(std::cin)), {});
    rapidjson::Document doc;
    doc.Parse(input.c_str());
    traced_execution trace;
    auto add = [&](uint64_t pc, std::string op, bool pred = false,
                   bool neg = false, uint64_t target = 0) {
      auto inst = std::make_shared<traced_instruction>();
      inst->op = op; inst->predicated = pred; inst->negate = neg; inst->target = target;
      trace.instructions[pc] = inst;
      return inst;
    };
    add(0x100, "SYNCS.PHASECHK.TRANS64.TRYWAIT")->operands =
        {{"P0", 0, TraceEnhancedOperandType::PRED}, {"[R1]"}, {"R2"}};
    add(0x90, "BRA", true, false, 0x200);
    add(0x110, "BRA", true, true, 0x400);
    add(0x400, "SYNCS.PHASECHK")->operands =
        {{"P0", 0, TraceEnhancedOperandType::PRED}, {"[R1]"}, {"R2"}};
    add(0x410, "BRA", true, true, 0x400);
    add(0x420, "BRA", false, false, 0x120);
    add(0x200, "WARPSYNC.ALL");
    add(0x210, "NOP");
    MissingWaitRecord record;
    record.partial_hold_proof_version = 3;
    record.entry_probe_pc = 0x100; record.entry_branch_pc = 0x110;
    record.entry_arming_pcs = {0x110}; record.captured_ready_pc = 0x120;
    record.miss_arm_entry_pc = 0x400; record.miss_arm_wait_pcs = {0x400};
    record.miss_arm_loopback_pc = 0x410; record.miss_arm_exit_pc = 0x420;
    record.canonical_wait_state_operands = {"R2"};
    record.partial_entry_visits = 1;
    record.observed_entry_masks = {1}; record.observed_live_masks = {15};
    gpgpu_sim gpu;
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode.rfind("lane-group", 0) == 0) {
      auto head = add(0x500, "SYNCS.PHASECHK.TRANS64.TRYWAIT");
      head->operands = {{"P0", 0, TraceEnhancedOperandType::PRED}, {"[R1]"}, {"R2"}};
      add(0x510, "NANOSLEEP.SYNCS", true, true);
      add(0x520, "SYNCS.PHASECHK.TRANS64", true, true)->operands = head->operands;
      add(0x530, "BRA", true, true, 0x500);
      if (mode == "lane-group-uniform") head->operands[2].text = "UR4";
      if (mode == "lane-group-predicate") trace.instructions.at(0x510)->negate = false;
      record.partial_hold_proof_version = 4;
      if (mode == "lane-group-full") {
        input_document = &doc;
        load_wait_block_sidecar_v5("fixture", trace, gpu.m_wait_block_sidecar_db, false);
      } else {
        load_lane_group_spans(doc, record, 5, 1, trace, gpu.m_wait_block_sidecar_db, "fixture");
        load_partial_join_occurrences(doc, record, 5, 1, 16, trace,
                                     gpu.m_wait_block_sidecar_db, "fixture");
      }
      warp_inst_t inst;
      gpu.validate_lane_group_wait(inst);
      inst.pc = 0x500; inst.trace_instruction_index = 2;
      inst.trace_active_mask = inst.trace_predicate_mask = 14;
      LaneGroupSpanRecord span;
      if (!gpu.lookup_lane_group_start(inst, span) || span.merge_index != 22) return 9;
      if (!gpu.m_wait_block_sidecar_db.partial_join_occurrences.empty()) return 10;
      return 0;
    }
    if (mode.rfind("collective", 0) == 0) {
      auto mask = add(0x1f0, "IMAD.MOV.U32");
      mask->operands = {{"R8"}, {"RZ"}, {"RZ"}, {"-0x1"}};
      auto join = add(0x200, "WARPSYNC.COLLECTIVE", false, false, 0x230);
      join->operands = {{"R8", 8, TraceEnhancedOperandType::REG}, {"0x230"}};
      auto body = add(0x210, "SHFL.IDX");
      auto end = add(0x220, "ENDCOLLECTIVE");
      add(0x230, "BSYNC")->operands = {{"B0"}};
      if (mode == "collective-mask") mask->operands[3].text = "0x1";
      if (mode == "collective-register") mask->operands[0].text = "R9";
      if (mode == "collective-predicated") mask->predicated = true;
      if (mode == "collective-target") join->target = 0x210;
      if (mode == "collective-end") end->op = "NOP";
      if (mode == "collective-body") body->op = "STG.E";
    }
    if (mode == "full-loader" || mode == "all-live-two-gate") {
      input_document = &doc;
      load_wait_block_sidecar_v5("fixture", trace, gpu.m_wait_block_sidecar_db, false);
      record = gpu.m_wait_block_sidecar_db.missing_wait_by_entry.begin()->second;
      if (mode == "all-live-two-gate") {
        return record.entry_probe_pc == 0x100 &&
                       record.entry_branch_pc == 0x140 &&
                       record.captured_ready_pc == 0x150 &&
                       record.all_live_entry_visits == 1 &&
                       record.partial_entry_visits == 0 &&
                       !record.partial_deferred_replay &&
                       record.partial_modeled_miss_aborts &&
                       record.entry_probe_kind == SyncInstructionKind::TRYWAIT
                   ? 0 : 8;
      }
    } else {
      load_partial_join_occurrences(doc, record, 5, 1, 16, trace,
                                   gpu.m_wait_block_sidecar_db, "fixture");
    }
    warp_inst_t inst;
    if (mode == "missing-index") inst.trace_instruction_index++;
    if (mode == "wrong-mask") inst.trace_predicate_mask = 3;
    if (mode == "wrong-barrier") inst.sync_barrier_addr++;
    gpu.resolve_partial_wait_occurrence(inst, record);
    if (record.join_pc != 0x200 ||
        record.partial_hold_proof_version !=
            (mode == "full-loader" ? 2 :
             doc["partial_join_variants"][0]["proof_version"].GetUint())) return 2;
    PartialWaitRegionRecord region;
    inst.pc = 0x200; inst.trace_instruction_index = 5;
    inst.trace_active_mask = inst.trace_predicate_mask = 14;
    if (!lookup_partial_occurrence_event(gpu.m_wait_block_sidecar_db.partial_join_occurrences,
                                        gpu.m_wait_block_sidecar_db, &inst, region)) return 3;
    if (region.sibling_masks != std::set<uint64_t>{14}) return 6;
    inst.trace_instruction_index = 20;
    inst.trace_active_mask = inst.trace_predicate_mask = 1;
    if (!lookup_partial_occurrence_event(gpu.m_wait_block_sidecar_db.partial_join_occurrences,
                                        gpu.m_wait_block_sidecar_db, &inst, region) ||
        !region.sibling_masks.empty()) return 7;
    inst.trace_instruction_index = 6;
    if (lookup_partial_occurrence_event(gpu.m_wait_block_sidecar_db.partial_join_occurrences,
                                       gpu.m_wait_block_sidecar_db, &inst, region)) return 4;
    gpu.m_wait_block_sidecar_db.clear();
    if (!gpu.m_wait_block_sidecar_db.partial_wait_occurrences.empty() ||
        !gpu.m_wait_block_sidecar_db.partial_join_occurrences.empty() ||
        !gpu.m_wait_block_sidecar_db.partial_post_join_occurrences.empty()) return 5;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what();
    return 1;
  }
}
'''
        result = subprocess.run(
            [compiler, "-std=c++17", "-Wno-deprecated-declarations",
             "-I", str(ROOT), "-x", "c++", "-", "-o", str(cls.binary)],
            input=harness, text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stderr)

    def setUp(self):
        self.document = {
            "partial_join_pending_visits": 0,
            "partial_join_already_issued_visits": 1,
            "partial_join_variants": [{
                "proof_version": 2, "sibling_branch_pc": None,
                "join_pc": 0x200, "post_join_pc": 0x210, "join_kind": "WARPSYNC",
                "bssy_pc": None, "join_token": None,
            }],
            "partial_join_occurrences": [{
                "variant": 0, "cta": [0, 0, 0], "warp_id": 0,
                "source_index": 10, "active_mask": 1, "predicate_mask": 1,
                "barrier_addr": 0x1000, "wait_state_raw": 0,
                "sibling_join_index": 5, "blocked_join_index": 20,
                "post_join_index": 21, "sibling_mask": 14, "live_mask": 15,
            }],
        }

    def run_loader(self, mode=""):
        return subprocess.run(
            [str(self.binary), mode], input=json.dumps(self.document),
            text=True, capture_output=True)

    def test_valid_occurrence_loads_and_selects_without_changing_generation(self):
        result = self.run_loader()
        self.assertEqual(result.returncode, 0, result.stderr)

    def lane_group_document(self):
        self.document.update(
            partial_join_already_issued_visits=0, partial_join_variants=[],
            partial_join_occurrences=[],
            lane_group_spans=[{
                "cta": [0, 0, 0], "warp_id": 0, "start_index": 2, "merge_index": 22,
                "poll_start_index": 2, "start_pc": 0x500, "start_mask": 14,
                "start_barrier_addr": 0x1000, "start_wait_state_raw": 0,
                "prefix_poll_pcs": [],
                "reorderable_until": 21, "poll_head_pc": 0x500, "poll_probe_pc": 0x520,
                "poll_loopback_pc": 0x530, "merge_pc": 0x520, "main_mask": 1,
                "polling_mask": 14, "live_mask": 15, "main_count": 10,
                "polling_count": 10, "reorderable_count": 10, "barrier_addr": 0x1000,
                "wait_state_raw": 0, "merge_kind": "POLL",
            }],
            lane_group_occurrences=[{
                "cta": [0, 0, 0], "warp_id": 0, "source_index": 10, "start_index": 2,
                "active_mask": 1, "predicate_mask": 1, "barrier_addr": 0x1000,
                "wait_state_raw": 0,
            }])

    def test_lane_group_loader_and_exact_lookup(self):
        self.lane_group_document()
        result = self.run_loader("lane-group")
        self.assertEqual(result.returncode, 0, result.stderr)
        for mode in ("lane-group-uniform", "lane-group-predicate"):
            self.assertNotEqual(self.run_loader(mode).returncode, 0)

    def test_lane_group_corrupt_partition_is_rejected(self):
        for field, value in (("main_mask", 14), ("merge_index", 2),
                             ("main_count", 11), ("reorderable_until", 22),
                             ("reorderable_count", 0), ("merge_pc", 0x540)):
            self.lane_group_document()
            with self.subTest(field=field):
                self.document["lane_group_spans"][0][field] = value
                self.assertNotEqual(self.run_loader("lane-group").returncode, 0)
        self.lane_group_document()
        self.document["lane_group_occurrences"][0]["source_index"] = 22
        self.assertNotEqual(self.run_loader("lane-group").returncode, 0)

    def test_lane_group_full_loader_checks_census_and_poll_ownership(self):
        self.test_full_sidecar_loader_admits_valid_occurrence_proof()
        document = self.document
        block = document["kernels"][0]["wait_blocks"][0]
        self.setUp()
        self.lane_group_document()
        block.update(self.document, partial_hold_proof_version=4)
        pcs = [0x500, 0x510, 0x520, 0x530]
        poll = {
            "block_id": "poll", "unique_function_id": 1,
            "runtime_strategy": "FOLD_CAPTURED_POLL", "fold_proof_version": 1,
            "poll_loop_head_pc": 0x500, "poll_probe_pc": 0x520,
            "poll_probe_opcode": "SYNCS.PHASECHK.TRANS64", "poll_probe_kind": "PHASECHK",
            "poll_loopback_pc": 0x530, "poll_loop_exit_pc": 0x540,
            "semantic_free_body": True, "control_safe": True, "wait_condition_safe": True,
            "effect_safe": True, "barrier_self_tied": True,
            "poll_loop_body_pcs": pcs[:3], "poll_advance_allowed_pcs": pcs,
            "poll_wait_pcs": [0x500, 0x520],
            "control_proof": {
                "all_wait_predicates_match": True, "loopback_predicate_negated": True,
                "loopback_target_pc": 0x500, "loopback_fallthrough_pc": 0x540,
            },
            "wait_condition_proof": {"sites": [
                {"pc": pc, "barrier_operand": "[R1]", "wait_state_operand": "R2"}
                for pc in [0x500, 0x520]]},
            "effect_proof": [{"pc": pc, "opcode": "poll", "effect": "poll"} for pc in pcs],
            "scoreboard_proof": {
                "checked_pcs": pcs, "wait_on_externally_armed_ids": [], "async_producer_ops": [],
            },
            "dynamic": {
                "is_spin_loop": True, "active_wait_payload_complete": True,
                "active_wait_occurrences": 1, "wait_sync_payload_missing": 0,
                "wait_condition_mismatches": 0,
            },
        }
        document["kernels"][0]["wait_blocks"].append(poll)
        document["wait_block_report"].update(
            discovered=2, passed=2, fold_captured_poll=1, partial_join_already_issued_visits=0)
        self.document = document
        result = self.run_loader("lane-group-full")
        self.assertEqual(result.returncode, 0, result.stderr)
        document["isize_bytes"] = 8
        self.assertNotEqual(self.run_loader("lane-group-full").returncode, 0)
        document["isize_bytes"] = 16
        document["wait_block_report"]["ready_only_partial_visits"] = 2
        self.assertNotEqual(self.run_loader("lane-group-full").returncode, 0)
        document["wait_block_report"]["ready_only_partial_visits"] = 1
        document["kernels"][0]["wait_blocks"].pop()
        self.assertNotEqual(self.run_loader("lane-group-full").returncode, 0)

    def test_direct_join_variant_keeps_existing_static_proof(self):
        variant = self.document["partial_join_variants"][0]
        variant.update(proof_version=1, sibling_branch_pc=0x90)
        result = self.run_loader()
        self.assertEqual(result.returncode, 0, result.stderr)
        variant["sibling_branch_pc"] = 0x110
        self.assertNotEqual(self.run_loader().returncode, 0)

    def test_corrupt_metadata_fails_closed(self):
        for field, bad in (
            ("variant", 8), ("sibling_mask", 1), ("sibling_join_index", 12),
            ("post_join_index", 19), ("active_mask", 0),
        ):
            with self.subTest(field=field):
                item = self.document["partial_join_occurrences"][0]
                original = item[field]
                item[field] = bad
                self.assertNotEqual(self.run_loader().returncode, 0)
                item[field] = original

    def test_missing_coverage_and_incorrect_join_are_rejected(self):
        self.document["partial_join_occurrences"] = []
        self.assertNotEqual(self.run_loader().returncode, 0)
        self.setUp()
        self.document["partial_join_variants"][0]["post_join_pc"] = 0x200
        self.assertNotEqual(self.run_loader().returncode, 0)

    def test_runtime_identity_mismatch_fails_closed(self):
        for mode in ("missing-index", "wrong-mask", "wrong-barrier"):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.run_loader(mode).returncode, 0)

    def test_collective_join_requires_a_proven_full_mask_and_scope(self):
        result = self.run_loader("collective")
        self.assertEqual(result.returncode, 0, result.stderr)
        for mode in ("mask", "register", "predicated", "target", "end", "body"):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.run_loader("collective-" + mode).returncode, 0)

    def test_full_sidecar_loader_admits_valid_occurrence_proof(self):
        block = {
            **self.document,
            "block_id": "test", "unique_function_id": 1,
            "runtime_strategy": "INJECT_MISSING_WAIT",
            "entry_probe_pc": 0x100, "entry_branch_pc": 0x110,
            "captured_ready_pc": 0x120, "entry_arming_pcs": [0x110],
            "miss_arm_entry_pc": 0x400, "miss_arm_wait_pcs": [0x400],
            "miss_arm_loopback_pc": 0x410, "miss_arm_exit_pc": 0x420,
            "entry_probe_opcode": "SYNCS.PHASECHK.TRANS64.TRYWAIT",
            "entry_probe_kind": "TRYWAIT",
            "entry_branch_not_ready_polarity": "PREDICATE_FALSE",
            "partial_modeled_miss_action": "DEFER_BLOCKED_SUFFIX_THEN_HOLD",
            "partial_hold_proof_version": 3, "sync_payload_missing": 0,
            "canonical_wait_condition": {
                "barrier_offset": "0x0", "wait_state_operands": ["R2"],
            },
            "observed_path_counts": {"ready_arm": 1, "second_gate_arm": 0, "captured_poll": 0},
            "all_live_entry_visits": 0, "partial_entry_visits": 1,
            "observed_effective_masks": {"0x1": 1}, "observed_live_masks": {"0xf": 1},
        }
        self.document = {
            "version": 5, "schema": "wait_block_sidecar",
            "validation_complete": True, "isize_bytes": 16,
            "wait_block_report": {
                "discovered": 1, "passed": 1, "failed": 0, "skipped": 0,
                "inject_missing_wait": 1, "fold_captured_poll": 0, "metadata_only": 0,
                "ready_only_all_live_visits": 0, "ready_only_partial_visits": 1,
                "ready_only_records_with_partial_visits": 1, "partial_defer_records": 1,
                "partial_join_pending_visits": 0, "partial_join_already_issued_visits": 1,
            },
            "kernels": [{"trace_kernel_id": 5, "unique_function_id": 1, "wait_blocks": [block]}],
        }
        result = self.run_loader("full-loader")
        self.assertEqual(result.returncode, 0, result.stderr)
        block["partial_join_pending_visits"] = 1
        self.assertNotEqual(self.run_loader("full-loader").returncode, 0)

    def test_generated_two_gate_record_uses_existing_all_live_loader(self):
        from test_waitblock_two_gate_setup import TwoGateSetupTest, SIDECAR

        fixture = TwoGateSetupTest()
        fixture.setUp()
        candidate = fixture.candidate()
        candidate["dynamic"] = fixture.validate(candidate)
        _, _, canonical = SIDECAR.prove_wait_condition(
            1, candidate, fixture.byp, fixture.sites)
        candidate["canonical_wait_condition"] = canonical
        block = SIDECAR.make_inject_missing_wait_record(
            5, candidate, fixture.sites, fixture.byp)
        self.document = {
            "version": 5, "schema": "wait_block_sidecar",
            "validation_complete": True, "isize_bytes": 16,
            "wait_block_report": {
                "discovered": 1, "passed": 1, "failed": 0, "skipped": 0,
                "inject_missing_wait": 1, "fold_captured_poll": 0, "metadata_only": 0,
                "ready_only_all_live_visits": 1, "ready_only_partial_visits": 0,
                "ready_only_records_with_partial_visits": 0, "partial_defer_records": 0,
                "partial_join_pending_visits": 0, "partial_join_already_issued_visits": 0,
            },
            "kernels": [{"trace_kernel_id": 5, "unique_function_id": 1,
                         "wait_blocks": [block]}],
        }
        result = self.run_loader("all-live-two-gate")
        self.assertEqual(result.returncode, 0, result.stderr)
        block["entry_probe_opcode"] = "SYNCS.PHASECHK.TRANS64"
        self.assertNotEqual(self.run_loader("all-live-two-gate").returncode, 0)
        block["entry_probe_opcode"] = "SYNCS.PHASECHK.TRANS64.TRYWAIT"
        block["observed_path_counts"]["second_gate_arm"] = 1
        self.assertNotEqual(self.run_loader("all-live-two-gate").returncode, 0)


if __name__ == "__main__":
    unittest.main()
