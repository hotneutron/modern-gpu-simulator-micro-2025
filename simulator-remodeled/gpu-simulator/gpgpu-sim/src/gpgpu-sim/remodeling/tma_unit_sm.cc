#include "tma_unit_sm.h"

#include <cassert>
#include <iostream>
#include <set>
#include <tuple>

#include "../gpu-sim.h"
#include "sm.h"

namespace {

TMADirection classify_tma_direction(const warp_inst_t &inst) {
  if (inst.op == TMA_LOAD_OP) {
    return TMADirection::GMEM_TO_SMEM;
  }
  if (inst.op == TMA_STORE_OP) {
    return TMADirection::SMEM_TO_GMEM;
  }
  return TMADirection::NONE;
}

TMATransferType classify_tma_transfer_type(const warp_inst_t &inst,
                                           TMAOpcodeFamily family) {
  switch (family) {
    case TMAOpcodeFamily::UTMAPF:
    case TMAOpcodeFamily::UBLKPF:
      return TMATransferType::PREFETCH;
    case TMAOpcodeFamily::UTMACCTL:
    case TMAOpcodeFamily::UTMACMDFLUSH:
      return TMATransferType::CONTROL;
    case TMAOpcodeFamily::UBLKRED:
    case TMAOpcodeFamily::UTMAREDG:
      return TMATransferType::REDUCTION;
    default:
      break;
  }
  if (inst.op == TMA_LOAD_OP) {
    return TMATransferType::LOAD;
  }
  if (inst.op == TMA_STORE_OP) {
    return TMATransferType::STORE;
  }
  return TMATransferType::UNKNOWN;
}

TMAOperandForm classify_tma_operand_form(TMAOpcodeFamily family) {
  switch (family) {
    case TMAOpcodeFamily::UTMALDG:
      return TMAOperandForm::EXPLICIT_DESC;
    case TMAOpcodeFamily::UTMASTG:
      return TMAOperandForm::DESC_LIKE_PAIR;
    case TMAOpcodeFamily::UTMAREDG:
      // Always descriptor-backed; no non-descriptor form known.
      return TMAOperandForm::EXPLICIT_DESC;
    case TMAOpcodeFamily::UBLKCP:
    case TMAOpcodeFamily::UBLKPF:
      return TMAOperandForm::BULK_OPERAND;
    case TMAOpcodeFamily::UBLKRED:
      // FA3 backward trace uses descriptor-backed UBLKRED (EXPLICIT_DESC).
      // Non-descriptor bulk UBLKRED also exists but is a Phase 4 extension.
      // Phase 2 metadata binding will override this per-site via the operand
      // resolver; BULK_OPERAND here is only the static default.
      return TMAOperandForm::BULK_OPERAND;
    default:
      return TMAOperandForm::GENERIC;
  }
}

uint32_t infer_minimum_request_count(const TMACommand &cmd) {
  if (cmd.requests_total > 0) {
    return cmd.requests_total;
  }
  if (cmd.total_bytes > 0) {
    return 1;
  }
  return 0;
}

uint32_t infer_descriptor_total_bytes(
    const TMADescriptorConfigMetadata &descriptor_config) {
  if (descriptor_config.element_size == 0) {
    return 0;
  }
  uint64_t volume = 1;
  bool has_box_extent = false;
  for (uint32_t dim : descriptor_config.box_dim) {
    if (dim == 0) {
      continue;
    }
    volume *= dim;
    has_box_extent = true;
  }
  if (!has_box_extent) {
    return 0;
  }
  return static_cast<uint32_t>(volume * descriptor_config.element_size);
}

uint32_t infer_descriptor_request_total(
    const TMADescriptorConfigMetadata &descriptor_config) {
  if (descriptor_config.element_size == 0 || descriptor_config.box_dim[0] == 0) {
    return 0;
  }
  uint64_t row_bytes =
      static_cast<uint64_t>(descriptor_config.box_dim[0]) *
      descriptor_config.element_size;
  uint64_t requests_per_row = (row_bytes + 127) / 128;
  uint64_t outer_iters = 1;
  for (unsigned int i = 1; i < descriptor_config.box_dim.size(); ++i) {
    uint32_t dim = descriptor_config.box_dim[i];
    if (dim == 0) {
      continue;
    }
    outer_iters *= dim;
  }
  return static_cast<uint32_t>(requests_per_row * outer_iters);
}

uint32_t infer_request_total_from_covered_bytes(uint32_t covered_bytes) {
  if (covered_bytes == 0) {
    return 0;
  }
  return (covered_bytes + 127) / 128;
}

bool tma_family_requires_descriptor(TMAOpcodeFamily family) {
  switch (family) {
    case TMAOpcodeFamily::UTMALDG:
    case TMAOpcodeFamily::UTMAPF:
    case TMAOpcodeFamily::UTMASTG:
    case TMAOpcodeFamily::UTMAREDG:
      return true;
    default:
      return false;
  }
}

bool tma_family_requires_operand_metadata(TMAOpcodeFamily family) {
  switch (family) {
    case TMAOpcodeFamily::UBLKCP:
    case TMAOpcodeFamily::UBLKPF:
    case TMAOpcodeFamily::UBLKRED:
    case TMAOpcodeFamily::UTMAPF:
    case TMAOpcodeFamily::UTMACCTL:
      return true;
    default:
      return false;
  }
}

void log_tma_phase2_binding_once(const warp_inst_t &inst,
                                 const TMACommand &cmd) {
  static std::set<std::tuple<unsigned int, uint64_t, uint64_t>> logged_sites;
  auto key = std::make_tuple(inst.unique_function_id, static_cast<uint64_t>(inst.pc),
                             inst.tma_descriptor_ptr);
  if (!logged_sites.insert(key).second) {
    return;
  }
  std::cerr << "[TMA][Phase2] ufid=" << inst.unique_function_id
            << " pc=0x" << std::hex << static_cast<uint64_t>(inst.pc)
            << " descriptor_ptr=0x" << inst.tma_descriptor_ptr
            << " handle_hi=0x" << inst.tma_handle_hi << std::dec
            << " family=" << static_cast<int>(cmd.opcode_family)
            << " meta_source=" << static_cast<int>(cmd.meta_source)
            << " config_id=" << cmd.config_id
            << " total_bytes=" << cmd.total_bytes
            << " requests_total=" << cmd.requests_total
            << " covered_bytes=" << cmd.covered_bytes
            << " operand3_raw=" << cmd.operand3_raw
            << " operand_form=" << static_cast<int>(cmd.operand_form)
            << std::endl;
}

void log_tma_phase2_missing_descriptor_once(
    const warp_inst_t &inst, const TMACommand &cmd,
    const TMAResolvedSiteMetadata &metadata) {
  static std::set<std::tuple<unsigned int, uint64_t, uint64_t>> logged_sites;
  auto key = std::make_tuple(inst.unique_function_id,
                             static_cast<uint64_t>(inst.pc),
                             inst.tma_descriptor_ptr);
  if (!logged_sites.insert(key).second) {
    return;
  }
  const std::string trace_opcode =
      inst.has_extra_trace_instruction_info()
          ? inst.get_extra_trace_instruction_info().get_op_code()
          : "<no-trace-opcode>";
  std::cerr << "[TMA][Phase2][MissingDescriptor] ufid="
            << inst.unique_function_id << " pc=0x" << std::hex
            << static_cast<uint64_t>(inst.pc) << " handle_hi=0x"
            << inst.tma_handle_hi << " descriptor_ptr=0x"
            << inst.tma_descriptor_ptr << std::dec
            << " trace_opcode=" << trace_opcode
            << " family=" << static_cast<int>(cmd.opcode_family)
            << " meta_source=" << static_cast<int>(cmd.meta_source)
            << " metadata.valid=" << metadata.valid
            << " metadata.runtime_observed=" << metadata.runtime_observed
            << " metadata.operand_lookup_hit=" << metadata.operand_lookup_hit
            << " metadata.has_descriptor_metadata="
            << metadata.has_descriptor_metadata
            << " metadata.has_operand_metadata="
            << metadata.has_operand_metadata
            << " config_id=" << cmd.config_id
            << " total_bytes=" << cmd.total_bytes
            << " covered_bytes=" << cmd.covered_bytes
            << " operand_form=" << static_cast<int>(cmd.operand_form)
            << std::endl;
}

const char *tma_phase2_family_label(TMAOpcodeFamily family) {
  switch (family) {
    case TMAOpcodeFamily::UTMALDG:
      return "UTMALDG";
    case TMAOpcodeFamily::UTMAPF:
      return "UTMAPF";
    case TMAOpcodeFamily::UTMASTG:
      return "UTMASTG";
    case TMAOpcodeFamily::UBLKRED:
      return "UBLKRED";
    default:
      return nullptr;
  }
}

struct TMAPhase2FamilyStats {
  std::set<std::tuple<unsigned int, uint64_t, uint64_t>> descriptor_key_sites;
  std::set<std::tuple<unsigned int, uint64_t, uint64_t>> operand_only_sites;
  std::set<std::tuple<unsigned int, uint64_t, uint64_t>> mixed_sites;
  std::set<std::tuple<unsigned int, uint64_t, uint64_t>> unresolved_sites;
  uint64_t descriptor_key_commands = 0;
  uint64_t operand_only_commands = 0;
  uint64_t mixed_commands = 0;
  uint64_t unresolved_commands = 0;

  void record(const std::tuple<unsigned int, uint64_t, uint64_t> &key,
              const TMAResolvedSiteMetadata &metadata) {
    if (metadata.descriptor_lookup_hit && metadata.operand_lookup_hit) {
      mixed_sites.insert(key);
      ++mixed_commands;
      return;
    }
    if (metadata.descriptor_lookup_hit) {
      descriptor_key_sites.insert(key);
      ++descriptor_key_commands;
      return;
    }
    if (metadata.operand_lookup_hit) {
      operand_only_sites.insert(key);
      ++operand_only_commands;
      return;
    }
    unresolved_sites.insert(key);
    ++unresolved_commands;
  }

  uint64_t total_unique_sites() const {
    return descriptor_key_sites.size() + operand_only_sites.size() +
           mixed_sites.size() + unresolved_sites.size();
  }

  uint64_t total_commands() const {
    return descriptor_key_commands + operand_only_commands + mixed_commands +
           unresolved_commands;
  }
};

struct TMAPhase2BindingStats {
  TMAPhase2FamilyStats overall;
  TMAPhase2FamilyStats utmaldg;
  TMAPhase2FamilyStats utmapf;
  TMAPhase2FamilyStats utmastg;
  TMAPhase2FamilyStats ublkred;

  TMAPhase2FamilyStats *select_family_stats(TMAOpcodeFamily family) {
    switch (family) {
      case TMAOpcodeFamily::UTMALDG:
        return &utmaldg;
      case TMAOpcodeFamily::UTMAPF:
        return &utmapf;
      case TMAOpcodeFamily::UTMASTG:
        return &utmastg;
      case TMAOpcodeFamily::UBLKRED:
        return &ublkred;
      default:
        return nullptr;
    }
  }

  void record(const warp_inst_t &inst, const TMAResolvedSiteMetadata &metadata) {
    auto key = std::make_tuple(inst.unique_function_id,
                               static_cast<uint64_t>(inst.pc),
                               inst.tma_descriptor_ptr);
    overall.record(key, metadata);
    TMAPhase2FamilyStats *family_stats = select_family_stats(inst.tma_opcode_family);
    if (family_stats != nullptr) {
      family_stats->record(key, metadata);
    }
  }

  void dump_one(const char *label, const TMAPhase2FamilyStats &stats) const {
    if (stats.total_unique_sites() == 0 && stats.total_commands() == 0) {
      return;
    }
    std::cerr << "[TMA][Phase2][Stats][" << label << "] unique_sites total="
              << stats.total_unique_sites()
              << " descriptor_key=" << stats.descriptor_key_sites.size()
              << " operand_only=" << stats.operand_only_sites.size()
              << " mixed=" << stats.mixed_sites.size()
              << " unresolved=" << stats.unresolved_sites.size() << std::endl;
    std::cerr << "[TMA][Phase2][Stats][" << label << "] commands total="
              << stats.total_commands()
              << " descriptor_key=" << stats.descriptor_key_commands
              << " operand_only=" << stats.operand_only_commands
              << " mixed=" << stats.mixed_commands
              << " unresolved=" << stats.unresolved_commands << std::endl;
  }

  ~TMAPhase2BindingStats() {
    if (overall.total_unique_sites() == 0 && overall.total_commands() == 0) {
      return;
    }
    dump_one("ALL", overall);
    dump_one("UTMALDG", utmaldg);
    dump_one("UTMAPF", utmapf);
    dump_one("UTMASTG", utmastg);
    dump_one("UBLKRED", ublkred);
  }
};

TMAPhase2BindingStats &get_tma_phase2_binding_stats() {
  static TMAPhase2BindingStats stats;
  return stats;
}

}

tma_unit_sm::tma_unit_sm(std::vector<register_set_uniptr *> result_ports,
                         std::vector<register_set_uniptr *> reception_ports,
                         const shader_core_config *config, SM *sm)
    : functional_unit_shared_sm_part(
          result_ports, config, 1, "TMA_SM_shared", sm, MEM__OP, false, false,
          1, reception_ports, 1, nullptr, 0, false,
          TraceEnhancedOperandType::NONE) {}

void tma_unit_sm::issue(register_set_uniptr &source_reg) {
  warp_inst_t *ready_inst = source_reg.get_ready();
  if (ready_inst != nullptr) {
    if (ready_inst->active_count() > 0) {
      TMACommand cmd = build_tma_command(*ready_inst);
      cmd.completion_id = allocate_completion_object(cmd);
      m_command_queue.push(cmd);
    }
  }
  functional_unit_shared_sm_part::issue(source_reg);
}

void tma_unit_sm::cycle() {
  functional_unit_shared_sm_part::cycle();
  enqueue_issued_commands();
  advance_in_flight_transfers();
}

TMACommand tma_unit_sm::build_tma_command(const warp_inst_t &inst) const {
  TMACommand cmd;
  TMAResolvedSiteMetadata metadata;
  cmd.warp_id = inst.warp_id();
  cmd.sm_id = m_sm->get_sid();
  cmd.subcore_id = inst.get_subcore_id();
  if (m_sm->get_shd_warp(inst.warp_id()) != nullptr) {
    cmd.cta_id = m_sm->get_shd_warp(inst.warp_id())->get_cta_id();
  }
  cmd.opcode_family = inst.tma_opcode_family;
  cmd.direction = classify_tma_direction(inst);
  cmd.transfer_type = classify_tma_transfer_type(inst, cmd.opcode_family);
  cmd.operand_form = classify_tma_operand_form(cmd.opcode_family);
  cmd.meta_source = TMAMetadataSource::NONE;
  if (m_sm->get_gpu()->lookup_tma_site_metadata(inst.unique_function_id, inst.pc,
                                                inst.tma_descriptor_ptr,
                                                metadata)) {
    if (metadata.has_descriptor_metadata) {
      cmd.config_id = metadata.config_id;
      cmd.mapping_method = metadata.mapping_method;
      cmd.resolver_confidence = metadata.resolver_confidence;
      cmd.rank = metadata.descriptor_config.tensor_rank;
      cmd.box_dim = metadata.descriptor_config.box_dim;
      cmd.element_size = metadata.descriptor_config.element_size;
      cmd.swizzle = metadata.descriptor_config.swizzle;
      cmd.interleave = metadata.descriptor_config.interleave;
      cmd.oob_fill = metadata.descriptor_config.oob_fill;
      cmd.l2_promotion = metadata.descriptor_config.l2_promotion;
      cmd.total_bytes = infer_descriptor_total_bytes(metadata.descriptor_config);
      cmd.requests_total = infer_descriptor_request_total(
          metadata.descriptor_config);
      cmd.meta_source = TMAMetadataSource::DESCRIPTOR;
    }
    if (metadata.has_operand_metadata) {
      cmd.operand_form = metadata.operand_form;
      if (metadata.has_covered_bytes) {
        cmd.covered_bytes = metadata.covered_bytes;
        if (cmd.opcode_family == TMAOpcodeFamily::UBLKRED) {
          cmd.total_bytes = metadata.covered_bytes;
          cmd.requests_total =
              infer_request_total_from_covered_bytes(metadata.covered_bytes);
        } else if (cmd.total_bytes == 0) {
          cmd.total_bytes = metadata.covered_bytes;
          cmd.requests_total =
              infer_request_total_from_covered_bytes(metadata.covered_bytes);
        }
      }
      if (metadata.has_operand3_raw) {
        cmd.operand3_raw = metadata.operand3_raw;
      }
      if (cmd.meta_source == TMAMetadataSource::DESCRIPTOR) {
        cmd.meta_source = TMAMetadataSource::MIXED;
      } else {
        cmd.meta_source = TMAMetadataSource::OPERAND;
        cmd.mapping_method = metadata.mapping_method;
        cmd.resolver_confidence = metadata.resolver_confidence;
      }
    }
  }
  assert(metadata.operand_lookup_hit &&
         "Phase 2 expected runtime-observed operand resolver entry for executed TMA op");
  assert(metadata.runtime_observed &&
         "Phase 2 expected runtime_observed=true for executed TMA op");
  if (tma_family_requires_descriptor(cmd.opcode_family)) {
    assert(metadata.valid && "Phase 2 expected TMA metadata for descriptor-backed family");
    if (!metadata.has_descriptor_metadata) {
      log_tma_phase2_missing_descriptor_once(inst, cmd, metadata);
    }
    assert(metadata.has_descriptor_metadata &&
           "Phase 2 missing descriptor metadata for descriptor-backed TMA family");
  }
  if (tma_family_requires_operand_metadata(cmd.opcode_family)) {
    assert(metadata.valid && "Phase 2 expected TMA metadata for operand-sensitive family");
  }
  if (cmd.opcode_family == TMAOpcodeFamily::UTMALDG ||
      cmd.opcode_family == TMAOpcodeFamily::UTMASTG ||
      cmd.opcode_family == TMAOpcodeFamily::UTMAPF ||
      cmd.opcode_family == TMAOpcodeFamily::UTMAREDG) {
    assert(!cmd.config_id.empty() &&
           "Phase 2 missing config_id for descriptor-backed TMA command");
    assert(cmd.total_bytes > 0 &&
           "Phase 2 expected nonzero total_bytes for descriptor-backed TMA command");
    assert(cmd.requests_total > 0 &&
           "Phase 2 expected nonzero requests_total for descriptor-backed TMA command");
  }
  if (cmd.opcode_family == TMAOpcodeFamily::UBLKRED &&
      cmd.operand_form == TMAOperandForm::EXPLICIT_DESC) {
    assert(cmd.covered_bytes > 0 &&
           "Phase 2 expected covered_bytes for descriptor-backed UBLKRED");
    assert(cmd.total_bytes == cmd.covered_bytes &&
           "Phase 2 descriptor-backed UBLKRED should use covered_bytes as total_bytes");
    assert(cmd.requests_total ==
               infer_request_total_from_covered_bytes(cmd.covered_bytes) &&
           "Phase 2 descriptor-backed UBLKRED should derive requests_total from covered_bytes");
  }
  if (cmd.opcode_family == TMAOpcodeFamily::UBLKCP ||
      cmd.opcode_family == TMAOpcodeFamily::UBLKPF) {
    assert(cmd.covered_bytes > 0 &&
           "Phase 2 bulk UBLKCP/UBLKPF expected nonzero covered_bytes");
    assert(cmd.total_bytes == cmd.covered_bytes &&
           "Phase 2 bulk UBLKCP/UBLKPF should use covered_bytes as total_bytes");
    assert(cmd.requests_total ==
               infer_request_total_from_covered_bytes(cmd.covered_bytes) &&
           "Phase 2 bulk UBLKCP/UBLKPF should derive requests_total from covered_bytes");
  }
  get_tma_phase2_binding_stats().record(inst, metadata);
  log_tma_phase2_binding_once(inst, cmd);
  return cmd;
}

uint32_t tma_unit_sm::allocate_completion_object(const TMACommand &cmd) {
  TMACompletionObject completion;
  completion.expected_tx_bytes = cmd.total_bytes;
  completion.completed_tx_bytes = 0;
  completion.phase = 0;
  completion.ready = false;
  completion.warp_id = cmd.warp_id;
  completion.cta_id = cmd.cta_id;
  completion.cycle_ready = -1;
  m_completion_objects.push_back(completion);
  return static_cast<uint32_t>(m_completion_objects.size() - 1);
}

void tma_unit_sm::enqueue_issued_commands() {
  if (m_command_queue.empty()) {
    return;
  }

  TMATransferEntry entry;
  entry.cmd = m_command_queue.front();
  entry.state = TMATransferEntry::State::ENQUEUED;
  entry.cycle_enqueued = static_cast<int>(m_sm->get_current_gpu_cycle());
  entry.completion_id = entry.cmd.completion_id;
  m_in_flight_transfers.push_back(entry);
  m_command_queue.pop();
}

void tma_unit_sm::advance_in_flight_transfers() {
  int current_cycle = static_cast<int>(m_sm->get_current_gpu_cycle());
  for (auto &entry : m_in_flight_transfers) {
    switch (entry.state) {
      case TMATransferEntry::State::ENQUEUED:
        entry.state = TMATransferEntry::State::AGU_READY;
        entry.cycle_agu_ready = current_cycle;
        break;
      case TMATransferEntry::State::AGU_READY:
        entry.state = TMATransferEntry::State::IN_FLIGHT;
        entry.cycle_first_request = current_cycle;
        break;
      case TMATransferEntry::State::IN_FLIGHT: {
        uint32_t request_goal = infer_minimum_request_count(entry.cmd);
        if (request_goal == 0) {
          entry.state = TMATransferEntry::State::COMPLETED;
          entry.cycle_last_completion = current_cycle;
          break;
        }
        if (entry.requests_issued < request_goal) {
          entry.requests_issued++;
        }
        if (entry.requests_completed < entry.requests_issued) {
          entry.requests_completed++;
        }
        if (entry.requests_completed >= request_goal) {
          entry.state = TMATransferEntry::State::COMPLETED;
          entry.cycle_last_completion = current_cycle;
          if (entry.completion_id < m_completion_objects.size()) {
            TMACompletionObject &completion =
                m_completion_objects[entry.completion_id];
            completion.completed_tx_bytes = entry.cmd.total_bytes;
            completion.ready = true;
            completion.cycle_ready = current_cycle;
          }
          m_sm->notify_tma_completion(entry.cmd.warp_id, entry.cmd.total_bytes);
        }
        break;
      }
      case TMATransferEntry::State::COMPLETED:
      case TMATransferEntry::State::WAIT_SATISFIED:
      case TMATransferEntry::State::ISSUED:
        break;
    }
  }
}
