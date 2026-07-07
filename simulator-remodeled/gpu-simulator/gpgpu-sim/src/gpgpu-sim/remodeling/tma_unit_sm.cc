#include "tma_unit_sm.h"

#include <cassert>
#include <iostream>
#include <set>
#include <tuple>

#include "../../abstract_hardware_model.h"
#include "../gpu-sim.h"
#include "../mem_fetch.h"
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
      // Both forms are handled in Phase 4 via the same covered-span size source
      // and REDUCTION transfer type (see classify_tma_transfer_type, which keys
      // off the family, not the operand form). The FA3 backward trace only
      // exercises the descriptor-backed form (EXPLICIT_DESC, set per-site by the
      // operand resolver below); the bulk non-descriptor form also routes here
      // but is NOT validated (no FA3 coverage). BULK_OPERAND is the static
      // default before the resolver overrides it.
      return TMAOperandForm::BULK_OPERAND;
    default:
      return TMAOperandForm::GENERIC;
  }
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

bool tma_site_requires_descriptor(TMAOpcodeFamily family,
                                  TMAOperandForm operand_form) {
  if (tma_family_requires_descriptor(family)) {
    return true;
  }
  if (family == TMAOpcodeFamily::UBLKRED &&
      operand_form == TMAOperandForm::EXPLICIT_DESC) {
    return true;
  }
  return false;
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
                                 const TMACommand &cmd,
                                 bool enable) {
  if (!enable) {
    return;
  }
  static std::set<std::tuple<unsigned int, uint64_t, uint32_t>> logged_sites;
  auto key = std::make_tuple(inst.unique_function_id, static_cast<uint64_t>(inst.pc),
                             inst.tma_handle_hi);
  if (!logged_sites.insert(key).second) {
    return;
  }
  std::cerr << "[TMA][Phase2] ufid=" << inst.unique_function_id
            << " pc=0x" << std::hex << static_cast<uint64_t>(inst.pc)
            << " handle_hi=0x" << inst.tma_handle_hi << std::dec
            << " family=" << static_cast<int>(cmd.opcode_family)
            << " meta_source=" << static_cast<int>(cmd.meta_source)
            << " config_id=" << cmd.config_id
            << " total_bytes=" << cmd.total_bytes
            << " requests_total=" << cmd.requests_total
            << " covered_bytes=" << cmd.covered_bytes
            << " operand3_raw=" << cmd.operand3_raw
            << " operand_form=" << static_cast<int>(cmd.operand_form)
            << " has_real_base=" << (cmd.has_real_base ? 1 : 0)
            << " global_base=0x" << std::hex << cmd.global_base << std::dec
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
  std::set<std::tuple<unsigned int, uint64_t, uint32_t>> descriptor_key_sites;
  std::set<std::tuple<unsigned int, uint64_t, uint32_t>> operand_only_sites;
  std::set<std::tuple<unsigned int, uint64_t, uint32_t>> mixed_sites;
  std::set<std::tuple<unsigned int, uint64_t, uint32_t>> unresolved_sites;
  uint64_t descriptor_key_commands = 0;
  uint64_t operand_only_commands = 0;
  uint64_t mixed_commands = 0;
  uint64_t unresolved_commands = 0;
  // Real-base coverage (only meaningful when -tma_real_base_addr_enable is on):
  // real_base_* = a static base was applied; synthetic_* = fell back to the
  // transfer_uid scheme (operand_addressed sites like UBLKRED/UBLKCP, or flag off).
  std::set<std::tuple<unsigned int, uint64_t, uint32_t>> real_base_sites;
  std::set<std::tuple<unsigned int, uint64_t, uint32_t>> synthetic_sites;
  uint64_t real_base_commands = 0;
  uint64_t synthetic_commands = 0;

  void record(const std::tuple<unsigned int, uint64_t, uint32_t> &key,
              const TMAResolvedSiteMetadata &metadata, bool has_real_base) {
    if (has_real_base) {
      real_base_sites.insert(key);
      ++real_base_commands;
    } else {
      synthetic_sites.insert(key);
      ++synthetic_commands;
    }
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

  void record(const warp_inst_t &inst, const TMAResolvedSiteMetadata &metadata,
              bool has_real_base) {
    auto key = std::make_tuple(inst.unique_function_id,
                               static_cast<uint64_t>(inst.pc),
                               inst.tma_handle_hi);
    overall.record(key, metadata, has_real_base);
    TMAPhase2FamilyStats *family_stats = select_family_stats(inst.tma_opcode_family);
    if (family_stats != nullptr) {
      family_stats->record(key, metadata, has_real_base);
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
    // Real-base coverage: how many sites/commands used the exact GMEM base vs the
    // synthetic fallback. With the flag on, real_base should cover all descriptor
    // sites; synthetic should be only operand_addressed (UBLKRED/UBLKCP).
    std::cerr << "[TMA][Phase2][Stats][" << label << "] real_base sites="
              << stats.real_base_sites.size()
              << " synthetic_sites=" << stats.synthetic_sites.size()
              << " | real_base_commands=" << stats.real_base_commands
              << " synthetic_commands=" << stats.synthetic_commands << std::endl;
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
                         const shader_core_config *config, SM *sm,
                         mem_fetch_interface *icnt,
                         mem_fetch_allocator *mf_allocator)
    : functional_unit_shared_sm_part(
          result_ports, config, 1, "TMA_SM_shared", sm, MEM__OP, false, false,
          1, reception_ports, 1, nullptr, 0, false,
          TraceEnhancedOperandType::NONE),
      m_icnt(icnt),
      m_mf_allocator(mf_allocator) {}

tma_unit_sm::~tma_unit_sm() { debug_dump_tma_counters(); }

void tma_unit_sm::issue(register_set_uniptr &source_reg) {
  warp_inst_t *ready_inst = source_reg.get_ready();
  if (ready_inst != nullptr) {
    if (ready_inst->active_count() > 0) {
      TMACommand cmd = build_tma_command(*ready_inst);
      m_command_queue.push(cmd);
      ++m_stat_commands_issued;
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
  // UTMALDG.MULTICAST distributes one descriptor load across multiple CTAs via
  // a ctaMask. The ctaMask is not tracked in the trace/sidecar, so modeling it
  // as a single-CTA load would silently misattribute traffic. Refuse it
  // explicitly until ctaMask handling is implemented (FA2-only feature).
  if (inst.tma_is_multicast) {
    m_sm->debug_log_tma_event(
        "MULTICAST-refused warp=" + std::to_string(inst.warp_id()) +
        " pc=" + std::to_string(inst.pc) +
        " (UTMALDG.MULTICAST not implemented, ctaMask unmodeled)");
  }
  assert(!inst.tma_is_multicast &&
         "UTMALDG.MULTICAST not implemented (ctaMask unmodeled)");
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
                                                inst.tma_handle_hi,
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
  // Real per-site GMEM base (tma_pc_base_map.json), looked up here because (uid,pc) is
  // still available — the mover only sees TMATransferEntry and cannot re-derive it.
  // Gated by -tma_real_base_addr_enable; when off, has_real_base stays false and the
  // mover keeps the synthetic address. Only tensormap-addressed sites carry a static
  // base (UBLKRED/UBLKCP are operand_addressed -> no real base, synthetic retained).
  if (m_config->tma_real_base_addr_enable) {
    TMABaseRecord base_record;
    bool base_hit = m_sm->get_gpu()->lookup_tma_base_record(
        inst.unique_function_id, inst.pc, base_record);
    if (base_hit && base_record.has_static_base) {
      cmd.global_base = base_record.global_base;
      cmd.has_real_base = true;
      // SIZE CROSS-CHECK (M1 prerequisite for cutting the config_id path in M3):
      // the base map carries box_dim/element_size for the exact site, so it must
      // produce the SAME total_bytes/requests_total as the config_id path already
      // stored in cmd. A mismatch means the two descriptor sources disagree about
      // how much data moves -> wrong traffic volume. Fail on the FIRST such command
      // so a 12h run dies immediately instead of producing bad numbers.
      TMADescriptorConfigMetadata base_shape;
      base_shape.box_dim = base_record.box_dim;
      base_shape.element_size = base_record.element_size;
      uint32_t base_total_bytes = infer_descriptor_total_bytes(base_shape);
      uint32_t base_requests_total = infer_descriptor_request_total(base_shape);
      if (cmd.total_bytes != 0 && base_total_bytes != cmd.total_bytes) {
        std::cerr << "[TMA][RealBase][FATAL] total_bytes mismatch ufid="
                  << inst.unique_function_id << " pc=0x" << std::hex
                  << static_cast<uint64_t>(inst.pc) << std::dec
                  << " base_map=" << base_total_bytes
                  << " config_path=" << cmd.total_bytes
                  << " (base_map box/element_size disagrees with config_id path)"
                  << std::endl;
      }
      assert((cmd.total_bytes == 0 || base_total_bytes == cmd.total_bytes) &&
             "real-base total_bytes must match config_id path (size cross-check)");
      if (cmd.requests_total != 0 &&
          base_requests_total != cmd.requests_total) {
        std::cerr << "[TMA][RealBase][FATAL] requests_total mismatch ufid="
                  << inst.unique_function_id << " pc=0x" << std::hex
                  << static_cast<uint64_t>(inst.pc) << std::dec
                  << " base_map=" << base_requests_total
                  << " config_path=" << cmd.requests_total << std::endl;
      }
      assert((cmd.requests_total == 0 ||
              base_requests_total == cmd.requests_total) &&
             "real-base requests_total must match config_id path (size cross-check)");
      // M2 (visit-counter tile spread): base-only collapses every transfer of a tensor
      // to global_base+agu_index*128 (one 16KB tile), erasing the cold miss of the
      // tensor's other tiles. Spread transfers across all tiles: tile_bytes = one box
      // (=base_total_bytes), tensor_bytes = Πglobal_dim·element_size, num_tiles =
      // ⌈tensor_bytes/tile_bytes⌉. The per-tensor visit counter picks tile_idx =
      // count % num_tiles so repeated visits to the same tile still hit while distinct
      // tiles occupy distinct L2 lines. Deterministic approximation of the real
      // schedule (coords are not in the trace).
      uint64_t tile_bytes = base_total_bytes;  // Πbox_dim · element_size
      if (tile_bytes > 0) {
        uint64_t tensor_elems = 1;
        bool has_extent = false;
        for (uint32_t d : base_record.global_dim) {
          if (d == 0) continue;
          tensor_elems *= d;
          has_extent = true;
        }
        uint64_t tensor_bytes =
            has_extent ? tensor_elems * base_record.element_size : 0;
        uint64_t num_tiles =
            tensor_bytes > 0 ? (tensor_bytes + tile_bytes - 1) / tile_bytes : 1;
        if (num_tiles == 0) num_tiles = 1;
        uint64_t &visit = m_tensor_visit_count[base_record.global_base];
        uint64_t tile_idx = visit % num_tiles;
        ++visit;
        cmd.tile_offset_bytes = tile_idx * tile_bytes;
      }
    } else if (tma_family_requires_descriptor(cmd.opcode_family)) {
      // MISSING-MAP ASSERT: a descriptor-required op (UTMALDG/UTMASTG/UTMAPF/
      // UTMAREDG) must have an exact base when the flag is on. UBLKRED/UBLKCP are
      // operand_addressed (no static base) and are excluded by this branch, so they
      // fall through to the synthetic path without asserting. Fail early so a base-map
      // coverage gap is caught before a 12h run rather than silently falling back.
      std::cerr << "[TMA][RealBase][FATAL] descriptor-required site has no static base "
                << "ufid=" << inst.unique_function_id << " pc=0x" << std::hex
                << static_cast<uint64_t>(inst.pc) << std::dec
                << " family=" << tma_phase2_family_label(cmd.opcode_family)
                << " base_hit=" << (base_hit ? 1 : 0)
                << " (tma_pc_base_map.json missing this (uid,pc) or operand_addressed)"
                << std::endl;
      assert(false &&
             "descriptor-required TMA site missing real base (base-map coverage gap)");
    }
  }
  assert(metadata.operand_lookup_hit &&
         "Phase 2 expected runtime-observed operand resolver entry for executed TMA op");
  assert(metadata.runtime_observed &&
         "Phase 2 expected runtime_observed=true for executed TMA op");
  if (tma_site_requires_descriptor(cmd.opcode_family, cmd.operand_form)) {
    assert(metadata.valid &&
           "Phase 2 expected TMA metadata for descriptor-required TMA site");
    assert(metadata.has_descriptor_metadata &&
           "Phase 2 missing descriptor metadata for descriptor-required TMA site");
    assert(!cmd.config_id.empty() &&
           "Phase 2 missing config_id for descriptor-required TMA site");
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
  get_tma_phase2_binding_stats().record(inst, metadata, cmd.has_real_base);
  log_tma_phase2_binding_once(inst, cmd, m_config->sync_debug_enable);
  return cmd;
}

void tma_unit_sm::enqueue_issued_commands() {
  if (m_command_queue.empty()) {
    return;
  }

  TMATransferEntry entry;
  entry.cmd = m_command_queue.front();
  entry.state = TMATransferEntry::State::ENQUEUED;
  entry.cycle_enqueued = static_cast<int>(m_sm->get_current_gpu_cycle());
  entry.transfer_uid = m_next_transfer_uid++;
  m_in_flight_transfers.push_back(entry);
  m_command_queue.pop();

  // Track store-class transfers (SMEM->GMEM: UTMASTG / UTMAREDG / UBLKRED) per
  // warp from enqueue until completion. A UTMACMDFLUSH (wait_group 0) drains all
  // of them for its warp. Counting from enqueue (not from IN_FLIGHT) keeps the
  // drain-all semantics exact even for stores still waiting in the AGU stage.
  if (entry.cmd.direction == TMADirection::SMEM_TO_GMEM) {
    uint32_t outstanding = ++m_outstanding_stores_per_warp[entry.cmd.warp_id];
    m_sm->debug_log_tma_event(
        "store-outstanding++ uid=" + std::to_string(entry.transfer_uid) +
        " warp=" + std::to_string(entry.cmd.warp_id) +
        " reduce=" +
        std::to_string(
            entry.cmd.transfer_type == TMATransferType::REDUCTION ? 1 : 0) +
        " outstanding=" + std::to_string(outstanding));
  }

  // Prefetch (UTMAPF / UBLKPF) shares the GMEM->SMEM direction and read-shape of
  // a load, but is fire-and-forget: it has no consumer mbarrier and must not be
  // counted as an outstanding store. Logged here so its issue is observable in
  // the trace distinct from a real load.
  if (entry.cmd.transfer_type == TMATransferType::PREFETCH) {
    m_sm->debug_log_tma_event(
        "prefetch-issue uid=" + std::to_string(entry.transfer_uid) +
        " warp=" + std::to_string(entry.cmd.warp_id) +
        " family=" +
        std::to_string(static_cast<int>(entry.cmd.opcode_family)) +
        " total_bytes=" + std::to_string(entry.cmd.total_bytes));
  }

  m_sm->debug_log_tma_event(
      "enqueue uid=" + std::to_string(entry.transfer_uid) +
      " warp=" + std::to_string(entry.cmd.warp_id) +
      " dir=" + std::to_string(static_cast<int>(entry.cmd.direction)) +
      " ttype=" + std::to_string(static_cast<int>(entry.cmd.transfer_type)) +
      " requests_total=" + std::to_string(entry.cmd.requests_total) +
      " total_bytes=" + std::to_string(entry.cmd.total_bytes));
}

void tma_unit_sm::advance_in_flight_transfers() {
  int current_cycle = static_cast<int>(m_sm->get_current_gpu_cycle());
  for (auto &entry : m_in_flight_transfers) {
    switch (entry.state) {
      case TMATransferEntry::State::ENQUEUED:
        // Descriptor / AGU block: compute transfer geometry. requests_total is
        // already derived from box_dim/element_size in Phase 2.
        entry.state = TMATransferEntry::State::AGU_READY;
        entry.cycle_agu_ready = current_cycle;
        m_sm->debug_log_tma_event(
            "agu-ready uid=" + std::to_string(entry.transfer_uid) +
            " warp=" + std::to_string(entry.cmd.warp_id) +
            " requests_total=" + std::to_string(entry.cmd.requests_total) +
            " enqueued_cycle=" + std::to_string(entry.cycle_enqueued) +
            " cycle=" + std::to_string(current_cycle));
        break;
      case TMATransferEntry::State::AGU_READY:
        // CONTROL transfers (UTMACCTL / UTMACMDFLUSH) carry no data movement.
        // The store-side wait modeled by UTMACMDFLUSH is handled separately
        // (warp-stall gate); here it just completes without issuing traffic.
        if (entry.cmd.transfer_type == TMATransferType::CONTROL ||
            entry.cmd.direction == TMADirection::NONE) {
          entry.state = TMATransferEntry::State::COMPLETED;
          entry.cycle_last_completion = current_cycle;
          m_sm->debug_log_tma_event(
              "control-passthrough uid=" + std::to_string(entry.transfer_uid) +
              " family=" +
              std::to_string(static_cast<int>(entry.cmd.opcode_family)));
          break;
        }
        // Both GMEM->SMEM (load) and SMEM->GMEM (store/reduce) data movement
        // are issued through the same mover; mover_issue_requests selects the
        // access type (read / write / read+write RMW) from the transfer type.
        entry.state = TMATransferEntry::State::IN_FLIGHT;
        entry.cycle_first_request = current_cycle;
        m_sm->debug_log_tma_event(
            "in-flight uid=" + std::to_string(entry.transfer_uid) +
            " dir=" + std::to_string(static_cast<int>(entry.cmd.direction)) +
            " cycle=" + std::to_string(current_cycle));
        break;
      case TMATransferEntry::State::IN_FLIGHT:
        mover_issue_requests(entry, current_cycle);
        break;
      case TMATransferEntry::State::COMPLETED:
      case TMATransferEntry::State::WAIT_SATISFIED:
      case TMATransferEntry::State::ISSUED:
        break;
    }
  }

  // Reclaim finished transfers so the deque does not grow without bound.
  while (!m_in_flight_transfers.empty() &&
         (m_in_flight_transfers.front().state ==
              TMATransferEntry::State::COMPLETED ||
          m_in_flight_transfers.front().state ==
              TMATransferEntry::State::WAIT_SATISFIED)) {
    m_in_flight_transfers.pop_front();
  }
}

// ---- Hopper data mover (Blackwell plugs a different mover at these hooks) ----

void tma_unit_sm::mover_issue_requests(TMATransferEntry &entry,
                                       int current_cycle) {
  // AGU throughput is modelled in units of 128B cache-line requests (the rate
  // at which the TMA address-generation unit can emit line addresses). The
  // descriptor's requests_total already counts those 128B requests.
  uint32_t agu_request_goal = entry.cmd.requests_total;
  if (agu_request_goal == 0 && entry.cmd.total_bytes > 0) {
    agu_request_goal = 1;
  }
  if (agu_request_goal == 0) {
    // Nothing to move (e.g. control op routed here): complete immediately.
    entry.state = TMATransferEntry::State::COMPLETED;
    entry.cycle_last_completion = current_cycle;
    return;
  }

  // Direction / access shape selects how each 32B sector is moved:
  //   LOAD  (GMEM->SMEM): one read  (GLOBAL_ACC_R) per sector
  //   STORE (SMEM->GMEM): one write (GLOBAL_ACC_W) per sector
  //   REDUCE-STORE      : one read  + one write per sector (elementwise RMW,
  //                       NOT a many-to-one atomic; see TMA_ARCH.md Phase 4)
  const bool is_reduction =
      (entry.cmd.transfer_type == TMATransferType::REDUCTION);
  const bool is_store =
      (entry.cmd.direction == TMADirection::SMEM_TO_GMEM) && !is_reduction;
  // Number of sector mem_fetches emitted per 32B sector of the moved region.
  const uint32_t mfs_per_sector = is_reduction ? 2u : 1u;

  // Each 128B AGU request is sent to memory as SECTOR_CHUNCK_SIZE (4) separate
  // 32B sector mem_fetches, exactly like a normal ldst request that has been
  // coalesced to 32B sectors. Emitting 32B + a single-bit sector mask keeps the
  // L2 from re-splitting the request (see
  // memory_sub_partition::breakdown_request_to_sector_requests), so one issued
  // mf maps to exactly one response and the parent/child mismatch disappears.
  // requests_issued / requests_completed are therefore counted in sector mfs;
  // a reduce-store counts 2x (read + write) per sector.
  const uint32_t kSectorMfGoal =
      agu_request_goal * SECTOR_CHUNCK_SIZE * mfs_per_sector;

  if (entry.requests_issued < kSectorMfGoal) {
    ++entry.issue_active_cycles;
  }

  uint32_t agu_requests_this_cycle = 0;
  while (entry.requests_issued < kSectorMfGoal &&
         agu_requests_this_cycle < kMaxRequestsPerCycle) {
    if (m_icnt == nullptr || m_mf_allocator == nullptr) {
      break;
    }

    // Convert the flat sector-mf index back to (128B AGU line, 32B sector,
    // rmw slot). For non-reduction mfs_per_sector == 1, so rmw_slot is always 0.
    uint32_t sector_unit = entry.requests_issued / mfs_per_sector;
    uint32_t agu_index = sector_unit / SECTOR_CHUNCK_SIZE;

    // GMEM base for this 128B AGU request. When the exact per-site base is available
    // (build_tma_command set has_real_base from tma_pc_base_map.json), use it plus the
    // M2 per-transfer tile offset (tile_offset_bytes) so different tiles of the same
    // tensor land in different L2 lines while same-tile revisits still hit. Without the
    // tile offset every transfer would collapse to the tensor's first tile (base-only).
    // Otherwise fall back to the synthetic, deterministic per-transfer range that only
    // exercises memory-hierarchy timing (the trace lacked the descriptor base).
    new_addr_type agu_base =
        entry.cmd.has_real_base
            ? (static_cast<new_addr_type>(entry.cmd.global_base) +
               static_cast<new_addr_type>(entry.cmd.tile_offset_bytes) +
               static_cast<new_addr_type>(agu_index) * MAX_MEMORY_ACCESS_SIZE)
            : ((static_cast<new_addr_type>(entry.transfer_uid) << 20) +
               (static_cast<new_addr_type>(agu_index) * MAX_MEMORY_ACCESS_SIZE));

    bool icnt_blocked = false;
    // Emit one 128B AGU line worth of sector mfs (mfs_per_sector each), resuming
    // mid-line from wherever the previous cycle stopped.
    while (entry.requests_issued < kSectorMfGoal) {
      sector_unit = entry.requests_issued / mfs_per_sector;
      if (sector_unit / SECTOR_CHUNCK_SIZE != agu_index) {
        break;  // crossed into the next 128B AGU line
      }
      uint32_t sector_in_line = sector_unit % SECTOR_CHUNCK_SIZE;
      uint32_t rmw_slot = entry.requests_issued % mfs_per_sector;
      // rmw_slot 0 = read (fetch dst), rmw_slot 1 = write (store reduced value).
      const bool this_mf_is_write = is_store || (is_reduction && rmw_slot == 1);

      // Back-pressure: stop on the first sector mf the interconnect cannot
      // accept; resume from the same mf next cycle. Stores/reductions use the
      // write side of the interconnect.
      if (m_icnt->full(SECTOR_SIZE, /*write=*/this_mf_is_write)) {
        icnt_blocked = true;
        ++entry.icnt_full_cycles;
        // Interconnect back-pressure: the transfer cannot drain its sector mfs
        // and will resume next cycle. This is a TMA-side stall source (feeds the
        // long_scoreboard axis); log once per transfer to keep it observable
        // without flooding the trace on a sustained stall.
        if (!entry.logged_backpressure) {
          entry.logged_backpressure = true;
          ++m_stat_icnt_backpressure_events;
          m_sm->debug_log_tma_event(
              "icnt-backpressure uid=" + std::to_string(entry.transfer_uid) +
              " warp=" + std::to_string(entry.cmd.warp_id) +
              " write=" + std::to_string(this_mf_is_write ? 1 : 0) +
              " requests_issued=" + std::to_string(entry.requests_issued) +
              " sector_goal=" + std::to_string(kSectorMfGoal) +
              " cycle=" + std::to_string(current_cycle));
        }
        break;
      }

      new_addr_type addr = agu_base + sector_in_line * SECTOR_SIZE;

      // 32B request carrying a single-sector mask, matching what the ldst path
      // produces after coalescing. byte_mask covers the 32 bytes of the sector.
      mem_access_sector_mask_t sector_mask;
      sector_mask.set(sector_in_line);
      mem_access_byte_mask_t byte_mask;
      for (unsigned b = 0; b < SECTOR_SIZE; ++b) {
        byte_mask.set(sector_in_line * SECTOR_SIZE + b);
      }

      mem_access_type acc_type = this_mf_is_write ? GLOBAL_ACC_W : GLOBAL_ACC_R;
      mem_fetch *mf = m_mf_allocator->alloc(
          addr, acc_type, active_mask_t().set(0), byte_mask, sector_mask,
          SECTOR_SIZE, /*wr=*/this_mf_is_write, m_sm->get_current_gpu_cycle(),
          /*wid=*/(unsigned)-1, m_sm->get_sid(), m_sm->get_tpc_id(),
          /*original_mf=*/nullptr);
      mf->set_is_tma(true);

      if (entry.cycle_first_request_issued < 0) {
        entry.cycle_first_request_issued = current_cycle;
      }
      entry.cycle_last_request_issued = current_cycle;

      if (entry.requests_issued == 0) {
        m_sm->debug_log_tma_event(
            "first-request uid=" + std::to_string(entry.transfer_uid) +
            " family=" +
            std::to_string(static_cast<int>(entry.cmd.opcode_family)) +
            " ttype=" +
            std::to_string(static_cast<int>(entry.cmd.transfer_type)) +
            " addr=" + std::to_string(addr) +
            " dir=" + std::to_string(static_cast<int>(entry.cmd.direction)) +
            " store=" + std::to_string(is_store ? 1 : 0) +
            " reduce=" + std::to_string(is_reduction ? 1 : 0) +
            " mfs_per_sector=" + std::to_string(mfs_per_sector) +
            " agu_requests=" + std::to_string(agu_request_goal) +
            " sector_mfs=" + std::to_string(kSectorMfGoal) +
            " (32B sector, L1-bypass, shared icnt)");
      }
      // For store/reduce, log the first read mf and first write mf separately so
      // the RMW (read+write) issue of a reduce-store is observable in the trace.
      if ((is_store || is_reduction)) {
        if (this_mf_is_write && !entry.logged_first_write) {
          entry.logged_first_write = true;
          m_sm->debug_log_tma_event(
              "store-write-issue uid=" + std::to_string(entry.transfer_uid) +
              " addr=" + std::to_string(addr) + " acc=GLOBAL_ACC_W" +
              " reduce=" + std::to_string(is_reduction ? 1 : 0));
        } else if (!this_mf_is_write && !entry.logged_first_read) {
          entry.logged_first_read = true;
          m_sm->debug_log_tma_event(
              "reduce-read-issue uid=" + std::to_string(entry.transfer_uid) +
              " addr=" + std::to_string(addr) + " acc=GLOBAL_ACC_R" +
              " (RMW dst fetch)");
        }
      }

      m_outstanding_requests[mf] = entry.transfer_uid;
      m_icnt->push(mf);

      ++entry.requests_issued;
      ++m_stat_requests_issued;
      m_stat_bytes_issued += SECTOR_SIZE;
      if (is_reduction) {
        m_stat_reduce_bytes_issued += SECTOR_SIZE;
      } else if (is_store) {
        m_stat_store_bytes_issued += SECTOR_SIZE;
      } else {
        m_stat_load_bytes_issued += SECTOR_SIZE;
      }
    }

    if (icnt_blocked) {
      break;
    }
    // One 128B AGU request worth of sectors emitted this iteration.
    ++agu_requests_this_cycle;
  }
}

void tma_unit_sm::mover_on_response(TMATransferEntry &entry, mem_fetch *mf,
                                    int current_cycle) {
  ++entry.requests_completed;
  ++m_stat_requests_completed;
  entry.bytes_completed += mf->get_data_size();
  m_stat_bytes_completed += mf->get_data_size();
  const bool is_reduction =
      (entry.cmd.transfer_type == TMATransferType::REDUCTION);
  const bool is_store =
      (entry.cmd.direction == TMADirection::SMEM_TO_GMEM) && !is_reduction;
  if (is_reduction) {
    m_stat_reduce_bytes_completed += mf->get_data_size();
  } else if (is_store) {
    m_stat_store_bytes_completed += mf->get_data_size();
  } else {
    m_stat_load_bytes_completed += mf->get_data_size();
  }

  uint32_t agu_request_goal = entry.cmd.requests_total;
  if (agu_request_goal == 0 && entry.cmd.total_bytes > 0) {
    agu_request_goal = 1;
  }
  // requests are counted in 32B sector mfs (see mover_issue_requests). A
  // reduce-store emits 2 mfs (read + write) per sector, so its goal is 2x.
  const uint32_t mfs_per_sector = is_reduction ? 2u : 1u;
  uint32_t sector_mf_goal =
      agu_request_goal * SECTOR_CHUNCK_SIZE * mfs_per_sector;
  if (entry.requests_completed >= sector_mf_goal) {
    entry.state = TMATransferEntry::State::COMPLETED;
    entry.cycle_last_completion = current_cycle;
    ++m_stat_transfers_completed;
    int cycle_first_request_issued = entry.cycle_first_request_issued;
    if (cycle_first_request_issued < 0) {
      cycle_first_request_issued = entry.cycle_first_request;
    }
    int cycle_last_request_issued = entry.cycle_last_request_issued;
    if (cycle_last_request_issued < 0) {
      cycle_last_request_issued = cycle_first_request_issued;
    }
    int lat_to_first_request = cycle_first_request_issued - entry.cycle_agu_ready;
    int lat_emit = cycle_last_request_issued - cycle_first_request_issued;
    int lat_drain = current_cycle - cycle_last_request_issued;
    if (cycle_first_request_issued >= 0 && cycle_last_request_issued >= 0) {
      ++m_stat_timed_transfers;
      m_stat_issue_active_cycles += entry.issue_active_cycles;
      m_stat_icnt_full_cycles += entry.icnt_full_cycles;
      m_stat_to_first_request_cycles +=
          lat_to_first_request > 0 ? lat_to_first_request : 0;
      m_stat_emit_span_cycles += lat_emit > 0 ? lat_emit : 0;
      m_stat_drain_cycles += lat_drain > 0 ? lat_drain : 0;
    }
    double requests_per_issue_active_cycle =
        entry.issue_active_cycles
            ? (double)entry.requests_issued / (double)entry.issue_active_cycles
            : 0.0;
    m_sm->debug_log_tma_event(
        "complete uid=" + std::to_string(entry.transfer_uid) +
        " warp=" + std::to_string(entry.cmd.warp_id) +
        " dir=" + std::to_string(static_cast<int>(entry.cmd.direction)) +
        " reduce=" + std::to_string(is_reduction ? 1 : 0) +
        " bytes=" + std::to_string(entry.cmd.total_bytes) +
        " requests=" + std::to_string(entry.requests_completed) +
        " sector_goal=" + std::to_string(sector_mf_goal) +
        // Per-transfer latency breakdown (cycles). This is the primary signal
        // for the HW-vs-sim cycle gap: it splits the modeled transfer latency
        // into queueing / issue-serialization / memory-roundtrip so the
        // dominant over-estimated stage is identifiable per transfer.
        //   lat_total = enqueue -> complete
        //   lat_queue = enqueue -> agu_ready   (descriptor/AGU wait)
        //   lat_issue = agu_ready -> first_req  (request-issue serialization)
        //   lat_mem   = first_req -> complete   (interconnect+L2+DRAM roundtrip)
        " lat_total=" + std::to_string(current_cycle - entry.cycle_enqueued) +
        " lat_queue=" + std::to_string(entry.cycle_agu_ready - entry.cycle_enqueued) +
        " lat_issue=" + std::to_string(entry.cycle_first_request - entry.cycle_agu_ready) +
        " lat_mem=" + std::to_string(current_cycle - entry.cycle_first_request) +
        " lat_to_first_request=" + std::to_string(lat_to_first_request) +
        " lat_emit=" + std::to_string(lat_emit) +
        " lat_drain=" + std::to_string(lat_drain) +
        " issue_active_cycles=" + std::to_string(entry.issue_active_cycles) +
        " icnt_full_cycles=" + std::to_string(entry.icnt_full_cycles) +
        " requests_per_issue_active_cycle=" +
        std::to_string(requests_per_issue_active_cycle) +
        " cycle=" + std::to_string(current_cycle));
    // Only real GMEM->SMEM loads (UTMALDG/UBLKCP) credit the Hopper mbarrier
    // transaction-count (complete_tx). Prefetch (UTMAPF/UBLKPF) also has
    // direction GMEM_TO_SMEM but must NOT credit any mbarrier: it carries no
    // arrive/complete_tx contract, so crediting it would corrupt the
    // transaction-count accounting of an unrelated UTMALDG load. Store/reduce
    // completion uses the bulk async-group (commit-group / wait-group)
    // mechanism, modeled separately by the UTMACMDFLUSH warp-stall.
    if (entry.cmd.transfer_type == TMATransferType::LOAD) {
      m_sm->notify_tma_completion(entry.cmd.warp_id, entry.cmd.total_bytes);
      // Symmetric with the prefetch branch's "mbarrier_credited=0": make the
      // producer->consumer handshake observable. A real load credits the
      // consumer mbarrier's transaction-count (complete_tx) with total_bytes,
      // which is what releases a warp stalled in wait_barrier (the TMA axis).
      m_sm->debug_log_tma_event(
          "load-mbarrier-credit uid=" + std::to_string(entry.transfer_uid) +
          " warp=" + std::to_string(entry.cmd.warp_id) +
          " bytes=" + std::to_string(entry.cmd.total_bytes) +
          " mbarrier_credited=1"
          " cycle=" + std::to_string(current_cycle));
    } else if (entry.cmd.transfer_type == TMATransferType::PREFETCH) {
      // Fire-and-forget prefetch: data has landed in L2/SMEM-staging but there
      // is no consumer barrier to signal. Just log and retire silently.
      // ttype is logged so the trace can prove this transfer took the PREFETCH
      // branch and therefore did NOT call notify_tma_completion (the latent bug
      // this branch closes).
      m_sm->debug_log_tma_event(
          "prefetch-complete uid=" + std::to_string(entry.transfer_uid) +
          " warp=" + std::to_string(entry.cmd.warp_id) +
          " family=" +
          std::to_string(static_cast<int>(entry.cmd.opcode_family)) +
          " ttype=" + std::to_string(static_cast<int>(entry.cmd.transfer_type)) +
          " bytes=" + std::to_string(entry.cmd.total_bytes) +
          " mbarrier_credited=0"
          " cycle=" + std::to_string(current_cycle));
    } else {
      // Store-class transfer finished: retire it from the warp's bulk
      // async-group so a UTMACMDFLUSH waiting on this warp can make progress.
      auto it = m_outstanding_stores_per_warp.find(entry.cmd.warp_id);
      uint32_t outstanding = 0;
      if (it != m_outstanding_stores_per_warp.end() && it->second > 0) {
        outstanding = --(it->second);
      }
      m_sm->debug_log_tma_event(
          "store-outstanding-- uid=" + std::to_string(entry.transfer_uid) +
          " warp=" + std::to_string(entry.cmd.warp_id) +
          " reduce=" + std::to_string(is_reduction ? 1 : 0) +
          " outstanding=" + std::to_string(outstanding));
    }
  }
}

bool tma_unit_sm::warp_has_outstanding_stores(unsigned int warp_id) const {
  auto it = m_outstanding_stores_per_warp.find(warp_id);
  return it != m_outstanding_stores_per_warp.end() && it->second > 0;
}

void tma_unit_sm::fill(mem_fetch *mf) {
  auto it = m_outstanding_requests.find(mf);
  assert(it != m_outstanding_requests.end() &&
         "TMA fill received a mem_fetch not issued by the TMA unit");
  uint64_t transfer_uid = it->second;
  m_outstanding_requests.erase(it);

  int current_cycle = static_cast<int>(m_sm->get_current_gpu_cycle());
  for (auto &entry : m_in_flight_transfers) {
    if (entry.transfer_uid == transfer_uid) {
      mover_on_response(entry, mf, current_cycle);
      break;
    }
  }
}

void tma_unit_sm::debug_dump_tma_counters() const {
  if (m_stat_commands_issued == 0) {
    return;
  }
  const double elapsed_seconds =
      (double)m_sm->get_current_gpu_cycle() *
      m_sm->get_gpu()->get_config().get_core_period();
  const double issued_bw =
      elapsed_seconds > 0.0
          ? ((double)m_stat_bytes_issued / elapsed_seconds) / 1000000000.0
          : 0.0;
  const double completed_bw =
      elapsed_seconds > 0.0
          ? ((double)m_stat_bytes_completed / elapsed_seconds) / 1000000000.0
          : 0.0;
  const double load_issued_bw =
      elapsed_seconds > 0.0
          ? ((double)m_stat_load_bytes_issued / elapsed_seconds) /
                1000000000.0
          : 0.0;
  const double store_issued_bw =
      elapsed_seconds > 0.0
          ? ((double)m_stat_store_bytes_issued / elapsed_seconds) /
                1000000000.0
          : 0.0;
  const double reduce_issued_bw =
      elapsed_seconds > 0.0
          ? ((double)m_stat_reduce_bytes_issued / elapsed_seconds) /
                1000000000.0
          : 0.0;
  const double avg_issue_active_cycles =
      m_stat_timed_transfers
          ? (double)m_stat_issue_active_cycles / (double)m_stat_timed_transfers
          : 0.0;
  const double avg_icnt_full_cycles =
      m_stat_timed_transfers
          ? (double)m_stat_icnt_full_cycles / (double)m_stat_timed_transfers
          : 0.0;
  const double avg_to_first_request_cycles =
      m_stat_timed_transfers
          ? (double)m_stat_to_first_request_cycles /
                (double)m_stat_timed_transfers
          : 0.0;
  const double avg_emit_span_cycles =
      m_stat_timed_transfers
          ? (double)m_stat_emit_span_cycles / (double)m_stat_timed_transfers
          : 0.0;
  const double avg_drain_cycles =
      m_stat_timed_transfers
          ? (double)m_stat_drain_cycles / (double)m_stat_timed_transfers
          : 0.0;
  const double avg_requests_per_issue_active_cycle =
      m_stat_issue_active_cycles
          ? (double)m_stat_requests_issued / (double)m_stat_issue_active_cycles
          : 0.0;
  std::cerr << "[TMA][Phase3][Stats] sm=" << m_sm->get_sid()
            << " commands_issued=" << m_stat_commands_issued
            << " transfers_completed=" << m_stat_transfers_completed
            << " requests_issued=" << m_stat_requests_issued
            << " requests_completed=" << m_stat_requests_completed
            << " bytes_issued=" << m_stat_bytes_issued
            << " bytes_completed=" << m_stat_bytes_completed
            << " load_bytes_issued=" << m_stat_load_bytes_issued
            << " store_bytes_issued=" << m_stat_store_bytes_issued
            << " reduce_bytes_issued=" << m_stat_reduce_bytes_issued
            << " load_bytes_completed=" << m_stat_load_bytes_completed
            << " store_bytes_completed=" << m_stat_store_bytes_completed
            << " reduce_bytes_completed=" << m_stat_reduce_bytes_completed
            << " icnt_backpressure_events=" << m_stat_icnt_backpressure_events
            << " BW_issued_GBps=" << issued_bw
            << " BW_completed_GBps=" << completed_bw
            << " BW_load_issued_GBps=" << load_issued_bw
            << " BW_store_issued_GBps=" << store_issued_bw
            << " BW_reduce_issued_GBps=" << reduce_issued_bw
            << " timed_transfers=" << m_stat_timed_transfers
            << " avg_issue_active_cycles=" << avg_issue_active_cycles
            << " avg_icnt_full_cycles=" << avg_icnt_full_cycles
            << " avg_to_first_request_cycles="
            << avg_to_first_request_cycles
            << " avg_emit_span_cycles=" << avg_emit_span_cycles
            << " avg_drain_cycles=" << avg_drain_cycles
            << " avg_requests_per_issue_active_cycle="
            << avg_requests_per_issue_active_cycle
            << " (TMA traffic counted separately from L1/ldst)" << std::endl;
}
