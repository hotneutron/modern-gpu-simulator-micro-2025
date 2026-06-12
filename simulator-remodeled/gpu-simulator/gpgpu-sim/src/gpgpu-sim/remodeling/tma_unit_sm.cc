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
                                 const TMACommand &cmd) {
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

  void record(const std::tuple<unsigned int, uint64_t, uint32_t> &key,
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
                               inst.tma_handle_hi);
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
  get_tma_phase2_binding_stats().record(inst, metadata);
  log_tma_phase2_binding_once(inst, cmd);
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

  m_sm->debug_log_tma_event(
      "enqueue uid=" + std::to_string(entry.transfer_uid) +
      " warp=" + std::to_string(entry.cmd.warp_id) +
      " dir=" + std::to_string(static_cast<int>(entry.cmd.direction)) +
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

    // Synthetic, deterministic GMEM base address for this 128B AGU request. The
    // trace does not carry the descriptor base, so we fabricate a per-transfer
    // address range purely to exercise memory-hierarchy timing. TMA transfers
    // bypass L1 and go directly to L2/DRAM via the shared interconnect.
    new_addr_type agu_base =
        (static_cast<new_addr_type>(entry.transfer_uid) << 20) +
        (static_cast<new_addr_type>(agu_index) * MAX_MEMORY_ACCESS_SIZE);

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

  uint32_t agu_request_goal = entry.cmd.requests_total;
  if (agu_request_goal == 0 && entry.cmd.total_bytes > 0) {
    agu_request_goal = 1;
  }
  // requests are counted in 32B sector mfs (see mover_issue_requests). A
  // reduce-store emits 2 mfs (read + write) per sector, so its goal is 2x.
  const bool is_reduction =
      (entry.cmd.transfer_type == TMATransferType::REDUCTION);
  const uint32_t mfs_per_sector = is_reduction ? 2u : 1u;
  uint32_t sector_mf_goal =
      agu_request_goal * SECTOR_CHUNCK_SIZE * mfs_per_sector;
  if (entry.requests_completed >= sector_mf_goal) {
    entry.state = TMATransferEntry::State::COMPLETED;
    entry.cycle_last_completion = current_cycle;
    ++m_stat_transfers_completed;
    m_sm->debug_log_tma_event(
        "complete uid=" + std::to_string(entry.transfer_uid) +
        " warp=" + std::to_string(entry.cmd.warp_id) +
        " dir=" + std::to_string(static_cast<int>(entry.cmd.direction)) +
        " reduce=" + std::to_string(is_reduction ? 1 : 0) +
        " bytes=" + std::to_string(entry.cmd.total_bytes) +
        " requests=" + std::to_string(entry.requests_completed) +
        " sector_goal=" + std::to_string(sector_mf_goal) +
        " cycle=" + std::to_string(current_cycle));
    // Only GMEM->SMEM loads credit the Hopper mbarrier transaction-count
    // (complete_tx). Store/reduce completion uses the bulk async-group
    // (commit-group / wait-group) mechanism, modeled separately by the
    // UTMACMDFLUSH warp-stall; it must NOT drive the load-side mbarrier.
    if (entry.cmd.direction == TMADirection::GMEM_TO_SMEM) {
      m_sm->notify_tma_completion(entry.cmd.warp_id, entry.cmd.total_bytes);
    }
  }
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
  std::cerr << "[TMA][Phase3][Stats] sm=" << m_sm->get_sid()
            << " commands_issued=" << m_stat_commands_issued
            << " transfers_completed=" << m_stat_transfers_completed
            << " requests_issued=" << m_stat_requests_issued
            << " requests_completed=" << m_stat_requests_completed
            << " bytes_issued=" << m_stat_bytes_issued
            << " bytes_completed=" << m_stat_bytes_completed
            << " (TMA traffic counted separately from L1/ldst)" << std::endl;
}
