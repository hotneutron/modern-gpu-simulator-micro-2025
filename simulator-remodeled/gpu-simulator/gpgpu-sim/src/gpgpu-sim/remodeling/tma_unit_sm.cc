#include "tma_unit_sm.h"

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
    case TMAOpcodeFamily::UBLKCP:
    case TMAOpcodeFamily::UBLKPF:
    case TMAOpcodeFamily::UBLKRED:
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
    TMACommand cmd = build_tma_command(*ready_inst);
    cmd.completion_id = allocate_completion_object(cmd);
    m_command_queue.push(cmd);
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
  cmd.warp_id = inst.warp_id();
  cmd.sm_id = m_sm->get_sid();
  cmd.subcore_id = inst.get_subcore_id();
  if (m_sm->get_shd_warp(inst.warp_id()) != nullptr) {
    cmd.cta_id = m_sm->get_shd_warp(inst.warp_id())->get_cta_id();
  }
  cmd.opcode_family = TMAOpcodeFamily::UNKNOWN;
  cmd.direction = classify_tma_direction(inst);
  cmd.transfer_type = classify_tma_transfer_type(inst, cmd.opcode_family);
  cmd.operand_form = classify_tma_operand_form(cmd.opcode_family);
  cmd.meta_source = TMAMetadataSource::NONE;
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
