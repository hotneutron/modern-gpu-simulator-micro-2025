// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cassert>
#include <cstdio>
#include <memory>

#include "subcore.h"
#include "functional_unit.h"
#include "sm.h"


#include "../../../../trace-driven/trace_driven.h"
#include "../../../libcuda/gpgpu_context.h"
#include "../gpu-sim.h"
#include "../shader_trace.h"
#include "../stat-tool.h"
#include "first_level_instruction_cache.h"
#include "ldst_unit_sm.h"
#include "register_file.h"

#include "../../../../../util/traces_enhanced/src/traced_instruction.h"



Subcore::Subcore(unsigned subcore_id, const shader_core_config *config,
                 shader_core_stats *stats, SM *sm,
                 register_set_uniptr *EX_DP_shared_sm_reception_latch,
                 register_set_uniptr *EX_MEM_shared_sm_reception_latch,
                 register_set_uniptr *EX_TMA_shared_sm_reception_latch) :
                        m_regular_fixed_latency_rf_write_queue(config->max_size_register_file_write_queue_for_fixed_latency_instructions, "regular_fixed_latency_rf_write_queue"),
                        m_uniform_fixed_latency_rf_write_queue(config->max_size_register_file_write_queue_for_fixed_latency_instructions, "uniform_fixed_latency_rf_write_queue") {
  m_subcore_id = subcore_id;
  m_config = config;
  m_stats = stats;
  m_sm = sm;
  m_num_warps_per_subcore =
      m_config->max_warps_per_shader / sm->get_num_subcores();
  m_EX_DP_shared_sm_reception_latch = EX_DP_shared_sm_reception_latch;
  m_EX_MEM_shared_sm_reception_latch = EX_MEM_shared_sm_reception_latch;
  m_EX_TMA_shared_sm_reception_latch = EX_TMA_shared_sm_reception_latch;
  m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp = 0;
  m_num_pending_cycles_with_issue_port_busy = 0;
  m_pipeline_read_stage_latency_reg.resize(MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST);
  for (unsigned i = 0; i < MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST; i++) {
    m_pipeline_read_stage_latency_reg[i] = std::make_unique<warp_inst_t>(config);
  }
  m_reserved_slots_regular_fixed_latency_rf_write_queue = 0;
  m_reserved_slots_uniform_fixed_latency_rf_write_queue = 0;
  m_num_active_warps_subcore = 0;
  m_is_next_stage_of_issue_busy = false;
}

Subcore::~Subcore() {
  if(!m_config->is_fp32_and_int_unified_pipeline) {
    delete m_int_pipeline;
  }
  delete m_sp_pipeline;
  delete m_uniform_pipeline;
  delete m_tensor_pipeline;
  delete m_branch_pipeline;
  delete m_sfu_pipeline;
  delete m_miscellaneous_with_queue_pipeline;
  delete m_miscellaneous_no_queue_pipeline;
  delete m_memory_unit_subcore;
  delete m_tma_pipeline;
  delete m_dp_pipeline;
  delete m_uniform_rf;
  delete m_regular_rf;
  delete m_L0I;
  delete m_L0C_cache;
  m_all_subcore_ex_pipelines.clear();
  m_pipeline_read_stage_latency_reg.clear();
}

Subcore* Subcore::getptr() { return this; }

void Subcore::cycle() {
  if(m_num_active_warps_subcore > 0) {
    writeback(m_sm);
    execute();
    read_rf(m_sm);
    allocate(m_sm);
    control_stage(m_sm);
    issue(m_sm);
    decode(m_sm);
    fetch(m_sm);
    m_greedy_pointer_fetch = m_greedy_pointer_issue;
  }
  // if(m_sm->get_sid() == 0 && m_subcore_id == 0 && m_sm->get_current_gpu_cycle() == 3837) {
  //   fflush(stdout);
  // }
  m_L0C_cache->cycle();
  m_L0I->cycle();
}

void Subcore::decrease_active_warp() {
  m_num_active_warps_subcore--;
  assert(m_num_active_warps_subcore >= 0);
}

void Subcore::increase_active_warp() {
  m_num_active_warps_subcore++;
}


void Subcore::num_cycles_to_stall(unsigned int num_cycles) {
  m_num_pending_cycles_with_issue_port_busy += num_cycles;
}

int Subcore::get_fixed_latency_result_queue_size() {
  return m_reserved_slots_regular_fixed_latency_rf_write_queue;
}

bool Subcore::is_subcore_with_problems_of_fordward_progress() const {
  return m_is_next_stage_of_issue_busy;
}


bool Subcore::has_regular_fixed_latency_rf_result_queue_space() {
  bool res = m_reserved_slots_regular_fixed_latency_rf_write_queue < m_config->max_size_register_file_write_queue_for_fixed_latency_instructions;
  return res;
}

bool Subcore::has_uniform_fixed_latency_rf_result_queue_space() {
  bool res = m_reserved_slots_uniform_fixed_latency_rf_write_queue < m_config->max_size_register_file_write_queue_for_fixed_latency_instructions;
  return res;
}

void Subcore::reserve_slot_regular_fixed_latency_rf_result_queue_space() {
  assert(has_regular_fixed_latency_rf_result_queue_space());
  m_reserved_slots_regular_fixed_latency_rf_write_queue++;
}

void Subcore::reserve_slot_uniform_fixed_latency_rf_result_queue_space() {
  assert(has_uniform_fixed_latency_rf_result_queue_space());
  m_reserved_slots_uniform_fixed_latency_rf_write_queue++;
}

void Subcore::free_slot_regular_fixed_latency_rf_result_queue_space() {
  assert(m_reserved_slots_regular_fixed_latency_rf_write_queue > 0);
  m_reserved_slots_regular_fixed_latency_rf_write_queue--;
}

void Subcore::free_slot_uniform_fixed_latency_rf_result_queue_space() {
  assert(m_reserved_slots_uniform_fixed_latency_rf_write_queue > 0);
  m_reserved_slots_uniform_fixed_latency_rf_write_queue--;
}

bool Subcore::writeback_latch_proccess(SM *shared_sm, register_set_uniptr &latch, bool is_from_shared_sm_structure) {
  warp_inst_t *ready_reg = latch.get_ready();
  bool is_retirement_allowed = true;
  unsigned int num_uses = 0;
  unsigned int num_encoded_dsts = 0;
  bool conflict_wb_with_sm_shared_unit = false;
  Register_file *dst_rf = nullptr;
  bool has_rf_modeled = true;
  if (ready_reg && !ready_reg->empty()) {
    if(ready_reg->get_extra_trace_instruction_info().has_destination_registers()) {    
      num_encoded_dsts = ready_reg->get_extra_trace_instruction_info().get_num_destination_registers();
      num_uses = get_number_of_uses_per_operand(ready_reg->get_extra_trace_instruction_info(), ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number(), 0, ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_type());
      TraceEnhancedOperandType dst_type = TraceEnhancedOperandType::NONE;
      dst_type = get_reg_type_eval(ready_reg->get_extra_trace_instruction_info().get_operand(0));
      
      if(dst_type == TraceEnhancedOperandType::REG)  {
        dst_rf = m_regular_rf;
      }else if(dst_type == TraceEnhancedOperandType::UREG) {
        dst_rf = m_uniform_rf;
      }else {
        has_rf_modeled = false;
        assert((dst_type == TraceEnhancedOperandType::UPRED) || (dst_type == TraceEnhancedOperandType::PRED) || (dst_type == TraceEnhancedOperandType::BREG));
      }
      if(has_rf_modeled) {
        num_uses = std::min(num_uses, dst_rf->get_num_banks() * dst_rf->get_num_write_ports_per_bank());
      }
          
      if((num_encoded_dsts>0) && has_rf_modeled) {
        if(ready_reg->m_has_wb_from_sm_struct_to_subcore) {
          unsigned int target_reg_id = ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number();
          target_reg_id = ready_reg->get_final_dst_reg(target_reg_id);
          unsigned int bank_id = dst_rf->calculate_target_bank(target_reg_id);
          bool can_write = dst_rf->is_rf_bank_write_port_available_this_cycle(bank_id);
          bool reserve_wb = ready_reg->sm_shared_wb_consumed(can_write, m_config->num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore, conflict_wb_with_sm_shared_unit);
          if(reserve_wb) {
            dst_rf->allocate_rf_bank_write_port_this_cycle(bank_id);
          }
          is_retirement_allowed = ready_reg->has_sm_shared_wb_finished();
        }else if(has_rf_modeled) {
          for(unsigned j = 0; (j < num_uses) && is_retirement_allowed; j++) {
            unsigned int current_reg_id = ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number() + j;
            unsigned int bank_id = dst_rf->calculate_target_bank(current_reg_id);
            is_retirement_allowed = dst_rf->is_rf_bank_write_port_available_this_cycle(bank_id);
          }
        }
      }
    }
    
    if(is_retirement_allowed) {
      std::unique_ptr<warp_inst_t> &inst_to_retire = latch.get_ready_smartptr();
      assert(inst_to_retire.get() == ready_reg);
      for(unsigned j = 0; (j < num_uses) && !ready_reg->m_has_wb_from_sm_struct_to_subcore && has_rf_modeled && (num_encoded_dsts>0); j++) {
        unsigned int current_reg_id = ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number() + j;
        unsigned int bank_id = m_regular_rf->calculate_target_bank(current_reg_id);
        dst_rf->allocate_rf_bank_write_port_this_cycle(bank_id);
      }
      shared_sm->instruction_retirement(inst_to_retire.get());
    }else {
      if(!ready_reg->m_has_wb_from_sm_struct_to_subcore || (ready_reg->m_has_wb_from_sm_struct_to_subcore && conflict_wb_with_sm_shared_unit)) {
        shared_sm->m_sm_stats.m_stats_map["total_num_times_wb_port_conflict"]->increment_with_integer(1);
      }
    }

    shared_sm->m_sm_stats.m_stats_map["total_num_times_wb_evaluated"]->increment_with_integer(1);
  }

  return is_retirement_allowed;
}

void Subcore::writeback_process_fixed_latency_write_queue(register_set_uniptr &latch, SM *shared_sm, unsigned int max_num_pops, TraceEnhancedOperandType dst_result_queue_type) {
  for(unsigned int i = 0; (i < max_num_pops) && latch.has_ready(); i++) {
    bool retired = writeback_latch_proccess(shared_sm, latch, false);
    if(retired) {
      if(dst_result_queue_type == TraceEnhancedOperandType::UREG) {
        free_slot_uniform_fixed_latency_rf_result_queue_space();
      }else {
        free_slot_regular_fixed_latency_rf_result_queue_space();
      }
    }
  }
}

void Subcore::writeback(SM *shared_sm) {
  writeback_process_fixed_latency_write_queue(m_regular_fixed_latency_rf_write_queue, shared_sm, m_config->max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions, TraceEnhancedOperandType::REG);
  writeback_process_fixed_latency_write_queue(m_uniform_fixed_latency_rf_write_queue, shared_sm, m_config->max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions, TraceEnhancedOperandType::UREG);
  writeback_latch_proccess(shared_sm, m_EX_WB_sm_variable_latency_latch, false);
  writeback_latch_proccess(shared_sm, m_EX_WB_sm_shared_units_latch, true);
}

void Subcore::execute() {
  for (auto fu : m_all_subcore_ex_pipelines) {
    fu->cycle();
  }
}

void Subcore::read_rf(SM *shared_sm) {
  if(!m_pipeline_read_stage_latency_reg[0]->empty()) {
    functional_unit* fu = m_pipeline_read_stage_latency_reg[0]->get_fu_assigned();
    assert(fu);
    assert(m_read_stage_aux_latch.has_free());
    fu->release_read_barrier(m_pipeline_read_stage_latency_reg[0]);
    m_read_stage_aux_latch.move_in(m_pipeline_read_stage_latency_reg[0]);
    fu->issue(m_read_stage_aux_latch);
  }
  for (unsigned stage = 0; (stage + 1) < MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST; stage++) {
    if (m_pipeline_read_stage_latency_reg[stage]->empty()) {
      move_warp_uniptr(m_pipeline_read_stage_latency_reg[stage], m_pipeline_read_stage_latency_reg[stage + 1]);
    }
  }
  m_regular_rf->cycle();
  m_uniform_rf->cycle();
}

void Subcore::allocate(SM *shared_sm) {
  if(m_CONTROL_ALLOCATE_latch.has_ready()) {
    warp_inst_t *current_ins = m_CONTROL_ALLOCATE_latch.get_ready();
    functional_unit* fu = current_ins->get_fu_assigned();
    RF_requests rf_requests;
    assert(fu->is_fixed_latency_unit());
    unsigned int latency_read_fixed_latency_inst = current_ins->is_tensor_core_op_with_4_registers_per_op() ? MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST : NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST;
    if(m_pipeline_read_stage_latency_reg[latency_read_fixed_latency_inst - 1]->empty()) {
      unsigned int sm_warp_id = current_ins->warp_id();
      rf_requests.m_regular = m_regular_rf->is_possible_to_read_cacheable(current_ins, sm_warp_id, fu->get_rf_num_read_cycles());
      rf_requests.m_uniform = m_uniform_rf->is_possible_to_read_cacheable(current_ins, sm_warp_id, m_config->warp_size);
      bool is_read_available = rf_requests.is_possible_to_read();
      unsigned int target_latency_execution = latency_read_fixed_latency_inst + current_ins->latency  + current_ins->initiation_interval;
      // #region debug-point bitset-latency-overflow
      if (target_latency_execution >= 512) {
        std::fprintf(stderr,
                     "[BITSETDBG][Subcore] overflow sid=%u subcore=%u warp=%u "
                     "ufid=%u pc=0x%llx op=%u op_pipe=%u tma_family=%u "
                     "lat_read=%u latency=%u initiation=%u target=%u\n",
                     shared_sm->get_sid(), m_subcore_id, current_ins->warp_id(),
                     current_ins->unique_function_id,
                     static_cast<unsigned long long>(current_ins->pc),
                     static_cast<unsigned>(current_ins->op),
                     static_cast<unsigned>(current_ins->op_pipe),
                     static_cast<unsigned>(current_ins->tma_opcode_family),
                     latency_read_fixed_latency_inst, current_ins->latency,
                     current_ins->initiation_interval, target_latency_execution);
        assert(target_latency_execution < 512 &&
               "bitset latency overflow before functional_unit::is_latency_available");
      }
      // #endregion debug-point bitset-latency-overflow
      bool is_fu_latency_available = fu->is_latency_available(target_latency_execution);
      bool is_rf_ready = is_read_available;
      shared_sm->m_sm_stats.m_stats_map["total_num_evals_rf"]->increment_with_integer(1);
      if(is_rf_ready && is_fu_latency_available) {
        allocate_reads(rf_requests, current_ins, sm_warp_id, fu->get_rf_num_read_cycles());
        shared_sm->m_sm_stats.m_stats_map["total_num_register_file_cache_hits"]->increment_with_integer(rf_requests.m_regular.m_rf_cache_read_requests.size());
        shared_sm->m_sm_stats.m_stats_map["total_num_register_file_cache_allocations"]->increment_with_integer(rf_requests.m_regular.m_rf_cache_allocate_requests.size());
        fu->reserve_latency(target_latency_execution);
        m_CONTROL_ALLOCATE_latch.move_out_to(m_pipeline_read_stage_latency_reg[latency_read_fixed_latency_inst - 1]);
      }else {
        fu->add_extra_cycle_initiation_interval();
        shared_sm->m_sm_stats.m_stats_map["total_num_evals_rf_with_conflict"]->increment_with_integer(1);
        // [WGMMA Opt6 Step-0] (VII) tensor-only: RF/latency conflict extended the tensor
        // re-issue lockout beyond the static initiation_interval.
        if(m_config->wgmma_step0_instrument_enable && current_ins->op == TENSOR_CORE_OP) {
          shared_sm->m_sm_stats.m_stats_map["total_num_tensor_add_extra_cycle_initiation_interval"]->increment_with_integer(1);
        }
        // [Head-of-line lever] CONTROL_ALLOCATE cannot drain because RF read and/or the FU latency
        // bitset is unavailable — the root cause behind control_allocate_full one stage up.
        if(m_config->headofline_instrument_enable) {
          if(!is_rf_ready)
            shared_sm->m_sm_stats.m_stats_map["total_num_hol_reason_rf_conflict"]->increment_with_integer(1);
          if(!is_fu_latency_available)
            shared_sm->m_sm_stats.m_stats_map["total_num_hol_reason_fu_latency_full"]->increment_with_integer(1);
        }
      }
    }else {
      fu->add_extra_cycle_initiation_interval();
      shared_sm->m_sm_stats.m_stats_map["total_num_evals_rf_with_conflict"]->increment_with_integer(1);
      // [WGMMA Opt6 Step-0] (VII) tensor-only lockout extension.
      if(m_config->wgmma_step0_instrument_enable && current_ins->op == TENSOR_CORE_OP) {
        shared_sm->m_sm_stats.m_stats_map["total_num_tensor_add_extra_cycle_initiation_interval"]->increment_with_integer(1);
      }
      // [Head-of-line lever] CONTROL_ALLOCATE cannot drain because the read_stage pipeline reg for this
      // (WGMMA=6-deep / other=3-deep) latency class is still occupied.
      if(m_config->headofline_instrument_enable) {
        shared_sm->m_sm_stats.m_stats_map["total_num_hol_reason_read_stage_full"]->increment_with_integer(1);
      }
    }
  }
}

void Subcore::control_stage(SM *shared_sm) {
  if(m_ISSUE_CONTROL_latch.has_ready()) {
    warp_inst_t *current_ins = m_ISSUE_CONTROL_latch.get_ready();
    functional_unit* fu = current_ins->get_fu_assigned();
    bool is_fixed_latency_inst = fu->is_fixed_latency_unit();
    if(!current_ins->m_has_perform_control_stage) {
      if (m_sm->get_config()->is_trace_mode && !((!m_sm->get_shd_warp(current_ins->warp_id())->get_kernel_info()->is_captured_from_binary) || m_sm->get_config()->is_remodeling_scoreboarding_enabled)) {
        if (current_ins->get_extra_trace_instruction_info().get_control_bits().get_is_new_read_barrier()) {
          m_sm->add_pending_wait_barrier_increment(current_ins, READ_WAIT_BARRIER, current_ins->get_extra_trace_instruction_info().get_control_bits().get_id_new_read_barrier());
        }
        if (current_ins->get_extra_trace_instruction_info().get_control_bits().get_is_new_write_barrier()) {
          m_sm->add_pending_wait_barrier_increment(current_ins, WRITE_WAIT_BARRIER, current_ins->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier());
        }
      }
      current_ins->m_has_perform_control_stage = true;
    }
    if(m_CONTROL_ALLOCATE_latch.has_free() || !is_fixed_latency_inst) {
      if(is_fixed_latency_inst) {
        assert(m_CONTROL_ALLOCATE_latch.has_free());
        move_warp_between_reg_sets(m_CONTROL_ALLOCATE_latch, 0, m_ISSUE_CONTROL_latch, 0);
      }else {
        // If it is not a fixed_latency_instruction, it is direclty issued to its functional unit
        if(fu->can_issue(current_ins)) {
          fu->issue(m_ISSUE_CONTROL_latch);
        } else if(m_config->headofline_instrument_enable) {
          // [Head-of-line lever] non-fixed op stuck: target FU cannot accept it (queue full) -> the
          // ISSUE_CONTROL latch stays occupied and blocks the whole subcore next issue.
          shared_sm->m_sm_stats.m_stats_map["total_num_hol_reason_fu_cannot_issue"]->increment_with_integer(1);
        }
      }
    }else {
      if(is_fixed_latency_inst) {
        fu->add_extra_cycle_initiation_interval();
        // [WGMMA Opt6 Step-0] (VII) tensor-only lockout extension (CONTROL->ALLOCATE latch full).
        if(m_config->wgmma_step0_instrument_enable && current_ins->op == TENSOR_CORE_OP) {
          shared_sm->m_sm_stats.m_stats_map["total_num_tensor_add_extra_cycle_initiation_interval"]->increment_with_integer(1);
        }
        // [Head-of-line lever] fixed-latency op stuck: CONTROL_ALLOCATE latch is full -> ISSUE_CONTROL
        // cannot drain -> whole-subcore head-of-line stall.
        if(m_config->headofline_instrument_enable) {
          shared_sm->m_sm_stats.m_stats_map["total_num_hol_reason_control_allocate_full"]->increment_with_integer(1);
        }
      }
    }
  }
}

// [Head-of-line lever] Read-only re-scan on a next_stage_not_available cycle (ISSUE_CONTROL latch
// full, so Subcore::issue() skipped its warp-scan). Counts warps that COULD have issued if the latch
// were free — i.e. valid head + warp-side conditions satisfied (FU-side excluded, since a full latch
// blocks FU entry regardless). Also classifies the FU of the instruction holding the latch. Pure read,
// no side effects (never calls the side-effecting waiting()/warp_waiting_at_tma_flush()); gated by
// -headofline_instrument_enable so default runs stay bit-identical. See .plan/CONSUMER_COMPUTE_BOUND.md.
void Subcore::scan_head_of_line_when_blocked(SM *shared_sm) {
  // Classify what is holding the head of line (the instruction occupying ISSUE_CONTROL).
  if (m_ISSUE_CONTROL_latch.has_ready()) {
    warp_inst_t *held = m_ISSUE_CONTROL_latch.get_ready();
    const char *key;
    if (held->op == TENSOR_CORE_OP)                 key = "total_num_next_stage_blocked_by_tensor";
    else if (held->is_load() || held->is_store())   key = "total_num_next_stage_blocked_by_mem";
    else                                            key = "total_num_next_stage_blocked_by_other";
    shared_sm->m_sm_stats.m_stats_map[key]->increment_with_integer(1);
    // finer split of "other" so the fix knows which pipe's read_stage/FU-latency to deepen.
    if (held->op != TENSOR_CORE_OP && !held->is_load() && !held->is_store()) {
      const char *okey;
      switch (held->op) {
        case SFU_OP: okey = "total_num_next_stage_blocked_by_sfu"; break;
        case SP_OP: case INTP_OP: case DP_OP: case HALF_OP: case UNIFORM_OP:
                     okey = "total_num_next_stage_blocked_by_sp_int_dp"; break;
        default:     okey = "total_num_next_stage_blocked_by_branch_other"; break;
      }
      shared_sm->m_sm_stats.m_stats_map[okey]->increment_with_integer(1);
    }
  }

  unsigned long long n_ready = 0;
  for (auto *c_warp : m_warps_of_subcore) {
    if (c_warp == NULL || c_warp->done_exit()) continue;
    if (!c_warp->get_IBuffer_remodeled()->is_next_valid()) continue;  // no valid head
    unsigned int sm_warp_id = c_warp->get_warp_id();
    unsigned int subcore_warp_id =
        translate_warp_id_of_sm_to_subcore(sm_warp_id, shared_sm->get_num_subcores());
    warp_inst_t *pI = c_warp->get_IBuffer_remodeled()->next_inst();
    if (pI == nullptr) continue;
    shared_sm->m_sm_stats.m_stats_map["total_num_next_stage_valid_head_warps"]->increment_with_integer(1);

    bool use_traditional_scoreboarding =
        !c_warp->get_kernel_info()->is_captured_from_binary ||
        m_config->is_remodeling_scoreboarding_enabled || !m_config->is_trace_mode;
    bool are_traditional_scoreaboards_ready = true;
    bool is_stall_counter_0 = true;
    bool are_wait_barriers_ready = true;
    bool is_not_yield = true;
    if (use_traditional_scoreboarding) {
      are_traditional_scoreaboards_ready =
          !(shared_sm->get_scoreboard()->checkCollision_remodeling(sm_warp_id, pI) ||
            shared_sm->get_scoreboard_WAR()->checkCollision_remodeling(sm_warp_id, pI));
    } else {
      is_stall_counter_0 = c_warp->get_dependency_state()->is_stall_counter_0();
      are_wait_barriers_ready = is_wait_barriers_ready_entry_point(pI, subcore_warp_id);
      is_not_yield = c_warp->get_dependency_state()->is_yield_ready();
    }
    bool is_not_warp_waiting_ldgdepbar = !is_waiting_ldgdepbar(pI, subcore_warp_id);
    // Warp-side ready = every issue condition EXCEPT the FU-side ones (fu_available, result-queue,
    // l1c) and the side-effecting programmer-barrier/tma_flush (skipped like the read-only tail).
    bool warp_side_ready = are_traditional_scoreaboards_ready && is_stall_counter_0 &&
                           are_wait_barriers_ready && is_not_yield && is_not_warp_waiting_ldgdepbar;
    if (warp_side_ready) {
      n_ready++;
    } else {
      // [Head-of-line lever] why this warp was NOT ready (non-exclusive: a warp can miss several).
      // If `with_ready_warp` is low, the dominant reason here tells us whether the whole warpgroup is
      // stuck lockstep on the same dependency (over-serialization) vs. genuinely diverse waits.
      if (!are_wait_barriers_ready)
        shared_sm->m_sm_stats.m_stats_map["total_num_next_stage_notready_wait_barrier"]->increment_with_integer(1);
      if (!are_traditional_scoreaboards_ready)
        shared_sm->m_sm_stats.m_stats_map["total_num_next_stage_notready_scoreboard"]->increment_with_integer(1);
      if (!is_stall_counter_0)
        shared_sm->m_sm_stats.m_stats_map["total_num_next_stage_notready_stall_count"]->increment_with_integer(1);
      if (!is_not_yield)
        shared_sm->m_sm_stats.m_stats_map["total_num_next_stage_notready_yield"]->increment_with_integer(1);
      if (!is_not_warp_waiting_ldgdepbar)
        shared_sm->m_sm_stats.m_stats_map["total_num_next_stage_notready_ldgdepbar"]->increment_with_integer(1);
    }
  }

  shared_sm->m_sm_stats.m_stats_map["total_num_cycles_next_stage_scanned"]->increment_with_integer(1);
  if (n_ready > 0) {
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_next_stage_with_ready_warp"]->increment_with_integer(1);
    shared_sm->m_sm_stats.m_stats_map["total_num_ready_warps_during_next_stage"]->increment_with_integer(n_ready);
  }
}

void Subcore::issue(SM *shared_sm) {
  bool is_valid_inst =
      false;  // there was one warp with a valid instruction to issue
  bool is_issued_inst = false;  // Achieved to issue an instruction?
  // [NANOSLEEP spin lever] set true iff the issued winner this cycle was a producer mbarrier spin
  // poll (PHASECHK/TRYWAIT). Finalized against n_eligible_this_cycle in the stats block. Observe-only.
  bool issued_spin_op_this_cycle = false;
  bool is_issue_port_busy = true;
  bool is_next_stage_availabe = true;
  bool is_any_invalid_head_decode_pending = false;
  bool is_any_invalid_head_l0i_response_ready = false;
  bool is_any_invalid_head_waiting_frontend = false;
  bool is_any_invalid_head_waiting_frontend_miss = false;
  bool is_any_invalid_head_waiting_frontend_in_l0i_response_queue = false;
  bool is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait = false;
  bool is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_not_allocated = false;
  bool is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_issued_not_ready = false;
  bool is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_ready_not_promoted = false;
  bool is_any_invalid_head_waiting_frontend_in_l0i_response_queue_other = false;
  bool is_any_invalid_head_waiting_frontend_hit_status = false;
  bool is_any_invalid_head_ibuffer_empty = false;
  bool is_any_invalid_head_unknown = false;
  // [Step-0] sub-cause of ibuffer-empty: is the head PC's fetch already in flight in L0I
  // (awaiting a response) or has no fetch been issued yet (fetch scheduling behind)?
  bool is_any_invalid_head_ibuffer_empty_fetch_inflight = false;
  bool is_any_invalid_head_ibuffer_empty_fetch_not_issued = false;
  // Per-reason "no_warps_ready" stall attribution. Each flag becomes true if, in
  // this cycle, at least one warp with a valid head instruction was blocked for
  // that reason. Reasons are grouped so they can be compared against NCU's
  // smsp__average_warps_issue_stalled_<reason> decomposition:
  //   TMA axis     : wait_barrier (mbarrier) + tma_flush + inst_barrier(ldgdepbar/named)
  //   non-TMA axis : fu_occupied (gmma/math pipe) + stall_count (wait) + l1c (short scoreboard)
  //                  + scoreboard (traditional RAW/WAR) + yield
  bool is_any_waiting_in_inst_barrier = false;   // named barrier / ldgdepbar (TMA)
  bool is_any_waiting_in_stall_count = false;    // fixed-latency dep -> NCU "wait" (non-TMA)
  bool is_any_waiting_in_wait_barrier = false;   // mbarrier -> NCU "barrier/long_scoreboard" (TMA)
  bool is_any_waiting_in_tma_flush = false;      // cp.async.bulk.wait_group drain (TMA)
  bool is_any_waiting_in_yield = false;          // YIELD control bit (non-TMA)
  bool is_any_waiting_in_fu_occupied = false;    // FU busy -> NCU "gmma/math_pipe_throttle" (non-TMA)
  // [WGMMA Opt6 Step-0] per-cycle flags (observe-only).
  bool is_any_fu_occupied_tensor = false;        // (I) blocked warp's head op is TENSOR_CORE_OP
  bool is_any_fu_occupied_sfu = false;           // (I) SFU_OP
  bool is_any_fu_occupied_sp_int_dp = false;     // (I) SP/INT/DP/UNIFORM
  bool is_any_fu_occupied_other = false;         // (I) any other fixed-latency pipe
  bool is_any_tensor_reissue_lockout_only = false; // (III) tensor head blocked ONLY by tensor fu
  bool is_any_tensor_fu_occupied_and_wait_barrier = false; // (VI) tensor fu busy AND wait_barrier not-ready on same warp
  bool is_any_waiting_in_scoreboard = false;     // traditional scoreboard collision (non-TMA)
  bool is_any_waiting_in_result_queue_full = false; // RF result-queue full (non-TMA)
  bool is_any_waiting_l1c = false;               // const cache -> NCU "short_scoreboard" (non-TMA)
  // [NCU stall-taxonomy] warpgroup-arrive wait (WGMMA WARPGROUP.ARRIVE / .DEPBAR) split out of
  // the wait_barrier bucket to match NCU `warpgroup_arrive`. See .plan/NCU_STALL_TAXONOMY_METRICS_IMPL.md.
  bool is_any_waiting_in_warpgroup_arrive = false;
  // [NCU stall-taxonomy] dispatch/issue-port + RF-result-queue backpressure (NCU `dispatch_stall`),
  // re-derived into the per-warp axis (a warp was otherwise-ready but its result queue was full).
  bool is_any_waiting_in_dispatch = false;
  // [NCU stall-taxonomy] not_selected + eligible accounting (Phase 2): once the loop would have
  // broken, keep iterating in read-only mode over the post-stop tail; count eligible-but-not-picked.
  bool tail_readonly = false;
  unsigned long long n_not_selected_this_cycle = 0;
  unsigned long long n_eligible_this_cycle = 0;
  // [FWD drain-idle 축1] mutually-exclusive sole-block accounting for this subcore this cycle.
  // We scan valid-head warps that failed to issue; for each we count how many issue conditions are
  // unmet. If exactly one is unmet it yields an ONLY_* reason, else SB_MULTI. Across warps we keep
  // the highest recover-value reason (WAIT_BARRIER>TENSOR>STALL_COUNT>FU_NONTENSOR>NEXT_STAGE>L1C>
  // OTHER>MULTI). If NO warp had a valid head at all -> SB_DRAINED (floor). Gated by wgmma step0.
  bool any_valid_head_this_cycle = false;
  int best_sole_block_rank = -1;         // higher rank wins; -1 = none seen
  Step0SoleBlock best_sole_block = SB_MULTI;

  // [intra-SMSP warp-switch] set true iff, this cycle, at least one warp's head targeted a busy
  // no-queue FU (SFU) and was filtered at issue time by the new gate (can_issue=false). Combined
  // with is_issued_inst in the stats block to record: fix fired / another warp then issued (the
  // recovered slot) / still nobody issued. Only meaningful when -intra_smsp_warpswitch_enable is on.
  bool intra_ws_sfu_filtered_this_cycle = false;

  modify_warp_state();
  if(m_num_pending_cycles_with_issue_port_busy > 0) {
    m_num_pending_cycles_with_issue_port_busy--;
  }else if(m_ISSUE_CONTROL_latch.has_free()) {
    is_issue_port_busy = false;
    is_next_stage_availabe = true;
    std::vector<unsigned int> priority_ordered_for_issue = order_greedy_then_highest_id(shared_sm, m_greedy_pointer_issue);
    for (auto c_warp_id : priority_ordered_for_issue) {
      shd_warp_t *c_warp = m_warps_of_subcore[c_warp_id];
      // Don't consider warps that are not yet valid
      if (c_warp == NULL || c_warp->done_exit()) {
        continue;
      }
      unsigned int sm_warp_id = c_warp->get_warp_id();
      unsigned int subcore_warp_id = translate_warp_id_of_sm_to_subcore(
          sm_warp_id, shared_sm->get_num_subcores());
      bool is_the_greedy_warp = (m_greedy_pointer_issue == subcore_warp_id);
      assert(c_warp_id == subcore_warp_id);
      bool is_valid_inst_in_the_warp =
          c_warp->get_IBuffer_remodeled()->is_next_valid();

      // [NCU stall-taxonomy] Phase 2 tail: once the loop has already stopped this cycle, a warp with
      // NO valid head instruction cannot be `not_selected` (it is a frontend/no_valid case), so skip
      // it without running the frontend-classification path (which today never executes after the
      // break, and would otherwise perturb the no_valid_instruction sub-counters). Valid-head tail
      // warps fall through to the read-only eligibility check inside the valid block below.
      if (tail_readonly && !is_valid_inst_in_the_warp) {
        continue;
      }

      if (!is_valid_inst_in_the_warp) {
        IBuffer_Remodeled *ibuffer = c_warp->get_IBuffer_remodeled();
        if (ibuffer->get_is_empty()) {
          is_any_invalid_head_ibuffer_empty = true;
          // [Step-0] classify ibuffer-empty: is a fetch for this warp's next PC already
          // in flight in L0I (awaiting response), or has no fetch been issued yet?
          // Guard: only read get_pc() when the warp still has a next PC to fetch — once the
          // trace is exhausted (used_insts == traced_pcs.size()) get_pc() asserts. A drained
          // warp with an empty ibuffer is just winding down; leave it unclassified.
          bool empty_warp_has_next_pc =
              m_config->is_trace_mode
                  ? !static_cast<trace_shd_warp_t *>(c_warp)->trace_done()
                  : true;
          if ((m_config->wgmma_step0_instrument_enable ||
               m_config->l1i_frontend_step0_instrument_enable) &&
              empty_warp_has_next_pc) {
            address_type empty_local_pc =
                m_config->is_trace_mode
                    ? static_cast<trace_shd_warp_t *>(c_warp)->get_pc()
                    : c_warp->get_pc();
            unsigned int empty_ufid =
                c_warp->get_current_unique_function_id_call();
            address_type empty_global_pc =
                shared_sm->from_local_pc_to_global_pc_address(empty_local_pc,
                                                              empty_ufid);
            cache_request_status empty_status = RESERVATION_FAIL;
            if (m_L0I->get_access_status_for_warp_pc(sm_warp_id, empty_global_pc,
                                                     empty_status)) {
              is_any_invalid_head_ibuffer_empty_fetch_inflight = true;
            } else {
              is_any_invalid_head_ibuffer_empty_fetch_not_issued = true;
            }
          }
          continue;
        }

        address_type local_pc_head = ibuffer->get_next_pc_to_issue();
        if (m_inst_fetch_decode_latch.m_valid &&
            (m_inst_fetch_decode_latch.m_warp_id == subcore_warp_id) &&
            (m_inst_fetch_decode_latch.m_pc == local_pc_head)) {
          is_any_invalid_head_decode_pending = true;
          continue;
        }

        unsigned int unique_function_id =
            c_warp->get_current_unique_function_id_call();
        address_type global_pc_head =
            shared_sm->from_local_pc_to_global_pc_address(local_pc_head,
                                                          unique_function_id);
        if (m_L0I->is_first_access_ready_for_warp_pc(sm_warp_id,
                                                     global_pc_head)) {
          is_any_invalid_head_l0i_response_ready = true;
          continue;
        }

        cache_request_status head_status = RESERVATION_FAIL;
        if (m_L0I->get_access_status_for_warp_pc(sm_warp_id, global_pc_head,
                                                 head_status)) {
          if (head_status == MISS) {
            is_any_invalid_head_waiting_frontend = true;
            is_any_invalid_head_waiting_frontend_miss = true;
          } else if (head_status == IN_L0I_RESPONSE_QUEUE) {
            is_any_invalid_head_waiting_frontend = true;
            is_any_invalid_head_waiting_frontend_in_l0i_response_queue = true;
            status_element::origin head_origin = status_element::NONE;
            if (m_L0I->get_access_origin_for_warp_pc(sm_warp_id, global_pc_head,
                                                     head_origin) &&
                (head_origin == status_element::STREAM_BUFFER_WAIT)) {
              is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait = true;
              bool is_prefetch_ready = false;
              if (m_L0I->classify_stream_buffer_wait_state(global_pc_head,
                                                           is_prefetch_ready)) {
                if (is_prefetch_ready) {
                  is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_ready_not_promoted = true;
                } else {
                  is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_issued_not_ready = true;
                }
              } else {
                is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_not_allocated = true;
              }
            } else {
              is_any_invalid_head_waiting_frontend_in_l0i_response_queue_other = true;
            }
          } else if (head_status == HIT) {
            is_any_invalid_head_waiting_frontend = true;
            is_any_invalid_head_waiting_frontend_hit_status = true;
          } else {
            is_any_invalid_head_unknown = true;
          }
        } else {
          is_any_invalid_head_unknown = true;
        }
        continue;
      }

      if (is_valid_inst_in_the_warp) {
        is_valid_inst = true;

        bool use_traditional_scoreboarding = !c_warp->get_kernel_info()->is_captured_from_binary || m_config->is_remodeling_scoreboarding_enabled || !m_config->is_trace_mode;

        warp_inst_t *pI = c_warp->get_IBuffer_remodeled()->next_inst();
        assert(pI != nullptr);

        bool are_traditional_scoreaboards_ready = true;
        bool is_stall_counter_0 = true;
        bool are_wait_barriers_ready = true;
        bool is_not_yield = true;
        
        if(use_traditional_scoreboarding) {
          are_traditional_scoreaboards_ready = !(shared_sm->get_scoreboard()->checkCollision_remodeling(sm_warp_id, pI) || shared_sm->get_scoreboard_WAR()->checkCollision_remodeling(sm_warp_id, pI));
        }else {
          is_stall_counter_0 =
            c_warp->get_dependency_state()->is_stall_counter_0();
          are_wait_barriers_ready =
            is_wait_barriers_ready_entry_point(pI, subcore_warp_id);
          is_not_yield = c_warp->get_dependency_state()->is_yield_ready(); 
        }

        bool is_not_warp_waiting_ldgdepbar = !is_waiting_ldgdepbar(pI, subcore_warp_id);
        // [NCU stall-taxonomy] On the read-only tail (already stopped this cycle) do NOT call the
        // side-effecting waiting()/warp_waiting_at_tma_flush(): they map to NCU barrier/membar/tma_flush,
        // NOT not_selected, and re-invoking them would perturb membar/log state (bit-identity).
        // See .plan/NCU_STALL_TAXONOMY_METRICS_IMPL.md Phase 2.
        bool is_not_warp_waiting_in_programmer_barrier =
            tail_readonly ? true : !c_warp->waiting();
        // UTMACMDFLUSH (cp.async.bulk.wait_group 0) stalls its warp until all of
        // that warp's outstanding store-class TMA transfers drain (warp-local).
        bool is_not_warp_waiting_tma_flush =
            tail_readonly ? true : !shared_sm->warp_waiting_at_tma_flush(sm_warp_id, pI);
        functional_unit* fu = get_fu(pI);
        bool is_fu_available = true;;
        bool is_fixed_latency_inst = fu->is_fixed_latency_unit();
        if(is_fixed_latency_inst) {
          is_fu_available = fu->can_issue(pI);
        } else if(m_config->intra_smsp_warpswitch_enable && !fu->get_has_queue()) {
          // [intra-SMSP warp-switch] A no-queue FU (SFU) holds its dispatch reg for a DETERMINISTIC
          // II (dispatch_delay counts initiation_interval down), so — like a fixed-latency unit — we
          // know at issue time whether it can accept an op. Checking can_issue() here filters a busy
          // SFU MUFU BEFORE it enters the 1-deep ISSUE_CONTROL latch, so the warp-scan continues and
          // GTO picks another ready warp instead of the whole subcore stalling. Queue-based FUs are
          // excluded (their drain is non-deterministic; deferred check is correct for them). II is
          // untouched. Gated; default off = baseline bit-identical. See .plan/CONSUMER_COMPUTE_BOUND.md.
          is_fu_available = fu->can_issue(pI);
          // record that the fix actually filtered a busy-SFU head this cycle (for the effect counters).
          if(!is_fu_available) intra_ws_sfu_filtered_this_cycle = true;
        }
        bool is_l1c_ready = tail_readonly ? true : are_l1c_operands_ready(shared_sm, pI);
        bool is_write_available_result_queue_for_fixed_latency_available = true;
        bool has_dst_regs = false;
        TraceEnhancedOperandType dst_type = TraceEnhancedOperandType::NONE;
        if(fu->is_fixed_latency_unit()) {
          if(pI->get_extra_trace_instruction_info().has_destination_registers()) {
            has_dst_regs = true;
            dst_type = fu->get_result_queue_type();
            if(dst_type == TraceEnhancedOperandType::UREG) {
              is_write_available_result_queue_for_fixed_latency_available = has_uniform_fixed_latency_rf_result_queue_space();           
            }else {
              is_write_available_result_queue_for_fixed_latency_available = has_regular_fixed_latency_rf_result_queue_space();
            }
          }
        }

        // Despite the name, this is the "this warp is eligible to issue now"
        // predicate, not a literal warp switch. If false, the warp is simply
        // skipped this cycle and the scheduler tries another warp; the warp
        // becomes issuable again once its condition clears (e.g. is_not_warp_
        // waiting_tma_flush goes true when its store-class transfers drain). So
        // a stalling UTMACMDFLUSH only parks its own warp, never the whole core.
        bool are_switch_warp_conditions_ready =
            is_not_yield && is_stall_counter_0 && are_wait_barriers_ready &&
            is_fu_available && is_not_warp_waiting_in_programmer_barrier &&
            is_not_warp_waiting_ldgdepbar && is_not_warp_waiting_tma_flush && are_traditional_scoreaboards_ready && is_write_available_result_queue_for_fixed_latency_available;

        bool can_l1c_switch_warp = true;

        if(m_greedy_pointer_issue == subcore_warp_id ) {
          if(is_l1c_ready) {
            m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp = m_config->num_const_cache_cycle_misses_before_switch_to_other_warp;
          }else if(m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp > 0) {
            m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp--;
          }
          if(m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp > 0) {
            can_l1c_switch_warp = false;
          }
        }

        bool is_inst_ready_to_issue = are_switch_warp_conditions_ready && is_l1c_ready;
        // [NCU stall-taxonomy] Phase 2: on the read-only tail (a warp strictly AFTER the cycle's
        // stop) never issue. Count it as not_selected iff it satisfies the read-only eligibility
        // (are_switch_warp_conditions_ready; l1c excluded by design — see plan). No side effects.
        if (tail_readonly) {
          if (are_switch_warp_conditions_ready) {
            ++n_eligible_this_cycle;
            ++n_not_selected_this_cycle;
          }
          continue;
        }
        // [FWD drain-idle 축1] observe-only sole-block classification for this valid-head warp.
        // Reached only for warps WITH a valid head (invalid heads `continue` above), so mark that
        // this subcore had >=1 valid head this cycle (=> NOT drained). Count unmet issue conditions;
        // exactly-one => the matching ONLY_* reason, >=2 => SB_MULTI. Keep the highest recover-value
        // reason across warps. Pure read (no side effects); gated by wgmma step0.
        if (m_config->wgmma_step0_instrument_enable) {
          any_valid_head_this_cycle = true;
          if (!is_inst_ready_to_issue) {
            bool head_is_tensor = (pI->op == TENSOR_CORE_OP);
            // condition -> (met?, sole-reason-if-this-is-the-only-unmet, rank)
            struct { bool met; Step0SoleBlock reason; int rank; } conds[] = {
              { are_wait_barriers_ready,
                SB_ONLY_WAIT_BARRIER, 7 },
              { !( !is_fu_available && head_is_tensor ),
                SB_ONLY_TENSOR, 6 },
              { is_stall_counter_0,
                SB_ONLY_STALL_COUNT, 5 },
              { !( !is_fu_available && !head_is_tensor ),
                SB_ONLY_FU_NONTENSOR, 4 },
              { is_write_available_result_queue_for_fixed_latency_available,
                SB_ONLY_NEXT_STAGE, 3 },
              { is_l1c_ready,
                SB_ONLY_L1C, 2 },
              { is_not_yield && is_not_warp_waiting_in_programmer_barrier &&
                is_not_warp_waiting_ldgdepbar && is_not_warp_waiting_tma_flush &&
                are_traditional_scoreaboards_ready,
                SB_ONLY_OTHER, 1 },
            };
            int unmet = 0; Step0SoleBlock sole = SB_MULTI; int sole_rank = 0;
            for (auto &c : conds) {
              if (!c.met) { unmet++; sole = c.reason; sole_rank = c.rank; }
            }
            Step0SoleBlock warp_reason; int warp_rank;
            if (unmet == 1) { warp_reason = sole; warp_rank = sole_rank; }
            else            { warp_reason = SB_MULTI; warp_rank = 0; }
            if (warp_rank > best_sole_block_rank) {
              best_sole_block_rank = warp_rank;
              best_sole_block = warp_reason;
            }
          }
        }
        if (is_inst_ready_to_issue) {
          // This warp is eligible AND selected (the winner). Counts toward eligible, not not_selected.
          ++n_eligible_this_cycle;
          const active_mask_t &active_mask =
              shared_sm->get_active_mask(sm_warp_id, pI);
          assert(c_warp->inst_in_pipeline());
          set_num_pending_cycles_with_issue_port_busy(pI);
          if(m_config->is_interwarp_coalescing_enabled && ((m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC) ||
              (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID)))  {
            remove_interwarp_coalescing_dep_counter_at_decode_tracking(pI, sm_warp_id);
          }
          issue_warp(shared_sm, m_ISSUE_CONTROL_latch, pI, active_mask, sm_warp_id, fu, is_fixed_latency_inst, use_traditional_scoreboarding, has_dst_regs, dst_type);
          is_issued_inst = true;
          // [NANOSLEEP spin lever] record whether the winning op is a producer mbarrier spin poll
          // (PHASECHK/TRYWAIT). Used with n_eligible_this_cycle (finalized on the tail) to detect
          // spin displacing a co-eligible warp. Observe-only; gated at the stats site.
          issued_spin_op_this_cycle =
              (pI->sync_kind == SyncInstructionKind::PHASECHK ||
               pI->sync_kind == SyncInstructionKind::TRYWAIT);
          m_greedy_pointer_issue = subcore_warp_id;
          m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp = m_config->num_const_cache_cycle_misses_before_switch_to_other_warp;
          // [NCU stall-taxonomy] was: break; — now keep iterating the SAME loop in read-only mode
          // over the post-winner tail to count not_selected. No further issue_warp / side effects.
          tail_readonly = true;
          continue;
        }else {
          if(!are_switch_warp_conditions_ready) {
            if(!is_fu_available) {
              is_any_waiting_in_fu_occupied = true;
              // [WGMMA Opt6 Step-0] (I) classify which pipe the blocked head targets.
              if(m_config->wgmma_step0_instrument_enable) {
              switch(pI->op) {
                case TENSOR_CORE_OP: is_any_fu_occupied_tensor = true; break;
                case SFU_OP:         is_any_fu_occupied_sfu = true; break;
                case SP_OP: case HALF_OP: case INTP_OP: case DP_OP: case UNIFORM_OP:
                                     is_any_fu_occupied_sp_int_dp = true; break;
                default:             is_any_fu_occupied_other = true; break;
              }
              if(pI->op == TENSOR_CORE_OP) {
                // (III) tensor head blocked ONLY by the tensor fu: every other issue
                // condition for THIS warp is satisfied (so the warp would issue if the
                // tensor re-issue lockout were gone). This is the recoverable-cycle ceiling.
                bool blocked_only_by_fu =
                    is_not_yield && is_stall_counter_0 && are_wait_barriers_ready &&
                    is_not_warp_waiting_in_programmer_barrier &&
                    is_not_warp_waiting_ldgdepbar && is_not_warp_waiting_tma_flush &&
                    are_traditional_scoreaboards_ready &&
                    is_write_available_result_queue_for_fixed_latency_available;
                if(blocked_only_by_fu) {
                  is_any_tensor_reissue_lockout_only = true;
                }
                // (VI) tensor fu busy AND this same warp would next block on a wait_barrier
                // (WGMMA.WAIT / DEPBAR). High value => lowering II only shifts the stall.
                if(!are_wait_barriers_ready) {
                  is_any_tensor_fu_occupied_and_wait_barrier = true;
                }
              }
              } // m_config->wgmma_step0_instrument_enable
            }
            if(!is_not_warp_waiting_in_programmer_barrier || !is_not_warp_waiting_ldgdepbar) {
              is_any_waiting_in_inst_barrier = true;
            }
            if(!is_not_warp_waiting_tma_flush) {
              is_any_waiting_in_tma_flush = true;
            }
            if(!is_not_yield) {
              is_any_waiting_in_yield = true;
            }
            if(!is_stall_counter_0) {
              is_any_waiting_in_stall_count = true;
            }
            if(!are_wait_barriers_ready) {
              is_any_waiting_in_wait_barrier = true;
            }
            if(!are_traditional_scoreaboards_ready) {
              is_any_waiting_in_scoreboard = true;
              // [NCU stall-taxonomy] split NCU `warpgroup_arrive` out of scoreboard stalls: the warp
              // is RAW-blocked on a register that is a pending dst of an in-flight tensor (WGMMA) op.
              // This is where a consumer waiting on a warpgroup-MMA result actually stalls (WGMMA is a
              // tensor-FU op, so it surfaces as a scoreboard RAW, NOT an mbarrier wait). Read-only.
              if(shared_sm->get_scoreboard()->checkTensorCollision_remodeling(sm_warp_id, pI)) {
                is_any_waiting_in_warpgroup_arrive = true;
              }
            }
            if(!is_write_available_result_queue_for_fixed_latency_available) {
              is_any_waiting_in_result_queue_full = true;
              // [NCU stall-taxonomy] result-queue backpressure at the dispatch/RF-write port maps to
              // NCU `dispatch_stall` (re-derived into the per-warp axis).
              is_any_waiting_in_dispatch = true;
            }
            if(!is_l1c_ready) {
              is_any_waiting_l1c = true;
            }
          }else {
            if(!is_the_greedy_warp || (can_l1c_switch_warp)) {
              if(!is_l1c_ready) {
                is_any_waiting_l1c = true;
              }
            }else {
              // [NCU stall-taxonomy] was: break; — the greedy warp is blocked only on the const cache
              // and cannot yield. It is the stop point (l1c wait = NCU short_scoreboard/imc_miss, NOT
              // not_selected). Keep iterating read-only over the tail instead of breaking.
              tail_readonly = true;
              continue;
            }
          }
        
        }
      }
    }
  }else {
    is_next_stage_availabe = false;
  }

  // Stats
  // [NCU stall-taxonomy] Phase 2 accumulators — emitted every cycle regardless of the branch below.
  // n_eligible = warps that satisfied read-only eligibility this cycle (winner + tail); n_not_selected
  // = eligible tail warps that were not the winner. See .plan/NCU_STALL_TAXONOMY_METRICS_IMPL.md.
  shared_sm->m_sm_stats.m_stats_map["total_num_warps_eligible_accumulator"]->increment_with_integer(n_eligible_this_cycle);
  shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_not_selected"]->increment_with_integer(n_not_selected_this_cycle);
  // [NANOSLEEP spin lever] observe-only: count cycles where the winner was a producer mbarrier spin
  // poll, and (of those) cycles where >=1 other warp was eligible but not selected (n_not_selected>0)
  // = spin displaced a co-eligible warp. Gated so default runs stay bit-identical.
  if (m_config->spin_instrument_enable && is_issued_inst && issued_spin_op_this_cycle) {
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_spin_ops_issued"]->increment_with_integer(1);
    shared_sm->m_sm_stats.m_stats_map["total_num_spin_phasechk_issued"]->increment_with_integer(1);
    if (n_not_selected_this_cycle > 0) {
      shared_sm->m_sm_stats.m_stats_map["total_num_cycles_spin_won_over_eligible"]->increment_with_integer(1);
    }
  }
  // [intra-SMSP warp-switch] effect counters — the DIRECT causal evidence for the fix (gated; only
  // fires when -intra_smsp_warpswitch_enable filtered a busy-SFU head this cycle). Records, per cycle
  // where the fix acted: how often ANOTHER warp then issued into the freed slot (the recovered slot =
  // proof the warp-switch worked) vs. nobody issued (structural idle the fix cannot recover).
  if (m_config->intra_smsp_warpswitch_enable && intra_ws_sfu_filtered_this_cycle) {
    shared_sm->m_sm_stats.m_stats_map["total_num_intra_warpswitch_sfu_filtered"]->increment_with_integer(1);
    if (is_issued_inst) {
      shared_sm->m_sm_stats.m_stats_map["total_num_intra_warpswitch_other_warp_issued"]->increment_with_integer(1);
    } else {
      shared_sm->m_sm_stats.m_stats_map["total_num_intra_warpswitch_still_idle"]->increment_with_integer(1);
    }
  }
  if(is_issued_inst) {
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_issuing"]->increment_with_integer(1);
    // [NCU stall-taxonomy] `selected` == the issued winner (NCU per-issue denominator).
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_selected"]->increment_with_integer(1);
  }else if(!is_next_stage_availabe){
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_next_stage_not_available"]->increment_with_integer(1);
    // [Head-of-line lever] the warp-scan loop was skipped this cycle (ISSUE_CONTROL latch full);
    // read-only re-scan to size how much of this is recoverable head-of-line blocking. Gated.
    if (m_config->headofline_instrument_enable) {
      scan_head_of_line_when_blocked(shared_sm);
    }
  }else if(is_issue_port_busy) { // IMAD.WIDE scenario
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_issue_port_busy"]->increment_with_integer(1);
  }else if(!is_valid_inst) {
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction"]->increment_with_integer(1);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_decode_pending"]->increment_with_integer(is_any_invalid_head_decode_pending);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_l0i_response_ready"]->increment_with_integer(is_any_invalid_head_l0i_response_ready);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_head_invalid_waiting_frontend"]->increment_with_integer(is_any_invalid_head_waiting_frontend);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_head_invalid_waiting_frontend_miss"]->increment_with_integer(is_any_invalid_head_waiting_frontend_miss);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_head_invalid_waiting_frontend_in_l0i_response_queue"]->increment_with_integer(is_any_invalid_head_waiting_frontend_in_l0i_response_queue);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_head_invalid_waiting_frontend_in_l0i_response_queue_stream_buffer_wait"]->increment_with_integer(is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_head_invalid_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_not_allocated"]->increment_with_integer(is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_not_allocated);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_head_invalid_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_issued_not_ready"]->increment_with_integer(is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_issued_not_ready);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_head_invalid_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_ready_not_promoted"]->increment_with_integer(is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait_prefetch_ready_not_promoted);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_head_invalid_waiting_frontend_in_l0i_response_queue_other"]->increment_with_integer(is_any_invalid_head_waiting_frontend_in_l0i_response_queue_other);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_head_invalid_waiting_frontend_hit_status"]->increment_with_integer(is_any_invalid_head_waiting_frontend_hit_status);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_ibuffer_empty"]->increment_with_integer(is_any_invalid_head_ibuffer_empty);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction_unknown"]->increment_with_integer(is_any_invalid_head_unknown);
  }else { // It has been possible to switch to another warp, but none where ready to issue
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_warps_ready"]->increment_with_integer(1);
    // Per-reason attribution of the no_warps_ready stall (at least one warp
    // blocked for the given reason this cycle). Reasons are not mutually
    // exclusive across warps, so the sub-counters can sum to more than
    // no_warps_ready; compare relative shape against NCU stall decomposition.
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied"]->increment_with_integer(is_any_waiting_in_fu_occupied);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier"]->increment_with_integer(is_any_waiting_in_inst_barrier);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_tma_flush"]->increment_with_integer(is_any_waiting_in_tma_flush);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield"]->increment_with_integer(is_any_waiting_in_yield);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count"]->increment_with_integer(is_any_waiting_in_stall_count);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier"]->increment_with_integer(is_any_waiting_in_wait_barrier);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_scoreboard"]->increment_with_integer(is_any_waiting_in_scoreboard);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_result_queue_full"]->increment_with_integer(is_any_waiting_in_result_queue_full);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c"]->increment_with_integer(is_any_waiting_l1c);
    // [NCU stall-taxonomy] warpgroup_arrive (WGMMA-group wait split out of wait_barrier) +
    // dispatch (RF result-queue / issue-port backpressure re-derived into the per-warp axis).
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_warpgroup_arrive"]->increment_with_integer(is_any_waiting_in_warpgroup_arrive);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_dispatch"]->increment_with_integer(is_any_waiting_in_dispatch);
    // [WGMMA Opt6 Step-0] (I)/(III)/(VI) per-pipe + tensor-specific stall attribution.
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied_tensor"]->increment_with_integer(is_any_fu_occupied_tensor);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied_sfu"]->increment_with_integer(is_any_fu_occupied_sfu);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied_sp_int_dp"]->increment_with_integer(is_any_fu_occupied_sp_int_dp);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied_other"]->increment_with_integer(is_any_fu_occupied_other);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_tensor_reissue_lockout_only"]->increment_with_integer(is_any_tensor_reissue_lockout_only);
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_tensor_fu_occupied_and_wait_barrier_coupled"]->increment_with_integer(is_any_tensor_fu_occupied_and_wait_barrier);
  }
  // [WGMMA Opt6 Step-0] (V) export this subcore's per-cycle issue outcome for SM-level
  // aggregation in SM::cycle(): did it issue, and (if not) was it blocked ONLY by the
  // tensor pipe this cycle? Used to count true SM-wide idle cycles vs tensor-only idle.
  m_step0_issued_this_cycle = is_issued_inst;
  m_step0_blocked_by_tensor_only_this_cycle = is_any_tensor_reissue_lockout_only;
  // [FWD drain-idle 축1] finalize this subcore's mutually-exclusive sole-block reason:
  // issued -> SB_ISSUED; else no valid head all cycle -> SB_DRAINED (floor); else the highest
  // recover-value ONLY_*/MULTI reason seen among valid-head warps. Observe-only.
  if (is_issued_inst)                 m_step0_sole_block_this_cycle = SB_ISSUED;
  else if (!any_valid_head_this_cycle) m_step0_sole_block_this_cycle = SB_DRAINED;
  else                                 m_step0_sole_block_this_cycle = best_sole_block;
  // [CTA-imbalance diag] tensor-FU-busy stall: >=1 warp's head is a WGMMA blocked because the
  // tensor FU is still busy (== NCU `mma`). In this trace-driven model WGMMA is a fixed-latency
  // SPECIALIZED FU op and there is NO separate consumer-side "warpgroup_arrive" scoreboard wait
  // (result waits fold into wait_barrier/DEPBAR, indistinguishable from mbarrier). So the
  // tensor-attributable stall the model DOES expose is this producer/FU-busy signal; we export
  // it (not the scoreboard warpgroup_arrive, which is 0 in trace mode) for per-CTA attribution.
  m_step0_blocked_by_fu_occupied_tensor_this_cycle = is_any_fu_occupied_tensor;
  // [Frontend Step-0] also export whether this subcore had >=1 warp blocked on the L1I
  // stream-buffer frontend this cycle, for the SM-level frontend-idle measurement.
  m_step0_blocked_by_frontend_sbwait_this_cycle = is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait;
  // [Step-0 full SM-idle decomposition] build this subcore's per-cycle non-issue reason mask
  // (only meaningful when it did NOT issue). SM::cycle() ORs these across the 4 subcores on
  // true SM-idle cycles, so one run attributes ALL of sm_all_subcores_idle to every reason.
  // Computed when EITHER Step-0 flag is on (shared infra for WGMMA + frontend measurements).
  if(m_config->wgmma_step0_instrument_enable || m_config->l1i_frontend_step0_instrument_enable) {
    unsigned int mask = 0;
    if(!is_issued_inst) {
      if(!is_next_stage_availabe)      mask |= STEP0_R_NEXT_STAGE;
      else if(is_issue_port_busy)      mask |= STEP0_R_ISSUE_PORT_BUSY;
      else if(!is_valid_inst) {
        if(is_any_invalid_head_waiting_frontend) mask |= STEP0_R_NO_VALID_FRONTEND;
        if(is_any_invalid_head_waiting_frontend_in_l0i_response_queue_stream_buffer_wait) mask |= STEP0_R_NO_VALID_SBWAIT;
        if(!is_any_invalid_head_waiting_frontend) mask |= STEP0_R_NO_VALID_OTHER;
        // Sub-reasons of the "no_valid but NOT frontend-wait" case, so the SM-idle decomposition
        // can tell ibuffer-empty (fetch behind) from decode-pending from response-ready.
        if(is_any_invalid_head_ibuffer_empty)       mask |= STEP0_R_NV_IBUFFER_EMPTY;
        if(is_any_invalid_head_ibuffer_empty_fetch_inflight)   mask |= STEP0_R_NV_IBUF_FETCH_INFLIGHT;
        if(is_any_invalid_head_ibuffer_empty_fetch_not_issued) mask |= STEP0_R_NV_IBUF_FETCH_NOT_ISSUED;
        if(is_any_invalid_head_decode_pending)      mask |= STEP0_R_NV_DECODE_PENDING;
        if(is_any_invalid_head_l0i_response_ready)  mask |= STEP0_R_NV_L0I_RESP_READY;
        if(is_any_invalid_head_unknown)             mask |= STEP0_R_NV_UNKNOWN;
      } else { // no_warps_ready: overlapping per-reason flags
        if(is_any_waiting_in_fu_occupied)        mask |= STEP0_R_FU_OCCUPIED;
        if(is_any_fu_occupied_tensor)            mask |= STEP0_R_FU_OCCUPIED_TENSOR;
        if(is_any_waiting_in_inst_barrier)       mask |= STEP0_R_INST_BARRIER;
        if(is_any_waiting_in_wait_barrier)       mask |= STEP0_R_WAIT_BARRIER;
        if(is_any_waiting_in_tma_flush)          mask |= STEP0_R_TMA_FLUSH;
        if(is_any_waiting_in_stall_count)        mask |= STEP0_R_STALL_COUNT;
        if(is_any_waiting_in_scoreboard)         mask |= STEP0_R_SCOREBOARD;
        if(is_any_waiting_l1c)                   mask |= STEP0_R_L1C;
        if(is_any_waiting_in_result_queue_full)  mask |= STEP0_R_RESULT_QUEUE_FULL;
        if(is_any_waiting_in_yield)              mask |= STEP0_R_YIELD;
      }
    }
    m_step0_reason_mask_this_cycle = mask;
  }
  shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_evaluated"]->increment_with_integer(1);

  m_is_next_stage_of_issue_busy = !is_next_stage_availabe;
}

void Subcore::modify_warp_state() {
  for (auto warp : m_warps_of_subcore) {
    warp->get_dependency_state()->cycle();
    if(m_config->is_interwarp_coalescing_enabled && ((m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC) ||
        (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID)))  {
      if (warp == NULL || warp->done_exit()) {
        continue;
      }
      unsigned int sm_warp_id = warp->get_warp_id();
      bool is_valid_inst_in_the_warp =
      warp->get_IBuffer_remodeled()->is_next_valid();
      if(is_valid_inst_in_the_warp) {
        warp_inst_t *pI = warp->get_IBuffer_remodeled()->next_inst();
        assert(pI != nullptr);
        add_interwarp_coalescing_dep_counter_at_decode_tracking(pI, sm_warp_id);
      }
    }
  }
}

void Subcore::add_interwarp_coalescing_dep_counter_at_decode_tracking(warp_inst_t * pI, unsigned sm_warp_id) {
  std::vector<Wait_Barrier_Checking> wait_barriers_checking_generic;
  if(pI->is_any_kind_of_barrier()) {
    for(unsigned int i = 0; i < m_config->num_wait_barriers_per_warp; i++) {
      wait_barriers_checking_generic.push_back(Wait_Barrier_Checking(i, 0));
    }
  }else{
    wait_barriers_checking_generic = wait_barriers_to_check_generic(pI, sm_warp_id);
    if(pI->op == DEPBAR_OP) {
      std::vector<Wait_Barrier_Checking> wait_barriers_checking_depbar = wait_barriers_to_check_depbar(pI, sm_warp_id);
        wait_barriers_checking_generic.insert(wait_barriers_checking_generic.end(), wait_barriers_checking_depbar.begin(), wait_barriers_checking_depbar.end());
    }
  }
  for(auto &wait_bar : wait_barriers_checking_generic) {
    m_sm->m_interwarp_coal_warps_waiting_dep_counter->m_waiting_dep_counters_per_warp[sm_warp_id].increase_dep_counter(wait_bar.barrier_id);
  }
}

void Subcore::remove_interwarp_coalescing_dep_counter_at_decode_tracking(warp_inst_t * pI, unsigned sm_warp_id) {
  std::vector<Wait_Barrier_Checking> wait_barriers_checking_generic;
  if(pI->is_any_kind_of_barrier()) {
    for(unsigned int i = 0; i < m_config->num_wait_barriers_per_warp; i++) {
      wait_barriers_checking_generic.push_back(Wait_Barrier_Checking(i, 0));
    }
  }else{
    wait_barriers_checking_generic = wait_barriers_to_check_generic(pI, sm_warp_id);
    if(pI->op == DEPBAR_OP) {
      std::vector<Wait_Barrier_Checking> wait_barriers_checking_depbar = wait_barriers_to_check_depbar(pI, sm_warp_id);
        wait_barriers_checking_generic.insert(wait_barriers_checking_generic.end(), wait_barriers_checking_depbar.begin(), wait_barriers_checking_depbar.end());
    }
  }
  for(auto &wait_bar : wait_barriers_checking_generic) {
    m_sm->m_interwarp_coal_warps_waiting_dep_counter->m_waiting_dep_counters_per_warp[sm_warp_id].decrease_dep_counter(wait_bar.barrier_id);
  }
}

void Subcore::set_num_pending_cycles_with_issue_port_busy(const warp_inst_t *pI) {
  if(m_config->is_trace_mode && pI->get_extra_trace_instruction_info().get_is_imad()) {
    m_num_pending_cycles_with_issue_port_busy = m_config->num_cycles_issue_port_busy_after_imadwide;
  }else {
    m_num_pending_cycles_with_issue_port_busy = 0;
  }
}

bool Subcore::is_waiting_ldgdepbar(const warp_inst_t *pI, unsigned int subcore_warp_id) {
  bool res = false;
  if(pI->op ==  LDGDEPBAR_OP) {
    res = m_warps_of_subcore[subcore_warp_id]->get_dependency_state()->are_ldgsts_pending();
  }
  return res;
}

bool Subcore::is_wait_barriers_ready_entry_point(const warp_inst_t *inst,
                                                 unsigned int subcore_warp_id) {
  std::vector<Wait_Barrier_Checking> wait_barriers_checking_generic =
      wait_barriers_to_check_generic(inst, subcore_warp_id);
                                              
  bool are_wait_barriers_ready =
    is_wait_barriers_ready(wait_barriers_checking_generic, subcore_warp_id);
  if (are_wait_barriers_ready && inst->op == DEPBAR_OP) {
    std::vector<Wait_Barrier_Checking> wait_barriers_checking_depbar = wait_barriers_to_check_depbar(inst, subcore_warp_id);
    are_wait_barriers_ready =
    is_wait_barriers_ready(wait_barriers_checking_depbar, subcore_warp_id);
  }
  return are_wait_barriers_ready;
}

std::vector<Wait_Barrier_Checking> Subcore::wait_barriers_to_check_generic(const warp_inst_t* inst, unsigned int subcore_warp_id) {
  int wait_barrier_mask_int = inst->get_extra_trace_instruction_info()
                                  .get_control_bits()
                                  .get_wait_barrier_bits();
  std::bitset<6> wait_barrier_mask(wait_barrier_mask_int);
  std::vector<Wait_Barrier_Checking> wait_barriers_checking;
  for (unsigned int i = 0; i < m_config->num_wait_barriers_per_warp; i++) {
    if (wait_barrier_mask[i]) {
      wait_barriers_checking.push_back(Wait_Barrier_Checking(i, 0));
    }
  }
  return wait_barriers_checking;
}

std::vector<Wait_Barrier_Checking> Subcore::wait_barriers_to_check_depbar(const warp_inst_t* inst, unsigned int subcore_warp_id) {
  std::size_t num_operands = inst->get_extra_trace_instruction_info().get_num_operands();
  assert( num_operands > 1);
  traced_operand& op_sb = inst->get_extra_trace_instruction_info().get_operand(0);
  traced_operand& op_val = inst->get_extra_trace_instruction_info().get_operand(1);
  assert(op_sb.get_operand_type() == TraceEnhancedOperandType::SB);
  assert(op_sb.get_has_reg());
  assert(op_val.get_operand_type() == TraceEnhancedOperandType::IMM_UINT64);
  assert(op_val.get_has_inmediate());
  unsigned int sb_reg = op_sb.get_operand_reg_number();
  unsigned int sb_max_allowed_val = op_val.get_operands_inmediates()[0];
  assert(sb_reg < m_config->num_wait_barriers_per_warp );
  std::vector<Wait_Barrier_Checking> wait_barriers_checking;
  wait_barriers_checking.push_back(Wait_Barrier_Checking(sb_reg, sb_max_allowed_val));
  for (unsigned int i = 2; i < num_operands; i++) {
    assert(inst->get_extra_trace_instruction_info().get_operand(i).get_has_inmediate());
    sb_reg = inst->get_extra_trace_instruction_info().get_operand(i).get_operands_inmediates()[0];
    assert(sb_reg < m_config->num_wait_barriers_per_warp );
    wait_barriers_checking.push_back(Wait_Barrier_Checking(sb_reg, 0));
  }
  return wait_barriers_checking;
}

bool Subcore::is_wait_barriers_ready(std::vector<Wait_Barrier_Checking> &wait_barriers_checking,
                                             unsigned int subcore_warp_id) {
  bool are_wait_barriers_ready =
      m_warps_of_subcore[subcore_warp_id]
          ->get_dependency_state()
          ->are_wait_barriers_ready(wait_barriers_checking);
  return are_wait_barriers_ready;
}

void Subcore::generate_fixed_latency_constant_accesses(warp_inst_t *pI) {
  if(!(pI->is_load() && (pI->space.get_type() == const_space)) && pI->has_extra_trace_instruction_info() && !pI->m_has_the_constant_addr_already_calculated) {
    for(unsigned int i = 0; !pI->get_generated_constant_accesses() && (i < pI->get_extra_trace_instruction_info().get_num_operands()); i++) {
      if(pI->get_extra_trace_instruction_info().get_operand(i).get_operand_type() == TraceEnhancedOperandType::CBANK) {
        new_addr_type addr = calculate_constant_address(0, pI->get_extra_trace_instruction_info().get_operand(i));
        pI->generate_fixed_latency_constant_accesses(addr);
        pI->set_generated_constant_accesses(true);
      }
    }
  }
}

bool Subcore::are_l1c_operands_ready(SM *shared_sm, const warp_inst_t *pI) {
  bool are_l1c_operands_ready = true;
  if(pI->get_generated_constant_accesses() && !pI->accessq_empty()) {
    for(const auto mem_acc : pI->get_mem_accesses()) {
      assert(mem_acc.get_type() == CONST_ACC_R);
      new_addr_type addr = mem_acc.get_addr();
      mem_fetch *mf = shared_sm->get_memf_fetch_allocator().alloc(*pI, mem_acc, shared_sm->get_current_gpu_cycle());
      mf->set_subcore(m_subcore_id);
      mf->set_is_fixed_latency_constant_access(true);
      enum cache_request_status status = HIT;
      if(!m_config->perfect_inst_const_cache && !m_config->perfect_constant_cache) {
        std::list<cache_event> events;
        bool useless = false;
        status = m_L0C_cache->access(addr, mf, shared_sm->get_current_gpu_cycle(), events, useless);
      }
      are_l1c_operands_ready = are_l1c_operands_ready ? (status == HIT) : false;
      if( (status == HIT) || (status == HIT_RESERVED) || (status == RESERVATION_FAIL) ) {
        delete mf;
      }
    }
  }
  return are_l1c_operands_ready;
}

void Subcore::assign_instruction_warp_id(warp_inst_t *pI, unsigned int subcore_warp_id, unsigned int sm_warp_id) {
  unsigned int dynamic_warp_id = m_warps_of_subcore[subcore_warp_id]->get_dynamic_warp_id();
  pI->set_some_warp_attributes(sm_warp_id, dynamic_warp_id);
}

void Subcore::allocate_reads(RF_requests rf_requests, const warp_inst_t *pI, unsigned int sm_warp_id, unsigned int regular_rf_num_read_cycles) {
  assert(rf_requests.m_regular.m_is_possible_to_read);
  m_regular_rf->allocate_reads_cacheable(rf_requests.m_regular, pI, sm_warp_id, regular_rf_num_read_cycles);
  assert(rf_requests.m_uniform.m_is_possible_to_read);
  m_uniform_rf->allocate_reads_cacheable(rf_requests.m_uniform, pI, sm_warp_id, m_config->warp_size);
}


bool Subcore::is_possible_to_write(warp_inst_t *inst, Register_file *dst_rf, unsigned int target_latency_execution_wb, unsigned int &num_uses) {
  bool is_write_available = true;
  num_uses = get_number_of_uses_per_operand(inst->get_extra_trace_instruction_info(), inst->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number(), 0, inst->get_extra_trace_instruction_info().get_operand(0).get_operand_type());
  num_uses = std::min(num_uses, dst_rf->get_num_banks() * dst_rf->get_num_write_ports_per_bank());
  for(unsigned j = 0; (j < num_uses) && is_write_available; j++) {
    unsigned int current_reg_id = inst->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number() + j;
    unsigned int bank_id = dst_rf->calculate_target_bank(current_reg_id);
    is_write_available = dst_rf->is_rf_bank_write_port_available_at_given_cycle(bank_id, target_latency_execution_wb);
  }
  return is_write_available;
}

void Subcore::allocate_writes(warp_inst_t *inst, Register_file *dst_rf, unsigned int num_uses, unsigned int target_latency_execution_wb) {
  for(unsigned j = 0; (j < num_uses) && !inst->m_has_wb_from_sm_struct_to_subcore; j++) {
    unsigned int current_reg_id = inst->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number() + j;
    unsigned int bank_id = dst_rf->calculate_target_bank(current_reg_id);
    dst_rf->allocate_rf_bank_write_port_at_given_cycle(bank_id, target_latency_execution_wb);
  }
}

void Subcore::issue_warp(SM *shared_sm, register_set_uniptr &dispatch_latch, warp_inst_t *pI,
                         const active_mask_t &active_mask, unsigned sm_warp_id,
                         functional_unit* fu, bool is_fixed_latency_inst,
                         bool use_traditional_scoreboarding, bool has_dst_reg, TraceEnhancedOperandType dst_result_queue_type) {
  pI->set_fu_assigned(fu);
  manage_instruction_operand_stats(shared_sm, pI);
  shared_sm->issue_warp(dispatch_latch, pI, active_mask, sm_warp_id, m_subcore_id, use_traditional_scoreboarding);
  if(has_dst_reg && fu->is_fixed_latency_unit()) {
    if(dst_result_queue_type == TraceEnhancedOperandType::UREG) {
      reserve_slot_uniform_fixed_latency_rf_result_queue_space();
    }else {
      reserve_slot_regular_fixed_latency_rf_result_queue_space();
    }
  }
  if(is_fixed_latency_inst) {
    fu->reserve_unit(dispatch_latch);
  }
}

std::vector<unsigned int> Subcore::order_greedy_then_highest_id(SM *shared_sm, unsigned int greedy_pointer) {
  std::vector<unsigned int> result_list;
  std::vector<shd_warp_t*> temp = m_warps_of_subcore;
  result_list.push_back(greedy_pointer);
  std::sort(temp.begin(), temp.end(), sort_warps_by_highest_id_dynamic_id);
  for(auto c_warp : temp) {
    unsigned int warp_subcore_id = translate_warp_id_of_sm_to_subcore( c_warp->get_warp_id(), shared_sm->get_num_subcores() );
    if(!c_warp->done_exit() && (warp_subcore_id != greedy_pointer)) {
      result_list.push_back(warp_subcore_id);
    }
  }
  return result_list;
}

bool Subcore::sort_warps_by_highest_id_dynamic_id(shd_warp_t *lhs,
                                                shd_warp_t *rhs) {
  if (rhs && lhs) {
    if (lhs->done_exit() || lhs->waiting()) {
      return false;
    } else if (rhs->done_exit() || rhs->waiting()) {
      return true;
    } else {
      return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id();
    }
  } else {
    return lhs > rhs;
  }
}

functional_unit* Subcore::get_fu(const warp_inst_t *pI) {
  functional_unit* fu = nullptr;
  switch (pI->op) {
    case PREDICATE_OP:
    case INTP_OP:
      if(m_config->is_fp32_and_int_unified_pipeline) {
        fu = m_sp_pipeline;
      }else {
        fu = m_int_pipeline;
      }
      break;
    case HALF_OP:
      fu = m_sp_pipeline;
    case SP_OP:
      fu = m_sp_pipeline;
      if(m_config->is_fp32ops_allowed_in_int_pipeline && m_int_pipeline->can_issue(pI) && !pI->get_extra_trace_instruction_info().get_is_imad()) { /// INCLUIR AQUI IMAD
        fu = m_int_pipeline;
      }
      break;
    case DP_OP:
      fu = m_dp_pipeline;
      break;
    case UNIFORM_OP:
      fu = m_uniform_pipeline;
      break;
    case TENSOR_CORE_OP:
      fu = m_tensor_pipeline;
      break;
    case CALL_OPS:
    case RET_OPS:
    case EXIT_OPS:
    case BRANCH_OP:
      fu = m_branch_pipeline;
      break;
    case SFU_OP:
      fu = m_sfu_pipeline;
      break;
    case MISCELLANEOUS_QUEUE_OP:
      fu = m_miscellaneous_with_queue_pipeline;
      break;
    case LDGDEPBAR_OP:
    case BARRIER_OP:
    case MBARRIER_OP:
    case DEPBAR_OP:
    case MISCELLANEOUS_NO_QUEUE_OP:
      fu = m_miscellaneous_no_queue_pipeline;
      break;
    case TEXTURE_OP:
    case SURFACE_OP:
    case MEMORY_BARRIER_OP:
    case GRID_BARRIER_OP:
    case MEMORY_MISCELLANEOUS_OP:
      fu = m_memory_unit_subcore;
      break;
    case TMA_LOAD_OP:
    case TMA_STORE_OP:
    case TMA_MISCELLANEOUS_OP:
      fu = m_tma_pipeline;
      break;
    case LOAD_OP:
    case STORE_OP:
      fu = m_memory_unit_subcore;
      break;
    default:
      fflush(stdout);
      std::cout << "ERROR. EXECUTION PIPELINE FOR THIS INSTRUCTION NOT "
                   "IMPLEMENTED"
                << std::endl;
      abort();
  }
  return fu;
}

void Subcore::single_decode(SM *shared_sm, warp_inst_t *pI,
                            IBuffer_Entry &ibuffer_entry,
                            unsigned int sm_warp_id,
                            unsigned int subcore_warp_id, shd_warp_t *warp) {
  if (pI) {
    assign_instruction_warp_id(pI, subcore_warp_id, sm_warp_id);
    generate_fixed_latency_constant_accesses(pI);
    pI->assign_predicate_latencies_if_needed(m_sm->get_gpu());
    warp->inc_inst_in_pipeline();
    pI->set_unique_inst_id(warp->m_last_unique_inst_id);
    if(pI->is_tensor_core_op()) {
      pI->get_tensor_core_instruction_info();
    }
    warp->m_last_unique_inst_id++;
    if (pI->is_tma_op()) {
      pI->generate_other_mem_ops_latencies(m_sm->get_gpu());
    } else if (pI->is_load() || pI->is_store()) {
      pI->generate_mem_latencies(m_sm->get_gpu());
    }else if(pI->is_memory_barrier() || pI->is_grid_barrier() || pI->is_memory_miscelanous()) {
      pI->generate_other_mem_ops_latencies(m_sm->get_gpu());
    }else if(pI->is_texture()) {
      pI->generate_texture_latencies(m_sm->get_gpu());
    } else if (pI->is_dp_op()) {
      pI->generate_dp_latencies(m_sm->get_gpu());      
    }else if(pI->is_tensor_core_op()) {
      pI->generate_tensor_core_latencies(m_sm->get_gpu());
    }else if(pI->is_sfu_useful()) {
      pI->m_num_cycles_to_wait_to_free_WAR = pI->latency + pI->initiation_interval - 3;
    }else if(pI->is_miscellaneous_queue()) {
      pI->generate_miscellaneous_queue_latencies(m_sm->get_gpu());
    }
    ibuffer_entry.m_valid = true;
    
    ibuffer_entry.m_inst = pI;
    assert(ibuffer_entry.m_inst->pc == ibuffer_entry.m_pc);
    if ((pI->oprnd_type == INT_OP) ||
        (pI->oprnd_type == UN_OP)) {  // these counters get added up in
                                      // mcPat to compute scheduler power
      m_stats->m_num_INTdecoded_insn[shared_sm->get_sid()]++;
    } else if (pI->oprnd_type == FP_OP) {
      m_stats->m_num_FPdecoded_insn[shared_sm->get_sid()]++;
    }

    if(m_config->is_interwarp_coalescing_enabled && ( (m_config->interwarp_coalescing_selection_policy == InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC) || (m_config->interwarp_coalescing_selection_policy == InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID) ) ) {
      add_interwarp_coalescing_dep_counter_at_decode_tracking(pI, sm_warp_id);
    }
  }
}

void Subcore::decode(SM *shared_sm) {
  if (m_inst_fetch_decode_latch.m_valid) {
    address_type pc = m_inst_fetch_decode_latch.m_pc;
    unsigned long long current_cycle = shared_sm->get_current_gpu_cycle();
    m_stats->m_num_decoded_insn[shared_sm->get_sid()]++;
    if(m_config->ibuffer_coalescing) {
      for (auto warp : m_warps_of_subcore) {
        if (!warp->functional_done()) {
          unsigned int sm_warp_id = warp->get_warp_id();
          unsigned int subcore_warp_id = translate_warp_id_of_sm_to_subcore(
              sm_warp_id, shared_sm->get_num_subcores());
          for (auto &ibuffer_entry :
              warp->get_IBuffer_remodeled()->get_remodeled_ibuffer()) {
            if (!ibuffer_entry.m_valid && (ibuffer_entry.m_pc == pc)) {
              shared_sm->m_sm_stats.m_stats_map["total_num_ibuffer_entries_decoded"]->increment_with_integer(1);
              warp_inst_t *pI = get_next_inst(shared_sm, subcore_warp_id, pc);
              single_decode(shared_sm, pI, ibuffer_entry, sm_warp_id,
                            subcore_warp_id, warp);
            }
          }
        }
      }
    }else {
      
      unsigned int subcore_warp_id = m_inst_fetch_decode_latch.m_warp_id;
      shd_warp_t *warp = m_warps_of_subcore[subcore_warp_id];
      unsigned int sm_warp_id = warp->get_warp_id();
      for (auto &ibuffer_entry :
           warp->get_IBuffer_remodeled()->get_remodeled_ibuffer()) {
        if (!ibuffer_entry.m_valid && (ibuffer_entry.m_pc == pc)) {
          shared_sm->m_sm_stats.m_stats_map["total_num_ibuffer_entries_decoded"]->increment_with_integer(1);
          warp_inst_t *pI = get_next_inst(shared_sm, subcore_warp_id, pc);
          single_decode(shared_sm, pI, ibuffer_entry, sm_warp_id,
                        subcore_warp_id, warp);
        }
      }
    }
    m_inst_fetch_decode_latch.m_valid = false;
  }
}

warp_inst_t *Subcore::get_next_inst(SM *shared_sm, unsigned int warp_id, address_type pc) {
  assert(warp_id < m_warps_of_subcore.size());
  if (m_config->is_trace_mode) {
    // read the inst from the traces
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(
        m_warps_of_subcore[warp_id]);
    return m_trace_warp->get_next_trace_inst(pc);
  } else {
    return shared_sm->get_gpu()->gpgpu_ctx->ptx_fetch_inst(pc);
  }
}

void Subcore::fetch(SM *shared_sm) {
  if (!m_inst_fetch_decode_latch.m_valid) {
    unsigned long long current_cycle = shared_sm->get_current_gpu_cycle();
    auto mark_response_ready = [&](unsigned int target_subcore_warp_id,
                                   address_type pc) {
      if (m_config->ibuffer_coalescing) {
        for (auto warp : m_warps_of_subcore) {
          if (warp == NULL || warp->functional_done()) {
            continue;
          }
          for (auto &ibuffer_entry :
               warp->get_IBuffer_remodeled()->get_remodeled_ibuffer()) {
            if (!ibuffer_entry.m_valid && (ibuffer_entry.m_pc == pc)) {
              shared_sm->m_sm_stats.m_stats_map["total_num_ibuffer_entries_response_ready"]->increment_with_integer(1);
            }
          }
        }
      } else {
        shd_warp_t *warp = m_warps_of_subcore[target_subcore_warp_id];
        for (auto &ibuffer_entry :
             warp->get_IBuffer_remodeled()->get_remodeled_ibuffer()) {
          if (!ibuffer_entry.m_valid && (ibuffer_entry.m_pc == pc)) {
            shared_sm->m_sm_stats.m_stats_map["total_num_ibuffer_entries_response_ready"]->increment_with_integer(1);
          }
        }
      }
    };
    if (m_L0I->is_first_access_ready()) {
      mem_fetch *mf = m_L0I->next_first_access();
      unsigned int unique_function_id = mf->get_unique_function_id();
      unsigned int subcore_warp_id = translate_warp_id_of_sm_to_subcore(
          mf->get_wid(), shared_sm->get_num_subcores());
      address_type local_pc_response =
          shared_sm->from_global_pc_address_to_local_pc(mf->get_addr(), unique_function_id);
      mark_response_ready(subcore_warp_id, local_pc_response);
      m_inst_fetch_decode_latch = ifetch_buffer_t(
          local_pc_response, mf->get_access_size(), subcore_warp_id);
      // if(shared_sm->get_sid() == 1 && m_subcore_id == 0) {
      //   std::cout << "Fetch Response. SM: " << shared_sm->get_sid() << ". Subcore: " << m_subcore_id << ". Warp_ID: " << mf->get_wid() << ". PC: " << std::hex << local_pc_response << std::dec << ". Cycle: " << shared_sm->get_current_gpu_cycle() << std::endl;
      //   fflush(stdout);
      // }
      m_inst_fetch_decode_latch.m_valid = true;
      m_warps_of_subcore[subcore_warp_id]->set_last_fetch(
          shared_sm->get_gpu()->gpu_sim_cycle);
      delete mf;
    }
    std::vector<unsigned int> priority_ordered_for_fetch = order_greedy_then_highest_id(shared_sm, m_greedy_pointer_fetch);
    for (auto c_warp_id : priority_ordered_for_fetch) {
      shd_warp_t *c_warp = m_warps_of_subcore[c_warp_id];
      shared_sm->check_if_warp_has_finished_executing_and_can_be_reclaim(c_warp);
      if (c_warp->functional_done()) {
        continue;
      }
      unsigned int sm_warp_id = c_warp->get_warp_id();
      unsigned int subcore_warp_id = translate_warp_id_of_sm_to_subcore(
          sm_warp_id, shared_sm->get_num_subcores());
      assert(subcore_warp_id < m_warps_of_subcore.size() &&
             sm_warp_id < shared_sm->get_config()->max_warps_per_shader);
      bool is_ibuffer_with_space = m_warps_of_subcore[subcore_warp_id]
                                       ->get_IBuffer_remodeled()
                                       ->can_fetch();
      if (is_ibuffer_with_space) {
        address_type local_pc_request = m_warps_of_subcore[subcore_warp_id]
                                            ->get_IBuffer_remodeled()
                                            ->get_next_pc_to_fetch_request();
        // Request different address for different kernels
        unsigned int unique_function_id = c_warp->get_current_unique_function_id_call();
        address_type global_pc_addr =
            shared_sm->from_local_pc_to_global_pc_address(local_pc_request, unique_function_id);
        unsigned int line_size = m_config->m_L0I_config.get_line_sz();
        unsigned int nbytes = num_bytes_cache_req(line_size, local_pc_request);

        // TODO: replace with use of allocator
        // mem_fetch *mf = m_mem_fetch_allocator->alloc()
        mem_access_t acc(INST_ACC_R, global_pc_addr, nbytes, false,
                         shared_sm->get_gpu()->gpgpu_ctx);
        mem_fetch *mf =
            new mem_fetch(acc, NULL /*we don't have an instruction yet*/,
                          READ_PACKET_SIZE, sm_warp_id, shared_sm->get_sid(),
                          shared_sm->get_tpc_id(), shared_sm->get_memory_config(),
                          shared_sm->get_gpu()->gpu_tot_sim_cycle +
                              shared_sm->get_gpu()->gpu_sim_cycle, NULL, NULL, unique_function_id);
        mf->set_subcore(m_subcore_id);
        std::list<cache_event> events;
        enum cache_request_status status;
        if (m_config->perfect_inst_const_cache || m_config->perfect_instruction_cache) {
          status = HIT;
          shader_cache_access_log(shared_sm->get_sid(), INSTRUCTION, 0);
          m_inst_fetch_decode_latch =
              ifetch_buffer_t(local_pc_request, nbytes, subcore_warp_id);
          delete mf;
        } else {
          status = m_L0I->access((new_addr_type)global_pc_addr, mf,
                                 shared_sm->get_current_gpu_cycle(),
                                 events);
          if ((status == HIT) && !m_inst_fetch_decode_latch.m_valid &&
              m_L0I->is_first_access_ready()) {
            mem_fetch *hit_mf = m_L0I->next_first_access();
            unsigned int hit_unique_function_id =
                hit_mf->get_unique_function_id();
            unsigned int hit_subcore_warp_id =
                translate_warp_id_of_sm_to_subcore(
                    hit_mf->get_wid(), shared_sm->get_num_subcores());
            address_type local_pc_response =
                shared_sm->from_global_pc_address_to_local_pc(
                    hit_mf->get_addr(), hit_unique_function_id);
            mark_response_ready(hit_subcore_warp_id, local_pc_response);
            m_inst_fetch_decode_latch = ifetch_buffer_t(
                local_pc_response, hit_mf->get_access_size(),
                hit_subcore_warp_id);
            m_inst_fetch_decode_latch.m_valid = true;
            m_warps_of_subcore[hit_subcore_warp_id]->set_last_fetch(
                shared_sm->get_gpu()->gpu_sim_cycle);
            delete hit_mf;
          }
        }
        
        // if(shared_sm->get_sid() == 1 && m_subcore_id == 0) {
        //   std::cout << "Fetch Request. SM: " << shared_sm->get_sid() << ". Subcore: " << m_subcore_id << ". Warp_ID: " << mf->get_wid() << ". PC: " << std::hex << local_pc_request << std::dec << ". Status: " << status << ". Cycle: " << shared_sm->get_current_gpu_cycle() << std::endl;
        //   fflush(stdout);
        // }

        if ((status == MISS) || (status == HIT) ||
            (status == IN_L0I_RESPONSE_QUEUE)) {
          m_warps_of_subcore[subcore_warp_id]->set_last_fetch(
              shared_sm->get_gpu()->gpu_sim_cycle);
        } else {
          assert(status == RESERVATION_FAIL);
          m_warps_of_subcore[subcore_warp_id]->get_IBuffer_remodeled()
                                            ->remove_entry(local_pc_request);
        }
        if( (status == RESERVATION_FAIL) ||  (status == IN_L0I_RESPONSE_QUEUE)) {
          delete mf;
        }

        break;
      }
    }
  }
}

void Subcore::assign_warp_to_subcore(shd_warp_t *warp) {
  m_warps_of_subcore.push_back(warp);
  warp->m_subcore = this;
}

void Subcore::finilized_warps_assignation() {
  m_greedy_pointer_issue = m_warps_of_subcore.size() - 1;
  m_greedy_pointer_fetch = m_warps_of_subcore.size() - 1;
}

void Subcore::create_pipeline() {  
  SM *shared_sm = get_sm();
  create_register_file(shared_sm);
  unsigned int num_intermediate_cycles_until_fu_execution = NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST;
  if(!m_config->is_fp32_and_int_unified_pipeline) {
    m_int_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->max_int_latency, "INT", shared_sm, INTP__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
      &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
    m_all_subcore_ex_pipelines.push_back(m_int_pipeline);
  }
  m_sp_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->max_sp_latency, "SP", shared_sm, SP__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
      &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  m_uniform_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->uniform_latency, "UNIFORM", shared_sm, SPECIALIZED__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
      &m_uniform_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::UREG);
  m_tensor_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->tensor_latency, "TENSOR", shared_sm, SPECIALIZED__OP, true, false, 1, NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST_TENSOR_CORE_INSTS_WITH_4_REGS_PER_OP, 
      &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  m_branch_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->branch_latency, "BRANCH", shared_sm, SPECIALIZED__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
    &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  m_sfu_pipeline = new functional_unit_sfu(nullptr, m_regular_rf, m_config, m_config->sfu_latency, "SFU", shared_sm, SFU__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
    &m_EX_WB_sm_variable_latency_latch, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  m_miscellaneous_with_queue_pipeline = new functional_unit_with_queue( //nullptr, 0////////////////////// ?
      nullptr, m_regular_rf, m_config, m_config->miscellaneous_queue_latency, "MISC_QUEUE", shared_sm, SPECIALIZED__OP, true, true, m_config->miscellaneous_queue_size,
      num_intermediate_cycles_until_fu_execution,  &m_EX_WB_sm_variable_latency_latch, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, 1, 0, TraceEnhancedOperandType::REG);
  m_miscellaneous_no_queue_pipeline = new functional_unit( nullptr, m_regular_rf, m_config, m_config->miscellaneous_no_queue_latency, "MISC_NO_QUEUE", shared_sm, SPECIALIZED__OP,
      true, false, 1, num_intermediate_cycles_until_fu_execution, &m_regular_fixed_latency_rf_write_queue,  m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
      
  m_memory_unit_subcore = new functional_unit_with_queue( m_EX_MEM_shared_sm_reception_latch, m_regular_rf, m_config, 1, "MEM_SUBCORE_UNIT", shared_sm, MEM__OP, true, true, m_config->memory_subcore_queue_size,
      num_intermediate_cycles_until_fu_execution, nullptr, 0, true, m_config->memory_intermidiate_stages_subcore_unit, 
      m_config->num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst, TraceEnhancedOperandType::NONE);

  m_tma_pipeline = new functional_unit_with_queue(m_EX_TMA_shared_sm_reception_latch, m_regular_rf, m_config, 1, "TMA_SUBCORE_UNIT", shared_sm, MEM__OP, true, true, m_config->memory_subcore_queue_size,
      num_intermediate_cycles_until_fu_execution, nullptr, 0, true, m_config->memory_intermidiate_stages_subcore_unit,
      m_config->num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst, TraceEnhancedOperandType::NONE);

  if (m_config->is_dp_pipeline_shared_for_subcores) {
    m_dp_pipeline = new functional_unit_with_queue(m_EX_DP_shared_sm_reception_latch, m_regular_rf, m_config, m_config->dp_subcore_max_latency, "DP_SUBCORE_UNIT", shared_sm, DP__OP, true, true, 
        m_config->dp_subcore_queue_size, num_intermediate_cycles_until_fu_execution, nullptr, 0, true, m_config->dp_shared_intermidiate_stages, 
        m_config->num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_dp_inst, TraceEnhancedOperandType::NONE);
  } else {
    m_dp_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->max_dp_latency, "DP", shared_sm, DP__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
        &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  }
  
  m_all_subcore_ex_pipelines.push_back(m_sp_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_uniform_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_tensor_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_branch_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_miscellaneous_no_queue_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_memory_unit_subcore);
  m_all_subcore_ex_pipelines.push_back(m_tma_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_dp_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_sfu_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_miscellaneous_with_queue_pipeline);
}

void Subcore::create_register_file(SM *shared_sm) {
  m_num_regular_rf_banks = m_config->gpgpu_num_reg_banks / shared_sm->get_num_subcores();
  assert( (m_config->gpgpu_num_reg_banks % shared_sm->get_num_subcores()) == 0);
  m_regular_rf = new Register_file(m_num_regular_rf_banks, m_config->num_regular_register_file_read_ports_per_bank, m_config->num_regular_register_file_write_ports_per_bank, m_config->max_latency_regular_register_file_latency, false, false, m_config->max_operands_regular_register_file, m_stats, getptr(), TraceEnhancedOperandType::REG, m_config->is_rf_cache_enabled);
  m_uniform_rf = new Register_file(m_num_regular_rf_banks, MAX_SRC, MAX_DST, m_config->max_latency_regular_register_file_latency, true, true, m_config->max_operands_regular_register_file, m_stats, getptr(), TraceEnhancedOperandType::UREG, false);
  m_regular_rf->init();
  m_uniform_rf->init();
}

void Subcore::create_L0s(mem_fetch_interface *icnt_icache) {
#define STRSIZE 1024
  SM *shared_sm = get_sm();
  char nameL0I[STRSIZE];
  char nameL0C[STRSIZE];
  snprintf(nameL0I, STRSIZE, "L0I_%03d_%d", shared_sm->get_sid(), m_subcore_id);
  m_L0I = new first_level_instruction_cache(
      "L0I", m_config->m_L0I_config, shared_sm->get_sid(),
      get_shader_instruction_cache_id(), icnt_icache, IN_L0_MISS_QUEUE, m_config->is_instruction_prefetching_enabled, m_subcore_id, get_sm(), m_config->num_instruction_prefetches_per_cycle, m_config->ibuffer_coalescing, m_config->prefetch_per_stream_buffer_size, m_config->prefetch_num_stream_buffers);
  m_L0I->initiate_stream_buffers();

  snprintf(nameL0C, STRSIZE, "L0C_%03d_%d", shared_sm->get_sid(), m_subcore_id);
  m_L0C_cache = new read_only_cache(nameL0C, m_config->m_L0C_config, shared_sm->get_sid(),
                              get_shader_constant_cache_id(), icnt_icache,
                              IN_L1C_MISS_QUEUE);
}

first_level_instruction_cache* Subcore::get_L0I() { return m_L0I; }
read_only_cache* Subcore::get_L0C() { return m_L0C_cache; }

void Subcore::get_L0I_sub_stats(struct cache_sub_stats &css) const {
  m_L0I->get_sub_stats(css);
}

register_set_uniptr *Subcore::get_EX_WB_sm_shared_units_latch() {
  return &m_EX_WB_sm_shared_units_latch;
}

unsigned int Subcore::get_subcore_id() { return m_subcore_id; }

SM *Subcore::get_sm() { 
  return m_sm;
}

void Subcore::manage_instruction_operand_stats(SM *shared_sm, warp_inst_t *pI) {
  unsigned int first_read_operand = pI->get_extra_trace_instruction_info().get_num_destination_registers();
  if(pI->get_extra_trace_instruction_info().has_destination_registers()) {
    unsigned int num_of_accesses = get_number_of_uses_per_operand(pI->get_extra_trace_instruction_info(), pI->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number(), 0, pI->get_extra_trace_instruction_info().get_operand(0).get_operand_type());
    switch(pI->get_extra_trace_instruction_info().get_operand(0).get_operand_type()) {
      case TraceEnhancedOperandType::MREF: // Only these two are used for power compute
      case TraceEnhancedOperandType::REG:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_regular_regfile_writes);
        break;
      case TraceEnhancedOperandType::UREG:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_uniform_regfile_writes);
        break;
      case TraceEnhancedOperandType::PRED:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_predicate_regfile_writes);
        break;
      case TraceEnhancedOperandType::UPRED:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_uniform_predicate_regfile_writes);
        break;
      default:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::incnon_rf_operands);
        break;
    }
  }

  for(unsigned int i = first_read_operand; i < pI->get_extra_trace_instruction_info().get_num_operands(); i++) {
    unsigned int reg_id = pI->get_extra_trace_instruction_info().get_operand(i).get_operand_reg_number();
    TraceEnhancedOperandType reg_type = pI->get_extra_trace_instruction_info().get_operand(i).get_operand_type();
    if(pI->get_extra_trace_instruction_info().get_operand(i).get_has_reg() && !is_reserved_reg(reg_id, reg_type)) {
      unsigned int num_of_accesses = get_number_of_uses_per_operand(pI->get_extra_trace_instruction_info(), reg_id , i, reg_type);
      if(pI->is_tensor_core_op() && num_of_accesses == 4) {
        pI->m_is_tensor_core_op_with_4_registers_per_op = true;
      }
      switch(reg_type) {
        case TraceEnhancedOperandType::MREF: // Only these two are used for power compute
        case TraceEnhancedOperandType::REG:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_regular_regfile_reads);
          break;
        case TraceEnhancedOperandType::UREG:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_uniform_regfile_reads);
          break;
        case TraceEnhancedOperandType::PRED:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_predicate_regfile_reads);
          break;
        case TraceEnhancedOperandType::UPRED:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_uniform_predicate_regfile_reads);
          break;
        case TraceEnhancedOperandType::CBANK:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_constant_cache_reads);
          break;
        default: // Operands that might seems like registers but we are not considering like that. (e.g SR, SB, B)
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::incnon_rf_operands);
          break;
      }
    }else {
      // Operands that does not have registers as source (e.g. immediate values)
      // For the moment, we count reserved registers (RZ, URZ, PT, UPT) as non RF accesses.
      manage_operand_stat(shared_sm, pI, 1, &Subcore::incnon_rf_operands);
      break;
    }
  }
}


void Subcore::manage_operand_stat(SM *shared_sm, const warp_inst_t *pI, unsigned int num_accesses_per_operand,
                                  void (Subcore::*increase_stat)(unsigned int, SM *shared_sm)) {
  if (shared_sm->get_config()->gpgpu_clock_gated_reg_file) {
    unsigned active_count = 0;
    for (unsigned i = 0; i < shared_sm->get_config()->warp_size;
         i = i + shared_sm->get_config()->n_regfile_gating_group) {
      for (unsigned j = 0; j < shared_sm->get_config()->n_regfile_gating_group;
           j++) {
        if (pI->get_active_mask().test(i + j)) {
          active_count += shared_sm->get_config()->n_regfile_gating_group;
          break;
        }
      }
    }
    (this->*increase_stat)(num_accesses_per_operand * active_count, shared_sm);
  } else {
    (this->*increase_stat)(num_accesses_per_operand * (shared_sm->get_config()->warp_size), shared_sm);
  }
}

void Subcore::inc_regular_regfile_reads(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_regular_regfile_reads"]->increment_with_integer(active_count);
  get_sm()->incregfile_reads(active_count);
}

void Subcore::inc_regular_regfile_writes(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_regular_regfile_writes"]->increment_with_integer(active_count);
  get_sm()->incregfile_writes(active_count); 
}

void Subcore::inc_uniform_regfile_reads(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_uniform_regfile_reads"]->increment_with_integer(active_count);
}

void Subcore::inc_uniform_regfile_writes(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_uniform_regfile_writes"]->increment_with_integer(active_count);
}

void Subcore::inc_predicate_regfile_reads(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_predicate_regfile_reads"]->increment_with_integer(active_count);
}

void Subcore::inc_predicate_regfile_writes(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_predicate_regfile_writes"]->increment_with_integer(active_count);
}

void Subcore::inc_uniform_predicate_regfile_reads(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_uniform_predicate_regfile_reads"]->increment_with_integer(active_count);
}

void Subcore::inc_uniform_predicate_regfile_writes(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_uniform_predicate_regfile_writes"]->increment_with_integer(active_count);
}

void Subcore::inc_constant_cache_reads(unsigned int active_count, SM *shared_sm) { 
  shared_sm->m_sm_stats.m_stats_map["total_num_constant_cache_reads"]->increment_with_integer(active_count);
}

void Subcore::incnon_rf_operands(unsigned int active_count, SM *shared_sm) { 
  get_sm()->incnon_rf_operands(active_count); 
}
