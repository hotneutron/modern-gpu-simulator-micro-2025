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


#include "sm.h"

#include "../../../../trace-driven/trace_driven.h"
#include "../../abstract_hardware_model.h"
#include "../../cuda-sim/cuda-sim.h"
#include "../../cuda-sim/ptx_ir.h"

#include "../gpu-sim.h"
#include "l0_icnt.h"
#include "../shader.h"
#include "../shader_trace.h"
#include "tma_unit_sm.h"
#include "../stat-tool.h"

#include "first_level_instruction_cache.h"
#include "functional_unit.h"
#include "warp_dependency_state.h"

#include "../../../../../util/traces_enhanced/src/traced_operand.h"

#include <algorithm>
#include <sstream>

#define STRSIZE 1024

unsigned int translate_warp_id_of_sm_to_subcore(unsigned int warp_id,
                                                unsigned int num_subcores) {
  return warp_id / num_subcores;
}

TraceEnhancedOperandType get_reg_type_eval(traced_operand& op) {
  TraceEnhancedOperandType reg_type = op.get_operand_type();
  if( (reg_type == TraceEnhancedOperandType::MREF) || (reg_type == TraceEnhancedOperandType::CBANK) || (reg_type == TraceEnhancedOperandType::DESC)) {
    if(op.get_operand_string().find("UR") != std::string::npos) {
      reg_type = TraceEnhancedOperandType::UREG;
    }else if(op.get_operand_string().find("R") != std::string::npos) {
      reg_type = TraceEnhancedOperandType::REG;
    }
  }
  return reg_type;
}

bool check_is_reserved_regs_remodeling(int reg, TraceEnhancedOperandType reg_type, bool is_trace_mode) {
  bool res = false;
  if(is_trace_mode) {
    if(reg_type == TraceEnhancedOperandType::REG) {
      res = (reg == 255);
    }else if(reg_type == TraceEnhancedOperandType::UREG) {
      res = (reg == 63);
    }else if(reg_type == TraceEnhancedOperandType::PRED) {
      res = (reg == 7);
    }else if(reg_type == TraceEnhancedOperandType::UPRED) {
      res = (reg == 7);
    }
  }
  return res;
}

unsigned int translate_reg_to_global_id(int reg, TraceEnhancedOperandType reg_type) {
  unsigned int global_id = 0;
  if(reg_type == TraceEnhancedOperandType::REG) {
    global_id = reg;
  }else if(reg_type == TraceEnhancedOperandType::UREG) {
    global_id = 256 + reg;
  }else if(reg_type == TraceEnhancedOperandType::PRED) {
    global_id = 512;
    if(reg != PR) {
      global_id += reg;
    }
  }else if(reg_type == TraceEnhancedOperandType::UPRED) {
    global_id = 520;
    if(reg != UPR) {
      global_id += reg;
    }
  }
  return global_id;
}

namespace {

static constexpr uint32_t kSyncExchDecodeBase = 0x200000;
// Policy thresholds from WAIT_BLOCK.md 2.6 and SPIN_WAIT_SIM_MODELING.md 6.
// They are strategy-wide safety limits, not trace/application identities.
static constexpr uint64_t kSpinWaitObserveWatchdogCycles = 400000;
static constexpr uint64_t kWaitBlockHoldWatchdogCycles = 32000;
static constexpr uint64_t kWaitBlockHeartbeatCycles = 4096;

uint32_t decode_sync_exch_expected_arrive_count(uint64_t raw) {
  if (raw >= kSyncExchDecodeBase) {
    return 0;
  }
  // EXCH encodes 2x the logical expected-arrival count (validated against
  // microbenchmarks: knob 3 -> raw giving 6, 4 -> 8, 8 -> 16). The logical
  // arrival count is in thread units, so divide the encoded delta by 2.
  return static_cast<uint32_t>((kSyncExchDecodeBase - raw) / 2);
}

uint32_t decode_sync_wait_phase(uint64_t wait_state_raw) {
  // The PHASECHK / TRYWAIT phaseParity operand is encoded in bit 31 of the
  // captured wait_state (validated against FA3 traces: the operand only ever
  // takes the values 0x0 and 0x80000000, differing solely in bit 31). Reading
  // bit 0 folded both values to 0 and discarded the parity entirely.
  return static_cast<uint32_t>((wait_state_raw >> 31) & 0x1ULL);
}

std::string format_hex_u64(uint64_t value) {
  std::ostringstream oss;
  oss << std::hex << value;
  return oss.str();
}

const char *sync_kind_to_string(SyncInstructionKind kind) {
  switch (kind) {
    case SyncInstructionKind::EXCH:
      return "EXCH";
    case SyncInstructionKind::ARRIVE:
      return "ARRIVE";
    case SyncInstructionKind::ARRIVE_EXPECT_TX:
      return "ARRIVE_EXPECT_TX";
    case SyncInstructionKind::ARRIVE_COUNTED:
      return "ARRIVE_COUNTED";
    case SyncInstructionKind::PHASECHK:
      return "PHASECHK";
    case SyncInstructionKind::TRYWAIT:
      return "TRYWAIT";
    case SyncInstructionKind::NONE:
    default:
      return "NONE";
  }
}

const char *wait_block_mode_to_string(WaitBlockPendingMode mode) {
  switch (mode) {
    case WaitBlockPendingMode::MISSING_WAIT_HOLD:
      return "INJECT_MISSING_WAIT";
    case WaitBlockPendingMode::PARTIAL_MISSING_WAIT_HOLD:
      return "DEFER_PARTIAL_MISSING_WAIT";
    case WaitBlockPendingMode::FOLD_AUX_WAIT_HOLD:
      return "FOLD_AUX_WAIT";
    case WaitBlockPendingMode::CAPTURED_POLL_HOLD:
      return "FOLD_CAPTURED_POLL";
    case WaitBlockPendingMode::NONE:
    default:
      return "NONE";
  }
}

// A memory barrier (MEMBAR / FENCE) is an ordering/visibility fence, NOT a
// thread rendezvous: only the issuing warp waits (until its own outstanding
// memory writes are visible at the requested scope), and it never waits for the
// other warps of the CTA to arrive. Such barriers must therefore bypass the
// CTA barrier engine (warp_reaches_barrier()) and be modeled purely as a
// per-warp memory-dependency wait (set_membar() -> warp_waiting_at_mem_barrier).
//
// Returns true for every memory-barrier form whose non-rendezvous semantics
// have been verified against the FA3 fwd/bwd traces:
//   - FENCE.*            (e.g. FENCE.VIEW.ASYNC.S / .G)  async-proxy fences
//   - MEMBAR.ALL.CTA / MEMBAR.ALL.GPU                    CTA/GPU-scope fences
// Any other MEMBAR form (e.g. MEMBAR.*.SYS) has not been characterized; rather
// than silently mis-model it, abort so an unverified form is never treated as a
// fence (mirrors the OP_BAR decode guard).
bool is_non_rendezvous_memory_barrier(const warp_inst_t &inst) {
  if (inst.op != MEMORY_BARRIER_OP || !inst.has_extra_trace_instruction_info()) {
    return false;
  }
  const std::string opcode =
      inst.get_extra_trace_instruction_info().get_op_code();
  if (opcode.rfind("FENCE", 0) == 0) return true;
  if (opcode.rfind("MEMBAR", 0) == 0) {
    if (opcode == "MEMBAR.ALL.CTA" || opcode == "MEMBAR.ALL.GPU") {
      return true;
    }
    assert(false &&
           "Unverified MEMBAR form reached the memory-barrier path (only "
           "MEMBAR.ALL.CTA and MEMBAR.ALL.GPU have been characterized as "
           "non-rendezvous fences); characterize it before modeling.");
  }
  return false;
}

// Determine the visibility scope a memory fence enforces, from the trace opcode.
//   MEMBAR.ALL.GPU            -> GPU scope (drain to L2, all SMs)
//   MEMBAR.ALL.CTA           -> CTA scope (drain to shared/L1, same SM)
//   FENCE.* (async proxy)    -> CTA scope (intra-SM ordering); its real SB wait is
//                               enforced separately via wait_barrier_bits.
// Only forms verified for the FA3 fwd/bwd traces are mapped; anything else that
// reaches here returns NONE (no per-warp store drain) but is still gated by the
// generic wait-barrier path. (MEMBAR.*.SYS is already asserted out upstream.)
static membar_scope_t fence_scope_of(const warp_inst_t &inst) {
  if (inst.op != MEMORY_BARRIER_OP || !inst.has_extra_trace_instruction_info()) {
    return MEMBAR_SCOPE_NONE;
  }
  const std::string opcode =
      inst.get_extra_trace_instruction_info().get_op_code();
  if (opcode == "MEMBAR.ALL.GPU") return MEMBAR_SCOPE_GPU;
  if (opcode == "MEMBAR.ALL.CTA") return MEMBAR_SCOPE_CTA;
  if (opcode.rfind("FENCE", 0) == 0) return MEMBAR_SCOPE_CTA;
  return MEMBAR_SCOPE_NONE;
}

// SYNCS.ARRIVE.TRANS64 variants whose runtime semantics have been validated end
// to end (see .plan/SYNC_ISA.md). The simulator classifies arrive vs
// arrive-expect-tx purely from the captured semantic_raw, so these only need a
// confirmed operand layout. Any other arrive variant that actually executes is
// unverified and must trip the guard in handle_sync_instruction().
bool is_validated_arrive_opcode(const std::string &opcode) {
  return opcode == "SYNCS.ARRIVE.TRANS64" ||             // suffix-less, tx bytes
         opcode == "SYNCS.ARRIVE.TRANS64.RED.A1T0" ||    // operand2 RZ, plain
         opcode == "SYNCS.ARRIVE.TRANS64.RED.A0TR" ||    // nvcc, tx bytes
         opcode == "SYNCS.ARRIVE.TRANS64.ART0" ||        // nvcc, plain
         opcode == "SYNCS.ARRIVE.TRANS64.A1T0";          // nvcc, plain
}

bool is_wait_block_timing_semantic_opcode(const std::string &opcode) {
  static const char *const prefixes[] = {
      "HGMMA",  "IGMMA",   "QGMMA",  "BGMMA", "OGMMA", "MMA",
      "WGMMA",  "MUFU",    "UTMALDG", "UTMASTG", "UTMACMDFLUSH",
      "UBLKCP", "UBLKPF",  "UTMAPF", "UTMAREDG", "UBLKRED",
      "UTMACCTL", "LDGSTS", "STG", "LDG", "STS", "LDS", "ATOM",
      "ATOMS", "RED", "SYNCS.ARRIVE", "SYNCS.EXCH", "BAR"};
  for (const char *prefix : prefixes) {
    if (opcode.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

void debug_print_sm_barrier_issue(const warp_inst_t &inst,
                                  unsigned int warp_id,
                                  unsigned int sm_id,
                                  bool enable,
                                  unsigned int issue_budget) {
  if (!enable) {
    return;
  }
  static unsigned int budget_left = 0;
  static bool budget_init = false;
  if (!budget_init) {
    budget_left = issue_budget;
    budget_init = true;
  }
  if (budget_left == 0) {
    return;
  }
  const std::string trace_opcode =
      inst.has_extra_trace_instruction_info()
          ? inst.get_extra_trace_instruction_info().get_op_code()
          : "<no-trace-opcode>";
  int wait_barrier_bits =
      inst.has_extra_trace_instruction_info()
          ? inst.get_extra_trace_instruction_info().get_control_bits().get_wait_barrier_bits()
          : -1;
  std::ostringstream oss;
  oss << "[BARDBG][issue] sm=" << sm_id
      << " warp=" << warp_id
      << " pc=0x" << format_hex_u64(inst.pc)
      << " op=" << static_cast<int>(inst.op)
      << " trace_opcode=" << trace_opcode
      << " bar_type=" << static_cast<int>(inst.bar_type)
      << " bar_subop=" << static_cast<int>(inst.bar_subop)
      << " bar_id=" << inst.bar_id
      << " bar_count=" << static_cast<int>(inst.bar_count)
      << " wait_bits=" << wait_barrier_bits << "\n";
  std::cerr << oss.str();
  --budget_left;
}

}  // namespace

SM::SM(unsigned int num_subcores, gpgpu_sim *gpu, simt_core_cluster *cluster,
       unsigned shader_id, unsigned tpc_id, const shader_core_config *config,
       const memory_config *mem_config, shader_core_stats *stats)
    : core_t(gpu, NULL, config->warp_size, config->n_thread_per_shader),
      m_sm_stats("SM_" + std::to_string(shader_id)),
      m_barriers(this, config->max_warps_per_shader, config->max_cta_per_core,
                 config->max_barriers_per_cta, config->warp_size) {
  m_sm_id = shader_id;
  m_tpc_id = tpc_id;
  m_num_subcores = num_subcores;
  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;
  // Jin: for concurrent kernels on a SM
  m_occupied_n_threads = 0;
  m_occupied_shmem = 0;
  m_occupied_regs = 0;
  m_occupied_ctas = 0;
  m_occupied_hwtid.reset();
  m_occupied_cta_to_hwtid.clear();
  m_active_warps = 0;
  m_dynamic_warp_id = 0;
  m_gpu = gpu;
  m_cluster = cluster;
  m_stats = stats;
  m_memory_config = mem_config;
  m_config = config;
  m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline = 0;
  m_pending_sync_waits.resize(config->max_warps_per_shader);
  m_partial_wait_arms.resize(config->max_warps_per_shader);
  m_partial_replay_progress.resize(config->max_warps_per_shader);
  m_partial_joined_regions.resize(config->max_warps_per_shader);
  m_wait_block_last_semantic_anchor_by_warp.resize(
      config->max_warps_per_shader);
  m_tma_open_contracts_per_warp.resize(config->max_warps_per_shader, 0);
  m_tma_outstanding_bindings_per_warp.resize(
      config->max_warps_per_shader, 0);
  m_tma_max_completed_transfer_uid_per_warp.resize(
      config->max_warps_per_shader, 0);
  if (config->sync_debug_enable) {
    m_sync_debug_print_budget = config->sync_debug_print_budget;
    m_sync_debug_skip_runtime_budget = config->sync_debug_skip_runtime_budget;
  }
  if (config->tma_debug_enable) {
    m_tma_debug_print_budget = config->tma_debug_print_budget;
  }
  if(config->is_interwarp_coalescing_enabled && is_using_interwarp_coal_warps_waiting_dep_counter()) {
    m_interwarp_coal_warps_waiting_dep_counter = new InterWarp_Coalescing_Waiting_Dep_Counters(config->max_warps_per_shader);
  }
}

SM::~SM() {
  debug_dump_tma_binding_counters();
  debug_dump_sync_counters();
  for (auto warp : m_physical_warp) {
    delete warp;
  }
  for(unsigned int i = 0; i < m_EX_MEM_reception_latches_per_subcore.size(); i++) {
    delete m_EX_MEM_reception_latches_per_subcore[i];
  }
  for (unsigned int i = 0; i < m_EX_TMA_reception_latches_per_subcore.size(); i++) {
    delete m_EX_TMA_reception_latches_per_subcore[i];
  }
  m_EX_WB_sm_shared_units_subcore_latches.clear();
  delete m_ldst_unit_shared_of_sm;
  delete m_tma_unit_shared_of_sm;
  if(m_config->is_dp_pipeline_shared_for_subcores) {
    delete m_shared_dp_unit;
  }
  delete m_icnt_L0s;
  delete m_icnt;
  delete m_scaling_coeffs;
  free(m_threadState);
  for(unsigned int i = 0; i < m_subcores.size(); i++) {
    delete m_subcores[i];
  }
  if(m_config->is_interwarp_coalescing_enabled && is_using_interwarp_coal_warps_waiting_dep_counter()) {
    delete m_interwarp_coal_warps_waiting_dep_counter;
  }
}

void SM::init() {
  create_logical_structures();
  create_memory_interfaces();

  if (get_gpu()->get_config().g_power_simulation_enabled) {
    m_scaling_coeffs = get_gpu()->get_scaling_coeffs();
  }
}

bool SM::is_using_interwarp_coal_warps_waiting_dep_counter() {
  bool res = (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC) ||
              (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID) ||
              (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC) ||
              (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID);
  return res;
}

void SM::enqueue_instruction_region_prewarm(kernel_info_t &kernel) {
  if (!m_config->is_instruction_region_prewarm_enabled) {
    return;
  }
  if (!m_config->is_trace_mode) {
    return;
  }
  if (kernel.function_unique_id == 0) {
    return;
  }
  std::vector<new_addr_type> region_bases = get_instruction_regions_to_prewarm(
      kernel.function_unique_id,
      m_config->instruction_region_prewarm_min_late_miss_count,
      m_config->instruction_region_prewarm_min_observed_warps,
      m_config->instruction_region_prewarm_max_regions);
  if (region_bases.empty()) {
    return;
  }
  unsigned int line_size = m_config->m_L1I_L1_half_C_cache_config.get_line_sz();
  unsigned int region_blocks = get_instruction_region_size_in_blocks();
  for (new_addr_type region_base : region_bases) {
    std::pair<unsigned int, new_addr_type> activation_key(
        kernel.function_unique_id, region_base);
    if (!m_activated_instruction_region_prewarm.insert(activation_key).second) {
      continue;
    }
    for (unsigned int i = 0; i < region_blocks; ++i) {
      m_pending_instruction_region_prewarm.push(
          std::make_pair(region_base + static_cast<new_addr_type>(i) * line_size,
                         kernel.function_unique_id));
    }
  }
}

void SM::issue_pending_instruction_region_prewarm() {
  if (!m_config->is_instruction_region_prewarm_enabled) {
    return;
  }
  if (m_pending_instruction_region_prewarm.empty()) {
    return;
  }
  L0_icnt *l0_icnt = static_cast<L0_icnt *>(m_icnt_L0s);
  for (unsigned int i = 0;
       i < m_config->instruction_region_prewarm_max_lines_per_cycle &&
       !m_pending_instruction_region_prewarm.empty();
       ++i) {
    std::pair<new_addr_type, unsigned int> entry =
        m_pending_instruction_region_prewarm.front();
    unsigned int subcore_id =
        m_next_instruction_region_prewarm_subcore % m_num_subcores;
    if (!l0_icnt->issue_instruction_prewarm(entry.first, entry.second,
                                            subcore_id)) {
      break;
    }
    m_next_instruction_region_prewarm_subcore =
        (m_next_instruction_region_prewarm_subcore + 1) % m_num_subcores;
    m_pending_instruction_region_prewarm.pop();
  }
}

void SM::num_cycles_to_stall_SM(unsigned int num_cycles) {
  for(unsigned int i = 0; i < m_subcores.size(); i++) {
    m_subcores[i]->num_cycles_to_stall(num_cycles);
  }
}

void SM::debug_log_sync_event(const std::string &message) {
  if (m_sync_debug_print_budget == 0) {
    return;
  }
  // cyc= prefix makes every sync event a point on a reconstructable timeline
  // (needed to correlate a not-ready wait with the later phase flip / tma
  // completion that releases it -- i.e. to size or diagnose a would-be fold
  // park and spot a barrier that never flips). See .plan/SPIN_WAIT_POSTPROCESS.md.
  std::cerr << "[SYNCDBG][SM" << m_sm_id << "] cyc="
            << get_current_gpu_cycle() << " " << message << std::endl;
  --m_sync_debug_print_budget;
}

// TMA per-event log. Two independent sinks:
//  1) [TMADBG] on stderr, gated by -tma_debug_enable (its own budget, separate
//     from -sync_debug_enable so TMA logs come without SYNCDBG/SMDBG noise).
//  2) The GPGPU-Sim trace system (-trace_enabled 1 -trace_components TMA),
//     which prints on stdout in the standard "GPGPU-Sim Cycle N: TMA - ..."
//     format and honors -trace_sampling_core. The TMA unit is a newly added
//     architecture component, so its activity is exposed as a first-class
//     trace stream alongside WARP_SCHEDULER / SCOREBOARD / INTERCONNECT.
void SM::debug_log_tma_event(const std::string &message) {
  if (DTRACE(TMA) &&
      (Trace::sampling_core == m_sm_id ||
       Trace::sampling_core == (unsigned int)-1)) {
    DPRINTF(TMA, "SM %u - %s\n", m_sm_id, message.c_str());
  }
  if (m_tma_debug_print_budget == 0) {
    return;
  }
  std::cerr << "[TMADBG][SM" << m_sm_id << "] cyc="
            << get_current_gpu_cycle() << " " << message << std::endl;
  --m_tma_debug_print_budget;
}

void SM::debug_log_tma_binding_event(const char *event,
                                     const std::string &message) {
  if (!m_config->tma_debug_enable || m_tma_debug_print_budget == 0) {
    return;
  }
  std::cerr << "[TMABIND][" << event << "][SM" << m_sm_id << "] cyc="
            << get_current_gpu_cycle() << " " << message << std::endl;
  --m_tma_debug_print_budget;
}

void SM::debug_log_wait_block_event(const char *event,
                                    const std::string &message) {
  if (!m_config->wait_block_debug ||
      m_wait_block_debug_print_budget == 0) {
    return;
  }
  std::cerr << "[WAITBLOCK][" << event << "][SM" << m_sm_id << "] cyc="
            << get_current_gpu_cycle() << " " << message << std::endl;
  --m_wait_block_debug_print_budget;
}

void SM::debug_dump_tma_binding_snapshot(const char *reason) const {
  uint64_t warp_open_contracts = 0;
  for (uint32_t count : m_tma_open_contracts_per_warp) {
    warp_open_contracts += count;
  }
  uint64_t barrier_open_contracts = 0;
  uint64_t barrier_outstanding = 0;
  for (const auto &entry : m_hopper_mbarriers) {
    if (entry.second.assignment_open) {
      ++barrier_open_contracts;
    }
    barrier_outstanding += entry.second.outstanding_tma_bindings;
  }
  uint64_t warp_outstanding = 0;
  for (uint32_t count : m_tma_outstanding_bindings_per_warp) {
    warp_outstanding += count;
  }
  const uint64_t bindings_in_flight =
      m_tma_bindings_created >= m_tma_bindings_completed
          ? m_tma_bindings_created - m_tma_bindings_completed
          : 0;
  const bool accounting_valid =
      m_tma_contracts_opened >= m_tma_contracts_fully_assigned &&
      m_tma_contracts_opened - m_tma_contracts_fully_assigned ==
          barrier_open_contracts &&
      m_tma_bindings_created >= m_tma_bindings_completed &&
      m_tma_assigned_credit_bytes >= m_tma_completed_credit_bytes &&
      warp_open_contracts == barrier_open_contracts &&
      bindings_in_flight == barrier_outstanding &&
      barrier_outstanding == warp_outstanding;
  const bool no_failures =
      m_tma_cross_barrier_applications == 0 &&
      m_tma_contract_mismatches == 0 && m_tma_stale_generation == 0 &&
      m_tma_launch_or_cta_mismatch == 0 &&
      m_tma_duplicate_completion == 0 && m_tma_orphan_cleanup == 0;
  std::cerr << "[TMABIND][snapshot] status="
            << (accounting_valid && no_failures ? "PASS" : "FAIL")
            << " reason=" << reason << " sm=" << m_sm_id
            << " launch_uid=" << m_tma_binding_launch_uid
            << " contracts_opened=" << m_tma_contracts_opened
            << " contracts_fully_assigned="
            << m_tma_contracts_fully_assigned
            << " open_contracts=" << barrier_open_contracts
            << " warp_open_contracts=" << warp_open_contracts
            << " max_open_contracts_per_warp="
            << m_tma_max_open_contracts_per_warp
            << " bindings_created=" << m_tma_bindings_created
            << " bindings_completed=" << m_tma_bindings_completed
            << " bindings_in_flight=" << bindings_in_flight
            << " barrier_outstanding=" << barrier_outstanding
            << " warp_outstanding=" << warp_outstanding
            << " assigned_credit_bytes=" << m_tma_assigned_credit_bytes
            << " completed_credit_bytes=" << m_tma_completed_credit_bytes
            << " uid_reorders=" << m_tma_uid_reorders
            << " cross_barrier_applications="
            << m_tma_cross_barrier_applications
            << " contract_mismatches=" << m_tma_contract_mismatches
            << " stale_generation=" << m_tma_stale_generation
            << " launch_or_cta_mismatch="
            << m_tma_launch_or_cta_mismatch
            << " duplicate_completion=" << m_tma_duplicate_completion
            << " orphan_cleanup=" << m_tma_orphan_cleanup << std::endl;
}

void SM::debug_dump_tma_binding_counters() const {
  if (m_tma_contracts_opened == 0 && m_tma_bindings_created == 0 &&
      m_tma_contract_mismatches == 0 &&
      m_tma_launch_or_cta_mismatch == 0 &&
      m_tma_duplicate_completion == 0 && m_tma_orphan_cleanup == 0) {
    return;
  }
  uint64_t warp_open_contracts = 0;
  for (uint32_t count : m_tma_open_contracts_per_warp) {
    warp_open_contracts += count;
  }
  uint64_t barrier_open_contracts = 0;
  uint64_t barrier_outstanding = 0;
  for (const auto &entry : m_hopper_mbarriers) {
    if (entry.second.assignment_open) {
      ++barrier_open_contracts;
    }
    barrier_outstanding += entry.second.outstanding_tma_bindings;
  }
  uint64_t warp_outstanding = 0;
  for (uint32_t count : m_tma_outstanding_bindings_per_warp) {
    warp_outstanding += count;
  }
  const uint64_t bindings_in_flight =
      m_tma_bindings_created >= m_tma_bindings_completed
          ? m_tma_bindings_created - m_tma_bindings_completed
          : 0;
  const bool pass =
      m_tma_contracts_opened == m_tma_contracts_fully_assigned &&
      warp_open_contracts == 0 && barrier_open_contracts == 0 &&
      m_tma_bindings_created == m_tma_bindings_completed &&
      bindings_in_flight == 0 && barrier_outstanding == 0 &&
      warp_outstanding == 0 &&
      m_tma_assigned_credit_bytes == m_tma_completed_credit_bytes &&
      m_tma_cross_barrier_applications == 0 &&
      m_tma_contract_mismatches == 0 && m_tma_stale_generation == 0 &&
      m_tma_launch_or_cta_mismatch == 0 &&
      m_tma_duplicate_completion == 0 && m_tma_orphan_cleanup == 0;
  std::cerr << "[TMABIND][FINAL] status=" << (pass ? "PASS" : "FAIL")
            << " sm=" << m_sm_id
            << " launch_uid=" << m_tma_binding_launch_uid
            << " contracts_opened=" << m_tma_contracts_opened
            << " contracts_fully_assigned="
            << m_tma_contracts_fully_assigned
            << " open_contracts=" << barrier_open_contracts
            << " warp_open_contracts=" << warp_open_contracts
            << " max_open_contracts_per_warp="
            << m_tma_max_open_contracts_per_warp
            << " bindings_created=" << m_tma_bindings_created
            << " bindings_completed=" << m_tma_bindings_completed
            << " bindings_in_flight=" << bindings_in_flight
            << " barrier_outstanding=" << barrier_outstanding
            << " warp_outstanding=" << warp_outstanding
            << " assigned_credit_bytes=" << m_tma_assigned_credit_bytes
            << " completed_credit_bytes=" << m_tma_completed_credit_bytes
            << " uid_reorders=" << m_tma_uid_reorders
            << " cross_barrier_applications="
            << m_tma_cross_barrier_applications
            << " contract_mismatches=" << m_tma_contract_mismatches
            << " stale_generation=" << m_tma_stale_generation
            << " launch_or_cta_mismatch="
            << m_tma_launch_or_cta_mismatch
            << " duplicate_completion=" << m_tma_duplicate_completion
            << " orphan_cleanup=" << m_tma_orphan_cleanup << std::endl;
}

void SM::debug_dump_sync_counters() const {
  if (m_sync_debug_sync_insts == 0 && m_sync_debug_tma_completions == 0 &&
      m_sync_debug_missing_runtime == 0) {
    return;
  }
  std::cerr << "[SYNCDBG][SM" << m_sm_id << "] summary"
            << " sync_insts=" << m_sync_debug_sync_insts
            << " exch=" << m_sync_debug_exch
            << " arrive=" << m_sync_debug_arrive
            << " arrive_expect_tx=" << m_sync_debug_arrive_expect_tx
            << " wait_pending=" << m_sync_debug_wait_pending
            << " wait_released=" << m_sync_debug_wait_released
            << " phase_flip=" << m_sync_debug_phase_flip
            << " tma_completions=" << m_sync_debug_tma_completions
            << " missing_runtime=" << m_sync_debug_missing_runtime
            << std::endl;
  // [FWD drain-idle 축3] mbarrier wait-duration histogram (per SM). Emitted only when the
  // dedicated gate is on; buckets = {0, 1-16, 17-64, 65-256, 257-1024, 1024+}. Observe-only.
  if (m_config->sync_wait_hist_instrument_enable) {
    std::cerr << "[SYNCWAITHIST][SM" << m_sm_id << "]"
              << " b0_eq0=" << m_sync_wait_cycle_hist[0]
              << " b1_1_16=" << m_sync_wait_cycle_hist[1]
              << " b2_17_64=" << m_sync_wait_cycle_hist[2]
              << " b3_65_256=" << m_sync_wait_cycle_hist[3]
              << " b4_257_1024=" << m_sync_wait_cycle_hist[4]
              << " b5_1024plus=" << m_sync_wait_cycle_hist[5]
              << std::endl;
  }
}

void SM::create_gpu_per_sm_stats(Element_stats &all_stats) {
  for(auto stat_name : all_stats.m_stats_name) {
    auto stat = all_stats.m_stats_map[stat_name];
    if(stat->get_allowed_type() == AllowedTypesStats::UNSIGNED_LONG_LONG) {
      m_sm_stats.add_unsigned_long_long_stat(stat->get_name(), stat->get_allowed_type(), stat->get_value(), stat->get_between_name_and_value(), stat->get_suffix(),
                                            stat->get_is_reset_allowed() ,stat->get_is_erase_after_gather_in_sm(), true);
    }else if(stat->get_allowed_type() == AllowedTypesStats::DOUBLE) {
      m_sm_stats.add_double_stat(stat->get_name(), stat->get_allowed_type(), stat->get_value(), stat->get_between_name_and_value(), stat->get_suffix(),
                                            stat->get_is_reset_allowed() ,stat->get_is_erase_after_gather_in_sm(), true);
    }else {
      std::cout << "Error. Stat type not allowed" << std::endl;
      stat->print(stdout);
      fflush(stdout);
      abort();
    }
    
  }
}

void SM::reset_cycless_access_history() {
  m_ldst_unit_shared_of_sm->reset_coalescingHistory();
}

void SM::gather_gpu_per_sm_stats(Element_stats &all_stats, coalescingStatsAcrossSms& coal_stats_l1d, coalescingStatsAcrossSms& coal_stats_const, coalescingStatsAcrossSms& coal_stats_sharedmem) {
  for(auto stat_name : all_stats.m_stats_name) {
    auto stat_value = m_sm_stats.m_stats_map[stat_name]->get_value();
    all_stats.m_stats_map[stat_name]->increment_with_integer(stat_value);
    if(all_stats.m_stats_map[stat_name]->get_is_erase_after_gather_in_sm()) {
      m_sm_stats.m_stats_map[stat_name]->reset();
    }
  }
  coal_stats_l1d.addStats(m_ldst_unit_shared_of_sm->get_coalescingStatPerSm_l1d());
  coal_stats_const.addStats(m_ldst_unit_shared_of_sm->get_coalescingStatPerSm_const());
  coal_stats_sharedmem.addStats(m_ldst_unit_shared_of_sm->get_coalescingStatPerSm_sharedmem());
}

void SM::gather_gpu_per_sm_single_stat(Element_stats &all_stats, std::string stat_name) {
  auto stat_value = m_sm_stats.m_stats_map[stat_name]->get_value();
  all_stats.m_stats_map[stat_name]->increment_with_integer(stat_value);
  if(all_stats.m_stats_map[stat_name]->get_is_erase_after_gather_in_sm()) {
    m_sm_stats.m_stats_map[stat_name]->reset();
  }
}

void SM::increment_sm_stat_by_integer(std::string stat_name, int val_to_increment) {
  m_sm_stats.m_stats_map[stat_name]->increment_with_integer(val_to_increment);
}

void SM::gather_stall_reason_align_stats(Subcore::StallPopulationTotals &dst) const {
  for (auto subcore : m_subcores) {
    subcore->gather_stall_reason_align_stats(dst);
  }
}

void SM::reset_stall_reason_align_stats() {
  for (auto subcore : m_subcores) {
    subcore->reset_stall_reason_align_stats();
  }
}

void SM::record_wait_block_timing_issue(unsigned int warp_id,
                                        const warp_inst_t *inst) {
  if (!m_config->wait_block_late_arrival_instrument_enable ||
      inst == nullptr ||
      warp_id >= m_wait_block_last_semantic_anchor_by_warp.size() ||
      !inst->has_extra_trace_instruction_info()) {
    return;
  }
  const std::string opcode =
      inst->get_extra_trace_instruction_info().get_op_code();
  if (!is_wait_block_timing_semantic_opcode(opcode)) {
    return;
  }

  WaitBlockSemanticAnchor &anchor =
      m_wait_block_last_semantic_anchor_by_warp[warp_id];
  anchor.valid = true;
  anchor.cycle = get_current_gpu_cycle();
  anchor.trace_kernel_id = inst->trace_kernel_id;
  anchor.unique_function_id = inst->unique_function_id;
  anchor.cta_warp_id = inst->trace_cta_warp_id;
  anchor.pc = inst->pc;
  anchor.source_index = inst->trace_instruction_index;
  anchor.opcode = opcode;
}

void SM::cycle() {
  if (!isactive() && get_not_completed() == 0) return;

  m_stats->shader_cycles[m_sm_id]++;

  // MOD. Begin. Custom Stats
  m_stats->shader_cycles_per_kernel[m_stats->m_current_kernel_pos][m_sm_id]++;
  m_stats->shader_active_warps_per_kernel[m_stats->m_current_kernel_pos]
                                         [m_sm_id] += m_active_warps;
  m_stats->shader_maximum_theoretical_warps_per_kernel
      [m_stats->m_current_kernel_pos][m_sm_id] +=
      m_config->max_warps_per_shader;
  // MOD. End. Custom Stats

  L0_icnt *l0_icnt = static_cast<L0_icnt *>(m_icnt_L0s);
  if (m_config->is_instruction_region_prewarm_enabled &&
      m_pending_instruction_region_prewarm.empty() && m_kernel != nullptr) {
    enqueue_instruction_region_prewarm(*m_kernel);
  }
  issue_pending_instruction_region_prewarm();
  l0_icnt->cycle();
  m_L1I_L1_half_C_cache->cycle();

  if(m_config->is_dp_pipeline_shared_for_subcores) {
    m_shared_dp_unit->cycle();
  }

  m_ldst_unit_shared_of_sm->cycle();
  m_tma_unit_shared_of_sm->cycle();

  if(m_config->is_interwarp_coalescing_enabled && 
        ((m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC) ||
        (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID))) {
    m_interwarp_coal_warps_waiting_dep_counter->clear();
  }
  
  for (auto subcore : m_subcores) {
    subcore->cycle();
  }

  if (m_config->producer_tail_instrument_enable) {
    account_producer_tail_cycle();
  }

  // [overlap probe] SM-level tensor-vs-SFU pipe overlap this cycle (read-only, gated). OR the busy flags
  // across the 4 subcores, then classify the cycle into both / tensor-only / sfu-only / neither. HW
  // overlaps tensor(46%)+MUFU(48%) so 'both' should be large on HW; if the sim shows mostly tensor-only
  // + sfu-only, the pipes are being run SERIALLY (the suspected root of the 1.56x issue-density gap).
  // Compare 'both' fraction against HW min(tensor%,xu%) which is the max possible overlap.
  if (m_config->overlap_instrument_enable) {
    bool any_tensor_busy = false, any_sfu_busy = false;
    for (auto subcore : m_subcores) {
      if (subcore->is_tensor_pipe_busy_this_cycle()) any_tensor_busy = true;
      if (subcore->is_sfu_pipe_busy_this_cycle())    any_sfu_busy = true;
    }
    const char *key;
    if (any_tensor_busy && any_sfu_busy)      key = "total_num_cycles_overlap_tensor_and_sfu";
    else if (any_tensor_busy)                 key = "total_num_cycles_overlap_tensor_only";
    else if (any_sfu_busy)                    key = "total_num_cycles_overlap_sfu_only";
    else                                      key = "total_num_cycles_overlap_neither";
    auto it = m_sm_stats.m_stats_map.find(key);
    if (it != m_sm_stats.m_stats_map.end()) it->second->increment_with_integer(1);
  }
  // [WGMMA Opt6 Step-0] (V) SM-level idle accounting. A subcore being blocked on the
  // tensor pipe is only a real cycle loss if NO subcore on this SM issued this cycle.
  if(m_config->wgmma_step0_instrument_enable || m_config->l1i_frontend_step0_instrument_enable) {
    bool any_subcore_issued = false;
    bool any_subcore_tensor_only_block = false;
    bool any_subcore_frontend_block = false;
    bool any_subcore_fu_occupied_tensor_block = false;
    unsigned int sm_idle_reason_or = 0;  // OR of all subcores' non-issue reason masks
    // [FWD drain-idle 축1] SM-level mutually-exclusive sole-block: pick the highest recover-value
    // reason across the 4 subcores (only meaningful on SM-idle cycles). DRAINED is the floor and
    // only wins if NO subcore had a recoverable (ONLY_*/MULTI) reason.
    int sm_best_sole_rank = -1;                 // -1 = nothing recoverable seen
    Subcore::Step0SoleBlock sm_best_sole = Subcore::SB_DRAINED;
    bool any_subcore_non_drained_block = false; // >=1 subcore had a valid head that failed to issue
    for (auto subcore : m_subcores) {
      if (subcore->step0_issued_this_cycle()) any_subcore_issued = true;
      if (subcore->step0_blocked_by_tensor_only_this_cycle()) any_subcore_tensor_only_block = true;
      if (subcore->step0_blocked_by_frontend_sbwait_this_cycle()) any_subcore_frontend_block = true;
      if (subcore->step0_blocked_by_fu_occupied_tensor_this_cycle()) any_subcore_fu_occupied_tensor_block = true;
      sm_idle_reason_or |= subcore->step0_reason_mask_this_cycle();
      // rank map (mirror of subcore.cc): WAIT_BARRIER>TENSOR>STALL_COUNT>FU_NONTENSOR>NEXT_STAGE>L1C>OTHER>MULTI
      Subcore::Step0SoleBlock sb = subcore->step0_sole_block_this_cycle();
      int rank = -1;
      switch (sb) {
        case Subcore::SB_ONLY_WAIT_BARRIER: rank = 7; break;
        case Subcore::SB_ONLY_TENSOR:       rank = 6; break;
        case Subcore::SB_ONLY_STALL_COUNT:  rank = 5; break;
        case Subcore::SB_ONLY_FU_NONTENSOR: rank = 4; break;
        case Subcore::SB_ONLY_NEXT_STAGE:   rank = 3; break;
        case Subcore::SB_ONLY_L1C:          rank = 2; break;
        case Subcore::SB_ONLY_OTHER:        rank = 1; break;
        case Subcore::SB_MULTI:             rank = 0; break;
        default:                            rank = -1; break;  // SB_ISSUED / SB_DRAINED
      }
      if (rank >= 0) {
        any_subcore_non_drained_block = true;
        if (rank > sm_best_sole_rank) { sm_best_sole_rank = rank; sm_best_sole = sb; }
      }
    }
    // [CTA-imbalance diag] tensor-FU-busy stall (NCU `mma`), counted EVERY cycle (not just
    // SM-idle: another subcore may still issue while this WGMMA waits on the tensor FU) and
    // attributed to each resident CTA slot. This is the tensor-attributable stall the trace
    // model exposes (there is no separate consumer-side warpgroup_arrive). Exact for 1-CTA/SM
    // (fwd); upper bound for bwd.
    if (m_config->wgmma_step0_instrument_enable && any_subcore_fu_occupied_tensor_block) {
      for (unsigned s = 0; s < MAX_CTA_PER_SHADER; s++) {
        if (m_cta_status[s] > 0) m_fu_occupied_tensor_cyc_by_cta_slot[s]++;
      }
    }
    if (!any_subcore_issued) {
      m_sm_stats.m_stats_map["total_num_cycles_sm_all_subcores_idle"]->increment_with_integer(1);
      // [CTA stall breakdown] Per-CTA drain-idle: attribute every true SM-idle cycle (no
      // subcore issued) to each resident CTA slot, and separately its ibuffer-empty subset
      // (trace-drained head). This is the drain-idle that dominates fwd's finish variance
      // (1.39x); correlating it per-CTA with tensor_ops tells fwd tensor-vs-nontensor apart.
      // Gated separately so it can be toggled without the tensor counters.
      if (m_config->cta_stall_breakdown_instrument_enable) {
        bool idle_ibuffer_empty = (sm_idle_reason_or & Subcore::STEP0_R_NV_IBUFFER_EMPTY) != 0;
        for (unsigned s = 0; s < MAX_CTA_PER_SHADER; s++) {
          if (m_cta_status[s] > 0) {
            m_sm_idle_cyc_by_cta_slot[s]++;
            if (idle_ibuffer_empty) m_sm_idle_ibuffer_empty_cyc_by_cta_slot[s]++;
          }
        }
      }
      // WGMMA-specific: SM-idle cycles with >=1 subcore blocked only by the tensor lockout.
      if (m_config->wgmma_step0_instrument_enable && any_subcore_tensor_only_block) {
        m_sm_stats.m_stats_map["total_num_cycles_sm_idle_all_blocked_by_tensor"]->increment_with_integer(1);
        // [CTA-imbalance diag] attribute this tensor-caused idle cycle to every
        // CTA slot resident this cycle (exact for 1-CTA/SM fwd; upper bound for
        // multi-CTA bwd). Correlates per-CTA tensor-idle with density in [CTAFIN].
        for (unsigned s = 0; s < MAX_CTA_PER_SHADER; s++) {
          if (m_cta_status[s] > 0) m_sm_idle_tensor_cyc_by_cta_slot[s]++;
        }
      }
      // Frontend-specific: SM-idle cycles with >=1 subcore blocked on the L1I stream-buffer frontend.
      if (m_config->l1i_frontend_step0_instrument_enable && any_subcore_frontend_block) {
        m_sm_stats.m_stats_map["total_num_cycles_sm_idle_blocked_by_frontend_sbwait"]->increment_with_integer(1);
      }
      // [FWD drain-idle 축1] mutually-exclusive SM-idle partition: attribute THIS idle cycle to
      // exactly one reason (highest recover-value across subcores; DRAINED floor if none). These
      // counters sum EXACTLY to sm_all_subcores_idle (unlike the overlapping sm_idle_reason_* below).
      // Also attribute the two recoverable-vs-floor signals per-CTA-slot for [CTAFIN].
      if (m_config->wgmma_step0_instrument_enable) {
        Subcore::Step0SoleBlock part =
            any_subcore_non_drained_block ? sm_best_sole : Subcore::SB_DRAINED;
        const char *pkey = nullptr;
        switch (part) {
          case Subcore::SB_DRAINED:           pkey = "total_num_cycles_sm_idle_partition_drained"; break;
          case Subcore::SB_ONLY_WAIT_BARRIER: pkey = "total_num_cycles_sm_idle_partition_only_wait_barrier"; break;
          case Subcore::SB_ONLY_TENSOR:       pkey = "total_num_cycles_sm_idle_partition_only_tensor"; break;
          case Subcore::SB_ONLY_STALL_COUNT:  pkey = "total_num_cycles_sm_idle_partition_only_stall_count"; break;
          case Subcore::SB_ONLY_FU_NONTENSOR: pkey = "total_num_cycles_sm_idle_partition_only_fu_nontensor"; break;
          case Subcore::SB_ONLY_NEXT_STAGE:   pkey = "total_num_cycles_sm_idle_partition_only_next_stage"; break;
          case Subcore::SB_ONLY_L1C:          pkey = "total_num_cycles_sm_idle_partition_only_l1c"; break;
          case Subcore::SB_ONLY_OTHER:        pkey = "total_num_cycles_sm_idle_partition_only_other"; break;
          default:                            pkey = "total_num_cycles_sm_idle_partition_multi"; break;
        }
        m_sm_stats.m_stats_map[pkey]->increment_with_integer(1);
        // Independent ceiling (mirrors sm_idle_all_blocked_by_tensor): >=1 subcore blocked ONLY by
        // wait_barrier. Used to cross-check the partition's only_wait_barrier column.
        bool any_wb_only = false;
        for (auto subcore : m_subcores)
          if (subcore->step0_sole_block_this_cycle() == Subcore::SB_ONLY_WAIT_BARRIER) any_wb_only = true;
        if (any_wb_only)
          m_sm_stats.m_stats_map["total_num_cycles_sm_idle_all_blocked_by_wait_barrier"]->increment_with_integer(1);
        // Per-CTA: recoverable (only_wait_barrier) and floor (drained) attributed to resident slots.
        bool part_is_wb   = (part == Subcore::SB_ONLY_WAIT_BARRIER);
        bool part_is_drain = (part == Subcore::SB_DRAINED);
        if (part_is_wb || part_is_drain) {
          for (unsigned s = 0; s < MAX_CTA_PER_SHADER; s++) {
            if (m_cta_status[s] > 0) {
              if (part_is_wb)    m_wait_barrier_only_cyc_by_cta_slot[s]++;
              if (part_is_drain) m_drained_cyc_by_cta_slot[s]++;
            }
          }
        }
      }
      // FULL SM-idle decomposition: for each reason, count SM-idle cycles where >=1 subcore
      // was blocked for that reason. Reasons overlap (sum can exceed sm_all_subcores_idle),
      // so read as relative intensity — this attributes ALL true SM-idle in one run.
      if (sm_idle_reason_or & Subcore::STEP0_R_NEXT_STAGE)         m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_next_stage"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_ISSUE_PORT_BUSY)    m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_issue_port_busy"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_NO_VALID_FRONTEND)  m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_no_valid_frontend"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_NO_VALID_SBWAIT)    m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_no_valid_sbwait"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_NO_VALID_OTHER)     m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_no_valid_other"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_NV_IBUFFER_EMPTY)   m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_nv_ibuffer_empty"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_NV_IBUF_FETCH_INFLIGHT)   m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_nv_ibuf_fetch_inflight"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_NV_IBUF_FETCH_NOT_ISSUED) m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_nv_ibuf_fetch_not_issued"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_NV_DECODE_PENDING)  m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_nv_decode_pending"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_NV_L0I_RESP_READY)  m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_nv_l0i_resp_ready"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_NV_UNKNOWN)         m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_nv_unknown"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_FU_OCCUPIED)        m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_fu_occupied"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_FU_OCCUPIED_TENSOR) m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_fu_occupied_tensor"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_INST_BARRIER)       m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_inst_barrier"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_WAIT_BARRIER)       m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_wait_barrier"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_TMA_FLUSH)          m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_tma_flush"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_STALL_COUNT)        m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_stall_count"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_SCOREBOARD)         m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_scoreboard"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_L1C)                m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_l1c"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_RESULT_QUEUE_FULL)  m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_result_queue_full"]->increment_with_integer(1);
      if (sm_idle_reason_or & Subcore::STEP0_R_YIELD)              m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_yield"]->increment_with_integer(1);
      // Cycles where the SM was idle and NO subcore reported any reason (pure structural idle,
      // e.g. all warps done/sleeping) — should be small; catches unattributed idle.
      if (sm_idle_reason_or == 0)                                 m_sm_stats.m_stats_map["total_num_cycles_sm_idle_reason_none"]->increment_with_integer(1);
    }
  }

  consume_pending_wait_barrier_actions(m_pending_wait_barrier_increments);
  consume_pending_wait_barrier_actions(m_pending_wait_barrier_decrements);
  if(m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline > 0) {
    m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline--;
  }
}

void SM::set_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline(unsigned int num_cycles) {
  m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline = num_cycles;
}

bool SM::producer_tail_active_for_warp(unsigned wid) const {
  if (!m_config->producer_tail_instrument_enable ||
      wid >= MAX_WARP_PER_SHADER || m_physical_warp[wid] == nullptr ||
      !m_warp_is_producer[wid] || m_warp_drain_cycle[wid] != 0) {
    return false;
  }

  const unsigned cta_slot = m_physical_warp[wid]->get_cta_id();
  if (cta_slot >= MAX_CTA_PER_SHADER || m_cta_status[cta_slot] == 0) {
    return false;
  }

  bool saw_consumer = false;
  for (unsigned w = 0; w < MAX_WARP_PER_SHADER; ++w) {
    if (m_physical_warp[w] == nullptr ||
        m_physical_warp[w]->get_cta_id() != cta_slot ||
        !m_warp_is_consumer[w]) {
      continue;
    }
    saw_consumer = true;
    if (m_warp_drain_cycle[w] == 0) {
      return false;
    }
  }
  return saw_consumer;
}

void SM::record_producer_tail_issue(unsigned wid, const warp_inst_t *inst) {
  if (!producer_tail_active_for_warp(wid) || inst == nullptr) {
    return;
  }
  const unsigned cta_slot = m_physical_warp[wid]->get_cta_id();
  ProducerTailCounters &tail = m_producer_tail_by_cta_slot[cta_slot];
  tail.issued++;
  tail.last_producer_issue_cycle = get_current_gpu_cycle();

  const bool is_phasechk =
      inst->sync_kind == SyncInstructionKind::PHASECHK ||
      inst->sync_kind == SyncInstructionKind::TRYWAIT;
  std::string opcode;
  if (inst->has_extra_trace_instruction_info()) {
    opcode = inst->get_extra_trace_instruction_info().get_op_code();
  }
  if (is_phasechk) {
    tail.issued_phasechk++;
  } else if (opcode.rfind("NANOSLEEP", 0) == 0) {
    tail.issued_nanosleep++;
  } else if (inst->op == BRANCH_OP) {
    tail.issued_branch++;
  } else if (inst->is_tma_op()) {
    tail.issued_tma++;
  } else {
    tail.issued_other++;
  }
}

void SM::record_producer_tail_phasechk_result(unsigned wid, bool ready) {
  if (!producer_tail_active_for_warp(wid)) {
    return;
  }
  const unsigned cta_slot = m_physical_warp[wid]->get_cta_id();
  ProducerTailCounters &tail = m_producer_tail_by_cta_slot[cta_slot];
  if (ready) {
    tail.phasechk_ready++;
  } else {
    tail.phasechk_not_ready++;
  }
}

// [Producer-spin ready-retry probe] whole-producer-lifetime opcode accounting.
// Unlike the tail probe, this fires for every producer-role warp instruction
// regardless of consumer-drain state, so it captures bwd main-loop spin too.
void SM::record_producer_spin_issue(unsigned wid, const warp_inst_t *inst) {
  if (!m_config->producer_spin_ready_instrument_enable || inst == nullptr ||
      wid >= MAX_WARP_PER_SHADER || m_physical_warp[wid] == nullptr ||
      !m_warp_is_producer[wid]) {
    return;
  }
  const unsigned cta_slot = m_physical_warp[wid]->get_cta_id();
  if (cta_slot >= MAX_CTA_PER_SHADER) {
    return;
  }
  ProducerSpinCounters &spin = m_producer_spin_by_cta_slot[cta_slot];
  const bool is_phasechk =
      inst->sync_kind == SyncInstructionKind::PHASECHK ||
      inst->sync_kind == SyncInstructionKind::TRYWAIT;
  std::string opcode;
  if (inst->has_extra_trace_instruction_info()) {
    opcode = inst->get_extra_trace_instruction_info().get_op_code();
  }
  if (is_phasechk) {
    spin.issued_phasechk++;
  } else if (opcode.rfind("NANOSLEEP", 0) == 0) {
    spin.issued_nanosleep++;
  } else if (inst->op == BRANCH_OP) {
    spin.issued_branch++;
  } else {
    spin.issued_other++;
  }
}

// [Producer-spin ready-retry probe] sample the existing mbarrier result at each
// producer PHASECHK/TRYWAIT. "ready" means the modeled Hopper mbarrier is already
// satisfied yet the captured trace still issued this poll. Consecutive ready
// tests on the same warp (a ready-run) beyond the first are redundant re-polls.
void SM::record_producer_spin_phasechk_result(unsigned wid, bool ready) {
  if (!m_config->producer_spin_ready_instrument_enable ||
      wid >= MAX_WARP_PER_SHADER || m_physical_warp[wid] == nullptr ||
      !m_warp_is_producer[wid]) {
    return;
  }
  const unsigned cta_slot = m_physical_warp[wid]->get_cta_id();
  if (cta_slot >= MAX_CTA_PER_SHADER) {
    return;
  }
  ProducerSpinCounters &spin = m_producer_spin_by_cta_slot[cta_slot];
  spin.phasechk_total++;
  if (ready) {
    spin.phasechk_ready++;
    // A ready test that continues an already-open ready-run is a redundant
    // re-poll (the barrier was already satisfied on the prior test).
    if (m_producer_spin_in_ready_run[wid]) {
      spin.ready_run_retries++;
    }
    m_producer_spin_in_ready_run[wid] = true;
  } else {
    spin.phasechk_not_ready++;
    m_producer_spin_in_ready_run[wid] = false;
  }
  m_producer_spin_prev_ready[wid] = ready;
}

// [Spin-wait observe] Observe-only. Called on every PHASECHK/TRYWAIT that maps to
// a spin_wait_sidecar primary_wait_pc. Measures, PER REGION, how long the modeled
// mbarrier stays not-ready after the first not-ready poll of an episode (= the
// cycles the warp WOULD be parked under fold), buckets them, and runs a
// deadlock watchdog: if a single would-be park crosses the threshold, it emits a
// [SPINWAITDEADLOCK] line immediately (once per episode) with the barrier state so
// a never-flipping / pathologically-long wait is caught in the SAME 12h run.
// It never blocks, never advances the trace, and never touches
// m_pending_sync_waits. See .plan/SPIN_WAIT_POSTPROCESS.md.
void SM::record_spin_wait_observe(unsigned wid, uint64_t wait_pc,
                                  uint64_t barrier_addr, unsigned cta_id,
                                  bool ready) {
  if (!m_config->spin_wait_observe_enable || wid >= MAX_WARP_PER_SHADER) {
    return;
  }
  SpinWaitObserveState &st = m_spin_wait_observe_by_warp[wid];
  SpinWaitRegionObserve &reg = m_spin_wait_observe_by_region[wait_pc];
  reg.primary_wait_pc = wait_pc;
  const unsigned long long now = get_current_gpu_cycle();
  const unsigned long long threshold = kSpinWaitObserveWatchdogCycles;
  if (!ready) {
    // Open a new episode on the first not-ready poll for this warp+region, or
    // keep the earliest not-ready cycle if an episode for this region is open.
    if (!st.episode_open || st.active_wait_pc != wait_pc) {
      st.episode_open = true;
      st.active_wait_pc = wait_pc;
      st.active_barrier_addr = barrier_addr;
      st.active_cta = cta_id;
      st.first_not_ready_cycle = now;
      st.watchdog_fired = false;
    } else {
      // Episode continues: check the deadlock watchdog once per episode.
      const unsigned long long parked = now - st.first_not_ready_cycle;
      if (!st.watchdog_fired && parked >= threshold) {
        st.watchdog_fired = true;
        reg.watchdog_hits++;
        fprintf(stdout,
                "[SPINWAITDEADLOCK] sm=%u cta=%u warp=%u wait_pc=0x%llx "
                "barrier=0x%llx parked_cyc=%llu first_not_ready_cyc=%llu "
                "now=%llu threshold=%llu (would-be fold park exceeded "
                "threshold: deadlock candidate)\n",
                m_sm_id, st.active_cta, wid,
                (unsigned long long)wait_pc,
                (unsigned long long)st.active_barrier_addr, parked,
                st.first_not_ready_cycle, now, threshold);
      }
    }
    return;
  }
  // ready: close the open episode iff it is waiting on THIS barrier (the barrier
  // flip is the real spin-loop exit). Keying close on the BARRIER, not the
  // wait_pc/wait_state, is essential: a producer leaves its spin loop as soon as
  // the barrier flips, and on the next poll the phase parity has changed, so the
  // same wait_pc is re-polled with a DIFFERENT wait_state and never matched the
  // open episode -- leaving it falsely "open" even though the barrier flipped
  // normally (K5 SM27 barrier 0x16c00 flips at cyc 10049/30658 yet produced 1371
  // false [SPINWAITOPEN] under the old wait_pc-keyed close). Closing on
  // "same barrier observed ready" tracks the real exit and equals the would-be
  // fold park (park ends when the barrier is ready).
  if (st.episode_open && st.active_barrier_addr == barrier_addr) {
    SpinWaitRegionObserve &open_reg =
        m_spin_wait_observe_by_region[st.active_wait_pc];
    open_reg.primary_wait_pc = st.active_wait_pc;
    const unsigned long long delta =
        now >= st.first_not_ready_cycle ? now - st.first_not_ready_cycle : 0;
    open_reg.episodes++;
    open_reg.park_cycles_total += delta;
    if (delta > open_reg.park_cycles_max) {
      open_reg.park_cycles_max = delta;
    }
    const unsigned b = (delta == 0)     ? 0
                     : (delta <= 16)    ? 1
                     : (delta <= 64)    ? 2
                     : (delta <= 256)   ? 3
                     : (delta <= 1024)  ? 4
                                        : 5;
    open_reg.park_hist[b]++;
    st.episode_open = false;
    st.watchdog_fired = false;
  } else if (!st.episode_open) {
    // ready with no open episode: the barrier was already satisfied at the first
    // poll of this region (would-be park = 0).
    reg.immediate_ready_polls++;
  }
  // (ready while an episode is open on a DIFFERENT barrier: leave that episode
  // open; it closes when its own barrier flips.)
}

void SM::print_spin_wait_observe_summary() const {
  if (!m_config->spin_wait_observe_enable) {
    return;
  }
  // Per-region would-be-park summary: attributes park latency to a specific spin
  // region so fold can be applied/diagnosed region by region.
  for (const auto &kv : m_spin_wait_observe_by_region) {
    const SpinWaitRegionObserve &r = kv.second;
    const double avg = r.episodes ? (double)r.park_cycles_total / r.episodes : 0.0;
    fprintf(stdout,
            "[SPINWAITOBS] sm=%u wait_pc=0x%llx episodes=%llu avg_park=%.1f "
            "max_park=%llu total_park=%llu immediate_ready=%llu watchdog_hits=%llu "
            "hist[0/1-16/17-64/65-256/257-1024/>1024]=%llu/%llu/%llu/%llu/%llu/%llu\n",
            m_sm_id, (unsigned long long)r.primary_wait_pc, r.episodes, avg,
            r.park_cycles_max, r.park_cycles_total, r.immediate_ready_polls,
            r.watchdog_hits, r.park_hist[0], r.park_hist[1], r.park_hist[2],
            r.park_hist[3], r.park_hist[4], r.park_hist[5]);
  }
  // Residual open episodes at print time: a producer wait whose modeled barrier
  // never flipped for the remainder of the run -- the strongest deadlock signal.
  for (unsigned w = 0; w < MAX_WARP_PER_SHADER; ++w) {
    const SpinWaitObserveState &st = m_spin_wait_observe_by_warp[w];
    if (st.episode_open) {
      fprintf(stdout,
              "[SPINWAITOPEN] sm=%u cta=%u warp=%u wait_pc=0x%llx barrier=0x%llx "
              "first_not_ready_cyc=%llu (episode never released: never-flip "
              "deadlock candidate)\n",
              m_sm_id, st.active_cta, w, (unsigned long long)st.active_wait_pc,
              (unsigned long long)st.active_barrier_addr,
              st.first_not_ready_cycle);
    }
  }
}

void SM::account_producer_tail_cycle() {
  const unsigned long long now = get_current_gpu_cycle();
  for (unsigned cta_slot = 0; cta_slot < MAX_CTA_PER_SHADER; ++cta_slot) {
    if (m_cta_status[cta_slot] == 0) {
      continue;
    }

    bool active_tail = false;
    bool producer_subcore_issued = false;
    unsigned int producer_reason_mask = 0;
    for (unsigned w = 0; w < MAX_WARP_PER_SHADER; ++w) {
      if (!producer_tail_active_for_warp(w)) {
        continue;
      }
      if (m_physical_warp[w]->get_cta_id() != cta_slot) {
        continue;
      }
      active_tail = true;
      const unsigned subcore_id =
          translate_warp_id_of_sm_to_subcore(w, get_num_subcores());
      assert(subcore_id < m_subcores.size());
      producer_subcore_issued |=
          m_subcores[subcore_id]->step0_issued_this_cycle();
      producer_reason_mask |=
          m_subcores[subcore_id]->step0_reason_mask_this_cycle();
    }
    if (!active_tail) {
      continue;
    }

    ProducerTailCounters &tail = m_producer_tail_by_cta_slot[cta_slot];
    if (tail.start_cycle == 0) {
      tail.start_cycle = now;
    }
    tail.cycles++;
    if (tail.last_producer_issue_cycle == now) {
      continue;
    }

    tail.stall_cycles++;
    bool attributed = false;
    if (producer_subcore_issued) {
      tail.stall_not_selected++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_NEXT_STAGE) {
      tail.stall_next_stage++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_ISSUE_PORT_BUSY) {
      tail.stall_issue_port++;
      attributed = true;
    }
    if (producer_reason_mask &
        (Subcore::STEP0_R_NO_VALID_FRONTEND |
         Subcore::STEP0_R_NO_VALID_OTHER)) {
      tail.stall_no_valid++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_WAIT_BARRIER) {
      tail.stall_wait_barrier++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_STALL_COUNT) {
      tail.stall_stall_count++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_FU_OCCUPIED) {
      tail.stall_fu_occupied++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_INST_BARRIER) {
      tail.stall_inst_barrier++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_TMA_FLUSH) {
      tail.stall_tma_flush++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_SCOREBOARD) {
      tail.stall_scoreboard++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_L1C) {
      tail.stall_l1c++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_RESULT_QUEUE_FULL) {
      tail.stall_result_queue++;
      attributed = true;
    }
    if (producer_reason_mask & Subcore::STEP0_R_YIELD) {
      tail.stall_yield++;
      attributed = true;
    }
    if (!attributed) {
      tail.stall_unattributed++;
    }
  }
}

bool SM::can_send_inst_from_subcore_to_sm_shared_pipeline() const {
  return (m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline == 0);
}

void SM::consume_pending_wait_barrier_actions(std::stack<Wait_Barrier_Entry_Modifier> &actions) {
  while (!actions.empty()) {
    Wait_Barrier_Entry_Modifier current_action = actions.top();
    m_physical_warp[current_action.sm_warp_id]->get_dependency_state()->action_over_wait_barrier(&current_action);
    actions.pop();
  }
}

void SM::add_pending_wait_barrier_decrement(warp_inst_t *inst,
                                            Wait_Barrier_Type barrier_type, unsigned int barrier_id) {                                    
  m_pending_wait_barrier_decrements.push(Wait_Barrier_Entry_Modifier(
    inst->warp_id(), barrier_id, barrier_type, Wait_Barrier_Action::DECREASE_COUNTER, inst->pc));
}

void SM::add_pending_wait_barrier_increment(warp_inst_t *inst,
                                            Wait_Barrier_Type barrier_type, unsigned int barrier_id) {                                    
  // [wait_barrier-kind probe] tag the SB with the op-class arming it, so a later stall on this SB can
  // be attributed to WGMMA-result (recoverable by de-phasing) vs TMA (producer floor) vs other.
  Wait_Barrier_Armed_Op armed_op = WB_ARM_OTHER;
  if (inst->is_tensor_core_op())      armed_op = WB_ARM_TENSOR;
  else if (inst->is_tma_op())         armed_op = WB_ARM_TMA;
  m_pending_wait_barrier_increments.push(Wait_Barrier_Entry_Modifier(
    inst->warp_id(), barrier_id, barrier_type, Wait_Barrier_Action::INCREASE_COUNTER, inst->pc, armed_op));
}


void SM::instruction_retirement(warp_inst_t *instruction) {
  unsigned int warp_id = instruction->warp_id();
  // if(m_sm_id == 0 && warp_id == 0) { {
  //   std::cout << "WB. SM: " << m_sm_id << ". Subcore: " << instruction->get_subcore_id() << ". Warp_ID: " << warp_id << ". PC: " << std::hex << instruction->pc << std::dec << ". Cycle: " << get_current_gpu_cycle() << std::endl;
  //   fflush(stdout);
  // }
  bool use_traditional_scoreboarding = !m_physical_warp[warp_id]->get_kernel_info()->is_captured_from_binary;
  if (use_traditional_scoreboarding ||
    m_config->is_remodeling_scoreboarding_enabled ||
    !m_config->is_trace_mode) {
    if ( (use_traditional_scoreboarding && m_config->is_trace_mode) || (m_config->is_trace_mode && m_config->is_remodeling_scoreboarding_enabled) ) {
      if ((m_scoreboard_WAR->getMode() ==
           scoreboard_reads_mode::RELEASE_AT_WB)) {
        m_scoreboard_WAR->releaseRegisters_remodeling(instruction);
      }
      m_scoreboard->releaseRegisters_remodeling(instruction);
    } else {
      if ((m_scoreboard_WAR->getMode() ==
           scoreboard_reads_mode::RELEASE_AT_WB)) {
        m_scoreboard_WAR->releaseRegisters(instruction);
      }
      m_scoreboard->releaseRegisters(instruction);
    }

  } else {
    if (instruction->get_extra_trace_instruction_info()
            .get_control_bits()
            .get_is_new_write_barrier()) {
      add_pending_wait_barrier_decrement(instruction, Wait_Barrier_Type::WRITE_WAIT_BARRIER,
      instruction->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier());       
    }
    if(instruction->m_is_ldgsts) {
      m_physical_warp[warp_id]->get_dependency_state()->decrease_num_pending_ldgsts();
    }
  }
  m_physical_warp[warp_id]->dec_inst_in_pipeline();
  warp_inst_complete(*instruction);
  get_gpu()->gpu_sim_insn_last_update_sid = get_sid();
  get_gpu()->gpu_sim_insn_last_update = get_gpu()->gpu_sim_cycle;
  set_last_inst_gpu_sim_cycle(get_gpu()->gpu_sim_cycle);
  set_last_inst_gpu_tot_sim_cycle(get_gpu()->gpu_tot_sim_cycle);
  instruction->clear();
}

void SM::issue_warp(register_set_uniptr &pipe_reg_set, warp_inst_t *next_inst,
                    const active_mask_t &active_mask, unsigned warp_id,
                    unsigned subcore_id, bool use_traditional_scoreboarding) {
  assert(next_inst->valid());
  std::unique_ptr<warp_inst_t> &pipe_reg = pipe_reg_set.get_free_smartptr(); 
  std::unique_ptr<warp_inst_t> inst_smart(next_inst);
  move_warp_uniptr(pipe_reg, inst_smart);

  trace_warp_inst_t *trace_inst = nullptr;
  trace_shd_warp_t *trace_warp = nullptr;
  if (m_config->is_trace_mode) {
    trace_inst = static_cast<trace_warp_inst_t *>(pipe_reg.get());
    trace_warp =
        static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id]);
    if (m_config->wait_block_enable &&
        trace_inst->wait_block_trace_origin ==
            WaitBlockTraceOrigin::SYNTHETIC &&
        (trace_warp->partial_capsule_active() ||
         m_partial_replay_progress[warp_id].valid)) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=partial-replay-synthetic-issued"
                << " trace_kernel=" << trace_inst->trace_kernel_id
                << " function=" << trace_inst->unique_function_id
                << " cta=" << m_physical_warp[warp_id]->get_cta_id()
                << " warp=" << warp_id
                << " requested_pc=0x" << format_hex_u64(trace_inst->pc)
                << " expected_pc=0x" << format_hex_u64(trace_warp->get_pc())
                << " action=abort" << std::endl;
      abort();
    }
  }

  m_physical_warp[warp_id]->get_IBuffer_remodeled()->issued();
  if (m_config->is_trace_mode) {
    const unsigned int replay_remaining_before =
        trace_inst->wait_block_trace_origin ==
                WaitBlockTraceOrigin::PARTIAL_BLOCKED_REPLAY
            ? trace_warp->partial_replay_remaining()
            : 0;
    trace_warp->commit_trace_inst(trace_inst);
    if (trace_inst->wait_block_trace_origin ==
        WaitBlockTraceOrigin::PARTIAL_BLOCKED_REPLAY) {
      m_partial_replayed_occurrences_total++;
      note_partial_replay_progress(
          warp_id, trace_warp->partial_replay_remaining(),
          trace_inst->pc, 1);
      if (replay_remaining_before == 1) {
        debug_log_wait_block_event(
            "partial-replay-drained",
            "trace_kernel=" +
                std::to_string(trace_inst->trace_kernel_id) +
                " function=" +
                std::to_string(trace_inst->unique_function_id) +
                " cta=" +
                std::to_string(m_physical_warp[warp_id]->get_cta_id()) +
                " warp=" + std::to_string(warp_id) +
                " last_pc=0x" + format_hex_u64(trace_inst->pc) +
                " expected_post_join_pc=0x" +
                format_hex_u64(trace_inst->next_traced_pc));
      }
    }
  }

  pipe_reg->issue(active_mask, warp_id,
                     m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
                     m_physical_warp[warp_id]->get_dynamic_warp_id(),
                     subcore_id);  // dynamic instruction information
  int num_active_threads = pipe_reg->active_count();

  if (num_active_threads > 0) {
    m_sm_stats.m_stats_map["warp_occ_dist" + std::to_string(num_active_threads)]->increment_with_integer(1);
  }
  
  func_exec_inst(*pipe_reg);

  // if(m_sm_id == 0 && warp_id == 0) { //&& get_stats()->m_last_kernel_id == 11) { // && warp_id == 0) {
  //   std::cout << "Issue. SM: " << m_sm_id << ". Subcore: " << subcore_id << ". Warp_ID: " << warp_id << ". PC: " << std::hex << pipe_reg->pc << ". Next traced PC:" << pipe_reg->next_traced_pc << std::dec << ". Cycle: " << get_current_gpu_cycle() << std::endl;
  //   fflush(stdout);
  // }

  // if(m_sm_id == 0 && subcore_id == 0 && ((*pipe_reg)->pc==0x260 || (*pipe_reg)->pc==0x2c0)) { // && warp_id == 0) {
  //   std::cout << "Measure. Issue. SM: " << m_sm_id << ". Subcore: " << subcore_id << ". Warp_ID: " << warp_id << ". PC: " << std::hex << (*pipe_reg)->pc << std::dec << ". Cycle: " << get_current_gpu_cycle() << std::endl;
  //   fflush(stdout);
  // }

  m_stats->warp_issues_from_last_power_sample[m_sm_id]
                                             [warp_id]++;  // MOD. Custom
                                                           // powermodel stats

  // [TRIMCHK] Always-on producer-spin issue accounting (no flag). Count the
  // spin-loop ops the simulator actually issues so a trimmed vs full run is
  // distinguishable in the log without re-running: PHASECHK/TRYWAIT (mbarrier
  // poll), NANOSLEEP (spin sleep), and the loop-back BRA. A trimmed trace issues
  // ~48-86x fewer of these. Emitted per kernel in gpu_print_stat.
  {
    if (pipe_reg->sync_kind == SyncInstructionKind::PHASECHK ||
        pipe_reg->sync_kind == SyncInstructionKind::TRYWAIT) {
      m_gpu->trimchk_spin_phasechk_issued++;
    } else if (pipe_reg->has_extra_trace_instruction_info() &&
               pipe_reg->get_extra_trace_instruction_info()
                       .get_op_code()
                       .rfind("NANOSLEEP", 0) == 0) {
      m_gpu->trimchk_spin_nanosleep_issued++;
    } else if (pipe_reg->op == BRANCH_OP) {
      m_gpu->trimchk_spin_branch_issued++;
    }
  }

  if (pipe_reg->op == MBARRIER_OP) {
    handle_sync_instruction(*pipe_reg, warp_id);
  } else if (pipe_reg->op == BARRIER_OP) {
    PartialWaitRegionRecord replay_join_region;
    const WaitBlockTraceOrigin trace_origin =
        m_config->is_trace_mode
            ? static_cast<trace_warp_inst_t *>(pipe_reg.get())
                  ->wait_block_trace_origin
            : WaitBlockTraceOrigin::CANONICAL;
    const bool is_partial_replay_join =
        m_config->is_trace_mode &&
        (trace_origin == WaitBlockTraceOrigin::PARTIAL_SIBLING_JOIN ||
         trace_origin == WaitBlockTraceOrigin::PARTIAL_BLOCKED_REPLAY) &&
        m_gpu->lookup_partial_wait_join(
            pipe_reg->trace_kernel_id, pipe_reg->unique_function_id,
            pipe_reg->pc, replay_join_region);
    // A predicated-off BAR (active_count()==0, e.g. a warp-specialized BAR.ARV/BAR.SYNC
    // executed only by a subset of warps) is a no-op for this warp: the warp does not
    // actually arrive at the barrier on hardware. Skip the barrier-set update entirely;
    // otherwise it would (a) register a spurious arrival and (b) decode to a fallback
    // bar_id=0 since no register operand value was captured for the inactive lanes.
    if (pipe_reg->active_count() == 0 || is_partial_replay_join) {
      // nothing to do for this warp
    } else {
      debug_print_sm_barrier_issue(*pipe_reg, warp_id, m_sm_id,
                                   m_config->bar_debug_enable,
                                   m_config->bar_debug_issue_budget);
      // B2: an ARRIVE (BAR.ARV) only signals arrival and keeps issuing; it must not be
      // recorded as "last inst at barrier" (that bookkeeping is for blocking SYNC/RED warps).
      // Still call warp_reaches_barrier so the arrival is registered on the id.
      if (pipe_reg->bar_type != ARRIVE) {
        m_physical_warp[warp_id]->store_info_of_last_inst_at_barrier(pipe_reg.get());
      }
      m_barriers.warp_reaches_barrier(m_physical_warp[warp_id]->get_cta_id(),
                                      warp_id, pipe_reg.get());
    }
  } else if (pipe_reg->op == MEMORY_BARRIER_OP) {
      // A memory barrier is an ordering/visibility fence, not a CTA rendezvous.
      // Record the fence scope so warp_waiting_at_mem_barrier can drain only the
      // stores visible at that scope (CTA = shared/L1, GPU = L2 + TMA). Modeling
      // this as a full-CTA rendezvous + fixed SM-wide stall (the legacy path) was
      // both wrong (membar is per-warp, scope-only) and the dominant inflation in
      // FA3 bwd, where MEMBAR.ALL.CTA executes 55k+ times.
      membar_scope_t scope = fence_scope_of(*pipe_reg);
      m_physical_warp[warp_id]->set_membar(scope);
      if (m_config->bar_debug_enable &&
          m_config->bar_debug_issue_budget > 0) {
        const std::string fop =
            pipe_reg->has_extra_trace_instruction_info()
                ? pipe_reg->get_extra_trace_instruction_info().get_op_code()
                : "<no-op>";
        // Build the whole line first, then emit with a single stream insertion so
        // concurrent SM threads do not interleave tokens within one log line.
        std::ostringstream oss;
        oss << "[MEMBARDBG][fence-enter] sm=" << m_sm_id
            << " warp=" << warp_id << " op=" << fop
            << " scope=" << (int)scope
            << " cta=" << m_physical_warp[warp_id]->get_pending_stores_cta_visible()
            << " gpu=" << m_physical_warp[warp_id]->get_pending_stores_gpu_visible()
            << " tma=" << (m_tma_unit_shared_of_sm->warp_has_outstanding_stores(warp_id) ? 1 : 0)
            << " cycle=" << get_current_gpu_cycle() << "\n";
        std::cerr << oss.str();
      }
      // Non-rendezvous fences (FENCE.*, MEMBAR.ALL.CTA/GPU) must NOT enter the CTA
      // barrier engine. Only an unverified/legacy form would fall through here.
      if (!is_non_rendezvous_memory_barrier(*pipe_reg)) {
        debug_print_sm_barrier_issue(*pipe_reg, warp_id, m_sm_id,
                                     m_config->bar_debug_enable,
                                     m_config->bar_debug_issue_budget);
        m_physical_warp[warp_id]->store_info_of_last_inst_at_barrier(
            pipe_reg.get());
        m_barriers.warp_reaches_barrier(m_physical_warp[warp_id]->get_cta_id(),
                                        warp_id, pipe_reg.get());
      }
  }else if(pipe_reg->op == GRID_BARRIER_OP) {
    m_physical_warp[warp_id]->set_gridbar();
    m_physical_warp[warp_id]->store_info_of_last_inst_at_barrier(pipe_reg.get());
  }

  ib_ooo_simt_info ib_ooo_simt_status;
  ib_ooo_simt_status.m_is_inst_reissued =
      pipe_reg->get_vpreg_need_to_reissue();
  if (!m_config->is_trace_mode) {
    updateSIMTStack(warp_id, pipe_reg.get());
  }
  if (use_traditional_scoreboarding ||
      m_config->is_remodeling_scoreboarding_enabled ||
      !m_config->is_trace_mode) {

    if ( (use_traditional_scoreboarding && m_config->is_trace_mode) || (m_config->is_trace_mode && m_config->is_remodeling_scoreboarding_enabled) ) {
      m_scoreboard->reserveRegisters_remodeling(pipe_reg.get());
      if (m_scoreboard_WAR->isEnabled()) {
        m_scoreboard_WAR->reserveRegisters_remodeling(pipe_reg.get());
      }
    } else {
      m_scoreboard->reserveRegisters(pipe_reg.get());
      if (m_scoreboard_WAR->isEnabled()) {
        m_scoreboard_WAR->reserveRegisters(pipe_reg.get());
      }
    }
  } else {
    bool is_yield = pipe_reg
                        ->get_extra_trace_instruction_info()
                        .get_control_bits()
                        .get_is_yield();
    unsigned int stall_count = pipe_reg
                                   ->get_extra_trace_instruction_info()
                                   .get_control_bits()
                                   .get_stall_count();
    if (is_yield) {
      m_physical_warp[warp_id]->get_dependency_state()->set_yield();
      stall_count = (stall_count == 0) ? m_config->num_stall_cycles_wait_after_bits_stall_0_and_yield : stall_count;
    }
    m_physical_warp[warp_id]->get_dependency_state()->set_stall_counter(
        stall_count);
    if( pipe_reg->m_is_ldgsts ) {
      m_physical_warp[warp_id]->get_dependency_state()->increase_num_pending_ldgsts();
    }
  }

  m_physical_warp[warp_id]->set_next_pc(pipe_reg->pc + pipe_reg->isize);

  if(m_config->is_trace_mode) {
    assert(m_physical_warp[warp_id]->has_function_call_context());
    assert(pipe_reg->unique_function_id == m_physical_warp[warp_id]->get_current_unique_function_id_call());
    bool is_any_thread_active = pipe_reg->get_active_mask().any();
    if(pipe_reg->op == CALL_OPS && is_any_thread_active) {
      new_addr_type call_address = pipe_reg->get_first_addr_valid();
      search_func_addr_result search_result = get_gpu()->get_extra_trace_info().search_function_addr(call_address);
      if(search_result.m_has_been_traced) {
        m_physical_warp[warp_id]->push_function_call(search_result.m_unique_function_id, pipe_reg->get_active_mask());
      }else if(pipe_reg->get_extra_trace_instruction_info().get_is_call_or_ret_with_relative()) {
        m_physical_warp[warp_id]->push_function_call(pipe_reg->unique_function_id, pipe_reg->get_active_mask());
      }else{
        assert(pipe_reg->next_traced_pc == (pipe_reg->pc + pipe_reg->isize) );
      }
    } else if(pipe_reg->op == RET_OPS && is_any_thread_active) {
      if (m_physical_warp[warp_id]->can_pop_non_root_function_call()) {
        m_physical_warp[warp_id]->pop_function_call(
            pipe_reg->get_active_mask());
      }
    }
    if (m_config->wait_block_enable) {
      handle_partial_wait_instruction_issue(*pipe_reg, warp_id);
    }
  }
}

void SM::func_exec_inst(warp_inst_t &inst) {
  if (m_config->is_trace_mode) {
    for (unsigned t = 0; t < m_warp_size; t++) {
      if (inst.active(t)) {
        unsigned warpId = inst.warp_id();
        unsigned tid = m_warp_size * warpId + t;

        // virtual function
        checkExecutionStatusAndUpdate(inst, t, tid);
      }
    }

    // here, we generate memory acessess and set the status if thread (done?)
    if (inst.is_load() || inst.is_store()) {
      inst.generate_mem_accesses();
    }

    trace_shd_warp_t *m_trace_warp =
        static_cast<trace_shd_warp_t *>(m_physical_warp[inst.warp_id()]);
    if (m_trace_warp->trace_done() && m_trace_warp->functional_done()) {
      m_trace_warp->get_IBuffer_remodeled()->flush(true);
      m_barriers.warp_exit(inst.warp_id());
      // [FWD drain-idle 축2] stamp the cycle this warp's trace drained (first time only).
      if (m_config->wgmma_step0_instrument_enable ||
          m_config->producer_tail_instrument_enable) {
        unsigned wid = inst.warp_id();
        if (wid < MAX_WARP_PER_SHADER && m_warp_drain_cycle[wid] == 0) {
          m_warp_drain_cycle[wid] = get_current_gpu_cycle();
        }
      }
    }
  } else {
    execute_warp_inst_t(inst);
    if (inst.is_load() || inst.is_store()) {
      inst.generate_mem_accesses();
    }
  }
}

void SM::check_if_warp_has_finished_executing_and_can_be_reclaim(
    shd_warp_t *warp) {
  unsigned int warp_id = warp->get_warp_id();
  if (warp->hardware_done() && !warp->done_exit() &&
      !m_scoreboard->pendingWrites(warp_id) &&
      !m_scoreboard_WAR->pendingReads(warp_id) && !warp->get_dependency_state()->are_pending_dependencies() &&
      !warp->is_atomic_pending()) {
    bool did_exit = false;
    for (unsigned t = 0; t < m_config->warp_size; t++) {
      unsigned tid = warp_id * m_config->warp_size + t;
      if (m_threadState[tid].m_active == true) {
        m_threadState[tid].m_active = false;
        unsigned cta_id = warp->get_cta_id();
        if (m_thread[tid] == NULL) {
          register_cta_thread_exit(cta_id, warp->get_kernel_info());
        } else {
          register_cta_thread_exit(cta_id, &(m_thread[tid]->get_kernel()));
        }
        m_not_completed -= 1;
        m_active_threads.reset(tid);
        did_exit = true;
      }
    }
    if (did_exit) {
      warp->set_done_exit();
      warp->get_dependency_state()->reset();
    }
    unsigned int m_subcore_id = warp_id %  m_num_subcores;
    m_subcores[m_subcore_id]->decrease_active_warp();
    --m_active_warps;
    assert(m_active_warps >= 0);
  }
}

unsigned int SM::get_kernel_id(unsigned warp_id) {
  return m_physical_warp[warp_id]->m_kernel_id;
}

void SM::init_warps(unsigned cta_id, unsigned start_thread, unsigned end_thread,
                    unsigned ctaid, int cta_size, kernel_info_t &kernel) {
  address_type start_pc = next_pc(start_thread);
  unsigned kernel_id = kernel.get_uid();
  // [CTA-imbalance diag] Stamp this CTA slot's launch cycle and clear its
  // tensor-op counter so [CTAFIN] can report per-CTA elapsed cycles + density.
  // Observe-only; no timing effect.
  if (cta_id < MAX_CTA_PER_SHADER) {
    m_cta_slot_start_cycle[cta_id] = (unsigned long long)m_gpu->gpu_sim_cycle;
    m_tensor_ops_by_cta_slot[cta_id] = 0;
    m_sm_idle_tensor_cyc_by_cta_slot[cta_id] = 0;
    m_fu_occupied_tensor_cyc_by_cta_slot[cta_id] = 0;
    m_sm_idle_cyc_by_cta_slot[cta_id] = 0;
    m_sm_idle_ibuffer_empty_cyc_by_cta_slot[cta_id] = 0;
    // [FWD drain-idle 축1·2·4] reset this CTA slot's new per-CTA counters at launch.
    m_wait_barrier_only_cyc_by_cta_slot[cta_id] = 0;
    m_drained_cyc_by_cta_slot[cta_id] = 0;
    m_wait_pending_by_cta_slot[cta_id] = 0;
    m_wait_released_by_cta_slot[cta_id] = 0;
    m_producer_tail_by_cta_slot[cta_id] = ProducerTailCounters();
    m_wait_block_first_late_reported_by_cta_slot[cta_id] = false;
  }
  if (m_config->model == POST_DOMINATOR) {
    unsigned start_warp = start_thread / m_config->warp_size;
    unsigned warp_per_cta = cta_size / m_config->warp_size;
    unsigned end_warp = end_thread / m_config->warp_size +
                        ((end_thread % m_config->warp_size) ? 1 : 0);

    m_stats->number_of_warps_per_kernel[m_stats->m_current_kernel_pos] +=
        end_warp - start_warp;  // MOD. Custom Stats
    for (unsigned i = start_warp; i < end_warp; ++i) {
      unsigned n_active = 0;
      simt_mask_t active_threads;
      for (unsigned t = 0; t < m_config->warp_size; t++) {
        unsigned hwtid = i * m_config->warp_size + t;
        if (hwtid < end_thread) {
          n_active++;
          assert(!m_active_threads.test(hwtid));
          m_active_threads.set(hwtid);
          active_threads.set(t);
        }
      }
      m_simt_stack[i]->launch(start_pc, active_threads);
      // [FWD drain-idle 축2·3] reset this warp's per-warp observe-only state at (re)launch so a
      // reused hardware warp slot (multi-wave SMs) does not inherit a prior CTA's role/drain/miss.
      if (i < MAX_WARP_PER_SHADER) {
        m_warp_is_producer[i] = false;
        m_warp_is_consumer[i] = false;
        m_warp_drain_cycle[i] = 0;
        m_sync_first_miss_barrier[i] = 0;
        m_sync_first_miss_cycle[i] = 0;
        if (i < m_wait_block_last_semantic_anchor_by_warp.size()) {
          m_wait_block_last_semantic_anchor_by_warp[i] =
              WaitBlockSemanticAnchor();
        }
      }
      if (m_gpu->resume_option == 1 && kernel_id == m_gpu->resume_kernel &&
          ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        char fname[2048];
        snprintf(fname, 2048, "checkpoint_files/warp_%d_%d_simt.txt",
                 i % warp_per_cta, ctaid);
        unsigned pc, rpc;
        m_simt_stack[i]->resume(fname);
        m_simt_stack[i]->get_pdom_stack_top_info(&pc, &rpc);
        for (unsigned t = 0; t < m_config->warp_size; t++) {
          if (m_thread != NULL) {
            m_thread[i * m_config->warp_size + t]->set_npc(pc);
            m_thread[i * m_config->warp_size + t]->update_pc();
          }
        }
        start_pc = pc;
      }
      unsigned int subcore_id = i % m_num_subcores;
      m_subcores[subcore_id]->increase_active_warp();
      m_physical_warp[i]->init(start_pc, cta_id, i, active_threads,
                               m_dynamic_warp_id, m_sm_id);
      m_physical_warp[i]->m_kernel_id = kernel_id;
      ++m_dynamic_warp_id;
      m_not_completed += n_active;
      ++m_active_warps;
    }
  }
  if (m_config->is_trace_mode) {
    std::string kernel_name = kernel.name();
    unsigned int kernel_unique_function_id = get_gpu()->get_extra_trace_info().get_unique_function_id(kernel_name);
    // Init traces
    unsigned start_warp = start_thread / m_config->warp_size;
    unsigned end_warp = end_thread / m_config->warp_size +
                        ((end_thread % m_config->warp_size) ? 1 : 0);

    // init_traces(start_warp, end_warp, kernel);
    std::vector<std::map<address_type, traced_instructions_by_pc> *> threadblock_traces;
    std::vector<std::vector<address_type> *> threadblock_traced_pcs;
    for (unsigned i = start_warp; i < end_warp; ++i) {
      trace_shd_warp_t *m_trace_warp =
          static_cast<trace_shd_warp_t *>(m_physical_warp[i]);
      m_trace_warp->clear();
      threadblock_traces.push_back(&(m_trace_warp->map_warp_traces));
      threadblock_traced_pcs.push_back(&(m_trace_warp->traced_pcs));
    }
    trace_kernel_info_t &trace_kernel =
        static_cast<trace_kernel_info_t &>(kernel);
    trace_kernel.get_next_threadblock_traces(threadblock_traces, threadblock_traced_pcs, m_gpu, m_gpu->get_extra_trace_info());

    // set the pc from the traces and ignore the functional model
    for (unsigned i = start_warp; i < end_warp; ++i) {
      m_physical_warp[i]->push_function_call(kernel_unique_function_id, m_physical_warp[i]->get_active_mask());
      trace_shd_warp_t *m_trace_warp =
          static_cast<trace_shd_warp_t *>(m_physical_warp[i]);
      m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
      m_trace_warp->set_kernel(&trace_kernel);
    }
  }
}

void SM::checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                       unsigned tid) {
  if (m_config->is_trace_mode) {
    if (inst.isatomic()) {
      m_physical_warp[inst.warp_id()]->inc_n_atomic();
    }

    if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
      new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
      unsigned num_addrs;
      num_addrs = translate_local_memaddr(
          inst.get_addr(t), tid,
          m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
          inst.data_size, (new_addr_type *)localaddrs);
      inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
    }

    if (inst.op == EXIT_OPS) {
      m_physical_warp[inst.warp_id()]->set_completed(t);
    }
  } else {
    if (inst.isatomic()) m_physical_warp[inst.warp_id()]->inc_n_atomic();
    if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
      new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
      unsigned num_addrs;
      num_addrs = translate_local_memaddr(
          inst.get_addr(t), tid,
          m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
          inst.data_size, (new_addr_type *)localaddrs);
      inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
    }
    if (ptx_thread_done(tid)) {
      m_physical_warp[inst.warp_id()]->set_completed(t);
      m_physical_warp[inst.warp_id()]->get_IBuffer_remodeled()->flush(false);
    }

    // PC-Histogram Update
    unsigned warp_id = inst.warp_id();
    unsigned pc = inst.pc;
    for (unsigned t = 0; t < m_config->warp_size; t++) {
      if (inst.active(t)) {
        int tid = warp_id * m_config->warp_size + t;
        cflog_update_thread_pc(m_sm_id, tid, pc);
      }
    }
  }
}

// Returns numbers of addresses in translated_addrs, each addr points to a 4B
// (32-bit) word
unsigned int SM::translate_local_memaddr(address_type localaddr, unsigned tid,
                                         unsigned num_shader, unsigned datasize,
                                         new_addr_type *translated_addrs) {
  // During functional execution, each thread sees its own memory space for
  // local memory, but these need to be mapped to a shared address space for
  // timing simulation.  We do that mapping here.

  address_type thread_base = 0;
  unsigned max_concurrent_threads = 0;
  if (m_config->gpgpu_local_mem_map) {
    // Dnew = D*N + T%nTpC + nTpC*C
    // N = nTpC*nCpS*nS (max concurent threads)
    // C = nS*K + S (hw cta number per gpu)
    // K = T/nTpC   (hw cta number per core)
    // D = data index
    // T = thread
    // nTpC = number of threads per CTA
    // nCpS = number of CTA per shader
    //
    // for a given local memory address threads in a CTA map to contiguous
    // addresses, then distribute across memory space by CTAs from successive
    // shader cores first, then by successive CTA in same shader core
    thread_base =
        4 *
        (kernel_padded_threads_per_cta *
             (m_sm_id + num_shader * (tid / kernel_padded_threads_per_cta)) +
         tid % kernel_padded_threads_per_cta);
    max_concurrent_threads =
        kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader;
  } else {
    // legacy mapping that maps the same address in the local memory space of
    // all threads to a single contiguous address region
    thread_base = 4 * (m_config->n_thread_per_shader * m_sm_id + tid);
    max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
  }
  assert(thread_base < 4 /*word size*/ * max_concurrent_threads);

  // If requested datasize > 4B, split into multiple 4B accesses
  // otherwise do one sub-4 byte memory access
  unsigned num_accesses = 0;

  if (datasize >= 4) {
    // >4B access, split into 4B chunks
    assert(datasize % 4 == 0);  // Must be a multiple of 4B
    num_accesses = datasize / 4;
    assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);  // max 32B
    assert(
        localaddr % 4 ==
        0);  // Address must be 4B aligned - required if accessing 4B per
             // request, otherwise access will overflow into next thread's space
    for (unsigned i = 0; i < num_accesses; i++) {
      address_type local_word = localaddr / 4 + i;
      address_type linear_address = local_word * max_concurrent_threads * 4 +
                                    thread_base + LOCAL_GENERIC_START;
      translated_addrs[i] = linear_address;
    }
  } else {
    // Sub-4B access, do only one access
    assert(datasize > 0);
    num_accesses = 1;
    address_type local_word = localaddr / 4;
    address_type local_word_offset = localaddr % 4;
    assert((localaddr + datasize - 1) / 4 ==
           local_word);  // Make sure access doesn't overflow into next 4B chunk
    address_type linear_address = local_word * max_concurrent_threads * 4 +
                                  local_word_offset + thread_base +
                                  LOCAL_GENERIC_START;
    translated_addrs[0] = linear_address;
  }
  return num_accesses;
}

void SM::create_logical_structures() {
  m_threadState = (thread_ctx_t *)calloc(sizeof(thread_ctx_t),
                                         m_config->n_thread_per_shader);
  m_not_completed = 0;
  m_active_threads.reset();
  m_n_active_cta = 0;
  for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_cta_status[i] = 0;
  for (unsigned i = 0; i < m_config->n_thread_per_shader; i++) {
    m_thread[i] = NULL;
    m_threadState[i].m_cta_id = -1;
    m_threadState[i].m_active = false;
  }
  create_shd_warp();
  std::string latch_name;
  for (unsigned int i = 0; i < m_num_subcores; i++) {
    latch_name =
        "EX_MEM_shared_reception_latch_for_subcore_" + std::to_string(i);
    register_set_uniptr *mem_sm_reception_latch_for_subcore = new register_set_uniptr(1, latch_name.c_str());
    m_EX_MEM_reception_latches_per_subcore.push_back(mem_sm_reception_latch_for_subcore);
    latch_name =
        "EX_TMA_shared_reception_latch_for_subcore_" + std::to_string(i);
    register_set_uniptr *tma_sm_reception_latch_for_subcore =
        new register_set_uniptr(1, latch_name.c_str());
    m_EX_TMA_reception_latches_per_subcore.push_back(
        tma_sm_reception_latch_for_subcore);
    
    Subcore *subcore = new Subcore(i, m_config, m_stats, this,
                                   &m_EX_DP_shared_sm_reception_latch,
                                   mem_sm_reception_latch_for_subcore,
                                   tma_sm_reception_latch_for_subcore);
    subcore->create_pipeline();
    m_subcores.push_back(subcore);
    m_EX_WB_sm_shared_units_subcore_latches.push_back(
        subcore->get_EX_WB_sm_shared_units_latch());
  }

  latch_name = "MEM_icnt_latch"; // Used for LDGSTS communication inside ldst_unit_sm
  register_set_uniptr *mem_sm_icnt_latch = new register_set_uniptr(1, latch_name.c_str());
  m_EX_MEM_reception_latches_per_subcore.push_back(mem_sm_icnt_latch);

  for (unsigned int i = 0; i < m_physical_warp.size(); i++) {
    m_subcores[i % m_num_subcores]->assign_warp_to_subcore(m_physical_warp[i]);
  }

  for (auto subcore : m_subcores) {
    subcore->finilized_warps_assignation();
  }

  m_scoreboard = std::make_shared<Scoreboard>(m_sm_id, m_config->max_warps_per_shader, m_gpu, m_config->is_trace_mode);
  m_scoreboard_WAR =
      std::make_shared<Scoreboard_reads>(m_sm_id, m_config->max_warps_per_shader, m_gpu,
                           m_config->scoreboard_war_reads_mode,
                           m_config->scoreboard_war_max_uses_per_reg,
                           m_config->is_trace_mode, m_stats);
  if (m_config->is_dp_pipeline_shared_for_subcores) {
    std::vector<register_set_uniptr*> m_EX_DP_shared_sm_reception_latches;
    m_EX_DP_shared_sm_reception_latches.push_back(&m_EX_DP_shared_sm_reception_latch);
    m_shared_dp_unit = new functional_unit_shared_sm_part(
        m_EX_WB_sm_shared_units_subcore_latches, m_config,
        m_config->max_dp_latency, "DP_SM_shared", this, DP__OP, false, false, 1,
        m_EX_DP_shared_sm_reception_latches, NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST, nullptr, 0, false, TraceEnhancedOperandType::NONE);
  }
}

void SM::create_memory_interfaces() {
  if (m_config->gpgpu_perfect_mem) {
    m_icnt = new perfect_memory_interface(this, m_cluster);
  } else {
    m_icnt = new shader_memory_interface(this, m_cluster);
  }
  m_mem_fetch_allocator = std::make_shared<shader_core_mem_fetch_allocator>(m_sm_id, m_tpc_id, m_memory_config);

  char name[STRSIZE];
  snprintf(name, STRSIZE, "L1I_%03d", m_sm_id);
  m_L1I_L1_half_C_cache = std::make_shared<read_only_cache>(name, m_config->m_L1I_L1_half_C_cache_config, m_sm_id,
                              get_shader_instruction_cache_id(), m_icnt,
                              IN_L1I_MISS_QUEUE);

  m_icnt_L0s =
      new L0_icnt(m_L1I_L1_half_C_cache.get(), m_gpu, this, m_config->max_reply_allowed_from_L1I,
                   m_config->max_request_allowed_to_L1I,
                   m_config->latency_L0_to_L1,
                   m_config->latency_L1_to_L0);

  for (auto subcore : m_subcores) {
    subcore->create_L0s(m_icnt_L0s);
    static_cast<L0_icnt *>(m_icnt_L0s)
        ->add_L0(static_cast<read_only_cache *>(subcore->get_L0I()));
  }
  for (auto subcore : m_subcores) {
    static_cast<L0_icnt *>(m_icnt_L0s)
        ->add_L0(static_cast<read_only_cache *>(subcore->get_L0C()));
  }
  m_subcore_req_fetch_L1I_priority = 0;
  m_ldst_unit_shared_of_sm = new ldst_unit_sm(
      m_EX_WB_sm_shared_units_subcore_latches,
      m_EX_MEM_reception_latches_per_subcore, m_icnt, m_icnt_L0s, m_mem_fetch_allocator,
      this, m_scoreboard, m_scoreboard_WAR, m_config, m_memory_config, m_stats,
      m_sm_id, m_tpc_id, m_config->memory_sm_prt_size);
  m_tma_unit_shared_of_sm =
      new tma_unit_sm(m_EX_WB_sm_shared_units_subcore_latches,
                      m_EX_TMA_reception_latches_per_subcore, m_config, this,
                      m_icnt, m_mem_fetch_allocator.get());
  static_cast<L0_icnt *>(m_icnt_L0s)
        ->add_L0(static_cast<read_only_cache *>(m_ldst_unit_shared_of_sm->get_L1C()));
}

void SM::create_shd_warp() {
  if (m_config->is_trace_mode) {
    m_physical_warp.resize(m_config->max_warps_per_shader);
    for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
      m_physical_warp[k] = new trace_shd_warp_t(
          this, m_config->warp_size, m_stats);
    }
  } else {
    m_physical_warp.resize(m_config->max_warps_per_shader);
    for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
      m_physical_warp[k] = new shd_warp_t(
          this, m_config->warp_size, m_stats);
    }
  }
}

unsigned int SM::sim_init_thread(kernel_info_t &kernel,
                                 ptx_thread_info **thread_info, int sid,
                                 unsigned tid, unsigned threads_left,
                                 unsigned num_threads, core_t *core,
                                 unsigned hw_cta_id, unsigned hw_warp_id,
                                 gpgpu_t *gpu) {
  if (m_config->is_trace_mode) {
    if (kernel.no_more_ctas_to_run()) {
      return 0;  // finished!
    }

    if (kernel.more_threads_in_cta()) {
      kernel.increment_thread_id();
    }

    if (!kernel.more_threads_in_cta()) kernel.increment_cta_id();

    return 1;
  } else {
    return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                               num_threads, core, hw_cta_id, hw_warp_id, gpu);
  }
}

void SM::reinit(unsigned start_thread, unsigned end_thread,
                bool reset_not_completed) {
  const unsigned first_warp = start_thread / m_config->warp_size;
  const unsigned end_warp = end_thread / m_config->warp_size;
  if (m_config->wait_block_enable) {
    for (unsigned i = first_warp; i < end_warp; ++i) {
      const bool pending_hold =
          i < m_pending_sync_waits.size() &&
          m_pending_sync_waits[i].valid;
      const bool partial_arm =
          i < m_partial_wait_arms.size() &&
          m_partial_wait_arms[i].valid;
      const bool replay_progress =
          i < m_partial_replay_progress.size() &&
          m_partial_replay_progress[i].valid;
      const bool joined_region =
          i < m_partial_joined_regions.size() &&
          !m_partial_joined_regions[i].empty();
      const bool trace_capsule =
          m_config->is_trace_mode &&
          static_cast<trace_shd_warp_t *>(m_physical_warp[i])
              ->partial_capsule_active();
      if (!pending_hold && !partial_arm && !replay_progress &&
          !joined_region && !trace_capsule) {
        continue;
      }
      const HopperMBarrierPendingWait &pending = m_pending_sync_waits[i];
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=warp-slot-reuse-with-active-wait-block-state"
                << " pending_hold=" << pending_hold
                << " partial_arm=" << partial_arm
                << " replay_progress=" << replay_progress
                << " joined_regions=" << joined_region
                << " trace_capsule=" << trace_capsule
                << " strategy="
                << wait_block_mode_to_string(pending.mode)
                << " warp=" << i
                << " action=abort" << std::endl;
      abort();
    }
  }
  for (unsigned i = first_warp;
       i < end_warp && i < m_tma_open_contracts_per_warp.size(); ++i) {
    const uint32_t open_contracts = m_tma_open_contracts_per_warp[i];
    const uint32_t outstanding =
        i < m_tma_outstanding_bindings_per_warp.size()
            ? m_tma_outstanding_bindings_per_warp[i]
            : 0;
    if (open_contracts == 0 && outstanding == 0) {
      continue;
    }
    ++m_tma_orphan_cleanup;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=warp-slot-reuse-with-active-ownership"
              << " warp=" << i
              << " open_contracts=" << open_contracts
              << " outstanding=" << outstanding << std::endl;
    debug_dump_tma_binding_snapshot("warp-slot-reuse");
    abort();
  }
  if (reset_not_completed) {
    for (const auto &entry : m_hopper_mbarriers) {
      const HopperMBarrierObject &barrier = entry.second;
      if (barrier.assignment_open ||
          barrier.expected_tx_bytes != barrier.assigned_tx_bytes ||
          barrier.assigned_tx_bytes != barrier.completed_tx_bytes ||
          barrier.outstanding_tma_bindings != 0) {
        ++m_tma_orphan_cleanup;
        std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=sm-reset-with-active-ownership"
                  << " trace_kernel=" << entry.first.trace_kernel_id
                  << " cta_slot=" << entry.first.cta_id << " barrier=0x"
                  << format_hex_u64(entry.first.barrier_addr)
                  << " generation=" << barrier.gen
                  << " contract_id=" << barrier.tx_contract_id
                  << " assignment_open=" << barrier.assignment_open
                  << " expected=" << barrier.expected_tx_bytes
                  << " assigned=" << barrier.assigned_tx_bytes
                  << " completed=" << barrier.completed_tx_bytes
                  << " outstanding=" << barrier.outstanding_tma_bindings
                  << std::endl;
        debug_dump_tma_binding_snapshot("sm-reset-active-ownership");
        abort();
      }
    }
    m_not_completed = 0;
    m_active_threads.reset();

    // Jin: for concurrent kernels on a SM
    m_occupied_n_threads = 0;
    m_occupied_shmem = 0;
    m_occupied_regs = 0;
    m_occupied_ctas = 0;
    m_occupied_hwtid.reset();
    m_occupied_cta_to_hwtid.clear();
    m_active_warps = 0;
    m_hopper_mbarriers.clear();
  }
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].n_insn = 0;
    m_threadState[i].m_cta_id = -1;
  }
  for (unsigned i = first_warp; i < end_warp; ++i) {
    m_physical_warp[i]->reset();
    m_simt_stack[i]->reset();
    if (i < m_pending_sync_waits.size()) {
      m_pending_sync_waits[i] = HopperMBarrierPendingWait();
    }
    if (i < m_partial_wait_arms.size()) {
      m_partial_wait_arms[i] = HopperPartialWaitArm();
    }
    if (i < m_partial_replay_progress.size()) {
      m_partial_replay_progress[i] = HopperPartialReplayProgress();
    }
    if (i < m_partial_joined_regions.size()) {
      m_partial_joined_regions[i].clear();
    }
    if (i < m_tma_open_contracts_per_warp.size()) {
      m_tma_open_contracts_per_warp[i] = 0;
    }
    if (i < m_tma_outstanding_bindings_per_warp.size()) {
      m_tma_outstanding_bindings_per_warp[i] = 0;
    }
    if (i < m_tma_max_completed_transfer_uid_per_warp.size()) {
      m_tma_max_completed_transfer_uid_per_warp[i] = 0;
    }
  }
}

void SM::register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel) {
  assert(m_cta_status[cta_num] > 0);
  m_cta_status[cta_num]--;
  if (!m_cta_status[cta_num]) {
    if (m_config->wait_block_enable) {
      for (unsigned warp_id = 0; warp_id < m_pending_sync_waits.size();
           ++warp_id) {
        const HopperMBarrierPendingWait &pending =
            m_pending_sync_waits[warp_id];
        if (!pending.valid || pending.key.cta_id != cta_num) {
          continue;
        }
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=cta-exit-with-active-hold"
                  << " strategy=" << wait_block_mode_to_string(pending.mode)
                  << " trace_kernel=" << pending.key.trace_kernel_id
                  << " cta=" << cta_num << " warp=" << warp_id
                  << " action_pc=0x" << format_hex_u64(pending.action_pc)
                  << " action=abort" << std::endl;
        abort();
      }
      for (unsigned warp_id = 0; warp_id < m_partial_wait_arms.size();
           ++warp_id) {
        if (m_physical_warp[warp_id]->get_cta_id() != cta_num) {
          continue;
        }
        const bool partial_arm = m_partial_wait_arms[warp_id].valid;
        const bool replay_progress =
            m_partial_replay_progress[warp_id].valid;
        const bool joined_region =
            !m_partial_joined_regions[warp_id].empty();
        const bool trace_capsule =
            m_config->is_trace_mode &&
            static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id])
                ->partial_capsule_active();
        if (!partial_arm && !replay_progress && !joined_region &&
            !trace_capsule) {
          continue;
        }
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=cta-exit-with-active-partial-state"
                  << " cta=" << cta_num << " warp=" << warp_id
                  << " partial_arm=" << partial_arm
                  << " replay_progress=" << replay_progress
                  << " joined_regions=" << joined_region
                  << " trace_capsule=" << trace_capsule
                  << " action=abort" << std::endl;
        abort();
      }
    }
    // Record first completed CTA for ground truth validation
    unsigned global_cta_id = m_local_to_global_cta_id[cta_num];
    m_gpu->record_first_completed_cta(global_cta_id, m_sm_id);

    if (m_config->producer_tail_instrument_enable) {
      unsigned producer_warps = 0, consumer_warps = 0;
      unsigned long long producer_last_drain = 0;
      unsigned long long consumer_last_drain = 0;
      for (unsigned w = 0; w < MAX_WARP_PER_SHADER; ++w) {
        if (m_physical_warp[w] == nullptr ||
            m_physical_warp[w]->get_cta_id() != cta_num) {
          continue;
        }
        if (m_warp_is_producer[w]) {
          producer_warps++;
          producer_last_drain =
              std::max(producer_last_drain, m_warp_drain_cycle[w]);
        }
        if (m_warp_is_consumer[w]) {
          consumer_warps++;
          consumer_last_drain =
              std::max(consumer_last_drain, m_warp_drain_cycle[w]);
        }
      }
      const ProducerTailCounters &tail =
          m_producer_tail_by_cta_slot[cta_num];
      printf(
          "[PRODTAIL] sm=%u cta_slot=%u global_cta=%u "
          "producer_warps=%u consumer_warps=%u "
          "producer_last_drain=%llu consumer_last_drain=%llu "
          "tail_start=%llu tail_cycles=%llu issued=%llu "
          "issued_phasechk=%llu issued_nanosleep=%llu "
          "issued_branch=%llu issued_tma=%llu issued_other=%llu "
          "phasechk_ready=%llu phasechk_not_ready=%llu "
          "stall_cycles=%llu stall_not_selected=%llu "
          "stall_next_stage=%llu stall_issue_port=%llu "
          "stall_no_valid=%llu stall_wait_barrier=%llu "
          "stall_stall_count=%llu stall_fu_occupied=%llu "
          "stall_inst_barrier=%llu stall_tma_flush=%llu "
          "stall_scoreboard=%llu stall_l1c=%llu "
          "stall_result_queue=%llu stall_yield=%llu "
          "stall_unattributed=%llu\n",
          m_sm_id, cta_num, global_cta_id, producer_warps,
          consumer_warps, producer_last_drain, consumer_last_drain,
          tail.start_cycle, tail.cycles, tail.issued,
          tail.issued_phasechk, tail.issued_nanosleep,
          tail.issued_branch, tail.issued_tma, tail.issued_other,
          tail.phasechk_ready, tail.phasechk_not_ready,
          tail.stall_cycles, tail.stall_not_selected,
          tail.stall_next_stage, tail.stall_issue_port,
          tail.stall_no_valid, tail.stall_wait_barrier,
          tail.stall_stall_count, tail.stall_fu_occupied,
          tail.stall_inst_barrier, tail.stall_tma_flush,
          tail.stall_scoreboard, tail.stall_l1c,
          tail.stall_result_queue, tail.stall_yield,
          tail.stall_unattributed);
      m_producer_tail_by_cta_slot[cta_num] = ProducerTailCounters();
    }

    if (m_config->producer_spin_ready_instrument_enable) {
      const ProducerSpinCounters &spin = m_producer_spin_by_cta_slot[cta_num];
      printf(
          "[PRODSPIN] sm=%u cta_slot=%u global_cta=%u "
          "phasechk_total=%llu phasechk_ready=%llu phasechk_not_ready=%llu "
          "ready_run_retries=%llu issued_phasechk=%llu issued_nanosleep=%llu "
          "issued_branch=%llu issued_other=%llu\n",
          m_sm_id, cta_num, m_local_to_global_cta_id[cta_num],
          spin.phasechk_total, spin.phasechk_ready, spin.phasechk_not_ready,
          spin.ready_run_retries, spin.issued_phasechk, spin.issued_nanosleep,
          spin.issued_branch, spin.issued_other);
      m_producer_spin_by_cta_slot[cta_num] = ProducerSpinCounters();
      // Reset per-warp ready-run trackers for warps owned by this CTA slot so a
      // reused slot (bwd multi-wave) starts clean.
      for (unsigned w = 0; w < MAX_WARP_PER_SHADER; ++w) {
        if (m_physical_warp[w] != nullptr &&
            m_physical_warp[w]->get_cta_id() == cta_num) {
          m_producer_spin_prev_ready[w] = false;
          m_producer_spin_in_ready_run[w] = false;
        }
      }
    }

    // [Spin-wait observe] Emit the SM-global cumulative would-be-park summary.
    // Counters accumulate across the whole SM, so the final CTA-exit print holds
    // the complete totals; a post-processing pass takes the last/max per SM.
    if (m_config->spin_wait_observe_enable) {
      print_spin_wait_observe_summary();
    }
    if (m_config->wait_block_enable) {
      print_wait_block_summary();
    }

    // [CTA-imbalance diag] Emit this CTA's finish cycle, elapsed cycles, and raw
    // tensor-op (WGMMA) count so tail-drain can be correlated with tensor density
    // AND per-CTA duration. Keyed by hardware CTA slot so multi-wave SMs (bwd)
    // report each CTA separately; the slot counters are reset for reuse by the
    // next CTA scheduled into this slot. Observe-only, gated.
    if (m_config->wgmma_step0_instrument_enable) {
      unsigned long long finish_cyc = (unsigned long long)m_gpu->gpu_sim_cycle;
      unsigned long long start_cyc = m_cta_slot_start_cycle[cta_num];
      printf("[CTAFIN] sm=%u cta_slot=%u global_cta=%u start_cyc=%llu "
             "finish_cyc=%llu elapsed_cyc=%llu tensor_ops=%llu sm_idle_tensor_cyc=%llu "
             "fu_occupied_tensor_cyc=%llu",
             m_sm_id, cta_num, global_cta_id, start_cyc, finish_cyc,
             finish_cyc - start_cyc, m_tensor_ops_by_cta_slot[cta_num],
             m_sm_idle_tensor_cyc_by_cta_slot[cta_num],
             m_fu_occupied_tensor_cyc_by_cta_slot[cta_num]);
      // [CTA stall breakdown] append the non-tensor drain-idle columns on the SAME line
      // (only when enabled) so the [CTAFIN] parser sees one record per CTA.
      if (m_config->cta_stall_breakdown_instrument_enable) {
        printf(" sm_idle_cyc=%llu sm_idle_ibuffer_empty_cyc=%llu",
               m_sm_idle_cyc_by_cta_slot[cta_num],
               m_sm_idle_ibuffer_empty_cyc_by_cta_slot[cta_num]);
        // [FWD drain-idle 축1] mutually-exclusive recoverable(mbarrier-only) vs floor(drained) idle.
        // [FWD drain-idle 축4] per-CTA mbarrier test counts.
        printf(" sm_idle_wait_barrier_only_cyc=%llu sm_idle_drained_cyc=%llu "
               "wait_pending=%llu wait_released=%llu",
               m_wait_barrier_only_cyc_by_cta_slot[cta_num],
               m_drained_cyc_by_cta_slot[cta_num],
               m_wait_pending_by_cta_slot[cta_num],
               m_wait_released_by_cta_slot[cta_num]);
        // [FWD drain-idle 축2] producer vs consumer last trace-drain cycle for this CTA: scan the
        // warps that belong to this CTA slot and take the max drain cycle per role (0 = none).
        unsigned long long prod_drain = 0, cons_drain = 0;
        for (unsigned w = 0; w < MAX_WARP_PER_SHADER; w++) {
          if (m_physical_warp[w] == NULL) continue;
          if (m_physical_warp[w]->get_cta_id() != cta_num) continue;
          if (m_warp_is_producer[w] && m_warp_drain_cycle[w] > prod_drain)
            prod_drain = m_warp_drain_cycle[w];
          if (m_warp_is_consumer[w] && m_warp_drain_cycle[w] > cons_drain)
            cons_drain = m_warp_drain_cycle[w];
        }
        printf(" producer_last_drain_cyc=%llu consumer_last_drain_cyc=%llu",
               prod_drain, cons_drain);
      }
      printf("\n");
      m_tensor_ops_by_cta_slot[cta_num] = 0;
      m_cta_slot_start_cycle[cta_num] = 0;
      m_sm_idle_tensor_cyc_by_cta_slot[cta_num] = 0;
      m_fu_occupied_tensor_cyc_by_cta_slot[cta_num] = 0;
      m_sm_idle_cyc_by_cta_slot[cta_num] = 0;
      m_sm_idle_ibuffer_empty_cyc_by_cta_slot[cta_num] = 0;
      // [FWD drain-idle 축1·2·4] reset the new per-CTA counters for slot reuse.
      m_wait_barrier_only_cyc_by_cta_slot[cta_num] = 0;
      m_drained_cyc_by_cta_slot[cta_num] = 0;
      m_wait_pending_by_cta_slot[cta_num] = 0;
      m_wait_released_by_cta_slot[cta_num] = 0;
    }

    // Increment the completed CTAs
    m_sm_stats.m_stats_map["ctas_completed"]->increment_with_integer(1);
    m_n_active_cta--;
    m_barriers.deallocate_barrier(cta_num);
    clear_sync_barrier_state_for_cta(cta_num);
    if (m_config->wait_block_enable && m_n_active_cta == 0 &&
        (m_missing_wait_holds_total != m_missing_wait_releases_total ||
         m_fold_parks_total != m_fold_releases_total ||
         m_partial_join_holds_total != m_partial_join_releases_total ||
         m_partial_deferred_occurrences_total !=
             m_partial_replayed_occurrences_total +
                 m_partial_folded_occurrences_total)) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=terminal-hold-release-imbalance"
                << " missing_holds=" << m_missing_wait_holds_total
                << " missing_releases=" << m_missing_wait_releases_total
                << " poll_holds=" << m_fold_parks_total
                << " poll_releases=" << m_fold_releases_total
                << " partial_holds=" << m_partial_join_holds_total
                << " partial_releases=" << m_partial_join_releases_total
                << " partial_deferred="
                << m_partial_deferred_occurrences_total
                << " partial_replayed="
                << m_partial_replayed_occurrences_total
                << " partial_folded="
                << m_partial_folded_occurrences_total
                << " action=abort" << std::endl;
      abort();
    }
    shader_CTA_count_unlog(m_sm_id, 1);

    SHADER_DPRINTF(
        LIVENESS,
        "GPGPU-Sim uArch: Finished CTA #%u (%lld,%lld), %u CTAs running\n",
        cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle,
        m_n_active_cta);

    if (m_n_active_cta == 0) {
      SHADER_DPRINTF(
          LIVENESS,
          "GPGPU-Sim uArch: Empty (last released kernel %u \'%s\').\n",
          kernel->get_uid(), kernel->name().c_str());
      fflush(stdout);

      // Shader can only be empty when no more cta are dispatched
      if (kernel != m_kernel) {
        assert(m_kernel == NULL || !m_gpu->kernel_more_cta_left(m_kernel));
      }
      m_kernel = NULL;
    }

    // Jin: for concurrent kernels on sm
    release_shader_resource_1block(cta_num, *kernel);
    #pragma omp critical
    {
      m_gpu->inc_completed_cta();
      m_gpu->decrease_num_threads_kernel(kernel->get_uid(), kernel->threads_per_cta());
      kernel->dec_running();
      if (!m_gpu->kernel_more_cta_left(kernel)) {
        if (!kernel->running()) {
          SHADER_DPRINTF(LIVENESS,
                        "GPGPU-Sim uArch: GPU detected kernel %u \'%s\' "
                        "finished on shader %u.\n",
                        kernel->get_uid(), kernel->name().c_str(), m_sm_id);

          if (m_kernel == kernel) m_kernel = NULL;
          m_gpu->set_kernel_done(kernel);
        }
      }
    }
  }
}

address_type SM::next_pc(int tid) const {
  if (tid == -1) return -1;
  ptx_thread_info *the_thread = m_thread[tid];
  if (the_thread == NULL) return -1;
  return the_thread
      ->get_pc();  // PC should already be updatd to next PC at this point (was
                   // set in shader_decode() last time thread ran)
}

void SM::set_kernel(kernel_info_t *k) {
  assert(k);
  const uint64_t next_launch_uid = k->get_uid();
  if (m_tma_binding_launch_uid != 0 &&
      m_tma_binding_launch_uid != next_launch_uid) {
    for (unsigned int warp_id = 0;
         warp_id < m_tma_open_contracts_per_warp.size(); ++warp_id) {
      const uint32_t open_contracts =
          m_tma_open_contracts_per_warp[warp_id];
      const uint32_t outstanding =
          m_tma_outstanding_bindings_per_warp[warp_id];
      if (open_contracts != 0 || outstanding != 0) {
        ++m_tma_orphan_cleanup;
        std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=kernel-switch-with-active-warp-ownership"
                  << " old_launch=" << m_tma_binding_launch_uid
                  << " next_launch=" << next_launch_uid
                  << " warp=" << warp_id
                  << " open_contracts=" << open_contracts
                  << " outstanding=" << outstanding << std::endl;
        debug_dump_tma_binding_snapshot(
            "kernel-switch-active-warp-ownership");
        abort();
      }
    }
    for (const auto &entry : m_hopper_mbarriers) {
      const HopperMBarrierObject &barrier = entry.second;
      if (barrier.assignment_open ||
          barrier.expected_tx_bytes != barrier.assigned_tx_bytes ||
          barrier.assigned_tx_bytes != barrier.completed_tx_bytes ||
          barrier.outstanding_tma_bindings != 0) {
        ++m_tma_orphan_cleanup;
        std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=kernel-switch-with-active-ownership"
                  << " old_launch=" << m_tma_binding_launch_uid
                  << " next_launch=" << next_launch_uid
                  << " contract_id=" << barrier.tx_contract_id
                  << " expected=" << barrier.expected_tx_bytes
                  << " assigned=" << barrier.assigned_tx_bytes
                  << " completed=" << barrier.completed_tx_bytes
                  << " outstanding=" << barrier.outstanding_tma_bindings
                  << std::endl;
        debug_dump_tma_binding_snapshot("kernel-switch-active-ownership");
        abort();
      }
    }
    debug_dump_tma_binding_counters();
    m_tma_contracts_opened = 0;
    m_tma_contracts_fully_assigned = 0;
    m_tma_max_open_contracts_per_warp = 0;
    m_tma_bindings_created = 0;
    m_tma_bindings_completed = 0;
    m_tma_assigned_credit_bytes = 0;
    m_tma_completed_credit_bytes = 0;
    m_tma_uid_reorders = 0;
    m_tma_cross_barrier_applications = 0;
    m_tma_contract_mismatches = 0;
    m_tma_stale_generation = 0;
    m_tma_launch_or_cta_mismatch = 0;
    m_tma_duplicate_completion = 0;
    m_tma_orphan_cleanup = 0;
    std::fill(m_tma_open_contracts_per_warp.begin(),
              m_tma_open_contracts_per_warp.end(), 0);
    std::fill(m_tma_outstanding_bindings_per_warp.begin(),
              m_tma_outstanding_bindings_per_warp.end(), 0);
    std::fill(m_tma_max_completed_transfer_uid_per_warp.begin(),
              m_tma_max_completed_transfer_uid_per_warp.end(), 0);
  }
  m_tma_binding_launch_uid = next_launch_uid;
  m_kernel = k;
  //        k->inc_running();
  printf(
      "GPGPU-Sim uArch: Shader %d bind to kernel launch_uid=%u trace_kernel_id=%u \'%s\'\n",
      m_sm_id, m_kernel->get_uid(), m_kernel->get_trace_kernel_id(),
      m_kernel->name().c_str());
}
kernel_info_t *SM::get_kernel() { return this->core_t::get_kernel_info(); }
kernel_info_t *SM::get_kernel_info() { return this->core_t::get_kernel_info(); }

unsigned long long SM::get_current_gpu_cycle() {
  return m_gpu->get_current_gpu_cycle();
}

unsigned int SM::get_num_subcores() { return m_num_subcores; }
unsigned int SM::get_sid() const { return m_sm_id; }
unsigned int SM::get_tpc_id() const { return m_tpc_id; }
gpgpu_sim *SM::get_gpu() { return this->core_t::get_gpu(); }
shader_core_mem_fetch_allocator &SM::get_memf_fetch_allocator() { return *m_mem_fetch_allocator; }
read_only_cache* SM::get_L1C() { return m_ldst_unit_shared_of_sm->get_L1C(); }
std::shared_ptr<Scoreboard_reads> SM::get_scoreboard_WAR() { return m_scoreboard_WAR; }
std::shared_ptr<Scoreboard> SM::get_scoreboard() { return m_scoreboard; }
const memory_config *SM::get_memory_config() const { return m_memory_config; }
const shader_core_config *SM::get_config() const { return m_config; }
shader_core_stats *SM::get_stats() { return m_stats; }

std::list<unsigned> SM::get_regs_written(const inst_t &fvt) const {
  std::list<unsigned> result;
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = fvt.arch_reg.dst[op];  // this math needs to match that used
                                         // in function_info::ptx_decode_inst
    if (reg_num >= 0)                    // valid register
      result.push_back(reg_num);
  }
  return result;
}

shd_warp_t *SM::get_shd_warp(int id) { return m_physical_warp[id]; }
int SM::get_subcore_req_fetch_L1I_priority() {
  return m_subcore_req_fetch_L1I_priority;
}
void SM::set_subcore_req_fetch_L1I_priority(
    int new_subcore_req_fetch_L1I_priority) {
  m_subcore_req_fetch_L1I_priority = new_subcore_req_fetch_L1I_priority;
}

void SM::set_last_inst_gpu_sim_cycle(
    unsigned long long last_inst_gpu_sim_cycle) {
  m_last_inst_gpu_sim_cycle = last_inst_gpu_sim_cycle;
}
void SM::set_last_inst_gpu_tot_sim_cycle(
    unsigned long long last_inst_gpu_tot_sim_cycle) {
  m_last_inst_gpu_tot_sim_cycle = last_inst_gpu_tot_sim_cycle;
}

bool SM::is_any_subcore_problems_of_fordward_progress() const {
  bool res = false;
  for(auto subcore : m_subcores) {
    res = res || subcore->is_subcore_with_problems_of_fordward_progress();
  }
  return res;
}

bool SM::get_is_loog_enabled() { return m_config->is_loog_enabled; }
RRS *SM::get_loog_rrs() {
  throw std::logic_error(
      "LOOG is not compatible with this new accurate remodeling");
}

void SM::get_pdom_stack_top_info(unsigned tid, unsigned *pc,
                                 unsigned *rpc) const {
  unsigned warp_id = tid / m_config->warp_size;
  m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
}

void SM::get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                 unsigned *pc, unsigned *rpc) {
  if (m_config->is_trace_mode) {
    // In trace-driven mode, we assume no control hazard
    *pc = pI->pc;
    *rpc = pI->pc;
  } else {
    m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
  }
}

const active_mask_t &SM::get_active_mask(unsigned warp_id,
                                         const warp_inst_t *pI) {
  if (m_config->is_trace_mode) {
    return pI->get_active_mask();
  } else {
    return m_simt_stack[warp_id]->get_active_mask();
  }
}

void SM::warp_exit(unsigned warp_id) {
  bool done = true;
  for (unsigned i = warp_id * get_config()->warp_size;
       i < (warp_id + 1) * get_config()->warp_size; i++) {
    //		if(this->m_thread[i]->m_functional_model_thread_state &&
    // this->m_thread[i].m_functional_model_thread_state->donecycle()==0) {
    // done = false;
    //		}

    if (m_thread[i] && !m_thread[i]->is_done()) done = false;
  }
  // if (m_warp[warp_id].get_n_completed() == get_config()->warp_size)
  // if (this->m_simt_stack[warp_id]->get_num_entries() == 0)
  if (done) m_barriers.warp_exit(warp_id);
}

void SM::cache_flush() { m_ldst_unit_shared_of_sm->flush(); }

void SM::cache_invalidate() { 
  m_ldst_unit_shared_of_sm->invalidate(); 
  if(m_config->invalidate_instruction_caches_at_kernel_end) {
    while(!m_pending_instruction_region_prewarm.empty()) {
      m_pending_instruction_region_prewarm.pop();
    }
    m_activated_instruction_region_prewarm.clear();
    m_icnt_L0s->flush();
    m_L1I_L1_half_C_cache->invalidate();
    for(auto subcore : m_subcores) {
      subcore->get_L0I()->invalidate();
    }
  }  
}
void SM::data_cache_invalidate() { 
  m_ldst_unit_shared_of_sm->invalidate(); 
}

void SM::accept_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  m_L1I_L1_half_C_cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

void SM::accept_ldst_unit_response(mem_fetch *mf) {
  // TMA bulk responses are owned by the TMA unit, not the ldst unit. This is
  // the only behavioral change to the response path; ordinary LD/ST responses
  // continue to flow to the ldst unit unchanged.
  if (mf != nullptr && mf->is_tma()) {
    m_tma_unit_shared_of_sm->fill(mf);
    return;
  }
  m_ldst_unit_shared_of_sm->fill(mf);
}

bool SM::fetch_unit_response_buffer_full() const { return false; }

bool SM::ldst_unit_response_buffer_full() const {
  return m_ldst_unit_shared_of_sm->response_buffer_full();
}

// Cuda barrier management

bool SM::check_if_non_released_reduction_barrier(warp_inst_t &inst) {
  unsigned warp_id = inst.warp_id();
  bool bar_red_op = (inst.op == BARRIER_OP) && (inst.bar_type == RED);
  bool non_released_barrier_reduction = false;
  bool warp_stucked_at_barrier = warp_waiting_at_barrier(warp_id);
  bool single_inst_in_pipeline =
      (m_physical_warp[warp_id]->num_issued_inst_in_pipeline() == 1);
  non_released_barrier_reduction =
      single_inst_in_pipeline and warp_stucked_at_barrier and bar_red_op;
  printf("non_released_barrier_reduction=%u\n", non_released_barrier_reduction);
  return non_released_barrier_reduction;
}

bool SM::warp_waiting_at_barrier(unsigned warp_id) const {
  if (m_barriers.warp_waiting_at_barrier(warp_id)) {
    return true;
  }
  if (warp_id >= m_pending_sync_waits.size()) {
    return false;
  }
  const HopperMBarrierPendingWait &pending_wait =
      m_pending_sync_waits[warp_id];
  if (!pending_wait.valid) {
    return false;
  }
  // Pure peek: report hold state, do NOT clear here. The clear + "wait released"
  // accounting and any captured-poll cursor skip happen exactly once on the
  // issue path in service_wait_block_release(), never from this query
  // from a std::sort comparator). See design doc 6.5.1.
  return !is_wait_block_pending_satisfied(pending_wait);
}

// Non-mutating alias used off the issue path (comparator, stall scans). Same
// value as warp_waiting_at_barrier; kept as a distinct name so callers document
// that they must not rely on any side effect.
bool SM::warp_held_on_mbarrier_wait(unsigned warp_id) const {
  return warp_waiting_at_barrier(warp_id);
}

// [Stage 1] Common cursor-skip shared by the parked-release and immediate-ready
// fold paths. Flushes younger prefetched IBuffer entries, builds the allowed
// skip-PC set from the region (body_pcs + loopback_bra_pc), and advances the
// trace cursor to region.exit_pc. All generation/counter/pending-wait/park
// bookkeeping is intentionally left to the caller so the two paths cannot
// diverge in their skip mechanics again. See plan Stage 1.
unsigned SM::fold_cursor_skip_to_exit(unsigned warp_id,
                                      const SpinWaitRegionRecord &region,
                                      FoldSkipStats &out_stats) {
  trace_shd_warp_t *tw =
      static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id]);
  const bool in_partial_replay =
      tw->partial_replay_active() &&
      tw->partial_replay_origin() ==
          WaitBlockTraceOrigin::PARTIAL_BLOCKED_REPLAY;
  const unsigned int cursor_before_flush = tw->used_insts;
  tw->get_IBuffer_remodeled()->flush(false);
  const unsigned int cursor_after_flush = tw->used_insts;
  if (cursor_after_flush > cursor_before_flush) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id
              << "] cyc=" << get_current_gpu_cycle()
              << " reason=poll-prefetch-rollback-direction"
              << " block=" << region.block_id << " warp=" << warp_id
              << " cursor_before_flush=" << cursor_before_flush
              << " cursor_after_flush=" << cursor_after_flush
              << " action=abort" << std::endl;
    abort();
  }
  std::set<address_type> allowed_skip_pcs;
  for (uint64_t bp : region.body_pcs) {
    allowed_skip_pcs.insert(static_cast<address_type>(bp));
  }
  // body_pcs deliberately excludes the loopback BRA; add it so the guard lets
  // the cursor advance over the loop-closing branch on its way to exit_pc.
  allowed_skip_pcs.insert(static_cast<address_type>(region.loopback_bra_pc));
  const unsigned int skipped =
      in_partial_replay
          ? tw->skip_partial_replay_to_exit(
                static_cast<address_type>(region.exit_pc),
                static_cast<address_type>(region.loopback_bra_pc),
                allowed_skip_pcs, &out_stats)
          : tw->skip_spin_to_exit(
                static_cast<address_type>(region.exit_pc),
                static_cast<address_type>(region.loopback_bra_pc),
                allowed_skip_pcs, &out_stats);
  const unsigned int cursor_after_skip = tw->used_insts;
  if (in_partial_replay) {
    m_partial_folded_occurrences_total += skipped;
  }
  if (!in_partial_replay) {
    out_stats.cursor_before_flush = cursor_before_flush;
    out_stats.cursor_after_flush = cursor_after_flush;
    out_stats.cursor_after_skip = cursor_after_skip;
    out_stats.prefetched_rollback = cursor_before_flush - cursor_after_flush;
  }
  const bool cursor_accounting_ok =
      in_partial_replay
          ? (out_stats.cursor_after_skip >= out_stats.cursor_after_flush &&
             out_stats.cursor_after_skip - out_stats.cursor_after_flush ==
                 skipped)
          : (cursor_after_skip >= cursor_after_flush &&
             cursor_after_skip - cursor_after_flush == skipped);
  if (!cursor_accounting_ok ||
      out_stats.warp_insts != skipped ||
      tw->get_pc() != region.exit_pc) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id
              << "] cyc=" << get_current_gpu_cycle()
              << " reason=poll-cursor-accounting"
              << " block=" << region.block_id << " warp=" << warp_id
              << " expected_exit_pc=0x" << format_hex_u64(region.exit_pc)
              << " actual_landing_pc=0x" << format_hex_u64(tw->get_pc())
              << " cursor_before_flush=" << cursor_before_flush
              << " cursor_after_flush=" << cursor_after_flush
              << " prefetched_rollback=" << out_stats.prefetched_rollback
              << " cursor_after_skip=" << cursor_after_skip
              << " replay_cursor_after_skip="
              << out_stats.cursor_after_skip
              << " source="
              << (in_partial_replay ? "partial-replay" : "canonical")
              << " skipped=" << skipped << " action=abort" << std::endl;
    abort();
  }
  if (in_partial_replay && skipped != 0) {
    note_partial_replay_progress(
        warp_id, tw->partial_replay_remaining(), tw->get_pc(), skipped);
  }
  return skipped;
}

void SM::note_partial_replay_progress(
    unsigned int warp_id, unsigned int remaining, uint64_t pc,
    unsigned int advanced_occurrences) {
  if (warp_id >= m_partial_replay_progress.size()) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=partial-replay-progress-warp-range"
              << " warp=" << warp_id << " action=abort" << std::endl;
    abort();
  }
  HopperPartialReplayProgress &progress =
      m_partial_replay_progress[warp_id];
  if (advanced_occurrences == 0 || !progress.valid ||
      remaining > progress.last_remaining ||
      progress.last_remaining - remaining != advanced_occurrences ||
      progress.progressed_occurrences + advanced_occurrences >
          progress.initial_remaining) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=partial-replay-progress-accounting"
              << " warp=" << warp_id
              << " valid=" << progress.valid
              << " previous_remaining=" << progress.last_remaining
              << " remaining=" << remaining
              << " advanced=" << advanced_occurrences
              << " pc=0x" << format_hex_u64(pc)
              << " action=abort" << std::endl;
    abort();
  }
  progress.last_remaining = remaining;
  progress.last_pc = pc;
  progress.progressed_occurrences += advanced_occurrences;
  progress.last_progress_cycle = get_current_gpu_cycle();
  progress.last_heartbeat_cycle = progress.last_progress_cycle;
}

void SM::service_partial_replay_progress(unsigned int warp_id) {
  if (warp_id >= m_partial_replay_progress.size()) {
    return;
  }
  HopperPartialReplayProgress &progress =
      m_partial_replay_progress[warp_id];
  if (!progress.valid) {
    return;
  }

  const uint64_t now = get_current_gpu_cycle();
  if (m_physical_warp[warp_id]->get_cta_id() != progress.cta_id) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
              << " reason=partial-replay-progress-cta-mismatch"
              << " trace_kernel=" << progress.trace_kernel_id
              << " cta=" << progress.cta_id
              << " actual_cta="
              << m_physical_warp[warp_id]->get_cta_id()
              << " warp=" << warp_id << " action=abort" << std::endl;
    abort();
  }

  HopperPartialWaitArm &arm = m_partial_wait_arms[warp_id];
  HopperMBarrierPendingWait &pending_wait =
      m_pending_sync_waits[warp_id];
  if (arm.valid || pending_wait.valid) {
    if (arm.valid && !arm.source_was_deferred_replay) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                << " reason=partial-replay-overlapped-by-canonical-arm"
                << " owner_region=" << progress.region_id
                << " nested_region=" << arm.record.partial_region_id
                << " warp=" << warp_id << " action=abort" << std::endl;
      abort();
    }
    // The nested partial or mbarrier hold owns forward-progress checking until
    // it releases. Its transition/hold watchdog remains active independently.
    progress.last_progress_cycle = now;
    progress.last_heartbeat_cycle = now;
    return;
  }

  trace_shd_warp_t *tw =
      static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id]);
  const unsigned int remaining = tw->partial_replay_remaining();
  const bool replay_active = tw->partial_replay_active();
  if (remaining != progress.last_remaining ||
      progress.progressed_occurrences + remaining !=
          progress.initial_remaining ||
      (replay_active &&
       tw->partial_replay_origin() !=
           WaitBlockTraceOrigin::PARTIAL_BLOCKED_REPLAY) ||
      (!replay_active && remaining != 0)) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
              << " reason=partial-replay-progress-state-drift"
              << " block=" << progress.block_id
              << " region=" << progress.region_id
              << " warp=" << warp_id
              << " replay_active=" << replay_active
              << " expected_remaining=" << progress.last_remaining
              << " actual_remaining=" << remaining
              << " initial=" << progress.initial_remaining
              << " progressed=" << progress.progressed_occurrences
              << " action=abort" << std::endl;
    abort();
  }

  const uint64_t no_progress_cycles = now - progress.last_progress_cycle;
  if (now - progress.last_heartbeat_cycle >=
      kWaitBlockHeartbeatCycles) {
    debug_log_wait_block_event(
        "partial-replay-heartbeat",
        "block=" + progress.block_id +
            " region=" + progress.region_id +
            " trace_kernel=" +
            std::to_string(progress.trace_kernel_id) +
            " function=" +
            std::to_string(progress.unique_function_id) +
            " cta=" + std::to_string(progress.cta_id) +
            " warp=" + std::to_string(warp_id) +
            " remaining=" + std::to_string(remaining) +
            " progressed=" +
            std::to_string(progress.progressed_occurrences) +
            " last_pc=0x" + format_hex_u64(progress.last_pc) +
            " expected_post_join_pc=0x" +
            format_hex_u64(progress.post_join_pc) +
            " age=" + std::to_string(now - progress.start_cycle) +
            " no_progress_cycles=" +
            std::to_string(no_progress_cycles));
    progress.last_heartbeat_cycle = now;
  }
  if (no_progress_cycles > kWaitBlockHoldWatchdogCycles) {
    m_partial_replay_watchdog_aborts_total++;
    m_wait_block_watchdog_aborts_total++;
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
              << " reason=partial-replay-watchdog"
              << " block=" << progress.block_id
              << " region=" << progress.region_id
              << " trace_kernel=" << progress.trace_kernel_id
              << " function=" << progress.unique_function_id
              << " cta=" << progress.cta_id
              << " global_cta="
              << m_local_to_global_cta_id[progress.cta_id]
              << " warp=" << warp_id
              << " remaining=" << remaining
              << " initial=" << progress.initial_remaining
              << " progressed=" << progress.progressed_occurrences
              << " last_pc=0x" << format_hex_u64(progress.last_pc)
              << " expected_post_join_pc=0x"
              << format_hex_u64(progress.post_join_pc)
              << " age=" << now - progress.start_cycle
              << " no_progress_cycles=" << no_progress_cycles
              << " threshold=" << kWaitBlockHoldWatchdogCycles
              << " action=abort" << std::endl;
    abort();
  }
}

void SM::start_partial_wait_hold(unsigned int warp_id,
                                 HopperPartialWaitArm &arm) {
  if (!arm.valid || arm.mode == PartialWaitArmMode::NONE ||
      warp_id >= m_pending_sync_waits.size() ||
      m_pending_sync_waits[warp_id].valid ||
      m_partial_joined_regions[warp_id].count(
          arm.record.partial_region_id) == 0) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=invalid-partial-hold-transition"
              << " warp=" << warp_id
              << " block=" << arm.record.block_id
              << " region=" << arm.record.partial_region_id
              << " action=abort" << std::endl;
    abort();
  }

  trace_shd_warp_t *tw =
      static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id]);
  tw->get_IBuffer_remodeled()->flush(false);

  HopperMBarrierPendingWait held;
  held.valid = true;
  held.is_trywait =
      arm.record.entry_probe_kind == SyncInstructionKind::TRYWAIT;
  held.key = arm.key;
  held.wait_state_raw = arm.wait_state_raw;
  held.mode = WaitBlockPendingMode::PARTIAL_MISSING_WAIT_HOLD;
  held.block_id = arm.record.block_id;
  held.action_pc = arm.record.entry_probe_pc;
  held.unique_function_id = arm.unique_function_id;
  held.expected_gen = arm.expected_gen;
  held.uses_exact_generation_ticket =
      arm.uses_exact_generation_ticket;
  held.hold_start_cycle = get_current_gpu_cycle();
  held.last_heartbeat_cycle = held.hold_start_cycle;
  m_pending_sync_waits[warp_id] = held;
  arm.mode = PartialWaitArmMode::HOLDING;
  m_missing_wait_holds_total++;
  m_partial_join_holds_total++;
  debug_log_wait_block_event(
      "partial-join-hold",
      "block=" + arm.record.block_id +
          " region=" + arm.record.partial_region_id +
          " trace_kernel=" + std::to_string(arm.key.trace_kernel_id) +
          " function=" + std::to_string(held.unique_function_id) +
          " cta=" + std::to_string(arm.key.cta_id) +
          " warp=" + std::to_string(warp_id) +
          " entry_pc=0x" + format_hex_u64(arm.record.entry_probe_pc) +
          " join_pc=0x" + format_hex_u64(arm.record.join_pc) +
          " blocked_mask=0x" + format_hex_u64(arm.blocked_mask) +
          " sibling_mask=0x" + format_hex_u64(arm.sibling_mask) +
          " expected_gen=" + std::to_string(arm.expected_gen) +
          " current_gen=" + std::to_string(sync_barrier_gen(arm.key)) +
          " deferred=" +
          std::to_string(tw->partial_deferred_size()));
}

void SM::handle_partial_wait_instruction_issue(const warp_inst_t &inst,
                                               unsigned int warp_id) {
  if (!m_config->is_trace_mode ||
      warp_id >= m_partial_wait_arms.size() ||
      warp_id >= m_partial_joined_regions.size()) {
    return;
  }

  PartialWaitRegionRecord join_region;
  if (m_gpu->lookup_partial_wait_join(
          inst.trace_kernel_id, inst.unique_function_id, inst.pc,
          join_region)) {
    const uint64_t issued_mask = inst.get_active_mask().to_ullong();
    if (join_region.sibling_masks.count(issued_mask) != 0) {
      if (!m_partial_joined_regions[warp_id]
               .insert(join_region.region_id)
               .second) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=duplicate-partial-sibling-join"
                  << " region=" << join_region.region_id
                  << " warp=" << warp_id << " action=abort" << std::endl;
        abort();
      }
      debug_log_wait_block_event(
          "partial-sibling-join",
          "region=" + join_region.region_id +
              " trace_kernel=" + std::to_string(inst.trace_kernel_id) +
              " function=" + std::to_string(inst.unique_function_id) +
              " cta=" +
              std::to_string(m_physical_warp[warp_id]->get_cta_id()) +
              " warp=" + std::to_string(warp_id) +
              " join_pc=0x" + format_hex_u64(inst.pc) +
              " sibling_mask=0x" + format_hex_u64(issued_mask));
    }
  }

  HopperPartialWaitArm &arm = m_partial_wait_arms[warp_id];
  if (arm.valid) {
    if (inst.trace_kernel_id != arm.key.trace_kernel_id ||
        inst.unique_function_id != arm.unique_function_id) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=partial-arm-identity-mismatch"
                << " block=" << arm.record.block_id
                << " warp=" << warp_id
                << " trace_kernel=" << inst.trace_kernel_id
                << " function=" << inst.unique_function_id
                << " action=abort" << std::endl;
      abort();
    }
    if (arm.mode == PartialWaitArmMode::ARMED) {
      if (inst.pc == arm.record.entry_probe_pc) {
        return;
      }
      const uint64_t arming_owner =
          static_cast<uint64_t>(inst.trace_active_mask) &
          (arm.blocked_mask | arm.sibling_mask);
      if (!inst.trace_masks_valid ||
          (arming_owner != 0 &&
           (arming_owner & ~arm.blocked_mask) != 0)) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=partial-arming-mask-escape"
                  << " block=" << arm.record.block_id
                  << " warp=" << warp_id << " pc=0x"
                  << format_hex_u64(inst.pc)
                  << " trace_active_mask=0x"
                  << format_hex_u64(inst.trace_active_mask)
                  << " blocked_mask=0x"
                  << format_hex_u64(arm.blocked_mask)
                  << " action=abort" << std::endl;
        abort();
      }
      if (inst.pc != arm.record.entry_branch_pc) {
        if (std::find(arm.record.entry_arming_pcs.begin(),
                      arm.record.entry_arming_pcs.end(),
                      static_cast<uint64_t>(inst.pc)) ==
            arm.record.entry_arming_pcs.end()) {
          std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                    << get_current_gpu_cycle()
                    << " reason=partial-arming-path-escape"
                    << " block=" << arm.record.block_id
                    << " warp=" << warp_id << " pc=0x"
                    << format_hex_u64(inst.pc)
                    << " expected_branch_pc=0x"
                    << format_hex_u64(arm.record.entry_branch_pc)
                    << " action=abort" << std::endl;
          abort();
        }
        return;
      }

      if (m_partial_joined_regions[warp_id].count(
              arm.record.partial_region_id) != 0) {
        m_partial_join_already_issued_total++;
        start_partial_wait_hold(warp_id, arm);
        return;
      }

      trace_shd_warp_t *tw =
          static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id]);
      const unsigned int cursor_before_flush = tw->used_insts;
      tw->get_IBuffer_remodeled()->flush(false);
      const unsigned int cursor_after_flush = tw->used_insts;
      if (cursor_after_flush > cursor_before_flush) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=partial-prefetch-rollback-direction"
                  << " block=" << arm.record.block_id
                  << " warp=" << warp_id
                  << " cursor_before_flush=" << cursor_before_flush
                  << " cursor_after_flush=" << cursor_after_flush
                  << " action=abort" << std::endl;
        abort();
      }
      PartialTraceExtractStats stats;
      std::string failure;
      if (!tw->extract_partial_wait_capsule(
              arm.blocked_mask, arm.sibling_mask,
              arm.blocked_mask | arm.sibling_mask,
              arm.unique_function_id,
              static_cast<address_type>(arm.record.captured_ready_pc),
              static_cast<address_type>(arm.record.join_pc),
              static_cast<address_type>(arm.record.post_join_pc),
              &stats, &failure)) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=partial-capsule-extraction-failed"
                  << " detail=" << failure
                  << " block=" << arm.record.block_id
                  << " region=" << arm.record.partial_region_id
                  << " warp=" << warp_id
                  << " source_pc=0x"
                  << format_hex_u64(tw->canonical_trace_pc())
                  << " expected_ready_pc=0x"
                  << format_hex_u64(arm.record.captured_ready_pc)
                  << " action=abort" << std::endl;
        abort();
      }
      if (stats.cursor_before != cursor_after_flush ||
          stats.cursor_after < stats.cursor_before ||
          stats.cursor_after - stats.cursor_before !=
              stats.blocked_occurrences + stats.sibling_join_occurrences) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=partial-extraction-cursor-accounting"
                  << " block=" << arm.record.block_id
                  << " warp=" << warp_id
                  << " cursor_before_flush=" << cursor_before_flush
                  << " cursor_after_flush=" << cursor_after_flush
                  << " cursor_after_extract=" << stats.cursor_after
                  << " blocked_occurrences=" << stats.blocked_occurrences
                  << " sibling_join_occurrences="
                  << stats.sibling_join_occurrences
                  << " action=abort" << std::endl;
        abort();
      }
      arm.mode = PartialWaitArmMode::WAITING_FOR_SIBLING_JOIN;
      arm.extracted_capsule = true;
      m_partial_capsules_total++;
      m_partial_deferred_occurrences_total += stats.blocked_occurrences;
      debug_log_wait_block_event(
          "partial-extract",
          "block=" + arm.record.block_id +
              " region=" + arm.record.partial_region_id +
              " warp=" + std::to_string(warp_id) +
              " cursor_before_flush=" +
              std::to_string(cursor_before_flush) +
              " cursor_after_flush=" +
              std::to_string(cursor_after_flush) +
              " prefetched_rollback=" +
              std::to_string(cursor_before_flush - cursor_after_flush) +
              " cursor_after_extract=" +
              std::to_string(stats.cursor_after) +
              " blocked_occurrences=" +
              std::to_string(stats.blocked_occurrences) +
              " sibling_join_occurrences=" +
              std::to_string(stats.sibling_join_occurrences) +
              " landing_pc=0x" +
              format_hex_u64(tw->canonical_trace_pc()));
      return;
    }

    if (arm.mode == PartialWaitArmMode::WAITING_FOR_SIBLING_JOIN &&
        inst.pc == arm.record.join_pc &&
        inst.get_active_mask().to_ullong() == arm.sibling_mask) {
      if (m_partial_joined_regions[warp_id].count(
              arm.record.partial_region_id) == 0) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=partial-join-issue-not-recorded"
                  << " block=" << arm.record.block_id
                  << " warp=" << warp_id << " action=abort" << std::endl;
        abort();
      }
      start_partial_wait_hold(warp_id, arm);
      return;
    }
  }

  PartialWaitRegionRecord post_region;
  if (m_gpu->lookup_partial_wait_post_join(
          inst.trace_kernel_id, inst.unique_function_id, inst.pc,
          post_region)) {
    const uint64_t issued_mask = inst.get_active_mask().to_ullong();
    if (post_region.live_masks.count(issued_mask) == 0) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=partial-post-join-mask-mismatch"
                << " region=" << post_region.region_id
                << " warp=" << warp_id
                << " mask=0x" << format_hex_u64(issued_mask)
                << " action=abort" << std::endl;
      abort();
    }
    trace_shd_warp_t *tw =
        static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id]);
    if (tw->partial_capsule_active()) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=post-join-with-active-partial-capsule"
                << " region=" << post_region.region_id
                << " warp=" << warp_id << " action=abort" << std::endl;
      abort();
    }
    HopperPartialReplayProgress &progress =
        m_partial_replay_progress[warp_id];
    const bool replay_region_matches =
        progress.valid && progress.region_id == post_region.region_id;
    const bool replay_pc_matches =
        progress.valid && progress.post_join_pc == inst.pc;
    const bool closes_replay_progress =
        replay_region_matches && replay_pc_matches;
    if (progress.valid &&
        (replay_region_matches != replay_pc_matches ||
         (closes_replay_progress &&
          (progress.trace_kernel_id != inst.trace_kernel_id ||
           progress.unique_function_id != inst.unique_function_id ||
           progress.cta_id !=
               m_physical_warp[warp_id]->get_cta_id() ||
           progress.last_remaining != 0 ||
           progress.progressed_occurrences !=
               progress.initial_remaining)))) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=partial-replay-post-join-accounting"
                << " owner_region=" << progress.region_id
                << " observed_region=" << post_region.region_id
                << " warp=" << warp_id
                << " expected_post_join_pc=0x"
                << format_hex_u64(progress.post_join_pc)
                << " actual_post_join_pc=0x"
                << format_hex_u64(inst.pc)
                << " initial=" << progress.initial_remaining
                << " remaining=" << progress.last_remaining
                << " progressed=" << progress.progressed_occurrences
                << " action=abort" << std::endl;
      abort();
    }
    if (m_partial_joined_regions[warp_id].erase(post_region.region_id) != 1) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=post-join-without-sibling-join"
                << " region=" << post_region.region_id
                << " warp=" << warp_id << " action=abort" << std::endl;
      abort();
    }
    debug_log_wait_block_event(
        "partial-region-close",
        "region=" + post_region.region_id +
            " trace_kernel=" + std::to_string(inst.trace_kernel_id) +
            " function=" + std::to_string(inst.unique_function_id) +
            " cta=" +
            std::to_string(m_physical_warp[warp_id]->get_cta_id()) +
            " warp=" + std::to_string(warp_id) +
            " post_join_pc=0x" + format_hex_u64(inst.pc) +
            " replay_progress_closed=" +
            std::to_string(closes_replay_progress) +
            " replay_occurrences=" +
            std::to_string(closes_replay_progress
                               ? progress.progressed_occurrences
                               : 0));
    if (closes_replay_progress) {
      progress = HopperPartialReplayProgress();
    }
  }
}

void SM::service_wait_block_release(unsigned warp_id) {
  if (!m_config->wait_block_enable ||
      warp_id >= m_pending_sync_waits.size()) {
    return;
  }
  service_partial_replay_progress(warp_id);
  HopperPartialWaitArm &partial_arm = m_partial_wait_arms[warp_id];
  if (partial_arm.valid &&
      partial_arm.mode != PartialWaitArmMode::HOLDING) {
    const uint64_t now = get_current_gpu_cycle();
    const uint64_t age = now - partial_arm.arm_start_cycle;
    if (now - partial_arm.last_heartbeat_cycle >=
        kWaitBlockHeartbeatCycles) {
      debug_log_wait_block_event(
          "partial-transition-heartbeat",
          "block=" + partial_arm.record.block_id +
              " region=" + partial_arm.record.partial_region_id +
              " warp=" + std::to_string(warp_id) +
              " mode=" +
              std::to_string(static_cast<int>(partial_arm.mode)) +
              " age=" + std::to_string(age));
      partial_arm.last_heartbeat_cycle = now;
    }
    if (age > kWaitBlockHoldWatchdogCycles) {
      m_wait_block_watchdog_aborts_total++;
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                << " reason=partial-transition-watchdog"
                << " block=" << partial_arm.record.block_id
                << " region=" << partial_arm.record.partial_region_id
                << " warp=" << warp_id
                << " mode=" << static_cast<int>(partial_arm.mode)
                << " age=" << age << " action=abort" << std::endl;
      abort();
    }
  }
  HopperMBarrierPendingWait &pending_wait = m_pending_sync_waits[warp_id];
  if (partial_arm.valid &&
      partial_arm.mode == PartialWaitArmMode::HOLDING &&
      !pending_wait.valid) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=partial-hold-lost-pending-state"
              << " block=" << partial_arm.record.block_id
              << " warp=" << warp_id << " action=abort" << std::endl;
    abort();
  }
  if (!pending_wait.valid) {
    return;
  }
  if (pending_wait.mode == WaitBlockPendingMode::PARTIAL_MISSING_WAIT_HOLD &&
      (!partial_arm.valid ||
       partial_arm.mode != PartialWaitArmMode::HOLDING)) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=partial-pending-hold-lost-arm-state"
              << " warp=" << warp_id << " action=abort" << std::endl;
    abort();
  }
  if (pending_wait.mode == WaitBlockPendingMode::NONE ||
      pending_wait.block_id.empty()) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle() << " warp=" << warp_id
              << " reason=invalid-pending-hold-state" << std::endl;
    abort();
  }

  const uint64_t now = get_current_gpu_cycle();
  const uint64_t cur_gen = sync_barrier_gen(pending_wait.key);
  const uint64_t exp_gen = pending_wait.expected_gen;
  const uint64_t held_cycles = now - pending_wait.hold_start_cycle;
  const std::string &block = pending_wait.block_id;
  std::ostringstream barrier_detail_stream;
  auto barrier_it = m_hopper_mbarriers.find(pending_wait.key);
  if (barrier_it == m_hopper_mbarriers.end()) {
    barrier_detail_stream << " barrier_state=unseen";
  } else {
    const HopperMBarrierObject &barrier = barrier_it->second;
    barrier_detail_stream
        << " barrier_state=present phase=" << barrier.phase
        << " arrive=" << barrier.arrive_count << "/"
        << barrier.expected_arrive_count << " tx="
        << barrier.completed_tx_bytes << "/" << barrier.expected_tx_bytes
        << " assigned=" << barrier.assigned_tx_bytes
        << " outstanding=" << barrier.outstanding_tma_bindings
        << " contract_id=" << barrier.tx_contract_id
        << " prev_flip_cycle=" << barrier.prev_flip_cycle
        << " last_flip_cycle=" << barrier.last_flip_cycle;
  }
  const std::string barrier_detail = barrier_detail_stream.str();

  // Once an exact target has completed, the live barrier object may already
  // describe a later generation. Its contract cannot invalidate the completed
  // dependency represented by this ticket.
  if (barrier_it != m_hopper_mbarriers.end() &&
      (!pending_wait.uses_exact_generation_ticket ||
       cur_gen < exp_gen)) {
    const HopperMBarrierObject &barrier = barrier_it->second;
    const bool missing_assignment =
        barrier.expected_tx_bytes > barrier.completed_tx_bytes &&
        barrier.assigned_tx_bytes < barrier.expected_tx_bytes &&
        !barrier.assignment_open;
    const bool missing_in_flight_binding =
        barrier.expected_tx_bytes > barrier.completed_tx_bytes &&
        barrier.assigned_tx_bytes == barrier.expected_tx_bytes &&
        barrier.outstanding_tma_bindings == 0;
    if (missing_assignment || missing_in_flight_binding) {
      ++m_tma_contract_mismatches;
      std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc=" << now
                << " reason=unfulfillable-tx-contract"
                << " trace_kernel=" << pending_wait.key.trace_kernel_id
                << " cta_slot=" << pending_wait.key.cta_id
                << " warp=" << warp_id << " barrier=0x"
                << format_hex_u64(pending_wait.key.barrier_addr)
                << " generation=" << barrier.gen
                << " contract_id=" << barrier.tx_contract_id
                << " assignment_open=" << barrier.assignment_open
                << " expected=" << barrier.expected_tx_bytes
                << " assigned=" << barrier.assigned_tx_bytes
                << " completed=" << barrier.completed_tx_bytes
                << " outstanding=" << barrier.outstanding_tma_bindings
                << std::endl;
      debug_dump_tma_binding_snapshot("unfulfillable-tx-contract");
      abort();
    }
  }

  if (cur_gen > exp_gen) {
    if (pending_wait.uses_exact_generation_ticket) {
      m_wait_block_late_arrivals_total++;
      debug_log_wait_block_event(
          "exact-generation-late-release",
          "strategy=" +
              std::string(wait_block_mode_to_string(pending_wait.mode)) +
              " block=" + block +
              " trace_kernel=" +
              std::to_string(pending_wait.key.trace_kernel_id) +
              " function=" +
              std::to_string(pending_wait.unique_function_id) +
              " cta=" + std::to_string(pending_wait.key.cta_id) +
              " warp=" + std::to_string(warp_id) +
              " action_pc=0x" +
              format_hex_u64(pending_wait.action_pc) +
              " barrier=0x" +
              format_hex_u64(pending_wait.key.barrier_addr) +
              " target_gen=" + std::to_string(exp_gen) +
              " current_gen=" + std::to_string(cur_gen) +
              " held_cycles=" + std::to_string(held_cycles));
    } else {
      m_wait_block_generation_overshoots_total++;
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                << " reason=generation-overshoot"
                << " strategy="
                << wait_block_mode_to_string(pending_wait.mode)
                << " block=" << block
                << " trace_kernel=" << pending_wait.key.trace_kernel_id
                << " function=" << pending_wait.unique_function_id
                << " cta=" << pending_wait.key.cta_id
                << " global_cta="
                << m_local_to_global_cta_id[pending_wait.key.cta_id]
                << " warp=" << warp_id << " action_pc=0x"
                << format_hex_u64(pending_wait.action_pc)
                << " barrier=0x"
                << format_hex_u64(pending_wait.key.barrier_addr)
                << " expected_gen=" << exp_gen
                << " current_gen=" << cur_gen
                << " held_cycles=" << held_cycles
                << barrier_detail
                << " action=abort" << std::endl;
      debug_dump_tma_binding_snapshot("waitblock-generation-overshoot");
      abort();
    }
  }

  if (cur_gen < exp_gen) {
    if (m_config->wait_block_enable &&
        now - pending_wait.last_heartbeat_cycle >=
            kWaitBlockHeartbeatCycles) {
      debug_log_wait_block_event(
          "heartbeat",
          "strategy=" +
              std::string(wait_block_mode_to_string(pending_wait.mode)) +
              " block=" + block +
              " trace_kernel=" +
              std::to_string(pending_wait.key.trace_kernel_id) +
              " function=" +
              std::to_string(pending_wait.unique_function_id) +
              " cta=" + std::to_string(pending_wait.key.cta_id) +
              " warp=" + std::to_string(warp_id) + " action_pc=0x" +
              format_hex_u64(pending_wait.action_pc) + " barrier=0x" +
              format_hex_u64(pending_wait.key.barrier_addr) +
              " expected_gen=" + std::to_string(exp_gen) +
              " current_gen=" + std::to_string(cur_gen) +
              " held_cycles=" + std::to_string(held_cycles) +
              barrier_detail);
      pending_wait.last_heartbeat_cycle = now;
    }
    const uint64_t threshold = kWaitBlockHoldWatchdogCycles;
    if (held_cycles > threshold) {
      m_wait_block_watchdog_aborts_total++;
      if (pending_wait.mode == WaitBlockPendingMode::CAPTURED_POLL_HOLD ||
          pending_wait.mode == WaitBlockPendingMode::FOLD_AUX_WAIT_HOLD) {
        m_fold_watchdog_hits++;
      }
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                << " reason=hold-watchdog"
                << " strategy=" << wait_block_mode_to_string(pending_wait.mode)
                << " block=" << block
                << " trace_kernel=" << pending_wait.key.trace_kernel_id
                << " function=" << pending_wait.unique_function_id
                << " cta=" << pending_wait.key.cta_id
                << " global_cta="
                << m_local_to_global_cta_id[pending_wait.key.cta_id]
                << " warp=" << warp_id << " action_pc=0x"
                << format_hex_u64(pending_wait.action_pc)
                << " barrier=0x"
                << format_hex_u64(pending_wait.key.barrier_addr)
                << " wait_state=0x"
                << format_hex_u64(pending_wait.wait_state_raw)
                << " expected_gen=" << exp_gen
                << " current_gen=" << cur_gen
                << " held_cycles=" << held_cycles
                << " threshold=" << threshold << barrier_detail
                << " action=abort"
                << std::endl;
      debug_dump_tma_binding_snapshot("waitblock-hold-watchdog");
      abort();
    }
    return;
  }

  if (!is_wait_block_pending_satisfied(pending_wait)) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
              << " reason=generation-ready-state-mismatch"
              << " strategy=" << wait_block_mode_to_string(pending_wait.mode)
              << " block=" << block << " warp=" << warp_id
              << " expected_gen=" << exp_gen << " current_gen=" << cur_gen
              << " wait_state=0x"
              << format_hex_u64(pending_wait.wait_state_raw)
              << " action=abort" << std::endl;
    abort();
  }

  m_sync_debug_wait_released++;
  if (pending_wait.mode == WaitBlockPendingMode::PARTIAL_MISSING_WAIT_HOLD) {
    HopperPartialWaitArm &arm = m_partial_wait_arms[warp_id];
    if (!arm.valid || arm.mode != PartialWaitArmMode::HOLDING ||
        arm.record.block_id != pending_wait.block_id ||
        arm.expected_gen != pending_wait.expected_gen) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                << " reason=partial-release-state-mismatch"
                << " block=" << block << " warp=" << warp_id
                << " action=abort" << std::endl;
      abort();
    }

    trace_shd_warp_t *tw =
        static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id]);
    HopperPartialReplayProgress &replay_progress =
        m_partial_replay_progress[warp_id];
    tw->get_IBuffer_remodeled()->flush(false);
    if (arm.extracted_capsule) {
      if (replay_progress.valid) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                  << " reason=partial-replay-owner-already-active"
                  << " owner_region=" << replay_progress.region_id
                  << " new_region=" << arm.record.partial_region_id
                  << " block=" << block << " warp=" << warp_id
                  << " action=abort" << std::endl;
        abort();
      }
      if (!tw->activate_partial_blocked_replay()) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                  << " reason=partial-deferred-replay-activation-failed"
                  << " block=" << block << " warp=" << warp_id
                  << " deferred=" << tw->partial_deferred_size()
                  << " action=abort" << std::endl;
        abort();
      }
      const unsigned int replay_remaining =
          tw->partial_replay_remaining();
      if (replay_remaining == 0) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                  << " reason=partial-replay-activated-empty"
                  << " block=" << block << " warp=" << warp_id
                  << " action=abort" << std::endl;
        abort();
      }
      replay_progress.valid = true;
      replay_progress.block_id = arm.record.block_id;
      replay_progress.region_id = arm.record.partial_region_id;
      replay_progress.trace_kernel_id =
          pending_wait.key.trace_kernel_id;
      replay_progress.unique_function_id =
          pending_wait.unique_function_id;
      replay_progress.cta_id = pending_wait.key.cta_id;
      replay_progress.start_cycle = now;
      replay_progress.last_heartbeat_cycle = now;
      replay_progress.last_progress_cycle = now;
      replay_progress.last_pc = arm.record.join_pc;
      replay_progress.post_join_pc = arm.record.post_join_pc;
      replay_progress.initial_remaining = replay_remaining;
      replay_progress.last_remaining = replay_remaining;
    } else if (arm.source_was_deferred_replay) {
      if (!replay_progress.valid ||
          replay_progress.trace_kernel_id !=
              pending_wait.key.trace_kernel_id ||
          replay_progress.unique_function_id !=
              pending_wait.unique_function_id ||
          replay_progress.cta_id != pending_wait.key.cta_id ||
          !tw->partial_replay_active() ||
          tw->partial_replay_origin() !=
              WaitBlockTraceOrigin::PARTIAL_BLOCKED_REPLAY ||
          replay_progress.last_remaining !=
              tw->partial_replay_remaining()) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                  << " reason=partial-nested-replay-state-lost"
                  << " block=" << block << " warp=" << warp_id
                  << " action=abort" << std::endl;
        abort();
      }
      replay_progress.last_progress_cycle = now;
      replay_progress.last_heartbeat_cycle = now;
    } else if (tw->partial_capsule_active()) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                << " reason=unexpected-partial-capsule-on-direct-release"
                << " block=" << block << " warp=" << warp_id
                << " action=abort" << std::endl;
      abort();
    } else if (replay_progress.valid) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
                << " reason=canonical-partial-release-with-replay-owner"
                << " owner_region=" << replay_progress.region_id
                << " block=" << block << " warp=" << warp_id
                << " action=abort" << std::endl;
      abort();
    }

    m_missing_wait_releases_total++;
    m_partial_join_releases_total++;
    debug_log_wait_block_event(
        "partial-join-release",
        "block=" + block +
            " region=" + arm.record.partial_region_id +
            " trace_kernel=" +
            std::to_string(pending_wait.key.trace_kernel_id) +
            " function=" + std::to_string(pending_wait.unique_function_id) +
            " cta=" + std::to_string(pending_wait.key.cta_id) +
            " warp=" + std::to_string(warp_id) +
            " entry_pc=0x" + format_hex_u64(pending_wait.action_pc) +
            " expected_gen=" + std::to_string(exp_gen) +
            " current_gen=" + std::to_string(cur_gen) +
            " held_cycles=" + std::to_string(held_cycles) +
            " replay_remaining=" +
            std::to_string(tw->partial_replay_remaining()));
    arm = HopperPartialWaitArm();
    pending_wait = HopperMBarrierPendingWait();
    return;
  }

  if (pending_wait.mode == WaitBlockPendingMode::MISSING_WAIT_HOLD) {
    m_missing_wait_releases_total++;
    debug_log_wait_block_event(
        "missing-wait-release",
        "block=" + block +
            " trace_kernel=" +
            std::to_string(pending_wait.key.trace_kernel_id) +
            " function=" + std::to_string(pending_wait.unique_function_id) +
            " cta=" + std::to_string(pending_wait.key.cta_id) +
            " warp=" + std::to_string(warp_id) + " entry_pc=0x" +
            format_hex_u64(pending_wait.action_pc) + " barrier=0x" +
            format_hex_u64(pending_wait.key.barrier_addr) +
            " expected_gen=" + std::to_string(exp_gen) +
            " current_gen=" + std::to_string(cur_gen) +
            " held_cycles=" + std::to_string(held_cycles) +
            " cursor_action=none");
    pending_wait = HopperMBarrierPendingWait();
    return;
  }

  if (pending_wait.mode == WaitBlockPendingMode::FOLD_AUX_WAIT_HOLD) {
    m_fold_aux_releases_total++;
    debug_log_wait_block_event(
        "fold-aux-wait-release",
        "block=" + block +
            " trace_kernel=" +
            std::to_string(pending_wait.key.trace_kernel_id) +
            " function=" + std::to_string(pending_wait.unique_function_id) +
            " cta=" + std::to_string(pending_wait.key.cta_id) +
            " warp=" + std::to_string(warp_id) + " wait_pc=0x" +
            format_hex_u64(pending_wait.action_pc) + " barrier=0x" +
            format_hex_u64(pending_wait.key.barrier_addr) +
            " target_gen=" + std::to_string(exp_gen) +
            " current_gen=" + std::to_string(cur_gen) +
            " held_cycles=" + std::to_string(held_cycles) +
            " cursor_action=none");
    pending_wait = HopperMBarrierPendingWait();
    return;
  }

  if (pending_wait.mode != WaitBlockPendingMode::CAPTURED_POLL_HOLD) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
              << " reason=unknown-pending-mode warp=" << warp_id
              << " action=abort" << std::endl;
    abort();
  }

  SpinWaitRegionRecord region;
  if (!m_gpu->lookup_spin_wait_region(
          pending_wait.key.trace_kernel_id, pending_wait.unique_function_id,
          pending_wait.action_pc, region) ||
      !region.foldable || region.exit_pc != pending_wait.poll_loop_exit_pc ||
      (m_config->wait_block_enable &&
       region.block_id != pending_wait.block_id)) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc=" << now
              << " reason=captured-poll-sidecar-runtime-drift"
              << " block=" << block << " warp=" << warp_id
              << " poll_pc=0x" << format_hex_u64(pending_wait.action_pc)
              << " expected_exit_pc=0x"
              << format_hex_u64(pending_wait.poll_loop_exit_pc)
              << " action=abort" << std::endl;
    abort();
  }

  trace_shd_warp_t *tw =
      static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id]);
  FoldSkipStats skip_stats;
  const unsigned skipped =
      fold_cursor_skip_to_exit(warp_id, region, skip_stats);

  if (m_fold_pace_by_warp.size() <= warp_id) {
    m_fold_pace_by_warp.resize(MAX_WARP_PER_SHADER);
  }
  FoldPaceStats &pace =
      m_fold_pace_by_warp[warp_id]
                         [FoldPaceKey(pending_wait.key,
                                      pending_wait.action_pc)];
  pace.releases++;
  pace.park_cycles_total += held_cycles;
  if (held_cycles > pace.park_cycles_max) {
    pace.park_cycles_max = held_cycles;
  }
  pace.last_end_cycle = now;
  pace.last_expected_gen = exp_gen;
  pace.last_barrier_gen = cur_gen;
  m_fold_releases_total++;
  m_fold_skipped_insts_total += skipped;
  m_fold_park_cycles_total += held_cycles;
  if (held_cycles > m_fold_park_cycles_max) {
    m_fold_park_cycles_max = held_cycles;
  }

  std::ostringstream per_pc;
  bool first_pc = true;
  for (const auto &entry : skip_stats.pc_occurrences) {
    if (!first_pc) {
      per_pc << ",";
    }
    first_pc = false;
    per_pc << "0x" << format_hex_u64(entry.first) << ":" << entry.second;
  }
  debug_log_wait_block_event(
      "poll-advance",
      "block=" + block +
          " trace_kernel=" +
          std::to_string(pending_wait.key.trace_kernel_id) +
          " function=" + std::to_string(pending_wait.unique_function_id) +
          " cta=" + std::to_string(pending_wait.key.cta_id) +
          " warp=" + std::to_string(warp_id) + " poll_pc=0x" +
          format_hex_u64(pending_wait.action_pc) +
          " cursor_before_flush=" +
          std::to_string(skip_stats.cursor_before_flush) +
          " cursor_after_flush=" +
          std::to_string(skip_stats.cursor_after_flush) +
          " prefetched_rollback=" +
          std::to_string(skip_stats.prefetched_rollback) +
          " cursor_after_skip=" +
          std::to_string(skip_stats.cursor_after_skip) +
          " skipped=" + std::to_string(skipped) +
          " per_pc=" + (first_pc ? std::string("<none>") : per_pc.str()) +
          " phasechk_trywait=" +
          std::to_string(skip_stats.phasechk_trywait) +
          " nanosleep=" + std::to_string(skip_stats.nanosleep) +
          " imad_scratch=" + std::to_string(skip_stats.imad_scratch) +
          " loopback_bra=" + std::to_string(skip_stats.loopback_bra) +
          " other=" + std::to_string(skip_stats.other) +
          " expected_exit_pc=0x" + format_hex_u64(region.exit_pc) +
          " actual_landing_pc=0x" + format_hex_u64(tw->get_pc()));
  debug_log_wait_block_event(
      "poll-release",
      "block=" + block +
          " trace_kernel=" +
          std::to_string(pending_wait.key.trace_kernel_id) +
          " function=" + std::to_string(pending_wait.unique_function_id) +
          " cta=" + std::to_string(pending_wait.key.cta_id) +
          " warp=" + std::to_string(warp_id) + " poll_pc=0x" +
          format_hex_u64(pending_wait.action_pc) + " exit_pc=0x" +
          format_hex_u64(region.exit_pc) +
          " expected_gen=" + std::to_string(exp_gen) +
          " current_gen=" + std::to_string(cur_gen) +
          " held_cycles=" + std::to_string(held_cycles));
  pending_wait = HopperMBarrierPendingWait();
}

void SM::print_wait_block_summary() const {
  if (!m_config->wait_block_enable) {
    return;
  }
  unsigned long long active_holds = 0;
  for (const HopperMBarrierPendingWait &pending : m_pending_sync_waits) {
    if (pending.valid) {
      active_holds++;
    }
  }
  unsigned long long active_partial_arms = 0;
  unsigned long long active_partial_replay_progress = 0;
  unsigned long long active_partial_regions = 0;
  unsigned long long active_partial_capsules = 0;
  for (unsigned warp_id = 0; warp_id < m_partial_wait_arms.size();
       ++warp_id) {
    active_partial_arms += m_partial_wait_arms[warp_id].valid;
    active_partial_replay_progress +=
        m_partial_replay_progress[warp_id].valid;
    active_partial_regions += m_partial_joined_regions[warp_id].size();
    if (m_config->is_trace_mode &&
        static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id])
            ->partial_capsule_active()) {
      active_partial_capsules++;
    }
  }
  const bool balanced =
      active_holds == 0 &&
      active_partial_arms == 0 &&
      active_partial_replay_progress == 0 &&
      active_partial_regions == 0 &&
      active_partial_capsules == 0 &&
      m_missing_wait_holds_total == m_missing_wait_releases_total &&
      m_fold_parks_total == m_fold_releases_total &&
      m_fold_aux_holds_total == m_fold_aux_releases_total &&
      m_partial_join_holds_total == m_partial_join_releases_total &&
      m_partial_deferred_occurrences_total ==
          m_partial_replayed_occurrences_total +
              m_partial_folded_occurrences_total;
  const double captured_poll_avg_hold =
      m_fold_releases_total
          ? static_cast<double>(m_fold_park_cycles_total) /
                m_fold_releases_total
          : 0.0;
  fprintf(
      stdout,
      "[WAITBLOCK][summary] status=%s sm=%u missing_wait_ready=%llu "
      "missing_wait_holds=%llu missing_wait_releases=%llu "
      "partial_missing_wait_model_misses=%llu "
      "partial_capsules=%llu partial_join_already_issued=%llu "
      "partial_deferred_occurrences=%llu partial_replayed_occurrences=%llu "
      "partial_folded_occurrences=%llu "
      "partial_join_holds=%llu partial_join_releases=%llu "
      "captured_poll_immediate_folds=%llu captured_poll_holds=%llu "
      "captured_poll_releases=%llu "
      "captured_poll_aux_holds=%llu captured_poll_aux_releases=%llu "
      "captured_poll_advanced_occurrences=%llu "
      "captured_poll_immediate_advanced_occurrences=%llu "
      "captured_poll_hold_cycles_total=%llu "
      "captured_poll_hold_cycles_avg=%.1f captured_poll_hold_cycles_max=%llu "
      "captured_poll_watchdog_aborts=%llu active_holds=%llu "
      "active_partial_arms=%llu active_partial_replay_progress=%llu "
      "active_partial_regions=%llu "
      "active_partial_capsules=%llu "
      "partial_replay_watchdog_aborts=%llu "
      "generation_overshoots=%llu exact_generation_targets=%llu "
      "generation_fallbacks=%llu late_arrival_divergences=%llu "
      "first_late_timing_reports=%llu "
      "watchdog_aborts=%llu\n",
      balanced ? "PASS" : "ACTIVE", m_sm_id, m_missing_wait_ready_total,
      m_missing_wait_holds_total,
      m_missing_wait_releases_total, m_partial_missing_wait_misses_total,
      m_partial_capsules_total, m_partial_join_already_issued_total,
      m_partial_deferred_occurrences_total,
      m_partial_replayed_occurrences_total,
      m_partial_folded_occurrences_total, m_partial_join_holds_total,
      m_partial_join_releases_total,
      m_fold_immediate_folds_total, m_fold_parks_total,
      m_fold_releases_total, m_fold_aux_holds_total,
      m_fold_aux_releases_total, m_fold_skipped_insts_total,
      m_fold_immediate_skipped_insts_total, m_fold_park_cycles_total,
      captured_poll_avg_hold, m_fold_park_cycles_max, m_fold_watchdog_hits,
      active_holds,
      active_partial_arms, active_partial_replay_progress,
      active_partial_regions, active_partial_capsules,
      m_partial_replay_watchdog_aborts_total,
      m_wait_block_generation_overshoots_total,
      m_wait_block_exact_targets_total,
      m_wait_block_generation_fallbacks_total,
      m_wait_block_late_arrivals_total,
      m_wait_block_first_late_reports_total,
      m_wait_block_watchdog_aborts_total);

  if (!m_config->wait_block_debug) {
    return;
  }
  for (unsigned warp_id = 0; warp_id < m_fold_pace_by_warp.size();
       ++warp_id) {
    const char *role =
        warp_id < MAX_WARP_PER_SHADER && m_warp_is_producer[warp_id]
            ? "producer"
            : (warp_id < MAX_WARP_PER_SHADER &&
                       m_warp_is_consumer[warp_id]
                   ? "consumer"
                   : "unknown");
    for (const auto &entry : m_fold_pace_by_warp[warp_id]) {
      const HopperMBarrierKey &barrier = entry.first.first;
      const uint64_t wait_pc = entry.first.second;
      const FoldPaceStats &pace = entry.second;
      const double avg_hold =
          pace.releases
              ? static_cast<double>(pace.park_cycles_total) / pace.releases
              : 0.0;
      fprintf(
          stdout,
          "[WAITBLOCK][poll-pace] sm=%u warp=%u role=%s trace_kernel=%u "
          "cta=%u global_cta=%u barrier=0x%s poll_pc=0x%s "
          "holds=%llu releases=%llu immediate_folds=%llu "
          "immediate_advanced_occurrences=%llu total_hold_cycles=%llu "
          "avg_hold_cycles=%.1f max_hold_cycles=%llu "
          "first_hold_cycle=%llu last_end_cycle=%llu "
          "last_expected_gen=%llu last_barrier_gen=%llu\n",
          m_sm_id, warp_id, role, barrier.trace_kernel_id, barrier.cta_id,
          m_local_to_global_cta_id[barrier.cta_id],
          format_hex_u64(barrier.barrier_addr).c_str(),
          format_hex_u64(wait_pc).c_str(), pace.parks, pace.releases,
          pace.immediate_folds, pace.immediate_skipped_insts,
          pace.park_cycles_total, avg_hold, pace.park_cycles_max,
          pace.first_park_cycle, pace.last_end_cycle,
          static_cast<unsigned long long>(pace.last_expected_gen),
          static_cast<unsigned long long>(pace.last_barrier_gen));
    }
  }
}


HopperMBarrierKey SM::build_sync_barrier_key(const warp_inst_t &inst,
                                             unsigned int warp_id) const {
  HopperMBarrierKey key;
  key.trace_kernel_id = inst.trace_kernel_id;
  key.cta_id = m_physical_warp[warp_id]->get_cta_id();
  key.barrier_addr = inst.sync_barrier_addr;
  return key;
}

HopperMBarrierObject &SM::get_or_create_sync_barrier(
    const HopperMBarrierKey &key) {
  return m_hopper_mbarriers[key];
}

void SM::recompute_sync_barrier_ready_and_maybe_flip_phase(
    const HopperMBarrierKey &key, HopperMBarrierObject &barrier) {
  const bool arrive_ready =
      barrier.arrive_count >= barrier.expected_arrive_count;
  const bool tx_ready =
      barrier.completed_tx_bytes >= barrier.expected_tx_bytes;
  barrier.ready = arrive_ready && tx_ready;
  if (!barrier.ready) {
    return;
  }
  if (barrier.expected_tx_bytes > 0 &&
      (barrier.assignment_open ||
       barrier.assigned_tx_bytes != barrier.expected_tx_bytes ||
       barrier.completed_tx_bytes != barrier.assigned_tx_bytes ||
       barrier.outstanding_tma_bindings != 0 ||
       barrier.tx_contract_id == 0 ||
       barrier.assignment_owner_warp_id == ~0u ||
       barrier.assignment_generation != barrier.gen ||
       barrier.assignment_kernel_launch_uid == 0 ||
       barrier.assignment_global_cta_id == ~0u)) {
    ++m_tma_contract_mismatches;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=phase-flip-with-incomplete-contract"
              << " barrier=0x" << format_hex_u64(key.barrier_addr)
              << " generation=" << barrier.gen
              << " contract_id=" << barrier.tx_contract_id
              << " assignment_open=" << barrier.assignment_open
              << " owner_warp=" << barrier.assignment_owner_warp_id
              << " assignment_generation="
              << barrier.assignment_generation
              << " expected=" << barrier.expected_tx_bytes
              << " assigned=" << barrier.assigned_tx_bytes
              << " completed=" << barrier.completed_tx_bytes
              << " outstanding=" << barrier.outstanding_tma_bindings
              << std::endl;
    debug_dump_tma_binding_snapshot("phase-flip-incomplete-contract");
    abort();
  }
  m_sync_debug_phase_flip++;
  debug_log_sync_event(
      "phase flip barrier=0x" + format_hex_u64(key.barrier_addr) + " phase=" +
      std::to_string(barrier.phase) + "->" +
      std::to_string(barrier.phase ^ 1) + " arrive=" +
      std::to_string(barrier.arrive_count) + "/" +
      std::to_string(barrier.expected_arrive_count) + " tx=" +
      std::to_string(barrier.completed_tx_bytes) + "/" +
      std::to_string(barrier.expected_tx_bytes) + " assigned=" +
      std::to_string(barrier.assigned_tx_bytes) + " outstanding=" +
      std::to_string(barrier.outstanding_tma_bindings) + " contract_id=" +
      std::to_string(barrier.tx_contract_id) + " gen=" +
      std::to_string(barrier.gen) + "->" + std::to_string(barrier.gen + 1) +
      " prev_flip_cyc=" + std::to_string(barrier.last_flip_cycle));
  if (m_config->wait_block_late_arrival_instrument_enable) {
    WaitBlockGenerationTiming &timing =
        barrier.wait_block_timing[barrier.gen + 1];
    timing.completion_valid = true;
    timing.completion_cycle = get_current_gpu_cycle();
  }
  barrier.prev_flip_cycle = barrier.last_flip_cycle;
  barrier.last_flip_cycle = get_current_gpu_cycle();
  barrier.phase ^= 1;
  // [Spin-wait fold audit] Monotonic generation, incremented on every flip
  // in lockstep with `phase`, never reset. Read only by the fold-path gen
  // audit; the 1-bit `phase` remains the sole driver of non-fold issue logic
  // so feature-off stays bit-identical. See .plan/SPIN_WAIT_SIM_MODELING.md 13.4.
  barrier.gen++;
  // Reset only the per-phase accumulators. expected_arrive_count is set once by
  // EXCH (mbarrier.init) and reused on every phase: EXCH runs exactly once per
  // barrier (validated in FA3 traces: 1 EXCH vs many PHASECHK/arrive phases),
  // so clearing it here would let the next phase flip after a single arrive.
  // expected_tx_bytes IS reset because the expect-tx arrive re-sets it each
  // phase.
  barrier.arrive_count = 0;
  barrier.expected_tx_bytes = 0;
  barrier.assigned_tx_bytes = 0;
  barrier.completed_tx_bytes = 0;
  barrier.outstanding_tma_bindings = 0;
  barrier.tx_contract_id = 0;
  barrier.assignment_open = false;
  barrier.assignment_owner_warp_id = ~0u;
  barrier.assignment_generation = ~0ull;
  barrier.assignment_kernel_launch_uid = 0;
  barrier.assignment_global_cta_id = ~0u;
  barrier.ready = false;
}

bool SM::is_sync_wait_satisfied(
    const HopperMBarrierPendingWait &pending_wait) const {
  auto it = m_hopper_mbarriers.find(pending_wait.key);
  if (it == m_hopper_mbarriers.end()) {
    return false;
  }
  // Hopper mbarrier try_wait/test_wait completes when the consumer's input
  // phaseParity DIFFERS from the barrier's current phase parity (the phase the
  // consumer was waiting on has flipped away). Equality means the barrier is
  // still in that phase -> keep waiting. See NVIDIA CUDA Programming Guide
  // (4.9 Asynchronous Barriers) and .plan/SYNC_ISA.md.
  return it->second.phase != decode_sync_wait_phase(pending_wait.wait_state_raw);
}

bool SM::is_wait_block_pending_satisfied(
    const HopperMBarrierPendingWait &pending_wait) const {
  if (pending_wait.uses_exact_generation_ticket) {
    return sync_barrier_gen(pending_wait.key) >= pending_wait.expected_gen;
  }
  return is_sync_wait_satisfied(pending_wait);
}

uint64_t SM::sync_barrier_gen(const HopperMBarrierKey &key) const {
  auto it = m_hopper_mbarriers.find(key);
  return it == m_hopper_mbarriers.end() ? 0 : it->second.gen;
}

void SM::record_wait_block_first_late_arrival(
    const warp_inst_t &inst, unsigned int warp_id,
    const HopperMBarrierKey &key, uint64_t target_generation,
    uint64_t current_generation, WaitBlockPendingMode mode) {
  if (!m_config->wait_block_late_arrival_instrument_enable ||
      key.cta_id >= MAX_CTA_PER_SHADER ||
      m_wait_block_first_late_reported_by_cta_slot[key.cta_id]) {
    return;
  }
  m_wait_block_first_late_reported_by_cta_slot[key.cta_id] = true;
  ++m_wait_block_first_late_reports_total;

  WaitBlockGenerationTiming target_timing;
  WaitBlockGenerationTiming next_timing;
  auto barrier_it = m_hopper_mbarriers.find(key);
  if (barrier_it != m_hopper_mbarriers.end()) {
    auto target_it =
        barrier_it->second.wait_block_timing.find(target_generation);
    if (target_it != barrier_it->second.wait_block_timing.end()) {
      target_timing = target_it->second;
    }
    auto next_it =
        barrier_it->second.wait_block_timing.find(target_generation + 1);
    if (next_it != barrier_it->second.wait_block_timing.end()) {
      next_timing = next_it->second;
    }
  }

  const WaitBlockSemanticAnchor *consumer_anchor = nullptr;
  if (warp_id < m_wait_block_last_semantic_anchor_by_warp.size() &&
      m_wait_block_last_semantic_anchor_by_warp[warp_id].valid) {
    consumer_anchor = &m_wait_block_last_semantic_anchor_by_warp[warp_id];
  }
  const uint32_t producer_warp =
      next_timing.contract_open_valid
          ? next_timing.contract_owner_warp_id
          : target_timing.contract_owner_warp_id;
  const WaitBlockSemanticAnchor *producer_anchor = nullptr;
  if (producer_warp < m_wait_block_last_semantic_anchor_by_warp.size() &&
      m_wait_block_last_semantic_anchor_by_warp[producer_warp].valid) {
    producer_anchor =
        &m_wait_block_last_semantic_anchor_by_warp[producer_warp];
  }

  const uint64_t now = get_current_gpu_cycle();
  std::ostringstream line;
  line << "[WAITBLOCK][late-timing-first]"
       << " sm=" << m_sm_id << " cycle=" << now
       << " trace_kernel=" << inst.trace_kernel_id
       << " cta=" << inst.trace_block_idx_x << ","
       << inst.trace_block_idx_y << "," << inst.trace_block_idx_z
       << " cta_slot=" << key.cta_id
       << " global_cta=" << m_local_to_global_cta_id[key.cta_id]
       << " cta_warp=" << inst.trace_cta_warp_id
       << " hardware_warp=" << warp_id
       << " function=" << inst.unique_function_id
       << " action_pc=0x" << format_hex_u64(inst.pc)
       << " source_index=" << inst.wait_block_ticket_source_index
       << " barrier=0x" << format_hex_u64(key.barrier_addr)
       << " strategy=" << wait_block_mode_to_string(mode)
       << " target_gen=" << target_generation
       << " current_gen=" << current_generation
       << " generation_distance="
       << current_generation - target_generation;

  const auto append_generation =
      [&line](const char *name, uint64_t generation,
              const WaitBlockGenerationTiming &timing) {
        line << " " << name << "_gen=" << generation
             << " " << name << "_open_valid="
             << timing.contract_open_valid
             << " " << name << "_open_cycle="
             << timing.contract_open_cycle
             << " " << name << "_producer_hardware_warp="
             << timing.contract_owner_warp_id
             << " " << name << "_producer_cta_warp="
             << timing.contract_owner_cta_warp_id
             << " " << name << "_open_function="
             << timing.contract_open_function_id
             << " " << name << "_open_pc=0x"
             << format_hex_u64(timing.contract_open_pc)
             << " " << name << "_open_source_index="
             << timing.contract_open_source_index
             << " " << name << "_tma_issue_valid="
             << timing.tma_issue_valid
             << " " << name << "_first_tma_issue_cycle="
             << timing.first_tma_issue_cycle
             << " " << name << "_last_tma_issue_cycle="
             << timing.last_tma_issue_cycle
             << " " << name << "_tma_commands="
             << timing.tma_commands_issued
             << " " << name << "_completion_valid="
             << timing.completion_valid
             << " " << name << "_completion_cycle="
             << timing.completion_cycle;
      };
  append_generation("target", target_generation, target_timing);
  append_generation("next", target_generation + 1, next_timing);

  const auto append_anchor =
      [&line, now](const char *name,
                   const WaitBlockSemanticAnchor *anchor) {
        line << " " << name << "_valid=" << (anchor != nullptr);
        if (anchor != nullptr) {
          line << " " << name << "_cycle=" << anchor->cycle
               << " " << name << "_age=" << now - anchor->cycle
               << " " << name << "_cta_warp=" << anchor->cta_warp_id
               << " " << name << "_function="
               << anchor->unique_function_id
               << " " << name << "_pc=0x"
               << format_hex_u64(anchor->pc)
               << " " << name << "_source_index="
               << anchor->source_index
               << " " << name << "_opcode=" << anchor->opcode;
        }
      };
  append_anchor("producer_anchor", producer_anchor);
  append_anchor("consumer_anchor", consumer_anchor);

  if (target_timing.completion_valid) {
    line << " target_complete_to_consumer="
         << now - target_timing.completion_cycle;
  }
  if (next_timing.contract_open_valid) {
    line << " next_open_to_consumer="
         << now - next_timing.contract_open_cycle;
  }
  if (next_timing.completion_valid) {
    line << " next_complete_to_consumer="
         << now - next_timing.completion_cycle;
  }
  std::cerr << line.str() << std::endl;
}

uint64_t SM::expected_gen_for_wait(unsigned int warp_id,
                                   const HopperMBarrierKey &key,
                                   uint64_t wait_state_raw) const {
  // [gen audit v3] Pin the generation this wait pairs with from the CURRENT
  // barrier state at first observation — NOT from a per-(warp,barrier) consumed
  // history. gen advances only on a real flip and is independent of expected.
  //
  //   parity = waited-on phase bit.  phase == (gen & 1).
  //   ready now (phase != parity)     -> expected = cur_gen       (this flip)
  //   not-ready (phase == parity)     -> expected = cur_gen + 1   (next flip)
  //
  // Rationale: the old "last_consumed + 1" forced every satisfied wait onto a
  // strictly newer gen, which double-counted a single flip observed by several
  // waits sharing one runtime barrier address and could over/under-count on
  // shared barriers. Anchoring to the live barrier gen makes immediate and held
  // waits use one identical rule and removes both false-audit directions while
  // still catching a genuine double-flip: a held wait pins expected=cur_gen+1
  // and, if the barrier
  // over-flips past it, release sees cur_gen > expected and aborts.
  (void)warp_id;
  auto barrier_it = m_hopper_mbarriers.find(key);
  if (barrier_it == m_hopper_mbarriers.end()) {
    // An unseen barrier is modeled not-ready by is_sync_wait_satisfied().
    // Pair such a hold with its first real flip instead of treating gen=0
    // parity as a completed wait on an object that does not yet exist.
    return 1;
  }
  const unsigned parity = decode_sync_wait_phase(wait_state_raw);
  const uint64_t cur_gen = barrier_it->second.gen;
  return cur_gen + (((cur_gen & 1ULL) == (uint64_t)parity) ? 1ULL : 0ULL);
}

void SM::handle_sync_instruction(warp_inst_t &inst, unsigned int warp_id) {
  if (inst.active_count() == 0) {
    if (m_config->wait_block_enable && m_config->wait_block_debug) {
      MissingWaitRecord missing_record;
      SpinWaitRegionRecord poll_record;
      const bool has_missing_action = m_gpu->lookup_missing_wait(
          inst.trace_kernel_id, inst.unique_function_id, inst.pc,
          missing_record);
      const bool has_poll_action = m_gpu->lookup_spin_wait_region(
          inst.trace_kernel_id, inst.unique_function_id, inst.pc, poll_record);
      if (has_missing_action || has_poll_action) {
        debug_log_wait_block_event(
            "predicated-off-noop",
            "strategy=" +
                std::string(has_missing_action ? "INJECT_MISSING_WAIT"
                                               : "FOLD_CAPTURED_POLL") +
                " block=" +
                (has_missing_action ? missing_record.block_id
                                    : poll_record.block_id) +
                " trace_kernel=" + std::to_string(inst.trace_kernel_id) +
                " function=" + std::to_string(inst.unique_function_id) +
                " cta=" +
                std::to_string(m_physical_warp[warp_id]->get_cta_id()) +
                " warp=" + std::to_string(warp_id) + " action_pc=0x" +
                format_hex_u64(inst.pc) + " effective_mask=0x" +
                format_hex_u64(inst.get_active_mask().to_ullong()) +
                " trace_active_mask=0x" +
                format_hex_u64(inst.trace_active_mask) +
                " trace_predicate_mask=0x" +
                format_hex_u64(inst.trace_predicate_mask) +
                " sync_runtime_valid=" +
                std::to_string(inst.sync_runtime_valid));
      }
    }
    return;
  }
  if (!inst.sync_site_valid || !inst.sync_runtime_valid) {
    if (m_config->wait_block_enable) {
      MissingWaitRecord missing_record;
      SpinWaitRegionRecord poll_record;
      const bool is_runtime_action =
          m_gpu->lookup_missing_wait(inst.trace_kernel_id,
                                     inst.unique_function_id, inst.pc,
                                     missing_record) ||
          m_gpu->lookup_spin_wait_region(inst.trace_kernel_id,
                                         inst.unique_function_id, inst.pc,
                                         poll_record);
      if (is_runtime_action) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=runtime-action-payload-missing"
                  << " trace_kernel=" << inst.trace_kernel_id
                  << " function=" << inst.unique_function_id
                  << " cta=" << m_physical_warp[warp_id]->get_cta_id()
                  << " warp=" << warp_id << " action_pc=0x"
                  << format_hex_u64(inst.pc)
                  << " sync_site_valid=" << inst.sync_site_valid
                  << " sync_runtime_valid=" << inst.sync_runtime_valid
                  // #region debug-point F:runtime-action-mask
                  << " effective_mask=0x"
                  << format_hex_u64(inst.get_active_mask().to_ullong())
                  << " trace_masks_valid=" << inst.trace_masks_valid
                  << " trace_active_mask=0x"
                  << format_hex_u64(inst.trace_active_mask)
                  << " trace_predicate_mask=0x"
                  << format_hex_u64(inst.trace_predicate_mask)
                  // #endregion
                  << " action=abort" << std::endl;
        abort();
      }
    }
    m_sync_debug_missing_runtime++;
    if (m_sync_debug_skip_runtime_budget > 0) {
      const std::string opcode =
          inst.has_extra_trace_instruction_info()
              ? inst.get_extra_trace_instruction_info().get_op_code()
              : "<no-trace-opcode>";
      debug_log_sync_event(
          "skip sync warp=" + std::to_string(warp_id) +
          " trace_kernel_id=" + std::to_string(inst.trace_kernel_id) +
          " site_valid=" + std::to_string(inst.sync_site_valid) +
          " runtime_valid=" + std::to_string(inst.sync_runtime_valid) +
          " kind=" + sync_kind_to_string(inst.sync_kind) + " pc=0x" +
          format_hex_u64(inst.pc) + " opcode=" + opcode +
          " barrier_idx=" + std::to_string(inst.sync_barrier_operand_index) +
          " semantic_idx=" + std::to_string(inst.sync_semantic_operand_index));
      --m_sync_debug_skip_runtime_budget;
    }
    return;
  }

  m_sync_debug_sync_insts++;
  HopperMBarrierKey key = build_sync_barrier_key(inst, warp_id);
  MissingWaitRecord missing_record;
  const bool has_missing_action = m_gpu->lookup_missing_wait(
      inst.trace_kernel_id, inst.unique_function_id, inst.pc, missing_record);
  SpinWaitRegionRecord poll_region;
  const bool has_poll_record = m_gpu->lookup_spin_wait_region(
      inst.trace_kernel_id, inst.unique_function_id, inst.pc, poll_region);
  const bool has_fold_wait_action =
      has_poll_record && poll_region.foldable;
  const bool is_primary_fold_wait =
      has_fold_wait_action && poll_region.primary_wait_pc == inst.pc;
  const bool is_auxiliary_fold_wait =
      has_fold_wait_action && !is_primary_fold_wait;
  if (has_missing_action && has_fold_wait_action) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=conflicting-runtime-actions"
              << " trace_kernel=" << inst.trace_kernel_id
              << " function=" << inst.unique_function_id
              << " pc=0x" << format_hex_u64(inst.pc)
              << " action=abort" << std::endl;
    abort();
  }
  const bool is_wait_probe =
      inst.sync_kind == SyncInstructionKind::PHASECHK ||
      inst.sync_kind == SyncInstructionKind::TRYWAIT;
  if ((has_missing_action || has_fold_wait_action) && !is_wait_probe) {
    std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=runtime-action-kind-mismatch"
              << " trace_kernel=" << inst.trace_kernel_id
              << " function=" << inst.unique_function_id
              << " pc=0x" << format_hex_u64(inst.pc)
              << " runtime_kind=" << sync_kind_to_string(inst.sync_kind)
              << " action=abort" << std::endl;
    abort();
  }
  if (is_wait_probe) {
    if (warp_id < m_pending_sync_waits.size()) {
      if (m_pending_sync_waits[warp_id].valid) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=held-warp-issued-another-sync"
                  << " trace_kernel=" << inst.trace_kernel_id
                  << " function=" << inst.unique_function_id
                  << " cta=" << key.cta_id << " warp=" << warp_id
                  << " pc=0x" << format_hex_u64(inst.pc)
                  << " action=abort" << std::endl;
        abort();
      }
      const bool is_trywait = inst.sync_kind == SyncInstructionKind::TRYWAIT;
      HopperMBarrierPendingWait probe;
      probe.valid = true;
      probe.is_trywait = is_trywait;
      probe.key = key;
      probe.wait_state_raw = inst.sync_semantic_raw;
      const bool parity_wait_ready = is_sync_wait_satisfied(probe);

      if ((has_missing_action &&
           (missing_record.entry_probe_pc != inst.pc ||
            missing_record.entry_probe_kind != inst.sync_kind ||
            (!missing_record.partial_modeled_miss_aborts &&
             !missing_record.partial_deferred_replay) ||
            !inst.sync_has_semantic_raw || !inst.trace_masks_valid)) ||
          (has_fold_wait_action &&
           (std::find(
                poll_region.wait_pcs.begin(), poll_region.wait_pcs.end(),
                static_cast<uint64_t>(inst.pc)) ==
                poll_region.wait_pcs.end() ||
            (is_primary_fold_wait &&
             poll_region.primary_wait_kind != inst.sync_kind) ||
            !inst.sync_has_semantic_raw))) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=action-kind-or-proof-mismatch"
                  << " trace_kernel=" << inst.trace_kernel_id
                  << " function=" << inst.unique_function_id
                  << " pc=0x" << format_hex_u64(inst.pc)
                  << " runtime_kind=" << sync_kind_to_string(inst.sync_kind)
                  << " action=abort" << std::endl;
        abort();
      }

      bool wait_ready = parity_wait_ready;
      bool action_uses_exact_generation = false;
      uint64_t action_expected_gen = 0;
      if (m_config->wait_block_enable &&
          (has_missing_action || has_fold_wait_action)) {
        const std::string &action_block_id =
            has_missing_action ? missing_record.block_id
                               : poll_region.block_id;
        if (m_config->wait_block_exact_generation_enable) {
          if (!inst.wait_block_generation_ticket_present) {
            std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                      << get_current_gpu_cycle()
                      << " reason=missing-generation-ticket"
                      << " block=" << action_block_id
                      << " trace_kernel=" << inst.trace_kernel_id
                      << " function=" << inst.unique_function_id
                      << " cta=" << key.cta_id << " warp=" << warp_id
                      << " action_pc=0x" << format_hex_u64(inst.pc)
                      << " action=abort" << std::endl;
            abort();
          }
          if (inst.wait_block_exact_target_valid) {
            action_uses_exact_generation = true;
            action_expected_gen =
                inst.wait_block_target_release_generation;
            m_wait_block_exact_targets_total++;
            const uint64_t cur_gen = sync_barrier_gen(key);
            wait_ready = cur_gen >= action_expected_gen;
            if (cur_gen > action_expected_gen) {
              m_wait_block_late_arrivals_total++;
              const WaitBlockPendingMode late_mode =
                  has_missing_action
                      ? WaitBlockPendingMode::MISSING_WAIT_HOLD
                      : (is_primary_fold_wait
                             ? WaitBlockPendingMode::CAPTURED_POLL_HOLD
                             : WaitBlockPendingMode::FOLD_AUX_WAIT_HOLD);
              if (has_missing_action) {
                record_wait_block_first_late_arrival(
                    inst, warp_id, key, action_expected_gen, cur_gen,
                    late_mode);
              }
              debug_log_wait_block_event(
                  "exact-generation-late-immediate",
                  "strategy=" +
                      std::string(wait_block_mode_to_string(late_mode)) +
                      " block=" + action_block_id +
                      " trace_kernel=" +
                      std::to_string(inst.trace_kernel_id) +
                      " function=" +
                      std::to_string(inst.unique_function_id) +
                      " cta=" + std::to_string(key.cta_id) +
                      " warp=" + std::to_string(warp_id) +
                      " action_pc=0x" + format_hex_u64(inst.pc) +
                      " source_index=" +
                      std::to_string(
                          inst.wait_block_ticket_source_index) +
                      " barrier=0x" + format_hex_u64(key.barrier_addr) +
                      " target_gen=" +
                      std::to_string(action_expected_gen) +
                      " current_gen=" + std::to_string(cur_gen));
            }
          } else {
            m_wait_block_generation_fallbacks_total++;
            action_expected_gen = expected_gen_for_wait(
                warp_id, key, inst.sync_semantic_raw);
          }
        } else {
          action_expected_gen = expected_gen_for_wait(
              warp_id, key, inst.sync_semantic_raw);
        }
      }

      if (m_config->producer_tail_instrument_enable) {
        record_producer_tail_phasechk_result(warp_id, wait_ready);
      }
      if (m_config->producer_spin_ready_instrument_enable) {
        record_producer_spin_phasechk_result(warp_id, wait_ready);
      }
      // [Spin-wait observe] observe-only would-be-park measurement: only for
      // polls that are the primary_wait_pc of a detected foldable spin loop.
      if (m_config->spin_wait_observe_enable && is_primary_fold_wait &&
          poll_region.semantic_free_body && poll_region.is_spin_loop) {
        record_spin_wait_observe(warp_id, inst.pc, key.barrier_addr, key.cta_id,
                                 wait_ready);
      }

      if (wait_ready) {
        m_sync_debug_wait_released++;
        if (has_missing_action) {
          m_missing_wait_ready_total++;
        }
        if (m_config->wait_block_enable && is_primary_fold_wait) {
          const uint64_t expected_gen = action_expected_gen;
          const uint64_t cur_gen = sync_barrier_gen(key);
          if (!action_uses_exact_generation &&
              expected_gen != cur_gen) {
            std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                      << get_current_gpu_cycle()
                      << " reason=immediate-generation-mismatch"
                      << " block=" << poll_region.block_id
                      << " warp=" << warp_id
                      << " expected_gen=" << expected_gen
                      << " current_gen=" << cur_gen
                      << " action=abort" << std::endl;
            abort();
          }
          FoldSkipStats skip_stats;
          const unsigned skipped =
              fold_cursor_skip_to_exit(warp_id, poll_region, skip_stats);
          m_fold_immediate_folds_total++;
          m_fold_immediate_skipped_insts_total += skipped;
          m_fold_skipped_insts_total += skipped;
          if (m_fold_pace_by_warp.size() <= warp_id) {
            m_fold_pace_by_warp.resize(MAX_WARP_PER_SHADER);
          }
          FoldPaceStats &pace =
              m_fold_pace_by_warp[warp_id][FoldPaceKey(key, inst.pc)];
          pace.immediate_folds++;
          pace.immediate_skipped_insts += skipped;
          pace.last_expected_gen = expected_gen;
          pace.last_barrier_gen = cur_gen;
          pace.last_end_cycle = get_current_gpu_cycle();
          if (m_config->wait_block_enable) {
            std::ostringstream per_pc;
            bool first_pc = true;
            for (const auto &entry : skip_stats.pc_occurrences) {
              if (!first_pc) {
                per_pc << ",";
              }
              first_pc = false;
              per_pc << "0x" << format_hex_u64(entry.first) << ":"
                     << entry.second;
            }
            debug_log_wait_block_event(
                "poll-advance",
                "mode=immediate block=" + poll_region.block_id +
                    " trace_kernel=" + std::to_string(inst.trace_kernel_id) +
                    " function=" + std::to_string(inst.unique_function_id) +
                    " cta=" + std::to_string(key.cta_id) +
                    " warp=" + std::to_string(warp_id) + " poll_pc=0x" +
                    format_hex_u64(inst.pc) +
                    " cursor_before_flush=" +
                    std::to_string(skip_stats.cursor_before_flush) +
                    " cursor_after_flush=" +
                    std::to_string(skip_stats.cursor_after_flush) +
                    " prefetched_rollback=" +
                    std::to_string(skip_stats.prefetched_rollback) +
                    " cursor_after_skip=" +
                    std::to_string(skip_stats.cursor_after_skip) +
                    " skipped=" + std::to_string(skipped) +
                    " per_pc=" +
                    (first_pc ? std::string("<none>") : per_pc.str()) +
                    " phasechk_trywait=" +
                    std::to_string(skip_stats.phasechk_trywait) +
                    " nanosleep=" + std::to_string(skip_stats.nanosleep) +
                    " imad_scratch=" +
                    std::to_string(skip_stats.imad_scratch) +
                    " loopback_bra=" +
                    std::to_string(skip_stats.loopback_bra) +
                    " other=" + std::to_string(skip_stats.other) +
                    " expected_exit_pc=0x" +
                    format_hex_u64(poll_region.exit_pc) +
                    " actual_landing_pc=0x" +
                    format_hex_u64(
                        static_cast<trace_shd_warp_t *>(
                            m_physical_warp[warp_id])
                            ->get_pc()));
          }
        }

        // [FWD drain-idle 축4] per-CTA mbarrier test count (released = immediate hit).
        if (m_config->cta_stall_breakdown_instrument_enable &&
            key.cta_id < MAX_CTA_PER_SHADER) {
          m_wait_released_by_cta_slot[key.cta_id]++;
        }
        // [FWD drain-idle 축3] observe-only: if this warp had an outstanding first-miss on THIS
        // barrier, the wait is now satisfied → bucket (now - first_miss_cycle) and clear. Dedicated
        // array only; never touches m_pending_sync_waits / issue path. Gated, default-off.
        if (m_config->sync_wait_hist_instrument_enable && warp_id < MAX_WARP_PER_SHADER &&
            m_sync_first_miss_barrier[warp_id] == key.barrier_addr) {
          unsigned long long delta =
              get_current_gpu_cycle() - m_sync_first_miss_cycle[warp_id];
          unsigned b = (delta == 0)      ? 0
                     : (delta <= 16)     ? 1
                     : (delta <= 64)     ? 2
                     : (delta <= 256)    ? 3
                     : (delta <= 1024)   ? 4
                                         : 5;
          m_sync_wait_cycle_hist[b]++;
          m_sync_first_miss_barrier[warp_id] = 0;
        }
        std::string hit_phase_state = " phase=<unseen> parity=" +
            std::to_string(decode_sync_wait_phase(inst.sync_semantic_raw));
        auto hit = m_hopper_mbarriers.find(key);
        if (hit != m_hopper_mbarriers.end()) {
          hit_phase_state =
              " phase=" + std::to_string(hit->second.phase) + " parity=" +
              std::to_string(decode_sync_wait_phase(inst.sync_semantic_raw));
        }
        debug_log_sync_event(
            "wait immediate-hit warp=" + std::to_string(warp_id) + " cta=" +
            std::to_string(key.cta_id) + " barrier=0x" +
            format_hex_u64(key.barrier_addr) + " wait_pc=0x" +
            format_hex_u64(inst.pc) + " wait_state=" +
            std::to_string(inst.sync_semantic_raw) + hit_phase_state);
        m_pending_sync_waits[warp_id] = HopperMBarrierPendingWait();
      } else {
        // Unselected probes remain nonblocking predicate tests. Runtime actions
        // below create a hold only for a sidecar-owned strategy.
        m_sync_debug_wait_pending++;
        // [FWD drain-idle 축4] per-CTA mbarrier test count (pending = found not-ready).
        if (m_config->cta_stall_breakdown_instrument_enable &&
            key.cta_id < MAX_CTA_PER_SHADER) {
          m_wait_pending_by_cta_slot[key.cta_id]++;
        }
        // [FWD drain-idle 축3] observe-only: record the FIRST miss cycle for this warp on this
        // barrier (spin retries keep the earliest). Uses a DEDICATED array (never m_pending_sync_waits,
        // which feeds the issue-path warp_waiting_at_barrier). Gated so default runs are unchanged.
        if (m_config->sync_wait_hist_instrument_enable && warp_id < MAX_WARP_PER_SHADER) {
          if (m_sync_first_miss_barrier[warp_id] != key.barrier_addr) {
            m_sync_first_miss_barrier[warp_id] = key.barrier_addr;
            m_sync_first_miss_cycle[warp_id] = get_current_gpu_cycle();
          }
        }
        bool hold_created = false;
        if (m_config->wait_block_enable && has_missing_action) {
          const active_mask_t effective_mask = inst.get_active_mask();
          const active_mask_t live_mask =
              m_physical_warp[warp_id]->get_active_mask();
          if ((effective_mask & ~live_mask).any() || live_mask.none()) {
            std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                      << get_current_gpu_cycle()
                      << " reason=invalid-entry-mask-state"
                      << " block=" << missing_record.block_id
                      << " trace_kernel=" << inst.trace_kernel_id
                      << " function=" << inst.unique_function_id
                      << " cta=" << key.cta_id << " warp=" << warp_id
                      << " entry_pc=0x" << format_hex_u64(inst.pc)
                      << " effective_mask=0x"
                      << format_hex_u64(effective_mask.to_ullong())
                      << " live_mask=0x"
                      << format_hex_u64(live_mask.to_ullong())
                      << " action=abort" << std::endl;
            abort();
          }
          const uint64_t expected_gen = action_expected_gen;
          if (effective_mask != live_mask) {
            m_partial_missing_wait_misses_total++;
            const uint64_t effective_bits = effective_mask.to_ullong();
            const uint64_t live_bits = live_mask.to_ullong();
            const uint64_t sibling_bits = live_bits & ~effective_bits;
            if (!missing_record.partial_deferred_replay ||
                missing_record.partial_modeled_miss_aborts ||
                missing_record.partial_region_id.empty() ||
                missing_record.observed_entry_masks.count(effective_bits) == 0 ||
                missing_record.observed_sibling_masks.count(sibling_bits) == 0 ||
                missing_record.observed_live_masks.count(live_bits) == 0 ||
                sibling_bits == 0 ||
                m_partial_wait_arms[warp_id].valid) {
              std::cerr
                  << "[WAITBLOCK][UNSUPPORTED_PARTIAL_MISSING_WAIT][SM"
                  << m_sm_id << "] cyc=" << get_current_gpu_cycle()
                  << " block=" << missing_record.block_id
                  << " trace_kernel=" << inst.trace_kernel_id
                  << " function=" << inst.unique_function_id
                  << " cta=" << key.cta_id
                  << " global_cta="
                  << m_local_to_global_cta_id[key.cta_id]
                  << " warp=" << warp_id << " entry_probe_pc=0x"
                  << format_hex_u64(inst.pc)
                  << " effective_mask=0x"
                  << format_hex_u64(effective_bits)
                  << " live_mask=0x" << format_hex_u64(live_bits)
                  << " sibling_mask=0x"
                  << format_hex_u64(sibling_bits)
                  << " partial_deferred_replay="
                  << missing_record.partial_deferred_replay
                  << " arm_already_valid="
                  << m_partial_wait_arms[warp_id].valid
                  << " action=abort" << std::endl;
              abort();
            }

            HopperPartialWaitArm arm;
            arm.valid = true;
            arm.mode = PartialWaitArmMode::ARMED;
            arm.record = missing_record;
            arm.key = key;
            arm.wait_state_raw = inst.sync_semantic_raw;
            arm.expected_gen = expected_gen;
            arm.uses_exact_generation_ticket =
                action_uses_exact_generation;
            arm.blocked_mask = effective_bits;
            arm.sibling_mask = sibling_bits;
            arm.unique_function_id = inst.unique_function_id;
            arm.arm_start_cycle = get_current_gpu_cycle();
            arm.last_heartbeat_cycle = arm.arm_start_cycle;
            arm.source_was_deferred_replay =
                static_cast<const trace_warp_inst_t &>(inst)
                    .wait_block_trace_origin ==
                WaitBlockTraceOrigin::PARTIAL_BLOCKED_REPLAY;
            m_partial_wait_arms[warp_id] = arm;
            hold_created = true;
            debug_log_wait_block_event(
                "partial-arm",
                "block=" + missing_record.block_id +
                    " region=" + missing_record.partial_region_id +
                    " trace_kernel=" +
                    std::to_string(inst.trace_kernel_id) +
                    " function=" +
                    std::to_string(inst.unique_function_id) +
                    " cta=" + std::to_string(key.cta_id) +
                    " warp=" + std::to_string(warp_id) +
                    " entry_pc=0x" + format_hex_u64(inst.pc) +
                    " branch_pc=0x" +
                    format_hex_u64(missing_record.entry_branch_pc) +
                    " blocked_mask=0x" +
                    format_hex_u64(effective_bits) +
                    " sibling_mask=0x" +
                    format_hex_u64(sibling_bits) +
                    " join_already_issued=" +
                    std::to_string(
                        m_partial_joined_regions[warp_id].count(
                            missing_record.partial_region_id) != 0) +
                    " source_is_deferred_replay=" +
                    std::to_string(arm.source_was_deferred_replay));
          } else {
            HopperMBarrierPendingWait held;
            held.valid = true;
            held.is_trywait = is_trywait;
            held.key = key;
            held.wait_state_raw = inst.sync_semantic_raw;
            held.mode = WaitBlockPendingMode::MISSING_WAIT_HOLD;
            held.block_id = missing_record.block_id;
            held.action_pc = inst.pc;
            held.unique_function_id = inst.unique_function_id;
            held.expected_gen = expected_gen;
            held.uses_exact_generation_ticket =
                action_uses_exact_generation;
            held.hold_start_cycle = get_current_gpu_cycle();
            held.last_heartbeat_cycle = held.hold_start_cycle;
            m_pending_sync_waits[warp_id] = held;
            m_missing_wait_holds_total++;
            hold_created = true;
            debug_log_wait_block_event(
                "missing-wait-hold",
                "block=" + missing_record.block_id +
                    " trace_kernel=" + std::to_string(inst.trace_kernel_id) +
                    " function=" + std::to_string(inst.unique_function_id) +
                    " cta=" + std::to_string(key.cta_id) +
                    " warp=" + std::to_string(warp_id) + " entry_pc=0x" +
                    format_hex_u64(inst.pc) + " effective_mask=0x" +
                    format_hex_u64(effective_mask.to_ullong()) +
                    " live_mask=0x" +
                    format_hex_u64(live_mask.to_ullong()) + " barrier=0x" +
                    format_hex_u64(key.barrier_addr) +
                    " wait_state=0x" +
                    format_hex_u64(inst.sync_semantic_raw) +
                    " expected_gen=" + std::to_string(expected_gen) +
                    " exact_target=" +
                    std::to_string(
                        action_uses_exact_generation) +
                    " current_gen=" +
                    std::to_string(sync_barrier_gen(key)));
          }
        } else if (m_config->wait_block_enable &&
                   is_primary_fold_wait) {
          HopperMBarrierPendingWait held;
          held.valid = true;
          held.is_trywait = is_trywait;
          held.key = key;
          held.wait_state_raw = inst.sync_semantic_raw;
          held.mode = WaitBlockPendingMode::CAPTURED_POLL_HOLD;
          held.block_id = poll_region.block_id;
          held.action_pc = inst.pc;
          held.poll_loop_exit_pc = poll_region.exit_pc;
          held.unique_function_id = inst.unique_function_id;
          held.expected_gen = action_expected_gen;
          held.uses_exact_generation_ticket =
              action_uses_exact_generation;
          held.hold_start_cycle = get_current_gpu_cycle();
          held.last_heartbeat_cycle = held.hold_start_cycle;
          m_pending_sync_waits[warp_id] = held;
          hold_created = true;
          m_fold_parks_total++;
          if (m_fold_pace_by_warp.size() <= warp_id) {
            m_fold_pace_by_warp.resize(MAX_WARP_PER_SHADER);
          }
          FoldPaceStats &pace =
              m_fold_pace_by_warp[warp_id][FoldPaceKey(key, inst.pc)];
          pace.parks++;
          if (pace.first_park_cycle == 0) {
            pace.first_park_cycle = get_current_gpu_cycle();
          }
          pace.last_expected_gen = held.expected_gen;
          pace.last_barrier_gen = sync_barrier_gen(key);
          if (m_config->wait_block_enable) {
            debug_log_wait_block_event(
                "poll-hold",
                "block=" + poll_region.block_id +
                    " trace_kernel=" + std::to_string(inst.trace_kernel_id) +
                    " function=" + std::to_string(inst.unique_function_id) +
                    " cta=" + std::to_string(key.cta_id) +
                    " warp=" + std::to_string(warp_id) + " poll_pc=0x" +
                    format_hex_u64(inst.pc) + " exit_pc=0x" +
                    format_hex_u64(poll_region.exit_pc) + " barrier=0x" +
                    format_hex_u64(key.barrier_addr) +
                    " expected_gen=" + std::to_string(held.expected_gen) +
                    " exact_target=" +
                    std::to_string(action_uses_exact_generation) +
                    " current_gen=" +
                    std::to_string(sync_barrier_gen(key)));
          }
        } else if (
            m_config->wait_block_enable &&
            is_auxiliary_fold_wait &&
            action_uses_exact_generation) {
          HopperMBarrierPendingWait held;
          held.valid = true;
          held.is_trywait = is_trywait;
          held.key = key;
          held.wait_state_raw = inst.sync_semantic_raw;
          held.mode = WaitBlockPendingMode::FOLD_AUX_WAIT_HOLD;
          held.block_id = poll_region.block_id;
          held.action_pc = inst.pc;
          held.unique_function_id = inst.unique_function_id;
          held.expected_gen = action_expected_gen;
          held.uses_exact_generation_ticket = true;
          held.hold_start_cycle = get_current_gpu_cycle();
          held.last_heartbeat_cycle = held.hold_start_cycle;
          m_pending_sync_waits[warp_id] = held;
          hold_created = true;
          m_fold_aux_holds_total++;
          debug_log_wait_block_event(
              "fold-aux-wait-hold",
              "block=" + poll_region.block_id +
                  " trace_kernel=" +
                  std::to_string(inst.trace_kernel_id) +
                  " function=" +
                  std::to_string(inst.unique_function_id) +
                  " cta=" + std::to_string(key.cta_id) +
                  " warp=" + std::to_string(warp_id) +
                  " wait_pc=0x" + format_hex_u64(inst.pc) +
                  " primary_pc=0x" +
                  format_hex_u64(poll_region.primary_wait_pc) +
                  " barrier=0x" + format_hex_u64(key.barrier_addr) +
                  " target_gen=" +
                  std::to_string(action_expected_gen) +
                  " current_gen=" +
                  std::to_string(sync_barrier_gen(key)) +
                  " cursor_action=none");
        }
        if (!hold_created) {
          m_pending_sync_waits[warp_id] = HopperMBarrierPendingWait();
        }

        // Deadlock diagnosis: dump the current barrier counters so a barrier
        // that never becomes ready (arrive_count < expected, or tx bytes short)
        // is visible even late in a long run.
        std::string barrier_state = " barrier-state=<unseen>";
        auto bit = m_hopper_mbarriers.find(key);
        if (bit != m_hopper_mbarriers.end()) {
          const HopperMBarrierObject &b = bit->second;
          barrier_state =
              " arrive=" + std::to_string(b.arrive_count) + "/" +
              std::to_string(b.expected_arrive_count) + " tx=" +
              std::to_string(b.completed_tx_bytes) + "/" +
              std::to_string(b.expected_tx_bytes) + " assigned=" +
              std::to_string(b.assigned_tx_bytes) + " outstanding=" +
              std::to_string(b.outstanding_tma_bindings) +
              " contract_id=" + std::to_string(b.tx_contract_id) +
              " phase=" +
              std::to_string(b.phase) + " ready=" +
              std::to_string(b.ready ? 1 : 0);
        }
        debug_log_sync_event(
            "wait miss nonblocking warp=" + std::to_string(warp_id) +
            " cta=" +
            std::to_string(key.cta_id) + " barrier=0x" +
            format_hex_u64(key.barrier_addr) + " wait_state=" +
            std::to_string(inst.sync_semantic_raw) + " parity=" +
            std::to_string(decode_sync_wait_phase(inst.sync_semantic_raw)) +
            " trywait=" +
            std::to_string(is_trywait) + barrier_state);
      }
    }
    return;
  }

  HopperMBarrierObject &barrier = get_or_create_sync_barrier(key);
  // Guard: any arrive that actually executes must be a validated variant. The
  // resolver labels all arrives ARRIVE_EXPECT_TX and the real arrive vs
  // expect-tx behavior is decided below from sync_semantic_raw. Unverified
  // arrive opcodes must not be silently trusted (see .plan/SYNC_ISA.md).
  if (inst.sync_kind == SyncInstructionKind::ARRIVE ||
      inst.sync_kind == SyncInstructionKind::ARRIVE_COUNTED ||
      inst.sync_kind == SyncInstructionKind::ARRIVE_EXPECT_TX) {
    const std::string arrive_opcode =
        inst.has_extra_trace_instruction_info()
            ? inst.get_extra_trace_instruction_info().get_op_code()
            : std::string("<no-trace-opcode>");
    if (!is_validated_arrive_opcode(arrive_opcode)) {
      fprintf(stderr,
              "[sync] FATAL: unvalidated SYNCS.ARRIVE variant executed: '%s' "
              "(pc=0x%s cta=%u barrier=0x%s). Validate its operand/semantics "
              "(see .plan/SYNC_ISA.md) before trusting it.\n",
              arrive_opcode.c_str(), format_hex_u64(inst.pc).c_str(),
              key.cta_id, format_hex_u64(key.barrier_addr).c_str());
      abort();
    }
  }
  // mbarrier tracks arrivals in thread units: a SYNCS.ARRIVE instruction adds
  // one arrival per active thread (validated against FA3 traces, where the
  // count barrier has active_mask=0xffffffff -> +32 and the tx barrier has
  // active_mask=0x1 -> +1, matching the logical expected-arrival count).
  const uint32_t active_threads =
      static_cast<uint32_t>(inst.active_count());
  const std::string active_mask_hex =
      format_hex_u64(inst.get_active_mask().to_ullong());
  switch (inst.sync_kind) {
    case SyncInstructionKind::EXCH:
      m_sync_debug_exch++;
      if (barrier.assignment_open || barrier.tx_contract_id != 0 ||
          barrier.expected_tx_bytes != 0 ||
          barrier.assigned_tx_bytes != 0 ||
          barrier.completed_tx_bytes != 0 ||
          barrier.outstanding_tma_bindings != 0) {
        ++m_tma_orphan_cleanup;
        std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=barrier-reinit-with-active-contract"
                  << " trace_kernel=" << key.trace_kernel_id
                  << " cta=" << key.cta_id << " barrier=0x"
                  << format_hex_u64(key.barrier_addr)
                  << " generation=" << barrier.gen
                  << " contract_id=" << barrier.tx_contract_id
                  << " assignment_open=" << barrier.assignment_open
                  << " expected=" << barrier.expected_tx_bytes
                  << " assigned=" << barrier.assigned_tx_bytes
                  << " completed=" << barrier.completed_tx_bytes
                  << " outstanding=" << barrier.outstanding_tma_bindings
                  << std::endl;
        debug_dump_tma_binding_snapshot(
            "barrier-reinit-with-active-contract");
        abort();
      }
      barrier.expected_arrive_count = inst.sync_has_semantic_raw
                                          ? decode_sync_exch_expected_arrive_count(
                                                inst.sync_semantic_raw)
                                          : 0;
      barrier.arrive_count = 0;
      barrier.expected_tx_bytes = 0;
      barrier.assigned_tx_bytes = 0;
      barrier.completed_tx_bytes = 0;
      barrier.outstanding_tma_bindings = 0;
      barrier.tx_contract_id = 0;
      barrier.assignment_open = false;
      barrier.assignment_owner_warp_id = ~0u;
      barrier.assignment_generation = ~0ull;
      barrier.assignment_kernel_launch_uid = 0;
      barrier.assignment_global_cta_id = ~0u;
      barrier.ready = false;
      if (m_config->wait_block_late_arrival_instrument_enable) {
        barrier.wait_block_timing.clear();
      }
      debug_log_sync_event(
          "exch warp=" + std::to_string(warp_id) + " cta=" +
          std::to_string(key.cta_id) + " barrier=0x" +
          format_hex_u64(key.barrier_addr) + " raw=0x" +
          format_hex_u64(inst.sync_semantic_raw) + " (0x200000-raw)=" +
          std::to_string(0x200000ULL - inst.sync_semantic_raw) +
          " expected_arrive(logical=/2)=" +
          std::to_string(barrier.expected_arrive_count));
      break;
    case SyncInstructionKind::ARRIVE:
      m_sync_debug_arrive++;
      barrier.arrive_count += active_threads;
      debug_log_sync_event(
          "arrive warp=" + std::to_string(warp_id) + " cta=" +
          std::to_string(key.cta_id) + " barrier=0x" +
          format_hex_u64(key.barrier_addr) + " pc=0x" +
          format_hex_u64(inst.pc) + " gen=" + std::to_string(barrier.gen) +
          " phase=" + std::to_string(barrier.phase) + " +active=" +
          std::to_string(active_threads) + "(mask=0x" + active_mask_hex +
          ") arrive=" + std::to_string(barrier.arrive_count) + "/" +
          std::to_string(barrier.expected_arrive_count));
      recompute_sync_barrier_ready_and_maybe_flip_phase(key, barrier);
      break;
    case SyncInstructionKind::ARRIVE_COUNTED:
      m_sync_debug_arrive++;
      barrier.arrive_count +=
          inst.sync_has_semantic_raw
              ? static_cast<uint32_t>(inst.sync_semantic_raw)
              : active_threads;
      debug_log_sync_event(
          "arrive_counted warp=" + std::to_string(warp_id) + " cta=" +
          std::to_string(key.cta_id) + " barrier=0x" +
          format_hex_u64(key.barrier_addr) + " pc=0x" +
          format_hex_u64(inst.pc) + " gen=" + std::to_string(barrier.gen) +
          " phase=" + std::to_string(barrier.phase) + " +active=" +
          std::to_string(active_threads) + "(mask=0x" + active_mask_hex +
          ") arrive=" + std::to_string(barrier.arrive_count) + "/" +
          std::to_string(barrier.expected_arrive_count));
      recompute_sync_barrier_ready_and_maybe_flip_phase(key, barrier);
      break;
    case SyncInstructionKind::ARRIVE_EXPECT_TX: {
      // The resolver labels every arrive ARRIVE_EXPECT_TX; the real behavior is
      // decided here from the captured semantic_raw (see .plan/SYNC_ISA.md):
      //   semantic_raw == 0 -> plain arrive (operand2 = RZ, no tx bytes)
      //   semantic_raw != 0 -> arrive + expect tx bytes
      const bool is_expect_tx =
          inst.sync_has_semantic_raw && inst.sync_semantic_raw != 0;
      if (is_expect_tx) {
        m_sync_debug_arrive_expect_tx++;
        // [FWD drain-idle 축2] a warp that issues arrive+expect-tx announces an incoming TMA
        // transfer → it is a data-producer warp (sticky).
        mark_producer_warp(warp_id);
        barrier.arrive_count += active_threads;
        uint32_t tx_bytes = static_cast<uint32_t>(inst.sync_semantic_raw);
        open_tma_assignment_contract(inst, warp_id, key, tx_bytes);
        debug_log_sync_event(
            "arrive_expect_tx warp=" + std::to_string(warp_id) + " cta=" +
            std::to_string(key.cta_id) + " barrier=0x" +
            format_hex_u64(key.barrier_addr) + " pc=0x" +
            format_hex_u64(inst.pc) + " gen=" + std::to_string(barrier.gen) +
            " phase=" + std::to_string(barrier.phase) + " +active=" +
            std::to_string(active_threads) + "(mask=0x" + active_mask_hex +
            ") arrive=" + std::to_string(barrier.arrive_count) + "/" +
            std::to_string(barrier.expected_arrive_count) + " +tx=" +
            std::to_string(tx_bytes) + " tx=" +
            std::to_string(barrier.expected_tx_bytes));
      } else {
        m_sync_debug_arrive++;
        barrier.arrive_count += active_threads;
        debug_log_sync_event(
            "arrive warp=" + std::to_string(warp_id) + " cta=" +
            std::to_string(key.cta_id) + " barrier=0x" +
            format_hex_u64(key.barrier_addr) + " pc=0x" +
            format_hex_u64(inst.pc) + " gen=" + std::to_string(barrier.gen) +
            " phase=" + std::to_string(barrier.phase) + " +active=" +
            std::to_string(active_threads) + "(mask=0x" + active_mask_hex +
            ") arrive=" + std::to_string(barrier.arrive_count) + "/" +
            std::to_string(barrier.expected_arrive_count));
      }
      recompute_sync_barrier_ready_and_maybe_flip_phase(key, barrier);
      break;
    }
    case SyncInstructionKind::NONE:
    case SyncInstructionKind::PHASECHK:
    case SyncInstructionKind::TRYWAIT:
      break;
  }
}

void SM::clear_sync_barrier_state_for_cta(unsigned int cta_id) {
  for (unsigned warp_id = 0; warp_id < m_pending_sync_waits.size();
       ++warp_id) {
    HopperMBarrierPendingWait &pending_wait =
        m_pending_sync_waits[warp_id];
    if (pending_wait.valid && pending_wait.key.cta_id == cta_id) {
      if (m_config->wait_block_enable) {
        std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=cta-cleanup-with-active-hold"
                  << " strategy="
                  << wait_block_mode_to_string(pending_wait.mode)
                  << " trace_kernel=" << pending_wait.key.trace_kernel_id
                  << " cta=" << cta_id << " warp=" << warp_id
                  << " action_pc=0x"
                  << format_hex_u64(pending_wait.action_pc)
                  << " action=abort" << std::endl;
        abort();
      }
      pending_wait = HopperMBarrierPendingWait();
    }
  }
  for (unsigned warp_id = 0; warp_id < m_partial_wait_arms.size();
       ++warp_id) {
    if (m_physical_warp[warp_id]->get_cta_id() != cta_id) {
      continue;
    }
    if (m_config->wait_block_enable &&
        (m_partial_wait_arms[warp_id].valid ||
         m_partial_replay_progress[warp_id].valid ||
         !m_partial_joined_regions[warp_id].empty() ||
         (m_config->is_trace_mode &&
          static_cast<trace_shd_warp_t *>(m_physical_warp[warp_id])
              ->partial_capsule_active()))) {
      std::cerr << "[WAITBLOCK][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=cta-cleanup-with-active-partial-state"
                << " cta=" << cta_id << " warp=" << warp_id
                << " partial_arm="
                << m_partial_wait_arms[warp_id].valid
                << " replay_progress="
                << m_partial_replay_progress[warp_id].valid
                << " joined_regions="
                << m_partial_joined_regions[warp_id].size()
                << " action=abort" << std::endl;
      abort();
    }
    m_partial_wait_arms[warp_id] = HopperPartialWaitArm();
    m_partial_replay_progress[warp_id] =
        HopperPartialReplayProgress();
    m_partial_joined_regions[warp_id].clear();
  }
  for (auto it = m_hopper_mbarriers.begin(); it != m_hopper_mbarriers.end();) {
    if (it->first.cta_id == cta_id) {
      const HopperMBarrierObject &barrier = it->second;
      if (barrier.assignment_open ||
          barrier.expected_tx_bytes != barrier.assigned_tx_bytes ||
          barrier.assigned_tx_bytes != barrier.completed_tx_bytes ||
          barrier.outstanding_tma_bindings != 0) {
        ++m_tma_orphan_cleanup;
        std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
                  << get_current_gpu_cycle()
                  << " reason=cta-cleanup-with-inflight-completion"
                  << " trace_kernel=" << it->first.trace_kernel_id
                  << " cta_slot=" << it->first.cta_id << " barrier=0x"
                  << format_hex_u64(it->first.barrier_addr)
                  << " generation=" << barrier.gen
                  << " contract_id=" << barrier.tx_contract_id
                  << " assignment_open=" << barrier.assignment_open
                  << " expected=" << barrier.expected_tx_bytes
                  << " assigned=" << barrier.assigned_tx_bytes
                  << " completed=" << barrier.completed_tx_bytes
                  << " outstanding=" << barrier.outstanding_tma_bindings
                  << std::endl;
        debug_dump_tma_binding_snapshot(
            "cta-cleanup-with-inflight-completion");
        abort();
      }
      it = m_hopper_mbarriers.erase(it);
    } else {
      ++it;
    }
  }
  // Poll pace is emitted immediately before this cleanup at CTA exit. Drop the
  // just-printed CTA-slot entries so a later CTA reusing the slot gets an
  // independent pacing record rather than a cross-wave aggregate.
  for (auto &per_warp : m_fold_pace_by_warp) {
    for (auto it = per_warp.begin(); it != per_warp.end();) {
      if (it->first.first.cta_id == cta_id) {
        it = per_warp.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void SM::open_tma_assignment_contract(const warp_inst_t &inst,
                                      unsigned int warp_id,
                                      const HopperMBarrierKey &key,
                                      uint32_t expected_tx_bytes) {
  if (warp_id >= m_tma_open_contracts_per_warp.size() ||
      key.cta_id >= MAX_CTA_PER_SHADER || expected_tx_bytes == 0) {
    ++m_tma_contract_mismatches;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=invalid-contract-open"
              << " warp=" << warp_id << " trace_kernel="
              << key.trace_kernel_id << " cta=" << key.cta_id
              << " barrier=0x" << format_hex_u64(key.barrier_addr)
              << " expected=" << expected_tx_bytes << std::endl;
    debug_dump_tma_binding_snapshot("invalid-contract-open");
    abort();
  }

  auto barrier_it = m_hopper_mbarriers.find(key);
  if (barrier_it == m_hopper_mbarriers.end()) {
    ++m_tma_contract_mismatches;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=contract-barrier-missing"
              << " warp=" << warp_id << " trace_kernel="
              << key.trace_kernel_id << " cta=" << key.cta_id
              << " barrier=0x" << format_hex_u64(key.barrier_addr)
              << std::endl;
    debug_dump_tma_binding_snapshot("contract-open");
    abort();
  }

  HopperMBarrierObject &barrier = barrier_it->second;
  if (barrier.assignment_open || barrier.tx_contract_id != 0 ||
      barrier.expected_tx_bytes != 0 || barrier.assigned_tx_bytes != 0 ||
      barrier.completed_tx_bytes != 0 ||
      barrier.outstanding_tma_bindings != 0 ||
      barrier.assignment_owner_warp_id != ~0u ||
      barrier.assignment_generation != ~0ull ||
      barrier.assignment_kernel_launch_uid != 0 ||
      barrier.assignment_global_cta_id != ~0u) {
    ++m_tma_contract_mismatches;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=contract-generation-not-empty"
              << " warp=" << warp_id << " trace_kernel="
              << key.trace_kernel_id << " cta=" << key.cta_id
              << " barrier=0x" << format_hex_u64(key.barrier_addr)
              << " generation=" << barrier.gen
              << " owner_warp=" << barrier.assignment_owner_warp_id
              << " expected=" << barrier.expected_tx_bytes
              << " assigned=" << barrier.assigned_tx_bytes
              << " completed=" << barrier.completed_tx_bytes
              << " outstanding=" << barrier.outstanding_tma_bindings
              << std::endl;
    debug_dump_tma_binding_snapshot("contract-generation-not-empty");
    abort();
  }

  const uint64_t launch_uid = m_physical_warp[warp_id]->m_kernel_id;
  const uint32_t global_cta_id = get_global_cta_id(key.cta_id);
  if (launch_uid == 0) {
    ++m_tma_launch_or_cta_mismatch;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=contract-owner-launch-missing"
              << " warp=" << warp_id << " cta_slot=" << key.cta_id
              << std::endl;
    debug_dump_tma_binding_snapshot("contract-owner-launch-missing");
    abort();
  }
  if (m_tma_binding_launch_uid == 0) {
    m_tma_binding_launch_uid = launch_uid;
  }
  if (m_tma_binding_launch_uid != launch_uid) {
    ++m_tma_launch_or_cta_mismatch;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=contract-launch-mismatch"
              << " stats_launch=" << m_tma_binding_launch_uid
              << " current_launch=" << launch_uid << std::endl;
    debug_dump_tma_binding_snapshot("contract-launch-mismatch");
    abort();
  }

  const uint64_t contract_id = m_next_tma_contract_id++;
  barrier.tx_contract_id = contract_id;
  barrier.assignment_open = true;
  barrier.assignment_owner_warp_id = warp_id;
  barrier.assignment_generation = barrier.gen;
  barrier.assignment_kernel_launch_uid = launch_uid;
  barrier.assignment_global_cta_id = global_cta_id;
  barrier.expected_tx_bytes = expected_tx_bytes;
  barrier.assigned_tx_bytes = 0;
  barrier.completed_tx_bytes = 0;
  barrier.outstanding_tma_bindings = 0;
  ++m_tma_open_contracts_per_warp[warp_id];
  m_tma_max_open_contracts_per_warp =
      std::max(m_tma_max_open_contracts_per_warp,
               m_tma_open_contracts_per_warp[warp_id]);
  ++m_tma_contracts_opened;

  if (m_config->wait_block_late_arrival_instrument_enable) {
    WaitBlockGenerationTiming &timing =
        barrier.wait_block_timing[barrier.gen + 1];
    timing.contract_open_valid = true;
    timing.contract_open_cycle = get_current_gpu_cycle();
    timing.contract_owner_warp_id = warp_id;
    timing.contract_owner_cta_warp_id = inst.trace_cta_warp_id;
    timing.contract_open_function_id = inst.unique_function_id;
    timing.contract_open_pc = inst.pc;
    timing.contract_open_source_index = inst.trace_instruction_index;
  }

  debug_log_tma_binding_event(
      "contract-open",
      "kernel_launch_uid=" + std::to_string(launch_uid) +
          " trace_kernel=" + std::to_string(key.trace_kernel_id) +
          " global_cta=" + std::to_string(global_cta_id) +
          " cta_slot=" + std::to_string(key.cta_id) +
          " warp=" + std::to_string(warp_id) + " barrier=0x" +
          format_hex_u64(key.barrier_addr) +
          " generation=" + std::to_string(barrier.gen) +
          " contract_id=" + std::to_string(contract_id) +
          " expected=" + std::to_string(expected_tx_bytes) +
          " warp_open_contracts=" +
          std::to_string(m_tma_open_contracts_per_warp[warp_id]));
}

void SM::claim_tma_completion_target(const warp_inst_t &inst,
                                     TMACommand &cmd) {
  if (cmd.transfer_type != TMATransferType::LOAD) {
    if (inst.tma_completion_runtime_valid) {
      ++m_tma_contract_mismatches;
      std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=nonload-has-completion-target"
                << " trace_kernel=" << inst.trace_kernel_id
                << " warp=" << inst.warp_id() << " pc=0x"
                << format_hex_u64(inst.pc) << std::endl;
      debug_dump_tma_binding_snapshot("nonload-has-completion-target");
      abort();
    }
    return;
  }

  const unsigned int warp_id = inst.warp_id();
  if (!inst.tma_completion_runtime_valid ||
      inst.tma_completion_barrier_addr == 0 ||
      (inst.tma_completion_barrier_addr & 0x7) != 0 ||
      inst.tma_completion_barrier_ureg_id == 0 ||
      warp_id >= m_tma_open_contracts_per_warp.size() ||
      cmd.total_bytes == 0 ||
      cmd.warp_id != warp_id || cmd.sm_id != m_sm_id ||
      cmd.trace_kernel_id != inst.trace_kernel_id ||
      cmd.unique_function_id != inst.unique_function_id ||
      cmd.tma_pc != inst.pc || cmd.cta_id >= MAX_CTA_PER_SHADER ||
      cmd.cta_id != m_physical_warp[warp_id]->get_cta_id()) {
    ++m_tma_contract_mismatches;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=missing-or-invalid-completion-target"
              << " trace_kernel=" << inst.trace_kernel_id
              << " warp=" << warp_id << " pc=0x"
              << format_hex_u64(inst.pc)
              << " payload_valid=" << inst.tma_completion_runtime_valid
              << " barrier=0x"
              << format_hex_u64(inst.tma_completion_barrier_addr)
              << " barrier_ureg=" << inst.tma_completion_barrier_ureg_id
              << " memory_bytes=" << cmd.total_bytes
              << " cmd_warp=" << cmd.warp_id
              << " cmd_sm=" << cmd.sm_id
              << " cmd_trace_kernel=" << cmd.trace_kernel_id
              << " cmd_function=" << cmd.unique_function_id
              << " cmd_pc=0x" << format_hex_u64(cmd.tma_pc)
              << " cmd_cta=" << cmd.cta_id << std::endl;
    debug_dump_tma_binding_snapshot("missing-completion-target");
    abort();
  }

  HopperMBarrierKey key;
  key.trace_kernel_id = inst.trace_kernel_id;
  key.cta_id = cmd.cta_id;
  key.barrier_addr = inst.tma_completion_barrier_addr;
  const uint64_t launch_uid = m_physical_warp[warp_id]->m_kernel_id;
  const uint32_t global_cta_id = get_global_cta_id(cmd.cta_id);

  auto barrier_it = m_hopper_mbarriers.find(key);
  if (barrier_it == m_hopper_mbarriers.end()) {
    ++m_tma_contract_mismatches;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=claim-barrier-missing"
              << " trace_kernel=" << inst.trace_kernel_id
              << " global_cta=" << global_cta_id
              << " cta_slot=" << cmd.cta_id << " warp=" << warp_id
              << " pc=0x" << format_hex_u64(inst.pc)
              << " payload_barrier=0x"
              << format_hex_u64(key.barrier_addr) << std::endl;
    debug_dump_tma_binding_snapshot("claim-barrier-missing");
    abort();
  }

  HopperMBarrierObject &barrier = barrier_it->second;
  if (launch_uid == 0 || !barrier.assignment_open ||
      barrier.assignment_owner_warp_id != warp_id ||
      barrier.assignment_kernel_launch_uid != launch_uid ||
      barrier.assignment_global_cta_id != global_cta_id ||
      m_tma_open_contracts_per_warp[warp_id] == 0) {
    ++m_tma_contract_mismatches;
    if (barrier.assignment_kernel_launch_uid != launch_uid ||
        barrier.assignment_global_cta_id != global_cta_id) {
      ++m_tma_launch_or_cta_mismatch;
    }
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=tx-contract-unresolved"
              << " trace_kernel=" << inst.trace_kernel_id
              << " global_cta=" << global_cta_id
              << " cta_slot=" << cmd.cta_id << " warp=" << warp_id
              << " pc=0x" << format_hex_u64(inst.pc)
              << " payload_barrier=0x"
              << format_hex_u64(key.barrier_addr)
              << " assignment_open=" << barrier.assignment_open
              << " owner_warp=" << barrier.assignment_owner_warp_id
              << " assignment_launch="
              << barrier.assignment_kernel_launch_uid
              << " assignment_global_cta="
              << barrier.assignment_global_cta_id
              << " contract_id=" << barrier.tx_contract_id
              << " warp_open_contracts="
              << m_tma_open_contracts_per_warp[warp_id]
              << std::endl;
    debug_dump_tma_binding_snapshot("tx-contract-unresolved");
    abort();
  }

  if (barrier.assignment_generation != barrier.gen ||
      barrier.tx_contract_id == 0) {
    ++m_tma_contract_mismatches;
    if (barrier.assignment_generation != barrier.gen) {
      ++m_tma_stale_generation;
    }
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=claim-contract-state-mismatch"
              << " contract_id=" << barrier.tx_contract_id
              << " contract_generation=" << barrier.assignment_generation
              << " barrier_generation=" << barrier.gen
              << " assignment_open=" << barrier.assignment_open << std::endl;
    debug_dump_tma_binding_snapshot("claim-contract-state-mismatch");
    abort();
  }

  const uint64_t assigned_after =
      static_cast<uint64_t>(barrier.assigned_tx_bytes) + cmd.total_bytes;
  if (assigned_after > barrier.expected_tx_bytes) {
    ++m_tma_contract_mismatches;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=tx-contract-overassigned"
              << " contract_id=" << barrier.tx_contract_id
              << " expected=" << barrier.expected_tx_bytes
              << " assigned_before=" << barrier.assigned_tx_bytes
              << " credit=" << cmd.total_bytes << std::endl;
    debug_dump_tma_binding_snapshot("tx-contract-overassigned");
    abort();
  }

  TmaCompletionTarget target;
  target.valid = true;
  target.key = key;
  target.barrier_generation = barrier.gen;
  target.kernel_launch_uid = launch_uid;
  target.global_cta_id = global_cta_id;
  target.owner_warp_id = warp_id;
  target.source_barrier_ureg_id = inst.tma_completion_barrier_ureg_id;
  target.contract_id = barrier.tx_contract_id;
  target.binding_id = m_next_tma_binding_id++;
  target.credit_bytes = cmd.total_bytes;
  cmd.completion = target;

  barrier.assigned_tx_bytes = static_cast<uint32_t>(assigned_after);
  ++barrier.outstanding_tma_bindings;
  ++m_tma_outstanding_bindings_per_warp[warp_id];
  ++m_tma_bindings_created;
  m_tma_assigned_credit_bytes += target.credit_bytes;
  if (m_config->wait_block_late_arrival_instrument_enable) {
    WaitBlockGenerationTiming &timing =
        barrier.wait_block_timing[barrier.gen + 1];
    const uint64_t now = get_current_gpu_cycle();
    if (!timing.tma_issue_valid) {
      timing.tma_issue_valid = true;
      timing.first_tma_issue_cycle = now;
    }
    timing.last_tma_issue_cycle = now;
    ++timing.tma_commands_issued;
  }

  if (barrier.assigned_tx_bytes == barrier.expected_tx_bytes) {
    if (m_tma_open_contracts_per_warp[warp_id] == 0) {
      ++m_tma_contract_mismatches;
      std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
                << get_current_gpu_cycle()
                << " reason=open-contract-counter-underflow"
                << " warp=" << warp_id
                << " contract_id=" << barrier.tx_contract_id << std::endl;
      debug_dump_tma_binding_snapshot("open-contract-counter-underflow");
      abort();
    }
    barrier.assignment_open = false;
    --m_tma_open_contracts_per_warp[warp_id];
    ++m_tma_contracts_fully_assigned;
  }

  debug_log_tma_binding_event(
      "assign",
      "kernel_launch_uid=" + std::to_string(launch_uid) +
          " trace_kernel=" + std::to_string(key.trace_kernel_id) +
          " global_cta=" + std::to_string(global_cta_id) +
          " cta_slot=" + std::to_string(key.cta_id) +
          " warp=" + std::to_string(warp_id) + " tma_pc=0x" +
          format_hex_u64(inst.pc) + " opcode=" +
          std::to_string(static_cast<int>(cmd.opcode_family)) +
          " barrier=0x" + format_hex_u64(key.barrier_addr) +
          " generation=" + std::to_string(target.barrier_generation) +
          " contract_id=" + std::to_string(target.contract_id) +
          " binding_id=" + std::to_string(target.binding_id) +
          " memory_bytes=" + std::to_string(cmd.total_bytes) +
          " credit_bytes=" + std::to_string(target.credit_bytes) +
          " expected=" + std::to_string(barrier.expected_tx_bytes) +
          " assigned=" + std::to_string(barrier.assigned_tx_bytes) +
          " outstanding=" +
          std::to_string(barrier.outstanding_tma_bindings) +
          " warp_open_contracts=" +
          std::to_string(m_tma_open_contracts_per_warp[warp_id]));
}

void SM::complete_tma_load(TmaCompletionTarget &target,
                           uint32_t memory_bytes, uint64_t transfer_uid) {
  if (!target.valid || target.completed || target.credit_bytes == 0 ||
      target.binding_id == 0 || target.contract_id == 0 ||
      target.key.barrier_addr == 0 ||
      target.key.cta_id >= MAX_CTA_PER_SHADER ||
      target.owner_warp_id >=
          m_tma_max_completed_transfer_uid_per_warp.size()) {
    if (target.completed) {
      ++m_tma_duplicate_completion;
    } else {
      ++m_tma_contract_mismatches;
    }
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason="
              << (target.completed ? "duplicate-binding-completion"
                                   : "invalid-completion-target")
              << " transfer_uid=" << transfer_uid
              << " binding_id=" << target.binding_id
              << " credit_bytes=" << target.credit_bytes << std::endl;
    debug_dump_tma_binding_snapshot("invalid-completion");
    abort();
  }

  const uint64_t live_launch_uid =
      m_physical_warp[target.owner_warp_id]->m_kernel_id;
  const uint32_t live_cta_id =
      m_physical_warp[target.owner_warp_id]->get_cta_id();
  const uint32_t live_global_cta = get_global_cta_id(target.key.cta_id);
  if (live_launch_uid == 0 ||
      target.kernel_launch_uid != live_launch_uid ||
      target.key.cta_id != live_cta_id ||
      target.global_cta_id != live_global_cta) {
    ++m_tma_launch_or_cta_mismatch;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=completion-launch-or-cta-mismatch"
              << " transfer_uid=" << transfer_uid
              << " binding_id=" << target.binding_id
              << " target_launch=" << target.kernel_launch_uid
              << " live_launch=" << live_launch_uid
              << " target_cta_slot=" << target.key.cta_id
              << " live_cta_slot=" << live_cta_id
              << " target_global_cta=" << target.global_cta_id
              << " live_global_cta=" << live_global_cta << std::endl;
    debug_dump_tma_binding_snapshot("completion-identity-mismatch");
    abort();
  }

  auto barrier_it = m_hopper_mbarriers.find(target.key);
  if (barrier_it == m_hopper_mbarriers.end()) {
    ++m_tma_orphan_cleanup;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=completion-barrier-missing"
              << " transfer_uid=" << transfer_uid
              << " binding_id=" << target.binding_id << " barrier=0x"
              << format_hex_u64(target.key.barrier_addr) << std::endl;
    debug_dump_tma_binding_snapshot("completion-barrier-missing");
    abort();
  }

  HopperMBarrierObject &barrier = barrier_it->second;
  if (target.barrier_generation != barrier.gen ||
      target.barrier_generation != barrier.assignment_generation) {
    ++m_tma_stale_generation;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=stale-barrier-generation"
              << " transfer_uid=" << transfer_uid
              << " binding_id=" << target.binding_id
              << " target_generation=" << target.barrier_generation
              << " barrier_generation=" << barrier.gen
              << " assignment_generation="
              << barrier.assignment_generation << std::endl;
    debug_dump_tma_binding_snapshot("stale-barrier-generation");
    abort();
  }

  const uint64_t completed_after =
      static_cast<uint64_t>(barrier.completed_tx_bytes) +
      target.credit_bytes;
  if (target.contract_id == 0 ||
      target.contract_id != barrier.tx_contract_id ||
      target.owner_warp_id != barrier.assignment_owner_warp_id ||
      target.kernel_launch_uid !=
          barrier.assignment_kernel_launch_uid ||
      target.global_cta_id != barrier.assignment_global_cta_id ||
      memory_bytes != target.credit_bytes ||
      completed_after > barrier.assigned_tx_bytes ||
      barrier.assigned_tx_bytes > barrier.expected_tx_bytes ||
      barrier.outstanding_tma_bindings == 0 ||
      m_tma_outstanding_bindings_per_warp[target.owner_warp_id] == 0) {
    ++m_tma_contract_mismatches;
    std::cerr << "[TMABIND][abort][SM" << m_sm_id << "] cyc="
              << get_current_gpu_cycle()
              << " reason=completion-contract-mismatch"
              << " transfer_uid=" << transfer_uid
              << " binding_id=" << target.binding_id
              << " target_contract=" << target.contract_id
              << " barrier_contract=" << barrier.tx_contract_id
              << " target_warp=" << target.owner_warp_id
              << " barrier_owner_warp="
              << barrier.assignment_owner_warp_id
              << " memory_bytes=" << memory_bytes
              << " credit_bytes=" << target.credit_bytes
              << " expected=" << barrier.expected_tx_bytes
              << " assigned=" << barrier.assigned_tx_bytes
              << " completed_before=" << barrier.completed_tx_bytes
              << " outstanding=" << barrier.outstanding_tma_bindings
              << " warp_outstanding="
              << m_tma_outstanding_bindings_per_warp[target.owner_warp_id]
              << std::endl;
    debug_dump_tma_binding_snapshot("completion-contract-mismatch");
    abort();
  }

  const uint32_t completed_before = barrier.completed_tx_bytes;
  target.completed = true;
  --barrier.outstanding_tma_bindings;
  --m_tma_outstanding_bindings_per_warp[target.owner_warp_id];
  barrier.completed_tx_bytes = static_cast<uint32_t>(completed_after);
  ++m_tma_bindings_completed;
  m_tma_completed_credit_bytes += target.credit_bytes;
  uint64_t &max_completed_uid =
      m_tma_max_completed_transfer_uid_per_warp[target.owner_warp_id];
  if (max_completed_uid != 0 && transfer_uid < max_completed_uid) {
    ++m_tma_uid_reorders;
  }
  max_completed_uid = std::max(max_completed_uid, transfer_uid);
  ++m_sync_debug_tma_completions;

  debug_log_tma_binding_event(
      "complete",
      "kernel_launch_uid=" + std::to_string(target.kernel_launch_uid) +
          " trace_kernel=" +
          std::to_string(target.key.trace_kernel_id) +
          " global_cta=" + std::to_string(target.global_cta_id) +
          " cta_slot=" + std::to_string(target.key.cta_id) +
          " warp=" + std::to_string(target.owner_warp_id) +
          " transfer_uid=" + std::to_string(transfer_uid) +
          " barrier=0x" + format_hex_u64(target.key.barrier_addr) +
          " generation=" + std::to_string(target.barrier_generation) +
          " contract_id=" + std::to_string(target.contract_id) +
          " binding_id=" + std::to_string(target.binding_id) +
          " credit_bytes=" + std::to_string(target.credit_bytes) +
          " completed_before=" + std::to_string(completed_before) +
          " completed_after=" +
          std::to_string(barrier.completed_tx_bytes) +
          " expected=" + std::to_string(barrier.expected_tx_bytes) +
          " outstanding=" +
          std::to_string(barrier.outstanding_tma_bindings));
  debug_log_sync_event(
      "tma_complete warp=" + std::to_string(target.owner_warp_id) +
      " cta=" + std::to_string(target.key.cta_id) + " barrier=0x" +
      format_hex_u64(target.key.barrier_addr) + " gen=" +
      std::to_string(barrier.gen) + " phase=" +
      std::to_string(barrier.phase) + " applied=" +
      std::to_string(target.credit_bytes) + " completed_tx=" +
      std::to_string(barrier.completed_tx_bytes) + "/" +
      std::to_string(barrier.expected_tx_bytes) + " binding_id=" +
      std::to_string(target.binding_id));
  recompute_sync_barrier_ready_and_maybe_flip_phase(target.key, barrier);
}

bool SM::are_all_wait_barrier_ready(unsigned int warp_id) {
  std::vector<Wait_Barrier_Checking> wait_barriers_checking;
  for (unsigned int i = 0; i < m_config->num_wait_barriers_per_warp; i++) {
    wait_barriers_checking.push_back(Wait_Barrier_Checking(i, 0));
  }
  bool are_wait_barriers_ready =
      m_physical_warp[warp_id]
          ->get_dependency_state()
          ->are_wait_barriers_ready(wait_barriers_checking);
  return are_wait_barriers_ready;
}

bool SM::warp_waiting_at_mem_barrier(unsigned warp_id) {
  if (!m_physical_warp[warp_id]->get_membar()) return false;
  bool use_traditional_scoreboarding = !m_physical_warp[warp_id]->get_kernel_info()->is_captured_from_binary || m_config->is_remodeling_scoreboarding_enabled || !m_config->is_trace_mode;
  bool clear_membar = false;
  if (use_traditional_scoreboarding) {
    // Baseline / non-trace model: keep the original register-scoreboard drain.
    clear_membar = (!m_scoreboard->pendingWrites(warp_id) && !m_scoreboard_WAR->pendingReads(warp_id)) ;
  } else {
    // Trace model: scope-aware fence. A MEMBAR/FENCE is an ordering fence (not a
    // CTA rendezvous): drain only this warp's stores that must be visible at the
    // fence scope.
    //   CTA -> shared + L1-level stores; GPU -> + L2-level + TMA stores.
    membar_scope_t scope = m_physical_warp[warp_id]->get_membar_scope();
    if (scope == MEMBAR_SCOPE_NONE) {
      // Unmapped/legacy fence form: nothing scope-specific to wait on.
      clear_membar = true;
    } else {
      clear_membar = !warp_has_pending_fence_stores(warp_id, (int)scope);
    }
  }
  if(clear_membar) {
    if (m_config->bar_debug_enable &&
        m_config->bar_debug_issue_budget > 0) {
      std::ostringstream oss;
      oss << "[MEMBARDBG][fence-release] sm=" << m_sm_id
          << " warp=" << warp_id
          << " scope=" << (int)m_physical_warp[warp_id]->get_membar_scope()
          << " cycle=" << get_current_gpu_cycle() << "\n";
      std::cerr << oss.str();
    }
    m_membar_wait_start_cycle.erase(warp_id);
    m_membar_last_stuck_warn_cycle.erase(warp_id);
    m_physical_warp[warp_id]->clear_membar();
    if (m_gpu->get_config().flush_l1()) {
      // Mahmoud fixed this on Nov 2019
      // Invalidate L1 cache
      // Based on Nvidia Doc, at MEM barrier, we have to
      //(1) wait for all pending writes till they are acked
      //(2) invalidate L1 cache to ensure coherence and avoid reading stall data
      data_cache_invalidate();
    }
  } else {
    // Deadlock watchdog: a scope-aware fence should drain in at most a few hundred
    // cycles. If a warp stays blocked far longer, a per-warp store counter likely
    // never got its matching dec (counter leak -> deadlock). Emit a periodic dump
    // of the exact pending counts so the offending level is obvious.
    unsigned long long now = get_current_gpu_cycle();
    auto it = m_membar_wait_start_cycle.find(warp_id);
    if (it == m_membar_wait_start_cycle.end()) {
      m_membar_wait_start_cycle[warp_id] = now;
    } else {
      const unsigned long long kStuckThreshold = 5000;   // cycles blocked
      const unsigned long long kWarnInterval = 5000;     // re-warn cadence
      unsigned long long waited = now - it->second;
      if (waited >= kStuckThreshold) {
        unsigned long long last = m_membar_last_stuck_warn_cycle.count(warp_id)
                                      ? m_membar_last_stuck_warn_cycle[warp_id]
                                      : 0;
        if (now - last >= kWarnInterval) {
          m_membar_last_stuck_warn_cycle[warp_id] = now;
          std::ostringstream oss;
          oss << "[MEMBARDBG][stuck] sm=" << m_sm_id << " warp=" << warp_id
              << " scope=" << (int)m_physical_warp[warp_id]->get_membar_scope()
              << " waited=" << waited
              << " cta=" << m_physical_warp[warp_id]->get_pending_stores_cta_visible()
              << " gpu=" << m_physical_warp[warp_id]->get_pending_stores_gpu_visible()
              << " tma=" << (m_tma_unit_shared_of_sm->warp_has_outstanding_stores(warp_id) ? 1 : 0)
              << " stores_outstanding=" << (m_physical_warp[warp_id]->stores_done() ? 0 : 1)
              << " cycle=" << now << "\n";
          std::cerr << oss.str();
        }
      }
    }
  }
  return !clear_membar;
}


bool SM::warp_waiting_grid_barrier(unsigned warp_id) {
  return m_physical_warp[warp_id]->get_gridbar();
}

// [SR-M0A] Pure, non-mutating wait-state peeks. See STALL_REASON_ALIGN.md SR-M0A.6.
// These reproduce the readiness predicates used by the authoritative issue path
// WITHOUT any of its side effects (no clear, no wait-set insert/erase, no cache
// invalidate, no watchdog-map update, no log emission).

WaitBlockPendingMode SM::peek_wait_block_hold_mode(unsigned warp_id) const {
  // Inspect ONLY the pending WaitBlock state, never the programmer/named-barrier
  // set (that is peek_named_barrier_wait's job). Returns NONE when there is no
  // unresolved WaitBlock hold.
  if (warp_id >= m_pending_sync_waits.size()) {
    return WaitBlockPendingMode::NONE;
  }
  const HopperMBarrierPendingWait &pending_wait = m_pending_sync_waits[warp_id];
  if (!pending_wait.valid ||
      is_wait_block_pending_satisfied(pending_wait)) {
    return WaitBlockPendingMode::NONE;
  }
  return pending_wait.mode;
}

bool SM::peek_named_barrier_wait(unsigned warp_id) const {
  return m_barriers.warp_waiting_at_barrier(warp_id);
}

bool SM::peek_membar_wait(unsigned warp_id) const {
  // Mirror warp_waiting_at_mem_barrier()'s drain predicate with no clear/log/
  // watchdog/L1-invalidate side effects.
  if (!m_physical_warp[warp_id]->get_membar()) return false;
  bool use_traditional_scoreboarding =
      !m_physical_warp[warp_id]->get_kernel_info()->is_captured_from_binary ||
      m_config->is_remodeling_scoreboarding_enabled || !m_config->is_trace_mode;
  bool clear_membar = false;
  if (use_traditional_scoreboarding) {
    clear_membar = (!m_scoreboard->pendingWrites(warp_id) &&
                    !m_scoreboard_WAR->pendingReads(warp_id));
  } else {
    membar_scope_t scope = m_physical_warp[warp_id]->get_membar_scope();
    if (scope == MEMBAR_SCOPE_NONE) {
      clear_membar = true;
    } else {
      clear_membar = !warp_has_pending_fence_stores(warp_id, (int)scope);
    }
  }
  return !clear_membar;
}

bool SM::peek_grid_barrier_wait(unsigned warp_id) const {
  return m_physical_warp[warp_id]->get_gridbar();
}

bool SM::peek_tma_flush_wait(unsigned warp_id, const warp_inst_t *inst) const {
  if (inst == nullptr ||
      inst->tma_opcode_family != TMAOpcodeFamily::UTMACMDFLUSH) {
    return false;
  }
  return m_tma_unit_shared_of_sm->warp_has_outstanding_stores(warp_id);
}

bool SM::warp_waiting_at_tma_flush(unsigned warp_id, const warp_inst_t *pI) {
  if (pI == nullptr ||
      pI->tma_opcode_family != TMAOpcodeFamily::UTMACMDFLUSH) {
    return false;
  }
  bool waiting =
      m_tma_unit_shared_of_sm->warp_has_outstanding_stores(warp_id);
  bool was_waiting =
      m_warps_waiting_tma_flush.find(warp_id) != m_warps_waiting_tma_flush.end();
  if (waiting && !was_waiting) {
    // Stall episode begins: log once on entry (not every cycle).
    m_warps_waiting_tma_flush.insert(warp_id);
    debug_log_tma_event("flush-wait-enter warp=" + std::to_string(warp_id) +
                        " pc=" + format_hex_u64(pI->pc));
  } else if (!waiting && was_waiting) {
    // All store-class transfers drained: warp may now issue the flush.
    m_warps_waiting_tma_flush.erase(warp_id);
    debug_log_tma_event("flush-wait-release warp=" + std::to_string(warp_id) +
                        " pc=" + format_hex_u64(pI->pc));
  } else if (!waiting && !was_waiting) {
    // Flush issued with no outstanding store-class transfers (pass-through).
    debug_log_tma_event("flush-pass warp=" + std::to_string(warp_id) +
                        " pc=" + format_hex_u64(pI->pc));
  }
  return waiting;
}

void SM::clear_gridbar(unsigned int kernel_id) {
  for (unsigned int i = 0; i < m_config->max_warps_per_shader; i++) {
    if (m_physical_warp[i]->m_kernel_id == kernel_id) {
      m_physical_warp[i]->clear_gridbar();
    }
  }
}

void SM::decrement_atomic_count(unsigned wid, unsigned n) {
  assert(m_physical_warp[wid]->get_n_atomic() >= n);
  m_physical_warp[wid]->dec_n_atomic(n);
}

unsigned int SM::get_atomic_count(unsigned wid) {
  return m_physical_warp[wid]->get_n_atomic();
}

void SM::broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                     warp_set_t warps) {
  for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
    if (warps.test(i)) {
      const warp_inst_t *inst =
          m_physical_warp[i]->restore_info_of_last_inst_at_barrier();
      const_cast<warp_inst_t *>(inst)->broadcast_barrier_reduction(
          inst->get_active_mask());
    }
  }
}

// Block Occupancy Functions

void SM::set_max_cta(const kernel_info_t &kernel) {
  // calculate the max cta count and cta size for local memory address mapping
  kernel_max_cta_per_shader = m_config->max_cta(kernel);
  unsigned int gpu_cta_size = kernel.threads_per_cta();
  kernel_padded_threads_per_cta =
      (gpu_cta_size % m_config->warp_size)
          ? m_config->warp_size * ((gpu_cta_size / m_config->warp_size) + 1)
          : gpu_cta_size;
}

unsigned int SM::get_n_active_cta() const { return m_n_active_cta; }
unsigned int SM::get_not_completed() const { return m_not_completed; }

unsigned int SM::isactive() const {
  if (m_n_active_cta > 0)
    return 1;
  else
    return 0;
}

float SM::get_current_occupancy(unsigned long long &active,
                                unsigned long long &total) const {
  // To match the achieved_occupancy in nvprof, only SMs that are active are
  // counted toward the occupancy.
  if (m_active_warps > 0) {
    total += m_physical_warp.size();
    active += m_active_warps;
    return float(active) / float(total);
  } else {
    return 0;
  }
}

void SM::issue_block2core(kernel_info_t &kernel) {
  if (!m_config->gpgpu_concurrent_kernel_sm)
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  kernel.inc_running();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  // Record local→global CTA ID mapping for ground truth validation
  m_local_to_global_cta_id[free_cta_hw_id] = ctaid;
  std::unique_ptr<checkpoint> g_checkpoint = std::make_unique<checkpoint>();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sm_id, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  enqueue_instruction_region_prewarm(kernel);
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  shader_CTA_count_log(m_sm_id, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(),
                 kernel.get_name().c_str());
}

bool SM::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int SM::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool SM::occupy_shader_resource_1block(kernel_info_t &k, bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas, on shader %d\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas, m_sm_id);
  }

  return true;
}

void SM::release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

void SM::warp_inst_complete(const warp_inst_t &inst) {
#if 0
      printf("[warp_inst_complete] uid=%u core=%u warp=%u pc=%#x @ time=%llu \n",
             inst.get_uid(), m_sm_id, inst.warp_id(), inst.pc,  m_gpu->gpu_tot_sim_cycle +  m_gpu->gpu_sim_cycle);
#endif

  if (inst.op_pipe == SP__OP)
    m_stats->m_num_sp_committed[m_sm_id]++;
  else if (inst.op_pipe == SFU__OP)
    m_stats->m_num_sfu_committed[m_sm_id]++;
  else if (inst.op_pipe == MEM__OP)
    m_stats->m_num_mem_committed[m_sm_id]++;

  if (m_config->gpgpu_clock_gated_lanes == false)
    m_stats->m_num_sim_insn[m_sm_id] += m_config->warp_size;
  else
    m_stats->m_num_sim_insn[m_sm_id] += inst.active_count();

  m_stats->m_num_sim_winsn[m_sm_id]++;

  m_stats
      ->m_num_sim_winsn_per_shader_per_kernel[m_stats->m_current_kernel_pos]
                                             [m_sm_id]++;  // MOD. Custom Stats
  m_stats->m_num_sim_winsn_per_shader[m_sm_id]++;          // MOD. Custom Stats

  m_sm_stats.m_stats_map["gpu_sim_insn"]->increment_with_integer(inst.active_count());
  m_sm_stats.m_stats_map["total_num_warp_instructions"]->increment_with_integer(1);

  if(inst.is_load() || inst.is_store()) {
    m_sm_stats.m_stats_map["total_num_ldst_unit_instructions"]->increment_with_integer(1);
  }else if(inst.is_dp_op()) {
    m_sm_stats.m_stats_map["total_num_dp_instructions"]->increment_with_integer(1);
  }

  inst.completed(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

  if (inst.active_count() < 32 && inst.active_count() != 0) {
    m_sm_stats.m_stats_map["Total_effective_incomplete_warps"]->increment_with_integer(1);
  }
}

bool SM::ptx_thread_done(unsigned hw_thread_id) const {
  return this->core_t::ptx_thread_done(hw_thread_id);
}

void SM::dec_inst_in_pipeline(unsigned warp_id) {
  m_physical_warp[warp_id]->dec_inst_in_pipeline();
}

void SM::store_ack(class mem_fetch *mf) {
  assert(mf->get_type() == WRITE_ACK ||
         (m_config->gpgpu_perfect_mem && mf->get_is_write()));
  unsigned warp_id = mf->get_wid();
  m_physical_warp[warp_id]->dec_store_req();
  // Scope-aware fence accounting: decrement the per-warp visibility-level store
  // counter this request was tagged with at issue time (see ldst_unit_sm store
  // issue sites). One mem_fetch == one sector ack, matching the inc granularity.
  mem_fetch::fence_visibility_level_t lvl = mf->get_fence_visibility_level();
  if (lvl != mem_fetch::FENCE_VIS_NONE) {
    dec_fence_store(warp_id, (int)lvl, 1);
  }
}

void SM::inc_store_req(unsigned warp_id) {
  m_physical_warp[warp_id]->inc_store_req();
}

void SM::inc_fence_store(unsigned warp_id, int fence_vis_level, unsigned n) {
  if (fence_vis_level == mem_fetch::FENCE_VIS_CTA) {
    m_physical_warp[warp_id]->inc_pending_stores_cta_visible(n);
  } else if (fence_vis_level == mem_fetch::FENCE_VIS_GPU) {
    m_physical_warp[warp_id]->inc_pending_stores_gpu_visible(n);
  }
  if (m_config->bar_debug_enable &&
      m_config->bar_debug_issue_budget > 0) {
    std::ostringstream oss;
    oss << "[MEMBARDBG][store++] sm=" << m_sm_id << " warp=" << warp_id
        << " level=" << fence_vis_level << " n=" << n
        << " cta=" << m_physical_warp[warp_id]->get_pending_stores_cta_visible()
        << " gpu=" << m_physical_warp[warp_id]->get_pending_stores_gpu_visible()
        << " cycle=" << get_current_gpu_cycle() << "\n";
    std::cerr << oss.str();
  }
}

void SM::dec_fence_store(unsigned warp_id, int fence_vis_level, unsigned n) {
  if (fence_vis_level == mem_fetch::FENCE_VIS_CTA) {
    m_physical_warp[warp_id]->dec_pending_stores_cta_visible(n);
  } else if (fence_vis_level == mem_fetch::FENCE_VIS_GPU) {
    m_physical_warp[warp_id]->dec_pending_stores_gpu_visible(n);
  }
  if (m_config->bar_debug_enable &&
      m_config->bar_debug_issue_budget > 0) {
    std::ostringstream oss;
    oss << "[MEMBARDBG][store--] sm=" << m_sm_id << " warp=" << warp_id
        << " level=" << fence_vis_level << " n=" << n
        << " cta=" << m_physical_warp[warp_id]->get_pending_stores_cta_visible()
        << " gpu=" << m_physical_warp[warp_id]->get_pending_stores_gpu_visible()
        << " cycle=" << get_current_gpu_cycle() << "\n";
    std::cerr << oss.str();
  }
}

bool SM::warp_has_pending_fence_stores(unsigned warp_id, int membar_scope) const {
  unsigned cta_pending = m_physical_warp[warp_id]->get_pending_stores_cta_visible();
  if (membar_scope == MEMBAR_SCOPE_CTA) {
    // CTA fence: shared + L1-level stores must be visible (same SM).
    return cta_pending > 0;
  }
  if (membar_scope == MEMBAR_SCOPE_GPU) {
    // GPU fence subsumes CTA: also wait on L2-level global stores and TMA stores.
    unsigned gpu_pending = m_physical_warp[warp_id]->get_pending_stores_gpu_visible();
    bool tma_pending =
        m_tma_unit_shared_of_sm->warp_has_outstanding_stores(warp_id);
    return (cta_pending > 0) || (gpu_pending > 0) || tma_pending;
  }
  return false;
}

void SM::display_simt_state(FILE *fout, int mask) const {
  if ((mask & 4) && m_config->model == POST_DOMINATOR) {
    fprintf(fout, "per warp SIMT control-flow state:\n");
    unsigned n = m_config->n_thread_per_shader / m_config->warp_size;
    for (unsigned i = 0; i < n; i++) {
      unsigned nactive = 0;
      for (unsigned j = 0; j < m_config->warp_size; j++) {
        unsigned tid = i * m_config->warp_size + j;
        int done = ptx_thread_done(tid);
        nactive += (ptx_thread_done(tid) ? 0 : 1);
        if (done && (mask & 8)) {
          unsigned done_cycle = m_thread[tid]->donecycle();
          if (done_cycle) {
            printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle);
          }
        }
      }
      if (nactive == 0) {
        continue;
      }
      m_simt_stack[i]->print(fout);
    }
    fprintf(fout, "\n");
  }
}

void SM::display_pipeline(FILE *fout, int print_mem, int mask3bit) const {
  display_SM(fout, print_mem, mask3bit);
}

void SM::display_SM(FILE *fout, int print_mem, int mask3bit) const {
  throw std::logic_error("Function not yet implemented");
}

void SM::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_physical_warp[w]->print(fout);
}

// Stats Functions

void SM::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                           unsigned &dl1_misses) {
  m_ldst_unit_shared_of_sm->print_cache_stats(fp, dl1_accesses, dl1_misses);
}

void SM::get_cache_stats(cache_stats &cs) {
  // Adds stats from each cache to 'cs'
  cs += m_L1I_L1_half_C_cache->get_stats();                       // Get L1I stats
  m_ldst_unit_shared_of_sm->get_cache_stats(cs);  // Get L1D, L1C, L1T stats
}

void SM::get_L0I_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (auto subcore : m_subcores) {
    subcore->get_L0I_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

void SM::get_L1I_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1I_L1_half_C_cache) m_L1I_L1_half_C_cache->get_sub_stats(css);
}

void SM::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit_shared_of_sm->get_L1D_sub_stats(css);
}
void SM::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit_shared_of_sm->get_L1C_sub_stats(css);
}
void SM::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit_shared_of_sm->get_L1T_sub_stats(css);
}

void SM::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_sm_stats.m_stats_map["gpgpu_n_shmem_insn"]->increment_with_integer(
            active_count);
      break;
    case sstarr_space:
      m_sm_stats.m_stats_map["gpgpu_n_sstarr_insn"]->increment_with_integer(
            active_count);
      break;
    case const_space:
      m_sm_stats.m_stats_map["gpgpu_n_const_mem_insn"]->increment_with_integer(
            active_count);
      break;
    case param_space_kernel:
    case param_space_local:
      m_sm_stats.m_stats_map["gpgpu_n_param_mem_insn"]->increment_with_integer(
            active_count);
      break;
    case tex_space:
      m_sm_stats.m_stats_map["gpgpu_n_tex_insn"]->increment_with_integer(
            active_count);
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_sm_stats.m_stats_map["gpgpu_n_store_insn"]->increment_with_integer(
            active_count);
      else
        m_sm_stats.m_stats_map["gpgpu_n_load_insn"]->increment_with_integer(
            active_count);
      break;
    default:
      abort();
  }
}

unsigned SM::inactive_lanes_accesses_sfu(unsigned active_count,
                                         double latency) {
  return (((32 - active_count) >> 1) * latency) +
         (((32 - active_count) >> 3) * latency) +
         (((32 - active_count) >> 3) * latency);
}
unsigned SM::inactive_lanes_accesses_nonsfu(unsigned active_count,
                                            double latency) {
  return (((32 - active_count) >> 1) * latency);
}
void SM::incload_stat() { m_stats->m_num_loadqueued_insn[m_sm_id]++; }
void SM::incstore_stat() { m_stats->m_num_storequeued_insn[m_sm_id]++; }
void SM::incialu_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_ialu_acesses[m_sm_id] =
        m_stats->m_num_ialu_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_ialu_acesses[m_sm_id] =
        m_stats->m_num_ialu_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incimul_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_imul_acesses[m_sm_id] =
        m_stats->m_num_imul_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_imul_acesses[m_sm_id] =
        m_stats->m_num_imul_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incimul24_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_imul24_acesses[m_sm_id] =
        m_stats->m_num_imul24_acesses[m_sm_id] +
        (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_imul24_acesses[m_sm_id] =
        m_stats->m_num_imul24_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incimul32_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_imul32_acesses[m_sm_id] =
        m_stats->m_num_imul32_acesses[m_sm_id] +
        (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_imul32_acesses[m_sm_id] =
        m_stats->m_num_imul32_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incidiv_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_idiv_acesses[m_sm_id] =
        m_stats->m_num_idiv_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_idiv_acesses[m_sm_id] =
        m_stats->m_num_idiv_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incfpalu_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_fp_acesses[m_sm_id] =
        m_stats->m_num_fp_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_fp_acesses[m_sm_id] =
        m_stats->m_num_fp_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incfpmul_stat(unsigned active_count, double latency) {
  // printf("FP MUL stat increament\n");
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_fpmul_acesses[m_sm_id] =
        m_stats->m_num_fpmul_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_fpmul_acesses[m_sm_id] =
        m_stats->m_num_fpmul_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incfpdiv_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_fpdiv_acesses[m_sm_id] =
        m_stats->m_num_fpdiv_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_fpdiv_acesses[m_sm_id] =
        m_stats->m_num_fpdiv_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incdpalu_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_dp_acesses[m_sm_id] =
        m_stats->m_num_dp_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_dp_acesses[m_sm_id] =
        m_stats->m_num_dp_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incdpmul_stat(unsigned active_count, double latency) {
  // printf("FP MUL stat increament\n");
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_dpmul_acesses[m_sm_id] =
        m_stats->m_num_dpmul_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_dpmul_acesses[m_sm_id] =
        m_stats->m_num_dpmul_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incdpdiv_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_dpdiv_acesses[m_sm_id] =
        m_stats->m_num_dpdiv_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_dpdiv_acesses[m_sm_id] =
        m_stats->m_num_dpdiv_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::incsqrt_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_sqrt_acesses[m_sm_id] =
        m_stats->m_num_sqrt_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_sqrt_acesses[m_sm_id] =
        m_stats->m_num_sqrt_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::inclog_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_log_acesses[m_sm_id] =
        m_stats->m_num_log_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_log_acesses[m_sm_id] =
        m_stats->m_num_log_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::incexp_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_exp_acesses[m_sm_id] =
        m_stats->m_num_exp_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_exp_acesses[m_sm_id] =
        m_stats->m_num_exp_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::incsin_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_sin_acesses[m_sm_id] =
        m_stats->m_num_sin_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_sin_acesses[m_sm_id] =
        m_stats->m_num_sin_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::inctensor_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_tensor_core_acesses[m_sm_id] =
        m_stats->m_num_tensor_core_acesses[m_sm_id] +
        (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_tensor_core_acesses[m_sm_id] =
        m_stats->m_num_tensor_core_acesses[m_sm_id] +
        (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::inctex_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_tex_acesses[m_sm_id] =
        m_stats->m_num_tex_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_tex_acesses[m_sm_id] =
        m_stats->m_num_tex_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::inc_const_accesses(unsigned active_count) {
  m_stats->m_num_const_acesses[m_sm_id] =
      m_stats->m_num_const_acesses[m_sm_id] + active_count;
}

void SM::incsfu_stat(unsigned active_count, double latency) {
  m_stats->m_num_sfu_acesses[m_sm_id] =
      m_stats->m_num_sfu_acesses[m_sm_id] + (double)active_count * latency;
}
void SM::incsp_stat(unsigned active_count, double latency) {
  m_stats->m_num_sp_acesses[m_sm_id] =
      m_stats->m_num_sp_acesses[m_sm_id] + (double)active_count * latency;
}
void SM::incmem_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_mem_acesses[m_sm_id] =
        m_stats->m_num_mem_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_mem_acesses[m_sm_id] =
        m_stats->m_num_mem_acesses[m_sm_id] + (double)active_count * latency;
  }
}

void SM::incregfile_reads(unsigned active_count) {
  m_stats->m_read_regfile_acesses[m_sm_id] =
      m_stats->m_read_regfile_acesses[m_sm_id] + active_count;
}
void SM::incregfile_writes(unsigned active_count) {
  m_stats->m_write_regfile_acesses[m_sm_id] =
      m_stats->m_write_regfile_acesses[m_sm_id] + active_count;
}
void SM::incnon_rf_operands(unsigned active_count) {
  m_stats->m_non_rf_operands[m_sm_id] =
      m_stats->m_non_rf_operands[m_sm_id] + active_count;
}

void SM::incspactivelanes_stat(unsigned active_count) {
  m_stats->m_active_sp_lanes[m_sm_id] =
      m_stats->m_active_sp_lanes[m_sm_id] + active_count;
}
void SM::incsfuactivelanes_stat(unsigned active_count) {
  m_stats->m_active_sfu_lanes[m_sm_id] =
      m_stats->m_active_sfu_lanes[m_sm_id] + active_count;
}
void SM::incfuactivelanes_stat(unsigned active_count) {
  m_stats->m_active_fu_lanes[m_sm_id] =
      m_stats->m_active_fu_lanes[m_sm_id] + active_count;
}
void SM::incfumemactivelanes_stat(unsigned active_count) {
  m_stats->m_active_fu_mem_lanes[m_sm_id] =
      m_stats->m_active_fu_mem_lanes[m_sm_id] + active_count;
}

void SM::inc_simt_to_mem(unsigned n_flits) {
  m_stats->n_simt_to_mem[m_sm_id] += n_flits;
}

void SM::incexecstat(warp_inst_t *&inst) {
  // Latency numbers for next operations are used to scale the power values
  // for special operations, according observations from microbenchmarking
  // TODO: put these numbers in the xml configuration
  if (get_gpu()->get_config().g_power_simulation_enabled) {
    switch (inst->sp_op) {
      case INT__OP:
        incialu_stat(inst->active_count(), m_scaling_coeffs->int_coeff);
        break;
      case INT_MUL_OP:
        incimul_stat(inst->active_count(), m_scaling_coeffs->int_mul_coeff);
        break;
      case INT_MUL24_OP:
        incimul24_stat(inst->active_count(), m_scaling_coeffs->int_mul24_coeff);
        break;
      case INT_MUL32_OP:
        incimul32_stat(inst->active_count(), m_scaling_coeffs->int_mul32_coeff);
        break;
      case INT_DIV_OP:
        incidiv_stat(inst->active_count(), m_scaling_coeffs->int_div_coeff);
        break;
      case FP__OP:
        incfpalu_stat(inst->active_count(), m_scaling_coeffs->fp_coeff);
        break;
      case FP_MUL_OP:
        incfpmul_stat(inst->active_count(), m_scaling_coeffs->fp_mul_coeff);
        break;
      case FP_DIV_OP:
        incfpdiv_stat(inst->active_count(), m_scaling_coeffs->fp_div_coeff);
        break;
      case DP___OP:
        incdpalu_stat(inst->active_count(), m_scaling_coeffs->dp_coeff);
        break;
      case DP_MUL_OP:
        incdpmul_stat(inst->active_count(), m_scaling_coeffs->dp_mul_coeff);
        break;
      case DP_DIV_OP:
        incdpdiv_stat(inst->active_count(), m_scaling_coeffs->dp_div_coeff);
        break;
      case FP_SQRT_OP:
        incsqrt_stat(inst->active_count(), m_scaling_coeffs->sqrt_coeff);
        break;
      case FP_LG_OP:
        inclog_stat(inst->active_count(), m_scaling_coeffs->log_coeff);
        break;
      case FP_SIN_OP:
        incsin_stat(inst->active_count(), m_scaling_coeffs->sin_coeff);
        break;
      case FP_EXP_OP:
        incexp_stat(inst->active_count(), m_scaling_coeffs->exp_coeff);
        break;
      case TENSOR__OP:
        inctensor_stat(inst->active_count(), m_scaling_coeffs->tensor_coeff);
        break;
      case TEX__OP:
        inctex_stat(inst->active_count(), m_scaling_coeffs->tex_coeff);
        break;
      default:
        break;
    }
    if (inst->const_cache_operand)  // warp has const address space load as one
                                    // operand
      inc_const_accesses(1);
  }
}

void SM::get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const {
  n_simt_to_mem += m_stats->n_simt_to_mem[m_sm_id];
  n_mem_to_simt += m_stats->n_mem_to_simt[m_sm_id];
}

address_type SM::from_local_pc_to_global_pc_address(address_type local_pc, unsigned int unique_function_id) {
  address_type first_pc_of_kernel = get_gpu()->get_extra_trace_info().get_kernel_by_unique_function_id(unique_function_id).get_function_addr();
  address_type res = local_pc + first_pc_of_kernel;
  return res;
}

address_type SM::from_global_pc_address_to_local_pc(address_type global_pc, unsigned int unique_function_id) {
  address_type first_pc_of_kernel = get_gpu()->get_extra_trace_info().get_kernel_by_unique_function_id(unique_function_id).get_function_addr();
  address_type res = global_pc - first_pc_of_kernel;
  return res;
}
