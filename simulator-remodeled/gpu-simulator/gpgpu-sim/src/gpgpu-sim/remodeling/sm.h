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

#pragma once

#include "../../abstract_hardware_model.h"

#include <stdio.h>
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <stack>
#include <memory>
#include "ldst_unit_sm.h"

#include "../shader_core_wrapper.h"
#include "../shader.h"
#include "../../constants.h"
// #include "../../../../trace-driven/trace_driven.h"

#include "subcore.h"
#include "new_stats.h"

#define NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST 3
#define MULTIPLIER_LATENCY_READ_FIXED_LATENCY_INST_TENSOR_CORE_INSTS_WITH_4_REGS_PER_OP 2
#define MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST*MULTIPLIER_LATENCY_READ_FIXED_LATENCY_INST_TENSOR_CORE_INSTS_WITH_4_REGS_PER_OP
#define NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST+2 // 1 CONTROL, 1 ALLOCATE and 3 READ
#define NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST_TENSOR_CORE_INSTS_WITH_4_REGS_PER_OP MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST+2 // 1 CONTROL, 1 ALLOCATE and 3 READ


class shader_core_config;
class read_only_cache;
class gpgpu_sim;
class simt_core_cluster;
class shader_core_stats;
class memory_config;
class shader_core_mem_fetch_allocator;
class Scoreboard;
class Scoreboard_reads;
class functional_unit;
class coalescingStatsPerSm;
class coalescingStatsAcrossSms;
class tma_unit_sm;

enum subcore_dispatch_latch_name_t {
  DISPATCH_SP = 0, // FP32
  DISPATCH_DP, // FP64
  DISPATCH_HP, // FP16
  DISPATCH_INT,
  DISPATCH_SFU,
  DISPATCH_TENSOR_CORE,
  DISPATCH_UNIFORM,
  DISPATCH_BRANCH,
  DISPATCH_MISCELLANEOUS, // NOPS go here
  N_DISPATCH_LATCHES
};

const char *const subcore_dispatch_latch_decode[] = {
  "DISPATCH_SP",
  "DISPATCH_DP",
  "DISPATCH_HP", 
  "DISPATCH_INT",
  "DISPATCH_SFU",
  "DISPATCH_TENSOR_CORE",
  "DISPATCH_UNIFORM",
  "DISPATCH_BRANCH",
  "DISPATCH_MISCELLANEOUS",
  "N_DISPATCH_LATCHES"
};

class Waiting_Dep_Counters_per_Warp {
  public:
    Waiting_Dep_Counters_per_Warp(unsigned int warp_id) : m_warp_id(warp_id) {}
    Waiting_Dep_Counters_per_Warp() : m_warp_id(0) {}
    void increase_dep_counter(unsigned int dep_counter_id) {
      if(m_waiting_dep_counters.find(dep_counter_id) == m_waiting_dep_counters.end()) {
        m_waiting_dep_counters[dep_counter_id] = 1;
      } else {
        m_waiting_dep_counters[dep_counter_id]++;
      }
    }
    void decrease_dep_counter(unsigned int dep_counter_id) {
      assert(m_waiting_dep_counters.find(dep_counter_id) != m_waiting_dep_counters.end());
      assert(m_waiting_dep_counters[dep_counter_id] > 0);
      m_waiting_dep_counters[dep_counter_id]--;
      if(m_waiting_dep_counters[dep_counter_id] == 0) {
        m_waiting_dep_counters.erase(dep_counter_id);
      }
    }
    void clear() {
      m_waiting_dep_counters.clear();
    }
    unsigned int m_warp_id;
    std::map<int, unsigned int> m_waiting_dep_counters;
};
class InterWarp_Coalescing_Waiting_Dep_Counters {
  public:
    InterWarp_Coalescing_Waiting_Dep_Counters(unsigned int num_warps) {
      m_waiting_dep_counters_per_warp.resize(num_warps);
      for(unsigned int i = 0; i < num_warps; i++) {
        m_waiting_dep_counters_per_warp[i].m_warp_id = i;
      }
    }
    void clear() {
      for(unsigned int i = 0; i < m_waiting_dep_counters_per_warp.size(); i++) {
        m_waiting_dep_counters_per_warp[i].clear();
      }
    }
    std::vector<Waiting_Dep_Counters_per_Warp> m_waiting_dep_counters_per_warp;
};


unsigned int translate_warp_id_of_sm_to_subcore(unsigned int warp_id, unsigned int num_subcores);

TraceEnhancedOperandType get_reg_type_eval(traced_operand& op);

bool check_is_reserved_regs_remodeling(int reg, TraceEnhancedOperandType reg_type, bool is_trace_mode);

unsigned int translate_reg_to_global_id(int reg, TraceEnhancedOperandType reg_type);

class SM : public core_t, public shader_core_ctx_wrapper {
 public:
  SM(unsigned int num_subcores, gpgpu_sim *gpu, simt_core_cluster *cluster,
     unsigned shader_id, unsigned tpc_id, const shader_core_config *config,
     const memory_config *mem_config, shader_core_stats *stats);

  ~SM() override;
  void init() override;

  void num_cycles_to_stall_SM(unsigned int num_cycles);

  // Map a hardware CTA slot (get_cta_id()) to the linear global block index
  // (blockIdx.x + gridDim.x*blockIdx.y + ...). Recorded at issue_block2core.
  // Used by the TMA unit's M2 tile-spread so different CTAs address different
  // tiles of the same tensor instead of every CTA collapsing to tile 0.
  unsigned int get_global_cta_id(unsigned int hw_cta_slot) const {
    if (hw_cta_slot >= MAX_CTA_PER_SHADER) return 0;
    return m_local_to_global_cta_id[hw_cta_slot];
  }

  void cycle() override;

  void consume_pending_wait_barrier_actions(std::stack<Wait_Barrier_Entry_Modifier> &actions);
  void add_pending_wait_barrier_decrement(warp_inst_t *inst, Wait_Barrier_Type barrier_type, unsigned int barrier_id);
  void add_pending_wait_barrier_increment(warp_inst_t *inst, Wait_Barrier_Type barrier_type, unsigned int barrier_id);
  void instruction_retirement(warp_inst_t *instruction);
  void issue_warp(register_set_uniptr &warp, warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned subcore_id, bool use_traditional_scoreboarding);
  virtual void func_exec_inst(warp_inst_t &inst);
  void check_if_warp_has_finished_executing_and_can_be_reclaim(shd_warp_t *warp);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  // Returns numbers of addresses in translated_addrs, each addr points to a 4B (32-bit) word                                           
  unsigned int translate_local_memaddr(address_type localaddr, unsigned tid, unsigned num_shader, unsigned datasize,
                                   new_addr_type *translated_addrs);
  void create_logical_structures();
  void create_memory_interfaces();
  virtual void create_shd_warp();
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t &kernel);
  virtual unsigned int sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu);
  void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed) override;

  void register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel);
  address_type next_pc(int tid) const; // return the next pc of a thread

  void set_kernel(kernel_info_t *k) override;
  kernel_info_t *get_kernel() override;
  kernel_info_t *get_kernel_info() override;
  unsigned long long get_current_gpu_cycle() override;
  unsigned int get_num_subcores();
  unsigned int get_sid() const override;
  unsigned int get_tpc_id() const;
  unsigned int get_kernel_id(unsigned warp_id);
  gpgpu_sim *get_gpu() override;
  shader_core_mem_fetch_allocator &get_memf_fetch_allocator();
  read_only_cache *get_L1C();
  std::shared_ptr<Scoreboard_reads> get_scoreboard_WAR();
  std::shared_ptr<Scoreboard> get_scoreboard();
  const memory_config *get_memory_config() const;
  const shader_core_config *get_config() const override;
  shader_core_stats* get_stats() override;
  std::list<unsigned> get_regs_written(const inst_t &fvt) const;
  shd_warp_t *get_shd_warp(int id) override;
  int get_subcore_req_fetch_L1I_priority();
  void set_subcore_req_fetch_L1I_priority(int new_subcore_req_fetch_L1I_priority) override;
  void set_last_inst_gpu_sim_cycle(unsigned long long last_inst_gpu_sim_cycle);
  void set_last_inst_gpu_tot_sim_cycle(unsigned long long last_inst_gpu_tot_sim_cycle);
  bool is_any_subcore_problems_of_fordward_progress() const;
  bool get_is_loog_enabled() override;
  RRS* get_loog_rrs() override;

  void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const override;
  void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI, unsigned *pc, unsigned *rpc) override;
  virtual const active_mask_t &get_active_mask(unsigned warp_id, const warp_inst_t *pI);

  virtual void warp_exit(unsigned warp_id);

  void cache_flush() override;
  void data_cache_invalidate();
  void cache_invalidate() override;

  void accept_fetch_response(mem_fetch *mf) override;
  void accept_ldst_unit_response(class mem_fetch *mf) override;
  bool fetch_unit_response_buffer_full() const override;
  bool ldst_unit_response_buffer_full() const override;
  bool are_all_wait_barrier_ready(unsigned int warp_id);
  bool warp_waiting_at_barrier(unsigned warp_id) const override;
  // Non-mutating peek of the same hold condition as
  // warp_waiting_at_barrier, but with NO clear and NO counter bump. Used off the
  // authoritative issue-eligibility path (sort comparator, stall scans) so the
  // release side effect fires exactly once. See design doc 6.5.1.
  bool warp_held_on_mbarrier_wait(unsigned warp_id) const;
  // Once-per-warp-per-cycle authoritative release service. All-live
  // missing-wait holds only clear pending state; partial holds activate or
  // resume their owned replay stream; captured-poll holds fold residual loop
  // occurrences to their exact local exit.
  void service_wait_block_release(unsigned warp_id);
  // [Stage 1] Common cursor-skip used by BOTH the parked-release and the
  // immediate-ready fold paths so their skip semantics can never diverge again.
  // Responsibility is deliberately narrow: flush younger IBuffer entries, build
  // the allowed-skip PC set (region.body_pcs + loopback_bra_pc), advance the
  // trace cursor to region.exit_pc via skip_spin_to_exit (which asserts the
  // post-skip get_pc()==exit_pc and the no-real-work guard), and return the
  // skipped count plus phase-aware cursor/rollback and per-class statistics.
  // It does NOT touch generation bookkeeping, pending-wait state, counters,
  // hold duration, or histograms; those stay owned by each caller.
  unsigned fold_cursor_skip_to_exit(unsigned warp_id,
                                    const SpinWaitRegionRecord &region,
                                    FoldSkipStats &out_stats);
  bool check_if_non_released_reduction_barrier(warp_inst_t &inst);
  bool warp_waiting_at_mem_barrier(unsigned warp_id) override;
  // True when `pI` is a UTMACMDFLUSH (cp.async.bulk.wait_group 0) and the
  // issuing warp still has outstanding store-class TMA transfers. The warp must
  // stall (warp-local drain-all) until all its store/reduce transfers complete.
  bool warp_waiting_at_tma_flush(unsigned warp_id, const warp_inst_t *pI);
  bool warp_waiting_grid_barrier(unsigned warp_id) override;
  // [SR-M0A] Pure, non-mutating wait-state queries used only by the stall-reason
  // alignment snapshot. None of them clear a barrier/membar, insert/erase a
  // wait-set entry, invalidate a cache, update a watchdog map, or emit a log.
  // See .plan/STALL_REASON_ALIGN.md Stage SR-M0A.6.
  //
  // peek_wait_block_hold_mode inspects ONLY the pending WaitBlock state
  // (m_pending_sync_waits) and returns NONE when no unresolved hold exists.
  // It deliberately does NOT consult the programmer/named-barrier set, unlike
  // warp_waiting_at_barrier / warp_held_on_mbarrier_wait.
  WaitBlockPendingMode peek_wait_block_hold_mode(unsigned warp_id) const;
  // Named/programmer barrier hold only (m_barriers), no mbarrier fold state.
  bool peek_named_barrier_wait(unsigned warp_id) const;
  // Scope-aware fence drain condition, observed without clearing the membar,
  // invalidating L1, or touching the stuck-watchdog maps.
  bool peek_membar_wait(unsigned warp_id) const;
  // Grid-barrier flag only.
  bool peek_grid_barrier_wait(unsigned warp_id) const;
  // UTMACMDFLUSH with outstanding store-class TMA transfers, without touching
  // m_warps_waiting_tma_flush or emitting an episode log.
  bool peek_tma_flush_wait(unsigned warp_id, const warp_inst_t *inst) const;
  void clear_gridbar(unsigned int kernel_id);
  void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                   warp_set_t warps);
  void decrement_atomic_count(unsigned wid, unsigned n);
  unsigned int get_atomic_count(unsigned wid);

  void set_max_cta(const kernel_info_t &kernel);
  unsigned int get_n_active_cta() const override;
  unsigned int get_not_completed() const override;
  unsigned int isactive() const override;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const override;

  void issue_block2core(class kernel_info_t &kernel) override;
  bool can_issue_1block(kernel_info_t &kernel) override;
  int find_available_hwtid(unsigned int cta_size, bool occupy);
  bool occupy_shader_resource_1block(kernel_info_t &k, bool occupy);
  void release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t &k);

  void warp_inst_complete(const warp_inst_t &inst) override;
  void dec_inst_in_pipeline(unsigned warp_id) override;
  void store_ack(class mem_fetch *mf) override;
  void inc_store_req(unsigned warp_id) override;
  bool ptx_thread_done(unsigned hw_thread_id) const override;

  // Scope-aware memory-fence store tracking (per warp). These mirror the existing
  // inc_store_req/store_ack accounting but split it by fence visibility level so a
  // MEMBAR can drain only the stores visible at its scope. inc happens when a store
  // is issued (level chosen from is_l1d_bypass / shared), dec when that store is
  // acked (level read back from the mem_fetch tag, or directly for shared stores).
  void inc_fence_store(unsigned warp_id, int fence_vis_level, unsigned n = 1);
  void dec_fence_store(unsigned warp_id, int fence_vis_level, unsigned n = 1);
  // True while warp_id still has stores not yet visible at the given fence scope
  // (MEMBAR_SCOPE_CTA / MEMBAR_SCOPE_GPU). Used by warp_waiting_at_mem_barrier.
  bool warp_has_pending_fence_stores(unsigned warp_id, int membar_scope) const;

  // debug:
  void display_simt_state(FILE *fout, int mask) const;
  void display_pipeline(FILE *fout, int print_mem, int mask3bit) const override;
  void display_SM(FILE *fout, int print_mem, int mask3bit) const;
  void dump_warp_state(FILE *fout) const;

  // Stats
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) override;
  void get_cache_stats(cache_stats &cs) override;
  void get_L0I_sub_stats(struct cache_sub_stats &css) const override;
  void get_L1I_sub_stats(struct cache_sub_stats &css) const override;
  void get_L1D_sub_stats(struct cache_sub_stats &css) const override;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const override;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const override;

  unsigned int inactive_lanes_accesses_sfu(unsigned active_count, double latency);
  unsigned int inactive_lanes_accesses_nonsfu(unsigned active_count,
                                          double latency);
  void mem_instruction_stats(const warp_inst_t &inst) override;
  void incload_stat();
  void incstore_stat();
  void incialu_stat(unsigned active_count, double latency);
  void incimul_stat(unsigned active_count, double latency);
  void incimul24_stat(unsigned active_count, double latency);
  void incimul32_stat(unsigned active_count, double latency);
  void incidiv_stat(unsigned active_count, double latency);
  void incfpalu_stat(unsigned active_count, double latency);
  void incfpmul_stat(unsigned active_count, double latency);
  void incfpdiv_stat(unsigned active_count, double latency);
  void incdpalu_stat(unsigned active_count, double latency);
  void incdpmul_stat(unsigned active_count, double latency);
  void incdpdiv_stat(unsigned active_count, double latency);
  void incsqrt_stat(unsigned active_count, double latency);
  void inclog_stat(unsigned active_count, double latency);
  void incexp_stat(unsigned active_count, double latency);
  void incsin_stat(unsigned active_count, double latency);
  void inctensor_stat(unsigned active_count, double latency);
  void inctex_stat(unsigned active_count, double latency);
  void inc_const_accesses(unsigned active_count);
  void incsfu_stat(unsigned active_count, double latency);
  void incsp_stat(unsigned active_count, double latency);
  void incmem_stat(unsigned active_count, double latency);
  void incregfile_reads(unsigned active_count);
  void incregfile_writes(unsigned active_count);
  void incnon_rf_operands(unsigned active_count);
  void incspactivelanes_stat(unsigned active_count);
  void incsfuactivelanes_stat(unsigned active_count);
  void incfuactivelanes_stat(unsigned active_count);
  void incfumemactivelanes_stat(unsigned active_count);
  void inc_simt_to_mem(unsigned n_flits);
  void incexecstat(warp_inst_t *&inst);
  void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const override;

  address_type from_local_pc_to_global_pc_address(address_type local_pc, unsigned int unique_function_id) override;
  address_type from_global_pc_address_to_local_pc(address_type global_pc, unsigned int unique_function_id) override;
  
  bool can_send_inst_from_subcore_to_sm_shared_pipeline() const;

  void set_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline(unsigned int num_cycles);

  Element_stats m_sm_stats;

  void create_gpu_per_sm_stats(Element_stats &all_stats) override;
  void reset_cycless_access_history() override;
  void gather_gpu_per_sm_stats(Element_stats &all_stats, coalescingStatsAcrossSms& coal_stats_l1d, coalescingStatsAcrossSms& coal_stats_const, coalescingStatsAcrossSms& coal_stats_sharedmem) override;
  void gather_gpu_per_sm_single_stat(Element_stats &all_stats, std::string stat_name) override;

  // [SR-M0A] Fold every resident Subcore's stall-reason-align totals into dst
  // (additive) and reset them. Boundary-only; gated by the caller.
  void gather_stall_reason_align_stats(Subcore::StallPopulationTotals &dst) const;
  void reset_stall_reason_align_stats();

  void increment_sm_stat_by_integer(std::string stat_name, int val_to_increment) override;
  
  bool is_using_interwarp_coal_warps_waiting_dep_counter();
  void enqueue_instruction_region_prewarm(kernel_info_t &kernel);
  void issue_pending_instruction_region_prewarm();
  void handle_sync_instruction(warp_inst_t &inst, unsigned int warp_id);
  void handle_partial_wait_instruction_issue(const warp_inst_t &inst,
                                              unsigned int warp_id);
  void start_partial_wait_hold(unsigned int warp_id,
                               HopperPartialWaitArm &arm);
  void note_partial_replay_progress(unsigned int warp_id,
                                    unsigned int remaining,
                                    uint64_t pc,
                                    unsigned int advanced_occurrences);
  void service_partial_replay_progress(unsigned int warp_id);
  HopperMBarrierKey build_sync_barrier_key(const warp_inst_t &inst,
                                           unsigned int warp_id) const;
  HopperMBarrierObject &get_or_create_sync_barrier(
      const HopperMBarrierKey &key);
  void recompute_sync_barrier_ready_and_maybe_flip_phase(
      const HopperMBarrierKey &key, HopperMBarrierObject &barrier);
  bool is_sync_wait_satisfied(
      const HopperMBarrierPendingWait &pending_wait) const;
  bool is_wait_block_pending_satisfied(
      const HopperMBarrierPendingWait &pending_wait) const;
  // Pin a modeled wait to the current generation when already ready, or to the
  // next generation when not ready. Release audits this absolute generation,
  // not only parity, so an overshoot cannot be mistaken for a valid release.
  uint64_t expected_gen_for_wait(unsigned int warp_id,
                                 const HopperMBarrierKey &key,
                                 uint64_t wait_state_raw) const;
  // Current monotonic generation of a barrier (0 if the object does not exist).
  uint64_t sync_barrier_gen(const HopperMBarrierKey &key) const;
  void clear_sync_barrier_state_for_cta(unsigned int cta_id);
  void open_tma_assignment_contract(const warp_inst_t &inst,
                                    unsigned int warp_id,
                                    const HopperMBarrierKey &key,
                                    uint32_t expected_tx_bytes);
  void record_wait_block_first_late_arrival(
      const warp_inst_t &inst, unsigned int warp_id,
      const HopperMBarrierKey &key, uint64_t target_generation,
      uint64_t current_generation, WaitBlockPendingMode mode);
  void claim_tma_completion_target(const warp_inst_t &inst, TMACommand &cmd);
  void complete_tma_load(TmaCompletionTarget &target, uint32_t memory_bytes,
                         uint64_t transfer_uid);
  void debug_dump_tma_binding_snapshot(const char *reason) const;
  void debug_dump_tma_binding_counters() const;
  void debug_log_sync_event(const std::string &message);
  void debug_log_tma_event(const std::string &message);
  void debug_log_tma_binding_event(const char *event,
                                   const std::string &message);
  void debug_log_wait_block_event(const char *event,
                                  const std::string &message);
  void debug_dump_sync_counters() const;
  void record_wait_block_timing_issue(unsigned int warp_id,
                                      const warp_inst_t *inst);

  InterWarp_Coalescing_Waiting_Dep_Counters *m_interwarp_coal_warps_waiting_dep_counter;

 private:
  unsigned int m_sm_id;
  unsigned int m_tpc_id;
  unsigned int m_num_subcores;
  // Warps currently parked on a UTMACMDFLUSH store-drain wait. Used only to
  // emit a single enter/release log per stall episode (avoids per-cycle spam).
  std::set<unsigned int> m_warps_waiting_tma_flush;

  // Deadlock watchdog for scope-aware memory fences: cycle at which each warp
  // first started waiting at its current MEMBAR/FENCE. Used to emit a periodic
  // "[MEMBARDBG][stuck]" warning if a warp stays blocked far too long (likely a
  // counter that never drains -> deadlock). Cleared on release.
  std::map<unsigned int, unsigned long long> m_membar_wait_start_cycle;
  std::map<unsigned int, unsigned long long> m_membar_last_stuck_warn_cycle;

  unsigned int m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline;

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  const shader_core_config *m_config;
  const memory_config *m_memory_config;
  simt_core_cluster *m_cluster;
  gpgpu_sim *m_gpu;
  shader_core_stats *m_stats;
  std::vector<Subcore*> m_subcores;

  std::vector<register_set_uniptr*> m_EX_WB_sm_shared_units_subcore_latches; 
  std::vector<register_set_uniptr*> m_EX_MEM_reception_latches_per_subcore;
  std::vector<register_set_uniptr*> m_EX_TMA_reception_latches_per_subcore;
  register_set_uniptr m_EX_DP_shared_sm_reception_latch = register_set_uniptr(1,"EX_DP_shared_sm_reception_latch");
  functional_unit *m_shared_dp_unit;

  mem_fetch_interface *m_icnt;
  std::shared_ptr<read_only_cache> m_L1I_L1_half_C_cache;
  mem_fetch_interface *m_icnt_L0s;
  unsigned int m_subcore_req_fetch_L1I_priority;  // MOD. Added L0I
  // In order to be perfect, another icnt from memory_unit_subcore to shared
  // structures of memory of the subcore needs to be places here.
  ldst_unit_sm *m_ldst_unit_shared_of_sm;
  tma_unit_sm *m_tma_unit_shared_of_sm = nullptr;
  std::shared_ptr<shader_core_mem_fetch_allocator> m_mem_fetch_allocator;

  std::vector<shd_warp_t *> m_physical_warp;  // per warp information array

  // Scoreboard to fully track data hazards in order to have retrocompatibilty with modes with no control codes
  std::shared_ptr<Scoreboard> m_scoreboard; // RAW and WAW Hazards
  std::shared_ptr<Scoreboard_reads> m_scoreboard_WAR; // WAR Hazards
  std::stack<Wait_Barrier_Entry_Modifier> m_pending_wait_barrier_decrements;
  std::stack<Wait_Barrier_Entry_Modifier> m_pending_wait_barrier_increments;

  barrier_set_t m_barriers;
  std::map<HopperMBarrierKey, HopperMBarrierObject> m_hopper_mbarriers;
  mutable std::vector<HopperMBarrierPendingWait> m_pending_sync_waits;
  std::vector<HopperPartialWaitArm> m_partial_wait_arms;
  std::vector<HopperPartialReplayProgress> m_partial_replay_progress;
  std::vector<std::set<std::string>> m_partial_joined_regions;
  std::vector<uint32_t> m_tma_open_contracts_per_warp;
  std::vector<uint32_t> m_tma_outstanding_bindings_per_warp;
  std::vector<uint64_t> m_tma_max_completed_transfer_uid_per_warp;
  struct WaitBlockSemanticAnchor {
    bool valid = false;
    uint64_t cycle = 0;
    uint32_t trace_kernel_id = 0;
    uint32_t unique_function_id = 0;
    uint32_t cta_warp_id = 0;
    uint64_t pc = 0;
    uint64_t source_index = 0;
    std::string opcode;
  };
  std::vector<WaitBlockSemanticAnchor>
      m_wait_block_last_semantic_anchor_by_warp;
  bool m_wait_block_first_late_reported_by_cta_slot[MAX_CTA_PER_SHADER] = {
      false};
  uint64_t m_next_tma_contract_id = 1;
  uint64_t m_next_tma_binding_id = 1;
  uint64_t m_tma_binding_launch_uid = 0;
  uint64_t m_tma_contracts_opened = 0;
  uint64_t m_tma_contracts_fully_assigned = 0;
  uint32_t m_tma_max_open_contracts_per_warp = 0;
  uint64_t m_tma_bindings_created = 0;
  uint64_t m_tma_bindings_completed = 0;
  uint64_t m_tma_assigned_credit_bytes = 0;
  uint64_t m_tma_completed_credit_bytes = 0;
  uint64_t m_tma_uid_reorders = 0;
  uint64_t m_tma_cross_barrier_applications = 0;
  uint64_t m_tma_contract_mismatches = 0;
  uint64_t m_tma_stale_generation = 0;
  uint64_t m_tma_launch_or_cta_mismatch = 0;
  uint64_t m_tma_duplicate_completion = 0;
  uint64_t m_tma_orphan_cleanup = 0;
  // Initialized from shader_core_config in SM::SM(); 0 disables logging.
  uint64_t m_sync_debug_print_budget = 0;
  uint64_t m_sync_debug_skip_runtime_budget = 0;
  // Independent budget for TMA-only stderr logs (TMADBG); 0 disables.
  uint64_t m_tma_debug_print_budget = 0;
  // Independent budget for WaitBlock event logs; 0 disables.
  uint64_t m_wait_block_debug_print_budget = 20000000;
  uint64_t m_sync_debug_sync_insts = 0;
  uint64_t m_sync_debug_exch = 0;
  uint64_t m_sync_debug_arrive = 0;
  uint64_t m_sync_debug_arrive_expect_tx = 0;
  uint64_t m_sync_debug_wait_pending = 0;
  uint64_t m_sync_debug_wait_released = 0;
  uint64_t m_sync_debug_phase_flip = 0;
  uint64_t m_sync_debug_tma_completions = 0;
  uint64_t m_sync_debug_missing_runtime = 0;

  // thread contexts
  thread_ctx_t *m_threadState;

  // Used for handing out dynamic warp_ids to new warps.
  // the differnece between a warp_id and a dynamic_warp_id
  // is that the dynamic_warp_id is a running number unique to every warp
  // run on this shader, where the warp_id is the static warp slot.
  unsigned int m_dynamic_warp_id;

  // used for local address mapping with single kernel launch
  unsigned int kernel_max_cta_per_shader;
  unsigned int kernel_padded_threads_per_cta;

  // CTA scheduling / hardware thread allocation
  unsigned int m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                                // currently running on this shader.
  unsigned int m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  unsigned int m_local_to_global_cta_id[MAX_CTA_PER_SHADER];  // Map local CTA slot to global CTA ID
  unsigned int m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  std::queue<std::pair<new_addr_type, unsigned int>>
      m_pending_instruction_region_prewarm;
  std::set<std::pair<unsigned int, new_addr_type>>
      m_activated_instruction_region_prewarm;
  unsigned int m_next_instruction_region_prewarm_subcore = 0;
  
  // SM occupancy structures
  int m_active_warps;
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;
  unsigned int m_occupied_n_threads;
  unsigned int m_occupied_shmem;
  unsigned int m_occupied_regs;
  unsigned int m_occupied_ctas;
  std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
  std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;

  // Power
  PowerscalingCoefficients *m_scaling_coeffs;

  // [CTA-imbalance diag] Per-CTA-slot raw count of TENSOR_CORE (WGMMA) ops.
  // Indexed by hardware CTA slot (get_cta_id, 0..MAX_CTA_PER_SHADER-1) so a SM
  // that runs multiple CTAs sequentially (bwd: Waves/SM=2.91) counts each CTA
  // separately — a SM-wide scalar would wrongly fold earlier CTAs into later
  // ones. Also stamps each slot's launch cycle so both tensor density AND
  // elapsed cycles can be correlated with the finish cycle. Observe-only;
  // increment is always-on, emission is gated by -wgmma_step0_instrument_enable.
  unsigned long long m_tensor_ops_by_cta_slot[MAX_CTA_PER_SHADER] = {0};
  unsigned long long m_cta_slot_start_cycle[MAX_CTA_PER_SHADER] = {0};
  // SM-idle cycles (no subcore issued) with >=1 subcore blocked only by the tensor
  // re-issue lockout, attributed to the CTA slot(s) resident this cycle. For 1-CTA/SM
  // (fwd) this exactly attributes the tensor-caused drain to that CTA; for multi-CTA
  // SMs it credits every resident slot (read as an upper bound there). Lets [CTAFIN]
  // correlate a CTA's tensor-caused idle with its tensor density in ONE line.
  unsigned long long m_sm_idle_tensor_cyc_by_cta_slot[MAX_CTA_PER_SHADER] = {0};
  // Cycles with >=1 subcore whose head WGMMA was blocked because the tensor FU is busy
  // (NCU `mma`, the producer/FU-busy tensor stall), attributed to each resident CTA slot.
  // Counted every cycle (not only SM-idle), since a WGMMA can wait on the tensor FU while
  // another subcore still issues. In this trace-driven model there is no separate consumer-
  // side warpgroup_arrive (WGMMA result waits fold into wait_barrier), so this is THE tensor-
  // attributable per-CTA stall. Exact for 1-CTA/SM (fwd); upper bound for bwd.
  unsigned long long m_fu_occupied_tensor_cyc_by_cta_slot[MAX_CTA_PER_SHADER] = {0};
  // [CTA stall breakdown] Per-CTA-slot NON-tensor drain-idle counters, for correlating
  // fwd CTA finish-cycle variance with non-tensor causes. Gated by
  // -cta_stall_breakdown_instrument_enable. `m_sm_idle_cyc_by_cta_slot` counts every
  // true SM-idle cycle (no subcore issued) resident to this CTA = the drain-idle that
  // dominates fwd (1.39x). `m_sm_idle_ibuffer_empty_cyc_by_cta_slot` is the subset where
  // the idle head warp had an empty ibuffer (trace-drained), the suspected fwd cause.
  unsigned long long m_sm_idle_cyc_by_cta_slot[MAX_CTA_PER_SHADER] = {0};
  unsigned long long m_sm_idle_ibuffer_empty_cyc_by_cta_slot[MAX_CTA_PER_SHADER] = {0};
  // [FWD drain-idle 축1] per-CTA mutually-exclusive partition signals: recoverable mbarrier-only
  // idle vs floor (all-drained) idle. Gated by wgmma_step0_instrument_enable; reset at launch/exit.
  unsigned long long m_wait_barrier_only_cyc_by_cta_slot[MAX_CTA_PER_SHADER] = {0};
  unsigned long long m_drained_cyc_by_cta_slot[MAX_CTA_PER_SHADER] = {0};
  // [FWD drain-idle 축2] producer/consumer role split. sticky per-warp booleans (set on first
  // TMA/arrive_expect_tx = producer, first WGMMA = consumer), and per-warp last trace-drain cycle
  // (0 = not yet drained). At CTA exit the [CTAFIN] line derives per-role max drain from these.
  bool m_warp_is_producer[MAX_WARP_PER_SHADER] = {false};
  bool m_warp_is_consumer[MAX_WARP_PER_SHADER] = {false};
  unsigned long long m_warp_drain_cycle[MAX_WARP_PER_SHADER] = {0};
  // [FWD drain-idle 축4] per-CTA mbarrier test counts (per-SM SYNCDBG re-attributed per CTA slot).
  unsigned long long m_wait_pending_by_cta_slot[MAX_CTA_PER_SHADER] = {0};
  unsigned long long m_wait_released_by_cta_slot[MAX_CTA_PER_SHADER] = {0};
  // [FWD drain-idle 축3] mbarrier wait-duration histogram (observe-only; does NOT touch
  // m_pending_sync_waits, which feeds the issue-path warp_waiting_at_barrier). first-miss cycle +
  // barrier addr per warp; on the pass (next hit) we bucket (now - first_miss). Gated by
  // sync_wait_hist_instrument_enable. Buckets: {0, 1-16, 17-64, 65-256, 257-1024, 1024+}.
  address_type m_sync_first_miss_barrier[MAX_WARP_PER_SHADER] = {0};
  unsigned long long m_sync_first_miss_cycle[MAX_WARP_PER_SHADER] = {0};
  unsigned long long m_sync_wait_cycle_hist[6] = {0};
  // [Producer-tail probe] CTA-cycle and producer-issue decomposition after all
  // consumer-role warps have drained. PHASECHK ready/not-ready is sampled from
  // the existing Hopper mbarrier state machine at execution; no state is changed.
  struct ProducerTailCounters {
    unsigned long long start_cycle = 0;
    unsigned long long cycles = 0;
    unsigned long long issued = 0;
    unsigned long long issued_phasechk = 0;
    unsigned long long issued_nanosleep = 0;
    unsigned long long issued_branch = 0;
    unsigned long long issued_tma = 0;
    unsigned long long issued_other = 0;
    unsigned long long phasechk_ready = 0;
    unsigned long long phasechk_not_ready = 0;
    unsigned long long last_producer_issue_cycle = 0;
    unsigned long long stall_cycles = 0;
    unsigned long long stall_not_selected = 0;
    unsigned long long stall_next_stage = 0;
    unsigned long long stall_issue_port = 0;
    unsigned long long stall_no_valid = 0;
    unsigned long long stall_wait_barrier = 0;
    unsigned long long stall_stall_count = 0;
    unsigned long long stall_fu_occupied = 0;
    unsigned long long stall_inst_barrier = 0;
    unsigned long long stall_tma_flush = 0;
    unsigned long long stall_scoreboard = 0;
    unsigned long long stall_l1c = 0;
    unsigned long long stall_result_queue = 0;
    unsigned long long stall_yield = 0;
    unsigned long long stall_unattributed = 0;
  };
  ProducerTailCounters m_producer_tail_by_cta_slot[MAX_CTA_PER_SHADER];

  // [Producer-spin ready-retry probe] Whole-producer-lifetime spin accounting,
  // NOT gated by consumer drain. Sizes captured excess polling for fwd AND bwd
  // (bwd spin is spread across the main loop, so the tail-only probe misses it).
  // All fields are read-only samples of the existing Hopper mbarrier result.
  struct ProducerSpinCounters {
    unsigned long long phasechk_total = 0;       // all PHASECHK/TRYWAIT by producers
    unsigned long long phasechk_ready = 0;       // mbarrier already satisfied at test
    unsigned long long phasechk_not_ready = 0;   // mbarrier not yet satisfied
    unsigned long long issued_phasechk = 0;      // PHASECHK/TRYWAIT issued
    unsigned long long issued_nanosleep = 0;     // NANOSLEEP issued
    unsigned long long issued_branch = 0;        // BRA issued
    unsigned long long issued_other = 0;         // any other producer op
    // Consecutive-ready run tracking: a maximal run of back-to-back ready
    // PHASECHK on one warp with no intervening not-ready is "redundant re-poll".
    unsigned long long ready_run_retries = 0;    // ready tests beyond the 1st in a run
  };
  ProducerSpinCounters m_producer_spin_by_cta_slot[MAX_CTA_PER_SHADER];
  // Per-warp last-PHASECHK-was-ready flag, for consecutive-ready run detection.
  bool m_producer_spin_prev_ready[MAX_WARP_PER_SHADER] = {false};
  bool m_producer_spin_in_ready_run[MAX_WARP_PER_SHADER] = {false};

  // [Spin-wait observe] Observe-only measurement of the would-be park time if the
  // producer spin-wait fold feature were enabled. For each (warp,region)
  // episode we remember the cycle of the FIRST not-ready poll and, when the same
  // warp next polls the same region ready, bucket the elapsed cycles PER REGION
  // (keyed by primary_wait_pc). This never changes control flow; it only sizes
  // park latency, attributes it to a specific spin region, and flags never-flip
  // barriers via a deadlock watchdog. Gated by -spin_wait_observe_enable.
  struct SpinWaitObserveState {
    uint64_t active_wait_pc = 0;        // primary_wait_pc of the in-progress episode
    uint64_t active_barrier_addr = 0;   // barrier being polled (for watchdog dump)
    unsigned active_cta = 0;            // cta owning the episode (for watchdog dump)
    unsigned long long first_not_ready_cycle = 0;  // cycle of first not-ready poll
    bool episode_open = false;          // a not-ready episode is in progress
    bool watchdog_fired = false;        // deadlock watchdog already reported this episode
  };
  SpinWaitObserveState m_spin_wait_observe_by_warp[MAX_WARP_PER_SHADER];
  // Per-region (keyed by primary_wait_pc) would-be-park accounting.
  struct SpinWaitRegionObserve {
    uint64_t primary_wait_pc = 0;
    unsigned long long episodes = 0;              // completed not-ready->ready episodes
    unsigned long long park_cycles_total = 0;     // summed would-be park cycles
    unsigned long long park_cycles_max = 0;       // longest single would-be park
    unsigned long long immediate_ready_polls = 0; // ready with no open episode
    unsigned long long watchdog_hits = 0;         // episodes that crossed the deadlock threshold
    unsigned long long park_hist[6] = {0, 0, 0, 0, 0, 0};  // 0,1-16,17-64,65-256,257-1024,>1024
  };
  std::map<uint64_t, SpinWaitRegionObserve> m_spin_wait_observe_by_region;
  struct FoldPaceStats {
    unsigned long long parks = 0;
    unsigned long long releases = 0;
    unsigned long long park_cycles_total = 0;
    unsigned long long park_cycles_max = 0;
    unsigned long long first_park_cycle = 0;
    unsigned long long last_end_cycle = 0;
    uint64_t last_expected_gen = 0;
    uint64_t last_barrier_gen = 0;
    // [Stage 1.1] Immediate-ready folds are NOT parks/releases: the primary poll
    // was already satisfied at issue, so no warp ever parked. Counted separately
    // so parks == releases stays a real park-balance invariant.
    unsigned long long immediate_folds = 0;
    unsigned long long immediate_skipped_insts = 0;
  };
  // Per-(warp, barrier, wait_pc) pacing decomposition. wait_pc is included
  // because multiple logical waits may share one runtime barrier.
  using FoldPaceKey = std::pair<HopperMBarrierKey, uint64_t>;
  std::vector<std::map<FoldPaceKey, FoldPaceStats>> m_fold_pace_by_warp;
  // Captured-poll validation counters (only move when -wait_block_enable).
  // At a clean diagnostic exit, holds == releases. Generation overshoot aborts
  // rather than creating a late-release path.
  unsigned long long m_fold_parks_total = 0;
  unsigned long long m_fold_releases_total = 0;
  unsigned long long m_fold_skipped_insts_total = 0;
  unsigned long long m_fold_park_cycles_total = 0;
  unsigned long long m_fold_park_cycles_max = 0;
  unsigned long long m_fold_watchdog_hits = 0;
  // [Stage 1.1] Immediate-ready fold totals (ready-at-issue path, no park).
  // total_folds = releases + immediate_folds; immediate_skipped_insts is the
  // subset of skipped_insts contributed by the immediate path.
  unsigned long long m_fold_immediate_folds_total = 0;
  unsigned long long m_fold_immediate_skipped_insts_total = 0;
  unsigned long long m_fold_aux_holds_total = 0;
  unsigned long long m_fold_aux_releases_total = 0;
  unsigned long long m_missing_wait_ready_total = 0;
  unsigned long long m_missing_wait_holds_total = 0;
  unsigned long long m_missing_wait_releases_total = 0;
  unsigned long long m_partial_missing_wait_misses_total = 0;
  unsigned long long m_partial_capsules_total = 0;
  unsigned long long m_partial_join_already_issued_total = 0;
  unsigned long long m_partial_deferred_occurrences_total = 0;
  unsigned long long m_partial_replayed_occurrences_total = 0;
  unsigned long long m_partial_folded_occurrences_total = 0;
  unsigned long long m_partial_join_holds_total = 0;
  unsigned long long m_partial_join_releases_total = 0;
  unsigned long long m_partial_replay_watchdog_aborts_total = 0;
  unsigned long long m_wait_block_generation_overshoots_total = 0;
  unsigned long long m_wait_block_exact_targets_total = 0;
  unsigned long long m_wait_block_generation_fallbacks_total = 0;
  unsigned long long m_wait_block_late_arrivals_total = 0;
  unsigned long long m_wait_block_first_late_reports_total = 0;
  unsigned long long m_wait_block_watchdog_aborts_total = 0;
 public:
  // Increment the tensor-op count for the CTA slot that owns hardware warp wid.
  void inc_tensor_ops_for_warp(unsigned wid) {
    unsigned cta_slot = m_physical_warp[wid]->get_cta_id();
    if (cta_slot < MAX_CTA_PER_SHADER) m_tensor_ops_by_cta_slot[cta_slot]++;
  }
  // [FWD drain-idle 축2] sticky role marks (observe-only; set on first occurrence).
  void mark_producer_warp(unsigned wid) { if (wid < MAX_WARP_PER_SHADER) m_warp_is_producer[wid] = true; }
  void mark_consumer_warp(unsigned wid) { if (wid < MAX_WARP_PER_SHADER) m_warp_is_consumer[wid] = true; }
  // [Producer-tail probe] read-only hooks used by Subcore::issue() and the
  // existing mbarrier execution path.
  bool producer_tail_active_for_warp(unsigned wid) const;
  void record_producer_tail_issue(unsigned wid, const warp_inst_t *inst);
  void record_producer_tail_phasechk_result(unsigned wid, bool ready);
  void account_producer_tail_cycle();
  // [Producer-spin ready-retry probe] whole-lifetime hooks (not tail-gated).
  void record_producer_spin_issue(unsigned wid, const warp_inst_t *inst);
  void record_producer_spin_phasechk_result(unsigned wid, bool ready);
  // [Spin-wait observe] observe-only would-be-park measurement (never blocks).
  void record_spin_wait_observe(unsigned wid, uint64_t wait_pc,
                                uint64_t barrier_addr, unsigned cta_id,
                                bool ready);
  void print_spin_wait_observe_summary() const;
  void print_wait_block_summary() const;
};
