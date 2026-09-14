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

// Copyright (c) 2018-2021, Mahmoud Khairy, Vijay Kandiah, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Northwestern University, Purdue University,
//    The University of British Columbia nor the names of their contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <set>

#ifndef TRACE_DRIVEN_H
#define TRACE_DRIVEN_H

#include "../ISA_Def/trace_opcode.h"
#include "../trace-parser/trace_parser.h"
#include "lane_group_replay.h"
#include "../gpgpu-sim/src/abstract_hardware_model.h"
#include "../gpgpu-sim/src/gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/src/gpgpu-sim/shader.h"
#include "../gpgpu-sim/src/cuda-sim/ptx_ir.h"


void advance_trace_cta_id(kernel_trace_t *kernel_trace_info);

class trace_function_info : public function_info {
 public:
  trace_function_info(const struct gpgpu_ptx_sim_info &info,
                      gpgpu_context *m_gpgpu_context)
      : function_info(0, m_gpgpu_context) {
    m_kernel_info = info;
  }

  virtual const struct gpgpu_ptx_sim_info *get_kernel_info() const {
    return &m_kernel_info;
  }

  virtual const void set_kernel_info(const struct gpgpu_ptx_sim_info &info) {
    m_kernel_info = info;
  }

  ~trace_function_info() override {}
};

enum class WaitBlockTraceOrigin {
  CANONICAL,
  PARTIAL_SIBLING_JOIN,
  PARTIAL_BLOCKED_REPLAY,
  LANE_GROUP_REPLAY,
  SYNTHETIC,
};

class trace_warp_inst_t : public warp_inst_t {
 public:
  trace_warp_inst_t() {
    m_opcode = 0;
    should_do_atomic = false;
  }

  trace_warp_inst_t(const class core_config *config) : warp_inst_t(config) {
    m_opcode = 0;
    should_do_atomic = false;
  }

  bool parse_from_trace_struct(
      inst_trace_t &trace,
      const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
      const class trace_config *tconfig,
      const class kernel_trace_t *kernel_trace_info,
      traced_execution &static_trace_info);
      
  bool is_s2r() { return m_opcode == OP_S2R; }
  WaitBlockTraceOrigin wait_block_trace_origin =
      WaitBlockTraceOrigin::CANONICAL;
  // [WaitBlock partial replay] When this instruction was decoded from a
  // reserved replay occurrence, the index of that occurrence in the warp's
  // replay stream. Used by commit_trace_inst to enforce exact-once, in-order
  // (FIFO) issue. Meaningful only when wait_block_trace_origin is a replay
  // origin (not CANONICAL or SYNTHETIC).
  unsigned int replay_occurrence_index = 0;

 private:
  unsigned m_opcode;
};

struct PartialTraceExtractStats {
  unsigned int cursor_before = 0;
  unsigned int cursor_after = 0;
  unsigned int blocked_occurrences = 0;
  unsigned int sibling_join_occurrences = 0;
};

class trace_kernel_info_t : public kernel_info_t {
 public:
  trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                      trace_function_info *m_function_info,
                      trace_parser *parser, class trace_config *config,
                      kernel_trace_t *kernel_trace_info);

  void get_next_threadblock_traces(
      std::vector<std::map<address_type, traced_instructions_by_pc> *> threadblock_traces, std::vector<std::vector<address_type> *> threadblock_traced_pcs, gpgpu_sim *gpu, traced_execution &static_trace_info);

  unsigned long get_cuda_stream_id() {
    return m_kernel_trace_info->cuda_stream_id;
  }

  unsigned get_trace_kernel_id() const override {
    return m_kernel_trace_info ? m_kernel_trace_info->kernel_id : 0;
  }

  kernel_trace_t *get_trace_info() { return m_kernel_trace_info; }

  bool was_launched() { return m_was_launched; }

  void set_launched() { m_was_launched = true; }

  trace_config *m_tconfig;
  const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
  kernel_trace_t *m_kernel_trace_info;
 private:
  trace_parser *m_parser;
  bool m_was_launched;

  friend class trace_shd_warp_t;
};

class trace_config {
 public:
  trace_config();

  void set_latency(unsigned category, unsigned &latency,
                   unsigned &initiation_interval) const;
  // [NANOSLEEP spin lever] opcode-aware overload: identical to the category-only version except that
  // when category==MISCELLANEOUS_NO_QUEUE_OP and opcode==OP_NANOSLEEP it applies the dedicated
  // nanosleep latency/init (so widening NANOSLEEP does NOT perturb other MISC_NO_QUEUE ops).
  void set_latency(unsigned category, unsigned opcode, unsigned &latency,
                   unsigned &initiation_interval) const;
  void parse_config();
  void reg_options(option_parser_t opp);
  char *get_traces_filename() { return g_traces_filename; }

  bool get_is_extra_traces_enabled(){ return is_extra_traces_enabled; } // MOD. Improved tracer

  unsigned int get_int_latency() const { return int_latency; }
  unsigned int get_fp_latency() const { return fp_latency; }
  unsigned int get_dp_latency() const { return dp_latency; }
  unsigned int get_sfu_latency() const { return sfu_latency; }
  unsigned int get_tensor_latency() const { return tensor_latency; }
  unsigned int get_int_init() const { return int_init; }
  unsigned int get_fp_init() const { return fp_init; }
  unsigned int get_dp_init() const { return dp_init; }
  unsigned int get_sfu_init() const { return sfu_init; }
  unsigned int get_tensor_init() const { return tensor_init; }
  unsigned int get_branch_latency() const { return branch_latency; }
  unsigned int get_branch_init() const { return branch_init; }
  unsigned int get_half_latency() const { return half_latency; }
  unsigned int get_half_init() const { return half_init; }
  unsigned int get_uniform_latency() const { return uniform_latency; }
  unsigned int get_uniform_init() const { return uniform_init; }
  unsigned int get_predicate_latency() const { return predicate_latency; }
  unsigned int get_predicate_init() const { return predicate_init; }
  unsigned int get_miscellaneous_queue_latency() const {
    return miscellaneous_queue_latency;
  }
  unsigned int get_miscellaneous_queue_init() const {
    return miscellaneous_queue_init;
  }
  unsigned int get_miscellaneous_no_queue_latency() const {
    return miscellaneous_no_queue_latency;
  }
  unsigned int get_miscellaneous_no_queue_init() const {
    return miscellaneous_no_queue_init;
  }
  // [NANOSLEEP spin lever] dedicated nanosleep latency/init (default 1,1 = old behavior).
  unsigned int get_nanosleep_latency() const { return nanosleep_latency; }
  unsigned int get_nanosleep_init() const { return nanosleep_init; }

 private:
  unsigned int_latency, fp_latency, dp_latency, sfu_latency, tensor_latency;
  unsigned int_init, fp_init, dp_init, sfu_init, tensor_init;
  unsigned int branch_latency, branch_init, half_latency, half_init,
      uniform_latency, uniform_init, predicate_latency, predicate_init,
      miscellaneous_queue_latency, miscellaneous_queue_init,
      miscellaneous_no_queue_latency,
      miscellaneous_no_queue_init;
  // [NANOSLEEP spin lever] dedicated nanosleep timing (default 1,1).
  unsigned int nanosleep_latency, nanosleep_init;
  unsigned specialized_unit_latency[SPECIALIZED_UNIT_NUM];
  unsigned specialized_unit_initiation[SPECIALIZED_UNIT_NUM];

  char *g_traces_filename;
  char *trace_opcode_latency_initiation_int;
  char *trace_opcode_latency_initiation_sp;
  char *trace_opcode_latency_initiation_dp;
  char *trace_opcode_latency_initiation_sfu;
  char *trace_opcode_latency_initiation_tensor;
  char *trace_opcode_latency_initiation_branch;
  char *trace_opcode_latency_initiation_half;
  char *trace_opcode_latency_initiation_uniform;
  char *trace_opcode_latency_initiation_predicate;
  char *trace_opcode_latency_initiation_miscellaneous_queue;
  char *trace_opcode_latency_initiation_miscellaneous_no_queue;
  // [NANOSLEEP spin lever] dedicated nanosleep opt string (default "1,1").
  char *trace_opcode_latency_initiation_nanosleep;
  char *trace_opcode_latency_initiation_specialized_op[SPECIALIZED_UNIT_NUM];

  bool is_extra_traces_enabled; // MOD. Improved tracer

};

class trace_shd_warp_t : public shd_warp_t {
 public:
  trace_shd_warp_t(class shader_core_ctx_wrapper *shader, unsigned warp_size, shader_core_stats *stats)
      : shd_warp_t(shader, warp_size, stats) {
    m_kernel_info = NULL;
    used_insts = 0;
  }

  ~trace_shd_warp_t() {}

  std::map<address_type, traced_instructions_by_pc> map_warp_traces;
  std::vector<address_type> traced_pcs;
  unsigned int used_insts;
  trace_warp_inst_t *get_next_trace_inst(address_type pc); // MOD. VPREG
  // [WaitBlock partial replay] Reservation-indexed decode. When the IBuffer
  // slot was allocated against a replay occurrence, decode the exact reserved
  // occurrence (is_replay=true, replay_index, replay_epoch) instead of the
  // global decode head. This keeps slot<->occurrence identity stable under
  // out-of-order instruction-cache fills. A canonical slot (is_replay=false)
  // falls back to the ordinary pc-keyed path.
  trace_warp_inst_t *get_next_trace_inst(address_type pc, bool is_replay,
                                         unsigned int replay_index,
                                         uint64_t replay_epoch);
  void decrement_trace_pc(); // MOD. VPREG
  void decrease_num_used_inst(address_type pc);
  void rollback_trace_inst(const trace_warp_inst_t *inst);
  void commit_trace_inst(const trace_warp_inst_t *inst);
  bool extract_partial_wait_capsule(
      uint64_t blocked_mask, uint64_t sibling_mask, uint64_t live_mask,
      unsigned int unique_function_id,
      address_type captured_ready_pc, address_type join_pc,
      address_type post_join_pc, PartialTraceExtractStats *out_stats,
      std::string *failure_reason);
  bool activate_partial_blocked_replay();
  unsigned int skip_partial_replay_to_exit(
      address_type exit_pc, address_type loopback_bra_pc,
      const std::set<address_type> &allowed_skip_pcs,
      struct FoldSkipStats *out_stats);
  bool partial_replay_active() const;
  bool partial_blocked_replay_pending() const;
  bool partial_capsule_active() const;
  WaitBlockTraceOrigin partial_replay_origin() const;
  unsigned int partial_replay_remaining() const;
  unsigned int partial_deferred_size() const;
  address_type canonical_trace_pc() const;
  // [WaitBlock partial replay] Fetch-time reservation API. When a replay stream
  // is active and has unreserved occurrences, reserve the next one and return
  // its PC, occurrence index, and stream epoch through the out-params so the
  // IBuffer slot can bind to it. Returns false when no replay reservation is
  // available (caller then uses the canonical sequential fetch PC).
  bool reserve_next_partial_replay(address_type *pc_out,
                                   unsigned int *index_out,
                                   uint64_t *epoch_out);
  // [WaitBlock partial replay] Undo exactly one (the most recent) replay
  // reservation. Called per-removed-entry by the IBuffer flush and once by
  // remove_entry after a RESERVATION_FAIL, so an un-fetched occurrence is
  // re-reserved on the next fetch. The in-flight (already pop_front'd but not
  // yet committed) occurrence is never un-reserved.
  void unreserve_last_partial_replay();
  uint64_t partial_replay_epoch() const;
  void clear_partial_wait_capsule();
  LaneGroupReplay<inst_trace_t> &lane_groups() { return m_lane_groups; }
  const LaneGroupReplay<inst_trace_t> &lane_groups() const { return m_lane_groups; }
  const LaneGroupSpanRecord &lane_group_span() const { return m_lane_group_span; }
  void begin_lane_group_replay(const LaneGroupSpanRecord &span, unsigned fid);
  unsigned fold_lane_group_poll(unsigned group,
                               const SpinWaitRegionRecord &region,
                               FoldSkipStats *out_stats);
  bool lane_group_can_fetch() const;
  // [Spin-wait fold] Forward analogue of get_next_trace_inst's cursor
  // bookkeeping: advance both used_insts and each skipped pc's per-occurrence
  // counter until traced_pcs[used_insts] == exit_pc, WITHOUT issuing the skipped
  // instructions. Lands the cursor AT exit_pc (so the exit BRA issues normally).
  // Returns the number of skipped instructions. Only used by the fold path;
  // see .plan/SPIN_WAIT_SIM_MODELING.md 4b.
  //
  // [Stage 1.4] Semantic guard: allowed_skip_pcs is body_pcs + loopback_bra_pc.
  // Before advancing over any occurrence the guard asserts (a) the current PC is
  // in allowed_skip_pcs, (b) the PC has a live trace occurrence, and (c) the
  // occurrence's opcode/sync class carries no semantic/local-memory/HGMMA/MUFU/
  // TMA/BAR work. Any violation emits [FOLDSKIP_AUDIT] and aborts. The per-class
  // breakdown of what was skipped is returned in *out_stats when non-null.
  unsigned int skip_spin_to_exit(address_type exit_pc,
                                 address_type loopback_bra_pc,
                                 const std::set<address_type> &allowed_skip_pcs,
                                 struct FoldSkipStats *out_stats);
  void clear();
  bool trace_done();
  address_type get_start_trace_pc();
  virtual address_type get_pc();
  virtual kernel_info_t *get_kernel_info() const { return m_kernel_info; }
  void set_kernel(trace_kernel_info_t *kernel_info) {
    m_kernel_info = kernel_info;
  }

 private:
  LaneGroupReplay<inst_trace_t> m_lane_groups;
  LaneGroupSpanRecord m_lane_group_span;
  std::vector<inst_trace_t> m_partial_replay;
  std::vector<inst_trace_t> m_partial_deferred;
  unsigned int m_partial_replay_issued = 0;
  // [WaitBlock partial replay] Fetch-time reservation cursor and stream epoch.
  // m_partial_replay_reserved is the number of replay occurrences already bound
  // to an IBuffer slot; it satisfies
  //   0 <= issued <= reserved <= size.
  // Decode does not use a separate decoded cursor: each IBuffer slot carries the
  // exact reserved occurrence index, so decode is reservation-indexed and safe
  // under out-of-order instruction-cache fills.
  // m_partial_replay_epoch changes whenever a replay stream is activated,
  // drained, or replaced, so a stale instruction-cache response carrying an
  // older epoch cannot fill a slot in a newer stream.
  unsigned int m_partial_replay_reserved = 0;
  uint64_t m_partial_replay_epoch = 0;
  WaitBlockTraceOrigin m_partial_replay_origin =
      WaitBlockTraceOrigin::CANONICAL;
  trace_kernel_info_t *m_kernel_info;
};

class trace_gpgpu_sim : public gpgpu_sim {
 public:
  trace_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  virtual void createSIMTCluster();
};

class trace_simt_core_cluster : public simt_core_cluster {
 public:
  trace_simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                          const shader_core_config *config,
                          const memory_config *mem_config,
                          class shader_core_stats *stats,
                          class memory_stats_t *mstats)
      : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {
    create_shader_core_ctx();
  }

  virtual void create_shader_core_ctx();
};

class trace_shader_core_ctx : public shader_core_ctx {
 public:
  trace_shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                        unsigned shader_id, unsigned tpc_id,
                        const shader_core_config *config,
                        const memory_config *mem_config,
                        shader_core_stats *stats)
      : shader_core_ctx(gpu, cluster, shader_id, tpc_id, config, mem_config,
                        stats) {
    create_front_pipeline();
    create_shd_warp();
    create_schedulers();
    create_exec_pipeline();
  }

  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t &kernel);
  virtual void func_exec_inst(warp_inst_t &inst);
  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu);
  virtual void create_shd_warp();
  virtual warp_inst_t *get_next_inst(unsigned warp_id, address_type pc); // MOD. VPREG
  virtual void decrement_trace_pc(unsigned warp_id); // MOD. VPREG
  virtual void updateSIMTStack(unsigned warpId, warp_inst_t *inst, ib_ooo_simt_info *ib_ooo_simt_status);
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc);
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI);
  virtual void issue_warp(register_set &warp, const warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned sch_id);
  virtual RRS* get_loog_rrs() override { return nullptr; }
  virtual bool get_is_loog_enabled() override { return m_config->is_loog_enabled; }

 private:
  void init_traces(unsigned start_warp, unsigned end_warp,
                   kernel_info_t &kernel);
};

types_of_operands get_oprnd_type(op_type op, special_ops sp_op);

#endif
