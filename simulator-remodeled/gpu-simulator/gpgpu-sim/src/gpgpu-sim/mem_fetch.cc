// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
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
// The University of British Columbia nor the names of its contributors may be
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

#include "mem_fetch.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"

unsigned mem_fetch::sm_next_mf_request_uid = 1;

// Opt6 4.11.2 latency-bucket instrumentation (observe-only, timing-neutral).
unsigned long long mem_fetch::s_tma_status_cycles[NUM_MEM_REQ_STAT] = {0};
unsigned long long mem_fetch::s_tma_status_visits[NUM_MEM_REQ_STAT] = {0};

mem_fetch::mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
                     unsigned ctrl_size, unsigned wid, unsigned sid,
                     unsigned tpc, const memory_config *config,
                     unsigned long long cycle, mem_fetch *m_original_mf,
                     mem_fetch *m_original_wr_mf, unsigned int unique_function_id)
    : m_access(access)

{
  m_request_uid = sm_next_mf_request_uid++;
  // std::cerr << "Creating mem_fetch: " << m_request_uid << std::endl; fflush(stdout);
  m_access = access;
  if (inst) {
    m_inst = *inst;
    assert(wid == m_inst.warp_id());
  }
  m_data_size = access.get_size();
  m_ctrl_size = ctrl_size;
  m_sid = sid;
  m_tpc = tpc;
  m_wid = wid;
  config->m_address_mapping.addrdec_tlx(access.get_addr(), &m_raw_addr);
  m_partition_addr =
      config->m_address_mapping.partition_address(access.get_addr());
  m_type = m_access.is_write() ? WRITE_REQUEST : READ_REQUEST;
  m_timestamp = cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = cycle;
  m_mem_config = config;
  icnt_flit_size = config->icnt_flit_size;
  original_mf = m_original_mf;
  original_wr_mf = m_original_wr_mf;
  if (m_original_mf) {
    m_raw_addr.chip = m_original_mf->get_tlx_addr().chip;
    m_raw_addr.sub_partition = m_original_mf->get_tlx_addr().sub_partition;
    // Inherit the TMA tag so that L2 sector-split children are still routed
    // back to the TMA unit (not the ldst unit) on response. Ordinary requests
    // keep m_is_tma=false because their parents are not TMA either.
    m_is_tma = m_original_mf->is_tma();
  }
  m_subcore = -1; // MOD. Added L0I
  m_is_filling_L0 = false; // MOD. Added L0I
  m_is_fixed_latency_constant_access= false;
  m_unique_function_id = unique_function_id;
  m_is_prefetch = false;
  m_is_instruction_region_prewarm = false;
  m_stream_buffer_id = std::numeric_limits<unsigned int>::max();
  m_prefetch_l1i_fate = 0;
  m_kernel_id = 0;

  m_tlb_set_idx = -1;
  m_tlb_way_idx = -1;
  m_tlb_tag = 0;
}

mem_fetch::~mem_fetch() {
  // std::cerr << "Destroying mem_fetch: " << m_request_uid << std::endl; fflush(stdout);
  m_status = MEM_FETCH_DELETED; 
}

#define MF_TUP_BEGIN(X) static const char *Status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) \
  }                   \
  ;
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

void mem_fetch::print(FILE *fp, bool print_inst) const {
  fprintf(fp, "  mf: uid=%6u, sid%02u:w%02u, part=%u, ", m_request_uid, m_sid,
          m_wid, m_raw_addr.chip);
  m_access.print(fp);
  if ((unsigned)m_status < NUM_MEM_REQ_STAT)
    fprintf(fp, " status = %s (%llu), ", Status_str[m_status], m_status_change);
  else
    fprintf(fp, " status = %u??? (%llu), ", m_status, m_status_change);
  if (!m_inst.empty() && print_inst)
    m_inst.print(fp);
  else
    fprintf(fp, "\n");
}

void mem_fetch::set_status(enum mem_fetch_status status,
                           unsigned long long cycle) {
  // Opt6 4.11.2: attribute the time just spent in the PREVIOUS status to a
  // TMA-only per-stage bucket before overwriting it. Observe-only: it reads the
  // existing m_status / m_status_change and writes only to static counters, so
  // it cannot change any simulated timing. Guarded to TMA mfs and to valid,
  // monotonic transitions so parent/child sector splits or re-probes cannot
  // corrupt the totals.
  if (m_is_tma && (unsigned)m_status < NUM_MEM_REQ_STAT &&
      cycle >= m_status_change) {
    s_tma_status_cycles[m_status] += (cycle - m_status_change);
    ++s_tma_status_visits[m_status];
  }
  m_status = status;
  m_status_change = cycle;
}

// Opt6 4.11.2: dump the TMA-only per-stage residency table plus the three
// summary buckets used in the plan (req_side / reply_side / queue_wait). Called
// once from gpu_print_stat(). Bucket assignment follows the 4.10 stage map:
//   req_side  = SM-inject .. DRAM-accept   (ICNT_TO_MEM, ROP_DELAY, ICNT_TO_L2,
//               L2_TO_DRAM, DRAM_LATENCY, DRAM, plus MC_* interface queues)
//   reply_side= DRAM-return .. SM-receive  (DRAM_TO_L2, L2_FILL, L2_TO_ICNT,
//               ICNT_TO_SHADER, CLUSTER_TO_SHADER, LDST_RESPONSE_FIFO)
//   queue_wait= anything not clearly on either side (INITIALIZED etc.)
void mem_fetch::print_tma_status_residency(FILE *fp) {
  unsigned long long total = 0;
  for (unsigned i = 0; i < NUM_MEM_REQ_STAT; ++i) total += s_tma_status_cycles[i];
  fprintf(fp, "TMA_status_residency_total_cycles = %llu\n", total);
  if (total == 0) return;

  unsigned long long req_side = 0, reply_side = 0, queue_wait = 0;
  for (unsigned i = 0; i < NUM_MEM_REQ_STAT; ++i) {
    unsigned long long c = s_tma_status_cycles[i];
    if (c == 0) continue;
    switch ((enum mem_fetch_status)i) {
      // MEM_FETCH_INITIALIZED = creation (in the TMA mover) until first partition
      // entry = time in the shared REQ icnt (injection + traversal). The plan's
      // req_side == icnt2mem is measured from mf creation, so this belongs here.
      case MEM_FETCH_INITIALIZED:
      case IN_ICNT_TO_MEM:
      case IN_PARTITION_ROP_DELAY:
      case IN_PARTITION_ICNT_TO_L2_QUEUE:
      case IN_PARTITION_L2_TO_DRAM_QUEUE:
      case IN_PARTITION_DRAM_LATENCY_QUEUE:
      case IN_PARTITION_L2_MISS_QUEUE:
      case IN_PARTITION_MC_INTERFACE_QUEUE:
      case IN_PARTITION_MC_INPUT_QUEUE:
      case IN_PARTITION_MC_BANK_ARB_QUEUE:
      case IN_PARTITION_DRAM:
      case IN_PARTITION_MC_RETURNQ:
        req_side += c;
        break;
      case IN_PARTITION_DRAM_TO_L2_QUEUE:
      case IN_PARTITION_L2_FILL_QUEUE:
      case IN_PARTITION_L2_TO_ICNT_QUEUE:
      case IN_ICNT_TO_SHADER:
      case IN_CLUSTER_TO_SHADER_QUEUE:
      case IN_SHADER_LDST_RESPONSE_FIFO:
        reply_side += c;
        break;
      default:
        queue_wait += c;
        break;
    }
    unsigned long long v = s_tma_status_visits[i];
    fprintf(fp, "TMA_status[%-28s] cycles=%llu visits=%llu avg=%.2f pct=%.2f\n",
            Status_str[i], c, v, v ? (double)c / (double)v : 0.0,
            100.0 * (double)c / (double)total);
  }
  fprintf(fp, "TMA_status_bucket_req_side_cycles   = %llu (%.2f%%)\n", req_side,
          100.0 * (double)req_side / (double)total);
  fprintf(fp, "TMA_status_bucket_reply_side_cycles = %llu (%.2f%%)\n",
          reply_side, 100.0 * (double)reply_side / (double)total);
  fprintf(fp, "TMA_status_bucket_queue_wait_cycles = %llu (%.2f%%)\n",
          queue_wait, 100.0 * (double)queue_wait / (double)total);
}

bool mem_fetch::isatomic() const {
  if (m_inst.empty()) return false;
  return m_inst.isatomic();
}

void mem_fetch::do_atomic() { m_inst.do_atomic(m_access.get_warp_mask()); }

bool mem_fetch::istexture() const {
  if (m_inst.empty()) return false;
  return m_inst.space.get_type() == tex_space;
}

bool mem_fetch::isconst() const {
  if (m_inst.empty()) return false;
  return (m_inst.space.get_type() == const_space) ||
         (m_inst.space.get_type() == param_space_kernel);
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the
/// direction
unsigned mem_fetch::get_num_flits(bool simt_to_mem) {
  unsigned sz = 0;
  // If atomic, write going to memory, or read coming back from memory, size =
  // ctrl + data. Else, only ctrl
  if (isatomic() || (simt_to_mem && get_is_write()) ||
      !(simt_to_mem || get_is_write()))
    sz = size();
  else
    sz = get_ctrl_size();

  return (sz / icnt_flit_size) + ((sz % icnt_flit_size) ? 1 : 0);
}
