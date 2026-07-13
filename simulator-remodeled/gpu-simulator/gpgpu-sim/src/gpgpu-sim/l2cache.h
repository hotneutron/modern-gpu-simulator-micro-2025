// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
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
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "../abstract_hardware_model.h"
#include "dram.h"

#include <list>
#include <queue>

class mem_fetch;
class memory_stats_t;

class partition_mf_allocator : public mem_fetch_allocator {
 public:
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;
  }
  ~partition_mf_allocator() {}
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    abort();
    return NULL;
  }
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr,
                           unsigned long long cycle) const;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf) const;

 private:
  const memory_config *m_memory_config;
};

// Memory partition unit contains all the units assolcated with a single DRAM
// channel.
// - It arbitrates the DRAM channel among multiple sub partitions.
// - It does not connect directly with the interconnection network.
class memory_partition_unit {
 public:
  memory_partition_unit(unsigned partition_id, const memory_config *config,
                        class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_partition_unit();

  bool busy() const;

  void cache_cycle(unsigned cycle);
  void dram_cycle();
  void simple_dram_model_cycle();

  void set_done(mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file) const;
  void print_stat(FILE *fp) { m_dram->print_stat(fp); }
  void visualize() const { m_dram->visualize(); }
  void print(FILE *fp) const;
  void handle_memcpy_to_gpu(size_t dst_start_addr, unsigned subpart_id,
                            mem_access_sector_mask_t mask);

  class memory_sub_partition *get_sub_partition(int sub_partition_id) {
    return m_sub_partition[sub_partition_id];
  }

  // Power model
  void set_dram_power_stats(unsigned &n_cmd, unsigned &n_activity,
                            unsigned &n_nop, unsigned &n_act, unsigned &n_pre,
                            unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
                            unsigned &n_req) const;

  int global_sub_partition_id_to_local_id(int global_sub_partition_id) const;

  unsigned get_mpid() const { return m_id; }

  class gpgpu_sim *get_mgpu() const {
    return m_gpu;
  }

  memory_stats_t* get_memory_partition_stats() { return m_stats; };

 private:
  unsigned m_id;
  const memory_config *m_config;
  memory_stats_t *m_stats;
  class memory_sub_partition **m_sub_partition;
  class dram_t *m_dram;

  class arbitration_metadata {
   public:
    arbitration_metadata(const memory_config *config);

    // check if a subpartition still has credit
    bool has_credits(int inner_sub_partition_id) const;
    // borrow a credit for a subpartition
    void borrow_credit(int inner_sub_partition_id);
    // return a credit from a subpartition
    void return_credit(int inner_sub_partition_id);

    // return the last subpartition that borrowed credit
    int last_borrower() const { return m_last_borrower; }

    void print(FILE *fp) const;

   private:
    // id of the last subpartition that borrowed credit
    int m_last_borrower;

    int m_shared_credit_limit;
    int m_private_credit_limit;

    // credits borrowed by the subpartitions
    std::vector<int> m_private_credit;
    int m_shared_credit;
  };
  arbitration_metadata m_arbitration_metadata;

  // determine wheither a given subpartition can issue to DRAM
  bool can_issue_to_dram(int inner_sub_partition_id);

  // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
  struct dram_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::list<dram_delay_t> m_dram_latency_queue;

  class gpgpu_sim *m_gpu;
};

class memory_sub_partition {
 public:
  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_sub_partition();

  unsigned get_id() const { return m_id; }

  bool busy() const;

  void cache_cycle(unsigned cycle);

  bool full() const;
  bool full(unsigned size) const;
  void push(class mem_fetch *mf, unsigned long long clock_cycle);
  class mem_fetch *pop();
  class mem_fetch *top();
  void set_done(mem_fetch *mf);

  unsigned flushL2();
  unsigned invalidateL2();

  // interface to L2_dram_queue
  bool L2_dram_queue_empty() const;
  class mem_fetch *L2_dram_queue_top() const;
  void L2_dram_queue_pop();

  // interface to dram_L2_queue
  bool dram_L2_queue_full() const;
  void dram_L2_queue_push(class mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file);
  void print_cache_stat(unsigned &accesses, unsigned &misses) const;
  void print(FILE *fp) const;

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
  void clear_L2cache_stats_pw();

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    m_memcpy_cycle_offset += 1;
  }

  // Opt6 Part-0 TMA L2 diagnosis (timing-neutral observers).
  unsigned long long get_tma_l2_hits() const { return m_tma_l2_hits; }
  unsigned long long get_tma_l2_pending_hits() const {
    return m_tma_l2_pending_hits;
  }
  unsigned long long get_tma_l2_misses() const { return m_tma_l2_misses; }
  unsigned long long get_tma_l2_res_fails() const { return m_tma_l2_res_fails; }
  unsigned long long get_tma_l2_output_full_cycles() const {
    return m_tma_l2_output_full_cycles;
  }
  unsigned long long get_tma_l2_port_busy_cycles() const {
    return m_tma_l2_port_busy_cycles;
  }

 private:
  // data
  unsigned m_id;  //< the global sub partition ID
  const memory_config *m_config;
  class l2_cache *m_L2cache;
  class L2interface *m_L2interface;
  class gpgpu_sim *m_gpu;
  partition_mf_allocator *m_mf_allocator;

  // model delay of ROP units with a fixed latency
  struct rop_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::queue<rop_delay_t> m_rop;

  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_dram_queue;
  fifo_pipeline<mem_fetch> *m_dram_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue

  class mem_fetch *L2dramout;
  unsigned long long int wb_addr;

  class memory_stats_t *m_stats;

  std::set<mem_fetch *> m_request_tracker;

  friend class L2interface;

  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to
  // the gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;

  // Opt6 Part-0: TMA-only L2 admission outcome counters. The aggregate
  // L2_total_cache_* stats mix TMA with normal LDG/STG, so they cannot tell
  // whether the RESERVATION_FAIL re-probe storm (ADDR_MERGE synthetic-address
  // hotspot) is TMA-driven. These count only is_tma() requests at the admission
  // probe in cache_cycle(). Pure observers; no timing effect.
  //  - m_tma_l2_hits        : true HIT (line resident, data returned now)
  //  - m_tma_l2_pending_hits: HIT_RESERVED / MSHR_HIT (line already in-flight;
  //                           merged onto an outstanding miss -> no new DRAM
  //                           traffic). This is the bucket that answers "isn't a
  //                           single synthetic base always an L2 hit?": cross-SM
  //                           reuse of one fabricated line shows up here, not as
  //                           a true hit, and it still costs the miss latency of
  //                           the first requester.
  //  - m_tma_l2_misses      : MISS / SECTOR_MISS (goes to DRAM)
  //  - m_tma_l2_res_fails   : RESERVATION_FAIL re-probe cycles (head-of-line
  //                           blocking; the hotspot pressure signal)
  //
  // The four counters above only advance when access() is actually called,
  // i.e. when the admission gate (!output_full && port_free) is open. A TMA mf
  // can also be stuck at the queue head for two *downstream-backpressure*
  // reasons that prevent access() entirely; without separating them, a low
  // res_fail could be misread as "admission is not the limiter" when in fact
  // the head is jammed downstream. These two count those blocked cycles so the
  // four head-stall causes (dram-queue-full -> gpu_stall_dramfull; output-full;
  // port-busy; reservation-fail) are fully separable from one TMA run:
  //  - m_tma_l2_output_full_cycles: head is TMA and m_L2_icnt_queue is full
  //                                 (L2->ICNT reply queue backpressure)
  //  - m_tma_l2_port_busy_cycles  : head is TMA, output not full, but the L2
  //                                 data port is busy this cycle
  unsigned long long m_tma_l2_hits = 0;
  unsigned long long m_tma_l2_pending_hits = 0;
  unsigned long long m_tma_l2_misses = 0;
  unsigned long long m_tma_l2_res_fails = 0;
  unsigned long long m_tma_l2_output_full_cycles = 0;
  unsigned long long m_tma_l2_port_busy_cycles = 0;
  // Opt8 admission-parallelism instrumentation (timing-neutral). Per-sub-partition
  // so gpu_print_stat can build the across-slice histogram that reveals whether the
  // ROP wall is a few hot slices (spread problem) or a genuine per-slice throughput
  // limit (the Opt8 lever). See L2_SLICE_PARALLELISM_H100.md section 8.
  //  - m_l2_admissions      : total accepted probes (== this slice's L2 accesses)
  //  - m_l2_active_cycles   : L2 ticks where this slice accepted >=1 probe
  //  - m_l2_multi_admit_cycles: L2 ticks where this slice accepted >1 (proves the
  //                            widened budget was actually used; 0 when knob==1)
  unsigned long long m_l2_admissions = 0;
  unsigned long long m_l2_active_cycles = 0;
  unsigned long long m_l2_multi_admit_cycles = 0;

 public:
  unsigned long long get_l2_admissions() const { return m_l2_admissions; }
  unsigned long long get_l2_active_cycles() const { return m_l2_active_cycles; }
  unsigned long long get_l2_multi_admit_cycles() const {
    return m_l2_multi_admit_cycles;
  }

 private:
};

class L2interface : public mem_fetch_interface {
 public:
  L2interface(memory_sub_partition *unit) { m_unit = unit; }
  virtual ~L2interface() {}
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_L2_dram_queue->full();
  }
  // Defined out-of-line in l2cache.cc so it can stamp the real GPU cycle. The
  // old inline body used a hard-coded 0 for the status-change timestamp (the
  // original FIXME), which corrupted any per-stage residency measured off
  // m_status_change: the next transition computed (cycle - 0) = an absolute
  // timestamp instead of a delta. This is the L2 miss port, so every enabled-L2
  // miss hit it.
  virtual void push(mem_fetch *mf);

  virtual void flush() {}

 private:
  memory_sub_partition *m_unit;
};

#endif
