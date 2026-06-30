#pragma once

#include <cstdint>
#include <deque>
#include <queue>
#include <unordered_map>
#include <vector>

#include "functional_unit.h"
#include "tma_types.h"

class shader_core_config;
class register_set_uniptr;
class SM;
class mem_fetch;
class mem_fetch_interface;
class mem_fetch_allocator;

class tma_unit_sm : public functional_unit_shared_sm_part {
 public:
  tma_unit_sm(std::vector<register_set_uniptr *> result_ports,
              std::vector<register_set_uniptr *> reception_ports,
              const shader_core_config *config, SM *sm,
              mem_fetch_interface *icnt, mem_fetch_allocator *mf_allocator);
  ~tma_unit_sm() override;

  void issue(register_set_uniptr &source_reg) override;
  void cycle() override;

  // Called by the SM when a TMA-tagged response returns from the memory
  // hierarchy (L2/DRAM). Distinct from the ordinary ldst_unit fill path.
  void fill(mem_fetch *mf);

  void debug_dump_tma_counters() const;

  // True while `warp_id` still has store-class TMA transfers (UTMASTG /
  // UTMAREDG / UBLKRED) that have been enqueued but not yet completed. A
  // UTMACMDFLUSH (cp.async.bulk.wait_group 0) must stall its issuing warp until
  // this drains to zero (warp-local drain-all). Loads do not count: their
  // completion is tracked by the mbarrier transaction count, not the bulk
  // async-group.
  bool warp_has_outstanding_stores(unsigned int warp_id) const;

 private:
  // Maximum number of bulk requests the TMA engine launches into the shared
  // interconnect per cycle. Conservative first-model bandwidth bound.
  static constexpr uint32_t kMaxRequestsPerCycle = 2;

  std::queue<TMACommand> m_command_queue;
  std::deque<TMATransferEntry> m_in_flight_transfers;

  // Shared physical interconnect to L2/DRAM (NOT owned). TMA shares the same
  // icnt as the ldst unit because L2/DRAM is a shared resource in real HW, but
  // request generation, L1 bypass and completion tracking remain TMA-private.
  mem_fetch_interface *m_icnt = nullptr;
  mem_fetch_allocator *m_mf_allocator = nullptr;

  // Maps an outstanding TMA mem_fetch back to the transfer that launched it,
  // so a returning response credits the correct in-flight transfer.
  std::unordered_map<mem_fetch *, uint64_t> m_outstanding_requests;
  uint64_t m_next_transfer_uid = 1;

  // Per-warp count of store-class transfers (UTMASTG / UTMAREDG / UBLKRED) that
  // have been enqueued but not yet completed. Incremented when a store-class
  // command is enqueued, decremented when its data movement completes. Drives
  // the UTMACMDFLUSH warp-local drain-all wait (see warp_has_outstanding_stores).
  std::unordered_map<unsigned int, uint32_t> m_outstanding_stores_per_warp;

  // ---- TMA-private debug counters (separate from L1/ldst counters) ----
  uint64_t m_stat_commands_issued = 0;
  uint64_t m_stat_transfers_completed = 0;
  uint64_t m_stat_requests_issued = 0;
  uint64_t m_stat_requests_completed = 0;
  uint64_t m_stat_bytes_issued = 0;
  uint64_t m_stat_bytes_completed = 0;
  uint64_t m_stat_load_bytes_issued = 0;
  uint64_t m_stat_store_bytes_issued = 0;
  uint64_t m_stat_reduce_bytes_issued = 0;
  uint64_t m_stat_load_bytes_completed = 0;
  uint64_t m_stat_store_bytes_completed = 0;
  uint64_t m_stat_reduce_bytes_completed = 0;
  uint64_t m_stat_icnt_backpressure_events = 0;
  uint64_t m_stat_timed_transfers = 0;
  uint64_t m_stat_issue_active_cycles = 0;
  uint64_t m_stat_icnt_full_cycles = 0;
  uint64_t m_stat_to_first_request_cycles = 0;
  uint64_t m_stat_emit_span_cycles = 0;
  uint64_t m_stat_drain_cycles = 0;

  TMACommand build_tma_command(const warp_inst_t &inst) const;
  void enqueue_issued_commands();
  void advance_in_flight_transfers();

  // ---- Blackwell seam ----
  // The mechanics of moving data for one transfer are isolated behind these
  // hooks. Hopper performs explicit GMEM->SMEM bulk reads that bypass L1 and
  // complete on memory-hierarchy responses. Blackwell's data movement differs
  // fundamentally and will plug a different mover in here (additive, later).
  void mover_issue_requests(TMATransferEntry &entry, int current_cycle);
  void mover_on_response(TMATransferEntry &entry, mem_fetch *mf,
                         int current_cycle);
};
