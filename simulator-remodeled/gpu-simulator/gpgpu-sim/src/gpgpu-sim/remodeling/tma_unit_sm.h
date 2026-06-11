#pragma once

#include <deque>
#include <queue>
#include <vector>

#include "functional_unit.h"
#include "tma_types.h"

class shader_core_config;
class register_set_uniptr;
class SM;

class tma_unit_sm : public functional_unit_shared_sm_part {
 public:
  tma_unit_sm(std::vector<register_set_uniptr *> result_ports,
              std::vector<register_set_uniptr *> reception_ports,
              const shader_core_config *config, SM *sm);
  ~tma_unit_sm() override = default;

  void issue(register_set_uniptr &source_reg) override;
  void cycle() override;

 private:
  std::queue<TMACommand> m_command_queue;
  std::deque<TMATransferEntry> m_in_flight_transfers;
  std::vector<TMACompletionObject> m_completion_objects;

  TMACommand build_tma_command(const warp_inst_t &inst) const;
  uint32_t allocate_completion_object(const TMACommand &cmd);
  void enqueue_issued_commands();
  void advance_in_flight_transfers();
};
