"""Exercise production copy-group tracking and scheduler readiness without CUDA."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling"


class CopyGroupRuntimeTests(unittest.TestCase):
    def test_group_frontier_and_generic_scoreboard(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler required")
        header = (SOURCE / "warp_dependency_state.h").read_text().replace(
            '#include "../../abstract_hardware_model.h"',
            "using new_addr_type = unsigned long long;").replace("#pragma once", "")
        implementation = (SOURCE / "warp_dependency_state.cc").read_text()
        implementation = implementation[implementation.index("Wait_Barrier::Wait_Barrier"):]
        subcore = (SOURCE / "subcore.cc").read_text()
        predicate = subcore[subcore.index("bool Subcore::is_waiting_ldgdepbar("):
                            subcore.index("bool Subcore::is_wait_barriers_ready_entry_point(")]
        program = r"""
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
struct shader_core_config { unsigned num_wait_barriers_per_warp = 6; };
struct shader_core_stats {};
"""
        program += header + implementation
        program += r"""
constexpr unsigned LDGDEPBAR_OP = 42;
struct warp_inst_t { unsigned op = LDGDEPBAR_OP; unsigned long long pc = 0x1050; };
struct Warp {
  Dependency_State *dep;
  Dependency_State *get_dependency_state() { return dep; }
};
struct SM {
  unsigned get_sid() { return 0; }
  unsigned get_current_gpu_cycle() { return 10; }
};
struct Subcore {
  unsigned m_subcore_id = 0;
  SM *m_sm;
  std::vector<Warp *> m_warps_of_subcore;
  bool is_waiting_ldgdepbar(const warp_inst_t *, unsigned);
};
"""
        program += predicate
        program += r"""
void arm(Dependency_State &state, unsigned sb) {
  Wait_Barrier_Entry_Modifier action(0, sb, WRITE_WAIT_BARRIER, INCREASE_COUNTER, 0);
  state.action_over_wait_barrier(&action);
}
void release(Dependency_State &state, const std::vector<unsigned> &barriers) {
  for (unsigned sb : barriers) {
    Wait_Barrier_Entry_Modifier action(0, sb, WRITE_WAIT_BARRIER, DECREASE_COUNTER, 0);
    state.action_over_wait_barrier(&action);
  }
}
void commit(Dependency_State &state, unsigned sb) {
  // Commit must expose its credit before the same-cycle issue readiness
  // check. Only completion decrements are deferred to the SM cycle boundary.
  release(state, state.commit_ldgsts_group(sb));
}
int main() {
  shader_core_config config;
  Dependency_State state(&config, nullptr);
  Warp warp{&state}; SM sm; Subcore sub{0, &sm, {&warp}};
  warp_inst_t instruction;
  for (unsigned token : {1, 2, 3}) state.increase_num_pending_ldgsts(token);
  assert(!sub.is_waiting_ldgdepbar(&instruction, 0));
  commit(state, 0);
  state.increase_num_pending_ldgsts(4);
  commit(state, 0);
  assert(!state.peek_wait_barrier_ready(0, 1));
  // A newer single-copy group finishes first. wait_group 1 must still wait
  // for every copy in the older three-copy group.
  release(state, state.decrease_num_pending_ldgsts(4));
  assert(!state.peek_wait_barrier_ready(0, 1));
  release(state, state.decrease_num_pending_ldgsts(2));
  release(state, state.decrease_num_pending_ldgsts(1));
  assert(!state.are_wait_barriers_ready({Wait_Barrier_Checking(0, 1)}));
  release(state, state.decrease_num_pending_ldgsts(3));
  assert(state.peek_wait_barrier_ready(0, 0));
  assert(!state.are_pending_dependencies());

  // Older completes first: wait_group 1 releases, wait_group 0 blocks.
  state.increase_num_pending_ldgsts(5); commit(state, 0);
  state.increase_num_pending_ldgsts(6); commit(state, 0);
  release(state, state.decrease_num_pending_ldgsts(5));
  assert(state.peek_wait_barrier_ready(0, 1));
  assert(!state.peek_wait_barrier_ready(0, 0));
  release(state, state.decrease_num_pending_ldgsts(6));

  // Retirement before commit and truly empty commits must not leak credits.
  state.increase_num_pending_ldgsts(7);
  release(state, state.decrease_num_pending_ldgsts(7));
  commit(state, 0);
  commit(state, 0);
  assert(!state.are_pending_dependencies());

  // Ordinary scoreboard users survive copy group completion on the same SB.
  arm(state, 0);
  state.increase_num_pending_ldgsts(8); commit(state, 0);
  release(state, state.decrease_num_pending_ldgsts(8));
  assert(!state.peek_wait_barrier_ready(0, 0));
  release(state, {0});
  assert(!state.are_pending_dependencies());
  // Credits follow the decoded barrier, not a hardcoded SB0.
  state.increase_num_pending_ldgsts(9); commit(state, 3);
  assert(state.peek_wait_barrier_ready(0, 0));
  assert(!state.peek_wait_barrier_ready(3, 0));
  release(state, state.decrease_num_pending_ldgsts(9));
  state.reset();
  assert(!state.are_pending_dependencies());
  // Slot reuse with a fresh warp cannot inherit a completed group.
  state.increase_num_pending_ldgsts(1); commit(state, 0);
  release(state, state.decrease_num_pending_ldgsts(1));
  assert(!state.are_pending_dependencies());
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "copy_groups.cc"
            binary = Path(tmp) / "copy_groups"
            source.write_text(program)
            subprocess.run([compiler, "-std=c++14", str(source), "-o", str(binary)], check=True)
            env = os.environ.copy()
            env.pop("GCOM_COPY_DEBUG", None)
            subprocess.run([str(binary)], check=True, env=env)


if __name__ == "__main__":
    unittest.main()

# ## Revision History
# 2026-09-22: Require commit itself to publish the credit for an adjacent wait.
# 2026-09-22: Production runtime checks for nonblocking commit, unequal groups,
# out-of-order completion, empty groups, generic SB coexistence, and warp reuse.
