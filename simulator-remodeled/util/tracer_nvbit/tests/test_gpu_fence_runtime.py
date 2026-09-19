"""Execute production fence classification, store drain and error-barrier decode."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def section(text, start, end):
    return text[text.index(start):text.index(end)]


class FenceRuntimeTests(unittest.TestCase):
    def test_gpu_store_drain_and_misc_error_barrier(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler required")
        sm = (ROOT / "gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc").read_text()
        trace = (ROOT / "gpu-simulator/trace-driven/trace_driven.cc").read_text()
        program = r'''
#include <cassert>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "gpu-simulator/ISA_Def/hopper_opcode.h"
enum membar_scope_t { MEMBAR_SCOPE_NONE, MEMBAR_SCOPE_CTA, MEMBAR_SCOPE_GPU };
struct Info {
  std::string opcode;
  std::string get_op_code() const { return opcode; }
};
struct warp_inst_t {
  unsigned op;
  Info info;
  bool has_extra_trace_instruction_info() const { return true; }
  const Info &get_extra_trace_instruction_info() const { return info; }
};
struct Kernel { bool is_captured_from_binary=true; };
struct Warp {
  Kernel kernel;
  bool membar=false;
  unsigned cta=0, gpu=0;
  membar_scope_t scope=MEMBAR_SCOPE_NONE;
  bool get_membar() { return membar; }
  Kernel *get_kernel_info() { return &kernel; }
  membar_scope_t get_membar_scope() { return scope; }
  void clear_membar() { membar=false; }
  unsigned get_pending_stores_cta_visible() const { return cta; }
  unsigned get_pending_stores_gpu_visible() const { return gpu; }
  bool stores_done() { return cta==0 && gpu==0; }
};
struct Config {
  bool is_remodeling_scoreboarding_enabled=false, is_trace_mode=true;
  bool bar_debug_enable=false;
  unsigned bar_debug_issue_budget=0;
  bool flush_l1() { return false; }
};
struct GPU { Config config; Config &get_config() { return config; } };
struct Scoreboard {
  bool pendingWrites(unsigned) { return false; }
  bool pendingReads(unsigned) { return false; }
};
struct TMA {
  bool pending=false;
  bool warp_has_outstanding_stores(unsigned) { return pending; }
};
struct SM {
  Warp warp, sibling;
  std::vector<Warp*> m_physical_warp{&warp, &sibling};
  Config config, *m_config=&config;
  GPU gpu, *m_gpu=&gpu;
  Scoreboard sb, *m_scoreboard=&sb, *m_scoreboard_WAR=&sb;
  TMA tma, *m_tma_unit_shared_of_sm=&tma;
  unsigned m_sm_id=0;
  std::map<unsigned,unsigned long long> m_membar_wait_start_cycle, m_membar_last_stuck_warn_cycle;
  unsigned long long get_current_gpu_cycle() { return 100; }
  void data_cache_invalidate() {}
  bool warp_waiting_at_mem_barrier(unsigned);
  bool warp_has_pending_fence_stores(unsigned, int) const;
};
'''
        program += section(sm, "bool is_non_rendezvous_memory_barrier(",
                           "// SYNCS.ARRIVE.TRANS64 variants")
        program += section(sm, "bool SM::warp_waiting_at_mem_barrier(",
                           "bool SM::warp_waiting_grid_barrier(")
        program += section(sm, "bool SM::warp_has_pending_fence_stores(",
                           "void SM::display_simt_state(")
        program += r'''
int main(int argc, char **) {
  if(argc>1) {
    warp_inst_t bad{MEMORY_BARRIER_OP, {"MEMBAR.SC.SYS"}};
    is_non_rendezvous_memory_barrier(bad); // still unqualified
    return 0;
  }
  for(const std::string op : {"MEMBAR.ALL.GPU", "MEMBAR.SC.GPU"}) {
    warp_inst_t inst{Hopper_OpcodeMap.at("MEMBAR").opcode_category, {op}};
    assert(is_non_rendezvous_memory_barrier(inst));
    assert(fence_scope_of(inst)==MEMBAR_SCOPE_GPU);
    SM sm;
    sm.warp.membar=true; sm.warp.scope=fence_scope_of(inst);
    sm.warp.cta=1; sm.warp.gpu=1; sm.tma.pending=true;
    assert(sm.warp_waiting_at_mem_barrier(0));
    sm.warp.cta=0;
    assert(sm.warp_waiting_at_mem_barrier(0));
    sm.warp.gpu=0;
    assert(sm.warp_waiting_at_mem_barrier(0));
    sm.tma.pending=false;
    // Other warps need not arrive or drain for this per-warp fence.
    sm.sibling.gpu=99;
    assert(!sm.warp_waiting_at_mem_barrier(0));
    assert(!sm.warp.membar);
  }
  warp_inst_t error{Hopper_OpcodeMap.at("CGAERRBAR").opcode_category, {"CGAERRBAR"}};
  assert(error.op==MISCELLANEOUS_QUEUE_OP);
  assert(!is_non_rendezvous_memory_barrier(error));
  assert(fence_scope_of(error)==MEMBAR_SCOPE_NONE);
  const int SYNC=1;
  unsigned bar_id=99, bar_count=42;
  int bar_type=99;
  switch(Hopper_OpcodeMap.at("CGAERRBAR").opcode) {
''' + section(trace, "    case OP_CGAERRBAR:", "    case OP_BAR: {") + r'''
    default: assert(false);
  }
  assert(bar_id==99 && bar_count==42 && bar_type==99);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / "fence-runtime")
            built = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT),
                 "-x", "c++", "-", "-o", binary], input=program, text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([binary, "unqualified"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

# ## Revision History
# 2026-09-19: verify GPU fence completion waits for CTA/L2/TMA stores, never
# rendezvous with sibling warps, and keep CGAERRBAR on the misc queue path.
