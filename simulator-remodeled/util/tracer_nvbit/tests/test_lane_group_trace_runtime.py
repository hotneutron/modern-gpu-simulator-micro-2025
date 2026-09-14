import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACE = ROOT / "gpu-simulator/trace-driven/trace_driven.cc"
SM = ROOT / "gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc"


class LaneGroupTraceRuntimeTest(unittest.TestCase):
    def test_production_extraction_fold_and_sm_release(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler not installed")
        trace = TRACE.read_text()
        sm = SM.read_text()

        def section(text, start, end):
            return text[text.index(start):text.index(end)]

        harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "gpu-simulator/trace-driven/lane_group_replay.h"
#include "gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h"
using address_type = uint64_t;
struct inst_trace_t {
  uint64_t m_pc = 0, trace_instruction_index = 0, m_next_traced_pc = 0;
  unsigned m_unique_function_id = 1, active_mask = 0, mask = 0;
  std::string opcode;
  bool sync_runtime_valid = true, sync_has_semantic_raw = true;
  uint64_t sync_barrier_addr = 0x800, sync_semantic_raw = 0;
  std::vector<uint64_t> addresses;
};
struct ByPC {
  unsigned num_used_instructions[1] = {0}, num_traced_instructions = 0;
  std::vector<inst_trace_t> instructions;
};
struct IBuffer {
  LaneGroupReplay<inst_trace_t> *groups;
  void flush(bool) { while(groups->reserved()) groups->unreserve(); }
};
struct trace_shd_warp_t {
  LaneGroupReplay<inst_trace_t> m_lane_groups;
  LaneGroupSpanRecord m_lane_group_span;
  unsigned used_insts = 1;
  std::vector<address_type> traced_pcs;
  std::map<address_type, ByPC> map_warp_traces;
  IBuffer ib{&m_lane_groups};
  auto &lane_groups() { return m_lane_groups; }
  IBuffer *get_IBuffer_remodeled() { return &ib; }
  bool partial_capsule_active() { return m_lane_groups.active(); }
  uint64_t m_partial_replay_epoch=5;
  unsigned m_partial_replay_reserved=0;
  std::vector<inst_trace_t> m_partial_replay;
  enum class WaitBlockTraceOrigin { CANONICAL, PARTIAL };
  WaitBlockTraceOrigin m_partial_replay_origin=WaitBlockTraceOrigin::CANONICAL;
  uint64_t partial_replay_epoch() const { return m_partial_replay_epoch+m_lane_groups.epoch(); }
  bool reserve_next_partial_replay(address_type *, unsigned *, uint64_t *);
  void begin_lane_group_replay(const LaneGroupSpanRecord &, unsigned);
  bool lane_group_can_fetch() const;
  unsigned fold_lane_group_poll(unsigned, const SpinWaitRegionRecord &, FoldSkipStats *);
};
struct Config {
  bool is_trace_mode = true;
  uint64_t wait_block_hold_watchdog_cycles = 100, wait_block_heartbeat_cycles = 25;
};
struct GPU {
  SpinWaitRegionRecord poll;
  bool lookup_spin_wait_region(unsigned, unsigned, uint64_t, SpinWaitRegionRecord &r) {
    r = poll; return true;
  }
};
struct SM {
  Config config, *m_config=&config;
  GPU gpu, *m_gpu=&gpu;
  trace_shd_warp_t warp;
  std::vector<trace_shd_warp_t*> m_physical_warp{&warp};
  std::vector<std::array<HopperMBarrierPendingWait, 2>> m_lane_group_waits{1};
  std::vector<uint64_t> m_lane_group_progress_cycle{0};
  unsigned m_sm_id=5;
  std::vector<unsigned> m_local_to_global_cta_id{17};
  std::map<HopperMBarrierKey, HopperMBarrierObject> m_hopper_mbarriers;
  uint64_t m_wait_block_generation_overshoots_total=0, m_wait_block_watchdog_aborts_total=0;
  std::vector<std::string> events;
  uint64_t now=0, gen_a=0, gen_b=0;
  uint64_t m_lane_group_folded=0, m_fold_skipped_insts_total=0, m_fold_park_cycles_total=0;
  uint64_t m_fold_park_cycles_max=0;
  uint64_t m_fold_releases_total=0, m_missing_wait_releases_total=0;
  uint64_t m_fold_aux_releases_total=0, m_sync_debug_wait_released=0;
  uint64_t get_current_gpu_cycle() { return now; }
  uint64_t sync_barrier_gen(const HopperMBarrierKey &k) { return k.barrier_addr==0x800?gen_b:gen_a; }
  bool is_wait_block_pending_satisfied_at_cycle_start(const HopperMBarrierPendingWait &h) {
    return sync_barrier_gen(h.key) >= h.expected_gen;
  }
  void debug_log_wait_block_event(const char *tag, const std::string &detail) {
    events.push_back(std::string(tag)+" "+detail);
  }
  void debug_dump_tma_binding_snapshot(const char *) const {}
  void select_lane_group(unsigned);
  void service_lane_group_waits(unsigned);
  unsigned fold_lane_group(unsigned, unsigned, const SpinWaitRegionRecord &);
};
'''
        harness += section(trace, "void trace_shd_warp_t::begin_lane_group_replay(",
                           "bool trace_shd_warp_t::partial_replay_active()")
        harness += section(trace, "bool trace_shd_warp_t::reserve_next_partial_replay(",
                           "void trace_shd_warp_t::unreserve_last_partial_replay()")
        harness += section(sm, "void SM::select_lane_group(",
                           "void SM::service_wait_block_release(")
        harness += r'''
void load(SM &sm) {
  auto &w = sm.warp;
  std::vector<uint64_t> pcs{0x100,0x200,0x110,0x120,0x130,0x100,0x110,0x120};
  std::vector<unsigned> masks{14,1,14,14,14,14,14,15};
  for(size_t i=0;i<pcs.size();++i) {
    inst_trace_t t;
    t.m_pc=pcs[i]; t.trace_instruction_index=i; t.active_mask=t.mask=masks[i];
    t.opcode=t.m_pc==0x200 ? "UTMALDG.4D" : t.m_pc==0x110 ? "NANOSLEEP.SYNCS"
        : t.m_pc==0x130 ? "BRA" : t.m_pc==0x100 ? "SYNCS.PHASECHK.TRANS64.TRYWAIT"
                                             : "SYNCS.PHASECHK.TRANS64";
    t.addresses={0x100000+i, 0x200000+i};
    w.traced_pcs.push_back(t.m_pc);
    auto &v=w.map_warp_traces[t.m_pc]; v.instructions.push_back(t); ++v.num_traced_instructions;
  }
  w.map_warp_traces[0x100].num_used_instructions[0]=1;
}
LaneGroupSpanRecord span() {
  LaneGroupSpanRecord s;
  s.start_index=0; s.merge_index=7; s.reorderable_until=6; s.merge_pc=0x120;
  s.start_mask=14; s.start_pc=0x100; s.start_barrier_addr=0x800;
  s.main_mask=1; s.polling_mask=14; s.live_mask=15;
  s.main_count=1; s.polling_count=s.reorderable_count=6;
  s.poll_head_pc=0x100; s.poll_probe_pc=0x120; s.poll_loopback_pc=0x130;
  s.barrier_addr=0x800; s.merge_kind="POLL";
  return s;
}
void long_sibling_wait(bool no_progress, bool overshoot, bool late_ready,
                       bool stop_after_progress, bool exact_late) {
  SM sm; load(sm); sm.warp.begin_lane_group_replay(span(),1);
  auto &g=sm.warp.lane_groups();
  auto &held=sm.m_lane_group_waits[0][1];
  held.valid=true; held.expected_gen=1; held.key.barrier_addr=0x800;
  held.uses_exact_generation_ticket=exact_late;
  held.mode=WaitBlockPendingMode::CAPTURED_POLL_HOLD;
  held.block_id="poll"; held.action_pc=0x120; held.unique_function_id=1;
  sm.gpu.poll.block_id="poll"; sm.gpu.poll.foldable=true;
  sm.gpu.poll.loop_head_pc=0x100; sm.gpu.poll.primary_wait_pc=0x120;
  sm.gpu.poll.loopback_bra_pc=0x130; sm.gpu.poll.exit_pc=0x140;
  sm.gpu.poll.body_pcs={0x100,0x110,0x120};
  auto &barrier=sm.m_hopper_mbarriers[held.key];
  barrier.expected_arrive_count=1; barrier.phase=0;
  g.set_waiting(1,true);
  for(sm.now=25;sm.now<=100;sm.now+=25) sm.service_lane_group_waits(0);
  sm.now=100;
  assert(sm.events.size()==4 && held.last_heartbeat_cycle==100);
  assert(sm.events.back().find("held_cycles=100 replay_idle_cycles=100")!=std::string::npos);
  assert(sm.m_lane_group_progress_cycle[0]==0 && held.hold_start_cycle==0);
  assert(g.waiting(1) && g.selected()==0);
  if(!no_progress) {
    size_t index; uint64_t epoch;
    assert(g.reserve(index,epoch));
    assert(g.reserved_item(index,epoch).instruction.addresses==
           std::vector<uint64_t>({0x100001,0x200001}));
    g.commit(index);
    sm.m_lane_group_progress_cycle[0]=sm.now;
  }
  sm.now=101;
  if(overshoot) sm.gen_b=2;
  if(late_ready) sm.gen_b=1;
  if(exact_late) sm.gen_b=2;
  sm.service_lane_group_waits(0);
  if(late_ready || exact_late) {
    assert(!g.active() && sm.m_fold_releases_total==1);
    return;
  }
  // Long total age is not permission to release the still-pending dependency.
  assert(g.active() && g.waiting(1) && held.valid && held.expected_gen==1);
  assert(held.hold_start_cycle==0 && sm.m_fold_releases_total==0);
  sm.now=200; sm.service_lane_group_waits(0);
  if(no_progress) return; // Must already have aborted at 101.
  assert(held.hold_start_cycle==0 && sm.m_lane_group_progress_cycle[0]==100);
  if(stop_after_progress) { sm.now=201; sm.service_lane_group_waits(0); return; }
  sm.now=201; sm.gen_b=1; sm.service_lane_group_waits(0);
  assert(!g.active() && sm.m_fold_releases_total==1);
  assert(sm.m_fold_park_cycles_total==201);
  assert(sm.warp.used_insts==7 && sm.m_lane_group_folded==5);
}
int main(int argc, char **argv) {
  std::string mode=argc>1?argv[1]:"";
  if(mode=="long-sibling" || mode=="no-progress" ||
     mode=="overshoot-with-progress" || mode=="late-ready" ||
     mode=="stalled-after-progress" || mode=="exact-late") {
    long_sibling_wait(mode=="no-progress", mode=="overshoot-with-progress",
                      mode=="late-ready", mode=="stalled-after-progress",
                      mode=="exact-late");
    return 0;
  }
  for(bool poll_first : {true,false}) {
    SM sm; load(sm); auto s=span();
    if(mode=="mask") sm.warp.map_warp_traces[0x200].instructions[0].active_mask=3;
    if(mode=="identity") sm.warp.map_warp_traces[0x200].instructions[0].trace_instruction_index=2;
    if(mode=="count") ++s.main_count;
    if(mode=="condition") sm.warp.map_warp_traces[0x120].instructions[0].sync_barrier_addr=0x900;
    sm.warp.begin_lane_group_replay(s,1);
    auto &g=sm.warp.lane_groups();
    assert(sm.warp.used_insts==7 && g.remaining(0)+g.remaining(1)==6);
    assert(g.head(0).source_index==1);
    assert(g.head(0).instruction.addresses==std::vector<uint64_t>({0x100001,0x200001}));
    assert(g.head(0).instruction.active_mask==1);
    g.select(0); size_t index; uint64_t epoch;
    assert(g.reserve(index,epoch)); sm.warp.ib.flush(false);
    assert(g.issued(0)==0 && !g.reserved());
    for(unsigned group=0;group<2;++group) {
      auto &h=sm.m_lane_group_waits[0][group];
      h.valid=true; h.expected_gen=1; h.uses_exact_generation_ticket=true;
      h.key.barrier_addr=group==0?0x900:0x800;
      h.mode=group==0?WaitBlockPendingMode::MISSING_WAIT_HOLD:WaitBlockPendingMode::CAPTURED_POLL_HOLD;
      h.block_id="poll"; h.action_pc=0x120; h.unique_function_id=1;
      g.set_waiting(group,true);
    }
    sm.gpu.poll.block_id="poll"; sm.gpu.poll.foldable=true;
    sm.gpu.poll.loop_head_pc=0x100; sm.gpu.poll.primary_wait_pc=0x120;
    sm.gpu.poll.loopback_bra_pc=0x130; sm.gpu.poll.exit_pc=0x140;
    sm.gpu.poll.body_pcs={0x100,0x110,0x120};
    sm.select_lane_group(0);
    assert(!sm.warp.lane_group_can_fetch());
    sm.now=1;
    if(poll_first) sm.gen_b=1; else sm.gen_a=1;
    if(mode=="overshoot") {
      sm.m_lane_group_waits[0][1].uses_exact_generation_ticket=false;
      sm.gen_b=2;
    }
    if(mode=="watchdog") { sm.now=101; sm.gen_a=sm.gen_b=0; }
    sm.service_lane_group_waits(0);
    assert(g.active() && g.waiting(poll_first?0:1));
    if(poll_first) { assert(g.remaining(1)==0 && g.remaining(0)==1); sm.gen_a=1; }
    else {
      assert(g.selected()==0 && sm.warp.lane_group_can_fetch());
      assert(g.reserve(index,epoch)); g.commit(index);
      sm.select_lane_group(0); assert(g.active()); sm.gen_b=1;
    }
    sm.now=2; sm.service_lane_group_waits(0);
    if(poll_first) {
      assert(g.reserve(index,epoch)); g.commit(index); sm.select_lane_group(0);
    }
    assert(!g.active());
    assert(sm.m_lane_group_folded==5);
    assert(sm.m_missing_wait_releases_total==1 && sm.m_fold_releases_total==1);
    assert(sm.warp.used_insts==7);
    // The full-mask occurrence is still canonical, neither copied nor folded.
    auto &merge=sm.warp.map_warp_traces[0x120];
    assert(merge.num_used_instructions[0]==1 && merge.instructions[1].mask==15);
    sm.warp.m_partial_replay_origin=trace_shd_warp_t::WaitBlockTraceOrigin::PARTIAL;
    sm.warp.m_partial_replay.push_back(merge.instructions[1]);
    unsigned slot; address_type pc;
    assert(sm.warp.reserve_next_partial_replay(&pc,&slot,&epoch));
    assert(epoch==sm.warp.partial_replay_epoch() && pc==0x120 && slot==0);
  }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / "runtime")
            built = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-I", str(ROOT), "-x", "c++", "-", "-o", binary],
                input=harness, text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for mode in ("long-sibling", "late-ready", "exact-late"):
                with self.subTest(mode=mode):
                    result = subprocess.run([binary, mode], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
            for mode in ("mask", "identity", "count", "condition", "overshoot", "watchdog",
                         "no-progress", "overshoot-with-progress", "stalled-after-progress"):
                with self.subTest(mode=mode):
                    result = subprocess.run([binary, mode], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    if mode in ("watchdog", "no-progress", "stalled-after-progress"):
                        self.assertIn("reason=lane-group-progress-watchdog", result.stderr)
                        self.assertIn("replay_idle_cycles=101 threshold=100", result.stderr)
                        self.assertIn("global_cta=17", result.stderr)
                    if mode in ("overshoot", "overshoot-with-progress"):
                        self.assertIn("reason=lane-group-generation-overshoot", result.stderr)

if __name__ == "__main__":
    unittest.main()
