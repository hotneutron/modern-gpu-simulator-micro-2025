#include "stat_gating.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

StatGating::StatGating() : enabled_(false), total_cycles_(0) {}

StatGating::~StatGating() {}

void StatGating::reset() {
  enabled_ = false;
  total_cycles_ = 0;
  segments_by_cta_.clear();
  stats_.clear();
}

bool StatGating::load_segments(const std::string& txt_path) {
  reset();
  std::ifstream f(txt_path);
  if (!f.is_open()) {
    fprintf(stderr, "StatGating: cannot open %s\n", txt_path.c_str());
    return false;
  }

  // Format: "start_inst end_inst" per line
  unsigned id = 0;
  unsigned long long start_inst, end_inst;
  while (f >> start_inst >> end_inst) {
    SegmentDef sd;
    sd.segment_id = id++;
    sd.start_inst = start_inst;
    sd.end_inst = end_inst;
    sd.cta_x = 0;
    sd.cta_y = 0;
    sd.cta_z = 0;
    segments_by_cta_[0].push_back(sd);
  }

  enabled_ = !segments_by_cta_[0].empty();
  total_cycles_ = 0;

  if (enabled_) {
    fprintf(stderr, "StatGating: loaded %zu segments from %s\n",
            segments_by_cta_[0].size(), txt_path.c_str());
  } else {
    fprintf(stderr, "StatGating: no segments found in %s\n", txt_path.c_str());
  }
  return enabled_;
}

unsigned StatGating::current_segment_for_cta(unsigned cta_id,
                                              unsigned long long cta_total_inst) {
  auto it = segments_by_cta_.find(cta_id);
  if (it == segments_by_cta_.end()) return 0;
  for (const auto& seg : it->second) {
    if (cta_total_inst >= seg.start_inst && cta_total_inst <= seg.end_inst)
      return seg.segment_id;
  }
  return it->second.size() - 1;
}

void StatGating::record_cycle(unsigned cta_id) {
  if (!enabled_) return;
  total_cycles_++;
}

void StatGating::record_event(unsigned cta_id, unsigned event_type,
                               unsigned long long val) {
  if (!enabled_) return;
  (void)cta_id;
  (void)event_type;
  (void)val;
}

void StatGating::print_stats(FILE* out) {
  if (!enabled_ || stats_.empty()) return;
  fprintf(out, "gpu_segment_count = %zu\n", stats_.size());
  for (auto& [seg_id, st] : stats_) {
    double cpi = st.instructions > 0
                     ? (double)st.cycles / st.instructions
                     : 0.0;
    fprintf(out, "gpu_segment_%u_cycles = %llu\n", seg_id, st.cycles);
    fprintf(out, "gpu_segment_%u_instructions = %llu\n", seg_id, st.instructions);
    fprintf(out, "gpu_segment_%u_cpi = %.4f\n", seg_id, cpi);
  }
  bool ok = validate_sum();
  fprintf(out, "gpu_segment_validate_sum = %s\n", ok ? "PASS" : "FAIL");
}

bool StatGating::validate_sum() const {
  if (!enabled_) return true;
  unsigned long long seg_sum = 0;
  for (auto& [_, st] : stats_)
    seg_sum += st.cycles;
  bool ok = seg_sum == total_cycles_;
  if (!ok)
    fprintf(stderr, "StatGating: segment sum (%llu) != total (%llu), diff=%lld\n",
            seg_sum, total_cycles_, (long long)(total_cycles_ - seg_sum));
  return ok;
}
