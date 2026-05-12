#include "stat_gating.h"
#include <stdio.h>
#include <fstream>
#include <json/json.h>

StatGating::StatGating() : enabled_(false), total_cycles_(0) {}

StatGating::~StatGating() {}

void StatGating::reset() {
  enabled_ = false;
  total_cycles_ = 0;
  segments_by_cta_.clear();
  stats_.clear();
}

bool StatGating::load_segments(const std::string& json_path) {
  reset();
  std::ifstream f(json_path);
  if (!f.is_open()) {
    fprintf(stderr, "StatGating: cannot open %s\n", json_path.c_str());
    return false;
  }

  Json::Value root;
  f >> root;

  const Json::Value& segs = root["segments"];
  for (const auto& js : segs) {
    SegmentDef sd;
    sd.segment_id = js["segment_id"].asUInt();
    sd.start_inst = js["start_interval"].asUInt64() * 1024;
    sd.end_inst = (js["end_interval"].asUInt64() + 1) * 1024 - 1;
    sd.cta_x = 0;
    sd.cta_y = 0;
    sd.cta_z = 0;
    segments_by_cta_[0].push_back(sd);
  }

  enabled_ = true;
  total_cycles_ = 0;
  fprintf(stderr, "StatGating: loaded %zu segments\n", segments_by_cta_[0].size());
  return true;
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
  if (!enabled_) return;
  fprintf(out, "gpu_segment_count = %zu\n", stats_.size());
  for (auto& [seg_id, st] : stats_) {
    double cpi = st.instructions > 0
                     ? (double)st.cycles / st.instructions
                     : 0.0;
    fprintf(out, "gpu_segment_%u_cycles = %llu\n", seg_id, st.cycles);
    fprintf(out, "gpu_segment_%u_instructions = %llu\n", seg_id, st.instructions);
    fprintf(out, "gpu_segment_%u_cpi = %.4f\n", seg_id, cpi);
    fprintf(out, "gpu_segment_%u_l1_hit_rate = %.3f\n", seg_id,
            (st.l1_hits + st.l1_misses) > 0
                ? (double)st.l1_hits / (st.l1_hits + st.l1_misses)
                : 0.0);
    fprintf(out, "gpu_segment_%u_l2_hit_rate = %.3f\n", seg_id,
            (st.l2_hits + st.l2_misses) > 0
                ? (double)st.l2_hits / (st.l2_hits + st.l2_misses)
                : 0.0);
  }
}

void StatGating::emit_json(const std::string& out_path) {
  Json::Value root;
  root["total_cycles"] = (Json::Value::UInt64)total_cycles_;

  for (auto& [seg_id, st] : stats_) {
    Json::Value js;
    js["segment_id"] = seg_id;
    js["cycles"] = (Json::Value::UInt64)st.cycles;
    js["instructions"] = (Json::Value::UInt64)st.instructions;
    js["cpi"] = st.instructions > 0
                    ? (double)st.cycles / st.instructions
                    : 0.0;
    js["l1_hit_rate"] = (st.l1_hits + st.l1_misses) > 0
                            ? (double)st.l1_hits / (st.l1_hits + st.l1_misses)
                            : 0.0;
    js["l2_hit_rate"] = (st.l2_hits + st.l2_misses) > 0
                            ? (double)st.l2_hits / (st.l2_hits + st.l2_misses)
                            : 0.0;
    root["segments"].append(js);
  }

  std::ofstream f(out_path);
  f << root;
}

bool StatGating::validate_sum() const {
  if (!enabled_) return true;
  unsigned long long seg_sum = 0;
  for (auto& [_, st] : stats_)
    seg_sum += st.cycles;
  bool ok = seg_sum == total_cycles_;
  if (!ok)
    fprintf(stderr, "StatGating: segment sum (%llu) != total (%llu)\n",
            seg_sum, total_cycles_);
  return ok;
}
