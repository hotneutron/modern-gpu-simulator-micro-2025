#ifndef STAT_GATING_H_
#define STAT_GATING_H_

#include <json/json.h>
#include <map>
#include <string>
#include <vector>

#define N_STALL_REASONS 16

struct SegmentDef {
  unsigned segment_id;
  unsigned long long start_inst;
  unsigned long long end_inst;  // inclusive
  unsigned cta_x, cta_y, cta_z;
};

struct PerSegmentStats {
  unsigned segment_id;
  unsigned long long cycles;
  unsigned long long instructions;
  unsigned long long l1_hits, l1_misses;
  unsigned long long l2_hits, l2_misses;
  unsigned long long dram_reads, dram_writes;
  unsigned long long stall_cycles[N_STALL_REASONS];
  double active_warps_avg;
  unsigned samples;

  PerSegmentStats() : segment_id(0), cycles(0), instructions(0),
      l1_hits(0), l1_misses(0), l2_hits(0), l2_misses(0),
      dram_reads(0), dram_writes(0), active_warps_avg(0.0), samples(0) {
    for (int i = 0; i < N_STALL_REASONS; i++)
      stall_cycles[i] = 0;
  }
};

class StatGating {
 public:
  StatGating();
  ~StatGating();

  bool load_segments(const std::string& json_path);
  unsigned current_segment_for_cta(unsigned cta_id,
                                   unsigned long long cta_total_inst);
  void record_cycle(unsigned cta_id);
  void record_event(unsigned cta_id, unsigned event_type, unsigned long long val);
  void print_stats(FILE* out);
  void emit_json(const std::string& out_path);
  bool validate_sum() const;
  void reset();

 private:
  bool enabled_;
  unsigned long long total_cycles_;
  std::map<unsigned, std::vector<SegmentDef>> segments_by_cta_;
  std::map<unsigned, PerSegmentStats> stats_;
};

#endif  // STAT_GATING_H_
