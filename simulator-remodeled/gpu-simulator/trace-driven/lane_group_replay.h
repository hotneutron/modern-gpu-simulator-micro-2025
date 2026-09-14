#ifndef LANE_GROUP_REPLAY_H
#define LANE_GROUP_REPLAY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

// Two disjoint streams terminate at one *recorded* full-mask instruction.
// The boundary is not a hardware barrier and changes no barrier state.
template <typename Instruction>
class LaneGroupReplay {
 public:
  struct Item {
    uint64_t source_index;
    Instruction instruction;
    bool reorderable_poll = false;
  };

  void begin(std::array<std::vector<Item>, 2> streams,
             uint32_t first_mask, uint32_t second_mask, uint32_t live_mask,
             uint64_t merge_index) {
    require(!active_, "lane-group replay already active");
    require(first_mask && second_mask && !(first_mask & second_mask) &&
                (first_mask | second_mask) == live_mask,
            "invalid lane-group mask partition");
    for (const auto &stream : streams) {
      uint64_t previous = 0;
      bool first = true;
      for (const auto &item : stream) {
        require(item.source_index < merge_index &&
                    (first || item.source_index > previous),
                "lane-group source order is not increasing");
        previous = item.source_index;
        first = false;
      }
    }
    // A source occurrence must not be owned by both groups.
    size_t a = 0, b = 0;
    while (a < streams[0].size() && b < streams[1].size()) {
      const auto x = streams[0][a].source_index;
      const auto y = streams[1][b].source_index;
      require(x != y, "duplicate lane-group source occurrence");
      if (x < y) ++a;
      else ++b;
    }
    streams_ = std::move(streams);
    for (unsigned g = 0; g != 2; ++g) {
      auto &next = next_semantic_[g];
      next.assign(streams_[g].size() + 1, streams_[g].size());
      for (size_t i = streams_[g].size(); i != 0; --i) {
        next[i - 1] = streams_[g][i - 1].reorderable_poll ? next[i] : i - 1;
      }
    }
    issued_ = {{0, 0}};
    waiting_ = {{false, false}};
    masks_ = {{first_mask, second_mask}};
    merge_index_ = merge_index;
    selected_ = -1;
    reserved_ = 0;
    active_ = true;
    ++epoch_;
  }

  bool active() const { return active_; }
  uint64_t epoch() const { return epoch_; }
  int selected() const { return selected_; }
  uint32_t mask(unsigned group) const { return masks_.at(group); }
  uint64_t merge_index() const { return merge_index_; }
  size_t remaining(unsigned group) const {
    return streams_.at(group).size() - issued_.at(group);
  }
  bool waiting(unsigned group) const { return waiting_.at(group); }
  size_t issued(unsigned group) const { return issued_.at(group); }
  size_t reserved() const { return reserved_; }
  const Item &item(unsigned group, size_t index) const {
    return streams_.at(group).at(index);
  }

  bool runnable(unsigned group) const {
    if (!active_ || waiting_[group] || !remaining(group)) return false;
    const auto &current = head(group);
    if (current.reorderable_poll) return true;
    const unsigned other = 1 - group;
    const size_t semantic = next_semantic_[other][issued_[other]];
    return semantic == streams_[other].size() ||
           streams_[other][semantic].source_index > current.source_index;
  }

  void set_waiting(unsigned group, bool waiting) {
    require(active_, "wait without lane-group replay");
    waiting_.at(group) = waiting;
  }

  int next_group() const {
    int next = -1;
    for (unsigned g = 0; g != 2; ++g) {
      if (!runnable(g)) continue;
      if (next < 0 || head(g).source_index < head(next).source_index) next = g;
    }
    return next;
  }

  bool at_merge() const {
    return active_ && remaining(0) == 0 && remaining(1) == 0 &&
           !waiting_[0] && !waiting_[1];
  }

  // The caller rolls back IBuffer reservations before changing streams.
  void select(int group) {
    require(active_ && group >= -1 && group < 2, "invalid lane-group selection");
    require(reserved_ == 0, "lane-group switch with outstanding reservations");
    require(group < 0 || runnable(group),
            "selected lane group is not runnable");
    if (group != selected_) ++epoch_;
    selected_ = group;
  }

  const Item &head(unsigned group) const {
    require(remaining(group) != 0, "lane-group stream is exhausted");
    return streams_.at(group).at(issued_.at(group));
  }

  bool reserve(size_t &index, uint64_t &epoch) {
    if (selected_ < 0 || waiting_[selected_] ||
        reserved_ >= remaining(selected_)) return false;
    index = issued_[selected_] + reserved_++;
    epoch = epoch_;
    return true;
  }

  const Item &reserved_item(size_t index, uint64_t epoch) const {
    require(active_ && selected_ >= 0 && epoch == epoch_,
            "stale lane-group reservation epoch");
    require(index >= issued_[selected_] &&
                index < issued_[selected_] + reserved_,
            "lane-group reservation outside outstanding window");
    return streams_[selected_].at(index);
  }

  Instruction &reserved_instruction(size_t index, uint64_t epoch) {
    reserved_item(index, epoch);
    return streams_[selected_].at(index).instruction;
  }

  void unreserve() {
    require(reserved_ != 0, "lane-group reservation underflow");
    --reserved_;
  }

  void commit(size_t index) {
    require(active_ && selected_ >= 0 && reserved_ &&
                index == issued_[selected_],
            "lane-group commit is not the reserved head");
    ++issued_[selected_];
    --reserved_;
  }

  // Only a caller-proven pure polling occurrence may be folded.
  void fold_head(unsigned group) {
    require(active_ && reserved_ == 0 && remaining(group),
            "lane-group fold with outstanding reservation or empty stream");
    ++issued_.at(group);
  }

  void finish() {
    require(at_merge() && reserved_ == 0,
            "lane-group replay ended before both groups were ready");
    streams_ = {};
    active_ = false;
    selected_ = -1;
    ++epoch_;
  }

 private:
  static void require(bool condition, const char *message) {
    if (!condition) throw std::logic_error(message);
  }
  std::array<std::vector<Item>, 2> streams_;
  std::array<std::vector<size_t>, 2> next_semantic_;
  std::array<size_t, 2> issued_{{0, 0}};
  std::array<bool, 2> waiting_{{false, false}};
  std::array<uint32_t, 2> masks_{{0, 0}};
  uint64_t merge_index_ = 0;
  uint64_t epoch_ = 0;
  size_t reserved_ = 0;
  int selected_ = -1;
  bool active_ = false;
};

#endif
