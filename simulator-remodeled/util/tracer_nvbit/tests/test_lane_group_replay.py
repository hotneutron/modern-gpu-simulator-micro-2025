import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class LaneGroupReplayTest(unittest.TestCase):
    def test_independent_completion_reservations_and_conservation(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler is not installed")
        harness = r'''
#include <cassert>
#include <functional>
#include "gpu-simulator/trace-driven/lane_group_replay.h"
using Replay = LaneGroupReplay<int>;
void rejected(std::function<void()> operation) {
  bool caught = false;
  try { operation(); } catch (const std::logic_error &) { caught = true; }
  assert(caught);
}
void issue(Replay &r, unsigned group, int payload) {
  r.select(group);
  size_t index;
  uint64_t epoch;
  assert(r.reserve(index, epoch));
  assert(r.reserved_item(index, epoch).instruction == payload);
  r.commit(index);
}
int main() {
  for (bool poll_first : {false, true}) {
    Replay r;
    r.begin({{{{10, 101}, {30, 102}}, {{20, 201, true}, {40, 202, true}}}},
            1, 14, 15, 50);
    assert(r.next_group() == 0);
    issue(r, 0, 101);
    r.set_waiting(0, true);
    assert(r.next_group() == 1);
    issue(r, 1, 201);
    r.set_waiting(1, true);
    r.select(-1);
    assert(r.next_group() == -1 && !r.at_merge());
    unsigned first = poll_first ? 1 : 0, second = 1 - first;
    r.set_waiting(first, false);
    assert(r.waiting(second));
    assert(r.next_group() == static_cast<int>(first));
    issue(r, first, first == 0 ? 102 : 202);
    assert(r.remaining(first) == 0 && !r.at_merge());
    assert(r.next_group() == -1);
    rejected([&] { r.finish(); });
    r.set_waiting(second, false);
    issue(r, second, second == 0 ? 102 : 202);
    assert(r.at_merge());
    r.finish();
    assert(!r.active());
  }
  Replay r;
  rejected([&] { r.begin({{{{1, 1}}, {{1, 2}}}}, 1, 14, 15, 10); });
  rejected([&] { r.begin({{{{1, 1}}, {{2, 2}}}}, 1, 15, 15, 10); });
  rejected([&] { r.begin({{{{5, 1}, {4, 2}}, {{2, 2}}}}, 1, 14, 15, 10); });
  r.begin({{{{1, 1}, {3, 3}}, {{2, 2}}}}, 1, 14, 15, 10);
  r.select(0);
  size_t a, b;
  uint64_t epoch;
  assert(r.reserve(a, epoch) && r.reserve(b, epoch));
  rejected([&] { r.select(1); });
  rejected([&] { r.commit(b); });
  r.unreserve();
  rejected([&] { r.reserved_item(b, epoch); });
  r.commit(a);
  r.select(1);
  rejected([&] { r.reserved_item(a, epoch); });
  issue(r, 1, 2);
  r.select(0);
  r.fold_head(0);
  assert(r.at_merge());
  r.finish();
  r.begin({{{{10, 10}, {30, 30}}, {{20, 20, true}, {40, 40}}}}, 1, 14, 15, 50);
  r.set_waiting(0, true);
  issue(r, 1, 20);
  // A semantic tail cannot pass another group's earlier semantic work.
  assert(r.next_group() == -1 && !r.runnable(1));
  rejected([&] { r.select(1); });
  r.set_waiting(0, false);
  issue(r, 0, 10);
  issue(r, 0, 30);
  issue(r, 1, 40);
  r.finish();
  r.begin({{{{1, 1}}, {{2, 2, true}, {3, 3, true}, {4, 4, true}}}},
          1, 14, 15, 5);
  issue(r, 0, 1);
  issue(r, 1, 2);
  issue(r, 1, 3);
  issue(r, 1, 4);
  r.finish();
  // Reuse after a larger stream: its old semantic-index sentinel must not leak.
  r.begin({{{{1, 1}}, {{2, 2, true}}}}, 1, 14, 15, 3);
  assert(r.runnable(0) && r.next_group() == 0);
  issue(r, 0, 1);
  issue(r, 1, 2);
  r.finish();
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / "test-lane-group")
            compiled = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=address,undefined",
                 "-I", str(ROOT), "-x", "c++", "-", "-o", binary],
                input=harness, text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
