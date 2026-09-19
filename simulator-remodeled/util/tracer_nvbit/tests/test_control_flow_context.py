"""Execute production call-context methods, including divergent return masks."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class CallContextTests(unittest.TestCase):
    def test_nested_partial_predicated_and_invalid_returns(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler required")
        source = (ROOT / "gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h").read_text()
        methods = source[source.index("  void push_function_call("):
                         source.index("  bool functional_done() const;")]
        program = r'''
#include <bitset>
#include <cassert>
#include <stack>
#include <string>
using active_mask_t = std::bitset<32>;
struct function_call_entry_info {
  unsigned unique_function_id;
  active_mask_t active_mask;
};
struct Warp {
  std::stack<function_call_entry_info> m_function_call_stack;
''' + methods + r'''
};
int main(int argc, char **argv) {
  Warp w;
  w.push_function_call(1, 0xff);
  // Predicated-off CALL must not create a frame.
  w.push_function_call(1, 0);
  assert(w.get_function_call_depth() == 1);
  w.push_function_call(1, 0x0f);  // same-function CALL.REL.NOINC
  w.push_function_call(2, 0x03);  // nested traced callee
  w.pop_function_call(0x01);     // one lane returns first
  assert(w.get_current_unique_function_id_call() == 2);
  if(argc > 1 && std::string(argv[1]) == "duplicate") {
    w.pop_function_call(0x01);   // used to silently reactivate the lane
    return 0;
  }
  if(argc > 1 && std::string(argv[1]) == "foreign") {
    w.pop_function_call(0x80);
    return 0;
  }
  w.pop_function_call(0x02);
  assert(w.get_function_call_depth() == 2);
  assert(w.get_current_unique_function_id_call() == 1);
  w.pop_function_call(0x05);
  w.pop_function_call(0);        // predicated-off RET
  assert(w.can_pop_non_root_function_call());
  w.pop_function_call(0x0a);
  assert(!w.can_pop_non_root_function_call());
  assert(w.get_function_call_depth() == 1);
  w.pop_function_call(0xff);    // root exit
  assert(!w.has_function_call_context());
}
'''
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / "call-context")
            built = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-x", "c++", "-", "-o", binary], input=program,
                text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            passed = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            for mode in ("duplicate", "foreign"):
                result = subprocess.run([binary, mode], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, mode)


if __name__ == "__main__":
    unittest.main()

# ## Revision History
# 2026-09-19: regress same-function and nested calls, split lane returns,
# predicated-off events and invalid return masks using production methods.
