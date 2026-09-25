"""Run the production L0I destructor against interrupted stream-buffer waits."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling"


class InstructionCacheTeardownTests(unittest.TestCase):
    def test_freed_wait_request_and_owned_responses(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler required")
        source = (SOURCE / "first_level_instruction_cache.cc").read_text()
        destructor = source[
            source.index("first_level_instruction_cache::~"):
            source.index("void first_level_instruction_cache::initiate_stream_buffers")
        ]
        header = (SOURCE / "first_level_instruction_cache.h").read_text()
        status = header[header.index("struct status_element {"):
                        header.index("class first_level_instruction_cache :")]
        program = r"""
#include <cassert>
#include <map>
using new_addr_type = unsigned long long;
enum cache_request_status { HIT, MISS, RESERVATION_FAIL, IN_L0I_RESPONSE_QUEUE };
struct mem_fetch {
  static unsigned destroyed;
  ~mem_fetch() { ++destroyed; }
};
unsigned mem_fetch::destroyed = 0;
"""
        program += status
        program += r"""
class first_level_instruction_cache {
 public:
  std::map<unsigned, std::map<new_addr_type, status_element>>
      m_regular_access_status_without_IB_coalescing;
  int *m_stream_buffers = new int(0);
  ~first_level_instruction_cache();
  void clear_cache() { m_regular_access_status_without_IB_coalescing.clear(); }
};
"""
        program += destructor
        program += r"""
int main() {
  {
    first_level_instruction_cache cache;
    mem_fetch *request = new mem_fetch;
    cache.m_regular_access_status_without_IB_coalescing[0].emplace(
        0x1000, status_element(IN_L0I_RESPONSE_QUEUE, request, true,
                              status_element::STREAM_BUFFER_WAIT));
    // Subcore frees this request as soon as access returns a stream wait.
    delete request;
    assert(mem_fetch::destroyed == 1);
  }
  assert(mem_fetch::destroyed == 1);
  {
    first_level_instruction_cache cache;
    // fill_from_stream_buffer replaces an erased request with a live response.
    // The historical mf_erased flag remains true, but cache_status becomes HIT.
    cache.m_regular_access_status_without_IB_coalescing[0].emplace(
        0x2000, status_element(HIT, new mem_fetch, true,
                              status_element::STREAM_BUFFER_WAIT));
    cache.m_regular_access_status_without_IB_coalescing[1].emplace(
        0x3000, status_element(MISS, new mem_fetch, false,
                              status_element::NORMAL_ACCESS));
    cache.m_regular_access_status_without_IB_coalescing[2].emplace(
        0x4000, status_element(HIT, new mem_fetch, false,
                              status_element::NORMAL_ACCESS));
  }
  assert(mem_fetch::destroyed == 4);
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp) / "teardown.cc"
            binary = Path(tmp) / "teardown"
            code.write_text(program)
            subprocess.run(
                [compiler, "-std=c++14", "-fsanitize=address",
                 "-fno-omit-frame-pointer", "-g", str(code), "-o", str(binary)],
                check=True,
            )
            env = os.environ.copy()
            env["ASAN_OPTIONS"] = "detect_leaks=0"
            subprocess.run([str(binary)], check=True, env=env)


if __name__ == "__main__":
    unittest.main()

# ## Revision History
# 2026-09-22: Reproduce the observed forced-stop double free with the production
# destructor under ASan, and require live replacement responses to be reclaimed.
