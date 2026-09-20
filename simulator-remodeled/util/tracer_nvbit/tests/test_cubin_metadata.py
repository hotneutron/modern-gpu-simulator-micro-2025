"""Exercise real ELF symbol selection and loaded-object retention in native C++."""
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

TRACER = Path(__file__).resolve().parents[1] / "tracer_tool"


def cubin():
    # Deliberately include a substring match, undefined function and data symbol.
    names = b"\0target\0target_extra\0unloaded\0data\0"
    data = bytearray(64 + 4 * 64)
    data[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<H", data, 18, 190)
    struct.pack_into("<Q", data, 40, 64)
    struct.pack_into("<HH", data, 58, 64, 4)
    entries = [(0, 0, 0), (1, 2, 3), (8, 2, 3), (21, 2, 0), (30, 1, 3)]
    symbols = b"".join(struct.pack("<IBBHQQ", n, t, 0, s, 0, 16)
                       for n, t, s in entries)
    struct.pack_into("<IIQQQQIIQQ", data, 128, 0, 2, 0, 0, len(data),
                     len(symbols), 2, 0, 8, 24)
    data.extend(symbols)
    struct.pack_into("<IIQQQQIIQQ", data, 192, 0, 3, 0, 0, len(data),
                     len(names), 0, 0, 1, 0)
    data.extend(names)
    return bytes(data)


class CubinMetadataTests(unittest.TestCase):
    def test_exact_symbols_bounds_and_byte_deduplication(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler required")
        program = r'''
#include "cubin_metadata.hpp"
#include <cassert>
int main(int argc, char **argv) {
  assert(argc == 2);
  namespace c = cubin_metadata;
  const std::filesystem::path root = argv[1], input = root / "input.cubin";
  assert(c::function_indices(input, {"target"}) == "1");
  assert(c::function_indices(input, {"target_extra", "target"}) == "1,2");
  for (const auto &names : std::vector<std::unordered_set<std::string>>{
      {}, {"unloaded"}, {"data"}, {"TARGET"}, {"targ"}, {"absent"}}) {
    bool rejected = false;
    try { c::function_indices(input, names); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
  }
  auto bytes = c::read(input);
  for (size_t length = 0; length < bytes.size(); ++length) {
    bool rejected = false;
    try { c::function_symbols({bytes.begin(), bytes.begin() + length}); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
  }
  auto corrupt = bytes;
  for (int i = 0; i < 8; ++i) corrupt[40 + i] = 255;
  bool rejected = false;
  try { c::function_symbols(corrupt); }
  catch (const std::runtime_error &) { rejected = true; }
  assert(rejected);
  c::CapturedCubins capture;
  assert(capture.retain(input) == input);
  const auto duplicate = root / "duplicate.cubin";
  std::filesystem::copy_file(input, duplicate);
  assert(capture.retain(duplicate) == input);
  assert(!std::filesystem::exists(duplicate));
  const auto different = root / "different.cubin";
  bytes[8] = 1;  // Same size and function symbols, distinct code-object bytes.
  std::ofstream out(different, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  out.close();
  assert(capture.retain(different) == different);
  assert(std::filesystem::exists(input) && std::filesystem::exists(different));
  // Pinned NVBit returns false after writing a valid ELF. Its return value
  // cannot authorize stale/missing/empty output, nor reject a valid capture.
  const auto fresh = root / "fresh.cubin";
  bool called = false;
  assert(capture.capture(fresh, [&](const char *path) {
    called = true;
    std::filesystem::copy_file(input, path);
    return false;
  }) == input);
  assert(called && !std::filesystem::exists(fresh));
  called = false;
  rejected = false;
  try { capture.capture(input, [&](const char *) { called = true; return true; }); }
  catch (const std::runtime_error &) { rejected = true; }
  assert(rejected && !called);
  for (bool api_result : {false, true}) {
    rejected = false;
    try { capture.capture(fresh, [&](const char *) { return api_result; }); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
  }
  rejected = false;
  try { capture.capture(fresh, [](const char *path) {
    std::ofstream output(path);
    output << "incomplete ELF";
    return false;
  }); }
  catch (const std::runtime_error &) { rejected = true; }
  assert(rejected);
  assert(c::shell_quote("x'y $z") == "'x'\\''y $z'");
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.cubin").write_bytes(cubin())
            binary = str(root / "metadata")
            built = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-I", str(TRACER), "-x", "c++", "-", "-o", binary],
                input=program, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([binary, directory], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()

# ## Revision History
# 2026-09-20 00:51 UTC: exercise the real capture path with NVBit's observed
# false return; reject stale, missing and malformed output independently.
# 2026-09-20: cover exact scoped RFU indices, malformed/truncated ELF rejection,
# and content-verified loaded cubin deduplication without CUDA dependencies.
