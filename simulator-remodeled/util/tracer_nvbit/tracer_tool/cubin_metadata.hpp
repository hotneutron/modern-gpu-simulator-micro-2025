#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace cubin_metadata {

inline std::vector<unsigned char> read(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open cubin: " + path.string());
  std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input), {}};
  if (input.bad()) throw std::runtime_error("cannot read cubin: " + path.string());
  return bytes;
}

// Read ELF64 fields explicitly: no host alignment, byte order or elf.h dependency.
inline void bounds(const std::vector<unsigned char> &b, uint64_t off, uint64_t n) {
  if (off > b.size() || n > b.size() - off)
    throw std::runtime_error("truncated cubin ELF");
}

inline uint64_t field(const std::vector<unsigned char> &b, uint64_t off, unsigned n) {
  bounds(b, off, n);
  uint64_t value = 0;
  for (unsigned i = 0; i < n; ++i) value |= uint64_t(b[off + i]) << (8 * i);
  return value;
}

inline std::map<std::string, uint64_t> function_symbols(
    const std::vector<unsigned char> &b) {
  bounds(b, 0, 64);
  if (b[0] != 0x7f || b[1] != 'E' || b[2] != 'L' || b[3] != 'F' ||
      b[4] != 2 || b[5] != 1 || field(b, 18, 2) != 190)
    throw std::runtime_error("expected little-endian ELF64 CUDA cubin");
  const uint64_t sections = field(b, 40, 8);
  const uint64_t stride = field(b, 58, 2);
  uint64_t count = field(b, 60, 2);
  if (stride != 64 || sections == 0) throw std::runtime_error("invalid ELF sections");
  bounds(b, sections, stride);
  if (count == 0) count = field(b, sections + 32, 8);
  if (count > (b.size() - sections) / stride)
    throw std::runtime_error("truncated ELF sections");
  std::map<std::string, uint64_t> symbols;
  bool found_table = false;
  for (uint64_t section = 0; section < count; ++section) {
    const uint64_t sh = sections + section * stride;
    if (field(b, sh + 4, 4) != 2) continue;  // SHT_SYMTAB
    if (found_table) throw std::runtime_error("ambiguous cubin symbol tables");
    found_table = true;
    const uint64_t offset = field(b, sh + 24, 8);
    const uint64_t size = field(b, sh + 32, 8);
    const uint64_t link = field(b, sh + 40, 4);
    if (field(b, sh + 56, 8) != 24 || size % 24 != 0 || link >= count)
      throw std::runtime_error("invalid cubin symbol table");
    bounds(b, offset, size);
    const uint64_t strings_sh = sections + link * stride;
    if (field(b, strings_sh + 4, 4) != 3)
      throw std::runtime_error("invalid cubin string table");
    const uint64_t strings = field(b, strings_sh + 24, 8);
    const uint64_t strings_size = field(b, strings_sh + 32, 8);
    bounds(b, strings, strings_size);
    for (uint64_t i = 0; i < size / 24; ++i) {
      const uint64_t sym = offset + i * 24;
      const uint64_t code_section = field(b, sym + 6, 2);
      if ((b[sym + 4] & 15) != 2 || code_section == 0) continue; // STT_FUNC
      if (code_section >= count)
        throw std::runtime_error("invalid function section");
      const uint64_t name_offset = field(b, sym, 4);
      if (name_offset >= strings_size) throw std::runtime_error("invalid symbol name");
      const auto begin = b.begin() + strings + name_offset;
      const auto limit = b.begin() + strings + strings_size;
      const auto end = std::find(begin, limit, 0);
      if (end == limit) throw std::runtime_error("unterminated symbol name");
      const std::string name(begin, end);
      if (!symbols.emplace(name, i).second)
        throw std::runtime_error("ambiguous cubin function: " + name);
    }
  }
  if (!found_table) throw std::runtime_error("cubin has no symbol table");
  return symbols;
}

inline std::string function_indices(
    const std::filesystem::path &path,
    const std::unordered_set<std::string> &names) {
  if (names.empty()) throw std::runtime_error("empty RFU function selection");
  const auto symbols = function_symbols(read(path));
  std::vector<uint64_t> indices;
  for (const auto &name : names) {
    const auto found = symbols.find(name);
    if (found == symbols.end()) throw std::runtime_error("missing cubin function: " + name);
    indices.push_back(found->second);
  }
  std::sort(indices.begin(), indices.end());
  std::string result;
  for (auto index : indices) {
    if (!result.empty()) result += ",";
    result += std::to_string(index);
  }
  return result;
}

inline std::string shell_quote(const std::string &value) {
  std::string result = "'";
  for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c);
  return result + "'";
}

// The fingerprint only selects candidates. Full byte equality decides identity.
class CapturedCubins {
  std::map<std::pair<uint64_t, uint64_t>, std::vector<std::filesystem::path>> paths_;
 public:
  template <typename Dump>
  std::filesystem::path capture(const std::filesystem::path &path, Dump dump) {
    // NVBit 1.7.6 returns false even after a successful write. A fresh,
    // structurally valid ELF is the capture contract; never reuse stale output.
    if (std::filesystem::exists(path))
      throw std::runtime_error("cubin capture destination already exists");
    (void)dump(path.c_str());
    return retain(path);
  }

  std::filesystem::path retain(const std::filesystem::path &path) {
    const auto bytes = read(path);
    function_symbols(bytes);  // Reject incomplete/unsupported NVBit dumps.
    uint64_t hash = UINT64_C(14695981039346656037);
    for (auto byte : bytes) hash = (hash ^ byte) * UINT64_C(1099511628211);
    auto &candidates = paths_[{bytes.size(), hash}];
    for (const auto &candidate : candidates) {
      if (read(candidate) == bytes) {
        if (path != candidate) std::filesystem::remove(path);
        return candidate;
      }
    }
    candidates.push_back(path);
    return path;
  }
};

}  // namespace cubin_metadata

// ## Revision History
// 2026-09-20 00:51 UTC: validate fresh ELF output because pinned NVBit returns
// false on successful dumps; reject missing, malformed and stale captures.
// 2026-09-20: resolve exact CUDA ELF function indices for scoped RFU and retain
// loaded code objects with full-byte duplicate verification.
