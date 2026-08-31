// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <algorithm>
#include <atomic>
#include <assert.h>
#include <bitset>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <initializer_list>
#include <inttypes.h>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <regex>

// MOD. Begin. Enhanced Tracer
#include <execinfo.h> /* backtrace, backtrace_symbols_fd */
#include <unistd.h> /* STDOUT_FILENO */
#include <filesystem>
#include <fstream>

#include "../../traces_enhanced/src/traced_execution.h"
#include "../../traces_enhanced/src/string_utilities.h"
#include "../../../gpu-simulator/ISA_Def/trace_opcode.h"
// MOD. End. Enhanced Tracer

#include "../../traces_enhanced/pb_trace/include/trace.pb.h"
#include "../../traces_enhanced/pb_trace/include/gpu_device.pb.h"
#include "../../traces_enhanced/pb_trace/include/cuda_stream.pb.h"
#include "../../traces_enhanced/pb_trace/include/kernel.pb.h"
#include "../../traces_enhanced/pb_trace/include/threadblock.pb.h"
#include "../../traces_enhanced/pb_trace/include/warp.pb.h"
#include "../../traces_enhanced/pb_trace/include/instruction.pb.h"
#include "../../traces_enhanced/pb_trace/include/address.pb.h"
#include "../../traces_enhanced/pb_trace/include/dim3d.pb.h"

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

#define TRACER_VERSION 4

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)

int num_devices = 0;

enum class RecvThreadState {
  WORKING,
  STOP,
  FINISHED,
};

struct CTXstate {
  /* context id */
  int id = -1;

  /* Channel used to communicate from GPU to CPU receiving thread */
  ChannelDev* channel_dev = nullptr;
  ChannelHost channel_host;

  // After initialization, set it to WORKING to make recv thread get data,
  // parent thread sets it to STOP to make recv thread stop working.
  // recv thread sets it to FINISHED when it cleans up.
  // parent thread should wait until the state becomes FINISHED to clean up.
  volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;

  /* Published by the receiver when it consumes the end-of-kernel token. */
  std::atomic<uint64_t> received_records_per_kernel{0};
  std::atomic<uint64_t> sparse_retained_instructions_per_kernel{0};
  std::atomic<uint64_t> sparse_reconstructed_instructions_per_kernel{0};

  /* Stable registry pointer used only when MBARRIER_CONTRACT_CAPTURE=1. */
  mbarrier_contract_registry_t *mbarrier_contract_registry = nullptr;
};

/* lock */
pthread_mutex_t mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;
bool is_first_init_context_call = true;
std::map<CUcontext, bool> recv_thread_receiving;
bool *stop_report;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int intermediate_extra_files_persistance = 0;
int enable_compress = 1;
int print_core_id = 0;
int exclude_pred_off = 1;
int active_from_start = 1;
/* used to select region of interest when active from start is 0 */
bool active_region = true;
/* Probe-only mode: instrument SASS EXIT and record one final clock64 timestamp
 * per CTA-local warp. Full instruction tracing stays unchanged when disabled. */
int warp_lifetime_only = 0;
/* Resolved LIGHTER_TRACER feature state. */
int compact_wire_enabled = 0;
int prune_operands_enabled = 0;
int lighter_tracer_host_count = 0;
int lighter_tracer_sparse_reconstruct = 0;
std::string lighter_tracer_canonical = "off";
/* Sidecar mode: keep the complete dynamic trace path enabled and additionally
 * timestamp SASS EXIT. This measures the lifetime perturbation of full tracing
 * itself, using the same capture/output buffer as WARP_LIFETIME_ONLY. */
int warp_lifetime_sidecar = 0;
warp_lifetime_capture_t *warp_lifetime_capture = nullptr;
bool warp_lifetime_capture_active = false;
bool warp_lifetime_output_initialized = false;
unsigned int warp_lifetime_grid_x = 0;
unsigned int warp_lifetime_grid_y = 0;
unsigned int warp_lifetime_grid_z = 0;
unsigned int warp_lifetime_warps_per_cta = 0;
int warp_lifetime_kernel_id = 0;
std::string warp_lifetime_kernel_name;
/* Capture-only B0-E feasibility instrumentation. Default-off and isolated
 * from the instruction channel to avoid device-to-host backpressure. */
int mbarrier_contract_capture_enabled = 0;
int mbarrier_contract_barrier_slots_per_cta = 64;
int mbarrier_contract_records_per_cta = 8192;
int mbarrier_contract_max_records = 4194304;
uint64_t next_mbarrier_contract_launch_uid = 1;
std::unordered_set<CUfunction> mbarrier_contract_capture_launches;

/* Should we terminate the program once we are done tracing? */
int terminate_after_limit_number_of_kernels_reached = 0;
int user_defined_folders = 0;

static std::string trim_config_token(const std::string &value) {
  size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin]))) {
    begin++;
  }
  size_t end = value.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    end--;
  }
  return value.substr(begin, end - begin);
}

static int parse_strict_integer_config(
    const char *name, int default_value, int minimum, int maximum) {
  const char *raw = getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  if (*raw == '\0') {
    fprintf(stderr, "%s must not be empty\n", name);
    abort();
  }
  char *end = nullptr;
  const long parsed = strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed < minimum ||
      parsed > maximum) {
    fprintf(stderr, "%s must be an integer in [%d,%d], got '%s'\n",
            name, minimum, maximum, raw);
    abort();
  }
  return static_cast<int>(parsed);
}

[[noreturn]] static void lighter_tracer_config_error(
    const std::string &message) {
  fprintf(stderr, "[LIGHTER-TRACER] %s\n", message.c_str());
  abort();
}

static void validate_lighter_tracer_dependencies(
    const std::unordered_set<std::string> &features) {
  auto require = [&](const char *feature,
                     std::initializer_list<const char *> dependencies) {
    if (features.count(feature) == 0) {
      return;
    }
    std::vector<std::string> missing;
    for (const char *dependency : dependencies) {
      if (features.count(dependency) == 0) {
        missing.emplace_back(dependency);
      }
    }
    if (!missing.empty()) {
      std::ostringstream message;
      message << "feature '" << feature << "' requires: ";
      for (size_t i = 0; i < missing.size(); i++) {
        if (i != 0) {
          message << ",";
        }
        message << missing[i];
      }
      lighter_tracer_config_error(message.str());
    }
  };

  require("prune_operands", {"compact_wire"});
  require("host_count", {"compact_wire"});
  require("metadata_record", {"compact_wire"});
  require("sparse_reconstruct", {"compact_wire", "prune_operands"});
  require("warp_batch_1k", {"compact_wire"});
  require("warp_batch_2k", {"compact_wire"});
  require("warp_batch_4k", {"compact_wire"});
  require("spin_fusion",
          {"compact_wire", "prune_operands", "sparse_reconstruct"});

  int batch_features = 0;
  batch_features += features.count("warp_batch_1k") != 0;
  batch_features += features.count("warp_batch_2k") != 0;
  batch_features += features.count("warp_batch_4k") != 0;
  if (batch_features > 1) {
    lighter_tracer_config_error(
        "warp_batch_1k/2k/4k features are mutually exclusive");
  }
}

static void resolve_lighter_tracer_config() {
  static const std::vector<std::string> feature_order = {
      "compact_wire",      "prune_operands", "host_count",
      "metadata_record",   "sparse_reconstruct",
      "warp_batch_1k",     "warp_batch_2k",  "warp_batch_4k",
      "spin_fusion",
  };
  static const std::unordered_set<std::string> implemented_features = {
      "compact_wire",
      "prune_operands",
      "host_count",
      "sparse_reconstruct",
  };
  // sparse_reconstruct remains explicit-only until opcode ordering is fixed.
  static const std::unordered_set<std::string> production_features = {
      "compact_wire",
      "prune_operands",
  };

  const char *lighter_value = getenv("LIGHTER_TRACER");
  std::unordered_set<std::string> features;
  std::string requested = lighter_value == nullptr ? "all" : lighter_value;
  const char *source = lighter_value == nullptr ? "default" : "LIGHTER_TRACER";
  std::string raw =
      lighter_value == nullptr ? "all" : trim_config_token(lighter_value);
  if (raw.empty()) {
    lighter_tracer_config_error(
        "LIGHTER_TRACER must not be empty; use 'off'");
  }
  if (raw == "all") {
    features.insert(production_features.begin(), production_features.end());
  } else if (raw != "off") {
    size_t begin = 0;
    while (begin <= raw.size()) {
      size_t comma = raw.find(',', begin);
      size_t end = comma == std::string::npos ? raw.size() : comma;
      std::string token =
          trim_config_token(raw.substr(begin, end - begin));
      if (token.empty()) {
        lighter_tracer_config_error(
            "LIGHTER_TRACER contains an empty feature token");
      }
      if (token == "off" || token == "all") {
        lighter_tracer_config_error("feature '" + token +
                                    "' cannot be combined");
      }
      if (std::find(feature_order.begin(), feature_order.end(), token) ==
          feature_order.end()) {
        lighter_tracer_config_error("unknown feature '" + token + "'");
      }
      if (!features.insert(token).second) {
        lighter_tracer_config_error("duplicate feature '" + token + "'");
      }
      if (comma == std::string::npos) {
        break;
      }
      begin = comma + 1;
    }
  }

  validate_lighter_tracer_dependencies(features);
  std::vector<std::string> unimplemented;
  for (const auto &feature : features) {
    if (implemented_features.count(feature) == 0) {
      unimplemented.push_back(feature);
    }
  }
  if (!unimplemented.empty()) {
    std::sort(unimplemented.begin(), unimplemented.end());
    std::ostringstream message;
    message << "features are recognized but not implemented: ";
    for (size_t i = 0; i < unimplemented.size(); i++) {
      if (i != 0) {
        message << ",";
      }
      message << unimplemented[i];
    }
    lighter_tracer_config_error(message.str());
  }

  std::ostringstream canonical;
  bool first_feature = true;
  for (const auto &feature : feature_order) {
    if (features.count(feature) == 0) {
      continue;
    }
    if (!first_feature) {
      canonical << ",";
    }
    canonical << feature;
    first_feature = false;
  }
  lighter_tracer_canonical =
      features.empty() ? "off" : canonical.str();
  compact_wire_enabled = features.count("compact_wire") != 0;
  prune_operands_enabled = features.count("prune_operands") != 0;
  lighter_tracer_host_count = features.count("host_count") != 0;
  lighter_tracer_sparse_reconstruct =
      features.count("sparse_reconstruct") != 0;

  printf("[LIGHTER-TRACER] requested=%s canonical=%s source=%s "
         "host_count=%d sparse_reconstruct=%d\n",
         requested.c_str(), lighter_tracer_canonical.c_str(), source,
         lighter_tracer_host_count, lighter_tracer_sparse_reconstruct);
}

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

std::string cwd = getcwd(NULL,0);
std::string traces_location = cwd + "/traces/";
std::string stats_location = cwd + "/traces/stats.csv";

static std::string make_abs_path(const std::string &path) {
  if (!path.empty() && path[0] == '/') {
    return path;
  }
  return cwd + "/" + path;
}

/* kernel instruction counter, updated by the GPU */
int dynamic_kernel_limit_start =
    0;                                 // 0 means start from the begging kernel
int dynamic_kernel_limit_end = 0; // 0 means no limit

enum address_format { list_all = 0, base_stride = 1, base_delta = 2 };

int binary_version;
std::vector<int> kernel_id;
std::vector<int> current_stream_id;

// MOD. Begin. Enhanced Tracer
int next_candidate_unique_function_id = 0;
std::set<int> opcodes_id_ldgsts;
int threshold_unique_kernel_checking;
std::string variant_delimiter_str = "___";

// =============================================================================
// Incremental Flush Feature - prevents memory explosion for large kernels
// =============================================================================
// When a threadblock accumulates more than FLUSH_THRESHOLD instructions,
// it is immediately written to disk to prevent memory exhaustion.
// Enable via: INCREMENTAL_FLUSH_THRESHOLD=50000 (0 = disabled)
int incremental_flush_threshold = 0;  // 0 means disabled
std::unordered_map<std::string, uint64_t> tb_instruction_count;  // per-TB instruction counter
std::unordered_map<std::string, int> tb_flush_part_count;  // per-TB part file counter
uint64_t total_incremental_flushes = 0;  // stats counter
uint64_t total_instructions_flushed = 0;  // stats counter
int enable_tma_desc = 0;
int aux_htod_dump_max_bytes = 4096;
/* Phase 0b (TMA_BASE_ADDR.md §2.13): param-base capture for host-side descriptor
 * deref. tma_param_base_cbank_index is the c[0x0][K] index the entry ULDC.64 loads
 * param_base from (0x198 in FA3 fwd; auto-detected by extract_tma_descriptor_offsets.py
 * and passed via TMA_PARAM_BASE_CBANK_INDEX). tma_param_base_region_bytes is how many
 * bytes of the param block to cuMemcpyDtoH so the offline harness can slice each 128B
 * descriptor at param_base + tensormap_offset. */
int tma_param_base_cbank_index = 0x198;
int tma_param_base_region_bytes = 8192;
tma_param_base_capture_t *tma_param_base_capture = nullptr;
/* SPIKE 6 device fact-check (TMA_BASE_ADDR.md §2.18): read 128B at the descriptor VA
 * of executed UTMALDG/UTMASTG and record qword0 (expected == real base). Gated by
 * TMA_DESC_FACTCHECK; dumped to tma_desc_factcheck.csv at kernel exit. */
int tma_desc_factcheck_enable = 0;
int tma_desc_factcheck_read = 0;
tma_desc_factcheck_t *tma_desc_factcheck = nullptr;
std::unordered_set<int> tma_desc_factcheck_header_written;
std::unordered_set<int> tma_param_base_deref_header_written;
std::unordered_set<int> tma_desc_runtime_debug_header_written;
std::unordered_set<int> tma_desc_producer_debug_header_written;
std::unordered_set<int> tma_memcpy_dump_header_written;
std::unordered_set<int> tensor_map_encode_dump_header_written;
/* Path A (TMA_BASE_ADDR.md §2.23): host-side launch-arg dump. At cuLaunchKernel we walk
 * p->kernelParams (a host void** of per-argument pointers) and record, per argument, its
 * byte offset within the packed param block and the first 16 qwords read from that host
 * pointer. A by-value __grid_constant__ CUtensorMap argument yields its 128B descriptor
 * (qword0 = real base) directly; a pointer argument yields the pointer value (followed
 * offline). Host memory only — no device load, so this cannot fault like the device
 * fact-check did. Joined offline to (uid,pc) via the SASS static tensormap_offset. */
std::unordered_set<int> tma_launch_param_dump_header_written;
std::unordered_set<int> tma_consumer_opcode_ids;
std::map<int, std::map<int, uint32_t>> tma_desc_ureg_by_pc;
std::map<int, std::map<int, uint32_t>> first_dest_ureg_by_pc;
std::map<int, std::map<int, std::string>> instruction_text_by_pc;
struct sync_runtime_capture_site_info {
  sync_runtime_capture_site_info()
      : enabled(false),
        barrier_callback_index(-1),
        barrier_mref_index(-1),
        semantic_callback_index(-1),
        semantic_is_zero_literal(false),
        semantic_reg_type(TRACED_REG_TYPE::NO_REGS),
        semantic_reg_id(-1) {}
  bool enabled;
  int barrier_callback_index;
  int barrier_mref_index;
  int semantic_callback_index;
  bool semantic_is_zero_literal;
  TRACED_REG_TYPE semantic_reg_type;
  int semantic_reg_id;
};
std::map<int, std::map<int, sync_runtime_capture_site_info>>
    sync_runtime_capture_sites_by_pc;
// BAR.SYNC / BAR.ARV named-barrier register-operand capture. Mirrors the sync site map:
// records, per (function_id, pc), which device callback_index carries the runtime value of
// a register-form barrier id (operand 0) and/or count (operand 1). Immediate operands are
// not captured here (the decoder reads them statically).
struct bar_runtime_capture_site_info {
  bar_runtime_capture_site_info()
      : enabled(false),
        id_callback_index(-1),
        count_callback_index(-1) {}
  bool enabled;
  int id_callback_index;
  int count_callback_index;
};
std::map<int, std::map<int, bar_runtime_capture_site_info>>
    bar_runtime_capture_sites_by_pc;
struct tma_completion_runtime_capture_site_info {
  tma_completion_runtime_capture_site_info()
      : enabled(false),
        barrier_callback_index(-1),
        barrier_ureg_id(SECRET_UREG_DESC_NOT_USED) {}
  bool enabled;
  int barrier_callback_index;
  uint32_t barrier_ureg_id;
};
std::map<int, std::map<int, tma_completion_runtime_capture_site_info>>
    tma_completion_runtime_capture_sites_by_pc;
uint64_t tma_memcpy_dump_id = 0;
uint64_t tensor_map_encode_dump_id = 0;
struct tma_desc_producer_candidate_t {
  int unique_function_id;
  int pc;
  uint32_t dest_ureg_id;
  uint32_t pre_value_lo;
  uint32_t pre_value_hi;
};
std::unordered_map<std::string, std::deque<tma_desc_producer_candidate_t>> recent_tma_desc_producers_by_warp;
// =============================================================================

struct traced_operand_instrument {
  traced_operand_instrument() = default; // Add this default constructor
  traced_operand_instrument(TRACED_REG_TYPE reg_type, int num_regs, int first_reg_id) {
    this->reg_type = reg_type;
    this->num_regs = num_regs;
    this->first_reg_id = first_reg_id;
  }
  TRACED_REG_TYPE reg_type;
  int num_regs;
  int first_reg_id;
};

struct traced_kernel_id {
  traced_kernel_id(std::string kernel_name, unsigned int variant_id, unsigned int candidate_unique_function_id,std::map<int, std::string> key_instructions_by_pc, std::map<int, std::string> call_or_ret_by_pc, uint64_t func_addr) {
    this->original_kernel_name = kernel_name;
    this->variant_id = variant_id;
    this->unique_function_id = candidate_unique_function_id;
    this->key_instructions_by_pc = key_instructions_by_pc;
    this->call_or_ret_by_pc = call_or_ret_by_pc;
    this->func_addr = func_addr;
    sass_has_been_parsed = false;
    rfu_has_been_parsed = false;
  }
  std::string original_kernel_name;
  unsigned int unique_function_id; // ID to identify statically the code of the kernel. It should be repetead over all the kernels with the same name and variant
  unsigned int variant_id;
  bool sass_has_been_parsed;
  bool rfu_has_been_parsed;
  std::map<int, std::string> key_instructions_by_pc;
  std::map<int, std::string> call_or_ret_by_pc;
  uint64_t func_addr;
};

// This structures are used to uniquely identify kernels. Some of them have the same name, even that they have different code
std::map<std::string,std::vector<traced_kernel_id>> all_kernels_key_instructions_by_pc; // Stores all the kernels informations ofthe execution
std::map<int, std::string> current_kernel_key_instructions_by_pc; // Only stores the information of the current analyzed kernel.
std::map<int, std::string> current_kernel_call_or_ret_by_pc; // Only stores the information of the current analyzed kernel.
std::map<CUfunction, std::tuple<std::string,int>> map_function_to_kernel_name_and_variant_id; // This is used to store the kernel name and variant id of the kernels that are being instrumented
std::map<std::string, int> map_function_name_to_unique_function_id_with_variant; // This is used to store the unique function identifier of the kernels that are being instrumented
std::map<std::string, int> map_function_name_to_unique_function_id_without_variant;
std::map<uint64_t, std::string> map_func_addr_to_kernel_name;
std::map<uint64_t, std::map<int,std::string>> map_func_addr_to_pc_to_sass_instr; // Used for kernels that does not appear in the binary
std::map<std::string, uint64_t> map_kernel_name_to_func_addr;
// First element of tuple is kernel name and second the variant id

traced_execution *m_enhanced_traced_execution;
std::string traces_path = "traces";
std::string extrainfo_path = traces_path + "/extra_info";
std::string tma_htod_blob_path = extrainfo_path + "/tma_htod_blobs";
std::string tensor_map_blob_path = extrainfo_path + "/tensor_map_encode_blobs";
std::string cubin_path = extrainfo_path + "/cubin";
std::string sass_path = extrainfo_path + "/sass";
std::string register_usage_path = extrainfo_path + "/register_usage";
std::string threadblock_trace_path = traces_path + "/threadblocks";
std::string threadblock_register_values_path = traces_path + "/threadblocks/register_values";

std::string get_program_path() {
    size_t size;
    enum Constexpr { MAX_SIZE = 1024 };
    void *array[MAX_SIZE];
    size = backtrace(array, MAX_SIZE);
    char** calls;
    calls = backtrace_symbols(array, size);
    std::string program_caller = calls[size-1];
    std::string program_path = program_caller.substr(0, program_caller.find_last_of("("));
    return program_path;
}

dynamic_trace::Trace dyn_trace;
// Key: d_{device}_s_{stream}_k_{kernel}_{cta_id_x},{cta_id_y},{cta_id_z}
std::unordered_map<std::string, dynamic_trace::threadblock> threadblocks;
//TB key, warp id, int remaining injects
std::unordered_map<std::string, std::map<unsigned int, unsigned int>> remaining_injects_to_current_instruction;
const std::unordered_map<std::string, OpcodeChar> *OpcodeMap = nullptr;

void create_folder(const char * folder_path);

static bool is_tma_desc_consumer_opcode(const std::string &opcode) {
  return opcode.rfind("UTMALDG", 0) == 0 ||
         opcode.rfind("UTMASTG", 0) == 0 ||
         opcode.rfind("UBLKCP", 0) == 0 ||
         opcode.rfind("UBLKPF", 0) == 0 ||
         opcode.rfind("UBLKRED", 0) == 0 ||
         opcode.rfind("UTMACCTL", 0) == 0 ||
         opcode.rfind("UTMACMDFLUSH", 0) == 0 ||
         opcode.rfind("UTMAPF", 0) == 0 ||
         opcode.rfind("UTMAREDG", 0) == 0;
}

static bool is_tma_completion_bearing_load_opcode(
    const std::string &opcode) {
  return opcode.rfind("UTMALDG", 0) == 0 ||
         opcode.rfind("UBLKCP", 0) == 0;
}

enum class sync_runtime_capture_kind {
  NONE,
  EXCH,
  ARRIVE_EXPECT_TX,
  PHASECHK,
};

static sync_runtime_capture_kind get_sync_runtime_capture_kind(
    const std::string &opcode) {
  if (opcode.rfind("SYNCS.EXCH", 0) == 0) {
    return sync_runtime_capture_kind::EXCH;
  }
  // Both SYNCS.ARRIVE.TRANS64 variants (the bare form and the ".RED.*"
  // reduction form) are captured identically: operand 2 is recorded as the
  // candidate expect-tx semantic. Statically the ".RED" form shows RZ in
  // operand 2 (captured as a zero literal -> semantic_raw == 0) while the bare
  // form shows a real register. Capturing both lets the runtime semantic_raw
  // value decide ARRIVE vs ARRIVE_EXPECT_TX instead of baking in an unverified
  // assumption about what ".RED" means.
  if (opcode.rfind("SYNCS.ARRIVE.TRANS64", 0) == 0) {
    return sync_runtime_capture_kind::ARRIVE_EXPECT_TX;
  }
  if (opcode.rfind("SYNCS.PHASECHK", 0) == 0) {
    return sync_runtime_capture_kind::PHASECHK;
  }
  return sync_runtime_capture_kind::NONE;
}

static bool sync_runtime_capture_enabled(sync_runtime_capture_kind kind) {
  return kind != sync_runtime_capture_kind::NONE;
}

static MBARRIER_CONTRACT_EVENT_KIND get_mbarrier_contract_event_kind(
    const std::string &opcode) {
  const sync_runtime_capture_kind kind =
      get_sync_runtime_capture_kind(opcode);
  switch (kind) {
    case sync_runtime_capture_kind::EXCH:
      return MBARRIER_CONTRACT_EVENT_EXCH;
    case sync_runtime_capture_kind::ARRIVE_EXPECT_TX:
      return MBARRIER_CONTRACT_EVENT_ARRIVE;
    case sync_runtime_capture_kind::PHASECHK:
      return MBARRIER_CONTRACT_EVENT_WAIT;
    case sync_runtime_capture_kind::NONE:
      return MBARRIER_CONTRACT_EVENT_NONE;
  }
  return MBARRIER_CONTRACT_EVENT_NONE;
}

static bool is_sync_semantic_operand_type(InstrType::OperandType type) {
  return type == InstrType::OperandType::REG ||
         type == InstrType::OperandType::UREG ||
         type == InstrType::OperandType::PRED ||
         type == InstrType::OperandType::UPRED;
}

static bool is_zero_sync_semantic_operand(const std::string &operand_str) {
  return operand_str == "RZ" || operand_str == "URZ";
}

static sync_runtime_capture_site_info build_sync_runtime_capture_site_info(
    const std::string &opcode, int operand_position, int callback_index,
    int mref_index, InstrType::OperandType operand_type,
    const std::string &operand_str, int register_id) {
  sync_runtime_capture_site_info info;
  sync_runtime_capture_kind kind = get_sync_runtime_capture_kind(opcode);
  if (!sync_runtime_capture_enabled(kind)) {
    return info;
  }

  if (operand_position == 1 && operand_type == InstrType::OperandType::MREF) {
    info.enabled = true;
    info.barrier_callback_index = callback_index;
    info.barrier_mref_index = mref_index;
  } else if (operand_position == 2 &&
             is_sync_semantic_operand_type(operand_type)) {
    info.enabled = true;
    if (is_zero_sync_semantic_operand(operand_str)) {
      info.semantic_is_zero_literal = true;
    } else {
      info.semantic_callback_index = callback_index;
      info.semantic_reg_id = register_id;
      if (operand_type == InstrType::OperandType::REG) {
        info.semantic_reg_type = TRACED_REG_TYPE::REGULAR;
      } else if (operand_type == InstrType::OperandType::UREG) {
        info.semantic_reg_type = TRACED_REG_TYPE::UNIFORM;
      } else {
        info.semantic_reg_type = TRACED_REG_TYPE::NO_REGS;
      }
    }
  }
  return info;
}

static bar_runtime_capture_site_info build_bar_runtime_capture_site_info(
    const std::string &opcode, int operand_position, int callback_index,
    InstrType::OperandType operand_type) {
  bar_runtime_capture_site_info info;
  // Only BAR.SYNC / BAR.ARV named barriers. (MEMBAR/CGAERRBAR are not BAR.*)
  if (opcode.rfind("BAR.", 0) != 0 && opcode != "BAR") {
    return info;
  }
  // operand 0 = barrier id, operand 1 = count. Capture only when register-form; the
  // decoder reads immediate operands statically. A register-form operand emits a
  // device-side reg-value callback, so its runtime value is recoverable at callback_index.
  bool is_reg = (operand_type == InstrType::OperandType::REG ||
                 operand_type == InstrType::OperandType::UREG);
  if (!is_reg) {
    return info;
  }
  if (operand_position == 0) {
    info.enabled = true;
    info.id_callback_index = callback_index;
  } else if (operand_position == 1) {
    info.enabled = true;
    info.count_callback_index = callback_index;
  }
  return info;
}

static int get_first_predicated_lane(uint32_t active_mask, uint32_t predicate_mask) {
  uint32_t relevant = active_mask & predicate_mask;
  if (relevant == 0) {
    return -1;
  }
  for (int lane = 0; lane < 32; ++lane) {
    if (relevant & (1u << lane)) {
      return lane;
    }
  }
  return -1;
}

static bool inst_trace_has_lane_values(const inst_trace_t *record) {
  return (record->record_flags & INST_TRACE_HAS_LANE_VALUES) != 0;
}

static uint64_t inst_trace_value_0(const inst_trace_t *record, int lane) {
  assert(lane >= 0 && lane < 32);
  if (inst_trace_has_lane_values(record)) {
    return record->lane_values[lane];
  }
  return record->scalar_value_0;
}

static int64_t inst_trace_operand_reg_id(const inst_trace_t *record) {
  switch (record->per_operand_type) {
    case TRACED_REG_TYPE::REGULAR:
    case TRACED_REG_TYPE::REGULAR_2_REGS:
    case TRACED_REG_TYPE::REGULAR_4_REGS:
    case TRACED_REG_TYPE::UNIFORM:
    case TRACED_REG_TYPE::UNIFORM_2_REGS:
    case TRACED_REG_TYPE::PREDICATE:
    case TRACED_REG_TYPE::UNIFORM_PREDICATE:
      return static_cast<int64_t>(record->reg_id);
    case TRACED_REG_TYPE::NO_REGS:
    case TRACED_REG_TYPE::MEMORY_REF:
    default:
      return -1;
  }
}

static void normalize_legacy_inst_trace(const legacy_inst_trace_t *legacy,
                                        inst_trace_t *record) {
  record->record_version = INST_TRACE_RECORD_VERSION;
  record->record_flags = 0;
  record->cta_id_x = legacy->cta_id_x;
  if (legacy->cta_id_x == -1) {
    record->record_size = INST_TRACE_COMPACT_SIZE;
    record->mem_type = MEM_TYPE::NONE;
    return;
  }

  record->cta_id_y = legacy->cta_id_y;
  record->cta_id_z = legacy->cta_id_z;
  record->warpid_tb = legacy->warpid_tb;
  record->warpid_sm = legacy->warpid_sm;
  record->sm_id = legacy->sm_id;
  record->reg_id =
      legacy->mem_type == MEM_TYPE::NONE
          ? legacy->reg_id
          : SECRET_UREG_DESC_NOT_USED;
  record->vpc = legacy->vpc;
  record->unique_function_id = legacy->unique_function_id;
  record->mem_type = legacy->mem_type;
  record->width = legacy->width;
  record->ureg_desc_id = legacy->ureg_desc_id;
  record->ureg_desc_value = legacy->ureg_desc_value;
  record->ureg_desc_value_hi = legacy->ureg_desc_value_hi;
  record->active_mask = legacy->active_mask;
  record->predicate_mask = legacy->predicate_mask;
  record->num_of_injects = legacy->num_of_injects;
  record->per_operand_type = legacy->per_operand_type;
  record->scalar_value_0 = 0;
  record->scalar_value_1 = 0;
  record->scalar_value_2 = 0;
  record->scalar_value_3 = 0;
  record->reserved = 0;

  if (legacy->mem_type != MEM_TYPE::NONE) {
    record->record_size = INST_TRACE_MEMORY_SIZE;
    record->record_flags |= INST_TRACE_HAS_LANE_VALUES;
    for (int lane = 0; lane < 32; ++lane) {
      record->lane_values[lane] = legacy->addrs_or_reg_val_0[lane];
    }
    return;
  }

  record->record_size = INST_TRACE_COMPACT_SIZE;
  int value_lane =
      get_first_predicated_lane(legacy->active_mask, legacy->predicate_mask);
  if (value_lane < 0) {
    value_lane = 0;
  }
  if (legacy->per_operand_type != TRACED_REG_TYPE::NO_REGS) {
    record->scalar_value_0 = legacy->addrs_or_reg_val_0[value_lane];
  }
  if (legacy->per_operand_type == TRACED_REG_TYPE::REGULAR_2_REGS ||
      legacy->per_operand_type == TRACED_REG_TYPE::UNIFORM_2_REGS) {
    record->scalar_value_1 = legacy->reg_val_1[value_lane];
  } else if (legacy->per_operand_type == TRACED_REG_TYPE::REGULAR_4_REGS) {
    record->scalar_value_2 = legacy->reg_val_2[value_lane];
    record->scalar_value_3 = legacy->reg_val_3[value_lane];
  }
}

static const char *traced_reg_type_to_string(TRACED_REG_TYPE reg_type) {
  switch (reg_type) {
    case TRACED_REG_TYPE::NO_REGS: return "NO_REGS";
    case TRACED_REG_TYPE::MEMORY_REF: return "MEMORY_REF";
    case TRACED_REG_TYPE::REGULAR: return "REGULAR";
    case TRACED_REG_TYPE::REGULAR_2_REGS: return "REGULAR_2_REGS";
    case TRACED_REG_TYPE::REGULAR_4_REGS: return "REGULAR_4_REGS";
    case TRACED_REG_TYPE::UNIFORM: return "UNIFORM";
    case TRACED_REG_TYPE::UNIFORM_2_REGS: return "UNIFORM_2_REGS";
    case TRACED_REG_TYPE::PREDICATE: return "PREDICATE";
    case TRACED_REG_TYPE::UNIFORM_PREDICATE: return "UNIFORM_PREDICATE";
    default: return "UNKNOWN";
  }
}

static const char *mem_type_to_string(MEM_TYPE mem_type) {
  switch (mem_type) {
    case MEM_TYPE::NONE: return "NONE";
    case MEM_TYPE::STANDARD_MEM: return "STANDARD_MEM";
    case MEM_TYPE::CONSTANT_MEM: return "CONSTANT_MEM";
    case MEM_TYPE::CALL_OR_RET: return "CALL_OR_RET";
    default: return "UNKNOWN";
  }
}

static std::string get_tma_desc_producer_debug_key(const std::string &tb_string_id, int warp_id_tb) {
  return tb_string_id + "_w_" + std::to_string(warp_id_tb);
}

static std::string bytes_to_hex_preview(const uint8_t *data, size_t size, size_t max_preview_bytes) {
  size_t preview_size = std::min(size, max_preview_bytes);
  std::ostringstream oss;
  oss << std::hex;
  for (size_t i = 0; i < preview_size; ++i) {
    if (i != 0) {
      oss << " ";
    }
    oss.width(2);
    oss.fill('0');
    oss << static_cast<unsigned int>(data[i]);
  }
  return oss.str();
}

static std::string uint64_array_to_string(const cuuint64_t *values, uint32_t count) {
  if (values == nullptr || count == 0) {
    return "";
  }
  std::ostringstream oss;
  for (uint32_t i = 0; i < count; ++i) {
    if (i != 0) {
      oss << " ";
    }
    oss << values[i];
  }
  return oss.str();
}

static std::string uint32_array_to_string(const cuuint32_t *values, uint32_t count) {
  if (values == nullptr || count == 0) {
    return "";
  }
  std::ostringstream oss;
  for (uint32_t i = 0; i < count; ++i) {
    if (i != 0) {
      oss << " ";
    }
    oss << values[i];
  }
  return oss.str();
}

static std::string uint64_to_hex_string(uint64_t value) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return oss.str();
}

static void append_tma_memcpy_dump_event(int device_id, uint64_t stream_key, CUdeviceptr dst_device,
                                         const void *src_host, size_t byte_count) {
  if (!enable_tma_desc || src_host == nullptr || byte_count == 0 ||
      byte_count > aux_htod_dump_max_bytes) {
    return;
  }
  create_folder(extrainfo_path.c_str());
  create_folder(tma_htod_blob_path.c_str());
  std::string csv_path = extrainfo_path + "/tma_htod_dump.csv";
  std::ofstream csv;
  bool needs_header = tma_memcpy_dump_header_written.empty();
  if (needs_header) {
    csv.open(csv_path, std::ios::out);
  } else {
    csv.open(csv_path, std::ios::app);
  }
  if (!csv.is_open()) {
    return;
  }
  if (needs_header) {
    tma_memcpy_dump_header_written.insert(device_id);
    csv << "dump_id,device_id,stream_key,dst_device_hex,byte_count,preview_hex,blob_path\n";
  }
  uint64_t dump_id = tma_memcpy_dump_id++;
  std::string blob_path = tma_htod_blob_path + "/" + std::to_string(dump_id) + ".bin";
  std::ofstream blob(blob_path, std::ios::binary | std::ios::out);
  if (blob.is_open()) {
    blob.write(reinterpret_cast<const char *>(src_host), byte_count);
  }
  std::ostringstream dst_stream;
  dst_stream << "0x" << std::hex << static_cast<uint64_t>(dst_device);
  std::string preview_hex = bytes_to_hex_preview(reinterpret_cast<const uint8_t *>(src_host), byte_count, 64);
  csv << dump_id << ","
      << device_id << ","
      << stream_key << ","
      << dst_stream.str() << ","
      << byte_count << ","
      << "\"" << preview_hex << "\"" << ","
      << blob_path << "\n";
}

/* Path A (TMA_BASE_ADDR.md §2.23): dump the host launch-argument buffer. Called at
 * cuLaunchKernel (!is_exit). p->kernelParams is a host array of per-argument pointers;
 * nvbit_get_kernel_argument_sizes gives each argument's byte size, from which we derive
 * the packed param-block offset (the same static tensormap_offset the SASS def-chain
 * yields). For each argument we read up to 128B from its host pointer and record the
 * first 16 qwords. A by-value CUtensorMap arg exposes qword0 = the real base directly.
 * All reads are host-side, so this never triggers the device illegal-access that killed
 * the on-device fact-check. */
static void append_tma_launch_param_dump_event(CUcontext ctx, int device_id,
                                               int kernel_trace_id,
                                               unsigned int unique_function_id,
                                               const cuLaunchKernel_params *p) {
  if (!enable_tma_desc || p == nullptr || p->kernelParams == nullptr) {
    return;
  }
  std::vector<int> arg_sizes = nvbit_get_kernel_argument_sizes(ctx, p->f);
  if (arg_sizes.empty()) {
    return;
  }
  create_folder(extrainfo_path.c_str());
  std::string blob_dir = extrainfo_path + "/launch_param_blobs";
  create_folder(blob_dir.c_str());
  std::string csv_path = extrainfo_path + "/tma_launch_param_dump.csv";
  bool needs_header = tma_launch_param_dump_header_written.empty();
  std::ofstream csv(csv_path, needs_header ? std::ios::out : std::ios::app);
  if (!csv.is_open()) {
    return;
  }
  if (needs_header) {
    tma_launch_param_dump_header_written.insert(device_id);
    csv << "device_id,kernel_id,unique_function_id,arg_index,param_offset_hex,arg_size,"
           "arg_ptr_hex,blob_path,qword0_hex,qword1_hex,qword2_hex,qword3_hex\n";
  }
  /* CUDA packs each argument at its natural alignment; the param-block offset for a TMA
   * descriptor is what the SASS c[0x0][K]+UIADD3 chain computes. We approximate the pack
   * offset by aligning the running cursor to min(size, 8) before placing each argument,
   * which matches the observed 0x1f0/0x2b0/... spacing for 128B descriptors. */
  uint64_t offset = 0;
  for (size_t i = 0; i < arg_sizes.size(); ++i) {
    int sz = arg_sizes[i];
    if (sz <= 0) {
      continue;
    }
    uint64_t align = (uint64_t)(sz < 8 ? sz : 8);
    if (align > 0 && (offset % align) != 0) {
      offset += align - (offset % align);
    }
    const void *arg_ptr = p->kernelParams[i];
    /* Persistent-kernel params are passed as one big by-value struct (e.g. 2368B) whose
     * tensor bases sit at various inner offsets, so 128B is not enough. Dump the FULL arg
     * bytes to a blob and let the offline verifier scan every 8-aligned qword for a base.
     * (Host memory only — cannot fault the device.) */
    std::string blob_path = blob_dir + "/k" + std::to_string(kernel_trace_id) +
                            "_uid" + std::to_string(unique_function_id) + "_arg" +
                            std::to_string(i) + ".bin";
    if (arg_ptr != nullptr) {
      std::ofstream blob(blob_path, std::ios::binary | std::ios::out);
      if (blob.is_open()) {
        blob.write(reinterpret_cast<const char *>(arg_ptr), sz);
      }
    }
    const uint64_t *qw = reinterpret_cast<const uint64_t *>(arg_ptr);
    int nqw = sz / 8;
    csv << device_id << "," << kernel_trace_id << "," << unique_function_id << ","
        << i << ",0x" << std::hex << offset << std::dec << "," << sz << ",0x"
        << std::hex << reinterpret_cast<uintptr_t>(arg_ptr) << std::dec << ","
        << blob_path;
    for (int q = 0; q < 4; ++q) {
      uint64_t v = (arg_ptr != nullptr && q < nqw) ? qw[q] : 0ULL;
      csv << ",0x" << std::hex << v << std::dec;
    }
    csv << "\n";
    offset += (uint64_t)sz;
  }
}

static void append_tensor_map_encode_dump_event(int device_id, const cuTensorMapEncodeTiled_params *p) {
  if (!enable_tma_desc || p == nullptr || p->tensorMap == nullptr) {
    return;
  }
  create_folder(extrainfo_path.c_str());
  create_folder(tensor_map_blob_path.c_str());
  std::string csv_path = extrainfo_path + "/tensor_map_encode_dump.csv";
  std::ofstream csv;
  bool needs_header = tensor_map_encode_dump_header_written.empty();
  if (needs_header) {
    csv.open(csv_path, std::ios::out);
  } else {
    csv.open(csv_path, std::ios::app);
  }
  if (!csv.is_open()) {
    return;
  }
  if (needs_header) {
    tensor_map_encode_dump_header_written.insert(device_id);
    csv << "dump_id,device_id,tensor_map_ptr_hex,global_address_hex,tensor_data_type,tensor_rank,global_dim,global_strides,box_dim,element_strides,interleave,swizzle,l2_promotion,oob_fill,qword0_hex,qword1_hex,qword2_hex,qword3_hex,qword4_hex,qword5_hex,qword6_hex,qword7_hex,blob_path\n";
  }
  uint64_t dump_id = tensor_map_encode_dump_id++;
  std::string blob_path = tensor_map_blob_path + "/" + std::to_string(dump_id) + ".bin";
  std::ofstream blob(blob_path, std::ios::binary | std::ios::out);
  if (blob.is_open()) {
    blob.write(reinterpret_cast<const char *>(p->tensorMap), sizeof(CUtensorMap));
  }
  std::ostringstream tensor_map_ptr_stream;
  tensor_map_ptr_stream << "0x" << std::hex << reinterpret_cast<uintptr_t>(p->tensorMap);
  std::ostringstream global_addr_stream;
  global_addr_stream << "0x" << std::hex << reinterpret_cast<uintptr_t>(p->globalAddress);
  uint32_t global_stride_count = (p->tensorRank > 0) ? p->tensorRank - 1 : 0;
  std::string global_dim = uint64_array_to_string(p->globalDim, p->tensorRank);
  std::string global_strides = uint64_array_to_string(p->globalStrides, global_stride_count);
  std::string box_dim = uint32_array_to_string(p->boxDim, p->tensorRank);
  std::string element_strides = uint32_array_to_string(p->elementStrides, p->tensorRank);
  const uint64_t *qwords = reinterpret_cast<const uint64_t *>(p->tensorMap);
  csv << dump_id << ","
      << device_id << ","
      << tensor_map_ptr_stream.str() << ","
      << global_addr_stream.str() << ","
      << static_cast<int>(p->tensorDataType) << ","
      << p->tensorRank << ","
      << "\"" << global_dim << "\"" << ","
      << "\"" << global_strides << "\"" << ","
      << "\"" << box_dim << "\"" << ","
      << "\"" << element_strides << "\"" << ","
      << static_cast<int>(p->interleave) << ","
      << static_cast<int>(p->swizzle) << ","
      << static_cast<int>(p->l2Promotion) << ","
      << static_cast<int>(p->oobFill) << ","
      << uint64_to_hex_string(qwords[0]) << ","
      << uint64_to_hex_string(qwords[1]) << ","
      << uint64_to_hex_string(qwords[2]) << ","
      << uint64_to_hex_string(qwords[3]) << ","
      << uint64_to_hex_string(qwords[4]) << ","
      << uint64_to_hex_string(qwords[5]) << ","
      << uint64_to_hex_string(qwords[6]) << ","
      << uint64_to_hex_string(qwords[7]) << ","
      << blob_path << "\n";
}

// Phase 0b (TMA_BASE_ADDR.md §2.13, host-deref Gate 0). Called from the
// cuLaunchKernel EXIT handler, AFTER the kernel has synced and the channel has
// flushed (so the captured param_base is populated and the param buffer is still
// alive). Reads the device-global kernel param block at the captured param_base and
// dumps a raw region blob + a small param-base record. The OFFSET slicing (128B at
// param_base + tensormap_offset) and encode-blob comparison happen offline in
// verify_tma_descriptor_deref.py, so the tracer needs no SASS/offset knowledge.
static void append_tma_param_base_deref_event(int device_id, int kernel_trace_id) {
  if (!enable_tma_desc || tma_param_base_capture == nullptr) {
    return;
  }
  if (!tma_param_base_capture->valid || tma_param_base_capture->param_base == 0) {
    // No entry ULDC c[0x0][K] captured this launch (kernel may not use TMA, or K
    // mismatch). Leave a note once so a wrong TMA_PARAM_BASE_CBANK_INDEX is visible.
    return;
  }
  create_folder(extrainfo_path.c_str());
  std::string region_dir = extrainfo_path + "/tma_desc_deref_blobs";
  create_folder(region_dir.c_str());

  unsigned long long param_base = tma_param_base_capture->param_base;
  unsigned int uid = tma_param_base_capture->unique_function_id;
  int region_bytes = tma_param_base_region_bytes;
  if (region_bytes <= 0) {
    region_bytes = 8192;
  }

  std::vector<uint8_t> region(region_bytes, 0);
  CUresult rc = cuMemcpyDtoH(region.data(),
                             (CUdeviceptr)param_base, (size_t)region_bytes);

  std::string csv_path = extrainfo_path + "/tma_param_base_deref.csv";
  bool needs_header = tma_param_base_deref_header_written.find(device_id) ==
                      tma_param_base_deref_header_written.end();
  std::ofstream csv(csv_path, needs_header ? std::ios::out : std::ios::app);
  if (csv.is_open()) {
    if (needs_header) {
      tma_param_base_deref_header_written.insert(device_id);
      csv << "device_id,kernel_id,unique_function_id,param_base_hex,cbank_index_hex,"
             "region_bytes,memcpy_ok,region_blob_path\n";
    }
    std::string blob_path = region_dir + "/uid_" + std::to_string(uid) + "_k_" +
                            std::to_string(kernel_trace_id) + ".bin";
    if (rc == CUDA_SUCCESS) {
      std::ofstream blob(blob_path, std::ios::binary | std::ios::out);
      if (blob.is_open()) {
        blob.write(reinterpret_cast<const char *>(region.data()), region_bytes);
      }
    }
    std::ostringstream pb_stream;
    pb_stream << "0x" << std::hex << param_base;
    std::ostringstream cbank_stream;
    cbank_stream << "0x" << std::hex << tma_param_base_cbank_index;
    csv << device_id << "," << kernel_trace_id << "," << uid << ","
        << pb_stream.str() << "," << cbank_stream.str() << "," << region_bytes
        << "," << (rc == CUDA_SUCCESS ? 1 : 0) << ","
        << (rc == CUDA_SUCCESS ? blob_path : std::string("")) << "\n";
  }

  // Reset for the next launch so each launch re-captures its own param_base.
  tma_param_base_capture->valid = 0;
  tma_param_base_capture->param_base = 0;
  tma_param_base_capture->unique_function_id = 0;
}

/* SPIKE 6 fact-check (§2.18): dump the device-read 128B descriptor samples (qword0 =
 * expected real base) collected this launch, then reset the buffer. */
static void append_tma_desc_factcheck_event(int device_id, int kernel_trace_id) {
  if (!enable_tma_desc || !tma_desc_factcheck_enable ||
      tma_desc_factcheck == nullptr) {
    return;
  }
  if (tma_desc_factcheck->count == 0) {
    return;  // no executed UTMALDG/UTMASTG this launch
  }
  create_folder(extrainfo_path.c_str());
  std::string csv_path = extrainfo_path + "/tma_desc_factcheck.csv";
  bool needs_header = tma_desc_factcheck_header_written.find(device_id) ==
                      tma_desc_factcheck_header_written.end();
  std::ofstream csv(csv_path, needs_header ? std::ios::out : std::ios::app);
  if (csv.is_open()) {
    if (needs_header) {
      tma_desc_factcheck_header_written.insert(device_id);
      csv << "device_id,kernel_id,unique_function_id,pc_hex,mref_ord,space,read_ok,"
             "desc_va_hex,qword0_hex,qword1_hex\n";
    }
    unsigned int n = tma_desc_factcheck->stored;
    if (n > TMA_DESC_FACTCHECK_SLOTS) n = TMA_DESC_FACTCHECK_SLOTS;
    for (unsigned int i = 0; i < n; ++i) {
      csv << device_id << "," << kernel_trace_id << ","
          << tma_desc_factcheck->unique_function_id[i] << ",0x" << std::hex
          << tma_desc_factcheck->pc[i] << std::dec << ","
          << tma_desc_factcheck->mref_ord[i] << ","
          << tma_desc_factcheck->space[i] << ","
          << tma_desc_factcheck->read_ok[i] << ",0x" << std::hex
          << tma_desc_factcheck->desc_va[i]
          << ",0x" << tma_desc_factcheck->qword0[i] << ",0x"
          << tma_desc_factcheck->qword1[i] << std::dec << "\n";
    }
  }
  // Reset for the next launch.
  memset(tma_desc_factcheck, 0, sizeof(tma_desc_factcheck_t));
}


struct cuMemcpyHtoDAsync_v2_params_proxy {
  CUdeviceptr dstDevice;
  const void *srcHost;
  size_t ByteCount;
  CUstream hStream;
};

static void append_tma_desc_runtime_debug_event(
    int device_id, int stream_id, int kernel_trace_id, const inst_trace_t *ma) {
  if (!enable_tma_desc || ma->ureg_desc_id == SECRET_UREG_DESC_NOT_USED) {
    return;
  }
  std::string filename = extrainfo_path + "/tma_desc_runtime_debug.csv";
  create_folder(extrainfo_path.c_str());
  std::ofstream ofs;
  bool needs_header = tma_desc_runtime_debug_header_written.find(device_id) ==
                      tma_desc_runtime_debug_header_written.end();
  if (needs_header) {
    ofs.open(filename, std::ios::out);
  } else {
    ofs.open(filename, std::ios::app);
  }
  if (!ofs.is_open()) {
    return;
  }
  if (needs_header) {
    tma_desc_runtime_debug_header_written.insert(device_id);
    ofs << "device_id,stream_id,kernel_id,unique_function_id,pc_hex,cta_x,cta_y,cta_z,warp_id_tb,sm_id,active_mask,predicate_mask,desc_reg_id,desc_value_lo,desc_value_hi,first_lane_addr\n";
  }
  uint64_t first_lane_addr = 0;
  std::bitset<32> mask(ma->active_mask & ma->predicate_mask);
  for (int lane = 0; lane < 32; ++lane) {
    if (mask.test(lane)) {
      first_lane_addr = inst_trace_value_0(ma, lane);
      break;
    }
  }
  std::ostringstream pc_stream;
  pc_stream << "0x" << std::hex << ma->vpc;
  ofs << device_id << ","
      << stream_id << ","
      << kernel_trace_id << ","
      << ma->unique_function_id << ","
      << pc_stream.str() << ","
      << ma->cta_id_x << ","
      << ma->cta_id_y << ","
      << ma->cta_id_z << ","
      << ma->warpid_tb << ","
      << ma->sm_id << ","
      << ma->active_mask << ","
      << ma->predicate_mask << ","
      << ma->ureg_desc_id << ","
      << ma->ureg_desc_value << ","
      << ma->ureg_desc_value_hi << ","
      << first_lane_addr << "\n";
}

static void append_tma_runtime_operand_debug_event(
    int device_id, int stream_id, int kernel_trace_id, const inst_trace_t *ma,
    const std::string &opcode, unsigned int callback_index) {
  if (!enable_tma_desc) {
    return;
  }
  int first_lane = get_first_predicated_lane(ma->active_mask, ma->predicate_mask);
  if (first_lane < 0) {
    return;
  }
  create_folder(extrainfo_path.c_str());
  std::ofstream ofs(extrainfo_path + "/tma_runtime_operand_debug.jsonl", std::ios::app);
  if (!ofs.is_open()) {
    return;
  }
  uint64_t first_lane_addr = inst_trace_value_0(ma, first_lane);
  // #region debug-point tracer-desc-validity
  const bool desc_valid = ma->ureg_desc_id != SECRET_UREG_DESC_NOT_USED;
  // #endregion debug-point tracer-desc-validity
  std::ostringstream pc_stream;
  pc_stream << "0x" << std::hex << ma->vpc;
  ofs << "{"
      << "\"device_id\":" << device_id << ","
      << "\"stream_id\":" << stream_id << ","
      << "\"kernel_id\":" << kernel_trace_id << ","
      << "\"unique_function_id\":" << ma->unique_function_id << ","
      << "\"pc_hex\":\"" << pc_stream.str() << "\","
      << "\"opcode\":\"" << opcode << "\","
      << "\"callback_index\":" << callback_index << ","
      << "\"operand_type\":\"" << traced_reg_type_to_string(static_cast<TRACED_REG_TYPE>(ma->per_operand_type)) << "\","
      << "\"mem_type\":\"" << mem_type_to_string(ma->mem_type) << "\","
      << "\"operand_reg_id\":" << inst_trace_operand_reg_id(ma) << ","
      << "\"value_lo\":" << inst_trace_value_0(ma, first_lane) << ","
      << "\"value_hi\":" << ma->scalar_value_1 << ","
      << "\"value_2\":" << ma->scalar_value_2 << ","
      << "\"value_3\":" << ma->scalar_value_3 << ","
      << "\"first_lane_addr\":" << first_lane_addr << ","
      << "\"width\":" << ma->width << ","
      // #region debug-point tracer-desc-validity
      << "\"desc_valid\":" << (desc_valid ? "true" : "false") << ","
      // #endregion debug-point tracer-desc-validity
      << "\"desc_reg_id\":" << ma->ureg_desc_id << ","
      << "\"desc_value_lo\":" << ma->ureg_desc_value << ","
      << "\"desc_value_hi\":" << ma->ureg_desc_value_hi << ","
      << "\"cta_x\":" << ma->cta_id_x << ","
      << "\"cta_y\":" << ma->cta_id_y << ","
      << "\"cta_z\":" << ma->cta_id_z << ","
      << "\"warp_id_tb\":" << ma->warpid_tb << ","
      << "\"sm_id\":" << ma->sm_id
      << "}\n";
}

static void append_tma_desc_producer_debug_event(
    int device_id, int stream_id, int kernel_trace_id, const inst_trace_t *ma,
    const std::string &tb_string_id, int warp_id_tb) {
  if (!enable_tma_desc || ma->ureg_desc_id == SECRET_UREG_DESC_NOT_USED) {
    return;
  }
  std::string recent_key = get_tma_desc_producer_debug_key(tb_string_id, warp_id_tb);
  auto recent_it = recent_tma_desc_producers_by_warp.find(recent_key);
  if (recent_it == recent_tma_desc_producers_by_warp.end()) {
    return;
  }
  std::string filename = extrainfo_path + "/tma_desc_producer_debug.csv";
  create_folder(extrainfo_path.c_str());
  std::ofstream ofs;
  bool needs_header = tma_desc_producer_debug_header_written.find(device_id) ==
                      tma_desc_producer_debug_header_written.end();
  if (needs_header) {
    ofs.open(filename, std::ios::out);
  } else {
    ofs.open(filename, std::ios::app);
  }
  if (!ofs.is_open()) {
    return;
  }
  if (needs_header) {
    tma_desc_producer_debug_header_written.insert(device_id);
    ofs << "device_id,stream_id,kernel_id,consumer_function_id,consumer_pc_hex,consumer_desc_reg_id,consumer_desc_value_lo,consumer_desc_value_hi,producer_function_id,producer_pc_hex,producer_dest_ureg_id,producer_pre_value_lo,producer_pre_value_hi,producer_inst_text\n";
  }
  uint32_t consumer_desc_hi_reg = ma->ureg_desc_id + 1;
  for (auto it = recent_it->second.rbegin(); it != recent_it->second.rend(); ++it) {
    if (it->dest_ureg_id != ma->ureg_desc_id && it->dest_ureg_id != consumer_desc_hi_reg) {
      continue;
    }
    std::ostringstream consumer_pc_stream;
    consumer_pc_stream << "0x" << std::hex << ma->vpc;
    std::ostringstream producer_pc_stream;
    producer_pc_stream << "0x" << std::hex << it->pc;
    std::string producer_inst_text = instruction_text_by_pc[it->unique_function_id][it->pc];
    ofs << device_id << ","
        << stream_id << ","
        << kernel_trace_id << ","
        << ma->unique_function_id << ","
        << consumer_pc_stream.str() << ","
        << ma->ureg_desc_id << ","
        << ma->ureg_desc_value << ","
        << ma->ureg_desc_value_hi << ","
        << it->unique_function_id << ","
        << producer_pc_stream.str() << ","
        << it->dest_ureg_id << ","
        << it->pre_value_lo << ","
        << it->pre_value_hi << ","
        << producer_inst_text << "\n";
  }
}

dynamic_trace::threadblock& get_threadblock(std::string tb_key) {
  if(threadblocks.find(tb_key) == threadblocks.end()) {
    threadblocks[tb_key] = dynamic_trace::threadblock();
  }
  return threadblocks[tb_key];
}

// Get by reference the bool of is_last_warp_inst_ldgsts_half
unsigned int& get_remaining_injects_to_current_instruction(std::string tb_key, unsigned int warp_id) {
  if(remaining_injects_to_current_instruction.find(tb_key) == remaining_injects_to_current_instruction.end()) {
    remaining_injects_to_current_instruction[tb_key] = std::map<unsigned int, unsigned int>();
  }
  if(remaining_injects_to_current_instruction[tb_key].find(warp_id) == remaining_injects_to_current_instruction[tb_key].end()) {
    remaining_injects_to_current_instruction[tb_key][warp_id] = 0;
  }
  return remaining_injects_to_current_instruction[tb_key][warp_id];
}

// =============================================================================
// Incremental Flush: Write a single threadblock to disk immediately
// =============================================================================
// This function is called when a threadblock's instruction count exceeds the
// threshold. It writes the threadblock to disk and clears it from memory.
// Returns true if flush was successful.
bool flush_threadblock_immediate(const std::string& tb_string_id,
                                  dynamic_trace::threadblock& tb,
                                  int device_id, int stream_id, int kern_id) {
  // Create hierarchical folder structure
  std::string device_folder = threadblock_trace_path + "/device_" + std::to_string(device_id);
  std::string stream_folder = device_folder + "/stream_" + std::to_string(stream_id);
  std::string kernel_folder = stream_folder + "/kernel_" + std::to_string(kern_id);

  // Create folders (create_folder handles existing folders)
  mkdir(device_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  mkdir(stream_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  mkdir(kernel_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  // Get current part number for this TB
  int part_num = tb_flush_part_count[tb_string_id]++;

  // Write threadblock to PART file (e.g., tb_id.part0.pb)
  std::string tb_file = kernel_folder + "/" + tb_string_id + ".part" + std::to_string(part_num) + ".pb";
  std::ofstream ofs_tb(tb_file, std::ios::out | std::ios::binary);

  if (!tb.SerializeToOstream(&ofs_tb)) {
    std::cerr << "[INCREMENTAL_FLUSH] ERROR: Failed to write " << tb_file << std::endl;
    return false;
  }

  ofs_tb.close();

  // Get instruction count for stats
  uint64_t inst_count = tb_instruction_count[tb_string_id];

  // Update stats
  total_incremental_flushes++;
  total_instructions_flushed += inst_count;

  // Debug output
  if (verbose) {
    printf("[INCREMENTAL_FLUSH] Flushed %s part %d (%lu instructions) -> %s\n",
           tb_string_id.c_str(), part_num, inst_count, tb_file.c_str());
  }

  // Clear the threadblock from memory
  tb.Clear();

  // Reset instruction counter for this TB
  tb_instruction_count[tb_string_id] = 0;

  // Also clear remaining_injects state for this TB
  remaining_injects_to_current_instruction.erase(tb_string_id);

  return true;
}

// =============================================================================
// Merge all part files with remaining in-memory data into final file
// =============================================================================
void merge_and_write_threadblock(const std::string& tb_string_id,
                                  dynamic_trace::threadblock& tb_remaining,
                                  const std::string& kernel_folder) {
  int num_parts = tb_flush_part_count[tb_string_id];

  if (num_parts == 0) {
    // No incremental flushes - just write the remaining data directly
    std::string tb_file = kernel_folder + "/" + tb_string_id + ".pb";
    std::ofstream ofs_tb(tb_file, std::ios::out | std::ios::binary);
    if (!tb_remaining.SerializeToOstream(&ofs_tb)) {
      std::cerr << "Failed to write threadblock content." << std::endl;
      abort();
    }
    ofs_tb.close();
    return;
  }

  // Create merged threadblock starting with the remaining in-memory data
  dynamic_trace::threadblock merged_tb;

  // Copy block_id from remaining (or first part)
  if (tb_remaining.has_block_id()) {
    *merged_tb.mutable_block_id() = tb_remaining.block_id();
  }

  // Read and merge all part files in order
  for (int i = 0; i < num_parts; i++) {
    std::string part_file = kernel_folder + "/" + tb_string_id + ".part" + std::to_string(i) + ".pb";
    std::ifstream ifs_part(part_file, std::ios::in | std::ios::binary);

    if (!ifs_part.is_open()) {
      std::cerr << "[INCREMENTAL_FLUSH] ERROR: Cannot open part file " << part_file << std::endl;
      abort();
    }

    dynamic_trace::threadblock part_tb;
    if (!part_tb.ParseFromIstream(&ifs_part)) {
      std::cerr << "[INCREMENTAL_FLUSH] ERROR: Cannot parse part file " << part_file << std::endl;
      abort();
    }
    ifs_part.close();

    // Copy block_id if not set
    if (!merged_tb.has_block_id() && part_tb.has_block_id()) {
      *merged_tb.mutable_block_id() = part_tb.block_id();
    }

    // Merge warps: append instructions from part to merged
    for (const auto& warp_pair : part_tb.warps()) {
      int warp_id = warp_pair.first;
      const dynamic_trace::warp& part_warp = warp_pair.second;

      // Get or create warp in merged
      dynamic_trace::warp* merged_warp = &(*merged_tb.mutable_warps())[warp_id];
      merged_warp->set_id(warp_id);

      // Append all instructions from part to merged
      for (const auto& inst : part_warp.instructions()) {
        *merged_warp->add_instructions() = inst;
      }
    }

    // Delete part file
    std::remove(part_file.c_str());
  }

  // Now merge remaining in-memory data
  for (const auto& warp_pair : tb_remaining.warps()) {
    int warp_id = warp_pair.first;
    const dynamic_trace::warp& remaining_warp = warp_pair.second;

    // Get or create warp in merged
    dynamic_trace::warp* merged_warp = &(*merged_tb.mutable_warps())[warp_id];
    merged_warp->set_id(warp_id);

    // Append all instructions from remaining to merged
    for (const auto& inst : remaining_warp.instructions()) {
      *merged_warp->add_instructions() = inst;
    }
  }

  // Write final merged file
  std::string tb_file = kernel_folder + "/" + tb_string_id + ".pb";
  std::ofstream ofs_tb(tb_file, std::ios::out | std::ios::binary);
  if (!merged_tb.SerializeToOstream(&ofs_tb)) {
    std::cerr << "Failed to write merged threadblock content." << std::endl;
    abort();
  }
  ofs_tb.close();

  if (verbose) {
    printf("[INCREMENTAL_FLUSH] Merged %d parts + remaining -> %s\n", num_parts, tb_file.c_str());
  }

  // Clear part count for this TB
  tb_flush_part_count.erase(tb_string_id);
}
// =============================================================================

void create_folder(const char * folder_path) {
  if (mkdir(folder_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    if (errno == EEXIST) {
      // alredy exists
    } else {
      // something else
      std::cout << "cannot create folder error:" << strerror(errno)
                << std::endl;
      return;
    }
  }
}

void remove_folder(const char * folder_path) {
  std::filesystem::remove_all(folder_path);
}

std::string read_stripped_line(std::ifstream &ifs) {
  std::string line;
  std::getline(ifs, line);
  std::string clear_line = ReplaceAll(line, "/*", " ");
  clear_line = ReplaceAll(clear_line, "*/", " ");
  clear_line = ReplaceAll(clear_line, ",", " ");
  clear_line = ReplaceAll(clear_line, ";", " ");
  return strip_string(clear_line);
}

std::string getEnclosedSubstring(std::string str) {
  size_t start_pos = str.find('(');
  if(start_pos == std::string::npos)
      return "";
  size_t end_pos = str.find(')', start_pos);
  if(end_pos == std::string::npos)
      return "";
  return str.substr(start_pos + 1, end_pos - start_pos - 1);
}

// Regex to remove dependency requirements, e.g. " &req={1}"
std::regex reqRegex("\\s*&req=\\{[^}]+\\}");
// Regex to remove write-slot annotations, e.g. " &wr=0x4"
std::regex wrRegex("\\s*&wr=0x[0-9A-Fa-f]+");
// Regex to remove write-slot annotations, e.g. " &wr=0x4"
std::regex rdRegex("\\s*&rd=0x[0-9A-Fa-f]+");
// Regex to remove transaction/synchronization annotations, e.g. " ?trans1;" or " ?WAIT4_END_GROUP;"
std::regex transRegex("\\s*\\?[A-Za-z0-9_]+");

std::string replaceInstructionNewExtraInformation(std::string original_sass_string) {
  std::string transformed = original_sass_string;
  transformed = std::regex_replace(transformed, reqRegex, "");
  transformed = std::regex_replace(transformed, wrRegex, "");
  transformed = std::regex_replace(transformed, rdRegex, "");
  transformed = std::regex_replace(transformed, transRegex, "");
  return transformed;
}

void print_map(const std::map<int, std::string> &map_to_print) {
  std::cout << "Number of elements in the map: " << map_to_print.size() << std::endl;
  for(auto it = map_to_print.begin(); it != map_to_print.end(); ++it) {
    std::cout << std::hex << it->first << std::dec << " " << it->second << std::endl;
  }
  std::cout << std::endl;
}

void print_all_traced_kernels() {
  for(auto it = all_kernels_key_instructions_by_pc.begin(); it != all_kernels_key_instructions_by_pc.end(); ++it) {
    for(unsigned int i = 0; i < it->second.size(); i++) {
      std::cout << "Kernel " << it->second[i].original_kernel_name << " variant " << i << " has been traced. SASS parsed? " <<  it->second[i].sass_has_been_parsed
        << " RFU parsed? " << it->second[i].rfu_has_been_parsed << std::endl;
    }
  }
}

void erase_not_tracked_call_or_rets(std::map<int, std::string> &kernel_call_or_ret_by_pc, std::set<unsigned int> &call_or_ret_pcs_not_to_consider) {
  for(auto it = call_or_ret_pcs_not_to_consider.begin(); it != call_or_ret_pcs_not_to_consider.end(); ++it) {
    auto it_candidate = kernel_call_or_ret_by_pc.find(*it);
    if(it_candidate != kernel_call_or_ret_by_pc.end()) {
      kernel_call_or_ret_by_pc.erase(it_candidate);
    }
  }
}

bool are_two_kernels_equal(traced_kernel_id &kernel_to_be_compared, std::map<int, std::string> &candidate_key_instructions_by_pc, std::map<int, std::string> &candidate_call_or_ret_by_pc, bool compare_call_or_ret) {
  bool are_same_key_instructions = (kernel_to_be_compared.key_instructions_by_pc == candidate_key_instructions_by_pc);
  bool are_same_call_or_ret = true;
  if(compare_call_or_ret) {
    are_same_call_or_ret = (kernel_to_be_compared.call_or_ret_by_pc == candidate_call_or_ret_by_pc);
  }
  return are_same_key_instructions && are_same_call_or_ret;
}

static std::string strip_variant_suffix(const std::string &kernel_name) {
  const std::size_t pos = kernel_name.rfind(variant_delimiter_str);
  if (pos == std::string::npos) {
    return kernel_name;
  }
  const std::string suffix = kernel_name.substr(pos + variant_delimiter_str.size());
  if (suffix.empty()) {
    return kernel_name;
  }
  for (char c : suffix) {
    if (c < '0' || c > '9') {
      return kernel_name;
    }
  }
  return kernel_name.substr(0, pos);
}

static std::string to_lower_ascii(std::string value);

static std::string normalize_kernel_match_name(const std::string &kernel_name) {
  return to_lower_ascii(strip_variant_suffix(strip_string(kernel_name)));
}

static int kernel_symbol_match_score(const std::string &symbol_name, const std::string &kernel_name) {
  const std::string symbol_norm = normalize_kernel_match_name(symbol_name);
  const std::string kernel_norm = normalize_kernel_match_name(kernel_name);
  if (symbol_norm.empty() || kernel_norm.empty()) {
    return 0;
  }
  if (symbol_norm == kernel_norm) {
    return 1000;
  }
  if (symbol_norm.find(kernel_norm) != std::string::npos ||
      kernel_norm.find(symbol_norm) != std::string::npos) {
    return static_cast<int>(std::min(symbol_norm.size(), kernel_norm.size()));
  }
  return 0;
}

static int best_kernel_symbol_match_score(const std::string &symbol_name,
                                          const std::unordered_set<std::string> &kernel_names) {
  int best_score = 0;
  for (const auto &kernel_name : kernel_names) {
    best_score = std::max(best_score, kernel_symbol_match_score(symbol_name, kernel_name));
  }
  return best_score;
}

static std::string resolve_traced_kernel_key(const std::string &sass_kernel_name) {
  if (all_kernels_key_instructions_by_pc.find(sass_kernel_name) != all_kernels_key_instructions_by_pc.end()) {
    return sass_kernel_name;
  }

  const std::string sass_base = strip_variant_suffix(sass_kernel_name);
  if (sass_base != sass_kernel_name &&
      all_kernels_key_instructions_by_pc.find(sass_base) != all_kernels_key_instructions_by_pc.end()) {
    return sass_base;
  }

  std::string best;
  std::size_t best_score = 0;
  for (const auto &kv : all_kernels_key_instructions_by_pc) {
    const std::string &candidate = kv.first;
    if (candidate == sass_kernel_name) {
      return candidate;
    }
    const std::string candidate_base = strip_variant_suffix(candidate);
    if (candidate_base == sass_kernel_name || candidate_base == sass_base) {
      return candidate;
    }

    if (candidate.find(sass_kernel_name) != std::string::npos ||
        sass_kernel_name.find(candidate) != std::string::npos ||
        candidate.find(sass_base) != std::string::npos ||
        sass_base.find(candidate_base) != std::string::npos) {
      const std::size_t score = std::min(candidate.size(), sass_kernel_name.size());
      if (score > best_score) {
        best_score = score;
        best = candidate;
      }
    }
  }
  return best;
}

bool has_the_kernel_been_traced(std::string kernel_name, std::map<int, std::string> candidate_parsed_kernel, std::map<int, std::string> &candidate_call_or_ret_by_pc, unsigned int &variant_id, bool is_rfu_parsing, bool compare_call_or_ret, std::set<unsigned int> &call_or_ret_pcs_not_to_consider) {
  bool has_been_traced = false;
  auto it_already_traced = all_kernels_key_instructions_by_pc.find(kernel_name);
  if(it_already_traced != all_kernels_key_instructions_by_pc.end()) {
    for(unsigned int i = 0; !has_been_traced && (i < all_kernels_key_instructions_by_pc[kernel_name].size()); i++) {
      bool is_alread_parsed = is_rfu_parsing ? it_already_traced->second[i].rfu_has_been_parsed : it_already_traced->second[i].sass_has_been_parsed;
      if(compare_call_or_ret) {
        erase_not_tracked_call_or_rets(it_already_traced->second[i].call_or_ret_by_pc, call_or_ret_pcs_not_to_consider);
      }
      if( are_two_kernels_equal(it_already_traced->second[i], candidate_parsed_kernel, candidate_call_or_ret_by_pc, compare_call_or_ret ) && !is_alread_parsed) {
        has_been_traced = true;
        variant_id = it_already_traced->second[i].variant_id;
      }
    }
  }
  return has_been_traced;
}

struct parsed_sass_result {
  unsigned int matched_kernels = 0;
  std::unordered_set<std::string> matched_kernel_names;
};

static bool kernel_name_selected_for_rfu(const std::string &kernel_name,
                                         const std::unordered_set<std::string> &target_kernel_names) {
  if (target_kernel_names.empty()) {
    return true;
  }
  return target_kernel_names.find(normalize_kernel_match_name(kernel_name)) != target_kernel_names.end();
}

parsed_sass_result parse_sass(int binary_version, const std::filesystem::directory_entry &entry) {
  std::string absolute_sass_path = make_abs_path(sass_path);
  std::string sass_file = absolute_sass_path + "/" + entry.path().stem().string() + ".sass";
  std::ifstream ifs_sass; // input file stream
  ifs_sass.open(sass_file, std::ios::in);
  if (!ifs_sass)
  {
    std::cerr << "Error opening file " << sass_file << std::endl;
    fflush(stderr);
    abort();
  }
  
  parsed_sass_result result;
  bool is_starting_reading_kernel = false;
  std::string kernel_name = "";
  traced_kernel *current_traced_kernel = nullptr;
  std::map<int, std::string> full_inst_call_or_ret_by_pc;
  std::set<unsigned int> call_or_ret_pcs_not_to_consider;

  while (!ifs_sass.eof())
  {
    std::string stripped_line = read_stripped_line(ifs_sass);

    if (!is_starting_reading_kernel)
    {
      if ((stripped_line.find("Function") != std::string::npos))
      {
        std::vector<std::string> aux_splitted = split_string(stripped_line, ' ');
        assert(aux_splitted.size() == 3);
        kernel_name = aux_splitted[2];
        current_traced_kernel = new traced_kernel(kernel_name, static_cast<unsigned>(binary_version));
        is_starting_reading_kernel = true;
      }
    }
    else
    {
      if (stripped_line.find("..........") != std::string::npos)
      {
        is_starting_reading_kernel = false;

        // Check if the kernel was captured during the execution
        unsigned int variant_id = 0;
        std::string traced_kernel_key = resolve_traced_kernel_key(kernel_name);
        const std::string &kernel_key = traced_kernel_key.empty() ? kernel_name : traced_kernel_key;
        bool has_been_traced = has_the_kernel_been_traced(kernel_key, current_traced_kernel->get_key_instructions_pcs(), full_inst_call_or_ret_by_pc, variant_id, false, false, call_or_ret_pcs_not_to_consider);

        if(has_been_traced) {
          std::string final_kernel_name = kernel_key + variant_delimiter_str + std::to_string(variant_id);
          auto it_search_funct = map_kernel_name_to_func_addr.find(kernel_key);
          assert(it_search_funct != map_kernel_name_to_func_addr.end());
          all_kernels_key_instructions_by_pc[kernel_key][variant_id].sass_has_been_parsed = true;
          current_traced_kernel->set_kernel_name(final_kernel_name);
          m_enhanced_traced_execution->add_traced_kernel(final_kernel_name, all_kernels_key_instructions_by_pc[kernel_key][variant_id].unique_function_id, current_traced_kernel, it_search_funct->second, true);
          current_traced_kernel = nullptr;
          result.matched_kernels++;
          result.matched_kernel_names.insert(normalize_kernel_match_name(kernel_key));
        }else {
          delete current_traced_kernel;
        }
      }
      else if (stripped_line.find("headerflags") == std::string::npos)
      {
        stripped_line = replaceInstructionNewExtraInformation(stripped_line);
        std::vector<std::string> aux_list = split_string(stripped_line, ' ');
        std::vector<std::string> aux_list2;
        if (binary_version >= 70)
        {
          assert(!ifs_sass.eof());
          aux_list2 = split_string(read_stripped_line(ifs_sass), ' ');
        }
        current_traced_kernel->add_instruction(aux_list, aux_list2, threshold_unique_kernel_checking, stripped_line);
      }
    }
  }
  ifs_sass.close();
  return result;
}

std::string replaceEnclosedSubstring(std::string str, const std::string &replacement)
{
  size_t start_pos = str.find('`');
  if (start_pos == std::string::npos)
    return str;
  size_t end_pos = str.find(')', start_pos);
  if (end_pos == std::string::npos)
    return str;
  str.replace(start_pos, end_pos - start_pos + 1, replacement);
  return str;
}

void parse_rfu_instruction_info(std::vector<std::string> splitted_text, std::vector<std::string> reg_order, std::string full_instruction_str, std::map<int, std::string> &key_instructions_by_pc,
  std::map<unsigned int, std::map<std::string, unsigned int>> &reg_usage_by_pc, std::map<unsigned int, std::tuple<std::string, int>> &call_target_by_pc, std::map<int, std::string> &full_inst_call_or_ret_by_pc, std::set<unsigned int> &call_or_ret_pcs_not_to_consider)
{
  unsigned pc = std::stoul(splitted_text[0], nullptr, 16);
  bool is_all_reg_use_gathered = false;
  int num_already_visited_reg_files = 0;
  std::string instruction_str = full_instruction_str;
  instruction_str = replaceInstructionNewExtraInformation(instruction_str);
  std::string sass_string = create_sass_instr(instruction_str, true, "//");
  if(track_this_instruction(reg_usage_by_pc.size(), threshold_unique_kernel_checking, instruction_str)){
    key_instructions_by_pc[pc] = sass_string;
  }

  if((sass_string.find("CALL")!= std::string::npos) || (sass_string.find("RET")!= std::string::npos)) {
    std::string funct_name = getEnclosedSubstring(sass_string);
    auto it_search_funct = map_kernel_name_to_func_addr.find(funct_name);
    bool is_rel_type = (sass_string.find("REL") != std::string::npos);
    if(!is_rel_type && (it_search_funct != map_kernel_name_to_func_addr.end())) {
      uint64_t func_addr = it_search_funct->second;
      std::stringstream ss;
      ss << "0x" << std::hex << func_addr;
      std::string new_sass_string = replaceEnclosedSubstring(sass_string, ss.str());
      full_inst_call_or_ret_by_pc[pc] = new_sass_string;
    }else if(!is_rel_type && ((sass_string.find("printf") != std::string::npos) || (sass_string.find("assert") != std::string::npos) ) ) {
      call_or_ret_pcs_not_to_consider.insert(pc);
    }else if(!is_rel_type){
      full_inst_call_or_ret_by_pc[pc] = sass_string;
    }
  }

  if(sass_string.find("CALL") != std::string::npos) {
    std::size_t first_char =  sass_string.find("(");
    std::size_t last_char =  sass_string.find(")");
    if((first_char != std::string::npos) && (last_char != std::string::npos)) {
      std::string call_target = sass_string.substr(first_char+1, last_char - first_char - 1);
      auto search_unique_function_id = map_function_name_to_unique_function_id_without_variant.find(call_target);
      if(search_unique_function_id != map_function_name_to_unique_function_id_without_variant.end()) {
        int candidate_unique_function_id = search_unique_function_id->second;
        call_target_by_pc[pc]= std::make_tuple(call_target, candidate_unique_function_id);
      }
    }
  }

  for (unsigned int i = (splitted_text.size() - 1); (i > 0) && !is_all_reg_use_gathered; --i)
  {
    if (splitted_text[i] == "//")
    {
      is_all_reg_use_gathered = true;
    }
    else if (splitted_text[i] == "|")
    {
      num_already_visited_reg_files++;
    }
    else
    {
      unsigned num_regs = std::stoul(splitted_text[i]);
      std::string reg_file_name = reg_order[reg_order.size() - num_already_visited_reg_files];
      reg_usage_by_pc[pc][reg_file_name] = num_regs;
    }
  }
}

void parse_rfu(const std::filesystem::directory_entry &entry,
               const std::unordered_set<std::string> &target_kernel_names) {
  std::string absolute_rfu_path = make_abs_path(register_usage_path);
  std::string rfu_file = absolute_rfu_path + "/" + entry.path().stem().string() + ".rfu";
  std::ifstream ifs_rfu; // input file stream
  ifs_rfu.open(rfu_file, std::ios::in);
  if (!ifs_rfu)
  {
    std::cerr << "Error opening file " << rfu_file << std::endl;
    fflush(stderr);
    abort();
  }
  
  bool is_starting_reading_kernel = false;
  bool is_code_region_started = false;
  bool should_parse_current_kernel = false;
  std::string kernel_name = "";
  std::vector<std::string> reg_order;
  std::map<int, std::string> key_instructions_by_pc;
  std::map<unsigned int, std::map<std::string, unsigned int>> reg_usage_by_pc;
  std::map<unsigned int, std::tuple<std::string, int>> call_target_by_pc;
  std::map<int, std::string> full_call_or_ret_inst_by_pc;
  std::set<unsigned int> call_or_ret_pcs_not_to_consider;
  key_instructions_by_pc.clear();
  reg_usage_by_pc.clear();
  call_target_by_pc.clear();

  while (!ifs_rfu.eof())
  {
    std::string stripped_line = read_stripped_line(ifs_rfu);
    
    if (!is_starting_reading_kernel)
    {
      if ((stripped_line.find("GPR") != std::string::npos))
      {
        stripped_line = ReplaceAll(stripped_line, "|", "");
        std::vector<std::string> aux_splitted = split_string(stripped_line, ' ');
        aux_splitted.erase(aux_splitted.begin());
        reg_order = aux_splitted;
        is_starting_reading_kernel = true;
      }
    }
    else
    {
      if (stripped_line.find("Legend:") != std::string::npos)
      {
        is_starting_reading_kernel = false;
        is_code_region_started = false;
        if (should_parse_current_kernel) {
          unsigned int variant_id = 0;
          bool has_been_traced = has_the_kernel_been_traced(kernel_name, key_instructions_by_pc, full_call_or_ret_inst_by_pc, variant_id, true, true, call_or_ret_pcs_not_to_consider);
          if(has_been_traced) {
            std::string final_kernel_name = kernel_name + variant_delimiter_str + std::to_string(variant_id);
            all_kernels_key_instructions_by_pc[kernel_name][variant_id].rfu_has_been_parsed = true;
            m_enhanced_traced_execution->add_register_usage_to_a_kernel(final_kernel_name, reg_usage_by_pc, call_target_by_pc);
          }
        }
        key_instructions_by_pc.clear();
        reg_usage_by_pc.clear();
        call_target_by_pc.clear();
        full_call_or_ret_inst_by_pc.clear();
        call_or_ret_pcs_not_to_consider.clear();
        should_parse_current_kernel = false;
      }else {
        std::vector<std::string> splitted_text = split_string(stripped_line, ' ');
        if(!splitted_text.empty()) {
          if(should_parse_current_kernel && is_code_region_started && (stripped_line.find(":") == std::string::npos) &&
            (stripped_line.find(".weak") == std::string::npos) && (stripped_line.find(".type") == std::string::npos)
            && (stripped_line.find(".size") == std::string::npos)) {
            parse_rfu_instruction_info(splitted_text, reg_order, stripped_line, key_instructions_by_pc, reg_usage_by_pc, call_target_by_pc, full_call_or_ret_inst_by_pc, call_or_ret_pcs_not_to_consider);
          }else if( should_parse_current_kernel && (splitted_text[0].find("0000") != std::string::npos) && (stripped_line.find("//") != std::string::npos) && (stripped_line.find(":") == std::string::npos) &&
            (stripped_line.find(".weak") == std::string::npos) && (stripped_line.find(".type") == std::string::npos)
            && (stripped_line.find(".size") == std::string::npos)) {
            is_code_region_started = true;
            parse_rfu_instruction_info(splitted_text, reg_order, stripped_line, key_instructions_by_pc, reg_usage_by_pc, call_target_by_pc, full_call_or_ret_inst_by_pc, call_or_ret_pcs_not_to_consider);
          }else if(splitted_text[0].find(".text") != std::string::npos) {
            kernel_name = splitted_text[0].substr(6);
            kernel_name.pop_back(); // Remove the last  : char
            should_parse_current_kernel = kernel_name_selected_for_rfu(kernel_name, target_kernel_names);
          }
        }
      }
    }
  }
  ifs_rfu.close();
}
// MOD. End. Enhanced Tracer

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
              "Exclude predicated off instruction from count");
  GET_VAR_INT(dynamic_kernel_limit_end, "DYNAMIC_KERNEL_LIMIT_END", 0,
              "Limit of the number kernel to be printed, 0 means no limit");
  GET_VAR_INT(dynamic_kernel_limit_start, "DYNAMIC_KERNEL_LIMIT_START", 0,
              "start to report kernel from this kernel id, 0 means starts from "
              "the beginning, i.e. first kernel");
  GET_VAR_INT(
         active_from_start, "ACTIVE_FROM_START", 1,
         "Start instruction tracing from start or wait for cuProfilerStart "
         "and cuProfilerStop. If set to 0, DYNAMIC_KERNEL_LIMIT options have no effect");
  GET_VAR_INT(
      warp_lifetime_only, "WARP_LIFETIME_ONLY", 0,
      "Probe-only mode: record clock64 at SASS EXIT per CTA-local warp and "
      "disable full dynamic instruction tracing");
  GET_VAR_INT(
      warp_lifetime_sidecar, "WARP_LIFETIME_SIDECAR", 0,
      "Record clock64 at SASS EXIT while retaining the full dynamic trace "
      "and normal TMA/sync capture");
  if (warp_lifetime_only && warp_lifetime_sidecar) {
    fprintf(stderr,
            "WARP_LIFETIME_ONLY and WARP_LIFETIME_SIDECAR are mutually "
            "exclusive\n");
    abort();
  }
  resolve_lighter_tracer_config();
  if (lighter_tracer_sparse_reconstruct &&
      (instr_begin_interval != 0 || instr_end_interval != UINT32_MAX)) {
    fprintf(stderr,
            "[TRACE-SPARSE] sparse_reconstruct requires the full instruction "
            "interval\n");
    abort();
  }
  if (lighter_tracer_sparse_reconstruct && warp_lifetime_only) {
    fprintf(stderr,
            "[TRACE-SPARSE] sparse_reconstruct is incompatible with "
            "WARP_LIFETIME_ONLY\n");
    abort();
  }
  printf("[TRACE-WIRE] selected mode=%s legacy_size=%u compact_size=%u "
         "memory_size=%u\n",
         compact_wire_enabled ? "compact" : "legacy",
         INST_TRACE_LEGACY_SIZE, INST_TRACE_COMPACT_SIZE,
         INST_TRACE_MEMORY_SIZE);
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(intermediate_extra_files_persistance, "INTERMEDIATE_EXTRA_FILES_PERSISTANCE", 0, "Enable to not delete intermediate files from the extra trace generation");
  GET_VAR_INT(print_core_id, "TOOL_TRACE_CORE", 0,
              "write the core id in the traces");
  GET_VAR_INT(terminate_after_limit_number_of_kernels_reached, "TERMINATE_UPON_LIMIT", 0, 
              "Stop the process once the current kernel > DYNAMIC_KERNEL_LIMIT_END");
  GET_VAR_INT(user_defined_folders, "USER_DEFINED_FOLDERS", 0, "Uses the user defined "
              "folder TRACES_FOLDER path environment");
  GET_VAR_INT(threshold_unique_kernel_checking, "THRESHOLD_UNIQUE_KERNEL_CHECKING", 10,
              "Number of instructions used to check if kernels with the same name are different");
  printf("[TRACE-PRUNE] enabled=%d\n", prune_operands_enabled);
  GET_VAR_INT(incremental_flush_threshold, "INCREMENTAL_FLUSH_THRESHOLD", 0,
              "Flush threadblocks to disk when instruction count exceeds this threshold. "
              "0 = disabled (default). Recommended: 50000-100000 for large kernels.");
  mbarrier_contract_capture_enabled = parse_strict_integer_config(
      "MBARRIER_CONTRACT_CAPTURE", 0, 0, 1);
  if (mbarrier_contract_capture_enabled) {
    mbarrier_contract_barrier_slots_per_cta =
        parse_strict_integer_config(
            "MBARRIER_CONTRACT_BARRIERS_PER_CTA", 64, 1, 4096);
    mbarrier_contract_records_per_cta =
        parse_strict_integer_config(
            "MBARRIER_CONTRACT_RECORDS_PER_CTA", 8192, 1, 0x7fffffff);
    mbarrier_contract_max_records =
        parse_strict_integer_config(
            "MBARRIER_CONTRACT_MAX_RECORDS", 4194304, 1, 0x7fffffff);
    if (warp_lifetime_only || warp_lifetime_sidecar ||
        incremental_flush_threshold != 0) {
      fprintf(stderr,
              "MBARRIER_CONTRACT_CAPTURE=1 requires full in-memory tracing "
              "without warp-lifetime probes "
              "(WARP_LIFETIME_ONLY=0, WARP_LIFETIME_SIDECAR=0, "
              "INCREMENTAL_FLUSH_THRESHOLD=0)\n");
      abort();
    }
    printf("[MBARRIER-CONTRACT] enabled=1 version=%u "
           "barriers_per_cta=%d records_per_cta=%d max_records=%d "
           "transport=launch_managed_buffer\n",
           MBARRIER_CONTRACT_RECORD_VERSION,
           mbarrier_contract_barrier_slots_per_cta,
           mbarrier_contract_records_per_cta,
           mbarrier_contract_max_records);
  }
  GET_VAR_INT(enable_tma_desc, "ENABLE_TMA_DESC", 1,
              "Enable TMA descriptor capture, including runtime desc handle debug and tensor-map descriptor dumps.");
  GET_VAR_INT(aux_htod_dump_max_bytes, "AUX_HTOD_DUMP_MAX_BYTES", 4096,
              "Maximum auxiliary HtoD memcpy payload size to dump when ENABLE_TMA_DESC is enabled.");
  GET_VAR_INT(tma_param_base_cbank_index, "TMA_PARAM_BASE_CBANK_INDEX", 0x198,
              "Phase 0b: c[0x0][K] index the entry ULDC.64 loads param_base from (host-deref descriptor recovery).");
  GET_VAR_INT(tma_param_base_region_bytes, "TMA_PARAM_BASE_REGION_BYTES", 8192,
              "Phase 0b: bytes of the kernel param block to cuMemcpyDtoH for descriptor slicing.");
  GET_VAR_INT(tma_desc_factcheck_enable, "TMA_DESC_FACTCHECK", 0,
              "SPIKE 6 (§2.18): at executed UTMALDG/UTMASTG, classify the descriptor VA address space (crash-safe) and dump it to tma_desc_factcheck.csv.");
  GET_VAR_INT(tma_desc_factcheck_read, "TMA_DESC_FACTCHECK_READ", 0,
              "SPIKE 6 (§2.18): opt-in — also device-read 128B at the descriptor VA (only when GLOBAL). Off by default because a generic load of a SMEM/raw operand faults.");
  if (warp_lifetime_only) {
    enable_tma_desc = 0;
    incremental_flush_threshold = 0;
    printf("[WARP-LIFETIME] EXIT-only probe enabled; full trace and TMA capture disabled\n");
  } else if (warp_lifetime_sidecar) {
    printf("[WARP-LIFETIME] EXIT sidecar enabled; full trace and TMA capture retained\n");
  }
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());

  // Print incremental flush status
  if (incremental_flush_threshold > 0) {
    printf("[INCREMENTAL_FLUSH] Enabled with threshold = %d instructions per threadblock\n",
           incremental_flush_threshold);
  }

  if (active_from_start == 0) {
    active_region = false;
  }

  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&mutex, &attr);
}

// Key is unique function ID. The next key is PC and finally the Value is the datawidh of the opcode ID of the instruction.
std::map<int,std::map<int, int>>  pc_to_opcode;

/* For each retained instruction, the static unpredicated/payload-free
 * instructions immediately preceding it in the same basic block. */
std::map<int, std::map<int, std::vector<uint32_t>>>
    sparse_prefix_pcs_by_function;

static bool sparse_semantic_anchor(const std::string &opcode) {
  static const std::vector<std::string> prefixes = {
      "BAR",       "BSSY",      "BSYNC",    "BRA",
      "BREAK",     "BRX",       "CALL",     "DEPBAR",
      "EXIT",      "JMP",       "LDGDEPBAR", "MEMBAR",
      "NANOSLEEP", "RET",       "SYNCS",    "UTMA",
      "UBLK",      "WARPSYNC",  "YIELD",    "ULDC",
  };
  for (const auto &prefix : prefixes) {
    if (opcode.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void instrument_function_if_needed(CUcontext ctx, CUfunction func, int device_id) {
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);
  if (mbarrier_contract_capture_enabled) {
    bool launch_has_capture_site = false;
    for (CUfunction related : related_functions) {
      for (Instr *candidate : nvbit_get_instrs(ctx, related)) {
        const std::string opcode =
            candidate->getOpcode() ? candidate->getOpcode() : "";
        if (get_mbarrier_contract_event_kind(opcode) !=
            MBARRIER_CONTRACT_EVENT_NONE) {
          launch_has_capture_site = true;
          break;
        }
      }
      if (launch_has_capture_site) {
        break;
      }
    }
    if (launch_has_capture_site) {
      mbarrier_contract_capture_launches.insert(func);
    }
  }

  /* iterate on function */
  for (auto f : related_functions) {
    current_kernel_key_instructions_by_pc.clear();
    current_kernel_call_or_ret_by_pc.clear();
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second) {
      continue;
    }

    std::string current_kernel_name(nvbit_get_func_name(ctx, f, true));
    uint64_t addr_funct = (uint64_t)nvbit_get_func_addr(ctx, f);
    
    next_candidate_unique_function_id++;

    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    const CFG_t &cfg = nvbit_get_CFG(ctx, f);
    std::unordered_set<Instr *> sparse_block_entries;
    std::unordered_set<Instr *> sparse_block_exits;
    std::unordered_map<Instr *, size_t> sparse_block_index;
    std::vector<std::vector<uint32_t>> sparse_pending_pcs(cfg.bbs.size());
    if (lighter_tracer_sparse_reconstruct && !cfg.is_degenerate) {
      for (size_t bb_index = 0; bb_index < cfg.bbs.size(); bb_index++) {
        const auto &instructions = cfg.bbs[bb_index]->instrs;
        if (instructions.empty()) {
          continue;
        }
        sparse_block_entries.insert(instructions.front());
        sparse_block_exits.insert(instructions.back());
        for (Instr *bb_instruction : instructions) {
          auto inserted =
              sparse_block_index.emplace(bb_instruction, bb_index);
          if (!inserted.second) {
            fprintf(stderr,
                    "[TRACE-SPARSE] instruction belongs to multiple basic "
                    "blocks: function=%s pc=0x%x\n",
                    current_kernel_name.c_str(),
                    static_cast<unsigned int>(
                        bb_instruction->getOffset()));
            abort();
          }
        }
      }
    } else if (lighter_tracer_sparse_reconstruct) {
      printf("[TRACE-SPARSE] function=%s mode=direct reason=degenerate_cfg\n",
             current_kernel_name.c_str());
    }
    if (verbose) {
      printf("Inspecting function %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), addr_funct);
    }

    uint32_t cnt = 0;
    uint64_t sparse_static_retained = 0;
    uint64_t sparse_static_omitted = 0;
    uint64_t sparse_static_cfg_misses = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {

      if (!warp_lifetime_only &&
          (cnt < instr_begin_interval || cnt >= instr_end_interval)) {
        cnt++;
        continue;
      }
      if (verbose) {
        instr->printDecoded();
      }

      if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = instr->getOpcode();
        if (is_tma_desc_consumer_opcode(instr->getOpcode())) {
          tma_consumer_opcode_ids.insert(opcode_id);
        }
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];
      int vpc = (int)instr->getOffset();
      std::string opcode_str =
          instr->getOpcode() ? std::string(instr->getOpcode()) : "";
      pc_to_opcode[next_candidate_unique_function_id][vpc] = opcode_id;

      if (warp_lifetime_only) {
        if (opcode_str.rfind("EXIT", 0) == 0) {
          nvbit_insert_call(instr, "record_warp_exit", IPOINT_BEFORE);
          nvbit_add_call_arg_guard_pred_val(instr);
          nvbit_add_call_arg_const_val64(
              instr, reinterpret_cast<uint64_t>(warp_lifetime_capture));
        }
        cnt++;
        continue;
      }

      if (warp_lifetime_sidecar && opcode_str.rfind("EXIT", 0) == 0) {
        nvbit_insert_call(instr, "record_warp_exit", IPOINT_BEFORE);
        nvbit_add_call_arg_guard_pred_val(instr);
        nvbit_add_call_arg_const_val64(
            instr, reinterpret_cast<uint64_t>(warp_lifetime_capture));
      }

      std::string inst_str = ReplaceAll(instr->getSass(), ",", " ");
      inst_str = replaceInstructionNewExtraInformation(inst_str);
      inst_str = ReplaceAll(inst_str, ";", "");
      inst_str = strip_string(inst_str);

      if(std::string(instr->getOpcode()).find("LDGSTS") != std::string::npos) {
        opcodes_id_ldgsts.insert(opcode_id);
      }

      bool is_call_or_ret = (std::string(instr->getOpcode()).find("CALL") != std::string::npos) || (std::string(instr->getOpcode()).find("RET") != std::string::npos);
      bool is_rel_type = (std::string(instr->getOpcode()).find("REL") != std::string::npos);
      bool is_call_or_ret_with_reg = false;
      bool is_tma_desc_consumer_instruction = is_tma_desc_consumer_opcode(instr->getOpcode());

      if(track_this_instruction(cnt, threshold_unique_kernel_checking ,instr->getOpcode())) {
        current_kernel_key_instructions_by_pc[vpc] = inst_str;
      }

      std::shared_ptr<traced_instruction> inst_parsed = create_no_binay_instruction(vpc, inst_str);
      inst_parsed->set_simulation_opcode(OpcodeMap, inst_parsed->get_op_code());
      instruction_text_by_pc[next_candidate_unique_function_id][vpc] = inst_str;
      if (inst_parsed->get_num_operands() > 0) {
        traced_operand &first_operand = inst_parsed->get_operand(0);
        if (first_operand.get_operand_type() == TraceEnhancedOperandType::UREG &&
            first_operand.get_has_reg()) {
          first_dest_ureg_by_pc[next_candidate_unique_function_id][vpc] =
              first_operand.get_operand_reg_number();
        }
      }

      if(is_call_or_ret && !is_rel_type) {
        current_kernel_call_or_ret_by_pc[vpc] = inst_str;
      }

      map_func_addr_to_pc_to_sass_instr[addr_funct][vpc] = inst_str;
      
      /* Build the original per-operand callback list first. Compact mode may
       * prune callbacks after sync/BAR/TMA semantic roles have been identified. */
      int mem_oper_idx = -1;
      int num_mref = 0;

      bool has_ldc_with_reg = false;
      bool has_ldc_with_ureg = false;
      uint32_t tma_desc_ureg_id = SECRET_UREG_DESC_NOT_USED;
      std::map<uint32_t, uint32_t> memref_idx_with_desc;
      std::map<uint32_t, traced_operand_instrument> per_operand_type;
      int ldc_reg_id = -1;

      std::vector<int> aux_reg_ids;
      uint64_t call_ret_imm = 0;
      uint32_t num_of_injects = 0;
      sync_runtime_capture_site_info sync_site_info;
      bar_runtime_capture_site_info bar_site_info;
      tma_completion_runtime_capture_site_info tma_completion_site_info;
      uint32_t tma_completion_barrier_ureg_id =
          SECRET_UREG_DESC_NOT_USED;

      if(inst_parsed->is_tensor_core_op()) {
        inst_parsed->set_tensor_core_instruction_info();
      }
      for(int i = 0; i < instr->getNumOperands(); ++i){
        const InstrType::operand_t *op = instr->getOperand(i);
        std::string operand_str = op->str ? std::string(op->str) : "";
        const int operand_mref_index =
            op->type == InstrType::OperandType::MREF
                ? mem_oper_idx + 1
                : -1;
        const int operand_register_id =
            is_sync_semantic_operand_type(op->type)
                ? static_cast<int>(op->u.reg.num)
                : -1;
        if (is_tma_completion_bearing_load_opcode(opcode_str) && i == 0) {
          if (op->type != InstrType::OperandType::MREF ||
              operand_str.rfind("[UR", 0) != 0) {
            fprintf(stderr,
                    "[TMA-COMPLETION] unsupported destination-state operand: "
                    "function=%s pc=0x%x opcode=%s operand=%s\n",
                    current_kernel_name.c_str(), vpc, opcode_str.c_str(),
                    operand_str.c_str());
            abort();
          }
          const uint32_t destination_ureg_id =
              get_ur_register(operand_str);
          tma_completion_barrier_ureg_id = destination_ureg_id + 1;
        }
        sync_runtime_capture_site_info sync_operand_info =
            build_sync_runtime_capture_site_info(opcode_str, i, num_of_injects,
                                                 operand_mref_index, op->type,
                                                 operand_str,
                                                 operand_register_id);
        if (sync_operand_info.enabled) {
          sync_site_info.enabled = true;
          if (sync_operand_info.barrier_callback_index >= 0) {
            sync_site_info.barrier_callback_index =
                sync_operand_info.barrier_callback_index;
            sync_site_info.barrier_mref_index =
                sync_operand_info.barrier_mref_index;
          }
          if (sync_operand_info.semantic_callback_index >= 0) {
            sync_site_info.semantic_callback_index =
                sync_operand_info.semantic_callback_index;
            sync_site_info.semantic_reg_type =
                sync_operand_info.semantic_reg_type;
            sync_site_info.semantic_reg_id =
                sync_operand_info.semantic_reg_id;
          }
          if (sync_operand_info.semantic_is_zero_literal) {
            sync_site_info.semantic_is_zero_literal = true;
          }
        }
        bar_runtime_capture_site_info bar_operand_info =
            build_bar_runtime_capture_site_info(opcode_str, i, num_of_injects,
                                                op->type);
        if (bar_operand_info.enabled) {
          bar_site_info.enabled = true;
          if (bar_operand_info.id_callback_index >= 0) {
            bar_site_info.id_callback_index = bar_operand_info.id_callback_index;
          }
          if (bar_operand_info.count_callback_index >= 0) {
            bar_site_info.count_callback_index =
                bar_operand_info.count_callback_index;
          }
        }
        if(is_tma_desc_consumer_instruction && operand_str.find("desc[UR") != std::string::npos) {
          tma_desc_ureg_id = get_ur_register(operand_str);
        } else if (opcode_str.rfind("UTMASTG", 0) == 0 && i == 0 &&
                   operand_str.find("[UR") != std::string::npos) {
          tma_desc_ureg_id = get_ur_register(operand_str);
        }
        if (op->type == InstrType::OperandType::MREF) {
          mem_oper_idx++;
          num_mref++;
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          if(op->u.mref.has_desc) {
            memref_idx_with_desc[mem_oper_idx] = op->u.mref.desc_ureg_num;
          }
          num_of_injects++;
        }else if( (op->type == InstrType::OperandType::CBANK) && (instr->getOperand(i)->u.cbank.has_reg_offset || (std::string(op->str).find("UR") != std::string::npos) ) ) {
          if(std::string(op->str).find("UR") != std::string::npos) {
            has_ldc_with_ureg = true;
            ldc_reg_id = get_ur_register(std::string(op->str));
          }else {
            has_ldc_with_reg = true;
            ldc_reg_id = instr->getOperand(i)->u.cbank.reg_offset;
          }
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::IMM_UINT64 && is_call_or_ret) {
          call_ret_imm = op->u.imm_uint64.value;
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::REG && is_call_or_ret) {
          aux_reg_ids.push_back(op->u.reg.num);
          aux_reg_ids.push_back(op->u.reg.num + 1);
          is_call_or_ret_with_reg = true;
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::REG) {
          TraceEnhancedOperandType reg_type = TraceEnhancedOperandType::REG;
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, reg_type);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::REGULAR, num_uses, op->u.reg.num);
          if(num_uses == 2) {
            per_operand_type[num_of_injects].reg_type = TRACED_REG_TYPE::REGULAR_2_REGS;
          }else if(num_uses == 4) {
            per_operand_type[num_of_injects].reg_type = TRACED_REG_TYPE::REGULAR_4_REGS;
          }
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::UREG) {
          TraceEnhancedOperandType reg_type = TraceEnhancedOperandType::UREG;
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, reg_type);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::UNIFORM, num_uses, op->u.reg.num);
          if(num_uses == 2) {
            per_operand_type[num_of_injects].reg_type = TRACED_REG_TYPE::UNIFORM_2_REGS;
          }
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::PRED) {
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, TraceEnhancedOperandType::PRED);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::PREDICATE, num_uses, op->u.reg.num);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::UPRED) {
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, TraceEnhancedOperandType::UPRED);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::UNIFORM_PREDICATE, num_uses, op->u.reg.num);
          num_of_injects++;
        }
      }
      if (is_tma_completion_bearing_load_opcode(opcode_str)) {
        if (tma_completion_barrier_ureg_id ==
            SECRET_UREG_DESC_NOT_USED) {
          fprintf(stderr,
                  "[TMA-COMPLETION] missing barrier UReg: function=%s "
                  "pc=0x%x opcode=%s\n",
                  current_kernel_name.c_str(), vpc, opcode_str.c_str());
          abort();
        }
        tma_completion_site_info.enabled = true;
        tma_completion_site_info.barrier_callback_index =
            static_cast<int>(num_of_injects);
        tma_completion_site_info.barrier_ureg_id =
            tma_completion_barrier_ureg_id;
        per_operand_type[num_of_injects] =
            traced_operand_instrument(TRACED_REG_TYPE::UNIFORM, 1,
                                      tma_completion_barrier_ureg_id);
        num_of_injects++;
      }
      /*
       * The replay protobuf has no generic register-value field. In compact
       * mode, keep only callbacks whose runtime values have a proven consumer:
       *
       *   - MEMORY_REF: replay addresses, sync barrier addresses, LDC/call
       *   - sync semantic: PHASECHK/ARRIVE/EXCH runtime semantic or phase
       *   - BAR register operands: runtime named-barrier id/count
       *   - TMA family: every operand, because tma_runtime_operand_debug.jsonl
       *     is consumed by strict TMA sidecar generation
       *
       * Omitting prune_operands restores the all-operand compact callback
       * stream for diagnostics. The legacy transport is intentionally
       * unchanged.
       */
      if (compact_wire_enabled && prune_operands_enabled &&
          !is_tma_desc_consumer_instruction) {
        std::map<uint32_t, traced_operand_instrument> retained_operands;
        std::map<int, int> old_to_new_callback_index;

        for (const auto &entry : per_operand_type) {
          const int old_index = static_cast<int>(entry.first);
          const bool is_memory =
              entry.second.reg_type == TRACED_REG_TYPE::MEMORY_REF;
          const bool is_sync_semantic =
              sync_site_info.semantic_callback_index == old_index;
          const bool is_bar_runtime =
              bar_site_info.id_callback_index == old_index ||
              bar_site_info.count_callback_index == old_index;
          if (!is_memory && !is_sync_semantic && !is_bar_runtime) {
            continue;
          }

          const int new_index =
              static_cast<int>(retained_operands.size());
          retained_operands[static_cast<uint32_t>(new_index)] = entry.second;
          old_to_new_callback_index[old_index] = new_index;
        }

        auto remap_required_callback =
            [&](int &callback_index, const char *role) {
              if (callback_index < 0) {
                return;
              }
              auto remap_it =
                  old_to_new_callback_index.find(callback_index);
              if (remap_it == old_to_new_callback_index.end()) {
                fprintf(stderr,
                        "[TRACE-PRUNE] required %s callback was pruned: "
                        "function=%d pc=0x%x old_index=%d\n",
                        role, next_candidate_unique_function_id, vpc,
                        callback_index);
                abort();
              }
              callback_index = remap_it->second;
            };

        remap_required_callback(sync_site_info.barrier_callback_index,
                                "sync-barrier");
        remap_required_callback(sync_site_info.semantic_callback_index,
                                "sync-semantic");
        remap_required_callback(bar_site_info.id_callback_index, "bar-id");
        remap_required_callback(bar_site_info.count_callback_index,
                                "bar-count");

        per_operand_type.swap(retained_operands);
        num_of_injects =
            static_cast<uint32_t>(per_operand_type.size());
      }

      if (is_tma_desc_consumer_instruction && tma_desc_ureg_id != SECRET_UREG_DESC_NOT_USED) {
        tma_desc_ureg_by_pc[next_candidate_unique_function_id][vpc] = tma_desc_ureg_id;
      }
      if (sync_site_info.enabled) {
        sync_runtime_capture_sites_by_pc[next_candidate_unique_function_id][vpc] =
            sync_site_info;
      }
      if (bar_site_info.enabled) {
        bar_runtime_capture_sites_by_pc[next_candidate_unique_function_id][vpc] =
            bar_site_info;
      }
      if (tma_completion_site_info.enabled) {
        tma_completion_runtime_capture_sites_by_pc
            [next_candidate_unique_function_id][vpc] =
                tma_completion_site_info;
      }

      if (lighter_tracer_sparse_reconstruct) {
        bool retain_directly = true;
        auto bb_it = sparse_block_index.find(instr);
        if (!cfg.is_degenerate && bb_it != sparse_block_index.end()) {
          const bool is_block_entry =
              sparse_block_entries.count(instr) != 0;
          const bool is_block_exit =
              sparse_block_exits.count(instr) != 0;
          const bool has_runtime_payload = num_of_injects != 0;
          retain_directly =
              is_block_entry || is_block_exit || instr->hasPred() ||
              has_runtime_payload || sync_site_info.enabled ||
              bar_site_info.enabled || is_tma_desc_consumer_instruction ||
              is_call_or_ret || sparse_semantic_anchor(opcode_str);

          std::vector<uint32_t> &pending =
              sparse_pending_pcs[bb_it->second];
          if (!retain_directly) {
            pending.push_back(static_cast<uint32_t>(vpc));
            sparse_static_omitted++;
            cnt++;
            continue;
          }
          if (!pending.empty()) {
            auto &function_spans =
                sparse_prefix_pcs_by_function[
                    next_candidate_unique_function_id];
            if (function_spans.count(vpc) != 0) {
              fprintf(stderr,
                      "[TRACE-SPARSE] duplicate retained PC span: "
                      "function=%s pc=0x%x\n",
                      current_kernel_name.c_str(), vpc);
              abort();
            }
            function_spans[vpc] = pending;
            pending.clear();
          }
        } else if (!cfg.is_degenerate) {
          sparse_static_cfg_misses++;
        }
        sparse_static_retained++;
      }

      if(num_of_injects == 0) {
        per_operand_type[0] = traced_operand_instrument(TRACED_REG_TYPE::NO_REGS, 0, 0);
        num_of_injects++;
      }

      for(unsigned int i = 0; i < num_of_injects; i++) {
          /* insert call to the instrumentation function with its arguments */
          nvbit_insert_call(
              instr,
              compact_wire_enabled ? "instrument_inst_compact"
                                   : "instrument_inst",
              IPOINT_BEFORE);
          /* pass predicate value */
          nvbit_add_call_arg_guard_pred_val(instr);
          /* send Unique Function Identifier and PC */
          nvbit_add_call_arg_const_val32(instr, next_candidate_unique_function_id);
          nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
          /* send number of injects and type of operand */
          nvbit_add_call_arg_const_val32(instr, num_of_injects);
          nvbit_add_call_arg_const_val32(instr, per_operand_type[i].reg_type);

          if(per_operand_type[i].reg_type == TRACED_REG_TYPE::MEMORY_REF) {
            if (mem_oper_idx >= 0) {
              nvbit_add_call_arg_const_val32(instr, MEM_TYPE::STANDARD_MEM);
              assert(num_mref <= 2);
              nvbit_add_call_arg_mref_addr64(instr, mem_oper_idx);
              nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());
              uint32_t desc_ureg_id = SECRET_UREG_DESC_NOT_USED;
              if(memref_idx_with_desc.find(mem_oper_idx) != memref_idx_with_desc.end()) {
                desc_ureg_id = memref_idx_with_desc[mem_oper_idx];
              } else if(is_tma_desc_consumer_instruction && tma_desc_ureg_id != SECRET_UREG_DESC_NOT_USED) {
                desc_ureg_id = tma_desc_ureg_id;
              }
              if(desc_ureg_id != SECRET_UREG_DESC_NOT_USED) {
                nvbit_add_call_arg_ureg_val(instr, desc_ureg_id);
                nvbit_add_call_arg_const_val32(instr, desc_ureg_id);
                nvbit_add_call_arg_ureg_val(instr, desc_ureg_id + 1);
              }else {
                nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
                nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
                nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              }
              mem_oper_idx--;
            }else if(has_ldc_with_reg || has_ldc_with_ureg) {
              nvbit_add_call_arg_const_val32(instr, MEM_TYPE::CONSTANT_MEM);
              nvbit_add_call_arg_const_val64(instr, 0);
              if(has_ldc_with_ureg) {
                nvbit_add_call_arg_ureg_val(instr, ldc_reg_id);
              }else {
                nvbit_add_call_arg_reg_val(instr, ldc_reg_id);
              }
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            }else if(is_call_or_ret) {
              nvbit_add_call_arg_const_val32(instr, MEM_TYPE::CALL_OR_RET);
              nvbit_add_call_arg_const_val64(instr, call_ret_imm);
              if(is_call_or_ret_with_reg) {
                nvbit_add_call_arg_reg_val(instr, aux_reg_ids[0]);
                nvbit_add_call_arg_reg_val(instr, aux_reg_ids[1]);
              }else {
                nvbit_add_call_arg_const_val32(instr, 0);
                nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              }
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            }
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::REGULAR) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id); 
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::REGULAR_2_REGS) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 1);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::REGULAR_4_REGS) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 1);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 2);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 3);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::UNIFORM) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_ureg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::UNIFORM_2_REGS) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_ureg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_ureg_val(instr, per_operand_type[i].first_reg_id + 1);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::PREDICATE) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_pred_reg(instr);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::UNIFORM_PREDICATE) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_upred_reg(instr);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else {
            assert(per_operand_type[i].reg_type == TRACED_REG_TYPE::NO_REGS);
            assert(num_of_injects == 1);
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }

          /* add pointer to channel_dev and other counters*/
          nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
          nvbit_add_call_arg_const_val64(instr,
                                        (uint64_t)&total_dynamic_instr_counter);
          nvbit_add_call_arg_const_val64(instr,
                                        (uint64_t)&reported_dynamic_instr_counter);
          nvbit_add_call_arg_const_val64(instr, (uint64_t)&stop_report[device_id]);
          /* host_count replaces successful compact-callback device atomics
           * with receiver-side per-kernel accounting. */
          nvbit_add_call_arg_const_val32(
              instr, lighter_tracer_host_count ? 0 : 1);
      }

      /* Phase 0b (TMA_BASE_ADDR.md §2.13): instrument the entry
       * ULDC.64 URx, c[0x0][K] that loads param_base. K == tma_param_base_cbank_index.
       * Inject capture_tma_param_base at IPOINT_AFTER so the destination UREG pair
       * holds the loaded value (param_base). One call per matching site; the device
       * function keeps only the first write per launch. */
      if (enable_tma_desc && tma_param_base_capture != nullptr &&
          opcode_str.rfind("ULDC", 0) == 0) {
        int dest_ureg = -1;
        bool matches_param_cbank = false;
        for (int oi = 0; oi < instr->getNumOperands(); ++oi) {
          const InstrType::operand_t *op = instr->getOperand(oi);
          std::string op_text = op->str ? std::string(op->str) : "";
          if (op->type == InstrType::OperandType::UREG && dest_ureg < 0) {
            dest_ureg = get_ur_register(op_text);
          }
          if (op->type == InstrType::OperandType::CBANK) {
            /* Match c[0x0][K] by parsing the operand string (avoids depending on
             * the NVBit cbank struct field name across versions). The second
             * bracketed value is the constant offset K. */
            size_t first_lb = op_text.find('[');
            size_t second_lb = op_text.find('[', first_lb + 1);
            if (second_lb != std::string::npos) {
              int parsed_offset = (int)strtol(
                  op_text.c_str() + second_lb + 1, nullptr, 0);
              if (parsed_offset == tma_param_base_cbank_index) {
                matches_param_cbank = true;
              }
            }
          }
        }
        if (matches_param_cbank && dest_ureg >= 0) {
          nvbit_insert_call(instr, "capture_tma_param_base", IPOINT_AFTER);
          nvbit_add_call_arg_guard_pred_val(instr);
          nvbit_add_call_arg_const_val32(instr, next_candidate_unique_function_id);
          nvbit_add_call_arg_ureg_val(instr, dest_ureg);      /* param_base lo */
          nvbit_add_call_arg_ureg_val(instr, dest_ureg + 1);  /* param_base hi */
          nvbit_add_call_arg_const_val64(instr, (uint64_t)tma_param_base_capture);
        }
      }

      /* SPIKE 6 device fact-check (TMA_BASE_ADDR.md §2.18): at an executed
       * UTMALDG/UTMASTG, read 128B at the descriptor-carrying memref address (the
       * generic-SMEM tensormap VA, §2.17f) and record qword0/qword1. Uses the FIRST
       * memory-ref operand (operand-1 for UTMALDG, the desc-like first pair for
       * UTMASTG), which is the address UTMACCTL.PF prefetches. Offline we check
       * qword0 in the 7 encode bases. Gated by TMA_DESC_FACTCHECK. */
      if (enable_tma_desc && tma_desc_factcheck != nullptr &&
          (opcode_str.rfind("UTMALDG", 0) == 0 ||
           opcode_str.rfind("UTMASTG", 0) == 0)) {
        /* Instrument EVERY memory-ref operand (max 2), each tagged with its 0-based
         * MREF ordinal. SPIKE 7 showed MREF-0 is a raw-shared cursor (0xe800), so we
         * must also capture MREF-1 to find which operand is the actual descriptor VA.
         * The device fn classifies each VA's space and only reads a plausible GLOBAL. */
        int mref_index = 0;
        for (int oi = 0; oi < instr->getNumOperands(); ++oi) {
          if (instr->getOperand(oi)->type != InstrType::OperandType::MREF) {
            continue;
          }
          nvbit_insert_call(instr, "factcheck_tma_descriptor", IPOINT_BEFORE);
          nvbit_add_call_arg_guard_pred_val(instr);
          nvbit_add_call_arg_const_val32(instr, next_candidate_unique_function_id);
          nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
          nvbit_add_call_arg_const_val32(instr, mref_index);
          nvbit_add_call_arg_mref_addr64(instr, mref_index);
          nvbit_add_call_arg_const_val32(instr, tma_desc_factcheck_read);
          nvbit_add_call_arg_const_val64(instr, (uint64_t)tma_desc_factcheck);
          mref_index++;
        }
      }

      if (mbarrier_contract_capture_enabled) {
        const MBARRIER_CONTRACT_EVENT_KIND event_kind =
            get_mbarrier_contract_event_kind(opcode_str);
        if (event_kind != MBARRIER_CONTRACT_EVENT_NONE) {
          if (ctx_state->mbarrier_contract_registry == nullptr ||
              sync_site_info.barrier_mref_index < 0 ||
              (!sync_site_info.semantic_is_zero_literal &&
               sync_site_info.semantic_reg_id < 0) ||
              (!sync_site_info.semantic_is_zero_literal &&
               sync_site_info.semantic_reg_type !=
                   TRACED_REG_TYPE::REGULAR &&
               sync_site_info.semantic_reg_type !=
                   TRACED_REG_TYPE::UNIFORM)) {
            fprintf(stderr,
                    "[MBARRIER-CONTRACT] unsupported capture operands: "
                    "function=%s pc=0x%x opcode=%s mref=%d semantic_type=%d "
                    "semantic_reg=%d zero=%d\n",
                    current_kernel_name.c_str(), vpc, opcode_str.c_str(),
                    sync_site_info.barrier_mref_index,
                    static_cast<int>(sync_site_info.semantic_reg_type),
                    sync_site_info.semantic_reg_id,
                    sync_site_info.semantic_is_zero_literal ? 1 : 0);
            abort();
          }

          auto add_contract_begin_arguments = [&]() {
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_const_val32(
                instr, next_candidate_unique_function_id);
            nvbit_add_call_arg_const_val32(
                instr, static_cast<uint32_t>(instr->getOffset()));
            nvbit_add_call_arg_const_val32(
                instr, static_cast<uint32_t>(event_kind));
            nvbit_add_call_arg_mref_addr64(
                instr, sync_site_info.barrier_mref_index);
            if (sync_site_info.semantic_is_zero_literal) {
              nvbit_add_call_arg_const_val32(instr, 0);
            } else if (sync_site_info.semantic_reg_type ==
                       TRACED_REG_TYPE::REGULAR) {
              nvbit_add_call_arg_reg_val(
                  instr, sync_site_info.semantic_reg_id);
            } else {
              nvbit_add_call_arg_ureg_val(
                  instr, sync_site_info.semantic_reg_id);
            }
            nvbit_add_call_arg_const_val64(
                instr,
                reinterpret_cast<uint64_t>(
                    ctx_state->mbarrier_contract_registry));
          };

          nvbit_insert_call(
              instr, "record_mbarrier_contract_begin", IPOINT_BEFORE);
          add_contract_begin_arguments();
          nvbit_insert_call(
              instr, "record_mbarrier_contract_end", IPOINT_AFTER);
          nvbit_add_call_arg_guard_pred_val(instr);
          nvbit_add_call_arg_const_val32(
              instr, next_candidate_unique_function_id);
          nvbit_add_call_arg_const_val32(
              instr, static_cast<uint32_t>(instr->getOffset()));
          nvbit_add_call_arg_const_val32(
              instr, static_cast<uint32_t>(event_kind));
          nvbit_add_call_arg_const_val64(
              instr,
              reinterpret_cast<uint64_t>(
                  ctx_state->mbarrier_contract_registry));
        }
      }
      cnt++;
    }

    if (lighter_tracer_sparse_reconstruct) {
      for (const auto &pending : sparse_pending_pcs) {
        if (!pending.empty()) {
          fprintf(stderr,
                  "[TRACE-SPARSE] unterminated basic-block span: "
                  "function=%s omitted=%zu\n",
                  current_kernel_name.c_str(), pending.size());
          abort();
        }
      }
      printf("[TRACE-SPARSE] function=%s static_retained=%llu "
             "static_omitted=%llu cfg_misses=%llu\n",
             current_kernel_name.c_str(),
             static_cast<unsigned long long>(sparse_static_retained),
             static_cast<unsigned long long>(sparse_static_omitted),
             static_cast<unsigned long long>(sparse_static_cfg_misses));
    }

    pthread_mutex_lock(&mutex);
    int variant_id = 0;
    auto it_map_already_instrumented = map_function_to_kernel_name_and_variant_id.find(f);
    assert(it_map_already_instrumented == map_function_to_kernel_name_and_variant_id.end());

    auto it_already_traced = all_kernels_key_instructions_by_pc.find(current_kernel_name);
    if (it_already_traced == all_kernels_key_instructions_by_pc.end())
    {
      all_kernels_key_instructions_by_pc[current_kernel_name].push_back(traced_kernel_id(current_kernel_name, 0, next_candidate_unique_function_id, current_kernel_key_instructions_by_pc, current_kernel_call_or_ret_by_pc, addr_funct));
    }
    else
    {
      bool is_already_traced = false;
      for (unsigned int i = 0; !is_already_traced && (i < it_already_traced->second.size()); i++)
      {
        if (are_two_kernels_equal(it_already_traced->second[i], current_kernel_key_instructions_by_pc, current_kernel_call_or_ret_by_pc, true)) // CAMBIAR A FUNCION
        {
          is_already_traced = true;
          variant_id = it_already_traced->second[i].variant_id;
        }
      }

      if (!is_already_traced)
      {
        variant_id = it_already_traced->second.size();
        all_kernels_key_instructions_by_pc[current_kernel_name].push_back(traced_kernel_id(current_kernel_name, it_already_traced->second.size(), next_candidate_unique_function_id, current_kernel_key_instructions_by_pc, current_kernel_call_or_ret_by_pc, addr_funct));
      }
    }
    
    std::string candidate_final_kernel_name = current_kernel_name + variant_delimiter_str + std::to_string(variant_id);
    map_function_to_kernel_name_and_variant_id[f] = std::make_tuple(current_kernel_name, variant_id);
    map_function_name_to_unique_function_id_with_variant[candidate_final_kernel_name] = next_candidate_unique_function_id;
    map_function_name_to_unique_function_id_without_variant[current_kernel_name] = next_candidate_unique_function_id;
    map_func_addr_to_kernel_name[addr_funct] = current_kernel_name;
    map_kernel_name_to_func_addr[current_kernel_name] = addr_funct;
    pthread_mutex_unlock(&mutex);
  }
}

__global__ void flush_channel(ChannelDev* ch_dev, int compact_transport) {
  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  if (compact_transport) {
    inst_trace_t ma;
    ma.record_size = INST_TRACE_COMPACT_SIZE;
    ma.record_version = INST_TRACE_RECORD_VERSION;
    ma.record_flags = 0;
    ma.cta_id_x = -1;
    ma.mem_type = MEM_TYPE::NONE;
    ch_dev->push(&ma, ma.record_size);
  } else {
    legacy_inst_trace_t ma;
    ma.cta_id_x = -1;
    ch_dev->push(&ma, sizeof(legacy_inst_trace_t));
  }

  /* flush channel */
  ch_dev->flush();
}

static const mbarrier_contract_record_t *
validated_mbarrier_contract_record(const mbarrier_contract_record_t *record) {
  if (record->record_size != sizeof(mbarrier_contract_record_t) ||
      record->record_version != MBARRIER_CONTRACT_RECORD_VERSION ||
      record->event_kind < MBARRIER_CONTRACT_EVENT_EXCH ||
      record->event_kind > MBARRIER_CONTRACT_EVENT_WAIT ||
      record->launch_uid == 0 ||
      (record->predicate_mask & ~record->active_mask) != 0 ||
      record->predicate_mask == 0) {
    fprintf(stderr,
            "[MBARRIER-CONTRACT] invalid capture record: size=%u version=%u "
            "kind=%u launch=%llu active=0x%x predicate=0x%x\n",
            record->record_size, record->record_version,
            record->event_kind,
            static_cast<unsigned long long>(record->launch_uid),
            record->active_mask, record->predicate_mask);
    abort();
  }
  return record;
}

static void acquire_mbarrier_contract_launch(
    CTXstate *ctx_state, int device_id, int trace_kernel_id, int stream_id,
    const cuLaunchKernel_params *params) {
  assert(mbarrier_contract_capture_enabled);
  assert(ctx_state->mbarrier_contract_registry != nullptr);
  mbarrier_contract_registry_t *registry =
      ctx_state->mbarrier_contract_registry;
  if (registry->active != 0 || registry->launch.state != 0 ||
      registry->error_flags != 0) {
    fprintf(stderr,
            "[MBARRIER-CONTRACT] concurrent or stale launch: active=%d "
            "state=%u errors=0x%x kernel=%d\n",
            registry->active, registry->launch.state,
            registry->error_flags, trace_kernel_id);
    abort();
  }

  const uint64_t num_ctas =
      static_cast<uint64_t>(params->gridDimX) * params->gridDimY *
      params->gridDimZ;
  const uint64_t num_threads =
      static_cast<uint64_t>(params->blockDimX) * params->blockDimY *
      params->blockDimZ;
  const uint64_t warps_per_cta = (num_threads + 31u) / 32u;
  if (num_ctas == 0 || num_ctas > UINT32_MAX ||
      warps_per_cta == 0 ||
      warps_per_cta > MBARRIER_CONTRACT_MAX_WARPS_PER_CTA) {
    fprintf(stderr,
            "[MBARRIER-CONTRACT] unsupported launch dimensions: "
            "ctas=%llu warps_per_cta=%llu kernel=%d\n",
            static_cast<unsigned long long>(num_ctas),
            static_cast<unsigned long long>(warps_per_cta),
            trace_kernel_id);
    abort();
  }

  const uint64_t barrier_entries =
      num_ctas *
      static_cast<uint64_t>(mbarrier_contract_barrier_slots_per_cta);
  const uint64_t scratch_entries = num_ctas * warps_per_cta;
  if (barrier_entries >
          SIZE_MAX / sizeof(mbarrier_contract_barrier_state_t) ||
      scratch_entries >
          SIZE_MAX / sizeof(mbarrier_contract_warp_scratch_t) ||
      num_ctas >
          static_cast<uint64_t>(mbarrier_contract_max_records)) {
    fprintf(stderr,
            "[MBARRIER-CONTRACT] launch allocation overflow: kernel=%d "
            "ctas=%llu barrier_entries=%llu scratch_entries=%llu\n",
            trace_kernel_id,
            static_cast<unsigned long long>(num_ctas),
            static_cast<unsigned long long>(barrier_entries),
            static_cast<unsigned long long>(scratch_entries));
    abort();
  }

  const uint64_t records_per_cta = std::min(
      static_cast<uint64_t>(mbarrier_contract_records_per_cta),
      static_cast<uint64_t>(mbarrier_contract_max_records) / num_ctas);
  const uint64_t record_capacity = num_ctas * records_per_cta;
  if (record_capacity == 0 ||
      record_capacity >
          SIZE_MAX / sizeof(mbarrier_contract_record_t)) {
    fprintf(stderr,
            "[MBARRIER-CONTRACT] record allocation overflow: kernel=%d "
            "capacity=%llu\n",
            trace_kernel_id,
            static_cast<unsigned long long>(record_capacity));
    abort();
  }

  mbarrier_contract_launch_t *launch = &registry->launch;
  if (launch->barrier_state_ptr != 0 ||
      launch->warp_scratch_ptr != 0 ||
      launch->cta_record_count_ptr != 0 ||
      launch->record_buffer_ptr != 0) {
    fprintf(stderr,
            "[MBARRIER-CONTRACT] stale launch pointers: kernel=%d\n",
            trace_kernel_id);
    abort();
  }

  const bool prior_skip_callback = skip_callback_flag;
  skip_callback_flag = true;
  CUDA_SAFECALL(cuMemAllocManaged(
      reinterpret_cast<CUdeviceptr *>(&launch->barrier_state_ptr),
      barrier_entries * sizeof(mbarrier_contract_barrier_state_t),
      CU_MEM_ATTACH_GLOBAL));
  CUDA_SAFECALL(cuMemAllocManaged(
      reinterpret_cast<CUdeviceptr *>(&launch->warp_scratch_ptr),
      scratch_entries * sizeof(mbarrier_contract_warp_scratch_t),
      CU_MEM_ATTACH_GLOBAL));
  CUDA_SAFECALL(cuMemAllocManaged(
      reinterpret_cast<CUdeviceptr *>(&launch->cta_record_count_ptr),
      num_ctas * sizeof(uint64_t), CU_MEM_ATTACH_GLOBAL));
  CUDA_SAFECALL(cuMemAllocManaged(
      reinterpret_cast<CUdeviceptr *>(&launch->record_buffer_ptr),
      record_capacity * sizeof(mbarrier_contract_record_t),
      CU_MEM_ATTACH_GLOBAL));
  CUDA_SAFECALL(cuMemsetD8(
      static_cast<CUdeviceptr>(launch->barrier_state_ptr), 0,
      barrier_entries * sizeof(mbarrier_contract_barrier_state_t)));
  CUDA_SAFECALL(cuMemsetD8(
      static_cast<CUdeviceptr>(launch->warp_scratch_ptr), 0,
      scratch_entries * sizeof(mbarrier_contract_warp_scratch_t)));
  CUDA_SAFECALL(cuMemsetD8(
      static_cast<CUdeviceptr>(launch->cta_record_count_ptr), 0,
      num_ctas * sizeof(uint64_t)));
  skip_callback_flag = prior_skip_callback;

  if (next_mbarrier_contract_launch_uid == 0) {
    fprintf(stderr, "[MBARRIER-CONTRACT] launch UID overflow\n");
    abort();
  }
  launch->device_id = static_cast<uint32_t>(device_id);
  launch->stream_id = static_cast<uint32_t>(stream_id);
  launch->trace_kernel_id = static_cast<uint32_t>(trace_kernel_id);
  launch->launch_uid = next_mbarrier_contract_launch_uid++;
  launch->grid_dim_x = params->gridDimX;
  launch->grid_dim_y = params->gridDimY;
  launch->grid_dim_z = params->gridDimZ;
  launch->num_ctas = static_cast<uint32_t>(num_ctas);
  launch->warps_per_cta = static_cast<uint32_t>(warps_per_cta);
  launch->barrier_slots_per_cta =
      static_cast<uint32_t>(
          mbarrier_contract_barrier_slots_per_cta);
  launch->record_capacity = record_capacity;
  launch->records_per_cta = records_per_cta;
  launch->overflow_records = 0;
  launch->state = 1;
  registry->error_flags = 0;
  __sync_synchronize();
  registry->active = 1;

  printf("[MBARRIER-CONTRACT] launch_begin uid=%llu kernel=%d ctas=%u "
         "warps_per_cta=%u barrier_slots_per_cta=%u "
         "records_per_cta=%llu record_buffer_mib=%.2f\n",
         static_cast<unsigned long long>(launch->launch_uid),
         trace_kernel_id, launch->num_ctas, launch->warps_per_cta,
         launch->barrier_slots_per_cta,
         static_cast<unsigned long long>(launch->records_per_cta),
         static_cast<double>(
             launch->record_capacity *
             sizeof(mbarrier_contract_record_t)) /
             (1024.0 * 1024.0));
  fflush(stdout);
}

static MBARRIER_CONTRACT_EVENT_KIND mbarrier_contract_event_kind_for_pc(
    uint32_t unique_function_id, uint32_t pc) {
  auto function_it = pc_to_opcode.find(unique_function_id);
  if (function_it == pc_to_opcode.end()) {
    return MBARRIER_CONTRACT_EVENT_NONE;
  }
  auto pc_it = function_it->second.find(pc);
  if (pc_it == function_it->second.end()) {
    return MBARRIER_CONTRACT_EVENT_NONE;
  }
  auto opcode_it = id_to_opcode_map.find(pc_it->second);
  if (opcode_it == id_to_opcode_map.end()) {
    return MBARRIER_CONTRACT_EVENT_NONE;
  }
  return get_mbarrier_contract_event_kind(opcode_it->second);
}

static void attach_mbarrier_contract_records(
    const mbarrier_contract_launch_t &launch) {
  const uint64_t *cta_record_counts =
      reinterpret_cast<const uint64_t *>(
          launch.cta_record_count_ptr);
  const mbarrier_contract_record_t *records =
      reinterpret_cast<const mbarrier_contract_record_t *>(
          launch.record_buffer_ptr);
  uint64_t attached = 0;

  for (uint32_t cta = 0; cta < launch.num_ctas; ++cta) {
    const uint32_t cta_x = cta % launch.grid_dim_x;
    const uint32_t cta_y =
        (cta / launch.grid_dim_x) % launch.grid_dim_y;
    const uint32_t cta_z =
        cta / (launch.grid_dim_x * launch.grid_dim_y);
    const std::string tb_key =
        "d_" + std::to_string(launch.device_id) +
        "_s_" + std::to_string(launch.stream_id) +
        "_k_" + std::to_string(launch.trace_kernel_id) +
        "_" + std::to_string(cta_x) + "," +
        std::to_string(cta_y) + "," + std::to_string(cta_z);
    auto tb_it = threadblocks.find(tb_key);
    if (tb_it == threadblocks.end()) {
      if (cta_record_counts[cta] == 0) {
        continue;
      }
      fprintf(stderr,
              "[MBARRIER-CONTRACT] capture references missing CTA: "
              "kernel=%u cta=%u records=%llu\n",
              launch.trace_kernel_id, cta,
              static_cast<unsigned long long>(
                  cta_record_counts[cta]));
      abort();
    }

    using occurrence_key_t = std::tuple<int, int, int>;
    std::map<occurrence_key_t,
             std::vector<dynamic_trace::instruction *>>
        protobuf_occurrences;
    for (auto &warp_pair : *tb_it->second.mutable_warps()) {
      const int warp_id = warp_pair.first;
      dynamic_trace::warp &warp = warp_pair.second;
      for (int index = 0; index < warp.instructions_size(); ++index) {
        dynamic_trace::instruction *instruction =
            warp.mutable_instructions(index);
        if ((instruction->active_mask() &
             instruction->predicate_mask()) == 0 ||
            mbarrier_contract_event_kind_for_pc(
                instruction->function_unique_id(),
                instruction->pc()) ==
                MBARRIER_CONTRACT_EVENT_NONE) {
          continue;
        }
        protobuf_occurrences[
            std::make_tuple(
                warp_id, instruction->function_unique_id(),
                static_cast<int>(instruction->pc()))]
            .push_back(instruction);
      }
    }

    std::map<occurrence_key_t, size_t> next_occurrence;
    const uint64_t cta_records = cta_record_counts[cta];
    if (cta_records > launch.records_per_cta) {
      fprintf(stderr,
              "[MBARRIER-CONTRACT] CTA record count exceeds capacity: "
              "kernel=%u cta=%u records=%llu capacity=%llu\n",
              launch.trace_kernel_id, cta,
              static_cast<unsigned long long>(cta_records),
              static_cast<unsigned long long>(
                  launch.records_per_cta));
      abort();
    }
    const uint64_t cta_base =
        static_cast<uint64_t>(cta) * launch.records_per_cta;
    for (uint64_t local = 0; local < cta_records; ++local) {
      const mbarrier_contract_record_t &record =
          *validated_mbarrier_contract_record(
              &records[cta_base + local]);
      if (record.launch_uid != launch.launch_uid ||
          record.linear_cta_id != cta ||
          record.warp_id_tb >= launch.warps_per_cta) {
        fprintf(stderr,
                "[MBARRIER-CONTRACT] record launch/CTA identity mismatch: "
                "kernel=%u cta=%u index=%llu\n",
                launch.trace_kernel_id, cta,
                static_cast<unsigned long long>(local));
        abort();
      }
      const occurrence_key_t key = std::make_tuple(
          static_cast<int>(record.warp_id_tb),
          static_cast<int>(record.unique_function_id),
          static_cast<int>(record.vpc));
      auto occurrences_it = protobuf_occurrences.find(key);
      const size_t ordinal = next_occurrence[key]++;
      if (occurrences_it == protobuf_occurrences.end() ||
          ordinal >= occurrences_it->second.size()) {
        fprintf(stderr,
                "[MBARRIER-CONTRACT] no protobuf occurrence for capture: "
                "kernel=%u cta=%u warp=%u function=%u pc=0x%x "
                "ordinal=%zu\n",
                launch.trace_kernel_id, cta, record.warp_id_tb,
                record.unique_function_id, record.vpc, ordinal);
        abort();
      }

      dynamic_trace::instruction *instruction =
          occurrences_it->second[ordinal];
      if (static_cast<uint32_t>(
              mbarrier_contract_event_kind_for_pc(
                  instruction->function_unique_id(),
                  instruction->pc())) != record.event_kind ||
          instruction->active_mask() != record.active_mask ||
          instruction->predicate_mask() != record.predicate_mask ||
          !instruction->has_sync() ||
          !instruction->sync().valid() ||
          instruction->sync().barrier_addr() !=
              record.barrier_addr ||
          (instruction->sync().has_semantic_raw()
               ? instruction->sync().semantic_raw()
               : 0ULL) != record.semantic_raw) {
        fprintf(stderr,
                "[MBARRIER-CONTRACT] protobuf/capture payload mismatch: "
                "kernel=%u cta=%u warp=%u function=%u pc=0x%x "
                "ordinal=%zu\n",
                launch.trace_kernel_id, cta, record.warp_id_tb,
                record.unique_function_id, record.vpc, ordinal);
        abort();
      }

      auto *capture = instruction->mutable_mbarrier_contract();
      if (capture->valid()) {
        fprintf(stderr,
                "[MBARRIER-CONTRACT] duplicate protobuf attachment: "
                "kernel=%u cta=%u warp=%u function=%u pc=0x%x "
                "ordinal=%zu\n",
                launch.trace_kernel_id, cta, record.warp_id_tb,
                record.unique_function_id, record.vpc, ordinal);
        abort();
      }
      capture->set_valid(true);
      capture->set_event_kind(record.event_kind);
      capture->set_barrier_addr(record.barrier_addr);
      capture->set_initialization_epoch_before(
          record.initialization_epoch_before);
      capture->set_initialization_epoch_after(
          record.initialization_epoch_after);
      capture->set_opened_contract_count_before(
          record.opened_contract_count_before);
      capture->set_opened_contract_count_after(
          record.opened_contract_count_after);
      capture->set_has_contract_ordinal(
          record.contract_ordinal != 0);
      capture->set_contract_ordinal(record.contract_ordinal);
      capture->set_capture_flags(record.capture_flags);
      capture->set_barrier_error_flags(
          record.barrier_error_flags);
      capture->set_capture_launch_uid(record.launch_uid);
      capture->set_has_semantic_raw(true);
      capture->set_semantic_raw(record.semantic_raw);
      attached++;
    }

    for (const auto &entry : protobuf_occurrences) {
      const size_t consumed = next_occurrence[entry.first];
      if (consumed != entry.second.size()) {
        fprintf(stderr,
                "[MBARRIER-CONTRACT] missing capture records: "
                "kernel=%u cta=%u warp=%d function=%d pc=0x%x "
                "captured=%zu protobuf=%zu\n",
                launch.trace_kernel_id, cta,
                std::get<0>(entry.first), std::get<1>(entry.first),
                std::get<2>(entry.first), consumed,
                entry.second.size());
        abort();
      }
    }
  }

  printf("[MBARRIER-CONTRACT] attach kernel=%u records=%llu status=PASS\n",
         launch.trace_kernel_id,
         static_cast<unsigned long long>(attached));
}

static void release_mbarrier_contract_launch(CTXstate *ctx_state) {
  assert(mbarrier_contract_capture_enabled);
  mbarrier_contract_registry_t *registry =
      ctx_state->mbarrier_contract_registry;
  if (registry == nullptr || registry->active != 1 ||
      registry->launch.state != 1) {
    fprintf(stderr,
            "[MBARRIER-CONTRACT] missing active launch at kernel exit\n");
    abort();
  }
  mbarrier_contract_launch_t *launch = &registry->launch;
  if (registry->error_flags != 0 ||
      launch->overflow_records != 0) {
    fprintf(stderr,
            "[MBARRIER-CONTRACT] launch validation failed: "
            "uid=%llu kernel=%u errors=0x%x overflow=%llu\n",
            static_cast<unsigned long long>(launch->launch_uid),
            launch->trace_kernel_id, registry->error_flags,
            static_cast<unsigned long long>(
                launch->overflow_records));
    abort();
  }

  const uint64_t scratch_entries =
      static_cast<uint64_t>(launch->num_ctas) *
      launch->warps_per_cta;
  const mbarrier_contract_warp_scratch_t *scratch =
      reinterpret_cast<
          const mbarrier_contract_warp_scratch_t *>(
          launch->warp_scratch_ptr);
  for (uint64_t index = 0; index < scratch_entries; ++index) {
    if (scratch[index].state != 0) {
      fprintf(stderr,
              "[MBARRIER-CONTRACT] unterminated callback interval: "
              "uid=%llu kernel=%u scratch=%llu state=%u\n",
              static_cast<unsigned long long>(launch->launch_uid),
              launch->trace_kernel_id,
              static_cast<unsigned long long>(index),
              scratch[index].state);
      abort();
    }
  }

  attach_mbarrier_contract_records(*launch);

  uint64_t emitted = 0;
  const uint64_t *record_counts =
      reinterpret_cast<const uint64_t *>(
          launch->cta_record_count_ptr);
  for (uint32_t cta = 0; cta < launch->num_ctas; ++cta) {
    emitted += record_counts[cta];
  }
  printf("[MBARRIER-CONTRACT] launch_end uid=%llu kernel=%u "
         "records=%llu status=PASS\n",
         static_cast<unsigned long long>(launch->launch_uid),
         launch->trace_kernel_id,
         static_cast<unsigned long long>(emitted));
  fflush(stdout);

  registry->active = 0;
  launch->state = 0;
  const bool prior_skip_callback = skip_callback_flag;
  skip_callback_flag = true;
  CUDA_SAFECALL(cuMemFree(
      static_cast<CUdeviceptr>(launch->barrier_state_ptr)));
  CUDA_SAFECALL(cuMemFree(
      static_cast<CUdeviceptr>(launch->warp_scratch_ptr)));
  CUDA_SAFECALL(cuMemFree(
      static_cast<CUdeviceptr>(launch->cta_record_count_ptr)));
  CUDA_SAFECALL(cuMemFree(
      static_cast<CUdeviceptr>(launch->record_buffer_ptr)));
  skip_callback_flag = prior_skip_callback;
  launch->barrier_state_ptr = 0;
  launch->warp_scratch_ptr = 0;
  launch->cta_record_count_ptr = 0;
  launch->record_buffer_ptr = 0;
  launch->record_capacity = 0;
  launch->records_per_cta = 0;
  launch->overflow_records = 0;
}

static void reset_warp_lifetime_capture(
    int kernel_trace_id, const char *kernel_name,
    const cuLaunchKernel_params *params) {
  assert(warp_lifetime_capture != nullptr);
  const unsigned long long num_ctas =
      static_cast<unsigned long long>(params->gridDimX) *
      params->gridDimY * params->gridDimZ;
  const unsigned int num_threads =
      params->blockDimX * params->blockDimY * params->blockDimZ;
  const unsigned int num_warps = (num_threads + 31) / 32;
  if (num_ctas > WARP_LIFETIME_MAX_CTAS ||
      num_warps > WARP_LIFETIME_MAX_WARPS) {
    fprintf(stderr,
            "[WARP-LIFETIME] launch exceeds capture bounds: ctas=%llu/%u "
            "warps_per_cta=%u/%u\n",
            num_ctas, WARP_LIFETIME_MAX_CTAS, num_warps,
            WARP_LIFETIME_MAX_WARPS);
    abort();
  }

  const bool prior_skip_callback = skip_callback_flag;
  skip_callback_flag = true;
  CUDA_SAFECALL(cuMemsetD8(
      reinterpret_cast<CUdeviceptr>(warp_lifetime_capture), 0,
      sizeof(warp_lifetime_capture_t)));
  skip_callback_flag = prior_skip_callback;
  warp_lifetime_grid_x = params->gridDimX;
  warp_lifetime_grid_y = params->gridDimY;
  warp_lifetime_grid_z = params->gridDimZ;
  warp_lifetime_warps_per_cta = num_warps;
  warp_lifetime_kernel_id = kernel_trace_id;
  warp_lifetime_kernel_name = kernel_name ? kernel_name : "<unknown>";
  warp_lifetime_capture_active = true;

  if (!warp_lifetime_output_initialized) {
    std::ofstream out(traces_path + "/warp_exit_timestamps.csv",
                      std::ios::out | std::ios::trunc);
    out << "kernel_id,kernel_name,cta_x,cta_y,cta_z,warp_id,sm_id,"
           "exit_clock64\n";
    out.close();
    warp_lifetime_output_initialized = true;
  }
}

static void dump_warp_lifetime_capture() {
  if (!warp_lifetime_capture_active || warp_lifetime_capture == nullptr) {
    return;
  }

  std::ofstream out(traces_path + "/warp_exit_timestamps.csv",
                    std::ios::out | std::ios::app);
  const unsigned int num_ctas =
      warp_lifetime_grid_x * warp_lifetime_grid_y * warp_lifetime_grid_z;
  unsigned int rows = 0;
  for (unsigned int cta = 0; cta < num_ctas; ++cta) {
    const unsigned int cta_x = cta % warp_lifetime_grid_x;
    const unsigned int cta_y =
        (cta / warp_lifetime_grid_x) % warp_lifetime_grid_y;
    const unsigned int cta_z =
        cta / (warp_lifetime_grid_x * warp_lifetime_grid_y);
    for (unsigned int warp = 0; warp < warp_lifetime_warps_per_cta; ++warp) {
      const unsigned int slot =
          cta * WARP_LIFETIME_MAX_WARPS + warp;
      const unsigned long long exit_clock =
          warp_lifetime_capture->exit_clock[slot];
      if (exit_clock == 0) {
        continue;
      }
      out << warp_lifetime_kernel_id << "," << warp_lifetime_kernel_name
          << "," << cta_x << "," << cta_y << "," << cta_z
          << "," << warp << "," << warp_lifetime_capture->sm_id[slot]
          << "," << exit_clock << "\n";
      rows++;
    }
  }
  out.close();

  printf("[WARP-LIFETIME] kernel=%d rows=%u overflow=%u output=%s\n",
         warp_lifetime_kernel_id, rows,
         warp_lifetime_capture->overflow_count,
         (traces_path + "/warp_exit_timestamps.csv").c_str());
  warp_lifetime_capture_active = false;
}

static FILE *statsFile = NULL;
static bool first_call = true;
static unsigned int pending_devices_to_finish = 0;

unsigned old_total_insts = 0;
unsigned old_total_reported_insts = 0;

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  // std::cout << "nvbit_at_cuda_event: "  << cbid << std::endl; fflush(stdout);
  pthread_mutex_lock(&mutex);
  /* we prevent re-entry on this callback when issuing CUDA functions inside
    * this function */
  if (skip_callback_flag) {
      pthread_mutex_unlock(&mutex);
      return;
  }

  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  if (first_call == true) {

    first_call = false;

    if (active_from_start && !dynamic_kernel_limit_start || dynamic_kernel_limit_start == 1)
      active_region = true;
    else {
      if (active_from_start)
        active_region = false;
    }
    
    if(user_defined_folders == 1)
    {
      const char* usr_folder_env = std::getenv("TRACES_FOLDER");
      if (usr_folder_env) {
        std::string usr_folder = usr_folder_env;
        traces_path = usr_folder;
        extrainfo_path = traces_path + "/extra_info";
        tma_htod_blob_path = extrainfo_path + "/tma_htod_blobs";
        cubin_path = extrainfo_path + "/cubin";
        sass_path = extrainfo_path + "/sass";
        register_usage_path = extrainfo_path + "/register_usage";
        threadblock_trace_path = traces_path + "/threadblocks";
        threadblock_register_values_path = traces_path + "/threadblocks/register_values";

        traces_location = usr_folder;
        stats_location = usr_folder + "/stats.csv";
        printf("\n Traces location is %s \n", traces_location.c_str());
        printf("Stats location is %s \n", stats_location.c_str());
      }
    }

    create_folder(traces_path.c_str());
    create_folder(extrainfo_path.c_str());
    if (enable_tma_desc) {
      create_folder(tma_htod_blob_path.c_str());
      create_folder(tensor_map_blob_path.c_str());
      std::ofstream tma_dump_ofs(extrainfo_path + "/tma_htod_dump.csv", std::ios::out);
      if (tma_dump_ofs.is_open()) {
        tma_dump_ofs << "dump_id,device_id,stream_key,dst_device_hex,byte_count,preview_hex,blob_path\n";
      }
      std::ofstream tensor_map_dump_ofs(extrainfo_path + "/tensor_map_encode_dump.csv", std::ios::out);
      if (tensor_map_dump_ofs.is_open()) {
        tensor_map_dump_ofs << "dump_id,device_id,tensor_map_ptr_hex,global_address_hex,tensor_data_type,tensor_rank,global_dim,global_strides,box_dim,element_strides,interleave,swizzle,l2_promotion,oob_fill,qword0_hex,qword1_hex,qword2_hex,qword3_hex,qword4_hex,qword5_hex,qword6_hex,qword7_hex,blob_path\n";
      }
    }
    if (enable_tma_desc) {
      std::ofstream ofs(extrainfo_path + "/tma_desc_runtime_debug.csv", std::ios::out);
      if (ofs.is_open()) {
        ofs << "device_id,stream_id,kernel_id,unique_function_id,pc_hex,cta_x,cta_y,cta_z,warp_id_tb,sm_id,active_mask,predicate_mask,desc_reg_id,desc_value_lo,desc_value_hi,first_lane_addr\n";
      }
      std::ofstream producer_ofs(extrainfo_path + "/tma_desc_producer_debug.csv", std::ios::out);
      if (producer_ofs.is_open()) {
        producer_ofs << "device_id,stream_id,kernel_id,consumer_function_id,consumer_pc_hex,consumer_desc_reg_id,consumer_desc_value_lo,consumer_desc_value_hi,producer_function_id,producer_pc_hex,producer_dest_ureg_id,producer_pre_value_lo,producer_pre_value_hi,producer_inst_text\n";
      }
      std::ofstream operand_jsonl_ofs(extrainfo_path + "/tma_runtime_operand_debug.jsonl", std::ios::out);
      operand_jsonl_ofs.close();
    }

    statsFile = fopen(stats_location.c_str(), "w");
    fprintf(statsFile,
            "device_id, stream_id, kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, "
            "#blocks, block_dimX, block_dimY, block_dimZ, #threads, "
            "total_insts, total_reported_insts\n");
    fclose(statsFile);
  }

  if (cbid == API_CUDA_cuMemcpyHtoD_v2) {
    if (!is_exit) {
      cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;
      uint64_t stream_key = 0;// Memcpy unless they are asynchronous, they are always in stream 0.
      char buffer[1024];
      int device_id;
      cuCtxGetDevice(&device_id);
      append_tma_memcpy_dump_event(device_id, stream_key, p->dstDevice, p->srcHost, p->ByteCount);
      sprintf(buffer, "MemcpyHtoD,0x%016llx,%lu", p->dstDevice, p->ByteCount);
      dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[device_id];
      dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
      stream.add_ordered_cuda_events(buffer);
    }

#if defined(API_CUDA_cuMemcpyHtoDAsync_v2)
  } else if (cbid == API_CUDA_cuMemcpyHtoDAsync_v2
#if defined(API_CUDA_cuMemcpyHtoDAsync_v2_ptsz)
             || cbid == API_CUDA_cuMemcpyHtoDAsync_v2_ptsz
#endif
  ) {
    if (!is_exit) {
      cuMemcpyHtoDAsync_v2_params_proxy *p =
          (cuMemcpyHtoDAsync_v2_params_proxy *)params;
      uint64_t stream_key = (uint64_t)p->hStream;
      char buffer[1024];
      int device_id;
      cuCtxGetDevice(&device_id);
      append_tma_memcpy_dump_event(device_id, stream_key, p->dstDevice, p->srcHost, p->ByteCount);
      sprintf(buffer, "MemcpyHtoDAsync,0x%016llx,%lu", p->dstDevice, p->ByteCount);
      dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[device_id];
      dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
      stream.add_ordered_cuda_events(buffer);
    }
#endif

  } else if (
#if defined(API_CUDA_cuTensorMapEncodeTiled)
             cbid == API_CUDA_cuTensorMapEncodeTiled ||
#endif
             (name != nullptr && std::string(name) == "cuTensorMapEncodeTiled")) {
    if (is_exit) {
      cuTensorMapEncodeTiled_params *p = (cuTensorMapEncodeTiled_params *)params;
      int device_id;
      cuCtxGetDevice(&device_id);
      append_tensor_map_encode_dump_event(device_id, p);
    }

  } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
             cbid == API_CUDA_cuLaunchKernel) {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

    if (!is_exit) {
      int device_id;
      cuCtxGetDevice(&device_id);
      ctx_state->received_records_per_kernel.store(
          0, std::memory_order_relaxed);
      ctx_state->sparse_retained_instructions_per_kernel.store(
          0, std::memory_order_relaxed);
      ctx_state->sparse_reconstructed_instructions_per_kernel.store(
          0, std::memory_order_relaxed);
      if (active_from_start && dynamic_kernel_limit_start && kernel_id[device_id] == dynamic_kernel_limit_start)
        active_region = true;

      if (terminate_after_limit_number_of_kernels_reached && dynamic_kernel_limit_end != 0 && kernel_id[device_id] > dynamic_kernel_limit_end)
      {
        pthread_mutex_unlock(&mutex);
        exit(0);
      }
      
      int nregs = 0;
      CUDA_SAFECALL(
          cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

      int shmem_static_nbytes = 0;
      CUDA_SAFECALL(cuFuncGetAttribute(
          &shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

      CUDA_SAFECALL(cuFuncGetAttribute(&binary_version,
                                       CU_FUNC_ATTRIBUTE_BINARY_VERSION, p->f));

      get_opcode_map(OpcodeMap, binary_version);
      instrument_function_if_needed(ctx, p->f, device_id);

      if (active_region) {
        nvbit_enable_instrumented(ctx, p->f, true);
        stop_report[device_id] = false;
      } else {
        nvbit_enable_instrumented(ctx, p->f, false);
        stop_report[device_id] = true;
      }

      char buffer[1024];
      sprintf(buffer, std::string(traces_location+"/kernel-%d.trace").c_str(), kernel_id[device_id]);
      dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[device_id];

      if (!stop_report[device_id]) {
        int variant_id = 0;
        auto it_map_already_instrumented = map_function_to_kernel_name_and_variant_id.find(p->f);
        assert(it_map_already_instrumented != map_function_to_kernel_name_and_variant_id.end());
        variant_id = std::get<1>(it_map_already_instrumented->second);
        std::string kernel_name(nvbit_get_func_name(ctx, p->f, true));
        if (warp_lifetime_only || warp_lifetime_sidecar) {
          reset_warp_lifetime_capture(kernel_id[device_id],
                                      kernel_name.c_str(), p);
        }

        // std::vector<int> kernel_argument_sizes = nvbit_get_kernel_argument_sizes(p->f);
        
        std::string final_kernel_name = kernel_name + variant_delimiter_str + std::to_string(variant_id);
        
        auto it_map_unique_function_id = map_function_name_to_unique_function_id_with_variant.find(final_kernel_name);
        assert(it_map_unique_function_id != map_function_name_to_unique_function_id_with_variant.end());
        dyn_trace.set_binary_version(binary_version);
        auto stream_map = gpu_dev.streams();
        uint64_t stream_key = (uint64_t)p->hStream;
        stream_key = 0; // We use 0 as stream key for now, as we do not support multiple streams yet. WIPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
        current_stream_id[device_id] = stream_key;
        if (mbarrier_contract_capture_enabled &&
            mbarrier_contract_capture_launches.count(p->f) != 0) {
          acquire_mbarrier_contract_launch(
              ctx_state, device_id, kernel_id[device_id],
              static_cast<int>(stream_key), p);
        }
        dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
        stream.set_id(stream_key);
        dynamic_trace::kernel *ker = stream.add_kernels();
        ker->set_id(kernel_id[device_id]);
        ker->set_name(final_kernel_name);
        ker->set_function_unique_id(it_map_unique_function_id->second);
        /* Path A (§2.23): dump the host launch-arg buffer once per launch (host reads
         * only, cannot fault). Joined offline to (uid,pc) by the static tensormap_offset. */
        if (enable_tma_desc) {
          append_tma_launch_param_dump_event(ctx, device_id, kernel_id[device_id],
                                             it_map_unique_function_id->second, p);
        }
        ker->set_size_shared_memory(shmem_static_nbytes + p->sharedMemBytes);
        ker->set_number_of_registers(nregs);
        ker->set_shared_memory_base_address((uint64_t)nvbit_get_shmem_base_addr(ctx));
        ker->set_local_memory_base_address((uint64_t)nvbit_get_local_mem_base_addr(ctx));
        dynamic_trace::dim3d *grid_dim = ker->mutable_grid_dim();
        grid_dim->set_x(p->gridDimX);
        grid_dim->set_y(p->gridDimY);
        grid_dim->set_z(p->gridDimZ);
        dynamic_trace::dim3d *block_dim = ker->mutable_block_dim();
        block_dim->set_x(p->blockDimX);
        block_dim->set_y(p->blockDimY);
        block_dim->set_z(p->blockDimZ);
      }

      // This will be a relative path to the traces file
      sprintf(buffer,"kernel-%d.trace", kernel_id[device_id]);
      if (!stop_report[device_id]) {
        uint64_t stream_key = (uint64_t)p->hStream;
        stream_key = 0; // We use 0 as stream key for now, as we do not support multiple streams yet. WIPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
        dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
        stream.add_ordered_cuda_events(buffer);
      }

      statsFile = fopen(stats_location.c_str(), "a");
      unsigned blocks = p->gridDimX * p->gridDimY * p->gridDimZ;
      unsigned threads = p->blockDimX * p->blockDimY * p->blockDimZ;

      fprintf(statsFile, "%d, %d, %s, %s, %d, %d, %d, %d, %d, %d, %d, %d, ", device_id , current_stream_id[device_id], buffer,
              nvbit_get_func_name(ctx, p->f, true), p->gridDimX, p->gridDimY,
              p->gridDimZ, blocks, p->blockDimX, p->blockDimY, p->blockDimZ,
              threads);

      fclose(statsFile);

      kernel_id[device_id]++;
      recv_thread_receiving[ctx] = true;

    } else {
      int device_id;
      cuCtxGetDevice(&device_id);
      /* make sure current kernel is completed */
      cudaDeviceSynchronize();
      // GET CUDA ERROR:
      cudaError_t cuErr = cudaGetLastError();
      if (cuErr != cudaSuccess) {
        fprintf(stdout, "CUDA error in file '%s' in line %i : %s.\n", __FILE__,
                __LINE__, cudaGetErrorString(cuErr));
        fflush(stdout);
        abort();
      }
      assert(cuErr == cudaSuccess);

      if ((warp_lifetime_only || warp_lifetime_sidecar) &&
          !stop_report[device_id] &&
          warp_lifetime_capture_active &&
          warp_lifetime_kernel_id == kernel_id[device_id] - 1) {
        dump_warp_lifetime_capture();
      }

      /* make sure we prevent re-entry on the nvbit_callback when issuing
       * the flush_channel kernel */
      skip_callback_flag = true;

      /* issue flush of channel so we are sure all the memory accesses
       * have been pushed */
      flush_channel<<<1, 1>>>(ctx_state->channel_dev, compact_wire_enabled);
      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);

      /* unset the skip flag */
      skip_callback_flag = false;

      /* wait here until the receiving thread has not finished with the
       * current kernel */
      while (recv_thread_receiving[ctx]) {
        pthread_yield();
      }
      if (mbarrier_contract_capture_enabled &&
          ctx_state->mbarrier_contract_registry != nullptr &&
          ctx_state->mbarrier_contract_registry->active == 1) {
        release_mbarrier_contract_launch(ctx_state);
      }

      /* Phase 0b (TMA_BASE_ADDR.md §2.13): the kernel is fully done and the channel
       * is drained, so the captured param_base is populated and the kernel param
       * buffer is still alive. Deref the param region on the app thread here (never
       * from the recv thread). kernel_id was incremented at launch, so the just-
       * finished kernel is kernel_id-1. */
      if (enable_tma_desc && !stop_report[device_id]) {
        append_tma_param_base_deref_event(device_id, kernel_id[device_id] - 1);
        append_tma_desc_factcheck_event(device_id, kernel_id[device_id] - 1);
      }

      uint64_t total_insts_per_kernel =
          total_dynamic_instr_counter - old_total_insts;
      old_total_insts = total_dynamic_instr_counter;

      uint64_t reported_insts_per_kernel =
          reported_dynamic_instr_counter - old_total_reported_insts;
      old_total_reported_insts = reported_dynamic_instr_counter;

      uint64_t received_records =
          ctx_state->received_records_per_kernel.load(
              std::memory_order_acquire);
      if (lighter_tracer_host_count && !stop_report[device_id]) {
        total_insts_per_kernel = received_records;
        reported_insts_per_kernel = received_records;
      }
      if (lighter_tracer_host_count) {
        printf("[TRACE-COUNT] kernel=%d received_records=%llu "
               "stats_total=%llu stats_reported=%llu stop_report=%d "
               "source=%s\n",
               kernel_id[device_id] - 1,
               static_cast<unsigned long long>(received_records),
               static_cast<unsigned long long>(total_insts_per_kernel),
               static_cast<unsigned long long>(reported_insts_per_kernel),
               stop_report[device_id] ? 1 : 0,
               stop_report[device_id] ? "device-suppressed" : "host");
      }
      if (lighter_tracer_sparse_reconstruct) {
        uint64_t retained_instructions =
            ctx_state->sparse_retained_instructions_per_kernel.load(
                std::memory_order_acquire);
        uint64_t reconstructed_instructions =
            ctx_state->sparse_reconstructed_instructions_per_kernel.load(
                std::memory_order_acquire);
        printf("[TRACE-SPARSE] kernel=%d received_records=%llu "
               "retained_instructions=%llu reconstructed_instructions=%llu "
               "final_instructions=%llu stop_report=%d\n",
               kernel_id[device_id] - 1,
               static_cast<unsigned long long>(received_records),
               static_cast<unsigned long long>(retained_instructions),
               static_cast<unsigned long long>(reconstructed_instructions),
               static_cast<unsigned long long>(
                   retained_instructions + reconstructed_instructions),
               stop_report[device_id] ? 1 : 0);
      }

      statsFile = fopen(stats_location.c_str(), "a");
      fprintf(statsFile, "%llu,%llu",
              static_cast<unsigned long long>(total_insts_per_kernel),
              static_cast<unsigned long long>(reported_insts_per_kernel));
      fprintf(statsFile, "\n");
      fclose(statsFile);
      
      if (!stop_report[device_id]) {
        create_folder(threadblock_trace_path.c_str());
        create_folder(threadblock_register_values_path.c_str());
        auto it_tb = threadblocks.begin();
        while(it_tb != threadblocks.end()) {
          std::string tb_string_id = it_tb->first;
          ThreadblockStringParseInfo tb_info = parse_tb_string_id(tb_string_id);
          if((tb_info.device_id == device_id) && (tb_info.stream_id == current_stream_id[device_id]) && (tb_info.kernel_id == (kernel_id[device_id]-1))) {
            // Create hierarchical folder structure
            std::string device_folder = threadblock_trace_path + "/device_" + std::to_string(tb_info.device_id);
            std::string stream_folder = device_folder + "/stream_" + std::to_string(tb_info.stream_id);
            std::string kernel_folder = stream_folder + "/kernel_" + std::to_string(tb_info.kernel_id);

            create_folder(device_folder.c_str());
            create_folder(stream_folder.c_str());
            create_folder(kernel_folder.c_str());

            dynamic_trace::threadblock &tb = it_tb->second;

            // Use merge_and_write_threadblock to handle incremental flush parts
            // This will merge any part files with remaining in-memory data
            merge_and_write_threadblock(tb_string_id, tb, kernel_folder);
            tb.Clear();

            it_tb = threadblocks.erase(it_tb);
          }else {
            it_tb++;
          }
        }
      }
      if (active_from_start && dynamic_kernel_limit_end && kernel_id[device_id] > dynamic_kernel_limit_end)
        active_region = false;
    }
  } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
      if (!active_from_start) {
        active_region = true;
      }
  } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
      if (!active_from_start) {
        active_region = false;
      }
  }

  pthread_mutex_unlock(&mutex);
}

bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  return !s.empty() && it == s.end();
}

unsigned get_datawidth_from_opcode(const std::vector<std::string> &opcode) {
  for (unsigned i = 0; i < opcode.size(); ++i) {
    if (is_number(opcode[i])) {
      return (std::stoi(opcode[i], NULL) / 8);
    } else if (opcode[i][0] == 'U' && is_number(opcode[i].substr(1))) {
      // handle the U* case
      unsigned bits;
      sscanf(opcode[i].c_str(), "U%u", &bits);
      return bits / 8;
    }
  }

  return 4; // default is 4 bytes
}

bool check_opcode_contain(const std::vector<std::string> &opcode,
                          std::string param) {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param)
      return true;

  return false;
}

bool base_stride_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                          uint64_t &base_addr, int &stride) {

  // calulcate the difference between addresses
  // write cosnsctive addresses with constant stride in a more
  // compressed way (i.e. start adress and stride)
  bool const_stride = true;
  bool first_bit1_found = false;
  bool last_bit1_found = false;

  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      first_bit1_found = true;
      base_addr = addrs[s];
      if (s < 31 && mask.test(s + 1))
        stride = addrs[s + 1] - addrs[s];
      else {
        const_stride = false;
        break;
      }
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask.test(s)) {
        int diff_addr = addrs[s] - addrs[s - 1];
        if (stride != diff_addr) {
          const_stride = false;
          break;
        }
      } else
        last_bit1_found = true;
    } else if (last_bit1_found) {
      if (mask.test(s)) {
        const_stride = false;
        break;
      }
    }
  }

  return const_stride;
}

void base_delta_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                         uint64_t &base_addr, std::vector<long long> &deltas) {

  // save the delta from the previous address
  bool first_bit1_found = false;
  uint64_t last_address = 0;
  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      base_addr = addrs[s];
      first_bit1_found = true;
      last_address = addrs[s];
    } else if (mask.test(s) && first_bit1_found) {
      deltas.push_back(addrs[s] - last_address);
      last_address = addrs[s];
    }
  }
}

static uint32_t validated_inst_trace_record_size(const char *data,
                                                 uint32_t available_bytes) {
  if (available_bytes < INST_TRACE_COMPACT_SIZE) {
    std::cerr << "[TRACE-WIRE] truncated compact record: available="
              << available_bytes << " required=" << INST_TRACE_COMPACT_SIZE
              << std::endl;
    abort();
  }

  const inst_trace_t *record =
      reinterpret_cast<const inst_trace_t *>(data);
  if (record->record_version != INST_TRACE_RECORD_VERSION) {
    std::cerr << "[TRACE-WIRE] unsupported record version "
              << record->record_version << std::endl;
    abort();
  }

  const uint16_t known_flags = INST_TRACE_HAS_LANE_VALUES;
  if ((record->record_flags & ~known_flags) != 0) {
    std::cerr << "[TRACE-WIRE] unknown record flags 0x" << std::hex
              << record->record_flags << std::dec << std::endl;
    abort();
  }

  const uint32_t expected_size =
      inst_trace_has_lane_values(record) ? INST_TRACE_MEMORY_SIZE
                                         : INST_TRACE_COMPACT_SIZE;
  if (record->record_size != expected_size ||
      record->record_size > available_bytes) {
    std::cerr << "[TRACE-WIRE] invalid record size=" << record->record_size
              << " expected=" << expected_size
              << " available=" << available_bytes << std::endl;
    abort();
  }
  if (record->cta_id_x != -1) {
    const bool memory_record = record->mem_type != MEM_TYPE::NONE;
    if (memory_record != inst_trace_has_lane_values(record)) {
      std::cerr << "[TRACE-WIRE] payload/type mismatch: mem_type="
                << static_cast<int>(record->mem_type)
                << " flags=0x" << std::hex << record->record_flags
                << std::dec << std::endl;
      abort();
    }
  }
  return record->record_size;
}

void *recv_thread_fun(void *args) {
  CUcontext ctx = (CUcontext)args;
  pthread_mutex_lock(&mutex);
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];
  CUcontext current_ctx;
  int device_id;
  cuCtxGetCurrent(&current_ctx);
  cuCtxSetCurrent(ctx);
  cuCtxGetDevice(&device_id);
  cuCtxSetCurrent(current_ctx);
  ChannelHost* ch_host = &ctx_state->channel_host;
  dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[device_id];
  gpu_dev.set_id(device_id);
  pthread_mutex_unlock(&mutex);

  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);
  uint64_t legacy_records = 0;
  uint64_t compact_records = 0;
  uint64_t memory_records = 0;
  uint64_t wire_bytes = 0;
  uint64_t kernel_received_records = 0;
  uint64_t kernel_sparse_retained_instructions = 0;
  uint64_t kernel_sparse_reconstructed_instructions = 0;

  while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
    if (num_recv_bytes > 0) {
      if (warp_lifetime_only) {
        uint32_t offset = 0;
        while (offset < num_recv_bytes) {
          uint32_t record_size = 0;
          int cta_id_x = 0;
          if (compact_wire_enabled) {
            record_size = validated_inst_trace_record_size(
                &recv_buffer[offset], num_recv_bytes - offset);
            const inst_trace_t *token =
                reinterpret_cast<const inst_trace_t *>(&recv_buffer[offset]);
            cta_id_x = token->cta_id_x;
          } else {
            if (num_recv_bytes - offset < INST_TRACE_LEGACY_SIZE) {
              std::cerr << "[TRACE-WIRE] truncated legacy record"
                        << std::endl;
              abort();
            }
            record_size = INST_TRACE_LEGACY_SIZE;
            const legacy_inst_trace_t *token =
                reinterpret_cast<const legacy_inst_trace_t *>(
                    &recv_buffer[offset]);
            cta_id_x = token->cta_id_x;
          }
          if (cta_id_x == -1) {
            ctx_state->received_records_per_kernel.store(
                0, std::memory_order_release);
            ctx_state->sparse_retained_instructions_per_kernel.store(
                0, std::memory_order_release);
            ctx_state->sparse_reconstructed_instructions_per_kernel.store(
                0, std::memory_order_release);
            recv_thread_receiving[ctx] = false;
            break;
          }
          offset += record_size;
        }
        continue;
      }
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        uint32_t record_size = 0;
        inst_trace_t normalized_record;
        inst_trace_t *ma = nullptr;
        if (compact_wire_enabled) {
          record_size = validated_inst_trace_record_size(
              &recv_buffer[num_processed_bytes],
              num_recv_bytes - num_processed_bytes);
          ma = reinterpret_cast<inst_trace_t *>(
              &recv_buffer[num_processed_bytes]);
        } else {
          if (num_recv_bytes - num_processed_bytes <
              INST_TRACE_LEGACY_SIZE) {
            std::cerr << "[TRACE-WIRE] truncated legacy record"
                      << std::endl;
            abort();
          }
          const legacy_inst_trace_t *legacy =
              reinterpret_cast<const legacy_inst_trace_t *>(
                  &recv_buffer[num_processed_bytes]);
          normalize_legacy_inst_trace(legacy, &normalized_record);
          ma = &normalized_record;
          record_size = INST_TRACE_LEGACY_SIZE;
        }

        /* when we receive a CTA_id_x it means all the kernels
        * completed, this is the special token we receive from the
        * flush channel kernel that is issues at the end of the
        * context */
        if (ma->cta_id_x == -1) {
          ctx_state->received_records_per_kernel.store(
              kernel_received_records, std::memory_order_release);
          ctx_state->sparse_retained_instructions_per_kernel.store(
              kernel_sparse_retained_instructions,
              std::memory_order_release);
          ctx_state->sparse_reconstructed_instructions_per_kernel.store(
              kernel_sparse_reconstructed_instructions,
              std::memory_order_release);
          kernel_received_records = 0;
          kernel_sparse_retained_instructions = 0;
          kernel_sparse_reconstructed_instructions = 0;
          recv_thread_receiving[ctx] = false;
          break;
        }
        if (lighter_tracer_host_count ||
            lighter_tracer_sparse_reconstruct) {
          kernel_received_records++;
        }
        wire_bytes += record_size;
        if (!compact_wire_enabled) {
          legacy_records++;
        } else if (inst_trace_has_lane_values(ma)) {
          memory_records++;
        } else {
          compact_records++;
        }
        while(!recv_thread_receiving[ctx]) {
          pthread_yield();
        }
        // Key: d_{device}_s_{stream}_k_{kernel}_{cta_id_x},{cta_id_y},{cta_id_z}
        std::string tb_string_id = "d_" + std::to_string(device_id) + "_s_" + std::to_string(current_stream_id[device_id]) + "_k_" + std::to_string(kernel_id[device_id]-1) + "_" + std::to_string(ma->cta_id_x) + "," +
                                   std::to_string(ma->cta_id_y) + "," +
                                   std::to_string(ma->cta_id_z);
        dynamic_trace::threadblock &tb = get_threadblock(tb_string_id);
        dynamic_trace::dim3d *cta_id = tb.mutable_block_id();
        cta_id->set_x(ma->cta_id_x);
        cta_id->set_y(ma->cta_id_y);
        cta_id->set_z(ma->cta_id_z);
        int warp_id_tb = ma->warpid_tb;
        dynamic_trace::warp &wp = (*tb.mutable_warps())[warp_id_tb];
        wp.set_id(warp_id_tb);
        // if (print_core_id) {
        //   tb.set_sm_id(ma->sm_id);
        //   wp.set_warp_id_in_sm(ma->warpid_sm);
        // }
        int opcode_id = pc_to_opcode[ma->unique_function_id][ma->vpc];
        unsigned int &remaining_injects = get_remaining_injects_to_current_instruction(tb_string_id, warp_id_tb);
        dynamic_trace::instruction *inst;
        if(remaining_injects == 0) {
          if (lighter_tracer_sparse_reconstruct) {
            auto function_it =
                sparse_prefix_pcs_by_function.find(
                    ma->unique_function_id);
            if (function_it != sparse_prefix_pcs_by_function.end()) {
              auto span_it = function_it->second.find(ma->vpc);
              if (span_it != function_it->second.end()) {
                for (uint32_t omitted_pc : span_it->second) {
                  dynamic_trace::instruction *reconstructed =
                      wp.add_instructions();
                  reconstructed->set_pc(omitted_pc);
                  reconstructed->set_active_mask(ma->active_mask);
                  reconstructed->set_predicate_mask(ma->active_mask);
                  reconstructed->set_function_unique_id(
                      ma->unique_function_id);
                }
                kernel_sparse_reconstructed_instructions +=
                    span_it->second.size();
              }
            }
            kernel_sparse_retained_instructions++;
          }
          inst = wp.add_instructions();
          inst->set_pc(ma->vpc);
          inst->set_active_mask(ma->active_mask);
          inst->set_predicate_mask(ma->predicate_mask);
          inst->set_function_unique_id(ma->unique_function_id);
          remaining_injects = ma->num_of_injects;
        }else {
          inst = wp.mutable_instructions(wp.instructions_size()-1);
        }
        remaining_injects--;
        unsigned int callback_index = ma->num_of_injects - remaining_injects - 1;
        auto opcode_it = id_to_opcode_map.find(opcode_id);
        const std::string opcode = opcode_it == id_to_opcode_map.end() ? "" : opcode_it->second;
        auto sync_function_it =
            sync_runtime_capture_sites_by_pc.find(ma->unique_function_id);
        if (sync_function_it != sync_runtime_capture_sites_by_pc.end()) {
          auto sync_pc_it = sync_function_it->second.find(ma->vpc);
          if (sync_pc_it != sync_function_it->second.end()) {
            const sync_runtime_capture_site_info &sync_site = sync_pc_it->second;
            int first_lane =
                get_first_predicated_lane(ma->active_mask, ma->predicate_mask);
            if (first_lane >= 0) {
              if (callback_index ==
                      static_cast<unsigned int>(sync_site.barrier_callback_index) &&
                  ma->mem_type != MEM_TYPE::NONE) {
                dynamic_trace::instruction::sync_runtime_info *sync =
                    inst->mutable_sync();
                sync->set_valid(true);
                sync->set_barrier_addr(
                    inst_trace_value_0(ma, first_lane));
                if (sync_site.semantic_is_zero_literal) {
                  sync->set_has_semantic_raw(true);
                  sync->set_semantic_raw(0);
                }
              } else if (callback_index ==
                             static_cast<unsigned int>(
                                 sync_site.semantic_callback_index) &&
                         ma->mem_type == MEM_TYPE::NONE) {
                dynamic_trace::instruction::sync_runtime_info *sync =
                    inst->mutable_sync();
                sync->set_valid(true);
                sync->set_has_semantic_raw(true);
                sync->set_semantic_raw(
                    inst_trace_value_0(ma, first_lane));
              }
            }
          }
        }
        auto bar_function_it =
            bar_runtime_capture_sites_by_pc.find(ma->unique_function_id);
        if (bar_function_it != bar_runtime_capture_sites_by_pc.end()) {
          auto bar_pc_it = bar_function_it->second.find(ma->vpc);
          if (bar_pc_it != bar_function_it->second.end()) {
            const bar_runtime_capture_site_info &bar_site = bar_pc_it->second;
            // Register-form barrier operands take the REGULAR callback path, where the
            // live register value lands in scalar_value_0 with mem_type==NONE.
            // Pick the first predicated-active lane: if the warp did not actually execute
            // this BAR (no active+predicated lane), nothing is captured for this instance.
            int first_lane =
                get_first_predicated_lane(ma->active_mask, ma->predicate_mask);
            if (first_lane >= 0 && ma->mem_type == MEM_TYPE::NONE) {
              if (bar_site.id_callback_index >= 0 &&
                  callback_index ==
                      static_cast<unsigned int>(bar_site.id_callback_index)) {
                dynamic_trace::instruction::bar_runtime_info *bar =
                    inst->mutable_bar_runtime();
                bar->set_valid(true);
                bar->set_has_id(true);
                bar->set_id(static_cast<uint32_t>(
                    inst_trace_value_0(ma, first_lane)));
              } else if (bar_site.count_callback_index >= 0 &&
                         callback_index ==
                             static_cast<unsigned int>(
                                 bar_site.count_callback_index)) {
                dynamic_trace::instruction::bar_runtime_info *bar =
                    inst->mutable_bar_runtime();
                bar->set_valid(true);
                bar->set_has_count(true);
                bar->set_count(static_cast<uint32_t>(
                    inst_trace_value_0(ma, first_lane)));
              }
            }
          }
        }
        auto tma_completion_function_it =
            tma_completion_runtime_capture_sites_by_pc.find(
                ma->unique_function_id);
        if (tma_completion_function_it !=
            tma_completion_runtime_capture_sites_by_pc.end()) {
          auto tma_completion_pc_it =
              tma_completion_function_it->second.find(ma->vpc);
          if (tma_completion_pc_it !=
              tma_completion_function_it->second.end()) {
            const tma_completion_runtime_capture_site_info &tma_site =
                tma_completion_pc_it->second;
            int first_lane =
                get_first_predicated_lane(ma->active_mask,
                                          ma->predicate_mask);
            if (first_lane >= 0 && ma->mem_type == MEM_TYPE::NONE &&
                callback_index == static_cast<unsigned int>(
                                      tma_site.barrier_callback_index)) {
              dynamic_trace::instruction::tma_completion_runtime_info *tma =
                  inst->mutable_tma_completion();
              tma->set_valid(true);
              tma->set_barrier_addr(
                  inst_trace_value_0(ma, first_lane));
              tma->set_barrier_ureg_id(tma_site.barrier_ureg_id);
            }
          }
        }
        if (tma_consumer_opcode_ids.count(opcode_id)) {
          append_tma_runtime_operand_debug_event(device_id, current_stream_id[device_id],
                                                 kernel_id[device_id] - 1, ma, opcode,
                                                 callback_index);
        }
        std::bitset<32> mask(ma->active_mask & ma->predicate_mask);
        if (ma->mem_type == MEM_TYPE::NONE && callback_index == 0 &&
            (ma->per_operand_type == TRACED_REG_TYPE::UNIFORM ||
             ma->per_operand_type == TRACED_REG_TYPE::UNIFORM_2_REGS)) {
          auto function_it = first_dest_ureg_by_pc.find(ma->unique_function_id);
          if (function_it != first_dest_ureg_by_pc.end()) {
            auto pc_it = function_it->second.find(ma->vpc);
            if (pc_it != function_it->second.end()) {
              std::string recent_key = get_tma_desc_producer_debug_key(tb_string_id, warp_id_tb);
              auto &recent_queue = recent_tma_desc_producers_by_warp[recent_key];
              tma_desc_producer_candidate_t candidate;
              candidate.unique_function_id = ma->unique_function_id;
              candidate.pc = ma->vpc;
              candidate.dest_ureg_id = pc_it->second;
              candidate.pre_value_lo =
                  static_cast<uint32_t>(ma->scalar_value_0);
              candidate.pre_value_hi =
                  (ma->per_operand_type == TRACED_REG_TYPE::UNIFORM_2_REGS)
                      ? ma->scalar_value_1
                      : SECRET_UREG_DESC_NOT_USED;
              recent_queue.push_back(candidate);
              if (recent_queue.size() > 16) {
                recent_queue.pop_front();
              }
            }
          }
        }
        if (ma->mem_type != MEM_TYPE::NONE) {
          dynamic_trace::address *addr = inst->add_addresses();
          std::istringstream iss(id_to_opcode_map[opcode_id]);
          std::vector<std::string> tokens;
          std::string token;
          addr->set_udesc_value(ma->ureg_desc_value);
          addr->set_udesc_value_hi(ma->ureg_desc_value_hi);
          if(ma->mem_type == MEM_TYPE::CALL_OR_RET) {
            addr->set_data_width(1);
          }else {
            while (std::getline(iss, token, '.'))
            {
              if (!token.empty())
                tokens.push_back(token);
            }
            addr->set_data_width(get_datawidth_from_opcode(tokens));
          }

          bool base_stride_success = false;
          uint64_t base_addr = 0;
          int stride = 0;
          std::vector<long long> deltas;

          if (enable_compress) {
            // try base+stride format
            base_stride_success =
                base_stride_compress(ma->lane_values, mask, base_addr,
                                     stride);
            if (!base_stride_success) {
              // if base+stride fails, try base+delta format
              base_delta_compress(ma->lane_values, mask, base_addr, deltas);
            }
          }

          if (base_stride_success && enable_compress) {
            // base + stride format
            addr->set_compression_format(address_format::base_stride);
            addr->set_base_address(base_addr);
            addr->set_stride(stride);
          } else if (!base_stride_success && enable_compress) {
            // base + delta format
            addr->set_compression_format(address_format::base_delta);
            addr->set_base_address(base_addr);
            for (unsigned int s = 0; s < deltas.size(); s++) {
              addr->add_addrs(deltas[s]);
            }
          } else {
            // list all the addresses
            addr->set_compression_format(address_format::list_all);
            for (int s = 0; s < 32; s++) {
              if (mask.test(s))
                addr->add_addrs(ma->lane_values[s]);
            }
          }
          if (ma->mem_type == MEM_TYPE::STANDARD_MEM && enable_tma_desc &&
              ma->ureg_desc_id != SECRET_UREG_DESC_NOT_USED) {
            append_tma_desc_runtime_debug_event(device_id, current_stream_id[device_id],
                                                kernel_id[device_id] - 1, ma);
            append_tma_desc_producer_debug_event(device_id, current_stream_id[device_id],
                                                 kernel_id[device_id] - 1, ma, tb_string_id,
                                                 warp_id_tb);
          }
        }

        // =============================================================================
        // Incremental Flush Check: Flush threadblock if instruction count exceeds threshold
        // =============================================================================
        if (incremental_flush_threshold > 0) {
          // Increment instruction count for this threadblock
          tb_instruction_count[tb_string_id]++;

          // Check if we need to flush
          if (tb_instruction_count[tb_string_id] >= (uint64_t)incremental_flush_threshold) {
            // Ensure trace folder structure exists
            create_folder(threadblock_trace_path.c_str());

            // Flush this threadblock immediately
            int kern_id = kernel_id[device_id] - 1;  // kernel_id is already incremented
            int stream_id = current_stream_id[device_id];
            flush_threadblock_immediate(tb_string_id, tb, device_id, stream_id, kern_id);

            // Note: tb is now cleared but still in map - subsequent instructions
            // will add to fresh TB, and final flush at kernel end will write remainder
          }
        }
        // =============================================================================

        num_processed_bytes += record_size;
      }
    }
  }
  free(recv_buffer);

  const uint64_t total_records =
      legacy_records + compact_records + memory_records;
  const uint64_t legacy_bytes =
      total_records * static_cast<uint64_t>(INST_TRACE_LEGACY_SIZE);
  const double reduction_pct =
      legacy_bytes == 0
          ? 0.0
          : 100.0 * (1.0 - static_cast<double>(wire_bytes) /
                               static_cast<double>(legacy_bytes));
  printf("[TRACE-WIRE] mode=%s legacy_records=%" PRIu64
         " compact_records=%" PRIu64
         " memory_records=%" PRIu64 " wire_bytes=%" PRIu64
         " legacy_bytes=%" PRIu64 " reduction=%.2f%%\n",
         compact_wire_enabled ? "compact" : "legacy", legacy_records,
         compact_records, memory_records, wire_bytes, legacy_bytes,
         reduction_pct);

  // Print incremental flush stats at thread end
  if (incremental_flush_threshold > 0 && total_incremental_flushes > 0) {
    printf("[INCREMENTAL_FLUSH] Stats: %lu flushes, %lu instructions flushed to disk during kernel execution\n",
           total_incremental_flushes, total_instructions_flushed);
  }

  ctx_state->recv_thread_done = RecvThreadState::FINISHED;
  return NULL;
}

int check_system_call(int system_res, const char* syscall) {
  if(system_res != 0) {
    std::cout << "Error. System call failed. System call: " << syscall << std::endl;
    fflush(stdout);
    std::string str_error(syscall);
    if(str_error.find("nvdisasm -lrm count") == std::string::npos) {
      abort();
    }
  }
  return system_res;
}

static std::string get_basename(const std::string &path) {
  const std::size_t pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

static std::string to_lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

static std::vector<std::string> get_loaded_shared_objects() {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  std::unordered_set<std::string> seen;
  std::vector<std::string> out;
  while (std::getline(maps, line)) {
    const std::size_t pos = line.find('/');
    if (pos == std::string::npos) {
      continue;
    }
    std::string path = line.substr(pos);
    if (path.find(".so") == std::string::npos) {
      continue;
    }
    if (seen.insert(path).second) {
      out.push_back(path);
    }
  }
  return out;
}

static int score_cubin_candidate(const std::string &path, bool include_torch_libs) {
  const std::string lower = to_lower_ascii(path);
  const bool is_libtorch_cuda = (lower.find("libtorch_cuda") != std::string::npos);
  if (!include_torch_libs) {
    if (lower.find("libc10") != std::string::npos) {
      return -1;
    }
    if (!is_libtorch_cuda && lower.find("libtorch") != std::string::npos) {
      return -1;
    }
  }
  if (lower.find("flash_attn_3/_c") != std::string::npos || lower.find("flash_attn_3") != std::string::npos) {
    return 120;
  }
  if (lower.find("flash_attn") != std::string::npos || lower.find("flashattn") != std::string::npos) {
    return 100;
  }
  if (lower.find("cutlass") != std::string::npos) {
    return 90;
  }
  if (is_libtorch_cuda) {
    return 70;
  }
  if (lower.find("libtorch") != std::string::npos) {
    return 60;
  }
  if (lower.find("cudnn") != std::string::npos) {
    return 50;
  }
  if (lower.find("cuda") != std::string::npos) {
    return 10;
  }
  return 0;
}

static bool extract_cubin_from_binary(const std::string &binary_path, unsigned int binary_version,
                                     const std::string &out_dir) {
  std::filesystem::remove_all(out_dir);
  create_folder(out_dir.c_str());
  std::unordered_map<int, std::string> elf_by_index;
  {
    std::string cmd = "cuobjdump -lelf -arch=sm_" + std::to_string(binary_version) + " " + binary_path + " 2>/dev/null";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (pipe) {
      char buf[8192];
      while (fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        if (line.find("ELF file") == std::string::npos) {
          continue;
        }
        std::size_t pos_colon = line.find(':');
        if (pos_colon == std::string::npos) {
          continue;
        }
        std::string left = line.substr(0, pos_colon);
        std::string right = line.substr(pos_colon + 1);
        int idx = 0;
        try {
          std::size_t pos_file = left.find("file");
          std::string num = left.substr(pos_file + 4);
          idx = std::stoi(num);
        } catch (...) {
          continue;
        }
        right.erase(0, right.find_first_not_of(" \t\r\n"));
        right.erase(right.find_last_not_of(" \t\r\n") + 1);
        if (!right.empty()) {
          elf_by_index[idx] = right;
        }
      }
      pclose(pipe);
    }
  }

  const std::string bin_lower = to_lower_ascii(binary_path);
  const bool is_flashattn_binary = (bin_lower.find("flash_attn_3") != std::string::npos) ||
                                  (bin_lower.find("flash-attn") != std::string::npos);
  const char* fast_xelf_env = std::getenv("TRACER_FAST_XELF");
  const bool enable_fast_xelf = is_flashattn_binary || (fast_xelf_env && std::string(fast_xelf_env) == "1");
  if (enable_fast_xelf && !elf_by_index.empty()) {
    bool want_bwd = false;
    bool want_fwd = false;
    bool want_bf16 = false;
    bool want_fp16 = false;
    bool want_fp8 = false;
    bool want_hdim64 = false;
    bool want_hdim96 = false;
    bool want_hdim128 = false;
    bool want_hdim192 = false;
    bool want_hdim256 = false;
    for (const auto &kv : all_kernels_key_instructions_by_pc) {
      const std::string k = to_lower_ascii(kv.first);
      if (k.find("flashattnbwd") != std::string::npos || k.find("bwd") != std::string::npos) {
        want_bwd = true;
      }
      if (k.find("flashattnfwd") != std::string::npos || k.find("fwd") != std::string::npos) {
        want_fwd = true;
      }
      if (k.find("bfloat16") != std::string::npos || k.find("bf16") != std::string::npos) {
        want_bf16 = true;
      }
      if (k.find("half") != std::string::npos || k.find("fp16") != std::string::npos || k.find("f16") != std::string::npos) {
        want_fp16 = true;
      }
      if (k.find("e4m3") != std::string::npos || k.find("e5m2") != std::string::npos || k.find("fp8") != std::string::npos) {
        want_fp8 = true;
      }
      if (k.find("ili64ee") != std::string::npos) {
        want_hdim64 = true;
      }
      if (k.find("ili96ee") != std::string::npos) {
        want_hdim96 = true;
      }
      if (k.find("ili128ee") != std::string::npos) {
        want_hdim128 = true;
      }
      if (k.find("ili192ee") != std::string::npos) {
        want_hdim192 = true;
      }
      if (k.find("ili256ee") != std::string::npos) {
        want_hdim256 = true;
      }
    }

    std::vector<std::pair<int, std::string>> scored;
    scored.reserve(elf_by_index.size());
    std::string prefer_substr;
    const char* prefer_env = std::getenv("TRACER_XELF_PREFER_SUBSTR");
    if (prefer_env) {
      prefer_substr = to_lower_ascii(std::string(prefer_env));
    }
    if (prefer_substr.empty()) {
      if (want_hdim128) prefer_substr = "hdim128";
      else if (want_hdim64) prefer_substr = "hdim64";
      else if (want_hdim96) prefer_substr = "hdim96";
      else if (want_hdim192) prefer_substr = "hdim192";
      else if (want_hdim256) prefer_substr = "hdim256";
    }
    for (const auto &kv : elf_by_index) {
      const std::string name_lower = to_lower_ascii(kv.second);
      int score = 0;
      if (name_lower.find("flash_") != std::string::npos) score += 50;
      if (want_bwd && name_lower.find("bwd") != std::string::npos) score += 40;
      if (want_fwd && name_lower.find("fwd") != std::string::npos) score += 30;
      if (want_bf16 && name_lower.find("bf16") != std::string::npos) score += 30;
      if (want_fp16 && name_lower.find("fp16") != std::string::npos) score += 20;
      if (want_fp8 && (name_lower.find("e4m3") != std::string::npos || name_lower.find("e5m2") != std::string::npos)) score += 20;
      if (name_lower.find("sm90") != std::string::npos) score += 10;
      if (!prefer_substr.empty() && name_lower.find(prefer_substr) != std::string::npos) score += 1000;
      if (score > 0) {
        scored.emplace_back(score, kv.second);
      }
    }
    std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.first > b.first; });

    int max_extract = 8;
    const char* max_extract_env = std::getenv("TRACER_MAX_CUBIN_FILES");
    if (max_extract_env) {
      try {
        max_extract = std::max(1, std::stoi(max_extract_env));
      } catch (...) {
      }
    }
    int extracted = 0;
    std::vector<std::string> chosen;
    chosen.reserve(static_cast<size_t>(max_extract));
    std::unordered_set<std::string> chosen_set;

    auto pick_first = [&](auto pred) {
      for (const auto &kv : scored) {
        if (!pred(kv.second)) {
          continue;
        }
        if (chosen_set.insert(kv.second).second) {
          chosen.push_back(kv.second);
          return;
        }
      }
    };

    if (want_bwd && want_fwd && max_extract >= 2) {
      pick_first([&](const std::string &name) {
        const std::string nl = to_lower_ascii(name);
        return nl.find("bwd") != std::string::npos;
      });
      pick_first([&](const std::string &name) {
        const std::string nl = to_lower_ascii(name);
        return nl.find("fwd") != std::string::npos;
      });
    }

    for (const auto &kv : scored) {
      if (static_cast<int>(chosen.size()) >= max_extract) {
        break;
      }
      if (chosen_set.insert(kv.second).second) {
        chosen.push_back(kv.second);
      }
    }

    for (const auto &name : chosen) {
      std::string cmd = "cd " + out_dir + " && cuobjdump -xelf " + name + " -arch=sm_" +
                        std::to_string(binary_version) + " " + binary_path + " > /dev/null 2>&1";
      if (system(cmd.c_str()) == 0) {
        extracted++;
        if (extracted >= max_extract) break;
      }
    }
    if (!std::filesystem::is_empty(out_dir)) {
      return true;
    }
  }

  std::unordered_map<int, int> needed_index_scores;
  {
    std::unordered_set<std::string> kernel_names;
    kernel_names.reserve(all_kernels_key_instructions_by_pc.size() * 2 + 8);
    for (const auto &kv : all_kernels_key_instructions_by_pc) {
      if (kv.first.empty()) {
        continue;
      }
      kernel_names.insert(kv.first);
      const std::size_t pos = kv.first.rfind(variant_delimiter_str);
      if (pos != std::string::npos) {
        kernel_names.insert(kv.first.substr(0, pos));
      }
    }

    std::string cmd = "cuobjdump --dump-elf-symbols -arch=sm_" + std::to_string(binary_version) + " " + binary_path + " 2>/dev/null";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (pipe) {
      char buf[8192];
      int current_idx = 0;
      while (fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        if (line.find("Fatbin elf code") != std::string::npos) {
          current_idx++;
          continue;
        }
        if (current_idx == 0) {
          continue;
        }
        if (line.rfind("STT_", 0) != 0) {
          continue;
        }
        const std::size_t end = line.find_last_not_of(" \t\r\n");
        if (end == std::string::npos) {
          continue;
        }
        const std::size_t start = line.find_last_of(" \t", end);
        if (start == std::string::npos) {
          continue;
        }
        std::string sym = line.substr(start + 1, end - start);
        const int match_score = best_kernel_symbol_match_score(sym, kernel_names);
        if (match_score > 0) {
          auto it = needed_index_scores.find(current_idx);
          if (it == needed_index_scores.end()) {
            needed_index_scores[current_idx] = match_score;
          } else {
            it->second = std::max(it->second, match_score);
          }
        }
      }
      pclose(pipe);
    }
  }

  int extracted = 0;
  int max_extract = 32;
  const char* max_extract_env2 = std::getenv("TRACER_MAX_CUBIN_FILES");
  if (max_extract_env2) {
    try {
      max_extract = std::max(1, std::stoi(max_extract_env2));
    } catch (...) {
    }
  }
  std::vector<std::pair<int, int>> ranked_indices;
  ranked_indices.reserve(needed_index_scores.size());
  for (const auto &kv : needed_index_scores) {
    ranked_indices.emplace_back(kv.second, kv.first);
  }
  std::sort(ranked_indices.begin(), ranked_indices.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
  for (const auto &ranked_idx : ranked_indices) {
    auto it = elf_by_index.find(ranked_idx.second);
    if (it == elf_by_index.end()) {
      continue;
    }
    std::string cmd = "cd " + out_dir + " && cuobjdump -xelf " + it->second + " -arch=sm_" +
                      std::to_string(binary_version) + " " + binary_path + " > /dev/null 2>&1";
    if (system(cmd.c_str()) == 0) {
      extracted++;
      if (extracted >= max_extract) {
        break;
      }
    }
  }

  if (!std::filesystem::is_empty(out_dir)) {
    return true;
  }

  const char* allow_full = std::getenv("TRACER_ALLOW_FULL_XELF");
  if (allow_full && std::string(allow_full) == "1") {
    std::string cmd = "cd " + out_dir + " && cuobjdump " + binary_path + " -xelf all -arch=sm_" +
                      std::to_string(binary_version) + " > /dev/null 2>&1";
    if (system(cmd.c_str()) == 0 && !std::filesystem::is_empty(out_dir)) {
      return true;
    }
  }

  return false;
}

void enhanced_tracer() {
  create_folder(extrainfo_path.c_str());
  create_folder(cubin_path.c_str());
  create_folder(sass_path.c_str());
  create_folder(register_usage_path.c_str());
  std::string program_path = get_program_path();
  std::size_t found = program_path.find_last_of("/");
  std::string program_name = program_path.substr(found + 1);
  std::cout << "Generating extra information for the enhanced traces of benchmark: " << program_name << std::endl;
  const bool skip_rfu = (std::getenv("TRACER_SKIP_RFU") != nullptr) && (std::string(std::getenv("TRACER_SKIP_RFU")) == "1");
  bool include_torch_libs = false;
  const char* include_torch_env = std::getenv("TRACER_INCLUDE_TORCH_LIBS");
  if (include_torch_env && std::string(include_torch_env) == "1") {
    include_torch_libs = true;
  }
  std::vector<std::string> cubin_sources;
  const char* override_binary = std::getenv("TRACER_CUBIN_BINARY");
  if (override_binary && std::string(override_binary).size() > 0) {
    cubin_sources.push_back(std::string(override_binary));
  }
  cubin_sources.push_back(program_path);
  std::vector<std::string> loaded = get_loaded_shared_objects();
  std::stable_sort(loaded.begin(), loaded.end(),
                   [&](const std::string &a, const std::string &b) {
                     return score_cubin_candidate(a, include_torch_libs) >
                            score_cubin_candidate(b, include_torch_libs);
                   });
  for (const auto &p : loaded) {
    if (score_cubin_candidate(p, include_torch_libs) <= 0) {
      continue;
    }
    cubin_sources.push_back(p);
  }

  std::vector<std::string> extracted_cubin_dirs;
  std::unordered_set<std::string> attempted;
  int extracted = 0;
  int max_sources = 6;
  const char* max_sources_env = std::getenv("TRACER_MAX_CUBIN_SOURCES");
  if (max_sources_env) {
    try {
      max_sources = std::max(1, std::stoi(max_sources_env));
    } catch (...) {
    }
  }
  for (const auto &src : cubin_sources) {
    if (attempted.insert(src).second == false) {
      continue;
    }
    if (!std::filesystem::exists(src)) {
      continue;
    }
    std::string out_dir = cubin_path + "/src_" + std::to_string(extracted);
    if (extract_cubin_from_binary(src, binary_version, out_dir)) {
      extracted_cubin_dirs.push_back(out_dir);
      extracted++;
      if (extracted >= max_sources) {
        break;
      }
    }
  }
  if (extracted_cubin_dirs.empty()) {
    std::cout << "Warning. Failed to extract cubin from program binary, continuing with traced instructions only." << std::endl;
    fflush(stdout);
  }
  m_enhanced_traced_execution = new traced_execution(program_name);
  for (const auto &cubin_dir : extracted_cubin_dirs) {
    std::string absolute_cubin_path = make_abs_path(cubin_dir);
    for (const auto &entry : std::filesystem::directory_iterator(absolute_cubin_path)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      std::string aux_cubin = entry.path().filename().string();
      std::string base_name = entry.path().stem().string();
      std::string command_get_sass = "cd " + sass_path + " && cuobjdump -sass " + absolute_cubin_path + "/" + aux_cubin + " > " + base_name + ".sass";
      int call_code_sass = check_system_call(system(command_get_sass.c_str()), command_get_sass.c_str());
      if(call_code_sass == 0) {
        parsed_sass_result sass_result = parse_sass(binary_version, entry);
        if(!skip_rfu && sass_result.matched_kernels > 0) {
          std::string command_get_register_usage = "cd " + register_usage_path + " && nvdisasm -lrm count " + absolute_cubin_path + "/" + aux_cubin + " > " + base_name + ".rfu";
          int call_code_rfu = check_system_call(system(command_get_register_usage.c_str()), command_get_register_usage.c_str());
          if(call_code_rfu == 0) {
            parse_rfu(entry, sass_result.matched_kernel_names);
          }
        }
      }
    }
  }
  for(auto kernel_name : all_kernels_key_instructions_by_pc) {
    for(auto variant : kernel_name.second) {
      if(!variant.sass_has_been_parsed || (!skip_rfu && !variant.rfu_has_been_parsed)) {
        std::cout << "Error. Kernel " << kernel_name.first << " variant " << variant.variant_id << " has not been parsed." << std::endl;
        std::cout << "SASS parsed: " << variant.sass_has_been_parsed << std::endl;
        std::cout << "RFU parsed: " << variant.rfu_has_been_parsed << std::endl;
        std::cout << "Traced instructions: " << std::endl;
        print_map(variant.key_instructions_by_pc);
        auto it_already_captured_instr = map_func_addr_to_pc_to_sass_instr.find(variant.func_addr);
        assert(it_already_captured_instr != map_func_addr_to_pc_to_sass_instr.end());
        std::string kernel_name_to_add = variant.original_kernel_name + variant_delimiter_str + std::to_string(variant.variant_id);
        m_enhanced_traced_execution->add_no_binary_kernel(kernel_name_to_add, variant.unique_function_id, variant.func_addr, binary_version, it_already_captured_instr->second, false);
      }
    }
  }
  std::cout << "Enhanced tracer has parsed " << all_kernels_key_instructions_by_pc.size() << "/" << already_instrumented.size() << " kernels" << std::endl;
  if(!intermediate_extra_files_persistance) {
    remove_folder(cubin_path.c_str());
    remove_folder(sass_path.c_str());
    remove_folder(register_usage_path.c_str());
  }
  m_enhanced_traced_execution->remove_useless_kernels();
  m_enhanced_traced_execution->SerializeToFile(extrainfo_path +"/enhanced_execution_info.json");
}

void nvbit_at_ctx_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  pending_devices_to_finish++;
  if (verbose) {
      printf("Tracer: Starting context %p\n", ctx);
  }
  assert(ctx_state_map.find(ctx) == ctx_state_map.end());
  CTXstate* ctx_state = new CTXstate;
  ctx_state_map[ctx] = ctx_state;
  pthread_mutex_unlock(&mutex);
}

void init_context_state(CUcontext ctx) {
  if(is_first_init_context_call) {
    is_first_init_context_call = false;
    CUDA_SAFECALL(cuDeviceGetCount(&num_devices));
    CUDA_SAFECALL(cuMemAllocManaged(reinterpret_cast<CUdeviceptr *>(&stop_report),
                                   num_devices * sizeof(bool), CU_MEM_ATTACH_GLOBAL));
    for(int i = 0; i < num_devices; ++i) {
      stop_report[i] = false;
      kernel_id.push_back(1);
      current_stream_id.push_back(0);
    }
    if (warp_lifetime_only || warp_lifetime_sidecar) {
      CUDA_SAFECALL(cuMemAllocManaged(
          reinterpret_cast<CUdeviceptr *>(&warp_lifetime_capture),
          sizeof(warp_lifetime_capture_t), CU_MEM_ATTACH_GLOBAL));
      CUDA_SAFECALL(cuMemsetD8(
          reinterpret_cast<CUdeviceptr>(warp_lifetime_capture), 0,
          sizeof(warp_lifetime_capture_t)));
    }
    /* Phase 0b: one managed param-base capture slot (single device assumed here,
     * matching the rest of the tool's single-stream simplification). */
    if (enable_tma_desc) {
      CUDA_SAFECALL(cuMemAllocManaged(
          reinterpret_cast<CUdeviceptr *>(&tma_param_base_capture),
          sizeof(tma_param_base_capture_t), CU_MEM_ATTACH_GLOBAL));
      tma_param_base_capture->param_base = 0;
      tma_param_base_capture->unique_function_id = 0;
      tma_param_base_capture->valid = 0;
    }
    /* SPIKE 6 fact-check (§2.18): one managed buffer for device-side 128B descriptor reads. */
    if (enable_tma_desc && tma_desc_factcheck_enable) {
      CUDA_SAFECALL(cuMemAllocManaged(
          reinterpret_cast<CUdeviceptr *>(&tma_desc_factcheck),
          sizeof(tma_desc_factcheck_t), CU_MEM_ATTACH_GLOBAL));
      memset(tma_desc_factcheck, 0, sizeof(tma_desc_factcheck_t));
    }
  }
  CTXstate* ctx_state = ctx_state_map[ctx];
  ctx_state->recv_thread_done = RecvThreadState::WORKING;
  CUDA_SAFECALL(cuMemAllocManaged(reinterpret_cast<CUdeviceptr *>(&ctx_state->channel_dev),
                                  sizeof(ChannelDev), CU_MEM_ATTACH_GLOBAL));
  ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                               ctx_state->channel_dev, recv_thread_fun, ctx);
  nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
  if (mbarrier_contract_capture_enabled) {
    CUDA_SAFECALL(cuMemAllocManaged(
        reinterpret_cast<CUdeviceptr *>(
            &ctx_state->mbarrier_contract_registry),
        sizeof(mbarrier_contract_registry_t), CU_MEM_ATTACH_GLOBAL));
    CUDA_SAFECALL(cuMemsetD8(
        reinterpret_cast<CUdeviceptr>(
            ctx_state->mbarrier_contract_registry),
        0, sizeof(mbarrier_contract_registry_t)));
    ctx_state->mbarrier_contract_registry->magic =
        MBARRIER_CONTRACT_REGISTRY_MAGIC;
    ctx_state->mbarrier_contract_registry->version =
        MBARRIER_CONTRACT_REGISTRY_VERSION;
  }
}

void nvbit_tool_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  init_context_state(ctx);
  pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  skip_callback_flag = true;
  if (verbose) {
      printf("Tracer: Terminating context %p\n", ctx);
  }
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];
  pending_devices_to_finish--;
  if(pending_devices_to_finish == 0 && !warp_lifetime_only) {
    dyn_trace.set_nvbit_version(NVBIT_VERSION);
    dyn_trace.set_accelsim_version(TRACER_VERSION);
    dyn_trace.set_is_gathered_registers_values(
        compact_wire_enabled && !prune_operands_enabled);
    std::string program_name = get_program_path();
    std::size_t found = program_name.find_last_of("/");
    if(found != std::string::npos) {
      program_name = program_name.substr(found + 1);
    }
    dyn_trace.set_name(program_name);
    std::string filename = traces_path + "/dynamic_trace.pb";
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file " << filename << " for writing." << std::endl;
  } else {
      if (!dyn_trace.SerializeToOstream(&ofs)) {
          std::cerr << "Failed to serialize dyn_trace." << std::endl;
      } else {
          std::cout << "Serialized protocol buffer written to " << filename << std::endl;
          dyn_trace.Clear();
      }
      ofs.close();
  }
    std::cout << "Starting the enhanced tracer" << std::endl;
    enhanced_tracer();
    std::cout << "Terminated the enhanced tracer" << std::endl;
  }
  ctx_state->recv_thread_done = RecvThreadState::STOP;
  while (ctx_state->recv_thread_done != RecvThreadState::FINISHED);
  ctx_state->channel_host.destroy(false);
  CUDA_SAFECALL(cuMemFree(reinterpret_cast<CUdeviceptr>(ctx_state->channel_dev)));
  if (ctx_state->mbarrier_contract_registry != nullptr) {
    if (ctx_state->mbarrier_contract_registry->active != 0 ||
        ctx_state->mbarrier_contract_registry->launch.state != 0 ||
        ctx_state->mbarrier_contract_registry->error_flags != 0) {
      fprintf(stderr,
              "[MBARRIER-CONTRACT] context terminated with live or failed "
              "capture state: active=%d state=%u errors=0x%x\n",
              ctx_state->mbarrier_contract_registry->active,
              ctx_state->mbarrier_contract_registry->launch.state,
              ctx_state->mbarrier_contract_registry->error_flags);
      abort();
    }
    CUDA_SAFECALL(cuMemFree(
        reinterpret_cast<CUdeviceptr>(
            ctx_state->mbarrier_contract_registry)));
    ctx_state->mbarrier_contract_registry = nullptr;
  }
  if (pending_devices_to_finish == 0 && warp_lifetime_capture != nullptr) {
    CUDA_SAFECALL(cuMemFree(
        reinterpret_cast<CUdeviceptr>(warp_lifetime_capture)));
    warp_lifetime_capture = nullptr;
  }
  skip_callback_flag = false;
  delete ctx_state;
  pthread_mutex_unlock(&mutex);
}
