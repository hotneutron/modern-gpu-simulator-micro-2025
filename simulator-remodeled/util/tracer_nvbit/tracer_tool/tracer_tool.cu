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
#include <assert.h>
#include <bitset>
#include <cctype>
#include <cstdio>
#include <deque>
#include <inttypes.h>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
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
  int id;

  /* Channel used to communicate from GPU to CPU receiving thread */
  ChannelDev* channel_dev;
  ChannelHost channel_host;

  // After initialization, set it to WORKING to make recv thread get data,
  // parent thread sets it to STOP to make recv thread stop working.
  // recv thread sets it to FINISHED when it cleans up.
  // parent thread should wait until the state becomes FINISHED to clean up.
  volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;
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
int gather_registers = 0;
int enable_compress = 1;
int print_core_id = 0;
int exclude_pred_off = 1;
int active_from_start = 1;
/* used to select region of interest when active from start is 0 */
bool active_region = true;

/* Should we terminate the program once we are done tracing? */
int terminate_after_limit_number_of_kernels_reached = 0;
int user_defined_folders = 0;

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
std::unordered_set<int> tma_desc_runtime_debug_header_written;
std::unordered_set<int> tma_desc_producer_debug_header_written;
std::unordered_set<int> tma_memcpy_dump_header_written;
std::unordered_set<int> tensor_map_encode_dump_header_written;
std::unordered_set<int> tma_consumer_opcode_ids;
std::map<int, std::map<int, uint32_t>> tma_desc_ureg_by_pc;
std::map<int, std::map<int, uint32_t>> first_dest_ureg_by_pc;
std::map<int, std::map<int, std::string>> instruction_text_by_pc;
struct sync_runtime_capture_site_info {
  sync_runtime_capture_site_info()
      : enabled(false),
        barrier_callback_index(-1),
        semantic_callback_index(-1),
        semantic_is_zero_literal(false) {}
  bool enabled;
  int barrier_callback_index;
  int semantic_callback_index;
  bool semantic_is_zero_literal;
};
std::map<int, std::map<int, sync_runtime_capture_site_info>>
    sync_runtime_capture_sites_by_pc;
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
    InstrType::OperandType operand_type, const std::string &operand_str) {
  sync_runtime_capture_site_info info;
  sync_runtime_capture_kind kind = get_sync_runtime_capture_kind(opcode);
  if (!sync_runtime_capture_enabled(kind)) {
    return info;
  }

  if (operand_position == 1 && operand_type == InstrType::OperandType::MREF) {
    info.enabled = true;
    info.barrier_callback_index = callback_index;
  } else if (operand_position == 2 &&
             is_sync_semantic_operand_type(operand_type)) {
    info.enabled = true;
    if (is_zero_sync_semantic_operand(operand_str)) {
      info.semantic_is_zero_literal = true;
    } else {
      info.semantic_callback_index = callback_index;
    }
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
      first_lane_addr = ma->addrs_or_reg_val_0[lane];
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
  uint64_t first_lane_addr = ma->addrs_or_reg_val_0[first_lane];
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
      << "\"operand_reg_id\":" << ma->reg_id << ","
      << "\"value_lo\":" << static_cast<uint64_t>(ma->addrs_or_reg_val_0[first_lane]) << ","
      << "\"value_hi\":" << ma->reg_val_1[first_lane] << ","
      << "\"value_2\":" << ma->reg_val_2[first_lane] << ","
      << "\"value_3\":" << ma->reg_val_3[first_lane] << ","
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
  GET_VAR_INT(gather_registers, "GATHER_REGISTERS", 0, "Enable gathering of GPU register values. Not available in this version.");
  GET_VAR_INT(incremental_flush_threshold, "INCREMENTAL_FLUSH_THRESHOLD", 0,
              "Flush threadblocks to disk when instruction count exceeds this threshold. "
              "0 = disabled (default). Recommended: 50000-100000 for large kernels.");
  GET_VAR_INT(enable_tma_desc, "ENABLE_TMA_DESC", 0,
              "Enable TMA descriptor capture, including runtime desc handle debug and tensor-map descriptor dumps.");
  GET_VAR_INT(aux_htod_dump_max_bytes, "AUX_HTOD_DUMP_MAX_BYTES", 4096,
              "Maximum auxiliary HtoD memcpy payload size to dump when ENABLE_TMA_DESC is enabled.");
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
    if (verbose) {
      printf("Inspecting function %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), addr_funct);
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {

      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
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
      pc_to_opcode[next_candidate_unique_function_id][vpc] = opcode_id;

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
      
      /* We only report memory addresses */
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
      std::string opcode_str = instr->getOpcode() ? std::string(instr->getOpcode()) : "";
      sync_runtime_capture_site_info sync_site_info;

      if(inst_parsed->is_tensor_core_op()) {
        inst_parsed->set_tensor_core_instruction_info();
      }
      for(int i = 0; i < instr->getNumOperands(); ++i){
        const InstrType::operand_t *op = instr->getOperand(i);
        std::string operand_str = op->str ? std::string(op->str) : "";
        sync_runtime_capture_site_info sync_operand_info =
            build_sync_runtime_capture_site_info(opcode_str, i, num_of_injects,
                                                 op->type, operand_str);
        if (sync_operand_info.enabled) {
          sync_site_info.enabled = true;
          if (sync_operand_info.barrier_callback_index >= 0) {
            sync_site_info.barrier_callback_index =
                sync_operand_info.barrier_callback_index;
          }
          if (sync_operand_info.semantic_callback_index >= 0) {
            sync_site_info.semantic_callback_index =
                sync_operand_info.semantic_callback_index;
          }
          if (sync_operand_info.semantic_is_zero_literal) {
            sync_site_info.semantic_is_zero_literal = true;
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
      if (is_tma_desc_consumer_instruction && tma_desc_ureg_id != SECRET_UREG_DESC_NOT_USED) {
        tma_desc_ureg_by_pc[next_candidate_unique_function_id][vpc] = tma_desc_ureg_id;
      }
      if (sync_site_info.enabled) {
        sync_runtime_capture_sites_by_pc[next_candidate_unique_function_id][vpc] =
            sync_site_info;
      }

      if(num_of_injects == 0) {
        per_operand_type[0] = traced_operand_instrument(TRACED_REG_TYPE::NO_REGS, 0, 0);
        num_of_injects++;
      }

      for(unsigned int i = 0; i < num_of_injects; i++) {
          /* insert call to the instrumentation function with its arguments */
          nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);
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
      }
      cnt++;
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

__global__ void flush_channel(ChannelDev* ch_dev) {
  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  inst_trace_t ma;
  ma.cta_id_x = -1;
  ch_dev->push(&ma, sizeof(inst_trace_t));

  /* flush channel */
  ch_dev->flush();
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

        // std::vector<int> kernel_argument_sizes = nvbit_get_kernel_argument_sizes(p->f);
        
        std::string final_kernel_name = kernel_name + variant_delimiter_str + std::to_string(variant_id);
        
        auto it_map_unique_function_id = map_function_name_to_unique_function_id_with_variant.find(final_kernel_name);
        assert(it_map_unique_function_id != map_function_name_to_unique_function_id_with_variant.end());
        dyn_trace.set_binary_version(binary_version);
        auto stream_map = gpu_dev.streams();
        uint64_t stream_key = (uint64_t)p->hStream;
        stream_key = 0; // We use 0 as stream key for now, as we do not support multiple streams yet. WIPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
        current_stream_id[device_id] = stream_key;
        dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
        stream.set_id(stream_key);
        dynamic_trace::kernel *ker = stream.add_kernels();
        ker->set_id(kernel_id[device_id]);
        ker->set_name(final_kernel_name);
        ker->set_function_unique_id(it_map_unique_function_id->second);
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

      /* make sure we prevent re-entry on the nvbit_callback when issuing
       * the flush_channel kernel */
      skip_callback_flag = true;

      /* issue flush of channel so we are sure all the memory accesses
       * have been pushed */
      flush_channel<<<1, 1>>>(ctx_state->channel_dev);
      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);

      /* unset the skip flag */
      skip_callback_flag = false;

      /* wait here until the receiving thread has not finished with the
       * current kernel */
      while (recv_thread_receiving[ctx]) {
        pthread_yield();
      }

      unsigned total_insts_per_kernel =
          total_dynamic_instr_counter - old_total_insts;
      old_total_insts = total_dynamic_instr_counter;

      unsigned reported_insts_per_kernel =
          reported_dynamic_instr_counter - old_total_reported_insts;
      old_total_reported_insts = reported_dynamic_instr_counter;

      statsFile = fopen(stats_location.c_str(), "a");
      fprintf(statsFile, "%d,%d", total_insts_per_kernel,
              reported_insts_per_kernel);
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

  while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
    if (num_recv_bytes > 0) {
      uint32_t num_processed_bytes = 0;
      dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[current_stream_id[device_id]];
      dynamic_trace::kernel &ker = (*stream.mutable_kernels())[kernel_id[device_id]-2]; 
      while (num_processed_bytes < num_recv_bytes) {
        inst_trace_t *ma = (inst_trace_t *)&recv_buffer[num_processed_bytes];

        /* when we receive a CTA_id_x it means all the kernels
        * completed, this is the special token we receive from the
        * flush channel kernel that is issues at the end of the
        * context */
        if (ma->cta_id_x == -1) {
          recv_thread_receiving[ctx] = false;
          break;
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
                sync->set_barrier_addr(ma->addrs_or_reg_val_0[first_lane]);
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
                sync->set_semantic_raw(ma->addrs_or_reg_val_0[first_lane]);
              }
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
              candidate.pre_value_lo = static_cast<uint32_t>(ma->addrs_or_reg_val_0[0]);
              candidate.pre_value_hi =
                  (ma->per_operand_type == TRACED_REG_TYPE::UNIFORM_2_REGS) ? ma->reg_val_1[0]
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
                base_stride_compress(ma->addrs_or_reg_val_0, mask, base_addr, stride);
            if (!base_stride_success) {
              // if base+stride fails, try base+delta format
              base_delta_compress(ma->addrs_or_reg_val_0, mask, base_addr, deltas);
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
                addr->add_addrs(ma->addrs_or_reg_val_0[s]);
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

        num_processed_bytes += sizeof(inst_trace_t);
      }
    }
  }
  free(recv_buffer);

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
  }
  CTXstate* ctx_state = ctx_state_map[ctx];
  ctx_state->recv_thread_done = RecvThreadState::WORKING;
  CUDA_SAFECALL(cuMemAllocManaged(reinterpret_cast<CUdeviceptr *>(&ctx_state->channel_dev),
                                  sizeof(ChannelDev), CU_MEM_ATTACH_GLOBAL));
  ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                               ctx_state->channel_dev, recv_thread_fun, ctx);
  nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
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
  if(pending_devices_to_finish == 0) {
    dyn_trace.set_nvbit_version(NVBIT_VERSION);
    dyn_trace.set_accelsim_version(TRACER_VERSION);
    dyn_trace.set_is_gathered_registers_values(static_cast<bool>(gather_registers));
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
  skip_callback_flag = false;
  delete ctx_state;
  pthread_mutex_unlock(&mutex);
}
