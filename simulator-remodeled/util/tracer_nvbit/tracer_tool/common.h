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

#include <stddef.h>
#include <stdint.h>

static __managed__ uint64_t total_dynamic_instr_counter = 0;
static __managed__ uint64_t reported_dynamic_instr_counter = 0;
/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
#define MAX_SRC 5

enum MEM_TYPE { NONE = 0, STANDARD_MEM = 1, CONSTANT_MEM = 2, CALL_OR_RET = 3 };
enum TRACED_REG_TYPE {NO_REGS = 0, MEMORY_REF = 1, REGULAR = 2, REGULAR_2_REGS = 3, REGULAR_4_REGS = 4, UNIFORM = 5, UNIFORM_2_REGS = 6, PREDICATE = 7, UNIFORM_PREDICATE = 8};

/* Original fixed-size GPU-to-host record, retained for
 * LIGHTER_TRACER=off and legacy compatibility. */
typedef struct {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warpid_tb;
  int warpid_sm;
  int sm_id;
  uint64_t addrs_or_reg_val_0[32];
  uint32_t reg_val_1[32];
  uint32_t reg_val_2[32];
  uint32_t reg_val_3[32];
  uint32_t reg_id;
  uint32_t vpc;
  uint32_t unique_function_id;
  MEM_TYPE mem_type;
  int32_t width;
  uint32_t ureg_desc_id;
  uint32_t ureg_desc_value;
  uint32_t ureg_desc_value_hi;
  uint32_t active_mask;
  uint32_t predicate_mask;
  uint32_t num_of_injects;
  uint32_t per_operand_type;
} legacy_inst_trace_t;

enum INST_TRACE_RECORD_FLAGS {
  INST_TRACE_HAS_LANE_VALUES = 1u << 0,
};

#define INST_TRACE_RECORD_VERSION 1u

/*
 * GPU-to-host transport record.
 *
 * The old fixed record carried four 32-lane value arrays for every callback
 * (712 bytes), although the final protobuf only consumes lane values for
 * memory-like callbacks. Non-memory callbacks need the common metadata and at
 * most one predicated-active lane's operand values. Keep those scalar values
 * in the compact prefix and append the 32-lane payload only when
 * INST_TRACE_HAS_LANE_VALUES is set.
 *
 * The host receiver expands either wire form into the same protobuf contract;
 * this changes capture transport only, not the dynamic trace schema.
 */
typedef struct {
  uint32_t record_size;
  uint16_t record_version;
  uint16_t record_flags;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warpid_tb;
  int warpid_sm;
  int sm_id;
  uint32_t reg_id;
  uint32_t vpc;
  uint32_t unique_function_id;
  MEM_TYPE mem_type;
  int32_t width;
  uint32_t ureg_desc_id;
  uint32_t ureg_desc_value;
  uint32_t ureg_desc_value_hi;
  uint32_t active_mask;
  uint32_t predicate_mask;
  uint32_t num_of_injects;
  uint32_t per_operand_type;
  uint64_t scalar_value_0;
  uint32_t scalar_value_1;
  uint32_t scalar_value_2;
  uint32_t scalar_value_3;
  uint32_t reserved;
  uint64_t lane_values[32];
} inst_trace_t;

#define INST_TRACE_COMPACT_SIZE \
  ((uint32_t)offsetof(inst_trace_t, lane_values))
#define INST_TRACE_MEMORY_SIZE ((uint32_t)sizeof(inst_trace_t))
#define INST_TRACE_LEGACY_SIZE ((uint32_t)sizeof(legacy_inst_trace_t))

#ifdef __cplusplus
static_assert(INST_TRACE_LEGACY_SIZE == 712,
              "unexpected legacy inst trace wire size");
static_assert(INST_TRACE_COMPACT_SIZE == 104,
              "unexpected compact inst trace wire size");
static_assert(INST_TRACE_MEMORY_SIZE == 360,
              "unexpected memory inst trace wire size");
#endif

/*
 * Optional barrier-local contract capture.
 *
 * This transport is launch-owned managed memory, not a ChannelDev stream.
 * Device callbacks therefore cannot block waiting for a second host receiver.
 * All capacities are checked at launch and every overflow is fatal.
 */
#define MBARRIER_CONTRACT_RECORD_VERSION 1u
#define MBARRIER_CONTRACT_REGISTRY_VERSION 1u
#define MBARRIER_CONTRACT_REGISTRY_MAGIC 0x4d42434fu
#define MBARRIER_CONTRACT_MAX_WARPS_PER_CTA 32u

enum MBARRIER_CONTRACT_EVENT_KIND {
  MBARRIER_CONTRACT_EVENT_NONE = 0,
  MBARRIER_CONTRACT_EVENT_EXCH = 1,
  MBARRIER_CONTRACT_EVENT_ARRIVE = 2,
  MBARRIER_CONTRACT_EVENT_WAIT = 3,
};

enum MBARRIER_CONTRACT_CAPTURE_FLAGS {
  MBARRIER_CONTRACT_CAPTURE_NONE = 0,
  MBARRIER_CONTRACT_CAPTURE_PLAIN_ARRIVE = 1u << 0,
  MBARRIER_CONTRACT_CAPTURE_ARRIVE_BEFORE_INIT = 1u << 1,
  MBARRIER_CONTRACT_CAPTURE_DUPLICATE_INIT = 1u << 2,
};

enum MBARRIER_CONTRACT_ERROR_FLAGS {
  MBARRIER_CONTRACT_ERROR_NONE = 0,
  MBARRIER_CONTRACT_ERROR_NO_ACTIVE_LAUNCH = 1u << 0,
  MBARRIER_CONTRACT_ERROR_INVALID_LAUNCH = 1u << 1,
  MBARRIER_CONTRACT_ERROR_CTA_OUT_OF_RANGE = 1u << 2,
  MBARRIER_CONTRACT_ERROR_WARP_OUT_OF_RANGE = 1u << 3,
  MBARRIER_CONTRACT_ERROR_BARRIER_TABLE_OVERFLOW = 1u << 4,
  MBARRIER_CONTRACT_ERROR_NESTED_BEGIN = 1u << 5,
  MBARRIER_CONTRACT_ERROR_MISSING_BEGIN = 1u << 6,
  MBARRIER_CONTRACT_ERROR_IDENTITY_MISMATCH = 1u << 7,
  MBARRIER_CONTRACT_ERROR_RECORD_OVERFLOW = 1u << 8,
  MBARRIER_CONTRACT_ERROR_EPOCH_OVERFLOW = 1u << 9,
  MBARRIER_CONTRACT_ERROR_ORDINAL_OVERFLOW = 1u << 10,
  MBARRIER_CONTRACT_ERROR_INVALID_BARRIER_ADDRESS = 1u << 11,
};

typedef struct {
  uint64_t key_plus_one;
  uint64_t opened_contract_count;
  uint32_t initialization_epoch;
  uint32_t initialization_count;
  uint32_t plain_arrive_count;
  uint32_t semantic_error_flags;
} mbarrier_contract_barrier_state_t;

typedef struct {
  uint32_t state;
  uint32_t event_kind;
  uint32_t unique_function_id;
  uint32_t vpc;
  uint32_t active_mask;
  uint32_t predicate_mask;
  uint32_t sm_id;
  uint32_t capture_flags;
  uint64_t barrier_addr;
  uint64_t semantic_raw;
  uint64_t opened_contract_count_before;
  uint64_t contract_ordinal;
  uint32_t initialization_epoch_before;
  uint32_t barrier_error_flags_before;
} mbarrier_contract_warp_scratch_t;

typedef struct {
  uint32_t record_size;
  uint16_t record_version;
  uint16_t event_kind;
  uint64_t launch_uid;
  uint32_t linear_cta_id;
  uint32_t warp_id_tb;
  uint32_t sm_id;
  uint32_t unique_function_id;
  uint32_t vpc;
  uint32_t active_mask;
  uint32_t predicate_mask;
  uint32_t initialization_epoch_before;
  uint32_t initialization_epoch_after;
  uint32_t capture_flags;
  uint32_t barrier_error_flags;
  uint64_t barrier_addr;
  uint64_t semantic_raw;
  uint64_t opened_contract_count_before;
  uint64_t opened_contract_count_after;
  uint64_t contract_ordinal;
} mbarrier_contract_record_t;

typedef struct {
  uint32_t state;
  uint32_t device_id;
  uint32_t stream_id;
  uint32_t trace_kernel_id;
  uint64_t launch_uid;
  uint32_t grid_dim_x;
  uint32_t grid_dim_y;
  uint32_t grid_dim_z;
  uint32_t num_ctas;
  uint32_t warps_per_cta;
  uint32_t barrier_slots_per_cta;
  uint64_t barrier_state_ptr;
  uint64_t warp_scratch_ptr;
  uint64_t cta_record_count_ptr;
  uint64_t record_buffer_ptr;
  uint64_t record_capacity;
  uint64_t records_per_cta;
  uint64_t overflow_records;
} mbarrier_contract_launch_t;

typedef struct {
  uint32_t magic;
  uint32_t version;
  int32_t active;
  uint32_t error_flags;
  mbarrier_contract_launch_t launch;
} mbarrier_contract_registry_t;

#ifdef __cplusplus
static_assert(sizeof(mbarrier_contract_barrier_state_t) == 32,
              "unexpected mbarrier contract state size");
static_assert(sizeof(mbarrier_contract_warp_scratch_t) == 72,
              "unexpected mbarrier contract scratch size");
static_assert(sizeof(mbarrier_contract_record_t) == 104,
              "unexpected mbarrier contract record size");
static_assert(sizeof(mbarrier_contract_launch_t) == 104,
              "unexpected mbarrier contract launch size");
static_assert(sizeof(mbarrier_contract_registry_t) == 120,
              "unexpected mbarrier contract registry size");
#endif

/* EXIT timestamp capture for FA3 producer/consumer lifetime, shared by the
 * EXIT-only probe and the full-trace sidecar. The current workload uses at most
 * 384 CTAs and 16 warps/CTA; the larger fixed bounds keep the tool reusable
 * without adding device-side allocation logic. */
#define WARP_LIFETIME_MAX_CTAS 4096
#define WARP_LIFETIME_MAX_WARPS 32
#define WARP_LIFETIME_MAX_SLOTS \
  (WARP_LIFETIME_MAX_CTAS * WARP_LIFETIME_MAX_WARPS)

typedef struct {
  unsigned long long exit_clock[WARP_LIFETIME_MAX_SLOTS];
  unsigned int sm_id[WARP_LIFETIME_MAX_SLOTS];
  unsigned int overflow_count;
} warp_lifetime_capture_t;

/* Phase 0b (TMA_BASE_ADDR.md §2.13, host-deref Gate 0):
 * one managed slot per device that the entry ULDC.64 URx, c[0x0][K] writes its
 * loaded value (param_base, the device-global pointer to the kernel param block)
 * into, once per launch. The host reads it after cuLaunchKernel sync and
 * cuMemcpyDtoH's the param region so the offline harness can slice each 128B
 * descriptor at param_base + tensormap_offset. */
typedef struct {
  unsigned long long param_base;
  unsigned int unique_function_id;
  int valid;
} tma_param_base_capture_t;

/* SPIKE 6 device fact-check (TMA_BASE_ADDR.md §2.18):
 * at an executed UTMALDG/UTMASTG, read 128B from the descriptor VA (the operand that
 * holds the generic-SMEM tensormap address, e.g. 0xffffffffc428xxxx) and record the
 * first two qwords. Offline we check qword0 in {7 encode bases} to confirm the
 * SMEM-staged descriptor really carries the real base. A tiny fixed ring is enough —
 * we only need a handful of distinct samples, not every dynamic issue. */
#define TMA_DESC_FACTCHECK_SLOTS 256
/* address-space classification of desc_va (isspacep.* is a pure predicate that never
 * faults, so we can record it for every sample even when the byte read is skipped). */
enum TMA_DESC_VA_SPACE {
  TMA_VA_UNKNOWN = 0,  /* generic addr in none of the mapped windows below */
  TMA_VA_GLOBAL = 1,
  TMA_VA_SHARED = 2,
  TMA_VA_CONSTANT = 3,
  TMA_VA_LOCAL = 4
};
typedef struct {
  unsigned int count;                              /* total attempts (atomic) */
  unsigned int stored;                             /* slots actually written  */
  unsigned int unique_function_id[TMA_DESC_FACTCHECK_SLOTS];
  unsigned int pc[TMA_DESC_FACTCHECK_SLOTS];
  unsigned int mref_ord[TMA_DESC_FACTCHECK_SLOTS];       /* which memref (0/1) */
  unsigned int space[TMA_DESC_FACTCHECK_SLOTS];          /* TMA_DESC_VA_SPACE  */
  unsigned int read_ok[TMA_DESC_FACTCHECK_SLOTS];        /* 1 if q0/q1 valid   */
  unsigned long long desc_va[TMA_DESC_FACTCHECK_SLOTS];  /* address inspected  */
  unsigned long long qword0[TMA_DESC_FACTCHECK_SLOTS];   /* expected = base    */
  unsigned long long qword1[TMA_DESC_FACTCHECK_SLOTS];
} tma_desc_factcheck_t;
