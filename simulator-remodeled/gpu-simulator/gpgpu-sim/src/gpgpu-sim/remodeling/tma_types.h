#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

enum class TMAOpcodeFamily {
  UTMALDG,
  UTMAPF,
  UTMASTG,
  UTMAREDG,
  UBLKCP,
  UBLKPF,
  UBLKRED,
  UTMACCTL,
  UTMACMDFLUSH,
  UNKNOWN,
};

enum class TMADirection {
  GMEM_TO_SMEM,
  SMEM_TO_GMEM,
  NONE,
};

enum class TMATransferType {
  LOAD,
  STORE,
  PREFETCH,
  REDUCTION,
  CONTROL,
  UNKNOWN,
};

enum class TMAMetadataSource {
  DESCRIPTOR,
  OPERAND,
  MIXED,
  NONE,
};

enum class TMAOperandForm {
  EXPLICIT_DESC,
  DESC_LIKE_PAIR,
  BULK_OPERAND,
  GENERIC,
};

struct TMADescriptorConfigMetadata {
  std::string config_id;
  uint32_t tensor_rank = 0;
  uint32_t tensor_data_type = 0;
  std::array<uint32_t, 5> global_dim = {0, 0, 0, 0, 0};
  std::array<uint64_t, 5> global_strides = {0, 0, 0, 0, 0};
  std::array<uint32_t, 5> box_dim = {0, 0, 0, 0, 0};
  std::array<uint32_t, 5> element_strides = {0, 0, 0, 0, 0};
  uint32_t interleave = 0;
  uint32_t swizzle = 0;
  uint32_t l2_promotion = 0;
  uint32_t oob_fill = 0;
  uint32_t element_size = 0;
};

struct TMADescriptorLookupKey {
  unsigned int unique_function_id = 0;
  uint64_t pc = 0;
  uint32_t handle_hi = 0;

  bool operator<(const TMADescriptorLookupKey &other) const {
    return std::tie(unique_function_id, pc, handle_hi) <
           std::tie(other.unique_function_id, other.pc, other.handle_hi);
  }
};

struct TMAOperandLookupKey {
  unsigned int unique_function_id = 0;
  uint64_t pc = 0;

  bool operator<(const TMAOperandLookupKey &other) const {
    return std::tie(unique_function_id, pc) <
           std::tie(other.unique_function_id, other.pc);
  }
};

struct TMADescriptorSiteRecord {
  std::string config_id;
  std::string mapping_method;
  float resolver_confidence = 0.0f;
  bool config_is_ambiguous = false;
  bool has_descriptor_metadata = false;
};

struct TMAOperandSiteRecord {
  std::string mapping_method;
  float resolver_confidence = 0.0f;
  bool runtime_observed = false;
  bool has_operand_metadata = false;
  bool has_linked_descriptor_metadata = false;
  std::string linked_config_id;
  TMAOperandForm operand_form = TMAOperandForm::GENERIC;
  uint32_t covered_bytes = 0;
  bool has_covered_bytes = false;
  uint32_t operand3_raw = 0;
  bool has_operand3_raw = false;
};

struct TMAResolvedSiteMetadata {
  bool valid = false;
  bool has_descriptor_metadata = false;
  bool has_operand_metadata = false;
  bool descriptor_lookup_hit = false;
  bool operand_lookup_hit = false;
  bool runtime_observed = false;
  uint32_t handle_hi = 0;
  std::string config_id;
  std::string mapping_method;
  float resolver_confidence = 0.0f;
  TMAOperandForm operand_form = TMAOperandForm::GENERIC;
  uint32_t covered_bytes = 0;
  bool has_covered_bytes = false;
  uint32_t operand3_raw = 0;
  bool has_operand3_raw = false;
  TMADescriptorConfigMetadata descriptor_config;
};

struct TMASidecarMetadataDB {
  std::map<std::string, TMADescriptorConfigMetadata> descriptor_configs;
  std::map<TMADescriptorLookupKey, TMADescriptorSiteRecord> descriptor_site_records;
  std::map<TMAOperandLookupKey, TMAOperandSiteRecord> operand_site_records;

  void clear() {
    descriptor_configs.clear();
    descriptor_site_records.clear();
    operand_site_records.clear();
  }

  bool empty() const {
    return descriptor_configs.empty() && descriptor_site_records.empty() &&
           operand_site_records.empty();
  }
};

enum class SyncInstructionKind {
  NONE,
  EXCH,
  ARRIVE,
  ARRIVE_EXPECT_TX,
  ARRIVE_COUNTED,
  PHASECHK,
  TRYWAIT,
};

enum class SyncSemanticOperandRole {
  NONE,
  EXCH_ARRIVE_COUNT_ENCODED,
  EXPECT_TX_BYTES,
  WAIT_STATE,
  ARRIVE_COUNT,
};

struct SyncOperandLookupKey {
  unsigned int unique_function_id = 0;
  uint64_t pc = 0;

  bool operator<(const SyncOperandLookupKey &other) const {
    return std::tie(unique_function_id, pc) <
           std::tie(other.unique_function_id, other.pc);
  }
};

struct SyncSiteRecord {
  std::string opcode;
  SyncInstructionKind sync_kind = SyncInstructionKind::NONE;
  int barrier_operand_index = -1;
  int semantic_operand_index = -1;
  SyncSemanticOperandRole semantic_operand_role =
      SyncSemanticOperandRole::NONE;
  std::string barrier_operand_text;
  std::string semantic_operand_text;
};

struct SyncResolvedSiteMetadata {
  bool valid = false;
  std::string opcode;
  SyncInstructionKind sync_kind = SyncInstructionKind::NONE;
  int barrier_operand_index = -1;
  int semantic_operand_index = -1;
  SyncSemanticOperandRole semantic_operand_role =
      SyncSemanticOperandRole::NONE;
  std::string barrier_operand_text;
  std::string semantic_operand_text;
};

struct SyncSidecarMetadataDB {
  std::map<SyncOperandLookupKey, SyncSiteRecord> site_records;

  void clear() { site_records.clear(); }

  bool empty() const { return site_records.empty(); }
};

struct HopperMBarrierKey {
  uint32_t trace_kernel_id = 0;
  uint32_t cta_id = 0;
  uint64_t barrier_addr = 0;

  bool operator<(const HopperMBarrierKey &other) const {
    return std::tie(trace_kernel_id, cta_id, barrier_addr) <
           std::tie(other.trace_kernel_id, other.cta_id, other.barrier_addr);
  }
};

struct HopperMBarrierObject {
  uint32_t expected_arrive_count = 0;
  uint32_t arrive_count = 0;
  uint32_t expected_tx_bytes = 0;
  uint32_t completed_tx_bytes = 0;
  uint32_t phase = 0;
  bool ready = false;
};

struct HopperMBarrierPendingWait {
  bool valid = false;
  bool is_trywait = false;
  HopperMBarrierKey key;
  uint64_t wait_state_raw = 0;
};

struct HopperMBarrierPendingTxBinding {
  HopperMBarrierKey key;
  uint32_t pending_tx_bytes = 0;
};

struct TMACommand {
  uint32_t warp_id = 0;
  uint32_t cta_id = 0;
  uint32_t sm_id = 0;
  uint32_t subcore_id = 0;
  TMAOpcodeFamily opcode_family = TMAOpcodeFamily::UNKNOWN;
  TMADirection direction = TMADirection::NONE;
  TMATransferType transfer_type = TMATransferType::UNKNOWN;
  std::string config_id;
  TMAMetadataSource meta_source = TMAMetadataSource::NONE;
  std::string mapping_method;
  float resolver_confidence = 0.0f;
  uint32_t rank = 0;
  std::array<uint32_t, 5> box_dim = {0, 0, 0, 0, 0};
  std::array<uint32_t, 5> coords = {0, 0, 0, 0, 0};
  uint32_t element_size = 0;
  uint64_t smem_ptr = 0;
  uint32_t requests_total = 0;
  uint32_t total_bytes = 0;
  uint32_t covered_bytes = 0;
  uint32_t operand3_raw = 0;
  uint32_t swizzle = 0;
  uint32_t interleave = 0;
  uint32_t oob_fill = 0;
  uint32_t l2_promotion = 0;
  TMAOperandForm operand_form = TMAOperandForm::GENERIC;
};

struct TMATransferEntry {
  enum class State {
    ISSUED,
    ENQUEUED,
    AGU_READY,
    IN_FLIGHT,
    COMPLETED,
    WAIT_SATISFIED,
  };

  TMACommand cmd;
  State state = State::ISSUED;
  uint32_t requests_issued = 0;
  uint32_t requests_completed = 0;
  uint32_t bytes_completed = 0;
  uint64_t transfer_uid = 0;
  int cycle_enqueued = -1;
  int cycle_agu_ready = -1;
  int cycle_first_request = -1;
  int cycle_last_completion = -1;
};
