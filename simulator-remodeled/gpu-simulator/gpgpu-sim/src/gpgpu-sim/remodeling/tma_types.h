#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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

struct TMAOperandLookupKey {
  unsigned int trace_kernel_id = ~0u;
  unsigned int unique_function_id = 0;
  uint64_t pc = 0;

  bool operator<(const TMAOperandLookupKey &other) const {
    return std::tie(trace_kernel_id, unique_function_id, pc) <
           std::tie(other.trace_kernel_id, other.unique_function_id, other.pc);
  }
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

// Exact per-site descriptor binding recovered offline
// (build_tma_pc_base_map.py). Version-2 records are keyed by
// (trace_kernel_id, unique_function_id, pc), which prevents static variants
// sharing a function id and PC from overwriting each other.
struct TMABaseLookupKey {
  unsigned int trace_kernel_id = ~0u;
  unsigned int unique_function_id = 0;
  uint64_t pc = 0;

  bool operator<(const TMABaseLookupKey &other) const {
    return std::tie(trace_kernel_id, unique_function_id, pc) <
           std::tie(other.trace_kernel_id, other.unique_function_id, other.pc);
  }
};

struct TMABaseRecord {
  // has_static_base: this site is tensormap-addressed and we recovered the real base
  // (UTMALDG/UTMASTG/UTMACCTL.PF). operand_addressed: this site is raw-pointer addressed
  // (UBLKRED/UBLKCP) — no tensormap; the base is the raw kernel-arg GMEM pointer read
  // offline from the by-value params struct (M2.5). raw_pointer_addressed marks that
  // operand_addressed case where we DID recover a real base + mock-tiling metadata, so
  // the simulator applies real base + visit-counter tiling instead of the synthetic
  // address (size still comes from operand covered_bytes, not a descriptor box).
  bool has_static_base = false;
  bool operand_addressed = false;
  bool raw_pointer_addressed = false;
  uint64_t global_base = 0;
  uint32_t tensor_rank = 0;
  uint32_t tensor_data_type = 0;
  uint32_t element_size = 0;
  std::array<uint32_t, 5> global_dim = {0, 0, 0, 0, 0};
  std::array<uint64_t, 5> global_strides = {0, 0, 0, 0, 0};
  std::array<uint32_t, 5> box_dim = {0, 0, 0, 0, 0};
  std::array<uint32_t, 5> element_strides = {0, 0, 0, 0, 0};
  uint32_t interleave = 0;
  uint32_t swizzle = 0;
  uint32_t l2_promotion = 0;
  uint32_t oob_fill = 0;
  // M2.5 raw-pointer mock-tiling metadata (only set when raw_pointer_addressed):
  // tile_bytes = covered_bytes (one transfer's span), num_tiles = region/tile_bytes.
  // The mover spreads transfers as global_base + (visit%num_tiles)*tile_bytes.
  uint32_t tile_bytes_operand = 0;
  uint32_t num_tiles = 0;
  std::string source;  // "utmacctl_chain" | "utmacctl_prefetch_edge" |
                       // "operand_raw_pointer"
};

struct TMASidecarMetadataDB {
  unsigned int base_map_version = 0;
  unsigned int operand_map_version = 0;
  bool completion_contracts_validated = false;
  std::map<std::string, TMADescriptorConfigMetadata> descriptor_configs;
  std::map<TMAOperandLookupKey, TMAOperandSiteRecord> operand_site_records;
  std::map<TMABaseLookupKey, TMABaseRecord> base_records;

  void clear() {
    base_map_version = 0;
    operand_map_version = 0;
    completion_contracts_validated = false;
    descriptor_configs.clear();
    operand_site_records.clear();
    base_records.clear();
  }

  bool empty() const {
    return descriptor_configs.empty() &&
           operand_site_records.empty() && base_records.empty();
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

// ---- Producer spin-wait sidecar (spin_wait_sidecar.json) --------------------
// A producer mbarrier spin loop, described statically by PC so the simulator can
// (behind a gate) block the warp at the loop's primary wait until the modeled
// barrier is ready, instead of replaying the capture-time poll count. See
// build_spin_wait_sidecar.py and .plan/SPIN_WAIT_POSTPROCESS.md.
struct SpinWaitRegionRecord {
  std::string block_id;
  uint64_t loop_head_pc = 0;
  uint64_t primary_wait_pc = 0;   // the poll whose barrier readiness gates the loop
  uint64_t loopback_bra_pc = 0;   // backward BRA that closes the loop
  uint64_t exit_pc = 0;           // fall-through target when the loop is left
  SyncInstructionKind primary_wait_kind = SyncInstructionKind::NONE;
  bool semantic_free_body = false;  // body has no TMA/GMMA/MUFU/arrive/memory op
  bool is_spin_loop = false;        // observed to iterate > 1 (not a single-shot wait)
  bool barrier_self_tied = false;   // whole-loop skip is SASS scoreboard net-zero
  // [Stage 1.4] Loop body PCs (loopback BRA excluded). The cursor-skip guard
  // allows advancing over exactly these PCs plus loopback_bra_pc; any other PC
  // encountered before exit_pc is a sidecar/trace drift and aborts loudly.
  std::vector<uint64_t> body_pcs;
  // The single authoritative fold gate (sidecar v2): the simulator folds
  // a region ONLY when this is true = is_spin_loop && semantic_free_body &&
  // barrier_self_tied. See .plan/SPIN_WAIT_SIM_MODELING.md 6.6/12.7.
  bool foldable = false;
};

// [Stage 1.4] Decomposition of the instructions the fold cursor-skip advanced
// over, classified from each skipped occurrence's opcode/sync class. The fold is
// only sound when NO semantic/local-memory/HGMMA/MUFU/TMA/BAR work is skipped;
// those fields MUST be zero (asserted at the skip site).
struct FoldSkipStats {
  unsigned int cursor_before_flush = 0;  // includes prefetched occurrences
  unsigned int cursor_after_flush = 0;   // after IBuffer rollback
  unsigned int cursor_after_skip = 0;    // after residual poll consumption
  unsigned int prefetched_rollback = 0;  // before_flush - after_flush
  unsigned int warp_insts = 0;          // total dynamic warp-instruction entries skipped
  unsigned int phasechk_trywait = 0;    // PHASECHK / TRYWAIT polls
  unsigned int nanosleep = 0;           // backoff/yield
  unsigned int imad_scratch = 0;        // loop-local scratch ALU
  unsigned int loopback_bra = 0;        // the closing backward BRA
  unsigned int local_memory = 0;        // STL/LDL — must be 0
  unsigned int semantic_memory = 0;     // global/shared memory refs — must be 0
  unsigned int hgmma = 0;               // must be 0
  unsigned int mufu = 0;                // must be 0
  unsigned int tma_family = 0;          // must be 0
  unsigned int bar_arrive_exch = 0;     // must be 0
  unsigned int other = 0;               // vetted scratch/control only
  std::map<uint64_t, unsigned int> pc_occurrences;
};

// Keyed by (funcid, primary_wait_pc): the simulator matches a PHASECHK/TRYWAIT at
// issue time against this to know it is inside a foldable spin loop.
struct SpinWaitSidecarDB {
  std::map<SyncOperandLookupKey, SpinWaitRegionRecord> regions_by_wait_pc;

  void clear() { regions_by_wait_pc.clear(); }

  bool empty() const { return regions_by_wait_pc.empty(); }
};

struct WaitBlockLookupKey {
  uint32_t trace_kernel_id = 0;
  uint32_t unique_function_id = 0;
  uint64_t action_pc = 0;

  bool operator<(const WaitBlockLookupKey &other) const {
    return std::tie(trace_kernel_id, unique_function_id, action_pc) <
           std::tie(other.trace_kernel_id, other.unique_function_id,
                    other.action_pc);
  }
};

struct MissingWaitRecord {
  std::string block_id;
  uint64_t entry_probe_pc = 0;
  uint64_t entry_branch_pc = 0;
  uint64_t captured_ready_pc = 0;
  SyncInstructionKind entry_probe_kind = SyncInstructionKind::NONE;
  bool partial_modeled_miss_aborts = false;
  bool partial_deferred_replay = false;
  std::vector<uint64_t> entry_arming_pcs;
  uint64_t miss_arm_entry_pc = 0;
  std::vector<uint64_t> miss_arm_wait_pcs;
  uint64_t miss_arm_loopback_pc = 0;
  uint64_t miss_arm_exit_pc = 0;
  uint64_t canonical_barrier_offset = 0;
  std::set<std::string> canonical_wait_state_operands;
  std::string partial_region_id;
  uint64_t sibling_branch_pc = 0;
  uint64_t join_pc = 0;
  uint64_t post_join_pc = 0;
  std::string join_kind;
  std::set<uint64_t> observed_entry_masks;
  std::set<uint64_t> observed_sibling_masks;
  std::set<uint64_t> observed_live_masks;
  uint64_t captured_ready_visits = 0;
  uint64_t captured_second_gate_visits = 0;
  uint64_t captured_poll_visits = 0;
  uint64_t all_live_entry_visits = 0;
  uint64_t partial_entry_visits = 0;
};

struct PartialWaitRegionRecord {
  std::string region_id;
  uint32_t trace_kernel_id = 0;
  uint32_t unique_function_id = 0;
  uint64_t sibling_branch_pc = 0;
  uint64_t join_pc = 0;
  uint64_t post_join_pc = 0;
  std::string join_kind;
  uint64_t bssy_pc = 0;
  std::string join_token;
  std::set<uint64_t> sibling_masks;
  std::set<uint64_t> live_masks;
};

struct WaitBlockSidecarDB {
  std::map<WaitBlockLookupKey, MissingWaitRecord> missing_wait_by_entry;
  std::map<WaitBlockLookupKey, SpinWaitRegionRecord> captured_poll_by_probe;
  std::map<std::string, PartialWaitRegionRecord> partial_regions_by_id;
  std::map<WaitBlockLookupKey, std::string> partial_region_id_by_join;
  std::map<WaitBlockLookupKey, std::string> partial_region_id_by_post_join;
  uint64_t discovered = 0;
  uint64_t passed = 0;
  uint64_t failed = 0;
  uint64_t skipped = 0;
  uint64_t metadata_only = 0;
  uint64_t ready_only_all_live_visits = 0;
  uint64_t ready_only_partial_visits = 0;
  uint64_t ready_only_records_with_partial_visits = 0;
  uint64_t partial_defer_records = 0;
  uint64_t partial_join_pending_visits = 0;
  uint64_t partial_join_already_issued_visits = 0;
  bool validation_complete = false;

  void clear() {
    missing_wait_by_entry.clear();
    captured_poll_by_probe.clear();
    partial_regions_by_id.clear();
    partial_region_id_by_join.clear();
    partial_region_id_by_post_join.clear();
    discovered = 0;
    passed = 0;
    failed = 0;
    skipped = 0;
    metadata_only = 0;
    ready_only_all_live_visits = 0;
    ready_only_partial_visits = 0;
    ready_only_records_with_partial_visits = 0;
    partial_defer_records = 0;
    partial_join_pending_visits = 0;
    partial_join_already_issued_visits = 0;
    validation_complete = false;
  }

  bool empty() const {
    return missing_wait_by_entry.empty() && captured_poll_by_probe.empty() &&
           partial_regions_by_id.empty();
  }
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
  // [Spin-wait fold audit] Monotonically-increasing generation: gen++ on
  // every phase flip, NEVER reset. The 1-bit `phase` above stays the sole driver
  // of the (bit-identical) non-fold issue logic; `gen` is read ONLY by the
  // fold-path generation audit, which needs to tell "my flip already passed"
  // (gen > expected) from "not yet" (gen < expected) — a distinction 1-bit parity
  // cannot make on a low-flip cross-warp barrier. See
  // .plan/SPIN_WAIT_SIM_MODELING.md 13.4/13.5.
  uint64_t gen = 0;
  // Last two flip coordinates are diagnostic-only. They make a poison audit
  // self-contained: expected/current generations can be correlated with the
  // exact gen1/gen2 window without replaying the full SYNCDBG stream.
  uint64_t prev_flip_cycle = 0;
  uint64_t last_flip_cycle = 0;
};

enum class WaitBlockPendingMode {
  NONE,
  MISSING_WAIT_HOLD,
  PARTIAL_MISSING_WAIT_HOLD,
  CAPTURED_POLL_HOLD,
};

enum class PartialWaitArmMode {
  NONE,
  ARMED,
  WAITING_FOR_SIBLING_JOIN,
  HOLDING,
};

struct HopperPartialWaitArm {
  bool valid = false;
  PartialWaitArmMode mode = PartialWaitArmMode::NONE;
  MissingWaitRecord record;
  HopperMBarrierKey key;
  uint64_t wait_state_raw = 0;
  uint64_t expected_gen = 0;
  uint64_t blocked_mask = 0;
  uint64_t sibling_mask = 0;
  uint32_t unique_function_id = 0;
  uint64_t arm_start_cycle = 0;
  uint64_t last_heartbeat_cycle = 0;
  bool source_was_deferred_replay = false;
  bool extracted_capsule = false;
};

struct HopperPartialReplayProgress {
  bool valid = false;
  std::string block_id;
  std::string region_id;
  uint32_t trace_kernel_id = 0;
  uint32_t unique_function_id = 0;
  uint32_t cta_id = 0;
  uint64_t start_cycle = 0;
  uint64_t last_heartbeat_cycle = 0;
  uint64_t last_progress_cycle = 0;
  uint64_t last_pc = 0;
  uint64_t post_join_pc = 0;
  uint64_t progressed_occurrences = 0;
  unsigned int initial_remaining = 0;
  unsigned int last_remaining = 0;
};

struct HopperMBarrierPendingWait {
  bool valid = false;
  bool is_trywait = false;
  HopperMBarrierKey key;
  uint64_t wait_state_raw = 0;
  WaitBlockPendingMode mode = WaitBlockPendingMode::NONE;
  std::string block_id;
  uint64_t action_pc = 0;
  uint64_t poll_loop_exit_pc = 0;
  unsigned int unique_function_id = 0;
  // Absolute barrier generation paired with this hold. Both strategies pin it
  // once after the action probe executes. A captured poll consumes residual
  // loop occurrences on release; an all-live missing wait never changes either
  // cursor; a partial missing wait replays occurrences extracted before hold.
  uint64_t expected_gen = 0;
  uint64_t hold_start_cycle = 0;
  uint64_t last_heartbeat_cycle = 0;
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
  // Real GMEM base recovered from the exact (uid,pc) base map. When has_real_base is
  // true the mover computes addr = global_base + agu_index*128 (HW-faithful, L2 reuse)
  // instead of the synthetic (transfer_uid<<20)+agu_index*128. Populated in
  // build_tma_command where (uid,pc) is still known; carried to mover_issue_requests.
  uint64_t global_base = 0;
  bool has_real_base = false;
  // M2 (visit-counter tile spread): per-tensor tile this transfer targets, so the
  // mover adds tile_idx*tile_bytes to spread transfers across the tensor's tiles
  // (restores the cold-miss of first-touch that base-only folds away). Set in
  // build_tma_command from the per-base visit counter; only meaningful when
  // has_real_base is true. tile_bytes = Πbox_dim·element_size (one tile's byte span).
  uint64_t tile_offset_bytes = 0;
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
  int cycle_first_request_issued = -1;
  int cycle_last_request_issued = -1;
  int cycle_last_completion = -1;
  uint32_t issue_active_cycles = 0;
  uint32_t icnt_full_cycles = 0;
  // Debug-only: ensure the first read mf and first write mf of a store/reduce
  // transfer are each logged exactly once (so RMW issue is observable).
  bool logged_first_write = false;
  bool logged_first_read = false;
  // Debug-only: log interconnect back-pressure (icnt full) at most once per
  // transfer, so a recurring stall is observable without flooding the trace.
  bool logged_backpressure = false;
};
