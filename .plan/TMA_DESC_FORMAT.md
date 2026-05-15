# TMA Descriptor Config and Resolver Format

## Purpose

This document defines the simulator-facing TMA descriptor artifacts produced from trace generation.

The goal is:

- do **not** make the simulator interpret raw `desc[URx]` handles directly
- do **not** make the simulator parse raw `CUtensorMap` blobs
- instead, provide:
  - a normalized descriptor config table
  - a resolver table that maps runtime TMA consumers to one config

This keeps the simulator logic stable even as more TMA ops are added later.

---

## Output files

The trace-generation flow now produces these TMA-related outputs under `traces/extra_info/`:

- `tensor_map_encode_dump.csv`
  - raw semantic descriptor captures from `cuTensorMapEncodeTiled`
- `tensor_map_encode_blobs/*.bin`
  - exact encoded `CUtensorMap` blobs
- `tma_desc_runtime_debug.csv`
  - runtime consumer-side `desc[URx]` data
- `tma_desc_producer_debug.csv`
  - producer-side uniform-register setup info
- `tma_descriptor_configs.json`
  - normalized simulator-facing descriptor configs
- `tma_descriptor_resolver.json`
  - mapping from runtime TMA consumer to `config_id`
- `tma_runtime_operand_debug.jsonl`
  - raw runtime operand callback captures for TMA-family instructions
- `tma_operand_resolver.json`
  - simulator-facing operand roles, runtime-observed values, and static decode formulas

For simulator modeling, the most important outputs are:

- `tma_descriptor_configs.json`
- `tma_descriptor_resolver.json`
- `tma_operand_resolver.json`

---

## Design overview

The design splits into two layers:

### Layer 1 — Descriptor configs

This file contains the normalized semantic tensor-map families.

It stores the fields the simulator needs:

- `tensor_rank`
- `tensor_data_type`
- `global_dim`
- `global_strides`
- `box_dim`
- `element_strides`
- `interleave`
- `swizzle`
- `l2_promotion`
- `oob_fill`

It intentionally does **not** require:

- `tensor_map_ptr_hex`
- `global_address_hex`
- raw blob bytes

unless later address-sensitive modeling is required.

### Layer 2 — Resolver

This file maps a runtime TMA consumer instance to a descriptor config.

The current best lookup key is:

- `unique_function_id`
- `pc_hex`
- `handle_hi_hex`

This is better than using only the descriptor register number because:

- the same `desc[URx]` register pair can carry different descriptor families at different PCs
- the observed high 32-bit descriptor handle is the stable family signal in the current trace

However, later trace work shows that this key is not always sufficient by itself:

- `UTMALDG.*`
  - usually has a useful nonzero `handle_hi_hex`
- `UBLKRED.*`
  - may need rank inference from same-function related uses because the opcode text does not always encode `4D` / `5D`
- `UTMASTG.*`
  - may expose `handle_hi_hex = 0x00000000`
  - so final resolution may need same-function `desc_reg_ids` reuse or single-rank-candidate fallback

---

## `tma_descriptor_configs.json`

### File role

`tma_descriptor_configs.json` is the source of truth for simulator-side TMA descriptor semantics.

The simulator should load this file once and index it by `config_id`.

### Schema

```json
{
  "version": 1,
  "source": {
    "tensor_map_rows": 0,
    "normalized_config_count": 0
  },
  "configs": [
    {
      "config_id": "string",
      "tensor_rank": 0,
      "tensor_data_type": 0,
      "global_dim": [0],
      "global_strides": [0],
      "box_dim": [0],
      "element_strides": [0],
      "interleave": 0,
      "swizzle": 0,
      "l2_promotion": 0,
      "oob_fill": 0,
      "source_dump_ids": [0]
    }
  ]
}
```

### Field meanings

- `version`
  - schema version

- `source.tensor_map_rows`
  - number of raw rows read from `tensor_map_encode_dump.csv`

- `source.normalized_config_count`
  - number of unique semantic descriptor families after deduplication

- `config_id`
  - stable normalized identifier for one descriptor family

- `tensor_rank`
  - number of tensor dimensions

- `tensor_data_type`
  - data type enum from CUDA tensor-map encoding

- `global_dim`
  - logical tensor shape

- `global_strides`
  - explicit outer-dimension strides

- `box_dim`
  - tensor tile / transfer box dimensions

- `element_strides`
  - per-dimension element stepping

- `interleave`
  - tensor-map interleave mode

- `swizzle`
  - tensor-map swizzle mode

- `l2_promotion`
  - cache / L2 promotion policy

- `oob_fill`
  - out-of-bounds fill mode

- `source_dump_ids`
  - trace-local dump rows that were merged into this normalized family

### Example

```json
{
  "version": 1,
  "source": {
    "tensor_map_rows": 12,
    "normalized_config_count": 2
  },
  "configs": [
    {
      "config_id": "tm_r4_dt9_box_64x192x1x1",
      "tensor_rank": 4,
      "tensor_data_type": 9,
      "global_dim": [64, 2048, 24, 1],
      "global_strides": [3072, 128, 6291456],
      "box_dim": [64, 192, 1, 1],
      "element_strides": [1, 1, 1, 1],
      "interleave": 0,
      "swizzle": 3,
      "l2_promotion": 2,
      "oob_fill": 0,
      "source_dump_ids": [0]
    },
    {
      "config_id": "tm_r4_dt9_box_64x128x1x1",
      "tensor_rank": 4,
      "tensor_data_type": 9,
      "global_dim": [64, 2048, 24, 1],
      "global_strides": [3072, 128, 6291456],
      "box_dim": [64, 128, 1, 1],
      "element_strides": [1, 1, 1, 1],
      "interleave": 0,
      "swizzle": 3,
      "l2_promotion": 2,
      "oob_fill": 0,
      "source_dump_ids": [1, 2, 3, 4]
    }
  ]
}
```

---

## `tma_descriptor_resolver.json`

### File role

`tma_descriptor_resolver.json` maps a dynamic or semi-dynamic TMA consumer to one config family.

The simulator should not inspect raw dump ids. It should ask the resolver which `config_id` applies to the current instruction instance.

### Schema

```json
{
  "version": 1,
  "source": {
    "runtime_group_count": 0,
    "mapping_method": "string"
  },
  "resolver": [
    {
      "unique_function_id": 0,
      "pc_hex": "0x0",
      "handle_hi_hex": "0x0",
      "desc_reg_ids": [0],
      "sample_count": 0,
      "confidence": "medium",
      "mapping_method": "string",
      "config_id": "string"
    }
  ]
}
```

### Field meanings

- `version`
  - schema version

- `source.runtime_group_count`
  - number of grouped runtime consumer signatures

- `source.mapping_method`
  - high-level mapping strategy used by the generator

- `unique_function_id`
  - traced kernel/function identity used by the enhanced trace

- `pc_hex`
  - TMA consumer PC

- `handle_hi_hex`
  - high 32-bit runtime handle observed in `desc[URx+1]`
  - may be `0x00000000` for some desc-like ops such as `UTMASTG.*`

- `desc_reg_ids`
  - uniform descriptor register ids observed for this consumer group
  - for `UTMASTG.*`, this records the first bracketed uniform-register pair base, treated as desc-like

- `sample_count`
  - number of runtime samples grouped into this entry

- `confidence`
  - current confidence level of the mapping

- `mapping_method`
  - resolver-level mapping method for this entry
  - examples now include:
    - `handle_hi_to_box_dim_family_with_opcode_rank`
    - `handle_hi_to_box_dim_family_with_inferred_rank_from_desc_reg_reuse`
    - `handle_hi_to_box_dim_family_with_inferred_rank_from_handle_reuse`
    - `same_function_desc_reg_config_reuse`
    - `single_rank_candidate_config`

- `config_id`
  - normalized config selected for the consumer

### Example

```json
{
  "version": 1,
  "source": {
    "runtime_group_count": 3,
    "mapping_method": "handle_hi_to_box_dim_family"
  },
  "resolver": [
    {
      "unique_function_id": 3,
      "pc_hex": "0x94e0",
      "handle_hi_hex": "0x14f00000",
      "desc_reg_ids": [10],
      "sample_count": 18,
      "confidence": "medium",
      "mapping_method": "handle_hi_to_box_dim_family",
      "config_id": "tm_r4_dt9_box_64x192x1x1"
    },
    {
      "unique_function_id": 3,
      "pc_hex": "0x96b0",
      "handle_hi_hex": "0x12f00000",
      "desc_reg_ids": [18],
      "sample_count": 20,
      "confidence": "medium",
      "mapping_method": "handle_hi_to_box_dim_family",
      "config_id": "tm_r4_dt9_box_64x128x1x1"
    }
  ]
}
```

---

## How the simulator should use these files

### Recommended flow

At instruction issue time, when the simulator sees a TMA consumer like:

- `UTMALDG.* ... desc[URx]`
- `UBLKRED.* ... desc[URx]`
- `UTMASTG.* [URa], [URb]`

it should:

1. extract:
   - `unique_function_id`
   - `pc_hex`
   - `handle_hi`
   - `desc_reg_id` when available
2. build a resolver key:
   - `(unique_function_id, pc_hex, handle_hi_hex)`
3. query `tma_descriptor_resolver.json`
4. obtain `config_id`
5. look up that `config_id` in `tma_descriptor_configs.json`
6. simulate the TMA op using the descriptor semantics from the config

### Pseudocode

```cpp
ResolverKey key = {
    unique_function_id,
    pc_hex,
    handle_hi_hex
};

auto resolver_it = resolver_table.find(key);
if (resolver_it == resolver_table.end()) {
    return TM_CONFIG_UNKNOWN;
}

auto cfg_it = config_table.find(resolver_it->second.config_id);
if (cfg_it == config_table.end()) {
    return TM_CONFIG_UNKNOWN;
}

const TmaConfig &cfg = cfg_it->second;
simulate_utmaldg(cfg, inst_state);
```

### Why this is the right abstraction

This design avoids:

- making the simulator decode CUDA blob bytes
- making the simulator infer descriptor identity from register numbers
- coupling the simulator to raw tracer dump ids

---

## `tma_operand_resolver.json`

### File role

`tma_operand_resolver.json` is the simulator-facing operand metadata file for TMA-family instructions whose runtime behavior depends on operand values beyond descriptor resolution alone.

This is especially important for:

- bulk `UBLKRED` without `desc[URx]`
- bulk `UBLKCP` without `desc[URx]`
- future mixed forms where a descriptor exists but additional control operands still affect execution semantics

It is generated from:

- `tma_discovery.json`
- `tma_runtime_operand_debug.jsonl`
- optionally `tma_descriptor_resolver.json` when a descriptor-backed association exists

### Design split

Keep descriptor and operand semantics separate:

- `tma_descriptor_configs.json`
  - tensor-map semantics
- `tma_descriptor_resolver.json`
  - descriptor-backed PC → config mapping
- `tma_operand_resolver.json`
  - operand roles, runtime-observed values, and static decode formulas

### Schema

```json
{
  "version": 1,
  "source": {
    "discovery_entries": 0,
    "runtime_events": 0
  },
  "resolver": [
    {
      "unique_function_id": 1,
      "pc_hex": "0x370",
      "opcode": "UBLKCP.S.G",
      "text": "/*0390*/ UBLKCP.S.G [UR12], [UR6], UR10 ;",
      "role": "copy",
      "operand_form": "bulk",
      "operands": {
        "operand_1": {
          "position": 1,
          "text": "[UR12]",
          "reg_ids": [12],
          "kind": "dst_smem_base_or_cursor"
        },
        "operand_2": {
          "position": 2,
          "text": "[UR6]",
          "reg_ids": [6],
          "kind": "src_gmem_base_or_cursor"
        },
        "operand_3": {
          "position": 3,
          "text": "UR10",
          "reg_ids": [10],
          "kind": "covered_bytes_or_encoded_span"
        }
      },
      "runtime_observed_values": {
        "operand_3": {
          "callback_index": 2,
          "operand_type": "UNIFORM",
          "mem_type": "NONE",
          "operand_reg_ids": [10],
          "raw_value_lo_samples": [16],
          "sample_count": 1,
          "decoded_byte_samples": [256]
        }
      },
      "static_decode_formula": {
        "kind": "sass_validated_formula",
        "applies_to": "bulk_ublkcp",
        "operands": {
          "operand_3": {
            "kind": "covered_bytes",
            "encoding": "16B_units",
            "formula": {
              "covered_bytes": "operand_3 * 16"
            }
          }
        }
      }
    }
  ]
}
```

### Bulk `UBLKCP` interpretation

For the validated non-descriptor bulk form:

```text
UBLKCP.S.G [UR12], [UR6], UR10
```

the resolver should interpret:

- operand 1
  - shared-memory destination base / cursor
- operand 2
  - global-memory source base / cursor
- operand 3
  - encoded covered span in units of 16 bytes

Decode rule:

```text
covered_bytes = operand_3 * 16
```

This matches the resolver style already used by bulk `UBLKRED`:

- top-level operand entry
  - `kind = covered_bytes_or_encoded_span`
- semantic decode entry
  - `kind = covered_bytes`
  - `encoding = 16B_units`
  - `formula.covered_bytes = operand_3 * 16`

### Bulk `UBLKRED` interpretation

For the validated non-descriptor bulk form:

```text
UBLKRED.G.S.ADD.F32.RN [UR8], [UR4], UR6
```

the resolver should interpret:

- operand 1
  - destination base
- operand 2
  - source base
- operand 3
  - encoded covered span in units of 16 bytes

Decode rule:

```text
covered_bytes = operand_3 * 16
covered_elements_f32 = operand_3 * 4
```

### Descriptor-backed `UBLKRED` interpretation

For descriptor-backed forms such as:

```text
UBLKRED.G.S.ADD.F32.RN [UR10], [UR4], UR12, desc[UR16]
```

the current FA3 evidence supports a more conservative model:

- operand 1
  - destination-side runtime state
  - not yet proven to be a complete final destination address by itself
- operand 2
  - source-side runtime state
- operand 3
  - size/span-style control operand
  - observed raw runtime value is stable across multiple descriptor-backed `UBLKRED` PCs in the analyzed FA3 trace
- `desc[URx]`
  - tensor-map layout / address-resolution context

The key observation is that descriptor-backed `UBLKRED` still needs the descriptor even when operand 3 already carries size-like meaning, because the descriptor contributes tensor-layout metadata that the other operands do not encode.

For descriptor-backed entries:

- keep `raw_value_lo_samples` as the authoritative runtime observation
- do **not** automatically apply the bulk `operand_3 * 16` decode rule unless it has been validated for that descriptor-backed form
- use `static_decode_formula.kind = "descriptor_backed_formula_pending"` until the descriptor-backed size rule is directly proven

### Recommended simulator consumption path

For descriptor-backed TMA forms:

1. resolve descriptor semantics through:
   - `tma_descriptor_resolver.json`
   - `tma_descriptor_configs.json`
2. resolve extra operand semantics through:
   - `tma_operand_resolver.json`

For bulk non-descriptor `UBLKRED`:

1. look up `(unique_function_id, pc_hex)` in `tma_operand_resolver.json`
2. read:
   - `operand_form`
   - `operands`
   - `runtime_observed_values`
   - `static_decode_formula`
3. decode the covered span from operand 3
4. simulate the reduce-store using operand 1 / operand 2 / operand 3 semantics

For bulk non-descriptor `UBLKCP`:

1. look up `(unique_function_id, pc_hex)` in `tma_operand_resolver.json`
2. read:
   - `operand_form`
   - `operands`
   - `runtime_observed_values`
   - `static_decode_formula`
3. decode the covered span from operand 3
4. simulate the GMEM → SMEM copy using operand 1 / operand 2 / operand 3 semantics

For descriptor-backed `UBLKRED`:

1. look up `(unique_function_id, pc_hex)` in `tma_operand_resolver.json`
2. read:
   - `operands`
   - `runtime_observed_values`
   - `descriptor_ref`
   - `static_decode_formula`
3. use `tma_descriptor_resolver.json` and `tma_descriptor_configs.json` to recover the referenced tensor-map layout
4. interpret operand 1 as destination-side runtime state resolved through the descriptor
5. treat operand 3 as the runtime size/span control operand, but keep its decode rule conservative until directly validated for that form

### Notes on raw runtime fields

- `raw_value_lo_samples`
  - primary captured runtime payload for the operand callback
- `raw_value_hi_samples`
  - optional companion high slot when the traced operand really uses an additional value
- `raw_value_2_samples`, `raw_value_3_samples`
  - only kept when they contain nontrivial data
  - absent for simple bulk `UBLKRED` operand-3 cases because those slots are not meaningful there

### `UTMAREDG` simulator guidance

For `UTMAREDG`, the right abstraction depends on simulator scope:

- if the simulator only needs a coarse throughput-level model
  - amount of data
  - reduction op kind
  - issue / completion cost
  may be sufficient
- if the simulator needs layout-aware placement, locality, or correctness-sensitive address behavior
  - descriptor-derived tensor-map semantics still matter

So `UTMAREDG` can be simplified to amount + op kind only for a deliberately coarse model, but not for a descriptor-faithful model.

### `UTMALDG` / `UTMASTG` resolver labels

`UTMALDG` and `UTMASTG` now use explicit resolver labels when a descriptor-backed tensor-map association exists.

The intended interpretation is:

- `operand_form = "descriptor_shape_driven"`
  - the logical transfer shape is primarily carried by the descriptor
- `inferred_runtime_semantics.descriptor.shape_role = "descriptor_shape_driven"`
  - the descriptor is the main source of tensor-map shape / tile interpretation
- `inferred_runtime_semantics.operand_model.kind = "operand_state_driven"`
  - runtime operands behave like address/state/coordinate inputs rather than a simple direct byte-count field

Typical operand labels are:

- `UTMALDG`
  - `operand_1 = load_dst_state`
  - `operand_2 = load_coord_or_state`
  - `operand_3 = tensor_map_descriptor`
- `UTMASTG`
  - `operand_1 = store_dst_state`
  - `operand_2 = store_src_or_state`

This labeling is intended to distinguish these instructions from bulk `UBLKRED`, where a dedicated runtime operand appears to directly control the covered span.

And it enables:

- one semantic config reused across many dump rows
- one resolver method reused across many TMA sites
- incremental extension to more `desc[URx]` TMA ops later
- trace-backed mapping even when a TMA op does not expose a useful nonzero `handle_hi`

---

## Current known mappings

For the currently analyzed FlashAttention-3 trace:

- `pc 0x94e0`, `handle_hi 0x14f00000`
  - maps to `tm_r4_dt9_box_64x192x1x1`

- `pc 0x96b0`, `handle_hi 0x12f00000`
  - maps to `tm_r4_dt9_box_64x128x1x1`

- `pc 0x9a80`, `handle_hi 0x14f00000`
  - likely maps to `tm_r4_dt9_box_64x192x1x1`

- `UBLKRED` first family in `unique_function_id = 8`
  - `pc 0x90a0`, `0x91d0`, `0x94e0`, `0x95b0`, `0x96f0`, `0x97c0`
  - maps to `tm_r4_dt9_box_64x192x1x1`
  - resolved generically through inferred-rank reuse, not opcode-specific hard-coding

- `UTMASTG.5D` at `pc 0x8210`
  - uses desc-like first operand pair `UR8/UR9`
  - maps to `tm_r5_dt9_box_64x192x1x1x1`
  - resolved by `single_rank_candidate_config`

- `UTMASTG.4D` at `pc 0x7ed0` and `0x7ef0`
  - use desc-like first operand pair `UR8/UR9`
  - map to `tm_r4_dt9_box_64x192x1x1`
  - resolved by `same_function_desc_reg_config_reuse` because `handle_hi_hex` is zero at runtime

This is only the first TMA family. The same format should be extended to later `desc[URx]` TMA ops as more mappings are derived.
