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
- `utmaldg_runtime_debug.csv`
  - runtime consumer-side `desc[URx]` data
- `utmaldg_producer_debug.csv`
  - producer-side uniform-register setup info
- `tma_descriptor_configs.json`
  - normalized simulator-facing descriptor configs
- `tma_descriptor_resolver.json`
  - mapping from runtime TMA consumer to `config_id`

For simulator modeling, the most important outputs are:

- `tma_descriptor_configs.json`
- `tma_descriptor_resolver.json`

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

- `desc_reg_ids`
  - uniform descriptor register ids observed for this consumer group

- `sample_count`
  - number of runtime samples grouped into this entry

- `confidence`
  - current confidence level of the mapping

- `mapping_method`
  - resolver-level mapping method for this entry

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

it should:

1. extract:
   - `unique_function_id`
   - `pc_hex`
   - `handle_hi`
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

And it enables:

- one semantic config reused across many dump rows
- one resolver method reused across many TMA sites
- incremental extension to more `desc[URx]` TMA ops later

---

## Current known mappings

For the currently analyzed FlashAttention-3 trace:

- `pc 0x94e0`, `handle_hi 0x14f00000`
  - maps to `tm_r4_dt9_box_64x192x1x1`

- `pc 0x96b0`, `handle_hi 0x12f00000`
  - maps to `tm_r4_dt9_box_64x128x1x1`

- `pc 0x9a80`, `handle_hi 0x14f00000`
  - likely maps to `tm_r4_dt9_box_64x192x1x1`

This is only the first TMA family. The same format should be extended to later `desc[URx]` TMA ops as more mappings are derived.
