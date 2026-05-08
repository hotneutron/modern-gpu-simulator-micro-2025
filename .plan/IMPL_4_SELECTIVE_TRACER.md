# Implementation Plan: [4] Selective Tracer

**File:** `simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu` (modify)
**Depends on:** `representatives.json` from [3]
**Produces:** Sparse `traces/threadblocks/` with only K `.pb` files per kernel

---

## How the Existing Tracer Works (Relevant Parts)

The tracer receives memory access records through a GPU→CPU channel. Each record
contains `cta_id_x`, `cta_id_y`, `cta_id_z`. The consumer loop at line ~1666:

```cpp
if (ma->cta_id_x == -1) {
    // kernel-end sentinel
}
// Key: d_{device}_s_{stream}_k_{kernel}_{x},{y},{z}
std::string tb_string_id = "d_" + ... + std::to_string(ma->cta_id_x) + "," + ...;
dynamic_trace::dim3d *cta_id = tb.mutable_block_id();
cta_id->set_x(ma->cta_id_x);
```

All CTAs are currently written unconditionally. The filter goes here.

---

## Changes Required

### 1. Add allowlist data structure (global, loaded at startup)

```cpp
// tracer_tool.cu — add near top with other globals

#include <unordered_set>
#include <nlohmann/json.hpp>   // or manual JSON parse to avoid dependency

struct CTAKey {
    int32_t x, y, z;
    bool operator==(const CTAKey &o) const {
        return x==o.x && y==o.y && z==o.z;
    }
};
struct CTAKeyHash {
    size_t operator()(const CTAKey &k) const {
        return std::hash<int64_t>()(((int64_t)k.x << 32) ^ ((int64_t)k.y << 16) ^ k.z);
    }
};

// kernel_id (0-based) → set of allowed CTAs
// empty map = no filter (trace all CTAs)
static std::unordered_map<int, std::unordered_set<CTAKey, CTAKeyHash>> g_cta_allowlist;
static bool g_allowlist_active = false;
```

### 2. Load allowlist at `nvbit_at_init()`

```cpp
void nvbit_at_init() {
    // ... existing init code ...

    const char *allowlist_path = std::getenv("TRACER_CTA_ALLOWLIST");
    if (allowlist_path) {
        load_cta_allowlist(allowlist_path);
        g_allowlist_active = true;
        std::cout << "CTA allowlist loaded from: " << allowlist_path << "\n";
    }
}

void load_cta_allowlist(const char *path) {
    // Parse representatives.json
    // Format: {"kernel_3": {"representatives": [{"x":0,"y":0,"z":0}, ...]}, ...}
    // kernel name → kernel_id mapping done lazily at first kernel launch
    std::ifstream f(path);
    // ... parse JSON, populate g_cta_allowlist ...
}
```

JSON parsing: use a minimal header-only parser (nlohmann/json.hpp single-header,
or a simple hand-rolled parser to avoid adding a dependency to the NVBit build).

### 3. Map kernel names to IDs at kernel launch

```cpp
// In nvbit_at_cuda_event, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
void nvbit_at_cuda_event(...) {
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel && !is_exit) {
        // ... existing kernel-launch handling ...

        if (g_allowlist_active) {
            // Map kernel name → internal kernel_id
            // g_kernel_name_to_allowlist_id populated from JSON keys
            register_kernel_allowlist(kernel_name, kernel_id[device_id]);
        }
    }
}
```

### 4. Filter in the consumer loop (line ~1666)

```cpp
// In the channel consumer thread:
while (recv_thread_started) {
    // ... receive ma from channel ...

    if (ma->cta_id_x == -1) { /* sentinel */ }

    // ---- ADD FILTER HERE ----
    if (g_allowlist_active) {
        int kid = kernel_id[device_id] - 1;
        auto it = g_cta_allowlist.find(kid);
        if (it != g_cta_allowlist.end()) {
            CTAKey key{ma->cta_id_x, ma->cta_id_y, ma->cta_id_z};
            if (it->second.find(key) == it->second.end()) {
                continue;   // skip this CTA's records
            }
        }
    }
    // ---- END FILTER ----

    // existing: write to protobuf
    std::string tb_string_id = ...;
}
```

### 5. Write sidecar weights file

After tracing completes, write `cta_weights.json` alongside `dynamic_trace.pb`:

```cpp
void nvbit_at_term() {
    // ... existing cleanup ...

    if (g_allowlist_active) {
        write_cta_weights_sidecar(output_dir);
    }
}

void write_cta_weights_sidecar(const std::string &dir) {
    // Emit JSON: kernel_id → [{x,y,z,weight}, ...]
    // Weights come from the loaded allowlist (parsed from representatives.json)
    std::ofstream f(dir + "/cta_weights.json");
    // ... write ...
}
```

---

## Environment Variable Interface

```bash
# Enable CTA filter
TRACER_CTA_ALLOWLIST=./representatives.json \
  LD_PRELOAD=./tracer_nvbit.so \
  ./workload args

# Disable (default — trace all CTAs)
LD_PRELOAD=./tracer_nvbit.so ./workload args
```

No changes to the build system needed — the allowlist path is runtime-configurable.

---

## Instrumentation Side (inject_funcs.cu)

No changes needed. The filter operates on the CPU side (consumer thread), after
data has been sent through the channel. This avoids any GPU-side overhead for
non-representative CTAs — their instructions are still instrumented and sent,
but dropped on the CPU side.

**Future optimization:** push the filter to the GPU side by passing the allowlist
as a constant memory array, reducing channel bandwidth for large kernels. Not
needed for correctness; defer.

---

## Output

The sparse `traces/threadblocks/` directory contains only `.pb` files for the K
representative CTAs. The `cta_weights.json` sidecar enables the simulator to:
1. Load the correct weight per representative
2. Apply weight-scaled stat aggregation

---

## Testing

1. Run full trace → count `.pb` files
2. Run with `TRACER_CTA_ALLOWLIST` → verify only K `.pb` files exist
3. Verify `.pb` files correspond exactly to the listed `{x,y,z}` coordinates
4. Simulate both; compare per-representative stats (should be identical)
5. Verify `cta_weights.json` weights sum to 1.0 per kernel

---

## Notes

- The filter key is `(kernel_id, cta_x, cta_y, cta_z)`. Kernel ID is the
  internal sequential counter (`kernel_id[device_id]`), not the CUDA kernel
  registration order. The JSON uses kernel names; mapping to IDs is done at
  first launch.
- Multiple invocations of the same kernel name get the same allowlist applied.
  This is correct: representative CTAs are stable across invocations for
  regular kernels.
- For kernels not in the allowlist JSON, all CTAs are traced (safe default).
