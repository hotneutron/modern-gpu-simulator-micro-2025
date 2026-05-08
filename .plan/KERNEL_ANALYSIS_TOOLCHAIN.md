# Kernel/Trace Analysis Toolchain Plan

## Vision

An end-to-end pipeline that takes a GPU workload and produces accurate simulated
performance metrics at a fraction of the cost of full simulation.

```
Workload binary + dataset
        │
        ▼
[1] Kernel Profiler
        │  per-kernel: latency, hotspots, instruction mix, memory pressure
        │  per-CTA: behavior variation, feature vectors
        ▼
[2] Hotspot Analyzer + Classifier
        │  rank kernels by simulation cost × importance
        │  classify: compute / memory / mixed (roofline)
        ▼
[3] CTA Clustering
        │  K-Means on per-CTA features → K representative CTAs per kernel
        ▼
[4] Selective Tracer
        │  NVBit with CTA allowlist → full SASS trace only for K reps
        ▼
[5] Adaptive CTA Simulator
        │  replicate reps to fill active SMs, pilot loop, weighted stats
        ▼
    Per-kernel performance report
```

---

## Component Design

### [1] Kernel Profiler

**Goal:** Characterize every kernel invocation — latency, hotspots, CTA-level variation.

**Tool:** `ncu --csv --metrics <list>` (Nsight Compute CLI), two passes:

**Pass A — kernel-level (fast, full workload):**
- Execution time / cycle count per invocation
- Instruction mix: FP32, FP64, INT, tensor core (TC), SFU, loads, stores, branches
- L1/L2 hit rates, DRAM bandwidth utilization
- Warp occupancy, stall breakdown (memory, compute, sync, other)
- Memory coalescing efficiency, transaction size distribution

**Pass B — CTA-level (targeted, hot kernels only):**
- Same metrics at per-CTA granularity
- Input to [3] CTA clustering
- Only run on kernels above hotspot threshold (top N% by execution time)

**Hotspot identification:**
```
hotspot_score(k) = mean_exec_time(k) × invocation_count(k)
```
Rank kernels by score; top-K are candidates for trace collection.

**Outputs:**
- `kernel_report.csv` — one row per kernel invocation
- `cta_features_{kernel_id}.csv` — one row per CTA (Pass B only)
- `hotspot_ranking.txt` — sorted by hotspot_score

**Implementation:** `tools/kernel_profiler.py`

---

### [2] Hotspot Analyzer + Classifier

**Goal:** Decide which kernels are worth tracing/simulating, and how to simulate them.

**Roofline classification:**
```
ridge_point = peak_FLOPS / peak_DRAM_BW          (from GPU config)
kernel_AI   = (FP + INT + W_tc·TC + W_sfu·SFU) / bytes_to_DRAM
ridge_ratio = kernel_AI / ridge_point

class = compute   if ridge_ratio >= T_high AND mem_stall_frac < 0.2
      = memory    if ridge_ratio <= T_low  OR  mem_stall_frac > 0.5
      = mixed     otherwise
```
Initial thresholds: `T_low=0.9, T_high=1.3` (calibrate per architecture).

**Per-kernel recommendations:**
- Classification (compute/memory/mixed)
- Recommended `sim_ctas` for the adaptive simulator
- Estimated simulation speedup vs. full run
- Whether CTA clustering (Pass B) is needed:
  - Regular (GEMM, stencil): coordinate heuristic sufficient, skip clustering
  - Irregular (sparse, graph, reduction): clustering required

**Output:** `kernel_classifier_report.csv`

**Implementation:** `tools/kernel_classifier.py`

---

### [3] CTA Clustering

**Goal:** Select K representative CTAs per kernel that cover the behavioral diversity
of all CTAs, so simulating K reps gives accurate aggregate statistics.

**Algorithm:**
1. Load per-CTA feature vectors from `cta_features_{kernel_id}.csv`
2. Normalize features (z-score per dimension)
3. K-Means with silhouette score to select K:
   - Regular kernels: K=2–5
   - Irregular kernels: K=10–20
4. Select medoid per cluster (CTA closest to centroid)
5. Cluster weight = cluster_size / total_CTAs

**Feature vector (~7D):**
| Feature | Captures |
|---|---|
| Active warps | Occupancy variation |
| L1 hit rate | Memory locality |
| L2 hit rate | Off-chip traffic |
| Issued instructions | Work imbalance |
| Memory transactions | Load/store divergence |
| Branch divergence | Control flow variation |
| Shared memory accesses | Collaborative patterns |

**Output:** `representatives.json`
```json
{
  "kernel_1": [
    {"x": 0, "y": 0, "z": 0, "weight": 0.25},
    {"x": 4, "y": 3, "z": 0, "weight": 0.75}
  ]
}
```

**Fallback for regular kernels:** Use coordinate heuristic from `compute_sampled_ctas()`
in `main.cc` — zero cost, K≈9, valid for GEMM/stencil/conv.

**Implementation:** `tools/cta_cluster.py`

---

### [4] Selective Tracer

**Goal:** Generate full SASS traces only for the K representative CTAs, not all CTAs.

**Tool:** Modify `simulator-remodeled/util/tracer_nvbit`

**Change:** In the NVBit instrumentation callback, check `blockIdx` against the allowlist
from `representatives.json`; skip recording for non-representative CTAs.

**Interface:**
```bash
tracer_nvbit --cta-allowlist representatives.json -- ./workload args
```

**Output:** Sparse `traces/threadblocks/` directory — only K `.pb` files per kernel
instead of `total_CTAs` files.

**Speedup:** `total_CTAs / K` for trace generation (200–500× for regular kernels).

**Note:** CTA weights from [3] are embedded in a sidecar file alongside the traces
so the simulator can apply them at stat aggregation time.

---

### [5] Adaptive CTA Simulator

**Goal:** Simulate only the K representative CTA traces with enough active SMs to
reproduce real occupancy and inter-CTA contention, then scale stats by weight.

**Already partially implemented** on `cta-sampling` branch. See `CTA_SAMPLING.md`
and `.plan/CTA_SAMPLING_STATUS.md` for full status.

**Key behaviors:**
- Replicate K reps to fill `sim_ctas` slots using stratified shuffle (not round-robin)
- Adaptive pilot loop: double `sim_ctas` until DRAM BW stabilizes
- Stat scaling: cycles unscaled (wall-clock time), instructions and CTA count × weight
- Cycle estimation: throughput-conservation formula or log-fit from pilot history

**Remaining work:**
- Fix 3 failure modes (see `.plan/CTA_SAMPLING_STATUS.md` §3)
- Accept `representatives.json` weights from [3] instead of uniform weight
- Per-kernel accuracy report in validation mode

---

## Glue Script

**`tools/trace_selector.py`** — orchestrates the full pipeline:

```bash
trace_selector.py \
  --workload ./app \
  --args "input1 input2" \
  --gpu-config tools/configs/sm86_rtx3080.json \
  --hotspot-top-k 5 \
  --output-dir ./output/
```

Steps run in sequence:
1. kernel_profiler.py Pass A (full workload)
2. kernel_classifier.py (rank + classify)
3. kernel_profiler.py Pass B (hot kernels only)
4. cta_cluster.py (for irregular kernels needing clustering)
5. tracer_nvbit (selective trace collection)
6. accel-sim.out with -cta_sampling_mode 1
7. sim_compare.py (accuracy report if ground-truth is available)

---

## File Layout

```
tools/
  kernel_profiler.py       # ncu wrapper, per-kernel + per-CTA feature extraction
  kernel_classifier.py     # roofline classifier, sim_ctas recommendations
  cta_cluster.py           # K-Means clustering, outputs representatives.json
  trace_selector.py        # end-to-end pipeline orchestration
  sim_compare.py           # sampled vs. full comparison, error/speedup report
  configs/
    sm86_rtx3080.json      # peak FLOPS, peak DRAM BW, num_SMs
    sm80_a100.json
    sm90_h100.json

simulator-remodeled/
  util/tracer_nvbit/       # add --cta-allowlist option (NVBit modification)
  gpu-simulator/
    main.cc                # fix failure modes, accept external weights
    gpgpu-sim/src/gpgpu-sim/gpu-sim.cc   # metric extraction (mostly done)
```

---

## Implementation Sequence

| Step | Task | Effort | Unblocks |
|---|---|---|---|
| 1 | Fix 3 simulator failure modes (§3 in STATUS.md) | Medium | Reliable simulation |
| 2 | `kernel_profiler.py` — ncu wrapper, kernel_report.csv | Small | Classifier, clustering |
| 3 | `kernel_classifier.py` — roofline, recommendations | Small | Trace selector |
| 4 | `cta_cluster.py` — K-Means, representatives.json | Medium | Heartwall-fix, irregular kernels |
| 5 | `tracer_nvbit` CTA allowlist filter | Medium | End-to-end trace pipeline |
| 6 | Simulator accepts external weights from representatives.json | Small | Accurate cluster weighting |
| 7 | `trace_selector.py` glue script | Small | Full pipeline |
| 8 | `sim_compare.py` validation report | Small | Accuracy measurement |
| 9 | Validate ≥5 kernel types end-to-end | Medium | Confidence in pipeline |

---

## Validation Plan

Test on ≥5 kernel types:

| Kernel | Type | Tracing need | Expected challenge |
|---|---|---|---|
| hotspot (stencil) | memory-regular | coordinate heuristic | Occupancy (already tested) |
| GEMM (cutlass) | compute-regular | coordinate heuristic | None expected |
| backprop | compute-irregular | clustering | Log-fit extrapolation |
| BFS (rodinia) | memory-irregular | clustering | Irregular CTA behavior |
| srad_v2 | mixed | clustering | Mixed classification |

Acceptance targets:
- Median cycle error < 8%, p90 < 15%
- Speedup > 20× on large kernels (total_CTAs >> num_SMs)

---

## Kernel Selection: Modern Benchmark Suite

### Motivation

The existing Rodinia 2.0 benchmarks (hotspot, BFS, backprop, etc.) were designed for
pre-Volta GPUs. They do not cover:
- Tensor Core (TC) workloads dominant in modern ML inference/training
- Warp-specialized kernels (producer/consumer split — FlashAttention-3, CUTLASS 3.x)
- Persistent kernels with grid-level synchronization
- Kernels using CUDA TMA (Tensor Memory Accelerator) on Hopper/Blackwell
- Fine-grained pipelining via async copy + barriers (cp.async, mbarrier)

A modern benchmark suite should cover these classes to make the simulator relevant for
current hardware (SM90 H100, SM100 B200).

### Selection Methodology

#### Step 1: Identify target kernel classes

| Class | Example workloads | Key hardware features exercised |
|---|---|---|
| Dense GEMM — compute-bound | CUTLASS GEMM, cuBLAS SGEMM | TC (HMMA/IMMA), register file, warp scheduler |
| Dense GEMM — memory-bound | Small-batch GEMM, batched GEMM | L2 bandwidth, DRAM BW |
| Convolution | CUTLASS Conv2d, cuDNN | TC, shared memory tiling |
| Attention (fused) | FlashAttention-2/3, xFormers | TC + DRAM BW, TMA (FA3), warp specialization |
| Normalization | LayerNorm, RMSNorm, GroupNorm | Reduction, shared memory, memory-bound |
| Elementwise / fusion | GELU, bias+add+relu | DRAM BW, memory-bound |
| Sparse matmul | cuSPARSE SpMM, blocked sparse | Irregular memory, L2 miss-heavy |
| Graph workload | SSSP, BFS, PageRank | Irregular access, atomic-heavy |
| Stencil / physics | hotspot, diffusion | Regular memory, cache-friendly |
| Reduction / scan | CUB DeviceReduce, thrust scan | Warp shuffle, multi-wave dependency |

#### Step 2: Source kernels from CuTeDSL and CUTLASS

**CUTLASS 3.x** (https://github.com/NVIDIA/cutlass) is the primary source for modern
TC kernels. Relevant targets:

| Target | CUTLASS component | Why it's interesting |
|---|---|---|
| SGEMM SM90 (Hopper) | `examples/48_hopper_warp_specialized_gemm` | Warp-specialized, TMA, persistent |
| SGEMM SM86 (Ampere) | `examples/14_ampere_tf32_tensorop_gemm` | Baseline TC GEMM |
| Batched GEMM | `examples/40_cutlass_py/gemm/gemm_grouped` | Memory-bound variant |
| Conv2d | `examples/09_turing_tensorop_conv2d` | TC conv, shared mem tiling |
| FlashAttention kernel | `examples/41_fused_multi_head_attention` | Fused attention, memory-bound |

**CuTeDSL** (CUDA/Python DSL built on CuTe layout algebra) targets: custom fused kernels
written by the user as Python → PTX/SASS via CuTeDSL compilation. Selection criteria:
- Cover boundary between compute-bound and memory-bound (near ridge point)
- Include at least one warp-specialized kernel (warp roles differ in behavior)
- Include at least one TMA-using kernel (tests TMA modeling gap)

#### Step 3: Selection criteria for the benchmark set

A kernel is included if it satisfies ≥2 of:

1. **Relevance** — appears in a major ML framework (PyTorch, JAX, TensorRT) as a hot kernel
2. **Architectural coverage** — exercises a hardware feature not covered by existing benchmarks
   (TC units, TMA, warp specialization, persistent grid, async copy)
3. **Simulator stress** — exposes a known simulator weakness (irregular access, multi-wave
   cycle estimation, warp divergence, atomic contention)
4. **CTA heterogeneity** — CTAs have meaningfully different behavior (clustering adds value
   over coordinate heuristic) — measured by silhouette score > 0.3 on per-CTA features

Exclude:
- Micro-benchmarks with synthetic access patterns (not representative of real workloads)
- Kernels with < 2× num_SMs CTAs (too small to benefit from CTA sampling)
- Kernels that already run in < 1ms on target hardware (not worth sampling)

#### Step 4: Coverage matrix

Ensure the selected set spans all four quadrants of the classification space:

```
                 Regular CTA behavior
                 (coord. heuristic OK)
                        │
      Compute   ────────┼────────  Memory
      bound             │           bound
                        │
                 Irregular CTA behavior
                 (clustering required)
```

Target: ≥2 kernels per quadrant.

#### Step 5: Trace collection plan per kernel

| Kernel | Source | Grid typical | Heuristic or cluster? | TMA? |
|---|---|---|---|---|
| SGEMM SM86 (M=N=K=4096) | CUTLASS ex14 | ~1024 CTAs | coordinate heuristic | No |
| SGEMM SM90 warp-specialized | CUTLASS ex48 | ~512 CTAs | clustering (warp roles differ) | Yes |
| Conv2d (ResNet-50 layer) | CUTLASS ex09 | ~1000 CTAs | coordinate heuristic | No |
| FlashAttention-2 | FA repo | ~512 CTAs | clustering | No |
| FlashAttention-3 (Hopper) | FA3 repo | ~256 CTAs | clustering | Yes (UTMALDG) |
| LayerNorm (large batch) | custom/CuTeDSL | ~4096 CTAs | coordinate heuristic | No |
| GELU elementwise | CuTeDSL | ~8192 CTAs | coordinate heuristic | No |
| SpMM (sparse 50%) | cuSPARSE | variable | clustering | No |
| BFS (road network) | Rodinia / Gunrock | irregular | clustering | No |
| DeviceReduce (CUB) | CUB | ~2048 CTAs | coordinate heuristic | No |

### Integration with Toolchain

The kernel selection feeds directly into the pipeline:

1. **Profiler Pass A** on each candidate → classify with [2]
2. Kernels with silhouette score > 0.3 on per-CTA features → [3] CTA clustering
3. All selected kernels → [4] selective tracer → [5] adaptive simulator
4. Coverage matrix validated against the four quadrants above

### Notes on CuTeDSL Kernels

CuTeDSL produces kernels as Python code compiled to PTX/SASS at runtime. To trace them:
- Compile with `CUTLASS_DEBUG_TRACE_LEVEL=1` to emit kernel names
- Use NVBit tracer as-is (NVBit instruments SASS regardless of source)
- No special handling needed — the pipeline treats them as any other kernel

TMA-using kernels (FA3, Hopper GEMM) are a special case — see Open Questions §4.

---

## Open Questions

1. **Real GPU availability** — CTA clustering (Approach 2) requires NVBit counter
   profiling on real hardware. Coordinate heuristic works without a GPU (traces already
   collected). Clustering needed for irregular kernels.

2. **Multi-wave cycle model** — the current log-fit handles some cases but is not
   validated on kernels with total_CTAs >> 10× num_SMs. Needs investigation.

3. **Threshold calibration** — `T_low`, `T_high`, AI weights `W_tc`, `W_sfu` are
   initial guesses. Should be calibrated per architecture config.

4. **TMA kernels** — FlashAttention-3 and future kernels using CUDA TMA bypass standard
   memory pipelines. Pressure signals and clustering features don't capture TMA behavior.
   This is a known gap; separate modeling needed.
