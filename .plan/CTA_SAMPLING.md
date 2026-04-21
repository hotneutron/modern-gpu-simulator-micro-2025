# CTA-Level Sampling for Single-Kernel Simulation Speedup

## Problem

GCL-Sampler clusters kernel *invocations* across a multi-kernel workload. When the workload under test is a **single kernel**, there is nothing to cluster at the kernel level. Yet individual kernels can be extremely expensive:

- **Trace generation** (NVBit): instrumenting every CTA × warp × instruction can take hours for large kernels
- **Simulation** (GCoM / Accel-Sim): simulation time scales roughly linearly with CTA count

The solution is to apply the same sampling idea one level down: **sample thread blocks (CTAs) within the kernel**.

---

## Approach 1: Coordinate-Based Heuristic (Fast Baseline)

For **regular compute kernels** (GEMM, convolution, stencil), CTAs doing identical work are trivially identifiable by their grid position. The dominant source of CTA heterogeneity is **boundary effects**: CTAs at the edges of the grid handle partial tiles or boundary conditions; interior CTAs are nearly identical.

**Sampling rule:**
- Select a small set of CTAs that covers the distinct structural positions: corners, edges (X-face, Y-face, Z-face), and one interior representative.
- For a 2D grid this gives 4–9 representative CTAs regardless of grid size.
- Simulate those, weight results by how many CTAs fall into each position category.

**When it works well:** GEMM, convolution, pooling, elementwise kernels, stencil codes.  
**When it fails:** Sparse kernels, graph algorithms, reductions with irregular access — CTA behavior is determined by data, not grid position.

**Cost:** Zero preprocessing. Can be implemented in ~100 lines by modifying which CTA trace files the simulator loads.

---

## Approach 2: Two-Pass Lightweight Profiling + Full Trace of Representatives

### Overview

Instead of full SASS traces for every CTA, run two passes:

1. **Pass 1 (fast profiling):** Collect a cheap per-CTA feature vector using hardware performance counters — no instruction-level tracing needed.
2. **Cluster** CTAs by feature similarity (K-Means or hierarchical).
3. **Pass 2 (selective tracing):** Run NVBit full SASS trace only for the representative CTA(s) per cluster.
4. **Simulate** only the representative CTAs; scale stats by cluster weight.

### Pass 1: Lightweight Feature Collection

Use **NVIDIA Nsight Compute** (or NVBit counter instrumentation) to collect per-CTA hardware counters in a single kernel run. No instruction-level tracing; this is 10–50× faster than full SASS trace.

Useful per-CTA features:

| Feature | Captures |
|---|---|
| Active warps (avg) | Occupancy variation |
| L1 hit rate | Memory access locality |
| L2 hit rate | Off-chip traffic variation |
| Issued instructions | Work imbalance |
| Memory transactions | Load/store divergence |
| Branch divergence count | Control flow variation |
| Shared memory accesses | Collaborative patterns |

Each CTA becomes a point in ~7-dimensional feature space.

### Clustering

Apply K-Means on the feature vectors. Use silhouette coefficient to select K (same approach as GCL-Sampler at the kernel level). For most regular kernels K=2–5 is sufficient; irregular kernels may need K=10–20.

Select one representative CTA per cluster (e.g., the medoid — closest to centroid).

### Pass 2: Selective SASS Tracing

Modify the NVBit tracer to only instrument selected CTAs (filter by `blockIdx`). NVBit already supports this — the Accel-Sim tracer checks CTA coordinates when deciding whether to record.

Only the K representative CTAs are traced. For a kernel with 1,000 CTAs and K=5, trace generation is ~200× faster.

### Simulation

The simulator already stores per-CTA trace files as:
```
traces/threadblocks/device_0/stream_0/kernel_1/{x},{y},{z}.pb
```

Load only the representative CTA trace files. After simulation, multiply each representative's stats by its cluster weight (cluster size / total CTAs).

### Stat Aggregation with Weights

For metric M (IPC, L1 hit rate, etc.):
```
M_estimated = Σ_c  (weight_c × M_representative_c)
weight_c = |CTAs in cluster c| / total CTAs
```

---

## Expected Speedup

| Kernel type | CTA regularity | K (clusters needed) | Speedup |
|---|---|---|---|
| GEMM, convolution | Very high | 2–5 | 200–500× |
| Stencil, pooling | High | 3–8 | 100–300× |
| Reduction, scan | Medium | 5–15 | 50–100× |
| Graph BFS/SSSP | Low | 10–30 | 30–100× |
| Sparse matrix (SpMM) | Low-medium | 5–20 | 50–200× |

Speedup = (total CTAs) / K for both trace generation and simulation.

---

## Existing Open Source Implementations

### 1. Principal Kernel Analysis (PKA) — Accel-Sim
- **Repo:** https://github.com/accel-sim/accel-sim-framework
- **Paper:** "Principal Kernel Analysis: A Tractable Methodology to Simulate Scaled GPU Workloads" (MICRO 2021)
- **What it does:** Two-level sampling — inter-kernel selection AND intra-kernel CTA projection. The intra-kernel component is exactly the two-pass idea: a fast profiling pass identifies representative CTAs, then only those are simulated.
- **Accuracy:** ~27% average cycle error vs. silicon across 147 workloads (error is higher than GCL-Sampler because PKA uses hand-crafted features, not learned embeddings)
- **Relevance:** Most directly applicable. The CTA-sampling infrastructure is already in Accel-Sim; could be ported to `simulator-remodeled` with moderate effort.

### 2. Photon — Fine-Grained Sampled GPU Simulation
- **Paper:** "Photon: A Fine-grained Sampled Simulation Methodology for GPU Workloads" (MICRO 2023)
- **What it does:** Multi-level intra-kernel sampling at warp granularity and basic-block granularity. No upfront analysis; uses online sampling. Reduced ResNet-152 simulation from 7.05 days to 1.7 hours (99× speedup) at 10.7% error.
- **Relevance:** More aggressive than PKA — samples within a CTA at the warp level, enabling further speedup on very large CTAs.
- **Status:** Paper is published but code availability is unclear; check with authors.

### 3. GPGPU-Sim Checkpoint Mechanism
- **Repo:** https://github.com/gpgpu-sim/gpgpu-sim_distribution
- **What it does:** Functional simulation to a checkpoint (specified by CTA count + instruction count), then resume in performance mode. Not ML-based — it's a manual skip, not clustering-based sampling.
- **Relevance:** Useful as a fast-forward mechanism before the representative CTAs; could be combined with clustering.

### 4. NVBit with Per-CTA Filtering
- **Repo:** https://github.com/NVlabs/NVBit
- **What it does:** NVBit tool examples include memory tracing with CTA-coordinate-aware filtering. The existing `tracer_nvbit` in `simulator-remodeled/util/` is already NVBit-based.
- **Relevance:** Pass 2 (selective tracing) can be implemented by modifying `util/tracer_nvbit` to accept a list of `(blockIdx.x, blockIdx.y, blockIdx.z)` coordinates to trace.

---

## Approach 1 Revised: Roofline-Guided Adaptive Occupancy Filling

### Problem with Naive K-Rep Simulation

Simulating only K representative CTAs (K≈9 for an 8×8 grid) on a GPU with 68 SMs leaves
59 SMs idle. This causes two classes of error:

1. **Occupancy collapse**: per-SM warp count drops, latency hiding degrades, IPC is
   underestimated even for compute-bound kernels (warp scheduler, L0I pressure, and
   shared-memory bank conflicts all depend on CTA concurrency, not just instruction mix).
2. **Inter-CTA contention absence**: with K/num_SMs of real traffic, the following are
   absent or underrepresented:
   - L2 slice/set contention (reps underfill sets, overpredict hit rate)
   - DRAM partition camping and bank conflicts
   - MSHR and miss-queue pressure (nonlinear with active CTAs)
   - NoC/crossbar arbitration contention
   - Atomic hotspot amplification (reductions, hash-table inserts)
   - Instruction-cache and warp-scheduler pressure with many active warps

### Inputs and Derived Metrics

From the K-CTA simulation run:

```
# Hardware prior (from GPU config)
ridge_point = peak_FLOPS / peak_DRAM_BW          (FLOP/byte)

# Kernel arithmetic intensity (broadened to cover tensor/SFU ops)
kernel_AI   = (ALU_insns + tensor_insns + SFU_insns) / bytes_to_DRAM
              # prefer bytes_to_DRAM over bytes_requested if available

ridge_ratio = kernel_AI / ridge_point

# Memory pressure signals from K-CTA run
mem_stall_fraction  = cycles_stalled_on_memory / total_cycles
achieved_BW_ratio   = observed_DRAM_BW / peak_DRAM_BW
dram_queue_occupancy = average DRAM queue depth (normalized)
L2_miss_rate        = L2_misses / L2_accesses
```

**Circularity caveat:** all pressure signals from the K-CTA run are biased low because
contention is absent. They provide a *lower bound* on memory pressure, not an exact
measure. Borderline cases should be resolved conservatively toward memory-bound.

### Classification: Three-Way

Use a band around the ridge point defined by two thresholds (`T_low`, `T_high`),
plus direct pressure signals as co-classifiers:

| Class | Condition | Default sim_ctas |
|---|---|---|
| **Compute-bound** | `ridge_ratio >= T_high` AND `mem_stall_fraction < 0.2` AND `dram_queue_occupancy` low | K |
| **Memory-bound** | `ridge_ratio <= T_low` OR `mem_stall_fraction > 0.5` OR `achieved_BW_ratio >= 0.6` | N_sat_est |
| **Mixed/uncertain** | neither above | 1.5 × N_sat_est |

Initial threshold values (calibrate per architecture):
- `T_low = 0.9`, `T_high = 1.3`
- memory pressure "high" if any of: `achieved_BW_ratio >= 0.6`, `dram_queue_occupancy`
  above config threshold, `mem_stall_fraction` above config threshold

Mixed/uncertain cases use `1.5 × N_sat_est` as a conservative starting point; this
multiplier is a tunable config knob, not a fixed constant.

### Simulation Strategy

**Step 1: K-rep baseline run** (always)
- Coordinate heuristic selects K≈9 CTAs (corners, edge midpoints, interior)
- Simulate on K SMs; collect `kernel_AI`, `mem_stall_fraction`, `achieved_BW_ratio`,
  `dram_queue_occupancy`, `L2_miss_rate`
- Classify as compute-bound, memory-bound, or mixed

**Step 2: Choose initial sim_ctas**

*Compute-bound* (`sms_floor_compute` = K — no replication needed):
```
sim_ctas = K
weight   = total_ctas / K
```
Note: even compute-bound kernels can exhibit CTA-concurrency effects via shared-memory
bank conflicts or L0I pressure. Validate per-SM independence before accepting.

*Memory-bound:*
```
per_SM_mem_BW    = (mem_insns / total_insns) × bytes_per_mem_insn × SM_issue_rate
N_sat_est        = ceil(peak_DRAM_BW / per_SM_mem_BW)   # initial estimate only
sim_ctas         = min(total_ctas, max(K, N_sat_est))
```

*Mixed/uncertain:*
```
sim_ctas = min(total_ctas, max(K, ceil(1.5 × N_sat_est)))
```

`N_sat_est` is an initial guess. `per_SM_mem_BW` is noisy and architecture-sensitive;
peak DRAM BW as denominator tends to overestimate required SMs for latency-limited
kernels and underestimate for partition-camped traffic. The adaptive loop below corrects
for this.

**Step 3: Adaptive pilot scaling (contention guardrail)**

Run at `sim_ctas`, check absolute pressure targets, double and re-run only if needed:

```
sim_ctas = sim_ctas_initial
while (memory_bound or mixed) and sim_ctas < total_ctas:
    run simulation at sim_ctas
    if all stop conditions met:
        break
    sim_ctas = min(total_ctas, sim_ctas * 2)

weight = total_ctas / sim_ctas
```

Stop conditions (all must hold):
- `achieved_BW_ratio >= 0.8` (primary — absolute DRAM saturation target)
- change in `achieved_BW_ratio` vs. previous run < 5% (secondary — stabilization)
- change in IPC vs. previous run < 5%
- change in `dram_queue_occupancy` vs. previous run < 10%

In practice, 1–2 doublings are sufficient for most kernels.

**Step 4: CTA assignment to SMs (stratified shuffle)**

Do **not** use pure round-robin replication — it creates artificial periodicity in
address streams (L2 set mapping, DRAM bank mapping). Instead:
- Each of the K reps is assigned to approximately `sim_ctas / K` slots
- Slot-to-rep mapping is randomized (stratified shuffle) to break address alignment
  with memory striping patterns

### Stat Aggregation Rules

Additive counts scale linearly with CTA count — apply weight directly:

| Stat | Rule | Rationale |
|---|---|---|
| `gpu_tot_sim_cycle` | **No scaling** | Wall-clock time of the simulated wave; valid only when Step 3 ensures representative contention |
| `gpu_tot_sim_insn` | × weight | Linear with CTA count |
| `gpu_tot_issued_cta` | × weight | Linear with CTA count |
| Memory transactions/bytes | × weight | Linear with CTA count |

Ratio metrics must **not** be averaged directly. Recompute from weighted
numerators and denominators:

```
IPC       = (gpu_sim_insn × weight) / gpu_sim_cycle   # weighted insns / raw cycles
hit_rate  = weighted_hits / weighted_accesses
```

`gpu_tot_ipc` is only meaningful when `sim_ctas` is large enough that contention is
representative. Flag it as unreliable if the adaptive loop hit `total_ctas` without
achieving `achieved_BW_ratio >= 0.8`.

Whole-kernel *time* estimation (multi-wave model) is explicitly deferred — it requires
a wave-count model that is not yet defined.

### Contention Guardrails

Before accepting a sampled result, verify these effects are adequately represented:

- L2 slice/set contention (check L2 miss rate vs. expected from full occupancy)
- DRAM partition camping (check per-partition traffic balance)
- MSHR/miss-queue pressure (check queue saturation fraction)
- NoC/crossbar pressure proxies
- Atomic hotspot amplification (flag kernels with global atomics)
- Warp scheduler pressure (`eligible warps per cycle`)

If any guardrail fails, escalate `sim_ctas` or fall back to full simulation.

### Speedup

| Kernel type | total_ctas | num_SMs | sim_ctas (typical) | Speedup |
|---|---|---|---|---|
| Compute-bound, large | 10,000 | 108 | K≈9 | ~1,100× |
| Memory-bound, large | 10,000 | 108 | ~20–40 (adaptive) | ~250–500× |
| Mixed, large | 10,000 | 108 | ~40–80 (conservative) | ~125–250× |
| Small grid (any) | 64 | 68 | K≈9–64 | 1–7× |

### Known Limitations

- Coordinate heuristic only captures structural boundary effects; fails for
  data-irregular kernels (sparse, graph, reduction). Use Approach 2 for those.
- All K-CTA pressure signals are biased low — classification errs toward memory-bound
  by design.
- Adaptive pilot adds up to 2× overhead vs. a single fixed sim_ctas run; still
  negligible for large kernels vs. full simulation.
- Whole-kernel cycle estimation (multi-wave) is not yet modeled.

## Recommended Implementation Path

1. **Done:** Coordinate-based heuristic — K representative CTAs selected, simulator
   loads only those. (`main.cc`, `trace_driven.cc`, `trace_parser.h`)

2. **Done (partial):** Stat scaling — `gpu_tot_sim_cycle` unscaled, `gpu_tot_sim_insn`
   and `gpu_tot_issued_cta` scaled by weight. (`gpu-sim.cc`, `gpu-sim.h`)

3. **Next — metric extraction:** Add helpers to collect `kernel_AI`, `mem_stall_fraction`,
   `achieved_BW_ratio`, `dram_queue_occupancy`, `L2_miss_rate` from the simulator after
   each kernel run.

4. **Next — classifier:** Implement 3-way classifier with configurable `T_low`, `T_high`,
   and pressure thresholds in `trace_config`. Compute `N_sat_est` from GPU config.

5. **Next — replication:** Expand `sampled_ctas` list using stratified shuffle to
   `sim_ctas` slots. Update `gridDim` accordingly in `create_kernel_info`.

6. **Next — adaptive loop:** Wire pilot scaling loop in `main.cc`; re-invoke
   `create_kernel_info` + simulation with doubled `sim_ctas` until stop conditions met.

7. **Validate:** Test on ≥5 kernels covering compute-regular (GEMM), memory-regular
   (stencil/hotspot), reduction/scan, and if traces available, sparse/graph irregular.
   Acceptance targets:
   - Median cycle error < 8%
   - p90 cycle error < 15%
   - Speedup > 20× on large kernels (total_ctas >> num_SMs)

8. **Deliverables:** Config knobs for all thresholds and tolerances; benchmark script
   for sampled vs. full comparison; per-kernel accuracy/speedup report.

9. **Later (Approach 2):** Two-pass clustering — Nsight Compute Pass 1 for per-CTA
   feature vectors, K-Means clustering, selective NVBit re-trace of representatives.
   Handles irregular kernels where coordinate heuristic fails.
