# Progress overview — `cta-sampling` branch

## What this repo is

`modern-gpu-simulator-micro-2025` is an enhanced **GPU microarchitecture simulator** forked from Accel-Sim and published at MICRO 2025. It takes a recorded SASS instruction trace from a real NVIDIA GPU (collected via NVBit) and re-executes it on a cycle-accurate model of the GPU's pipeline, caches, NoC, and DRAM, producing per-kernel cycle, IPC, memory-traffic, and energy stats.

The MICRO 2025 contribution is a redesigned SM ("streaming multiprocessor") model living in `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/`. All active work happens under `simulator-remodeled/`. The end-to-end flow is:

```
Real GPU
  → [NVBit Tracer]   → dynamic_trace.pb + enhanced_execution_info.json
  → [accel-sim.out]  → cycles, IPC, DRAM bytes, energy
```

You don't need a real GPU to *use* the simulator — there are example traces in `simulator-remodeled/exampleTraces/` (Turing, Ampere, Blackwell). You only need a real GPU to *generate* new traces.

(Note on terminology: a *kernel* is one CUDA `<<<grid, block>>>` call; the *grid* is split into *CTAs* — cooperative thread arrays, also called thread blocks — each running on a single SM. A typical kernel might launch 1,000–100,000 CTAs across ~80 SMs.)

## The problem this branch solves

Simulating a single large GPU kernel is **slow** — simulator runtime scales roughly linearly with the CTA count, so kernels with 10k+ CTAs take many minutes to many hours. Existing kernel-level sampling tools (e.g. GCoM / GCL-Sampler) cluster *kernels* in multi-kernel workloads; if the workload is just one kernel, they don't help.

The fix is to sample one level down: **simulate only a few representative CTAs and scale the stats up.** For regular compute kernels (GEMM, stencils, conv), CTAs only differ at grid boundaries — so a coordinate heuristic (corners, edge-midpoints, one interior CTA) covers all the structural variation in ~9 reps regardless of grid size.

The complication is that running only K=9 CTAs underfills the GPU: 9 SMs busy, 70+ idle. That destroys inter-CTA contention (L2 set conflicts, DRAM bank camping, MSHR pressure) and biases stats. So the design also calls for **expanding** K reps to fill more SMs (replicating each rep), and ideally **adapting** how many SMs we fill to match what the full kernel would experience.

## What was already done before this branch's recent work

- `5a5f068` "cta sampling": added the K-rep coordinate heuristic, the `-cta_sampling_mode 0/1` knob, and a `cta_sampling_weight` field on `kernel_trace_t`. Sampling enabled, but stats weren't scaled correctly.
- `9d04a35` "Fix CTA sampling stat scaling": made `gpu_tot_sim_insn` and `gpu_tot_issued_cta` scale by weight, while leaving `gpu_tot_sim_cycle` unscaled (cycles represent wall-clock of the *sampled wave*, not the full kernel).

Three design documents in `.plan/` propose how to extend this: `CTA_SAMPLING.md` (full design with two approaches), `CTA_SAMPLING_REVISED_PLAN.md` (3-way classifier + adaptive pilot loop), `CTA_SAMPLING_SINGLE_PASS_PLAN.md` (simpler single-shot variant).

## What was just built (revised plan, 7 commits)

We implemented the **revised plan with adaptive pilot loop**. Each commit is independently buildable and testable; they layer on top of each other:

| Commit | What it adds |
|---|---|
| `f2447e1` Fix build on modern toolchains | Adds missing `#include <functional>` to `ldst_unit_sm.h` and drops dead `-lGL` from the link line. Required to build at all on g++-13 / glvnd-less systems. |
| `3bf0d58` Per-kernel pressure signal extraction | New `gpgpu_sim::compute_kernel_pressure_signals()` returns per-kernel deltas (DRAM bytes, L2 hits/misses, queue occupancy, achieved BW ratio) plus roofline derivatives (`kernel_ai`, `ridge_ratio`). Snapshot taken in `launch()` so deltas are correct despite cumulative DRAM/L2 counters. Logged unconditionally as `CTA_PRESSURE_SIGNALS:...`. |
| `726608a` 3-way roofline classifier | `classify_kernel()` returns COMPUTE / MEMORY / MIXED. New knobs `-cta_sampling_t_low` (0.9), `-cta_sampling_t_high` (1.3), `-cta_sampling_pressure_bw` (0.6), `-cta_sampling_pressure_queue` (0.5). High BW or queue depth overrides ridge_ratio (treats as memory-bound regardless of AI). |
| `22a9fc7` `N_sat_est` + initial sim_ctas selector | `compute_n_sat_est()` (peak DRAM BW / per-SM BW from K-rep) and `compute_initial_sim_ctas()` (COMPUTE→sms/2, MEMORY→N_sat_est, MIXED→1.5×N_sat_est). |
| `7ce6074` Stratified-shuffle replication | `expand_sampled_ctas(reps, target, seed)` deterministically replicates K reps to N slots with Fisher-Yates shuffle to break L2/DRAM striping periodicity. Knobs `-cta_sampling_target_ctas` (0=disabled), `-cta_sampling_seed` (1). Weight becomes `total/N`. |
| `8efcddb` Adaptive pilot loop with rollback | After K-rep run, classifier picks a target; if not COMPUTE, rebuild the kernel with that target and re-run; double until DRAM-saturated or stable. Rejected iters are rolled back via `pilot_snapshot/restore` of `gpu_tot_*` so they don't pollute totals. Knobs: `-cta_sampling_pilot_max_doublings` (0=off), `-cta_sampling_pilot_stop_bw_target` (0.8), `-cta_sampling_pilot_stop_delta` (0.05). |
| `c63f17c` Validation harness | `util/cta_sampling/validate.py` runs accel-sim.out across 4 modes on a configurable workload set and prints a comparison table. |

Validation across hotspot / backprop / pathfinder (rodinia2 / Turing):

```
workload    mode      insn_err%   cycle_err%   wall vs baseline
hotspot     K-rep        -5.1        -28        ~same (small grid)
hotspot     pilot        -5.1        -28        ~same
backprop    K-rep        +0.3        -56        7.5× faster
backprop    pilot        +0.3        -56        7.5× faster
pathfinder  K-rep        -5.2        -0.3       ~same (only 20 CTAs)
```

Insn error is **0.3–5%** across kernels — weight scaling preserves total work well. Cycle error is large by design: cycles report the sampled wave's wall-clock, not a whole-kernel estimate (multi-wave model deferred). Sampling pays off when the original grid has many more CTAs than SMs (backprop = 512 CTAs → 7.5× speedup); for tiny grids it's a no-op.

## How to build and run

The build needs a careful environment (Ubuntu 24.04 / g++-13 won't work directly). One-time setup:

```bash
# Create a dedicated conda env with g++-11 and pre-Abseil protobuf
conda create -n accelsim-build -c conda-forge -y \
    "gcc_linux-64=11.4" "gxx_linux-64=11.4" "libprotobuf=3.21.12" zlib
```

Build:

```bash
ENV=/home/afa55/.conda/envs/accelsim-build
export IS_SERT=0 CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=$ENV/bin:$PATH
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc
export CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export CPATH=$ENV/include:${CPATH:-}
export LIBRARY_PATH=$ENV/lib:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}

cd simulator-remodeled
source ./gpu-simulator/setup_environment_no_git.sh
make -j$(nproc) -C ./gpu-simulator/
```

Run a single trace:

```bash
TRACE=/tmp/traces_extracted/rodinia2/12.8/hotspot-rodinia-2.0-ft/.../traces/dynamic_trace.pb
GCFG=./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM75_RTX2070_S/gpgpusim.config
TCFG=./gpu-simulator/configs/tested-cfgs/SM75_RTX2070_S/trace.config

./gpu-simulator/bin/release/accel-sim.out -trace $TRACE -config $GCFG -config $TCFG \
    -cta_sampling_mode 1 -cta_sampling_pilot_max_doublings 2
```

Run validation across multiple kernels:

```bash
./util/cta_sampling/validate.py
```

Example traces are in `simulator-remodeled/exampleTraces/rodinia2{Turing,Ampere,Blackwell}.tar.gz`. Extract one and point `--trace-root` at it.

## Code map (new code only)

| File | What's there |
|---|---|
| `gpu-simulator/main.cc` | `compute_sampled_ctas` (K-rep heuristic), `expand_sampled_ctas` (stratified shuffle), `classify_kernel`, `compute_n_sat_est`, `compute_initial_sim_ctas`, `pilot_decide_accept`, `pilot_next_target`, the kernel cleanup block with the pilot loop |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h/.cc` | `pressure_signals_t`, `pressure_snapshot_t`, `pilot_stats_snapshot_t` structs; `compute_kernel_pressure_signals`, `snapshot_pressure_signals_kernel_start`, `pilot_snapshot`, `pilot_restore` methods |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/dram.h` | three small accessors `get_n_req`, `get_bwutil`, `get_ave_mrqs` |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h` | `get_dram_ro()` accessor on `memory_partition_unit` |
| `gpu-simulator/trace-driven/trace_driven.h/.cc` | seven new `-cta_sampling_*` knobs registered in `reg_options` |
| `util/cta_sampling/validate.py` | sampled-vs-baseline error table generator |

## Known limitations / next steps

- **Arithmetic-intensity proxy is crude.** `kernel_ai = gpu_sim_insn / dram_bytes` over-counts (memory and control-flow ops are included as if they were FLOPs), so memory-heavy kernels are mis-classified as COMPUTE and the pilot loop accepts on iter 0 without expanding. A finer breakdown (separate ALU / load-store / tensor counters) would fix this.
- **Whole-kernel cycle estimation is deferred.** `gpu_tot_sim_cycle` is the wall-clock of the *sampled wave*, not the full kernel. A multi-wave model needs to be added before cycle/IPC numbers can be compared 1:1 to a non-sampled baseline.
- **Pilot loop requires `window_size==1`** (no concurrent kernels). It auto-disables and warns otherwise.
- **No memory-stall fraction yet.** The classifier uses DRAM achieved-BW and queue occupancy as memory pressure proxies; per-SM stall counters exist (`shader.h:2467+`) but aren't plumbed in.

See `.plan/HANDOFF.md` for the original task list and `.plan/CTA_SAMPLING_REVISED_PLAN.md` for the design these commits implement.
