# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **modern-gpu-simulator-micro-2025**, an enhanced GPU microarchitecture simulator forked from Accel-Sim, published at MICRO 2025. It adds a redesigned SM model, control-bit-based dependency tracking, an OpenMP-parallelized simulation, and Protocol Buffer-based trace format. All active code lives under `simulator-remodeled/`.

**Requirements:** Ubuntu 20.04–24.04, g++/gcc ≤ 11 (RapidJSON compatibility), CUDA 11.4 or 12.8, Google Protocol Buffers.

## Build Commands

All commands run from `simulator-remodeled/`:

```bash
# Set up build environment (sets ACCELSIM_CONFIG=release)
source ./gpu-simulator/setup_environment_no_git.sh
# For debug: source ./gpu-simulator/setup_environment_no_git.sh debug

# Build the simulator
make -j -C ./gpu-simulator/
# Output: ./gpu-simulator/bin/release/accel-sim.out

# Install Python dependencies
pip install -r requirements.txt
```

**Build the NVBit tracer** (requires a real GPU):
```bash
export CUDA_INSTALL_PATH=<path>
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/
```

## Running Simulations

**Single workload:**
```bash
./gpu-simulator/bin/release/accel-sim.out \
  -trace ./hw_run/<arch>/<app>/traces/dynamic_trace.pb \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/<SM_CFG>/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/<SM_CFG>/trace.config
```

**Batch simulations** (supports SLURM):
```bash
./util/job_launching/run_simulations.py \
  -B rodinia_2.0-ft \
  -C RTX3080-Accelwattch_SASS_SIM \
  -T ./hw_run/traces/device-<N>/<cuda-version>/ \
  -N myTestName

./util/job_launching/get_stats.py -N myTestName | tee stats.csv
```

Example traces are in `exampleTraces/` (`.tar.gz` archives for Turing, Ampere, Blackwell).

## Architecture

### End-to-End Flow

```
Real GPU
  → [NVBit Tracer] → dynamic_trace.pb + enhanced_execution_info.json
  → [accel-sim.out] → stats CSV + AccelWattch power report
  → [Correlator/Plotter] → APE vs. hardware counters
```

### Key Source Locations (all under `simulator-remodeled/`)

| Path | Role |
|---|---|
| `gpu-simulator/main.cc` | Entry point |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/` | GPGPU-Sim 4.0 base: `shader.cc`, `gpu-sim.cc`, `gpu-cache.cc`, `dram.cc` |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/` | **MICRO 2025 SM model** (see below) |
| `gpu-simulator/trace-driven/` | Trace-driven front-end; feeds warps to performance model |
| `gpu-simulator/trace-parser/` | Reads `.pb` trace files |
| `gpu-simulator/ISA_Def/` | SASS opcode definitions per architecture (Kepler–Blackwell) |
| `gpu-simulator/configs/tested-cfgs/` | Per-GPU config pairs (`gpgpusim.config` + `trace.config`) |
| `util/traces_enhanced/` | Enhanced trace library; parses static JSON metadata and PB dynamic traces |
| `util/traces_enhanced/dynamic_trace/*.proto` | Protocol Buffer schema for dynamic traces |
| `util/tracer_nvbit/tracer_tool/` | NVBit SASS-level instrumentation (`tracer_tool.cu`, `inject_funcs.cu`) |
| `util/job_launching/` | Batch simulation scripts |
| `util/tuner/` | Auto-tuner: microbenchmarks + `tuner.py` to generate GPU configs |
| `APEs/` | Per-GPU Absolute Percentage Error reports vs. hardware |

### The MICRO 2025 SM Model (`remodeling/`)

This is the primary contribution of the paper. The new SM model replaces the GPGPU-Sim shader with:

- **`sm.cc/h`** — Top-level SM: orchestrates sub-cores, fetch/decode, memory unit
- **`subcore.cc/h`** — Sub-core pipeline: issue, execute, writeback; dispatches to per-type latches (SP/DP/HP/INT/SFU/TensorCore/Uniform/Branch/Misc)
- **`ldst_unit_sm.cc/h`** — Load/store unit
- **`warp_dependency_state.cc/h`** — Control-bit-based dependency tracking (the actual Turing/Ampere hardware mechanism)
- **`scoreboard.cc/h` + `scoreboard_reads.cc/h`** — Enhanced scoreboards covering uniform, predicate, and uniform-predicate registers; separate WAR scoreboard
- **`first_level_instruction_cache.cc/h`** — L0 instruction cache
- **`stream_buffer.cc/h`** — Stream-buffer instruction prefetcher
- **`functional_unit.cc/h`** — SP, DP, HP, INT, SFU, Tensor Core models
- **`register_file.cc/h`** — Register file model
- **`new_stats.h`** — Polymorphic stats infrastructure for OpenMP-parallelized stat gathering
- **`fusedMemory/coalescingStats.cc/h`** — Memory coalescing statistics

### GPU Configurations

42 configs spanning SM75 (Turing), SM86 (Ampere), SM90 (Hopper H100), SM100 (Blackwell B200), SM120 (Blackwell RTX 5070 Ti). SM86 has many experimental variants tuning prefetch depth, scoreboard consumers, and R/W port counts.

### Trace Format

Two-part: (1) **Static metadata** in `enhanced_execution_info.json` — control bits, operand types, register usage per instruction. (2) **Dynamic traces** in `dynamic_trace.pb` (Protocol Buffers); per-CTA traces in `traces/threadblocks/{x,y,z}.pb`. The `traces_enhanced` library ties these together and is linked into `accel-sim.out`.

## Known Issues

Three benchmarks cannot build with `sm_86` (tracked in `KNOWN_BUILD_FAILURES.md`): `ispass-MUM` (deprecated 2D texture API), `ispass-DG` (MPI/ParMetis Makefile issue), `ispass-WP` (Fortran + deprecated nvcc options). Overall build success: 102/105 (97.1%).

## Active Work

**Branch `cta-sampling`:** CTA-level sampling to speed up simulation. Foundation (sampling + classifier + pilot loop), Phase A (refined classifier inputs), Phase B (whole-kernel cycle estimator), and three wider-validation gap fixes (C1 tiny-grid skip, C2 K-rep replication fix, C3a adaptive doublings) are implemented and validated. Cycle accuracy meets p50<15% / p90<25% on the wider 9-workload set; wall-time speedup is workload-scale dependent. Start with `HANDOFF.md` for the current self-contained state and next steps; `CTA_SAMPLING_Debate_Log_Fit.md` is the active design debate on replacing the log-fit concurrency model; `history/` holds the two walkthrough-derived progression records.
