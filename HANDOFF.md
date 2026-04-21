# HANDOFF.md — CTA Sampling Work

## Project Context

This is **modern-gpu-simulator-micro-2025**, a GPU microarchitecture simulator forked from Accel-Sim (MICRO 2025). Branch `cta-sampling` adds CTA-level sampling to speed up single-kernel simulation.

## Current State

**Branch:** `cta-sampling` (unmerged)
**Last commit:** `5a5f068 cta sampling`

### Modified Files

- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc`
  - Changed `gpu_tot_sim_cycle` to NOT scale by weight (leaves cycles unscaled as wall-clock time of sampled wave)
  - Fixed `gpu_tot_sim_insn` and `gpu_tot_issued_cta` to scale by weight
  - This is a behavioral fix, notfeature addition

### Design Documents (in repo root)

| File | Status | Description |
|---|---|---|
| `CTA_SAMPLING.md` | Complete | Full design with 2 approaches, roofline classification, adaptive pilot loop |
| `CTA_SAMPLING_REVISED_PLAN.md` | Draft | Simplified 3-way classifier + adaptiveloop |
| `CTA_SAMPLING_SINGLE_PASS_PLAN.md` | Draft | Single-pass variant (no pilot), conservative SM sizing |

All three describe the same overall goal. `CTA_SAMPLING_SINGLE_PASS_PLAN.md` is the simplest to implement if you want a quick baseline.

## What Was In Progress

1. **Stat scaling fix**: Done in `gpu-sim.cc` (see diff above)
2. **Next**: Implement metric extraction helpers (AI, memory pressure)
3. **Next**: Implement 3-way classifier with thresholds
4. **Next**: Add adaptive SM filling loop
5. **Validation**: Test on GEMM, stencil, reduction, sparse/graph kernels

## Build/Test Commands

```bash
# Build simulator
cd simulator-remodeled/
source ./gpu-simulator/setup_environment_no_git.sh
make -j -C ./gpu-simulator/

# Run test workload
./gpu-simulator/bin/release/accel-sim.out \
  -trace <path-to-trace.pb> \
  -config <path-to-gpgpusim.config> \
  -config <path-to-trace.config>
```

## Key Source Locations

| Path | Role |
|---|---|
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc` | Main GPU sim, stat aggregation |
| `simulator-remodeled/gpu-simulator/trace-driven/` | Trace-driven front-end |
| `simulator-remodeled/gpu-simulator/trace-parser/` | PB trace reading |
| `simulator-remodeled/gpu-simulator/configs/tested-cfgs/` | GPU configs (SM75–SM120) |

## Open Questions

1. Which plan to implement: single-pass (simple) vs revised with adaptive loop?
2. Classification thresholds need calibration per architecture
3. Whole-kernel cycle estimation (multi-wave model) is deferred — not in scope

## Take-Over Instructions

1. Review `CTA_SAMPLING_SINGLE_PASS_PLAN.md` for simplest path forward
2. Build and verify existing changes work: run a trace with CTA sampling enabled
3. Add metric extraction in `gpu-sim.cc` or new helper class
4. Wire classifier + SM filling into `main.cc` or `gpu-sim.cc`
5. Validate against full simulation on ≥3 kernels

## Contact

Original author: see `git log 5a5f068` for context