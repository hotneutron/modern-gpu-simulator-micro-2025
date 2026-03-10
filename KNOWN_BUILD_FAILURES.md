# Known Build Failures

This document lists benchmarks that cannot be built with `CUDA_ARCH=sm_86` due to deep API or build system issues that require significant refactoring.

## Summary

- **Total Benchmarks**: 105
- **Successfully Building**: 102 (97.1%)
- **Known Failures**: 3

## Failed Benchmarks

| Benchmark | Suite | Issue | Root Cause | Fix Complexity |
|-----------|-------|-------|------------|----------------|
| ispass-MUM | ISPASS-2009 | 2D Texture API | Uses `texture<ulong4, 2>` and `tex2D()` throughout kernel logic for suffix tree traversal. No conditional fallback exists. | High - requires complete kernel rewrite to use surface objects or global memory |
| ispass-DG | ISPASS-2009 | MPI + ParMetis Integration | Requires MPI headers and ParMetis library. Build system uses `common.mk` which doesn't propagate MPI compiler flags correctly. | High - requires makefile surgery to integrate MPI with CUDA SDK build system |
| ispass-WP | ISPASS-2009 | Fortran + m4 + Deprecated CUDA | Uses m4 preprocessor, Fortran compiler (gfortran), and deprecated nvcc options (`-Xopencc`, `compute_10`). Complex multi-stage build pipeline. | High - requires updating entire build pipeline and removing deprecated options |

## Technical Details

### ispass-MUM (MUMmerGPU)
- **File**: `ispass-2009/MUM/mummergpu_kernel.cu`
- **Texture declarations** (lines 16-19):
  ```cuda
  texture<ulong4, 2, cudaReadModeElementType> nodetex;
  texture<ulong4, 2, cudaReadModeElementType> childrentex;
  texture<char, 2, cudaReadModeElementType> reftex;
  ```
- **tex2D usage**: Lines 47, 121, 151, 217, 296, 326, 392 - deeply embedded in algorithm logic
- Unlike Rodinia's mummergpu which has conditional texture usage (`TREETEX` flag), this version has no fallback path

### ispass-DG (Discontinuous Galerkin)
- **File**: `ispass-2009/DG/Makefile`
- **Dependencies**: OpenMPI, ParMetis-3.1 (included in 3rdParty/)
- **Issue**: `common.mk` compiles C files with gcc but doesn't include MPI headers
- ParMetis library builds successfully, but main DG build fails on MPI includes

### ispass-WP (Weather Prediction - WSM5)
- **File**: `ispass-2009/WP/makefile`
- **Build stages**:
  1. m4 preprocessing: `m4 wsm5.cu | sed ...`
  2. Fortran compilation: `gfortran` for `.F` files
  3. CUDA compilation with deprecated options
- **Deprecated options**: `-Xopencc -noinline`, `compute_10`, `compute_20`
- **Issue**: Modern nvcc (CUDA 11+) doesn't support these options

## Workaround

These benchmarks are excluded from trace collection. For research purposes, if these specific benchmarks are needed:

1. **ispass-MUM**: Port to use CUDA texture objects (bindless textures) or surface memory
2. **ispass-DG**: Create a standalone build script using `mpicc` directly
3. **ispass-WP**: Update makefile to remove deprecated options and use modern architecture flags only
