#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Options {
  int global_d0 = 1;
  int global_d1 = 1;
  int global_d2 = 64;
  int global_d3 = 64;
  int pitch_d3 = 64;
  int box_d0 = 1;
  int box_d1 = 1;
  int box_d2 = 16;
  int box_d3 = 16;
  int coord_d0 = 0;
  int coord_d1 = 0;
  int coord_d2 = 0;
  int coord_d3 = 0;
  int smem_offset = 0;
  int alu_gap_iters = 64;
  int predicate_flag = 1;
  int fill_a = 100;
  int fill_b = 10000;
  std::string variant = "baseline";
};

enum class Variant {
  Baseline,
  Operand1Alt,
  Operand2Alt,
  Predicate,
};

static void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

static void check_driver(CUresult status, const char *what) {
  if (status != CUDA_SUCCESS) {
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    std::fprintf(stderr,
                 "%s failed: %s (%s)\n",
                 what,
                 name != nullptr ? name : "unknown",
                 message != nullptr ? message : "unknown");
    std::exit(EXIT_FAILURE);
  }
}

static void print_usage(const char *argv0) {
  std::printf("Usage: %s [options]\n", argv0);
  std::printf("  --variant baseline|op1_alt|op2_alt|pred\n");
  std::printf("  --global-d0 N\n");
  std::printf("  --global-d1 N\n");
  std::printf("  --global-d2 N\n");
  std::printf("  --global-d3 N\n");
  std::printf("  --pitch-d3 N\n");
  std::printf("  --box-d0 N\n");
  std::printf("  --box-d1 N\n");
  std::printf("  --box-d2 N\n");
  std::printf("  --box-d3 N\n");
  std::printf("  --coord-d0 N\n");
  std::printf("  --coord-d1 N\n");
  std::printf("  --coord-d2 N\n");
  std::printf("  --coord-d3 N\n");
  std::printf("  --smem-offset N\n");
  std::printf("  --alu-gap-iters N\n");
  std::printf("  --predicate-flag 0|1\n");
  std::printf("  --fill-a N\n");
  std::printf("  --fill-b N\n");
}

static int parse_int_arg(const char *flag, const char *value) {
  if (value == nullptr) {
    std::fprintf(stderr, "Missing value for %s\n", flag);
    std::exit(EXIT_FAILURE);
  }
  return std::atoi(value);
}

static Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--variant") == 0) {
      const char *value = argv[++i];
      if (value == nullptr) {
        std::fprintf(stderr, "Missing value for --variant\n");
        std::exit(EXIT_FAILURE);
      }
      options.variant = value;
    } else if (std::strcmp(argv[i], "--global-d0") == 0) {
      options.global_d0 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--global-d1") == 0) {
      options.global_d1 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--global-d2") == 0) {
      options.global_d2 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--global-d3") == 0) {
      options.global_d3 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--pitch-d3") == 0) {
      options.pitch_d3 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--box-d0") == 0) {
      options.box_d0 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--box-d1") == 0) {
      options.box_d1 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--box-d2") == 0) {
      options.box_d2 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--box-d3") == 0) {
      options.box_d3 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--coord-d0") == 0) {
      options.coord_d0 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--coord-d1") == 0) {
      options.coord_d1 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--coord-d2") == 0) {
      options.coord_d2 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--coord-d3") == 0) {
      options.coord_d3 = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--smem-offset") == 0) {
      options.smem_offset = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--alu-gap-iters") == 0) {
      options.alu_gap_iters = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--predicate-flag") == 0) {
      options.predicate_flag = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--fill-a") == 0) {
      options.fill_a = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--fill-b") == 0) {
      options.fill_b = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  return options;
}

static Variant parse_variant(const std::string &variant) {
  if (variant == "baseline") return Variant::Baseline;
  if (variant == "op1_alt") return Variant::Operand1Alt;
  if (variant == "op2_alt") return Variant::Operand2Alt;
  if (variant == "pred") return Variant::Predicate;
  std::fprintf(stderr, "Unknown variant: %s\n", variant.c_str());
  std::exit(EXIT_FAILURE);
}

static void validate_options(const Options &options) {
  if (options.global_d0 <= 0 || options.global_d1 <= 0 || options.global_d2 <= 0 ||
      options.global_d3 <= 0 || options.pitch_d3 <= 0 || options.box_d0 <= 0 ||
      options.box_d1 <= 0 || options.box_d2 <= 0 || options.box_d3 <= 0 ||
      options.smem_offset < 0 || options.alu_gap_iters < 0) {
    std::fprintf(stderr, "All dimensions must be positive; offsets and gap must be non-negative.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.pitch_d3 < options.global_d3) {
    std::fprintf(stderr, "pitch-d3 must be >= global-d3.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.coord_d0 < 0 || options.coord_d1 < 0 || options.coord_d2 < 0 || options.coord_d3 < 0) {
    std::fprintf(stderr, "Coordinates must be non-negative.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.coord_d0 + options.box_d0 > options.global_d0 ||
      options.coord_d1 + options.box_d1 > options.global_d1 ||
      options.coord_d2 + options.box_d2 > options.global_d2 ||
      options.coord_d3 + options.box_d3 > options.global_d3) {
    std::fprintf(stderr, "Requested box exceeds tensor-map bounds.\n");
    std::exit(EXIT_FAILURE);
  }
  if (!(options.predicate_flag == 0 || options.predicate_flag == 1)) {
    std::fprintf(stderr, "predicate-flag must be 0 or 1.\n");
    std::exit(EXIT_FAILURE);
  }
}

static size_t linear_index(const Options &options, int d0, int d1, int d2, int d3) {
  return ((static_cast<size_t>(d0) * static_cast<size_t>(options.global_d1) + static_cast<size_t>(d1)) *
              static_cast<size_t>(options.global_d2) +
          static_cast<size_t>(d2)) *
             static_cast<size_t>(options.pitch_d3) +
         static_cast<size_t>(d3);
}

static __device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(const void *ptr) {
  uint32_t out;
  asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr;}"
      : "=r"(out)
      : "l"(ptr));
  return out;
}

static __device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

static __device__ __forceinline__ void mbarrier_init(uint64_t *bar, unsigned arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(cast_smem_ptr_to_uint(bar)), "r"(arrivals)
               : "memory");
}

static __device__ __forceinline__ void mbarrier_expect_tx(uint64_t *bar, unsigned bytes) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
               :
               : "r"(cast_smem_ptr_to_uint(bar)), "r"(bytes)
               : "memory");
}

static __device__ __forceinline__ uint64_t mbarrier_arrive(uint64_t *bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
               : "=l"(state)
               : "r"(cast_smem_ptr_to_uint(bar))
               : "memory");
  return state;
}

static __device__ __forceinline__ void mbarrier_wait(uint64_t *bar, uint64_t state) {
  uint32_t done = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], %2; selp.b32 %0, 1, 0, p; }"
                 : "=r"(done)
                 : "r"(cast_smem_ptr_to_uint(bar)), "l"(state)
                 : "memory");
  } while (done == 0);
}

static __device__ __forceinline__ void issue_utmapf(const void *tensor_map,
                                                    int32_t coord0,
                                                    int32_t coord1,
                                                    int32_t coord2,
                                                    int32_t coord3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];"
               :
               : "l"(tensor_map), "r"(coord0), "r"(coord1), "r"(coord2), "r"(coord3)
               : "memory");
#endif
}

static __device__ __forceinline__ void issue_utmapf_predicated(const void *tensor_map,
                                                               int32_t coord0,
                                                               int32_t coord1,
                                                               int32_t coord2,
                                                               int32_t coord3,
                                                               int predicate_flag) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("{ .reg .pred p; setp.ne.s32 p, %5, 0; @p cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}]; }"
               :
               : "l"(tensor_map),
                 "r"(coord0),
                 "r"(coord1),
                 "r"(coord2),
                 "r"(coord3),
                 "r"(predicate_flag)
               : "memory");
#endif
}

static __device__ __forceinline__ void issue_utmaldg(const void *tensor_map,
                                                     void *dst_mem,
                                                     int32_t coord0,
                                                     int32_t coord1,
                                                     int32_t coord2,
                                                     int32_t coord3,
                                                     uint64_t *bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
               "[%0], [%1, {%2, %3, %4, %5}], [%6];"
               :
               : "r"(cast_smem_ptr_to_uint(dst_mem)),
                 "l"(tensor_map),
                 "r"(coord0),
                 "r"(coord1),
                 "r"(coord2),
                 "r"(coord3),
                 "r"(cast_smem_ptr_to_uint(bar))
               : "memory");
#endif
}

static __device__ __forceinline__ void do_alu_gap(int iters) {
  uint32_t x = static_cast<uint32_t>(iters + 1);
  for (int i = 0; i < iters; ++i) {
    asm volatile("add.u32 %0, %0, 3;" : "+r"(x));
  }
  asm volatile("" : : "r"(x));
}

template <Variant V>
__device__ void run_probe(const CUtensorMap &tensor_map_a,
                          const CUtensorMap &tensor_map_b,
                          int coord0,
                          int coord1,
                          int coord2,
                          int coord3,
                          int alt_coord0,
                          int alt_coord1,
                          int alt_coord2,
                          int alt_coord3,
                          int predicate_flag,
                          int smem_offset,
                          int tile_elems,
                          int transaction_bytes,
                          int alu_gap_iters,
                          float *output) {
  extern __shared__ __align__(128) float smem[];
  __shared__ alignas(16) uint64_t bar[1];
  for (int idx = threadIdx.x; idx < tile_elems + smem_offset; idx += blockDim.x) {
    smem[idx] = -1.0f;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const CUtensorMap *prefetch_map = &tensor_map_a;
    int pf_coord0 = coord0;
    int pf_coord1 = coord1;
    int pf_coord2 = coord2;
    int pf_coord3 = coord3;
    if constexpr (V == Variant::Operand1Alt) {
      prefetch_map = &tensor_map_b;
    }
    if constexpr (V == Variant::Operand2Alt) {
      pf_coord0 = alt_coord0;
      pf_coord1 = alt_coord1;
      pf_coord2 = alt_coord2;
      pf_coord3 = alt_coord3;
    }
    mbarrier_init(bar, 1);
    mbarrier_expect_tx(bar, static_cast<unsigned>(transaction_bytes));
    if constexpr (V == Variant::Predicate) {
      if (predicate_flag != 0) {
        issue_utmapf_predicated(prefetch_map, pf_coord0, pf_coord1, pf_coord2, pf_coord3, 1);
      }
    } else {
      issue_utmapf(prefetch_map, pf_coord0, pf_coord1, pf_coord2, pf_coord3);
    }
    do_alu_gap(alu_gap_iters);
    issue_utmaldg(&tensor_map_a, smem + smem_offset, coord0, coord1, coord2, coord3, bar);
    uint64_t state = mbarrier_arrive(bar);
    mbarrier_wait(bar, state);
    fence_proxy_async_shared_cta();
    for (int idx = 0; idx < tile_elems; ++idx) {
      output[idx] = smem[smem_offset + idx];
    }
  }
}

__global__ void utmapf_probe_kernel_baseline(const __grid_constant__ CUtensorMap tensor_map_a,
                                             const __grid_constant__ CUtensorMap tensor_map_b,
                                             int coord0,
                                             int coord1,
                                             int coord2,
                                             int coord3,
                                             int alt_coord0,
                                             int alt_coord1,
                                             int alt_coord2,
                                             int alt_coord3,
                                             int smem_offset,
                                             int tile_elems,
                                             int transaction_bytes,
                                             int alu_gap_iters,
                                             float *output) {
  run_probe<Variant::Baseline>(tensor_map_a,
                               tensor_map_b,
                               coord0,
                               coord1,
                               coord2,
                               coord3,
                               alt_coord0,
                               alt_coord1,
                               alt_coord2,
                               alt_coord3,
                               1,
                               smem_offset,
                               tile_elems,
                               transaction_bytes,
                               alu_gap_iters,
                               output);
}

__global__ void utmapf_probe_kernel_op1_alt(const __grid_constant__ CUtensorMap tensor_map_a,
                                            const __grid_constant__ CUtensorMap tensor_map_b,
                                            int coord0,
                                            int coord1,
                                            int coord2,
                                            int coord3,
                                            int alt_coord0,
                                            int alt_coord1,
                                            int alt_coord2,
                                            int alt_coord3,
                                            int smem_offset,
                                            int tile_elems,
                                            int transaction_bytes,
                                            int alu_gap_iters,
                                            float *output) {
  run_probe<Variant::Operand1Alt>(tensor_map_a,
                                  tensor_map_b,
                                  coord0,
                                  coord1,
                                  coord2,
                                  coord3,
                                  alt_coord0,
                                  alt_coord1,
                                  alt_coord2,
                                  alt_coord3,
                                  1,
                                  smem_offset,
                                  tile_elems,
                                  transaction_bytes,
                                  alu_gap_iters,
                                  output);
}

__global__ void utmapf_probe_kernel_op2_alt(const __grid_constant__ CUtensorMap tensor_map_a,
                                            const __grid_constant__ CUtensorMap tensor_map_b,
                                            int coord0,
                                            int coord1,
                                            int coord2,
                                            int coord3,
                                            int alt_coord0,
                                            int alt_coord1,
                                            int alt_coord2,
                                            int alt_coord3,
                                            int smem_offset,
                                            int tile_elems,
                                            int transaction_bytes,
                                            int alu_gap_iters,
                                            float *output) {
  run_probe<Variant::Operand2Alt>(tensor_map_a,
                                  tensor_map_b,
                                  coord0,
                                  coord1,
                                  coord2,
                                  coord3,
                                  alt_coord0,
                                  alt_coord1,
                                  alt_coord2,
                                  alt_coord3,
                                  1,
                                  smem_offset,
                                  tile_elems,
                                  transaction_bytes,
                                  alu_gap_iters,
                                  output);
}

__global__ void utmapf_probe_kernel_predicated(const __grid_constant__ CUtensorMap tensor_map_a,
                                               const __grid_constant__ CUtensorMap tensor_map_b,
                                               int coord0,
                                               int coord1,
                                               int coord2,
                                               int coord3,
                                               int alt_coord0,
                                               int alt_coord1,
                                               int alt_coord2,
                                               int alt_coord3,
                                               int predicate_flag,
                                               int smem_offset,
                                               int tile_elems,
                                               int transaction_bytes,
                                               int alu_gap_iters,
                                               float *output) {
  run_probe<Variant::Predicate>(tensor_map_a,
                                tensor_map_b,
                                coord0,
                                coord1,
                                coord2,
                                coord3,
                                alt_coord0,
                                alt_coord1,
                                alt_coord2,
                                alt_coord3,
                                predicate_flag,
                                smem_offset,
                                tile_elems,
                                transaction_bytes,
                                alu_gap_iters,
                                output);
}

static void fill_tensor(std::vector<float> &host, const Options &options, int base) {
  for (int d0 = 0; d0 < options.global_d0; ++d0) {
    for (int d1 = 0; d1 < options.global_d1; ++d1) {
      for (int d2 = 0; d2 < options.global_d2; ++d2) {
        for (int d3 = 0; d3 < options.global_d3; ++d3) {
          size_t index = linear_index(options, d0, d1, d2, d3);
          host[index] = static_cast<float>(base + d0 * 100000 + d1 * 10000 + d2 * 100 + d3);
        }
      }
    }
  }
}

static void print_window(const std::vector<float> &host, int count) {
  int limit = std::min(count, static_cast<int>(host.size()));
  for (int i = 0; i < limit; ++i) {
    std::printf("%8.1f", host[i]);
  }
  std::printf("\n");
}

int main(int argc, char **argv) {
  Options options = parse_options(argc, argv);
  validate_options(options);
  Variant variant = parse_variant(options.variant);

  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  if (prop.major < 9) {
    std::fprintf(stderr, "utmapf_probe requires Hopper-or-newer hardware.\n");
    return EXIT_FAILURE;
  }

  check_driver(cuInit(0), "cuInit");

  size_t alloc_elems = static_cast<size_t>(options.global_d0) * static_cast<size_t>(options.global_d1) *
                       static_cast<size_t>(options.global_d2) * static_cast<size_t>(options.pitch_d3);
  std::vector<float> host_a(alloc_elems, 0.0f);
  std::vector<float> host_b(alloc_elems, 0.0f);
  fill_tensor(host_a, options, options.fill_a);
  fill_tensor(host_b, options, options.fill_b);

  float *device_input_a = nullptr;
  float *device_input_b = nullptr;
  int tile_elems = options.box_d0 * options.box_d1 * options.box_d2 * options.box_d3;
  float *device_output = nullptr;
  std::vector<float> host_output(tile_elems, 0.0f);

  check_cuda(cudaMalloc(&device_input_a, alloc_elems * sizeof(float)), "cudaMalloc input_a");
  check_cuda(cudaMalloc(&device_input_b, alloc_elems * sizeof(float)), "cudaMalloc input_b");
  check_cuda(cudaMalloc(&device_output, static_cast<size_t>(tile_elems) * sizeof(float)), "cudaMalloc output");
  check_cuda(cudaMemcpy(device_input_a, host_a.data(), alloc_elems * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy input_a");
  check_cuda(cudaMemcpy(device_input_b, host_b.data(), alloc_elems * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy input_b");
  check_cuda(cudaMemset(device_output, 0, static_cast<size_t>(tile_elems) * sizeof(float)), "cudaMemset output");

  CUtensorMap tensor_map_a{};
  CUtensorMap tensor_map_b{};
  cuuint64_t global_dim[4] = {static_cast<cuuint64_t>(options.global_d3),
                              static_cast<cuuint64_t>(options.global_d2),
                              static_cast<cuuint64_t>(options.global_d1),
                              static_cast<cuuint64_t>(options.global_d0)};
  cuuint64_t global_strides[3] = {static_cast<cuuint64_t>(options.pitch_d3) * sizeof(float),
                                  static_cast<cuuint64_t>(options.global_d2) *
                                      static_cast<cuuint64_t>(options.pitch_d3) * sizeof(float),
                                  static_cast<cuuint64_t>(options.global_d1) *
                                      static_cast<cuuint64_t>(options.global_d2) *
                                      static_cast<cuuint64_t>(options.pitch_d3) * sizeof(float)};
  cuuint32_t box_dim[4] = {static_cast<cuuint32_t>(options.box_d3),
                           static_cast<cuuint32_t>(options.box_d2),
                           static_cast<cuuint32_t>(options.box_d1),
                           static_cast<cuuint32_t>(options.box_d0)};
  cuuint32_t element_strides[4] = {1u, 1u, 1u, 1u};

  check_driver(cuTensorMapEncodeTiled(&tensor_map_a,
                                      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                      4,
                                      device_input_a,
                                      global_dim,
                                      global_strides,
                                      box_dim,
                                      element_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled tensor_map_a");
  check_driver(cuTensorMapEncodeTiled(&tensor_map_b,
                                      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                      4,
                                      device_input_b,
                                      global_dim,
                                      global_strides,
                                      box_dim,
                                      element_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled tensor_map_b");

  int alt_coord2 = std::min(options.coord_d2 + options.box_d2, options.global_d2 - options.box_d2);
  int alt_coord3 = std::min(options.coord_d3 + options.box_d3, options.global_d3 - options.box_d3);
  int transaction_bytes = tile_elems * static_cast<int>(sizeof(float));
  size_t shared_bytes = static_cast<size_t>(tile_elems + options.smem_offset) * sizeof(float);
  int threads = 128;

  switch (variant) {
    case Variant::Baseline:
      utmapf_probe_kernel_baseline<<<1, threads, shared_bytes>>>(tensor_map_a,
                                                                 tensor_map_b,
                                                                 options.coord_d3,
                                                                 options.coord_d2,
                                                                 options.coord_d1,
                                                                 options.coord_d0,
                                                                 alt_coord3,
                                                                 alt_coord2,
                                                                 options.coord_d1,
                                                                 options.coord_d0,
                                                                 options.smem_offset,
                                                                 tile_elems,
                                                                 transaction_bytes,
                                                                 options.alu_gap_iters,
                                                                 device_output);
      break;
    case Variant::Operand1Alt:
      utmapf_probe_kernel_op1_alt<<<1, threads, shared_bytes>>>(tensor_map_a,
                                                                tensor_map_b,
                                                                options.coord_d3,
                                                                options.coord_d2,
                                                                options.coord_d1,
                                                                options.coord_d0,
                                                                alt_coord3,
                                                                alt_coord2,
                                                                options.coord_d1,
                                                                options.coord_d0,
                                                                options.smem_offset,
                                                                tile_elems,
                                                                transaction_bytes,
                                                                options.alu_gap_iters,
                                                                device_output);
      break;
    case Variant::Operand2Alt:
      utmapf_probe_kernel_op2_alt<<<1, threads, shared_bytes>>>(tensor_map_a,
                                                                tensor_map_b,
                                                                options.coord_d3,
                                                                options.coord_d2,
                                                                options.coord_d1,
                                                                options.coord_d0,
                                                                alt_coord3,
                                                                alt_coord2,
                                                                options.coord_d1,
                                                                options.coord_d0,
                                                                options.smem_offset,
                                                                tile_elems,
                                                                transaction_bytes,
                                                                options.alu_gap_iters,
                                                                device_output);
      break;
    case Variant::Predicate:
      utmapf_probe_kernel_predicated<<<1, threads, shared_bytes>>>(tensor_map_a,
                                                                   tensor_map_b,
                                                                   options.coord_d3,
                                                                   options.coord_d2,
                                                                   options.coord_d1,
                                                                   options.coord_d0,
                                                                   alt_coord3,
                                                                   alt_coord2,
                                                                   options.coord_d1,
                                                                   options.coord_d0,
                                                                   options.predicate_flag,
                                                                   options.smem_offset,
                                                                   tile_elems,
                                                                   transaction_bytes,
                                                                   options.alu_gap_iters,
                                                                   device_output);
      break;
  }

  check_cuda(cudaPeekAtLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  check_cuda(cudaMemcpy(host_output.data(),
                        device_output,
                        static_cast<size_t>(tile_elems) * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy output");

  double checksum = 0.0;
  for (float value : host_output) {
    checksum += static_cast<double>(value);
  }

  std::printf("device=%s\n", prop.name);
  std::printf("variant=%s predicate_flag=%d\n", options.variant.c_str(), options.predicate_flag);
  std::printf("global=%dx%dx%dx%d pitch_d3=%d box=%dx%dx%dx%d coord=(%d,%d,%d,%d) alt_coord=(%d,%d,%d,%d)\n",
              options.global_d0,
              options.global_d1,
              options.global_d2,
              options.global_d3,
              options.pitch_d3,
              options.box_d0,
              options.box_d1,
              options.box_d2,
              options.box_d3,
              options.coord_d0,
              options.coord_d1,
              options.coord_d2,
              options.coord_d3,
              options.coord_d0,
              options.coord_d1,
              alt_coord2,
              alt_coord3);
  std::printf("smem_offset=%d alu_gap_iters=%d fill_a=%d fill_b=%d\n",
              options.smem_offset,
              options.alu_gap_iters,
              options.fill_a,
              options.fill_b);
  std::printf("checksum=%0.1f\n", checksum);
  print_window(host_output, 16);

  check_cuda(cudaFree(device_output), "cudaFree output");
  check_cuda(cudaFree(device_input_b), "cudaFree input_b");
  check_cuda(cudaFree(device_input_a), "cudaFree input_a");
  return EXIT_SUCCESS;
}
