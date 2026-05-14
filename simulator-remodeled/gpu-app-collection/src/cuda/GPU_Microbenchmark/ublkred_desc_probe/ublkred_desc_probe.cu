#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Options {
  int global_rows = 64;
  int global_cols = 64;
  int pitch_cols = 64;
  int box_rows = 16;
  int box_cols = 16;
  int coord_row = 0;
  int coord_col = 0;
  int smem_offset = 0;
  float init = 1.0f;
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
    std::fprintf(stderr, "%s failed: %s (%s)\n",
                 what,
                 name != nullptr ? name : "unknown",
                 message != nullptr ? message : "unknown");
    std::exit(EXIT_FAILURE);
  }
}

static void print_usage(const char *argv0) {
  std::printf("Usage: %s [options]\n", argv0);
  std::printf("  --global-rows N\n");
  std::printf("  --global-cols N\n");
  std::printf("  --pitch-cols N\n");
  std::printf("  --box-rows N\n");
  std::printf("  --box-cols N\n");
  std::printf("  --coord-row N\n");
  std::printf("  --coord-col N\n");
  std::printf("  --smem-offset N\n");
  std::printf("  --init F\n");
}

static int parse_int_arg(const char *flag, const char *value) {
  if (value == nullptr) {
    std::fprintf(stderr, "Missing value for %s\n", flag);
    std::exit(EXIT_FAILURE);
  }
  return std::atoi(value);
}

static float parse_float_arg(const char *flag, const char *value) {
  if (value == nullptr) {
    std::fprintf(stderr, "Missing value for %s\n", flag);
    std::exit(EXIT_FAILURE);
  }
  return std::strtof(value, nullptr);
}

static Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--global-rows") == 0) {
      options.global_rows = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--global-cols") == 0) {
      options.global_cols = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--pitch-cols") == 0) {
      options.pitch_cols = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--box-rows") == 0) {
      options.box_rows = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--box-cols") == 0) {
      options.box_cols = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--coord-row") == 0) {
      options.coord_row = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--coord-col") == 0) {
      options.coord_col = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--smem-offset") == 0) {
      options.smem_offset = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--init") == 0) {
      options.init = parse_float_arg(argv[i], argv[++i]);
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

static __device__ __forceinline__ void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

static __device__ __forceinline__ void cp_async_bulk_wait_group_read_0() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

static __device__ __forceinline__ void issue_ublkred_desc_add_2d(const void *tensor_map,
                                                                 const void *src_mem,
                                                                 int32_t coord_row,
                                                                 int32_t coord_col) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t smem_ptr = cast_smem_ptr_to_uint(src_mem);
  asm volatile(
      "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];"
      :
      : "l"(tensor_map), "r"(smem_ptr), "r"(coord_row), "r"(coord_col)
      : "memory");
#endif
}

__global__ void ublkred_desc_probe_kernel(const __grid_constant__ CUtensorMap tensor_map,
                                          int box_rows,
                                          int box_cols,
                                          int coord_row,
                                          int coord_col,
                                          int smem_offset,
                                          float init_value) {
  extern __shared__ __align__(128) float smem[];
  int tile_elems = box_rows * box_cols;
  int total_elems = tile_elems + smem_offset;
  for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
    smem[idx] = init_value + static_cast<float>(idx);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    fence_proxy_async_shared_cta();
    issue_ublkred_desc_add_2d(&tensor_map, smem + smem_offset, coord_row, coord_col);
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group_read_0();
  }
  __syncthreads();
}

static void validate_options(const Options &options) {
  if (options.global_rows <= 0 || options.global_cols <= 0 || options.pitch_cols <= 0 ||
      options.box_rows <= 0 || options.box_cols <= 0 || options.smem_offset < 0) {
    std::fprintf(stderr, "All dimensions must be positive and smem-offset must be non-negative.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.pitch_cols < options.global_cols) {
    std::fprintf(stderr, "pitch-cols must be >= global-cols.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.coord_row < 0 || options.coord_col < 0) {
    std::fprintf(stderr, "Coordinates must be non-negative.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.coord_row + options.box_rows > options.global_rows ||
      options.coord_col + options.box_cols > options.global_cols) {
    std::fprintf(stderr, "Requested tile exceeds encoded tensor-map bounds.\n");
    std::exit(EXIT_FAILURE);
  }
}

static void print_window(const std::vector<float> &host,
                         const Options &options,
                         int rows,
                         int cols) {
  int max_rows = rows < options.box_rows ? rows : options.box_rows;
  int max_cols = cols < options.box_cols ? cols : options.box_cols;
  for (int r = 0; r < max_rows; ++r) {
    for (int c = 0; c < max_cols; ++c) {
      int index = (options.coord_row + r) * options.pitch_cols + (options.coord_col + c);
      std::printf("%8.1f", host[index]);
    }
    std::printf("\n");
  }
}

int main(int argc, char **argv) {
  Options options = parse_options(argc, argv);
  validate_options(options);

  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  if (prop.major < 9) {
    std::fprintf(stderr, "ublkred_desc_probe requires Hopper-or-newer hardware.\n");
    return EXIT_FAILURE;
  }

  check_driver(cuInit(0), "cuInit");

  size_t alloc_elems = static_cast<size_t>(options.global_rows) * static_cast<size_t>(options.pitch_cols);
  std::vector<float> host(alloc_elems, 0.0f);
  float *device_output = nullptr;
  check_cuda(cudaMalloc(&device_output, alloc_elems * sizeof(float)), "cudaMalloc");
  check_cuda(cudaMemset(device_output, 0, alloc_elems * sizeof(float)), "cudaMemset");

  CUtensorMap tensor_map{};
  cuuint64_t global_dim[2] = {
      static_cast<cuuint64_t>(options.global_rows),
      static_cast<cuuint64_t>(options.global_cols)};
  cuuint64_t global_strides[1] = {
      static_cast<cuuint64_t>(options.pitch_cols * static_cast<int>(sizeof(float)))};
  cuuint32_t box_dim[2] = {
      static_cast<cuuint32_t>(options.box_rows),
      static_cast<cuuint32_t>(options.box_cols)};
  cuuint32_t element_strides[2] = {1u, 1u};

  check_driver(
      cuTensorMapEncodeTiled(&tensor_map,
                             CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                             2,
                             device_output,
                             global_dim,
                             global_strides,
                             box_dim,
                             element_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");

  int threads = 256;
  size_t shared_bytes = static_cast<size_t>(options.box_rows * options.box_cols + options.smem_offset) * sizeof(float);
  ublkred_desc_probe_kernel<<<1, threads, shared_bytes>>>(
      tensor_map,
      options.box_rows,
      options.box_cols,
      options.coord_row,
      options.coord_col,
      options.smem_offset,
      options.init);
  check_cuda(cudaPeekAtLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  check_cuda(cudaMemcpy(host.data(), device_output, alloc_elems * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy");

  double checksum = 0.0;
  for (float value : host) {
    checksum += static_cast<double>(value);
  }

  std::printf("device=%s\n", prop.name);
  std::printf("global=%dx%d pitch=%d box=%dx%d coord=(%d,%d) smem_offset=%d init=%0.1f\n",
              options.global_rows,
              options.global_cols,
              options.pitch_cols,
              options.box_rows,
              options.box_cols,
              options.coord_row,
              options.coord_col,
              options.smem_offset,
              options.init);
  std::printf("checksum=%0.1f\n", checksum);
  print_window(host, options, 4, 8);

  check_cuda(cudaFree(device_output), "cudaFree");
  return EXIT_SUCCESS;
}
