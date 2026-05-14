#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Options {
  int total_floats = 1024;
  int store_floats = 256;
  int repeats = 1;
  float dst_init = 1000.0f;
};

static void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

static void print_usage(const char *argv0) {
  std::printf("Usage: %s [options]\n", argv0);
  std::printf("  --total-floats N\n");
  std::printf("  --store-floats N\n");
  std::printf("  --repeats N\n");
  std::printf("  --dst-init F\n");
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
    if (std::strcmp(argv[i], "--total-floats") == 0) {
      options.total_floats = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--store-floats") == 0) {
      options.store_floats = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--repeats") == 0) {
      options.repeats = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--dst-init") == 0) {
      options.dst_init = parse_float_arg(argv[i], argv[++i]);
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

static __device__ __forceinline__ void issue_ublkred_add(float *dst_mem,
                                                         const float *src_mem,
                                                         int32_t store_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t smem_ptr = cast_smem_ptr_to_uint(src_mem);
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
      :
      : "l"(dst_mem), "r"(smem_ptr), "r"(store_bytes)
      : "memory");
#endif
}

__global__ void ublkred_count_probe_kernel(float *dst_mem,
                                           int total_floats,
                                           int store_floats,
                                           int repeats,
                                           float dst_init) {
  extern __shared__ __align__(128) float smem[];
  for (int idx = threadIdx.x; idx < store_floats; idx += blockDim.x) {
    smem[idx] = static_cast<float>(idx + 1);
  }
  for (int idx = threadIdx.x; idx < total_floats; idx += blockDim.x) {
    dst_mem[idx] = dst_init;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    fence_proxy_async_shared_cta();
    for (int iter = 0; iter < repeats; ++iter) {
      issue_ublkred_add(dst_mem, smem, store_floats * static_cast<int>(sizeof(float)));
      cp_async_bulk_commit_group();
      cp_async_bulk_wait_group_read_0();
    }
  }
  __syncthreads();
}

static void validate_options(const Options &options) {
  if (options.total_floats <= 0 || options.store_floats <= 0 || options.repeats <= 0) {
    std::fprintf(stderr, "All sizes and repeats must be positive.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.store_floats > options.total_floats) {
    std::fprintf(stderr, "store-floats cannot exceed total-floats.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.store_floats % 4 != 0) {
    std::fprintf(stderr, "store-floats must be a multiple of 4 floats (16 bytes).\n");
    std::exit(EXIT_FAILURE);
  }
}

static void print_window(const std::vector<float> &host, int count) {
  int printed = 0;
  while (printed < count) {
    int row_count = count - printed < 8 ? count - printed : 8;
    for (int c = 0; c < row_count; ++c) {
      std::printf("%8.1f", host[printed + c]);
    }
    std::printf("\n");
    printed += row_count;
  }
}

int main(int argc, char **argv) {
  Options options = parse_options(argc, argv);
  validate_options(options);

  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  if (prop.major < 9) {
    std::fprintf(stderr, "ublkred_count_probe requires Hopper-or-newer hardware.\n");
    return EXIT_FAILURE;
  }

  std::vector<float> host(static_cast<size_t>(options.total_floats), 0.0f);
  float *device_output = nullptr;
  check_cuda(cudaMalloc(&device_output, host.size() * sizeof(float)), "cudaMalloc");

  int threads = 256;
  size_t shared_bytes = static_cast<size_t>(options.store_floats) * sizeof(float);
  ublkred_count_probe_kernel<<<1, threads, shared_bytes>>>(
      device_output,
      options.total_floats,
      options.store_floats,
      options.repeats,
      options.dst_init);
  check_cuda(cudaPeekAtLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  check_cuda(cudaMemcpy(host.data(), device_output, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy");

  int changed_count = 0;
  bool one_to_one_match = true;
  for (int idx = 0; idx < options.total_floats; ++idx) {
    float expected = options.dst_init;
    if (idx < options.store_floats) {
      expected += static_cast<float>(options.repeats * (idx + 1));
    }
    if (host[idx] != options.dst_init) {
      changed_count++;
    }
    if (host[idx] != expected) {
      one_to_one_match = false;
    }
  }

  double checksum = 0.0;
  for (float value : host) {
    checksum += static_cast<double>(value);
  }

  std::printf("device=%s\n", prop.name);
  std::printf("total_floats=%d store_floats=%d repeats=%d dst_init=%0.1f\n",
              options.total_floats,
              options.store_floats,
              options.repeats,
              options.dst_init);
  std::printf("changed_count=%d\n", changed_count);
  std::printf("one_to_one_match=%s\n", one_to_one_match ? "true" : "false");
  std::printf("checksum=%0.1f\n", checksum);
  print_window(host, options.store_floats < 32 ? options.store_floats : 32);

  check_cuda(cudaFree(device_output), "cudaFree");
  return one_to_one_match ? EXIT_SUCCESS : EXIT_FAILURE;
}
