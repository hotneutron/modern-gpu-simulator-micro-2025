#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

enum class ElementFormat {
  F32,
  F16,
  S32,
};

struct Options {
  ElementFormat format = ElementFormat::F32;
  int total_elems = 1024;
  int op_bytes = 1024;
  int repeats = 1;
  int dst_init_int = 1000;
  float dst_init_float = 1000.0f;
};

static void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

static void print_usage(const char *argv0) {
  std::printf("Usage: %s [options]\n", argv0);
  std::printf("  --format f32|f16|s32\n");
  std::printf("  --total-elems N\n");
  std::printf("  --op-bytes N\n");
  std::printf("  --repeats N\n");
  std::printf("  --dst-init-int N\n");
  std::printf("  --dst-init-float F\n");
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

static ElementFormat parse_format_arg(const char *value) {
  if (value == nullptr) {
    std::fprintf(stderr, "Missing value for --format\n");
    std::exit(EXIT_FAILURE);
  }
  if (std::strcmp(value, "f32") == 0) {
    return ElementFormat::F32;
  }
  if (std::strcmp(value, "f16") == 0) {
    return ElementFormat::F16;
  }
  if (std::strcmp(value, "s32") == 0) {
    return ElementFormat::S32;
  }
  std::fprintf(stderr, "Unsupported format: %s\n", value);
  std::exit(EXIT_FAILURE);
}

static Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--format") == 0) {
      options.format = parse_format_arg(argv[++i]);
    } else if (std::strcmp(argv[i], "--total-elems") == 0) {
      options.total_elems = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--op-bytes") == 0) {
      options.op_bytes = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--repeats") == 0) {
      options.repeats = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--dst-init-int") == 0) {
      options.dst_init_int = parse_int_arg(argv[i], argv[++i]);
    } else if (std::strcmp(argv[i], "--dst-init-float") == 0) {
      options.dst_init_float = parse_float_arg(argv[i], argv[++i]);
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

template <typename T>
static __device__ __forceinline__ void issue_ublkred_add(T *dst_mem,
                                                         const T *src_mem,
                                                         int32_t op_bytes);

template <>
__device__ __forceinline__ void issue_ublkred_add<float>(float *dst_mem,
                                                         const float *src_mem,
                                                         int32_t op_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t smem_ptr = cast_smem_ptr_to_uint(src_mem);
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
      :
      : "l"(dst_mem), "r"(smem_ptr), "r"(op_bytes)
      : "memory");
#endif
}

template <>
__device__ __forceinline__ void issue_ublkred_add<__half>(__half *dst_mem,
                                                          const __half *src_mem,
                                                          int32_t op_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t smem_ptr = cast_smem_ptr_to_uint(src_mem);
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%0], [%1], %2;"
      :
      : "l"(dst_mem), "r"(smem_ptr), "r"(op_bytes)
      : "memory");
#endif
}

template <>
__device__ __forceinline__ void issue_ublkred_add<int32_t>(int32_t *dst_mem,
                                                           const int32_t *src_mem,
                                                           int32_t op_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t smem_ptr = cast_smem_ptr_to_uint(src_mem);
  asm volatile(
      "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s32 [%0], [%1], %2;"
      :
      : "l"(dst_mem), "r"(smem_ptr), "r"(op_bytes)
      : "memory");
#endif
}

template <typename T>
static __host__ __device__ __forceinline__ T initial_value_for_index(int idx, T base);

template <>
__host__ __device__ __forceinline__ float initial_value_for_index<float>(int idx, float base) {
  return base + static_cast<float>(idx + 1);
}

template <>
__host__ __device__ __forceinline__ __half initial_value_for_index<__half>(int idx, __half base) {
  return __float2half(__half2float(base) + static_cast<float>(idx + 1));
}

template <>
__host__ __device__ __forceinline__ int32_t initial_value_for_index<int32_t>(int idx, int32_t base) {
  return base + static_cast<int32_t>(idx + 1);
}

template <typename T>
static __host__ __device__ __forceinline__ double value_to_double(T value) {
  return static_cast<double>(value);
}

template <>
__host__ __device__ __forceinline__ double value_to_double<__half>(__half value) {
  return static_cast<double>(__half2float(value));
}

template <typename T>
static __host__ __device__ __forceinline__ bool values_equal(T a, T b) {
  return a == b;
}

template <>
__host__ __device__ __forceinline__ bool values_equal<__half>(__half a, __half b) {
  return __half2float(a) == __half2float(b);
}

template <typename T>
static __host__ __device__ __forceinline__ T add_reduction_value(T base, int repeats, int idx) {
  return static_cast<T>(base + static_cast<T>(repeats * (idx + 1)));
}

template <>
__host__ __device__ __forceinline__ __half add_reduction_value<__half>(__half base, int repeats, int idx) {
  return __float2half(__half2float(base) + static_cast<float>(repeats * (idx + 1)));
}

template <typename T>
__global__ void ublkred_type_probe_kernel(T *dst_mem,
                                          int total_elems,
                                          int op_bytes,
                                          int repeats,
                                          T dst_init) {
  extern __shared__ __align__(128) unsigned char shared_storage[];
  T *smem = reinterpret_cast<T *>(shared_storage);
  int op_elems = op_bytes / static_cast<int>(sizeof(T));
  for (int idx = threadIdx.x; idx < op_elems; idx += blockDim.x) {
    smem[idx] = initial_value_for_index<T>(idx, static_cast<T>(0));
  }
  for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
    dst_mem[idx] = dst_init;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    fence_proxy_async_shared_cta();
    for (int iter = 0; iter < repeats; ++iter) {
      issue_ublkred_add<T>(dst_mem, smem, op_bytes);
      cp_async_bulk_commit_group();
      cp_async_bulk_wait_group_read_0();
    }
  }
  __syncthreads();
}

static void validate_options(const Options &options) {
  if (options.total_elems <= 0 || options.op_bytes <= 0 || options.repeats <= 0) {
    std::fprintf(stderr, "All sizes and repeats must be positive.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.op_bytes % 16 != 0) {
    std::fprintf(stderr, "op-bytes must be a multiple of 16.\n");
    std::exit(EXIT_FAILURE);
  }
  int element_bytes = 4;
  if (options.format == ElementFormat::F16) {
    element_bytes = 2;
  }
  if (options.op_bytes / element_bytes > options.total_elems) {
    std::fprintf(stderr, "total-elems must cover the selected format footprint.\n");
    std::exit(EXIT_FAILURE);
  }
}

template <typename T>
static void print_window(const std::vector<T> &host, int count) {
  int printed = 0;
  while (printed < count) {
    int row_count = count - printed < 8 ? count - printed : 8;
    for (int c = 0; c < row_count; ++c) {
      if constexpr (std::is_same<T, int32_t>::value) {
        std::printf("%8d", static_cast<int>(host[printed + c]));
      } else {
        std::printf("%8.1f", value_to_double(host[printed + c]));
      }
    }
    std::printf("\n");
    printed += row_count;
  }
}

template <typename T>
static int run_probe_typed(const Options &options, const char *format_name, T dst_init) {
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  if (prop.major < 9) {
    std::fprintf(stderr, "ublkred_type_probe requires Hopper-or-newer hardware.\n");
    return EXIT_FAILURE;
  }

  std::vector<T> host(static_cast<size_t>(options.total_elems), T{});
  T *device_output = nullptr;
  check_cuda(cudaMalloc(&device_output, host.size() * sizeof(T)), "cudaMalloc");

  int threads = 256;
  size_t shared_bytes = static_cast<size_t>(options.op_bytes);
  ublkred_type_probe_kernel<T><<<1, threads, shared_bytes>>>(
      device_output,
      options.total_elems,
      options.op_bytes,
      options.repeats,
      dst_init);
  check_cuda(cudaPeekAtLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  check_cuda(cudaMemcpy(host.data(), device_output, host.size() * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy");

  int changed_count = 0;
  bool one_to_one_match = true;
  int expected_changed = options.op_bytes / static_cast<int>(sizeof(T));
  for (int idx = 0; idx < options.total_elems; ++idx) {
    T expected = dst_init;
    if (idx < expected_changed) {
      expected = add_reduction_value<T>(expected, options.repeats, idx);
    }
    if (!values_equal(host[idx], dst_init)) {
      changed_count++;
    }
    if (!values_equal(host[idx], expected)) {
      one_to_one_match = false;
    }
  }

  double checksum = 0.0;
  for (T value : host) {
    checksum += value_to_double(value);
  }

  std::printf("device=%s\n", prop.name);
  std::printf("format=%s total_elems=%d op_bytes=%d repeats=%d\n",
              format_name,
              options.total_elems,
              options.op_bytes,
              options.repeats);
  std::printf("changed_count=%d\n", changed_count);
  std::printf("expected_changed=%d\n", expected_changed);
  std::printf("actual_changed_bytes=%d\n", changed_count * static_cast<int>(sizeof(T)));
  std::printf("one_to_one_match=%s\n", one_to_one_match ? "true" : "false");
  std::printf("checksum=%0.1f\n", checksum);
  print_window(host, expected_changed < 32 ? expected_changed : 32);

  check_cuda(cudaFree(device_output), "cudaFree");
  return one_to_one_match && changed_count == expected_changed ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv) {
  Options options = parse_options(argc, argv);
  validate_options(options);

  if (options.format == ElementFormat::F32) {
    return run_probe_typed<float>(options, "f32", options.dst_init_float);
  }
  if (options.format == ElementFormat::F16) {
    return run_probe_typed<__half>(options, "f16", __float2half(options.dst_init_float));
  }
  return run_probe_typed<int32_t>(options, "s32", options.dst_init_int);
}
