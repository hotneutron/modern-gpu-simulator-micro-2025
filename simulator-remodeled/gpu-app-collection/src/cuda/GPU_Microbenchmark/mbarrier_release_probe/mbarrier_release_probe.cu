#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kTileD0 = 1;
constexpr int kTileD1 = 1;
constexpr int kTileD2 = 16;
constexpr int kTileD3 = 16;
constexpr int kTileElements = kTileD0 * kTileD1 * kTileD2 * kTileD3;
constexpr int kTransactionBytes = kTileElements * sizeof(float);

enum class ProbeCase {
  CountMany,
  CountAlias,
  TmaMany,
  TmaAlias,
};

struct Options {
  ProbeCase probe_case = ProbeCase::CountMany;
  unsigned long long delay_cycles = 4096;
};

struct ProbeResult {
  unsigned long long wait_completed;
  unsigned int observed_value;
  unsigned int reserved;
};

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

void check_driver(CUresult status, const char *what) {
  if (status != CUDA_SUCCESS) {
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    std::fprintf(stderr, "%s failed: %s (%s)\n", what,
                 name != nullptr ? name : "unknown",
                 message != nullptr ? message : "unknown");
    std::exit(EXIT_FAILURE);
  }
}

const char *case_name(ProbeCase probe_case) {
  switch (probe_case) {
    case ProbeCase::CountMany:
      return "count_many";
    case ProbeCase::CountAlias:
      return "count_alias";
    case ProbeCase::TmaMany:
      return "tma_many";
    case ProbeCase::TmaAlias:
      return "tma_alias";
  }
  return "unknown";
}

void print_usage(const char *argv0) {
  std::printf("Usage: %s --case count_many|count_alias|tma_many|tma_alias "
              "[--delay-cycles N]\n",
              argv0);
}

unsigned long long parse_ull(const char *flag, const char *value) {
  if (value == nullptr || value[0] == '\0') {
    std::fprintf(stderr, "Missing value for %s\n", flag);
    std::exit(EXIT_FAILURE);
  }
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0') {
    std::fprintf(stderr, "Invalid integer for %s: %s\n", flag, value);
    std::exit(EXIT_FAILURE);
  }
  return parsed;
}

Options parse_options(int argc, char **argv) {
  Options options;
  bool case_seen = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--case") == 0) {
      if (++i >= argc) {
        std::fprintf(stderr, "Missing value for --case\n");
        std::exit(EXIT_FAILURE);
      }
      case_seen = true;
      if (std::strcmp(argv[i], "count_many") == 0) {
        options.probe_case = ProbeCase::CountMany;
      } else if (std::strcmp(argv[i], "count_alias") == 0) {
        options.probe_case = ProbeCase::CountAlias;
      } else if (std::strcmp(argv[i], "tma_many") == 0) {
        options.probe_case = ProbeCase::TmaMany;
      } else if (std::strcmp(argv[i], "tma_alias") == 0) {
        options.probe_case = ProbeCase::TmaAlias;
      } else {
        std::fprintf(stderr, "Unknown --case value: %s\n", argv[i]);
        std::exit(EXIT_FAILURE);
      }
    } else if (std::strcmp(argv[i], "--delay-cycles") == 0) {
      const char *flag = argv[i];
      const char *value = (++i < argc) ? argv[i] : nullptr;
      options.delay_cycles = parse_ull(flag, value);
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (!case_seen) {
    std::fprintf(stderr, "--case is required\n");
    print_usage(argv[0]);
    std::exit(EXIT_FAILURE);
  }
  if (options.delay_cycles == 0) {
    std::fprintf(stderr, "--delay-cycles must be positive\n");
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__device__ __forceinline__ uint32_t shared_address(const void *ptr) {
  uint32_t address;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
      "cvt.u32.u64 %0, smem_ptr; }"
      : "=r"(address)
      : "l"(ptr));
  return address;
}

__device__ __forceinline__ void mbarrier_init(uint64_t *barrier,
                                               unsigned int arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(shared_address(barrier)), "r"(arrivals)
               : "memory");
}

__device__ __forceinline__ uint64_t mbarrier_arrive(uint64_t *barrier) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
               : "=l"(state)
               : "r"(shared_address(barrier))
               : "memory");
  return state;
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t *barrier,
                                                    unsigned int bytes) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
               :
               : "r"(shared_address(barrier)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ bool mbarrier_test_wait_parity(
    uint32_t barrier_address, unsigned int parity) {
  unsigned int ready;
  asm volatile(
      "{ .reg .pred p; "
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; "
      "selp.b32 %0, 1, 0, p; }"
      : "=r"(ready)
      : "r"(barrier_address), "r"(parity)
      : "memory");
  return ready != 0;
}

__device__ __forceinline__ void mbarrier_wait_state(uint64_t *barrier,
                                                     uint64_t state) {
  const uint32_t barrier_address = shared_address(barrier);
  unsigned int ready = 0;
  while (ready == 0) {
    asm volatile(
        "{ .reg .pred p; mbarrier.test_wait.shared::cta.b64 p, [%1], %2; "
        "selp.b32 %0, 1, 0, p; }"
        : "=r"(ready)
        : "r"(barrier_address), "l"(state)
        : "memory");
  }
}

__device__ __forceinline__ void wait_for_parity(
    uint64_t *barrier, unsigned int parity) {
  const uint32_t barrier_address = shared_address(barrier);
  while (!mbarrier_test_wait_parity(barrier_address, parity)) {
    __nanosleep(64);
  }
}

__device__ __forceinline__ void delay_for_cycles(
    unsigned long long delay_cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < delay_cycles) {
    asm volatile("" ::: "memory");
  }
}

__device__ __forceinline__ unsigned int load_shared_counter(
    unsigned int *counter) {
  return atomicAdd(counter, 0u);
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void issue_tma_load_4d(
    const void *tensor_map, void *destination, uint64_t *barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile."
      "mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5}], [%6];"
      :
      : "r"(shared_address(destination)), "l"(tensor_map), "r"(0),
        "r"(0), "r"(0), "r"(0), "r"(shared_address(barrier))
      : "memory");
#endif
}

__device__ __forceinline__ uint64_t issue_tma_phase(
    const void *tensor_map, void *destination, uint64_t *barrier) {
  mbarrier_expect_tx(barrier, kTransactionBytes);
  issue_tma_load_4d(tensor_map, destination, barrier);
  return mbarrier_arrive(barrier);
}

__global__ void count_many_kernel(unsigned long long delay_cycles,
                                  ProbeResult *results) {
  __shared__ alignas(16) uint64_t barrier_storage[2];
  uint64_t *barrier = &barrier_storage[0];
  const unsigned int warp = threadIdx.x / 32;
  const unsigned int lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    mbarrier_init(barrier, 1);
  }
  __syncthreads();

  if (warp == 0) {
    if (lane == 0) {
      delay_for_cycles(delay_cycles);
      (void)mbarrier_arrive(barrier);
    }
    return;
  }

  wait_for_parity(barrier, 0);
  if (lane == 0) {
    results[warp] = {1u, 0u, 0u};
  }
}

__global__ void count_alias_kernel(unsigned long long delay_cycles,
                                   ProbeResult *results) {
  __shared__ alignas(16) uint64_t barrier_storage[2];
  __shared__ unsigned int first_waiters_done;
  __shared__ unsigned int second_flip_done;
  __shared__ unsigned int late_wait_started;
  uint64_t *barrier = &barrier_storage[0];
  const unsigned int warp = threadIdx.x / 32;
  const unsigned int lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    mbarrier_init(barrier, 1);
    first_waiters_done = 0;
    second_flip_done = 0;
    late_wait_started = 0;
  }
  __syncthreads();

  if (warp == 0) {
    if (lane == 0) {
      delay_for_cycles(delay_cycles);
      uint64_t state = mbarrier_arrive(barrier);
      mbarrier_wait_state(barrier, state);
      while (load_shared_counter(&first_waiters_done) != 2) {
        __nanosleep(64);
      }

      delay_for_cycles(delay_cycles);
      state = mbarrier_arrive(barrier);
      mbarrier_wait_state(barrier, state);
      __threadfence_block();
      atomicExch(&second_flip_done, 1u);

      while (load_shared_counter(&late_wait_started) == 0) {
        __nanosleep(64);
      }
      delay_for_cycles(delay_cycles);
      (void)mbarrier_arrive(barrier);
    }
    return;
  }

  if (warp == 1 || warp == 2) {
    wait_for_parity(barrier, 0);
    __syncwarp();
    if (lane == 0) {
      results[warp] = {1u, 0u, 0u};
      atomicAdd(&first_waiters_done, 1u);
    }
    return;
  }

  if (lane == 0) {
    while (load_shared_counter(&second_flip_done) == 0) {
      __nanosleep(64);
    }
    atomicExch(&late_wait_started, 1u);
  }
  __syncwarp();

  // The delayed waiter still asks for parity 0 after events 1 and 2 have
  // already passed. It can only succeed on event 3.
  wait_for_parity(barrier, 0);
  if (lane == 0) {
    results[warp] = {1u, 0u, 0u};
  }
}

__global__ void tma_many_kernel(
    const CUtensorMap *tensor_map,
    unsigned long long delay_cycles, ProbeResult *results) {
  extern __shared__ __align__(128) float tile[];
  __shared__ alignas(16) uint64_t barrier_storage[2];
  uint64_t *barrier = &barrier_storage[0];
  const unsigned int warp = threadIdx.x / 32;
  const unsigned int lane = threadIdx.x % 32;

  for (int index = threadIdx.x; index < kTileElements;
       index += blockDim.x) {
    tile[index] = -1.0f;
  }
  if (threadIdx.x == 0) {
    mbarrier_init(barrier, 1);
  }
  __syncthreads();

  if (warp == 0) {
    if (lane == 0) {
      delay_for_cycles(delay_cycles);
      (void)issue_tma_phase(tensor_map, tile, barrier);
    }
    return;
  }

  wait_for_parity(barrier, 0);
  fence_proxy_async_shared_cta();
  if (lane == 0) {
    results[warp] = {
        1u,
        static_cast<unsigned int>(tile[0]),
        0u,
    };
  }
}

__global__ void tma_alias_kernel(
    const CUtensorMap *tensor_map,
    unsigned long long delay_cycles, ProbeResult *results) {
  extern __shared__ __align__(128) float tile[];
  __shared__ alignas(16) uint64_t barrier_storage[2];
  __shared__ unsigned int first_waiters_done;
  __shared__ unsigned int second_flip_done;
  __shared__ unsigned int late_wait_started;
  uint64_t *barrier = &barrier_storage[0];
  const unsigned int warp = threadIdx.x / 32;
  const unsigned int lane = threadIdx.x % 32;

  for (int index = threadIdx.x; index < kTileElements;
       index += blockDim.x) {
    tile[index] = -1.0f;
  }
  if (threadIdx.x == 0) {
    mbarrier_init(barrier, 1);
    first_waiters_done = 0;
    second_flip_done = 0;
    late_wait_started = 0;
  }
  __syncthreads();

  if (warp == 0) {
    if (lane == 0) {
#pragma unroll 1
      for (unsigned int event = 1; event <= 3; ++event) {
        if (event == 2) {
          while (load_shared_counter(&first_waiters_done) != 2) {
            __nanosleep(64);
          }
        } else if (event == 3) {
          __threadfence_block();
          atomicExch(&second_flip_done, 1u);
          while (load_shared_counter(&late_wait_started) == 0) {
            __nanosleep(64);
          }
        }

        delay_for_cycles(delay_cycles);
        const uint64_t state = issue_tma_phase(tensor_map, tile, barrier);
        if (event < 3) {
          mbarrier_wait_state(barrier, state);
        }
      }
    }
    return;
  }

  if (warp == 1 || warp == 2) {
    wait_for_parity(barrier, 0);
    fence_proxy_async_shared_cta();
    __syncwarp();
    if (lane == 0) {
      results[warp] = {
          1u, static_cast<unsigned int>(tile[0]), 0u,
      };
      atomicAdd(&first_waiters_done, 1u);
    }
    return;
  }

  if (lane == 0) {
    while (load_shared_counter(&second_flip_done) == 0) {
      __nanosleep(64);
    }
    atomicExch(&late_wait_started, 1u);
  }
  __syncwarp();

  wait_for_parity(barrier, 0);
  fence_proxy_async_shared_cta();
  if (lane == 0) {
    results[warp] = {
        1u, static_cast<unsigned int>(tile[0]), 0u,
    };
  }
}

void initialize_tensor_map(CUtensorMap *tensor_map, float *device_input) {
  const cuuint64_t global_dim[4] = {kTileD3, kTileD2, kTileD1, kTileD0};
  const cuuint64_t global_strides[3] = {
      kTileD3 * sizeof(float),
      kTileD2 * kTileD3 * sizeof(float),
      kTileD1 * kTileD2 * kTileD3 * sizeof(float),
  };
  const cuuint32_t box_dim[4] = {kTileD3, kTileD2, kTileD1, kTileD0};
  const cuuint32_t element_strides[4] = {1u, 1u, 1u, 1u};
  check_driver(
      cuTensorMapEncodeTiled(
          tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, device_input,
          global_dim, global_strides, box_dim, element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
}

bool validate_results(ProbeCase probe_case,
                      const std::vector<ProbeResult> &results) {
  bool pass = true;
  for (int warp = 1; warp < kWarps; ++warp) {
    const ProbeResult &result = results[warp];
    std::printf(
        "[MBREL][RESULT] case=%s warp=%d wait_completed=%llu "
        "observed_value=%u\n",
        case_name(probe_case), warp, result.wait_completed,
        result.observed_value);
    if (result.wait_completed != 1) {
      pass = false;
    }
  }
  if (probe_case == ProbeCase::TmaMany ||
      probe_case == ProbeCase::TmaAlias) {
    for (int warp = 1; warp < kWarps; ++warp) {
      if (results[warp].observed_value != 100u) {
        pass = false;
      }
    }
  }
  std::printf("[MBREL][FINAL] status=%s case=%s\n",
              pass ? "PASS" : "FAIL", case_name(probe_case));
  return pass;
}

}  // namespace

int main(int argc, char **argv) {
  const Options options = parse_options(argc, argv);

  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0),
             "cudaGetDeviceProperties");
  if (properties.major < 9) {
    std::fprintf(stderr,
                 "mbarrier_release_probe requires Hopper-or-newer hardware\n");
    return EXIT_FAILURE;
  }
  check_driver(cuInit(0), "cuInit");

  ProbeResult *device_results = nullptr;
  std::vector<ProbeResult> results(kWarps);
  check_cuda(cudaMalloc(&device_results, sizeof(ProbeResult) * kWarps),
             "cudaMalloc(results)");
  check_cuda(cudaMemset(device_results, 0, sizeof(ProbeResult) * kWarps),
             "cudaMemset(results)");

  float *device_input = nullptr;
  CUtensorMap *device_tensor_map = nullptr;
  if (options.probe_case == ProbeCase::CountMany) {
    count_many_kernel<<<1, kThreads>>>(options.delay_cycles, device_results);
  } else if (options.probe_case == ProbeCase::CountAlias) {
    count_alias_kernel<<<1, kThreads>>>(options.delay_cycles, device_results);
  } else {
    std::vector<float> input(kTileElements);
    for (int index = 0; index < kTileElements; ++index) {
      input[index] = static_cast<float>(100 + index);
    }
    check_cuda(cudaMalloc(&device_input, sizeof(float) * input.size()),
               "cudaMalloc(input)");
    check_cuda(cudaMemcpy(device_input, input.data(),
                          sizeof(float) * input.size(),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(input)");
    CUtensorMap tensor_map{};
    initialize_tensor_map(&tensor_map, device_input);
    check_cuda(cudaMalloc(&device_tensor_map, sizeof(CUtensorMap)),
               "cudaMalloc(tensor_map)");
    check_cuda(cudaMemcpy(device_tensor_map, &tensor_map,
                          sizeof(CUtensorMap), cudaMemcpyHostToDevice),
               "cudaMemcpy(tensor_map)");
    if (options.probe_case == ProbeCase::TmaMany) {
      tma_many_kernel<<<1, kThreads, kTransactionBytes>>>(
          device_tensor_map, options.delay_cycles, device_results);
    } else {
      tma_alias_kernel<<<1, kThreads, kTransactionBytes>>>(
          device_tensor_map, options.delay_cycles, device_results);
    }
  }

  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  check_cuda(cudaMemcpy(results.data(), device_results,
                        sizeof(ProbeResult) * results.size(),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(results)");

  const bool pass = validate_results(options.probe_case, results);
  if (device_input != nullptr) {
    check_cuda(cudaFree(device_input), "cudaFree(input)");
  }
  if (device_tensor_map != nullptr) {
    check_cuda(cudaFree(device_tensor_map), "cudaFree(tensor_map)");
  }
  check_cuda(cudaFree(device_results), "cudaFree(results)");
  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
