#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Options {
  int barrier_slot = 0;
  int alu_gap_iters = 32;
  int init_arrivals = 1;
  int arrive_count = 1;
  int arrive_repeats = 1;
  int phase_cycles = 4;
  int reuse_barrier = 1;
};

static void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

static void print_usage(const char *argv0) {
  std::printf("Usage: %s [options]\n", argv0);
  std::printf("  --barrier-slot 0..3\n");
  std::printf("  --init-arrivals N\n");
  std::printf("  --arrive-count N\n");
  std::printf("  --arrive-repeats N\n");
  std::printf("  --phase-cycles N\n");
  std::printf("  --reuse-barrier 0|1\n");
  std::printf("  --alu-gap-iters N\n");
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
    if (std::strcmp(argv[i], "--barrier-slot") == 0) {
      const char *flag = argv[i];
      const char *value = (i + 1 < argc) ? argv[++i] : nullptr;
      options.barrier_slot = parse_int_arg(flag, value);
    } else if (std::strcmp(argv[i], "--init-arrivals") == 0) {
      const char *flag = argv[i];
      const char *value = (i + 1 < argc) ? argv[++i] : nullptr;
      options.init_arrivals = parse_int_arg(flag, value);
    } else if (std::strcmp(argv[i], "--arrive-count") == 0) {
      const char *flag = argv[i];
      const char *value = (i + 1 < argc) ? argv[++i] : nullptr;
      options.arrive_count = parse_int_arg(flag, value);
    } else if (std::strcmp(argv[i], "--arrive-repeats") == 0) {
      const char *flag = argv[i];
      const char *value = (i + 1 < argc) ? argv[++i] : nullptr;
      options.arrive_repeats = parse_int_arg(flag, value);
    } else if (std::strcmp(argv[i], "--phase-cycles") == 0) {
      const char *flag = argv[i];
      const char *value = (i + 1 < argc) ? argv[++i] : nullptr;
      options.phase_cycles = parse_int_arg(flag, value);
    } else if (std::strcmp(argv[i], "--reuse-barrier") == 0) {
      const char *flag = argv[i];
      const char *value = (i + 1 < argc) ? argv[++i] : nullptr;
      options.reuse_barrier = parse_int_arg(flag, value);
    } else if (std::strcmp(argv[i], "--alu-gap-iters") == 0) {
      const char *flag = argv[i];
      const char *value = (i + 1 < argc) ? argv[++i] : nullptr;
      options.alu_gap_iters = parse_int_arg(flag, value);
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

  if (options.barrier_slot < 0 || options.barrier_slot > 3) {
    std::fprintf(stderr, "--barrier-slot must be in [0, 3]\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.init_arrivals <= 0 || options.arrive_count <= 0 ||
      options.arrive_repeats <= 0 || options.phase_cycles <= 0) {
    std::fprintf(stderr,
                 "init/arrive/phase options must all be positive integers\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.reuse_barrier != 0 && options.reuse_barrier != 1) {
    std::fprintf(stderr, "--reuse-barrier must be 0 or 1\n");
    std::exit(EXIT_FAILURE);
  }
  return options;
}

static __device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(
    const void *ptr) {
  uint32_t out;
  asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
      "smem_ptr;}"
      : "=r"(out)
      : "l"(ptr));
  return out;
}

static __device__ __forceinline__ void mbarrier_init(uint64_t *bar,
                                                     unsigned arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(cast_smem_ptr_to_uint(bar)), "r"(arrivals)
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

static __device__ __forceinline__ uint64_t mbarrier_arrive_count(uint64_t *bar,
                                                                 unsigned count) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1], %2;"
               : "=l"(state)
               : "r"(cast_smem_ptr_to_uint(bar)), "r"(count)
               : "memory");
  return state;
}

static __device__ __forceinline__ void mbarrier_wait(uint64_t *bar,
                                                     uint64_t state) {
  uint32_t done = 0;
  do {
    asm volatile(
        "{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], %2; "
        "selp.b32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(cast_smem_ptr_to_uint(bar)), "l"(state)
        : "memory");
  } while (done == 0);
}

static __device__ __forceinline__ void do_alu_gap(int iters) {
  uint32_t x = static_cast<uint32_t>(iters + 1);
  for (int i = 0; i < iters; ++i) {
    asm volatile("add.u32 %0, %0, 3;" : "+r"(x));
  }
  asm volatile("" : : "r"(x));
}

__global__ void sync_arrive_probe_kernel(int barrier_slot,
                                         int init_arrivals,
                                         int arrive_count,
                                         int arrive_repeats,
                                         int phase_cycles,
                                         int reuse_barrier,
                                         int alu_gap_iters,
                                         unsigned long long *output) {
  __shared__ alignas(16) uint64_t barriers[8];
  if (threadIdx.x != 0) {
    return;
  }

  uint64_t *bar = &barriers[barrier_slot * 2];
  if (reuse_barrier != 0) {
    mbarrier_init(bar, static_cast<unsigned>(init_arrivals));
  }

  for (int cycle = 0; cycle < phase_cycles; ++cycle) {
    if (reuse_barrier == 0) {
      mbarrier_init(bar, static_cast<unsigned>(init_arrivals));
    }
    do_alu_gap(alu_gap_iters);

    uint64_t state = 0;
    for (int arrive_idx = 0; arrive_idx < arrive_repeats; ++arrive_idx) {
      state = (arrive_count == 1)
                  ? mbarrier_arrive(bar)
                  : mbarrier_arrive_count(bar, static_cast<unsigned>(arrive_count));
    }
    mbarrier_wait(bar, state);
    output[cycle] = state;
  }
}

int main(int argc, char **argv) {
  const Options options = parse_options(argc, argv);

  std::vector<unsigned long long> host_output(
      static_cast<size_t>(options.phase_cycles), 0ULL);
  unsigned long long *device_output = nullptr;

  check_cuda(cudaMalloc(&device_output,
                        sizeof(unsigned long long) * host_output.size()),
             "cudaMalloc(device_output)");
  check_cuda(cudaMemset(device_output,
                        0,
                        sizeof(unsigned long long) * host_output.size()),
             "cudaMemset(device_output)");

  sync_arrive_probe_kernel<<<1, 32>>>(
      options.barrier_slot, options.init_arrivals, options.arrive_count,
      options.arrive_repeats, options.phase_cycles, options.reuse_barrier,
      options.alu_gap_iters, device_output);
  check_cuda(cudaGetLastError(), "sync_arrive_probe_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  check_cuda(cudaMemcpy(host_output.data(),
                        device_output,
                        sizeof(unsigned long long) * host_output.size(),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(device_output->host_output)");
  check_cuda(cudaFree(device_output), "cudaFree(device_output)");

  std::printf("sync_arrive_probe output states:");
  for (size_t i = 0; i < host_output.size(); ++i) {
    std::printf(" %llu", host_output[i]);
  }
  std::printf("\n");

  return EXIT_SUCCESS;
}
