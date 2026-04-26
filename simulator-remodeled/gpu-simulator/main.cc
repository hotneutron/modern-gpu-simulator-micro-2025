// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <math.h>
#include <set>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <tuple>
#include <vector>

#include "gpgpu_context.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "../ISA_Def/trace_opcode.h"
#include "trace_driven.h"
#include "../trace-parser/trace_parser.h"
#include "accelsim_version.h"

#include <omp.h>

#include <signal.h>
#include <unistd.h>
#include <execinfo.h>
#include <stdlib.h>

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           class trace_config *m_config, option_parser_t &opp);

trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser);

// Roofline classification for the adaptive sim_ctas selector.
enum kernel_class { KCLASS_COMPUTE, KCLASS_MEMORY, KCLASS_MIXED };

static const char* kernel_class_name(kernel_class c) {
  switch (c) {
    case KCLASS_COMPUTE: return "compute";
    case KCLASS_MEMORY:  return "memory";
    case KCLASS_MIXED:   return "mixed";
  }
  return "unknown";
}

// Three-way classifier: ridge_ratio + DRAM-side pressure proxies.
// "Pressure-high" beats ridge_ratio: a kernel reading at >=60% peak BW is
// effectively memory-bound regardless of where its AI puts it on roofline.
// Note: K-rep pressure signals are biased low (under-fills caches/queues),
// so borderline cases are biased toward MEMORY by design.
static kernel_class classify_kernel(const pressure_signals_t& s,
                                    const trace_config& tc) {
  bool memory_pressure_high =
      (s.achieved_bw_ratio >= tc.get_cta_sampling_pressure_bw()) ||
      (s.dram_queue_occupancy_avg >= tc.get_cta_sampling_pressure_queue());
  if (s.ridge_ratio >= tc.get_cta_sampling_t_high() && !memory_pressure_high)
    return KCLASS_COMPUTE;
  if (s.ridge_ratio <= tc.get_cta_sampling_t_low() || memory_pressure_high)
    return KCLASS_MEMORY;
  return KCLASS_MIXED;
}

// Estimate the number of CTAs/SMs needed to saturate DRAM bandwidth based on
// the K-rep run. per_sm_bw_est = (dram_bytes/sim_cycles)/k_reps; N_sat_est =
// ceil(peak_dram_bw / per_sm_bw_est). Plan acknowledges this is noisy: peak
// DRAM BW as denominator over-estimates required SMs for latency-limited
// kernels and under-estimates for partition-camped traffic. The pilot loop
// (later commit) corrects via doubling.
static unsigned compute_n_sat_est(const pressure_signals_t& s, unsigned k_reps) {
  if (s.dram_bytes == 0 || s.sim_cycles == 0 || k_reps == 0)
    return 1;
  double per_sm_bw = ((double)s.dram_bytes / (double)s.sim_cycles) /
                     (double)k_reps;
  if (per_sm_bw <= 0.0 || s.peak_dram_bw_bytes_per_cycle <= 0.0)
    return 1;
  double n = s.peak_dram_bw_bytes_per_cycle / per_sm_bw;
  if (!std::isfinite(n) || n < 1.0) return 1;
  return (unsigned)std::ceil(n);
}

// Pick an initial sim_ctas from the classification. Memory => N_sat_est;
// mixed => 1.5x; compute => sms_floor_compute (half the GPU is sufficient for
// per-SM effects without needing DRAM saturation).
static unsigned compute_initial_sim_ctas(kernel_class kc, unsigned k_reps,
                                         unsigned total_ctas, unsigned total_sms,
                                         unsigned n_sat_est) {
  unsigned target = k_reps;
  switch (kc) {
    case KCLASS_COMPUTE:
      target = std::max(k_reps, total_sms / 2u);
      break;
    case KCLASS_MEMORY:
      target = std::max(k_reps, n_sat_est);
      break;
    case KCLASS_MIXED:
      target = std::max(k_reps,
                        (unsigned)std::ceil(1.5 * (double)n_sat_est));
      break;
  }
  if (total_sms > 0 && target > total_sms) target = total_sms;
  if (target > total_ctas) target = total_ctas;
  if (target < k_reps) target = k_reps;  // never below K
  return target;
}

// Coordinate-based CTA heuristic: selects boundary (corners, edge-midpoints) and
// one interior representative. Works best for regular compute kernels (GEMM, conv,
// stencil) where CTA behavior varies only at grid boundaries.
static std::vector<std::tuple<unsigned,unsigned,unsigned>>
compute_sampled_ctas(unsigned gx, unsigned gy, unsigned gz) {
  std::set<std::tuple<unsigned,unsigned,unsigned>> seen;
  auto add = [&](unsigned x, unsigned y, unsigned z) {
    x = std::min(x, gx - 1);
    y = std::min(y, gy - 1);
    z = std::min(z, gz - 1);
    seen.insert({x, y, z});
  };

  // corners
  for (unsigned z : {0u, gz - 1}) {
    for (unsigned y : {0u, gy - 1}) {
      for (unsigned x : {0u, gx - 1}) {
        add(x, y, z);
      }
    }
  }
  // mid-edge representatives and interior
  add(gx / 2, 0,      0);       add(gx / 2, gy - 1, 0);
  add(0,      gy / 2, 0);       add(gx - 1, gy / 2, 0);
  add(gx / 2, gy / 2, 0);      // interior (z=0 face)
  if (gz > 1) {
    add(gx / 2, gy / 2, gz / 2); // interior z
    add(gx / 2, gy / 2, gz - 1);
  }
  return std::vector<std::tuple<unsigned,unsigned,unsigned>>(seen.begin(), seen.end());
}


void handler(int sig) {
  void *array[50];
  size_t size;

  fprintf(stderr, "Error: signal %d:\n", sig);
  fflush(stderr);

  // get void*'s for all entries on the stack
  size = backtrace(array, 50);

  // print out all the frames to stderr
  fprintf(stdout, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDOUT_FILENO);
  fflush(stdout);
  exit(1);
}

int main(int argc, const char **argv) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  gpgpu_context *m_gpgpu_context = new gpgpu_context();
  trace_config tconfig;
  option_parser_t opp;
  gpgpu_sim *m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig, opp);
  m_gpgpu_sim->init();
  
  m_gpgpu_sim->m_current_omp_scheduler = omp_sched_t::omp_sched_static;
  omp_set_schedule(m_gpgpu_sim->m_current_omp_scheduler, 1);

  trace_parser tracer(tconfig.get_traces_filename(), tconfig.get_is_extra_traces_enabled(), m_gpgpu_sim->getShaderCoreConfig()->filter_first_kernel_id, m_gpgpu_sim->getShaderCoreConfig()->filter_last_kernel_id); // MOD. Improved tracer

  tconfig.parse_config(); 

  m_gpgpu_sim->parse_extra_trace_info(tracer.get_extra_trace_info_filename(), tconfig.get_is_extra_traces_enabled()); // MOD. Improved tracer

  // for each kernel
  // load file
  // parse and create kernel info
  // launch
  // while loop till the end of the end kernel execution
  // prints stats
  bool concurrent_kernel_sm =  m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
  unsigned window_size = concurrent_kernel_sm ? m_gpgpu_sim->get_config().get_max_concurrent_kernel() : 1;
  assert(window_size > 0);
  std::vector<trace_command> commandlist = tracer.parse_commandlist_file();
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t*> kernels_info;
  kernels_info.reserve(window_size);

  bool active = false;
  bool sim_cycles = false;
  unsigned finished_kernel_uid = 0;
  bool is_cta_max_hit = false;
  bool can_continue_simulation = true;

  unsigned i = 0;
  signal(SIGSEGV, handler);
  signal(SIGILL, handler);
  signal(SIGABRT, handler);
  signal(SIGTERM, handler);
  signal(SIGFPE, handler);
  while (true) {
    if((i >= commandlist.size() && kernels_info.empty()) || !can_continue_simulation){
      break;
    }
    //gulp up as many commands as possible - either cpu_gpu_mem_copy 
    //or kernel_launch - until the vector "kernels_info" has reached
    //the window_size or we have read every command from commandlist
    while (kernels_info.size() < window_size && i < commandlist.size()) {
      trace_kernel_info_t *kernel_info = NULL;
      if (commandlist[i].m_type == command_type::cpu_gpu_mem_copy) {
        size_t addre, Bcount;
        tracer.parse_memcpy_info(commandlist[i].command_string, addre, Bcount);
        std::cout << "launching memcpy command : " << commandlist[i].command_string << std::endl;
        m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
        i++;
      } else if (commandlist[i].m_type == command_type::kernel_launch) {
        // Read trace header info for window_size number of kernels
        kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string, m_gpgpu_sim->get_extra_trace_info());
        kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
        kernels_info.push_back(kernel_info);
        std::cout << "Header info loaded for kernel command : " << commandlist[i].command_string << std::endl;
        i++;
      }
      else{
        //unsupported commands will fail the simulation
        assert(0 && "Undefined Command");
      }
    }

    // Launch all kernels within window that are on a stream that isn't already running
    for (auto k : kernels_info) {
      bool stream_busy = false;
      for (auto s: busy_streams) {
        if (s == k->get_cuda_stream_id())
          stream_busy = true;
      }
      if (!stream_busy && m_gpgpu_sim->can_start_kernel() && !k->was_launched()) {
        std::cout << "launching kernel name: " << k->get_name() << " uid: " << k->get_uid() << std::endl;
        m_gpgpu_sim->launch(k);
        k->set_launched();
        busy_streams.push_back(k->get_cuda_stream_id());
      }
    }
    active = m_gpgpu_sim->active();
    sim_cycles = false;
    finished_kernel_uid = 0;
    is_cta_max_hit = false;
    
    while (true) {
      if (active) {
        m_gpgpu_sim->cycle();
      }

      if (active) {
          sim_cycles = true;
          m_gpgpu_sim->deadlock_check();
      } else if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
          m_gpgpu_context->the_gpgpusim->g_stream_manager
              ->stop_all_running_kernels();
          is_cta_max_hit = true;
      }
      active = m_gpgpu_sim->active();
      finished_kernel_uid = m_gpgpu_sim->finished_kernel();

      if (!active || finished_kernel_uid || is_cta_max_hit) {
        break;
      }
    }

    // cleanup finished kernel
    unsigned finished_kernel_total_ctas = 0;
    unsigned finished_kernel_k_reps = 0;
    if ( (finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit()
        || !m_gpgpu_sim->active()) && !kernels_info.empty() ) {
      trace_kernel_info_t* k = NULL;
      float finished_kernel_cta_weight = 1.0f;
      for (unsigned j = 0; j < kernels_info.size(); j++) {
        k = kernels_info.at(j);
        if (k->get_uid() == finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit()
            || !m_gpgpu_sim->active()) {
          for (std::size_t l = 0; l < busy_streams.size(); l++) {
            if (busy_streams.at(l) == k->get_cuda_stream_id()) {
              busy_streams.erase(busy_streams.begin()+l);
              break;
            }
          }
          finished_kernel_cta_weight = k->get_trace_info()->cta_sampling_weight;
          // Capture original grid info before kernel_finalizer frees trace_info.
          finished_kernel_total_ctas = k->get_trace_info()->grid_dim_x
                                     * k->get_trace_info()->grid_dim_y
                                     * k->get_trace_info()->grid_dim_z;
          finished_kernel_k_reps = (unsigned)k->get_trace_info()->sampled_ctas.size();
          tracer.kernel_finalizer(k->get_trace_info());
          delete k->entry();
          delete k;
          kernels_info.erase(kernels_info.begin()+j);
          if (!m_gpgpu_sim->cycle_insn_cta_max_hit() && m_gpgpu_sim->active())
            break;
        }
      }
      assert(k);
      m_gpgpu_sim->set_cta_sampling_weight(finished_kernel_cta_weight);
      m_gpgpu_sim->print_stats();
    }

    if (sim_cycles) {
      // CTA sampling: capture per-kernel pressure signals before update_stats
      // resets the per-kernel counters. Always logged so the values are
      // available even when sampling is off (sanity check + roofline view).
      pressure_signals_t ps;
      m_gpgpu_sim->compute_kernel_pressure_signals(ps);
      kernel_class kc = classify_kernel(ps, tconfig);
      // K reps: when sampling is off, treat the kernel as if K = total_ctas
      // so the selector returns total_ctas and "no expansion needed".
      unsigned k_reps = finished_kernel_k_reps
          ? finished_kernel_k_reps
          : finished_kernel_total_ctas;
      unsigned total_sms = m_gpgpu_sim->get_config().num_shader();
      unsigned n_sat_est = compute_n_sat_est(ps, k_reps);
      unsigned target_sim_ctas = compute_initial_sim_ctas(
          kc, k_reps, finished_kernel_total_ctas, total_sms, n_sat_est);
      printf("CTA_PRESSURE_SIGNALS:"
             " sim_cycles=%llu sim_insns=%llu ctas_launched=%llu"
             " dram_bytes=%llu dram_reqs=%llu"
             " l2_accesses=%llu l2_misses=%llu l2_miss_rate=%.4f"
             " dram_queue_occupancy_avg=%.4f achieved_bw_ratio=%.4f"
             " peak_dram_bw_bytes_per_cycle=%.2f peak_flops_per_cycle=%.2f"
             " ridge_point_flop_per_byte=%.4f"
             " kernel_ai=%.4f ridge_ratio=%.4f"
             " class=%s k_reps=%u total_ctas=%u total_sms=%u"
             " n_sat_est=%u target_sim_ctas=%u\n",
             ps.sim_cycles, ps.sim_insns, ps.ctas_launched,
             ps.dram_bytes, ps.dram_reqs,
             ps.l2_accesses, ps.l2_misses, ps.l2_miss_rate,
             ps.dram_queue_occupancy_avg, ps.achieved_bw_ratio,
             ps.peak_dram_bw_bytes_per_cycle, ps.peak_flops_per_cycle,
             ps.ridge_point_flop_per_byte,
             ps.kernel_ai, ps.ridge_ratio,
             kernel_class_name(kc), k_reps, finished_kernel_total_ctas, total_sms,
             n_sat_est, target_sim_ctas);
      fflush(stdout);
      m_gpgpu_sim->update_stats();
      m_gpgpu_context->print_simulation_time();
    }
    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      printf("GPGPU-Sim: ** break due to reaching the maximum cycles (or "
            "instructions) **\n");
      fflush(stdout);
      can_continue_simulation = false;
    }
  }
  option_parser_destroy(opp);
  delete m_gpgpu_sim;
  delete m_gpgpu_context;
  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}


trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser){

  gpgpu_ptx_sim_info info;
  info.smem = kernel_trace_info->shmem;
  info.regs = kernel_trace_info->nregs;
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y, kernel_trace_info->tb_dim_z);

  unsigned gx = kernel_trace_info->grid_dim_x;
  unsigned gy = kernel_trace_info->grid_dim_y;
  unsigned gz = kernel_trace_info->grid_dim_z;
  unsigned total_ctas = gx * gy * gz;

  dim3 gridDim(gx, gy, gz);

  if (config->get_cta_sampling_mode() == 1 && total_ctas > 1) {
    auto sampled = compute_sampled_ctas(gx, gy, gz);
    unsigned k = (unsigned)sampled.size();
    kernel_trace_info->sampled_ctas = sampled;
    kernel_trace_info->sampled_cta_idx = 0;
    kernel_trace_info->cta_sampling_weight = (float)total_ctas / (float)k;
    // set first representative as the starting trace position
    kernel_trace_info->next_tb_to_parse_x = std::get<0>(sampled[0]);
    kernel_trace_info->next_tb_to_parse_y = std::get<1>(sampled[0]);
    kernel_trace_info->next_tb_to_parse_z = std::get<2>(sampled[0]);
    // tell the simulator the kernel has only K CTAs so dispatch terminates after K
    gridDim = dim3(k, 1, 1);
    std::cout << "CTA sampling: kernel " << kernel_trace_info->kernel_name
              << " grid=(" << gx << "," << gy << "," << gz << ") total=" << total_ctas
              << " sampled=" << k << " weight=" << kernel_trace_info->cta_sampling_weight
              << "\n";
  }

  trace_function_info *function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_trace_info->kernel_name.c_str());
  trace_kernel_info_t *kernel_info =
      new trace_kernel_info_t(gridDim, blockDim, function_info,
    		  parser, config, kernel_trace_info);
  kernel_info->function_unique_id = kernel_trace_info->func_unique_id;
  kernel_info->is_captured_from_binary = kernel_trace_info->is_cap_from_binary;
  return kernel_info;
}

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           trace_config *m_config, option_parser_t &opp) {
  srand(1);
  print_splash();

  opp = option_parser_create();

  m_gpgpu_context->ptx_reg_options(opp);
  m_gpgpu_context->func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config =
      new gpgpu_sim_config(m_gpgpu_context);
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->reg_options(
      opp); // register GPU microrachitecture options
  m_config->reg_options(opp);
  m_gpgpu_context->the_gpgpusim->g_trace_config = m_config;

  option_parser_cmdline(opp, argc, argv); // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->set_custom_options(true); // MOD. General parse options

  m_gpgpu_context->the_gpgpusim->g_the_gpu = new trace_gpgpu_sim(
      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config), m_gpgpu_context);

  m_gpgpu_context->the_gpgpusim->g_stream_manager =
      new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                         m_gpgpu_context->func_sim->g_cuda_launch_blocking);

  m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);
  
  return m_gpgpu_context->the_gpgpusim->g_the_gpu;
}
