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
#include <map>
#include <math.h>
#include <random>
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

// Three-way classifier: ridge_ratio + DRAM-side pressure proxies + issue-stage
// memory-stall fraction. "Pressure-high" beats ridge_ratio: a kernel reading
// at >=60% peak BW or stalling on l1c-wait above the mstall threshold is
// effectively memory-bound regardless of where its AI puts it on roofline.
// Note: K-rep pressure signals are biased low (under-fills caches/queues), so
// borderline cases are biased toward MEMORY by design.
static kernel_class classify_kernel(const pressure_signals_t& s,
                                    const trace_config& tc) {
  bool memory_pressure_high =
      (s.achieved_bw_ratio        >= tc.get_cta_sampling_pressure_bw())     ||
      (s.dram_queue_occupancy_avg >= tc.get_cta_sampling_pressure_queue())  ||
      (s.mem_stall_frac           >= tc.get_cta_sampling_pressure_mstall());
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

// One pilot iteration's measurement, captured for the concurrency-throughput
// fit. Per-SM throughput grows nonlinearly with concurrent CTAs/SM (memory
// latency hiding), and the constant-throughput cycle estimator under-counts
// that growth. After 2+ iterations at distinct CTAs/SM densities we fit a
// log model to extrapolate to the full-grid density.
struct pilot_iter_obs_t {
  unsigned long long sampled_cycles;  // gpu_sim_cycle for this iter
  unsigned           sampled_ctas;    // ps.ctas_launched
  unsigned           active_sms;      // min(sampled_ctas, total_sms)
};

// Per-kernel state carried across iterations of the adaptive pilot loop.
struct pilot_state_t {
  unsigned iter;            // 0 = K-rep run; 1+ = expansion / doubling iters
  unsigned target_sim_ctas; // current iter's target
  unsigned k_reps;          // K (constant per-kernel)
  unsigned total_ctas;      // grid total (constant per-kernel)
  unsigned total_sms;       // hardware SM count (constant per kernel)
  pilot_stats_snapshot_t snapshot;  // gpu_tot_* before this iter's contribution
  pressure_signals_t prev_signals;  // for stability check
  bool prev_signals_valid;
  // Log of every iteration's measurement (rejected and accepted alike), so
  // the concurrency-throughput fit at accept time can use all data points.
  std::vector<pilot_iter_obs_t> history;
};

// Mutate kernel_trace_info's sampling fields for the next pilot iteration.
// target == 0 leaves sampling off (mode 0); target == K uses the K-rep list;
// target > K stratified-shuffle replicates.
static void update_sampling_on_trace_info(kernel_trace_t* ti, unsigned target, unsigned seed);

// Fit T(N) = a + b*log(N+1) to per-SM throughput vs CTAs-per-SM, where
// throughput is sampled_ctas/(active_sms * sampled_cycles) and N is
// sampled_ctas/active_sms (CTAs per SM in that iteration). Per-SM
// throughput grows nonlinearly with concurrent CTAs/SM (memory-latency
// hiding), and a constant-throughput extrapolation under-counts that
// growth on memory-stall-bound kernels (notably backprop's 1D-grid
// kernels where K-rep collapses to 1-CTA-per-SM samples).
//
// Requires at least 2 distinct CTAs-per-SM densities in the history to
// fit a slope. Returns false (no fit) when fewer densities are present
// or the fit yields a non-monotone slope. When multiple iterations share
// a density the per-SM throughputs are averaged before fitting.
static bool pilot_fit_log_throughput(const std::vector<pilot_iter_obs_t>& history,
                                     double& fit_a, double& fit_b) {
  fit_a = 0.0; fit_b = 0.0;
  if (history.size() < 2) return false;
  // Aggregate by CTAs-per-SM (rounded to nearest integer). Average the
  // throughputs at each density.
  std::map<unsigned, std::pair<double, unsigned>> agg;  // N -> (sum_T, count)
  for (const auto& obs : history) {
    if (obs.active_sms == 0 || obs.sampled_cycles == 0) continue;
    unsigned ctas_per_sm = obs.sampled_ctas / obs.active_sms;
    if (ctas_per_sm == 0) ctas_per_sm = 1;
    double t = (double)obs.sampled_ctas
             / ((double)obs.active_sms * (double)obs.sampled_cycles);
    auto& entry = agg[ctas_per_sm];
    entry.first += t;
    entry.second += 1;
  }
  if (agg.size() < 2) return false;
  // Least-squares fit on (log(N+1), T_avg) pairs.
  std::vector<std::pair<double,double>> pts;
  pts.reserve(agg.size());
  for (auto& kv : agg) {
    double x = std::log((double)kv.first + 1.0);
    double y = kv.second.first / (double)kv.second.second;
    pts.push_back({x, y});
  }
  double n = (double)pts.size();
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (auto& p : pts) { sx += p.first; sy += p.second;
                        sxx += p.first*p.first; sxy += p.first*p.second; }
  double denom = n*sxx - sx*sx;
  if (std::fabs(denom) < 1e-30) return false;
  fit_b = (n*sxy - sx*sy) / denom;
  fit_a = (sy - fit_b*sx) / n;
  // Reject non-monotone (decreasing) fits -- those mean per-SM throughput
  // dropped as we added concurrency, which is a noisy signal we shouldn't
  // extrapolate from.
  if (fit_b <= 0.0) return false;
  return true;
}

// Decide whether to accept this iteration's run.
//   iter 0 (K-rep): accept if classifier says COMPUTE AND the K-rep sample
//     is large enough to be representative. The K-rep heuristic collapses
//     to ~3 CTAs on 1D grids (e.g. backprop's 256x1x1 kernels), and at that
//     size a 1-CTA-per-SM sample misses the steady-state SM concurrency that
//     the full grid sees -- with high mem_stall_frac (memory-latency-bound),
//     concurrent CTAs hide latency and raise per-SM throughput in ways the
//     1-CTA-per-SM sample can't measure. Force at least one pilot expansion
//     in that regime so the next iteration has more representative
//     concurrency.
//   iter > 0: accept on stop_bw_target reached, or stable deltas, or hit the
//     doublings cap, or sim_ctas == total_ctas.
static bool pilot_decide_accept(const pilot_state_t& pst, const pressure_signals_t& ps,
                                kernel_class kc, const trace_config& tc) {
  if (pst.iter == 0) {
    if (kc != KCLASS_COMPUTE) return false;
    // If K-rep happens to cover all CTAs, expansion is a no-op -- accept.
    if (ps.ctas_launched >= pst.total_ctas) return true;
    // Force expansion when projecting the sample to the full grid would
    // cross a large concurrency gap. The cycle estimator's
    // constant-per-SM-throughput assumption breaks badly when the full grid
    // runs at >>1 CTA/SM while the sample runs at 1 CTA/SM -- per-SM
    // throughput grows nonlinearly with CTAs/SM (memory + functional-unit
    // latency hiding). Trigger on EITHER:
    //   (a) high projection ratio -- full_ctas_per_sm > 4 * sampled_per_sm.
    //       Threshold catches backprop (6.4×), nn (23×), and other large-
    //       grid kernels but spares hotspot (1.6×) and lud-like grids
    //       where the full-grid concurrency is close to the sample's.
    //   (b) memory-stall signal -- legacy heuristic for grids that aren't
    //       large but show high mem_stall_frac, where even modest
    //       concurrency increases shift throughput materially.
    unsigned tsms = pst.total_sms ? pst.total_sms : 1;
    unsigned sampled_active = std::min((unsigned)ps.ctas_launched, tsms);
    if (sampled_active == 0) sampled_active = 1;
    unsigned full_active = std::min(pst.total_ctas, tsms);
    if (full_active == 0) full_active = 1;
    double sampled_per_sm = (double)ps.ctas_launched / (double)sampled_active;
    double full_per_sm    = (double)pst.total_ctas / (double)full_active;
    if (full_per_sm > 4.0 * sampled_per_sm) return false;
    bool undersized = (ps.ctas_launched * 2u < pst.total_ctas)
                   && (ps.ctas_launched < 8u);
    if (undersized && ps.mem_stall_frac > 0.10) return false;
    return true;
  }
  if (pst.target_sim_ctas >= pst.total_ctas) return true;
  if (pst.iter > tc.get_cta_sampling_pilot_max_doublings()) return true;
  if (ps.achieved_bw_ratio >= tc.get_cta_sampling_pilot_stop_bw_target()) return true;
  if (pst.prev_signals_valid) {
    double tol = tc.get_cta_sampling_pilot_stop_delta();
    double bw_delta = std::fabs(ps.achieved_bw_ratio -
                                pst.prev_signals.achieved_bw_ratio);
    double ipc_prev = pst.prev_signals.sim_cycles
        ? (double)pst.prev_signals.sim_insns / (double)pst.prev_signals.sim_cycles : 0.0;
    double ipc_curr = ps.sim_cycles
        ? (double)ps.sim_insns / (double)ps.sim_cycles : 0.0;
    double ipc_delta = (ipc_prev > 0.0)
        ? std::fabs(ipc_curr - ipc_prev) / ipc_prev : 0.0;
    if (bw_delta < tol && ipc_delta < tol) return true;
  }
  return false;
}

// Choose sim_ctas for the next iteration. iter 0 -> classifier-driven initial
// target; subsequent iters double the previous target.
static unsigned pilot_next_target(const pilot_state_t& pst,
                                  const pressure_signals_t& ps, kernel_class kc,
                                  unsigned total_sms) {
  if (pst.iter == 0) {
    unsigned n_sat = compute_n_sat_est(ps, pst.k_reps);
    return compute_initial_sim_ctas(kc, pst.k_reps, pst.total_ctas, total_sms, n_sat);
  }
  unsigned next = pst.target_sim_ctas * 2u;
  if (next > pst.total_ctas) next = pst.total_ctas;
  return next;
}

// Stratified replication of K representative CTAs into a target-slot list.
// Each rep gets ceil(target/K) or floor(target/K) slots; the order is then
// deterministically Fisher-Yates shuffled (seeded) to break alignment with
// L2-set / DRAM-bank striping that pure round-robin replication would create.
static std::vector<std::tuple<unsigned,unsigned,unsigned>>
expand_sampled_ctas(const std::vector<std::tuple<unsigned,unsigned,unsigned>>& reps,
                    unsigned target_sim_ctas, unsigned seed) {
  unsigned K = (unsigned)reps.size();
  if (K == 0 || target_sim_ctas <= K) return reps;
  std::vector<std::tuple<unsigned,unsigned,unsigned>> out;
  out.reserve(target_sim_ctas);
  unsigned per_rep = target_sim_ctas / K;
  unsigned remainder = target_sim_ctas % K;
  for (unsigned i = 0; i < K; ++i) {
    unsigned count = per_rep + (i < remainder ? 1u : 0u);
    for (unsigned c = 0; c < count; ++c) out.push_back(reps[i]);
  }
  std::mt19937 rng(seed);
  for (size_t i = out.size() - 1; i > 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    std::swap(out[i], out[dist(rng)]);
  }
  return out;
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


// Implementation: rebuild sampled_ctas for a target sim_ctas count.
static void update_sampling_on_trace_info(kernel_trace_t* ti, unsigned target,
                                          unsigned seed) {
  unsigned gx = ti->grid_dim_x, gy = ti->grid_dim_y, gz = ti->grid_dim_z;
  unsigned total_ctas = gx * gy * gz;
  if (total_ctas <= 1) return;
  auto reps = compute_sampled_ctas(gx, gy, gz);
  unsigned k = (unsigned)reps.size();
  std::vector<std::tuple<unsigned,unsigned,unsigned>> sampled;
  if (target > k && target <= total_ctas) {
    sampled = expand_sampled_ctas(reps, target, seed);
  } else {
    sampled = reps;
  }
  unsigned n_slots = (unsigned)sampled.size();
  ti->sampled_ctas = sampled;
  ti->sampled_cta_idx = 0;
  ti->cta_sampling_weight = (float)total_ctas / (float)n_slots;
  ti->next_tb_to_parse_x = std::get<0>(sampled[0]);
  ti->next_tb_to_parse_y = std::get<1>(sampled[0]);
  ti->next_tb_to_parse_z = std::get<2>(sampled[0]);
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

  // Adaptive pilot loop state: keyed by kernel_trace_t* (stable across pilot
  // iterations of the same kernel; the trace_t is only freed on accept).
  std::map<kernel_trace_t*, pilot_state_t> pilot_states;
  bool pilot_loop_enabled = (tconfig.get_cta_sampling_mode() == 1) &&
                            (tconfig.get_cta_sampling_pilot_max_doublings() > 0);
  if (pilot_loop_enabled && window_size != 1) {
    fprintf(stderr,
            "CTA sampling: pilot loop disabled because window_size=%u (>1); "
            "concurrent kernels are not supported by the pilot loop.\n",
            window_size);
    pilot_loop_enabled = false;
  }

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
        if (pilot_loop_enabled) {
          // Initialize pilot state for this kernel. Iter 0 = K-rep run.
          pilot_state_t& pst = pilot_states[kernel_trace_info];
          pst.iter = 0;
          pst.k_reps = (unsigned)kernel_trace_info->sampled_ctas.size();
          pst.total_ctas = kernel_trace_info->grid_dim_x
                         * kernel_trace_info->grid_dim_y
                         * kernel_trace_info->grid_dim_z;
          pst.target_sim_ctas = pst.k_reps;
          pst.total_sms = m_gpgpu_sim->get_config().num_shader();
          pst.prev_signals_valid = false;
        }
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
        // Capture gpu_tot_* baseline before this iteration's contribution so a
        // rejected pilot iteration can be rolled back. snapshot_pressure_signals
        // (DRAM/L2 baseline for delta-extraction) is taken inside launch().
        if (pilot_loop_enabled) {
          auto it = pilot_states.find(k->get_trace_info());
          if (it != pilot_states.end()) m_gpgpu_sim->pilot_snapshot(it->second.snapshot);
        }
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

    // cleanup finished kernel — but capture pressure signals first so the
    // pilot loop can re-launch this kernel before kernel_finalizer wipes the
    // trace_info. The classification + selector logging happens for both
    // accepted and pilot iterations so the trace is visible end-to-end.
    unsigned finished_kernel_total_ctas = 0;
    unsigned finished_kernel_k_reps = 0;
    bool kernel_finished_this_iter = false;
    trace_kernel_info_t* finished_k = nullptr;
    unsigned finished_idx_in_kernels_info = 0;
    if ( (finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit()
        || !m_gpgpu_sim->active()) && !kernels_info.empty() ) {
      for (unsigned j = 0; j < kernels_info.size(); j++) {
        trace_kernel_info_t* k = kernels_info.at(j);
        if (k->get_uid() == finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit()
            || !m_gpgpu_sim->active()) {
          finished_k = k;
          finished_idx_in_kernels_info = j;
          finished_kernel_total_ctas = k->get_trace_info()->grid_dim_x
                                     * k->get_trace_info()->grid_dim_y
                                     * k->get_trace_info()->grid_dim_z;
          finished_kernel_k_reps = (unsigned)k->get_trace_info()->sampled_ctas.size();
          kernel_finished_this_iter = true;
          if (!m_gpgpu_sim->cycle_insn_cta_max_hit() && m_gpgpu_sim->active())
            break;
        }
      }
    }

    if (sim_cycles && kernel_finished_this_iter) {
      pressure_signals_t ps;
      m_gpgpu_sim->compute_kernel_pressure_signals(
          ps,
          tconfig.get_cta_sampling_ai_w_dp(),
          tconfig.get_cta_sampling_ai_w_tc(),
          tconfig.get_cta_sampling_ai_w_sfu());
      kernel_class kc = classify_kernel(ps, tconfig);
      unsigned k_reps = finished_kernel_k_reps
          ? finished_kernel_k_reps
          : finished_kernel_total_ctas;
      unsigned total_sms = m_gpgpu_sim->get_config().num_shader();
      unsigned n_sat_est = compute_n_sat_est(ps, k_reps);
      unsigned target_sim_ctas = compute_initial_sim_ctas(
          kc, k_reps, finished_kernel_total_ctas, total_sms, n_sat_est);

      // Pilot decide
      bool accept = true;
      pilot_state_t* pst = nullptr;
      if (pilot_loop_enabled) {
        auto it = pilot_states.find(finished_k->get_trace_info());
        if (it != pilot_states.end()) {
          pst = &it->second;
          // Record this iteration's measurement for the concurrency-throughput
          // fit. Done before the accept decision so a forced-expansion path
          // still preserves the data point.
          pilot_iter_obs_t obs;
          obs.sampled_cycles = ps.sim_cycles;
          obs.sampled_ctas   = (unsigned)ps.ctas_launched;
          obs.active_sms     = std::min((unsigned)ps.ctas_launched, total_sms);
          if (obs.active_sms == 0) obs.active_sms = 1;
          pst->history.push_back(obs);
          accept = pilot_decide_accept(*pst, ps, kc, tconfig);
        }
      }

      printf("CTA_PRESSURE_SIGNALS:"
             " sim_cycles=%llu sim_insns=%llu ctas_launched=%llu"
             " dram_bytes=%llu dram_reqs=%llu"
             " l2_accesses=%llu l2_misses=%llu l2_miss_rate=%.4f"
             " dram_queue_occupancy_avg=%.4f achieved_bw_ratio=%.4f"
             " peak_dram_bw_bytes_per_cycle=%.2f peak_flops_per_cycle=%.2f"
             " ridge_point_flop_per_byte=%.4f"
             " n_fp_decoded=%llu n_int_decoded=%llu"
             " n_dp_acc=%.0f n_tc_acc=%.0f n_sfu_acc=%.0f"
             " n_load_insn=%llu n_store_insn=%llu"
             " compute_ops=%.0f mem_ops=%llu"
             " issue_eval_cycles=%llu stall_l1c_cycles=%llu stall_total_cycles=%llu"
             " mem_stall_frac=%.4f total_stall_frac=%.4f"
             " kernel_ai=%.4f ridge_ratio=%.4f"
             " class=%s k_reps=%u total_ctas=%u total_sms=%u"
             " n_sat_est=%u target_sim_ctas=%u"
             " pilot_iter=%u pilot_accepted=%d\n",
             ps.sim_cycles, ps.sim_insns, ps.ctas_launched,
             ps.dram_bytes, ps.dram_reqs,
             ps.l2_accesses, ps.l2_misses, ps.l2_miss_rate,
             ps.dram_queue_occupancy_avg, ps.achieved_bw_ratio,
             ps.peak_dram_bw_bytes_per_cycle, ps.peak_flops_per_cycle,
             ps.ridge_point_flop_per_byte,
             ps.n_fp_decoded, ps.n_int_decoded,
             ps.n_dp_acc, ps.n_tc_acc, ps.n_sfu_acc,
             ps.n_load_insn, ps.n_store_insn,
             ps.compute_ops, ps.mem_ops,
             ps.issue_eval_cycles, ps.stall_l1c_cycles, ps.stall_total_cycles,
             ps.mem_stall_frac, ps.total_stall_frac,
             ps.kernel_ai, ps.ridge_ratio,
             kernel_class_name(kc), k_reps, finished_kernel_total_ctas, total_sms,
             n_sat_est, target_sim_ctas,
             pst ? pst->iter : 0u, accept ? 1 : 0);
      fflush(stdout);

      if (accept) {
        // Final iteration — finalize the kernel and update gpu_tot_* normally.
        for (std::size_t l = 0; l < busy_streams.size(); l++) {
          if (busy_streams.at(l) == finished_k->get_cuda_stream_id()) {
            busy_streams.erase(busy_streams.begin()+l);
            break;
          }
        }
        float weight = finished_k->get_trace_info()->cta_sampling_weight;
        kernel_trace_t* trace_info = finished_k->get_trace_info();
        m_gpgpu_sim->set_cta_sampling_weight(weight);
        // Phase B: hand the pilot-final wave context to gpu-sim so print_stats
        // can project a whole-grid cycle estimate from the sampled-wave clock.
        last_kernel_wave_info_t wave_info;
        wave_info.kernel_class_int = (int)kc;
        wave_info.target_sim_ctas  = target_sim_ctas;
        wave_info.sampled_ctas     = (unsigned)ps.ctas_launched;
        wave_info.total_ctas       = finished_kernel_total_ctas;
        wave_info.max_cta_per_core = (unsigned)m_gpgpu_sim->max_cta_per_core();
        wave_info.total_sms        = total_sms;
        // Concurrency-throughput fit: if the pilot ran multiple iterations at
        // distinct CTAs-per-SM densities, fit a log model so the estimator
        // can extrapolate per-SM throughput to the full-grid density rather
        // than assuming it stays constant.
        if (pst) {
          double a = 0.0, b = 0.0;
          if (pilot_fit_log_throughput(pst->history, a, b)) {
            wave_info.has_log_fit = true;
            wave_info.log_fit_a   = a;
            wave_info.log_fit_b   = b;
          }
        }
        m_gpgpu_sim->set_last_kernel_wave_info(wave_info);
        m_gpgpu_sim->print_stats();
        if (pst) pilot_states.erase(trace_info);
        tracer.kernel_finalizer(trace_info);
        delete finished_k->entry();
        delete finished_k;
        kernels_info.erase(kernels_info.begin() + finished_idx_in_kernels_info);
        m_gpgpu_sim->update_stats();
        m_gpgpu_context->print_simulation_time();
      } else {
        // Pilot reject — pick next target, rebuild kernel_info from the same
        // trace_info, replace in kernels_info, restore gpu_tot_* and reset
        // per-kernel state so the next iteration starts clean.
        unsigned next_target = pilot_next_target(*pst, ps, kc, total_sms);
        pst->iter += 1;
        pst->target_sim_ctas = next_target;
        pst->prev_signals = ps;
        pst->prev_signals_valid = true;
        kernel_trace_t* trace_info = finished_k->get_trace_info();
        update_sampling_on_trace_info(trace_info, next_target,
                                      tconfig.get_cta_sampling_seed());
        trace_kernel_info_t* new_k = create_kernel_info(
            trace_info, m_gpgpu_context, &tconfig, &tracer);
        kernels_info[finished_idx_in_kernels_info] = new_k;
        for (std::size_t l = 0; l < busy_streams.size(); l++) {
          if (busy_streams.at(l) == finished_k->get_cuda_stream_id()) {
            busy_streams.erase(busy_streams.begin()+l);
            break;
          }
        }
        delete finished_k->entry();
        delete finished_k;
        // update_stats moves gpu_sim_* into gpu_tot_*; pilot_restore then
        // undoes that increment so the rejected iteration has zero net effect
        // on cross-kernel totals.
        m_gpgpu_sim->update_stats();
        m_gpgpu_sim->pilot_restore(pst->snapshot);
        m_gpgpu_context->print_simulation_time();
      }
    } else if (sim_cycles) {
      // No finished kernel found this iteration but we did sim cycles
      // (e.g. cta_max_hit). Preserve original behavior: just update stats.
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
    std::vector<std::tuple<unsigned,unsigned,unsigned>> sampled;
    unsigned k;
    if (!kernel_trace_info->sampled_ctas.empty()) {
      // Pilot loop has already populated sampled_ctas via
      // update_sampling_on_trace_info; honor it instead of recomputing.
      sampled = kernel_trace_info->sampled_ctas;
      // K (the original heuristic count) is recoverable from
      // total_ctas / cta_sampling_weight, but we only need it for logging.
      k = kernel_trace_info->cta_sampling_weight > 0.0f
          ? (unsigned)((float)total_ctas / kernel_trace_info->cta_sampling_weight)
          : (unsigned)sampled.size();
    } else {
      auto reps = compute_sampled_ctas(gx, gy, gz);
      k = (unsigned)reps.size();
      unsigned target = config->get_cta_sampling_target_ctas();
      if (target > k && target <= total_ctas) {
        sampled = expand_sampled_ctas(reps, target, config->get_cta_sampling_seed());
      } else {
        sampled = reps;
      }
    }
    unsigned n_slots = (unsigned)sampled.size();
    kernel_trace_info->sampled_ctas = sampled;
    kernel_trace_info->sampled_cta_idx = 0;
    // Weight scales recorded per-CTA stats up to the full grid. With n_slots
    // simulated CTAs, the simulator records n_slots * per_cta_avg work, so
    // weight = total_ctas / n_slots gives back total_ctas * per_cta_avg.
    kernel_trace_info->cta_sampling_weight = (float)total_ctas / (float)n_slots;
    kernel_trace_info->next_tb_to_parse_x = std::get<0>(sampled[0]);
    kernel_trace_info->next_tb_to_parse_y = std::get<1>(sampled[0]);
    kernel_trace_info->next_tb_to_parse_z = std::get<2>(sampled[0]);
    gridDim = dim3(n_slots, 1, 1);
    std::cout << "CTA sampling: kernel " << kernel_trace_info->kernel_name
              << " grid=(" << gx << "," << gy << "," << gz << ") total=" << total_ctas
              << " sampled=" << k
              << (n_slots != k ? " (expanded to " + std::to_string(n_slots) + ")" : "")
              << " weight=" << kernel_trace_info->cta_sampling_weight
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
