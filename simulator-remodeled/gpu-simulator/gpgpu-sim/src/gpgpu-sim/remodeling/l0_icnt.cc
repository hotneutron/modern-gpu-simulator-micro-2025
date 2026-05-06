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

#include "l0_icnt.h"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <set>
#include <unordered_set>

#include "../gpu-cache.h"
#include "../gpu-sim.h"
#include "../shader_core_wrapper.h"

namespace {
constexpr unsigned int k_instruction_region_size_in_blocks = 8;

struct instruction_region_tracking_key {
    unsigned int sm_id;
    unsigned int unique_function_id;
    new_addr_type region_base;

    bool operator==(const instruction_region_tracking_key &other) const {
        return sm_id == other.sm_id &&
               unique_function_id == other.unique_function_id &&
               region_base == other.region_base;
    }
};

struct instruction_region_tracking_key_hash {
    std::size_t operator()(const instruction_region_tracking_key &key) const {
        return std::hash<unsigned int>()(key.sm_id) ^
               (std::hash<unsigned int>()(key.unique_function_id) << 1) ^
               (std::hash<new_addr_type>()(key.region_base) << 2);
    }
};

struct instruction_region_tracking_entry {
    unsigned long long total_l1i_miss_count = 0;
    unsigned long long demand_l1i_miss_count = 0;
    unsigned long long prefetch_l1i_miss_count = 0;
    unsigned long long late_l1i_miss_count = 0;
    unsigned long long last_observed_cycle = 0;
    std::set<unsigned int> observed_warp_ids;
};

std::unordered_map<instruction_region_tracking_key,
                   instruction_region_tracking_entry,
                   instruction_region_tracking_key_hash>
    g_instruction_region_tracking;
std::mutex g_instruction_region_tracking_mutex;
std::unordered_set<new_addr_type> g_instruction_region_prewarm_resident_blocks;
std::mutex g_instruction_region_prewarm_resident_blocks_mutex;

new_addr_type get_instruction_region_base(new_addr_type block_addr,
                                          unsigned int line_size) {
    new_addr_type region_size =
        static_cast<new_addr_type>(line_size) *
        static_cast<new_addr_type>(k_instruction_region_size_in_blocks);
    return block_addr - (block_addr % region_size);
}

void mark_instruction_region_prewarm_resident(new_addr_type block_addr) {
    std::lock_guard<std::mutex> lock(
        g_instruction_region_prewarm_resident_blocks_mutex);
    g_instruction_region_prewarm_resident_blocks.insert(block_addr);
}

bool is_instruction_region_prewarm_resident(new_addr_type block_addr) {
    std::lock_guard<std::mutex> lock(
        g_instruction_region_prewarm_resident_blocks_mutex);
    return g_instruction_region_prewarm_resident_blocks.count(block_addr) > 0;
}

bool erase_instruction_region_prewarm_resident(new_addr_type block_addr) {
    std::lock_guard<std::mutex> lock(
        g_instruction_region_prewarm_resident_blocks_mutex);
    return g_instruction_region_prewarm_resident_blocks.erase(block_addr) > 0;
}
}


unsigned num_bytes_cache_req(unsigned line_size, address_type pc) {
    assert((line_size % 8) == 0);
    unsigned nbytes = line_size / 8;
    unsigned offset_in_block = pc & (line_size - 1);
    if ((offset_in_block + nbytes) > line_size) {
        nbytes = (line_size - offset_in_block);
    }
    return nbytes;
}

address_type get_pc_of_request(address_type pc) {
    return pc - PROGRAM_MEM_START;
}

void record_instruction_region_l1i_miss_observation(
    unsigned int sm_id, unsigned int line_size, new_addr_type block_addr,
    unsigned int unique_function_id, unsigned int warp_id, bool is_prefetch,
    unsigned long long cycle) {
    std::lock_guard<std::mutex> lock(g_instruction_region_tracking_mutex);
    instruction_region_tracking_key key{
        sm_id, unique_function_id,
        get_instruction_region_base(block_addr, line_size)};
    auto &entry = g_instruction_region_tracking[key];
    entry.total_l1i_miss_count++;
    if (is_prefetch) {
        entry.prefetch_l1i_miss_count++;
    } else {
        entry.demand_l1i_miss_count++;
    }
    entry.last_observed_cycle = cycle;
    entry.observed_warp_ids.insert(warp_id);
}

void record_instruction_region_late_miss_observation(
    unsigned int sm_id, unsigned int line_size, new_addr_type block_addr,
    unsigned int unique_function_id, unsigned int warp_id,
    unsigned long long cycle) {
    std::lock_guard<std::mutex> lock(g_instruction_region_tracking_mutex);
    instruction_region_tracking_key key{
        sm_id, unique_function_id,
        get_instruction_region_base(block_addr, line_size)};
    auto &entry = g_instruction_region_tracking[key];
    entry.late_l1i_miss_count++;
    entry.last_observed_cycle = cycle;
    entry.observed_warp_ids.insert(warp_id);
}

std::string get_instruction_region_top_late_miss_regions(unsigned int top_n) {
    std::vector<std::pair<instruction_region_tracking_key,
                          instruction_region_tracking_entry>> entries;
    {
        std::lock_guard<std::mutex> lock(g_instruction_region_tracking_mutex);
        if (g_instruction_region_tracking.empty()) {
            return "none";
        }
        entries.assign(g_instruction_region_tracking.begin(),
                       g_instruction_region_tracking.end());
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto &lhs, const auto &rhs) {
                  if (lhs.second.late_l1i_miss_count != rhs.second.late_l1i_miss_count) {
                      return lhs.second.late_l1i_miss_count >
                             rhs.second.late_l1i_miss_count;
                  }
                  return lhs.second.total_l1i_miss_count >
                         rhs.second.total_l1i_miss_count;
              });
    std::ostringstream oss;
    unsigned int count = 0;
    for (const auto &entry : entries) {
        if (entry.second.late_l1i_miss_count == 0) {
            continue;
        }
        if (count++ > 0) {
            oss << ";";
        }
        oss << "sm=" << entry.first.sm_id
            << ":ufid=" << entry.first.unique_function_id
            << ":region=0x" << std::hex << entry.first.region_base << std::dec
            << ":late=" << entry.second.late_l1i_miss_count
            << ":miss=" << entry.second.total_l1i_miss_count
            << ":demand=" << entry.second.demand_l1i_miss_count
            << ":prefetch=" << entry.second.prefetch_l1i_miss_count
            << ":warps=" << entry.second.observed_warp_ids.size();
        if (count >= top_n) {
            break;
        }
    }
    return count == 0 ? "none" : oss.str();
}

std::string get_instruction_region_top_miss_regions(unsigned int top_n) {
    std::vector<std::pair<instruction_region_tracking_key,
                          instruction_region_tracking_entry>> entries;
    {
        std::lock_guard<std::mutex> lock(g_instruction_region_tracking_mutex);
        if (g_instruction_region_tracking.empty()) {
            return "none";
        }
        entries.assign(g_instruction_region_tracking.begin(),
                       g_instruction_region_tracking.end());
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto &lhs, const auto &rhs) {
                  if (lhs.second.total_l1i_miss_count != rhs.second.total_l1i_miss_count) {
                      return lhs.second.total_l1i_miss_count >
                             rhs.second.total_l1i_miss_count;
                  }
                  return lhs.second.late_l1i_miss_count >
                         rhs.second.late_l1i_miss_count;
              });
    std::ostringstream oss;
    unsigned int count = 0;
    for (const auto &entry : entries) {
        if (entry.second.total_l1i_miss_count == 0) {
            continue;
        }
        if (count++ > 0) {
            oss << ";";
        }
        oss << "sm=" << entry.first.sm_id
            << ":ufid=" << entry.first.unique_function_id
            << ":region=0x" << std::hex << entry.first.region_base << std::dec
            << ":miss=" << entry.second.total_l1i_miss_count
            << ":late=" << entry.second.late_l1i_miss_count
            << ":demand=" << entry.second.demand_l1i_miss_count
            << ":prefetch=" << entry.second.prefetch_l1i_miss_count
            << ":warps=" << entry.second.observed_warp_ids.size();
        if (count >= top_n) {
            break;
        }
    }
    return count == 0 ? "none" : oss.str();
}

std::vector<new_addr_type> get_instruction_regions_to_prewarm(
    unsigned int unique_function_id, unsigned long long min_late_miss_count,
    unsigned int min_observed_warps, unsigned int max_regions) {
    struct aggregated_instruction_region_entry {
        new_addr_type region_base;
        unsigned long long total_late_l1i_miss_count = 0;
        unsigned long long total_l1i_miss_count = 0;
        std::set<unsigned int> observed_warp_ids;
    };
    std::unordered_map<new_addr_type, aggregated_instruction_region_entry>
        aggregated_entries;
    {
        std::lock_guard<std::mutex> lock(g_instruction_region_tracking_mutex);
        for (const auto &entry : g_instruction_region_tracking) {
            if (entry.first.unique_function_id != unique_function_id) {
                continue;
            }
            auto &aggregated_entry = aggregated_entries[entry.first.region_base];
            aggregated_entry.region_base = entry.first.region_base;
            aggregated_entry.total_late_l1i_miss_count +=
                entry.second.late_l1i_miss_count;
            aggregated_entry.total_l1i_miss_count +=
                entry.second.total_l1i_miss_count;
            aggregated_entry.observed_warp_ids.insert(
                entry.second.observed_warp_ids.begin(),
                entry.second.observed_warp_ids.end());
        }
    }
    std::vector<aggregated_instruction_region_entry> entries;
    for (const auto &entry : aggregated_entries) {
        if (entry.second.total_late_l1i_miss_count < min_late_miss_count) {
            continue;
        }
        if (entry.second.observed_warp_ids.size() < min_observed_warps) {
            continue;
        }
        entries.push_back(entry.second);
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto &lhs, const auto &rhs) {
                  if (lhs.total_late_l1i_miss_count != rhs.total_late_l1i_miss_count) {
                      return lhs.total_late_l1i_miss_count >
                             rhs.total_late_l1i_miss_count;
                  }
                  if (lhs.total_l1i_miss_count != rhs.total_l1i_miss_count) {
                      return lhs.total_l1i_miss_count >
                             rhs.total_l1i_miss_count;
                  }
                  return lhs.region_base < rhs.region_base;
              });
    std::vector<new_addr_type> region_bases;
    unsigned int limit = std::min<unsigned int>(max_regions, entries.size());
    for (unsigned int i = 0; i < limit; ++i) {
        region_bases.push_back(entries[i].region_base);
    }
    return region_bases;
}

unsigned int get_instruction_region_size_in_blocks() {
    return k_instruction_region_size_in_blocks;
}

struct instruction_region_prewarm_debug_counters {
    unsigned long long issued = 0;
    unsigned long long blocked_memport_full = 0;
    unsigned long long l1i_accesses = 0;
    unsigned long long l1i_hits = 0;
    unsigned long long l1i_misses = 0;
    unsigned long long l1i_reservation_fails = 0;
    unsigned long long fills = 0;
    unsigned long long useful_demand_hits = 0;
    unsigned long long useful_prefetch_hits = 0;
    unsigned long long lines_evicted_by_demand_miss = 0;
};

instruction_region_prewarm_debug_counters &
get_instruction_region_prewarm_debug_counters_ref() {
    static instruction_region_prewarm_debug_counters counters;
    return counters;
}

std::mutex &get_instruction_region_prewarm_debug_counters_mutex() {
    static std::mutex counters_mutex;
    return counters_mutex;
}

template <typename Func>
void update_instruction_region_prewarm_debug_counters(Func updater) {
    std::lock_guard<std::mutex> lock(
        get_instruction_region_prewarm_debug_counters_mutex());
    updater(get_instruction_region_prewarm_debug_counters_ref());
}

std::string get_instruction_region_prewarm_debug_stats() {
    std::lock_guard<std::mutex> lock(
        get_instruction_region_prewarm_debug_counters_mutex());
    const auto &counters = get_instruction_region_prewarm_debug_counters_ref();
    std::ostringstream oss;
    oss << "issued=" << counters.issued
        << ";blocked_memport_full=" << counters.blocked_memport_full
        << ";l1i_accesses=" << counters.l1i_accesses
        << ";l1i_hits=" << counters.l1i_hits
        << ";l1i_misses=" << counters.l1i_misses
        << ";l1i_reservation_fails=" << counters.l1i_reservation_fails
        << ";fills=" << counters.fills
        << ";useful_demand_hits=" << counters.useful_demand_hits
        << ";useful_prefetch_hits=" << counters.useful_prefetch_hits
        << ";lines_evicted_by_demand_miss="
        << counters.lines_evicted_by_demand_miss;
    return oss.str();
}

L0_icnt::L0_icnt(read_only_cache *L1, gpgpu_sim *gpu, shader_core_ctx_wrapper* shader, int max_num_L1_reply_ports_allowed, int max_num_L1_request_ports_allowed, int latency_L0_to_L1, int latency_L1_to_L0) {
    m_L1 = L1;
    m_gpu = gpu;
    m_shader = shader;
    m_max_num_L1_reply_ports_allowed = max_num_L1_reply_ports_allowed;
    m_max_num_L1_request_ports_allowed = max_num_L1_request_ports_allowed;
    m_latency_of_L1_to_L0s_icnt_queue = latency_L1_to_L0;
    m_latency_of_L0s_icnt_to_L1_queue = latency_L0_to_L1;

    assert(m_max_num_L1_reply_ports_allowed>0);
    assert(m_max_num_L1_request_ports_allowed>0);
    assert(m_latency_of_L1_to_L0s_icnt_queue>0);
    assert(m_latency_of_L0s_icnt_to_L1_queue>0);
    
    m_icnt_to_L1_queue.resize(m_max_num_L1_request_ports_allowed);
    for(int i = 0; i < m_max_num_L1_request_ports_allowed; i++) {
        m_icnt_to_L1_queue[i].resize(m_latency_of_L0s_icnt_to_L1_queue, nullptr); 
    }
    m_L1_to_icnt_queue.resize(m_max_num_L1_reply_ports_allowed);
    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        m_L1_to_icnt_queue[i].resize(m_latency_of_L1_to_L0s_icnt_queue, nullptr); 
    }

    m_max_size_icnt_L1_TLB_to_cache =1;
}

L0_icnt::~L0_icnt() {
    flush();
}

void L0_icnt::add_L0(read_only_cache *L0) {
    m_L0.push_back(L0);
}

bool L0_icnt::full(unsigned size, bool write) const {
    bool res = true;
    for(int i = 0; i < m_max_num_L1_request_ports_allowed; i++) {
        if(m_icnt_to_L1_queue[i][m_latency_of_L0s_icnt_to_L1_queue-1] == nullptr) {
            res = false;
            break;
        }
    }
    return res;
}

bool L0_icnt::is_L1_to_icnt_queue_full() {
    bool res = true;
    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        if(m_L1_to_icnt_queue[i][m_latency_of_L1_to_L0s_icnt_queue-1] == nullptr) {
            res = false;
            break;
        }
    }
    return res;
}

int L0_icnt::get_available_L1_to_icnt_port_id() {
    int res = -1;
    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        if(m_L1_to_icnt_queue[i][m_latency_of_L1_to_L0s_icnt_queue-1] == nullptr) {
            res = i;
            break;
        }
    }
    assert(res != -1);
    return res;
}

// L0_icnt pushes are execute before L0_icnt cycle
void L0_icnt::push(mem_fetch *mf) {
    bool inserted = false;
    for(int i = 0; i < m_max_num_L1_request_ports_allowed; i++) {
        if(m_icnt_to_L1_queue[i][m_latency_of_L0s_icnt_to_L1_queue-1] == nullptr) {
            m_icnt_to_L1_queue[i][m_latency_of_L0s_icnt_to_L1_queue-1] = mf;
            inserted = true;
            break;
        }
    }
    assert(inserted);

    m_shader->set_subcore_req_fetch_L1I_priority( (mf->get_subcore() + 1) % m_shader->get_num_subcores() );
}

bool L0_icnt::issue_instruction_prewarm(new_addr_type addr,
                                        unsigned int unique_function_id,
                                        unsigned int subcore_id) {
    if(full(m_L1->get_config().get_line_sz(), false)) {
        update_instruction_region_prewarm_debug_counters(
            [](auto &counters) { counters.blocked_memport_full++; });
        return false;
    }
    unsigned int line_size = m_L1->get_config().get_line_sz();
    unsigned int nbytes = num_bytes_cache_req(line_size, addr);
    unsigned int sid = m_shader->get_sid();
    unsigned int tpc = m_gpu->getShaderCoreConfig()->sid_to_cluster(sid);
    mem_access_t acc(INST_ACC_R, addr, nbytes, false, m_gpu->gpgpu_ctx);
    mem_fetch *mf = new mem_fetch(
        acc, NULL, READ_PACKET_SIZE, 0, sid,
        tpc, m_gpu->getMemoryConfig(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
        NULL, NULL, unique_function_id);
    mf->set_subcore(subcore_id);
    mf->set_is_prefetch(true);
    mf->set_is_instruction_region_prewarm(true);
    update_instruction_region_prewarm_debug_counters(
        [](auto &counters) { counters.issued++; });
    push(mf);
    return true;
}

void L0_icnt::cycle() {

    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        if(m_L1_to_icnt_queue[i][0] != nullptr) {
            mem_fetch *mf = m_L1_to_icnt_queue[i][0];
            bool safe_to_pop = true;
            bool not_used = true;
            bool is_prefetch = mf->get_original_mf()->get_is_prefetch();
            bool is_instruction_region_prewarm =
                mf->get_original_mf()->get_is_instruction_region_prewarm();
            unsigned int mf_subcore_id = mf->get_original_mf()->get_subcore();
            mem_access_type type = mf->get_access().get_type();
            if(is_prefetch) {
                mf->set_is_prefetch(true);
                mf->set_stream_buffer_id(mf->get_original_mf()->get_stream_buffer_id());
                mf->set_is_instruction_region_prewarm(is_instruction_region_prewarm);
                if(mf->get_original_mf()->get_prefetch_l1i_fate() == 2) {
                    m_gpu->record_l1i_prefetch_fill(
                        m_L1->get_config().block_addr(mf->get_access_address()));
                }
            }
            if (is_instruction_region_prewarm) {
                mark_instruction_region_prewarm_resident(
                    m_L1->get_config().block_addr(mf->get_access_address()));
                update_instruction_region_prewarm_debug_counters(
                    [](auto &counters) { counters.fills++; });
                m_L1_to_icnt_queue[i][0] = nullptr;
                delete mf->get_original_mf();
                delete mf;
            } else {
                unsigned int cache_id = mf_subcore_id;
                if( (type == CONST_ACC_R) && (cache_id < m_shader->get_num_subcores()) ) {
                    cache_id += m_shader->get_num_subcores();
                }
                assert(cache_id < m_L0.size());
                bool is_mf_for_this_subcore = m_L0[cache_id]->waiting_for_fill(mf->get_original_mf());
                bool is_L0_port_subcore_free = m_L0[cache_id]->fill_port_free();
                if(is_mf_for_this_subcore && !is_L0_port_subcore_free) {
                    safe_to_pop = false;
                }
                if (is_mf_for_this_subcore && is_L0_port_subcore_free) {
                    mf->set_is_filling_L0(true);
                    mf->set_status(IN_L0_FILL_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    m_L0[cache_id]->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    not_used = false;
                }
                if(safe_to_pop) {
                    m_L1_to_icnt_queue[i][0] = nullptr;
                }
                if( (safe_to_pop && not_used) || (safe_to_pop && is_prefetch) ) {
                    delete mf->get_original_mf();
                    delete mf;
                }
            }
        }
        for (int stage = 0; stage < m_latency_of_L1_to_L0s_icnt_queue - 1; stage++) {
            if (m_L1_to_icnt_queue[i][stage] == nullptr) {
                m_L1_to_icnt_queue[i][stage] = m_L1_to_icnt_queue[i][stage + 1];
                m_L1_to_icnt_queue[i][stage + 1] = nullptr;
            }
        }
    }

    bool inserted = true;
    // From TLB Stage to Cache
    for(int i = 0; (i < m_max_num_L1_request_ports_allowed) && !is_L1_to_icnt_queue_full() && m_L1->data_port_free() && inserted && !m_icnt_L1_TLB_to_cache.empty(); i++) {
        mem_fetch *mf = m_icnt_L1_TLB_to_cache.front();
        std::list<cache_event> events;
        enum cache_request_status status = cache_request_status::NOT_INITIALIZED;
        address_type addr_req = mf->get_access_address();
        new_addr_type block_addr = m_L1->get_config().block_addr(addr_req);
        unsigned int set_idx = m_L1->get_config().set_index(block_addr);
        unsigned int victim_index = (unsigned)-1;
        enum cache_request_status probe_status =
            m_L1->get_tag_array()->probe(block_addr, victim_index, mf,
                                         mf->is_write());
        bool victim_valid = false;
        new_addr_type victim_block_addr = 0;
        bool victim_was_prefetch_resident = false;
        bool victim_was_instruction_region_prewarm_resident = false;
        if((probe_status != HIT) && (probe_status != RESERVATION_FAIL)) {
            cache_block_t *victim_block = m_L1->get_tag_array()->get_block(victim_index);
            victim_valid = victim_block->is_valid_line();
            if(victim_valid) {
                victim_block_addr = victim_block->m_block_addr;
                victim_was_prefetch_resident =
                    m_gpu->is_l1i_prefetch_resident(victim_block_addr);
                victim_was_instruction_region_prewarm_resident =
                    is_instruction_region_prewarm_resident(victim_block_addr);
            }
        }
        unsigned nbytes = num_bytes_cache_req(m_gpu->getShaderCoreConfig()->m_L1I_L1_half_C_cache_config.get_line_sz(), addr_req);
        mem_access_t acc(mf->get_access().get_type(), addr_req, nbytes, false, m_gpu->gpgpu_ctx);
        mem_fetch *mf_n = new mem_fetch(acc, NULL /*we don't have an instruction yet*/, READ_PACKET_SIZE,
                mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf->get_mem_config(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf, NULL, mf->get_unique_function_id());
        mf_n->set_is_instruction_region_prewarm(
            mf->get_is_instruction_region_prewarm());
        bool erase_orifinal_mf = false;
        status = m_L1->access((new_addr_type) mf_n->get_access_address(), mf_n, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events, erase_orifinal_mf);
        if(mf->get_is_instruction_region_prewarm()) {
            update_instruction_region_prewarm_debug_counters(
                [](auto &counters) { counters.l1i_accesses++; });
            if(status == HIT) {
                update_instruction_region_prewarm_debug_counters(
                    [](auto &counters) { counters.l1i_hits++; });
            } else if(status == MISS) {
                update_instruction_region_prewarm_debug_counters(
                    [](auto &counters) { counters.l1i_misses++; });
            } else if(status == RESERVATION_FAIL) {
                update_instruction_region_prewarm_debug_counters(
                    [](auto &counters) { counters.l1i_reservation_fails++; });
            }
        } else if(mf->get_is_prefetch()) {
            m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_l1i_accesses"]->increment_with_integer(1);
            if(status == HIT) {
                mf->set_prefetch_l1i_fate(1);
                mf_n->set_prefetch_l1i_fate(1);
                m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_l1i_hits"]->increment_with_integer(1);
                if (is_instruction_region_prewarm_resident(block_addr)) {
                    update_instruction_region_prewarm_debug_counters(
                        [](auto &counters) { counters.useful_prefetch_hits++; });
                }
            } else if(status == MISS) {
                mf->set_prefetch_l1i_fate(2);
                mf_n->set_prefetch_l1i_fate(2);
                m_gpu->record_l1i_prefetch_miss_observation(
                    block_addr, mf->get_unique_function_id(), set_idx,
                    victim_valid, victim_block_addr,
                    victim_was_prefetch_resident, true);
                m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_l1i_misses"]->increment_with_integer(1);
            } else if(status == RESERVATION_FAIL) {
                mf->set_prefetch_l1i_fate(3);
                mf_n->set_prefetch_l1i_fate(3);
                m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_l1i_reservation_fails"]->increment_with_integer(1);
            }
        } else {
            if ((status == HIT) && (mf->get_access().get_type() == INST_ACC_R) &&
                is_instruction_region_prewarm_resident(block_addr)) {
                update_instruction_region_prewarm_debug_counters(
                    [](auto &counters) { counters.useful_demand_hits++; });
            }
            if((status == MISS) && victim_valid && victim_was_prefetch_resident) {
                m_gpu->record_l1i_prefetched_line_evicted(victim_block_addr, false);
            }
            if((status == MISS) && victim_valid &&
               victim_was_instruction_region_prewarm_resident &&
               erase_instruction_region_prewarm_resident(victim_block_addr)) {
                update_instruction_region_prewarm_debug_counters(
                    [](auto &counters) {
                        counters.lines_evicted_by_demand_miss++;
                    });
            }
        }
        if ((status == MISS) && (mf->get_access().get_type() == INST_ACC_R)) {
            record_instruction_region_l1i_miss_observation(
                m_shader->get_sid(), m_L1->get_config().get_line_sz(),
                block_addr, mf->get_unique_function_id(), mf->get_wid(),
                mf->get_is_prefetch(),
                m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
        }

        if(status == RESERVATION_FAIL) {
            inserted = false;
            delete mf_n;
        }else if(status == MISS) {
            inserted = true;
        }else if(status == HIT) {
            inserted = true;
            int port = get_available_L1_to_icnt_port_id();
            m_L1_to_icnt_queue[port][m_latency_of_L1_to_L0s_icnt_queue - 1] = mf_n;
        }else {
            inserted = false;
            delete mf_n;
            assert(0);
        }

        if(erase_orifinal_mf) {
            delete mf;
        }
        if(inserted) {
           m_icnt_L1_TLB_to_cache.pop();
        }
    }

    // ICNT to IL1 TLB
    for(int i = 0; (i < m_max_num_L1_request_ports_allowed); i++) {
        if(m_icnt_to_L1_queue[i][0] != nullptr) {
            inserted = false;
            mem_fetch *mf = m_icnt_to_L1_queue[i][0];
            cache_request_status tlb_acc = HIT;
            if((tlb_acc == HIT)) {
                if(m_icnt_L1_TLB_to_cache.size() < m_max_size_icnt_L1_TLB_to_cache) {
                    m_icnt_L1_TLB_to_cache.push(mf);
                    inserted = true;
                }
            }else if((tlb_acc == MISS) || (tlb_acc == MSHR_HIT)) {
                inserted = true;
            }else {
                assert(tlb_acc == RESERVATION_FAIL);
                inserted = false;
            }

            if(inserted) {
                m_icnt_to_L1_queue[i][0] = nullptr;
            }
        }
        for (int stage = 0; stage < m_latency_of_L0s_icnt_to_L1_queue - 1; stage++) {
            if (m_icnt_to_L1_queue[i][stage] == nullptr) {
                m_icnt_to_L1_queue[i][stage] = m_icnt_to_L1_queue[i][stage + 1];
                m_icnt_to_L1_queue[i][stage + 1] = nullptr;
            }
        }
    }
    
    // L1 to ICNT
    for(int i = 0; (i < m_max_num_L1_reply_ports_allowed) && m_L1->access_ready() &&  m_L1->data_port_free(); i++) {
        if(m_L1_to_icnt_queue[i][m_latency_of_L1_to_L0s_icnt_queue - 1] == nullptr) {
            mem_fetch *mf = m_L1->next_access();
            mf->set_reply();
            m_L1_to_icnt_queue[i][m_latency_of_L1_to_L0s_icnt_queue - 1] = mf;
        }
    }
}

void L0_icnt::flush() {
    for(int i = 0; i < m_max_num_L1_request_ports_allowed; i++) {
        for(int j = 0; j < m_latency_of_L0s_icnt_to_L1_queue; j++) {
            if(m_icnt_to_L1_queue[i][j] != nullptr) {
                delete m_icnt_to_L1_queue[i][j];
                m_icnt_to_L1_queue[i][j] = nullptr;
            }
        }
    }
    
    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        for(int j = 0; j < m_latency_of_L1_to_L0s_icnt_queue; j++) {
            if(m_L1_to_icnt_queue[i][j] != nullptr) {
                if(m_L1_to_icnt_queue[i][j]->get_original_mf() != nullptr) {
                    delete m_L1_to_icnt_queue[i][j]->get_original_mf();
                }
                delete m_L1_to_icnt_queue[i][j];
                m_L1_to_icnt_queue[i][j] = nullptr;
            }
        }
    }
}
