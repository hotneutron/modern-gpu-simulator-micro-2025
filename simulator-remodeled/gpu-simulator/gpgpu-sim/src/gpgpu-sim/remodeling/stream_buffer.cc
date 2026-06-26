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

#include "stream_buffer.h"

#include "first_level_instruction_cache.h"
#include "sm.h"
#include "../gpu-cache.h"
#include "../gpu-sim.h"

void record_instruction_region_late_miss_observation(
    unsigned int sm_id, unsigned int line_size, new_addr_type block_addr,
    unsigned int unique_function_id, unsigned int warp_id,
    unsigned long long cycle);

single_stream_buffer::single_stream_buffer(unsigned int sb_id, int core_id, bool is_prefetching_enabled,
        unsigned int subcore_id, SM * sm, first_level_instruction_cache *cache,
        unsigned int max_size, unsigned int line_size, mem_fetch_interface *memport) {
    m_sm_id = core_id;
    m_is_enabled = is_prefetching_enabled;
    m_subcore_id = subcore_id;
    m_sm = sm;
    m_cache = cache;
    m_max_size = max_size;
    m_line_size = line_size;
    m_memport = memport;
    gpu_cycle_hit = std::numeric_limits<unsigned long long>::max();
    m_stream_buffer_id = sb_id;
    m_next_addr_to_prefetch = std::numeric_limits<new_addr_type>::max();
    m_current_unique_function_id = std::numeric_limits<unsigned int>::max();
    m_first_sm_warp_id_reserved = 0;
}

single_stream_buffer::~single_stream_buffer() {
    flush();
}

SM* single_stream_buffer::get_sm() { 
    return m_sm;
}

bool single_stream_buffer::is_active() {
    return m_is_currently_prefetching;
}

bool single_stream_buffer::is_idle_for_replacement() {
    return !m_is_currently_prefetching && m_queue_ordered_prefetches.empty();
}

bool single_stream_buffer::is_inactive_with_entries() {
    return !m_is_currently_prefetching && !m_queue_ordered_prefetches.empty();
}

unsigned long long single_stream_buffer::get_gpu_cycle_hit() {
    return gpu_cycle_hit;
}

first_level_instruction_cache *single_stream_buffer::get_cache() { 
    return m_cache;
}

bool single_stream_buffer::is_full() {
    return m_queue_ordered_prefetches.size() == m_max_size;
}

bool single_stream_buffer::is_a_pending_request(new_addr_type addr, unsigned long long gpu_cycle) {
    auto it = m_all_prefetches.find(addr);
    bool hit = it != m_all_prefetches.end();
    return hit;
}

bool single_stream_buffer::is_hit(new_addr_type addr, unsigned long long gpu_cycle) {
    assert(m_queue_ordered_prefetches.size() == m_all_prefetches.size());
    bool hit = false;
    if(!m_queue_ordered_prefetches.empty()) {
        new_addr_type first_addr = m_queue_ordered_prefetches.front();
        hit = addr == first_addr;
    }
    if(hit) {
        gpu_cycle_hit = gpu_cycle;
    }
    return hit;
}

bool single_stream_buffer::is_found_requested_addr_deeper_than_head(
    new_addr_type addr, bool &is_ready) const {
    is_ready = false;
    if(m_queue_ordered_prefetches.empty()) {
        return false;
    }
    auto it = m_all_prefetches.find(addr);
    if(it == m_all_prefetches.end()) {
        return false;
    }
    if(m_queue_ordered_prefetches.front() == addr) {
        return false;
    }
    is_ready = it->second.is_ready;
    return true;
}

void single_stream_buffer::flush() {
    m_all_prefetches.clear();
    while (!m_queue_ordered_prefetches.empty()) {
        m_queue_ordered_prefetches.pop();
    }
    m_is_currently_prefetching = false;
    m_next_addr_to_prefetch = std::numeric_limits<new_addr_type>::max();
    m_current_unique_function_id = std::numeric_limits<unsigned int>::max();
}

void single_stream_buffer::set_new_stream(new_addr_type addr, unsigned int unique_function_id, unsigned long long gpu_cycle, unsigned int warp_id, bool is_early_trigger_candidate) {
    SM *sm = get_sm();
    bool safe_to_set = true;
    bool replacing_active_stream = false;
    if(!m_queue_ordered_prefetches.empty()) {
        new_addr_type top_addr = m_queue_ordered_prefetches.front();
        auto it = m_all_prefetches.find(top_addr);
        assert(it != m_all_prefetches.end());
        safe_to_set = !it->second.is_request_to_cache;
    }
    if(m_is_currently_prefetching || !m_queue_ordered_prefetches.empty()) {
        replacing_active_stream = true;
    }
    if(is_early_trigger_candidate) {
        sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_set_new_stream_calls"]->increment_with_integer(1);
    }
    if(safe_to_set && ((addr != m_next_addr_to_prefetch) || (unique_function_id != m_current_unique_function_id)) ) {
        sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_new_stream_accepted"]->increment_with_integer(1);
        if(is_early_trigger_candidate) {
            sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_set_new_stream_accepted"]->increment_with_integer(1);
        }
        if(replacing_active_stream) {
            sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_new_stream_flush_replaced_active_stream"]->increment_with_integer(1);
            if(is_early_trigger_candidate) {
                sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_set_new_stream_accepted_replace_active"]->increment_with_integer(1);
            }
        } else if(is_early_trigger_candidate) {
            sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_set_new_stream_accepted_idle_buffer"]->increment_with_integer(1);
        }
        flush();
        m_is_currently_prefetching = true;
        m_next_addr_to_prefetch = addr;
        m_current_unique_function_id = unique_function_id;
        m_first_sm_warp_id_reserved = warp_id;
    } else if(!safe_to_set) {
        sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_new_stream_rejected_head_waiting_for_cache"]->increment_with_integer(1);
        if(is_early_trigger_candidate) {
            sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_set_new_stream_rejected_head_waiting_for_cache"]->increment_with_integer(1);
        }
    } else if(is_early_trigger_candidate) {
        sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_set_new_stream_redundant_same_stream"]->increment_with_integer(1);
        sm->m_sm_stats.m_stats_map["total_sum_l0i_stream_buffer_early_trigger_redundant_same_stream_queue_depth"]->increment_with_integer(m_queue_ordered_prefetches.size());
        if(m_is_currently_prefetching) {
            sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_redundant_same_stream_active"]->increment_with_integer(1);
        } else {
            sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_redundant_same_stream_inactive"]->increment_with_integer(1);
        }
        auto it_target = m_all_prefetches.find(addr);
        if(it_target != m_all_prefetches.end()) {
            sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_redundant_same_stream_target_allocated"]->increment_with_integer(1);
            if(it_target->second.is_ready) {
                sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_redundant_same_stream_target_ready"]->increment_with_integer(1);
            } else {
                sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_redundant_same_stream_target_not_ready"]->increment_with_integer(1);
            }
        } else {
            sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_redundant_same_stream_target_not_allocated"]->increment_with_integer(1);
        }
    }
}

bool single_stream_buffer::fill(mem_fetch *mf, unsigned time) {
    bool res = false;
    new_addr_type addr = mf->get_addr();
    auto it = m_all_prefetches.find(addr);
    if(it != m_all_prefetches.end()) {
        get_sm()->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_fill_matched"]->increment_with_integer(1);
        it->second.m_prefetch_l1i_fate = mf->get_prefetch_l1i_fate();
        if(it->second.m_prefetch_l1i_fate == 2) {
            get_sm()->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_l1i_miss_fill_matched"]->increment_with_integer(1);
        }
        it->second.is_ready = true;
        it->second.m_ready_cycle = get_sm()->get_current_gpu_cycle();
        if (it->second.m_has_first_demand) {
            unsigned long long ready_cycle = get_sm()->get_current_gpu_cycle();
            if (ready_cycle > it->second.m_first_demand_cycle) {
                unsigned long long stall_cycles = ready_cycle - it->second.m_first_demand_cycle;
                get_sm()->m_sm_stats.m_stats_map["total_num_l0i_sb_head_demand_arrived_before_ready"]->increment_with_integer(1);
                get_sm()->m_sm_stats.m_stats_map["total_sum_cycles_l0i_sb_demand_wait_for_ready"]->increment_with_integer(stall_cycles);
                if(it->second.m_prefetch_l1i_fate == 2) {
                    get_sm()->m_sm_stats.m_stats_map["total_num_l0i_sb_l1i_miss_head_demand_arrived_before_ready"]->increment_with_integer(1);
                    record_instruction_region_late_miss_observation(
                        get_sm()->get_sid(), get_cache()->get_config().get_line_sz(),
                        addr, it->second.unique_function_id, it->second.sm_warp_id,
                        ready_cycle);
                }
            } else {
                get_sm()->m_sm_stats.m_stats_map["total_num_l0i_sb_head_demand_arrived_after_ready"]->increment_with_integer(1);
                if(it->second.m_prefetch_l1i_fate == 2) {
                    get_sm()->m_sm_stats.m_stats_map["total_num_l0i_sb_l1i_miss_head_demand_arrived_after_ready"]->increment_with_integer(1);
                }
            }
        }
    }else {
        get_sm()->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_fill_orphaned"]->increment_with_integer(1);
        if(mf->get_prefetch_l1i_fate() == 2) {
            get_sm()->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_l1i_miss_fill_orphaned"]->increment_with_integer(1);
        }
        res = true;
    }
    return res;
}

bool single_stream_buffer::send_to_cache() {
    bool can_continue_send_to_cache = true;
    if(!m_queue_ordered_prefetches.empty()) {
        new_addr_type addr = m_queue_ordered_prefetches.front();
        auto it = m_all_prefetches.find(addr);
        assert(it != m_all_prefetches.end());
        if(it->second.is_ready && it->second.is_request_to_cache) {
            get_sm()->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_send_to_cache_attempts"]->increment_with_integer(1);
            can_continue_send_to_cache = false;
            bool safe_to_pop = m_cache->fill_from_stream_buffer(addr, m_sm->get_current_gpu_cycle(), it->second);
            if(safe_to_pop) {
                get_sm()->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_send_to_cache_full_service"]->increment_with_integer(1);
                m_queue_ordered_prefetches.pop();
                m_all_prefetches.erase(addr);
            } else {
                get_sm()->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_send_to_cache_partial_service"]->increment_with_integer(1);
            }
        }
    }
    return can_continue_send_to_cache;
}

bool single_stream_buffer::has_ready_requested_head() const {
    if(m_queue_ordered_prefetches.empty()) {
        return false;
    }
    new_addr_type addr = m_queue_ordered_prefetches.front();
    auto it = m_all_prefetches.find(addr);
    assert(it != m_all_prefetches.end());
    return it->second.is_ready && it->second.is_request_to_cache;
}

// L1I eager-promote (Option B). Promote the head entry into the L1I tag array if
// it is ready and has NOT yet been demanded. Guards:
//  - is_ready == true (data actually returned into the stream buffer)
//  - is_request_to_cache == false (no demand has claimed it yet; the demand path
//    owns those, and double-driving would race the L0I response)
//  - waiting_warp_ids_and_its_addrs empty (Risk B: never strand a waiter)
//  - fill-port available (Risk: Option B defers rather than dropping)
// Returns true iff a promote was performed (head popped).
bool single_stream_buffer::try_eager_promote_head() {
    if(!m_is_enabled) return false;
    if(m_queue_ordered_prefetches.empty()) return false;
    first_level_instruction_cache *cache = get_cache();
    if(!cache->eager_promote_enabled()) return false;

    new_addr_type addr = m_queue_ordered_prefetches.front();
    auto it = m_all_prefetches.find(addr);
    assert(it != m_all_prefetches.end());
    prefetch_element &elem = it->second;

    if(!elem.is_ready) return false;             // data not back yet
    if(elem.is_request_to_cache) return false;   // demand owns this; leave it
    SM *sm = get_sm();
    if(!elem.waiting_warp_ids_and_its_addrs.empty() ||
       !elem.waiting_addrs_of_the_block.empty()) {  // Risk B guard
        sm->m_sm_stats.m_stats_map["total_num_l0i_sb_eager_promote_skipped_has_waiter"]->increment_with_integer(1);
        return false;
    }
    if(cache->is_eager_promote_blocked_by_port()) {  // Option B: defer
        sm->m_sm_stats.m_stats_map["total_num_l0i_sb_eager_promote_skipped_fill_port_busy"]->increment_with_integer(1);
        return false;
    }

    bool promoted = cache->promote_prefetch_to_cache(addr, elem);
    if(promoted) {
        m_queue_ordered_prefetches.pop();
        m_all_prefetches.erase(addr);
    } else {
        // probe-skip (already cached / misaligned): drop the SB entry too, since
        // the line is (or will be) serviced by the cache directly.
        m_queue_ordered_prefetches.pop();
        m_all_prefetches.erase(addr);
    }
    return promoted;
}

bool single_stream_buffer::classify_waiting_requested_head(new_addr_type addr,
                                                           bool &is_ready) const {
    if(m_queue_ordered_prefetches.empty()) {
        return false;
    }
    new_addr_type head_addr = m_queue_ordered_prefetches.front();
    if(head_addr != addr) {
        return false;
    }
    auto it = m_all_prefetches.find(addr);
    if(it == m_all_prefetches.end()) {
        return false;
    }
    is_ready = it->second.is_ready;
    return true;
}

void single_stream_buffer::do_prefetch() {
    bool continue_prefetching = false;
    if(m_is_enabled && m_is_currently_prefetching ) {
        continue_prefetching = true;
        if(!m_memport->full(m_line_size, false) && !is_full()) {
            first_level_instruction_cache* sh_cache = get_cache();
            SM* sm = get_sm();
            unsigned int cache_idx;
            std::list<cache_event> events;
            unsigned long long gpu_cycle = sm->get_current_gpu_cycle();
            unsigned int nbytes = m_line_size;
            mem_access_t acc(INST_ACC_R, m_next_addr_to_prefetch, nbytes, false,
                            sm->get_gpu()->gpgpu_ctx);
            mem_fetch *mf =
            new mem_fetch(acc, NULL /*we don't have an instruction yet*/,
                            READ_PACKET_SIZE, m_first_sm_warp_id_reserved, sm->get_sid(),
                            sm->get_tpc_id(), sm->get_memory_config(),
                            gpu_cycle, NULL, NULL, m_current_unique_function_id);
            mf->set_subcore(m_subcore_id);
            mf->set_is_prefetch(true);
            mf->set_stream_buffer_id(m_stream_buffer_id);
            cache_request_status status = sh_cache->get_tag_array()->probe(m_next_addr_to_prefetch, cache_idx, mf, mf->is_write());
            new_addr_type mshr_addr = sh_cache->get_config().mshr_addr(mf->get_addr());
            bool mshr_hit = sh_cache->get_mshr().probe(mshr_addr);
            if((status != HIT) && (status != RESERVATION_FAIL) && !mshr_hit) {
                sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_issued"]->increment_with_integer(1);
                m_all_prefetches[m_next_addr_to_prefetch] = prefetch_element(m_first_sm_warp_id_reserved, m_current_unique_function_id, false, false);
                m_all_prefetches[m_next_addr_to_prefetch].m_prefetch_issue_cycle = gpu_cycle;
                m_queue_ordered_prefetches.push(m_next_addr_to_prefetch);
                m_memport->push(mf);
                m_next_addr_to_prefetch += m_line_size;
            }else{
                continue_prefetching = false;
                if(status == HIT) {
                    sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_stopped_tag_hit"]->increment_with_integer(1);
                } else if(status == RESERVATION_FAIL) {
                    sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_stopped_reservation_fail"]->increment_with_integer(1);
                } else if(mshr_hit) {
                    sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_stopped_mshr_hit"]->increment_with_integer(1);
                }
                delete mf;
            }
        } else if(m_memport->full(m_line_size, false)) {
            get_sm()->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_blocked_memport_full"]->increment_with_integer(1);
        } else if(is_full()) {
            get_sm()->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_prefetch_blocked_sb_full"]->increment_with_integer(1);
        }
    }
    m_is_currently_prefetching = continue_prefetching;
}


void single_stream_buffer::set_waiting_fill_in_cache(new_addr_type base_addr, new_addr_type request_addr, unsigned int warp_id) {
    auto it = m_all_prefetches.find(base_addr);
    assert(it != m_all_prefetches.end());
    it->second.is_request_to_cache = true;
    if (!it->second.m_has_first_demand) {
        it->second.m_has_first_demand = true;
        it->second.m_first_demand_cycle = get_sm()->get_current_gpu_cycle();
        unsigned long long lead_time = it->second.m_first_demand_cycle - it->second.m_prefetch_issue_cycle;
        get_sm()->m_sm_stats.m_stats_map["total_num_l0i_sb_head_first_demand_events"]->increment_with_integer(1);
        get_sm()->m_sm_stats.m_stats_map["total_sum_cycles_l0i_sb_prefetch_issue_to_first_demand"]->increment_with_integer(lead_time);
        if(!it->second.is_ready) {
            get_sm()->m_sm_stats.m_stats_map["total_num_l0i_sb_head_first_demand_before_ready_issue_age_samples"]->increment_with_integer(1);
            get_sm()->m_sm_stats.m_stats_map["total_sum_cycles_l0i_sb_prefetch_issue_to_first_demand_before_ready"]->increment_with_integer(lead_time);
        }
    }
    it->second.waiting_addrs_of_the_block.insert(request_addr);
    auto it_wid = it->second.waiting_warp_ids_and_its_addrs.find(warp_id);
    if(it_wid == it->second.waiting_warp_ids_and_its_addrs.end()) {
        it->second.waiting_warp_ids_and_its_addrs[warp_id] = std::set<new_addr_type>();
    }
    it->second.waiting_warp_ids_and_its_addrs[warp_id].insert(request_addr);
}




multiple_stream_buffers::multiple_stream_buffers(int core_id, bool is_prefetching_enabled,
    unsigned int subcore_id, SM *sm, first_level_instruction_cache *cache,
    unsigned int max_size_per_stream_buffer, unsigned int line_size, unsigned int num_stream_buffers,
    unsigned int max_num_prefetches_per_cycle, mem_fetch_interface *memport) {
    m_sm_id = core_id;
    m_is_enabled = is_prefetching_enabled;
    m_subcore_id = subcore_id;
    m_sm = sm;
    m_max_size_per_stream_buffer = max_size_per_stream_buffer;
    m_line_size = line_size;
    m_cache = cache;
    m_num_stream_buffers = num_stream_buffers;
    m_next_sb_send_request = 0;
    m_max_num_prefetches_per_cycle = max_num_prefetches_per_cycle;
    m_memport = memport;
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        m_stream_buffers.push_back(new single_stream_buffer(i, core_id, is_prefetching_enabled, subcore_id, sm, cache, max_size_per_stream_buffer, line_size, memport));
    }
}

multiple_stream_buffers::~multiple_stream_buffers() {
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        delete m_stream_buffers[i];
    }
    m_stream_buffers.clear();
}

stream_buffer_search_result multiple_stream_buffers::search(new_addr_type base_addr_request, new_addr_type base_addr_prefetch, unsigned long long gpu_cycle) {
    stream_buffer_search_result result;
    unsigned int id_sb_hit_prefetch = std::numeric_limits<unsigned int>::max();
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        if(m_stream_buffers[i]->is_hit(base_addr_request, gpu_cycle)) {
            result.is_hit_requested_addr = true;
            result.stream_buffer_id = i;
            break;
        } else if(m_stream_buffers[i]->is_hit(base_addr_prefetch, gpu_cycle)) {
            id_sb_hit_prefetch = i;
        }
        if(!result.is_found_requested_addr_deeper_than_head) {
            bool deeper_requested_addr_ready = false;
            if(m_stream_buffers[i]->is_found_requested_addr_deeper_than_head(
                   base_addr_request, deeper_requested_addr_ready)) {
                result.is_found_requested_addr_deeper_than_head = true;
                result.is_found_requested_addr_deeper_than_head_ready =
                    deeper_requested_addr_ready;
            }
        }
    }
    if(!result.is_hit_requested_addr && (id_sb_hit_prefetch != std::numeric_limits<unsigned int>::max())) {
        result.is_hit_prefetch_addr = true;
        result.stream_buffer_id = id_sb_hit_prefetch;
    }
    return result;
}

void multiple_stream_buffers::set_new_stream(new_addr_type addr, unsigned int unique_function_id, unsigned long long gpu_cycle, unsigned int warp_id, bool is_early_trigger_candidate) {
    unsigned long long far_gpu_cycle_hit = std::numeric_limits<unsigned long long>::max();
    unsigned int idx_buffer = 0;
    bool selected_inactive_buffer = false;
    bool saw_inactive_with_entries = false;
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        if(m_stream_buffers[i]->is_idle_for_replacement()) {
            idx_buffer = i;
            selected_inactive_buffer = true;
            break;
        }else {
            if(m_stream_buffers[i]->is_inactive_with_entries()) {
                saw_inactive_with_entries = true;
            }
            unsigned long long gpu_cycle_hit = m_stream_buffers[i]->get_gpu_cycle_hit();
            if(gpu_cycle_hit < far_gpu_cycle_hit) {
                far_gpu_cycle_hit = gpu_cycle_hit;
                idx_buffer = i;
            }
        }
    }
    if(saw_inactive_with_entries) {
        m_sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_new_stream_selection_saw_inactive_nonempty_buffer"]->increment_with_integer(1);
    }
    if(selected_inactive_buffer) {
        m_sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_new_stream_selection_selected_idle_buffer"]->increment_with_integer(1);
    } else {
        m_sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_new_stream_selection_selected_active_buffer"]->increment_with_integer(1);
        if(saw_inactive_with_entries) {
            m_sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_new_stream_selection_selected_active_while_inactive_nonempty_exists"]->increment_with_integer(1);
        }
    }
    if(is_early_trigger_candidate) {
        if(selected_inactive_buffer) {
            m_sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_selected_inactive_buffer"]->increment_with_integer(1);
        } else {
            m_sm->m_sm_stats.m_stats_map["total_num_l0i_stream_buffer_early_trigger_selected_active_buffer"]->increment_with_integer(1);
        }
    }
    m_stream_buffers[idx_buffer]->set_new_stream(addr, unique_function_id, gpu_cycle, warp_id, is_early_trigger_candidate);
}

void multiple_stream_buffers::cycle(bool can_sb_send_to_cache) {
    bool can_send_to_cache = can_sb_send_to_cache;
    if(!can_send_to_cache && has_ready_requested_head()) {
        m_sm->m_sm_stats.m_stats_map["total_num_cycles_l0i_stream_buffer_ready_head_blocked_by_response_slot"]->increment_with_integer(1);
    }
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        for(unsigned int j = 0; j < m_max_num_prefetches_per_cycle; j++) {
            m_stream_buffers[i]->do_prefetch();
        }
        if(can_send_to_cache) {
            can_send_to_cache = m_stream_buffers[m_next_sb_send_request]->send_to_cache();
            m_next_sb_send_request = (m_next_sb_send_request + 1) % m_num_stream_buffers;
        }  
    }
}

void multiple_stream_buffers::eager_promote_cycle() {
    if(!m_is_enabled) return;
    // Best-effort: try to promote one ready/not-demanded head per stream buffer.
    // single_stream_buffer::try_eager_promote_head() internally gates on the
    // cache fill port (Option B), so at most one promote actually consumes the
    // port per cycle; the rest see the port busy and defer.
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        m_stream_buffers[i]->try_eager_promote_head();
    }
}

bool multiple_stream_buffers::has_ready_requested_head() const {
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        if(m_stream_buffers[i]->has_ready_requested_head()) {
            return true;
        }
    }
    return false;
}

bool multiple_stream_buffers::classify_waiting_requested_head(new_addr_type base_addr,
                                                              bool &is_ready) const {
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        if(m_stream_buffers[i]->classify_waiting_requested_head(base_addr,
                                                                is_ready)) {
            return true;
        }
    }
    return false;
}

bool multiple_stream_buffers::fill(mem_fetch *mf, unsigned time) {
    return m_stream_buffers[mf->get_stream_buffer_id()]->fill(mf, time);
}

void multiple_stream_buffers::set_waiting_fill_in_cache(unsigned int stream_buffer_id, new_addr_type base_addr, new_addr_type request_addr, unsigned int warp_id) {
    m_stream_buffers[stream_buffer_id]->set_waiting_fill_in_cache(base_addr, request_addr, warp_id);
}

bool multiple_stream_buffers::is_already_allocated(new_addr_type addr, unsigned long long gpu_cycle, unsigned int stream_buffer_id) {
    return m_stream_buffers[stream_buffer_id]->is_a_pending_request(addr, gpu_cycle);
}

void multiple_stream_buffers::flush() {
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        m_stream_buffers[i]->flush();
    }
}
