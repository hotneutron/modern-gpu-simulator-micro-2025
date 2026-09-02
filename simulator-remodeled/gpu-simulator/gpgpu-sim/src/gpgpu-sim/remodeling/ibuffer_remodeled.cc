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


#include "ibuffer_remodeled.h"
#include "subcore.h"
#include "../../cuda-sim/cuda-sim.h"
#include "../shader.h"
#include "../../../../trace-driven/trace_driven.h"

IBuffer_Remodeled::IBuffer_Remodeled(const shader_core_config* config, shd_warp_t *shd_warp, shader_core_stats *stats) {
    assert(config->fetch_decode_width <= config->ibuffer_remodeled_size && "Fetch decode width is bigger than remodeled IBuffer size");
    m_is_enabled = config->is_ibuffer_remodeled_enabled;
    m_num_max_entries = config->ibuffer_remodeled_size;
    m_fetch_decode_width = config->fetch_decode_width;
    m_num_entries = 0;
    m_shd_warp = shd_warp;
    m_stats = stats;
    m_is_init_next_pc = false;
    m_is_ret_reached = false;
    m_next_pc_to_fetch_request = 0;
    m_config = config;
}

IBuffer_Remodeled::~IBuffer_Remodeled() {}

bool IBuffer_Remodeled::get_is_enabled() {
    return m_is_enabled;
}

bool IBuffer_Remodeled::get_is_empty() {
    assert( ( (m_num_entries == 0) == m_remodeled_ibuffer.empty()) && "Extended Ibuffer is empty but num_entries is not 0" );
    return m_num_entries == 0;
}

unsigned int IBuffer_Remodeled::get_num_entries() {
    return m_num_entries;
}

bool IBuffer_Remodeled::can_fetch() {
    return ((m_num_max_entries - m_num_entries) >= m_fetch_decode_width) && !m_is_ret_reached;
}

bool IBuffer_Remodeled::is_full() {
    return m_remodeled_ibuffer.size() == m_num_max_entries;
}

bool IBuffer_Remodeled::get_is_ret_reached() {
    return m_is_ret_reached;
}

void IBuffer_Remodeled::set_is_ret_reached(bool is_ret_reached) {
    m_is_ret_reached = is_ret_reached;
}

address_type IBuffer_Remodeled::get_next_pc_to_fetch_request() {
    address_type res;
    if(m_is_init_next_pc) {
        res = m_next_pc_to_fetch_request;
    }else {
        res = m_config->is_trace_mode ? static_cast<trace_shd_warp_t*>(m_shd_warp)->get_pc() : (m_shd_warp)->get_pc();
    }
    m_is_init_next_pc = true;
    m_num_entries += m_fetch_decode_width;
    // [WaitBlock partial replay] When a deferred-replay stream is active, each
    // IBuffer slot must be bound to a specific replay occurrence at allocation
    // time. Its requested PC comes from the reserved occurrence, NOT from
    // res + instruction_size, so a branch target or repeated PC inside the
    // deferred stream is fetched correctly and out-of-order cache fills stay
    // matched to the right occurrence.
    trace_shd_warp_t *trace_warp =
        m_config->is_trace_mode
            ? static_cast<trace_shd_warp_t *>(m_shd_warp)
            : nullptr;
    // [WaitBlock partial replay] The reservation model assumes one IBuffer slot
    // is allocated (and re-fetched on RESERVATION_FAIL) at a time. remove_entry
    // rolls back the single most recent reservation and asserts back().m_pc ==
    // the returned request PC. With m_fetch_decode_width > 1 a request would
    // allocate several slots at once, mixing reserved replay slots with
    // sequential canonical slots and breaking that back()/first_pc identity.
    // Fail closed instead of silently corrupting replay ordering if the width is
    // ever raised above 1 while a replay stream is active.
    if (trace_warp != nullptr && m_fetch_decode_width > 1 &&
        trace_warp->partial_replay_active()) {
        fprintf(stderr,
                "[WAITBLOCK][abort] partial replay requires "
                "fetch_decode_width==1 (got %u)\n",
                m_fetch_decode_width);
        abort();
    }
    bool used_replay_reservation = false;
    address_type first_pc = res;
    for(unsigned int i = 0; i < m_fetch_decode_width; i++) {
        address_type pc_to_fetch = res + 16 * i;
        IBuffer_Entry entry(false, pc_to_fetch, NULL);
        if (trace_warp != nullptr) {
            address_type replay_pc = 0;
            unsigned int replay_index = 0;
            uint64_t replay_epoch = 0;
            if (trace_warp->reserve_next_partial_replay(
                    &replay_pc, &replay_index, &replay_epoch)) {
                entry.m_pc = replay_pc;
                entry.m_is_replay = true;
                entry.m_replay_index = replay_index;
                entry.m_replay_epoch = replay_epoch;
                if (i == 0) {
                    first_pc = replay_pc;
                }
                used_replay_reservation = true;
            }
        }
        m_remodeled_ibuffer.push_back(entry);
    }
    // The trace source can switch from the canonical stream to a deferred
    // replay while this IBuffer is uninitialized. Advance from the PC selected
    // above, not from the stale pre-switch fetch cursor. For a replay stream the
    // next fetch base is re-derived from the reservation cursor via get_pc(),
    // so leave m_next_pc_to_fetch_request uninitialized-driven in that case.
    if (used_replay_reservation) {
        m_is_init_next_pc = false;
        return first_pc;
    }
    m_next_pc_to_fetch_request = res + 16 * m_fetch_decode_width;
    return res;
}

void IBuffer_Remodeled::remove_entry(address_type pc) {
    assert(!m_remodeled_ibuffer.empty() && "Trying to remove entry from empty remodeled IBuffer");
    assert(m_remodeled_ibuffer.back().m_pc == pc && "Trying to remove entry from remodeled IBuffer that is not the first one");
    // [WaitBlock partial replay] If the removed slot held a replay reservation,
    // give that one reservation back so the occurrence is re-reserved next fetch.
    if (m_config->is_trace_mode && m_remodeled_ibuffer.back().m_is_replay) {
        static_cast<trace_shd_warp_t*>(m_shd_warp)
            ->unreserve_last_partial_replay();
    }
    m_remodeled_ibuffer.pop_back();
    m_next_pc_to_fetch_request = pc;
    m_num_entries--;
}

address_type IBuffer_Remodeled::get_next_pc_to_issue() {
    assert(!m_remodeled_ibuffer.empty() && "Trying to get next pc to issue from empty remodeled IBuffer");
    return m_remodeled_ibuffer.front().m_pc;
}

bool IBuffer_Remodeled::is_next_valid() { 
    return !m_remodeled_ibuffer.empty() && m_remodeled_ibuffer.front().m_valid; 
}

warp_inst_t *IBuffer_Remodeled::next_inst() {
    return m_remodeled_ibuffer.front().m_inst; 
}

void IBuffer_Remodeled::issued() {
    assert( (!m_remodeled_ibuffer.empty()) && "Trying to issue more entries than allowed in remodeled IBuffer");
    address_type next_traced_pc = m_remodeled_ibuffer.front().m_inst->next_traced_pc;
    address_type next_not_taken_pc = m_remodeled_ibuffer.front().m_inst->pc + m_remodeled_ibuffer.front().m_inst->isize;
    m_remodeled_ibuffer.pop_front();
    m_num_entries--;
    if(m_config->is_trace_mode && (next_not_taken_pc != next_traced_pc)) {
        flush(true);
        m_next_pc_to_fetch_request = next_traced_pc;
        m_is_init_next_pc = true;
    }
    
}

void IBuffer_Remodeled::flush(bool reset_pc_to_0) {
    while(!m_remodeled_ibuffer.empty()) {
        // [WaitBlock partial replay] Every removed slot that carried a replay
        // reservation gives that reservation back, whether or not its cache
        // response ever arrived (unfilled slots are m_valid=false and would
        // otherwise leak their reservation). This is done per-removed-entry so
        // the occurrence currently in flight (already pop_front'd by issued()
        // but not yet committed) is NOT un-reserved: it is no longer in this
        // deque, so commit_trace_inst can still advance issued past it.
        if (m_config->is_trace_mode && m_remodeled_ibuffer.front().m_is_replay) {
            static_cast<trace_shd_warp_t*>(m_shd_warp)
                ->unreserve_last_partial_replay();
        }
        if(m_remodeled_ibuffer.front().m_valid) {
            m_shd_warp->dec_inst_in_pipeline();
            if(m_config->is_interwarp_coalescing_enabled && ((m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC) ||
            (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID))) {
                if(m_config->is_trace_mode && (m_remodeled_ibuffer.front().m_inst != NULL)) {
                    m_shd_warp->m_subcore->remove_interwarp_coalescing_dep_counter_at_decode_tracking(m_remodeled_ibuffer.front().m_inst, m_shd_warp->get_warp_id());
                }
            }
            if(m_config->is_trace_mode && (m_remodeled_ibuffer.front().m_inst != NULL)
                && m_remodeled_ibuffer.front().m_inst->m_has_the_instruction_been_traced) {
                static_cast<trace_shd_warp_t*>(m_shd_warp)
                    ->rollback_trace_inst(
                        static_cast<trace_warp_inst_t*>(
                            m_remodeled_ibuffer.front().m_inst));
                delete m_remodeled_ibuffer.front().m_inst;
            }else if(m_remodeled_ibuffer.front().m_inst != NULL){
                delete m_remodeled_ibuffer.front().m_inst;
            }
        }
        m_remodeled_ibuffer.pop_front();
    }
    m_num_entries = 0;
    m_is_init_next_pc = false;
    m_is_ret_reached = false;
    m_next_pc_to_fetch_request = reset_pc_to_0 ? 0 : static_cast<trace_shd_warp_t*>(m_shd_warp)->get_pc();
    if(reset_pc_to_0) {
        m_next_pc_to_fetch_request = 0;
    }else {
        m_next_pc_to_fetch_request = m_config->is_trace_mode ? static_cast<trace_shd_warp_t*>(m_shd_warp)->get_pc() : m_shd_warp->get_pc();
    }
}

void IBuffer_Remodeled::print(FILE *fout) {
    fprintf(fout, "Remodeled_Ibuffer of warp %d, has %d/%d entries as valid\n", m_shd_warp->get_warp_id(), m_num_entries, m_num_max_entries);
    if(m_remodeled_ibuffer.empty()) {
        fprintf(fout, "Empty\n");
    }else {
        unsigned int entry_id = 0;
        for(auto &ib_entry : m_remodeled_ibuffer) {
            if(ib_entry.m_valid) {
                fprintf(fout, "Entry %d. Valid. PC: %llx: . Inst:", entry_id, ib_entry.m_pc);
                if(ib_entry.m_inst != nullptr) {
                    ib_entry.m_inst->print_insn(fout);
                    fprintf(fout, "\n");
                }else {
                    fprintf(fout, "Not filled\n");
                }
                
            }else {
                fprintf(fout, "Entry %d. Not Filled. PC: %llx.\n", entry_id, ib_entry.m_pc);
            }
            entry_id++;
        }
    }
    fprintf(fout, "\n");
    fflush(stdout);
}
