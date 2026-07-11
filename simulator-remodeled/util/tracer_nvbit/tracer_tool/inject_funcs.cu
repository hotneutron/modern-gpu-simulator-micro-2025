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

#include <cstdarg>
#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

#include "../../traces_enhanced/src/string_utilities.h"

/* Instrumentation function that we want to inject, please note the use of
 *  extern "C" __device__ __noinline__
 *    To prevent "dead"-code elimination by the compiler.
 */
extern "C" __device__ __noinline__ void instrument_inst(
    int32_t pred, uint32_t unique_function_id, uint32_t vpc, uint32_t num_of_injects,
    uint32_t per_operand_type, uint32_t memory_type, uint64_t addr_or_offset_or_reg_id,
    uint32_t mem_width_or_op_reg_val_0, uint32_t op_reg_val_1, uint32_t op_reg_val_2,
    uint32_t op_reg_val_3, uint64_t pchannel_dev, uint64_t ptotal_dynamic_instr_counter,
    uint64_t preported_dynamic_instr_counter, uint64_t pstop_report) {

  const int active_mask = __ballot_sync(__activemask(), 1);
  const int predicate_mask = __ballot_sync(__activemask(), pred);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  if ((*((bool *)pstop_report))) {
    if (first_laneid == laneid) {
      atomicAdd((unsigned long long *)ptotal_dynamic_instr_counter, 1);
      return;
    }
  }

  inst_trace_t ma;
  ma.ureg_desc_id = SECRET_UREG_DESC_NOT_USED;
  ma.ureg_desc_value = SECRET_UREG_DESC_NOT_USED;
  ma.ureg_desc_value_hi = SECRET_UREG_DESC_NOT_USED;
  ma.num_of_injects = num_of_injects;
  ma.per_operand_type = per_operand_type;
  if (memory_type == MEM_TYPE::STANDARD_MEM) {
    /* collect memory address information */
    for (int i = 0; i < 32; i++) {
      ma.addrs_or_reg_val_0[i] = __shfl_sync(active_mask, addr_or_offset_or_reg_id, i);
    }
    ma.width = mem_width_or_op_reg_val_0;
    ma.mem_type = MEM_TYPE::STANDARD_MEM;
    if(op_reg_val_1 != SECRET_UREG_DESC_NOT_USED) {
      ma.ureg_desc_value = op_reg_val_1;
    }
    if(op_reg_val_2 != SECRET_UREG_DESC_NOT_USED) {
      ma.ureg_desc_id = op_reg_val_2;
    }
    if(op_reg_val_3 != SECRET_UREG_DESC_NOT_USED) {
      ma.ureg_desc_value_hi = op_reg_val_3;
    }
  }else if(memory_type == MEM_TYPE::CONSTANT_MEM) {
    ma.mem_type = MEM_TYPE::CONSTANT_MEM;
    uint32_t reg_value = mem_width_or_op_reg_val_0;
    ma.width = 4;
    for (int tid = 0; tid < 32; tid++) {
        ma.addrs_or_reg_val_0[tid] = __shfl_sync(active_mask, reg_value, tid);
    }
  }else if(memory_type == MEM_TYPE::CALL_OR_RET) {
    ma.mem_type = MEM_TYPE::CALL_OR_RET;
    ma.width = 1;
    uint32_t reg_value_1 = mem_width_or_op_reg_val_0;
    uint32_t reg_value_2 = op_reg_val_1;
    for (int tid = 0; tid < 32; tid++) {
        uint32_t th_reg_val1 = __shfl_sync(active_mask, reg_value_1, tid);
        uint32_t th_reg_val2 = __shfl_sync(active_mask, reg_value_2, tid);
        uint64_t final_addr = static_cast<uint64_t>(th_reg_val2) << 32;
        final_addr += th_reg_val1 + addr_or_offset_or_reg_id;
        ma.addrs_or_reg_val_0[tid] = final_addr;
    }
  }else {
    ma.mem_type = MEM_TYPE::NONE;
    ma.width = 0;
    ma.reg_id = addr_or_offset_or_reg_id;
    if((per_operand_type == TRACED_REG_TYPE::REGULAR) || (per_operand_type == TRACED_REG_TYPE::REGULAR_2_REGS) || 
        (per_operand_type == TRACED_REG_TYPE::REGULAR_4_REGS) || (per_operand_type == TRACED_REG_TYPE::PREDICATE)) {
      for (int tid = 0; tid < 32; tid++) {
        ma.addrs_or_reg_val_0[tid] = __shfl_sync(active_mask, mem_width_or_op_reg_val_0, tid);
      }
      if(per_operand_type == TRACED_REG_TYPE::REGULAR_2_REGS) {
        for (int tid = 0; tid < 32; tid++) {
          ma.reg_val_1[tid] = __shfl_sync(active_mask, op_reg_val_1, tid);
        }
      }
      if(per_operand_type == TRACED_REG_TYPE::REGULAR_4_REGS) {
        for (int tid = 0; tid < 32; tid++) {
          ma.reg_val_2[tid] = __shfl_sync(active_mask, op_reg_val_2, tid);
        }
        for (int tid = 0; tid < 32; tid++) {
          ma.reg_val_3[tid] = __shfl_sync(active_mask, op_reg_val_3, tid);
        }
      }
    }else if((per_operand_type == TRACED_REG_TYPE::UNIFORM) || (per_operand_type == TRACED_REG_TYPE::UNIFORM_2_REGS) ||
        (per_operand_type == TRACED_REG_TYPE::UNIFORM_PREDICATE)) {
      for (int tid = 0; tid < 32; tid++) {
        ma.addrs_or_reg_val_0[tid] = mem_width_or_op_reg_val_0;
      }
      if(per_operand_type == TRACED_REG_TYPE::UNIFORM_2_REGS) {
        for (int tid = 0; tid < 32; tid++) {
          ma.reg_val_1[tid] = op_reg_val_1;
        }
      }
    }
  }

  int4 cta = get_ctaid();
  int uniqe_threadId = threadIdx.z * blockDim.y * blockDim.x +
                       threadIdx.y * blockDim.x + threadIdx.x;
  ma.warpid_tb = uniqe_threadId / 32;

  ma.cta_id_x = cta.x;
  ma.cta_id_y = cta.y;
  ma.cta_id_z = cta.z;
  ma.warpid_sm = get_warpid();
  ma.vpc = vpc;
  ma.unique_function_id = unique_function_id;
  ma.active_mask = active_mask;
  ma.predicate_mask = predicate_mask;
  ma.sm_id = get_smid();

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&ma, sizeof(inst_trace_t));
    atomicAdd((unsigned long long *)ptotal_dynamic_instr_counter, 1);
    atomicAdd((unsigned long long *)preported_dynamic_instr_counter, 1);
  }
}

/* Phase 0b (TMA_BASE_ADDR.md §2.13). Injected at IPOINT_AFTER on the kernel-entry
 * ULDC.64 URx, c[0x0][K] that loads param_base. dst_lo/dst_hi are the loaded
 * destination UREG pair (read post-execution => the param_base value). First
 * active lane records it once per launch into the managed slot; the host reads it
 * after the launch syncs and cuMemcpyDtoH's the param region. */
extern "C" __device__ __noinline__ void capture_tma_param_base(
    int32_t pred, uint32_t unique_function_id, uint32_t dst_lo, uint32_t dst_hi,
    uint64_t pcapture) {
  if (!pred) {
    return;
  }
  const int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;
  if (first_laneid != laneid) {
    return;
  }
  tma_param_base_capture_t *slot = (tma_param_base_capture_t *)pcapture;
  /* first writer per launch wins (grid-uniform value; slot->valid reset by host
   * before each launch). */
  if (atomicCAS(&slot->valid, 0, 1) == 0) {
    slot->param_base =
        ((unsigned long long)dst_hi << 32) | (unsigned long long)dst_lo;
    slot->unique_function_id = unique_function_id;
  }
}

/* SPIKE 6 device fact-check (TMA_BASE_ADDR.md §2.18). Injected at IPOINT_BEFORE on an
 * executed UTMALDG/UTMASTG. desc_va is the effective address of the descriptor-carrying
 * memory-ref operand (the generic-SMEM tensormap address, e.g. 0xffffffffc428xxxx),
 * passed via nvbit_add_call_arg_mref_addr64. Read 128B there and store qword0/qword1
 * into a bounded managed ring. First active lane only; a handful of samples across pcs
 * is enough to confirm qword0 == the real base. */
extern "C" __device__ __noinline__ void factcheck_tma_descriptor(
    int32_t pred, uint32_t unique_function_id, uint32_t vpc, uint32_t mref_ord,
    uint64_t desc_va, uint32_t do_read, uint64_t pfact) {
  if (!pred) {
    return;
  }
  const int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;
  if (first_laneid != laneid) {
    return;
  }
  tma_desc_factcheck_t *fc = (tma_desc_factcheck_t *)pfact;
  atomicAdd(&fc->count, 1);

  /* Classify the generic address with non-faulting predicates (isspacep.*). This is
   * pure classification — it never touches memory, so recording the space for every
   * sample is always crash-safe. CAVEAT (SPIKE 7 CSV): isspacep.global returns TRUE for
   * a small raw-shared cursor like 0xe800 (it is not in the shared/local/const window,
   * so "global" by elimination), so GLOBAL alone does NOT mean "safe to deref". The read
   * below additionally requires a plausibly-mapped high VA. */
  void *gp = (void *)desc_va;
  unsigned int space = TMA_VA_UNKNOWN;
  if (__isShared(gp)) {
    space = TMA_VA_SHARED;
  } else if (__isConstant(gp)) {
    space = TMA_VA_CONSTANT;
  } else if (__isLocal(gp)) {
    space = TMA_VA_LOCAL;
  } else if (__isGlobal(gp)) {
    space = TMA_VA_GLOBAL;
  }

  unsigned int read_ok = 0;
  unsigned long long q0 = 0, q1 = 0;
  /* Read a GLOBAL VA that is above the tiny-cursor band. SPIKE 7 CSV: mref-0 is a tiny
   * raw-shared cursor (<= 0x18400 ≈ 99 KiB) that isspacep mislabels GLOBAL and that
   * faults when read; mref-1 is the descriptor cursor (~0x562804c0 ≈ 1.4 GiB, +0xc0=192
   * stride = box_dim). A 1 MiB floor rejects the tiny cursors while admitting the
   * descriptor cursor. NOTE: reading mref-1 may still fault if it is a shared/generic
   * alias rather than a true global — that outcome is itself the D1-vs-D2 answer. */
  const unsigned long long MIN_READ_VA = 0x100000ULL;  /* 1 MiB */
  if (do_read && space == TMA_VA_GLOBAL && desc_va >= MIN_READ_VA) {
    const unsigned long long *p = (const unsigned long long *)desc_va;
    q0 = p[0];
    q1 = p[1];
    read_ok = 1;
  }

  unsigned int idx = atomicAdd(&fc->stored, 1);
  if (idx < TMA_DESC_FACTCHECK_SLOTS) {
    fc->unique_function_id[idx] = unique_function_id;
    fc->pc[idx] = vpc;
    fc->mref_ord[idx] = mref_ord;
    fc->space[idx] = space;
    fc->read_ok[idx] = read_ok;
    fc->desc_va[idx] = desc_va;
    fc->qword0[idx] = q0;
    fc->qword1[idx] = q1;
  }
}
