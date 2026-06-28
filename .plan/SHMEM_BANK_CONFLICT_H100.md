# Opt 6 — Shared-Memory Bank-Conflict Model Fix (H100 / FA3)

> Target: FA3 fwd (kernel 5) and FA3 bwd (kernel 10). This is the planned **Opt 6**,
> applied on top of Opt 5 (eager-promote). Frontend (L1I prefetch) is already handled by
> Opt 4/5, so the dominant remaining buckets are now execution-side
> (`non_tma_axis`, `fu_occupied`) and downstream back-pressure (`next_stage_not_available`).
> This document is the root-cause analysis + fix design + verification plan. Implementation
> follows after review.

## 1. Why this optimization

### 1.1 The symptom — sim massively over-counts shared bank conflicts

| Kernel | sim `gpgpu_n_shmem_bkconflict` | HW (NCU `...bank_conflicts_pipe_lsu_mem_shared_op_st`) | sim / HW |
|---|---|---|---|
| FA3 fwd (k5, Opt 5 `.o18`) | **38,016** | **281** | ~135× |
| FA3 bwd (k10, Opt 4 `.o1`) | **1,327,104** | **35,493** | ~37× |

- bwd is the heavier shared-memory consumer (1.33M vs 38k), and bwd is exactly the kernel
  whose largest remaining stall buckets are execution-side
  (`non_tma_axis = 27.15%`, `fu_occupied = 18.08%`, `next_stage_not_available = 18.96%`).
- HW NCU explicitly says this kernel is **not** memory-bandwidth bound; the FA3 smem layout
  is **swizzled to be (near) conflict-free** by design. So a 37–135× over-count is a
  **model bug**, not a latency-tuning gap.

> **CRITICAL — measured mechanism (verified, corrects the first draft).** The big numbers
> 38,016 / 1,327,104 are **NOT** the per-instruction conflict size. The real per-instruction
> cost is small:
>
> | | fwd `.o18` | bwd `.o1` |
> |---|---|---|
> | `total_accesses_per_shared_instruction` (= avg `cycles` set per shared inst) | **3.3607** | **3.0377** |
>
> This stat is `total_conflicts / total_shared_instructions`
> ([shader.cc:964](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L964)), i.e. the average `cycles` (= `total_accesses`)
> assigned to each shared LOAD/STORE. So the over-count decomposes as
> **`gpgpu_n_shmem_bkconflict ≈ (per-inst ≈ 3) × (dynamic shared-inst count) × (Defect-B per-cycle
> double-count)`**. The headline 37–135× is dominated by **Defect B (per-cycle accumulation,
> §2 Defect B)**, not by a huge per-instruction conflict.
>
> Why per-inst ≈ 3–4: for a 16B (`STS.128`/`STSM`) access the 32 lanes are ~16B apart, so under
> `bank = (addr/4)%32` they land on only 8 distinct banks with **4 lanes per bank** ⇒
> `max_bank_accesses ≈ 4`; mixing in the 4B `LDS` (which is 1) gives the measured ~3. HW services
> the same 16B access as conflict-free wavefronts (HW `lsu_wavefronts_mem_shared ≈ 56.6%` of peak,
> `mio_throttle + short_scoreboard` stalls only ~15% of warp-cycles), i.e. effectively ~1, not ~4.

### 1.2 Why the over-count inflates cycles — and how much

The bank-conflict count is not just a stat — it is used as the **serialization (dispatch)
delay** of the shared-memory access:

- In [generate_mem_accesses()](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L675-L772),
  the shared path computes `total_accesses` = max distinct 4-byte words mapped to any single
  bank, then sets `cycles = total_accesses` ([abstract_hardware_model.cc:768](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L767-L771)).
- `cycles` is the instruction's dispatch delay ([abstract_hardware_model.h:1654-1663](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L1654-L1663)),
  decremented once per cycle in the LDST PRT retire loop.
- The PRT keeps re-emitting the same shared entry every cycle until `dispatch_delay()` hits 0
  ([ldst_unit_sm.cc:1860-1873](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/ldst_unit_sm.cc#L1860-L1873)),
  and increments `gpgpu_n_shmem_bkconflict` **on each of those extra cycles**.

So an N-way (false) conflict costs **N−1 extra serialization cycles** on the shared pipe,
feeding `mio_throttle` / `short_scoreboard` (the `non_tma_axis`) and back-pressuring the
LDST sub-pipeline (`next_stage_not_available`).

> **Realistic cycle-impact estimate (do not over-promise).** Because the per-instruction
> over-charge is only ~`(3.36 − 1) ≈ 2.4` cycles (fwd) / `(3.04 − 1) ≈ 2` cycles (bwd) per shared
> access, the *cycle* gain from fixing it is bounded. On HW the shared-serialization stalls
> (`mio_throttle` 6.3% + `short_scoreboard` 8.5% ≈ 15% of warp-cycles for bwd) are **not** the
> dominant bucket — `fu_occupied`/WGMMA (18%) is larger. So Opt 6 is expected to trim the
> `non_tma_axis`/`next_stage` buckets modestly, **not** to be a large single-step win. The other
> half of its value is **accuracy**: bringing the reported `gpgpu_n_shmem_bkconflict` from 37–135×
> down to HW order (Defect B), so the metric becomes trustworthy.

### 1.3 Which operations are (and are NOT) affected — verified from the trace decode

The bank-conflict block is only reached when **(a)** `op ∈ {LOAD_OP, TENSOR_CORE_LOAD_OP,
STORE_OP, TENSOR_CORE_STORE_OP}` (op filter,
[abstract_hardware_model.cc:629-631](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L629-L631))
**and (b)** `space ∈ {shared_space, sstarr_space}` (switch,
[abstract_hardware_model.cc:674-773](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L674-L773)).

**Reaches the shared bank-conflict path** (space set to `shared_space` in the trace decoder
[trace_driven.cc](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc)):

| Op | Decode site | Width (`data_size`) source |
|---|---|---|
| `LDS` | trace_driven.cc:478-482 | numeric token in opcode (`.32`→4B, `.128`→16B) |
| `STS` / `STS.128` | trace_driven.cc:483-487 | same `OP_STS`, width via `data_size` |
| `ATOMS` | trace_driven.cc:488-493 | — |
| `LDSM` (ldmatrix) | trace_driven.cc:494-497 | `num_matrix*32/8` from `.2`/`.4` suffix |
| `STSM` (stmatrix) | trace_driven.cc:498-502 | `num_matrix*32/8` |
| generic `LD`/`ST` resolved to shared | trace_driven.cc:515 / :524 | — |
| `LDGSTS` (cp.async) **shared-store stage only** | abstract_hardware_model.cc:378-393 | load stage is `global_space` (trace_driven.cc:431); only the STS stage hits the shared path |

**Actual FA3 shared opcodes (verified by counting the trace SASS, static count per CTA):**

| Kernel | Shared opcodes present | Widths |
|---|---|---|
| fwd (`flash_fwd_hdim64_bf16_sm90.sass`) | `STSM.16.M88.4`, `LDS.128`, `STS.128` | **all 16B** (4 phases) |
| bwd (`flash_bwd_hdim64_bf16_softcapall_sm90.sass`) | `LDS` (no suffix, **736**), `STS.128` (1120), `STSM.16.M88.4`/`MT88.4` (939), `LDS.128` (129); `LDSM` = 0 | **mixed: 4B (`LDS`) + 16B (rest)** |

This confirms two things directly from data:
- It is **not** only `STS.128`/`LDSM` — bwd's most frequent shared op is the plain **4-byte `LDS`** (736 static sites).
- Therefore the fix must be **opcode-agnostic and driven purely by `data_size`**: a 4B `LDS`
  resolves to `phases = 1` (identical to today's behavior), while a 16B `STS.128`/`STSM` resolves
  to `phases = 4`. No per-opcode branching is needed or wanted (see §3 Step 1).

**Does NOT reach the path (important — confirms the user's point):**

- **TMA** (`UTMALDG`/`UTMASTG`/`UBLKCP` → `TMA_LOAD_OP`/`TMA_STORE_OP`) is excluded by the op
  filter at [abstract_hardware_model.cc:631](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L629-L631),
  never sets a memory `space` in decode, is routed to a **separate** `m_tma_pipeline`
  ([subcore.cc:979-983](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L979-L983)),
  and is modeled by byte-volume / 128B-request granularity in
  [tma_unit_sm.cc](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc).
  **TMA carries its own `swizzle` field** (from the TMA descriptor:
  [tma_types.h:60](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h#L60), `:257`;
  [tma_unit_sm.cc:380](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L380);
  parsed at [gpu-sim.cc:375-377](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L375-L377)).
  So **swizzle is a TMA-only concept today** and this Opt does NOT touch TMA at all.

> Correction vs the first draft: it is wrong to say "FA3 uses swizzled layouts so the generic
> bank model is swizzle-unaware". The swizzle that the TMA path applies when it *writes* a tile
> into shared memory is already reflected in the **actual byte addresses** that the later
> `LDSM`/`LDS`/`STS` instructions use — and those addresses are captured in the trace
> (see §1.4). So we do **not** need to model swizzle in the bank function. The only real defect
> is the **word granularity** (§2, Defect A).

### 1.4 What is available at the bank-conflict site (so a fix is feasible) — verified

All three inputs needed for a width-correct model are present at
[generate_mem_accesses()](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L675-L772) time:

1. **Real per-thread byte addresses** — `m_per_scalar_thread[thread].memreqaddr[0]`
   ([:690](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L690)),
   populated from genuine trace addresses (`set_addr` at trace_driven.cc:401; parsed in
   trace_parser.cc:258-265). For LDGSTS, the **memref2** (shared destination) addresses are
   copied in before re-running (abstract_hardware_model.cc:385-389). These addresses already
   embody whatever swizzle the producer applied.
2. **Real per-instruction access width** — `data_size` (set at trace_driven.cc:398 from
   `trace.memadd_info[0]->width`, derived per opcode in trace_parser.cc:183-207).
   **The current model ignores it** and hardcodes `WORD_SIZE = 4`.
3. **Opcode string** — `m_extra_trace_instruction_info->get_op_code()`, already called from a
   sibling member at [abstract_hardware_model.cc:606](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L604-L617),
   hence callable here too (guard with `has_extra_trace_instruction_info()`).

## 2. Root cause (two independent defects)

### Defect A — fixed `WORD_SIZE = 4` word granularity (NOT a swizzle defect)

[abstract_hardware_model.h:430-434](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L430-L434):

```
static const address_type WORD_SIZE = 4;
unsigned shmem_bank_func(address_type addr) const {
  return ((addr / WORD_SIZE) % num_shmem_bank);   // (addr/4) % 32
}
```

The shared loop ([abstract_hardware_model.cc:685-765](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L685-L765))
reads **one start address per lane** (`memreqaddr[0]`, verified — `set_addr` stores a single
addr per lane, [abstract_hardware_model.h:1438-1448](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L1438-L1448)),
maps it with `bank = (addr/4)%32`, `word = addr & ~3`
([line_size_based_tag_func](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L259-L263)),
builds `bank_accs[bank][word]`, then sets `cycles = total_accesses =`
`max_bank(#distinct words in that bank)` (broadcast-off path,
[:753-765](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L753-L765)).

Why this over-counts for **vectorized** shared accesses (`STS.128`, `STSM`) — the **correct**
mechanism (the first draft's "one access is split into 4 words" was wrong; the model only ever
records ONE word per lane):
- The model treats each lane as touching a **single 4B word at its start address**, ignoring
  that the access is actually 16B wide. For a real 16B access the 32 lanes are ~16B apart, so
  `bank = (addr/4)%32` maps them onto only **8 distinct banks, 4 lanes each**, and since each of
  those 4 lanes has a different start `word`, `max_bank_accesses ≈ 4`. Measured: avg per-inst
  `total_accesses ≈ 3.36` fwd / `3.04` bwd (§1.1).
- HW does the opposite: a 16B shared transaction is split into **4 sub-word phases**, and within
  each phase the (swizzled) layout spreads the 32 lanes across all 32 banks ⇒ ~1 per phase ⇒
  effectively conflict-free. The sim never models phases, so it charges the ~4× collision once.
- Net effect: the model charges ~3–4 serialization cycles for an access HW treats as ~1. This is
  the **per-instruction** over-charge; the headline 37–135× counter inflation on top of it is
  Defect B.
- Note: LDSM/STSM are recognized today **only for latency** at
  [abstract_hardware_model.cc:604-617](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L604-L617);
  the conflict/`cycles` computation ignores them and `data_size` entirely.

### Defect B — counter semantics (conflict-cycles vs conflict-events)

[ldst_unit_sm.cc:1871](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/ldst_unit_sm.cc#L1860-L1873)
increments `gpgpu_n_shmem_bkconflict` **every cycle** the entry still has remaining
`dispatch_delay`. So an N-way conflict adds ~N−1 to the counter.

- HW's `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st = 281 / 35,493` is closer to a
  **per-conflict-event / extra-wavefront count**, not a per-cycle accumulation.
- Even after Defect A is fixed, the **axis** of this counter must match HW before the
  number is comparable. Right now sim and HW measure different things.

## 3. How to implement

> Order chosen for lowest correctness risk and clean A/B. All changes are behind a config
> flag so the run can be compared 1:1 against Opt 5.

### Step 1 — Make the bank-conflict model access-width (phase) aware (Defect A)

**The fix is NOT about swizzle.** Swizzle is already baked into the trace addresses (§1.3/§1.4),
and TMA — the only place a swizzle field lives — never touches this path. The defect is purely
that the model treats every shared access as a single 4-byte word. The fix is to model a
wide (vectorized) shared access the way HW does: as a sequence of **phases**, each phase being a
32-bank-wide slice, and to compute the conflict per phase rather than collapsing all 4B words of
all lanes into one window.

Concretely, in the `case shared_space / sstarr_space` block of
[generate_mem_accesses()](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L675-L772):

- Add a config flag `-gpgpu_shmem_phase_aware_bank_conflict` (default **off** for clean A/B).
- Read `data_size` (the real access width) and the opcode string (already available, §1.4).
- Replace the single per-lane `bank_accs[bank][word]++` (which uses `addr/4`) with a
  **phase loop**. A 128-bit HW shared transaction is serviced in
  `phases = max(1, data_size / 4)` 4-byte phases (e.g. STS.128 → 4 phases, 32-bit → 1 phase).
  For each phase `p`:
  - the byte address each lane touches in that phase is `addr + p*4`;
  - bank = `(addr + p*4)/4 % 32`, word = `(addr + p*4)/4`;
  - build a **per-phase** `bank_accs` map and compute that phase's `max_bank_accesses`
    (max distinct words mapped to any single bank within this phase only — phases never share
    a map, which is what removes the false 4× collapse for STS.128).
- **Phase cost combination — selectable via a second config flag**
  `-gpgpu_shmem_phase_conflict_combine {sum|max}` (both implemented; chosen by A/B):
  - `sum` (**expected more accurate, default-on-when-phase-aware**): `total_accesses = Σ_p phase_cost[p]`.
    Models phases as **serially issued**, so a conflict-free `STS.128` still costs `phases`
    (e.g. 4) cycles — matching the fact that a 128-bit shared access is drained over multiple
    4B phases on HW even with no conflict.
  - `max`: `total_accesses = max_p phase_cost[p]`. Models phases as overlapped; a conflict-free
    access costs 1.
  - Run both and keep whichever matches HW (`281` fwd / `35,493` bwd) and the cycle trend best.
- Result with `sum`: a conflict-free vectorized access yields `phase_cost[p] == 1` per phase ⇒
  `total_accesses == phases` (the genuine, small serialization), instead of today's
  4×(num lanes mapped to one 4B bank) false conflict.
- LDSM/STSM: the trace addresses are the per-lane shared addresses ldmatrix actually issues;
  with the phase-aware per-4B-word counting these come out conflict-free when the layout is
  conflict-free, **without any special-case "treat as conflict-free" hack**. (The earlier draft's
  idea of force-flagging LDSM as conflict-free was wrong — let the real addresses decide.)
- Keep the legacy single-4B-word path intact when the phase-aware flag is off, so existing
  non-FA3 configs are byte-for-byte unchanged.

**Opcode-agnostic — no per-opcode branching.** The phase loop is keyed only on `data_size`, so
it covers *every* shared LOAD/STORE uniformly:
- 4B `LDS`/`STS` (bwd has 736 plain `LDS`) → `phases = 1` ⇒ identical to today's single-word
  result (no behavior change for these);
- 8B → `phases = 2`; 16B `STS.128`/`STSM.16`/`LDS.128` → `phases = 4`;
- `LDSM`/`STSM` widths come straight from `data_size` (matrix-count derived), so they need no
  special case. Letting `data_size` drive it is exactly why mixed-width kernels (bwd) are handled
  correctly without enumerating opcodes.

**Scope — shared memory only; L1D / L2 are NOT touched.** Bank conflicts are a shared-memory
concept and live only in the `shared_space`/`sstarr_space` branch
([abstract_hardware_model.cc:674-773](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L674-L773)).
Global/local accesses take a **different** path —
[memory_coalescing_arch()](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L830-L862) — which models
**32B-sector / 128B-segment coalescing** for L1D/L2, not banks. There is no "phase" or
shared-bank concept there, and this Opt does not modify it. (On H100, FA3 moves global data
mostly via TMA, which bypasses L1D entirely.) So the phase change is strictly confined to the
shared-memory bank model.

> Open item to confirm during implementation (one targeted check, no full run): for a sample
> FA3 `STS.128`/`LDSM` instruction, dump the 32 per-lane addresses from the trace and hand-verify
> that the phase-aware count is ~1/phase. This validates the model before the 40 h run. The
> `-enable_ptx_file_line_stats 1` per-line attribution (already wired via
> `ptx_file_line_stats_add_smem_bank_conflict`, abstract_hardware_model.cc:770) can localize which
> shared PCs still report conflicts after the fix.

### Step 2 — Fix the counter semantics (Defect B)

- Change `gpgpu_n_shmem_bkconflict` to count **conflict events / extra wavefronts**
  (`total_accesses - 1` once per instruction when `total_accesses > 1`), recorded at
  `assign`/issue time, **not** once per delay cycle.
- Keep the per-cycle serialization behavior (the `cycles` delay) for timing; only the
  *counter* increment moves out of the per-cycle loop. (Optionally keep the old per-cycle
  number under a separate name `gpgpu_n_shmem_bkconflict_cycles` for continuity.)

### Step 3 — Rebuild + A/B run both kernels

- Run fwd (k5) and bwd (k10) with the flag **off** (sanity: must reproduce Opt 5 / Opt 4
  numbers exactly) and **on**.
- Compare `gpgpu_n_shmem_bkconflict` against HW (281 / 35,493) and check the cycle delta.

*Files expected to change:* `abstract_hardware_model.h` (`WORD_SIZE` usage / bank func / flags),
`abstract_hardware_model.cc` (`generate_mem_accesses()` shared path — phase loop + `data_size`,
sum/max combine), `ldst_unit_sm.cc` (counter increment site), `gpu-sim.cc` (register the two
config flags `-gpgpu_shmem_phase_aware_bank_conflict` and `-gpgpu_shmem_phase_conflict_combine`),
`gpgpusim.config` (expose the flags).

## 4. Verification

- **Primary**: `gpgpu_n_shmem_bkconflict` drops from 38,016 → O(281) (fwd) and
  1,327,104 → O(35,493) (bwd), i.e. same order of magnitude as HW.
- **Cycle effect**: expect a reduction concentrated in the `non_tma_axis` /
  `next_stage_not_available` buckets, largest in **bwd** (the heavier smem consumer).
- **Safety**: flag-off run must be bit-for-bit identical to Opt 5 / Opt 4 cycle counts.
- **Sanity**: `gpu_sim_insn` unchanged (model change must not alter instruction count).

## 5. Result

— (pending implementation + run) —
