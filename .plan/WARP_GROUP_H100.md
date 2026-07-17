# WARP_GROUP_H100 — the simulator has no warpgroup execution model (WGMMA is executed 4× redundantly)

> ## ⛔ VERDICT (2026-07-17, MEASURED `.o25`/`.o42`): the "4× over-execution" hypothesis is REFUTED.
>
> The confirmation run measured **sim `Σ tensor_ops` = 835,584** vs **HW NCU gmma ≈ 835,506** →
> **ratio 1.0001 (per CTA: 2,176 vs 2,176)**. The sim executes **exactly the same number of tensor
> instructions as HW**, NOT 4×. Resolution of the earlier confusion:
> - WGMMA is *collective* in **FLOP** terms (one tile / warpgroup), BUT the **instruction count** that
>   both NCU (`sm__inst_executed_pipe_tensor_op_gmma`) and the sim report is **per-warp** — all 4 warps
>   of the warpgroup each issue the HGMMA and each is counted. sim (per-warp) == HW (per-warp). No 4×.
> - So there is **no tensor-work over-count** and **no warpgroup lever here**. `mma` FU-occupancy is
>   HW-faithful in *count*; the sim/HW `mma`-share gap is a per-op **latency/II** (config) matter, which
>   ASYNC_WGMMA.md already showed is config-tunable, not a code bug.
> - **This item is CLOSED (no action).** The remaining fwd/bwd gap is NOT tensor over-execution. See the
>   measured breakdown recorded in FA3_progress.md (Ongoing item 4 → closed; the live gap is fwd
>   drain-idle, Ongoing item 2/3). The analysis below is retained for history but its premise (4×) is
>   wrong.

> **Status: ~~PARKED~~ CLOSED — hypothesis refuted by measurement (see verdict above).** Originally
> discovered 2026-07-16 while designing async-WGMMA ([ASYNC_WGMMA.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/ASYNC_WGMMA.md)).

## 1. The finding

Real Hopper `wgmma.mma_async` is a **warpgroup** instruction: 4 warps (128 threads) collectively
issue **one** WGMMA that runs **once** on the SM tensor core. NCU counts it as **one** `gmma`
instruction.

This simulator has **no warpgroup (nor thread-block-cluster) execution unit**. It only models warps.
Every occurrence of "warpgroup" in the source is a **stall-taxonomy label** for matching an NCU
counter name, not an execution construct:
- [subcore.cc:432-434](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L432-L434) (`is_any_waiting_in_warpgroup_arrive` — a stall bucket)
- [sm.cc:569](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L569) / [sm.h:509](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h#L509) — comments explicitly state "there is no separate
  consumer-side warpgroup_arrive".

## 2. What the sim actually does (4× redundant)

**HW ground truth (authoritative sources, web-verified 2026-07-16):** `wgmma.mma_async` "is
**executed collectively by all 128 threads in a warpgroup**" and computes **one** M×N×K tile once,
with the accumulator fragments distributed across the 4 warps' registers — it is NOT 4 warps each
computing the full tile. Sources: [Colfax/CUTLASS WGMMA tutorial](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/),
[ThunderKittens "GPUs Go Brrr" (Stanford)](https://hazyresearch.stanford.edu/blog/2024-05-12-tk),
[Triton Gluon WGMMA docs](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html).
So real HW tensor work = **1 tile / warpgroup**.

Confirmed by source (search subagent + direct reads, 2026-07-16):
- The NVBit trace records the HGMMA in **every** warp of the warpgroup that executed it (all 4 warps),
  because each of the 128 threads executes `wgmma.mma_async`.
- The sim parses trace **per warp** ([trace_driven.cc:236](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L236), one `warp_inst_t` per record; no
  warpgroup fan-in / leader election anywhere).
- Each warp's HGMMA is classified `TENSOR_CORE_OP` and routed to **that warp's own subcore tensor
  pipe** ([subcore.cc:1155-1156](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1155-L1156)).
- Warps map to subcores by `warp_id % num_subcores` ([sm.cc:1195](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1195)), so the 4 warps of a
  warpgroup land on subcores 0,1,2,3 and their 4 tensor pipes run **in parallel**.
- **The shape is the FULL warpgroup tile, parsed from the opcode string.**
  `set_tensor_core_instruction_info()` ([traced_instruction.cc:538-570](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/util/traces_enhanced/src/traced_instruction.cc#L538-L570)) regex-extracts M/N/K
  from `HGMMA.64x128x16` → 64/128/16, i.e. the whole-warpgroup tile, and applies it **identically to
  each of the 4 warps' ops** ([abstract_hardware_model.cc:429-435](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L429-L435)) — **no**
  division of the warpgroup tile across the 4 warps.

**Net: one warpgroup WGMMA tile is executed 4 times (once per warp on 4 separate tensor pipes),
each doing the FULL tile.** The sim does 4× the real tensor instruction work per warpgroup tile.

## 3. Evidence / numbers

- **HW (NCU, kernel 9 `FlashAttnBwdSm90`, this trace):** `sm__inst_executed_pipe_tensor_op_gmma`
  = 5.36% of active cycles; `sm__cycles_active.sum` = 15,587,794 → **≈ 835,500 gmma instructions**
  total (grid 384 → ≈ 2,176 gmma / CTA). smsp `inst_executed.sum` = 19,952,816 (matches sim).
- **Throughput sanity (from arXiv:2501.12084, see ASYNC_WGMMA.md §9):** the sim's per-tensor-pipe
  rate is ~correct (sim useful ≈ 4096 FLOP/cyc/SM vs HW peak 3781, ratio 0.92). So the tensor-pipe
  *rate* is fine; the problem is the **4× instruction count**, which floods all 4 subcores with
  tensor work and inflates the `mma` issue-stall (sim bwd `mma` 12.5% vs HW 5.3% = 2.4×).
- **Not yet numerically confirmed sim-side:** the `[CTAFIN] tensor_ops` counter that would directly
  show the sim's tensor-op count was 0 in `.o40` (ran on the pre-fix build) and the only post-fix
  run (`.o41`) crashed on the Design-A assert. **A clean post-fix run is needed to confirm sim
  tensor_ops ≈ 4 × HW gmma.**

## 4. Why this is likely the real `mma`/`math_pipe` over-model (not async)

The original Ongoing-item-3 premise (sim over-serializes WGMMA because II is too big) was **falsified**
by the throughput back-calculation (ASYNC_WGMMA.md §9.3): sim II=32 is actually SMALLER than HW II≈72.
The 4× redundant execution is a much more plausible root cause of the sim's excess tensor-pipe stall:
4 subcores each busy with a full tile means the issue stage is tensor-bound far more of the time than
on HW, where the warpgroup drives the tensor core once.

## 5. Possible fixes (design sketch — NOT chosen yet)

- **Option W1 — warpgroup-leader-only tensor execution.** Only the leader warp of a warpgroup issues
  the TENSOR_CORE_OP into a tensor pipe; the other 3 warps' HGMMA become no-ops (or a cheap marker).
  Their `WARPGROUP.ARRIVE`/`DEPBAR` must still be honored for scoreboard/wait correctness — this is
  the main correctness risk.
- **Option W2 — quarter-tile per warp.** Keep 4 warps issuing, but each computes 1/4 of the tile
  (number_of_cycles/4). Preserves the 4-warp structure and the DEPBAR bookkeeping, only fixes the
  work量. Lower correctness risk, but is a modeling approximation (real HW is not 4 independent
  quarter-tiles).
- Either requires deciding how the 4 warps' independent DEPBAR/write-barrier counters reconcile into
  the single warpgroup completion.

## 6. Confirmation step (before any implementation)

Run the current build (tensor_ops fix `ee96251` is in; Design A reverted `760ffe5` → synchronous
baseline; `-wgmma_step0_instrument_enable 1`) and check `Σ [CTAFIN] tensor_ops` vs HW gmma ≈ 835,500.
**If sim ≈ 4× HW (≈ 3.34M), the 4× redundancy is confirmed** and this becomes a real lever. All of
this is timing-neutral instrumentation.

**Instrumentation available for this run (all gated, timing-neutral):**
- `-wgmma_step0_instrument_enable 1` → emits per-CTA `[CTAFIN]` with `tensor_ops` (the 4× check),
  `sm_idle_tensor_cyc`, `fu_occupied_tensor_cyc` (= NCU `mma`).
- `-cta_stall_breakdown_instrument_enable 1` (NEW, added 2026-07-16) → appends `sm_idle_cyc` and
  `sm_idle_ibuffer_empty_cyc` to the same `[CTAFIN]` line — the non-tensor drain-idle columns needed
  for **Open item 2 (fwd finish-cycle variance)**. Lets one run decide, per CTA, whether the slow
  CTAs are tensor-bound (tensor_ops / fu_occupied_tensor_cyc) or drain-idle/frontend-bound
  (sm_idle_cyc / sm_idle_ibuffer_empty_cyc).
- Verified no instrumentation bug: `inc_tensor_ops_for_warp` keys off `is_tensor_core_op()`
  ([functional_unit.cc:178](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L178)) on the real fixed-latency issue path
  ([subcore.cc:277](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L277)); the earlier `tensor_ops=0` was only because `.o40` ran on the
  pre-fix build and `.o41` crashed on the Design-A assert.

**Expected `[CTAFIN]` line format (with both flags on):**
```
[CTAFIN] sm=.. cta_slot=.. global_cta=.. start_cyc=.. finish_cyc=.. elapsed_cyc=.. tensor_ops=.. sm_idle_tensor_cyc=.. fu_occupied_tensor_cyc=.. sm_idle_cyc=.. sm_idle_ibuffer_empty_cyc=..
```

Once confirmed, choose W1 vs W2 (§5). Note the correctness caveat there: NCU counting (per-warp vs
per-warpgroup) is a *metric* nuance, but the real tensor **work** is unambiguously 1 tile/warpgroup
(authoritative sources, §2), so the sim's 4× is a genuine over-execution regardless of how NCU tallies
the `gmma` counter.
