# Opt 6 — TMA transfer latency / injection over-modeling

> **Current decision.** This plan still uses the evidence from the address-hotspot / sector-merge
> and injection-bandwidth analyses, but implementation must be split into separately-gated
> experiments:
>
> - **Opt 6A (first):** run observe-only TMA latency decomposition, then reduce TMA transfer
>   injection/sectorization only if the measured bottleneck is upstream of L2 sector splitting.
> - **Opt 6B (separate):** revisit the TMA address model only after 6A, because address changes can
>   improve, fake, or even regress cycle count depending on their L2-hit effect.
>
> This supersedes and combines two earlier drafts that attacked the same symptom from different
> angles:
> - [TMA_ADDR_MERGE_PLAN.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/TMA_ADDR_MERGE_PLAN.md)
>   (synthetic-address L2 hotspot + 32B→128B sector merge). Its "Phase A" (mock base + 128B mf)
>   was **rolled back** (see §5) and is **not** in the current code tree — only its `-tma_debug_*`
>   logging infra was merged.
> - the injection-bandwidth analysis (`kMaxRequestsPerCycle` + shared-icnt back-pressure).
>
> They are coupled symptoms, not one safe monolithic fix. The old `config_id`-only mock-base path
> was too risky because it could overstate L2 reuse and produce a fake cycle win. A more realistic
> address model may also **increase** TMA/memory cycles if it lowers the simulator's already-high L2
> hit rate toward HW.
>
> Opt-6 number reused: the original Opt-6 (L1I frontend `stream_buffer_wait`) is deferred
> ([L1I_PREFETCH_LOOKAHEAD_H100.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/L1I_PREFETCH_LOOKAHEAD_H100.md)).
>
> Target: FA3 fwd (k5) / bwd (k10), on top of Opt 5. Goal for **6A**: decide whether reducing TMA
> injection/sectorization is actually worth a 12h behavior run. Goal for **6B**: improve address
> realism without accepting fake locality.

## 1. Symptom — poor IPC, dominated by a TMA-side wait

The remaining gap (fwd **2.24x** = 151,350 vs 67,696; bwd **1.82x** = 241,238 vs 132,901)
decomposes into two independent factors:

1. **Instruction count is +11–13% over HW** — sim `gpgpu_n_tot_w_icount` fwd 16,063,168 / bwd
   22,630,632 vs NCU `Executed Instructions` fwd 14,482,551 / bwd 19,952,816. (Secondary; tracked
   separately.)
2. **IPC is 0.50x (fwd) / 0.62x (bwd) of HW** — the primary lever. Per-scheduler
   warp-inst/elapsed-cyc: sim 0.201 / 0.178 vs HW 0.405 / 0.285.

Poor IPC is a *result*. The cause is what leaves schedulers with no eligible warp. Decomposing the
**true SM-idle** (`sm_all_subcores_idle` — counted only on cycles where **no** subcore on the SM
issued; the per-subcore percentages over-count ~3–7x, the WGMMA/frontend mirage) by reason
(fwd `.o23` / bwd `.o5`):

| SM-idle reason | fwd | bwd | Recoverable? |
|---|---|---|---|
| `sm_all_subcores_idle` (sum) | 18.68% | 18.17% | — |
| `nv_ibuffer_empty` (tail-drain) | 12.21% | 10.08% | ❌ (HW shows same; FA3_progress Deferred Opts) |
| **`wait_barrier` (TMA load mbarrier wait)** | **9.96%** | **10.90%** | ✅ **#1 lever** |
| `tma_flush` (`UTMACMDFLUSH` store drain) | 0.00% | 4.73% | ✅ (bwd) |
| `stall_count` / `fu_occupied` / `next_stage` | small | small | partial |

After removing the unrecoverable tail-drain, the dominant SM-idle is the **TMA axis**
(`wait_barrier` + `tma_flush`): consumer warpgroups parked on the mbarrier a TMA load must satisfy.
So the lever is the modeled TMA latency.

## 2. Root cause (code + measured): latency symptom plus risky address model

A single TMA load transfer carries **24,576 bytes = 768 × 32B sector mem_fetches**, emitted as 768
individual 32B mfs over the SM's single shared interconnect port. Measured per-transfer latency is
huge: fwd median `lat_total` 3,445 / mean 4,103 / max 11,025 cyc; bwd median 2,638 / mean 2,774 cyc
(SM-0 sampled). With `lat_queue=0`, `lat_issue=1`, essentially all of it is `lat_mem` — and
`lat_mem` is **not** memory round-trip. Configured memory latency is small
(`-gpgpu_l2_rop_latency 100`, `-dram_latency 243`). Traced lifecycle of one bwd load (uid=115):

```
104207 first-request  sector_mfs=768  (32B sector, L1-bypass, shared icnt)
104213 icnt-backpressure requests_issued=49 sector_goal=768   <- blocked after 49 / 768
109916 complete   lat_total=5711 lat_queue=0 lat_issue=1 lat_mem=5710
```

The same "768 individual 32B sector mfs" emission causes three coupled symptoms — all in the mover
loop [tma_unit_sm.cc:614-738](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L614-L738):

**(A) Interconnect injection serialization (from the LATENCY analysis).**
- `kMaxRequestsPerCycle = 2`
  ([tma_unit_sm.h:47](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.h#L47))
  → ≤ 8 sector mfs/cyc even when the icnt is free, so injecting 768 sectors is ≥ 96 cyc best case.
- `m_icnt->full(SECTOR_SIZE, write)`
  ([:653](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L653))
  stops the transfer after ~49 sectors; the SM's single `m_icnt` (shared with ldst,
  [sm.cc:1226-1229](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1226-L1229))
  then dribbles the remaining 719 over ~5,700 cyc. 94 of 106 sampled bwd transfers log a
  back-pressure event.
- **Recent pre-run check:** pure issue throttle is only ~96 cycles for a 768-sector load, while
  sampled load latency is thousands of cycles (fwd mean ~4,103 cyc, bwd mean ~3,529 cyc for SM0
  load transfers). So `kMaxRequestsPerCycle=2` alone cannot explain the gap. 128B emission helps
  only if the dominant cost is TMA-side injection / shared-icnt queueing before the memory
  sub-partition. If `lat_drain` dominates, the bottleneck is downstream.

**(B) 32B×4 sector explosion at L2 admission (from ADDR_MERGE §1-C / §2-2).**
- Each 128B AGU request is emitted as **4 separate 32B sector mfs** instead of one 128B line, so the
  L2 input queue carries 4x the mf count and in-flight duplicates of the same line are not merged at
  the admission stage. L2 MSHR merge exists but the `icnt_L2_queue` admission probes one at a time,
  so `RESERVATION_FAIL` retries accumulate.
- Important caveat: sending one 128B parent mf from TMA does **not** remove sectorization entirely.
  In the current sector-L2 path, `memory_sub_partition::push()` calls
  `breakdown_request_to_sector_requests()` and splits a 128B parent back into 4 x 32B children.
  Therefore 128B emission mainly reduces pressure **before** that split (TMA mover + shared ICNT).
  It may not help if L2 admission / ROP delay / memory partition queues are the real limiter.

**(C) Synthetic-address hotspot across SMs (from ADDR_MERGE §1-A) — real, but not the first fix.**
- AGU base = `(transfer_uid << 20) + agu_index*128`
  ([:633-635](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L633-L635)).
  `transfer_uid` is an **SM-local** counter, so the Nth transfer on every SM hits the **same** L2
  line → an artificial all-SM hotspot → `RESERVATION_FAIL` storms (one sector re-probed 132x over
  cycles 8948–13493). This is the FA3_progress **Arch TODO-2** limitation: the trace cannot supply
  the real GMEM base (NVBit can't read the TMA descriptor cache), so the address is fabricated.
- Current finding from [TMA_ADDR_MERGE_PLAN.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/TMA_ADDR_MERGE_PLAN.md):
  `first_lane_addr` cannot recover the TMA GMEM base, and `(ufid, pc, handle_hi)` cannot uniquely
  select one base because multiple real tensor-map bases collapse into the same normalized config.
  A `config_id`-only mock base can therefore overstate L2 hits; a more realistic spread can instead
  lower L2 hits and increase actual TMA/memory cycles. Treat address modeling as a separate
  validation problem, not as the first cycle-reduction lever.

> **"If there's no real base model, isn't every TMA access just an L2 hit anyway? And wouldn't
> fixing it only *increase* cycles?"** — Important question; the answer is nuanced, and it is why
> 6B is gated rather than assumed to be a win.
>
> 1. **It is NOT always a hit, even today.** Within a single transfer, `agu_index` increments, so
>    the 24 KB load spans many *distinct* 128B lines (`agu_base + agu_index*128`). The first probe
>    of each line is a true MISS to DRAM; only the cross-SM / cross-transfer collisions (same `N`th
>    `transfer_uid` on every SM → same `(N<<20)` base) reuse a line. So the current model is a mix
>    of (a) genuine first-touch misses, (b) **pending-hits** where a second SM's request merges onto
>    the first SM's still-in-flight miss, and (c) RESERVATION_FAIL re-probes when that merge can't be
>    admitted. The `L2_TMA_*` counters now separate exactly these three so we stop guessing.
> 2. **The collision is a double-edged artifact.** The shared synthetic base **inflates L2 hit/
>    pending-hit rate** (fake locality → fewer DRAM trips → fewer cycles) *and at the same time*
>    **creates the RESERVATION_FAIL storm** (one hot line → head-of-line blocking → more cycles).
>    The two effects push cycles in opposite directions, so the net sign is genuinely unknown until
>    measured.
> 3. **Therefore a more realistic address model can move cycles either way.** Spreading the base
>    across SMs removes the RESERVATION_FAIL storm (↓ cycles) but also lowers the fake L2 reuse
>    (↑ cycles, more DRAM traffic). Per the user's "no fake wins" constraint, a 6B result that
>    **raises** cycles while bringing L2 hit rate **down toward HW** is an accuracy *improvement*,
>    not a regression — and a 6B result that lowers cycles by pushing L2 hit rate further **above**
>    HW is rejected. This is precisely why §4 gates 6B on L2-hit realism, not on cycle delta.

**Why this is wrong for HW.** A real Hopper TMA engine issues a bulk descriptor copy pipelined at
high bandwidth, addressing real per-tensor GMEM regions — not 768 serialized 32B injections into
one shared port at one synthetic hotspot. Bandwidth is a supporting signal, but compare like with
like: simulator `DRAM_BW_total` (fwd **15.2** / bwd **44.3 GB/s**) should be checked against NCU
DRAM bytes/sec or DRAM-throughput-derived bandwidth, while NCU's broader `Memory Throughput`
should be checked against simulator L2/TMA-side traffic. The primary evidence remains the direct
TMA logs: inflated `lat_mem` parks consumer warpgroups on their mbarriers →
`wait_barrier`/`tma_flush` SM-idle → IPC ceiling.

## 3. Fix design

Implementation must be staged. Do **not** combine address modeling with the first latency test.

### Opt 6A — latency / injection path

First step is **instrumentation only**, not a behavior change. Expected cycle improvement from the
later behavior change is not guaranteed. It is bounded by the recoverable TMA SM-idle budget
(`wait_barrier` ≈ 10% on both kernels, plus bwd `tma_flush` ≈ 4.7%), and the actual delta may be
much smaller if TMA latency is overlapped or if pressure simply moves downstream.

**Part 0 — observe-only TMA timing decomposition (implemented / next run).**
- Add per-transfer fields to the TMA `complete` event:
  - `lat_to_first_request`: AGU ready -> first mf actually issued
  - `lat_emit`: first issued mf -> last issued mf
  - `lat_drain`: last issued mf -> transfer complete
  - `issue_active_cycles`
  - `icnt_full_cycles`
  - `requests_per_issue_active_cycle`
- Add TMA summary averages:
  - `avg_issue_active_cycles`
  - `avg_icnt_full_cycles`
  - `avg_to_first_request_cycles`
  - `avg_emit_span_cycles`
  - `avg_drain_cycles`
  - `avg_requests_per_issue_active_cycle`
- Decision rule:
  - If `lat_emit` and/or `icnt_full_cycles` dominate, then 128B emission / injection bandwidth is
    likely worth testing.
  - If `lat_drain` dominates, then 128B emission is likely to shift the bottleneck downstream; look
    at L2 admission, memory partition pressure, ROP delay, and address behavior instead.

**Part 1 — optional behavior test: emit one 128B line mf instead of 4×32B sectors.**
- AGU 128B request → a single 128B mf (`data_size=128`, full sector/byte mask). Goal counter
  switches from `kSectorMfGoal = agu_requests*4*mfs_per_sector` to `kLineMfGoal = agu_requests *
  mfs_per_line` (reduction RMW = 2 line mfs: read+write).
- `memory_sub_partition::push` → `breakdown_request_to_sector_requests` splits into 32B children +
  MSHR-merges, exactly like the normal L1→L2 path. Parent 128B kept as `original_mf`; children
  inherit the TMA tag.
- Response path: `fill(mf)` resolves a returning 32B child to its parent via `get_original_mf()`,
  retiring the parent only when all children are back (`m_outstanding_sectors[parent]`); handle the
  L2-bypass case where the 128B parent returns directly (remaining=1).
- This cuts injected mf count before the memory sub-partition by up to 4x. It does **not** guarantee
  a 4x latency drop because the sector L2 path will split the parent back into children. Treat it as
  a gated experiment, not a guaranteed fix.

**Part 2 — injection bandwidth knob (finishes A).**
- Make `kMaxRequestsPerCycle` a config knob (not a hardcoded `2`) and/or give TMA its own injection
  budget so a bulk transfer drains faster. Tune only after Part 1. Target is not "free memory"; it is
  eliminating artificial 32B-mf injection/backpressure. Per-transfer `lat_mem` should move toward
  the rop+dram scale (hundreds of cycles), not collapse to zero.

### Opt 6B — address model (separate gated experiment)

Do this only after 6A establishes the latency/injection delta.

- Re-implement the rolled-back mock-base idea only behind a separate flag. Do **not** use
  `config_id` as a single fixed base for all transfers: that was the Phase-A flaw because it can
  collapse distinct logical tiles and overstate L2 hit rate.
- Since the trace has no tile `coords` (`TMACommand.coords` is never set), any synthetic tile spread
  is a heuristic. It may reduce the artificial cross-SM hotspot, but it may also lower L2 hit rate
  toward HW and **increase** actual TMA/memory cycles. That can be more accurate even if it is not a
  cycle win.
- This is explicitly **not** real-base recovery (that needs trace-gen work; see Arch TODO-2 and
  [tma_tx256b_revert_plan.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/tma_tx256b_revert_plan.md)).
  Accept 6B only if L2 hit rate and TMA/L2 traffic move toward HW without hiding the cost through
  fake locality.

> **Relation to FA3_progress Arch TODO-2 (real TMA base address).** 6B (synthetic spread) and
> [FA3_progress.md Arch TODO-2](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.result/FA3_progress.md)
> are the **same defect at two fidelity levels**. TODO-2 says the TMA unit fabricates
> `agu_base = (transfer_uid << 20) + …` because the trace cannot supply the real descriptor base
> (NVBit can't read the TMA descriptor cache). That single fabricated-base fact is the *root* of
> everything in §2-C: the cross-SM collision, the fake L2 reuse, and the RESERVATION_FAIL storm all
> follow from it.
> - **6B is the cheap heuristic mitigation** (spread the synthetic base to kill the artificial
>   hotspot) — it can make L2 behavior *more* realistic but is still a guess, since real tile
>   `coords` are absent.
> - **TODO-2 is the real fix** (feed the true GMEM base + per-transfer offsets from trace-gen),
>   which is the prerequisite for TODO-1 (SMEM swizzle) and for validating any 6B heuristic against
>   real addresses.
> So if the Part-0 gate points at the address axis, the decision is not just "do 6B" but "is the
> 6B heuristic enough, or do we escalate to the TODO-2 trace-gen work?" — answer that with the
> `L2_TMA_*` realism check, not with the cycle delta.

*Files:* `tma_unit_sm.{h,cc}` (128B emission, parent/child accounting, base helper, injection
knob), `gpu-sim.cc` + `shader.h` (config flags). Reuse the existing `-tma_debug_*` logging.

## 4. Verification plan

1. **Run 6A Part 0 first (instrumentation only)** with current TMA behavior.
   Recommended config for a one-run diagnosis:
   - Keep `-trace_enabled 1`.
   - Keep `TMA` in `-trace_components` and `-trace_sampling_core 0` for SM0 timeline events.
   - Keep scheduler / scoreboard / memory / interconnect traces if one all-purpose diagnosis run is
     preferred.
   - Set `-tma_debug_enable 1` only if all-SM TMA per-event stderr logs are desired. Otherwise,
     `trace_components TMA` is enough for SM0 per-transfer timing.
   - Keep `-l1i_prefetch_debug_enable 0`, `-bar_debug_enable 0`, and `-sync_debug_enable 0` to avoid
     unrelated log explosion.
2. **Analyze the instrumentation gate.**
   - **TMA latency axis (where in the transfer is the time spent):**
     - If `avg_emit_span_cycles` and `avg_icnt_full_cycles` are large relative to
       `avg_drain_cycles`, run the 128B / injection-bandwidth behavior test (Part 1).
     - If `avg_drain_cycles` dominates, skip 128B emission for now and inspect downstream pressure
       (next bullet). 128B emission only relieves the TMA-mover → shared-ICNT → `push()` segment;
       the sector L2 path re-splits a 128B parent into 4×32B inside
       `memory_sub_partition::push()`, so anything past that point is unchanged.
     - If `requests_per_issue_active_cycle` is already near the current theoretical limit while
       `lat_drain` is high, the TMA issue loop is not the primary bottleneck.
   - **TMA L2 admission axis (is the downstream cost a hotspot or genuine latency):** read the new
     TMA-only L2 counters (`L2_TMA_*`, separated from the aggregate `L2_total_cache_*` which mixes
     in normal LDG/STG):
     - `L2_TMA_res_fail_per_probe` **high** AND `L2_TMA_true_hit_rate` (or `pending_hit_rate`)
       above HW (fwd 69.58% / bwd 82.26%) ⇒ **synthetic-address hotspot** (the ADDR_MERGE §1-A
       symptom). This is a **6B address** problem; 6A 128B emission cannot fix it because the
       re-probing items are 32B children regardless of TMA emission format.
     - `L2_TMA_res_fail_per_probe` **low** is **not** conclusive on its own — the hit/miss/res_fail
       counters only advance when the admission probe actually runs. Disambiguate with the two
       head-of-line backpressure counters before concluding:
       - low res_fail **and** low `L2_TMA_output_full_cycles` / `L2_TMA_port_busy_cycles` (and low
         `gpu_stall_dramfull`) ⇒ admission is genuinely not the limiter; the `lat_drain` is real
         memory round-trip. Neither 6A nor 6B helps, and shaving it would be a fake win.
       - low res_fail **but** high `L2_TMA_output_full_cycles` / `L2_TMA_port_busy_cycles` ⇒ the TMA
         head is jammed by **downstream reply/port backpressure**, a separate fix axis (reply-queue
         depth / port width), not the address hotspot and not 6A injection.
       - This is the four-way head-of-line split (dram-queue-full via `gpu_stall_dramfull`,
         reply-queue-full, port-busy, reservation-fail) that lets a single Part-0 run decide the
         direction without a follow-up run.
     - A high `L2_TMA_pending_hit_rate` specifically (vs `true_hit_rate`) is the fingerprint of the
       cross-SM single-base collision: many requests merging onto one in-flight miss line. See the
       "always an L2 hit?" note in §2-C.
3. **Only if the gate passes, run 6A Part 1** with address behavior unchanged: confirm
   per-transfer `lat_emit` / `icnt_full_cycles` drops, `wait_barrier` and bwd `tma_flush` SM-idle
   fall, and `nv_ibuffer_empty` stays flat.
4. **HW alignment cross-check for any 6A behavior run:**
   - sim `DRAM_BW_total` → compare against NCU DRAM bytes/sec or DRAM-throughput-derived bandwidth
   - simulator L2/TMA traffic → compare against NCU memory/L2-side throughput where available
   - sim **L2 Hit Rate should not become less realistic**. With address unchanged, large L2-hit
     changes are suspicious and likely indicate accounting bugs in the 128B parent/child path.
   - sim per-scheduler IPC → toward HW 0.405 / 0.285.
5. **Run 6B only after 6A**. Gate 6B on L2 hit rate staying near HW (fwd 69.58% / bwd 82.26%) and on
   no fake `line-reuse` collapse. A cycle regression is acceptable if it is the result of more
   realistic L2 locality; a cycle win is rejected if L2 hit rate balloons above HW.
6. Record cycle deltas vs Opt 5 (fwd 149,727 / bwd 241,425) and vs 6A in `FA3_progress.md`.

## 5. Rollback history & risks (why Phase A failed, must not repeat)

- **Phase A was rolled back and never committed** (no `tma_mock_config_base`/`kLineMfGoal`/
  `m_outstanding_sectors` anywhere in the tree; git shows only the `-tma_debug_*` logging landed).
  The runs analyzed here (`.o23`/`.o5`) are the **pre-Phase-A** code (`32B sector`, `sector_mfs=768`,
  synthetic `transfer_uid` base).
- **Risk 1 — L2-hit over-statement (the main Phase-A flaw).** A `config_id`-only fixed base
  collapses all tiles of a tensor onto one region → L2 hit rate balloons above HW, giving a fake
  cycle win. This is why address modeling is split into 6B and gated on HW L2 hit rate.
- **Risk 2 — 128B↔32B accounting.** Parent(1)↔children(4) retire bookkeeping and the reduce/store
  RMW path (2 line mfs) are error-prone; assert children-complete == 4 and that parents retire
  exactly once.
- **Risk 3 — don't make memory free.** The goal is to remove *injection* serialization, not memory
  pressure; keep rop/L2/DRAM accounting intact and validate against HW DRAM and L2/TMA-side metrics,
  not just cycles.
- **Risk 4 — address realism can increase cycles.** Current sim L2 hit rate is already higher than
  HW. A better address model may increase actual TMA/memory latency by reducing fake locality. That
  is an accuracy improvement, not a failed 6B result, but it must not be mixed into the 6A latency
  experiment.
- **Out of scope:** real GMEM base recovery (trace-gen / NVBit descriptor-cache limitation, Arch
  TODO-2). Tracked separately; this plan deliberately uses a hotspot-free synthetic address only.
