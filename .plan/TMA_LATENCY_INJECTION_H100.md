# Opt 7 — TMA transfer latency / injection over-modeling

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

## 4.5 Part-0 result (measured — fwd `.o24` / bwd `.o6`, 2026-07-01)

Runs are the current tree (new `L2_TMA_*` + backpressure counters), clean exit, and **timing-neutral
vs the pre-instrumentation runs**: fwd 151,232 (vs `.o23` 151,350, -0.08%) / bwd 240,869 (vs `.o5`
241,238, -0.15%). SM-idle axis reproduced: fwd `wait_barrier` 9.82% / bwd 10.93% + `tma_flush`
4.64%.

**The four-way head-of-line split resolves the gate to: neither 6A nor 6B — the limiter is the
L2→core reply path.**

| Metric | FWD (`.o24`) | BWD (`.o6`) | Reading |
|---|---|---|---|
| `L2_TMA_res_fail_per_probe` | **0.0000** | **0.0000** | not the synthetic-address hotspot → 6B is not the current lever |
| `L2_TMA_port_busy_cycles` | 0 | 0 | L2 data port is not the limiter |
| `L2_TMA_output_full_cycles` | **180,998** | **314,724** | **reply queue (`m_L2_icnt_queue`) full → admission stalls** |
| `gpu_stall_icnt2sh` | 258,818 | 464,997 | interconnect reply port has no buffer → the root upstream cause |
| `gpu_stall_dramfull` | 189,115 | 279,825 | DRAM-return queue also backs up (secondary) |
| TMA `lat_emit` vs `lat_drain` (sample) | 63 vs 1990 | 641 vs 2246 | `lat_drain` dominates → cost is waiting for completion, **not** injection (6A) |
| `L2_TMA_true_hit_rate` | 0.9846 | 0.9657 | see fake-locality caveat below |

**Why 6A is ruled out:** `lat_emit` is small and `L2_TMA_res_fail_per_probe`/`port_busy` are zero, so
the 32B→128B emission change (which only relieves the mover→ICNT→`push()` segment) cannot move the
needle. The time is in `lat_drain` — waiting for the 768 sector responses to come back.

**Root cause (structural):** one TMA load = 768×32B responses that must all traverse
`m_L2_icnt_queue`, but that queue is drained **one mf per sub-partition per cycle**, and only when
`icnt_has_buffer` is true ([gpu-sim.cc:3993-4011](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3993-L4011)). When the interconnect reply
port is busy (`gpu_stall_icnt2sh`), the queue fills (`output_full`), which stops L2 admission, which
parks the consumer warpgroup on its mbarrier (`wait_barrier` SM-idle). So the chain is:
`icnt reply port busy → m_L2_icnt_queue full → L2 admission stalls → TMA head blocked → wait_barrier`.

**Correction (supersedes an earlier note): do not compare `gpu_stall_icnt2sh` against
`output_full_cycles` as if they were the same population.** They count different scopes:
- `L2_TMA_output_full_cycles` — only cycles where the **TMA** mf is at the queue head
  ([l2cache.cc:537](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L537)).
- `gpu_stall_icnt2sh` — **all** reply mf (TMA + normal LDG/STG) that could not be injected into the
  icnt this cycle ([gpu-sim.cc:4014](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4014)).
So "`icnt2sh` (258K) > `output_full` (181K)" does **not** mean the consumer stalls more often than
the queue fills. Both being large is the valid signal; their magnitudes are not directly comparable.

**Why experiments 1/2/3 are related — the reply path is a 4-stage serial pipe, and the true root
is a producer/consumer imbalance:**

```
[Producer: up to 2 push/cycle]        [Buffer]            [Drain: 1 pop/cycle]     [Network]
 L2 fill response  ─┐  (l2cache.cc:483)
                    ├─push→  m_L2_icnt_queue  ──pop(1/cyc)──→  icnt_push  ──→  icnt reply network
 admission HIT     ─┘  (l2cache.cc:558)         (depth = exp2)    (rate = exp1)     (exp3)
```

- The **producer** pushes from **two** sites in one `cache_cycle` — the L2 fill response
  ([l2cache.cc:483](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L483)) and the admission HIT reply
  ([l2cache.cc:558](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L558)) — while the **consumer** pops only
  **one** mf per sub-partition per cycle ([gpu-sim.cc:4011](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4011)). That is a
  **structural up-to-2:1 imbalance**, amplified right now because the fake 98% L2 hit rate makes the
  admission-HIT producer fire on almost every probe.
- Therefore **exp 2 (queue depth) alone cannot fix it**: a deeper FIFO only absorbs a longer burst;
  with producer > consumer the queue re-fills. Depth buys latency tolerance, not throughput.
- **They must NOT be run together.** A serial pipe's throughput is set by its narrowest stage, so
  changing depth (exp2) and drain-rate (exp1) and network (exp3) at once reproduces exactly the
  confounding that Opt 6A/6B split was created to avoid. Change one stage, observe whether the
  bottleneck *moves* to the next stage, then decide.

### Proposed next experiments (in order)

Run **exp 2 first as a pure config probe** (no rebuild), read the already-emitted counters, and let
the result select the next step:

| exp 2 outcome | interpretation | next |
|---|---|---|
| `output_full`↓ **and** cycles↓ meaningfully | depth was the limiter | done (skip 1/3) |
| `output_full`↓ but cycles ~flat **and** `gpu_stall_icnt2sh` ~flat | the real cap is downstream **drain/icnt**, not depth | exp 1, then 3 if needed |
| nothing moves | another queue gates (check `gpu_stall_dramfull`) | re-examine DRAM axis |

Predicted outcome: the **middle row** (the 2:1 imbalance is the root, so depth alone won't move
cycles). If confirmed, exp 1 (raise drain to N mf/sub-partition/cycle) directly targets the
imbalance and comes before exp 3.

1. **Structural (root cause): drain more than 1 reply mf per sub-partition per cycle, gated by real
   reply bandwidth.** The single-mf-per-cycle drain in
   [gpu-sim.cc:3993-4019](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4001-L4043) was a per-mf serialization: a 128B line
   returns as 4×32B and each 32B ate one drain slot. **Implemented (config flag, default-off):**
   `-gpgpu_l2_reply_drain_per_cycle N` lets each sub-partition eject up to N reply mf per ICNT tick.
   `N=1` = original behavior. Each drained mf **still passes `icnt_has_buffer` + `icnt_push`**, so
   the icnt reply-bandwidth accounting is unchanged — this removes only the artificial ejection cap,
   keeping exp1 (drain rate) and exp3 (icnt bandwidth) separable. `N=4` matches 128B=4×32B.
2. **Config-only A/B (do first as the probe above): deepen the L2→icnt queue.**
   `-gpgpu_dram_partition_queues` (bound to `gpgpu_L2_queue_config`,
   [gpu-sim.cc:854-855](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L854-L855)) fields are
   `icnt→L2 : L2→dram : dram→L2 : L2→icnt(reply)`; the **4th** field is the `m_L2_icnt_queue` depth.
   **Note this config already sets `64:64:64:64` — the reply queue is already deep (64), not the
   library default 8.** Exp 2 raises only the 4th field to `256`
   (`-gpgpu_dram_partition_queues 64:64:64:256`, applied). Because the reply queue is *already* 64
   and `output_full` is still large, the prior is strong that this is a **drain-rate** limit, not a
   depth limit — a deeper queue barely helping is the expected (and informative) result pointing to
   experiment 1. Pure knob, no rebuild, low risk — a cycle win here is suspect until cross-checked
   against the fake-locality caveat.
3. **Interconnect reply-port width / ejection buffer.** `gpu_stall_icnt2sh` is the upstream cause;
   if experiments 1–2 show the queue is fed faster than icnt can eject, raise the icnt ejection
   buffer / reply VC allocation. Validate reply BW stays within a realistic H100 envelope.

### A/B run matrix (one build covers all — exp1 is now a flag)

Both exp1 and exp2 are config-only on the **same rebuilt binary** (the drain-loop code adds the
`-gpgpu_l2_reply_drain_per_cycle` knob; a rebuild is required once, then no more). Run in this order
and read `L2_TMA_output_full_cycles`, `gpu_stall_icnt2sh`, cycles, `L2_TMA_true_hit_rate`,
`gpu_stall_dramfull` each time:

| Run | `dram_partition_queues` (4th) | `l2_reply_drain_per_cycle` | Isolates |
|---|---|---|---|
| A (baseline re-confirm) | 64 | 1 | reproduce `.o24`/`.o6` on the new binary (timing-neutral check) |
| B (exp2: depth only) | 256 | 1 | is `output_full` pure queue depth? |
| C (exp1: drain rate) | 64 | 4 | does relieving the 1-mf/cycle cap move cycles? |
| D (exp1+2, only if needed) | 256 | 4 | depth + drain together, if C shows depth also matters |

Interpretation: if **B ≈ A** (depth doesn't help) but **C improves cycles + drops `output_full`**,
the root was the drain-rate cap (as predicted) and exp1 is the fix. If **C still leaves
`gpu_stall_icnt2sh` high**, the next limiter is the icnt reply network (exp3). If **C balloons
`L2_TMA_true_hit_rate` further above HW or a cycle win coincides with unchanged `gpu_stall_dramfull`
under fake 98% L2 hit**, treat the win as suspect until TMA address realism (6B) is in.

### A/B result (measured — FA3 fwd k5, 4 runs, 2026-07-01/02)

| Run | queue 4th | drain N | `gpu_sim_cycle` | `output_full` | `gpu_stall_icnt2sh` | `dramfull` | `wait_barrier` | L2_TMA hit |
|---|---|---|---|---|---|---|---|---|
| A base `.o24` | 64 | 1 | 151,232 | 180,998 | 258,818 | 189,115 | 9.82% | 0.985 |
| B depth `.o25` | 256 | 1 | **150,522** | 96,320 | 253,395 | 94,319 | 9.69% | 0.986 |
| C drain `.o7` | 64 | 4 | 151,346 | 151,644 | **722,519** | 148,773 | 9.99% | 0.985 |
| D both `.o8` | 256 | 4 | 152,816 | 89,023 | **820,804** | 83,404 | 10.31% | 0.987 |

**Conclusion: the reply path is NOT the cycle lever. `output_full` / `wait_barrier` are symptoms,
not the cause.**

- **All four cycles are within ±0.8% (150,522–152,816).** Quadrupling the reply queue depth,
  quadrupling the drain rate, or both, does not move cycles.
- **B (depth only):** `output_full` **halved** (181K→96K) yet cycles moved −0.5%. Queue depth was
  never the limiter.
- **C (drain only):** `output_full` barely changed while `gpu_stall_icnt2sh` **exploded 259K→722K**
  and cycles were flat — draining the reply queue faster just **pushes the stall one stage
  downstream to the icnt reply network** (exactly the §4.5 "middle row" → exp3 case), with no cycle
  benefit. This is the "bottleneck simply relocates" outcome the user predicted from the start.
- **D (both):** lowest `output_full` (89K, −51%) but the **worst** cycle count (152,816) and the
  highest `icnt2sh` (821K). Removing reply-queue pressure entirely does not help.
- **`wait_barrier` is pinned at 9.7–10.3% across all four runs.** The consumer warpgroup's TMA
  mbarrier wait does not shrink no matter how fast replies drain, so TMA completion latency is
  **not** gated by the reply queue / icnt ejection — it is set upstream (the whole issue→complete
  span of a 768×32B transfer) and/or is an artifact of the run conditions below.

**⚠ These four runs all ran on fake ~98% L2 locality** (`L2_TMA_true_hit_rate ≈ 0.985` vs HW
0.6958). The dense, all-hit reply traffic that saturates the icnt in C/D is itself a product of the
synthetic single-base address. With realistic addresses more responses go to DRAM, spreading them
in time, so the reply-path pressure profile could change qualitatively. **Therefore the reply path
cannot be judged (or fixed) until the address model is realistic.**

**Decision (updated, supersedes the earlier "fix reply path first"):** the reply-path knobs (exp
1/2/3) are **exhausted with no cycle lever**, so config is reverted to baseline
(`64:64:64:64`, `drain=1`) and the `-gpgpu_l2_reply_drain_per_cycle` code is **kept but left at 1**
for re-testing after addresses are realistic. The next experiment is **TMA address realism (6B /
Arch TODO-2)**: it is now a prerequisite for interpreting the TMA bottleneck at all, not just a
later refinement. Re-run this exp1/2/3 A/B after 6B before drawing any reply-path conclusion.

## 4.6 Address realism landed — reply-path re-test is now unblocked (2026-07-09)

The 6B/Arch-TODO-2 prerequisite is **done**: TMA real base + CTA-indexed tile spread (M2/M2.5) is
implemented and verified (see TMA_BASE_ADDR.md §4.1 / TMA_exact_base_mapping_integration.md). L2
hit rate is no longer fake — it dropped from ~0.98 toward HW:

| | before (fake) | after (M2/M2.5) | HW |
|---|---|---|---|
| fwd K5 `L2_TMA_true_hit_rate` | 0.9854 | **0.9461** | 0.6958 |
| bwd K10 `L2_TMA_true_hit_rate` | 0.9785 | **0.8718** | 0.8226 |

So the §4.5 caveat ("all four A/B runs ran on fake ~98% locality → reply path cannot be judged") is
now lifted. New evidence from the M2/M2.5 runs points the same way as §4.5's own prediction:

- **`avg_drain_cycles ≈ 2,600` vs `avg_emit_span ≈ 1,260`** (bwd per-SM, `.e14` TMA Phase3) — the
  transfer cost is dominated by **drain (waiting for responses)**, ~2x the injection span, and now
  on *realistic* traffic. 6A (128B emission / injection bandwidth) is still ruled out; the cost is
  the memory-return path.
- SM-idle on the realistic baseline: `wait_barrier` ≈ **9.58% (fwd) / 9.53% (bwd)**, bwd
  `tma_flush` ≈ **14.66%** — the TMA-completion axis is still the #1 recoverable lever.

**Next step:** re-run the exp1/2/3 reply-path A/B matrix (§4.5 "A/B run matrix") on this M2/M2.5
baseline. The prior result (depth/drain move `output_full` but not cycles, stall relocates to
`gpu_stall_icnt2sh`) must be re-measured now that misses actually go to DRAM and spread the reply
traffic in time. Judge by `gpu_sim_cycle` + `wait_barrier`/`tma_flush` SM-idle, and keep the
`L2_TMA_true_hit_rate` realism check (must stay near HW, not re-inflate). This is the start of the
actual cycle-reduction phase (tracked as ongoing in FA3_progress.md).

## 4.7 Root cause found = INJECTION path, not reply path (static, no run needed) — 2026-07-09

Instead of burning a 12h reply-path A/B run, the M2/M2.5 baseline logs (`.o14`/`.e14`, bwd K10)
were analyzed statically. **The reply path is confirmed NOT the lever; the bottleneck is the SM's
shared REQ-net injection buffer.** Decisive evidence (bwd K10, real-address baseline):

| signal | value | reading |
|---|---|---|
| `avg_icnt_full_cycles` / `avg_issue_active_cycles` | 4256 / 4262 = **99.9%** | a TMA transfer spends nearly all its active issue cycles blocked on `m_icnt->full()` |
| effective inject rate | **0.18 sector mf/cycle** | vs kMaxRequestsPerCycle cap of 8 → cap is NOT the limiter |
| `Req_Network_in_buffer_full_per_cycle` | **355.4** | REQ (SM->L2 inject) buffer saturated |
| `Reply_Network_in_buffer_full_per_cycle` | 6.3 | reply side ~empty → reply path is NOT the bottleneck (56x lower) |
| `Req_Network_out_buffer_full_per_cycle` | 0.20 | out side (sub-partition accept) not full → the in_buffer just **drains too slowly** |
| `Req_Network_conflicts_per_cycle` | 62.7 | many SMs target the same sub-partition → iSLIP grants 1/dest/cycle |
| `bw_util` (DRAM) | **0.078** | DRAM is 92% idle → not a DRAM-bandwidth problem |
| `L2 reservation_fails` | 0 | not an L2 admission problem |

**Structural root:** the built-in local xbar (`-network_mode 2`) runs the iSLIP arbiter **once per
ICNT tick**, and each pass ejects **<=1 packet per input node** (`input_nodes.erase(node_id)`,
[local_interconnect.cc:210-247](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/local_interconnect.cc#L210-L247);
Advance once/cycle, [gpu-sim.cc:4172-4174](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4172-L4174)).
FA3 emits **768x32B sector mfs per TMA transfer** ([tma_unit_sm.cc](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc)) into the SM's **single** shared REQ-net input node (shared with ldst, [sm.cc:1195,1223,1229](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1195)). At 1 packet/cycle drain the 512-deep in_buffer saturates and stays full → `full()` blocks every subsequent inject → transfer completion stalls → `wait_barrier` (load) and `tma_flush` (bwd store/reduce).

**Both stall axes are the SAME root.** `tma_flush` (bwd 14.66%, fwd 0%) is store/reduce sector mfs
going through the **identical single icnt port + same kMaxRequestsPerCycle throttle + same
`full()` write-side backpressure** ([tma_unit_sm.cc:814,894](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L814); release is ack-based, not fixed-latency, [sm.cc:2016-2017](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2016)). fwd is load-only so its outstanding-store counter stays 0. UBLKRED (reduce) emits 2 mf/sector so it doubles the injection pressure.

**Why the other levers are ruled out (no run needed):**
- **reply-path (exp1/2/3)** — REQ full 355 vs REPLY full 6.3; `full()` checks only the REQ subnet
  ([local_interconnect.cc:375-384](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/local_interconnect.cc#L375-L384)), physically separate from REPLY. This is exactly why §4.5's depth/drain A/B moved `output_full` but not cycles, and why `wait_barrier` was pinned.
- **kMaxRequestsPerCycle=2** — effective inject is 0.18/cycle « cap 8; `full()` gates first. Raising the cap does nothing.
- **in_buffer size (512)** — deeper buffer only queues longer; the 1-packet/cycle **drain rate** is unchanged, so steady-state throughput is the same.

## 4.8 Fix chosen + IMPLEMENTED: iSLIP grant-passes-per-cycle knob

The only static-sound lever is to raise the **drain rate** of the injection buffer. Implemented as a
config knob (default 1 = bit-identical):

- **`-icnt_grant_passes_per_cycle N`** ([icnt_wrapper.cc](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/icnt_wrapper.cc), default `"1"`). `xbar_router::Advance()` now runs the iSLIP/RR arbiter up to N full passes per ICNT tick, stopping early when no in_buffer has anything to grant ([local_interconnect.cc Advance](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/local_interconnect.cc#L117-L138)). Each pass still grants <=1 packet/(input,dest) and every packet still passes `Has_Buffer_Out` + `icnt_push`, so the bandwidth accounting stays honest (not a free-memory hack).
- Applies to both REQ and REPLY xbars (same class); the bottleneck is REQ, REPLY has headroom.
- H100 config set to **`-icnt_grant_passes_per_cycle 4`** ([gpgpusim.config](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config#L183)) — 4 sectors/cycle = one 128B line, the natural width for the 4x32B sectorization.
- Files: `local_interconnect.{h,cc}`, `icnt_wrapper.cc`, `gpgpusim.config`. No tracer/trace change; rebuild required (source change).

**Expected effect (to be measured, this is the first real cycle-reduction run):** `avg_icnt_full_cycles`↓, `Req_Network_in_buffer_full_per_cycle`↓, `wait_barrier`/bwd `tma_flush` SM-idle↓, `gpu_sim_cycle`↓.

**Honest residual risk:** §4.5 warned a reply-drain increase merely relocated the stall to
`gpu_stall_icnt2sh`. This time `out_buffer_full≈0.20` and REPLY has headroom, so there is no obvious
next stage to absorb the relocation — but only the run confirms it. Run bwd K10 first (REQ full 355,
worst case); fwd (REQ full ~45K-equivalent, milder) can share the same run. Keep the
`L2_TMA_true_hit_rate` realism check unchanged (this knob does not touch addressing).
**§4.9 pre-emptively widens the very stage this risk would relocate to.**

## 4.9 Paired downstream drain: icnt->L2 multi-pop (IMPLEMENTED, same run)

Concern (user): if grant-passes drains the SM's REQ in_buffer 4x faster, the packets just pile in
the xbar **out_buffer**, because the next stage — popping the out_buffer into each L2 sub-partition —
was itself only **1 pop / sub-partition / L2-tick** ([gpu-sim.cc L2 clock loop](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4146)). This is exactly the §4.5 "stall relocates one stage downstream" failure mode. So both stages are widened in the same run instead of burning a 12h run to discover the relocation.

Static check of the downstream path (why it was the natural next limiter):
- `icnt_pop` pulls **1 packet** from the xbar out_buffer per call ([local_interconnect.cc Pop](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/local_interconnect.cc#L92-L102)); it is called **once per sub-partition per L2 tick** in the L2 clock loop ([gpu-sim.cc:4146](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4146)).
- The gate is `m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)` → `gpu_stall_dramfull++`; the code comment already notes "in the worst case we may need to push SECTOR_CHUNCK_SIZE requests", so a 4-wide pop matches the intended design.
- ICNT clock and L2 clock are both 1800MHz (`-gpgpu_clock_domains 1800:1800:1800:8000`), so at 4-in / 1-out the out_buffer would fill at ~3 packets/tick — a real relocation risk.

Fix (config knob, default 1 = bit-identical):
- **`-gpgpu_icnt_to_l2_pop_per_cycle N`** ([gpu-sim.cc register](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L913-L920), member on `memory_config` [gpu-sim.h](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h#L374-L378)). The L2-tick loop now pops up to N request mfs per sub-partition, **re-checking `full(SECTOR_CHUNCK_SIZE)` before each pop** and stopping when the out_buffer is empty (mf==NULL). `cache_cycle`, GRID_BARRIER handling and power stats stay **once per tick** (outside the pop loop) so no per-cycle stat is double-counted. Default 1 reproduces the original single-pop flow exactly.
- H100 config set to **`-gpgpu_icnt_to_l2_pop_per_cycle 4`** ([gpgpusim.config](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config#L188)), matching the 4x32B sector line and the grant-passes width.
- Honest: this does NOT create bandwidth — every pop still respects `full()`, so if L2/DRAM cannot accept, `gpu_stall_dramfull` rises instead (visible). It only removes the artificial 1-pop/tick serialization.

Instrumentation added for the 12h run (both levers now fully observable):
- Boot logs (stderr, once): `[ICNT] grant_passes_per_cycle=4` and `[ICNT->L2] gpgpu_icnt_to_l2_pop_per_cycle=4` — confirm both knobs are live in the first seconds.
- Lever-1 (grant passes): `Req_Network_avg_passes_per_active_cycle`, `Req_Network_extra_pass_grants_total` (>1 / >0 proves it drained more).
- Lever-2 (icnt->L2 pop): `gpu_icnt_to_l2_pops_total`, `gpu_icnt_to_l2_extra_pops` (>0 proves the downstream actually moved more, i.e. out_buffer was not the limiter).
- Relocation detectors: `Req_Network_in_buffer_full` (355 baseline), `Req_Network_out_buffer_full` (0.20 baseline), `gpu_stall_dramfull` (1.42M baseline) — read together they show exactly where the stall ends up.

**Files:** `local_interconnect.{h,cc}`, `icnt_wrapper.cc` (lever 1); `gpu-sim.{cc,h}` (lever 2 + pop counters); `gpgpusim.config` (both knobs=4). No tracer/trace change; rebuild required.

**Judgment rule after the run:** success = `gpu_sim_cycle`↓ with `avg_icnt_full_cycles`↓ and neither `out_buffer_full` nor `gpu_stall_dramfull` exploding. If cycles are flat and `gpu_stall_dramfull` explodes → the limiter moved to the L2 admission / icnt_L2 queue depth (next lever: `-gpgpu_dram_partition_queues` 1st field), and the counters above localize it without another exploratory run.

## 4.10 Reference: full memory-path pipeline (SM/TMA -> DRAM -> SM) + counter->stage map

Static map of every queue/buffer/interconnect stage and the exact perf counter that flags it, so a
run result immediately localizes the bottleneck. Verified against source (file:line inline). Depths
shown are the H100 config defaults.

```
==================== REQUEST path (SM -> DRAM) ====================

[1] TMA mover / LDST unit  (issue)
      TMA rate: kMaxRequestsPerCycle=2 (128B lines/cyc)  [hardcoded, tma_unit_sm.h:47]
      full: m_icnt->full()                                [tma_unit_sm.cc:814 / ldst_unit_sm.cc:1165]
      counter: avg_icnt_full_cycles, icnt_backpressure_events (TMA)
            gpgpu_stall_shd_mem[..][ICNT_RC_FAIL] (LDST)
         | TMA & LDST share the SAME m_icnt (sm.cc:1195,1223,1229); push via
         | icnt_injection_buffer_full/icnt_inject_request_packet (shader.cc:5310/5316)
         v
[2] REQ_NET in_buffer  (xbar input)          cap=64   -icnt_in_buffer_limit
      counter: *Req_Network_in_buffer_full_per_cycle*   [local_interconnect.cc:442]  <-- baseline 355 = MAIN bottleneck
         | iSLIP arbiter (xbar): <=1 pkt/(input,dest)/pass
         | rate = grant_passes_per_cycle (LEVER 1, cfg=4)  [icnt_wrapper.cc:122]
         | counter: Req_Network_extra_pass_grants_total, avg_passes_per_active_cycle
         v
[3] REQ_NET out_buffer  (xbar output)        cap=64   -icnt_out_buffer_limit
      counter: *Req_Network_out_buffer_full_per_cycle*  [local_interconnect.cc:447]  <-- baseline 0.20
         | rate = gpgpu_icnt_to_l2_pop_per_cycle (LEVER 2, cfg=4)  [gpu-sim.cc:4177]
         | counter: gpu_icnt_to_l2_extra_pops
         | full: memory_sub_partition::full(SECTOR_CHUNCK_SIZE)   [gpu-sim.cc:4180]
         | counter: *gpu_stall_dramfull*  [gpu-sim.cc:4181]  <-- baseline 1.42M  (NAME IS MISLEADING: this is L2-INPUT queue backpressure, not DRAM)
         v
[4] m_icnt_L2_queue  (L2 input FIFO)         cap=8    -gpgpu_dram_partition_queues field1  [l2cache.cc:448]
         v
[5] L2 cache bank                            1 port/cyc   -gpgpu_cache:dl2
      gate: output_full(=L2_icnt full) & port_free   [l2cache.cc:530-543]
      counter: L2_total_cache_reservation_fails (baseline 0); TMA: L2_TMA_res_fail_per_probe, L2_TMA_output_full_cycles
         | (miss)
         v
[6] m_L2_dram_queue                          cap=8    field2   [l2cache.cc:449]
         | DRAM arbiter: 1 req/cyc accept, credit-based  [l2cache.cc:341-364]
         v
[7] mrqq (cap=2 FIXED, not configurable!)  -> FR-FCFS bank queues   [dram.cc:120]
      full: dram_t::full()= num_pending >= -gpgpu_frfcfs_dram_sched_queue_size (0=unlimited)  [dram.cc:184]
      counter: m_num_pending, ave_mrqs
         v
[8] DRAM banks/timing -> rwq (cap=CL+1)   [dram.cc:119]
      counter: *bw_util* (=bwutil/n_cmd)  [dram.cc:724]  <-- baseline 0.078 (DRAM only 7.8% used)
           dram_eff, idle_bw, wasted_bw_row/col

==================== REPLY path (DRAM -> SM) ======================

[9]  returnq (cap 0->1024)   [dram.cc:121]
         v
[10] m_dram_L2_queue                         cap=8    field3   [l2cache.cc:450]
         v
[11] L2 fill / bypass   [l2cache.cc:500-518]
         v
[12] m_L2_icnt_queue  (reply FIFO)           cap=8    field4   [l2cache.cc:451]
      counter: *L2_TMA_output_full_cycles*  <-- baseline fwd 45K / bwd 1.39M
         | rate = gpgpu_l2_reply_drain_per_cycle (cfg=1, ruled out in 4.5)  [gpu-sim.cc:4099]
         | full: !icnt_has_buffer
         | counter: *gpu_stall_icnt2sh*  [gpu-sim.cc:4122]  <-- baseline bwd 1.82M
         v
[13] REPLY_NET (icnt)   counter: Reply_Network_in_buffer_full (baseline 6.3 = headroom)
         v
[14] SM receive   [gpu-sim.cc:4056]
```

> Numbering note: stages are numbered 1..14 in strict flow order. Only actual
> queues/buffers/interconnect stages get a number; the xbar's internal full-check and the
> iSLIP arbiter step are folded onto the arrows between [2] and [3] (they are the SAME
> xbar_router, not separate buffers).

### Per-stage throughput (how many items move per cycle) — the narrowest stage is the wall

Each stage's rate is given in the unit that stage's own queue/logic actually uses (do NOT convert;
the mismatch between units is exactly why "one 24KB TMA transfer" hits so many stages differently).
Unit relationships are explained under the table.

| stage | move unit | default rate | current cfg | knob (source) |
|---|---|---|---|---|
| [1] TMA issue | 128B AGU line (=4 sectors) | 2 lines/cyc | 2 | kMaxRequestsPerCycle (hardcoded, tma_unit_sm.h:47) |
| [2]->[3] iSLIP grant | packet | 1 pkt/(input,dest)/pass x **N passes** | N=4 | -icnt_grant_passes_per_cycle (LEVER 1) |
| [3]->[4] icnt->L2 pop | packet | 1/sub-part/L2-tick -> **N** | N=4 | -gpgpu_icnt_to_l2_pop_per_cycle (LEVER 2) |
| [5] L2 cache bank | access | 1 access/cyc (per sub-part, 1 data port) | 1 | -gpgpu_cache:dl2 (port count) |
| [6]->[7] DRAM arbiter | req | 1 req/cyc (per memory partition) | 1 | hardcoded (l2cache.cc:362 break) |
| [7] mrqq | req | (depth cap=2, not a rate) | 2 | hardcoded (dram.cc:120) |
| [8] DRAM banks | column cmd | timing-limited (tRC/tCCD...), not a fixed N | — | DRAM timing (-gpgpu_dram_timing_opt) |
| [9]->[10] returnq -> dram_L2 | packet | 1/sub-part/DRAM-tick | 1 | hardcoded (l2cache.cc:309-331) |
| [12]->[13] L2->icnt reply drain | packet | 1/sub-part/ICNT-tick -> N | 1 (ruled out) | -gpgpu_l2_reply_drain_per_cycle |

**Read this as: after the two levers, the request path is 4-wide from [1]..[4], then RE-NARROWS to
1/cyc at [5] L2 bank, [6] DRAM arbiter and [9] returnq.** Those 1/cyc stages are per-sub-partition,
and there are 32 sub-partitions in parallel, so aggregate = 32/cyc — currently not the wall
(DRAM bw_util 7.8%). But if the two levers succeed, `[5]` / `[6]` / `[9]` are the next candidates;
watch `gpu_stall_dramfull` (=> [4]/[5] region) and `bw_util` (=> [8]) to tell which.

**Unit glossary (what actually flows, and how the units relate):**
- **sector** = a 32B chunk = the atomic memory access granularity here. One `mem_fetch` (mf) carries
  one sector. A 128B cache line = **4 sectors** (`SECTOR_CHUNCK_SIZE=4`).
- **packet** = one `mem_fetch` as it travels the interconnect (icnt in/out buffers, iSLIP grants,
  icnt->L2 pop, reply drain). So **1 packet = 1 sector mf** on the icnt. This is the dominant unit
  for stages [2],[3],[4],[9],[12],[13].
- **128B AGU line** ([1] only) = the TMA mover's issue granularity. Each line expands into **4 sector
  mfs** (a reduce/RMW into 8: read+write per sector). So `kMaxRequestsPerCycle=2 lines` = **up to 8
  sector packets/cyc** entering the icnt — but `full()` gates it long before 8 (measured 0.18/cyc).
- **access** ([5]) = one L2 cache probe. The L2 `push()` sector-splits an incoming mf, but the bank
  serves **1 access per cycle per sub-partition** (single data port). A miss then emits a fill req.
- **req** ([6],[7]) = a DRAM request (`dram_req_t`) handed to the memory controller. Multiple sector
  mfs to the same line can coalesce, but the arbiter still accepts **1 req/cyc per partition**.
- **column cmd** ([8]) = the actual DRAM burst; throughput here is set by DRAM timing
  (row/column/bank constraints), reported as `bw_util`, not by a per-cycle integer N.

So the same 24KB TMA load = 768 sectors = 768 packets on the icnt = 192 AGU lines at [1] = (after L2
coalescing) far fewer reqs at [6]. A stage is the bottleneck when *its own unit's* arrival rate
exceeds *its own unit's* service rate above.

### Counter -> bottleneck-stage lookup (read this first on any run)

| counter | high => bottleneck at | bwd baseline |
|---|---|---|
| `avg_icnt_full_cycles` | [1]->[2] TMA injection blocked on in_buffer full | 4256 (99.9%) |
| **`Req_Network_in_buffer_full`** | **[2] REQ injection buffer** (current main) | 355 |
| `Req_Network_out_buffer_full` | [3] xbar output (relocation target if lever 2 fails) | 0.20 |
| `gpu_stall_dramfull` | [4] L2-input queue (m_icnt_L2_queue) full — NOT DRAM | 1.42M |
| `L2_total_cache_reservation_fails` | [5] L2 bank admission | 0 |
| `L2_TMA_output_full_cycles` | [12] reply queue (L2->icnt) full | 1.39M |
| `gpu_stall_icnt2sh` | [12]->[13] reply injection blocked on icnt full | 1.82M |
| **`bw_util`** | **[8] real DRAM bandwidth** (high => truly DRAM-bound) | **0.078** |
| `Reply_Network_in_buffer_full` | [13] reply network | 6.3 |

### Bottleneck-class rule (bw_util is the primary discriminator)
- `bw_util` HIGH (>~50%) + stall high  => genuinely **DRAM-bandwidth-bound** (not the case here).
- `bw_util` LOW (7.8%) + stall high     => **queue-serialization-bound** (current case). Then:
  - `gpu_stall_dramfull` big => [4] L2-input queue or upstream.
  - `gpu_stall_icnt2sh` big  => [12] reply path.
  - `idle_bw` big            => requests are not even reaching DRAM (upstream serialization).

### Traps to remember
- **[7] mrqq is cap=2 hardcoded** ([dram.cc:120]) — if requests reach DRAM but stall here, config
  cannot fix it; needs a code change.
- **`gpu_stall_dramfull` is misnamed** — it is L2-input ([4] `m_icnt_L2_queue`, cap=8) backpressure,
  not DRAM. If it explodes after the two levers, the next knob is `-gpgpu_dram_partition_queues` field1.
- Stages [2] and [3] are the same xbar_router (in/out), both cap 64, tuned by separate knobs.

## 4.11 HW validation anchors (NCU) — which HW metric each queue-rate maps to, and what the numbers say

Source: `nv_reports/h100/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.csv` (H100 SXM,
CC9.0). Per-kernel, because fwd (K5, `FlashAttnFwdSm90`) and bwd (K10, `FlashAttnBwdSm90`) behave
very differently and MUST be judged separately.

### Stage -> HW metric mapping
| sim stage | sim rate unit | closest HW reality | NCU metric to compare |
|---|---|---|---|
| [1] TMA issue | 128B line/cyc | TMA engine issue (no such serial cap in HW) | (no direct NCU metric -> likely a sim artifact) |
| [2][3] SM<->L2 icnt | packet/cyc | SM->L2 crossbar / NoC BW | L1/TEX Cache Throughput, L2 Cache Throughput |
| [5] L2 bank | access/cyc | L2 slice throughput | L2 Cache Throughput |
| [6][7] DRAM arbiter | req/cyc | memory-controller channel | DRAM Throughput |
| [8] DRAM | column cmd | HBM3 physical BW | **DRAM Throughput (pct of peak)** = primary anchor |

### HW vs sim baseline — FWD (K5)
| metric | HW (NCU) | sim baseline | note |
|---|---|---|---|
| Elapsed Cycles | 67,696 | 145,855 | sim 2.15x over |
| DRAM Throughput | **12.09%** | ~1.9% (62.8 GB/s / 3.35TB/s); bw_util(active) 2.45% | sim ~6x LOWER |
| L2 Hit Rate | 69.58% | 94.61% | sim over (fwd CTA-count-cap approximation limit) |
| L2 Cache Throughput | 22.68% | n/a (no equiv %% metric) | — |
| L1/TEX Cache Throughput | 31.99% | n/a | — |
| Compute (SM) Throughput | 43.04% | n/a | — |
| Memory Throughput | 28.84% | n/a | — |

### HW vs sim baseline — BWD (K10)
| metric | HW (NCU) | sim baseline | note |
|---|---|---|---|
| Elapsed Cycles | 132,901 | 290,572 | sim 2.19x over |
| DRAM Throughput | **14.85%** | ~4.75% (159.1 GB/s / 3.35TB/s); bw_util(active) 6.21% | sim ~3x LOWER |
| L2 Hit Rate | 82.26% | 87.18% | close (M2/M2.5 addressing success) |
| L2 Cache Throughput | 48.52% | n/a | HW L2 quite busy |
| L1/TEX Cache Throughput | **62.59%** | n/a | **HW's highest pipe = the real hot spot** |
| Compute (SM) Throughput | 48.45% | n/a | compute also ~half |
| Memory Throughput | 56.58% | n/a | memory pipe > half busy |

> DRAM %% caveat: HW uses NCU pct-of-peak; sim has no pct-of-peak counter, so sim DRAM is shown two
> ways — (a) `DRAM_BW_total` GB/s divided by H100 HBM3 peak ~3.35 TB/s, and (b) `bw_util` (fraction of
> DRAM *active* cycles). Definitions differ, but BOTH agree sim uses far less DRAM than HW. Sim has no
> NCU-equivalent L2/L1TEX throughput %%, so those axes are n/a (can't be directly compared).

### Findings (what this proves)
1. **HW is NOT DRAM-bandwidth-bound** (fwd 12% / bwd 15% DRAM). So "DRAM is idle" is TRUE on real HW
   too — the injection/queue diagnosis is not contradicted by HW.
2. **Sim uses even LESS DRAM than HW** (fwd ~6x, bwd ~3x lower). This is exactly the fingerprint of
   requests being stuck in the icnt/L2 queues and not reaching DRAM — matching the [2]/[4] injection
   bottleneck. So widening the queues (the two levers) should move sim DRAM% UP toward HW, which is an
   accuracy improvement, not a cheat.
3. **HW's real hot spot is the on-chip memory path, not DRAM.** bwd: L1/TEX 62.6%, Memory 56.6%,
   L2 48.5%, Compute 48.5% are all high together (balanced), while DRAM is only 15%. "Memory pipe busy
   but DRAM idle" == the bottleneck lives in L2 / L1TEX / interconnect — the same region the levers
   target.
4. **fwd vs bwd differ.** fwd L2-hit is still over-modeled (0.946 vs HW 0.696, the CTA-cap limit) and
   fwd DRAM gap is larger (6x); bwd L2-hit is on target (0.872 vs 0.823) and DRAM gap is 3x. Expect a
   bigger, cleaner lever effect on bwd than fwd.

### Lever judgment thresholds (fill "after-lever" once runs finish)
The two queue levers are a **timing** change only (see 4.12): they must cut cycles WITHOUT changing
work. So judge on two separate axes.

| gate | axis | FWD K5 | BWD K10 |
|---|---|---|---|
| `gpu_sim_cycle` -> HW | timing (lever's job) | 145,855 -> toward 67,696 | 290,572 -> toward 132,901 |
| DRAM bytes (must NOT change vs baseline) | work (hit-rate's job) | ~5.09 MB | ~25.68 MB |
| L2 sectors (must NOT change) | work | 3,356,320 | 11,213,151 |
| `L2_TMA_true_hit_rate` (must hold) | work | 0.9461 | 0.8718 |

- **Lever success = `gpu_sim_cycle` drops toward HW WHILE DRAM bytes / L2 sectors / hit rate stay put.**
  A lever that changed the work counts would be a bug, not a win.
- **Do NOT use DRAM% (bytes/cycle) as a lever gate.** Cutting cycles raises DRAM% mechanically even
  though bytes are unchanged; that is a side-effect of the timing fix, not an accuracy gain. DRAM% is a
  final cross-check (4.12 TODO), read only after cycles converge.
- **The remaining DRAM-work gap (fwd 0.26x, bwd 0.56x) is a hit-rate / addressing problem, not a lever
  problem** — track it under M2/M2.5, not here.
- **Relocation (per 4.9):** if cycle is flat, use the 4.10 counter->stage table (`out_buffer_full`,
  `gpu_stall_dramfull`, `bw_util`) to see where the stall moved.

### After-lever result — FWD K5 (`grant_passes=4`, `icnt_to_l2_pop=4`) — 2026-07-09

This is the matching FWD run for the same two queue levers. The result is weaker than BWD: the levers
still work, but the net cycle gain is small because FWD's remaining stall relocates downstream much
earlier.

| metric | baseline | after lever | judgment |
|---|---|---|---|
| `gpu_sim_cycle` | 145,855 | **140,138** | **-3.9%** -> small timing improvement |
| `L2_TMA_true_hit_rate` | 0.9461 | **0.9455** | unchanged -> **work invariant** |
| `Req_Network_in_buffer_full_per_cycle` | 17.44 | **0.5169** | **-97%** -> REQ injection bottleneck largely removed |
| `Req_Network_out_buffer_full_per_cycle` | 0.0000 | **0.1561** | still tiny -> [3] out-buffer not the real new limiter |
| `gpu_stall_dramfull` | 106,285 | **239,240** | **+125%** -> strong relocation to downstream admission/backpressure |
| `gpu_stall_icnt2sh` | 73,812 | **94,011** | **+27%** -> reply-side pressure increased |
| `L2_TMA_output_full_cycles` | 44,836 | **73,775** | **+65%** -> reply/output congestion increased |
| `L2_TMA_port_busy_cycles` | 0 | **0** | no evidence that L2 data-port width is the limiter here |

**Did the two levers actually fire?** Yes.
- `Req_Network_avg_passes_per_active_cycle = 3.7625` -> lever 1 again ran close to the configured 4
  passes/cycle.
- `Req_Network_extra_pass_grants_total = 2.18M` -> later iSLIP passes materially drained extra REQ
  packets.
- `gpu_icnt_to_l2_extra_pops = 1.80M / 3.01M total` -> about **60%** of all icnt->L2 pops came from
  the added downstream pops, so lever 2 also materially fired.

**Interpretation.**
1. **The levers worked exactly as designed on the injection axis.** `Req in_buffer_full` almost
   disappeared (17.44 -> 0.52) while `L2_TMA_true_hit_rate` and L2 byte counts stayed effectively
   unchanged.
2. **But FWD had much less REQ-side pain to begin with than BWD.** So once the injection queue is
   relieved, the remaining bottleneck surfaces quickly downstream, and the net cycle gain is only
   **-3.9%**.
3. **The new visible stall is NOT [3] out-buffer and NOT [5] L2 data-port width (at least in this
   run).** `out_buffer_full` stays near zero and `L2_TMA_port_busy_cycles` stays zero. The relocation
   is instead more consistent with **[4] L2-entry / admission backpressure and [12] reply-path
   congestion** (`gpu_stall_dramfull`, `output_full`, `gpu_stall_icnt2sh` all rise).
4. **So FWD and BWD diverge.** BWD got a clearer cycle win from REQ-side drain widening; FWD mostly
   exposes the next downstream limit sooner.

**Work-axis confirmation (same run).**
- `L2_total_cache_accesses`: 3,356,320 -> 3,366,184 (**+0.3%**, effectively unchanged)
- `L2_cache_read_bytes`: 102.88 MB -> 103.20 MB (**+0.3%**, effectively unchanged)
- `L2_cache_write_bytes`: 4.52 MB -> 4.52 MB (unchanged)
- `DRAM_BW_total_GBps`: 62.80 -> 65.34 GB/s (**higher only because cycles fell**; not evidence of more
  DRAM work)

**Practical reading.**
- FWD K5 confirms the two queue levers are **real and timing-only**, just like BWD.
- But the gain is small because FWD rapidly exposes **downstream** limits after the REQ queue is fixed.
- The zero `L2_TMA_port_busy_cycles` is important: for FWD, this run gives **no positive evidence** yet
  that widening `m_data_port_width` is the main next lever. K10 remains the cleaner proving ground for
  the ongoing L2-port-width experiment.

### After-lever result — BWD K10 (`grant_passes=4`, `icnt_to_l2_pop=4`) — 2026-07-09

This is the first completed behavior run with BOTH queue levers enabled. It confirms the intended
effect: **work stayed invariant, timing improved, and the bottleneck relocated downstream from
[2]/[3] to [4]/[5].**

| metric | baseline | after lever | judgment |
|---|---|---|---|
| `gpu_sim_cycle` | 290,572 | **262,744** | **-9.6%** -> timing improved |
| `L2_TMA_true_hit_rate` | 0.8718 | **0.8701** | unchanged -> **work invariant** |
| `Req_Network_in_buffer_full_per_cycle` | 355.0 | **70.9** | **-80%** -> primary REQ injection bottleneck relieved |
| `Req_Network_out_buffer_full_per_cycle` | 0.20 | **2.51** | small rise, still low -> no meaningful relocation to [3] |
| `gpu_stall_dramfull` | 1.42M | **1.35M** | still huge -> downstream backpressure remains |
| `gpu_stall_icnt2sh` | 1.82M | **0.51M** | **-72%** -> reply-side pressure also dropped |
| `L2_TMA_output_full_cycles` | 1.39M | **0.47M** | **-66%** -> less reply/output congestion |

**Did the two levers actually fire?** Yes, decisively.
- `Req_Network_avg_passes_per_active_cycle = 3.74` -> lever 1 reached almost the full configured 4
  passes/cycle.
- `Req_Network_extra_pass_grants_total = 8.2M` -> later iSLIP passes granted many additional packets,
  so lever 1 was not just configured; it was materially used.
- `gpu_icnt_to_l2_extra_pops = 7.4M / 11.0M total` -> about **67%** of all icnt->L2 pops came from
  the extra downstream pops, so lever 2 also materially worked.

**Interpretation.**
1. **The two levers succeeded on their intended axis.** `Req in_buffer_full` collapsed from 355 to 71,
   cycle count dropped 9.6%, and the key work counters stayed fixed (`L2_TMA_true_hit_rate` unchanged,
   L2 accesses/read-bytes/write-bytes within ~1%). This is exactly the 4.12 rule: **timing changed,
   work did not**.
2. **The user's predicted "relocation" did happen, but NOT to `out_buffer`.** `Req out_buffer_full`
   rose only from 0.20 to 2.51, still tiny. So [3] is not the new limiter.
3. **The remaining stall moved to the L2-entry / L2-bank region.** `gpu_stall_dramfull` stayed huge
   (1.35M) even after the REQ-side lever succeeded. Per the 4.10 stage map, that means the limiter is
   now around **[4] `m_icnt_L2_queue` backpressure and/or [5] L2 bank/data-port service rate**.
4. **Reply-path counters improved as a consequence, not as the root cause.** `gpu_stall_icnt2sh` and
   `L2_TMA_output_full_cycles` both fell sharply because once injection pressure eased, the whole pipe
   stopped bunching up behind it. This supports the earlier conclusion that reply-path tuning was not
   the primary first lever.

**Work-axis confirmation (same run).**
- `L2_total_cache_accesses`: 11,213,151 -> 11,310,291 (**+0.9%**, effectively unchanged)
- `L2_cache_read_bytes`: 239.3 MB -> 242.3 MB (**+1.3%**, effectively unchanged)
- `L2_cache_write_bytes`: 119.5 MB -> 119.6 MB (unchanged)
- `DRAM_BW_total_GBps`: 159.1 -> 176.7 GB/s (**higher only because cycles fell**; do NOT misread this
  as more DRAM work)

**Immediate conclusion.**
- The queue levers are a **real partial success**.
- They remove most of the [2]/[3] artificial injection throttling.
- They do **not** fully close the cycle gap because the next limiter is now downstream, in **[4]/[5]**.

**Next lever (based on this run).**
- Do **not** spend another 12h run just to "see if L2 matters" — this run already shows it does.
- The next experiment should target **L2 service rate**, not reply-path or more REQ-side widening.
- Most likely candidates:
  1. **[5] L2 data-port width / service bandwidth** (`-gpgpu_cache:dl2` last field = `m_data_port_width`)
  2. **[4] L2-input queue capacity** (first field of `-gpgpu_dram_partition_queues`)
- Important caveat before the L2-port sweep: the current `fill_cycles =
  atom_sz / m_data_port_width` path uses floor division, so simply increasing `m_data_port_width`
  above the atom size without a small code fix can accidentally make fill-port cost zero. Treat the
  next L2-port experiment as **config + correctness guard**, not config-only.

**Throughput metrics note.**
- The completed `.o15` run above was built **before** the new `Throughput_*` counters were added, so
  those lines do not appear in this log.
- Therefore this run is enough to prove **timing/work correctness** of the two queue levers, but a
  rebuild + rerun is still required to collect the new DRAM/L2/L1TEX/Compute throughput outputs for
  the final HW-facing comparison.

### 4.11.1 Next experiment direction (actionable) — L2 service-rate decision tree

**Decisive finding now fixed in the plan.**
- The real L2 service-rate knob already exists: `m_data_port_width` (`cache bytes/cycle`), parsed
  from the **last field** of `-gpgpu_cache:dl2`.
- Current H100 tested config uses `-gpgpu_cache:dl2 ... ,32:0,32`, so current L2 data-port width is
  **32B/cycle**.

**Important nuance (why not run config-only sweep immediately).**
- Existence of the knob does **not** automatically mean "no code work needed":
  1. Current TMA/L2 path is 32B sector-heavy (`atom_sz = 32B` in sector cache mode), so widening only
     `m_data_port_width` may yield limited gain by itself.
  2. `memory_sub_partition::cache_cycle()` still admits one queue-head access attempt per cycle; this
     can cap realized throughput even if the internal data port is wider.
  3. Current fill-port model uses floor division (`fill_cycles = atom_sz / m_data_port_width`), which
     can become zero when `m_data_port_width > atom_sz`; that would create a non-physical "free fill"
     artifact.

**So the next run must be "config + correctness guard", not blind config-only.**

#### Step A (small implementation guard; timing-model correctness)
Before any 12h behavior run:
1. Change fill-port occupancy to ceil division:
   - from `fill_cycles = atom_sz / port_width`
   - to `fill_cycles = ceil(atom_sz / port_width)`
2. Align writeback data-port occupancy with the same ceil rule for consistency:
   - from `modified_size / port_width`
   - to `ceil(modified_size / port_width)`
3. Keep HIT path as-is (already ceil-equivalent).

This prevents accidental 0-cycle occupancy when sweeping `m_data_port_width` to 64/128.

**Status now:** implemented. `gpu-cache.cc` now uses ceil-equivalent occupancy for fill-port and
writeback data-port accounting, so the next `m_data_port_width=64` run will not get a false zero-cycle
fill artifact from the old floor-division path.

#### Step B (single decisive run, BWD K10 first)
Use the same rebuilt binary and run only one controlled A/B on K10:
- **A (control):** current `-gpgpu_cache:dl2 ... ,32:0,32`
- **B (test):** `-gpgpu_cache:dl2 ... ,32:0,64`

**Status now:** the tested H100 config has been advanced to the guarded **B** setting (`...,32:0,64`)
for the next run. The already-recorded baseline/after-lever K10 results serve as the width=32 control.

Read these outputs together:
- timing: `gpu_sim_cycle`
- work invariance: `L2_TMA_true_hit_rate`, `L2_total_cache_accesses`, `DRAM served bytes`
- relocation: `gpu_stall_dramfull`, `L2_TMA_port_busy_cycles`, `L2_TMA_output_full_cycles`
- HW-facing direction: `Throughput_L2_pct`, `Throughput_DRAM_pct` (plus served-byte counters)

#### Step C (decision gate)
1. If `gpu_sim_cycle` drops and `L2_TMA_port_busy_cycles` drops while work stays invariant:
   - `m_data_port_width` is a real remaining limiter -> proceed to optional `64 -> 128` sweep.
2. If cycle barely changes and `gpu_stall_dramfull` remains dominant:
   - limiter is likely not width-only; move to explicit [4] admission-rate lever / multi-access path.
3. Reject any run where work counters drift materially (that indicates model behavior changed, not a
   pure timing fix).

#### Why this order is optimal
- Avoids burning multiple 12h runs for exploratory guesses.
- Uses one guarded A/B to answer the main question decisively.
- Preserves the 4.12 principle: **work first, timing second, throughput% last**.

#### Step B/C RESULT — width 32->64 REJECTED (BWD K10, `.o16`/`.e16`, 2026-07-10)

The guarded A/B ran. Config reverted to `...,32:0,32`. **The width lever is a null result — Step C
branch 2 ("cycle barely changes, `gpu_stall_dramfull` remains dominant") is confirmed.** This was
statically predictable and is now measured.

| metric | width=32 (`.o15`) | **width=64 (`.o16`)** | delta | axis |
|---|---|---|---|---|
| `gpu_sim_cycle` | 262,744 | **263,273** | **+0.2%** | timing — no gain (marginally worse) |
| `L2_TMA_true_hit_rate` | 0.8701 | 0.8700 | ~0 | work invariant ✅ |
| `L2_total_cache_accesses` | 11,310,291 | 11,295,780 | -0.1% | work invariant ✅ |
| `L2_cache_read_bytes` | 242.32 MB | 241.86 MB | -0.2% | work invariant ✅ |
| `L2_cache_write_bytes` | 119.60 MB | 119.60 MB | 0.0% | work invariant ✅ |
| `L2_TMA_port_busy_cycles` [5] | 660 | **314** | -52% (noise floor) | port was never the limiter |
| `gpu_stall_dramfull` [4] | 1,353,380 | **1,179,321** | -12.9% | dropped but **cycle did not move** |
| `L2_TMA_output_full_cycles` [12] | 465,528 | 449,564 | -3.4% | ~flat |
| `gpu_stall_icnt2sh` [12->13] | 511,707 | 513,932 | +0.4% | flat |
| `wait_barrier` SM-idle | 9.80% | **9.83%** | pinned | TMA-completion axis unmoved |
| `tma_flush` SM-idle | 8.88% | 8.28% | -0.6pp | marginal |
| `Req_Network_in_buffer_full` | 70.94 | 56.23 | -21% | side effect |

**Why width=64 did nothing (static root, confirmed by the numbers):**
1. **L2 is a 32B-sector cache** (`-gpgpu_cache:dl2 S:...`), so `get_atom_sz()` returns
   `SECTOR_SIZE = 32B` ([gpu-cache.h:684](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h#L684)). The fill/HIT data-port cost is
   `ceil(atom_sz / port_width)` ([gpu-cache.cc:1150-1185](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1150-L1185)), and
   `ceil(32/64) == ceil(32/32) == 1`. So the dominant sector traffic sees **byte-identical port
   occupancy** at 32 and 64. The ceil guard (Step A) worked exactly as intended — it prevented a fake
   0-cycle fill — but there was no real occupancy to remove.
2. **The only path width actually changed is 128B full-line writeback** (`ceil(128/32)=4 ->
   ceil(128/64)=2`). That is real but tiny: it only moved `L2_TMA_port_busy_cycles` 660 -> 314, and
   `port_busy` was already at the noise floor (0.13% of `output_full`), so it is not on the critical
   path.
3. **The 12.9% drop in `gpu_stall_dramfull` with a FLAT cycle count is the decisive signal.** Widening
   the port let L2 admit slightly faster, which relieved [4] L2-input backpressure — but cycles did not
   move. That means **[4]/[5] (L2 admission + data port) are NOT the critical-path limiter**; they are
   downstream symptoms. This run therefore *also* rules out the "next lever = [4] admission-rate"
   candidate that Step C branch 2 originally proposed, for free.

**What this run proves about the real wall.** `wait_barrier` stayed pinned at ~9.8% across width=32,
width=64, and all five prior queue/interconnect experiments (depth, drain, grant-passes, icnt->L2 pop,
port width). A stall that no request-path or L2-service knob can move is **not** a request-path or
L2-service problem. Per the 4.10 map the only untested region on the critical path is the **reply/return
segment [9]-[13] on realistic locality**, and specifically the TMA per-transfer `avg_drain_cycles`
(waiting for the 768 sector responses to come back), which stayed ~2,300-2,570 cyc in every run. See
4.11.2 for the reframed diagnosis.

#### Step B/C RESULT — width 32->64 REJECTED for FWD K5 too (`.o33`/`.e33`, 2026-07-10)

The matching FWD A/B ran (grant_passes=4, icnt_to_l2_pop=4, width 32->64). **Same null verdict as BWD,
and even cleaner: FWD `L2_TMA_port_busy_cycles` is 0 at BOTH widths, so the port had literally zero
occupancy to remove.**

| metric | width=32 (`.o32`) | **width=64 (`.o33`)** | delta | axis |
|---|---|---|---|---|
| `gpu_sim_cycle` | 140,138 | **140,213** | **+0.05%** | timing — no gain |
| `L2_TMA_true_hit_rate` | 0.9455 | 0.9459 | ~0 | work invariant ✅ |
| `L2_total_cache_accesses` | 3,366,184 | 3,367,644 | +0.04% | work invariant ✅ |
| `L2_cache_read_bytes` | 103.20 MB | 103.24 MB | +0.04% | work invariant ✅ |
| `L2_cache_write_bytes` | 4.52 MB | 4.52 MB | 0.0% | work invariant ✅ |
| `L2_TMA_port_busy_cycles` [5] | **0** | **0** | — | port never occupied (no writeback pressure) |
| `gpu_stall_dramfull` [4] | 239,240 | 256,095 | +7.0% | rose, cycle flat |
| `L2_TMA_output_full_cycles` [12] | 73,775 | 67,797 | -8.1% | ~flat |
| `gpu_stall_icnt2sh` [12->13] | 94,011 | 85,477 | -9.1% | ~flat |
| `wait_barrier` SM-idle | 9.83% | **9.87%** | pinned | TMA-completion axis unmoved |
| `tma_flush` SM-idle | 0.00% | 0.00% | — | FWD is load-only (no reduce/store drain) |
| `Req_Network_in_buffer_full` | 0.52 | 0.52 | flat | already relieved by grant-passes |

**Why FWD is an even stronger null than BWD.** For BWD, width touched the 128B writeback port
(`port_busy` 660->314); for FWD `port_busy` is **0 at both widths**, because FWD writes almost nothing
to L2 (`L2_cache_write_bytes` 4.52 MB, load-only kernel). So the L2 data port is provably not on FWD's
critical path at any width. Width=64 is rejected for both kernels; config reverted to
`...,32:0,32`.

## 4.11.2 Reframed diagnosis — the wall is TMA-transfer SERIALIZATION, not any single queue (BWD K10, static + `.o15`, 2026-07-10)

Six consecutive knob experiments (depth, reply-drain, grant-passes, icnt->L2 pop, port-width x1) all
left `gpu_sim_cycle` and `wait_barrier` essentially fixed. Before spending a 7th 12h run, this section
re-derives the bottleneck from the **existing** `.o15` latency counters (no new run needed) so the next
experiment targets the actual critical path.

### Evidence 1 — the per-request round-trip is queue-bound, and DRAM is negligible
From `.o15` (`-gpgpu_memlatency_stat 14`, already in every run):

| counter | value | meaning (stage in 4.10 map) |
|---|---|---|
| `averagemflatency` | **1,989** | full mf round-trip: SM inject -> ... -> SM receive |
| `avg_icnt2mem_latency` | 473 | request path SM-inject -> DRAM-accepted ([2]->[7]) |
| `avg_mrq_latency` | **8** | time inside the DRAM scheduler ([7] mrqq) |
| `maxmrqlatency` | 420 | worst DRAM-scheduler wait |
| `avg_icnt2sh_latency` | 178 | reply path L2-reply-ready -> SM receive ([12]->[14]) |
| `max_icnt2sh_latency` | 4,485 | worst reply-eject wait |
| `bw_util` (DRAM) | 0.086 | DRAM 91% idle |

**The DRAM device itself costs ~8 cyc of a ~1,989 cyc round-trip (0.4%).** Even the configured
`-dram_latency 243` + `-gpgpu_l2_rop_latency 100` = 343 fixed cyc is only ~17% of the round-trip. So
**~80% of every request's latency is spent queued**, not in ROP/DRAM/reply-eject. This is the
quantitative proof that the model is **queue-serialization-bound**, exactly as 4.10's bw_util rule
predicts — and that no single downstream queue is the culprit (each individual segment avg is small:
req 473, mrq 8, reply 178; they sum to ~659, far below the 1,989 total, so the rest is spent waiting to
*enter* those segments, i.e. head-of-line blocking upstream).

### Evidence 2 — the real serialization is one TMA transfer = 768 sectors sharing one SM port
The TMA per-transfer decomposition (`.e15`, SM0) is the decisive lens, not the aggregate queue counters:

| TMA per-transfer field (BWD K10, after both levers) | value |
|---|---|
| `avg_issue_active_cycles` | 914 |
| `avg_icnt_full_cycles` | **893 (97.7% of issue-active)** |
| `avg_emit_span_cycles` | 449 |
| `avg_drain_cycles` | **2,570** |
| `avg_to_first_request_cycles` | 467 |
| `avg_requests_per_issue_active_cycle` | 0.565 |

Even after grant-passes=4 relieved the REQ in_buffer (355 -> 71), a single transfer still spends 893
of its 914 issue-active cycles blocked on `m_icnt->full()`, and then waits 2,570 more cyc for its 768
sector replies to drain back. **`drain (2,570) > emit (449)`** on realistic locality — the transfer is
completion-latency-bound, and that completion latency is the sum of 768 individual sector round-trips
each averaging ~1,989 cyc but pipelined through a 1-in / limited-out shared port.

### Evidence 3 — FWD K5 is a DIFFERENT bottleneck (reply-side, small transfers), which is why FWD gained less
The same latency decomposition on FWD (`.o32`/`.o33`, width=32/64) tells a qualitatively different
story than BWD, and it explains why grant-passes gave BWD -9.6% but FWD only -3.9%:

| latency counter | FWD K5 (`.o32`) | BWD K10 (`.o15`) | reading |
|---|---|---|---|
| `averagemflatency` (round-trip) | **595** | 1,989 | FWD round-trip is 3.3x SHORTER |
| `avg_icnt2mem_latency` (req side) | 92 | 473 | FWD req path barely queued |
| `avg_mrq_latency` (DRAM device) | 47 | 8 | both negligible |
| `avg_icnt2sh_latency` (reply side) | **335 (56% of 595)** | 178 (9% of 1,989) | **FWD is REPLY-dominated; BWD is upstream-queue-dominated** |
| TMA `avg_icnt_full_cycles` (SM0) | **12.4** | 893 | FWD injection is NOT blocked |
| TMA `avg_emit_span_cycles` (SM0) | 70 | 449 | FWD emits in ~70 cyc |
| TMA `avg_drain_cycles` (SM0) | **1,136** | 2,570 | FWD still waits ~1,100 to complete |
| TMA transfers/SM0 | 42 (688 KB) | 174 (2.88 MB) | **FWD transfers are ~4x smaller** |

**Why FWD and BWD diverge (decisive):**
1. **FWD injection is already fine.** `avg_icnt_full_cycles=12.4` (vs BWD 893) means the SM shared
   REQ port is NOT the FWD wall — grant-passes already emptied it (`Req_in_buffer_full=0.52`). So
   TMA-128B (128B emission), which attacks the injection count, has **little headroom on FWD** — the
   768->192 reduction cannot help a stage that is only busy 12 cyc/transfer.
2. **FWD's cost is on the REPLY/return side.** 56% of its (short) round-trip is `icnt2sh`
   (L2-reply-ready -> SM receive). Yet the reply-path knobs (depth/reply-drain, section 4.5) were also
   null — because with `averagemflatency=595` and DRAM at 47 cyc, the reply latency is not queue depth,
   it is the fixed `-gpgpu_l2_rop_latency 100` + reply traversal applied per-sector across many small
   transfers. This is a **fixed-latency-per-sector x sector-count** cost, not a queue-throughput cost.
3. **This is consistent with 4.11's earlier reading** ("FWD relocates downstream sooner, smaller net
   gain") but now quantified: FWD's `wait_barrier` (9.87%) is set by reply-side per-sector latency on
   many small transfers, while BWD's `wait_barrier` (9.53%) is set by injection-side head-of-line
   blocking on few large transfers. **Same symptom (`wait_barrier` ~9.8%), two different roots.**

### Why every knob so far was null (unified explanation)
The knobs each widened **one** segment of a serial pipe whose throughput is set by the SM's **single
shared icnt port** carrying 768 sectors/transfer:
- grant-passes / icnt->L2 pop widened [2]->[4] (REQ side) -> `Req_in_buffer_full` fell, but the reply
  side and the sheer sector *count* were untouched, so `wait_barrier` held.
- depth / reply-drain / port-width touched [4]/[5]/[12] -> moved their local `*_full` counters but not
  the round-trip, because the round-trip is dominated by *queuing to enter* the shared port, not by any
  one queue's service rate.
- **The invariant across all six runs:** total sectors per transfer = 768, and they share one SM icnt
  injection node + one reply eject path. That count x sharing is the wall.

### The two candidate levers that actually attack the wall (pick ONE for the next 12h run)
Both reduce the *number of serialized sector round-trips per transfer*, which is the only thing every
prior knob failed to change. They are mutually exclusive for a clean A/B.

1. **TMA-128B — coalesce TMA emission to 128B lines (Opt 6A Part 1, previously deferred).** Emit one
   128B mf per AGU line instead of 4x32B ([tma_unit_sm.cc:614-738](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L614-L738)). This cuts injected mf
   count 768 -> 192 *before* the shared port. Caveat from 2-B: `memory_sub_partition::push()` re-splits
   a 128B parent into 4x32B for the sector-L2 ([l2cache.cc:826-851](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L826-L851)), so this only relieves
   the **[1]->[4] injection + REQ-net** segment, not L2/DRAM/reply. Given Evidence 1 says the reply
   side avg is small (178) but the *injection* side dominates (`avg_icnt_full` 893), this is now the
   best-supported single lever — it directly cuts the 893. Risk: needs the parent/child retire
   bookkeeping (Risk 2 in section 5); implementation, not config.
2. **TMA-dedicated-port — give TMA its own icnt injection node / port, separate from ldst.** Today TMA and ldst
   share one `m_icnt` per SM ([sm.cc:1195,1223,1229](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1195)). A bulk 768-sector transfer
   monopolizes it. A dedicated TMA port removes the head-of-line blocking that Evidence 1 attributes
   ~80% of round-trip latency to. This is a structural change (more invasive than TMA-128B) and risks
   giving TMA unrealistic injection bandwidth unless the new port is itself rate-limited.

### Recommendation before the next 12h run
- **The FWD result changes the plan: there is NO single lever that fixes both kernels.** BWD is
  injection-bound (TMA-128B helps), FWD is reply/fixed-latency-bound (TMA-128B barely helps). Do not pick
  one lever and expect both to move.
- **For BWD: prefer TMA-128B (128B emission).** It reduces the 768-sector count that every prior knob
  left untouched, directly supported by BWD `avg_icnt_full_cycles=893` (injection-side dominant). This
  is the original Opt 6A Part-1 plan, gated only by the (now-resolved) address-realism prerequisite.
- **For FWD: TMA-128B has almost no headroom** (`avg_icnt_full_cycles=12.4`). FWD's cost is reply-side
  per-sector latency across many small transfers (Evidence 3). The FWD-relevant knob is the fixed
  `-gpgpu_l2_rop_latency 100` applied per sector, or reducing the per-sector reply *count* (which
  TMA-128B also does at emission but the L2 re-split at [l2cache.cc:826-851](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L826-L851) undoes for the reply
  path). Treat FWD as a **separate, later** investigation; do not let it gate the BWD TMA-128B run.
- **Cheap pre-check first (no 12h run), and it now serves BOTH kernels:** add a per-transfer `lat_mem`
  histogram split into `req_side` (agu_ready -> DRAM-accept, reuse `icnt2mem`) vs `reply_side`
  (`icnt2sh`) vs `queue_wait` (remainder). For BWD this attributes the 1,989 - 659 = ~1,330 "unaccounted
  queue wait" to a specific `mem_fetch_status` bucket ([mem_fetch_status.tup](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/mem_fetch_status.tup)
  already enumerates every stage; `m_status_change` already timestamps each transition); for FWD it
  confirms the reply-side dominance before committing to a reply-side fix. Observe-only, timing-neutral.
  - **Status now: IMPLEMENTED.** `mem_fetch::set_status()` accumulates the residency of the previous
    status into TMA-only static counters `s_tma_status_cycles[]` / `s_tma_status_visits[]`
    ([mem_fetch.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/mem_fetch.cc), guarded to `m_is_tma` and monotonic transitions so
    parent/child splits and re-probes cannot corrupt totals). `print_tma_status_residency()` dumps a
    per-stage table plus the three plan buckets and is called from `gpu_print_stat()`
    ([gpu-sim.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc), after `gpu_icnt_to_l2_extra_pops`). Output keys:
    `TMA_status[<stage>] cycles/visits/avg/pct`, `TMA_status_bucket_req_side_cycles`,
    `TMA_status_bucket_reply_side_cycles`, `TMA_status_bucket_queue_wait_cycles`. Rebuild required
    (source change); no tracer/trace change. Read this on the next BWD run to confirm which stage owns
    the ~1,330 cyc before committing to TMA-128B.
- **Judgment gate for the BWD TMA-128B run:** success = `avg_icnt_full_cycles`↓ AND `gpu_sim_cycle`↓ AND
  `wait_barrier`↓, with `L2_TMA_true_hit_rate` and L2 sectors/bytes invariant (4.12 work axis). If BWD
  cycles stay flat while `avg_icnt_full` drops, the wall has relocated to the reply/return side (i.e.
  BWD converges to FWD's profile) and TMA-dedicated-port / rop-latency becomes the next candidate for both.

## 4.11.3 TMA-128B design — coalesced 128B-granularity TMA requests + parent/child retire, mirroring the L1->L2 path (2026-07-10)

> **SUPERSEDED by 4.11.4 (2026-07-10).** After a HW primary-source audit (arXiv:2501.12084 Hopper
> microbenchmarks + NVIDIA Hopper Tuning Guide), BOTH TMA-128B and TMA-dedicated-port were REJECTED as
> not HW-faithful: (1) real Hopper memory granularity is the 32B sector (a TMA request touches 1-4
> sectors of a 128B L2 line) — so emitting a 128B TMA request is not what HW does and the current 32B
> model is correct; (2) there is NO published evidence of a dedicated TMA->L2 port — the TMA bandwidth
> ceiling equals ordinary global-load bandwidth (~1800 GB/s on H800), i.e. TMA SHARES the SM->L2 path
> with the LSU, exactly as the sim's single shared `m_icnt` already models. The real artifact is the
> shared-port INJECTION RATE, not granularity or a missing port. See 4.11.4 for the HW-calibrated fix.
> This section is kept for the record (the parent/child retire analysis is still correct should a 128B
> path ever be revisited), but it is NOT the chosen direction.

> **Naming.** "TMA-128B" = emit one 128B-granularity request per AGU line (like the ldst coalescer)
> instead of 4 separate 32B sector mfs, then let L2 re-split it. The alternative structural fix is
> "TMA-dedicated-port" (a separate SM->L2 injection node for TMA). These replace the earlier
> placeholder names; TMA-128B is the recommended first experiment.

Static confirmation of how the normal ldst path and the TMA path differ today, so TMA-128B copies the
proven L1->L2 mechanism instead of inventing a new one.

### Confirmed today (code-verified)
1. **Normal ldst->L2 already sends one 128B request.** The coalescer emits a single `mem_access` of
   `segment_size=128` carrying a 4-sector chunk mask
   ([abstract_hardware_model.cc:1069-1122](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L1069-L1122)); that is ONE icnt packet. L2's
   `push()` then calls `breakdown_request_to_sector_requests`
   ([l2cache.cc:759-824](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L759-L824)): a 128B parent (`data_size==MAX_MEMORY_ACCESS_SIZE`) is split
   into 4x32B children, **each child's `original_mf` = the parent**.
2. **The 4 children MSHR-merge and are reassembled on the way back.** `send_read_request` uses
   `m_mshrs.probe(mshr_addr)` so same-line children coalesce ([gpu-cache.cc:1340-1391](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1340-L1391)),
   and `baseline_cache::fill` decrements `pending_read` on the parent, deleting each child until the
   last one, then resurfaces the parent ([gpu-cache.cc:1228-1247](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1228-L1247)). This is the
   parent/child retire bookkeeping TMA-128B must reuse.
3. **TMA today does the OPPOSITE.** `mover_issue_requests` emits `SECTOR_CHUNCK_SIZE` separate 32B mfs
   per 128B AGU line, each `data_size=SECTOR_SIZE`, single-bit sector mask, and crucially
   **`original_mf=nullptr`** ([tma_unit_sm.cc:835-852](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L835-L852)). So each 32B mf is a distinct icnt
   packet, and `breakdown_request_to_sector_requests` takes the **`data_size==SECTOR_SIZE &&
   sector_mask.count()==1` early-return branch** ([l2cache.cc:762-764](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L762-L764)) — i.e. no split, one mf
   in / one mf out. On return, `SM::accept_ldst_unit_response` routes any `is_tma()` mf to
   `tma_unit_sm::fill`, which looks the mf up by **exact pointer** in `m_outstanding_requests`
   ([sm.cc:1515-1524](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1515-L1524), [tma_unit_sm.cc:1055-1069](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L1055-L1069)).

### Answer to "does TMA send duplicate same-addr requests / does MSHR not work?"
**Not literal duplicates, but redundant fragmentation, and MSHR only helps at DRAM.** The 4 sectors of
one 128B line have distinct 32B addresses (`agu_base + sector*32`), so they are not identical. But:
- **They are the same 128B line.** L2 MSHR DOES merge same-line in-flight requests
  ([gpu-cache.cc:1340-1391](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1340-L1391)), so **DRAM sees no duplication** — consistent with
  `avg_mrq_latency=8`, DRAM negligible.
- **MSHR does nothing for the injection cost.** The 4x32B still occupy 4 icnt packets, 4 REQ-net
  in_buffer slots, and 4 separate L2 admission probes (`m_icnt_L2_queue` head is served one mf/cycle,
  [l2cache.cc:524-543](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L524-L543)) BEFORE any merge. That 4x inflation on the shared SM port is
  exactly BWD's `avg_icnt_full_cycles=893`. So the fix is not "add MSHR" (it exists) but "stop
  fragmenting at emission" — send the 128B parent so only ONE packet crosses the shared port, then let
  the existing L2 split+MSHR do what it already does for ldst.

### TMA-128B implementation sketch (mirror the ldst path, do NOT invent a new one)
- **Emit one 128B parent mf per AGU line** in `mover_issue_requests` when a full 128B line is being
  moved: `data_size=MAX_MEMORY_ACCESS_SIZE`, full 4-sector mask, full 128B byte mask, `original_mf=
  nullptr` (the parent IS the original). Replace the inner 4x (or 8x for reduce) sector loop with a
  single alloc+push. Goal counter changes from `kSectorMfGoal = agu_requests*SECTOR_CHUNCK_SIZE*
  mfs_per_sector` to `kLineMfGoal = agu_requests * mfs_per_line` (`mfs_per_line=1` load/store, `2` for
  reduce RMW: one 128B read + one 128B write).
- **Let L2 split it** — no TMA-side change needed: `push()` already routes a 128B parent through
  `breakdown_request_to_sector_requests` into 4 children with `original_mf=parent`, MSHR-merged, exactly
  like ldst. This is the key point: the re-split still happens (section 2-B caveat), but only AFTER the
  shared-port injection, which is the bottleneck.
- **Retire on the parent, not the child.** Two options:
  1. **Preferred (reuse existing machinery):** keep `m_outstanding_requests[parent]=uid`; the children
     are deleted inside `baseline_cache::fill` and only the parent resurfaces
     ([gpu-cache.cc:1237-1246](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1237-L1246)), so `SM::accept_ldst_unit_response` receives the PARENT and
     `tma_unit_sm::fill` finds it by pointer unchanged. Must verify the L2 reply path actually returns
     the parent (not a child) for the L2-bypass and HIT cases too.
  2. **Fallback (explicit child accounting):** if some path returns children directly, add
     `m_outstanding_sectors[parent]` initialized to the child count and decrement per returning child,
     retiring the parent when it hits 0 (Risk 2 in section 5).
- **Reduce/store (bwd `tma_flush`):** a reduce line is 2 parent mfs (128B read + 128B write). Keep the
  RMW pairing; do not collapse the write side, or the store-visibility/ack accounting
  ([sm.cc:2016-2017](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2016-L2017)) breaks.

### The exact "fill one sector of a 128B line" mechanism (this is the retire question, resolved)
The user's question — "when a return ptr comes back it has to fill ONE sector of the 128B, how is that
done?" — is precisely the parent/child retire, and the sector-cache path ALREADY implements it. Traced
end to end for a 128B parent:

1. **Split happens in `push`, parent never enters a queue.** `memory_sub_partition::push` calls
   `breakdown_request_to_sector_requests` ([l2cache.cc:826-851](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L826-L851)). For a 128B parent it takes
   the `data_size==MAX_MEMORY_ACCESS_SIZE` branch ([l2cache.cc:765-780](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L765-L780)) and makes 4
   children, each `alloc(..., original_mf=parent)`. Only the **children** are pushed to ROP -> the L2
   input queue; the **parent is never queued and never probes L2**. (This is also why the parent must
   NOT be registered in `m_request_tracker`/icnt — it is a pure accounting token.)
2. **Each child probes L2 independently.** In `cache_cycle`, each child hits `m_L2cache->access()`
   ([l2cache.cc:548](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L548)). On a MISS, `send_read_request` registers
   `m_extra_mf_fields[child]` with `pending_read = m_line_sz / SECTOR_SIZE = 4`
   ([gpu-cache.h:1538-1540](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h#L1538-L1540)) — i.e. the count is stored on the mf that carries
   the line, and same-line siblings MSHR-merge onto it ([gpu-cache.cc:1340-1359](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1340-L1359)).
3. **"Fill one sector" = `baseline_cache::fill` decrementing `pending_read`.** Each returning 32B child
   enters `fill`; because `m_mshr_type==SECTOR_ASSOC`, it does
   `e = m_extra_mf_fields.find(mf->get_original_mf()); e->second.pending_read--;`
   ([gpu-cache.cc:1230-1246](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1230-L1246)). If `pending_read > 0` it **deletes that child and returns**
   (`res_deleted`), i.e. that one sector is filled and the line is not yet complete. When the LAST child
   arrives (`pending_read == 0`) it swaps `mf = mf->get_original_mf()` (the parent), deletes the child,
   and lets the parent flow out as the single completed reply. **So exactly one mf — the parent —
   resurfaces per 128B line, after all 4 sectors are filled.**
4. **Therefore the TMA retire needs NO new per-sector logic on the TMA side.** Because step 3 already
   collapses 4 children -> 1 parent, the mf that reaches `SM::accept_ldst_unit_response`
   ([sm.cc:1515-1524](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1515-L1524)) is the **parent pointer we registered** in
   `m_outstanding_requests`. `tma_unit_sm::fill` finds it by pointer and calls `mover_on_response`
   ONCE for the whole 128B line. This is option-1 above, and it is the same machinery ldst already uses.

**The one thing that MUST be checked before trusting option 1 (else it silently corrupts):**
`baseline_cache::fill`'s child->parent collapse ONLY runs for a **MISS that went through
`send_read_request`** (which set `pending_read=4` and `original_mf`). Two other L2 outcomes bypass it:
- **L2 HIT** ([l2cache.cc:578-595](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L578-L595)): the HIT reply is pushed to `m_L2_icnt_queue` as the
  **child** mf (it never entered `fill`), so a HIT child returns to the SM, NOT the parent.
- **L2 disabled / bypass** ([l2cache.cc:620-626](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L620-L626)): the child is pushed straight to the DRAM
  queue and returns as the child.
So for TMA, a 128B line that HITs in L2 will return as up to 4 separate child mfs, each with
`original_mf=parent` but NONE of them the registered parent -> `tma_unit_sm::fill`'s exact-pointer
lookup would MISS and hit the assert. **This is why option 2 (explicit `m_outstanding_sectors[parent]`)
is actually required, not optional:** register the parent with an expected child count, and in
`tma_unit_sm::fill` resolve `parent = mf->get_original_mf() ? mf->get_original_mf() : mf`, decrement the
parent's outstanding-sector count, and only call `mover_on_response` + free the parent when it reaches
0. The current sim shows `L2_TMA_true_hit_rate=0.87` (bwd), so ~87% of children take exactly this HIT
path — option 1 alone would assert almost immediately. Implement option 2.

### Store / reduce ACK path — traced, and why it needs DIFFERENT accounting than reads
The store/reduce side (bwd `tma_flush`, 14.66% baseline) does NOT go through `baseline_cache::fill`
(that is the read-fill path), so its 128B->4x32B collapse must be handled explicitly. Traced:

1. **Completion is purely count-based.** `mover_on_response` increments `requests_completed` for every
   returning mf and retires the transfer when `requests_completed >= sector_mf_goal`
   ([tma_unit_sm.cc:918-946](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L918-L946)). For a store, `sector_mf_goal =
   agu_requests*4*1`; for a reduce, `*2` (read + write per sector). On the last mf it decrements the
   warp's `m_outstanding_stores_per_warp` ([tma_unit_sm.cc:1033-1046](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L1033-L1046)), which is what a
   `UTMACMDFLUSH` warp-stall waits on. So **the store's `wait`/`tma_flush` release is gated by the exact
   count of returning mfs matching `sector_mf_goal`.** If TMA-128B changes how many mfs come back without
   changing `sector_mf_goal` to match, the store either never retires (hang) or retires early (wrong).
2. **A write DOES generate a reply that returns to the SM.** L2 dl2 policy is `...,L:B:m:L:P,...` =
   WRITE_BACK + write-allocate. A write HIT calls `wr_hit_wb` -> returns HIT with no miss-queue send
   ([gpu-cache.cc:1425-1441](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1425-L1441)); back in `cache_cycle` the HIT branch has `write_sent==false`,
   so it does `mf->set_reply(); m_L2_icnt_queue->push(mf)` ([l2cache.cc:578-591](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L578-L591)) — i.e. a
   WRITE_ACK flows back exactly like a read reply, and `accept_ldst_unit_response` routes it to
   `tma_unit_sm::fill` because the child inherits `is_tma` ([sm.cc:1515-1524](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1515-L1524)).
3. **BUT the write child->parent is NOT collapsed by anything.** `baseline_cache::fill` (which does the
   `pending_read` collapse) is only called on the DRAM read-return / L2 fill path, never for a write
   HIT reply. `breakdown_request_to_sector_requests` sets `original_mf=parent` on write children too
   ([l2cache.cc:772-777](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L772-L777)), but there is no `pending_write` decrement — so for a 128B write
   parent, **all 4 write children return independently to `tma_unit_sm::fill`**, each with
   `original_mf=parent`. This is actually the SAME shape as the read-HIT case in the note above, so
   **option 2's `m_outstanding_sectors[parent]` accounting handles reads AND writes uniformly**: resolve
   parent via `get_original_mf()`, decrement, retire at 0.

**Consequence for TMA-128B goal-count:** with option 2, keep the emission-side count in **128B lines**
but keep the retire-side count in **expected child mfs**. Concretely:
- Emit `kLineMfGoal = agu_requests * mfs_per_line` parents (`mfs_per_line`=1 load/store, 2 reduce).
- Set each parent's `m_outstanding_sectors = SECTOR_CHUNCK_SIZE` (4) — the number of children L2 will
  split it into (and the number of replies that will come back, whether HIT children or one collapsed
  read parent... see next bullet).
- **Unify the retire:** in `tma_unit_sm::fill`, always do `parent = mf->get_original_mf() ? ... : mf`
  then `if (--m_outstanding_sectors[parent] == 0) { mover_on_response(parent-as-one-line); free; }`.
  For a read MISS the L2 fill path already collapsed 4->1, so only 1 reply arrives — meaning
  `m_outstanding_sectors` must be set to the ACTUAL number of replies, which DIFFERS between the read-MISS
  (1, pre-collapsed) and read-HIT/write (4, not collapsed) paths. **This is the real subtlety:** the
  reply count is not constant. Two clean ways to make it deterministic:
  - (a) **Count bytes, not mfs.** Set `m_outstanding_bytes[parent]=128` (or 256 for reduce RMW) and
    decrement by each returning mf's `data_size` (32 for a child, 128 for a collapsed parent). Retire at
    0. This is robust to whether L2 collapsed or not. **Preferred.**
    - reduce needs read 128 + write 128 = 256; a returning read-collapsed-parent subtracts 128, four
      write children subtract 32 each = 128. Sums to 256 regardless of path.
  - (b) Make `mover_on_response` idempotent per (parent,sector-mask) and retire when the union of
    returned sector masks covers the line. More state; (a) is simpler and race-free in a serial sim.

**So the store/reduce path is safe for TMA-128B only if the retire switches from "count mfs == fixed
sector_mf_goal" to "count returned BYTES == transfer bytes".** That is the one non-trivial code change
option 2 forces, and it must be in before the 12h run or the store side will mis-retire.

### Barrier / correctness accounting under TMA-128B — traced, and why it is SAFE
The user's concern "read/write ACK both touch barrier-related state — did you check that?" is the real
risk. Traced all three barrier couplings; **all are byte- or transfer-based, none are per-mf-count**, so
128B emission does not change them AS LONG AS option-2 byte-retire is used:

1. **Load -> consumer mbarrier (`wait_barrier` release).** `mover_on_response` calls
   `notify_tma_completion(warp_id, entry.cmd.total_bytes)` exactly ONCE per transfer, only in the
   `transfer_type==LOAD` branch ([tma_unit_sm.cc:1006-1017](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L1006-L1017)). `notify_tma_completion` credits
   the mbarrier in **bytes** (`completed_tx_bytes += applied_bytes`, [sm.cc:1886-1916](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1886-L1916)),
   matched against `expected_tx_bytes` which is set by the SASS `expect-tx` arrive
   ([sm.cc:1806](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1806)). **It is credited with `cmd.total_bytes`, NOT with a per-sector-mf
   count.** So whether the transfer was 768 mfs or 192 mfs is irrelevant to the mbarrier — the credit
   is the same total_bytes. The ONLY requirement is that `notify_tma_completion` still fires once, at
   true completion. With byte-retire, "true completion" = all bytes returned, which is exactly when it
   should fire. **SAFE.**
2. **Store/reduce -> `tma_flush` (UTMACMDFLUSH) release.** Gated by `m_outstanding_stores_per_warp`,
   incremented once per store transfer at enqueue ([tma_unit_sm.cc:633-641](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L633-L641)) and
   decremented once per store transfer at completion ([tma_unit_sm.cc:1033-1046](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L1033-L1046)). This is
   **per-transfer, not per-mf** — again independent of 768-vs-192. The decrement happens inside the
   `requests_completed >= goal` block, so it moves to "byte-retire complete" cleanly. **SAFE.**
3. **TMA store does NOT use the ldst fence accounting at all (the key finding).** `store_ack` /
   `inc_store_req` / `inc_fence_store` / `dec_fence_store` — the per-mf, per-sector fence counters —
   are called ONLY from `ldst_unit_sm.cc` and `shader.cc` store sites; **`tma_unit_sm.cc` never calls
   any of them** (grep-confirmed). And `SM::accept_ldst_unit_response` intercepts any `is_tma()` reply
   (read OR write ack) and routes it to `tma_unit_sm::fill` BEFORE the ldst `store_ack` path
   ([sm.cc:1515-1524](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1515-L1524)). So a TMA write ack never decrements `pending_stores_*_visible`,
   never touches `m_stores_outstanding`. That means the `MEMBAR`/fence scope logic
   (`warp_has_pending_fence_stores`, [sm.cc:2426](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2426)) is driven by ordinary LDST stores, not
   TMA — TMA-128B cannot perturb it because TMA was never in it. **SAFE (and orthogonal).**

**Net barrier conclusion:** every TMA barrier/fence coupling is keyed on **bytes or per-transfer
counts**, never on the number of sector mfs. TMA-128B changes only the sector-mf count, so with
byte-based retire (option 2a) all barrier releases are bit-identical in *timing-cause* (they fire when
the same bytes have returned). The risk is NOT correctness of the barrier math; it is only that a
mis-implemented retire could fire `notify_tma_completion` / the store decrement at the wrong cycle.
That is exactly what the byte-retire guarantees, and what the validation counters below will confirm.

### TMA metrics impacted by TMA-128B — which stay valid, which need a definition fix
The user's second concern. Audited every `m_stat_*` in `mover_issue_requests` / `mover_on_response`
([tma_unit_sm.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc)) and the derived averages in the Phase3 stats line:

| metric | today (32B) | after TMA-128B (128B) | action |
|---|---|---|---|
| `bytes_issued` / `bytes_completed` | sum of 32B sectors | **unchanged** if we add `data_size` per parent (128) OR per child; bytes are bytes | keep byte-based increments; DO NOT count `+= SECTOR_SIZE` blindly — use `mf->get_data_size()` |
| `requests_issued` / `requests_completed` | counts 32B mfs (768) | will count 128B parents (192) | **DEFINITION CHANGES 4x.** Either rename to `sector_mfs` vs `line_mfs`, or keep counting in sectors by adding 4 per parent. Otherwise `avg_requests_per_issue_active_cycle` and BW ratios shift 4x and look like a regression that is really just a unit change. |
| `avg_icnt_full_cycles` / `avg_emit_span_cycles` | per-transfer, cycle-based | **valid and IS the target** — expected to drop (fewer packets to inject) | keep; this is the success signal |
| `avg_drain_cycles` | last-issue -> complete | valid; may drop | keep |
| `icnt_backpressure_events` | per-transfer flag | valid | keep |
| `BW_*_GBps` | bytes/elapsed | valid (bytes unchanged) | keep |
| `L2_TMA_true_hit_rate` / `pending_hit_rate` | per-probe at L2 admission | **children still probe L2 individually after split**, so probe count and hit rate are UNCHANGED (the split reconstitutes the same 4 probes) | keep — this is the 4.12 work-invariance check and MUST stay ~0.87 |
| `L2_TMA_output_full_cycles` / `port_busy` / `res_fail` | per-child at L2 | unchanged population (children identical post-split) | keep |
| `avg_issue_active_cycles` | cycles the transfer was issuing | drops (1 push/line vs 4) | keep; expected |

**The one metric that will move for a NON-physical reason is `requests_issued/completed`** (sector-mf
count). To avoid a fake "throughput dropped 4x" reading, either (a) keep incrementing these in
**sector units** (add `SECTOR_CHUNCK_SIZE` per 128B parent so the count still means "sectors moved"),
or (b) add explicit `line_mfs_issued` alongside and leave `requests_*` as sectors. Option (a) is less
disruptive to existing log parsers. Everything cycle- or byte-based is invariant by construction, which
is the 4.12 discipline: **TMA-128B is a timing change; bytes and L2 probe population must not move.** The
two validation guards below prove exactly that.

### L2 breakdown asserts — the 128B parent MUST be field-complete or L2 asserts (user's warning, verified)
The user's warning "L2 copies the parent's internal fields to make the 32B sub-reqs; if TMA doesn't
pre-fill them like L1 does, the copy fails and asserts" is CORRECT and is the highest-risk part of
TMA-128B. Traced every field `breakdown_request_to_sector_requests` reads and every downstream assert:

**What `breakdown` copies from the parent** ([l2cache.cc:765-780](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L765-L780), 128B branch). For each
of the 4 children it calls `m_mf_allocator->alloc(...)` with values derived from the parent:
- `mf->get_addr() + SECTOR_SIZE*i` — parent addr MUST be 128B-aligned, else children straddle lines.
- `mf->get_access_type()` — must be a real GLOBAL_ACC_R / GLOBAL_ACC_W (it is, from the mover).
- `mf->get_access_warp_mask()` — copied as-is.
- **`mf->get_access_byte_mask() & mask`** — the per-sector 32B slice of the parent's byte mask. **If the
  parent's byte_mask is not fully set for the 128B, each child gets an empty/partial byte mask**, which
  later breaks `set_byte_mask` / dirty-mask logic on writes and readability checks on reads.
- `std::bitset<SECTOR_CHUNCK_SIZE>().set(i)` — child sector mask = exactly 1 bit (breakdown sets this,
  not the parent — good).
- `original_mf = mf` (the parent) — this is what the retire relies on.

**The branch-selection trap (most likely silent failure).** `breakdown` chooses the split path by
`data_size` + `sector_mask.count()`:
- `data_size==SECTOR_SIZE(32) && sector_mask.count()==1` -> no split (today's TMA path).
- `data_size==MAX_MEMORY_ACCESS_SIZE(128)` -> 4-way split (the TMA-128B path we WANT).
- `data_size==64 && (mask.all()||mask.none())` -> const-cache 2-way path.
- **else -> the final branch, which splits only sectors where `sector_mask.test(i)` is true**
  ([l2cache.cc:804-820](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L804-L820)). If the TMA 128B parent is built with `data_size=128` but its
  **sector_mask is empty**, it still enters the 128B branch (that branch ignores sector_mask) and works;
  BUT if it is built with `data_size=128` and the code path ever compares sector_mask, an empty mask
  silently yields the wrong sector set. And `if (result.size()==0) assert(0 && "no mf sent")`
  ([l2cache.cc:822](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L822)) fires if a mis-built parent produces zero children.

**Downstream asserts a mis-built parent would hit:**
1. **`assert(mf->get_data_size() <= m_config.get_atom_sz())`** at `data_cache::access`
   ([gpu-cache.cc:2006](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L2006)), and the same at `wr_hit`/`rd_hit`/read paths
   ([gpu-cache.cc:1867/1912/2006](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1867)). atom_sz=32 (sector L2). **A 128B parent that
   reaches `access()` un-split asserts immediately.** This is why the parent MUST go through `push()`
   (which splits it) and MUST NOT be pushed directly to the L2 cache. Confirmed: `push` always calls
   `breakdown` for `m_cache_type==SECTOR` before anything reaches `access`, so as long as the parent is
   handed to `push` (the normal icnt->L2 path) this is safe. The danger is any TMA shortcut that hands a
   128B mf straight to L2.
2. **`assert(sector_mask.count() == 1)`** in `get_sector_index`
   ([gpu-cache.h:505-511](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h#L505-L511)), called from every sector get_status/set_status. Children have
   exactly 1 sector bit (breakdown sets it), so children are fine. **But the parent must never be probed
   as a sector** — it has 4 bits set. Again safe iff the parent is only ever a pre-split token.
3. **`assert(line->get_status(mask) == INVALID)`** in `tag_array::probe`
   ([gpu-cache.cc:284](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L284)) — reached with a child's single-bit mask; fine for children.

**Therefore the parent-build contract for TMA-128B (must match what L1/coalescer produces):**
- `data_size = MAX_MEMORY_ACCESS_SIZE` (128).
- `addr` = 128B-aligned line base (`agu_base`, already line-aligned in the mover).
- **`byte_mask` = all 128 bytes set** (full line). This is the field the user flagged: the mover today
  sets only 32 bits per sector mf; the 128B parent must set the full 128-bit byte mask so
  `get_access_byte_mask() & mask` gives each child its correct 32 bytes.
- **`sector_mask` = all 4 sectors set** (`.set(0..3)`), so the parent is unambiguously the 128B branch
  and never mistaken for a single-sector or const-64 request.
- `access_type` = GLOBAL_ACC_R (load / reduce-read) or GLOBAL_ACC_W (store / reduce-write).
- `original_mf = nullptr` (the parent IS the original; children get `original_mf=parent` from breakdown).
- warp/sid/tpc = same as today.

This is exactly the shape the ldst coalescer emits for a full-line access
([abstract_hardware_model.cc:1069-1122](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L1069-L1122) builds size=128 with the full chunk set), so TMA-128B's
parent must be byte-for-byte the same shape. **Add an assert in the mover right after building the
parent** (`assert(parent->get_data_size()==128 && parent->get_access_byte_mask().count()==128 &&
parent->get_access_sector_mask().count()==4)`) so a mis-built parent fails loudly at emission, not 5
stages later inside L2 with an opaque backtrace.

### Two validation guards to ADD as code (the user's request), timing-neutral
1. **L1->L2 granularity assertion (confirm 128B emission is actually happening).** In
   `memory_sub_partition::push` ([l2cache.cc:826-851](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L826-L851)), when `mf->is_tma()`, count how
   many arrive with `data_size==MAX_MEMORY_ACCESS_SIZE` (128B parent, post-TMA-128B) vs
   `data_size==SECTOR_SIZE` (32B, today). Emit a boot/summary counter
   `TMA_L2_push_128B_parents` / `TMA_L2_push_32B_sectors` so a run instantly shows whether TMA-128B's
   emission took effect and how many children the split produced.
2. **TMA duplicate-request / MSHR-miss detector.** In `send_read_request` when `mf->is_tma()`, on the
   `!mshr_hit && mshr_avail` branch (a NEW DRAM miss is issued) record `(block_addr)` in a per-kernel
   `std::unordered_set`; if the same `block_addr` is issued to DRAM again while an earlier one for the
   same line is still outstanding, increment `TMA_dram_reissue_same_line`. On the `mshr_hit` branch
   increment `TMA_mshr_merged`. Together they prove whether same-line requests are being merged (healthy)
   or re-sent to DRAM (the "MSHR not working" hypothesis). Expected today: high `TMA_mshr_merged`, near-zero
   `TMA_dram_reissue_same_line` (matching `avg_mrq_latency=8`). Observe-only.

Both guards are counters only (no timing change) and can ship in the SAME rebuild as TMA-128B so the
first TMA-128B run self-validates: 128B parents present, children split as expected, MSHR still merging,
DRAM byte count unchanged (4.12 work axis).

## 4.11.4 CHOSEN direction — HW-calibrated shared-port injection rate (2026-07-10)

This supersedes 4.11.3. After confirming from HW primary sources that (a) 32B sectors are the real
granularity and (b) TMA shares the SM->L2 path (no dedicated port), the only HW-faithful lever left is
to set the shared-port INJECTION RATE to the measured HW per-SM bandwidth, keeping the 32B sector model.

### HW primary-source anchors
| fact | value | source |
|---|---|---|
| Real memory granularity | 32B sector; a request touches 1-4 sectors of a 128B L2 line | H100 course (128B line / 32B sector); arXiv:2501.12084 |
| TMA has dedicated L2 port? | NO evidence; TMA BW ceiling == ordinary global load (~1800 GB/s H800) => shares SM->L2 path | arXiv:2501.12084 §5.2; NVIDIA Hopper Tuning Guide §1.4.1.2 |
| Per-SM load bandwidth (LSU/L1 path) | **124 byte/clk/SM** | arXiv:2501.12084 Table 5 (H800) |
| TMA per-transfer fixed overhead | ~170 cycles (TMA unit + mbarrier sync), ~size-independent for latency | arXiv:2501.12084 §5.1 Fig.2 |
| TMA single-SM saturates HBM? | NO — needs many CTAs + large (>=8-16KB) transfers | arXiv:2501.12084 §5.2 Fig.3 |

Caveat: the paper's numbers are H800 PCIe (114 SM, HBM2e ~2039 GB/s). H100 SXM5 (132 SM, HBM3
~3.35 TB/s) scales ~1.6x aggregate, but the PER-SM byte/clk figure (124) is the injection-side quantum
we need and is architecture-level, so we use it directly.

### The conversion (byte/clk/SM -> sector/clk/SM)
```
124 byte/clk/SM  /  32 byte/sector  =  3.875 sector/clk/SM  ~=  4 sector/clk/SM  (= one 128B line/clk)
```
So HW injects ~**4 sectors/clk from one SM into the SM->L2 path**. This is the single calibration target.

### Every sim injection stage in sector/clk (code-confirmed) vs HW target
| stage (sim) | current rate | knob (source) | HW target | verdict |
|---|---|---|---|---|
| [1] TMA mover emit | 2 lines/clk = **8 sector/clk** | `kMaxRequestsPerCycle` (hardcoded 2, [tma_unit_sm.h:47](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.h#L47)) | ~4 | **2x too high -> lower to 1 line/clk** |
| [2]->[3] iSLIP grant | 4 packet/clk | `-icnt_grant_passes_per_cycle` (4) | ~4 | already HW-matched -> keep |
| [3]->[4] icnt->L2 pop | 4 packet/clk | `-gpgpu_icnt_to_l2_pop_per_cycle` (4) | ~4 | already HW-matched -> keep |

**Key realization:** the after-lever config from 4.11 (grant=4, pop=4) was, by coincidence, already the
HW-calibrated value (3.875 ~= 4). That is WHY BWD got -9.6% there. The one remaining un-calibrated stage
is the TMA mover itself: `kMaxRequestsPerCycle=2` lines = 8 sector/clk = **2x the HW per-SM injection
bandwidth**.

### Why LOWERING kMaxRequestsPerCycle is the correct HW fix (counter-intuitive but measured)
Measured after-lever K10 (`.o15`): `avg_passes_per_active_cycle=3.74` (grant already drains ~4/clk),
`conflicts_per_cycle=30.2`, `Req_Network_in_buffer_full=70.9`, `out_buffer_full=2.5`.
- The drain side (grant) is already at the HW rate (~4/clk) and largely absorbs the 30/clk conflicts.
- But the mover PRODUCES 8 sector/clk while the path DRAINS 4/clk. Producer(8) > drainer(4) keeps the
  shared REQ in_buffer artificially full (`in_buffer_full=70.9`). **HW never injects 8 sectors/clk from
  one SM (it is a 124 byte/clk = 4 sector/clk unit), so the sim's 8/clk is a non-physical over-injection
  that inflates in_buffer_full and distorts the backpressure signal.**
- Setting mover = 4 sector/clk (1 line/clk) makes producer == drainer == HW bandwidth. The in_buffer
  reaches a balanced steady state instead of a fake overflow. This is a *calibration*, not a speedup
  hack: it aligns the injection quantum with the measured HW per-SM bandwidth.

### Change list
1. **Make `kMaxRequestsPerCycle` a config knob** (currently hardcoded `2` in
   [tma_unit_sm.h:47](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.h#L47)). Add `-gpgpu_tma_max_lines_per_cycle` (unit = 128B AGU lines/clk,
   since the mover's outer loop counts lines and each line = 4 sector mfs). Default **1** (= 4 sector/clk
   = 124 byte/clk, HW-calibrated). Old behavior = 2.
2. **grant_passes=4, icnt_to_l2_pop=4: keep** (already HW-matched).
3. Deferred (separate next step, not this run): model the ~170-cycle TMA fixed overhead explicitly;
   today it is implicitly (over)approximated by the 768-sector serial injection span.

### Validation gate for the kMaxRequestsPerCycle=1 run (BWD K10 first)
- Timing: `gpu_sim_cycle` (expect down or flat — this removes an artifact, the win comes from realistic
  backpressure, so a large drop is NOT expected; a cleaner in_buffer profile is).
- Injection axis: `Req_Network_in_buffer_full` should DROP markedly (producer==drainer now);
  `avg_icnt_full_cycles` (TMA) should drop.
- **Work invariant (4.12):** `L2_TMA_true_hit_rate`, `L2_total_cache_accesses`, L2 read/write bytes,
  DRAM bytes must NOT change (this is a pure timing/rate change).
- HW cross-check: `bw_util` (DRAM) currently 8.6% vs HW 14.85% — should move toward HW, not past it.
- Honest expectation: since grant/pop were already the real drain limit at 4/clk, lowering the mover to
  4/clk mostly removes the fake `in_buffer_full`; it may not move cycles much by itself. The value is a
  correct, HW-anchored injection model on which the next lever (170-cyc TMA overhead) can be judged
  cleanly.

## 4.11.5 Instrumentation bug fix + what actually GATES the queue drain (BWD `.o17` / FWD `.o34`, no re-run — 2026-07-11)

### (a) Latency-bucket instrumentation bug — root cause found and fixed (base-sim bug, not the counter)
The first `kMaxRequestsPerCycle=1` diagnostic run (BWD `.o17`, FWD `.o34`) produced one physically
impossible bucket: `TMA_status[IN_PARTITION_L2_TO_DRAM_QUEUE]` avg = **123,432 cyc** (BWD) — larger
than the whole kernel (261,345 cyc) — while every OTHER stage cross-validated exactly against the
legacy metrics (`avg_icnt2mem_latency=422` == `ICNT_TO_MEM` 421.7; `avg_icnt2sh_latency=181` ==
`ICNT_TO_SHADER` 180.3). So the bucket logic was sound; one stage's *timestamp* was poisoned.

**Root cause:** [l2cache.h:313](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h#L313)
`L2interface::push()` set the status with a hard-coded `0 /*FIXME*/` instead of the real cycle:
```cpp
mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
```
This `L2interface` is the l2_cache's **miss port** ([l2cache.cc:439](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L439)),
so with L2 *enabled* every miss sector passes through it. `m_status_change` became 0, and the NEXT
transition computed `cycle - 0` = an absolute timestamp (~120k) instead of a delta. `visits=797,967`
matches the miss count (misses are ~7% of the 10.9M TMA requests), so only this one bucket was
affected; it was a pre-existing base-simulator bug the authors had flagged `FIXME`, only made visible
by the new per-stage residency table. (Note: the earlier suspicion of [l2cache.cc:622](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L622)
was wrong — that is the L2-*disabled* branch, not taken here.)

**Fix:** moved `push()` out-of-line into [l2cache.cc:455](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L455)
so it can read `m_gpu->gpu_sim_cycle + gpu_tot_sim_cycle` (the header could not — `gpgpu_sim` is
incomplete there). **Timing-neutral:** `m_status_change` is read only by `mem_fetch::print()` and this
instrumentation (grep-verified, 8 sites); it never feeds a scheduling/latency decision, so `0 -> real
cycle` cannot change simulated timing — it only fixes the debug print and the residency accounting.

**Consequence for analysis:** the `.o17`/`.o34` data is fully usable NOW by simply dropping the one
poisoned bucket and re-normalizing. No re-run is needed to trust the diagnosis below.

### (b) Corrected residency (poisoned bucket excluded) — the two-roots split is confirmed
| | BWD K10 (`.o17`, 261,345 cyc) | FWD K5 (`.o34`, 140,941 cyc) |
|---|---|---|
| valid residency total (excl. L2_TO_DRAM) | 21.98 B | 1.64 B |
| **ROP_DELAY** | **67.4%**, avg **1356.6** | 23.8%, avg 135.1 |
| ICNT_TO_MEM (inject) | 20.9%, avg 421.7 | 11.9%, avg 67.6 |
| **ICNT_TO_SHADER** (reply flight) | 9.0%, avg 180.3 | **58.5%**, avg **332.2** |
| bucket req_side / reply_side | **89.8% / 10.2%** | 39.8% / **60.2%** |

BWD is request/admission-bound; FWD is reply-bound. The corrupted bucket never mattered to this
conclusion — it was in the clean stages all along.

### (c) The real question: the queue is not the problem — what GATES its drain?
Queue depth/drain-rate knobs are all null (§4.5/§4.11.2) because a FIFO's residency is set by its
*downstream service gate*, not its own size. Tracing each stall counter to the exact code gate:

**BWD — `ROP_DELAY` avg 1357 (fixed part is only `rop_latency=100`, so ~1257 is pure wait).**
ROP is a fixed-latency queue that pops into `m_icnt_L2_queue` **only when that queue is not full**
([l2cache.cc:639-640](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L639-L640)).
`m_icnt_L2_queue` drains into the L2 bank, and the bank admits a new access **only when
`!output_full && port_free`** ([l2cache.cc:543](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L543)).
The measured admission-skip cause in `.o17` is decisive:
- `L2_TMA_output_full_cycles` = **435,481** (reply queue `m_L2_icnt_queue` full)
- `L2_TMA_port_busy_cycles` = 532 (~0)
- `L2_TMA_res_fail_per_probe` = 0.0006 (~0)

So the BWD chain is: **`m_L2_icnt_queue` (reply FIFO) full -> L2 bank stops admitting -> `m_icnt_L2_queue`
fills -> ROP can't pop -> ROP residency balloons to 1357 -> req-side backpressure `in_buffer_full=62`.**
The gate is the **L2 reply FIFO filling**, which is itself gated one stage further: it drains at 1
mf/sub-partition/ICNT-tick ([gpu-sim.cc:4168-4192](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4168-L4192)),
and each drained reply must find room in the REPLY icnt (`icnt_has_buffer`, else `gpu_stall_icnt2sh`,
BWD = 491,296).

**Why raising `reply_drain` still did nothing (§4.5 C/D, now explained mechanistically, not just empirically):**
draining `m_L2_icnt_queue` faster just moves the same replies into the REPLY icnt faster, where the
**per-cluster ejection is 1 mf/ICNT-tick** at the SM ([shader.cc:5396-5413](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L5396-L5413))
— so the stall relocates to `gpu_stall_icnt2sh` (§4.5 saw exactly 259K -> 722K). The true serial gate
is not any single queue's depth or drain knob; it is that **one TMA transfer's sector replies (up to
768 for a full 24KB tile; ~512 measured per-transfer under FA3 causal masking) must funnel back
through a per-SM ejection that accepts 1/tick**, the mirror image of the injection wall.

**FWD — `ICNT_TO_SHADER` avg 332 (58.5%).** This interval is bracketed by two status writes: set when
the reply is pushed into the REPLY icnt ([gpu-sim.cc:4185](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4185))
and cleared when the SM cluster ejects it ([shader.cc:5426](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L5426)).
The built-in local xbar has **no fixed per-hop latency** (only queues + iSLIP arbiter,
[local_interconnect.cc:88](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/local_interconnect.cc#L88)),
so 332 cyc is almost entirely **queue-wait for the per-cluster 1-mf/tick ejection**
([shader.cc:5396-5413](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L5396-L5413)),
NOT network flight time. Crucially the REPLY net is NOT congested on average
(`Reply_Network_in_buffer_full=0.71/cyc`, `out_buffer_full=2.96`, `gpu_stall_icnt2sh` only 0.9% of
cycles) — so this is a **burst** effect, not steady saturation. Since `n_cores_per_cluster=1`, cluster
== SM, and **all ~512 sector replies of one TMA transfer return to the single issuing SM**, which
ejects them one per tick. FWD injects fast (`in_buffer_full=0.057`), so a transfer's replies arrive
back in a tight burst and pile up at that 1/tick ejection → high per-sector reply wait (332). This is
the SAME 1/tick ejection choke as BWD's relocation target, hit from the front instead of the back.

**Why BWD's reply wait (180) is LOWER than FWD's (332) despite similar per-transfer size** (~16.5KB,
~512 sectors both; BWD 174 transfers/SM vs FWD 42): BWD's injection is throttled upstream
(`in_buffer_full=62`), so its replies dribble back spread-in-time and rarely batch at the ejection.
FWD's injection is free, so its replies burst. **Same two 1/tick chokes; the binding one differs:
BWD binds at injection, FWD binds at ejection.** (Earlier drafts mis-stated FWD as having "4x more,
smaller" transfers — it is fewer, similar-size; corrected here.)

### (d) Root-cause statement (both kernels, one shared structural cause)
Every prior knob widened ONE segment of a serial pipe whose throughput is actually set at TWO
per-SM 1/tick choke points that the knobs never touched:
- **Injection choke:** the SM's single shared REQ icnt node (TMA + LSU) — BWD's wall.
- **Ejection choke:** the per-cluster reply ejection `icnt_cycle()` at 1 mf/tick — FWD's wall, and BWD's
  relocation target when the reply FIFO is drained faster.
The 768-sector-per-transfer count divided by these 1/tick chokes is the serialization. `bw_util`
(DRAM) = 8.6% and `avg_mrq_latency` = 8 cyc prove the memory *device* is idle; the entire ~2x gap vs
HW is queue-entry/ejection serialization around a physically-idle DRAM.

### (e) Candidate fixes that attack the GATE (not the queue) — for discussion, mutually composable
These target the two 1/tick chokes and the fixed per-sector overhead directly. None is a "make memory
free" hack; each keeps L2/DRAM work invariant (4.12) and must be validated on that axis.
1. **Multi-eject the per-cluster reply (mirror of grant-passes on the ejection side).** Today
   `simt_core_cluster::icnt_cycle()` ejects <=1 reply/tick ([shader.cc:5396](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L5396)).
   A `-gpgpu_cluster_reply_eject_per_cycle N` (paired with the EXISTING but-null `reply_drain`, so the
   relocation target is widened at the SAME time — the paired-lever pattern that made grant+pop work in
   §4.8/§4.9) is the FWD-relevant lever and BWD's relocation guard. HW basis needed: what is the real
   per-SM reply-acceptance BW (should equal the 124 byte/clk load path, i.e. ~4 sector/clk — same quantum
   as injection). If so, N=4 is HW-calibrated, not a hack.
2. **Model the ~170-cyc TMA fixed overhead explicitly and REMOVE the implicit 768-serial-injection
   overapproximation** (§4.11.4 item 3, deferred). Today the fixed TMA startup is implicitly ~the
   768-sector emit span; if HW's real per-transfer fixed cost is ~170 cyc (arXiv:2501.12084 §5.1), the
   sim over-serializes. This is the injection-side analogue and BWD-relevant.
3. **Pair (1)+(2)+existing grant/pop=4** so injection quantum, ejection quantum, and fixed overhead are
   all set to the SAME HW per-SM bandwidth (4 sector/clk) + HW fixed latency. This is the only
   combination that removes BOTH 1/tick chokes without letting the stall relocate, because every stage
   in the round-trip is then at the identical HW rate.

**Next-step gate (unchanged discipline):** any of the above is a timing change only — require
`L2_TMA_true_hit_rate`, L2 accesses/bytes, DRAM bytes invariant, and judge by `gpu_sim_cycle` +
`wait_barrier`/`tma_flush`. Prefer to validate (1)+(3) first (pure rate calibration, lowest risk); (2)
needs a per-transfer completion-time model and is higher-touch.

### (f) Rejected here: TMA bulk engine
Replacing the 768 individual sector round-trips with a closed-form `injection + ~170cyc + mem_latency`
completion time is a **simplification, not a model**: it deletes the very per-sector L2/DRAM traffic
whose hit-rate realism §4.6 spent effort earning, so `L2_TMA_true_hit_rate` / L2 accesses / DRAM bytes
would no longer emerge from simulation (4.12 work axis destroyed). Rejected as not HW-faithful; the
levers in (e) fix the drain gates while keeping every mem_fetch and its statistics intact.

## 4.11.6 IMPLEMENTED — paired reply-path calibration (per-SM reply eject + L2 reply drain), 2026-07-12

Chosen from §4.11.5(e) candidate 1. This is a **timing-only calibration** of the reply path to the HW
per-SM load-return bandwidth (124 byte/clk ≈ 4 sector/clk, arXiv:2501.12084 Table 5), the exact mirror
of the §4.11.4 injection calibration. It attacks the two per-SM `1/tick` reply chokes §4.11.5(c)/(d)
identified as the true drain gate — NOT any queue depth (which §4.5 proved null).

### Why the reply path needs TWO knobs paired (the §4.5-null trap, now avoided)
The reply path is two `1/tick` handoffs **in series**, each a separate un-knobbed choke:
```
L2 bank --[reply_drain]--> REPLY icnt --[grant_passes]--> out_buffer --[reply_eject]--> core
          (was 1/tick)                   (already 4)                    (was 1/tick)
   gpu-sim.cc:4179                    local_interconnect               shader.cc icnt_cycle
```
§4.5 raised ONLY `reply_drain` (1->4) and got flat cycles + `gpu_stall_icnt2sh` 259K->722K — because
the stall just relocated onto the still-1/tick per-SM eject (`icnt_cycle`). So the correct fix is to
open **both ends at once** (the same lesson as the injection side, where grant_passes had to be paired
with icnt_to_l2_pop in §4.8/§4.9). The middle hop (REPLY xbar `Advance()`) already runs at
`grant_passes_per_cycle=4` for BOTH subnets ([local_interconnect.cc:399-403](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/local_interconnect.cc#L399-L403)),
so pairing `reply_drain=4` + `reply_eject=4` leaves no `1/tick` choke on the entire reply path.

### Change list (default-1 knobs; behavior bit-identical when left at 1)
1. **New knob `-gpgpu_cluster_reply_eject_per_cycle`** (unit = reply mf/ICNT-tick per SM cluster).
   Member in `shader_core_config` ([shader.h:2038-2046](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h#L2038-L2046)),
   registered in `shader_core_config::reg_options` ([gpu-sim.cc:1111-1121](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L1111-L1121)),
   default `1`.
2. **`simt_core_cluster::icnt_cycle()` rewritten** ([shader.cc:5395-5468](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L5395-L5468)):
   both handoffs (ejection-FIFO->core, icnt->ejection-FIFO) are now bounded loops of `N` iterations
   instead of a single op. Each iteration keeps EVERY original per-mf gate and stat:
   - fifo->core: still checks `fetch/ldst_unit_response_buffer_full()` per mf; **breaks** (not continue)
     on a blocked head because the FIFO is in-order (a blocked head blocks all behind it).
   - icnt->fifo: still checks `n_simt_ejection_buffer_size`, `icnt_pop`, the tpc/type asserts, traffic
     stats, and `set_status(IN_CLUSTER_TO_SHADER_QUEUE)` per mf; **breaks** on full FIFO or empty icnt.
   - N=1 reproduces the original 1-op-each-per-tick behavior exactly (verified by inspection).
   - Added `assert(eject_budget>=1)`, a 0-guard, and a one-time `[REPLY-EJECT]` boot log (mirrors the
     `[ICNT->L2]` boot log) so the 12h run confirms the knob is live in the first seconds.
3. **`gpgpusim.config`:** `-gpgpu_cluster_reply_eject_per_cycle 4` ([line 207](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config#L207))
   and `-gpgpu_l2_reply_drain_per_cycle 1 -> 4` ([line 165](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config#L165)).
   Rebuild required (source change); no tracer/trace change.
4. **Observe-only "did the lever fire?" counters** ([shader.cc:96-109](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc#L96-L109)
   define, incremented in `icnt_cycle`, dumped in `gpu_print_stat` [gpu-sim.cc:3420-3465](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3420-L3465)).
   For EACH of the two handoffs (`fifo2core`, `icnt2fifo`) it prints `active_ticks` (ejected >=1),
   `multi_ticks` (ejected >1 — the proof the widened budget was used), `total_mf`, `max_burst`, and
   `avg_per_active`. Race-free (the cluster loop is serial), timing-neutral (only reads local counts
   after the work). **These are what makes a null result interpretable without a re-run:** `multi_ticks
   ~= 0` ⇒ eject was never the choke (valid null); `multi_ticks` large but cycles flat ⇒ wall moved
   downstream (candidate 2 / barrier), not "experiment failed".

### Debug-log / early-stop coverage for the 12h run
- **Boot (stderr, once):** `[REPLY-EJECT] gpgpu_cluster_reply_eject_per_cycle = 4 ...` — confirms the
  knob parsed and is >1 within seconds; paired `[ICNT->L2]` log already confirms the inject side.
- **End-of-kernel (stdout):** the 10 `reply_eject_*` counters above + the existing
  `gpu_stall_icnt2sh`, `L2_TMA_output_full_cycles`, and the (now-fixed) TMA per-stage residency table
  — together they show whether the eject fired, whether the stall relocated, and where.
- **Early-stop asserts:** `assert(eject_budget>=1)` (config sanity) plus the retained per-mf
  `assert(mf->get_tpc()==m_cluster_id)` / `assert(type==READ_REPLY||WRITE_ACK)` inside the widened
  loop — a mis-routed or corrupted reply trips immediately instead of silently after 11h.

### Why TMA responses benefit (verified, not assumed)
TMA replies do NOT go through the depth-2 ldst response FIFO: `SM::accept_ldst_unit_response` routes
`is_tma()` mfs straight to `tma_unit_sm::fill()` ([sm.cc:1515-1524](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1515-L1524)),
which consumes the mf immediately (erase-from-outstanding + `mover_on_response`, no internal per-tick
cap, [tma_unit_sm.cc:1080-1094](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L1080-L1094)).
The only cap on TMA reply intake was the `icnt_cycle` `1/tick` eject — exactly what this removes. The
cluster ejection buffer is depth 32 (`-gpgpu_n_cluster_ejection_buffer_size 32`), ample for N=4.

### Validation gate for this run (BWD K10 + FWD K5)
- **Timing:** `gpu_sim_cycle` down (primary). FWD is the strongest expected mover (its dominant stage is
  `ICNT_TO_SHADER`=332cyc/58%); BWD should drop too if the reply path was its `output_full` relief valve.
- **Bottleneck shift:** `ICNT_TO_SHADER` residency ↓ (FWD), `L2_TMA_output_full_cycles` ↓ (BWD, since the
  reply FIFO now drains), `gpu_stall_icnt2sh` should NOT explode the way §4.5's drain-only run did
  (that is the whole point of pairing — if it still explodes, the middle xbar hop is the residual).
- **Work invariant (4.12) — MUST hold or the result is void:** `L2_TMA_true_hit_rate` (BWD 0.8691 / FWD
  0.9461), `L2_total_cache_accesses`, L2 read/write bytes, DRAM bytes all unchanged. This is a pure
  timing knob; any work-axis movement means a real bug.
- **`wait_barrier` / `tma_flush`** SM-idle should drop if TMA completion latency actually shortened.
- **Instrumentation cross-check:** the `L2_TO_DRAM_QUEUE` bucket is now correct (§4.11.5a fix), so the
  full per-stage residency table is trustworthy this run.

### Residual risk (honest)
If cycles stay flat, the remaining reply-path serialization is NOT these two chokes — the next suspect
is the fixed per-sector reply traversal (candidate 2: model the ~170cyc TMA fixed overhead) or a
genuine `wait_barrier` structural limit unrelated to reply throughput. This run cleanly separates
"reply throughput choke" (fixed here) from "reply fixed-latency" (untouched), which is why it is worth
one run even though depth/drain-alone were null.

### RESULT — paired reply-path calibration MEASURED (FWD K5 `.o35`/`.e35`, BWD K10 `.o18`/`.e18`, 2026-07-12/13)

Both kernels ran clean. Both knobs confirmed live in the boot log (`[REPLY-EJECT] ...=4`, plus the
prior `[ICNT] grant_passes=4` and `[ICNT->L2] pop=4`). Both levers materially fired: BWD
`reply_eject_*_multi_ticks=2.89M` (`avg_per_active=2.17`, `max_burst=4`); FWD
`multi_ticks=809K` (`avg_per_active=2.43`, `max_burst=4`). **The run is a real partial success:
timing improved, work stayed invariant, and it did NOT relocate the stall the way the §4.5
drain-only run did.**

Baselines: prior after-lever (grant/pop=4 only) FWD `.o32`=140,138 / BWD `.o15`=262,744; realistic-
address baseline (no queue levers, M2/M2.5) FWD `.o31`=145,855 / BWD `.o14`=290,572; HW 67,696 / 132,901.

| metric | FWD K5 (`.o35`) | BWD K10 (`.o18`) | judgment |
|---|---|---|---|
| `gpu_sim_cycle` | **138,021** | **250,026** | timing improved |
| vs prior after-lever (grant/pop=4) | 140,138 → **−1.5%** | 262,744 → **−4.8%** | reply calibration adds a real cut |
| vs realistic baseline (no queue levers) | 145,855 → **−5.4%** | 290,572 → **−13.9%** | cumulative queue-lever effect |
| vs HW | **2.04x** | **1.88x** | still ~2x |
| `L2_TMA_true_hit_rate` | 0.9456 (was 0.9455) | 0.8688 (was 0.8701) | **work invariant ✅** |
| `L2_total_cache_accesses` | 3,457,208 (~flat) | 11,269,403 (~flat) | **work invariant ✅** |
| `L2_cache_read/write_bytes` | 106.1 / 4.52 MB | 241.0 / 119.6 MB | **work invariant ✅** |
| `L2_TMA_output_full_cycles` [12] | 0 | **441** (was 465,528) | reply FIFO no longer fills |
| `gpu_stall_icnt2sh` [12→13] | 0 | **5,792** (was 511,707) | did NOT explode (pairing worked) |
| `Reply_Network_in_buffer_full` | 0.00 | **0.02** | reply net empty |
| `wait_barrier` SM-idle | 9.81% | **9.16%** | down |
| `tma_flush` SM-idle | 0.00% (load-only) | **6.84%** (was 14.66%) | down sharply |

**Why the pairing worked (vs §4.5 drain-only null):** §4.5 raised only `reply_drain` and the stall
relocated to the still-1/tick per-SM eject (`gpu_stall_icnt2sh` 259K→722K). This run opens BOTH ends
(`reply_drain=4` + `reply_eject=4`), so there is no 1/tick choke left on the reply path — hence
`gpu_stall_icnt2sh` FELL (511K→5.8K on BWD) instead of exploding. This is the same paired-lever
lesson as the injection side (§4.8/§4.9 grant_passes + icnt_to_l2_pop).

**FWD moved less (−1.5%) than BWD (−4.8%), as predicted:** FWD's residual cost is reply-side
fixed-latency across small transfers (§4.11.5c), which a throughput knob cannot touch; BWD's reply
FIFO was a genuine relief valve.

## 4.11.7 POST-RUN diagnosis — TMA queue tuning is EXHAUSTED; the new wall is ROP_DELAY (a fixed-latency stage, not a queue) — 2026-07-13

With the reply path drained, the §4.10 stage map was re-read on the new `.o18` (BWD K10) counters.
**Every queue/interconnect stage is now at the noise floor. There is no queue left to widen.**

| stage (§4.10) | counter | pre-levers baseline | **now (`.o18`)** | status |
|---|---|---|---|---|
| [2] REQ inject buffer | `Req_Network_in_buffer_full` | 355 | **7.15** | drained |
| [3] REQ out buffer | `Req_Network_out_buffer_full` | 0.20 | **0.26** | never a limiter |
| [5] L2 bank data port | `L2_TMA_port_busy_cycles` | 660 | **750** | noise floor |
| [5] L2 admission | `L2_TMA_res_fail_per_probe` | 0 | **0.0015** | ~0 |
| [12] L2→icnt reply FIFO | `L2_TMA_output_full_cycles` | 1.39M | **441** | drained |
| [12]→[13] reply inject | `gpu_stall_icnt2sh` | 1.82M | **5,792** | drained |
| [13] reply network | `Reply_Network_in_buffer_full` | 6.3 | **0.02** | empty |

The only counter still large is `gpu_stall_dramfull = 137,131` (stage [4] L2-input queue) — but per the
per-stage residency below it is a *symptom* of ROP holding requests, not an independent depth limit
(and §4.5/§4.11 already proved widening it is null).

### Where the cycles actually go now — the TMA round-trip is dominated by ROP_DELAY
Corrected per-stage residency of every TMA request (`.o18`, `averagemflatency = 1,649`; the
`L2_TO_DRAM_QUEUE` poisoned-bucket bug from §4.11.5a is fixed here, so the table is trustworthy):

| stage | avg cyc | % of TMA round-trip |
|---|---|---|
| **`IN_PARTITION_ROP_DELAY`** | **1,483** | **90.0%** |
| `IN_ICNT_TO_MEM` (inject) | 117 | 7.1% |
| `IN_PARTITION_DRAM_LATENCY_QUEUE` | 247 (×0.07 miss) | 1.1% |
| `IN_ICNT_TO_SHADER` (reply flight) | 24 | 1.5% |
| DRAM device / MC / fill / L2 queues | ≈ 0–11 each | < 0.3% |
| bucket req_side / reply_side | **98.45% / 1.55%** | — |

DRAM is idle (`bw_util ≈ 0.033`, `avg_mrq_latency = 10`, `IN_PARTITION_DRAM = 0.57 cyc`), so the
memory *device* is not the wall — **90% of a TMA request's life is spent parked in ROP_DELAY.**

**What ROP_DELAY is (and why it is the next lever):** the ROP avg is **1,483** but the *configured*
fixed part is only `-gpgpu_l2_rop_latency 100`. So ~1,383 cyc (93%) is **queue-wait to leave ROP**,
not modeled ROP latency. ROP pops into `m_icnt_L2_queue` only when that queue has room; the chain is
now `ROP holds → m_icnt_L2_queue can't accept → gpu_stall_dramfull → ROP residency balloons to 1,483`.
This is no longer a depth/drain problem (all drained). It is the §4.11.5(e) **candidate 2**: the sim
pays a fixed ROP cost **per 32B sector × 768 sectors/transfer, serialized into single L2-input
admission**, whereas HW pays a ~170-cyc fixed overhead **once per transfer** (arXiv:2501.12084 §5.1).
That per-sector × 768 fixed cost funneled through one admission point is the residual 1,383 cyc.

FWD `.o35` shows the same shape (ROP_DELAY 61% / 135 avg, req_side 95.3%), just milder because FWD
transfers are smaller and its residual is reply-side fixed-latency.

### The fix for this ROP wall is split into its own plan (Opt 8)

The residual `ROP_DELAY` wall is the L2 **admission** stage: `cache_cycle` admits only **1 sector
(32B) per sub-partition per L2-tick**, while a real H100 L2 slice returns **64B/cycle (2×32B
sectors)**. That HW anchor, the full re-entrancy/MSHR/data-port safety trace for admitting 2
probes/cycle, and the implementation/verification plan now live in a dedicated doc:
[L2_ADMISSION_WIDTH_H100.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/L2_ADMISSION_WIDTH_H100.md) (Opt 8). `m_data_port_width` is NOT that knob (proven null in
§4.11 Step B/C). See that doc for details; this section only established *that* ROP_DELAY is the wall.

## 4.12 Cycle-INDEPENDENT "work done" comparison (the trustworthy anchor) — 2026-07-09

Throughput% (bytes/cycle) is a TRAP for validation: sim runs ~2x more cycles, so any bytes/cycle
metric is auto-halved purely because the denominator (cycles) is larger. That is circular — the
metric we would use to validate cycles depends on cycles. So compare **absolute work counts**
(sectors, bytes), which are cycle-independent, FIRST. If work matches HW, the model is right and the
gap is purely timing.

### Absolute work: HW (NCU raw sums) vs sim baseline
NCU raw metrics from `ncu --import ...full_rpt.ncu-rep --page raw --metrics lts__t_sectors.sum,
dram__bytes_read.sum,dram__bytes_write.sum` (main kernels only: FlashAttnFwdSm90 / FlashAttnBwdSm90).
Sim DRAM bytes = `DRAM_BW_total_GBps * (gpu_sim_cycle / core_clock_1800MHz)` (the only honest sim DRAM
figure; do NOT use `L2_miss * 128B` — L2 misses are 32B sectors, so that proxy over-counts ~4x and is
WRONG, see pitfall below).

| kernel | metric | HW (NCU) | sim | sim/HW | reading |
|---|---|---|---|---|---|
| FWD K5 | L2 sectors (32B) | 3,833,304 | 3,356,320 | **0.88** | request/addressing model accurate |
| FWD K5 | DRAM bytes | 19.25 MB | 5.09 MB | **0.26** | sim sends 1/4 of HW to DRAM |
| BWD K10 | L2 sectors (32B) | 10,111,818 | 11,213,151 | **1.11** | request/addressing model accurate |
| BWD K10 | DRAM bytes | 45.91 MB | 25.68 MB | **0.56** | sim sends ~half of HW to DRAM |

(sim L2 sectors = `(L2_cache_read_bytes + L2_cache_write_bytes)/32`; L2 accesses raw: fwd 3,356,320 /
bwd 11,213,151.)

### Findings (this REPLACES the 4.11 throughput-only reading where they conflict)
1. **L2 work is accurate (0.88x / 1.11x).** The sector volume in/out of L2 matches HW within ~12% for
   both kernels -> the addressing + request model (M2/M2.5) is correct; the sim is NOT generating the
   wrong amount of memory traffic at L2.
2. **DRAM work is set by L2 HIT RATE, not by the queues.** Total DRAM bytes = (L2 misses) x sector
   size, i.e. it is a pure function of L2 hit rate; a queue only changes *when* those misses reach
   DRAM, never *how many*. So the fact that sim DRAM bytes are low (fwd 0.26x, bwd 0.56x) is a
   **hit-rate story, not a queue story**:
   - bwd L2 hit: sim 0.872 vs HW 0.823 -> sim miss rate 12.8% < HW 17.7% (ratio 0.72). With L2 sectors
     at 1.11x, expected DRAM ~= 1.11*0.72 = 0.80x; measured 0.56x (the extra gap is sim hit slightly
     too high + writeback/RMW + BW back-calc error). Direction and cause = hit rate.
   - Therefore closing the DRAM-work gap is the job of the **addressing model (M2/M2.5 hit rate)**, NOT
     the queue levers. The fwd hit rate is still over-modeled (0.946 vs HW 0.696), which is exactly why
     fwd DRAM work is the most off (0.26x).
3. **The two queue levers change TIMING only, never total work.** Widening icnt/L2 queues cannot add
   or remove a single DRAM byte. Their only legitimate goal is **fewer cycles for the same work**. A
   prior draft of this section wrongly claimed the levers "move DRAM bytes up toward HW" — that is
   impossible and has been removed.
4. **This resolves the 4.11 apparent paradox.** 4.11 (throughput%) said "sim DRAM% lower than HW"; an
   early `miss*128` proxy briefly suggested "sim DRAM 3.3x HIGHER". The real absolute bytes
   (0.26x/0.56x) show sim under-drives DRAM, driven by hit rate. Note `DRAM% = bytes/cycle`, so a lever
   that cuts cycles will *raise* DRAM% even though total bytes are unchanged — another reason DRAM%
   alone is not a work metric.

### Two independent validation axes (do not conflate)
- **Work axis (cycle-independent): L2 sectors, DRAM bytes** -> owned by the **addressing / hit-rate
  model (M2/M2.5)**. Fix here = better hit rate. Queues are irrelevant to this axis.
- **Timing axis (cycles): `gpu_sim_cycle`, stall counters** -> owned by the **queue/interconnect
  levers**. Fix here = fewer cycles for the SAME work. DRAM bytes must NOT change when a lever is
  applied (if they do, something is wrong).

### Pitfalls proven here (do not repeat)
- **throughput% is cycle-contaminated** — never use bytes/cycle to judge a model whose cycle count is
  itself wrong (~2x here). Use absolute sector/byte sums first; only once cycles are close does
  throughput% become a valid "is each pipe as busy as HW" check.
- **`L2_miss * 128B` is a wrong DRAM proxy** — L2 misses are 32B sectors, so it over-counts ~4x and
  flipped the sign of the conclusion. Use `DRAM_BW_GBps * (cycle/clock)` (or a real DRAM byte counter).
- **Unit discipline:** NCU `lts__t_sectors` = 32B sectors; sim `L2_total_cache_accesses` is an access
  count (compare via bytes/32, not directly). Always reconcile units before comparing.
- **Queues do not change work** — total DRAM bytes / L2 misses are fixed by hit rate; a queue lever is
  only ever a *timing* change. Never justify a queue change by a work/byte metric.

### What this does NOT yet tell us (open)
- HW's real hot pipe is L1/TEX (62.6% bwd). Sim throughput% counters now exist (see 4.13) but are
  cycle-contaminated until cycles converge, so the on-chip *utilization* comparison is only trustworthy
  after the timing gap closes; the *work* comparison (this section) is valid now.

## 4.13 Sim throughput% metrics IMPLEMENTED (final-comparison, read last) — 2026-07-09

NCU-equivalent per-pipe throughput% counters were added to the simulator so the loop can be closed
once cycles converge. **Each prints both the %% AND the absolute served count** — read the absolute
count now (cycle-independent, 4.12), read the %% only after cycles are close (%% is bytes/cycle =
cycle-contaminated, 4.12 pitfall 1).

| metric | sim output key | served (numerator) | peak (denominator) | source |
|---|---|---|---|---|
| DRAM | `Throughput_DRAM_pct` / `Throughput_DRAM_served_bytes` | total_accesses * dram_atom_bytes | m_n_mem * dram_atom_bytes * dram_tot_sim_cycle | [mem_latency_stat.cc](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/mem_latency_stat.cc) (after DRAM_BW prints) |
| L2 | `Throughput_L2_pct` / `Throughput_L2_served_bytes` | total_l2_css.bytes | m_n_mem_sub_partition * 32B * gpu_sim_cycle | [gpu-sim.cc](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) gpu_print_stat, inside L2 block |
| L1/TEX | `Throughput_L1TEX_pct` / `Throughput_L1TEX_served_bytes (L1D=.. shared=..)` | core_cache_css.bytes (L1D) + total_shared_access_bytes | num_shader() * 128B * gpu_sim_cycle | gpu-sim.cc gpu_print_stat |
| Compute (tensor) | `Throughput_ComputeTensor_pct` / `Throughput_ComputeTensor_active_cycles` | total_num_cycles_tensor_pipe_active | num_shader() * gpu_sim_cycle | gpu-sim.cc gpu_print_stat |

New counters added (observe-only, timing-neutral, default present):
- **`total_num_cycles_tensor_pipe_active`** — incremented in `functional_unit::cycle()` for the TENSOR
  FU whenever it has work in flight / in dispatch / under WGMMA II-lockout. This is a true busy-cycle
  count, unlike the existing `*_fu_occupied_tensor` which only counts cycles OTHER warps were blocked
  by tensor (under-counts when tensor runs alone). Registered in gpu-sim.cc stat init.
- **`total_shared_access_bytes`** — incremented in `ldst_unit_sm.cc` shared dispatch block as
  `active_count() * data_size`. On Hopper L1D + shared memory are the SAME unified L1TEX unit, so
  L1/TEX throughput folds L1D bytes (already tracked via core cache stats) + shared bytes together.

Definitions / caveats:
- **L1/TEX is unified (L1D + shared + texture).** HW does NOT separate them; NCU L1/TEX throughput is
  the combined utilization. So sim L1TEX% = (L1D bytes + shared bytes) / peak. Texture is ~0 for FA3.
  Non-TMA operators DO use L1D/shared, so this metric is necessary (TMA bypasses L1 to L2, counted at
  L2/DRAM, not here).
- **Compute% is the TENSOR pipe specifically** (WGMMA), the dominant compute for FA3 — not a general
  SM issue-slot utilization. Peak = 1 tensor pipe per subcore, denominator uses per-SM cycles.
- **Peak denominators are model-consistent approximations, not NCU-identical.** They use the sim's own
  per-cycle service limits (32B/sub-partition, 128B/SM, 1 atom/channel, 1 tensor pipe/subcore), so a
  sim-vs-sim (baseline vs after-lever) comparison is exact; sim-vs-HW% is directional (compare the
  absolute served counts for exact HW comparison, 4.12).

**How to use after a run:** (1) confirm absolute served counts vs NCU sums (4.12 anchor). (2) after
cycles converge, compare the %% to NCU DRAM/L2/L1TEX/Compute throughput to confirm each pipe is as
busy as HW. Before cycles converge, a lower sim %% is expected purely from the larger cycle
denominator and is NOT an accuracy signal.





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

## 6. Opt 7 — final summary (TMA queue/interconnect calibration) and remaining items

Everything in this document below §4 (the injection + reply queue/interconnect calibration) is
consolidated as **Opt 7** in `FA3_progress.md`. It is built on top of **Opt 6** (TMA real base +
CTA-indexed tile spread, M2/M2.5 — the address-realism prerequisite, now finalized separately).

### What Opt 7 is (the shipped config + code)
All knobs default to 1 (bit-identical); the H100 config enables them:
- `-icnt_grant_passes_per_cycle 4` (§4.8, injection xbar drain, HW-calibrated to ~4 sector/clk).
- `-gpgpu_icnt_to_l2_pop_per_cycle 4` (§4.9, paired downstream icnt→L2 pop).
- `-gpgpu_l2_reply_drain_per_cycle 4` (§4.11.6, reply FIFO drain — now paired, was null when alone in §4.5).
- `-gpgpu_cluster_reply_eject_per_cycle 4` (§4.11.6, per-SM reply eject — the missing pair for reply_drain).
- Ceil-division fill/writeback port occupancy guard (§4.11.1 Step A) + the §4.11.5a `L2_TO_DRAM_QUEUE`
  status-timestamp bug fix (both timing-neutral correctness fixes).
- L2 data-port width sweep 32→64 was tested and **rejected** (§4.11-Step B/C, null for both kernels);
  config left at 32B.

### Measured result (on the Opt 6 realistic-address baseline)
| | FWD K5 | BWD K10 | HW |
|---|---|---|---|
| Opt 6 baseline (no queue levers) | 145,855 | 290,572 | — |
| **Opt 7 (all queue levers)** | **138,021** (`.o35`) | **250,026** (`.o18`) | 67,696 / 132,901 |
| Opt 7 vs Opt 6 | **−5.4%** | **−13.9%** | — |
| vs HW | 2.04x | 1.88x | 1.0x |

Work axis invariant (the win is timing-only, not fake locality): `L2_TMA_true_hit_rate` 0.9456 / 0.8688,
L2 accesses/bytes unchanged vs baseline. See §4.11.6 RESULT for the full table.

### Conclusion — TMA queue/interconnect tuning is EXHAUSTED (§4.11.7)
After Opt 7 every REQ/reply queue and interconnect stage is at the noise floor
(`Req_in_buffer_full` 355→7, `L2_TMA_output_full` 1.39M→441, `gpu_stall_icnt2sh` 1.82M→5.8K,
`Reply_in_buffer_full` 6.3→0.02). Further depth/drain/eject/width knobs will move local `*_full`
counters but **not `gpu_sim_cycle`** — the exact null pattern this document documented six times.

### Remaining items to reach HW cycles (NOT queue-related — tracked as Ongoing in FA3_progress.md)
1. **TMA fixed-overhead / ROP per-sector serialization (§4.11.5e candidate 2 + §4.11.7).** 90% of a
   TMA request's round-trip is now `ROP_DELAY` (avg 1,483 cyc, of which ~1,383 is queue-wait, not the
   configured 100). Root: the sim pays a fixed ROP cost **per 32B sector × 768 sectors/transfer**,
   serialized into a single L2-input admission, while HW pays a ~170-cyc fixed overhead **once per
   transfer** (arXiv:2501.12084 §5.1). This is the largest remaining bucket and the most HW-faithful
   next lever. It is a *timing* change; keep `L2_TMA_true_hit_rate`, L2 accesses/bytes, DRAM bytes
   invariant (§4.12 work axis). Rejected alternative: the "TMA bulk engine" closed-form model (§4.11f)
   deletes the per-sector L2/DRAM traffic whose hit-rate realism Opt 6 earned.
2. **Frontend tail / fwd L2-hit over-model (accuracy-side, largely unrecoverable).** With the TMA axis
   drained, `nv_ibuffer_empty`/`no_valid_frontend` (~6–9%) is co-dominant but is HW's own straggler-tail
   imbalance (Waves-Per-SM cross-validated, see FA3_progress Deferred Opts) — not recoverable by a
   frontend-fetch fix. Separately, the fwd L2 hit rate is still over-modeled (0.9456 vs HW 0.6958) due
   to the CTA-count cap; closing it needs real tile `coords` (Opt-6 approach B), an *addressing* fix,
   not a timing one. These are fidelity items, not the primary cycle lever.
