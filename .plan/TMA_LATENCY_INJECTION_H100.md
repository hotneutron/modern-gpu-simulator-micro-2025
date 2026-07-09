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
