# Opt 6 — TMA transfer latency is over-modeled (merged: address-hotspot + sector-merge + injection bandwidth)

> **Merged plan.** This supersedes and combines two earlier drafts that attacked the same symptom
> from different angles:
> - [TMA_ADDR_MERGE_PLAN.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/TMA_ADDR_MERGE_PLAN.md)
>   (synthetic-address L2 hotspot + 32B→128B sector merge). Its "Phase A" (mock base + 128B mf)
>   was **rolled back** (see §5) and is **not** in the current code tree — only its `-tma_debug_*`
>   logging infra was merged.
> - the injection-bandwidth analysis (`kMaxRequestsPerCycle` + shared-icnt back-pressure).
>
> Both are the **same root code defect**; this plan re-implements the fix as one coherent change.
>
> Opt-6 number reused: the original Opt-6 (L1I frontend `stream_buffer_wait`) is deferred
> ([L1I_PREFETCH_LOOKAHEAD_H100.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/L1I_PREFETCH_LOOKAHEAD_H100.md)).
>
> Target: FA3 fwd (k5) / bwd (k10), on top of Opt 5. Goal: reduce sim cycle toward HW **and** make
> the SM-idle breakdown resemble HW.

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

## 2. Root cause (code + measured): one defect, three coupled symptoms

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

**(B) 32B×4 sector explosion at L2 admission (from ADDR_MERGE §1-C / §2-2).**
- Each 128B AGU request is emitted as **4 separate 32B sector mfs** instead of one 128B line, so the
  L2 input queue carries 4x the mf count and in-flight duplicates of the same line are not merged at
  the admission stage. L2 MSHR merge exists but the `icnt_L2_queue` admission probes one at a time,
  so `RESERVATION_FAIL` retries accumulate.

**(C) Synthetic-address hotspot across SMs (from ADDR_MERGE §1-A).**
- AGU base = `(transfer_uid << 20) + agu_index*128`
  ([:633-635](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L633-L635)).
  `transfer_uid` is an **SM-local** counter, so the Nth transfer on every SM hits the **same** L2
  line → an artificial all-SM hotspot → `RESERVATION_FAIL` storms (one sector re-probed 132x over
  cycles 8948–13493). This is the FA3_progress **Arch TODO-2** limitation: the trace cannot supply
  the real GMEM base (NVBit can't read the TMA descriptor cache), so the address is fabricated.

**Why this is wrong for HW.** A real Hopper TMA engine issues a bulk descriptor copy pipelined at
high bandwidth, addressing real per-tensor GMEM regions — not 768 serialized 32B injections into
one shared port at one synthetic hotspot. Bandwidth shows it directly: sim `DRAM_BW_total` fwd
**15.2** / bwd **44.3 GB/s** vs HW NCU `Memory Throughput` fwd **404.9** / bwd **497.8 GB/s**
(1/10–1/27 of HW). The inflated `lat_mem` parks consumer warpgroups on their mbarriers →
`wait_barrier`/`tma_flush` SM-idle → IPC ceiling.

## 3. Fix design (re-implementation; Phase A was rolled back)

The fix is one change with three parts, mapping 1:1 to §2 (A)(B)(C). Implement behind a default-off
flag for clean A/B, since Phase A's rollback showed correctness risk (§5).

**Part 1 — emit one 128B line mf instead of 4×32B sectors (fixes B, and relieves A).**
- AGU 128B request → a single 128B mf (`data_size=128`, full sector/byte mask). Goal counter
  switches from `kSectorMfGoal = agu_requests*4*mfs_per_sector` to `kLineMfGoal = agu_requests *
  mfs_per_line` (reduction RMW = 2 line mfs: read+write).
- `memory_sub_partition::push` → `breakdown_request_to_sector_requests` splits into 32B children +
  MSHR-merges, exactly like the normal L1→L2 path. Parent 128B kept as `original_mf`; children
  inherit the TMA tag.
- Response path: `fill(mf)` resolves a returning 32B child to its parent via `get_original_mf()`,
  retiring the parent only when all children are back (`m_outstanding_sectors[parent]`); handle the
  L2-bypass case where the 128B parent returns directly (remaining=1).
- This alone cuts injected mf count 4x, so the `kMaxRequestsPerCycle`/icnt-full serialization (A)
  shrinks ~4x without touching the icnt model.

**Part 2 — injection bandwidth knob (finishes A).**
- Make `kMaxRequestsPerCycle` a config knob (not a hardcoded `2`) and/or give TMA its own injection
  budget so a bulk transfer drains in tens of cycles. Tune so per-transfer `lat_mem` lands at the
  `rop+dram` round-trip scale (hundreds of cyc), matching HW. Try config-only A/B first.

**Part 3 — address model that removes the cross-SM hotspot without faking L2 hits (fixes C).**
- Re-implement the rolled-back mock base **carefully** (the rollback reason, §5): derive a base from
  `config_id` so the *same tensor* reused across transfers/SMs maps to a shared region (models real
  reuse → legitimate L2 hits), but **spread distinct logical tiles** so a single `config_id` does
  not collapse all tiles onto one line (which over-states L2 hit rate). Since the trace has no tile
  `coords` (confirmed: `TMACommand.coords` is never set), spread by a cheap proxy
  (e.g. `transfer_uid`/`sm_id`-derived tile offset within the config region) instead of a single
  fixed base.
- This is explicitly **not** real-base recovery (that needs trace-gen work; see Arch TODO-2 and
  [tma_tx256b_revert_plan.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/tma_tx256b_revert_plan.md)).
  Goal: remove the artificial hotspot while keeping L2 hit rate close to HW.

*Files:* `tma_unit_sm.{h,cc}` (128B emission, parent/child accounting, base helper, injection
knob), `gpu-sim.cc` + `shader.h` (config flags). Reuse the existing `-tma_debug_*` logging.

## 4. Verification plan (single re-run)

1. **A/B behind the flag** with `-tma_debug_enable 1` on SM 0: confirm per-transfer `lat_mem` drops
   from ~3,000–5,700 to the rop+dram scale; back-pressure events fall; `fill-retire` shows correct
   128B→4×32B split+merge; `line-reuse` shows real reuse (not all tiles collapsed).
2. **Re-run Step-0** (fwd k5 / bwd k10): `wait_barrier` and `tma_flush` SM-idle should fall;
   `nv_ibuffer_empty` (tail-drain) stays flat.
3. **HW alignment cross-check (critical, ties §2 + the rollback risk):**
   - sim `DRAM_BW_total` → toward HW `Memory Throughput` (404.9 / 497.8 GB/s)
   - sim **L2 Hit Rate must stay near HW** (fwd 69.58% / bwd 82.26%) — if Part 3 over-states it,
     the cycle win is an artifact; spread tiles more.
   - sim per-scheduler IPC → toward HW 0.405 / 0.285.
4. Record cycle deltas vs Opt 5 (fwd 149,727 / bwd 241,425) in `FA3_progress.md`.

## 5. Rollback history & risks (why Phase A failed, must not repeat)

- **Phase A was rolled back and never committed** (no `tma_mock_config_base`/`kLineMfGoal`/
  `m_outstanding_sectors` anywhere in the tree; git shows only the `-tma_debug_*` logging landed).
  The runs analyzed here (`.o23`/`.o5`) are the **pre-Phase-A** code (`32B sector`, `sector_mfs=768`,
  synthetic `transfer_uid` base).
- **Risk 1 — L2-hit over-statement (the main Phase-A flaw).** A `config_id`-only fixed base
  collapses all tiles of a tensor onto one region → L2 hit rate balloons above HW, giving a fake
  cycle win. Part 3 must spread tiles and §4.3 must gate on HW L2 hit rate.
- **Risk 2 — 128B↔32B accounting.** Parent(1)↔children(4) retire bookkeeping and the reduce/store
  RMW path (2 line mfs) are error-prone; assert children-complete == 4 and that parents retire
  exactly once.
- **Risk 3 — don't make memory free.** The goal is to remove *injection* serialization, not memory
  pressure; keep rop/L2/DRAM accounting intact and validate against HW `Memory Throughput` /
  `DRAM Throughput`, not just cycles.
- **Out of scope:** real GMEM base recovery (trace-gen / NVBit descriptor-cache limitation, Arch
  TODO-2). Tracked separately; this plan deliberately uses a hotspot-free synthetic address only.
