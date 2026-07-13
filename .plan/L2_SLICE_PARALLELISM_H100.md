# Opt 8 — L2 slice parallelism: admission-rate (throughput) + balanced sub-partition hashing (placement)

> **Scope.** This plan is a single optimization (**Opt 8**) with TWO orthogonal parts on the L2
> sub-partition (=slice), both born from the post-Opt-7 diagnosis that the residual wall is
> `ROP_DELAY` (90% of a TMA request's round-trip):
> - **Part 1 — admission-rate (throughput / "how fast each slice drains"):** the sim admits only
>   **1 sector (32B) per slice per L2-tick** while a real H100 L2 slice returns **64B/cycle (2×32B)**.
>   §3–§9.
> - **Part 2 — balanced sub-partition hashing (placement / "which slice a sector goes to"):** the
>   40-channel (non-2^n) config makes IPoly hash into 128 then fold with `% 80`, double-counting slices
>   0..47 ⇒ up to **2:1 spatial load imbalance** across slices. §10–§11.
>
> The two parts are **complementary, not alternatives**: Part 1 makes each slice faster; Part 2 makes
> the load even across slices. Both are **timing-only** and must keep the §4.12 work axis invariant
> (`L2_TMA_true_hit_rate`, L2 accesses/bytes, DRAM bytes). Both are default-off (bit-identical); the
> H100 config enables both, and they are validated in the same 12h run.
>
> **Origin.** Split out from [TMA_LATENCY_INJECTION_H100.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/TMA_LATENCY_INJECTION_H100.md).
> Opt 7 (that doc) drained every TMA REQ/reply queue and interconnect stage to the noise floor. The
> post-Opt-7 diagnosis (§4.11.7 there) found the residual wall is **`ROP_DELAY` = 90% of a TMA
> request's round-trip (avg 1,483 cyc)**, and traced it to the L2 **admission** stage (Part 1). The
> placement-imbalance root (Part 2) was found later while auditing why sim L2 slices are under-utilized
> vs HW (§7.2, §9). (Formerly named `L2_ADMISSION_WIDTH_H100.md`, renamed once the placement part was
> folded in.)

## 1. Symptom (measured, post-Opt-7)

From bwd K10 `.o18` (Opt 7 baseline, all queue levers enabled), TMA per-request stage residency
(`averagemflatency = 1,649`):

| stage | avg cyc | % of TMA round-trip |
|---|---|---|
| **`IN_PARTITION_ROP_DELAY`** | **1,483** | **90.0%** |
| `IN_ICNT_TO_MEM` (inject) | 117 | 7.1% |
| `IN_PARTITION_DRAM_LATENCY_QUEUE` | 247 (×0.07 miss) | 1.1% |
| `IN_ICNT_TO_SHADER` (reply flight) | 24 | 1.5% |
| DRAM device / MC / fill / L2 queues | ≈ 0–11 each | < 0.3% |
| bucket req_side / reply_side | **98.45% / 1.55%** | — |

`bw_util ≈ 0.033`, `avg_mrq_latency = 10`, `IN_PARTITION_DRAM = 0.57` — the DRAM **device** is 97%
idle. The time is spent **queued to enter the L2 bank**, one sector at a time, not waiting for memory.
The configured ROP fixed part is only `-gpgpu_l2_rop_latency 100`, so **~1,383 cyc (93% of ROP_DELAY)
is queue-wait**, inherited backpressure from the 1-sector/cycle L2 admission downstream
(`gpu_stall_dramfull = 137,131` = L2-input `m_icnt_L2_queue` full).

Root cause in one line: **one 24KB TMA transfer = 768×32B sectors, tiled onto a few sub-partitions,
each draining at 1 sector/cycle.** HW pipelines that as a bulk line stream; the sim serializes it.

## 2. HW anchor — one L2 slice returns 64B/cycle (2 sectors), NOT 128B and NOT 32B

Verified against HW primary/secondary sources so the fix matches HW rather than guessing.

| fact | value | source |
|---|---|---|
| Access/tag granularity | **32B sector**; a 128B line = 4 sectors; sectors of different lines fetch independently | Cornell CVW GPU memory ("128-byte cache lines consist of four 32-byte sectors … 32-byte sector size persists"); arXiv:1509.02308 (Dissecting GPU Memory Hierarchy) |
| Per-slice L2 data throughput (HBM 100-class: V100→A100→H100) | **64 B/cycle per slice** (V100 was 32 B/cycle; 100-class HBM doubled it) | NVIDIA dev-forum L2-throughput thread (Nanodeoclus/Curefab): "later GPUs … 64B/cycle per slice"; GDDR = 4 slices × 32B/cycle per controller |
| L2 organization | slices bound to memory controllers; H100 = 50MB, 5120-bit HBM3 bus | NVIDIA Hopper whitepaper / Wikipedia Hopper |

So the physical L2-slice quantum is **64B/cycle = 2 sectors/cycle**, sitting *between* the sim's
current 1-sector (32B)/cycle admission and a full 128B line/cycle.

**Do NOT conflate with the injection quantum.** The SM→L2 *injection* path is ~4 sector/clk/SM
(124 B/clk, Opt 7 §4.11.4) because it aggregates a whole SM's LSU+TMA. The L2 *slice* is 2 sector/clk
(64 B/clk) because it is one slice. **This Opt's target is 2, not 4.**

## 3. Background — HW memory hierarchy (partition / slice / bank / port)

To see why this Opt fixes the **per-slice throughput** rather than the **number of slices**, you need
the real HW hierarchy. This describes **how the HW actually behaves**, not the sim code.

### Hierarchy (top → bottom)
```
GPU
 └─ Memory Partition (= Memory Controller, owns one HBM channel)   sim: -gpgpu_n_mem 40
     └─ L2 Slice (= sub-partition)  ← the real unit of parallelism  sim: x2/channel = 80
         └─ Data Banks (several SRAM banks)                         sim: not modeled
             └─ Port (SRAM read/write physical path)                sim: m_data_port_width (occupancy only)
```

### (1) Memory Partition / Channel
- One HBM memory controller = one partition = one HBM channel. They run fully in parallel to produce
  the aggregate bandwidth (H100 ~3.35 TB/s).

### (2) Sub-partition = L2 Slice ← the real unit of parallelism
- A partition is typically split into **2 sub-partitions (= L2 slices)** (matches sim
  `n_sub_partition_per_mchannel 2`, 80 total).
- Each slice is an **independent L2 chunk**: its own tag array + its own data banks + its own access port.
- **Why split into slices?** Bandwidth. Splitting one big L2 into many chunks lets each process an
  access simultaneously, so **aggregate bandwidth = #slices × per-slice bandwidth**. Which address goes
  to which slice is spread by physical address hashing (in the sim: `addrdec.cc` + `hashing.cc` IPoly
  indexing, `-gpgpu_memory_partition_indexing 2`, already implemented — see §9).
- **HW throughput: 64 B/cycle per slice** (100-class HBM). This is the "per-sub-partition throughput"
  this Opt targets.

### (3) Bank — why it exists
The data SRAM inside a slice is further divided into multiple **banks**, for two reasons:
1. **Latency/conflict hiding (bank interleaving).** One SRAM bank is briefly busy (cycle time) after an
   access. With several banks, consecutive addresses interleave across different banks so a new bank can
   be hit every cycle without stalling on any single bank's cycle time.
2. **Concurrent access width.** Multiple banks each emitting their share fill the slice's target
   bandwidth in one cycle (e.g. 2× 32B banks × 32B = 64B/cycle).
- If addresses collide on the same bank you get a **bank conflict** → serialization. Interleaving
  spreads them out.

### (4) Port — is it on the bank or the sub-partition?
It exists at different meanings on different levels:
- **slice-level port** = the slice's external interface bandwidth (to/from the upstream interconnect) =
  **64 B/cycle**.
- **bank-level port** = each SRAM bank's physical read/write port. One bank reads **or** writes its
  width in one cycle.
- So **the port fundamentally belongs to the bank**, and the slice's "port bandwidth" is the sum of its
  banks' ports.
- **Read/write independent?** Most GPU L2 SRAM banks are single-ported (R/W shared), so a bank does read
  or write (not both) in a cycle. But **across the slice**, bank A can read while bank B writes in the
  same cycle, so it looks R/W-parallel in aggregate. That parallelism comes from "**many banks**", not
  from "a R/W-split port".

### (5) sim ↔ HW mapping (key point)
| HW concept | Simulator | per-slice throughput |
|---|---|---|
| Memory Partition (channel) | `m_n_mem = 40` | — |
| **Sub-partition = L2 Slice** | x2/channel → **80** | — |
| per-slice admission rate | single `top()`/`pop()` in `cache_cycle` | **1 sector (32B)/cycle** ← HW is **64B (2 sectors)** |
| **Bank** | **not modeled** (no bank interleaving/conflict) | — |
| data port | `m_data_port_width` (occupancy cycles only, not admission count) | — |

**Conclusion:** the unit of parallelism is the slice (sub-partition), and the sim models down to the
slice only (banks/ports are abstracted). So the HW fact "64 B/cycle per slice" is reflected in the sim
as the **per-slice admission count (1→2)** — this Opt. The problem is not too few banks; it is that the
per-slice rate is modeled at half.

## 4. Current code — where the 1-sector/cycle cap lives

The cap is NOT `m_data_port_width` (already proven null in Opt 7 §4.11 Step B/C). It is the
single-`top()`-per-tick admission loop.

### 4.1 The admission loop ([l2cache.cc:532-636](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L532-L636), inside `memory_sub_partition::cache_cycle`)
Called **once per sub-partition per L2-tick** ([gpu-sim.cc:4334](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4334)). Per call it does exactly ONE admission:

```
if (!m_L2_dram_queue->full() && !m_icnt_L2_queue->empty()) {
    mem_fetch *mf = m_icnt_L2_queue->top();       // <-- ONE head only
    bool output_full = m_L2_icnt_queue->full();
    bool port_free   = m_L2cache->data_port_free();
    if (!output_full && port_free) {
        status = m_L2cache->access(mf->get_addr(), mf, ...);   // one probe
        if (status == HIT)              { push reply; m_icnt_L2_queue->pop(); }
        else if (status != RESERVATION_FAIL) { send to DRAM; m_icnt_L2_queue->pop(); }
        // RESERVATION_FAIL: leave at head, retry next tick
    }
}
```

### 4.2 Why `m_data_port_width` does NOT fix it
`use_data_port` charges `ceil(data_size / port_width)` occupancy cycles ([gpu-cache.cc:1144-1176](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1144-L1176)); with `data_size=32, width=32` (and even width=64) that is `ceil(32/32)=ceil(32/64)=1`. It only meters the *occupancy of an already-admitted sector*; it does **not** increase the *number of admissions per cycle*. That number is fixed at 1 by the single `top()`/`pop()` above.

## 5. Trace — is admitting 2 probes/cycle SAFE? (done, before implementing)

Traced every piece of state a second `access()` in the same `cache_cycle` would touch. **Conclusion:
safe, provided each of the 2 probes independently re-checks the same gates the single probe checks
today, and the once-per-tick port-replenish is respected.** Details:

### 5.1 `data_cache::access()` is re-entrant within a cycle — with ONE gate to respect
[gpu-cache.cc:2003-2024](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L2003-L2024) → `process_tag_probe` → `m_rd_hit`/`m_rd_miss`/`m_wr_*`. It carries no
"already-ran-this-cycle" flag. The only intra-cycle side effect that couples two calls is the **data
port**:
- Each `access()` ends with `m_bandwidth_management.use_data_port(mf, status, events)` ([gpu-cache.cc:1994](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1994)), which on a HIT adds `ceil(32/32)=1` to `m_data_port_occupied_cycles`.
- `data_port_free()` is `true` iff `m_data_port_occupied_cycles == 0` ([gpu-cache.cc:1202-1204](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1202-L1204)).
- **Implication:** after the FIRST HIT admission this cycle, `data_port_occupied_cycles` becomes 1, so
  `data_port_free()` would return **false** for the second probe → the naive 2-probe loop that
  re-checks `port_free` would admit only 1. **This is the design decision (see §6):** to model a 64B
  slice we must let the port carry **2 sectors' worth of occupancy per cycle** (i.e. treat the port as
  64B-wide), not gate the 2nd probe on the 1st probe's 32B occupancy. Options in §6.
- `assert(mf->get_data_size() <= m_config.get_atom_sz())` ([gpu-cache.cc:2006](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L2006)) — atom_sz=32, and TMA children are 32B, so each probe satisfies it. **No new assert risk** as long as we still admit one 32B sector *per probe* (we do — 2 probes × 32B, not 1 probe × 64B).

### 5.2 MSHR accepts 2 adds/cycle — it is capacity-bounded, not rate-bounded
`send_read_request` ([gpu-cache.cc:1333-1392](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1333-L1392)):
- `mshr_hit = m_mshrs.probe(addr)`, `mshr_avail = !m_mshrs.full(addr)`, then `m_mshrs.add(...)`. These
  are **capacity** checks (per-entry merge count + total entries), **not** a per-cycle "1 add" limit.
  Two calls in one cycle each probe/add independently and correctly (the 2nd sees the 1st's add — good,
  same-line siblings will MSHR-merge exactly as they do across cycles today).
- dl2 config `A:192:96` → **192 MSHR entries, 96 merges/entry**. Enormous headroom; 2 adds/cycle cannot
  overflow it under FA3 traffic. If it ever did, `send_read_request` already returns
  `MSHR_ENTRY_FAIL`/`RESERVATION_FAIL` (the 2nd probe just fails gracefully and stays at head next
  cycle) — no assert.

### 5.3 miss_queue accepts 2 pushes/cycle — but drains at 1/tick (the real relocation risk)
- The miss path pushes to `m_miss_queue` gated by `m_miss_queue.size() < m_config.m_miss_queue_size`
  ([gpu-cache.cc:1362-1363](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1362-L1363)). `m_miss_queue_size = 32` (dl2 field 4 of `...,32:0,32`). Two pushes in one cycle is fine (capacity check re-evaluated each push).
- **BUT `baseline_cache::cycle()` drains only ONE mf from `m_miss_queue` to the DRAM port per tick**
  ([gpu-cache.cc:1213-1219](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1213-L1219)). So if both probes MISS, we admit 2 misses/cycle but only send 1 to DRAM/cycle → `m_miss_queue` fills, `access()` starts returning `RESERVATION_FAIL` (or the miss-queue-full guard blocks), and the stall relocates to the miss-queue.
- **Mitigation:** this is the exact Opt-7 §4.9-style paired-lever situation. However, note bwd `.o18` has
  ~87% L2 HIT rate (`L2_TMA_true_hit_rate=0.8688`) — **most admissions are HITs, which do NOT touch
  `m_miss_queue`** (a HIT pushes straight to the reply queue). So the 1/tick miss-queue drain is only a
  concern for the ~13% miss fraction, and DRAM is 97% idle anyway. Expect the relocation to be small,
  but **the miss-queue drain rate is the first thing to watch** in the run (see §7 gate). If it
  relocates, pair with a 2/tick `baseline_cache::cycle()` miss drain.

### 5.4 Port replenish stays once per tick (must NOT be doubled)
`baseline_cache::cycle()` calls `replenish_port_bandwidth()` once ([gpu-cache.cc:1223](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc#L1223)), decrementing `m_data_port_occupied_cycles` by 1/tick. If we widen the port to 64B (§6 option A) so 2 sectors = 1 occupancy-unit, the replenish stays correct. If instead we let 2 separate 32B probes each add 1 occupancy (total 2/tick) but only replenish 1/tick, the port would monotonically saturate — a bug. **So §6 option A (treat the L2 data port as 64B-wide) is the clean design; a naive "loop twice + keep 32B occupancy" is not.**

### 5.5 Reply/output side already widened by Opt 7
Each HIT admission pushes to `m_L2_icnt_queue` (reply FIFO). Opt 7 already set `reply_drain=4` +
`cluster_reply_eject=4`, so the reply side can absorb 2 HIT replies/cycle from this partition. The gate
`output_full = m_L2_icnt_queue->full()` is re-checked per probe (both probes must check it). No new risk.

### Trace verdict
**Admitting 2 sectors/cycle is safe** with these rules:
1. Loop the admission up to N=2 times, **re-checking `output_full` per probe** and stopping when the
   L2-input queue is empty or a probe RESERVATION_FAILs.
2. Model the L2 data port as **64B-wide** (occupancy semantics), NOT as two independent 32B charges
   against a 32B port — otherwise §5.4 saturation bug.
3. Keep `baseline_cache::cycle()` (miss-queue drain + port replenish) **once per tick**; only the
   admission loop repeats.
4. Watch the miss-queue relocation (§5.3); pair a 2/tick miss drain only if the run shows it.

No `assert` in `access()`/MSHR/tag-array is violated by a 2nd 32B probe in the same cycle.

## 6. Implementation options (Part 1 — admission rate)

### Option A (recommended) — config knob `-gpgpu_l2_admit_sectors_per_cycle N`, port modeled 64B-wide
- New `memory_config` member, default **1** (bit-identical). H100 config sets **2** (= 64B/cycle slice).
- In `cache_cycle` new-access block ([l2cache.cc:532-636](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L532-L636)) wrap the single admission in a bounded
  `for (p=0; p<N; ++p)` loop that: `break`s if `m_icnt_L2_queue->empty()`; re-reads `top()`; re-checks
  `output_full`; on RESERVATION_FAIL `break`s (head unchanged, retry next tick); pops on success.
- Port width: either bump the effective data-port occupancy accounting so N sectors = 1 occupancy unit
  (cleanest: gate the whole loop on a single `data_port_free()` at the top, and charge the port once for
  the batch), OR keep per-probe `use_data_port` but widen `m_data_port_width` to `N*32` **and** confirm
  `replenish` still matches. Option A's single-gate-at-top form avoids the §5.4 pitfall entirely.
- Everything else (`access()`, MSHR, tag array, reply push, miss push) is unchanged per-probe, so
  hit-rate/DRAM work is invariant by construction (§7 work axis).

### Option B (rejected for now) — full 128B line admission
Admitting a whole 128B line (4 sectors) per cycle overshoots HW (64B/cycle) by 2x and would be a
bandwidth cheat, not a calibration. Rejected on the same "no fake wins" principle as Opt 7.

### Option C (separate, larger) — model L2 slices/banks explicitly
Multiple parallel admission ports per sub-partition. More faithful but far more invasive; Option A
captures the throughput at the HW quantum, so C is deferred unless A relocates unresolvably.

*Files (Option A):* `gpu-sim.{cc,h}` (knob + `memory_config` member + boot log), `l2cache.cc`
(`cache_cycle` admission loop), `gpu-cache.{cc,h}` (port-width/occupancy alignment if needed),
`gpgpusim.config` (knob=2). No tracer/trace change; rebuild required.

## 7. Verification plan

Run bwd K10 first (worst ROP_DELAY), fwd K5 for the milder case.

**Success = `gpu_sim_cycle` ↓ WITH work invariant AND no unresolved relocation.**

- **Timing:** `gpu_sim_cycle` down; `IN_PARTITION_ROP_DELAY` avg ↓ (the target — it should drop toward
  the configured 100 + a small queue term); `wait_barrier` / `tma_flush` SM-idle ↓.
- **Work axis (MUST hold — else it's a bug, §4.12):** `L2_TMA_true_hit_rate` (bwd 0.8688 / fwd 0.9456),
  `L2_total_cache_accesses`, L2 read/write bytes, DRAM bytes all unchanged. A rate knob cannot change
  work.
- **Relocation detectors:** `gpu_stall_dramfull` (L2-input queue) should ↓; watch the **miss-queue**
  (§5.3) — if cycles stay flat and misses pile up, pair a 2/tick `baseline_cache::cycle()` miss drain.
  Check `bw_util` moves toward HW (bwd HW DRAM 14.85%) but does not exceed it.
- **Boot log:** `[L2-ADMIT] gpgpu_l2_admit_sectors_per_cycle = 2 ...` to confirm the knob is live in the
  first seconds.
- **Honest expectation:** since most admissions are HITs and DRAM is idle, this should give a real
  cycle cut (the ROP backlog is HIT-dominated serialization). But if the 87%-HIT reply traffic simply
  re-bunches at the next stage, cycles may move less than the ROP% suggests — the per-stage residency
  table localizes where it went.

### 7.1 HW (NCU) metrics to compare — which ones this Opt should move, and how (verified 2026-07-13)

Pulled from `nv_reports/h100/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24_full_rpt.ncu-rep` (bwd
main kernel = FlashAttnBwdSm90, the ID-9 instance). NCU commands run inside `docker exec gpu-sim bash`,
report at `/modern-gpu-simulator-micro-2025/modern-gpu-simulator-micro-2025/nv_reports/h100/...ncu-rep`,
`ncu` at `/usr/local/cuda/bin/ncu`.

**The single most relevant HW metric for this Opt is `L2 Cache Throughput %`** — it is NCU's measure of
how busy the L2 slices are, i.e. exactly the "slice utilization" this Opt raises. The sim currently
under-drives it (slice admits 1/2 the HW rate), so Opt 8 should move sim L2 utilization UP toward HW —
that is an *accuracy improvement*, not a cheat, as long as the work axis (sectors/hit-rate) is unchanged.

Both fwd (K5, FlashAttnFwdSm90) and bwd (K10, FlashAttnBwdSm90) will be run, so they are tabled
separately. "sim now" = the Opt-7 baseline (fwd `.o35` 138,021 / bwd `.o18` 250,026), i.e. the state
right BEFORE this Opt. NCU values are the main-kernel GPU-Speed-Of-Light + Memory-Workload sections.

> **Note on sim `Throughput_*_pct`.** The sim's own throughput% counters (Opt 4.13) are
> **cycle-contaminated** (bytes/cycle, and sim runs ~2x more cycles), so they read far LOWER than NCU
> and are NOT directly comparable until cycles converge (§4.12 pitfall 1). They are shown only as the
> sim's internal before-value; judge Opt 8 on `gpu_sim_cycle` + `ROP_DELAY` + the work axis, and use the
> NCU %s as directional anchors.

### FWD K5 (FlashAttnFwdSm90) — HW vs sim
| axis | HW (NCU) | sim now (`.o35`, Opt7) | sim/HW | Opt 8 expectation |
|---|---|---|---|---|
| **timing** — Elapsed Cycles | **67,696** | 138,021 | 2.04x | ↓ toward HW |
| **L2 busy %** (direct target) | **L2 Cache Throughput 22.68%** | `Throughput_L2_pct` 31.31 (cycle-contaminated) | n/a | slice util ↑; use §9 proxy |
| Memory Throughput % | 28.84% | — | — | ↑ |
| DRAM Throughput % | 12.09% | `Throughput_DRAM_pct` 0.65 / `bw_util` 0.017 | ~0.05x | ↑ toward HW, must NOT exceed |
| L1/TEX Cache Throughput % | 31.99% | `Throughput_L1TEX_pct` 1.83 | — | not this Opt's lever |
| Compute (SM) Throughput % | 43.04% | `Throughput_ComputeTensor_pct` 63.26 | — | side-effect only |
| **L2 sectors (work, invariant)** | 3,833,304 | `L2_total_cache_accesses` 3,457,208 | 0.90x | MUST stay put |
| **L2 Hit Rate (work, invariant)** | 69.58% | `L2_TMA_true_hit_rate` 0.9456 | over-modeled (CTA-cap, Ongoing item 2) | MUST stay put |

### BWD K10 (FlashAttnBwdSm90) — HW vs sim
| axis | HW (NCU) | sim now (`.o18`, Opt7) | sim/HW | Opt 8 expectation |
|---|---|---|---|---|
| **timing** — Elapsed Cycles | **132,901** | 250,026 | 1.88x | ↓ toward HW |
| **L2 busy %** (direct target) | **L2 Cache Throughput 48.52%** | `Throughput_L2_pct` 56.34 (cycle-contaminated) | n/a | slice util ↑; use §9 proxy |
| Memory Throughput % | 56.58% | — | — | ↑ |
| DRAM Throughput % | 14.85% | `Throughput_DRAM_pct` 1.82 / `bw_util` 0.045 | ~0.12x | ↑ toward HW, must NOT exceed |
| L1/TEX Cache Throughput % | 62.59% (HW's hottest pipe) | `Throughput_L1TEX_pct` 6.74 | — | not this Opt's lever |
| Compute (SM) Throughput % | 48.45% | `Throughput_ComputeTensor_pct` 73.85 | — | side-effect only |
| **L2 sectors (work, invariant)** | 10,111,818 | `L2_total_cache_accesses` 11,269,403 | 1.11x | MUST stay put |
| **L2 Hit Rate (work, invariant)** | 82.26% | `L2_TMA_true_hit_rate` 0.8688 | on target | MUST stay put |

**Reading both:** bwd is the cleaner proving ground — its L2 hit rate already matches HW (0.87 vs 0.82)
and HW L2 is already quite busy (48.5%), so widening admission should move sim L2 utilization toward HW
with less confound. fwd's L2 hit rate is still over-modeled (0.95 vs 0.70, the CTA-count cap = Ongoing
item 2), so its DRAM-work gap (0.05x) is a hit-rate story, not an admission story; expect a smaller,
noisier Part-1 effect on fwd.

### 7.2 Per-slice (sub-partition) utilization + bank-conflict: what HW exposes (verified 2026-07-13)

Checked whether NCU has a per-L2-slice utilization metric and an L2 bank-conflict metric — because
those are the direct HW counterparts of §9's per-slice admission histogram and of the "bank" abstraction.

**(a) L2 bank conflict — NCU has NONE for L2.** The only bank-conflict metrics are
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared*` — all **shared-memory** only (bwd: total 45,471 =
ld 24 + st 35,493). There is **no `lts__*bank_conflict` metric**: L2-slice-internal banks are HW-hashed
and not user-controllable, so NVIDIA does not expose L2 bank conflicts (consistent with §3: the sim
also does not model L2 banks). The shared-mem bank conflict is a TODO-1 (SMEM swizzle) axis, orthogonal
to this Opt.

**(b) Per-slice L2 utilization — NCU HAS it: `lts__cycles_active.{avg,max,min}`** (`lts` = L2 slice), and
this is the direct HW anchor for §9's `L2_slice_util_*`. Measured (bwd K10 main kernel):

| NCU per-slice metric | avg | max | min | max/min spread |
|---|---|---|---|---|
| `lts__cycles_active` (slice busy cycles) | 143,159 | 145,071 | 141,035 | **2.9%** |
| `lts__t_sectors` (sectors per slice) | 126,398 | 129,937 | 123,519 | **5.2%** |

**Decisive reading — HW L2 slices are almost perfectly EVEN (≤5% spread across slices).** So on HW the
L2 is NOT "a few hot slices + the rest idle"; the traffic is fully spread and every slice is ~equally,
near-fully busy. Contrast with the sim baseline `partiton_level_parallism = 44 / 80` (~55% of slices
active per cycle, §9.1) — the sim is LESS spread / lower per-slice occupancy than HW.

**This is exactly the fork §9.3 was built to resolve, now with the HW answer in hand:**
- If, after Part 1, sim `L2_slice_util_p50/p95/max` become **even and high like HW** (`lts__cycles_active`
  avg≈max), the per-slice-throughput model was the gap → Part 1 is HW-faithful and the right lever.
- If sim `L2_slice_util` stays **skewed (hot p95/max ≫ p50)** while HW is even, then part of the wall is
  a **spread / address→slice mapping** problem — which Part 2 (§10/§11) fixes, NOT throughput — Part 1
  alone won't fully close it. `lts__cycles_active.{avg,max,min}` is the HW yardstick for that judgment.

**Add these to the NCU pull command** (`--page raw --metrics ...`):
`lts__cycles_active.avg,lts__cycles_active.max,lts__cycles_active.min,lts__t_sectors.avg,lts__t_sectors.max,lts__t_sectors.min`
(fwd K5 equivalents to be read the same way when judging fwd).

**How to read it after the run (two independent axes, do not conflate — §4.12):**
1. **Work axis first (cycle-independent):** confirm sim `L2_total_cache_accesses` and
   `L2_TMA_true_hit_rate` are UNCHANGED vs the Opt-7 baseline above (fwd 3,457,208 / 0.9456; bwd
   11,269,403 / 0.8688). If they move, the knob has a bug (a pure rate change cannot alter work).
2. **Timing axis:** `gpu_sim_cycle` should drop toward HW (fwd 67,696 / bwd 132,901);
   `IN_PARTITION_ROP_DELAY` avg should fall.
3. **Throughput% is a CROSS-CHECK, read only after cycles converge** (bytes/cycle is cycle-contaminated,
   §4.12 pitfall). Once cycles are closer, sim `DRAM%`/`bw_util` should approach HW `DRAM Throughput`
   (fwd 12.09% / bwd 14.85%) — WITHOUT overshooting (overshoot = fake bandwidth).

**Caveat — no sim metric is byte-for-byte identical to NCU `L2 Cache Throughput %`.** NCU's is a
pct-of-peak-sustained; the sim has no such counter. So the primary sim gate stays `gpu_sim_cycle` +
`ROP_DELAY` + the §9 per-slice admission histogram; the NCU `L2 Cache Throughput` / `DRAM Throughput` /
`Memory Throughput` are directional HW anchors (sim should move toward them, and the fact that HW L2 is
48.5%/22.7% busy while the model serializes admission is the independent evidence that widening it is
HW-faithful).

**Reusable NCU command (per-kernel raw metrics):**
```
ncu --import <report>.ncu-rep --csv --page raw \
    --metrics gpc__cycles_elapsed.max,lts__t_sectors.sum,lts__t_sector_hit_rate.pct,\
dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed
# GPU Speed-Of-Light %s via --page details, grep each main-kernel section:
#   FWD K5:  Elapsed 67,696  / L2 22.68% / DRAM 12.09% / Memory 28.84% / L1TEX 31.99% /
#            Compute 43.04% / L2 Hit 69.58% / L2 sectors 3,833,304
#   BWD K10: Elapsed 132,901 / L2 48.52% / DRAM 14.85% / Memory 56.58% / L1TEX 62.59% /
#            Compute 48.45% / L2 Hit 82.26% / L2 sectors 10,111,818
```

## 8. Relationship to the other remaining item (fixed-overhead alternative)

This Opt (L2 admission width) attacks the **throughput/serialization** half of ROP_DELAY. The other
FA3_progress "Ongoing item 1" framing — modeling a per-transfer ~170-cyc TMA fixed overhead instead of
768 serialized sector round-trips — attacks the same wall from the **fixed-latency** side and is more
invasive (needs a per-transfer completion model, and risks deleting the per-sector L2/DRAM traffic that
earns the realistic hit rate). **Do this Opt (Option A) first** — it is the direct, HW-anchored,
work-invariant rate calibration and is the clean continuation of Opt 7. Escalate to the fixed-overhead
model only if admission-width relocates the stall without closing the cycle gap.

## 9. Instrumentation — measure per-sub-partition admission parallelism BEFORE/AFTER (add to next run)

The core hypothesis (§1) is that the ROP wall is a **burst concentrated on a few sub-partitions**, not
a chip-wide bandwidth limit. That must be *measured*, not assumed. Two levels of instrumentation:

### 9.1 Already-present coarse counter (re-confirm; no build needed)
`partiton_level_parallism` is **already printed** ([gpu-sim.cc:3475-3492](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L3475-L3492)) and shows the
chip-wide average. Measured on the Opt-7 baseline (bwd K10 `.o18`):
- `partiton_level_parallism = 44.04` — avg **~44 of 80** sub-partitions receive a request per cycle.
- `partiton_level_parallism_util = 48.21` — even counting only non-idle cycles, only ~48 of 80.

**Caveat — this counter is NOT L2 admission.** It increments at `icnt_pop`→`push` into the
sub-partition ([gpu-sim.cc:4320](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L4320)), i.e. it counts **inject-side** arrivals, not the
`cache_cycle` admission we widen. It also averages over the whole kernel, so it hides the *temporal*
concentration (a hot sub-partition serializing its backlog while others idle). Useful as a sanity
number, insufficient as the gate.

### 9.2 New counter to ADD (build required) — true per-sub-partition admission distribution
Add a **per-sub-partition L2-admission histogram** so the next run directly shows the burst
concentration and how much headroom the 1→2 widening actually uses.

- **Where:** in `memory_sub_partition::cache_cycle` new-access block ([l2cache.cc:552-635](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.cc#L552-L635)),
  count an **actual admission** (a probe that did NOT `RESERVATION_FAIL`, i.e. an mf that was
  `pop()`ed this tick) per sub-partition.
- **Counters (per sub-partition `i`):**
  - `m_l2_admissions[i]` — total admissions (should equal that slice's share of `L2_total_cache_accesses`).
  - `m_l2_active_cycles[i]` — cycles this slice admitted ≥1 (for per-slice utilization).
  - After Part 1 lands: `m_l2_admit2_cycles[i]` — cycles this slice admitted **2** (proves the widened
    budget was actually used, mirror of Opt 7's `*_multi_ticks`).
- **Dump (in `gpu_print_stat`):**
  - `L2_admit_per_active_cycle` = Σ`m_l2_admissions` / Σ`m_l2_active_cycles` — the real admission rate
    on busy slices (baseline expectation ≈ 1.0, capped by the 1/cycle limit; a value pinned at ~1.0
    while `gpu_stall_dramfull` is high is the smoking gun that the 1/cycle cap is binding).
  - `L2_admit_p50 / p95 / max` across the 80 slices (histogram tail = the burst concentration).
  - `L2_slice_util_p50/p95` = per-slice active-cycle fraction (baseline: the hot slices near 100%
    while median is far lower ⇒ concentration confirmed).
- **Timing-neutral:** observe-only counters, incremented on the existing admission path, no scheduling
  effect. Race-free (the sub-partition loop is serial).

### 9.3 What the numbers decide
- **Baseline run (Opt 7 binary + these counters):** if the hot slices show `admit_per_active ≈ 1.0`
  pinned AND `admit_p95/max` ≫ median utilization, the burst-serialization hypothesis is confirmed and
  Part 1 (1→2) is the right lever.
- **After Part 1:** `m_l2_admit2_cycles` > 0 on the hot slices proves the widened budget was used;
  `admit_per_active` should rise toward ~2 on those slices, `gpu_stall_dramfull` and `ROP_DELAY` should
  fall, with the §7 work axis unchanged.
- If the hot slices already sit at `admit_per_active ≈ 1.0` but the p95 across slices is *low* (traffic
  actually spread), then Part 1 helps little and the real issue is the **address→slice mapping** (see
  §10/§11) — i.e. the burst is not being spread across slices in the first place.

**Files:** `l2cache.{h,cc}` (per-sub-partition counters + accessors), `gpu-sim.cc` (aggregate + print).
No tracer/trace change; rebuild required.

## 10. Part 2 — CONFIRMED spatial root cause: the `% 80` partition-index modulo bias (static, 2026-07-13)

> **User's hunch (correct):** "changing the L2 slice return to 64B/cycle won't fix the sub-partition
> imbalance — it only changes how much a slice returns, not WHERE the sectors go inside L2." Confirmed
> by static analysis: Part 1 (throughput) and the imbalance (placement) are **orthogonal**, and there is
> a **structural placement bias in the address→slice mapping** that Part 1 cannot touch. Part 2 fixes it.

### The mechanism (code-proven, not a guess)
The H100 config has `-gpgpu_n_mem 40` × `n_sub_partition_per_mchannel 2` = **80 sub-partitions**, but 40
is **not** a power of two → `gap = true`. The IPoly path in
[addrdec.cc:144-161](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/addrdec.cc#L144-L161) does:
```
sub_partition = ipoly_hash_function(high_bits, sub_partition, nextPow2(40)*2 = 128);  // 0..127
if (gap) sub_partition = sub_partition % (40*2 = 80);                                  // 0..79
```
`ipoly_hash_function` only supports power-of-two `bank_set_num` (16/32/64/128/…), so it is called with
**128** and then folded to 80 by `% 80`. Boot log confirms the live config: `IPoly`, `sub partition = 80`,
`gap` path taken.

### The bias, quantified (assuming the hash is perfectly uniform over 0..127)
```
128 hash values % 80:
  sub-partitions  0..47  (48 slices) ← hit by TWO hash values (N and N+80) → ~2x traffic  (HOT)
  sub-partitions 48..79  (32 slices) ← hit by ONE hash value               → ~1x traffic  (COLD)
=> up to 2:1 spatial load imbalance from the modulo ALONE, independent of the address pattern.
```
This is a **pure simulator artifact of the 40-channel (non-2^n) config** — real H100 partition counts
are chosen hash-friendly, which is exactly why NCU shows the HW L2 slices even to ≤5% (§7.2) while the
sim baseline runs at `partiton_level_parallism = 44/80 (~55%)`: the ~32 cold slices are structurally
under-fed.

### Why Part 1 (admission rate) cannot fix this (the user's point, formalized)
Part 1 raises each slice's **service rate** (1→2 sector/cycle). It does **not** change **which** slice a
sector maps to. If slices 0..47 receive ~2x the sectors, making every slice 2x faster leaves the *ratio*
unchanged — the hot half still finishes its double load in the same relative time as the cold half
finishes its single load, so the per-slice utilization histogram stays skewed. Part 1 lowers the absolute
ROP wait (fewer serialized cycles per slice) but does not flatten the slice distribution.

### The TMA address does NOT rescue it either
A TMA transfer's sectors are `global_base + tile_offset + agu_index*128 + sector*32`
([tma_unit_sm.cc:815-860](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L815-L860)) — a run of consecutive 128B lines. IPoly spreads consecutive
lines across the 128-wide hash space well, but the **final `% 80` fold re-concentrates them 2:1 onto the
low 48 slices** regardless of how good the hash is. So the placement bias is downstream of both the hash
and the TMA address; neither the address model nor Part 1 removes it.

### Fix direction (Part 2 — address→slice mapping)
Remove the `% 80` fold-bias so all 80 slices are equally likely. Options, cheapest first:
1. **Config probe (no code):** does the sim accept an 8th/16th memory config that is a power of two, or a
   different `-gpgpu_memory_partition_indexing`? If `n_mem` could be 32 or 64 the `gap` path disappears
   and IPoly is bijective. But changing `n_mem` changes L2 size/DRAM channels → NOT work-invariant, so
   this is a modeling change, not a free knob. **Rejected** (changes model params / HW spec).
2. **Bias-free hash (code, CHOSEN + IMPLEMENTED):** see §11.
3. **Validate against §7.2 HW anchor:** after the fix, sim `L2_slice_util p50≈p95≈max` (even, like HW's
   `lts__cycles_active` avg≈max) and `partiton_level_parallism` should rise from ~44 toward ~80.

### Decision order (do NOT skip the measurement)
The §9.2 histogram from the *current* run is still the arbiter of **how much** of the ROP wall is
placement (this §10 bias) vs throughput (Part 1's target):
- If cycles drop a lot and `L2_slice_util` flattens → throughput dominated, bias is secondary.
- If cycles barely move and `L2_slice_util` stays skewed with hot∈{0..47} → **this %80 bias is the
  real wall**, and Part 2 (bias-free hash) is the fix. The static proof above says the bias is *present*;
  the run says how *binding* it is. **Part 1 and Part 2 are complementary, not alternatives.**

## 11. Part 2 fix — balanced avalanche sub-partition hash (IMPLEMENTED, 2026-07-13)

Per the decision to run Part 1 (admission rate) + Part 2 (this placement fix) together (one 12h run),
the `% 80` bias fix is implemented.

### Why not the "obvious" fixes (both empirically rejected)
- **Any deterministic 128→80 fold is 2:1** (pigeonhole: 128 hash values into 80 slices ⇒ 48 slices get
  2, 32 get 1). `h % 80` and `h*80>>7` both measured min/max = 1/2. So keeping the IPoly-128 space and
  folding cannot be uniform.
- **Plain multiplicative hash `(x*C) % 80` collapses on strided traffic.** Measured: consecutive
  addresses uniform (cv 0.005) but **stride-128 puts everything on ~5 slices** (`gcd(128,80)=16`,
  cv 3.87). GPU traffic is heavily strided, which is the whole reason IPoly exists — so a naive
  multiplicative hash is worse than the bias.

### The chosen hash — avalanche (bit-mix) then modulo
`balanced_subpartition_hash(high_bits, ipoly_index, n_slices)`
([hashing.cc](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/hashing.cc), [hashing.h](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/hashing.h)): combine the address high bits with the IPoly index
(`x = high_bits ^ (index<<40)`), run a **SplitMix64 finalizer** (two multiply+xorshift rounds =
full avalanche), then `% n_slices`. The avalanche decorrelates any power-of-two stride from the
modulus, so `% 80` becomes uniform **for every 2^k stride**. Measured (python model, 400k refs):

| stream | `% 80` (current) | balanced hash |
|---|---|---|
| consecutive | cv 0.005 (2:1) | cv 0.003 |
| stride-128 | 2:1 bias | **cv 0.001** |
| stride-1024 | 2:1 bias | **cv 0.001** |
| stride-4096 | 2:1 bias | **cv 0.001** |

Properties that make it safe:
- **Deterministic + address-stable:** pure function of `(high_bits, index, n_slices)` → same line always
  maps to the same slice (L2 caching / MSHR / reuse unaffected).
- **Work-invariant:** it only changes **which** slice `tlx->sub_partition` is; it does NOT touch
  `partition_address()` (which strips the chip/sub bits to form the in-partition address — verified
  independent of the slice choice), so L2 hit rate / access counts / DRAM bytes are unchanged.
- **Only on the biased path:** applied ONLY when `gap && memory_partition_indexing==IPOLY`; power-of-two
  configs (no gap) are untouched. Default off.

### Change list
- `hashing.{h,cc}` — `balanced_subpartition_hash()`.
- `addrdec.{h,cc}` — member `l2_slice_balanced_hash` (ctor-init 0), option
  `-gpgpu_l2_slice_balanced_hash`, IPoly-gap branch calls the balanced hash instead of `ipoly%N`, and a
  `[L2-SLICE-HASH]` boot log (once) confirming it is live and that the config actually hits the `gap` path.
- `gpgpusim.config` — `-gpgpu_l2_slice_balanced_hash 1` (default 0 = original ipoly%N, bit-identical).

### Validation gate (same run as Part 1)
- **Boot log:** `[L2-SLICE-HASH] balanced sub-partition hash ENABLED (n_channel=40 non-2^n, n_slices=80 …)`.
- **Imbalance fixed (the target):** `L2_slice_util_p50 ≈ p95 ≈ max` (was skewed), `partiton_level_parallism`
  ↑ from ~44 toward ~80, `L2_slice_admissions_p50 ≈ max`. Compare to HW `lts__cycles_active` avg≈max (§7.2).
- **Work-invariant (MUST hold — else a bug):** `L2_TMA_true_hit_rate` (bwd 0.8688 / fwd 0.9456),
  `L2_total_cache_accesses`, L2 bytes, DRAM bytes all unchanged. A placement change cannot alter work.
  ⚠ **Caveat:** re-hashing which line → which slice *can* subtly change L2 set-mapping and thus hit rate
  by a tiny amount if slice==set-selecting bits overlap; watch hit rate closely — a >1% move means the
  hash is perturbing set selection, not just slice selection, and must be revisited.
- **Timing:** `gpu_sim_cycle` ↓ (together with Part 1); this is the combined injection-throughput +
  placement-balance run.

## 12. Relationship to TMA swizzle (FA3_progress TODO-1) — related GOAL, different LAYER

The user's intuition — "swizzle spreads the transfer across all L2 sub-partitions to approach peak
bandwidth" — points at a real and correct HW effect, but it is important **not** to attach it to the
wrong mechanism. There are TWO distinct "swizzle/spread" concepts at TWO different memory layers, and
FA3_progress TODO-1 is only the first one:

| | layer | what it spreads | goal | sim status |
|---|---|---|---|---|
| **TODO-1: TMA descriptor swizzle** | **shared memory (SMEM)** | the tile's bytes across **SMEM banks** (32 banks) | avoid SMEM bank conflicts on the GMEM→SMEM write and the downstream `LDSM`/`LDS` reads | descriptor field carried, **not applied** (TODO) |
| **L2 partition/slice spread** | **L2 (GMEM side)** | GMEM lines across **L2 slices / memory channels** | approach peak DRAM/L2 bandwidth by using all slices | **already implemented** — `addrdec.cc` + `hashing.cc`, IPoly (`-gpgpu_memory_partition_indexing 2`) |

**Key clarification:** the "spread across all L2 sub-partitions to approach peak bandwidth" effect is
governed by the **L2 partition indexing (address hashing)**, which the sim **already does**. It is NOT
what TODO-1 (SMEM swizzle) controls. SMEM swizzle only rearranges bytes *within one CTA's shared
memory* to dodge the 32-bank SMEM conflict; it does not change which L2 slice a GMEM line lands on.

**So how does this interact with Part 1 (the L2-admission wall)?** Two separate questions, and §9.2's
histogram answers the crucial one:
1. **Are TMA transfers actually spread across L2 slices, or concentrated?** If the M2/M2.5 CTA-indexed
   tiling + IPoly hashing already spreads them well (median slice utilization ≈ p95), then the ROP wall
   is a genuine **per-slice throughput** limit → Part 1 (1→2 admission) is the fix, and it moves sim
   toward HW peak-bandwidth behavior *by making each slice as fast as HW*, not by re-spreading.
2. **If instead the histogram shows a few hot slices (concentration)**, then part of the wall is a
   *spread* problem, and the lever is the **address→slice mapping** for TMA tiles — which is exactly
   **Part 2** (§10/§11), **not** SMEM swizzle (TODO-1) and not the admission width. Part 1 and Part 2
   are complementary.

**Bottom line for the docs:** TODO-1 (SMEM swizzle) is a **shared-memory bank-conflict** fidelity item,
orthogonal to the L2-admission cycle wall. The "spread across L2 to hit peak BW" idea belongs to the
**L2 partition indexing** (Part 2 fixes its `% 80` bias); whether TMA bursts exploit it is exactly what
§9.2's per-slice admission histogram will reveal.

## 13. Implementation status + risk review (IMPLEMENTED, pre-build) — 2026-07-13

Part 1 (admission) + Part 2 (balanced hash) + §9 instrumentation are coded (build pending on the user).
Files changed:
- `gpu-sim.h` — `memory_config::gpgpu_l2_admit_sectors_per_cycle` (default 1).
- `gpu-sim.cc` — knob registration (`-gpgpu_l2_admit_sectors_per_cycle`, default "1"); `[L2-ADMIT]`
  boot log (once, when >1); §9 per-slice admission histogram print in `gpu_print_stat`.
- `l2cache.{h,cc}` — the N-probe admission loop in `cache_cycle` + per-sub-partition counters
  (`m_l2_admissions/active_cycles/multi_admit_cycles`) + accessors.
- `gpu-cache.h` — `bandwidth_management::replenish_data_port_extra(reps)` + `baseline_cache`
  pass-through (the N-wide data-port model).
- `hashing.{h,cc}` — `balanced_subpartition_hash()` (Part 2).
- `addrdec.{h,cc}` — `l2_slice_balanced_hash` member + `-gpgpu_l2_slice_balanced_hash` option +
  IPoly-gap branch + `[L2-SLICE-HASH]` boot log (Part 2).
- `gpgpusim.config` — `-gpgpu_l2_admit_sectors_per_cycle 2` (Part 1) + `-gpgpu_l2_slice_balanced_hash 1`
  (Part 2).

### The one subtle correctness point (got it right, documented so it is not "fixed" back)
The data port must be gated **only on the first probe** (`port_ok = (ap==0) ? port_free : true`) AND the
loop must be followed by `(N-1)` extra data-port replenishes. These two are a **matched pair**:
- `access()` charges the port once per probe. If we re-checked `port_free` on ap>0, the 1st probe's
  occupancy would block the 2nd within the same tick → N=2 would silently admit only 1 (no effect).
- If we skip the ap>0 port check but do NOT add the extra replenish, occupancy accumulates N/tick while
  base replenish removes 1/tick → the port saturates and throttles the NEXT tick back to 1 (also wrong).
- With both: steady state = 1 residual occupancy at tick start → base replenish → 0 → admit up to N →
  occupancy N → extra (N-1) → 1 residual. Port is free at every tick start ⇒ N admissions/tick every
  tick. `replenish_data_port_extra` has an `if(occupied>0)` guard so it never underflows when fewer than
  N were admitted. **An earlier draft re-checked the port every probe; that was a bug and was reverted.**

### Risks considered (and why each is OK)
1. **Work-axis drift (would be a bug).** Each probe still runs the real `access()`+MSHR+tag-array, so
   hit/miss/sector counts are identical to N=1; only *how many per tick* changes. Gate: `L2_TMA_true_hit_rate`
   / `L2_total_cache_accesses` must be unchanged (§7.1 axis-1). If they move → real bug, reject.
2. **Miss-queue relocation (real, expected small).** The loop can admit up to N misses/tick, but
   `baseline_cache::cycle()` still drains only 1 miss/tick to DRAM. With ~87% L2 hits and DRAM 97% idle
   this should be minor; if `gpu_stall_dramfull` explodes and cycles stay flat, pair a 2/tick miss drain
   (§5.3). This is the honest Opt-7 §4.9-style relocation risk, and the §9 counters + stage residency
   localize it without a second exploratory run.
3. **RESERVATION_FAIL head-of-line.** On a RESERVATION_FAIL the loop `break`s (head unchanged, retried
   next tick) — identical to N=1 semantics, no busy-spin.
4. **L2-disabled / texture path.** Counted as an admission (it pops) but the extra replenish is guarded
   to `!disabled`, and that path never touches the data port, so no port bookkeeping issue.
5. **Default-off safety.** Knob default 1 reproduces the original single-probe flow exactly (loop runs
   once, `port_ok=port_free`, no extra replenish). Bit-identical when not enabled.

### Debug-log coverage for the 12h run (sufficient?)
- **Boot (stderr, once):** `[L2-ADMIT] gpgpu_l2_admit_sectors_per_cycle = 2 ...` — confirms the knob is
  live in the first seconds (mirrors `[ICNT->L2]` / `[REPLY-EJECT]`).
- **End-of-kernel (stdout), lever-fired proof:** `L2_admit_multi_cycles_total` > 0 and
  `L2_admit_per_active_cycle` > 1.0 prove the 2nd probe actually admitted (mirror of Opt 7's
  `*_multi_ticks`). If both are ~0/1.0 the widened budget was never used (valid null, not a failed run).
- **End-of-kernel, relocation localizers (already present):** `gpu_stall_dramfull`,
  `L2_TMA_output_full_cycles`, the TMA per-stage residency table (`IN_PARTITION_ROP_DELAY` avg), `bw_util`
  — together they show whether the stall moved and where.
- **Slice-concentration (new §9):** `L2_slice_util_p50/p95/max`, `L2_slice_admissions_p50/p95/max` — the
  burst-vs-spread evidence.
- **Assessment: coverage is sufficient** for a single decisive run. The only thing NOT logged is a
  per-cycle time series (deliberately — it would flood a 12h run); the aggregate histogram is enough to
  decide. No early-stop assert was added because the loop reuses existing gated ops (no new invariant that
  could silently corrupt); the `replenish_data_port_extra` guard prevents port underflow.
