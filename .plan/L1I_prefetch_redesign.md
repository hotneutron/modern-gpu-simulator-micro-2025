# L1I Prefetch Redesign: Demand-Independent Eager Promote

## 1. Motivation

After the MEMBAR scope-aware fix (Opt 3), the FA3 forward kernel (kernel 5) is no
longer barrier-bound. The dominant remaining bottleneck is the instruction
frontend. In the `o15` run (`gpu_tot_sim_cycle = 158,990`):

- `no_valid_instruction = 37.01%` is the #1 issue-stage stall.
  - Almost all of it is `head_invalid_waiting_frontend = 36.57%`.
  - Of that, `..._in_l0i_response_queue_stream_buffer_wait = 17.67%`.
  - Inside that, **`prefetch_issued_not_ready = 11,710,629 cycles`** dominates
    (vs `prefetch_ready_not_promoted = 92,444`).

So the fetch unit is waiting on the *exact* line a prefetch was already issued
for, but that prefetch is not ready in time.

### Why the prefetch is "always late" (quantified from `o15`)

| Counter | Value | Meaning |
|---|---|---|
| `prefetch_issue_to_first_demand` avg | 28,741,108 / 121,841 = **236 cyc** | Prefetch is issued ~236 cycles *ahead* of demand. Lead time is plenty. |
| `head_demand_arrived_before_ready` | 66,850 (55% of waiters) | More than half the time the prefetch is still not ready when demand arrives. |
| `demand_wait_for_ready` avg | 17,938,201 / 66,850 = **268 cyc** | When late, demand waits ~268 extra cycles. |
| **`head_demand_arrived_after_ready`** | **0** | **The prefetch has NEVER become ready before demand. Not once.** |

The lead time is large, yet readiness *never* beats demand. This is not a
"prefetch too late" timing problem; it is a structural one.

## 2. Root Cause

### Where the prefetched data actually lives

The instruction path is `L0I -> L1I -> (L2/mem)`. The stream buffer is a staging
structure that sits *beside* L1I, not inside it.

1. **Issue** (`single_stream_buffer::do_prefetch`): a prefetch `mem_fetch` is
   pushed via the L1I `m_memport` toward L2/mem. **Nothing is written into the
   L1I tag array.** Only a queue entry is recorded.
2. **Response** (`first_level_instruction_cache::fill` ->
   `single_stream_buffer::fill`): when data returns,
   `if (mf->get_is_prefetch()) return m_stream_buffers->fill(...)`. The line is
   stored **in the stream buffer entry** and marked `is_ready = true`. **It is
   NOT placed in the L1I tag array.**
3. **Promote** (`send_to_cache` -> `fill_from_stream_buffer`): the line is only
   written into the L1I tag array (`m_tag_array->fill(...)`) **after a demand
   arrives** and sets `is_request_to_cache = true`. The same call **also**
   generates the L0I response (`m_next_response`).

Conclusion: the prefetched data is in the **stream buffer**, not in L1I and not
in L0I. So "is the data already in L1I?" — no, only in the staging buffer.

### Why prefetch goes through the demand response slot

`fill_from_stream_buffer` fuses two actions into one:
(a) fill the L1I tag array, and (b) create the L0I response `m_next_response`.
And the promote is gated on demand arrival. Worse, the per-cycle scheduler gates
promote on the response slot being free:

```cpp
// first_level_instruction_cache::cycle()
m_stream_buffers->cycle(!m_next_response);  // promote only when response slot empty
```

`m_next_response` is the single L0I reply slot (`max_reply_allowed_from_L1I 1`).
When demands stream in, this slot is almost always occupied, so prefetch lines
never get a chance to promote ahead of demand. This is the direct cause of
`head_demand_arrived_after_ready = 0` and of
`ready_head_blocked_by_response_slot = 273,680`.

### Ports are NOT the bottleneck

| Port | util |
|---|---|
| `L1I_cache_data_port_util` | 0.000 |
| `L1I_cache_fill_port_util` | 0.002 |
| `L0I_cache_data_port_util` | 0.000 |
| `issue_port_busy` | 0.71% |

Bandwidth is idle. Also note the existing promote path calls
`get_bw_manager().use_fill_port(nullptr)` which only *accumulates* occupancy; it
does **not** check `fill_port_free()` and never blocks on the port today. The
blocker is purely the logical `m_next_response` dependency, not physical ports.

### Difference vs real hardware

- **HW**: the I-prefetcher fills the line straight into L1I (or L0I). A later
  demand fetch is simply a cache hit. Prefetch-fill and demand-response are
  independent events.
- **This simulator**: the prefetched line stays in the stream buffer and is only
  promoted into L1I *when a demand arrives*, fused with the demand response. The
  "warm the cache ahead of time" effect therefore structurally cannot happen.

## 3. Goal

When a prefetch becomes ready (`is_ready = true`), promote it into the L1I tag
array **immediately, without waiting for a demand**, and **without** producing an
L0I response. A subsequent demand then hits in L1I via the normal
`read_only_cache::access` HIT path.

### Success criteria (vs `o15`)

- `head_demand_arrived_after_ready`: **0 -> large positive** (prefetch starts
  beating demand).
- `prefetch_issued_not_ready` stall (11.7M cyc): large reduction.
- `ready_head_blocked_by_response_slot` (273,680): reduction.
- `gpu_tot_sim_cycle` (158,990): reduction.
- Correctness: no deadlock, `fill_orphaned = 0` preserved, clean kernel exit.

## 4. Design: Two coexisting promote modes

Redefine the stream buffer as an L1I prefetch staging area. Split "promote into
L1I" from "produce L0I response".

| Mode | Trigger | Action |
|---|---|---|
| **(new) eager promote** | prefetch fill complete (`is_ready == true`, `is_request_to_cache == false`) | fill the L1I tag array + pop from SB. **Does NOT create `m_next_response`.** Gated by `fill_port_free()` (option B). |
| (existing) demand path | demand arrives + SB hit | unchanged; lines already eager-promoted are naturally served as L1I HIT. |

### Port handling — Option B (chosen)

eager promote uses the same fill port. We model the 1-port constraint strictly
and give demand fill priority:

```cpp
// pseudocode in the per-cycle promote logic
if (eager_promote_enabled && head.is_ready && !head.is_request_to_cache) {
    if (cache.fill_port_free()) {            // only when the port is free
        cache.promote_prefetch_to_cache(head_addr); // tag_array fill, NO m_next_response
        cache.use_fill_port();
        pop(head);
    }
    // else: skip this cycle, keep the entry in the SB, retry next cycle
}
```

Rationale:
1. Accuracy: fill port is a single port; demand fill and eager promote must not
   both occupy it in the same cycle.
2. Safety / no deadlock: eager promote is best-effort. If the port is busy it is
   simply deferred to a later cycle; the data stays safely in the SB. The demand
   path is unaffected.
3. Practically rarely blocked: demand-side `fill_port_util` is 0.2%, so eager
   promote will succeed almost every cycle.

## 5. Change points (by file)

### A. `stream_buffer.cc` — `single_stream_buffer::fill`
When a prefetch response lands and `is_ready` becomes true, mark the entry as
eligible for eager promote. Do **not** fill the tag array in the same cycle
(avoid re-entrancy / port-ordering issues); let `cycle()` do it.

### B. `stream_buffer.cc` — `multiple_stream_buffers::cycle` / `single_stream_buffer`
Each cycle, for a head entry with `is_ready && !is_request_to_cache`, perform the
Option-B eager promote: if `fill_port_free()`, call the new cache promote method,
consume the fill port, and pop the entry. This promote must **not** touch
`m_next_response`.

### C. `first_level_instruction_cache` — new method `promote_prefetch_to_cache(addr)`
- Probe first; if already HIT or MSHR-pending, skip (reuse the
  `do_prefetch` probe pattern) and count it.
- Otherwise `m_tag_array->fill(mshr_addr, time, ...)` only. **No
  `m_next_response`, no `m_regular_access_status_*` mutation** (no demand yet).
- Build a lightweight synthetic `mem_fetch` for the tag-array fill (reuse the
  `mf_new` pattern from `fill_from_stream_buffer`).

### D. `first_level_instruction_cache::access` — eager-promoted lines hit naturally
Minimal change. Once a line is eager-promoted into L1I,
`read_only_cache::access` returns HIT and the existing HIT branch creates
`m_next_response` (1-cycle response). `sb_check.is_hit_requested_addr` now only
catches the case where demand arrived before promote; existing logic stands.

### E. `gpgpusim.config` — safety switch
Add `-is_instruction_prefetch_eager_promote_enabled` (default **0** = legacy
behavior). Enable only for the experiment to isolate regression risk and allow
A/B comparison.

## 6. Correctness — failure modes and defenses

### 6.0 Port-metric reflection (verified)

The per-cycle order inside `first_level_instruction_cache::cycle()` is:

```
1) read_only_cache::cycle() == baseline_cache::cycle()
      -> sample_cache_port_utility(...)   // metric sampled from CURRENT occupancy
      -> replenish_port_bandwidth()       // occupancy -= 1
2) demand promote path (m_next_response handling)
3) m_stream_buffers->cycle(...)           // eager promote calls use_fill_port()
```

- eager promote's `use_fill_port()` adds `fill_cycles = atom_sz / data_port_width`
  to `m_fill_port_occupied_cycles`. It is observed by `sample_cache_port_utility`
  on the **next** cycle, so it **does** show up in `L1I_cache_fill_port_util`
  (currently 0.2%). Metric impact confirmed.
- For il1 (`N`, non-sector) `atom_sz = line_sz = 128B`; `data_port_width` defaults
  to `line_sz = 128B` => `fill_cycles = 1`. So one promote occupies the fill port
  for exactly 1 cycle -> at most one eager promote per cycle, no throttling of the
  effect.
- Only the **fill** port is touched (correct: promote is a fill). The L1I
  **data** port stays 0% as before.
- If a demand fill (`fill_from_stream_buffer`, step 2) takes the fill port first
  in the same cycle, the step-3 eager promote sees `fill_port_free() == false` and
  defers => Option B gives demand priority, as intended.

### 6.1 Failure modes

| Risk | Defense |
|---|---|
| **(A) Demand MISSes after eager promote** (mshr_addr mismatch -> line not actually in L1I) | **Critical.** eager promote must fill with the *same* key demand probes: `m_config.mshr_addr(prefetch_addr)` where `prefetch_addr == base_addr`. Verify `prefetch_addr == base_addr` and log `[L1IPFDBG][demand-MISS-after-promote]` as a first-class abort signal. |
| **(B) Waiter leak / deadlock on same-cycle race** | eager promote pops the head; if a demand registered a waiter via `set_waiting_fill_in_cache` just before, that waiter would never get its L0I response. **Defense: promote only when `is_request_to_cache == false` AND `waiting_warp_ids_and_its_addrs.empty()` (both).** Log `[L1IPFDBG][skip-has-waiter]`. |
| **(C) Duplicate tag-array fill / MSHR conflict** | A demand may have already put the line in L1I MSHR via normal access. Probe before promote; skip if HIT or MSHR-pending (do_prefetch pattern). Log `[L1IPFDBG][skip-already-cached]`. |
| Demand looks up SB after eager promote and misses | Intended. Demand is served by L1I HIT (`read_only_cache::access`). Fine (this is the goal of A being correct). |
| `m_next_response` assert in `fill_from_stream_buffer` | eager promote never creates `m_next_response`, so the assert is untouched. |
| (D) Useless prefetch polluting L1I (eviction) | monitor `prefetched_l1i_lines_evicted` (currently 0; low risk). |
| (E) invalidate/flush consistency at kernel end | eager promote drains SB earlier, so teardown is safer. Confirm interaction with `-invalidate_instruction_caches_at_kernel_end 1`. |
| Deadlock interaction with `num_stream_buffers` | eager promote drains the head without demand, which *reduces* HOL/deadlock pressure. Validate first with `num_stream_buffers = 1`. |
| Port starvation of eager promote | Option B defers (does not drop). Data stays in SB; retried next cycle. |

## 7. Instrumentation (new counters)

- `total_num_l0i_sb_eager_promote_to_cache` — number of eager promotes.
- `total_num_l0i_sb_eager_promote_skipped_already_cached` — probe HIT/MSHR skips.
- `total_num_l0i_sb_eager_promote_skipped_fill_port_busy` — Option-B deferrals.
- `total_num_l0i_sb_eager_promote_skipped_has_waiter` — risk-B defense fired.
- `total_num_l0i_sb_eager_promote_demand_hit_later` — eager-promoted line later
  hit by demand (usefulness metric; the healthy direction of `after_ready`).
- `total_num_l0i_sb_eager_promote_demand_miss_after_promote` — **critical**: a
  demand MISSed even though the line was eager-promoted (risk A). Must stay 0.

## 7.1 Debug logging (for 24h-run early triage)

All gated by a new config flag `-l1i_prefetch_debug_enable` (default 0), sharing
a print budget `-l1i_prefetch_debug_budget` to bound stderr volume. Logs are
kept on one line (OpenMP-safe), prefix `[L1IPFDBG]`.

Event logs (budgeted):
- `[L1IPFDBG][eager-promote] sm=.. sb=.. addr=0x.. base=0x.. mshr=0x.. issue_cyc=.. ready_cyc=.. lead=.. cycle=..`
- `[L1IPFDBG][skip-port-busy] sm=.. addr=0x.. cycle=..`
- `[L1IPFDBG][skip-already-cached] sm=.. addr=0x.. probe=HIT|MSHR cycle=..`
- `[L1IPFDBG][skip-has-waiter] sm=.. addr=0x.. nwaiters=.. cycle=..`
- `[L1IPFDBG][demand-hit-promoted] sm=.. addr=0x.. promote_cyc=.. demand_cyc=.. gap=..` (success signal)
- `[L1IPFDBG][demand-MISS-after-promote] sm=.. addr=0x.. cycle=..` (**critical**: kill the run, recheck mshr_addr)

Watchdog (always on, budget-independent):
- `[L1IPFDBG][stuck] sm=.. warp=.. addr=0x.. waited=.. cta=.. cycle=..` — a warp
  parked in SB_WAIT / promote-pending for more than N cycles (e.g. 5000) is
  reported once, for early deadlock detection.

Periodic summary (e.g. every 50k cycles, core 0 only):
- `[L1IPFDBG][summary] cycle=.. eager_promote=.. skip_port=.. skip_cached=.. skip_waiter=.. demand_hit_promoted=.. demand_miss_after_promote=.. after_ready_running=..`
  - `after_ready_running` = cumulative `head_demand_arrived_after_ready`. If it
    starts rising early -> success. If it stays 0 -> stop early.

Early go/no-go (observe first ~10 minutes):
- GO: `eager_promote > 0`, `demand_miss_after_promote == 0`, no `[stuck]`.
- NO-GO (kill immediately): `demand_miss_after_promote > 0` or any `[stuck]` ->
  recheck risk A / risk B.

## 8. Milestones

1. **M1**: config flag + new method stub + counters (no behavior change; build).
2. **M2**: implement the eager promote logic (flag ON).
3. **M3**: debug build, single-SM, `num_stream_buffers = 1` -> verify
   `eager_promote_to_cache > 0`, `after_ready > 0`, no deadlock.
4. **M4**: full run (kernel 5) -> compare cycles/stalls; record as Opt 4 in
   `.result/FA3_progress.md`.
5. **M5**: once stable, combine with the `num_stream_buffers` experiment.

## 9. Out of scope (this change)

- Prefetch degree/distance (keep next-line degree 1).
- Prewarm parameters (revisit after prefetch is stable).
- L1I capacity / ports (shown unnecessary).
- The head serial-service structure itself.

## 10. Reference: relevant code locations

- `first_level_instruction_cache::access` — prefetch trigger + demand handling.
- `first_level_instruction_cache::fill` — routes prefetch to stream buffer.
- `first_level_instruction_cache::fill_from_stream_buffer` — current fused
  promote + L0I response.
- `first_level_instruction_cache::cycle` — `m_stream_buffers->cycle(!m_next_response)`.
- `single_stream_buffer::do_prefetch` / `fill` / `send_to_cache` / `set_new_stream`.
- `multiple_stream_buffers::cycle` — round-robin promote.
- `baseline_cache::bandwidth_management::use_fill_port` / `fill_port_free`.

## 11. Hardware spec note (L1I sizing)

NVIDIA public docs do not disclose the instruction-cache hierarchy; the "256 KB
L1 per SM" figures refer to the unified data L1 + shared memory, not the I-cache.
Microbenchmarking literature (Jia et al. 2018 for Volta/Turing; arXiv 2501.12084
for Hopper) indicates L0I ~16-32 KB per sub-partition and L1I ~32 KB (some
measurements up to 128 KB). The current config already uses
`il1 = 64:128:16 = 128 KB` and `il0 = 8:128:32 = 32 KB`, i.e. L1I is already at
or above the measured upper bound. Therefore increasing L1I capacity is NOT
pursued; it lacks hardware justification and would only inflate an
already-generous setting.
