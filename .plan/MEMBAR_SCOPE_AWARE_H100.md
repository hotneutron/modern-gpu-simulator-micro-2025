# MEMBAR scope-aware memory fence on H100 (FA3 bwd `inst_barrier` follow-up to BAR_OP)

## Summary

After the `OP_BAR` fix (see `BAR_OP_H100.md`), FA3 fwd (kernel 5) dropped to a clean
162,582 cycles, but FA3 bwd (kernel 10 = `FlashAttnBwdSm90`) still overestimates by ~2.47×
(sim 328,643 vs HW 132,901), and `inst_barrier` is *still* the #1 stall at **44.78%** of
issue cycles / **76.84%** of `no_warps_ready`.

Root cause (verified from the `.e13` BARDBG log and the kernel SASS): the dominant barrier
event in bwd is **`MEMBAR.ALL.CTA` (55,296 dynamic issues, #1 by far)**, which the simulator
decodes as a **full-CTA blocking SYNC barrier** (`bar_type=SYNC, bar_count=-1`) and routes
through the CTA barrier engine (`warp_reaches_barrier()`). A `MEMBAR` is an
**ordering/visibility fence**, not a thread rendezvous: only the issuing warp waits, and only
until its own outstanding writes are visible at the requested scope. Forcing a full-CTA
rendezvous serializes the warp-specialized pipeline exactly like the old BAR bug did.

fwd was unaffected only because it issues `MEMBAR.ALL.CTA` 17× less often (3,168 vs 55,296).

This plan makes the memory barrier a **per-warp, scope-aware fence**:
- Remove the CTA rendezvous and the fixed SM-wide stall latency for MEMBAR/FENCE.
- Model the wait as "the issuing warp's outstanding stores are visible at the fence scope":
  - `MEMBAR.ALL.CTA` → CTA-visible stores drained (shared + L1-level global).
  - `MEMBAR.ALL.GPU` → GPU-visible stores drained (L2-level global + TMA), which subsumes CTA.

---

## Evidence (grounded in this workload)

### Dynamic barrier mix (from `.e13` `[BARDBG][issue]`, kernel 10 bwd)

| trace_opcode | bar_type | bar_count | dynamic issues | note |
|---|---|---|---|---|
| **MEMBAR.ALL.CTA** | 1 | -1 | **55,296** | #1 — wrongly treated as full-CTA SYNC |
| BAR.SYNC.DEFER_BLOCKING (id=9) | 2 | 256 | 26,112 | partial-count (handled by BAR fix) |
| BAR.SYNC.DEFER_BLOCKING (id=10/11) | 2 | 160 | 13,056 each | partial-count |
| BAR.ARV (id=13/14) | 2 | 160 | 13,056 each | arrive-only |

fwd (`.e12`) has the same `MEMBAR.ALL.CTA` decode but only **3,168** dynamic issues.

### Static MEMBAR / FENCE forms (SASS)

| form | bwd | fwd | classification |
|---|---|---|---|
| `FENCE.VIEW.ASYNC.S` | 399 | 78 | async-proxy fence (already lightweight) |
| `FENCE.VIEW.ASYNC.G` | 0 | 3 | async-proxy fence |
| `MEMBAR.ALL.CTA` | 372 | 24 | CTA-scope fence (mostly after `STS.128` / `HGMMA`) |
| `MEMBAR.ALL.GPU` | 120 | 0 | GPU-scope fence (after TMA store / `UBLKRED`) |

`MEMBAR.ALL.CTA` predominantly follows **shared stores (`STS.128`) / WGMMA** → shared-memory
ordering, *not* an L2 store fence. `MEMBAR.ALL.GPU` follows TMA global stores / reductions →
must wait until those reach L2.

---

## Hardware semantics (what we are modeling)

A memory fence orders the issuing warp's prior memory writes against later ones, up to a
**scope** (set of observers). It does **not** wait for other warps to arrive.

| scope | observers | visibility point | fence waits for |
|---|---|---|---|
| `.CTA` | other threads of the same CTA | shared mem + L1 (same SM) | shared stores + L1-level global stores |
| `.GPU` | other CTAs (other SMs) | L2 (shared by all SMs) | L2-level global stores + TMA stores |

Cost is **not** a fixed cycle count; it is proportional to draining the warp's outstanding
writes to that scope. With no outstanding writes, the fence passes almost immediately. This
is why HW NCU shows membar stalls at ~0% for this kernel.

### WGMMA / async-proxy ordering is handled by the SB wait-barrier path, independent of MEMBAR's rendezvous

Verified by decoding **all** SASS control words (`CCPos_arch_7x_8x = 41`;
`wait_bits=(cw>>41>>11)&0x3f`, `id_w=(cw>>41>>5)&7` [7=none], `id_r=(cw>>41>>8)&7`):

| instruction | counts (wait_bits, id_w, id_r) | reading |
|---|---|---|
| `HGMMA` (WGMMA) | all 3008 = `(wait=0, id_w=7, id_r=7)` | WGMMA sets **no** SB write-barrier and waits on **no** SB |
| `MEMBAR.ALL.CTA` | `wait=0`, `id_r` ∈ {1,2,3,4} | only **arms a read barrier id**; never waits |
| `FENCE.VIEW.ASYNC.S` | `wait` ∈ {0,2,4,8,16}, `id_w` ∈ {0,1,2,3,4} | waits on / sets SB barriers; **`wait=0` cases (~147) just take fixed latency** |

Corrected understanding (supersedes earlier drafts):
- `MEMBAR.ALL.CTA` genuinely has `wait_bits=0` (the sim parses it correctly); it is **not** a
  WGMMA wait and **not** a parse bug.
- **WGMMA completion is not tracked by the SB wait-barrier system at all** (HGMMA: id_w=7,
  wait=0). WGMMA ordering is enforced by the separate `WARPGROUP`/`DEPBAR` mechanism, which is
  untouched by this change.
- The `MEMBAR.ALL.CTA` + `FENCE.VIEW.ASYNC.S` pair forms a **generic↔async-proxy ordering**
  for **memory ops** (shared/STS etc.) via SB read/write barriers, *not* for WGMMA.
- `FENCE.VIEW.ASYNC.S` with `wait_bits!=0` (252 sites) is **really gated** by the generic
  wait-barrier check; with `wait_bits=0` (~147 sites) it just takes the FU fixed latency
  (matches the "FENCE is just default-latency" intuition for those cases).

**Why this is safe under the rendezvous removal (Open items 1 & 2 — resolved):**
- The generic wait-barrier check is at the issue stage
  (`subcore.cc:514-515,553-556`: `are_wait_barriers_ready` ∈ `are_switch_warp_conditions_ready`)
  and applies to **every** op regardless of type — it is **independent** of the
  MEMORY_BARRIER_OP barrier-engine path. Bypassing `warp_reaches_barrier()` for MEMBAR/FENCE
  does **not** disable it. ✔
- The MEMBAR/FENCE **control-bit bookkeeping** (arm/decrement read/write SB barriers,
  stall_count, yield) runs in the **common tail of `func_exec_inst`** (`sm.cc:696-725`),
  *after* the `if/else-if` op branch (`670-694`). Skipping the `warp_reaches_barrier()` call
  inside the MEMORY_BARRIER_OP branch leaves this tail fully intact. ✔
- Wait-barrier **increments** happen on the FU dispatch path (`subcore.cc:349-350`), also
  independent of the barrier engine. ✔

**Consequence:** the scope-aware MEMBAR change only needs to model **store visibility**;
WGMMA/FENCE SB ordering keeps working unchanged.

---

## Existing infrastructure (verified — reused, not rebuilt)

| visibility level | signal | inc site | dec site (= reaches level) | status |
|---|---|---|---|---|
| L2 TMA store | `tma_unit_sm::warp_has_outstanding_stores(warp)` | TMA store enqueue (`tma_unit_sm.cc:484`) | mover L2 response complete (`tma_unit_sm.cc:824`) | exists, already used by `UTMACMDFLUSH` (`sm.cc:1869-1888`) |
| L2 global store | `m_stores_outstanding` (L2-bypass path) | `dispatch_directly_to_l2` (`ldst_unit_sm.cc:1168`) | L2 `WRITE_ACK` (`ldst_unit_sm.cc:908`) | exists |
| L1 global store | `m_stores_outstanding` (L1 path) | `process_cache_access` / latency queue (`ldst_unit_sm.cc:305`, `439`) | L1 hit ack (`ldst_unit_sm.cc:647`) | exists |
| shared store (STS/STSM) | `m_current_num_shared_mem_inst` (SM-wide, load+store) | shared dispatch (`ldst_unit_sm.cc:1079`/`1101`) | PRT store retire, `is_shared()` branch (`ldst_unit_sm.cc:1559-1567`) | **per-warp store-only counter missing — must add** |
| WGMMA / async-proxy write | WGMMA group-scoreboard (gsb) completion | HGMMA issue | WARPGROUP/scoreboard completion | exists for `DEPBAR`; **not wired to MEMBAR (wait_bits=0)** |

Verified facts (this investigation):
- **#5 shared store path confirmed**: every shared instruction goes through the PRT
  (`m_prt->assign_entry`, `ldst_unit_sm.cc:773`); shared **stores** are retired (no
  write-back) at `ldst_unit_sm.cc:1559-1567` inside the `res->space.is_shared()` branch,
  where `m_current_num_shared_mem_inst--`. inc is at `1079`/`1101`. The existing counter is
  **SM-wide and counts loads+stores**, so a **new per-warp, store-only** counter is required
  (inc/dec at the same two sites, gated on `is_store()`).
- **#6 config confirmed**: `-gpgpu_gmem_skip_L1D 0` and dl1 = `S:4:128:512,L:T:...`
  (write-**through**). L1D is **not** globally skipped, so L1-level global stores *do* exist;
  `is_l1d_bypass()` is true only for `cache_op==CACHE_GLOBAL` (`.cg`). → CTA-vs-GPU split is
  meaningful and must be kept.
- **#4 sector granularity confirmed**: store inc/dec is per **sector** mem_fetch
  (`inc_ack = data_size/SECTOR_SIZE`, `ldst_unit_sm.cc:300-305`, dec mirrored at `647`/`908`).
  The new per-warp counters must inc/dec at the **same sector granularity** so they return to
  0 exactly. Tagging the visibility level on the `mem_fetch` makes each sector carry its tag,
  keeping inc/dec balanced regardless of the ack path.

Key facts established earlier:
- A store has no destination register, so it is **not** tracked by the scoreboard; the
  scoreboard models register dependencies, not memory visibility. The scoreboard / mbarrier
  paths must **not** be used as the fence condition.
- The `is_l1d_bypass()` flag on `mem_access_t` is set at issue time
  (`ldst_unit_sm.cc:1860-1862`), so the CTA-vs-GPU visibility level of a global store is
  **known at the inc site**.
- The single `m_stores_outstanding` counter mixes L1-hit dec (`647`) and L2-ack dec (`908`),
  so it cannot distinguish CTA-visible from GPU-visible — but it is **also used by
  `stores_done()` for warp/kernel exit** (`shader.cc:4694`). Therefore do **not** repurpose
  it; add **separate** per-warp counters alongside it.

---

## Design — per-warp, two-level store tracking + scope-aware fence

### 1. Add two per-warp visibility-level store counters (do NOT repurpose `m_stores_outstanding`)

Keep `m_stores_outstanding` (used by `stores_done()` for warp/kernel exit) untouched. Add
**two new per-warp counters** alongside it (per `shd_warp_t`, or a per-warp map on `SM`):

```
m_pending_stores_cta_visible   // shared stores (STS/STSM) + L1-level global stores
m_pending_stores_gpu_visible   // L2-level (L1-bypass, .cg) global stores
```

(TMA stores keep their own existing counter in `tma_unit_sm` and are folded into the
GPU-visible condition; no new TMA counter is needed.)

| counter | inc (per sector) | dec (per sector) |
|---|---|---|
| `cta_visible` | shared-store dispatch (new, `1079`/`1101`, gated `is_store()`) + global store when `!acc->is_l1d_bypass()` (`305`/`439`) | shared-store PRT retire (new, `1560`, `is_shared()&&is_store()`) + L1 hit ack (`647`) |
| `gpu_visible` | global store when `acc->is_l1d_bypass()` (`1168`) | L2 `WRITE_ACK` (`908`) |

Inc/dec must be at **sector granularity** (matching the existing `inc_ack`/`dec_ack`), so the
counters return to exactly 0. To keep dec correct regardless of which ack site fires, tag the
visibility level on the `mem_fetch` at issue time (one enum/bit: `CTA` vs `GPU`), then dec the
matching counter in `store_ack` / the L1-hit path using that tag. Shared stores have no
`mem_fetch` (handled via PRT), so they inc/dec `cta_visible` directly at the shared dispatch
and the PRT store-retire site.

### 2. Carry the fence scope on the warp

`MEMBAR` issue currently only sets a boolean (`set_membar()`); it does not remember the
scope. Extend it:

- `shd_warp_t`: add `m_membar_scope` (enum `{CTA, GPU}`; SYS reserved/asserted).
- At MEMORY_BARRIER_OP issue, derive scope from the trace opcode
  (`MEMBAR.ALL.CTA` → CTA, `MEMBAR.ALL.GPU` → GPU) and `set_membar(scope)`.

### 3. Scope-aware fence wait

Rewrite `SM::warp_waiting_at_mem_barrier(warp_id)`. WGMMA/async-proxy ordering is **not** part
of this condition — it is already enforced by the adjacent `FENCE.VIEW.ASYNC.S` through the
generic wait-barrier path (`subcore.cc:762-773`). The fence only waits on store visibility:

```
if (!get_membar(warp_id)) return false;            // not at a fence
switch (get_membar_scope(warp_id)) {
  case CTA:
    done = (cta_visible(warp) == 0);               // shared + L1 global drained
    break;
  case GPU:                                          // GPU subsumes CTA
    done = (cta_visible(warp) == 0)
        && (gpu_visible(warp) == 0)
        && !m_tma_unit_shared_of_sm->warp_has_outstanding_stores(warp);
    break;
}
if (done) { clear_membar(warp); /* keep existing L1-invalidate-on-flush hook */ }
return !done;
```

No fixed scope latency (`53/186/2900`) is applied — the wait is the real data-drain time.

### 4. Remove the rendezvous + dead config (already partially done)

- `sm.cc` MEMORY_BARRIER_OP issue: skip `warp_reaches_barrier()` /
  `store_info_of_last_inst_at_barrier()` for every memory fence (`is_non_rendezvous_memory_barrier`,
  extended to `FENCE.*` + `MEMBAR.ALL.CTA/GPU`, with an `assert` guard for unverified forms
  such as `MEMBAR.*.SYS`). **Done.**
- Drop the `m_num_cycles_to_stall_SM_at_{cta,gpu,system}_memory_barrier` assignment on the
  MEMBAR path (per-warp drain replaces SM-wide stall). **Done.**
- `trace_driven.cc` MEMBAR placeholder (`bar_type=SYNC, bar_count=-1`) annotated as dead
  (never consumed once MEMBAR bypasses the barrier engine). **Done.**
- The now-unreachable MEMORY_BARRIER_OP branch in `shader.cc:4365-4366`
  (`num_cycles_to_stall_SM` consumption after a full-CTA rendezvous) can be left as-is for
  the baseline (non-trace) model or removed if confirmed unused.

---

## Files to touch

| file | change |
|---|---|
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc` | `is_non_rendezvous_memory_barrier` (done); MEMBAR issue rendezvous/latency removal (done); `warp_waiting_at_mem_barrier` scope-aware rewrite; `set_membar(scope)` plumbing |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h` | `shd_warp_t`: add `m_membar_scope`, two per-warp store counters + accessors |
| `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/ldst_unit_sm.cc` | split inc/dec into cta/gpu counters by `is_l1d_bypass()`; add shared-store inc (enqueue) / dec (PRT retire `1560`) |
| `gpu-simulator/gpgpu-sim/src/abstract_hardware_model.*` (mem_fetch) | add fence-visibility-level tag set at store issue, read at `store_ack` |
| `gpu-simulator/trace-driven/trace_driven.cc` | MEMBAR placeholder comment (done) |

---

## Verification plan

1. **Build** then re-run kernel 10 bwd with `-bar_debug_enable 1`.
2. **Correctness guards**:
   - `.e13`: `MEMBAR.ALL.CTA` no longer appears in `[BARDBG][issue]` (it bypasses the barrier
     engine); `leaked_ids=0` maintained.
   - Per-warp `cta_visible` / `gpu_visible` counters return to 0 at kernel end (no leaks).
3. **fwd regression**: re-run kernel 5; cycles must not regress from 162,582 (fwd has only
   24 static / 3,168 dynamic MEMBAR.ALL.CTA, so the effect should be small but positive).
4. **Expected effect (bwd)**: `inst_barrier` (44.78%) drops sharply; `no_warps_ready`
   (58.27%) drops; `gpu_tot_sim_cycle` falls from 328,643 toward HW (132,901). Memory-fence
   stalls should land in a small per-warp wait bucket, consistent with HW membar ≈ 0%.
5. Compare the TMA/non-TMA stall distribution shape against the HW table in
   `.result/FA3_kernel_10_bwd.md` rather than only absolute cycles.

---

## Open items — all resolved

1. **FENCE/MEMBAR SB wait-barrier under the bypass change** → **safe.** The generic
   `wait_barrier_bits` check is at the issue stage (`subcore.cc:514-515,553-556`), independent
   of the MEMORY_BARRIER_OP barrier-engine path. Bypassing `warp_reaches_barrier()` does not
   disable it.
2. **MEMBAR `id_r` / read-write barrier bookkeeping under the bypass change** → **safe.** Arm
   /decrement of read/write SB barriers (plus stall_count, yield) runs in the common tail of
   `func_exec_inst` (`sm.cc:696-725`) after the op `if/else-if`, and increments run on the FU
   dispatch path (`subcore.cc:349-350`); neither is inside the skipped `warp_reaches_barrier()`
   call.
3. **WGMMA dependency** → **not relevant.** WGMMA neither sets nor waits on SB barriers
   (HGMMA: id_w=7, wait=0); its completion is handled by the `WARPGROUP`/`DEPBAR` mechanism,
   untouched here.

---

## Risks / notes

- **CTA-MEMBAR models store visibility only.** WGMMA neither sets nor waits on SB barriers
  (verified: HGMMA id_w=7/wait=0); its ordering is the separate `WARPGROUP`/`DEPBAR` path. The
  `MEMBAR+FENCE.VIEW.ASYNC.S` SB ordering is for memory ops and is enforced independently at
  the issue stage. Do **not** re-introduce a rendezvous, and do **not** add a WGMMA wait to
  the fence condition.
- **Scope subsumption**: GPU-scope must also wait on `cta_visible` (a GPU-visible fence
  implies CTA visibility). The condition above includes it.
- **Do not repurpose `m_stores_outstanding`** — it backs `stores_done()` for warp/kernel exit
  (`shader.cc:4694`). Add separate per-warp counters.
- **`MEMBAR.ALL.SYS`** does not appear in either traced kernel; it is asserted out in
  `is_non_rendezvous_memory_barrier` until characterized (mirrors the OP_BAR guard).
- **mem_fetch tagging** must be set once at issue and read at every ack site (`647`, `684`,
  `908`) at sector granularity so the correct level is decremented no matter which path the
  store took.
- **Shared-store counters**: the existing `m_current_num_shared_mem_inst` is SM-wide and
  counts loads+stores; the new per-warp counter must be store-only (gate on `is_store()`) and
  inc/dec at `1079`/`1101` and the `is_shared()` retire branch (`1559-1567`).
- Keep `-sync_debug_enable 1` while iterating so `[TMADBG]` store-outstanding ++/-- events
  can be correlated with the new fence wait.
