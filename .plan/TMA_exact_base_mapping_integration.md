# TMA Exact Base-Mapping Integration Plan (remove heuristics → exact mapping)

## Context (why this work)

The simulator currently synthesizes the TMA transfer address:
`agu_base = (transfer_uid << 20) + agu_index*128` ([tma_unit_sm.cc:637](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L637)).
Because `transfer_uid` is unique per transfer, **every transfer gets a different address**, so L2 locality is fake (repeated reads of the same tensor never hit). Real HW reads the tensor's **real GMEM base** from the CUtensorMap descriptor and computes `addr = base + Σ coord·stride`, so repeated accesses hit in L2. HW L2 hit targets: fwd ~70%, bwd ~82%.

We have **already solved** exact base recovery. `build_tma_pc_base_map.py` produces `tma_pc_base_map.json`, giving each descriptor op's `(uid,pc) → exact real base`. On FA3: **23/23 mapped**, zero heuristic fallback (direct + UTMACCTL prefetch-chain cover 100%). This plan (A) wires that exact mapping into the trace-gen pipeline, (B) makes the simulator use the real base instead of the synthetic one, and (C) removes the old handle_hi heuristic.

Core principle (user requirement): **every opcode the old heuristic mapped a descriptor to must also be covered by the new pipeline.** The data-confirmed per-opcode handling is in §"Per-opcode handling" below.

## Per-opcode handling (verified against the real FA3 trace)

| opcode | desc_refs | new-mapping handling | evidence |
|---|---|---|---|
| **UTMALDG**(.4D) | yes | ci=0 SMEM-window offset = tensor ID → direct/chain → real base | already in `DESC_OPS`, included in 23/23 |
| **UTMASTG**(.4D/.5D) | yes | same (ci=0 SMEM window) | already in `DESC_OPS` |
| **UBLKRED**.G.S.ADD.F32 | yes (but bare handle) | **NOT a tensormap descriptor → synthetic base in M1, size exact via covered_bytes. M2.5 upgrades base to real (raw-ptr from param blob) + mock tiling (§M2.5).** The executed sites (0x90a0 etc., 6 total) set `desc[UR16]` via `UMOV UR16,0x0`+`UMOV UR17,0x14f00000` = a bare handle. dst is a raw dQaccum GMEM pointer (`c[0x0][0x280]/0x2a0`) + dynamic tile offset | executed: 0x90a0/0x91d0/0x94e0/0x95b0/0x96f0/0x97c0. The extractor's offset 0x30d96 is a **false positive** from unmodeled UMOV (outside the 0x700 struct). Only `covered_bytes` (operand-3) is exact |
| UBLKCP.S.G | no | not a tensormap (desc_valid=false). Synthetic base in M1; **M2.5 = per-pc real base + mock tiling (§M2.5)**; size from covered_bytes | ci=0/ci=1 are raw addrs, not tensor IDs |
| **UTMACCTL.PF** | prologue | **exact mapping target (for prefetch modeling).** prologue `UIADD3 URx, param_base, off` (off ∈ {0xb0,0x170,0x230,0x2f0,0x4f0,0x5b0}) → `off−0x30` = struct slot → base. clean 6→6 | must know which tensor is prefetched (user requirement). single-source ⇒ no ambiguity |
| UTMAPF | — | **absent in the bwd kernel (count=0).** No UTMACCTL.PF→UTMAPF→UTMALDG three-step | grep-confirmed |
| UTMACMDFLUSH | no | not mapped | control-only (matches user expectation) |

**Unifying rule**: base-map targets = **only sites that reach a 128B tensormap descriptor in the struct**. UTMALDG/UTMASTG (runtime SMEM-window offset) and UTMACCTL.PF (prologue param offset) qualify. UBLKRED/UBLKCP are raw GMEM pointers + dynamic offsets, so **their base is synthetic in M1; M2.5 upgrades them to a real base read offline from the by-value param struct + mock tiling (§M2.5)**, while size (covered_bytes) is exact throughout. UTMACMDFLUSH issues no data request.

**Required extractor fix (remove the UBLKRED false-positive)**: `extract_tma_descriptor_offsets.py`'s `build_defs()` does not model `UMOV`, so for executed UBLKRED (0x90a0 etc.) it skips the live `UMOV UR16,0x0` and walks a stale UIADD3, fabricating a **bogus offset (0x30d96, outside the 0x700 struct)**. Fix: (1) model `UMOV URd,imm` as a def → a umov terminus returns `None` (dead-end) in `trace_offset()`; (2) **struct-bounds guard**: accept `resolved` only when `0 ≤ offset < struct_size(uid)` (else `offset_outside_param_struct`). Then UBLKRED honestly falls to "unresolved," leaving only UTMALDG/UTMACCTL.

## Per-opcode **dest (where) + size (how much)** — data-confirmed sources (direct answer to the user's question)

**Premise (data-confirmed):** every runtime-operand GMEM address keeps only its low 32 bits (e.g. `fla=0x722804c0, value_hi=0x85`, hi not merged). So **no op's operand value is itself the real GMEM dest.** There are two classes:

| opcode | **dest (GMEM address) source** | **size source** | simulator path |
|---|---|---|---|
| **UTMALDG/UTMASTG** | operand ci=0 = SMEM descriptor window (`0x72..04c0`), whose **low16 = tensor ID (0x4c0)** → base-map lookup → **descriptor qword0 = real base** | descriptor **box_dim × element_size** (no span operand; no ci=2) | build_tma_command: (uid,pc)→base map→`global_base`+box. mover: `base + agu_index*128` |
| **UTMACCTL.PF** (bwd) / **UTMAPF.L2** (fwd) | prologue `param_base+off` → `off−0x30` = struct slot → **descriptor qword0** (the prefetched tensor) | prefetch (whole) — descriptor box | emit per-pc into base map → which tensor the prefetch warms |
| **UBLKRED** | **dest = raw dQaccum GMEM ptr**. desc is a bare handle (no base). synthetic in M1; **M2.5 = real base (offline from `launch_param_blobs`) + mock tiling** | operand **ci=2 span** (`0x400*16 = 16384B`) → covered_bytes | size exact from operand; base synthetic in M1, real in M2.5 |
| **UBLKCP** | GMEM addr (distinct per transfer) + SMEM cursor — not a tensormap. **measured cb: GMEM=cb0, SMEM=cb1, span=cb2** | operand **ci=2 span** (`0x20*16 = 512B`) → covered_bytes | size exact; base synthetic in M1, **M2.5 = per-pc real base + mock tiling** |
| **UTMACMDFLUSH** | none | none | control, no data request |

**Summary:** dest is obtained (a) **through the descriptor** (UTMALDG/UTMASTG/UTMACCTL — the base map provides it exactly), or (b) via a **raw pointer** (UBLKRED/UBLKCP — synthetic in M1; **real base in M2.5**, read offline from the by-value param struct in `launch_param_blobs`). size comes from (a) the **descriptor box** (UTMALDG family) or (b) the **operand span ci=2** (bulk family), **both already present in the trace and therefore exact**. Net: **size is exact for every op; dest is exact for the descriptor family in M1 and for UBLKRED/UBLKCP in M2.5**.

## Per-opcode **transfer size** (how much moves) — basis for issuing data requests

base (address) and size (amount) are **separate data**. base decides "where," size decides "how many 128B requests to issue." The simulator issues `requests_total` requests to icnt ([tma_unit_sm.cc:583](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L583)); a wrong size means wrong traffic volume. Today's size comes from **two sources depending on opcode** (all already implemented/verified, [build_tma_command:373-401](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L373)):

| opcode | size source | formula | data origin |
|---|---|---|---|
| **UTMALDG / UTMASTG / UTMAPF / UTMAREDG** | **descriptor box** | `total_bytes = Πbox_dim[d] × element_size`; `requests_total = ⌈box_dim[0]·elem/128⌉ × Πbox_dim[1..]` ([:75-113](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L75)) | `box_dim`·`element_size` in `tma_descriptor_configs.json` |
| **UBLKRED** | **operand covered_bytes** | `total_bytes = covered_bytes`; `requests_total = ⌈covered_bytes/128⌉` ([:393-396](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L393)) | operand-3 (covered_bytes) in `tma_operand_resolver.json` |
| **UBLKCP / UBLKPF** | **operand covered_bytes** | same (⌈covered_bytes/128⌉) ([:451-459](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L451)) | operand-3 |
| UTMACMDFLUSH | none | no data request (control) | — |

**Key gap (raised by the user — captured in the plan):** the descriptor family's size comes from `box_dim/element_size` (= `descriptor_config`), but the key that attaches that config to a site is **`config_id` via the handle_hi resolver**. Removing the handle_hi resolver in M3 breaks **not only base but also the size path (= the config link)**. So the new mapping must provide not only base but also the **(uid,pc) → config_id (or box_dim/element_size directly)** link.

**Resolution (added to M1 Part A):** `build_tma_pc_base_map.py` already knows each (uid,pc)'s tensor (SMEM offset → struct descriptor), so **parse box_dim·element_size directly from that 128B struct descriptor** and carry it in the base-map entry. That is, extend each `tma_pc_base_map.json` entry to `{base_hex, box_dim[], element_size, element_stride[], source}`. Then:
- descriptor family: (uid,pc) → base **and** box/element_size via one path → no handle_hi/config_id needed.
- UBLKRED/UBLKCP: size still uses operand covered_bytes (the real value in the trace) — unchanged. The base map marks them `source=operand_addressed` only.

This lets `tma_descriptor_configs.json` (box/stride) itself stay in M1 (coexisting with the config_id path initially), while **removing the handle_hi resolver in M3 is safe because size is now covered by the new base map**. strides (for coord math, M2) come from the same struct descriptor, so this extension is also the prerequisite for M2.

## Replace **all** descriptor fields the handle_hi path supplies with the real mapping (user requirement)

The fields the handle_hi resolver → `config_id` path attaches to a site are **all 11** of `TMADescriptorConfigMetadata` ([tma_types.h:51-64](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h#L51)):
`tensor_rank, tensor_data_type, global_dim[5], global_strides[5], box_dim[5], element_strides[5], interleave, swizzle, l2_promotion, oob_fill, element_size`.

All 11 live **inside the 128B CUtensorMap descriptor**, and `tensor_map_encode_dump.csv` already carries them as columns (verified):
`global_address_hex` (=qword0=base), `tensor_data_type=9`, `tensor_rank=4`, `global_dim="64 2048 24 1"`, `global_strides="3072 128 6291456"`, `box_dim="64 192 1 1"`, `element_strides="1 1 1 1"`, `interleave/swizzle/l2_promotion/oob_fill`, + `blob_path` (128B original).

**Replacement method (same exact path as base):** struct discovery matches each (uid,pc)'s 128B **byte-for-byte to an encode blob** (that's how we already get base=qword0). So reading **all 11 fields from that one matched encode-dump row** yields every field without handle_hi/config_id. That is, extend the `tma_pc_base_map.json` entry to the **full descriptor**:
```
"uid:pc": { "source":"direct|chain", "encode_id": <dump_id>,
            "base_hex", "tensor_rank", "tensor_data_type",
            "global_dim":[...], "global_strides":[...],
            "box_dim":[...], "element_strides":[...],
            "interleave","swizzle","l2_promotion","oob_fill","element_size" }
```
UBLKRED/UBLKCP (`operand_addressed`) carry no descriptor fields and use only trace operands (address·covered_bytes) — unchanged.

With this extension, §transfer size's box/element_size is automatically included (same row), and M2's coord-math global_strides/element_strides are also secured. Net: **one (uid,pc)→exact descriptor path replaces everything the handle_hi path supplied**.

## pool vs exact — answer to the user's question

**Every descriptor op (UTMALDG/UTMASTG) knows the exact addr (and all fields). pool is not used.** Server re-verification on the latest trace (M0):
```
uid3: direct=6 chain=5 pool=0 unresolved=0
uid8: direct=4 chain=8 pool=0 unresolved=0   → 23/23, pool=0
```
- **direct**: runtime SMEM offset == struct descriptor slot → that 128B directly → exact.
- **chain**: UTMACCTL.PF copies struct→SMEM (param_source−0x30==slot) → that 128B → exact.
- **pool**: the earlier fallback that "deterministically assigns from a per-kernel base pool when a SMEM offset matches neither direct nor chain." **0 occurrences in this data** — i.e. never fires. Keep it in the code as a locality safety net (same SMEM offset = same base), but make the strict gate fail if pool is used, to enforce "100% exact mapping."

In short: **descriptor ops (UTMALDG/UTMASTG) + UTMACCTL.PF pin the 128B tensormap byte-for-byte → all fields exact, pool unused.** **UBLKRED/UBLKCP do not use a tensormap descriptor** (UBLKRED desc = UMOV bare handle, base = raw dQaccum GMEM pointer + dynamic offset). In M1 keep base synthetic, size exact via covered_bytes. **M2.5 makes their base real** by reading the raw pointer offline from the by-value param struct (`launch_param_blobs/*.bin`) — a *tensormap* mapping is impossible (proven), but the raw pointer itself is a plain GMEM base and is already captured. Verified on the local trace (§M2.5).

## Architecture: data flow and injection point

```
.traceg → trace_driven.cc (tma_handle_hi, tma_opcode_family)
   → tma_unit issue → build_tma_command(inst)   ← the only point where (uid,pc) are live
       → lookup base(uid,pc) → store into TMACommand.global_base
   → enqueue (transfer_uid assigned)
   → mover_issue_requests → compute agu_base   ← use TMACommand.global_base here
```
Key constraint: `mover_issue_requests` ([tma_unit_sm.cc:578](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L578)) only sees `TMATransferEntry` and does not know (uid,pc). So the base must be looked up in `build_tma_command` ([:344](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L344)) and carried on `TMACommand` to the mover.

## Milestones

### M0 — re-verification (read-only, no build) — DONE
- `build_tma_pc_base_map.py` server re-verification: `mapped: 23, direct+chain, pool=0, locality OK` ✓
- UBLKRED characterized (done): executed sites are not tensormap descriptors. desc is a UMOV bare handle, base is a raw GMEM pointer. The extractor offset is a UMOV-unmodeled false positive. → M1 excludes base, size from covered_bytes.
- UTMACCTL.PF characterized (done): prologue `param_base+off` (6), `off−0x30` = struct slot → the prefetched tensor is exactly mappable.

### M1 — base-only simulator injection + trace-gen integration

**Part A (Python)**
0. **Fix `extract_tma_descriptor_offsets.py` (prerequisite)**: (a) add `UMOV URd,imm` modeling in `build_defs()` → a umov terminus returns `None` in `trace_offset()` (constant dead-end); (b) on accepting `resolved`, add a **struct-bounds guard** (`0 ≤ offset < struct_size(uid)`, struct_size from `tma_launch_param_dump.csv`). → removes the executed-UBLKRED bogus offset 0x30d96; UBLKRED honestly falls to unresolved.
1. Extend `build_tma_pc_base_map.py`:
   - UTMALDG/UTMASTG/UTMAREDG: keep the existing SMEM-offset direct/chain path.
   - **Add UTMACCTL.PF emit**: output the prologue `prefetch_sites` (uid,pc)→`off−0x30`=struct slot→base as **per-pc entries** (currently only consumed as a chain linker, not emitted). Lets the simulator know which tensor each prefetch warms. clean 6→6.
   - **Carry size fields (§transfer size)**: parse `box_dim[]`·`element_size`·`element_stride[]`·`global_strides[]` from the 128B struct descriptor each (uid,pc) points to, and record them in the entry → the descriptor family gets size (and M2 strides) without handle_hi/config_id.
   - **UBLKRED/UBLKCP**: not added to the base map (not tensormaps). Simulator keeps synthetic; size from covered_bytes (operand-3, §Part A-3).
   - **strict gate**: if any base-map target site (UTMALDG/UTMASTG/UTMACCTL.PF) fails to resolve to an exact base, `SystemExit`. Also fail if pool is used (enforce 100% exact). UBLKRED is not a gate target (synthetic is fine).
   - **UBLKRED 1:1-absence diagnostic (for proof)**: attempt, per executed UBLKRED, to reach an in-struct slot → **0 resolved** is the expected result (desc = umov dead-end, dst = c[0x0][0x280]/0x2a0). That null is the evidence that "UBLKRED cannot be 1:1-mapped to a tensormap."
2. Add `--configs-only` to `build_tma_descriptor_mapping.py`: emit only `tma_descriptor_configs.json` (box_dim/strides/swizzle — still needed by the simulator), and stop emitting `tma_descriptor_resolver.json` (handle_hi heuristic).
3. `build_tma_operand_mapping.py`: drop the resolver dependence in `load_descriptor_refs` ([:77](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/build_tma_operand_mapping.py#L77)); switch the descriptor-required check to be based on `tma_pc_base_map.json`. Keep the covered_bytes (operand-3) extraction path for UBLKRED/UBLKCP — that is where size comes from.
4. Wire into `run_hw_trace.py` post-processing order ([:203](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/run_hw_trace.py#L203)):
   `discover_tma_producers` → `extract_tma_descriptor_offsets` (fixed) → `build_tma_descriptor_mapping --configs-only` → `build_tma_pc_base_map` (new, strict gate) → `build_tma_operand_mapping` (rewired) → `build_sync_operand_mapping` (kept). Guard each step with `os.path.exists`.

**Part B (C++)**
1. [tma_types.h](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h): add `TMABaseLookupKey{uid,pc}` + `TMABaseRecord{has_static_base, global_base, operand_addressed, box_dim[5], element_size, element_stride[5]}` (base and size in one record), add a `base_records` map to `TMASidecarMetadataDB` ([:128](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h#L128)). Add carrier fields to `TMACommand` ([:236](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h#L236)): `uint64_t global_base; bool has_real_base;` (coords[5]·box_dim already exist).
2. [gpu-sim.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc): new `load_tma_pc_base_map` (split key `"uid:pc"`, uid base-10, pc base-0; also load box_dim/element_size). Call it from `parse_extra_trace_info` ([:629](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L629)).
3. `build_tma_command` ([:344](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L344)): look up `TMABaseRecord` by (uid,pc). For static base, set `cmd.global_base=rec.global_base` **and `cmd.box_dim/element_size=rec.*`** → compute `total_bytes/requests_total` from the new record (reuse the existing [:384-386](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L384) formulas). For operand_addressed (UBLKRED/UBLKCP), base stays synthetic and size uses covered_bytes (keep the existing [:391-401](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L391) path). Set `cmd.has_real_base=true` for descriptor sites. **In M1 keep the config_id path coexisting** (cross-check that size matches the old path); remove the config_id path in M3.
4. `mover_issue_requests` ([:637](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L637)): if the flag is on and has_real_base, `agu_base = cmd.global_base + agu_index*MAX_MEMORY_ACCESS_SIZE`; else the existing synthetic. [:678](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L678) addr computation unchanged.
5. Feature flag `-tma_real_base_addr_enable` (default off): off ⇒ bit-identical to today. Add a bool to `shader_core_config`.
6. assert-on-miss (flag-gated): if a descriptor-required op has `has_real_base=false`, assert (mirror the existing Phase-2 block [:415](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L415)). Non-descriptor ops keep the synthetic fallback and do not assert.

**Verification (M1)**: build the simulator → run FA3 fwd/bwd with `-tma_real_base_addr_enable 0` (baseline) and `1`.
- **Size cross-check (prerequisite)**: confirm that `requests_total` computed from the base-map box_dim/element_size **matches** the old config_id path **at every (uid,pc)** (assert on mismatch). This must hold before cutting the config path in M3.
- **Base effect**: with the flag on, the `(transfer_uid<<20)` cross-SM hotspot disappears; distinct tensors land in distinct L2 regions. The decision metric is not cycles but the **L2 hit-rate direction toward HW** (from the inflated ~0.98 toward fwd 70% / bwd 82%). Base-only can transiently raise hit rate before coords land — judge by direction.

### M2 — coordinates (later, needs a tracer mini-spike)
per-transfer coords are not captured today (§0). To complete `addr = base + Σ coord·stride`, add coord capture + combine with `element_stride[]` (already carried in the base map in M1). UBLKRED needs no coords (operand_addressed).

## M1 verification results (FA3 fwd K5 + bwd K10, real-base flag on) — DONE

Both kernels ran to completion with `-tma_real_base_addr_enable 1` (+ debug on), **no assert/crash**, size cross-check passed, base map loaded (`base_records=33`). Real-base coverage is exactly as designed:

| kernel | uid | real_base coverage | L2_TMA_true_hit_rate | HW target |
|---|---|---|---|---|
| fwd K5 | 3 | UTMALDG 9 / UTMASTG 1 / UTMAPF 1 = 100%; synthetic = 0 | **0.9972** | ~0.70 |
| bwd K10 | 8 | UTMALDG 10 / UTMASTG 2 = 100%; UBLKRED 6 = synthetic (operand_addressed) | **0.9785** | ~0.82 |

**Base mapping is correct** (asserts would have caught any gap). The hit-rate over-estimation is the **predicted M1 limitation** (§ "locality over-estimation"): no coord offset yet.

**Root cause confirmed by first-request address analysis (both kernels):** real-base descriptor tensors collapse to **one address per tensor** because `agu_base = global_base + agu_index*128` has no tile coordinate (`agu_index` is the 0..requests_total-1 index *within one transfer*, identical across transfers).
- fwd K5: 5112 UTMALDG transfers → **2 distinct addresses** (K base, V base). The whole 6 MB tensor's 384 tiles collapse to the first 16 KB tile → 383 tiles' cold misses vanish → 0.9972.
- bwd K10: family=0 (UTMALDG) 7293 transfers → **3 distinct addr**; family=2 (UTMASTG) 768 → **2 distinct addr**; UBLKRED (synthetic, `transfer_uid<<20`) → **~226 distinct addr**. Ironically the synthetic UBLKRED *scatters* and creates the misses that pull bwd (0.9785) below fwd, while the real-base UTMALDG over-hits from tile collapse.

**Key sizing fact (decides M2):** a K/V tensor is `global_dim=[64,2048,24,1]`, tile `box=[64,128]`, element 2B → tensor = **6 MB**, **384 tiles** of 16 KB (128 lines each), 49152 lines total. L2 = **50 MB ≫ 6 MB**, so once loaded the whole tensor stays resident; repeated-tile revisits *should* hit. The missing misses are the **cold miss of first-touch across the 384 distinct tiles**, which the collapse erases.

## M2 chosen approach — A: visit-counter tile spread (decided, this is what we implement)

We evaluated two ways to add coords without a common trace field (coords are NOT in the trace, §0):
- **A (chosen): visit-counter tile spread.** Keep a per-tensor (keyed by `global_base`) monotonic visit counter in the TMA unit. For each descriptor transfer compute `tile_idx = counter % num_tiles`, where `num_tiles = ⌈tensor_bytes / tile_bytes⌉` from the descriptor (`global_dim`·element_size / `box_dim`·element_size). Address becomes `agu_base = global_base + tile_idx*tile_bytes + agu_index*128`. This spreads transfers across all `num_tiles` tiles → the cold-miss of first-touch is restored (≈ `num_tiles` cold misses per tensor), while repeated visits to the same tile still hit. Deterministic; needs **no tracer change, no trace regen** — simulator-only, flag-gated.
- **B (rejected for now): stride-based coord reconstruction.** Map `cta_x` to the seq axis and the inner-loop iteration to another axis, then `addr = base + Σ coord·stride`. More structural but requires guessing the FA3 tile scheduler's CTA→tile assignment, which is not in the trace and easy to get wrong.

**Why A is sound:** what L2 realism depends on is *how many distinct tiles are touched and revisited*, not the exact HW tile order. A produces the correct count of distinct tiles (`num_tiles`) and preserves same-tile reuse. The `% num_tiles` wrap is a deterministic approximation of the real schedule, not the real schedule.

**A — design details:**
- Only applies to descriptor sites with a real base (`has_real_base`, i.e. UTMALDG/UTMASTG/UTMAPF/UTMAREDG). In M2 UBLKRED/UBLKCP keep the synthetic path (operand_addressed); **M2.5 gives them their own real base + mock tiling under a separate flag (§M2.5)** — the same tile-spread mechanism, just with `tile_bytes=covered_bytes` and a raw-pointer base instead of the descriptor box.
- `num_tiles` and `tile_bytes` are derived from the base-map descriptor already carried in `TMACommand` (box_dim, element_size, global_dim). Carry them into `TMACommand`/`TMABaseRecord` if not already present.
- The visit counter is per `global_base` (per tensor), stored in the `tma_unit_sm` instance (a `std::unordered_map<uint64_t,uint64_t>`). Incremented once per enqueued descriptor transfer, read in `build_tma_command` (or at enqueue) so `tile_idx` is fixed per transfer and carried to the mover in `TMACommand`.
- Gate behind the existing `-tma_real_base_addr_enable` (M2 is the coord half of the same feature). Optionally add `-tma_real_coords_enable` if we want base-only vs base+coords A/B; default to combined.
- Verification: rerun fwd K5 + bwd K10. Expect `L2_TMA_true_hit_rate` to drop from ~0.99 toward the HW direction (fwd 0.70 / bwd 0.82) as the 384-tile cold misses reappear. Judge by direction, not exact match (the wrap is an approximation).

### M2.5 — UBLKRED/UBLKCP **real base + mock tiling** (CHOSEN, verified on the local trace)

> **Scope confirmed: BOTH halves ship together as one feature.** (1) **real base** — anchor UBLKRED/UBLKCP at their true GMEM base; (2) **mock tiling** — reuse the M2 visit-counter for the per-transfer tile offset. Neither alone is enough: real base without tiling collapses all transfers onto the tensor's first tile (over-hit); tiling without real base cannot model cross-op L2 residency. Both are gated behind the single new flag `-tma_operand_addr_tiling_enable` (default off).

**Decision (user):** give UBLKRED/UBLKCP the **real GMEM base** (so they live in the *same 64-bit address space* as the descriptor tensors) and reuse the **M2 visit-counter mock tiling** for the per-transfer tile offset. This **supersedes the M1 synthetic-base** treatment for these two ops. Base and size are both exact; only the *tile visitation order* is a deterministic approximation — identical in spirit to how UTMALDG already does M2.

**Why real base (directly answers "what if a prior UTMALDG already put that line in L2?"):** an isolated/synthetic base cannot model cross-op L2 residency — dQaccum lines a prior op left resident would *falsely miss* (the mirror image of the synthetic-hotspot false-hit). Anchoring UBLKRED/UBLKCP at their **true** base puts them in the descriptor tensors' coordinate system, so cross-op reuse is decided **structurally by the L2 model**, never assumed in code. A synthetic "band" was rejected for exactly this reason.

**The base is a raw kernel-argument GMEM pointer (NOT a tensormap), and it is already in the trace.** UBLKRED/UBLKCP desc is a `UMOV` bare handle; the real dst/src is a raw pointer passed **by value in the params struct**, which is already dumped to `launch_param_blobs/*.bin`. So the base is read **offline from an existing artifact** — no device deref, no crash risk (kills the old "static mapping impossible" blocker: we never needed a tensormap, just the raw pointer).

**Local-trace verification** (`flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24`, bwd uid8):

*UBLKRED.G.S.ADD.F32.RN* — executed pcs 0x90a0/0x91d0/0x94e0/0x95b0/0x96f0/0x97c0:

| operand (callback_index) | role | value | distinct |
|---|---|---|---|
| cb=0 | SMEM source cursor | 0x8400 / 0xc400 | 1 |
| **cb=1** | **GMEM dst = dQaccum** | low32 0x61400000..0x61ffc000 (`value_hi=0x85` not merged) | 768 |
| cb=2 | span (covered_bytes) | 0x400 → ×16 = **16384B** | 1 |

- **base = `0x7fd661400000`** — read directly from `launch_param_blobs/k10_uid8_arg0.bin` at struct offset **0x40** (a GMEM pointer whose low32 `0x61400000` == the floor of the runtime cb=1 addresses).
- **containment 6528/6528**: every runtime cb=1 addr ∈ `[base, base+12.58MB)`. tensor = `[1,2048,24,64]·fp32` = 12.58MB.
- **num_tiles = 768** = 12.58MB / 16384B = distinct-address count. Three independent derivations agree.

*UBLKCP.S.G* — executed, direction GMEM→SMEM (load). **Same mechanism, two differences:** (1) operand order is reversed — **cb=0 = GMEM addr** (distinct), cb=1 = SMEM cursor, cb=2 = span (0x20 → ×16 = **512B**); (2) it is **multi-region per pc** — most pcs target a scratch base `0x7fd533e00000`, while pc 0x4f0 targets the dQaccum base `0x7fd661400000`. So UBLKCP needs **per-pc base resolution** (match each pc's runtime GMEM floor to a param-blob pointer), not one base for the whole op.

**Cross-op overlap check (decisive evidence for the user's question):** dQaccum `[0x7fd661400000, +12.58MB)` vs **every** descriptor tensor (K/V/dK/dV/Q, from the base map + encode dump) → **overlap = 0**, nearest gap 8.39MB. So in THIS kernel cross-op reuse is genuinely zero; real-base anchoring reproduces that *structurally* (and would automatically capture overlap in a different config) instead of hard-coding "no overlap".

**Zero trace changes required** (answering the user directly). Everything needed is already present in the current trace, and **both files regenerate automatically on every trace run** — they are gated only by `enable_tma_desc`, whose **default is 1** ([tracer_tool.cu:1636](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu#L1636) `GET_VAR_INT(enable_tma_desc,"ENABLE_TMA_DESC",1,...)`). So no env is needed (only an explicit `ENABLE_TMA_DESC=0` would suppress them):
- **real base** → `launch_param_blobs/*.bin` + `tma_launch_param_dump.csv`, written per `cuLaunchKernel` at [tracer_tool.cu:581-602](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu#L581) / [:2353](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu#L2353) (guard `if(!enable_tma_desc) return;`).
- **per-transfer GMEM offset** (used **offline only**, to derive `num_tiles`) → `tma_runtime_operand_debug.jsonl`, written at [tracer_tool.cu:884](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu#L884) (same guard).
- **size** → operand-3 `covered_bytes` (already reaches the sim).

No proto change, no coord capture (mock tiling *replaces* real coords), no GPU re-run, no tracer source change. Only `build_tma_pc_base_map.py` (offline) and the C++ base-map consumer change.

**Caveat — `blob_path` is an absolute host path.** `tma_launch_param_dump.csv` records `blob_path` as the server's absolute path (`/home/jihyun/…`), so on any other machine `exists=False` (same bug fixed in §2.29). The offline emit must **not** trust `blob_path`; reconstruct it from the current `extra_info` as `extra/"launch_param_blobs"/Path(raw).name` (the pattern `build_tma_pc_base_map.py` already uses).

**Rebuild note:** the `ENABLE_TMA_DESC` default-1 lives in `tracer_tool.cu`, so the server's `tracer_tool.so` must be built from that revision for env-free emission; passing `ENABLE_TMA_DESC=1` explicitly is harmless insurance.

**Implementation (offline + C++):**
- *Offline* (`build_tma_pc_base_map.py`): add a UBLKRED/UBLKCP emit path. For each executed `(uid,pc)`: pick the GMEM operand (**UBLKRED cb=1, UBLKCP cb=0**), take its low32 floor, match to a `launch_param_blobs` GMEM pointer by containment, emit `{base_hex, num_tiles, tile_bytes=covered_bytes, raw_pointer_addressed:true, operand_addressed:true}`. `num_tiles = ⌈region_bytes/tile_bytes⌉`, cross-checked against the distinct-address count.
- *C++* (`tma_types.h`, `tma_unit_sm.cc`): add `raw_pointer_addressed` + `num_tiles` to `TMABaseRecord`. In `build_tma_command`, for a raw-pointer record: set `global_base`/`has_real_base`, **bypass the descriptor size cross-check** (box_dim is 0 → would FATAL at [:463-484](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L463) and also skip the tile-spread block), and compute the M2 tile offset with `tile_bytes=covered_bytes`, `num_tiles=record.num_tiles`. The mover ([:754](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L754)) is unchanged (already `global_base + tile_offset_bytes + agu_index*128`). Gate behind a new `-tma_operand_addr_tiling_enable` (default off) so M1/M2 stay bit-identical.

**[WARN] operand-address read API** — only relevant if the base is ever read at sim-time instead of offline: `get_addr(0)` asserts on `m_per_scalar_thread_valid`; guard with `get_per_scalar_thread_valid()` + `get_first_addr_valid()`, and note memref order is opcode-dependent (measured: UBLKRED GMEM=cb=1, UBLKCP GMEM=cb=0). The chosen design reads the base **offline** from the param blob, so the sim never touches these APIs.

## Next steps (M2+)
Address realism (not runnability — the five ops already run). Moved here from `TMA_ISA.md` so the plan stays canonical:
- **M2 coords** — spread each tensor's transfers across its tiles so the L2 hit-rate stops over-counting (base-only collapses all 384 tiles to one address). **Implemented as visit-counter tile spread (approach A, above).** Next: verify hit-rate direction toward HW (fwd 0.70 / bwd 0.82) on server rebuild.
- **M2.5 UBLKRED/UBLKCP real base + mock tiling** — anchor at the true GMEM base (read **offline** from `launch_param_blobs/*.bin`, no trace change) + reuse the M2 visit-counter tiling with `tile_bytes=covered_bytes`, `num_tiles=region/tile`. Verified on the local trace: UBLKRED base `0x7fd661400000` / num_tiles 768; UBLKCP per-pc base (scratch + dQaccum). Cross-op overlap with descriptor tensors = 0, so real-base anchoring reproduces L2 residency structurally (§M2.5).
- **M3 heuristic removal** — remove the legacy `handle_hi → config_id` heuristic entirely; base + all 11 descriptor fields + size now come from the exact base map (§M3).
- **M4 flag promotion** — after M1/M2/M2.5/M3 verified, drop `-tma_real_base_addr_enable` / `-tma_operand_addr_tiling_enable` from the config and make real base + mock tiling the default (delete the flags/gates). The base map is then the sole, always-on address source (§M4).

### M3 — remove heuristics (after M1 verified)
- Python: delete the resolver heuristic functions in `build_tma_descriptor_mapping.py` (`derive_handle_family_map_by_rank`, etc.) and the `tma_descriptor_resolver.json` write; keep only config emission.
- C++: remove handle_hi from `TMADescriptorLookupKey`; delete `load_tma_descriptor_resolver` ([gpu-sim.cc:392](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L392)); drop the handle_hi arg from `lookup_tma_site_metadata`; remove the `inst.tma_handle_hi` path ([tma_unit_sm.cc:370](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L370), [trace_driven.cc:399](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven/trace_driven.cc#L399)). **Since size has moved to the base map** (replace the config_id-based total_bytes/requests_total in [build_tma_command](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L384) with base-map-based), the config_id dependency can be removed.
- Verification: rebuild; run FA3 fwd/bwd with the flag on; confirm that no descriptor site previously bound becomes unbound vs the `.bak` baseline, and that requests_total is identical.

### M4 — promote flags to default + drop from config (after M1/M2/M2.5/M3 all verified)
Once the real-base + mock-tiling path is the ONLY correct behavior (heuristics deleted in M3, L2 hit rate confirmed toward HW fwd ~0.70 / bwd ~0.82), the two opt-in flags become vestigial and should be retired so the feature is always on. Do this **only after** all prior milestones pass — until then the flags are needed for A/B baselines and for not crashing non-FA3 traces.
- **Remove from config**: delete `-tma_real_base_addr_enable 1` and `-tma_operand_addr_tiling_enable 1` from [gpgpusim.config](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config#L377) (no longer opt-in per run).
- **Make default in code**: either flip the `option_parser_register` default from `"0"` to `"1"` for both, or (cleaner) delete both flags entirely and remove the `if (m_config->...)` gates in `build_tma_command` so real base + mock tiling run unconditionally. Deleting is preferred once heuristics are gone — a dead flag that can only be 1 is noise.
- **Prerequisite guard**: real base asserts on base-map load failure, so before removing the flag confirm the trace pipeline ALWAYS emits `tma_pc_base_map.json` (it does — `run_hw_trace.py` calls `build_tma_pc_base_map.py` in post-processing whenever the inputs exist). If a non-FA3 trace could legitimately lack a base map, keep a load-success guard (skip real base if the map is empty) instead of asserting.
- **Heuristic deletion is a hard prerequisite** (already M3): `handle_hi → config_id` resolver, `tma_descriptor_resolver.json`, and the config_id size path must all be gone, so the base map is the sole source of base + size + descriptor fields. Only then is "always on" unambiguous.

## Pre-implementation audit — risks/open items (must be reflected)

Problems found via a simulator source audit + real trace data. **Some earlier conclusions are corrected.**

### [BLOCKER→RESOLVED in M2.5] UBLKRED/UBLKCP have no complete GMEM base in the operand — correcting the "use the operand address directly" conclusion
> **Resolution:** the operand only keeps low-32 (below), so the operand *value* alone can't be the address — but the **full 64-bit base is read offline from the by-value param struct** (`launch_param_blobs/*.bin`), and the low-32 operand offset is used only to pick `num_tiles`. So this is not a blocker for M2.5's "real base + mock tiling" (§M2.5). The facts below stand as the reason the operand value is not used directly.

Data re-check (runtime jsonl):
- UBLKRED ci=1 = `0xDE08000` (~222MB), with `value_hi=133(0x85)` **not merged** into `first_lane_addr`. The earlier `0x850d40xxxx` was a case where hi happened to be merged; in general only the low 32 bits remain in fla (max non-neg = `0xdffc000`).
- UBLKCP ci=0 = `0xffffffff...` (SMEM neg), ci=1 = small. **No `0x7f...` GMEM base in any operand.** (The 64-bit base is recovered from the param blob, not the operand — §M2.5.)
- Yet both UBLKRED/UBLKCP carry `desc_value_hi=0x14f00000` (handle) → they are descriptor-referencing ops (UBLKRED discovery `desc_refs=[16]`).

**Corrected conclusion (finalized via SASS + executed trace):** UBLKRED is **not a tensormap descriptor op.** The executed sites (0x90a0/0x91d0/0x94e0/0x95b0/0x96f0/0x97c0) set `desc[UR16]` via `UMOV UR16,0x0`+`UMOV UR17,0x14f00000` (a bare handle with no base, a constant shared with UTMALDG). dst is a raw dQaccum GMEM pointer (`c[0x0][0x280]/0x2a0`) + dynamic tile offset. (The 0x61d0 I analyzed earlier is **dead code**, executed=False.) Therefore:
1. **M1 excludes real-base for UBLKRED/UBLKCP, keeps synthetic fallback** (even with the flag on). base-map targets are only UTMALDG/UTMASTG/UTMACCTL.PF.
2. **The extractor's UBLKRED offset (0x30d96 etc.) is a false positive**: `build_defs()` doesn't model UMOV → skips the live `UMOV UR16,0x0`, walks a stale UIADD3, fabricates a value outside the struct (0x700). §M1 Part A-0 removes it via UMOV modeling + struct-bounds guard → UBLKRED honestly unresolved.
3. UBLKRED/UBLKCP's real base is **M2.5**: read the raw GMEM pointer **offline** from the by-value param struct (`launch_param_blobs/*.bin`), then apply **mock tiling** (M2 visit-counter, `tile_bytes=covered_bytes`, `num_tiles=region/tile`). A static *tensormap* mapping is impossible (0 in-struct slots resolved), but the raw pointer is a plain GMEM base and is already in the trace — **no per-tile coord capture needed** (mock tiling replaces coords). Verified on the local trace (§M2.5): UBLKRED base `0x7fd661400000`, num_tiles 768, containment 6528/6528.
4. **Size is unaffected**: `covered_bytes` (operand-3) is valid → data-request volume is exact. "How much moves" is correct; "where" is synthetic in M1 and **real in M2.5** (§M2.5).

### [WARN] operand-address read API caution (for M2.5)
`get_addr(0)` asserts on `m_per_scalar_thread_valid`. Always guard with `get_per_scalar_thread_valid()` + use `get_first_addr_valid()`. The memref[0] vs memref2 order is also opcode-dependent and unknown → measure before forcing.

### [BLOCKER for M3] config_id/descriptor path · Phase-2 assert removal order
- `config_id` is the current supplier of size (box_dim/element_size) ([tma_unit_sm.cc:374-387](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L374)). Do **not remove** it before the base map's size fields are cross-checked to match `requests_total`.
- The Phase-2 assert block ([:415-459](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L415)) enforces `has_descriptor_metadata`/`!config_id.empty()`/`total_bytes>0` → removing the resolver first **crashes every descriptor op**. Rewrite the asserts against the new basis only after base-map sizing is verified.

### [WARN] locality over-estimation (M1 limitation, intended)
M1 only does `base + agu_index*128` — **no coord (tile) offset** ([:637](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L637)). Different tiles of the same tensor collapse to the same address → L2 hit is over-stated. Correctness is safe (each mf still returns). Resolved in M2 (coords). Judge M1 only by "synthetic hotspot gone + HW direction."

### [OK] items confirmed safe
- Response matching uses `transfer_uid` (an mf-pointer key), independent of agu_base ([:736](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L736), [:898-911](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L898)) — traffic count unchanged.
- pc encoding: `inst.pc == pc_hex` (base-0) confirmed. The existing resolver already works this way ([gpu-sim.cc:415](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L415)).
- The feature flag follows the `tma_debug_enable` pattern ([shader.h:2236](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h#L2236), [gpu-sim.cc:1890](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L1890)).

### M3 sites missed when removing handle_hi (audit addendum)
The `std::tuple<uint,uint64,uint32>` stats/log tuples in `tma_unit_sm.cc` where the 3rd element = handle_hi: [:165-167, :201-204, :210, :264-267]. Also fix `TMAResolvedSiteMetadata.handle_hi` ([tma_types.h:116](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h#L116)) and the `lookup_tma_site_metadata` signature ([gpu-sim.h:883-885](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h#L883)).

## Critical Files
- [build_tma_pc_base_map.py](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/build_tma_pc_base_map.py) — UTMACCTL.PF emit + full descriptor fields + strict gate + UBLKRED absence diagnostic; **M2.5: UBLKRED/UBLKCP raw-pointer base emit (offline param-blob match) + num_tiles**
- [extract_tma_descriptor_offsets.py](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/extract_tma_descriptor_offsets.py) — UMOV modeling + struct-bounds guard
- [run_hw_trace.py](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/run_hw_trace.py) — wire in the new scripts
- [tma_types.h](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_types.h) — base-map types + TMACommand carrier
- [gpu-sim.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc) — loader + parse_extra_trace_info
- [tma_unit_sm.cc](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc) — build_tma_command lookup + mover injection

Secondary: `build_tma_descriptor_mapping.py` (--configs-only), `build_tma_operand_mapping.py` (rewire), `trace_driven.cc`, `abstract_hardware_model.h`, `shader.h`.
