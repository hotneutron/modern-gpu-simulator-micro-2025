# TMA Real Base Address + Real Coords (Arch TODO-2, "same as HW")

> **Goal.** Make the TMA unit resolve the *actual* per-transfer GMEM address the
> way real Hopper HW does — `addr = global_base + Σ coord[d]·global_stride[d]` —
> instead of the synthetic `agu_base = (transfer_uid<<20) + agu_index*128`
> ([tma_unit_sm.cc:633-639](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L633-L639)).
> This is the true fix for FA3_progress **Arch TODO-2** and the prerequisite for
> TODO-1 (SMEM swizzle) and for Opt-6B address realism.
>
> **Also:** delete the trace-gen resolver *rescue heuristics* (exact-bind-or-fail),
> per `TMA_TRACING_revised.md` policy, taken to its strict end.

## 0.0 The problem in plain terms (why base recovery is hard)

Think of a TMA load as: "copy a tile of a big GPU tensor into shared memory." To
model it faithfully the simulator needs the tensor's **GPU start address (base)**.
That base lives in the `CUtensorMap` descriptor. The problem is a **two-world
identity gap**:

- **Host world (CPU, at `cuTensorMapEncodeTiled`):** we DO see the real base —
  `global_address_hex`. There are **7 distinct bases** in FA3. But here we only
  know "a tensor map was built," not "which `UTMALDG` instruction will use it."
- **Device world (GPU, at the executed `UTMALDG`):** we see the instruction and its
  operands, but **none of them contain the base.** Measured on FA3 (24,816 executed
  loads):
  - `desc[URx]` handle = `{lo=0, hi=desc_value_hi}`, and `desc_value_hi` takes only
    **3 values** (`0x14f00000`, `0x12f00000`, `0x10000000`). `lo=0` is not
    "predicate-off / not executed" — it is genuinely `0` because the handle's low
    word is a constant `0` (SASS `UMOV UR20, 0x0`). All 24,816 *executed* rows show
    `lo=0`.
  - Operand-1 and operand-2 are **shared-memory addresses** (e.g. `0xe800`), not
    GMEM. No base anywhere.

**So the core problem is a mapping gap, not a "missing value":**

```
   7 real bases (host, cuTensorMapEncodeTiled)          many UTMALDG sites (device)
   ┌───────────────────────────────┐                   ┌───────────────────────────┐
   │ base 0x7f3fc2c00000  box192    │        ???        │ pc 0x94e0  desc_hi 0x14f...│
   │ base 0x7f3fc3200000  box128    │  ◄─── no shared ──►│ pc 0x96b0  desc_hi 0x12f...│
   │ base 0x7f3fc3800000  box128    │       key         │ pc 0x9c00  desc_hi 0x14f...│
   │ ... (7 total) ...              │                   │ ... (many, only 3 desc_hi) │
   └───────────────────────────────┘                   └───────────────────────────┘
```

- The device side has only **3 `desc_hi` values** to distinguish **7 bases** →
  **`desc_hi` is not unique; the same `desc_hi` covers several different bases.**
  (Answering the user: yes — same `desc_value_hi`, different GPU base.)
- Worse, `desc_hi` does not even appear in the host encode dump (the blob's
  qwords are all `0x3f00...`, no `0x14f...`), so there is **no common field** to
  join host↔device at all.

That is the whole difficulty: the real base exists (host side), the instruction
exists (device side), but there is **no reliable key that ties one executed
`UTMALDG` to one of the 7 host bases**. Every prior attempt failed on this exact
join. §2.9 gives the only join that actually works (host launch-argument order).

## 0.05 Key concepts (plain-language glossary)

Before the mechanics, the five recurring terms, by analogy to an **apartment
complex**:

- **kernel argument** — the values the CPU passes when launching the kernel
  (`kernel(Q, K, V, tensormap1, tensormap2, …)`). On the GPU they are copied into a
  read-only region called the **constant bank**.
- **param base pointer (`param_base`)** — CUDA packs all kernel arguments into one
  memory block and gives the kernel the block's **start address** = `param_base`.
  *Analogy: the complex's front-gate address.* SASS loads it with
  `ULDC.64 UR4, c[0x0][0x198]` (so `c[0x0][0x198]` is the signboard holding the gate
  address; `UR4` becomes the gate address itself).
- **tensormap (= TMA descriptor)** — the **128-byte struct** built by
  `cuTensorMapEncodeTiled`, containing the tensor's **base (GPU start address),
  shape, strides, swizzle**. The TMA hardware reads it to know what to fetch.
  *Analogy: the info board inside each building.* In FA3 this 128B struct is passed
  **by value** as a kernel argument, so it physically sits inside the param block.
- **static offset** — the **fixed distance** from `param_base` to each tensormap,
  fixed at compile time. *Analogy: "building 1 is +0x1f0 from the gate, building 2
  is +0x2b0…".* SASS: `UIADD3 UR6, UR4, 0x1f0` computes
  `tensormap1_addr = param_base + 0x1f0`.
- **def-chain** — tracing backward in SASS to see where a value came from: a
  `UTMALDG` operand ← the `UIADD3` that built it ← the root `ULDC c[0x0][0x198]`.
  Fixed per pc (instruction location), so recoverable statically.

Putting it together (the sentence that was unclear before):

```
[GPU constant bank]
param_base ─────────────► value read by  ULDC c[0x0][0x198]   (the "front gate")
   │
   ├─ +0x1f0 ─► [ tensormap #1 : 128B ]  qword0 = base 0x7f3fc2c00000, strides…
   ├─ +0x2b0 ─► [ tensormap #2 : 128B ]  qword0 = base 0x7f3fc3200000, strides…
   ├─ +0x370 ─► [ tensormap #3 : 128B ]  qword0 = base 0x7f3fc3800000, …
   └─ +0x7f0 ─► [ tensormap #4 : 128B ]  …
```

- `c[0x0][0x198]` is **not** the descriptor; it is the **param base pointer** (the
  gate). Each tensormap lives at a **distinct static offset** from it.
- Why that offset is the missing **per-tensor key:** `desc_hi` had only 3 values for
  7 tensors, so it could not tell them apart. But each tensor sits at a **unique
  offset** in the param block, so "this `UTMALDG` reads +0x2b0" ⇒ "tensor #2",
  uniquely. Which offset a given `UTMALDG` pc uses is **fixed per pc** (from the
  def-chain), so it does not vary at runtime.
- Final base recovery: `param_base` (captured once at runtime) `+ static_offset`
  (fixed per pc) = that tensor's descriptor address → read the **128B** there →
  **qword0 = the true GPU base**.

## 0.1 What is `desc_value_hi` then? (it is NOT the descriptor)

`desc_value_hi` is often mistaken for "the TMA descriptor." It is not. Traced end
to end:

- **Where it is read:** the device instrument reads the `desc[URx]` uniform-register
  pair and stores lo/hi into `ureg_desc_value` / `ureg_desc_value_hi`
  ([inject_funcs.cu:79-87](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/inject_funcs.cu#L79-L87)).
  So `desc_value_hi` = the **upper 32 bits of the `desc[URx]` register pair**.
- **What sets it (SASS):** `UMOV UR20, 0x0` + `UMOV UR21, 0x14f00000` (§2.5). Both
  halves are **compile-time immediates**. `lo` is a constant `0`; `hi` is one of
  three constants: `0x14f00000`, `0x12f00000`, `0x10000000`.
- **It is a per-argument "tensor-map slot tag", not an address.** It is the
  compiler's normalized handle for *which by-value `CUtensorMap` kernel argument*
  this site uses — a coarse tag, nothing more.
- **It does not identify a tensor, let alone a base:**
  - only **3 values** exist, but there are **7 GPU bases** → cannot be unique;
  - one value (`0x14f00000`) is shared by **16 pcs** spanning multiple tensors;
  - `0x10000000` **and** `0x14f00000` both map to the *same* box family
    (`box_64x192`), so it is not even 1:1 with the box family;
  - it never appears in the host encode blob (all qwords `0x3f00…`), so it cannot
    be joined to `global_address_hex` at all.

Analogy: `desc_value_hi` is like the **first two digits of a ZIP code** — it hints
at a rough region (box family), but never the street address (GPU base), and it is
not even printed on the shipping record (host encode dump), so it cannot be
cross-checked. This is why keying the resolver on it leaves 50/82 sites unresolved
(§2.8).

## 0.2 How real NVIDIA HW knows the base, and two recovery paths

The user's intuition is **correct**: HW reads the descriptor to get the base. The
nuance is *where* the descriptor lives (confirmed vs NVIDIA/CUTLASS docs,
`cuTensorMapEncodeTiled`, and FA3 SASS):

1. Host builds a **128B `CUtensorMap`** (`cuTensorMapEncodeTiled(&map, …,
   globalAddress, …)`); **qword0 of the blob = tensor GMEM base**.
2. It is passed **by value** as a `__grid_constant__ CUtensorMap` argument, so the
   128B struct is **copied into the constant/param block** (`c[0x0][…]`).
3. On device, `UTMALDG` / `cp.async.bulk.tensor` passes only a **pointer to that
   descriptor + coords**; the **TMA HW unit reads the 128B descriptor** and computes
   `addr = base + Σ coord·stride`. Threads never compute the address.

**Key reversal:** the base *is* on the device, inside the **128B descriptor in the
param block**. It looked absent only because the tracer reads the `desc[URx]`
**handle** (`{0, 0x14f00000}`) and never dereferences the descriptor bytes.

Two paths; per the user, **B first**:
- **Path A (host launch-arg):** dump the by-value `CUtensorMap` args at
  `cuLaunchKernel`, match bytes to an encoded blob. Robust, currently unimplemented.
- **Path B (device param dereference; try first):** recover per-site the descriptor
  address `param_base + static_offset`, read the 128B there, take qword0 as base.
  Closest to HW; needs no host change if feasible. Verified in §2.10.

## 0. The question this answers

> "Make this the same as what HW does: find the base address of the TMA
> descriptor, map it correctly when the TMA descriptor access is needed. I assume
> the actual data base address exists somewhere in the TMA descriptor. Am I
> correct?"

**Yes — confirmed by the trace artifacts.** On Hopper a `UTMALDG` names a
`CUtensorMap` descriptor; the TMA engine reads qword0 of that 128B blob as the
tensor **global base address**, then adds the per-transfer tile coordinates
scaled by the descriptor strides. Both halves are visible in the trace:

- **Base address is captured.** `tensor_map_encode_dump.csv` (host-side
  `cuTensorMapEncodeTiled` hook) records `global_address_hex`, and it is exactly
  `qword0_hex` of the encoded blob. FA3 rows:
  - dump 0: `global_address_hex=0x7f3fc2c00000` == `qword0_hex=0x00007f3fc2c00000`
  - dump 1: `global_address_hex=0x7f3fc3200000` == `qword0_hex=0x00007f3fc3200000`
  So the base the HW reads *is* in the trace already; the current config pipeline
  simply **drops it** (`TMA_DESC_FORMAT.md` Layer 1: "does not require
  `global_address_hex` unless later address-sensitive modeling is required").
- **Strides are captured** (`global_strides = [3072,128,6291456]`) and already
  reach the sim via the config.
- **Per-transfer coordinates are NOT captured** — this is the one genuinely
  missing ingredient (see §2).

## 1. Why the previous attempt hit a wall (the "resolver key problem")

The last attempt tried to attach the base to the resolver key
`(unique_function_id, pc_hex, handle_hi)` → `config_id`. That cannot work, for
two independent reasons grounded in the trace:

1. **`config_id` is many-to-one over base addresses.** In the FA3 dump, the single
   config `tm_r4_dt9_box_64x128x1x1` is produced by dump rows 1,2,3,4,6,8,9,10,11
   — **nine distinct `global_address_hex` values**. `config_id → base` is not a
   function, so binding base to `config_id` (the Opt-6B "mock base" flaw) is
   fundamentally lossy and inflates L2 reuse.
2. **The runtime disambiguator is zeroed / non-unique.** The consumer capture
   `tma_desc_runtime_debug.csv` records `desc_value_lo=0, desc_value_hi=0,
   first_lane_addr=0`. `handle_hi` (the key field) is only a coarse *box family*
   signal — several distinct GPU base tensors share one `handle_hi`.

**Conclusion (corrected — see §2.5/§2.6):** the base is **not observable on the GPU
side at all.** The `desc[URx]` operand is a constant `{0, handle_hi}` handle, and
the `UTMALDG` operand-2 pointer only points at the kernel-param buffer (the by-value
`CUtensorMap` copy) — its *pointer value* is a param VA, not a GPU base. The GPU
base address exists in exactly **one** place: the host-side `cuTensorMapEncodeTiled`
hook (`globalAddress`), already dumped as `global_address_hex`. So the real problem
is **not "find the base"** (it is already captured) but **"uniquely map each
executed TMA site to the correct encode-dump row"** — and the current trace lacks a
GPU-observable key that is unique per tensor.

## 2. What HW does, and the exact data we still need to capture

HW `UTMALDG.4D [UR8], [UR18], desc[UR20]` (confirmed FA3 SASS form). Naive reading
of the operand roles is **wrong for FA3** — the spike in §2.5 corrects it:

| SASS operand | naive role | **actual role in FA3 (spike)** |
|---|---|---|
| `[UR8]` (operand 1) | SMEM destination | SMEM dst cursor (not needed for GMEM addr) |
| `[UR18]` (operand 2) | packed tile coords | **pointer to the tensormap struct in kernel-param/SMEM** (contains base+strides+coords) |
| `desc[UR20]` (operand 3) | descriptor → base+strides | `{lo=0, hi=handle_hi}` **constant handle only — carries NO base** |

HW address: `gmem_addr = global_base + Σ_d coord[d] * global_stride[d]`.

## 2.5 SPIKE RESULT (measured on FA3 fwd SASS) — the base lives in the operand-2 pointer, NOT in `desc[URx]`

Traced the real def chain of `UTMALDG.4D [UR8], [UR18], desc[UR20]` at
`pc 0x7e90` in `flash_fwd_hdim64_bf16_sm90.sm_90a.sass`:

```
0x0090  ULDC.64  UR4, c[0x0][0x198]      ; {UR4,UR5} = KERNEL PARAM BASE (const bank)
0x00a0  UIADD3   UR6, UP0, UR4, 0x1f0    ; UR6 = param_base + 0x1f0  (a tensormap ptr)
...
0x7da0  UIADD3   UR18, UP0, UR6, 0x2b0   ; [UR18:UR19] = (param_base+0x1f0) + 0x2b0  ← operand 2
0x7de0  UIADD3.X UR19, URZ, UR7, ...     ;   = pointer into the param/tensormap region
0x7e00  UMOV     UR20, 0x0               ; desc[UR20] lo = 0
0x7e20  UMOV     UR21, 0x14f00000        ; desc[UR21] hi = 0x14f00000  (== handle_hi)
0x7e90  UTMALDG.4D [UR8], [UR18], desc[UR20]
```

**Findings (this resolves the §2 open risk and rewrites the capture target):**

1. **`desc[URx]` is a constant `{0, handle_hi}`.** That is exactly why the trace
   shows `desc_value_lo=0`. The handle carries no address — so capturing the
   `desc[URx]` *value* (plan's old "(P)") would still yield `0/handle_hi` and would
   NOT recover a base. The previous attempt's wall is now fully explained.
2. **Operand 2 (`[UR18]`) is a pointer into the kernel-param buffer**, computed as
   `ULDC.64 c[0x0][0x198] (kernel param base) + const offsets`. This is the
   **by-value `CUtensorMap` argument** the CPU passed at `cuLaunchKernel`. Its
   *pointer value* is a param-region VA, **not a GPU base address**. (This confirms
   the user's point: the launch-argument path cannot expose a GPU base as a
   register/pointer value — the base is a *field inside* the by-value blob.)
3. **Where the GPU base actually is.** Two facts from the encode dump make this
   decisive:
   - `tensor_map_ptr_hex` is a **host CPU address** (`0x7ffda2b9c380`,
     `0x7f3db15f8240` — stack/heap VAs) and is **reused across many tensors**
     (only 2 distinct ptrs for 12 encode rows). So it cannot pick one row.
   - `global_address_hex` (7 distinct GPU VAs) is the real base and comes
     **only** from the host `cuTensorMapEncodeTiled` hook (`p->globalAddress`,
     [tracer_tool.cu:578](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu#L578)).
   Dereferencing the operand-2 param VA at runtime would just re-read the same
   by-value blob the host already dumped — no new information, and it needs a
   GPU-side memory read that the tracer does not currently do for param space.
4. **So the capture target is NOT the base** (already have it) but a
   **GPU-observable unique key** that ties an executed site to one encode row. The
   only candidates are:
   - **(a) the operand-2 param VA (pointer value)** — distinguishes *which param
     slot / which tensormap argument* is used, if each tensor argument sits at a
     distinct param offset. This is the most promising GPU-side signal.
   - **(b) launch-argument by-value capture** (currently **unimplemented**;
     `nvbit_get_kernel_argument_sizes` is commented out at
     [tracer_tool.cu:2049](file:///home/jihyun/project/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu#L2049)).
     Dumping the param blob at launch (the `TMA_TRACING_revised.md`
     `kernel_launch_arg_dump.jsonl` idea) lets us match the launch-arg
     `CUtensorMap` bytes to an encoded blob, then map the param **offset** →
     tensor → base. Combined with (a) this is the exact-bind path.

**Revised feasibility gate:** the base is a solved input; the open question is
whether the **operand-2 param offset** (from the pointer VA) is unique per tensor
argument. That must be verified next (§2.7), not assumed.

## 2.6 Per-opcode descriptor-access differences (per `TMA_TRACING_revised.md`)

`TMA_TRACING_revised.md` (Opcode-Family Coverage) is explicit that each family
reaches its descriptor differently. The base-recovery mechanism must therefore be
**per family**, not one rule. Confirmed FA3 forms:

| Opcode | FA3 form | Where the descriptor / base comes from | Path B bind rule |
|---|---|---|---|
| `UTMALDG` | `[dst],[ptr],desc[URx]` | operand-2 is the runtime **descriptor VA** (`param_base + static_offset`); dump 128B there | dump descriptor at operand-2 VA; decode all fields from dumped 128B |
| `UTMASTG` | `[URa],[URb]` (**no `desc[]`**) | desc-like **first operand pair** is the runtime descriptor VA; `handle_hi=0` is irrelevant | dump descriptor at operand-1 pair VA; decode all fields from dumped 128B |
| `UBLKRED` (desc-backed) | `[d],[s],URn,desc[URx]` | explicit `desc[URx]` supplies the runtime descriptor VA; operand-3 remains span semantics only | dump descriptor at `desc[URx]` VA; keep operand-3 span separate |
| `UBLKCP` / `UBLKPF` | `[d],[s],URn` (**bulk, no desc**) | no tensormap descriptor; operand addresses are already the moved-region cursors | no descriptor bind; require operand metadata only |
| `UTMAPF` | `.L2.4D [URa],[URb]` (**no desc**) | no direct descriptor operand; must inherit descriptor identity through an exact link to a later consumer | exact forward link to a later bound `UTMALDG`/descriptor consumer |
| `UTMAREDG` | (not present in FA3 trace) | descriptor-backed like `UTMALDG` | same as `UTMALDG` when it appears |
| `UTMACCTL.PF` / `UTMACMDFLUSH` | control-only | no data-moving descriptor bind for the simulator address path | keep as control metadata only; no GMEM base generation |

Key consequence: every descriptor-carrying family reaches its tensormap through a
runtime **descriptor VA** (operand-2 for `UTMALDG`/`UTMAREDG`, operand-1 pair for
`UTMASTG`, `desc[URx]` for desc-backed `UBLKRED`). Under Path B the bind key is no
longer `handle_hi` or launch-order identity; it is the **descriptor VA itself**,
and the authority is the dumped **128B descriptor bytes** read at that VA.
`UBLKCP`/`UBLKPF` need no tensormap base; `UTMAPF` still inherits from its linked
consumer.

### 2.6.1 Coverage Status Against `TMA_TRACING_revised.md`

The document-level design now covers **all opcode families named in**
`TMA_TRACING_revised.md`, but the **current code implementation does not yet fully
cover all of them**. This distinction must remain explicit.

| Family | Required by `TMA_TRACING_revised.md` | Path B design status | Current code status | Remaining work |
|---|---|---|---|---|
| `UTMALDG` | direct descriptor-backed family | **covered** | **partially implemented** | runtime 128B dump path added; still need end-to-end trace proof + coord/stride address generation in simulator |
| `UTMAREDG` | direct descriptor-backed family | **covered** | **partially implemented** | same Path B rule as `UTMALDG`; needs runtime proof because FA3 sample did not exercise it |
| `UTMASTG` | indirect / desc-like descriptor family | **covered** | **partially implemented** | first operand pair must be treated as descriptor VA everywhere; needs runtime proof end-to-end |
| descriptor-backed `UBLKRED` | direct descriptor-backed family | **covered** | **not complete yet** | must keep descriptor semantics separate from operand-3 span and dump/decode the descriptor VA path |
| non-descriptor `UBLKRED` | operand-only family split | **covered** | **not complete yet** | must stay operand-metadata-only and not be forced through descriptor binding |
| `UTMAPF` | exact linked-descriptor path | **covered** | **not complete yet** | must inherit descriptor identity only through exact forward link to a later bound consumer |
| `UBLKCP` | operand/control-only family | **covered** | **not complete yet** | no descriptor bind needed, but operand metadata path still needs explicit completion/safety checks |
| `UBLKPF` | operand/control-only family | **covered** | **not complete yet** | same as `UBLKCP`; operand-only, not descriptor-bound |
| `UTMACCTL.PF` | control family | **covered** | **not complete yet** | control metadata path must be explicit; no GMEM base generation |
| `UTMACMDFLUSH` | control family | **covered** | **not complete yet** | keep as control/wait semantics only, with explicit metadata path |

So the implementation status right now is:

1. **Design coverage:** all families from `TMA_TRACING_revised.md` are now covered in
   the document.
2. **Code coverage:** only the `UTMALDG` / `UTMAREDG` / `UTMASTG` Path B scaffold is
   partially in place today.
3. **Still missing before we can say "all ops are handled":**
   descriptor-backed vs non-descriptor `UBLKRED` split, `UTMAPF` exact-link rule,
   and explicit operand/control completion for `UBLKCP` / `UBLKPF` /
   `UTMACCTL.PF` / `UTMACMDFLUSH`.

## 2.7 Next verification (before any code) — is the param offset unique per tensor?

The whole exact-bind plan hinges on one unverified assumption from §2.5-4(a): that
the **operand's param offset** (e.g. `UTMALDG` operand-2 VA = `param_base + K`)
differs per distinct tensor argument, so it can select one encode row. Check:

1. For every executed `UTMALDG`/`UTMASTG` site, extract its operand param offset
   `K` from the SASS def chain (`ULDC c[0x0][off]` + `UIADD3` const adds).
2. Count distinct `K` values vs distinct `global_address_hex` (7). If `K` is
   1:1 (or finer) with tensors, offset+launch-arg is sufficient. If several
   tensors collapse to one `K` (e.g. a loop reuses one param slot as a cursor),
   then the param offset is NOT unique and we must fall to launch-arg **value**
   capture (2.5-4(b)) or accept a coarser model.
3. Also confirm whether the launch-arg buffer actually contains the `CUtensorMap`
   **by value** (128B blobs) vs by pointer — dump `p->kernelParams` at
   `cuLaunchKernel` and compare bytes to `tensor_map_encode_blobs/*.bin`.

This §2.7 SASS/param study is the real "spike 2" and must complete before touching
`tracer_tool.cu`.

## 2.8 SPIKE 2 RESULT (measured on FA3 k5) — no GPU-observable base signal exists

Ran the runtime-operand and resolver analysis on the FA3 kernel-5 trace. Result is
**negative for the operand-pointer approach**, which confirms the user's position.

**(a) Both `UTMALDG` operands are SMEM addresses, not GMEM.** For every executed
`UTMALDG` site (`tma_runtime_operand_debug.jsonl`), the two `MEMORY_REF` operands
resolve to small **shared-memory offsets**, not global VAs:

| pc | operand-1 (ci=0) | operand-2 (ci=1) distinct values | desc_value_hi |
|---|---|---|---|
| 0x94e0 | 1 SMEM addr | `{0xe800, 0x12800}` (SMEM) | `0x14f00000` |
| 0x96b0 | 1 | `{0x8800}` | `0x12f00000` |
| 0x9c00 | 1 | `{0x800, 0x4800}` | `0x14f00000` |
| 0xa240 | 1 | `{0x10400}` | `0x14f00000` |
| … (20 pcs total) | 1 | 1–2 SMEM offsets | one handle each |

So operand-2 is a **destination/staging SMEM cursor**, not a param VA pointing at a
tensormap. The def chain in §2.5 (`UR18 = param_base + off`) was a *different*
site/kernel; at the actually-executed load sites the operand value is SMEM. **No
GMEM base or param-tensormap pointer is observable in any operand at runtime.**

**(b) `desc_value_hi` (handle_hi) is the only tensor signal, and it is not unique.**
Every site exposes only `desc_value_lo=0, desc_value_hi=handle_hi`. Multiple pcs
share the same `handle_hi` (`0x14f00000` appears on ~12 pcs), and one `handle_hi`
maps to a whole box family, which in turn maps to **7 GMEM bases**.

**(c) The resolver already fails at the family level, before base even matters.**
`tma_descriptor_resolver.json` (82 executed descriptor sites):
- **50 / 82 `unresolved`**, each with **3 candidate `config_id`s** and
  `confidence=low` — handle_hi cannot even pick the box family for the majority.
- Only 32 resolved, and those lean on exactly the rescue heuristics we intended to
  delete (`single_rank_candidate_config`, `same_function_desc_reg_config_reuse`,
  `..._inferred_rank_from_handle_reuse`, `..._from_desc_reg_reuse`).

**Conclusion of spike 2 — historical, now superseded by spike 3.** At this point we
were still missing the crucial fact that the descriptor itself sits at
`param_base + static_offset` and can be dereferenced. So the "operand/param route is
a dead end" conclusion here should now be read only as a record of the failed
`desc_hi`/operand interpretation, not as the final design.

## 2.9 Decision — Path B is the design, and it must become the only authority

Spike 3 overturns the old "GPU-side is impossible" conclusion above. The correct
decision is now:

1. **Adopt Path B as the primary and only binding path.** For every executed
   descriptor-carrying TMA site, recover the site's **descriptor VA**
   (`param_base + static_offset`) and dump the **full 128B descriptor bytes** from
   that VA.
2. **Use the dumped 128B as the source of truth for *all* descriptor fields.** Not
   only `global_base` (qword0), but also the values needed to recover rank,
   `global_dim`, `global_strides`, `box_dim`, `element_strides`, `swizzle`,
   `interleave`, `l2_promotion`, and `oob_fill` must come from this same dumped
   descriptor path.
3. **Demote host encode dumps to validation only.** `tensor_map_encode_dump.csv` /
   `tensor_map_encode_blobs/*.bin` remain useful as a defense-in-depth cross-check,
   but they are no longer the binding authority. The binding authority is the
   runtime-visible device descriptor that the HW actually dereferences.
4. **Delete the old resolver heuristics completely.** Once the per-site 128B blob is
   dumped, there is no remaining need for `handle_hi` family matching, rank reuse,
   desc-reg reuse, or candidate selection.
5. **Fail hard on mapping problems.** If an executed descriptor-carrying site does
   not produce exactly one well-formed 128B descriptor and exactly one decoded
   descriptor record, the pipeline must `assert`/abort immediately. No unresolved
   entries, no low-confidence binding, no synthetic fallback.

So the new contract is simple:

`executed site -> descriptor VA -> dumped 128B -> decoded descriptor -> simulator`

and every old side path becomes either validation-only or dead code.

## 2.10 SPIKE 3 RESULT (Path B) — the descriptor address IS recoverable on device

Measured on FA3 fwd SASS + producer/runtime traces. **Path B is structurally sound
and is the chosen approach.**

**(a) Descriptors live at `param_base + fixed offset`.** Kernel entry:

```
0x0090  ULDC.64 UR4, c[0x0][0x198]   ; {UR4,UR5} = param_base (a device pointer)
0x00a0  UIADD3  UR6,  UR4, 0x1f0     ; tensormap #1 addr = param_base + 0x1f0
0x00b0  UIADD3  UR8,  UR4, 0x2b0     ; tensormap #2 addr = param_base + 0x2b0
0x00c0  UIADD3  UR10, UR4, 0x370     ; tensormap #3 addr = param_base + 0x370
0x00d0  UIADD3  UR4,  UR4, 0x7f0     ; tensormap #4 addr = param_base + 0x7f0
0x0120  UTMACCTL.PF [UR6]            ; TMA prefetches the descriptor at UR6
0x0130  UTMACCTL.PF [UR8]            ; ... one per tensormap
```

So `c[0x0][0x198]` is **not** the descriptor; it is the **param base pointer**, and
each tensormap sits at a **distinct static offset**. That offset is the per-tensor
key we were missing (stable per pc via the def-chain).

**(b) Real GPU base values are already visible in the trace.**
`tma_desc_producer_debug.csv` already captured 64-bit values decoding to real device
VAs, e.g. `(lo=0x5C600000, hi=0x7F3F) = 0x7F3F5C600000`, which **exactly matches**
`global_address_hex` row 7 in the encode dump. So the param/constant space really
does hold the true bases.

**(c) Caveat — current capture is incomplete, not wrong.** The existing producer
capture only reliably logs one `ULDC c[0x0][0x208]` slot, so only 1/7 bases matched
directly and some captured VAs (`0x7f3f9a000000`) are *other* tensor args, not the
descriptor qword0. The descriptor is 128B spanning several const words; the tracer
does not yet dump the whole 128B at `param_base + offset`. That is the piece to add.

**Conclusion — Path B works. Required trace-gen work:**
1. For each executed `UTMALDG` (and family, §2.6), recover its descriptor **static
   offset** from `param_base` via the def-chain (constant per pc).
2. Capture `param_base` (the `ULDC c[0x0][0x198]` runtime value) once per launch.
3. Dereference `param_base + offset` and dump the **128B descriptor**; take qword0 =
   base, read strides for `addr = base + Σ coord·stride`.
4. Cross-check the dumped 128B against `tensor_map_encode_blobs/*.bin` to bind to the
   exact encode row (also yields strides/swizzle as defense-in-depth).

The binding key becomes the **descriptor address** (unique), so this sidesteps the
`desc_hi` non-uniqueness entirely. Device-only, matches HW semantics.

## 2.11 SPIKE 3 CONFIRMATION (def-chain auto-trace) — offset is a unique, stronger key

Auto-traced the def-chain of **every** `UTMALDG` in the FA3 fwd SASS (operand-2 back
to `ULDC c[0x0][0x198]` + `UIADD3` const adds). Every site resolves to
`param_base@0x198 + <fixed offset>`, and there are exactly **5 offsets**. Joining
each offset to the runtime `desc_hi` and the resolver box config:

| param offset | desc_hi seen | box config | pcs |
|---|---|---|---|
| 0x388 | 0x12f00000 | box_64x128 | 9 |
| 0x448 | 0x14f00000 | box_64x192 | 36 |
| 0x508 | 0x14f00000 | box_64x192 | 36 |
| 0x5c8 | (not executed in sample) | — | 6 |
| 0x688 | (not executed in sample) | — | 6 |

Two decisive facts:
1. **Each offset maps to exactly one desc_hi / box** — the offset is a consistent
   per-tensor key.
2. **Offsets 0x448 and 0x508 share the same `desc_hi` (0x14f00000) but are distinct
   offsets.** `desc_hi` cannot tell these two tensors apart; **the offset can.** This
   is direct proof the param offset is *strictly stronger* than `desc_hi` — it
   separates tensors that `desc_hi` collapses. Combined with the 128B dereference
   (qword0=base), it gives the true per-tensor base and full descriptor.

So Path B is confirmed twice: the address is recoverable (§2.10) **and** the key it
provides is unique where `desc_hi` was not (§2.11).

## 3. Design

### 3.1 Trace-gen (NVBit) — dereference the descriptor and dump ALL fields (Path B)

Files: `util/tracer_nvbit/tracer_tool/tracer_tool.cu`,
`util/tracer_nvbit/tracer_tool/inject_funcs.cu`.

The device job is to recover, per executed descriptor-carrying TMA site, the
**descriptor address** and dump the **whole 128B descriptor** (not just the base),
so every descriptor field comes from **one authoritative source** and all inference
is eliminated.

1. **Static per-pc offset (def-chain).** Recover the descriptor operand's origin
   `param_base@c[0x0][K0] + fixed_offset` via the def-chain (§2.11; constant per pc).
   Operand per family (§2.6): operand-2 for `UTMALDG`/`UTMAREDG`, operand-1 pair for
   `UTMASTG`, `desc[URx]` for desc-backed `UBLKRED`.
2. **Runtime descriptor VA.** Capture the runtime value of that operand — it already
   evaluates to `param_base + fixed_offset`, i.e. the exact 64-bit **descriptor VA**.
3. **Dereference + dump 128B.** Read 128 bytes at that VA and emit the 8 qwords per
   site (qword0 = base; rest = dim/stride/box/swizzle/etc).
4. Emit to `tma_transfer_addr_debug.csv`, keyed by
   `(unique_function_id, pc_hex, descriptor_va, cta_x/y/z, warp_id_tb, sm_id)`, plus
   the coordinate operand value for `UTMALDG`.

### 3.2 Trace-gen (python) — decode the 128B descriptor, DELETE all heuristics, assert-on-fail

Files: `util/tracer_nvbit/build_tma_descriptor_mapping.py`.

1. **Decode the dumped 128B directly.** For each executed site, decode its 8 qwords
   into the full descriptor (base, rank, global_dim, global_strides, box_dim,
   element_strides, interleave, swizzle, l2_promotion, oob_fill). This is the exact
   descriptor HW used — no `handle_hi` family lookup, no rank inference, no
   box-family guessing.
2. **Cross-check against the host encode dump (defense-in-depth).** Match the dumped
   128B (or decoded fields) to a row in `tensor_map_encode_dump.csv` /
   `tensor_map_encode_blobs/*.bin`, validating the device decode against host truth.
3. **DELETE every heuristic (per user).** Remove entirely, not merely as "final
   authority": `handle_hi_to_box_dim_family_with_opcode_rank`,
   `single_rank_candidate_config`, `same_function_desc_reg_config_reuse`,
   `*_inferred_rank_from_handle_reuse`, `*_inferred_rank_from_desc_reg_reuse`. With
   the descriptor bytes in hand the resolver becomes a direct decode, not a matcher;
   `handle_hi`/`config_id` family logic is obsolete.
4. **Assert-on-fail (hard, per user).** Every executed descriptor-carrying site must
   yield exactly one decoded descriptor. If a site has no dumped 128B, a malformed
   blob, or a device-decode that does NOT match any host encode row, `assert`/abort
   with `(uid, pc, opcode, descriptor_va, dumped qwords, nearest encode row)`. No
   silent fallback, no "weakly bound".

### 3.3 Simulator — carry the full decoded descriptor, compute HW address, assert on miss

Files: `tma_types.h`, `gpu-sim.cc` (loader), `tma_unit_sm.{h,cc}`.

1. `TMADescriptorConfigMetadata` / `TMACommand`: add `global_base` (u64), keep
   `global_strides`, populate `TMACommand.coords[]` (tma_types.h:250, currently 0).
2. Loader (gpu-sim.cc:340-389): parse the per-site decoded descriptor (base + all
   fields) keyed by `(uid, pc, descriptor_va)`.
3. Mover (tma_unit_sm.cc:633-639): replace the synthetic base with
   `agu_base = global_base + sum(coord[d]*global_stride[d]) + agu_index*128`.
4. **Assert on lookup miss.** When a TMA command is issued and the flag is on,
   resolving its descriptor must succeed; if the loader has no entry for `(uid, pc)`
   or no base is present, `assert` — never silently fall back to the synthetic base.
5. **Gated flag** `-tma_real_base_addr_enable` (default off) for clean A/B. When on,
   the synthetic-base path is dead code (the assert guarantees real base is used).

## 4. Verification (aligns with TMA_LATENCY_INJECTION §4 gates)

1. **Correctness first (not cycles).** With real base on, distinct FA3 tensors must
   land in distinct L2 regions; the cross-SM synthetic hotspot
   (`L2_TMA_res_fail` storm, ADDR_MERGE §1-A) must disappear.
2. **L2-hit realism gate (the decision metric, cycle-sign-agnostic).** Real
   addresses will very likely **lower** the currently-inflated `L2_TMA_true_hit_rate`
   (fwd 0.9846 / bwd 0.9657 in `.o24`/`.o6`) toward HW (fwd 69.58% / bwd 82.26%).
   Per the user's "no fake wins" rule, a cycle **increase** that moves L2 hit rate
   toward HW is an **accuracy improvement**, not a regression.
3. **FA3 binding safety gate** (`TMA_TRACING_revised.md` §FA3 Safety Gate): compare
   new `tma_descriptor_resolver.json` against the `.bak` baseline; previously-bound
   sites must not silently change `config_id`; newly unbound executed
   descriptor-involved sites are failures.
4. Record cycle deltas + L2 hit rate vs Opt 5 and vs the reply-path 6A results in
   `FA3_progress.md` (this supersedes Arch TODO-2).

## 5. Sequencing

The reply-path A/B matrix runs on a separate server and is **independent of the
address model** (`TMA_LATENCY_INJECTION` §4.5), so this work proceeds in parallel.
Note its pressure magnitude is inflated by the fake 98% L2 hit rate, so the two
must be cross-checked together at the very end.

1. **Spike 1/2 historical note:** keep them only as background. They explain why the
   old `desc_hi` / heuristic path was misleading, but they are no longer the chosen
   design.
2. **Spike 3 — DONE (§2.10, §2.11):** confirmed that the runtime descriptor address
   is recoverable as `param_base + static_offset`, and that this key is strictly
   stronger than `desc_hi`.
3. **Trace-gen NVBit (§3.1):** dump the executed site's **descriptor VA + full 128B
   descriptor bytes** for every descriptor-carrying family.
4. **Trace-gen python (§3.2):** decode that dumped 128B directly into the final
   descriptor metadata, delete all heuristic binding code, and `assert` on any
   missing / malformed / ambiguous mapping.
5. **Simulator (§3.3):** consume the decoded per-site descriptor, replace the
   synthetic base path with real descriptor-derived address generation, and `assert`
   on lookup miss when the feature is enabled.
6. **Validation:** compare device-dumped 128B against host encode blobs, then run the
   real-base A/B and judge success by L2-hit realism first, not by cycle sign.

*Files touched (summary):* `tracer_tool.cu`, `inject_funcs.cu`,
`build_tma_descriptor_mapping.py`, `tma_types.h`, `gpu-sim.cc`,
`tma_unit_sm.{h,cc}`.
