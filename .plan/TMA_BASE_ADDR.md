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

## 0.01 How the gap dissolves (resolution summary — read this first)

The gap was unsolvable **only because we tried to find a common field** (a value
present on both sides). There is none: `desc_hi` is device-only + non-unique,
`global_address` is host-only, `tensor_map_ptr` is a reused host VA. So stop matching
fields — instead **go to the memory where the descriptor lives and read the base
directly**, exactly like the HW does.

Two device-observable facts make this work (proven in §2.10/§2.11):
- the by-value `CUtensorMap` copy physically sits at `param_base + static_offset` in
  **device global memory**, and
- `static_offset` is recoverable **per pc and unique per tensor** from the SASS
  def-chain (strictly stronger than `desc_hi` — it separates the two tensors that share
  `desc_hi=0x14f00000`).

```
 [Host: cuTensorMapEncodeTiled]                 [Device: executed UTMALDG at pc]
   writes 128B descriptor                         pc ──def-chain(§2.11)──▶ static_offset (unique)
   base = qword0                                  param_base = runtime value of c[0x0][0x198]
        │                                                  │
        │  passed BY VALUE → the same 128B is             descriptor_VA = param_base + offset
        │  copied into the device param buffer            │
        │                                          deref 128B at that VA ─▶ base = qword0
        │                                                  │
        └── they meet HERE: the memory holding the by-value copy ──┘
                    (host encode dump is now only a cross-check)
```

- **The "missing common key" was the wrong thing to look for.** The real join is the
  **descriptor's memory location (VA)**, computed entirely from device-side signals
  (`param_base` + `offset`). Whoever reads that VA — the HW, a device-side load, or a
  host `cuMemcpyDtoH` — gets the identical bytes.
- **Identity is decided on the device (offset), not by host matching.** `param_base`
  (runtime) and `offset` (static per pc) both come from the device world; the host only
  performs the final *read*. So host deref does **not** reintroduce the join problem.
- **Host encode dump is demoted to validation.** It is no longer the authority; it is a
  defense-in-depth cross-check that the run-time bytes equal the encode-time bytes.

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

## 2.12 OPEN GATE (Gate 0) — prove `deref(param_base+offset)` is the right 128B for ALL offsets

§2.11 proved the **key** (offset↔tensor is 1:1). It did **not** prove the **value**:
that reading 128B at `param_base + (offset−0x198)` returns the descriptor whose
`qword0` equals that tensor's `global_address_hex`. §2.10(c) only matched **1 of 7**
bases directly; the rest is still assumed. This must be closed **before** writing the
tracer, because it is the single load-bearing assumption of Path B.

Gate 0 is a spike whose by-product **is** the `offset → encode-row` binding table, so
verification and deliverable are the same artifact:

1. Device: capture the runtime 64-bit `param_base` (the `ULDC.64 c[0x0][0x198]` value)
   once per launch.
2. For each of the 5 offsets, obtain 128B at `param_base + (offset−0x198)` (mechanism
   per §2.13) and byte-compare against every `tensor_map_encode_blobs/*.bin`.
3. **Pass = each offset matches exactly one encode blob, and its `qword0` equals that
   row's `global_address_hex`.** All 5 unique ⇒ Path B holds.
4. **Fail** (a VA is not host-readable, or a blob does not match) ⇒ fall back to
   **Path A** (launch-arg by-value dump, §2.5-4(b)): dump the by-value `CUtensorMap`
   bytes from `p->kernelParams` at `cuLaunchKernel` and match to an encode blob — no
   runtime deref at all.

## 2.13 DECISION (chosen: HOST DEREF) — how to obtain the 128B

Both device and host deref read the **same device-global bytes** (§3.1 "where the
descriptor lives"). **Decision: host deref**, because it needs no `inst_trace_t`/channel
change and reuses the existing encode-blob infrastructure.

| Option | Reads what HW reads | Trace-gen cost | Main risk | Status |
|---|---|---|---|---|
| **Host deref** (`cuMemcpyDtoH(param_base+off,128)`) | yes, same bytes | small (capture `param_base` only) | param-buffer lifetime; VA host-copyable | **CHOSEN** |
| Device deref (inject 128B load at site) | yes, at exec time | `inst_trace_t` + channel + inject_funcs | larger diff | not chosen |
| Path A (launch-arg by-value dump) | yes (the copied blob) | enable `nvbit_get_kernel_argument_sizes` | — | **fallback if Gate 0 fails** |

**Concrete host-deref mechanism (what Phase 0b implements):**
- **Capture `param_base`.** Instrument the kernel-entry `ULDC.64 URx, c[0x0][K]`
  (K = the param-base cbank index, `0x198` in FA3 fwd; auto-detected by §3.1 step 1)
  and read its **destination** UREG pair at `IPOINT_AFTER` — that is the loaded
  `param_base` (grid-uniform, capture once per `(device, uid)`).
- **Deref at a deadlock-free time.** Do the `cuMemcpyDtoH` from the **application
  thread in the `cuLaunchKernel` exit handler, after `cuCtxSynchronize`** (kernel done,
  channel drained, param buffer still alive before returning to the app). Never memcpy
  from the recv thread mid-kernel (that can deadlock the channel).
- **Which offsets.** Read the 5 offsets from `tma_descriptor_offsets.json` (Phase 0a),
  compute `VA = param_base + tensormap_offset`, dump 128B + qwords to
  `tma_descriptor_deref.csv` (+ per-offset `.bin`).

This is exactly the pair of risks Gate 0 (§2.12) exists to test: (a) is the const-bank
param VA host-copyable, and (b) is the post-sync buffer still valid. If either fails,
switch to Path A.

## 2.14 SPIKE 4 (measured on FA3 bwd k=uid8) — fwd is clean by-value; bwd is by-pointer + operand-based, so DO FWD FIRST

Ran the def-chain extractor + runtime-operand analyzer on the **bwd** trace
(`extract_tma_descriptor_offsets.py`, `analyze_tma_runtime_operands.py`). The bwd
kernel does NOT match the clean fwd by-value model. Decision: **land the whole
pipeline on fwd first (100% clean), treat bwd as a separate follow-up.**

**(a) fwd is uniformly clean.** Every fwd kernel variant resolves to exactly 2
param-base offsets (`0x2f0/0x3b0` or `0x470/0x530`) via `param_base@0x198 + offset`.
This is the §2.10/§2.11 model and host-deref (§2.13) covers it fully.

**(b) bwd executed UTMALDG operands are SMEM at runtime — same as fwd §2.8.** The
runtime analyzer shows all 10 executed UTMALDG sites are `no_gmem_operand` (operands
are SMEM staging cursors `0x10400/0x18400` and SMEM-window `0x2e2a0xxx`). So a
device-side operand capture cannot recover the descriptor VA either — the **static
param_base+offset path is still the only route** for bwd UTMALDG.

**(c) but bwd passes tensormaps BY POINTER, not by value.** Executed bwd UTMALDG
def-chains terminate at **multiple, non-0x198 cbank slots with large offsets**:
```
pc=0xa240  UR? <- ULDC c[0x0][0x198] + 0x2b0        (clean, by-value — 1 of 10)
pc=0x6ea0  UR6 <- ULDC c[0x0][0x2ac] + 0x0
pc=0x74f0  UR6 <- ULDC c[0x0][0x2ac] + 0x40600      (264 KB)
pc=0x7670  UR14<- ULDC c[0x0][0x280] + 0x8000       (32 KB)
pc=0xa320  UR32: no_def  (desc reg comes from an LDG / memory load)
```
`c[0x0][0x2ac]` / `c[0x0][0x280]` are **pointers inside the param block to a GMEM
tensormap array**, and the large offsets index into that array. So the bwd recovery is
a **two-level dereference**: `deref(param_base + 0x2ac)` = array pointer, then
`deref(array_ptr + index)` = the 128B descriptor. The `no_def` sites take the pointer
from an `LDG`, which static analysis cannot follow at all.

**(d) bwd UBLKRED is operand-based, not descriptor-based.** All 6 executed UBLKRED
sites expose a real **GMEM operand** (ci=1 = `0xffffffffc9......`, a sign-extended
device VA = the actual reduction target), with a desc handle `{lo=0x85,hi=0x85}` that
is span metadata, not a tensormap. These need no descriptor bind (TMA_TRACING §2.6
"non-descriptor UBLKRED"): the GMEM address is already in the operand.

**Decision (recorded):**
1. **Phase 0b/0c/2/3 run on the FWD trace first.** fwd is 100% clean by-value; prove
   real-base end-to-end (L2 hotspot removal, L2-hit realism) there.
2. **bwd is a separate design** with three sub-cases, deferred until fwd lands:
   - by-value UTMALDG (minority) → same host-deref path;
   - by-pointer UTMALDG (majority) → **two-level deref** (array pointer then 128B),
     plus a way to handle `LDG`-sourced descriptor pointers;
   - UBLKRED → operand-based GMEM address, no descriptor bind.
3. Do **not** force bwd through the fwd model — it would fabricate wrong offsets
   (the `0x40600`/`no_def`/multi-source cases prove it).

## 2.15 SPIKE 5 CORRECTION (per-uid split on the SAME bwd-causal trace) — the EXECUTED workload is by-pointer for BOTH uids; §2.14(a) "fwd is clean" was about NON-executed static sites

Added per-uid executed stats and re-read the trace. **§2.14's "fwd is uniformly clean"
is misleading and is corrected here.** That statement was true only of the *static*
(mostly non-executed) fwd kernel variants; the *executed* loads tell a different story.

**(a) Both executed uids are dominated by by-pointer, not by-value.**
```
by uid (executed): uid 3 = 12/13 (92.3%)   uid 8 = 7/23 (30.4%)
```
The `--only-uid 3 --dump-chains` output shows uid 3's executed UTMALDG sites use the
**same** non-param-base pattern as uid 8:
```
uid=3 pc=0x6d20 UTMALDG  UR6 <- ULDC c[0x0][0x2ac]                       (off 0x0)
uid=3 pc=0x7370 UTMALDG  UR6 += 0x20400; UR6 += 0x20200; <- c[0x0][0x2ac] (off 0x40600)
uid=3 pc=0x6d50 UTMALDG  UR32: no_def                                     (LDG-sourced)
uid=3 pc=0x9650 UTMALDG  UR22 <- UR8 + 0x20000; <- c[0x0][0x2ac]          (off 0x20000)
```
So the "clean fwd by-value" kernel effectively **does not run** in this bwd-causal
trace. The 92.3% is an artifact of counting a couple of by-value Q-loads plus the fact
that most sites are static/non-executed.

**(b) Self-accumulation is real, not an artifact.** `UR6 += 0x20400; UR6 += 0x20200`
folds correctly to a fixed `0x40600` (verified). These are loop-induction walks over a
**GMEM tensormap array** — the large offset is the array index, terminating at the
pointer slot `c[0x0][0x2ac]` / `c[0x0][0x280]` / `c[0x0][0x2a0]`.

**(c) `no_def` (UR32/UR34) = descriptor pointer comes from an `LDG`** (a memory load),
which static analysis cannot follow at all.

**Corrected conclusion:** for THIS trace the recovery is fundamentally **two-level**
(param pointer → GMEM tensormap array → 128B descriptor), plus an `LDG`-sourced subset
that static analysis cannot reach. The single-level host-deref (§2.13) covers only the
minority by-value sites.

**Next step (hypothesis proof before any redesign code):** prove level-1 first — that
`c[0x0][0x2ac]` / `c[0x0][0x280]` hold **GMEM pointers** — by reading those 8 bytes out
of the Phase-0b param-region dump (`prove_tma_tensormap_pointer.py`). If confirmed,
design the two-level deref (dump the pointed-at GMEM range, index by the large offset,
take qword0). A pure fwd-only trace (separate run) is still worth capturing to confirm
the clean by-value path exists there, but it is not what this bwd-causal trace exercises.

## 2.16 Per-opcode descriptor recovery — the OLD `handle_hi` join vs the NEW real-address specific mapping

This is the spine of the whole effort, re-stated so it does not drift again.

**What "getting the descriptor" means.** For each executed TMA op the simulator must
know **which tensor-map (128B descriptor) that op uses**, because that descriptor holds
`global_base + strides + box` — the inputs to the real HW address
`addr = global_base + Σ coord·stride`. The question per opcode is only: *how do we name,
for this exact executed site, the one descriptor it consumes?*

**OLD (TMA_TRACING.md, being replaced).** Descriptor **values** came from the host
`cuTensorMapEncodeTiled` dump; the executed site was tied to a value by a **join key**
`(uid, pc, handle_hi)` → `config_id`. This is the part that fails: `handle_hi` (=
`desc_value_hi`) takes only 3 values for 7 bases, so the join is **not unique** and
needs rescue heuristics (rank reuse, desc-reg reuse, candidate lists). It never yields a
specific GPU base.

**NEW (Path B, this doc).** Stop joining by a shared field. Instead, for each executed
site compute the **descriptor's memory address (VA)** the way HW does, then **read the
128B there** and decode every field from those bytes. The "mapping" is no longer a
fuzzy tag match — it is the *physical location of the descriptor the HW itself
dereferences*. `handle_hi` is demoted to a debug column; the host encode dump is demoted
to a cross-check.

The only per-opcode difference is **where that descriptor VA comes from** (which SASS
operand carries it). Note: `desc[URx]` is NOT the descriptor — it is a constant handle
`{lo=0, hi=handle_hi}` (§2.5). The descriptor VA is an *address operand*, not the desc
handle.

| Opcode (FA3 form) | Descriptor-VA source (NEW key) | Level | How the 128B is obtained |
|---|---|---|---|
| `UTMALDG.4D [dst],[ptr],desc[URx]` | **operand-2 `[ptr]`** VA = def-chain from `ULDC c[0x0][K]` (+const adds) | **by-value** when `K` = param-base (`0x198`): `VA = param_base + off`. **by-pointer** when `K` ≠ param-base (`0x2ac/0x280/0x2a0`): `VA = deref(param_base+K) + index` | deref VA → 128B → qword0=base |
| `UTMAREDG` (not in FA3 run) | same as `UTMALDG` (operand-2) | same rule | same |
| `UTMASTG.4D/5D [URa],[URb]` | **operand-1 pair `[URa]`** VA (no `desc[]`; first pair is desc-like) | by-value in FA3 (reaches `0x198`) | deref VA → 128B |
| `UBLKRED …,desc[URx]` (desc-backed) | **`desc[URx]` as a VA** *(needs recheck — see below)*; operand-3 stays span-only | mixed | deref VA → 128B; keep operand-3 separate |
| `UBLKRED …` (non-desc / bulk) | **none** — operand already holds the real GMEM target (`0xffff…` sign-extended VA) | n/a | no descriptor bind; use operand address + `op3*16` span |
| `UBLKCP`,`UBLKPF` | none (bulk; operands are the moved-region cursors) | n/a | no descriptor bind |
| `UTMAPF.L2.4D [URa],[URb]` (no desc) | **inherit** from an exact-linked later `UTMALDG` (same `uid`, equal operand-1 samples) | follows the linked load | inherit that load's descriptor |
| `UTMACCTL.PF`,`UTMACMDFLUSH` | none (control-only) | n/a | no base generation |

**Two important corrections that the bwd data forced (§2.14/§2.15):**

1. **`UTMALDG` has two levels, not one.** The clean fwd case is *by-value* (K=0x198,
   descriptor inline in the param block). The executed bwd case is dominated by
   *by-pointer*: the param slot `c[0x0][0x2ac]` holds a **GMEM pointer to a tensormap
   array**, and the large def-chain offset (`0x40600`, `0x8000`, …) is the **array
   index**. So `UTMALDG` recovery must support both: level-1 `VA = param_base+off`
   (by-value) and level-2 `VA = deref(param_base+K) + index` (by-pointer). A subset is
   `LDG`-sourced (`no_def`) and cannot be recovered statically at all.

2. **`UBLKRED` splits by form (already in §2.6, reconfirmed on data).** The desc-backed
   form has `desc[URx]`, but the FA3 bwd runtime shows its **operand ci=1 is a real GMEM
   address** (`0xffffffffc9…`) and the handle is span-ish `{0x85,0x85}` — i.e. these
   executed `UBLKRED` behave **operand-based**, needing no tensormap deref. This must be
   rechecked per site before treating any `UBLKRED` as descriptor-backed: if the operand
   already carries the GMEM target, do **not** force a descriptor VA.

**Net rule for the redesign:** every descriptor-carrying site is keyed by a **real
descriptor VA**, and the base comes from the **128B read at that VA** — never from
`handle_hi`. `UTMALDG`/`UTMAREDG` use operand-2 (with by-value vs by-pointer levels),
`UTMASTG` uses operand-1 pair, desc-backed `UBLKRED` uses `desc[URx]` *only if* the
operand doesn't already carry the GMEM target, `UTMAPF` inherits via exact link, and
bulk/control ops need no descriptor. This is the "specific per-address mapping" that
replaces the non-unique `handle_hi` tag.

## 2.17 SPIKE 6 (measured on the copied trace, fwd uid3 + bwd uid8) — the real mechanism: param holds a POINTER to the descriptor, and both fwd & bwd are identical

This is the decisive spike. It resolves the §2.8 "operand is SMEM but def-chain says
param-global" contradiction, corrects my operand-numbering error, and proves fwd and
bwd obey **one** rule (answering the user's "fwd-only can't pass").

**Data used:** `tensor_map_encode_dump.csv` (7 distinct bases, qword0=base),
`tma_runtime_operand_debug.jsonl` (real operand values), `tma_desc_producer_debug.csv`
(producer reg values), and the fwd SASS `flash_fwd_hdim64_bf16_sm90.sm_90a.sass`
(uid3, verified to contain UTMALDG@0x94e0 exactly once).

**(a) Operand numbering corrected.** For `UTMALDG.4D [UR16], [UR8], desc[UR10]`
@0x94e0 the def-chains + runtime values are:
```
operand-1 [UR16] (ci=0): R2UR UR16,R12; +0xe400   runtime 0xffffffffc42804c0  = SMEM dst cursor
operand-2 [UR8]  (ci=1): UR24+0x2b0; UR24=ULDC c[0x0][0x198]   runtime 0xe800/0x12800
desc      [UR10]        : UMOV UR10,0x0            runtime {0,0x14f00000}     = constant handle
```
So operand-2 IS `param_base + 0x2b0` (as §2.10/§2.11 predicted), **but its runtime value
is a SMEM address `0xe800`.** My earlier §2.14/§2.16 "by-value vs by-pointer" framing was
wrong about which operand and about by-value.

**(b) What the param slot actually contains — a POINTER, not the 128B descriptor.**
`operand-2 = param_base + 0x2b0` is the *address of a param slot*; the value the kernel
reads from that slot (and puts into the operand at issue) is `0xe800`, a **SMEM address**.
The kernel entry proves the intent:
```
0x090 ULDC.64 UR4, c[0x0][0x198]        ; UR4 = param_base
0x0a0 UIADD3  UR6, UR4, 0x1f0           ; &tensormap_ptr[0]
0x0b0 UIADD3  UR8, UR4, 0x2b0           ; &tensormap_ptr[1]
0x0c0 UIADD3  UR10,UR4, 0x370           ; &tensormap_ptr[2]
0x0d0 UIADD3  UR4, UR4, 0x7f0           ; &tensormap_ptr[3]
0x120 UTMACCTL.PF [UR6]                 ; TMA prefetches the descriptor pointed at by each slot
0x130 UTMACCTL.PF [UR8]                 ; (one per tensormap)
0x140 UTMACCTL.PF [UR10]
0x150 UTMACCTL.PF [UR4]
```
So the param block at `+0x1f0/+0x2b0/+0x370/+0x7f0` holds **tensormap descriptor
pointers**; `UTMACCTL.PF` prefetches each descriptor into the TMA descriptor cache, and
the kernel also stages a copy the TMA engine reads (the SMEM `0xe800` seen at issue).
The descriptor is **not** inline-by-value in the param block.

**(c) The real base IS in the trace.** `tma_desc_producer_debug.csv` holds device VAs
that match the encode dump exactly, e.g. `0x7f3f5c600000` (= encode row 7 base) appears
66,410 times. So the true base is observable; it is reached by dereferencing the
descriptor pointer, not by reading the param slot as 128B.

**(d) Why host-deref of `param_base+offset` FAILED to map (root cause).**
`cuMemcpyDtoH(param_base+0x2b0, 128)` reads the *param slot*, which contains the pointer
value (→ `0xe800`, a SMEM address), **not** the 128B descriptor. The descriptor lives
where that pointer points (descriptor cache / a staged SMEM copy / the pointed global
buffer), which host `cuMemcpyDtoH` cannot follow — SMEM is not host-addressable. That is
exactly why the mapping never matched. The §2.10(c) "1/7 matched" was luck: one slot
happened to hold a value that looked like a base.

**(e) fwd and bwd are the SAME.** uid8 `UTMASTG.4D` shows the identical shape:
ci=0=`0xffffffffc42a07c0` (SMEM dst cursor), ci=1=`0x4400` (SMEM), desc handle small.
Both kernels are CUTLASS **persistent** (`VarlenDynamicPersistentTileScheduler`) and both
route the descriptor through a param pointer + staged copy. **There is no fwd-only path**
— confirming the user. The 92.3% vs 30.4% "resolved" numbers were an artifact of the
broken static assumption, not a real fwd/bwd difference.

**(f) FACT-NAILED: the live descriptor is SMEM-staged, and its address is observable.**
Cross-checking the entry `UTMACCTL.PF` (descriptor prefetch) with the `UTMALDG` operands
and their runtime values pins the location exactly:
```
UTMACCTL.PF [UR6]   runtime 0xffffffffc4280400   ; descriptor prefetch target
UTMALDG op1 [UR16]  runtime 0xffffffffc42804c0   ; SAME generic window, +0xc0
UTMALDG op2 [UR8]   runtime 0xe800               ; separate raw-shared cursor (staging/barrier)
UTMASTG op1 (uid8)  runtime 0xffffffffc42a07c0   ; identical scheme in bwd
```
`0xffffffffc428xxxx` is a **generic-address-space SMEM window** address (Hopper maps
shared memory into the high generic window `0xffffff…`). `UTMACCTL.PF` and `UTMALDG`
operand-1 point into the **same** window, and the strides between consecutive descriptors
are `0xc0 = 192` — matching the `box_dim 64×192`. So the descriptor the HW dereferences
at issue is a **SMEM-staged tensormap** at that generic-SMEM VA. This is the same in fwd
(uid3) and bwd (uid8).

Correcting my own §2.17(a): the descriptor VA is **operand-1 / the `UTMACCTL.PF` target**
(`0xffffffffc428xxxx`, SMEM), **not** operand-2 (`0xe800`, which is a raw-shared staging/
barrier cursor). Both are SMEM; neither is host-readable.

**Consequences (this is the design break):**
- **Host deref (§2.13) is dead** for these kernels — the descriptor is in SMEM, which
  `cuMemcpyDtoH` cannot read. The "same device-global bytes" premise of §2.13 was false.
- **Path A launch-arg dump (§2.12 fallback) is also dead** — at `cuLaunchKernel` the SMEM
  copy does not exist yet; the param block holds only a pointer.
- **Consistent with producer_debug:** 0/10 (uid3) and 0/12 (uid8) executed sites captured
  a real base — because their bases only ever appear inside SMEM staging, which the
  current producer capture does not read.
- **A global original still exists** (encode dump has the 7 real bases), but for uid3/8 it
  is only reached through the SMEM-staged copy at issue; the raw-global-pointer scheme
  (`c[0x0][0x208]`, seen feeding `producer_debug` bases) belongs to *other* kernels
  (uid1/6/7), not the FA3 persistent fwd/bwd.

So the recovery must read the **SMEM-staged descriptor at issue** on the device. That is
the capture decision that replaces §2.13 (see §2.18).

**(g) Not every opcode knows a descriptor location.** Per the user: this does NOT mean
all ops carry a descriptor VA. `UBLKRED` in this trace is operand-based (ci=1 is a real
target address like `0x5d408000`, handle is span-ish `{0x85}`); `UBLKCP`/`UTMACCTL`/
`UTMACMDFLUSH` are control/bulk with no descriptor to locate. The per-opcode table in
§2.16 stands, but with (f)'s two-level pointer deref replacing the by-value read for
`UTMALDG`/`UTMASTG`/desc-backed `UBLKRED`.

## 2.18 What is confirmed vs still open, and the device-side capture decision (replaces §2.13)

**Confirmed by trace data (SPIKE 6):**
1. Param slots hold a pointer/staging path, **not** the 128B descriptor.
2. The descriptor the HW reads at issue is **SMEM-staged** (`0xffffffffc428xxxx` generic
   window; `UTMACCTL.PF` and `UTMALDG` op-1 agree; 192-byte stride = box_dim).
3. **fwd and bwd are identical** (persistent-kernel staging) — no fwd-only path.
4. **Host deref and Path A are both dead** for these kernels (SMEM not host-visible;
   SMEM copy absent at launch).
5. `producer_debug` captured 0 real bases for uid3/8, consistent with the above.
6. `c[0x0][0x208]` is a *global* pointer used by `LDG.E desc[UR14]` — a **different**
   scheme belonging to other kernels, not the FA3 persistent fwd/bwd descriptor path.

**Still open (cannot be closed offline — SMEM is not in the trace files):**
- Whether the SMEM-staged descriptor's `qword0` equals the real base (expected, but
  must be verified on device).
- Whether a global original is reachable for uid3/8 *before* staging.

**Capture options (all device-side; pick after the fact-check reruns):**

| Option | What it reads | Trace-gen cost | Accuracy | Risk |
|---|---|---|---|---|
| **D1: read SMEM descriptor at issue** | inject a 128B shared-mem load at each `UTMALDG`/`UTMASTG` from operand-1's generic-SMEM VA | `inst_trace_t` + channel + `inject_funcs` | highest — the exact bytes HW uses, incl. per-tile coords | larger diff; must read at issue (predicated lanes) |
| **D2: read global original before staging** | find the `LDG`/bulk source that fills the SMEM copy, dump 128B there once | medium (identify the fill site) | base + static fields correct; per-tile coords may be pre-staging | must prove the fill source exists & is stable for uid3/8 |
| **D3: capture in producer chain** | extend existing producer capture to record the staged descriptor VA + a 128B read | reuse producer path | same bytes as D1 if it reads SMEM | producer path currently only logs const/reg values, not SMEM contents |

D1 is the faithful, HW-matching route (and the only one guaranteed to include per-tile
coord edits); D2 is cheaper if a stable global source exists. The next server step is a
**device-side fact-check**: at one `UTMALDG` issue, read 128B at operand-1's VA and
confirm `qword0` ∈ the 7 encode bases. That single check decides D1 vs D2 and closes the
last open item.

**Fact-check tooling (implemented, ready to build/run on server):**
- Tracer (`common.h`, `inject_funcs.cu`, `tracer_tool.cu`): gated by
  `TMA_DESC_FACTCHECK=1`. At each executed `UTMALDG`/`UTMASTG` it device-reads 128B at
  the descriptor-carrying **first memref** VA (the generic-SMEM tensormap address) and
  records `qword0/qword1` into a bounded managed buffer, dumped to
  `extra_info/tma_desc_factcheck.csv` at kernel exit.
- Verifier `verify_tma_desc_factcheck.py`: checks each sampled `qword0` against the 7
  bases in `tensor_map_encode_dump.csv`. Prints PASS (all match ⇒ D1 confirmed),
  PARTIAL (matches + `base+offset` near-hits ⇒ coords-adjusted copy, subtract coord
  term), or FAIL (none match ⇒ wrong operand / transformed bytes ⇒ consider D2).

Run recipe:
```
# rebuild tracer, then re-run the app with:
ENABLE_TMA_DESC=1 TMA_DESC_FACTCHECK=1 <existing trace-gen run cmd>
# then:
python3 verify_tma_desc_factcheck.py --traces "$TRACES"
```
Expected on PASS: samples for uid3 (fwd) and uid8 (bwd) with `desc_va` in the
`0xffffffffc428xxxx` SMEM window and `qword0` ∈ {the 7 bases}. This is the same rule for
both kernels — the whole point of §2.17(e).

## 2.19 SPIKE 7 (device fact-check crash) — the descriptor operand is NOT generic-readable

Running the §2.18 fact-check with an **unconditional** 128B read of the first-MREF VA
crashed the app: `CUDA error … an illegal memory access was encountered` at the tool's
post-launch `cudaDeviceSynchronize()` (before any host deref), i.e. the fault is inside
the injected `factcheck_tma_descriptor`, not the host code. This is itself decisive data:

- The descriptor-carrying operand VA (`0xffffffffc428xxxx`, §2.17f) is **not** servable by
  a plain generic load from the instrumentation context — a blind `*(u64*)desc_va` faults.
  Whether that is because it is a SMEM/TMA window a generic load can't reach, or because
  some sampled sites hand back a raw staging cursor (e.g. `0xe800`), the read is unsafe.
- **Conclusion:** classification must be decoupled from reading. `isspacep.*`
  (`__isGlobal/__isShared/…`) is a pure predicate that never touches memory, so we can
  label every `desc_va`'s address space with zero crash risk. That label alone answers the
  §2.18 open item (SMEM vs GMEM) — which is what actually decides D1 vs D2.

**Two-step design (implemented):**
1. `TMA_DESC_FACTCHECK=1` — classify only. Dump `space` (GLOBAL/SHARED/CONSTANT/LOCAL/
   UNKNOWN-generic) + `read_ok=0` per sample. Always crash-safe. **Run this first.**
2. `TMA_DESC_FACTCHECK_READ=1` — opt-in. Additionally read 128B, but only when the space
   is `GLOBAL` (guaranteed safe). SHARED/generic remain unread until a shared-aware probe
   (`ld.shared`) is added, since that is where the D1 vs D2 branch lands.

Interpreting step 1:
- mostly `SHARED` ⇒ descriptor is genuinely SMEM-staged ⇒ **D1** (must read as shared at
  issue; a generic/host read can never work — kills §2.13 for good).
- mostly `GLOBAL` ⇒ a global original is reachable ⇒ **D2** (and step 2's read will fill
  `qword0`, verifiable against the 7 bases).
- `UNKNOWN`/small ⇒ the picked operand is a raw cursor, not the descriptor ⇒ revisit the
  operand/MREF choice (try the `UTMACCTL.PF` target).

Updated recipe:
```
# rebuild tracer, then FIRST classify (crash-safe, no reads):
ENABLE_TMA_DESC=1 TMA_DESC_FACTCHECK=1 <existing trace-gen run cmd>
python3 verify_tma_desc_factcheck.py --traces "$TRACES"   # read the space histogram
# only if the histogram shows GLOBAL, opt into the byte read:
ENABLE_TMA_DESC=1 TMA_DESC_FACTCHECK=1 TMA_DESC_FACTCHECK_READ=1 <run cmd>
python3 verify_tma_desc_factcheck.py --traces "$TRACES"
```

## 2.20 SPIKE 7 RESULT (classify step, bwd uid8) — desc_va is GLOBAL, not SMEM. §2.17(f) corrected.

The crash-safe classify step ran clean (no illegal access) and returned, for the bwd
`flashattn-fa3-bf16-bwd` run:
```
desc_va address space (distinct samples):
    GLOBAL           x7
    -> read_ok (byte read performed): 0/7   (READ step not enabled yet)
encode bases: 7 distinct (0x7f001aa00000 … 0x7f01ad800000)
```

**All 7 distinct UTMALDG first-MREF VAs classify as GLOBAL.** This **overturns my
§2.17(f) claim** that the descriptor lives in a generic-SMEM window. That claim was an
**offline mis-read**: I saw `0xffffffff…`-looking values in the operand JSONL and assumed
generic→shared. The device predicate `__isGlobal` is authoritative and says the address
the HW reads at issue is in **global memory**.

Consequences:
- The descriptor (or its live copy) is **GMEM-resident and directly readable** at issue.
  So the recovery path is essentially **D2 / a direct global read** — *not* the SMEM-only
  D1 that §2.17–§2.18 concluded. Host `cuMemcpyDtoH` of that GMEM VA could even work
  (host-deref, §2.13, is **not** dead after all — it just needs the *runtime* VA, which
  only exists at issue, not the static `param_base+offset`).
- `read_ok=0/7` only because this run set `TMA_DESC_FACTCHECK=1` alone (classify), not
  `TMA_DESC_FACTCHECK_READ=1`. GLOBAL is safe to read, so the read step is the immediate
  next action — it will fill `qword0` and we check it against the 7 encode bases.
- If the read step shows `qword0 ∈ {7 bases}` (exact or base+coord near-hit), the real
  base recovery is **done from the device**, and the offline SMEM-staging detour is
  closed. If it does not match, we inspect the actual `desc_va`/`qword0` values (the VA is
  GLOBAL, so the bytes are trustworthy) to see whether it is a coords-adjusted tile
  address or a different structure.

Lesson: trust the on-device `isspacep` classification over any offline guess about what a
`0xffffff…` operand "looks like". This one predicate invalidated three sections of prior
SMEM reasoning.

## 2.21 SPIKE 7 CSV (read step, uid3 fwd + uid8 bwd) — §2.20 was WRONG; captured operand is the raw cursor, and isspacep mislabels it GLOBAL

The read step (`TMA_DESC_FACTCHECK_READ=1`) crashed again (illegal access), but the
classify CSV from the prior run is intact and decisive. Every captured `desc_va` is a
**small** value, not a device pointer:
```
uid3 (fwd): pc 0x94e0 → 0xe800 ;  0x96b0 → 0x8800 ;  0x9a80/0x9fc0 → 0x12800
uid8 (bwd): pc 0xa240 → 0x10400 ; 0xa320 → 0x400 ;  0xa330 → 0x4400
all rows: space=1 (GLOBAL), read_ok=0, qword0=0
```

**Two corrections to my own notes:**
1. **§2.20 ("desc_va is GLOBAL, descriptor is in global memory") is retracted.** `space=1`
   was an `isspacep.global` **false positive**: `0xe800` is a raw-shared *offset* that is
   not inside the shared/local/const windows, so `isspacep.global` returns true by
   elimination. It is **not** a global pointer — reading it faulted, which is why the read
   step crashed. So §2.17(f)'s SMEM-staging picture is *not* overturned after all.
2. **The captured operand was the wrong one.** These small values are exactly the
   §2.17(a) **operand-2 raw-shared cursors** (`0xe800`, `0x12800`), i.e. NVBit's mref-0
   for these ops maps to operand-2, not the operand-1 descriptor cursor
   (`0xffffffffc428xxxx`). We were dereferencing the staging cursor, never the descriptor.

**Fixes applied (this iteration):**
- Instrument **all** MREF operands of each `UTMALDG`/`UTMASTG`, each tagged with a
  `mref_ord` column, so the CSV shows which operand carries `0xffffffffc428xxxx` vs the
  small cursor.
- Harden the read guard: read only when `space==GLOBAL` **and** `desc_va >= 4 GiB`
  (`MIN_GLOBAL_VA`). This rejects the isspacep-mislabeled tiny cursors that caused the
  crash — the read step is now crash-safe regardless of which operand is picked.

**What the next CSV will show (and the decision):**
- For each site we now get mref-0 and mref-1. Expect one to be the small cursor and one to
  be `0xffffffffc428xxxx` (the §2.17f generic-SMEM descriptor window).
- If the `0xffffff…` operand classifies as **SHARED** ⇒ §2.17(f) confirmed, descriptor is
  SMEM-staged ⇒ **D1** needs a `ld.shared` probe (a generic load can't read it, and
  neither can the host) — the current read guard will (correctly) leave it unread.
- If some operand is a real **GLOBAL** VA `>= 4 GiB` ⇒ that is the descriptor/global
  original ⇒ the read fills `qword0`, checkable against the 7 bases (**D2**).

Lesson (again): `isspacep.global` is "not shared/local/const", **not** "valid global
pointer". Always pair it with a VA-magnitude sanity bound before dereferencing.

## 2.22 SPIKE 7 (all-MREF classify) — mref-1 IS the descriptor cursor (192-byte stride). Read floor lowered.

Capturing **both** MREF operands nailed the operand question. Per site:
```
uid=3 pc=0x94e0  mref0 0xe800   mref1 0x562804c0
uid=3 pc=0x96b0  mref0 0x8800   mref1 0x56280400
uid=8 pc=0xa240  mref0 0x10400  mref1 0x562a02c0
uid=8 pc=0xa330  mref0 0x4400   mref1 0x562a0380   (+0xc0)
uid=8 pc=0xa320  mref0 0x400    mref1 0x562a0440   (+0xc0)
uid=8 pc=0xa610  mref0 0x18400  mref1 0x562a0500   (+0xc0)
```

- **mref-0** = tiny raw-shared **staging cursor** (`0xe800 … 0x18400`, all ≤ ~99 KiB) —
  the §2.17(a) operand-2. NVBit orders this as mref-0. Not the descriptor.
- **mref-1** = the **descriptor cursor** (`0x5628xxxx`/`0x562axxxx`). Two proofs it is the
  descriptor: (1) consecutive uid8 sites step by exactly `0xc0 = 192` = the `box_dim`
  (64×**192**) — the descriptor array stride from §2.17(f); (2) the low bits match the
  §2.17(f) `0xffffffffc42804c0` observation (here without the generic high bits, so it
  reads as ~`0x562804c0` ≈ 1.4 GiB).
- Both classify GLOBAL, but that includes the isspacep false-positive for mref-0, so
  address space alone can't separate them — the **magnitude** does.

**Why the read step still showed 0/12:** the earlier guard floor was 4 GiB; the descriptor
cursor is ~1.4 GiB, below it, so it was (correctly) not read — but neither was it probed.
Lowered the floor to **1 MiB** (`MIN_READ_VA`): rejects every tiny mref-0 cursor
(≤ 99 KiB) while admitting mref-1. The next `TMA_DESC_FACTCHECK_READ=1` run will attempt
to read mref-1's 128B.

**Decision this pins next:**
- If reading mref-1 **succeeds** and `qword0 ∈ {7 bases}` (exact or base+coord near-hit) ⇒
  the descriptor is directly readable at that VA ⇒ real base recovered on device (**D2**,
  and host `cuMemcpyDtoH` of the same VA would also work).
- If reading mref-1 **faults** (crash-safe now: only mref-1 is attempted, tiny cursors are
  skipped) ⇒ that VA is a shared/generic alias the HW resolves specially ⇒ **D1** (needs a
  shared-aware probe, not a generic load).

## 2.23 SPIKE 8 — device read is DEAD; switch to Path A (host launch-arg dump)

**Device read is conclusively ruled out (5 crashes).** read-off classify = clean;
reading mref-0 (tiny cursor) = fault; reading mref-1 (descriptor cursor ~0x562804c0,
~1.4 GiB) = fault. A GPU device load that faults **aborts the whole context** — it can't
be guarded like a CPU read. mref-1 faulting confirms §2.17(f): that VA is a TMA-engine
generic/SMEM alias, not a warp-loadable global. So **no on-device UTMALDG-time read can
recover the base**.

**Also confirmed the base is NOT in producer_debug for uid3/8.** The only producer row is
`ULDC.64 UR6, c[0x0][0x208]` with value 0 — the uid1 global-pointer scheme (§2.18(6)),
not our persistent fwd/bwd. §2.17(c)'s "base is in producer_debug" was a different uid.

**Everything except the join already exists on the host.** `tensor_map_encode_dump.csv`
has the 7 real bases + all descriptor fields (strides/box/swizzle, already consumed by the
sim). The only missing link is "which encode event ↔ which UTMALDG pc" — the original §0.0
gap. `tensor_map_ptr_hex` is a host stack addr (`&map`, identical for all 6), so it can't
key tensors.

**Decision (user-approved): Path A — dump the host launch-argument buffer.** At
`cuLaunchKernel` the driver hands us `p->kernelParams`, a **host** array of per-argument
pointers. For a by-value `__grid_constant__ CUtensorMap` argument, that host pointer points
at the 128B descriptor whose qword0 = the real base. Reading it is a **host** memory access
— it cannot trigger the device illegal-access that killed every D1/D2 attempt.

**Implemented (this iteration):**
- Tracer `append_tma_launch_param_dump_event` (gated by `ENABLE_TMA_DESC`): at each
  `cuLaunchKernel`, walk `p->kernelParams`, use `nvbit_get_kernel_argument_sizes(ctx, f)`
  to derive each argument's packed **param_offset**, and dump the first 16 qwords read from
  the host arg pointer to `extra_info/tma_launch_param_dump.csv` (columns: `device_id,
  kernel_id,unique_function_id,arg_index,param_offset_hex,arg_size,arg_ptr_hex,qword0..15`).
- Verifier `verify_tma_launch_param.py`: (1) flags args whose `qword0 ∈ {7 encode bases}`
  (⇒ that arg is a by-value CUtensorMap), (2) joins `param_offset == tensormap_offset`
  (from `extract_tma_descriptor_offsets.py`) to map **(uid,pc) → real base**, writing
  `tma_launch_param_join.json`. Local self-test (2 by-value args, offsets 0x1f0/0x2b0)
  passes end-to-end.

**Gate outcomes (what the next run decides):**
- **PASS** — some arg's qword0 is a real base AND its param_offset matches a site's
  tensormap_offset ⇒ `(uid,pc)→base` recovered on the host. Build the real-address mover
  on `tma_launch_param_join.json`.
- **PARTIAL** — bases present but offsets don't line up ⇒ fix the arg-alignment/packing
  model vs the SASS `c[0x0][K]+UIADD3` offsets.
- **INVESTIGATE** — no arg exposes a base in qword0 ⇒ args are by-pointer; add an offline
  host `cuMemcpyDtoH` of the pointer value (still host-side, still crash-free).

Run recipe (no READ env; classification/dump only, cannot crash):
```
# rebuild tracer, then:
ENABLE_TMA_DESC=1 <existing trace-gen run cmd>
python3 extract_tma_descriptor_offsets.py --traces "$TRACES"   # (uid,pc)->tensormap_offset
python3 verify_tma_launch_param.py       --traces "$TRACES"    # join to base
```

## 3. Design

### 3.1 Trace-gen (NVBit) — capture param_base + static offset, deref the descriptor, dump ALL fields (Path B)

Files: `util/tracer_nvbit/tracer_tool/tracer_tool.cu`,
`util/tracer_nvbit/tracer_tool/inject_funcs.cu`,
`util/tracer_nvbit/discover_tma_producers.py`.

**Where the descriptor actually lives (corrects a common misread).** `c[0x0][0x198]`
does **not** hold the descriptor. It holds a 64-bit **pointer** (`param_base`) into
**device global memory** where the by-value `CUtensorMap` copies sit (§0.05). SASS
proof (§2.10): `ULDC.64` loads `param_base` from the constant bank, then
`[param_base + 0x1f0]` is dereferenced as a **global address** by `UTMACCTL.PF` /
`UTMALDG`. So the descriptor bytes are device-global and are exactly the bytes the TMA
engine reads. This is why *both* deref options below read the identical, authoritative
bytes — they differ only in *who* issues the read, not *which* memory.

The device job is to recover, per executed descriptor-carrying TMA site, the
**descriptor VA** and dump the **whole 128B descriptor** (not just the base), so every
descriptor field comes from **one authoritative source** and all inference is
eliminated. Two ingredients, combined per site:

1. **Static per-pc offset (def-chain, offline).** Trace the descriptor operand
   (per family, §2.6: operand-2 for `UTMALDG`/`UTMAREDG`, operand-1 pair for
   `UTMASTG`, `desc[URx]` for desc-backed `UBLKRED`) back to
   `c[0x0][0x198] + Σ UIADD3 immediates`. This yields a per-pc **key** =
   `0x198 + tensormap_offset`, exactly the 5 values verified in §2.11
   (`0x388/0x448/0x508/0x5c8/0x688`). The global offset to add to `param_base` is
   `(key − 0x198) = tensormap_offset ∈ {0x1f0,0x2b0,0x370,0x430,0x4f0}`.
   **IMPORTANT:** the *executed operand's runtime value is SMEM* (§2.8) and must **NOT**
   be used as the VA — only this static offset is the key. Fold the §2.11 script into
   `discover_tma_producers.py` as a first-class `(uid, pc) → tensormap_offset` emitter.
2. **Runtime `param_base`.** Capture the runtime 64-bit value loaded by
   `ULDC.64 c[0x0][0x198]` **once per launch** (it is grid-uniform). The existing
   `CONSTANT_MEM` / producer-capture path already reads `c[0x0][…]` values, so this is
   cheap.

`descriptor_VA = param_base + (key − 0x198)`. Deref 128B there and emit the 8 qwords
(qword0 = base; rest = dim/stride/box/swizzle/…). The deref needs only **one read per
distinct offset** (5 for fwd), so gate it to first-touch per `(uid, pc)`.

**Deref mechanism (the one open decision — see §2.13):**
- **(primary, "same as HW") device-side deref.** Inject a 128B global load of
  `param_base+offset` at first-touch of each descriptor site; push the bytes on the
  channel. Reads exactly what the TMA engine reads, at kernel-exec time, **no lifetime
  risk**. Needs `inst_trace_t`/channel extension.
- **(alternative) host-side deref.** Capture `param_base` on device, then
  `cuMemcpyDtoH(param_base+offset, 128)` in the launch callback. Reads the **identical**
  device-global bytes; lighter (no channel change). Caveat: param-buffer **lifetime** +
  confirm the VA is host-copyable (Gate 0).
- Either way, **cross-check** the dumped 128B against the host encode blobs
  (`tensor_map_encode_blobs/*.bin`). This both binds `offset → encode row` and validates
  that the encode-time base still equals the run-time base.

Emit to `tma_transfer_addr_debug.csv`, keyed by
`(unique_function_id, pc_hex, tensormap_offset, descriptor_va, cta_x/y/z, warp_id_tb,
sm_id)`, plus the 8 qwords. **Per-transfer coordinates remain uncaptured** (§0) — that
is a separate mini-spike and is only needed for milestone 2 (§3.3).

### 3.2 Trace-gen (python) — decode the 128B descriptor, DELETE all heuristics, assert-on-fail

Files: `util/tracer_nvbit/build_tma_descriptor_mapping.py`.

1. **Decode the dumped 128B directly.** For each executed site, decode its 8 qwords
   into the full descriptor (base, rank, global_dim, global_strides, box_dim,
   element_strides, interleave, swizzle, l2_promotion, oob_fill). This is the exact
   descriptor HW used — no `handle_hi` family lookup, no rank inference, no
   box-family guessing.
2. **Re-key the resolver on `(uid, pc, tensormap_offset)`, not `handle_hi`.** The
   output binding key becomes the def-chain offset (§2.11), which is strictly stronger
   than `handle_hi` (it separates `0x448`/`0x508` that share `desc_hi=0x14f00000`).
   `handle_hi_hex` may stay as a debug column but is no longer part of the key.
3. **Cross-check against the host encode dump (defense-in-depth).** Match the dumped
   128B (or decoded fields) to a row in `tensor_map_encode_dump.csv` /
   `tensor_map_encode_blobs/*.bin`, validating the device decode against host truth.
4. **DELETE every heuristic (per user).** Remove entirely, not merely as "final
   authority": `handle_hi_to_box_dim_family_with_opcode_rank`,
   `single_rank_candidate_config`, `same_function_desc_reg_config_reuse`,
   `*_inferred_rank_from_handle_reuse`, `*_inferred_rank_from_desc_reg_reuse`
   (build_tma_descriptor_mapping.py:152-322). With the descriptor bytes in hand the
   resolver becomes a direct decode, not a matcher; `handle_hi`/`config_id` family
   logic is obsolete.
5. **Assert-on-fail (hard, per user).** Every executed descriptor-carrying site must
   yield exactly one decoded descriptor. If a site has no dumped 128B, a malformed
   blob, or a device-decode that does NOT match any host encode row, `assert`/abort
   with `(uid, pc, opcode, descriptor_va, dumped qwords, nearest encode row)`. No
   silent fallback, no "weakly bound".

### 3.3 Simulator — carry the full decoded descriptor, compute HW address, assert on miss

Files: `tma_types.h`, `gpu-sim.cc` (loader + `lookup_tma_site_metadata`),
`tma_unit_sm.{h,cc}`, and the trace-delivery path
(`address.proto`, `trace_parser.cc`, `trace_driven.cc`, `abstract_hardware_model.h`).

1. **Types (`tma_types.h`).** Add `global_base` (u64) to
   `TMADescriptorConfigMetadata`; change the binding key
   `TMADescriptorLookupKey` from `{uid, pc, handle_hi}` (tma_types.h:66-75) to
   `{uid, pc, tensormap_offset}`; populate `TMACommand.global_base`,
   `TMACommand.global_strides`, and `TMACommand.coords[]` (tma_types.h:250, currently 0).
2. **Trace delivery (only if the offset must travel per-instruction).** The offset is
   static per pc, so it can be joined in the loader from the resolver JSON **without a
   proto change** — prefer that. A proto/`trace_parser`/`trace_driven` field
   (address.proto:5-13 → trace_driven.cc:399) is only needed if per-transfer `coords`
   are later delivered from the trace (milestone 2).
3. **Loader (`gpu-sim.cc:330-438`, `lookup_tma_site_metadata:647-701`).** Parse the
   per-site decoded descriptor (base + all fields) keyed by
   `{uid, pc, tensormap_offset}`; drop `handle_hi` consumption. Set
   `metadata.descriptor_config.global_base`.
4. **Command build (`tma_unit_sm.cc:344-388`).** Copy `global_base`/`global_strides`
   into the `TMACommand`; **assert on lookup miss** when the flag is on (extend the
   existing Phase-2 assert block at tma_unit_sm.cc:415-459) — never silently fall back
   to the synthetic base.
5. **Mover (`tma_unit_sm.cc:633-639`), two milestones.** Replace the synthetic
   `(transfer_uid<<20)` base:
   - **Milestone 1 (base-only, no coords):**
     `agu_base = global_base + agu_index*128`. Kills the cross-SM `(uid<<20)` hotspot.
   - **Milestone 2 (coords):** `+ Σ_d coord[d]*global_stride[d]` once the coord
     mini-spike (§3.1) lands.
   > Accuracy caveat: milestone 1 alone makes every CTA reading the same tensor overlap
   > on `base..base+24KB`, which can push `L2_TMA_true_hit_rate` *higher*, not lower.
   > The hotspot artifact is gone, but real L2 realism needs coords — judge each
   > milestone by §4.2 L2-hit realism, not by cycle sign.
6. **Gated flag** `-tma_real_base_addr_enable` (default off) for clean A/B. When on,
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
3. **Gate 0 — TODO first (§2.12/§2.13):** prove `deref(param_base+offset)` returns the
   correct 128B for **all 5 offsets** (not just 1/7), and pick the deref mechanism
   (device vs host). Its by-product is the `offset → encode-row` binding table. Nothing
   below starts until Gate 0 passes (or Path A fallback is chosen).
4. **Trace-gen NVBit (§3.1):** fold the §2.11 def-chain into
   `discover_tma_producers.py` to emit `(uid,pc)→tensormap_offset`; capture
   `param_base`; dump the site's **128B descriptor** for every descriptor-carrying
   family. **Do not** use the executed operand's runtime value (SMEM, §2.8) as the VA.
5. **Trace-gen python (§3.2):** decode that dumped 128B directly into the final
   descriptor metadata, re-key on the offset, delete all heuristic binding code, and
   `assert` on any missing / malformed / ambiguous mapping.
6. **Simulator (§3.3):** consume the decoded per-site descriptor keyed by offset,
   replace the synthetic base path with real descriptor-derived address generation
   (milestone 1 base-only, then milestone 2 coords), and `assert` on lookup miss when
   the feature is enabled.
7. **Validation:** compare device-dumped 128B against host encode blobs, then run the
   real-base A/B and judge success by L2-hit realism first, not by cycle sign.

*Files touched (summary):* `discover_tma_producers.py`, `tracer_tool.cu`,
`inject_funcs.cu` (+ `common.h`, channel — device-deref only),
`build_tma_descriptor_mapping.py`, `tma_types.h`, `gpu-sim.cc`, `tma_unit_sm.{h,cc}`;
`address.proto` + `trace_parser.cc` + `trace_driven.cc` + `abstract_hardware_model.h`
only if per-transfer coords are delivered (milestone 2).

## 6. Implementation status (host-deref plan, confirmed)

Phased execution. Phase 0 (Gate 0) gates everything else.

**Scope decision (§2.14): land the pipeline on the FWD trace first** — fwd is 100%
clean by-value (`param_base@0x198 + {0x2f0,0x3b0,0x470,0x530}`). bwd is by-pointer +
operand-based and is a separate follow-up (two-level deref; not in scope for Phase 0-3).

| Phase | What | Where it runs | Status |
|---|---|---|---|
| **0a** | def-chain offset extractor `(uid,pc)→tensormap_offset` + operand discovery + executed stats + multi-source fork guard | offline (needs SASS) | **DONE** — `extract_tma_descriptor_offsets.py` (validated on real fwd+bwd; fwd clean) |
| **0d** | runtime operand analyzer (SMEM/GMEM bucket, decides by-value vs by-pointer) | offline | **DONE** — `analyze_tma_runtime_operands.py` (confirmed fwd by-value, bwd by-pointer) |
| **0c** | Gate 0 verify harness: dumped 128B ↔ encode blob, base==qword0, executed-only coverage | offline (pure python) | **DONE** — `verify_tma_descriptor_deref.py` (self-tested: good + 6 failure modes) |
| **0b** | tracer: capture `param_base` (ULDC c[0x0][K] dest, IPOINT_AFTER) + `cuMemcpyDtoH` the param **region** at launch-exit → `tma_param_base_deref.csv` + region blob | GPU server (build tracer) | **CODE READY** — run on FWD trace |
| **1** | run 0a on real **fwd** SASS; confirm all executed UTMALDG resolve to param_base | GPU server | fwd verified in §2.14(a) |
| **2** | python direct-decode + re-key on offset + delete heuristics + assert (fwd) | offline | pending Gate 0 pass |
| **3** | simulator: `global_base` field, offset key, real-addr mover (M1 base-only → M2 coords), assert (fwd) | GPU server | pending Gate 0 pass |
| **bwd** | two-level deref for by-pointer UTMALDG + operand-based UBLKRED | later | **DEFERRED** (§2.14) |

**Gate 0 run recipe — use the FWD trace (`$TRACES_FWD`):**
```
# 1. extract static offsets from FWD SASS (writes extra_info/tma_descriptor_offsets.json)
python3 extract_tma_descriptor_offsets.py --traces $TRACES_FWD
#    → expect EXECUTED ~100% reach param_base, offsets {0x2f0,0x3b0} / {0x470,0x530}
#    → also prints the auto-detected param-base cbank index (0x198 for FA3 fwd)

# 2. rebuild the tracer, then re-run the app with descriptor capture on:
#      ENABLE_TMA_DESC=1
#      TMA_PARAM_BASE_CBANK_INDEX=0x198   # from step 1 if not the default
#      TMA_PARAM_BASE_REGION_BYTES=8192   # >= max(tensormap_offset)+128
#    tracer captures param_base at the entry ULDC and, at each launch-exit (post-sync),
#    cuMemcpyDtoH's the param region → extra_info/tma_param_base_deref.csv
#      + extra_info/tma_desc_deref_blobs/*.bin

# 3. verify: slice each 0a offset out of the region blob, match to an encode blob,
#    check qword0 == real base, executed-only coverage
python3 verify_tma_descriptor_deref.py --traces <kernel_traces_dir> --strict
#    → writes extra_info/tma_descriptor_offset_binding.json (the base binding table)
```
PASS ⇒ proceed to Phase 2/3. FAIL ⇒ Path A fallback (§2.13).

**Design note (why the tracer dumps a region, not per-offset 128B):** the tracer has no
SASS/offset knowledge at runtime, so it dumps one raw param-region blob per launch
(`param_base` + `TMA_PARAM_BASE_REGION_BYTES`). The offline harness slices each 128B
descriptor at `region[tensormap_offset : +128]` using the 0a offsets. This keeps 0b
independent of 0a (no chicken-and-egg) and puts all matching logic in one testable
place.
