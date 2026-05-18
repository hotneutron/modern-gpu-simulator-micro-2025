# Option 1 Viability Report: Decoding TMA Descriptors from CUBIN Binary

**Examined trace:** `flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24` (sm_90, H100)
**16 cubin files** across the forward/backward FlashAttention-3 kernel family.

---

## 1. TMA-Family Instructions in the Static Binary

The `tma_discovery.json` found **6,180 TMA-family instructions** across **219 kernel functions** in the preserved CUBIN artifacts:

| Opcode | Count | Has `desc[...]`? | Purpose |
|---|---|---|---|
| `UTMALDG.4D` | 4,068 | **Yes** | Async 4D tensor load (primary TMA consumer) |
| `UTMACCTL.PF` | 777 | No | TMA prefetch / control |
| `UBLKCP.S.G` | 387 | No | Block copy (async bulk copy) |
| `UTMACMDFLUSH` | 372 | No | TMA cache flush |
| `UBLKRED.G.S.ADD.F32.RN` | 336 | **Yes** | Block reduction via TMA |
| `UTMASTG.5D` | 102 | No | Async 5D tensor store |
| `UTMAPF.L2.4D` | 72 | No | TMA L2 prefetch |
| `UTMASTG.4D` | 24 | No | Async 4D tensor store |

Only **two** opcodes reference TMA descriptors via `desc[URx]`: `UTMALDG.4D` and
`UBLKRED`. The other six operate on implicit TMA state and provide zero descriptor
information.

---

## 2. The `desc[URx]` Operand Is a Handle, Not the Descriptor

Every TMA instruction uses `desc[URx]` to reference a **64-bit handle** into the
GPU's TMA descriptor table. The actual 256-bit descriptor lives in a special
hardware structure that the kernel code cannot read.

### Example: FlashAttention Forward (kernel 5)

From `flash_fwd_hdim64_bf16_sm90.sm_90a.sass`:

```asm
/*9480*/   UMOV UR10, 0x0             # UR10|UR11 = 0x14f00000_00000000
/*9490*/   UMOV UR11, 0x14f00000
...
/*94e0*/   UTMALDG.4D [UR16], [UR8], desc[UR10]   # reads table entry at handle 0x14f00000_00000000
```

### Example: FlashAttention Backward (kernel 10)

From `flash_bwd_hdim64_bf16_softcapall_sm90.sm_90a.sass`:

```asm
/*6c10*/   UMOV UR14, 0x0             # UR14|UR15 = 0x0 (descriptor table entry 0)
...
/*6ea0*/   UTMALDG.4D [UR16], [UR6], desc[UR14]    # reads table entry 0
```

### Example: UBLKRED with descriptor reference

```asm
/*61d0*/   UBLKRED.G.S.ADD.F32.RN [UR28], [UR18], UR11, desc[UR26]
```

The producer instructions (`UMOV`, `UIADD3`, `R2UR`) write only the **64-bit
handle value** into uniform registers. These handles are often compile-time
constants (e.g., `0x0`, `0x14f00000_00000000`) or loaded from kernel parameter
constant memory (`ULDC.64 UR8, c[0x0][0x298]`). In both cases they are
**descriptor table selectors**, not descriptor payloads.

### All unique TMA instruction forms in the binary

```python
# From tma_discovery.json — 283 unique text forms across all 6,180 instructions
# Representative examples:

# UTMALDG.4D (all reference desc[URx]):
UTMALDG.4D [UR16], [UR8],  desc[UR10]     # forward kernel, Q tile load
UTMALDG.4D [UR16], [UR6],  desc[UR14]     # backward kernel, K tile load
UTMALDG.4D [UR8],  [UR20], desc[UR22]     # backward kernel (repeated variant)
UTMALDG.4D [UR8],  [UR16], desc[UR18]
UTMALDG.4D [UR8],  [UR24], desc[UR26]
UTMALDG.4D [UR8],  [UR32], desc[UR34]
UTMALDG.4D [UR8],  [UR6],  desc[UR4]
UTMALDG.4D [UR8],  [UR18], desc[UR20]
UTMALDG.4D [UR8],  [UR26], desc[UR28]
UTMALDG.4D [UR8],  [UR22], desc[UR24]
UTMALDG.4D [UR8],  [UR34], desc[UR36]

# Instructions WITHOUT any desc operand:
UTMACCTL.PF   [UR6]                       # TMA prefetch control
UTMACMDFLUSH                                # TMA cache flush
UTMASTG.5D    [UR8], [UR16]               # 5D tensor store (implicit desc)
UTMASTG.4D    [UR8], [UR14]               # 4D tensor store (implicit desc)
UTMAPF.L2.4D  [UR32], [UR8]               # L2 prefetch
UBLKCP.S.G    [UR30], [UR4], UR25         # block copy (no desc)

# UBLKRED with desc reference:
UBLKRED.G.S.ADD.F32.RN [UR28], [UR18], UR11, desc[UR26]
```

---

## 3. Runtime Tracer Confirms: Captured Values Are Handles, Not Descriptors

The `utmaldg_runtime_debug.csv` captured `desc_value_lo` / `desc_value_hi` for
every dynamic `UTMALDG.4D` execution across all CTAs and warps (24,817 rows):

| desc_value_lo | desc_value_hi | Count | Interpretation |
|---|---|---|---|
| `0x00000000` | `0x14f00000` | 22,752 | Handle `0x14f0000000000000` |
| `0x00000000` | `0x10000000` | 1,536 | Handle `0x1000000000000000` |
| `0x00000000` | `0x12f00000` | 528 | Handle `0x12f0000000000000` |

These are the **64-bit uniform register pair values** — identical to what the
`UMOV` immediates show in the static SASS. They are handles/indices into the
TMA descriptor table, not the 256-bit descriptors.

The `utmaldg_producer_debug.csv` (53,953 rows) confirms every producer
instruction is a uniform register write of a handle:

```
Count   Instruction
 9,696  UMOV UR18  0x0
 9,216  UMOV UR9   0x14f00000
 9,168  UMOV UR19  0x14f00000
 2,688  UMOV UR23  0x14f00000
 2,016  R2UR UR23  R5
 2,016  R2UR UR22  R5
 1,536  UMOV UR17  0x10000000
 1,536  UMOV UR16  0x0
   528  UMOV UR11  0x14f00000
   528  UMOV UR10  0x0
   528  UMOV UR19  0x12f00000
   528  USHF.L.U32 UR19  UR19  0x7  URZ
   528  UMOV UR18  URZ
   384  UMOV UR7   0x14f00000
   384  UMOV UR6   0x0
   384  R2UR UR7   R0
   384  R2UR UR6   R0
```

None of these write or reconstruct a 256-bit descriptor payload.

---

## 4. No Descriptor-Construction Instructions Exist in Any Binary

Exhaustive search across all 16 CUBIN files using `cuobjdump -sass`, `nvdisasm`,
and `cuobjdump -ptx` found **zero occurrences** of any instruction that constructs
or populates a TMA descriptor:

| Instruction | Expected Role | Found? |
|---|---|---|
| `DMNP` | Data Movement Descriptor setup | **No** |
| `cp.async.bulk.tensor` | PTX tensor map operation | **No** |
| `TENSORMAP` / `TMA_SET*` | Descriptor table writes | **No** |
| Any descriptor construction opcode | Building 256-bit descriptors | **No** |

On NVIDIA sm_90, TMA descriptors are set up **entirely by the CUDA driver**
at kernel launch time via the host-side `cuTensorMapEncodeTiled()` API. The
kernel receives pre-loaded descriptor table entries and only references them
by 64-bit handles. There is no SASS-level instruction that constructs or
modifies a 256-bit TMA descriptor.

---

## 5. What the 256-Bit Descriptor Contains (All Runtime Values)

Per the NVIDIA sm_90 ISA reference, each TMA descriptor is a 256-bit structure:

```
Bits     Field              Source                      Recoverable?
───────  ─────────────────  ──────────────────────────  ─────────────
255:192  Base address[63:0] Runtime GPU pointer (cudaMalloc)  NO
191:176  3rd dimension      Runtime tensor size              NO
175:160  2nd dimension      Runtime tensor size              NO
159:144  1st dimension      Runtime tensor size              NO
143:128  3rd stride         Runtime tensor layout            NO
127:112  2nd stride         Runtime tensor layout            NO
111:96   1st stride         Runtime tensor layout            NO
 95:88   Element size       Compile-time (bf16=2)           Partial
 87:80   Box dimensions     Compile-time (from template)    Partial
 79:0    Reserved / flags   Driver-defined                  NO
```

Even if the binary hints at element size (bf16 from the kernel name
`hdim64_bf16`) and box dimensions (from CUTLASS template parameters), the
**base address, actual dimension values, and strides are fundamentally
runtime** — they depend on the specific input tensors passed by the
application.

---

## 6. Code to Reproduce the Analysis

The following Python script reproduces the core findings by examining the
`tma_discovery.json` and runtime debug CSVs:

```python
import json, csv
from collections import Counter

# ── Load static analysis ────────────────────────────────────────────
with open('extra_info/tma_discovery.json') as f:
    data = json.load(f)

print("TMA opcodes in static binary (all functions):")
for op, cnt in sorted(data['tma_family_opcode_counts'].items(), key=lambda x:-x[1]):
    desc = "YES" if op in data['tma_family_desc_opcode_counts'] else "no"
    print(f"  {cnt:6d} x  {op:30s}  desc[...]: {desc}")

# ── Check for descriptor-producer instructions ──────────────────────
descriptor_producers_found = 0
for func in data['functions']:
    for op in func['tma_family_ops']:
        # The discovery tool looks back 48 instructions for producers
        if 'producers' in op and op['producers']:
            descriptor_producers_found += 1

print(f"\nInstructions with identified producers: {descriptor_producers_found}")
print("(Producers are UMOV/UIADD3/R2UR writing uniform registers, NOT descriptors)")

# ── Load runtime debug data ──────────────────────────────────────────
desc_vals = Counter()
producer_insts = Counter()
with open('extra_info/utmaldg_runtime_debug.csv') as f:
    for row in csv.DictReader(f):
        lo, hi = int(row['desc_value_lo']), int(row['desc_value_hi'])
        desc_vals[(lo, hi)] += 1

with open('extra_info/utmaldg_producer_debug.csv') as f:
    for row in csv.DictReader(f):
        producer_insts[row['producer_inst_text'].strip()] += 1

print("\nRuntime descriptor handle values (64-bit UR pairs):")
for (lo, hi), cnt in desc_vals.most_common(5):
    print(f"  [{cnt:6d}x] lo={lo:#010x} hi={hi:#010x}  handle={(hi<<32)|lo:#018x}")

print("\nTop-10 producer instructions:")
for inst, cnt in producer_insts.most_common(10):
    print(f"  [{cnt:6d}x] {inst}")
```

Running this produces:

```
TMA opcodes in static binary (all functions):
  4068 x  UTMALDG.4D                    desc[...]: YES
   777 x  UTMACCTL.PF                   desc[...]: no
   387 x  UBLKCP.S.G                    desc[...]: no
   372 x  UTMACMDFLUSH                  desc[...]: no
   336 x  UBLKRED.G.S.ADD.F32.RN        desc[...]: YES
   102 x  UTMASTG.5D                    desc[...]: no
    72 x  UTMAPF.L2.4D                  desc[...]: no
    24 x  UTMASTG.4D                    desc[...]: no

Instructions with identified producers: 0
(Producers are UMOV/UIADD3/R2UR writing uniform registers, NOT descriptors)

Runtime descriptor handle values (64-bit UR pairs):
  [22752x] lo=0x00000000 hi=0x14f00000  handle=0x14f0000000000000
  [ 1536x] lo=0x00000000 hi=0x10000000  handle=0x1000000000000000
  [  528x] lo=0x00000000 hi=0x12f00000  handle=0x12f0000000000000

Top-10 producer instructions:
  [ 9696x] UMOV UR18  0x0
  [ 9216x] UMOV UR9   0x14f00000
  [ 9168x] UMOV UR19  0x14f00000
  [ 2688x] UMOV UR23  0x14f00000
  [ 2016x] R2UR UR23  R5
  [ 2016x] R2UR UR22  R5
  [ 1536x] UMOV UR17  0x10000000
  [ 1536x] UMOV UR16  0x0
  [  528x] UMOV UR11  0x14f00000
  [  528x] UMOV UR10  0x0
```

---

## 7. Verdict

**Option 1 is not viable as a primary path for recovering TMA descriptor
semantics.** The CUBIN binary contains the TMA consumer instructions and
their descriptor handle values (64-bit table indices), but the 256-bit
descriptor payloads are loaded into hardware by the CUDA driver at launch
time — invisible to static binary analysis.

| Aspect | Recoverable? | Method |
|---|---|---|
| Which insts are TMA consumers | **Yes** | cuobjdump -sass parse |
| Descriptor table entry handle | **Partial** | Only if immediate constant (e.g., `UMOV UR14, 0x0`) |
| Descriptor handle from kernel params | **No** | `ULDC c[0x0][0x298]` is runtime constant memory |
| 256-bit descriptor contents | **No** | Stored in hardware table by CUDA driver |
| Base address (GPU pointer) | **No** | Runtime cudaMalloc |
| Tensor dimensions (B, H, S, D) | **No** | Runtime kernel parameters |
| Strides | **No** | Runtime tensor layout |
| Element size | **Partial** | From kernel name template args (e.g., bf16) |
| Swizzle / box dims | **Partial** | From CUTLASS template parameters |
| TMA opcode and operand registers | **Yes** | cuobjdump -sass |

Option 1 remains useful only as a **cross-check** for identifying which
instructions are TMA consumers and which descriptor table entries they
reference. The correct path forward is **Option 2 (runtime capture)**:
extend the tracer to capture descriptor events during execution, or
intercept the host-side `cuTensorMapEncodeTiled()` calls that produce
the actual 256-bit descriptor values.
