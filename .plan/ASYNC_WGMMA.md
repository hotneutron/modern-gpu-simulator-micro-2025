# ASYNC_WGMMA — model WGMMA re-issue as async (fwd/bwd tensor lever)

> ## ⭐ TL;DR / FINAL CONCLUSION (2026-07-16) — READ THIS FIRST
>
> **The sim is ALREADY effectively async, and any remaining latency/II mismatch is a CONFIG change,
> not a code change. This whole "async-WGMMA" investigation did NOT find a code lever.** Concretely,
> verified in source (see §11):
> 1. **Producer is already free during the WGMMA's latency.** The issue gate `can_issue()` only checks
>    the II lockout (`m_dispatch_pending_reserved_cycles`, [functional_unit.cc:137-139](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L137-L139)); a
>    non-tensor next inst goes to a different FU and is not blocked at all; the consumer waits only on
>    the `gsb0` write-barrier (DEPBAR), which is real data dependency (HW waits too). So "producer runs
>    ahead during latency" — the defining property of async — **already holds**.
> 2. **latency/II are just numbers.** completion = `latency + II = number_of_cycles`. To make them match
>    HW (latency ~64–128, §9) you only need to raise the config `-tensor_latency` (= the tensor FU
>    pipeline depth) and/or adjust the `number_of_cycles/2` split. **No async machinery required.**
> 3. **Design A (II divisor) was reverted** — it tried to hold completion constant by inflating
>    `latency` past the fixed-pipe depth cap (`tensor_latency=32`) and hit an assert (§8). Structurally
>    impossible in the fixed-latency FU; and per §9 the "lower II" premise was falsified anyway
>    (HW II ≈ 72 > sim II = 32).
> 4. **Design B (background completion) gives the SAME timing as raising the config** — it only differs
>    in implementation structure (a background list vs. a deeper shift-register pipe), i.e. "write down
>    when it finishes and count down" ≈ "make the pipe deeper". Its only real value is future-structure
>    (warpgroup/cluster), NOT cycles. **Not implemented; parked.**
> 5. **The real gap is a DIFFERENT problem: no warpgroup execution model → WGMMA is run 4× (once per
>    warp of the warpgroup, each computing the full tile).** That is [WARP_GROUP_H100.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/WARP_GROUP_H100.md)
>    and FA3_progress Ongoing item 4 — the current active lever. async does NOT fix it.
>
> **Net: async-WGMMA is closed as "already modeled / config-tunable". Next work = warpgroup 4×.**

> **Primary target: bwd K10.** The per-CTA correlation ([CTA_FINISH_TENSOR_CORRELATION.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/CTA_FINISH_TENSOR_CORRELATION.md))
> showed bwd's CTA-imbalance (58% elapsed spread) is **near-perfectly** correlated with the tensor
> re-issue lockout (`r(sm_idle_tensor_cyc, elapsed_cyc)=+0.99`; slowest-decile CTAs carry 11.6× the
> tensor idle of the fastest). fwd is weakly coupled (r=0.38) — not its lever. Tracked in
> [FA3_progress.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.result/FA3_progress.md) Ongoing item 3 (Suspect #1).

## 1. What the sim does today (synchronous WGMMA)

WGMMA (`TENSOR_CORE_OP`) is dispatched into `m_tensor_pipeline`, a **fixed-latency `functional_unit`**
(created `SPECIALIZED__OP`, `has_queue=false`, [subcore.cc:1469](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1469)). Per-instruction latencies are
computed in [generate_tensor_core_latencies()](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L429):
`number_of_cycles = M·N·K·bits / tensor_rate_per_cycle`, then **`initiation_interval = number_of_cycles/2`,
`latency = number_of_cycles − II`**. Measured this run:

| WGMMA shape | number_of_cycles | II | latency |
|---|---:|---:|---:|
| m64n128k16 bf16 (HGMMA) | 64 | **32** | 32 |
| m64n64k16 bf16 | 32 | 16 | 16 |

Two mechanisms serialize the **producer** (the warpgroup issuing back-to-back WGMMAs):

1. **Re-issue lockout** — on issue, `reserve_unit()` sets `m_dispatch_pending_reserved_cycles = II`
   ([functional_unit.cc:129](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L129)); `can_issue()` returns false until it counts to 0
   ([functional_unit.cc:137-139](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L137)). So a warpgroup cannot launch the next WGMMA for **II=32
   cycles**. This is exactly the `sm_idle_tensor_cyc` / `tensor_reissue_lockout_only` signal.
2. **Latency-bitset reservation** — `allocate()` reserves `target = read(6) + latency(32) + II(32) = 70`
   slots in the FU `occupied` bitset ([subcore.cc:300](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L300),[:326](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L326)); a conflict extends
   the lockout via `add_extra_cycle_initiation_interval()` ([subcore.cc:329](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L329)).

The **consumer** (a later inst that reads the WGMMA accumulator) waits on the fixed `latency` (32) via
the normal scoreboard/writeback path ([functional_unit.cc:244-246](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L244) → writeback → `instruction_retirement`).
There is **no separate consumer-side warpgroup_arrive event** in trace mode (it folds into wait_barrier);
that is why `warpgroup_arrive_cyc` was 0 and we measure the producer signal instead.

## 2. Why this is wrong vs H100

Real Hopper `wgmma.mma_async` is **asynchronous**: the warpgroup *issues* the WGMMA into the tensor core
and continues; the tensor core runs it in the background at the true tensor-core throughput; the
warpgroup only blocks later at `wgmma.wait_group`. The effective **issue-to-issue interval** on HW is the
tensor-core pipeline throughput (a few cycles), **not 32**. The sim's II=32 (= number_of_cycles/2) makes
the producer serialize as if each WGMMA occupies the issue slot for half its whole compute time. This
over-serialization is:
- fwd: `mma` (fu_occupied_tensor) 5.65% vs HW 1.4% (4×), `math_pipe` 11% vs 3.2% (3.4×);
- bwd: `mma` 12.5% vs HW 5.3% (2.4×), `math_pipe` 9.6% vs 1.2% (8×) — and it is what drives the bwd
  drain-tail (r=0.99).

## 3. The fix — decouple the re-issue interval from the compute latency

**Key insight:** the sim conflates two separate HW quantities into `number_of_cycles/2`:
- **issue throughput II** — how soon the *next* WGMMA can be issued (HW: small, pipelined);
- **completion latency** — when the *result* is ready for the consumer (HW: the full compute time).

Today both are ≈32. HW has small II, full latency. The fix is to **lower the re-issue II while keeping
the completion latency** so the consumer dependency is unchanged (result still takes ~number_of_cycles),
but the producer can pipeline back-to-back WGMMAs.

### Design A (recommended, minimal, bit-identity-safe): a config-gated II scale

Add `-wgmma_async_issue_interval_divisor N` (default **1** = today's behavior, bit-identical).
**Chosen value for the next run: N=4.** When N>1, in `generate_tensor_core_latencies()`
([abstract_hardware_model.cc:439](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L439)) split:

```
completion_cycles = number_of_cycles;            // consumer waits on this (unchanged)
initiation_interval = max(1, (number_of_cycles/2) / N);   // producer re-issue (shrunk)
latency = completion_cycles - initiation_interval;         // keep issue->result == number_of_cycles
```

- `initiation_interval` feeds `m_dispatch_pending_reserved_cycles` (the lockout) → **directly shrinks the
  producer stall** we measured.
- `latency + II` (the bitset reservation `target`, [subcore.cc:300](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L300)) stays == number_of_cycles, so the
  **consumer sees the same completion time** and the tensor pipe's total occupancy is unchanged (no fake
  compute speedup — only the *serialization* is removed, matching async issue).
- **Default N=1 → byte-identical to current** (gate off = no behavior change). This preserves the
  135,999 / 215,895 baselines when the knob is absent.

**Exact effect at N=4 (computed from this run's measured shapes):**

| WGMMA shape | N | II | latency | bitset target (=6+lat+II) | issue→result |
|---|---:|---:|---:|---:|---:|
| m64n128k16 (HGMMA, bwd/fwd dominant) | 1 | 32 | 32 | 70 | 64 |
| m64n128k16 | **4** | **8** | **56** | **70** | **64** |
| m64n64k16 | 1 | 16 | 16 | 38 | 32 |
| m64n64k16 | **4** | **4** | **28** | **38** | **32** |

The **bitset target and issue→result are IDENTICAL** for N=1 and N=4 — only the producer re-issue
interval shrinks (32→8 for the dominant HGMMA), so this is a pure de-serialization with no change to
completion time or total tensor occupancy. Expected: the ~11.8% bwd `tensor_reissue_lockout` and the
12.5% `mma` share should drop toward the HW anchors (bwd `mma` 5.3%, `math_pipe` 1.2%).

**Why N=4 as the first point** (not a full sweep): sim II=32 for the dominant HGMMA; an HW-plausible
issue throughput is ~8 cyc → divisor 4. It is the single most likely calibration point, and one 12h run
is expensive. If N=4 overshoots (sim `mma` < HW 5.3%) or undershoots, adjust N next round — the knob
makes that a one-line config change, no rebuild.

### Design B (heavier, only if A under-delivers): true background completion

Convert WGMMA to a `functional_unit_with_queue` + background `in_flight` list like TMA
([tma_unit_sm.cc:680](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L680) advance + [sm.cc:1568](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1568) async completion), releasing the
consumer via an event instead of fixed latency. This is the "correct" async model but is a large change
and risks perturbing the scoreboard/writeback path. **Defer** unless A cannot hit the HW anchors — A
already removes the exact serialization the measurement implicated.

### Strategy decision (user, 2026-07-16): A first, then B if A works

The user's goal is a **correct architecture design**, not just a quick experiment. The agreed sequence is:

1. **Implement Design A now** (config-gated producer-II divisor) — DONE (see §7 Status). It is the minimal,
   bit-identity-safe change that directly targets the *only* measured tensor stall (producer re-issue
   lockout, bwd r=0.99).
2. **Run N=4 and check against the HW anchors** (§6). Design A is the measurement gate that tells us
   whether the producer-side fix alone is sufficient.
3. **If Design A is effective** (bwd `mma` moves toward HW 5.3%, `math_pipe` toward 1.2%, cycles drop
   toward the anchor) → **proceed to implement Design B** as the proper full-async architecture
   (background completion + event-driven consumer release), using A's result to size the remaining gap.
4. If Design A under-delivers, re-examine before committing to B (the confound in
   [CTA_FINISH_TENSOR_CORRELATION.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.plan/CTA_FINISH_TENSOR_CORRELATION.md) §Results must be cleared by the post-fix `[CTAFIN]` re-run
   that N=4 also produces).

So B is **not** cancelled — it is the intended end state for the correct async model. A is the safe,
measured first step that both delivers a cycle result and de-risks B.

## 4. Why Design A is the right first step

- **Surgical:** one knob, one formula site, one gate. No pipeline-structure rewrite.
- **Bit-identity-safe:** N=1 default = current. The 12h baselines are protected.
- **Targets the measured signal directly:** the II *is* `m_dispatch_pending_reserved_cycles` = the
  producer lockout = `sm_idle_tensor_cyc` = the bwd r=0.99 driver.
- **No fake win:** completion latency (consumer dependency) and total tensor-pipe occupancy are held ≈
  constant; only the issue-slot serialization shrinks — which is precisely what async issue removes.

## 5. Implementation checklist (verified insertion sites — for the NEXT session)

1. **Register knob** `-wgmma_async_issue_interval_divisor` (OPT_INT32, default **1**) in
   `gpgpu_sim_config::reg_options`, next to `-tensor_rate_per_cycle` at
   [gpu-sim.cc:1676](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L1676). Bind to a new `int wgmma_async_issue_interval_divisor;` field in
   `shader_core_config` next to `tensor_rate_per_cycle` at [shader.h:2141](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h#L2141).
2. **Apply the split** in `generate_tensor_core_latencies()` at
   [abstract_hardware_model.cc:439-440](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L439):
   ```
   int N = shader_config.wgmma_async_issue_interval_divisor;   // default 1
   initiation_interval = (N > 1) ? std::max(1u, (number_of_cycles/2) / (unsigned)N)
                                 : number_of_cycles/2;          // N==1 -> EXACT current value
   latency = number_of_cycles - initiation_interval;
   ```
   Keep the `is_sparse` halving ([:437](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L437)) and the `is_16816_fp32` extra ([:441-444](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L441)) applied
   AFTER the split exactly as today, so those paths are untouched at N=1.
3. **No bitset change needed:** `target = read + latency + II` is invariant under the split (verified:
   70 and 38 unchanged), and it only ever *decreases* vs today for N>1, so the `<512` guard
   ([subcore.cc:302](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L302)) cannot trip. Just confirm `latency ≥ 1` (holds: 56, 28).
4. **Extend `[WGMMADBG-MNK]`** ([abstract_hardware_model.cc:457-461](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc#L457)) to also print N and the
   resulting II so the applied divisor is verifiable in the log.
5. **Config for the run:** set `-wgmma_async_issue_interval_divisor 4` in
   [gpgpusim.config](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config) (value only, no comment edits). Keep
   `-wgmma_step0_instrument_enable 1` so `[CTAFIN]` + NCU taxonomy come out for verification.

**Header changed (shader.h) → `make clean` required.**

## 6. Verification plan for the N=4 run

1. **Bit-identity pre-check (cheap):** a throwaway N=1 build/run must reproduce fwd 135,999 / bwd 215,895
   exactly — proves the knob is truly gated. (Or trust the `N==1` branch = current formula and skip.)
2. **N=4 acceptance (the payoff):** on fwd `.oNN` / bwd `.oNN`, read the NCU taxonomy + `[CTAFIN]`:
   - **Primary (bwd):** `mma` 12.5% → toward HW **5.3%**; `math_pipe` 9.6% → toward **1.2%**;
     `tensor_reissue_lockout` 11.8% → down ~4×; `warp-cyc/issued` 8.6 → toward HW **7.53**;
     total bwd cycles 215,895 → down (the tail shrinks).
   - **Per-CTA (bwd):** re-run the correlation — `sm_idle_tensor_cyc` slow-decile should collapse toward
     the fast-decile (the 11.6× gap narrows), and `elapsed_cyc` spread (58%) should shrink.
   - **fwd:** expected small improvement only (weak coupling); watch it does not regress and `mma`
     1.4%-HW is not overshot.
3. **No-fake-win check:** `gpu_sim_insn`, `tensor_ops` totals, and `L2_TMA_true_hit_rate` must be
   unchanged (work-invariant); only cycles/stall-shape move.

## 7. Status

- [x] Root cause mapped (synchronous II=32 producer lockout; consumer folds into fixed latency).
- [x] Design chosen: Design A (config-gated II divisor), default-off bit-identity-safe. **N=4 selected.**
- [x] Insertion sites verified; per-shape N=4 effect computed (II 32→8, target/completion unchanged).
- [x] Strategy agreed with user (2026-07-16): implement A now → if effective, implement Design B as the
      correct full-async architecture (§3 "Strategy decision").
- [x] Implement the 5-step checklist (2026-07-16). Verified against current source
      (`/home/jihyun/...` tree; the doc's `/Users/bytedance/...` line numbers are stale but the sites
      matched by symbol):
  - `int wgmma_async_issue_interval_divisor;` added to `shader_core_config` next to
    `tensor_rate_per_cycle` ([shader.h](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h)).
  - `-wgmma_async_issue_interval_divisor` (OPT_INT32, default 1) registered in
    `shader_core_config::reg_options` next to `-tensor_rate_per_cycle`
    ([gpu-sim.cc](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc)).
  - II/latency split applied in `generate_tensor_core_latencies()` after the `is_sparse` halving and
    before the `is_16816_fp32` extra, using `std::max(1u, (number_of_cycles/2)/N)` for N>1 and the
    EXACT current value for N==1 ([abstract_hardware_model.cc](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc)).
  - `[WGMMADBG-MNK]` log extended with `async_div=%d` so the applied divisor is verifiable.
  - `-wgmma_async_issue_interval_divisor 4` set in
    [gpgpusim.config](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config).
  - **Header changed (shader.h) → `make clean` required before rebuild.**
- [ ] Run N=4; verify against HW anchors (§6); record in FA3_progress.md.
- [ ] If effective → implement Design B (true background completion).

## 8. Design A N=4 run FAILED (2026-07-16) — structural assert

The N=4 run crashed (bwd `.e41`):

```
[WGMMADBG-MNK] m=64 n=128 k=16 bits=16 sparse=0 number_of_cycles=64 async_div=4 II=8 latency=56
accel-sim.out: functional_unit.cc:308: virtual void functional_unit::cycle():
    Assertion `(unsigned)start_stage < m_pipeline_depth' failed.
```

**Root cause (structural, not a tuning bug).** The fixed-latency FU uses `latency` as a **pipeline
STAGE INDEX**, not a countdown:
- [functional_unit.cc:306-308](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L306-L308):
  `int start_stage = m_dispatch_reg->latency - 1; assert((unsigned)start_stage < m_pipeline_depth);`
- The tensor FU is created with `m_pipeline_depth = m_config->tensor_latency = 32`
  ([subcore.cc:1469](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L1469); depth set at [functional_unit.cc:58](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L58)).

So **no WGMMA `latency` can exceed 32** in this model. Design A's `latency = number_of_cycles - II`
grows to 56 when II shrinks to 8 → `start_stage=55 ≥ 32` → assert. The original `latency=II=32` was
not a coincidence: it is the maximum the 32-deep shift-register pipeline can hold. **Design A is
structurally impossible here** — you cannot hold completion time constant by inflating `latency` when
the pipe depth caps `latency` at `tensor_latency`. This is why we move to Design B. (The knob/formula
code is left in place, default N=1 = bit-identical; the config was the only thing that made it fire.)

## 9. HW-measured WGMMA latency + II back-calculation (2026-07-16)

> Source: arXiv:2501.12084 *Dissecting the NVIDIA Hopper Architecture* (the same paper already used as
> the HW anchor for Opt 7 TMA bandwidth). Device = **H800** (Hopper): 114 SM, 1755 MHz, FP16 peak
> 756.5 TFLOPS. §6.2 Table 8 (dense wgmma) + line 1143.

### 9.1 Measured (Table 8, dense, SS/Zero — bf16/FP16 path matches this kernel)

| A/B | C/D | shape | LAT (cyc) | throughput (TFLOPS) | notes |
|---|---|---|---:|---:|---|
| FP16 | FP32 | m64n256k16 | **128.0** | 728.5 | this kernel's accum type (bf16→fp32 ≈ FP16→FP32 pipe) |
| FP16 | FP16 | m64n256k16 | 128.0 | 729.3 | — |

Paper line 1143: "with N set to 128, the **latency for all data types is 128.0**." (Table only lists
N=256; the quoted line says the latency is the same at N=128.) Note the paper defines LAT as
**completion latency** = issue → result-usable (line 904).

### 9.2 II back-calculation from throughput (verified numerically)

OPS/instruction = `2 · M · N · K`. instructions/s = TFLOPS / OPS. II(cyc/inst/SM) =
clock / (inst/s / SM). With clock=1755 MHz, SM=114, throughput 728.5 TFLOPS:

| shape | OPS/inst | **effective II (cyc/inst/SM)** | ideal cyc @ peak (756.5) |
|---|---:|---:|---:|
| m64n256k16 FP16→FP32 | 524,288 | **144** | 138.7 |
| m64n128k16 (this kernel, scaled ½) | 262,144 | **72** | 69.3 |

### 9.3 ⚠️ The decisive finding — HW II ≈ HW latency (near-serial), and it CONTRADICTS the "lower II" premise

- **HW `m64n128k16`: latency ≈ 64–72, effective II ≈ 72.** The two are ~equal. A single m64n128k16 is
  262,144 MAC and the TC peak is ~1890 MAC/cyc/SM, so one instruction physically occupies the tensor
  core for ~69 cycles. **There is almost no pipeline overlap headroom** — one warpgroup already ~96%
  saturates the TC. So real Hopper wgmma is **NOT** "small II, big latency"; for these large shapes it
  is "II ≈ latency ≈ number_of_cycles".
- **sim today: II=32, latency=32** (number_of_cycles/2 each). So the sim's **II=32 is actually SMALLER
  than HW's ~72** — the sim already re-issues WGMMA *more* often than HW, not less.
- **This breaks the original Ongoing-item-3 premise** that fwd/bwd are slow because the sim over-serializes
  the WGMMA producer (II too big). By the throughput math the sim's II is if anything too *small*.
- **sim number_of_cycles=64 ≈ HW ideal 69.3** — the *total* work-cycle estimate is right; only its
  split into II/latency (32/32) and its mapping onto a 32-deep pipeline are the modeling artifacts.

### 9.4 What this means for the design (to resolve before implementing Design B)

The measured `mma` over-model (sim 12.5% vs HW 5.3% bwd) must therefore come from something OTHER than
"II too large". Candidate re-interpretations to check next:
- The sim's completion latency is **32, but HW is ~64–128** → the sim finishes WGMMA results too *early*,
  which would make the consumer wait *less*, not more — so latency-under-model cannot explain a *higher*
  sim stall either. The `mma`/`math_pipe` over-count may be a **counter-attribution** effect, or driven
  by the FU `occupied`-bitset structural-hazard model rather than the II/latency values.
- Need to separate: (a) is the extra sim `mma` stall real added cycles, or a taxonomy mis-attribution?
  (b) does the near-serial HW reality mean the correct sim change is to set **latency≈number_of_cycles,
  II≈number_of_cycles** (i.e. LARGER, more serial) rather than smaller?
- Design B (background completion) is still the structural vehicle (it removes the 32-deep-pipe cap so
  latency can be the true ~64–128), but the **direction of the II change is now open** and must be
  settled against the taxonomy before coding.

## 10. Design B IMPLEMENTATION PLAN — true async WGMMA (chosen 2026-07-16)

> Decision (user): implement **real async** (interpretation 2), not "just widen the pipe depth"
> (interpretation 1). Rationale: interpretation 1 keeps the fixed-latency shift-register semantics,
> which will fight future warpgroup / cluster work. A proper background-completion unit is the correct
> long-term structure. Design A is reverted (§8). The warpgroup-4× issue (WARP_GROUP_H100.md) is
> PARKED and NOT addressed here — async keeps the current per-warp execution.

### 10.1 What already exists (reuse, do NOT rebuild)

The consumer side is **already** a proper async barrier mechanism in this config
(`is_remodeling_scoreboarding_enabled=0`, trace mode, captured-from-binary ⇒ `use_traditional_scoreboarding=false`):
- **Producer registers a write barrier** at `Subcore::control_stage` when the op has a new write
  barrier ([subcore.cc:358-360](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L358-L360) → `add_pending_wait_barrier_increment(WRITE_WAIT_BARRIER, id)`).
  This maps to the SASS `HGMMA ... gsb0`.
- **Completion decrements it** at `SM::instruction_retirement` else-branch
  ([sm.cc:682-688](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L682-L688) → `add_pending_wait_barrier_decrement(WRITE_WAIT_BARRIER, id)`).
- **Consumer waits** via `WARPGROUP.DEPBAR.LE gsb0, N` = `DEPBAR_OP`, checked by
  `wait_barriers_to_check_depbar` ([subcore.cc:981-1002](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L981-L1002)) and gated at issue.

**So the ONLY thing wrong is WHEN the write-barrier decrement fires:** today it is tied to the WGMMA
walking off stage 0 of the 32-deep fixed pipe (`instruction_finishing_execution` → `rf_write_queue` →
`writeback` → `instruction_retirement`). Design B must fire the SAME decrement from a **background
completion event scheduled at `issue_cycle + completion_cycles`**, and NOT route the WGMMA through the
fixed shift-register pipe at all.

### 10.2 Core design — a per-SM background WGMMA completion list

Model after TMA's `m_in_flight_transfers` + `advance()` + `notify_tma_completion`
([tma_unit_sm.cc:680-738](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L680-L738), [sm.cc:1934-1964](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L1934-L1964)), but simpler — WGMMA
completion is a pure cycle countdown (no memory response).

**New structure (SM-level):**
```cpp
struct InFlightWgmma {
  unsigned warp_id;
  unsigned long long completion_cycle;   // issue_cycle + completion_cycles
  warp_inst_t inst_snapshot;             // enough to drive retirement (dst regs, write-barrier id, pc)
};
std::deque<InFlightWgmma> m_in_flight_wgmma;   // per SM (or per subcore)
```

**Issue path (producer freed):** when a `TENSOR_CORE_OP` is issued, instead of entering the fixed
pipe:
1. Keep the existing `control_stage` write-barrier **increment** (unchanged — already correct).
2. Reserve only a small **issue-throughput II** on the tensor FU (`m_dispatch_pending_reserved_cycles`),
   so the producer can re-issue after II — NOT after latency. II value: config-gated (see 10.4).
3. Push an `InFlightWgmma{warp_id, gpu_sim_cycle + completion_cycles, snapshot}` onto
   `m_in_flight_wgmma`. Do **not** touch `m_pipeline_reg` / the `occupied` bitset stage index (this is
   what avoids the 32-deep assert entirely).
4. The producer `warp_inst_t` retires from the issue path immediately (like TMA's issue()).

**Advance path (background completion):** once per SM cycle, walk `m_in_flight_wgmma`; for every entry
with `completion_cycle <= gpu_sim_cycle`:
1. Fire the **write-barrier decrement** exactly as `instruction_retirement`'s else-branch does today
   (`add_pending_wait_barrier_decrement(WRITE_WAIT_BARRIER, id)`), plus `dec_inst_in_pipeline()` /
   `warp_inst_complete()` bookkeeping that `instruction_retirement` performs.
2. Remove the entry. (FIFO by construction since completion_cycle is monotonic per fixed latency; if
   variable, scan.)

### 10.3 completion_cycles / II values

- `completion_cycles = number_of_cycles` (the full compute time; e.g. m64n128k16 → 64 in this config,
  ≈ HW ideal 69). This is now free of the 32-cap because it is a stored cycle, not a pipe index.
- `II` (producer re-issue) = config knob. Per §9 the HW effective II ≈ latency (near-serial), so the
  first HW-faithful point is **II ≈ number_of_cycles** (NOT smaller). Provide a knob to sweep; default
  must reproduce the synchronous baseline for the bit-identity gate (see 10.4).

### 10.4 Bit-identity / safety gating

- Config flag `-wgmma_async_enable` (default 0). When 0, WGMMA takes the **exact current** fixed-pipe
  path → byte-identical to the 135,999 / 215,895 baselines.
- When 1, WGMMA takes the background-completion path. This is a **behavioral** change (not just a
  formula tweak like Design A), so the two code paths coexist behind the flag.

### 10.5 Files / insertion points (to verify at implement time)

- `sm.h` / `sm.cc`: the `m_in_flight_wgmma` deque, an `advance_in_flight_wgmma()` called from the SM
  cycle, and a helper that performs the retirement-equivalent bookkeeping (refactor the else-branch of
  `instruction_retirement` [sm.cc:682-692](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L682-L692) into a reusable
  `retire_async_wgmma(warp_id, write_barrier_id, ...)`).
- `subcore.cc`: at the WGMMA issue site, when `-wgmma_async_enable`, branch to the async enqueue
  instead of the fixed-pipe allocate/`reserve_latency`; still set the small II lockout via
  `reserve_unit`/`m_dispatch_pending_reserved_cycles`.
- `gpu-sim.{cc,h}`: register `-wgmma_async_enable` (+ optional `-wgmma_async_issue_ii`), `shader_core_config` fields.
- Reuse the existing `[CTAFIN]` instrumentation (tensor_ops fix already in `ee96251`) so the SAME run
  also confirms the warpgroup-4× count (WARP_GROUP_H100.md §6).

### 10.6 Correctness risks to watch

- **Where the write-barrier increment stays.** Keep it at `control_stage` (issue). The async path must
  only move the **decrement** to completion. Do not double-count.
- **WAR / read barriers.** WGMMA read operands (A/B in SMEM/regs) — the read-barrier release path
  ([functional_unit.cc:205-227](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L205-L227)) currently runs on the fixed-pipe path; the async
  path must reproduce any read-barrier decrement it owns, or WAR hazards break.
- **Ordering.** Multiple in-flight WGMMA with the same write-barrier id must decrement in the right
  order so DEPBAR `LE N` thresholds are met at the right cycle (FIFO deque handles this for equal
  latencies).
- **Teardown.** Warp/CTA exit must drain or account for still-in-flight WGMMA (mirror TMA's exit
  handling) to avoid a hung DEPBAR at kernel end.

### 10.7 Status
- [x] Design chosen: interpretation 2 (true async background completion). Design A reverted.
- [~] **SUPERSEDED — see §11 and the TL;DR.** Design B was fully planned above, but the §11 source
  audit then showed the sim is *already* effectively async and B would only duplicate a config change
  in timing terms. **Design B is PARKED (not implemented)**; it is kept as the future-structure option
  for when warpgroup/cluster modeling is added. The active lever is now warpgroup-4× (WARP_GROUP_H100.md).
- [ ] (if ever revived) Implement 10.2–10.5 behind `-wgmma_async_enable`, bit-identical default.

## 11. Why the sim is ALREADY async (source audit, 2026-07-16) — the decision behind the TL;DR

Verified in `Subcore::issue()` ([subcore.cc:572-634](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc#L572-L634)) that in this config
(`is_remodeling_scoreboarding_enabled=0`, trace mode, captured-from-binary ⇒ `use_traditional_scoreboarding=false`)
the ONLY things that gate a warp's next issue after a WGMMA are:
1. `is_fu_available = fu->can_issue()` — for a **tensor** next inst, the II lockout
   (`m_dispatch_pending_reserved_cycles`, [functional_unit.cc:137-139](file:///home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc#L137-L139)). A **non-tensor** next inst
   uses a different FU and is not gated by the tensor pipe at all.
2. `are_wait_barriers_ready` — the `gsb0`/DEPBAR write-barrier, i.e. a genuine data dependency on the
   WGMMA *result*. HW's `WARPGROUP.DEPBAR.LE gsb0,N` waits exactly the same way.
3. `is_stall_counter_0` — generic stall bits.

There is **no** register-RAW scoreboard on the WGMMA path (traditional scoreboard is off). So:
- the producer is **not** held for the WGMMA `latency`; it re-issues after the small II and runs other
  work meanwhile — the defining behavior of async;
- the consumer blocks only on the real result dependency (`gsb0`), as on HW.

The only non-HW-faithful piece is the **magnitude** of latency/II (§9: sim latency 32 vs HW ~64–128,
sim II 32 vs HW ~72), and those are set by `generate_tensor_core_latencies()` +
`-tensor_latency`/`-tensor_rate_per_cycle` — a **config/formula** knob, capped only by the fixed-pipe
depth (which is itself `-tensor_latency`). Hence: raising the config is the whole "async" fix in timing
terms; Design B adds no timing beyond that. This is why async-WGMMA is closed and the effort moves to
the warpgroup-4× over-execution (WARP_GROUP_H100.md), which async cannot address.
