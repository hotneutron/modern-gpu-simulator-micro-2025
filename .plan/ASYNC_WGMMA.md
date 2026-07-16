# ASYNC_WGMMA — model WGMMA re-issue as async (fwd/bwd tensor lever)

> **Primary target: bwd K10.** The per-CTA correlation ([CTA_FINISH_TENSOR_CORRELATION.md](file:///Users/bytedance/Documents/github/modern-gpu-simulator-micro-2025/.plan/CTA_FINISH_TENSOR_CORRELATION.md))
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
