# Producer Spin-Wait Sidecar

## 0. Purpose

This document describes the **static spin-wait sidecar** produced by
[`build_spin_wait_sidecar.py`](../simulator-remodeled/util/tracer_nvbit/build_spin_wait_sidecar.py)
and the contract it establishes for the simulator.

The sidecar is the *static half* of a fix for the FA3 Producer/Consumer timing
error: the simulator replays each warp's recorded PC stream and takes every
branch from the captured `next_traced_pc`, so a producer mbarrier spin loop is
replayed exactly as many times as it happened to spin during the (instrumented)
hardware capture. That recorded poll count is a capture-timing artifact, not
real work. The correct number of polls depends on when the *modeled* mbarrier
flips, which only the simulator can decide at replay time.

The sidecar does not modify the trace. It annotates, per kernel, where each
producer spin loop is (by PC), what barrier poll it hinges on, and where it
exits, so a later gated simulator feature can **block the warp at the loop until
the modeled barrier is ready and then jump to the loop exit**, instead of
replaying the captured poll count.

This file is read-only with respect to the trace payload. It is safe to
regenerate at any time on a CPU-only server.

## 1. Background: why the poll count cannot be replayed

Established earlier in this investigation (see
`.plan/PRODUCER_CONSUMER_HANDSHAKE.md` for the full analysis):

- `PHASECHK` / `TRYWAIT` are modeled as nonblocking predicate tests in
  `sm.cc::handle_sync_instruction()`; the modeled mbarrier readiness is computed
  but never changes the replayed control flow.
- Branch direction is decided solely by the captured `next_traced_pc` in
  `ibuffer_remodeled.cc::issued()`.
- Therefore the only thing that paced the producer was the *number of spin
  iterations baked into the trace*. Opt 11-B (sparse) removed most of that
  accidental pacing and exposed the missing timing model.

Because the trace stores a fixed integer number of poll iterations, no amount of
in-place blocking can match hardware: the modeled barrier flips at a different
cycle than it did on the instrumented capture. The only consistent fix is to
stop treating the recorded poll count as input and let the simulator regenerate
it. The sidecar makes that possible without altering semantic work.

## 2. Input trace selection

The sidecar is generated from **Opt 11-A**
(`LIGHTER_TRACER="compact_wire,prune_operands"`), chosen after verifying:

1. **Semantic work is identical** across fullrun / Opt 11-A / Opt 11-B. The
   per-warp CORE opcode sequence (HGMMA / TMA / MUFU / BAR / arrive / global and
   shared memory) is byte-for-byte order-identical in all three
   (K5 132 CTAs, K10 384 CTAs). No lighter trace dropped a semantic operator.
2. **Opt 11-A preserves spin-loop position.** Each spin loop's first-occurrence
   order relative to semantic ops matches fullrun. Only the *iteration count*
   differs (which the sidecar discards anyway).
3. **Opt 11-B (sparse) does NOT preserve spin-loop position.** Sparse
   reconstruction deleted several spin loops and relocated one to the end of the
   producer stream. Its bodies are still valid SASS (CFG-legal), but the dynamic
   placement of "which barrier was polled at which point" is lost, which is
   exactly the information a wait-region needs. Sparse is therefore unsuitable
   for this fix.
4. **fullrun is the most inflated** (48x / 86x more polls than the real HW NCU
   `NANOSLEEP` totals of 8,461 / 7,860) and is a slow-tracer artifact, not a
   hardware reference. Opt 11-A is lighter and, because the poll count is
   discarded, its reduced spin is irrelevant to the collapsed result.

Opt 11-A is thus the least-processed trace that keeps everything the fix needs.

## 3. What a spin region is

A spin region is a short **backward self-loop** whose body contains an mbarrier
poll:

- the loopback branch is a `BRA` whose immediate target is *below* its own PC
  (a backward branch) and within `MAX_LOOP_SPAN` (0x100 bytes);
- the loop body (from the branch target up to, but excluding, the loopback
  `BRA`) contains at least one `SYNCS.PHASECHK` or
  `SYNCS.PHASECHK.TRANS64.TRYWAIT`.

First and later iterations may differ (e.g. K10 `0xb8c0` has a leading
`STL/LDL` pair, some loops carry a leading `WARPSYNC.ALL` on the first pass), so
the region is anchored on **static PCs**, never on a fixed instruction body or a
dynamic instruction index.

Representative body (K5 `0xacb0`):

```text
0xacb0  IMAD.U32
0xacc0  SYNCS.PHASECHK.TRANS64.TRYWAIT
0xacd0  NANOSLEEP.SYNCS
0xace0  SYNCS.PHASECHK.TRANS64          <- primary_wait_pc
0xacf0  BRA -> 0xacb0                    <- loopback_bra_pc
0xad00  BRA -> 0x1380                    <- exit_pc (fall-through of the loopback BRA)
```

## 4. Detection algorithm

Implemented in `build_spin_wait_sidecar.py`. Reuses the dependency-free varint
protobuf parser and `_peek_funcid` from `trim_spin_from_ncu.py`, so it runs on
any server with no protobuf toolchain.

1. **Load static CFG** from `extra_info/enhanced_execution_info.json`
   (`pc -> opcode`, `BRA` immediate target).
2. **Load sync sites** from `extra_info/sync_operand_resolver.json`
   (`(funcid, pc) -> {sync_kind, opcode, barrier_operand_text, ...}`).
3. **Detect regions** per kernel: for every backward, short `BRA`, take the body
   between its target and itself; keep it only if the body contains a
   `PHASECHK` / `TRYWAIT`.
4. **Classify the body**:
   - `semantic_free_body = True` iff the body has no tensor (`*GMMA`), `MUFU`,
     TMA (`UTMA*` / `UBLK*`), global/shared memory (`LDG/STG/LDGSTS/LDS/STS/
     ATOM/RED`), `SYNCS.ARRIVE`, `SYNCS.EXCH`, or `BAR` op. Only such regions may
     be collapsed; a region with semantic ops is emitted with the flag `False`
     so the simulator can refuse to collapse it.
   - thread-local stack ops (`STL` / `LDL`) are allowed (warp-local side effect)
     and reported in `local_mem_ops_in_body`.
5. **Pick the primary wait**: the last `PHASECHK` in the body if present, else
   the last `TRYWAIT`. This is the poll whose result the loop branch depends on,
   and whose captured `barrier_addr` the simulator must test for readiness.
6. **Exit PC**: the fall-through of the predicated loopback `BRA`
   (`loopback_bra_pc + isize`).
7. **Dynamic validation** over all CTAs (`--validate-ctas all`): count loopback
   executions per `(cta, warp)` to record min/max/total iteration counts and
   mark `is_spin_loop = (max_iters > 1)`. Regions that only ever execute their
   loopback once are single-shot waits (collapsing them is a no-op).

`--strict` fails if any detected region has semantic ops in its body.

## 5. Sidecar schema

Written to `<traces>/extra_info/spin_wait_sidecar.json`.

```json
{
  "version": 1,
  "schema": "spin_wait_sidecar",
  "source_traces_dir": "<abs path>",
  "isize_bytes": 16,
  "note": "...",
  "kernels": [
    {
      "kernel_id": 5,
      "unique_function_id": 3,
      "kernel_name": "...",
      "num_regions": 21,
      "validated_ctas": 132,
      "spin_regions": [
        {
          "region_id": "k_fid3_h0xb290",
          "loop_head_pc": 45712,
          "body_pcs": [45712, 45728, 45744],
          "wait_pcs": [45712, 45744],
          "primary_wait_pc": 45744,
          "primary_wait_kind": "PHASECHK",
          "primary_wait_opcode": "SYNCS.PHASECHK.TRANS64",
          "barrier_operand_text": "[R5+URZ+0x16840]",
          "barrier_operand_index": 1,
          "loopback_bra_pc": 45760,
          "exit_pc": 45776,
          "semantic_free_body": true,
          "semantic_ops_in_body": [],
          "local_mem_ops_in_body": [],
          "dynamic": {
            "warp_visits": 132,
            "total_iters": 35694,
            "max_iters": 368,
            "min_iters": 214,
            "single_shot_visits": 0,
            "is_spin_loop": true
          }
        }
      ]
    }
  ]
}
```

Key design choices:

- **PC-anchored, not index-anchored.** The trace is stored per CTA and the poll
  count varies per CTA/warp, so PC anchors are both stable and match the
  simulator's `traced_pcs[used_insts]` cursor model.
- **Barrier address is runtime-only.** The poll's barrier operand is
  register-formed (`[R5+URZ+0x16840]`); the real address is known only from the
  instruction's captured `sync.barrier_addr` payload at replay time.
  `barrier_operand_text` is recorded for cross-checking/debugging only.
- **`semantic_free_body` gates collapse.** The simulator must never collapse a
  region whose body carries real work.

## 6. Validation results (Opt 11-A, all CTAs)

| Kernel | funcid | regions | semantic-free | CTAs | strict |
|---|---:|---:|---:|---:|---|
| K5  | 3 | 21 | 21/21 | 132 | PASS |
| K10 | 8 | 17 | 17/17 | 384 | PASS |
| K11 | 9 | 1  | 1/1   | 384 | PASS |

All detected regions are semantic-free. Iteration counts vary widely per
CTA/warp (e.g. K5 `0xb290`: 214..368; K10 `0xb960`: 84..616), which quantitively
confirms that the poll count is a timing artifact — precisely the value the
simulator must regenerate rather than replay.

## 7. Pipeline integration

`build_spin_wait_sidecar_if_available(trace_folder)` is called from
[`run_hw_trace.py`](../simulator-remodeled/util/tracer_nvbit/run_hw_trace.py)
right after `build_sync_operand_mapping_if_available`, so the sidecar is produced
automatically when a trace is captured. It is also fully standalone:

```bash
python3 simulator-remodeled/util/tracer_nvbit/build_spin_wait_sidecar.py \
    --traces <.../traces> \
    [--kernels 5,10] [--validate-ctas all|N] [--strict] [--out <path>]
```

Because detection is purely static and validation only reads the `.pb` files, it
runs on the CPU-only simulation server with no GPU and no re-capture.

## 8. Simulator consumption contract (next stage, not yet implemented)

The simulator, behind a default-off flag, should:

1. Load `spin_wait_sidecar.json` and build a `(funcid, primary_wait_pc) ->
   region` map.
2. When a producer warp issues the `primary_wait_pc` poll, test the modeled
   mbarrier at the captured `barrier_addr` (reusing `is_sync_wait_satisfied`).
   - **not ready**: park the warp (revive the currently-dead
     `m_pending_sync_waits` valid path consumed by
     `SM::warp_waiting_at_barrier`). Do not advance.
   - **ready**: advance `used_insts` to the next `exit_pc` occurrence, consuming
     the remaining recorded poll iterations without issuing them, then continue
     into the semantic block.
3. Only collapse regions with `semantic_free_body = true` and
   `is_spin_loop = true`.
4. Guard against a modeled barrier that never flips: after a threshold of parked
   cycles, surface it as a model bug (assert/log), never silently hang. The HW
   trace guarantees the loop did exit.

### Acceptance gates (updated)

- **Gate 0 — feature-off bit identity**: with the flag off, all totals match the
  current Opt 11-A/11-B baseline exactly.
- **Gate 1 — semantic-work invariant (relaxed from total icount)**: HGMMA / MUFU
  / TMA / BAR / global-shared-memory / arrive counts, TMA command and byte
  totals, and the captured control-flow of semantic blocks are unchanged. Only
  the spin-poll instruction count (`PHASECHK` / `NANOSLEEP` / loopback `BRA`) is
  allowed to change, because it is regenerated by the model.
- **Gate 2 — integrity**: TMA unresolved = 0, BAR leak = 0, no deadlock, clean
  exit.
- **Gate 3 — timing direction**: K5 Producer-minus-Consumer tail moves from
  `-61.4K` toward HW `+7.95K`; TMA interconnect-full and first-request cycles
  fall.
- **Gate 4 — no fake win**: no widened interconnect, no reinserted fake spins, no
  trace trimming, no HW-timestamp-based issue delay, no total-cycle-only match.

## 9. Observe-only stage (implemented; runs before collapse)

Before any control-flow change, an **observe-only** pass measures what collapse
*would* do, so a single (12h) run decides whether collapse is safe and where it
helps. It is gated by `-spin_wait_observe_enable` (default 0) and **never blocks,
advances the trace, or changes timing** — it only reads the modeled mbarrier
result already computed for each `PHASECHK`/`TRYWAIT`.

Scope: only polls whose `(funcid, pc)` matches a sidecar `primary_wait_pc` that
is `is_spin_loop && semantic_free_body`. For each `(warp, region)` episode it
records the cycle of the first not-ready poll and, when the same region next
polls ready, buckets the elapsed cycles = the cycles the warp **would** be parked
under collapse.

### Log lines (stdout)

- `[SPINWAITOBS] sm=.. wait_pc=0x.. episodes=.. avg_park=.. max_park=.. total_park=.. immediate_ready=.. watchdog_hits=.. hist[0/1-16/17-64/65-256/257-1024/>1024]=..`
  Per-region (per `primary_wait_pc`) would-be-park summary, emitted per SM. Cumulative;
  the last CTA-exit print for an SM holds the totals. Attributes park latency to a
  specific spin region so collapse can be applied/diagnosed region by region.
- `[SPINWAITDEADLOCK] sm=.. cta=.. warp=.. wait_pc=0x.. barrier=0x.. parked_cyc=.. first_not_ready_cyc=.. now=.. threshold=..`
  Emitted **immediately** (once per episode) when a single would-be park crosses the
  watchdog threshold (default 400,000 cycles, ~3x the K10 HW target). This is the
  primary deadlock-candidate signal and does not depend on any print budget.
- `[SPINWAITOPEN] sm=.. cta=.. warp=.. wait_pc=0x.. barrier=0x.. first_not_ready_cyc=..`
  Emitted at summary time for an episode still open at kernel end — a wait whose
  modeled barrier never flipped (strongest never-flip signal).

### Deadlock decision rule

Collapse is safe to implement iff, across the run, there are **no
`[SPINWAITDEADLOCK]` and no `[SPINWAITOPEN]` lines**. If any appear, the modeled
mbarrier for that `wait_pc`/`barrier` never becomes ready in a bounded window;
fix that barrier's accounting (phase parity, expected-arrive, expect-tx credit)
BEFORE enabling collapse, because collapse would turn that unbounded wait into a
real hang.

### Correlating with existing sync/TMA logs

`debug_log_sync_event` (`[SYNCDBG]`, gated by `-sync_debug_enable`, budget
`-sync_debug_print_budget`) and `debug_log_tma_event` (`[TMADBG]`, gated by
`-tma_debug_enable`) now prefix every line with `cyc=<gpu_sim_cycle>`. A
`[SPINWAITDEADLOCK]`/`[SPINWAITOPEN]` `barrier=0x..` can be traced on the same
cycle timeline through the existing `arrive_expect_tx` / `phase flip` /
`tma_complete` / `wait miss nonblocking` events (which already carry
arrive/expected/tx/phase/ready) to see exactly which credit never arrived.

### Config

`gpgpusim.config` sets `-spin_wait_observe_enable 1`. For a deadlock-diagnosis
run also raise `-sync_debug_enable 1` (and optionally `-tma_debug_enable 1`) so
the `barrier=0x..` from a watchdog line can be followed on the `cyc=`-stamped
`[SYNCDBG]`/`[TMADBG]` timeline. Set the observe flag back to 0 for normal runs.

`[BARDBG][issue]` / `[MEMBARDBG]` per-event lines (named barrier + memory fence,
also `cyc=`-stamped) are enabled via `-bar_debug_enable 1 -bar_debug_issue_budget
20000000`, so a deadlock candidate that involves a named barrier or a pending
fence (not just an mbarrier) is visible on the same timeline.

Budget sizing (checked): the sync/tma budgets are **per SM** and K10 schedules at
most ~3 CTAs per SM (384 CTAs / 132 SMs), so a busy SM emits only ~9K sync and
~2K TMA events — thousands of times under the 40M / 20M caps. `bar_debug`'s budget
is a single **process-global** static (~53K named-barrier events for all of K10),
still ~80x under 20M. No budget needs raising for a full K10 run.

### What this can and cannot prove (limits)

The observe pass is a strong but not complete safety check. Be explicit about the
boundary before trusting a GO decision:

- **Proves:** absence of never-flip / pathologically-long waits under the observed
  schedule (`[SPINWAITOPEN]` / `[SPINWAITDEADLOCK]`); the per-region park
  magnitude; and whether waits are mostly immediate-ready (collapse would be a
  no-op) vs genuinely blocking.
- **Does NOT prove — mutual collapse interaction:** the sidecar detects BOTH the
  producer's "buffer-empty" wait and the consumer's "buffer-full" wait, and
  collapse applies to both. Observe measures each wait independently under the
  *non-collapsed* schedule, so it cannot rule out a circular-wait deadlock that
  only appears when both sides are collapsed at once (double-buffering normally
  breaks the cycle, but a wrong modeled buffer depth could not).
- **Does NOT prove — exact timing:** in observe the producer still issues its spin
  polls and consumes issue slots. Under collapse those freed slots let the
  consumer run faster, flip the barrier earlier, and shorten the park. So the
  measured park is a *pessimistic upper bound*, not the collapsed timing.
- **Residual risk is covered at runtime:** the collapse implementation MUST carry
  the §8.4 watchdog (assert/log on unbounded park, never silently hang). That is
  the real safety net for the two gaps above.

## 10. How to proceed from the observe-only results (decision tree)

A single (12h) observe run selects the next step:

| Observed result | Decision | Action |
|---|---|---|
| No `[SPINWAITDEADLOCK]` / `[SPINWAITOPEN]`, and park histogram has real weight in the `65-256` / `257-1024` / `>1024` buckets | **GO** | Implement collapse (cursor jump) WITH the runtime watchdog. Validate the K5 tail moves from `-61.4K` toward HW `+7.95K` and TMA burst metrics fall. |
| No deadlock signals, but episodes are overwhelmingly `immediate_ready` / park ≈ 0 | **PIVOT** | Collapse is a near no-op: the producer is not actually blocked (barriers are ready when polled), so it races ahead for a different reason. Investigate barrier flip timing / expected-tx / arrive credit instead of collapse. Note this is consistent with the current "producer ends ~61K before consumer" symptom, so treat it as a live possibility. |
| Any `[SPINWAITDEADLOCK]` or `[SPINWAITOPEN]` | **NO-GO (as-is)** | Take the reported `barrier=0x..` and follow it on the `cyc=` timeline through `arrive_expect_tx` / `phase flip` / `tma_complete` / named-barrier events to find the credit that never arrives. Fix that barrier's accounting, re-run observe, then re-decide. |
| Some regions clean, others flagged | **SELECTIVE** | Enable collapse only for regions with `semantic_free_body && is_spin_loop && watchdog_hits==0`; leave flagged regions nonblocking until their barrier accounting is fixed. |

The GO path's collapse work is specified in §8. The watchdog there is mandatory,
not optional, because it is the only mechanism that covers the two limits in §9.

## 11. First observe-only run results (Opt 11-A, K5 `.o70` / K10 `.o44`)

Verdict: **GO** (no deadlock; real park exists). Detail:

| Signal | K5 | K10 | Meaning |
|---|---:|---:|---|
| `watchdog_hits` (park > 400K) | 0 | 0 | no unbounded wait |
| `[SPINWAITDEADLOCK]` | 0 | 0 | no over-threshold park |
| closed-episode `total_park` | 0.63M | 3.10M | real would-be pacing exists |
| hottest region | `0xb2b0` 403K | `0xb980` 1.33M | most park weight in `>1024` bucket |

- **No deadlock.** Every barrier flips normally. Verified directly on the `cyc=`
  timeline: K5 SM27 barrier `0x16c00` flips at `cyc=10049` and `30658`
  (`arrive_expect_tx` -> two `tma_complete` -> `phase flip`), and the kernel ends
  cleanly at `gpu_sim_cycle=83705`.
- **Large park confirms collapse is worth doing** (not a PIVOT): the closed
  episodes carry hundreds of thousands (K5) to millions (K10) of would-be park
  cycles, mostly in the `>1024`-cycle bucket.

### Measurement artifact found and fixed (episode close)

The first run also printed many `[SPINWAITOPEN]` lines (K5 1471, K10 2539), which
initially looked like never-flip deadlocks. They were **not**: 93% were a single
consumer wait (`wait_pc=0xace0`, `episodes=0` — never closed). Root cause was an
observe bug, not a simulator deadlock: the episode was closed only when the same
`wait_pc` was re-polled ready, but once the barrier flips the phase parity
changes, so the next poll of that `wait_pc` uses a different `wait_state` and
never matched. `record_spin_wait_observe` now closes an episode when the **same
`barrier_addr`** is observed ready (the real spin-loop exit), and the residual
`[SPINWAITOPEN]` count reflects true never-flip only. The fix is committed but the
12h run was **not** re-executed — the GO decision already holds from the barrier
flips confirmed on the timeline and the non-zero closed-episode park.

## 12. First COLLAPSE run results + K5 deadlock root cause (2026-08-12)

The first collapse-ON experiment (`-spin_wait_collapse_enable 1`,
`-spin_wait_collapse_debug 1`) was run on K5 and K10 simultaneously. **K10 passed;
K5 deadlocked.** This section is the self-contained handoff for the next session:
the log paths, the confirmed root cause, and the redefined direction. Everything
below is re-derived from the final logs (not the earlier live-streaming reads,
which were wrong — see §12.6).

> **Direction was redefined on 2026-08-12 (§12.4).** The earlier proposal
> (generation-counter-as-release + watchdog force-release, so K5 would "complete"
> by skipping the missed flip) has been **removed**. Reasoning: a producer that
> misses a consumer phase edge is a **simulator-accuracy bug**, not something to
> paper over with a skip. If collapse were done correctly the producer/consumer
> handshake would line up exactly as it does on real (tracer-free) hardware, so a
> missed edge must be **surfaced as a model defect (assert), never force-released**.
> The generation counter is retained but repurposed as a **consistency auditor**,
> not a release mechanism. See §12.4 for the confirmed root cause and the redefined
> plan.

### 12.1 Run outcome

| Kernel | stdout / stderr | Result | Final cycle | vs target |
|---|---|---|---:|---|
| **K10** bwd | `.o45` / `.e45` | ✅ clean `exit detected` | **135,352** | target 132,901 → **+1.84%** |
| **K5** fwd | `.o71` / `.e71` | ❌ `deadlock detected @ cyc 85333` | — (hang) | — |

- K10 collapse-ON 135,352 vs collapse-OFF (Opt 11-A) 136,898 = **−1,546 cyc
  (−1.13%)** → collapse moved K10 **toward** the HW target. Direction is correct.
- Full log dir base:
  `simulator-remodeled/sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/`
  - K5: `H100_80GB-OnlyKernel5/...-63a73d452237.{o71,e71}`
  - K10: `H100_80GB-OnlyKernel10/...-63a73d452237.{o45,e45}`
  - `[COLLAPSESUM]` (per-SM summary) is on **stdout** (`.o*`); per-event
    `[COLLAPSE][SMx] ... park/release+skip` is on **stderr** (`.e*`, K5 has 7,259).

### 12.2 Confirmed root cause — realized phase double-flip on a low-flip cross-warp barrier

The deadlock is **not** in the collapse bookkeeping (that is internally
consistent). It is a single poison park site, proven by set-equality:

- **The 21 SMs that parked at `wait_pc=0xb0b0` are EXACTLY the 21 deadlocked SMs**
  (`diff` of the two sets is empty). The GPGPU-Sim deadlock dump's stuck-core list
  (`6 8 9 16 23 24 25 27 ...`) equals the set of SMs that never printed a
  `[COLLAPSESUM]`.
- Park/release balance **by barrier** (all SMs, from `.e71`): every barrier is
  perfectly balanced **except one**:

  | barrier | parks | releases | dangling | how it is advanced (on SM6) |
  |---|---:|---:|---:|---|
  | 0x16c00 | 1370 | 1371 | 0 | **TMA-credited** (`tma_complete`, issue-independent) |
  | 0x16c30 | — | — | 0 | **TMA-credited** |
  | 0x16c40 | 979 | 979 | 0 | consumer `arrive` (warp-arrived) |
  | 0x16c48 | 907 | 907 | 0 | consumer `arrive` |
  | 0x16c60 | 228 | 228 | 0 | consumer `arrive` |
  | 0x16c68 | 126 | 126 | 0 | consumer `arrive` |
  | **0x16c20** | **29** | **8** | **21** | **warp15 `arrive` ONLY** (`tma_complete`=0) |

- **Two distinct wait sites share barrier `0x16c20`** (both `collapsible=True` in
  the sidecar):
  - dangling: `wait_pc=0xb0b0 exit_pc=0xb0d0` → **0/21 released** (poison)
  - successful: `wait_pc=0xb3d0 exit_pc=0xb3f0` → **8/8 released**
- The mechanism (`.e71`, barrier `0x16c20` only flips **twice** the whole kernel —
  warp15 arrives exactly 2×):
  - **Success (SM26):** producer reaches `0xb0b0` **while phase is still 1** (its
    prior generation) → **immediate-hit, never parks** → sails to `0xb3d0`, parks,
    and warp15's 2nd arrive (`@60782`) releases it. Clean 1:1.
  - **Dangling (SM6):** warp15 has **already done both arrives** (flip `0→1 @52212`,
    `1→0 @58477`) before the producer even polls `0xb0b0` (`@59786`). Phase is back
    to 0; the producer waits for `0→1`, but that flip **already passed** and no 3rd
    arrive exists → parks `@59821` **forever**. warp15 then blocks on downstream
    data the parked producer owes → circular wait closes → whole SM hangs.
  - A 1-bit parity cannot tell "my flip already passed" from "my flip has not
    happened yet", so the producer parks on an edge that will never recur.

### 12.2.1 Root cause quantified — this run only (`.e71`), poison vs success margin

**This is derived entirely from the collapse-ON run `.e71`.** The earlier
observe/collapse-OFF logs (`.o70`/`.e70`) are **not** a valid reference here: that
pass is observe-only, so the mbarrier does not drive timing at all and every wait
is walked past by construction — it is the first time mbarrier is actually in the
timing model that matters, and that is exactly this `.e71` run. The root cause must
be (and is) established inside `.e71`.

Splitting all 132 SMs by outcome (parked at `0xb0b0` = poison, immediate-hit at
`0xb0b0` = success) and measuring where the producer reaches `0xb0b0` relative to
the two flips of its own barrier:

| quantity (per SM, `.e71`) | POISON (21 SMs) | SUCCESS (111 SMs) |
|---|---|---|
| producer arrival − **gen2** flip | **+1 … +5,648** (always AFTER) | **−711 … −20,445** (always BEFORE) |
| gen1 flip cycle | med **50,697** (late) | med **45,779** (early) |
| gen2 − gen1 gap (the gen1 window) | med **7,085** (narrow) | med **13,474** (wide) |
| producer total collapse-park | med **40,055** | med **47,653** |

Reading:

- **Clean separator:** success iff the producer reaches `0xb0b0` **before gen2**;
  poison iff **after gen2**. SM16 fails by **+1 cycle** — this is a *timing-margin*
  fault, not a structural one.
- **It is NOT "poison producers park more".** The opposite is true (poison med
  40,055 < success med 47,653 collapse-park cycles). So the fault is not the
  absolute amount of collapse-injected park; it is the **relative** placement of
  the producer against the consumer-driven flip timeline.
- **The discriminating variable is the gen1 window width.** Poison SMs have gen1
  arriving *late* AND a *narrow* gen1→gen2 gap (~7K); the two consumer flips are
  bunched, so the window in which the producer's `0xb0b0` (a gen-1 wait) can land
  is short and the producer — paced by its own collapse parks on the empty barriers
  `0x16c40/48/68` — misses it. Success SMs open gen1 early and hold a wide (~13K)
  window.
- **Where the gen1/gen2 flip timing comes from:** `0x16c20` is flipped only by
  warp15's two arrives, and warp15 is itself gated between them by TMA/consumer
  barriers (`0x16c00/30/58`) whose timing is set by the modeled consumer-compute
  and TMA-fill cadence. So the accuracy fault that produces a too-narrow gen1
  window is in **that consumer/TMA pacing**, not in the collapse bookkeeping.

### 12.3 Why the design review missed it (corrections to SPIN_WAIT_COLLAPSE_DESIGN.md)

- **§12.8 "circular wait precluded (TMA completion is issue-independent)"** — true
  ONLY for TMA-credited barriers (0x16c00/0x16c30). `0x16c20` is **warp15-arrived**,
  so this safeguard does not apply to it.
- **§6.1 "double-flip precluded"** — assumed the parked barrier flips at most once
  between polls. A **cross-warp, low-flip barrier** (only 2 flips per kernel) can
  flip twice before the producer even reaches its poll — and §12.2.1 shows this is
  a *timing-margin* miss (poison SMs have a narrow ~7K gen1→gen2 window vs ~13K on
  success SMs), not a rare corner case: it happens on 21 of 132 CTAs.
- **Detector `collapsible` gate** (`is_spin_loop && semantic_free_body &&
  barrier_self_tied`) marked `0xb0b0` collapsible. `barrier_self_tied` is a
  **scoreboard** property; it does not capture **who advances the mbarrier** or
  **how few times it flips**.
- **Watchdog never fired** (`COLLAPSEDEADLOCK=0`): threshold 400K ≫ K5's ~85K
  cycles, and GPGPU-Sim's global detector (114K window) fired first. Max
  *legitimate* park across all SMs is only **8,104 cyc** (from `[COLLAPSESUM]
  max_park`). The watchdog currently only **logs then `return`**
  ([sm.cc:2294-2319](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2294-L2319)).
  Note (per the redefined direction §12.4): the watchdog must **assert/abort**, not
  force-release — an unbounded park is an accuracy-bug signal to surface, not to
  paper over.

### 12.4 Redefined direction — collapse is an accuracy audit, generation is the auditor (2026-08-12)

**Framing (decided with the user).** The reason collapse exists is to remove the
*tracer-induced* spin count so the sim paces the producer/consumer handshake the
way tracer-free **hardware** does. It follows that **if collapse were done
correctly, the producer would reach each wait while its matching consumer edge is
still the current phase — exactly as on HW — and every wait would line up 1:1 with
a flip.** A producer that arrives *after* its edge already passed (the SM6 poison)
is therefore **evidence of a simulator-accuracy defect somewhere in the timing
model**, not a situation collapse should "recover" from by skipping.

This inverts the earlier proposal. The generation counter is **kept**, but its job
changes from *release mechanism* to *consistency auditor*:

- **Collapse honors the mbarrier for real.** At a collapsible `primary_wait_pc`:
  not-ready → park; ready → `skip_spin_to_exit`. (Unchanged from the current
  implementation; this is the correct, HW-faithful behavior.)
- **Generation counter audits the 1:1 pairing and ASSERTS on violation — it never
  force-releases.**
  - Add a monotonically-increasing `gen` per barrier: `gen++` on every phase flip
    in `recompute_sync_barrier_ready_and_maybe_flip_phase`
    ([sm.cc:2391-2421](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2391-L2421)),
    **never reset**. Keep the existing 1-bit `phase` untouched → Gate 0
    (feature-off bit-identity) preserved.
  - Each collapsible wait carries the **absolute generation it is supposed to pair
    with** (`expected_gen`; see the formula caveat below). At wait entry, compare
    `barrier.gen` to `expected_gen`:
    - `barrier.gen < expected_gen` → the edge has not happened yet → **park**
      (normal, correct wait).
    - `barrier.gen == expected_gen` → the edge just landed → **release + skip**
      (normal, correct release).
    - `barrier.gen > expected_gen` → **the producer's edge already passed** → this
      is the poison. **ASSERT/abort with a full diagnostic** (sm/cta/warp/wait_pc/
      barrier/`expected_gen`/`barrier.gen`/first-flip-cyc/now). Do **NOT** skip,
      do **NOT** park-forever. The run stops loudly at the exact accuracy fault.
- **Watchdog = assert, not force-release.** The `service_collapse_release`
  over-threshold path
  ([sm.cc:2294-2319](../simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc#L2294-L2319))
  becomes a hard assert/abort as well (threshold right-sized to ~16–32K, above the
  8.1K legitimate max, below the 114K global detector). An unbounded park is, by
  the same logic, an accuracy defect to surface — never something to `skip` past.

**Why not just skip the missed edge (the rejected earlier plan).** Skipping makes
K5 "complete", but it launders an accuracy bug into a green run: the producer would
be advanced past a consumer edge it never actually observed, so the reported cycle
count would be a fiction. The user's explicit priority is correctness over a
passing run — "한번의 run으로 맞지 않더라도 반드시 풀어야 한다." The assert makes
the defect impossible to ignore and pins it to a coordinate.

**`expected_gen` formula — must be parity-aware, NOT `waits_completed + 1`.** The
earlier `target_gen = waits_completed + 1` is off-by-one for the producer's first
wait. Confirmed on the logs: the producer's very first wait on `0x16c20`
(`@7780`, parity=1) is an **immediate-hit at gen 0** (no flip yet), i.e. wait #1 ↔
gen 0, not flip #1. Because the barrier starts at `phase=0, gen=0` and keeps
`phase == gen & 1`, a parity-`p` wait is satisfied at the smallest `gen` with
`(gen & 1) != p`. So:

```
expected_gen(p) = smallest g >= (last consumed gen for this warp+barrier)+1
                  such that (g & 1) != p        # p = bit31 of wait_state_raw
# with the very first wait allowed to match g = 0 when phase already differs
```

This must be **validated offline** (no GPU) against the captured traces before any
coding: confirm, per barrier, that each warp's successive collapsible waits map to
strictly increasing generations and that the parity alternation holds. The CTA
structure is already confirmed identical (see §12.4.1), which is what makes an
absolute `expected_gen` well-defined.

**What the assert will most likely reveal (the real root cause to chase).** Per
§12.2.1 (this run, `.e71`), the discriminator is **not** the absolute amount of
producer collapse-park (poison SMs actually park *less*), but the **relative**
placement: poison SMs open gen1 late and keep a **narrow gen1→gen2 window (~7K
cyc)**, so the producer's `0xb0b0` gen-1 wait misses it. `0x16c20` is flipped only
by warp15's two arrives, and warp15 is gated between them by TMA/consumer barriers
(`0x16c00/30/58`) whose timing is set by modeled consumer-compute (WGMMA/MUFU) and
TMA-fill cadence. So the accuracy fault the assert pins is almost certainly **"the
modeled consumer/TMA pacing that sets the gen1→gen2 window is off relative to HW,"**
not the collapse bookkeeping. The assert coordinate (which barrier, `expected_gen`
vs `barrier.gen`, the gen1/gen2 flip cycles, first-not-ready cycle) is the lead for
that investigation.

### 12.4.1 CTA-consistency of the handshake (validated, all 132 CTAs)

The `0x16c20` handshake is structurally identical across every CTA, which is what
licenses an absolute-generation audit:

| property | value | scope |
|---|---|---|
| `EXCH` expected_arrive | `= 1` | 132/132 CTAs |
| flips per CTA on `0x16c20` | **exactly 2** | 132/132 SMs |
| arrive owner | **warp15 only** | 264/264 arrives |
| `wait_pc=0xb0b0` waits on | **parity 0** (→ 1st flip `0→1`) | 21/21 parks |
| `wait_pc=0xb3d0` waits on | **parity 1** (→ 2nd flip `1→0`) | 8/8 parks |

So `0xb0b0 ↔ generation 1` and `0xb3d0 ↔ generation 2` is a stable, per-CTA
invariant. The poison is purely a *timing* misorder (producer arrives at
generation 2 for a generation-1 wait), never a structural ambiguity.

### 12.5 Where the fix touches code (for next session)

Implementation is **audit-only** additions plus the assert; the collapse
park/skip logic itself is already correct and stays as-is.

- `tma_types.h` `HopperMBarrierObject` (~line 272): add `uint64_t gen = 0;`.
- `sm.cc::recompute_sync_barrier_ready_and_maybe_flip_phase` (2391-2421):
  `gen++` alongside `phase ^= 1` (never reset).
- `HopperMBarrierPendingWait` (tma_types.h): carry `expected_gen`.
- park-set branch in `handle_sync_instruction` (~2466-2534): compute
  `expected_gen` from a new per-`(warp,barrier)` generation-progress map (parity-
  aware formula, §12.4); then **audit**: `gen > expected_gen` → assert/abort with
  the diagnostic; `gen == expected_gen` → release+skip; `gen < expected_gen` →
  park. Do **not** replace the parity path used by the non-collapse issue logic
  (Gate 0).
- `sm.cc::service_collapse_release` (2285-2355): the over-threshold `else`/`return`
  (2294-2319) becomes an **assert/abort** (not `skip_spin_to_exit`), threshold
  lowered to ~16–32K.
- Add a `[COLLAPSEAUDIT]` diagnostic line (cyc-stamped) at the assert site so a
  future run's fault is greppable on the same timeline as `[SYNCDBG]`/`[TMADBG]`.

### 12.6 Corrections to the earlier (live-streaming) analysis — do NOT trust these

The mid-run notes were from partial logs and are **refuted** by the final logs:

- ❌ "20 unreleased parks" → ✅ actually **21**, all at `wait_pc=0xb0b0` on
  barrier `0x16c20`.
- ❌ "all collapse activity is warp=0 only; consumers never park" → ✅ consumers
  **do** park (park events show warp=12/13/14 on 0x16c00; 0x16c40/48/60/68 are
  consumer barriers with thousands of balanced parks). The dangling park happens
  to be producer warp0, but that is site-specific, not warp-role-specific.
- ❌ "watchdog too small" confusion → ✅ watchdog was too **large** (400K vs ~85K
  kernel) AND only logs, so it never fired.
- ❌ earlier proposed fix "exclude warp-arrived barriers" → ✅ wrong; see §12.4.
- ❌ earlier proposed fix "generation counter as edge-triggered **release** +
  watchdog **force-release**" (so K5 completes by skipping the missed edge) →
  ✅ **removed** on 2026-08-12. A missed edge is an accuracy defect to **assert**
  on, not to skip. The generation counter is kept only as the audit that raises
  that assert. See §12.4.

### 12.7 Quick re-verification commands (next session)

```bash
DIR=simulator-remodeled/sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1
K5E=$DIR/H100_80GB-OnlyKernel5/*-63a73d452237.e71   # collapse-ON run — the ONLY valid reference
# park/release balance per barrier (expect only 0x16c20 imbalanced):
grep -aoE "cyc=[0-9]+ park .*barrier=0x[0-9a-f]+ "    $K5E | grep -oE "barrier=0x[0-9a-f]+" | sort | uniq -c
grep -aoE "cyc=[0-9]+ release.*barrier=0x[0-9a-f]+ "  $K5E | grep -oE "barrier=0x[0-9a-f]+" | sort | uniq -c
# the two wait sites sharing 0x16c20 (0xb0b0 poison vs 0xb3d0 ok):
grep -aoE "park .*barrier=0x16c20 wait_pc=0x[0-9a-f]+ exit_pc=0x[0-9a-f]+" $K5E | sed -E 's/warp=[0-9]+ cta=[0-9]+ //' | sort | uniq -c
# ROOT CAUSE (§12.2.1) — poison=producer reaches 0xb0b0 AFTER gen2; success=BEFORE gen2.
# On a poison SM (e.g. SM6): the two 0x16c20 flips (gen1,gen2) both precede the 0xb0b0 park:
grep -aE "\[SM6\].*phase flip barrier=0x16c20" $K5E          # gen1 @52212 (0->1), gen2 @58477 (1->0)
grep -aE "\[SM6\] cyc=[0-9]+ park warp=0 cta=0 barrier=0x16c20 wait_pc=0xb0b0" $K5E   # @59821 (> gen2)
# On a success SM (e.g. SM26): producer immediate-hits 0xb0b0 BEFORE gen2:
grep -aE "\[SM26\].*wait immediate-hit warp=0 cta=0 barrier=0x16c20 .*parity=0" $K5E | head -1  # < gen2
# NOTE: do NOT use .o70/.e70 (observe/collapse-OFF) as a baseline for this wait —
# that pass does not put mbarrier in the timing model, so walking past the wait there
# is by construction, not evidence about the collapse-ON handshake.
```

Sidecar of record (both wait sites are `collapsible=True`, `barrier_self_tied=True`):
`simulator-remodeled/hw_run/traces/device-0/12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1_lighter-compact_wire+prune_operands-69a20aed/traces/extra_info/spin_wait_sidecar.json`
(schema v2, K5 = kernel_id 5, funcid 3, 21 spin_regions; the poison region is
`region_id=k_fid3_h0xb090`, `primary_wait_pc=0xb0b0`, only **2** dynamic flips of
its barrier per kernel).
