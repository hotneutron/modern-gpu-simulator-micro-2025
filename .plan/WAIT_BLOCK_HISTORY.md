# WaitBlock History: Server Runs and the Rejected Generation-Ticket Design

**Work type:** Chronological execution record. This file preserves the
server-run evidence (`.e80` through `.e85`, `.e59`/`.e61`/`.e62`) and the two
closed design lines (the rejected ordered-generation ticket design and the
reverted post-`951b058` ordered-candidate experiment).

**Parent index:** [WAIT_BLOCK.md](WAIT_BLOCK.md). Nothing in this file is
implementation authority. It exists so the resolved bugs are not re-debugged and
the rejected designs are not re-proposed. The current authoritative design lives
in the parent index plus [WAIT_BLOCK_POSTPROC.md](WAIT_BLOCK_POSTPROC.md),
[WAIT_BLOCK_SIMULATOR.md](WAIT_BLOCK_SIMULATOR.md), and
[WAIT_BLOCK_TMA_OWNERSHIP.md](WAIT_BLOCK_TMA_OWNERSHIP.md).

---

## 1. K5 `.e80`: first actual partial modeled miss

The third server run reached the previously hypothetical v1 partial boundary
(the `UNSUPPORTED_PARTIAL_MISSING_WAIT` abort defined in
[WAIT_BLOCK_SIMULATOR.md](WAIT_BLOCK_SIMULATOR.md) section 10.1). Its first
fatal event was:

```text
[WAITBLOCK][UNSUPPORTED_PARTIAL_MISSING_WAIT][SM103]
cyc=44865
block=wb_fid3_e0x93b0
trace_kernel=5
function=3
cta=0
global_cta=103
warp=0
entry_probe_pc=0x93b0
active_mask=0x1
predicate_mask=0x1
effective_mask=0x1
live_mask=0xffffffff
barrier=0x16c48
wait_state=0x0
current_gen=4
expected_gen=5
captured_ready_visits=264
all_live_entry_visits=0
partial_entry_visits=264
scheduler_mutation=none
cursor_mutation=none
action=abort
```

This proved that a captured partial entry can become a modeled miss. The
partial-lane design was therefore required for K5 completion and was no longer a
hypothetical future extension.

The abort was not a regression in the captured-poll path:

- K5 completed 2,669 poll advances before the guard;
- all 2,667 non-interleaved poll-advance records satisfy rollback direction,
  forward-skip distance, per-PC/opcode accounting, and exact local-exit landing;
- two poll-advance log lines were split by concurrent stderr output and are not
  runtime invariant failures;
- maximum completed hold duration was 7,625 cycles and maximum heartbeat age was
  8,192 cycles, both below the 32,000-cycle watchdog;
- no generation, payload, sidecar, or cursor fatal preceded the partial guard;
- the subsequent signal 11, undefined `UIADD3`, and final signal 6 are secondary
  failures after the deliberate abort.

The source trace for global CTA 103, warp 0 shows why the guard cannot be
replaced by passthrough or cursor advance. After the active lane-0 `0x93b0`
probe, the captured ready path contains lane-0 IMAD/R2UR/UIADD3 setup,
`SYNCS.ARRIVE.TRANS64` at `0x94d0`, `UTMALDG.4D` at `0x94e0`, and lane-0
`WARPSYNC.ALL` at `0x9500`; the complementary `0xfffffffe` WARPSYNC occurrence
follows. A modeled miss must not execute or consume that lane-0 semantic suffix
before release.

For SM103, barrier `0x16c48` is configured for three arrivals and prior
generations are completed by warps 4, 8, and 12. The failing occurrence had one
of three arrivals for generation 5. This makes a conservative whole-warp hold
worth evaluating, because the known producer warps are distinct from held warp
0, but it does not prove timing equivalence for the live sibling lanes.

This result required deferred handling of the blocked lane suffix, which became
the approved v2 partial policy in
[WAIT_BLOCK_SIMULATOR.md](WAIT_BLOCK_SIMULATOR.md) section 10.2.

---

## 2. First simulator v5 full-run result (`.e55`, `.e81`)

Server runs:

- K10: `.o55/.e55`, completed after 136,997 cycles and 6h30m.
- K5: `.o81/.e81`, aborted at cycle 69,031.

K10 completed with all 132 final SM summaries at `status=PASS`, balanced
missing-wait and captured-poll hold/release counts, zero generation overshoots,
and zero watchdog aborts. It had no modeled partial misses. The 384 observed
partial sibling joins followed the captured ready path and closed normally, so
K10 is a regression result but did not exercise deferred replay.

K5 exercised deferred replay:

```text
partial arms/extractions/holds:
  entry 0x93b0: 36
  entry 0x9930: 38

partial releases:
  entry 0x93b0: 11
  entry 0x9930: 38

deferred occurrences:
  entry 0x93b0: 17 per capsule
  entry 0x9930: 46 per capsule
```

No partial-specific assertion fired. However, none of the 49 released capsules
reached its `post_join_pc`. The release path flushed the IBuffer while the
canonical source was still selected, then activated the deferred stream.
`get_next_pc_to_fetch_request()` selected the first replay PC correctly but
advanced its internal fetch cursor from the stale canonical PC. The fix is to
compute the next fetch PC from the selected `res`:

```text
m_next_pc_to_fetch_request = res + fetch_width * instruction_size
```

A `partial-replay-drained` event is emitted after the final deferred occurrence
issues. The next server run must show one drain and one region close for every
partial release.

The K5 primary fatal was separate from partial replay:

```text
SM124 global_cta=124 warp=6
entry=0x5a50 barrier=0x16c38
expected_gen=5 current_gen=4
held_cycles=32001
strategy=INJECT_MISSING_WAIT
```

SM124 had no modeled partial miss. At abort, warp 0 was held at captured poll
`0xb370` waiting on `0x16c60`, while warps 4-15 were held at existing all-live
missing-wait entries waiting on `0x16c38`, `0x16c50`, and `0x16c58`. No CTA
warp remained runnable, so increasing the watchdog threshold is not a fix.
After verifying the replay-cursor correction, this all-live cross-warp
dependency cycle remains a separate design problem (now B0-E in
[WAIT_BLOCK.md](WAIT_BLOCK.md)).

---

## 3. Second K5 v5 run (`.e82`): replay progress confirmed, reservation gap exposed

### 3.1 Run result

The K5 `.o82/.e82` server run includes both the initial replay fetch-base fix
(`9f8e4fd`) and the replay lifecycle hardening (`f5168dd`). It aborts earlier
than `.e81`:

```text
[WAITBLOCK][abort][SM117] cyc=50173
reason=partial-replay-synthetic-issued
trace_kernel=5 function=3 cta=0 warp=0
requested_pc=0x9970 expected_pc=0x9970
```

The primary failure is the intentional fail-closed guard. The later signal 11
and `undefined instruction: UIADD3` messages are secondary failures emitted by
other simulator threads after the primary abort. This run does not reach the
old cycle-69,031 all-live watchdog, so it provides no new evidence about that
separate deadlock.

Observed partial lifecycle totals before the abort:

```text
partial-arm=23
partial-extract=21
partial-join-hold=23
partial-join-release=11
partial-replay-drained=7
partial-region-close with replay_progress_closed=1: 7
partial-replay-heartbeat=0
partial replay accounting/watchdog aborts=0
synthetic issue aborts=1
```

Seven capsules therefore complete the entire sequence:

```text
release -> replay all deferred occurrences -> replay drain -> post_join close
```


Representative successful cases:

```text
SM113:
  cycle 46450 release, remaining=17
  cycle 46502 drained at 0x9500
  cycle 46511 close at 0x9510, replay_occurrences=17

SM103:
  cycle 48164 release, remaining=46
  cycle 48324 drained at 0x9c20
  cycle 48328 close at 0x9c30, replay_occurrences=46
```

This proves that `9f8e4fd` fixed the original stale-base failure from `.e81`.
It does not prove that the replay frontend is correct for every instruction
cache response order.

### 3.2 Failing SM117 capsule

SM117 follows the expected partial state machine until replay:

```text
cycle 47362  arm at 0x9930
cycle 47371  extract 46 blocked occurrences and one sibling join
cycle 47374  issue sibling BSYNC 0x9c20 and begin hold
cycle 50154  release generation 6 and activate 46-entry blocked replay
cycle 50173  reject synthetic 0x9970 before issue
```

The source trace for CTA 117 warp 0 contains this exact blocked suffix:

```text
0x9950, 0x9960, 0x9970, 0x9980, ...,
0x9aa0, 0x9ab0, 0x9ac0, ..., 0x9c20
```

`0x9970` is therefore a real replay occurrence, not an absent or wrong-path
instruction.

### 3.3 What a synthetic instruction means

Synthetic instructions predate partial replay. The remodeled frontend can
speculatively request sequential instruction PCs before an older branch
resolves. Dynamic traces contain only executed-path occurrences, so a fetched
wrong-path PC may have no matching dynamic record. The legacy trace reader
materializes a zero-active synthetic NOP for such a response:

```text
pc = requested speculative PC
opcode = NOP
active mask = 0
dynamic payload = absent
is_instruction_traced = false
```

The synthetic instruction is not intended to issue. The older branch should
issue first and flush all younger wrong-path IBuffer entries. Returning no
instruction would leave an IBuffer entry unfillable after its cache response
has already been consumed.

This is compatible with in-order warp issue:

```text
IBuffer allocation order:  A, B, C
response/fill order:       C, A, B   (allowed)
issue order:               A, B, C   (FIFO)
```

Only response completion and IBuffer filling may be out of order. Warp
instruction issue remains in order.

### 3.4 Why replay misclassifies an early response

The current replay reader has one sequential decode cursor:

```cpp
if (pc != m_partial_replay[m_partial_replay_decoded].m_pc) {
  // construct a synthetic instruction
}
```

The server configuration has:

```text
fetch_decode_width=1
ibuffer_remodeled_size=5
ibuffer_coalescing=0
```

One request is generated at a time, but up to five IBuffer entries and fetches
may be outstanding. Their cache responses are not required to return in PC
order. A valid failing ordering is:

```text
replay sequence: 0x9950 -> 0x9960 -> 0x9970

decode 0x9950:
  expected=0x9950
  bind real occurrence
  replay decoded cursor now expects 0x9960

response 0x9970 arrives before 0x9960:
  requested=0x9970
  expected=0x9960
  current code incorrectly creates synthetic 0x9970
  replay decoded cursor remains at 0x9960

decode 0x9960:
  expected=0x9960
  bind real occurrence
  replay decoded cursor now expects 0x9970

issue:
  real 0x9950
  real 0x9960
  synthetic 0x9970 reaches the FIFO head
```

The abort prints `requested_pc=expected_pc=0x9970` because those values are
sampled at issue time. They differed when the synthetic object was created;
the real `0x9960` decode advanced the replay cursor before the synthetic
`0x9970` reached the issue head.

The fail-closed guard must remain. Allowing the synthetic instruction to issue
would silently replace a real deferred occurrence with a zero-active NOP and
violate exact-once replay.

### 3.5 Required fix: reserve replay occurrences at fetch allocation

The replay queue is not a new execution model. It is a temporary source of
ordinary traced instructions; the existing IBuffer may still fetch, fill, and
issue them with its normal timing. The missing contract is stable identity
between an IBuffer slot and the replay occurrence that the slot represents.

At IBuffer allocation time, reserve the next replay occurrence:

```text
IBuffer entry:
  requested_pc
  replay_epoch
  replay_occurrence_index
  replay_origin
```

Then bind a response using the entry's reservation rather than the global
`m_partial_replay_decoded` head:

```text
fetch allocation:
  reserve replay[index]
  entry.pc = replay[index].pc
  entry.replay_index = index

response/decode:
  locate the matching unfilled IBuffer entry
  decode replay[entry.replay_index]
  require replay PC/function/origin/epoch identity

issue:
  require entry.replay_index == replay_issued
  commit exactly one occurrence
  replay_issued++
```

Out-of-order fill is then safe:

```text
entries:        [index 0:0x9950] [index 1:0x9960] [index 2:0x9970]
response order: index 0, index 2, index 1
issue order:    index 0, index 1, index 2
```

The implementation must not derive later replay requests as `previous_pc +
instruction_size`; it must obtain every requested PC from the reserved dynamic
occurrence. This also supports a branch target or repeated PC inside the
deferred stream.

### 3.6 Reservation lifecycle and guards

A robust implementation needs these monotonic bounds:

```text
0 <= replay_issued <= replay_reserved <= replay_size
```

Decoded readiness belongs to each IBuffer entry and need not be represented by
a single monotonic `replay_decoded` cursor. Required invariants:

1. Reservation advances once when an IBuffer slot is allocated.
2. A cache response may fill only the slot carrying the same replay epoch and
   occurrence index.
3. Filling a slot never consumes or commits the replay occurrence.
4. FIFO issue requires `entry.replay_occurrence_index == replay_issued`.
5. Issue advances `replay_issued` exactly once.
6. IBuffer flush cancels every unissued reservation, including entries whose
   cache response has not arrived, and resets `replay_reserved` to
   `replay_issued`.
7. A replay epoch changes whenever a replay stream is activated, drained, or
   replaced. A response from an older epoch cannot fill a new replay entry even
   when its PC is identical.
8. FOLD inside replay first flushes/cancels reservations, then advances
   `replay_issued` over the proven poll occurrences.
9. Nested partial hold preserves the active replay and its issued position but
   cancels younger speculative reservations before resuming.
10. No synthetic instruction may issue while a replay owner or post-join
    progress record is active.

Serializing replay to one outstanding IBuffer entry would avoid this specific
response reordering, but it is only a conservative stopgap. Occurrence-index
reservation preserves the normal IBuffer behavior and is the selected design.

### 3.7 Next-session implementation checklist

1. Extend `IBuffer_Entry` with optional replay epoch/origin/index metadata.
2. Replace replay's global decode-head lookup with fetch-time occurrence
   reservation and reservation-indexed decode.
3. Keep canonical trace lookup unchanged.
4. Make flush cancel both decoded and not-yet-decoded replay reservations.
5. Tag or otherwise reject late instruction-cache responses from an older
   replay epoch.
6. Make commit verify FIFO replay index and exact PC/function identity.
7. Make deferred FOLD reconcile reserved, issued, and skipped positions.
8. Add focused tests for in-order response, `0x9970`-before-`0x9960` response,
   branch flush, repeated PC, stale response after release, nested hold, and
   replay drain to post-join.
9. Rerun K5 and require:

   ```text
   synthetic issue aborts=0
   partial replay accounting/watchdog aborts=0
   partial-replay-drained == extracted capsules that release
   replay_progress_closed=1 == partial-replay-drained
   ```

10. Only after replay passes the `.e82` failure point should the independent
    all-live cross-warp deadlock be analyzed again. Passing cycle 50,173 proves
    only that this reservation failure did not recur; it does not resolve the
    old cycle-69,031 dependency cycle.

The design is complete when each runtime action is explained by exactly one of
the two strategies. No entry action is permitted for a captured-poll record, and
the all-live missing-wait path performs no cursor action. The partial
missing-wait sub-policy may move source occurrences only into an owned replay
queue under the approved extraction and replay invariants.

### 3.8 Reservation implementation review: flush-vs-commit ordering bug

The reservation design of 3.5-3.6 was implemented and then re-reviewed against
the actual issue/commit ordering in `SM::issue_warp`. One real bug was found and
fixed; the rest of the reservation logic checked out.

Ordering fact: `issue_warp` calls the IBuffer `issued()` **before**
`commit_trace_inst`:

```text
ibuffer->issued();          // pop_front; may trigger flush(true) on a
                            // non-sequential next_traced_pc
commit_trace_inst(inst);    // replay_issued++ only here
```

Checklist item 3.7.4 ("flush cancels both decoded and not-yet-decoded replay
reservations") was first implemented literally, as a bulk
`reserved = decoded = issued` cancel at the end of flush. That is wrong because
the just-issued replay occurrence has already been `pop_front`'d by `issued()`
but its `replay_issued` has not advanced yet. A bulk cancel rewinds `reserved`
onto that in-flight occurrence, and the following `commit_trace_inst` then makes
`issued > reserved`, so the next fetch re-reserves an already-issued index and
double-issues it (or trips the FIFO identity assert).

This is triggered in practice: the K5 `0x9930` blocked suffix is not fully
sequential (`... 0x9ac0 -> 0x9c20`), so `next_traced_pc != pc + isize` fires
`flush(true)` from inside `issued()` mid-replay.

Fix: reservation rollback is per-removed-entry, not bulk. The flush loop calls
`unreserve_last_partial_replay()` once for every removed IBuffer slot that
carries a replay reservation (both filled and never-filled slots, so unfilled
reservations do not leak). The in-flight occurrence is already out of the deque,
so it is not un-reserved, and the subsequent commit advances `issued` past it
correctly. This mirrors the original per-entry `rollback_trace_inst` semantics
that the bulk cancel had replaced. The unused bulk
`cancel_partial_replay_reservations()` was removed.

Items confirmed correct during the review:

- The `.e81` stale-fetch-base fix still holds: flush sets `m_is_init_next_pc =
  false`, so the next fetch re-derives its base from `get_pc()` after the replay
  stream is activated, not from a stale canonical cursor.
- Stream activation (`extract_partial_wait_capsule`,
  `activate_partial_blocked_replay`) is always preceded by a flush, so a
  stale-epoch response never survives to decode; the epoch assert is a
  fail-closed guard, not a normal path.
- `RESERVATION_FAIL` in `remove_entry` gives back exactly one reservation via
  the same `unreserve_last_partial_replay()`.
- Deferred FOLD (`skip_partial_replay_to_exit`) runs after a flush with
  `reserved == issued`, and advances reserved lock-step with issued over the
  skipped occurrences.

### 3.9 Post-review cleanup

Two review notes from the full replay-queue re-read were addressed. Neither was
a correctness bug in the fixed reservation model; both remove latent hazards.

- Removed `m_partial_replay_decoded`. After the reservation rewrite each IBuffer
  slot decodes its own reserved occurrence index, so a separate global decode
  cursor was dead state whose only remaining reader was a redundant
  `decoded == issued` assert in the deferred FOLD path (already covered by
  `reserved == issued`). The cursor is gone; decode is purely
  reservation-indexed and the invariant is now `0 <= issued <= reserved <=
  size`.
- Added a fail-closed guard in `get_next_pc_to_fetch_request`: partial replay
  requires `fetch_decode_width == 1`. The reservation rollback (`remove_entry`
  after RESERVATION_FAIL, and per-entry flush unreserve) assumes one slot is
  allocated at a time and that `back().m_pc` equals the returned request PC. A
  width > 1 request would allocate several slots at once and mix reserved replay
  slots with sequential canonical slots, breaking that identity. The server
  config already uses width 1; the guard aborts instead of silently corrupting
  replay ordering if the width is ever raised while a replay stream is active.

---

## 4. Post-reservation run (`.e83`): reservation fixed, TMA tx-credit deadlock isolated

The `.e83` K5 run built on the reservation fix. Result:

- Reservation bug is fixed. The run passed the old `.e82` failure point at cycle
  50,173 with `synthetic issue aborts = 0`, and completed 48 real
  deferred-replay drains (`replay_occurrences > 0`) with zero partial-replay
  accounting/watchdog aborts.
- It then aborted at cycle 68,271 on a 32,000-cycle hold-watchdog for
  `INJECT_MISSING_WAIT` block `wb_fid3_e0x5a50` (barrier `0x16c38`), i.e. the
  independent all-live cross-warp deadlock predicted by 3.7.10. This is not a
  partial-replay failure.

### 4.1 Deadlock state (SM121, CTA 0, all 13 participating warps held)

```text
warp 4,7      block 0x5a50  barrier 0x16c38  exp/cur gen 5/4  arrive 0/1
warp 12,13,14 block 0x13b0  barrier 0x16c38  exp/cur gen 5/4  arrive 0/1
warp 8,9,10,11,15 block 0x74e0 barrier 0x16c50 exp/cur gen 5/4 arrive 0/1
warp 5,6      block 0x5bc0  barrier 0x16c58  exp/cur gen 4/3  arrive 1/1 tx=8192/16384
warp 0        poll 0xb370   barrier 0x16c60  exp/cur gen 4/3  arrive 2/3
```

These records cover warp IDs `0` and `4-15`, which are the 13 participating
warps visible in the kernel's synchronization traffic. The earlier "all 16
warps held" wording was inaccurate: `.e83` has no terminal WaitBlock state for
warp IDs `1-3`. All 13 participating warps are held, so no logged producer can
supply the missing arrival/tx that another warp is waiting on.

### 4.2 Root cause audit: identity-free TMA credit plus an unresolved size mismatch

This is not a partial-replay reservation bug. The deadlocked SM121 has no
deferred-replay miss, while the full run has 48 real
`partial-replay-drained`/`replay_progress_closed=1` pairs and no replay
reservation or synthetic-issue abort.

The prior analysis correctly identified an invalid byte-routing mechanism in
`notify_tma_completion`, but it overstated the conclusion in three ways:

1. The runtime metadata is not size-consistent. The `0x16c00` expect-tx
   occurrence requests 24,576 bytes, but the following `UTMALDG` at `0x96b0`
   is modeled from descriptor metadata as 16,384 bytes (`128` 128-byte AGU
   requests).
2. SM121 already has captured-poll holds at cycle 6,581 and
   `INJECT_MISSING_WAIT` holds at cycle 6,655. The old claim that no WaitBlock
   hold existed before cycle 38,122 is false. The first hold in the final
   non-releasing set is warp 7 at cycle 36,270.
3. The terminal table contains 13 participating warps, not 16.

The exact SM121 sequence is:

```text
cyc=5774  bind 0x16c30 expect=16384
cyc=5787  enqueue uid=6 load_bytes=16384
cyc=6896  bind 0x16c00 expect=24576
cyc=6917  enqueue uid=7 load_bytes=16384
cyc=7984  uid=6 completes -> 0x16c30 gets 16384/16384
cyc=8182  uid=7 completes -> 0x16c00 gets 16384/24576

cyc=9094  bind 0x16c38 expect=16384
cyc=9107  enqueue uid=8 load_bytes=16384
cyc=9767  uid=8 completes
          -> stale FIFO front 0x16c00 gets 8192 and flips
          -> remaining 8192 spills into 0x16c38
```

`notify_tma_completion(warp_id, completed_tx_bytes)` has no transfer identity.
It applies bytes to the oldest per-warp binding and permits one completion to
cross binding boundaries. UID 8 therefore credits both the stale UID 7 barrier
and its own later barrier. From then on the queue is permanently offset by
8,192 bytes. At the terminal state `0x16c58` remains at `8192/16384`, never
flips generation 3 to 4, and the 13 participating warps form the observed wait
cycle.

The same structural problem exists even without the 24-KiB/16-KiB mismatch:
`.e83` contains at least 40 parseable cases where LOAD completion UIDs arrive
out of issue order. A completion-order FIFO cannot preserve
TMA-command-to-barrier identity under that response ordering.

The byte mismatch is not resolved by `.e83` alone. Post-review evidence in
[WAIT_BLOCK_TMA_OWNERSHIP.md](WAIT_BLOCK_TMA_OWNERSHIP.md) section 3 resolves it
as the first case below:

1. `0x96b0` really moves 24,576 bytes and its descriptor/geometry mapping is
   wrong. **Confirmed.**
2. The 24,576-byte expect count covers the 16,384-byte UTMALDG plus another
   8,192-byte async producer that the simulator does not model. **Rejected by
   full K5 occurrence grouping.**
3. The TMA instruction has a distinct architectural completion-credit size
   that must be carried separately from modeled memory traffic bytes.
   **Not needed for this site.**

The captured `semantic_raw=24576` is hardware runtime evidence; the
descriptor-derived 16,384-byte command is a medium-confidence simulator
metadata mapping. The FA3 source-level transaction formula, kernel tile shape,
and all dynamic occurrences independently prove that this is the 24,576-byte Q
transfer. The resolver must fix the descriptor association generically; do not
hardcode the affected PC/barrier.

WaitBlock makes this invalid accounting architecturally visible by holding a
modeled-not-ready warp. The byte mismatch is present before the final
non-releasing hold set, but `.e83` does not justify the stronger claim that
WaitBlock never changes the relative timing/order of bindings and completions.

### 4.3 Required fix boundary (design review)

The fix has two mandatory parts.

1. Give every load `TMACommand` an exact completion binding before it enters the
   asynchronous mover. Prefer the barrier identity carried by the TMA
   instruction. Both observed `UTMALDG` and `UBLKCP` forms carry the completion
   barrier in the high UReg of their first destination-state register pair.
   Carry that runtime value and a unique issue-time binding ID through
   `TMATransferEntry`.
2. On completion, update only that binding's `(kernel, CTA, barrier,
   generation)` object. Never stream remaining bytes into another binding.
   Completion order must be irrelevant.

The 24-KiB/16-KiB contract is resolved in
[WAIT_BLOCK_TMA_OWNERSHIP.md](WAIT_BLOCK_TMA_OWNERSHIP.md) section 3, but the
corrected descriptor mapping and direct barrier payload were not implemented at
`.e83`. Merely adding UID identity while retaining the 16,384-byte
misclassification leaves `0x16c00` permanently at `16384/24576` and deadlocks
earlier. Merely forbidding cross-binding spill is a useful fail-closed
diagnostic, not a fix.

Required guards and validation:

- every completion-bearing TMA load owns exactly one explicit barrier binding;
- a binding is claimed and completed exactly once;
- completion key and generation match the object that was armed;
- one completion never updates two barrier keys;
- per-barrier completed bytes never exceed expected bytes;
- CTA cleanup has no orphan binding or in-flight completion;
- logs include TMA UID, TMA PC, barrier key/generation, memory bytes, and
  completion-credit bytes;
- tests cover equal-size binding, the current 24-KiB/16-KiB mismatch,
  out-of-order UID completion, repeated barrier generations, and CTA cleanup;
- K5 rerun requires the old cycle-50,173 replay failure to remain absent, all
  released capsules to drain/close, no TMA binding invariant failure, no
  WaitBlock watchdog, and a final grep-able `FINAL status=PASS`.

This fix boundary became B0-D, whose design is in
[WAIT_BLOCK_TMA_OWNERSHIP.md](WAIT_BLOCK_TMA_OWNERSHIP.md).

---

## 5. B0-C server result (`K5 .e84/.o84`, `K10 .e59/.o59`)

The size-only experiment used sidecar schema v2 with
`base_map_version=2`, `operand_map_version=2`, and
`contracts_validated=1`.

K10 is the clean regression:

- all 6,528 UBLKCP completions carry 512 bytes;
- all 7,296 descriptor LOAD completions carry 16,384 bytes;
- aggregate LOAD bytes are
  `issued=completed=122,880,000`, equal to the offline contract total;
- all 384 WaitBlock summaries report `status=PASS`, zero active state, and
  zero watchdog aborts;
- the kernel exits cleanly at `gpu_sim_cycle=138,456`.

K5 passes the primary size/byte criterion but not the clean-exit diagnostic:

- `0x96b0` forms a 24,576-byte command with 192 128-byte AGU requests, and
  reached data-mover completions report `requests=sector_goal=768` 32-byte
  sector requests;
- every reached Q completion carries 24,576 bytes; 178 such completions occur
  before the abort, while the offline full-trace proof covers all 264 Q
  occurrences and closes every byte contract;
- the run passes cycle 50,173, then aborts at cycle 68,176 on SM124,
  `wb_fid3_e0x5a50`, barrier `0x16c38`, expected/current generation `5/4`;
- the previous terminal `0x16c58 tx=8192/16384` deficit is absent.

The run proves two independent remaining problems.

First, B0-D is still mandatory. At SM84 cycle 9,898, corrected 24-KiB command
UID 7 completes while the legacy FIFO still names an older 16-KiB
`0x16c30` binding first. The completion applies 16,384 bytes to `0x16c30` and
spills its remaining 8,192 bytes into `0x16c00`. Global bytes are conserved,
but command-to-barrier identity is wrong.

Second, that FIFO event is not the direct cause of the first K5 abort. The
terminal SM124 sequence has complete, in-order TMA applications through its
last completion at cycle 40,613. At abort, all 13 participating warps are
held:

```text
warp 0       captured poll 0xb370  barrier 0x16c60  expected/current 4/3
warps 4-7    missing wait  0x5a50  barrier 0x16c38  expected/current 5/4
warps 8-15   missing wait  0x74e0  barrier 0x16c50  expected/current 5/4
```

Barrier `0x16c60` is at `arrive=1/3`, while the consumer warps that can
provide the remaining progress are held waiting for later producer TMA
generations. This is an all-live cross-warp WaitBlock dependency cycle. It is
a separate model defect because the hardware kernel completes and because the
cycle persists after the Q-byte deficit is removed. The exact correction is
not yet selected: the missing-wait generation/hold placement and the producer
captured-poll dependency must be audited after B0-D removes completion
identity corruption. B1 traversal ordering must not be presumed to solve a
32,000-cycle state with no runnable participating warp.

B0-D component acceptance, independent of the separate K5 WaitBlock cycle:

```text
TMA binding coverage = 100%
contracts_opened == contracts_fully_assigned
bindings_created == bindings_completed
assigned_credit_bytes == completed_credit_bytes
TMA cross-barrier applications = 0
TMA contract mismatches = 0
stale_generation = 0
launch_or_cta_mismatch = 0
duplicate_completion = 0
orphan_cleanup = 0
an observed UID inversion routes each command only to its payload target
.e85 SM86 UID 7 -> only 0x16c00/24576
.e85 SM86 UID 6 -> only 0x16c30/16384
clean focused/K10 runs: every participating SM emits [TMABIND][FINAL] status=PASS
K5 WaitBlock abort: aborting SM emits [TMABIND][snapshot] with no ownership error
```

B0-D first runtime validation at `4de5c80` produced two distinct results.

K5 `.e85` validates the exact completion half of B0-D:

```text
contract-open/assign/enqueue/complete events = 4,167 each
logged assigned/completed credit            = 69,894,144 bytes each
TMABIND ownership aborts                    = 0
observed UID inversions                      = 3
SM124 terminal ownership snapshot           = PASS
```

In particular, SM86 completes UID 7 before UID 6 and still applies UID 7 only
to `0x16c00` with 24,576 bytes and UID 6 only to `0x16c30` with 16,384 bytes.
The old 16-KiB-plus-8-KiB spill does not occur. K5 then reaches the independent
13-warp SM124 WaitBlock cycle at 68,240; the aborting SM reports no open
contract, in-flight binding, byte mismatch, stale generation, or ownership
error.

K10 `.e61` invalidates the original one-slot assignment representation before
any LOAD is claimed:

```text
cycle 5590: ARRIVE_EXPECT_TX 0x31010 expected=16896 opens contract 1
cycle 5633: ARRIVE_EXPECT_TX 0x31000 expected=32768 reaches sync execution
             while contract 1 is still waiting for its LOADs to reach TMA issue
result:      previous-contract-unassigned abort
```

The dynamic trace remains correctly ordered:

```text
ARRIVE 0x31010
UTMALDG 0xa240 -> 0x31010
UBLKCP  0xa260 -> 0x31010
ARRIVE 0x31000
UTMALDG 0xa320 -> 0x31000
UTMALDG 0xa330 -> 0x31000
```

The overlap is caused by different pipeline service points, not missing trace
ordering: sync execution can process the later ARRIVE before the older LOADs
reach `tma_unit_sm::issue()`. The barrier-local repair permits both exact
contracts to remain open and lets each LOAD payload select its own contract.

B0-D final K10 regression acceptance:

```text
clean kernel exit
max_open_contracts_per_warp >= 2
0x31010: 16384 + 512 == 16896
0x31000: 16384 + 16384 == 32768
contracts=6912, bindings=13824
expected_credit=completed_credit=122880000
no semantic TMA/BAR/memory-work loss
no TMA binding/lifecycle abort
WaitBlock holds/releases balanced
SR-M0A population conservation PASS
```

---

## 6. B0-D final validation (`K5 .e85`, `K10 .e62`)

K5 `.e85` reaches its WaitBlock watchdog at cycle 68,240 on SM124 only after
all TMA work issued on that SM has closed exactly:

```text
[TMABIND][snapshot] status=PASS
contracts_opened=18
contracts_fully_assigned=18
bindings_created=18
bindings_completed=18
open_contracts=0
bindings_in_flight=0
assigned_credit_bytes=303104
completed_credit_bytes=303104
cross_barrier_applications=0
contract_mismatches=0
stale_generation=0
```

The 18 contracts and 303,104 bytes describe only commands that reached issue
before the cycle formed. They are not the complete CTA program. The complete
CTA124 trace contains 38 contracts, 38 completion-bearing LOADs, and 638,976
assigned bytes. Therefore the clean snapshot proves that no issued completion
was lost or misrouted; it does not prove that warp 0 issued its remaining
future TMA work.

The barrier-local assignment repair is commit `39abefb`. K10 `.e62` completed
at cycle 136,931 with all 384 CTAs issued and completed and a normal simulator
exit. It passed the old `.e61` abort at cycle 5,633 and logged 768 dynamic
records with `warp_open_contracts=2`, proving that contracts for `0x31010` and
`0x31000` can overlap.

The final B0-D reconciliation across all 132 SMs is:

```text
[TMABIND][FINAL] status=PASS: 132/132
contracts_opened=contracts_fully_assigned=6,912
bindings_created=bindings_completed=13,824
assigned_credit_bytes=completed_credit_bytes=122,880,000
max_open_contracts_per_warp=2
uid_reorders=6,379
open_contracts=0
warp_open_contracts=0
bindings_in_flight=0
barrier_outstanding=0
warp_outstanding=0
cross_barrier_applications=0
contract_mismatches=0
stale_generation=0
launch_or_cta_mismatch=0
duplicate_completion=0
orphan_cleanup=0
```

K10 also emitted 384/384 `[WAITBLOCK][summary] status=PASS` records. Every
record has balanced missing-wait holds/releases and captured-poll
holds/releases, zero active state, zero generation overshoots, and zero
watchdog aborts. The maximum captured-poll hold was 13,819 cycles, below the
32,000-cycle watchdog. No runtime abort, assertion failure, deadlock, signal,
or failed status appears in `.e62` or `.o62`.

Therefore B0-D is complete. B0-E work uses `39abefb` plus the `.e62` result
as its fixed exact-ownership baseline.

End-to-end K5 acceptance (still a later combined gate) and the B0-D
new-canonical-baseline statement are recorded in
[WAIT_BLOCK_TMA_OWNERSHIP.md](WAIT_BLOCK_TMA_OWNERSHIP.md) section 14.

The `.e85` terminal 13-warp cycle, its source-trace classification, and the
current B0-E root-cause direction are in [WAIT_BLOCK.md](WAIT_BLOCK.md).

---

## 6A. K10 `.e51` whole-warp park deadlock (pre-generation evidence)

This is the earliest and clearest proof that a whole-warp park, not the
generation counter, causes a WaitBlock hang. [WAIT_BLOCK.md](WAIT_BLOCK.md)
cites this evidence in "Why generation removal is not the fix"; the primary log
is retained here.

The failed Stage 1.7/1.8 universal WaitBlock implementation (later reverted for
the two-strategy v4 design) parked the whole warp at an admitted entry probe
even when only one lane group executed it. In K10 `.e51` this produced a
watchdog stall at `gen=0`:

```text
.e51:86459  K10 watchdog at SM24/warp0, entry 0xa9c0, barrier 0x31020,
            gen=0 for 32,001 cycles
```

The clean pre-implementation run and the regressed run differ only in park
placement, not in generation arithmetic:

```text
clean .e48:
  a9c0 miss -> disjoint b2c0 executes -> park b980 @ 12758
  barrier gen0->1 -> release b980 @ 22756

regressed .e51:
  park a9c0 @ 9129 -> b2c0 never executes
  barrier remains gen0 -> watchdog @ 41130
```

The mechanism: K10 CTA24 `0xa9c0` executes with `active_mask=0x1`. Before lane 0
reaches `0xa9e0`, lanes 1-31 execute `WARPSYNC.ALL 0xb2c0` with mask
`0xfffffffe`. Parking the whole warp at the lane-0 entry suppresses that
disjoint `0xb2c0`, so the producer ARRIVE that would flip `0x31020` gen0->gen1
never issues. The hold sits at `gen=0`; no generation compare is involved.

K5 `.e78` showed the same contract defect at release time instead of park time:
a lane-0 Type-2 visit at `0x9e80` tried to skip toward semantic `resume_pc` with
one linear warp cursor and hit lane 1-31 `BSYNC 0xa150`, which the offline
mask-filtered proof had ignored.

The root cause is that a lane-group semantic proof was used as authority for a
whole-warp park and a whole-warp linear skip. Dynamic protobuf streams are
linear warp issue streams, not one scalar CFG path: `b1e0(0x1) ->
a150(0xfffffffe) -> b1f0(0x1)` is SIMT lane-group interleaving, not a same-lane
CFG edge. This is exactly the boundary the two-strategy v4 design and the
partial-lane v2 replay policy were built to respect (see
[WAIT_BLOCK_SIMULATOR.md](WAIT_BLOCK_SIMULATOR.md)).

The failed universal implementation was reverted; full-CTA strict
post-processing then succeeded with the v4 two-strategy design
(`discovered=43 passed=36 failed=0 skipped=7`, `INJECT_MISSING_WAIT=22`,
`FOLD_CAPTURED_POLL=14`).

## 6B. K5 `.e85` producer/consumer timing divergence measurements

These are the quantitative timing findings behind the "K5 first late-arrival
root cause" work item in [WAIT_BLOCK.md](WAIT_BLOCK.md). They are diagnostic
evidence, not an accepted correction.

Phase-window miss on the terminal occurrence:

```text
SM124 0x16c50 generation 3 parity-0 window:
  opens (gen 2->3) at cycle 30,135
  closes (gen 3->4) at cycle 34,575
warp 12 reaches 0x74e0 at cycle 36,624   (2,049 cycles after closure)
```

Across the parity-0 terminal cohort, successful warp-12 occurrences have
3,118-4,923 cycles from their preceding `0x16c60` ARRIVE to the `0x16c50`
close; failed occurrences have only 512-863 cycles. Their raw consumer
intervals overlap (success 2,429-3,741; failure 2,541-2,909), so absolute
consumer latency alone does not classify the result. Remaining
producer/consumer headroom does.

The relevant TMA completion is not the defect:

```text
SM124 generation-3 0x16c50 contract opens at 34,183, flips at 34,575
  -> 392-cycle TMA interval
independently measured K5 TMA latency ~= 402 cycles/tile
```

Restoring the old identity-incorrect `.e76` 3,043-cycle interval would
reintroduce an accidental delay, not correct TMA timing. The residual is in
relative issue density: simulator consumer wall time is ~3,300 cycles/tile
versus ~1,700 on HW, and matched issue density is 0.71x HW.

Rejected correction axes (each independently disproven):

- **TMA latency inflation:** the 392-cycle transfer already matches the ~402-cycle
  mean; the multi-thousand-cycle `.e76` window came from an identity-incorrect
  completion path.
- **Fixed WGMMA/SFU latency reduction:** tensor occupancy is already lighter than
  HW and SFU latency is already HW-faithful.
- **Scheduler ordering alone (GTLRR):** a controlled A/B changed K5 cycles by only
  -1.6%, lockstep 74.6% -> 73.8%, SM-idle 31.7% -> 31.5%.
- **B1 release traversal:** the terminal SCC has no release event to reorder.

Rejected HW-timestamp route (do not repeat):

- NCU cannot recover an elapsed interval between low-frequency events; its data
  is cumulative with no time axis.
- Full-trace NVBit plus `clock64()` inflated the measured K5 producer tail by
  ~7,073x and destroyed the natural CTA distribution.
- The kernel is a prebuilt `flash_attn_3` binary, so there is no in-kernel
  timestamp source-rebuild path in this repo.
- Adding more NVBit callbacks measures the instrumented schedule, not the
  uninstrumented HW schedule whose window is needed.

Valid HW constraints are therefore aggregate only (EXIT-only producer/consumer
tail, NCU issue density, operation counts, pipe/stall metrics). They can reject
an incorrect timing change but cannot assign an exact per-occurrence barrier
window.

Two explicit semantic contracts were defined for any future schedule-robust
WaitBlock change; neither is selected, and the exact-generation Stage E1 result
made the liveness question moot for FA3:

```text
STRICT_PREDICTIVE
    a modeled miss may hold only when the next satisfying transition remains
    reachable without semantic work behind the candidate hold;
    an unsafe schedule aborts immediately.

TRACE_REPLAY
    a captured-ready INJECT_MISSING_WAIT is observation-only;
    FOLD_CAPTURED_POLL remains a modeled wait because the poll path was captured.
```

---

## 7. Rejected design: ordered occurrence-generation tickets

Status: **REJECTED / RETAINED AS A FAIL-CLOSED ALTERNATIVE.** This section does
not authorize implementation. It is recorded so the design is not re-proposed.
The approved B0-E direction is timing realignment plus global park safety; see
[WAIT_BLOCK.md](WAIT_BLOCK.md).

### 7.1 Generation-ticket feasibility result (why parity alone fails)

Static reaching-definition inspection found one unambiguous register chain at
each terminal wait:

```text
0x5a50: R7  <- UR19 <- UR5 phase bit
0x74e0: R11 <- UR7  <- UR5 phase bit
0xb370: R3  <- R2 phase bit
```

The recovered value is only `0x0` or `0x80000000`. It uniquely identifies the
one-bit phase/parity consumed by the instruction, but the same parity is reused
by multiple monotonic generations:

```text
phase 1: generation 1, 3, 5, ...
phase 0: generation 0, 2, 4, ...
```

The current trace has per-warp instruction order but no authoritative
cross-warp hardware execution order, hardware barrier generation, or hardware
TMA completion order at the wait. Consequently:

```text
register reaching definition = UNIQUE
phase provenance             = UNIQUE
exact monotonic generation   = AMBIGUOUS
```

An offline generation ticket or runtime shadow generation derived only from
the current phase value would guess among multiple valid generations. It is
rejected under the fail-closed policy. The existing
`expected_gen_for_wait()` calculation remains a simulator-local interpretation
of parity against current modeled state; it is not evidence of the hardware
generation that produced the captured ready result.

### 7.2 Rejected shortcuts

- Do not force a barrier phase or generation.
- Do not release an arbitrary held warp or add a deadlock watchdog escape.
- Do not make every captured-ready gate unconditionally pass; that could issue
  post-wait semantic work before the modeled barrier is ready.
- Do not generate exact tickets or shadow generations from one-bit parity.
- Do not treat a larger watchdog or B1 traversal ordering as a fix for a stable
  cycle with no runnable participating warp.
- Do not weaken B0-D ownership to recover the legacy FIFO's accidental
  progress.

### 7.3 The proposed correction (recorded, not authorized)

The correction targeted the information loss common to both the historical
pre-WaitBlock stale-poll failure and the current all-live cycle. The compact
trace preserves instruction order within each warp, but it discards the
cross-warp order in which one CTA executed barrier and TMA events. A one-bit
wait parity therefore cannot identify which logical barrier epoch produced a
captured result after the simulator chooses a different inter-warp schedule.

The design bracketed synchronization events with CTA-local sequence endpoints
during trace capture and resolved an exact occurrence ticket offline. It did
not derive an epoch from parity alone and did not alter B0-D completion
ownership.

Terminology:

```text
barrier generation g
    current simulator epoch before the next phase flip

release generation r
    monotonic generation immediately after the phase transition that satisfies
    one captured wait occurrence

waited generation
    r - 1 for a wait that requires the current parity to flip
```

For the terminal CTA124 parity-0 `0x74e0` occurrence, the intended proposal is
therefore `target_release_generation=3`, not 4. Generation 4 has phase 0 and
does not satisfy a parity-0 wait. The current live-state calculation instead
interprets the occurrence as waiting for the `4 -> 5` transition.

#### Pre-schema feasibility gate

Before freezing a transport record or protobuf schema, build a probe-only
microtest for each supported `SYNCS.PHASECHK*` form:

```text
PHASECHK
PHASECHK.TRYWAIT
predicated-on
predicated-off
destination P0 and non-P0
```

The microtest must prove that an `IPOINT_AFTER` callback can read the
instruction's destination predicate without changing the branch outcome. If
NVBit cannot expose that predicate reliably, the production design uses the
already-proven following branch path as the captured-result source. It must not
infer the captured result from barrier parity.

This probe uses a temporary fixed capture buffer and does not change the
production trace schema. Its result selects the production record shape before
the transport implementation begins.

#### Trace capture contract

Add one monotonically increasing 64-bit counter per launched CTA:

```text
cta_event_seq[linear_cta_id]
```

The counter starts at zero for every kernel launch. A relevant instruction is
bracketed by two lightweight callbacks. The first active lane records:

```text
begin_seq = atomicAdd(&cta_event_seq[linear_cta_id], 1)  // IPOINT_BEFORE
target instruction executes
end_seq   = atomicAdd(&cta_event_seq[linear_cta_id], 1)  // IPOINT_AFTER
```

for each dynamic occurrence in the following closed event set:

```text
SYNCS.EXCH*
SYNCS.ARRIVE*
SYNCS.PHASECHK*
completion-bearing UTMALDG / UBLKCP
```

The sequence is CTA-local, not device-global. CUDA does not migrate a resident
CTA between SMs, and no ordering between independent CTAs is needed. A
device-global counter is rejected because it would add unrelated cross-CTA
serialization. `clock64()` is not part of this design.

A single callback sequence is not sufficient. A callback inserted before an
instruction can execute before another warp's callback even when the target
instructions later execute in the opposite order. Bracketing derives only
sound happens-before relations:

```text
A.end_seq < B.begin_seq  -> A definitely completed before B began
intervals overlap        -> A and B remain unordered
```

The resolver enumerates legal orders for overlapping intervals. It must not use
warp ID, host receive order, or `begin_seq` alone as a tie-break.

The callback rules are:

- insert one matched begin/end capture pair per dynamic instruction, not one
  pair per traced operand;
- execute the begin and end callbacks at `IPOINT_BEFORE` and `IPOINT_AFTER`;
- let the first active lane publish both endpoints;
- capture the destination predicate in the end event for waits;
- retain active and predicate masks so predicated-off occurrences remain
  distinguishable but have no semantic effect;
- compute `linear_cta_id` from all three CTA coordinates and the launch grid;
- bounds-check the allocated counter array and abort trace generation on
  overflow or an invalid CTA index;
- reset counters at kernel launch and free them only after the channel drains.

The order capture must not use a host receive index. The current trace receiver
groups records into per-warp protobuf arrays, and host/channel arrival order is
not an explicit cross-warp execution-order contract.

The recommended transport is a versioned order-event record:

```cpp
struct sync_order_event_t {
  uint32_t record_size;
  uint16_t record_version;
  uint16_t record_kind;
  int32_t cta_x;
  int32_t cta_y;
  int32_t cta_z;
  int32_t warp_id_tb;
  int32_t sm_id;
  uint32_t unique_function_id;
  uint32_t pc;
  uint32_t active_mask;
  uint32_t predicate_mask;
  uint64_t begin_seq;
  uint64_t end_seq;
  bool has_wait_result;
  bool wait_result;
};
```

Runtime barrier addresses, wait-state values, and TMA completion targets remain
owned by the existing protobuf instruction payload. The order event is joined
to the raw, untrimmed protobuf exactly once by:

```text
(kernel, CTA xyz, warp, function, PC, per-PC occurrence ordinal)
```

This ordinal is only a bootstrap join key. It is forbidden after any trace
transformation. Immediately after the raw join, stamp the directly captured
protobuf instruction with:

```text
trace_event_id
begin_seq
end_seq
source_warp_instruction_index
captured wait result, when available
capture SM ID
```

The receiver also writes `extra_info/sync_event_order.jsonl` as an independently
auditable artifact. Spin trimming copies the stamped fields on retained
instructions. Sparse reconstruction never invents them. All post-transform
joins use `trace_event_id` plus `source_warp_instruction_index`, never a newly
computed per-PC ordinal.

Required capture validation:

```text
one complete begin/end pair per relevant protobuf occurrence
no duplicate sequence endpoint within one CTA
begin_seq < end_seq for every event
strictly increasing, non-overlapping intervals for each warp
one SM ID for the lifetime of one CTA
active/predicate masks equal the protobuf occurrence
no missing EXCH/ARRIVE/PHASECHK/completion-bearing LOAD event
```

Because the additional per-CTA atomic can perturb an instrumented execution,
the regenerated trace must also preserve the existing semantic invariants:

```text
same kernel and CTA counts
same active semantic TMA/BAR/SYNCS occurrence counts
same ready-only versus captured-poll classification
same K5/K10 completion-payload coverage
```

Any path-classification change makes the capture invalid. Do not accept a
different branch path merely because it produces an order.

#### Capture feature gate

Add one independent opt-in tracer configuration:

```text
SYNC_EVENT_ORDER=1
```

This is synchronization provenance, not a trace-size or transport optimization,
so it is intentionally independent of `LIGHTER_TRACER`. It must work with both
legacy and compact wire formats and must not be added to
`LIGHTER_TRACER_FEATURE_ORDER`, `LIGHTER_TRACER_IMPLEMENTED_FEATURES`, or
`LIGHTER_TRACER_PRODUCTION_FEATURES`.

`SYNC_EVENT_ORDER` is disabled by default. `run_hw_trace.py` and the tracer both
parse it as a strict boolean and print the resolved state at startup:

```text
[SYNC-ORDER] enabled=1 mode=cta-interval-v1
```

Feature-off contract:

```text
no begin/end callbacks inserted
no CTA counter allocation
no order-event records emitted
no sync_event_order.jsonl created
existing compact-wire bytes and protobuf output unchanged
```

Feature-on startup rejects:

```text
WARP_LIFETIME_ONLY
ACTIVE_FROM_START=0
INSTR_BEGIN != 0
INSTR_END != UINT32_MAX
```

`LIGHTER_TRACER=off`, compact wire, operand pruning, host counting, and sparse
reconstruction remain orthogonal choices. Enabling or disabling any of them
must not implicitly enable or disable sync-event ordering. Unsupported
combinations fail explicitly rather than rewriting either configuration.

Static instrumentation validation requires that every opcode in the closed
event set receives both callbacks. Post-capture join validation detects any
missing relevant dynamic occurrence. These checks do not belong to startup,
where the dynamic occurrence set does not yet exist.

The sidecar generator exposes a separate explicit requirement:

```text
--require-sync-event-order
```

Schema-v6 occurrence tickets may be emitted only with that option and a
complete validated `sync_event_order.jsonl`. Older traces remain usable for
schema-v5 analysis but cannot silently enable ticketed WaitBlock.

#### Offline generation resolver

The resolver consumes the final simulator-facing protobuf, the existing sync
and TMA metadata, and `sync_event_order.jsonl`. It processes each CTA
independently using the begin/end partial order.

For the current FA3 subset:

1. `SYNCS.EXCH` initializes generation zero and the expected arrival count.
   More than one active EXCH for one barrier fails as unsupported.
2. A count-only barrier advances when ordered active ARRIVEs reach its exact
   expected count. Active thread contribution is derived from the captured
   active/predicate mask.
3. The first active `ARRIVE_EXPECT_TX` for a barrier opens contract generation
   zero. The nth contract for the same barrier opens generation `n - 1`.
4. Opening contract `n` proves that release generation `n - 1` has already
   occurred. A valid program cannot reopen the same barrier while the prior
   generation still owns residual expected, assigned, completed, or in-flight
   state.
5. At a wait after `n` contracts have opened, a TMA-backed barrier's current
   generation is restricted to `{n - 1, n}`: the newest contract is either
   still active or has completed, and every older contract must already have
   completed.
6. The captured wait result and parity select between those adjacent
   generations because they have opposite parity.
7. Captured polling loops are grouped into episodes. Earlier loop observations
   are misses and the exit-producing observation is ready. The ticket belongs
   to the action occurrence at which the simulator may create the hold, but its
   target is the generation proven by the episode's final ready observation.
8. Ready-only entries use their captured ready arm as the observed result.
9. Per-warp program order, non-overlapping event intervals, and producer
   ownership add happens-before constraints. A candidate generation that
   requires an event downstream of the wait is rejected.
10. Enumerate all legal orders for overlapping intervals and all assignments
    that satisfy these constraints. Exactly one generation result across every
    legal ordering is required for each actionable occurrence.

Resolver outcomes are fail-closed:

```text
0 assignments -> INCONSISTENT_TRACE
1 assignment  -> emit exact ticket
2+ assignments -> AMBIGUOUS_GENERATION
```

There is no smallest, largest, nearest, or latest-compatible tie-break. The
current trace demonstrates why this matters: without cross-warp ordering, the
CTA124 warp-12 `0x74e0` occurrence retains release-generation candidates
`{1, 3}` after generation 5 and later are rejected by the provider cycle.

#### Ticket schema and identity

Add a schema-v6 occurrence table:

```json
{
  "trace_kernel_id": 5,
  "global_cta_id": 124,
  "warp_id": 12,
  "unique_function_id": 3,
  "block_id": "wb_fid3_e0x74e0",
  "pc": 29920,
  "trace_event_id": 1234,
  "begin_seq": 1234,
  "end_seq": 1237,
  "canonical_occurrence_index": 4256,
  "barrier_addr": 93264,
  "wait_parity": 0,
  "captured_result": "READY",
  "target_release_generation": 3,
  "proof": "CTA_EVENT_ORDER_V1"
}
```

`trace_event_id` is allocated from `begin_seq` and is the primary occurrence
identity. The canonical protobuf instruction index is retained as an
independent consistency check and to preserve identity through partial
extraction and replay. A static PC or a per-PC count alone is insufficient.

The post-processing pipeline must preserve this identity across:

```text
raw full trace
-> bootstrap ordinal join and immutable identity stamp
-> spin trimming
-> sparse reconstruction
-> strict WaitBlock sidecar generation by stable identity
```

Deleting a spin occurrence may remove its order event, but it must never
renumber retained events. Reconstructed instructions do not invent an event
sequence. Every runtime-action instruction must be a directly captured
occurrence with an exact ticket.

#### Simulator runtime contract

Add the captured `trace_event_id` and canonical occurrence index to
`trace_warp_inst_t`. IBuffer rollback, partial extraction, deferred replay, and
replay reservation preserve both values unchanged.

For every active WaitBlock action:

1. Find exactly one ticket by kernel launch, global CTA, CTA-local warp, and
   `trace_event_id`.
2. Require function, block, PC, barrier address, wait parity, and canonical
   occurrence index to match.
3. Set `pending.expected_gen` from `target_release_generation`; do not call
   `expected_gen_for_wait()` for a ticketed occurrence.
4. Apply trace-latched readiness:

```text
current generation < target release generation
    hold

current generation == target release generation
    release normally

current generation > target release generation
    release as an already-observed captured event
    increment and log waitblock_trace_generation_overshoots
```

The `current > target` case is not an arbitrary deadlock escape. It is permitted
only when an exact ticket with captured-ready provenance proves that this
specific occurrence observed the older release. Missing, duplicate, ambiguous,
or identity-mismatched tickets abort.

This trace-latched comparison applies to both `INJECT_MISSING_WAIT` and
`FOLD_CAPTURED_POLL`. Applying it only to Strategy A would leave the historical
pre-WaitBlock stale-poll failure unresolved.

B0-D remains unchanged:

```text
TMA completion target generation == live barrier generation
```

A stale TMA completion still aborts. Wait tickets describe captured observation
history; TMA generation identifies the only object a completion may mutate.
These are different contracts and must not share the `>=` rule.

#### Implementation order (of the rejected design)

1. Add and test the independent opt-in `SYNC_EVENT_ORDER=1` configuration,
   startup banner, invalid-mode checks, and feature-off identity in both
   `run_hw_trace.py` and the tracer.
2. Before changing the production schema, run the probe-only PHASECHK microtest
   for `IPOINT_AFTER` destination-predicate capture. Select either direct
   predicate capture or the proven following-branch fallback.
3. Add the versioned CTA-order transport record, protobuf identity fields, and
   per-launch counter allocation/lifecycle.
4. Emit one complete begin/end order record for every relevant active or
   predicated-off sync/TMA instruction and write `sync_event_order.jsonl`.
   Join only against raw, untrimmed protobuf and stamp immutable event/source
   identities before any transformation.
5. Implement and test raw-join validation using synthetic and microtrace
   fixtures.
6. Implement count-barrier and TMA-contract generation reconstruction using
   fixtures only.
7. Implement the uniqueness solver and negative ambiguity fixtures.
8. Enable the schema-v6 emission code path using fixture data only. Do not
   claim or require real K5/K10 tickets at this step.
9. Add stable occurrence identity to trace decode and replay.
10. Load and validate fixture tickets before enabling WaitBlock.
11. Replace live `expected_gen_for_wait()` binding for ticketed actions with the
    exact target and trace-latched comparison.
12. Run all focused fixture and feature-combination tests.
13. Perform the single K5/K10 retrace with `SYNC_EVENT_ORDER=1`. In this same
    stage, join the raw traces, stamp stable identities, run transformations,
    resolve generations, require zero missing/duplicate/ambiguous tickets, and
    emit the real schema-v6 sidecars.
14. Run one final WaitBlock-on K5 and K10 simulator validation using those
    tickets.

Focused acceptance:

```text
CTA-local sequence endpoints are unique and per-warp intervals do not overlap
order-event/protobuf join is bijective
SYNC_EVENT_ORDER=0 preserves the previous trace artifacts byte-for-byte
SYNC_EVENT_ORDER=1 emits a complete validated order sidecar
SYNC_EVENT_ORDER=0 x LIGHTER_TRACER=off is bit-identical
SYNC_EVENT_ORDER=0 x compact_wire is bit-identical
SYNC_EVENT_ORDER=0 x compact_wire,prune_operands is bit-identical
SYNC_EVENT_ORDER=1 x LIGHTER_TRACER=off captures and joins every required event
SYNC_EVENT_ORDER=1 x compact_wire captures and joins every required event
SYNC_EVENT_ORDER=1 x compact_wire,prune_operands captures and joins every required event
every unsupported host_count/sparse-reconstruct combination rejects explicitly
all actionable K5/K10 occurrences have one ticket
CTA124 warp12 0x74e0 resolves to release generation 3
CTA124 warp0 0xb370 resolves to release generation 4
current > target releases only with captured-ready ticket provenance
partial replay preserves the original ticket identity
no semantic TMA/BAR/memory occurrence is added, removed, or reordered
TMABIND stale-generation and exact ownership guards are unchanged
K5 exits with WAITBLOCK and TMABIND final PASS
K10 retains B0-D and replay/cursor invariants
```

Known risks:

- The two CTA-local atomics add trace-time perturbation. Limit them to the closed
  event set and reject any regenerated trace whose control-path classification
  changes.
- Callback intervals may overlap across warps. Such events remain unordered;
  the resolver must prove one result across every legal ordering or fail closed.
- Verify `IPOINT_AFTER` support and destination-predicate capture for each
  PHASECHK form in the pre-schema microtest. If the predicate cannot be captured
  reliably, use the already-proven following branch path as the result source;
  never infer it from parity.
- The order is the instrumented execution order, not an uninstrumented hardware
  timing oracle. It is valid for reconstructing the captured trace's logical
  epochs, not for calibrating simulator latency.
- TMA completion is asynchronous and is not directly timestamped. The resolver
  relies on same-barrier contract lifecycle plus captured wait outcomes; if
  those constraints do not produce one generation, sidecar generation aborts.
- Repeated EXCH epochs remain unsupported until the ticket key includes an
  explicit initialization epoch.

### 7.4 Review dispositions

Reviewer follow-up (2026-08-26) raised four additional blocking concerns beyond
the first five. Their dispositions:

- **Overshoot data-lifetime hazard is NOT critical for this simulator.** The
  barrier model stores only counters and ownership metadata, never tensor data.
  `recompute_sync_barrier_ready_and_maybe_flip_phase` resets `arrive_count`,
  `expected/assigned/completed_tx_bytes`, `outstanding_tma_bindings`,
  `tx_contract_id`, and `assignment_*` only; there is no shared-memory buffer
  whose reuse could corrupt a value. A `current_gen > target_gen` release
  therefore cannot produce numeric corruption here. Its only effect is timing
  progression plus counter accounting. The overshoot abort at
  `sm.cc` L3473-L3493 is thus a MODEL-DIVERGENCE auditor, not a
  data-correctness guard. It should be described that way and kept fail-closed
  for divergence detection, not for data safety.
- Transport mixing (issue 6), per-launch counter pointer lifecycle (issue 7),
  and the contract-count `{n-1,n}` premise (issue 8) remain valid
  implementation blockers and are independent of the direction chosen. They are
  the blockers to revive only if this design is ever reopened.

The corrected conclusion (generation removal is not a deadlock fix; the primary
axis is timing realignment plus global park safety) is the current authority in
[WAIT_BLOCK.md](WAIT_BLOCK.md).

---

## 8. Closed experiment: post-`951b058` ordered candidate-set

This experiment tested whether stronger CTA-local trace ordering could recover
a complete set of legal release generations without an application, kernel,
PC, CTA, barrier, warp, occurrence, or generation special case. It comprised
the five commits from `9ee8f35` through `97c1bae`: ordered event capture,
offline joining, candidate generation, baseline-path handling, managed-buffer
transport, and the callback deadlock correction. Simulator source and
configuration remained frozen throughout.

**Capture and conservation result**

The final ordered K5/K10 run completed all 11 kernels:

```text
order records            = 1,635,080
protobuf semantic events = 1,635,080
joined events             = 1,635,080
join bijective            = true
record overflow           = 0

TMA contracts             = 12,408
completion-bearing LOADs  = 19,320
expected/assigned bytes   = 221,380,608 / 221,380,608
TMA errors                = 0
```

The non-PHASECHK semantic multisets were conserved globally:

```text
K5  non-PHASECHK semantic events = 268,536 / 268,536
K10 non-PHASECHK semantic events = 448,128 / 448,128
```

The instrumentation nevertheless changed timing-sensitive polling and K5
persistent work placement:

```text
                         baseline   ordered   delta
K5 PHASECHK               235,692   196,342  -39,350
K10 PHASECHK              623,254   507,330 -115,924
K5 actionable waits        26,934    26,956      +22
K10 actionable waits       60,829    60,856      +27
K5 per-warp non-poll mismatches              364 / 2,112
K10 per-warp non-poll mismatches               0 / 4,608
```

Consequently, the ordered run could validate capture completeness and global
semantic conservation, but it could not transfer per-CTA/per-warp wait
identities or causal authority back to the baseline trace.

**V2 result**

The timestamp-only candidate resolver covered all 87,812 ordered actions and
failed closed without simulator-log input. For the critical ordered occurrence:

```text
K5 CTA124 warp12 0x74e0 / barrier 0x16c50
candidate set = {1,3,5}
```

The `.e85` baseline is stuck at `current_gen=4`, so this set does not authorize
the required must-complete release. V2 therefore did not solve B0-E.

**V3 hypothesis and pre-implementation audit**

V3 proposed rebuilding canonical events from the baseline trace, using the
ordered trace only for global conservation, deriving phase-token lower bounds
with reviewed scalar/uniform ALU semantics, and testing each candidate with a
candidate-specific graph/CSP. Its critical candidate-5 hypothesis required
this must-dependency:

```text
W_consumer(B50)
  -> A_consumer(B60)
  -> R(B60,4)
  -> W_producer(B60)
  -> C_producer(B50,5)
```

The pre-implementation CTA124 audit invalidated the required middle edge:

```text
B60 expected_arrive = (0x200000 - 0x1ffffa) / 2 = 3
B60 contributing warps = {4,8,12}
effective B60 ARRIVEs per contributing warp = 9
effective B60 ARRIVEs available before the target continuation = 4 per warp
```

Warp 12's first four contributions are in same-warp program order before its
target B50 wait. Warps 4 and 8 are cross-warp unordered relative to that wait,
so a legal topological assignment may place their first four contributions
before it as well. Thus one legal provider assignment can reach `R(B60,4)`
without the target wait continuation. Proving that the producer wait has
minimum generation 4 would not prove that generation 4 depends on the consumer
wait.

Candidate pruning requires every legal assignment to be unsatisfiable. Because
the baseline evidence does not exclude this acyclic assignment, a sound solver
must retain candidate 5 or return unresolved; it cannot emit the proposed
`SELF_DEPENDENT_RELEASE_SCC`. Adding an ordered timestamp, simulator terminal
state, PC rule, or manually injected edge would violate the input boundary and
fail-closed policy.

**Disposition**

- No schema-v7 sidecar or simulator consumer was produced.
- The incomplete canonicalizer/symbolic prototype is discarded.
- The five Stage T commits from `9ee8f35` through `97c1bae` are reverted from
  the working code to the `951b058` state.
- Ordered trace artifacts may be retained as diagnostic evidence, but are not
  production inputs and do not justify another retrace.
- Do not resume candidate-set pruning unless new baseline-authoritative
  semantics prove a must-edge that excludes the independent B60 generation-4
  assignment without simulator logs or hardcoded identities.

The concrete next step is the first-divergence and park-safety analysis
described in [WAIT_BLOCK.md](WAIT_BLOCK.md), not a new trace schema.
