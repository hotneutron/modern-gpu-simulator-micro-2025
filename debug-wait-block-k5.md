# Debug Session: wait-block-k5
- **Status**: [OPEN]
- **Issue**: Kernel 5 still fails after the latest WaitBlock implementation; verify the root cause and the prior analysis in `.plan/WAIT_BLOCK.md`.
- **Debug Server**: Not started; existing `.e83` and `.o83` logs are the initial pre-fix evidence.
- **Log File**: `simulator-remodeled/sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/H100_80GB-OnlyKernel5/*.e83` and `*.o83`

## Reproduction Steps
1. Run the Kernel 5 simulation using the configuration represented by the `.e83` and `.o83` logs.
2. Observe the terminal failure, watchdog, or lack-of-progress report.
3. Compare the terminal state with the expected WaitBlock release and replay lifecycle in `.plan/WAIT_BLOCK.md`.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Expected Signal |
|----|------------|------------|--------|-----------------|
| A | WaitBlock release is stronger than the actual producer progress condition and leaves a circular all-live wait. | High | Low | **Rejected as primary cause.** WaitBlock observes genuinely incomplete barriers; TMA credit state is already corrupt before the terminal hold set. |
| B | Partial replay restores an inconsistent IBuffer or scoreboard state despite using `selected_pc + width`. | Medium | Medium | **Rejected.** The run has 48 real replay drains and 48 matching closes, no synthetic/replay abort, and SM121 has no deferred-replay miss. |
| C | Barrier lifecycle ordering leaves a stale dependency after warp exit, CTA drain, or barrier reuse. | High | Medium | **Rejected for this failure.** The stale state is produced by identity-free completion routing within one live CTA generation, not slot reuse or cleanup. |
| D | The watchdog falsely classifies a long-latency but progressing interval as deadlock. | Low | Low | **Rejected.** SM121's last TMA completion is at cycle 37,845; all 13 participating warps remain held until the cycle-68,271 abort. |
| E | The prior document correctly describes the terminal state but misidentifies the earlier causal transition. | High | Medium | **Confirmed.** The terminal dependency cycle and FIFO spill are real, but the document missed the upstream 24-KiB/16-KiB contract mismatch and contained chronology/count errors. |

## Instrumentation Plan
The existing `.e83` instrumentation is sufficient to prove the invalid
completion routing. Before a logic fix, the remaining observation must bind
each LOAD command to:

1. TMA PC and command UID.
2. Exact completion barrier key and generation.
3. Captured expect-tx bytes.
4. Descriptor-derived memory bytes.

This must distinguish whether the `0x16c00` 24-KiB expectation means a 24-KiB
transfer, a 16-KiB transfer plus another 8-KiB producer, or separate
completion-credit semantics.

## Log Evidence
- `.e83:199103`: primary abort at cycle 68,271, SM121, warp 7,
  `INJECT_MISSING_WAIT`, barrier `0x16c38`, generation `4 -> 5`, held 32,001
  cycles.
- Reconstructed terminal pending map contains warp IDs `0` and `4-15`: 13
  participating warps, not 16.
- `.e83:18275-18316`: `0x16c30` binds 16,384 bytes and LOAD uid 6 enqueues
  16,384 bytes.
- `.e83:23764-23883`: `0x16c00` binds 24,576 bytes, but LOAD uid 7 enqueues
  16,384 bytes.
- `.e83:26014-26016`: uid 7 credits only 16,384/24,576 to `0x16c00`.
- `.e83:27765-27935`: `0x16c38` binds 16,384 bytes and LOAD uid 8 enqueues
  16,384 bytes.
- `.e83:32823-32827`: uid 8 completion is split across two barriers:
  8,192 bytes finishes stale `0x16c00`, and 8,192 bytes incorrectly enters
  `0x16c38`.
- Whole-log audit: 4,108 bind records and 4,108 LOAD enqueues; 192 parseable
  issue-order pairs have the same 24-KiB/16-KiB mismatch at `0x16c00`.
- Whole-log audit: at least 40 LOAD completion UID inversions, proving that a
  completion-order per-warp FIFO cannot preserve command-to-barrier identity.
- Whole-log audit: 48 `partial-replay-drained` events and 48 matching
  `replay_progress_closed=1` events; no partial replay, reservation, or
  synthetic-issue abort.
- `.plan/TMA_ARCH.md` previously contradicted its own geometry formula by
  pairing `box_dim=[64,128]` with a 24,576-byte/192-request result. K5 `.e70`
  through `.e83` consistently report 16,384 bytes/128 requests for `0x96b0`;
  the stale validation wording was corrected.
- Prior-design comparison:
  - `.trae/documents/SPIN_WAIT_FOLD_ROOT_CAUSE_PLAN.md` Batch B Stage 0.1 is the
    directly matching prior work item: transfer-UID-owned exact barrier credit.
    Its K5/K10 UID-inversion evidence is confirmed by `.e83`.
  - Its original allowance for one transfer to span multiple pending barrier
    ranges is insufficient for SM121. The relevant uid6-uid9 completions are
    in order; the 24-KiB/16-KiB mismatch causes the cross-barrier split even
    without completion reordering.
  - `.plan/H100_sync_impl.md` already identified the failure condition
    "waiter on barrier A, completion updates barrier B, kernel deadlocks."
  - `.plan/Full_sync_impl.md` prescribed instruction-derived barrier identity
    (`UTMALDG` destination register + 1; explicit UBLKCP barrier operand) and
    warned that a mutable per-warp active slot could be overwritten.
  - Commit `dcd9566b` did not implement that exact command-owned identity. It
    introduced the current per-warp byte FIFO directly, and `TMACommand` never
    carried a completion barrier.
  - `.e83` is therefore the same fundamental identity-loss bug, expressed
    through FIFO ordering/byte spill rather than active-slot overwrite.

## Verification Conclusion
The `.e83` reservation fix is validated. The current abort is a TMA
completion-accounting failure exposed by WaitBlock.

The prior `WAIT_BLOCK.md` analysis is directionally correct about the
identity-free FIFO spill, but incomplete:

- exact completion identity is mandatory because completions reorder;
- the first unresolved contract is `expected_tx=24576` versus modeled
  `load_bytes=16384`;
- identity-only routing would deadlock `0x16c00` earlier and is not a complete
  fix;
- the previous "all 16 warps" and "first WaitBlock hold at cycle 38,122"
  statements are factually incorrect.

No simulator logic change has been made. The session remains `[OPEN]` pending
design approval and a post-fix server run.

## Consolidated Design
`WAIT_BLOCK_TMA_OWNERSHIP.md` is now the authoritative, self-contained post-`.e83`
design. It incorporates the former Batch B work, supersedes cross-barrier
partial-range assignment, and includes implementation state, data structures,
identity and byte-contract gates, release ordering, lifecycle, metrics,
diagnostics, implementation order, and K5/K10 acceptance criteria.

## Post-Review Evidence Gates

- Existing `instruction.proto` has no TMA completion-barrier payload. Generic
  TMA operand values are not replayed from the protobuf, so production
  command-owned identity requires a trace schema extension and retrace.
- Static SASS shows completion-bearing `UTMALDG` and `UBLKCP` use the high UReg
  of the first destination-state pair as the mbarrier address.
- K5 full-trace grouping covers 5,112 LOAD occurrences. All contracts close
  exactly except `0x9680 -> 0x96b0`, which occurs 264 times and is always
  24,576 expected bytes versus one 16,384-byte modeled command.
- FA3 source defines the Q transaction as `M*K*sizeof(BF16)`. The traced kernel
  tile `(192,128,64)` gives 24,576 bytes, proving the `0x96b0` 64x128 mapping is
  wrong; it is not a distinct completion-credit-size case.
- K10 covers 13,824 LOAD occurrences with no contract mismatch:
  16,896-byte contracts are `16,384 + 512`, and 32,768-byte contracts are
  `16,384 + 16,384`.
- The observation-only trace patch now captures a dedicated TMA completion
  barrier payload and plumbs it through the trace parser. Scheduler and
  completion routing are unchanged. Retrace is pending.
