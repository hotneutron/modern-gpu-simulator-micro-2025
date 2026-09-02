# FWD drain-idle — 4축 통합 계측 (1회 12h run으로 전부 판정)

## Context (왜 한 번에 전부 재야 하는가)

정본 run(fwd `OnlyKernel5/.o42`, bwd `OnlyKernel10/.o25`) 측정: fwd·bwd 모두 SM drain-idle이 ~45%로
HW(~9.7~11%)보다 큼. 증상은 같지만 원인이 다름:
- **bwd**: causal-mask 삼각 부하 불균형(elapsed spread 92.2%, r(tensor_ops,elapsed)=+0.99) → HW도 겪는
  실제 workload, **충실 재현, 회수 대상 아님**.
- **fwd**: 모든 CTA 균일 ~47% idle(spread 12.5%). warp-specialization 구조에서 생산자(TMA) warp는 trace를
  일찍 소진, 소비자 warp는 mbarrier에서 대기. per-SM `wait_pending`↔elapsed r=+0.69(bwd r=+0.06) →
  **fwd 고유 회수 후보**.

> ⚠️ **용어 주의 — `wait_barrier`(mbarrier) ≠ NCU `barrier`.** 이 문서에서 말하는 mbarrier 대기는 Hopper
> **async barrier**(소비자가 TMA 데이터 도착을 기다림)이며, NCU taxonomy에서는 **`long_scoreboard`**
> (=`wait_barrier`+`tma_flush`, fwd 12.8%)로 매핑된다. NCU **`barrier`**(=`BAR.SYNC`/`__syncthreads` CTA
> 랑데부, fwd 0.08%)와는 **다른 메커니즘**이다. 아래 모든 "mbarrier/wait_barrier"는 async barrier(12.8%)를
> 뜻하며, NCU `barrier`(0.08%)가 아니다.

**한 run(12h)으로 전부 답해야 할 질문:**
1. fwd 43.5% SM-idle을 겹침 없이 분해하면 **floor**(전부 trace 소진, 회수 불가) vs **mbarrier-only**(회수
   가능) vs 나머지 단일원인/mixed가 각각 몇 %?
2. 그 idle의 주체가 **생산자(early-drain)**냐 **소비자(mbarrier wait)**냐? (mechanism-1 vs 2 분리)
3. 소비자 mbarrier 대기가 **짧은 것 다수**(HW가 오버랩으로 숨김 → 회수 가능)냐 **긴 것 소수**(실제
   데이터 지연 → floor)냐?
4. per-CTA로 위 신호들이 elapsed와 어떻게 상관되나 (fwd 인과 확정 + bwd 대조)?

현재 카운터로는 불가: `sm_idle_reason_*`는 OR-집계라 겹침(합>idle), warp-role 구분 없음, mbarrier 대기는
count만 있고 지속시간 없음. → **4축 계측을 한 번에** 넣는다.

전부 **observe-only, 기존 gate 재사용, default-off → timing-neutral**(bit-identity 유지). 단 축3은 이미
timing에 반영된 trace spin을 *관측만* 함(모델 불변).

---

## 축 1 — SM-idle 완전분해 (겹침 0, 합 == sm_all_subcores_idle)

subcore별 "단독 차단 이유(sole-block)"를 이미 계산된 `is_*` 플래그만 읽어 결정(side-effect 없음):

| 코드 | 의미 |
|---|---|
| `ISSUED` | 이 subcore가 issue함 |
| `DRAINED` | valid head warp 없음(전부 trace 소진) = **floor** |
| `ONLY_WAIT_BARRIER` | 유일 미충족 = `are_wait_barriers_ready`(mbarrier) |
| `ONLY_TENSOR` | 유일 미충족 = tensor FU lockout (기존 ceiling과 동형) |
| `ONLY_STALL_COUNT` | 유일 미충족 = fixed-latency dep (NCU `wait`) |
| `ONLY_FU_NONTENSOR` | 유일 미충족 = 비-tensor FU busy |
| `ONLY_NEXT_STAGE` | 유일 미충족 = 다음 stage back-pressure |
| `ONLY_L1C` | 유일 미충족 = const-cache |
| `MULTI` | 미충족 조건 ≥2 (단일 lever로 회수 불가) |

"단독" 판정은 기존 `blocked_only_by_fu`(subcore.cc:696-704)와 동일. subcore에 valid-head warp가 여럿이면
회수 가치 우선순위(WAIT_BARRIER>TENSOR>STALL_COUNT>FU_NONTENSOR>NEXT_STAGE>L1C>MULTI)로 대표 1개.
**SM 레벨**: `!any_subcore_issued`인 idle cycle에서 4 subcore의 sole-reason을 우선순위 최댓값 1개로 묶어
그 cycle을 **정확히 1개 partition에 귀속** → 합 == `sm_all_subcores_idle`(겹침 0).

**카운터**: `total_num_cycles_sm_idle_partition_{drained, only_wait_barrier, only_tensor, only_stall_count,
only_fu_nontensor, only_next_stage, only_l1c, multi}` (8개). + 자기검증용 독립 ceiling
`total_num_cycles_sm_idle_all_blocked_by_wait_barrier`(기존 tensor ceiling과 교차검증).

---

## 축 2 — 생산자/소비자 warp-role 분해 (mechanism-1 vs 2)

Explore 확인: role 필드는 없지만 신호는 이미 관측됨. **sticky per-warp 플래그 2개**를 첫 발생 시 set:
- **producer**: TMA/arrive_expect_tx 발행 — `sm.cc:1881` `bind_tma_completion_to_sync_barrier(warp_id,...)`
  또는 `tma_unit_sm.cc:398` `cmd.warp_id`. (FA3에서 producer warpgroup만 UTMALDG+expect-tx.)
- **consumer**: WGMMA(TENSOR_CORE_OP) 발행 — `functional_unit.cc:178` `is_tensor_core_op()`
  (기존 `inc_tensor_ops_for_warp` 패턴 그대로).

**role 겹침/미분류 규칙(명시)**: 두 플래그는 독립 sticky bool이라 4가지 상태가 가능:
`producer-only`, `consumer-only`, `both`(TMA와 WGMMA 둘 다 발행), `neither`(예: reduction/epilogue-only warp).
집계는 **4개 버킷 모두** 유지(합쳐서 뭉개지 않음). FA3는 producer/consumer가 분리돼 있어 `both`≈0으로
예상되지만, **`both`가 크면 role-분리 가정 자체가 틀린 것**이므로 반드시 관측해 검증. `neither` warp도
drain/idle을 따로 집계(어느 role에도 안 잡히는 idle이 얼마인지).

**per-warp drain cycle**: trace 소진 시점 stamp — `sm.cc:902`(`trace_done() && functional_done()`)에서
`m_warp_drain_cycle[wid]=gpu_sim_cycle`. (기존엔 `m_last_fetch`만 있음=fetch≠drain, 신규 필요.)

**집계**: per-CTA로 producer 최종 drain cycle vs consumer 최종 drain cycle, 그리고 role별 idle 기여.
→ "생산자가 X% 지점에 소진 후 놀고, 소비자가 mbarrier로 Y cycle 대기" 를 직접 수치화.

**카운터(per-CTA slot 배열)**: `m_producer_last_drain_cyc_by_cta_slot`, `m_consumer_last_drain_cyc_by_cta_slot`,
그리고 축1의 `only_wait_barrier`를 role로 나눈 `m_consumer_wait_barrier_only_cyc_by_cta_slot`.

---

## 축 3 — mbarrier 대기 지속시간 히스토그램

**중요 제약(Explore 확인)**: PHASECHK/TRYWAIT는 **완전 non-blocking**, 재시도는 trace의 spin loop가 담당.
sim은 warp를 blocked로 두지 않음(`sm.cc:1756-1760`). `m_pending_sync_waits`는 현재 매 호출 리셋되어
사실상 dead.

> ⛔ **결함 수정(2026-07-17 소스 검증) — `m_pending_sync_waits`를 되살리면 bit-identity가 깨진다.**
> `warp_waiting_at_barrier`(sm.cc:1621)는 `m_pending_sync_waits[warp_id].valid`가 true면 true를 반환하고
> (sm.cc:1642), 이 값은 `check_if_non_released_reduction_barrier`(sm.cc:1612, **issue/실행 경로의
> BARRIER_OP+RED 처리**)에서 소비된다. 지금은 `valid`가 항상 false라 dead지만, 축3이 첫 miss에서
> `valid=true`를 저장하면 그 dead 경로가 부활해 `non_released_barrier_reduction` 결정이 바뀌고 **timing이
> 변한다**. 게다가 그 경로엔 `m_sync_debug_wait_released++` side-effect(sm.cc:1633)까지 있다.
> **따라서 `m_pending_sync_waits`를 절대 건드리지 않는다. 완전 분리된 관측 전용 배열을 새로 만든다.**

- **신규(관측 전용, issue 경로와 무관)**: `sm.h`에
  `address_type m_sync_first_miss_barrier[MAX_WARPS] = {0};`(대기 중인 barrier_addr, 0=없음),
  `uint64_t m_sync_first_miss_cycle[MAX_WARPS] = {0};`,
  히스토그램 `uint64_t m_sync_debug_wait_cycle_hist[6] = {0};`.
- **첫 miss 기록**: `sm.cc:1759`(wait_pending++ 분기) 옆에서 — `m_sync_first_miss_barrier[warp_id]`가
  현재 key.barrier_addr와 다르면(=새 대기 시작) `barrier=key.barrier_addr; cycle=get_current_gpu_cycle()`
  저장; 같으면 보존(반복 spin에서 시작 시점 유지). **`m_pending_sync_waits`는 손대지 않음.**
- **통과 측정**: `sm.cc:1737`(hit 분기) 옆에서 — `m_sync_first_miss_barrier[warp_id]==key.barrier_addr`면
  `delta=get_current_gpu_cycle()-m_sync_first_miss_cycle[warp_id]` → 버킷++, 그리고
  `m_sync_first_miss_barrier[warp_id]=0`(클리어). 첫 시도 즉시 hit(=miss 이력 없음)이면 barrier가 0이라
  건너뜀 = 버킷0(즉시) 의미.
- **버킷**: {0, 1-16, 17-64, 65-256, 257-1024, 1024+} (sm.h `m_sync_debug_*` 블록).
- **CTA/warp reset 시 클리어**: 기존 리셋 지점(sm.cc:1344-1345, 1921-1924)에서 두 신규 배열도 0으로
  (누수 방지). 이 배열은 어떤 issue 결정에도 읽히지 않으므로 timing-neutral 보장.

**주의**: 이는 순수 관측 — spin loop 자체가 이미 timing에 반영돼 있으므로 measurement가 cycle을 바꾸지
않음(bit-identity 유지).

**해석**: 짧은 대기(버킷 1-64) 다수 = HW가 warp 스위칭으로 숨길 수 있는 부분(회수 후보). 긴 대기(257+)
소수 = 실제 데이터 지연(floor). → 축1 `only_wait_barrier` 회수분의 상/하한을 좁힘.

---

## 축 4 — per-CTA [CTAFIN] 원시값 확장

기존 `[CTAFIN]` 컬럼(`elapsed_cyc, tensor_ops, sm_idle_tensor_cyc, fu_occupied_tensor_cyc, sm_idle_cyc,
sm_idle_ibuffer_empty_cyc`)에 추가:
- `sm_idle_wait_barrier_only_cyc` (축1, 회수 후보 per-CTA)
- `sm_idle_drained_cyc` (축1, floor per-CTA)
- `producer_last_drain_cyc`, `consumer_last_drain_cyc` (축2)
- `wait_pending`, `wait_released` (기존 per-SM SYNCDBG를 per-CTA로 — mbarrier 테스트 count)

→ fwd/bwd 각각 r(각 신호, elapsed) 계산으로 인과 확정 + 대조.

---

## 구현 (Explore로 확정된 삽입 지점; 파일 `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/`)

### subcore.cc / subcore.h (축1)
- `enum Step0SoleBlock {...}` (subcore.h). member `m_step0_sole_block_this_cycle` + getter
  (subcore.h:123/163 부근). set at `subcore.cc:828`.
- sole-block 계산: issue 루프 valid-head 처리부(`subcore.cc:660-760`), 기존 `blocked_only_by_fu`(696-704)
  일반화. 변수 전부 존재(`are_wait_barriers_ready` 587, `is_fu_available` 606, `is_stall_counter_0`,
  `is_l1c_ready` 609 등). gate `wgmma_step0_instrument_enable`.

### functional_unit.cc (축2 consumer)
- `functional_unit.cc:178` `is_tensor_core_op()` 옆에 per-warp consumer sticky set(신규 `mark_consumer_warp(wid)`).

### tma_unit_sm.cc / sm.cc (축2 producer)
- `sm.cc:1881` (bind_tma_completion) 또는 `tma_unit_sm.cc:398`에 producer sticky set.
- per-warp drain stamp: `sm.cc:902`.

### sm.cc / sm.h (축1·2·4 집계 + CTAFIN)
- 집계부 `sm.cc:553-634` idle 블록에 축1 partition 8개 + role별 귀속.
- per-CTA 배열(sm.h:504/511 부근): `m_sm_idle_partition_*_by_cta_slot`는 과하니 핵심만 —
  `m_wait_barrier_only_cyc_by_cta_slot`, `m_drained_cyc_by_cta_slot`,
  `m_producer_last_drain_cyc_by_cta_slot`, `m_consumer_last_drain_cyc_by_cta_slot`,
  `m_wait_pending_by_cta_slot`, `m_wait_released_by_cta_slot`.
- launch 리셋 `sm.cc:962-963`, exit 리셋 `sm.cc:1389`.
- `[CTAFIN]` printf `sm.cc:1369-1375`에 축4 컬럼 append.

### sm.cc / sm.h (축3 — 관측 전용 별도 배열; `m_pending_sync_waits` 미사용)
- `sm.h`: `m_sync_first_miss_barrier[MAX_WARPS]`, `m_sync_first_miss_cycle[MAX_WARPS]`,
  `m_sync_debug_wait_cycle_hist[6]` (기존 `m_sync_debug_*` 블록 옆).
- `sm.cc:1759`(첫 miss 기록, 별도 배열만), `sm.cc:1737`(통과 delta→버킷, 배열 클리어).
- reset: `sm.cc:1344-1345`(warp/CTA start), `sm.cc:1921-1924`(teardown)에 두 배열 0.
- gate `-sync_wait_hist_instrument_enable`(신규, default 0). **`tma_types.h`/`HopperMBarrierPendingWait`는
  건드리지 않음**(bit-identity 결함 회피 — §축3 결함 수정 참조).

### gpu-sim.cc (stat-key 등록 + 신규 플래그)
- `gpu-sim.cc:2838` 옆에 partition 8개 + `sm_idle_all_blocked_by_wait_barrier` +
  히스토그램 버킷 6개를 `add_unsigned_long_long_stat(...)` 등록.
- `-sync_wait_hist_instrument_enable`(OPT_INT32, default 0) 등록(축3 gate).

**gate 요약**: 축1·2(`-wgmma_step0_instrument_enable`) + 축4 per-CTA 컬럼(`-cta_stall_breakdown_instrument_enable`,
정본 `.o42` 이미 on) + 축3(`-sync_wait_hist_instrument_enable`, 신규). 셋 다 default-off → 미설정 시
byte-identical. 축별 독립 gate라 bit-identity 실패 시 이분 진단 가능.
**헤더 변경(subcore.h, sm.h) → `make clean` 필수.**

---

## config (12h run — 값만, 주석 보존)
`SM90_H100_L2_50MB_80GB/gpgpusim.config`:
- `-wgmma_step0_instrument_enable 1`, `-cta_stall_breakdown_instrument_enable 1` (이미 on) — 축1·2·4.
- `-sync_wait_hist_instrument_enable 1` (신규, default 0) — 축3. 독립 gate라 bit-identity 실패 시 이것만
  꺼서 축3을 용의선상에서 이분 가능.
- Opt9 timing 노브 그대로. `-tma_debug_enable 0`, `INTERCONNECT` trace 제외 유지(로그량).
- **주의**: per-event 로그는 절대 켜지 말 것(12h run 로그 폭증) — 모든 축은 집계값만 end-of-run/[CTAFIN] 출력.

## 판정 규칙 (run 후, fwd; bwd 대조)

**핵심: 회수 판정은 sim 절대%가 아니라 HW 앵커 대비다.** HW도 mbarrier 대기(`long_scoreboard` 9.8%)와
SM-idle(9.7%)이 있으므로, 회수 가능분 = **sim의 mbarrier-idle 중 HW가 오버랩으로 숨기는 초과분**.
고정 HW 앵커(fwd NCU, 이미 문서에 있음):

| HW 앵커 (fwd) | 값 | 용도 |
|---|---|---|
| SMSP-Active | **88.9%** | sim subcore-active 63.9%와 비교 → 회수 상한 = (88.9−63.9)=25pp |
| `long_scoreboard`(mbarrier) | **9.8%** | sim `only_wait_barrier` 정규화값과 직접 비교 |
| Eligible Warps/sched | **0.83** (sim 0.53) | mbarrier를 숨기는 능력의 대리지표 |
| SM-idle | **9.7%** | sim 43.5%와 비교 → floor 대비 초과 idle |

판정(`only_wait_barrier`를 `gpu_sim_cycle×132`로 정규화 후 **HW 9.8%와 비교**):

| sim `only_wait_barrier` vs HW 9.8% + 축2·3 교차 | 결론 |
|---|---|
| ≫ 9.8% (예: 25%+) & 축3 짧은버킷 다수 & 축2 consumer-mbarrier 지배 | HW가 숨기는 초과분 큼 → **fwd mbarrier-overlap lever 추진** |
| ~9.8% 근처 | sim이 이미 HW만큼 → mbarrier는 lever 아님, 다른 partition(next_stage 등) 재검토 |
| `drained`(floor)가 idle의 대부분 & 축3 긴버킷(257+) 지배 | 실제 데이터 지연 = faithful floor → fwd 2.01×를 floor로 인정 → **Deferred** |

교차검증(내부 일관성):
- partition `only_tensor` ≈ 기존 `sm_idle_all_blocked_by_tensor`(machinery 검증).
- 축2: producer가 이른 %지점에 drain하고 consumer가 mbarrier로 대기하면 mechanism-2 확정. `both`≈0 확인.
- 축3: 히스토그램이 짧은쪽(1-64)이면 회수↑, 긴쪽(257+)이면 floor↑ — 축1 회수분의 상/하한을 좁힘.
- bwd 대조: 같은 컬럼에서 `only_wait_barrier` 낮고 `drained`/density 지배 예상(회수 대상 아님 재확인).

## Verification (timing-neutral + 실패 시 진단)
1. `make clean && make` (헤더 subcore.h/sm.h 변경).
2. fwd K5 / bwd K10 재실행. **bit-identity gate**: `gpu_sim_cycle` == fwd 136,293 / bwd 215,537 정확 일치.
3. **bit-identity 실패 시 fallback(축 독립 gate)**: 만약 cycle이 어긋나면 어느 축이 범인인지 이분하기 위해
   각 축을 독립 플래그로 분리 가능하게 구현한다 — 축1·2(`wgmma_step0_instrument_enable`), 축3(신규
   `-sync_wait_hist_instrument_enable` default 0). 우선 축3만 끄고 재현 → 축3(sync 경로 인접)이 가장
   위험하므로 첫 용의자. 축1·2는 순수 read-only 카운터라 안전할 것으로 예상.
4. `[CTAFIN]` 신규 컬럼 132/384줄; **축1 partition 합 == `sm_all_subcores_idle`**(fwd 7,827,622) — 겹침0 검증.
5. `only_tensor` ≈ 기존 `sm_idle_all_blocked_by_tensor` — 교차검증. `only_wait_barrier` < 기존 overlap
   `sm_idle_reason_wait_barrier`(fwd 4.48M) — 부분집합 sanity.
6. 축3 히스토그램 총합 ≈ 기존 `wait_pending`(fwd~3124/SM×132) sanity(첫-miss만 세므로 wait_pending 이하).
7. `both` role 버킷 ≈ 0 확인(아니면 role 가정 재검토).

## Run 후 분석 산출물 (재현 가능하게 명시)
- `grep '^\[CTAFIN\]' <out> > ctafin.txt` → python으로 per-CTA 파싱, fwd/bwd 각각:
  r(only_wait_barrier, elapsed), r(drained, elapsed), r(wait_pending, elapsed) 및 각 신호 mean%elapsed.
- end-of-run 집계: partition 8개 정규화%, `sm_idle_all_blocked_by_wait_barrier`, 히스토그램 6버킷,
  role 4버킷 drain-cycle/idle.
- 결과 표를 `.result/FA3_progress.md` live Ongoing 항목에 추가(HW 앵커 대비 회수분 포함).

## 범위 밖
- 타이밍 모델 변경 없음(계측만). lever 추진은 run 결과가 결정.
- bwd 부하 불균형 미변경(HW-충실).
