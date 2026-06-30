# TMA 주소 합성 + L2 Sector Merge 개선 Plan

> **SUPERSEDED — 통합 Opt 6로 병합됨.** 이 문서의 §2 수정1(mock base)·수정2(128B 발사)
> "Phase A"는 **롤백되어 현재 코드 트리에 없다** (머지된 것은 §5 `-tma_debug_*` 로깅뿐).
> 본 plan은 [TMA_LATENCY_INJECTION_H100.md](file:///home/jihyun/modern-gpu-simulator-micro-2025/.plan/TMA_LATENCY_INJECTION_H100.md)
> (통합 Opt 6)로 대체되었다. 그 문서가 같은 root code defect를 (A) 인터커넥트 주입 직렬화,
> (B) 32B×4 sector explosion, (C) 합성주소 hotspot의 세 결합 증상으로 묶어 **재구현**한다.
> 아래 내용은 root-cause 근거(§1)와 base 복원 난점(§1-B), 검증용 로깅(§5)의 **참고용**으로 보존.

대상 커널: `flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24`, OnlyKernel5
현재 sim cycle = 220,024 / HW(NCU) = 67,696 → **3.25x over**
주 원인: `inst_barrier` stall 56% → TMA `lat_mem` 평균 ~3,000cyc (HW L2-hit ~200-400cyc)

---

## 1. 근본 원인 (trace + 코드로 확정)

### (A) TMA 합성 주소에 `sm_id` / 실제 base 미반영
- 현재 AGU 주소: `agu_base = (transfer_uid << 20) + agu_index*128 + sector*32`
  - 위치: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.cc#L626-L640`
- `transfer_uid`는 **SM-local 카운터** → 모든 SM의 N번째 transfer가 동일 주소(`0xN00000`대) 발사
- 결과: 전 SM이 같은 L2 line을 두드림 → 인위적 hotspot → RESERVATION_FAIL 재시도 폭증
  - trace 증거: sector `0x5023c0`가 cycle 8948~13493 동안 132회 재프로브 (status=2→0)
  - 한 transfer 내 4 sector(c0/e0/400/420)는 서로 다른 주소라 transfer-내 중복은 없음
  - 진짜 중복 = transfer-간 / SM-간 동일 합성주소

### (B) descriptor 실제 base는 trace에 존재하나 누락됨 — 그리고 first_lane_addr로는 복원 불가 (확정)

#### (B-1) first_lane_addr가 안 잡힌 근본 원인 (코드+trace+SASS 교차검증 완료)
- kernel5(ufid=3) UTMALDG 실제 형태: `UTMALDG.4D [UR16], [UR8], desc[UR10]`
  (enhanced_execution_info.json 확인) — MREF 2개(`[UR16]`=SMEM dst, `[UR8]`=**TMA 좌표 레지스터**),
  DESC 1개(`desc[UR10]`)
- tracer는 `nvbit_add_call_arg_mref_addr64(instr, mem_oper_idx)`로 **MREF의 effective address**를
  잡는데(tracer_tool.cu#L1639-L1660), UTMALDG의 MREF는 **GMEM 포인터가 아니라 좌표 패킹값**
  → NVBit이 좌표에서 만든 의미없는 값이 찍힘
  - CSV 증거: desc_hi≠0인 진짜 UTMALDG의 first_lane_addr는 전부 음수형 garbage
    (`0xffffffffe628xxxx`), desc_hi=0(일반 LDG/STG)만 정상 GMEM(`0x00007fcb...`)
- `desc[UR10]`은 operand_type=DESC라 mref_addr64로 캡처조차 안 됨
- NVBit은 GPU **descriptor cache**를 읽을 훅이 없음 (TMA_TRACING.md §"Where the Gap Is")
  → **UTMALDG에서 GMEM base 재포착은 구조적으로 불가능**
- 결론: 기존 plan의 "first_lane_addr → base" 가정은 **폐기**. (B안=tracer 수정 재포착 불가)

#### (B-2) 진짜 base의 유일한 source = cuTensorMapEncodeTiled
- `tensor_map_encode_dump.csv`의 `global_address_hex` + blob 첫 8바이트 = 실제 GMEM base
  - box=64x192 → dump0 base=0x7fcce4c00000 / dump5(rank5) base=0x7fcb52a00000
  - box=64x128 → dump 1,2,3,4,6,7,8,9,10,11 (**10개 서로 다른 base**: Q/K/V/dO/dQ 등)
- `tma_descriptor_configs.json`은 정규화 시 **base를 의도적으로 제거**, 형상만 보존
  (TMA_TRACING.md L769: "differ only in global_address ... merge into one family")

#### (B-3) 현재 키로는 site→단일 base 매핑 불가 (핵심 난점)
- config `tm_r4_dt9_box_64x128x1x1` 하나에 **10개 base가 섞여 있음** (`source_dump_ids`)
- 같은 box_dim dump들의 handle_hi(q1)가 **전부 동일(0x...46530)**
  → runtime `handle_hi`로 base 구분 불가
- resolver 키 `(ufid, pc, handle_hi)`로는 어떤 텐서(base)인지 식별 못 함
- 즉 사용자 요청("desc mapping에 시작 주소만 추가")은 **encode dump 레벨에선 base가 있지만**,
  **runtime site → 그 중 어느 base인지 연결할 키가 현재 없다**는 게 진짜 막힌 지점

### (C) TMA sector merge(coalescing) 부재
- TMA는 일반 LD/ST 경로의 2단 coalescing(warp-내, InterWarpCoalescingUnit)을 모두 우회
- 128B AGU request → 32B sector mf **4개**를 개별 alloc + icnt push (tma_unit_sm.cc#L640-L731)
  - 의도: L2가 `breakdown_request_to_sector_requests`에서 재분해 못 하게 (TMA_ARCH Phase 3)
  - 부작용: L2 입력 큐 mf 개수 4배, 같은 line in-flight 중복 미제거
- L2 도착 후 MSHR merge는 존재(A:192:96)하나, admission 단(icnt_L2_queue top 1개씩 probe)에서
  RESERVATION_FAIL 재시도가 누적됨

---

## 2. 수정 방안

> **현재 단계 (Phase A — 구현 완료):** descriptor mapping 변경(정확한 base 해소)은
> 변경량이 커서 보류하고, 먼저 **mock base addr + 128B 발사(merge)**로 성능 향상이
> 실제로 나오는지 검증한다. 정확한 base 복원(Phase B)은 효과 확인 후 진행.

### 수정 1 (구현 완료): mock base addr — config_id 기반 고정 base

**결정: 정확한 GMEM base 대신 `config_id` 기반 mock base 사용.**
이유: NVBit이 TMA descriptor cache를 못 읽어 진짜 base 복원은 trace-gen 대수술이 필요(§1-B).
mock의 목표는 **SM-간 인위적 hotspot 제거 + 같은 텐서 재사용을 L2 hit로 모델링**.

- 헬퍼 `tma_mock_config_base(config_id)` (tma_unit_sm.cc 익명 namespace):
  - `base = kMockRegionBase(1TiB) + (hash(config_id) & 0xffff) * kPerConfigRegion(256MiB)`
  - 같은 config_id ⇒ 모든 SM·모든 transfer가 같은 region 공유 → 재사용 hit
  - **sm_id / transfer_uid 미반영** (HW도 SM이 같은 GMEM 공유; 섞으면 재사용이 깨짐)
- AGU 주소: `addr = tma_mock_config_base(config_id) + agu_index * 128B`
  - 기존 `(transfer_uid<<20)` 합성 경로 **삭제**
- 한계(의도된 mock): trace에 타일 `coords`가 없어(§아래) config 단위로만 분리됨
  → 같은 config의 타일들이 연속 주소를 공유 → L2 hit이 HW보다 **과대**할 수 있음.
  성능 향상 방향(충돌 제거) 확인엔 충분. 과대하면 coords/cta 분산을 추후 추가.
- 확인 사실: `TMACommand.config_id`/`box_dim`은 채워지나 **`coords`는 채워지지 않음**
  (시뮬레이터 어디서도 set 안 함) → coords 기반 offset 불가

### 수정 2 (구현 완료): 128B mf 1개 발사 (L2 sector breakdown + MSHR merge)

목표: 일반 L1→L2 경로와 동일 포맷(128B mf 1개)으로 발사해 icnt/L2 admission 부담을
sector 4개→1개로 축소하고, L2 MSHR merge로 중복 line을 합친다.

#### 2-1. 128B mf 1개 발사 (구현 완료)
- 변경: AGU 128B request → **128B mf 1개** (data_size=128, sector_mask=full, byte_mask=full)
  - 단위가 32B sector → **128B line**으로 바뀜: `kLineMfGoal = agu_requests * mfs_per_line`
  - reduction(RMW)은 line당 read+write 2개(`mfs_per_line=2`)
  - `memory_sub_partition::push` → `breakdown_request_to_sector_requests`가 32B 자식 4개로
    분해 + MSHR merge (일반 경로와 동일). 부모 128B는 `original_mf`로 보존, 자식은 TMA tag 상속
- **완료/응답 경로 수정 (구현 완료)**:
  - `fill(mf)`: 돌아오는 게 32B 자식이면 `mf->get_original_mf()`로 부모(128B) 조회.
    `m_outstanding_sectors[parent]`로 자식 4개가 모두 돌아왔을 때만 부모 retire
    (L2 비활성 등으로 부모가 직접 오면 remaining=1로 즉시 처리)
  - `mover_on_response`: 완료 카운팅을 **128B line 기준**으로 (`line_mf_goal`)
  - `m_outstanding_requests` 키 = 발사한 128B 부모 mf, `m_outstanding_sectors` = 남은 자식 수

#### 2-2. transfer-간 merge (미구현 — 보류)
- 현재는 **L2의 MSHR merge에 의존**한다. 같은 config_id면 같은 mock base를 공유하므로
  여러 transfer/SM이 같은 128B line을 요청 → L2에서 자연히 merge/hit.
- 명시적 transfer-간 piggyback 테이블은 효과 확인 후 필요 시 추가(아래 설계 보존).

영향 파일 (실제 변경):
- `tma_unit_sm.cc`: `tma_mock_config_base` 헬퍼, 128B 발사 루프, `fill` 부모/자식 매칭,
  `mover_on_response` 128B-line 카운팅
- `tma_unit_sm.h`: `m_outstanding_sectors` 멤버 추가

---

### (Phase B, 보류) 정확한 base 복원 설계 — 참고용 보존

> 효과 확인 후 진행. 아래는 원 설계.

**base 해소 후보 (trace-gen 단계, 우선순위 순)**
1. **encode 호출 순서 ↔ launch별 site 매칭**: dump_id 순서와 UTMALDG site/desc_reg 순서를
   launch window 단위로 1:1 정렬. 검증: dump 수 == launch의 distinct descriptor site 수
2. **desc_reg_id / tensor_map_ptr 기반 매칭**: encode dump `tensor_map_ptr_hex` ↔ producer chain
3. **결정 불가 시 assert**: 합성 fallback 대신 trace-gen SystemExit (`--fail-on-missing-binding`)

**[A]** `build_tma_descriptor_mapping.py`: site별 base 해소 + resolver JSON에
`(ufid,pc,desc_reg)→global_base_address` 기록 + 실패 시 assert
**[B]** `gpu-sim.cc`/`tma_types.h`: base 로드 + 누락 시 assert (`TMACommand.global_base_address`)
**[C]** `tma_unit_sm.cc` AGU: `agu_base = global_base_address + tile_offset`

**transfer-간 merge 테이블 (Phase B 동반 가능)**: key=128B line 주소, value=대표 부모 mf +
piggyback transfer_uid 목록. 발사 전 조회해 in-flight면 piggyback, 완료 시 일괄 credit.
store/reduce(RMW)는 merge 제외.

---

## 3. 검증 계획 (한 번에 재실행)

1. 빌드 후 동일 커널 재실행 (trace config 유지, 단 trace 로그는 stats만 비교 시 축소 가능)
2. 핵심 지표 비교:
   | 지표 | 현재 | 목표(HW 기준) |
   |---|---|---|
   | gpu_sim_cycle | 220,024 | → 67,696 방향 |
   | inst_barrier stall % | 56.09% | ↓ (HW barrier ~10.9%) |
   | TMA lat_mem 평균 | ~3,000cyc | → 수백 cyc |
   | L2 RESERVATION_FAIL 재프로브 | 132회/sector | ↓ |
   | L2 hit rate | (미측정) | NCU 69.58%에 근접 |
3. 동일 sector(예시) 재프로브 횟수와 lat_mem 분포를 trace로 재확인
4. L2 hit rate가 HW(69.58%)와 크게 어긋나면 base 해소 규칙(encode 순서/desc_reg 매칭) 재검토

---

## 4. 상태 / 남은 리스크
- [완료] Phase A = **mock base addr (config_id 기반) + 128B mf 1개 발사**
  - 빌드는 사용자가 직접(현 환경에 CUDA toolkit 없음). 재실행은 ~12시간 소요 예상
- [보류] Phase B = 정확한 base 복원 + 명시적 transfer-간 merge (효과 확인 후)
- [리스크] mock base가 config 단위로만 분리 → **L2 hit 과대** 가능 (coords 부재).
  과대하면 coords/cta 분산 추가로 완화.
- [리스크] 128B 전환 시 fill/완료 회계(자식 4개↔부모 1개) — `m_outstanding_sectors`로 처리.
  reduce/store(RMW) 경로는 line당 2 mf(read+write)로 회귀 주의.
- [확인필요] 12시간 장기 실행이므로 **stderr 로깅으로 중간 동작이 의도대로인지** 조기 검증
  (아래 §5)

---

## 5. 검증용 로깅 (TMADBG, stderr)

12시간 실행 중 mock base / 128B 발사 / merge가 의도대로 됐는지 **조기에** 확인하기 위한
전용 로깅. `-sync_debug_enable`(SYNCDBG/SMDBG, 전체 SM 폭주)과 **분리된 독립 플래그**로,
SYNC 노이즈 없이 TMA 이벤트만 stderr `[TMADBG][SM<n>]`로 출력.

### 켜는 법
```
-tma_debug_enable 1            # 기본 0 (off)
-tma_debug_print_budget 20000000   # SM당 라인 상한(폭주 방지)
```
(config 파일 SM90_H100_L2_50MB_80GB/gpgpusim.config L334-344에 추가됨)

### 나오는 이벤트
- `first-request`  : transfer 발사 시 1회. `config=`, `mock_base=`, `sid=`, `addr=`,
  `mfs_per_line=`, `line_mfs=` → **같은 config가 여러 SM/transfer에서 같은 mock_base를
  쓰는지**(재사용) 확인
- `line-reuse`     : 같은 128B mock line이 2회+ 요청될 때. `times=` → **L2 merge/hit 대상**
  이 실제로 생기는지 확인
- `fill-retire`    : 128B 부모가 32B 자식 4개로 쪼개졌다 합쳐져 retire될 때 1회.
  `as_child=`(자식경유 여부), `sectors=4`, `latency=` → **128B→4×32B split+merge 동작** 확인
- `complete`       : transfer 완료 시. `lat_total/lat_queue/lat_issue/lat_mem` →
  **lat_mem 개선**(핵심 지표) 확인
- `icnt-backpressure` / `store-write-issue` / `reduce-read-issue` : 보조

### 구현 위치
- 게이트 분리: `shader.h`(tma_debug_enable/budget), `gpu-sim.cc`(옵션 register),
  `sm.h`(`m_tma_debug_print_budget`), `sm.cc`(`SM::debug_log_tma_event` stderr를
  새 budget으로 게이트)
- 이벤트: `tma_unit_sm.cc`(first-request/line-reuse/fill-retire/complete 등)
- DTRACE(TMA) stdout 경로는 그대로 유지(`-trace_components TMA`, core-0)
