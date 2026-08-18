<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# 프롬프트 난이도 인지 기반 경량 LLM 라우터 — 설계서 v2

> 2026 오픈소스 개발자대회 · SK텔레콤 지정과제 "Efficient LLM Routing Challenge"
> v2 (2026-07-16): 공식 저장소 공개로 스펙 확정 — **cascade/승급 금지, 단일 선택 방식**으로 전면 개정
> 공식 저장소: https://github.com/sktelecom/ossp-2026-llm-router-challenge (로컬 클론: d:\opensource\ossp-2026-llm-router-challenge)

## 1. 확정 스펙 (v1 설계와 달라진 점)

| 항목 | v1 가정 | **v2 확정** |
|---|---|---|
| 결정 방식 | 순차 호출·승급(cascade) 가능 | **문항당 모델 1개 단일 선택. 호출·비교·승급 금지** |
| 라우터 입력 | 프롬프트+tier+호출 이력+메타 | **프롬프트(또는 messages)+tier만.** 문항ID·출처·순서 사용 금지 |
| 후보 모델 | 미공개 | **ax31-light / ax31 / axk1-think** 3종 고정 |
| 예산 초과 | 감점 추정 | **해당 tier 0점 (하드 제약)** |
| 데이터 | 미공개 | Train 1,760 / Dev 880. 모델별 score·num_generations·input/output tokens |
| 제출 | GitHub 게시 | **공식 repo fork + linux/arm64 컨테이너 + submission-ossp-skt.json + 고정 커밋 스냅샷 URL** |

- Verifier·cascade(§v1-2C, 2E 일부)는 **폐기**. 관측 후 판단이 불가능하므로 순수 pre-hoc 예측 문제.
- 해시·정규식·n-gram·**임베딩** 등 프롬프트 내용에서 계산한 정보는 명시적으로 허용.

## 2. 채점 공식 (docs/SCORING.md 확정)

```
episode_cost = input_tokens·rate_in + output_tokens·rate_out   (per 1M tokens)
  ax31-light: 1 / 4      ax31: 2.127 / 8.509      axk1-think: 6.565 / 26.260
budget_limit = (전체 문항을 ax31-light로 선택한 비용) × multiplier
  fast 1.25 / balanced 2.0 / premium 4.0   — 초과 시 해당 tier 0점
tier_score  = 선택 모델 score 평균 (예산 통과 시)
final_score = 0.4·fast + 0.3·balanced + 0.3·premium
동점 시: arm64 컨테이너 실행 시간(시작~제출 생성) 중앙값 합산이 낮은 쪽 우선
```

핵심 함의:
1. **비용의 지배 변수는 output_tokens**(입력 대비 단가 4배). axk1-think는 추론(think) 모델이라
   출력 토큰 분산이 커서, **모델별 출력 토큰 수 예측기가 사실상 비용 예측기**다.
2. 예산은 하드 제약 + 평가셋 분포 비공개 → 공식 hash-regex 베이스라인조차 Dev에서 3.985로
   통과했지만 **비공개 평가에서 4.2로 초과해 Premium 0점**을 맞았다(주최 측 공개 사례).
   → 안전 마진은 성능 튜닝이 아니라 생존 조건. 목표 소진율 ≤ 한도의 85~90%.
3. Fast(가중치 0.4, 한도 1.25)가 최대 배점 — all-light 대비 여유가 25%뿐이므로
   "확실히 어려운 소수 문항만 정확히 승격"하는 정밀도가 승부처.

## 3. 이겨야 할 스코어보드 (공개 Dev 880, 주최 측 제공)

| Baseline | 방식 | 최종 점수 |
|---|---|---|
| all-light | 전부 경량 모델 | 0.6193 |
| feature-budget | 규칙 피처 + 예산 배분 | 0.6430 |
| prompt-heuristic | 규칙 피처 임계값 | 0.6553 |
| **hash-regex** | **해싱 n-gram + ridge 6헤드 학습형** | **0.6954** |

→ 우리 목표: 임베딩 기반 예측기로 hash-regex를 유의미하게 상회 + 3 tier 전부 예산 통과.
   상한 참고: 문항별 최고 score 모델을 항상 고르는 oracle 점수를 Dev에서 산출해 gap 측정(1주차).

## 4. 우리 라우터 아키텍처

```
프롬프트 ─┬─ [A] 표면 피처 (길이·한글비율·코드·수식·메시지 구조 …)
          ├─ [B] 해싱 n-gram 피처 (공식 baseline 계승·확장)
          └─ [C] 경량 다국어 임베딩 (증류·int8 ONNX, 팀 강점)
                    ↓ (스태킹)
   [D] 예측 헤드 6+3개 (k-fold OOF 앙상블: ridge/GBM)
       · score_m(x)  — 모델별 기대 품질 (3헤드)
       · logtok_m(x) — 모델별 출력 토큰 (3헤드) → 비용 추정 ĉ_m(x)
       · 불확실도    — 비용 상위 분위수(안전 마진용)
                    ↓
   [E] Tier 정책 (전역 배분 = 제약 있는 배낭 문제)
       · 이득/비용비 Δq̂/Δĉ 기준 전역 그리디 승격 (λ 임계값과 동치)
       · 비용은 상위 분위수로 보수 집계, 목표 소진율 85~90%
       · tier별 안전계수는 소스 단위 leave-one-out 스트레스로 결정
       · fast: light↔ax31 위주, premium: K1 승격은 출력토큰 예측 신뢰 구간 내에서만
```

- 실행 규칙 준수: 입력은 프롬프트 내용+tier만, 문항 간 독립·결정적 선택(공식 테스트 통과 필수).
  전역 배분은 **학습 시점에 임계값(λ, 안전계수)으로 고정**해 실행 시엔 문항별 독립 적용.
- 런타임: 표준 라이브러리 + (필요시) int8 ONNX 단일 파일. 컨테이너 시작 포함 실행 시간이
  tie-break이므로 무거운 프레임워크는 학습 단계에만 사용.

## 5. SOTA 및 자산 활용

- **RouteLLM(LMSYS)**: BERT/행렬분해 라우터 — 임베딩+경량 헤드 구조의 근거. 단일 선택 스펙과 정확히 동형.
- **Zooter / RouterDC / EmbedLLM**: 보상 증류·대조학습 기반 모델별 적합도 예측 — [D] 헤드 학습 개선 옵션.
- **RouterBench**: 비용-품질 트레이드오프 평가 프레임 — 오프라인 평가 지표 설계 참고.
- **Inference Scaling Laws**(공식 참고자료): num_generations 해석·compute-optimal 관점.
- **팀 자산**: KoBERT/한국어 분류기 학습 노하우(scam-models), ONNX int8 경량화·온디바이스(AllThatFinder),
  모델 단가·비용 감각(nodaero). ditto 등 기존 LLM 라우터 연구 자산은 위치 확인 후 통합.

## 6. 제출 파이프라인 (docs/SUBMISSION.md)

1. 공식 repo를 팀 GitHub 계정으로 fork → `src/ossp_router/heuristic.py` 교체가 최단 경로
2. `tools/materialize_public_data.py`로 Train/Dev 입력 생성(AIME 원문 별도 수급)
3. `self-check` CLI로 형식·예산·점수 검증 (`near_budget` 95% 경고 확인)
4. linux/arm64 이미지 빌드(Windows: Docker buildx+QEMU) → 공개 레지스트리 push
5. `submission-ossp-skt.json` 커밋 → 고정 스냅샷 URL을 결과보고서에 기재
6. 전 코드 OSI 라이선스(공식 repo가 Apache-2.0), 가중치·아티팩트 공개, 5년 유지

## 7. 일정 (오늘 8/11 기준, 제출 8/27 목 18:00 — 16일)

| 구간 | 기간 | 목표 |
|---|---|---|
| D0 | 8/11 | 하네스 검증(toy self-check ✅), 데이터 materialize, fork(leelang7) |
| D1 | 8/12–8/13 | EDA(소스·score·토큰 분포), oracle gap, hash-regex 로컬 재현 |
| D2 | 8/14–8/17 | 임베딩/피처 + OOF 헤드 학습 → Dev에서 hash-regex(0.6954) 초과 |
| D3 | 8/18–8/20 | 출력 토큰 예측·비용 분위수, tier 배분 최적화, 안전계수 스트레스 |
| D4 | 8/21–8/22 | 경량화(int8/증류 또는 순수 파이썬 변환), arm64 컨테이너 빌드·90초 실측 |
| D5 | 8/23–8/24 | 강건성 최종화, 마진 확정, check_runtime 통과, 코드 정리 |
| D6 | 8/25–8/27 | 프리즈. 결과보고서 5p·SBOM·붙임2, 시연영상, 이미지 push·스냅샷 커밋 |

런타임 제약(RUNTIME.md): linux/arm64, CPU 2코어·메모리 2GiB·GPU 없음·네트워크 없음,
tier당 90초(전체 배치), 이미지 압축 1GiB/루트 2GiB, 진입점 `router-run --input --tier --output`.
→ 추론 모델은 초소형이어야 함(문항당 수십 ms 예산). 학습은 로컬 RTX 4070, 배포는 증류·경량 변환.

## 8. 리스크

1. **비공개 평가 분포 이동으로 예산 초과 → 0점** (최대 리스크): 분위수 비용 + 85~90% 소진 목표
   + 소스 leave-one-out 스트레스로 대응. "점수 조금 손해 < 0점"이 항상 우선.
2. Train 1,760개 소표본 과적합: OOF 앙상블, Dev는 안전계수 보정 전용(공식 베이스라인 방식 준수).
3. arm64 빌드 환경: Windows Docker Desktop + buildx 사전 검증(4주차 이전).
4. AIME 원문 수급 실패: materialize 절차 조기 실행으로 1주차에 확인.
