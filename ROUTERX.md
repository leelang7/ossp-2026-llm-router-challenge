<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# routerx — 프롬프트 내용만으로 모델을 고르는 라우터

SK텔레콤 지정과제 *Efficient LLM Routing Challenge* 제출 구현입니다.
프롬프트(또는 대화 메시지)와 예산 등급만 보고 `ax31-light` / `ax31` / `axk1-think`
중 하나를 고릅니다. 모델을 호출하지 않고, 네트워크도 쓰지 않습니다.

## 빠르게 실행하기

```console
# 1) 공개 Train/Dev 자료 생성 (최초 1회)
python3 -m venv .venv-data
.venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
.venv-data/bin/python tools/materialize_public_data.py

# 2) 학습 — 아티팩트 생성
PYTHONPATH=src python3 train_routerx/train.py \
  --train-input data/materialized/train/inputs.json \
  --train-outcomes data/train/outcomes.json \
  --dev-input data/materialized/dev/inputs.json \
  --dev-outcomes data/dev/outcomes.json \
  --artifact src/routerx/artifact.npz \
  --fit-on train+dev \
  --tier-margin fast=0.83 --tier-margin balanced=0.80 --tier-margin premium=0.85 \
  --tier-k1-cap fast=0.0 --tier-k1-cap balanced=0.01 --tier-k1-cap premium=0.11

# 3) 라우팅 — 등급 하나당 제출 파일 하나
PYTHONPATH=src python3 -m routerx.cli \
  --input data/materialized/dev/inputs.json --tier fast --output build/fast.json

# 4) 자체 점검 — 규칙 준수·성능·예산 여유를 한 번에
PYTHONPATH=src python3 -m routerx.audit \
  --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json

# 5) 컨테이너 (제출 형식)
docker build --platform linux/arm64 -f container/Dockerfile.routerx -t routerx:arm64 .
docker run --rm --platform linux/arm64 --network none --read-only \
  --user 65532:65532 --cpus 2 --memory 2g --pids-limit 32 --tmpfs /tmp:size=256m \
  -v "$PWD/data/materialized/dev:/challenge/input:ro" -v "$PWD/build/out:/challenge/output" \
  routerx:arm64 --input /challenge/input/inputs.json --tier premium \
  --output /challenge/output/submission.json
```

## 동작 방식

### 1. 예측 — 무엇을 맞히려 하는가

프롬프트에서 세 종류의 특징을 뽑아 ridge 회귀 7개 헤드에 넣습니다.

| 특징 | 내용 |
| --- | --- |
| TF-IDF (단어) | 1–2그램, 최대 6만 차원 |
| TF-IDF (문자) | `char_wb` 3–5그램, 최대 12만 차원 |
| 직접 계산 특징 36개 | 길이·한글 비율·코드/수식 표지·객관식 표지·대화 구조 등 |

예측 대상은 모델별 품질 3개, 모델별 출력 토큰 3개, 입력 토큰 1개입니다.
**토큰 수를 예측 대상에 넣은 이유**는 라우팅 시점에 실제 토큰 수가 주어지지 않기
때문입니다. 학습에서도 실제값 대신 예측값으로 비용을 계산해 추론과 조건을 맞췄습니다.

비용은 공개 비용 정책으로 계산하되 두 단계로 보정합니다. 로그 공간에서 회귀한 값을
지수로 되돌리면 평균이 과소평가되므로(Jensen), 잔차 분위수만큼 상향한 뒤 학습 분할의
실제 총비용과 맞도록 모델별 스케일을 곱합니다.

### 2. 정책 — 어떻게 고르는가

모든 문항을 경량 모델에서 시작해, 예측 이득 대비 예측 비용이 큰 순서로 승격합니다.
동률은 **프롬프트 내용 해시**로만 깹니다. 문항 ID·입력 순서를 쓰지 않으므로 입력을
섞어도 결과가 같습니다(운영자 감사 항목).

여기에 **추론 모델 선택 건수 상한**을 겁니다. `axk1-think`는 출력 토큰 분포의 꼬리가
두꺼워 한 문항이 경량 총비용의 26%까지 쓸 수 있고, 몇 건만 빗나가도 등급 전체가
0점이 됩니다. 예측 비용에 상한을 두거나 분위수 회귀로 위험을 추정하는 방식은
"예측은 작은데 실제가 큰" 문항을 걸러내지 못해 효과가 없었지만, 선택 건수를 직접
묶으면 예측 오차와 무관하게 노출이 제한됩니다. 고비용 구간은 승격 이득 대비 비용도
가장 나쁜 구간이라 점수 손실도 작습니다.

### 3. 등급별 설정

2,640문항 교차검증(5-fold·8-fold)으로 정했습니다. 양쪽 fold 구성에서 예산 초과 0건입니다.

| 등급 | 추론모델 상한 | 예산 마진 | 공개 Dev 예산 사용률 |
| --- | ---: | ---: | ---: |
| fast | 0% | 0.83 | 한도의 83.4% |
| balanced | 1% | 0.80 | 한도의 83.1% |
| premium | 11% | 0.85 | 한도의 69.6% |

## 성능

공개 Dev 880문항, Train만 학습한 정직한 조건입니다.

| 항목 | 값 |
| --- | --- |
| 최종 점수 | 0.672869 |
| fast | 0.62897 (예산 83.4%) |
| balanced | 0.68806 (예산 83.1%) |
| premium | 0.71619 (예산 69.6%) |
| 실행 시간 | 등급당 4–5초(880문항, 약 5ms/문항) — 한도 90초의 6% |
| 이미지 크기 | 290MB |
| 아티팩트 | 5.5MB |

## 설계에서 내린 판단

- **입력 토큰도 예측 대상에 포함**: 라우팅 시점에 주어지지 않으므로 학습·추론 조건을 맞췄습니다.
- **비용 예측을 보수적으로**: 로그-지수 변환 편향과 총액 스케일을 함께 보정합니다.
- **추론 모델은 건수로 제한**: 예측 기반 필터는 실패했고 건수 제한만 효과가 있었습니다.
- **예산은 남기고 씁니다**: 초과하면 그 등급이 0점이므로, 한도까지 채우는 정책보다
  여유를 두는 정책이 기대 점수가 높습니다.
- **런타임 의존성은 NumPy 하나**: 학습에만 scikit-learn을 쓰고, 컨테이너에서는
  scikit-learn `TfidfVectorizer`의 동작을 표준 라이브러리로 재현합니다.
  두 경로의 예측 차이는 1.2e-15(부동소수점 한계)입니다.

## 구성

```
src/routerx/
├── features.py   프롬프트 전용 특징과 TF-IDF 재현
├── policy.py     예산·상한을 반영한 배치 선택
├── router.py     아티팩트 로드와 벡터화 추론
├── cli.py        router-run 인터페이스
├── audit.py      제출 전 자체 점검
└── artifact.npz  전역 계수·어휘·IDF·등급별 설정
train_routerx/train.py   학습과 아티팩트 생성
container/               linux/arm64 이미지 정의
```

## 라이선스

코드는 Apache-2.0입니다. 런타임 의존성은 NumPy(BSD-3-Clause) 하나이며,
학습에는 scikit-learn·SciPy(BSD-3-Clause)를 씁니다. 상세 목록은 결과보고서
붙임1(SBOM)에 있습니다.
