<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# routerx — 프롬프트 난이도 인지 기반 경량 LLM 라우터

**2026 오픈소스 개발자대회 · SK텔레콤 지정과제 [Efficient LLM Routing Challenge](CHALLENGE.md) · 팀 트리아지**

질문 하나하나의 난이도를 프롬프트만 보고 가늠해, 세 후보 모델 중 가장 알맞은
하나에 배정합니다. 모델을 호출하지 않고 네트워크도 쓰지 않습니다. 학습해 둔
계수와 어휘만으로 문항당 약 5밀리초에 판단합니다.

**시연영상 (1분 40초)** — https://youtu.be/ejKfuvM4vdg

## 무엇을 했나

같은 예산으로 더 나은 답을 얻는 것이 목표입니다. 쉬운 질문에 큰 모델을 쓰면
예산만 쓰고 품질은 그대로이고, 어려운 질문에 작은 모델을 쓰면 답이 나빠집니다.
그 경계를 프롬프트 텍스트만으로 그었습니다.

| | 점수 | 비고 |
|---|---|---|
| **본 구현 — 공개 Dev 880문항** | **0.7167** | 세 등급 모두 예산 내 |
| 본 구현 — 교차검증 2,640문항 | 0.6643 | 5-fold·8-fold 양쪽 예산 초과 0건 |
| 공식 최고 baseline | 0.6954 | |
| 전량 경량 모델 | 0.6193 | |

점수는 `0.4·fast + 0.3·balanced + 0.3·premium`이며, 어느 등급이든 예산을 넘기면
그 등급은 0점입니다. 그래서 **점수를 올리는 일보다 0점을 만들지 않는 일이 어려웠습니다.**

## 어떻게 동작하나

```
프롬프트  ─→  특징 추출  ─→  ridge 7개 헤드  ─→  배치 선택  ─→  model_id
              직접계산 36종      품질 3 · 출력토큰 3        이득/비용 비로 승격
              단어 TF-IDF 6만    입력토큰 1               예산·건수·항목 상한
              문자 TF-IDF 12만
```

- **[`src/routerx/features.py`](src/routerx/features.py)** — 프롬프트 내용만 사용합니다.
  문항 ID나 입력 순서는 특징에 넣지 않습니다.
- **[`src/routerx/router.py`](src/routerx/router.py)** — 학습은 scikit-learn으로 하고,
  실행 이미지에는 NumPy만 넣습니다. `TfidfVectorizer` 동작을 표준 라이브러리로
  재현했고, 두 경로의 예측 차이는 **1.2e-15** 입니다.
- **[`src/routerx/policy.py`](src/routerx/policy.py)** — 예산은 배치 전체 제약이라
  문항을 따로 처리할 수 없습니다. 이득/비용 비로 정렬하되 **동률은 프롬프트
  SHA-256 해시로만** 판정해, 입력 순서를 바꿔도 결과가 같습니다.
- **[`src/routerx/audit.py`](src/routerx/audit.py)** — 제출 전 자체 점검 10항목.

## 설계에서 가장 오래 붙든 문제

**예산 초과를 예측 정확도로 막을 수 없었습니다.** 추론 모델의 출력 토큰은
중앙값 1,570에 최댓값 130,504로, 단일 문항 하나가 경량 모델 전체 비용의 26%를
차지할 수 있습니다. 예측 비용에 상한을 두는 방식은 통하지 않았습니다 — 사고를
일으키는 문항이 하필 **예측은 적게 나오고 실제는 크게 나오는** 유형이라 필터를
그대로 통과했기 때문입니다.

그래서 예측을 믿는 대신 **노출 규모 자체를 제한**했습니다. 추론 모델 선택 건수에
상한을 두면 예측이 틀려도 손실이 한정됩니다. 다행히 고비용 구간은 승격 이득
대비 비용도 최저여서 점수 손실이 작았습니다. 등급별 상한값은 2,640문항
5-fold·8-fold 교차검증에서 **양쪽 모두 예산 초과 0건**인 조합만 채택했습니다.

## 실행

```console
# 1) 공개 Train/Dev 자료 생성 (최초 1회)
python3 -m venv .venv-data
.venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
.venv-data/bin/python tools/materialize_public_data.py

# 2) 라우팅
PYTHONPATH=src python -m routerx.cli \
  --input data/materialized/dev/inputs.json --tier premium --output out.json

# 3) 자체 점검 — 순서 불변·결정성·엣지 케이스·실행 시간·예산 여유
PYTHONPATH=src python -m routerx.audit \
  --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json
```

제출 이미지를 공식 자원 한도로 그대로 돌려볼 수도 있습니다.

```console
docker run --rm --platform linux/arm64 --network none --read-only \
  --user 65532:65532 --cpus 2 --memory 2g --tmpfs /tmp:size=256m \
  -v "$PWD/data/materialized/dev:/challenge/input:ro" -v "$PWD/out:/challenge/output" \
  ghcr.io/leelang7/routerx@sha256:ea01be4aa373f1358450c56105f4f595619b7fa2bd272d418c9bc71f8b75016f \
  --input /challenge/input/inputs.json --tier fast --output /challenge/output/submission.json
```

## 자체 점검 결과 (공개 Dev 880문항)

| 항목 | 결과 |
|---|---|
| 문항 ID·입력 순서 불변성 | 400문항 불일치 0건 |
| 결정성 (같은 입력 → 같은 출력) | 300문항 불일치 0건 |
| 엣지 케이스 7종 | 통과 (빈 프롬프트, 초장문, 어휘 외 문자, 이모지, 장문 대화 등) |
| 실행 시간 | 등급당 4.2초 (한도 90초의 5%) |
| 예산 | 세 등급 전량 통과, 여유 17~35% |
| 단위 시험 | 21건 통과 |

## 문서

| | |
|---|---|
| [ROUTERX.md](ROUTERX.md) | 구현 상세 — 특징, 학습, 정책, 재현 절차 |
| [SBOM.md](SBOM.md) | 소프트웨어 자재명세서 (결과보고서 붙임1과 동일) |
| [notes/](notes/) | 개발 기록 — 설계 판단, 실험 결과, 제출 점검표 |
| [CHALLENGE.md](CHALLENGE.md) | SK텔레콤 지정과제 원문 안내 |
| [docs/SCORING.md](docs/SCORING.md) | 채점 규칙 |

## 라이선스

직접 작성한 코드는 [Apache-2.0](LICENSE)입니다. 데이터셋 출처와 라이선스는
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [DATA_LICENSES.md](DATA_LICENSES.md)에
있습니다. 학습된 가중치([`src/routerx/artifact.npz`](src/routerx/artifact.npz), 5.5MB,
1.44M 파라미터)도 저장소에 함께 공개합니다.
