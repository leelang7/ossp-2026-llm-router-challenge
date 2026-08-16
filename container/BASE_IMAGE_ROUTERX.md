<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# routerx 제출 이미지의 기반 이미지와 의존성

`container/Dockerfile.routerx`가 사용하는 기반 이미지와 런타임 의존성의 출처·라이선스를
기록한다. 과제 규칙(`docs/CHALLENGE_RULES.md`)이 요구하는 항목이다.

## 기반 이미지

| 항목 | 내용 |
| --- | --- |
| 이미지 | `python:3.11-slim-bookworm` |
| 고정 다이제스트 | `sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91` |
| 다이제스트 종류 | 다중 플랫폼 인덱스(`linux/arm64` 포함) |
| 출처 | https://hub.docker.com/_/python |
| 구성 | Debian 12(bookworm) slim + CPython 3.11 |
| 라이선스 | CPython은 PSF-2.0. Debian 기본 구성요소는 GPL/LGPL/MIT/BSD 등 다중이며, 모두 상업적 이용과 재배포가 허용된다. 각 패키지의 사본은 이미지 안 `/usr/share/doc/<패키지>/copyright`에 포함된다. |

과제가 제공하는 참고 Dockerfile은 Alpine 기반이지만, 이 라우터는 NumPy가 필요하고
musl 대상 공식 휠이 없어 Debian slim을 쓴다. 규칙은 기반 이미지를 강제하지 않는다.

### 플랫폼 확인

빌드 전에 인덱스에 `linux/arm64`가 있는지 확인한다. amd64 단일 manifest 다이제스트를
쓰면 `--platform linux/arm64` 빌드가 실패한다.

```console
docker buildx imagetools inspect \
  python:3.11-slim-bookworm@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91
```

## 런타임 의존성

이미지에 설치하는 파이썬 패키지는 하나뿐이다(`container/requirements-runtime.txt`).

| 패키지 | 버전 | 라이선스 | 출처 |
| --- | --- | --- | --- |
| NumPy | 1.26.4 | BSD-3-Clause | https://github.com/numpy/numpy |

NumPy의 manylinux aarch64 휠은 아래 네이티브 라이브러리를 함께 담고 있다. 재배포에
문제가 없으나 고지를 위해 기록한다.

| 번들 구성요소 | 라이선스 | 비고 |
| --- | --- | --- |
| OpenBLAS | BSD-3-Clause | 선형대수 커널 |
| libgfortran, libgcc, libquadmath | GPL-3.0-with-GCC-exception | GCC 런타임 예외 조항에 따라 자유롭게 재배포 가능 |

학습에만 쓰는 scikit-learn·SciPy·PyArrow는 이미지에 넣지 않는다. 실행 중에는 어떤
파일도 내려받지 않으며 네트워크가 없어도 동작한다.

## 이미지에 포함하는 학습 산출물

| 항목 | 내용 |
| --- | --- |
| 파일 | `src/routerx/artifact.npz` |
| 용도 | 품질 예측 회귀 계수, TF-IDF 어휘·IDF, 토큰 예측 부스팅 트리, 등급별 안전계수·상한 |
| 생성 방법 | `train_routerx/train.py`가 과제 제공 공개 Train/Dev 자료로 학습해 생성 |
| 기반 모델 | 없음(사전학습 가중치를 쓰지 않는다) |
| 라이선스 | Apache-2.0 (`REUSE.toml`에 선언) |
| 공개 위치 | 이 저장소에 커밋되어 있으며 별도 승인 없이 접근 가능 |

외부에서 받은 사전, 토크나이저, 언어 모델은 사용하지 않는다.

## 학습 환경(재현용)

실행 이미지에는 들어가지 않지만 아티팩트를 다시 만들려면 필요하다.

| 패키지 | 버전 | 라이선스 |
| --- | --- | --- |
| Python | 3.12 | PSF-2.0 |
| NumPy | 1.26.4 | BSD-3-Clause |
| SciPy | 1.17.1 | BSD-3-Clause |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| PyArrow | 23.0.1 | Apache-2.0 (공개 자료 생성 단계에만 사용) |

`Ridge(solver="sparse_cg")`는 반복 해법이라 BLAS 구현에 따라 마지막 자리가 달라질 수
있다. 제출 이미지는 커밋된 아티팩트를 그대로 쓰므로 이 차이가 평가 결과에 영향을 주지
않는다.
