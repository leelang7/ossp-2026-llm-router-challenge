<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# SBOM (Software Bill of Materials)

2026 오픈소스 개발자대회 결과보고서 붙임1과 같은 내용이다. 심사 과정에서 별도
문의 없이 확인할 수 있도록 저장소 최상위에 함께 둔다.

수록 기준은 대회 「[부록1] SBOM 작성 가이드」를 따른다. 팀이 직접 작성한 코드는
적지 않으며, 바깥에서 가져다 쓴 것만 우선순위 순으로 적는다.

- ① GPL·AGPL·LGPL 계열 라이선스
- ② 핵심 기능 담당 — 빠지면 프로그램이 돌지 않는 것
- ③ 주요 프레임워크·SDK
- ④ 핵심 빌드·실행 도구

데이터셋 출처와 라이선스는 별도 문서에 있다 — [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[DATA_LICENSES.md](DATA_LICENSES.md). AI 모델 관련 사항은 결과보고서 붙임2에 적는다.

| 번호 | 라이브러리명 | 버전 | 라이선스 | 공식 저장소 | 사용 목적 및 결합 방식 |
|---|---|---|---|---|---|
| 1 | libgfortran | 5.0.0 | GPL-3.0-with-GCC-exception | https://github.com/gcc-mirror/gcc | Fortran 런타임. NumPy 휠에 번들되어 동적 링크로 적재 |
| 2 | NumPy | 1.26.4 | BSD-3-Clause | https://github.com/numpy/numpy | 실행 의존성. 계수 행렬 연산 및 배치 선택 계산 / 라이브러리로 불러 씀 |
| 3 | OpenBLAS | 0.3.23.dev | BSD-3-Clause | https://github.com/OpenMathLib/OpenBLAS | 행렬 연산 가속. NumPy 휠에 번들되어 동적 링크로 적재 |
| 4 | CPython | 3.11.16 | PSF-2.0 | https://github.com/python/cpython | 실행 런타임. 컨테이너 기반 이미지 `python:3.11-slim-bookworm` |
| 5 | scikit-learn | 1.8.0 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn | 학습 전용. TF-IDF 어휘·IDF 생성, ridge 회귀 학습 / 라이브러리로 불러 씀 |
| 6 | SciPy | 1.17.1 | BSD-3-Clause | https://github.com/scipy/scipy | 학습 전용. 희소 행렬 연산 / 라이브러리로 불러 씀 |
| 7 | PyArrow | 23.0.1 | Apache-2.0 | https://github.com/apache/arrow | 자료 준비 전용. 공개 Train/Dev 원본 변환 / 라이브러리로 불러 씀 |
| 8 | ossp_router (과제 제공) | v1 | Apache-2.0 | https://github.com/sktelecom/ossp-2026-llm-router-challenge | 입출력 규격 및 채점 도구 / 라이브러리로 불러 씀 |

## 실행 이미지에 실제로 들어가는 것

제출 이미지의 런타임 의존성은 NumPy 하나뿐이다. scikit-learn·SciPy·PyArrow는
학습과 자료 준비에만 쓰고 이미지에 넣지 않는다. libgfortran과 OpenBLAS는 NumPy
휠에 이미 번들되어 함께 들어간다.

```
ghcr.io/leelang7/routerx@sha256:ea01be4aa373f1358450c56105f4f595619b7fa2bd272d418c9bc71f8b75016f
```

버전은 위 이미지에서 실제로 적재된 값을 확인해 적었다.

```sh
docker run --rm --platform linux/arm64 --entrypoint python <image> \
  -c "import numpy; print(numpy.__config__.CONFIG['Build Dependencies']['blas'])"
```

## 라이선스 전문

각 라이선스 전문은 [LICENSES/](LICENSES/) 아래에 둔다. 본 프로젝트가 직접 작성한
코드는 [Apache-2.0](LICENSE)을 적용한다.
