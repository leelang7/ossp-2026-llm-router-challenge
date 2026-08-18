<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# 작업 노트

routerx 라우터를 만들면서 남긴 진행 기록, 실험 스크립트, 제출 준비 문서 모음이다.
구현 자체의 설명은 저장소 루트의 [`ROUTERX.md`](../ROUTERX.md)에 있다.

## 무엇부터 보면 되나

| 문서 | 내용 |
| --- | --- |
| [STATUS.md](STATUS.md) | **여기부터.** 현재 성적, 확정 정책, 검증 상태, 잡은 결함, 채택하지 않은 시도, 남은 작업 |
| [SUBMIT_CHECKLIST.md](SUBMIT_CHECKLIST.md) | 최종 제출 절차. 순서가 중요하다(이미지→커밋→JSON→보고서) |
| [REPORT_DRAFT.md](REPORT_DRAFT.md) | 결과보고서 원고 (본문 + SBOM + AI 모델 명세) |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | 시연영상 3분 대본. 장면별 멘트와 촬영 준비 목록 |
| [DESIGN.md](DESIGN.md) | 초기 설계와 과제 스펙 분석 |
| `2026 오픈소스 개발자대회 결과보고서_접수번호(팀명).docx` | 공식 양식에 채운 보고서. 팀명·커밋 SHA·영상 URL만 비어 있다 |

## 지금 상태 요약

- 공개 Dev 880문항 **0.7167**, 교차검증 2,640문항 **0.6643** (예산 초과 0건)
- 자체 점검 10항목 전부 통과, 단위 시험 21건 통과
- 등급당 실행 4.2초 (한도 90초의 5%), 이미지 73.4MB
- 확정 정책: fast K1 0%/마진 0.85, balanced K1 1%/0.80, premium K1 11%/0.85,
  문항별 비용 상한 5%

## 남은 일

1. **팀명** — 상장에 인쇄된다. 정해지면 보고서 자리표시자를 채운다.
2. **시연영상** — 3분 촬영·유튜브 업로드 (대본은 DEMO_SCRIPT.md)
3. **레지스트리 push** — `write:packages` 권한 인증 필요. 이미지 다이제스트 확보
4. **PDF 변환** — 규정이 원본 + PDF 두 부를 요구한다
5. **osscontest.kr 업로드** — 마감 2026-08-27(목) 18:00

## lab/ — 실험 스크립트

결론에 이르기까지 실제로 돌린 코드다. 재현하려면 `d:\opensource\skt-router\lab`
기준 절대경로를 자기 환경에 맞게 고쳐야 한다.

| 스크립트 | 무엇을 확인했나 |
| --- | --- |
| `cv.py`, `cv_margin.py` | 교차검증 엔진. 마진만으로는 예산 초과를 못 막는다는 것 |
| `cv_cap.py`, `item_cap.py` | 문항별 비용 상한. 예측 기반 필터가 왜 실패하는지 |
| `k1_limit.py`, `k1_fine.py`, `k1_cost.py` | 추론 모델 사용량 제한. 건수 상한이 유효한 이유 |
| `ceiling.py` | 정보 수준별 도달 가능 상한 (0.605 / 0.683 / 0.725 / 0.792) |
| `embed_test.py` | 다국어 임베딩이 오히려 예측을 떨어뜨린다는 측정 |
| `ridge_bag.py` | 계수 앙상블 효과가 +0.001에 그친다는 측정 |
| `compare_cost.py` | ridge vs 부스팅 트리 비용 예측 최종 비교 |
| `fill_report.py` | 결과보고서 DOCX 자동 생성 |
| `exp1.py` ~ `exp9.py` | 초기 탐색 (예측기 후보, 병목 판별, 캘리브레이션) |

`*_result.txt`는 각 실험의 출력 원본이다.
