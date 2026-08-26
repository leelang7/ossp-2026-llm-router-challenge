<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# 시연영상

`routerx_demo.mp4` — 2분 25초, 1920x1080, 2.3MB, 한국어 음성 해설 포함.

유튜브에 업로드한 뒤 URL을 결과보고서의 `시연영상` 칸에 적는다. 규정상 영상 파일을
직접 제출할 수는 없다.

## 구성

| 장면 | 내용 |
| --- | --- |
| 1 | 문제와 제약 (프롬프트만 입력, 호출 불가, 예산 초과 시 0점) |
| 2 | 라우팅 실행 — 등급당 4.2초 |
| 3 | 공식 채점기 검증 — 0.7167, 세 등급 통과 |
| 4 | 특징과 예측 구조 |
| 5 | 순서 불변성 설계 |
| 6 | 자체 점검 10항목 통과 |
| 7 | 예산 안전 — 왜 건수 제한인가 |
| 8 | 교차검증과 두 번의 되돌림 |
| 9 | 정리 |

수치는 모두 실제 실행 결과다. 슬라이드는 개조식으로 쓰고 표와 수치 강조로
구성했다.

## 다시 만들려면

```console
python3 notes/lab/make_video2.py
```

edge-tts로 음성을 만들고 ffmpeg로 합성한다. 한글이 섞인 줄은 맑은 고딕,
순수 영문·숫자는 Consolas로 자동 분기한다(Consolas에는 한글 글리프가 없어
그대로 쓰면 전부 두부로 깨진다).

## 수정하고 싶다면

- 문구: `make_video2.py`의 `s01` ~ `s09` 함수에서 수정
- 해설: 같은 파일 `SCENES` 리스트의 세 번째 항목
- 속도: `rate="+15%"` 조정
- 목소리: `ko-KR-InJoonNeural` (남성) → `ko-KR-SunHiNeural` (여성)
