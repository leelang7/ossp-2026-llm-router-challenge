<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# 시연영상

`routerx_demo.mp4` — 2분 34초, 1920x1080, 2.6MB, 한국어 음성 해설 포함.

유튜브에 업로드한 뒤 URL을 결과보고서의 `시연영상` 칸에 적는다. 규정상 영상 파일을
직접 제출할 수는 없다.

## 구성

| 장면 | 내용 |
| --- | --- |
| 1 | 문제와 제약 (프롬프트만 입력, 호출 불가, 예산 초과 시 0점) |
| 2 | 실제 라우팅 실행 화면 |
| 3 | 공식 채점기 검증 결과 (세 등급 통과, 0.7167) |
| 4 | 특징 추출과 예측 구조 |
| 5 | 순서 불변성 설계 |
| 6 | 자체 점검 도구 10항목 통과 |
| 7 | 예산 안전 — 왜 건수 제한인가 |
| 8 | 교차검증과 두 번의 되돌림 |
| 9 | 정리 |

2·3·6번 장면은 실제로 실행한 출력을 그대로 캡처한 것이다
(`build/video/cap_*.txt`).

## 다시 만들려면

```console
python3 notes/lab/make_video_short.py
```

`make_video.py`가 슬라이드 렌더링과 합성을, `make_video_short.py`가 3분에 맞춘
나레이션을 담당한다. edge-tts로 음성을 만들고 ffmpeg로 합성한다.
캡처 파일을 새로 뜨려면 `routerx.cli`, `ossp_router.cli self-check`,
`routerx.audit`의 출력을 `build/video/cap_route.txt`, `cap_check.txt`,
`cap_audit.txt`로 저장하면 된다.

## 수정하고 싶다면

- 문구: `make_video.py`의 `SCENES`에서 각 장면 `lines` 수정
- 해설: `make_video_short.py`의 `SHORT` 딕셔너리 수정
- 속도: `make_video.py`의 `rate="+18%"` 조정
- 목소리: `ko-KR-InJoonNeural` (남성) → `ko-KR-SunHiNeural` (여성)
