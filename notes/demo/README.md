<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# 시연영상

`routerx_demo.mp4` — 2분 37초 / 1920x1080 / 4.5MB / 한국어 해설·자막 포함.

유튜브에 업로드한 뒤 URL을 결과보고서의 `시연영상` 칸에 적는다.
규정상 영상 파일을 직접 제출할 수는 없다.

## 구성

발표 슬라이드가 아니라 **프로그램이 실제로 동작하는 화면 녹화**가 본편이다.
표지와 결과 카드만 앞뒤에 붙였다.

| 구간 | 내용 |
| --- | --- |
| 0:00 | 표지 |
| 0:05 | 라우팅 실행 — 공개 Dev 880문항, premium 등급 |
| 0:35 | 선택 결과 확인 — 문항 수, 모델 분포 |
| 0:49 | 공식 채점기 검증 — 0.716704, 세 등급 통과 |
| 1:07 | 자체 점검 도구 — 10항목 |
| 1:49 | 제출 이미지 실행 — 공식 자원 한도, 다이제스트 지정 |
| 2:29 | 결과 정리 |

터미널에 보이는 명령과 출력은 전부 실제 실행 결과다. 편집으로 만든 화면이 아니다.

## 다시 만들려면

세 단계다.

```console
# 1) 화면 녹화 — 실행 중 화면을 건드리면 그대로 찍힌다 (약 145초)
powershell -File notes/lab/demo_session.ps1        # 별도 창에서 실행
ffmpeg -f gdigrab -framerate 15 -video_size 2560x1440 -i desktop -t 145 \
       -vf scale=1920:1080 -c:v h264_nvenc build/rec/session.mp4

# 2) 해설 음성과 자막 — Gemini TTS(Charon, 차분한 남성 발표 톤)
python notes/lab/build_narration.py

# 3) 합성 — 녹화 + 음성 + 자막 + 앞뒤 표지
python notes/lab/compose_demo.py
```

`build_narration.py`는 `AuraView/secrets/tts_key*.txt`의 키를 읽는다.
키가 여러 개면 할당량 초과(429) 시 자동으로 다음 키를 쓴다.

## 알아둘 것

- **PowerShell 스크립트는 UTF-8 BOM으로 저장해야 한다.** BOM이 없으면 PowerShell 5.1이
  CP949로 읽어 한글이 깨지고 함수 호출까지 실패한다.
- 자막을 굽을 때 `subtitles` 필터에 `original_size`를 주지 않으면 기본 해상도 기준으로
  확대되어 화면을 덮는다.
- 녹화는 GPU 인코딩(`h264_nvenc`)을 쓴다.

## 수정하고 싶다면

- 실행 순서·명령 : `notes/lab/demo_session.ps1`
- 해설 문구·자막 : `notes/lab/build_narration.py`의 `BLOCKS`
- 목소리 : 같은 파일 `VOICE = "Charon"` (대안 Schedar, Iapetus, Rasalgethi)
- 표지·결과 카드 : `notes/lab/compose_demo.py`의 `title_card`, `closing_card`
- 자막 크기 : 같은 파일 `FontSize=15`
