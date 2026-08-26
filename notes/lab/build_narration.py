# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""시연영상 나레이션 생성 — Gemini TTS 프리미엄 음성.

AuraView/scripts/build_tts.py와 같은 방식이다. 차분한 남성 발표 톤(Charon)을 쓰고,
키가 여러 개면 429(할당량 초과) 시 다음 키로 넘어간다.

  python lab/build_narration.py            전체 생성
  python lab/build_narration.py 0 3        특정 블록만 재생성
"""
from __future__ import annotations

import base64
import re
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

KEY_DIR = Path(r"c:\Users\leesc\Documents\ThinkU\AuraView\secrets")
OUT = Path(r"d:\opensource\ossp-2026-llm-router-challenge\build\demo_video\tts")
OUT.mkdir(parents=True, exist_ok=True)

KEYS = []
for name in ("tts_key.txt", "tts_key2.txt", "tts_key3.txt"):
    path = KEY_DIR / name
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value and value not in KEYS:
            KEYS.append(value)
if not KEYS:
    raise SystemExit(f"키 없음: {KEY_DIR}")
print(f"[keys] {len(KEYS)}개 로드")

# 녹화 시작 직후 터미널이 화면을 덮기 전 구간(바탕화면)을 잘라내는 폭.
# 나레이션·자막 시각을 모두 이만큼 당긴다. compose_demo가 같은 값을 쓴다.
TRIM = 3.2
REC_END = 145.0     # 녹화 원본 길이(초)

VOICE = "Charon"
MODELS = ["gemini-3.1-flash-tts-preview"]
STYLE = ("차분하고 또렷한 전문 발표 톤으로, 신뢰감 있게, 너무 빠르지 않게, "
         "핵심 숫자는 분명하게 읽어줘:\n")

# (녹화 시각(초), TTS 낭독문, 자막문)
# TTS는 숫자를 한글로 적어야 자연스럽게 읽고, 자막은 숫자 그대로 보여야 읽기 쉽다.
BLOCKS = [
    (0,
     "질문 하나하나의 난이도를 프롬프트만 보고 가늠해, 세 후보 모델 중 가장 알맞은 "
     "하나에 배정하는 경량 LLM 라우터입니다.",
     "질문 하나하나의 난이도를 프롬프트만 보고 가늠해, 세 후보 모델 중 가장 알맞은 "
     "하나에 배정하는 경량 LLM 라우터입니다."),
    (14,
     "먼저 라우팅을 실행합니다. 공개 검증 자료 팔백팔십 문항을 프리미엄 등급으로 "
     "처리합니다. 사 점 이 초가 걸렸고, 제한은 등급당 구십 초입니다.",
     "먼저 라우팅을 실행합니다. 공개 검증 자료 880문항을 premium 등급으로 처리합니다. "
     "4.2초가 걸렸고 제한은 등급당 90초입니다."),
    (30,
     "선택 결과를 확인합니다. 팔백팔십 문항 전부에 모델이 하나씩 배정되었고, "
     "경량 모델과 중형 모델, 추론 모델의 분포를 볼 수 있습니다.",
     "선택 결과를 확인합니다. 880문항 전부에 모델이 하나씩 배정되었고, 경량·중형·추론 "
     "모델의 분포를 볼 수 있습니다."),
    (44,
     "공식 채점기로 검증합니다. 최종 점수 영 점 칠일육칠, 세 등급 모두 예산 안에 "
     "있습니다. 예산 사용률은 각각 팔십삼, 팔십, 육십오 퍼센트입니다.",
     "공식 채점기로 검증합니다. 최종 점수 0.7167, 세 등급 모두 예산 안에 있습니다. "
     "예산 사용률은 각각 83%, 80%, 65%입니다."),
    (62,
     "직접 만든 자체 점검 도구입니다. 문항 아이디와 입력 순서를 바꿔 재실행해도 "
     "결과가 같은지, 같은 입력에 같은 출력이 나오는지, 엣지 케이스에서 죽지 않는지, "
     "실행 시간과 예산 여유가 충분한지를 한 번에 확인합니다.",
     "직접 만든 자체 점검 도구입니다. 문항 ID와 입력 순서를 바꿔 재실행해도 결과가 "
     "같은지, 같은 입력에 같은 출력이 나오는지, 엣지 케이스에서 죽지 않는지, 실행 "
     "시간과 예산 여유가 충분한지를 한 번에 확인합니다."),
    (86,
     "열 항목 모두 통과입니다. 순서 불변성은 사백 문항에서 불일치 영 건, 실행 시간은 "
     "한도의 오 퍼센트, 예산은 세 등급 모두 여유를 남기고 통과했습니다.",
     "10항목 모두 통과입니다. 순서 불변성은 400문항에서 불일치 0건, 실행 시간은 한도의 "
     "5%, 예산은 세 등급 모두 여유를 남기고 통과했습니다."),
    (104,
     "마지막으로 실제 제출 이미지를 공식 자원 한도에서 실행합니다. 네트워크 차단, "
     "읽기 전용 루트, 씨피유 두 개, 메모리 이 기가바이트 조건입니다. 레지스트리에서 "
     "내려받은 이미지를 다이제스트로 지정해 실행합니다.",
     "마지막으로 실제 제출 이미지를 공식 자원 한도에서 실행합니다. 네트워크 차단, "
     "읽기 전용 루트, CPU 2코어, 메모리 2GiB 조건입니다. 레지스트리에서 내려받은 "
     "이미지를 다이제스트로 지정해 실행합니다."),
    (128,
     "정상 종료되었습니다. 학습은 사이킷런으로 하고, 실행 이미지에는 넘파이 하나만 "
     "넣었습니다. 두 경로의 예측 차이는 부동소수점 한계까지 일치합니다.",
     "정상 종료되었습니다. 학습은 scikit-learn으로 하고 실행 이미지에는 NumPy 하나만 "
     "넣었습니다. 두 경로의 예측 차이는 부동소수점 한계까지 일치합니다."),
]


def synth(text: str):
    body = {
        "contents": [{"parts": [{"text": STYLE + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}},
        },
    }
    last = ""
    for model in MODELS:
        for index, key in enumerate(KEYS):
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            for attempt in range(2):
                request = urllib.request.Request(
                    url, data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"})
                try:
                    payload = json.load(urllib.request.urlopen(request, timeout=300))
                    audio = payload["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                    return base64.b64decode(audio), model
                except urllib.error.HTTPError as exc:
                    last = f"key#{index + 1} {model}: {exc.code}"
                    if exc.code == 429:
                        print(f"    key#{index + 1} 429 할당량 — 다음 키로")
                    break
                except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
                    last = f"key#{index + 1} {model}: timeout {str(exc)[:50]}"
                    if attempt < 1:
                        print(f"    key#{index + 1} timeout — 10초 후 재시도")
                        time.sleep(10)
                        continue
                    break
    raise RuntimeError("모든 키/모델 실패 · " + last)


def duration_of(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nokey=1", str(path)],
                         capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def silence_marks(path: Path) -> list[float]:
    """음성 안의 쉼(무음) 중앙 시각을 모두 찾아 돌려준다.

    자막을 글자 수 비례로만 나누면 실제 발화 시점과 어긋난다. 낭독 사이의
    쉼을 찾아 두었다가 자막 경계를 거기에 붙이면 목소리와 자막이 맞는다.
    """
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path),
         "-af", "silencedetect=noise=-38dB:d=0.13", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    marks, start = [], None
    for line in (out.stderr or "").splitlines():
        if "silence_start:" in line:
            start = float(line.rsplit("silence_start:", 1)[1].split()[0])
        elif "silence_end:" in line and start is not None:
            end = float(line.rsplit("silence_end:", 1)[1].split("|")[0].strip())
            marks.append((start + end) / 2)
            start = None
    span = duration_of(path)
    return [m for m in marks if 0.4 < m < span - 0.4]


def phrases(text: str, low: int = 10, high: int = 32) -> list[str]:
    """자막 한 장에 들어갈 크기로 문장을 나눈다.

    한 장이 10초씩 떠 있으면 읽기 힘들고, 두 줄을 넘으면 뒷부분이 화면에서
    잘린다. 마침표·쉼표 뒤 공백에서만 끊어야 "0.7167"이나 "4.2초"의 소수점이
    문장 끝으로 오인되지 않는다.
    """
    tokens = [t for t in re.split(r"(?<=[.,])\s+", text.strip()) if t]
    parts, buf = [], ""
    for token in tokens:
        joined = f"{buf} {token}".strip()
        if buf and len(joined) > high:
            parts.append(buf)
            buf = token
        else:
            buf = joined
        if len(buf) >= low and buf.endswith("."):
            parts.append(buf)
            buf = ""
    if buf:
        if parts and len(buf) < low:
            parts[-1] = f"{parts[-1]} {buf}"
        else:
            parts.append(buf)
    return [x.rstrip(",") for x in parts]


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--srt":
        targets = []          # 음성은 그대로 두고 자막만 다시 만든다
    elif args:
        targets = [int(a) for a in args]
    else:
        targets = range(len(BLOCKS))
    total = 0.0
    used = None
    for i in targets:
        start, say, _caption = BLOCKS[i]
        pcm, used = synth(say)
        raw = OUT / f"n{i:02d}.raw"
        raw.write_bytes(pcm)
        mp3 = OUT / f"n{i:02d}.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", "24000",
                        "-ac", "1", "-i", str(raw), "-c:a", "libmp3lame", "-q:a", "2",
                        str(mp3)], check=True)
        raw.unlink(missing_ok=True)
        seconds = duration_of(mp3)
        total += seconds
        print(f"  n{i:02d}  {start:>4}s 시작  {seconds:5.1f}초  {say[:30]}…")
        time.sleep(16)     # 분당 한도 회피

    # 자막(SRT) — 긴 문장은 마침표 단위로 쪼개고, 음성 길이에 비례해 시간을 나눈다.
    def stamp(t: float) -> str:
        t = max(0.0, t)
        h, rem = divmod(t, 3600)
        m, sec = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}".replace(".", ",")

    def wrap(line: str, width: int = 30) -> str:
        out, cur = [], ""
        for word in line.split():
            if cur and len(cur) + 1 + len(word) > width:
                out.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur:
            out.append(cur)
        return "\n".join(out[:2])

    # 블록이 다음 블록 시작을 넘기면 음성 두 개가 동시에 들리고 자막도 겹친다.
    # 여기서 먼저 잡아 두어야 합성 뒤에 발견하는 일이 없다.
    starts = [b[0] for b in BLOCKS]
    placed = [max(0.0, t - TRIM) for t in starts] + [REC_END - TRIM]
    spans = {}
    overlap = []
    for i, (_start, _say, _cap) in enumerate(BLOCKS):
        mp3 = OUT / f"n{i:02d}.mp3"
        if not mp3.exists():
            continue
        spans[i] = duration_of(mp3)
        room = placed[i + 1] - placed[i]
        if spans[i] > room:
            overlap.append((i, spans[i], room))
    for i, length, room in overlap:
        print(f"  경고: 블록 {i} 음성 {length:.1f}초 > 구간 {room:.1f}초 "
              f"— {length - room:.1f}초 겹침. 문안을 줄이십시오.")

    entries = []
    for i, (_start, _say, text) in enumerate(BLOCKS):
        if i not in spans:
            continue
        # 트림한 만큼 당기고, 다음 블록 시작을 넘지 않도록 자막을 자른다
        begin, limit = placed[i], placed[i + 1]
        pieces = phrases(text)
        marks = silence_marks(OUT / f"n{i:02d}.mp3")
        weights = [len(x) for x in pieces]
        total_w = sum(weights) or 1
        # 글자 수 비례로 경계를 잡고, 근처(±1.2초)에 쉼이 있으면 그쪽으로 옮긴다
        bounds, acc, prev = [0.0], 0.0, 0.0
        for w in weights[:-1]:
            acc += spans[i] * w / total_w
            near = [m for m in marks if abs(m - acc) <= 1.2 and m > prev + 0.4]
            prev = min(near, key=lambda m: abs(m - acc)) if near else acc
            bounds.append(prev)
        bounds.append(spans[i])
        for k, piece in enumerate(pieces):
            a = min(begin + bounds[k], limit)
            b = min(begin + bounds[k + 1], limit)
            if b > a + 0.15:
                entries.append((a, b, wrap(piece)))

    lines = [f"{n}\n{stamp(a)} --> {stamp(b)}\n{body}\n"
             for n, (a, b, body) in enumerate(entries, start=1)]
    srt = OUT / "narration.srt"
    srt.write_text("\n".join(lines), encoding="utf-8")
    print(f"자막 {len(entries)}개 · {srt}")
    if used:
        print(f"음성 {used} · {VOICE}")
    return 1 if overlap else 0


if __name__ == "__main__":
    raise SystemExit(main())
