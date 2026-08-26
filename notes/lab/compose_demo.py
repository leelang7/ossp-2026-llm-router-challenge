# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""시연영상 합성 — 실행 녹화 + 성우 나레이션 + 자막 + 앞뒤 표지.

본편은 프로그램이 실제로 동작하는 화면 녹화다. 여기에 Gemini TTS로 만든
해설 음성을 구간별로 얹고, 같은 문장을 자막으로 화면 하단에 굽는다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"d:\opensource\ossp-2026-llm-router-challenge")
REC = ROOT / "build" / "rec" / "session.mp4"
WORK = ROOT / "build" / "demo_video"
TTS = WORK / "tts"
W, H = 1920, 1080
BG = (15, 17, 23)
FG = (234, 238, 246)
DIM = (146, 156, 174)
ACC = (86, 204, 242)
OK = (94, 224, 160)
FONTS = Path(r"C:\Windows\Fonts")
# 녹화 시작 직후 몇 초는 터미널이 화면을 덮기 전이라 바탕화면이 그대로 찍힌다.
# 그만큼을 잘라내고, 나레이션·자막 시각도 같은 폭으로 당긴다.
TRIM = 3.2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_narration import BLOCKS  # noqa: E402


def font(name: str, size: int):
    return ImageFont.truetype(str(FONTS / name), size)


def title_card(path: Path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((150, 360), "프롬프트 난이도 인지 기반", font=font("malgunbd.ttf", 74), fill=FG)
    d.text((150, 464), "경량 LLM 라우터", font=font("malgunbd.ttf", 74), fill=FG)
    d.line([(150, 600), (1770, 600)], fill=(60, 66, 80), width=3)
    d.text((150, 642), "팀 트리아지", font=font("malgunbd.ttf", 46), fill=ACC)
    d.text((150, 716), "SK텔레콤 지정과제 · Efficient LLM Routing Challenge",
           font=font("malgun.ttf", 34), fill=DIM)
    d.text((150, 776), "2026 오픈소스 개발자대회", font=font("malgun.ttf", 34), fill=DIM)
    img.save(path)


def closing_card(path: Path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((150, 150), "결과", font=font("malgunbd.ttf", 64), fill=FG)
    d.line([(150, 252), (1770, 252)], fill=(60, 66, 80), width=3)
    y = 310
    for left, right, color in (("공개 Dev 880문항", "0.7167", OK),
                               ("교차검증 2,640문항", "0.6643", OK),
                               ("공식 최고 baseline", "0.6954", DIM),
                               ("전량 경량 모델", "0.6193", DIM)):
        d.text((150, y), left, font=font("malgun.ttf", 42), fill=FG)
        f = font("consolab.ttf", 46)
        d.text((1770 - d.textbbox((0, 0), right, font=f)[2], y), right, font=f, fill=color)
        y += 76
    y += 44
    for line in ("교차검증 5-fold · 8-fold 양쪽에서 예산 초과 0건",
                 "실행 의존성 NumPy 단일 패키지 · 이미지 73MB · 등급당 4.2초",
                 "Apache-2.0 공개"):
        d.text((150, y), line, font=font("malgun.ttf", 34), fill=DIM)
        y += 58
    d.text((150, H - 112), "github.com/leelang7/ossp-2026-llm-router-challenge",
           font=font("malgun.ttf", 30), fill=ACC)
    img.save(path)


def duration_of(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nokey=1", str(path)],
                         capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def main() -> int:
    rec_len = duration_of(REC) - TRIM
    print(f"녹화 {rec_len:.1f}초 (앞 {TRIM}초 잘라냄)")

    # 1) 나레이션을 시각에 맞춰 배치하고 하나로 섞는다
    inputs = ["-ss", f"{TRIM}", "-i", str(REC)]
    filters = []
    count = 0
    for index, (start, _say, _caption) in enumerate(BLOCKS):
        mp3 = TTS / f"n{index:02d}.mp3"
        if not mp3.exists():
            continue
        inputs += ["-i", str(mp3)]
        count += 1
        delay = int(max(0.0, start - TRIM) * 1000)
        filters.append(f"[{count}:a]adelay={delay}|{delay}[d{count}]")
    mix = "".join(f"[d{i + 1}]" for i in range(count))
    filters.append(f"{mix}amix=inputs={count}:normalize=0:duration=longest[aout]")

    # 2) 자막을 화면 하단에 굽는다.
    #    Windows 경로는 subtitles 필터에서 이스케이프가 까다로우므로
    #    작업 디렉터리를 자막 폴더로 옮기고 파일 이름만 넘긴다(아래 cwd 참고).
    #    original_size를 주지 않으면 자막이 기본 해상도 기준으로 확대되어 화면을 덮는다.
    srt = "narration.srt"
    style = ("FontName=Malgun Gothic,FontSize=15,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&HC0000000,BackColour=&HB0000000,BorderStyle=3,"
             "Outline=3,Shadow=0,MarginV=28,Alignment=2")
    filters.append(
        f"[0:v]subtitles='{srt}':original_size={W}x{H}:force_style='{style}'[vout]")

    body = WORK / "body.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs,
                    "-filter_complex", ";".join(filters),
                    "-map", "[vout]", "-map", "[aout]",
                    "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                    "-pix_fmt", "yuv420p", "-r", "15", "-t", f"{rec_len:.2f}",
                    str(body)], check=True, cwd=str(TTS))
    print(f"본편 합성 완료 {duration_of(body):.1f}초")

    # 3) 앞뒤 표지 (본편과 같은 규격으로 맞춰야 이어붙일 때 깨지지 않는다)
    parts = []
    for name, render, hold in (("title", title_card, 4.5), ("closing", closing_card, 7.5)):
        png = WORK / f"{name}.png"
        render(png)
        seg = WORK / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-loop", "1", "-t", str(hold), "-i", str(png),
                        "-f", "lavfi", "-t", str(hold),
                        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                        "-pix_fmt", "yuv420p", "-r", "15", "-shortest", str(seg)], check=True)
        parts.append(seg)

    # 4) 이어붙이기 — 규격이 같으므로 필터 concat으로 안전하게
    final = WORK / "routerx_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(parts[0]), "-i", str(body), "-i", str(parts[1]),
                    "-filter_complex",
                    "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24",
                    "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
                    str(final)], check=True)

    length = duration_of(final)
    print(f"\n완성: {final}")
    print(f"  {int(length // 60)}분 {length % 60:04.1f}초 / 제한 3분   "
          f"{final.stat().st_size / 1e6:.1f}MB")
    if length > 180:
        print("  경고: 3분 초과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
