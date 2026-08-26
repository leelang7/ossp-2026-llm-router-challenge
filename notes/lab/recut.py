# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""녹화 재편집 — 대기는 줄이고, 결과 화면은 설명할 만큼 남긴다.

한 번에 쭉 녹화한 session.mp4는 시연영상으로 쓰기에 두 가지가 어긋난다.

  - 실행 대기가 길다. 자체 점검 33초, 컨테이너 52초 동안 화면이 멈춰 있다.
  - 반대로 결과는 순식간에 지나간다. 라우팅 결과가 뜬 지 2초 만에 다음
    단계로 넘어가 해설을 얹을 자리가 없다.

그래서 대기 구간은 잘라내고, 설명이 필요한 결과 화면은 그 프레임을 정지로
늘려 붙인다. PLAN의 각 항목이 편집본에서 차지하는 구간을 함께 출력하므로,
lab/build_narration.py의 BLOCKS 시각을 여기에 맞추면 된다.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(r"d:\opensource\ossp-2026-llm-router-challenge")
SRC = ROOT / "build" / "rec" / "session.mp4"
SEG = ROOT / "build" / "rec" / "seg"
OUT = ROOT / "build" / "rec" / "cut.mp4"
ENCODE = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "22",
          "-pix_fmt", "yuv420p", "-r", "15", "-vf", "scale=1920:1080,setsar=1"]

# (이름, clip=그대로 / hold=그 프레임 정지, 원본 시각, 길이)
PLAN = [
    ("a", "clip",   3.2, 16.0),   # 인트로·라우팅 실행·선택 결과
    ("b", "hold",  19.1,  6.0),   # 선택 결과 화면 유지
    ("c", "clip",  19.2,  5.8),   # 채점기 명령
    ("d", "clip",  36.0, 14.0),   # 채점 결과·자체 점검 시작
    ("e", "hold",  49.9, 10.0),   # 점검 진행 중
    ("f", "clip",  78.0,  4.0),   # 점검 결과
    ("g", "hold",  81.9,  5.0),   # 점검 결과 화면 유지
    ("h", "clip",  82.0,  8.0),   # 컨테이너 명령
    ("i", "hold",  89.9,  8.0),   # 컨테이너 실행 중
    ("j", "clip", 138.0,  5.8),   # 컨테이너 결과·종료
    # 144.8초에서 터미널이 닫히며 바탕화면이 찍혔다. 그 앞 프레임으로 마무리한다.
    ("k", "hold", 143.6,  4.2),   # 마무리 여운
]


def main() -> int:
    SEG.mkdir(parents=True, exist_ok=True)
    at = 0.0
    for name, kind, src, length in PLAN:
        seg = SEG / f"{name}.mp4"
        if kind == "clip":
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(src), "-t", str(length),
                            "-i", str(SRC), *ENCODE, str(seg)], check=True)
        else:
            png = SEG / f"{name}.png"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(src),
                            "-i", str(SRC), "-frames:v", "1", str(png)], check=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-t", str(length),
                            "-i", str(png), *ENCODE, str(seg)], check=True)
            png.unlink()
        print(f"  {name}  {kind:<4} 원본 {src:>6.1f}s → 편집본 {at:>5.1f}~{at + length:<5.1f}")
        at += length

    listing = SEG / "list.txt"
    listing.write_text("".join(f"file '{n}.mp4'\n" for n, _k, _s, _d in PLAN), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", listing.name, "-c", "copy", str(OUT)], check=True, cwd=str(SEG))
    print(f"\n편집본 {at:.1f}초 · {OUT}")
    print("  build_narration.py의 REC_END와 BLOCKS 시각을 위 구간에 맞추십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
