# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""시연영상 생성 — 슬라이드 이미지와 TTS 해설을 ffmpeg로 합성한다.

화면 녹화 대신 실제 실행 출력을 캡처해 슬라이드로 렌더링한다. 타이밍을 정확히
제어할 수 있고 각 장면 길이를 해설 음성에 맞출 수 있다.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"d:\opensource\ossp-2026-llm-router-challenge")
OUT = ROOT / "build" / "video"
W, H = 1920, 1080
BG = (18, 20, 26)
FG = (232, 236, 244)
DIM = (140, 150, 166)
ACC = (94, 200, 255)
OK = (110, 220, 150)
WARN = (255, 190, 90)
COLORS = {"fg": FG, "dim": DIM, "acc": ACC, "ok": OK, "warn": WARN}


def font(name: str, size: int):
    path = Path(r"C:\Windows\Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def styled(kind: str, size: int):
    return font({"t": "malgunbd.ttf", "b": "malgun.ttf",
                 "s": "malgun.ttf", "m": "consola.ttf"}[kind], size)


SIZES = {"t": 46, "b": 40, "s": 32, "m": 32}
STEPS = {"t": 74, "b": 60, "s": 50, "m": 46}


def slide(lines, path: Path, title: str = "", foot: str = ""):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    y = 90
    if title:
        draw.text((110, y), title, font=font("malgunbd.ttf", 64), fill=FG)
        y += 110
        draw.line([(110, y), (W - 110, y)], fill=(60, 66, 80), width=3)
        y += 50
    for text, kind, color in lines:
        draw.text((110, y), text, font=styled(kind, SIZES[kind]), fill=COLORS[color])
        y += STEPS[kind]
    if foot:
        draw.text((110, H - 90), foot, font=font("malgun.ttf", 30), fill=DIM)
    img.save(path)


def terminal_slide(capture: Path, path: Path, title: str, limit: int = 20):
    rows = []
    for line in capture.read_text(encoding="utf-8", errors="replace").splitlines()[:limit]:
        color = "fg"
        if line.startswith("$"):
            color = "acc"
        elif "[ OK ]" in line or "통과" in line:
            color = "ok"
        elif "최종 점수" in line:
            color = "warn"
        rows.append((line[:96], "m", color))
    slide(rows, path, title)


CAPTURES = {
    1: ("cap_route.txt", "실행 — 라우팅"),
    2: ("cap_check.txt", "실행 — 공식 채점기 검증"),
    3: ("cap_audit.txt", "실행 — 자체 점검 도구"),
}

SCENES = [
    {
        "name": "s01", "kind": 0, "title": "프롬프트만 보고 모델을 고르는 라우터",
        "foot": "SK텔레콤 지정과제 · Efficient LLM Routing Challenge",
        "lines": [
            ("LLM 서비스에서 추론 비용은 가장 큰 지출입니다.", "b", "fg"),
            ("그런데 트래픽의 상당수는 값싼 모델로도 충분한 질의입니다.", "b", "fg"),
            ("", "b", "fg"),
            ("이 라우터는 프롬프트만 보고 세 모델 중 하나를 골라", "b", "fg"),
            ("정해진 예산 안에서 답변 품질을 최대로 끌어올립니다.", "b", "fg"),
            ("", "b", "fg"),
            ("제약", "t", "acc"),
            ("· 프롬프트 내용과 예산 등급만 입력됩니다", "s", "dim"),
            ("· 모델을 호출하거나 답을 비교할 수 없습니다", "s", "dim"),
            ("· 예산을 넘기면 해당 등급은 0점입니다", "s", "warn"),
        ],
        "say": "LLM 서비스에서 추론 비용은 가장 큰 지출입니다. 그런데 실제 트래픽의 "
               "상당수는 값싼 모델로도 충분한 단순 질의입니다. 이 라우터는 프롬프트만 "
               "보고 세 모델 중 하나를 골라, 정해진 예산 안에서 답변 품질을 최대로 "
               "끌어올립니다. 모델을 호출하지도, 답을 비교하지도 않습니다. 한 번에 "
               "정해야 합니다. 그리고 예산을 넘기면 그 등급은 통째로 0점입니다.",
    },
    {
        "name": "s02", "kind": 1,
        "say": "먼저 실제로 돌려보겠습니다. 공개 검증 자료 880문항을 프리미엄 등급으로 "
               "라우팅합니다. 4초가 걸렸고 제한은 등급당 90초입니다.",
    },
    {
        "name": "s03", "kind": 2,
        "say": "공식 채점기로 확인하면 세 등급 모두 예산 안에 있고, 최종 점수는 "
               "영 점 칠일육칠입니다. 예산은 각각 한도의 83, 80, 65 퍼센트만 썼습니다.",
    },
    {
        "name": "s04", "kind": 0, "title": "어떻게 정하는가", "foot": "",
        "lines": [
            ("프롬프트에서 세 가지를 뽑습니다", "t", "acc"),
            ("· 직접 계산 특징 36개 — 길이, 한글 비율, 수식·코드 표지, 객관식 표지", "s", "fg"),
            ("· TF-IDF 단어 1–2그램 6만 차원", "s", "fg"),
            ("· TF-IDF 문자 3–5그램 12만 차원", "s", "fg"),
            ("", "s", "fg"),
            ("여기서 품질과 함께 토큰 사용량까지 예측합니다", "t", "acc"),
            ("라우팅 시점에는 토큰 수가 주어지지 않아 비용도 예측해야 합니다", "s", "dim"),
            ("", "s", "fg"),
            ("그다음 이득 대비 비용이 큰 순서로 승격합니다", "t", "acc"),
        ],
        "say": "프롬프트에서 세 가지를 뽑습니다. 길이와 한글 비율, 수식과 코드 표지 같은 "
               "직접 계산 특징 36개, 단어 티에프 아이디에프, 문자 티에프 아이디에프입니다. "
               "여기서 모델별 품질과 함께 토큰 사용량까지 예측합니다. 라우팅 시점에는 "
               "토큰 수가 주어지지 않기 때문에 비용도 예측해야 합니다. 그다음 이득 대비 "
               "비용이 큰 순서로 승격합니다.",
    },
    {
        "name": "s05", "kind": 0, "title": "설계에서 신경 쓴 것 ①  순서 불변성", "foot": "",
        "lines": [
            ("예산은 배치 전체에 걸린 제약이라", "b", "fg"),
            ("문항을 독립적으로 처리할 수 없습니다.", "b", "fg"),
            ("그런데 입력 순서에 의존하면 규칙 위반입니다.", "b", "fg"),
            ("", "b", "fg"),
            ("→ 동률은 프롬프트 내용의 SHA-256 해시로만 정렬", "t", "ok"),
            ("→ 내용이 같은 문항은 그룹으로 묶어 통째로 승격", "t", "ok"),
            ("", "b", "fg"),
            ("운영자가 순서를 섞고 문항 ID를 바꿔 재실행해도 결과가 같습니다.", "s", "dim"),
        ],
        "say": "설계에서 두 가지를 신경 썼습니다. 첫째, 순서 불변성입니다. 예산은 배치 "
               "전체에 걸린 제약이라 문항을 독립적으로 처리할 수 없습니다. 그런데 입력 "
               "순서에 의존하면 규칙 위반입니다. 그래서 동률은 프롬프트 내용의 해시로만 "
               "정렬하고, 내용이 같은 문항은 그룹으로 묶어 통째로 승격합니다. 운영자가 "
               "순서를 섞고 문항 아이디를 바꿔 재실행해도 결과가 같습니다.",
    },
    {
        "name": "s06", "kind": 3,
        "say": "직접 만든 자체 점검 도구입니다. 아이디와 순서 불변성, 결정성, 엣지 케이스, "
               "실행 시간, 예산 여유를 한 번에 확인합니다. 지금 열 항목 모두 통과입니다. "
               "실행 시간은 등급당 4.2초로 한도 90초의 5퍼센트입니다.",
    },
    {
        "name": "s07", "kind": 0, "title": "설계에서 신경 쓴 것 ②  예산 안전", "foot": "",
        "lines": [
            ("추론 모델은 출력 토큰이 극단적으로 튑니다", "t", "warn"),
            ("중앙값 1,570 토큰      최대 130,504 토큰", "m", "fg"),
            ("한 문항이 전체 경량 비용의 26%를 먹을 수 있습니다", "b", "warn"),
            ("", "b", "fg"),
            ("예측 비용으로 위험한 문항을 거르려 했지만 실패", "t", "dim"),
            ("사고를 내는 문항은 '예측은 작은데 실제가 큰' 것들이라", "s", "dim"),
            ("예측 기반 필터를 그대로 통과합니다", "s", "dim"),
            ("", "b", "fg"),
            ("→ 선택 건수 자체를 묶었습니다", "t", "ok"),
            ("예측이 틀려도 노출이 제한됩니다", "s", "ok"),
        ],
        "say": "둘째, 예산 안전입니다. 가장 비싼 추론 모델은 출력 토큰이 극단적으로 "
               "튑니다. 중앙값은 천오백칠십 토큰인데 최대는 십삼만 토큰입니다. 한 문항이 "
               "전체 경량 비용의 26퍼센트를 먹을 수 있어서, 몇 건만 빗나가도 등급 전체가 "
               "0점이 됩니다. 예측 비용으로 위험한 문항을 걸러내려 했지만 실패했습니다. "
               "사고를 내는 문항은 예측은 작은데 실제가 큰 것들이라 예측 기반 필터를 "
               "그대로 통과합니다. 그래서 선택 건수 자체를 묶었습니다. 예측이 틀려도 "
               "노출이 제한됩니다.",
    },
    {
        "name": "s08", "kind": 0, "title": "검증과 판단", "foot": "",
        "lines": [
            ("2,640문항을 5-fold와 8-fold로 나눠", "b", "fg"),
            ("양쪽에서 예산 초과가 0건인 조합만 후보로 삼았습니다", "b", "fg"),
            ("", "b", "fg"),
            ("이 과정에서 두 번 되돌렸습니다", "t", "warn"),
            ("· 부스팅 트리 — 토큰 예측 상관 0.37 → 0.65, 단일 측정 최고점", "s", "fg"),
            ("  그러나 교차검증에서 어떤 마진으로도 예산을 못 지킴", "s", "dim"),
            ("· 문장 임베딩 — 오히려 예측이 나빠짐 (0.425 → 0.372)", "s", "fg"),
            ("  난이도 신호가 의미 유사도가 아니라 표면 구조에 있기 때문", "s", "dim"),
            ("", "b", "fg"),
            ("점수 0.01을 얻으려다 등급 0점이면 최종의 30%를 잃습니다", "t", "ok"),
        ],
        "say": "설정은 감으로 정하지 않았습니다. 2640문항을 5겹과 8겹으로 나눠, 양쪽에서 "
               "예산 초과가 한 건도 없는 조합만 후보로 삼았습니다. 이 과정에서 두 번 "
               "되돌렸습니다. 부스팅 트리는 토큰 예측 상관을 영 점 삼칠에서 영 점 육오로 "
               "올렸고 단일 측정에서도 가장 높았지만, 교차검증에서는 어떤 마진으로도 "
               "예산을 지키지 못했습니다. 임베딩 모델은 오히려 예측이 나빠졌습니다. "
               "이 과제의 난이도 신호가 의미 유사도가 아니라 표면 구조에 있기 "
               "때문입니다. 점수 영 점 영일을 얻으려다 등급 0점을 맞으면 최종 점수의 "
               "30퍼센트를 잃습니다. 측정 결과를 따랐습니다.",
    },
    {
        "name": "s09", "kind": 0, "title": "정리", "foot": "github.com/leelang7/ossp-2026-llm-router-challenge",
        "lines": [
            ("공개 Dev 880문항          0.7167", "m", "ok"),
            ("교차검증 2,640문항        0.6643    예산 초과 0건", "m", "ok"),
            ("공식 최강 baseline        0.6954", "m", "dim"),
            ("전부 경량 모델            0.6193", "m", "dim"),
            ("", "b", "fg"),
            ("학습은 scikit-learn, 실행은 NumPy 하나", "b", "fg"),
            ("TF-IDF를 표준 라이브러리로 재현해 두 경로 예측 차이 1.2e-15", "s", "dim"),
            ("이미지 73MB · 등급당 4.2초 · Apache-2.0 공개", "s", "dim"),
        ],
        "say": "학습은 사이킷런으로, 실행은 넘파이 하나로 합니다. 티에프 아이디에프 "
               "동작을 표준 라이브러리로 재현해 두 경로의 예측 차이가 부동소수점 "
               "한계까지 일치합니다. 이미지는 73메가바이트, 코드는 아파치 2.0으로 "
               "공개했습니다. 감사합니다.",
    },
]


async def synthesize(text: str, path: Path) -> None:
    import edge_tts
    speech = edge_tts.Communicate(text, "ko-KR-InJoonNeural", rate="+18%")
    await speech.save(str(path))


def duration_of(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(result.stdout.strip())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    segments = []
    total = 0.0
    for scene in SCENES:
        name = scene["name"]
        png = OUT / f"{name}.png"
        if scene["kind"] == 0:
            slide(scene["lines"], png, scene.get("title", ""), scene.get("foot", ""))
        else:
            capture, title = CAPTURES[scene["kind"]]
            terminal_slide(OUT / capture, png, title)

        mp3 = OUT / f"{name}.mp3"
        asyncio.run(synthesize(scene["say"], mp3))
        seconds = duration_of(mp3) + 0.8
        total += seconds

        segment = OUT / f"{name}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png),
            "-i", str(mp3), "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac",
            "-b:a", "128k", "-pix_fmt", "yuv420p", "-r", "12",
            "-t", f"{seconds:.2f}", str(segment)], check=True)
        segments.append(segment)
        print(f"  {name}  {seconds:5.1f}초")

    listing = OUT / "concat.txt"
    listing.write_text("".join("file '%s'\n" % p.name for p in segments), encoding="utf-8")
    final = OUT / "routerx_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", "concat.txt", "-c", "copy", final.name],
                   check=True, cwd=str(OUT))

    length = duration_of(final)
    print(f"\n완성: {final}")
    print(f"  길이 {int(length // 60)}분 {length % 60:04.1f}초 / 제한 3분")
    print(f"  크기 {final.stat().st_size / 1e6:.1f} MB")
    if length > 180:
        print("  경고: 3분을 넘습니다. 나레이션을 줄이십시오.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
