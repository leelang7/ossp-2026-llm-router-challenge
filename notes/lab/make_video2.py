# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""시연영상 생성 v2 — 개조식 슬라이드와 TTS 해설을 ffmpeg로 합성한다.

v1의 문제를 고쳤다.
  · Consolas에 한글이 없어 터미널 캡처의 한글이 전부 두부로 깨졌다
    → 한글이 섞인 줄은 맑은 고딕, 순수 영문·숫자는 Consolas로 자동 분기
  · 터미널 원문을 그대로 붙여 화면이 비고 읽기 어려웠다
    → 결과를 카드와 수치 강조로 재구성
  · 문장형 서술이 갑갑했다
    → 개조식으로 다시 씀
"""
from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"d:\opensource\ossp-2026-llm-router-challenge")
OUT = ROOT / "build" / "video2"
W, H = 1920, 1080
PAD = 130

BG = (15, 17, 23)
PANEL = (24, 27, 35)
LINE = (48, 54, 68)
FG = (234, 238, 246)
DIM = (146, 156, 174)
ACC = (86, 204, 242)
OK = (94, 224, 160)
WARN = (255, 196, 92)
RED = (255, 120, 120)
COLORS = {"fg": FG, "dim": DIM, "acc": ACC, "ok": OK, "warn": WARN, "red": RED}

FONTS = Path(r"C:\Windows\Fonts")
_HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
_cache: dict = {}


def load(name: str, size: int):
    key = (name, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(str(FONTS / name), size)
    return _cache[key]


def pick(text: str, size: int, bold: bool = False, mono: bool = False):
    """한글이 섞이면 맑은 고딕, 순수 영문·숫자면 Consolas를 쓴다."""
    if mono and not _HANGUL.search(text):
        return load("consolab.ttf" if bold else "consola.ttf", size)
    return load("malgunbd.ttf" if bold else "malgun.ttf", size)


def text_width(draw, s, font):
    return draw.textbbox((0, 0), s, font=font)[2]


class Slide:
    def __init__(self, title: str = "", kicker: str = ""):
        self.img = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.img)
        self.y = 96
        if kicker:
            self.d.text((PAD, self.y), kicker, font=pick(kicker, 30, True), fill=ACC)
            self.y += 48
        if title:
            self.d.text((PAD, self.y), title, font=pick(title, 62, True), fill=FG)
            self.y += 96
            self.d.line([(PAD, self.y), (W - PAD, self.y)], fill=LINE, width=3)
            self.y += 46

    def bullet(self, text: str, color: str = "fg", size: int = 40, gap: int = 20):
        self.d.ellipse([PAD + 6, self.y + size // 2 - 4, PAD + 16, self.y + size // 2 + 6],
                       fill=COLORS[color] if color != "fg" else DIM)
        self.d.text((PAD + 38, self.y), text, font=pick(text, size), fill=COLORS[color])
        self.y += size + gap

    def sub(self, text: str, color: str = "dim", size: int = 32, gap: int = 14):
        self.d.text((PAD + 40, self.y), text, font=pick(text, size), fill=COLORS[color])
        self.y += size + gap

    def line(self, text: str, color: str = "fg", size: int = 40, gap: int = 18,
             bold: bool = False, indent: int = 0):
        self.d.text((PAD + indent, self.y), text,
                    font=pick(text, size, bold), fill=COLORS[color])
        self.y += size + gap

    def gap(self, n: int = 26):
        self.y += n

    def panel(self, rows, width: int = None, head=None, mid_x: float = 0.62):
        """수치 표를 카드로 그린다.

        rows = [(왼쪽, 오른쪽, 색)] 또는 [(왼쪽, 가운데, 오른쪽, 색)]
        """
        width = width or (W - PAD * 2)
        row_h = 66
        height = row_h * (len(rows) + (1 if head else 0)) + 44
        self.d.rounded_rectangle([PAD, self.y, PAD + width, self.y + height],
                                 radius=18, fill=PANEL, outline=LINE, width=2)
        right_edge = PAD + width - 34
        mid = PAD + int(width * mid_x)
        yy = self.y + 22
        if head:
            self.d.text((PAD + 34, yy), head[0], font=pick(head[0], 30, True), fill=DIM)
            hf = pick(head[-1], 30, True)
            self.d.text((right_edge - text_width(self.d, head[-1], hf), yy),
                        head[-1], font=hf, fill=DIM)
            yy += row_h
        for row in rows:
            left, color = row[0], row[-1]
            self.d.text((PAD + 34, yy), left, font=pick(left, 38, mono=True), fill=FG)
            if len(row) == 4:
                center = row[1]
                cf = pick(center, 40, True, mono=True)
                self.d.text((mid - text_width(self.d, center, cf), yy), center,
                            font=cf, fill=COLORS[color])
            right = row[-2]
            rf = pick(right, 40, True, mono=True)
            self.d.text((right_edge - text_width(self.d, right, rf), yy), right,
                        font=rf, fill=COLORS[color])
            yy += row_h
        self.y += height + 30

    def big(self, value: str, label: str, color: str = "ok"):
        f = load("consolab.ttf", 128)
        self.d.text((PAD, self.y), value, font=f, fill=COLORS[color])
        vw = text_width(self.d, value, f)
        self.d.text((PAD + vw + 34, self.y + 62), label,
                    font=pick(label, 36), fill=DIM)
        self.y += 158

    def foot(self, text: str):
        self._foot = text

    def save(self, path: Path):
        # 내용을 수직 중앙에 두어 아래쪽이 비어 보이지 않게 한다.
        content = self.y
        room = H - content
        if room > 180:
            shift = min(room // 2 - 40, 150)
            if shift > 0:
                canvas = Image.new("RGB", (W, H), BG)
                canvas.paste(self.img.crop((0, 0, W, content + 20)), (0, shift))
                self.img = canvas
                self.d = ImageDraw.Draw(self.img)
        if getattr(self, "_foot", None):
            self.d.text((PAD, H - 84), self._foot,
                        font=pick(self._foot, 28), fill=DIM)
        self.img.save(path)


def s01(p):
    s = Slide("쉬운 질문은 싸게, 어려운 질문만 비싸게",
              "SK텔레콤 지정과제 · Efficient LLM Routing Challenge")
    s.bullet("추론 비용은 LLM 서비스 최대 지출 항목", "fg")
    s.bullet("트래픽 다수는 경량 모델로도 충분", "fg")
    s.gap(10)
    s.line("프롬프트만 보고 세 모델 중 하나를 선택", "acc", 46, bold=True)
    s.gap(24)
    s.line("제약 조건", "warn", 36, bold=True)
    s.sub("입력 — 프롬프트 내용과 예산 등급뿐")
    s.sub("금지 — 모델 호출, 답변 비교, 선택 번복")
    s.sub("예산 초과 시 해당 등급 0점", "red")
    s.foot("팀 트리아지")
    s.save(p)


def s02(p):
    s = Slide("실행", "1 / 라우팅")
    s.line("$ python3 -m routerx.cli --tier premium", "acc", 36, bold=True, indent=0)
    s.sub("공개 검증 자료 880문항 · premium 등급", "dim", 32)
    s.gap(20)
    s.big("4.2초", "등급당 실행 시간", "ok")
    s.panel([("한도", "90초", "dim"),
             ("사용", "4.7%", "ok"),
             ("문항당", "4.8ms", "ok")])
    s.save(p)


def s03(p):
    s = Slide("공식 채점기 검증", "2 / 결과")
    s.big("0.7167", "최종 점수 · 공개 Dev 880문항", "ok")
    s.panel([("fast", "0.6744", "예산 83%", "ok"),
             ("balanced", "0.7303", "예산 80%", "ok"),
             ("premium", "0.7593", "예산 65%", "ok")],
            head=("등급", "점수", "예산 사용률"))
    s.line("세 등급 모두 예산 통과", "ok", 40, bold=True)
    s.save(p)


def s04(p):
    s = Slide("무엇을 예측하는가", "3 / 구조")
    s.line("입력 특징", "acc", 38, bold=True)
    s.sub("직접 계산 36개 — 길이, 한글 비율, 수식·코드 표지, 객관식 표지")
    s.sub("TF-IDF 단어 1–2그램 6만 차원")
    s.sub("TF-IDF 문자 3–5그램 12만 차원")
    s.gap(22)
    s.line("예측 대상 — ridge 회귀 7개 헤드", "acc", 38, bold=True)
    s.sub("모델별 품질 3")
    s.sub("모델별 출력 토큰 3 + 입력 토큰 1  →  비용 환산")
    s.sub("라우팅 시점에 토큰 수가 없어 비용도 예측 대상", "warn")
    s.gap(22)
    s.line("선택 — 이득 ÷ 비용이 큰 순서로 승격", "ok", 42, bold=True)
    s.save(p)


def s05(p):
    s = Slide("순서에 의존하지 않는 배치 정책", "4 / 설계 ①")
    s.bullet("예산은 배치 전체 제약 → 문항 독립 처리 불가", "fg")
    s.bullet("입력 순서 의존 시 규칙 위반", "red")
    s.gap(18)
    s.line("해결", "acc", 38, bold=True)
    s.sub("동률은 프롬프트 내용 SHA-256 해시로만 정렬", "ok", 34)
    s.sub("내용이 같은 문항은 그룹으로 묶어 통째 승격", "ok", 34)
    s.gap(24)
    s.panel([("순서 섞기 + ID 변경 재실행", "불일치 0건", "ok"),
             ("동일 입력 재실행", "불일치 0건", "ok")])
    s.save(p)


def s06(p):
    s = Slide("자체 점검 도구", "5 / 검증")
    s.sub("python3 -m routerx.audit", "acc", 34)
    s.gap(16)
    s.panel([("ID·순서 불변성", "400문항 0건", "ok"),
             ("결정성", "300문항 0건", "ok"),
             ("엣지 케이스", "7종 통과", "ok"),
             ("실행 시간", "한도의 5%", "ok"),
             ("예산 여유", "17 – 35%", "ok")],
            head=("점검 항목", "결과"))
    s.line("10항목 전부 통과 · 단위 시험 21건 통과", "ok", 40, bold=True)
    s.save(p)


def s07(p):
    s = Slide("예산 초과를 구조로 차단", "6 / 설계 ②")
    s.line("추론 모델 출력 토큰 분포", "warn", 36, bold=True)
    s.panel([("중앙값", "1,570 토큰", "fg"),
             ("최댓값", "130,504 토큰", "red"),
             ("한 문항 최대 점유", "전체 비용의 26%", "red")])
    s.line("예측 기반 필터 — 실패", "dim", 36, bold=True)
    s.sub("사고 문항은 '예측은 작고 실제는 큰' 유형 → 필터 통과")
    s.gap(16)
    s.line("선택 건수 상한 — 유효", "ok", 42, bold=True)
    s.sub("예측 오차와 무관하게 노출 제한", "ok", 34)
    s.save(p)


def s08(p):
    s = Slide("측정에 근거한 판단", "7 / 검증과 되돌림")
    s.line("2,640문항 · 5-fold + 8-fold 교차검증", "acc", 38, bold=True)
    s.sub("양쪽에서 예산 초과 0건인 조합만 채택")
    s.gap(22)
    s.line("되돌린 것 둘", "warn", 38, bold=True)
    s.panel([("부스팅 트리", "단일 측정 최고", "CV 예산 실패", "red"),
             ("문장 임베딩", "0.425 → 0.372", "예측 악화", "red")],
            mid_x=0.72)
    s.line("점수 0.01  <  등급 0점 (최종의 30%)", "ok", 44, bold=True)
    s.save(p)


def s09(p):
    s = Slide("정리", "8 / 결과")
    s.panel([("공개 Dev 880문항", "0.7167", "ok"),
             ("교차검증 2,640문항", "0.6643", "ok"),
             ("공식 최강 baseline", "0.6954", "dim"),
             ("전부 경량 모델", "0.6193", "dim")],
            head=("측정", "최종 점수"))
    s.bullet("학습 scikit-learn · 실행 NumPy 하나", "fg", 38)
    s.bullet("두 경로 예측 차이 1.2e-15", "fg", 38)
    s.bullet("이미지 73MB · 등급당 4.2초 · Apache-2.0", "fg", 38)
    s.foot("github.com/leelang7/ossp-2026-llm-router-challenge")
    s.save(p)


SCENES = [
    ("s01", s01, "추론 비용은 LLM 서비스의 최대 지출 항목입니다. 그런데 트래픽 다수는 "
                 "경량 모델로도 충분합니다. 이 라우터는 프롬프트만 보고 세 모델 중 "
                 "하나를 고릅니다. 모델을 호출하거나 답변을 비교할 수 없고, 예산을 "
                 "넘기면 해당 등급은 0점입니다."),
    ("s02", s02, "공개 검증 자료 880문항을 프리미엄 등급으로 라우팅합니다. "
                 "4.2초, 한도의 5퍼센트입니다."),
    ("s03", s03, "공식 채점기 결과입니다. 최종 점수 영 점 칠일육칠, 세 등급 모두 "
                 "예산을 통과했습니다."),
    ("s04", s04, "직접 계산 특징 36개와 단어, 문자 티에프 아이디에프를 뽑습니다. "
                 "여기서 모델별 품질과 토큰 사용량을 함께 예측합니다. 라우팅 시점에 "
                 "토큰 수가 주어지지 않아 비용도 예측 대상입니다. 그다음 이득 대비 "
                 "비용이 큰 순서로 승격합니다."),
    ("s05", s05, "첫째, 순서 불변성입니다. 예산은 배치 전체 제약이라 문항을 독립 "
                 "처리할 수 없는데, 입력 순서에 의존하면 규칙 위반입니다. 동률을 "
                 "프롬프트 내용 해시로만 정렬해 해결했습니다."),
    ("s06", s06, "자체 점검 도구입니다. 순서 불변성부터 예산 여유까지 열 항목을 "
                 "한 번에 확인하고, 전부 통과했습니다."),
    ("s07", s07, "둘째, 예산 안전입니다. 추론 모델 출력 토큰은 중앙값 천오백에 최대 "
                 "십삼만, 한 문항이 전체 비용의 26퍼센트를 점유할 수 있습니다. 예측 "
                 "기반 필터는 실패했습니다. 사고 문항이 예측은 작고 실제는 큰 유형이라 "
                 "그대로 통과하기 때문입니다. 선택 건수 자체를 묶어 해결했습니다."),
    ("s08", s08, "2640문항을 5겹과 8겹으로 교차검증해 양쪽에서 예산 초과가 없는 "
                 "조합만 채택했습니다. 이 과정에서 둘을 되돌렸습니다. 부스팅 트리는 "
                 "단일 측정 최고점이었지만 교차검증에서 예산을 못 지켰고, 임베딩은 "
                 "예측이 오히려 나빠졌습니다. 점수 영 점 영일보다 등급 0점이 훨씬 "
                 "큽니다."),
    ("s09", s09, "학습은 사이킷런, 실행은 넘파이 하나입니다. 두 경로의 예측 차이는 "
                 "부동소수점 한계까지 일치합니다. 아파치 2.0으로 공개했습니다."),
]


async def speak(text: str, path: Path):
    import edge_tts
    await edge_tts.Communicate(text, "ko-KR-InJoonNeural", rate="+15%").save(str(path))


def seconds_of(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    segments, total = [], 0.0
    for name, render, narration in SCENES:
        png = OUT / f"{name}.png"
        render(png)
        mp3 = OUT / f"{name}.mp3"
        asyncio.run(speak(narration, mp3))
        dur = seconds_of(mp3) + 0.9
        total += dur
        seg = OUT / f"{name}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png),
            "-i", str(mp3), "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac",
            "-b:a", "160k", "-pix_fmt", "yuv420p", "-r", "12", "-crf", "20",
            "-t", f"{dur:.2f}", str(seg)], check=True)
        segments.append(seg)
        print(f"  {name}  {dur:5.1f}초")

    (OUT / "concat.txt").write_text(
        "".join("file '%s'\n" % s.name for s in segments), encoding="utf-8")
    final = OUT / "routerx_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", "concat.txt", "-c", "copy", final.name],
                   check=True, cwd=str(OUT))
    length = seconds_of(final)
    print(f"\n완성: {final}")
    print(f"  {int(length // 60)}분 {length % 60:04.1f}초 / 제한 3분   "
          f"{final.stat().st_size / 1e6:.1f}MB")
    if length > 180:
        print("  경고: 3분 초과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
