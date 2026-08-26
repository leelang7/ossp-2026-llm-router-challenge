# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""제출 패키지(zip) 생성 — 공지 39호 '출품작 제출 가이드' 기준.

홈페이지에 올리는 것은 zip 하나다. 그 안에 들어가야 하는 것은 세 가지이며,
셋째는 해당 시에만 넣는다.

  ① 결과보고서 원본파일(한글 또는 워드)
  ② 결과보고서 PDF 변환파일
  ③ 출품작 중복수혜 여부 확인서(해당 시)

기타 산출물은 별도 파일로 낼 수 없고 보고서나 저장소에 포함해야 한다.
자리표시자가 남아 있으면 만들지 않고 멈춘다 — 시연영상 URL이 비어 있는 채로
제출되는 사고를 막기 위해서다.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

LAB = Path(r"d:\opensource\skt-router")
NAME = "2026 오픈소스 개발자대회 결과보고서_830(트리아지)"
DOCX = LAB / f"{NAME}.docx"
PDF = LAB / f"{NAME}.pdf"
EXTRA = LAB / "출품작 중복수혜 여부 확인서.pdf"   # 해당 시에만
OUT = LAB / "제출_트리아지.zip"
PLACEHOLDER = "유튜브 URL"


def check_placeholder() -> list[str]:
    """보고서에 채우지 않은 자리가 남아 있는지 본다."""
    from docx import Document
    left = []
    doc = Document(DOCX)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if PLACEHOLDER in cell.text:
                    left.append(" ".join(cell.text.split()))
    return left


def main() -> int:
    missing = [p.name for p in (DOCX, PDF) if not p.exists()]
    if missing:
        print("없는 파일: " + ", ".join(missing))
        return 1

    left = check_placeholder()
    if left:
        print("아직 채우지 않은 자리가 있습니다. 제출 패키지를 만들지 않습니다.")
        for text in dict.fromkeys(left):
            print(f"  - {text}")
        print("\nlab/fill_report.py의 VIDEO에 유튜브 링크를 넣고 다시 만드십시오.")
        return 1

    items = [DOCX, PDF] + ([EXTRA] if EXTRA.exists() else [])
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in items:
            zf.write(path, path.name)
            print(f"  담음: {path.name}  ({path.stat().st_size / 1024:.0f}KB)")
    if not EXTRA.exists():
        print("  중복수혜 확인서: 해당 없음(미포함)")
    print(f"\n{OUT}  {OUT.stat().st_size / 1024:.0f}KB")
    print("홈페이지 > 접수 및 조회 > 출품작 제출 > 제출하기 에서 올린 뒤")
    print("상태가 '제출 완료'로 바뀌었는지, 안내 메일이 왔는지 반드시 확인하십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
