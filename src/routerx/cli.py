# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""router-run 인터페이스 — 한 등급의 선택 결과 JSON 하나를 만든다.

    router-run --input inputs.json --tier fast --output submission.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from ossp_router.protocol import (
    TIERS,
    Decision,
    ProtocolError,
    Submission,
    dumps_json,
    load_input,
    submission_to_dict,
)

from .router import ARTIFACT_NAME, Artifact, route


def write_atomic(path: Path, payload: str) -> None:
    """같은 디렉터리 임시 파일에 완전히 쓴 뒤 원자적으로 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        # 입력 규격이 고립 서로게이트를 허용하므로 출력에서도 막히지 않게 한다.
        temporary.write_text(payload, encoding="utf-8", errors="surrogatepass")
        os.chmod(temporary, 0o644)
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="프롬프트 내용만으로 모델을 선택합니다.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=tuple(TIERS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact_path = args.artifact or (Path(__file__).resolve().parent / ARTIFACT_NAME)
        artifact = Artifact(artifact_path)
        inputs = load_input(args.input)
        chosen = route(inputs.episodes, artifact, args.tier)
        submission = Submission(
            schema_version=inputs.schema_version,
            challenge_id=inputs.challenge_id,
            policy_id=artifact.policy_id,
            split=inputs.split,
            tier=args.tier,
            decisions=tuple(
                Decision(episode.episode_id, model_id)
                for episode, model_id in zip(inputs.episodes, chosen)
            ),
        )
        write_atomic(args.output, dumps_json(submission_to_dict(submission)))
    except Exception as exc:      # noqa: BLE001 - 어떤 실패든 규격이 정한 종료 코드로 알린다
        print(f"오류: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 선택 결과를 {args.output}에 기록했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
