# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""제출 전 자체 점검 — 규칙 준수·성능·예산 여유를 한 번에 확인한다.

    python3 -m routerx.audit --input data/materialized/dev/inputs.json \
        --outcomes data/dev/outcomes.json

공식 self-check가 확인하지 않는 항목까지 본다.
  · 문항 ID·입력 순서를 바꿔도 같은 선택이 나오는가 (운영자 감사 항목)
  · 같은 입력을 두 번 돌렸을 때 결과가 같은가 (결정성)
  · 등급별 실행 시간이 한도(90초) 대비 얼마나 여유가 있는가
  · 등급별 예산을 얼마나 남기고 있는가 (초과 시 그 등급 0점)
  · 엣지 케이스(빈 프롬프트, 초장문, 어휘 밖 문서)에서 죽지 않는가
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Episode,
    Message,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

from .router import ARTIFACT_NAME, Artifact, route

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    numbers: Dict[str, object] = field(default_factory=dict)


def _fmt(check: Check) -> str:
    mark = {PASS: "[ OK ]", FAIL: "[FAIL]", WARN: "[WARN]"}[check.status]
    return f"  {mark} {check.name}: {check.detail}"


def check_order_invariance(episodes: Sequence, artifact: Artifact,
                           sample: int = 400, seed: int = 12345) -> Check:
    """문항 ID와 입력 순서를 바꿔도 선택이 같아야 한다(운영자 감사 항목)."""
    subset = list(episodes[:sample])
    base = route(subset, artifact, "premium")
    baseline = {ep.episode_id: model for ep, model in zip(subset, base)}

    shuffled = list(subset)
    random.Random(seed).shuffle(shuffled)
    renamed = [
        Episode(episode_id=f"audit-{index:04d}", prompt=ep.prompt, messages=ep.messages)
        for index, ep in enumerate(shuffled)
    ]
    after = route(renamed, artifact, "premium")
    mismatch = sum(
        1 for ep, model in zip(shuffled, after) if baseline[ep.episode_id] != model
    )
    status = PASS if mismatch == 0 else FAIL
    return Check("ID·순서 불변성", status,
                 f"{len(subset)}문항 중 불일치 {mismatch}건",
                 {"episodes": len(subset), "mismatch": mismatch})


def check_determinism(episodes: Sequence, artifact: Artifact, sample: int = 300) -> Check:
    subset = list(episodes[:sample])
    first = route(subset, artifact, "balanced")
    second = route(subset, artifact, "balanced")
    mismatch = sum(1 for a, b in zip(first, second) if a != b)
    return Check("결정성(동일 입력 재실행)", PASS if mismatch == 0 else FAIL,
                 f"{len(subset)}문항 중 불일치 {mismatch}건", {"mismatch": mismatch})


def check_edge_cases(artifact: Artifact) -> Check:
    cases = {
        "한 글자": Episode("e1", prompt="?", messages=None),
        "공백만": Episode("e2", prompt="   ", messages=None),
        "초장문": Episode("e3", prompt="가나다 " * 20000, messages=None),
        "어휘 밖": Episode("e4", prompt="ϴϵϷϸϹϺ ⑨⑩ ｱｲｳ", messages=None),
        "이모지": Episode("e5", prompt="🚀🛰️🌏 계산해줘 2+2", messages=None),
        "메시지 1개": Episode("e6", prompt=None,
                          messages=(Message(role="user", content="안녕"),)),
        "긴 대화": Episode("e7", prompt=None, messages=tuple(
            Message(role="user" if i % 2 == 0 else "assistant", content=f"턴 {i} " * 50)
            for i in range(40))),
    }
    failures = []
    for name, episode in cases.items():
        try:
            picked = route([episode], artifact, "fast")
            if len(picked) != 1 or picked[0] not in MODEL_IDS:
                failures.append(f"{name}(잘못된 출력)")
        except Exception as exc:                     # noqa: BLE001 - 진단 목적
            failures.append(f"{name}({type(exc).__name__}: {exc})")
    status = PASS if not failures else FAIL
    detail = f"{len(cases)}종 통과" if not failures else "; ".join(failures)
    return Check("엣지 케이스", status, detail, {"cases": len(cases)})


def check_latency(episodes: Sequence, artifact: Artifact, limit: float = 90.0) -> List[Check]:
    checks = []
    for tier in TIERS:
        started = time.perf_counter()
        route(list(episodes), artifact, tier)
        elapsed = time.perf_counter() - started
        ratio = elapsed / limit
        status = PASS if ratio < 0.5 else (WARN if ratio < 0.9 else FAIL)
        checks.append(Check(
            f"실행 시간({tier})", status,
            f"{elapsed:.1f}초 / 한도 {limit:.0f}초 = {ratio:.0%} "
            f"({len(episodes)}문항, {elapsed/max(len(episodes),1)*1000:.1f}ms/문항)",
            {"seconds": round(elapsed, 3), "ratio": round(ratio, 4)}))
    return checks


def check_budget(episodes: Sequence, artifact: Artifact, outcomes_path: Path) -> List[Check]:
    """실제 결과가 있을 때만 예산 사용률과 점수를 계산한다."""
    policy = load_bundled_policy()
    outcomes = load_outcomes(outcomes_path)
    index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    unit = Decimal(policy.token_unit)

    def cost_of(episode_id: str, model_id: str) -> Decimal:
        row = index[(episode_id, model_id)]
        rates = policy.models[model_id]
        return (rates.fixed_cost
                + Decimal(row.input_tokens) * rates.input_token_rate / unit
                + Decimal(row.output_tokens) * rates.output_token_rate / unit)

    light_total = sum((cost_of(ep.episode_id, policy.light_model_id) for ep in episodes),
                      Decimal(0))
    checks, weighted = [], Decimal(0)
    for tier in TIERS:
        picked = route(list(episodes), artifact, tier)
        used = sum((cost_of(ep.episode_id, m) for ep, m in zip(episodes, picked)), Decimal(0))
        ratio = used / light_total
        limit = Decimal(policy.tiers[tier].budget_multiplier)
        share = ratio / limit
        quality = sum((Decimal(index[(ep.episode_id, m)].score)
                       for ep, m in zip(episodes, picked)), Decimal(0)) / len(episodes)
        passed = ratio <= limit
        weighted += Decimal(policy.tiers[tier].weight) * (quality if passed else Decimal(0))
        heavy = sum(1 for m in picked if m == MODEL_IDS[2])
        if not passed:
            status, note = FAIL, "예산 초과 → 이 등급 0점"
        elif share > Decimal("0.97"):
            status, note = WARN, "한도에 근접(채점셋 분포가 나빠지면 0점 위험)"
        else:
            status, note = PASS, "여유 있음"
        checks.append(Check(
            f"예산({tier})", status,
            f"사용 {float(ratio):.3f}/{float(limit):.2f} = 한도의 {float(share):.1%}, "
            f"점수 {float(quality):.4f}, 추론모델 {heavy}건({heavy/len(episodes):.1%}) — {note}",
            {"ratio": float(ratio), "share": float(share), "quality": float(quality)}))
    checks.append(Check("가중 최종 점수", PASS, f"{float(weighted):.6f}",
                        {"final_score": float(weighted)}))
    return checks


def run_audit(input_path: Path, artifact_path: Path,
              outcomes_path: Optional[Path]) -> Dict[str, object]:
    artifact = Artifact(artifact_path)
    episodes = load_input(input_path).episodes
    print(f"라우터 자체 점검 — 문항 {len(episodes)}개, 아티팩트 {artifact_path.name}")
    print(f"  설정: {json.dumps(artifact.meta, ensure_ascii=False)}")
    print(f"  안전계수: {artifact.safety}")
    print(f"  추론모델 상한: {artifact.k1_cap}\n")

    checks: List[Check] = [
        check_order_invariance(episodes, artifact),
        check_determinism(episodes, artifact),
        check_edge_cases(artifact),
    ]
    checks.extend(check_latency(episodes, artifact))
    if outcomes_path is not None:
        checks.extend(check_budget(episodes, artifact, outcomes_path))

    for check in checks:
        print(_fmt(check))
    failed = [c for c in checks if c.status == FAIL]
    warned = [c for c in checks if c.status == WARN]
    print(f"\n결과: 통과 {len(checks)-len(failed)-len(warned)} / 경고 {len(warned)} / 실패 {len(failed)}")
    if failed:
        print("실패 항목을 고치기 전에는 제출하지 마십시오.")
    return {
        "checks": [{"name": c.name, "status": c.status, "detail": c.detail, **c.numbers}
                   for c in checks],
        "failed": len(failed), "warned": len(warned),
    }


def _force_utf8_stdout() -> None:
    """Windows 기본 콘솔 인코딩(cp949)에서도 보고서가 깨지지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="제출 전 자체 점검을 수행합니다.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, default=None,
                        help="공개 outcome. 주면 예산 사용률과 점수까지 계산한다")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact_path = args.artifact or (Path(__file__).resolve().parent / ARTIFACT_NAME)
    try:
        result = run_audit(args.input, artifact_path, args.outcomes)
    except (OSError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"보고서: {args.report}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
