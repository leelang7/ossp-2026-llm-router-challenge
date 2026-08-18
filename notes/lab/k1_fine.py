# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""K1 상한 세밀 탐색 — 등급별 최적 (K1 상한, 마진) 확정.

k1_limit 결과: K1 사용을 상한으로 묶으면 안전성과 점수를 동시에 얻는다.
  premium: K1 8% 상한 + margin 0.95 → 실패 0, 점수 0.697, 예산 사용률 0.69~0.80
  이는 K1 고비용 구간이 이득/비용도 최저였다는 exp8 관찰과 일치한다.
여기서는 경계를 좁혀 등급별 최적점을 정한다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MULT, TIERS, WEIGHT  # noqa: E402
from k1_limit import tier_eval  # noqa: E402

SPACE = {
    "fast": (np.arange(0.0, 0.061, 0.01), np.arange(0.80, 0.941, 0.02)),
    "balanced": (np.arange(0.0, 0.101, 0.01), np.arange(0.76, 0.921, 0.02)),
    "premium": (np.arange(0.04, 0.181, 0.01), np.arange(0.85, 1.001, 0.03)),
}

if __name__ == "__main__":
    best = {}
    for tier in TIERS:
        k1s, margins = SPACE[tier]
        rows = []
        for k1 in k1s:
            for m in margins:
                p5, r5, f5, used5 = tier_eval(5, tier, float(m), float(k1))
                p8, r8, f8, used8 = tier_eval(8, tier, float(m), float(k1))
                exp = 0.5 * p5.mean() + 0.5 * p8.mean()
                worst = max(r5.max(), r8.max())
                rows.append((exp, f5 + f8, float(k1), float(m), worst, used5))
        safe = [r for r in rows if r[1] == 0]
        pool = safe if safe else rows
        pick = max(pool, key=lambda r: r[0])
        best[tier] = {"k1": pick[2], "margin": pick[3], "exp": pick[0],
                      "worst_usage": pick[4], "k1_used": pick[5]}
        print(f"\n===== {tier} =====")
        top = sorted(pool, key=lambda r: -r[0])[:6]
        for exp, fails, k1, m, worst, used in top:
            print(f"  K1상한={k1:.2f} margin={m:.2f}  기대점수={exp:.5f}  "
                  f"최대사용률={worst:.3f}  실제K1={used:.1%}  실패={fails}")
        print(f"  → 선택: K1상한={pick[2]:.2f} margin={pick[3]:.2f} "
              f"기대점수={pick[0]:.5f} 최대사용률={pick[4]:.3f}")

    total = sum(WEIGHT[t] * best[t]["exp"] for t in TIERS)
    print("\n===== 최종 정책 =====")
    for t in TIERS:
        b = best[t]
        print(f"  {t:9s} K1상한={b['k1']:.2f} margin={b['margin']:.2f} "
              f"점수={b['exp']:.5f} 최대사용률={b['worst_usage']:.3f}(한도 대비) 실제K1={b['k1_used']:.1%}")
    print(f"  종합 CV 기대점수 = {total:.6f}")
    print(f"  (참고) 기존 무제한+margin0.80 CV = 0.614924, 공식 hash-regex Dev = 0.695369")
    import json
    with open(r"d:\opensource\skt-router\lab\best_policy.json", "w", encoding="utf-8") as fh:
        json.dump({t: {k: float(v) for k, v in best[t].items()} for t in TIERS}, fh,
                  ensure_ascii=False, indent=2)
    print("  저장: lab/best_policy.json")
