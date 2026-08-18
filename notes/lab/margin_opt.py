# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""등급별 마진 최적화 — 캐시된 fold 예측으로 초 단위 탐색.

목표: 5-fold(528문항)와 8-fold(330문항) 홀드아웃 모두에서 예산 초과 0건을
유지하면서 CV 점수를 최대화하는 등급별 마진을 찾는다.

예산 초과는 해당 등급 0점이므로 기대점수 관점에서 항상 보수 쪽이 유리하다.
  fast 0점 → 최종 -40%,  balanced/premium 0점 → 각 -30%
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MULT, TIERS, WEIGHT  # noqa: E402
from routerx.policy import select_batch  # noqa: E402

C = np.load(r"d:\opensource\skt-router\lab\fold_cache.npz")
SCORE, COST, KEYS = C["score"], C["cost"], C["keys"]
GRID = np.round(np.arange(0.30, 1.401, 0.005), 4)[::-1]


def tier_eval(n_folds: int, tier: str, margin: float):
    """해당 등급만 평가 → (fold별 점수, fold별 사용률, 실패수)."""
    fold = C[f"fold{n_folds}"]
    s_te_all, c_te_all = C[f"s_te{n_folds}"], C[f"c_te{n_folds}"]
    pts, ratios, fails = [], [], 0
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        s_oof, c_oof = C[f"f{n_folds}_{f}_s_oof"], C[f"f{n_folds}_{f}_c_oof"]
        safety = 0.30
        for s in GRID:
            sel = select_batch(s_oof, c_oof, MULT[tier], float(s), KEYS[tr])
            r = COST[tr][np.arange(len(sel)), sel].sum() / COST[tr][:, 0].sum()
            if r <= MULT[tier] * margin:
                safety = float(s)
                break
        sel_te = select_batch(s_te_all[te], c_te_all[te], MULT[tier], safety, KEYS[te])
        n = len(sel_te)
        ratio = COST[te][np.arange(n), sel_te].sum() / COST[te][:, 0].sum()
        ok = ratio <= MULT[tier] + 1e-12
        pts.append(SCORE[te][np.arange(n), sel_te].mean() if ok else 0.0)
        ratios.append(ratio / MULT[tier])
        fails += 0 if ok else 1
    return np.array(pts), np.array(ratios), fails


if __name__ == "__main__":
    margins = np.round(np.arange(0.70, 0.981, 0.01), 3)
    best = {}
    for tier in TIERS:
        print(f"\n===== {tier} (예산 {MULT[tier]}, 가중치 {WEIGHT[tier]}) =====")
        print(f"{'margin':>7s} {'5fold점수/실패/최대사용':>28s} {'8fold점수/실패/최대사용':>28s}")
        rows = []
        for m in margins:
            p5, r5, f5 = tier_eval(5, tier, float(m))
            p8, r8, f8 = tier_eval(8, tier, float(m))
            rows.append((m, p5.mean(), f5, r5.max(), p8.mean(), f8, r8.max()))
            mark = "  <-- 양쪽 실패 0" if (f5 == 0 and f8 == 0) else ""
            print(f"  {m:.2f}  {p5.mean():.5f}/{f5}/{r5.max():.3f}      "
                  f"{p8.mean():.5f}/{f8}/{r8.max():.3f}{mark}")
        safe = [r for r in rows if r[2] == 0 and r[5] == 0]
        if safe:
            pick = max(safe, key=lambda r: 0.5 * r[1] + 0.5 * r[4])
            best[tier] = pick[0]
            print(f"  → 선택: margin={pick[0]:.2f} (5fold {pick[1]:.5f}, 8fold {pick[4]:.5f})")
        else:
            best[tier] = 0.70
            print("  → 실패 0 구간 없음. 하한 사용")

    print(f"\n===== 최종 등급별 마진 = {best} =====")
    total5 = total8 = 0.0
    for tier in TIERS:
        p5, r5, f5 = tier_eval(5, tier, best[tier])
        p8, r8, f8 = tier_eval(8, tier, best[tier])
        total5 += WEIGHT[tier] * p5.mean()
        total8 += WEIGHT[tier] * p8.mean()
        print(f"  {tier:9s} margin={best[tier]:.2f} 5fold={p5.mean():.5f}(max {r5.max():.3f}) "
              f"8fold={p8.mean():.5f}(max {r8.max():.3f})")
    print(f"  종합 CV: 5-fold={total5:.6f}  8-fold={total8:.6f}")
