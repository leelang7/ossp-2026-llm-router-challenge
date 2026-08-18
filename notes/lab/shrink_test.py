# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""d1 축소(shrinkage) 실험 — 예산이 남는 진짜 원인을 찾는다.

k1_cost 결과: 비용 총액 상한은 실패. premium 건수 11%가 이미 안전 한계.
그런데도 예산이 30% 남는다. 가설:
  light→ax31 승격 이득(d1)은 예측 상관이 0.01~0.10으로 사실상 무의미하다.
  그래서 많은 문항에서 예측상 dq<=0이 되어 승격 후보에서 빠지고, 예산이 남는다.
  d1 예측을 전역 평균(+0.081, 항상 양수)으로 대체하면 '비용 싼 순으로 최대한 승격'이
  되어 예산을 다 쓰게 된다. 실제 d1 평균이 양수이므로 기대 이득도 양수다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MULT, TIERS, WEIGHT  # noqa: E402
from routerx.policy import select_batch  # noqa: E402

C = np.load(r"d:\opensource\skt-router\lab\fold_cache.npz")
SCORE, COST, KEYS = C["score"], C["cost"], C["keys"]
GRID = np.round(np.arange(0.30, 2.001, 0.005), 4)[::-1]

D1_MEAN = float((SCORE[:, 1] - SCORE[:, 0]).mean())
D2_MEAN = float((SCORE[:, 2] - SCORE[:, 1]).mean())
print(f"전역 평균 이득: d1={D1_MEAN:+.4f} d2={D2_MEAN:+.4f}")


def shrink(S, lam1, lam2):
    """d1은 lam1, d2는 lam2 비율만 예측을 신뢰하고 나머지는 전역 평균으로 되돌린다."""
    base = S[:, 0]
    d1 = S[:, 1] - S[:, 0]
    d2 = S[:, 2] - S[:, 1]
    n1 = D1_MEAN + lam1 * (d1 - d1.mean())
    n2 = D2_MEAN + lam2 * (d2 - d2.mean())
    return np.clip(np.column_stack([base, base + n1, base + n1 + n2]), 0.0, 1.0)


def tier_eval(n_folds, tier, margin, k1_cap, lam1, lam2):
    fold = C[f"fold{n_folds}"]
    s_te_all, c_te_all = C[f"s_te{n_folds}"], C[f"c_te{n_folds}"]
    pts, ratios, fails, k1s = [], [], 0, []
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        s_oof = shrink(C[f"f{n_folds}_{f}_s_oof"], lam1, lam2)
        c_oof = C[f"f{n_folds}_{f}_c_oof"]
        safety = 0.30
        for s in GRID:
            sel = select_batch(s_oof, c_oof, MULT[tier], float(s), KEYS[tr], k1_cap)
            r = COST[tr][np.arange(len(sel)), sel].sum() / COST[tr][:, 0].sum()
            if r <= MULT[tier] * margin:
                safety = float(s)
                break
        sel_te = select_batch(shrink(s_te_all[te], lam1, lam2), c_te_all[te],
                              MULT[tier], safety, KEYS[te], k1_cap)
        n = len(sel_te)
        ratio = COST[te][np.arange(n), sel_te].sum() / COST[te][:, 0].sum()
        ok = ratio <= MULT[tier] + 1e-12
        pts.append(SCORE[te][np.arange(n), sel_te].mean() if ok else 0.0)
        ratios.append(ratio / MULT[tier])
        k1s.append((sel_te == 2).mean())
        fails += 0 if ok else 1
    return np.array(pts), np.array(ratios), fails, float(np.mean(k1s))


SETUP = {"fast": (0.0, [0.83, 0.86]), "balanced": (0.01, [0.80, 0.85]),
         "premium": (0.11, [0.85, 0.92])}

if __name__ == "__main__":
    best_of = {}
    for tier in TIERS:
        k1_cap, margins = SETUP[tier]
        print(f"\n===== {tier} (K1상한 {k1_cap:.0%}) =====")
        print(f"{'λ1':>5s} {'λ2':>5s} {'margin':>7s}   5fold 점수/실패/최대   8fold 점수/실패/최대")
        rows = []
        for lam1 in (0.0, 0.2, 0.5, 1.0):
            for lam2 in (1.0, 1.5):
                for m in margins:
                    p5, r5, f5, k5 = tier_eval(5, tier, m, k1_cap, lam1, lam2)
                    p8, r8, f8, k8 = tier_eval(8, tier, m, k1_cap, lam1, lam2)
                    exp = 0.5 * p5.mean() + 0.5 * p8.mean()
                    safe = (f5 + f8) == 0
                    rows.append((exp, safe, lam1, lam2, m, max(r5.max(), r8.max())))
                    print(f"  {lam1:.1f}  {lam2:.1f}   {m:.2f}   "
                          f"{p5.mean():.5f}/{f5}/{r5.max():.2f}   "
                          f"{p8.mean():.5f}/{f8}/{r8.max():.2f}{'  SAFE' if safe else ''}")
        safe_rows = [r for r in rows if r[1]]
        pick = max(safe_rows or rows, key=lambda r: r[0])
        best_of[tier] = pick
        print(f"  → 최적: λ1={pick[2]} λ2={pick[3]} margin={pick[4]:.2f} "
              f"기대점수={pick[0]:.5f} 최대사용률={pick[5]:.2f}")

    total = sum(WEIGHT[t] * best_of[t][0] for t in TIERS)
    print("\n===== 종합 =====")
    for t in TIERS:
        b = best_of[t]
        print(f"  {t:9s} λ1={b[2]} λ2={b[3]} margin={b[4]:.2f} 점수={b[0]:.5f}")
    print(f"  CV 기대점수 = {total:.6f}  (현재 정책 0.664872)")
