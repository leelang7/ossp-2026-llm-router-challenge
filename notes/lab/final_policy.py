# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""최종 정책 검증 — GBM 비용 예측 기준으로 등급별 (마진, K1 상한)을 확정한다.

Dev 단일 측정으로 고른 값(fast 0.88 / balanced 0.93 / premium 0.95 + K1 20%)이
실제로 안전한지 5-fold·8-fold 양쪽에서 확인하고, 필요하면 되돌린다.
예산 초과는 해당 등급 0점이므로 실패가 하나라도 나면 채택하지 않는다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MULT, TIERS, WEIGHT  # noqa: E402
from routerx.policy import select_batch  # noqa: E402

C = np.load(r"d:\opensource\skt-router\lab\fold_cache2.npz")
SCORE, COST, KEYS = C["score"], C["cost"], C["keys"]
GRID = np.round(np.arange(0.30, 1.001, 0.005), 4)[::-1]
ITEM_CAP = 0.05


def tier_eval(n_folds, tier, margin, k1_cap):
    fold = C[f"fold{n_folds}"]
    s_te_all, c_te_all = C[f"s_te{n_folds}"], C[f"c_te{n_folds}"]
    pts, ratios, fails, k1s = [], [], 0, []
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        s_oof, c_oof = C[f"f{n_folds}_{f}_s_oof"], C[f"f{n_folds}_{f}_c_oof"]
        safety = 0.30
        for s in GRID:
            sel = select_batch(s_oof, c_oof, MULT[tier], float(s), KEYS[tr], k1_cap, ITEM_CAP)
            r = COST[tr][np.arange(len(sel)), sel].sum() / COST[tr][:, 0].sum()
            if r <= MULT[tier] * margin:
                safety = float(s)
                break
        sel_te = select_batch(s_te_all[te], c_te_all[te], MULT[tier], safety,
                              KEYS[te], k1_cap, ITEM_CAP)
        n = len(sel_te)
        ratio = COST[te][np.arange(n), sel_te].sum() / COST[te][:, 0].sum()
        ok = ratio <= MULT[tier] + 1e-12
        pts.append(SCORE[te][np.arange(n), sel_te].mean() if ok else 0.0)
        ratios.append(ratio / MULT[tier])
        k1s.append((sel_te == 2).mean())
        fails += 0 if ok else 1
    return np.array(pts), np.array(ratios), fails, float(np.mean(k1s))


SPACE = {
    "fast": ([0.0], [0.80, 0.84, 0.88, 0.92]),
    "balanced": ([0.01, 0.03], [0.85, 0.89, 0.93, 0.96]),
    "premium": ([0.11, 0.15, 0.20, 0.25], [0.85, 0.90, 0.95, 0.99]),
}

if __name__ == "__main__":
    best = {}
    for tier in TIERS:
        caps, margins = SPACE[tier]
        print(f"\n===== {tier} (예산 {MULT[tier]}, 가중치 {WEIGHT[tier]}) =====")
        print(f"{'K1상한':>7s} {'margin':>7s}   5fold 점수/실패/최대   8fold 점수/실패/최대   실제K1")
        rows = []
        for cap in caps:
            for m in margins:
                p5, r5, f5, k5 = tier_eval(5, tier, m, cap)
                p8, r8, f8, k8 = tier_eval(8, tier, m, cap)
                exp = 0.5 * p5.mean() + 0.5 * p8.mean()
                safe = (f5 + f8) == 0
                rows.append((exp, safe, cap, m, max(r5.max(), r8.max())))
                print(f"  {cap:5.2f}  {m:.2f}   {p5.mean():.5f}/{f5}/{r5.max():.2f}   "
                      f"{p8.mean():.5f}/{f8}/{r8.max():.2f}   {k5:.1%}"
                      f"{'  SAFE' if safe else ''}")
        safe_rows = [r for r in rows if r[1]]
        pick = max(safe_rows or rows, key=lambda r: r[0])
        best[tier] = pick
        print(f"  → 채택: K1상한={pick[2]:.2f} margin={pick[3]:.2f} "
              f"기대점수={pick[0]:.5f} 최대사용률={pick[4]:.2f}"
              f"{'' if pick[1] else '  (실패 있음 — 재검토 필요)'}")

    total = sum(WEIGHT[t] * best[t][0] for t in TIERS)
    print("\n===== 최종 권고 =====")
    args = []
    for t in TIERS:
        b = best[t]
        print(f"  {t:9s} K1상한={b[2]:.2f} margin={b[3]:.2f} 점수={b[0]:.5f} 최대사용률={b[4]:.2f}")
        args.append(f"--tier-margin {t}={b[3]:.2f} --tier-k1-cap {t}={b[2]:.2f}")
    print(f"  CV 기대점수 = {total:.6f}")
    print("\n  학습 인자:")
    print("   " + " ".join(args) + f" --k1-item-cap {ITEM_CAP}")
    print("\n  (참고) 이전 구성 CV 0.664872 / Dev 단일 측정으로 고른 값은 "
          "fast 0.88·balanced 0.93·premium 0.95 + K1 20%")
