# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""문항별 추론모델 비용 상한(heavy_item_cap) 탐색.

감사 지적: 건수 상한만으로는 비용이 묶이지 않는다. 11%를 가장 비싼 문항으로
채우면 실제 비율이 한도의 3배를 넘을 수 있고, 단일 문항이 경량 총비용의 78%를
쓴 사례도 있다. 예측 비용 기준 개별 상한을 더해 최악의 경우를 잘라낸다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MULT, TIERS, WEIGHT  # noqa: E402
from routerx.policy import select_batch  # noqa: E402

C = np.load(r"d:\opensource\skt-router\lab\fold_cache.npz")
SCORE, COST, KEYS = C["score"], C["cost"], C["keys"]
GRID = np.round(np.arange(0.30, 1.001, 0.005), 4)[::-1]

print("=== 참고: 실제 K1 비용의 경량 총비용 대비 분포 ===")
rel = COST[:, 2] / COST[:, 0].sum()
for q in (0.5, 0.9, 0.99, 0.999, 1.0):
    print(f"  분위 {q:<6} {np.quantile(rel, q):.4%}")


def tier_eval(n_folds, tier, margin, k1_cap, item_cap):
    fold = C[f"fold{n_folds}"]
    s_te_all, c_te_all = C[f"s_te{n_folds}"], C[f"c_te{n_folds}"]
    pts, ratios, fails, k1s = [], [], 0, []
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        s_oof, c_oof = C[f"f{n_folds}_{f}_s_oof"], C[f"f{n_folds}_{f}_c_oof"]
        safety = 0.30
        for s in GRID:
            sel = select_batch(s_oof, c_oof, MULT[tier], float(s), KEYS[tr], k1_cap, item_cap)
            r = COST[tr][np.arange(len(sel)), sel].sum() / COST[tr][:, 0].sum()
            if r <= MULT[tier] * margin:
                safety = float(s)
                break
        sel_te = select_batch(s_te_all[te], c_te_all[te], MULT[tier], safety,
                              KEYS[te], k1_cap, item_cap)
        n = len(sel_te)
        ratio = COST[te][np.arange(n), sel_te].sum() / COST[te][:, 0].sum()
        ok = ratio <= MULT[tier] + 1e-12
        pts.append(SCORE[te][np.arange(n), sel_te].mean() if ok else 0.0)
        ratios.append(ratio / MULT[tier])
        k1s.append((sel_te == 2).mean())
        fails += 0 if ok else 1
    return np.array(pts), np.array(ratios), fails, float(np.mean(k1s))


if __name__ == "__main__":
    print("\n=== premium: 개별 상한 스윕 (건수 11%, margin 0.85 고정) ===")
    print(f"{'item_cap':>10s}   5fold 점수/실패/최대   8fold 점수/실패/최대   K1")
    rows = []
    for item_cap in (1.0, 0.05, 0.03, 0.02, 0.01, 0.005):
        p5, r5, f5, k5 = tier_eval(5, "premium", 0.85, 0.11, item_cap)
        p8, r8, f8, k8 = tier_eval(8, "premium", 0.85, 0.11, item_cap)
        safe = (f5 + f8) == 0
        rows.append((0.5 * p5.mean() + 0.5 * p8.mean(), safe, item_cap))
        label = "없음" if item_cap >= 1.0 else f"{item_cap:.1%}"
        print(f"  {label:>10s}   {p5.mean():.5f}/{f5}/{r5.max():.2f}   "
              f"{p8.mean():.5f}/{f8}/{r8.max():.2f}   {k5:.1%}{'  SAFE' if safe else ''}")
    print("\n=== balanced: 개별 상한 스윕 (건수 1%, margin 0.80) ===")
    for item_cap in (1.0, 0.03, 0.02, 0.01):
        p5, r5, f5, k5 = tier_eval(5, "balanced", 0.80, 0.01, item_cap)
        p8, r8, f8, k8 = tier_eval(8, "balanced", 0.80, 0.01, item_cap)
        safe = (f5 + f8) == 0
        label = "없음" if item_cap >= 1.0 else f"{item_cap:.1%}"
        print(f"  {label:>10s}   {p5.mean():.5f}/{f5}/{r5.max():.2f}   "
              f"{p8.mean():.5f}/{f8}/{r8.max():.2f}   {k5:.1%}{'  SAFE' if safe else ''}")
    best = max([r for r in rows if r[1]] or rows, key=lambda r: r[0])
    print(f"\n  → premium 권장 item_cap = {best[2]} (기대점수 {best[0]:.5f})")
