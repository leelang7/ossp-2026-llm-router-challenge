# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""K1 사용 비율 제한 실험 — 기대점수 관점의 재검토.

지금까지: cap 규칙 무효, 분위수 회귀 무효(오히려 악화), 마진만이 유효.
premium이 가장 자주 실패하고 그 원인은 전적으로 K1의 heavy tail이다.

기대점수로 따지면
    K1 적극 사용:  높은 점수 × (1 - 실패확률)
    K1 제한:       낮은 점수 × 1.0
실패확률이 15%만 넘어도 후자가 유리할 수 있다. 실제로 그런지 측정한다.
"""
from __future__ import annotations

import heapq
import sys

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MULT, TIERS, WEIGHT  # noqa: E402

C = np.load(r"d:\opensource\skt-router\lab\fold_cache.npz")
SCORE, COST, KEYS = C["score"], C["cost"], C["keys"]
GRID = np.round(np.arange(0.30, 1.401, 0.005), 4)[::-1]
N_M = 3


def select_limited(pred_score, pred_cost, mult, safety, keys, max_k1_ratio=1.0):
    """예산 안에서 승격하되 K1 선택 개수를 상한으로 묶는다."""
    n = len(pred_score)
    sel = np.zeros(n, dtype=np.int64)
    light_total = float(pred_cost[:, 0].sum())
    spent = light_total
    cap = light_total * max(1.0, mult * safety)
    k1_budget = int(np.floor(n * max_k1_ratio))
    k1_used = 0
    heap: list = []

    def push(i):
        cur = sel[i]
        for m in range(N_M):
            if m == cur:
                continue
            dq = pred_score[i, m] - pred_score[i, cur]
            dc = pred_cost[i, m] - pred_cost[i, cur]
            if dq <= 0:
                continue
            heapq.heappush(heap, (-1e18 if dc <= 0 else -dq / dc, float(keys[i]), m, i))

    for i in range(n):
        push(i)
    while heap:
        _, _, m, i = heapq.heappop(heap)
        cur = sel[i]
        dq = pred_score[i, m] - pred_score[i, cur]
        dc = pred_cost[i, m] - pred_cost[i, cur]
        if dq <= 0:
            continue
        if m == 2 and cur != 2 and k1_used >= k1_budget:
            continue
        if spent + dc <= cap:
            spent += dc
            if m == 2 and cur != 2:
                k1_used += 1
            elif cur == 2 and m != 2:
                k1_used -= 1
            sel[i] = m
            push(i)
    return sel


def tier_eval(n_folds, tier, margin, max_k1=1.0):
    fold = C[f"fold{n_folds}"]
    s_te_all, c_te_all = C[f"s_te{n_folds}"], C[f"c_te{n_folds}"]
    pts, ratios, fails, k1s = [], [], 0, []
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        s_oof, c_oof = C[f"f{n_folds}_{f}_s_oof"], C[f"f{n_folds}_{f}_c_oof"]
        safety = 0.30
        for s in GRID:
            sel = select_limited(s_oof, c_oof, MULT[tier], float(s), KEYS[tr], max_k1)
            r = COST[tr][np.arange(len(sel)), sel].sum() / COST[tr][:, 0].sum()
            if r <= MULT[tier] * margin:
                safety = float(s)
                break
        sel_te = select_limited(s_te_all[te], c_te_all[te], MULT[tier], safety, KEYS[te], max_k1)
        n = len(sel_te)
        ratio = COST[te][np.arange(n), sel_te].sum() / COST[te][:, 0].sum()
        ok = ratio <= MULT[tier] + 1e-12
        pts.append(SCORE[te][np.arange(n), sel_te].mean() if ok else 0.0)
        ratios.append(ratio / MULT[tier])
        k1s.append((sel_te == 2).mean())
        fails += 0 if ok else 1
    return np.array(pts), np.array(ratios), fails, float(np.mean(k1s))


if __name__ == "__main__":
    print("K1 상한별 등급 성적 (점수는 실패 시 0점 반영한 기대값)")
    for tier in TIERS:
        print(f"\n===== {tier} (예산 {MULT[tier]}, 가중치 {WEIGHT[tier]}) =====")
        print(f"{'K1상한':>7s} {'margin':>7s}   5fold: 점수/실패/최대사용   8fold: 점수/실패/최대사용")
        best = None
        for max_k1 in (0.0, 0.03, 0.05, 0.08, 0.12, 0.20, 1.0):
            for m in (0.95, 0.90, 0.85, 0.80):
                p5, r5, f5, k5 = tier_eval(5, tier, m, max_k1)
                p8, r8, f8, k8 = tier_eval(8, tier, m, max_k1)
                exp = 0.5 * p5.mean() + 0.5 * p8.mean()
                safe = (f5 == 0 and f8 == 0)
                mark = "  SAFE" if safe else ""
                print(f"  {max_k1:5.2f}  {m:.2f}   {p5.mean():.5f}/{f5}/{r5.max():.3f}   "
                      f"{p8.mean():.5f}/{f8}/{r8.max():.3f}   K1={k5:.1%}{mark}")
                if best is None or exp > best[0]:
                    best = (exp, max_k1, m, safe, f5 + f8)
        print(f"  → 기대값 최대: K1상한={best[1]:.2f} margin={best[2]:.2f} "
              f"기대점수={best[0]:.5f} {'(실패 0)' if best[3] else f'(실패 {best[4]}건)'}")
