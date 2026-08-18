# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""K1 제한 방식 비교 — 건수 상한 vs 비용 총액 상한.

현재 premium은 예산의 70%만 쓰고 30%를 놀린다. 건수 상한(11%)이 병목이기 때문이다.
비용 총액으로 묶으면 '싼 K1은 많이, 비싼 K1은 적게' 쓰면서 남는 예산을 활용할 수 있다.
셋을 비교한다: 건수 상한 / 비용 상한 / 둘 다.
"""
from __future__ import annotations

import heapq
import sys

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MULT, TIERS, WEIGHT  # noqa: E402

C = np.load(r"d:\opensource\skt-router\lab\fold_cache.npz")
SCORE, COST, KEYS = C["score"], C["cost"], C["keys"]
GRID = np.round(np.arange(0.30, 1.601, 0.005), 4)[::-1]
HEAVY = 2


def select(pred_score, pred_cost, mult, safety, keys, k1_count=1.0, k1_cost=1.0):
    """예측 예산 안에서 승격. 추론 모델은 건수·비용 두 상한을 함께 적용한다."""
    n = len(pred_score)
    sel = np.zeros(n, dtype=np.int64)
    light_total = float(pred_cost[:, 0].sum())
    spent = light_total
    cap = light_total * max(1.0, mult * safety)
    headroom = cap - light_total
    count_budget = int(n * min(max(k1_count, 0.0), 1.0))
    cost_budget = headroom * k1_cost if k1_cost < 1.0 else float("inf")
    used_count, used_cost = 0, 0.0
    heap: list = []

    def push(i):
        cur = sel[i]
        for m in range(3):
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
        if m == HEAVY and cur != HEAVY:
            if used_count >= count_budget:
                continue
            extra_heavy = float(pred_cost[i, HEAVY] - pred_cost[i, 0])
            if used_cost + extra_heavy > cost_budget:
                continue
        if spent + dc <= cap:
            spent += dc
            if m == HEAVY and cur != HEAVY:
                used_count += 1
                used_cost += float(pred_cost[i, HEAVY] - pred_cost[i, 0])
            elif cur == HEAVY and m != HEAVY:
                used_count -= 1
                used_cost -= float(pred_cost[i, HEAVY] - pred_cost[i, 0])
            sel[i] = m
            push(i)
    return sel


def tier_eval(n_folds, tier, margin, k1_count=1.0, k1_cost=1.0):
    fold = C[f"fold{n_folds}"]
    s_te_all, c_te_all = C[f"s_te{n_folds}"], C[f"c_te{n_folds}"]
    pts, ratios, fails, k1s = [], [], 0, []
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        s_oof, c_oof = C[f"f{n_folds}_{f}_s_oof"], C[f"f{n_folds}_{f}_c_oof"]
        safety = 0.30
        for s in GRID:
            sel = select(s_oof, c_oof, MULT[tier], float(s), KEYS[tr], k1_count, k1_cost)
            r = COST[tr][np.arange(len(sel)), sel].sum() / COST[tr][:, 0].sum()
            if r <= MULT[tier] * margin:
                safety = float(s)
                break
        sel_te = select(s_te_all[te], c_te_all[te], MULT[tier], safety, KEYS[te], k1_count, k1_cost)
        n = len(sel_te)
        ratio = COST[te][np.arange(n), sel_te].sum() / COST[te][:, 0].sum()
        ok = ratio <= MULT[tier] + 1e-12
        pts.append(SCORE[te][np.arange(n), sel_te].mean() if ok else 0.0)
        ratios.append(ratio / MULT[tier])
        k1s.append((sel_te == HEAVY).mean())
        fails += 0 if ok else 1
    return np.array(pts), np.array(ratios), fails, float(np.mean(k1s))


if __name__ == "__main__":
    print("=== premium 등급: 제한 방식 비교 (예산 4.0, 가중치 0.3) ===")
    print(f"{'설정':44s} {'5fold':>18s} {'8fold':>18s}  실제K1")
    rows = []
    for tag, kc, kx, m in [
        ("건수 11% (현재)", 0.11, 1.0, 0.85),
        ("건수 15%", 0.15, 1.0, 0.85),
        ("건수 20%", 0.20, 1.0, 0.85),
        ("비용 30%", 1.0, 0.30, 0.85),
        ("비용 40%", 1.0, 0.40, 0.85),
        ("비용 50%", 1.0, 0.50, 0.85),
        ("비용 60%", 1.0, 0.60, 0.85),
        ("비용 70%", 1.0, 0.70, 0.85),
        ("건수20%+비용50%", 0.20, 0.50, 0.85),
        ("건수20%+비용60%", 0.20, 0.60, 0.85),
        ("건수30%+비용50%", 0.30, 0.50, 0.85),
        ("건수15%+비용40%", 0.15, 0.40, 0.85),
        ("건수20%+비용50% m=0.95", 0.20, 0.50, 0.95),
        ("건수20%+비용60% m=0.95", 0.20, 0.60, 0.95),
        ("건수30%+비용60% m=0.95", 0.30, 0.60, 0.95),
    ]:
        p5, r5, f5, k5 = tier_eval(5, "premium", m, kc, kx)
        p8, r8, f8, k8 = tier_eval(8, "premium", m, kc, kx)
        safe = (f5 + f8) == 0
        rows.append((0.5 * p5.mean() + 0.5 * p8.mean(), safe, tag, kc, kx, m))
        print(f"  {tag:42s} {p5.mean():.5f}/{f5}/{r5.max():.2f} {p8.mean():.5f}/{f8}/{r8.max():.2f}"
              f"  {k5:.1%}{'  SAFE' if safe else ''}")
    best = max([r for r in rows if r[1]] or rows, key=lambda r: r[0])
    print(f"\n  → premium 최적: {best[2]} (기대점수 {best[0]:.5f})")

    print("\n=== balanced 등급 비교 (예산 2.0) ===")
    for tag, kc, kx, m in [
        ("건수 1% (현재)", 0.01, 1.0, 0.80),
        ("건수 5%", 0.05, 1.0, 0.80),
        ("비용 20%", 1.0, 0.20, 0.80),
        ("비용 30%", 1.0, 0.30, 0.80),
        ("건수10%+비용30%", 0.10, 0.30, 0.80),
        ("건수10%+비용30% m=0.85", 0.10, 0.30, 0.85),
    ]:
        p5, r5, f5, k5 = tier_eval(5, "balanced", m, kc, kx)
        p8, r8, f8, k8 = tier_eval(8, "balanced", m, kc, kx)
        safe = (f5 + f8) == 0
        print(f"  {tag:42s} {p5.mean():.5f}/{f5}/{r5.max():.2f} {p8.mean():.5f}/{f8}/{r8.max():.2f}"
              f"  {k5:.1%}{'  SAFE' if safe else ''}")

    print("\n=== fast 등급 비교 (예산 1.25) ===")
    for tag, kc, kx, m in [
        ("건수 0% (현재)", 0.0, 1.0, 0.83),
        ("건수 0% m=0.86", 0.0, 1.0, 0.86),
        ("건수 0% m=0.90", 0.0, 1.0, 0.90),
        ("비용 10%", 1.0, 0.10, 0.86),
        ("건수2%+비용10%", 0.02, 0.10, 0.86),
    ]:
        p5, r5, f5, k5 = tier_eval(5, "fast", m, kc, kx)
        p8, r8, f8, k8 = tier_eval(8, "fast", m, kc, kx)
        safe = (f5 + f8) == 0
        print(f"  {tag:42s} {p5.mean():.5f}/{f5}/{r5.max():.2f} {p8.mean():.5f}/{f8}/{r8.max():.2f}"
              f"  {k5:.1%}{'  SAFE' if safe else ''}")
