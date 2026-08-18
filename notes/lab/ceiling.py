# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""도달 가능 상한 분석 — 예측을 얼마나 개선해야 얼마를 얻는가.

완전정보 오라클은 0.80이지만, light↔ax31 이득(d1)은 생성 2~4회 평균의 흔들림이라
원리적으로 예측이 어렵다. 그렇다면 'd1은 끝까지 모른 채 d2만 완벽히 아는' 라우터가
현실적 상한이다. 이 값과 현재 점수의 간격이 예측 개선으로 얻을 수 있는 최대치다.
"""
from __future__ import annotations

import heapq
import sys

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MULT, TIERS, WEIGHT  # noqa: E402

C = np.load(r"d:\opensource\skt-router\lab\fold_cache.npz")
SCORE, COST = C["score"], C["cost"]
N = len(SCORE)
D1_MEAN = float((SCORE[:, 1] - SCORE[:, 0]).mean())


def greedy(score, cost, mult, k1_cap=1.0):
    """주어진 (score, cost)로 예산 안에서 이득/비용비 순 승격."""
    n = len(score)
    sel = np.zeros(n, dtype=int)
    spent = float(cost[:, 0].sum())
    cap = float(cost[:, 0].sum()) * mult
    k1_budget, k1_used = int(n * k1_cap), 0
    heap = []

    def push(i):
        cur = sel[i]
        for m in range(3):
            if m == cur:
                continue
            dq, dc = score[i, m] - score[i, cur], cost[i, m] - cost[i, cur]
            if dq <= 0:
                continue
            heapq.heappush(heap, (-1e18 if dc <= 0 else -dq / dc, i, m))

    for i in range(n):
        push(i)
    while heap:
        _, i, m = heapq.heappop(heap)
        cur = sel[i]
        dq, dc = score[i, m] - score[i, cur], cost[i, m] - cost[i, cur]
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


def measure(name, score_view, cost_view, k1_cap=1.0):
    total, parts = 0.0, []
    for tier in TIERS:
        sel = greedy(score_view, cost_view, MULT[tier], k1_cap)
        used = COST[np.arange(N), sel].sum() / COST[:, 0].sum()
        q = SCORE[np.arange(N), sel].mean()
        ok = used <= MULT[tier] + 1e-12
        total += WEIGHT[tier] * (q if ok else 0.0)
        parts.append(f"{tier[:4]}={q:.4f}/{used/MULT[tier]:.0%}")
    print(f"  {name:46s} {total:.6f}   " + "  ".join(parts))
    return total


print("=== 정보 수준별 도달 가능 점수 (Train+Dev 2,640, 실제 비용 사용) ===")

# 1) 아무것도 모름
measure("전부 경량 모델", np.zeros((N, 3)), COST)

# 2) 완전정보 오라클
measure("완전정보 오라클 (실제 score·cost)", SCORE, COST)

# 3) d1만 모름 (전역 평균으로 대체), d2·비용은 완전정보
s_no_d1 = SCORE.copy()
base = SCORE[:, 0]
s_no_d1[:, 1] = base + D1_MEAN
s_no_d1[:, 2] = base + D1_MEAN + (SCORE[:, 2] - SCORE[:, 1])
measure("d1 무지 + d2·비용 완전정보", s_no_d1, COST)
measure("d1 무지 + d2·비용 완전정보 + K1 11% 상한", s_no_d1, COST, 0.11)

# 4) score 전부 모름, 비용만 완전정보 (비용 싼 순 승격)
s_flat = np.column_stack([np.zeros(N), np.full(N, D1_MEAN),
                          np.full(N, D1_MEAN + float((SCORE[:, 2] - SCORE[:, 1]).mean()))])
measure("score 전부 무지 + 비용 완전정보", s_flat, COST)

# 5) 실제 비용을 모를 때: 예측 비용으로 승격하되 실제 비용으로 채점
c_pred = C["c_te8"]
measure("d1 무지 + d2 완전정보 + 예측 비용", s_no_d1, c_pred, 0.11)
measure("완전정보 score + 예측 비용", SCORE, c_pred, 0.11)

print("\n=== 우리 현재 위치 ===")
print("  현재 정책 (CV 기대점수)                        0.664872")
print("  현재 정책 (공개 Dev)                           0.672869")
print("\n  → 위 표의 'd1 무지 + d2 완전정보 + K1 상한'이 현실적 상한이다.")
print("     그 값과 0.665의 차이가 예측 개선으로 얻을 수 있는 최대치다.")
