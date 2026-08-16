# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""등급 예산 안에서 문항별 모델을 정하는 배치 정책.

예측 이득 대비 예측 비용이 큰 순서로 승격하되, 동률은 프롬프트 내용에서
계산한 키로만 깬다. 문항 ID·입력 순서는 어떤 단계에서도 쓰지 않으므로
입력을 섞어도 같은 결과가 나온다.
"""
from __future__ import annotations

import heapq
from typing import Sequence

import numpy as np

_FREE_GAIN_KEY = -1e18


HEAVY_MODEL = 2      # axk1-think


def select_batch(pred_score: np.ndarray, pred_cost: np.ndarray,
                 budget_multiplier: float, safety_ratio: float,
                 tie_keys: Sequence[float],
                 heavy_share_cap: float = 1.0) -> np.ndarray:
    """모든 문항을 경량 모델에서 시작해 예측 예산 한도까지 승격한다.

    추론 모델(axk1-think)은 출력 토큰 분포의 꼬리가 매우 두껍다. 한 문항이 전체
    경량 비용의 26%까지 쓸 수 있어, 몇 건만 빗나가도 등급 전체가 0점이 된다.
    게다가 그 고비용 구간은 승격 이득 대비 비용이 가장 나쁜 구간이기도 하다.
    그래서 예측 예산과 별개로 추론 모델 선택 건수 자체에 상한을 둔다.

    Args:
        pred_score: (n, 3) 모델별 예측 품질
        pred_cost:  (n, 3) 모델별 예측 비용 (양수)
        budget_multiplier: 등급 예산 배수
        safety_ratio: 예측 예산에 적용할 안전계수
        tie_keys: (n,) 프롬프트 내용에서 계산한 동률 정렬 키
        heavy_share_cap: 추론 모델을 고를 수 있는 문항 비율 상한
    Returns:
        (n,) 선택한 모델 인덱스
    """
    n_rows, n_models = pred_score.shape
    selected = np.zeros(n_rows, dtype=np.int64)
    spent = float(pred_cost[:, 0].sum())
    cap = float(pred_cost[:, 0].sum()) * max(1.0, budget_multiplier * safety_ratio)
    heavy_budget = int(n_rows * min(max(heavy_share_cap, 0.0), 1.0))
    heavy_used = 0

    heap: list = []

    def push(row: int) -> None:
        current = selected[row]
        for model in range(n_models):
            if model == current:
                continue
            gain = float(pred_score[row, model] - pred_score[row, current])
            extra = float(pred_cost[row, model] - pred_cost[row, current])
            if gain <= 0.0:
                continue
            key = _FREE_GAIN_KEY if extra <= 0.0 else -gain / extra
            heapq.heappush(heap, (key, float(tie_keys[row]), model, row))

    for row in range(n_rows):
        push(row)

    while heap:
        _, _, model, row = heapq.heappop(heap)
        current = selected[row]
        gain = float(pred_score[row, model] - pred_score[row, current])
        extra = float(pred_cost[row, model] - pred_cost[row, current])
        if gain <= 0.0:
            continue
        if model == HEAVY_MODEL and current != HEAVY_MODEL and heavy_used >= heavy_budget:
            continue
        if spent + extra <= cap:
            spent += extra
            if model == HEAVY_MODEL and current != HEAVY_MODEL:
                heavy_used += 1
            elif current == HEAVY_MODEL and model != HEAVY_MODEL:
                heavy_used -= 1
            selected[row] = model
            push(row)
    return selected
