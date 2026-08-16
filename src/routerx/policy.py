# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""등급 예산 안에서 문항별 모델을 정하는 배치 정책.

예측 이득 대비 예측 비용이 큰 순서로 승격한다. 프롬프트 내용이 같은 문항은
예측도 같으므로 하나의 그룹으로 묶어 통째로 승격한다. 그룹 안에서 일부만
고르면 어느 문항을 고를지가 입력 순서에 좌우되는데, 운영자는 문항 ID와 순서를
바꿔 재실행하는 감사를 수행하므로 그런 의존이 남아서는 안 된다.

문항 ID·입력 순서는 어떤 단계에서도 쓰지 않는다.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Sequence, Tuple

import numpy as np

_FREE_GAIN_KEY = -1e18
HEAVY_MODEL = 2      # axk1-think


def _group_rows(tie_keys: Sequence[float]) -> List[List[int]]:
    """프롬프트 내용 키가 같은 문항을 한 그룹으로 묶는다(입력 순서 무관)."""
    buckets: Dict[float, List[int]] = {}
    for row, key in enumerate(tie_keys):
        buckets.setdefault(float(key), []).append(row)
    return [buckets[key] for key in sorted(buckets)]


def select_batch(pred_score: np.ndarray, pred_cost: np.ndarray,
                 budget_multiplier: float, safety_ratio: float,
                 tie_keys: Sequence[float],
                 heavy_share_cap: float = 1.0,
                 heavy_item_cap: float = 1.0) -> np.ndarray:
    """모든 문항을 경량 모델에서 시작해 예측 예산 한도까지 승격한다.

    추론 모델(axk1-think)은 출력 토큰 분포의 꼬리가 매우 두껍다. 공개 자료에서
    출력 토큰 중앙값은 1,570인데 최대는 130,504이고, 생성 반복까지 곱하면 한 문항이
    경량 총비용의 78%를 쓰는 경우가 있다. 몇 건만 빗나가도 등급 전체가 0점이 된다.

    예측 비용 분위수로 위험을 추정해 거르는 방식은 '예측은 작은데 실제가 큰' 문항을
    통과시켜 효과가 없었다. 그래서 두 가지 예측 무관 장치를 쓴다.
      · heavy_share_cap: 추론 모델을 고를 수 있는 문항 수 자체를 묶는다.
      · heavy_item_cap: 한 문항이 예측상 경량 총비용의 이 비율을 넘게 쓰면
        추론 모델로 올리지 않는다. 건수 상한만으로는 비싼 문항 몇 건이 예산을
        독차지할 수 있어, 개별 노출의 크기도 함께 제한한다.

    Args:
        pred_score: (n, 3) 모델별 예측 품질
        pred_cost:  (n, 3) 모델별 예측 비용 (양수)
        budget_multiplier: 등급 예산 배수
        safety_ratio: 예측 예산에 적용할 안전계수 (1.0 이하로 쓴다)
        tie_keys: (n,) 프롬프트 내용에서 계산한 그룹·정렬 키
        heavy_share_cap: 추론 모델을 고를 수 있는 문항 비율 상한
        heavy_item_cap: 추론 모델 승격 1건이 쓸 수 있는 경량 총비용 대비 비율 상한
    Returns:
        (n,) 선택한 모델 인덱스
    """
    n_rows, n_models = pred_score.shape
    selected = np.zeros(n_rows, dtype=np.int64)
    light_total = float(pred_cost[:, 0].sum())
    spent = light_total
    cap = light_total * max(1.0, budget_multiplier * safety_ratio)
    heavy_budget = int(n_rows * min(max(heavy_share_cap, 0.0), 1.0))
    heavy_used = 0
    item_limit = light_total * heavy_item_cap if heavy_item_cap < 1.0 else float("inf")

    groups = _group_rows(tie_keys)
    # 그룹 대표 행 하나로 이득·비용을 계산한다(같은 그룹은 예측이 동일하다).
    current: List[int] = [0] * len(groups)
    heap: List[Tuple[float, float, int, int]] = []

    def push(group_index: int) -> None:
        rows = groups[group_index]
        head = rows[0]
        size = len(rows)
        cur = current[group_index]
        for model in range(n_models):
            if model == cur:
                continue
            gain = float(pred_score[head, model] - pred_score[head, cur]) * size
            extra = float(pred_cost[head, model] - pred_cost[head, cur]) * size
            if gain <= 0.0:
                continue
            key = _FREE_GAIN_KEY if extra <= 0.0 else -gain / extra
            heapq.heappush(heap, (key, float(tie_keys[head]), model, group_index))

    for index in range(len(groups)):
        push(index)

    while heap:
        _, _, model, group_index = heapq.heappop(heap)
        rows = groups[group_index]
        head, size = rows[0], len(rows)
        cur = current[group_index]
        if model == cur:
            continue
        gain = float(pred_score[head, model] - pred_score[head, cur]) * size
        extra = float(pred_cost[head, model] - pred_cost[head, cur]) * size
        if gain <= 0.0:
            continue
        if model == HEAVY_MODEL and cur != HEAVY_MODEL:
            if heavy_used + size > heavy_budget:
                continue
            if float(pred_cost[head, HEAVY_MODEL]) > item_limit:
                continue
        if spent + extra <= cap:
            spent += extra
            if model == HEAVY_MODEL and cur != HEAVY_MODEL:
                heavy_used += size
            elif cur == HEAVY_MODEL and model != HEAVY_MODEL:
                heavy_used -= size
            current[group_index] = model
            for row in rows:
                selected[row] = model
            push(group_index)
    return selected
