# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""부스팅 트리 앙상블을 NumPy 배열만으로 평가한다.

출력 토큰 수는 프롬프트 길이에 선형이 아니라서 선형 회귀만으로는 잘 맞지 않는다
(공개 자료에서 상관 0.17~0.39). 부스팅 트리를 쓰면 0.37~0.56까지 오르고, 비용
예측이 정확해지면 같은 안전 수준에서 예산을 더 쓸 수 있다.

학습에는 scikit-learn을 쓰되 실행 이미지에는 넣지 않는다. 학습된 트리를 평탄한
배열로 내보내고, 여기서는 깊이만큼만 반복하는 벡터화 순회로 값을 구한다.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def export_forest(models) -> Dict[str, np.ndarray]:
    """scikit-learn HistGradientBoostingRegressor 목록을 배열 묶음으로 내보낸다.

    각 모델의 트리를 하나로 이어 붙이고, 트리 시작 위치와 모델 경계를 함께 담는다.
    결측값은 학습·추론 모두에서 발생하지 않으므로 결측 분기는 내보내지 않는다.
    """
    feature: List[int] = []
    threshold: List[float] = []
    left: List[int] = []
    right: List[int] = []
    value: List[float] = []
    is_leaf: List[int] = []
    tree_start: List[int] = []
    model_tree_start: List[int] = []
    baseline: List[float] = []

    for model in models:
        model_tree_start.append(len(tree_start))
        baseline.append(float(np.ravel(model._baseline_prediction)[0]))
        for stage in model._predictors:
            for predictor in stage:
                nodes = predictor.nodes
                offset = len(feature)
                tree_start.append(offset)
                for node in nodes:
                    leaf = bool(node["is_leaf"])
                    is_leaf.append(1 if leaf else 0)
                    value.append(float(node["value"]))
                    feature.append(0 if leaf else int(node["feature_idx"]))
                    threshold.append(0.0 if leaf else float(node["num_threshold"]))
                    left.append(offset + int(node["left"]))
                    right.append(offset + int(node["right"]))
    model_tree_start.append(len(tree_start))
    return {
        "tree_feature": np.asarray(feature, dtype=np.int32),
        "tree_threshold": np.asarray(threshold, dtype=np.float64),
        "tree_left": np.asarray(left, dtype=np.int32),
        "tree_right": np.asarray(right, dtype=np.int32),
        "tree_value": np.asarray(value, dtype=np.float64),
        "tree_is_leaf": np.asarray(is_leaf, dtype=np.int8),
        "tree_start": np.asarray(tree_start, dtype=np.int32),
        "tree_model_start": np.asarray(model_tree_start, dtype=np.int32),
        "tree_baseline": np.asarray(baseline, dtype=np.float64),
    }


class Forest:
    """내보낸 트리 배열로 예측을 계산한다."""

    def __init__(self, data):
        self.feature = np.asarray(data["tree_feature"], dtype=np.int64)
        self.threshold = np.asarray(data["tree_threshold"], dtype=np.float64)
        self.left = np.asarray(data["tree_left"], dtype=np.int64)
        self.right = np.asarray(data["tree_right"], dtype=np.int64)
        self.value = np.asarray(data["tree_value"], dtype=np.float64)
        self.is_leaf = np.asarray(data["tree_is_leaf"], dtype=bool)
        self.start = np.asarray(data["tree_start"], dtype=np.int64)
        self.model_start = np.asarray(data["tree_model_start"], dtype=np.int64)
        self.baseline = np.asarray(data["tree_baseline"], dtype=np.float64)
        self.n_models = len(self.model_start) - 1

    def predict(self, features: np.ndarray) -> np.ndarray:
        """(n, d) 특징에서 (n, n_models) 예측을 낸다."""
        n_rows = features.shape[0]
        out = np.empty((n_rows, self.n_models), dtype=np.float64)
        rows = np.arange(n_rows)
        for model_index in range(self.n_models):
            total = np.full(n_rows, self.baseline[model_index], dtype=np.float64)
            first = self.model_start[model_index]
            last = self.model_start[model_index + 1]
            for tree_index in range(first, last):
                node = np.full(n_rows, self.start[tree_index], dtype=np.int64)
                # 트리 깊이만큼만 반복한다. 잎에 닿은 행은 제자리에 머문다.
                while True:
                    leaf = self.is_leaf[node]
                    if leaf.all():
                        break
                    picked = features[rows, self.feature[node]]
                    go_left = picked <= self.threshold[node]
                    nxt = np.where(go_left, self.left[node], self.right[node])
                    node = np.where(leaf, node, nxt)
                total += self.value[node]
            out[:, model_index] = total
        return out
