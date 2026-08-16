# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""아티팩트를 읽어 프롬프트만으로 모델을 선택하는 라우터."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .features import (
    char_wb_ngrams,
    dense_features,
    episode_text,
    l2_normalize,
    tfidf_row,
    word_ngrams,
)
from .policy import select_batch
from .trees import Forest

ARTIFACT_NAME = "artifact.npz"


class Artifact:
    """전역 계수와 어휘·IDF만 담은 학습 산출물."""

    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=True)
        # (features, targets) 로 두면 행 인덱싱 한 번으로 배치 내적이 된다.
        self.coef_t = np.ascontiguousarray(
            np.asarray(data["coef"], dtype=np.float64).T
        )
        self.intercept = np.asarray(data["intercept"], dtype=np.float64)
        self.vocab_word: Dict[str, int] = {
            str(term): i for i, term in enumerate(data["vocab_word"])
        }
        self.idf_word = np.asarray(data["idf_word"], dtype=np.float64)
        self.vocab_char: Dict[str, int] = {
            str(term): i for i, term in enumerate(data["vocab_char"])
        }
        self.idf_char = np.asarray(data["idf_char"], dtype=np.float64)
        self.dense_mean = np.asarray(data["dense_mean"], dtype=np.float64)
        self.dense_scale = np.asarray(data["dense_scale"], dtype=np.float64)
        self.token_bump = np.asarray(data["token_bump"], dtype=np.float64)
        self.cost_scale = np.asarray(data["cost_scale"], dtype=np.float64)
        self.rate_in = np.asarray(data["rate_in"], dtype=np.float64)
        self.rate_out = np.asarray(data["rate_out"], dtype=np.float64)
        self.token_unit = float(data["token_unit"])
        self.tiers: List[str] = [str(t) for t in data["tiers"]]
        self.safety = {t: float(v) for t, v in zip(self.tiers, data["safety"])}
        caps = data["k1_cap"] if "k1_cap" in data.files else np.ones(len(self.tiers))
        self.k1_cap = {t: float(v) for t, v in zip(self.tiers, caps)}
        self.k1_item_cap = float(data["k1_item_cap"]) if "k1_item_cap" in data.files else 1.0
        self.model_ids: List[str] = [str(m) for m in data["model_ids"]]
        self.policy_id = str(data["policy_id"])
        self.meta = json.loads(str(data["meta"]))
        self.n_word = len(self.vocab_word)
        self.n_char = len(self.vocab_char)
        self.n_models = len(self.model_ids)
        dense_offset = self.n_word + self.n_char
        self.dense_coef = np.ascontiguousarray(
            self.coef_t[dense_offset:dense_offset + self.dense_mean.size]
        )
        # 토큰 수는 부스팅 트리로 예측한다(선형 모델보다 상관이 크게 높다).
        if "tree_start" not in data.files:
            raise ValueError(
                f"{path.name}에 토큰 예측 트리가 없습니다. "
                "train_routerx/train.py로 다시 학습하십시오."
            )
        self.forest = Forest(data)


def _tie_key(text: str) -> float:
    """프롬프트 내용만으로 만든 그룹·정렬 키 (입력 순서와 무관).

    입력 규격은 고립 서로게이트를 허용하므로 surrogatepass로 인코딩한다.
    기본 인코딩은 UnicodeEncodeError를 내고, 그 예외가 등급 실행 실패로 이어진다.
    """
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def predict(episodes: Sequence, artifact: Artifact):
    """모든 문항의 모델별 예측 품질과 예측 비용을 계산한다."""
    n_rows = len(episodes)
    raw = np.tile(artifact.intercept, (n_rows, 1))
    tie_keys = np.empty(n_rows, dtype=np.float64)
    dense_raw = np.empty((n_rows, artifact.dense_mean.size), dtype=np.float64)
    coef_t = artifact.coef_t

    for row, episode in enumerate(episodes):
        text = episode_text(episode)
        tie_keys[row] = _tie_key(text)
        # 학습에서는 word·char 두 TfidfVectorizer가 각자 L2 정규화한 결과를 이어붙인다.
        # 합친 뒤 한 번만 정규화하면 블록 간 비중이 달라져 예측이 어긋난다.
        sparse_row = l2_normalize(
            tfidf_row(word_ngrams(text), artifact.vocab_word, artifact.idf_word)
        )
        sparse_row.update(
            l2_normalize(
                tfidf_row(char_wb_ngrams(text), artifact.vocab_char, artifact.idf_char,
                          offset=artifact.n_word)
            )
        )
        if sparse_row:
            columns = np.fromiter(sparse_row.keys(), dtype=np.int64, count=len(sparse_row))
            values = np.fromiter(sparse_row.values(), dtype=np.float64, count=len(sparse_row))
            raw[row] += values @ coef_t[columns]
        values = np.asarray(dense_features(text, episode))
        dense_raw[row] = values
        dense = (values - artifact.dense_mean) / artifact.dense_scale
        raw[row] += dense @ artifact.dense_coef

    n_m = artifact.n_models
    pred_score = np.clip(raw[:, :n_m], 0.0, 1.0)
    tokens = artifact.forest.predict(dense_raw)
    pred_out = np.expm1(np.clip(tokens[:, :n_m], -50.0, 50.0)).clip(0.0)
    pred_out = pred_out * artifact.token_bump
    # 입력 토큰 수도 라우팅 시점에 주어지지 않으므로 학습과 같은 예측값을 쓴다.
    pred_in = np.expm1(np.clip(tokens[:, n_m:n_m + 1], -50.0, 50.0)).clip(1.0)
    in_rep = np.repeat(pred_in, n_m, axis=1)
    pred_cost = (in_rep * artifact.rate_in + pred_out * artifact.rate_out) / artifact.token_unit
    pred_cost = pred_cost * artifact.cost_scale
    pred_cost = np.maximum(pred_cost, 1e-9)
    return pred_score, pred_cost, tie_keys


def route(episodes: Sequence, artifact: Artifact, tier: str) -> List[str]:
    pred_score, pred_cost, tie_keys = predict(episodes, artifact)
    multiplier = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}[tier]
    selected = select_batch(pred_score, pred_cost, multiplier,
                            artifact.safety.get(tier, 0.85), tie_keys,
                            artifact.k1_cap.get(tier, 1.0), artifact.k1_item_cap)
    return [artifact.model_ids[int(i)] for i in selected]
