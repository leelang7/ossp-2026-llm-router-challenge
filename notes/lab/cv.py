# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""교차검증 평가 엔진 — 모든 개선을 여기서 측정한다.

Dev 880문항 단일 측정은 노이즈가 커서 0.005 미만의 개선을 구분할 수 없다.
Train+Dev 2,640문항을 K-fold로 나눠 fold마다
  (1) TF-IDF·회귀를 학습하고
  (2) fold 내부 OOF로 등급별 안전계수를 정한 뒤
  (3) 홀드아웃에서 공식 채점식으로 점수를 낸다.
누출을 막기 위해 어휘까지 fold 안에서만 만든다.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

CHALLENGE = r"d:\opensource\ossp-2026-llm-router-challenge"
sys.path.insert(0, CHALLENGE + r"\src")
sys.path.insert(0, r"d:\opensource\skt-router\lab")

from common import MODEL_IDS, POLICY, TIERS, load_split, episode_text  # noqa: E402
from routerx.features import dense_features  # noqa: E402
from routerx.policy import select_batch  # noqa: E402
from routerx.router import _tie_key  # noqa: E402

MULT = {t: float(POLICY.tiers[t].budget_multiplier) for t in TIERS}
WEIGHT = {t: float(POLICY.tiers[t].weight) for t in TIERS}
RATE_IN = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RATE_OUT = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
UNIT = float(POLICY.token_unit)


@dataclass
class Data:
    texts: List[str]
    dense: np.ndarray
    score: np.ndarray
    cost: np.ndarray
    out_tok: np.ndarray
    in_tok: np.ndarray
    keys: np.ndarray


def load_all() -> Data:
    texts, dense, score, cost, out_tok, in_tok = [], [], [], [], [], []
    for split in ("train", "dev"):
        inputs, _, rows = load_split(split)
        for ep, rec in zip(inputs.episodes, rows):
            text = episode_text(ep)
            texts.append(text)
            dense.append(dense_features(text, ep))
            score.append([rec[m]["score"] for m in MODEL_IDS])
            cost.append([rec[m]["cost"] for m in MODEL_IDS])
            out_tok.append([rec[m]["out_tok"] for m in MODEL_IDS])
            in_tok.append([rec[m]["in_tok"] for m in MODEL_IDS])
    return Data(
        texts=texts,
        dense=np.asarray(dense, dtype=np.float64),
        score=np.asarray(score, dtype=np.float64),
        cost=np.asarray(cost, dtype=np.float64),
        out_tok=np.asarray(out_tok, dtype=np.float64),
        in_tok=np.asarray(in_tok, dtype=np.float64),
        keys=np.asarray([_tie_key(t) for t in texts], dtype=np.float64),
    )


@dataclass
class Config:
    """실험 설정 — 이 값만 바꿔가며 비교한다."""
    name: str = "base"
    alpha: float = 10.0
    word_ngram: tuple = (1, 2)
    char_ngram: tuple = (3, 5)
    word_features: int = 60000
    char_features: int = 120000
    min_df: int = 3
    cost_quantile: float = 0.3
    margins: Dict[str, float] = field(default_factory=lambda: {"fast": 0.95, "balanced": 0.93, "premium": 0.88})
    use_dense: bool = True
    sublinear: bool = True
    extra: Dict = field(default_factory=dict)


def build_features(cfg: Config, fit_texts: Sequence[str], fit_dense: np.ndarray,
                   apply_sets: Sequence[tuple]):
    tf_w = TfidfVectorizer(analyzer="word", ngram_range=cfg.word_ngram, min_df=cfg.min_df,
                           max_features=cfg.word_features, sublinear_tf=cfg.sublinear,
                           lowercase=True, dtype=np.float32)
    tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=cfg.char_ngram, min_df=cfg.min_df,
                           max_features=cfg.char_features, sublinear_tf=cfg.sublinear,
                           lowercase=True, dtype=np.float32)
    Ww = tf_w.fit_transform(fit_texts)
    Wc = tf_c.fit_transform(fit_texts)
    mu, sd = fit_dense.mean(0), fit_dense.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    blocks = [Ww, Wc]
    if cfg.use_dense:
        blocks.append(sparse.csr_matrix((fit_dense - mu) / sd))
    X_fit = sparse.hstack(blocks).tocsr()
    outs = []
    for texts, dense in apply_sets:
        parts = [tf_w.transform(texts), tf_c.transform(texts)]
        if cfg.use_dense:
            parts.append(sparse.csr_matrix((dense - mu) / sd))
        outs.append(sparse.hstack(parts).tocsr())
    return X_fit, outs


def fit_predict(cfg: Config, X_fit, Y_fit, X_apply_list, folds: int = 5):
    model = Ridge(alpha=cfg.alpha, solver="sparse_cg", random_state=0).fit(X_fit, Y_fit)
    oof = np.empty_like(Y_fit)
    fid = np.arange(X_fit.shape[0]) % folds
    for f in range(folds):
        va = fid == f
        sub = Ridge(alpha=cfg.alpha, solver="sparse_cg", random_state=0).fit(X_fit[~va], Y_fit[~va])
        oof[va] = sub.predict(X_fit[va])
    return oof, [model.predict(X) for X in X_apply_list]


def to_score_cost(cfg: Config, raw: np.ndarray, bump: np.ndarray, cost_scale: np.ndarray):
    n_m = len(MODEL_IDS)
    pred_score = np.clip(raw[:, :n_m], 0.0, 1.0)
    pred_out = np.expm1(np.clip(raw[:, n_m:2 * n_m], -50, 50)).clip(0) * bump
    pred_in = np.expm1(np.clip(raw[:, 2 * n_m:2 * n_m + 1], -50, 50)).clip(1.0)
    cost = (np.repeat(pred_in, n_m, axis=1) * RATE_IN + pred_out * RATE_OUT) / UNIT
    return pred_score, np.maximum(cost * cost_scale, 1e-9)


def tier_scores(sel: np.ndarray, real_score: np.ndarray, real_cost: np.ndarray, tier: str):
    n = len(sel)
    used = real_cost[np.arange(n), sel].sum()
    light = real_cost[:, 0].sum()
    ratio = used / light
    passed = ratio <= MULT[tier] + 1e-12
    quality = real_score[np.arange(n), sel].mean()
    return (quality if passed else 0.0), ratio, passed, quality


def evaluate_config(cfg: Config, data: Data, n_folds: int = 5, seed: int = 0, verbose: bool = True):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data.texts))
    fold_of = np.empty(len(order), dtype=int)
    for i, idx in enumerate(order):
        fold_of[idx] = i % n_folds

    per_tier = {t: [] for t in TIERS}
    per_tier_ratio = {t: [] for t in TIERS}
    finals, fails = [], 0
    for f in range(n_folds):
        te = fold_of == f
        tr = ~te
        X_fit, (X_te,) = build_features(
            cfg, [data.texts[i] for i in np.where(tr)[0]], data.dense[tr],
            [([data.texts[i] for i in np.where(te)[0]], data.dense[te])],
        )
        Y = np.hstack([data.score[tr], np.log1p(data.out_tok[tr]), np.log1p(data.in_tok[tr][:, :1])])
        oof, (raw_te,) = fit_predict(cfg, X_fit, Y, [X_te])

        n_m = len(MODEL_IDS)
        bump = np.exp(np.quantile(np.log1p(data.out_tok[tr]) - oof[:, n_m:2 * n_m],
                                  cfg.cost_quantile, axis=0))
        s_oof, c_oof = to_score_cost(cfg, oof, bump, np.ones(n_m))
        cost_scale = data.cost[tr].sum(0) / c_oof.sum(0)
        s_oof, c_oof = to_score_cost(cfg, oof, bump, cost_scale)
        s_te, c_te = to_score_cost(cfg, raw_te, bump, cost_scale)

        final = 0.0
        for tier in TIERS:
            safety = 0.30
            for s in np.round(np.arange(0.30, 1.401, 0.005), 4)[::-1]:
                sel = select_batch(s_oof, c_oof, MULT[tier], float(s), data.keys[tr])
                ratio = data.cost[tr][np.arange(len(sel)), sel].sum() / data.cost[tr][:, 0].sum()
                if ratio <= MULT[tier] * cfg.margins[tier]:
                    safety = float(s)
                    break
            sel_te = select_batch(s_te, c_te, MULT[tier], safety, data.keys[te])
            pts, ratio, passed, _q = tier_scores(sel_te, data.score[te], data.cost[te], tier)
            per_tier[tier].append(pts)
            per_tier_ratio[tier].append(ratio)
            fails += 0 if passed else 1
            final += WEIGHT[tier] * pts
        finals.append(final)

    mean, std = float(np.mean(finals)), float(np.std(finals))
    if verbose:
        detail = "  ".join(
            f"{t[:4]}={np.mean(per_tier[t]):.4f}/{np.mean(per_tier_ratio[t]):.3f}" for t in TIERS
        )
        print(f"{cfg.name:34s} CV={mean:.6f}±{std:.4f}  {detail}"
              f"{'  !FAIL' + str(fails) if fails else ''}")
    return mean, std, fails, per_tier, per_tier_ratio


if __name__ == "__main__":
    t0 = time.time()
    data = load_all()
    print(f"[{time.time()-t0:.0f}s] loaded n={len(data.texts)}")
    evaluate_config(Config(name="baseline(현재 제출 설정)"), data)
    print(f"[{time.time()-t0:.0f}s] done")
