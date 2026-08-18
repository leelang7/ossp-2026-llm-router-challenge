# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""동일 조건 비교 — 이전 정책 vs K1 상한 정책을 같은 척도(Dev 880)로 측정한다.

CV(0.66)와 Dev(0.69)는 서로 다른 척도라 직접 비교할 수 없다.
  · CV: fold별로 2,112~2,310문항만 학습 → 예측이 불리, 홀드아웃 330~528문항 → 변동 큼
  · Dev: 1,760문항 학습 후 880문항 평가 → 단일 측정
따라서 두 정책을 모두 Dev에서 재보고, CV에서도 모두 재본다.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import Config, MODEL_IDS, MULT, POLICY, TIERS, WEIGHT, load_split, episode_text  # noqa: E402
from routerx.features import dense_features  # noqa: E402
from routerx.policy import select_batch  # noqa: E402
from routerx.router import _tie_key  # noqa: E402

RATE_IN = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RATE_OUT = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
UNIT = float(POLICY.token_unit)
N_M = 3
CFG = Config()


def pack(split):
    inputs, _, rows = load_split(split)
    texts = [episode_text(e) for e in inputs.episodes]
    dense = np.asarray([dense_features(t, e) for t, e in zip(texts, inputs.episodes)])
    g = lambda k: np.asarray([[r[m][k] for m in MODEL_IDS] for r in rows], dtype=float)
    return dict(texts=texts, dense=dense, score=g("score"), cost=g("cost"),
                out_tok=g("out_tok"), in_tok=g("in_tok"),
                keys=np.asarray([_tie_key(t) for t in texts]))


TR, DV = pack("train"), pack("dev")
tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=60000,
                       sublinear_tf=True, dtype=np.float32)
tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000,
                       sublinear_tf=True, dtype=np.float32)
Ww, Wc = tf_w.fit_transform(TR["texts"]), tf_c.fit_transform(TR["texts"])
mu, sd = TR["dense"].mean(0), TR["dense"].std(0)
sd = np.where(sd > 1e-12, sd, 1.0)
Xtr = sparse.hstack([Ww, Wc, sparse.csr_matrix((TR["dense"] - mu) / sd)]).tocsr()
Xdv = sparse.hstack([tf_w.transform(DV["texts"]), tf_c.transform(DV["texts"]),
                     sparse.csr_matrix((DV["dense"] - mu) / sd)]).tocsr()

Y = np.hstack([TR["score"], np.log1p(TR["out_tok"]), np.log1p(TR["in_tok"][:, :1])])
ridge = Ridge(alpha=CFG.alpha, solver="sparse_cg", random_state=0).fit(Xtr, Y)
oof = np.empty_like(Y)
inner = np.arange(Xtr.shape[0]) % 5
for g_ in range(5):
    va = inner == g_
    sub = Ridge(alpha=CFG.alpha, solver="sparse_cg", random_state=0).fit(Xtr[~va], Y[~va])
    oof[va] = sub.predict(Xtr[va])
raw_dv = ridge.predict(Xdv)
bump = np.exp(np.quantile(np.log1p(TR["out_tok"]) - oof[:, N_M:2 * N_M], CFG.cost_quantile, axis=0))


def mk(raw):
    s = np.clip(raw[:, :N_M], 0, 1)
    o = np.expm1(np.clip(raw[:, N_M:2 * N_M], -50, 50)).clip(0) * bump
    i_ = np.expm1(np.clip(raw[:, 2 * N_M:2 * N_M + 1], -50, 50)).clip(1)
    return s, (np.repeat(i_, N_M, axis=1) * RATE_IN + o * RATE_OUT) / UNIT


s_oof, c_oof = mk(oof)
scale = TR["cost"].sum(0) / c_oof.sum(0)
c_oof = np.maximum(c_oof * scale, 1e-9)
s_dv, c_dv = mk(raw_dv)
c_dv = np.maximum(c_dv * scale, 1e-9)

GRID = np.round(np.arange(0.30, 1.401, 0.005), 4)[::-1]


def dev_score(policy: dict, tag: str):
    total, detail = 0.0, []
    for tier in TIERS:
        margin = policy[tier]["margin"]
        k1 = policy[tier]["k1"]
        safety = 0.30
        for s in GRID:
            sel = select_batch(s_oof, c_oof, MULT[tier], float(s), TR["keys"], k1)
            r = TR["cost"][np.arange(len(sel)), sel].sum() / TR["cost"][:, 0].sum()
            if r <= MULT[tier] * margin:
                safety = float(s)
                break
        sel = select_batch(s_dv, c_dv, MULT[tier], safety, DV["keys"], k1)
        n = len(sel)
        ratio = DV["cost"][np.arange(n), sel].sum() / DV["cost"][:, 0].sum()
        ok = ratio <= MULT[tier] + 1e-12
        q = DV["score"][np.arange(n), sel].mean()
        total += WEIGHT[tier] * (q if ok else 0.0)
        detail.append(f"{tier[:4]}={q:.4f}/{ratio:.3f}({ratio/MULT[tier]:.0%})"
                      f"K1={np.mean(sel==2):.0%}{'' if ok else '!OVER'}")
    print(f"  {tag:34s} Dev={total:.6f}  " + "  ".join(detail))
    return total


if __name__ == "__main__":
    print("=== 같은 척도(Dev 880, Train만 학습) 비교 ===")
    old = {t: {"margin": m, "k1": 1.0} for t, m in
           (("fast", 0.95), ("balanced", 0.93), ("premium", 0.88))}
    dev_score(old, "이전 정책(K1 무제한, 제출본)")
    for k1p, mp in ((0.08, 0.95), (0.06, 0.95), (0.10, 0.95)):
        new = {"fast": {"margin": 0.85, "k1": 0.0},
               "balanced": {"margin": 0.80, "k1": 0.05},
               "premium": {"margin": mp, "k1": k1p}}
        dev_score(new, f"K1상한 정책(prem k1={k1p} m={mp})")
    print("\n  참고: 공식 hash-regex Dev=0.695369 (Dev로 안전계수 보정한 값)")
