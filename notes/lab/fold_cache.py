# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""fold별 예측을 캐시한다 — 정책·마진 탐색을 초 단위로 반복하기 위해.

cv_cap 결과: 예측 비용 기반 상한(cap)은 효과가 없었다.
문제는 '예측은 작은데 실제가 큰' 문항이므로 예측값 필터로는 잡히지 않는다.
따라서 남은 수단은 (1) 등급별 마진 정밀 조정, (2) 비용 예측 자체의 개선이다.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import Config, MODEL_IDS, POLICY, load_all  # noqa: E402

RATE_IN = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RATE_OUT = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
UNIT = float(POLICY.token_unit)
N_M = len(MODEL_IDS)
OUT = r"d:\opensource\skt-router\lab\fold_cache.npz"

if __name__ == "__main__":
    t0 = time.time()
    data = load_all()
    cfg = Config()
    tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=60000,
                           sublinear_tf=True, dtype=np.float32)
    tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000,
                           sublinear_tf=True, dtype=np.float32)
    Ww, Wc = tf_w.fit_transform(data.texts), tf_c.fit_transform(data.texts)
    mu, sd = data.dense.mean(0), data.dense.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    X = sparse.hstack([Ww, Wc, sparse.csr_matrix((data.dense - mu) / sd)]).tocsr()
    print(f"[{time.time()-t0:.0f}s] X={X.shape}")

    store = {"score": data.score, "cost": data.cost, "out_tok": data.out_tok,
             "in_tok": data.in_tok, "keys": data.keys}
    for n_folds in (5, 8, 10):
        rng = np.random.default_rng(0)
        order = rng.permutation(X.shape[0])
        fold = np.empty(X.shape[0], dtype=int)
        for i, idx in enumerate(order):
            fold[idx] = i % n_folds
        store[f"fold{n_folds}"] = fold
        s_oof_all = np.zeros((X.shape[0], N_M))
        c_oof_all = np.zeros((X.shape[0], N_M))
        s_te_all = np.zeros((X.shape[0], N_M))
        c_te_all = np.zeros((X.shape[0], N_M))
        for f in range(n_folds):
            te, tr = fold == f, fold != f
            Ytr = np.hstack([data.score[tr], np.log1p(data.out_tok[tr]),
                             np.log1p(data.in_tok[tr][:, :1])])
            ridge = Ridge(alpha=cfg.alpha, solver="sparse_cg", random_state=0).fit(X[tr], Ytr)
            Xtr = X[tr]
            oof = np.empty_like(Ytr)
            inner = np.arange(tr.sum()) % 5
            for g in range(5):
                va = inner == g
                sub = Ridge(alpha=cfg.alpha, solver="sparse_cg", random_state=0).fit(Xtr[~va], Ytr[~va])
                oof[va] = sub.predict(Xtr[va])
            raw_te = ridge.predict(X[te])
            bump = np.exp(np.quantile(np.log1p(data.out_tok[tr]) - oof[:, N_M:2 * N_M],
                                      cfg.cost_quantile, axis=0))

            def mk(raw):
                s = np.clip(raw[:, :N_M], 0, 1)
                o = np.expm1(np.clip(raw[:, N_M:2 * N_M], -50, 50)).clip(0) * bump
                i_ = np.expm1(np.clip(raw[:, 2 * N_M:2 * N_M + 1], -50, 50)).clip(1)
                return s, (np.repeat(i_, N_M, axis=1) * RATE_IN + o * RATE_OUT) / UNIT

            s_o, c_o = mk(oof)
            scale = data.cost[tr].sum(0) / c_o.sum(0)
            c_o = np.maximum(c_o * scale, 1e-9)
            s_t, c_t = mk(raw_te)
            c_t = np.maximum(c_t * scale, 1e-9)
            # OOF는 학습 부분에, 홀드아웃 예측은 평가 부분에 저장
            s_oof_all[tr] += s_o / (n_folds - 1)
            c_oof_all[tr] += c_o / (n_folds - 1)
            s_te_all[te] = s_t
            c_te_all[te] = c_t
            store[f"f{n_folds}_{f}_s_oof"] = s_o
            store[f"f{n_folds}_{f}_c_oof"] = c_o
        store[f"s_te{n_folds}"] = s_te_all
        store[f"c_te{n_folds}"] = c_te_all
        print(f"[{time.time()-t0:.0f}s] {n_folds}-fold 예측 캐시 완료")
    np.savez_compressed(OUT, **store)
    print(f"[{time.time()-t0:.0f}s] saved {OUT}")
