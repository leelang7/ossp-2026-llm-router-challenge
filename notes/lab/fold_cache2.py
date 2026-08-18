# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""fold 예측 캐시 v2 — 토큰 예측을 부스팅 트리로 바꾼 구성.

score는 ridge(전체 특징), 출력·입력 토큰은 GBM(직접 계산 특징 36개).
공개 Dev에서 토큰 상관이 0.14~0.37 → 0.38~0.65로 오르면서 비용 예측이 정확해졌고,
그만큼 예산 마진을 올릴 수 있게 되었다. 그 여유가 실제로 안전한지 교차검증한다.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import Config, MODEL_IDS, POLICY, load_all  # noqa: E402

RATE_IN = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RATE_OUT = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
UNIT = float(POLICY.token_unit)
N_M = 3
OUT = r"d:\opensource\skt-router\lab\fold_cache2.npz"
ALPHA = 5.0
TREE = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=15,
            l2_regularization=1.0, early_stopping=False, random_state=0)

if __name__ == "__main__":
    t0 = time.time()
    data = load_all()
    cfg = Config()
    tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=60000,
                           sublinear_tf=True, dtype=np.float32)
    tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000,
                           sublinear_tf=True, dtype=np.float32)
    mu, sd = data.dense.mean(0), data.dense.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    X = sparse.hstack([tf_w.fit_transform(data.texts), tf_c.fit_transform(data.texts),
                       sparse.csr_matrix((data.dense - mu) / sd)]).tocsr()
    D = data.dense
    TOK = np.hstack([np.log1p(data.out_tok), np.log1p(data.in_tok[:, :1])])
    print(f"[{time.time()-t0:.0f}s] X={X.shape} dense={D.shape}")

    store = {"score": data.score, "cost": data.cost, "out_tok": data.out_tok,
             "in_tok": data.in_tok, "keys": data.keys}

    def fit_tokens(train_mask, apply_matrix):
        models = [HistGradientBoostingRegressor(**TREE).fit(D[train_mask], TOK[train_mask, j])
                  for j in range(TOK.shape[1])]
        return np.column_stack([m.predict(apply_matrix) for m in models])

    for n_folds in (5, 8):
        rng = np.random.default_rng(0)
        order = rng.permutation(X.shape[0])
        fold = np.empty(X.shape[0], dtype=int)
        for i, idx in enumerate(order):
            fold[idx] = i % n_folds
        store[f"fold{n_folds}"] = fold
        s_te_all = np.zeros((X.shape[0], N_M))
        c_te_all = np.zeros((X.shape[0], N_M))
        for f in range(n_folds):
            te, tr = fold == f, fold != f
            ridge = Ridge(alpha=ALPHA, solver="sparse_cg", random_state=0).fit(X[tr], data.score[tr])
            Xtr = X[tr]
            s_oof = np.empty((tr.sum(), N_M))
            inner = np.arange(tr.sum()) % 5
            for g in range(5):
                va = inner == g
                sub = Ridge(alpha=ALPHA, solver="sparse_cg", random_state=0).fit(Xtr[~va], data.score[tr][~va])
                s_oof[va] = sub.predict(Xtr[va])
            s_te = ridge.predict(X[te])

            # 토큰: full-fit 예측으로 보정값을 정하고, 안전계수 탐색용 OOF도 만든다
            t_fit = fit_tokens(tr, D[tr])
            t_te = fit_tokens(tr, D[te])
            t_oof = np.empty((tr.sum(), TOK.shape[1]))
            idx_tr = np.where(tr)[0]
            for g in range(5):
                va = inner == g
                inner_train = idx_tr[~va]
                mask = np.zeros(X.shape[0], dtype=bool)
                mask[inner_train] = True
                t_oof[va] = fit_tokens(mask, D[idx_tr[va]])

            def to_cost(tok, bump, scale):
                out = np.expm1(np.clip(tok[:, :N_M], -50, 50)).clip(0) * bump
                inp = np.expm1(np.clip(tok[:, N_M:N_M + 1], -50, 50)).clip(1)
                c = (np.repeat(inp, N_M, axis=1) * RATE_IN + out * RATE_OUT) / UNIT
                return np.maximum(c * scale, 1e-9)

            # bump는 처음 보는 자료에서의 오차를 담아야 하므로 OOF 잔차로 구한다.
            # full-fit 잔차로 구하면 트리 모델에서 거의 0이 되어 비용을 크게 과소평가한다.
            bump = np.exp(np.quantile(TOK[tr, :N_M] - t_oof[:, :N_M], cfg.cost_quantile, axis=0))
            scale = data.cost[tr].sum(0) / to_cost(t_fit, bump, np.ones(N_M)).sum(0)
            store[f"f{n_folds}_{f}_s_oof"] = np.clip(s_oof, 0, 1)
            store[f"f{n_folds}_{f}_c_oof"] = to_cost(t_oof, bump, scale)
            s_te_all[te] = np.clip(s_te, 0, 1)
            c_te_all[te] = to_cost(t_te, bump, scale)
            print(f"[{time.time()-t0:.0f}s]   {n_folds}-fold {f+1}/{n_folds}")
        store[f"s_te{n_folds}"] = s_te_all
        store[f"c_te{n_folds}"] = c_te_all
    np.savez_compressed(OUT, **store)
    print(f"[{time.time()-t0:.0f}s] saved {OUT}")
