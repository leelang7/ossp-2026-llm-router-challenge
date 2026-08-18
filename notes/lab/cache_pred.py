# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""예측 캐시 생성 — 정책 실험을 빠르게 반복하기 위해 예측 결과를 npz로 저장."""
from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy import sparse

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from common import MODEL_IDS, POLICY, build_matrix, load_split, episode_text  # noqa: E402

SEED = 0
t0 = time.time()
tr_in, tr_out, tr_rows = load_split("train")
dv_in, dv_out, dv_rows = load_split("dev")
txt_tr = [episode_text(e) for e in tr_in.episodes]
txt_dv = [episode_text(e) for e in dv_in.episodes]
g = lambda rows, k: np.array([[r[m][k] for m in MODEL_IDS] for r in rows], dtype=float)
Str, Ctr, Otr, Itr = (g(tr_rows, k) for k in ("score", "cost", "out_tok", "in_tok"))
Sdv, Cdv, Odv, Idv = (g(dv_rows, k) for k in ("score", "cost", "out_tok", "in_tok"))

Dtr = build_matrix(tr_in, 16)[:, :-16]
Ddv = build_matrix(dv_in, 16)[:, :-16]
tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)
tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000, sublinear_tf=True)
Xs_tr = sparse.hstack([tf_w.fit_transform(txt_tr), tf_c.fit_transform(txt_dv if False else txt_tr)]).tocsr()
Xs_dv = sparse.hstack([tf_w.transform(txt_dv), tf_c.transform(txt_dv)]).tocsr()
svd = TruncatedSVD(n_components=200, random_state=SEED)
Vtr, Vdv = svd.fit_transform(Xs_tr), svd.transform(Xs_dv)
Gtr, Gdv = np.hstack([Dtr, Vtr]), np.hstack([Ddv, Vdv])
print(f"[{time.time()-t0:.0f}s] features ready sparse={Xs_tr.shape} G={Gtr.shape}")


def oof(fit, X, Y, folds=5):
    P = np.empty((X.shape[0], Y.shape[1]))
    fid = np.arange(X.shape[0]) % folds
    for f in range(folds):
        va = fid == f
        P[va] = fit(X[~va], Y[~va])(X[va])
    return P


def ridge_fit(alpha):
    def f(X, Y):
        m = Ridge(alpha=alpha, solver="sparse_cg", random_state=SEED).fit(X, Y)
        return lambda Xn: m.predict(Xn)
    return f


def gbm_fit(**kw):
    def f(X, Y):
        p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=15,
                 l2_regularization=1.0, early_stopping=False, random_state=SEED)
        p.update(kw)
        ms = [HistGradientBoostingRegressor(**p).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        return lambda Xn: np.column_stack([m.predict(Xn) for m in ms])
    return f


out = {}
for a in (3.0, 10.0, 30.0):
    out[f"s_oof_r{a:g}"] = oof(ridge_fit(a), Xs_tr, Str)
    out[f"s_dev_r{a:g}"] = ridge_fit(a)(Xs_tr, Str)(Xs_dv)
    print(f"[{time.time()-t0:.0f}s] ridge a={a:g} done")
out["s_oof_g"] = oof(gbm_fit(), Gtr, Str)
out["s_dev_g"] = gbm_fit()(Gtr, Str)(Gdv)
print(f"[{time.time()-t0:.0f}s] gbm score done")

Ytok = np.log1p(Otr)
tf = gbm_fit(max_iter=350, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20)
out["t_oof"] = oof(tf, Gtr, Ytok)
out["t_dev"] = tf(Gtr, Ytok)(Gdv)
out.update(Str=Str, Ctr=Ctr, Otr=Otr, Itr=Itr, Sdv=Sdv, Cdv=Cdv, Odv=Odv, Idv=Idv)
np.savez_compressed(r"d:\opensource\skt-router\lab\pred_cache.npz", **out)
print(f"[{time.time()-t0:.0f}s] saved pred_cache.npz")
