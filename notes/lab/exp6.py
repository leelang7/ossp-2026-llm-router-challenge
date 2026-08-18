# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 6 — score 예측기 집중 개선 (TF-IDF + SVD, 규칙상 어휘·IDF 허용).

exp5 판별: 병목은 score 예측 (특히 K1: corr 0.309 vs 공식 0.398, d2 AUC 0.649 vs 0.708).
우리 cost 예측은 이미 공식보다 우수하므로 그대로 재사용.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy import sparse

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from common import (  # noqa: E402
    MODEL_IDS, TIERS, POLICY, build_matrix, load_split, episode_text,
    lagrangian_select, official_score,
)

SEED = 0
t0 = time.time()
tr_in, tr_out, tr_rows = load_split("train")
dv_in, dv_out, dv_rows = load_split("dev")
txt_tr = [episode_text(e) for e in tr_in.episodes]
txt_dv = [episode_text(e) for e in dv_in.episodes]
g = lambda rows, k: np.array([[r[m][k] for m in MODEL_IDS] for r in rows], dtype=float)
Str, Ctr, Otr, Itr = (g(tr_rows, k) for k in ("score", "cost", "out_tok", "in_tok"))
Sdv, Cdv, Odv, Idv = (g(dv_rows, k) for k in ("score", "cost", "out_tok", "in_tok"))
d1_dv, d2_dv = Sdv[:, 1] - Sdv[:, 0], Sdv[:, 2] - Sdv[:, 1]

# dense 피처 (기존)
Dtr_full = build_matrix(tr_in, 16)   # bins 최소화 → dense만 사용
Ddv_full = build_matrix(dv_in, 16)
nd = Dtr_full.shape[1] - 16
Dtr, Ddv = Dtr_full[:, :nd], Ddv_full[:, :nd]
print(f"[{time.time()-t0:.0f}s] dense={Dtr.shape}")

# TF-IDF
tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=60000,
                       sublinear_tf=True, lowercase=True)
tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000,
                       sublinear_tf=True, lowercase=True)
Wtr, Wdv = tf_w.fit_transform(txt_tr), tf_w.transform(txt_dv)
Ctr_s, Cdv_s = tf_c.fit_transform(txt_tr), tf_c.transform(txt_dv)
print(f"[{time.time()-t0:.0f}s] tfidf word={Wtr.shape} char={Ctr_s.shape}")

Xs_tr = sparse.hstack([Wtr, Ctr_s]).tocsr()
Xs_dv = sparse.hstack([Wdv, Cdv_s]).tocsr()

svd = TruncatedSVD(n_components=160, random_state=SEED)
Vtr = svd.fit_transform(Xs_tr)
Vdv = svd.transform(Xs_dv)
print(f"[{time.time()-t0:.0f}s] SVD={Vtr.shape} evr={svd.explained_variance_ratio_.sum():.3f}")

Gtr = np.hstack([Dtr, Vtr])
Gdv = np.hstack([Ddv, Vdv])


def oof_generic(make_fit, X, Y, folds=5):
    P = np.empty((X.shape[0], Y.shape[1]))
    fid = np.arange(X.shape[0]) % folds
    for f in range(folds):
        va = fid == f
        Xa = X[~va] if not sparse.issparse(X) else X[~va]
        pred = make_fit(Xa, Y[~va])
        P[va] = pred(X[va])
    return P


def make_ridge_sparse(alpha):
    def fit(X, Y):
        m = Ridge(alpha=alpha, solver="sparse_cg", random_state=SEED).fit(X, Y)
        return lambda Xn: m.predict(Xn)
    return fit


def make_gbm(**kw):
    def fit(X, Y):
        p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=15,
                 l2_regularization=1.0, early_stopping=False, random_state=SEED)
        p.update(kw)
        ms = [HistGradientBoostingRegressor(**p).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        return lambda Xn: np.column_stack([m.predict(Xn) for m in ms])
    return fit


def quality(name, P_dv):
    c = [round(float(np.corrcoef(P_dv[:, j], Sdv[:, j])[0, 1]), 3) for j in range(3)]
    pd1, pd2 = P_dv[:, 1] - P_dv[:, 0], P_dv[:, 2] - P_dv[:, 1]
    a1 = roc_auc_score((d1_dv > 0).astype(int), pd1)
    a2 = roc_auc_score((d2_dv > 0).astype(int), pd2)
    print(f"  {name:32s} corr={c}  AUC d1={a1:.3f} d2={a2:.3f}")
    return a2, c


print("\n=== score 예측기 후보 (Dev 품질) ===")
print("  [기준] 공식 hash-regex     corr=[0.384, 0.446, 0.398]  AUC d1=0.548 d2=0.708")
cands = {}

for alpha in (1.0, 3.0, 10.0):
    P = make_ridge_sparse(alpha)(Xs_tr, Str)(Xs_dv)
    O = oof_generic(make_ridge_sparse(alpha), Xs_tr, Str)
    cands[f"tfidf-ridge-a{alpha:g}"] = (O, P)
    quality(f"TFIDF ridge a={alpha:g}", P)

P = make_gbm()(Gtr, Str)(Gdv); O = oof_generic(make_gbm(), Gtr, Str)
cands["svd-gbm"] = (O, P)
quality("SVD+dense GBM", P)

P = make_ridge_sparse(3.0)(np.hstack([Dtr, Vtr]), Str)(np.hstack([Ddv, Vdv]))
O = oof_generic(make_ridge_sparse(3.0), Gtr, Str)
cands["svd-ridge"] = (O, P)
quality("SVD+dense ridge", P)

# 앙상블
best_pairs = [("tfidf-ridge-a3", 0.5), ("svd-gbm", 0.5)]
for w in (0.3, 0.5, 0.7):
    key = f"ens-tfr{w:.1f}"
    O = w * cands["tfidf-ridge-a3"][0] + (1 - w) * cands["svd-gbm"][0]
    P = w * cands["tfidf-ridge-a3"][1] + (1 - w) * cands["svd-gbm"][1]
    cands[key] = (O, P)
    quality(f"ENS tfidf-ridge {w:.1f} + gbm", P)

# ---------------- 비용 예측 (exp5와 동일, 우리 것이 이미 우수)
Ytok = np.log1p(Otr)
tok_fit = make_gbm(max_iter=350, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20)
t_o = oof_generic(tok_fit, Gtr, Ytok)
t_d = tok_fit(Gtr, Ytok)(Gdv)
RI = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RO = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
cost_of = lambda i, o: (i * RI + o * RO) / float(POLICY.token_unit)
in_tr, in_dv = np.repeat(Itr[:, :1], 3, 1), np.repeat(Idv[:, :1], 3, 1)
bump = np.exp(np.quantile(Ytok - t_o, 0.3, axis=0))
pc_o = cost_of(in_tr, np.expm1(t_o).clip(0) * bump)
sc = Ctr.sum(0) / pc_o.sum(0)
pc_d = cost_of(in_dv, np.expm1(t_d).clip(0) * bump) * sc
print(f"\n[{time.time()-t0:.0f}s] cost DEVcorr="
      f"{[round(float(np.corrcoef(pc_d[:,j], Cdv[:,j])[0,1]),3) for j in range(3)]} "
      f"총합비={np.round(pc_d.sum(0)/Cdv.sum(0),3)}")


def evaluate(S_d, tag):
    grid = np.round(np.arange(0.30, 1.001, 0.005), 4)[::-1]
    idx_dv, ch = {}, {}
    for tier in TIERS:
        mult = float(POLICY.tiers[tier].budget_multiplier)
        pick = grid[-1]
        for s in grid:
            idx, _ = lagrangian_select(S_d, pc_d, mult, s)
            if Cdv[np.arange(len(idx)), idx].sum() / Cdv[:, 0].sum() <= mult * 0.995:
                pick = s; break
        ch[tier] = pick
        idx_dv[tier], _ = lagrangian_select(S_d, pc_d, mult, pick)
    rep = official_score(dv_in, dv_out, idx_dv)
    ln = [f"{tag:32s} DEV={rep['final_score'][:8]}"]
    for tier in TIERS:
        t = rep["tiers"][tier]
        ln.append(f" {tier[:4]}={t['tier_score'][:6]}/{t['budget_ratio'][:5]}")
    print("".join(ln))
    return float(rep["final_score"])


print("\n=== 정책 평가 ===")
res = {}
for k, (O, P) in cands.items():
    res[k] = evaluate(np.clip(P, 0, 1), k)
print(f"\n  hash-regex 공식 DEV=0.695369")
bk = max(res, key=res.get)
print(f"  BEST: {bk} = {res[bk]:.6f}  [{time.time()-t0:.0f}s]")
