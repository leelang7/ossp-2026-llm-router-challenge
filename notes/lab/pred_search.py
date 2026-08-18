# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""예측 개선 탐색 — 남은 기회는 d2(K1 이득)와 비용 예측뿐이다.

상한 분석 결과
  · score를 전혀 몰라도 비용만 정확하면 0.6828 (현재 0.6649)
  · d2를 완벽히 알면 0.7248
따라서 (1) d2 예측 상관을 올리고 (2) 출력 토큰 예측을 정확히 하는 것이 전부다.
설정을 바꿔가며 두 지표를 직접 측정한다. 정책은 건드리지 않는다.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MODEL_IDS, load_all  # noqa: E402

t0 = time.time()
DATA = load_all()
N = len(DATA.texts)
SCORE, OTOK = DATA.score, DATA.out_tok
D2 = SCORE[:, 2] - SCORE[:, 1]
D2_POS = (D2 > 0).astype(int)
FOLD = np.arange(N) % 5
print(f"[{time.time()-t0:.0f}s] n={N}  d2>0 비율={D2_POS.mean():.1%}")


def oof_predict(make_model, X, Y):
    P = np.empty((X.shape[0], Y.shape[1]))
    for f in range(5):
        va = FOLD == f
        P[va] = make_model(X[~va], Y[~va])(X[va])
    return P


def ridge_of(alpha):
    def fit(X, Y):
        m = Ridge(alpha=alpha, solver="sparse_cg", random_state=0).fit(X, Y)
        return lambda Z: m.predict(Z)
    return fit


def gbm_of(**kw):
    def fit(X, Y):
        p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=15,
                 l2_regularization=1.0, early_stopping=False, random_state=0)
        p.update(kw)
        ms = [HistGradientBoostingRegressor(**p).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        return lambda Z: np.column_stack([m.predict(Z) for m in ms])
    return fit


def report(tag, pred_score, pred_tok):
    d2p = pred_score[:, 2] - pred_score[:, 1]
    auc = roc_auc_score(D2_POS, d2p)
    c2 = float(np.corrcoef(d2p, D2)[0, 1])
    cs = [round(float(np.corrcoef(pred_score[:, j], SCORE[:, j])[0, 1]), 3) for j in range(3)]
    ct = [round(float(np.corrcoef(np.expm1(pred_tok[:, j]), OTOK[:, j])[0, 1]), 3) for j in range(3)]
    print(f"  {tag:40s} d2corr={c2:.3f} AUC={auc:.3f} score={cs} tok={ct}")
    return c2, auc


def build(word_ng, char_ng, min_df, max_w, max_c, sublinear=True):
    tw = TfidfVectorizer(analyzer="word", ngram_range=word_ng, min_df=min_df,
                         max_features=max_w, sublinear_tf=sublinear, dtype=np.float32)
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ng, min_df=min_df,
                         max_features=max_c, sublinear_tf=sublinear, dtype=np.float32)
    mu, sd = DATA.dense.mean(0), DATA.dense.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return sparse.hstack([tw.fit_transform(DATA.texts), tc.fit_transform(DATA.texts),
                          sparse.csr_matrix((DATA.dense - mu) / sd)]).tocsr()


print("\n=== A. 기준 설정에서 alpha 스윕 ===")
X = build((1, 2), (3, 5), 3, 60000, 120000)
print(f"[{time.time()-t0:.0f}s] X={X.shape}")
Ytok = np.log1p(OTOK)
best_alpha, best = None, -1
for a in (3.0, 10.0, 30.0, 60.0):
    ps = oof_predict(ridge_of(a), X, SCORE)
    pt = oof_predict(ridge_of(a), X, Ytok)
    c2, auc = report(f"ridge alpha={a:g}", np.clip(ps, 0, 1), pt)
    if c2 > best:
        best, best_alpha = c2, a
print(f"  최적 alpha={best_alpha} (d2corr {best:.3f})")

print("\n=== B. 특징 구성 변형 (alpha 고정) ===")
for tag, args in [
    ("word(1,2) char(3,5) mindf3 [기준]", ((1, 2), (3, 5), 3, 60000, 120000)),
    ("word(1,3) char(3,5)", ((1, 3), (3, 5), 3, 60000, 120000)),
    ("word(1,2) char(2,6)", ((1, 2), (2, 6), 3, 60000, 120000)),
    ("word(1,2) char(3,5) mindf2", ((1, 2), (3, 5), 2, 60000, 200000)),
    ("word(1,2) char(3,5) 대용량", ((1, 2), (3, 5), 3, 120000, 250000)),
]:
    Xv = build(*args)
    ps = oof_predict(ridge_of(best_alpha), Xv, SCORE)
    pt = oof_predict(ridge_of(best_alpha), Xv, Ytok)
    report(tag, np.clip(ps, 0, 1), pt)

print("\n=== C. GBM 앙상블 및 d2 전용 모델 ===")
from sklearn.decomposition import TruncatedSVD  # noqa: E402

svd = TruncatedSVD(n_components=220, random_state=0)
V = svd.fit_transform(X)
G = np.hstack([DATA.dense, V])
ps_r = oof_predict(ridge_of(best_alpha), X, SCORE)
pt_r = oof_predict(ridge_of(best_alpha), X, Ytok)
ps_g = oof_predict(gbm_of(), G, SCORE)
pt_g = oof_predict(gbm_of(max_iter=350, learning_rate=0.06, max_leaf_nodes=15,
                          min_samples_leaf=20), G, Ytok)
report("ridge 단독", np.clip(ps_r, 0, 1), pt_r)
report("GBM 단독", np.clip(ps_g, 0, 1), pt_g)
for w in (0.3, 0.5, 0.7):
    report(f"앙상블 ridge{w:.1f}+gbm{1-w:.1f}",
           np.clip(w * ps_r + (1 - w) * ps_g, 0, 1), w * pt_r + (1 - w) * pt_g)

# d2를 직접 회귀하는 전용 헤드
d2_r = oof_predict(ridge_of(best_alpha), X, D2[:, None])[:, 0]
d2_g = oof_predict(gbm_of(), G, D2[:, None])[:, 0]
for tag, p in (("d2 전용 ridge", d2_r), ("d2 전용 GBM", d2_g),
               ("d2 전용 앙상블", 0.5 * d2_r + 0.5 * d2_g)):
    print(f"  {tag:40s} d2corr={np.corrcoef(p, D2)[0,1]:.3f} "
          f"AUC={roc_auc_score(D2_POS, p):.3f}")
print(f"\n[{time.time()-t0:.0f}s] done")
