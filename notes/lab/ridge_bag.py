# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""ridge 계수 앙상블 — 런타임 비용 0으로 예측을 개선한다.

선형 모델은 여러 설정으로 학습한 계수를 평균하면 그 자체가 하나의 선형 모델이다.
따라서 추론 시간과 아티팩트 크기가 거의 그대로다. 다음을 섞어 본다.
  · 서로 다른 정규화 세기(alpha)
  · 서로 다른 특징 구성(단어/문자 n-gram 범위)
  · 부트스트랩 표본

임베딩은 앞선 실험에서 오히려 나빴으므로 쓰지 않는다.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import load_all  # noqa: E402

t0 = time.time()
DATA = load_all()
N = len(DATA.texts)
D2 = DATA.score[:, 2] - DATA.score[:, 1]
D2_POS = (D2 > 0).astype(int)
FOLD = np.arange(N) % 5
mu, sd = DATA.dense.mean(0), DATA.dense.std(0)
sd = np.where(sd > 1e-12, sd, 1.0)
DZ = (DATA.dense - mu) / sd


def build(word_ng, char_ng, min_df, max_w, max_c):
    tw = TfidfVectorizer(analyzer="word", ngram_range=word_ng, min_df=min_df,
                         max_features=max_w, sublinear_tf=True, dtype=np.float32)
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ng, min_df=min_df,
                         max_features=max_c, sublinear_tf=True, dtype=np.float32)
    return sparse.hstack([tw.fit_transform(DATA.texts), tc.fit_transform(DATA.texts),
                          sparse.csr_matrix(DZ)]).tocsr()


VIEWS = {
    "base": build((1, 2), (3, 5), 3, 60000, 120000),
    "wide-char": build((1, 2), (2, 6), 3, 60000, 160000),
    "word3": build((1, 3), (3, 5), 3, 90000, 120000),
}
print(f"[{time.time()-t0:.0f}s] 특징 구성 {len(VIEWS)}종")


def oof_pred(X, alpha, seed=None, frac=1.0):
    P = np.empty((N, 3))
    for f in range(5):
        va = FOLD == f
        idx = np.where(~va)[0]
        if frac < 1.0:
            rng = np.random.default_rng(seed)
            idx = rng.choice(idx, size=int(len(idx) * frac), replace=True)
        m = Ridge(alpha=alpha, solver="sparse_cg", random_state=0).fit(X[idx], DATA.score[idx])
        P[va] = m.predict(X[va])
    return P


def report(tag, P):
    d2p = P[:, 2] - P[:, 1]
    c2 = float(np.corrcoef(d2p, D2)[0, 1])
    auc = roc_auc_score(D2_POS, d2p)
    cs = [round(float(np.corrcoef(P[:, j], DATA.score[:, j])[0, 1]), 3) for j in range(3)]
    print(f"  {tag:46s} d2corr={c2:.4f} AUC={auc:.4f} score={cs}")
    return c2, auc


print("\n=== 단일 모델 ===")
single = {}
for name, X in VIEWS.items():
    for a in (3.0, 5.0, 8.0):
        key = f"{name}-a{a:g}"
        single[key] = oof_pred(X, a)
        report(key, np.clip(single[key], 0, 1))

print("\n=== 계수 평균 앙상블 (런타임 비용 동일) ===")
combos = {
    "base a3+a5+a8": ["base-a3", "base-a5", "base-a8"],
    "3구성 a5": ["base-a5", "wide-char-a5", "word3-a5"],
    "3구성 x 3alpha (전체 9)": list(single),
    "base+wide a3,a5": ["base-a3", "base-a5", "wide-char-a3", "wide-char-a5"],
}
best = None
for tag, keys in combos.items():
    P = np.mean([single[k] for k in keys], axis=0)
    c2, auc = report(tag, np.clip(P, 0, 1))
    if best is None or c2 > best[0]:
        best = (c2, auc, tag, keys)

print("\n=== 부트스트랩 배깅 (base 구성, alpha=5) ===")
X = VIEWS["base"]
bags = [oof_pred(X, 5.0, seed=s, frac=0.8) for s in range(5)]
for k in (2, 3, 5):
    P = np.mean(bags[:k], axis=0)
    report(f"부트스트랩 {k}개", np.clip(P, 0, 1))
P = np.mean(bags + [single["base-a5"]], axis=0)
report("부트스트랩 5 + 전체적합", np.clip(P, 0, 1))

print(f"\n  → 최고: {best[2]} (d2corr {best[0]:.4f}, AUC {best[1]:.4f})")
print(f"  기준(base-a5): d2corr {np.corrcoef(single['base-a5'][:,2]-single['base-a5'][:,1], D2)[0,1]:.4f}")
print(f"[{time.time()-t0:.0f}s] done")
