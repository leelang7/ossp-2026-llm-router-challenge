# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""임베딩이 예측을 실제로 개선하는지 검증한다.

현재 병목은 K1 승격 이득(d2) 예측이다(상관 0.425). 상한 분석상 d2를 완벽히 알면
0.7248, 현재는 0.665다. 임베딩으로 d2 상관을 올릴 수 있으면 그 격차의 일부를 얻는다.

여기서는 GPU로 임베딩을 뽑아 예측 성능만 본다. 개선이 확인되면 그때 ONNX 양자화와
런타임 통합을 검토한다(등급당 90초, CPU 2코어 제약).
"""
from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import load_all  # noqa: E402

MODELS = [
    ("e5-small", "intfloat/multilingual-e5-small", "query: "),
    ("mini-L12", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
]
MAX_LEN = 192

t0 = time.time()
DATA = load_all()
N = len(DATA.texts)
D2 = DATA.score[:, 2] - DATA.score[:, 1]
D2_POS = (D2 > 0).astype(int)
FOLD = np.arange(N) % 5
print(f"[{time.time()-t0:.0f}s] n={N}")


def oof(fit, X, Y):
    P = np.empty((X.shape[0], Y.shape[1]))
    for f in range(5):
        va = FOLD == f
        P[va] = fit(X[~va], Y[~va])(X[va])
    return P


def ridge_fit(alpha):
    def f(A, B):
        m = Ridge(alpha=alpha, random_state=0).fit(A, B)
        return lambda Z: m.predict(Z)
    return f


def gbm_fit(**kw):
    def f(A, B):
        p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=15,
                 l2_regularization=1.0, early_stopping=False, random_state=0)
        p.update(kw)
        ms = [HistGradientBoostingRegressor(**p).fit(A, B[:, j]) for j in range(B.shape[1])]
        return lambda Z: np.column_stack([m.predict(Z) for m in ms])
    return f


def report(tag, pred_score):
    d2p = pred_score[:, 2] - pred_score[:, 1]
    c2 = float(np.corrcoef(d2p, D2)[0, 1])
    auc = roc_auc_score(D2_POS, d2p)
    cs = [round(float(np.corrcoef(pred_score[:, j], DATA.score[:, j])[0, 1]), 3) for j in range(3)]
    print(f"  {tag:42s} d2corr={c2:.3f} AUC={auc:.3f} score={cs}")
    return c2


print("\n=== 기준: TF-IDF ridge (현재 구성) ===")
from scipy import sparse  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

tw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=60000,
                     sublinear_tf=True, dtype=np.float32)
tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000,
                     sublinear_tf=True, dtype=np.float32)
mu, sd = DATA.dense.mean(0), DATA.dense.std(0)
sd = np.where(sd > 1e-12, sd, 1.0)
DZ = (DATA.dense - mu) / sd
Xtf = sparse.hstack([tw.fit_transform(DATA.texts), tc.fit_transform(DATA.texts),
                     sparse.csr_matrix(DZ)]).tocsr()


def ridge_sparse(alpha):
    def f(A, B):
        m = Ridge(alpha=alpha, solver="sparse_cg", random_state=0).fit(A, B)
        return lambda Z: m.predict(Z)
    return f


base = oof(ridge_sparse(5.0), Xtf, DATA.score)
report("TF-IDF ridge a=5", np.clip(base, 0, 1))

for name, repo, prefix in MODELS:
    print(f"\n=== {name} ({repo}) ===")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(repo, device="cuda")
        model.max_seq_length = MAX_LEN
        texts = [prefix + t for t in DATA.texts]
        s = time.time()
        emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        print(f"  임베딩 {emb.shape} 추출 {time.time()-s:.1f}s (GPU)")
        np.save(rf"d:\opensource\skt-router\lab\emb_{name}.npy", emb)
    except Exception as exc:      # noqa: BLE001
        print(f"  실패: {type(exc).__name__}: {exc}")
        continue

    e_ridge = oof(ridge_fit(1.0), emb, DATA.score)
    report(f"{name} ridge", np.clip(e_ridge, 0, 1))
    e_gbm = oof(gbm_fit(), emb, DATA.score)
    report(f"{name} GBM", np.clip(e_gbm, 0, 1))
    both = np.hstack([emb, DZ])
    e_both = oof(ridge_fit(1.0), both, DATA.score)
    report(f"{name}+dense ridge", np.clip(e_both, 0, 1))
    for w in (0.3, 0.5, 0.7):
        mix = w * base + (1 - w) * e_both
        report(f"TFIDF{w:.1f} + {name}+dense{1-w:.1f}", np.clip(mix, 0, 1))
    mix3 = 0.4 * base + 0.3 * e_both + 0.3 * e_gbm
    report(f"3방향 앙상블 ({name})", np.clip(mix3, 0, 1))

print(f"\n[{time.time()-t0:.0f}s] done")
