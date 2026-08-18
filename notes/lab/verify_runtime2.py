# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""수정 후 재검증 — 런타임 예측이 학습 파이프라인과 일치하는가.

블록별 L2 정규화로 고친 뒤, sklearn 경로와 순수 파이썬 경로의 예측을 직접 비교한다.
word/char 각 블록의 재현 정확도도 따로 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

CHALLENGE = Path(r"d:\opensource\ossp-2026-llm-router-challenge")
sys.path.insert(0, str(CHALLENGE / "src"))

from ossp_router.protocol import load_input  # noqa: E402
from routerx.features import (  # noqa: E402
    char_wb_ngrams, dense_features, episode_text, l2_normalize, tfidf_row, word_ngrams,
)
from routerx.router import Artifact, predict  # noqa: E402

art = Artifact(CHALLENGE / "build" / "routerx" / "art_tiered.npz")
eps = load_input(CHALLENGE / "data" / "materialized" / "dev" / "inputs.json").episodes[:300]
texts = [episode_text(e) for e in eps]

# --- sklearn 경로 (학습과 동일한 방식)
vw = {t: i for i, t in enumerate(art.vocab_word)}
vc = {t: i for i, t in enumerate(art.vocab_char)}
tw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                     vocabulary=vw, dtype=np.float64).fit(texts)
tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                     vocabulary=vc, dtype=np.float64).fit(texts)
tw.idf_, tc.idf_ = art.idf_word, art.idf_char
Ww, Wc = tw.transform(texts), tc.transform(texts)

# --- 순수 파이썬 경로
py_w = [l2_normalize(tfidf_row(word_ngrams(t), art.vocab_word, art.idf_word)) for t in texts]
py_c = [l2_normalize(tfidf_row(char_wb_ngrams(t), art.vocab_char, art.idf_char)) for t in texts]


def block_diff(sk_matrix, py_rows, name):
    worst_val, worst_nnz = 0.0, 0
    for i in range(len(texts)):
        sk = {int(c): float(v) for c, v in zip(sk_matrix[i].tocoo().col, sk_matrix[i].tocoo().data)}
        py = {int(k): float(v) for k, v in py_rows[i].items()}
        keys = set(sk) | set(py)
        worst_val = max(worst_val, max((abs(sk.get(k, 0.0) - py.get(k, 0.0)) for k in keys), default=0.0))
        worst_nnz = max(worst_nnz, abs(len(sk) - len(py)))
    print(f"  {name:6s} 값 최대오차={worst_val:.3e}  nnz 최대차이={worst_nnz}")
    return worst_val


print("=== 블록별 TF-IDF 재현 정확도 (300문항) ===")
dw = block_diff(Ww, py_w, "word")
dc = block_diff(Wc, py_c, "char")

# --- 최종 예측 비교
dense = np.asarray([dense_features(t, e) for t, e in zip(texts, eps)])
Dz = (dense - art.dense_mean) / art.dense_scale
X = sparse.hstack([Ww, Wc, sparse.csr_matrix(Dz)]).tocsr()
raw_sk = X @ art.coef_t + art.intercept
s_sk = np.clip(raw_sk[:, :3], 0, 1)

s_rt, c_rt, _ = predict(eps, art)
print("\n=== 최종 예측 비교 ===")
print(f"  score 최대오차 = {np.abs(s_sk - s_rt).max():.3e}")
print(f"  sklearn score 평균 = {s_sk.mean(0).round(5)}")
print(f"  런타임  score 평균 = {s_rt.mean(0).round(5)}")
ok = np.abs(s_sk - s_rt).max() < 1e-6
print(f"\n  → {'PASS: 런타임이 학습 파이프라인과 일치' if ok else 'FAIL: 여전히 불일치'}")
