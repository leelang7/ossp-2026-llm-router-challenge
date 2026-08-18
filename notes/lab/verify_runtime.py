# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""학습(scikit-learn) 예측과 런타임(순수 파이썬 TF-IDF) 예측이 일치하는지 검증한다.

dev_compare에서 같은 정책이 CLI 경로로는 예산을 통과하고 학습 코드 경로로는
초과하는 불일치가 나왔다. 어느 쪽이 맞는지 원인을 찾는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CHALLENGE = Path(r"d:\opensource\ossp-2026-llm-router-challenge")
sys.path.insert(0, str(CHALLENGE / "src"))
sys.path.insert(0, r"d:\opensource\skt-router\lab")

from ossp_router.protocol import load_input  # noqa: E402
from routerx.router import Artifact, predict  # noqa: E402

art = Artifact(CHALLENGE / "build" / "routerx" / "art_tiered.npz")
inputs = load_input(CHALLENGE / "data" / "materialized" / "dev" / "inputs.json")
eps = inputs.episodes[:200]

s_rt, c_rt, keys = predict(eps, art)
print("=== 런타임 예측 (앞 200문항) ===")
print(f"  score  평균={s_rt.mean(0).round(4)}")
print(f"  cost   합계={c_rt.sum(0).round(4)}  비율={np.round(c_rt.sum(0)/c_rt[:,0].sum(),3)}")
print(f"  safety={art.safety}  meta={art.meta}")

# 같은 아티팩트 계수를 sklearn 파이프라인으로 재현
from scipy import sparse  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from routerx.features import dense_features, episode_text  # noqa: E402

texts = [episode_text(e) for e in eps]
vocab_w = {t: i for i, t in enumerate(art.vocab_word)}
vocab_c = {t: i for i, t in enumerate(art.vocab_char)}
tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                       vocabulary=vocab_w, dtype=np.float64)
tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                       vocabulary=vocab_c, dtype=np.float64)
tf_w.fit(texts); tf_c.fit(texts)
tf_w.idf_ = art.idf_word
tf_c.idf_ = art.idf_char
Ww = tf_w.transform(texts)
Wc = tf_c.transform(texts)
dense = np.asarray([dense_features(t, e) for t, e in zip(texts, eps)])
Dz = (dense - art.dense_mean) / art.dense_scale

# sklearn은 word/char 블록을 각각 L2 정규화하지만 학습 때는 hstack 후 정규화하지 않았다.
# 런타임은 두 블록을 합친 뒤 한 번 L2 정규화한다 → 여기서 차이가 나는지 확인.
X_sep = sparse.hstack([Ww, Wc, sparse.csr_matrix(Dz)]).tocsr()
raw_sep = X_sep @ art.coef_t + art.intercept
print("\n=== 블록별 L2 정규화(=학습 시 방식)로 계산한 예측 ===")
print(f"  score 평균={np.clip(raw_sep[:, :3], 0, 1).mean(0).round(4)}")

joint = []
for i in range(len(texts)):
    row = {}
    w = Ww[i].tocoo()
    for col, val in zip(w.col, w.data):
        row[col] = val
    c = Wc[i].tocoo()
    for col, val in zip(c.col, c.data):
        row[art.n_word + col] = val
    norm = np.sqrt(sum(v * v for v in row.values()))
    joint.append({k: v / norm for k, v in row.items()} if norm > 0 else row)
raw_joint = np.tile(art.intercept, (len(texts), 1))
for i, row in enumerate(joint):
    cols = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
    vals = np.fromiter(row.values(), dtype=np.float64, count=len(row))
    raw_joint[i] += vals @ art.coef_t[cols]
    raw_joint[i] += Dz[i] @ art.dense_coef
print("\n=== 합쳐서 한 번 L2 정규화(=런타임 방식) ===")
print(f"  score 평균={np.clip(raw_joint[:, :3], 0, 1).mean(0).round(4)}")
print(f"\n두 방식 차이(최대 절대값): {np.abs(raw_sep - raw_joint).max():.6f}")
print(f"런타임 predict()와 합산방식 차이: {np.abs(np.clip(raw_joint[:,:3],0,1) - s_rt).max():.6f}")
