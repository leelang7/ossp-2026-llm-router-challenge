# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""빠른 CV — 어휘는 전체에서 한 번만 만들고(비지도, 라벨 누출 없음) 회귀만 fold별 학습.

정직한 fold별 어휘 재적합(cv.py)보다 낙관적일 수 있으나 설정 간 상대 비교에는 충분하고
반복 속도가 10배 빠르다. 최종 후보만 cv.py로 다시 검증한다.

핵심 실험: 비용을 평균이 아닌 상위 분위수로 예측하면(분위수 회귀)
출력 토큰이 폭발하는 문항을 자동으로 피해 예산 초과를 막을 수 있는가?
"""
from __future__ import annotations

import sys
import time
from typing import Dict

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import (  # noqa: E402
    Config, MODEL_IDS, MULT, TIERS, WEIGHT, load_all, select_batch,
)

RATE_IN = np.array([float(__import__("cv").POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RATE_OUT = np.array([float(__import__("cv").POLICY.models[m].output_token_rate) for m in MODEL_IDS])
UNIT = float(__import__("cv").POLICY.token_unit)
N_M = len(MODEL_IDS)

t0 = time.time()
DATA = load_all()
CFG = Config()
print(f"[{time.time()-t0:.0f}s] n={len(DATA.texts)}")

tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=60000,
                       sublinear_tf=True, dtype=np.float32)
tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000,
                       sublinear_tf=True, dtype=np.float32)
Ww, Wc = tf_w.fit_transform(DATA.texts), tf_c.fit_transform(DATA.texts)
dmu, dsd = DATA.dense.mean(0), DATA.dense.std(0)
dsd = np.where(dsd > 1e-12, dsd, 1.0)
DZ = (DATA.dense - dmu) / dsd
X = sparse.hstack([Ww, Wc, sparse.csr_matrix(DZ)]).tocsr()
print(f"[{time.time()-t0:.0f}s] X={X.shape}")

SCORE, COST, OTOK, ITOK, KEYS = DATA.score, DATA.cost, DATA.out_tok, DATA.in_tok, DATA.keys


def folds_of(n_folds: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    fold = np.empty(X.shape[0], dtype=int)
    for i, idx in enumerate(order):
        fold[idx] = i % n_folds
    return fold


def cost_from(pred_out, pred_in, scale):
    c = (np.repeat(pred_in, N_M, axis=1) * RATE_IN + pred_out * RATE_OUT) / UNIT
    return np.maximum(c * scale, 1e-9)


def run(cost_mode: str, margins: Dict[str, float], n_folds: int = 5, seed: int = 0,
        quantile: float = 0.8, tag: str = ""):
    fold = folds_of(n_folds, seed)
    finals, fails = [], {t: 0 for t in TIERS}
    usage = {t: [] for t in TIERS}
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        Ytr = np.hstack([SCORE[tr], np.log1p(OTOK[tr]), np.log1p(ITOK[tr][:, :1])])
        ridge = Ridge(alpha=CFG.alpha, solver="sparse_cg", random_state=0).fit(X[tr], Ytr)
        oof = np.empty_like(Ytr)
        inner = np.arange(tr.sum()) % 5
        Xtr = X[tr]
        for g in range(5):
            va = inner == g
            sub = Ridge(alpha=CFG.alpha, solver="sparse_cg", random_state=0).fit(Xtr[~va], Ytr[~va])
            oof[va] = sub.predict(Xtr[va])
        raw_te = ridge.predict(X[te])

        s_oof = np.clip(oof[:, :N_M], 0, 1)
        s_te = np.clip(raw_te[:, :N_M], 0, 1)
        in_oof = np.expm1(oof[:, 2 * N_M:2 * N_M + 1]).clip(1)
        in_te = np.expm1(raw_te[:, 2 * N_M:2 * N_M + 1]).clip(1)

        if cost_mode == "global_bump":
            bump = np.exp(np.quantile(np.log1p(OTOK[tr]) - oof[:, N_M:2 * N_M], CFG.cost_quantile, axis=0))
            o_oof = np.expm1(oof[:, N_M:2 * N_M]).clip(0) * bump
            o_te = np.expm1(raw_te[:, N_M:2 * N_M]).clip(0) * bump
        elif cost_mode == "quantile_gbm":
            # 문항별 상위 분위수 출력 토큰을 직접 회귀 → 폭발 위험 문항의 비용이 커진다
            Dtr, Dte = DZ[tr], DZ[te]
            o_oof = np.empty((tr.sum(), N_M))
            o_te = np.empty((te.sum(), N_M))
            for j in range(N_M):
                q = HistGradientBoostingRegressor(
                    loss="quantile", quantile=quantile, max_iter=250, learning_rate=0.06,
                    max_leaf_nodes=15, min_samples_leaf=20, early_stopping=False, random_state=0)
                q.fit(Dtr, np.log1p(OTOK[tr][:, j]))
                o_oof[:, j] = np.expm1(q.predict(Dtr)).clip(0)
                o_te[:, j] = np.expm1(q.predict(Dte)).clip(0)
        else:
            raise ValueError(cost_mode)

        c_tmp = cost_from(o_oof, in_oof, np.ones(N_M))
        scale = COST[tr].sum(0) / c_tmp.sum(0)
        c_oof = cost_from(o_oof, in_oof, scale)
        c_te = cost_from(o_te, in_te, scale)

        final = 0.0
        for tier in TIERS:
            safety = 0.30
            for s in np.round(np.arange(0.30, 1.401, 0.005), 4)[::-1]:
                sel = select_batch(s_oof, c_oof, MULT[tier], float(s), KEYS[tr])
                r = COST[tr][np.arange(len(sel)), sel].sum() / COST[tr][:, 0].sum()
                if r <= MULT[tier] * margins[tier]:
                    safety = float(s)
                    break
            sel_te = select_batch(s_te, c_te, MULT[tier], safety, KEYS[te])
            n = len(sel_te)
            ratio = COST[te][np.arange(n), sel_te].sum() / COST[te][:, 0].sum()
            ok = ratio <= MULT[tier] + 1e-12
            pts = SCORE[te][np.arange(n), sel_te].mean() if ok else 0.0
            if not ok:
                fails[tier] += 1
            usage[tier].append(ratio / MULT[tier])
            final += WEIGHT[tier] * pts
        finals.append(final)
    tf = sum(fails.values())
    use = "  ".join(f"{t[:4]}={np.mean(usage[t]):.2f}/{np.max(usage[t]):.2f}" for t in TIERS)
    print(f"{tag:38s} CV={np.mean(finals):.6f}  실패={tf:2d}  사용률 {use}"
          f"{'  ' + str({k: v for k, v in fails.items() if v}) if tf else ''}")
    return float(np.mean(finals)), tf


if __name__ == "__main__":
    print("\n=== A. 전역 bump (현재 방식) — 마진 스윕 ===")
    for m in (0.95, 0.90, 0.85, 0.80, 0.75):
        run("global_bump", {t: m for t in TIERS}, tag=f"global_bump margin={m:.2f}")
    print("\n=== B. 분위수 회귀 비용 (문항별 상위 분위수) ===")
    for q in (0.7, 0.8, 0.9):
        for m in (0.95, 0.90, 0.85):
            run("quantile_gbm", {t: m for t in TIERS}, quantile=q,
                tag=f"quantile q={q} margin={m:.2f}")
    print(f"\n[{time.time()-t0:.0f}s] done")
