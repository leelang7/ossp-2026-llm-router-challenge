# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 5 — 병목 판별: 예측기 vs 정책.

공식 hash-regex의 예측을 그대로 우리 평가 루프에 넣어 비교한다.
  · 공식예측 + 우리정책 ≈ 0.695 이면 → 병목은 우리 '예측기'
  · 공식예측 + 우리정책 > 0.695 이면 → 우리 '정책'이 더 좋음
"""
from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from common import (  # noqa: E402
    MODEL_IDS, TIERS, POLICY, CHALLENGE, build_matrix, load_split,
    lagrangian_select, official_score,
)
import hash_regex  # noqa: E402

BINS, SEED = 4096, 0
t0 = time.time()
tr_in, tr_out, tr_rows = load_split("train")
dv_in, dv_out, dv_rows = load_split("dev")
g = lambda rows, k: np.array([[r[m][k] for m in MODEL_IDS] for r in rows], dtype=float)
Str, Ctr, Otr, Itr = (g(tr_rows, k) for k in ("score", "cost", "out_tok", "in_tok"))
Sdv, Cdv, Odv, Idv = (g(dv_rows, k) for k in ("score", "cost", "out_tok", "in_tok"))

# ---------------- 공식 hash-regex 예측 추출
art = hash_regex.parse_artifact(
    __import__("json").load(open(CHALLENGE / "baselines" / "hash-regex-public.v1.json", encoding="utf-8"))
)
def official_pred(inputs):
    S, C = [], []
    for ep in inputs.episodes:
        s, c = hash_regex.predict_episode(ep, art)
        S.append([s[m] for m in MODEL_IDS]); C.append([c[m] for m in MODEL_IDS])
    return np.array(S), np.array(C)

oS_tr, oC_tr = official_pred(tr_in)
oS_dv, oC_dv = official_pred(dv_in)
print(f"[{time.time()-t0:.0f}s] 공식 예측 추출 완료")


def evaluate(S_d, pc_d, tag):
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
    ln = [f"{tag:34s} DEV={rep['final_score'][:8]}"]
    for tier in TIERS:
        t = rep["tiers"][tier]
        ln.append(f" {tier[:4]}={t['tier_score'][:6]}/{t['budget_ratio'][:5]}")
    print("".join(ln))
    return float(rep["final_score"])


print("\n=== A. 공식 예측 + 우리 정책 ===")
evaluate(oS_dv, oC_dv, "공식score + 공식cost")

# ---------------- 우리 예측
Xtr, Xdv = build_matrix(tr_in, BINS), build_matrix(dv_in, BINS)
nd = Xtr.shape[1] - BINS
Xd, Xdd = Xtr[:, :nd], Xdv[:, :nd]


def fit_ridge(X, Y, alpha=30.0):
    mu, sd = X.mean(0), X.std(0); sd = np.where(sd > 1e-12, sd, 1.0)
    Z, b = (X - mu) / sd, Y.mean(0); n, d = Z.shape
    W = (Z.T @ np.linalg.solve(Z @ Z.T + alpha * np.eye(n), Y - b) if n <= d
         else np.linalg.solve(Z.T @ Z + alpha * np.eye(d), Z.T @ (Y - b)))
    return lambda Xn: ((Xn - mu) / sd) @ W + b


def fit_gbm(X, Y, **kw):
    p = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20,
             l2_regularization=1.0, early_stopping=False, random_state=SEED)
    p.update(kw)
    ms = [HistGradientBoostingRegressor(**p).fit(X, Y[:, j]) for j in range(Y.shape[1])]
    return lambda Xn: np.column_stack([m.predict(Xn) for m in ms])


def oof(fit, X, Y, folds=5):
    P = np.empty_like(Y); fid = np.arange(len(X)) % folds
    for f in range(folds):
        va = fid == f; P[va] = fit(X[~va], Y[~va])(X[va])
    return P


s_o = 0.6 * oof(fit_gbm, Xd, Str) + 0.4 * oof(fit_ridge, Xtr, Str)
s_d = 0.6 * fit_gbm(Xd, Str)(Xdd) + 0.4 * fit_ridge(Xtr, Str)(Xdv)
Ytok = np.log1p(Otr)
t_o = 0.6 * oof(fit_gbm, Xd, Ytok) + 0.4 * oof(fit_ridge, Xtr, Ytok)
t_d = 0.6 * fit_gbm(Xd, Ytok)(Xdd) + 0.4 * fit_ridge(Xtr, Ytok)(Xdv)
RI = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RO = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
cost_of = lambda i, o: (i * RI + o * RO) / float(POLICY.token_unit)
in_tr, in_dv = np.repeat(Itr[:, :1], 3, 1), np.repeat(Idv[:, :1], 3, 1)
resid = Ytok - t_o
bump = np.exp(np.quantile(resid, 0.3, axis=0))
pc_o = cost_of(in_tr, np.expm1(t_o).clip(0) * bump)
sc = Ctr.sum(0) / pc_o.sum(0)
pc_dv_ours = cost_of(in_dv, np.expm1(t_d).clip(0) * bump) * sc

print("\n=== B. 교차 조합 (병목 판별) ===")
evaluate(np.clip(s_d, 0, 1), pc_dv_ours, "우리score + 우리cost")
evaluate(np.clip(s_d, 0, 1), oC_dv, "우리score + 공식cost")
evaluate(oS_dv, pc_dv_ours, "공식score + 우리cost")

print("\n=== C. 예측 품질 직접 비교 (Dev) ===")
d1_dv = Sdv[:, 1] - Sdv[:, 0]
d2_dv = Sdv[:, 2] - Sdv[:, 1]
for nm, S in (("공식", oS_dv), ("우리", s_d)):
    for j, m in enumerate(MODEL_IDS):
        pass
    print(f"  [{nm}] score corr per model = "
          f"{[round(float(np.corrcoef(S[:,j], Sdv[:,j])[0,1]),3) for j in range(3)]}")
    pd1, pd2 = S[:, 1] - S[:, 0], S[:, 2] - S[:, 1]
    auc1 = roc_auc_score((d1_dv > 0).astype(int), pd1) if (d1_dv > 0).any() else float("nan")
    auc2 = roc_auc_score((d2_dv > 0).astype(int), pd2) if (d2_dv > 0).any() else float("nan")
    print(f"       Δ랭킹 AUC: d1>0={auc1:.3f}  d2>0={auc2:.3f}   "
          f"Δcorr d1={np.corrcoef(pd1,d1_dv)[0,1]:.3f} d2={np.corrcoef(pd2,d2_dv)[0,1]:.3f}")
for nm, C in (("공식", oC_dv), ("우리", pc_dv_ours)):
    rel = C / C[:, :1]
    real_rel = Cdv / Cdv[:, :1]
    print(f"  [{nm}] cost corr per model = "
          f"{[round(float(np.corrcoef(C[:,j], Cdv[:,j])[0,1]),3) for j in range(3)]}"
          f"  총합비={np.round(C.sum(0)/Cdv.sum(0),3)}")

print(f"\n[{time.time()-t0:.0f}s] hash-regex 공식 DEV=0.695369")
