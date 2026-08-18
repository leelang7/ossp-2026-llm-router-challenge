# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 4 — 핵심 통찰 반영.

exp3 발견:
  · d1(ax31-light) 예측 불가 (Dev corr 0.014) → 예측 노이즈가 오히려 해가 됨.
    d1 평균 +0.081 > 0 이므로 "싼 문항부터 최대한 많이 ax31 승격"이 최적.
  · d2(K1-ax31)은 예측 가능 (Dev corr 0.229) → 선별 승격이 유효.
  · hash-regex가 premium에서 K1을 192개나 쓰면서 예산을 지킴 → 비용 예측 정확도가 관건.

→ score 예측을 Δ별로 shrink: d1은 강하게 죽이고(λ1↓), d2는 살린다(λ2↑).
"""
from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from common import (  # noqa: E402
    MODEL_IDS, TIERS, POLICY, build_matrix, load_split,
    lagrangian_select, official_score,
)

BINS, SEED = 4096, 0
t0 = time.time()
tr_in, tr_out, tr_rows = load_split("train")
dv_in, dv_out, dv_rows = load_split("dev")
Xtr, Xdv = build_matrix(tr_in, BINS), build_matrix(dv_in, BINS)
nd = Xtr.shape[1] - BINS
g = lambda rows, k: np.array([[r[m][k] for m in MODEL_IDS] for r in rows], dtype=float)
Str, Ctr, Otr, Itr = (g(tr_rows, k) for k in ("score", "cost", "out_tok", "in_tok"))
Sdv, Cdv, Odv, Idv = (g(dv_rows, k) for k in ("score", "cost", "out_tok", "in_tok"))
RI = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RO = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
cost_of = lambda i, o: (i * RI + o * RO) / float(POLICY.token_unit)


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
        va = fid == f
        P[va] = fit(X[~va], Y[~va])(X[va])
    return P


Xd, Xdd = Xtr[:, :nd], Xdv[:, :nd]
# score 예측 (앙상블)
s_o = 0.6 * oof(fit_gbm, Xd, Str) + 0.4 * oof(fit_ridge, Xtr, Str)
s_d = 0.6 * fit_gbm(Xd, Str)(Xdd) + 0.4 * fit_ridge(Xtr, Str)(Xdv)
# 출력 토큰 예측
Ytok = np.log1p(Otr)
t_o = 0.6 * oof(fit_gbm, Xd, Ytok) + 0.4 * oof(fit_ridge, Xtr, Ytok)
t_d = 0.6 * fit_gbm(Xd, Ytok)(Xdd) + 0.4 * fit_ridge(Xtr, Ytok)(Xdv)
resid = Ytok - t_o
in_tr, in_dv = np.repeat(Itr[:, :1], 3, 1), np.repeat(Idv[:, :1], 3, 1)
print(f"[{time.time()-t0:.0f}s] 예측 완료. out_tok DEVcorr="
      f"{[round(float(np.corrcoef(np.expm1(t_d[:,j]), Odv[:,j])[0,1]),3) for j in range(3)]}")


def make_costs(q):
    bump = np.exp(np.quantile(resid, q, axis=0))
    po = cost_of(in_tr, np.expm1(t_o).clip(0) * bump)
    pd_ = cost_of(in_dv, np.expm1(t_d).clip(0) * bump)
    sc = Ctr.sum(0) / po.sum(0)
    return po * sc, pd_ * sc


def shrink(S, lam1, lam2):
    """d1은 lam1, d2는 lam2로 축소. 기준선은 예측 평균이 아니라 전역 평균 Δ."""
    base = S[:, 0]
    d1 = S[:, 1] - S[:, 0]
    d2 = S[:, 2] - S[:, 1]
    m1, m2 = 0.081, 0.133          # Train 전역 평균 Δ
    n1 = m1 + lam1 * (d1 - d1.mean())
    n2 = m2 + lam2 * (d2 - d2.mean())
    return np.column_stack([base, base + n1, base + n1 + n2])


def evaluate(S_o, S_d, pc_o, pc_d, calib="dev", tag=""):
    grid = np.round(np.arange(0.30, 1.001, 0.005), 4)[::-1]
    idx_dv, ch = {}, {}
    for tier in TIERS:
        mult = float(POLICY.tiers[tier].budget_multiplier)
        pick = grid[-1]
        for s in grid:
            if calib == "dev":
                idx, _ = lagrangian_select(S_d, pc_d, mult, s)
                real = Cdv[np.arange(len(idx)), idx].sum() / Cdv[:, 0].sum()
            else:
                idx, _ = lagrangian_select(S_o, pc_o, mult, s)
                real = Ctr[np.arange(len(idx)), idx].sum() / Ctr[:, 0].sum()
            if real <= mult * 0.995:
                pick = s
                break
        ch[tier] = pick
        idx_dv[tier], _ = lagrangian_select(S_d, pc_d, mult, pick)
    rep = official_score(dv_in, dv_out, idx_dv)
    ok = all(rep["tiers"][t]["budget_passed"] for t in TIERS)
    return float(rep["final_score"]), rep, ch, ok


print(f"\n=== shrinkage 그리드 (Dev 보정) ===")
best = None
for q in (0.3, 0.5, 0.7):
    pc_o, pc_d = make_costs(q)
    for lam1 in (0.0, 0.15, 0.35, 0.6, 1.0):
        for lam2 in (0.6, 1.0, 1.5, 2.2):
            So = np.clip(shrink(s_o, lam1, lam2), 0, 1)
            Sd = np.clip(shrink(s_d, lam1, lam2), 0, 1)
            sc, rep, ch, ok = evaluate(So, Sd, pc_o, pc_d)
            if ok and (best is None or sc > best[0]):
                best = (sc, q, lam1, lam2, rep, ch)
print(f"  최고: DEV={best[0]:.6f} (q={best[1]} λ1={best[2]} λ2={best[3]})")
for tier in TIERS:
    t = best[4]["tiers"][tier]
    print(f"    {tier:9s} score={t['tier_score'][:8]} cost={t['budget_ratio'][:6]} "
          f"safety={best[5][tier]:.3f} picks={t['model_counts']}")

# 비교: hash-regex
print(f"\n  hash-regex  fast=0.663068/1.2360  bal=0.693750/1.9615  prem=0.740057/3.9852  최종=0.695369")
print(f"  오라클      fast=0.7531  bal=0.8040  prem=0.8585")
print(f"\n[{time.time()-t0:.0f}s] 완료")
