# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 1 — 예측기 후보 비교 (ridge vs GBM vs 앙상블), Dev 공식 점수로 평가.

안전계수는 Train OOF에서 결정하고 Dev는 평가만 (과적합 방지).
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

BINS = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
SEED = 0
rng = np.random.default_rng(SEED)

t0 = time.time()
tr_in, tr_out, tr_rows = load_split("train")
dv_in, dv_out, dv_rows = load_split("dev")
Xtr = build_matrix(tr_in, BINS)
Xdv = build_matrix(dv_in, BINS)
n_dense = Xtr.shape[1] - BINS
print(f"[{time.time()-t0:.1f}s] X train={Xtr.shape} dev={Xdv.shape} dense={n_dense} bins={BINS}")

def targets(rows):
    S = np.array([[r[m]["score"] for m in MODEL_IDS] for r in rows])
    C = np.array([[r[m]["cost"] for m in MODEL_IDS] for r in rows])
    O = np.array([[r[m]["out_tok"] for m in MODEL_IDS] for r in rows])
    I = np.array([[r[m]["in_tok"] for m in MODEL_IDS] for r in rows])
    return S, C, O, I

Str, Ctr, Otr, Itr = targets(tr_rows)
Sdv, Cdv, Odv, Idv = targets(dv_rows)

RATE_IN = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RATE_OUT = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])


def cost_from_tokens(in_tok, out_tok):
    return (in_tok * RATE_IN + out_tok * RATE_OUT) / float(POLICY.token_unit)


# ---------------------------------------------------------------- 학습기

def fit_ridge(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    b = Y.mean(0)
    Yc = Y - b
    n, d = Z.shape
    if n <= d:
        W = Z.T @ np.linalg.solve(Z @ Z.T + alpha * np.eye(n), Yc)
    else:
        W = np.linalg.solve(Z.T @ Z + alpha * np.eye(d), Z.T @ Yc)
    return lambda Xn: ((Xn - mu) / sd) @ W + b


def fit_gbm(X, Y, **kw):
    params = dict(max_iter=400, learning_rate=0.06, max_depth=None, max_leaf_nodes=15,
                  min_samples_leaf=20, l2_regularization=1.0, early_stopping=False,
                  random_state=SEED)
    params.update(kw)
    models = [HistGradientBoostingRegressor(**params).fit(X, Y[:, j]) for j in range(Y.shape[1])]
    return lambda Xn: np.column_stack([m.predict(Xn) for m in models])


def oof(fit, X, Y, folds=5):
    P = np.empty_like(Y)
    fid = np.arange(len(X)) % folds
    for f in range(folds):
        va = fid == f
        P[va] = fit(X[~va], Y[~va])(X[va])
    return P


# ---------------------------------------------------------------- 평가

def evaluate(name, pS_tr, pC_tr, pS_dv, pC_dv, safety_grid=None):
    """Train OOF로 tier별 안전계수 결정 → Dev 공식 점수."""
    if safety_grid is None:
        safety_grid = np.linspace(0.55, 1.0, 46)
    idx_tr, idx_dv, chosen = {}, {}, {}
    for tier in TIERS:
        mult = float(POLICY.tiers[tier].budget_multiplier)
        best = None
        for s in safety_grid:
            idx, _ = lagrangian_select(pS_tr, pC_tr, mult, s)
            real = Ctr[np.arange(len(idx)), idx].sum() / Ctr[:, 0].sum()
            if real > mult:      # Train 실비용 초과 → 탈락
                continue
            sc = Str[np.arange(len(idx)), idx].mean()
            if best is None or sc > best[0]:
                best = (sc, s, idx)
        if best is None:
            best = (0.0, safety_grid[0], np.zeros(len(pS_tr), dtype=int))
        chosen[tier] = best[1]
        idx_tr[tier] = best[2]
        idx_dv[tier], _ = lagrangian_select(pS_dv, pC_dv, mult, best[1])
    rep_dv = official_score(dv_in, dv_out, idx_dv)
    rep_tr = official_score(tr_in, tr_out, idx_tr)
    line = [f"{name:24s} DEV={rep_dv['final_score'][:8]} TRAIN={rep_tr['final_score'][:8]}"]
    for tier in TIERS:
        t = rep_dv["tiers"][tier]
        flag = "" if t["budget_passed"] else " !OVER"
        line.append(f"  {tier[:4]}={t['tier_score'][:6]}/{t['budget_ratio'][:5]}(s={chosen[tier]:.2f}){flag}")
    print("".join(line))
    return float(rep_dv["final_score"]), chosen


results = {}

# --- A: ridge (dense+hashed), 공식 방식 = log_cost 직접 예측
Ytr_a = np.column_stack([Str, np.log(Ctr)])
for alpha in (1.0, 10.0, 100.0):
    p_oof = oof(lambda X, Y: fit_ridge(X, Y, alpha), Xtr, Ytr_a)
    pred = fit_ridge(Xtr, Ytr_a, alpha)(Xdv)
    sc_tr, c_tr = np.clip(p_oof[:, :3], 0, 1), np.exp(p_oof[:, 3:])
    sc_dv, c_dv = np.clip(pred[:, :3], 0, 1), np.exp(pred[:, 3:])
    results[f"ridge-a{alpha:g}"] = evaluate(f"ridge a={alpha:g}", sc_tr, c_tr, sc_dv, c_dv)

# --- B: GBM (dense only) + out_tok 예측 기반 비용
Xtr_d, Xdv_d = Xtr[:, :n_dense], Xdv[:, :n_dense]
Ytr_b = np.column_stack([Str, np.log1p(Otr)])
p_oof = oof(fit_gbm, Xtr_d, Ytr_b)
pred = fit_gbm(Xtr_d, Ytr_b)(Xdv_d)
in_tr = np.repeat(Itr[:, :1], 3, axis=1)   # 실제 in_tok은 모델별 거의 동일
in_dv = np.repeat(Idv[:, :1], 3, axis=1)
sc_tr, c_tr = np.clip(p_oof[:, :3], 0, 1), cost_from_tokens(in_tr, np.expm1(p_oof[:, 3:]).clip(0))
sc_dv, c_dv = np.clip(pred[:, :3], 0, 1), cost_from_tokens(in_dv, np.expm1(pred[:, 3:]).clip(0))
results["gbm-dense"] = evaluate("GBM dense (tok-cost)", sc_tr, c_tr, sc_dv, c_dv)

# --- C: GBM(dense) + ridge(hashed) 앙상블
p_oof_r = oof(lambda X, Y: fit_ridge(X, Y, 10.0), Xtr, Ytr_a)
pred_r = fit_ridge(Xtr, Ytr_a, 10.0)(Xdv)
for w in (0.3, 0.5, 0.7):
    s_tr = np.clip(w * p_oof[:, :3] + (1 - w) * p_oof_r[:, :3], 0, 1)
    s_dv = np.clip(w * pred[:, :3] + (1 - w) * pred_r[:, :3], 0, 1)
    ct = np.maximum(c_tr, 0.5 * c_tr + 0.5 * np.exp(p_oof_r[:, 3:]))
    cd = np.maximum(c_dv, 0.5 * c_dv + 0.5 * np.exp(pred_r[:, 3:]))
    results[f"ens-w{w}"] = evaluate(f"ENS gbm{w:.1f}+ridge", s_tr, ct, s_dv, cd)

print(f"\n[{time.time()-t0:.1f}s] baseline hash-regex DEV=0.695369")
best = max(results.items(), key=lambda kv: kv[1][0])
print(f"BEST: {best[0]} = {best[1][0]:.6f}")
