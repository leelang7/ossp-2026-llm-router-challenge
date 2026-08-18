# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 2 — 진단 + 비용 캘리브레이션 도입.

핵심 가설: 예산 초과의 원인은 (1) log→exp 변환의 Jensen 편향,
(2) 모델별 비용 스케일 미보정, (3) 라그랑지안 선택 편향(승자의 저주).
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

BINS = 4096
SEED = 0
t0 = time.time()

tr_in, tr_out, tr_rows = load_split("train")
dv_in, dv_out, dv_rows = load_split("dev")
Xtr, Xdv = build_matrix(tr_in, BINS), build_matrix(dv_in, BINS)
n_dense = Xtr.shape[1] - BINS


def targets(rows):
    g = lambda k: np.array([[r[m][k] for m in MODEL_IDS] for r in rows], dtype=float)
    return g("score"), g("cost"), g("out_tok"), g("in_tok")


Str, Ctr, Otr, Itr = targets(tr_rows)
Sdv, Cdv, Odv, Idv = targets(dv_rows)
RATE_IN = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RATE_OUT = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
cost_of = lambda i, o: (i * RATE_IN + o * RATE_OUT) / float(POLICY.token_unit)


def fit_ridge(X, Y, alpha=30.0):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z, b = (X - mu) / sd, Y.mean(0)
    n, d = Z.shape
    W = (Z.T @ np.linalg.solve(Z @ Z.T + alpha * np.eye(n), Y - b) if n <= d
         else np.linalg.solve(Z.T @ Z + alpha * np.eye(d), Z.T @ (Y - b)))
    return lambda Xn: ((Xn - mu) / sd) @ W + b


def fit_gbm(X, Y, **kw):
    p = dict(max_iter=350, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20,
             l2_regularization=1.0, early_stopping=False, random_state=SEED)
    p.update(kw)
    ms = [HistGradientBoostingRegressor(**p).fit(X, Y[:, j]) for j in range(Y.shape[1])]
    return lambda Xn: np.column_stack([m.predict(Xn) for m in ms])


def oof(fit, X, Y, folds=5):
    P = np.empty_like(Y)
    fid = np.arange(len(X)) % folds
    for f in range(folds):
        va = fid == f
        P[va] = fit(X[~va], Y[~va])(X[va])
    return P


# ---------------- 예측: score(GBM+ridge 앙상블), out_tok(GBM)
Yscore = Str
Ytok = np.log1p(Otr)
Xd, Xdd = Xtr[:, :n_dense], Xdv[:, :n_dense]

s_oof_g = oof(fit_gbm, Xd, Yscore); s_dv_g = fit_gbm(Xd, Yscore)(Xdd)
s_oof_r = oof(fit_ridge, Xtr, Yscore); s_dv_r = fit_ridge(Xtr, Yscore)(Xdv)
t_oof = oof(fit_gbm, Xd, Ytok); t_dv = fit_gbm(Xd, Ytok)(Xdd)

print(f"[{time.time()-t0:.0f}s] === 예측 진단 ===")
for j, m in enumerate(MODEL_IDS):
    for nm, p in (("gbm", s_oof_g), ("ridge", s_oof_r), ("ens", 0.6 * s_oof_g + 0.4 * s_oof_r)):
        r = np.corrcoef(p[:, j], Str[:, j])[0, 1]
        if nm == "ens":
            print(f"  score[{m:11s}] {nm:5s} OOF corr={r:.3f} mae={np.abs(np.clip(p[:,j],0,1)-Str[:,j]).mean():.3f}")

print("  --- out_tok 예측 (log1p→expm1 편향) ---")
for j, m in enumerate(MODEL_IDS):
    pred_tok = np.expm1(t_oof[:, j]).clip(0)
    real, pred = Otr[:, j].sum(), pred_tok.sum()
    print(f"  {m:11s} 실제합={real:9.0f} 예측합={pred:9.0f} 비율={pred/real:.3f} "
          f"corr={np.corrcoef(pred_tok, Otr[:,j])[0,1]:.3f}")

# ---------------- 비용 캘리브레이션 (Train OOF 기준 모델별 스케일)
def calib_scales(pred_tok_log, in_tok, real_cost):
    pt = np.expm1(pred_tok_log).clip(0)
    pc = cost_of(in_tok, pt)
    return real_cost.sum(0) / pc.sum(0), pc

in_tr = np.repeat(Itr[:, :1], 3, 1)
in_dv = np.repeat(Idv[:, :1], 3, 1)
scales, pc_tr = calib_scales(t_oof, in_tr, Ctr)
print(f"  비용 캘리브레이션 스케일 = {np.round(scales,3)}")
pc_tr = pc_tr * scales
pc_dv = cost_of(in_dv, np.expm1(t_dv).clip(0)) * scales

s_oof = np.clip(0.6 * s_oof_g + 0.4 * s_oof_r, 0, 1)
s_dv = np.clip(0.6 * s_dv_g + 0.4 * s_dv_r, 0, 1)


# ---------------- 정책 평가 (Train OOF에서 안전계수 → Dev 검증 하향)
def run(name, pS_tr, pC_tr, pS_dv, pC_dv, margin=1.0):
    grid = np.round(np.arange(0.30, 1.001, 0.01), 3)
    idx_dv, chosen, diag = {}, {}, {}
    for tier in TIERS:
        mult = float(POLICY.tiers[tier].budget_multiplier)
        pick = None
        for s in grid[::-1]:                      # 큰 안전계수부터 (점수 우선)
            idx, _ = lagrangian_select(pS_tr, pC_tr, mult, s)
            real = Ctr[np.arange(len(idx)), idx].sum() / Ctr[:, 0].sum()
            if real <= mult * margin:             # Train 실비용이 마진 안
                pick = (s, Str[np.arange(len(idx)), idx].mean(), real)
                break
        if pick is None:
            pick = (grid[0], 0.0, 0.0)
        chosen[tier] = pick[0]
        diag[tier] = pick
        idx_dv[tier], _ = lagrangian_select(pS_dv, pC_dv, mult, pick[0])
    rep = official_score(dv_in, dv_out, idx_dv)
    out = [f"{name:28s} DEV={rep['final_score'][:8]}"]
    for tier in TIERS:
        t = rep["tiers"][tier]
        out.append(f" {tier[:4]}={t['tier_score'][:6]}/{t['budget_ratio'][:5]}"
                   f"(s={chosen[tier]:.2f},tr={diag[tier][2]:.2f}){'' if t['budget_passed'] else '!OVER'}")
    print("".join(out))
    return float(rep["final_score"])


print(f"\n[{time.time()-t0:.0f}s] === 정책 평가 (margin = Train 실비용 허용 비율) ===")
for margin in (1.00, 0.95, 0.90, 0.85, 0.80):
    run(f"calib margin={margin:.2f}", s_oof, pc_tr, s_dv, pc_dv, margin)

# 보수적 비용: K1 예측에 잔차 상위 분위수 반영
resid = np.log1p(Otr) - t_oof
for q in (0.6, 0.75, 0.9):
    bump = np.exp(np.quantile(resid, q, axis=0))
    pc_tr_q = cost_of(in_tr, np.expm1(t_oof).clip(0) * bump) * scales
    pc_dv_q = cost_of(in_dv, np.expm1(t_dv).clip(0) * bump) * scales
    run(f"quantile q={q} bump={np.round(bump,2)}", s_oof, pc_tr_q, s_dv, pc_dv_q, 1.0)

print(f"\n[{time.time()-t0:.0f}s] hash-regex DEV=0.695369 | 오라클 DEV≈0.80")
