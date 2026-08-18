# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 3 — 타겟 구조 분석 + Δ(차이) 직접 예측 / 분류 접근.

결정에 필요한 건 score 절대값이 아니라 승격 이득 Δ다.
"""
from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

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
GENS = np.array([[r[m]["gens"] for m in MODEL_IDS] for r in tr_rows])

print("=== score 값 분포 ===")
for j, m in enumerate(MODEL_IDS):
    c = Counter(np.round(Str[:, j], 4))
    top = sorted(c.items(), key=lambda kv: -kv[1])[:6]
    print(f"  {m:11s} 고유값={len(c):3d} 상위={[(f'{v:g}', n) for v, n in top]}")
print(f"  num_generations 분포={Counter(GENS[:,0].tolist()).most_common(5)}")

d1, d2 = Str[:, 1] - Str[:, 0], Str[:, 2] - Str[:, 1]
print(f"\n=== 승격 이득 Δ (Train) ===")
print(f"  d1(ax31-light): mean={d1.mean():+.3f} >0={np.mean(d1>0):.1%} <0={np.mean(d1<0):.1%} =0={np.mean(d1==0):.1%}")
print(f"  d2(K1-ax31)   : mean={d2.mean():+.3f} >0={np.mean(d2>0):.1%} <0={np.mean(d2<0):.1%} =0={np.mean(d2==0):.1%}")
print(f"  비용비: ax31/light={(Ctr[:,1]/Ctr[:,0]).mean():.2f}  K1/light={(Ctr[:,2]/Ctr[:,0]).mean():.2f}")


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

# --- (a) 개별 score 예측 → Δ 유도
sa_o = oof(fit_gbm, Xd, Str); sa_d = fit_gbm(Xd, Str)(Xdd)
ra_o = oof(fit_ridge, Xtr, Str); ra_d = fit_ridge(Xtr, Str)(Xdv)
ea_o, ea_d = 0.6 * sa_o + 0.4 * ra_o, 0.6 * sa_d + 0.4 * ra_d

# --- (b) Δ 직접 예측
D = np.column_stack([d1, d2])
sb_o = oof(fit_gbm, Xd, D); sb_d = fit_gbm(Xd, D)(Xdd)
rb_o = oof(fit_ridge, Xtr, D); rb_d = fit_ridge(Xtr, D)(Xdv)
eb_o, eb_d = 0.6 * sb_o + 0.4 * rb_o, 0.6 * sb_d + 0.4 * rb_d

Ddv = np.column_stack([Sdv[:, 1] - Sdv[:, 0], Sdv[:, 2] - Sdv[:, 1]])
print(f"\n=== Δ 예측 정확도 (OOF / Dev) ===")
for k, nm in ((0, "d1"), (1, "d2")):
    ind_o, ind_d = ea_o[:, k + 1] - ea_o[:, k], ea_d[:, k + 1] - ea_d[:, k]
    print(f"  {nm} 개별차분 OOFcorr={np.corrcoef(ind_o, D[:,k])[0,1]:.3f} DEVcorr={np.corrcoef(ind_d, Ddv[:,k])[0,1]:.3f}")
    print(f"  {nm} 직접예측 OOFcorr={np.corrcoef(eb_o[:,k], D[:,k])[0,1]:.3f} DEVcorr={np.corrcoef(eb_d[:,k], Ddv[:,k])[0,1]:.3f}")

# --- 비용 예측 (공통)
def fit_tok():
    Y = np.log1p(Otr)
    o, d = oof(fit_gbm, Xd, Y), fit_gbm(Xd, Y)(Xdd)
    return o, d


t_o, t_d = fit_tok()
RI = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RO = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
cost_of = lambda i, o: (i * RI + o * RO) / float(POLICY.token_unit)
in_tr, in_dv = np.repeat(Itr[:, :1], 3, 1), np.repeat(Idv[:, :1], 3, 1)
resid = np.log1p(Otr) - t_o


def costs(bump_q):
    bump = np.exp(np.quantile(resid, bump_q, axis=0))
    pc_o = cost_of(in_tr, np.expm1(t_o).clip(0) * bump)
    pc_d = cost_of(in_dv, np.expm1(t_d).clip(0) * bump)
    sc = Ctr.sum(0) / pc_o.sum(0)
    return pc_o * sc, pc_d * sc


def run(name, S_o, S_d, q=0.75, calib_on="dev"):
    pc_o, pc_d = costs(q)
    grid = np.round(np.arange(0.30, 1.01, 0.01), 3)[::-1]
    idx_dv, ch = {}, {}
    for tier in TIERS:
        mult = float(POLICY.tiers[tier].budget_multiplier)
        pick = grid[-1]
        for s in grid:
            if calib_on == "dev":     # 공식 baseline과 동일 조건(Dev로 안전계수 보정)
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
    ln = [f"{name:30s} DEV={rep['final_score'][:8]}"]
    for tier in TIERS:
        t = rep["tiers"][tier]
        ln.append(f" {tier[:4]}={t['tier_score'][:6]}/{t['budget_ratio'][:5]}{'' if t['budget_passed'] else '!OVER'}")
    print("".join(ln))
    return float(rep["final_score"])


print(f"\n[{time.time()-t0:.0f}s] === 정책 평가 (Dev 보정 = 공식 baseline과 동일 조건) ===")
S_ind_o, S_ind_d = np.clip(ea_o, 0, 1), np.clip(ea_d, 0, 1)
run("(a) 개별 score q=0.75", S_ind_o, S_ind_d, 0.75)
run("(a) 개별 score q=0.5", S_ind_o, S_ind_d, 0.5)

# Δ 예측을 절대 score로 재구성 (light 기준선 + 누적 Δ)
base_o = np.clip(ea_o[:, 0], 0, 1)
base_d = np.clip(ea_d[:, 0], 0, 1)
S_del_o = np.column_stack([base_o, base_o + eb_o[:, 0], base_o + eb_o[:, 0] + eb_o[:, 1]])
S_del_d = np.column_stack([base_d, base_d + eb_d[:, 0], base_d + eb_d[:, 0] + eb_d[:, 1]])
run("(b) Δ직접 q=0.75", np.clip(S_del_o, 0, 1), np.clip(S_del_d, 0, 1), 0.75)
run("(b) Δ직접 q=0.5", np.clip(S_del_o, 0, 1), np.clip(S_del_d, 0, 1), 0.5)

# (c) 혼합: 개별 + Δ 평균
S_mix_o = np.clip(0.5 * S_ind_o + 0.5 * np.clip(S_del_o, 0, 1), 0, 1)
S_mix_d = np.clip(0.5 * S_ind_d + 0.5 * np.clip(S_del_d, 0, 1), 0, 1)
run("(c) 혼합 q=0.75", S_mix_o, S_mix_d, 0.75)
run("(c) 혼합 q=0.5", S_mix_o, S_mix_d, 0.5)

print(f"\n[{time.time()-t0:.0f}s] hash-regex DEV=0.695369 | 오라클 DEV≈0.80")
