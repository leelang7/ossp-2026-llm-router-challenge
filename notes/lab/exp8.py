# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 8 — '싼 K1의 함정' 검증 + 정책 개선.

가설: 비용효율(dq/dc)로 정렬하면 '싼 K1'을 먼저 고르는데,
     싼 K1 = 출력이 짧은 문제 = 이미 light로도 맞는 문제 = 이득 0.
     → 정렬 키를 dq/dc^β (β<1)로 완화하거나 dq 임계값을 두면 개선될 것.
"""
from __future__ import annotations

import heapq
import sys
import time

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from common import (  # noqa: E402
    MODEL_IDS, TIERS, POLICY, load_split, lagrangian_select, official_score,
)

t0 = time.time()
C = np.load(r"d:\opensource\skt-router\lab\pred_cache.npz")
Str, Ctr, Otr, Itr = C["Str"], C["Ctr"], C["Otr"], C["Itr"]
Sdv, Cdv, Odv, Idv = C["Sdv"], C["Cdv"], C["Odv"], C["Idv"]
tr_in, tr_out, _ = load_split("train")
dv_in, dv_out, _ = load_split("dev")
RI = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RO = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
cost_of = lambda i, o: (i * RI + o * RO) / float(POLICY.token_unit)

# ---------------- 진단: 비용과 이득의 관계
print("=== 가설 검증: 비용이 싼 문항일수록 승격 이득이 없는가? (Train) ===")
d1 = Str[:, 1] - Str[:, 0]
d2 = Str[:, 2] - Str[:, 1]
k1_rel = Ctr[:, 2] / Ctr[:, 0]          # K1 상대비용
ax_rel = Ctr[:, 1] / Ctr[:, 0]
print(f"  corr(K1비용, d2)   = {np.corrcoef(k1_rel, d2)[0,1]:+.3f}")
print(f"  corr(ax31비용, d1) = {np.corrcoef(ax_rel, d1)[0,1]:+.3f}")
qs = np.quantile(Ctr[:, 2], [0, .2, .4, .6, .8, 1.0])
print("  K1 실비용 5분위별 통계:")
for i in range(5):
    m = (Ctr[:, 2] >= qs[i]) & (Ctr[:, 2] <= qs[i + 1])
    print(f"    Q{i+1} cost[{qs[i]:.4f},{qs[i+1]:.4f}] n={m.sum():4d} "
          f"d2평균={d2[m].mean():+.3f} d2>0={np.mean(d2[m]>0):.1%} "
          f"light점수={Str[m,0].mean():.3f} K1점수={Str[m,2].mean():.3f} "
          f"이득/비용={d2[m].mean()/ (Ctr[m,2]/Ctr[m,0]).mean():.4f}")

# ---------------- 예측 준비
t_o, t_d = C["t_oof"], C["t_dev"]
bump = np.exp(np.quantile(np.log1p(Otr) - t_o, 0.3, axis=0))
in_tr, in_dv = np.repeat(Itr[:, :1], 3, 1), np.repeat(Idv[:, :1], 3, 1)
pc_o = cost_of(in_tr, np.expm1(t_o).clip(0) * bump)
scale = Ctr.sum(0) / pc_o.sum(0)
pc_o *= scale
pc_d = cost_of(in_dv, np.expm1(t_d).clip(0) * bump) * scale
S_o = np.clip(0.7 * C["s_oof_r10"] + 0.3 * C["s_oof_g"], 0, 1)
S_d = np.clip(0.7 * C["s_dev_r10"] + 0.3 * C["s_dev_g"], 0, 1)


def greedy_beta(pS, pC, cap, beta, min_gain):
    """정렬 키 = dq / dc^beta, dq < min_gain 이면 후보에서 제외."""
    n = len(pS)
    idx = np.zeros(n, dtype=int)
    spent = float(pC[:, 0].sum())
    heap = []

    def push(i):
        for m in range(3):
            dq = pS[i, m] - pS[i, idx[i]]
            dc = pC[i, m] - pC[i, idx[i]]
            if dq <= min_gain:
                continue
            if dc <= 0:
                heapq.heappush(heap, (-1e18, i, m))
            else:
                heapq.heappush(heap, (-dq / (dc ** beta), i, m))

    for i in range(n):
        push(i)
    while heap:
        _, i, m = heapq.heappop(heap)
        dq = pS[i, m] - pS[i, idx[i]]
        dc = pC[i, m] - pC[i, idx[i]]
        if dq <= min_gain:
            continue
        if spent + dc <= cap:
            spent += dc
            idx[i] = m
            push(i)
    return idx


def evaluate(pS, pC, real, inputs, outcomes, beta, min_gain, tag, verbose=True):
    grid = np.round(np.arange(0.30, 1.301, 0.005), 4)[::-1]
    idx_all, ch = {}, {}
    for tier in TIERS:
        mult = float(POLICY.tiers[tier].budget_multiplier)
        chosen = None
        for s in grid:
            cap = float(pC[:, 0].sum()) * max(1.0, mult * s)
            idx = greedy_beta(pS, pC, cap, beta, min_gain)
            if real[np.arange(len(idx)), idx].sum() / real[:, 0].sum() <= mult * 0.9985:
                chosen, ch[tier] = idx, s
                break
        idx_all[tier] = chosen if chosen is not None else np.zeros(len(pS), dtype=int)
    rep = official_score(inputs, outcomes, idx_all)
    if verbose:
        ln = [f"{tag:30s} DEV={rep['final_score'][:8]}"]
        for tier in TIERS:
            t = rep["tiers"][tier]
            mc = t["model_counts"]
            ln.append(f" {tier[:4]}={t['tier_score'][:6]}/{t['budget_ratio'][:5]}"
                      f"[{mc.get('ax31-light',0)}/{mc.get('ax31',0)}/{mc.get('axk1-think',0)}]")
        print("".join(ln))
    return float(rep["final_score"]), rep, ch


print(f"\n=== 정책 스윕 (β=비용지수, g=최소이득 임계) [{time.time()-t0:.0f}s] ===")
best = None
for beta in (1.0, 0.7, 0.5, 0.3, 0.0):
    for mg in (0.0, 0.02, 0.05):
        sc, rep, ch = evaluate(S_d, pc_d, Cdv, dv_in, dv_out, beta, mg, f"β={beta} g={mg}")
        if best is None or sc > best[0]:
            best = (sc, beta, mg, rep, ch)

print(f"\n  hash-regex=0.695369 | exp7 최고=0.697301 | 오라클≈0.80")
print(f"  BEST: β={best[1]} g={best[2]} → {best[0]:.6f}")
for tier in TIERS:
    t = best[3]["tiers"][tier]
    print(f"    {tier:9s} score={t['tier_score'][:8]} cost={t['budget_ratio'][:6]} "
          f"safety={best[4].get(tier,0):.3f} picks={t['model_counts']}")
print(f"[{time.time()-t0:.0f}s]")
