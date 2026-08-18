# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 7 — 정책 개선: 라그랑지안 + 잔여예산 그리디 fill.

exp6 관찰: 예측 품질은 공식을 앞서는데 premium 예산을 3.69만 써서 손해(공식 3.99).
라그랑지안은 penalty 이산성 때문에 예산을 정확히 소진하지 못한다.
→ 선택 후 잔여 예산을 이득/비용비 순으로 채우는 그리디 단계를 추가.
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
in_tr, in_dv = np.repeat(Itr[:, :1], 3, 1), np.repeat(Idv[:, :1], 3, 1)

t_o, t_d = C["t_oof"], C["t_dev"]
bump = np.exp(np.quantile(np.log1p(Otr) - t_o, 0.3, axis=0))
pc_o = cost_of(in_tr, np.expm1(t_o).clip(0) * bump)
scale = Ctr.sum(0) / pc_o.sum(0)
pc_o *= scale
pc_d = cost_of(in_dv, np.expm1(t_d).clip(0) * bump) * scale


def greedy_fill(idx, pS, pC, cap_pred):
    """현재 선택에서 잔여 예측 예산을 이득/비용비 순으로 채운다."""
    n = len(idx)
    idx = idx.copy()
    spent = float(pC[np.arange(n), idx].sum())
    heap = []
    for i in range(n):
        for m in range(3):
            dq = pS[i, m] - pS[i, idx[i]]
            dc = pC[i, m] - pC[i, idx[i]]
            if dq > 0 and dc > 0:
                heapq.heappush(heap, (-dq / dc, i, m))
            elif dq > 0 and dc <= 0:
                heapq.heappush(heap, (-1e18, i, m))
    while heap:
        _, i, m = heapq.heappop(heap)
        dq = pS[i, m] - pS[i, idx[i]]
        dc = pC[i, m] - pC[i, idx[i]]
        if dq <= 0:
            continue
        if spent + dc <= cap_pred:
            spent += dc
            idx[i] = m
            for mm in range(3):
                dq2 = pS[i, mm] - pS[i, idx[i]]
                dc2 = pC[i, mm] - pC[i, idx[i]]
                if dq2 > 0 and dc2 > 0:
                    heapq.heappush(heap, (-dq2 / dc2, i, mm))
    return idx, spent


def select(pS, pC, mult, safety, mode):
    if mode == "lag":
        return lagrangian_select(pS, pC, mult, safety)[0]
    cap = float(pC[:, 0].sum()) * max(1.0, mult * safety)
    if mode == "greedy":
        return greedy_fill(np.zeros(len(pS), dtype=int), pS, pC, cap)[0]
    idx = lagrangian_select(pS, pC, mult, safety)[0]
    return greedy_fill(idx, pS, pC, cap)[0]


def evaluate(pS_d, pC_d, mode, tag, real=None, inputs=None, outcomes=None):
    real = Cdv if real is None else real
    inputs = dv_in if inputs is None else inputs
    outcomes = dv_out if outcomes is None else outcomes
    grid = np.round(np.arange(0.30, 1.201, 0.005), 4)[::-1]
    idx_all, ch = {}, {}
    for tier in TIERS:
        mult = float(POLICY.tiers[tier].budget_multiplier)
        pick, chosen_idx = grid[-1], None
        for s in grid:
            idx = select(pS_d, pC_d, mult, s, mode)
            if real[np.arange(len(idx)), idx].sum() / real[:, 0].sum() <= mult * 0.9985:
                pick, chosen_idx = s, idx
                break
        if chosen_idx is None:
            chosen_idx = np.zeros(len(pS_d), dtype=int)
        ch[tier] = pick
        idx_all[tier] = chosen_idx
    rep = official_score(inputs, outcomes, idx_all)
    ln = [f"{tag:34s} DEV={rep['final_score'][:8]}"]
    for tier in TIERS:
        t = rep["tiers"][tier]
        ln.append(f" {tier[:4]}={t['tier_score'][:6]}/{t['budget_ratio'][:5]}")
    print("".join(ln))
    return float(rep["final_score"]), rep, ch


print("=== 예측기 × 정책 조합 ===")
preds = {
    "ridge3": (C["s_oof_r3"], C["s_dev_r3"]),
    "ridge10": (C["s_oof_r10"], C["s_dev_r10"]),
    "ridge30": (C["s_oof_r30"], C["s_dev_r30"]),
    "gbm": (C["s_oof_g"], C["s_dev_g"]),
    "ens5": (0.5 * C["s_oof_r10"] + 0.5 * C["s_oof_g"], 0.5 * C["s_dev_r10"] + 0.5 * C["s_dev_g"]),
    "ens7": (0.7 * C["s_oof_r10"] + 0.3 * C["s_oof_g"], 0.7 * C["s_dev_r10"] + 0.3 * C["s_dev_g"]),
}
best = None
for name, (_, Pd) in preds.items():
    S = np.clip(Pd, 0, 1)
    for mode in ("lag", "lag+fill", "greedy"):
        sc, rep, ch = evaluate(S, pc_d, mode, f"{name} / {mode}")
        if best is None or sc > best[0]:
            best = (sc, name, mode, rep, ch)

print(f"\n  hash-regex 공식 DEV=0.695369 | 오라클 DEV≈0.80")
print(f"  BEST: {best[1]} / {best[2]} = {best[0]:.6f}")
for tier in TIERS:
    t = best[3]["tiers"][tier]
    print(f"    {tier:9s} score={t['tier_score'][:8]} cost={t['budget_ratio'][:6]} "
          f"safety={best[4][tier]:.3f} picks={t['model_counts']}")
print(f"[{time.time()-t0:.0f}s]")
