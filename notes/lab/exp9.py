# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 9 — 예측 캘리브레이션 + 정직한 평가 프로토콜.

exp8 발견: K1 이득/비용은 중간 비용대에서 최대(역U자). β 조정으로는 못 잡는다.
→ 원인은 예측값의 '스케일 왜곡'. 랭킹은 맞아도 기대이득의 절대값이 틀리면
  이득/비용 트레이드오프 계산이 어긋난다. isotonic 캘리브레이션으로 교정.

동시에 평가 프로토콜을 정직하게: 안전계수를 Train OOF에서 정하고 Dev는 평가만.
"""
from __future__ import annotations

import heapq
import sys
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from common import (  # noqa: E402
    MODEL_IDS, TIERS, POLICY, load_split, official_score,
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

t_o, t_d = C["t_oof"], C["t_dev"]
bump = np.exp(np.quantile(np.log1p(Otr) - t_o, 0.3, axis=0))
in_tr, in_dv = np.repeat(Itr[:, :1], 3, 1), np.repeat(Idv[:, :1], 3, 1)
pc_o = cost_of(in_tr, np.expm1(t_o).clip(0) * bump)
scale = Ctr.sum(0) / pc_o.sum(0)
pc_o *= scale
pc_d = cost_of(in_dv, np.expm1(t_d).clip(0) * bump) * scale

S_o = np.clip(0.7 * C["s_oof_r10"] + 0.3 * C["s_oof_g"], 0, 1)
S_d = np.clip(0.7 * C["s_dev_r10"] + 0.3 * C["s_dev_g"], 0, 1)


def calibrate(S_train_oof, S_apply):
    """Δ 예측을 실제 Δ의 조건부 기댓값으로 isotonic 보정."""
    out_tr = S_train_oof.copy()
    out_ap = S_apply.copy()
    real_d = [Str[:, 1] - Str[:, 0], Str[:, 2] - Str[:, 1]]
    base_tr, base_ap = S_train_oof[:, 0].copy(), S_apply[:, 0].copy()
    cum_tr, cum_ap = base_tr.copy(), base_ap.copy()
    for k in (0, 1):
        pd_tr = S_train_oof[:, k + 1] - S_train_oof[:, k]
        pd_ap = S_apply[:, k + 1] - S_apply[:, k]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=-1.0, y_max=1.0)
        iso.fit(pd_tr, real_d[k])
        cum_tr = cum_tr + iso.predict(pd_tr)
        cum_ap = cum_ap + iso.predict(pd_ap)
        out_tr[:, k + 1] = cum_tr
        out_ap[:, k + 1] = cum_ap
    return np.clip(out_tr, 0, 1), np.clip(out_ap, 0, 1)


def greedy(pS, pC, cap):
    n = len(pS)
    idx = np.zeros(n, dtype=int)
    spent = float(pC[:, 0].sum())
    heap = []

    def push(i):
        for m in range(3):
            dq = pS[i, m] - pS[i, idx[i]]
            dc = pC[i, m] - pC[i, idx[i]]
            if dq <= 0:
                continue
            heapq.heappush(heap, (-1e18 if dc <= 0 else -dq / dc, i, m))

    for i in range(n):
        push(i)
    while heap:
        _, i, m = heapq.heappop(heap)
        dq, dc = pS[i, m] - pS[i, idx[i]], pC[i, m] - pC[i, idx[i]]
        if dq <= 0:
            continue
        if spent + dc <= cap:
            spent += dc
            idx[i] = m
            push(i)
    return idx


def pick_safety(pS, pC, real, mult, margin):
    """실제 비용이 mult*margin 이하가 되는 최대 안전계수."""
    for s in np.round(np.arange(0.30, 1.501, 0.005), 4)[::-1]:
        idx = greedy(pS, pC, float(pC[:, 0].sum()) * max(1.0, mult * s))
        if real[np.arange(len(idx)), idx].sum() / real[:, 0].sum() <= mult * margin:
            return s
    return 0.30


def run(tag, S_tr, S_dv, calib_on, margin=0.9985):
    idx_all, ch = {}, {}
    for tier in TIERS:
        mult = float(POLICY.tiers[tier].budget_multiplier)
        if calib_on == "train":
            s = pick_safety(S_tr, pc_o, Ctr, mult, margin)
        else:
            s = pick_safety(S_dv, pc_d, Cdv, mult, margin)
        ch[tier] = s
        idx_all[tier] = greedy(S_dv, pc_d, float(pc_d[:, 0].sum()) * max(1.0, mult * s))
    rep = official_score(dv_in, dv_out, idx_all)
    ln = [f"{tag:34s} DEV={rep['final_score'][:8]}"]
    for tier in TIERS:
        t = rep["tiers"][tier]
        ok = "" if t["budget_passed"] else "!OVER"
        ln.append(f" {tier[:4]}={t['tier_score'][:6]}/{t['budget_ratio'][:5]}{ok}")
    print("".join(ln))
    return float(rep["final_score"]), rep, ch


print("=== 캘리브레이션 효과 ===")
Sc_o, Sc_d = calibrate(S_o, S_d)
d1r, d2r = Sdv[:, 1] - Sdv[:, 0], Sdv[:, 2] - Sdv[:, 1]
for nm, S in (("raw", S_d), ("calibrated", Sc_d)):
    p1, p2 = S[:, 1] - S[:, 0], S[:, 2] - S[:, 1]
    print(f"  [{nm:10s}] Δ예측 평균 d1={p1.mean():+.4f}(실제{d1r.mean():+.4f}) "
          f"d2={p2.mean():+.4f}(실제{d2r.mean():+.4f})  "
          f"corr d1={np.corrcoef(p1,d1r)[0,1]:.3f} d2={np.corrcoef(p2,d2r)[0,1]:.3f}")

print("\n=== Dev 보정 (baseline과 동일 조건, 낙관적) ===")
run("raw", S_o, S_d, "dev")
run("calibrated", Sc_o, Sc_d, "dev")

print("\n=== Train 보정 → Dev 평가 (정직한 일반화) ===")
for margin in (0.9985, 0.97, 0.94, 0.90):
    run(f"raw m={margin}", S_o, S_d, "train", margin)
for margin in (0.9985, 0.97, 0.94, 0.90):
    run(f"calibrated m={margin}", Sc_o, Sc_d, "train", margin)

print(f"\n  hash-regex=0.695369 | exp7·8 최고=0.697301(Dev보정) | 오라클≈0.80")
print(f"[{time.time()-t0:.0f}s]")
