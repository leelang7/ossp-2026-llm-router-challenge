# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""마진 스윕 — fold별 예측을 한 번만 계산하고 마진만 바꿔가며 평가한다.

목표: 홀드아웃 예산 초과(등급 0점) 횟수를 0으로 만들면서 점수를 최대화.
비공개 평가셋 크기가 공개되지 않았으므로 fold 크기를 바꿔 작은 평가셋 위험도 함께 본다.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import (  # noqa: E402
    Config, Data, MODEL_IDS, MULT, TIERS, WEIGHT,
    build_features, fit_predict, load_all, select_batch, to_score_cost,
)


def fold_predictions(cfg: Config, data: Data, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data.texts))
    fold_of = np.empty(len(order), dtype=int)
    for i, idx in enumerate(order):
        fold_of[idx] = i % n_folds
    out = []
    n_m = len(MODEL_IDS)
    for f in range(n_folds):
        te = fold_of == f
        tr = ~te
        X_fit, (X_te,) = build_features(
            cfg, [data.texts[i] for i in np.where(tr)[0]], data.dense[tr],
            [([data.texts[i] for i in np.where(te)[0]], data.dense[te])],
        )
        Y = np.hstack([data.score[tr], np.log1p(data.out_tok[tr]), np.log1p(data.in_tok[tr][:, :1])])
        oof, (raw_te,) = fit_predict(cfg, X_fit, Y, [X_te])
        bump = np.exp(np.quantile(np.log1p(data.out_tok[tr]) - oof[:, n_m:2 * n_m],
                                  cfg.cost_quantile, axis=0))
        _s, c_tmp = to_score_cost(cfg, oof, bump, np.ones(n_m))
        cost_scale = data.cost[tr].sum(0) / c_tmp.sum(0)
        s_oof, c_oof = to_score_cost(cfg, oof, bump, cost_scale)
        s_te, c_te = to_score_cost(cfg, raw_te, bump, cost_scale)
        out.append(dict(tr=tr, te=te, s_oof=s_oof, c_oof=c_oof, s_te=s_te, c_te=c_te))
    return out


def eval_margins(folds, data: Data, margins: dict):
    finals, fails, ratios = [], {t: 0 for t in TIERS}, {t: [] for t in TIERS}
    scores = {t: [] for t in TIERS}
    for fd in folds:
        tr, te = fd["tr"], fd["te"]
        final = 0.0
        for tier in TIERS:
            safety = 0.30
            for s in np.round(np.arange(0.30, 1.401, 0.005), 4)[::-1]:
                sel = select_batch(fd["s_oof"], fd["c_oof"], MULT[tier], float(s), data.keys[tr])
                r = data.cost[tr][np.arange(len(sel)), sel].sum() / data.cost[tr][:, 0].sum()
                if r <= MULT[tier] * margins[tier]:
                    safety = float(s)
                    break
            sel_te = select_batch(fd["s_te"], fd["c_te"], MULT[tier], safety, data.keys[te])
            n = len(sel_te)
            ratio = data.cost[te][np.arange(n), sel_te].sum() / data.cost[te][:, 0].sum()
            passed = ratio <= MULT[tier] + 1e-12
            q = data.score[te][np.arange(n), sel_te].mean()
            pts = q if passed else 0.0
            if not passed:
                fails[tier] += 1
            ratios[tier].append(ratio / MULT[tier])
            scores[tier].append(pts)
            final += WEIGHT[tier] * pts
        finals.append(final)
    return float(np.mean(finals)), fails, ratios, scores


if __name__ == "__main__":
    t0 = time.time()
    data = load_all()
    cfg = Config()
    for n_folds in (5, 8):
        folds = fold_predictions(cfg, data, n_folds, seed=0)
        size = len(data.texts) // n_folds
        print(f"\n===== {n_folds}-fold (홀드아웃 약 {size}문항) [{time.time()-t0:.0f}s] =====")
        print(f"{'margins':32s} {'CV':>9s}  실패  등급별 예산사용률(평균/최대)")
        for m in (0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.78):
            margins = {"fast": m, "balanced": m, "premium": m}
            mean, fails, ratios, scores = eval_margins(folds, data, margins)
            total_fail = sum(fails.values())
            usage = "  ".join(
                f"{t[:4]}={np.mean(ratios[t]):.2f}/{np.max(ratios[t]):.2f}" for t in TIERS
            )
            flag = "" if total_fail == 0 else f"  FAIL={dict(fails)}"
            print(f"  uniform {m:.2f}{'':18s} {mean:.6f}  {total_fail:2d}  {usage}{flag}")
    print(f"\n[{time.time()-t0:.0f}s] done")
