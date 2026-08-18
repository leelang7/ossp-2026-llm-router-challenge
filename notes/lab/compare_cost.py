# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""비용 예측 방식 직접 비교 — ridge vs GBM을 같은 정책·같은 fold로 채점한다.

관찰: GBM 토큰 예측은 Dev 상관이 훨씬 높은데(0.65 vs 0.39) 교차검증 점수는 더 낮고
예산 초과가 난다. 원인 가설은 안전계수 탐색과 실제 적용의 불일치다.
안전계수는 OOF 예측으로 정하는데, 트리 모델은 OOF와 홀드아웃 예측의 성격 차이가
선형 모델보다 훨씬 크다(작은 학습셋에서 더 나빠지고 큰 학습셋에서 크게 좋아진다).
그러면 OOF 기준으로 고른 안전계수가 홀드아웃에서는 과도하게 공격적이 된다.

두 캐시를 같은 조건으로 돌려 어느 쪽을 제출할지 정한다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import MULT, TIERS, WEIGHT  # noqa: E402
from routerx.policy import select_batch  # noqa: E402

CACHES = {
    "ridge 비용": r"d:\opensource\skt-router\lab\fold_cache.npz",
    "GBM 비용": r"d:\opensource\skt-router\lab\fold_cache2.npz",
}
GRID = np.round(np.arange(0.30, 1.001, 0.005), 4)[::-1]
ITEM = 0.05


def evaluate(C, tier, margin, k1_cap, n_folds):
    score, cost, keys = C["score"], C["cost"], C["keys"]
    fold = C[f"fold{n_folds}"]
    s_te_all, c_te_all = C[f"s_te{n_folds}"], C[f"c_te{n_folds}"]
    pts, ratios, fails = [], [], 0
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        s_oof, c_oof = C[f"f{n_folds}_{f}_s_oof"], C[f"f{n_folds}_{f}_c_oof"]
        safety = 0.30
        for s in GRID:
            sel = select_batch(s_oof, c_oof, MULT[tier], float(s), keys[tr], k1_cap, ITEM)
            r = cost[tr][np.arange(len(sel)), sel].sum() / cost[tr][:, 0].sum()
            if r <= MULT[tier] * margin:
                safety = float(s)
                break
        sel_te = select_batch(s_te_all[te], c_te_all[te], MULT[tier], safety,
                              keys[te], k1_cap, ITEM)
        n = len(sel_te)
        ratio = cost[te][np.arange(n), sel_te].sum() / cost[te][:, 0].sum()
        ok = ratio <= MULT[tier] + 1e-12
        pts.append(score[te][np.arange(n), sel_te].mean() if ok else 0.0)
        ratios.append(ratio / MULT[tier])
        fails += 0 if ok else 1
    return np.mean(pts), max(ratios), fails


SPACE = {"fast": [0.0], "balanced": [0.01, 0.03], "premium": [0.08, 0.11, 0.15, 0.20]}
MARGINS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

if __name__ == "__main__":
    for name, path in CACHES.items():
        C = np.load(path)
        print(f"\n{'='*78}\n{name}\n{'='*78}")
        chosen = {}
        for tier in TIERS:
            print(f"  --- {tier} ---")
            best = None
            for cap in SPACE[tier]:
                for m in MARGINS:
                    p5, r5, f5 = evaluate(C, tier, m, cap, 5)
                    p8, r8, f8 = evaluate(C, tier, m, cap, 8)
                    exp = 0.5 * p5 + 0.5 * p8
                    safe = (f5 + f8) == 0
                    if safe:
                        print(f"    K1={cap:.2f} m={m:.2f}  점수={exp:.5f} "
                              f"최대사용률 5f={r5:.2f}/8f={r8:.2f}  SAFE")
                    if safe and (best is None or exp > best[0]):
                        best = (exp, cap, m, max(r5, r8))
            if best is None:
                print("    실패 0 구간 없음")
                best = (0.0, SPACE[tier][0], MARGINS[-1], 9.9)
            chosen[tier] = best
            print(f"    → {tier}: K1={best[1]:.2f} margin={best[2]:.2f} "
                  f"점수={best[0]:.5f} 최대사용률={best[3]:.2f}")
        total = sum(WEIGHT[t] * chosen[t][0] for t in TIERS)
        print(f"  ▶ {name} CV 종합 = {total:.6f}")
        print("    학습 인자: " + " ".join(
            f"--tier-margin {t}={chosen[t][2]:.2f} --tier-k1-cap {t}={chosen[t][1]:.2f}"
            for t in TIERS) + f" --k1-item-cap {ITEM}")
