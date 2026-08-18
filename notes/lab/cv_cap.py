# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""문항별 비용 상한(cap) 실험 — 예산 초과의 근본 원인을 차단한다.

관찰: K1 한 문항이 light 총액의 26.4%(premium 여유의 8.8%)를 먹을 수 있다.
     출력 토큰 최대 130,504. 한 문항의 예측 실패가 등급 전체를 0점으로 만든다.
     게다가 K1 출력이 긴 구간은 이득/비용도 최저였다(exp8 Q5).
     → 위험 문항은 애초에 승격 후보에서 빼면 안전성과 점수를 동시에 얻는다.

규칙 A: 승격 1건의 추가 비용이 '예산 여유'의 cap_share 를 넘으면 그 승격을 금지
규칙 B: K1 예측 출력 토큰 상위 drop_q 분위 문항은 K1 후보에서 제외
"""
from __future__ import annotations

import heapq
import sys
import time
from typing import Dict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, r"d:\opensource\skt-router\lab")
from cv import Config, MODEL_IDS, MULT, TIERS, WEIGHT, POLICY, load_all  # noqa: E402

RATE_IN = np.array([float(POLICY.models[m].input_token_rate) for m in MODEL_IDS])
RATE_OUT = np.array([float(POLICY.models[m].output_token_rate) for m in MODEL_IDS])
UNIT = float(POLICY.token_unit)
N_M = len(MODEL_IDS)

t0 = time.time()
DATA = load_all()
CFG = Config()
tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=60000,
                       sublinear_tf=True, dtype=np.float32)
tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120000,
                       sublinear_tf=True, dtype=np.float32)
Ww, Wc = tf_w.fit_transform(DATA.texts), tf_c.fit_transform(DATA.texts)
dmu, dsd = DATA.dense.mean(0), DATA.dense.std(0)
dsd = np.where(dsd > 1e-12, dsd, 1.0)
X = sparse.hstack([Ww, Wc, sparse.csr_matrix((DATA.dense - dmu) / dsd)]).tocsr()
SCORE, COST, OTOK, ITOK, KEYS = DATA.score, DATA.cost, DATA.out_tok, DATA.in_tok, DATA.keys
print(f"[{time.time()-t0:.0f}s] X={X.shape}")


def select_capped(pred_score, pred_cost, mult, safety, keys, cap_share=1.0, drop_q=1.0):
    """예측 예산 안에서 승격하되 규칙 A·B를 적용한다."""
    n = len(pred_score)
    sel = np.zeros(n, dtype=np.int64)
    light_total = float(pred_cost[:, 0].sum())
    spent = light_total
    cap = light_total * max(1.0, mult * safety)
    headroom = cap - light_total
    per_item_cap = headroom * cap_share if cap_share < 1.0 else float("inf")

    banned_k1 = np.zeros(n, dtype=bool)
    if drop_q < 1.0:
        k1_cost = pred_cost[:, 2]
        banned_k1 = k1_cost > np.quantile(k1_cost, drop_q)

    heap: list = []

    def push(i):
        cur = sel[i]
        for m in range(N_M):
            if m == cur or (m == 2 and banned_k1[i]):
                continue
            dq = pred_score[i, m] - pred_score[i, cur]
            dc = pred_cost[i, m] - pred_cost[i, cur]
            if dq <= 0:
                continue
            if dc > per_item_cap:
                continue
            heapq.heappush(heap, (-1e18 if dc <= 0 else -dq / dc, float(keys[i]), m, i))

    for i in range(n):
        push(i)
    while heap:
        _, _, m, i = heapq.heappop(heap)
        cur = sel[i]
        dq = pred_score[i, m] - pred_score[i, cur]
        dc = pred_cost[i, m] - pred_cost[i, cur]
        if dq <= 0 or dc > per_item_cap:
            continue
        if spent + dc <= cap:
            spent += dc
            sel[i] = m
            push(i)
    return sel


def folds_of(n_folds, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    fold = np.empty(X.shape[0], dtype=int)
    for i, idx in enumerate(order):
        fold[idx] = i % n_folds
    return fold


def prep(n_folds, seed=0):
    fold = folds_of(n_folds, seed)
    out = []
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        Ytr = np.hstack([SCORE[tr], np.log1p(OTOK[tr]), np.log1p(ITOK[tr][:, :1])])
        ridge = Ridge(alpha=CFG.alpha, solver="sparse_cg", random_state=0).fit(X[tr], Ytr)
        oof = np.empty_like(Ytr)
        Xtr = X[tr]
        inner = np.arange(tr.sum()) % 5
        for g in range(5):
            va = inner == g
            sub = Ridge(alpha=CFG.alpha, solver="sparse_cg", random_state=0).fit(Xtr[~va], Ytr[~va])
            oof[va] = sub.predict(Xtr[va])
        raw_te = ridge.predict(X[te])
        bump = np.exp(np.quantile(np.log1p(OTOK[tr]) - oof[:, N_M:2 * N_M], CFG.cost_quantile, axis=0))

        def mk(raw):
            s = np.clip(raw[:, :N_M], 0, 1)
            o = np.expm1(np.clip(raw[:, N_M:2 * N_M], -50, 50)).clip(0) * bump
            i_ = np.expm1(np.clip(raw[:, 2 * N_M:2 * N_M + 1], -50, 50)).clip(1)
            c = (np.repeat(i_, N_M, axis=1) * RATE_IN + o * RATE_OUT) / UNIT
            return s, c

        s_oof, c_oof = mk(oof)
        scale = COST[tr].sum(0) / c_oof.sum(0)
        c_oof = np.maximum(c_oof * scale, 1e-9)
        s_te, c_te = mk(raw_te)
        c_te = np.maximum(c_te * scale, 1e-9)
        out.append(dict(tr=tr, te=te, s_oof=s_oof, c_oof=c_oof, s_te=s_te, c_te=c_te))
    return out


def run(folds, margins: Dict[str, float], cap_share=1.0, drop_q=1.0, tag=""):
    finals, fails = [], {t: 0 for t in TIERS}
    usage = {t: [] for t in TIERS}
    k1n = []
    for fd in folds:
        tr, te = fd["tr"], fd["te"]
        final = 0.0
        for tier in TIERS:
            safety = 0.30
            for s in np.round(np.arange(0.30, 1.401, 0.005), 4)[::-1]:
                sel = select_capped(fd["s_oof"], fd["c_oof"], MULT[tier], float(s), KEYS[tr],
                                    cap_share, drop_q)
                r = COST[tr][np.arange(len(sel)), sel].sum() / COST[tr][:, 0].sum()
                if r <= MULT[tier] * margins[tier]:
                    safety = float(s)
                    break
            sel_te = select_capped(fd["s_te"], fd["c_te"], MULT[tier], safety, KEYS[te],
                                   cap_share, drop_q)
            n = len(sel_te)
            ratio = COST[te][np.arange(n), sel_te].sum() / COST[te][:, 0].sum()
            ok = ratio <= MULT[tier] + 1e-12
            pts = SCORE[te][np.arange(n), sel_te].mean() if ok else 0.0
            if not ok:
                fails[tier] += 1
            usage[tier].append(ratio / MULT[tier])
            if tier == "premium":
                k1n.append((sel_te == 2).mean())
            final += WEIGHT[tier] * pts
        finals.append(final)
    tf = sum(fails.values())
    use = "  ".join(f"{t[:4]}={np.mean(usage[t]):.2f}/{np.max(usage[t]):.2f}" for t in TIERS)
    print(f"{tag:44s} CV={np.mean(finals):.6f} 실패={tf:2d} {use} K1={np.mean(k1n):.1%}"
          f"{'  ' + str({k: v for k, v in fails.items() if v}) if tf else ''}")
    return float(np.mean(finals)), tf


if __name__ == "__main__":
    for n_folds in (5, 8):
        folds = prep(n_folds)
        print(f"\n===== {n_folds}-fold ({X.shape[0]//n_folds}문항 홀드아웃) [{time.time()-t0:.0f}s] =====")
        print("--- 기준: cap 없음 ---")
        for m in (0.90, 0.85, 0.82):
            run(folds, {t: m for t in TIERS}, tag=f"margin={m:.2f} cap=none")
        print("--- 규칙 A: 승격 1건 ≤ 여유의 cap_share ---")
        for cs in (0.20, 0.10, 0.05, 0.02):
            for m in (0.95, 0.90):
                run(folds, {t: m for t in TIERS}, cap_share=cs,
                    tag=f"margin={m:.2f} cap_share={cs}")
        print("--- 규칙 B: K1 예측비용 상위 분위 제외 ---")
        for dq in (0.9, 0.8, 0.7):
            for m in (0.95, 0.90):
                run(folds, {t: m for t in TIERS}, drop_q=dq,
                    tag=f"margin={m:.2f} drop_q={dq}")
        print("--- A+B 결합 ---")
        for cs, dq, m in ((0.10, 0.9, 0.95), (0.05, 0.9, 0.95), (0.05, 0.8, 0.95), (0.02, 0.8, 0.95)):
            run(folds, {t: m for t in TIERS}, cap_share=cs, drop_q=dq,
                tag=f"margin={m:.2f} cap={cs} drop={dq}")
    print(f"\n[{time.time()-t0:.0f}s] done")
