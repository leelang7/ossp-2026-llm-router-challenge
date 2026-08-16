# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""학습 스크립트 — 공개 Train/Dev로 라우터 아티팩트를 만든다.

산출물(artifact.npz)에는 전역 계수와 어휘·IDF, 등급별 안전계수만 담는다.
문항별 예측·선택·프롬프트는 저장하지 않는다.

  python3 train_routerx/train.py \
      --train-input data/materialized/train/inputs.json \
      --train-outcomes data/train/outcomes.json \
      --dev-input data/materialized/dev/inputs.json \
      --dev-outcomes data/dev/outcomes.json \
      --artifact src/routerx/artifact.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)
from routerx.features import DENSE_NAMES, dense_features, episode_text  # noqa: E402
from routerx.policy import select_batch  # noqa: E402
from routerx.router import _tie_key  # noqa: E402
from routerx.trees import Forest, export_forest  # noqa: E402

SEED = 0


def collect(input_path: Path, outcomes_path: Path, policy):
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    texts, dense, score, cost, out_tok, in_tok = [], [], [], [], [], []
    unit = Decimal(policy.token_unit)
    for ep in inputs.episodes:
        text = episode_text(ep)
        texts.append(text)
        dense.append(dense_features(text, ep))
        s, c, o_, i_ = [], [], [], []
        for m in MODEL_IDS:
            row = index[(ep.episode_id, m)]
            rates = policy.models[m]
            s.append(float(row.score))
            c.append(float(rates.fixed_cost
                           + Decimal(row.input_tokens) * rates.input_token_rate / unit
                           + Decimal(row.output_tokens) * rates.output_token_rate / unit))
            o_.append(row.output_tokens)
            i_.append(row.input_tokens)
        score.append(s), cost.append(c), out_tok.append(o_), in_tok.append(i_)
    return (inputs, outcomes, texts, np.asarray(dense, dtype=np.float64),
            np.asarray(score), np.asarray(cost),
            np.asarray(out_tok, dtype=np.float64), np.asarray(in_tok, dtype=np.float64))


def oof_ridge(X, Y, alpha, folds=5):
    P = np.empty((X.shape[0], Y.shape[1]))
    fid = np.arange(X.shape[0]) % folds
    for f in range(folds):
        va = fid == f
        model = Ridge(alpha=alpha, solver="sparse_cg", random_state=SEED).fit(X[~va], Y[~va])
        P[va] = model.predict(X[va])
    return P


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-input", type=Path, required=True)
    ap.add_argument("--train-outcomes", type=Path, required=True)
    ap.add_argument("--dev-input", type=Path, required=True)
    ap.add_argument("--dev-outcomes", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tree-iters", type=int, default=400,
                    help="토큰 예측 부스팅 트리 반복 수")
    ap.add_argument("--cost-quantile", type=float, default=0.3)
    ap.add_argument("--budget-margin", type=float, default=0.90,
                    help="학습 분할에서 예산의 이 비율까지만 사용하도록 안전계수를 정한다")
    ap.add_argument("--tier-margin", action="append", default=None, metavar="TIER=RATIO",
                    help="등급별 예산 마진 (예: premium=0.88). 여러 번 지정 가능")
    ap.add_argument("--tier-k1-cap", action="append", default=None, metavar="TIER=RATIO",
                    help="등급별 추론 모델 선택 비율 상한 (예: premium=0.11)")
    ap.add_argument("--k1-item-cap", type=float, default=0.05,
                    help="추론 모델 승격 1건이 쓸 수 있는 경량 총비용 대비 상한")
    ap.add_argument("--fit-on", choices=("train", "train+dev"), default="train+dev")
    args = ap.parse_args(argv)

    policy = load_bundled_policy()
    tr = collect(args.train_input, args.train_outcomes, policy)
    dv = collect(args.dev_input, args.dev_outcomes, policy)
    (tr_in, tr_out, tr_txt, tr_dense, tr_S, tr_C, tr_O, tr_I) = tr
    (dv_in, dv_out, dv_txt, dv_dense, dv_S, dv_C, dv_O, dv_I) = dv

    if args.fit_on == "train+dev":
        fit_txt = tr_txt + dv_txt
        fit_dense = np.vstack([tr_dense, dv_dense])
        fit_S, fit_C = np.vstack([tr_S, dv_S]), np.vstack([tr_C, dv_C])
        fit_O, fit_I = np.vstack([tr_O, dv_O]), np.vstack([tr_I, dv_I])
    else:
        fit_txt, fit_dense = tr_txt, tr_dense
        fit_S, fit_C, fit_O, fit_I = tr_S, tr_C, tr_O, tr_I

    tf_w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                           max_features=60000, sublinear_tf=True, lowercase=True, dtype=np.float32)
    tf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=120000, sublinear_tf=True, lowercase=True, dtype=np.float32)
    Ww = tf_w.fit_transform(fit_txt)
    Wc = tf_c.fit_transform(fit_txt)
    dmu, dsd = fit_dense.mean(0), fit_dense.std(0)
    dsd = np.where(dsd > 1e-12, dsd, 1.0)
    Dz = (fit_dense - dmu) / dsd
    X = sparse.hstack([Ww, Wc, sparse.csr_matrix(Dz)]).tocsr()
    print(f"features: word={Ww.shape[1]} char={Wc.shape[1]} dense={Dz.shape[1]} total={X.shape[1]}")

    # 품질은 선형 모델이 잘 맞지만 토큰 수는 그렇지 않다(상관 0.14~0.37).
    # 토큰은 부스팅 트리를 직접 계산 특징에 적합시켜 0.38~0.65까지 올린다.
    # 라우팅 시점에는 토큰 수가 주어지지 않으므로 학습·추론 모두 예측값을 쓴다.
    n_m = len(MODEL_IDS)
    model = Ridge(alpha=args.alpha, solver="sparse_cg", random_state=SEED).fit(X, fit_S)
    S_oof = oof_ridge(X, fit_S, args.alpha)

    tok_target = np.hstack([np.log1p(fit_O), np.log1p(fit_I[:, :1])])
    tree_params = dict(max_iter=args.tree_iters, learning_rate=0.05, max_leaf_nodes=31,
                       min_samples_leaf=15, l2_regularization=1.0,
                       early_stopping=False, random_state=SEED)
    tok_models = [HistGradientBoostingRegressor(**tree_params).fit(fit_dense, tok_target[:, j])
                  for j in range(tok_target.shape[1])]
    forest_arrays = export_forest(tok_models)
    T_fit = Forest(forest_arrays).predict(fit_dense)

    T_oof = np.empty_like(tok_target)
    fid = np.arange(fit_dense.shape[0]) % 5
    for f in range(5):
        va = fid == f
        sub = [HistGradientBoostingRegressor(**tree_params).fit(fit_dense[~va], tok_target[~va, j])
               for j in range(tok_target.shape[1])]
        T_oof[va] = np.column_stack([m.predict(fit_dense[va]) for m in sub])
    P_oof = np.hstack([S_oof, T_oof])
    P_fit = np.hstack([model.predict(X), T_fit])

    rate_in = np.array([float(policy.models[m].input_token_rate) for m in MODEL_IDS])
    rate_out = np.array([float(policy.models[m].output_token_rate) for m in MODEL_IDS])
    unit = float(policy.token_unit)

    def costs_from(raw, bump_vec, scale_vec):
        out_tok = np.expm1(np.clip(raw[:, n_m:2 * n_m], -50, 50)).clip(0) * bump_vec
        in_tok = np.expm1(np.clip(raw[:, 2 * n_m:2 * n_m + 1], -50, 50)).clip(1.0)
        cost = (np.repeat(in_tok, n_m, axis=1) * rate_in + out_tok * rate_out) / unit
        return np.maximum(cost * scale_vec, 1e-9)

    # 비용 보정은 두 조각의 목적이 달라 서로 다른 예측으로 계산한다.
    #  · bump(잔차 분위수)는 '예측이 처음 보는 자료에서 얼마나 빗나가는가'를
    #    담아야 하므로 OOF 예측으로 구한다. 학습에 쓴 자료의 잔차로 구하면
    #    특히 트리 모델에서 잔차가 거의 0이라 비용을 크게 과소평가한다.
    #  · cost_scale(총액 보정)은 배포될 모델의 총액을 맞추는 것이므로 full-fit
    #    예측으로 구한다. OOF로 구해 full-fit에 적용하면 모델별로 다른 편향이 남는다.
    bump = np.exp(np.quantile(np.log1p(fit_O) - P_oof[:, n_m:2 * n_m],
                              args.cost_quantile, axis=0))
    ones = np.ones(n_m)
    cost_scale = fit_C.sum(0) / costs_from(P_fit, bump, ones).sum(0)

    # 안전계수 탐색은 OOF 예측으로 한다(같은 자료에 과적합된 예측으로 마진을
    # 정하면 낙관적인 값이 나온다).
    pred_cost_oof = costs_from(P_oof, bump, cost_scale)
    pred_score_oof = np.clip(P_oof[:, :n_m], 0.0, 1.0)

    # 등급별 안전계수: OOF 예측으로 고르되 실제 비용이 margin 이하가 되는 최대값
    safety, diag = {}, {}
    keys = np.array([_tie_key(t) for t in fit_txt], dtype=np.float64)
    margins = {tier: args.budget_margin for tier in TIERS}
    for item in args.tier_margin or ():
        tier_name, _, value = item.partition("=")
        if tier_name not in margins:
            raise SystemExit(f"알 수 없는 등급: {tier_name}")
        margins[tier_name] = float(value)
    # 추론 모델(axk1-think)은 출력 토큰 꼬리가 두꺼워 한 문항이 경량 총비용의 26%까지
    # 쓸 수 있다. 교차검증 결과 선택 건수 자체를 묶는 편이 예산 초과를 막으면서
    # 점수도 높았다(고비용 구간은 이득/비용비도 최저였다).
    k1_caps = {tier: 1.0 for tier in TIERS}
    for item in args.tier_k1_cap or ():
        tier_name, _, value = item.partition("=")
        if tier_name not in k1_caps:
            raise SystemExit(f"알 수 없는 등급: {tier_name}")
        k1_caps[tier_name] = float(value)
    print("budget margins:", margins, "| k1 caps:", k1_caps)
    # 안전계수는 1.0을 넘지 않는다. 1.0 초과는 '예측 예산을 한도보다 크게 잡는다'는
    # 뜻이라 예산 가드가 사실상 풀린다. 탐색이 상한에 붙으면 경고한다.
    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        chosen = 0.30
        for s in np.round(np.arange(0.30, 1.001, 0.005), 4)[::-1]:
            idx = select_batch(pred_score_oof, pred_cost_oof, mult, float(s), keys,
                               k1_caps[tier], args.k1_item_cap)
            ratio = fit_C[np.arange(len(idx)), idx].sum() / fit_C[:, 0].sum()
            if ratio <= mult * margins[tier]:
                chosen = float(s)
                diag[tier] = {"fit_budget_ratio": float(ratio),
                              "fit_tier_score": float(fit_S[np.arange(len(idx)), idx].mean())}
                break
        safety[tier] = chosen
        if chosen >= 0.9999:
            print(f"  경고: {tier} 안전계수가 탐색 상한(1.0)에 도달했습니다. "
                  f"예측 예산이 실제보다 크게 잡혀 예산 가드가 느슨할 수 있습니다.")
    print("safety ratios:", {k: round(v, 3) for k, v in safety.items()}, diag)

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    vocab_w = sorted(tf_w.vocabulary_, key=tf_w.vocabulary_.get)
    vocab_c = sorted(tf_c.vocabulary_, key=tf_c.vocabulary_.get)
    np.savez_compressed(
        args.artifact,
        coef=model.coef_.astype(np.float32),
        intercept=model.intercept_.astype(np.float64),
        **forest_arrays,
        vocab_word=np.array(vocab_w, dtype=object),
        idf_word=tf_w.idf_.astype(np.float32),
        vocab_char=np.array(vocab_c, dtype=object),
        idf_char=tf_c.idf_.astype(np.float32),
        dense_mean=dmu, dense_scale=dsd,
        dense_names=np.array(DENSE_NAMES, dtype=object),
        token_bump=bump, cost_scale=cost_scale,
        rate_in=rate_in, rate_out=rate_out, token_unit=np.float64(unit),
        safety=np.array([safety[t] for t in TIERS], dtype=np.float64),
        k1_cap=np.array([k1_caps[t] for t in TIERS], dtype=np.float64),
        k1_item_cap=np.float64(args.k1_item_cap),
        tiers=np.array(list(TIERS), dtype=object),
        model_ids=np.array(list(MODEL_IDS), dtype=object),
        policy_id=np.array(policy.policy_id, dtype=object),
        meta=np.array(json.dumps({
            "alpha": args.alpha, "cost_quantile": args.cost_quantile,
            "budget_margins": margins, "k1_caps": k1_caps,
            "k1_item_cap": args.k1_item_cap, "fit_on": args.fit_on,
            "n_fit": int(X.shape[0]), "n_features": int(X.shape[1]),
        }), dtype=object),
    )
    size_mb = args.artifact.stat().st_size / 1e6
    print(f"OK: artifact 저장 {args.artifact} ({size_mb:.1f} MB)")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "safety_ratios": safety, "fit_diagnostics": diag,
            "n_fit": int(X.shape[0]), "n_features": int(X.shape[1]),
            "artifact_mb": size_mb,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
