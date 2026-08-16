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
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--cost-quantile", type=float, default=0.3)
    ap.add_argument("--budget-margin", type=float, default=0.90,
                    help="학습 분할에서 예산의 이 비율까지만 사용하도록 안전계수를 정한다")
    ap.add_argument("--tier-margin", action="append", default=None, metavar="TIER=RATIO",
                    help="등급별 예산 마진 (예: premium=0.88). 여러 번 지정 가능")
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

    # 타겟: 모델별 score(3) + 모델별 log 출력 토큰(3) + log 입력 토큰(1).
    # 입력 토큰은 라우팅 시점에 주어지지 않으므로 학습·추론 모두 예측값을 쓴다.
    n_m = len(MODEL_IDS)
    Y = np.hstack([fit_S, np.log1p(fit_O), np.log1p(fit_I[:, :1])])
    model = Ridge(alpha=args.alpha, solver="sparse_cg", random_state=SEED).fit(X, Y)
    P_oof = oof_ridge(X, Y, args.alpha)

    tok_resid = np.log1p(fit_O) - P_oof[:, n_m:2 * n_m]
    bump = np.exp(np.quantile(tok_resid, args.cost_quantile, axis=0))
    rate_in = np.array([float(policy.models[m].input_token_rate) for m in MODEL_IDS])
    rate_out = np.array([float(policy.models[m].output_token_rate) for m in MODEL_IDS])
    unit = float(policy.token_unit)
    pred_in_oof = np.expm1(P_oof[:, 2 * n_m:2 * n_m + 1]).clip(1.0)
    in_rep = np.repeat(pred_in_oof, n_m, axis=1)
    pred_cost_oof = (in_rep * rate_in
                     + np.expm1(P_oof[:, n_m:2 * n_m]).clip(0) * bump * rate_out) / unit
    cost_scale = fit_C.sum(0) / pred_cost_oof.sum(0)
    pred_cost_oof *= cost_scale
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
    print("budget margins:", margins)
    for tier in TIERS:
        mult = float(policy.tiers[tier].budget_multiplier)
        chosen = 0.30
        for s in np.round(np.arange(0.30, 1.401, 0.005), 4)[::-1]:
            idx = select_batch(pred_score_oof, pred_cost_oof, mult, float(s), keys)
            ratio = fit_C[np.arange(len(idx)), idx].sum() / fit_C[:, 0].sum()
            if ratio <= mult * margins[tier]:
                chosen = float(s)
                diag[tier] = {"fit_budget_ratio": float(ratio),
                              "fit_tier_score": float(fit_S[np.arange(len(idx)), idx].mean())}
                break
        safety[tier] = chosen
    print("safety ratios:", {k: round(v, 3) for k, v in safety.items()}, diag)

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    vocab_w = sorted(tf_w.vocabulary_, key=tf_w.vocabulary_.get)
    vocab_c = sorted(tf_c.vocabulary_, key=tf_c.vocabulary_.get)
    np.savez_compressed(
        args.artifact,
        coef=model.coef_.astype(np.float32),
        intercept=model.intercept_.astype(np.float64),
        vocab_word=np.array(vocab_w, dtype=object),
        idf_word=tf_w.idf_.astype(np.float32),
        vocab_char=np.array(vocab_c, dtype=object),
        idf_char=tf_c.idf_.astype(np.float32),
        dense_mean=dmu, dense_scale=dsd,
        dense_names=np.array(DENSE_NAMES, dtype=object),
        token_bump=bump, cost_scale=cost_scale,
        rate_in=rate_in, rate_out=rate_out, token_unit=np.float64(unit),
        safety=np.array([safety[t] for t in TIERS], dtype=np.float64),
        tiers=np.array(list(TIERS), dtype=object),
        model_ids=np.array(list(MODEL_IDS), dtype=object),
        policy_id=np.array(policy.policy_id, dtype=object),
        meta=np.array(json.dumps({
            "alpha": args.alpha, "cost_quantile": args.cost_quantile,
            "budget_margins": margins, "fit_on": args.fit_on,
            "n_fit": int(X.shape[0]), "n_features": int(X.shape[1]),
        }), dtype=object),
        allow_pickle=True,
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
