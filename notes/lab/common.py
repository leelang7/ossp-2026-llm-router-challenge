# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 공통 유틸 — 데이터 로드, 피처, 공식 채점기 연결.

공식 하네스(ossp_router)를 그대로 import해 Decimal 채점을 재현한다.
"""
from __future__ import annotations

import json
import math
import re
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

CHALLENGE = Path(r"d:\opensource\ossp-2026-llm-router-challenge")
sys.path.insert(0, str(CHALLENGE / "src"))
sys.path.insert(0, str(CHALLENGE / "baselines"))

from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    Decision,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
)
from ossp_router.scoring import score_submissions  # noqa: E402

POLICY = load_bundled_policy()
LIGHT, AX31, K1 = MODEL_IDS


def load_split(split: str):
    inputs = load_input(CHALLENGE / "data" / "materialized" / split / "inputs.json")
    outcomes = load_outcomes(CHALLENGE / "data" / split / "outcomes.json")
    index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    rows = []
    for ep in inputs.episodes:
        rec = {}
        for m in MODEL_IDS:
            o = index[(ep.episode_id, m)]
            rates = POLICY.models[m]
            cost = float(
                rates.fixed_cost
                + Decimal(o.input_tokens) * rates.input_token_rate / Decimal(POLICY.token_unit)
                + Decimal(o.output_tokens) * rates.output_token_rate / Decimal(POLICY.token_unit)
            )
            rec[m] = {
                "score": float(o.score),
                "cost": cost,
                "in_tok": o.input_tokens,
                "out_tok": o.output_tokens,
                "gens": o.num_generations,
            }
        rows.append(rec)
    return inputs, outcomes, rows


def episode_text(ep) -> str:
    if ep.prompt is not None:
        return ep.prompt
    return "\n".join(m.content for m in ep.messages)


# ---------------------------------------------------------------- 피처

_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_WORD = re.compile(r"[A-Za-z가-힣]+")
_SENT = re.compile(r"[.!?。！？]")
_CODE = re.compile(r"```|(?:^|\s)(?:def|class|function|SELECT|FROM|import|#include|return|public|void)\b|[{};]\s*$", re.I | re.M)
_MATH = re.compile(r"[=+\-*/^∑∫√≈≠≤≥]|\\(?:frac|sum|int|sqrt|begin|end|cdot|times|le|ge|pi|theta|alpha)\b")
_LATEX = re.compile(r"\$|\\\(|\\\[|\\text|\\mathrm|\\boxed")
_PROVE = re.compile(r"\b(prove|derive|theorem|lemma|counterexample|induction|증명|유도|정리|보조정리|반례|귀납)\b", re.I)
_ANALYZE = re.compile(r"\b(analyze|explain why|reason|complexity|big[- ]?o|algorithm|optimize|추론|분석|알고리즘|복잡도|최적화|이유)\b", re.I)
_PROGRAM = re.compile(r"\b(traceback|exception|stacktrace|compile|debug|refactor|unit test|예외|디버그|리팩터)\b", re.I)
_MCQ = re.compile(r"(?:^|\n)\s*(?:\(?[A-Ea-e][\).\]]|[①-⑩]|[1-5][\).]\s)", re.M)
_QUESTION = re.compile(r"[?？]")
_CONSTRAINT = re.compile(r"\b(must|should|constraint|require|각각|반드시|조건|제약|단,)\b", re.I)
_STEPWISE = re.compile(r"\b(step by step|first|second|finally|단계|먼저|다음으로|마지막)\b", re.I)
_SIMPLE_TASK = re.compile(r"\b(translate|summar|rewrite|list|define|번역|요약|정리해|나열)\b", re.I)
_INSTRUCT_FMT = re.compile(r"\b(json|yaml|csv|markdown|table|format|형식|표로)\b", re.I)

DENSE_NAMES = [
    "log_chars", "log_words", "log_sents", "msg_count", "log_msg_chars_max",
    "hangul_ratio", "ascii_ratio", "digit_ratio", "upper_ratio", "space_ratio",
    "punct_ratio", "unique_word_ratio", "avg_word_len", "max_word_len",
    "code_n", "math_n", "latex_n", "prove_n", "analyze_n", "program_n",
    "mcq_n", "question_n", "constraint_n", "stepwise_n", "simple_n", "fmt_n",
    "newline_n", "has_sys", "has_asst", "long_ctx", "very_long_ctx",
    "num_count", "big_num", "line_max_len", "eq_count", "paren_ratio",
]


def dense_features(text: str, ep) -> list[float]:
    n = len(text)
    nz = max(n, 1)
    words = _WORD.findall(text)
    wn = max(len(words), 1)
    uniq = len(set(w.lower() for w in words))
    lines = text.split("\n")
    nums = re.findall(r"\d+", text)
    msgs = ep.messages if ep.messages is not None else []
    return [
        math.log1p(n), math.log1p(len(words)), math.log1p(len(_SENT.findall(text))),
        float(len(msgs)), math.log1p(max((len(m.content) for m in msgs), default=n)),
        sum("\uac00" <= c <= "\ud7a3" for c in text) / nz,
        sum(c.isascii() and c.isalpha() for c in text) / nz,
        sum(c.isdigit() for c in text) / nz,
        sum(c.isupper() for c in text) / nz,
        sum(c.isspace() for c in text) / nz,
        sum(not c.isalnum() and not c.isspace() for c in text) / nz,
        uniq / wn,
        sum(len(w) for w in words) / wn,
        float(max((len(w) for w in words), default=0)),
        float(len(_CODE.findall(text))), float(len(_MATH.findall(text))),
        float(len(_LATEX.findall(text))), float(len(_PROVE.findall(text))),
        float(len(_ANALYZE.findall(text))), float(len(_PROGRAM.findall(text))),
        float(len(_MCQ.findall(text))), float(len(_QUESTION.findall(text))),
        float(len(_CONSTRAINT.findall(text))), float(len(_STEPWISE.findall(text))),
        float(len(_SIMPLE_TASK.findall(text))), float(len(_INSTRUCT_FMT.findall(text))),
        float(len(lines)),
        float(any(m.role == "system" for m in msgs)),
        float(any(m.role == "assistant" for m in msgs)),
        float(n > 2000), float(n > 8000),
        float(len(nums)), float(max((len(x) for x in nums), default=0)),
        float(max((len(l) for l in lines), default=0)),
        float(text.count("=")), text.count("(") / nz * 100,
    ]


_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_MASK = (1 << 64) - 1


def _fnv(s: str) -> int:
    h = _FNV_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * _FNV_PRIME) & _MASK
    return h


def hashed_features(text: str, bins: int) -> np.ndarray:
    """word 1/2-gram + char 4-gram signed hashing."""
    vec = np.zeros(bins, dtype=np.float64)
    toks = [t.lower() for t in _TOKEN.findall(text)][:600]
    grams = list(toks)
    grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    low = text.lower()[:3000]
    grams += [f"#{low[i:i+4]}" for i in range(0, max(0, len(low) - 3), 2)]
    for g in grams:
        h = _fnv(g)
        vec[h % bins] += 1.0 if (h >> 63) & 1 else -1.0
    norm = math.sqrt(float(np.dot(vec, vec)))
    if norm > 0:
        vec /= norm
    return vec


def build_matrix(inputs, bins: int) -> np.ndarray:
    out = []
    for ep in inputs.episodes:
        t = episode_text(ep)
        out.append(np.concatenate([np.asarray(dense_features(t, ep)), hashed_features(t, bins)]))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------- 정책 & 채점

def lagrangian_select(pred_score, pred_cost, budget_mult: float, safety: float):
    """공식 select_models와 동일 계열: penalty 이분탐색으로 예산 내 최대 효용."""
    n = len(pred_score)
    light_total = float(pred_cost[:, 0].sum())
    cap = light_total * max(1.0, budget_mult * safety)

    def choose(pen):
        util = pred_score - pen * pred_cost / light_total
        idx = np.argmax(util, axis=1)
        # 동점 시 낮은 인덱스(싼 모델) 우선
        best = util[np.arange(n), idx][:, None]
        idx = np.argmax(util >= best - 1e-15, axis=1)
        return idx, float(pred_cost[np.arange(n), idx].sum())

    idx, tot = choose(0.0)
    if tot > cap:
        lo, hi = 0.0, 1.0
        idx, tot = choose(hi)
        while tot > cap and hi < 2 ** 60:
            lo, hi = hi, hi * 2
            idx, tot = choose(hi)
        for _ in range(80):
            mid = (lo + hi) / 2
            cand, ct = choose(mid)
            if ct <= cap:
                hi, idx, tot = mid, cand, ct
            else:
                lo = mid
    if tot > cap:
        idx = np.zeros(n, dtype=int)
        tot = light_total
    return idx, tot / light_total


def official_score(inputs, outcomes, tier_idx: dict[str, np.ndarray]):
    subs = []
    for tier in TIERS:
        idx = tier_idx[tier]
        subs.append(Submission(
            schema_version=inputs.schema_version,
            challenge_id=inputs.challenge_id,
            policy_id=POLICY.policy_id,
            split=inputs.split,
            tier=tier,
            decisions=tuple(
                Decision(ep.episode_id, MODEL_IDS[int(i)])
                for ep, i in zip(inputs.episodes, idx)
            ),
        ))
    return score_submissions(inputs, outcomes, subs, POLICY)
