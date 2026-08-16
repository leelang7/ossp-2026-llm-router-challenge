# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""프롬프트 내용만으로 계산하는 특징.

문항 ID·입력 순서·출처 등 프롬프트 밖의 정보는 사용하지 않는다.
TF-IDF 변환은 학습에 쓴 scikit-learn TfidfVectorizer의 동작
(lowercase, token_pattern, char_wb, sublinear_tf, smooth idf, l2 norm)을
표준 라이브러리만으로 동일하게 재현한다.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple

# ------------------------------------------------------------------ dense

_WORD = re.compile(r"[A-Za-z가-힣]+")
_SENT = re.compile(r"[.!?。！？]")
_CODE = re.compile(
    r"```|(?:^|\s)(?:def|class|function|SELECT|FROM|import|#include|return|public|void)\b|[{};]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MATH = re.compile(r"[=+\-*/^∑∫√≈≠≤≥]|\\(?:frac|sum|int|sqrt|begin|end|cdot|times|le|ge|pi|theta|alpha)\b")
_LATEX = re.compile(r"\$|\\\(|\\\[|\\text|\\mathrm|\\boxed")
_PROVE = re.compile(r"\b(prove|derive|theorem|lemma|counterexample|induction|증명|유도|정리|보조정리|반례|귀납)\b", re.IGNORECASE)
_ANALYZE = re.compile(r"\b(analyze|explain why|reason|complexity|big[- ]?o|algorithm|optimize|추론|분석|알고리즘|복잡도|최적화|이유)\b", re.IGNORECASE)
_PROGRAM = re.compile(r"\b(traceback|exception|stacktrace|compile|debug|refactor|unit test|예외|디버그|리팩터)\b", re.IGNORECASE)
_MCQ = re.compile(r"(?:^|\n)\s*(?:\(?[A-Ea-e][\).\]]|[①-⑩]|[1-5][\).]\s)", re.MULTILINE)
_QUESTION = re.compile(r"[?？]")
_CONSTRAINT = re.compile(r"\b(must|should|constraint|require|각각|반드시|조건|제약|단,)\b", re.IGNORECASE)
_STEPWISE = re.compile(r"\b(step by step|first|second|finally|단계|먼저|다음으로|마지막)\b", re.IGNORECASE)
_SIMPLE_TASK = re.compile(r"\b(translate|summar|rewrite|list|define|번역|요약|정리해|나열)\b", re.IGNORECASE)
_INSTRUCT_FMT = re.compile(r"\b(json|yaml|csv|markdown|table|format|형식|표로)\b", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")

DENSE_NAMES: Tuple[str, ...] = (
    "log_chars", "log_words", "log_sents", "msg_count", "log_msg_chars_max",
    "hangul_ratio", "ascii_ratio", "digit_ratio", "upper_ratio", "space_ratio",
    "punct_ratio", "unique_word_ratio", "avg_word_len", "max_word_len",
    "code_n", "math_n", "latex_n", "prove_n", "analyze_n", "program_n",
    "mcq_n", "question_n", "constraint_n", "stepwise_n", "simple_n", "fmt_n",
    "newline_n", "has_sys", "has_asst", "long_ctx", "very_long_ctx",
    "num_count", "big_num", "line_max_len", "eq_count", "paren_ratio",
)


def episode_text(episode) -> str:
    """라우팅 시점에 볼 수 있는 프롬프트 본문만 반환한다."""
    if episode.prompt is not None:
        return episode.prompt
    return "\n".join(message.content for message in episode.messages)


def _char_counts(text: str):
    """문자 종류별 개수를 한 번의 순회로 센다."""
    hangul = latin = digit = upper = space = punct = 0
    for character in text:
        if "가" <= character <= "힣":
            hangul += 1
        elif character.isascii() and character.isalpha():
            latin += 1
        if character.isdigit():
            digit += 1
        if character.isupper():
            upper += 1
        if character.isspace():
            space += 1
        elif not character.isalnum():
            punct += 1
    return hangul, latin, digit, upper, space, punct


def dense_features(text: str, episode) -> List[float]:
    n = len(text)
    nz = max(n, 1)
    words = _WORD.findall(text)
    wn = max(len(words), 1)
    unique = len({w.lower() for w in words})
    lines = text.split("\n")
    numbers = _DIGITS.findall(text)
    messages = episode.messages if episode.messages is not None else ()
    hangul, latin, digit, upper, space, punct = _char_counts(text)
    return [
        math.log1p(n), math.log1p(len(words)), math.log1p(len(_SENT.findall(text))),
        float(len(messages)), math.log1p(max((len(m.content) for m in messages), default=n)),
        hangul / nz, latin / nz, digit / nz, upper / nz, space / nz, punct / nz,
        unique / wn,
        sum(len(w) for w in words) / wn,
        float(max((len(w) for w in words), default=0)),
        float(len(_CODE.findall(text))), float(len(_MATH.findall(text))),
        float(len(_LATEX.findall(text))), float(len(_PROVE.findall(text))),
        float(len(_ANALYZE.findall(text))), float(len(_PROGRAM.findall(text))),
        float(len(_MCQ.findall(text))), float(len(_QUESTION.findall(text))),
        float(len(_CONSTRAINT.findall(text))), float(len(_STEPWISE.findall(text))),
        float(len(_SIMPLE_TASK.findall(text))), float(len(_INSTRUCT_FMT.findall(text))),
        float(len(lines)),
        float(any(m.role == "system" for m in messages)),
        float(any(m.role == "assistant" for m in messages)),
        float(n > 2000), float(n > 8000),
        float(len(numbers)), float(max((len(x) for x in numbers), default=0)),
        float(max((len(line) for line in lines), default=0)),
        float(text.count("=")), text.count("(") / nz * 100,
    ]


# ------------------------------------------------------------------ TF-IDF

_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
_WHITESPACE = re.compile(r"\s\s+")


def word_ngrams(text: str, ngram_max: int = 2) -> List[str]:
    """scikit-learn word analyzer (lowercase + token_pattern + n-gram) 재현."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if ngram_max == 1:
        return tokens
    out = list(tokens)
    total = len(tokens)
    for n in range(2, min(ngram_max + 1, total + 1)):
        for i in range(total - n + 1):
            out.append(" ".join(tokens[i:i + n]))
    return out


def char_wb_ngrams(text: str, ngram_min: int = 3, ngram_max: int = 5) -> List[str]:
    """scikit-learn char_wb analyzer 재현 (단어를 공백으로 감싼 문자 n-gram)."""
    document = _WHITESPACE.sub(" ", text.lower())
    out: List[str] = []
    for word in document.split():
        padded = " " + word + " "
        length = len(padded)
        for n in range(ngram_min, ngram_max + 1):
            offset = 0
            out.append(padded[offset:offset + n])
            while offset + n < length:
                offset += 1
                out.append(padded[offset:offset + n])
            if offset == 0:      # 짧은 단어는 한 번만 센다
                break
    return out


def tfidf_row(grams: Iterable[str], vocabulary: Dict[str, int], idf: Sequence[float],
              offset: int = 0) -> Dict[int, float]:
    """sublinear_tf + smooth idf + l2 정규화는 호출부에서 합산 후 수행한다."""
    counts: Dict[int, int] = {}
    for gram in grams:
        column = vocabulary.get(gram)
        if column is not None:
            counts[column] = counts.get(column, 0) + 1
    return {offset + col: (1.0 + math.log(cnt)) * idf[col] for col, cnt in counts.items()}


def l2_normalize(row: Dict[int, float]) -> Dict[int, float]:
    norm = math.sqrt(sum(value * value for value in row.values()))
    if norm <= 0.0:
        return row
    return {key: value / norm for key, value in row.items()}
