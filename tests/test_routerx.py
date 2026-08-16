# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""routerx 라우터 단위 시험 — 표준 라이브러리만 사용한다.

규칙 위반과 런타임 실패로 이어지는 성질을 고정한다.
  · 선택이 문항 ID·입력 순서에 의존하지 않을 것
  · 같은 입력에 같은 출력이 나올 것
  · 예산과 추론 모델 상한을 넘지 않을 것
  · 어떤 프롬프트에도 예외 없이 유효한 모델 하나를 낼 것
"""
from __future__ import annotations

import random
import unittest
from pathlib import Path

import numpy as np

from ossp_router.protocol import MODEL_IDS, TIERS, Episode, Message, load_input
from routerx.features import (
    DENSE_NAMES,
    char_wb_ngrams,
    dense_features,
    episode_text,
    l2_normalize,
    tfidf_row,
    word_ngrams,
)
from routerx.policy import HEAVY_MODEL, select_batch
from routerx.router import Artifact, _tie_key, predict, route

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "src" / "routerx" / "artifact.npz"
DEV_INPUT = ROOT / "data" / "materialized" / "dev" / "inputs.json"


def _episode(text: str, episode_id: str = "e") -> Episode:
    return Episode(episode_id=episode_id, prompt=text, messages=None)


class FeatureTest(unittest.TestCase):
    def test_dense_length_matches_names(self):
        values = dense_features("hello 안녕 2+2", _episode("hello 안녕 2+2"))
        self.assertEqual(len(values), len(DENSE_NAMES))
        self.assertTrue(all(np.isfinite(v) for v in values))

    def test_dense_handles_degenerate_text(self):
        for text in ("", " ", "\n\n", "🚀", "a"):
            values = dense_features(text, _episode(text))
            self.assertEqual(len(values), len(DENSE_NAMES))
            self.assertTrue(all(np.isfinite(v) for v in values), text)

    def test_word_ngrams_follow_sklearn_token_pattern(self):
        # 기본 token_pattern은 두 글자 이상만 잡는다.
        self.assertEqual(word_ngrams("a bb ccc", 1), ["bb", "ccc"])
        self.assertIn("bb ccc", word_ngrams("a bb ccc", 2))

    def test_char_wb_pads_words_with_spaces(self):
        grams = char_wb_ngrams("ab", 3, 3)
        self.assertIn(" ab", grams)

    def test_char_wb_counts_short_word_once(self):
        # 단어가 n보다 짧으면 공백으로 감싼 문자열을 그대로 한 번만 낸다(sklearn 동작).
        self.assertEqual(char_wb_ngrams("a", 5, 5), [" a "])

    def test_l2_normalize_unit_norm(self):
        row = l2_normalize({0: 3.0, 1: 4.0})
        self.assertAlmostEqual(sum(v * v for v in row.values()), 1.0, places=12)

    def test_l2_normalize_all_zero_is_safe(self):
        self.assertEqual(l2_normalize({}), {})

    def test_tfidf_row_uses_sublinear_tf(self):
        row = tfidf_row(["a", "a"], {"a": 0}, [1.0])
        self.assertAlmostEqual(row[0], 1.0 + np.log(2.0), places=12)

    def test_episode_text_joins_messages(self):
        episode = Episode("e", prompt=None, messages=(
            Message(role="user", content="첫째"), Message(role="assistant", content="둘째")))
        self.assertEqual(episode_text(episode), "첫째\n둘째")


class PolicyTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 200
        self.score = rng.random((self.n, 3))
        self.score.sort(axis=1)                       # 무거운 모델일수록 좋게
        self.cost = np.sort(rng.random((self.n, 3)) + 0.1, axis=1)
        self.keys = rng.random(self.n)

    def test_respects_predicted_budget(self):
        for multiplier in (1.25, 2.0, 4.0):
            selected = select_batch(self.score, self.cost, multiplier, 1.0, self.keys)
            spent = self.cost[np.arange(self.n), selected].sum()
            self.assertLessEqual(spent, self.cost[:, 0].sum() * multiplier + 1e-9)

    def test_respects_heavy_share_cap(self):
        for cap in (0.0, 0.05, 0.25):
            selected = select_batch(self.score, self.cost, 4.0, 1.0, self.keys,
                                    heavy_share_cap=cap)
            heavy = int((selected == HEAVY_MODEL).sum())
            self.assertLessEqual(heavy, int(self.n * cap))

    def test_order_invariant(self):
        order = np.arange(self.n)
        rng = np.random.default_rng(7)
        shuffled = rng.permutation(order)
        base = select_batch(self.score, self.cost, 2.0, 1.0, self.keys)
        mixed = select_batch(self.score[shuffled], self.cost[shuffled], 2.0, 1.0,
                             self.keys[shuffled])
        np.testing.assert_array_equal(base[shuffled], mixed)

    def test_never_downgrades_below_light(self):
        selected = select_batch(self.score, self.cost, 1.0, 0.0, self.keys)
        np.testing.assert_array_equal(selected, np.zeros(self.n, dtype=selected.dtype))

    def test_tie_keys_break_ties_not_index(self):
        score = np.tile(np.array([0.0, 0.5, 1.0]), (10, 1))
        cost = np.tile(np.array([1.0, 2.0, 3.0]), (10, 1))
        keys = np.arange(10, dtype=float) / 10.0
        first = select_batch(score, cost, 1.5, 1.0, keys)
        perm = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        second = select_batch(score[perm], cost[perm], 1.5, 1.0, keys[perm])
        np.testing.assert_array_equal(first[perm], second)


@unittest.skipUnless(ARTIFACT.exists(), "artifact.npz가 없습니다")
class RouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = Artifact(ARTIFACT)

    def test_tie_key_depends_only_on_text(self):
        self.assertEqual(_tie_key("같은 글"), _tie_key("같은 글"))
        self.assertNotEqual(_tie_key("가"), _tie_key("나"))
        self.assertTrue(0.0 <= _tie_key("아무 글") < 1.0)

    def test_route_returns_valid_model_for_every_episode(self):
        episodes = [_episode(t, f"e{i}") for i, t in enumerate(
            ["", "  ", "2+2는?", "가나다" * 5000, "🚀🌏", "prove that 1+1=2"])]
        for tier in TIERS:
            picked = route(episodes, self.artifact, tier)
            self.assertEqual(len(picked), len(episodes))
            self.assertTrue(all(m in MODEL_IDS for m in picked))

    def test_route_is_deterministic(self):
        episodes = [_episode(f"문항 {i} 계산해줘", f"e{i}") for i in range(60)]
        self.assertEqual(route(episodes, self.artifact, "balanced"),
                         route(episodes, self.artifact, "balanced"))

    def test_route_ignores_episode_id_and_order(self):
        texts = [f"질문 {i}: {'추론' * (i % 7)} 계산" for i in range(80)]
        base = [_episode(t, f"a{i}") for i, t in enumerate(texts)]
        picked = dict(zip(texts, route(base, self.artifact, "premium")))
        shuffled = list(range(80))
        random.Random(3).shuffle(shuffled)
        renamed = [_episode(texts[i], f"zz{j}") for j, i in enumerate(shuffled)]
        after = route(renamed, self.artifact, "premium")
        for j, i in enumerate(shuffled):
            self.assertEqual(picked[texts[i]], after[j], f"문항 {i}에서 순서 의존 발생")

    def test_heavy_cap_enforced_end_to_end(self):
        episodes = [_episode(f"복잡한 증명 문제 {i} " + "추론 " * 40, f"e{i}") for i in range(120)]
        for tier in TIERS:
            picked = route(episodes, self.artifact, tier)
            heavy = sum(1 for m in picked if m == MODEL_IDS[HEAVY_MODEL])
            cap = self.artifact.k1_cap.get(tier, 1.0)
            self.assertLessEqual(heavy, int(len(episodes) * cap) + 1,
                                 f"{tier}에서 추론 모델 상한 초과")

    def test_predictions_are_finite_and_bounded(self):
        episodes = [_episode(t, f"e{i}") for i, t in enumerate(["", "짧음", "긴 " * 3000])]
        score, cost, keys = predict(episodes, self.artifact)
        self.assertTrue(np.isfinite(score).all() and np.isfinite(cost).all())
        self.assertTrue(((score >= 0.0) & (score <= 1.0)).all())
        self.assertTrue((cost > 0.0).all())
        self.assertTrue(((keys >= 0.0) & (keys < 1.0)).all())


@unittest.skipUnless(ARTIFACT.exists() and DEV_INPUT.exists(), "공개 자료가 없습니다")
class DevDataTest(unittest.TestCase):
    def test_all_dev_episodes_route_without_error(self):
        artifact = Artifact(ARTIFACT)
        episodes = load_input(DEV_INPUT).episodes[:250]
        picked = route(episodes, artifact, "fast")
        self.assertEqual(len(picked), len(episodes))
        self.assertTrue(all(m in MODEL_IDS for m in picked))


if __name__ == "__main__":
    unittest.main()
