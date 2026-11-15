"""Tests for the cross-encoder reranker.

The scorer is replaced by a fake, so the ranking and the fallback behaviour
are covered without loading a model.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from core.settings import RerankSettings, Settings
from libs.reranker.base_reranker import RerankerError
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.reranker_factory import RerankerFactory


def make_candidates(*ids: str) -> list[dict]:
    return [{"id": candidate_id, "text": f"text of {candidate_id}"} for candidate_id in ids]


def make_reranker(scores=None, error: Exception | None = None, top_k: int = 5):
    """Build a reranker whose scorer answers with `scores`."""

    def scorer(query: str, passages: Sequence[str]) -> Sequence[float]:
        if error is not None:
            raise error
        return scores if scores is not None else [0.0] * len(passages)

    return CrossEncoderReranker(
        RerankSettings(enabled=True, provider="cross_encoder", model="fake", top_k=top_k),
        scorer,
    )


def make_settings(provider: str = "cross_encoder", enabled: bool = True) -> Settings:
    return Settings(
        rerank=RerankSettings(enabled=enabled, provider=provider, model="fake", top_k=5)
    )


class TestRanking:
    def test_the_candidates_are_sorted_by_the_score(self):
        reranker = make_reranker(scores=[0.1, 0.9, 0.5])

        ranked = reranker.rerank("query", make_candidates("a", "b", "c"))

        assert [candidate["id"] for candidate in ranked] == ["b", "c", "a"]

    def test_the_candidate_records_are_preserved(self):
        reranker = make_reranker(scores=[0.1, 0.9])

        ranked = reranker.rerank("query", make_candidates("a", "b"))

        assert ranked[0] == {"id": "b", "text": "text of b"}

    def test_a_successful_ranking_is_not_a_fallback(self):
        reranker = make_reranker(scores=[0.1, 0.9])

        reranker.rerank("query", make_candidates("a", "b"))

        assert reranker.fallback_used is False

    def test_the_scorer_receives_the_query_and_the_passages(self):
        seen: dict[str, object] = {}

        def scorer(query: str, passages: Sequence[str]) -> Sequence[float]:
            seen["query"] = query
            seen["passages"] = list(passages)
            return [0.0] * len(passages)

        reranker = CrossEncoderReranker(
            RerankSettings(enabled=True, provider="cross_encoder", top_k=5), scorer
        )
        reranker.rerank("what is rag", make_candidates("a", "b"))

        assert seen["query"] == "what is rag"
        assert seen["passages"] == ["text of a", "text of b"]

    def test_only_the_top_candidates_are_scored(self):
        scored: list[str] = []

        def scorer(query: str, passages: Sequence[str]) -> Sequence[float]:
            scored.extend(passages)
            return [0.0] * len(passages)

        reranker = CrossEncoderReranker(
            RerankSettings(enabled=True, provider="cross_encoder", top_k=2), scorer
        )
        reranker.rerank("query", make_candidates("a", "b", "c", "d"))

        assert len(scored) == 2

    def test_the_candidates_below_the_top_k_keep_their_order(self):
        reranker = make_reranker(scores=[0.1, 0.9], top_k=2)

        ranked = reranker.rerank("query", make_candidates("a", "b", "c", "d"))

        assert [candidate["id"] for candidate in ranked] == ["b", "a", "c", "d"]

    def test_a_single_candidate_is_returned_unchanged(self):
        reranker = make_reranker()

        assert [c["id"] for c in reranker.rerank("query", make_candidates("a"))] == ["a"]


class TestFallback:
    def test_a_failing_scorer_keeps_the_original_order(self):
        reranker = make_reranker(error=TimeoutError("the model timed out"))

        ranked = reranker.rerank("query", make_candidates("a", "b"))

        assert [candidate["id"] for candidate in ranked] == ["a", "b"]

    def test_a_failing_scorer_marks_the_fallback(self):
        reranker = make_reranker(error=TimeoutError("the model timed out"))

        reranker.rerank("query", make_candidates("a", "b"))

        assert reranker.fallback_used is True

    def test_a_scorer_returning_the_wrong_number_of_scores_falls_back(self):
        reranker = make_reranker(scores=[0.1])

        ranked = reranker.rerank("query", make_candidates("a", "b"))

        assert [candidate["id"] for candidate in ranked] == ["a", "b"]
        assert reranker.fallback_used is True


class TestInputValidation:
    def test_an_empty_query_is_rejected(self):
        with pytest.raises(ValueError):
            make_reranker().rerank("   ", make_candidates("a"))

    def test_an_empty_candidate_list_is_rejected(self):
        with pytest.raises(ValueError):
            make_reranker().rerank("query", [])

    def test_a_candidate_without_an_id_is_rejected(self):
        with pytest.raises(RerankerError, match="carries no id"):
            make_reranker().rerank("query", [{"text": "no id here"}])


class TestFactoryRouting:
    def test_the_factory_creates_the_cross_encoder_reranker(self):
        reranker = RerankerFactory.create(make_settings())

        assert isinstance(reranker, CrossEncoderReranker)

    def test_the_strategy_name_is_matched_case_insensitively(self):
        reranker = RerankerFactory.create(make_settings("Cross_Encoder"))

        assert isinstance(reranker, CrossEncoderReranker)

    def test_the_configured_model_is_used(self):
        reranker = RerankerFactory.create(make_settings())

        assert reranker.model_name == "fake"

    def test_a_disabled_rerank_falls_back_to_the_none_reranker(self):
        reranker = RerankerFactory.create(make_settings(enabled=False))

        assert reranker.__class__.__name__ == "NoneReranker"
