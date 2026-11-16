"""Unit tests for the reranker orchestration.

The reranker puts the best chunks in front of a result list. The tests pin
that the order of the backend is kept, that a backend that fails does not
empty the answer, and that the fallback is said out loud.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.query_engine.reranker import Reranker
from core.settings import RerankSettings, Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker


class ReversingReranker(BaseReranker):
    """Backend that puts the last candidate in front."""

    def __init__(self) -> None:
        super().__init__(RerankSettings(enabled=True, provider="fake"))

    def rerank(self, query, candidates, trace=None):
        self.validate_query(query)
        self.validate_candidates(candidates)
        self.fallback_used = False
        return list(reversed([dict(candidate) for candidate in candidates]))


class BrokenReranker(BaseReranker):
    """Backend that fails."""

    def __init__(self) -> None:
        super().__init__(RerankSettings(enabled=True, provider="broken"))

    def rerank(self, query, candidates, trace=None):
        raise RuntimeError("the reranking model is down")


class SilentFallbackReranker(BaseReranker):
    """Backend that swallows the failure and keeps the order."""

    def __init__(self) -> None:
        super().__init__(RerankSettings(enabled=True, provider="silent"))
        self.fallback_used = False

    def rerank(self, query, candidates, trace=None):
        self.validate_query(query)
        self.validate_candidates(candidates)
        self.fallback_used = True
        return [dict(candidate) for candidate in candidates]


class DroppingReranker(BaseReranker):
    """Backend that leaves one candidate out."""

    def __init__(self) -> None:
        super().__init__(RerankSettings(enabled=True, provider="dropping"))

    def rerank(self, query, candidates, trace=None):
        self.validate_query(query)
        self.validate_candidates(candidates)
        self.fallback_used = False
        return [dict(candidates[0])]


def make_settings() -> Settings:
    return Settings(rerank=RerankSettings(enabled=True, provider="fake"))


def make_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=chunk_id,
            score=float(number),
            text=f"text of {chunk_id}",
            metadata={"source_path": "a.pdf"},
        )
        for number, chunk_id in enumerate(["a", "b", "c"])
    ]


class TestRerank:
    def test_the_order_of_the_backend_is_kept(self):
        reranker = Reranker(make_settings(), ReversingReranker())

        results = reranker.rerank("hello", make_results())

        assert [result.chunk_id for result in results] == ["c", "b", "a"]

    def test_the_content_of_the_result_is_kept(self):
        reranker = Reranker(make_settings(), ReversingReranker())

        results = reranker.rerank("hello", make_results())

        assert results[0].text == "text of c"
        assert results[0].metadata["source_path"] == "a.pdf"

    def test_the_score_of_the_fusion_is_kept(self):
        reranker = Reranker(make_settings(), ReversingReranker())

        results = reranker.rerank("hello", make_results())

        assert results[0].score == pytest.approx(2.0)

    def test_the_results_are_numbered_again(self):
        reranker = Reranker(make_settings(), ReversingReranker())

        results = reranker.rerank("hello", make_results())

        assert [result.rank for result in results] == [1, 2, 3]

    def test_the_number_of_results_can_be_capped(self):
        reranker = Reranker(make_settings(), ReversingReranker())

        results = reranker.rerank("hello", make_results(), top_k=2)

        assert len(results) == 2

    def test_no_result_gives_no_result(self):
        reranker = Reranker(make_settings(), ReversingReranker())

        assert reranker.rerank("hello", []) == []

    def test_the_order_of_the_input_is_not_changed(self):
        reranker = Reranker(make_settings(), ReversingReranker())
        results = make_results()

        reranker.rerank("hello", results)

        assert [result.chunk_id for result in results] == ["a", "b", "c"]

    def test_a_result_the_backend_leaves_out_stays_in_the_list(self):
        reranker = Reranker(make_settings(), DroppingReranker())

        results = reranker.rerank("hello", make_results())

        assert [result.chunk_id for result in results] == ["a", "b", "c"]


class TestFallback:
    def test_a_backend_that_fails_does_not_empty_the_answer(self):
        reranker = Reranker(make_settings(), BrokenReranker())

        results = reranker.rerank("hello", make_results())

        assert len(results) == 3

    def test_a_backend_that_fails_keeps_the_order(self):
        reranker = Reranker(make_settings(), BrokenReranker())

        results = reranker.rerank("hello", make_results())

        assert [result.chunk_id for result in results] == ["a", "b", "c"]

    def test_a_backend_that_fails_is_said_out_loud(self):
        reranker = Reranker(make_settings(), BrokenReranker())

        reranker.rerank("hello", make_results())

        assert reranker.fallback_used is True

    def test_a_backend_that_works_is_not_a_fallback(self):
        reranker = Reranker(make_settings(), ReversingReranker())

        reranker.rerank("hello", make_results())

        assert reranker.fallback_used is False

    def test_a_backend_that_keeps_the_order_on_its_own_is_a_fallback(self):
        reranker = Reranker(make_settings(), SilentFallbackReranker())

        results = reranker.rerank("hello", make_results())

        assert reranker.fallback_used is True
        assert [result.chunk_id for result in results] == ["a", "b", "c"]

    def test_the_fallback_is_forgotten_after_a_good_run(self):
        reranker = Reranker(make_settings(), ReversingReranker())
        reranker.fallback_used = True

        reranker.rerank("hello", make_results())

        assert reranker.fallback_used is False


class TestFailures:
    def test_a_top_k_of_zero_is_refused(self):
        reranker = Reranker(make_settings(), ReversingReranker())

        with pytest.raises(ValueError, match="top_k"):
            reranker.rerank("hello", make_results(), top_k=0)


class TestTrace:
    def test_the_trace_records_what_happened(self):
        reranker = Reranker(make_settings(), ReversingReranker())
        trace = TraceContext()

        reranker.rerank("hello", make_results(), trace=trace)

        data = trace.get_stage_data("rerank")
        assert data["candidates"] == 3
        assert data["fallback_used"] is False
        assert data["order_after"] == ["c", "b", "a"]

    def test_the_trace_records_the_fallback(self):
        reranker = Reranker(make_settings(), BrokenReranker())
        trace = TraceContext()

        reranker.rerank("hello", make_results(), trace=trace)

        data = trace.get_stage_data("rerank")
        assert data["fallback_used"] is True
        assert "error" in data


class TestBackend:
    def test_the_backend_of_the_settings_is_used(self):
        settings = Settings(rerank=RerankSettings(enabled=False, provider="none"))

        reranker = Reranker(settings)

        assert reranker.backend is not None

    def test_a_backend_can_be_handed_in(self):
        backend = ReversingReranker()

        reranker = Reranker(make_settings(), backend)

        assert reranker.backend is backend


def test_the_candidates_carry_the_text(monkeypatch) -> None:
    seen: list[dict[str, Any]] = []

    class PeekingReranker(BaseReranker):
        def __init__(self) -> None:
            super().__init__(RerankSettings(enabled=True, provider="peeking"))

        def rerank(self, query, candidates, trace=None):
            seen.extend(dict(candidate) for candidate in candidates)
            return [dict(candidate) for candidate in candidates]

    reranker = Reranker(make_settings(), PeekingReranker())

    reranker.rerank("hello", make_results())

    assert seen[0]["text"] == "text of a"
    assert seen[0]["id"] == "a"
