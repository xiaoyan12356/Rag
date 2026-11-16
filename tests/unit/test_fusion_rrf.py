"""Unit tests for the rank fusion.

The fusion puts the results of the dense route and of the sparse route into
one list. The tests pin that a chunk both routes found comes first, that the
score is the sum of the two places, and that the same input always gives the
same output.
"""

from __future__ import annotations

import pytest

from core.query_engine.fusion import RRFFusion
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


def make_result(chunk_id: str, score: float = 1.0, text: str = "some text") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source_path": "a.pdf"},
    )


class TestFuse:
    def test_the_results_of_both_routes_are_in_the_list(self):
        fusion = RRFFusion()

        results = fusion.fuse([make_result("a")], [make_result("b")])

        assert {result.chunk_id for result in results} == {"a", "b"}

    def test_a_chunk_that_both_routes_found_comes_first(self):
        fusion = RRFFusion()

        results = fusion.fuse(
            [make_result("shared"), make_result("dense_only")],
            [make_result("sparse_only"), make_result("shared")],
        )

        assert results[0].chunk_id == "shared"

    def test_the_score_is_the_sum_of_the_two_places(self):
        fusion = RRFFusion(k=60)

        results = fusion.fuse([make_result("shared")], [make_result("shared")])

        expected = 1 / 61 + 1 / 61
        assert results[0].score == pytest.approx(expected)

    def test_a_chunk_that_only_one_route_found_gets_one_place(self):
        fusion = RRFFusion(k=60)

        results = fusion.fuse([make_result("a")], [])

        assert results[0].score == pytest.approx(1 / 61)

    def test_an_earlier_place_is_worth_more(self):
        fusion = RRFFusion(k=60)

        results = fusion.fuse([make_result("first"), make_result("second")], [])

        assert results[0].chunk_id == "first"
        assert results[0].score > results[1].score

    def test_the_k_value_decides_how_flat_the_ranking_is(self):
        flat = RRFFusion(k=1000)
        steep = RRFFusion(k=1)

        flat_results = flat.fuse([make_result("first"), make_result("second")], [])
        steep_results = steep.fuse([make_result("first"), make_result("second")], [])

        flat_gap = flat_results[0].score - flat_results[1].score
        steep_gap = steep_results[0].score - steep_results[1].score
        assert steep_gap > flat_gap

    def test_the_k_value_can_be_chosen(self):
        fusion = RRFFusion(k=1)

        results = fusion.fuse([make_result("a")], [])

        assert results[0].score == pytest.approx(1 / 2)

    def test_the_same_input_gives_the_same_output(self):
        fusion = RRFFusion()

        first = fusion.fuse([make_result("a"), make_result("b")], [make_result("b")])
        second = fusion.fuse([make_result("a"), make_result("b")], [make_result("b")])

        assert [result.chunk_id for result in first] == [
            result.chunk_id for result in second
        ]
        assert [result.score for result in first] == [result.score for result in second]

    def test_a_tie_is_broken_by_the_chunk_id(self):
        fusion = RRFFusion()

        # Both chunks take one of the two places, so their scores are equal
        # and only the chunk id can tell them apart.
        results = fusion.fuse(
            [make_result("b"), make_result("a")],
            [make_result("a"), make_result("b")],
        )

        assert [result.chunk_id for result in results] == ["a", "b"]

    def test_the_number_of_results_can_be_capped(self):
        fusion = RRFFusion()

        results = fusion.fuse([make_result("a"), make_result("b"), make_result("c")], [], top_k=2)

        assert len(results) == 2

    def test_the_places_start_at_one(self):
        fusion = RRFFusion(k=60)

        results = fusion.fuse([make_result("a")], [])

        assert results[0].score == pytest.approx(1 / 61)

    def test_no_result_at_all_gives_an_empty_list(self):
        fusion = RRFFusion()

        assert fusion.fuse([], []) == []

    def test_the_score_of_the_dense_route_is_kept(self):
        fusion = RRFFusion()

        results = fusion.fuse([make_result("a", score=0.75)], [])

        assert results[0].dense_score == pytest.approx(0.75)

    def test_the_score_of_the_sparse_route_is_kept(self):
        fusion = RRFFusion()

        results = fusion.fuse([], [make_result("a", score=3.5)])

        assert results[0].sparse_score == pytest.approx(3.5)

    def test_a_chunk_that_only_the_dense_route_found_has_no_sparse_score(self):
        fusion = RRFFusion()

        results = fusion.fuse([make_result("a")], [])

        assert results[0].sparse_score is None

    def test_the_text_of_the_chunk_is_kept(self):
        fusion = RRFFusion()

        results = fusion.fuse([make_result("a", text="the text")], [])

        assert results[0].text == "the text"

    def test_the_results_are_numbered(self):
        fusion = RRFFusion()

        results = fusion.fuse([make_result("a"), make_result("b")], [])

        assert [result.rank for result in results] == [1, 2]

    def test_the_results_of_the_routes_are_not_changed(self):
        fusion = RRFFusion()
        dense = [make_result("a", score=0.75)]

        fusion.fuse(dense, [])

        assert dense[0].score == pytest.approx(0.75)
        assert dense[0].rank is None


class TestSetup:
    def test_a_k_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="k"):
            RRFFusion(k=0)


class TestFailures:
    def test_a_top_k_of_zero_is_refused(self):
        fusion = RRFFusion()

        with pytest.raises(ValueError, match="top_k"):
            fusion.fuse([make_result("a")], [], top_k=0)


class TestTrace:
    def test_the_trace_records_what_happened(self):
        fusion = RRFFusion()
        trace = TraceContext()

        fusion.fuse([make_result("a")], [make_result("b")], trace=trace)

        data = trace.get_stage_data("fusion")
        assert data["dense_count"] == 1
        assert data["sparse_count"] == 1
        assert data["fused_count"] == 2
