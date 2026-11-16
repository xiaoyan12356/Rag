"""Unit tests for the ragas evaluator.

These tests pin the wrapper around ragas, not ragas itself: the metric
classes are replaced by fakes, so the tests need no model, no credentials
and no network.
"""

from __future__ import annotations

import pytest

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import EvaluatorError
from observability.evaluation.ragas_evaluator import (
    ANSWER_RELEVANCY,
    CONTEXT_PRECISION,
    FAITHFULNESS,
    RagasEvaluator,
    ragas_is_available,
)

pytestmark = pytest.mark.skipif(
    not ragas_is_available(), reason="ragas is not installed"
)


class FakeResult:
    """What a ragas metric hands back: a value and a reason."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.reason = "because"


def make_metric_class(value: float, calls: list[dict]):
    """Build a stand-in for a ragas metric class that records its calls."""

    class FakeMetric:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def score(self, **kwargs) -> FakeResult:
            calls.append(kwargs)
            return FakeResult(value)

    return FakeMetric


@pytest.fixture
def metrics(monkeypatch):
    """Replace the ragas metrics with fakes that record what they were asked."""
    calls: dict[str, list[dict]] = {
        FAITHFULNESS: [],
        ANSWER_RELEVANCY: [],
        CONTEXT_PRECISION: [],
    }
    monkeypatch.setattr(
        "ragas.metrics.collections.Faithfulness", make_metric_class(0.9, calls[FAITHFULNESS])
    )
    monkeypatch.setattr(
        "ragas.metrics.collections.AnswerRelevancy",
        make_metric_class(0.8, calls[ANSWER_RELEVANCY]),
    )
    monkeypatch.setattr(
        "ragas.metrics.collections.ContextPrecisionWithoutReference",
        make_metric_class(0.7, calls[CONTEXT_PRECISION]),
    )
    return calls


def make_evaluator(metrics_list: list[str] | None = None) -> RagasEvaluator:
    """Build an evaluator that has a model to judge with."""
    config = EvaluationSettings(
        enabled=True,
        provider="ragas",
        metrics=metrics_list if metrics_list is not None else [],
    )
    return RagasEvaluator(config, llm=object(), embeddings=object())


class TestRagasEvaluator:
    def test_the_metrics_hold_the_faithfulness_and_the_answer_relevancy(self, metrics):
        evaluator = make_evaluator()

        result = evaluator.evaluate(
            "what is rag",
            [{"id": "c1", "text": "rag is retrieval augmented generation"}],
            generated_answer="rag is retrieval augmented generation",
        )

        assert result[FAITHFULNESS] == pytest.approx(0.9)
        assert result[ANSWER_RELEVANCY] == pytest.approx(0.8)

    def test_the_context_precision_is_computed_too(self, metrics):
        evaluator = make_evaluator()

        result = evaluator.evaluate(
            "what is rag",
            [{"id": "c1", "text": "rag is retrieval augmented generation"}],
            generated_answer="rag is retrieval augmented generation",
        )

        assert result[CONTEXT_PRECISION] == pytest.approx(0.7)

    def test_all_three_metrics_are_computed_when_none_are_named(self, metrics):
        evaluator = make_evaluator()

        result = evaluator.evaluate(
            "what is rag", [{"id": "c1", "text": "a chunk"}], generated_answer="an answer"
        )

        assert set(result) == {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}

    def test_only_the_requested_metrics_are_computed(self, metrics):
        evaluator = make_evaluator([FAITHFULNESS])

        result = evaluator.evaluate(
            "what is rag", [{"id": "c1", "text": "a chunk"}], generated_answer="an answer"
        )

        assert set(result) == {FAITHFULNESS}

    def test_a_metric_that_ragas_cannot_compute_is_rejected(self):
        with pytest.raises(EvaluatorError, match="hit_rate"):
            make_evaluator(["hit_rate"])

    def test_the_contexts_are_handed_over_to_ragas(self, metrics):
        evaluator = make_evaluator([FAITHFULNESS])

        evaluator.evaluate(
            "what is rag",
            [{"id": "c1", "text": "the first chunk"}, {"id": "c2", "text": "the second"}],
            generated_answer="an answer",
        )

        assert metrics[FAITHFULNESS][0]["retrieved_contexts"] == [
            "the first chunk",
            "the second",
        ]

    def test_the_text_is_read_from_a_chunk_object(self, metrics):
        class Chunk:
            text = "the text of the chunk"

        evaluator = make_evaluator([FAITHFULNESS])

        evaluator.evaluate("what is rag", [Chunk()], generated_answer="an answer")

        assert metrics[FAITHFULNESS][0]["retrieved_contexts"] == ["the text of the chunk"]

    def test_the_question_and_the_answer_are_handed_over_to_ragas(self, metrics):
        evaluator = make_evaluator([FAITHFULNESS])

        evaluator.evaluate(
            "what is rag", [{"id": "c1", "text": "a chunk"}], generated_answer="the answer"
        )

        call = metrics[FAITHFULNESS][0]
        assert call["user_input"] == "what is rag"
        assert call["response"] == "the answer"

    def test_an_answer_that_is_missing_is_refused(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError, match="generated_answer"):
            evaluator.evaluate("what is rag", [{"id": "c1", "text": "a chunk"}])

    def test_an_answer_that_is_empty_is_refused(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError, match="generated_answer"):
            evaluator.evaluate(
                "what is rag", [{"id": "c1", "text": "a chunk"}], generated_answer="   "
            )

    def test_an_empty_query_is_refused(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError):
            evaluator.evaluate(
                "   ", [{"id": "c1", "text": "a chunk"}], generated_answer="an answer"
            )

    def test_an_empty_retrieved_list_is_refused(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError):
            evaluator.evaluate("what is rag", [], generated_answer="an answer")

    def test_a_failing_metric_is_reported_as_a_evaluator_error(self, monkeypatch):
        class BrokenMetric:
            def __init__(self, **kwargs) -> None:
                pass

            def score(self, **kwargs):
                raise RuntimeError("the model is down")

        monkeypatch.setattr("ragas.metrics.collections.Faithfulness", BrokenMetric)
        evaluator = make_evaluator([FAITHFULNESS])

        with pytest.raises(EvaluatorError, match="the model is down"):
            evaluator.evaluate(
                "what is rag", [{"id": "c1", "text": "a chunk"}], generated_answer="an answer"
            )

    def test_the_work_is_recorded_in_the_trace(self, metrics):
        from core.trace.trace_context import TraceContext

        evaluator = make_evaluator([FAITHFULNESS])
        trace = TraceContext(trace_type="query")

        evaluator.evaluate(
            "what is rag",
            [{"id": "c1", "text": "a chunk"}],
            generated_answer="an answer",
            trace=trace,
        )

        assert trace.get_stage_data("ragas_evaluation")[FAITHFULNESS] == pytest.approx(0.9)

    def test_the_same_input_gives_the_same_metrics(self, metrics):
        evaluator = make_evaluator()

        first = evaluator.evaluate(
            "what is rag", [{"id": "c1", "text": "a chunk"}], generated_answer="an answer"
        )
        second = evaluator.evaluate(
            "what is rag", [{"id": "c1", "text": "a chunk"}], generated_answer="an answer"
        )

        assert first == second


class TestRagasProvider:
    def test_the_ragas_provider_is_registered(self):
        from libs.evaluator.evaluator_factory import EvaluatorFactory

        assert "ragas" in EvaluatorFactory.registered_providers()

    def test_the_ragas_provider_builds_the_ragas_evaluator(self):
        from libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = Settings(
            evaluation=EvaluationSettings(enabled=True, provider="ragas", metrics=[])
        )

        evaluator = EvaluatorFactory.create(settings)

        assert isinstance(evaluator, RagasEvaluator)

    def test_the_ragas_provider_uses_the_configured_metrics(self):
        from libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True, provider="ragas", metrics=[FAITHFULNESS]
            )
        )

        evaluator = EvaluatorFactory.create(settings)

        assert evaluator.metrics == [FAITHFULNESS]
