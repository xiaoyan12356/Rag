"""Unit tests for the composite evaluator.

These tests pin how the composite evaluator runs the evaluators it holds:
that their metrics are merged, that one failure does not take the others
down, and that the backends of the settings decide which evaluators are run.
"""

from __future__ import annotations

import pytest

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.composite_evaluator import CompositeEvaluator


class RecordingEvaluator(BaseEvaluator):
    """An evaluator that answers with the metrics it was given."""

    def __init__(self, config: EvaluationSettings, metrics: dict[str, float]) -> None:
        super().__init__(config)
        self._metrics = metrics
        self.calls: list[dict] = []

    def evaluate(
        self,
        query,
        retrieved_chunks,
        generated_answer=None,
        ground_truth=None,
        trace=None,
    ):
        self.calls.append(
            {
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "generated_answer": generated_answer,
                "ground_truth": ground_truth,
            }
        )
        return dict(self._metrics)


class BrokenEvaluator(BaseEvaluator):
    """An evaluator that always fails."""

    def __init__(self, config: EvaluationSettings) -> None:
        super().__init__(config)

    def evaluate(
        self,
        query,
        retrieved_chunks,
        generated_answer=None,
        ground_truth=None,
        trace=None,
    ):
        raise RuntimeError("the framework is down")


def make_config() -> EvaluationSettings:
    return EvaluationSettings(enabled=True, provider="composite", metrics=[])


def make_composite(*evaluators: BaseEvaluator) -> CompositeEvaluator:
    return CompositeEvaluator(make_config(), evaluators=list(evaluators))


class TestCompositeEvaluator:
    def test_the_metrics_of_every_evaluator_are_merged(self):
        first = RecordingEvaluator(make_config(), {"faithfulness": 0.9})
        second = RecordingEvaluator(make_config(), {"hit_rate": 1.0})
        composite = make_composite(first, second)

        metrics = composite.evaluate("query", ["a"], ground_truth=["a"])

        assert metrics == {"faithfulness": 0.9, "hit_rate": 1.0}

    def test_every_evaluator_is_run(self):
        first = RecordingEvaluator(make_config(), {"faithfulness": 0.9})
        second = RecordingEvaluator(make_config(), {"hit_rate": 1.0})
        composite = make_composite(first, second)

        composite.evaluate("query", ["a"])

        assert len(first.calls) == 1
        assert len(second.calls) == 1

    def test_the_evaluators_get_the_same_arguments(self):
        first = RecordingEvaluator(make_config(), {"faithfulness": 0.9})
        second = RecordingEvaluator(make_config(), {"hit_rate": 1.0})
        composite = make_composite(first, second)

        composite.evaluate(
            "query", ["a"], generated_answer="the answer", ground_truth=["a"]
        )

        assert first.calls[0] == second.calls[0]
        assert first.calls[0]["generated_answer"] == "the answer"

    def test_a_failing_evaluator_does_not_stop_the_others(self):
        broken = BrokenEvaluator(make_config())
        working = RecordingEvaluator(make_config(), {"hit_rate": 1.0})
        composite = make_composite(broken, working)

        metrics = composite.evaluate("query", ["a"])

        assert metrics == {"hit_rate": 1.0}

    def test_a_failing_evaluator_is_reported_when_everyone_fails(self):
        composite = make_composite(BrokenEvaluator(make_config()))

        with pytest.raises(EvaluatorError, match="the framework is down"):
            composite.evaluate("query", ["a"])

    def test_the_metrics_of_the_last_evaluator_win_when_they_share_a_name(self):
        first = RecordingEvaluator(make_config(), {"score": 0.5})
        second = RecordingEvaluator(make_config(), {"score": 0.8})
        composite = make_composite(first, second)

        metrics = composite.evaluate("query", ["a"])

        assert metrics["score"] == pytest.approx(0.8)

    def test_a_composite_without_evaluators_is_refused(self):
        with pytest.raises(EvaluatorError, match="at least one evaluator"):
            CompositeEvaluator(make_config(), evaluators=[])

    def test_an_empty_query_is_refused(self):
        composite = make_composite(RecordingEvaluator(make_config(), {"score": 1.0}))

        with pytest.raises(ValueError):
            composite.evaluate("   ", ["a"])

    def test_an_empty_retrieved_list_is_refused(self):
        composite = make_composite(RecordingEvaluator(make_config(), {"score": 1.0}))

        with pytest.raises(ValueError):
            composite.evaluate("query", [])

    def test_the_evaluators_can_be_read_back(self):
        first = RecordingEvaluator(make_config(), {"faithfulness": 0.9})
        composite = make_composite(first)

        assert composite.evaluators == [first]

    def test_the_same_input_gives_the_same_metrics(self):
        composite = make_composite(
            RecordingEvaluator(make_config(), {"faithfulness": 0.9}),
            RecordingEvaluator(make_config(), {"hit_rate": 1.0}),
        )

        first = composite.evaluate("query", ["a"])
        second = composite.evaluate("query", ["a"])

        assert first == second


class TestCompositeFromSettings:
    def test_the_backends_of_the_settings_decide_the_evaluators(self):
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="composite",
                backends=["custom"],
                metrics=["hit_rate", "mrr"],
            )
        )

        composite = CompositeEvaluator(settings.evaluation, settings=settings)

        assert len(composite.evaluators) == 1
        assert isinstance(composite.evaluators[0], CustomEvaluator)

    def test_the_metrics_of_the_custom_evaluator_come_through(self):
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="composite",
                backends=["custom"],
                metrics=["hit_rate", "mrr"],
            )
        )
        composite = CompositeEvaluator(settings.evaluation, settings=settings)

        metrics = composite.evaluate("query", ["a"], ground_truth=["a"])

        assert metrics == {"hit_rate": 1.0, "mrr": 1.0}

    def test_the_composite_never_builds_itself(self):
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="composite",
                backends=["composite", "custom"],
                metrics=["hit_rate"],
            )
        )

        composite = CompositeEvaluator(settings.evaluation, settings=settings)

        assert len(composite.evaluators) == 1

    def test_a_backend_that_cannot_be_built_is_left_out(self):
        settings = Settings(
            evaluation=EvaluationSettings(
                enabled=True,
                provider="composite",
                backends=["does-not-exist", "custom"],
                metrics=["hit_rate"],
            )
        )

        composite = CompositeEvaluator(settings.evaluation, settings=settings)

        assert len(composite.evaluators) == 1

    def test_a_composite_without_backends_is_refused(self):
        settings = Settings(
            evaluation=EvaluationSettings(enabled=True, provider="composite", backends=[])
        )

        with pytest.raises(EvaluatorError, match="at least one evaluator"):
            CompositeEvaluator(settings.evaluation, settings=settings)
