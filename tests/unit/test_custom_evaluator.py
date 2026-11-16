"""Unit tests for the custom evaluator and the evaluation provider factory."""

from __future__ import annotations

import pytest

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError, NoneEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


class RetrievedChunk:
    """A chunk carrying its identifier as an attribute rather than a mapping."""

    def __init__(self, chunk_id: str) -> None:
        self.id = chunk_id


class NamelessChunk:
    """A chunk that carries no identifier at all."""

    def __init__(self, text: str = "first") -> None:
        self.text = text


def make_evaluator(metrics: list[str] | None = None) -> CustomEvaluator:
    return CustomEvaluator(EvaluationSettings(provider="custom", metrics=metrics or []))


def make_settings(provider: str = "custom", enabled: bool = True) -> Settings:
    return Settings(
        evaluation=EvaluationSettings(
            enabled=enabled,
            provider=provider,
            metrics=["hit_rate", "mrr"],
        )
    )


class TestBaseEvaluator:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseEvaluator(EvaluationSettings())

    def test_subclass_must_implement_evaluate(self):
        class IncompleteEvaluator(BaseEvaluator):
            pass

        with pytest.raises(TypeError):
            IncompleteEvaluator(EvaluationSettings())

    def test_validate_query_rejects_a_query_that_is_not_a_string(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError, match="query must be a string"):
            evaluator.validate_query(None)

    def test_validate_query_rejects_an_empty_query(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError, match="query cannot be empty"):
            evaluator.validate_query("   ")

    def test_validate_retrieved_chunks_rejects_a_list_that_is_not_a_list(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError, match="must be a list"):
            evaluator.validate_retrieved_chunks("not a list")

    def test_validate_retrieved_chunks_rejects_an_empty_list(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError, match="cannot be empty"):
            evaluator.validate_retrieved_chunks([])


class TestNoneEvaluator:
    def test_it_returns_no_metrics(self):
        evaluator = NoneEvaluator(EvaluationSettings())

        assert evaluator.evaluate("query", ["a"], ground_truth=["a"]) == {}

    def test_it_still_refuses_an_empty_query(self):
        evaluator = NoneEvaluator(EvaluationSettings())

        with pytest.raises(ValueError):
            evaluator.evaluate("  ", ["a"])

    def test_it_still_refuses_an_empty_chunk_list(self):
        evaluator = NoneEvaluator(EvaluationSettings())

        with pytest.raises(ValueError):
            evaluator.evaluate("query", [])


class TestCustomEvaluator:
    def test_hit_rate_is_one_when_a_golden_chunk_is_retrieved(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a", "b"], ground_truth=["b"])

        assert metrics["hit_rate"] == 1.0

    def test_hit_rate_is_zero_when_no_golden_chunk_is_retrieved(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a", "b"], ground_truth=["c"])

        assert metrics["hit_rate"] == 0.0

    def test_mrr_is_the_reciprocal_of_the_first_matching_rank(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a", "b", "c"], ground_truth=["c"])

        assert metrics["mrr"] == pytest.approx(1 / 3)

    def test_mrr_is_zero_when_no_golden_chunk_is_retrieved(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a"], ground_truth=["z"])

        assert metrics["mrr"] == 0.0

    def test_metrics_default_to_hit_rate_and_mrr(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a"], ground_truth=["a"])

        assert set(metrics) == {"hit_rate", "mrr"}

    def test_only_the_requested_metrics_are_returned(self):
        evaluator = make_evaluator(metrics=["mrr"])

        metrics = evaluator.evaluate("query", ["a"], ground_truth=["a"])

        assert set(metrics) == {"mrr"}

    def test_an_unsupported_metric_is_rejected(self):
        with pytest.raises(EvaluatorError, match="faithfulness"):
            make_evaluator(metrics=["faithfulness"])

    def test_the_same_input_produces_the_same_metrics(self):
        evaluator = make_evaluator()

        first = evaluator.evaluate("query", ["a", "b"], ground_truth=["b"])
        second = evaluator.evaluate("query", ["a", "b"], ground_truth=["b"])

        assert first == second

    def test_ids_are_read_from_the_id_field(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate(
            "query", [{"id": "a", "text": "first"}], ground_truth=["a"]
        )

        assert metrics["hit_rate"] == 1.0

    def test_ids_are_read_from_alternative_id_fields(self):
        evaluator = make_evaluator()

        for field in ("chunk_id", "document_id", "doc_id"):
            metrics = evaluator.evaluate(
                "query", [{field: "a"}], ground_truth=["a"]
            )
            assert metrics["hit_rate"] == 1.0, field

    def test_ids_are_read_from_an_object_attribute(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", [RetrievedChunk("a")], ground_truth=["a"])

        assert metrics["hit_rate"] == 1.0

    def test_a_record_without_any_id_field_is_rejected(self):
        evaluator = make_evaluator()

        with pytest.raises(EvaluatorError, match="Missing id field"):
            evaluator.evaluate("query", [{"text": "first"}], ground_truth=["a"])

    def test_golden_ids_accept_a_plain_string(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a"], ground_truth="a")

        assert metrics["hit_rate"] == 1.0

    def test_golden_ids_accept_a_mapping_with_ids(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a"], ground_truth={"ids": ["a"]})

        assert metrics["hit_rate"] == 1.0

    def test_metrics_are_zero_without_golden_ids(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a"])

        assert metrics == {"hit_rate": 0.0, "mrr": 0.0}

    def test_an_empty_query_is_rejected(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError):
            evaluator.evaluate("   ", ["a"], ground_truth=["a"])

    def test_an_empty_retrieved_list_is_rejected(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError):
            evaluator.evaluate("query", [], ground_truth=["a"])

    def test_a_query_that_is_not_a_string_is_rejected(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError, match="query must be a string"):
            evaluator.evaluate(None, ["a"], ground_truth=["a"])

    def test_a_retrieved_list_that_is_not_a_list_is_rejected(self):
        evaluator = make_evaluator()

        with pytest.raises(ValueError, match="must be a list"):
            evaluator.evaluate("query", "not a list", ground_truth=["a"])

    def test_metric_names_are_matched_without_regard_to_case(self):
        evaluator = make_evaluator(metrics=["HIT_RATE"])

        metrics = evaluator.evaluate("query", ["a"], ground_truth=["a"])

        assert set(metrics) == {"hit_rate"}

    def test_metric_names_are_matched_without_regard_to_spaces(self):
        evaluator = make_evaluator(metrics=[" hit_rate "])

        metrics = evaluator.evaluate("query", ["a"], ground_truth=["a"])

        assert set(metrics) == {"hit_rate"}

    def test_a_repeated_metric_is_reported_once(self):
        evaluator = make_evaluator(metrics=["hit_rate", "hit_rate"])

        metrics = evaluator.evaluate("query", ["a"], ground_truth=["a"])

        assert set(metrics) == {"hit_rate"}

    def test_a_chunk_without_any_identifier_is_rejected(self):
        evaluator = make_evaluator()

        with pytest.raises(EvaluatorError, match="Unable to extract id"):
            evaluator.evaluate("query", [NamelessChunk()], ground_truth=["a"])

    def test_golden_ids_accept_a_mapping_without_an_ids_key(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a"], ground_truth={"chunk_id": "a"})

        assert metrics["hit_rate"] == 1.0

    def test_golden_ids_accept_a_list_of_mappings(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ["a"], ground_truth=[{"id": "a"}])

        assert metrics["hit_rate"] == 1.0

    def test_golden_ids_of_an_unsupported_type_are_rejected(self):
        evaluator = make_evaluator()

        with pytest.raises(EvaluatorError, match="Unsupported ground_truth type"):
            evaluator.evaluate("query", ["a"], ground_truth=42)

    def test_the_generated_answer_does_not_change_the_metrics(self):
        evaluator = make_evaluator()

        without = evaluator.evaluate("query", ["a"], ground_truth=["a"])
        with_answer = evaluator.evaluate(
            "query", ["a"], generated_answer="a long answer", ground_truth=["a"]
        )

        assert without == with_answer

    def test_a_tuple_of_chunks_is_accepted(self):
        evaluator = make_evaluator()

        metrics = evaluator.evaluate("query", ("a", "b"), ground_truth=["b"])

        assert metrics["hit_rate"] == 1.0


class TestEvaluatorFactory:
    def test_custom_provider_returns_the_custom_evaluator(self):
        evaluator = EvaluatorFactory.create(make_settings("custom"))

        assert isinstance(evaluator, CustomEvaluator)

    def test_custom_evaluator_uses_the_configured_metrics(self):
        settings = Settings(
            evaluation=EvaluationSettings(enabled=True, provider="custom", metrics=["mrr"])
        )

        evaluator = EvaluatorFactory.create(settings)

        assert evaluator.metrics == ["mrr"]

    def test_disabled_evaluation_falls_back_to_the_none_evaluator(self):
        evaluator = EvaluatorFactory.create(make_settings("custom", enabled=False))

        assert isinstance(evaluator, NoneEvaluator)

    def test_none_provider_returns_the_none_evaluator(self):
        evaluator = EvaluatorFactory.create(make_settings("none"))

        assert isinstance(evaluator, NoneEvaluator)

    def test_the_none_evaluator_returns_no_metrics(self):
        evaluator = EvaluatorFactory.create(make_settings("none"))

        assert evaluator.evaluate("query", ["a"], ground_truth=["a"]) == {}

    def test_provider_name_is_matched_case_insensitively(self):
        evaluator = EvaluatorFactory.create(make_settings("CUSTOM"))

        assert isinstance(evaluator, CustomEvaluator)

    def test_unknown_provider_raises_a_readable_error(self):
        with pytest.raises(EvaluatorError) as exc_info:
            EvaluatorFactory.create(make_settings("does-not-exist"))

        message = str(exc_info.value)
        assert "does-not-exist" in message
        assert "Registered providers" in message

    def test_missing_provider_raises_a_readable_error(self):
        with pytest.raises(EvaluatorError) as exc_info:
            EvaluatorFactory.create(make_settings("   "))

        assert "evaluation.provider" in str(exc_info.value)

    def test_builder_returning_a_non_evaluator_is_rejected(self):
        EvaluatorFactory.register("broken", lambda settings: "not an evaluator")
        try:
            with pytest.raises(EvaluatorError) as exc_info:
                EvaluatorFactory.create(make_settings("broken"))
        finally:
            EvaluatorFactory.unregister("broken")

        assert "BaseEvaluator" in str(exc_info.value)

    def test_registering_a_provider_twice_replaces_the_builder(self):
        class OtherEvaluator(BaseEvaluator):
            def evaluate(
                self,
                query,
                retrieved_chunks,
                generated_answer=None,
                ground_truth=None,
                trace=None,
            ):
                return {}

        EvaluatorFactory.register("custom", lambda settings: OtherEvaluator(settings.evaluation))
        try:
            evaluator = EvaluatorFactory.create(make_settings("custom"))
        finally:
            EvaluatorFactory.register(
                "custom", lambda settings: CustomEvaluator(settings.evaluation)
            )

        assert isinstance(evaluator, OtherEvaluator)

    def test_registered_providers_lists_registered_names(self):
        assert "custom" in EvaluatorFactory.registered_providers()

    def test_unregister_removes_the_provider(self):
        EvaluatorFactory.register(
            "temporary", lambda settings: CustomEvaluator(settings.evaluation)
        )
        EvaluatorFactory.unregister("temporary")

        assert "temporary" not in EvaluatorFactory.registered_providers()

    def test_registering_an_empty_provider_name_is_rejected(self):
        with pytest.raises(ValueError):
            EvaluatorFactory.register("", lambda settings: CustomEvaluator(settings.evaluation))

    def test_registering_a_provider_normalises_the_name(self):
        EvaluatorFactory.register(
            "  Temporary  ", lambda settings: CustomEvaluator(settings.evaluation)
        )
        try:
            assert "temporary" in EvaluatorFactory.registered_providers()
        finally:
            EvaluatorFactory.unregister("temporary")

    def test_unregistering_an_unknown_provider_is_ignored(self):
        EvaluatorFactory.unregister("never-registered")

        assert "never-registered" not in EvaluatorFactory.registered_providers()

    def test_a_provider_name_with_spaces_is_accepted(self):
        evaluator = EvaluatorFactory.create(make_settings("  custom  "))

        assert isinstance(evaluator, CustomEvaluator)

    def test_a_builder_that_fails_is_not_hidden(self):
        def broken(settings):
            raise RuntimeError("the framework is not there")

        EvaluatorFactory.register("broken", broken)
        try:
            with pytest.raises(RuntimeError, match="the framework is not there"):
                EvaluatorFactory.create(make_settings("broken"))
        finally:
            EvaluatorFactory.unregister("broken")
