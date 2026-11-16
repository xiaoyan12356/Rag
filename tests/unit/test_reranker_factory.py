"""Unit tests for the reranker abstraction and the strategy factory."""

from __future__ import annotations

import pytest

from core.settings import RerankSettings, Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankerError
from libs.reranker.reranker_factory import RerankerFactory


class FakeReranker(BaseReranker):
    """Stands in for a real strategy so the factory is exercised offline."""

    def __init__(self, config: RerankSettings) -> None:
        super().__init__(config)
        self.calls: list[tuple[str, list[dict]]] = []

    def rerank(self, query, candidates, trace=None):
        self.calls.append((query, list(candidates)))
        return list(candidates)


class OtherFakeReranker(FakeReranker):
    """Second fake, used to prove a re-registration replaces the builder."""


def make_settings(provider: str = "fake", enabled: bool = True) -> Settings:
    return Settings(
        rerank=RerankSettings(
            enabled=enabled,
            provider=provider,
            model="fake-model",
            top_k=5,
        )
    )


def make_candidates() -> list[dict]:
    return [
        {"id": "a", "text": "first", "score": 0.9},
        {"id": "b", "text": "second", "score": 0.8},
    ]


@pytest.fixture
def register_strategy():
    """Register strategies and remove them again once the test finishes."""
    registered: list[str] = []

    def _register(name: str, builder) -> str:
        RerankerFactory.register(name, builder)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        RerankerFactory.unregister(name)


class TestBaseReranker:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseReranker(RerankSettings())

    def test_subclass_must_implement_rerank(self):
        class IncompleteReranker(BaseReranker):
            pass

        with pytest.raises(TypeError):
            IncompleteReranker(RerankSettings())

    def test_config_is_available_to_the_implementation(self):
        config = RerankSettings(provider="fake", model="fake-model", top_k=3)
        reranker = FakeReranker(config)

        assert reranker.config is config
        assert reranker.config.top_k == 3

    def test_validate_query_rejects_an_empty_query(self):
        reranker = FakeReranker(RerankSettings())

        with pytest.raises(ValueError, match="query cannot be empty"):
            reranker.validate_query("   ")

    def test_validate_query_rejects_a_query_that_is_not_a_string(self):
        reranker = FakeReranker(RerankSettings())

        with pytest.raises(ValueError, match="query must be a string"):
            reranker.validate_query(None)

    def test_validate_candidates_rejects_an_empty_candidate_list(self):
        reranker = FakeReranker(RerankSettings())

        with pytest.raises(ValueError, match="candidates cannot be empty"):
            reranker.validate_candidates([])

    def test_validate_candidates_rejects_a_candidate_that_is_not_a_mapping(self):
        reranker = FakeReranker(RerankSettings())

        with pytest.raises(ValueError, match="not a mapping"):
            reranker.validate_candidates(["not a candidate"])

    def test_validate_candidates_rejects_a_candidate_list_that_is_not_a_list(self):
        reranker = FakeReranker(RerankSettings())

        with pytest.raises(ValueError, match="must be a list of records"):
            reranker.validate_candidates("not a list")

    def test_the_fallback_flag_starts_as_false(self):
        reranker = FakeReranker(RerankSettings())

        assert reranker.fallback_used is False


class TestNoneReranker:
    def test_keeps_the_original_order(self):
        reranker = NoneReranker(RerankSettings())

        candidates = make_candidates()
        ranked = reranker.rerank("query", candidates)

        assert [candidate["id"] for candidate in ranked] == ["a", "b"]

    def test_returns_a_copy_rather_than_the_same_list(self):
        reranker = NoneReranker(RerankSettings())

        candidates = make_candidates()
        ranked = reranker.rerank("query", candidates)

        assert ranked is not candidates

    def test_preserves_the_candidate_records(self):
        reranker = NoneReranker(RerankSettings())

        ranked = reranker.rerank("query", make_candidates())

        assert ranked[0]["text"] == "first"
        assert ranked[0]["score"] == 0.9

    def test_rejects_an_empty_query(self):
        reranker = NoneReranker(RerankSettings())

        with pytest.raises(ValueError):
            reranker.rerank("", make_candidates())

    def test_rejects_an_empty_candidate_list(self):
        reranker = NoneReranker(RerankSettings())

        with pytest.raises(ValueError):
            reranker.rerank("query", [])

    def test_rejects_a_candidate_that_is_not_a_mapping(self):
        reranker = NoneReranker(RerankSettings())

        with pytest.raises(ValueError, match="not a mapping"):
            reranker.rerank("query", ["not a candidate"])

    def test_keeps_the_same_records_rather_than_copies(self):
        reranker = NoneReranker(RerankSettings())
        candidates = make_candidates()

        ranked = reranker.rerank("query", candidates)

        assert ranked[0] is candidates[0]


class TestRerankerFactory:
    def test_none_provider_returns_the_none_reranker(self):
        reranker = RerankerFactory.create(make_settings("none"))

        assert isinstance(reranker, NoneReranker)

    def test_none_provider_wins_over_the_enabled_flag(self):
        reranker = RerankerFactory.create(make_settings("none", enabled=True))

        assert isinstance(reranker, NoneReranker)

    def test_disabled_reranking_falls_back_to_the_none_reranker(self, register_strategy):
        register_strategy("fake", lambda settings: FakeReranker(settings.rerank))

        reranker = RerankerFactory.create(make_settings("fake", enabled=False))

        assert isinstance(reranker, NoneReranker)

    def test_create_returns_the_registered_strategy(self, register_strategy):
        register_strategy("fake", lambda settings: FakeReranker(settings.rerank))

        reranker = RerankerFactory.create(make_settings("fake"))

        assert isinstance(reranker, FakeReranker)

    def test_create_passes_the_rerank_configuration_to_the_strategy(self, register_strategy):
        register_strategy("fake", lambda settings: FakeReranker(settings.rerank))

        reranker = RerankerFactory.create(make_settings("fake"))

        assert reranker.config.model == "fake-model"
        assert reranker.config.top_k == 5

    def test_provider_name_is_matched_case_insensitively(self, register_strategy):
        register_strategy("fake", lambda settings: FakeReranker(settings.rerank))

        reranker = RerankerFactory.create(make_settings("FAKE"))

        assert isinstance(reranker, FakeReranker)

    def test_unknown_provider_raises_a_readable_error(self):
        with pytest.raises(RerankerError) as exc_info:
            RerankerFactory.create(make_settings("does-not-exist"))

        message = str(exc_info.value)
        assert "does-not-exist" in message
        assert "Registered strategies" in message

    def test_unknown_provider_is_rejected_even_when_reranking_is_disabled(self):
        with pytest.raises(RerankerError):
            RerankerFactory.create(make_settings("does-not-exist", enabled=False))

    def test_missing_provider_raises_a_readable_error(self):
        with pytest.raises(RerankerError) as exc_info:
            RerankerFactory.create(make_settings("   "))

        assert "rerank.provider" in str(exc_info.value)

    def test_builder_returning_a_non_reranker_is_rejected(self, register_strategy):
        register_strategy("broken", lambda settings: "not a reranker")

        with pytest.raises(RerankerError) as exc_info:
            RerankerFactory.create(make_settings("broken"))

        assert "BaseReranker" in str(exc_info.value)

    def test_registering_a_strategy_twice_replaces_the_builder(self, register_strategy):
        register_strategy("fake", lambda settings: FakeReranker(settings.rerank))
        register_strategy("fake", lambda settings: OtherFakeReranker(settings.rerank))

        reranker = RerankerFactory.create(make_settings("fake"))

        assert isinstance(reranker, OtherFakeReranker)

    def test_registered_strategies_lists_registered_names(self, register_strategy):
        register_strategy("fake", lambda settings: FakeReranker(settings.rerank))

        assert "fake" in RerankerFactory.registered_strategies()

    def test_unregister_removes_the_strategy(self, register_strategy):
        register_strategy("fake", lambda settings: FakeReranker(settings.rerank))

        RerankerFactory.unregister("fake")

        assert "fake" not in RerankerFactory.registered_strategies()

    def test_registering_an_empty_strategy_name_is_rejected(self):
        with pytest.raises(ValueError):
            RerankerFactory.register("", lambda settings: FakeReranker(settings.rerank))

    def test_registering_a_strategy_normalises_the_name(self, register_strategy):
        register_strategy("  Fake  ", lambda settings: FakeReranker(settings.rerank))

        assert "fake" in RerankerFactory.registered_strategies()

    def test_unregistering_a_strategy_normalises_the_name(self, register_strategy):
        register_strategy("fake", lambda settings: FakeReranker(settings.rerank))

        RerankerFactory.unregister("  FAKE  ")

        assert "fake" not in RerankerFactory.registered_strategies()

    def test_unregistering_an_unknown_strategy_is_ignored(self):
        RerankerFactory.unregister("never-registered")

        assert "never-registered" not in RerankerFactory.registered_strategies()

    def test_a_provider_name_with_spaces_is_accepted(self, register_strategy):
        register_strategy("fake", lambda settings: FakeReranker(settings.rerank))

        reranker = RerankerFactory.create(make_settings("  fake  "))

        assert isinstance(reranker, FakeReranker)

    def test_a_builder_that_fails_is_not_hidden(self, register_strategy):
        def broken(settings):
            raise RuntimeError("the model is not there")

        register_strategy("fake", broken)

        with pytest.raises(RuntimeError, match="the model is not there"):
            RerankerFactory.create(make_settings("fake"))
