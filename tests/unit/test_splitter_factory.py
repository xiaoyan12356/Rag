"""Unit tests for the splitter abstraction and the strategy factory."""

from __future__ import annotations

import pytest

from core.settings import IngestionSettings, Settings
from libs.splitter.base_splitter import BaseSplitter, SplitterError
from libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """Stands in for a real strategy so the factory is exercised offline."""

    def __init__(self, config: IngestionSettings) -> None:
        super().__init__(config)
        self.sources: list[str] = []

    def split_text(self, text: str, trace=None) -> list[str]:
        self.sources.append(text)
        return [part for part in text.split() if part]


class FixedSplitter(FakeSplitter):
    """Second strategy, used to prove the factory routes by configuration."""

    def split_text(self, text: str, trace=None) -> list[str]:
        self.sources.append(text)
        size = self.config.chunk_size
        return [text[index : index + size] for index in range(0, len(text), size)]


def make_settings(strategy: str = "fake") -> Settings:
    return Settings(ingestion=IngestionSettings(splitter=strategy, chunk_size=10))


@pytest.fixture
def register_strategy():
    """Register strategies and remove them again once the test finishes."""
    registered: list[str] = []

    def _register(name: str, builder) -> str:
        SplitterFactory.register(name, builder)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        SplitterFactory.unregister(name)


class TestBaseSplitter:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseSplitter(IngestionSettings())

    def test_subclass_must_implement_split_text(self):
        class IncompleteSplitter(BaseSplitter):
            pass

        with pytest.raises(TypeError):
            IncompleteSplitter(IngestionSettings())

    def test_config_is_available_to_the_implementation(self):
        config = IngestionSettings(splitter="fake", chunk_size=10)
        splitter = FakeSplitter(config)

        assert splitter.config is config
        assert splitter.config.chunk_size == 10

    def test_split_text_returns_the_chunks(self):
        splitter = FakeSplitter(IngestionSettings())

        chunks = splitter.split_text("alpha beta")

        assert chunks == ["alpha", "beta"]

    def test_split_text_keeps_the_reading_order(self):
        splitter = FakeSplitter(IngestionSettings())

        chunks = splitter.split_text("first second third")

        assert chunks == ["first", "second", "third"]


class TestSplitterFactory:
    def test_create_returns_the_registered_strategy(self, register_strategy):
        register_strategy("fake", lambda settings: FakeSplitter(settings.ingestion))

        splitter = SplitterFactory.create(make_settings("fake"))

        assert isinstance(splitter, FakeSplitter)

    def test_create_passes_the_ingestion_configuration_to_the_strategy(
        self, register_strategy
    ):
        register_strategy("fake", lambda settings: FakeSplitter(settings.ingestion))

        splitter = SplitterFactory.create(make_settings("fake"))

        assert splitter.config.chunk_size == 10
        assert splitter.config.splitter == "fake"

    def test_different_configurations_return_different_strategies(
        self, register_strategy
    ):
        register_strategy("fake", lambda settings: FakeSplitter(settings.ingestion))
        register_strategy("fixed", lambda settings: FixedSplitter(settings.ingestion))

        assert isinstance(SplitterFactory.create(make_settings("fake")), FakeSplitter)
        assert isinstance(SplitterFactory.create(make_settings("fixed")), FixedSplitter)

    def test_strategy_name_is_matched_case_insensitively(self, register_strategy):
        register_strategy("fake", lambda settings: FakeSplitter(settings.ingestion))

        splitter = SplitterFactory.create(make_settings("FAKE"))

        assert isinstance(splitter, FakeSplitter)

    def test_unknown_strategy_raises_a_readable_error(self):
        with pytest.raises(SplitterError) as exc_info:
            SplitterFactory.create(make_settings("does-not-exist"))

        message = str(exc_info.value)
        assert "does-not-exist" in message
        assert "Registered strategies" in message

    def test_missing_strategy_raises_a_readable_error(self):
        with pytest.raises(SplitterError) as exc_info:
            SplitterFactory.create(make_settings("   "))

        assert "ingestion.splitter" in str(exc_info.value)

    def test_builder_returning_a_non_splitter_is_rejected(self, register_strategy):
        register_strategy("broken", lambda settings: "not a splitter")

        with pytest.raises(SplitterError) as exc_info:
            SplitterFactory.create(make_settings("broken"))

        assert "BaseSplitter" in str(exc_info.value)

    def test_registering_a_strategy_twice_replaces_the_builder(
        self, register_strategy
    ):
        register_strategy("fake", lambda settings: FakeSplitter(settings.ingestion))
        register_strategy("fake", lambda settings: FixedSplitter(settings.ingestion))

        splitter = SplitterFactory.create(make_settings("fake"))

        assert isinstance(splitter, FixedSplitter)

    def test_registered_strategies_lists_registered_names(self, register_strategy):
        register_strategy("fake", lambda settings: FakeSplitter(settings.ingestion))

        assert "fake" in SplitterFactory.registered_strategies()

    def test_unregister_removes_the_strategy(self, register_strategy):
        register_strategy("fake", lambda settings: FakeSplitter(settings.ingestion))

        SplitterFactory.unregister("fake")

        assert "fake" not in SplitterFactory.registered_strategies()

    def test_registering_an_empty_strategy_name_is_rejected(self):
        with pytest.raises(ValueError):
            SplitterFactory.register("", lambda settings: FakeSplitter(settings.ingestion))
