"""Unit tests for the embedding abstraction and the provider factory."""

from __future__ import annotations

import pytest

from core.settings import EmbeddingSettings, Settings
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """Stands in for a real provider so the factory is exercised offline."""

    def __init__(self, config: EmbeddingSettings) -> None:
        super().__init__(config)
        self.batches: list[list[str]] = []

    def embed(
        self, texts: list[str], trace=None
    ) -> list[list[float]]:
        self.validate_texts(texts)
        self.batches.append(list(texts))
        return [self.vector_of(text) for text in texts]

    @staticmethod
    def vector_of(text: str) -> list[float]:
        """A stable vector: the same text always yields the same numbers."""
        return [float(len(text)), float(sum(ord(char) for char in text) % 100)]


class OtherFakeEmbedding(FakeEmbedding):
    """Second fake, used to prove a re-registration replaces the builder."""


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        embedding=EmbeddingSettings(
            provider=provider, model="fake-model", api_key="fake-key"
        )
    )


@pytest.fixture
def register_provider():
    """Register providers and remove them again once the test finishes."""
    registered: list[str] = []

    def _register(name: str, builder) -> str:
        EmbeddingFactory.register(name, builder)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        EmbeddingFactory.unregister(name)


class TestBaseEmbedding:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseEmbedding(EmbeddingSettings())

    def test_subclass_must_implement_embed(self):
        class IncompleteEmbedding(BaseEmbedding):
            pass

        with pytest.raises(TypeError):
            IncompleteEmbedding(EmbeddingSettings())

    def test_config_is_available_to_the_implementation(self):
        config = EmbeddingSettings(provider="fake", model="fake-model")
        embedding = FakeEmbedding(config)

        assert embedding.config is config
        assert embedding.config.model == "fake-model"

    def test_embed_returns_one_vector_per_text(self):
        embedding = FakeEmbedding(EmbeddingSettings())

        vectors = embedding.embed(["a", "bb"])

        assert len(vectors) == 2
        assert all(isinstance(vector, list) for vector in vectors)

    def test_embed_keeps_the_order_of_the_input_texts(self):
        embedding = FakeEmbedding(EmbeddingSettings())

        vectors = embedding.embed(["a", "bb", "ccc"])

        assert vectors == [
            FakeEmbedding.vector_of("a"),
            FakeEmbedding.vector_of("bb"),
            FakeEmbedding.vector_of("ccc"),
        ]

    def test_embed_is_stable_across_calls(self):
        embedding = FakeEmbedding(EmbeddingSettings())

        first = embedding.embed(["hello"])
        second = embedding.embed(["hello"])

        assert first == second

    def test_embed_sends_the_whole_batch_in_one_call(self):
        embedding = FakeEmbedding(EmbeddingSettings())

        embedding.embed(["a", "b", "c"])

        assert embedding.batches == [["a", "b", "c"]]

    def test_empty_batch_is_rejected(self):
        embedding = FakeEmbedding(EmbeddingSettings())

        with pytest.raises(ValueError):
            embedding.embed([])

    def test_non_string_input_is_rejected(self):
        embedding = FakeEmbedding(EmbeddingSettings())

        with pytest.raises(ValueError):
            embedding.embed(["ok", 42])


class TestEmbeddingFactory:
    def test_create_returns_the_registered_provider(self, register_provider):
        register_provider("fake", lambda settings: FakeEmbedding(settings.embedding))

        embedding = EmbeddingFactory.create(make_settings("fake"))

        assert isinstance(embedding, FakeEmbedding)

    def test_create_passes_the_embedding_configuration_to_the_provider(
        self, register_provider
    ):
        register_provider("fake", lambda settings: FakeEmbedding(settings.embedding))

        embedding = EmbeddingFactory.create(make_settings("fake"))

        assert embedding.config.model == "fake-model"
        assert embedding.config.api_key == "fake-key"

    def test_provider_name_is_matched_case_insensitively(self, register_provider):
        register_provider("fake", lambda settings: FakeEmbedding(settings.embedding))

        embedding = EmbeddingFactory.create(make_settings("FAKE"))

        assert isinstance(embedding, FakeEmbedding)

    def test_unknown_provider_raises_a_readable_error(self):
        with pytest.raises(EmbeddingError) as exc_info:
            EmbeddingFactory.create(make_settings("does-not-exist"))

        message = str(exc_info.value)
        assert "does-not-exist" in message
        assert "Registered providers" in message

    def test_missing_provider_raises_a_readable_error(self):
        with pytest.raises(EmbeddingError) as exc_info:
            EmbeddingFactory.create(make_settings("   "))

        assert "embedding.provider" in str(exc_info.value)

    def test_builder_returning_a_non_embedding_is_rejected(self, register_provider):
        register_provider("broken", lambda settings: "not an embedding")

        with pytest.raises(EmbeddingError) as exc_info:
            EmbeddingFactory.create(make_settings("broken"))

        assert "BaseEmbedding" in str(exc_info.value)

    def test_registering_a_provider_twice_replaces_the_builder(
        self, register_provider
    ):
        register_provider("fake", lambda settings: FakeEmbedding(settings.embedding))
        register_provider("fake", lambda settings: OtherFakeEmbedding(settings.embedding))

        embedding = EmbeddingFactory.create(make_settings("fake"))

        assert isinstance(embedding, OtherFakeEmbedding)

    def test_registered_providers_lists_registered_names(self, register_provider):
        register_provider("fake", lambda settings: FakeEmbedding(settings.embedding))

        assert "fake" in EmbeddingFactory.registered_providers()

    def test_unregister_removes_the_provider(self, register_provider):
        register_provider("fake", lambda settings: FakeEmbedding(settings.embedding))

        EmbeddingFactory.unregister("fake")

        assert "fake" not in EmbeddingFactory.registered_providers()

    def test_registering_an_empty_provider_name_is_rejected(self):
        with pytest.raises(ValueError):
            EmbeddingFactory.register("", lambda settings: FakeEmbedding(settings.embedding))
