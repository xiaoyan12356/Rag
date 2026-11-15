"""Provider selection for the embedding layer.

The provider name comes from `config/settings.yaml`; the factory turns it into
the matching `BaseEmbedding` implementation, so switching backends is a config
edit rather than a code change.
"""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError

ProviderBuilder = Callable[[Settings], BaseEmbedding]


class EmbeddingFactory:
    """Registry mapping a configured provider name to its implementation."""

    _builders: dict[str, ProviderBuilder] = {}

    @classmethod
    def register(cls, provider: str, builder: ProviderBuilder) -> None:
        """Register `builder` as the implementation of `provider`."""
        name = provider.strip().lower()
        if not name:
            raise ValueError("Cannot register an embedding provider with an empty name")
        cls._builders[name] = builder

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration; unknown names are ignored."""
        cls._builders.pop(provider.strip().lower(), None)

    @classmethod
    def registered_providers(cls) -> list[str]:
        """Return the registered provider names, sorted."""
        return sorted(cls._builders)

    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        """Build the embedding provider named by `settings.embedding.provider`."""
        provider = (settings.embedding.provider or "").strip().lower()
        if not provider:
            raise EmbeddingError(
                "embedding.provider is not configured - set it in config/settings.yaml"
            )

        builder = cls._builders.get(provider)
        if builder is None:
            available = ", ".join(cls.registered_providers()) or "(none)"
            raise EmbeddingError(
                f"Unknown embedding provider '{provider}'. Registered providers: {available}"
            )

        embedding = builder(settings)
        if not isinstance(embedding, BaseEmbedding):
            raise EmbeddingError(
                f"Provider '{provider}' returned {type(embedding).__name__}, "
                "which is not a BaseEmbedding"
            )
        return embedding
