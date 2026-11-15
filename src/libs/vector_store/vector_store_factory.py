"""Provider selection for the vector store layer.

The provider name comes from `config/settings.yaml`; the factory turns it into
the matching `BaseVectorStore` implementation, so switching databases is a
config edit rather than a code change.
"""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError

ProviderBuilder = Callable[[Settings], BaseVectorStore]


class VectorStoreFactory:
    """Registry mapping a configured provider name to its implementation."""

    _builders: dict[str, ProviderBuilder] = {}

    @classmethod
    def register(cls, provider: str, builder: ProviderBuilder) -> None:
        """Register `builder` as the implementation of `provider`."""
        name = provider.strip().lower()
        if not name:
            raise ValueError("Cannot register a vector store provider with an empty name")
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
    def create(cls, settings: Settings) -> BaseVectorStore:
        """Build the vector store named by `settings.vector_store.provider`."""
        provider = (settings.vector_store.provider or "").strip().lower()
        if not provider:
            raise VectorStoreError(
                "vector_store.provider is not configured - set it in config/settings.yaml"
            )

        builder = cls._builders.get(provider)
        if builder is None:
            available = ", ".join(cls.registered_providers()) or "(none)"
            raise VectorStoreError(
                f"Unknown vector store provider '{provider}'. "
                f"Registered providers: {available}"
            )

        store = builder(settings)
        if not isinstance(store, BaseVectorStore):
            raise VectorStoreError(
                f"Provider '{provider}' returned {type(store).__name__}, "
                "which is not a BaseVectorStore"
            )
        return store
