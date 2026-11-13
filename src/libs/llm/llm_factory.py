"""Provider selection for the LLM layer.

The provider name comes from `config/settings.yaml`; the factory turns it into
the matching `BaseLLM` implementation, so switching backends is a config edit
rather than a code change.
"""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.llm.base_llm import BaseLLM, LLMError

ProviderBuilder = Callable[[Settings], BaseLLM]


class LLMFactory:
    """Registry mapping a configured provider name to its implementation."""

    _builders: dict[str, ProviderBuilder] = {}

    @classmethod
    def register(cls, provider: str, builder: ProviderBuilder) -> None:
        """Register `builder` as the implementation of `provider`."""
        name = provider.strip().lower()
        if not name:
            raise ValueError("Cannot register an LLM provider with an empty name")
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
    def create(cls, settings: Settings) -> BaseLLM:
        """Build the LLM provider named by `settings.llm.provider`."""
        provider = (settings.llm.provider or "").strip().lower()
        if not provider:
            raise LLMError(
                "llm.provider is not configured - set it in config/settings.yaml"
            )

        builder = cls._builders.get(provider)
        if builder is None:
            available = ", ".join(cls.registered_providers()) or "(none)"
            raise LLMError(
                f"Unknown LLM provider '{provider}'. Registered providers: {available}"
            )

        llm = builder(settings)
        if not isinstance(llm, BaseLLM):
            raise LLMError(
                f"Provider '{provider}' returned {type(llm).__name__}, "
                "which is not a BaseLLM"
            )
        return llm
