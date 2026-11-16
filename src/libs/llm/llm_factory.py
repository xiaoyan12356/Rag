"""Provider selection for the LLM layer.

The provider name comes from `config/settings.yaml`; the factory turns it into
the matching `BaseLLM` implementation, so switching backends is a config edit
rather than a code change. The vision providers live in their own registry,
because a vision model is a different kind of object than a text model.
"""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.llm.base_llm import BaseLLM, LLMError
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMError

ProviderBuilder = Callable[[Settings], BaseLLM]
VisionProviderBuilder = Callable[[Settings], BaseVisionLLM]


class LLMFactory:
    """Registry mapping a configured provider name to its implementation."""

    _builders: dict[str, ProviderBuilder] = {}
    _vision_builders: dict[str, VisionProviderBuilder] = {}

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

    @classmethod
    def register_vision(cls, provider: str, builder: VisionProviderBuilder) -> None:
        """Register `builder` as the implementation of a vision provider."""
        name = provider.strip().lower()
        if not name:
            raise ValueError("Cannot register a vision LLM provider with an empty name")
        cls._vision_builders[name] = builder

    @classmethod
    def unregister_vision(cls, provider: str) -> None:
        """Remove a vision provider registration; unknown names are ignored."""
        cls._vision_builders.pop(provider.strip().lower(), None)

    @classmethod
    def registered_vision_providers(cls) -> list[str]:
        """Return the registered vision provider names, sorted."""
        return sorted(cls._vision_builders)

    @classmethod
    def create_vision_llm(cls, settings: Settings) -> BaseVisionLLM:
        """Build the vision provider named by `settings.vision_llm.provider`.

        Whether the vision model is used at all is the caller's decision, so
        the `vision_llm.enabled` switch is not applied here.
        """
        provider = (settings.vision_llm.provider or "").strip().lower()
        if not provider:
            raise VisionLLMError(
                "vision_llm.provider is not configured - set it in config/settings.yaml"
            )

        builder = cls._vision_builders.get(provider)
        if builder is None:
            available = ", ".join(cls.registered_vision_providers()) or "(none)"
            raise VisionLLMError(
                f"Unknown vision LLM provider '{provider}'. "
                f"Registered vision providers: {available}"
            )

        vision = builder(settings)
        if not isinstance(vision, BaseVisionLLM):
            raise VisionLLMError(
                f"Provider '{provider}' returned {type(vision).__name__}, "
                "which is not a BaseVisionLLM"
            )
        return vision


def _openai_builder(settings: Settings) -> BaseLLM:
    from libs.llm.openai_llm import OpenAILLM

    return OpenAILLM(settings.llm)


def _azure_builder(settings: Settings) -> BaseLLM:
    from libs.llm.azure_llm import AzureLLM

    return AzureLLM(settings.llm)


def _deepseek_builder(settings: Settings) -> BaseLLM:
    from libs.llm.deepseek_llm import DeepSeekLLM

    return DeepSeekLLM(settings.llm)


def _ollama_builder(settings: Settings) -> BaseLLM:
    from libs.llm.ollama_llm import OllamaLLM

    return OllamaLLM(settings.llm)


def _azure_vision_builder(settings: Settings) -> BaseVisionLLM:
    from libs.llm.azure_vision_llm import AzureVisionLLM

    return AzureVisionLLM(settings.vision_llm)


LLMFactory.register("openai", _openai_builder)
LLMFactory.register("azure", _azure_builder)
LLMFactory.register("deepseek", _deepseek_builder)
LLMFactory.register("ollama", _ollama_builder)

LLMFactory.register_vision("azure", _azure_vision_builder)
