"""Unit tests for the LLM abstraction and the provider factory."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from core.settings import LLMSettings, Settings
from libs.llm.base_llm import BaseLLM, ChatMessage, LLMError
from libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    """Stands in for a real provider so the factory is exercised offline."""

    def __init__(self, config: LLMSettings) -> None:
        super().__init__(config)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return "fake reply"


class OtherFakeLLM(FakeLLM):
    """Second fake, used to prove a re-registration replaces the builder."""


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider=provider, model="fake-model", api_key="fake-key")
    )


@pytest.fixture
def register_provider():
    """Register providers and remove them again once the test finishes."""
    registered: list[str] = []

    def _register(name: str, builder) -> str:
        LLMFactory.register(name, builder)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        LLMFactory.unregister(name)


class TestBaseLLM:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseLLM(LLMSettings())

    def test_subclass_must_implement_chat(self):
        class IncompleteLLM(BaseLLM):
            pass

        with pytest.raises(TypeError):
            IncompleteLLM(LLMSettings())

    def test_config_is_available_to_the_implementation(self):
        config = LLMSettings(provider="fake", model="fake-model")
        llm = FakeLLM(config)

        assert llm.config is config
        assert llm.config.model == "fake-model"

    def test_chat_returns_the_provider_answer(self):
        llm = FakeLLM(LLMSettings())
        messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

        assert llm.chat(messages) == "fake reply"
        assert llm.calls == [messages]


class TestLLMFactory:
    def test_create_returns_the_registered_provider(self, register_provider):
        register_provider("fake", lambda settings: FakeLLM(settings.llm))

        llm = LLMFactory.create(make_settings("fake"))

        assert isinstance(llm, FakeLLM)

    def test_create_passes_the_llm_configuration_to_the_provider(
        self, register_provider
    ):
        register_provider("fake", lambda settings: FakeLLM(settings.llm))

        llm = LLMFactory.create(make_settings("fake"))

        assert llm.config.model == "fake-model"
        assert llm.config.api_key == "fake-key"

    def test_provider_name_is_matched_case_insensitively(self, register_provider):
        register_provider("fake", lambda settings: FakeLLM(settings.llm))

        llm = LLMFactory.create(make_settings("FAKE"))

        assert isinstance(llm, FakeLLM)

    def test_unknown_provider_raises_a_readable_error(self):
        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create(make_settings("does-not-exist"))

        message = str(exc_info.value)
        assert "does-not-exist" in message
        assert "Registered providers" in message

    def test_missing_provider_raises_a_readable_error(self):
        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create(make_settings("   "))

        assert "llm.provider" in str(exc_info.value)

    def test_builder_returning_a_non_llm_is_rejected(self, register_provider):
        register_provider("broken", lambda settings: "not an llm")

        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create(make_settings("broken"))

        assert "BaseLLM" in str(exc_info.value)

    def test_registering_a_provider_twice_replaces_the_builder(
        self, register_provider
    ):
        register_provider("fake", lambda settings: FakeLLM(settings.llm))
        register_provider("fake", lambda settings: OtherFakeLLM(settings.llm))

        llm = LLMFactory.create(make_settings("fake"))

        assert isinstance(llm, OtherFakeLLM)

    def test_registered_providers_lists_registered_names(self, register_provider):
        register_provider("fake", lambda settings: FakeLLM(settings.llm))

        assert "fake" in LLMFactory.registered_providers()

    def test_unregister_removes_the_provider(self, register_provider):
        register_provider("fake", lambda settings: FakeLLM(settings.llm))

        LLMFactory.unregister("fake")

        assert "fake" not in LLMFactory.registered_providers()

    def test_registering_an_empty_provider_name_is_rejected(self):
        with pytest.raises(ValueError):
            LLMFactory.register("", lambda settings: FakeLLM(settings.llm))
