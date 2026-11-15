"""Tests for the Ollama LLM provider.

Every request is served by `httpx.MockTransport`, so the tests exercise the
real request shaping and the error handling without touching the network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from core.settings import LLMSettings, Settings
from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM, OllamaLLMError

API_KEY = "sk-test-key-do-not-log"


def make_settings(**overrides) -> Settings:
    values = {
        "provider": "ollama",
        "model": "qwen2.5:7b",
        "api_key": API_KEY,
    }
    values.update(overrides)
    return Settings(llm=LLMSettings(**values))


def chat_payload(content: str = "hello") -> dict:
    return {
        "model": "qwen2.5:7b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 2,
    }


def recording_transport(payload=None, status_code: int = 200, error=None):
    """Return a transport that answers with `payload` and records the request."""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if error is not None:
            raise error
        return httpx.Response(
            status_code, json=payload if payload is not None else chat_payload()
        )

    return httpx.MockTransport(handler), recorded


def make_llm(settings: Settings, **kwargs) -> OllamaLLM:
    transport, _ = recording_transport(**kwargs)
    return OllamaLLM(settings.llm, transport=transport)


class TestOllamaLLM:
    def test_chat_returns_the_reply_text(self):
        llm = make_llm(make_settings())

        assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"

    def test_complete_returns_the_token_counts(self):
        llm = make_llm(make_settings())

        response = llm.complete([{"role": "user", "content": "hi"}])

        assert response.content == "hello"
        assert response.model == "qwen2.5:7b"
        assert response.usage == {"prompt_tokens": 5, "completion_tokens": 2}

    def test_the_request_targets_the_ollama_chat_endpoint(self):
        transport, recorded = recording_transport()

        OllamaLLM(make_settings().llm, transport=transport).chat(
            [{"role": "user", "content": "hi"}]
        )

        url = recorded[0].url
        assert url.host == "localhost"
        assert url.port == 11434
        assert url.path == "/api/chat"

    def test_the_request_carries_the_model_and_the_conversation(self):
        transport, recorded = recording_transport()

        OllamaLLM(make_settings().llm, transport=transport).chat(
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        )

        body = json.loads(recorded[0].content)
        assert body["model"] == "qwen2.5:7b"
        assert body["stream"] is False
        assert body["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_a_configured_base_url_is_honoured(self):
        transport, recorded = recording_transport()

        OllamaLLM(
            make_settings(base_url="http://ollama.internal:11434/").llm, transport=transport
        ).chat([{"role": "user", "content": "hi"}])

        assert str(recorded[0].url) == "http://ollama.internal:11434/api/chat"


class TestFactoryRouting:
    def test_the_factory_creates_the_ollama_provider(self):
        llm = LLMFactory.create(make_settings())

        assert isinstance(llm, OllamaLLM)

    def test_the_provider_name_is_matched_case_insensitively(self):
        llm = LLMFactory.create(make_settings(provider="Ollama"))

        assert isinstance(llm, OllamaLLM)


class TestErrorHandling:
    def test_invalid_messages_are_rejected_with_a_readable_error(self):
        llm = make_llm(make_settings())

        with pytest.raises(OllamaLLMError) as exc_info:
            llm.chat([])

        message = str(exc_info.value)
        assert "ollama" in message
        assert "empty" in message

    def test_a_connection_failure_is_reported_without_leaking_the_configuration(self):
        transport, _ = recording_transport(error=httpx.ConnectError("refused"))

        llm = OllamaLLM(
            make_settings(base_url="http://ollama.internal:11434").llm, transport=transport
        )

        with pytest.raises(OllamaLLMError) as exc_info:
            llm.chat([{"role": "user", "content": "hi"}])

        message = str(exc_info.value)
        assert "could not reach the endpoint" in message
        assert API_KEY not in message
        assert "ollama.internal" not in message

    def test_a_timeout_is_reported_without_leaking_the_configuration(self):
        transport, _ = recording_transport(error=httpx.ReadTimeout("too slow"))

        llm = OllamaLLM(make_settings().llm, transport=transport)

        with pytest.raises(OllamaLLMError) as exc_info:
            llm.chat([{"role": "user", "content": "hi"}])

        message = str(exc_info.value)
        assert "timed out" in message
        assert API_KEY not in message

    def test_an_unknown_model_is_reported_with_the_provider_detail(self):
        transport, _ = recording_transport(
            payload={"error": "model 'missing:1b' not found"}, status_code=404
        )

        llm = OllamaLLM(make_settings().llm, transport=transport)

        with pytest.raises(OllamaLLMError) as exc_info:
            llm.chat([{"role": "user", "content": "hi"}])

        message = str(exc_info.value)
        assert "404" in message
        assert "not found" in message

    def test_a_response_without_content_is_rejected(self):
        transport, _ = recording_transport(payload={"done": True})

        llm = OllamaLLM(make_settings().llm, transport=transport)

        with pytest.raises(OllamaLLMError, match="no message content"):
            llm.chat([{"role": "user", "content": "hi"}])
