"""Smoke tests for the OpenAI-compatible LLM providers.

Every request is served by `httpx.MockTransport`, so the tests exercise the
real request shaping and error handling without touching the network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from core.settings import LLMSettings, Settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAILLM, OpenAILLMError

API_KEY = "sk-test-key-do-not-log"


def make_llm_settings(**overrides) -> LLMSettings:
    values = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": API_KEY,
    }
    values.update(overrides)
    return LLMSettings(**values)


def make_settings(**overrides) -> Settings:
    return Settings(llm=make_llm_settings(**overrides))


def make_azure_settings(**overrides) -> Settings:
    values = {
        "provider": "azure",
        "model": "gpt-4o",
        "deployment_name": "gpt-4o-deployment",
        "azure_endpoint": "https://my-resource.openai.azure.com",
        "api_version": "2024-02-01",
    }
    values.update(overrides)
    return Settings(llm=make_llm_settings(**values))


def chat_payload(content: str = "hello", model: str = "gpt-4o-mini") -> dict:
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def recording_transport(payload=None, status_code: int = 200, error=None):
    """Return a transport that answers with `payload` and records the request."""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if error is not None:
            raise error
        return httpx.Response(status_code, json=payload if payload is not None else chat_payload())

    return httpx.MockTransport(handler), recorded


def make_llm(settings: Settings, **kwargs) -> OpenAILLM:
    transport, _ = recording_transport(**kwargs)
    return OpenAILLM(settings.llm, transport=transport)


class TestOpenAILLM:
    def test_chat_returns_the_reply_text(self):
        llm = make_llm(make_settings())

        assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"

    def test_complete_returns_the_usage_metadata(self):
        llm = make_llm(make_settings())

        response = llm.complete([{"role": "user", "content": "hi"}])

        assert response.content == "hello"
        assert response.model == "gpt-4o-mini"
        assert response.usage["total_tokens"] == 7

    def test_the_request_targets_the_chat_completions_endpoint(self):
        transport, recorded = recording_transport()

        OpenAILLM(make_settings().llm, transport=transport).chat(
            [{"role": "user", "content": "hi"}]
        )

        url = recorded[0].url
        assert url.host == "api.openai.com"
        assert url.path == "/v1/chat/completions"

    def test_the_request_carries_the_api_key_and_the_conversation(self):
        transport, recorded = recording_transport()

        OpenAILLM(make_settings().llm, transport=transport).chat(
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        )

        request = recorded[0]
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_a_custom_base_url_is_honoured(self):
        transport, recorded = recording_transport()

        OpenAILLM(
            make_settings(base_url="https://proxy.internal/v1").llm, transport=transport
        ).chat([{"role": "user", "content": "hi"}])

        assert str(recorded[0].url) == "https://proxy.internal/v1/chat/completions"


class TestAzureLLM:
    def test_the_request_targets_the_deployment_url(self):
        transport, recorded = recording_transport()

        AzureLLM(make_azure_settings().llm, transport=transport).chat(
            [{"role": "user", "content": "hi"}]
        )

        url = recorded[0].url
        assert url.path == "/openai/deployments/gpt-4o-deployment/chat/completions"
        assert url.params["api-version"] == "2024-02-01"

    def test_the_request_authenticates_with_the_api_key_header(self):
        transport, recorded = recording_transport()

        AzureLLM(make_azure_settings().llm, transport=transport).chat(
            [{"role": "user", "content": "hi"}]
        )

        assert recorded[0].headers["api-key"] == API_KEY
        assert "Authorization" not in recorded[0].headers

    def test_a_missing_azure_endpoint_is_rejected_at_creation(self):
        with pytest.raises(OpenAILLMError, match="azure_endpoint"):
            AzureLLM(make_settings(provider="azure").llm)

    def test_a_missing_deployment_name_is_rejected_at_creation(self):
        with pytest.raises(OpenAILLMError, match="deployment_name"):
            AzureLLM(make_azure_settings(model="", deployment_name="").llm)


class TestDeepSeekLLM:
    def test_the_request_targets_the_deepseek_endpoint(self):
        transport, recorded = recording_transport()

        DeepSeekLLM(make_settings(provider="deepseek").llm, transport=transport).chat(
            [{"role": "user", "content": "hi"}]
        )

        assert str(recorded[0].url) == "https://api.deepseek.com/v1/chat/completions"


class TestFactoryRouting:
    @pytest.mark.parametrize(
        ("provider", "expected"),
        [
            ("openai", OpenAILLM),
            ("azure", AzureLLM),
            ("deepseek", DeepSeekLLM),
        ],
    )
    def test_the_factory_routes_to_the_configured_provider(self, provider, expected):
        settings = (
            make_azure_settings() if provider == "azure" else make_settings(provider=provider)
        )

        llm = LLMFactory.create(settings)

        assert isinstance(llm, expected)

    def test_the_provider_name_is_matched_case_insensitively(self):
        llm = LLMFactory.create(make_settings(provider="OpenAI"))

        assert isinstance(llm, OpenAILLM)


class TestErrorHandling:
    def test_invalid_messages_are_rejected_with_a_readable_error(self):
        llm = make_llm(make_settings())

        with pytest.raises(OpenAILLMError) as exc_info:
            llm.chat([])

        message = str(exc_info.value)
        assert "openai" in message
        assert "empty" in message

    def test_an_unknown_role_is_rejected(self):
        llm = make_llm(make_settings())

        with pytest.raises(OpenAILLMError, match="role"):
            llm.chat([{"role": "wizard", "content": "hi"}])

    def test_a_non_mapping_message_is_rejected(self):
        llm = make_llm(make_settings())

        with pytest.raises(OpenAILLMError, match="mapping"):
            llm.chat(["hi"])

    def test_an_http_error_reports_the_status_and_the_provider_detail(self):
        transport, _ = recording_transport(
            payload={"error": {"code": "429", "message": "rate limit reached"}},
            status_code=429,
        )

        llm = OpenAILLM(make_settings().llm, transport=transport)

        with pytest.raises(OpenAILLMError) as exc_info:
            llm.chat([{"role": "user", "content": "hi"}])

        message = str(exc_info.value)
        assert "429" in message
        assert "rate limit reached" in message

    def test_a_timeout_is_reported_without_leaking_the_api_key(self):
        transport, _ = recording_transport(error=httpx.ReadTimeout("too slow"))

        llm = OpenAILLM(make_settings().llm, transport=transport)

        with pytest.raises(OpenAILLMError) as exc_info:
            llm.chat([{"role": "user", "content": "hi"}])

        message = str(exc_info.value)
        assert "timed out" in message
        assert API_KEY not in message

    def test_a_connection_failure_is_reported_without_leaking_the_api_key(self):
        transport, _ = recording_transport(error=httpx.ConnectError("refused"))

        llm = OpenAILLM(make_settings().llm, transport=transport)

        with pytest.raises(OpenAILLMError) as exc_info:
            llm.chat([{"role": "user", "content": "hi"}])

        message = str(exc_info.value)
        assert "could not reach" in message
        assert API_KEY not in message

    def test_a_response_without_choices_is_rejected(self):
        transport, _ = recording_transport(payload={"id": "chatcmpl-1"})

        llm = OpenAILLM(make_settings().llm, transport=transport)

        with pytest.raises(OpenAILLMError, match="no choices"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_a_response_without_content_is_rejected(self):
        transport, _ = recording_transport(payload={"choices": [{"message": {}}]})

        llm = OpenAILLM(make_settings().llm, transport=transport)

        with pytest.raises(OpenAILLMError, match="no message content"):
            llm.chat([{"role": "user", "content": "hi"}])
