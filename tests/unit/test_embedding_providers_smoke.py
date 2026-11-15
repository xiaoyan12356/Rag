"""Smoke tests for the OpenAI and Azure embedding providers.

Every request is served by `httpx.MockTransport`, so the tests exercise the
real request shaping and the error handling without touching the network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from core.settings import EmbeddingSettings, Settings
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding, OpenAIEmbeddingError

API_KEY = "sk-test-key-do-not-log"


def make_embedding_settings(**overrides) -> EmbeddingSettings:
    values = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "api_key": API_KEY,
    }
    values.update(overrides)
    return EmbeddingSettings(**values)


def make_settings(**overrides) -> Settings:
    return Settings(embedding=make_embedding_settings(**overrides))


def make_azure_settings(**overrides) -> Settings:
    values = {
        "provider": "azure",
        "model": "text-embedding-ada-002",
        "dimensions": 1536,
        "deployment_name": "ada-deployment",
        "azure_endpoint": "https://my-resource.openai.azure.com",
        "api_version": "2024-02-01",
        "api_key": API_KEY,
    }
    values.update(overrides)
    return Settings(embedding=make_embedding_settings(**values))


def embeddings_payload(*vectors: list[float]) -> dict:
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


def recording_transport(payload=None, status_code: int = 200, error=None):
    """Return a transport that answers with `payload` and records the request."""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if error is not None:
            raise error
        return httpx.Response(
            status_code,
            json=payload if payload is not None else embeddings_payload([0.1, 0.2]),
        )

    return httpx.MockTransport(handler), recorded


class TestOpenAIEmbedding:
    def test_embed_returns_one_vector_per_text(self):
        transport, _ = recording_transport(embeddings_payload([0.1, 0.2], [0.3, 0.4]))

        vectors = OpenAIEmbedding(
            make_settings().embedding, transport=transport
        ).embed(["first", "second"])

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]

    def test_the_request_targets_the_embeddings_endpoint(self):
        transport, recorded = recording_transport()

        OpenAIEmbedding(make_settings().embedding, transport=transport).embed(["hi"])

        url = recorded[0].url
        assert url.host == "api.openai.com"
        assert url.path == "/v1/embeddings"

    def test_the_request_carries_the_api_key_and_the_model(self):
        transport, recorded = recording_transport()

        OpenAIEmbedding(make_settings().embedding, transport=transport).embed(["hi"])

        assert recorded[0].headers["Authorization"] == f"Bearer {API_KEY}"
        body = json.loads(recorded[0].content)
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == ["hi"]

    def test_a_text_embedding_3_model_receives_the_requested_dimensions(self):
        transport, recorded = recording_transport()

        OpenAIEmbedding(
            make_settings(dimensions=512).embedding, transport=transport
        ).embed(["hi"])

        assert json.loads(recorded[0].content)["dimensions"] == 512

    def test_an_older_model_receives_no_dimensions(self):
        transport, recorded = recording_transport()

        OpenAIEmbedding(
            make_settings(model="text-embedding-ada-002").embedding, transport=transport
        ).embed(["hi"])

        assert "dimensions" not in json.loads(recorded[0].content)

    def test_the_vectors_keep_the_order_of_the_inputs(self):
        transport, _ = recording_transport(
            embeddings_payload([0.3, 0.4], [0.1, 0.2]),
        )

        vectors = OpenAIEmbedding(
            make_settings().embedding, transport=transport
        ).embed(["first", "second"])

        assert vectors == [[0.3, 0.4], [0.1, 0.2]]

    def test_an_empty_batch_is_rejected(self):
        transport, _ = recording_transport()

        with pytest.raises(EmbeddingError, match="empty"):
            OpenAIEmbedding(make_settings().embedding, transport=transport).embed([])

    def test_an_empty_text_is_rejected(self):
        transport, _ = recording_transport()

        with pytest.raises(EmbeddingError, match="is empty"):
            OpenAIEmbedding(make_settings().embedding, transport=transport).embed(["   "])

    def test_an_overlong_text_is_rejected(self):
        transport, _ = recording_transport()
        long_text = "x" * 32001

        with pytest.raises(EmbeddingError, match="exceeds the limit"):
            OpenAIEmbedding(make_settings().embedding, transport=transport).embed([long_text])

    def test_a_non_string_text_is_rejected(self):
        transport, _ = recording_transport()

        with pytest.raises(EmbeddingError, match="not a string"):
            OpenAIEmbedding(make_settings().embedding, transport=transport).embed([123])


class TestAzureEmbedding:
    def test_the_request_targets_the_deployment_url(self):
        transport, recorded = recording_transport()

        AzureEmbedding(make_azure_settings().embedding, transport=transport).embed(["hi"])

        url = recorded[0].url
        assert url.path == "/openai/deployments/ada-deployment/embeddings"
        assert url.params["api-version"] == "2024-02-01"

    def test_the_request_authenticates_with_the_api_key_header(self):
        transport, recorded = recording_transport()

        AzureEmbedding(make_azure_settings().embedding, transport=transport).embed(["hi"])

        assert recorded[0].headers["api-key"] == API_KEY
        assert "Authorization" not in recorded[0].headers

    def test_the_request_body_omits_the_model(self):
        transport, recorded = recording_transport()

        AzureEmbedding(make_azure_settings().embedding, transport=transport).embed(["hi"])

        body = json.loads(recorded[0].content)
        assert body == {"input": ["hi"]}

    def test_a_missing_azure_endpoint_is_rejected_at_creation(self):
        with pytest.raises(OpenAIEmbeddingError, match="azure_endpoint"):
            AzureEmbedding(make_settings(provider="azure").embedding)

    def test_a_missing_deployment_name_is_rejected_at_creation(self):
        with pytest.raises(OpenAIEmbeddingError, match="deployment_name"):
            AzureEmbedding(make_azure_settings(model="", deployment_name="").embedding)


class TestFactoryRouting:
    @pytest.mark.parametrize(
        ("provider", "expected"),
        [("openai", OpenAIEmbedding), ("azure", AzureEmbedding)],
    )
    def test_the_factory_routes_to_the_configured_provider(self, provider, expected):
        settings = (
            make_azure_settings() if provider == "azure" else make_settings(provider=provider)
        )

        embedding = EmbeddingFactory.create(settings)

        assert isinstance(embedding, expected)

    def test_the_provider_name_is_matched_case_insensitively(self):
        embedding = EmbeddingFactory.create(make_settings(provider="OpenAI"))

        assert isinstance(embedding, OpenAIEmbedding)


class TestErrorHandling:
    def test_an_http_error_reports_the_status_and_the_provider_detail(self):
        transport, _ = recording_transport(
            payload={"error": {"code": "429", "message": "rate limit reached"}},
            status_code=429,
        )

        embedding = OpenAIEmbedding(make_settings().embedding, transport=transport)

        with pytest.raises(OpenAIEmbeddingError) as exc_info:
            embedding.embed(["hi"])

        message = str(exc_info.value)
        assert "429" in message
        assert "rate limit reached" in message

    def test_a_timeout_is_reported_without_leaking_the_api_key(self):
        transport, _ = recording_transport(error=httpx.ReadTimeout("too slow"))

        embedding = OpenAIEmbedding(make_settings().embedding, transport=transport)

        with pytest.raises(OpenAIEmbeddingError) as exc_info:
            embedding.embed(["hi"])

        message = str(exc_info.value)
        assert "timed out" in message
        assert API_KEY not in message

    def test_a_connection_failure_is_reported_without_leaking_the_api_key(self):
        transport, _ = recording_transport(error=httpx.ConnectError("refused"))

        embedding = OpenAIEmbedding(make_settings().embedding, transport=transport)

        with pytest.raises(OpenAIEmbeddingError) as exc_info:
            embedding.embed(["hi"])

        message = str(exc_info.value)
        assert "could not reach the endpoint" in message
        assert API_KEY not in message

    def test_a_response_without_embeddings_is_rejected(self):
        transport, _ = recording_transport(payload={"data": []})

        embedding = OpenAIEmbedding(make_settings().embedding, transport=transport)

        with pytest.raises(OpenAIEmbeddingError, match="no embeddings"):
            embedding.embed(["hi"])

    def test_a_response_with_a_wrong_vector_count_is_rejected(self):
        transport, _ = recording_transport(embeddings_payload([0.1, 0.2]))

        embedding = OpenAIEmbedding(make_settings().embedding, transport=transport)

        with pytest.raises(OpenAIEmbeddingError, match="for 2 inputs"):
            embedding.embed(["first", "second"])
