"""Tests for the Ollama embedding provider.

Every request is served by `httpx.MockTransport`, so the tests exercise the
real request shaping and the error handling without touching the network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from core.settings import EmbeddingSettings, Settings
from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError

API_KEY = "sk-test-key-do-not-log"
NOMIC_DIMENSION = 768


def make_settings(**overrides) -> Settings:
    values = {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "api_key": API_KEY,
    }
    values.update(overrides)
    return Settings(embedding=EmbeddingSettings(**values))


def embedding_payload(dimension: int = NOMIC_DIMENSION) -> dict:
    return {"embedding": [0.01] * dimension}


def recording_transport(payload=None, status_code: int = 200, error=None):
    """Return a transport that answers with `payload` and records the request."""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if error is not None:
            raise error
        return httpx.Response(
            status_code,
            json=payload if payload is not None else embedding_payload(),
        )

    return httpx.MockTransport(handler), recorded


class TestOllamaEmbedding:
    def test_embed_returns_one_vector_per_text(self):
        transport, _ = recording_transport()

        vectors = OllamaEmbedding(
            make_settings().embedding, transport=transport
        ).embed(["first", "second"])

        assert len(vectors) == 2
        assert len(vectors[0]) == NOMIC_DIMENSION

    def test_the_vector_width_follows_the_model(self):
        transport, _ = recording_transport(embedding_payload(1024))

        vectors = OllamaEmbedding(
            make_settings(model="mxbai-embed-large").embedding, transport=transport
        ).embed(["hi"])

        assert len(vectors[0]) == 1024

    def test_the_request_targets_the_ollama_embeddings_endpoint(self):
        transport, recorded = recording_transport()

        OllamaEmbedding(make_settings().embedding, transport=transport).embed(["hi"])

        url = recorded[0].url
        assert url.host == "localhost"
        assert url.port == 11434
        assert url.path == "/api/embeddings"

    def test_the_request_carries_the_model_and_the_text(self):
        transport, recorded = recording_transport()

        OllamaEmbedding(make_settings().embedding, transport=transport).embed(["hi"])

        assert json.loads(recorded[0].content) == {
            "model": "nomic-embed-text",
            "prompt": "hi",
        }

    def test_a_batch_is_sent_as_one_request_per_text(self):
        transport, recorded = recording_transport()

        OllamaEmbedding(make_settings().embedding, transport=transport).embed(
            ["first", "second", "third"]
        )

        assert len(recorded) == 3
        assert [json.loads(request.content)["prompt"] for request in recorded] == [
            "first",
            "second",
            "third",
        ]

    def test_a_configured_base_url_is_honoured(self):
        transport, recorded = recording_transport()

        OllamaEmbedding(
            make_settings(base_url="http://ollama.internal:11434/").embedding,
            transport=transport,
        ).embed(["hi"])

        assert str(recorded[0].url) == "http://ollama.internal:11434/api/embeddings"


class TestFactoryRouting:
    def test_the_factory_creates_the_ollama_provider(self):
        embedding = EmbeddingFactory.create(make_settings())

        assert isinstance(embedding, OllamaEmbedding)

    def test_the_provider_name_is_matched_case_insensitively(self):
        embedding = EmbeddingFactory.create(make_settings(provider="Ollama"))

        assert isinstance(embedding, OllamaEmbedding)


class TestErrorHandling:
    def test_an_empty_batch_is_rejected(self):
        transport, _ = recording_transport()

        with pytest.raises(EmbeddingError, match="empty"):
            OllamaEmbedding(make_settings().embedding, transport=transport).embed([])

    def test_an_overlong_text_is_rejected(self):
        transport, _ = recording_transport()
        long_text = "x" * 32001

        with pytest.raises(EmbeddingError, match="exceeds the limit"):
            OllamaEmbedding(make_settings().embedding, transport=transport).embed([long_text])

    def test_a_connection_failure_is_reported_without_leaking_the_configuration(self):
        transport, _ = recording_transport(error=httpx.ConnectError("refused"))

        embedding = OllamaEmbedding(
            make_settings(base_url="http://ollama.internal:11434").embedding,
            transport=transport,
        )

        with pytest.raises(OllamaEmbeddingError) as exc_info:
            embedding.embed(["hi"])

        message = str(exc_info.value)
        assert "could not reach the endpoint" in message
        assert API_KEY not in message
        assert "ollama.internal" not in message

    def test_a_timeout_is_reported_without_leaking_the_api_key(self):
        transport, _ = recording_transport(error=httpx.ReadTimeout("too slow"))

        embedding = OllamaEmbedding(make_settings().embedding, transport=transport)

        with pytest.raises(OllamaEmbeddingError) as exc_info:
            embedding.embed(["hi"])

        message = str(exc_info.value)
        assert "timed out" in message
        assert API_KEY not in message

    def test_an_unknown_model_is_reported_with_the_provider_detail(self):
        transport, _ = recording_transport(
            payload={"error": "model 'missing' not found"}, status_code=404
        )

        embedding = OllamaEmbedding(make_settings().embedding, transport=transport)

        with pytest.raises(OllamaEmbeddingError) as exc_info:
            embedding.embed(["hi"])

        message = str(exc_info.value)
        assert "404" in message
        assert "not found" in message

    def test_a_response_without_an_embedding_is_rejected(self):
        transport, _ = recording_transport(payload={"embedding": []})

        embedding = OllamaEmbedding(make_settings().embedding, transport=transport)

        with pytest.raises(OllamaEmbeddingError, match="no embedding"):
            embedding.embed(["hi"])
