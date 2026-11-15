"""Ollama embedding client for locally hosted models.

Ollama's `/api/embeddings` endpoint serves one text per request, so a batch
is sent as one request per text. The width of the returned vector is decided
by the model (`nomic-embed-text` yields 768 dimensions), not by the
configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.provider_http import HTTPEmbeddingProvider

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class OllamaEmbeddingError(EmbeddingError):
    """Raised when an Ollama embedding request fails."""


class OllamaEmbedding(HTTPEmbeddingProvider):
    """Embeddings against a local Ollama server."""

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0
    provider_name = "ollama"

    def _embeddings_url(self) -> str:
        base = (self.config.base_url or "").strip()
        base = base.rstrip("/") if base else self.DEFAULT_BASE_URL
        return f"{base}/api/embeddings"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _request_body(self, text: str) -> dict[str, Any]:
        return {"model": self.config.model or "", "prompt": text}

    def embed(
        self, texts: list[str], trace: TraceContext | None = None
    ) -> list[list[float]]:
        """Return one vector per input text, in the same order as `texts`."""
        checked = self._checked_texts(texts)

        vectors: list[list[float]] = []
        for text in checked:
            response = self._post_json(
                self._embeddings_url(),
                self._request_body(text),
                self._headers(),
                OllamaEmbeddingError,
            )
            vectors.append(self._parse_vector(response.json()))
        return vectors

    def _parse_vector(self, payload: Any) -> list[float]:
        if not isinstance(payload, dict):
            raise OllamaEmbeddingError(
                f"[{self.provider_name}] unexpected response type "
                f"{type(payload).__name__}"
            )

        vector = payload.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise OllamaEmbeddingError(
                f"[{self.provider_name}] response carries no embedding"
            )
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in vector
        ):
            raise OllamaEmbeddingError(
                f"[{self.provider_name}] response carries a non numeric embedding"
            )
        return [float(value) for value in vector]
