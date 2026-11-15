"""OpenAI-compatible embedding client.

OpenAI and Azure OpenAI share the embeddings wire format and differ only in
the URL they serve it from and the header that carries the credential, so the
Azure implementation reuses everything here and overrides two methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.provider_http import HTTPEmbeddingProvider

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class OpenAIEmbeddingError(EmbeddingError):
    """Raised when an embedding request fails."""


class OpenAIEmbedding(HTTPEmbeddingProvider):
    """Embeddings against an OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    provider_name = "openai"

    def _base_url(self) -> str:
        base = (self.config.base_url or "").strip()
        return base.rstrip("/") if base else self.DEFAULT_BASE_URL

    def _embeddings_url(self) -> str:
        return f"{self._base_url()}/embeddings"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = (self.config.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _request_body(self, texts: Sequence[str]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.config.model or "", "input": list(texts)}
        if (self.config.model or "").startswith("text-embedding-3"):
            body["dimensions"] = self.config.dimensions
        return body

    def embed(
        self, texts: list[str], trace: TraceContext | None = None
    ) -> list[list[float]]:
        """Return one vector per input text, in the same order as `texts`."""
        checked = self._checked_texts(texts)

        response = self._post_json(
            self._embeddings_url(),
            self._request_body(checked),
            self._headers(),
            OpenAIEmbeddingError,
        )
        return self._parse_response(response.json(), len(checked))

    def _parse_response(self, payload: Any, expected: int) -> list[list[float]]:
        if not isinstance(payload, dict):
            raise OpenAIEmbeddingError(
                f"[{self.provider_name}] unexpected response type "
                f"{type(payload).__name__}"
            )

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise OpenAIEmbeddingError(
                f"[{self.provider_name}] response carries no embeddings"
            )

        ordered = sorted(data, key=lambda item: _index_of(item))
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not vector:
                raise OpenAIEmbeddingError(
                    f"[{self.provider_name}] response carries an invalid embedding"
                )
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in vector
            ):
                raise OpenAIEmbeddingError(
                    f"[{self.provider_name}] response carries a non numeric embedding"
                )
            vectors.append([float(value) for value in vector])

        if len(vectors) != expected:
            raise OpenAIEmbeddingError(
                f"[{self.provider_name}] response carries {len(vectors)} embeddings "
                f"for {expected} inputs"
            )
        return vectors


def _index_of(item: Any) -> int:
    if isinstance(item, dict):
        index = item.get("index")
        if isinstance(index, int) and not isinstance(index, bool):
            return index
    return 0
