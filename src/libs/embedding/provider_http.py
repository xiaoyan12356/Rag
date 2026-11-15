"""HTTP plumbing shared by the embedding providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.http_client import JSONHttpClient


class HTTPEmbeddingProvider(BaseEmbedding):
    """Base class for embedding providers that reach their endpoint over HTTP."""

    provider_name = "http"
    DEFAULT_TIMEOUT = 60.0
    MAX_BATCH_SIZE = 2048
    MAX_INPUT_CHARACTERS = 32000

    def __init__(
        self,
        config: EmbeddingSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(config)
        self.timeout = float(timeout) if timeout is not None else self.DEFAULT_TIMEOUT
        self._http = JSONHttpClient(
            self.provider_name, timeout=self.timeout, transport=transport
        )

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._http.close()

    def __enter__(self) -> HTTPEmbeddingProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        error_class: type[Exception],
    ) -> httpx.Response:
        return self._http.post_json(url, payload, headers, error_class)

    def _checked_texts(self, texts: Sequence[str]) -> list[str]:
        """Check the batch against the provider limits and return it as a list."""
        try:
            self.validate_texts(texts)
        except ValueError as exc:
            raise EmbeddingError(f"[{self.provider_name}] invalid input: {exc}") from exc

        if len(texts) > self.MAX_BATCH_SIZE:
            raise EmbeddingError(
                f"[{self.provider_name}] batch of {len(texts)} texts exceeds the limit "
                f"of {self.MAX_BATCH_SIZE}"
            )

        for index, text in enumerate(texts):
            if not text.strip():
                raise EmbeddingError(
                    f"[{self.provider_name}] text at index {index} is empty"
                )
            if len(text) > self.MAX_INPUT_CHARACTERS:
                raise EmbeddingError(
                    f"[{self.provider_name}] text at index {index} is {len(text)} characters, "
                    f"which exceeds the limit of {self.MAX_INPUT_CHARACTERS}"
                )
        return list(texts)
