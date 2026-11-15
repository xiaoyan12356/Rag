"""HTTP plumbing shared by the LLM providers."""

from __future__ import annotations

from typing import Any

import httpx

from core.settings import LLMSettings
from libs.http_client import JSONHttpClient
from libs.llm.base_llm import BaseLLM


class HTTPChatProvider(BaseLLM):
    """Base class for providers that reach their endpoint over HTTP."""

    provider_name = "http"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        config: LLMSettings,
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

    def __enter__(self) -> HTTPChatProvider:
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
