"""Shared JSON-over-HTTPS client for the remote provider implementations.

The LLM and embedding providers all reach a remote endpoint over the same
JSON-over-HTTPS convention, so the connection handling and the translation of
transport failures into readable errors live here once. The messages name the
provider and the failure type, never the credential or the endpoint.
"""

from __future__ import annotations

from typing import Any

import httpx


class JSONHttpClient:
    """A JSON-over-HTTPS client that reports failures in provider terms."""

    def __init__(
        self,
        provider_name: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        error_class: type[Exception],
    ) -> httpx.Response:
        """POST `payload` and translate a transport failure into `error_class`."""
        try:
            response = self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise error_class(
                f"[{self.provider_name}] request timed out after {self.timeout}s "
                f"({type(exc).__name__})"
            ) from exc
        except httpx.ConnectError as exc:
            raise error_class(
                f"[{self.provider_name}] could not reach the endpoint ({type(exc).__name__})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise error_class(
                f"[{self.provider_name}] HTTP {exc.response.status_code} from the provider: "
                f"{describe_error(exc.response)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise error_class(
                f"[{self.provider_name}] request failed ({type(exc).__name__})"
            ) from exc
        return response

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    def __enter__(self) -> JSONHttpClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def describe_error(response: httpx.Response) -> str:
    """Summarise a provider error body, keeping the provider's error code."""
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:200] if text else "empty response body"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            parts = [
                str(part) for part in (error.get("code"), error.get("message")) if part
            ]
            if parts:
                return ": ".join(parts)
        if isinstance(error, str) and error:
            return error
    return str(payload)[:200]
