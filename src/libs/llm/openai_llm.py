"""OpenAI-compatible chat completions.

OpenAI, Azure OpenAI and DeepSeek all speak the same chat completions wire
format and differ only in the URL they serve it from and the header that
carries the credential, so the request handling lives here once and the
variants override two methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from libs.llm.base_llm import ChatMessage, ChatResponse, LLMError
from libs.llm.provider_http import HTTPChatProvider

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class OpenAILLMError(LLMError):
    """Raised when an OpenAI-compatible request fails."""


class OpenAILLM(HTTPChatProvider):
    """Chat completions against an OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    provider_name = "openai"

    def _base_url(self) -> str:
        base = (self.config.base_url or "").strip()
        return base.rstrip("/") if base else self.DEFAULT_BASE_URL

    def _chat_completions_url(self) -> str:
        return f"{self._base_url()}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = (self.config.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _request_body(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {
            "model": self.config.model or "",
            "messages": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    def complete(
        self,
        messages: Sequence[ChatMessage],
        trace: TraceContext | None = None,
    ) -> ChatResponse:
        """Send the conversation and return the reply with its metadata."""
        try:
            self.validate_messages(messages)
        except ValueError as exc:
            raise OpenAILLMError(
                f"[{self.provider_name}] invalid chat messages: {exc}"
            ) from exc

        response = self._post_json(
            self._chat_completions_url(),
            self._request_body(messages),
            self._headers(),
            OpenAILLMError,
        )
        return self._parse_response(response.json())

    def chat(
        self,
        messages: Sequence[ChatMessage],
        trace: TraceContext | None = None,
    ) -> str:
        return self.complete(messages, trace).content

    def _parse_response(self, payload: Any) -> ChatResponse:
        if not isinstance(payload, dict):
            raise OpenAILLMError(
                f"[{self.provider_name}] unexpected response type "
                f"{type(payload).__name__}"
            )

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenAILLMError(f"[{self.provider_name}] response carries no choices")

        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OpenAILLMError(
                f"[{self.provider_name}] response carries no message content"
            )

        return ChatResponse(
            content=content,
            model=str(payload.get("model") or self.config.model or ""),
            usage=_usage_counts(payload.get("usage")),
        )


def _usage_counts(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    return {
        str(key): int(value)
        for key, value in usage.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
