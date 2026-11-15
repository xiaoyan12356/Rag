"""Ollama chat completions for locally hosted models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from libs.llm.base_llm import ChatMessage, ChatResponse, LLMError
from libs.llm.provider_http import HTTPChatProvider

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class OllamaLLMError(LLMError):
    """Raised when an Ollama request fails."""


class OllamaLLM(HTTPChatProvider):
    """Chat completions against a local Ollama server."""

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0
    provider_name = "ollama"

    def _base_url(self) -> str:
        base = (self.config.base_url or "").strip()
        return base.rstrip("/") if base else self.DEFAULT_BASE_URL

    def _chat_url(self) -> str:
        return f"{self._base_url()}/api/chat"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _request_body(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {
            "model": self.config.model or "",
            "messages": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
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
            raise OllamaLLMError(
                f"[{self.provider_name}] invalid chat messages: {exc}"
            ) from exc

        response = self._post_json(
            self._chat_url(),
            self._request_body(messages),
            self._headers(),
            OllamaLLMError,
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
            raise OllamaLLMError(
                f"[{self.provider_name}] unexpected response type "
                f"{type(payload).__name__}"
            )

        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaLLMError(
                f"[{self.provider_name}] response carries no message content"
            )

        usage: dict[str, int] = {}
        for source, target in (
            ("prompt_eval_count", "prompt_tokens"),
            ("eval_count", "completion_tokens"),
        ):
            value = payload.get(source)
            if isinstance(value, int) and not isinstance(value, bool):
                usage[target] = value

        return ChatResponse(
            content=content,
            model=str(payload.get("model") or self.config.model or ""),
            usage=usage,
        )
