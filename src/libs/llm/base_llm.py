"""LLM abstraction.

Every provider hides a different authentication scheme and request format
behind the same `chat` call, so callers depend on `BaseLLM` and never on a
concrete client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict

from core.settings import LLMSettings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

Role = Literal["system", "user", "assistant"]

ROLES = ("system", "user", "assistant")


class ChatMessage(TypedDict):
    """One turn of a conversation, in the shape every provider accepts."""

    role: Role
    content: str


@dataclass
class ChatResponse:
    """A provider reply, with the metadata a caller may want to log."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMError(Exception):
    """Raised when no provider can serve a request or a provider call fails."""


class BaseLLM(ABC):
    """Contract that every LLM provider implementation follows."""

    def __init__(self, config: LLMSettings) -> None:
        self.config = config

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        trace: TraceContext | None = None,
    ) -> str:
        """Return the assistant reply for `messages`."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        trace: TraceContext | None = None,
    ) -> ChatResponse:
        """Return the assistant reply together with the provider metadata."""
        return ChatResponse(
            content=self.chat(messages, trace),
            model=self.config.model or "",
        )

    def validate_messages(self, messages: Sequence[ChatMessage]) -> None:
        """Raise `ValueError` unless `messages` is a usable conversation."""
        if not isinstance(messages, (list, tuple)):
            raise ValueError("messages must be a list of chat messages")
        if not messages:
            raise ValueError("messages cannot be empty")
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(
                    f"message at index {index} is not a mapping "
                    f"(type: {type(message).__name__})"
                )
            role = message.get("role")
            if role not in ROLES:
                raise ValueError(f"message at index {index} has an unknown role: {role!r}")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"message at index {index} must have string content")
