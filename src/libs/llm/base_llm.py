"""LLM abstraction.

Every provider hides a different authentication scheme and request format
behind the same `chat` call, so callers depend on `BaseLLM` and never on a
concrete client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, TypedDict

from core.settings import LLMSettings

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    """One turn of a conversation, in the shape every provider accepts."""

    role: Role
    content: str


class LLMError(Exception):
    """Raised when no provider can serve a request or a provider call fails."""


class BaseLLM(ABC):
    """Contract that every LLM provider implementation follows."""

    def __init__(self, config: LLMSettings) -> None:
        self.config = config

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant reply for `messages`."""
