"""Embedding abstraction.

Providers disagree on request format, batch limits and vector width, so
callers depend on `BaseEmbedding` and never on a concrete client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from core.settings import EmbeddingSettings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class EmbeddingError(Exception):
    """Raised when no provider can serve a request or a provider call fails."""


class BaseEmbedding(ABC):
    """Contract that every embedding provider implementation follows."""

    def __init__(self, config: EmbeddingSettings) -> None:
        self.config = config

    @abstractmethod
    def embed(
        self, texts: list[str], trace: TraceContext | None = None
    ) -> list[list[float]]:
        """Return one vector per input text, in the same order as `texts`."""

    def validate_texts(self, texts: Sequence[str]) -> None:
        """Raise `ValueError` unless `texts` is a usable batch of strings."""
        if not isinstance(texts, (list, tuple)):
            raise ValueError("texts must be a list of strings")
        if not texts:
            raise ValueError("texts cannot be empty")
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"text at index {index} is not a string (type: {type(text).__name__})"
                )
