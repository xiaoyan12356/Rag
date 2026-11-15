"""Splitter abstraction.

Chunking strategies disagree on where a chunk may end - a recursive splitter
respects separators, a semantic splitter respects meaning - so callers depend
on `BaseSplitter` and never on a concrete strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.settings import IngestionSettings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class SplitterError(Exception):
    """Raised when no strategy can serve a request or a strategy call fails."""


class BaseSplitter(ABC):
    """Contract that every chunking strategy implementation follows."""

    def __init__(self, config: IngestionSettings) -> None:
        self.config = config

    @abstractmethod
    def split_text(self, text: str, trace: TraceContext | None = None) -> list[str]:
        """Split `text` into chunks, in reading order."""
