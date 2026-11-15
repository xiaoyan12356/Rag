"""Reranker abstraction.

A reranker reorders the candidates a retriever produced. Strategies differ in
how they judge - a cross-encoder scores the query and passage together, an
LLM writes a verdict, and the none strategy keeps the retriever's order - so
callers depend on `BaseReranker` and never on a concrete strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from core.settings import RerankSettings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class RerankerError(Exception):
    """Raised when no strategy can serve a request or a strategy call fails."""


class BaseReranker(ABC):
    """Contract that every reranking strategy implementation follows."""

    def __init__(self, config: RerankSettings) -> None:
        self.config = config
        self.fallback_used = False

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return `candidates` reordered by relevance to `query`.

        The candidate records themselves are preserved; only their order
        changes. `fallback_used` tells the caller whether the returned order
        is a real ranking or the retriever's order kept after a failure.
        """

    def validate_query(self, query: str) -> None:
        """Raise `ValueError` unless `query` is a non-empty string."""
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query cannot be empty or whitespace-only")

    def validate_candidates(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        """Raise `ValueError` unless `candidates` is a non-empty list of mappings."""
        if not isinstance(candidates, (list, tuple)):
            raise ValueError("candidates must be a list of records")
        if not candidates:
            raise ValueError("candidates cannot be empty")
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"candidate at index {index} is not a mapping "
                    f"(type: {type(candidate).__name__})"
                )


class NoneReranker(BaseReranker):
    """Keeps the candidates in the order the retriever produced them."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        return list(candidates)
