"""Evaluator abstraction.

An evaluator turns one query's retrieval result and its golden answer into
metrics. Frameworks differ in what they measure - Ragas scores faithfulness
and context precision, a custom evaluator counts hits - so callers depend on
`BaseEvaluator` and never on a concrete framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from core.settings import EvaluationSettings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class EvaluatorError(Exception):
    """Raised when no framework can serve a request or a framework call fails."""


class BaseEvaluator(ABC):
    """Contract that every evaluation framework implementation follows."""

    def __init__(self, config: EvaluationSettings) -> None:
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_chunks: Sequence[Any],
        generated_answer: str | None = None,
        ground_truth: Any = None,
        trace: TraceContext | None = None,
    ) -> dict[str, float]:
        """Return the metrics for one query, keyed by metric name."""

    def validate_query(self, query: str) -> None:
        """Raise `ValueError` unless `query` is a non-empty string."""
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query cannot be empty or whitespace-only")

    def validate_retrieved_chunks(self, retrieved_chunks: Sequence[Any]) -> None:
        """Raise `ValueError` unless `retrieved_chunks` is a non-empty list."""
        if not isinstance(retrieved_chunks, (list, tuple)):
            raise ValueError("retrieved_chunks must be a list")
        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")


class NoneEvaluator(BaseEvaluator):
    """Returns no metrics; used when evaluation is disabled."""

    def evaluate(
        self,
        query: str,
        retrieved_chunks: Sequence[Any],
        generated_answer: str | None = None,
        ground_truth: Any = None,
        trace: TraceContext | None = None,
    ) -> dict[str, float]:
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        return {}
