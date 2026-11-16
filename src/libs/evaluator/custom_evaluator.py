"""Lightweight metrics that need no model call.

The custom evaluator answers the questions a regression suite asks: did the
retrieval hit a golden chunk, and how high did that chunk rank.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class CustomEvaluator(BaseEvaluator):
    """Computes hit rate and MRR from retrieved and golden ids."""

    SUPPORTED_METRICS = ("hit_rate", "mrr")
    ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")

    def __init__(self, config: EvaluationSettings) -> None:
        super().__init__(config)
        requested = [str(metric).strip().lower() for metric in (config.metrics or [])]
        self.metrics = requested or list(self.SUPPORTED_METRICS)

        unsupported = [metric for metric in self.metrics if metric not in self.SUPPORTED_METRICS]
        if unsupported:
            raise EvaluatorError(
                f"Unsupported custom metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(self.SUPPORTED_METRICS)}"
            )

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

        retrieved_ids = self._extract_ids(retrieved_chunks, "retrieved_chunks")
        golden_ids = self._extract_golden_ids(ground_truth)

        results: dict[str, float] = {}
        if "hit_rate" in self.metrics:
            results["hit_rate"] = self._hit_rate(retrieved_ids, golden_ids)
        if "mrr" in self.metrics:
            results["mrr"] = self._mrr(retrieved_ids, golden_ids)
        return results

    def _extract_ids(self, items: Iterable[Any], label: str) -> list[str]:
        """Pull an identifier out of each item, whatever shape it arrives in."""
        ids: list[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, Mapping):
                for field in self.ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise EvaluatorError(
                        f"Missing id field in {label}[{index}]. "
                        f"Expected one of {', '.join(self.ID_FIELDS)}"
                    )
                continue
            for field in self.ID_FIELDS:
                if hasattr(item, field):
                    ids.append(str(getattr(item, field)))
                    break
            else:
                raise EvaluatorError(
                    f"Unable to extract id from {label}[{index}] of type {type(item).__name__}"
                )
        return ids

    def _extract_golden_ids(self, ground_truth: Any) -> list[str]:
        """Read the golden ids out of whichever shape the caller passed."""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, Mapping):
            if isinstance(ground_truth.get("ids"), list):
                return self._extract_ids(ground_truth["ids"], "ground_truth.ids")
            return self._extract_ids([ground_truth], "ground_truth")
        if isinstance(ground_truth, (list, tuple)):
            return self._extract_ids(ground_truth, "ground_truth")
        raise EvaluatorError(
            f"Unsupported ground_truth type: {type(ground_truth).__name__}. "
            "Expected a string, a mapping, a list or None."
        )

    def _hit_rate(self, retrieved_ids: Sequence[str], golden_ids: Sequence[str]) -> float:
        """Whether any retrieved chunk is one of the golden chunks."""
        if not golden_ids:
            return 0.0
        return 1.0 if any(item in golden_ids for item in retrieved_ids) else 0.0

    def _mrr(self, retrieved_ids: Sequence[str], golden_ids: Sequence[str]) -> float:
        """Reciprocal rank of the first golden chunk in the retrieved order."""
        if not golden_ids:
            return 0.0
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in golden_ids:
                return 1.0 / rank
        return 0.0
