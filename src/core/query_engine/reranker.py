"""Reranking.

The reranker puts the best chunks of a result list in front. The backends
differ in how they judge - a cross-encoder scores the question and the chunk
together, an LLM writes a verdict - but they all take the same list and
return it in a new order.

A backend that is down must not empty the answer, so a failure keeps the
order the fusion found and says so.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker
from libs.reranker.reranker_factory import RerankerFactory

logger = logging.getLogger(__name__)


class Reranker:
    """Reorder the results of a search, and fall back when that fails."""

    def __init__(
        self,
        settings: Settings,
        reranker: BaseReranker | None = None,
    ) -> None:
        """Set up the reranker.

        Args:
            settings: The settings that say which strategy to use.
            reranker: The backend that does the judging; one is built from
                the settings when this is left out.
        """
        self.settings = settings
        self.backend = reranker if reranker is not None else RerankerFactory.create(settings)
        self.fallback_used = False

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Return the results in the order the backend judged them best.

        Args:
            query: The question the results answer.
            results: The results to reorder.
            top_k: How many results to return; every result is returned when
                this is left out.
            trace: A trace to record the work in.
        """
        self.fallback_used = False

        if not results:
            return []

        started = time.monotonic()
        candidates = [
            {
                "id": result.chunk_id,
                "text": result.text,
                "metadata": dict(result.metadata),
            }
            for result in results
        ]

        try:
            reordered = self.backend.rerank(query, candidates, trace)
        except Exception as exc:
            logger.warning("The reranker failed, the original order is kept: %s", exc)
            self.fallback_used = True
            self._record(
                trace,
                query,
                results,
                results,
                fallback=True,
                error=str(exc),
                elapsed_ms=(time.monotonic() - started) * 1000.0,
            )
            return self._take(results, top_k)

        self.fallback_used = bool(getattr(self.backend, "fallback_used", False))
        merged = self._merge(results, reordered)

        if trace is not None:
            self._record(
                trace,
                query,
                results,
                merged,
                fallback=self.fallback_used,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
            )

        return self._take(merged, top_k)

    def _merge(
        self,
        results: list[RetrievalResult],
        reordered: list[dict[str, Any]],
    ) -> list[RetrievalResult]:
        """Put the results back in the order the backend gave them.

        A backend that leaves a result out does not lose it: it goes back to
        the end of the list, where the fusion had put it.
        """
        by_id = {result.chunk_id: result for result in results}
        merged: list[RetrievalResult] = []
        placed: set[str] = set()

        for candidate in reordered:
            chunk_id = str(candidate.get("id", ""))
            original = by_id.get(chunk_id)
            if original is None or chunk_id in placed:
                continue
            placed.add(chunk_id)
            merged.append(replace(original, rank=len(merged) + 1))

        for result in results:
            if result.chunk_id not in placed:
                placed.add(result.chunk_id)
                merged.append(replace(result, rank=len(merged) + 1))

        return merged

    @staticmethod
    def _take(
        results: list[RetrievalResult],
        top_k: int | None,
    ) -> list[RetrievalResult]:
        """Return the first `top_k` results, or all of them."""
        if top_k is None:
            return results
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        return results[:top_k]

    def _record(
        self,
        trace: TraceContext | None,
        query: str,
        before: list[RetrievalResult],
        after: list[RetrievalResult],
        fallback: bool,
        error: str | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        """Record what the reranker did."""
        if trace is None:
            return
        data: dict[str, Any] = {
            "provider": self.settings.rerank.provider or "none",
            "method": "rerank",
            "query": query,
            "candidates": len(before),
            "fallback_used": fallback,
            "order_before": [result.chunk_id for result in before],
            "order_after": [result.chunk_id for result in after],
        }
        if error is not None:
            data["error"] = error
        trace.record_stage("rerank", data, elapsed_ms=elapsed_ms)
