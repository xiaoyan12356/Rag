"""Rank fusion.

The dense route and the sparse route each bring their own list of chunks, and
the two lists rarely agree. Reciprocal Rank Fusion puts them into one list
without having to compare the scores of the two routes, which are not on the
same scale: a distance and a BM25 score mean nothing to each other.

Instead of the scores, the fusion uses the places of the chunks. A chunk that
both routes bring back is worth more than a chunk that only one of them does,
and a chunk that comes early is worth more than one that comes late.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from core.trace.trace_context import TraceContext
from core.types import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


class RRFFusion:
    """Put the results of several routes into one list.

    The score of a chunk is the sum of `1 / (k + place)` over the routes that
    found it, where the place starts at one. The `k` value decides how much
    the difference between the first place and the tenth place counts; a
    larger `k` makes the ranking flatter.
    """

    def __init__(self, k: int = DEFAULT_RRF_K) -> None:
        """Set up the fusion.

        Args:
            k: How strongly the place of a chunk counts.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def fuse(
        self,
        dense_results: list[RetrievalResult] | None = None,
        sparse_results: list[RetrievalResult] | None = None,
        top_k: int | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Return the results of all routes in one list.

        Args:
            dense_results: What the dense route found.
            sparse_results: What the sparse route found.
            top_k: How many results to return; every result is returned when
                this is left out.
            trace: A trace to record the work in.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        dense = dense_results or []
        sparse = sparse_results or []

        scores: dict[str, float] = {}
        sources: dict[str, RetrievalResult] = {}
        dense_by_id = {result.chunk_id: result for result in dense}
        sparse_by_id = {result.chunk_id: result for result in sparse}

        for results in (dense, sparse):
            for place, result in enumerate(results, start=1):
                scores[result.chunk_id] = (
                    scores.get(result.chunk_id, 0.0) + 1.0 / (self.k + place)
                )
                sources.setdefault(result.chunk_id, result)

        merged = [
            self._merge(
                source=sources[chunk_id],
                score=score,
                dense_result=dense_by_id.get(chunk_id),
                sparse_result=sparse_by_id.get(chunk_id),
            )
            for chunk_id, score in scores.items()
        ]
        merged.sort(key=lambda result: (-result.score, result.chunk_id))

        for rank, result in enumerate(merged, start=1):
            result.rank = rank

        if top_k is not None:
            merged = merged[:top_k]

        if trace is not None:
            trace.record_stage(
                "fusion",
                {
                    "k": self.k,
                    "dense_count": len(dense),
                    "sparse_count": len(sparse),
                    "fused_count": len(merged),
                    "chunk_ids": [result.chunk_id for result in merged],
                },
            )
        logger.debug("The fusion returned %d results", len(merged))
        return merged

    def _merge(
        self,
        source: RetrievalResult,
        score: float,
        dense_result: RetrievalResult | None,
        sparse_result: RetrievalResult | None,
    ) -> RetrievalResult:
        """Give the chunk its fused score, keeping the score of each route.

        The text and the metadata of the chunk are kept as they came, and the
        score of every route that found it is kept beside the fused score, so
        the result can be explained after the fact.
        """
        return replace(
            source,
            score=score,
            dense_score=dense_result.score if dense_result is not None else None,
            sparse_score=sparse_result.score if sparse_result is not None else None,
        )
