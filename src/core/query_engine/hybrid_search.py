"""Hybrid search.

The hybrid search puts the two routes together. The question is cut into the
words the search needs, the dense route and the sparse route look for chunks
in their own way, the fusion puts their answers into one list, and the
metadata filters are applied to what is left.

One route failing does not end the search: the answer of the other route is
used on its own. Only when both routes fail is the search a failure.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFusion
from core.query_engine.query_processor import ProcessedQuery, QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10
COLLECTION_KEY = "collection"

# The method and the provider of every stage of the query, so that a trace
# says which route did the work and not only that it was done.
STAGE_METHODS = {
    "query_processing": ("tokenize", "builtin"),
    "dense_retrieval": ("dense", "embedding"),
    "sparse_retrieval": ("bm25", "builtin"),
    "fusion": ("rrf", "builtin"),
}


class HybridSearch:
    """Search the corpus with the dense route, the sparse route, and both."""

    def __init__(
        self,
        settings: Settings,
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        fusion: RRFFusion | None = None,
        collection: str | None = None,
    ) -> None:
        """Set up the search.

        Args:
            settings: The settings that say which providers to use.
            query_processor: The one that cuts the question into words; one is
                built when this is left out.
            dense_retriever: The one that looks for chunks that mean the same;
                one is built when this is left out.
            sparse_retriever: The one that looks for chunks that use the same
                words; one is built when this is left out.
            fusion: The one that puts the two answers together; one is built
                when this is left out.
            collection: The collection to search.
        """
        self.settings = settings
        self.query_processor = query_processor or QueryProcessor()
        self.dense_retriever = dense_retriever or DenseRetriever(
            settings, collection=collection
        )
        self.sparse_retriever = sparse_retriever or SparseRetriever(
            settings, collection=collection
        )
        self.fusion = fusion or RRFFusion(k=settings.retrieval.rrf_k)
        self.collection = collection or settings.vector_store.collection_name

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Return the chunks that answer the question best.

        Args:
            query: The question to search for.
            top_k: How many results to return; the configured number is used
                when this is left out.
            filters: The metadata the results have to match.
            trace: A trace to record the work in.
        """
        started = time.monotonic()
        if trace is not None:
            # The search knows what kind of run this is, so the trace says
            # so too, whoever made it.
            trace.trace_type = "query"
        processed = self.query_processor.process(query, filters, trace)
        self._annotate(trace, "query_processing", started)

        limit = (
            top_k
            if top_k is not None
            else (self.settings.retrieval.fusion_top_k or DEFAULT_TOP_K)
        )
        if limit <= 0:
            raise ValueError(f"top_k must be positive, got {limit}")

        metadata_filters = self._metadata_filters(processed.filters)
        if not self._collection_matches(processed.filters):
            logger.info(
                "The question asks for the collection '%s' but the search runs on '%s'",
                processed.filters.get(COLLECTION_KEY),
                self.collection,
            )
            return []

        dense_results, sparse_results = self._retrieve_both(
            processed, limit, metadata_filters, trace
        )

        started = time.monotonic()
        fused = self.fusion.fuse(dense_results, sparse_results, trace=trace)
        self._annotate(trace, "fusion", started)

        filtered = self._apply_metadata_filters(fused, metadata_filters)

        return filtered[:limit]

    def _annotate(self, trace: TraceContext | None, stage: str, started: float) -> None:
        """Say how long a stage took and which method did it.

        The stage itself was recorded by the part that did the work, so only
        what that part cannot know is added here.
        """
        if trace is None:
            return
        method, provider = STAGE_METHODS[stage]
        if stage == "dense_retrieval":
            provider = self.settings.embedding.provider or "none"
        trace.annotate_stage(
            stage,
            method=method,
            provider=provider,
            _elapsed_ms=(time.monotonic() - started) * 1000.0,
        )

    def _retrieve_both(
        self,
        processed: ProcessedQuery,
        limit: int,
        filters: dict[str, Any],
        trace: TraceContext | None,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """Ask both routes, and let one of them carry the search alone.

        The dense route waits for the embedding provider and the sparse route
        works on the index on the disk, so they are asked at the same time.
        """
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(
                self._safe_dense, processed, limit, filters, trace
            )
            sparse_future = pool.submit(
                self._safe_sparse, processed, limit, trace
            )
            dense_results, dense_error = dense_future.result()
            sparse_results, sparse_error = sparse_future.result()

        if dense_error is not None and sparse_error is not None:
            raise RuntimeError(
                "Both retrieval routes failed: "
                f"dense: {dense_error}; sparse: {sparse_error}"
            ) from dense_error

        return dense_results, sparse_results

    def _safe_dense(
        self,
        processed: ProcessedQuery,
        limit: int,
        filters: dict[str, Any],
        trace: TraceContext | None,
    ) -> tuple[list[RetrievalResult], Exception | None]:
        """Ask the dense route, and report the failure instead of raising it."""
        started = time.monotonic()
        try:
            results = self.dense_retriever.retrieve(
                processed.original_query,
                top_k=limit,
                filters=filters or None,
                trace=trace,
            )
        except Exception as exc:
            logger.warning("The dense route failed, the sparse route carries on: %s", exc)
            return [], exc
        self._annotate(trace, "dense_retrieval", started)
        return list(results), None

    def _safe_sparse(
        self,
        processed: ProcessedQuery,
        limit: int,
        trace: TraceContext | None,
    ) -> tuple[list[RetrievalResult], Exception | None]:
        """Ask the sparse route, and report the failure instead of raising it."""
        started = time.monotonic()
        try:
            results = self.sparse_retriever.retrieve(
                processed.keywords,
                top_k=limit,
                trace=trace,
            )
        except Exception as exc:
            logger.warning("The sparse route failed, the dense route carries on: %s", exc)
            return [], exc
        self._annotate(trace, "sparse_retrieval", started)
        return list(results), None

    def _metadata_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Return the filters that name a field of the metadata.

        The collection is not one of them: the collection says which set of
        chunks is searched, not what a chunk looks like.
        """
        return {
            key: value for key, value in filters.items() if key != COLLECTION_KEY
        }

    def _collection_matches(self, filters: dict[str, Any]) -> bool:
        """Say whether the collection of the question is the one searched."""
        wanted = filters.get(COLLECTION_KEY)
        if wanted is None:
            return True
        return str(wanted) == str(self.collection)

    def _apply_metadata_filters(
        self,
        candidates: list[RetrievalResult],
        filters: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Leave out the results whose metadata does not match the filters.

        The dense route asks the vector store to do this already. The sparse
        route cannot, because the index knows nothing about the metadata, so
        the check is made here for the results of both routes.
        """
        if not filters:
            return candidates

        kept: list[RetrievalResult] = []
        for result in candidates:
            if self._matches(result.metadata, filters):
                kept.append(result)
        return kept

    @staticmethod
    def _matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Say whether the metadata of a chunk matches every filter.

        The value of the filter and the value of the metadata are compared as
        text, because one of them often comes from the question and the other
        from the document.
        """
        for key, wanted in filters.items():
            if key not in metadata:
                return False
            if str(metadata[key]) != str(wanted):
                return False
        return True
