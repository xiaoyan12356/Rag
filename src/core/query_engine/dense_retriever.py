"""Dense retrieval.

The dense route finds chunks that mean the same as the question. The question
is turned into a vector, the vector store is asked for the chunks closest to
it, and every match comes back as a `RetrievalResult`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10


class DenseRetriever:
    """Find the chunks that mean the same as the question.

    The embedding client and the vector store can be handed in, which is what
    lets the retriever be exercised without a network and without a database.
    """

    def __init__(
        self,
        settings: Settings,
        embedding_client: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
        collection: str | None = None,
    ) -> None:
        """Set up the retriever.

        Args:
            settings: The settings that say which providers to use.
            embedding_client: The client that turns the question into a
                vector; one is built from the settings when this is left out.
            vector_store: The store that holds the vectors; one is built from
                the settings when this is left out.
            collection: The collection to search; the one named in the
                settings is used when this is left out.
        """
        self.settings = settings
        if collection:
            settings = replace(
                settings,
                vector_store=replace(
                    settings.vector_store, collection_name=collection
                ),
            )
        self.collection = collection or settings.vector_store.collection_name
        self.embedding_client = embedding_client or EmbeddingFactory.create(settings)
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Return the chunks that are closest to the question.

        Args:
            query: The question to search for.
            top_k: How many results to return; the configured number is used
                when this is left out.
            filters: The metadata the results have to match.
            trace: A trace to record the work in.
        """
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query cannot be empty or whitespace-only")

        limit = (
            top_k
            if top_k is not None
            else (self.settings.retrieval.dense_top_k or DEFAULT_TOP_K)
        )
        if limit <= 0:
            raise ValueError(f"top_k must be positive, got {limit}")

        vectors = self.embedding_client.embed([query], trace)
        if not vectors:
            raise RuntimeError("The embedding provider returned no vector for the query")

        matches = self.vector_store.query(vectors[0], limit, filters, trace)
        results = [self._to_result(match) for match in matches]

        if trace is not None:
            trace.record_stage(
                "dense_retrieval",
                {
                    "collection": self.collection,
                    "top_k": limit,
                    "returned": len(results),
                    "chunk_ids": [result.chunk_id for result in results],
                    "filters": dict(filters or {}),
                },
            )
        logger.debug("The dense route returned %d results", len(results))
        return results

    @staticmethod
    def _to_result(match: dict[str, Any]) -> RetrievalResult:
        """Turn a match of the vector store into a retrieval result."""
        metadata = match.get("metadata") or {}
        text = match.get("text")
        if not isinstance(text, str) or not text:
            text = str(metadata.get("text", ""))
        score = float(match.get("score", 0.0))
        return RetrievalResult(
            chunk_id=str(match.get("id", "")),
            score=score,
            text=text,
            metadata=dict(metadata),
            dense_score=score,
        )
