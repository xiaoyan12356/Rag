"""Sparse retrieval.

The sparse route finds chunks that use the same words as the question. The
words are looked up in the BM25 index, which names the chunks that match and
how well they match. The text of those chunks is then read back from the
vector store, so the results carry the same shape as the dense route.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10


class SparseRetriever:
    """Find the chunks that use the same words as the question.

    The BM25 index and the vector store can be handed in, which is what lets
    the retriever be exercised without a network and without a database.
    """

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: BM25Indexer | None = None,
        vector_store: BaseVectorStore | None = None,
        collection: str | None = None,
    ) -> None:
        """Set up the retriever.

        Args:
            settings: The settings that say which providers to use.
            bm25_indexer: The index that holds the terms of the corpus; one
                is built from the settings when this is left out.
            vector_store: The store that holds the texts of the chunks; one is
                built from the settings when this is left out.
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
        self.bm25_indexer = bm25_indexer or BM25Indexer(
            index_dir=settings.ingestion.bm25_index_dir
        )
        self.vector_store = vector_store or VectorStoreFactory.create(settings)
        self._index_loaded = False

    def retrieve(
        self,
        keywords: list[str],
        top_k: int | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Return the chunks that match the words best.

        Args:
            keywords: The words of the question, as the query processor cut them.
            top_k: How many results to return; the configured number is used
                when this is left out.
            trace: A trace to record the work in.
        """
        if not isinstance(keywords, list):
            raise ValueError(f"keywords must be a list, got {type(keywords).__name__}")
        terms = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        if not terms:
            raise ValueError("keywords cannot be empty")

        limit = (
            top_k
            if top_k is not None
            else (self.settings.retrieval.sparse_top_k or DEFAULT_TOP_K)
        )
        if limit <= 0:
            raise ValueError(f"top_k must be positive, got {limit}")

        self._load_index()
        matches = self.bm25_indexer.query(terms, limit)
        results = self._to_results(matches)

        if trace is not None:
            trace.record_stage(
                "sparse_retrieval",
                {
                    "collection": self.collection,
                    "keywords": terms,
                    "top_k": limit,
                    "returned": len(results),
                    "chunk_ids": [result.chunk_id for result in results],
                },
            )
        logger.debug("The sparse route returned %d results", len(results))
        return results

    def _load_index(self) -> None:
        """Read the index from disk, once."""
        if self._index_loaded:
            return
        loaded = self.bm25_indexer.load(self.collection)
        if not loaded:
            logger.debug("No BM25 index was found for the collection '%s'", self.collection)
        self._index_loaded = True

    def _to_results(self, matches: list[dict[str, Any]]) -> list[RetrievalResult]:
        """Turn the matches of the index into retrieval results.

        The index knows which chunk matches and how well it matches, but not
        what the chunk says, so the texts are read back from the vector store.
        """
        if not matches:
            return []

        chunk_ids = [str(match["chunk_id"]) for match in matches]
        records = self.vector_store.get_by_ids(chunk_ids)
        by_id = {str(record.get("id", "")): record for record in records}

        results: list[RetrievalResult] = []
        for match in matches:
            chunk_id = str(match["chunk_id"])
            score = float(match.get("score", 0.0))
            record = by_id.get(chunk_id)
            if record is None:
                logger.debug("The chunk '%s' is in the index but not in the store", chunk_id)
                continue
            metadata = dict(record.get("metadata") or {})
            text = record.get("text")
            if not isinstance(text, str) or not text:
                text = str(metadata.get("text", ""))
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=text,
                    metadata=metadata,
                    sparse_score=score,
                )
            )
        return results
