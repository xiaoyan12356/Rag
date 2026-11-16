"""Sparse encoding.

A dense vector answers "is this about the same thing". A sparse one answers
"does this use the same words". The encoder counts the terms of every chunk,
which is what BM25 later turns into an index.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from core.query_engine.tokenizer import DEFAULT_MIN_TERM_LENGTH, tokenize
from core.trace.trace_context import TraceContext
from core.types import Chunk

logger = logging.getLogger(__name__)


class SparseEncoder:
    """Count the terms of every chunk, for the BM25 index to work on.

    The output of one chunk is what BM25 needs to know about it: how often
    each term occurs, how long the chunk is, and how many different terms it
    holds.
    """

    def __init__(
        self,
        min_term_length: int = DEFAULT_MIN_TERM_LENGTH,
        lowercase: bool = True,
    ) -> None:
        if min_term_length < 1:
            raise ValueError(f"min_term_length must be at least 1, got {min_term_length}")
        self.min_term_length = min_term_length
        self.lowercase = lowercase

    def encode(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return the term statistics of every chunk, in the order they came in."""
        if not chunks:
            raise ValueError("Cannot encode an empty list of chunks")

        results: list[dict[str, Any]] = []
        for chunk in chunks:
            terms = self._tokenize(chunk.text)
            frequencies = Counter(terms)
            results.append(
                {
                    "chunk_id": chunk.id,
                    "term_frequencies": dict(frequencies),
                    "doc_length": len(terms),
                    "unique_terms": len(frequencies),
                }
            )

        if trace is not None:
            trace.record_stage(
                "sparse_encoder",
                {
                    "total_chunks": len(chunks),
                    "total_terms": sum(result["doc_length"] for result in results),
                    "unique_terms": len(
                        {term for result in results for term in result["term_frequencies"]}
                    ),
                },
            )

        return results

    def get_corpus_stats(self, encoded_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Return what the index needs to know about the corpus as a whole."""
        if not encoded_chunks:
            return {"num_docs": 0, "avg_doc_length": 0.0, "document_frequency": {}}

        num_docs = len(encoded_chunks)
        total_length = sum(chunk["doc_length"] for chunk in encoded_chunks)

        document_frequency: dict[str, int] = {}
        for chunk in encoded_chunks:
            for term in chunk["term_frequencies"]:
                document_frequency[term] = document_frequency.get(term, 0) + 1

        return {
            "num_docs": num_docs,
            "avg_doc_length": total_length / num_docs,
            "document_frequency": document_frequency,
        }

    def _tokenize(self, text: str) -> list[str]:
        """Return the terms of the text.

        The text is cut the same way the question is cut at search time, so
        that the words of a question can be looked up in the index.
        """
        return tokenize(
            text,
            min_term_length=self.min_term_length,
            lowercase=self.lowercase,
        )
