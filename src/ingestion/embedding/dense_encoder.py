"""Dense encoding.

A chunk becomes a vector, so that a question can be turned into a vector as
well and the two compared. The encoder hands the text of the chunks to an
embedding provider and keeps the vectors in the order of the chunks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from core.trace.trace_context import TraceContext
from core.types import Chunk
from libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


class DenseEncoder:
    """Turn chunks into dense vectors.

    The provider is given the text of the chunks and answers with one vector
    per text. A chunk without text gets a vector of zeros, so that the
    vectors keep lining up with the chunks.
    """

    def __init__(self, embedding: BaseEmbedding, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.embedding = embedding
        self.batch_size = batch_size

    def encode(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[list[float]]:
        """Return one vector per chunk, in the order the chunks came in."""
        if not chunks:
            raise ValueError("Cannot encode an empty list of chunks")

        vectors: list[list[float] | None] = [None] * len(chunks)
        for start, batch in self._batches(chunks):
            self._encode_batch(batch, start, vectors, trace)

        dimension = next((len(vector) for vector in vectors if vector is not None), 0)
        if dimension == 0:
            dimension = self.embedding.config.dimensions

        for index, vector in enumerate(vectors):
            if vector is None:
                vectors[index] = [0.0] * dimension
            elif len(vector) != dimension:
                raise RuntimeError(
                    f"The vectors have different widths: the vector at index {index} has "
                    f"{len(vector)}, the first one has {dimension}"
                )

        if trace is not None:
            trace.record_stage(
                "dense_encoder",
                {
                    "total_chunks": len(chunks),
                    "batch_count": self.get_batch_count(len(chunks)),
                    "dimension": dimension,
                },
            )

        encoded: list[list[float]] = []
        for vector in vectors:
            if vector is None:
                raise RuntimeError("A chunk was left without a vector")
            encoded.append(vector)
        return encoded

    def get_batch_count(self, num_chunks: int) -> int:
        """Return how many batches a number of chunks is sent in."""
        if num_chunks <= 0:
            return 0
        return (num_chunks + self.batch_size - 1) // self.batch_size

    def _batches(self, chunks: list[Chunk]) -> Iterator[tuple[int, list[Chunk]]]:
        """Yield the batches of the chunks, with the index of the first one."""
        for start in range(0, len(chunks), self.batch_size):
            yield start, chunks[start : start + self.batch_size]

    def _encode_batch(
        self,
        batch: list[Chunk],
        start: int,
        vectors: list[list[float] | None],
        trace: TraceContext | None,
    ) -> None:
        """Ask the provider for the vectors of one batch and keep them."""
        askable = [
            (offset, chunk)
            for offset, chunk in enumerate(batch)
            if chunk.text and chunk.text.strip()
        ]
        if not askable:
            return

        texts = [chunk.text for _, chunk in askable]
        answer = self.embedding.embed(texts, trace)
        if len(answer) != len(texts):
            raise RuntimeError(
                f"The embedding provider returned {len(answer)} vectors for "
                f"{len(texts)} texts"
            )

        for (offset, _), vector in zip(askable, answer):
            vectors[start + offset] = vector
