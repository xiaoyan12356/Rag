"""Batch processing.

The encoders are asked one batch at a time, so that a long document does not
become one huge request, and so that the time each batch took can be written
down.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


@dataclass
class BatchResult:
    """What the encoding of a document came to."""

    dense_vectors: list[list[float]] = field(default_factory=list)
    sparse_stats: list[dict[str, Any]] = field(default_factory=list)
    batch_count: int = 0
    total_time: float = 0.0
    successful_chunks: int = 0
    failed_chunks: int = 0


class BatchProcessor:
    """Drive the dense and the sparse encoder over the chunks, batch by batch."""

    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def process(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> BatchResult:
        """Return the vectors and the term statistics of every chunk.

        A batch that cannot be encoded does not stop the ones after it; the
        chunks of that batch are counted as failed.
        """
        if not chunks:
            raise ValueError("Cannot process an empty list of chunks")

        started = time.monotonic()
        result = BatchResult(batch_count=self.get_batch_count(len(chunks)))

        for index, batch in enumerate(self._batches(chunks)):
            batch_started = time.monotonic()
            try:
                result.dense_vectors.extend(self.dense_encoder.encode(batch, trace))
                result.sparse_stats.extend(self.sparse_encoder.encode(batch, trace))
                result.successful_chunks += len(batch)
            except Exception as exc:
                logger.error("Could not encode the batch %s: %s", index, exc)
                result.failed_chunks += len(batch)
                if trace is not None:
                    trace.record_stage(
                        f"batch_{index}_error",
                        {"error": str(exc), "batch_size": len(batch)},
                    )

            if trace is not None:
                trace.record_stage(
                    f"batch_{index}",
                    {
                        "batch_size": len(batch),
                        "duration_ms": round((time.monotonic() - batch_started) * 1000, 2),
                        "chunks_processed": len(batch),
                    },
                )

        result.total_time = time.monotonic() - started

        if trace is not None:
            trace.record_stage(
                "batch_processing",
                {
                    "total_chunks": len(chunks),
                    "batch_count": result.batch_count,
                    "batch_size": self.batch_size,
                    "successful_chunks": result.successful_chunks,
                    "failed_chunks": result.failed_chunks,
                    "total_time_ms": round(result.total_time * 1000, 2),
                },
            )

        return result

    def get_batch_count(self, total_chunks: int) -> int:
        """Return how many batches a number of chunks is split into."""
        if total_chunks <= 0:
            return 0
        return (total_chunks + self.batch_size - 1) // self.batch_size

    def _batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split the chunks into batches, keeping their order."""
        return [
            chunks[start : start + self.batch_size]
            for start in range(0, len(chunks), self.batch_size)
        ]
