"""Unit tests for the batch processor.

The processor drives the dense and the sparse encoder over the chunks, a
batch at a time. The tests pin the order of the batches, and that one batch
failing does not stop the ones after it.
"""

from __future__ import annotations

import pytest

from core.settings import EmbeddingSettings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder
from libs.embedding.base_embedding import BaseEmbedding

DIMENSION = 3


class FakeEmbedding(BaseEmbedding):
    """Provider that answers with a vector per text."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        super().__init__(
            EmbeddingSettings(provider="fake", model="fake", dimensions=dimension)
        )
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, texts, trace=None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text))] * self.dimension for text in texts]


def make_chunk(text: str = "Some text to process.", chunk_id: str = "chunk_1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "data/documents/sample.pdf"},
        source_ref="doc_1",
    )


def make_chunks(count: int) -> list[Chunk]:
    return [make_chunk(f"Text number {index}.", f"chunk_{index}") for index in range(count)]


def make_processor(batch_size: int = 2, embedding=None) -> BatchProcessor:
    return BatchProcessor(
        DenseEncoder(embedding or FakeEmbedding()),
        SparseEncoder(),
        batch_size=batch_size,
    )


class TestBatching:
    def test_the_chunks_are_split_into_batches(self):
        processor = make_processor(batch_size=2)

        assert processor.get_batch_count(5) == 3
        assert processor.get_batch_count(4) == 2
        assert processor.get_batch_count(2) == 1
        assert processor.get_batch_count(1) == 1
        assert processor.get_batch_count(0) == 0

    def test_a_batch_size_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="batch_size"):
            make_processor(batch_size=0)

    def test_no_chunks_gives_nothing(self):
        processor = make_processor()

        with pytest.raises(ValueError, match="empty"):
            processor.process([])


class TestProcess:
    def test_every_chunk_gets_a_vector(self):
        processor = make_processor(batch_size=2)

        result = processor.process(make_chunks(5))

        assert len(result.dense_vectors) == 5
        assert len(result.sparse_stats) == 5

    def test_the_vectors_keep_the_order_of_the_chunks(self):
        processor = make_processor(batch_size=2)

        result = processor.process(
            [make_chunk("one", "chunk_1"), make_chunk("three", "chunk_2")]
        )

        assert result.dense_vectors[0][0] == 3.0
        assert result.dense_vectors[1][0] == 5.0

    def test_the_statistics_keep_the_order_of_the_chunks(self):
        processor = make_processor(batch_size=2)

        result = processor.process(
            [make_chunk("alpha", "chunk_1"), make_chunk("beta", "chunk_2")]
        )

        assert [item["chunk_id"] for item in result.sparse_stats] == ["chunk_1", "chunk_2"]

    def test_the_number_of_batches_is_counted(self):
        processor = make_processor(batch_size=2)

        result = processor.process(make_chunks(5))

        assert result.batch_count == 3
        assert result.successful_chunks == 5
        assert result.failed_chunks == 0

    def test_the_time_is_measured(self):
        processor = make_processor(batch_size=2)

        result = processor.process(make_chunks(3))

        assert result.total_time >= 0.0

    def test_the_provider_is_asked_batch_by_batch(self):
        embedding = FakeEmbedding()
        processor = make_processor(batch_size=2, embedding=embedding)

        processor.process(make_chunks(5))

        assert [len(call) for call in embedding.calls] == [2, 2, 1]


class TestFailure:
    def test_a_failing_batch_does_not_stop_the_ones_after_it(self):
        embedding = FakeEmbedding()
        calls: list[int] = []
        original_embed = embedding.embed

        def embed(texts, trace=None):
            calls.append(len(texts))
            if len(calls) == 2:
                raise RuntimeError("the provider is down")
            return original_embed(texts, trace)

        embedding.embed = embed
        processor = make_processor(batch_size=2, embedding=embedding)

        result = processor.process(make_chunks(5))

        assert result.failed_chunks == 2
        assert result.successful_chunks == 3
        assert len(result.dense_vectors) == 3

    def test_a_failing_batch_is_written_down(self):
        embedding = FakeEmbedding()
        calls: list[int] = []
        original_embed = embedding.embed

        def embed(texts, trace=None):
            calls.append(len(texts))
            if len(calls) == 2:
                raise RuntimeError("the provider is down")
            return original_embed(texts, trace)

        embedding.embed = embed
        processor = make_processor(batch_size=2, embedding=embedding)
        trace = TraceContext()

        processor.process(make_chunks(5), trace)

        error = trace.get_stage_data("batch_1_error")
        assert error is not None
        assert "the provider is down" in error["error"]
        assert error["batch_size"] == 2


class TestTrace:
    def test_the_trace_records_what_happened(self):
        processor = make_processor(batch_size=2)
        trace = TraceContext()

        processor.process(make_chunks(5), trace)

        data = trace.get_stage_data("batch_processing")
        assert data["total_chunks"] == 5
        assert data["batch_count"] == 3
        assert data["successful_chunks"] == 5
        assert data["failed_chunks"] == 0

    def test_the_trace_records_every_batch(self):
        processor = make_processor(batch_size=2)
        trace = TraceContext()

        processor.process(make_chunks(5), trace)

        first = trace.get_stage_data("batch_0")
        assert first["batch_size"] == 2
        assert first["chunks_processed"] == 2
        assert "duration_ms" in first
