"""Unit tests for the dense encoder.

The encoder turns chunks into vectors. The tests pin that there is one
vector per chunk, in the same order, and that the vectors all have the same
width.
"""

from __future__ import annotations

import pytest

from core.settings import EmbeddingSettings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder
from libs.embedding.base_embedding import BaseEmbedding

DIMENSION = 4


class FakeEmbedding(BaseEmbedding):
    """Provider that answers with a vector per text."""

    def __init__(self, dimension: int = DIMENSION, error: Exception | None = None) -> None:
        super().__init__(
            EmbeddingSettings(provider="fake", model="fake", dimensions=dimension)
        )
        self.dimension = dimension
        self.error = error
        self.calls: list[list[str]] = []

    def embed(self, texts, trace=None) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[float(len(text))] * self.dimension for text in texts]


def make_chunk(text: str = "Some text to encode.", chunk_id: str = "chunk_1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "data/documents/sample.pdf"},
        source_ref="doc_1",
    )


def make_chunks(count: int) -> list[Chunk]:
    return [make_chunk(f"Text number {index}.", f"chunk_{index}") for index in range(count)]


class TestEncode:
    def test_every_chunk_gets_a_vector(self):
        encoder = DenseEncoder(FakeEmbedding())

        vectors = encoder.encode(make_chunks(3))

        assert len(vectors) == 3

    def test_the_vectors_keep_the_order_of_the_chunks(self):
        encoder = DenseEncoder(FakeEmbedding())

        vectors = encoder.encode(
            [make_chunk("one"), make_chunk("two"), make_chunk("three")]
        )

        assert vectors[0][0] == 3.0
        assert vectors[1][0] == 3.0
        assert vectors[2][0] == 5.0

    def test_the_vectors_have_the_same_width(self):
        encoder = DenseEncoder(FakeEmbedding())

        vectors = encoder.encode(make_chunks(5))

        assert {len(vector) for vector in vectors} == {DIMENSION}

    def test_the_provider_is_asked_for_the_text_of_the_chunks(self):
        embedding = FakeEmbedding()
        encoder = DenseEncoder(embedding)

        encoder.encode([make_chunk("first"), make_chunk("second")])

        assert embedding.calls == [["first", "second"]]

    def test_the_chunks_are_asked_in_batches(self):
        embedding = FakeEmbedding()
        encoder = DenseEncoder(embedding, batch_size=2)

        encoder.encode(make_chunks(5))

        assert [len(call) for call in embedding.calls] == [2, 2, 1]

    def test_the_text_of_a_chunk_without_text_is_not_asked_for(self):
        embedding = FakeEmbedding()
        encoder = DenseEncoder(embedding)

        encoder.encode([make_chunk(""), make_chunk("text")])

        assert embedding.calls == [["text"]]

    def test_a_chunk_without_text_still_gets_a_vector(self):
        encoder = DenseEncoder(FakeEmbedding())

        vectors = encoder.encode([make_chunk(""), make_chunk("text")])

        assert len(vectors) == 2
        assert vectors[0] == [0.0] * DIMENSION

    def test_the_vectors_of_a_chunk_without_text_are_wide_enough(self):
        embedding = FakeEmbedding(dimension=7)
        encoder = DenseEncoder(embedding)

        vectors = encoder.encode([make_chunk("text"), make_chunk("")])

        assert vectors[1] == [0.0] * 7

    def test_no_chunks_gives_no_vectors(self):
        encoder = DenseEncoder(FakeEmbedding())

        with pytest.raises(ValueError, match="empty"):
            encoder.encode([])

    def test_a_batch_size_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="batch_size"):
            DenseEncoder(FakeEmbedding(), batch_size=0)

    def test_a_provider_that_answers_too_few_vectors_is_refused(self):
        class ShortAnswer(FakeEmbedding):
            def embed(self, texts, trace=None):
                return super().embed(texts, trace)[:-1]

        encoder = DenseEncoder(ShortAnswer())

        with pytest.raises(RuntimeError, match="vectors"):
            encoder.encode(make_chunks(2))

    def test_a_failing_provider_is_refused(self):
        encoder = DenseEncoder(FakeEmbedding(error=RuntimeError("the provider is down")))

        with pytest.raises(RuntimeError, match="the provider is down"):
            encoder.encode(make_chunks(1))


class TestBatchCount:
    def test_the_batches_are_counted(self):
        encoder = DenseEncoder(FakeEmbedding(), batch_size=2)

        assert encoder.get_batch_count(5) == 3
        assert encoder.get_batch_count(4) == 2
        assert encoder.get_batch_count(2) == 1
        assert encoder.get_batch_count(0) == 0


class TestTrace:
    def test_the_trace_records_what_happened(self):
        encoder = DenseEncoder(FakeEmbedding(), batch_size=2)
        trace = TraceContext()

        encoder.encode(make_chunks(5), trace)

        data = trace.get_stage_data("dense_encoder")
        assert data["total_chunks"] == 5
        assert data["batch_count"] == 3
        assert data["dimension"] == DIMENSION
