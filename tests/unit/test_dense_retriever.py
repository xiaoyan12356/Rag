"""Unit tests for the dense retriever.

The dense route turns the question into a vector and asks the vector store
for the chunks closest to it. The tests pin that the two are called in the
right order and that the matches come back as results that carry the text.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.query_engine.dense_retriever import DenseRetriever
from core.settings import EmbeddingSettings, RetrievalSettings, Settings, VectorStoreSettings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.vector_store.base_vector_store import BaseVectorStore

DIMENSION = 4


class FakeEmbedding(BaseEmbedding):
    """Provider that answers with one vector per text."""

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__(
            EmbeddingSettings(provider="fake", model="fake", dimensions=DIMENSION)
        )
        self.error = error
        self.calls: list[list[str]] = []

    def embed(self, texts, trace=None) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[float(len(text))] * DIMENSION for text in texts]


class FakeVectorStore(BaseVectorStore):
    """Store that answers with the records it was given."""

    def __init__(self) -> None:
        super().__init__(VectorStoreSettings(provider="fake", collection_name="kb"))
        self.records: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def upsert(self, records, trace=None) -> None:
        self.records = [dict(record) for record in records]

    def query(self, vector, top_k=10, filters=None, trace=None):
        self.calls.append(
            {"vector": list(vector), "top_k": top_k, "filters": filters}
        )
        return self.records[:top_k]

    def get_by_ids(self, ids, trace=None):
        return [record for record in self.records if record["id"] in set(ids)]


def make_settings(dense_top_k: int | None = 20) -> Settings:
    return Settings(
        embedding=EmbeddingSettings(provider="fake", model="fake", dimensions=DIMENSION),
        vector_store=VectorStoreSettings(provider="fake", collection_name="kb"),
        retrieval=RetrievalSettings(dense_top_k=dense_top_k),
    )


def make_match(record_id: str, score: float = 0.8, text: str = "some text") -> dict[str, Any]:
    return {
        "id": record_id,
        "score": score,
        "text": text,
        "metadata": {"source_path": "a.pdf", "text": text},
    }


class TestRetrieve:
    def test_the_question_is_turned_into_a_vector(self):
        embedding = FakeEmbedding()
        store = FakeVectorStore()
        retriever = DenseRetriever(make_settings(), embedding, store)

        retriever.retrieve("how to configure azure")

        assert embedding.calls == [["how to configure azure"]]

    def test_the_store_is_asked_with_that_vector(self):
        embedding = FakeEmbedding()
        store = FakeVectorStore()
        retriever = DenseRetriever(make_settings(), embedding, store)

        retriever.retrieve("hello")

        assert store.calls[0]["vector"] == [5.0] * DIMENSION

    def test_the_store_is_asked_for_the_right_number_of_results(self):
        store = FakeVectorStore()
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), store)

        retriever.retrieve("hello", top_k=3)

        assert store.calls[0]["top_k"] == 3

    def test_the_configured_number_of_results_is_used(self):
        store = FakeVectorStore()
        retriever = DenseRetriever(make_settings(dense_top_k=7), FakeEmbedding(), store)

        retriever.retrieve("hello")

        assert store.calls[0]["top_k"] == 7

    def test_the_filters_are_handed_to_the_store(self):
        store = FakeVectorStore()
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), store)

        retriever.retrieve("hello", filters={"doc_type": "pdf"})

        assert store.calls[0]["filters"] == {"doc_type": "pdf"}

    def test_every_match_becomes_a_result(self):
        store = FakeVectorStore()
        store.records = [make_match("a"), make_match("b")]
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), store)

        results = retriever.retrieve("hello")

        assert len(results) == 2
        assert all(isinstance(result, RetrievalResult) for result in results)

    def test_the_result_names_the_chunk(self):
        store = FakeVectorStore()
        store.records = [make_match("chunk_7")]
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), store)

        results = retriever.retrieve("hello")

        assert results[0].chunk_id == "chunk_7"

    def test_the_result_carries_the_score(self):
        store = FakeVectorStore()
        store.records = [make_match("a", score=0.42)]
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), store)

        results = retriever.retrieve("hello")

        assert results[0].score == pytest.approx(0.42)
        assert results[0].dense_score == pytest.approx(0.42)

    def test_the_result_carries_the_text(self):
        store = FakeVectorStore()
        store.records = [make_match("a", text="the text of the chunk")]
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), store)

        results = retriever.retrieve("hello")

        assert results[0].text == "the text of the chunk"

    def test_the_result_carries_the_metadata(self):
        store = FakeVectorStore()
        store.records = [make_match("a")]
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), store)

        results = retriever.retrieve("hello")

        assert results[0].metadata["source_path"] == "a.pdf"

    def test_the_text_is_taken_from_the_metadata_when_the_match_has_none(self):
        store = FakeVectorStore()
        store.records = [
            {"id": "a", "score": 0.5, "metadata": {"text": "from the metadata"}}
        ]
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), store)

        results = retriever.retrieve("hello")

        assert results[0].text == "from the metadata"

    def test_no_match_gives_no_result(self):
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), FakeVectorStore())

        assert retriever.retrieve("hello") == []


class TestSetup:
    def test_the_collection_of_the_settings_is_used(self):
        retriever = DenseRetriever(
            make_settings(), FakeEmbedding(), FakeVectorStore(), collection="other"
        )

        assert retriever.collection == "other"


class TestFailures:
    def test_a_question_that_is_empty_is_refused(self):
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), FakeVectorStore())

        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("   ")

    def test_a_question_that_is_not_text_is_refused(self):
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), FakeVectorStore())

        with pytest.raises(ValueError, match="string"):
            retriever.retrieve(None)

    def test_a_top_k_of_zero_is_refused(self):
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), FakeVectorStore())

        with pytest.raises(ValueError, match="top_k"):
            retriever.retrieve("hello", top_k=0)

    def test_a_provider_that_answers_nothing_is_reported(self):
        class EmptyAnswer(FakeEmbedding):
            def embed(self, texts, trace=None):
                return []

        retriever = DenseRetriever(make_settings(), EmptyAnswer(), FakeVectorStore())

        with pytest.raises(RuntimeError, match="no vector"):
            retriever.retrieve("hello")

    def test_a_failing_provider_is_reported(self):
        retriever = DenseRetriever(
            make_settings(), FakeEmbedding(error=RuntimeError("the provider is down")), FakeVectorStore()
        )

        with pytest.raises(RuntimeError, match="the provider is down"):
            retriever.retrieve("hello")


class TestTrace:
    def test_the_trace_records_what_happened(self):
        store = FakeVectorStore()
        store.records = [make_match("a")]
        retriever = DenseRetriever(make_settings(), FakeEmbedding(), store)
        trace = TraceContext()

        retriever.retrieve("hello", trace=trace)

        data = trace.get_stage_data("dense_retrieval")
        assert data["returned"] == 1
        assert data["chunk_ids"] == ["a"]
