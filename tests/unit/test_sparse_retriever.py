"""Unit tests for the sparse retriever.

The sparse route looks the words of the question up in the BM25 index and
reads the texts of the matching chunks back from the vector store. The tests
pin that the index is asked with the words, that the texts are read back for
the chunk ids the index named, and that the results carry both.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import (
    EmbeddingSettings,
    IngestionSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    """Store that answers with the records it was given."""

    def __init__(self) -> None:
        super().__init__(VectorStoreSettings(provider="fake", collection_name="kb"))
        self.records: list[dict[str, Any]] = []
        self.calls: list[list[str]] = []

    def upsert(self, records, trace=None) -> None:
        self.records = [dict(record) for record in records]

    def query(self, vector, top_k=10, filters=None, trace=None):
        return []

    def get_by_ids(self, ids, trace=None):
        wanted = set(ids)
        self.calls.append(list(ids))
        return [record for record in self.records if record["id"] in wanted]

    def delete_by_metadata(self, filters):
        self.validate_delete_filters(filters)
        keep = [
            record
            for record in self.records
            if not all(record.get("metadata", {}).get(k) == v for k, v in filters.items())
        ]
        removed = len(self.records) - len(keep)
        self.records = keep
        return removed

    def list_collections(self):
        return [{"name": self.collection_name, "chunk_count": len(self.records)}]


class FakeIndexer(BM25Indexer):
    """Indexer that answers with the matches it was given."""

    def __init__(self, matches: list[dict[str, Any]] | None = None) -> None:
        super().__init__(index_dir="data/db/bm25")
        self.matches = matches or []
        self.calls: list[tuple[list[str], int]] = []
        self.loads: list[str] = []

    def load(self, collection="default", trace=None) -> bool:
        self.loads.append(collection)
        return True

    def query(self, query_terms, top_k=10, trace=None):
        self.calls.append((list(query_terms), top_k))
        return [dict(match) for match in self.matches[:top_k]]


def make_settings(sparse_top_k: int | None = 20) -> Settings:
    return Settings(
        embedding=EmbeddingSettings(provider="fake", model="fake", dimensions=4),
        vector_store=VectorStoreSettings(provider="fake", collection_name="kb"),
        retrieval=RetrievalSettings(sparse_top_k=sparse_top_k),
        ingestion=IngestionSettings(bm25_index_dir="data/db/bm25"),
    )


def make_record(record_id: str, text: str = "some text") -> dict[str, Any]:
    return {
        "id": record_id,
        "text": text,
        "metadata": {"source_path": "a.pdf", "page": 3},
    }


class TestRetrieve:
    def test_the_index_is_asked_with_the_words(self):
        indexer = FakeIndexer()
        retriever = SparseRetriever(make_settings(), indexer, FakeVectorStore())

        retriever.retrieve(["azure", "openai"])

        assert indexer.calls[0][0] == ["azure", "openai"]

    def test_the_index_is_asked_for_the_right_number_of_results(self):
        indexer = FakeIndexer()
        retriever = SparseRetriever(make_settings(), indexer, FakeVectorStore())

        retriever.retrieve(["azure"], top_k=3)

        assert indexer.calls[0][1] == 3

    def test_the_configured_number_of_results_is_used(self):
        indexer = FakeIndexer()
        retriever = SparseRetriever(make_settings(sparse_top_k=7), indexer, FakeVectorStore())

        retriever.retrieve(["azure"])

        assert indexer.calls[0][1] == 7

    def test_the_texts_are_read_back_for_the_matched_chunks(self):
        indexer = FakeIndexer([{"chunk_id": "chunk_1", "score": 1.5}])
        store = FakeVectorStore()
        retriever = SparseRetriever(make_settings(), indexer, store)

        retriever.retrieve(["azure"])

        assert store.calls[0] == ["chunk_1"]

    def test_every_match_becomes_a_result(self):
        indexer = FakeIndexer(
            [{"chunk_id": "a", "score": 1.0}, {"chunk_id": "b", "score": 0.5}]
        )
        store = FakeVectorStore()
        store.records = [make_record("a"), make_record("b")]
        retriever = SparseRetriever(make_settings(), indexer, store)

        results = retriever.retrieve(["azure"])

        assert [result.chunk_id for result in results] == ["a", "b"]

    def test_the_result_carries_the_score_of_the_index(self):
        indexer = FakeIndexer([{"chunk_id": "a", "score": 2.25}])
        store = FakeVectorStore()
        store.records = [make_record("a")]
        retriever = SparseRetriever(make_settings(), indexer, store)

        results = retriever.retrieve(["azure"])

        assert results[0].score == pytest.approx(2.25)
        assert results[0].sparse_score == pytest.approx(2.25)

    def test_the_result_carries_the_text(self):
        indexer = FakeIndexer([{"chunk_id": "a", "score": 1.0}])
        store = FakeVectorStore()
        store.records = [make_record("a", text="the text of the chunk")]
        retriever = SparseRetriever(make_settings(), indexer, store)

        results = retriever.retrieve(["azure"])

        assert results[0].text == "the text of the chunk"

    def test_the_result_carries_the_metadata(self):
        indexer = FakeIndexer([{"chunk_id": "a", "score": 1.0}])
        store = FakeVectorStore()
        store.records = [make_record("a")]
        retriever = SparseRetriever(make_settings(), indexer, store)

        results = retriever.retrieve(["azure"])

        assert results[0].metadata["source_path"] == "a.pdf"
        assert results[0].metadata["page"] == 3

    def test_a_match_the_store_does_not_hold_is_left_out(self):
        indexer = FakeIndexer([{"chunk_id": "a", "score": 1.0}, {"chunk_id": "gone", "score": 0.5}])
        store = FakeVectorStore()
        store.records = [make_record("a")]
        retriever = SparseRetriever(make_settings(), indexer, store)

        results = retriever.retrieve(["azure"])

        assert [result.chunk_id for result in results] == ["a"]

    def test_no_match_gives_no_result(self):
        retriever = SparseRetriever(make_settings(), FakeIndexer(), FakeVectorStore())

        assert retriever.retrieve(["azure"]) == []

    def test_the_index_is_read_only_once(self):
        indexer = FakeIndexer([{"chunk_id": "a", "score": 1.0}])
        store = FakeVectorStore()
        store.records = [make_record("a")]
        retriever = SparseRetriever(make_settings(), indexer, store)

        retriever.retrieve(["azure"])
        retriever.retrieve(["openai"])

        assert indexer.loads == ["kb"]


class TestSetup:
    def test_the_collection_of_the_settings_is_used(self):
        retriever = SparseRetriever(
            make_settings(), FakeIndexer(), FakeVectorStore(), collection="other"
        )

        assert retriever.collection == "other"

    def test_the_index_is_read_for_the_collection(self):
        indexer = FakeIndexer()
        retriever = SparseRetriever(
            make_settings(), indexer, FakeVectorStore(), collection="other"
        )

        retriever.retrieve(["azure"])

        assert indexer.loads == ["other"]


class TestFailures:
    def test_words_that_are_empty_are_refused(self):
        retriever = SparseRetriever(make_settings(), FakeIndexer(), FakeVectorStore())

        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve([])

    def test_words_that_are_not_a_list_are_refused(self):
        retriever = SparseRetriever(make_settings(), FakeIndexer(), FakeVectorStore())

        with pytest.raises(ValueError, match="list"):
            retriever.retrieve("azure")

    def test_a_top_k_of_zero_is_refused(self):
        retriever = SparseRetriever(make_settings(), FakeIndexer(), FakeVectorStore())

        with pytest.raises(ValueError, match="top_k"):
            retriever.retrieve(["azure"], top_k=0)


class TestTrace:
    def test_the_trace_records_what_happened(self):
        indexer = FakeIndexer([{"chunk_id": "a", "score": 1.0}])
        store = FakeVectorStore()
        store.records = [make_record("a")]
        retriever = SparseRetriever(make_settings(), indexer, store)
        trace = TraceContext()

        retriever.retrieve(["azure"], trace=trace)

        data = trace.get_stage_data("sparse_retrieval")
        assert data["returned"] == 1
        assert data["chunk_ids"] == ["a"]
        assert data["keywords"] == ["azure"]
