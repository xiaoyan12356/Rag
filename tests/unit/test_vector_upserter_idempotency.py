"""Unit tests for the vector upserter.

The upserter writes chunks and their vectors into the vector store. The tests
pin that the same chunk always gets the same id, that changed text gets a
new id, and that writing twice leaves one record behind.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings, VectorStoreSettings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.storage.vector_upserter import VectorUpserter
from libs.vector_store.base_vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    """Store that keeps what it is given in memory."""

    def __init__(self) -> None:
        super().__init__(VectorStoreSettings(provider="fake", collection_name="kb"))
        self.records: dict[str, dict[str, Any]] = {}
        self.upsert_calls: list[int] = []
        self.error: Exception | None = None

    def upsert(self, records, trace=None) -> None:
        self.upsert_calls.append(len(records))
        if self.error is not None:
            raise self.error
        for record in records:
            self.records[record["id"]] = dict(record)

    def query(self, vector, top_k=10, filters=None, trace=None):
        return []

    def get_by_ids(self, ids, trace=None):
        return [self.records[i] for i in ids if i in self.records]


def make_chunk(
    text: str = "Some chunk text.",
    chunk_id: str = "chunk_1",
    chunk_index: int = 0,
    source_path: str = "data/documents/sample.pdf",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": source_path, "chunk_index": chunk_index},
        source_ref="doc_1",
    )


def make_upserter(store=None, collection_name: str | None = None) -> VectorUpserter:
    settings = Settings(vector_store=VectorStoreSettings(provider="fake", collection_name="kb"))
    return VectorUpserter(settings, collection_name=collection_name, vector_store=store or FakeVectorStore())


class TestChunkId:
    def test_the_same_chunk_always_gets_the_same_id(self):
        upserter = make_upserter()

        first = upserter.upsert([make_chunk("hello")], [[0.1, 0.2]])
        second = upserter.upsert([make_chunk("hello")], [[0.1, 0.2]])

        assert first == second

    def test_changed_text_gets_a_new_id(self):
        upserter = make_upserter()

        first = upserter.upsert([make_chunk("hello")], [[0.1, 0.2]])
        second = upserter.upsert([make_chunk("goodbye")], [[0.1, 0.2]])

        assert first != second

    def test_a_chunk_from_another_document_gets_another_id(self):
        upserter = make_upserter()

        first = upserter.upsert([make_chunk("hello", source_path="a.pdf")], [[0.1, 0.2]])
        second = upserter.upsert([make_chunk("hello", source_path="b.pdf")], [[0.1, 0.2]])

        assert first != second

    def test_the_id_says_where_the_chunk_came_from(self):
        upserter = make_upserter()

        chunk_id = upserter.upsert([make_chunk("hello", chunk_index=7)], [[0.1, 0.2]])[0]

        assert "_0007_" in chunk_id

    def test_a_chunk_without_a_source_path_is_refused(self):
        upserter = make_upserter()
        chunk = make_chunk("hello")
        del chunk.metadata["source_path"]

        with pytest.raises(ValueError, match="source_path"):
            upserter.upsert([chunk], [[0.1, 0.2]])

    def test_a_chunk_without_a_chunk_index_is_refused(self):
        upserter = make_upserter()
        chunk = make_chunk("hello")
        del chunk.metadata["chunk_index"]

        with pytest.raises(ValueError, match="chunk_index"):
            upserter.upsert([chunk], [[0.1, 0.2]])

    def test_a_chunk_index_that_is_not_a_number_is_refused(self):
        upserter = make_upserter()
        chunk = Chunk(
            id="c1", text="hello", metadata={"source_path": "a.pdf", "chunk_index": "first"}
        )

        with pytest.raises(ValueError, match="chunk_index"):
            upserter.upsert([chunk], [[0.1, 0.2]])


class TestUpsert:
    def test_the_chunks_are_written(self):
        store = FakeVectorStore()
        upserter = make_upserter(store)

        upserter.upsert([make_chunk("hello"), make_chunk("world")], [[0.1, 0.2], [0.3, 0.4]])

        assert len(store.records) == 2

    def test_the_ids_come_back_in_order(self):
        upserter = make_upserter()

        ids = upserter.upsert(
            [make_chunk("hello", chunk_index=0), make_chunk("world", chunk_index=1)],
            [[0.1, 0.2], [0.3, 0.4]],
        )

        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert "_0000_" in ids[0]
        assert "_0001_" in ids[1]

    def test_writing_the_same_chunk_twice_leaves_one_record(self):
        store = FakeVectorStore()
        upserter = make_upserter(store)

        upserter.upsert([make_chunk("hello")], [[0.1, 0.2]])
        upserter.upsert([make_chunk("hello")], [[0.1, 0.2]])

        assert len(store.records) == 1

    def test_the_text_travels_with_the_record(self):
        store = FakeVectorStore()
        upserter = make_upserter(store)

        upserter.upsert([make_chunk("hello")], [[0.1, 0.2]])

        record = next(iter(store.records.values()))
        assert record["metadata"]["text"] == "hello"

    def test_the_metadata_of_the_chunk_is_kept(self):
        store = FakeVectorStore()
        upserter = make_upserter(store)

        upserter.upsert([make_chunk("hello")], [[0.1, 0.2]])

        record = next(iter(store.records.values()))
        assert record["metadata"]["source_path"] == "data/documents/sample.pdf"

    def test_the_vector_is_written_as_it_is(self):
        store = FakeVectorStore()
        upserter = make_upserter(store)

        upserter.upsert([make_chunk("hello")], [[0.1, 0.2]])

        record = next(iter(store.records.values()))
        assert record["vector"] == [0.1, 0.2]

    def test_no_chunks_gives_nothing(self):
        upserter = make_upserter()

        with pytest.raises(ValueError, match="empty"):
            upserter.upsert([], [])

    def test_a_number_of_vectors_that_does_not_match_is_refused(self):
        upserter = make_upserter()

        with pytest.raises(ValueError, match="must be the same"):
            upserter.upsert([make_chunk("hello")], [[0.1], [0.2]])

    def test_a_store_that_fails_is_reported(self):
        store = FakeVectorStore()
        store.error = RuntimeError("the database is down")
        upserter = make_upserter(store)

        with pytest.raises(RuntimeError, match="the database is down"):
            upserter.upsert([make_chunk("hello")], [[0.1, 0.2]])


class TestBatch:
    def test_several_batches_are_written_in_one_go(self):
        store = FakeVectorStore()
        upserter = make_upserter(store)

        upserter.upsert_batch(
            [
                ([make_chunk("hello", chunk_index=0)], [[0.1, 0.2]]),
                ([make_chunk("world", chunk_index=1)], [[0.3, 0.4]]),
            ]
        )

        assert len(store.records) == 2

    def test_the_ids_of_every_batch_come_back_in_order(self):
        upserter = make_upserter()

        ids = upserter.upsert_batch(
            [
                ([make_chunk("hello", chunk_index=0)], [[0.1, 0.2]]),
                ([make_chunk("world", chunk_index=1)], [[0.3, 0.4]]),
            ]
        )

        assert "_0000_" in ids[0]
        assert "_0001_" in ids[1]


class TestCollection:
    def test_the_collection_of_the_settings_is_used(self):
        settings = Settings(vector_store=VectorStoreSettings(provider="fake", collection_name="kb"))
        upserter = VectorUpserter(settings, vector_store=FakeVectorStore())

        assert upserter.collection_name == "kb"

    def test_a_given_collection_wins(self):
        settings = Settings(vector_store=VectorStoreSettings(provider="fake", collection_name="kb"))
        upserter = VectorUpserter(settings, collection_name="other", vector_store=FakeVectorStore())

        assert upserter.collection_name == "other"


class TestTrace:
    def test_the_trace_records_what_happened(self):
        upserter = make_upserter()
        trace = TraceContext()

        upserter.upsert([make_chunk("hello"), make_chunk("world")], [[0.1], [0.2]], trace)

        data = trace.get_stage_data("vector_upserter")
        assert data["upserted"] == 2
        assert data["collection"] == "kb"
