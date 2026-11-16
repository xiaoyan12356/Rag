"""Unit tests for the document manager.

The manager is the one place that knows about the four stores a document
lives in, so the tests pin that a document is listed with the right counts,
and that deleting it leaves no trace in any of the four.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import VectorStoreSettings
from ingestion.document_manager import DocumentManager
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore

COLLECTION = "kb_test"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 32


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    """Return a Chroma store that keeps its collection in a temporary folder."""
    config = VectorStoreSettings(
        provider="chroma",
        persist_directory=str(tmp_path / "chroma"),
        collection_name=COLLECTION,
    )
    return ChromaStore(config)


@pytest.fixture
def manager(tmp_path: Path, store: ChromaStore) -> DocumentManager:
    """Return a manager whose four stores all live in a temporary folder."""
    bm25 = BM25Indexer(index_dir=tmp_path / "bm25")
    images = ImageStorage(db_path=tmp_path / "images.db", images_root=tmp_path / "images")
    integrity = SQLiteIntegrityChecker(tmp_path / "history.db")
    return DocumentManager(store, bm25, images, integrity)


def _write_chunks(
    manager: DocumentManager,
    source_hash: str,
    source_path: str,
    chunk_ids: list[str],
) -> None:
    """Put chunks of one document into the vector store and the index."""
    records = [
        {
            "id": chunk_id,
            "vector": [0.1, 0.2, 0.3],
            "metadata": {
                "text": f"the text of {chunk_id}",
                "doc_hash": source_hash,
                "source_path": source_path,
                "chunk_index": index,
                "collection": COLLECTION,
            },
        }
        for index, chunk_id in enumerate(chunk_ids)
    ]
    manager.chroma_store.upsert(records)

    manager.bm25_indexer.add_documents(
        [
            {
                "chunk_id": chunk_id,
                "term_frequencies": {"text": 1},
                "doc_length": 3,
            }
            for chunk_id in chunk_ids
        ],
        collection=COLLECTION,
    )


def _write_images(
    manager: DocumentManager,
    source_hash: str,
    image_ids: list[str],
) -> None:
    """Put the images of one document into the image storage."""
    for image_id in image_ids:
        manager.image_storage.save_image(
            image_id, PNG_BYTES, collection=COLLECTION, doc_hash=source_hash
        )


class TestListDocuments:
    def test_the_ingested_document_is_listed(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa"])

        documents = manager.list_documents()

        assert [document.source_path for document in documents] == ["data/a.pdf"]

    def test_the_document_carries_its_chunk_count(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa", "hash_a_0001_bbbb"])

        assert manager.list_documents()[0].chunk_count == 2

    def test_the_document_carries_its_image_count(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_images(manager, "hash_a", ["img_1", "img_2"])

        assert manager.list_documents()[0].image_count == 2

    def test_a_document_that_was_not_ingested_is_not_listed(self, manager: DocumentManager):
        assert manager.list_documents() == []

    def test_a_failed_ingestion_is_not_listed(self, manager: DocumentManager):
        manager.file_integrity.mark_failed("hash_a", "it failed", "data/a.pdf")

        assert manager.list_documents() == []

    def test_only_the_documents_of_the_collection_are_listed(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa"])

        assert manager.list_documents(collection=COLLECTION) != []
        assert manager.list_documents(collection="other") == []


class TestGetDocumentDetail:
    def test_the_detail_carries_the_ids_of_the_chunks(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa"])

        detail = manager.get_document_detail("hash_a")

        assert detail is not None
        assert detail.chunk_ids == ["hash_a_0000_aaaa"]

    def test_the_detail_carries_the_ids_of_the_images(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_images(manager, "hash_a", ["img_1"])

        detail = manager.get_document_detail("hash_a")

        assert detail is not None
        assert detail.image_ids == ["img_1"]

    def test_a_document_that_is_not_known_has_no_detail(self, manager: DocumentManager):
        assert manager.get_document_detail("hash_a") is None


class TestDeleteDocument:
    def test_the_chunks_are_gone_from_the_vector_store(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa"])

        result = manager.delete_document("data/a.pdf", COLLECTION, source_hash="hash_a")

        assert result.chunks_deleted == 1
        assert manager.chroma_store.collection.get(include=[])["ids"] == []

    def test_the_entries_are_gone_from_the_bm25_index(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa"])

        manager.delete_document("data/a.pdf", COLLECTION, source_hash="hash_a")

        assert manager.bm25_indexer.get_document_count() == 0

    def test_the_images_are_gone_from_the_image_storage(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_images(manager, "hash_a", ["img_1"])

        result = manager.delete_document("data/a.pdf", COLLECTION, source_hash="hash_a")

        assert result.images_deleted == 1
        assert manager.image_storage.list_images(doc_hash="hash_a") == []

    def test_the_record_is_gone_from_the_ingestion_history(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")

        result = manager.delete_document("data/a.pdf", COLLECTION, source_hash="hash_a")

        assert result.integrity_removed is True
        assert manager.file_integrity.list_processed() == []

    def test_the_document_is_not_listed_after_the_deletion(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa"])

        manager.delete_document("data/a.pdf", COLLECTION, source_hash="hash_a")

        assert manager.list_documents() == []

    def test_the_deletion_succeeds_when_every_store_answers(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa"])
        _write_images(manager, "hash_a", ["img_1"])

        result = manager.delete_document("data/a.pdf", COLLECTION, source_hash="hash_a")

        assert result.success is True
        assert result.errors == []

    def test_a_document_that_cannot_be_identified_is_reported(self, manager: DocumentManager):
        result = manager.delete_document("data/missing.pdf", COLLECTION)

        assert result.success is False
        assert result.errors

    def test_the_other_documents_survive_the_deletion(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        manager.file_integrity.mark_success("hash_b", "data/b.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa"])
        _write_chunks(manager, "hash_b", "data/b.pdf", ["hash_b_0000_cccc"])

        manager.delete_document("data/a.pdf", COLLECTION, source_hash="hash_a")

        remaining = manager.list_documents()
        assert [document.source_hash for document in remaining] == ["hash_b"]


class TestCollectionStats:
    def test_the_documents_are_counted(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        manager.file_integrity.mark_success("hash_b", "data/b.pdf")

        assert manager.get_collection_stats().document_count == 2

    def test_the_chunks_are_counted(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_chunks(manager, "hash_a", "data/a.pdf", ["hash_a_0000_aaaa", "hash_a_0001_bbbb"])

        assert manager.get_collection_stats().chunk_count == 2

    def test_the_images_are_counted(self, manager: DocumentManager):
        manager.file_integrity.mark_success("hash_a", "data/a.pdf")
        _write_images(manager, "hash_a", ["img_1", "img_2"])

        assert manager.get_collection_stats().image_count == 2

    def test_an_empty_collection_is_counted_as_empty(self, manager: DocumentManager):
        stats = manager.get_collection_stats()

        assert stats.document_count == 0
        assert stats.chunk_count == 0
        assert stats.image_count == 0
