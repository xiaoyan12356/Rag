"""Cross-store document lifecycle management.

An ingested document lives in four places at once: its chunks are in the
vector store, its terms are in the BM25 index, its images are in the image
storage, and its hash is in the ingestion history. This module is the one
place that knows about all four, so a document can be listed, inspected and
deleted without the caller having to reach into each store by hand.

The stores are told apart by the hash of the file they came from, which is
the one value all of them keep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "default"


@dataclass
class DocumentInfo:
    """What the manager knows about one ingested document."""

    source_path: str
    source_hash: str
    collection: str = DEFAULT_COLLECTION
    chunk_count: int = 0
    image_count: int = 0
    processed_at: str | None = None


@dataclass
class DocumentDetail(DocumentInfo):
    """A document with the ids of its chunks and its images."""

    chunk_ids: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)


@dataclass
class DeleteResult:
    """What a delete did, in every store it touched."""

    success: bool = True
    source_hash: str = ""
    chunks_deleted: int = 0
    bm25_removed: bool = False
    images_deleted: int = 0
    integrity_removed: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    """How much a collection holds."""

    collection: str = DEFAULT_COLLECTION
    document_count: int = 0
    chunk_count: int = 0
    image_count: int = 0


class DocumentManager:
    """List, inspect and delete documents across the four stores.

    Args:
        chroma_store: The store that holds the chunks and their vectors.
        bm25_indexer: The index that holds the terms of the chunks.
        image_storage: The storage that holds the images of the documents.
        file_integrity: The checker that holds the ingestion history.
    """

    def __init__(
        self,
        chroma_store: Any,
        bm25_indexer: Any,
        image_storage: Any,
        file_integrity: Any,
    ) -> None:
        """Keep hold of the four stores the documents live in."""
        self.chroma_store = chroma_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        """Return the documents that were ingested.

        Args:
            collection: The collection to look at; every collection is looked
                at when this is left out.

        Returns:
            One entry per ingested document, newest first.
        """
        documents: list[DocumentInfo] = []
        for record in self.file_integrity.list_processed():
            source_hash = str(record.get("file_hash", ""))
            source_path = str(record.get("file_path", ""))
            chunk_ids = self._chunk_ids(source_hash)
            image_ids = self._image_ids(source_hash)

            if collection is not None and not self._belongs_to_collection(
                source_hash, chunk_ids, collection
            ):
                continue

            documents.append(
                DocumentInfo(
                    source_path=source_path,
                    source_hash=source_hash,
                    collection=collection or DEFAULT_COLLECTION,
                    chunk_count=len(chunk_ids),
                    image_count=len(image_ids),
                    processed_at=record.get("processed_at"),
                )
            )
        return documents

    def get_document_detail(self, doc_id: str) -> DocumentDetail | None:
        """Return everything the manager knows about one document.

        Args:
            doc_id: The hash of the file the document came from.

        Returns:
            The detail of the document, or None when it is not known.
        """
        record = self._record_for_hash(doc_id)
        if record is None:
            return None

        chunk_ids = self._chunk_ids(doc_id)
        image_ids = self._image_ids(doc_id)
        return DocumentDetail(
            source_path=str(record.get("file_path", "")),
            source_hash=doc_id,
            collection=DEFAULT_COLLECTION,
            chunk_count=len(chunk_ids),
            image_count=len(image_ids),
            processed_at=record.get("processed_at"),
            chunk_ids=chunk_ids,
            image_ids=image_ids,
        )

    def delete_document(
        self,
        source_path: str,
        collection: str = DEFAULT_COLLECTION,
        source_hash: str | None = None,
    ) -> DeleteResult:
        """Remove a document from all four stores.

        The document is found by its hash: the one that is handed in is used,
        otherwise the file is hashed, and when the file is gone the ingestion
        history is asked which hash belonged to that path.

        A store that fails is reported in the result and does not stop the
        other stores, because a document that is only half deleted is worse
        than one whose leftovers are named.

        Args:
            source_path: The path the document was ingested from.
            collection: The collection the document belongs to.
            source_hash: The hash of the file, when the caller knows it.

        Returns:
            What was removed, and what failed.
        """
        result = DeleteResult(source_hash=source_hash or "")

        resolved_hash = source_hash or self._resolve_hash(source_path)
        if not resolved_hash:
            result.success = False
            result.errors.append(
                f"Cannot identify the document at '{source_path}': "
                "its hash is unknown"
            )
            return result
        result.source_hash = resolved_hash

        chunk_ids = self._chunk_ids(resolved_hash)

        try:
            result.chunks_deleted = self.chroma_store.delete_by_metadata(
                {"doc_hash": resolved_hash}
            )
        except Exception as exc:
            logger.warning("Could not delete the chunks of %s: %s", source_path, exc)
            result.errors.append(f"The vector store failed: {exc}")

        try:
            result.bm25_removed = self.bm25_indexer.remove_document(
                resolved_hash, collection, chunk_ids=chunk_ids
            )
        except Exception as exc:
            logger.warning("Could not delete the index entries of %s: %s", source_path, exc)
            result.errors.append(f"The BM25 index failed: {exc}")

        try:
            result.images_deleted = self._delete_images(resolved_hash)
        except Exception as exc:
            logger.warning("Could not delete the images of %s: %s", source_path, exc)
            result.errors.append(f"The image storage failed: {exc}")

        try:
            result.integrity_removed = self.file_integrity.remove_record(resolved_hash)
        except Exception as exc:
            logger.warning("Could not forget the history of %s: %s", source_path, exc)
            result.errors.append(f"The ingestion history failed: {exc}")

        if result.errors:
            result.success = False
        return result

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """Return how much a collection holds.

        Args:
            collection: The collection to count; every collection is counted
                when this is left out.

        Returns:
            The number of documents, chunks and images.
        """
        documents = self.list_documents(collection)
        return CollectionStats(
            collection=collection or DEFAULT_COLLECTION,
            document_count=len(documents),
            chunk_count=sum(document.chunk_count for document in documents),
            image_count=sum(document.image_count for document in documents),
        )

    def _resolve_hash(self, source_path: str) -> str:
        """Return the hash of the document at `source_path`.

        The file is hashed when it is still there; when it is gone the
        ingestion history is asked which hash belonged to that path.
        """
        try:
            return self.file_integrity.compute_sha256(source_path)
        except Exception:
            pass

        for record in self.file_integrity.list_processed():
            if str(record.get("file_path", "")) == str(source_path):
                return str(record.get("file_hash", ""))
        return ""

    def _record_for_hash(self, doc_id: str) -> dict[str, Any] | None:
        """Return the history record of the document with this hash."""
        for record in self.file_integrity.list_processed():
            if str(record.get("file_hash", "")) == doc_id:
                return record
        return None

    def _chunk_ids(self, source_hash: str) -> list[str]:
        """Return the ids of the chunks that came from the document."""
        if not source_hash:
            return []
        try:
            results = self.chroma_store.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
        except Exception as exc:
            logger.warning("Could not read the chunks of %s: %s", source_hash, exc)
            return []
        return [str(record_id) for record_id in (results.get("ids") or [])]

    def _image_ids(self, source_hash: str) -> list[str]:
        """Return the ids of the images that came from the document."""
        if not source_hash:
            return []
        try:
            images = self.image_storage.list_images(doc_hash=source_hash)
        except Exception as exc:
            logger.warning("Could not read the images of %s: %s", source_hash, exc)
            return []
        return [str(image.get("image_id", "")) for image in images]

    def _delete_images(self, source_hash: str) -> int:
        """Delete every image of the document and return how many went."""
        deleted = 0
        for image in self.image_storage.list_images(doc_hash=source_hash):
            if self.image_storage.delete_image(str(image.get("image_id", ""))):
                deleted += 1
        return deleted

    def _belongs_to_collection(
        self,
        source_hash: str,
        chunk_ids: list[str],
        collection: str,
    ) -> bool:
        """Say whether the document is part of `collection`.

        The ingestion history does not know about collections, so the chunks
        are asked. A document that has no chunks left is kept, because there
        is nothing left that could say where it belonged.
        """
        if not chunk_ids:
            return True
        try:
            results = self.chroma_store.collection.get(
                ids=chunk_ids, include=["metadatas"]
            )
        except Exception:
            return True
        for metadata in results.get("metadatas") or []:
            if metadata and metadata.get("collection") == collection:
                return True
        return False
