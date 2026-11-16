"""Data service for the dashboard.

The pages that show what was ingested need the documents, their chunks and
their images. They get them from here, so the pages do not have to know how
the stores are built or where they live.

The stores are built when they are first needed and then kept, because
opening a database is slow and a page is redrawn on every interaction.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from core.settings import DEFAULT_SETTINGS_PATH, Settings, load_settings, resolve_path
from ingestion.document_manager import CollectionStats, DocumentManager
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "default"


class DataService:
    """Give the pages read access to the ingested documents and their chunks.

    Args:
        settings: The configuration that says where the stores are; the
            default configuration is used when this is left out.
        settings_path: The configuration file to read, when the settings
            themselves are not handed in.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        settings_path: str | Path | None = None,
    ) -> None:
        """Set up the service without opening any store yet."""
        self._settings = settings
        self._settings_path = settings_path
        self._manager: DocumentManager | None = None
        self._vector_store: BaseVectorStore | None = None
        self._vector_store_collection: str = ""
        self._image_storage: ImageStorage | None = None
        self._manager_collection: str = ""

    @property
    def settings(self) -> Settings:
        """Return the configuration, reading the file the first time."""
        if self._settings is None:
            self._settings = load_settings(self._settings_path or DEFAULT_SETTINGS_PATH)
        return self._settings

    def list_collections(self) -> list[str]:
        """Return the names of the collections the vector store holds.

        The client is opened directly, because opening a collection would
        create it, and listing must not leave anything behind.
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            persist_directory = resolve_path(
                self.settings.vector_store.persist_directory or "./data/db/chroma"
            )
            client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            names = [str(collection.name) for collection in client.list_collections()]
        except Exception as exc:
            logger.warning("Could not list the collections: %s", exc)
            return []
        return sorted(names)

    def list_documents(self, collection: str | None = None) -> list[dict[str, Any]]:
        """Return the ingested documents, newest first.

        Args:
            collection: The collection to look at; every collection is looked
                at when this is left out.

        Returns:
            One dictionary per document, with the fields of `DocumentInfo`.
        """
        manager = self._get_manager(collection)
        return [asdict(document) for document in manager.list_documents(collection)]

    def get_document_detail(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> dict[str, Any] | None:
        """Return everything known about one document, or None."""
        manager = self._get_manager(collection)
        detail = manager.get_document_detail(doc_id)
        return asdict(detail) if detail is not None else None

    def get_chunks(
        self,
        source_hash: str,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the chunks of one document.

        Args:
            source_hash: The hash of the file the document came from.
            collection: The collection the document belongs to.

        Returns:
            One dictionary per chunk, with the keys `id`, `text` and
            `metadata`.
        """
        if not source_hash:
            return []
        store = self._get_vector_store(collection or DEFAULT_COLLECTION)
        try:
            results = store.collection.get(
                where={"doc_hash": source_hash},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("Could not read the chunks of %s: %s", source_hash, exc)
            return []

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        chunks: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            text = documents[index] if index < len(documents) else ""
            chunks.append(
                {
                    "id": str(chunk_id),
                    "text": text or str((metadata or {}).get("text", "")),
                    "metadata": dict(metadata or {}),
                }
            )
        return chunks

    def get_images(
        self,
        source_hash: str,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the images of one document."""
        if not source_hash:
            return []
        storage = self._get_image_storage()
        try:
            images = storage.list_images(doc_hash=source_hash)
        except Exception as exc:
            logger.warning("Could not read the images of %s: %s", source_hash, exc)
            return []
        if collection is not None:
            images = [
                image for image in images if image.get("collection") in (None, collection)
            ]
        return images

    def get_image_path(self, image_id: str) -> str | None:
        """Return where an image is, or None when it is not known."""
        return self._get_image_storage().get_image_path(image_id)

    def delete_document(
        self,
        source_path: str,
        collection: str | None = None,
        source_hash: str | None = None,
    ) -> Any:
        """Delete a document from all four stores."""
        manager = self._get_manager(collection)
        return manager.delete_document(
            source_path,
            collection or DEFAULT_COLLECTION,
            source_hash=source_hash,
        )

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """Return how much the collection holds."""
        return self._get_manager(collection).get_collection_stats(collection)

    def _get_manager(self, collection: str | None) -> DocumentManager:
        """Return the manager, built for `collection` when it changed."""
        target = collection or self.settings.vector_store.collection_name
        if self._manager is not None and self._manager_collection == target:
            return self._manager

        settings = self.settings
        vector_store = self._get_vector_store(target)
        bm25 = BM25Indexer(index_dir=settings.ingestion.bm25_index_dir)
        images = self._get_image_storage()
        integrity = SQLiteIntegrityChecker(settings.ingestion.history_db)

        self._manager = DocumentManager(vector_store, bm25, images, integrity)
        self._manager_collection = target
        return self._manager

    def _get_vector_store(self, collection: str) -> BaseVectorStore:
        """Return the vector store of `collection`."""
        if self._vector_store is not None and self._vector_store_collection == collection:
            return self._vector_store

        settings = self.settings
        settings = replace(
            settings,
            vector_store=replace(settings.vector_store, collection_name=collection),
        )
        store = VectorStoreFactory.create(settings)
        self._vector_store = store
        self._vector_store_collection = collection
        return store

    def _get_image_storage(self) -> ImageStorage:
        """Return the image storage."""
        if self._image_storage is None:
            settings = self.settings
            self._image_storage = ImageStorage(
                db_path=settings.ingestion.image_index_db,
                images_root=settings.ingestion.images_root,
            )
        return self._image_storage
