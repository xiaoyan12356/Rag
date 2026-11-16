"""ChromaDB vector store.

Chroma is the default local backend: it persists to a directory on disk and
ranks with cosine distance, which this store converts back into a similarity
score that grows with closeness.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.settings import VectorStoreSettings, resolve_path
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """ChromaDB implementation of the vector store contract."""

    def __init__(self, config: VectorStoreSettings) -> None:
        super().__init__(config)

        self.persist_directory = resolve_path(
            config.persist_directory or "./data/db/chroma"
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to open the Chroma collection '{self.collection_name}' at "
                f"{self.persist_directory}: {type(exc).__name__}: {exc}"
            ) from exc

    def upsert(
        self,
        records: Sequence[Mapping[str, Any]],
        trace: TraceContext | None = None,
    ) -> None:
        """Insert or replace `records`."""
        self.validate_records(records)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for record in records:
            metadata = record.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise VectorStoreError(
                    f"record '{record['id']}' carries metadata of type "
                    f"{type(metadata).__name__}. Expected a mapping."
                )
            sanitized = _sanitize_metadata(metadata)
            ids.append(str(record["id"]))
            embeddings.append([float(value) for value in record["vector"]])
            metadatas.append(sanitized or {"_placeholder": "true"})
            documents.append(str(metadata.get("text", record["id"])))

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert {len(records)} records into the Chroma collection "
                f"'{self.collection_name}': {type(exc).__name__}: {exc}"
            ) from exc

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return the `top_k` records closest to `vector`, closest first."""
        self.validate_query_vector(vector, top_k)

        try:
            results = self.collection.query(
                query_embeddings=[[float(value) for value in vector]],
                n_results=top_k,
                where=_build_where_clause(filters) if filters else None,
                include=["metadatas", "distances", "documents"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to query the Chroma collection '{self.collection_name}' "
                f"with top_k={top_k}: {type(exc).__name__}: {exc}"
            ) from exc

        ids = _first(results.get("ids"))
        if not ids:
            return []

        distances = _first(results.get("distances"))
        metadatas = _first(results.get("metadatas"))
        documents = _first(results.get("documents"))

        matches: list[dict[str, Any]] = []
        for index, record_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 0.0
            metadata = metadatas[index] if index < len(metadatas) else {}
            matches.append(
                {
                    "id": record_id,
                    "score": max(0.0, 1.0 - (distance / 2.0)),
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": metadata or {},
                }
            )
        return matches

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return the records with the given ids, skipping the unknown ones."""
        wanted = [str(record_id) for record_id in ids]
        if not wanted:
            return []

        try:
            results = self.collection.get(ids=wanted, include=["metadatas", "documents"])
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to read {len(wanted)} records from the Chroma collection "
                f"'{self.collection_name}': {type(exc).__name__}: {exc}"
            ) from exc

        found_ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []
        documents = results.get("documents") or []

        records: list[dict[str, Any]] = []
        for index, record_id in enumerate(found_ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            records.append(
                {
                    "id": record_id,
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": metadata or {},
                }
            )
        return records

    def delete_by_metadata(self, filters: Mapping[str, Any]) -> int:
        """Delete every record whose metadata matches `filters`.

        Args:
            filters: The metadata fields the records have to match.

        Returns:
            The number of records that were deleted.
        """
        self.validate_delete_filters(filters)

        try:
            existing = self.collection.get(
                where=_build_where_clause(filters), include=[]
            )
            ids = [str(record_id) for record_id in (existing.get("ids") or [])]
            if not ids:
                return 0
            self.collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete the records matching {dict(filters)} from the "
                f"Chroma collection '{self.collection_name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return len(ids)

    def list_collections(self) -> list[dict[str, Any]]:
        """Return the collections the store holds and how much is in them.

        Args:
            None.

        Returns:
            One entry per collection, with its name and how many chunks it
            holds.
        """
        names = {self.collection_name}
        try:
            for collection in self.client.list_collections():
                name = getattr(collection, "name", None)
                if name:
                    names.add(str(name))
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to list the collections of the Chroma store at "
                f"{self.persist_directory}: {type(exc).__name__}: {exc}"
            ) from exc

        listed: list[dict[str, Any]] = []
        for name in sorted(names):
            listed.append({"name": name, "chunk_count": self._count(name)})
        return listed

    def _count(self, name: str) -> int:
        """Return how many records the collection with this name holds."""
        try:
            return int(self.client.get_collection(name).count())
        except Exception as exc:
            logger.warning("Could not count the chunks of '%s': %s", name, exc)
            return 0


def _first(values: Any) -> list[Any]:
    """Chroma answers with one list per query embedding; keep the first."""
    if not isinstance(values, list) or not values:
        return []
    first = values[0]
    return first if isinstance(first, list) else []


def _sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce the metadata to the value types Chroma is able to store."""
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[str(key)] = value
        elif isinstance(value, (list, tuple)):
            sanitized[str(key)] = ",".join(str(item) for item in value)
        elif value is not None:
            sanitized[str(key)] = str(value)
    return sanitized


def _build_where_clause(filters: Mapping[str, Any]) -> dict[str, Any]:
    """Build a Chroma where clause that matches every filter by equality."""
    conditions = [{str(key): value} for key, value in filters.items()]
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
