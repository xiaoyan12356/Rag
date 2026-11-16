"""Vector storage.

The dense encoder gives every chunk a vector. This module writes those
vectors into the vector store, giving every chunk an id that is the same
every time the same chunk is written.

The id is built from where the chunk came from and what it says: the path of
the document, the place of the chunk in it, and the text itself. Write the
same chunk twice and it lands in the same place instead of twice; change the
text and it lands somewhere new, next to the old version.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)

ID_HASH_LENGTH = 8


class VectorUpserter:
    """Write chunks and their vectors into the vector store.

    The id of a chunk is `{source_hash}_{chunk_index:04d}_{content_hash}`,
    where the two hashes are the first eight characters of the SHA256 of the
    source path and of the text of the chunk.
    """

    def __init__(
        self,
        settings: Settings,
        collection_name: str | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        """Set up the upserter.

        Args:
            settings: The settings that say which vector store to use.
            collection_name: The collection to write to; the one named in the
                settings is used when this is left out.
            vector_store: A store to write to; one is built from the settings
                when this is left out.
        """
        self.settings = settings
        if collection_name:
            settings = replace(
                settings,
                vector_store=replace(
                    settings.vector_store, collection_name=collection_name
                ),
            )
        self.collection_name = collection_name or settings.vector_store.collection_name
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        trace: TraceContext | None = None,
    ) -> list[str]:
        """Write the chunks and their vectors, and return their ids.

        The ids come back in the order the chunks came in, so the caller can
        line them up with the vectors.
        """
        if not chunks:
            raise ValueError("Cannot upsert an empty list of chunks")
        if len(chunks) != len(vectors):
            raise ValueError(
                f"The number of chunks ({len(chunks)}) and the number of vectors "
                f"({len(vectors)}) must be the same"
            )

        records: list[dict[str, Any]] = []
        chunk_ids: list[str] = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = self._generate_chunk_id(chunk)
            chunk_ids.append(chunk_id)
            records.append(
                {
                    "id": chunk_id,
                    "vector": list(vector),
                    "metadata": {
                        **chunk.metadata,
                        "text": chunk.text,
                        "chunk_id": chunk_id,
                        "collection": self.collection_name,
                    },
                }
            )

        try:
            self.vector_store.upsert(records, trace=trace)
        except Exception as exc:
            raise RuntimeError(
                f"Writing {len(records)} chunks to the vector store failed: {exc}"
            ) from exc

        if trace is not None:
            trace.record_stage(
                "vector_upserter",
                {
                    "collection": self.collection_name,
                    "upserted": len(records),
                    "first_id": chunk_ids[0] if chunk_ids else "",
                },
            )
        return chunk_ids

    def upsert_batch(
        self,
        batches: list[tuple[list[Chunk], list[list[float]]]],
        trace: TraceContext | None = None,
    ) -> list[str]:
        """Write several batches of chunks and vectors in one go.

        The ids of every batch come back in order.
        """
        chunks: list[Chunk] = []
        vectors: list[list[float]] = []
        for batch_chunks, batch_vectors in batches:
            chunks.extend(batch_chunks)
            vectors.extend(batch_vectors)
        return self.upsert(chunks, vectors, trace)

    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """Return the id of a chunk: the same for the same chunk, every time."""
        source_path = chunk.metadata.get("source_path")
        if not source_path:
            raise ValueError("Chunk metadata must contain 'source_path'")
        chunk_index = chunk.metadata.get("chunk_index")
        if chunk_index is None:
            raise ValueError("Chunk metadata must contain 'chunk_index'")
        try:
            index = int(chunk_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"The chunk_index must be a number, got {chunk_index!r}") from exc

        source_hash = _short_hash(str(source_path))
        content_hash = _short_hash(chunk.text)
        return f"{source_hash}_{index:04d}_{content_hash}"


def _short_hash(text: str) -> str:
    """Return the first eight characters of the SHA256 of `text`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:ID_HASH_LENGTH]
