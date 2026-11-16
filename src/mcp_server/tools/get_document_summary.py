"""The tool that describes one document.

The chunks of a document carry the title, the summary and the tags that were
made when the document was taken in. The tool finds the chunks of a document
in the vector store and reads that information off them.

A document can be named in several ways - by its id, by the hash of its file
or by the path it was loaded from - so all of them are tried before the
document is called missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.response.response_builder import MCPToolResult
from core.settings import Settings
from mcp_server.tools import ToolArgumentError

logger = logging.getLogger(__name__)

TOOL_NAME = "get_document_summary"

DESCRIPTION = (
    "Return the title, the summary and the tags of one document of the "
    "knowledge hub."
)

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "The id of the document.",
        },
        "collection": {
            "type": "string",
            "description": (
                "The collection the document lives in. The configured one is used "
                "when it is left out."
            ),
        },
    },
    "required": ["doc_id"],
}


class DocumentLookup:
    """Find the chunks of a document in the vector store."""

    def __init__(
        self,
        settings: Settings,
        collection: str | None = None,
        store: Any | None = None,
    ) -> None:
        """Set up the lookup.

        Args:
            settings: The configuration that says where the store lives.
            collection: The collection the document lives in.
            store: The store the chunks are in; one is built from the
                settings when this is left out.
        """
        self.settings = settings
        self.collection = collection or settings.vector_store.collection_name
        self._store = store

    @property
    def store(self) -> Any:
        """Return the store the chunks are in."""
        if self._store is None:
            from libs.vector_store.vector_store_factory import VectorStoreFactory

            self._store = VectorStoreFactory.create(self.settings)
        return self._store

    def chunks_of(self, doc_id: str) -> list[dict[str, Any]]:
        """Return the chunks of a document.

        Args:
            doc_id: The id, the hash or the path of the document.
        """
        collection = getattr(self.store, "collection", None)
        if collection is None:
            raise RuntimeError("The vector store does not expose a collection to read")

        result = collection.get(include=["metadatas", "documents"])
        metadatas = result.get("metadatas") or []
        ids = result.get("ids") or []
        documents = result.get("documents") or []

        found: list[dict[str, Any]] = []
        for index, record_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            if not _belongs_to(metadata, record_id, doc_id):
                continue
            found.append(
                {
                    "id": record_id,
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": metadata,
                }
            )
        return found


def get_document_summary(
    doc_id: str,
    collection: str | None = None,
    *,
    settings: Settings | None = None,
    lookup: DocumentLookup | None = None,
) -> MCPToolResult:
    """Return what the knowledge hub knows about a document.

    Args:
        doc_id: The id of the document.
        collection: The collection the document lives in.
        settings: The configuration of the providers.
        lookup: The one that finds the chunks; one is built from the settings
            when this is left out.

    Returns:
        The title, the summary and the tags of the document.

    Raises:
        ToolArgumentError: The arguments do not fit the schema.
    """
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ToolArgumentError("The 'doc_id' argument must be a non-empty string")
    if collection is not None and (
        not isinstance(collection, str) or not collection.strip()
    ):
        raise ToolArgumentError("The 'collection' argument must be a non-empty string")

    if lookup is None:
        if settings is None:
            raise RuntimeError("The tool needs a configuration to look the document up in")
        lookup = DocumentLookup(settings, collection=collection)

    records = lookup.chunks_of(doc_id)
    if not records:
        return MCPToolResult(
            content=[
                {
                    "type": "text",
                    "text": f"No document with the id '{doc_id}' was found.",
                }
            ],
            structured_content={"doc_id": doc_id, "found": False},
            is_error=True,
        )

    metadata = _metadata_of(records)
    summary = {
        "doc_id": doc_id,
        "found": True,
        "title": str(metadata.get("title") or ""),
        "summary": str(metadata.get("summary") or ""),
        "tags": _tags_of(metadata),
        "source_path": str(metadata.get("source_path") or ""),
        "page_count": metadata.get("page_count"),
        "chunk_count": len(records),
    }
    return MCPToolResult(
        content=[{"type": "text", "text": _markdown(summary)}],
        structured_content=summary,
    )


def handle(arguments: dict[str, Any], settings: Settings | None = None) -> MCPToolResult:
    """Run the tool with the arguments a client sent.

    Args:
        arguments: The arguments of the call.
        settings: The configuration of the providers.
    """
    return get_document_summary(
        doc_id=arguments.get("doc_id"),
        collection=arguments.get("collection"),
        settings=settings,
    )


def _belongs_to(metadata: dict[str, Any], record_id: str, doc_id: str) -> bool:
    """Say whether a chunk belongs to the document that was asked for.

    A document can be named by its id, by the hash of its file or by the path
    it was loaded from, so all of those are tried.
    """
    wanted = doc_id.strip()

    stored_doc_id = str(metadata.get("doc_id") or "")
    if stored_doc_id and stored_doc_id == wanted:
        return True

    doc_hash = str(metadata.get("doc_hash") or "")
    if doc_hash and (wanted == doc_hash or doc_hash.startswith(wanted)):
        return True

    source_path = str(metadata.get("source_path") or "")
    if source_path:
        path = Path(source_path)
        if wanted in (source_path, path.name, path.stem):
            return True

    return record_id == wanted or record_id.startswith(f"{wanted}_")


def _metadata_of(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the metadata of the document: the first chunk carries it."""
    first = records[0]
    for record in records:
        metadata = record.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("chunk_index") == 0:
            first = record
            break
    metadata = first.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _tags_of(metadata: dict[str, Any]) -> list[str]:
    """Return the tags of a document, whichever way they were stored."""
    tags = metadata.get("tags")
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    if isinstance(tags, list):
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    return []


def _markdown(summary: dict[str, Any]) -> str:
    """Return what is known about a document as Markdown."""
    title = summary["title"] or summary["source_path"] or summary["doc_id"]
    lines = [f"## {title}", ""]

    if summary["tags"]:
        lines.append(f"- **Tags**: {', '.join(summary['tags'])}")
    lines.append(f"- **Chunks**: {summary['chunk_count']}")
    if summary["source_path"]:
        lines.append(f"- **Source**: {summary['source_path']}")
    lines.append("")

    if summary["summary"]:
        lines.append(summary["summary"])
    return "\n".join(lines)
