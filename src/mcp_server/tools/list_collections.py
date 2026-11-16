"""The tool that lists the collections.

The collections live in folders under `data/documents/`, one folder per
collection. A folder may hold a `description.txt` that says what the
collection is about.

The vector store also knows the collections that hold chunks, so both are
reported: the folder says what was put in, the store says what can be
searched.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.response.response_builder import MCPToolResult
from core.settings import Settings, resolve_path
from libs.vector_store.vector_store_factory import VectorStoreFactory
from mcp_server.tools import ToolArgumentError

logger = logging.getLogger(__name__)

TOOL_NAME = "list_collections"

DESCRIPTION = (
    "List the document collections of the knowledge hub, with how many "
    "documents and chunks each one holds and what it is about."
)

INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}

DEFAULT_DOCUMENTS_ROOT = "data/documents"
DESCRIPTION_FILE = "description.txt"
DOCUMENT_SUFFIXES = (".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".html", ".csv")


def list_collections(
    root: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> MCPToolResult:
    """Return the collections that are there.

    Args:
        root: The folder the collections live in; the default one is used
            when this is left out.
        settings: The configuration that says where the store lives.

    Returns:
        The names of the collections and what is known about each of them.
    """
    documents_root = Path(root) if root is not None else resolve_path(DEFAULT_DOCUMENTS_ROOT)

    collections: list[dict[str, Any]] = []
    if documents_root.is_dir():
        for entry in sorted(documents_root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            collections.append(_describe(entry))

    _add_stored_collections(collections, settings)

    markdown = _markdown(collections)
    return MCPToolResult(
        content=[{"type": "text", "text": markdown}],
        structured_content={"collections": collections},
    )


def handle(arguments: dict[str, Any], settings: Settings | None = None) -> MCPToolResult:
    """Run the tool with the arguments a client sent.

    Args:
        arguments: The arguments of the call.
        settings: The configuration of the providers.
    """
    root = arguments.get("root")
    if root is not None and not isinstance(root, str):
        raise ToolArgumentError("The 'root' argument must be a string")
    return list_collections(root, settings=settings)


def _describe(entry: Path) -> dict[str, Any]:
    """Return what is known about one collection."""
    return {
        "name": entry.name,
        "description": _description_of(entry),
        "document_count": _document_count(entry),
        "chunk_count": 0,
        "path": str(entry),
    }


def _add_stored_collections(
    collections: list[dict[str, Any]],
    settings: Settings | None,
) -> None:
    """Add the collections the vector store holds to the list.

    The documents of a collection can be ingested from any path, so the
    folders under the documents root do not tell the whole story: the store
    knows what can really be searched. A store that cannot be opened leaves
    the list as it is.

    Args:
        collections: The list to add to; it is changed in place.
        settings: The configuration that says where the store lives.
    """
    if settings is None:
        return

    try:
        store = VectorStoreFactory.create(settings)
        stored = store.list_collections()
    except Exception as exc:
        logger.warning("The collections of the store could not be read: %s", exc)
        return

    by_name = {collection["name"]: collection for collection in collections}
    for entry in stored:
        name = str(entry.get("name", ""))
        if not name:
            continue
        chunk_count = int(entry.get("chunk_count", 0))
        known = by_name.get(name)
        if known is None:
            collections.append(
                {
                    "name": name,
                    "description": "",
                    "document_count": 0,
                    "chunk_count": chunk_count,
                    "path": "",
                }
            )
        else:
            known["chunk_count"] = chunk_count


def _description_of(entry: Path) -> str:
    """Return the description of a collection, when it has one."""
    description_file = entry / DESCRIPTION_FILE
    if not description_file.is_file():
        return ""
    try:
        return description_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _document_count(entry: Path) -> int:
    """Return how many documents a collection holds.

    The description of the collection describes the collection, so it is not
    one of the documents in it.
    """
    count = 0
    for child in entry.rglob("*"):
        if not child.is_file() or child.suffix.lower() not in DOCUMENT_SUFFIXES:
            continue
        if child.name == DESCRIPTION_FILE:
            continue
        count += 1
    return count


def _markdown(collections: list[dict[str, Any]]) -> str:
    """Return the collections as Markdown."""
    if not collections:
        return (
            "## Collections\n\n"
            "There is no collection yet. Have the documents been ingested? Run "
            "scripts/ingest.py."
        )

    lines = ["## Collections", ""]
    for collection in collections:
        description = collection["description"]
        suffix = f" - {description}" if description else ""
        counts = f"{collection['document_count']} document(s)"
        if collection["chunk_count"]:
            counts += f", {collection['chunk_count']} chunk(s)"
        lines.append(f"- **{collection['name']}** - {counts}{suffix}")
    return "\n".join(lines)
