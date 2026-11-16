"""The data browser page.

The page shows the documents that were ingested and, for every one of them,
the chunks it was cut into and the images it holds. The text of a chunk can
be folded away, so a document with many chunks stays readable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from observability.dashboard.services.data_service import DataService


def render() -> None:
    """Draw the data browser page."""
    st.title("Data browser")
    st.caption("The documents that were ingested, their chunks and their images.")

    service = DataService()
    collections = service.list_collections()
    if not collections:
        st.info(
            "There is no collection yet. Ingest a document on the Ingestion page "
            "to see it here."
        )
        return

    collection = st.selectbox("Collection", collections)

    try:
        documents = service.list_documents(collection)
    except Exception as exc:
        st.error(f"The documents could not be read: {exc}")
        return

    if not documents:
        st.info("This collection holds no document yet.")
        return

    st.caption(f"{len(documents)} document(s) in '{collection}'.")
    for document in documents:
        _render_document(service, document, collection)


def _render_document(service: DataService, document: dict, collection: str) -> None:
    """Draw one document and the chunks it was cut into."""
    source_path = str(document.get("source_path", ""))
    source_hash = str(document.get("source_hash", ""))

    with st.expander(source_path or source_hash, expanded=False):
        columns = st.columns(3)
        columns[0].metric("Chunks", document.get("chunk_count", 0))
        columns[1].metric("Images", document.get("image_count", 0))
        columns[2].metric(
            "Ingested",
            str(document.get("processed_at") or "-")[:19].replace("T", " "),
        )
        st.caption(f"hash: `{source_hash}`")

        _render_images(service, source_hash, collection)
        _render_chunks(service, source_hash, collection)


def _render_images(service: DataService, source_hash: str, collection: str) -> None:
    """Draw the images of the document."""
    images = service.get_images(source_hash, collection)
    if not images:
        return

    st.markdown("**Images**")
    for image in images:
        image_id = str(image.get("image_id", ""))
        path = service.get_image_path(image_id)
        if path:
            st.image(path, caption=f"{image_id} (page {image.get('page_num')})", width=240)


def _render_chunks(service: DataService, source_hash: str, collection: str) -> None:
    """Draw the chunks of the document."""
    chunks = service.get_chunks(source_hash, collection)
    if not chunks:
        return

    st.markdown("**Chunks**")
    for position, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        title = metadata.get("title") or f"Chunk {position}"
        with st.expander(f"{position}. {title}", expanded=False):
            st.write(chunk.get("text", ""))
            _render_metadata(metadata)


def _render_metadata(metadata: dict) -> None:
    """Draw the fields of the metadata of a chunk."""
    fields = [
        ("source_path", metadata.get("source_path")),
        ("doc_type", metadata.get("doc_type")),
        ("chunk_index", metadata.get("chunk_index")),
        ("page", metadata.get("page")),
        ("tags", metadata.get("tags")),
    ]
    rows = {name: value for name, value in fields if value not in (None, "", [])}
    if not rows:
        return
    st.markdown("**Metadata**")
    st.dataframe(pd.DataFrame([rows]), hide_index=True, width="stretch")
