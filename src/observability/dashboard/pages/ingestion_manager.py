"""The ingestion manager page.

The page runs an ingestion and undoes one. A file that is uploaded is
written to a temporary folder and handed to the pipeline; the progress bar
follows the steps of the run, and the documents that are already there can be
deleted again.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from core.trace.trace_collector import TraceCollector
from core.trace.trace_context import TraceContext
from ingestion.pipeline import IngestionPipeline
from observability.dashboard.services.data_service import DataService

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "default"
BATCH_STAGE = "embed_batch"
BATCH_SHARE = 0.1


def render() -> None:
    """Draw the ingestion manager page."""
    st.title("Ingestion manager")
    st.caption("Ingest a document, watch it go, and take documents away again.")

    tabs = st.tabs(["Ingest a document", "Manage documents"])

    with tabs[0]:
        _render_upload()
    with tabs[1]:
        _render_documents()


def _render_upload() -> None:
    """Draw the upload form and run the ingestion it starts."""
    service = DataService()
    collections = service.list_collections()
    collection = st.selectbox(
        "Collection",
        collections or [DEFAULT_COLLECTION],
        key="ingestion_collection",
    )
    force = st.checkbox(
        "Ingest again even if the document was ingested before",
        value=False,
        key="ingestion_force",
    )

    uploaded = st.file_uploader(
        "Document (PDF)",
        type=["pdf"],
        key="ingestion_upload",
    )

    if uploaded is None:
        st.info("Choose a PDF to ingest it into the collection.")
        return

    if not st.button("Ingest", type="primary", key="ingestion_run"):
        return

    with tempfile.TemporaryDirectory(prefix="rag_ingest_") as workdir:
        target = Path(workdir) / uploaded.name
        target.write_bytes(uploaded.getvalue())
        _run_ingestion(target, collection, force)


def _run_ingestion(path: Path, collection: str, force: bool) -> None:
    """Ingest the file and follow the steps of the run."""
    service = DataService()
    progress = st.progress(0.0, text="Starting the ingestion...")

    def on_progress(stage: str, current: int, total: int) -> None:
        """Move the progress bar along as the steps are done.

        The pipeline also reports the batches it encodes, and those batches
        are counted against the batches of that step rather than against the
        steps of the run, so the bar does not jump back and forth.
        """
        if stage == BATCH_STAGE:
            fraction = current / total if total else 0.0
            progress.progress(
                min(fraction * BATCH_SHARE, 1.0),
                text=f"{stage} ({current}/{total})",
            )
            return
        fraction = current / total if total else 0.0
        progress.progress(min(fraction, 1.0), text=f"{stage} ({current}/{total})")

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["file_path"] = str(path)
    trace.metadata["collection"] = collection

    try:
        pipeline = IngestionPipeline(service.settings, collection=collection, force=force)
        result = pipeline.run(path, trace=trace, on_progress=on_progress)
    except Exception as exc:
        logger.warning("The ingestion of %s failed: %s", path, exc)
        progress.empty()
        st.error(f"The ingestion failed: {exc}")
        return

    progress.progress(1.0, text="Done.")
    _collect_trace(trace)

    if result.skipped:
        st.warning("This document was ingested before and was left alone.")
        return
    st.success(
        f"The document was ingested: {result.chunk_count} chunks, "
        f"{result.image_count} images."
    )


def _collect_trace(trace: TraceContext) -> None:
    """Write the trace of the run out, so the traces page can show it."""
    try:
        collector = TraceCollector()
        collector.collect(trace)
    except Exception as exc:
        logger.warning("The trace could not be written: %s", exc)


def _render_documents() -> None:
    """Draw the documents that are there, with a button to delete each."""
    service = DataService()
    collections = service.list_collections()
    collection = st.selectbox(
        "Collection",
        collections or [DEFAULT_COLLECTION],
        key="manage_collection",
    )

    try:
        documents = service.list_documents(collection)
    except Exception as exc:
        st.error(f"The documents could not be read: {exc}")
        return

    if not documents:
        st.info("This collection holds no document yet.")
        return

    for document in documents:
        _render_document_row(service, document, collection)


def _render_document_row(service: DataService, document: dict, collection: str) -> None:
    """Draw one document and the button that deletes it."""
    source_path = str(document.get("source_path", ""))
    source_hash = str(document.get("source_hash", ""))

    columns = st.columns([5, 1, 1, 1])
    columns[0].write(source_path or source_hash)
    columns[1].metric("Chunks", document.get("chunk_count", 0))
    columns[2].metric("Images", document.get("image_count", 0))

    if columns[3].button("Delete", key=f"delete_{source_hash}"):
        _delete_document(service, source_path, source_hash, collection)


def _delete_document(
    service: DataService,
    source_path: str,
    source_hash: str,
    collection: str,
) -> None:
    """Delete a document and say what happened."""
    result = service.delete_document(source_path, collection, source_hash=source_hash)
    if result.success:
        st.success(
            f"The document was deleted: {result.chunks_deleted} chunks, "
            f"{result.images_deleted} images."
        )
        st.rerun()
        return
    st.error("The document was not deleted completely: " + "; ".join(result.errors))
