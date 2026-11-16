"""The system overview page.

The page answers two questions at once: what is the system made of, and what
does it hold right now. The first is answered with the component cards, the
second with the numbers of the ingested documents, chunks and images.
"""

from __future__ import annotations

import streamlit as st

from observability.dashboard.services.config_service import ConfigService
from observability.dashboard.services.data_service import DataService


def render() -> None:
    """Draw the overview page."""
    st.title("System overview")
    st.caption("What the server is made of, and what it holds right now.")

    _render_components()
    st.divider()
    _render_data_statistics()


def _render_components() -> None:
    """Draw one card per configured component."""
    st.subheader("Components")

    service = ConfigService()
    try:
        cards = service.get_component_cards()
    except Exception as exc:
        st.error(f"The configuration could not be read: {exc}")
        return

    columns = st.columns(3)
    for index, card in enumerate(cards):
        with columns[index % len(columns)]:
            with st.container(border=True):
                st.markdown(f"**{card.name}**")
                st.caption(f"{card.provider} · {card.model}")
                for label, value in card.as_rows():
                    st.markdown(f"- {label}: `{value}`")


def _render_data_statistics() -> None:
    """Draw the numbers of what is stored."""
    st.subheader("Data statistics")

    service = DataService()
    try:
        collections = service.list_collections()
    except Exception as exc:
        st.error(f"The collections could not be read: {exc}")
        return

    if not collections:
        st.info(
            "There is no collection yet. Ingest a document on the Ingestion page "
            "to see what it holds."
        )
        return

    collection = st.selectbox("Collection", collections)
    try:
        stats = service.get_collection_stats(collection)
    except Exception as exc:
        st.error(f"The statistics could not be read: {exc}")
        return

    columns = st.columns(3)
    columns[0].metric("Documents", stats.document_count)
    columns[1].metric("Chunks", stats.chunk_count)
    columns[2].metric("Images", stats.image_count)
