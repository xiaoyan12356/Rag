"""The dashboard of the Modular RAG server.

The dashboard is a Streamlit application with one page per job: the overview
tells what the system is made of, the data browser shows what was ingested,
the ingestion manager runs and undoes an ingestion, the two trace pages show
how a run went, and the evaluation panel shows how good the answers are.

The pages are registered with `st.navigation`, so Streamlit keeps the
sidebar in one place and each page is a module under `pages/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PAGE_TITLE = "Modular RAG Dashboard"


def _overview() -> None:
    """Show the system overview."""
    from observability.dashboard.pages.overview import render

    render()


def _data_browser() -> None:
    """Show the ingested documents and their chunks."""
    from observability.dashboard.pages.data_browser import render

    render()


def _ingestion_manager() -> None:
    """Show the ingestion manager."""
    from observability.dashboard.pages.ingestion_manager import render

    render()


def _ingestion_traces() -> None:
    """Show the ingestion traces."""
    from observability.dashboard.pages.ingestion_traces import render

    render()


def _query_traces() -> None:
    """Show the query traces."""
    from observability.dashboard.pages.query_traces import render

    render()


def _evaluation_panel() -> None:
    """Show the evaluation panel."""
    from observability.dashboard.pages.evaluation_panel import render

    render()


def build_pages() -> list[st.Page]:
    """Return the pages of the dashboard, in the order they are shown."""
    return [
        st.Page(_overview, title="Overview", icon=":material/dashboard:", default=True),
        st.Page(_data_browser, title="Data Browser", icon=":material/search:"),
        st.Page(_ingestion_manager, title="Ingestion", icon=":material/upload:"),
        st.Page(_ingestion_traces, title="Ingestion Traces", icon=":material/timeline:"),
        st.Page(_query_traces, title="Query Traces", icon=":material/travel_explore:"),
        st.Page(_evaluation_panel, title="Evaluation", icon=":material/analytics:"),
    ]


def main() -> None:
    """Set up the page and run the one the user picked."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=":material/dashboard:", layout="wide")
    navigation = st.navigation(build_pages())
    navigation.run()


main()
