"""The ingestion traces page.

The page lists the ingestion runs that were recorded and draws the steps of
the one that is picked, so a slow run can be seen stage by stage.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from observability.dashboard.services.trace_service import (
    TRACE_TYPE_INGESTION,
    TraceService,
)

STAGE_ORDER = ("integrity", "load", "split", "transform", "encode", "store")


def render() -> None:
    """Draw the ingestion traces page."""
    st.title("Ingestion traces")
    st.caption("How the ingestion runs went, step by step.")

    service = TraceService()
    traces = service.list_traces(trace_type=TRACE_TYPE_INGESTION)

    if not traces:
        st.info(
            "No ingestion has been recorded yet. Ingest a document on the "
            "Ingestion page and it will show up here."
        )
        return

    labels = {trace.trace_id: _label(trace) for trace in traces}
    chosen = st.selectbox(
        "Ingestion run",
        options=[trace.trace_id for trace in traces],
        format_func=lambda trace_id: labels.get(trace_id, trace_id),
    )
    trace = service.get_trace(chosen)
    if trace is None:
        st.warning("That run is not in the file any more.")
        return

    _render_summary(trace)
    _render_waterfall(trace)
    _render_stages(trace)


def _label(trace) -> str:
    """Return the text that names a run in the list."""
    path = trace.metadata.get("file_path") or trace.metadata.get("source_path")
    started = trace.started_at[:19].replace("T", " ") if trace.started_at else "-"
    if path:
        return f"{started} - {path}"
    return f"{started} - {trace.trace_id}"


def _render_summary(trace) -> None:
    """Draw the numbers that describe the whole run."""
    columns = st.columns(3)
    columns[0].metric("Total", f"{trace.total_elapsed_ms:.0f} ms")
    columns[1].metric("Steps", len(trace.stages))
    columns[2].metric("Started", trace.started_at[:19].replace("T", " ") or "-")


def _render_waterfall(trace) -> None:
    """Draw how long each step of the run took."""
    timings = [
        {
            "stage": stage.name,
            "duration_ms": stage.elapsed_ms,
        }
        for stage in trace.stages
        if stage.elapsed_ms
    ]
    if not timings:
        st.info("This run did not record how long its steps took.")
        return

    frame = pd.DataFrame(timings)
    st.markdown("**Stage durations**")
    st.bar_chart(frame.set_index("stage")["duration_ms"], horizontal=True)
    st.dataframe(
        frame.sort_values("duration_ms", ascending=False),
        hide_index=True,
        width="stretch",
    )


def _render_stages(trace) -> None:
    """Draw what each step of the run recorded."""
    st.markdown("**Stages**")
    for stage in trace.stages:
        with st.expander(f"{stage.name} - {stage.elapsed_ms:.0f} ms", expanded=False):
            if not stage.data:
                st.caption("This step recorded nothing.")
                continue
            st.json(stage.data)
