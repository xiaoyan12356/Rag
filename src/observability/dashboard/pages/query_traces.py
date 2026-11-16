"""The query traces page.

The page lists the questions that were asked and shows how one of them was
answered: how long each step took, what the dense route and the sparse route
found on their own, and what the reranker did to the order.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from observability.dashboard.services.evaluation_service import EvaluationService
from observability.dashboard.services.trace_service import (
    TRACE_TYPE_QUERY,
    TraceService,
)


def render() -> None:
    """Draw the query traces page."""
    st.title("Query traces")
    st.caption("How the questions were answered, step by step.")

    service = TraceService()
    keyword = st.text_input("Search in the question", key="query_trace_search")
    traces = service.search(keyword, trace_type=TRACE_TYPE_QUERY)

    if not traces:
        st.info(
            "No question has been recorded yet. Ask one through the MCP server "
            "and it will show up here."
        )
        return

    labels = {trace.trace_id: _label(trace) for trace in traces}
    chosen = st.selectbox(
        "Question",
        options=[trace.trace_id for trace in traces],
        format_func=lambda trace_id: labels.get(trace_id, trace_id),
    )
    trace = service.get_trace(chosen)
    if trace is None:
        st.warning("That run is not in the file any more.")
        return

    _render_summary(trace)
    _render_waterfall(trace)
    _render_routes(trace)
    _render_rerank(trace)
    _render_evaluation(trace)


def _label(trace) -> str:
    """Return the text that names a run in the list."""
    question = _question_of(trace)
    started = trace.started_at[:19].replace("T", " ") if trace.started_at else "-"
    if question:
        return f"{started} - {question}"
    return f"{started} - {trace.trace_id}"


def _question_of(trace) -> str:
    """Return the question the run answered."""
    for key in ("query", "original_query"):
        value = trace.metadata.get(key)
        if value:
            return str(value)
    rerank = trace.stage_data("rerank")
    value = rerank.get("query")
    if value:
        return str(value)
    processing = trace.stage_data("query_processing")
    keywords = processing.get("keywords")
    if keywords:
        return " ".join(str(keyword) for keyword in keywords)
    return ""


def _render_summary(trace) -> None:
    """Draw the numbers that describe the whole run."""
    columns = st.columns(3)
    columns[0].metric("Total", f"{trace.total_elapsed_ms:.0f} ms")
    columns[1].metric("Steps", len(trace.stages))
    columns[2].metric("Started", trace.started_at[:19].replace("T", " ") or "-")


def _render_waterfall(trace) -> None:
    """Draw how long each step of the run took."""
    timings = [
        {"stage": stage.name, "duration_ms": stage.elapsed_ms}
        for stage in trace.stages
        if stage.elapsed_ms
    ]
    if not timings:
        st.info("This run did not record how long its steps took.")
        return

    frame = pd.DataFrame(timings)
    st.markdown("**Stage durations**")
    st.bar_chart(frame.set_index("stage")["duration_ms"], horizontal=True)


def _render_routes(trace) -> None:
    """Draw what the dense route and the sparse route found."""
    dense = trace.stage_data("dense_retrieval")
    sparse = trace.stage_data("sparse_retrieval")

    if not dense and not sparse:
        st.info("This run did not record the results of the two routes.")
        return

    st.markdown("**Dense vs sparse**")
    columns = st.columns(2)

    with columns[0]:
        st.markdown("Dense")
        st.metric("Returned", dense.get("returned", 0))
        st.caption(f"collection: `{dense.get('collection', '-')}`")
        dense_ids = list(dense.get("chunk_ids") or [])
        if dense_ids:
            st.dataframe(
                pd.DataFrame({"rank": range(1, len(dense_ids) + 1), "chunk_id": dense_ids}),
                hide_index=True,
                width="stretch",
            )

    with columns[1]:
        st.markdown("Sparse")
        st.metric("Returned", sparse.get("returned", 0))
        st.caption(f"keywords: `{', '.join(sparse.get('keywords') or []) or '-'}`")
        sparse_ids = list(sparse.get("chunk_ids") or [])
        if sparse_ids:
            st.dataframe(
                pd.DataFrame({"rank": range(1, len(sparse_ids) + 1), "chunk_id": sparse_ids}),
                hide_index=True,
                width="stretch",
            )

    fusion = trace.stage_data("fusion")
    if fusion:
        st.caption(
            f"The fusion put {fusion.get('fused_count', 0)} chunks together "
            f"(dense {fusion.get('dense_count', 0)}, sparse "
            f"{fusion.get('sparse_count', 0)})."
        )


def _render_rerank(trace) -> None:
    """Draw what the reranker did to the order."""
    rerank = trace.stage_data("rerank")
    if not rerank:
        st.info("This run did not go through the reranker.")
        return

    st.markdown("**What the reranker changed**")
    columns = st.columns(3)
    columns[0].metric("Candidates", rerank.get("candidates", 0))
    columns[1].metric("Provider", str(rerank.get("provider", "-")))
    columns[2].metric("Fallback", "yes" if rerank.get("fallback_used") else "no")

    before = list(rerank.get("order_before") or [])
    after = list(rerank.get("order_after") or [])
    if not before and not after:
        return

    moved = _moved(before, after)
    if moved:
        frame = pd.DataFrame(moved, columns=["chunk_id", "before", "after"])
        st.dataframe(frame, hide_index=True, width="stretch")
    else:
        st.caption("The reranker kept the order the fusion found.")


def _moved(before: list[str], after: list[str]) -> list[tuple[str, str, str]]:
    """Return the chunks that ended up in a different place."""
    positions_before = {chunk_id: rank for rank, chunk_id in enumerate(before, start=1)}
    positions_after = {chunk_id: rank for rank, chunk_id in enumerate(after, start=1)}
    moved: list[tuple[str, str, str]] = []
    for chunk_id, rank_after in positions_after.items():
        rank_before = positions_before.get(chunk_id)
        if rank_before is None or rank_before == rank_after:
            continue
        moved.append((chunk_id, str(rank_before), str(rank_after)))
    return moved


def _render_evaluation(trace) -> None:
    """Draw the button that judges the answer of this question."""
    question = _question_of(trace)
    if not question:
        return

    st.markdown("**How good is the answer**")
    st.caption(
        "The question is asked again and the answer is judged with the "
        "evaluator of the configuration."
    )

    if not st.button("Ragas Evaluate", key=f"ragas_{trace.trace_id}"):
        return

    service = EvaluationService()
    with st.spinner("The answer is being judged..."):
        try:
            chunks = service.retrieve(question)
            metrics = service.evaluate_answer(question, chunks, backend="ragas")
        except Exception as exc:
            st.error(f"The answer could not be judged: {exc}")
            return

    if not metrics:
        st.warning("The evaluator did not report any metric.")
        return

    columns = st.columns(len(metrics))
    for column, (key, value) in zip(columns, metrics.items()):
        column.metric(key.replace("_", " ").title(), f"{float(value):.3f}")
