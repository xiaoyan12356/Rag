"""The evaluation panel page.

The evaluation measures how good the answers are. A golden test set is a
list of questions with the chunks that should answer them, and running it
says how often the search finds those chunks and how high they end up.

The run takes a while, because every question is asked and every answer is
judged, so it is started with a button and the result is shown when it is
there. The result of the last run is kept, so the page has something to show
before a new run is started.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from observability.dashboard.services.evaluation_service import EvaluationService

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "hit_rate": "Hit rate",
    "mrr": "MRR",
}


def render() -> None:
    """Draw the evaluation panel page."""
    st.title("Evaluation panel")
    st.caption("How often the search finds the chunks it should.")

    service = EvaluationService()

    tabs = st.tabs(["Run an evaluation", "The last result"])

    with tabs[0]:
        _render_run(service)
    with tabs[1]:
        _render_last(service)


def _render_run(service: EvaluationService) -> None:
    """Draw the form that starts a run and the result of it."""
    test_sets = service.list_test_sets()
    chosen = st.selectbox(
        "Golden test set",
        options=test_sets,
        format_func=lambda path: path.rsplit("/", 1)[-1],
        key="evaluation_test_set",
    )
    backend = st.selectbox(
        "Evaluator",
        options=service.available_backends(),
        key="evaluation_backend",
    )
    top_k = st.number_input(
        "Chunks per question",
        min_value=1,
        max_value=50,
        value=10,
        key="evaluation_top_k",
    )

    if not st.button("Run the evaluation", type="primary", key="evaluation_run"):
        return

    with st.spinner("The questions are being asked..."):
        try:
            report = service.run(chosen, backend=backend, top_k=int(top_k))
        except Exception as exc:
            st.error(f"The evaluation could not run: {exc}")
            return

    _render_report(report.to_dict())


def _render_last(service: EvaluationService) -> None:
    """Draw the result of the last run."""
    payload = service.last_report()
    if payload is None:
        st.info(
            "No evaluation has been run yet. Run one on the other tab, or use "
            "`python scripts/evaluate.py` from the command line."
        )
        return

    st.caption(f"Last run: {payload.get('test_set_path', '-')}")
    _render_report(payload)


def _render_report(payload: dict) -> None:
    """Draw the numbers of a run and the detail of every question."""
    metrics = dict(payload.get("metrics") or {})
    metrics.setdefault("hit_rate", payload.get("hit_rate", 0.0))
    metrics.setdefault("mrr", payload.get("mrr", 0.0))

    st.markdown("**What the run found**")
    columns = st.columns(len(metrics))
    for column, (key, value) in zip(columns, metrics.items()):
        column.metric(METRIC_LABELS.get(key, key.replace("_", " ").title()), f"{value:.3f}")

    results = list(payload.get("query_results") or [])
    if not results:
        st.info("The run asked no questions.")
        return

    rows = []
    for result in results:
        rows.append(
            {
                "question": result.get("query", ""),
                "hit": "yes" if result.get("hit") else "no",
                "rank": result.get("rank") if result.get("rank") is not None else "-",
                "found": ", ".join(result.get("retrieved_chunk_ids") or []) or "-",
            }
        )

    st.markdown("**Every question**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
