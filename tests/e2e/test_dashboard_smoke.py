"""End-to-end test of the dashboard.

The pages are run with the test harness of Streamlit, so a page that raises
a Python error fails the test instead of showing a red screen to whoever
opens it.

The pages read the configuration of the project, the traces that were written
and the stores that were filled, so the test sets all three up: a temporary
configuration, a temporary trace file and a collection that was ingested with
a stand-in for the embedding provider.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest
from stub_embedding_server import DIMENSION, start_stub_server, stop_stub_server

from core.trace.trace_collector import TraceCollector
from core.trace.trace_context import TraceContext
from observability.dashboard.services import evaluation_service

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
INGEST_SCRIPT = REPO_ROOT / "scripts" / "ingest.py"
DASHBOARD_APP = REPO_ROOT / "src" / "observability" / "dashboard" / "app.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "sample_documents"
COLLECTION = "dashboard_smoke"

PAGES = (
    "overview",
    "data_browser",
    "ingestion_manager",
    "ingestion_traces",
    "query_traces",
    "evaluation_panel",
)


@pytest.fixture
def embedding_server():
    """A local server that answers the embedding requests."""
    address, server = start_stub_server()
    yield address
    stop_stub_server(server)


@pytest.fixture
def config_file(tmp_path: Path, embedding_server: str) -> Path:
    """A configuration that keeps every product in a temporary folder."""
    config = {
        "llm": {"provider": "openai", "model": "gpt-4o"},
        "embedding": {
            "provider": "ollama",
            "model": "stub",
            "dimensions": DIMENSION,
            "base_url": embedding_server,
        },
        "vision_llm": {"enabled": False},
        "vector_store": {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": COLLECTION,
        },
        "retrieval": {"dense_top_k": 20, "sparse_top_k": 20, "fusion_top_k": 10},
        "rerank": {"provider": "none"},
        "evaluation": {"provider": "custom"},
        "observability": {
            "log_level": "INFO",
            "trace_file": str(tmp_path / "logs" / "traces.jsonl"),
        },
        "ingestion": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "splitter": "recursive",
            "batch_size": 16,
            "bm25_index_dir": str(tmp_path / "bm25"),
            "images_root": str(tmp_path / "images"),
            "image_index_db": str(tmp_path / "image_index.db"),
            "history_db": str(tmp_path / "history.db"),
        },
    }
    path = tmp_path / "config" / "settings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def ingested(config_file: Path) -> Path:
    """Take the sample documents in, so the pages have something to show."""
    result = subprocess.run(
        [
            sys.executable,
            str(INGEST_SCRIPT),
            "--path",
            str(FIXTURES),
            "--collection",
            COLLECTION,
            "--config",
            str(config_file),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return config_file


@pytest.fixture
def dashboard(ingested: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the dashboard at the temporary stores and the temporary traces.

    The pages read the configuration through a path that is relative to the
    working directory, so the working directory is moved next to the
    temporary configuration. The last report is redirected as well, so the
    test never touches the files of the checkout.

    Args:
        ingested: The configuration of the temporary stores.
        tmp_path: The folder of the test.
        monkeypatch: The patcher that undoes the changes afterwards.

    Returns:
        The path of the temporary configuration.
    """
    monkeypatch.chdir(tmp_path)

    _write_traces(tmp_path / "logs" / "traces.jsonl")
    monkeypatch.setattr(
        evaluation_service,
        "REPORT_PATH",
        str(tmp_path / "reports" / "evaluation.json"),
    )
    return ingested


def _write_traces(path: Path) -> None:
    """Write one ingestion run and one answered question into the file.

    The runs are written with the collector and the context of the project,
    so the file has the shape the dashboard expects.
    """
    collector = TraceCollector(path)

    ingestion = TraceContext(trace_type="ingestion")
    ingestion.metadata["file_path"] = "tests/fixtures/sample_documents/guide.md"
    ingestion.metadata["collection"] = COLLECTION
    for stage in ("integrity", "load", "split", "transform", "encode", "store"):
        ingestion.record_stage(stage, {"ok": True}, elapsed_ms=5.0)
    collector.collect(ingestion)

    query = TraceContext(trace_type="query")
    query.metadata["query"] = "how is a document split into chunks?"
    query.record_stage(
        "query_processing",
        {"keywords": ["document", "chunks"]},
        elapsed_ms=3.0,
    )
    query.record_stage(
        "dense_retrieval",
        {
            "returned": 2,
            "collection": COLLECTION,
            "chunk_ids": ["a_0000_11111111", "b_0000_22222222"],
        },
        elapsed_ms=12.0,
    )
    query.record_stage(
        "sparse_retrieval",
        {
            "returned": 1,
            "keywords": ["chunks"],
            "chunk_ids": ["a_0000_11111111"],
        },
        elapsed_ms=4.0,
    )
    query.record_stage(
        "fusion",
        {"fused_count": 2, "dense_count": 2, "sparse_count": 1},
        elapsed_ms=1.0,
    )
    query.record_stage(
        "rerank",
        {
            "candidates": 2,
            "provider": "none",
            "fallback_used": True,
            "order_before": ["a_0000_11111111", "b_0000_22222222"],
            "order_after": ["b_0000_22222222", "a_0000_11111111"],
        },
        elapsed_ms=2.0,
    )
    collector.collect(query)


def _run(page_name: str) -> AppTest:
    """Run a page in the harness of Streamlit and return the result.

    The page is run as a small script that imports it and draws it, because
    the harness runs the source of a script and not the module a function
    was taken from.

    Args:
        page_name: The name of the module of the page.

    Returns:
        The result of the run.
    """
    source = (
        "import streamlit as st\n"
        f"from observability.dashboard.pages import {page_name}\n"
        f"{page_name}.render()\n"
    )
    app = AppTest.from_string(source)
    app.run()
    return app


class TestThePages:
    """Every page has to load without raising a Python error."""

    @pytest.mark.parametrize("page_name", PAGES)
    def test_the_page_loads(self, page_name: str, dashboard: Path):
        app = _run(page_name)

        assert not app.exception
        assert app.title

    def test_the_dashboard_opens_on_the_overview(self, dashboard: Path):
        app = AppTest.from_file(str(DASHBOARD_APP))
        app.run()

        assert not app.exception
        assert app.title[0].value == "System overview"


class TestTheOverview:
    def test_the_components_are_shown(self, dashboard: Path):
        app = _run("overview")

        assert "Components" in app.subheader[0].value

    def test_the_numbers_of_the_collection_are_shown(self, dashboard: Path):
        app = _run("overview")

        labels = [metric.label for metric in app.metric]
        assert "Documents" in labels
        assert "Chunks" in labels
        assert "Images" in labels


class TestTheDataBrowser:
    def test_the_ingested_documents_are_listed(self, dashboard: Path):
        app = _run("data_browser")

        assert app.expander
        assert any("simple.pdf" in item.label for item in app.expander)

    def test_the_collection_can_be_picked(self, dashboard: Path):
        app = _run("data_browser")

        assert app.selectbox
        assert COLLECTION in app.selectbox[0].options


class TestTheIngestionManager:
    def test_the_upload_form_is_shown(self, dashboard: Path):
        app = _run("ingestion_manager")

        assert app.file_uploader
        assert any("PDF" in item.label for item in app.file_uploader)

    def test_the_ingested_documents_can_be_deleted(self, dashboard: Path):
        app = _run("ingestion_manager")

        assert any(button.label == "Delete" for button in app.button)


class TestTheTraces:
    def test_the_ingestion_runs_are_listed(self, dashboard: Path):
        app = _run("ingestion_traces")

        assert app.selectbox
        assert app.selectbox[0].options

    def test_the_steps_of_a_run_are_shown(self, dashboard: Path):
        app = _run("ingestion_traces")

        labels = [expander.label for expander in app.expander]
        assert any("load" in label for label in labels)

    def test_the_questions_are_listed(self, dashboard: Path):
        app = _run("query_traces")

        assert app.selectbox
        assert app.selectbox[0].options

    def test_the_two_routes_of_a_question_are_shown(self, dashboard: Path):
        app = _run("query_traces")

        assert any("Dense" in item.value for item in app.markdown)
        assert any("Sparse" in item.value for item in app.markdown)

    def test_the_reranker_is_shown(self, dashboard: Path):
        app = _run("query_traces")

        labels = [metric.label for metric in app.metric]
        assert "Candidates" in labels
        assert "Provider" in labels


class TestTheEvaluationPanel:
    def test_the_golden_test_sets_are_offered(self, dashboard: Path):
        app = _run("evaluation_panel")

        assert app.selectbox
        assert app.selectbox[0].options

    def test_the_run_can_be_started(self, dashboard: Path):
        app = _run("evaluation_panel")

        assert any(button.label == "Run the evaluation" for button in app.button)
