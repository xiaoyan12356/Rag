"""End-to-end tests for the query script.

The script is run as it is run in real life: as a command, with a
configuration file of its own, over documents that were ingested before it.
The only thing that is replaced is the embedding provider, which a small
local server stands in for, so the test needs no credentials and no network.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from stub_embedding_server import DIMENSION, start_stub_server, stop_stub_server

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
INGEST_SCRIPT = REPO_ROOT / "scripts" / "ingest.py"
QUERY_SCRIPT = REPO_ROOT / "scripts" / "query.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "sample_documents"
COLLECTION = "e2e"


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
        "observability": {"log_level": "INFO"},
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
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def run_script(script: Path, *arguments: str) -> subprocess.CompletedProcess:
    """Run a script and wait for it to finish."""
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
    )


@pytest.fixture
def ingested(config_file: Path):
    """Take the sample documents in before the search runs."""
    result = run_script(
        INGEST_SCRIPT,
        "--path",
        str(FIXTURES),
        "--collection",
        COLLECTION,
        "--config",
        str(config_file),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return config_file


class TestQuery:
    def test_the_search_finds_something(self, ingested):
        result = run_script(
            QUERY_SCRIPT,
            "--query",
            "how is the document split into chunks",
            "--collection",
            COLLECTION,
            "--config",
            str(ingested),
        )

        assert result.returncode == 0
        assert "RESULTS" in result.stdout

    def test_the_results_are_printed(self, ingested):
        result = run_script(
            QUERY_SCRIPT,
            "--query",
            "chunking",
            "--collection",
            COLLECTION,
            "--config",
            str(ingested),
        )

        assert "score=" in result.stdout
        assert ".pdf" in result.stdout

    def test_the_number_of_results_can_be_capped(self, ingested):
        result = run_script(
            QUERY_SCRIPT,
            "--query",
            "chunking",
            "--top-k",
            "2",
            "--collection",
            COLLECTION,
            "--config",
            str(ingested),
        )

        assert result.stdout.count("score=") <= 2

    def test_the_collection_is_named(self, ingested):
        result = run_script(
            QUERY_SCRIPT,
            "--query",
            "chunking",
            "--collection",
            COLLECTION,
            "--config",
            str(ingested),
        )

        assert COLLECTION in result.stdout

    def test_the_verbose_mode_shows_every_stage(self, ingested):
        result = run_script(
            QUERY_SCRIPT,
            "--query",
            "chunking",
            "--verbose",
            "--collection",
            COLLECTION,
            "--config",
            str(ingested),
        )

        assert "Dense results" in result.stdout
        assert "Sparse results" in result.stdout
        assert "Fused results" in result.stdout

    def test_the_reranking_can_be_skipped(self, ingested):
        result = run_script(
            QUERY_SCRIPT,
            "--query",
            "chunking",
            "--no-rerank",
            "--collection",
            COLLECTION,
            "--config",
            str(ingested),
        )

        assert result.returncode == 0


class TestFailures:
    def test_a_collection_that_was_never_filled_gives_a_hint(self, ingested):
        result = run_script(
            QUERY_SCRIPT,
            "--query",
            "chunking",
            "--collection",
            "empty_collection",
            "--config",
            str(ingested),
        )

        assert result.returncode == 1
        assert "Nothing was found" in result.stdout
        assert "scripts/ingest.py" in result.stdout

    def test_a_configuration_that_is_not_there_is_reported(self, tmp_path):
        result = run_script(
            QUERY_SCRIPT,
            "--query",
            "chunking",
            "--config",
            str(tmp_path / "missing.yaml"),
        )

        assert result.returncode == 2
        assert "configuration could not be read" in result.stdout

    def test_a_question_is_needed(self):
        result = run_script(QUERY_SCRIPT)

        assert result.returncode != 0
