"""End-to-end tests for the ingestion script.

The script is run as it is run in real life: as a command, with a
configuration file of its own. The only thing that is replaced is the
embedding provider, which a small local server stands in for, so the test
needs no credentials and no network.
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
SCRIPT = REPO_ROOT / "scripts" / "ingest.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "sample_documents"


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
            "collection_name": "e2e",
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


def run_script(*arguments: str) -> subprocess.CompletedProcess:
    """Run the ingestion script and wait for it to finish."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
    )


class TestSingleDocument:
    def test_the_document_is_taken_in(self, config_file: Path):
        result = run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_products_are_in_the_data_folders(self, config_file: Path, tmp_path: Path):
        run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert (tmp_path / "chroma").exists()
        assert (tmp_path / "bm25" / "e2e_bm25.json").exists()
        assert (tmp_path / "image_index.db").exists()

    def test_the_script_says_what_it_did(self, config_file: Path):
        result = run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert "SUMMARY" in result.stdout
        assert "Chunks:" in result.stdout

    def test_the_document_with_images_is_taken_in(self, config_file: Path, tmp_path: Path):
        result = run_script(
            "--path", str(FIXTURES / "with_images.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert list((tmp_path / "images").rglob("*.png"))


class TestRepeatedRun:
    def test_the_second_run_skips_the_document(self, config_file: Path):
        run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        result = run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert result.returncode == 0
        assert "[SKIP]" in result.stdout

    def test_the_second_run_leaves_the_index_as_it_was(
        self, config_file: Path, tmp_path: Path
    ):
        run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
        )
        index = tmp_path / "bm25" / "e2e_bm25.json"
        before = index.read_text(encoding="utf-8")

        run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert index.read_text(encoding="utf-8") == before

    def test_forcing_takes_the_document_in_again(self, config_file: Path):
        run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        result = run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
            "--force",
        )

        assert result.returncode == 0
        assert "[SKIP]" not in result.stdout


class TestFolder:
    def test_every_document_of_the_folder_is_taken_in(self, config_file: Path):
        result = run_script(
            "--path", str(FIXTURES),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count("[OK]") >= 2

    def test_a_dry_run_takes_nothing_in(self, config_file: Path, tmp_path: Path):
        result = run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(config_file),
            "--dry-run",
        )

        assert result.returncode == 0
        assert not (tmp_path / "bm25").exists()


class TestFailure:
    def test_a_path_that_is_not_there_is_reported(self, config_file: Path, tmp_path: Path):
        result = run_script(
            "--path", str(tmp_path / "nothing"),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert result.returncode == 2
        assert "[FAIL]" in result.stdout

    def test_a_configuration_that_is_not_there_is_reported(self, tmp_path: Path):
        result = run_script(
            "--path", str(FIXTURES / "simple.pdf"),
            "--collection", "e2e",
            "--config", str(tmp_path / "nothing.yaml"),
        )

        assert result.returncode == 2
        assert "[FAIL]" in result.stdout

    def test_a_file_that_is_not_a_pdf_is_reported(self, config_file: Path, tmp_path: Path):
        notes = tmp_path / "notes.txt"
        notes.write_text("just some text", encoding="utf-8")

        result = run_script(
            "--path", str(notes),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert result.returncode == 2
        assert "[FAIL]" in result.stdout

    def test_a_broken_document_is_reported(self, config_file: Path, tmp_path: Path):
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"this is not a pdf")

        result = run_script(
            "--path", str(broken),
            "--collection", "e2e",
            "--config", str(config_file),
        )

        assert result.returncode == 2
        assert "[FAIL]" in result.stdout
