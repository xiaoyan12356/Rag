"""Unit tests for the dashboard trace service.

The service reads the file the logger writes, and the file is written while
the dashboard reads it. The tests pin that a file that is not there yet gives
no traces, that a half-written line is left out instead of raising, and that
the runs come back newest first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from observability.dashboard.services.trace_service import (
    DEFAULT_TRACES_PATH,
    TRACE_TYPE_INGESTION,
    TRACE_TYPE_QUERY,
    TraceService,
)


def _write(path: Path, payloads: list[dict]) -> Path:
    """Write the payloads as the lines of a trace file."""
    lines = [json.dumps(payload) for payload in payloads]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _trace(trace_id: str, trace_type: str, started_at: str, **metadata) -> dict:
    """Return the dictionary of one trace."""
    return {
        "trace_id": trace_id,
        "trace_type": trace_type,
        "started_at": started_at,
        "finished_at": started_at,
        "total_elapsed_ms": 120.0,
        "stages": [
            {
                "stage": "load",
                "timestamp": started_at,
                "data": {"image_count": 3},
                "elapsed_ms": 42.5,
            }
        ],
        "metadata": metadata,
    }


@pytest.fixture
def traces_file(tmp_path: Path) -> Path:
    """Return the path of a trace file that does not exist yet."""
    return tmp_path / "traces.jsonl"


class TestMissingFile:
    def test_a_file_that_is_not_there_gives_no_traces(self, traces_file: Path):
        service = TraceService(traces_file)

        assert service.list_traces() == []

    def test_a_trace_that_is_not_there_is_not_found(self, traces_file: Path):
        service = TraceService(traces_file)

        assert service.get_trace("nothing") is None


class TestReadingTraces:
    def test_the_traces_of_the_file_are_read(self, traces_file: Path):
        _write(
            traces_file,
            [_trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00")],
        )
        service = TraceService(traces_file)

        assert [trace.trace_id for trace in service.list_traces()] == ["a"]

    def test_the_newest_run_comes_first(self, traces_file: Path):
        _write(
            traces_file,
            [
                _trace("old", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00"),
                _trace("new", TRACE_TYPE_INGESTION, "2026-11-16T12:00:00+00:00"),
            ],
        )
        service = TraceService(traces_file)

        assert [trace.trace_id for trace in service.list_traces()] == ["new", "old"]

    def test_only_the_runs_of_one_kind_are_returned(self, traces_file: Path):
        _write(
            traces_file,
            [
                _trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00"),
                _trace("b", TRACE_TYPE_QUERY, "2026-11-16T11:00:00+00:00"),
            ],
        )
        service = TraceService(traces_file)

        assert [trace.trace_id for trace in service.list_traces(TRACE_TYPE_QUERY)] == ["b"]

    def test_the_runs_are_limited_to_the_number_that_is_asked_for(self, traces_file: Path):
        _write(
            traces_file,
            [
                _trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00"),
                _trace("b", TRACE_TYPE_INGESTION, "2026-11-16T11:00:00+00:00"),
            ],
        )
        service = TraceService(traces_file)

        assert len(service.list_traces(limit=1)) == 1


class TestBrokenFile:
    def test_a_half_written_line_is_left_out(self, traces_file: Path):
        good = json.dumps(_trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00"))
        traces_file.write_text(good + "\n" + '{"trace_id": "b", "sta', encoding="utf-8")
        service = TraceService(traces_file)

        assert [trace.trace_id for trace in service.list_traces()] == ["a"]

    def test_an_empty_line_is_left_out(self, traces_file: Path):
        good = json.dumps(_trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00"))
        traces_file.write_text(good + "\n\n   \n", encoding="utf-8")
        service = TraceService(traces_file)

        assert len(service.list_traces()) == 1


class TestTraceContents:
    def test_the_steps_of_the_run_are_read(self, traces_file: Path):
        _write(
            traces_file,
            [_trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00")],
        )
        service = TraceService(traces_file)

        trace = service.get_trace("a")

        assert trace is not None
        assert [stage.name for stage in trace.stages] == ["load"]
        assert trace.stages[0].elapsed_ms == 42.5

    def test_the_metadata_of_the_run_is_read(self, traces_file: Path):
        _write(
            traces_file,
            [
                _trace(
                    "a",
                    TRACE_TYPE_INGESTION,
                    "2026-11-16T10:00:00+00:00",
                    file_path="documents/a.pdf",
                )
            ],
        )
        service = TraceService(traces_file)

        trace = service.get_trace("a")

        assert trace is not None
        assert trace.metadata["file_path"] == "documents/a.pdf"
        assert trace.label == "documents/a.pdf"

    def test_the_timings_of_the_steps_are_listed(self, traces_file: Path):
        _write(
            traces_file,
            [_trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00")],
        )
        service = TraceService(traces_file)
        trace = service.get_trace("a")

        assert trace is not None
        timings = service.get_stage_timings(trace)

        assert timings[0]["stage_name"] == "load"
        assert timings[0]["elapsed_ms"] == 42.5


class TestSearch:
    def test_a_run_is_found_by_the_file_it_ingested(self, traces_file: Path):
        _write(
            traces_file,
            [
                _trace(
                    "a",
                    TRACE_TYPE_INGESTION,
                    "2026-11-16T10:00:00+00:00",
                    file_path="documents/report.pdf",
                )
            ],
        )
        service = TraceService(traces_file)

        assert [trace.trace_id for trace in service.search("report")] == ["a"]

    def test_a_run_is_found_by_its_question(self, traces_file: Path):
        _write(
            traces_file,
            [
                _trace(
                    "a",
                    TRACE_TYPE_QUERY,
                    "2026-11-16T10:00:00+00:00",
                    query="what is a vector store",
                )
            ],
        )
        service = TraceService(traces_file)

        assert [trace.trace_id for trace in service.search("VECTOR")] == ["a"]

    def test_a_run_that_does_not_match_is_not_returned(self, traces_file: Path):
        _write(
            traces_file,
            [_trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00")],
        )
        service = TraceService(traces_file)

        assert service.search("nothing") == []


class TestTheConfiguredFile:
    """The dashboard reads the file the configuration names."""

    def test_the_configured_file_is_read(self, tmp_path: Path, monkeypatch):
        traces = tmp_path / "logs" / "traces.jsonl"
        _write_config(tmp_path, monkeypatch, traces)
        _write(traces, [_trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00")])

        service = TraceService()

        assert [trace.trace_id for trace in service.list_traces()] == ["a"]

    def test_a_path_that_is_named_wins(self, tmp_path: Path, monkeypatch):
        traces = tmp_path / "logs" / "traces.jsonl"
        _write_config(tmp_path, monkeypatch, traces)
        _write(traces, [_trace("a", TRACE_TYPE_INGESTION, "2026-11-16T10:00:00+00:00")])

        service = TraceService(tmp_path / "other.jsonl")

        assert service.list_traces() == []

    def test_a_configuration_that_is_not_there_leaves_the_default(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        service = TraceService()

        assert service.traces_path == DEFAULT_TRACES_PATH


def _write_config(tmp_path: Path, monkeypatch, traces: Path) -> None:
    """Put a configuration in place that names the trace file."""
    config = {
        "llm": {"provider": "openai", "model": "gpt-4o"},
        "embedding": {"provider": "ollama", "model": "nomic", "dimensions": 8},
        "vector_store": {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
        },
        "retrieval": {"dense_top_k": 20, "sparse_top_k": 20, "fusion_top_k": 10},
        "rerank": {"provider": "none"},
        "evaluation": {"provider": "custom"},
        "observability": {"log_level": "INFO", "trace_file": str(traces)},
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    traces.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
