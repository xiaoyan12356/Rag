"""Unit tests for the JSON Lines logger.

The traces are written as one JSON object per line, so that a file of traces
can be read back one run at a time. These tests pin that the lines are valid
JSON, that they carry what a trace holds, and that the file grows by one line
per trace.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from observability.logger import JSONFormatter, get_trace_logger, write_trace


@pytest.fixture
def traces_file(tmp_path) -> Path:
    """Return the file the traces are written to."""
    return tmp_path / "logs" / "traces.jsonl"


@pytest.fixture
def trace_dict() -> dict:
    """Return a trace as `TraceContext.to_dict()` gives it."""
    return {
        "trace_id": "trace-1",
        "trace_type": "query",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "total_elapsed_ms": 1000.0,
        "stages": [{"stage": "load", "data": {"method": "markitdown"}}],
        "metadata": {},
    }


class TestJSONFormatter:
    def test_a_record_becomes_one_line_of_json(self):
        record = logging.LogRecord(
            name="modular_rag.trace",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="the trace was written",
            args=(),
            exc_info=None,
        )

        line = JSONFormatter().format(record)

        assert "\n" not in line
        assert json.loads(line)["message"] == "the trace was written"

    def test_the_line_says_the_time_the_level_and_the_logger(self):
        record = logging.LogRecord(
            name="modular_rag.trace",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="careful",
            args=(),
            exc_info=None,
        )

        payload = json.loads(JSONFormatter().format(record))

        assert payload["level"] == "WARNING"
        assert payload["logger"] == "modular_rag.trace"
        assert payload["timestamp"]

    def test_the_fields_the_caller_attached_are_kept(self):
        record = logging.LogRecord(
            name="modular_rag.trace",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="the trace was written",
            args=(),
            exc_info=None,
        )
        record.trace_type = "ingestion"

        payload = json.loads(JSONFormatter().format(record))

        assert payload["trace_type"] == "ingestion"

    def test_a_field_that_cannot_be_written_as_json_becomes_text(self):
        record = logging.LogRecord(
            name="modular_rag.trace",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="the trace was written",
            args=(),
            exc_info=None,
        )
        record.odd = object()

        payload = json.loads(JSONFormatter().format(record))

        assert isinstance(payload["odd"], str)

    def test_the_message_is_put_in_place_of_the_placeholder(self):
        record = logging.LogRecord(
            name="modular_rag.trace",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="the trace %s was written",
            args=("abc",),
            exc_info=None,
        )

        payload = json.loads(JSONFormatter().format(record))

        assert payload["message"] == "the trace abc was written"


class TestWriteTrace:
    def test_the_trace_is_written_to_the_file(self, traces_file, trace_dict):
        write_trace(trace_dict, traces_file)

        assert traces_file.exists()

    def test_the_file_holds_one_line_per_trace(self, traces_file, trace_dict):
        write_trace(trace_dict, traces_file)
        write_trace(trace_dict, traces_file)

        assert len(traces_file.read_text(encoding="utf-8").splitlines()) == 2

    def test_the_line_is_valid_json(self, traces_file, trace_dict):
        write_trace(trace_dict, traces_file)

        payload = json.loads(traces_file.read_text(encoding="utf-8").strip())

        assert payload["trace_id"] == "trace-1"

    def test_the_line_says_what_kind_of_trace_it_is(self, traces_file, trace_dict):
        write_trace(trace_dict, traces_file)

        payload = json.loads(traces_file.read_text(encoding="utf-8").strip())

        assert payload["trace_type"] == "query"

    def test_the_folder_is_made_when_it_is_not_there(self, tmp_path, trace_dict):
        target = tmp_path / "deep" / "down" / "traces.jsonl"

        write_trace(trace_dict, target)

        assert target.exists()

    def test_an_existing_trace_is_kept(self, traces_file, trace_dict):
        write_trace(trace_dict, traces_file)
        second = dict(trace_dict, trace_id="trace-2")
        write_trace(second, traces_file)

        lines = traces_file.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["trace_id"] == "trace-1"
        assert json.loads(lines[1])["trace_id"] == "trace-2"


class TestTraceLogger:
    def test_the_logger_writes_to_the_file(self, traces_file):
        logger = get_trace_logger(traces_file)
        logger.info("the trace was written")

        assert "the trace was written" in traces_file.read_text(encoding="utf-8")

    def test_the_line_the_logger_writes_is_valid_json(self, traces_file):
        logger = get_trace_logger(traces_file)
        logger.info("the trace was written")

        payload = json.loads(traces_file.read_text(encoding="utf-8").strip())

        assert payload["message"] == "the trace was written"

    def test_the_logger_is_not_asked_for_its_handlers_twice(self, traces_file):
        first = get_trace_logger(traces_file)
        second = get_trace_logger(traces_file)

        assert first is second
        assert len(first.handlers) == 1

    def test_the_logger_does_not_write_to_the_screen(self, traces_file, capsys):
        logger = get_trace_logger(traces_file)
        logger.info("the trace was written")

        assert "the trace was written" not in capsys.readouterr().err
