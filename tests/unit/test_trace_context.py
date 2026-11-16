"""Unit tests for the trace context.

The trace is the record of one run: it names the run, keeps the stages in the
order they happened, says how long each of them took, and can be written out
as JSON. These tests pin that behaviour.
"""

from __future__ import annotations

import json

import pytest

from core.settings import ObservabilitySettings, Settings
from core.trace.trace_collector import TraceCollector, collect_trace
from core.trace.trace_context import TraceContext


class TestTraceType:
    def test_a_new_trace_is_a_query(self):
        trace = TraceContext()

        assert trace.trace_type == "query"

    def test_the_type_can_be_set_to_ingestion(self):
        trace = TraceContext(trace_type="ingestion")

        assert trace.trace_type == "ingestion"

    def test_every_trace_has_its_own_id(self):
        first = TraceContext()
        second = TraceContext()

        assert first.trace_id != second.trace_id


class TestRecording:
    def test_a_stage_is_recorded(self):
        trace = TraceContext()

        trace.record_stage("load", {"doc_id": "doc_1"})

        assert trace.get_stage_data("load") == {"doc_id": "doc_1"}

    def test_the_stages_keep_the_order_they_happened_in(self):
        trace = TraceContext()

        trace.record_stage("load", {})
        trace.record_stage("split", {})
        trace.record_stage("transform", {})

        assert [entry["stage"] for entry in trace.stages] == [
            "load",
            "split",
            "transform",
        ]

    def test_the_time_of_a_stage_is_kept(self):
        trace = TraceContext()

        trace.record_stage("load", {}, elapsed_ms=12.345)

        assert trace.get_stage("load")["elapsed_ms"] == 12.35

    def test_a_stage_without_a_time_has_none(self):
        trace = TraceContext()

        trace.record_stage("load", {})

        assert "elapsed_ms" not in trace.get_stage("load")

    def test_a_stage_that_never_ran_is_not_found(self):
        trace = TraceContext()

        assert trace.get_stage_data("load") is None

    def test_the_last_record_of_a_stage_wins(self):
        trace = TraceContext()

        trace.record_stage("load", {"attempt": 1})
        trace.record_stage("load", {"attempt": 2})

        assert trace.get_stage_data("load") == {"attempt": 2}

    def test_the_fields_of_a_stage_can_be_added_later(self):
        trace = TraceContext()

        trace.record_stage("load", {"doc_id": "doc_1"})
        trace.annotate_stage("load", method="markitdown", provider="builtin")

        data = trace.get_stage_data("load")
        assert data["method"] == "markitdown"
        assert data["provider"] == "builtin"
        assert data["doc_id"] == "doc_1"

    def test_a_stage_that_never_ran_is_not_annotated(self):
        trace = TraceContext()

        trace.annotate_stage("load", method="markitdown")

        assert trace.stages == []


class TestTiming:
    def test_the_time_of_the_whole_run_is_measured(self):
        trace = TraceContext()

        trace.finish()

        assert trace.elapsed_ms() >= 0.0

    def test_the_time_of_a_stage_is_measured(self):
        trace = TraceContext()

        trace.record_stage("load", {}, elapsed_ms=25.0)

        assert trace.elapsed_ms("load") == 25.0

    def test_the_time_of_a_stage_without_a_time_is_not_found(self):
        trace = TraceContext()

        with pytest.raises(KeyError, match="load"):
            trace.elapsed_ms("load")

    def test_a_finished_run_has_a_finish_time(self):
        trace = TraceContext()

        trace.finish()

        assert trace.finished_at is not None

    def test_a_run_that_is_not_finished_has_no_finish_time(self):
        trace = TraceContext()

        assert trace.finished_at is None


class TestSerialization:
    def test_the_trace_is_a_dictionary(self):
        trace = TraceContext(trace_type="ingestion")

        assert isinstance(trace.to_dict(), dict)

    def test_the_dictionary_names_the_run(self):
        trace = TraceContext(trace_type="ingestion")

        data = trace.to_dict()

        assert data["trace_id"] == trace.trace_id
        assert data["trace_type"] == "ingestion"

    def test_the_dictionary_holds_the_stages(self):
        trace = TraceContext()
        trace.record_stage("load", {"doc_id": "doc_1"})
        trace.finish()

        data = trace.to_dict()

        assert len(data["stages"]) == 1
        assert data["stages"][0]["stage"] == "load"

    def test_the_dictionary_holds_the_times(self):
        trace = TraceContext()
        trace.finish()

        data = trace.to_dict()

        assert data["finished_at"] is not None
        assert data["total_elapsed_ms"] >= 0.0

    def test_the_dictionary_can_be_written_as_json(self):
        trace = TraceContext(trace_type="ingestion")
        trace.record_stage("split", {"chunk_count": 3}, elapsed_ms=1.5)
        trace.finish()

        text = json.dumps(trace.to_dict())

        assert json.loads(text)["trace_type"] == "ingestion"

    def test_the_metadata_is_kept(self):
        trace = TraceContext()
        trace.metadata["collection"] = "kb"

        assert trace.to_dict()["metadata"]["collection"] == "kb"


class TestCollector:
    def test_the_trace_is_written_to_the_file(self, tmp_path):
        collector = TraceCollector(tmp_path / "traces.jsonl")
        trace = TraceContext(trace_type="ingestion")

        collector.collect(trace)

        assert (tmp_path / "traces.jsonl").exists()

    def test_the_line_is_valid_json(self, tmp_path):
        collector = TraceCollector(tmp_path / "traces.jsonl")
        trace = TraceContext(trace_type="ingestion")

        collector.collect(trace)

        payload = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
        assert payload["trace_type"] == "ingestion"

    def test_the_file_grows_by_one_line_per_trace(self, tmp_path):
        collector = TraceCollector(tmp_path / "traces.jsonl")

        collector.collect(TraceContext())
        collector.collect(TraceContext())

        lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_a_trace_that_was_not_finished_is_finished(self, tmp_path):
        collector = TraceCollector(tmp_path / "traces.jsonl")
        trace = TraceContext()

        collector.collect(trace)

        assert trace.finished_at is not None


class TestCollectTrace:
    """The helper that writes a run out when the configuration asks for it."""

    def test_the_trace_is_written_to_the_configured_file(self, tmp_path):
        settings = Settings(
            observability=ObservabilitySettings(trace_file=str(tmp_path / "traces.jsonl"))
        )

        written = collect_trace(TraceContext(trace_type="query"), settings)

        assert written is True
        assert (tmp_path / "traces.jsonl").exists()

    def test_nothing_is_written_when_traces_are_switched_off(self, tmp_path):
        settings = Settings(
            observability=ObservabilitySettings(
                trace_enabled=False, trace_file=str(tmp_path / "traces.jsonl")
            )
        )

        written = collect_trace(TraceContext(trace_type="query"), settings)

        assert written is False
        assert not (tmp_path / "traces.jsonl").exists()

    def test_the_folder_of_the_trace_file_is_made(self, tmp_path):
        settings = Settings(
            observability=ObservabilitySettings(
                trace_file=str(tmp_path / "logs" / "traces.jsonl")
            )
        )

        collect_trace(TraceContext(), settings)

        assert (tmp_path / "logs" / "traces.jsonl").exists()
