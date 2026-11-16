"""Trace service for the dashboard.

The logger writes every finished run to `logs/traces.jsonl` as one line of
JSON. This service reads that file and turns the lines into the traces the
pages show: the runs of one kind, the steps of a run and how long each step
took.

The file is written while the dashboard reads it, so a line can be half
written and a run can be in the middle of being appended. Both are handled
here: a line that cannot be read is left out rather than stopping the page.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.settings import DEFAULT_SETTINGS_PATH, SettingsError, load_settings, resolve_path

logger = logging.getLogger(__name__)

DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")
TRACE_TYPE_INGESTION = "ingestion"
TRACE_TYPE_QUERY = "query"


@dataclass
class TraceStage:
    """One step of a run."""

    name: str
    elapsed_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class Trace:
    """One run of the pipeline or of a question."""

    trace_id: str
    trace_type: str = TRACE_TYPE_INGESTION
    started_at: str = ""
    finished_at: str | None = None
    total_elapsed_ms: float = 0.0
    stages: list[TraceStage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Return the text that names this run in a list."""
        if self.trace_type == TRACE_TYPE_QUERY:
            query = self.metadata.get("query") or self.metadata.get("original_query")
            if query:
                return str(query)
        source = self.metadata.get("file_path") or self.metadata.get("source_path")
        if source:
            return str(source)
        return self.trace_id

    def stage(self, name: str) -> TraceStage | None:
        """Return the step with this name, or None when it never ran."""
        for entry in self.stages:
            if entry.name == name:
                return entry
        return None

    def stage_data(self, name: str) -> dict[str, Any]:
        """Return what the step with this name recorded."""
        entry = self.stage(name)
        return entry.data if entry is not None else {}


class TraceService:
    """Read the traces out of the JSON Lines file.

    Args:
        traces_path: The file the traces are read from; the default one is
            used when this is left out.
    """

    def __init__(self, traces_path: str | Path | None = None) -> None:
        """Set up the service; the file is read when it is first needed."""
        self.traces_path = Path(traces_path) if traces_path else _configured_traces_path()
        self._traces: list[Trace] | None = None

    def reload(self) -> None:
        """Read the file again, so a run that just finished is seen."""
        self._traces = None

    def list_traces(
        self,
        trace_type: str | None = None,
        limit: int | None = None,
    ) -> list[Trace]:
        """Return the recorded runs, newest first.

        Args:
            trace_type: The kind of run to return, either `ingestion` or
                `query`; every kind is returned when this is left out.
            limit: How many runs to return at most; every run is returned
                when this is left out.

        Returns:
            The runs, newest first.
        """
        traces = self._load()
        if trace_type:
            traces = [trace for trace in traces if trace.trace_type == trace_type]
        if limit is not None and limit >= 0:
            traces = traces[:limit]
        return traces

    def get_trace(self, trace_id: str) -> Trace | None:
        """Return the run with this id, or None when it is not known."""
        for trace in self._load():
            if trace.trace_id == trace_id:
                return trace
        return None

    def search(self, keyword: str, trace_type: str | None = None) -> list[Trace]:
        """Return the runs whose text holds the keyword.

        The keyword is looked for in the query of a run and in the path of
        the file it ingested, which are the two things a person remembers
        about a run.

        Args:
            keyword: The text to look for; an empty keyword matches every
                run.
            trace_type: The kind of run to look at.

        Returns:
            The runs that match, newest first.
        """
        traces = self.list_traces(trace_type=trace_type)
        needle = keyword.strip().lower()
        if not needle:
            return traces
        return [
            trace
            for trace in traces
            if needle in trace.label.lower()
            or needle in json.dumps(trace.metadata, ensure_ascii=False).lower()
        ]

    def get_stage_timings(self, trace: Trace) -> list[dict[str, Any]]:
        """Return the steps of a run with how long each took.

        The steps come back in the order they happened, so they can be drawn
        as a waterfall.
        """
        return [
            {
                "stage_name": stage.name,
                "elapsed_ms": stage.elapsed_ms,
                "data": dict(stage.data),
            }
            for stage in trace.stages
        ]

    def _load(self) -> list[Trace]:
        """Read every trace out of the file.

        A file that is not there yet has no traces, and a line that cannot
        be read is left out: the file is written while it is read, so its
        last line can be half written.
        """
        if self._traces is not None:
            return self._traces

        traces: list[Trace] = []
        try:
            if self.traces_path.exists():
                traces = self._parse(self.traces_path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("Could not read the traces at %s: %s", self.traces_path, exc)
            traces = []

        traces.sort(key=lambda trace: trace.started_at, reverse=True)
        self._traces = traces
        return traces

    def _parse(self, content: str) -> list[Trace]:
        """Turn the lines of the file into traces.

        A line that is not valid JSON is left out, which is what happens to
        the last line while it is being written.
        """
        traces: list[Trace] = []
        for number, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                logger.debug("Skipping the unreadable trace at line %d", number)
                continue
            if not isinstance(payload, dict):
                continue
            traces.append(_to_trace(payload))
        return traces


def _configured_traces_path() -> Path:
    """Return the trace file the configuration names.

    The configuration decides where the runs are written, so the dashboard
    reads them from the same place instead of from a path of its own. A
    configuration that cannot be read leaves the default in place.
    """
    try:
        settings = load_settings(DEFAULT_SETTINGS_PATH)
    except SettingsError as exc:
        logger.warning("The configuration could not be read: %s", exc)
        return DEFAULT_TRACES_PATH

    configured = settings.observability.trace_file
    if not configured:
        return DEFAULT_TRACES_PATH
    return resolve_path(configured)


def _to_trace(payload: dict[str, Any]) -> Trace:
    """Turn the dictionary of a line into a trace."""
    stages: list[TraceStage] = []
    raw_stages = payload.get("stages")
    if isinstance(raw_stages, list):
        for entry in raw_stages:
            if not isinstance(entry, dict):
                continue
            data = entry.get("data")
            stages.append(
                TraceStage(
                    name=str(entry.get("stage", "")),
                    elapsed_ms=_as_float(entry.get("elapsed_ms")),
                    data=dict(data) if isinstance(data, dict) else {},
                    timestamp=str(entry.get("timestamp", "")),
                )
            )

    metadata = payload.get("metadata")
    return Trace(
        trace_id=str(payload.get("trace_id", "")),
        trace_type=str(payload.get("trace_type", TRACE_TYPE_INGESTION)),
        started_at=str(payload.get("started_at", "")),
        finished_at=payload.get("finished_at"),
        total_elapsed_ms=_as_float(payload.get("total_elapsed_ms")),
        stages=stages,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _as_float(value: Any) -> float:
    """Return the value as a number, and zero when it is not one."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
