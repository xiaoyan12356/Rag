"""Trace context.

A trace follows one run through the pipeline: it names the run, and records
what each stage did and how long it took. The stages are kept in the order
they happened, so a slow run can be explained after the fact.

A trace knows whether it belongs to a query or to an ingestion, so the two
kinds of run can be told apart once they are written out.

The collector that writes traces out is added in phase F; a trace is
something a caller holds on to until it is collected.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

TraceType = Literal["query", "ingestion"]

TRACE_TYPES: tuple[str, ...] = ("query", "ingestion")


@dataclass
class TraceContext:
    """What one run of the pipeline did."""

    trace_type: TraceType = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    _started: float = field(default_factory=time.monotonic, repr=False)
    _finished: float | None = field(default=None, repr=False)
    _stage_timings: dict[str, float] = field(default_factory=dict, repr=False)

    def record_stage(
        self,
        stage_name: str,
        data: dict[str, Any],
        elapsed_ms: float | None = None,
    ) -> None:
        """Record what a stage did.

        The stages are kept in the order they were recorded, so the trace
        reads like the run did.

        Args:
            stage_name: The name of the stage, such as "dense_retrieval".
            data: What the stage did; it names the method and the provider
                that did it, and holds whatever detail the stage wants to
                keep.
            elapsed_ms: How long the stage took, in milliseconds.
        """
        entry: dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
            self._stage_timings[stage_name] = elapsed_ms
        self.stages.append(entry)

    def get_stage_data(self, stage_name: str) -> dict[str, Any] | None:
        """Return what a stage recorded, or None when it never ran."""
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None

    def get_stage(self, stage_name: str) -> dict[str, Any] | None:
        """Return the whole record of a stage, or None when it never ran."""
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry
        return None

    def annotate_stage(self, stage_name: str, **fields: Any) -> None:
        """Add fields to a stage that was already recorded.

        A stage that was not recorded yet is left alone, so a caller that
        times a stage cannot invent one.

        Args:
            stage_name: The stage to add the fields to.
            **fields: The fields to add; the data of the stage is where they
                go, unless the name of a field starts with an underscore.
        """
        entry = self.get_stage(stage_name)
        if entry is None:
            return
        data = entry.setdefault("data", {})
        for key, value in fields.items():
            if key.startswith("_"):
                entry[key.lstrip("_")] = value
            else:
                data[key] = value

    def finish(self) -> None:
        """Mark the run as over."""
        self._finished = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """Return how long the run took, in milliseconds.

        Args:
            stage_name: The stage to measure. The time of the whole run is
                returned when this is left out.

        Returns:
            The milliseconds the stage took, or the milliseconds the run took.

        Raises:
            KeyError: When the named stage recorded no time.
        """
        if stage_name is not None:
            if stage_name not in self._stage_timings:
                raise KeyError(f"The stage '{stage_name}' has no recorded time")
            return self._stage_timings[stage_name]

        end = self._finished if self._finished is not None else time.monotonic()
        return (end - self._started) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Return the trace as a plain dictionary.

        The result can be handed to ``json.dumps()`` as it is.
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": round(self.elapsed_ms(), 2),
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }
