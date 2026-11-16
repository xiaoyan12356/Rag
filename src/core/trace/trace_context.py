"""Trace context.

A trace follows one run through the pipeline: it names the run, and records
what each stage did and how long it took. The stages are kept in the order
they happened, so a slow run can be explained after the fact.

The collector that writes traces out is added in phase F; for now a trace is
something a caller holds on to.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

TraceType = Literal["query", "ingestion"]


@dataclass
class TraceContext:
    """What one run of the pipeline did."""

    trace_type: TraceType = "ingestion"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    _started: float = field(default_factory=time.monotonic, repr=False)
    _finished: float | None = field(default=None, repr=False)

    def record_stage(
        self,
        stage_name: str,
        data: dict[str, Any],
        elapsed_ms: float | None = None,
    ) -> None:
        """Record what a stage did.

        The stages are kept in the order they were recorded, so the trace
        reads like the run did.
        """
        entry: dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
        self.stages.append(entry)

    def get_stage_data(self, stage_name: str) -> dict[str, Any] | None:
        """Return what a stage recorded, or None when it never ran."""
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None

    def finish(self) -> None:
        """Mark the run as over."""
        self._finished = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def elapsed_ms(self) -> float:
        """Return how long the run took, in milliseconds."""
        end = self._finished if self._finished is not None else time.monotonic()
        return (end - self._started) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Return the trace as a plain dictionary."""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": round(self.elapsed_ms(), 2),
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }
