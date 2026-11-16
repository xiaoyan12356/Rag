"""Trace collector.

The collector is the bridge between a trace that is still in memory and the
JSON Lines file the dashboard reads. It takes a finished trace and appends
one line for it, so the file is a list of runs in the order they happened.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.settings import Settings, resolve_path
from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

DEFAULT_TRACES_PATH = "logs/traces.jsonl"


class TraceCollector:
    """Collect finished traces and append them to a JSON Lines file.

    Args:
        traces_path: Where the traces are written. A relative path is
            resolved against the repository root, so the file is the same
            one whatever the working directory is. The folder is made when
            it is not there yet.
    """

    def __init__(self, traces_path: str | Path = DEFAULT_TRACES_PATH) -> None:
        """Set up the collector."""
        self.path = resolve_path(traces_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> None:
        """Write a trace out as one line of JSON.

        A trace that was not finished yet is finished here, so the line
        always says how long the run took.

        Args:
            trace: The trace to write out.
        """
        if trace.finished_at is None:
            trace.finish()

        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            logger.warning("The trace %s could not be written: %s", trace.trace_id, exc)


def collect_trace(trace: TraceContext, settings: Settings) -> bool:
    """Write a finished run out, so the dashboard can show it.

    Where the trace goes and whether it is written at all come from the
    configuration, so every run of the pipeline is traced the same way.

    Args:
        trace: The run that just finished.
        settings: The configuration of the project.

    Returns:
        Whether the trace was written.
    """
    if not settings.observability.trace_enabled:
        return False

    path = settings.observability.trace_file or DEFAULT_TRACES_PATH
    TraceCollector(path).collect(trace)
    return True
