"""Logging helper.

stdout carries MCP protocol traffic, so every log record is written to stderr.

The trace logger is the other kind of output: it writes JSON Lines, one
object per line, so that a file of traces can be read back one run at a time.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.settings import resolve_path

LOGGER_NAME = "modular_rag"
TRACE_LOGGER_NAME = "modular_rag.trace"
DEFAULT_TRACES_PATH = "logs/traces.jsonl"

# The fields of a log record that belong to logging itself; everything else
# was attached by the caller and belongs in the JSON output.
_INTERNAL_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def get_logger(name: str = LOGGER_NAME, log_level: str = "INFO") -> logging.Logger:
    """Return a logger that writes to stderr.

    Args:
        name: Logger name; use a child name to narrow the scope.
        log_level: Level name such as "DEBUG" or "INFO".
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)

    return logger


class JSONFormatter(logging.Formatter):
    """Format a log record as one line of JSON.

    The line always carries the time, the level, the name of the logger and
    the message. Whatever the caller attached to the record with `extra` is
    put into the same object.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as one line of JSON.

        Args:
            record: The record to format.

        Returns:
            The JSON text of the record.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _INTERNAL_RECORD_FIELDS or key in payload:
                continue
            payload[key] = value if _is_jsonable(value) else str(value)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _is_jsonable(value: Any) -> bool:
    """Say whether a value can be written into a JSON object as it is."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def get_trace_logger(
    traces_path: str | Path = DEFAULT_TRACES_PATH,
    name: str = TRACE_LOGGER_NAME,
) -> logging.Logger:
    """Return a logger that writes JSON Lines to a file.

    Args:
        traces_path: Where the lines are written; the folder is made when it
            is not there yet.
        name: Logger name.

    Returns:
        A logger whose records are written as JSON Lines.
    """
    path = resolve_path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    wanted = str(path.resolve())
    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == wanted
        for handler in logger.handlers
    ):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


def write_trace(
    trace_dict: dict[str, Any],
    traces_path: str | Path = DEFAULT_TRACES_PATH,
) -> None:
    """Append one trace to the traces file as a line of JSON.

    Args:
        trace_dict: The trace, as `TraceContext.to_dict()` gives it.
        traces_path: Where the line is written; the folder is made when it is
            not there yet.

    Raises:
        TypeError: When the trace cannot be written as JSON.
    """
    path = resolve_path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(trace_dict, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
