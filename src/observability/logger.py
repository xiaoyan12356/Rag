"""Logging helper.

stdout carries MCP protocol traffic, so every log record is written to stderr.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "modular_rag"


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
