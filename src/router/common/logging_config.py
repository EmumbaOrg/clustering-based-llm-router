"""Deliberately plain: one console handler, one line per log call, no structured fields, no run
id. `log_file` is opt-in (see `configure_logging`) — stderr alone stays the default.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT_LOGGER_NAME = "router"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Idempotent — calling this again replaces the previous handlers rather than stacking
    duplicates. `log_file`, if given, is appended to (not overwritten) alongside stderr, so
    re-running a command doesn't erase an earlier run's log."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
