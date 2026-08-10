"""Logging setup, shared by both the pipeline and the runtime.

Deliberately plain: one console handler, one line per log call, no structured fields, no run id,
no log file. Modules just use the idiomatic `logger = logging.getLogger(__name__)` and call
`logger.info("what's happening")` / `logger.warning(...)` / `logger.error(...)`.
"""
from __future__ import annotations

import logging
import sys

_ROOT_LOGGER_NAME = "router"


def configure_logging(level: str = "INFO") -> None:
    """Configure the shared `router` logger once per CLI invocation. Idempotent — calling this
    again replaces the previous handler rather than stacking duplicates."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
