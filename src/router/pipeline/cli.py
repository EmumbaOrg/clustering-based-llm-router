"""The `router pipeline` command group: `uv run router pipeline <command>`."""
from __future__ import annotations

import typer

from ..common.logging_config import configure_logging

app = typer.Typer(help="The offline pipeline: corpus -> embed -> cluster -> calibrate -> evaluate.")


@app.callback()
def main_callback(
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging verbosity (DEBUG, INFO, WARNING, ERROR)."
    ),
) -> None:
    """Configures logging once before any command runs."""
    configure_logging(level=log_level)
