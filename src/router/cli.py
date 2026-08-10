"""Top-level `router` command: `router pipeline <cmd>` for the offline pipeline, `router runtime
<cmd>` for the online routing runtime. See pipeline/cli.py and runtime/cli.py respectively."""
from __future__ import annotations

import typer

from .pipeline.cli import app as pipeline_app
from .runtime.cli import app as runtime_app

app = typer.Typer(help="Embedding-clustering based LLM router: offline pipeline + online runtime.")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(runtime_app, name="runtime")


def main() -> None:
    app()
