"""The `router runtime` command group: manual/offline testing of the routing decision from the
command line — no HTTP service, no host agent in front of it. See
docs/specs/2026-08-10-python-runtime-implementation-plan.md (Phase 4).

Not yet implemented.
"""
from __future__ import annotations

import typer

app = typer.Typer(help="The online routing runtime: embed one prompt, assign it to a cluster, select a model.")
