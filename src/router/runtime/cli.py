"""The `router runtime` command group: manual/offline testing of the routing decision from the
command line — no HTTP service, no host agent in front of it. See
docs/specs/2026-08-10-python-runtime-implementation-plan.md (Phase 4).
"""
from __future__ import annotations

import dataclasses
import json as json_lib
from pathlib import Path

import typer

from ..common.artifacts import ARTIFACTS_DIR
from .context import load_routing_context
from .decide import decide as run_decide

app = typer.Typer(help="The online routing runtime: embed one prompt, assign it to a cluster, select a model.")

DEFAULT_CLUSTER_MAP_PATH = ARTIFACTS_DIR / "cluster-map.json"
DEFAULT_MODEL_PROFILES_PATH = ARTIFACTS_DIR / "model-profiles.json"


@app.command()
def decide(
    prompt: str = typer.Option(..., help="The prompt to route."),
    lambda_: float = typer.Option(
        ..., "--lambda", help="Cost-weight lambda — no default; see config/calibration.yaml's lambda_sweep."
    ),
    cluster_map: str | None = typer.Option(None, help="Path to cluster-map.json. Defaults to artifacts/cluster-map.json."),
    model_profiles: str | None = typer.Option(None, help="Path to model-profiles.json. Defaults to artifacts/model-profiles.json."),
    json_output: bool = typer.Option(False, "--json", help="Print the full decision as one JSON object."),
) -> None:
    """Embed PROMPT, assign it to a cluster, and print the selected model and score table."""
    cluster_map_path = Path(cluster_map) if cluster_map else DEFAULT_CLUSTER_MAP_PATH
    model_profiles_path = Path(model_profiles) if model_profiles else DEFAULT_MODEL_PROFILES_PATH
    ctx = load_routing_context(cluster_map_path, model_profiles_path, lambda_)
    decision = run_decide(prompt, ctx)

    if json_output:
        typer.echo(json_lib.dumps(dataclasses.asdict(decision), indent=2))
        return

    typer.echo(
        f"cluster {decision.cluster.cluster_id} "
        f"(distance={decision.cluster.distance:.4f}, runner-up={decision.cluster.runner_up_cluster_id})"
    )
    typer.echo(
        f"selected {decision.selected.model_id} "
        f"(routing_score={decision.selected.routing_score:.4f}, error_source={decision.selected.error_source})"
    )
    typer.echo(f"lambda={decision.lambda_} embed_ms={decision.embed_ms:.1f} score_ms={decision.score_ms:.2f}")
    typer.echo(f"digest: {decision.digest}")
    typer.echo("\nmodel_id             predicted_error  normalised_cost  routing_score")
    for s in sorted(decision.scores, key=lambda s: s.routing_score):
        typer.echo(f"{s.model_id:<20} {s.predicted_error:<16.4f} {s.normalised_cost:<16.4f} {s.routing_score:.4f}")
    if decision.excluded:
        typer.echo("\nexcluded:")
        for model_id, reason in decision.excluded.items():
            typer.echo(f"  {model_id}: {reason}")


@app.command()
def validate(
    cluster_map: str | None = typer.Option(None, help="Path to cluster-map.json. Defaults to artifacts/cluster-map.json."),
    model_profiles: str | None = typer.Option(None, help="Path to model-profiles.json. Defaults to artifacts/model-profiles.json."),
) -> None:
    """Check both artifacts against config/models.yaml's candidate roster. No embedding, no model
    download — the load itself is the check, so a clean exit means every hard-fail rule in
    runtime/context.py passed."""
    cluster_map_path = Path(cluster_map) if cluster_map else DEFAULT_CLUSTER_MAP_PATH
    model_profiles_path = Path(model_profiles) if model_profiles else DEFAULT_MODEL_PROFILES_PATH
    # lambda doesn't affect validation; load_routing_context requires one, so 0.0 is inert here.
    ctx = load_routing_context(cluster_map_path, model_profiles_path, lambda_=0.0)

    k = ctx.cluster_map.centroids.shape[0]
    typer.echo(f"cluster map:    {ctx.cluster_map_id} (k={k}, dim={ctx.cluster_map.dimensions})")
    typer.echo(f"model profiles: {ctx.profiles_id}")
    typer.echo(f"embedding:      {ctx.embedding.model_id} ({ctx.embedding.dimensions}d)")
    typer.echo(f"candidates:     {', '.join(sorted(m.model_id for m in ctx.candidates))}")
    typer.echo(f"digest:         {ctx.digest}")

    gaps = [
        (m.model_id, cid)
        for m in sorted(ctx.candidates, key=lambda m: m.model_id)
        for cid in range(k)
        if str(cid) not in ctx.profiles_by_model[m.model_id].get("clusters", {})
    ]
    if gaps:
        typer.echo("\ncoverage gaps (these fall back to the model's global rate at decision time):")
        for model_id, cid in gaps:
            typer.echo(f"  {model_id}: cluster {cid}")
    else:
        typer.echo("\nno coverage gaps — every candidate has cluster-level calibration for every cluster")
