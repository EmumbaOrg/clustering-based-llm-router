"""The `router pipeline` command group: `uv run router pipeline <command>`."""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime

import typer

from ..common import embedding as embed_mod
from ..common.assign import load_cluster_map
from ..common.config import (
    REPO_ROOT,
    load_calibration_config,
    load_clustering_config,
    load_embedding_config,
    load_models_config,
)
from ..common.logging_config import configure_logging
from . import corpus as corpus_mod
from . import evaluate as evaluate_mod
from .calibration import calibrate as calibrate_mod
from .calibration import profiles as profiles_mod
from .calibration.grading import bigcodebench as bigcodebench_grading
from .calibration.grading import ds1000 as ds1000_grading
from .calibration.grading import swesmith as swesmith_grading
from .calibration.tasks import load_gradeable_tasks
from .clustering import cluster as cluster_mod
from .clustering import cluster_map as cluster_map_mod
from .clustering import viz as viz_mod

app = typer.Typer(help="The offline pipeline: corpus -> embed -> cluster -> calibrate -> evaluate.")


@app.callback()
def main_callback(
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging verbosity (DEBUG, INFO, WARNING, ERROR)."
    ),
) -> None:
    """Configures logging once before any command runs."""
    configure_logging(level=log_level)

WORK_DIR = REPO_ROOT / ".cache"
CORPUS_PATH = WORK_DIR / "corpus.jsonl"
EMBEDDINGS_PATH = WORK_DIR / "embeddings.npz"
VIZ_PATH = WORK_DIR / "visualization.json"
CLUSTER_MAP_PATH = cluster_map_mod.ARTIFACTS_DIR / "cluster-map.json"
PROFILES_PATH = profiles_mod.ARTIFACTS_DIR / "model-profiles.json"


def _run_corpus(sample: int | None) -> None:
    rows, sources = corpus_mod.build_corpus(sample=sample)
    corpus_mod.write_corpus_jsonl(rows, CORPUS_PATH)
    typer.echo(f"Wrote {len(rows)} deduped rows to {CORPUS_PATH}")
    for s in sources:
        typer.echo(f"  {s.name}: {s.rows} rows ({s.license})")


@app.command()
def corpus(
    sample: int | None = typer.Option(
        None, help="Cap EACH source at this many rows (dry run). Omit for the full corpus."
    ),
) -> None:
    """Load, tag, and dedup the corpus from all configured sources; write corpus.jsonl."""
    _run_corpus(sample)


def _run_embed(sample: int | None) -> None:
    if not CORPUS_PATH.exists():
        raise typer.BadParameter(f"{CORPUS_PATH} not found — run `corpus` first.")
    rows = corpus_mod.read_corpus_jsonl(CORPUS_PATH)
    if sample is not None:
        rows = rows[:sample]
    config = load_embedding_config()
    vectors = embed_mod.embed_texts([r.text for r in rows], config)
    embed_mod.save_embeddings(EMBEDDINGS_PATH, [r.id for r in rows], vectors)
    typer.echo(f"Embedded {len(rows)} rows ({vectors.shape[1]}-dim) to {EMBEDDINGS_PATH}")


@app.command()
def embed(
    sample: int | None = typer.Option(
        None, help="Only embed the first N rows of corpus.jsonl (dry run)."
    ),
) -> None:
    """Embed corpus.jsonl; write embeddings.npz."""
    _run_embed(sample)


def _run_cluster() -> None:
    if not EMBEDDINGS_PATH.exists():
        raise typer.BadParameter(f"{EMBEDDINGS_PATH} not found — run `embed` first.")
    _, vectors = embed_mod.load_embeddings(EMBEDDINGS_PATH)
    config = load_clustering_config()
    results = cluster_mod.run_candidates(vectors, config)
    for k in sorted(results):
        typer.echo(cluster_mod.format_diagnostics(results[k].diagnostics, len(vectors)))


@app.command()
def cluster() -> None:
    """Run K-means for every candidate k in config/clustering.yaml; print diagnostics."""
    _run_cluster()


def _run_build_artifact(k: int | None) -> None:
    if not EMBEDDINGS_PATH.exists():
        raise typer.BadParameter(f"{EMBEDDINGS_PATH} not found — run `embed` first.")
    if not CORPUS_PATH.exists():
        raise typer.BadParameter(f"{CORPUS_PATH} not found — run `corpus` first.")

    ids, vectors = embed_mod.load_embeddings(EMBEDDINGS_PATH)
    rows_by_id = {r.id: r for r in corpus_mod.read_corpus_jsonl(CORPUS_PATH)}
    used_rows = [rows_by_id[i] for i in ids if i in rows_by_id]

    clustering_config = load_clustering_config()
    embedding_config = load_embedding_config()
    chosen_k = k or clustering_config.default_k

    result = cluster_mod.run_kmeans(vectors, chosen_k, clustering_config.seed, clustering_config.n_init)
    typer.echo(cluster_mod.format_diagnostics(result.diagnostics, len(vectors)))

    # Re-derive per-source counts from what was ACTUALLY embedded/clustered (post-dedup, and
    # post any --sample cap used earlier in the run) rather than trusting the original loader
    # counts, so the artifact's provenance reflects reality.
    counts = Counter(r.source for r in used_rows)
    sources = [corpus_mod.provenance_for(name, count) for name, count in counts.items()]

    artifact = cluster_map_mod.build_cluster_map_dict(
        embedding_config, result, used_rows, sources, clustering_config.seed, clustering_config.n_init,
    )
    cluster_map_mod.validate_cluster_map(artifact)
    path = cluster_map_mod.write_cluster_map(artifact)
    typer.echo(f"Wrote validated artifact to {path}")


@app.command("build-artifact")
def build_artifact(
    k: int | None = typer.Option(
        None, help="Which k to promote to the final artifact. Defaults to clustering.yaml's default_k."
    ),
) -> None:
    """Cluster at the chosen k, assemble cluster-map.json, validate, and write it."""
    _run_build_artifact(k)


@app.command("export-viz")
def export_viz(
    k: int | None = typer.Option(None, help="Which k to visualize. Defaults to clustering.yaml's default_k."),
) -> None:
    """Project embeddings to 2D (t-SNE) with cluster labels; write visualization.json for the
    exploration page. Diagnostic only — not part of the routing artifact contract."""
    if not EMBEDDINGS_PATH.exists():
        raise typer.BadParameter(f"{EMBEDDINGS_PATH} not found — run `embed` first.")
    if not CORPUS_PATH.exists():
        raise typer.BadParameter(f"{CORPUS_PATH} not found — run `corpus` first.")

    ids, vectors = embed_mod.load_embeddings(EMBEDDINGS_PATH)
    rows_by_id = {r.id: r for r in corpus_mod.read_corpus_jsonl(CORPUS_PATH)}

    clustering_config = load_clustering_config()
    chosen_k = k or clustering_config.default_k

    result = cluster_mod.run_kmeans(vectors, chosen_k, clustering_config.seed, clustering_config.n_init)
    typer.echo("Projecting to 2D with t-SNE (this can take a moment)...")
    points = viz_mod.build_viz_points(ids, vectors, rows_by_id, result.labels, clustering_config.seed)
    payload = viz_mod.build_viz_payload(points, chosen_k)
    viz_mod.write_viz_payload(payload, VIZ_PATH)
    typer.echo(f"Wrote {len(points)} points across {chosen_k} clusters to {VIZ_PATH}")


@app.command("run-all")
def run_all(
    sample: int | None = typer.Option(None, help="Dry-run cap applied to corpus."),
    k: int | None = typer.Option(None, help="Which k to promote. Defaults to clustering.yaml's default_k."),
) -> None:
    """Convenience: corpus -> embed -> cluster -> build-artifact in one shot."""
    _run_corpus(sample)
    _run_embed(None)  # corpus already applied the sample cap; don't cap twice
    _run_cluster()
    _run_build_artifact(k)


@app.command("validate-graders")
def validate_graders(
    tasks_per_source: int = typer.Option(10, help="How many tasks per gradeable source to check."),
) -> None:
    """GATE: run the reference (gold solution) and null (empty solution) controls over a small
    sample per gradeable source. reference must score ~100% pass, null ~0% pass — this is what
    proves the grader itself is correct, independent of any model's actual coding ability. If this
    gate fails, nothing downstream (calibration, evaluation) means anything.

    SWE-smith's reference/null aren't graded as solution strings — grade_reference/grade_null
    establish the bug then reverse the same patch / apply no fix, matching calibrate.py's
    run_and_grade special-casing. Docker-based, so this source alone can take several minutes at
    the default --tasks-per-source; pass a smaller value for a quicker check."""
    calibration_config = load_calibration_config()
    reference_and_null_graders = {
        "bigcodebench": (
            lambda t, timeout: bigcodebench_grading.grade(t, t.reference_solution, timeout_seconds=timeout),
            lambda t, timeout: bigcodebench_grading.grade(t, "", timeout_seconds=timeout),
        ),
        "ds1000": (
            lambda t, timeout: ds1000_grading.grade(t, t.reference_solution, timeout_seconds=timeout),
            lambda t, timeout: ds1000_grading.grade(t, "", timeout_seconds=timeout),
        ),
        "swe-smith": (
            lambda t, timeout: swesmith_grading.grade_reference(t, timeout_seconds=timeout),
            lambda t, timeout: swesmith_grading.grade_null(t, timeout_seconds=timeout),
        ),
    }

    for source in calibration_config.gradeable_sources:
        graders = reference_and_null_graders.get(source)
        if graders is None:
            typer.echo(f"{source}: no self-contained grader wired into this gate (skipped)")
            continue
        grade_reference, grade_null = graders
        tasks = load_gradeable_tasks(source)[:tasks_per_source]
        timeout = calibration_config.task_timeout_seconds
        ref_outcomes = [grade_reference(t, timeout).outcome for t in tasks]
        null_outcomes = [grade_null(t, timeout).outcome for t in tasks]
        ref_pass = sum(1 for o in ref_outcomes if o == "pass")
        null_pass = sum(1 for o in null_outcomes if o == "pass")
        typer.echo(
            f"{source} (n={len(tasks)}): reference {ref_pass}/{len(tasks)} pass {dict(Counter(ref_outcomes))} | "
            f"null {null_pass}/{len(tasks)} pass {dict(Counter(null_outcomes))}"
        )


@app.command()
def calibrate() -> None:
    """Select tasks stratified by cluster, run every configured model (real + controls), and
    write the validated model-profiles.json artifact."""
    if not CLUSTER_MAP_PATH.exists():
        raise typer.BadParameter(f"{CLUSTER_MAP_PATH} not found — run `build-artifact` first.")

    cluster_map = load_cluster_map(CLUSTER_MAP_PATH)
    embedding_config = load_embedding_config()
    calibration_config = load_calibration_config()
    models = load_models_config()

    typer.echo(f"Selecting tasks across {cluster_map.centroids.shape[0]} clusters...")
    selected = calibrate_mod.select_tasks(calibration_config, embedding_config, cluster_map)
    n_calibration = sum(1 for s in selected if s.split == "calibration")
    n_holdout = sum(1 for s in selected if s.split == "holdout")
    typer.echo(f"Selected {len(selected)} tasks: {n_calibration} calibration, {n_holdout} holdout")

    results = []
    for model in models:
        typer.echo(f"Calibrating {model.model_id} ({model.runner})...")
        result = calibrate_mod.calibrate_model(model, selected, calibration_config)
        stats = result.global_stats
        typer.echo(
            f"  global: {stats.number_succeeded}/{stats.number_of_tasks} pass, "
            f"smoothed_error_rate={stats.smoothed_error_rate:.3f}, excluded={stats.excluded}"
        )
        results.append(result)

    calibration_run_id = f"cal-{datetime.now(UTC).strftime('%Y-%m-%d-%H%M%S')}"
    artifact = profiles_mod.build_profiles_dict(
        results, cluster_map, embedding_config, calibration_config, calibration_run_id,
    )
    profiles_mod.validate_profiles(artifact)
    path = profiles_mod.write_profiles(artifact)
    typer.echo(f"Wrote validated artifact to {path}")


@app.command()
def evaluate() -> None:
    """Re-select the same (deterministic) task split, run the holdout split against every model,
    and report the lambda-sweep resolution/cost table plus always-strongest/always-cheapest/oracle
    baselines."""
    if not PROFILES_PATH.exists():
        raise typer.BadParameter(f"{PROFILES_PATH} not found — run `calibrate` first.")
    if not CLUSTER_MAP_PATH.exists():
        raise typer.BadParameter(f"{CLUSTER_MAP_PATH} not found — run `build-artifact` first.")

    cluster_map = load_cluster_map(CLUSTER_MAP_PATH)
    embedding_config = load_embedding_config()
    calibration_config = load_calibration_config()
    models = load_models_config()
    profiles_artifact = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    profiles_by_model = {m["model_id"]: m for m in profiles_artifact["models"]}

    selected = calibrate_mod.select_tasks(calibration_config, embedding_config, cluster_map)
    typer.echo("Running holdout tasks against every model...")
    holdout_outcomes = evaluate_mod.run_holdout_outcomes(selected, models, calibration_config)

    report = evaluate_mod.evaluate(selected, models, profiles_by_model, calibration_config, holdout_outcomes)
    typer.echo(f"\nHoldout tasks: {report.holdout_task_count}")
    typer.echo(
        f"Always-strongest ({report.always_strongest.get('model_id')}): "
        f"{report.always_strongest.get('resolution_rate', 0):.1%}"
    )
    typer.echo(
        f"Always-cheapest ({report.always_cheapest.get('model_id')}): "
        f"{report.always_cheapest.get('resolution_rate', 0):.1%}"
    )
    typer.echo(f"Oracle: {report.oracle_resolution_rate:.1%}")
    typer.echo("\nlambda   resolution   mean_cost   selection")
    for lr in report.lambda_results:
        typer.echo(f"{lr.lambda_:<8} {lr.resolution_rate:<12.1%} {lr.mean_cost:<11.4f} {lr.selection_counts}")
