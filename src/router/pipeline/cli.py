from __future__ import annotations

import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import typer

from ..common import embedding as embed_mod
from ..common.assign import ClusterMap, load_cluster_map
from ..common.config import (
    REPO_ROOT,
    CalibrationConfig,
    EmbeddingConfig,
    ModelConfig,
    load_calibration_config,
    load_clustering_config,
    load_embedding_config,
    load_models_config,
)
from ..common.logging_config import configure_logging
from . import corpus as corpus_mod
from . import evaluate as evaluate_mod
from . import visualize as visualize_mod
from .calibration import calibrate as calibrate_mod
from .calibration import profiles as profiles_mod
from .calibration.calibrate import SelectedTask
from .calibration.tasks import load_gradeable_tasks
from .clustering import cluster as cluster_mod
from .clustering import cluster_map as cluster_map_mod
from .clustering import task_cluster_map as task_cluster_map_mod

app = typer.Typer(help="The offline pipeline: corpus -> embed -> cluster -> calibrate -> evaluate.")


@app.callback()
def main_callback(
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging verbosity (DEBUG, INFO, WARNING, ERROR)."
    ),
    log_file: str | None = typer.Option(
        None, "--log-file", help="Also append logs to this file, in addition to stderr."
    ),
) -> None:
    configure_logging(level=log_level, log_file=Path(log_file) if log_file else None)

WORK_DIR = REPO_ROOT / ".cache"
CORPUS_PATH = WORK_DIR / "corpus.jsonl"
EMBEDDINGS_PATH = WORK_DIR / "embeddings.npz"
CLUSTER_MAP_PATH = cluster_map_mod.ARTIFACTS_DIR / "cluster-map.json"
TASK_CLUSTER_MAP_PATH = cluster_map_mod.ARTIFACTS_DIR / "task-cluster-map.json"
PROFILES_PATH = profiles_mod.ARTIFACTS_DIR / "model-profiles.json"
CLUSTER_VISUALIZATION_PATH = cluster_map_mod.ARTIFACTS_DIR / "cluster-visualization.png"


def _details_csv_path(run_timestamp: str) -> Path:
    # Timestamp lives in the FILENAME, not a per-row column — this is what actually stops one
    # calibrate run (or a test run using the wrong path) from silently overwriting another's CSV,
    # which a per-row timestamp inside a single shared file never did (see CalibrationDetailRow's
    # docstring for the incident that motivated this).
    return profiles_mod.ARTIFACTS_DIR / f"calibration-details-{run_timestamp}.csv"


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
    """Build corpus.jsonl."""
    _run_corpus(sample)


def _run_embed(sample: int | None, per_source_sample: int | None = None, seed: int = 42) -> None:
    if not CORPUS_PATH.exists():
        raise typer.BadParameter(f"{CORPUS_PATH} not found — run `corpus` first.")
    rows = corpus_mod.read_corpus_jsonl(CORPUS_PATH)
    if per_source_sample is not None:
        # Stratified by source rather than `rows[:N]` — a plain prefix cap would just take
        # whichever source(s) happen to sort first in corpus.jsonl, not a balanced cross-section.
        rng = random.Random(seed)
        by_source: dict[str, list] = {}
        for r in rows:
            by_source.setdefault(r.source, []).append(r)
        sampled = []
        for source in sorted(by_source):
            source_rows = sorted(by_source[source], key=lambda r: r.id)  # deterministic before shuffling
            chosen = rng.sample(source_rows, min(per_source_sample, len(source_rows)))
            sampled.extend(chosen)
            typer.echo(f"  {source}: sampled {len(chosen)}/{len(source_rows)}")
        rows = sampled
    elif sample is not None:
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
    per_source_sample: int | None = typer.Option(
        None, "--per-source-sample",
        help="Randomly sample up to N rows PER SOURCE instead of the first N overall (e.g. for a "
        "source-balanced visualization sample). Overrides --sample.",
    ),
    seed: int = typer.Option(42, help="Random seed for --per-source-sample."),
) -> None:
    """Build embeddings.npz."""
    _run_embed(sample, per_source_sample, seed)


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
    """Print K-means diagnostics for every candidate k in config/clustering.yaml."""
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

    # Full per-task cluster labels, keyed by the same stable id calibration/tasks.py uses — lets
    # select_tasks() sample by cluster without a second, separate embedding pass. Built from
    # `rows_by_id` (every corpus row), not `used_rows` (already filtered to ids actually found),
    # so a task-cluster-map.json/embeddings.npz mismatch is handled by build_task_cluster_map_dict
    # itself rather than silently misaligning ids and labels by position.
    task_map_artifact = task_cluster_map_mod.build_task_cluster_map_dict(
        ids, result.labels, rows_by_id, artifact["artifact_id"],
    )
    task_map_path = task_cluster_map_mod.write_task_cluster_map(task_map_artifact)
    typer.echo(f"Wrote {task_map_path}")


@app.command("build-artifact")
def build_artifact(
    k: int | None = typer.Option(
        None, help="Which k to promote to the final artifact. Defaults to clustering.yaml's default_k."
    ),
) -> None:
    """Build and validate cluster-map.json."""
    _run_build_artifact(k)


@app.command("run-all")
def run_all(
    sample: int | None = typer.Option(None, help="Dry-run cap applied to corpus."),
    k: int | None = typer.Option(None, help="Which k to promote. Defaults to clustering.yaml's default_k."),
) -> None:
    _run_corpus(sample)
    _run_embed(None)  # corpus already applied the sample cap; don't cap twice
    _run_cluster()
    _run_build_artifact(k)


def _run_visualize_clusters(sample_size: int, highlight: list[str], seed: int, output: str | None) -> None:
    if not EMBEDDINGS_PATH.exists():
        raise typer.BadParameter(f"{EMBEDDINGS_PATH} not found — run `embed` first.")
    if not CLUSTER_MAP_PATH.exists():
        raise typer.BadParameter(f"{CLUSTER_MAP_PATH} not found — run `build-artifact` first.")

    ids, vectors = embed_mod.load_embeddings(EMBEDDINGS_PATH)
    cluster_map = load_cluster_map(CLUSTER_MAP_PATH)
    out_path = Path(output) if output else CLUSTER_VISUALIZATION_PATH

    result_path, missing = visualize_mod.plot_clusters(
        ids, vectors, cluster_map, out_path, sample_size=sample_size, highlight_ids=highlight, seed=seed,
    )
    if missing:
        typer.echo(f"Warning: {len(missing)} --highlight task_id(s) not found in embeddings.npz: {missing}")
    typer.echo(f"Wrote cluster visualization ({sample_size} sampled tasks) to {result_path}")


# Same reasoning as _SOURCE_OPTION above — a module-level singleton so ruff's B008 doesn't flag the
# list-typed Option default, while still supporting a repeatable `--highlight` flag.
_HIGHLIGHT_OPTION = typer.Option(
    None, "--highlight", help="task_id to annotate on the plot (repeatable), e.g. "
    "--highlight BigCodeBench/0 --highlight ds1000:42. Included even if not in the random sample."
)


@app.command("visualize-clusters")
def visualize_clusters(
    sample_size: int = typer.Option(500, help="How many tasks to randomly sample and plot."),
    highlight: list[str] | None = _HIGHLIGHT_OPTION,
    seed: int = typer.Option(42, help="Sampling/PCA random seed."),
    output: str | None = typer.Option(
        None, help="Output PNG path. Defaults to artifacts/cluster-visualization.png."
    ),
) -> None:
    """Plot a 2D PCA projection of sampled tasks colored by cluster, against the cluster centroids
    from cluster-map.json — a sanity-check view of where tasks actually land."""
    _run_visualize_clusters(sample_size, highlight or [], seed, output)


def _load_selected_tasks(
    tasks_file: Path | None = None,
) -> tuple[ClusterMap, EmbeddingConfig, CalibrationConfig, list[ModelConfig], list[SelectedTask]]:
    cluster_map = load_cluster_map(CLUSTER_MAP_PATH)
    embedding_config = load_embedding_config()
    calibration_config = load_calibration_config()
    models = load_models_config()

    if tasks_file is not None:
        # A pinned selection needs no task-cluster-map at all — cluster_id travels with each
        # entry already. This is what makes incremental single-model calibration (see the
        # `calibrate` command's `--model` option) safe: the new model grades against the EXACT
        # set an existing model-profiles.json was built from, not a freshly re-derived one.
        typer.echo(f"Loading pinned task selection from {tasks_file}...")
        selected = calibrate_mod.load_task_selection(tasks_file)
    else:
        if not TASK_CLUSTER_MAP_PATH.exists():
            raise typer.BadParameter(f"{TASK_CLUSTER_MAP_PATH} not found — run `build-artifact` first.")
        task_cluster_map = task_cluster_map_mod.load_task_cluster_map(TASK_CLUSTER_MAP_PATH)
        typer.echo(f"Selecting tasks across {cluster_map.centroids.shape[0]} clusters...")
        selected = calibrate_mod.select_tasks(calibration_config, cluster_map, task_cluster_map)
    return cluster_map, embedding_config, calibration_config, models, selected


# A module-level singleton, not an inline default — ruff's B008 flags any `list[...]`-typed
# Option/Argument default as a suspected mutable default, even though typer's own repeated-flag
# ("--source", repeatable) support requires exactly this pattern.
_SOURCE_OPTION = typer.Option(
    None, "--source", help="Gate only this source (repeatable). Defaults to every source in "
    "calibration.yaml's gradeable_sources."
)


@app.command("validate-graders")
def validate_graders(
    tasks_per_source: int = typer.Option(10, help="How many tasks per gradeable source to check."),
    source: list[str] | None = _SOURCE_OPTION,
) -> None:
    """GATE: run the reference (gold solution) and null (empty solution) controls over a small
    sample per gradeable source. reference must score ~100% pass, null ~0% pass — this is what
    proves the grader itself is correct, independent of any model's actual coding ability. If this
    gate fails, nothing downstream (calibration, evaluation) means anything.

    Uses `calibrate.py`'s `grade_reference`/`grade_null` — the SAME dispatch calibration itself
    uses — so an unwired source raises immediately instead of being silently skipped, and this
    gate can never drift from what a real calibration run actually does."""
    calibration_config = load_calibration_config()
    sources = source or calibration_config.gradeable_sources
    # random.sample, not [:n] — a source grouped by repo/instance (e.g. swe-smith's 128 repos)
    # would otherwise always sample the same handful of repos and never gate most of the dataset.
    rng = random.Random(calibration_config.seed)

    for src in sources:
        all_tasks = load_gradeable_tasks(src)
        tasks = rng.sample(all_tasks, min(tasks_per_source, len(all_tasks)))
        timeout = calibration_config.grading_timeout_for(src)
        ref_outcomes = [calibrate_mod.grade_reference(t, timeout).outcome for t in tasks]
        null_outcomes = [calibrate_mod.grade_null(t, timeout).outcome for t in tasks]
        ref_pass = sum(1 for o in ref_outcomes if o == "pass")
        null_pass = sum(1 for o in null_outcomes if o == "pass")
        typer.echo(
            f"{src} (n={len(tasks)}): reference {ref_pass}/{len(tasks)} pass {dict(Counter(ref_outcomes))} | "
            f"null {null_pass}/{len(tasks)} pass {dict(Counter(null_outcomes))}"
        )


_MODEL_OPTION = typer.Option(
    None, "--model", help="Calibrate only this model (repeatable). Requires --tasks-file — an "
    "incremental run grades against the exact task set an existing model-profiles.json was built "
    "from, and merges the result into it rather than overwriting the other models' entries."
)


@app.command()
def calibrate(
    tasks_file: Path | None = typer.Option(
        None, "--tasks-file",
        help="Load a pinned task selection (from a previous run's auto-written "
        "calibration-task-selection-<run>.json) instead of selecting fresh. Required for "
        "--model to be safe — an incremental run must grade against the exact set an existing "
        "model-profiles.json was built from.",
    ),
    model: list[str] | None = _MODEL_OPTION,
) -> None:
    """Build and validate model-profiles.json."""
    if not CLUSTER_MAP_PATH.exists():
        raise typer.BadParameter(f"{CLUSTER_MAP_PATH} not found — run `build-artifact` first.")
    if model and tasks_file is None:
        raise typer.BadParameter(
            "--model requires --tasks-file — an incremental run must grade against the exact "
            "task set the existing model-profiles.json was built from, not a fresh selection."
        )

    cluster_map, embedding_config, calibration_config, models, selected = _load_selected_tasks(tasks_file)

    if model:
        by_id = {m.model_id: m for m in models}
        unknown = [m for m in model if m not in by_id]
        if unknown:
            raise typer.BadParameter(f"unknown model_id(s) in config/models.yaml: {unknown}")
        models = [by_id[m] for m in model]

    n_calibration = sum(1 for s in selected if s.split == "calibration")
    n_holdout = sum(1 for s in selected if s.split == "holdout")
    typer.echo(f"Selected {len(selected)} tasks: {n_calibration} calibration, {n_holdout} holdout")

    # One timestamp for the whole run, shared by the details CSV filename, calibration_run_id
    # below, and the auto-written task-selection pin — so everything this run produced is
    # trivially matchable by name.
    run_timestamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S")
    details_path = _details_csv_path(run_timestamp)

    # Always written, whether this run selected fresh or loaded a pin — cheap, and it's what a
    # LATER incremental (--model) run, or anyone auditing this one, would pin against.
    selection_artifact = calibrate_mod.selected_tasks_to_dict(selected, k=cluster_map.centroids.shape[0])
    selection_path = ARTIFACTS_DIR / f"calibration-task-selection-{run_timestamp}.json"
    calibrate_mod.write_task_selection(selection_artifact, selection_path)
    typer.echo(f"Wrote task selection to {selection_path}")
    if selection_artifact["empty_clusters"]:
        typer.echo(f"WARNING: {len(selection_artifact['empty_clusters'])} cluster(s) have zero gradeable tasks: {selection_artifact['empty_clusters']}")

    # Tasks-outer / models-inner — see calibrate_models' docstring for why the loop order is worth
    # roughly a factor of len(models) on image-pull cost. Per-model progress is on the `router`
    # logger (--log-level INFO) rather than echoed here, since the run no longer proceeds
    # model-by-model.
    # calibrate_models writes the details CSV incrementally (one row per completed call, flushed
    # immediately) rather than only at the end — so a still-running or interrupted run's progress
    # is always inspectable on disk, not just held in memory until the whole run finishes.
    typer.echo(f"Grading {len(models)} models against each task (tasks-outer)...")
    typer.echo(f"Writing per-task-per-model details incrementally to {details_path}")
    results, detail_rows = calibrate_mod.calibrate_models(
        models, selected, calibration_config, details_csv_path=details_path,
    )
    for result in results:
        stats = result.global_stats
        model_cost = sum(row.cost_usd for row in detail_rows if row.model_id == result.model.model_id)
        typer.echo(
            f"  {result.model.model_id} ({result.model.runner}): "
            f"{stats.number_succeeded}/{stats.number_of_tasks} pass, "
            f"smoothed_error_rate={stats.smoothed_error_rate:.3f}, excluded={stats.excluded}, "
            f"cost=${model_cost:.4f}"
        )
    total_cost = sum(row.cost_usd for row in detail_rows)
    if total_cost:
        typer.echo(f"Total measured API cost this run: ${total_cost:.4f}")

    calibration_run_id = f"cal-{run_timestamp}"
    artifact = profiles_mod.build_profiles_dict(
        results, cluster_map, embedding_config, calibration_config, calibration_run_id,
    )
    if model:
        if not PROFILES_PATH.exists():
            raise typer.BadParameter(
                f"--model was given but {PROFILES_PATH} doesn't exist yet — run a full calibration "
                "first, then onboard additional models incrementally against it."
            )
        existing_artifact = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
        artifact = profiles_mod.merge_profiles_dict(existing_artifact, artifact)
        typer.echo(f"Merged {model} into the existing {len(existing_artifact['models'])}-model profile")
    profiles_mod.validate_profiles(artifact)
    path = profiles_mod.write_profiles(artifact)
    typer.echo(f"Wrote validated artifact to {path}")


@app.command()
def evaluate(
    tasks_file: Path | None = typer.Option(
        None, "--tasks-file",
        help="Reuse a pinned task selection (ideally the exact one the calibrate run this "
        "evaluates auto-wrote) instead of re-selecting. Without it, this re-selects fresh and "
        "relies on select_tasks() being deterministic given an unchanged config/cluster map/"
        "task-cluster-map — a pin removes that dependency entirely.",
    ),
) -> None:
    """Re-run the (same, ideally pinned) task split's holdout tasks and report the resolution/cost table."""
    if not PROFILES_PATH.exists():
        raise typer.BadParameter(f"{PROFILES_PATH} not found — run `calibrate` first.")
    if not CLUSTER_MAP_PATH.exists():
        raise typer.BadParameter(f"{CLUSTER_MAP_PATH} not found — run `build-artifact` first.")

    _, _, calibration_config, models, selected = _load_selected_tasks(tasks_file)
    profiles_artifact = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    profiles_by_model = {m["model_id"]: m for m in profiles_artifact["models"]}

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
