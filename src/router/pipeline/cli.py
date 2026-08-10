"""The `router pipeline` command group: `uv run router pipeline <command>`."""
from __future__ import annotations

from collections import Counter

import typer

from ..common import embedding as embed_mod
from ..common.config import REPO_ROOT, load_clustering_config, load_embedding_config
from ..common.logging_config import configure_logging
from . import corpus as corpus_mod
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
