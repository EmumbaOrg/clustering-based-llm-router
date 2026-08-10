"""The `router pipeline` command group: `uv run router pipeline <command>`."""
from __future__ import annotations

import typer

from ..common import embedding as embed_mod
from ..common.config import REPO_ROOT, load_embedding_config
from ..common.logging_config import configure_logging
from . import corpus as corpus_mod

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
