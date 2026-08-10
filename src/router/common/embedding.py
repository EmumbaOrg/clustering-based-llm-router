"""Text preprocessing + encoding — shared by the pipeline (batch corpus embedding) and, once
implemented, the runtime (single-query embedding). Kept as one module rather than a
preprocess/embed split: preprocessing only exists to feed encoding, no caller needs one without
the other, and both must always agree on the exact same rule (see `prepare_embedding_input`'s
docstring) — one file makes that impossible to drift apart by construction.

No server, no HTTP call — the encoder runs in-process via sentence-transformers. That was a
deliberate choice for the pipeline's batch corpus embedding, and
applies equally to the runtime once it exists: embedding one live prompt is just `embed_texts`
called with a one-element list, so there is no separate runtime-specific embedding path to build
or keep in sync with this one.
"""
from __future__ import annotations

import functools
import logging
import time
from pathlib import Path

import numpy as np

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def prepare_embedding_input(text: str, max_chars: int) -> str:
    """Deliberately no case folding and no whitespace collapsing — indentation and casing are
    signal for a code-oriented embedding model. If you change this rule, update embedding.yaml's
    `input` block to match: it's copied verbatim into cluster-map.json so any future consumer
    applies the exact same rule to a new query, and the two must never drift apart silently.
    """
    normalized = normalize_line_endings(text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]


# Measured empirically on this CPU-only setup: batch_size=32 spikes attention-matrix memory hard
# enough on this corpus's longer documents (up to 8000 chars / ~2000 tokens post-truncation) to
# get the process SIGKILL'd. batch_size=4 was both the safest AND fastest of the sizes tried
# (4/8/12 all measured; 4 won on both memory and wall-clock — larger batches pay more in padding
# waste across a wide length distribution than they gain in vectorization on CPU).
DEFAULT_BATCH_SIZE = 4


@functools.lru_cache(maxsize=2)
def _load_encoder(model_id: str):
    """Memoised per model id so a process that calls embed_texts/embed_one repeatedly (the
    runtime, or a pipeline command run in a loop) pays the load cost once, not per call. maxsize=2
    rather than 1: nothing in this codebase needs more than one model loaded at a time, but a
    fixed small cap costs nothing and avoids a hard "you may only ever load one model" assumption
    baked into the cache itself.
    """
    # Deferred import: sentence-transformers (and the torch it pulls in) is heavy: only pay for it
    # when actually embedding, not on every CLI invocation (e.g. `cluster`, which never needs it).
    from sentence_transformers import SentenceTransformer

    # trust_remote_code=True is required for jina-embeddings-v2-* — it ships custom JinaBert/ALiBi
    # modeling code on the Hub rather than a stock architecture.
    return SentenceTransformer(model_id, trust_remote_code=True)


def embed_texts(
    texts: list[str], config: EmbeddingConfig, batch_size: int = DEFAULT_BATCH_SIZE, show_progress_bar: bool = True,
) -> np.ndarray:
    prepared = [prepare_embedding_input(t, config.input.truncation.max) for t in texts]

    started = time.monotonic()
    logger.info(f"embedding {len(prepared)} rows with {config.model_id} (batch_size={batch_size})")

    model = _load_encoder(config.model_id)
    vectors = model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,   # L2-normalize here, matching config/embedding.yaml
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )

    if vectors.shape[1] != config.dimensions:
        raise ValueError(
            f"Embedding model '{config.model_id}' produced {vectors.shape[1]}-dim vectors, but "
            f"the configured dimensions is {config.dimensions}. Update the config/artifact to "
            f"match, or check you're loading the intended model."
        )

    logger.info(f"embedding completed in {time.monotonic() - started:.1f}s ({vectors.shape[1]}-dim)")

    # sentence-transformers returns float32; the artifact schema requires float64 centroids, so
    # cast at the source rather than relying on a later step to remember to do it.
    return vectors.astype(np.float64)


def embed_one(text: str, config: EmbeddingConfig) -> np.ndarray:
    """Thin wrapper over `embed_texts` so there is exactly one encoding code path, not a
    runtime-specific copy of it."""
    return embed_texts([text], config, batch_size=1, show_progress_bar=False)[0]


def save_embeddings(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids, dtype=object), vectors=vectors)


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return list(data["ids"]), data["vectors"]
