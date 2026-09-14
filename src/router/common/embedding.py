"""Text preprocessing + encoding, shared by the pipeline and the runtime so both always agree on
the exact same rule. In-process via sentence-transformers — no server, no HTTP call.
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
    """No case folding, no whitespace collapsing — indentation/casing are signal for a code
    embedding model. Keep in sync with embedding.yaml's `input` block if you change this."""
    normalized = normalize_line_endings(text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]


DEFAULT_BATCH_SIZE = 4


@functools.lru_cache(maxsize=2)
def _load_encoder(model_id: str):
    """Memoised per model id so repeated calls to embed_texts/embed_one pay the load cost once."""
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
