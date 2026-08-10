"""Exports a 2D projection of corpus embeddings + cluster assignments for visualization.

Not part of the routing artifact contract — this is a diagnostic/exploration export, not
validated against artifacts-schema/, and deliberately kept out of artifacts/.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

from ..corpus import CorpusRow

logger = logging.getLogger(__name__)

SNIPPET_MAX_CHARS = 220


@dataclasses.dataclass(frozen=True)
class VizPoint:
    id: str
    source: str
    cluster: int
    x: float
    y: float
    snippet: str


def project_2d(vectors: np.ndarray, seed: int) -> np.ndarray:
    # perplexity must be < n_samples; 30 is sklearn's default and fine for the few-hundred-to-low
    # -thousand point scale this pipeline runs at. Not tuned further — this is for exploration,
    # not a claim about the "true" structure of the embedding space.
    perplexity = min(30, max(5, vectors.shape[0] // 4))
    started = time.monotonic()
    logger.info(f"t-SNE projection started ({vectors.shape[0]} points, perplexity={perplexity})")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    coords = tsne.fit_transform(vectors)
    logger.info(f"t-SNE projection completed in {time.monotonic() - started:.1f}s")
    return coords.astype(np.float64)


def build_viz_points(
    ids: list[str], vectors: np.ndarray, rows_by_id: dict[str, CorpusRow], labels: np.ndarray, seed: int,
) -> list[VizPoint]:
    coords = project_2d(vectors, seed)
    points: list[VizPoint] = []
    for i, point_id in enumerate(ids):
        row = rows_by_id[point_id]
        snippet = row.text.strip().replace("\n", " ")
        if len(snippet) > SNIPPET_MAX_CHARS:
            snippet = snippet[:SNIPPET_MAX_CHARS].rstrip() + "…"
        points.append(
            VizPoint(
                id=point_id,
                source=row.source,
                cluster=int(labels[i]),
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                snippet=snippet,
            )
        )
    return points


def build_viz_payload(points: list[VizPoint], k: int) -> dict:
    cluster_sizes: dict[int, int] = {}
    cluster_sources: dict[int, dict[str, int]] = {}
    for p in points:
        cluster_sizes[p.cluster] = cluster_sizes.get(p.cluster, 0) + 1
        cluster_sources.setdefault(p.cluster, {})
        cluster_sources[p.cluster][p.source] = cluster_sources[p.cluster].get(p.source, 0) + 1

    clusters = [
        {"id": cid, "size": cluster_sizes[cid], "sources": cluster_sources[cid]}
        for cid in range(k)
    ]
    return {
        "k": k,
        "corpusSize": len(points),
        "points": [dataclasses.asdict(p) for p in points],
        "clusters": clusters,
    }


def write_viz_payload(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
