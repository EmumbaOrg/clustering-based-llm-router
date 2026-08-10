"""Assembles, validates, and writes the final cluster-map.json artifact. Mirrors calibration's
profiles.py: same three-function shape (build_*_dict / validate_* / write_*) for the sibling
artifact.

Validation is two layers, matching artifacts-schema/README.md: standard JSON Schema structural
checks, plus cross-field invariants the schema can't express on its own (centroid length vs
dimensions, cluster count vs k, contiguous ascending cluster ids). Both layers actually live in
../../common/artifacts.py — the runtime is a second consumer of this artifact and must run the
exact same checks, so they can't be private to this writer. `validate_cluster_map` below is a
thin wrapper kept here so call sites and tests don't need to reach into common/.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

from ...common import artifacts as artifacts_mod
from ...common.artifacts import ARTIFACTS_DIR, write_json_artifact
from ...common.config import EmbeddingConfig
from ..corpus import CorpusRow, SourceProvenance
from .cluster import ClusterResult

logger = logging.getLogger(__name__)

SCHEMA_PATH = artifacts_mod.CLUSTER_MAP_SCHEMA_PATH


def compute_corpus_digest(rows: list[CorpusRow]) -> str:
    """SHA-256 over sorted (id, text) pairs, so the digest is stable regardless of load order —
    a re-run on unchanged input reproduces the same digest byte-for-byte."""
    hasher = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r.id):
        hasher.update(row.id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(row.text.encode("utf-8"))
        hasher.update(b"\0")
    return f"sha256:{hasher.hexdigest()}"


def build_cluster_map_dict(
    embedding_config: EmbeddingConfig,
    cluster_result: ClusterResult,
    corpus_rows: list[CorpusRow],
    sources: list[SourceProvenance],
    seed: int,
    n_init: int,
) -> dict:
    k = cluster_result.k
    clusters = [
        {
            "id": cluster_id,
            "size": cluster_result.diagnostics.sizes[cluster_id],
            "centroid": cluster_result.centroids[cluster_id].tolist(),
        }
        for cluster_id in range(k)
    ]
    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": 1,
        "artifact_id": f"clustermap-{now[:10]}-k{k}-s{seed}",
        "created_at": now,
        "embedding": {
            "model_id": embedding_config.model_id,
            "dimensions": embedding_config.dimensions,
            "normalisation": embedding_config.normalisation,
            "distance": embedding_config.distance,
            "centroid_dtype": "float64",
            "input": {
                "normalisation": embedding_config.input.normalisation,
                "truncation": {
                    "unit": embedding_config.input.truncation.unit,
                    "max": embedding_config.input.truncation.max,
                    "strategy": embedding_config.input.truncation.strategy,
                },
            },
        },
        "kmeans": {
            "k": k,
            "seed": seed,
            "n_init": n_init,
            "corpus_size": len(corpus_rows),
            "corpus_digest": compute_corpus_digest(corpus_rows),
            "sources": [
                {
                    "name": s.name,
                    "hf_id": s.hf_id,
                    "split": s.split,
                    "field": s.field,
                    "rows": s.rows,
                    "license": s.license,
                }
                for s in sources
            ],
        },
        "clusters": clusters,
    }


def validate_cluster_map(artifact: dict) -> None:
    artifacts_mod.validate_cluster_map(artifact)


def write_cluster_map(artifact: dict, path: Path | None = None) -> Path:
    target = write_json_artifact(artifact, path or (ARTIFACTS_DIR / "cluster-map.json"))
    logger.info(f"wrote cluster-map artifact to {target} (k={artifact['kmeans']['k']})")
    return target
