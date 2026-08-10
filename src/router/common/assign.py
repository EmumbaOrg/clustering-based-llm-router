"""Nearest-centroid cluster assignment for a new embedding vector.

Shared by ../pipeline (calibration/evaluation reads cluster-map.json to assign held-out tasks) and
../runtime (the live router assigns one incoming prompt). Both read centroids from the same
cluster-map.json and must use the exact same tie-break/argmin rule, which is why this lives in
common/ rather than being reimplemented on either side.

Distance is SQUARED Euclidean (no sqrt) — monotone with true distance, one fewer floating-point
operation, and avoids sqrt's platform/library differences mattering for a comparison that doesn't
need them.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class ClusterMap:
    """The subset of cluster-map.json this module needs: centroids plus enough metadata to
    validate an incoming vector against them."""

    artifact_id: str
    embedding_model_id: str
    dimensions: int
    centroids: np.ndarray  # shape (k, dimensions), float64


@dataclasses.dataclass(frozen=True)
class ClusterAssignment:
    cluster_id: int
    distance: float  # squared Euclidean to the nearest centroid
    runner_up_cluster_id: int | None
    runner_up_distance: float | None


def cluster_map_from_dict(data: dict) -> ClusterMap:
    """Builds a ClusterMap from an already-parsed cluster-map.json. Separated from
    `load_cluster_map` so a caller that has already read the file once (to schema-validate it, or
    to pull the `embedding.input` preprocessing rule off it — see runtime/context.py) doesn't read
    it a second time just to get the centroids."""
    clusters = data["clusters"]
    dimensions = data["embedding"]["dimensions"]
    k = len(clusters)
    centroids = np.zeros((k, dimensions), dtype=np.float64)
    seen_ids = set()
    for cluster in clusters:
        cid = cluster["id"]
        if cid in seen_ids or not (0 <= cid < k):
            raise ValueError(f"cluster-map.json has a malformed cluster id: {cid}")
        seen_ids.add(cid)
        centroids[cid] = cluster["centroid"]
    return ClusterMap(
        artifact_id=data["artifact_id"],
        embedding_model_id=data["embedding"]["model_id"],
        dimensions=dimensions,
        centroids=centroids,
    )


def load_cluster_map(path: Path) -> ClusterMap:
    data = json.loads(path.read_text(encoding="utf-8"))
    return cluster_map_from_dict(data)


def assign_cluster(vector: np.ndarray, cluster_map: ClusterMap) -> ClusterAssignment:
    """Single-pass nearest + runner-up centroid, ascending id scan, strict `<` so an exact tie
    resolves to the lower id — deterministic by construction, no separate sort needed."""
    if vector.shape[0] != cluster_map.dimensions:
        raise ValueError(
            f"vector has {vector.shape[0]} dimensions, cluster map expects {cluster_map.dimensions}"
        )

    best_id: int | None = None
    best_dist = float("inf")
    runner_up_id: int | None = None
    runner_up_dist = float("inf")

    for cluster_id in range(cluster_map.centroids.shape[0]):
        diff = vector - cluster_map.centroids[cluster_id]
        dist = float(np.dot(diff, diff))
        if dist < best_dist:
            runner_up_id, runner_up_dist = best_id, best_dist
            best_id, best_dist = cluster_id, dist
        elif dist < runner_up_dist:
            runner_up_id, runner_up_dist = cluster_id, dist

    if best_id is None:
        raise ValueError("cluster map has no clusters — cannot assign")
    return ClusterAssignment(
        cluster_id=best_id,
        distance=best_dist,
        runner_up_cluster_id=runner_up_id,
        runner_up_distance=runner_up_dist if runner_up_id is not None else None,
    )
