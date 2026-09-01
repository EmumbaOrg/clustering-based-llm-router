"""Persists the full per-row cluster assignment K-means already computes during `build-artifact`
— `cluster.py::run_kmeans`'s `ClusterResult.labels` — as a standalone, human-inspectable artifact
keyed by the SAME stable task id `calibration/tasks.py` uses (`corpus.py::stable_task_id`, see its
own docstring for why the two now share one id scheme).

This is a derived, informational artifact, not a load-bearing pipeline input the way
`cluster-map.json`/`model-profiles.json` are — no formal JSON Schema, matching the lighter
treatment already given to files like `calibration-details-<run>.csv`. `calibrate.py::select_tasks`
is the one real consumer: it intersects this file's entries against
`calibration/tasks.py::load_gradeable_tasks`'s actual output rather than trusting it wholesale, so
a row this file has a label for but that source's own loader excludes (e.g. a DS-1000 Matplotlib
row — corpus.py's loader doesn't apply that filter, calibration's does) is simply never selected,
not a class of error to handle specially.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ...common.artifacts import ARTIFACTS_DIR, write_json_artifact
from ..corpus import CorpusRow

logger = logging.getLogger(__name__)


def build_task_cluster_map_dict(
    ids: list[str],
    labels: np.ndarray,
    rows_by_id: dict[str, CorpusRow],
    cluster_map_id: str,
) -> dict:
    """`ids`/`labels` come from the same `load_embeddings`/`run_kmeans` pair and are guaranteed
    the same length and order — paired by position. `rows_by_id` (ALL corpus rows, not the
    already-filtered `used_rows` list `_run_build_artifact` builds for its own source-count
    purposes) is looked up by id per entry rather than zipped positionally against `used_rows`,
    since `used_rows` can be shorter than `ids` if corpus.jsonl and embeddings.npz are ever out of
    sync — this function must not assume they're the same length."""
    tasks = []
    skipped = 0
    for task_id, label in zip(ids, labels):
        row = rows_by_id.get(task_id)
        if row is None:
            skipped += 1
            continue
        tasks.append({"task_id": task_id, "source": row.source, "cluster_id": int(label)})
    if skipped:
        logger.warning(f"task-cluster-map: {skipped} embedded id(s) had no matching corpus row, skipped")

    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": 1,
        "artifact_id": f"taskclustermap-{now[:10]}-{cluster_map_id}",
        "created_at": now,
        "cluster_map_id": cluster_map_id,
        "tasks": tasks,
    }


def write_task_cluster_map(artifact: dict, path: Path | None = None) -> Path:
    target = write_json_artifact(artifact, path or (ARTIFACTS_DIR / "task-cluster-map.json"))
    logger.info(f"wrote task-cluster-map artifact to {target} ({len(artifact['tasks'])} tasks)")
    return target


def load_task_cluster_map(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
