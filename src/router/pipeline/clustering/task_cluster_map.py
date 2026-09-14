"""Persists the full per-row cluster assignment K-means already computes during `build-artifact`
as a standalone, human-inspectable artifact keyed by the same stable task id `calibration/tasks.py`
uses. Derived and informational, not a load-bearing pipeline input like `cluster-map.json`/
`model-profiles.json` — no formal JSON Schema. `calibrate.py::select_tasks` is the one real
consumer, intersecting this file's entries against `load_gradeable_tasks`'s actual output rather
than trusting it wholesale.
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
    """`ids`/`labels` come from the same `load_embeddings`/`run_kmeans` pair, paired by position.
    `rows_by_id` (ALL corpus rows) is looked up by id per entry rather than zipped positionally,
    since it can be a different length than `ids` if corpus.jsonl and embeddings.npz are out of sync."""
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
