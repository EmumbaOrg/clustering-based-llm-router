"""Assembles, validates, and writes the final model-profiles.json artifact. Mirrors clustering's
cluster_map.py structure for cluster-map.json: two-layer validation (JSON Schema + cross-field
invariants the schema can't express — see artifacts-schema/README.md). Both layers actually live
in ../../common/artifacts.py — the runtime is a second consumer of this artifact and must run the
exact same checks, so they can't be private to this writer. `validate_profiles` below is a thin
wrapper kept here so call sites and tests don't need to reach into common/.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from ...common import artifacts as artifacts_mod
from ...common.artifacts import ARTIFACTS_DIR, write_json_artifact
from ...common.assign import ClusterMap
from ...common.config import CalibrationConfig, EmbeddingConfig
from .calibrate import ClusterStats, ModelCalibrationResult

logger = logging.getLogger(__name__)

SCHEMA_PATH = artifacts_mod.MODEL_PROFILES_SCHEMA_PATH


def _stats_dict(stats: ClusterStats) -> dict:
    d = {
        "number_of_tasks": stats.number_of_tasks,
        "number_succeeded": stats.number_succeeded,
        "number_failed": stats.number_failed,
        "raw_error_rate": stats.raw_error_rate,
        "smoothed_error_rate": stats.smoothed_error_rate,
    }
    if stats.excluded:
        d["excluded"] = stats.excluded
    return d


def build_profiles_dict(
    results: list[ModelCalibrationResult],
    cluster_map: ClusterMap,
    embedding_config: EmbeddingConfig,
    calibration_config: CalibrationConfig,
    calibration_run_id: str,
) -> dict:
    now = datetime.now(UTC).isoformat()
    models = [
        {
            "model_id": result.model.model_id,
            "provider": result.model.provider,
            "runner": result.model.runner,
            "is_control": result.model.is_control,
            "calibration_run_id": calibration_run_id,
            "global": _stats_dict(result.global_stats),
            "clusters": {str(cid): _stats_dict(stats) for cid, stats in result.cluster_stats.items()},
        }
        for result in results
    ]
    return {
        "schema_version": 1,
        "artifact_id": f"profiles-{now[:10]}-{calibration_run_id}",
        "created_at": now,
        "cluster_map_id": cluster_map.artifact_id,
        "cluster_count": cluster_map.centroids.shape[0],
        "embedding": {
            "model_id": embedding_config.model_id,
            "dimensions": embedding_config.dimensions,
        },
        "smoothing": {
            "method": calibration_config.smoothing.method,
            "prior_weight": calibration_config.smoothing.prior_weight,
            "formula": "(number_failed + prior_weight * global_raw_error_rate) / (number_of_tasks + prior_weight)",
        },
        "task_selection": {
            "gradeable_sources": calibration_config.gradeable_sources,
            "tasks_per_cluster": calibration_config.tasks_per_cluster,
            "seed": calibration_config.seed,
            "total_tasks": sum(r.global_stats.number_of_tasks for r in results) if results else 0,
        },
        "models": models,
    }


def validate_profiles(artifact: dict) -> None:
    artifacts_mod.validate_profiles(artifact)


def write_profiles(artifact: dict, path: Path | None = None) -> Path:
    target = write_json_artifact(artifact, path or (ARTIFACTS_DIR / "model-profiles.json"))
    logger.info(f"wrote model-profiles artifact to {target}")
    return target
