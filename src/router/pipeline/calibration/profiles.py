"""Assembles, validates, and writes the final model-profiles.json artifact. Mirrors clustering's
cluster_map.py structure for cluster-map.json: two-layer validation (JSON Schema + cross-field
invariants the schema can't express — see ../../../README.md's "Config and artifact schema").
Both layers live in ../../common/artifacts.py, since the runtime is a second consumer of this
artifact. `validate_profiles` below is a thin wrapper kept here so call sites and tests don't need
to reach into common/.
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


# Compatibility check for merge_profiles_dict below. Deliberately excludes `total_tasks`: it's
# summed across models in `results`, so an incremental single-model run's value is never equal to
# a multi-model existing artifact's — that's expected, not a mismatch.
_TASK_SELECTION_COMPATIBILITY_FIELDS = ("gradeable_sources", "tasks_per_cluster", "seed")


def merge_profiles_dict(existing_artifact: dict, new_artifact: dict) -> dict:
    """Appends/replaces `new_artifact`'s model entries into `existing_artifact` — the incremental
    single-model calibration workflow the spec calls for ("a new model can be onboarded by running
    only the calibration suite") and `task_selection`'s own schema description anticipates
    ("How the calibration task pool was chosen, for reproducibility"). Both artifacts must share
    the same cluster map and task-selection parameters — a mismatch means cluster id N wouldn't
    mean the same thing in both, exactly the failure mode the schema's task_selection field exists
    to let a caller detect; caught here as a hard error rather than silently producing a profiles
    file whose entries aren't mutually comparable."""
    if existing_artifact["cluster_map_id"] != new_artifact["cluster_map_id"]:
        raise ValueError(
            f"cannot merge: existing model-profiles.json was built against cluster map "
            f"{existing_artifact['cluster_map_id']!r}, but this run used "
            f"{new_artifact['cluster_map_id']!r} — cluster ids would not mean the same thing."
        )
    if existing_artifact["embedding"] != new_artifact["embedding"]:
        raise ValueError(
            f"cannot merge: existing model-profiles.json's embedding config "
            f"({existing_artifact['embedding']!r}) does not match this run's "
            f"({new_artifact['embedding']!r})."
        )
    for field in _TASK_SELECTION_COMPATIBILITY_FIELDS:
        existing_value = existing_artifact["task_selection"][field]
        new_value = new_artifact["task_selection"][field]
        if existing_value != new_value:
            raise ValueError(
                f"cannot merge: existing model-profiles.json's task_selection.{field} "
                f"({existing_value!r}) does not match this run's ({new_value!r}) — re-calibrate "
                "every model together instead of incrementally, or use the same --tasks-file."
            )

    merged_by_id = {m["model_id"]: m for m in existing_artifact["models"]}
    for model in new_artifact["models"]:
        merged_by_id[model["model_id"]] = model  # replaces, if re-calibrating an existing model

    now = datetime.now(UTC).isoformat()
    merged = dict(existing_artifact)
    merged["artifact_id"] = f"profiles-{now[:10]}-merged"
    merged["created_at"] = now
    merged["models"] = list(merged_by_id.values())
    return merged


def validate_profiles(artifact: dict) -> None:
    artifacts_mod.validate_profiles(artifact)


def write_profiles(artifact: dict, path: Path | None = None) -> Path:
    target = write_json_artifact(artifact, path or (ARTIFACTS_DIR / "model-profiles.json"))
    logger.info(f"wrote model-profiles artifact to {target}")
    return target
