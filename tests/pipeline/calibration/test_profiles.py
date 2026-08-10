import json

import numpy as np
import pytest

from router.common.assign import ClusterMap
from router.common.config import (
    CalibrationConfig,
    EmbeddingConfig,
    EmbeddingInputConfig,
    ModelConfig,
    SmoothingConfig,
)
from router.pipeline.calibration.calibrate import ClusterStats, ModelCalibrationResult
from router.pipeline.calibration.profiles import (
    SCHEMA_PATH,
    build_profiles_dict,
    validate_profiles,
)


def _cluster_map(k: int = 3) -> ClusterMap:
    return ClusterMap(
        artifact_id="clustermap-test", embedding_model_id="test-model", dimensions=4,
        centroids=np.zeros((k, 4), dtype=np.float64),
    )


def _embedding_config() -> EmbeddingConfig:
    from router.common.config import TruncationConfig

    return EmbeddingConfig(
        model_id="test-model", dimensions=4, normalisation="l2", distance="euclidean",
        input=EmbeddingInputConfig(normalisation="crlf-lf+trim", truncation=TruncationConfig(unit="chars", max=8000, strategy="head")),
    )


def _calibration_config() -> CalibrationConfig:
    return CalibrationConfig(
        gradeable_sources=["bigcodebench"], tasks_per_cluster=4, candidate_pool_oversample=4,
        task_timeout_seconds=60, smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        holdout_fraction=0.3, seed=42, lambda_sweep=[0, 0.1],
    )


def _stats(n=4, succeeded=3) -> ClusterStats:
    failed = n - succeeded
    rate = failed / n if n else 0.0
    return ClusterStats(number_of_tasks=n, number_succeeded=succeeded, number_failed=failed, raw_error_rate=rate, smoothed_error_rate=rate, excluded={})


def _result(model_id: str) -> ModelCalibrationResult:
    model = ModelConfig(model_id=model_id, provider="test", runner="pi", cost_input=0, cost_output=0, context_window=0, max_tokens=0)
    return ModelCalibrationResult(model=model, global_stats=_stats(), cluster_stats={0: _stats(), 1: _stats(n=2, succeeded=2)})


def test_build_profiles_dict_validates_against_the_real_schema():
    artifact = build_profiles_dict([_result("m1")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-test")
    validate_profiles(artifact)  # must not raise


def test_schema_file_exists_and_is_valid_json():
    assert SCHEMA_PATH.exists()
    json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_cluster_map_id_is_carried_through_for_the_linkage_check():
    cluster_map = _cluster_map()
    artifact = build_profiles_dict([_result("m1")], cluster_map, _embedding_config(), _calibration_config(), "cal-test")
    assert artifact["cluster_map_id"] == cluster_map.artifact_id


def test_rejects_a_cluster_id_outside_the_cluster_map_range():
    artifact = build_profiles_dict([_result("m1")], _cluster_map(k=3), _embedding_config(), _calibration_config(), "cal-test")
    artifact["models"][0]["clusters"]["99"] = artifact["models"][0]["clusters"]["0"]
    with pytest.raises(ValueError):
        validate_profiles(artifact)


def test_rejects_mismatched_success_failure_counts():
    artifact = build_profiles_dict([_result("m1")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-test")
    artifact["models"][0]["global"]["number_of_tasks"] = 999
    with pytest.raises(ValueError):
        validate_profiles(artifact)


def test_rejects_unknown_additional_fields():
    artifact = build_profiles_dict([_result("m1")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-test")
    artifact["unexpected"] = "nope"
    with pytest.raises(ValueError):
        validate_profiles(artifact)


def test_duplicate_model_ids_are_representable_but_the_pipeline_should_not_produce_them():
    # The schema itself doesn't forbid duplicate model_id (JSON Schema can't express "unique by
    # field" cleanly here) — this documents that build_profiles_dict is what's responsible for
    # not being called with two results for the same model, not the schema.
    artifact = build_profiles_dict([_result("m1"), _result("m1")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-test")
    validate_profiles(artifact)  # schema-valid despite the duplicate; a real caller must not do this
