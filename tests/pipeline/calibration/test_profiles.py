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
    merge_profiles_dict,
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
        gradeable_sources=["bigcodebench"], tasks_per_cluster=4,
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


# --- merge_profiles_dict: the incremental single-model calibration workflow the spec calls for
# ("a new model can be onboarded by running only the calibration suite"). --------------------

def test_merge_profiles_dict_appends_a_genuinely_new_model():
    existing = build_profiles_dict([_result("m1")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-1")
    incoming = build_profiles_dict([_result("m2")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-2")

    merged = merge_profiles_dict(existing, incoming)

    assert {m["model_id"] for m in merged["models"]} == {"m1", "m2"}
    validate_profiles(merged)  # must still validate against the real schema


def test_merge_profiles_dict_replaces_an_existing_model_of_the_same_id():
    existing = build_profiles_dict([_result("m1")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-1")
    updated_m1 = ModelCalibrationResult(
        model=ModelConfig(model_id="m1", provider="test", runner="pi", cost_input=0, cost_output=0, context_window=0, max_tokens=0),
        global_stats=_stats(n=4, succeeded=4), cluster_stats={0: _stats(n=4, succeeded=4)},
    )
    incoming = build_profiles_dict([updated_m1], _cluster_map(), _embedding_config(), _calibration_config(), "cal-2")

    merged = merge_profiles_dict(existing, incoming)

    assert len(merged["models"]) == 1
    assert merged["models"][0]["global"]["number_succeeded"] == 4  # the NEW result, not the old one


def test_merge_profiles_dict_does_not_mutate_the_pre_existing_models_entries():
    existing = build_profiles_dict([_result("m1")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-1")
    original_m1 = dict(existing["models"][0])
    incoming = build_profiles_dict([_result("m2")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-2")

    merged = merge_profiles_dict(existing, incoming)

    kept_m1 = next(m for m in merged["models"] if m["model_id"] == "m1")
    assert kept_m1 == original_m1


def test_merge_profiles_dict_rejects_a_mismatched_cluster_map():
    existing = build_profiles_dict([_result("m1")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-1")
    different_cluster_map = ClusterMap(artifact_id="clustermap-DIFFERENT", embedding_model_id="test-model", dimensions=4, centroids=np.zeros((3, 4)))
    incoming = build_profiles_dict([_result("m2")], different_cluster_map, _embedding_config(), _calibration_config(), "cal-2")

    with pytest.raises(ValueError, match="cluster map"):
        merge_profiles_dict(existing, incoming)


def test_merge_profiles_dict_rejects_a_mismatched_tasks_per_cluster():
    existing = build_profiles_dict([_result("m1")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-1")
    different_config = CalibrationConfig(
        gradeable_sources=["bigcodebench"], tasks_per_cluster=999,  # different from _calibration_config()'s 4
        task_timeout_seconds=60, smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        holdout_fraction=0.3, seed=42, lambda_sweep=[0, 0.1],
    )
    incoming = build_profiles_dict([_result("m2")], _cluster_map(), _embedding_config(), different_config, "cal-2")

    with pytest.raises(ValueError, match="tasks_per_cluster"):
        merge_profiles_dict(existing, incoming)


def test_merge_profiles_dict_ignores_the_expected_total_tasks_mismatch_between_single_and_multi_model_runs():
    # build_profiles_dict sums number_of_tasks ACROSS every model in `results` — a multi-model
    # existing artifact and a single-model incremental one will always disagree on total_tasks,
    # and that's expected, not a sign the two runs used different task sets (see the field's own
    # comment in profiles.py). Must NOT raise.
    existing = build_profiles_dict([_result("m1"), _result("m2")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-1")
    incoming = build_profiles_dict([_result("m3")], _cluster_map(), _embedding_config(), _calibration_config(), "cal-2")

    merged = merge_profiles_dict(existing, incoming)  # must not raise

    assert {m["model_id"] for m in merged["models"]} == {"m1", "m2", "m3"}
