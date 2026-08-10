import copy
import dataclasses
import math

import pytest

from router.common.config import ModelConfig
from router.runtime.context import load_routing_context


def test_loads_a_well_formed_pair_and_excludes_the_control(load_context):
    ctx = load_context(lambda_=0.5)
    assert {c.model_id for c in ctx.candidates} == {"cheap-model", "mid-model", "strong-model"}
    assert ctx.cluster_map_id == "clustermap-fixture-k3"
    assert ctx.profiles_id == "profiles-fixture-a"
    assert ctx.lambda_ == 0.5
    assert ctx.cluster_map.centroids.shape == (3, 4)


def test_digest_is_stable_across_reloads(load_context):
    assert load_context(lambda_=0.5).digest == load_context(lambda_=0.5).digest


def test_digest_changes_with_lambda(load_context):
    assert load_context(lambda_=0.0).digest != load_context(lambda_=0.5).digest


@pytest.mark.parametrize("bad_lambda", [-0.1, float("inf"), float("nan")])
def test_rejects_a_bad_lambda(load_context, bad_lambda):
    with pytest.raises(ValueError):
        load_context(lambda_=bad_lambda)


def test_rejects_a_missing_cluster_map_file(tmp_path, model_profiles_dict, write_artifacts, candidates, embedding_config):
    _, mp_path = write_artifacts({}, model_profiles_dict)
    with pytest.raises(ValueError):
        load_routing_context(
            tmp_path / "does-not-exist.json", mp_path, 0.0, candidates=candidates, embedding_config=embedding_config,
        )


def test_rejects_a_cluster_map_id_mismatch(write_artifacts, cluster_map_dict, model_profiles_dict, candidates, embedding_config):
    profiles = copy.deepcopy(model_profiles_dict)
    profiles["cluster_map_id"] = "some-other-cluster-map"
    cm_path, mp_path = write_artifacts(copy.deepcopy(cluster_map_dict), profiles)
    with pytest.raises(ValueError, match="cluster_map_id"):
        load_routing_context(cm_path, mp_path, 0.0, candidates=candidates, embedding_config=embedding_config)


def test_rejects_a_configured_embedding_model_mismatch(write_artifacts, cluster_map_dict, model_profiles_dict, candidates, embedding_config):
    mismatched = dataclasses.replace(embedding_config, model_id="a-completely-different-model")
    cm_path, mp_path = write_artifacts(copy.deepcopy(cluster_map_dict), copy.deepcopy(model_profiles_dict))
    with pytest.raises(ValueError, match="embedding model"):
        load_routing_context(cm_path, mp_path, 0.0, candidates=candidates, embedding_config=mismatched)


def test_rejects_a_profiles_embedding_model_mismatch(write_artifacts, cluster_map_dict, model_profiles_dict, candidates, embedding_config):
    profiles = copy.deepcopy(model_profiles_dict)
    profiles["embedding"]["model_id"] = "some-other-model"
    cm_path, mp_path = write_artifacts(copy.deepcopy(cluster_map_dict), profiles)
    with pytest.raises(ValueError, match="model-profiles.json embedding"):
        load_routing_context(cm_path, mp_path, 0.0, candidates=candidates, embedding_config=embedding_config)


def test_rejects_a_profiles_dimensions_mismatch(write_artifacts, cluster_map_dict, model_profiles_dict, candidates, embedding_config):
    profiles = copy.deepcopy(model_profiles_dict)
    profiles["embedding"]["dimensions"] = 8
    cm_path, mp_path = write_artifacts(copy.deepcopy(cluster_map_dict), profiles)
    with pytest.raises(ValueError, match="model-profiles.json embedding"):
        load_routing_context(cm_path, mp_path, 0.0, candidates=candidates, embedding_config=embedding_config)


def test_rejects_a_duplicate_model_id_in_profiles(write_artifacts, cluster_map_dict, model_profiles_dict, candidates, embedding_config):
    profiles = copy.deepcopy(model_profiles_dict)
    profiles["models"].append(copy.deepcopy(profiles["models"][0]))  # duplicate "cheap-model"
    cm_path, mp_path = write_artifacts(copy.deepcopy(cluster_map_dict), profiles)
    with pytest.raises(ValueError, match="duplicate model_id"):
        load_routing_context(cm_path, mp_path, 0.0, candidates=candidates, embedding_config=embedding_config)


def test_rejects_a_candidate_with_no_profile_entry(write_artifacts, cluster_map_dict, model_profiles_dict, embedding_config):
    uncalibrated = ModelConfig(
        model_id="uncalibrated-model", provider="test", runner="pi",
        cost_input=0.001, cost_output=0.001, context_window=8192, max_tokens=4096,
    )
    cm_path, mp_path = write_artifacts(copy.deepcopy(cluster_map_dict), copy.deepcopy(model_profiles_dict))
    with pytest.raises(ValueError, match="no entry in model-profiles.json"):
        load_routing_context(cm_path, mp_path, 0.0, candidates=[uncalibrated], embedding_config=embedding_config)


def test_rejects_zero_eligible_candidates(write_artifacts, cluster_map_dict, model_profiles_dict, embedding_config):
    only_control = [
        ModelConfig(model_id="reference-oracle", provider="stub", runner="reference", cost_input=0, cost_output=0, context_window=0, max_tokens=0)
    ]
    cm_path, mp_path = write_artifacts(copy.deepcopy(cluster_map_dict), copy.deepcopy(model_profiles_dict))
    with pytest.raises(ValueError, match="no eligible"):
        load_routing_context(cm_path, mp_path, 0.0, candidates=only_control, embedding_config=embedding_config)


def test_rejects_a_non_contiguous_cluster_map(write_artifacts, cluster_map_dict, model_profiles_dict, candidates, embedding_config):
    cm = copy.deepcopy(cluster_map_dict)
    cm["clusters"][1]["id"] = 7
    cm_path, mp_path = write_artifacts(cm, copy.deepcopy(model_profiles_dict))
    with pytest.raises(ValueError):
        load_routing_context(cm_path, mp_path, 0.0, candidates=candidates, embedding_config=embedding_config)


def test_rejects_a_non_finite_centroid(write_artifacts, cluster_map_dict, model_profiles_dict, candidates, embedding_config):
    cm = copy.deepcopy(cluster_map_dict)
    cm["clusters"][0]["centroid"][0] = math.nan
    cm_path, mp_path = write_artifacts(cm, copy.deepcopy(model_profiles_dict))
    with pytest.raises(ValueError):
        load_routing_context(cm_path, mp_path, 0.0, candidates=candidates, embedding_config=embedding_config)


def test_rejects_mismatched_success_failure_counts_in_profiles(write_artifacts, cluster_map_dict, model_profiles_dict, candidates, embedding_config):
    profiles = copy.deepcopy(model_profiles_dict)
    profiles["models"][0]["global"]["number_of_tasks"] = 999
    cm_path, mp_path = write_artifacts(copy.deepcopy(cluster_map_dict), profiles)
    with pytest.raises(ValueError):
        load_routing_context(cm_path, mp_path, 0.0, candidates=candidates, embedding_config=embedding_config)
