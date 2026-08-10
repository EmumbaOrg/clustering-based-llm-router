import dataclasses

import numpy as np

from router.runtime.decide import decide_from_vector

V0 = np.array([1.0, 0.0, 0.0, 0.0])  # nearest centroid: cluster 0
V1 = np.array([0.0, 1.0, 0.0, 0.0])  # cluster 1
V2 = np.array([0.0, 0.0, 1.0, 0.0])  # cluster 2, where mid-model has no calibration coverage


def test_assigns_to_the_nearest_cluster(load_context):
    decision = decide_from_vector(V0, load_context(lambda_=0.0))
    assert decision.cluster.cluster_id == 0


def test_lambda_zero_selects_lowest_error_regardless_of_cost(load_context):
    decision = decide_from_vector(V0, load_context(lambda_=0.0))
    assert decision.selected.model_id == "strong-model"
    assert decision.selected.error_source == "cluster"


def test_moderate_lambda_selects_the_middle_model(load_context):
    # error + lambda*normalised_cost: mid (0.45+0.5*0.135=0.518) beats strong (0.65) and cheap (0.70).
    decision = decide_from_vector(V0, load_context(lambda_=0.5))
    assert decision.selected.model_id == "mid-model"


def test_large_lambda_selects_the_cheapest_model(load_context):
    decision = decide_from_vector(V0, load_context(lambda_=100.0))
    assert decision.selected.model_id == "cheap-model"


def test_reference_oracle_is_never_selected_despite_a_zero_error_rate(load_context):
    for lambda_ in (0.0, 0.5, 100.0):
        decision = decide_from_vector(V0, load_context(lambda_=lambda_))
        assert decision.selected.model_id != "reference-oracle"
        assert all(s.model_id != "reference-oracle" for s in decision.scores)


def test_falls_back_to_the_models_global_rate_when_a_cluster_has_no_coverage(load_context, caplog):
    # mid-model has no cluster-2 entry; at lambda=1 it's the argmin using its global rate.
    decision = decide_from_vector(V2, load_context(lambda_=1.0))
    assert decision.selected.model_id == "mid-model"
    assert decision.selected.error_source == "model-global"
    assert "global error rate" in caplog.text


def test_cluster_two_uses_calibrated_rates_for_models_that_have_them(load_context):
    decision = decide_from_vector(V2, load_context(lambda_=0.0))
    strong = next(s for s in decision.scores if s.model_id == "strong-model")
    assert strong.error_source == "cluster"
    assert strong.predicted_error == 0.2


def test_excluded_is_empty_when_every_candidate_has_a_usable_profile(load_context):
    decision = decide_from_vector(V0, load_context(lambda_=0.5))
    assert decision.excluded == {}


def test_scaling_every_price_by_1000_changes_no_decision(load_context, candidates):
    scaled = [dataclasses.replace(m, cost_input=m.cost_input * 1000, cost_output=m.cost_output * 1000) for m in candidates]
    ctx_original = load_context(lambda_=0.5)
    ctx_scaled = dataclasses.replace(ctx_original, candidates=[m for m in scaled if not m.is_control])
    original = decide_from_vector(V0, ctx_original)
    rescaled = decide_from_vector(V0, ctx_scaled)
    assert original.selected.model_id == rescaled.selected.model_id


def test_decision_carries_the_context_identifiers(load_context):
    ctx = load_context(lambda_=0.05)
    decision = decide_from_vector(V1, ctx)
    assert decision.lambda_ == 0.05
    assert decision.digest == ctx.digest
    assert decision.cluster_map_id == ctx.cluster_map_id
    assert decision.profiles_id == ctx.profiles_id
    assert decision.embedding_model_id == ctx.embedding.model_id


def test_prompt_chars_and_embed_ms_are_none_when_the_vector_is_supplied_directly(load_context):
    decision = decide_from_vector(V0, load_context(lambda_=0.0))
    assert decision.prompt_chars is None
    assert decision.embed_ms is None
    assert decision.score_ms >= 0.0


def test_runner_up_is_the_second_nearest_cluster(load_context):
    # slightly off from e1 toward e2, still closest to cluster 0, second-closest cluster 1
    v = np.array([0.9, 0.4, 0.0, 0.0])
    decision = decide_from_vector(v, load_context(lambda_=0.0))
    assert decision.cluster.cluster_id == 0
    assert decision.cluster.runner_up_cluster_id == 1
