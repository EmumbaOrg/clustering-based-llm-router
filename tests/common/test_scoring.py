from router.common.config import ModelConfig
from router.common.scoring import (
    lookup_predicted_error,
    normalise_costs,
    score_candidates,
    select_model,
    static_price_per_1m,
)


def _model(model_id: str, cost_input: float, cost_output: float) -> ModelConfig:
    return ModelConfig(
        model_id=model_id, provider="test", runner="pi", cost_input=cost_input, cost_output=cost_output,
        context_window=8192, max_tokens=4096,
    )


def test_static_price_converts_per_1k_to_per_1m():
    model = _model("m", cost_input=0.001, cost_output=0.002)
    assert static_price_per_1m(model) == (0.001 + 0.002) * 1000


def test_normalise_costs_gives_cheapest_zero_and_priciest_one():
    result = normalise_costs([1.0, 5.0, 10.0])
    assert result[0] == 0.0
    assert result[2] == 1.0
    assert 0.0 < result[1] < 1.0


def test_normalise_costs_all_equal_price_is_zero_not_undefined():
    # All-local candidates at $0 (or any equal price) — the lambda term must vanish, not divide by zero.
    assert normalise_costs([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_normalise_costs_scale_invariance():
    # Per-1k vs per-1M unit choice cannot change a routing decision.
    raw = [1.0, 3.0, 7.0]
    scaled = [p * 1000 for p in raw]
    assert normalise_costs(raw) == normalise_costs(scaled)


def test_lookup_predicted_error_prefers_cluster_specific_over_global():
    profile = {"global": {"smoothed_error_rate": 0.5}, "clusters": {"3": {"smoothed_error_rate": 0.1}}}
    value, source = lookup_predicted_error(profile, cluster_id=3)
    assert value == 0.1
    assert source == "cluster"


def test_lookup_predicted_error_falls_back_to_global_when_cluster_missing():
    profile = {"global": {"smoothed_error_rate": 0.5}, "clusters": {}}
    value, source = lookup_predicted_error(profile, cluster_id=3)
    assert value == 0.5
    assert source == "model-global"


def test_lookup_predicted_error_returns_none_when_model_has_no_profile_at_all():
    assert lookup_predicted_error({}, cluster_id=3) is None


def test_lambda_zero_selects_lowest_error_regardless_of_cost():
    cheap_but_bad = _model("cheap", 0.0001, 0.0001)
    expensive_but_good = _model("expensive", 0.01, 0.01)
    profiles = {
        "cheap": {"global": {"smoothed_error_rate": 0.8}, "clusters": {}},
        "expensive": {"global": {"smoothed_error_rate": 0.1}, "clusters": {}},
    }
    scored = score_candidates(0, [cheap_but_bad, expensive_but_good], profiles, lambda_=0.0)
    selected = select_model(scored)
    assert selected.model_id == "expensive"


def test_large_lambda_selects_cheapest_regardless_of_error():
    cheap_but_bad = _model("cheap", 0.0001, 0.0001)
    expensive_but_good = _model("expensive", 0.01, 0.01)
    profiles = {
        "cheap": {"global": {"smoothed_error_rate": 0.8}, "clusters": {}},
        "expensive": {"global": {"smoothed_error_rate": 0.1}, "clusters": {}},
    }
    scored = score_candidates(0, [cheap_but_bad, expensive_but_good], profiles, lambda_=100.0)
    selected = select_model(scored)
    assert selected.model_id == "cheap"


def test_exact_score_tie_resolves_to_lower_model_id():
    a = _model("a", 0.001, 0.001)
    b = _model("b", 0.001, 0.001)
    profiles = {
        "a": {"global": {"smoothed_error_rate": 0.3}, "clusters": {}},
        "b": {"global": {"smoothed_error_rate": 0.3}, "clusters": {}},
    }
    scored = score_candidates(0, [a, b], profiles, lambda_=0.1)
    selected = select_model(scored)
    assert selected.model_id == "a"


def test_a_model_with_no_profile_is_excluded_not_defaulted():
    has_profile = _model("has-profile", 0.001, 0.001)
    no_profile = _model("no-profile", 0.001, 0.001)
    profiles = {"has-profile": {"global": {"smoothed_error_rate": 0.5}, "clusters": {}}}
    scored = score_candidates(0, [has_profile, no_profile], profiles, lambda_=0.1)
    assert len(scored) == 1
    assert scored[0].model_id == "has-profile"


def test_select_model_returns_none_when_nothing_is_eligible():
    assert select_model([]) is None
