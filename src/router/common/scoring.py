"""The routing-score arithmetic: predicted_error + lambda * normalised_cost -> selected model.

Shared between ../pipeline/evaluate.py (which replays this over a holdout split for the
lambda-sweep report) and ../runtime (which applies it to a single live request) — both must use
the exact same formula and tie-break rule, or the offline evaluation numbers stop describing what
the runtime actually does.
"""
from __future__ import annotations

import dataclasses

from .config import ModelConfig


def static_price_per_1m(model: ModelConfig) -> float:
    """$ per 1M tokens (input + output). Config stores $ per 1k, hence *1000. Min-max
    normalisation is invariant under this scaling, so the unit choice can't change any routing
    decision — converting anyway keeps this figure comparable to published provider pricing."""
    return (model.cost_input + model.cost_output) * 1000


def normalise_costs(prices: list[float]) -> list[float]:
    if not prices:
        return []
    lo, hi = min(prices), max(prices)
    span = hi - lo
    if span <= 0:
        # Every eligible candidate is the same price (e.g. an all-local set) — the lambda term
        # vanishes for all of them, which is correct: cost isn't a differentiator here.
        return [0.0 for _ in prices]
    return [min(1.0, max(0.0, (p - lo) / span)) for p in prices]


@dataclasses.dataclass(frozen=True)
class ScoredCandidate:
    model_id: str
    predicted_error: float
    error_source: str  # "cluster" | "model-global"
    static_price: float
    normalised_cost: float
    routing_score: float


def lookup_predicted_error(profile_entry: dict, cluster_id: int) -> tuple[float, str] | None:
    cluster_key = str(cluster_id)
    clusters = profile_entry.get("clusters", {})
    if cluster_key in clusters:
        return clusters[cluster_key]["smoothed_error_rate"], "cluster"
    if "global" in profile_entry:
        return profile_entry["global"]["smoothed_error_rate"], "model-global"
    return None


def score_candidates(
    cluster_id: int, candidates: list[ModelConfig], profiles_by_model: dict[str, dict], lambda_: float,
) -> list[ScoredCandidate]:
    priced = []
    for model in candidates:
        profile = profiles_by_model.get(model.model_id)
        if profile is None:
            continue  # no-profile: excluded, never defaulted (see artifacts-schema/README.md)
        lookup = lookup_predicted_error(profile, cluster_id)
        if lookup is None:
            continue
        error, source = lookup
        priced.append((model, error, source, static_price_per_1m(model)))
    if not priced:
        return []

    normalised = normalise_costs([p[3] for p in priced])
    return [
        ScoredCandidate(
            model_id=model.model_id,
            predicted_error=error,
            error_source=source,
            static_price=price,
            normalised_cost=norm_cost,
            routing_score=error + lambda_ * norm_cost,
        )
        for (model, error, source, price), norm_cost in zip(priced, normalised)
    ]


def select_model(scored: list[ScoredCandidate]) -> ScoredCandidate | None:
    if not scored:
        return None
    return min(scored, key=lambda s: (s.routing_score, s.model_id))
