"""Orchestrates one routing decision: embed -> assign -> score -> select.

Split into `decide_from_vector` (pure, no model download) and `decide` (adds the embedding call)
deliberately: every rule except the encoder call — assignment, scoring, tie-breaks, the
cluster-to-global fallback, the decision record itself — is covered by fast tests against a
hand-authored k=3/dim=4 fixture pair. See
docs/specs/2026-08-10-python-runtime-implementation-plan.md (Phase 3).
"""
from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from ..common.assign import ClusterAssignment, assign_cluster
from ..common.embedding import embed_one
from ..common.scoring import ScoredCandidate, score_candidates, select_model
from .context import RoutingContext

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RoutingDecision:
    selected: ScoredCandidate
    cluster: ClusterAssignment
    scores: list[ScoredCandidate]        # every scored candidate, for audit
    excluded: dict[str, str]             # model_id -> reason; normally empty, kept as a safety net
    lambda_: float
    digest: str
    cluster_map_id: str
    profiles_id: str
    embedding_model_id: str
    prompt_chars: int | None      # None when the caller supplied the vector directly
    embed_ms: float | None        # None when the caller supplied the vector directly
    score_ms: float


def decide_from_vector(
    vector: np.ndarray, ctx: RoutingContext, embed_ms: float | None = None, prompt_chars: int | None = None,
) -> RoutingDecision:
    started = time.monotonic()
    cluster = assign_cluster(vector, ctx.cluster_map)
    scores = score_candidates(cluster.cluster_id, ctx.candidates, ctx.profiles_by_model, ctx.lambda_)
    selected = select_model(scores)
    if selected is None:
        raise ValueError("no eligible candidates scored for this cluster — check model-profiles.json coverage")

    scored_ids = {s.model_id for s in scores}
    excluded = {
        m.model_id: "no usable profile entry (cluster and global lookup both failed)"
        for m in ctx.candidates
        if m.model_id not in scored_ids
    }

    if selected.error_source == "model-global":
        logger.warning(
            f"selected {selected.model_id} via its global error rate — cluster {cluster.cluster_id} "
            f"has no calibration coverage for this model"
        )

    score_ms = (time.monotonic() - started) * 1000
    return RoutingDecision(
        selected=selected,
        cluster=cluster,
        scores=scores,
        excluded=excluded,
        lambda_=ctx.lambda_,
        digest=ctx.digest,
        cluster_map_id=ctx.cluster_map_id,
        profiles_id=ctx.profiles_id,
        embedding_model_id=ctx.embedding.model_id,
        prompt_chars=prompt_chars,
        embed_ms=embed_ms,
        score_ms=score_ms,
    )


def decide(prompt: str, ctx: RoutingContext) -> RoutingDecision:
    started = time.monotonic()
    vector = embed_one(prompt, ctx.embedding)
    embed_ms = (time.monotonic() - started) * 1000
    return decide_from_vector(vector, ctx, embed_ms=embed_ms, prompt_chars=len(prompt))
