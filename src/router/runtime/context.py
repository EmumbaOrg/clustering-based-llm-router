"""Loads and hard-validates everything a routing decision needs: both artifacts, the candidate
roster, and lambda. See docs/specs/2026-08-10-python-runtime-implementation-plan.md (Phase 2) for
the reasoning behind each check below. Every check here raises rather than degrades.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
from pathlib import Path

from ..common.artifacts import validate_cluster_map, validate_profiles
from ..common.assign import ClusterMap, cluster_map_from_dict
from ..common.config import (
    EmbeddingConfig,
    EmbeddingInputConfig,
    ModelConfig,
    TruncationConfig,
    load_embedding_config,
    load_models_config,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RoutingContext:
    cluster_map: ClusterMap
    embedding: EmbeddingConfig      # derived from cluster-map.json, NOT config/embedding.yaml
    profiles_by_model: dict[str, dict]
    candidates: list[ModelConfig]   # non-control, every one profile-backed
    lambda_: float
    cluster_map_id: str
    profiles_id: str
    digest: str


def _read_json(path: Path, what: str) -> dict:
    if not path.exists():
        raise ValueError(f"{what} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _embedding_config_from_artifact(raw_map: dict) -> EmbeddingConfig:
    """Builds the EmbeddingConfig the runtime must actually embed with — from the artifact's
    `embedding` block, never config/embedding.yaml directly. This is the highest-risk silent-drift
    point in the runtime design — see docs/engineering-notes.md, "Embedding config source of
    truth" before touching this."""
    emb = raw_map["embedding"]
    truncation = TruncationConfig(**emb["input"]["truncation"])
    input_config = EmbeddingInputConfig(normalisation=emb["input"]["normalisation"], truncation=truncation)
    return EmbeddingConfig(
        model_id=emb["model_id"],
        dimensions=emb["dimensions"],
        normalisation=emb["normalisation"],
        distance=emb["distance"],
        input=input_config,
    )


def _compute_digest(lambda_: float, embedding_model_id: str, cluster_map_id: str, profiles_id: str, candidates: list[ModelConfig]) -> str:
    """Pins everything that fed the decision besides the embedding vector itself. Byte-exact
    embeddings aren't reproducible across runs (fp16 vs fp32, batch composition, kernel
    differences) — what's reproducible is the decision GIVEN the vector, so this digest is what
    makes "did anything change between run A and run B" answerable."""
    payload = {
        "lambda": lambda_,
        "embedding_model_id": embedding_model_id,
        "cluster_map_id": cluster_map_id,
        "profiles_id": profiles_id,
        "candidates": sorted((m.model_id, m.cost_input, m.cost_output) for m in candidates),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def load_routing_context(
    cluster_map_path: Path,
    model_profiles_path: Path,
    lambda_: float,
    candidates: list[ModelConfig] | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> RoutingContext:
    """`candidates` and `embedding_config` are injectable so tests can run this against fixtures
    with zero dependence on whatever happens to be in config/ on a given machine; both default to
    the real config/*.yaml loaders."""
    if not math.isfinite(lambda_) or lambda_ < 0:
        raise ValueError(f"lambda must be finite and >= 0, got {lambda_}")

    raw_map = _read_json(cluster_map_path, "cluster map")
    validate_cluster_map(raw_map)  # schema (incl. schema_version/normalisation/distance/dtype
    # consts) + cross-field invariants (contiguous ids, centroid length, non-finite values)
    cluster_map = cluster_map_from_dict(raw_map)
    artifact_embedding = _embedding_config_from_artifact(raw_map)

    configured_embedding = embedding_config if embedding_config is not None else load_embedding_config()
    if configured_embedding.model_id != artifact_embedding.model_id:
        raise ValueError(
            f"configured embedding model {configured_embedding.model_id!r} does not match "
            f"cluster-map.json's {artifact_embedding.model_id!r} — rebuild the artifact or fix "
            f"the config; they must agree."
        )

    raw_profiles = _read_json(model_profiles_path, "model profiles")
    validate_profiles(raw_profiles)  # schema + cross-field invariants

    if raw_profiles["cluster_map_id"] != cluster_map.artifact_id:
        raise ValueError(
            f"model-profiles.json.cluster_map_id={raw_profiles['cluster_map_id']!r} does not "
            f"match cluster-map.json.artifact_id={cluster_map.artifact_id!r} — these artifacts "
            f"were not built together; cluster ids would be silently permuted between them."
        )

    profiles_embedding = raw_profiles["embedding"]
    if (
        profiles_embedding["model_id"] != artifact_embedding.model_id
        or profiles_embedding["dimensions"] != artifact_embedding.dimensions
    ):
        raise ValueError(
            f"model-profiles.json embedding ({profiles_embedding['model_id']!r}, "
            f"{profiles_embedding['dimensions']}d) does not match cluster-map.json embedding "
            f"({artifact_embedding.model_id!r}, {artifact_embedding.dimensions}d) — calibration "
            f"tasks were clustered with a different model."
        )

    profiles_by_model: dict[str, dict] = {}
    for model_entry in raw_profiles["models"]:
        model_id = model_entry["model_id"]
        if model_id in profiles_by_model:
            raise ValueError(f"model-profiles.json has a duplicate model_id: {model_id!r}")
        profiles_by_model[model_id] = model_entry

    roster = candidates if candidates is not None else load_models_config()
    real_candidates = [m for m in roster if not m.is_control]
    if not real_candidates:
        raise ValueError("no eligible (non-control) candidate models configured")

    missing = [m.model_id for m in real_candidates if m.model_id not in profiles_by_model]
    if missing:
        raise ValueError(
            f"candidate model(s) {missing} have no entry in model-profiles.json — run calibration "
            f"for them before they can be routed to."
        )

    digest = _compute_digest(
        lambda_, artifact_embedding.model_id, cluster_map.artifact_id, raw_profiles["artifact_id"], real_candidates,
    )

    return RoutingContext(
        cluster_map=cluster_map,
        embedding=artifact_embedding,
        profiles_by_model=profiles_by_model,
        candidates=real_candidates,
        lambda_=lambda_,
        cluster_map_id=cluster_map.artifact_id,
        profiles_id=raw_profiles["artifact_id"],
        digest=digest,
    )
