"""Shared fixtures for runtime tests. One hand-authored k=3/dim=4 artifact pair
(fixtures/cluster-map.json, fixtures/model-profiles.json) covers every hard-fail rule and scoring
path as fast pure tests — every failure case is that pair, deep-copied and mutated in one field.

Fixture roster (see docs/specs/2026-08-10-python-runtime-implementation-plan.md, Phase 5):

| Model            | price/1M | global | c0   | c1   | c2                |
|------------------|----------|--------|------|------|-------------------|
| cheap-model      | 0.3      | 0.80   | 0.70 | 0.90 | 0.80              |
| mid-model        | 5.0      | 0.50   | 0.45 | 0.55 | (absent -> global) |
| strong-model     | 35.0     | 0.20   | 0.15 | 0.25 | 0.20              |
| reference-oracle | 0.0      | 0.00   | -    | -    | control, excluded |

Cluster centroids are the basis vectors e1/e2/e3 in 4 dimensions, so which cluster a hand-written
query vector assigns to is obvious by inspection.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from router.common.config import (
    EmbeddingConfig,
    EmbeddingInputConfig,
    ModelConfig,
    TruncationConfig,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def cluster_map_dict() -> dict:
    """A fresh deep copy every call, so a test can mutate it freely without affecting others."""
    return _load("cluster-map.json")


@pytest.fixture
def model_profiles_dict() -> dict:
    return _load("model-profiles.json")


@pytest.fixture
def write_artifacts(tmp_path):
    """Writes a (possibly mutated) cluster-map/model-profiles pair to tmp_path and returns their
    paths, so each hard-fail test is: deep-copy a fixture dict, mutate one field, write, assert."""

    def _write(cluster_map: dict, model_profiles: dict) -> tuple[Path, Path]:
        cm_path = tmp_path / "cluster-map.json"
        mp_path = tmp_path / "model-profiles.json"
        cm_path.write_text(json.dumps(cluster_map), encoding="utf-8")
        mp_path.write_text(json.dumps(model_profiles), encoding="utf-8")
        return cm_path, mp_path

    return _write


def _model(model_id: str, cost_input: float, cost_output: float, runner: str = "pi") -> ModelConfig:
    return ModelConfig(
        model_id=model_id, provider="test", runner=runner, cost_input=cost_input, cost_output=cost_output,
        context_window=8192, max_tokens=4096,
    )


@pytest.fixture
def candidates() -> list[ModelConfig]:
    """Prices chosen so (cost_input + cost_output) * 1000 == the price/1M column in the roster
    table above: 0.3, 5.0, 35.0."""
    return [
        _model("cheap-model", 0.00015, 0.00015),
        _model("mid-model", 0.0025, 0.0025),
        _model("strong-model", 0.0175, 0.0175),
        _model("reference-oracle", 0.0, 0.0, runner="reference"),
    ]


@pytest.fixture
def embedding_config() -> EmbeddingConfig:
    """Matches the fixture artifacts' `embedding` block exactly — used as the injected
    `embedding_config` override so context tests never depend on config/embedding.yaml."""
    return EmbeddingConfig(
        model_id="test-embedding-model", dimensions=4, normalisation="l2", distance="euclidean",
        input=EmbeddingInputConfig(normalisation="crlf-lf+trim", truncation=TruncationConfig(unit="chars", max=8000, strategy="head")),
    )


@pytest.fixture
def load_context(write_artifacts, cluster_map_dict, model_profiles_dict, candidates, embedding_config):
    """The common case: write the unmodified fixture pair and load a context at a given lambda.
    Returns a callable so each test can pick its own lambda without re-declaring every fixture."""

    def _load_context(lambda_: float = 0.0):
        from router.runtime.context import load_routing_context

        cm_path, mp_path = write_artifacts(copy.deepcopy(cluster_map_dict), copy.deepcopy(model_profiles_dict))
        return load_routing_context(cm_path, mp_path, lambda_, candidates=candidates, embedding_config=embedding_config)

    return _load_context
