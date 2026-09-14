"""Loads ../../config/*.yaml (the repo-root config directory, not this package) into dataclasses.

config/ is human-edited, so there's no validation beyond what YAML/attribute access give for free; schemas live in artifacts-schema/ for pipeline output instead.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


@dataclasses.dataclass(frozen=True)
class TruncationConfig:
    unit: str
    max: int
    strategy: str


@dataclasses.dataclass(frozen=True)
class EmbeddingInputConfig:
    normalisation: str
    truncation: TruncationConfig


@dataclasses.dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str
    dimensions: int
    normalisation: str
    distance: str
    input: EmbeddingInputConfig


def load_embedding_config(path: Path | None = None) -> EmbeddingConfig:
    data = yaml.safe_load((path or CONFIG_DIR / "embedding.yaml").read_text(encoding="utf-8"))
    truncation = TruncationConfig(**data["input"]["truncation"])
    input_config = EmbeddingInputConfig(normalisation=data["input"]["normalisation"], truncation=truncation)
    return EmbeddingConfig(
        model_id=data["model_id"],
        dimensions=data["dimensions"],
        normalisation=data["normalisation"],
        distance=data["distance"],
        input=input_config,
    )


@dataclasses.dataclass(frozen=True)
class ClusteringConfig:
    default_k: int
    seed: int
    n_init: int


def load_clustering_config(path: Path | None = None) -> ClusteringConfig:
    data = yaml.safe_load((path or CONFIG_DIR / "clustering.yaml").read_text(encoding="utf-8"))
    return ClusteringConfig(**data)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    model_id: str
    provider: str
    runner: str  # "reference" | "null" | "pi" — see config/models.yaml's own docstring
    cost_input: float
    cost_output: float
    context_window: int
    max_tokens: int
    supports_tool_calls: bool = True  # False for local llama.cpp providers

    @property
    def is_control(self) -> bool:
        """Grader-validation controls, not real candidates — a router must never select one."""
        return self.runner in ("reference", "null")


@dataclasses.dataclass(frozen=True)
class SmoothingConfig:
    method: str
    prior_weight: float


@dataclasses.dataclass(frozen=True)
class CalibrationConfig:
    gradeable_sources: list[str]
    tasks_per_cluster: int
    task_timeout_seconds: int
    smoothing: SmoothingConfig
    seed: int
    lambda_sweep: list[float]
    # Optional per-source grading timeout overrides (Docker-based sources only); absent/empty falls back to task_timeout_seconds.
    task_timeout_overrides: dict[str, int] = dataclasses.field(default_factory=dict)
    # Optional {category: fraction} sampling target; absent/empty keeps flat stratified-by-cluster sampling (see calibrate.py's CATEGORY_SOURCES).
    category_mix: dict[str, float] = dataclasses.field(default_factory=dict)

    def grading_timeout_for(self, source: str) -> int:
        return self.task_timeout_overrides.get(source, self.task_timeout_seconds)


def load_calibration_config(path: Path | None = None) -> CalibrationConfig:
    data = yaml.safe_load((path or CONFIG_DIR / "calibration.yaml").read_text(encoding="utf-8"))
    smoothing = SmoothingConfig(**data["smoothing"])
    return CalibrationConfig(
        gradeable_sources=data["gradeable_sources"],
        tasks_per_cluster=data["tasks_per_cluster"],
        task_timeout_seconds=data["task_timeout_seconds"],
        smoothing=smoothing,
        seed=data["seed"],
        lambda_sweep=data["lambda_sweep"],
        task_timeout_overrides=data.get("task_timeout_overrides", {}),
        category_mix=data.get("category_mix", {}),
    )


def load_models_config(path: Path | None = None) -> list[ModelConfig]:
    data = yaml.safe_load((path or CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))
    models = [ModelConfig(**entry) for entry in data["models"]]
    ids = [m.model_id for m in models]
    if len(ids) != len(set(ids)):
        raise ValueError(f"config/models.yaml has duplicate model_id values: {ids}")
    return models
