"""Loads ../../config/*.yaml (the repo-root config directory, not this package) into dataclasses.

Kept deliberately dumb — plain dataclasses, no validation beyond what YAML/attribute access give
for free. config/ is human-edited and small; a full schema is reserved for pipeline *output*
(see artifacts-schema/), not this input config.
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
    k_candidates: list[int]
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
    rate_limit_rpm: int | None = None  # the provider's documented free-tier requests/minute cap
    # for this model, or None for a provider with no meaningful limit (local, controls). Paced by
    # runner.py's RateLimiter before every real API call — see its docstring for why.
    supports_tool_calls: bool = True  # False for local llama.cpp providers, confirmed empirically
    # this session: the model attempts a tool call (in its own training-time dialect, e.g.
    # `<function-calls>{...}</function-calls>`) but llama.cpp's OpenAI-compatible endpoint never
    # translates that into the wire protocol's `message.tool_calls` field — Pi only recognizes a
    # tool call there, sees plain text instead, and returns the inert tool-call text as the "final
    # answer". runner.py's build_prompt uses this to avoid inviting tool use a provider can't
    # deliver on.

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
    holdout_fraction: float
    seed: int
    lambda_sweep: list[float]
    # Additive, backward-compatible: absent/empty means every source grades under the plain
    # task_timeout_seconds, same as before this field existed. Only GRADING calls (Docker-based
    # sources in particular, which can need far longer than a Groq completion) read this — `run_pi`
    # always uses task_timeout_seconds directly, so a slow grader can't also give a hung LLM call
    # the same long leash.
    task_timeout_overrides: dict[str, int] = dataclasses.field(default_factory=dict)

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
        holdout_fraction=data["holdout_fraction"],
        seed=data["seed"],
        lambda_sweep=data["lambda_sweep"],
        task_timeout_overrides=data.get("task_timeout_overrides", {}),
    )


def load_models_config(path: Path | None = None) -> list[ModelConfig]:
    data = yaml.safe_load((path or CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))
    models = [ModelConfig(**entry) for entry in data["models"]]
    ids = [m.model_id for m in models]
    if len(ids) != len(set(ids)):
        raise ValueError(f"config/models.yaml has duplicate model_id values: {ids}")
    return models
