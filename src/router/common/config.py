"""Loads ../../config/*.yaml (the repo-root config directory, not this package) into dataclasses.

Kept deliberately dumb — plain dataclasses, no validation beyond what YAML/attribute access give
for free. config/ is human-edited and small; a full schema is reserved for pipeline *output*
(see artifacts-schema/), not this input config.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


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

    @property
    def is_control(self) -> bool:
        """Grader-validation controls, not real candidates — a router must never select one."""
        return self.runner in ("reference", "null")
