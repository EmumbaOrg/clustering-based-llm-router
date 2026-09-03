"""Replays the routing formula against holdout tasks and reports resolution rate / cost / model
selection across a lambda sweep, plus always-strongest / always-cheapest / oracle baselines — the
metrics from spec §7.

Holdout tasks are actually EXECUTED against every configured model to get real graded outcomes,
but routing decisions themselves use ONLY the calibration profiles (predicted_error) — never
anything computed from the holdout run — matching what the live router has available.

Currently dormant: select_tasks() no longer produces a "holdout" split, so `holdout` below is
always empty and every metric here reports 0/empty until this is rewired against a real benchmark.
"""
from __future__ import annotations

import dataclasses
import logging
import time

from ..common.config import CalibrationConfig, ModelConfig
from ..common.scoring import score_candidates, select_model, static_price_per_1m
from .calibration.calibrate import SelectedTask, image_affinity_key, run_and_log

logger = logging.getLogger(__name__)


def run_holdout_outcomes(
    selected_tasks: list[SelectedTask], models: list[ModelConfig], calibration_config: CalibrationConfig,
) -> dict[tuple[str, str], str]:
    """Runs EVERY model (including controls) against every holdout task. Controls are included so
    the oracle/always-* baselines and a reference-oracle sanity check are all computable from one
    run, without a second pass. Returns {(model_id, task_id): outcome}."""
    # Tasks-outer / models-inner, for the same reason calibrate_models is — one multi-GB image pull
    # per task instead of one per (task, model). See calibrate_models' docstring for the numbers.
    holdout = sorted(
        (st for st in selected_tasks if st.split == "holdout"),
        key=lambda st: image_affinity_key(st.task),
    )
    total = len(models) * len(holdout)
    logger.info(
        f"holdout run started: {len(holdout)} tasks x {len(models)} models = {total} calls "
        "(tasks-outer, models-inner)"
    )
    started = time.monotonic()
    outcomes = {
        (model.model_id, st.task.task_id): run_and_log(st.task, model, calibration_config, i, total).result.outcome
        for i, (st, model) in enumerate(((s, m) for s in holdout for m in models), start=1)
    }
    logger.info(f"holdout run completed in {time.monotonic() - started:.1f}s")
    return outcomes


@dataclasses.dataclass(frozen=True)
class LambdaResult:
    lambda_: float
    resolution_rate: float
    mean_cost: float
    selection_counts: dict[str, int]


@dataclasses.dataclass(frozen=True)
class EvaluationReport:
    holdout_task_count: int
    lambda_results: list[LambdaResult]
    always_strongest: dict
    always_cheapest: dict
    oracle_resolution_rate: float


def evaluate(
    selected_tasks: list[SelectedTask],
    models: list[ModelConfig],
    profiles_by_model: dict[str, dict],
    calibration_config: CalibrationConfig,
    holdout_outcomes: dict[tuple[str, str], str],
) -> EvaluationReport:
    holdout = [st for st in selected_tasks if st.split == "holdout"]
    real_models = [m for m in models if not m.is_control]  # routing must never select a control
    n = len(holdout)

    lambda_results = []
    for lambda_ in calibration_config.lambda_sweep:
        resolved = 0
        total_cost = 0.0
        selection_counts: dict[str, int] = {}
        for st in holdout:
            selected = select_model(score_candidates(st.cluster_id, real_models, profiles_by_model, lambda_))
            if selected is None:
                continue
            selection_counts[selected.model_id] = selection_counts.get(selected.model_id, 0) + 1
            total_cost += selected.static_price
            if holdout_outcomes.get((selected.model_id, st.task.task_id)) == "pass":
                resolved += 1
        lambda_results.append(
            LambdaResult(
                lambda_=lambda_,
                resolution_rate=(resolved / n) if n else 0.0,
                mean_cost=(total_cost / n) if n else 0.0,
                selection_counts=selection_counts,
            )
        )

    def fixed_model_resolution(model_id: str) -> float:
        if n == 0:
            return 0.0
        resolved = sum(1 for st in holdout if holdout_outcomes.get((model_id, st.task.task_id)) == "pass")
        return resolved / n

    strongest = max(real_models, key=static_price_per_1m, default=None)
    cheapest = min(real_models, key=static_price_per_1m, default=None)
    oracle_resolved = sum(
        1 for st in holdout if any(holdout_outcomes.get((m.model_id, st.task.task_id)) == "pass" for m in real_models)
    )

    return EvaluationReport(
        holdout_task_count=n,
        lambda_results=lambda_results,
        always_strongest=(
            {"model_id": strongest.model_id, "resolution_rate": fixed_model_resolution(strongest.model_id)}
            if strongest
            else {}
        ),
        always_cheapest=(
            {"model_id": cheapest.model_id, "resolution_rate": fixed_model_resolution(cheapest.model_id)}
            if cheapest
            else {}
        ),
        oracle_resolution_rate=(oracle_resolved / n) if n else 0.0,
    )
