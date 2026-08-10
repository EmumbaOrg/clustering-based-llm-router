from __future__ import annotations

import dataclasses
import logging
import random
import time

import numpy as np

from ...common.assign import ClusterMap, assign_cluster
from ...common.config import CalibrationConfig, EmbeddingConfig, ModelConfig
from ...common.embedding import embed_texts
from . import runner as runner_mod
from .grading import base as grading_base
from .grading import bigcodebench, ds1000, swegym, swesmith
from .grading.base import GradeResult, Outcome, Task
from .tasks import load_gradeable_tasks

logger = logging.getLogger(__name__)

_GRADERS = {
    "bigcodebench": bigcodebench.grade,
    "ds1000": ds1000.grade,
    "swe-smith": swesmith.grade,
    "swe-gym": swegym.grade,
}


@dataclasses.dataclass(frozen=True)
class SelectedTask:
    task: Task
    cluster_id: int
    split: str  # "calibration" | "holdout"


def select_tasks(
    calibration_config: CalibrationConfig,
    embedding_config: EmbeddingConfig,
    cluster_map: ClusterMap,
) -> list[SelectedTask]:
    """Stratified-by-cluster task selection, bounded to a fixed embedding cost regardless of how
    large the underlying datasets are — see calibration.yaml's `candidate_pool_oversample` comment
    for why this doesn't just embed every available row."""
    rng = random.Random(calibration_config.seed)
    k = cluster_map.centroids.shape[0]
    pool_size_per_source = (
        calibration_config.tasks_per_cluster * k * calibration_config.candidate_pool_oversample
    )

    candidate_tasks: list[Task] = []
    for source in calibration_config.gradeable_sources:
        all_tasks = sorted(load_gradeable_tasks(source), key=lambda t: t.task_id)  # order-independent before shuffling
        rng.shuffle(all_tasks)
        pool = all_tasks[:pool_size_per_source]
        candidate_tasks.extend(pool)
        logger.info(f"candidate pool for {source}: {len(pool)} tasks")

    candidate_tasks.sort(key=lambda t: t.task_id)  # deterministic embedding order
    vectors = embed_texts([t.prompt for t in candidate_tasks], embedding_config)

    by_cluster: dict[int, list[Task]] = {}
    for task, vector in zip(candidate_tasks, vectors):
        assignment = assign_cluster(np.asarray(vector, dtype=np.float64), cluster_map)
        by_cluster.setdefault(assignment.cluster_id, []).append(task)

    selected: list[SelectedTask] = []
    for cluster_id, tasks_in_cluster in by_cluster.items():
        tasks_in_cluster = sorted(tasks_in_cluster, key=lambda t: t.task_id)
        rng.shuffle(tasks_in_cluster)
        chosen = tasks_in_cluster[: calibration_config.tasks_per_cluster]
        holdout_count = round(len(chosen) * calibration_config.holdout_fraction)
        for i, task in enumerate(chosen):
            split = "holdout" if i < holdout_count else "calibration"
            selected.append(SelectedTask(task=task, cluster_id=cluster_id, split=split))

    n_calibration = sum(1 for s in selected if s.split == "calibration")
    n_holdout = sum(1 for s in selected if s.split == "holdout")
    logger.info(
        f"selected {len(selected)} tasks across {len(by_cluster)} clusters "
        f"({n_calibration} calibration, {n_holdout} holdout)"
    )
    return selected


def run_and_grade(task: Task, model: ModelConfig, calibration_config: CalibrationConfig) -> GradeResult:
    grader = _GRADERS.get(task.source)
    if grader is None:
        raise ValueError(f"no grader for source {task.source!r}")
    timeout = calibration_config.task_timeout_seconds

    if model.runner == "reference":
        if task.source == "swe-smith":
            return swesmith.grade_reference(task, timeout_seconds=timeout)
        return grader(task, task.reference_solution, timeout_seconds=timeout)

    if model.runner == "null":
        if task.source == "swe-smith":
            return swesmith.grade_null(task, timeout_seconds=timeout)
        return grader(task, "", timeout_seconds=timeout)

    if model.runner == "pi":
        run_result = runner_mod.run_pi(task, model, timeout_seconds=timeout)
        if run_result.rate_limited:
            # An infra/quota rejection, not the model failing to answer — excluded rather than
            # counted as a wrong answer, same reasoning as error_timeout/error_missing_dep.
            return GradeResult(outcome="error_harness", detail=run_result.detail)
        if run_result.solution is None:
            return GradeResult(outcome="error_no_solution", detail=run_result.detail)
        return grader(task, run_result.solution, timeout_seconds=timeout)

    raise ValueError(f"unknown runner {model.runner!r} for model {model.model_id}")


def run_and_log(task: Task, model: ModelConfig, calibration_config: CalibrationConfig, index: int, total: int) -> GradeResult:
    """`run_and_grade` plus the timing/progress log line — shared by calibrate_model's per-model
    loop below and evaluate.py's per-model x per-task holdout loop, which otherwise duplicate this
    exact started/duration/excluded-vs-not branch."""
    started = time.monotonic()
    result = run_and_grade(task, model, calibration_config)
    duration_ms = round((time.monotonic() - started) * 1000)
    if result.outcome in grading_base.EXCLUDED_OUTCOMES:
        logger.warning(f"[{index}/{total}] {model.model_id} task {task.task_id} excluded: {result.outcome} ({duration_ms}ms)")
    else:
        logger.info(f"[{index}/{total}] {model.model_id} task {task.task_id} -> {result.outcome} ({duration_ms}ms)")
    return result


@dataclasses.dataclass(frozen=True)
class ClusterStats:
    number_of_tasks: int
    number_succeeded: int
    number_failed: int
    raw_error_rate: float
    smoothed_error_rate: float
    excluded: dict[str, int]


def _stats_from_outcomes(
    outcomes: list[Outcome], global_raw_error_rate: float | None, prior_weight: float,
) -> ClusterStats:
    graded = [o for o in outcomes if o in grading_base.GRADED_OUTCOMES]
    excluded = [o for o in outcomes if o in grading_base.EXCLUDED_OUTCOMES]
    n = len(graded)
    failed = sum(1 for o in graded if o != "pass")
    succeeded = n - failed
    raw_rate = failed / n if n > 0 else 0.0
    # When no global rate is supplied (we ARE computing the global stats), shrink toward the raw
    # rate itself — algebraically this makes smoothed_error_rate == raw_error_rate at global scope,
    # which is correct: there's no "more global" number to shrink a global rate toward.
    global_rate = global_raw_error_rate if global_raw_error_rate is not None else raw_rate
    denom = n + prior_weight
    smoothed_rate = (failed + prior_weight * global_rate) / denom if denom > 0 else 0.0
    excluded_counts: dict[str, int] = {}
    for o in excluded:
        excluded_counts[o] = excluded_counts.get(o, 0) + 1
    return ClusterStats(
        number_of_tasks=n,
        number_succeeded=succeeded,
        number_failed=failed,
        raw_error_rate=raw_rate,
        smoothed_error_rate=smoothed_rate,
        excluded=excluded_counts,
    )


@dataclasses.dataclass(frozen=True)
class ModelCalibrationResult:
    model: ModelConfig
    global_stats: ClusterStats
    cluster_stats: dict[int, ClusterStats]


def calibrate_model(
    model: ModelConfig, selected_tasks: list[SelectedTask], calibration_config: CalibrationConfig,
) -> ModelCalibrationResult:
    calibration_only = [st for st in selected_tasks if st.split == "calibration"]
    total = len(calibration_only)
    logger.info(f"starting calibration for {model.model_id}: {total} tasks")
    per_task_outcomes: list[tuple[SelectedTask, GradeResult]] = [
        (st, run_and_log(st.task, model, calibration_config, i, total))
        for i, st in enumerate(calibration_only, start=1)
    ]

    prior_weight = calibration_config.smoothing.prior_weight
    all_outcomes = [r.outcome for _, r in per_task_outcomes]
    global_stats = _stats_from_outcomes(all_outcomes, global_raw_error_rate=None, prior_weight=prior_weight)

    by_cluster: dict[int, list[Outcome]] = {}
    for st, result in per_task_outcomes:
        by_cluster.setdefault(st.cluster_id, []).append(result.outcome)

    cluster_stats = {
        cid: _stats_from_outcomes(outcomes, global_raw_error_rate=global_stats.raw_error_rate, prior_weight=prior_weight)
        for cid, outcomes in by_cluster.items()
    }
    logger.info(
        f"calibration completed for {model.model_id}: "
        f"{global_stats.number_succeeded}/{global_stats.number_of_tasks} pass, "
        f"smoothed_error_rate={global_stats.smoothed_error_rate:.3f}"
    )
    return ModelCalibrationResult(model=model, global_stats=global_stats, cluster_stats=cluster_stats)
