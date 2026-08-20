from __future__ import annotations

import csv
import dataclasses
import logging
import random
import time
from pathlib import Path

import numpy as np

from ...common.artifacts import ARTIFACTS_DIR
from ...common.assign import ClusterMap, assign_cluster
from ...common.config import CalibrationConfig, EmbeddingConfig, ModelConfig
from ...common.embedding import embed_texts
from . import runner as runner_mod
from .grading import base as grading_base
from .grading import bigcodebench, ds1000, multiswerl, swegym, swesmith
from .grading.base import GradeResult, Outcome, Task
from .tasks import load_gradeable_tasks

logger = logging.getLogger(__name__)

_GRADERS = {
    "bigcodebench": bigcodebench.grade,
    "ds1000": ds1000.grade,
    "swe-smith": swesmith.grade,
    "swe-gym": swegym.grade,
    "multi-swe-rl": multiswerl.grade,
}

# Patch-based sources leave `reference_solution` empty (see tasks.py) and grade the gold/empty
# patch through a dedicated entry point instead of the generic `grader(task, solution)` shape —
# see grade_reference/grade_null below, which are the single source of truth cli.py's
# validate-graders gate and run_and_grade's reference/null branches both call through.
_REFERENCE_GRADERS = {
    "swe-smith": swesmith.grade_reference,
    "swe-gym": swegym.grade_reference,
    "multi-swe-rl": multiswerl.grade_reference,
}
_NULL_GRADERS = {
    "swe-smith": swesmith.grade_null,
    "swe-gym": swegym.grade_null,
    "multi-swe-rl": multiswerl.grade_null,
}


def grade_reference(task: Task, timeout_seconds: int) -> GradeResult:
    special = _REFERENCE_GRADERS.get(task.source)
    if special is not None:
        return special(task, timeout_seconds=timeout_seconds)
    return _GRADERS[task.source](task, task.reference_solution, timeout_seconds=timeout_seconds)


def grade_null(task: Task, timeout_seconds: int) -> GradeResult:
    special = _NULL_GRADERS.get(task.source)
    if special is not None:
        return special(task, timeout_seconds=timeout_seconds)
    return _GRADERS[task.source](task, "", timeout_seconds=timeout_seconds)


# Debug-only preview length for expected-vs-provided logging (see run_and_grade's pi branch).
# Long enough to see the shape of a real answer; short enough that a 1.4MB swe-gym patch doesn't
# flood the log. --log-level DEBUG is scoped to the `router` logger only (logging_config.py), so
# this never mixes with third-party DEBUG noise from datasets/huggingface_hub/sentence-transformers.
_LOG_PREVIEW_CHARS = 1000


def _preview(text: str) -> str:
    text = text.strip()
    if len(text) <= _LOG_PREVIEW_CHARS:
        return text
    return text[:_LOG_PREVIEW_CHARS] + f" ...[{len(text) - _LOG_PREVIEW_CHARS} more chars]"


def _expected_solution(task: Task) -> str:
    # bigcodebench/ds1000 carry the gold answer directly in reference_solution. Patch-based
    # sources (swe-smith/swe-gym) leave it empty on purpose (see tasks.py) — their gold fix is
    # task.row["patch"] instead, which run_and_grade's reference/null branches use directly.
    return task.reference_solution or str(task.row.get("patch", ""))


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


def run_and_grade(task: Task, model: ModelConfig, calibration_config: CalibrationConfig) -> tuple[GradeResult, str | None]:
    """Returns (GradeResult, solution) — `solution` is what the candidate actually produced (the
    gold answer for `reference`, empty for `null`, the extracted/diff solution for `pi` — falling
    back to the raw agent response when no solution could be extracted, so a CSV/log consumer can
    still see what the agent said even when extraction failed). Kept alongside `GradeResult` rather
    than folded into it so `GradeResult` stays the small, stable type graders/tests already build
    on."""
    grader = _GRADERS.get(task.source)
    if grader is None:
        raise ValueError(f"no grader for source {task.source!r}")
    # Grading (Docker-based sources especially) and the model call itself can need very different
    # timeouts — see CalibrationConfig.task_timeout_overrides. run_pi always uses the plain,
    # unoverridden value so a slow grader can't also give a hung LLM call the same long leash.
    grading_timeout = calibration_config.grading_timeout_for(task.source)

    if model.runner == "reference":
        return grade_reference(task, grading_timeout), _expected_solution(task)

    if model.runner == "null":
        return grade_null(task, grading_timeout), ""

    if model.runner == "pi":
        run_result = runner_mod.run_pi(task, model, timeout_seconds=calibration_config.task_timeout_seconds)
        if run_result.context_unavailable:
            # Repo clone/checkout failed before pi was ever invoked — an infra problem, not the
            # model's fault, and per the spec's fairness requirement a task that can't be set up
            # consistently for every model shouldn't be scored for any of them.
            result = GradeResult(outcome="error_harness", detail=run_result.detail)
        elif run_result.rate_limited:
            # An infra/quota rejection, not the model failing to answer — excluded rather than
            # counted as a wrong answer, same reasoning as error_timeout/error_missing_dep.
            result = GradeResult(outcome="error_harness", detail=run_result.detail)
        elif run_result.solution is None:
            result = GradeResult(outcome="error_no_solution", detail=run_result.detail)
        else:
            result = grader(task, run_result.solution, timeout_seconds=grading_timeout)

        # Multi-line despite the "one line per call" logging convention (logging_config.py) —
        # deliberately: this is the one place meant for reading a code/diff block back, and
        # collapsing it to one line would make exactly the thing it's for unreadable. DEBUG-only
        # so it never appears in a normal run.
        logger.debug(
            f"{task.task_id} ({model.model_id}) -> {result.outcome}\n"
            f"  expected: {_preview(_expected_solution(task))}\n"
            f"  provided: {_preview(run_result.solution) if run_result.solution is not None else '(no solution extracted)'}\n"
            f"  detail: {result.detail}"
        )
        returned = run_result.solution if run_result.solution is not None else run_result.raw_response
        return result, returned

    raise ValueError(f"unknown runner {model.runner!r} for model {model.model_id}")


@dataclasses.dataclass(frozen=True)
class TaskRunRecord:
    """`run_and_grade`'s (GradeResult, solution) pair plus timing — one row of "what actually
    happened" for a single (task, model) call, kept separate from `GradeResult` for the same reason
    `run_and_grade` returns a tuple instead of widening it (see that function's docstring)."""
    result: GradeResult
    solution: str | None
    duration_ms: int


def run_and_log(task: Task, model: ModelConfig, calibration_config: CalibrationConfig, index: int, total: int) -> TaskRunRecord:
    """`run_and_grade` plus the timing/progress log line — shared by calibrate_model's per-model
    loop below and evaluate.py's per-model x per-task holdout loop, which otherwise duplicate this
    exact started/duration/excluded-vs-not branch."""
    logger.info(f"[{index}/{total}] {model.model_id} task {task.task_id} ({task.source}) starting...")
    started = time.monotonic()
    result, solution = run_and_grade(task, model, calibration_config)
    duration_ms = round((time.monotonic() - started) * 1000)
    if result.outcome in grading_base.EXCLUDED_OUTCOMES:
        logger.warning(f"[{index}/{total}] {model.model_id} task {task.task_id} excluded: {result.outcome} ({duration_ms}ms)")
    else:
        logger.info(f"[{index}/{total}] {model.model_id} task {task.task_id} -> {result.outcome} ({duration_ms}ms)")
    return TaskRunRecord(result=result, solution=solution, duration_ms=duration_ms)


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


def image_affinity_key(task: Task) -> tuple[str, str]:
    """Sort key that groups tasks sharing a Docker image next to each other, so consecutive grading
    calls hit a warm image instead of re-pulling. Matters most for swe-smith, whose `image_name` is
    per bug-injected repo (~128 distinct images across ~59K instances) rather than per instance —
    swe-gym and multi-swe-rl build one image PER instance, so for them this only groups by repo,
    which is still the right tiebreak for their shared git clones (repo_context.py)."""
    return (task.source, str(task.row.get("image_name") or task.row.get("repo") or task.task_id))


@dataclasses.dataclass(frozen=True)
class CalibrationDetailRow:
    """One (task, model) call, flattened for `calibration-details.csv` — everything `run_and_log`
    knows about a single call, alongside the task identity it was made for."""
    task_id: str
    source: str
    cluster_id: int
    split: str
    model_id: str
    provider: str
    outcome: str
    detail: str
    solution: str
    duration_ms: int


# Truncation length for the `solution` column — long enough to see the shape of a real answer
# (a full diff/patch can be huge), short enough that a multi-MB swe-gym patch doesn't blow up the
# CSV. Same idea as _LOG_PREVIEW_CHARS above, just a separate constant since a CSV meant to be
# opened in a spreadsheet tool can afford to keep more than a debug log line.
_CSV_SOLUTION_MAX_CHARS = 2000


def _csv_preview(text: str) -> str:
    if len(text) <= _CSV_SOLUTION_MAX_CHARS:
        return text
    return text[:_CSV_SOLUTION_MAX_CHARS] + f" ...[{len(text) - _CSV_SOLUTION_MAX_CHARS} more chars]"


def write_calibration_details_csv(rows: list[CalibrationDetailRow], path: Path | None = None) -> Path:
    target = path or (ARTIFACTS_DIR / "calibration-details.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["task_id", "source", "cluster_id", "split", "model_id", "provider", "outcome", "detail", "solution", "duration_ms"]
        )
        for row in rows:
            writer.writerow(
                [row.task_id, row.source, row.cluster_id, row.split, row.model_id, row.provider,
                 row.outcome, row.detail, row.solution, row.duration_ms]
            )
    return target


def calibrate_models(
    models: list[ModelConfig], selected_tasks: list[SelectedTask], calibration_config: CalibrationConfig,
) -> tuple[list[ModelCalibrationResult], list[CalibrationDetailRow]]:
    """Runs every (task, model) pair TASKS-OUTER / MODELS-INNER, then aggregates per model.

    The loop order is the optimization, and it is worth roughly a factor of `len(models)` on the
    dominant cost of a real run. swe-gym and multi-swe-rl build one multi-GB Docker image PER
    INSTANCE (measured 2.9-6.5GB each), so a models-outer loop walks every image once per model
    while `dockerexec`'s bounded LRU cache (15 images) is far too small to bridge the gap — at a
    few hundred selected tasks the hit rate collapses to ~0 and every image is re-pulled for every
    model. Grading each task against all models while its image is still hot pulls each image
    exactly once instead: for ~250 Docker-backed tasks and 5 configured models that's ~250 pulls
    (~875GB) rather than ~1,250 (~4.4TB), which at realistic bandwidth is the difference between
    hours and a day of pure download. It also collapses peak resident disk, since a task's image is
    finished with the moment its inner loop ends, and it fixes the same thrash for
    `repo_context.py`'s bare-clone cache.

    Purely a reordering: outcomes are per (model, task) pair and every statistic is computed after
    the fact by `_aggregate_outcomes`, so results are identical to the previous order. Task
    selection (and therefore the RNG) has already happened in `select_tasks` by this point, so
    determinism is unaffected too."""
    calibration_only = sorted(
        (st for st in selected_tasks if st.split == "calibration"),
        key=lambda st: image_affinity_key(st.task),
    )
    total = len(calibration_only) * len(models)
    logger.info(
        f"calibration started: {len(calibration_only)} tasks x {len(models)} models = {total} calls "
        "(tasks-outer, models-inner)"
    )
    outcomes_by_model: dict[str, list[tuple[SelectedTask, GradeResult]]] = {m.model_id: [] for m in models}
    detail_rows: list[CalibrationDetailRow] = []
    index = 0
    for st in calibration_only:
        for model in models:
            index += 1
            record = run_and_log(st.task, model, calibration_config, index, total)
            outcomes_by_model[model.model_id].append((st, record.result))
            detail_rows.append(CalibrationDetailRow(
                task_id=st.task.task_id,
                source=st.task.source,
                cluster_id=st.cluster_id,
                split=st.split,
                model_id=model.model_id,
                provider=model.provider,
                outcome=record.result.outcome,
                detail=record.result.detail,
                solution=_csv_preview(record.solution or ""),
                duration_ms=record.duration_ms,
            ))

    aggregated = [_aggregate_outcomes(m, outcomes_by_model[m.model_id], calibration_config) for m in models]
    return aggregated, detail_rows


def _aggregate_outcomes(
    model: ModelConfig,
    per_task_outcomes: list[tuple[SelectedTask, GradeResult]],
    calibration_config: CalibrationConfig,
) -> ModelCalibrationResult:
    """Pure aggregation — no grading calls, so it's independent of the order they were made in."""
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
