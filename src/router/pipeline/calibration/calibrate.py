from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import random
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ...common.artifacts import ARTIFACTS_DIR
from ...common.assign import ClusterMap
from ...common.config import CalibrationConfig, ModelConfig
from . import ground_truth_registry
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

# Patch-based sources grade the gold/empty patch through a dedicated entry point instead of the
# generic `grader(task, solution)` shape; single source of truth for the reference/null branches.
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


# Debug-only preview length for expected-vs-provided logging (see run_and_grade's pi branch) —
# long enough to see the shape of an answer, short enough not to flood the log with a huge patch.
_LOG_PREVIEW_CHARS = 1000


def _preview(text: str) -> str:
    text = text.strip()
    if len(text) <= _LOG_PREVIEW_CHARS:
        return text
    return text[:_LOG_PREVIEW_CHARS] + f" ...[{len(text) - _LOG_PREVIEW_CHARS} more chars]"


def _expected_solution(task: Task) -> str:
    # bigcodebench/ds1000 carry the gold answer in reference_solution; patch-based sources leave it
    # empty (see tasks.py) and use task.row["patch"] instead.
    return task.reference_solution or str(task.row.get("patch", ""))


@dataclasses.dataclass(frozen=True)
class SelectedTask:
    task: Task
    cluster_id: int
    split: str  # always "calibration" for now, kept for schema/consumer compatibility — see
    # docs/engineering-notes.md, "Holdout split removed"


# Fixed source->category mapping for spec §5.1's "aim for the following distribution" — a property
# of what each dataset IS, not a tunable; config only controls the target ratios (category_mix).
CATEGORY_SOURCES: dict[str, list[str]] = {
    "repo_python": ["swe-smith", "swe-gym"],
    "multilingual": ["multi-swe-rl"],
    "standalone": ["bigcodebench", "ds1000"],
}
_SOURCE_CATEGORY: dict[str, str] = {s: cat for cat, sources in CATEGORY_SOURCES.items() for s in sources}

# "aim for" (spec's own word), not "require exactly" — but a sum off by more than this is almost
# certainly a typo, so it's rejected rather than silently normalized.
_CATEGORY_MIX_SUM_TOLERANCE = 0.01


def _validate_category_mix(category_mix: dict[str, float]) -> None:
    unknown = sorted(set(category_mix) - set(CATEGORY_SOURCES))
    if unknown:
        raise ValueError(
            f"config/calibration.yaml's category_mix has unknown categor{'y' if len(unknown) == 1 else 'ies'} "
            f"{unknown} — known categories are {sorted(CATEGORY_SOURCES)}."
        )
    # A category missing here isn't "excluded on purpose" — that's what gradeable_sources is for.
    # Left unchecked, an omitted category's tasks are silently never selected (chosen/backfill both
    # only ever iterate over category_mix's own keys), with no warning and a config that still
    # "validly" sums to ~1.0.
    missing = sorted(set(CATEGORY_SOURCES) - set(category_mix))
    if missing:
        raise ValueError(
            f"config/calibration.yaml's category_mix is missing categor{'y' if len(missing) == 1 else 'ies'} "
            f"{missing} — every known category must have an explicit ratio (use 0.0 to deliberately "
            "exclude one), or its tasks are silently never selected."
        )
    total = sum(category_mix.values())
    if abs(total - 1.0) > _CATEGORY_MIX_SUM_TOLERANCE:
        raise ValueError(
            f"config/calibration.yaml's category_mix values sum to {total}, not ~1.0 "
            f"({category_mix!r}) — fix the ratios before calibrating."
        )


def _take_up_to(candidates: list[Task], n: int, verify: Callable[[Task], bool] | None) -> tuple[list[Task], list[Task]]:
    """Returns (taken, remaining) — `taken` is the first `n` USABLE candidates (all of them, in
    order, if `verify` is `None`; the first `n` for which `verify(task)` is true otherwise), and
    `remaining` is whatever in `candidates` was never examined, so it can still be offered as
    backfill surplus to a different category without re-examining (or re-verifying) anything.

    A candidate `verify` rejects is dropped for good — never re-offered as surplus — since
    rejecting it just recorded (via `verify`'s own side effect, see `calibrate_verified_tasks`)
    that it's not usable at all, not merely unlucky for this particular category/quota."""
    if verify is None:
        return candidates[:n], candidates[n:]
    taken: list[Task] = []
    i = 0
    while i < len(candidates) and len(taken) < n:
        if verify(candidates[i]):
            taken.append(candidates[i])
        i += 1
    return taken, candidates[i:]


def _select_with_category_mix(
    tasks_in_cluster: list[Task], budget: int, category_mix: dict[str, float], rng: random.Random,
    verify: Callable[[Task], bool] | None = None,
) -> tuple[list[Task], dict[str, int]]:
    """Splits `budget` across categories by `category_mix`'s ratios, then backfills any category's
    shortfall from other categories' surplus in the same cluster (a cluster naturally dominated by
    one source may have nothing to give another). Returns (chosen, shortfalls) — shortfalls maps
    category -> how much of its quota went unfilled, for logging; `chosen` can still total less
    than `budget` if every category in this cluster is already exhausted.

    `verify` (optional) turns "take the first `quota` candidates" into "take the first `quota`
    USABLE candidates, skipping and permanently discarding any it rejects, pulling further into the
    shuffled pool as needed" — see `_take_up_to`. `None` (the default, used by `select_tasks`)
    preserves the exact original behavior: unconditional, no extra work. Passing a real predicate
    (see `calibrate_verified_tasks`) is what turns this from "select, then separately filter and
    hope enough survive" into "select exactly `budget` already-known-good tasks per category,
    verifying lazily — only as many candidates as actually needed, never the whole pool"."""
    by_category: dict[str, list[Task]] = {}
    for task in tasks_in_cluster:
        category = _SOURCE_CATEGORY.get(task.source)
        if category is None:
            raise ValueError(
                f"gradeable source {task.source!r} has no entry in CATEGORY_SOURCES — add it there "
                "before using category_mix with this source enabled."
            )
        by_category.setdefault(category, []).append(task)
    for cat_tasks in by_category.values():
        cat_tasks.sort(key=lambda t: t.task_id)  # order-independent before shuffling
        rng.shuffle(cat_tasks)

    categories = list(category_mix)  # config's own order — the last one absorbs the rounding
    # remainder below so quotas always sum to exactly `budget`, never budget +/- 1 from independent
    # per-category rounding.
    quotas: dict[str, int] = {}
    allocated = 0
    for i, category in enumerate(categories):
        if i == len(categories) - 1:
            quotas[category] = max(budget - allocated, 0)
        else:
            quotas[category] = round(budget * category_mix[category])
            allocated += quotas[category]

    chosen: list[Task] = []
    shortfalls: dict[str, int] = {}
    leftover_by_category: dict[str, list[Task]] = {}
    remaining_budget = 0
    for category in categories:
        available = by_category.get(category, [])
        quota = quotas[category]
        take, leftover = _take_up_to(available, quota, verify)
        chosen.extend(take)
        leftover_by_category[category] = leftover
        shortfall = quota - len(take)
        if shortfall > 0:
            shortfalls[category] = shortfall
            remaining_budget += shortfall

    if remaining_budget > 0:
        for category in categories:
            if remaining_budget <= 0:
                break
            surplus = leftover_by_category.get(category, [])  # not already taken above
            take_extra, leftover = _take_up_to(surplus, remaining_budget, verify)
            chosen.extend(take_extra)
            leftover_by_category[category] = leftover
            remaining_budget -= len(take_extra)

    return chosen, shortfalls


def _group_gradeable_tasks_by_cluster(
    calibration_config: CalibrationConfig, cluster_map: ClusterMap, task_cluster_map: dict,
) -> tuple[dict[int, list[Task]], int]:
    """Shared setup for `select_tasks`/`select_verified_tasks`: validates `task_cluster_map` was
    built against the SAME `cluster_map` (a stale/mismatched mapping would silently make cluster
    ids mean different things than the centroids `cluster_map` carries, which must fail loudly
    rather than produce a quietly-wrong selection), then loads every gradeable source's full row
    data and groups it by cluster. Returns (by_cluster, k)."""
    if task_cluster_map["cluster_map_id"] != cluster_map.artifact_id:
        raise ValueError(
            f"task-cluster-map.json was built against cluster map {task_cluster_map['cluster_map_id']!r}, "
            f"but the current cluster-map.json is {cluster_map.artifact_id!r} — re-run `build-artifact` "
            "to regenerate a matching task-cluster-map.json."
        )
    k = cluster_map.centroids.shape[0]
    gradeable_sources = set(calibration_config.gradeable_sources)

    # Full row data per gradeable source — task_cluster_map only carries task_id/source/cluster_id.
    gradeable_by_id: dict[str, Task] = {}
    for source in calibration_config.gradeable_sources:
        for task in load_gradeable_tasks(source):
            gradeable_by_id[task.task_id] = task

    by_cluster: dict[int, list[Task]] = {}
    unmatched = 0
    for entry in task_cluster_map["tasks"]:
        if entry["source"] not in gradeable_sources:
            continue
        task = gradeable_by_id.get(entry["task_id"])
        if task is None:
            # Expected, not an error: a row the corpus includes but calibration's loader excludes
            # (e.g. a DS-1000 Matplotlib row) has a cluster label but was never gradeable.
            unmatched += 1
            continue
        by_cluster.setdefault(entry["cluster_id"], []).append(task)
    if unmatched:
        logger.info(f"{unmatched} task-cluster-map entries had no matching gradeable task, skipped")

    empty_clusters = [c for c in range(k) if c not in by_cluster]
    if empty_clusters:
        logger.warning(f"{len(empty_clusters)} of {k} clusters have zero gradeable tasks: {empty_clusters}")

    return by_cluster, k


def _log_category_mix_achieved(
    selected: list[SelectedTask], category_mix: dict[str, float], num_clusters: int, clusters_with_shortfall: int,
) -> None:
    achieved = {cat: sum(1 for s in selected if _SOURCE_CATEGORY[s.task.source] == cat) for cat in category_mix}
    total = len(selected) or 1
    achieved_pct = {cat: round(100 * n / total, 1) for cat, n in achieved.items()}
    logger.info(
        f"selected {len(selected)} tasks across {num_clusters} clusters — category mix achieved: "
        f"{achieved_pct} (target: {category_mix}, {clusters_with_shortfall} cluster(s) had a quota shortfall)"
    )


def select_tasks(
    calibration_config: CalibrationConfig,
    cluster_map: ClusterMap,
    task_cluster_map: dict,
) -> list[SelectedTask]:
    """Stratified-by-cluster task selection, driven by the full task->cluster mapping computed once
    at `build-artifact` time (`clustering/task_cluster_map.py`) rather than a per-run random
    pre-filter — see docs/engineering-notes.md, "Task-cluster mapping replaced a per-run random
    pre-filter" for why. When `calibration_config.category_mix` is set, each cluster's
    `tasks_per_cluster` budget is further split by category (see `_select_with_category_mix`) per
    spec §5.1; left empty (the default), behavior is a flat shuffle-and-cap per cluster.

    Fast and pure — no grading calls, no Docker. Ground-truth validity of what comes out is only
    discovered later, mid-calibration (`_ground_truth_invalid`'s in-run skip). For a selection
    that's already pre-verified, see `select_verified_tasks` — a separate, explicit, slower entry
    point; this function's behavior and callers (routine `calibrate` runs loading a pin) are
    unchanged by that existing."""
    if calibration_config.category_mix:
        _validate_category_mix(calibration_config.category_mix)
    rng = random.Random(calibration_config.seed)
    by_cluster, _k = _group_gradeable_tasks_by_cluster(calibration_config, cluster_map, task_cluster_map)

    selected: list[SelectedTask] = []
    clusters_with_shortfall = 0
    for cluster_id in sorted(by_cluster):  # explicit, not dict-insertion-order reliance
        tasks_in_cluster = by_cluster[cluster_id]
        if calibration_config.category_mix:
            chosen, shortfalls = _select_with_category_mix(
                tasks_in_cluster, calibration_config.tasks_per_cluster, calibration_config.category_mix, rng,
            )
            if shortfalls:
                clusters_with_shortfall += 1
                logger.info(f"cluster {cluster_id}: category quota shortfall (backfilled where possible): {shortfalls}")
        else:
            tasks_sorted = sorted(tasks_in_cluster, key=lambda t: t.task_id)  # order-independent before shuffling
            rng.shuffle(tasks_sorted)
            chosen = tasks_sorted[: calibration_config.tasks_per_cluster]

        for task in chosen:
            selected.append(SelectedTask(task=task, cluster_id=cluster_id, split="calibration"))

    if calibration_config.category_mix:
        _log_category_mix_achieved(selected, calibration_config.category_mix, len(by_cluster), clusters_with_shortfall)
    else:
        logger.info(f"selected {len(selected)} tasks across {len(by_cluster)} clusters")
    return selected


def select_verified_tasks(
    calibration_config: CalibrationConfig,
    cluster_map: ClusterMap,
    task_cluster_map: dict,
    registry: dict,
    registry_path: Path = ground_truth_registry.REGISTRY_PATH,
) -> list[SelectedTask]:
    """Like `select_tasks`, but every candidate is ground-truth-verified (via `verify_ground_truth`
    — registry-cached, live-graded only on a cache miss) before being counted toward its cluster's
    category quota — see `_select_with_category_mix`'s `verify` parameter. A task the verification
    rejects is skipped and never re-offered, including as backfill surplus for a different
    category; `select_tasks` itself is untouched by this — routine `calibrate` runs stay fast.

    Requires `calibration_config.category_mix` (a flat, uncategorized verified selection isn't a
    real use case this supports yet — every caller so far wants the category-mix targeting).
    `tasks_per_cluster` is the same per-cluster budget `select_tasks` uses; asking for a bigger
    overall pin means raising it (see `cli.py`'s `select-verified-tasks` command), the same lever
    `select_tasks` already exposes.

    `registry` is mutated in place AND written to `registry_path` after every new verification (see
    `verify_ground_truth`) — a run touching hundreds of tasks over hours must not risk losing
    everything to a crash near the end; callers still get the final in-memory `registry` back for
    logging, but don't need to write it again themselves for correctness."""
    if not calibration_config.category_mix:
        raise ValueError("select_verified_tasks requires calibration_config.category_mix to be set")
    _validate_category_mix(calibration_config.category_mix)
    rng = random.Random(calibration_config.seed)
    by_cluster, _k = _group_gradeable_tasks_by_cluster(calibration_config, cluster_map, task_cluster_map)

    def verify(task: Task) -> bool:
        return verify_ground_truth(task, calibration_config, registry, registry_path) == "valid"

    selected: list[SelectedTask] = []
    clusters_with_shortfall = 0
    for cluster_id in sorted(by_cluster):
        tasks_in_cluster = by_cluster[cluster_id]
        chosen, shortfalls = _select_with_category_mix(
            tasks_in_cluster, calibration_config.tasks_per_cluster, calibration_config.category_mix, rng,
            verify=verify,
        )
        if shortfalls:
            clusters_with_shortfall += 1
            logger.info(
                f"cluster {cluster_id}: category quota shortfall after ground-truth verification "
                f"(backfilled where possible): {shortfalls}"
            )
        for task in chosen:
            selected.append(SelectedTask(task=task, cluster_id=cluster_id, split="calibration"))

    _log_category_mix_achieved(selected, calibration_config.category_mix, len(by_cluster), clusters_with_shortfall)
    return selected


def compute_task_selection_digest(selected: list[SelectedTask]) -> str:
    """SHA-256 over sorted (task_id, prompt) pairs — same construction as
    `clustering/cluster_map.py::compute_corpus_digest`, applied to the selected tasks' prompt text.
    `load_task_selection` recomputes this and refuses to load on a mismatch."""
    hasher = hashlib.sha256()
    for selected_task in sorted(selected, key=lambda s: s.task.task_id):
        hasher.update(selected_task.task.task_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(selected_task.task.prompt.encode("utf-8"))
        hasher.update(b"\0")
    return f"sha256:{hasher.hexdigest()}"


def selected_tasks_to_dict(selected: list[SelectedTask], k: int) -> dict:
    """Only `task_id`/`source`/`cluster_id`/`split` are kept — `load_task_selection` re-derives the
    heavy `row` data fresh at load time. `k` is used only to report which clusters got zero tasks,
    the permanent record of what `select_tasks` already logs."""
    represented = {s.cluster_id for s in selected}
    empty_clusters = [c for c in range(k) if c not in represented]
    now = datetime.now(UTC).isoformat()
    digest = compute_task_selection_digest(selected)
    return {
        "schema_version": 1,
        "artifact_id": f"taskselection-{now[:10]}-{digest.split(':')[1][:12]}",
        "created_at": now,
        "content_digest": digest,
        "empty_clusters": empty_clusters,
        "tasks": [
            {"task_id": s.task.task_id, "source": s.task.source, "cluster_id": s.cluster_id, "split": s.split}
            for s in selected
        ],
    }


def write_task_selection(artifact: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"wrote task selection to {path} ({len(artifact['tasks'])} tasks)")
    return path


def load_task_selection(path: Path) -> list[SelectedTask]:
    """The reverse of `selected_tasks_to_dict` — reconstructs `SelectedTask`s with fresh `row` data.
    Raises (never silently drops) on a pinned task_id no longer in its source's gradeable pool, or
    a content-digest mismatch — an incremental run (cli.py's `--model`) must grade against the
    EXACT set an existing model-profiles.json was built from."""
    data = json.loads(path.read_text(encoding="utf-8"))

    sources = {entry["source"] for entry in data["tasks"]}
    gradeable_by_id: dict[str, Task] = {}
    for source in sources:
        for task in load_gradeable_tasks(source):
            gradeable_by_id[task.task_id] = task

    selected: list[SelectedTask] = []
    missing: list[str] = []
    for entry in data["tasks"]:
        task = gradeable_by_id.get(entry["task_id"])
        if task is None:
            missing.append(entry["task_id"])
            continue
        selected.append(SelectedTask(task=task, cluster_id=entry["cluster_id"], split=entry["split"]))
    if missing:
        raise ValueError(
            f"{len(missing)} pinned task_id(s) from {path} are no longer in their source's gradeable "
            f"pool (first few: {missing[:5]}) — the underlying dataset has likely changed since this "
            "pin was written."
        )

    actual_digest = compute_task_selection_digest(selected)
    if actual_digest != data["content_digest"]:
        raise ValueError(
            f"{path}'s content digest doesn't match its pinned tasks' current prompt text — the "
            "underlying dataset has changed since this pin was written. Re-select tasks rather than "
            "grade against content this pin didn't sign off on."
        )
    return selected


def run_and_grade(
    task: Task, model: ModelConfig, calibration_config: CalibrationConfig,
) -> tuple[GradeResult, str | None, runner_mod.TokenUsage | None]:
    """Returns (GradeResult, solution, usage) — `solution` is what the candidate actually produced
    (falling back to the raw agent response when extraction failed), `usage` is Pi's reported
    token/cost usage, or None for `reference`/`null` or an unparseable `pi` call. Kept as a tuple
    rather than folded into `GradeResult` so that type stays the small, stable one graders/tests
    already build on."""
    grader = _GRADERS.get(task.source)
    if grader is None:
        raise ValueError(f"no grader for source {task.source!r}")
    # Grading (Docker-based sources especially) and the model call itself can need very different
    # timeouts — see CalibrationConfig.task_timeout_overrides. run_pi always uses the plain,
    # unoverridden value so a slow grader can't also give a hung LLM call the same long leash.
    grading_timeout = calibration_config.grading_timeout_for(task.source)

    if model.runner == "reference":
        return grade_reference(task, grading_timeout), _expected_solution(task), None

    if model.runner == "null":
        return grade_null(task, grading_timeout), "", None

    if model.runner == "pi":
        run_result = runner_mod.run_pi(task, model, timeout_seconds=calibration_config.task_timeout_seconds)
        # All of these are infra/harness problems, not the model failing to answer, so they're
        # excluded (error_harness) rather than counted as a wrong answer.
        if run_result.context_unavailable:
            # Repo clone/checkout failed before pi was ever invoked.
            result = GradeResult(outcome="error_harness", detail=run_result.detail)
        elif run_result.harness_error:
            # Pi exits 0 even on a provider-level error — see docs/engineering-notes.md,
            # "Pi exits 0 on a provider-level error".
            result = GradeResult(outcome="error_harness", detail=run_result.detail)
        elif run_result.timed_out:
            # Our own subprocess timeout, distinct from a grader's own error_timeout (the test run).
            result = GradeResult(outcome="error_harness", detail=run_result.detail)
        elif run_result.solution is None:
            result = GradeResult(outcome="error_no_solution", detail=run_result.detail)
        else:
            result = grader(task, run_result.solution, timeout_seconds=grading_timeout)

        # Deliberately multi-line (unlike the rest of this module's one-line-per-call logging) so a
        # code/diff block stays readable; DEBUG-only so it never appears in a normal run.
        logger.debug(
            f"{task.task_id} ({model.model_id}) -> {result.outcome}\n"
            f"  expected: {_preview(_expected_solution(task))}\n"
            f"  provided: {_preview(run_result.solution) if run_result.solution is not None else '(no solution extracted)'}\n"
            f"  detail: {result.detail}"
        )
        returned = run_result.solution if run_result.solution is not None else run_result.raw_response
        return result, returned, run_result.usage

    raise ValueError(f"unknown runner {model.runner!r} for model {model.model_id}")


@dataclasses.dataclass(frozen=True)
class TaskRunRecord:
    """`run_and_grade`'s (GradeResult, solution, usage) triple plus timing — one row of "what
    actually happened" for a single (task, model) call."""
    result: GradeResult
    solution: str | None
    duration_ms: int
    usage: runner_mod.TokenUsage | None = None


def run_and_log(task: Task, model: ModelConfig, calibration_config: CalibrationConfig, index: int, total: int) -> TaskRunRecord:
    """`run_and_grade` plus the timing/progress log line — shared by calibrate_models' per-model
    loop and evaluate.py's per-model x per-task loop."""
    logger.info(f"[{index}/{total}] {model.model_id} task {task.task_id} ({task.source}) starting...")
    started = time.monotonic()
    result, solution, usage = run_and_grade(task, model, calibration_config)
    duration_ms = round((time.monotonic() - started) * 1000)
    if result.outcome in grading_base.EXCLUDED_OUTCOMES:
        logger.warning(f"[{index}/{total}] {model.model_id} task {task.task_id} excluded: {result.outcome} ({duration_ms}ms)")
    else:
        logger.info(f"[{index}/{total}] {model.model_id} task {task.task_id} -> {result.outcome} ({duration_ms}ms)")
    return TaskRunRecord(result=result, solution=solution, duration_ms=duration_ms, usage=usage)


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
    calls hit a warm image instead of re-pulling. See docs/engineering-notes.md, "Image affinity"."""
    return (task.source, str(task.row.get("image_name") or task.row.get("repo") or task.task_id))


@dataclasses.dataclass(frozen=True)
class CalibrationDetailRow:
    """One (task, model) call, flattened for `calibration-details-<run timestamp>.csv`. No per-row
    timestamp — the run's start time lives in the CSV's filename instead (see calibrate_models'
    `details_csv_path`), which is what actually distinguishes one run's file from another's."""
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
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0  # Pi's own reported cost — 0 for controls or unreadable usage, not "free".
    turns: int = 0  # Agent turns Pi took to reach a final answer — 0 for the same cases as above.


_CSV_FIELDNAMES = [
    "task_id", "source", "cluster_id", "split", "model_id", "provider",
    "outcome", "detail", "solution", "duration_ms", "input_tokens", "output_tokens", "cost_usd", "turns",
]


def _csv_row_values(row: CalibrationDetailRow) -> list:
    return [
        row.task_id, row.source, row.cluster_id, row.split, row.model_id,
        row.provider, row.outcome, row.detail, row.solution, row.duration_ms,
        row.input_tokens, row.output_tokens, row.cost_usd, row.turns,
    ]


# Truncation length for the `solution` column, so a multi-MB patch doesn't blow up the CSV. A
# separate constant from _LOG_PREVIEW_CHARS since a spreadsheet-opened CSV can afford more.
_CSV_SOLUTION_MAX_CHARS = 2000


def _csv_preview(text: str) -> str:
    if len(text) <= _CSV_SOLUTION_MAX_CHARS:
        return text
    return text[:_CSV_SOLUTION_MAX_CHARS] + f" ...[{len(text) - _CSV_SOLUTION_MAX_CHARS} more chars]"


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def _solutions_dir(details_csv_path: Path) -> Path:
    # Keyed by the CSV's own filename stem (embeds the run timestamp) so each run's persisted
    # solutions land in their own directory.
    return details_csv_path.parent / "solutions" / details_csv_path.stem


# `run_and_grade`'s pi branch falls back to `run_result.raw_response` (Pi's raw `--mode json`
# stdout, e.g. `{"type":"session",...}` one JSON object per line) whenever no solution could be
# extracted — see its own docstring. That raw stdout is never mistakable for a real diff/code
# solution, so it's used here purely to pick a filename extension that doesn't claim to be a diff
# when it isn't. Never affects grading, which already treats this text identically either way.
_RAW_PI_EVENT_STREAM_PREFIX = '{"type":'


def _persist_full_solution(solutions_dir: Path, task_id: str, model_id: str, solution: str) -> None:
    """Only called when `_csv_preview` truncated `solution` — see docs/engineering-notes.md,
    "Large solution truncation". Uses a `.raw.txt` extension instead of `.diff` when `solution` is
    actually Pi's raw, unparsed event-stream stdout rather than a real diff/code solution, so
    browsing this directory doesn't show a `.diff` file that isn't one."""
    solutions_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = _UNSAFE_FILENAME_CHARS.sub("_", task_id)
    safe_model_id = _UNSAFE_FILENAME_CHARS.sub("_", model_id)
    ext = "raw.txt" if solution.lstrip().startswith(_RAW_PI_EVENT_STREAM_PREFIX) else "diff"
    (solutions_dir / f"{safe_task_id}__{safe_model_id}.{ext}").write_text(solution, encoding="utf-8")


class _CalibrationDetailsWriter:
    """Writes calibration-details.csv incrementally — one row per call, flushed immediately — so an
    interrupted run still leaves a usable CSV and a still-running one can be inspected."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_CSV_FIELDNAMES)
        self._file.flush()

    def write(self, row: CalibrationDetailRow) -> None:
        self._writer.writerow(_csv_row_values(row))
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _ground_truth_invalid(reference_outcome: str | None, null_outcome: str | None) -> bool:
    """True once BOTH controls have actually run for this task and either says the ground truth is
    bad: the gold solution didn't pass (`reference_outcome != "pass"`) or an empty solution
    incorrectly did (`null_outcome == "pass"`). Either argument still `None` (a control wasn't in
    this run's roster at all — e.g. a `--model`-scoped incremental run drops both) means "can't
    tell," not "invalid" — real models still run in that case, same as before this existed."""
    if reference_outcome is None or null_outcome is None:
        return False
    return reference_outcome != "pass" or null_outcome == "pass"


def verify_ground_truth(
    task: Task, calibration_config: CalibrationConfig, registry: dict,
    registry_path: Path = ground_truth_registry.REGISTRY_PATH,
) -> str:
    """Returns `"valid"` or `"invalid"` for `task`, consulting `registry` first (a task already
    verified with a matching prompt digest costs zero grading calls) and running the real
    `grade_reference`/`grade_null` controls — upserting the result back into `registry` — only on a
    cache miss.

    Writes `registry` to `registry_path` immediately after every cache-miss upsert (never on a
    cache hit — nothing changed) — a real, Docker-based verification run can take hours, and this
    environment has a genuine history of crashing mid-run (credit exhaustion, encoding bugs); a
    write here costs milliseconds against a grading call that costs seconds to minutes, so batching
    writes to save on I/O would be a bad trade. Same discipline as `_CalibrationDetailsWriter`'s
    per-row flush, for the same reason: confirmed real — a `select-verified-tasks` run was killed
    after ~40 minutes of grading with the registry file's mtime never having moved once.

    This is the only intended way `registry` gets new entries — it grows accretively from whatever
    selection/calibration work already needs a task's ground truth, never from a separate,
    dedicated verification sweep. See `ground_truth_registry`'s module docstring."""
    digest = ground_truth_registry.compute_task_digest(task)
    cached = ground_truth_registry.lookup(registry, task.task_id, digest)
    if cached is not None:
        return cached["verdict"]

    grading_timeout = calibration_config.grading_timeout_for(task.source)
    reference_result = grade_reference(task, grading_timeout)
    null_result = grade_null(task, grading_timeout)
    verdict = "invalid" if _ground_truth_invalid(reference_result.outcome, null_result.outcome) else "valid"

    ground_truth_registry.upsert(
        registry,
        task_id=task.task_id,
        source=task.source,
        prompt_digest=digest,
        reference_outcome=reference_result.outcome,
        reference_detail=reference_result.detail,
        null_outcome=null_result.outcome,
        null_detail=null_result.detail,
        verdict=verdict,
    )
    ground_truth_registry.write_registry(registry, registry_path)
    return verdict


def backfill_registry_from_details_csv(csv_path: Path, registry: dict) -> int:
    """Recovers ground-truth verifications already paid for in a prior calibration run's details
    CSV into `registry`, instead of re-grading them from scratch via `verify_ground_truth` —
    idempotent and safe to run against any old CSV at any time: only adds an entry for a task that
    both (a) has both `reference-oracle` and `null-baseline` rows in this CSV, and (b) isn't
    ALREADY registered with a matching prompt digest — an existing, still-fresh entry (e.g. from a
    live verification in a later run) is never overwritten by older CSV data.

    The CSV itself doesn't carry prompt text, so a task's CURRENT prompt (needed to compute the
    digest that makes a cache hit trustworthy) is loaded fresh per source, same mechanism
    `select_tasks`/`load_task_selection` already use — a task the CSV mentions that's no longer in
    its source's gradeable pool (dataset drift) is skipped, not an error.

    Returns the number of NEW entries added."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    reference_rows = {r["task_id"]: r for r in rows if r["model_id"] == "reference-oracle"}
    null_rows = {r["task_id"]: r for r in rows if r["model_id"] == "null-baseline"}
    both = sorted(set(reference_rows) & set(null_rows))

    sources = {reference_rows[task_id]["source"] for task_id in both}
    task_by_id: dict[str, Task] = {}
    for source in sources:
        for task in load_gradeable_tasks(source):
            task_by_id[task.task_id] = task

    added = 0
    for task_id in both:
        task = task_by_id.get(task_id)
        if task is None:
            continue
        digest = ground_truth_registry.compute_task_digest(task)
        if ground_truth_registry.lookup(registry, task_id, digest) is not None:
            continue
        reference_row = reference_rows[task_id]
        null_row = null_rows[task_id]
        verdict = "invalid" if _ground_truth_invalid(reference_row["outcome"], null_row["outcome"]) else "valid"
        ground_truth_registry.upsert(
            registry,
            task_id=task_id,
            source=task.source,
            prompt_digest=digest,
            reference_outcome=reference_row["outcome"],
            reference_detail=reference_row["detail"],
            null_outcome=null_row["outcome"],
            null_detail=null_row["detail"],
            verdict=verdict,
        )
        added += 1
    return added


def calibrate_models(
    models: list[ModelConfig], selected_tasks: list[SelectedTask], calibration_config: CalibrationConfig,
    details_csv_path: Path | None = None,
) -> tuple[list[ModelCalibrationResult], list[CalibrationDetailRow]]:
    """Runs every (task, model) pair TASKS-OUTER / MODELS-INNER, then aggregates per model. This
    loop order is a deliberate optimization for Docker-image cache hit rate — see
    docs/engineering-notes.md, "Tasks-outer / models-inner loop order" for the measured GB/TB
    figures.

    Within a task's inner loop, controls are graded first; if together they show this task's
    ground truth is bad (`_ground_truth_invalid`), every real model for that task is skipped with a
    synthesized `error_harness` result instead of a real grading call — see "Ground-truth-invalid
    skip rate" for the measured skip rate. `index`/`total` still count a skipped call so `[i/total]`
    progress stays consistent with the logged total.

    Otherwise purely a reordering: statistics are computed after the fact by `_aggregate_outcomes`,
    so results are identical to a models-outer run, and determinism is unaffected since task
    selection (and its RNG) already happened in `select_tasks`."""
    calibration_only = sorted(
        (st for st in selected_tasks if st.split == "calibration"),
        key=lambda st: image_affinity_key(st.task),
    )
    # Controls before real models for every task — _ground_truth_invalid needs their outcome first.
    models = [m for m in models if m.is_control] + [m for m in models if not m.is_control]
    total = len(calibration_only) * len(models)
    logger.info(
        f"calibration started: {len(calibration_only)} tasks x {len(models)} models = {total} calls "
        "(tasks-outer, models-inner)"
    )
    outcomes_by_model: dict[str, list[tuple[SelectedTask, GradeResult]]] = {m.model_id: [] for m in models}
    detail_rows: list[CalibrationDetailRow] = []
    resolved_csv_path = details_csv_path or (ARTIFACTS_DIR / "calibration-details.csv")
    solutions_dir = _solutions_dir(resolved_csv_path)
    writer = _CalibrationDetailsWriter(resolved_csv_path)
    index = 0
    try:
        for st in calibration_only:
            reference_outcome: str | None = None
            null_outcome: str | None = None
            for model in models:
                index += 1
                if not model.is_control and _ground_truth_invalid(reference_outcome, null_outcome):
                    logger.info(f"[{index}/{total}] {model.model_id} task {st.task.task_id} skipped: bad ground truth (reference={reference_outcome}, null={null_outcome})")
                    result = GradeResult(
                        outcome="error_harness",
                        detail=f"skipped: ground truth check failed (reference={reference_outcome}, null={null_outcome})",
                    )
                    record = TaskRunRecord(result=result, solution=None, duration_ms=0)
                else:
                    record = run_and_log(st.task, model, calibration_config, index, total)
                    if model.runner == "reference":
                        reference_outcome = record.result.outcome
                    elif model.runner == "null":
                        null_outcome = record.result.outcome
                outcomes_by_model[model.model_id].append((st, record.result))
                full_solution = record.solution or ""
                if len(full_solution) > _CSV_SOLUTION_MAX_CHARS:
                    _persist_full_solution(solutions_dir, st.task.task_id, model.model_id, full_solution)
                row = CalibrationDetailRow(
                    task_id=st.task.task_id,
                    source=st.task.source,
                    cluster_id=st.cluster_id,
                    split=st.split,
                    model_id=model.model_id,
                    provider=model.provider,
                    outcome=record.result.outcome,
                    detail=record.result.detail,
                    solution=_csv_preview(full_solution),
                    duration_ms=record.duration_ms,
                    input_tokens=record.usage.input_tokens if record.usage else 0,
                    output_tokens=record.usage.output_tokens if record.usage else 0,
                    cost_usd=record.usage.cost_usd if record.usage else 0.0,
                    turns=record.usage.turn_count if record.usage else 0,
                )
                detail_rows.append(row)
                writer.write(row)  # flushed immediately — see _CalibrationDetailsWriter's docstring
    finally:
        writer.close()

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
