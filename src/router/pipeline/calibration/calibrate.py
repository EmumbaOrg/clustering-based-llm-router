from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from ...common.artifacts import ARTIFACTS_DIR
from ...common.assign import ClusterMap
from ...common.config import CalibrationConfig, ModelConfig
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
    cluster_map: ClusterMap,
    task_cluster_map: dict,
) -> list[SelectedTask]:
    """Stratified-by-cluster task selection, driven by the full task->cluster mapping computed
    once at `build-artifact` time (`clustering/task_cluster_map.py`) rather than a per-run random
    pre-filter followed by a fresh embedding pass. This is a deliberate replacement for an earlier
    design that embedded a small, randomly-oversampled candidate pool per source before assigning
    clusters — confirmed against a real run that the random pre-filter could (and did: 2 of 24
    configured clusters ended up with zero tasks) leave a cluster completely unrepresented purely
    by chance, before stratification ever got a chance to run. Grouping directly from a full
    per-task mapping fixes that by construction: every gradeable task is visible before sampling
    starts, and the only way a cluster ends up with zero selected tasks is if it genuinely has zero
    gradeable tasks anywhere in the underlying pool (logged below, not silently dropped) — and it
    removes the embedding step from every calibration run entirely, since the labels are already
    known.

    Raises if `task_cluster_map` wasn't built against the SAME cluster map passed in — a stale or
    mismatched mapping would silently make cluster ids mean different things than the centroids
    `cluster_map` carries, which must fail loudly rather than produce a quietly-wrong selection."""
    if task_cluster_map["cluster_map_id"] != cluster_map.artifact_id:
        raise ValueError(
            f"task-cluster-map.json was built against cluster map {task_cluster_map['cluster_map_id']!r}, "
            f"but the current cluster-map.json is {cluster_map.artifact_id!r} — re-run `build-artifact` "
            "to regenerate a matching task-cluster-map.json."
        )

    rng = random.Random(calibration_config.seed)
    k = cluster_map.centroids.shape[0]
    gradeable_sources = set(calibration_config.gradeable_sources)

    # Real Task objects (full row data, needed for actually running/grading) per gradeable source
    # — task_cluster_map only ever carries task_id/source/cluster_id, never the heavy row data.
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
            # Expected, not an error: a row the corpus includes but calibration's own loader
            # excludes (e.g. a DS-1000 Matplotlib row — corpus.py doesn't apply that filter,
            # calibration/tasks.py does) has a cluster label here but was never gradeable.
            unmatched += 1
            continue
        by_cluster.setdefault(entry["cluster_id"], []).append(task)
    if unmatched:
        logger.info(f"{unmatched} task-cluster-map entries had no matching gradeable task, skipped")

    empty_clusters = [c for c in range(k) if c not in by_cluster]
    if empty_clusters:
        logger.warning(f"{len(empty_clusters)} of {k} clusters have zero gradeable tasks: {empty_clusters}")

    selected: list[SelectedTask] = []
    for cluster_id in sorted(by_cluster):  # explicit, not dict-insertion-order reliance
        tasks_in_cluster = sorted(by_cluster[cluster_id], key=lambda t: t.task_id)  # order-independent before shuffling
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


def compute_task_selection_digest(selected: list[SelectedTask]) -> str:
    """SHA-256 over sorted (task_id, prompt) pairs — same construction as
    `clustering/cluster_map.py::compute_corpus_digest`, applied to the actual selected tasks'
    prompt text rather than the whole corpus. A pinned selection (see `write_task_selection`) is
    only as trustworthy as the guarantee that the underlying task content hasn't silently changed
    since it was written — `load_task_selection` recomputes this and refuses to load on a
    mismatch, rather than grading against content nobody signed off on."""
    hasher = hashlib.sha256()
    for selected_task in sorted(selected, key=lambda s: s.task.task_id):
        hasher.update(selected_task.task.task_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(selected_task.task.prompt.encode("utf-8"))
        hasher.update(b"\0")
    return f"sha256:{hasher.hexdigest()}"


def selected_tasks_to_dict(selected: list[SelectedTask], k: int) -> dict:
    """Only `task_id`/`source`/`cluster_id`/`split` are kept — never the heavy `row` data, which
    `load_task_selection` re-derives fresh from the source at load time (the same reasoning
    `calibration/tasks.py`'s stable ids exist for: the row is a lookup away, not something worth
    duplicating into a second file). `k`, passed by the caller rather than re-derived here, is
    ONLY used to report which clusters got zero tasks — a genuine gap in the gradeable pool
    (`select_tasks` already logs this; this is the file's permanent record of it)."""
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
    """The reverse of `selected_tasks_to_dict` — reconstructs `SelectedTask`s with fresh `row`
    data by calling `load_gradeable_tasks` once per distinct source the pin references (the same
    cost `select_tasks` already pays; embedding, not this, was ever the expensive step). Raises
    (never silently drops) on either a pinned task_id no longer present in its source's gradeable
    pool, or a content-digest mismatch — an incremental calibration run (see cli.py's `--model`)
    depends on grading the new model against the EXACT set an existing model-profiles.json was
    built from, so a silently-smaller or silently-changed set here would be worse than a hard
    failure telling the caller to investigate."""
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
    (the gold answer for `reference`, empty for `null`, the extracted/diff solution for `pi` —
    falling back to the raw agent response when no solution could be extracted, so a CSV/log
    consumer can still see what the agent said even when extraction failed). `usage` is Pi's own
    reported token/cost usage for the call, or None for `reference`/`null` (synthesized, no real
    call) and for a `pi` call whose stdout wasn't parseable JSON. Kept alongside `GradeResult`
    rather than folded into it so `GradeResult` stays the small, stable type graders/tests already
    build on."""
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
        if run_result.context_unavailable:
            # Repo clone/checkout failed before pi was ever invoked — an infra problem, not the
            # model's fault, and per the spec's fairness requirement a task that can't be set up
            # consistently for every model shouldn't be scored for any of them.
            result = GradeResult(outcome="error_harness", detail=run_result.detail)
        elif run_result.rate_limited:
            # An infra/quota rejection, not the model failing to answer — excluded rather than
            # counted as a wrong answer, same reasoning as error_timeout/error_missing_dep.
            result = GradeResult(outcome="error_harness", detail=run_result.detail)
        elif run_result.harness_error:
            # The provider itself rejected the call (e.g. an API auth failure) — pi exits 0 in this
            # case, so it never reached the model at all. Confirmed live this session: 15% of one
            # model's calls in one run hit this, previously miscounted as error_no_solution.
            result = GradeResult(outcome="error_harness", detail=run_result.detail)
        elif run_result.timed_out:
            # Our own subprocess timeout fired — a call that never finished isn't evidence the
            # model couldn't solve the task, just that it didn't in the time we gave it. Distinct
            # from a grader's own error_timeout (about the test run, not the model call).
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
        return result, returned, run_result.usage

    raise ValueError(f"unknown runner {model.runner!r} for model {model.model_id}")


@dataclasses.dataclass(frozen=True)
class TaskRunRecord:
    """`run_and_grade`'s (GradeResult, solution, usage) triple plus timing — one row of "what
    actually happened" for a single (task, model) call, kept separate from `GradeResult` for the
    same reason `run_and_grade` returns a tuple instead of widening it (see that function's
    docstring)."""
    result: GradeResult
    solution: str | None
    duration_ms: int
    usage: runner_mod.TokenUsage | None = None


def run_and_log(task: Task, model: ModelConfig, calibration_config: CalibrationConfig, index: int, total: int) -> TaskRunRecord:
    """`run_and_grade` plus the timing/progress log line — shared by calibrate_model's per-model
    loop below and evaluate.py's per-model x per-task holdout loop, which otherwise duplicate this
    exact started/duration/excluded-vs-not branch."""
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
    calls hit a warm image instead of re-pulling. Matters most for swe-smith, whose `image_name` is
    per bug-injected repo (~128 distinct images across ~59K instances) rather than per instance —
    swe-gym and multi-swe-rl build one image PER instance, so for them this only groups by repo,
    which is still the right tiebreak for their shared git clones (repo_context.py)."""
    return (task.source, str(task.row.get("image_name") or task.row.get("repo") or task.task_id))


@dataclasses.dataclass(frozen=True)
class CalibrationDetailRow:
    """One (task, model) call, flattened for `calibration-details-<run timestamp>.csv` —
    everything `run_and_log` knows about a single call, alongside the task identity it was made
    for. No per-row timestamp — the run's start time lives in the CSV's filename instead (see
    calibrate_models' `details_csv_path` and cli.py's `calibrate` command), which is what actually
    stops one run's file from overwriting another's; a per-row timestamp inside a single run's file
    never distinguished anything and just gave every row in that file the same-ish value anyway."""
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
    cost_usd: float = 0.0  # Pi's own reported cost for this call (see TokenUsage) — 0 for the
    # reference/null controls (synthesized, no real call) and for a call whose usage genuinely
    # couldn't be read back, not a claim that the call was free.


_CSV_FIELDNAMES = [
    "task_id", "source", "cluster_id", "split", "model_id", "provider",
    "outcome", "detail", "solution", "duration_ms", "input_tokens", "output_tokens", "cost_usd",
]


def _csv_row_values(row: CalibrationDetailRow) -> list:
    return [
        row.task_id, row.source, row.cluster_id, row.split, row.model_id,
        row.provider, row.outcome, row.detail, row.solution, row.duration_ms,
        row.input_tokens, row.output_tokens, row.cost_usd,
    ]


# Truncation length for the `solution` column — long enough to see the shape of a real answer
# (a full diff/patch can be huge), short enough that a multi-MB swe-gym patch doesn't blow up the
# CSV. Same idea as _LOG_PREVIEW_CHARS above, just a separate constant since a CSV meant to be
# opened in a spreadsheet tool can afford to keep more than a debug log line.
_CSV_SOLUTION_MAX_CHARS = 2000


def _csv_preview(text: str) -> str:
    if len(text) <= _CSV_SOLUTION_MAX_CHARS:
        return text
    return text[:_CSV_SOLUTION_MAX_CHARS] + f" ...[{len(text) - _CSV_SOLUTION_MAX_CHARS} more chars]"


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def _solutions_dir(details_csv_path: Path) -> Path:
    # Keyed by the details CSV's own filename stem (which already embeds the run timestamp — see
    # CalibrationDetailRow's docstring) so each run's persisted solutions land in their own
    # directory, the same way the CSV itself never overwrites a different run's file.
    return details_csv_path.parent / "solutions" / details_csv_path.stem


def _persist_full_solution(solutions_dir: Path, task_id: str, model_id: str, solution: str) -> None:
    """Only called when `_csv_preview` actually truncated `solution` — a short solution is already
    complete in the CSV, so a side file for it would be pure duplication. Confirmed this session
    why this matters: once `_csv_preview` truncates a large diff (e.g. Multi-SWE-RL's
    `checkstyle-15001`, 613,777 chars), the original text is gone for good — it only ever existed
    in memory for this one call — so there was no way to audit a large apply failure after the
    fact without re-running a fresh container by hand."""
    solutions_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = _UNSAFE_FILENAME_CHARS.sub("_", task_id)
    safe_model_id = _UNSAFE_FILENAME_CHARS.sub("_", model_id)
    (solutions_dir / f"{safe_task_id}__{safe_model_id}.diff").write_text(solution, encoding="utf-8")


class _CalibrationDetailsWriter:
    """Writes calibration-details.csv incrementally — one row per (task, model) call, flushed to
    disk immediately — rather than accumulating everything in memory and writing once at the end.
    A calibration run can take 30+ minutes (real Docker-based grading calls run several minutes
    each) and this session saw more than one run interrupted partway through; without incremental
    writes, an interrupted run left no CSV at all, and a still-running one couldn't be inspected."""

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


def write_calibration_details_csv(rows: list[CalibrationDetailRow], path: Path | None = None) -> Path:
    """One-shot variant, kept for regenerating the CSV from an already-in-memory row list (e.g. in
    a test or a notebook) — a real `calibrate` run writes incrementally via
    `_CalibrationDetailsWriter` instead, so its CSV survives an interrupted run."""
    target = path or (ARTIFACTS_DIR / "calibration-details.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_FIELDNAMES)
        for row in rows:
            writer.writerow(_csv_row_values(row))
    return target


def calibrate_models(
    models: list[ModelConfig], selected_tasks: list[SelectedTask], calibration_config: CalibrationConfig,
    details_csv_path: Path | None = None,
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
    resolved_csv_path = details_csv_path or (ARTIFACTS_DIR / "calibration-details.csv")
    solutions_dir = _solutions_dir(resolved_csv_path)
    writer = _CalibrationDetailsWriter(resolved_csv_path)
    index = 0
    try:
        for st in calibration_only:
            for model in models:
                index += 1
                record = run_and_log(st.task, model, calibration_config, index, total)
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
