"""Loads full task rows (including ground truth) for calibration/evaluation — distinct from
corpus.py, which loads text-only rows for the clustering corpus. Task ids here are STABLE
dataset-native identifiers (`task_id` / `instance_id` / `metadata.problem_id`), not positional
post-shuffle indices, so a task can be referenced across separate runs without depending on load
order.

Only the gradeable sources (see config/calibration.yaml) are covered. swe-smith and swe-gym are
both wired below — each Docker grader was validated against real instances before its loader was
added (see grading/swesmith.py, grading/swegym.py).
"""
from __future__ import annotations

import logging

from datasets import load_dataset

from ..corpus import SOURCE_METADATA
from .grading.base import Task

logger = logging.getLogger(__name__)


def _bigcodebench_tasks() -> list[Task]:
    meta = SOURCE_METADATA["bigcodebench"]
    ds = load_dataset(meta["hf_id"], split=meta["split"])
    return [
        Task(
            task_id=row["task_id"],
            source="bigcodebench",
            prompt=row["instruct_prompt"],
            reference_solution=row["canonical_solution"],
            row=dict(row),
        )
        for row in ds
    ]


def _ds1000_tasks() -> list[Task]:
    meta = SOURCE_METADATA["ds1000"]
    ds = load_dataset(meta["hf_id"], split=meta["split"])
    tasks = []
    for row in ds:
        # Matplotlib rows grade by rendered-PNG comparison against a reference image we'd first
        # have to render ourselves — excluded here rather than half-implemented in ds1000.py.
        if row["metadata"]["library"] == "Matplotlib":
            continue
        tasks.append(
            Task(
                task_id=f"ds1000:{row['metadata']['problem_id']}",
                source="ds1000",
                prompt=row["prompt"],
                reference_solution=row["reference_code"],
                row=dict(row),
            )
        )
    return tasks


# select_tasks only ever needs tasks_per_cluster * k * candidate_pool_oversample rows per source
# (well under 2,000 even at the spec's full-scale k=24/tasks_per_cluster=20) — loading and
# filtering all ~59K SWE-smith rows just to sample from them costs minutes on every
# calibrate/evaluate/validate-graders invocation for no benefit. Shuffled first, then filtered for
# a non-empty problem_statement (same order as corpus.py's own load for this dataset), so capping
# doesn't bias toward whichever rows the filter would have skipped anyway.
SWESMITH_TASK_POOL_CAP = 5_000
_SWESMITH_POOL_SEED = 42


def _swesmith_tasks() -> list[Task]:
    meta = SOURCE_METADATA["swe-smith"]
    ds = load_dataset(meta["hf_id"], split=meta["split"]).shuffle(seed=_SWESMITH_POOL_SEED)
    tasks = []
    for row in ds:
        if len(tasks) >= SWESMITH_TASK_POOL_CAP:
            break
        prompt = row.get("problem_statement")
        if not prompt or not str(prompt).strip():
            # A meaningful fraction of rows are synthetic mutation tasks with no generated NL
            # description — see corpus.py's own filter for the same dataset.
            continue
        tasks.append(
            Task(
                task_id=row["instance_id"],
                source="swe-smith",
                prompt=prompt,
                reference_solution="",  # unused for this source: calibrate.py's "reference"/"null"
                # branches special-case swe-smith to swesmith.grade_reference/grade_null (bug-patch
                # reversal / bug-only), never by grading this field as a solution string.
                row=dict(row),
            )
        )
    return tasks


def _swegym_tasks() -> list[Task]:
    meta = SOURCE_METADATA["swe-gym"]
    ds = load_dataset(meta["hf_id"], split=meta["split"])
    return [
        Task(
            task_id=row["instance_id"],
            source="swe-gym",
            prompt=row["problem_statement"],
            reference_solution="",  # unused: calibrate.py special-cases swe-gym to
            # swegym.grade_reference/grade_null, same reasoning as swe-smith above.
            row=dict(row),
        )
        for row in ds
    ]


_LOADERS = {
    "bigcodebench": _bigcodebench_tasks,
    "ds1000": _ds1000_tasks,
    "swe-smith": _swesmith_tasks,
    "swe-gym": _swegym_tasks,
}


def load_gradeable_tasks(source: str) -> list[Task]:
    if source not in _LOADERS:
        raise ValueError(f"no task loader for source {source!r} — gradeable sources are {sorted(_LOADERS)}")
    tasks = _LOADERS[source]()
    logger.info(f"loaded {len(tasks)} gradeable tasks from {source}")
    return tasks
