"""Loads full task rows (including ground truth) for calibration/evaluation — distinct from
corpus.py, which loads text-only rows for the clustering corpus. Task ids here are STABLE
dataset-native identifiers (`task_id` / `instance_id` / `metadata.problem_id`), not positional
post-shuffle indices, so a task can be referenced across separate runs without depending on load
order.

Only the gradeable sources (see config/calibration.yaml) are covered. swe-smith, swe-gym, and
multi-swe-rl's Go/JS/TS/Java/Rust slices are all wired below (see grading/swesmith.py,
grading/swegym.py, grading/multiswerl.py).
"""
from __future__ import annotations

import json
import logging

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ..corpus import (
    _MULTI_SWE_RL_FILE_SUFFIX,
    MULTI_SWE_RL_BATCH,
    SOURCE_METADATA,
    _extract_multi_swe_rl_text,
    _multi_swe_rl_repo_files,
    _multi_swe_rl_row_id,
    stable_task_id,
)
from .grading.base import Task

logger = logging.getLogger(__name__)


def _bigcodebench_tasks() -> list[Task]:
    meta = SOURCE_METADATA["bigcodebench"]
    ds = load_dataset(meta["hf_id"], split=meta["split"])
    tasks = []
    for row in ds:
        task_id = stable_task_id("bigcodebench", row)
        if task_id is None:
            continue
        tasks.append(
            Task(
                task_id=task_id,
                source="bigcodebench",
                prompt=row["instruct_prompt"],
                reference_solution=row["canonical_solution"],
                row=dict(row),
            )
        )
    return tasks


def _ds1000_tasks() -> list[Task]:
    meta = SOURCE_METADATA["ds1000"]
    ds = load_dataset(meta["hf_id"], split=meta["split"])
    tasks = []
    for row in ds:
        # Matplotlib rows grade by rendered-PNG comparison against a reference image we'd first
        # have to render ourselves — excluded here rather than half-implemented in ds1000.py.
        # corpus.py's own loader does NOT apply this filter (a broader corpus is fine including
        # them), so a Matplotlib row can appear in task-cluster-map.json without ever appearing in
        # this loader's output — select_tasks() intersects against this function's actual return
        # value rather than trusting the map wholesale, specifically to handle this.
        if row["metadata"]["library"] == "Matplotlib":
            continue
        task_id = stable_task_id("ds1000", row)
        if task_id is None:
            continue
        tasks.append(
            Task(
                task_id=task_id,
                source="ds1000",
                prompt=row["prompt"],
                reference_solution=row["reference_code"],
                row=dict(row),
            )
        )
    return tasks


# Matches corpus.py's own SWE_SMITH_SAMPLE_SIZE (same seed 42, same filter) so this loader walks
# the identical shuffled sequence used to build task-cluster-map.json's swe-smith cluster labels —
# see docs/engineering-notes.md, "SWE-smith task pool cap".
SWESMITH_TASK_POOL_CAP = 20_000
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
        task_id = stable_task_id("swe-smith", row)
        if task_id is None:
            continue
        tasks.append(
            Task(
                task_id=task_id,
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
    tasks = []
    for row in ds:
        task_id = stable_task_id("swe-gym", row)
        if task_id is None:
            continue
        tasks.append(
            Task(
                task_id=task_id,
                source="swe-gym",
                prompt=row["problem_statement"],
                reference_solution="",  # unused for this source — see swe-smith's loader above.
                row=dict(row),
            )
        )
    return tasks


# Keeps only the fields grading/repo_context actually touch — see docs/engineering-notes.md,
# "Multi-SWE-RL field trim". Each entry below names its reader so this stays checkable.
_MULTI_SWE_RL_GRADED_FIELDS = frozenset({
    "instance_id",  # not read by the grader — kept as the row's own stable id, and only ~21 B/row
    "org", "repo", "number",  # grading/multiswerl.py: _image / _repo_dir
    "base",  # repo_context.py: _multiswerl_remote_and_ref reads base["sha"]
    "fix_patch",  # grading/multiswerl.py: grade_reference (the GOLD fix)
    "test_patch",  # grading/multiswerl.py: _setup_script
    "f2p_tests", "n2p_tests", "s2p_tests", "p2p_tests",  # grading/multiswerl.py: _test_names
})


# Go (1,675 tasks), JS (619), TS (412), Java (976), and Rust (215) — see grading/multiswerl.py's
# module docstring for why the other 2 languages in this dataset (C, C++) aren't gradeable yet.
_MULTI_SWE_RL_GRADEABLE_LANGUAGES = ("go", "js", "ts", "java", "rust")


def _multi_swe_rl_tasks() -> list[Task]:
    """Filters `corpus.py`'s own file listing down to the gradeable-language directories above
    before downloading anything, so this loader never pays for the other 4 languages' files.
    Reuses `corpus.py`'s own text-extraction and row-id helpers rather than reimplementing them —
    this loader's only real job is the language filter and the `Task` wrapping."""
    meta = SOURCE_METADATA["multi-swe-rl"]
    prefixes = tuple(f"{MULTI_SWE_RL_BATCH}/{language}/" for language in _MULTI_SWE_RL_GRADEABLE_LANGUAGES)
    paths = sorted(
        path for path, _ in _multi_swe_rl_repo_files()
        if path.startswith(prefixes) and path.endswith(_MULTI_SWE_RL_FILE_SUFFIX)
    )
    tasks: list[Task] = []
    for path in paths:
        local_path = hf_hub_download(meta["hf_id"], filename=path, repo_type="dataset")
        with open(local_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                text = _extract_multi_swe_rl_text(row)
                task_id = _multi_swe_rl_row_id(row)
                if text is None or task_id is None:
                    continue
                tasks.append(
                    Task(
                        task_id=task_id,
                        source="multi-swe-rl",
                        prompt=text,
                        reference_solution="",  # unused for this source — see swe-smith's loader above.
                        row={k: v for k, v in row.items() if k in _MULTI_SWE_RL_GRADED_FIELDS},
                    )
                )
    return tasks


_LOADERS = {
    "bigcodebench": _bigcodebench_tasks,
    "ds1000": _ds1000_tasks,
    "swe-smith": _swesmith_tasks,
    "swe-gym": _swegym_tasks,
    "multi-swe-rl": _multi_swe_rl_tasks,
}


def load_gradeable_tasks(source: str) -> list[Task]:
    if source not in _LOADERS:
        raise ValueError(f"no task loader for source {source!r} — gradeable sources are {sorted(_LOADERS)}")
    tasks = _LOADERS[source]()
    logger.info(f"loaded {len(tasks)} gradeable tasks from {source}")
    return tasks
