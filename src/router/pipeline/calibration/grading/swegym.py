"""Grader for SWE-Gym — Docker-based, built on `dockerexec.py`'s sentinel protocol, mirroring
`swesmith.py`'s shape closely but with different row semantics: **`patch` here is the GOLD FIX**,
not a bug-injecting diff — `base_commit` is already the buggy, pre-fix state the prebuilt image is
built at, and a separate `test_patch` field carries the test changes needed to exercise
FAIL_TO_PASS/PASS_TO_PASS, applied in every mode.

Images aren't named in the row; `_image` constructs the Docker Hub tag from `instance_id`. Passes
FAIL_TO_PASS + PASS_TO_PASS directly as pytest node-id arguments rather than replicating the
official harness's whole-file-plus-log-parsing approach — no per-test granularity within one run,
in exchange for reusing `dockerexec.py`'s sentinel/classify protocol unchanged.
"""
from __future__ import annotations

import uuid

from . import dockerexec
from .base import GradeResult, Task

DOCKER_TIMEOUT_SECONDS = dockerexec.DOCKER_TIMEOUT_SECONDS
REPO_DIR = "/testbed"
CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate && conda activate testbed"


def _image(task: Task) -> str:
    return f"xingyaoww/sweb.eval.x86_64.{task.row['instance_id'].replace('__', '_s_')}:latest".lower()


def _setup_script(task: Task, nonce: str) -> str:
    """Shared by all three modes: apply the test changes every mode needs to exercise
    FAIL_TO_PASS/PASS_TO_PASS. A failure here is always `error_harness` — it's the dataset's own
    test_patch against its own prebuilt image, never the candidate's fault."""
    harness_no_repo = dockerexec.report_cmd(nonce, "HARNESS", f"missing {REPO_DIR}")
    harness_test_patch_apply = dockerexec.report_cmd(nonce, "HARNESS", "test_patch failed to apply")
    return (
        f"cd {REPO_DIR} || {{ {harness_no_repo}; }}\n"
        f"{dockerexec.write_file_cmd(task.row['test_patch'], '/tmp/test.patch')}\n"
        f"git apply /tmp/test.patch || {{ {harness_test_patch_apply}; }}\n"
    )


def _pytest_script(task: Task, nonce: str) -> str:
    """Run the test suite and report PASS/FAIL off its exit code. `dockerexec.pytest_collect_then_run`
    handles the case where some `FAIL_TO_PASS`/`PASS_TO_PASS` node ids can't be collected as given."""
    return dockerexec.pytest_collect_then_run(
        node_ids=task.row["FAIL_TO_PASS"] + task.row["PASS_TO_PASS"],
        nonce=nonce,
        repo_dir=REPO_DIR,
        conda_activate=CONDA_ACTIVATE,
        fail_detail="pytest reported failures",
    )


def grade(task: Task, solution: str, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """`solution` is a forward-apply unified diff, applied on top of the test-patched baseline.
    An inapplicable candidate diff is `fail` (the model's own failure), not `error_harness`. Any
    path `test_patch` already touches is excluded from the candidate's own patch."""
    if not solution.strip():
        return GradeResult(outcome="fail", detail="empty patch — bug remains unfixed")

    image = _image(task)
    nonce = uuid.uuid4().hex
    exclude_paths = dockerexec.diff_touched_paths(task.row["test_patch"])
    script = (
        _setup_script(task, nonce)
        + f"{dockerexec.write_file_cmd(solution, '/tmp/candidate.patch')}\n"
        + dockerexec.apply_patch_or_fail_cmd(nonce, "/tmp/candidate.patch", exclude_paths=exclude_paths)
        + _pytest_script(task, nonce)
    )
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds)
    finally:
        dockerexec.touch_image(image)


def grade_reference(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: apply the test patch, then the real GOLD fix (`task.row["patch"]`
    — opposite semantics from swesmith, where `patch` is the bug). Must score ~100% or the grader
    is broken. A gold-patch apply failure is `error_harness`, not `fail` — the dataset's own fix
    failing to apply is a dataset/image problem, never a signal about candidate quality."""
    image = _image(task)
    nonce = uuid.uuid4().hex
    harness_gold_apply = dockerexec.report_cmd(nonce, "HARNESS", "gold patch failed to apply")
    script = (
        _setup_script(task, nonce)
        + f"{dockerexec.write_file_cmd(task.row['patch'], '/tmp/gold.patch')}\n"
        + f"git apply /tmp/gold.patch || {{ {harness_gold_apply}; }}\n"
        + _pytest_script(task, nonce)
    )
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds)
    finally:
        dockerexec.touch_image(image)


def grade_null(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: apply the test patch and no fix at all. FAIL_TO_PASS tests are
    expected to fail. Must score ~0% or the grader is broken."""
    image = _image(task)
    nonce = uuid.uuid4().hex
    script = _setup_script(task, nonce) + _pytest_script(task, nonce)
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds)
    finally:
        dockerexec.touch_image(image)
