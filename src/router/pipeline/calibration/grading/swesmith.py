"""Grader for SWE-smith — Docker-based, built on `dockerexec.py`'s sentinel protocol. Each row's
`image_name` is a prebuilt image checked out at its CLEAN (pre-bug) state, so `_setup_script`
applies `patch` (the bug) forward first. `grade` then applies the candidate's fix; `grade_reference`
reverses the same patch instead; `grade_null` applies no fix at all.
"""
from __future__ import annotations

import uuid

from . import dockerexec
from .base import GradeResult, Task

DOCKER_TIMEOUT_SECONDS = dockerexec.DOCKER_TIMEOUT_SECONDS
REPO_DIR = "/testbed"  # confirmed against the one instance checked above
CONDA_ACTIVATE = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"


def _setup_script(task: Task, nonce: str) -> str:
    """Shared by all three modes: establish the buggy baseline. A failure here is always
    `error_harness` — it's the dataset's own patch against its own clean image, never the
    candidate's fault."""
    harness_no_repo = dockerexec.report_cmd(nonce, "HARNESS", f"missing {REPO_DIR}")
    harness_bug_apply = dockerexec.report_cmd(nonce, "HARNESS", "bug patch failed to apply")
    return (
        f"cd {REPO_DIR} || {{ {harness_no_repo}; }}\n"
        f"{dockerexec.write_file_cmd(task.row['patch'], '/tmp/bug.patch')}\n"
        f"git apply /tmp/bug.patch || {{ {harness_bug_apply}; }}\n"
    )


def _pytest_script(task: Task, nonce: str) -> str:
    """Run the test suite and report PASS/FAIL off its exit code. Same node-id collection handling
    as swegym.py's `_pytest_script`."""
    return dockerexec.pytest_collect_then_run(
        node_ids=task.row["FAIL_TO_PASS"] + task.row["PASS_TO_PASS"],
        nonce=nonce,
        repo_dir=REPO_DIR,
        conda_activate=CONDA_ACTIVATE,
        fail_detail="pytest reported failures",
    )


def grade(task: Task, solution: str, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """`solution` is a forward-apply unified diff — the shape a real candidate/agent produces,
    applied on top of the freshly-established buggy baseline. An inapplicable candidate diff is
    `fail` (the model's own failure), not `error_harness`."""
    if not solution.strip():
        return GradeResult(outcome="fail", detail="empty patch — bug remains unfixed")

    image = task.row["image_name"]
    nonce = uuid.uuid4().hex
    script = (
        _setup_script(task, nonce)
        + f"{dockerexec.write_file_cmd(solution, '/tmp/candidate.patch')}\n"
        + dockerexec.apply_patch_or_fail_cmd(nonce, "/tmp/candidate.patch")
        + _pytest_script(task, nonce)
    )
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds)
    finally:
        dockerexec.touch_image(image)


def grade_reference(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: establish the bug, then reverse the SAME patch — a round trip
    back to the clean state — and confirm tests pass. Must score ~100% or the grader is broken.
    A reverse-apply failure is `error_harness`, not `fail` — the gold patch failing to undo itself
    is a dataset/image problem, never a signal about candidate quality."""
    image = task.row["image_name"]
    nonce = uuid.uuid4().hex
    harness_reverse = dockerexec.report_cmd(nonce, "HARNESS", "bug patch failed to reverse-apply")
    script = (
        _setup_script(task, nonce)
        + f"git apply -R /tmp/bug.patch || {{ {harness_reverse}; }}\n"
        + _pytest_script(task, nonce)
    )
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds)
    finally:
        dockerexec.touch_image(image)


def grade_null(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: establish the bug and apply no fix. FAIL_TO_PASS tests are
    expected to fail. Must score ~0% or the grader is broken."""
    image = task.row["image_name"]
    nonce = uuid.uuid4().hex
    script = _setup_script(task, nonce) + _pytest_script(task, nonce)
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds)
    finally:
        dockerexec.touch_image(image)
