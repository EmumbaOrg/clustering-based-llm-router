"""Grader for SWE-smith — Docker-based, built on `dockerexec.py`'s sentinel protocol. Each row's
`image_name` is a prebuilt image with the target repo checked out at its CLEAN state and the test
suite present — no `test_patch`, no clone needed. (`repo` is a synthetic `swesmith/...` namespace,
not a real GitHub repo, so cloning isn't an option even in principle — see pipeline README.)

Confirmed empirically against `jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536`
(the calibration plan's "validate on one instance" step) — three things that were WRONG in the
first draft, before that check:

1. **The image starts clean, not buggy.** `patch` (the row's bug-introducing diff) has NOT been
   applied yet — `git log` shows a single "Initial commit" and the pre-bug source is what's
   checked out. Establishing the buggy baseline (`git apply` the row's `patch`, forward) is a
   setup step every grading mode needs, not something already done for you.
2. **Tests need a specific conda env activated first.** `python -m pytest` against the base
   interpreter fails with "No module named pytest" — `conda activate testbed` first is required.
3. Given (1), the modes are: `grade()` applies the bug forward, then the candidate's own fix diff,
   then tests. `grade_reference()` applies the bug forward, then REVERSES THE SAME PATCH (`git
   apply -R`) — a round trip back to the clean state — then tests (expect pass). `grade_null()`
   applies the bug forward and tests with no fix at all (expect fail).

Docker must be reachable wherever this runs (`docker info` should succeed) — no other special
setup is required today.

Failure classification, enforced via `dockerexec`'s sentinel protocol rather than trusting the
container's own exit code: applying the dataset's OWN bug patch (or, in `grade_reference`,
reversing it) failing is `error_harness` in every mode — it's the dataset's/image's problem, never
the candidate's. Only the candidate's own patch failing to apply (`grade`, only) is `fail`. The
actual test run's exit code (0 vs. nonzero) maps to pass/fail identically across all three modes.
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
    """Run the test suite and report PASS/FAIL off its exit code.

    One exception, exit codes 4/5 (pytest's own "usage error" / "no tests collected"): a
    FAIL_TO_PASS/PASS_TO_PASS node id that pytest can't collect as given — same class of issue
    documented at length in swegym.py's `_pytest_script` (a dataset-recorded id not matching what
    the installed pytest actually generates, whether from a non-ASCII parametrize case or an id
    truncated mid-value at an embedded comma, both confirmed live against real SWE-Gym rows this
    session). This module shares swegym.py's `FAIL_TO_PASS`/`PASS_TO_PASS` node-id shape and the
    same single-batch invocation, so it's exposed to the identical failure mode — previously
    unhandled here, which silently miscounted an infra/dataset problem as a genuine model `FAIL`.
    `dockerexec.pytest_collect_then_run` routes it to HARNESS instead, and additionally recovers
    whichever other ids in the same batch ARE collectible rather than losing the whole task's
    signal to one bad id."""
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
