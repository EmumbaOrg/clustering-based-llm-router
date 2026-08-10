"""Grader for SWE-smith — Docker-based. Each row's `image_name` is a prebuilt image with the
target repo checked out at its CLEAN state and the test suite present — no `test_patch`, no clone
needed. (`repo` is a synthetic `swesmith/...` namespace, not a real GitHub repo, so cloning isn't
an option even in principle — see pipeline README.)

Confirmed empirically against `jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536`
(the plan's "validate on one instance" step) — three things that were WRONG in the first draft,
before that check:

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

REQUIRES the command sandbox disabled for every `docker` call — the sandbox blocks the Docker
socket, which is what made the daemon look unreachable during initial environment checks (see the
calibration plan doc).
"""
from __future__ import annotations

import base64
import shlex
import subprocess

from .base import GradeResult, Task

DOCKER_TIMEOUT_SECONDS = 300  # image pull (if not cached) + container run
REPO_DIR = "/testbed"  # confirmed against the one instance checked above
CONDA_ACTIVATE = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"


def _run_docker(image: str, script: str, timeout_seconds: int) -> GradeResult:
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", "-i", image, "bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return GradeResult(outcome="error_timeout", detail=f"exceeded {timeout_seconds}s")
    except FileNotFoundError:
        return GradeResult(outcome="error_harness", detail="docker binary not found on PATH")

    output = (proc.stdout + proc.stderr)[-2000:]
    if "docker: " in output.lower() and "image" in output.lower() and proc.returncode != 0:
        return GradeResult(outcome="error_harness", detail=f"docker itself failed: {output}")
    if proc.returncode == 0:
        return GradeResult(outcome="pass", detail="")
    return GradeResult(outcome="fail", detail=f"exit {proc.returncode}: {output}")


def _node_ids(task: Task) -> str:
    ids = task.row["FAIL_TO_PASS"] + task.row["PASS_TO_PASS"]
    return " ".join(shlex.quote(t) for t in ids)


def _write_patch_cmd(patch_text: str, dest: str) -> str:
    patch_b64 = base64.b64encode(patch_text.encode("utf-8")).decode("ascii")
    return f"echo {patch_b64} | base64 -d > {dest}"


def _apply_bug_patch(task: Task) -> str:
    return f"{_write_patch_cmd(task.row['patch'], '/tmp/bug.patch')} && git apply /tmp/bug.patch"


def _test_cmd(task: Task) -> str:
    return f"{CONDA_ACTIVATE} && cd {REPO_DIR} && python -m pytest -q {_node_ids(task)}"


def grade(task: Task, solution: str, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """`solution` is a forward-apply unified diff — the shape a real candidate/agent produces,
    applied on top of the freshly-established buggy baseline."""
    image = task.row["image_name"]
    if not solution.strip():
        return GradeResult(outcome="fail", detail="empty patch — bug remains unfixed")

    script = (
        f"cd {REPO_DIR} && {_apply_bug_patch(task)} && "
        f"{_write_patch_cmd(solution, '/tmp/candidate.patch')} && git apply /tmp/candidate.patch && "
        f"{_test_cmd(task)}"
    )
    return _run_docker(image, script, timeout_seconds)


def grade_reference(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: establish the bug, then reverse the SAME patch — a round trip
    back to the clean state — and confirm tests pass. Must score ~100% or the grader is broken."""
    image = task.row["image_name"]
    script = (
        f"cd {REPO_DIR} && {_apply_bug_patch(task)} && "
        f"git apply -R /tmp/bug.patch && "
        f"{_test_cmd(task)}"
    )
    return _run_docker(image, script, timeout_seconds)


def grade_null(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: establish the bug and apply no fix. FAIL_TO_PASS tests are
    expected to fail. Must score ~0% or the grader is broken."""
    image = task.row["image_name"]
    script = f"cd {REPO_DIR} && {_apply_bug_patch(task)} && {_test_cmd(task)}"
    return _run_docker(image, script, timeout_seconds)
