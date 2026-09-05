"""Grader for SWE-Gym — Docker-based, built on `dockerexec.py`'s sentinel protocol, mirroring
`swesmith.py`'s shape closely but with different row semantics.

Confirmed empirically (Docker Hub API + real `docker pull`/`docker run` against `getmoto/moto` and
`pandas-dev/pandas` instances) rather than assumed from the row schema alone:

1. **Prebuilt per-instance Docker images exist and are public**, under Docker Hub namespace
   `xingyaoww/sweb.eval.x86_64.{instance_id}` with `__` replaced by `_s_` (dunders aren't allowed
   in Docker Hub image names) — confirmed for every one of the 11 repos in this dataset. SWE-Gym
   rows don't carry an `image_name` field the way swe-smith's do, so the tag is constructed instead
   of read; see `_image`.
2. **No repo-specific `install`/environment-setup step is needed at grade time** — every prebuilt
   image already has its package editable-installed. See docs/engineering-notes.md, "SWE-Gym /
   pandas meson rebuild".
3. **Working dir is always `/testbed`, conda env is always named `testbed`** — same universal
   convention as swesmith's images, though the exact activation script's path differs (confirmed
   directly, not assumed identical): `source /opt/miniconda3/bin/activate`, not
   `/opt/miniconda3/etc/profile.d/conda.sh`.
4. **Plain `git apply` is sufficient** — no `patch --fuzz` fallback needed. Verified across one
   real instance from each of the 11 repos in this dataset: every `test_patch` and gold `patch`
   applied cleanly.
5. **A small fraction of rows (~2.8%, confirmed by direct count) have a `FAIL_TO_PASS`/
   `PASS_TO_PASS` id pytest can't collect as given** — a non-ASCII character (e.g. an emoji in a
   parametrize case) recorded literally at dataset-collection time, while the grading image's own
   installed pytest version escapes it when generating node ids. Caught via pytest's exit codes 4
   ("usage error") / 5 ("no tests collected"), routed to `HARNESS` rather than `FAIL` — see
   `_pytest_script`. Discovered via a real `validate-graders` run (not assumed), a direct
   consequence of passing exact node ids rather than replicating the official harness's whole-file
   grading (see the "deliberate simplification" note below).

Row semantics differ from swesmith in one important way: **`patch` here is the GOLD FIX**, not a
bug-injecting diff — `base_commit` is already the buggy, pre-fix state the image is built at. A
separate `test_patch` field (absent from swesmith) carries the test changes needed to actually
exercise `FAIL_TO_PASS`/`PASS_TO_PASS` and must be applied in every mode.

Like swesmith's grader, this passes `FAIL_TO_PASS + PASS_TO_PASS` directly as pytest node-id
arguments rather than replicating the official harness's whole-file-plus-log-parsing approach —
verified valid (not just convenient) by checking that these fields are standard pytest node-id
strings across a getmoto/mypy/pandas sample, independent of what each repo's own `test_cmd`
convention happens to be upstream. Trade-off: no per-test granularity within one run, in exchange
for reusing `dockerexec.py`'s sentinel/classify protocol unchanged.
"""
from __future__ import annotations

import uuid

from . import dockerexec
from .base import GradeResult, Task

DOCKER_TIMEOUT_SECONDS = dockerexec.DOCKER_TIMEOUT_SECONDS
REPO_DIR = "/testbed"  # confirmed against real getmoto/moto and pandas-dev/pandas instances
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
    handles the case where some `FAIL_TO_PASS`/`PASS_TO_PASS` node ids can't be collected as given —
    see docs/engineering-notes.md, "Pytest collection mismatch (swegym/swesmith)"."""
    return dockerexec.pytest_collect_then_run(
        node_ids=task.row["FAIL_TO_PASS"] + task.row["PASS_TO_PASS"],
        nonce=nonce,
        repo_dir=REPO_DIR,
        conda_activate=CONDA_ACTIVATE,
        fail_detail="pytest reported failures",
    )


def grade(task: Task, solution: str, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """`solution` is a forward-apply unified diff — the shape a real candidate/agent produces,
    applied on top of the test-patched baseline. An inapplicable candidate diff is `fail` (the
    model's own failure), not `error_harness`.

    Excludes any path `test_patch` already touches from the candidate's own patch — see
    `dockerexec.apply_patch_or_fail_cmd`'s docstring for why this is safe (no gold fix ever needs
    those paths) and necessary (the agent's worktree never has `test_patch` applied, so a
    same-file edit is captured against a baseline that no longer matches at grading time)."""
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
