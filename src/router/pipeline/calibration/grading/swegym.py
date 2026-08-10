"""Grader for SWE-Gym — deliberately NOT implemented this pass.

Unlike SWE-smith (a prebuilt image per task), SWE-Gym rows give only `repo` + `base_commit` +
`version`, with NO `environment_setup_commit` — so building a working environment means resolving
each of 161 distinct version strings against SWE-bench's own per-repo environment-setup constants
(exact conda/pip commands, Python version, sometimes system packages), which is a research task,
not a porting task. Several rows (e.g. pandas) also carry 11,000+ PASS_TO_PASS tests, making even
a correctly-resolved environment expensive to run per task.

This is a stub with a clear failure, not a silent wrong answer — calling `grade()` raises rather
than returning a GradeResult, so a caller can't accidentally record a 0% (or 100%) rate for a
model that was never actually graded on SWE-Gym tasks. See pipeline-python/README.md and the
calibration plan doc for why this is deferred rather than attempted.
"""
from __future__ import annotations

from .base import Task


def grade(task: Task, solution: str, timeout_seconds: int = 60) -> None:
    raise NotImplementedError(
        f"SWE-Gym grading is not implemented (task {task.task_id}): environment resolution needs "
        "repo+version -> setup-commands mapping with no environment_setup_commit available, and "
        "some rows carry 11,000+ PASS_TO_PASS tests. See grading/swegym.py and the calibration "
        "plan doc for what this would take."
    )
