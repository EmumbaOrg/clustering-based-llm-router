"""Shared grading infrastructure: the Outcome taxonomy and a subprocess+timeout+temp-cwd runner
every grader builds on.

A task that fails because OUR sandbox lacks a library, or times out on our slow CPU, is NOT the
model failing — conflating the two would corrupt every error rate the router later trusts. So
grading scripts built by each grader module print a single `RESULT_<nonce>: <TAG> [detail]` line to
stdout as their last action, and this module classifies that line into an Outcome rather than
trusting a bare exit code — which can't tell "candidate code raised an exception" (a real FAIL,
since a broken solution crashing IS what a wrong answer looks like) apart from "our harness
couldn't even start" (an environment problem, excluded from error rates). The nonce (see
`NONCE_PLACEHOLDER`) is generated fresh per call, same reasoning as `dockerexec.py`'s own nonce: the
candidate code this classifies is `exec()`'d in the same process, so a fixed sentinel would be
forgeable by anything the candidate happens to print.

Isolation note: this runs candidate-generated code via `exec()` in a subprocess with a timeout and
a throwaway temp cwd — the same approach the benchmarks' own reference harnesses use. That's
process isolation, not a security sandbox; see the pipeline README for when Docker is warranted
instead.
"""
from __future__ import annotations

import dataclasses
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Literal

Outcome = Literal[
    "pass",
    "fail",
    "error_missing_dep",
    "error_timeout",
    "error_harness",
    "error_no_solution",
]

# pass/fail/error_no_solution feed error-rate calculations — a model that produced nothing is a
# real failure. error_missing_dep/error_timeout/error_harness are OUR environment's fault and are
# tallied separately (artifacts-schema/model-profiles.schema.json's `excluded` block) rather than
# counted as a model failure.
GRADED_OUTCOMES: frozenset[str] = frozenset({"pass", "fail", "error_no_solution"})
EXCLUDED_OUTCOMES: frozenset[str] = frozenset({"error_missing_dep", "error_timeout", "error_harness"})


@dataclasses.dataclass(frozen=True)
class Task:
    task_id: str  # stable, dataset-native id (e.g. "BigCodeBench/0", "ds1000:42")
    source: str  # "bigcodebench" | "ds1000" | "swe-smith" | "swe-gym"
    prompt: str  # natural-language instruction — what gets embedded and shown to the agent
    reference_solution: str  # gold solution, for the `reference` runner control
    row: dict  # raw HF row, for whatever source-specific fields a grader needs


@dataclasses.dataclass(frozen=True)
class GradeResult:
    outcome: Outcome
    detail: str = ""


# Every grading script template (bigcodebench.py, ds1000.py) writes its sentinel as
# `RESULT_NONCE_PLACEHOLDER:` (this literal text) — `run_graded_script` substitutes it with a fresh
# `uuid4().hex` per call before the script ever runs, exactly mirroring why `dockerexec.py`'s
# Docker-based graders use a per-call nonce: candidate code here runs via `exec()` in the SAME
# process as the grading script (see module docstring's isolation note), so it's just as capable of
# printing arbitrary text as an arbitrary Docker container's candidate patch is. Confirmed
# empirically this session that the OLD fixed `RESULT:` prefix was exploitable — a candidate
# solution that merely contains `print("RESULT: PASS")` short-circuited the whole grading run to
# `pass` before the real test suite ever ran, because the old code returned on the FIRST matching
# line rather than requiring anything unforgeable. A nonce generated fresh per call and unknown to
# the candidate ahead of time closes that off the same way it already does for Docker grading.
NONCE_PLACEHOLDER = "NONCE_PLACEHOLDER"


def _result_line_pattern(nonce: str) -> re.Pattern[str]:
    return re.compile(rf"^RESULT_{nonce}:\s*(PASS|FAIL|ERROR_MISSING_DEP)\b(.*)$")


def run_graded_script(script: str, timeout_seconds: int, extra_files: dict[str, bytes] | None = None) -> GradeResult:
    """`extra_files`, if given, is written into the same temp directory before the script runs
    (e.g. a reference image for a comparison-based grader).

    `script` must use `NONCE_PLACEHOLDER` (this module's constant) everywhere its `RESULT:` sentinel
    would otherwise go bare — substituted here via plain string replacement (not `.format()`, so it
    can't collide with the script's own f-string braces) into a real per-call nonce before the
    script is written to disk."""
    nonce = uuid.uuid4().hex
    script = script.replace(NONCE_PLACEHOLDER, nonce)
    result_line = _result_line_pattern(nonce)

    with tempfile.TemporaryDirectory(prefix="router-grade-") as tmp:
        tmp_path = Path(tmp)
        script_path = tmp_path / "grade.py"
        script_path.write_text(script, encoding="utf-8")
        for name, content in (extra_files or {}).items():
            (tmp_path / name).write_bytes(content)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return GradeResult(outcome="error_timeout", detail=f"exceeded {timeout_seconds}s")

        for line in proc.stdout.splitlines():
            match = result_line.match(line.strip())
            if not match:
                continue
            tag, detail = match.group(1), match.group(2).strip()
            if tag == "PASS":
                return GradeResult(outcome="pass", detail=detail)
            if tag == "FAIL":
                return GradeResult(outcome="fail", detail=detail)
            if tag == "ERROR_MISSING_DEP":
                return GradeResult(outcome="error_missing_dep", detail=detail)

        # No RESULT line found anywhere in stdout — the script crashed before it could report one
        # (a bug in the grading script itself, not a candidate-code failure, which unittest/our own
        # try-except would have already turned into a FAIL line).
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        return GradeResult(
            outcome="error_harness",
            detail=f"no RESULT line (exit {proc.returncode}); stderr tail:\n{stderr_tail}",
        )
