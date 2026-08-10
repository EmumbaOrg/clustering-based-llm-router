"""Shared grading infrastructure: the Outcome taxonomy and a subprocess+timeout+temp-cwd runner
every grader builds on.

A task that fails because OUR sandbox lacks a library, or times out on our slow CPU, is NOT the
model failing — conflating the two would corrupt every error rate the router later trusts. So
grading scripts built by each grader module print a single `RESULT: <TAG> [detail]` line to stdout
as their last action, and this module classifies that line into an Outcome rather than trusting a
bare exit code — which can't tell "candidate code raised an exception" (a real FAIL, since a
broken solution crashing IS what a wrong answer looks like) apart from "our harness couldn't even
start" (an environment problem, excluded from error rates).

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


_RESULT_LINE = re.compile(r"^RESULT:\s*(PASS|FAIL|ERROR_MISSING_DEP)\b(.*)$")


def run_graded_script(script: str, timeout_seconds: int, extra_files: dict[str, bytes] | None = None) -> GradeResult:
    """`extra_files`, if given, is written into the same temp directory before the script runs
    (e.g. a reference image for a comparison-based grader)."""
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
            match = _RESULT_LINE.match(line.strip())
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
