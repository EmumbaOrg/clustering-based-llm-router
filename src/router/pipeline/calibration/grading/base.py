"""Shared grading infrastructure: the Outcome taxonomy and a subprocess+timeout+temp-cwd runner
every self-contained grader builds on. Grading scripts print a single `RESULT_<nonce>: <TAG>
[detail]` line as their last action; this module classifies that line rather than trusting a bare
exit code, so an environment problem (missing library, timeout) is never miscounted as the model
being wrong.
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


# Grading script templates write their sentinel as `RESULT_NONCE_PLACEHOLDER:` — `run_graded_script`
# substitutes it with a fresh `uuid4().hex` per call, so a fixed sentinel can't be forged by
# anything the exec()'d candidate code happens to print.
NONCE_PLACEHOLDER = "NONCE_PLACEHOLDER"


def _result_line_pattern(nonce: str) -> re.Pattern[str]:
    return re.compile(rf"^RESULT_{nonce}:\s*(PASS|FAIL|ERROR_MISSING_DEP)\b(.*)$")


def run_graded_script(script: str, timeout_seconds: int, extra_files: dict[str, bytes] | None = None) -> GradeResult:
    """`extra_files`, if given, is written into the same temp directory before the script runs.
    `script` must use `NONCE_PLACEHOLDER` for its sentinel, substituted here (plain string
    replacement, not `.format()`) into a real per-call nonce before running."""
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
