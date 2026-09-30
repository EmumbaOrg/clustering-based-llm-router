"""Standalone BigCodeBench-Instruct loader + grader — companion to TUTORIAL.md.

Self-contained: the only imports below are third-party (`datasets`) and the standard library.
Run with:

    python tutorials/bigcodebench/quickstart.py --n-tasks 5

Requires: pip install datasets
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from datasets import load_dataset

# --- 1. Dataset loading (see TUTORIAL.md section 1 for the schema) -------------------------

HF_DATASET_ID = "bigcode/bigcodebench"
HF_SPLIT = "v0.1.4"  # the unversioned default split MERGES 5 historical versions (5,700 rows) —
# pinning v0.1.4 (1,140 rows) is deliberate, not incidental.


def load_tasks() -> list[dict]:
    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
    return [dict(row) for row in ds]


# --- 2. Custom grading harness, written inline (see TUTORIAL.md section 2) -----------------
#
# A solution here is a FRAGMENT — the indented function body meant to be appended directly after
# `code_prompt`, which ends mid-signature (`def task_func(...):\n`). A complete standalone script
# would be invalid once concatenated. An empty/missing solution is therefore a SyntaxError, which
# this grader classifies as "fail" (a candidate that produced nothing failed the task), not a
# harness error.

_GRADE_SCRIPT_TEMPLATE = '''
import sys
import unittest

ns = {{}}
try:
    with open("candidate.py", encoding="utf-8") as f:
        exec(compile(f.read(), "candidate.py", "exec"), ns)
    with open("test.py", encoding="utf-8") as f:
        exec(compile(f.read(), "test.py", "exec"), ns)
    suite = unittest.TestLoader().loadTestsFromTestCase(ns["TestCases"])
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=0).run(suite)
except (ImportError, ModuleNotFoundError) as e:
    print(f"RESULT_{nonce}: ERROR_MISSING_DEP {{type(e).__name__}}: {{e}}")
    sys.exit(0)
except Exception as e:
    print(f"RESULT_{nonce}: FAIL setup-exception {{type(e).__name__}}: {{e}}")
    sys.exit(0)

if result.wasSuccessful():
    print("RESULT_{nonce}: PASS")
else:
    tracebacks = "\\n".join(tb for _, tb in (result.failures + result.errors))
    if "ModuleNotFoundError" in tracebacks or "ImportError" in tracebacks:
        # A candidate whose function body only imports a missing package when actually CALLED (the
        # common case) raises DURING the test run, not while the function is merely defined —
        # unittest.TextTestRunner catches that internally and records it in result.errors rather
        # than letting it propagate to our own try/except above, so we inspect tracebacks too.
        print("RESULT_{nonce}: ERROR_MISSING_DEP raised during test execution (see tracebacks)")
    else:
        print(f"RESULT_{nonce}: FAIL {{len(result.failures)}} failures, {{len(result.errors)}} errors")
'''


def grade(row: dict, solution: str, timeout_seconds: int = 60) -> tuple[str, str]:
    """Returns (outcome, detail). outcome is one of "pass" / "fail" / "error_missing_dep" /
    "error_timeout" / "error_harness"."""
    nonce = uuid.uuid4().hex  # fresh per call — a candidate that merely PRINTS a fixed sentinel
    # string could otherwise forge a "PASS" result before the real test suite ever runs.
    script = _GRADE_SCRIPT_TEMPLATE.format(nonce=nonce)
    result_prefix = f"RESULT_{nonce}:"

    candidate_src = row["code_prompt"] + solution
    test_src = row["test"]

    with tempfile.TemporaryDirectory(prefix="bigcodebench-tutorial-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "grade.py").write_text(script, encoding="utf-8")
        (tmp_path / "candidate.py").write_text(candidate_src, encoding="utf-8")
        (tmp_path / "test.py").write_text(test_src, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(tmp_path / "grade.py")],
                cwd=tmp, capture_output=True, text=True, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return "error_timeout", f"exceeded {timeout_seconds}s"

        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith(result_prefix):
                continue
            rest = line[len(result_prefix):].strip()
            if rest.startswith("PASS"):
                return "pass", rest[4:].strip()
            if rest.startswith("FAIL"):
                return "fail", rest[4:].strip()
            if rest.startswith("ERROR_MISSING_DEP"):
                return "error_missing_dep", rest[len("ERROR_MISSING_DEP"):].strip()

        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        return "error_harness", f"no RESULT line (exit {proc.returncode}); stderr tail:\n{stderr_tail}"


# --- 3. Evaluation: custom-harness controls + optional official-harness cross-check --------


def run_custom_harness(tasks: list[dict]) -> None:
    print("\n--- custom harness (this script's own grade()) ---")
    for row in tasks:
        reference_outcome, _ = grade(row, row["canonical_solution"])
        null_outcome, _ = grade(row, "")
        print(f"{row['task_id']}: reference -> {reference_outcome}  null -> {null_outcome}")
        assert reference_outcome == "pass", (
            f"grader rejected the dataset's own gold solution for {row['task_id']} — "
            "the grader itself is broken, not the (nonexistent) model"
        )
        assert null_outcome != "pass", (
            f"grader accepted an EMPTY solution for {row['task_id']} — the grader is broken"
        )


def run_official_harness_note(tasks: list[dict]) -> None:
    print("\n--- official harness cross-check (pip install bigcodebench) ---")
    try:
        import bigcodebench  # noqa: F401
    except ImportError:
        print(
            "the `bigcodebench` package isn't installed — skipping. See TUTORIAL.md section 3 for "
            "why this needs a Docker-based run, not a bare pip install, to be a meaningful check."
        )
        return
    print(
        "`bigcodebench` is importable, but driving its generate/evaluate flow means writing a "
        "jsonl of {task_id, solution} pairs (a COMPLETE function per row, unlike this script's "
        "fragment convention above) and handing it to its own CLI against its own Docker image. "
        "See TUTORIAL.md section 3 for the full walkthrough and why exact flags aren't asserted here."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_tasks = load_tasks()
    print(f"loaded {len(all_tasks)} total BigCodeBench-Instruct tasks ({HF_SPLIT} split)")
    tasks = random.Random(args.seed).sample(all_tasks, min(args.n_tasks, len(all_tasks)))

    run_custom_harness(tasks)
    run_official_harness_note(tasks)


if __name__ == "__main__":
    main()
