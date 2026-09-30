"""Standalone DS-1000 loader + grader — companion to TUTORIAL.md.

Self-contained: only third-party (`datasets`) and standard library imports below. Run with:

    python tutorials/ds1000/quickstart.py --n-tasks 5

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

HF_DATASET_ID = "xlangai/DS-1000"
HF_SPLIT = "test"


def load_tasks() -> list[dict]:
    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
    tasks = []
    for row in ds:
        # Matplotlib rows grade by comparing a RENDERED PNG against a reference image, which needs
        # a rendering step this tutorial doesn't implement — excluded rather than half-implemented.
        if row["metadata"]["library"] == "Matplotlib":
            continue
        tasks.append(dict(row))
    return tasks


# --- 2. Custom grading harness, written inline (see TUTORIAL.md section 2) -----------------
#
# Unlike BigCodeBench, DS-1000 rows carry their OWN test harness inline: `code_context` is a
# Python source string defining a `test_execution(solution)` function (every row) and, on 159/1000
# rows, a `test_string(solution)` function too. Grading a candidate means exec()-ing `code_context`
# to load those functions, then calling them with the candidate's solution string.

_GRADE_SCRIPT_TEMPLATE = '''
import sys

ns = {{}}
try:
    with open("code_context.py", encoding="utf-8") as f:
        exec(compile(f.read(), "code_context.py", "exec"), ns)
    with open("solution.txt", encoding="utf-8") as f:
        solution = f.read()
    ns["test_execution"](solution)
    if "test_string" in ns:
        ns["test_string"](solution)
except (ImportError, ModuleNotFoundError) as e:
    print(f"RESULT_{nonce}: ERROR_MISSING_DEP {{type(e).__name__}}: {{e}}")
    sys.exit(0)
except AssertionError as e:
    print(f"RESULT_{nonce}: FAIL assertion: {{e}}")
    sys.exit(0)
except Exception as e:
    print(f"RESULT_{nonce}: FAIL {{type(e).__name__}}: {{e}}")
    sys.exit(0)

print("RESULT_{nonce}: PASS")
'''


def grade(row: dict, solution: str, timeout_seconds: int = 60) -> tuple[str, str]:
    """Returns (outcome, detail): "pass" / "fail" / "error_missing_dep" / "error_timeout" /
    "error_harness"."""
    nonce = uuid.uuid4().hex
    script = _GRADE_SCRIPT_TEMPLATE.format(nonce=nonce)
    result_prefix = f"RESULT_{nonce}:"

    with tempfile.TemporaryDirectory(prefix="ds1000-tutorial-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "grade.py").write_text(script, encoding="utf-8")
        (tmp_path / "code_context.py").write_text(row["code_context"], encoding="utf-8")
        (tmp_path / "solution.txt").write_text(solution, encoding="utf-8")

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


# --- 3. Evaluation: custom-harness controls + official-convention cross-check --------------


def run_custom_harness(tasks: list[dict]) -> None:
    print("\n--- custom harness (this script's own grade()) ---")
    for row in tasks:
        problem_id = row["metadata"]["problem_id"]
        reference_outcome, _ = grade(row, row["reference_code"])
        wrong_outcome, _ = grade(row, "None")
        print(f"ds1000:{problem_id}: reference -> {reference_outcome}  wrong -> {wrong_outcome}")
        assert reference_outcome == "pass", (
            f"grader rejected the dataset's own gold solution for ds1000:{problem_id} — "
            "the grader itself is broken"
        )


def run_official_convention_note() -> None:
    print("\n--- 'official harness' note ---")
    print(
        "DS-1000 has no separate official eval PACKAGE the way BigCodeBench does — the dataset's "
        "own code_context field IS the official grading convention this script just reimplemented "
        "(test_execution/test_string). See TUTORIAL.md section 3 for how to cross-check at scale "
        "instead: run the reference solution across a larger sample and confirm it passes at the "
        "expected rate, rather than diffing against a separate tool's verdict."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_tasks = load_tasks()
    print(f"loaded {len(all_tasks)} non-Matplotlib DS-1000 tasks ({HF_SPLIT} split)")
    tasks = random.Random(args.seed).sample(all_tasks, min(args.n_tasks, len(all_tasks)))

    run_custom_harness(tasks)
    run_official_convention_note()


if __name__ == "__main__":
    main()
