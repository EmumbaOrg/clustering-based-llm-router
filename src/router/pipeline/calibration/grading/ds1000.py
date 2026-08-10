"""Grader for DS-1000 — self-contained: exec(code_context) to load its own `test_execution` (and
`test_string`, when present) functions, then call them with the candidate solution string. No
repo checkout, no Docker.

Both grading functions run when present — `test_string` (159 rows) asserts a required API
appears in the solution text; skipping it lets a hardcoded answer pass `test_execution` alone.

Solution contract: the snippet to splice into `code_context`'s `[insert]` placeholder — the
splicing itself happens inside `test_execution`, defined by the row's own `code_context`, so this
grader never needs to know the placeholder convention itself.

Matplotlib-library rows are excluded from task selection entirely (see calibrate.py), not handled
here — their `exec_test` compares rendered PNGs against a reference image that would first need
to be rendered from `reference_code`, which is more machinery than this pass's scope covers. See
pipeline-python/README.md.

Known simplification shared with bigcodebench.py: any ImportError/ModuleNotFoundError anywhere
during grading is classified error_missing_dep rather than fail.
"""
from __future__ import annotations

from .base import GradeResult, Task, run_graded_script

_GRADE_SCRIPT = r"""
import sys

ns = {}
try:
    with open("code_context.py", encoding="utf-8") as f:
        exec(compile(f.read(), "code_context.py", "exec"), ns)
    with open("solution.txt", encoding="utf-8") as f:
        solution = f.read()

    ns["test_execution"](solution)
    if "test_string" in ns:
        ns["test_string"](solution)
except (ImportError, ModuleNotFoundError) as e:
    print(f"RESULT: ERROR_MISSING_DEP {type(e).__name__}: {e}")
    sys.exit(0)
except AssertionError as e:
    print(f"RESULT: FAIL assertion: {e}")
    sys.exit(0)
except Exception as e:
    print(f"RESULT: FAIL {type(e).__name__}: {e}")
    sys.exit(0)

print("RESULT: PASS")
"""


def grade(task: Task, solution: str, timeout_seconds: int = 60) -> GradeResult:
    return run_graded_script(
        _GRADE_SCRIPT,
        timeout_seconds=timeout_seconds,
        extra_files={
            "code_context.py": task.row["code_context"].encode("utf-8"),
            "solution.txt": solution.encode("utf-8"),
        },
    )
