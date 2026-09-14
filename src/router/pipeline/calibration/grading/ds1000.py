"""Grader for DS-1000 — self-contained: exec(code_context) to load its own `test_execution` (and
`test_string`, when present) functions, then call them with the candidate solution string. No repo
checkout, no Docker. Matplotlib-library rows (PNG comparison) are excluded from task selection
entirely, not handled here. Same ImportError/ModuleNotFoundError -> `error_missing_dep`
simplification as bigcodebench.py.
"""
from __future__ import annotations

from .base import GradeResult, NONCE_PLACEHOLDER, Task, run_graded_script

# "RESULT_NONCE_PLACEHOLDER:" below is substituted for a real per-call nonce by run_graded_script
# before this ever runs.
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
    print(f"RESULT_NONCE_PLACEHOLDER: ERROR_MISSING_DEP {type(e).__name__}: {e}")
    sys.exit(0)
except AssertionError as e:
    print(f"RESULT_NONCE_PLACEHOLDER: FAIL assertion: {e}")
    sys.exit(0)
except Exception as e:
    print(f"RESULT_NONCE_PLACEHOLDER: FAIL {type(e).__name__}: {e}")
    sys.exit(0)

print("RESULT_NONCE_PLACEHOLDER: PASS")
"""
assert NONCE_PLACEHOLDER in _GRADE_SCRIPT, "sentinel placeholder text drifted out of sync with base.py"


def grade(task: Task, solution: str, timeout_seconds: int = 60) -> GradeResult:
    return run_graded_script(
        _GRADE_SCRIPT,
        timeout_seconds=timeout_seconds,
        extra_files={
            "code_context.py": task.row["code_context"].encode("utf-8"),
            "solution.txt": solution.encode("utf-8"),
        },
    )
