"""Grader for BigCodeBench-Instruct — self-contained: exec(code_prompt + solution) then exec(test)
into the same namespace, then run `unittest.TestCases`. No repo checkout, no Docker.

Solution contract: a fragment appended directly after `code_prompt` (an indented function body),
not a complete standalone script — an empty/missing solution is a SyntaxError, graded `fail`.

Any ImportError/ModuleNotFoundError anywhere during grading is classified `error_missing_dep`
rather than `fail` — including one raised mid-test and caught inside `unittest`'s own result
object, not just one raised directly by our harness setup.
"""
from __future__ import annotations

from .base import GradeResult, NONCE_PLACEHOLDER, Task, run_graded_script

# "RESULT_NONCE_PLACEHOLDER:" below is substituted for a real per-call nonce by run_graded_script
# before this ever runs.
_GRADE_SCRIPT = r"""
import sys
import unittest

ns = {}
try:
    with open("candidate.py", encoding="utf-8") as f:
        exec(compile(f.read(), "candidate.py", "exec"), ns)
    with open("test.py", encoding="utf-8") as f:
        exec(compile(f.read(), "test.py", "exec"), ns)
    suite = unittest.TestLoader().loadTestsFromTestCase(ns["TestCases"])
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=0).run(suite)
except (ImportError, ModuleNotFoundError) as e:
    print(f"RESULT_NONCE_PLACEHOLDER: ERROR_MISSING_DEP {type(e).__name__}: {e}")
    sys.exit(0)
except Exception as e:
    print(f"RESULT_NONCE_PLACEHOLDER: FAIL setup-exception {type(e).__name__}: {e}")
    sys.exit(0)

if result.wasSuccessful():
    print("RESULT_NONCE_PLACEHOLDER: PASS")
else:
    tracebacks = "\n".join(tb for _, tb in (result.failures + result.errors))
    if "ModuleNotFoundError" in tracebacks or "ImportError" in tracebacks:
        print("RESULT_NONCE_PLACEHOLDER: ERROR_MISSING_DEP raised during test execution (see tracebacks)")
    else:
        print(f"RESULT_NONCE_PLACEHOLDER: FAIL {len(result.failures)} failures, {len(result.errors)} errors")
"""
assert NONCE_PLACEHOLDER in _GRADE_SCRIPT, "sentinel placeholder text drifted out of sync with base.py"


def grade(task: Task, solution: str, timeout_seconds: int = 60) -> GradeResult:
    candidate_src = task.row["code_prompt"] + solution
    test_src = task.row["test"]
    return run_graded_script(
        _GRADE_SCRIPT,
        timeout_seconds=timeout_seconds,
        extra_files={
            "candidate.py": candidate_src.encode("utf-8"),
            "test.py": test_src.encode("utf-8"),
        },
    )
