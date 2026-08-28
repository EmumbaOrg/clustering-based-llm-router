"""Grader for BigCodeBench-Instruct — self-contained: exec(code_prompt + solution) then exec(test)
into the SAME namespace (row 1's test calls `random.seed(42)` without importing `random` itself,
relying on the solution module's globals being shared), then run `unittest.TestCases`. No repo
checkout, no Docker.

Solution contract: a fragment to append directly after `code_prompt` (an indented function body,
matching `canonical_solution`'s own shape) — NOT a complete standalone script. `code_prompt` ends
mid-signature (`def task_func(...):\n`), so an empty/missing solution is a SyntaxError, which is
graded `fail`, not a harness error — a candidate that produced nothing failed the task.

Known simplification: any ImportError/ModuleNotFoundError anywhere during grading (harness setup
OR candidate code) is classified error_missing_dep rather than fail. Models overwhelmingly import
real, common libraries for these tasks (numpy/pandas/etc.); a genuinely nonexistent module is a
much stronger signal of an incomplete grading environment than of a deliberate model mistake. See
pipeline-python/README.md for the accepted limitations of this pass.

Implementation note this simplification depends on: `unittest.TextTestRunner` catches exceptions
raised *while a test runs* internally and records them in `result.errors` — it does NOT let them
propagate to an outer try/except. A candidate whose function body only imports a missing package
when actually CALLED (the common case — the import executes during the test's call, not while
merely defining the function) would otherwise be misclassified as `fail`. So after a failed run,
the tracebacks in `result.errors`/`result.failures` are inspected for the missing-dependency
signature too, not just exceptions raised directly by our own harness setup.
"""
from __future__ import annotations

from .base import GradeResult, NONCE_PLACEHOLDER, Task, run_graded_script

# Every "RESULT_NONCE_PLACEHOLDER:" below has that literal text substituted for a real per-call
# nonce by run_graded_script before this ever runs — see base.py's NONCE_PLACEHOLDER docstring for
# why a bare "RESULT:" sentinel is forgeable by the candidate code this script itself exec()s.
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
