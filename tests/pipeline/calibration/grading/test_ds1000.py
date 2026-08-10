from router.pipeline.calibration.grading.base import Task
from router.pipeline.calibration.grading.ds1000 import grade

_CODE_CONTEXT = (
    "def test_execution(solution):\n"
    "    test_env = {}\n"
    "    exec('result = ' + solution, test_env)\n"
    "    assert test_env['result'] == 5, f\"got {test_env['result']}\"\n"
    "\n"
    "def test_string(solution):\n"
    "    assert '+' in solution, 'must use addition'\n"
)

_CODE_CONTEXT_MISSING_DEP = "import totally_fake_nonexistent_package_xyz\n" + _CODE_CONTEXT


def _task(code_context: str = _CODE_CONTEXT) -> Task:
    return Task(task_id="ds1000:0", source="ds1000", prompt="compute 5", reference_solution="2 + 3", row={"code_context": code_context})


def test_correct_solution_passes_both_test_execution_and_test_string():
    result = grade(_task(), "2 + 3")
    assert result.outcome == "pass"


def test_wrong_value_fails():
    result = grade(_task(), "2 + 2")
    assert result.outcome == "fail"


def test_test_string_is_invoked_when_present():
    # Correct numeric result (5) but doesn't use '+' -> test_execution passes, test_string fails.
    # If test_string weren't invoked, this would incorrectly pass.
    result = grade(_task(), "10 - 5")
    assert result.outcome == "fail"


def test_empty_solution_fails_rather_than_erroring():
    result = grade(_task(), "")
    assert result.outcome == "fail"


def test_missing_dependency_in_the_harness_is_classified_separately():
    result = grade(_task(_CODE_CONTEXT_MISSING_DEP), "2 + 3")
    assert result.outcome == "error_missing_dep"


def test_reference_solution_from_the_task_itself_passes():
    task = _task()
    result = grade(task, task.reference_solution)
    assert result.outcome == "pass"
