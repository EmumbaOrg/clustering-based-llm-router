from router.pipeline.calibration.grading.base import Task
from router.pipeline.calibration.grading.bigcodebench import grade

_ROW = {
    "code_prompt": "def add(a, b):\n",
    "test": (
        "import unittest\n\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n"
    ),
}


def _task(row: dict = _ROW) -> Task:
    return Task(task_id="test:0", source="bigcodebench", prompt="add two numbers", reference_solution="    return a + b\n", row=row)


def test_correct_solution_passes():
    result = grade(_task(), "    return a + b\n")
    assert result.outcome == "pass"


def test_wrong_solution_fails():
    result = grade(_task(), "    return a - b\n")
    assert result.outcome == "fail"


def test_empty_solution_is_a_syntax_error_and_fails_not_errors():
    result = grade(_task(), "")
    assert result.outcome == "fail"


def test_missing_dependency_is_classified_separately_from_a_wrong_answer():
    solution = "    import totally_fake_nonexistent_package_xyz\n    return a + b\n"
    result = grade(_task(), solution)
    assert result.outcome == "error_missing_dep"
