from router.pipeline.calibration.grading.base import run_graded_script


def test_pass_line_is_classified_as_pass():
    result = run_graded_script("print('RESULT: PASS')\n", timeout_seconds=10)
    assert result.outcome == "pass"


def test_fail_line_is_classified_as_fail_with_detail():
    result = run_graded_script("print('RESULT: FAIL something went wrong')\n", timeout_seconds=10)
    assert result.outcome == "fail"
    assert "something went wrong" in result.detail


def test_missing_dep_line_is_classified_separately_from_fail():
    result = run_graded_script("print('RESULT: ERROR_MISSING_DEP no numpy')\n", timeout_seconds=10)
    assert result.outcome == "error_missing_dep"


def test_a_crash_with_no_result_line_is_a_harness_error_not_a_fail():
    result = run_graded_script("raise RuntimeError('boom')\n", timeout_seconds=10)
    assert result.outcome == "error_harness"
    assert "boom" in result.detail


def test_exceeding_the_timeout_is_classified_separately_from_a_fail():
    result = run_graded_script("import time\ntime.sleep(5)\nprint('RESULT: PASS')\n", timeout_seconds=1)
    assert result.outcome == "error_timeout"


def test_extra_files_are_readable_by_the_script():
    script = "print('RESULT: PASS ' + open('data.txt').read())\n"
    result = run_graded_script(script, timeout_seconds=10, extra_files={"data.txt": b"hello"})
    assert result.outcome == "pass"
    assert "hello" in result.detail


def test_only_the_result_line_is_used_even_with_other_output():
    script = "print('some noise')\nprint('RESULT: PASS')\nprint('more noise')\n"
    result = run_graded_script(script, timeout_seconds=10)
    assert result.outcome == "pass"
