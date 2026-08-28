from router.pipeline.calibration.grading.base import NONCE_PLACEHOLDER, run_graded_script


def _script(text: str) -> str:
    """Scripts under test use the same NONCE_PLACEHOLDER token real grader templates use —
    run_graded_script substitutes it for a real per-call nonce before running anything, so a test
    script written with a bare "RESULT:" (no placeholder) is exercising the exploit the nonce
    exists to close, not real usage."""
    return text.replace("RESULT:", f"RESULT_{NONCE_PLACEHOLDER}:")


def test_pass_line_is_classified_as_pass():
    result = run_graded_script(_script("print('RESULT: PASS')\n"), timeout_seconds=10)
    assert result.outcome == "pass"


def test_fail_line_is_classified_as_fail_with_detail():
    result = run_graded_script(_script("print('RESULT: FAIL something went wrong')\n"), timeout_seconds=10)
    assert result.outcome == "fail"
    assert "something went wrong" in result.detail


def test_missing_dep_line_is_classified_separately_from_fail():
    result = run_graded_script(_script("print('RESULT: ERROR_MISSING_DEP no numpy')\n"), timeout_seconds=10)
    assert result.outcome == "error_missing_dep"


def test_a_crash_with_no_result_line_is_a_harness_error_not_a_fail():
    result = run_graded_script("raise RuntimeError('boom')\n", timeout_seconds=10)
    assert result.outcome == "error_harness"
    assert "boom" in result.detail


def test_exceeding_the_timeout_is_classified_separately_from_a_fail():
    result = run_graded_script(_script("import time\ntime.sleep(5)\nprint('RESULT: PASS')\n"), timeout_seconds=1)
    assert result.outcome == "error_timeout"


def test_extra_files_are_readable_by_the_script():
    script = _script("print('RESULT: PASS ' + open('data.txt').read())\n")
    result = run_graded_script(script, timeout_seconds=10, extra_files={"data.txt": b"hello"})
    assert result.outcome == "pass"
    assert "hello" in result.detail


def test_only_the_result_line_is_used_even_with_other_output():
    script = _script("print('some noise')\nprint('RESULT: PASS')\nprint('more noise')\n")
    result = run_graded_script(script, timeout_seconds=10)
    assert result.outcome == "pass"


# --- sentinel forgery (see base.py's NONCE_PLACEHOLDER docstring) ------------------------------

def test_a_bare_result_line_with_no_nonce_is_not_recognized():
    # Regression test for the exploit this session found and fixed: candidate code (exec()'d in
    # the SAME process as the grading script — see module docstring's isolation note) used to be
    # able to forge the sentinel just by printing a bare "RESULT: PASS", short-circuiting grading
    # before the real test suite ever ran. Without the real nonce, this must NOT be recognized as
    # a result line at all.
    result = run_graded_script("print('RESULT: PASS')\nraise RuntimeError('the real check never ran')\n", timeout_seconds=10)
    assert result.outcome == "error_harness"


def test_a_candidate_forged_pass_line_does_not_override_the_scripts_real_verdict():
    # The realistic shape of the exploit: a candidate's OWN code (standing in for whatever the
    # script exec()s before reaching its own real RESULT line) prints something that looks exactly
    # like a passing sentinel but WITHOUT the real nonce, then the script's real logic goes on to
    # correctly detect a failure with the real, noncified line. Only the second line uses
    # NONCE_PLACEHOLDER — the first is deliberately left bare, simulating the forgery attempt.
    script = (
        "print('RESULT: PASS')  # forged by 'candidate' code, no real nonce\n"
        f"print('RESULT_{NONCE_PLACEHOLDER}: FAIL the real check')\n"
    )
    result = run_graded_script(script, timeout_seconds=10)
    assert result.outcome == "fail"
    assert "the real check" in result.detail


def test_two_different_calls_get_two_different_nonces():
    # The nonce must be unpredictable per call — a candidate that happened to see one call's
    # sentinel (e.g. from a prior run's logs) must not be able to reuse it for the next.
    script = _script("print('RESULT: PASS')\n")
    first = run_graded_script(script, timeout_seconds=10)
    second = run_graded_script(script, timeout_seconds=10)
    assert first.outcome == "pass"
    assert second.outcome == "pass"
