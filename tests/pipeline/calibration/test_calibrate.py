from router.common.config import CalibrationConfig, ModelConfig, SmoothingConfig
from router.pipeline.calibration import calibrate as calibrate_module
from router.pipeline.calibration.calibrate import (
    _expected_solution,
    _preview,
    _stats_from_outcomes,
    run_and_grade,
)
from router.pipeline.calibration.grading.base import GradeResult, Task
from router.pipeline.calibration.runner import RunResult


def test_global_scope_smoothed_rate_equals_raw_rate():
    # No external global rate supplied (we ARE the global scope) -> shrinking toward itself is a
    # no-op algebraically; smoothed must equal raw exactly, not merely approximately.
    stats = _stats_from_outcomes(["pass", "pass", "fail", "fail"], global_raw_error_rate=None, prior_weight=5)
    assert stats.raw_error_rate == 0.5
    assert stats.smoothed_error_rate == stats.raw_error_rate


def test_a_thin_cluster_is_pulled_toward_the_global_rate():
    # 0/1 failed (a lucky perfect score on a single task) should NOT read as a confident 0%.
    stats = _stats_from_outcomes(["pass"], global_raw_error_rate=0.4, prior_weight=5)
    assert stats.raw_error_rate == 0.0
    assert stats.smoothed_error_rate > 0.0
    assert stats.smoothed_error_rate < 0.4


def test_a_large_cluster_trusts_its_own_rate_more_than_a_thin_one():
    thin = _stats_from_outcomes(["pass"], global_raw_error_rate=0.4, prior_weight=5)
    large = _stats_from_outcomes(["pass"] * 100, global_raw_error_rate=0.4, prior_weight=5)
    assert large.smoothed_error_rate < thin.smoothed_error_rate


def test_excluded_outcomes_are_tallied_but_not_in_the_denominator():
    stats = _stats_from_outcomes(
        ["pass", "fail", "error_missing_dep", "error_timeout"], global_raw_error_rate=None, prior_weight=5,
    )
    assert stats.number_of_tasks == 2  # only pass/fail counted
    assert stats.excluded == {"error_missing_dep": 1, "error_timeout": 1}


def test_error_no_solution_counts_as_a_real_failure_not_excluded():
    stats = _stats_from_outcomes(["pass", "error_no_solution"], global_raw_error_rate=None, prior_weight=5)
    assert stats.number_of_tasks == 2
    assert stats.number_failed == 1
    assert stats.excluded == {}


def test_empty_outcomes_list_does_not_divide_by_zero():
    stats = _stats_from_outcomes([], global_raw_error_rate=None, prior_weight=5)
    assert stats.number_of_tasks == 0
    assert stats.raw_error_rate == 0.0


def _pi_model() -> ModelConfig:
    return ModelConfig(
        model_id="llama-3.1-8b-instant", provider="groq", runner="pi",
        cost_input=0.00005, cost_output=0.00008, context_window=131072, max_tokens=131072, rate_limit_rpm=30,
    )


def _task() -> Task:
    return Task(task_id="bigcodebench:0", source="bigcodebench", prompt="do the thing", reference_solution="x", row={})


def _calibration_config() -> CalibrationConfig:
    return CalibrationConfig(
        gradeable_sources=["bigcodebench"], tasks_per_cluster=4, candidate_pool_oversample=4,
        task_timeout_seconds=60, smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        holdout_fraction=0.3, seed=42, lambda_sweep=[0, 0.1],
    )


def test_run_and_grade_maps_a_rate_limited_run_result_to_error_harness_not_error_no_solution(monkeypatch):
    # A 429 is an infra/quota rejection, not the model failing to answer — it must be excluded
    # from the model's error rate (error_harness), not counted as a wrong answer (error_no_solution).
    monkeypatch.setattr(
        calibrate_module.runner_mod, "run_pi",
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="429 Too Many Requests", rate_limited=True),
    )
    result = run_and_grade(_task(), _pi_model(), _calibration_config())
    assert result.outcome == "error_harness"


def test_run_and_grade_still_treats_a_genuine_no_solution_as_a_real_failure(monkeypatch):
    monkeypatch.setattr(
        calibrate_module.runner_mod, "run_pi",
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="empty response", rate_limited=False),
    )
    result = run_and_grade(_task(), _pi_model(), _calibration_config())
    assert result.outcome == "error_no_solution"


def test_expected_solution_falls_back_to_the_gold_patch_when_reference_solution_is_empty():
    # swe-smith/swe-gym leave reference_solution empty on purpose (see tasks.py) — the
    # review-worthy "expected" text for those sources is the gold patch instead.
    task = Task(task_id="t", source="swe-smith", prompt="p", reference_solution="", row={"patch": "diff --git a/x"})
    assert _expected_solution(task) == "diff --git a/x"


def test_expected_solution_prefers_reference_solution_when_present():
    task = Task(task_id="t", source="bigcodebench", prompt="p", reference_solution="return 1", row={"patch": "unused"})
    assert _expected_solution(task) == "return 1"


def test_preview_truncates_long_text_and_reports_how_much_was_cut():
    text = "x" * 2000
    preview = _preview(text)
    assert preview.startswith("x" * 100)
    assert preview.endswith("more chars]")
    assert len(preview) < len(text)


def test_preview_leaves_short_text_untouched():
    assert _preview("  a short answer  ") == "a short answer"


def test_run_and_grade_logs_expected_vs_provided_solution_at_debug(monkeypatch, caplog):
    monkeypatch.setattr(
        calibrate_module.runner_mod, "run_pi",
        lambda task, model, timeout_seconds: RunResult(solution="def f():\n    return 1", detail="", rate_limited=False),
    )
    monkeypatch.setitem(
        calibrate_module._GRADERS, "bigcodebench",
        lambda task, solution, timeout_seconds: GradeResult(outcome="fail", detail="assertion: expected 2, got 1"),
    )
    # logger=... (not just a bare level) matters here: test_logging_config.py exercises
    # configure_logging, which sets the "router" logger's propagate=False — a bare caplog.at_level
    # only listens at the root logger, so it would see nothing once that's run earlier in the
    # suite. Naming the logger attaches caplog's handler directly to it instead.
    with caplog.at_level("DEBUG", logger="router.pipeline.calibration.calibrate"):
        result = run_and_grade(_task(), _pi_model(), _calibration_config())

    assert result.outcome == "fail"
    assert "-> fail" in caplog.text
    assert "expected: x" in caplog.text  # _task()'s reference_solution
    assert "provided: def f():" in caplog.text
    assert "assertion: expected 2, got 1" in caplog.text


def test_run_and_grade_logs_no_solution_extracted_when_pi_returns_none(monkeypatch, caplog):
    monkeypatch.setattr(
        calibrate_module.runner_mod, "run_pi",
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="empty response", rate_limited=False),
    )
    with caplog.at_level("DEBUG", logger="router.pipeline.calibration.calibrate"):
        run_and_grade(_task(), _pi_model(), _calibration_config())
    assert "provided: (no solution extracted)" in caplog.text
