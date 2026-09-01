from router.common.config import CalibrationConfig, ModelConfig, SmoothingConfig
from router.pipeline.calibration import calibrate as calibrate_module
from router.pipeline.calibration.calibrate import (
    SelectedTask,
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


def _named_model(model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id, provider="groq", runner="pi",
        cost_input=0.00005, cost_output=0.00008, context_window=131072, max_tokens=131072, rate_limit_rpm=30,
    )


def _selected(task_id: str, cluster_id: int) -> SelectedTask:
    return SelectedTask(
        task=Task(task_id=task_id, source="bigcodebench", prompt="p", reference_solution="x", row={}),
        cluster_id=cluster_id,
        split="calibration",
    )


def _outcome_for(task_id: str, model_id: str) -> str:
    # Depends only on the (task, model) pair, never on call order — which is exactly what makes the
    # tasks-outer/models-inner reordering safe.
    return "pass" if (task_id, model_id) in {("t1", "m1"), ("t2", "m2"), ("t3", "m1")} else "fail"


def test_calibrate_models_grades_every_model_against_a_task_before_moving_to_the_next_task(monkeypatch, tmp_path):
    # The whole point of the loop order: swe-gym/multi-swe-rl build one multi-GB image PER INSTANCE,
    # so all models must be graded while a task's image is still hot. A models-outer loop re-pulls
    # every image once per model (~5x the bytes at the configured roster size).
    calls = []

    def fake_run_and_grade(task, model, calibration_config):
        calls.append((task.task_id, model.model_id))
        return GradeResult(outcome=_outcome_for(task.task_id, model.model_id)), "", None

    monkeypatch.setattr(calibrate_module, "run_and_grade", fake_run_and_grade)
    models = [_named_model("m1"), _named_model("m2")]
    selected = [_selected("t1", 0), _selected("t2", 0)]

    # details_csv_path MUST be redirected to tmp_path — the default writes to the real repo's
    # artifacts/calibration-details.csv, which would silently clobber a real calibration run's
    # output. Confirmed the hard way this session: running this suite overwrote a just-completed
    # real run's CSV with this test's "m1"/"m2" fixture data.
    calibrate_module.calibrate_models(models, selected, _calibration_config(), details_csv_path=tmp_path / "details.csv")

    assert calls == [("t1", "m1"), ("t1", "m2"), ("t2", "m1"), ("t2", "m2")]


def test_calibrate_models_persists_the_full_solution_when_the_csv_preview_would_truncate_it(monkeypatch, tmp_path):
    # Regression test: confirmed this session investigating Luna's Multi-SWE-RL failures that a
    # solution over the CSV preview cap (e.g. `checkstyle-15001`, 613,777 chars) is gone for good
    # once truncated — it only ever existed in memory for that one call, with no way to audit a
    # large apply failure after the fact without re-running a fresh container by hand.
    huge_solution = "x" * (calibrate_module._CSV_SOLUTION_MAX_CHARS + 500)
    monkeypatch.setattr(
        calibrate_module, "run_and_grade",
        lambda task, model, cfg: (GradeResult(outcome="fail"), huge_solution, None),
    )
    models = [_named_model("m1")]
    selected = [_selected("t1", 0)]
    details_path = tmp_path / "details.csv"

    _results, detail_rows = calibrate_module.calibrate_models(models, selected, _calibration_config(), details_csv_path=details_path)

    assert len(detail_rows[0].solution) < len(huge_solution)  # the CSV row itself still just has the preview
    persisted = tmp_path / "solutions" / "details" / "t1__m1.diff"
    assert persisted.read_text(encoding="utf-8") == huge_solution


def test_calibrate_models_does_not_persist_a_side_file_for_a_solution_that_already_fits(monkeypatch, tmp_path):
    # A short solution is already complete in the CSV — a side file for it would be pure
    # duplication, and every (task, model) pair in a real run would get one otherwise.
    monkeypatch.setattr(
        calibrate_module, "run_and_grade",
        lambda task, model, cfg: (GradeResult(outcome="pass"), "short diff", None),
    )
    models = [_named_model("m1")]
    selected = [_selected("t1", 0)]
    details_path = tmp_path / "details.csv"

    calibrate_module.calibrate_models(models, selected, _calibration_config(), details_csv_path=details_path)

    assert not (tmp_path / "solutions").exists()


def test_calibrate_models_stats_are_identical_to_the_old_models_outer_aggregation(monkeypatch, tmp_path):
    # Guards the "purely a reordering" claim: every statistic is computed after all grading, so
    # reordering the calls must not move a single number.
    monkeypatch.setattr(
        calibrate_module, "run_and_grade",
        lambda task, model, cfg: (GradeResult(outcome=_outcome_for(task.task_id, model.model_id)), "", None),
    )
    models = [_named_model("m1"), _named_model("m2")]
    selected = [_selected("t1", 0), _selected("t2", 1), _selected("t3", 1)]
    config = _calibration_config()

    # See the sibling test above for why details_csv_path must be redirected to tmp_path.
    results, _detail_rows = calibrate_module.calibrate_models(
        models, selected, config, details_csv_path=tmp_path / "details.csv"
    )

    # Rebuild each model's outcome list the OLD way (models-outer, original task order) and
    # aggregate that instead — the two must agree exactly.
    expected = [
        calibrate_module._aggregate_outcomes(
            model,
            [(st, GradeResult(outcome=_outcome_for(st.task.task_id, model.model_id))) for st in selected],
            config,
        )
        for model in models
    ]
    assert [(r.model.model_id, r.global_stats, r.cluster_stats) for r in results] == [
        (r.model.model_id, r.global_stats, r.cluster_stats) for r in expected
    ]


def test_image_affinity_key_groups_tasks_that_share_a_docker_image():
    # swe-smith's image_name is per bug-injected repo, shared by many instances — those should sort
    # adjacently so the second instance reuses the first's pulled image.
    def swesmith(task_id, image):
        return Task(task_id=task_id, source="swe-smith", prompt="p", reference_solution="", row={"image_name": image})

    tasks = [swesmith("b", "img-2"), swesmith("a", "img-1"), swesmith("c", "img-1")]
    ordered = sorted(tasks, key=calibrate_module.image_affinity_key)
    assert [t.task_id for t in ordered] == ["a", "c", "b"]  # both img-1 tasks before the img-2 one


def test_image_affinity_key_falls_back_to_repo_then_task_id():
    # swe-gym/multi-swe-rl have one image per instance, so `repo` is the useful grouping (it's also
    # what repo_context.py's bare-clone cache is keyed on).
    gym = Task(task_id="x", source="swe-gym", prompt="p", reference_solution="", row={"repo": "getmoto/moto"})
    assert calibrate_module.image_affinity_key(gym) == ("swe-gym", "getmoto/moto")
    bare = Task(task_id="ds1000:7", source="ds1000", prompt="p", reference_solution="", row={})
    assert calibrate_module.image_affinity_key(bare) == ("ds1000", "ds1000:7")


def test_run_and_grade_maps_a_rate_limited_run_result_to_error_harness_not_error_no_solution(monkeypatch):
    # A 429 is an infra/quota rejection, not the model failing to answer — it must be excluded
    # from the model's error rate (error_harness), not counted as a wrong answer (error_no_solution).
    monkeypatch.setattr(
        calibrate_module.runner_mod, "run_pi",
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="429 Too Many Requests", rate_limited=True),
    )
    result, _solution, _usage = run_and_grade(_task(), _pi_model(), _calibration_config())
    assert result.outcome == "error_harness"


def test_run_and_grade_maps_context_unavailable_to_error_harness(monkeypatch):
    # A repo clone/checkout failure happens before pi is ever invoked — an infra problem, and per
    # the spec's fairness requirement a task that can't be set up consistently for every model
    # shouldn't be scored as a wrong answer for any of them.
    monkeypatch.setattr(
        calibrate_module.runner_mod, "run_pi",
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="clone failed", context_unavailable=True),
    )
    result, _solution, _usage = run_and_grade(_task(), _pi_model(), _calibration_config())
    assert result.outcome == "error_harness"


def test_run_and_grade_still_treats_a_genuine_no_solution_as_a_real_failure(monkeypatch):
    monkeypatch.setattr(
        calibrate_module.runner_mod, "run_pi",
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="empty response", rate_limited=False),
    )
    result, _solution, _usage = run_and_grade(_task(), _pi_model(), _calibration_config())
    assert result.outcome == "error_no_solution"


def test_run_and_grade_maps_a_harness_error_to_error_harness_not_error_no_solution(monkeypatch):
    # The provider itself rejected the call (e.g. an API auth failure) — pi exits 0 in this case, so
    # this never reached the model at all. Confirmed live this session: a real 401 got silently
    # miscounted as error_no_solution before this fix.
    monkeypatch.setattr(
        calibrate_module.runner_mod, "run_pi",
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="401 API key is invalid", harness_error=True),
    )
    result, _solution, _usage = run_and_grade(_task(), _pi_model(), _calibration_config())
    assert result.outcome == "error_harness"


def test_run_and_grade_maps_a_timed_out_run_result_to_error_harness_not_error_no_solution(monkeypatch):
    # A call that never finished isn't evidence the model couldn't solve the task, just that it
    # didn't in the time we gave it — must not count against the model's error rate.
    monkeypatch.setattr(
        calibrate_module.runner_mod, "run_pi",
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="pi timed out after 300s", timed_out=True),
    )
    result, _solution, _usage = run_and_grade(_task(), _pi_model(), _calibration_config())
    assert result.outcome == "error_harness"


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
        result, solution, _usage = run_and_grade(_task(), _pi_model(), _calibration_config())

    assert result.outcome == "fail"
    assert solution == "def f():\n    return 1"
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
