import numpy as np

from router.common.assign import ClusterMap
from router.common.config import CalibrationConfig, ModelConfig, SmoothingConfig
from router.pipeline.calibration import calibrate as calibrate_module
from router.pipeline.calibration.calibrate import (
    SelectedTask,
    _expected_solution,
    _preview,
    _stats_from_outcomes,
    run_and_grade,
    select_tasks,
)
from router.pipeline.calibration.grading.base import GradeResult, Task
from router.pipeline.calibration.runner import RunResult, TokenUsage


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
        model_id="llama-3.1-8b-instant", provider="openai", runner="pi",
        cost_input=0.00005, cost_output=0.00008, context_window=131072, max_tokens=131072,
    )


def _task() -> Task:
    return Task(task_id="bigcodebench:0", source="bigcodebench", prompt="do the thing", reference_solution="x", row={})


def _calibration_config() -> CalibrationConfig:
    return CalibrationConfig(
        gradeable_sources=["bigcodebench"], tasks_per_cluster=4,
        task_timeout_seconds=60, smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        seed=42, lambda_sweep=[0, 0.1],
    )


def _named_model(model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id, provider="openai", runner="pi",
        cost_input=0.00005, cost_output=0.00008, context_window=131072, max_tokens=131072,
    )


# --- select_tasks: stratified-by-cluster selection driven by a task-cluster-map artifact rather
# than a per-run random pre-filter + embedding pass. See select_tasks' own docstring for why the
# old design is gone (it silently left clusters with zero tasks — confirmed against a real run). --

def _cluster_map(k: int, artifact_id: str = "clustermap-x") -> ClusterMap:
    return ClusterMap(artifact_id=artifact_id, embedding_model_id="m", dimensions=4, centroids=np.zeros((k, 4)))


def _task_cluster_map(cluster_map_id: str, entries: list[tuple[str, str, int]]) -> dict:
    """`entries` is a list of (task_id, source, cluster_id) triples, matching
    task_cluster_map.py's own {"task_id", "source", "cluster_id"} shape."""
    return {
        "schema_version": 1, "artifact_id": "taskclustermap-x", "created_at": "2026-01-01T00:00:00Z",
        "cluster_map_id": cluster_map_id,
        "tasks": [{"task_id": t, "source": s, "cluster_id": c} for t, s, c in entries],
    }


def _gradeable_task(task_id: str, source: str = "bigcodebench") -> Task:
    return Task(task_id=task_id, source=source, prompt=f"prompt for {task_id}", reference_solution="x", row={})


def _stub_gradeable_tasks(monkeypatch, tasks_by_source: dict[str, list[Task]]) -> None:
    monkeypatch.setattr(calibrate_module, "load_gradeable_tasks", lambda source: tasks_by_source.get(source, []))


def test_select_tasks_raises_when_the_task_cluster_map_was_built_against_a_different_cluster_map(monkeypatch):
    _stub_gradeable_tasks(monkeypatch, {})
    cluster_map = _cluster_map(k=2, artifact_id="clustermap-current")
    stale_map = _task_cluster_map(cluster_map_id="clustermap-OLD", entries=[])

    try:
        select_tasks(_calibration_config(), cluster_map, stale_map)
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "clustermap-OLD" in str(e)
        assert "clustermap-current" in str(e)


def test_select_tasks_groups_by_the_cluster_id_the_map_declares(monkeypatch):
    tasks = {"bigcodebench": [_gradeable_task("t1"), _gradeable_task("t2"), _gradeable_task("t3")]}
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=2)
    task_map = _task_cluster_map("clustermap-x", [("t1", "bigcodebench", 0), ("t2", "bigcodebench", 0), ("t3", "bigcodebench", 1)])
    config = CalibrationConfig(
        gradeable_sources=["bigcodebench"], tasks_per_cluster=10, task_timeout_seconds=60,
        smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        seed=42, lambda_sweep=[0],
    )

    selected = select_tasks(config, cluster_map, task_map)

    by_cluster: dict[int, set[str]] = {}
    for s in selected:
        by_cluster.setdefault(s.cluster_id, set()).add(s.task.task_id)
    assert by_cluster == {0: {"t1", "t2"}, 1: {"t3"}}


def test_select_tasks_caps_at_tasks_per_cluster(monkeypatch):
    tasks = {"bigcodebench": [_gradeable_task(f"t{i}") for i in range(5)]}
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=1)
    task_map = _task_cluster_map("clustermap-x", [(f"t{i}", "bigcodebench", 0) for i in range(5)])
    config = CalibrationConfig(
        gradeable_sources=["bigcodebench"], tasks_per_cluster=2, task_timeout_seconds=60,
        smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        seed=42, lambda_sweep=[0],
    )

    selected = select_tasks(config, cluster_map, task_map)

    assert len(selected) == 2


def test_select_tasks_puts_every_selected_task_in_the_calibration_split(monkeypatch):
    # select_tasks() no longer carves out a "holdout" split (see config/calibration.yaml's comment
    # on the removed holdout_fraction) — every selected task must come back as "calibration".
    tasks = {"bigcodebench": [_gradeable_task(f"t{i}") for i in range(10)]}
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=1)
    task_map = _task_cluster_map("clustermap-x", [(f"t{i}", "bigcodebench", 0) for i in range(10)])
    config = CalibrationConfig(
        gradeable_sources=["bigcodebench"], tasks_per_cluster=10, task_timeout_seconds=60,
        smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        seed=42, lambda_sweep=[0],
    )

    selected = select_tasks(config, cluster_map, task_map)

    assert len(selected) == 10
    assert {s.split for s in selected} == {"calibration"}


def test_select_tasks_skips_a_mapped_task_id_the_gradeable_loader_does_not_actually_return(monkeypatch):
    # The task-cluster-map is built from corpus.py's broader row population, which doesn't apply
    # every filter calibration/tasks.py does (e.g. DS-1000's Matplotlib exclusion) — a mapped id
    # with no matching gradeable task must be silently skipped, not crash or get selected anyway.
    tasks = {"bigcodebench": [_gradeable_task("real-task")]}
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=1)
    task_map = _task_cluster_map(
        "clustermap-x", [("real-task", "bigcodebench", 0), ("filtered-out-elsewhere", "bigcodebench", 0)],
    )

    selected = select_tasks(_calibration_config(), cluster_map, task_map)

    assert [s.task.task_id for s in selected] == ["real-task"]


def test_select_tasks_ignores_entries_for_a_source_not_in_gradeable_sources(monkeypatch):
    tasks = {"bigcodebench": [_gradeable_task("t1")], "ds1000": [_gradeable_task("d1", source="ds1000")]}
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=1)
    task_map = _task_cluster_map("clustermap-x", [("t1", "bigcodebench", 0), ("d1", "ds1000", 0)])
    config = CalibrationConfig(  # gradeable_sources deliberately excludes ds1000
        gradeable_sources=["bigcodebench"], tasks_per_cluster=10, task_timeout_seconds=60,
        smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        seed=42, lambda_sweep=[0],
    )

    selected = select_tasks(config, cluster_map, task_map)

    assert [s.task.task_id for s in selected] == ["t1"]


def test_select_tasks_does_not_crash_when_a_cluster_has_zero_gradeable_tasks(monkeypatch):
    # A genuine gap in the gradeable pool (not a random-draw artifact — see the function's own
    # docstring on why this can no longer happen silently) must still produce a valid, if smaller,
    # selection for every OTHER cluster rather than failing the whole run.
    tasks = {"bigcodebench": [_gradeable_task("t1")]}
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=3)  # clusters 1 and 2 have no entries in the map at all
    task_map = _task_cluster_map("clustermap-x", [("t1", "bigcodebench", 0)])

    selected = select_tasks(_calibration_config(), cluster_map, task_map)

    assert {s.cluster_id for s in selected} == {0}
    assert [s.task.task_id for s in selected] == ["t1"]


def test_select_tasks_is_deterministic_given_the_same_seed(monkeypatch):
    tasks = {"bigcodebench": [_gradeable_task(f"t{i}") for i in range(10)]}
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=1)
    task_map = _task_cluster_map("clustermap-x", [(f"t{i}", "bigcodebench", 0) for i in range(10)])
    config = CalibrationConfig(
        gradeable_sources=["bigcodebench"], tasks_per_cluster=4, task_timeout_seconds=60,
        smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        seed=7, lambda_sweep=[0],
    )

    first = select_tasks(config, cluster_map, task_map)
    second = select_tasks(config, cluster_map, task_map)

    assert [(s.task.task_id, s.cluster_id, s.split) for s in first] == [
        (s.task.task_id, s.cluster_id, s.split) for s in second
    ]


# --- category_mix: spec §5.1's "aim for the following distribution" (repo_python/multilingual/
# standalone), applied within each cluster's tasks_per_cluster budget. See calibrate.py's
# CATEGORY_SOURCES for the fixed source->category mapping this all keys off. --------------------

def _config_with_category_mix(category_mix: dict, tasks_per_cluster: int = 10, gradeable_sources=None) -> CalibrationConfig:
    return CalibrationConfig(
        gradeable_sources=gradeable_sources or ["bigcodebench", "ds1000", "swe-smith", "swe-gym", "multi-swe-rl"],
        tasks_per_cluster=tasks_per_cluster, task_timeout_seconds=60,
        smoothing=SmoothingConfig(method="shrink_to_model_global", prior_weight=5),
        seed=42, lambda_sweep=[0], category_mix=category_mix,
    )


def test_validate_category_mix_rejects_an_unknown_category():
    try:
        calibrate_module._validate_category_mix({"bogus": 1.0})
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "bogus" in str(e)


def test_validate_category_mix_rejects_ratios_that_do_not_sum_to_one():
    try:
        calibrate_module._validate_category_mix({"repo_python": 0.5, "multilingual": 0.3, "standalone": 0.1})
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "sum" in str(e)


def test_select_tasks_splits_a_cluster_s_budget_by_category_when_every_category_has_enough_tasks(monkeypatch):
    tasks = {
        "bigcodebench": [_gradeable_task(f"bcb{i}", source="bigcodebench") for i in range(5)],
        "ds1000": [_gradeable_task(f"ds{i}", source="ds1000") for i in range(5)],
        "swe-smith": [_gradeable_task(f"ss{i}", source="swe-smith") for i in range(10)],
        "swe-gym": [_gradeable_task(f"sg{i}", source="swe-gym") for i in range(10)],
        "multi-swe-rl": [_gradeable_task(f"ms{i}", source="multi-swe-rl") for i in range(10)],
    }
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=1)
    entries = [(t.task_id, t.source, 0) for source_tasks in tasks.values() for t in source_tasks]
    task_map = _task_cluster_map("clustermap-x", entries)
    config = _config_with_category_mix({"repo_python": 0.60, "multilingual": 0.25, "standalone": 0.15})

    selected = select_tasks(config, cluster_map, task_map)

    by_category = {}
    for s in selected:
        by_category.setdefault(calibrate_module._SOURCE_CATEGORY[s.task.source], 0)
        by_category[calibrate_module._SOURCE_CATEGORY[s.task.source]] += 1
    assert len(selected) == 10
    assert by_category == {"repo_python": 6, "multilingual": 2, "standalone": 2}


def test_select_tasks_backfills_a_category_shortfall_from_other_categories_in_the_same_cluster(monkeypatch):
    # This cluster has ONLY standalone tasks — no repo_python or multilingual tasks exist in it at
    # all (a real, observed shape: some clusters are 100% one source). The 60/25/15 quota can't be
    # met for two of the three categories, but the cluster's tasks_per_cluster budget should still
    # be filled from what IS available, not silently left under-filled.
    tasks = {"bigcodebench": [_gradeable_task(f"bcb{i}", source="bigcodebench") for i in range(10)]}
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=1)
    task_map = _task_cluster_map("clustermap-x", [(f"bcb{i}", "bigcodebench", 0) for i in range(10)])
    config = _config_with_category_mix(
        {"repo_python": 0.60, "multilingual": 0.25, "standalone": 0.15}, gradeable_sources=["bigcodebench"],
    )

    selected = select_tasks(config, cluster_map, task_map)

    assert len(selected) == 10  # fully backfilled from the only category with any tasks
    assert {s.task.source for s in selected} == {"bigcodebench"}


def test_select_tasks_is_deterministic_with_category_mix(monkeypatch):
    tasks = {
        "bigcodebench": [_gradeable_task(f"bcb{i}", source="bigcodebench") for i in range(5)],
        "swe-smith": [_gradeable_task(f"ss{i}", source="swe-smith") for i in range(10)],
        "multi-swe-rl": [_gradeable_task(f"ms{i}", source="multi-swe-rl") for i in range(10)],
    }
    _stub_gradeable_tasks(monkeypatch, tasks)
    cluster_map = _cluster_map(k=1)
    entries = [(t.task_id, t.source, 0) for source_tasks in tasks.values() for t in source_tasks]
    task_map = _task_cluster_map("clustermap-x", entries)
    config = _config_with_category_mix(
        {"repo_python": 0.60, "multilingual": 0.25, "standalone": 0.15},
        gradeable_sources=["bigcodebench", "swe-smith", "multi-swe-rl"],
    )

    first = select_tasks(config, cluster_map, task_map)
    second = select_tasks(config, cluster_map, task_map)

    assert [s.task.task_id for s in first] == [s.task.task_id for s in second]


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
    # swe-gym/multi-swe-rl build one multi-GB image PER INSTANCE, so all models must be graded
    # while a task's image is still hot; a models-outer loop would re-pull every image per model.
    calls = []

    def fake_run_and_grade(task, model, calibration_config):
        calls.append((task.task_id, model.model_id))
        return GradeResult(outcome=_outcome_for(task.task_id, model.model_id)), "", None

    monkeypatch.setattr(calibrate_module, "run_and_grade", fake_run_and_grade)
    models = [_named_model("m1"), _named_model("m2")]
    selected = [_selected("t1", 0), _selected("t2", 0)]

    # details_csv_path must be redirected to tmp_path — the default writes to the real repo's
    # artifacts/calibration-details.csv.
    calibrate_module.calibrate_models(models, selected, _calibration_config(), details_csv_path=tmp_path / "details.csv")

    assert calls == [("t1", "m1"), ("t1", "m2"), ("t2", "m1"), ("t2", "m2")]


# --- skip real models on a task whose ground truth (reference-oracle/null-baseline, already part
# of the normal roster — no new grading calls) turns out to be bad. See _ground_truth_invalid. ----

def _control_model(runner: str, model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id, provider="stub", runner=runner, cost_input=0, cost_output=0, context_window=0, max_tokens=0,
    )


def test_ground_truth_invalid_requires_both_controls_to_have_actually_run():
    assert calibrate_module._ground_truth_invalid(None, "fail") is False
    assert calibrate_module._ground_truth_invalid("pass", None) is False
    assert calibrate_module._ground_truth_invalid(None, None) is False


def test_ground_truth_invalid_true_when_reference_fails():
    assert calibrate_module._ground_truth_invalid("fail", "fail") is True


def test_ground_truth_invalid_true_when_null_passes():
    assert calibrate_module._ground_truth_invalid("pass", "pass") is True


def test_ground_truth_invalid_false_when_reference_passes_and_null_fails():
    assert calibrate_module._ground_truth_invalid("pass", "fail") is False


def test_calibrate_models_skips_real_models_when_reference_fails(monkeypatch, tmp_path):
    calls = []

    def fake_run_and_grade(task, model, cfg):
        calls.append(model.model_id)
        return GradeResult(outcome="fail"), "", None  # reference fails -> bad ground truth

    monkeypatch.setattr(calibrate_module, "run_and_grade", fake_run_and_grade)
    models = [_named_model("real"), _control_model("reference", "reference-oracle"), _control_model("null", "null-baseline")]
    selected = [_selected("t1", 0)]

    _results, detail_rows = calibrate_module.calibrate_models(
        models, selected, _calibration_config(), details_csv_path=tmp_path / "details.csv",
    )

    assert "real" not in calls  # never actually called
    real_row = next(r for r in detail_rows if r.model_id == "real")
    assert real_row.outcome == "error_harness"
    assert "ground truth" in real_row.detail
    assert real_row.duration_ms == 0


def test_calibrate_models_skips_real_models_when_null_solution_passes(monkeypatch, tmp_path):
    def fake_run_and_grade(task, model, cfg):
        if model.runner == "reference":
            return GradeResult(outcome="pass"), "", None
        if model.runner == "null":
            return GradeResult(outcome="pass"), "", None  # empty solution incorrectly passes
        raise AssertionError("real model must not be called")

    monkeypatch.setattr(calibrate_module, "run_and_grade", fake_run_and_grade)
    models = [_named_model("real"), _control_model("reference", "reference-oracle"), _control_model("null", "null-baseline")]
    selected = [_selected("t1", 0)]

    _results, detail_rows = calibrate_module.calibrate_models(
        models, selected, _calibration_config(), details_csv_path=tmp_path / "details.csv",
    )

    real_row = next(r for r in detail_rows if r.model_id == "real")
    assert real_row.outcome == "error_harness"


def test_calibrate_models_runs_real_models_normally_when_ground_truth_is_valid(monkeypatch, tmp_path):
    calls = []

    def fake_run_and_grade(task, model, cfg):
        calls.append(model.model_id)
        if model.runner == "reference":
            return GradeResult(outcome="pass"), "", None
        if model.runner == "null":
            return GradeResult(outcome="fail"), "", None
        return GradeResult(outcome="pass"), "solution", None

    monkeypatch.setattr(calibrate_module, "run_and_grade", fake_run_and_grade)
    models = [_named_model("real"), _control_model("reference", "reference-oracle"), _control_model("null", "null-baseline")]
    selected = [_selected("t1", 0)]

    _results, detail_rows = calibrate_module.calibrate_models(
        models, selected, _calibration_config(), details_csv_path=tmp_path / "details.csv",
    )

    assert "real" in calls
    real_row = next(r for r in detail_rows if r.model_id == "real")
    assert real_row.outcome == "pass"


def test_calibrate_models_does_not_skip_when_controls_are_absent_from_the_roster(monkeypatch, tmp_path):
    # Mirrors a --model-scoped incremental run: no reference-oracle/null-baseline in the roster at
    # all, so there's nothing to check against — real models must run exactly as before this existed.
    calls = []

    def fake_run_and_grade(task, model, cfg):
        calls.append(model.model_id)
        return GradeResult(outcome="pass"), "solution", None

    monkeypatch.setattr(calibrate_module, "run_and_grade", fake_run_and_grade)
    models = [_named_model("real")]
    selected = [_selected("t1", 0)]

    calibrate_module.calibrate_models(models, selected, _calibration_config(), details_csv_path=tmp_path / "details.csv")

    assert calls == ["real"]


def test_calibrate_models_grades_controls_before_real_models_even_if_listed_after(monkeypatch, tmp_path):
    calls = []

    def fake_run_and_grade(task, model, cfg):
        calls.append(model.model_id)
        # Valid ground truth: reference passes, null fails — real model must actually run.
        outcome = "fail" if model.runner == "null" else "pass"
        return GradeResult(outcome=outcome), "", None

    monkeypatch.setattr(calibrate_module, "run_and_grade", fake_run_and_grade)
    # Real model listed FIRST in input order — calibrate_models must still grade controls first.
    models = [_named_model("real"), _control_model("reference", "reference-oracle"), _control_model("null", "null-baseline")]
    selected = [_selected("t1", 0)]

    calibrate_module.calibrate_models(models, selected, _calibration_config(), details_csv_path=tmp_path / "details.csv")

    assert calls == ["reference-oracle", "null-baseline", "real"]


def test_calibrate_models_persists_the_full_solution_when_the_csv_preview_would_truncate_it(monkeypatch, tmp_path):
    # A solution over the CSV preview cap must still be persisted in full, not silently truncated.
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


def test_calibrate_models_persists_a_raw_response_fallback_as_raw_txt_not_diff(monkeypatch, tmp_path):
    # run_and_grade's pi branch falls back to Pi's raw --mode json stdout (starts with `{"type":`)
    # when no solution could be extracted — that text isn't a real diff, so it must not get a
    # `.diff` extension. See docs/engineering-notes.md, "Large solution truncation".
    raw_event_stream = '{"type":"session","v":1}\n' + ("x" * calibrate_module._CSV_SOLUTION_MAX_CHARS)
    monkeypatch.setattr(
        calibrate_module, "run_and_grade",
        lambda task, model, cfg: (GradeResult(outcome="error_harness"), raw_event_stream, None),
    )
    models = [_named_model("m1")]
    selected = [_selected("t1", 0)]
    details_path = tmp_path / "details.csv"

    calibrate_module.calibrate_models(models, selected, _calibration_config(), details_csv_path=details_path)

    persisted = tmp_path / "solutions" / "details" / "t1__m1.raw.txt"
    assert persisted.read_text(encoding="utf-8") == raw_event_stream
    assert not (tmp_path / "solutions" / "details" / "t1__m1.diff").exists()


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


def test_calibrate_models_records_turn_count_from_usage_in_the_details_csv(monkeypatch, tmp_path):
    usage = TokenUsage(input_tokens=100, output_tokens=50, cost_usd=0.01, turn_count=4)
    monkeypatch.setattr(
        calibrate_module, "run_and_grade",
        lambda task, model, cfg: (GradeResult(outcome="pass"), "solution", usage),
    )
    models = [_named_model("m1")]
    selected = [_selected("t1", 0)]

    _results, detail_rows = calibrate_module.calibrate_models(
        models, selected, _calibration_config(), details_csv_path=tmp_path / "details.csv",
    )

    assert detail_rows[0].turns == 4


def test_calibrate_models_defaults_turns_to_zero_when_usage_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibrate_module, "run_and_grade",
        lambda task, model, cfg: (GradeResult(outcome="pass"), "solution", None),
    )
    models = [_named_model("m1")]
    selected = [_selected("t1", 0)]

    _results, detail_rows = calibrate_module.calibrate_models(
        models, selected, _calibration_config(), details_csv_path=tmp_path / "details.csv",
    )

    assert detail_rows[0].turns == 0


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
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="empty response"),
    )
    result, _solution, _usage = run_and_grade(_task(), _pi_model(), _calibration_config())
    assert result.outcome == "error_no_solution"


def test_run_and_grade_maps_a_harness_error_to_error_harness_not_error_no_solution(monkeypatch):
    # The provider itself rejected the call (e.g. an API auth failure) — pi exits 0 in this case, so
    # this never reached the model at all and must not be counted as error_no_solution.
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
        lambda task, model, timeout_seconds: RunResult(solution="def f():\n    return 1", detail=""),
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
        lambda task, model, timeout_seconds: RunResult(solution=None, detail="empty response"),
    )
    with caplog.at_level("DEBUG", logger="router.pipeline.calibration.calibrate"):
        run_and_grade(_task(), _pi_model(), _calibration_config())
    assert "provided: (no solution extracted)" in caplog.text


# --- Pinned task selection: serialize/deserialize SelectedTask lists so a later run (in
# particular incremental single-model calibration) can grade against the EXACT same set an
# existing model-profiles.json was built from, instead of trusting select_tasks() to reproduce it. -

def _selected_task(task_id: str, cluster_id: int, split: str = "calibration", source: str = "bigcodebench") -> SelectedTask:
    return SelectedTask(task=_gradeable_task(task_id, source=source), cluster_id=cluster_id, split=split)


def test_compute_task_selection_digest_is_order_independent():
    a = [_selected_task("t1", 0), _selected_task("t2", 1)]
    b = list(reversed(a))
    assert calibrate_module.compute_task_selection_digest(a) == calibrate_module.compute_task_selection_digest(b)


def test_compute_task_selection_digest_changes_when_a_task_s_prompt_changes():
    original = [_selected_task("t1", 0)]
    changed = [SelectedTask(task=Task(task_id="t1", source="bigcodebench", prompt="a DIFFERENT prompt", reference_solution="x", row={}), cluster_id=0, split="calibration")]
    assert calibrate_module.compute_task_selection_digest(original) != calibrate_module.compute_task_selection_digest(changed)


def test_selected_tasks_to_dict_records_which_clusters_got_zero_tasks():
    selected = [_selected_task("t1", 0), _selected_task("t2", 2)]
    artifact = calibrate_module.selected_tasks_to_dict(selected, k=4)
    assert artifact["empty_clusters"] == [1, 3]


def test_selected_tasks_to_dict_omits_the_heavy_row_data():
    selected = [_selected_task("t1", 0)]
    artifact = calibrate_module.selected_tasks_to_dict(selected, k=1)
    assert set(artifact["tasks"][0]) == {"task_id", "source", "cluster_id", "split"}


def test_write_then_load_task_selection_round_trips_to_identical_selected_tasks(monkeypatch, tmp_path):
    _stub_gradeable_tasks(monkeypatch, {"bigcodebench": [_gradeable_task("t1"), _gradeable_task("t2")]})
    original = [_selected_task("t1", 0, split="calibration"), _selected_task("t2", 1, split="holdout")]
    path = tmp_path / "pin.json"

    calibrate_module.write_task_selection(calibrate_module.selected_tasks_to_dict(original, k=2), path)
    reloaded = calibrate_module.load_task_selection(path)

    assert [(s.task.task_id, s.cluster_id, s.split) for s in reloaded] == [
        (s.task.task_id, s.cluster_id, s.split) for s in original
    ]


def test_load_task_selection_raises_when_a_pinned_task_id_is_no_longer_gradeable(monkeypatch, tmp_path):
    # The source's own loader no longer returns "t1" — simulates upstream dataset drift. Must
    # raise, not silently grade a smaller set than what was pinned.
    _stub_gradeable_tasks(monkeypatch, {"bigcodebench": []})
    path = tmp_path / "pin.json"
    calibrate_module.write_task_selection(calibrate_module.selected_tasks_to_dict([_selected_task("t1", 0)], k=1), path)

    try:
        calibrate_module.load_task_selection(path)
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "t1" in str(e)


def test_load_task_selection_raises_when_the_pinned_content_digest_no_longer_matches(monkeypatch, tmp_path):
    # The source's loader now returns "t1" with DIFFERENT prompt text than what was pinned —
    # simulates the dataset row itself having changed upstream since the pin was written.
    path = tmp_path / "pin.json"
    calibrate_module.write_task_selection(calibrate_module.selected_tasks_to_dict([_selected_task("t1", 0)], k=1), path)
    _stub_gradeable_tasks(monkeypatch, {"bigcodebench": [Task(task_id="t1", source="bigcodebench", prompt="CHANGED", reference_solution="x", row={})]})

    try:
        calibrate_module.load_task_selection(path)
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "digest" in str(e)
