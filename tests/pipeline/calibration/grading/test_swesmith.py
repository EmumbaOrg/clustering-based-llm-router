from router.pipeline.calibration.grading import dockerexec, swesmith
from router.pipeline.calibration.grading.base import GradeResult, Task


def _task(**row_overrides) -> Task:
    row = {
        "image_name": "swebench/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536",
        "patch": "diff --git a/bug.py b/bug.py\n+bug\n",
        "FAIL_TO_PASS": ["tests/test_x.py::test_one"],
        "PASS_TO_PASS": ["tests/test_x.py::test_two", "tests/test with space.py::test_three"],
    }
    row.update(row_overrides)
    return Task(task_id="swesmith:0", source="swe-smith", prompt="fix the bug", reference_solution="", row=row)


def test_node_ids_are_shell_quoted():
    node_ids = swesmith._node_ids(_task())
    assert "'tests/test with space.py::test_three'" in node_ids
    assert "tests/test_x.py::test_one" in node_ids


def test_grade_returns_fail_without_touching_docker_for_an_empty_solution(monkeypatch):
    called = []
    monkeypatch.setattr(dockerexec, "run", lambda *a, **k: called.append("run"))
    monkeypatch.setattr(dockerexec, "touch_image", lambda *a, **k: called.append("touch"))

    result = swesmith.grade(_task(), "   ")

    assert result.outcome == "fail"
    assert called == []


def test_grade_calls_dockerexec_run_and_touches_the_image(monkeypatch):
    captured = {}

    def fake_run(image, script, nonce, timeout_seconds):
        captured["image"] = image
        captured["script"] = script
        return GradeResult(outcome="pass")

    touched = []
    monkeypatch.setattr(dockerexec, "run", fake_run)
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: touched.append(image))

    task = _task()
    result = swesmith.grade(task, "diff --git a/fix.py b/fix.py\n+fix\n")

    assert result.outcome == "pass"
    assert captured["image"] == task.row["image_name"]
    assert touched == [task.row["image_name"]]


def test_grade_touches_the_image_even_if_run_raises(monkeypatch):
    def raising_run(*args, **kwargs):
        raise RuntimeError("boom")

    touched = []
    monkeypatch.setattr(dockerexec, "run", raising_run)
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: touched.append(image))

    task = _task()
    try:
        swesmith.grade(task, "diff --git a/fix.py b/fix.py\n+fix\n")
    except RuntimeError:
        pass

    assert touched == [task.row["image_name"]]


def test_grade_script_reports_fail_not_harness_when_candidate_patch_wont_apply(monkeypatch):
    # The model's own bad diff must count against it (FAIL), unlike the dataset's own bug patch
    # failing to apply (HARNESS, see test_setup_script_reports_harness_for_bug_patch_apply_failure).
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    swesmith.grade(_task(), "diff --git a/fix.py b/fix.py\n+fix\n")

    assert "candidate patch failed to apply" in captured["script"]
    assert "HARNESS:candidate" not in captured["script"]


def test_setup_script_reports_harness_for_bug_patch_apply_failure():
    script = swesmith._setup_script(_task(), "nonce")
    assert dockerexec.report_cmd("nonce", "HARNESS", "bug patch failed to apply") in script
    assert "git apply /tmp/bug.patch" in script


def test_grade_reference_reports_harness_for_a_failed_reverse_apply(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    swesmith.grade_reference(_task())

    assert "bug patch failed to reverse-apply" in captured["script"]
    assert "git apply -R /tmp/bug.patch" in captured["script"]


def test_pytest_script_reports_pass_and_fail_off_the_same_exit_code_check():
    script = swesmith._pytest_script(_task(), "nonce")
    assert "python -m pytest -q" in script
    assert dockerexec.report_cmd("nonce", "PASS") in script
    assert dockerexec.report_cmd("nonce", "FAIL", "pytest reported failures") in script
