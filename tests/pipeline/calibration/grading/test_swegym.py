import base64

from router.pipeline.calibration.grading import dockerexec, swegym
from router.pipeline.calibration.grading.base import GradeResult, Task


def _decoded_ids_file_content(script: str) -> str:
    """See test_dockerexec.py's identical helper — node ids are written via `write_file_cmd`'s
    base64 transport, never embedded raw in the script."""
    for line in script.splitlines():
        if line.startswith("echo ") and "base64 -d > /tmp/node_ids.txt" in line:
            encoded = line.split(" ", 2)[1]
            return base64.b64decode(encoded).decode("utf-8")
    raise AssertionError("no base64 write of /tmp/node_ids.txt found in script")


def _task(**row_overrides) -> Task:
    row = {
        "instance_id": "getmoto__moto-7365",
        "repo": "getmoto/moto",
        "base_commit": "7f6c9cb1deafb280fe7fcc7551c38e397f11a706",
        "patch": "diff --git a/fix.py b/fix.py\n+fix\n",
        "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n+test\n",
        "FAIL_TO_PASS": ["tests/test_x.py::test_one"],
        "PASS_TO_PASS": ["tests/test_x.py::test_two", "tests/test with space.py::test_three"],
    }
    row.update(row_overrides)
    return Task(task_id="swegym:0", source="swe-gym", prompt="fix the bug", reference_solution="", row=row)


def test_image_replaces_dunders_and_lowercases():
    image = swegym._image(_task())
    assert image == "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest"


def test_image_lowercases_mixed_case_instance_ids():
    image = swegym._image(_task(instance_id="Project-MONAI__MONAI-3837"))
    assert image == "xingyaoww/sweb.eval.x86_64.project-monai_s_monai-3837:latest"


def test_pytest_script_passes_every_declared_node_id_to_the_shared_collect_then_run_helper():
    # Confirms the right ids/repo_dir/conda env are threaded through from a real task row;
    # space-containing-id safety itself is covered at dockerexec.pytest_collect_then_run.
    script = swegym._pytest_script(_task(), "nonce")
    assert _decoded_ids_file_content(script) == (
        "tests/test_x.py::test_one\ntests/test_x.py::test_two\ntests/test with space.py::test_three\n"
    )


def test_grade_returns_fail_without_touching_docker_for_an_empty_solution(monkeypatch):
    called = []
    monkeypatch.setattr(dockerexec, "run", lambda *a, **k: called.append("run"))
    monkeypatch.setattr(dockerexec, "touch_image", lambda *a, **k: called.append("touch"))

    result = swegym.grade(_task(), "   ")

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
    result = swegym.grade(task, "diff --git a/fix.py b/fix.py\n+fix\n")

    assert result.outcome == "pass"
    assert captured["image"] == swegym._image(task)
    assert touched == [swegym._image(task)]


def test_grade_touches_the_image_even_if_run_raises(monkeypatch):
    def raising_run(*args, **kwargs):
        raise RuntimeError("boom")

    touched = []
    monkeypatch.setattr(dockerexec, "run", raising_run)
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: touched.append(image))

    task = _task()
    try:
        swegym.grade(task, "diff --git a/fix.py b/fix.py\n+fix\n")
    except RuntimeError:
        pass

    assert touched == [swegym._image(task)]


def test_grade_script_reports_fail_not_harness_when_candidate_patch_wont_apply(monkeypatch):
    # The model's own bad diff must count against it (FAIL), unlike the dataset's own test_patch
    # failing to apply (HARNESS, see test_setup_script_reports_harness_for_test_patch_apply_failure).
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    swegym.grade(_task(), "diff --git a/fix.py b/fix.py\n+fix\n")

    assert "candidate patch failed to apply" in captured["script"]
    assert "HARNESS:candidate" not in captured["script"]


def test_setup_script_reports_harness_for_test_patch_apply_failure():
    script = swegym._setup_script(_task(), "nonce")
    assert dockerexec.report_cmd("nonce", "HARNESS", "test_patch failed to apply") in script
    assert "git apply /tmp/test.patch" in script


def test_grade_reference_applies_the_gold_patch_and_reports_harness_on_failure(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    swegym.grade_reference(_task())

    assert "gold patch failed to apply" in captured["script"]
    assert "git apply /tmp/gold.patch" in captured["script"]


def test_grade_null_applies_no_fix_patch(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    swegym.grade_null(_task())

    assert "/tmp/gold.patch" not in captured["script"]
    assert "/tmp/candidate.patch" not in captured["script"]
    assert "/tmp/test.patch" in captured["script"]


def test_pytest_script_reports_pass_and_fail_off_the_same_exit_code_check():
    script = swegym._pytest_script(_task(), "nonce")
    assert "python -m pytest -q" in script
    assert swegym.CONDA_ACTIVATE in script
    assert f"{dockerexec.sentinel('nonce')}PASS" in script
    assert "pytest reported failures" in script


def test_pytest_script_reports_harness_for_uncollectable_node_ids(monkeypatch):
    # An uncollectable node id (pytest exit 4/5) must map to HARNESS, not FAIL, so a
    # dataset/pytest-version mismatch can't spuriously count against a candidate. Confirms
    # swegym.py wires into the shared dockerexec.pytest_collect_then_run mechanism.
    script = swegym._pytest_script(_task(), "nonce")
    assert dockerexec.report_cmd("nonce", "HARNESS", "pytest could not collect the specified test ids") in script
    assert "PYTEST_EXIT -eq 4" in script
    assert "PYTEST_EXIT -eq 5" in script
