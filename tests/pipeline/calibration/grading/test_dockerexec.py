import base64
import subprocess
from collections import OrderedDict

import pytest

from router.pipeline.calibration.grading import dockerexec
from router.pipeline.calibration.grading.dockerexec import (
    apply_patch_or_fail_cmd,
    classify,
    cleanup_image,
    pytest_collect_then_run,
    report_cmd,
    run,
    sentinel,
    touch_image,
    write_file_cmd,
)

NONCE = "abc123"


@pytest.fixture(autouse=True)
def _isolated_image_cache(monkeypatch):
    # touch_image's LRU set is module-level state — give each test a fresh one so cache contents
    # from one test can't leak into another's eviction assertions.
    monkeypatch.setattr(dockerexec, "_recently_used_images", OrderedDict())


def test_classify_pass_sentinel():
    result = classify(NONCE, 1, f"{sentinel(NONCE)}PASS\n", "")
    assert result.outcome == "pass"


def test_classify_fail_sentinel_preserves_detail():
    result = classify(NONCE, 0, f"{sentinel(NONCE)}FAIL:pytest reported failures\n", "")
    assert result.outcome == "fail"
    assert result.detail == "pytest reported failures"


def test_classify_harness_sentinel():
    result = classify(NONCE, 1, f"{sentinel(NONCE)}HARNESS:bug patch failed to apply\n", "")
    assert result.outcome == "error_harness"


def test_no_sentinel_and_nonzero_exit_is_error_harness_never_fail():
    # A container that exits nonzero WITHOUT our script ever reporting must never be misread as
    # the candidate failing the task.
    result = classify(NONCE, 1, "some unrelated output\n", "")
    assert result.outcome == "error_harness"
    assert result.outcome != "fail"


def test_real_daemon_down_string_is_error_harness():
    stderr = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"
    result = classify(NONCE, 1, "", stderr)
    assert result.outcome == "error_harness"


def test_missing_image_string_is_error_harness():
    stderr = "Unable to find image 'swebench/swesmith.x86_64.some_repo.abc123:latest' locally"
    result = classify(NONCE, 1, "", stderr)
    assert result.outcome == "error_harness"


def test_sentinel_with_a_different_nonce_is_ignored():
    result = classify(NONCE, 1, f"{sentinel('other-nonce')}PASS\n", "")
    assert result.outcome == "error_harness"


def test_last_matching_sentinel_wins():
    output = f"{sentinel(NONCE)}HARNESS:setup failed\n{sentinel(NONCE)}PASS\n"
    result = classify(NONCE, 0, output, "")
    assert result.outcome == "pass"


def test_report_cmd_round_trips_through_classify():
    cmd = report_cmd(NONCE, "FAIL", "candidate patch failed to apply")
    # report_cmd produces a shell command; simulate what running it would print to stdout.
    assert cmd.startswith("echo ")
    stdout = f"{sentinel(NONCE)}FAIL:candidate patch failed to apply\n"
    assert classify(NONCE, 0, stdout, "").outcome == "fail"


def test_write_file_cmd_round_trips_unicode_and_shell_special_chars():
    text = "diff --git a/x b/x\n+café \"quoted\" $VAR 'single' \n"
    cmd = write_file_cmd(text, "/tmp/dest.txt")
    assert "/tmp/dest.txt" in cmd
    # The command must not embed the raw text directly into the shell line — only its base64 form.
    assert text not in cmd


def test_run_timeout_expired_maps_to_error_timeout(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    result = run("some-image", "echo hi", NONCE, timeout_seconds=5)
    assert result.outcome == "error_timeout"


def test_run_names_the_container_so_it_can_be_killed_later(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout=f"{sentinel(NONCE)}PASS\n", stderr="")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    run("some-image", "echo hi", NONCE, timeout_seconds=5)

    name_index = captured["args"].index("--name")
    assert captured["args"][name_index + 1] == f"router-grade-{NONCE}"


def test_run_kills_the_container_on_timeout(monkeypatch):
    # Killing the `docker run` client does NOT stop the container — an explicit `docker kill` is
    # required, or a timed-out task keeps burning CPU for the rest of the calibration run.
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 0))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    result = run("some-image", "sleep 300", NONCE, timeout_seconds=5)

    assert result.outcome == "error_timeout"
    assert ["docker", "kill", f"router-grade-{NONCE}"] in calls


def test_kill_container_swallows_a_failing_docker_kill(monkeypatch):
    def fake_run(*args, **kwargs):
        raise OSError("docker daemon unreachable")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    dockerexec.kill_container("router-grade-whatever")  # must not raise


def test_run_file_not_found_maps_to_error_harness(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    result = run("some-image", "echo hi", NONCE, timeout_seconds=5)
    assert result.outcome == "error_harness"


def test_run_oserror_e2big_maps_to_error_harness_not_a_crash(monkeypatch):
    def fake_run(*args, **kwargs):
        raise OSError(7, "Argument list too long")  # errno 7 == E2BIG

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    result = run("some-image", "echo hi", NONCE, timeout_seconds=5)
    assert result.outcome == "error_harness"


def test_run_passes_script_via_stdin_never_as_an_argv_element(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(args, 0, stdout=f"{sentinel(NONCE)}PASS\n", stderr="")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    huge_script = "x" * 500_000  # comfortably over MAX_ARG_STRLEN if it were ever passed as argv
    result = run("some-image", huge_script, NONCE, timeout_seconds=5)

    assert result.outcome == "pass"
    assert captured["input"] == huge_script
    assert all(huge_script not in arg for arg in captured["args"])


def test_run_omits_volume_flags_by_default(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout=f"{sentinel(NONCE)}PASS\n", stderr="")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    run("some-image", "echo hi", NONCE, timeout_seconds=5)

    assert "-v" not in captured["args"]


def test_run_passes_each_volume_as_a_host_path_container_path_flag(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout=f"{sentinel(NONCE)}PASS\n", stderr="")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    run("some-image", "echo hi", NONCE, timeout_seconds=5, volumes={"/host/cache": "/root/.cache/go-build"})

    args = captured["args"]
    v_index = args.index("-v")
    assert args[v_index + 1] == "/host/cache:/root/.cache/go-build"
    # The volume flags must come before the image name, matching normal `docker run` argument order.
    assert v_index < args.index("some-image")


def test_cleanup_image_swallows_a_failing_docker_rmi(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="No such image")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    cleanup_image("some-image")  # must not raise


def test_cleanup_image_swallows_an_exception(monkeypatch):
    def fake_run(*args, **kwargs):
        raise OSError("docker daemon unreachable")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    cleanup_image("some-image")  # must not raise


def test_cleanup_image_passes_the_image_through_unmodified(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(dockerexec.subprocess, "run", fake_run)
    cleanup_image("swebench/swesmith.x86_64.some_repo.abc123:latest")
    assert "swebench/swesmith.x86_64.some_repo.abc123:latest" in captured["args"]


def test_touch_image_does_not_evict_while_under_the_cap(monkeypatch):
    cleaned_up = []
    monkeypatch.setattr(dockerexec, "cleanup_image", lambda image: cleaned_up.append(image))

    for i in range(dockerexec._MAX_CACHED_IMAGES):
        touch_image(f"image-{i}")

    assert cleaned_up == []


def test_touch_image_evicts_the_least_recently_used_image_once_over_the_cap(monkeypatch):
    cleaned_up = []
    monkeypatch.setattr(dockerexec, "cleanup_image", lambda image: cleaned_up.append(image))

    for i in range(dockerexec._MAX_CACHED_IMAGES + 1):
        touch_image(f"image-{i}")

    # image-0 was the first one touched and never re-touched — it's the oldest, so it's evicted.
    assert cleaned_up == ["image-0"]


def test_touch_image_refreshes_recency_so_a_re_touched_image_is_not_the_one_evicted(monkeypatch):
    cleaned_up = []
    monkeypatch.setattr(dockerexec, "cleanup_image", lambda image: cleaned_up.append(image))

    for i in range(dockerexec._MAX_CACHED_IMAGES):
        touch_image(f"image-{i}")
    touch_image("image-0")  # re-touch the oldest — it should no longer be least-recently-used
    touch_image("image-new")  # pushes the cache one over the cap

    # image-1, not image-0, is now the least-recently-used and gets evicted.
    assert cleaned_up == ["image-1"]


def _decoded_ids_file_content(script: str) -> str:
    """`pytest_collect_then_run` writes node ids via `write_file_cmd`'s base64 transport (never
    embedded raw in the script — same reasoning `test_write_file_cmd_round_trips...` covers), so a
    test needs to find and decode that blob to inspect what was actually written."""
    for line in script.splitlines():
        if line.startswith("echo ") and "base64 -d > /tmp/node_ids.txt" in line:
            encoded = line.split(" ", 2)[1]
            return base64.b64decode(encoded).decode("utf-8")
    raise AssertionError("no base64 write of /tmp/node_ids.txt found in script")


def test_pytest_collect_then_run_writes_every_node_id_one_per_line_not_shell_embedded():
    # Regression coverage for the property `swegym.py`'s old `_node_ids` used to guarantee via
    # shell-quoting: an id containing a space (a real, confirmed shape in this dataset) must
    # survive intact, not get split into two ids. The new mechanism (a file, one id per line, read
    # back with `readarray`) sidesteps shell-quoting entirely instead of relying on it.
    node_ids = ["tests/test_x.py::test_one", "tests/test with space.py::test_two"]
    script = pytest_collect_then_run(node_ids, NONCE, "/testbed", "conda activate testbed")

    assert _decoded_ids_file_content(script) == "tests/test_x.py::test_one\ntests/test with space.py::test_two\n"
    assert node_ids[0] not in script  # never embedded raw — only via the base64 blob
    assert node_ids[1] not in script


def test_pytest_collect_then_run_collects_before_executing():
    script = pytest_collect_then_run(["tests/test_x.py::test_one"], NONCE, "/testbed", "conda activate testbed")

    collect_index = script.index("--collect-only")
    execute_index = script.index('python -m pytest -q "${VALID[@]}"')
    assert collect_index < execute_index
    # Both passes must run in the same cwd/env — pytest resolves relative node ids against
    # getcwd(), so a mismatch would make the collect-only pass's "not found" lines use a different
    # absolute path than step 2 expects, silently excluding every id.
    assert script.count("conda activate testbed && cd /testbed") == 2


def test_pytest_collect_then_run_reports_harness_immediately_when_every_id_is_excluded():
    script = pytest_collect_then_run(["tests/test_x.py::test_one"], NONCE, "/testbed", "conda activate testbed")

    assert report_cmd(NONCE, "HARNESS", "pytest could not collect the specified test ids") in script
    # The real (potentially slow) execution pass must be skipped entirely in this branch.
    harness_report_index = script.index(report_cmd(NONCE, "HARNESS", "pytest could not collect the specified test ids"))
    execute_index = script.index('python -m pytest -q "${VALID[@]}"')
    assert harness_report_index < execute_index  # HARNESS branch precedes (and short-circuits before) it


def test_pytest_collect_then_run_checks_both_of_pytests_not_found_message_shapes():
    # pytest reports an unresolvable id one of two ways depending on whether the FILE itself
    # exists — "not found: <repo_dir>/<id>" (file exists but the specific test/class doesn't) or
    # "file or directory not found: <id>" (file doesn't exist at all) — both must be checked.
    script = pytest_collect_then_run(["tests/test_x.py::test_one"], NONCE, "/testbed", "conda activate testbed")
    assert 'grep -qxF "ERROR: not found: /testbed/$id"' in script
    assert 'grep -qxF "ERROR: file or directory not found: $id"' in script


def test_pytest_collect_then_run_folds_excluded_count_into_fail_and_pass_detail():
    script = pytest_collect_then_run(
        ["tests/test_x.py::test_one"], NONCE, "/testbed", "conda activate testbed", fail_detail="pytest reported failures"
    )

    assert "id(s) excluded as uncollectable" in script
    assert f'{sentinel(NONCE)}FAIL:pytest reported failures ($EXCL_NOTE)' in script
    assert f'{sentinel(NONCE)}PASS:$EXCL_NOTE' in script


def test_apply_patch_or_fail_cmd_folds_gits_real_stderr_into_the_fail_detail():
    # git's own error message must be preserved, not discarded behind a static failure string —
    # otherwise "malformed diff" is indistinguishable from "diff doesn't match this baseline".
    script = apply_patch_or_fail_cmd(NONCE, "/tmp/candidate.patch", fail_detail="candidate patch failed to apply")

    assert "git apply /tmp/candidate.patch 2>/tmp/apply_err.txt ||" in script
    assert f'echo "{sentinel(NONCE)}FAIL:candidate patch failed to apply: $ERR"' in script


def test_apply_patch_or_fail_cmd_never_reports_on_the_success_path():
    # The FAIL report must live entirely inside the `|| { ... }` block — a successful `git apply`
    # must fall through to whatever the caller appends next (the test stage), not report anything.
    script = apply_patch_or_fail_cmd(NONCE, "/tmp/candidate.patch")
    fail_line_index = script.index("echo")
    apply_line_index = script.index("git apply")
    assert apply_line_index < fail_line_index  # the echo is inside the || block, after the apply attempt
