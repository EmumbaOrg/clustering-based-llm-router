import base64

from router.pipeline.calibration.grading import dockerexec, multiswerl
from router.pipeline.calibration.grading.base import GradeResult, Task


def _b64(text: str) -> str:
    # Test-name patterns are transported into the container via dockerexec.write_file_cmd, which
    # base64-encodes them — a raw substring check against the script text would never match.
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _task(**row_overrides) -> Task:
    row = {
        "org": "gin-gonic",
        "repo": "gin",
        "number": 4048,
        "base": {"sha": "28e57f58b184b2305ace192e02496bb89f6fd8cb"},
        "fix_patch": "diff --git a/fix.go b/fix.go\n+fix\n",
        "test_patch": "diff --git a/fix_test.go b/fix_test.go\n+test\n",
        "f2p_tests": {"TestMappingMultipleDefaultWithCollectionFormat": {"fix": "PASS", "test": "FAIL", "run": "NONE"}},
        "n2p_tests": {},
        "s2p_tests": {},
        "p2p_tests": {
            "TestDisableBindValidation": {"fix": "PASS", "test": "PASS", "run": "PASS"},
            # A real, table-driven subtest name with regex-special characters (quotes, parens,
            # pipe) — confirmed live against a real prometheus instance this session: matching
            # just the top-level name exercises this correctly with zero escaping needed.
            'TestPostingsForMatchers/n!~"(1|2.5)"': {"fix": "PASS", "test": "PASS", "run": "PASS"},
        },
    }
    row.update(row_overrides)
    return Task(task_id="multi-swe-rl:gin-gonic__gin-4048", source="multi-swe-rl", prompt="fix the bug", reference_solution="", row=row)


def test_image_lowercases_org_and_repo_and_uses_pr_tag():
    image = multiswerl._image(_task())
    assert image == "mswebench/gin-gonic_m_gin:pr-4048"


def test_repo_dir_uses_the_repo_field_not_org():
    assert multiswerl._repo_dir(_task()) == "/home/gin"


def test_discriminating_names_come_from_f2p_n2p_s2p_only():
    names = multiswerl._discriminating_test_names(_task())
    # Top-level only, deduped, sorted — p2p_tests' names must NOT leak into this set.
    assert names == ["TestMappingMultipleDefaultWithCollectionFormat"]


def test_regression_guard_names_come_from_p2p_only_and_strip_subtest_paths():
    names = multiswerl._regression_guard_test_names(_task())
    # The regex-special subtest suffix must be stripped to just the top-level identifier.
    assert names == ["TestDisableBindValidation", "TestPostingsForMatchers"]


def test_grade_returns_fail_without_touching_docker_for_an_empty_solution(monkeypatch):
    called = []
    monkeypatch.setattr(dockerexec, "run", lambda *a, **k: called.append("run"))
    monkeypatch.setattr(dockerexec, "touch_image", lambda *a, **k: called.append("touch"))

    result = multiswerl.grade(_task(), "   ")

    assert result.outcome == "fail"
    assert called == []


def test_grade_calls_dockerexec_run_and_touches_the_image(monkeypatch):
    captured = {}

    def fake_run(image, script, nonce, timeout_seconds, volumes=None):
        captured["image"] = image
        captured["script"] = script
        captured["volumes"] = volumes
        return GradeResult(outcome="pass")

    touched = []
    monkeypatch.setattr(dockerexec, "run", fake_run)
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: touched.append(image))

    task = _task()
    result = multiswerl.grade(task, "diff --git a/fix.go b/fix.go\n+fix\n")

    assert result.outcome == "pass"
    assert captured["image"] == multiswerl._image(task)
    assert touched == [multiswerl._image(task)]
    # A persistent Go build cache is what makes repeated same-commit compiles (reference/null/
    # candidate, and once per model in a real calibrate run) fast — see the module docstring's
    # measured 34x warm-cache speedup.
    # Regression guard: mounting an empty GOMODCACHE would wipe the image's own pre-populated
    # module cache (confirmed directly: 214MB-1.3GB already present) for zero benefit — see the
    # module docstring for the measured ~50s network-redownload cost this avoids. Only GOCACHE
    # (which every image starts genuinely empty, confirmed) is mounted.
    assert captured["volumes"] == {str(multiswerl.GOCACHE_HOST_DIR): multiswerl._GOCACHE_CONTAINER_DIR}
    assert "/go/pkg/mod" not in captured["volumes"].values()


def test_grade_touches_the_image_even_if_run_raises(monkeypatch):
    def raising_run(*args, **kwargs):
        raise RuntimeError("boom")

    touched = []
    monkeypatch.setattr(dockerexec, "run", raising_run)
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: touched.append(image))

    task = _task()
    try:
        multiswerl.grade(task, "diff --git a/fix.go b/fix.go\n+fix\n")
    except RuntimeError:
        pass

    assert touched == [multiswerl._image(task)]


def test_grade_script_reports_fail_not_harness_when_candidate_patch_wont_apply(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds, volumes=None: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    multiswerl.grade(_task(), "diff --git a/fix.go b/fix.go\n+fix\n")

    assert "candidate patch failed to apply" in captured["script"]
    assert "HARNESS:candidate" not in captured["script"]


def test_setup_script_reports_harness_for_test_patch_apply_failure():
    script = multiswerl._setup_script(_task(), "nonce")
    assert dockerexec.report_cmd("nonce", "HARNESS", "test_patch failed to apply") in script
    assert "git apply /tmp/test.patch" in script
    assert "cd /home/gin" in script


def test_grade_reference_applies_the_fix_patch_and_reports_harness_on_failure(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds, volumes=None: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    multiswerl.grade_reference(_task())

    assert "fix patch failed to apply" in captured["script"]
    assert "git apply /tmp/fix.patch" in captured["script"]


def test_grade_null_applies_no_fix_patch(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds, volumes=None: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    multiswerl.grade_null(_task())

    assert "/tmp/fix.patch" not in captured["script"]
    assert "/tmp/candidate.patch" not in captured["script"]
    assert "/tmp/test.patch" in captured["script"]


# --- staged grading -----------------------------------------------------------------------------

def test_run_test_stage_reports_pass_fail_and_harness_off_exit_code_and_run_count():
    script = multiswerl._run_test_stage(_task(), "nonce", ["TestFoo"], "mystage", on_pass="NEXT\n")
    assert "go test ./..." in script
    assert dockerexec.report_cmd("nonce", "FAIL", "mystage tests failed") in script
    assert dockerexec.report_cmd("nonce", "HARNESS", "no mystage tests matched the expected names") in script
    # Regression guard: `go test -run` exits 0 even when the pattern matches nothing (confirmed
    # live this session, unlike pytest which errors loudly on an unknown node id) — the script
    # must gate on the `=== RUN` count, not exit code alone.
    assert "RUN_COUNT" in script
    assert "=== RUN" in script
    # A passing stage must fall through to whatever comes next, verbatim.
    assert script.endswith("NEXT\n")


def test_run_test_stage_builds_a_safe_anchored_pattern_with_no_raw_special_characters():
    # The names passed in are always already-top-level identifiers by the time they reach here —
    # this just confirms the OR-pattern itself is built cleanly around them. The pattern travels
    # into the container base64-encoded (dockerexec.write_file_cmd), not as a raw substring.
    script = multiswerl._run_test_stage(_task(), "nonce", ["TestA", "TestB"], "stage", on_pass="")
    assert _b64("^(TestA|TestB)$") in script


def test_discriminating_stage_reports_harness_when_no_discriminating_tests_exist():
    task = _task(f2p_tests={}, n2p_tests={}, s2p_tests={})
    script = multiswerl._discriminating_stage_script(task, "nonce", on_pass="UNREACHABLE")
    assert dockerexec.report_cmd("nonce", "HARNESS", "no discriminating tests found on this row") in script
    assert "UNREACHABLE" not in script  # never falls through on the empty-set case


def test_regression_guard_stage_falls_straight_through_when_p2p_is_empty():
    # Unlike the discriminating stage, an empty regression-guard set is normal (some rows have
    # none) — nothing to check, so it must be a silent pass-through, not a HARNESS.
    task = _task(p2p_tests={})
    script = multiswerl._regression_guard_stage_script(task, "nonce", on_pass="NEXT_STAGE")
    assert script == "NEXT_STAGE"


def test_grade_null_never_runs_the_regression_guard_stage(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds, volumes=None: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    multiswerl.grade_null(_task())

    # p2p_tests' names must never appear in null's script — checking them adds cost, not signal,
    # since they're defined as passing both before and after the fix, and null IS the "before"
    # state. Only the discriminating stage should run. Patterns are base64-encoded in transit
    # (dockerexec.write_file_cmd), so check for the encoded form, not a raw substring.
    assert _b64("^(TestDisableBindValidation|TestPostingsForMatchers)$") not in captured["script"]
    assert "regression_guard" not in captured["script"]
    assert _b64("^(TestMappingMultipleDefaultWithCollectionFormat)$") in captured["script"]


def test_grade_runs_the_discriminating_stage_before_the_regression_guard_stage(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds, volumes=None: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    multiswerl.grade(_task(), "diff --git a/fix.go b/fix.go\n+fix\n")

    script = captured["script"]
    assert _b64("^(TestMappingMultipleDefaultWithCollectionFormat)$") in script
    assert _b64("^(TestDisableBindValidation|TestPostingsForMatchers)$") in script
    # Nesting order matters: the discriminating stage's script must appear before the
    # regression-guard stage's, since a FAIL in the first must short-circuit before the second
    # ever runs (report_cmd's `exit 0` on the FAIL/HARNESS branch enforces this at runtime).
    assert script.index("discriminating") < script.index("regression_guard")


def test_grade_reference_runs_both_stages(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds, volumes=None: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    multiswerl.grade_reference(_task())

    script = captured["script"]
    assert _b64("^(TestMappingMultipleDefaultWithCollectionFormat)$") in script
    assert _b64("^(TestDisableBindValidation|TestPostingsForMatchers)$") in script
    assert script.index("discriminating") < script.index("regression_guard")


# --- JS/TS: whole-suite grading ---------------------------------------------------------------
# _REPO_CONFIG's build/test commands are copied verbatim from the official multi_swe_bench
# harness's own per-repo source (see multiswerl.py's module docstring) and confirmed live against
# two real images (express, zod) — not guessed, so these tests check OUR dispatch/script-building
# logic around that table, not the commands' own correctness.

def _js_task(**row_overrides) -> Task:
    row = {
        "org": "colinhacks", "repo": "zod", "number": 3887,
        "base": {"sha": "deadbeef"},
        "fix_patch": "diff --git a/fix.ts b/fix.ts\n+fix\n",
        "test_patch": "diff --git a/fix.test.ts b/fix.test.ts\n+test\n",
        "f2p_tests": {"src/__tests__/string.test.ts": {"fix": "PASS", "test": "FAIL", "run": "NONE"}},
        "n2p_tests": {}, "s2p_tests": {},
        "p2p_tests": {"src/__tests__/validations.test.ts": {"fix": "PASS", "test": "PASS", "run": "PASS"}},
    }
    row.update(row_overrides)
    return Task(task_id="multi-swe-rl:colinhacks__zod-3887", source="multi-swe-rl", prompt="fix the bug", reference_solution="", row=row)


def test_js_ts_config_lookup_is_keyed_by_org_and_repo():
    assert multiswerl._js_ts_config(_js_task()) is not None
    assert multiswerl._js_ts_config(_task()) is None  # the Go fixture from above


def test_js_ts_test_script_runs_build_then_test_and_reports_harness_on_build_failure():
    task = _js_task()  # zod: has a build step
    config = multiswerl._js_ts_config(task)
    script = multiswerl._js_ts_test_script(task, "nonce", config, on_pass="NEXT\n")
    assert "cd /home/zod && yarn build ||" in script
    assert "cd /home/zod && yarn test >" in script
    assert dockerexec.report_cmd("nonce", "HARNESS", "build step failed") in script
    assert dockerexec.report_cmd("nonce", "FAIL", "tests failed") in script
    assert script.endswith("NEXT\n")
    assert script.index("yarn build") < script.index("yarn test")


def test_js_ts_test_script_skips_the_build_step_when_the_repo_has_none():
    task = _js_task(org="Automattic", repo="mongoose", number=1)  # mongoose: no build step
    config = multiswerl._js_ts_config(task)
    script = multiswerl._js_ts_test_script(task, "nonce", config, on_pass="")
    assert "build" not in script
    assert "cd /home/mongoose && npm test >" in script


def test_js_ts_test_script_gates_on_the_test_commands_exit_code_alone():
    # Unlike Go, there's no per-name "zero tests matched" guard here — the confirmed harness
    # commands themselves give nothing more granular than whole-suite pass/fail (see module
    # docstring), so a non-zero exit is always FAIL, with no HARNESS branch for the test step.
    task = _js_task(org="expressjs", repo="express", number=1)
    config = multiswerl._js_ts_config(task)
    script = multiswerl._js_ts_test_script(task, "nonce", config, on_pass="")
    assert 'if [ "$TEST_EXIT" -ne 0 ]; then' in script
    assert dockerexec.report_cmd("nonce", "FAIL", "tests failed") in script


def test_test_stage_script_dispatches_js_ts_tasks_to_the_whole_suite_run():
    task = _js_task()
    script = multiswerl._test_stage_script(task, "nonce", on_pass="NEXT\n")
    assert "yarn build" in script and "yarn test" in script


def test_test_stage_script_dispatches_go_tasks_to_the_staged_run():
    script = multiswerl._test_stage_script(_task(), "nonce", on_pass="NEXT\n")
    assert "go test ./..." in script
    assert script.index("discriminating") < script.index("regression_guard")


def test_grade_runs_the_confirmed_js_ts_command_and_mounts_no_cache_volume(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds, volumes=None: captured.update(script=script, volumes=volumes))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    multiswerl.grade(_js_task(), "diff --git a/fix.ts b/fix.ts\n+fix\n")

    assert "yarn build" in captured["script"] and "yarn test" in captured["script"]
    # No Go build-cache mount for JS/TS — see _volumes' docstring for why nothing is mounted.
    assert captured["volumes"] is None


def test_grade_still_mounts_the_go_cache_for_go_tasks(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds, volumes=None: captured.update(volumes=volumes))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    multiswerl.grade(_task(), "diff --git a/fix.go b/fix.go\n+fix\n")

    assert captured["volumes"] == {str(multiswerl.GOCACHE_HOST_DIR): multiswerl._GOCACHE_CONTAINER_DIR}


def test_grade_null_never_runs_the_go_regression_guard_stage_for_go_but_still_runs_js_ts_whole_suite(monkeypatch):
    captured = {}
    monkeypatch.setattr(dockerexec, "run", lambda image, script, nonce, timeout_seconds, volumes=None: captured.update(script=script))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)

    multiswerl.grade_null(_js_task())

    # JS/TS has no separate regression-guard stage to skip — the one whole-suite run already
    # covers it (see module docstring), so build+test must both still appear.
    assert "yarn build" in captured["script"] and "yarn test" in captured["script"]
