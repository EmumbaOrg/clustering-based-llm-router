import json

from router.pipeline.calibration import repo_context
from router.pipeline.calibration import tasks as tasks_mod
from router.pipeline.calibration.grading import dockerexec, multiswerl
from router.pipeline.calibration.grading.base import GradeResult

# A realistic full row: everything the grader needs, PLUS the four fields measured at 796MB /57% of
# retained bytes across the Go slice that nothing in the grading path reads.
_FULL_ROW = {
    "instance_id": "gin-gonic__gin-4048", "org": "gin-gonic", "repo": "gin", "number": 4048,
    "base": {"sha": "28e57f58b184b2305ace192e02496bb89f6fd8cb", "ref": "master"},
    "fix_patch": "diff --git a/fix.go b/fix.go\n+fix\n",
    "test_patch": "diff --git a/fix_test.go b/fix_test.go\n+test\n",
    "f2p_tests": {"TestOne": {"fix": "PASS", "test": "FAIL", "run": "NONE"}},
    "n2p_tests": {}, "s2p_tests": {},
    "p2p_tests": {"TestTwo/sub!~\"(1|2)\"": {"fix": "PASS", "test": "PASS", "run": "PASS"}},
    "title": "a bug", "body": "b" * 100, "state": "closed",
    "resolved_issues": [{"title": "real issue title", "body": "c" * 200}],
    # The dead weight:
    "fix_patch_result": {"huge": "x" * 5000}, "run_result": {"huge": "y" * 5000},
    "test_patch_result": {"huge": "z" * 5000}, "fixed_tests": {"huge": "w" * 5000},
}


def _load_one_trimmed_task(monkeypatch, tmp_path):
    batch = tasks_mod.MULTI_SWE_RL_BATCH
    monkeypatch.setattr(
        tasks_mod, "_multi_swe_rl_repo_files",
        lambda: [(f"{batch}/go/gin-gonic__gin_dataset.jsonl", 100)],
    )
    path = tmp_path / "gin.jsonl"
    path.write_text(json.dumps(_FULL_ROW) + "\n")
    monkeypatch.setattr(tasks_mod, "hf_hub_download", lambda hf_id, filename, repo_type: str(path))
    tasks = tasks_mod._multi_swe_rl_tasks()
    assert len(tasks) == 1
    return tasks[0]


def test_multi_swe_rl_rows_drop_the_unused_execution_log_fields(monkeypatch, tmp_path):
    task = _load_one_trimmed_task(monkeypatch, tmp_path)
    for dead in ("fix_patch_result", "run_result", "test_patch_result", "fixed_tests"):
        assert dead not in task.row, f"{dead} is dead weight and must not be retained"
    # The prompt is extracted at load time, so the text fields it came from aren't needed after.
    assert "resolved_issues" not in task.row
    assert "real issue title" in task.prompt  # ...but the prompt genuinely carries the issue text


def test_trimmed_row_still_satisfies_every_real_grader_accessor(monkeypatch, tmp_path):
    # The allowlist is a correctness risk, not merely an optimization: drop a field some grader
    # reads and grading breaks at runtime, deep into a multi-hour run. So exercise the REAL
    # accessors against an actually-trimmed row rather than asserting against a hand-written set.
    task = _load_one_trimmed_task(monkeypatch, tmp_path)

    assert multiswerl._image(task) == "mswebench/gin-gonic_m_gin:pr-4048"
    assert multiswerl._repo_dir(task) == "/home/gin"
    assert multiswerl._discriminating_test_names(task) == ["TestOne"]
    assert multiswerl._regression_guard_test_names(task) == ["TestTwo"]
    assert "git apply /tmp/test.patch" in multiswerl._setup_script(task, "nonce")
    assert "go test ./..." in multiswerl._run_test_stage(task, "nonce", ["TestOne"], "stage", on_pass="")
    assert repo_context.remote_and_ref(task) == (
        "https://github.com/gin-gonic/gin.git", "28e57f58b184b2305ace192e02496bb89f6fd8cb",
    )

    # All three grading modes must build their scripts without a KeyError on a trimmed row.
    monkeypatch.setattr(dockerexec, "run", lambda *a, **k: GradeResult(outcome="pass"))
    monkeypatch.setattr(dockerexec, "touch_image", lambda image: None)
    assert multiswerl.grade(task, "diff --git a/x b/x\n+x\n").outcome == "pass"
    assert multiswerl.grade_reference(task).outcome == "pass"
    assert multiswerl.grade_null(task).outcome == "pass"


def test_multi_swe_rl_tasks_reads_go_js_ts_files_only_and_tags_rows_as_multi_swe_rl(monkeypatch, tmp_path):
    batch = tasks_mod.MULTI_SWE_RL_BATCH
    all_files = [
        (f"{batch}/go/gin-gonic__gin_dataset.jsonl", 100),
        (f"{batch}/rust/BurntSushi__ripgrep_dataset.jsonl", 100),  # not yet gradeable — must be skipped
        (f"{batch}/java/mockito__mockito_dataset.jsonl", 100),  # not yet gradeable — must be skipped
        (f"{batch}/js/colinhacks__zod_dataset.jsonl", 100),
        (f"{batch}/ts/vuejs__core_dataset.jsonl", 100),
        (f"{batch}/multi_swe_bench_discarded_instances.jsonl", 100),  # excluded suffix, see corpus.py
    ]
    monkeypatch.setattr(tasks_mod, "_multi_swe_rl_repo_files", lambda: all_files)

    go_row = {
        "instance_id": "gin-gonic__gin-4048", "org": "gin-gonic", "repo": "gin", "number": 4048,
        "resolved_issues": [{"title": "bug title here", "body": "a" * 100}],
    }
    js_row = {
        "instance_id": "colinhacks__zod-3887", "org": "colinhacks", "repo": "zod", "number": 3887,
        "resolved_issues": [{"title": "another bug", "body": "b" * 100}],
    }
    ts_row = {
        "instance_id": "vuejs__core-100", "org": "vuejs", "repo": "core", "number": 100,
        "resolved_issues": [{"title": "yet another bug", "body": "c" * 100}],
    }
    go_path = tmp_path / "gin.jsonl"
    go_path.write_text(json.dumps(go_row) + "\n")
    js_path = tmp_path / "zod.jsonl"
    js_path.write_text(json.dumps(js_row) + "\n")
    ts_path = tmp_path / "core.jsonl"
    ts_path.write_text(json.dumps(ts_row) + "\n")

    requested_paths = []

    def fake_download(hf_id, filename, repo_type):
        requested_paths.append(filename)
        if "gin-gonic" in filename:
            return str(go_path)
        if "zod" in filename:
            return str(js_path)
        if "vuejs" in filename:
            return str(ts_path)
        raise AssertionError(f"should never be called for a non-go/js/ts file: {filename}")

    monkeypatch.setattr(tasks_mod, "hf_hub_download", fake_download)

    result = tasks_mod._multi_swe_rl_tasks()

    assert requested_paths == [
        f"{batch}/go/gin-gonic__gin_dataset.jsonl",
        f"{batch}/js/colinhacks__zod_dataset.jsonl",
        f"{batch}/ts/vuejs__core_dataset.jsonl",
    ]
    assert {t.task_id for t in result} == {
        "multi-swe-rl:gin-gonic__gin-4048", "multi-swe-rl:colinhacks__zod-3887", "multi-swe-rl:vuejs__core-100",
    }
    assert all(t.source == "multi-swe-rl" for t in result)
