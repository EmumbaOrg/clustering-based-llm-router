import subprocess

import pytest

from router.common.config import ModelConfig
from router.pipeline.calibration import runner as runner_module
from router.pipeline.calibration.grading.base import Task
from router.pipeline.calibration.runner import run_pi


def _task() -> Task:
    return Task(task_id="bigcodebench:0", source="bigcodebench", prompt="do the thing", reference_solution="", row={})


def _swesmith_task() -> Task:
    return Task(
        task_id="oauthlib__oauthlib.1fd52536", source="swe-smith", prompt="fix the bug",
        reference_solution="", row={"repo": "swesmith/oauthlib__oauthlib.1fd52536"},
    )


def _model() -> ModelConfig:
    return ModelConfig(
        model_id="m", provider="openai", runner="pi", cost_input=0, cost_output=0,
        context_window=0, max_tokens=0,
    )


def test_run_pi_returns_the_failure_detail_on_a_nonzero_exit(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="connection refused")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    result = run_pi(_task(), _model(), timeout_seconds=60)

    assert result.solution is None
    assert "connection refused" in result.detail


# --- repo-context wiring ---------------------------------------------------------------------

def test_run_pi_skips_repo_context_entirely_for_sources_without_it(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(args, 0, stdout="```python\nreturn 1\n```", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(runner_module.repo_context, "extract_diff", lambda wt: pytest.fail("should not be called"))

    result = run_pi(_task(), _model(), timeout_seconds=60)

    assert captured["cwd"] is None
    assert result.solution == "return 1"


def test_run_pi_returns_context_unavailable_without_calling_pi_when_clone_fails(monkeypatch):
    def fake_run(args, **kwargs):
        pytest.fail("pi should never be invoked when repo context setup fails")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner_module.repo_context, "ensure_cached_clone",
        lambda url: (_ for _ in ()).throw(runner_module.repo_context.RepoContextError("clone failed")),
    )

    result = run_pi(_swesmith_task(), _model(), timeout_seconds=60)

    assert result.context_unavailable is True
    assert result.solution is None


def test_run_pi_passes_the_worktree_as_cwd_when_repo_context_is_available(monkeypatch, tmp_path):
    cached_clone = tmp_path / "cached_clone"
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    captured = {}
    removed = []

    monkeypatch.setattr(runner_module.repo_context, "ensure_cached_clone", lambda url: cached_clone)
    monkeypatch.setattr(runner_module.repo_context, "checkout_worktree", lambda clone, ref: worktree)
    monkeypatch.setattr(runner_module.repo_context, "extract_diff", lambda wt: "some diff")
    monkeypatch.setattr(
        runner_module.repo_context, "remove_worktree",
        lambda clone, wt: removed.append((clone, wt)),
    )

    def fake_run(args, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(args, 0, stdout="explored and fixed it", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    result = run_pi(_swesmith_task(), _model(), timeout_seconds=60)

    assert captured["cwd"] == worktree
    assert result.solution == "some diff"
    assert removed == [(cached_clone, worktree)]  # cleaned up after use


def test_run_pi_falls_back_to_text_extraction_when_the_worktree_is_clean(monkeypatch, tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    monkeypatch.setattr(runner_module.repo_context, "ensure_cached_clone", lambda url: tmp_path)
    monkeypatch.setattr(runner_module.repo_context, "checkout_worktree", lambda clone, ref: worktree)
    monkeypatch.setattr(runner_module.repo_context, "extract_diff", lambda wt: None)
    monkeypatch.setattr(runner_module.repo_context, "remove_worktree", lambda clone, wt: None)

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="```diff\nsome hand-written diff\n```", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    result = run_pi(_swesmith_task(), _model(), timeout_seconds=60)

    assert result.solution == "some hand-written diff"


def test_run_pi_cleans_up_the_worktree_even_when_pi_times_out(monkeypatch, tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    removed = []

    monkeypatch.setattr(runner_module.repo_context, "ensure_cached_clone", lambda url: tmp_path)
    monkeypatch.setattr(runner_module.repo_context, "checkout_worktree", lambda clone, ref: worktree)
    monkeypatch.setattr(
        runner_module.repo_context, "remove_worktree",
        lambda clone, wt: removed.append(wt),
    )

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    result = run_pi(_swesmith_task(), _model(), timeout_seconds=60)

    assert result.solution is None
    assert result.timed_out is True
    assert removed == [worktree]


# --- provider-level errors (see runner._parse_json_stream's NONCE_PLACEHOLDER-adjacent docstring) --

def test_run_pi_flags_a_provider_error_as_harness_error_not_no_solution(monkeypatch):
    # pi exits 0 even when the underlying provider call itself failed — the failure is only visible
    # as `stopReason: "error"` on the last assistant message in the JSON event stream, and must map
    # to error_harness (excluded), not error_no_solution (counts against the model).
    stdout = (
        '{"type":"session"}\n'
        '{"type":"agent_end","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},'
        '{"role":"assistant","content":[],"stopReason":"error",'
        '"errorMessage":"401 {\\"type\\":\\"error\\",\\"error\\":{\\"message\\":\\"API key is invalid.\\"}}"}]}\n'
    )

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    result = run_pi(_task(), _model(), timeout_seconds=60)

    assert result.solution is None
    assert result.harness_error is True
    assert "API key is invalid" in result.detail


def test_run_pi_does_not_flag_a_normal_completion_as_a_harness_error(monkeypatch):
    stdout = (
        '{"type":"session"}\n'
        '{"type":"agent_end","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},'
        '{"role":"assistant","content":[{"type":"text","text":"```python\\nreturn 1\\n```"}],'
        '"stopReason":"stop"}]}\n'
    )

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    result = run_pi(_task(), _model(), timeout_seconds=60)

    assert result.harness_error is False


# --- turn_count: how many agent turns Pi took, read off the same assistant-message list the usage
# totals are summed across (see TokenUsage.turn_count and _parse_json_stream's own docstring on why
# usage is per-message, not cumulative). ------------------------------------------------------------

def test_run_pi_reports_turn_count_one_for_a_single_turn_call(monkeypatch):
    stdout = (
        '{"type":"session"}\n'
        '{"type":"agent_end","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},'
        '{"role":"assistant","content":[{"type":"text","text":"```python\\nreturn 1\\n```"}],'
        '"stopReason":"stop","usage":{"input":10,"output":5,"cost":{"total":0.001}}}]}\n'
    )

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    result = run_pi(_task(), _model(), timeout_seconds=60)

    assert result.usage.turn_count == 1


def test_run_pi_reports_turn_count_for_a_multi_turn_tool_using_call(monkeypatch):
    # Mirrors the real 6-turn repo-context call described in _parse_json_stream's own docstring —
    # each assistant message is one turn, and turn_count must reflect ALL of them, not just the last.
    assistant_turns = ",".join(
        '{"role":"assistant","content":[{"type":"text","text":"turn"}],'
        '"stopReason":"stop","usage":{"input":3,"output":2,"cost":{"total":0.0001}}}'
        for _ in range(3)
    )
    stdout = (
        '{"type":"session"}\n'
        f'{{"type":"agent_end","messages":[{{"role":"user","content":[{{"type":"text","text":"hi"}}]}},{assistant_turns}]}}\n'
    )

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    result = run_pi(_task(), _model(), timeout_seconds=60)

    assert result.usage.turn_count == 3
    assert result.usage.input_tokens == 9  # summed across all 3 turns, not just the last
