import subprocess

import pytest

from router.common.config import ModelConfig
from router.pipeline.calibration import runner as runner_module
from router.pipeline.calibration.grading.base import Task
from router.pipeline.calibration.runner import (
    RateLimiter,
    _looks_rate_limited,
    _parse_retry_after_seconds,
    run_pi,
)


class FakeClock:
    """A controllable monotonic clock: advances only when told to, so a test can assert exact
    sleep durations without a real sleep."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _limiter():
    clock = FakeClock()
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)  # simulate time passing while "asleep"

    return RateLimiter(clock=clock, sleep=fake_sleep), clock, sleeps


def test_first_call_never_waits():
    limiter, _clock, sleeps = _limiter()
    limiter.wait("model-a", rpm=30)
    assert sleeps == []


def test_a_call_too_soon_after_the_last_one_sleeps_the_remaining_interval():
    limiter, clock, sleeps = _limiter()
    limiter.wait("model-a", rpm=30)  # min_interval = (60/30)*1.15 = 2.3s
    clock.advance(1.0)  # only 1s has passed
    limiter.wait("model-a", rpm=30)
    assert sleeps == pytest.approx([1.3])


def test_a_call_after_the_interval_has_elapsed_does_not_sleep():
    limiter, clock, sleeps = _limiter()
    limiter.wait("model-a", rpm=30)
    clock.advance(10.0)  # comfortably longer than the ~2.3s interval
    limiter.wait("model-a", rpm=30)
    assert sleeps == []


def test_none_rpm_never_throttles():
    limiter, _clock, sleeps = _limiter()
    limiter.wait("model-a", rpm=None)
    limiter.wait("model-a", rpm=None)
    assert sleeps == []


def test_zero_rpm_never_throttles():
    limiter, _clock, sleeps = _limiter()
    limiter.wait("model-a", rpm=0)
    limiter.wait("model-a", rpm=0)
    assert sleeps == []


def test_different_keys_are_paced_independently():
    limiter, clock, sleeps = _limiter()
    limiter.wait("model-a", rpm=30)
    clock.advance(0.1)
    limiter.wait("model-b", rpm=30)  # first call for this key — no history yet
    assert sleeps == []


def test_looks_rate_limited_detects_common_markers():
    assert _looks_rate_limited("Error: 429 Too Many Requests")
    assert _looks_rate_limited("rate limit exceeded, try again later")
    assert _looks_rate_limited("RATE_LIMIT_EXCEEDED")
    assert _looks_rate_limited("too many requests in this window")


def test_looks_rate_limited_ignores_unrelated_failures():
    assert not _looks_rate_limited("connection refused")
    assert not _looks_rate_limited("500 internal server error")
    assert not _looks_rate_limited("")


def test_parse_retry_after_seconds_parses_plain_seconds():
    assert _parse_retry_after_seconds("Please try again in 20.19s.") == pytest.approx(20.19)


def test_parse_retry_after_seconds_parses_milliseconds():
    assert _parse_retry_after_seconds("Please try again in 780ms.") == pytest.approx(0.78)


def test_parse_retry_after_seconds_returns_none_when_absent():
    assert _parse_retry_after_seconds("connection refused") is None


def _task() -> Task:
    return Task(task_id="bigcodebench:0", source="bigcodebench", prompt="do the thing", reference_solution="", row={})


def _swesmith_task() -> Task:
    return Task(
        task_id="oauthlib__oauthlib.1fd52536", source="swe-smith", prompt="fix the bug",
        reference_solution="", row={"repo": "swesmith/oauthlib__oauthlib.1fd52536"},
    )


def _model() -> ModelConfig:
    # rate_limit_rpm=None so RateLimiter.wait() is a no-op and doesn't add real sleeps of its own.
    return ModelConfig(
        model_id="m", provider="groq", runner="pi", cost_input=0, cost_output=0,
        context_window=0, max_tokens=0, rate_limit_rpm=None,
    )


_RATE_LIMITED_STDERR = (
    '{"error":{"message":"Rate limit reached for tokens per minute (TPM): '
    'Limit 6000, Used 4096, Requested 3923. Please try again in 2.5s.",'
    '"type":"tokens","code":"rate_limit_exceeded"}}'
)


def test_run_pi_retries_a_rate_limited_call_and_succeeds_on_retry(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr=_RATE_LIMITED_STDERR)
        return subprocess.CompletedProcess(args, returncode=0, stdout="```python\nreturn 1\n```", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    sleeps = []
    result = run_pi(_task(), _model(), timeout_seconds=60, sleep=sleeps.append)

    assert result.solution == "return 1"
    assert result.rate_limited is False
    assert len(calls) == 2
    assert sleeps == [pytest.approx(3.0)]  # parsed 2.5s + 0.5s buffer


def test_run_pi_gives_up_after_max_retries_and_marks_rate_limited(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr=_RATE_LIMITED_STDERR)

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    sleeps = []
    result = run_pi(_task(), _model(), timeout_seconds=60, sleep=sleeps.append)

    assert result.rate_limited is True
    assert result.solution is None
    assert len(sleeps) == runner_module.MAX_RATE_LIMIT_RETRIES  # slept before each retry, none after the final failure


def test_run_pi_caps_the_retry_wait_at_the_configured_maximum(monkeypatch):
    stderr = _RATE_LIMITED_STDERR.replace("2.5s", "9999s")

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr=stderr)

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    sleeps = []
    run_pi(_task(), _model(), timeout_seconds=60, sleep=sleeps.append)

    assert max(sleeps) <= runner_module.MAX_RATE_LIMIT_WAIT_SECONDS + 0.5


def test_run_pi_does_not_retry_a_non_rate_limit_failure(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="connection refused")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    result = run_pi(_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

    assert len(calls) == 1
    assert result.rate_limited is False


# --- repo-context wiring ---------------------------------------------------------------------

def test_run_pi_skips_repo_context_entirely_for_sources_without_it(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(args, 0, stdout="```python\nreturn 1\n```", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(runner_module.repo_context, "extract_diff", lambda wt: pytest.fail("should not be called"))

    result = run_pi(_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

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

    result = run_pi(_swesmith_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

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

    result = run_pi(_swesmith_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

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

    result = run_pi(_swesmith_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

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

    result = run_pi(_swesmith_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

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
    result = run_pi(_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

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
    result = run_pi(_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

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
    result = run_pi(_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

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
    result = run_pi(_task(), _model(), timeout_seconds=60, sleep=lambda s: None)

    assert result.usage.turn_count == 3
    assert result.usage.input_tokens == 9  # summed across all 3 turns, not just the last
