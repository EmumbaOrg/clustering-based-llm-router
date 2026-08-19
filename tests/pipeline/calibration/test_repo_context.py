import subprocess
from collections import OrderedDict
from pathlib import Path

import pytest

from router.pipeline.calibration import repo_context
from router.pipeline.calibration.grading.base import Task
from router.pipeline.calibration.repo_context import (
    RepoContextError,
    checkout_worktree,
    ensure_cached_clone,
    extract_diff,
    remote_and_ref,
    remove_worktree,
)


@pytest.fixture(autouse=True)
def _isolated_clone_cache(monkeypatch):
    # _touch_clone's LRU set is module-level state — give each test a fresh one so cache contents
    # from one test can't leak into another's eviction assertions.
    monkeypatch.setattr(repo_context, "_recently_used_clones", OrderedDict())


def _swesmith_task(repo: str = "swesmith/oauthlib__oauthlib.1fd52536") -> Task:
    return Task(
        task_id="oauthlib__oauthlib.1fd52536",
        source="swe-smith",
        prompt="fix the bug",
        reference_solution="",
        row={"repo": repo},
    )


def test_remote_and_ref_resolves_swesmith_repo_field():
    result = remote_and_ref(_swesmith_task())
    assert result == ("https://github.com/swesmith/oauthlib__oauthlib.1fd52536.git", "HEAD")


@pytest.mark.parametrize("source", ["bigcodebench", "ds1000"])
def test_remote_and_ref_is_a_true_no_op_for_self_contained_sources(source):
    task = Task(task_id="t", source=source, prompt="p", reference_solution="", row={})
    assert remote_and_ref(task) is None


def test_ensure_cached_clone_clones_once_and_reuses_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_context, "CLONE_DIR", tmp_path)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        dest = Path(args[-1])
        dest.mkdir(parents=True)  # simulate `git clone` creating the destination
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(repo_context.subprocess, "run", fake_run)
    remote_url = "https://github.com/swesmith/oauthlib__oauthlib.1fd52536.git"

    first = ensure_cached_clone(remote_url)
    second = ensure_cached_clone(remote_url)

    assert first == second
    assert len(calls) == 1  # second call reused the cache, no second `git clone`


def test_ensure_cached_clone_raises_on_git_failure_and_does_not_cache_a_partial_clone(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_context, "CLONE_DIR", tmp_path)

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: repository not found")

    monkeypatch.setattr(repo_context.subprocess, "run", fake_run)

    with pytest.raises(RepoContextError):
        ensure_cached_clone("https://github.com/example/missing.git")
    assert not any(tmp_path.iterdir())


def test_ensure_cached_clone_raises_on_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_context, "CLONE_DIR", tmp_path)

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(repo_context.subprocess, "run", fake_run)

    with pytest.raises(RepoContextError):
        ensure_cached_clone("https://github.com/example/slow.git")


def test_checkout_worktree_raises_on_git_failure(monkeypatch, tmp_path):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="fatal: invalid reference: HEAD")

    monkeypatch.setattr(repo_context.subprocess, "run", fake_run)

    with pytest.raises(RepoContextError):
        checkout_worktree(tmp_path, "HEAD")


def test_checkout_worktree_returns_the_created_directory(monkeypatch, tmp_path):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(repo_context.subprocess, "run", fake_run)

    worktree = checkout_worktree(tmp_path, "HEAD")
    assert worktree.exists()
    assert captured["cwd"] == tmp_path
    assert "HEAD" in captured["args"]


def test_remove_worktree_never_raises_even_if_git_fails(monkeypatch, tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()

    def fake_run(args, **kwargs):
        raise OSError("git binary not found")

    monkeypatch.setattr(repo_context.subprocess, "run", fake_run)
    remove_worktree(tmp_path, worktree)  # must not raise
    assert not worktree.exists()  # falls back to a plain directory removal


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def real_git_worktree(tmp_path) -> Path:
    # Real (small, local, throwaway) git state — extract_diff's `git add -A -N .` + `git diff`
    # mechanics are genuinely exercised here, not mocked into meaninglessness.
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["config", "user.name", "Test"], cwd=repo)
    (repo / "existing.py").write_text("value = 1\n")
    _run_git(["add", "existing.py"], cwd=repo)
    _run_git(["commit", "-m", "initial"], cwd=repo)
    return repo


def test_extract_diff_returns_none_when_nothing_changed(real_git_worktree):
    assert extract_diff(real_git_worktree) is None


def test_extract_diff_captures_an_edit_to_a_tracked_file(real_git_worktree):
    (real_git_worktree / "existing.py").write_text("value = 2\n")
    diff = extract_diff(real_git_worktree)
    assert diff is not None
    assert "-value = 1" in diff
    assert "+value = 2" in diff


def test_extract_diff_captures_a_newly_created_file(real_git_worktree):
    (real_git_worktree / "new_file.py").write_text("def fixed():\n    return True\n")
    diff = extract_diff(real_git_worktree)
    assert diff is not None
    assert "new_file.py" in diff
    assert "+def fixed():" in diff
