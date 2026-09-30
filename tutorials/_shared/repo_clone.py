"""Generic git bare-clone + worktree checkout, shared by the swe-smith, swe-gym, and multi-swe-rl
tutorials — for the optional "check out the real repo to read the buggy code the task describes"
side of those tutorials.

Two-tier caching: a **bare clone** is cached per remote URL under a local cache dir and reused
across calls to the same repo (cloning once, not once per task, is what makes this tractable at
all); a disposable **worktree** per call gives an isolated, clean checkout at a specific ref.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

GIT_TIMEOUT_SECONDS = 120
CLONE_CACHE_DIR = Path(tempfile.gettempdir()) / "router-tutorials" / "repo-clones"


class RepoCloneError(Exception):
    pass


def _clone_path(remote_url: str) -> Path:
    safe = remote_url.replace("://", "_").replace("/", "_")
    return CLONE_CACHE_DIR / safe


def ensure_cached_clone(remote_url: str) -> Path:
    """Bare-clones `remote_url` if not already cached locally; reuses it otherwise."""
    dest = _clone_path(remote_url)
    if dest.exists():
        return dest

    CLONE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["git", "clone", "--bare", remote_url, str(dest)],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise RepoCloneError(f"clone of {remote_url} timed out after {GIT_TIMEOUT_SECONDS}s") from e

    if proc.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)  # don't leave a partial clone cached as if valid
        raise RepoCloneError(f"git clone {remote_url} failed: {proc.stderr[-500:].strip()}")
    return dest


def checkout_worktree(cached_clone: Path, ref: str) -> Path:
    """Creates a disposable, isolated working copy of `cached_clone` at `ref`."""
    worktree = Path(tempfile.mkdtemp(prefix="router-tutorial-worktree-"))
    try:
        proc = subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), ref],
            cwd=cached_clone, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(worktree, ignore_errors=True)
        raise RepoCloneError(f"worktree checkout of {ref} timed out after {GIT_TIMEOUT_SECONDS}s") from e

    if proc.returncode != 0:
        shutil.rmtree(worktree, ignore_errors=True)
        raise RepoCloneError(f"git worktree add {ref} failed: {proc.stderr[-500:].strip()}")
    return worktree


def remove_worktree(cached_clone: Path, worktree: Path) -> None:
    """Best-effort cleanup — never raises."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=cached_clone, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
    except Exception:
        pass
    shutil.rmtree(worktree, ignore_errors=True)
