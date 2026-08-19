"""Gives `runner.run_pi` real repo access for sources whose tasks are "fix a bug in this
repository" rather than a self-contained snippet. Mirrors `grading/dockerexec.py`'s shape and
philosophy — a small, source-agnostic primitive layer, with per-source dispatch living in
`_REPO_SOURCES` — but for git, not Docker.

Two-tier caching, same lesson `dockerexec.touch_image` learned the hard way: a **bare clone** is
cached per repo (keyed by remote URL) and reused across every task/model that touches that repo —
cloning once, not once per call, is what makes this tractable at all. A disposable **worktree** per
call (`git worktree add --detach`) gives each `run_pi` invocation an isolated, clean copy without
re-cloning from network, removed (`remove_worktree`) right after that call finishes regardless of
outcome. The clone cache is bounded the same way `_MAX_CACHED_IMAGES` bounds Docker images — a git
mirror is far smaller than a Docker image, so the cap here is larger.

Also fixes a real (if likely rare) existing gap: before this module existed, `run_pi`'s
`subprocess.run` passed no `cwd` at all, so an agent that tried to use its already-enabled
read/bash/edit tools would be pointed at THIS ROUTER'S OWN REPO, not a sandbox. Every worktree this
module creates is a disposable directory outside this repo.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path

from ...common.config import REPO_ROOT
from .grading.base import Task

logger = logging.getLogger(__name__)

CLONE_DIR = REPO_ROOT / ".cache" / "repo_context" / "clones"
GIT_TIMEOUT_SECONDS = 120

# See dockerexec.py's _MAX_CACHED_IMAGES comment for the reasoning; a bare git mirror is a few
# MB-to-low-hundreds-of-MB, not multiple GB, so a larger cap is affordable here.
_MAX_CACHED_CLONES = 50
_recently_used_clones: OrderedDict[str, None] = OrderedDict()


class RepoContextError(Exception):
    """Raised when repo-context setup (clone/checkout) fails. Caught by `runner.run_pi` and mapped
    to `RunResult(context_unavailable=True)` — never allowed to crash a whole calibration run."""


def _swesmith_remote_and_ref(task: Task) -> tuple[str, str]:
    # task.row["repo"] is already an "org/repo" path, e.g. "swesmith/oauthlib__oauthlib.1fd52536" —
    # a real, public GitHub mirror pinned at the bug-injected commit (confirmed this session; see
    # the repo-context plan). Its default branch is the clean, pre-bug state.
    return f"https://github.com/{task.row['repo']}.git", "HEAD"


def _swegym_remote_and_ref(task: Task) -> tuple[str, str]:
    # Unlike swe-smith's single-commit synthetic mirror, swe-gym's `repo` is a real, unmodified
    # upstream OSS repo (e.g. "getmoto/moto") and `base_commit` a real commit SHA — the buggy,
    # pre-fix state the grading image is also built at (see grading/swegym.py).
    return f"https://github.com/{task.row['repo']}.git", task.row["base_commit"]


_REPO_SOURCES = {"swe-smith": _swesmith_remote_and_ref, "swe-gym": _swegym_remote_and_ref}


def remote_and_ref(task: Task) -> tuple[str, str] | None:
    """None means this source's tasks are self-contained (bigcodebench/ds1000) — no repo context
    to provide, and callers must treat that as a true no-op, not a default that happens to match."""
    resolver = _REPO_SOURCES.get(task.source)
    return resolver(task) if resolver is not None else None


def _clone_path(remote_url: str) -> Path:
    safe = remote_url.replace("://", "_").replace("/", "_")
    return CLONE_DIR / safe


def _touch_clone(dest: Path) -> None:
    key = str(dest)
    if key in _recently_used_clones:
        _recently_used_clones.move_to_end(key)
    else:
        _recently_used_clones[key] = None

    while len(_recently_used_clones) > _MAX_CACHED_CLONES:
        oldest, _ = _recently_used_clones.popitem(last=False)
        _remove_clone(Path(oldest))


def _remove_clone(dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)


def ensure_cached_clone(remote_url: str) -> Path:
    """Bare-clones `remote_url` if not already cached; reuses (and refreshes the recency of) the
    cached clone otherwise."""
    dest = _clone_path(remote_url)
    if dest.exists():
        _touch_clone(dest)
        return dest

    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"cloning {remote_url} -> {dest}")
    try:
        proc = subprocess.run(
            ["git", "clone", "--bare", remote_url, str(dest)],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise RepoContextError(f"clone of {remote_url} timed out after {GIT_TIMEOUT_SECONDS}s") from e

    if proc.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)  # don't leave a partial clone cached as if valid
        raise RepoContextError(f"git clone {remote_url} failed: {proc.stderr[-500:].strip()}")

    _touch_clone(dest)
    return dest


def checkout_worktree(cached_clone: Path, ref: str) -> Path:
    """Creates a disposable, isolated working copy at `ref` — safe for an agent's tools to read,
    run commands in, and edit, without touching the cached clone or any other call's worktree."""
    worktree = Path(tempfile.mkdtemp(prefix="router-repo-context-"))
    try:
        proc = subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), ref],
            cwd=cached_clone, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(worktree, ignore_errors=True)
        raise RepoContextError(f"worktree checkout of {ref} timed out after {GIT_TIMEOUT_SECONDS}s") from e

    if proc.returncode != 0:
        shutil.rmtree(worktree, ignore_errors=True)
        raise RepoContextError(f"git worktree add {ref} failed: {proc.stderr[-500:].strip()}")

    return worktree


def remove_worktree(cached_clone: Path, worktree: Path) -> None:
    """Best-effort, never raises — cleanup failing must not turn a completed (or already-failed)
    `run_pi` call into a crash. Matches `dockerexec.cleanup_image`'s discipline."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=cached_clone, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
    except Exception:
        logger.debug(f"failed to remove worktree {worktree} via git", exc_info=True)
    shutil.rmtree(worktree, ignore_errors=True)


def extract_diff(worktree: Path) -> str | None:
    """`git add -A -N .` stages new files as intent-to-add (so `git diff` reports their full
    content, not just "new file") without actually staging content — then `git diff` captures
    every change an agent's edit/write tools made against the checked-out ref. Preferred over
    parsing the agent's text response: a tool-using agent's actual edits, not its prose description
    of them, are the ground truth for what changed."""
    subprocess.run(
        ["git", "add", "-A", "-N", "."],
        cwd=worktree, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
    )
    proc = subprocess.run(
        ["git", "diff"],
        cwd=worktree, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
    )
    diff = proc.stdout.strip()
    return diff or None
