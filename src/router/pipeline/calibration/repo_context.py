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


def _multiswerl_remote_and_ref(task: Task) -> tuple[str, str]:
    # Same shape as swe-gym's entry — a real, unmodified upstream repo (`org`/`repo` fields) at a
    # real commit SHA (`base.sha`), the buggy pre-fix state the grading image is built at (see
    # grading/multiswerl.py).
    return f"https://github.com/{task.row['org']}/{task.row['repo']}.git", task.row["base"]["sha"]


_REPO_SOURCES = {
    "swe-smith": _swesmith_remote_and_ref,
    "swe-gym": _swegym_remote_and_ref,
    "multi-swe-rl": _multiswerl_remote_and_ref,
}


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


def _inject_swesmith_bug(worktree: Path, task: Task) -> None:
    """swe-smith's mirror repos (`_swesmith_remote_and_ref`) check out at the CLEAN, pre-bug
    commit — `task.row["patch"]` is the bug-injection diff that `grading/swesmith.py`'s Docker
    setup applies forward before anything else happens. Without doing the same here, an agent
    explores/edits code that doesn't have the bug its `problem_statement` describes, and writes a
    diff against the wrong baseline — confirmed empirically this session: such a diff applies
    cleanly against the clean worktree, then fails against the actually-buggy grading container,
    which is exactly the "candidate patch failed to apply" failure seen across every model on
    swe-smith tasks (not a model-quality or tool-use problem at all).

    Committed, not left as an uncommitted working-tree change: `extract_diff()` does `git diff`
    against HEAD, and must only ever capture the AGENT's own edits — if this injection stayed
    unstaged, it would be indistinguishable from the agent's own changes and get folded into
    `extract_diff()`'s output, corrupting it with a copy of the very step grading already applies
    on its own inside Docker. Committing moves HEAD to "bug injected" as the new baseline, so a
    later `git diff` reports only what the agent does on top of that."""
    patch = task.row.get("patch")
    if not patch:
        return
    patch_file = worktree / ".router-bug-injection.patch"
    patch_file.write_text(patch)
    try:
        proc = subprocess.run(
            ["git", "apply", str(patch_file)],
            cwd=worktree, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
    finally:
        patch_file.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RepoContextError(f"swe-smith bug-injection patch failed to apply: {proc.stderr[-500:].strip()}")

    subprocess.run(["git", "add", "-A"], cwd=worktree, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False)
    subprocess.run(
        # -c user.*  inline rather than relying on any global/repo git config being present — this
        # commit is purely an internal baseline marker, never pushed or attributed to anyone.
        ["git", "-c", "user.email=router@localhost", "-c", "user.name=router",
         "commit", "--no-verify", "-m", "router: inject swe-smith bug (pre-task baseline)"],
        cwd=worktree, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
    )


# Per-source post-checkout setup beyond the plain `git worktree add` — only swe-smith needs one
# today (see `_inject_swesmith_bug`); swe-gym and multi-swe-rl check out a real historical pre-fix
# commit directly (confirmed empirically this session: their own gold fix patches apply forward
# cleanly against the checked-out worktree, meaning the buggy code is already there — no injection
# needed). A source with no entry here is a deliberate no-op, not an oversight.
_POST_CHECKOUT_SETUP = {
    "swe-smith": _inject_swesmith_bug,
}


def apply_post_checkout_setup(worktree: Path, task: Task) -> None:
    setup = _POST_CHECKOUT_SETUP.get(task.source)
    if setup is not None:
        setup(worktree, task)


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
    # `.strip()` on the RETURNED value (not just to test for emptiness) used to eat the trailing
    # newline every valid unified diff needs after its last line — confirmed empirically this
    # session: `git apply` rejects such a diff outright with "corrupt patch", regardless of whether
    # the underlying edit was correct. `proc.stdout` itself is returned unmodified when non-empty;
    # `.strip()` is only used here to test for "nothing changed."
    return proc.stdout if proc.stdout.strip() else None
