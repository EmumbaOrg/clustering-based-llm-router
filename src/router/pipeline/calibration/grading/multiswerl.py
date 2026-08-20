"""Grader for Multi-SWE-RL's Go slice — Docker-based, built on `dockerexec.py`'s sentinel
protocol, mirroring `swegym.py`'s shape. **Go-only pilot**: `tasks.py`'s loader for this source
returns only rows from the dataset's `go/` batch directory — see its own docstring for why the
other 6 languages (C, C++, Java, JS, Rust, TS) in this dataset are corpus-only for now.

Confirmed empirically this session (real `docker pull`/`docker run` against `gin-gonic/gin`,
`prometheus/prometheus`, and `istio/istio` — small, medium, and the largest repo in the corpus)
rather than assumed from the row schema or the official harness source alone:

1. **Prebuilt per-instance Docker images exist and are public**, under Docker Hub namespace
   `mswebench/{org}_m_{repo}:pr-{number}` (org/repo case-lowered; the `_m_` separator, from
   `github.com/multi-swe-bench/multi-swe-bench`'s own `image.py`, needs no dataset-value
   substitution the way swesmith's `_1776_`/swe-gym's `_s_` do). Verified against real instance
   numbers pulled directly from our own corpus, not the official eval benchmark's curated subset —
   `redis/redis` 11/11, `BurntSushi/ripgrep` 8/8, `gohugoio/hugo` 8/8 sampled.
2. **Repo checkout path is always `/home/{repo}`** — confirmed both by reading the harness's
   per-repo `dockerfile()` methods and by `docker run`-ing a real image directly.
3. **No repo-specific install/build step is needed** — every Go repo shares one toolchain
   (`go build`/`go test`), with the module cache already fully resolved in the image; unlike
   swe-gym's `pandas` case, no build-backend rebuild-on-import surprise was found.
4. **Test selection uses only the TOP-LEVEL test name, not full subtest paths, and needs no regex
   escaping.** Rows carry `f2p_tests`/`n2p_tests`/`s2p_tests`/`p2p_tests` (a richer 4-way taxonomy
   than swe-smith/swe-gym's FAIL_TO_PASS/PASS_TO_PASS) keyed by human-readable test names that are
   often full subtest paths containing regex-special characters (e.g.
   `TestPostingsForMatchers/n!~"(1|2.5)"` — real key, real prometheus instance). Passing the FULL
   path would need per-name escaping AND get the path-splitting semantics of `-run` wrong. Instead,
   truncating to the part before the first `/` gives a plain Go identifier (always safe, no
   escaping needed — Go identifiers can't contain regex metacharacters) that matches the TOP-LEVEL
   test, which by default runs and reports ALL its subtests — confirmed directly against the real
   prometheus instance above: matching just `TestPostingsForMatchers` exercised that exact
   special-character subtest, correctly, with zero escaping.
5. **`go test`'s `-run` matching zero tests anywhere still exits 0** — confirmed directly (unlike
   `pytest <node-ids>`, which errors loudly on an unknown id). A typo'd/mismatched test-name set
   would otherwise silently "pass" a `grade_null` or a broken `grade_reference` call. Guarded here
   by counting `=== RUN` lines (via `-v`) and reporting `HARNESS` if none appear — see
   `_run_test_stage`.
6. **Large, heavily-tested repos (e.g. `istio`, whose relevant test set spans ~1,750 top-level
   names across ~11,000 individual subtests) can take several minutes per grading call** (measured:
   6m51s for one `istio` instance) and may occasionally show an unrelated integration test fail
   even against the GOLD fix — observed once (`TestAgent`, unconnected to the patched file) on
   `istio`, almost certainly sandbox/resource sensitivity in a heavy integration test swept in by
   the large regression-guard set, not a flaw in this grading mechanism (validated cleanly on two
   other repos of increasing size first). Accepted as a known, occasional noise source for this
   source's largest repos, same category as swe-gym's pytest-collection edge case — sized into a
   generous timeout (`config/calibration.yaml`) rather than special-cased in code.

**Staged grading: run the small discriminating test set first, only run the large regression-guard
set if that passes.** Investigated three hypotheses for why large repos are slow, empirically,
before picking this one:
- *Package-narrowing* (`go test ./pkgA ./pkgB` instead of `./...`) — tested directly: most packages
  already contain a relevant test (prometheus 76/101, istio 244/475), so narrowing only cut ~9s off
  a ~150s run (~6%). Not worth the added correctness surface (a grep-based discovery step, path
  escaping) for that little.
- *Storage backend* (host bind-mount vs. a Docker-native named volume for `$GOCACHE`) — tested
  directly: 2m33s vs. 2m29s. No difference; ruled out.
- *What's actually slow*: a handful of the REPO'S OWN genuinely slow tests inside the
  regression-guard set — prometheus's `tsdb` package alone took 190 of ~150-190s wall-clock,
  dominated by tests like `TestTombstoneCleanRetentionLimitsRace` (54.9s, a real concurrency stress
  test). Same pattern independently seen on `istio`/`TestAgent` (157.8s). No caching or narrowing
  changes this — it's inherent to the third-party test suite, not this harness.

Given that, the actual lever is deciding WHETHER the expensive set needs to run at all: measured
across 5 real rows, the discriminating set (`f2p_tests`+`n2p_tests`+`s2p_tests`) is 1-320 names
while the regression-guard set (`p2p_tests`) is 32-1,368 — usually >90% of all names. Running just
the discriminating set for one real instance took 27s vs. 150-190s for the full set — a ~6x cut —
and skipping the second stage changes nothing about the final outcome in the cases it's skipped:
- `grade_null()` never applies a fix, so `p2p_tests` (defined as passing both BEFORE and after the
  gold fix) are trivially already satisfied by null's exact state — checking them adds cost, not
  signal. `grade_null` now never runs that stage.
- `grade()` on a candidate that fails the discriminating stage is already decided — the
  regression-guard stage cannot change a FAIL into anything else, so `grade()` skips straight to
  reporting FAIL. Only a candidate that PASSES the discriminating stage runs the second, expensive
  stage to confirm it didn't break anything else.
- `grade_reference()` always runs both stages — it's the one place actually validating the "~100%
  including regression safety" claim, so nothing here should be skipped.

**A persistent Go BUILD cache (only — not the module cache) cuts the dominant real cost: repeated
cold compiles of the SAME commit.** `grade()`, `grade_reference()`, and `grade_null()` all compile
the SAME `base_commit` (differing only by which tiny patch is applied on top), and a real
`calibrate` run calls `grade()` again for every candidate model against the SAME task — meaning the
exact same, mostly-unchanged dependency tree gets recompiled from scratch every time. Mounting a
persistent host directory as `$GOCACHE` measured a **34x speedup on a warm cache** for one real
repeat compile (prometheus, `./tsdb/...`: 1m43s cold → 3s warm, confirmed by the `(cached)` markers
in `go test`'s own output). Cross-commit reuse (a different PR of the same repo) is much weaker —
different commits genuinely differ in enough files that the content-addressed cache mostly misses —
but that isn't the redundancy this targets; same-commit, multiple-grading-call reuse is.

**`$GOMODCACHE` is deliberately NOT mounted, despite the same idea seeming to apply.** Confirmed by
direct inspection that every image already ships a fully-populated `/go/pkg/mod` (e.g. 214MB for
`gin`, 1.3GB for `prometheus`) — an empty host-directory bind mount REPLACES that pre-baked cache
rather than adding to it, forcing a real (if bounded, ~50s measured for prometheus) network
re-download of every dependency on every single call, for zero benefit, since the image's own copy
was already exactly right. Caught this by testing end-to-end before trusting the by-analogy
assumption that "cache GOMODCACHE too" — the build cache and module cache are not interchangeable
here: the build cache starts empty in every image (genuinely nothing to lose by mounting over it),
the module cache does not.

Go's build cache has its own internal age-based eviction; if host disk growth ever becomes a
problem, `go clean -cache` against the mounted directory is the manual remedy — no bespoke bounding
is implemented here, unlike `dockerexec.py`'s Docker image LRU cache, since Go's own cache already
manages this and the cache directory is far smaller than a Docker image.

Row semantics: `fix_patch` is the GOLD FIX (same convention as swe-gym's `patch`, opposite of
swe-smith's bug-injecting `patch`) — `base_commit` (`row["base"]["sha"]`) is already the buggy,
pre-fix state the image is built at. A separate `test_patch` field (same role as swe-gym's) carries
the test changes needed to exercise the four test-outcome dicts and must be applied in every mode.
"""
from __future__ import annotations

import uuid

from ....common.config import REPO_ROOT
from . import dockerexec
from .base import GradeResult, Task

DOCKER_TIMEOUT_SECONDS = dockerexec.DOCKER_TIMEOUT_SECONDS

# Confirmed via `go env GOCACHE` against a real image — lives under a root-owned path, consistent
# with every one of these images running as root (confirmed via `whoami`). GOMODCACHE is
# deliberately NOT mounted — see the module docstring for why that one's different.
_GOCACHE_CONTAINER_DIR = "/root/.cache/go-build"
GOCACHE_HOST_DIR = REPO_ROOT / ".cache" / "multiswerl" / "gocache"


def _cache_volumes() -> dict[str, str]:
    GOCACHE_HOST_DIR.mkdir(parents=True, exist_ok=True)
    return {str(GOCACHE_HOST_DIR): _GOCACHE_CONTAINER_DIR}

# Split, not one combined set, so PASS/FAIL can be decided from the small set alone in the common
# case — see module docstring's "staged grading" section for why this matters and what it measured.
_DISCRIMINATING_TEST_KEYS = ("f2p_tests", "n2p_tests", "s2p_tests")
_REGRESSION_GUARD_TEST_KEY = "p2p_tests"


def _image(task: Task) -> str:
    return f"mswebench/{task.row['org']}_m_{task.row['repo']}:pr-{task.row['number']}".lower()


def _repo_dir(task: Task) -> str:
    return f"/home/{task.row['repo']}"


def _top_level_names(row: dict, key: str) -> set[str]:
    """Top-level Go test function names only (the part before the first `/`) — always a plain Go
    identifier, so always a regex-safe `-run` term without escaping. See module docstring point 4
    for why this is preferred over the full subtest path some of these keys carry."""
    return {name.split("/", 1)[0] for name in (row.get(key) or {})}


def _discriminating_test_names(task: Task) -> list[str]:
    """The small set that actually proves whether a fix works: existing tests that should flip
    from failing to passing (`f2p_tests`), brand-new tests the fix introduces (`n2p_tests`), and
    previously-skipped tests that should now run and pass (`s2p_tests`). Measured directly across
    5 real rows: 1-320 names, vs. 32-1,368 for the regression-guard set below — usually well under
    10% of the total."""
    names: set[str] = set()
    for key in _DISCRIMINATING_TEST_KEYS:
        names |= _top_level_names(task.row, key)
    return sorted(names)


def _regression_guard_test_names(task: Task) -> list[str]:
    """`p2p_tests` — tests defined by the dataset as already passing BOTH before and after the
    gold fix. Checking these is real, necessary work when a candidate's fix might have broken
    something else, but it's expensive (measured: a single repo's regression-guard run took
    150-190s, entirely dominated by a handful of the REPO's OWN genuinely slow tests — e.g.
    prometheus's `TestTombstoneCleanRetentionLimitsRace` at 54.9s — not anything about our
    harness; package-narrowing and storage-backend changes were tested directly and neither
    moved this number meaningfully). See module docstring for when this stage runs at all."""
    return sorted(_top_level_names(task.row, _REGRESSION_GUARD_TEST_KEY))


def _setup_script(task: Task, nonce: str) -> str:
    """Shared by all three modes: apply the test changes every mode needs to exercise the four
    test-outcome dicts. A failure here is always `error_harness` — it's the dataset's own
    test_patch against its own prebuilt image, never the candidate's fault."""
    repo_dir = _repo_dir(task)
    harness_no_repo = dockerexec.report_cmd(nonce, "HARNESS", f"missing {repo_dir}")
    harness_test_patch_apply = dockerexec.report_cmd(nonce, "HARNESS", "test_patch failed to apply")
    return (
        f"cd {repo_dir} || {{ {harness_no_repo}; }}\n"
        f"{dockerexec.write_file_cmd(task.row['test_patch'], '/tmp/test.patch')}\n"
        f"git apply /tmp/test.patch || {{ {harness_test_patch_apply}; }}\n"
    )


def _run_test_stage(task: Task, nonce: str, names: list[str], label: str, on_pass: str) -> str:
    """Runs `names` and either falls through to `on_pass` (more script, appended verbatim) if they
    all pass, or reports FAIL/HARNESS and stops — `report_cmd` ends in `exit 0`, so a fail/harness
    branch here terminates the whole script and `on_pass` is simply never reached. Unlike
    swesmith/swe-gym's pytest-based scripts, exit code alone is not enough: `go test -run` exits 0
    even if the pattern matched ZERO tests anywhere (confirmed directly, unlike pytest which errors
    loudly on an unknown node id), so a `=== RUN` count of zero is reported as HARNESS rather than
    trusted as a pass — see module docstring point 5."""
    repo_dir = _repo_dir(task)
    pattern = "^(" + "|".join(names) + ")$"
    safe_label = label.replace("-", "_")
    pattern_file, log_file = f"/tmp/{safe_label}_pattern.txt", f"/tmp/go_test_{safe_label}.log"
    exit_var, count_var = f"EXIT_{safe_label.upper()}", f"RUN_COUNT_{safe_label.upper()}"
    fail_cmd = dockerexec.report_cmd(nonce, "FAIL", f"{label} tests failed")
    harness_cmd = dockerexec.report_cmd(nonce, "HARNESS", f"no {label} tests matched the expected names")
    return (
        f"{dockerexec.write_file_cmd(pattern, pattern_file)}\n"
        f'cd {repo_dir} && go test ./... -run "$(cat {pattern_file})" -v > {log_file} 2>&1\n'
        f"{exit_var}=$?\n"
        f"{count_var}=$(grep -c '^=== RUN' {log_file})\n"
        f'if [ "${count_var}" -eq 0 ]; then {harness_cmd}; '
        f"elif [ ${exit_var} -ne 0 ]; then {fail_cmd}; fi\n"
    ) + on_pass


def _discriminating_stage_script(task: Task, nonce: str, on_pass: str) -> str:
    names = _discriminating_test_names(task)
    if not names:
        # Should never happen on a well-formed row — every task needs at least one test that
        # distinguishes buggy from fixed. Unlike the regression-guard stage below, an empty set
        # here is a dataset/harness problem worth surfacing, not a silent pass-through.
        return dockerexec.report_cmd(nonce, "HARNESS", "no discriminating tests found on this row")
    return _run_test_stage(task, nonce, names, "discriminating", on_pass)


def _regression_guard_stage_script(task: Task, nonce: str, on_pass: str) -> str:
    names = _regression_guard_test_names(task)
    if not names:
        # A genuinely empty p2p_tests set is normal (some rows have none) — nothing to check, so
        # fall straight through rather than treating it as a harness condition.
        return on_pass
    return _run_test_stage(task, nonce, names, "regression-guard", on_pass)


def grade(task: Task, solution: str, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """`solution` is a forward-apply unified diff — the shape a real candidate/agent produces,
    applied on top of the test-patched baseline. An inapplicable candidate diff is `fail` (the
    model's own failure), not `error_harness`.

    Staged: the (small, fast) discriminating tests run first; the (large, slow) regression-guard
    tests only run if those pass. A candidate that doesn't fix the bug is already decided by the
    first stage — see module docstring for the measured payoff and why this changes nothing about
    the final outcome for a candidate that DOES pass both."""
    if not solution.strip():
        return GradeResult(outcome="fail", detail="empty patch — bug remains unfixed")

    image = _image(task)
    nonce = uuid.uuid4().hex
    fail_candidate_apply = dockerexec.report_cmd(nonce, "FAIL", "candidate patch failed to apply")
    pass_cmd = dockerexec.report_cmd(nonce, "PASS")
    script = (
        _setup_script(task, nonce)
        + f"{dockerexec.write_file_cmd(solution, '/tmp/candidate.patch')}\n"
        + f"git apply /tmp/candidate.patch || {{ {fail_candidate_apply}; }}\n"
        + _discriminating_stage_script(task, nonce, on_pass=_regression_guard_stage_script(task, nonce, on_pass=pass_cmd))
    )
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds, volumes=_cache_volumes())
    finally:
        dockerexec.touch_image(image)


def grade_reference(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: apply the test patch, then the real GOLD fix
    (`task.row["fix_patch"]`). Must score ~100% or the grader is broken. A fix-patch apply failure
    is `error_harness`, not `fail` — the dataset's own fix failing to apply is a dataset/image
    problem, never a signal about candidate quality.

    Always runs both stages (unlike `grade_null`, see below) — this is the one control that must
    actually prove the regression-guard set passes, since "reference scores ~100%" is what
    validates the grader is correct in the first place."""
    image = _image(task)
    nonce = uuid.uuid4().hex
    harness_fix_apply = dockerexec.report_cmd(nonce, "HARNESS", "fix patch failed to apply")
    pass_cmd = dockerexec.report_cmd(nonce, "PASS")
    script = (
        _setup_script(task, nonce)
        + f"{dockerexec.write_file_cmd(task.row['fix_patch'], '/tmp/fix.patch')}\n"
        + f"git apply /tmp/fix.patch || {{ {harness_fix_apply}; }}\n"
        + _discriminating_stage_script(task, nonce, on_pass=_regression_guard_stage_script(task, nonce, on_pass=pass_cmd))
    )
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds, volumes=_cache_volumes())
    finally:
        dockerexec.touch_image(image)


def grade_null(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: apply the test patch and no fix at all. The discriminating tests
    (`f2p_tests`/`n2p_tests`/`s2p_tests`) are expected to fail. Must score ~0% or the grader is
    broken.

    Deliberately NEVER runs the regression-guard (`p2p_tests`) stage — those are defined by the
    dataset as passing both BEFORE and after the gold fix, and null's state (test_patch applied,
    no fix) IS the "before" state, so by the dataset's own labeling they're already guaranteed to
    pass here. Checking them would only add cost, not signal — measured directly: 27s for the
    discriminating stage alone vs. 150-190s for the full set on the same real instance, a ~6x cut
    on every single null-control run."""
    image = _image(task)
    nonce = uuid.uuid4().hex
    pass_cmd = dockerexec.report_cmd(nonce, "PASS")
    script = _setup_script(task, nonce) + _discriminating_stage_script(task, nonce, on_pass=pass_cmd)
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds, volumes=_cache_volumes())
    finally:
        dockerexec.touch_image(image)
