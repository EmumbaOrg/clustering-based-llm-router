"""Grader for Multi-SWE-RL's Go, JS, TS, Java, and Rust slices — Docker-based, built on
`dockerexec.py`'s sentinel protocol, mirroring `swegym.py`'s shape. C and C++ rows in this dataset
are corpus-only (not gradeable) — see `tasks.py`'s loader docstring for why.

Image tag: `mswebench/{org}_m_{repo}:pr-{number}` (org/repo lower-cased). Repo checkout is always
`/home/{repo}`.

Row semantics: `fix_patch` is the gold fix (`base_commit` is already the pre-fix buggy state the
image is built at); `test_patch` carries the test changes needed to exercise the four
`f2p_tests`/`n2p_tests`/`s2p_tests`/`p2p_tests` outcome dicts and must be applied in every mode.

Go and Rust share one uniform per-language test convention; JS, TS, and Java are handled instead by
`_REPO_CONFIG`, a per-repo table sourced from the official harness's own `run.sh`/`test-run.sh`
rather than guessed from `package.json`/`pom.xml`/`build.gradle` — see "JS/TS/Java command
sourcing" in docs/engineering-notes.md.

Go/Rust specifics (see docs/engineering-notes.md, "Grading — Multi-SWE-RL" section, for the
measurements behind each):
- `go test -run`/`cargo test -- --exact` both exit 0 on zero matched tests — see "Zero test
  matches still exit 0".
- Go test names are truncated to the top-level identifier before use in `-run` patterns — see "Go
  test name truncation for -run patterns".
- Large repos can take several minutes per grading call — see "Large repos can take several
  minutes per grading call".
- Grading is staged: the small discriminating set (f2p/n2p/s2p) runs first, and the large
  regression-guard set (p2p) only runs if that passes (skipped entirely for `grade_null`) — see
  "Staged grading rationale (Go/Rust)".
- Go's build cache (`$GOCACHE`) is mounted as a persistent host volume; the module cache
  (`$GOMODCACHE`) is not — see "Go build cache is mounted, module cache is not".
- Rust needs no per-repo config (`_RUST_REPOS` is just an `_is_rust` dispatch set): every repo uses
  bare `cargo test`.

Other known quirks documented there: "Mongoose flakiness", "Checkstyle baseline noise", "Go's
overall exit code is unreliable", "Lazygit needs a git identity", "Test-name substring collisions".
"""
from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass

from ....common.config import REPO_ROOT
from . import dockerexec
from .base import GradeResult, Task

DOCKER_TIMEOUT_SECONDS = dockerexec.DOCKER_TIMEOUT_SECONDS

# Real path from `go env GOCACHE` in these (root-owned) images. GOMODCACHE is deliberately not
# mounted — see docs/engineering-notes.md, "Go build cache is mounted, module cache is not".
_GOCACHE_CONTAINER_DIR = "/root/.cache/go-build"
GOCACHE_HOST_DIR = REPO_ROOT / ".cache" / "multiswerl" / "gocache"


def _cache_volumes() -> dict[str, str]:
    GOCACHE_HOST_DIR.mkdir(parents=True, exist_ok=True)
    return {str(GOCACHE_HOST_DIR): _GOCACHE_CONTAINER_DIR}

# Split, not one combined set, so PASS/FAIL can be decided from the small set alone in the common
# case — see docs/engineering-notes.md, "Staged grading rationale (Go/Rust)".
_DISCRIMINATING_TEST_KEYS = ("f2p_tests", "n2p_tests", "s2p_tests")
_REGRESSION_GUARD_TEST_KEY = "p2p_tests"


def _image(task: Task) -> str:
    return f"mswebench/{task.row['org']}_m_{task.row['repo']}:pr-{task.row['number']}".lower()


def _repo_dir(task: Task) -> str:
    return f"/home/{task.row['repo']}"


def _top_level_names(row: dict, key: str) -> set[str]:
    """Top-level Go test function name only (before the first `/`) — a plain Go identifier, so a
    regex-safe `-run` term with no escaping needed. See docs/engineering-notes.md, "Go test name
    truncation for -run patterns"."""
    return {name.split("/", 1)[0] for name in (row.get(key) or {})}


def _is_n2p_only_discriminating_set(task: Task) -> bool:
    """True when the discriminating set is entirely `n2p_tests` (brand-new tests the fix itself
    introduces, no `f2p_tests`/`s2p_tests`) — the expected shape for `grade_null` and for most real
    candidates that don't happen to reproduce the gold fix's exact new test name. Used only to
    label that case distinctly in the reported detail; the outcome is still `error_harness` either
    way."""
    f2p = task.row.get("f2p_tests") or {}
    s2p = task.row.get("s2p_tests") or {}
    n2p = task.row.get("n2p_tests") or {}
    return not f2p and not s2p and bool(n2p)


def _discriminating_test_names(task: Task) -> list[str]:
    """The small set that proves whether a fix works: `f2p_tests` (should flip fail→pass),
    `n2p_tests` (brand-new), `s2p_tests` (previously skipped, should now pass)."""
    names: set[str] = set()
    for key in _DISCRIMINATING_TEST_KEYS:
        names |= _top_level_names(task.row, key)
    return sorted(names)


def _regression_guard_test_names(task: Task) -> list[str]:
    """`p2p_tests` — tests already passing both before and after the gold fix. Expensive to check
    (dominated by a handful of genuinely slow tests in the target repo, not this harness) — see
    docs/engineering-notes.md, "Staged grading rationale (Go/Rust)" for when this stage runs."""
    return sorted(_top_level_names(task.row, _REGRESSION_GUARD_TEST_KEY))


def _full_test_names(row: dict, key: str) -> set[str]:
    """The original, untruncated test/subtest names for `key` — used to check each target test's
    own `--- PASS`/`--- FAIL` line after the run, never for the `-run` pattern itself (which needs
    `_top_level_names`'s truncated form instead). Needed because a top-level test with many named
    subtests (e.g. `TestIntegration/foo/bar`) requires the full name to tell our target subtest's
    result apart from an unrelated sibling's."""
    return set(row.get(key) or {})


def _discriminating_full_test_names(task: Task) -> list[str]:
    names: set[str] = set()
    for key in _DISCRIMINATING_TEST_KEYS:
        names |= _full_test_names(task.row, key)
    return sorted(names)


def _regression_guard_full_test_names(task: Task) -> list[str]:
    return sorted(_full_test_names(task.row, _REGRESSION_GUARD_TEST_KEY))


# Every repo here uses the exact same bare `cargo test` command (confirmed from the harness's own
# run.sh), with no per-repo config needed — unlike JS/TS/Java's `_REPO_CONFIG`.
_RUST_REPOS: frozenset[tuple[str, str]] = frozenset({
    ("BurntSushi", "ripgrep"),
    ("alacritty", "alacritty"),
    ("clap-rs", "clap"),
    ("fish-shell", "fish-shell"),
    ("helix-editor", "helix"),
    ("nushell", "nushell"),
    ("rusqlite", "rusqlite"),
    ("rust-lang", "mdBook"),
    ("serde-rs", "serde"),
    ("sharkdp", "bat"),
    ("sharkdp", "fd"),
    ("tokio-rs", "bytes"),
    ("tokio-rs", "tokio"),
    ("tokio-rs", "tracing"),
})


def _is_rust(task: Task) -> bool:
    return (task.row.get("org"), task.row.get("repo")) in _RUST_REPOS


def _rust_test_names(task: Task, keys: tuple[str, ...]) -> list[str]:
    """Rust's test-outcome keys are already exact `cargo test` filter targets
    (`module::submodule::test_name`) — no stripping needed, unlike Go's `_top_level_names`."""
    names: set[str] = set()
    for key in keys:
        names |= set(task.row.get(key) or {})
    return sorted(names)


def _cargo_test_stage(
    task: Task, nonce: str, names: list[str], label: str, on_pass: str, harness_detail_suffix: str = ""
) -> str:
    """Rust's equivalent of `_run_test_stage`: `cargo test -- --exact <name1> <name2> ...` runs
    exactly and only the named tests, OR'd — Rust's equivalent of Go's `-run "^(name1|name2)$"`
    anchor, needing no escaping since `::`-joined paths are always safe identifiers. Also like Go's
    `-run`, it exits 0 on zero matches (see docs/engineering-notes.md, "Zero test matches still
    exit 0"), so this sums `N passed`/`M failed` across every `test result:` line rather than
    trusting the exit code alone.

    `harness_detail_suffix` (non-empty only for the discriminating stage) distinguishes the
    expected n2p-only case from a genuine harness gap in the reported detail; outcome
    classification is unaffected either way."""
    repo_dir = _repo_dir(task)
    names_text = "\n".join(names)
    safe_label = label.replace("-", "_")
    names_file, log_file = f"/tmp/{safe_label}_names.txt", f"/tmp/cargo_test_{safe_label}.log"
    exit_var, count_var = f"EXIT_{safe_label.upper()}", f"RUN_COUNT_{safe_label.upper()}"
    fail_cmd = dockerexec.report_cmd(nonce, "FAIL", f"{label} tests failed")
    harness_cmd = dockerexec.report_cmd(
        nonce, "HARNESS", f"no {label} tests matched the expected names{harness_detail_suffix}"
    )
    return (
        f"{dockerexec.write_file_cmd(names_text, names_file)}\n"
        f"cd {repo_dir} && cargo test -- --exact $(cat {names_file}) > {log_file} 2>&1\n"
        f"{exit_var}=$?\n"
        f"{count_var}=$(grep -oE '[0-9]+ passed; [0-9]+ failed' {log_file} "
        f"| awk '{{sum += $1 + $3}} END {{print sum+0}}')\n"
        f'if [ "${count_var}" -eq 0 ]; then {harness_cmd}; '
        f"elif [ ${exit_var} -ne 0 ]; then {fail_cmd}; fi\n"
    ) + on_pass


@dataclass(frozen=True)
class _RepoConfig:
    """One row per JS/TS/Java repo (keyed by `(org, repo)` in `_REPO_CONFIG`), sourced from the
    official `multi_swe_bench` harness's own per-repo classes (`.../repos/{javascript,typescript,
    java}/{org}/{repo}.py`), not guessed from `package.json`/`pom.xml`/`build.gradle` — see
    docs/engineering-notes.md, "JS/TS/Java command sourcing".

    `build` is a per-grading-call compile step for repos whose tests run against compiled output
    (`zod`, `nuxt`, `react-router`). `test` is the confirmed test invocation. There is no `install`
    field: dependency install (`prepare.sh` / Maven's `~/.m2`) is baked into the image at build
    time, not repeated at grading time.

    A handful of repos have PR-range-specific override classes upstream (e.g. `commander.js`
    migrated from Jest to `node:test` at some point); this table uses the current/default class,
    which may not exactly match every instance's era — a mismatch surfaces as `error_harness`, not
    a silently wrong grade.

    `granularity="exit_code"` (default) trusts the confirmed command's own exit code.
    `granularity="class"` is for Maven repos whose test-outcome keys are exact class names
    (`checkstyle`, `fastjson2`) and reads per-class Surefire XML instead, because the whole-suite
    exit code is unreliable for them — see docs/engineering-notes.md, "Checkstyle baseline noise",
    and `_java_class_test_script`."""
    build: str | None
    test: str
    granularity: str = "exit_code"  # "exit_code" or "class"


_REPO_CONFIG: dict[tuple[str, str], _RepoConfig] = {
    # --- JS ---------------------------------------------------------------------------------
    ("anuraghazra", "github-readme-stats"): _RepoConfig(None, "npm run test -- --verbose"),
    ("Automattic", "mongoose"): _RepoConfig(None, "npm test"),
    ("axios", "axios"): _RepoConfig(None, "npm test -- --reporter console"),
    ("caolan", "async"): _RepoConfig(None, "npm test -- --verbose"),
    ("expressjs", "express"): _RepoConfig(None, "npm run test-ci -- --reporter json"),
    ("google", "zx"): _RepoConfig(None, "npm test -- --reporter=verbose"),
    # Drops the upstream script's own trailing `&& codecov` — a coverage upload with no token in
    # this environment would fail and flip an otherwise-passing run's exit code to non-zero.
    ("iamkun", "dayjs"): _RepoConfig(None, "npm test -- --verbose"),
    ("Kong", "insomnia"): _RepoConfig(None, "npm test -- --verbose"),
    ("sveltejs", "svelte"): _RepoConfig(None, "pnpm test -- --reporter verbose"),
    ("tj", "commander.js"): _RepoConfig(None, "npm test"),
    # --- TS ---------------------------------------------------------------------------------
    ("colinhacks", "zod"): _RepoConfig("yarn build", "yarn test"),
    ("darkreader", "darkreader"): _RepoConfig(
        None, "npm run test:ci -- --json --outputFile=test-results-unit.json"
    ),
    ("mui", "material-ui"): _RepoConfig(None, "yarn run test:unit --reporter json --exit"),
    ("nuxt", "nuxt"): _RepoConfig("pnpm build:stub", "pnpm test:unit -- --verbose && pnpm test:runtime --no-watch"),
    ("reduxjs", "redux"): _RepoConfig(None, "yarn test"),
    ("remix-run", "react-router"): _RepoConfig("yarn build", "yarn test -- --verbose"),
    ("trpc", "trpc"): _RepoConfig(None, "pnpm turbo --filter tests test-ci"),
    ("vuejs", "core"): _RepoConfig(None, "pnpm run test-unit --no-watch --reporter=verbose"),
    # --- Java -------------------------------------------------------------------------------
    # `clean` is part of every command below, so (unlike Go) a full recompile happens on every
    # grading call; no `build` field is needed since `clean test` does compile+test in one step.
    # checkstyle/fastjson2 use granularity="class" (see _RepoConfig's docstring); the 4 Gradle
    # repos key by build task name, not test name, so exit_code is the only option there.
    ("checkstyle", "checkstyle"): _RepoConfig(None, "mvn clean test -Dstyle.color=never", granularity="class"),
    ("mockito", "mockito"): _RepoConfig(None, "./gradlew test"),
    ("elastic", "logstash"): _RepoConfig(None, "./gradlew clean test --continue"),
    ("junit-team", "junit5"): _RepoConfig(None, "./gradlew clean test --continue"),
    ("spotbugs", "spotbugs"): _RepoConfig(None, "./gradlew clean test --continue"),
    ("alibaba", "fastjson2"): _RepoConfig(
        None,
        "./mvnw -V --no-transfer-progress -Pgen-javadoc -Pgen-dokka clean test "
        "-Dsurefire.useFile=false -Dmaven.test.skip=false -DfailIfNoTests=false",
        granularity="class",
    ),
}


def _whole_suite_config(task: Task) -> _RepoConfig | None:
    return _REPO_CONFIG.get((task.row.get("org"), task.row.get("repo")))


def _volumes(task: Task) -> dict[str, str] | None:
    """Go mounts a persistent build cache (see `_cache_volumes`) because the SAME `base_commit`
    gets recompiled from scratch on every grading call — see docs/engineering-notes.md, "Go build
    cache is mounted, module cache is not". JS/TS and Rust mount nothing: their dependency/build
    caches are already baked into the image, and a host-mounted build-output cache for either
    hasn't been measured as worth adding."""
    if _whole_suite_config(task) is not None or _is_rust(task):
        return None
    return _cache_volumes()


def _setup_script(task: Task, nonce: str) -> str:
    """Shared by all three modes: apply the test changes needed to exercise the four test-outcome
    dicts. A failure here is always `error_harness` — it's the dataset's own test_patch against its
    own prebuilt image, never the candidate's fault.

    The `git config --global` is scoped to this disposable `--rm` container's own throwaway
    `~/.gitconfig` (discarded on exit) — needed because some suites shell out to `git commit`
    internally. See docs/engineering-notes.md, "Lazygit needs a git identity"."""
    repo_dir = _repo_dir(task)
    harness_no_repo = dockerexec.report_cmd(nonce, "HARNESS", f"missing {repo_dir}")
    harness_test_patch_apply = dockerexec.report_cmd(nonce, "HARNESS", "test_patch failed to apply")
    return (
        'git config --global user.name "router" && '
        'git config --global user.email "router@localhost"\n'
        f"cd {repo_dir} || {{ {harness_no_repo}; }}\n"
        f"{dockerexec.write_file_cmd(task.row['test_patch'], '/tmp/test.patch')}\n"
        f"git apply /tmp/test.patch || {{ {harness_test_patch_apply}; }}\n"
    )


def _run_test_stage(
    task: Task,
    nonce: str,
    run_names: list[str],
    check_names: list[str],
    label: str,
    on_pass: str,
    harness_detail_suffix: str = "",
) -> str:
    """Runs `run_names` (top-level, regex-safe truncated names) via `-run`, then decides
    PASS/FAIL/HARNESS from each of `check_names`' (the original, untruncated) own `--- PASS:`/
    `--- FAIL:` line in `go test -v`'s output — never from the overall exit code or a blanket
    `=== RUN` count, both of which are unreliable here. See docs/engineering-notes.md, "Go's
    overall exit code is unreliable" and "Lazygit needs a git identity".

    `grep -F -f` (patterns read from a file, matched literally) is used instead of interpolating
    test names into the shell command, since real test names can contain characters that would
    need shell escaping (e.g. `TestPostingsForMatchers/n!~"(1|2.5)"`). `go test -run` still exits 0
    on zero matches (see "Zero test matches still exit 0"), which is what the "not every
    check_name got a PASS" branch below reports as HARNESS.

    `harness_detail_suffix` (non-empty only for the discriminating stage) distinguishes the
    expected n2p-only case from a genuine harness gap in the reported detail; outcome
    classification is unaffected either way."""
    repo_dir = _repo_dir(task)
    pattern = "^(" + "|".join(run_names) + ")$"
    safe_label = label.replace("-", "_")
    pattern_file = f"/tmp/{safe_label}_pattern.txt"
    log_file = f"/tmp/go_test_{safe_label}.log"
    pass_patterns_file = f"/tmp/{safe_label}_pass_patterns.txt"
    fail_patterns_file = f"/tmp/{safe_label}_fail_patterns.txt"
    pass_var, fail_var, total_var = f"PASS_{safe_label.upper()}", f"FAIL_{safe_label.upper()}", f"TOTAL_{safe_label.upper()}"
    fail_cmd = dockerexec.report_cmd(nonce, "FAIL", f"{label} tests failed")
    harness_cmd = dockerexec.report_cmd(
        nonce, "HARNESS", f"no {label} tests matched the expected names{harness_detail_suffix}"
    )

    # Trailing newline on every line (including the last) so `wc -l` counts correctly.
    #
    # Append " (" to each pattern (the text `go test -v` always prints right after a test name) so
    # a shorter name's pattern can't match as a substring of a longer name's result line (e.g.
    # `TestAddTree` vs. `TestAddTree2`) — see docs/engineering-notes.md, "Test-name substring
    # collisions".
    pass_patterns = "".join(f"--- PASS: {name} (\n" for name in check_names)
    fail_patterns = "".join(f"--- FAIL: {name} (\n" for name in check_names)

    return (
        f"{dockerexec.write_file_cmd(pattern, pattern_file)}\n"
        f"{dockerexec.write_file_cmd(pass_patterns, pass_patterns_file)}\n"
        f"{dockerexec.write_file_cmd(fail_patterns, fail_patterns_file)}\n"
        f'cd {repo_dir} && go test ./... -run "$(cat {pattern_file})" -v > {log_file} 2>&1\n'
        f"{total_var}=$(wc -l < {pass_patterns_file})\n"
        f"{pass_var}=$(grep -F -o -f {pass_patterns_file} {log_file} | sort -u | wc -l)\n"
        f"{fail_var}=$(grep -F -o -f {fail_patterns_file} {log_file} | sort -u | wc -l)\n"
        f'if [ "${fail_var}" -gt 0 ]; then {fail_cmd}; '
        f'elif [ "${pass_var}" -ne "${total_var}" ]; then {harness_cmd}; fi\n'
    ) + on_pass


def _discriminating_stage_script(task: Task, nonce: str, on_pass: str) -> str:
    n2p_suffix = " (n2p-only — expected unless this is the gold fix)" if _is_n2p_only_discriminating_set(task) else ""

    if _is_rust(task):
        names = _rust_test_names(task, _DISCRIMINATING_TEST_KEYS)
        if not names:
            return dockerexec.report_cmd(nonce, "HARNESS", "no discriminating tests found on this row")
        return _cargo_test_stage(task, nonce, names, "discriminating", on_pass, harness_detail_suffix=n2p_suffix)

    names = _discriminating_test_names(task)
    if not names:
        # Should never happen on a well-formed row — every task needs at least one test that
        # distinguishes buggy from fixed. Unlike the regression-guard stage below, an empty set
        # here is a dataset/harness problem worth surfacing, not a silent pass-through.
        return dockerexec.report_cmd(nonce, "HARNESS", "no discriminating tests found on this row")
    check_names = _discriminating_full_test_names(task)
    return _run_test_stage(task, nonce, names, check_names, "discriminating", on_pass, harness_detail_suffix=n2p_suffix)


def _regression_guard_stage_script(task: Task, nonce: str, on_pass: str) -> str:
    if _is_rust(task):
        names = _rust_test_names(task, (_REGRESSION_GUARD_TEST_KEY,))
        if not names:
            return on_pass
        return _cargo_test_stage(task, nonce, names, "regression-guard", on_pass)

    names = _regression_guard_test_names(task)
    if not names:
        # A genuinely empty p2p_tests set is normal (some rows have none) — nothing to check, so
        # fall straight through rather than treating it as a harness condition.
        return on_pass
    check_names = _regression_guard_full_test_names(task)
    return _run_test_stage(task, nonce, names, check_names, "regression-guard", on_pass)


def _whole_suite_test_script(task: Task, nonce: str, config: _RepoConfig, on_pass: str) -> str:
    """Runs the confirmed per-repo build step (if any) and test command ONCE, covering the
    discriminating and regression-guard sets in a single pass — see `_RepoConfig`'s docstring for
    why JS/TS doesn't stage the way Go does. A build-step failure is `error_harness` (it doesn't
    depend on the candidate at all beyond whatever source it touches); a non-zero test-command
    exit is `FAIL`, the same "no named-test breakdown available" signal the confirmed harness
    command itself reports (most of these, e.g. mongoose's bare `npm test`, give nothing more
    granular than whole-suite pass/fail either)."""
    repo_dir = _repo_dir(task)
    harness_build = dockerexec.report_cmd(nonce, "HARNESS", "build step failed")
    fail_cmd = dockerexec.report_cmd(nonce, "FAIL", "tests failed")
    lines = []
    if config.build:
        lines.append(f"cd {repo_dir} && {config.build} || {{ {harness_build}; }}")
    lines.append(f"cd {repo_dir} && {config.test} > /tmp/jsts_test.log 2>&1")
    lines.append("TEST_EXIT=$?")
    lines.append(f'if [ "$TEST_EXIT" -ne 0 ]; then {fail_cmd}; fi')
    return "\n".join(lines) + "\n" + on_pass


def _java_class_names(task: Task, keys: tuple[str, ...]) -> list[str]:
    """`checkstyle`/`fastjson2` rows' `f2p_tests`/`n2p_tests`/`s2p_tests`/`p2p_tests` keys are
    already exact fully-qualified Java class names — confirmed directly from real rows, no subtest
    suffix to strip the way Go's keys need (see `_top_level_names`)."""
    names: set[str] = set()
    for key in keys:
        names |= set(task.row.get(key) or {})
    return sorted(names)


def _java_class_ok_snippet(classes: list[str], result_var: str) -> str:
    """Sets shell variable `result_var` to `1` if every one of `classes` has a Maven Surefire XML
    report with zero errors and zero failures, `0` otherwise. Maven writes one
    `target/surefire-reports/TEST-{class}.xml` per test class, its `<testsuite ...>` opening tag
    (confirmed real format, one line) carrying `errors="N"` and `failures="M"` attributes —
    searched with `find` rather than a hardcoded `target/` path since a multi-module Maven project
    nests `target/` under each module. A MISSING report (the class never ran at all, e.g. a typo'd
    class name or a compile failure before tests could run) counts as NOT ok — that's not evidence
    the class passed, the same principle behind Go's `RUN_COUNT` guard."""
    if not classes:
        return f"{result_var}=1\n"
    lines = [f"{result_var}=1"]
    for cls in classes:
        report_name = shlex.quote(f"TEST-{cls}.xml")
        lines.append(
            f'REPORT=$(find . -path "*/surefire-reports/*" -name {report_name} 2>/dev/null | head -1)\n'
            f'[ -n "$REPORT" ] && grep -q \' errors="0"\' "$REPORT" && grep -q \' failures="0"\' "$REPORT" '
            f"|| {result_var}=0"
        )
    return "\n".join(lines) + "\n"


def _java_class_test_script(
    task: Task, nonce: str, config: _RepoConfig, on_pass: str, include_regression_guard: bool
) -> str:
    """`granularity="class"` path (see `_RepoConfig`'s docstring): runs the confirmed command ONCE
    (same as `_whole_suite_test_script`) but reads the outcome from the SPECIFIC named classes'
    Surefire XML reports instead of the overall exit code — immune to unrelated pre-existing
    failures elsewhere in a large suite (confirmed necessary for `checkstyle` specifically).
    `include_regression_guard=False` (from `grade_null`) skips inspecting `p2p_tests`' classes
    entirely, matching Go's semantics; JS/TS's whole-suite path has no such distinction because it
    has no per-class signal to selectively ignore in the first place."""
    repo_dir = _repo_dir(task)
    discriminating = _java_class_names(task, _DISCRIMINATING_TEST_KEYS)
    if not discriminating:
        return dockerexec.report_cmd(nonce, "HARNESS", "no discriminating classes found on this row")
    guard = _java_class_names(task, (_REGRESSION_GUARD_TEST_KEY,)) if include_regression_guard else []

    harness_build = dockerexec.report_cmd(nonce, "HARNESS", "build step failed")
    fail_cmd = dockerexec.report_cmd(nonce, "FAIL", "tests failed")
    lines = []
    if config.build:
        lines.append(f"cd {repo_dir} && {config.build} || {{ {harness_build}; }}")
    lines.append(f"cd {repo_dir} && {config.test} > /tmp/java_test.log 2>&1")
    lines.append(_java_class_ok_snippet(discriminating, "DISC_OK"))
    lines.append(_java_class_ok_snippet(guard, "GUARD_OK"))
    lines.append(f'if [ "$DISC_OK" -eq 0 ] || [ "$GUARD_OK" -eq 0 ]; then {fail_cmd}; fi')
    return "\n".join(lines) + "\n" + on_pass


def _test_stage_script(task: Task, nonce: str, on_pass: str, include_regression_guard: bool = True) -> str:
    """Dispatches to Java's class-based interpretation (`_java_class_test_script`), JS/TS/Gradle
    Java's single whole-suite run (`_whole_suite_test_script` — always covers both test sets in
    one pass, so `include_regression_guard` has no effect there), or Go's discriminating stage,
    optionally followed by the regression-guard stage. `grade_null` passes
    `include_regression_guard=False` for Go — see its own docstring for why."""
    config = _whole_suite_config(task)
    if config is not None:
        if config.granularity == "class":
            return _java_class_test_script(task, nonce, config, on_pass, include_regression_guard)
        return _whole_suite_test_script(task, nonce, config, on_pass)
    if not include_regression_guard:
        return _discriminating_stage_script(task, nonce, on_pass=on_pass)
    return _discriminating_stage_script(task, nonce, on_pass=_regression_guard_stage_script(task, nonce, on_pass=on_pass))


def grade(task: Task, solution: str, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """`solution` is a forward-apply unified diff — the shape a real candidate/agent produces,
    applied on top of the test-patched baseline. An inapplicable candidate diff is `fail` (the
    model's own failure), not `error_harness`.

    Go: staged — the (small, fast) discriminating tests run first; the (large, slow)
    regression-guard tests only run if those pass. JS/TS: one whole-suite run — see
    `_RepoConfig`'s docstring for why staging isn't worth it there.

    Excludes any path `test_patch` already touches from the candidate's own patch — see
    `dockerexec.apply_patch_or_fail_cmd`'s docstring for why this is safe (no gold fix ever needs
    those paths) and necessary (the agent's worktree never has `test_patch` applied, so a
    same-file edit is captured against a baseline that no longer matches at grading time)."""
    if not solution.strip():
        return GradeResult(outcome="fail", detail="empty patch — bug remains unfixed")

    image = _image(task)
    nonce = uuid.uuid4().hex
    pass_cmd = dockerexec.report_cmd(nonce, "PASS")
    exclude_paths = dockerexec.diff_touched_paths(task.row["test_patch"])
    script = (
        _setup_script(task, nonce)
        + f"{dockerexec.write_file_cmd(solution, '/tmp/candidate.patch')}\n"
        + dockerexec.apply_patch_or_fail_cmd(nonce, "/tmp/candidate.patch", exclude_paths=exclude_paths)
        + _test_stage_script(task, nonce, on_pass=pass_cmd)
    )
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds, volumes=_volumes(task))
    finally:
        dockerexec.touch_image(image)


def grade_reference(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: apply the test patch, then the real GOLD fix
    (`task.row["fix_patch"]`). Must score ~100% or the grader is broken. A fix-patch apply failure
    is `error_harness`, not `fail` — the dataset's own fix failing to apply is a dataset/image
    problem, never a signal about candidate quality.

    Go always runs both stages here — this is the one control that must actually prove the
    regression-guard set passes, since "reference scores ~100%" is what validates the grader is
    correct in the first place. JS/TS's single whole-suite run already covers both sets at once."""
    image = _image(task)
    nonce = uuid.uuid4().hex
    harness_fix_apply = dockerexec.report_cmd(nonce, "HARNESS", "fix patch failed to apply")
    pass_cmd = dockerexec.report_cmd(nonce, "PASS")
    script = (
        _setup_script(task, nonce)
        + f"{dockerexec.write_file_cmd(task.row['fix_patch'], '/tmp/fix.patch')}\n"
        + f"git apply /tmp/fix.patch || {{ {harness_fix_apply}; }}\n"
        + _test_stage_script(task, nonce, on_pass=pass_cmd)
    )
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds, volumes=_volumes(task))
    finally:
        dockerexec.touch_image(image)


def grade_null(task: Task, timeout_seconds: int = DOCKER_TIMEOUT_SECONDS) -> GradeResult:
    """Grader-validation control: apply the test patch and no fix at all. The discriminating tests
    (`f2p_tests`/`n2p_tests`/`s2p_tests`) are expected to fail. Must score ~0% or the grader is
    broken.

    Go deliberately never runs the regression-guard (`p2p_tests`) stage here — those tests are
    defined as passing both before and after the gold fix, and null's state already is the
    "before" state, so they're trivially guaranteed to pass and checking them adds cost, not
    signal — see docs/engineering-notes.md, "Staged grading rationale (Go/Rust)". JS/TS has no
    separate stage to skip: its one whole-suite run naturally fails here since the discriminating
    tests fail without a fix."""
    image = _image(task)
    nonce = uuid.uuid4().hex
    pass_cmd = dockerexec.report_cmd(nonce, "PASS")
    script = _setup_script(task, nonce) + _test_stage_script(task, nonce, on_pass=pass_cmd, include_regression_guard=False)
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds, volumes=_volumes(task))
    finally:
        dockerexec.touch_image(image)
