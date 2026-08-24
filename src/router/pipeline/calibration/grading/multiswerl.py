"""Grader for Multi-SWE-RL's Go, JS, TS, Java, and Rust slices — Docker-based, built on
`dockerexec.py`'s sentinel protocol, mirroring `swegym.py`'s shape. `tasks.py`'s loader for this
source returns rows from the dataset's `go/`, `js/`, `ts/`, `java/`, and `rust/` batch directories
— see its own docstring for why the other 2 languages (C, C++) in this dataset are corpus-only for
now. The Go section immediately below was the original (single-convention) pilot; the
"whole-suite languages" section further down covers JS, TS, and Java, which all ended up sharing
one considerably less uniform second design; Rust (further down still) turned out uniform enough
to rejoin Go's original staged design instead.

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

--- Whole-suite languages: JS, TS, Java (second and third pilots) ------------------------------

`_REPO_CONFIG` covers the JS (619 tasks, 10 repos), TS (412 tasks, 8 repos), and Java (976 tasks,
6 repos) slices — the second-, third-, and (by task count) largest-after-Go language buckets.

An early version of this guessed the test command from each repo's `package.json` and tried to
replicate Go's staged discriminating/regression-guard split by parsing a file path or test name out
of each `f2p_tests`/`p2p_tests` key. Both guesses turned out wrong, or at least far riskier than
necessary, once checked against ground truth: the official `multi_swe_bench` harness's own source
(`multi_swe_bench/harness/repos/{javascript,typescript}/{org}/{repo}.py` in
github.com/multi-swe-bench/multi-swe-bench) literally defines each repo's `prepare.sh`/`run.sh`/
`test-run.sh`, and pulling two real images (`expressjs/express`, `colinhacks/zod`) confirmed those
exact scripts are baked into every image at `/home/*.sh` — a much stronger source of truth than
inferring from `package.json` alone. Two real findings from reading that source, not the dataset
schema:

1. **`package.json` guessed the wrong framework for `zod`.** It looked like Vitest (a `test:vitest`
   script exists); the harness's own confirmed `run.sh` uses `yarn build && yarn test`, which
   resolves to Jest (`test:ts-jest`) — `package.json` had multiple test scripts and the wrong one
   was picked. Trusting the harness's own confirmed invocation instead of a heuristic avoided
   shipping a broken command for that repo.
2. **Almost none of the 18 confirmed commands support name-based narrowing.** Most are each repo's
   own `npm test`/`yarn test`/`pnpm test` wrapper (`mongoose`'s is a bare `npm test`), not a direct
   framework invocation we could append `--grep`/`--testNamePattern` to with any confidence it'd be
   forwarded. Given that, and that a live run confirmed these suites are fast — `express`'s 1,149
   Mocha tests ran in 4.9s — the staged discriminating-then-regression-guard split Go relies on
   (see above) isn't worth the narrowing complexity/risk here: JS/TS runs the confirmed command
   ONCE per grading call and reads the outcome from its exit code (`_whole_suite_test_script`), covering
   both test sets in a single pass. `grade_null` still fails correctly here with no special-casing:
   the discriminating tests are defined to fail without a fix, so the whole suite's exit code is
   already non-zero.

`_RepoConfig` has no install step: `prepare.sh` (dependency install) is a genuine one-time
IMAGE-BUILD step (`RUN bash /home/prepare.sh` in the harness's own `dockerfile()`), not something
to repeat at grading time — confirmed directly (both pulled images already had `node_modules`
present: 59MB for `express`, 732MB for `zod`), the same "pre-resolved, don't touch it" shape as
Go's `GOMODCACHE`. A handful of repos DO need a genuine per-call `build` step (`zod`, `nuxt`,
`react-router`) because their tests run against compiled output that a candidate's source patch
would otherwise leave stale — confirmed present in their own `run.sh`, not assumed.

This is still genuinely closer in shape to the official harness's own ~533 hand-written per-repo
classes than to Go's one clean convention — some of those classes exist specifically because a
repo's test command changed across its own history (e.g. `commander.js` migrated from Jest to
node's built-in `node:test` at some point) and needed a PR-range-specific override.
`_REPO_CONFIG` uses whichever class has no such suffix (current/default), which may not exactly
match every instance's era; a mismatch surfaces as a script failure (`error_harness`), not a
silently wrong grade — see `_RepoConfig`'s own docstring.

**Validated live against real containers** (`grade_null` ~0%, `grade_reference` ~100%, same bar as
Go's own pilot): `colinhacks/zod` (has a build step), `expressjs/express` (JSON reporter, no build
step), and `Automattic/mongoose` — the single largest JS/TS repo (302 of 1,031 tasks). Mongoose's
first `grade_reference` run FAILED once (`tests failed`) while a byte-identical manual replay and
an immediate retry both passed cleanly (3579 passing, 0 failing) — mongoose spins up a real
MongoDB instance per test run (its own suite prints a live "Downloading MongoDB ..." progress
line, confirmed in a real row's `f2p_tests` key), making it timing/network-sensitive per container.
Treated the same way istio's/prometheus's occasional unrelated-test noise is treated above: a
known, accepted flakiness source for this specific (large, heavyweight-setup) repo, not a grader
bug — re-run rather than trust a single `grade_reference` failure there as conclusive. The
remaining 15 repos share one of these three already-validated shapes (no build + JSON reporter,
no build + plain exit code, or a build step) and haven't each been individually run against a
container yet.

**Java (added after JS/TS, applying the same lesson from the start)**: went straight to the
harness's own source (`multi_swe_bench/harness/repos/java/{org}/{repo}.py`) for the confirmed
command rather than guessing from `pom.xml`/`build.gradle`, and found a much simpler shape than
JS/TS — every one of the 6 repos has exactly one class (no PR-range overrides), and every
confirmed command is already whole-suite with no name-based narrowing (`mvn clean test
-Dstyle.color=never` for `checkstyle`; `./gradlew test`/`./gradlew clean test --continue` for the
Gradle repos; a Maven-wrapper variant with the repo's own required profile flags for `fastjson2`).
No repo needs a separate `build` field — Maven/Gradle's `clean test` already does compile+test in
one command. `clean` wipes `target/`/`build/` on every call (the harness's own confirmed choice,
not a shortcut of ours to remove without evidence), so unlike Go's incremental `go test`, a Java
grading call always pays a full recompile.

`checkstyle` is 845 of Java's 976 tasks (86.6%) — overwhelmingly the repo that matters here; the
other 5 (`spotbugs` 57, `junit5` 55, `logstash`/`mockito` 7 each, `fastjson2` 5) are comparatively
low-volume.

**Whole-suite exit code turned out to be unreliable for `checkstyle` specifically — confirmed live,
not assumed.** Its ~2,900-5,100-test suite has at least one baked-in, always-reproducible failure
completely unrelated to any patch, reproducing even on the raw base commit with zero patches
applied: on one real PR, `ImportControlLoaderTest.testInputStreamThatFailsOnClose` fails a Mockito
`verify(times(1))` check (`Wanted 1 time but was 2 times`) — a JDK/library-version sensitivity, not
a real regression; on a different, much later PR, `MainTest.testExistingTargetFileButWithoutReadAccess`
fails because it expects a permission-denied file read to fail, but these containers all run as
**root**, which bypasses Unix permission checks entirely, so the read always succeeds. Two
different unrelated tests, both 100% deterministic (retried and reproduced on isolated single-class
runs too) — this is baseline noise inherent to running this suite as root in this environment, not
occasional flakiness like `istio`'s or `mongoose`'s. Trusting the overall exit code would make
`grade_reference`/`grade_null` indistinguishable from this noise for 86.6% of all Java tasks.

**Fix**: `checkstyle` and `fastjson2` (the two Maven repos whose `f2p_tests`/`p2p_tests` keys are
exact fully-qualified class names — confirmed real format) use `granularity="class"`
(`_java_class_test_script`): the confirmed command still runs exactly once, but the outcome is read
from the SPECIFIC named classes' Maven Surefire XML reports (`target/surefire-reports/
TEST-{class}.xml`, one line `<testsuite ... errors="N" failures="M">` per class — confirmed real
format by inspecting a pulled image), ignoring whatever happens in every OTHER class. `mockito`/
`junit5`/`logstash`/`spotbugs` key by Gradle *task* name (`buildSrc:generateExternalPluginSpecBuilders`)
or a mix of task names and JUnit display names, not exact class names — `granularity="exit_code"`
(no name-based check available) stays for those; lower volume (126 of 976 Java tasks) and no
evidence yet of the same systemic noise.

**Validated live against real containers** (`grade_null` fails, `grade_reference` passes):
`checkstyle` on BOTH problem PRs above (confirming the class-granularity fix actually resolves what
it was built for, not just that it runs), and `mockito` (Gradle, `exit_code` path). `~/.m2`
confirmed pre-baked in the pulled `checkstyle` image (84MB) — the same "resolved once, baked into
the image" shape as Go's `GOMODCACHE` and JS/TS's `node_modules`, not assumed by analogy alone.
`checkstyle`'s suite took 1m37s–4m12s per run across the two validated PRs — real but not
prohibitive. `fastjson2`/`logstash`/`junit5`/`spotbugs` (131 tasks combined) share an
already-validated shape but haven't each been individually run against a container yet.

--- Rust (fourth pilot) ---------------------------------------------------------------------

Rust turned out to be the most uniform language pilot yet, closer to Go's shape than to
JS/TS/Java's — confirmed from the harness's own source
(`multi_swe_bench/harness/repos/rust/{org}/{repo}.py`), not assumed from `Cargo.toml` conventions:
**every one of the 14 repos** (`ripgrep`, `alacritty`, `clap`, `fish-shell`, `helix`, `nushell`,
`rusqlite`, `mdBook`, `serde`, `bat`, `fd`, `bytes`, `tokio`, `tracing`) **uses the exact same
confirmed command, bare `cargo test`** — no per-repo variation, no PR-range overrides, no build
step. `f2p_tests`/`n2p_tests`/`s2p_tests`/`p2p_tests` keys are already exact `cargo test` filter
targets (`module::submodule::test_name`) — Rust identifiers joined by `::`, which can't contain
regex/shell metacharacters, so (unlike JS/TS) nothing needs parsing, stripping, or escaping.

Given that uniformity, Rust **reuses Go's staged discriminating/regression-guard design directly**
(`_cargo_test_stage`, dispatched from the same `_discriminating_stage_script`/
`_regression_guard_stage_script` Go already used) rather than JS/TS/Java's whole-suite compromise:
`cargo test -- --exact <name1> <name2> ...` runs exactly and only the OR'd named tests — confirmed
live against a real container, Rust's own equivalent of Go's `-run "^(name1|name2)$"` anchor. Like
Go's `-run`, it exits 0 even when it matches ZERO tests anywhere (confirmed live: a bogus name and
a real name checked before its introducing patch was applied both produced `0 passed; 0 failed`
across every test binary) — `_cargo_test_stage` sums `N passed`/`M failed` across every `test
result:` line the run prints (one per test binary/crate target) as the "did anything run" guard,
the same role Go's `RUN_COUNT` plays.

No `_RepoConfig`/`_REPO_CONFIG` entry needed at all: Rust repos are tracked in a separate
`_RUST_REPOS` set purely for `_is_rust` dispatch, since the confirmed command needs no per-repo
customization the way JS/TS/Java's does. No cache volume is mounted either — `target/` (cargo's
build cache) is confirmed pre-baked into the image at build time (`rusqlite`'s image shipped a
140MB `target/` from `prepare.sh`'s own `cargo test || true` run), the same "resolved once, baked
in" shape as Go's `GOMODCACHE`; a `$CARGO_TARGET_DIR` host mount (mirroring Go's `GOCACHE`) is a
candidate future optimization, but — like every other cache-mount decision in this module — only
worth adding after MEASURING repeated-call cost, not assumed from the Go precedent.

One accepted quirk carries over unchanged, not newly introduced by Rust: a row whose entire
discriminating set is `n2p_tests` (brand-new tests the fix introduces) shows "0 tests matched"
against the null state, since those tests don't exist in source without the fix (confirmed live on
a real `rusqlite` row) — reports `HARNESS`, the same outcome Go's own "zero tests matched" guard
already produces for this exact class of row (it doesn't distinguish f2p/n2p/s2p either).

**Validated live against real containers** (`grade_null` fails or reports the accepted `n2p`-only
`HARNESS` outcome, `grade_reference` passes): `rusqlite` (smallest Rust repo, 1 task,
discriminating set is 100% `n2p_tests` — exercises the accepted quirk above directly) and `clap`
(largest single Rust repo, 62 tasks; the validated row's `p2p_tests` set was 212 names, confirming
the multi-name `--exact` invocation holds up at scale, and its `grade_null` genuinely FAILED rather
than zero-matching, confirming `cargo test`'s exit code correctly reflects a real test failure —
closing the one risk this pilot's plan flagged as unconfirmed). Repo checkout path (`/home/{repo}`)
and image tag convention confirmed unchanged. The other 12 repos share this already-validated shape
but haven't each been individually run against a container yet.
"""
from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass

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


# Every one of these 14 repos is confirmed (from the official multi_swe_bench harness's own
# run.sh, not guessed) to use the exact same bare `cargo test` command, with no PR-range
# overrides — the most uniform of any language pilot so far, closer to Go's shape than to
# JS/TS/Java's. See `_cargo_test_stage`'s docstring and the module docstring's Rust section.
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
    """Rust's `f2p_tests`/`n2p_tests`/`s2p_tests`/`p2p_tests` keys are already exact `cargo test`
    filter targets (`module::submodule::test_name`, confirmed real format) — no stripping needed,
    unlike Go's `_top_level_names` (Rust identifiers joined by `::` can't contain the kind of
    regex-special subtest-path suffix Go's keys sometimes carry)."""
    names: set[str] = set()
    for key in keys:
        names |= set(task.row.get(key) or {})
    return sorted(names)


def _cargo_test_stage(task: Task, nonce: str, names: list[str], label: str, on_pass: str) -> str:
    """Rust's equivalent of `_run_test_stage`. `cargo test -- --exact <name1> <name2> ...` runs
    exactly and only the named tests, OR'd — confirmed live against a real container (two real
    test names ran and passed, correctly filtered out of every other test binary in the crate) —
    Rust's own equivalent of Go's `-run "^(name1|name2)$"` anchor, needing no escaping since `::`
    -joined paths are always safe identifiers.

    Like Go's `-run`, `cargo test -- --exact` exits 0 even when it matches ZERO tests anywhere
    (confirmed live: a bogus name and a real name checked before its introducing patch was applied
    both produced `0 passed; 0 failed` across every test binary, exit 0) — so this sums `N passed`/
    `M failed` across every `test result:` line the run prints (`cargo test` runs one such line per
    test binary/crate target) rather than trusting the exit code alone."""
    repo_dir = _repo_dir(task)
    names_text = "\n".join(names)
    safe_label = label.replace("-", "_")
    names_file, log_file = f"/tmp/{safe_label}_names.txt", f"/tmp/cargo_test_{safe_label}.log"
    exit_var, count_var = f"EXIT_{safe_label.upper()}", f"RUN_COUNT_{safe_label.upper()}"
    fail_cmd = dockerexec.report_cmd(nonce, "FAIL", f"{label} tests failed")
    harness_cmd = dockerexec.report_cmd(nonce, "HARNESS", f"no {label} tests matched the expected names")
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
    """One row per JS/TS repo (keyed by `(org, repo)` in `_REPO_CONFIG`), copied directly from the
    official `multi_swe_bench` harness's own per-repo source
    (`multi_swe_bench/harness/repos/{javascript,typescript}/{org}/{repo}.py` in
    github.com/multi-swe-bench/multi-swe-bench) — not guessed from `package.json`, and
    independently confirmed by pulling `expressjs/express`'s and `colinhacks/zod`'s real images and
    finding those exact commands baked into `/home/run.sh`/`/home/test-run.sh`.

    `build` is a per-grading-call compile step for the handful of repos whose tests run against
    compiled output rather than source directly (`yarn build` for zod/react-router, `pnpm
    build:stub` for nuxt) — confirmed present in their own `run.sh`, so it isn't optional. `test`
    is the confirmed test invocation. There is no `install` field: `prepare.sh` (dependency
    install) runs once at IMAGE BUILD time (`RUN bash /home/prepare.sh` in the harness's own
    `dockerfile()`), the same way Go's module cache is pre-resolved — confirmed directly by
    inspecting a pulled image (`express`'s `node_modules` was already 59MB; `zod`'s 732MB) rather
    than assumed by analogy.

    A handful of these repos have PR-range-specific override classes upstream (e.g. `commander.js`
    migrated from Jest to node's built-in `node:test` at some point in its history) — this table
    uses whichever class has no numeric PR-range suffix (the current/default one), which may not
    exactly match every instance's era. Treated as a known, low-blast-radius risk: a mismatched
    command surfaces as a script failure (`error_harness`), not a silently wrong grade.

    `granularity="exit_code"` (the default) trusts the confirmed command's own exit code — correct
    for every repo above, and for the Gradle-based Java repos (`mockito`/`junit5`/`logstash`/
    `spotbugs`), whose `f2p_tests`/`p2p_tests` keys are Gradle *task* names, not test names, so
    there's nothing more specific to check anyway. `granularity="class"` is for Maven repos whose
    keys ARE exact test class names (`checkstyle`, `fastjson2`) — confirmed necessary, not a
    speculative refinement: `checkstyle`'s ~2,900-5,100-test suite has at least one baked-in,
    always-reproducible failure unrelated to any patch (confirmed on two different PRs — a
    Mockito/JDK-sensitive stream-close count in one, a root-user-bypasses-file-permissions check in
    another), so trusting the overall exit code would make `grade_reference`/`grade_null`
    indistinguishable from noise for checkstyle's 845 tasks (86.6% of all Java tasks). See
    `_java_class_test_script`."""
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
    # `clean` is part of every confirmed command below (Maven/Gradle wipe their build output
    # before testing), so — unlike Go — a full recompile happens on every single grading call.
    # That's the harness's own confirmed choice, not a shortcut of ours to remove without
    # evidence; no `build` field is needed here since `clean test` already does compile+test in
    # one command, the same way it does for the repos above with `build=None`. `checkstyle`/
    # `fastjson2` use `granularity="class"` (confirmed necessary, see _RepoConfig's docstring);
    # the 4 Gradle repos key by build task name, not test name, so exit_code is the only option
    # and no repo-specific noise has been found there (lower volume, lower validated confidence).
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
    gets recompiled from scratch on every grading call. JS/TS mounts nothing: dependencies are
    baked into the image at build time (see `_RepoConfig`'s docstring) and never touched again, and
    the handful of repos with a genuine per-call build step compile from source into a small
    per-container `dist`/`lib` output — not yet measured as worth caching the way Go's build cache
    was (that mount was added only after measuring a 34x speedup, not assumed). Rust mounts nothing
    either: `target/` (cargo's own build cache) is confirmed pre-baked into the image the same way
    — a `$CARGO_TARGET_DIR` host mount is a candidate future optimization, same "measure first"
    bar as Go's `GOCACHE`, not something to add on assumption."""
    if _whole_suite_config(task) is not None or _is_rust(task):
        return None
    return _cache_volumes()


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
    if _is_rust(task):
        names = _rust_test_names(task, _DISCRIMINATING_TEST_KEYS)
        if not names:
            return dockerexec.report_cmd(nonce, "HARNESS", "no discriminating tests found on this row")
        return _cargo_test_stage(task, nonce, names, "discriminating", on_pass)

    names = _discriminating_test_names(task)
    if not names:
        # Should never happen on a well-formed row — every task needs at least one test that
        # distinguishes buggy from fixed. Unlike the regression-guard stage below, an empty set
        # here is a dataset/harness problem worth surfacing, not a silent pass-through.
        return dockerexec.report_cmd(nonce, "HARNESS", "no discriminating tests found on this row")
    return _run_test_stage(task, nonce, names, "discriminating", on_pass)


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
    return _run_test_stage(task, nonce, names, "regression-guard", on_pass)


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
    `_RepoConfig`'s docstring for why staging isn't worth it there."""
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

    Go deliberately never runs the regression-guard (`p2p_tests`) stage here — those are defined
    by the dataset as passing both BEFORE and after the gold fix, and null's state (test_patch
    applied, no fix) IS the "before" state, so by the dataset's own labeling they're already
    guaranteed to pass here. Checking them would only add cost, not signal — measured directly:
    27s for the discriminating stage alone vs. 150-190s for the full set on the same real
    instance, a ~6x cut on every single null-control run. JS/TS has no separate stage to skip
    (see `_RepoConfig`'s docstring) — its one whole-suite run naturally fails here because the
    discriminating tests fail without a fix, exactly the same signal a staged run would give."""
    image = _image(task)
    nonce = uuid.uuid4().hex
    pass_cmd = dockerexec.report_cmd(nonce, "PASS")
    script = _setup_script(task, nonce) + _test_stage_script(task, nonce, on_pass=pass_cmd, include_regression_guard=False)
    try:
        return dockerexec.run(image, script, nonce, timeout_seconds, volumes=_volumes(task))
    finally:
        dockerexec.touch_image(image)
