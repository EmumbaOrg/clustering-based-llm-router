"""Standalone multi-swe-rl (Go slice) loader + Docker-based grader — companion to TUTORIAL.md.

Uses the generic `tutorials/_shared/docker_exec.py` helper. Covers the Go slice in full — the
dataset's "pilot" language, and the one whose grading logic has a real, well-documented exit-code
pitfall worth learning from (see TUTORIAL.md). JS/TS/Java/Rust follow analogous but genuinely
different per-language conventions — TUTORIAL.md section 2 covers Rust's `cargo test` variant as a
second worked example and points at the official harness's own per-language source for the rest,
rather than reimplementing all 5.

Run with:

    python tutorials/multi-swe-rl/quickstart.py --n-tasks 2

Requires: pip install datasets huggingface_hub, and a running Docker daemon. Each task pulls a
multi-GB prebuilt image the FIRST time it's graded — real network/disk cost, not a quick demo.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import docker_exec  # noqa: E402

# --- 1. Dataset loading (see TUTORIAL.md section 1 for the schema) -------------------------

HF_DATASET_ID = "ByteDance-Seed/Multi-SWE-RL"
BATCH = "data_20240601_20250331"
FILE_SUFFIX = "_dataset.jsonl"  # excludes multi_swe_bench_discarded_instances.jsonl on purpose


def load_go_tasks(n_tasks: int, seed: int) -> list[dict]:
    api = HfApi()
    tree = api.list_repo_tree(HF_DATASET_ID, BATCH, repo_type="dataset", recursive=True)
    go_paths = sorted(
        entry.path for entry in tree
        if entry.path.startswith(f"{BATCH}/go/") and entry.path.endswith(FILE_SUFFIX)
    )
    rows: list[dict] = []
    for path in go_paths:
        local_path = hf_hub_download(HF_DATASET_ID, filename=path, repo_type="dataset")
        with open(local_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        if len(rows) >= n_tasks * 20:  # enough of a pool to sample from without reading every file
            break
    return random.Random(seed).sample(rows, min(n_tasks, len(rows)))


# --- 2. Custom grading harness, written inline (see TUTORIAL.md section 2) -----------------

_DISCRIMINATING_TEST_KEYS = ("f2p_tests", "n2p_tests", "s2p_tests")
_REGRESSION_GUARD_TEST_KEY = "p2p_tests"


def _image(row: dict) -> str:
    return f"mswebench/{row['org']}_m_{row['repo']}:pr-{row['number']}".lower()


def _repo_dir(row: dict) -> str:
    return f"/home/{row['repo']}"


def _setup_script(row: dict, nonce: str) -> str:
    repo_dir = _repo_dir(row)
    harness_no_repo = docker_exec.report_cmd(nonce, "HARNESS", f"missing {repo_dir}")
    harness_test_patch = docker_exec.report_cmd(nonce, "HARNESS", "test_patch failed to apply")
    return (
        f"cd {repo_dir} || {{ {harness_no_repo}; }}\n"
        f"{docker_exec.write_file_cmd(row['test_patch'], '/tmp/test.patch')}\n"
        f"git apply /tmp/test.patch || {{ {harness_test_patch}; }}\n"
    )


def _target_test_names(row: dict, include_regression_guard: bool) -> list[str]:
    keys = list(_DISCRIMINATING_TEST_KEYS)
    if include_regression_guard:
        keys.append(_REGRESSION_GUARD_TEST_KEY)
    names: list[str] = []
    for key in keys:
        names.extend(row.get(key) or [])
    return names


def _go_test_stage(row: dict, nonce: str, include_regression_guard: bool) -> str:
    """The critical piece — see TUTORIAL.md's pitfall section for the real-task bugs this exact
    approach exists to avoid: `go test ./...`'s OVERALL exit code is never trusted as a signal
    about any one target test.
    Instead: run with `-run` scoped to just the target tests' top-level names, capture full `-v`
    output to a log file, then check each TARGET test's own `--- PASS: {name}`/`--- FAIL: {name}`
    line — via `grep -F -o -f` against pattern FILES, never interpolating test names directly into
    the shell command, since some contain shell metacharacters."""
    full_names = _target_test_names(row, include_regression_guard)
    if not full_names:
        # p2p_tests can legitimately be empty for some rows — nothing to check is a pass, not a
        # harness problem.
        return docker_exec.report_cmd(nonce, "PASS")

    # `-run` needs only the TOP-LEVEL test name (before the first "/") — plain Go identifiers,
    # never containing characters that need escaping in a regex, unlike the full subtest names.
    top_level_names = sorted({name.split("/", 1)[0] for name in full_names})
    run_pattern = "^(" + "|".join(top_level_names) + ")$"

    pass_patterns = "\n".join(f"--- PASS: {name} (" for name in full_names)
    fail_patterns = "\n".join(f"--- FAIL: {name} (" for name in full_names)

    repo_dir = _repo_dir(row)
    pass_cmd = docker_exec.report_cmd(nonce, "PASS")
    fail_cmd = docker_exec.report_cmd(nonce, "FAIL", "one or more target tests failed")
    harness_cmd = docker_exec.report_cmd(
        nonce, "HARNESS", "not every target test produced a PASS line (build/setup problem, "
        "not necessarily a test failure)",
    )
    return (
        f"{docker_exec.write_file_cmd(run_pattern, '/tmp/run_pattern.txt')}\n"
        f"{docker_exec.write_file_cmd(pass_patterns, '/tmp/pass_patterns.txt')}\n"
        f"{docker_exec.write_file_cmd(fail_patterns, '/tmp/fail_patterns.txt')}\n"
        f'cd {repo_dir} && go test ./... -run "$(cat /tmp/run_pattern.txt)" -v > /tmp/go_test.log 2>&1\n'
        f"TOTAL=$(wc -l < /tmp/pass_patterns.txt)\n"
        f"PASSED=$(grep -F -o -f /tmp/pass_patterns.txt /tmp/go_test.log | sort -u | wc -l)\n"
        f"FAILED=$(grep -F -o -f /tmp/fail_patterns.txt /tmp/go_test.log | sort -u | wc -l)\n"
        f'if [ "$FAILED" -gt 0 ]; then {fail_cmd}; '
        f'elif [ "$PASSED" -ne "$TOTAL" ]; then {harness_cmd}; '
        f"else {pass_cmd}; fi\n"
    )


def grade(row: dict, solution: str, timeout_seconds: int = 1200) -> docker_exec.GradeResult:
    if not solution.strip():
        return docker_exec.GradeResult(outcome="fail", detail="empty patch — bug remains unfixed")

    nonce = uuid.uuid4().hex
    fail_apply = docker_exec.report_cmd(nonce, "FAIL", "candidate patch failed to apply")
    script = (
        _setup_script(row, nonce)
        + f"{docker_exec.write_file_cmd(solution, '/tmp/candidate.patch')}\n"
        f"git apply /tmp/candidate.patch || {{ {fail_apply}; }}\n"
        + _go_test_stage(row, nonce, include_regression_guard=True)
    )
    return docker_exec.run(_image(row), script, nonce, timeout_seconds)


def grade_reference(row: dict, timeout_seconds: int = 1200) -> docker_exec.GradeResult:
    """Applies the gold `fix_patch`. Runs BOTH the discriminating tests AND the regression guard
    (p2p_tests) — this is the one control that must actually prove the regression-guard set
    passes, unlike grade_null below."""
    nonce = uuid.uuid4().hex
    harness_fail = docker_exec.report_cmd(nonce, "HARNESS", "fix_patch failed to apply")
    script = (
        _setup_script(row, nonce)
        + f"{docker_exec.write_file_cmd(row['fix_patch'], '/tmp/fix.patch')}\n"
        f"git apply /tmp/fix.patch || {{ {harness_fail}; }}\n"
        + _go_test_stage(row, nonce, include_regression_guard=True)
    )
    return docker_exec.run(_image(row), script, nonce, timeout_seconds)


def grade_null(row: dict, timeout_seconds: int = 1200) -> docker_exec.GradeResult:
    """Setup only (test_patch applied, no fix). Skips the regression guard: p2p_tests are DEFINED
    as tests that pass both before and after the gold fix, so null's unfixed state trivially
    satisfies them — checking them here would just cost time for a guaranteed result."""
    nonce = uuid.uuid4().hex
    script = _setup_script(row, nonce) + _go_test_stage(row, nonce, include_regression_guard=False)
    return docker_exec.run(_image(row), script, nonce, timeout_seconds)


# --- 3. Evaluation ---------------------------------------------------------------------------


def run_custom_harness(tasks: list[dict]) -> None:
    print("\n--- custom harness (this script's own grade_reference()/grade_null(), Go slice) ---")
    for row in tasks:
        ref = grade_reference(row)
        null = grade_null(row)
        print(f"{row.get('org')}/{row.get('repo')}#{row.get('number')}: "
              f"reference -> {ref.outcome} ({ref.detail})  null -> {null.outcome}")


def run_official_harness_note() -> None:
    print("\n--- official harness note ---")
    print(
        "This dataset's image tags, per-language test commands, and result conventions are taken "
        "directly from multi-swe-bench/multi-swe-bench's own harness source "
        "(multi_swe_bench/harness/repos/go/{org}/{repo}.py and siblings for other languages). "
        "See TUTORIAL.md section 3 for how to cross-check and why this tutorial only reimplements "
        "the Go slice in full."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--structural-only", action="store_true",
                         help="build/print the grading scripts without calling docker (no daemon needed)")
    args = parser.parse_args()

    tasks = load_go_tasks(args.n_tasks, args.seed)
    print(f"sampled {len(tasks)} multi-swe-rl Go tasks")

    if args.structural_only:
        for row in tasks:
            nonce = "PREVIEW"
            print(f"\n{row.get('org')}/{row.get('repo')}#{row.get('number')} (image={_image(row)}):")
            print(_setup_script(row, nonce) + _go_test_stage(row, nonce, include_regression_guard=True))
        return

    run_custom_harness(tasks)
    run_official_harness_note()


if __name__ == "__main__":
    main()
