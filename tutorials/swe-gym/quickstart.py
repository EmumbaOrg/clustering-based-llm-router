"""Standalone SWE-Gym loader + Docker-based grader — companion to TUTORIAL.md.

Uses the generic `tutorials/_shared/docker_exec.py` helper. Run with:

    python tutorials/swe-gym/quickstart.py --n-tasks 2

Requires: pip install datasets, and a running Docker daemon. Each task pulls a multi-GB prebuilt
image the FIRST time it's graded — expect real network/disk cost, not a quick demo.
"""
from __future__ import annotations

import argparse
import random
import shlex
import sys
import uuid
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import docker_exec  # noqa: E402

# --- 1. Dataset loading (see TUTORIAL.md section 1 for the schema) -------------------------

HF_DATASET_ID = "SWE-Gym/SWE-Gym"
HF_SPLIT = "train"


def load_tasks(n_tasks: int, seed: int) -> list[dict]:
    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
    all_tasks = [dict(row) for row in ds]  # no filtering/capping — every row is usable as-is
    return random.Random(seed).sample(all_tasks, min(n_tasks, len(all_tasks)))


# --- 2. Custom grading harness, written inline (see TUTORIAL.md section 2) -----------------

REPO_DIR = "/testbed"
# NOTE this activation path is DIFFERENT from swe-smith's — confirmed directly against real
# images, not assumed identical (see TUTORIAL.md's pitfall note).
CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate && conda activate testbed"


def _image(row: dict) -> str:
    # No image_name field on SWE-Gym rows (unlike swe-smith) — constructed from instance_id.
    # Dunders replaced with "_s_" because Docker Hub disallows "__" in image names.
    return f"xingyaoww/sweb.eval.x86_64.{row['instance_id'].replace('__', '_s_')}:latest".lower()


def _node_ids(row: dict) -> str:
    ids = row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]
    return " ".join(shlex.quote(i) for i in ids)


def _setup_script(row: dict, nonce: str) -> str:
    """Unlike swe-smith, the image is ALREADY at the buggy `base_commit` — `row['patch']` here is
    the GOLD FIX, not a bug-injection diff. What every mode needs to apply first instead is
    `test_patch`: SWE-Gym-specific test changes that exercise FAIL_TO_PASS/PASS_TO_PASS."""
    harness_fail = docker_exec.report_cmd(nonce, "HARNESS", "test_patch failed to apply")
    return (
        f"cd {REPO_DIR}\n"
        f"{docker_exec.write_file_cmd(row['test_patch'], '/tmp/test.patch')}\n"
        f"git apply /tmp/test.patch || {{ {harness_fail}; }}\n"
    )


def _pytest_script(row: dict, nonce: str) -> str:
    pass_cmd = docker_exec.report_cmd(nonce, "PASS")
    fail_cmd = docker_exec.report_cmd(nonce, "FAIL", "tests did not pass")
    harness_cmd = docker_exec.report_cmd(nonce, "HARNESS", "pytest usage error / no tests collected")
    return (
        f"{CONDA_ACTIVATE} && cd {REPO_DIR} && python -m pytest -q {_node_ids(row)}\n"
        f"PYTEST_EXIT=$?\n"
        # Pitfall (see TUTORIAL.md): ~2.8% of rows record a node id with a literal non-ASCII
        # character that the installed pytest version escapes internally, so it never matches and
        # pytest exits 4 (usage error) or 5 (no tests collected) instead of running anything. That
        # is OUR harness failing to select the right tests, not the candidate's code being wrong —
        # must route to HARNESS, not FAIL, or a perfectly good solution gets scored as broken.
        f"if [ $PYTEST_EXIT -eq 0 ]; then {pass_cmd}; "
        f"elif [ $PYTEST_EXIT -eq 4 ] || [ $PYTEST_EXIT -eq 5 ]; then {harness_cmd}; "
        f"else {fail_cmd}; fi\n"
    )


def grade(row: dict, solution: str, timeout_seconds: int = 900) -> docker_exec.GradeResult:
    if not solution.strip():
        return docker_exec.GradeResult(outcome="fail", detail="empty patch — bug remains unfixed")

    nonce = uuid.uuid4().hex
    fail_apply = docker_exec.report_cmd(nonce, "FAIL", "candidate patch failed to apply")
    script = (
        _setup_script(row, nonce)
        + f"{docker_exec.write_file_cmd(solution, '/tmp/candidate.patch')}\n"
        f"git apply /tmp/candidate.patch || {{ {fail_apply}; }}\n"
        + _pytest_script(row, nonce)
    )
    return docker_exec.run(_image(row), script, nonce, timeout_seconds)


def grade_reference(row: dict, timeout_seconds: int = 900) -> docker_exec.GradeResult:
    """Applies the row's OWN `patch` field — here that's the real gold fix, not a bug-injection
    diff (contrast with swe-smith's grade_reference, which REVERSES a bug patch instead)."""
    nonce = uuid.uuid4().hex
    harness_fail = docker_exec.report_cmd(nonce, "HARNESS", "gold patch failed to apply")
    script = (
        _setup_script(row, nonce)
        + f"{docker_exec.write_file_cmd(row['patch'], '/tmp/gold.patch')}\n"
        f"git apply /tmp/gold.patch || {{ {harness_fail}; }}\n"
        + _pytest_script(row, nonce)
    )
    return docker_exec.run(_image(row), script, nonce, timeout_seconds)


def grade_null(row: dict, timeout_seconds: int = 900) -> docker_exec.GradeResult:
    nonce = uuid.uuid4().hex
    script = _setup_script(row, nonce) + _pytest_script(row, nonce)
    return docker_exec.run(_image(row), script, nonce, timeout_seconds)


# --- 3. Evaluation ---------------------------------------------------------------------------


def run_custom_harness(tasks: list[dict]) -> None:
    print("\n--- custom harness (this script's own grade_reference()/grade_null()) ---")
    for row in tasks:
        ref = grade_reference(row)
        null = grade_null(row)
        print(f"{row['instance_id']}: reference -> {ref.outcome} ({ref.detail})  null -> {null.outcome}")


def run_official_harness_note() -> None:
    print("\n--- official harness note ---")
    print(
        "SWE-Gym follows SWE-bench-style evaluation conventions (princeton-nlp/SWE-bench, pip "
        "package `swebench`), with one deliberate divergence: the official harness re-runs an "
        "`install` command per (repo, version) at eval time, while every image here already has "
        "the package editable-installed. See TUTORIAL.md section 3."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--structural-only", action="store_true",
                         help="build/print the grading scripts without calling docker (no daemon needed)")
    args = parser.parse_args()

    tasks = load_tasks(args.n_tasks, args.seed)
    print(f"sampled {len(tasks)} SWE-Gym tasks")

    if args.structural_only:
        for row in tasks:
            nonce = "PREVIEW"
            print(f"\n{row['instance_id']} (image={_image(row)}):")
            print(_setup_script(row, nonce) + _pytest_script(row, nonce))
        return

    run_custom_harness(tasks)
    run_official_harness_note()


if __name__ == "__main__":
    main()
