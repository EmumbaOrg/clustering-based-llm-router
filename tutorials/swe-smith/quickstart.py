"""Standalone SWE-smith loader + Docker-based grader — companion to TUTORIAL.md.

Uses only third-party (`datasets`) and the local `tutorials/_shared/docker_exec.py` helper
(generic Docker plumbing — see that file's own docstring). Run with:

    python tutorials/swe-smith/quickstart.py --n-tasks 2

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

HF_DATASET_ID = "SWE-bench/SWE-smith"
HF_SPLIT = "train"
POOL_SEED = 42
# Loading and filtering all ~59K rows just to sample a handful costs real minutes — cap the pool
# after shuffling (see TUTORIAL.md section 1 for why).
POOL_CAP = 500


def load_tasks(n_tasks: int, seed: int) -> list[dict]:
    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT).shuffle(seed=POOL_SEED)
    tasks = []
    for row in ds:
        if len(tasks) >= POOL_CAP:
            break
        if not row.get("problem_statement") or not str(row["problem_statement"]).strip():
            continue  # a meaningful fraction of rows are synthetic mutations with no NL description
        tasks.append(dict(row))
    return random.Random(seed).sample(tasks, min(n_tasks, len(tasks)))


# --- 2. Custom grading harness, written inline (see TUTORIAL.md section 2) -----------------

REPO_DIR = "/testbed"
CONDA_ACTIVATE = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"


def _node_ids(row: dict) -> str:
    ids = row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]
    return " ".join(shlex.quote(i) for i in ids)


def _setup_script(row: dict, nonce: str) -> str:
    """The image starts CLEAN, not buggy — every grading mode needs to establish the buggy
    baseline first by applying the row's own `patch` field forward."""
    harness_fail = docker_exec.report_cmd(nonce, "HARNESS", "bug patch failed to apply")
    return (
        f"cd {REPO_DIR}\n"
        f"{docker_exec.write_file_cmd(row['patch'], '/tmp/bug.patch')}\n"
        f"git apply /tmp/bug.patch || {{ {harness_fail}; }}\n"
    )


def _pytest_script(row: dict, nonce: str) -> str:
    pass_cmd = docker_exec.report_cmd(nonce, "PASS")
    fail_cmd = docker_exec.report_cmd(nonce, "FAIL", "tests did not pass")
    return (
        f"{CONDA_ACTIVATE} && cd {REPO_DIR} && python -m pytest -q {_node_ids(row)}\n"
        f"if [ $? -eq 0 ]; then {pass_cmd}; else {fail_cmd}; fi\n"
    )


def grade(row: dict, solution: str, timeout_seconds: int = 600) -> docker_exec.GradeResult:
    if not solution.strip():
        return docker_exec.GradeResult(outcome="fail", detail="empty patch — bug remains unfixed")

    nonce = uuid.uuid4().hex
    fail_apply = docker_exec.report_cmd(nonce, "FAIL", "candidate patch failed to apply")
    script = (
        _setup_script(row, nonce)
        + f"{docker_exec.write_file_cmd(solution, '/tmp/candidate.patch')}\n"
        # A candidate patch that fails to apply is the MODEL's failure, not ours — "fail", not
        # "harness" (unlike the bug-patch apply in _setup_script above, which is OUR setup step).
        f"git apply /tmp/candidate.patch || {{ {fail_apply}; }}\n"
        + _pytest_script(row, nonce)
    )
    return docker_exec.run(row["image_name"], script, nonce, timeout_seconds)


def grade_reference(row: dict, timeout_seconds: int = 600) -> docker_exec.GradeResult:
    """Applies the bug, then REVERSES it (`git apply -R`) — a round-trip back to the clean state.
    Must score ~100% pass; if it doesn't, the grader itself is broken."""
    nonce = uuid.uuid4().hex
    harness_reverse_fail = docker_exec.report_cmd(nonce, "HARNESS", "bug patch failed to reverse-apply")
    script = (
        _setup_script(row, nonce)
        + f"{docker_exec.write_file_cmd(row['patch'], '/tmp/bug.patch')}\n"
        f"git apply -R /tmp/bug.patch || {{ {harness_reverse_fail}; }}\n"
        + _pytest_script(row, nonce)
    )
    return docker_exec.run(row["image_name"], script, nonce, timeout_seconds)


def grade_null(row: dict, timeout_seconds: int = 600) -> docker_exec.GradeResult:
    """Bug applied, no fix — must score ~0% pass."""
    nonce = uuid.uuid4().hex
    script = _setup_script(row, nonce) + _pytest_script(row, nonce)
    return docker_exec.run(row["image_name"], script, nonce, timeout_seconds)


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
        "SWE-smith's images and grading conventions descend from SWE-bench's own harness "
        "(princeton-nlp/SWE-bench, pip package `swebench`). See TUTORIAL.md section 3 for how to "
        "cross-check against it and the specific divergences already known and accepted here "
        "(pytest node-id targeting rather than SWE-bench's whole-file-plus-log-parsing approach)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--structural-only", action="store_true",
                         help="build/print the grading scripts without calling docker (no daemon needed)")
    args = parser.parse_args()

    tasks = load_tasks(args.n_tasks, args.seed)
    print(f"sampled {len(tasks)} SWE-smith tasks with a non-empty problem_statement")

    if args.structural_only:
        for row in tasks:
            nonce = "PREVIEW"
            print(f"\n{row['instance_id']} (image={row['image_name']}):")
            print(_setup_script(row, nonce) + _pytest_script(row, nonce))
        return

    run_custom_harness(tasks)
    run_official_harness_note()


if __name__ == "__main__":
    main()
