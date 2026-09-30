# SWE-Gym

A self-contained, implementation-focused tutorial for SWE-Gym, using
[`tutorials/_shared/docker_exec.py`](../_shared/docker_exec.py) (generic Docker plumbing shared
across the three Docker-based tutorials in this directory).

Companion script: [quickstart.py](quickstart.py).

```bash
pip install datasets
python tutorials/swe-gym/quickstart.py --n-tasks 2 --structural-only   # no daemon needed, prints the scripts
python tutorials/swe-gym/quickstart.py --n-tasks 2                       # real run — pulls a multi-GB image per task
```

## 1. Dataset overview

SWE-Gym is a real-bug-fixing benchmark built from actual GitHub issues on real, unmodified open
-source repos, at real historical commits — unlike SWE-smith's synthetically injected bugs. Each
row is checked out at the actual pre-fix commit where the bug genuinely existed.

- **Source**: [`SWE-Gym/SWE-Gym`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) on the Hugging
  Face Hub. **Split: `train`.** No filtering or capping needed — every row is directly usable.

### Schema (fields this tutorial's code touches)

| Field | Meaning |
|---|---|
| `instance_id` | Dataset-native task id, e.g. `python__mypy-11207`. Also used to derive the Docker image name (see below — there's no separate image field). |
| `problem_statement` | The real GitHub issue text. |
| `base_commit` | The real commit SHA where the bug already exists — **the image is already checked out here**, unlike SWE-smith where the image starts clean. |
| `patch` | **The real gold fix** for this bug. Same field name as SWE-smith's `patch`, but the OPPOSITE direction: SWE-smith's `patch` applied forward CREATES a bug; SWE-Gym's `patch` applied forward FIXES one that's already there. Don't assume identical semantics from a shared field name across datasets. |
| `test_patch` | SWE-Gym-specific — a diff supplying the FAIL_TO_PASS/PASS_TO_PASS test changes. Applied in every grading mode, same role as SWE-smith's `patch`-forward setup step but a genuinely different field. |
| `FAIL_TO_PASS`, `PASS_TO_PASS` | Same meaning as SWE-smith: pytest node ids that must flip from failing to passing, or must keep passing, once the bug is genuinely fixed. |

## 2. Implementation

### Loading

```python
from datasets import load_dataset

ds = load_dataset("SWE-Gym/SWE-Gym", split="train")
tasks = [dict(row) for row in ds]
```

### Image naming

Unlike SWE-smith, there's no `image_name` field — it's derived from `instance_id`:

```python
def image_for(row: dict) -> str:
    # Docker Hub disallows "__" in image names, hence the "_s_" substitution.
    return f"xingyaoww/sweb.eval.x86_64.{row['instance_id'].replace('__', '_s_')}:latest".lower()
```

These are public, prebuilt per-instance images under the `xingyaoww/sweb.eval.x86_64.*` namespace
on Docker Hub — no build step, no separate `install` command needs to be re-run at grade time (the
package is already editable-installed in the image; even a patch that touches something like
pandas' `meson.build` triggers an automatic rebuild on next import via the editable install's own
build backend — in the rare worst case that rebuild can take several minutes, worth budgeting a
generous grading timeout for).

### Grading, step by step

1. **Setup (every mode)**: `cd /testbed`, write `row["test_patch"]` to `/tmp/test.patch`, `git
   apply` it. A failed apply here is `error_harness` — nothing candidate-specific has happened yet.
2. **`grade()`**: empty solution → `fail` directly (no Docker call). Otherwise, after setup, write
   the candidate's patch and `git apply` it — a failure here IS a real candidate failure (`fail`).
3. **`grade_reference()`**: after setup, apply `row["patch"]` — the real gold fix (not a reversal,
   unlike SWE-smith's reference control, because this dataset's `patch` already points the right
   direction). Must score ~100% pass.
4. **`grade_null()`**: setup only, no fix — must score ~0% pass.
5. **Running tests**, after
   `source /opt/miniconda3/bin/activate && conda activate testbed` — **a different activation path
   from SWE-smith's** (`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed`),
   confirmed directly against real images rather than assumed identical. Then `cd /testbed &&
   python -m pytest -q <FAIL_TO_PASS + PASS_TO_PASS node ids>`.

**Pitfall — pytest exit codes 4 and 5 are NOT failures**: ~2.8% of rows record a FAIL_TO_PASS or
PASS_TO_PASS node id containing a literal non-ASCII character, while the pytest version actually
installed in the corresponding image escapes non-ASCII parametrize ids internally — so the literal
node id from the row never matches anything, and pytest exits 4 ("usage error", nothing matched a
given `-k`/node-id selector in some pytest versions) or 5 ("no tests were collected") instead of
running any test at all. This is OUR harness failing to select the right tests, not a signal about
the candidate's code — it must route to `error_harness`, never `fail`. Getting this wrong silently
scores ~2.8% of otherwise-correct solutions as broken.

```python
# see quickstart.py for the full, runnable version
def _pytest_script(row: dict, nonce: str) -> str:
    # ... after activating conda and cd'ing to /testbed:
    #   run pytest, capture $?
    #   exit 0  -> PASS
    #   exit 4 or 5 -> HARNESS (see pitfall above)
    #   anything else -> FAIL
    ...
```

## 3. Evaluation

### Custom harness

```python
row = tasks[0]
print(grade_reference(row))   # expect outcome == "pass"
print(grade_null(row))        # expect outcome != "pass"
```

### Official harness cross-check

SWE-Gym follows SWE-bench-style evaluation conventions
([`princeton-nlp/SWE-bench`](https://github.com/princeton-nlp/SWE-bench), `pip install swebench`).
One deliberate, known divergence worth being explicit about when cross-checking: **the official
harness re-runs an `install` command per (repo, version) at evaluation time**, resolving environment
setup from SWE-bench's own version-to-install-command mapping. This tutorial's harness (section 2)
skips that entirely, because every prebuilt `xingyaoww/sweb.eval.*` image already has the package
editable-installed — re-running install would be redundant work, not a correctness gap, but
it IS a real difference in what actually executes, worth knowing about before assuming the two
harnesses are doing byte-for-byte the same thing. As with the other tutorials here, verify exact
current CLI flags against `swebench`'s own `--help`/README rather than trusting any specific
invocation asserted in this file.

### Commands

```bash
python tutorials/swe-gym/quickstart.py --n-tasks 2 --structural-only
python tutorials/swe-gym/quickstart.py --n-tasks 2
```
