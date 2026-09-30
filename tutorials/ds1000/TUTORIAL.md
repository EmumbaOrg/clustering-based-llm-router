# DS-1000

A self-contained, implementation-focused tutorial for DS-1000 — every mechanic (loading, grading,
evaluation) is written out directly and runnable end to end.

Companion script: [quickstart.py](quickstart.py).

```bash
pip install datasets
python tutorials/ds1000/quickstart.py --n-tasks 5
```

## 1. Dataset overview

DS-1000 is a data-science coding benchmark: 1,000 real Stack Overflow-derived problems across
NumPy/Pandas/Matplotlib/etc., each asking for a short snippet solving a specific data-manipulation
task. What makes it distinctive relative to BigCodeBench: **each row carries its own executable
test harness inline**, not just a reference answer — grading code doesn't need to know anything
data-science-specific, because the row tells it exactly how to check a solution.

- **Source**: [`xlangai/DS-1000`](https://huggingface.co/datasets/xlangai/DS-1000) on the Hugging
  Face Hub. **Split: `test`** — 1,000 rows.
- **License**: CC-BY-SA-4.0.
- **This tutorial excludes Matplotlib rows** (155 of 1,000) — they grade by rendering the
  candidate's code to a PNG and comparing it against a reference image, which needs a rendering
  step out of scope here. 845 rows remain gradeable.

### Schema (fields this tutorial's code touches)

| Field | Meaning |
|---|---|
| `metadata.problem_id` | Numeric id used to build a stable task id, e.g. `ds1000:42`. |
| `metadata.library` | Which library the problem is about (`Numpy`, `Pandas`, `Matplotlib`, ...) — used only to filter out Matplotlib rows here. |
| `prompt` | The natural-language + code-context problem statement shown to a model. |
| `reference_code` | The gold solution string. |
| `code_context` | **The row's own test harness** — Python source defining `test_execution(solution)` (present on every row) and, on 159/1000 rows, `test_string(solution)` too. Both must pass for a solution to be graded correct. |

The `[insert]` splice-placeholder convention DS-1000 uses internally is handled entirely inside
`test_execution` itself — this tutorial's grading code never touches it directly, it just calls the
function `code_context` defines.

## 2. Implementation

### Loading

```python
from datasets import load_dataset

HF_DATASET_ID = "xlangai/DS-1000"
HF_SPLIT = "test"

ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
tasks = [dict(row) for row in ds if row["metadata"]["library"] != "Matplotlib"]
```

### Grading, step by step

1. Write `row["code_context"]` to a `code_context.py` file, and the candidate `solution` string to
   `solution.txt`.
2. In a **subprocess**, with a fresh temp working directory and a wall-clock timeout: `exec()`
   `code_context.py` into a namespace dict, loading its `test_execution` (and, if present,
   `test_string`) functions. Read the candidate solution string back, call `test_execution(solution)`,
   then — if the row defines it — `test_string(solution)` too.

   **Why both, when both are present**: `test_execution` alone can be satisfied by a hardcoded
   answer that happens to match the expected output value without using the required approach —
   `test_string` (when the row has one) additionally checks the solution's *source text* itself
   (e.g. "must use `.sum()`"), which a hardcoded literal answer wouldn't pass.
3. Outcome from whichever exception (if any) is raised:
   - `ImportError`/`ModuleNotFoundError` (harness setup OR inside the candidate/test call) →
     `error_missing_dep` — same simplification and same reasoning as the BigCodeBench tutorial:
     models overwhelmingly import real, common libraries, so a genuinely missing module is a
     stronger signal of an incomplete grading environment than a deliberate model mistake.
   - `AssertionError` → `fail` (the expected, everyday "wrong answer" case — DS-1000's own tests
     are plain `assert` statements).
   - Any other `Exception` → `fail`.
   - No exception → `pass`.
4. The grading script's LAST action prints a `RESULT_<nonce>: <TAG>` line, exactly the same
   nonce-guarded sentinel protocol as the BigCodeBench tutorial (see that tutorial's section 2 for
   the full "why a nonce" rationale — it applies identically here, since the candidate's code again
   runs via `exec()` in the same process as the grading script).

```python
# see quickstart.py for the full, runnable version
def grade(row: dict, solution: str, timeout_seconds: int = 60) -> tuple[str, str]:
    # write code_context.py + solution.txt + a grading script to a temp dir, run as a subprocess
    # with a timeout, scan stdout for RESULT_<nonce>:, return (outcome, detail)
    ...
```

## 3. Evaluation

### Custom harness

```python
tasks = load_tasks()[:5]
for row in tasks:
    print(grade(row, row["reference_code"]))   # expect ("pass", "")
    print(grade(row, "None"))                    # almost certainly ("fail", ...)
```

Reference must pass; a deliberately wrong solution must fail. Same grader-validation logic as
every other dataset in this tutorial series — a low score from a real model is only meaningful if
these controls hold.

### "Official harness" — an honest caveat

Unlike BigCodeBench, **DS-1000 has no separate official evaluation package to install and diff
against.** The dataset's `code_context` field, with its `test_execution`/`test_string` convention,
*is* the paper's own grading approach, baked directly into each row — there's no third-party CLI
sitting on top of it the way `bigcodebench`'s pip package sits on top of BigCodeBench's rows. This
tutorial's custom harness in section 2 is, in effect, already a from-scratch implementation of "the
official approach," not a simplified stand-in for one.

The meaningful cross-check here isn't against a different tool — it's a **scale check**: run the
reference solution (known-correct by construction) across a much larger sample than the 5 used
above, and confirm it passes at (near) 100%. A gap indicates either a bug in this tutorial's
`grade()`, or a row whose `code_context` has an environment dependency this tutorial's Python
environment doesn't satisfy (which would show up as `error_missing_dep`, not a silent `fail`).

```bash
python tutorials/ds1000/quickstart.py --n-tasks 100   # widen the sample for a scale check
```

### Commands

```bash
python tutorials/ds1000/quickstart.py --n-tasks 5
```
