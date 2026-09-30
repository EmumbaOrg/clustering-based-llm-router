# BigCodeBench-Instruct

A self-contained, implementation-focused tutorial for BigCodeBench-Instruct — every mechanic
(loading, grading, evaluation) is written out directly and runnable end to end.

Companion script: [quickstart.py](quickstart.py). Run it after reading section 1:

```bash
pip install datasets
python tutorials/bigcodebench/quickstart.py --n-tasks 5
```

## 1. Dataset overview

BigCodeBench-Instruct is a Python function-completion benchmark: given a natural-language
instruction and a function signature, produce the function body. It's "gradeable" without any repo
checkout or container — every row carries its own executable test suite.

- **Source**: [`bigcode/bigcodebench`](https://huggingface.co/datasets/bigcode/bigcodebench) on
  the Hugging Face Hub. **Split: `v0.1.4`** — 1,140 rows. The unversioned default split merges 5
  historical dataset versions into 5,700 rows; pinning `v0.1.4` is a deliberate choice to match a
  single, stable version, not an oversight.
- **License**: Apache-2.0.

### Schema (fields this tutorial's code touches)

| Field | Meaning |
|---|---|
| `task_id` | Dataset-native id, e.g. `"BigCodeBench/0"` — stable across runs, safe to reference directly. |
| `instruct_prompt` | The natural-language instruction. There's also a `complete_prompt` (docstring-style, not instruction-style) — this tutorial uses `instruct_prompt` throughout. |
| `code_prompt` | The function signature the solution must complete. **Ends mid-signature** (`def task_func(...):\n`) — a solution is a body FRAGMENT to append after this, not a standalone script. |
| `canonical_solution` | The gold function body — itself a fragment in the same shape as `code_prompt` expects. |
| `test` | A full `unittest.TestCase` (class `TestCases`) that imports and exercises `task_func`. |

One non-obvious detail worth internalizing before writing any grading code: some tests rely on
globals the solution module sets up — e.g. a test that calls `random.seed(42)` without importing
`random` itself, because it expects the solution's own `import random` to already be in scope. That
only works if the candidate code and the test are `exec()`'d into the **same namespace**, not two
isolated ones (see section 2).

## 2. Implementation

### Loading

```python
from datasets import load_dataset

HF_DATASET_ID = "bigcode/bigcodebench"
HF_SPLIT = "v0.1.4"

ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
tasks = [dict(row) for row in ds]
```

### Grading, step by step

1. Concatenate `row["code_prompt"] + solution` into a `candidate.py` file. (**Pitfall**: passing a
   complete script instead of just the body fragment produces a `SyntaxError` — the signature line
   would appear twice. This is graded `fail`, not a harness error, exactly like an empty solution:
   `code_prompt` alone is invalid syntax once `exec()`'d as a full program, since it ends
   mid-signature.)
2. Write `row["test"]` to a `test.py` file.
3. In a **subprocess**, with a fresh temp working directory and a wall-clock timeout: `exec()`
   `candidate.py` into a namespace dict `ns`, then `exec()` `test.py` **into that same `ns`** (the
   shared-globals detail from section 1). Then load `TestCases` from `ns` and run it through
   `unittest.TextTestRunner`.
4. The grading script's LAST action is to print one line — `RESULT_<nonce>: PASS`,
   `RESULT_<nonce>: FAIL <detail>`, or `RESULT_<nonce>: ERROR_MISSING_DEP <detail>` — where
   `<nonce>` is a fresh `uuid4().hex` generated per call. The calling process scans stdout for a
   line with that exact prefix and classifies the outcome from it.

   **Why a nonce, not a fixed `RESULT:` string**: the candidate's OWN code runs inside this same
   subprocess (via `exec()`), so it's just as capable of printing arbitrary text as the grading
   script itself. A candidate solution containing `print("RESULT: PASS")` would otherwise
   short-circuit grading to a false pass before the real test suite ever executes. A nonce
   generated fresh per call and unknown to the candidate ahead of time closes that off — this is a
   real, previously-confirmed exploit against a fixed-sentinel design, not a hypothetical.

5. **Missing-dependency handling** (a deliberate simplification, not an accident): any
   `ImportError`/`ModuleNotFoundError` — whether raised by the harness's own setup OR by the
   candidate's code — is classified `error_missing_dep`, a separate outcome excluded from error
   rates, rather than `fail`. This requires checking two places, not one: an import missing at
   *module load time* raises directly and is caught by the harness's own `try/except`; but a
   function body that only `import`s a missing package when the function is actually *called* (the
   common case, since the import runs during the test's call, not while the function is merely
   defined) raises *during* `unittest.TextTestRunner.run()` — which catches that exception
   internally and records it in `result.errors`, and does **not** let it propagate to an outer
   `try/except`. So after a failed run, `result.errors`/`result.failures` tracebacks are inspected
   for the missing-dependency signature too, not just exceptions caught directly.

```python
# see quickstart.py for the full, runnable version with the exact grading script template
def grade(row: dict, solution: str, timeout_seconds: int = 60) -> tuple[str, str]:
    candidate_src = row["code_prompt"] + solution
    test_src = row["test"]
    # ... write both + a grading script to a temp dir, run it as a subprocess with a timeout,
    # scan stdout for the RESULT_<nonce>: line, return (outcome, detail)
```

## 3. Evaluation

### Custom harness

Run the reference (gold) solution and an empty solution through `grade()` above — this is the
grader-validation pattern: **reference must score ~100% pass, null must score ~0% pass.** If either
control is off, the grader itself is broken, and no result it produces for a real model means
anything.

```python
tasks = load_tasks()[:5]
for row in tasks:
    print(grade(row, row["canonical_solution"]))   # expect ("pass", "")
    print(grade(row, ""))                            # expect ("fail", ...) — SyntaxError, not a harness error
```

### Official harness cross-check

BigCodeBench ships its own evaluation package —
[`bigcode-project/bigcodebench`](https://github.com/bigcode-project/bigcodebench) on GitHub, `pip
install bigcodebench`. It is intentionally **not** a dependency of this tutorial's own custom
harness above — it's an external reference point for cross-checking, not something the custom
harness needs at runtime.

**Why cross-check at all**, given the custom harness above already works: the official harness runs
candidate code inside a **Docker image with a large, curated set of libraries preinstalled**
(numpy/pandas/sklearn/etc.). The custom harness above runs in-process, in whatever environment it's
invoked from. A solution that's genuinely correct but imports a library that environment lacks
grades `error_missing_dep` here (excluded from error rates, per section 2) — while the official
harness would actually execute and grade it `pass` or `fail`. Cross-checking on a sample is how
you'd notice a thin local environment is under-counting real failures as "excluded" rather than
scoring them.

Rough flow — **verify exact flags against the package's own `--help`/README before relying on
this**; the CLI surface isn't something this tutorial can guarantee stays current, and asserting
exact flags without being able to run them here would be a claim I can't back up:

1. Produce a `{task_id, solution}` jsonl for the same tasks graded above. The official format
   expects a **complete function** per row — `row["code_prompt"] + solution` — not the bare
   fragment this tutorial's own `Task`-equivalent convention uses.
2. Run the package's own generate/evaluate entry points against that jsonl, pointed at its own
   Docker image (not whatever environment ran the custom harness above).
3. Diff its per-task pass/fail against the outcomes from the custom harness for the identical
   solutions. Expect exact agreement on `pass`/`fail`; an `error_missing_dep` row from the custom
   harness is exactly where a disagreement is most likely, per the reasoning above.

### Commands

```bash
python tutorials/bigcodebench/quickstart.py --n-tasks 5   # custom-harness walkthrough, real HF data
```
