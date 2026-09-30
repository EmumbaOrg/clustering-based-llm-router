# multi-swe-rl (Go slice)

A self-contained, implementation-focused tutorial for multi-swe-rl, using
[`tutorials/_shared/docker_exec.py`](../_shared/docker_exec.py) (generic Docker plumbing shared
across the three Docker-based tutorials in this directory).

Companion script: [quickstart.py](quickstart.py). Covers the **Go slice in full** — the dataset's
most uniform language and the one whose grading logic has a real, previously-fixed correctness bug
worth learning from (see the pitfall below). JS/TS/Java/Rust use structurally similar but genuinely
different per-language conventions; section 2 covers Rust's variant as a second worked example and
points at the authoritative source for the rest rather than re-deriving all five independently.

```bash
pip install datasets huggingface_hub
python tutorials/multi-swe-rl/quickstart.py --n-tasks 2 --structural-only   # no daemon needed
python tutorials/multi-swe-rl/quickstart.py --n-tasks 2                       # real run — pulls a multi-GB image
```

## 1. Dataset overview

multi-swe-rl is a multi-language real-bug-fixing benchmark, one instance per real merged pull
request across 7 languages. Only 5 are gradeable today (Go, JS, TS, Java, Rust) — C/C++ have no
confirmed grading-image convention and remain corpus-only.

- **Source**: [`ByteDance-Seed/Multi-SWE-RL`](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-RL)
  on the Hugging Face Hub. **Batch/split: `data_20240601_20250331`.** License is ambiguous — the
  dataset card says CC0-1.0, the Hub's own license tag says "other"; check both before relying on
  it.
- Data is laid out as per-language, per-repo `.jsonl` files (e.g.
  `data_20240601_20250331/go/beego__beego_dataset.jsonl`), not a single flat table — loading means
  listing the repo tree and downloading the relevant files directly (see section 2), not a plain
  `load_dataset(..., split=...)` call the way the other 4 tutorials in this series work.
- **Row id convention differs from every other dataset in this series**: `instance_id` (or a
  reconstructed `{org}__{repo}-{number}`) is used directly, not a positional index — file read
  order is size-ranked, not stable, so a positional id would silently drift between reads.

### Schema (fields this tutorial's code touches)

| Field | Meaning |
|---|---|
| `org`, `repo`, `number` | Identify the PR — used to build both the Docker image tag and the repo directory path inside the container. |
| `base.sha` | The commit the buggy pre-fix state is checked out at (nested under `base`, not a top-level `base_commit` — different shape from SWE-Gym's equivalent field). |
| `fix_patch` | The gold fix — this dataset's name for what SWE-Gym calls `patch`. |
| `test_patch` | Test changes exercising the FAIL_TO_PASS-equivalent tests — applied in every grading mode, same role as SWE-Gym's `test_patch`. |
| `f2p_tests`, `n2p_tests`, `s2p_tests` | A 3-way **discriminating** taxonomy — tests that must go from some non-passing state to passing once the bug is genuinely fixed (fail→pass, new→pass, skip→pass respectively) — richer than SWE-bench-style FAIL_TO_PASS alone. |
| `p2p_tests` | The **regression guard** — tests that pass both BEFORE and AFTER the gold fix. Checking these is expensive (the largest repo in this dataset measured ~11,000 individual Go subtests) and, by definition, adds no discriminating signal for a null (unfixed) solution — see section 2 for when this tutorial's code skips them. |

## 2. Implementation

### Loading (Go slice)

```python
from huggingface_hub import HfApi, hf_hub_download
import json

api = HfApi()
tree = api.list_repo_tree("ByteDance-Seed/Multi-SWE-RL", "data_20240601_20250331",
                           repo_type="dataset", recursive=True)
go_paths = [e.path for e in tree
            if e.path.startswith("data_20240601_20250331/go/") and e.path.endswith("_dataset.jsonl")]
# (the discarded-instances file is a different suffix and is excluded on purpose)

rows = []
for path in go_paths:
    local_path = hf_hub_download("ByteDance-Seed/Multi-SWE-RL", filename=path, repo_type="dataset")
    rows.extend(json.loads(line) for line in open(local_path, encoding="utf-8") if line.strip())
```

### Image naming and repo layout

```python
def image(row):
    return f"mswebench/{row['org']}_m_{row['repo']}:pr-{row['number']}".lower()

def repo_dir(row):
    return f"/home/{row['repo']}"
```

### Grading, step by step

1. **Setup (every mode)**: `cd` into `repo_dir` (missing → `error_harness`), write `test_patch` to
   `/tmp/test.patch`, `git apply` it (failure → `error_harness`) — same shape as SWE-Gym's setup
   step, different field/path conventions.
2. **`grade()`**: empty solution → `fail` directly. Otherwise, after setup, apply the candidate
   patch (failure → `fail` — the model's own problem), then run the Go test stage (below) against
   the discriminating tests **and** the regression guard.
3. **`grade_reference()`**: applies `fix_patch` (the real gold fix). Runs **both** the
   discriminating tests and the regression guard — this is the one control that must actually prove
   `p2p_tests` passes, so it can't be skipped here the way `grade_null` skips it.
4. **`grade_null()`**: setup only, no fix. Skips the regression guard entirely —
   `p2p_tests` are *defined* as tests passing both before and after the fix, so an unfixed state
   trivially satisfies them; checking them here would only cost time (measured ~2-3 minutes saved
   on the largest repos) for a guaranteed result.

### The Go test stage — and the pitfall it exists to fix

**Never trust `go test ./...`'s overall exit code, or a blanket count of `=== RUN` lines, as a
proxy for whether one specific target test passed.** This is not a theoretical concern — it's a
real, confirmed failure mode that scores a genuinely correct fix as `fail` on real tasks:

- `gohugoio/hugo` PR 11029: an unrelated `go-internal/testscript` toolchain/stdlib mismatch failed
  the WHOLE MODULE's exit code, even though the actual target test
  (`TestReproCommentsIn10947`) printed its own `--- PASS` line correctly.
- `jesseduffield/lazygit` PR 3676: `TestIntegration` fans out to hundreds of subtests under one
  shared top-level name; an unrelated SIBLING subtest panicked (missing git identity inside the
  container) and failed the whole `-run` match, even though the two actual target subtests never
  touched git at all.

A naive whole-suite-exit-code grader misclassifies the correct fix as `fail` on both — a gold
solution being wrongly scored as broken is exactly the kind of grader bug the reference-oracle
control (section 3) exists to catch.

**The correct approach**, reproduced here:

1. Build a `-run` regex from just the TARGET tests' TOP-LEVEL names (the part before the first
   `/`) — always plain Go identifiers, never needing regex escaping.
2. Run `go test ./... -run "<pattern>" -v`, redirecting ALL output to a log file.
3. Write two PATTERN FILES (not a shell-interpolated string — some full test names contain shell
   metacharacters, e.g. `TestPostingsForMatchers/n!~"(1|2.5)"`): one line per target test of the
   form `--- PASS: {full name} (`, and the equivalent for `--- FAIL:`.
4. `grep -F -o -f` each pattern file against the log, count distinct matches.
5. Decide the outcome from THOSE per-target-test counts, never from `go test`'s own exit code:
   any `FAIL` line among the targets → `fail`; not every target test produced its own `PASS` line
   → `error_harness` (something about the run itself is broken — a build failure, a missing test,
   etc. — not necessarily that a target test genuinely failed); otherwise → `pass`.

```python
# see quickstart.py for the full, runnable version
def _go_test_stage(row, nonce, include_regression_guard):
    full_names = target_test_names(row, include_regression_guard)
    top_level_names = sorted({name.split("/", 1)[0] for name in full_names})
    run_pattern = "^(" + "|".join(top_level_names) + ")$"
    # write run_pattern.txt, pass_patterns.txt, fail_patterns.txt (base64-transported, see
    # _shared/docker_exec.write_file_cmd), then:
    #   go test ./... -run "$(cat run_pattern.txt)" -v > go_test.log 2>&1
    #   PASSED=$(grep -F -o -f pass_patterns.txt go_test.log | sort -u | wc -l)
    #   FAILED=$(grep -F -o -f fail_patterns.txt go_test.log | sort -u | wc -l)
    #   FAILED>0 -> FAIL ; PASSED != TOTAL -> HARNESS ; else -> PASS
    ...
```

### A second worked example: Rust

Rust's grading command (from the official harness's own Rust support) takes a different shape —
`cargo test` instead of `go test`, and a different result-counting convention:

```bash
cd {repo_dir} && cargo test -- --exact $(cat names_file) > log 2>&1
EXIT=$?
COUNT=$(grep -oE '[0-9]+ passed; [0-9]+ failed' log | awk '{sum += $1 + $3} END {print sum+0}')
```

This is enough to show the shape differs meaningfully per language — not just a find-and-replace of
`go test` → `cargo test`. JS/TS/Java each have their own further-different conventions again (see
section 3's official-harness pointer) — this tutorial doesn't reimplement all five in full, since
the point of this section is the *mechanism* (per-target-test result checking, never a suite-level
exit code), not exhaustive per-language coverage.

## 3. Evaluation

### Custom harness

```python
row = go_tasks[0]
print(grade_reference(row))   # expect outcome == "pass"
print(grade_null(row))        # expect outcome != "pass"
```

### Official harness cross-check

Every convention in this tutorial — image tags, per-language test commands, result parsing — is
taken directly from
[`multi-swe-bench/multi-swe-bench`](https://github.com/multi-swe-bench/multi-swe-bench)'s own
harness source, specifically `multi_swe_bench/harness/repos/go/{org}/{repo}.py` (and the sibling
`javascript/`, `typescript/`, `java/`, `rust/` directories for the other languages). This is the
most directly-traceable official-harness relationship of any dataset in this tutorial series — the
per-repo Python files there are the ground truth this tutorial's Go implementation was checked
against.

To cross-check: run the official harness against the same (instance, patch) pairs and confirm its
verdicts agree with this tutorial's PASS/FAIL/HARNESS outcomes — paying particular attention to any
task where they disagree on a Go task with many subtests under one top-level name, since that's
exactly the shape of the exit-code pitfall above. As with every other tutorial in this series,
verify exact current CLI invocation against the official harness's own README rather than trusting
a specific command asserted here.

### Commands

```bash
python tutorials/multi-swe-rl/quickstart.py --n-tasks 2 --structural-only
python tutorials/multi-swe-rl/quickstart.py --n-tasks 2
```
