# Dataset tutorials

Implementation-focused tutorials for 5 code/bug-fixing benchmark datasets: how each one is
structured, how to load it, and how to evaluate a solution against it — both with a from-scratch
custom harness and by cross-checking against the dataset's own official evaluation conventions.

Every tutorial is self-contained: it reimplements its own loading and grading logic directly and
visibly, rather than calling into an external library, so the mechanics are never hidden behind a
function call.

## Datasets

| Tutorial | Dataset | Docker required? |
|---|---|---|
| [bigcodebench/](bigcodebench/TUTORIAL.md) | BigCodeBench-Instruct | No |
| [ds1000/](ds1000/TUTORIAL.md) | DS-1000 | No |
| [swe-smith/](swe-smith/TUTORIAL.md) | SWE-smith | Yes |
| [swe-gym/](swe-gym/TUTORIAL.md) | SWE-Gym | Yes |
| [multi-swe-rl/](multi-swe-rl/TUTORIAL.md) | multi-swe-rl (Go slice) | Yes |

Each `TUTORIAL.md` has 3 sections: **Dataset overview** (what it is, full field-by-field schema),
**Implementation** (loading + the full custom grading mechanism, inline), and **Evaluation**
(running the custom harness's own controls, then cross-checking against the dataset's official
harness where one exists).

## Shared code

[`_shared/`](_shared/) holds generic, dataset-agnostic Docker/git plumbing reused by the 3
Docker-based tutorials (`docker_exec.py`, `repo_clone.py`) — raw `docker run` construction,
sentinel-based result parsing, and bare-clone/worktree checkout. This is shared to avoid
copy-pasting ~150-200 lines of subprocess code three times.

## Running the Docker-based tutorials

The swe-smith/swe-gym/multi-swe-rl tutorials pull real, multi-GB prebuilt evaluation images the
first time they grade a task — real network and disk cost, not a quick demo. Each one's
`quickstart.py` supports `--structural-only`, which builds and prints the exact grading script
without calling Docker at all, useful for reading through the mechanics without the download cost.
