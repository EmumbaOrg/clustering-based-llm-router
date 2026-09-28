# CLI reference

Two command groups under the `router` entry point (`src/router/cli.py`): `router pipeline ...`
(the offline pipeline — see [README.md](README.md)) and `router runtime ...` (the online routing
runtime). Every flag below is read straight from the actual `typer` command definitions, not
inferred from `--help` output.

## Global: `router pipeline` (any subcommand)

| Flag | Default | Effect |
|---|---|---|
| `--log-level` | `INFO` | Logging verbosity (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Use `DEBUG` when a run's outcome is confusing and you need per-task detail; the default is quiet enough for routine runs. |
| `--log-file <path>` | none (stderr only) | Also append logs to a file. Useful for a long unattended `calibrate` run you want a durable record of, since stdout's `typer.echo` summary lines don't capture the underlying `logger.info`/`logger.warning` detail. |

## `router pipeline corpus`

Builds `corpus.jsonl` from the 5 configured datasets (see [README.md](README.md#1-offline-pipeline-flow)).

| Flag | Default | Why you'd use it / effect |
|---|---|---|
| `--sample N` | none (full corpus) | Caps **each source** at N rows before dedup. A dry run to exercise every loader cheaply — for Multi-SWE-RL specifically, a capped sample reads the smallest file per language first, so it costs megabytes instead of the full ~4GB, but is biased toward small repos, not random. Omit for a real run. |

## `router pipeline embed`

Embeds `corpus.jsonl` into `embeddings.npz`, per `config/embedding.yaml`.

| Flag | Default | Why you'd use it / effect |
|---|---|---|
| `--sample N` | none | Only embed the first N rows (dry run, cheap CPU-throughput check before committing to a full embedding pass). |
| `--per-source-sample N` | none | Randomly samples up to N rows **per source** instead of the first N overall — a source-balanced cross-section (e.g. for `visualize-clusters`), rather than whichever sources happen to sort first. **Overrides `--sample`** if both are given. |
| `--seed` | `42` | Random seed for `--per-source-sample`'s sampling. Change it only if you deliberately want a different balanced sample; otherwise leave it for reproducibility. |

## `router pipeline build-artifact`

Runs k-means over the embeddings, writes `cluster-map.json` + `task-cluster-map.json`.

| Flag | Default | Why you'd use it / effect |
|---|---|---|
| `--k` | `config/clustering.yaml`'s `default_k` | Which k to promote to the final artifact. Use this to try a different cluster count without editing the config file — e.g. sweeping k to check cluster-size spread/inertia before settling on one. |

## `router pipeline run-all`

Convenience wrapper: `corpus` → `embed` → `build-artifact` in one call. **Does not** run
`validate-graders` or `calibrate` — those still need separate invocations.

| Flag | Default | Effect |
|---|---|---|
| `--sample N` | none | Passed through to `corpus` only (the embed step deliberately isn't re-capped — the corpus step's own cap already limited row count, so a second cap would compound it). |
| `--k` | `config/clustering.yaml`'s `default_k` | Passed through to `build-artifact`. |

## `router pipeline visualize-clusters`

Plots a 2D PCA projection of sampled tasks, colored by cluster, against the real centroids — a
sanity-check view of where tasks actually land. Requires `embed` and `build-artifact` to have run
first.

| Flag | Default | Why you'd use it / effect |
|---|---|---|
| `--sample-size` | `500` | How many tasks to randomly plot. Higher = a denser, slower-to-render picture. |
| `--highlight <task_id>` | none, repeatable | Annotates a specific task on the plot (e.g. `--highlight BigCodeBench/0 --highlight ds1000:42`) — included even if it wasn't in the random sample, useful for checking where one specific task of interest actually landed. |
| `--seed` | `42` | Sampling/PCA random seed — change it to see a different random slice; keep it fixed to compare two runs' plots apples-to-apples. |
| `--output <path>` | `artifacts/cluster-visualization.png` | Where to write the PNG. |

## `router pipeline validate-graders`

**GATE** — must be run (and pass) before trusting anything downstream. Grades the reference (gold)
and null (empty) solution for a sample of tasks per gradeable source, with no real model involved.
See [outcomes.md](outcomes.md) for what "pass" means for each outcome.

| Flag | Default | Why you'd use it / effect |
|---|---|---|
| `--tasks-per-source N` | `10` | How many tasks per source to sample for the gate. Raise it for a more thorough (slower, more Docker calls) gate before a high-stakes calibration run; the default is a quick sanity check. |
| `--source <name>` | none, repeatable — defaults to every source in `calibration.yaml`'s `gradeable_sources` | Narrows the gate to one or more specific sources (e.g. `--source swe-smith`) — use this after touching one grader's code, to re-gate just that source instead of re-running the whole (slower) suite. |

**What a passing gate looks like:** reference ~100% pass, null ~0% pass, for every source. Any
shortfall should appear as `error_harness`/`error_timeout` in the printed outcome counts — if it
shows up as `fail` instead, the *grader itself* is broken, not the (gold/empty) solution.

## `router pipeline select-verified-tasks`

Builds a ground-truth-**pre-verified** pinned task selection — every candidate task is checked via
the reference/null controls (cached in the registry) *before* being counted, so the resulting pin
never includes a task whose ground truth is already known-bad.

| Flag | Default | Why you'd use it / effect |
|---|---|---|
| `--target N` | **required** | Approximate total task count across all clusters. Achieved by scaling `tasks_per_cluster` up so the total lands near `N` (ceiling division against cluster count) — the exact achieved count still depends on category-mix rounding and any unfillable shortfall. |

Requires `config/calibration.yaml`'s `category_mix` to be set (category-aware selection only).
**Slow** — every not-yet-verified candidate costs a real grading call (Docker for repo-context
sources) — meant to be run occasionally, not as part of routine calibration. Writes both a normal
pinned selection (usable with `calibrate --tasks-file` exactly like any other pin) and the updated
registry.

## `router pipeline calibrate`

Builds/validates `model-profiles.json`. See [README.md](README.md#2-calibration-inner-loop-flow)
for the full inner-loop diagram.

| Flag | Default | Why you'd use it / effect |
|---|---|---|
| `--tasks-file <path>` | none (selects fresh via `select_tasks`) | Loads a previously pinned `calibration-task-selection-<run>.json` instead of re-selecting. **Required** for `--model` to be safe. Use it any time you want a run to grade against the *exact* same tasks as a previous run — required for a like-for-like comparison, and for onboarding a new model into an existing profile. |
| `--model <id>` | none, repeatable — omit to calibrate every model in `config/models.yaml` | Calibrate **only** this model (or these models). Grades against `--tasks-file`'s exact pinned set, then **merges** the result into the existing `model-profiles.json` (keeping every other model's entry untouched) rather than overwriting the whole file. Use this to add a new candidate without re-running (and re-paying for) every existing model. Errors if `--tasks-file` isn't also given, or if the named model isn't in `config/models.yaml`. |

**Effect of omitting both flags:** a full fresh run — new stratified task selection, every model in
the roster graded, a brand-new `model-profiles.json` written (not merged).

A full run (or a single `--model` onboarding run) can take hours — see
[running-calibration.md](running-calibration.md) for how to run it detached from the terminal with
a safe pause/resume/tail workflow.

## `router runtime validate`

Loads both artifacts against `config/models.yaml`'s candidate roster and reports coverage — no
embedding, no model download. A clean exit means every hard-fail rule in
[runtime-validation.md](runtime-validation.md) passed.

| Flag | Default | Why you'd use it / effect |
|---|---|---|
| `--cluster-map <path>` | `artifacts/cluster-map.json` | Point at a different artifact — e.g. to validate a candidate artifact before promoting it. |
| `--model-profiles <path>` | `artifacts/model-profiles.json` | Same, for the profiles artifact. |

Also prints any **coverage gaps** — `(model_id, cluster_id)` pairs where that model has no
cluster-level calibration and would fall back to its global rate at decision time (see
[scoring-formula.md](scoring-formula.md)). Use this after calibration to see, before ever routing a
real request, exactly which model/cluster combinations are running on the fallback path.

## `router runtime decide`

Prints the selected model and full score table for one prompt — manual/offline testing of a
routing decision, no HTTP service in front of it.

| Flag | Default | Why you'd use it / effect |
|---|---|---|
| `--prompt <text>` | **required** | The prompt to route. |
| `--lambda <float>` | **required, no default** | The cost-weight lambda. There's deliberately no default — see [scoring-formula.md](scoring-formula.md) for what changing it does to the selected model, and `config/calibration.yaml`'s `lambda_sweep` for a reference set of values worth trying one at a time. |
| `--cluster-map <path>` | `artifacts/cluster-map.json` | Point at a different artifact. |
| `--model-profiles <path>` | `artifacts/model-profiles.json` | Point at a different artifact. |
| `--json` | off (human-readable text) | Print the full `RoutingDecision` as one JSON object instead of formatted text — use this when scripting against the output rather than reading it. |

**Effect of `--json`:** the human-readable path prints cluster/selection/score-table/exclusions as
separate lines; `--json` instead dumps every field (including ones not shown in the text view, like
per-candidate `error_source` for each scored model) as a single machine-readable object.

See also: [README.md](README.md) (where these commands sit in the overall flow),
[config-map.md](config-map.md) (which config file each command reads).
