# clustering-based-llm-router

An embedding-clustering based LLM router: embed a prompt, assign it to a K-means cluster fit over
a coding-task corpus, and pick the model minimising `predicted_error + lambda * normalised_cost`
using per-cluster error rates measured by calibration. Two parts, both pure Python:

1. **The offline pipeline** (`src/router/pipeline/`) — builds the corpus, embeds it, runs K-means,
   calibrates per-model per-cluster error rates, and evaluates the routing formula over a held-out
   split.
2. **The online runtime** (`src/router/runtime/`) — given one live prompt, embeds it, assigns it
   to a cluster, and selects a model.

## Prerequisites

**Hard requirements — needed for anything here, including just running the tests:**

- **Python >= 3.11** (`pyproject.toml`'s `requires-python`).
- **[uv](https://docs.astral.sh/uv/) >= 0.11.5** — this repo's build backend is pinned to
  `uv_build>=0.11.5,<0.12.0` (`pyproject.toml`'s `[build-system]`); an older `uv` won't satisfy it.
- **git**, to clone the repo.
- Internet access and a few GB of free disk: `uv sync` downloads all Python dependencies
  (including `torch`/`sentence-transformers`), and the pipeline itself downloads datasets from
  Hugging Face on first use — the full corpus pull alone is a one-time ~4GB download (mostly
  Multi-SWE-RL; see "What the pipeline does" below), cached under `~/.cache/huggingface/hub`.

**Can be skipped initially** — none of this is needed to run `corpus` → `embed` →
`build-artifact`, or to run the test suite. It's only needed once you get to
`validate-graders`/`calibrate`/`evaluate`:

- **Node.js >= 22.19.0 + npm**, to install the
  [Pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) CLI
  (`npm install -g @earendil-works/pi-coding-agent`, engine requirement per its own
  `package.json`) — see "Aligning `config/models.yaml` with the Pi coding agent" below.
- **Docker** — required for `swe-smith` and `swe-gym`, both `gradeable_sources` now enabled in
  `config/calibration.yaml` (`bigcodebench`/`ds1000` alone don't need it — they're self-contained).
  `docker info` must succeed on whichever host runs `validate-graders`/`calibrate`/`evaluate`; no
  further setup beyond a running daemon. SWE-smith images are large (~3.2-3.5GB measured) and span
  ~128 repos; SWE-Gym images span its own 11 repos and vary more widely (moto ~2.9GB, pandas
  ~6.5GB measured) — `grading/dockerexec.py` keeps a bounded LRU cache of the 15 most recently used
  images (~50GB ceiling, shared across both sources) rather than either accumulating every one ever
  pulled or re-pulling on every single grading call; a `calibrate` run touching more than 15
  distinct tasks will still see some re-pulls once the cache rolls over. Budget real wall-clock
  time for this — a handful of tasks can take 10+ minutes when images aren't already cached, and a
  SWE-Gym task whose patch touches pandas' build config can trigger a several-minute Cython rebuild
  on top of that (automatic, via the image's own editable-install build backend — see
  `grading/swegym.py`'s module docstring).
- **Outbound access to github.com**, for the `pi` runner against swe-smith/swe-gym:
  `repo_context.py` clones each task's real repo (swe-smith's `swesmith/{owner}__{project}.{hash}`
  mirror, or swe-gym's own upstream repo at `base_commit` — both real, public GitHub repos — see
  "What the pipeline does" below) so the agent gets real repo access instead of a bare paragraph.
  Bare clones are cached per repo under `.cache/repo_context/clones/` (bounded LRU, same idea as
  the Docker image cache above); a disposable `git worktree` per task is what the agent's tools
  actually see, removed again once that task's `pi` call finishes.
- **A GPU** — entirely optional. `torch` is pinned to the CPU-only wheel in this repo (see below);
  a GPU only makes `embed` faster, and needs its own `torch` install to take advantage of.

### Setup

```bash
git clone https://github.com/hamad1safdar/clustering-based-llm-router.git
cd clustering-based-llm-router
uv sync
uv run pytest -q   # sanity check — fully offline, should pass with no setup beyond `uv sync`
```
From here, see "Running the pipeline" below.

## Design

A standalone, Python-only project: the offline pipeline and a from-scratch Python runtime, so the
routing logic that must agree between calibration/evaluation and live routing — nearest-centroid
assignment, the scoring formula — is one shared module (`src/router/common/`), not two
implementations kept in sync by hand.

## Runtime status

Deliberately **stateless** — no session, no escalation, no budget caps.

```bash
uv run router runtime validate                      # check both artifacts, no embedding, no model download
uv run router runtime decide --prompt "..." --lambda 0.05
```

- `runtime/context.py` — `load_routing_context` loads both artifacts and hard-fails on ~13 rules
  (schema/version/geometry, non-finite centroids, `profiles.cluster_map_id != cluster_map.artifact_id`,
  the configured embedding model disagreeing with the artifact's, a candidate with no calibration
  profile, a non-positive lambda, and more) before any decision is possible. A reproducibility
  digest (`sha256` over lambda + both artifact ids + the candidate roster) is computed once here.
- `runtime/decide.py` — `decide_from_vector` (pure: assign -> score -> select -> audit record) and
  `decide` (adds the `common/embedding.embed_one` call). Logs a WARNING whenever the selection
  falls back to a model's global error rate instead of the assigned cluster's.
- `runtime/cli.py` — `router runtime validate` / `decide` (`--json` for machine-readable output on
  `decide`).

**Verified two ways:**
1. 29 pure tests against a hand-authored k=3/dim=4 fixture pair (`tests/runtime/fixtures/`),
   covering every hard-fail rule and the full lambda-sweep scoring behaviour with zero
   network/model access.
2. **Live, against real pipeline-produced artifacts** — `validate` and `decide` both run
   successfully against a real (if tiny, k=5) `cluster-map.json`/`model-profiles.json` pair with
   the real embedding model and real calibrated candidates. Confirmed the lambda sweep actually
   changes the selected model on real calibration numbers, not just fixture math.

One caveat this exposed: each `decide` **invocation** pays one cold encoder load (~13s) since the
in-process cache in `common/embedding.py` doesn't survive across separate CLI processes. Nothing
yet keeps the encoder warm *across* invocations (would need a long-lived process, e.g. a future
HTTP service).

## Running the pipeline

Assumes you've already done the one-time `uv sync` from "Prerequisites" above. `torch` is pinned
to the CPU-only wheel there (see `pyproject.toml`'s `[tool.uv.sources]`) — PyPI's default `torch`
pulls a full CUDA stack (multiple GB) that this pipeline doesn't need.

```bash
# Recommended: dry run first. --sample caps EACH source at N rows, so it still exercises every
# loader. CPU throughput for jina-embeddings-v2-base-code on this corpus is not yet benchmarked —
# this is how you find out before committing to the full run. For multi-swe-rl specifically,
# --sample reads the smallest file per language first, so a dry run costs megabytes, not the full
# ~4GB — but that also means a capped multi-swe-rl sample is biased toward small repos, not random.
uv run router pipeline corpus --sample 200
uv run router pipeline embed
uv run router pipeline build-artifact

# Full run
uv run router pipeline corpus            # ~29,301 rows before dedup (~4GB one-time download for
                                          # multi-swe-rl, cached under ~/.cache/huggingface/hub)
uv run router pipeline embed
uv run router pipeline build-artifact --k 24

# Or all three steps in one shot
uv run router pipeline run-all --k 24
```

Intermediate files (`corpus.jsonl`, `embeddings.npz`) are written to `.cache/` at the repo root —
not `artifacts/`, since they're working files, not the final artifact.

Calibration and evaluation, once `artifacts/cluster-map.json` exists:

```bash
# GATE — run this first. If reference isn't ~100% and null isn't ~0%, stop; nothing below means anything.
uv run router pipeline validate-graders

uv run router pipeline calibrate   # writes artifacts/model-profiles.json
uv run router pipeline evaluate    # holdout run + lambda-sweep report
```

`evaluate` executes real model calls for every holdout task against every non-control model in
`config/models.yaml` — for a `runner: pi` entry that means a live `pi -p` subprocess per task, so
whatever it points at must be reachable first — see "Aligning `config/models.yaml` with the Pi
coding agent" below.

### Aligning `config/models.yaml` with the Pi coding agent

Every `runner: pi` candidate is invoked through the
[Pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) CLI
(`pi --provider <provider> --model <model_id> -p ...`, one subprocess per task — see
`pipeline/calibration/runner.py`'s `run_pi`). This repo only *names* providers/models in
`config/models.yaml`; it never manages credentials or talks to a model API directly, so each
`provider`/`model_id` pair needs a matching entry in your own Pi provider config
(`~/.pi/agent/models.json`, or `$PI_CODING_AGENT_DIR/models.json` if you've overridden that).

Minimal example matching this repo's OpenAI candidates:

```json
{
  "providers": {
    "openai": {
      "apiKey": "<your OpenAI API key>",
      "models": [
        {
          "id": "gpt-5-nano",
          "name": "GPT-5 Nano",
          "contextWindow": 400000,
          "maxTokens": 128000,
          "cost": { "input": 0.05, "output": 0.40, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

| `config/models.yaml` | Pi provider config | Note |
|---|---|---|
| `provider` | `providers.<name>` key | Must match exactly — `provider: openai` needs a `providers.openai` entry. |
| `model_id` | `providers.<name>.models[].id` | Must match exactly — this is the `--model` value `run_pi` passes. |
| `context_window` | `models[].contextWindow` | Informational on both sides; neither enforces it against the other. |
| `max_tokens` | `models[].maxTokens` | Keep this in line with what a real completion for these prompts actually needs. |
| `cost_input` / `cost_output` | `models[].cost.input` / `.output` | **Different units** — `config/models.yaml` is $ per 1k tokens, Pi's config is $ per 1M tokens; multiply by 1000 going from ours to theirs. |

`reference`/`null`-runner candidates (the grader-validation controls) never go through this path
at all — they're synthesized directly in `calibrate.py`, not run through `pi`.

## What the pipeline does

1. **`corpus`** — loads five benchmark/training datasets from Hugging Face, tags and dedups them into one corpus:

   | Source | Rows used | License |
   |---|---|---|
   | SWE-smith | 20,000 sampled (of ~59,136 available) | MIT |
   | SWE-Gym | 2,438 (all) | MIT |
   | BigCodeBench-Instruct | 1,140 (all, `v0.1.4` split) | Apache-2.0 |
   | DS-1000 | 1,000 (all) | CC-BY-SA-4.0 |
   | Multi-SWE-RL | ~4,723 (all of batch `data_20240601_20250331`) | unverified — see below |

   **Multi-SWE-RL is a clustering-corpus source only** — there's no grader for it (that would need
   a per-repo Docker image per instance), so `config/calibration.yaml`'s `gradeable_sources` never
   lists it. It also can't use the generic `load_dataset(hf_id, split=split)` path the other four
   sources share: its batch-1 files have per-repo-heterogeneous nested fields, and Arrow schema
   unification across them fails outright (`TypeError: Couldn't cast array of type string to
   null`) — the same reason the HF dataset viewer is broken for this dataset. `corpus.py` instead
   fetches each file individually via `huggingface_hub.hf_hub_download` and parses it with plain
   `json.loads`. The dataset has no `problem_statement` field; its `title`/`body` is the *pull
   request* (a solution description), so the corpus text comes from `resolved_issues[].title`+
   `.body` instead (the *issue*, i.e. the actual problem) — measured 750 median chars vs. 128 for
   the PR text, present in every record sampled. Only the initial release batch is pulled (a
   second, larger upstream batch is deliberately not); even so, that's ~4GB (JSONL is
   row-oriented, so the ~750 chars of usable text per record can't be separated from the multi-MB
   test-log fields stored alongside it). **License is recorded as `unverified`** rather than
   resolved one way or the other: the dataset card states "licensed under CC0 ... subject to any
   intellectual property rights in the dataset owned by Bytedance" and that use "must comply with
   [the underlying repositories'] respective licenses", while the repo's own HuggingFace metadata
   tag says `license: other` — get sign-off from whoever owns data licensing before using this
   corpus beyond local experimentation.

   **SWE-Gym is gradeable too, the same way swe-smith is.** An earlier assumption — that grading it
   would require hand-resolving 161 distinct `version` strings against SWE-bench's own
   environment-setup constants, since rows carry no `environment_setup_commit` — turned out to be
   wrong once checked directly: prebuilt per-instance Docker images already exist and are public
   (Docker Hub, `xingyaoww/sweb.eval.x86_64.*`), and no repo-specific install step is needed at
   grade time (verified against `getmoto/moto` and, more demandingly, `pandas-dev/pandas`, whose
   editable install rebuilds automatically on next import when its build config changes). See
   `grading/swegym.py`'s module docstring for the full verification.

2. **`embed`** — embeds the corpus in-process with `sentence-transformers`
   (`jinaai/jina-embeddings-v2-base-code`, 768-dim, CPU) per the rule declared in
   `config/embedding.yaml`. No server, no HTTP call.

3. **`build-artifact`** — runs K-means at `config/clustering.yaml`'s `default_k` (override with
   `--k`), prints diagnostics (inertia, cluster-size spread), assembles `cluster-map.json`,
   validates it against the schema, and writes it to `artifacts/` (git-ignored — outputs are
   local-only for now, regenerable by re-running the pipeline).

4. **`validate-graders`** — GATE, run before trusting anything downstream. Runs each gradeable
   source's own reference (gold) solution and an empty (null) solution through its grader with no
   model involved. Reference must score ~100% pass, null ~0% — this is what proves the grader
   itself discriminates correct from incorrect code, independent of any model's actual ability. Any
   shortfall should show up as `error_harness`/`error_timeout` in the printed outcome counts —
   never as `fail`, which would mean the grader itself, not the (gold/empty) solution, is broken.
   Repeatable `--source` narrows the gate to one or more sources at a time (e.g.
   `--source swe-smith`), and the sample is drawn with `random.sample` rather than the first N
   rows, so a source grouped by repo (like swe-smith's 128) doesn't always gate the same handful.

5. **`calibrate`** — selects tasks stratified across `cluster-map.json`'s clusters (from
   `config/calibration.yaml`'s gradeable sources), runs every model in `config/models.yaml`
   (including the `reference`/`null` controls) against them, and writes the smoothed per-cluster
   error rates to `model-profiles.json`.

6. **`evaluate`** — re-selects the same deterministic split, runs the *held-out* tasks against
   every model, and reports resolution rate / mean cost / model-selection distribution across
   `config/calibration.yaml`'s `lambda_sweep`, plus always-strongest / always-cheapest / oracle
   baselines. Routing decisions use only the calibration profiles, never anything from the holdout
   run itself — the same constraint a live router would have.

**swe-smith and swe-gym tasks give the `pi` agent real repo access, not just a paragraph.**
`repo_context.py` clones the task's real repo (swe-smith's GitHub mirror at `HEAD`, or swe-gym's
own upstream repo at `base_commit`) and checks out a disposable `git worktree` for each `run_pi`
call, so the agent's already-enabled read/bash/edit/write tools have a real, isolated working tree
to explore and fix rather than nothing to point them at. The resulting solution is captured via
`git diff` on that worktree — a tool-using agent's actual edits, not a hand-written diff parsed
from its text response — falling back to the old text-extraction path only if the agent responds
with prose instead of using its tools. A clone/checkout failure is recorded as `error_harness`
(never scored as a wrong answer) before `pi` is even invoked.
bigcodebench/ds1000 are unaffected — they're self-contained snippet tasks with no repo to clone.

**Calibration's target is pipeline completeness, not research-grade numbers.** The tiny task
volume this pass runs at (`tasks_per_cluster` in `config/calibration.yaml`) is chosen to exercise
the full pipeline cheaply, not to produce statistically confident error rates — see
`config/calibration.yaml`'s comments and `runner.py`'s docstring for the `reference`/`null`/`pi`
backend split that lets the grader itself be proven correct independent of any model's actual
coding ability.

## Config and artifact schema

`config/*.yaml` is human-authored input; `artifacts-schema/*.json` is a machine-checked contract on
the pipeline's *output*. Different formats for a reason: YAML supports the inline comments a person
editing config wants, while the artifacts are machine-generated and machine-consumed (by the
runtime), so they get a real JSON Schema (Draft 2020-12, validated via
`jsonschema.Draft202012Validator`) that both sides fail loudly against instead of silently drifting.

| Config file | Purpose |
|---|---|
| `embedding.yaml` | Embedding model, dimensionality, normalization/distance, and the text preprocessing rule — copied verbatim into `cluster-map.json`'s `embedding` block so the artifact is self-describing. |
| `clustering.yaml` | K-means run config: candidate `k` values, the default to promote, seed, `n_init`. |
| `models.yaml` | The candidate model roster (id, provider, cost, context window, max tokens). |

`models.yaml` is the single source of truth for the candidate roster — both
`pipeline/calibration/` and `runtime/` read the same file, and the runtime hard-fails at startup if
a configured candidate has no calibration profile (see "Runtime status" above), so repricing or
swapping a model here and forgetting to recalibrate is a loud failure, not silent drift.

Schema validation catches structure (missing fields, wrong types, `additionalProperties: false`)
but can't express cross-field invariants, so those are checked separately in code at write time:

- `cluster-map.json` (`pipeline/clustering/cluster_map.py`): every centroid's length equals
  `embedding.dimensions`; `clusters.length` equals `kmeans.k`; cluster ids are contiguous `0..k-1`
  ascending; every centroid value is finite.
- `model-profiles.json` (`pipeline/calibration/profiles.py`): `cluster_map_id` equals the
  referenced cluster map's `artifact_id` (this is the linkage that makes cluster id 7 mean the same
  thing in both files — a mismatch would silently permute the error table); every cluster key falls
  within `0..cluster_count-1`; `number_succeeded + number_failed == number_of_tasks`.

Any future consumer of these artifacts must re-check these invariants itself — passing schema
validation alone doesn't guarantee them.

## Logging

Every command logs progress through Python's stdlib `logging` — not `print`, not ad-hoc output.
Configured once per invocation by the CLI's Typer callback (`--log-level`, default `INFO`).

Deliberately plain — one console handler (`stderr`), one readable line per call, no structured
fields: `logger.info("loaded 1140 rows from bigcodebench")`. Call sites inline any detail worth
keeping directly into the message string rather than attaching it as a separate field, so there's
nothing to configure or look up beyond reading the line itself.

A log **file** is opt-in via `--log-file <path>` (appended to, not overwritten, so re-running a
command doesn't erase an earlier run) — useful for `calibrate`/`evaluate`, which can run long
enough that you want a persistent record to grep through afterward rather than relying on terminal
scrollback:
```bash
uv run router pipeline --log-level DEBUG --log-file calibrate.log calibrate
```
With `--log-level DEBUG`, `calibrate`/`evaluate` also log the expected vs. actual solution for
every `runner: pi` task (`calibrate.py`'s `run_and_grade`) — useful for seeing exactly what a model
produced when a task unexpectedly failed.

Level guidance applied consistently across the pipeline: **INFO** for stage boundaries and
per-item progress (the bulk of it — per-source/per-k/per-task, including a running progress
fraction for the two loops that run one real model/grading call per task —
`calibrate_model`/`run_holdout_outcomes`, via the shared `run_and_grade` dispatcher and
`runner.py`'s `run_pi`); **WARNING** for excluded grading outcomes
(`error_missing_dep`/`error_timeout`/`error_harness`) and a timed-out `pi` call — an environment
problem worth surfacing the moment it happens, not just in the final aggregate dict; **ERROR** for
hard failures (schema validation, a missing `pi` binary).

## Project layout

```
src/router/
  cli.py             # top-level `router` command: mounts pipeline/ and runtime/ as subcommands
  common/            # shared by pipeline AND runtime — nothing here is stage- or mode-specific
    config.py          # loads config/*.yaml
    assign.py          # nearest-centroid assignment
    scoring.py          # predicted_error + lambda*normalised_cost -> selected model
    embedding.py        # text preprocessing + sentence-transformers encoding (batch or single-query)
    artifacts.py        # shared JSON Schema validation + write helpers for pipeline output
    logging_config.py   # logging setup
  pipeline/          # the offline pipeline — see "What the pipeline does" above
    cli.py              # the `router pipeline` command group
    corpus.py           # dataset loading: tagging, exact dedup
    evaluate.py          # holdout replay of the routing formula, lambda sweep, baselines
    clustering/           # K-means + cluster-map.json assembly
    calibration/           # task selection -> grade -> smoothed rates -> model-profiles.json
      grading/               # one grader per data source
  runtime/           # the online routing runtime — see "Runtime status" above
    cli.py              # the `router runtime` command group: validate, decide
    context.py           # loads + hard-validates both artifacts -> RoutingContext
    decide.py            # decide_from_vector / decide -> RoutingDecision
config/              # human-edited YAML: embedding/clustering/calibration/model-roster config
artifacts-schema/    # versioned JSON Schema for cluster-map.json / model-profiles.json
artifacts/           # pipeline output (git-ignored, regenerable)
tests/               # mirrors src/router/'s common/pipeline/runtime split
```

## Tests

```bash
uv run pytest
```

All tests are pure — no network, no model downloads, no Docker, no `pi`/`llama` subprocess calls.
They cover preprocessing, dedup, K-means determinism/dtype, artifact assembly + schema validation
(including the cross-field invariants the schema itself can't express — see "Config and artifact
schema" above), nearest-centroid assignment, the smoothing/scoring arithmetic,
grading outcome classification, the logging setup, and the runtime's hard-fail validation +
lambda-sweep scoring against the `tests/runtime/fixtures/` artifact pair. Docker-based grading
(`grading/dockerexec.py`'s sentinel classification, `grading/swesmith.py`'s script construction) is
tested by monkeypatching `subprocess.run` — real `docker run` calls are exercised only by
`validate-graders`, not `pytest`. `test_registration.py` guards against a source being listed in
`gradeable_sources` but missing from one of the dispatch tables a real run depends on. The
`validate-graders` gate above is the actual correctness proof for grading — it needs real datasets
(and, for swe-smith, a reachable Docker daemon) and isn't part of `pytest`.
