# clustering-based-llm-router

An embedding-clustering based LLM router: embed a prompt, assign it to a K-means cluster fit over
a coding-task corpus, and pick the model minimising `predicted_error + lambda * normalised_cost`
using per-cluster error rates measured by calibration. Two parts, both pure Python:

1. **The offline pipeline** (`src/router/pipeline/`) — builds the corpus, embeds it, runs K-means,
   calibrates per-model per-cluster error rates, and evaluates the routing formula over a held-out
   split.
2. **The online runtime** (`src/router/runtime/`) — given one live prompt, embeds it, assigns it
   to a cluster, and selects a model.

## Design

Planned in `docs/specs/2026-08-04-cluster-routing-implementation-plan.md`, as a standalone,
Python-only project: the offline pipeline and a from-scratch Python runtime, so the routing logic
that must agree between calibration/evaluation and live routing — nearest-centroid assignment, the
scoring formula — is one shared module (`src/router/common/`), not two implementations kept in
sync by hand.

## Running the pipeline

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

`torch` is pinned to the CPU-only wheel (see `pyproject.toml`'s `[tool.uv.sources]`) — PyPI's
default `torch` pulls a full CUDA stack (multiple GB) that this pipeline doesn't need.

```bash
# Recommended: dry run first. --sample caps EACH source at N rows, so it still exercises every
# loader. CPU throughput for jina-embeddings-v2-base-code on this corpus is not yet benchmarked —
# this is how you find out before committing to the full run.
uv run router pipeline corpus --sample 200
uv run router pipeline embed
uv run router pipeline cluster

# Full run
uv run router pipeline corpus            # ~24,578 rows before dedup
uv run router pipeline embed
uv run router pipeline cluster           # review diagnostics for k=16/24/32
uv run router pipeline build-artifact --k 24

# Or all four steps in one shot
uv run router pipeline run-all --k 24
```

Intermediate files (`corpus.jsonl`, `embeddings.npz`) are written to `.cache/` at the repo root —
not `artifacts/`, since they're working files, not the final artifact.

Calibration and evaluation, once `artifacts/cluster-map.json` exists:

```bash
# GATE — run this first. If reference isn't ~100% and null isn't ~0%, stop; nothing below means anything.
uv run router pipeline validate-graders

uv run router pipeline calibrate   # writes artifacts/model-profiles.json
```

### Calibrating against Groq

The three real candidates in `config/models.yaml` (`llama-3.1-8b-instant`, `openai/gpt-oss-120b`,
`llama-3.3-70b-versatile`) run on [Groq](https://groq.com)'s free tier — no local model this pass.

**Setup, outside this repo:** add a `groq` provider to your own Pi provider config (e.g.
`~/.pi/agent/models.json`) pointing at Groq's API with a Groq API key. This repo only names the
provider in `config/models.yaml`; it doesn't manage credentials.

**Free-tier rate limits** (per Groq's own docs, `console.groq.com/docs/rate-limits`, checked
August 2026): each of these three models allows 30 requests/minute on the free tier, with daily
request caps from 1,000–14,400 and **tokens-per-minute (TPM) caps of 6,000–12,000** depending on
the model. In practice **TPM binds long before RPM or the daily cap do** — Groq reserves a
request's prompt + `max_tokens` against your TPM budget up front, so a single coding-task prompt
plus a few thousand completion tokens can exhaust a 6,000 TPM budget in one or two calls.
`pipeline/calibration/runner.py`'s `RateLimiter` paces every call's *request rate* against
`rate_limit_rpm` in `config/models.yaml`, but that alone won't prevent a TPM rejection. When Groq
rejects a call this way, its error names an exact cooldown ("Please try again in Ns") — `runner.py`
parses that and retries after precisely that long (`MAX_RATE_LIMIT_RETRIES` times, each capped at
`MAX_RATE_LIMIT_WAIT_SECONDS`) rather than giving up immediately. Only once retries are exhausted
is the task excluded from that model's error rate (`error_harness`) rather than counted as a wrong
answer. Keep `maxTokens` in your Pi provider config modest (e.g. 2048, not a model's full output
ceiling) — these calibration prompts explicitly ask for just a function body or code snippet, and
a needlessly large `max_tokens` reserves far more of your TPM budget than any real completion here
will use.

**Cost fields are not $0.** `config/models.yaml`'s `cost_input`/`cost_output` record Groq's real
published per-token prices, not what we're actually paying on the free tier — if every candidate
were priced at $0, `normalise_costs` would collapse to all-zero and the lambda sweep would have
nothing to trade accuracy against.

## What the pipeline does

1. **`corpus`** — loads four benchmark/training datasets from Hugging Face, tags and dedups them into one corpus:

   | Source | Rows used | License |
   |---|---|---|
   | SWE-smith | 20,000 sampled (of ~59,136 available) | MIT |
   | SWE-Gym | 2,438 (all) | MIT |
   | BigCodeBench-Instruct | 1,140 (all, `v0.1.4` split) | Apache-2.0 |
   | DS-1000 | 1,000 (all) | CC-BY-SA-4.0 |

   **Multi-SWE-RL is deliberately excluded** — a 23.8GB download with no independently-verifiable
   row count and a license conflict between its README (CC0) and its repo metadata tag (`other`).
   Revisit once someone confirms the actual license terms; adding it back is additive (a new entry
   in `corpus.py`'s `SOURCE_METADATA` plus a loader for its non-parquet JSONL layout), not a
   redesign.

2. **`embed`** — embeds the corpus in-process with `sentence-transformers`
   (`jinaai/jina-embeddings-v2-base-code`, 768-dim, CPU) per the rule declared in
   `config/embedding.yaml`. No server, no HTTP call; see `config/README.md`.

3. **`cluster`** — runs K-means for every candidate `k` in `config/clustering.yaml` and reports
   diagnostics (inertia, cluster-size spread). **Choosing which `k` to promote is a human decision**,
   not automated.

4. **`build-artifact`** — assembles `cluster-map.json` at the chosen `k`, validates it against the
   schema, and writes it to `artifacts/` (git-ignored — outputs are local-only for now,
   regenerable by re-running the pipeline).

5. **`validate-graders`** — GATE, run before trusting anything downstream. Runs each gradeable
   source's own reference (gold) solution and an empty (null) solution through its grader with no
   model involved. Reference must score ~100% pass, null ~0% — this is what proves the grader
   itself discriminates correct from incorrect code, independent of any model's actual ability.

6. **`calibrate`** — selects tasks stratified across `cluster-map.json`'s clusters (from
   `config/calibration.yaml`'s gradeable sources), runs every model in `config/models.yaml`
   (including the `reference`/`null` controls) against them, and writes the smoothed per-cluster
   error rates to `model-profiles.json`.

**Calibration's target is pipeline completeness, not research-grade numbers.** The tiny task
volume this pass runs at (`tasks_per_cluster: 4`) is chosen to exercise the full pipeline cheaply
on Groq's free tier, not to produce statistically confident error rates — see
`config/calibration.yaml`'s comments and `runner.py`'s docstring for the `reference`/`null`/`pi`
backend split that lets the grader itself be proven correct independent of any model's actual
coding ability.

## Logging

Every command logs progress through Python's stdlib `logging` — not `print`, not ad-hoc output.
Configured once per invocation by the CLI's Typer callback (`--log-level`, default `INFO`).

Deliberately plain — one console handler (`stderr`), one readable line per call, no structured
fields, no log file: `logger.info("loaded 1140 rows from bigcodebench")`. Call sites inline any
detail worth keeping directly into the message string rather than attaching it as a separate
field, so there's nothing to configure or look up beyond reading the line itself.

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
    clustering/           # K-means + cluster-map.json assembly + t-SNE viz
    calibration/           # task selection -> grade -> smoothed rates -> model-profiles.json
      grading/               # one grader per data source
config/              # human-edited YAML: embedding/clustering/calibration/model-roster config
artifacts-schema/    # versioned JSON Schema for cluster-map.json / model-profiles.json
artifacts/           # pipeline output (git-ignored, regenerable)
docs/specs/          # design docs, including the original cluster-routing implementation plan
tests/               # mirrors src/router/'s common/pipeline/runtime split
```

## Tests

```bash
uv run pytest
```

All tests are pure — no network, no model downloads, no Docker, no `pi`/`llama` subprocess calls.
They cover preprocessing, dedup, K-means determinism/dtype, artifact assembly + schema validation
(including the cross-field invariants the schema itself can't express — see
`artifacts-schema/README.md`), nearest-centroid assignment, the smoothing/scoring arithmetic,
grading outcome classification, and the logging setup. The `validate-graders` gate above is the
actual correctness proof for grading — it needs real datasets and isn't part of `pytest`.
