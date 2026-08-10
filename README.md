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

```bash
uv run router pipeline corpus --sample 200   # dry run: caps EACH source at N rows
uv run router pipeline corpus                # full run
```

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

## Tests

```bash
uv run pytest
```

All tests are pure — no network, no model downloads, no Docker, no subprocess calls.
