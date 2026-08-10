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

## Project layout

```
src/router/
  cli.py             # top-level `router` command: mounts pipeline/ and runtime/ as subcommands
  common/            # shared by pipeline AND runtime — nothing here is stage- or mode-specific
    config.py          # loads config/*.yaml
    scoring.py          # predicted_error + lambda*normalised_cost -> selected model
    embedding.py        # text preprocessing + sentence-transformers encoding (batch or single-query)
    artifacts.py        # shared JSON Schema validation + write helpers for pipeline output
    logging_config.py   # logging setup
  pipeline/          # the offline pipeline — see "What the pipeline does" above
    cli.py              # the `router pipeline` command group
    corpus.py           # dataset loading: tagging, exact dedup
    clustering/           # K-means + cluster-map.json assembly + t-SNE viz
config/              # human-edited YAML: embedding/clustering config
artifacts-schema/    # versioned JSON Schema for cluster-map.json
artifacts/           # pipeline output (git-ignored, regenerable)
docs/specs/          # design docs, including the original cluster-routing implementation plan
tests/               # mirrors src/router/'s common/pipeline/runtime split
```

## Tests

```bash
uv run pytest
```

All tests are pure — no network, no model downloads, no Docker, no subprocess calls.
