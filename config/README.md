# config/

Human-authored configuration for the offline pipeline (`src/router/pipeline/`). Plain YAML, no
schema validation — schema is reserved for pipeline *output* artifacts (see
`../artifacts-schema/`), not input config.

## Files

| File | Purpose |
|---|---|
| `embedding.yaml` | The embedding model, its dimensionality, normalization/distance choices, and the text preprocessing rule (truncation, line-ending normalization) applied before embedding. These values are copied verbatim into `artifacts/cluster-map.json`'s `embedding` block when the pipeline runs, so the artifact is self-describing — nothing downstream should need to re-read this file. |
| `clustering.yaml` | K-means run configuration: candidate K values, the default to promote, the random seed, and the number of initializations. |
| `models.yaml` | The candidate model roster (id, provider, cost, context window, max tokens) used by calibration and, eventually, by the runtime to know which models it's allowed to route to. |

## `models.yaml` is the single source of truth for the candidate roster

This project has no separate runtime configuration format — no `.env`-based tier ladder to keep in
sync — both `src/router/pipeline/calibration/` and `src/router/runtime/` read the same `models.yaml`. The
runtime hard-fails at startup if a configured candidate model has no calibration profile in
`artifacts/model-profiles.json` — see
`../docs/specs/2026-08-04-cluster-routing-implementation-plan.md` (Phase 2) for the fuller
reasoning — so adding, removing, or repricing a model here and forgetting to recalibrate becomes a
loud failure, not silent drift.

## Why YAML, and why not the artifact schema shape directly

Config here is meant to be edited by a person: YAML supports comments, which is why every field
above and in the files themselves is documented inline rather than in a separate schema. Pipeline
*output* (`artifacts/cluster-map.json`) is a different kind of file — machine-generated, consumed
by another program (the runtime) — which is why that one has a real JSON Schema contract in
`../artifacts-schema/` instead.
