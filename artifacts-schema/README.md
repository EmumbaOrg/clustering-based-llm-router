# artifacts-schema/

Versioned JSON Schema contracts for the offline pipeline's **output** — not its input config (see
`../config/` for that). Both `src/router/pipeline/` (which writes these artifacts) and
`src/router/runtime/` (which reads them) validate against these same files, so the two sides can
never silently disagree about the shape of an artifact.

## Files

| File | Validates |
|---|---|
| `cluster-map.schema.json` | `artifacts/cluster-map.json` — the embedding + K-means clustering output. Draft 2020-12. |
| `model-profiles.schema.json` | `artifacts/model-profiles.json` — the calibration output: per-model, per-cluster error rates. Draft 2020-12. Versioned independently from `cluster-map.schema.json`; linked to a specific cluster map via `cluster_map_id`, not by modifying that schema. |

## What the schema does and doesn't check

Standard JSON Schema validation (via `jsonschema.Draft202012Validator` on the Python side) catches
structural problems: missing required fields, wrong types, unrecognized extra fields
(`additionalProperties: false` throughout), and fixed-value fields like `normalisation: "l2"` or
`centroid_dtype: "float64"`.

It **cannot** express cross-field invariants — JSON Schema has no way to say "this array's length
must equal that sibling field's value." Three such invariants matter here and are enforced by
`src/router/pipeline/clustering/cluster_map.py` at write time instead:

- every `centroid` array's length equals `embedding.dimensions`
- `clusters.length` equals `kmeans.k`
- cluster `id`s are contiguous `0..k-1`, appearing in ascending order

Any future consumer of this artifact (including the runtime reader) must re-check these three
invariants itself — passing schema validation alone does not guarantee them.

`model-profiles.schema.json` has its own set, enforced by `src/router/pipeline/calibration/profiles.py`:

- `cluster_map_id` equals the referenced cluster map's `artifact_id` — this is the linkage that
  makes cluster id 7 mean the same thing in both files; a mismatch silently permutes the error
  table, so consumers must hard-fail on it rather than degrade
- every cluster key in a model's `clusters` object falls within `0..cluster_count-1`
- `number_succeeded + number_failed == number_of_tasks` for every `global`/per-cluster block
- `raw_error_rate`/`smoothed_error_rate` are consistent with the counts they were computed from

## Why a real schema file instead of a shared doc

A schema is executable — both sides can fail loudly on a mismatch instead of a human having to
notice two docs drifted apart. This choice was made explicitly during planning: see
`../docs/specs/2026-08-04-cluster-routing-implementation-plan.md` (Phase 2) for the fuller
reasoning, including why centroids are `float64` and why the artifact stores the embedding model's
preprocessing rule rather than leaving it to each consumer to reimplement. That plan's runtime
phases (1, 3, 4) targeted the TypeScript extension and are superseded here — the runtime is now
`src/router/runtime/`, built in Python so it can import `src/router/common/` directly instead of
re-deriving the same rules in another language — but Phase 2's artifact contract is unchanged and
still binding.
