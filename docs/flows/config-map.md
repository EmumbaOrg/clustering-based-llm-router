# Config file map

Which `config/*.yaml` feeds which pipeline stage (or the runtime), and what it actually controls.

```mermaid
flowchart LR
    subgraph Config
        EMB["embedding.yaml"]
        CLU["clustering.yaml"]
        CAL["calibration.yaml"]
        MOD["models.yaml"]
    end

    EMB --> Embed[embed]
    EMB -.copied verbatim into.-> CM[(cluster-map.json)]
    CLU --> Build["build-artifact"]
    CAL --> Calibrate[calibrate]
    MOD --> Calibrate
    MOD --> Runtime["runtime (decide)"]
    CM -.runtime reads embedding FROM HERE, not embedding.yaml.-> Runtime

    classDef cfg fill:#eef2ff,stroke:#4338ca;
    class EMB,CLU,CAL,MOD cfg
```

## `embedding.yaml` → `embed`, then frozen into `cluster-map.json`

Single source of truth for embedding choices: `model_id`, `dimensions`, `normalisation`,
`distance`, and text-preprocessing (`input.normalisation`, `input.truncation`). Its values are
copied verbatim into `cluster-map.json`'s own `embedding` block at `build-artifact` time — **the
runtime must use the artifact's copy, never re-read this file directly** (see
`runtime/context.py::_embedding_config_from_artifact`'s docstring, "the highest-risk silent-drift
point in the runtime design"). `load_routing_context` cross-checks the two only to catch a machine
whose config has drifted from what the artifact was actually built with — see
[runtime-validation.md](runtime-validation.md).

## `clustering.yaml` → `build-artifact`

`default_k` (k-means cluster count), `seed`, `n_init`. Nothing downstream reads this file directly
either — `cluster-map.json` records the `k` actually used (`kmeans.k`), which is what
`check_cluster_map_invariants` validates against.

## `calibration.yaml` → `calibrate` (and `validate-graders`)

The largest config, controlling:
- `gradeable_sources` — which sources `validate-graders`/`calibrate` will grade at all (Docker
  daemon required for swe-smith/swe-gym/multi-swe-rl).
- `tasks_per_cluster`, `category_mix` — see [task-selection.md](task-selection.md).
- `task_timeout_seconds`, `task_timeout_overrides` — per-source grading timeout ceilings (Docker
  sources need much more headroom than the base value; see the file's own inline comments for the
  measured worst cases behind each override).
- `smoothing.method`/`prior_weight` — see [scoring-formula.md](scoring-formula.md) for the shrinkage
  math this drives.
- `seed` — deterministic task selection.
- `lambda_sweep` — a reference set of lambda values worth trying one at a time via
  `router runtime decide --lambda` (not used by `calibrate` itself, and nothing sweeps it
  automatically).

## `models.yaml` → `calibrate` AND the runtime

The candidate model roster — the one file both sides of the project share. Each entry's `runner`
(`reference`/`null`/`pi`) decides how calibration produces a solution for it (see
[README.md](README.md)'s calibration inner-loop); `cost_input`/`cost_output` feed
[scoring-formula.md](scoring-formula.md)'s cost term at runtime. A candidate listed here with no
matching `model-profiles.json` entry is a hard runtime startup failure, not a silent gap — see
[runtime-validation.md](runtime-validation.md).

## What's conspicuously *not* here

There's no config file the runtime reads for `cluster-map.json`/`model-profiles.json`'s own
*paths* or for the embedding config used to score a live request — both come from the artifacts
themselves, by design, so a routing decision's inputs are fully pinned to what calibration actually
produced rather than to whatever `config/` happens to contain at request time.

See also: [README.md](README.md) (where each config file's stage sits in the full pipeline).
