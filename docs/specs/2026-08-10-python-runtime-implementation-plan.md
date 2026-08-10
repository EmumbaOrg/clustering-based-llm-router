# Plan: the Python routing runtime

> Status: approved plan, not yet built. Supersedes the runtime phases (1, 3, 4) of
> `2026-08-04-cluster-routing-implementation-plan.md`, which targeted a TypeScript Pi extension
> that never shipped cluster mode. That document's **Phase 2 (artifact contract) remains binding**
> and is implemented — `artifacts-schema/` is the same contract. Read it for the design *reasoning*
> behind the scoring formula, the fallback chain, and the determinism rules; read this document for
> what actually gets built in Python.

## Context

`routing-poc-background-research.md` §6 defines runtime routing as: embed the prompt → assign to
the nearest K-means centroid → score every eligible candidate as
`predicted_error + lambda x normalised_cost` → pick the argmin.

**That formula is already implemented and tested** in `src/router/common/scoring.py`
(`score_candidates`, `select_model`), because `pipeline/evaluate.py` needs the identical
arithmetic to replay a lambda sweep over the holdout split. Nearest-centroid assignment is
likewise done in `common/assign.py`. So the runtime's remaining work is *not* the routing
mathematics — it is everything around it:

1. Loading both artifacts and **validating them hard** before trusting a single number.
2. Embedding one live prompt using exactly the model and preprocessing rule the corpus was built
   with.
3. Emitting a decision record complete enough to satisfy §8's *"all routing decisions reproducible
   from versioned artifacts"*.

### Decisions already taken

- **Stateless, decision-only.** §6 also says to preserve escalation and budget mechanisms, and §2.4
  to hold the model "for the full agent run". Those are session-scoped: they need a host agent
  driving multi-turn runs, emitting turn events and token counts. With the Pi/TypeScript extension
  out of scope there is no host, so there is nothing to freeze a decision *for* and no usage to
  bill against a budget. Building them now means inventing an untestable session model. Deferred —
  see "Explicitly out of scope".
- **Library + CLI, no service.** No HTTP endpoint until something needs to call one.
- **In-process embedding.** `common/embedding.py` is shared with the pipeline. The old plan's
  HTTP embedding-server wire format existed only because Node cannot run
  `sentence-transformers`; that constraint is gone.

### Consequence worth stating plainly

The old plan's elaborate per-request **degrade** path ("notify and keep the current model" on an
embedding failure) does not survive statelessness: with no session there is no current model to
keep. Every failure is therefore a hard error and a non-zero exit. The one fallback that *does*
survive is cluster-entry → model `global`, because that is about artifact coverage rather than
session state.

---

## Phase 1 — Changes to `common/` (shared, prerequisite)

Three gaps that only surface once there is a second consumer of the artifacts.

### 1a. Share the cross-field invariant checks

Both schemas say, in their own `description` fields, that the invariants JSON Schema cannot express
"must be re-checked by any consumer before trusting the file". Today those checks are private
functions inside the *writers* (`pipeline/clustering/cluster_map.py::_check_cross_field_invariants`,
`pipeline/calibration/profiles.py::_check_cross_field_invariants`). The runtime is that second
consumer, and must not reach into a pipeline private or reimplement them.

Move into `common/artifacts.py`, which already holds `validate_against_schema` /
`write_json_artifact`:

```python
def check_cluster_map_invariants(artifact: dict) -> None
def check_profiles_invariants(artifact: dict) -> None
def validate_cluster_map(artifact: dict) -> None      # schema + invariants
def validate_profiles(artifact: dict) -> None         # schema + invariants
```

The pipeline modules keep thin delegating wrappers of the same names, so `pipeline/cli.py` and the
existing tests do not churn.

**Add one check that does not exist today: non-finite centroid values.** Python's `json.loads`
accepts `NaN` / `Infinity` by default, and `jsonschema`'s `"type": "number"` accepts a float `NaN`.
A `NaN` centroid coordinate would make every squared distance `NaN`, and every `<` comparison in
`assign_cluster` false — silently pinning assignment to cluster 0 rather than erroring.

Make `check_profiles_invariants` read `cluster_count` via `.get()`: the schema marks it optional
even though `profiles.py` always writes it, so a reader must not `KeyError` on a valid artifact.

### 1b. Cache the encoder

`embed_texts` constructs `SentenceTransformer(...)` on **every call**. That is the ~15s dominating
the smoke test earlier. Irrelevant for one batch corpus run; fatal for §8's *"the router adds
negligible latency compared with an agent session"*.

```python
@functools.lru_cache(maxsize=2)
def _load_encoder(model_id: str)                       # memoised per model id
def embed_one(text: str, config: EmbeddingConfig) -> np.ndarray
```

`embed_one` delegates to `embed_texts([text], ...)` with the progress bar suppressed — a one-row
tqdm bar is noise, not progress. Cost stays a one-time per-process load.

### 1c. Parse a cluster map without file IO

`load_cluster_map(path)` currently reads *and* parses. The runtime needs to read the raw JSON once
(to schema-validate it and to pull the `embedding.input` block), then build the `ClusterMap` from
the dict it already has, without a second read.

```python
def cluster_map_from_dict(data: dict) -> ClusterMap    # new; pure
def load_cluster_map(path: Path) -> ClusterMap         # read + delegate (unchanged signature)
```

**`ClusterMap` itself is deliberately left alone.** Its docstring scopes it to "the subset of
cluster-map.json this module needs" — assignment geometry. Preprocessing rules do not belong on it;
they belong on the runtime context (Phase 2).

---

## Phase 2 — `runtime/context.py`: load and validate

```python
@dataclasses.dataclass(frozen=True)
class RoutingContext:
    cluster_map: ClusterMap
    embedding: EmbeddingConfig        # DERIVED FROM THE ARTIFACT, never from config/embedding.yaml
    profiles_by_model: dict[str, dict]
    candidates: list[ModelConfig]     # non-control, profile-backed, cost-bearing
    lambda_: float
    cluster_map_id: str
    profiles_id: str
    digest: str

def load_routing_context(
    cluster_map_path: Path,
    model_profiles_path: Path,
    lambda_: float,
    candidates: list[ModelConfig] | None = None,     # None -> config/models.yaml
    embedding_config: EmbeddingConfig | None = None,  # None -> config/embedding.yaml, for cross-check only
) -> RoutingContext
```

### The artifact is the authority on embedding, not config

`RoutingContext.embedding` is constructed **from `cluster-map.json`'s `embedding` block** — model
id, dimensions, and the `input.normalisation` / `input.truncation` rule — and that is what gets
handed to `embed_one`. `config/embedding.yaml` is read only to cross-check and hard-fail on
mismatch.

This is the single highest-risk silent failure in the whole design, and the reason the artifact
carries the preprocessing rule at all (see the old plan's Phase 2). Config can be edited ahead of
a rebuild; the artifact records what the corpus was *actually* embedded with. Deriving from the
artifact makes offline/runtime preprocessing drift structurally impossible rather than merely
discouraged. The schema states the same rule normatively: *"Any consumer embedding a new query for
comparison against these centroids MUST use this same model."*

Injectable `candidates` / `embedding_config` exist so Phase 5's tests are pure — no dependence on
whatever happens to be in `config/` on a given machine.

### Hard-fail table

All of these raise at load, before any decision is possible. §8's reproducibility criterion is the
justification: a complete, healthy-looking decision that no artifact can explain is strictly worse
than a loud refusal.

| Check | Why it must fail rather than degrade |
|---|---|
| `lambda_` missing / non-finite / negative | A negative lambda inverts the cost preference silently |
| Either artifact unreadable or schema-invalid | — |
| `schema_version != 1` | Field meanings are not guaranteed across versions |
| Cluster ids not exactly `0..k-1` ascending; `len(clusters) != k` | A truncated file would route on a partial map |
| Any centroid length `!= embedding.dimensions` | — |
| Any non-finite centroid value | See 1a — silently pins assignment to cluster 0 |
| `normalisation != "l2"` / `distance != "euclidean"` / `centroid_dtype != "float64"` | The artifact targets geometry this runtime does not implement |
| **`profiles.cluster_map_id != map.artifact_id`** | Cluster 7 means different things in the two files: the error table is permuted relative to the centroids, and every decision is plausible but wrong |
| `profiles.embedding.model_id / dimensions` disagree with the map | Calibration tasks were clustered with a different model, invalidating every assignment |
| **`map.embedding.model_id != config/embedding.yaml model_id`** | Config drifted ahead of the artifact; names both in the message |
| Duplicate `model_id` in `profiles.models[]` | `models` is a JSON array, so duplicates survive parsing as distinct entries |
| A non-control candidate in `models.yaml` with no profile entry | Error message must name the model and say to run calibration |
| Zero eligible candidates after filtering | Nothing to route to |

### Controls are excluded from candidates

`reference-oracle` and `null-baseline` have profiles (the `reference` control scores ~0.0 error by
construction — it *is* the gold solution) and would win every lambda=0 decision outright. They are
grader-validation instruments, not models. Filter on `ModelConfig.is_control`, matching
`evaluate.py`'s existing `not m.is_control`. The profiles schema says the same thing normatively:
*"A router must never select a control as a real candidate."*

### Reproducibility digest

`sha256` over a canonical JSON encoding of `[lambda, embedding_model_id, cluster_map_id,
profiles_id, sorted((model_id, cost_input, cost_output) for candidates)]`.

Byte-exact embeddings are not reproducible in general (fp16 vs fp32, batch composition, kernel
differences). What *is* reproducible is the decision **given** the vector — so the digest pins
everything else that fed the decision, making "did anything change between run A and run B"
answerable. Plain `hashlib.sha256`; the old plan's hand-rolled FNV-1a existed solely to avoid
importing `node:crypto` into a pure TypeScript module.

---

## Phase 3 — `runtime/decide.py`: the decision

```python
@dataclasses.dataclass(frozen=True)
class RoutingDecision:
    selected: ScoredCandidate
    cluster: ClusterAssignment
    scores: list[ScoredCandidate]        # every candidate, for audit
    excluded: dict[str, str]             # model_id -> reason
    lambda_: float
    digest: str
    cluster_map_id: str
    profiles_id: str
    embedding_model_id: str
    prompt_chars: int
    embed_ms: float | None               # None when the caller supplied the vector
    score_ms: float

def decide_from_vector(vector: np.ndarray, ctx: RoutingContext, embed_ms=None) -> RoutingDecision
def decide(prompt: str, ctx: RoutingContext) -> RoutingDecision
```

The split is what makes the whole thing testable: `decide_from_vector` covers assignment, scoring,
tie-breaks, fallback, and the decision record with a 4-dimensional hand-written vector and no model
download. `decide` is then only `embed_one` plus that call.

`ClusterAssignment` already returns the runner-up cluster and its distance — logging both is what
makes a boundary case visible in a decision row rather than looking like a confident assignment.

**Log a WARNING when `error_source == "model-global"`.** That means the assigned cluster had no
calibration coverage for that model and the decision fell back to its global rate — a real
degradation in decision quality that must be visible in the log, not silent.

`excluded` will be empty in practice, since a missing profile is a startup hard-fail and the schema
requires a `global` block. It is populated defensively (a candidate that `score_candidates` drops,
or a non-finite score) so that if it is ever non-empty the reason is recorded rather than inferred
from a shorter score table.

---

## Phase 4 — `runtime/cli.py`

```
router runtime decide --prompt TEXT --lambda FLOAT [--cluster-map PATH] [--model-profiles PATH] [--json]
router runtime validate [--cluster-map PATH] [--model-profiles PATH]
```

- `--lambda` is **required, with no default**, on `decide`. §6 says explicitly not to pick a
  permanent lambda before evaluation; forcing it per-invocation stops a sweep run from being
  silently mislabeled. Ships documented with §6's sweep: `0 / 0.02 / 0.05 / 0.10 / 0.20 / 0.40`.
- Artifact paths default to `common/artifacts.ARTIFACTS_DIR`.
- `--json` prints the full `RoutingDecision` as one JSON object, so an evaluation harness can
  consume decisions without scraping the human-readable table.
- `validate` runs Phase 2's loading and validation and prints a summary (k, dimensions, artifact
  ids, candidate roster, per-cluster coverage gaps) **without embedding anything** — so an operator
  can check an artifact pair in milliseconds, with no model load. It takes no `--lambda`, since
  validation does not depend on it.

---

## Phase 5 — Tests

One hand-authored fixture pair in `tests/runtime/fixtures/` (`k=3`, `dim=4`), plus a `conftest.py`
helper that loads it as dicts, applies one targeted mutation, writes to `tmp_path`, and calls
`load_routing_context`. Every hard-fail case is then a deep copy plus a single field change: ~15
checks for the cost of one fixture pair, and they cannot drift out of sync with each other.

Fixture shape — centroids are the basis vectors `e1/e2/e3`, so expected assignment is obvious by
inspection:

| Model | price/1M | global | c0 | c1 | c2 |
|---|---|---|---|---|---|
| `cheap-model` | 0.3 | 0.80 | 0.70 | 0.90 | 0.80 |
| `mid-model` | 5.0 | 0.50 | 0.45 | 0.55 | *(absent — exercises global fallback)* |
| `strong-model` | 35.0 | 0.20 | 0.15 | 0.25 | 0.20 |
| `reference-oracle` | 0.0 | 0.00 | 0.00 | — | — |

These numbers are chosen so the three interesting regimes are all reachable on cluster 0, which is
what makes the lambda tests meaningful rather than tautological:

- `lambda=0` → `strong-model` (lowest error, cost ignored)
- `lambda=0.5` → `mid-model` (`0.45 + 0.5x0.135 = 0.518` beats `0.65` and `0.70`)
- `lambda=100` → `cheap-model` (cheapest, `normalised_cost = 0`)

Plus: `reference-oracle` never selected despite a 0.0 error rate; cluster 2 on `mid-model` yields
`error_source == "model-global"`; scaling every price by 1000 changes no decision (min-max
normalisation is scale-invariant); digest is stable across reloads and changes when lambda or an
artifact id changes.

## Verification

1. `uv run pytest` — all pure, no network, no model download, no artifacts on disk.
2. `uv run ruff check src tests`.
3. `router runtime validate` against the fixture pair — confirm it reports coverage gaps and does
   not load the encoder (assert on wall-clock or the absence of the load log line).
4. `router runtime decide --prompt "..." --lambda 0.05 --json` against the fixture pair, which
   *does* load the real 768-dim encoder and must therefore **hard-fail** on the dimension mismatch
   against `dim=4` fixtures — that failure is itself the check that the artifact-vs-model guard
   fires.
5. **Not verifiable in this pass:** an end-to-end decision on real artifacts. `artifacts/` is empty
   and gitignored, so this needs `corpus → embed → cluster → build-artifact → calibrate` first
   (a long CPU job). This must be reported as unverified rather than implied to work.

## Rejected alternatives

- **HTTP embedding server** (the old plan's Phase 4). Justified only by Node's inability to run
  `sentence-transformers`. In-process reuse of `common/embedding.py` is fewer moving parts and
  removes an entire class of drift. Revisit only if the runtime is ever deployed as many
  short-lived processes, where one warm shared server beats N model loads.
- **Re-normalising centroids at load.** They are raw K-means centroids of L2-normalised vectors
  and are *not* themselves unit-norm; re-normalising would move every decision boundary.
- **Recomputing smoothing in the runtime.** Reading the stored `smoothed_error_rate` is precisely
  what makes a decision reproducible from the artifact alone.
- **Defaulting a missing profile to some neutral error rate.** Would put a decision in the log that
  no artifact explains. Excluded at startup instead.
- **Putting the preprocessing rule on `ClusterMap`.** Keeps that type scoped to assignment
  geometry; the rule belongs to the routing context.
- **A `config/routing.yaml` holding lambda.** A persisted default is exactly what §6 warns against
  before evaluation completes.

## Explicitly out of scope

- Session state: frozen per-session route, retry escalation, budget caps, mid-run tier bumps.
  Needs a host agent to be meaningful or testable.
- Any agent integration (Pi extension, HTTP service, SDK adapter).
- Soft weighting over nearby clusters, and everything else in §9.
- Observed session cost as a routing input — §6 confines it to evaluation and savings reporting.
- Re-embedding or re-routing on follow-up messages (§9.6).
