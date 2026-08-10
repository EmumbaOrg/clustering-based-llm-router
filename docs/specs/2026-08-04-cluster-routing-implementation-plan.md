# Plan: cluster-based model routing (spec §3/§6) behind a strategy flag

> Status: approved implementation plan, not yet built. Tracks the request in
> `../routing-poc-background-research.md`. Phase 5 of this plan produces a
> separate, narrower design doc — `2026-08-04-cluster-routing-design.md` —
> that pins the artifact contract for the offline pipeline; this file is the
> plan for the runtime and generalisation work that ships ahead of it.
>
> **Migration note (2026-08-10):** this doc was written against the TypeScript Pi extension in
> `model-router-poc` (`core/*.ts`, `.pi/extensions/routing-layer.ts`). That extension only ever
> implemented *planner* mode — cluster mode (Phases 1, 3, and the TS-specific parts of Phase 4)
> was never built and is now superseded: this repo (`clustering-based-llm-router`) implements the
> routing runtime in Python instead (`src/router/runtime/`), reusing `src/router/common/` directly
> rather than porting logic across languages. **Phase 2 (the artifact schema and its validation
> rules) is unchanged and still the binding contract** — `artifacts-schema/` and `config/` here
> are the same files this plan describes. Read Phases 1/3/4 for the *design reasoning* (the
> scoring formula, the fallback chain, the determinism rules) — not as literal TypeScript
> instructions.

## Context

`Routing PoC and Background Research.md` replaces the router's decision logic. Today an LLM planner call classifies each request on three axes (`editScope`, `needsUnderstanding`, `openEndedness`) and a deterministic table maps that to one of four tiers ([core/planner.ts:152](../../../core/planner.ts#L152)). The spec replaces this with an offline-calibrated, embedding-based router: embed the prompt, assign it to a K-means cluster, and pick the model minimising `predicted_error + λ × normalised_cost`.

The spec explicitly says to reuse the existing extension structure and preserve the escalation and budget mechanisms. So this is a decision-logic swap, not a rewrite — but three things make it more than a drop-in:

1. The spec's candidate-model set is open-ended ("a new model can be onboarded by running only the calibration suite"), while the current ladder is a hardcoded 4-valued `TIERS` enum that `models.ts`, `budget-manager.ts`, `planner.ts` and the extension all key off.
2. Retry escalation's fileless "it's still broken" detection comes from `DemandProfile.reportsFailure` — supplied by the planner call the spec removes ([routing-layer.ts:448](../../../.pi/extensions/routing-layer.ts#L448)).
3. "All routing decisions reproducible from versioned artifacts" (§8) means artifact/config mismatches must fail loudly, not degrade silently.

Intended outcome: both strategies coexist behind `ROUTER_STRATEGY`, defaulting to `planner` so existing installs are untouched; cluster mode makes **no LLM call** on the routing path.

## Scope

Confirmed decisions: generalise to N candidates; drop the planner entirely in cluster mode; pipeline implementation out of scope.

**In scope (implement):** the runtime routing path, the N-tier generalisation, the versioned artifact schema + validation, the feature flag, extended logging, tests.

**In scope (specify only):** a design doc at `docs/superpowers/specs/2026-08-04-cluster-routing-design.md` pinning the artifact contract and the offline pipeline's obligations (corpus, embedding, K-means, calibration), so artifact production can proceed as a separate project.

**Out of scope:** building the corpus, running K-means, the calibration harness, the λ sweep, SWE-bench evaluation. Also deferred: soft cluster weighting and everything in spec §9.

Runtime is developed against hand-authored fixtures (k=3, dim=4). Cluster mode is unusable until real artifacts exist — that is expected and is why the flag defaults to `planner`.

---

## Phase 1 — Generalise the tier ladder to N candidates

Prerequisite for everything else. **Regression guarantee: behaviour is byte-identical at N=4 with an unchanged `.env`.**

### `core/models.ts` — rewritten

Delete the `TIERS` const object and the `CAPABILITIES` record. New surface:

```ts
export type Tier = number                      // 1-based cost rank; 1 = cheapest
export type EnvSource = Readonly<Record<string, string | undefined>>

export interface TierLadder {
  readonly tiers: readonly Tier[]              // [1..N]
  readonly minTier: Tier                       // 1
  readonly maxTier: Tier                       // N
  readonly config: Readonly<Record<Tier, TierModel>>
  readonly models: Readonly<Record<Tier, string>>
  readonly ordered: readonly TierModel[]       // cheapest first; index 0 = tier 1
}

export function buildTierLadder(env: EnvSource): TierLadder   // pure, throws naming the exact var
export function discoverTierNumbers(env: EnvSource): Tier[]
export function tierModel(tier: Tier, ladder?: TierLadder): TierModel   // boundary guard, throws
export const MAX_LADDER_SIZE = 32
export const LEGACY_TIER_CAPABILITIES: readonly string[]      // the 4 existing blurbs, verbatim

export const LADDER = buildTierLadder(process.env)            // module singleton, throws at startup
export const TIER_CONFIG = LADDER.config
export const TIER_MODELS = LADDER.models
export const ALL_TIERS = LADDER.tiers
export const MIN_TIER = LADDER.minTier
export const MAX_TIER = LADDER.maxTier
```

`TierModel` gains `tier: Tier` and `capability` becomes optional. `requireEnv`/`requireEnvNumber` take `env` as a first parameter but keep their message strings **byte-identical** — that string is the documented startup-error contract.

**Discovery: key-scan, not probe-until-missing.** Collect every `n` from keys matching `/^TIER_(\d+)_(MODEL|NAME|PROVIDER|COST_INPUT|COST_OUTPUT|CONTEXT_WINDOW|MAX_TOKENS|CAPABILITY)$/` with a non-empty value; `N = max(n)`; every `1..N` must have `TIER_n_MODEL`. A probe would *silently ignore* a `TIER_5_*` block sitting below a commented-out `TIER_4_MODEL` — the likeliest user mistake. Key-scan turns it into a named error. Reject non-canonical indices (`TIER_01_`), enforce `2 ≤ N ≤ 32`, and error on a duplicate `(provider, id)` pair (see Pre-existing issues #5).

**Cost rank: validate, don't sort.** Throw when `cost.input + cost.output` decreases across tiers, naming both tiers and both prices. Do not renumber internally: the tier number is a *user-facing identifier* — it keys `MAX_BUDGET_TIER_n`, the `usage` keys in `budget-state.json`, and the `tier` field in `routing.log.json`. Silent renumbering would make `MAX_BUDGET_TIER_3` cap a different model than the `.env` block the user edited. Ties allowed, so an all-local all-$0 ladder passes. Verified: the shipped `.env.example` ladder (0.00045 / 0.00145 / 0.00525 / 0.035) is already non-decreasing, as is its mixed local/hosted example.

**Capabilities** (planner mode only — cluster mode never reads them): `TIER_n_CAPABILITY` if set, else `LEGACY_TIER_CAPABILITIES[n-1]` **only when N===4**, else `undefined`. This is what makes N=4 need no `.env` changes.

### `core/planner.ts`

- Delete the eager `const TIER_MENU` at [line 57](../../../core/planner.ts#L57) — it snapshots `TIER_CONFIG` at import, which both defeats ladder injection in tests and would throw at module load for cluster-mode users missing capability text. Replace with a memoised `defaultTierMenu()` plus an exported `assertPlannerCapabilities(ladder?)` that the entry point calls **only in planner mode**, inside the existing `try`/`setupError` block. `buildTierMenu` takes a `TierLadder` and iterates `ladder.ordered`.
- Split `mapProfileToTier` into an N-independent 4-level demand scale and an N-dependent projection:

```ts
export type DemandLevel = 0 | 1 | 2 | 3
export function demandLevel(profile: DemandProfile): DemandLevel
export function tierForDemandLevel(level: DemandLevel, ladder = LADDER): Tier   // minTier + round(level*span/3)
export function mapProfileToTier(profile: DemandProfile, ladder = LADDER): Tier
```

  Proportional projection is exact identity at N=4 (1/2/3/4), monotone for all N≥2, and always uses the full ladder — an absolute clamp would leave tiers 5+ unreachable at N=7, and a top-anchored mapping is non-monotone at N=2.
- `requestTypeForTier` derives from the same projection instead of a 4-entry table.
- `applyContextGuard` takes the ladder and loops to `ladder.maxTier`. **It stays a no-op** — `buildTierModel` never sets `contextGuard`, so the function returns by identity today ([tests/planner.test.ts:118-119](../../../tests/planner.test.ts#L118-L119) documents this), which also makes the context branch at [routing-layer.ts:565-570](../../../.pi/extensions/routing-layer.ts#L565-L570) dead. Wiring it to `contextWindow` would activate a dormant escalation path — a real behaviour change that deserves its own commit. Keep the seam, don't delete it.

### `core/budget-manager.ts`

Delete all three 4-entry tables. `MAX_BUDGET_TIER_${tier}` inline; `ALL_TIERS` imported from models; `nextAvailableTier` gains an injectable `tiers` parameter for testing. Default budgets become a clamped curve so tiers 1-4 keep their exact existing values and 5+ inherit 60,000:

```ts
const DEFAULT_BUDGET_CURVE = [10_000, 20_000, 35_000, 60_000] as const
const defaultBudget = (tier: Tier) => DEFAULT_BUDGET_CURVE[Math.min(tier, 4) - 1]
```

Also fix `freshState()` at [line 52](../../../core/budget-manager.ts#L52), which hardcodes `{'1':0,'2':0,'3':0,'4':0}` — use `{}` (every read already does `?? 0`).

### `core/escalation-manager.ts`, `core/logger.ts`

No changes. `retryTierFloor` already takes its ceiling as a parameter; only its test's arguments change.

### `.pi/extensions/routing-layer.ts`

Mechanical: `TIERS.FRONTIER_CODE` → `MAX_TIER` (lines 460, 479, 485, 491-494, 588), `TIERS.LOCAL_SMALL_GENERAL` → `MIN_TIER` (422, 503, 598), `TIER_CONFIG[tier]` → `tierModel(tier)` at [line 366](../../../.pi/extensions/routing-layer.ts#L366), `LADDER.ordered` at 390. Also fix the message at [line 493](../../../.pi/extensions/routing-layer.ts#L493), which tells the user to "swap the Tier-4 model in `core/models.ts`" — models live in `.env`.

### Testing

Add `"tests/**/*.ts"` to `tsconfig.check.json`'s include. Tests are currently invisible to `npm run typecheck`, so deleting `TIERS` would surface only as a runtime `TypeError`.

New `tests/models.test.ts` is entirely pure — it calls `buildTierLadder(fakeEnv(n))` with synthetic env objects, so it needs no `.env` and no module-reload tricks. This is why the pure-builder extraction matters: `vi.resetModules()` + dynamic import would force `process.env` mutation (breaking parallelism) and desync other modules' singletons against the reloaded `models.ts`. Cover N=2/4/7, gaps, orphan fields, out-of-range, non-canonical index, missing/non-numeric fields, cost-order violation, duplicate models, and capability resolution at N=4 vs N=5.

Existing tests: `planner.test.ts` (14, 76-96, 110-113, 121-137, 156, 197-203), `escalation-manager.test.ts` (9, 64), `budget-manager.test.ts` (11 + ~24 usages) swap `TIERS.*` for numeric literals or `MIN_TIER`/`MAX_TIER`. Keep the N=4 cases as the identity regression. `routing-layer.test.ts` is unaffected.

---

## Phase 2 — Artifact schema and validation

Two separately-versioned JSON files. The split is what satisfies "adding a model does not require rebuilding K-means": onboarding appends one entry to `models[]` and bumps `artifact_id`; `cluster-map.json` is untouched.

New pure module `core/cluster-artifacts.ts` — types, `CLUSTER_MAP_SCHEMA_VERSION`, `parseClusterMap(raw, sourceLabel)`, `parseModelProfiles(...)`, `validateArtifactSet({map, profiles, configuredEmbeddingModel, candidateModelIds})`. All take already-parsed JSON and return `{ok:true, value, warnings} | {ok:false, errors}`; the entry point does the `readFileSync`.

`cluster-map.json` carries `schema_version`, `artifact_id`, `embedding: {model_id, dimensions, normalisation:'l2', distance:'euclidean', centroid_dtype:'float64', input: {normalisation, truncation}}`, `kmeans: {k, seed, n_init, corpus_size, corpus_digest}`, and `clusters: [{id, size, label?, centroid: number[]}]`.

`model-profiles.json` carries `schema_version`, `artifact_id`, `cluster_map_id`, `smoothing: {method, prior_weight, formula}`, and `models: [{model_id, calibration_run_id, global: {...}, clusters: {"<id>": {number_of_tasks, number_succeeded, number_failed, raw_error_rate, smoothed_error_rate, mean_session_cost, ...}}}]` — the full §5.2 field list.

Contract points the offline pipeline must honour, all hard-validated:

- `clusters` is an **array** whose index equals its `id`, contiguous `0..k-1` ascending. Kills iteration-order questions and catches a truncated file. Same for `models[]` (identity is `model_id`, duplicates rejected) — `JSON.parse` silently keeps the last of duplicate object keys.
- Centroids are the raw K-means centroids of L2-normalised vectors; they are **not** themselves unit-norm and the runtime must never re-normalise them. `centroid_dtype: 'float64'` is mandatory: float32 centroids would disagree with the runtime near boundaries.
- **`embedding.input` (normalisation + truncation) lives in the artifact, not in env.** Offline/runtime preprocessing drift shifts every vector and is the highest-risk silent failure in the whole design; reading the rule from the artifact makes drift impossible.

**Runtime uses `smoothed_error_rate`, always.** At K=24 with 800 calibration prompts, per-cluster `n ≈ 20–33`, so `raw_error_rate` is quantised to `1/n` and a lucky 20/20 hands a cheap model a `0.0` that pins every request in that cluster to it regardless of λ. `raw_error_rate` and the counts are logged for audit only. The runtime **never recomputes smoothing** — reading the stored value is what makes the decision reproducible from the artifact alone.

Fallback chain: cluster entry → model `global` (per-request degrade, logged as `errorSource: 'model-global'`, notified once) → **exclude the candidate** (`no-profile`), never default. Inventing a quality number would put a decision in the log that no artifact explains. Level 3 is normally unreachable because a candidate with no profile is a startup hard-fail.

### Startup validation — hard-fail vs degrade

Hard-fail (appended to `setupError`, surfaced at `session_start` because Pi swallows factory throws): unknown `ROUTER_STRATEGY`; unreadable/invalid artifact; schema-version mismatch; **`map.embedding.model_id ≠ ROUTER_EMBEDDING_MODEL`**; dimension mismatch; cluster ids not exactly `0..k-1`; non-finite centroid values; `normalisation`/`distance`/`centroid_dtype` not the one geometry implemented; **`profiles.cluster_map_id ≠ map.artifact_id`**; duplicate `model_id`; **a candidate model missing from `models[]`**; error rates outside `[0,1]` or counts that don't sum; `ROUTER_LAMBDA` missing/non-finite/negative; embedding provider unresolvable or `api === 'anthropic-messages'` (no embeddings endpoint).

Those three bolded checks are the ones that would otherwise produce a complete, healthy-looking log of decisions no artifact can explain — the exact failure §8's reproducibility criterion exists to prevent.

Per-request degrade (notify + **keep the current model**, mirroring the existing planner-failure convention at [routing-layer.ts:436-440](../../../.pi/extensions/routing-layer.ts#L436-L440)): embedding HTTP error, timeout, unparseable body, wrong dimension, zero-norm/non-finite vector, cluster coverage hole with a usable `global`, prompt below `MIN_ROUTABLE_INPUT_CHARS`.

Artifacts live in `router-artifacts/` and are referenced by absolute path via env. **Not `artifacts/`** — `.gitignore:88` ignores that, so the files would silently never commit. Don't ship them in the npm tarball (k=24×768 float64 ≈ 400 KB); `scripts/postinstall.cjs` adds the two path vars to `.env` commented out.

---

## Phase 3 — The scoring router

New pure module `core/cluster-router.ts`:

```ts
export interface Candidate { tier: Tier; modelId: string; costPer1kInput: number; costPer1kOutput: number }
export type ExclusionReason = 'below-retry-floor' | 'budget-exhausted' | 'no-profile' | 'invalid-score'

export function assignCluster(vector: number[], map: ClusterMapArtifact): ClusterAssignment
export function lookupPredictedError(profiles, modelId, clusterId): { value: number; source: ErrorSource } | null
export function staticPricePer1M(c: Candidate): number      // (input + output) * 1000
export function normaliseCosts(prices: number[]): number[]
export function selectModel(input: SelectionInput): SelectionResult
export function formatClusterDecision(assignment, selected, lambda, label?): string
```

`ClusterAssignment` returns `{clusterId, distance, runnerUpClusterId, runnerUpDistance}` — the runner-up makes boundary cases visible in the log.

**Divide-by-zero:** when `max - min <= 0` (one eligible model, or all at the same price — the all-local-$0 case), every `normalised_cost` is `0`. Any constant works since it shifts all scores equally, but `0` keeps `routing_score === predicted_error` in the log and preserves the "λ=0 ⇒ accuracy-only" invariant exactly. Clamp the general case to `[0,1]`.

**Cost units:** min-max normalisation is invariant under positive scaling, so the per-1k → per-1M conversion cannot change any decision. Convert anyway and log the per-1M figure — it is the unit the spec and every provider price page use, so a logged score row can be hand-checked. Note for the eval write-up: the spec's unweighted `input + output` penalises expensive-output models as if output volume equalled input, which is backwards for agent sessions; weighting would need expected token counts, which §6 excludes.

**Eligibility is pre-filtered, before scoring** — in cluster mode only. §6 says to normalise over the "currently eligible" set, and today's post-hoc substitution ([routing-layer.ts:501-518](../../../.pi/extensions/routing-layer.ts#L501-L518)) both normalises over a set containing an unchoosable model and substitutes by *cost adjacency* (`nextAvailableTier`), ignoring `predicted_error` entirely — it can land on a model that is terrible for the assigned cluster. Pre-filtering makes the fallback "second-best by routing score", which is what §7 wants to measure. Planner mode keeps post-hoc substitution unchanged.

Preserved verbatim: the all-exhausted notify string and keep-current-model behaviour; the escalation-ceiling terminal notice and `terminalNotified` set. `turn_end`'s mid-run bumps stay tier-based and unchanged — they fire on *capacity* signals (context overflow, `stopReason === 'length'`, looping) about which cluster profiles say nothing, so `tier + 1` is the right response and re-scoring would be a category error.

**Determinism:** total order is `routing_score` asc → `tier` asc → `modelId` asc. Single-pass argmin over the tier-ascending array with strict `<`, so the cheapest of a tie wins by construction; no `Array.sort` on the hot path. Fixed-index `for` loops for the distance accumulation (no `reduce`, no `Math.hypot`, no `Math.sqrt` — squared distance is monotone). Centroid ties resolve to the lowest id via the ascending scan. Non-finite scores are excluded as `invalid-score` rather than silently losing every `<` comparison.

Byte-exact embeddings are *not* achievable over HTTP (fp16 vs fp32, batch size, GPU kernels). What is reproducible is the decision *given* the vector — hence logging `clusterId`, both distances, and a `routingConfigDigest`: an FNV-1a hash (pure, no `node:crypto` in `core/`) over `[strategy, λ, embeddingModelId, mapArtifactId, profilesArtifactId, candidates(tier, modelId, prices)]`. `.env` is not a versioned artifact, so without this digest "did anything change between run A and run B" is unanswerable.

---

## Phase 4 — Wire into the extension

New pure module `core/embedding.ts`: `prepareEmbeddingInput(text, rule)` (CRLF→LF, trim, head-truncate — **no case folding, no whitespace collapsing**, since indentation is signal for a code embedding model), `parseEmbeddingResponse`, `l2Normalize`, `toRoutingVector(data, expectedDimensions)`, `isRoutableInput`. All fetch stays in the entry point as `buildEmbeddingRequest` + `embedQuery`, mirroring `buildPlannerRequest`/`planTier` exactly — Pi loads every `.ts` in `.pi/extensions/` as an extension, so there is no second entry-point file.

Wire format: `POST {baseUrl}/embeddings`, body `{model, input: [text], encoding_format: 'float'}`. `input` as a one-element **array** (a bare string is not accepted everywhere) and `encoding_format` explicit so a base64 response is rejected rather than parsed into garbage. This one shape covers Ollama, TEI, vLLM, LM Studio, OpenAI, and Jina's own API. The embedding provider resolves through the existing `resolveProviderConnection` rather than a parallel mechanism (widen its error string to mention the embedding provider).

**What gets embedded:** `before_agent_start`'s `event.prompt`, on the first agent run after each `session_start`, then **frozen for the session** in module state keyed on `ctx.sessionManager.getSessionId()`. Follow-up messages issue no embedding at all — zero added latency and cost. Not the `input` event: it fires for inputs that never reach the agent (slash commands, `!bash`) and its text is pre-expansion, so it would differ from what the agent receives. Post-expansion text can inject file contents absent from the clustering corpus, which is exactly why the artifact declares a head truncation (~8000 chars, inside jina-v2-base-code's 8192 window): the task statement leads, expanded dumps trail.

Freeze the *cluster assignment*, not the selection — re-score (no re-embed, so it is pure and instant) when eligibility changes: a retry floor rising above the frozen tier, or the frozen tier crossing its 85% cap. Per-message re-routing is explicitly a §9.6 future enhancement.

**`reportsFailure` without the planner.** Per the decision to drop the planner, cluster mode does **not** get a keyword/regex substitute — `core/escalation-manager.ts:53-55` rejected that approach and this plan honours it. `findRetryEntry`'s signature is unchanged; in cluster mode its second argument comes from new module state instead:

```ts
lastRunFailed = turnState.lastBashFailed && turnState.codeChanged   // set in recordRunOutcome()
```

This is factual state, not a heuristic: the immediately preceding run edited code *and* left a failing command. It is strictly more precise than the LLM flag — it never fires on an unrelated fileless follow-up like "now add tests". It does not cover a fileless failure report after a run that only edited and never ran a command; that recall loss is the accepted cost of dropping the planner.

Edits to `.pi/extensions/routing-layer.ts`, in handler order:

- **Module scope:** delete the three unconditional `PLANNER_*` throws ([lines 100-113](../../../.pi/extensions/routing-layer.ts#L100-L113)) — they run at module load, outside `setupError`, and would force cluster-mode users to configure a planner they never call. Replace with `readRouterConfig()` returning a discriminated `{strategy:'planner', planner:{...}} | {strategy:'cluster', cluster:{...}}`, called inside the factory's existing `try`.
- **Factory:** load + validate artifacts, resolve the embedding connection, compute `routingConfigDigest`, `assertPlannerCapabilities()` in planner mode — all in the same `try`.
- **`session_start`:** reset the frozen route and per-session flags. Optional warm-up probe (default on) with a fixed short string — validates dimensionality early with a precise message and absorbs a cold local model's first-load latency off the user's first prompt. Its vector is discarded.
- **`before_agent_start`:** new first line `if (setupError) { notify; return }` — see Pre-existing issues #1. Keep `extractFileReferences` (still needed for retry attribution) but **skip `gatherReferencedCode` entirely in cluster mode**, saving up to 96 KB of synchronous reads per request. Then branch on strategy; cluster path embeds-or-reuses, assigns, builds the `isEligible` predicate, and calls `selectModel`.
- **`turn_end` / `agent_end`:** add `agentTurns`/`toolCalls` counters and the new log fields; `agent_end` must **not** clear the frozen route.
- **New:** `pi.on('model_select')` to stamp `manualModelOverride` when a user hand-picks a model — without it the §7 accuracy–cost curve silently mixes in models the router never chose. Optionally `pi.registerCommand('router')` to print the frozen route and score table, making the mechanism inspectable during evaluation instead of log-only.

### Config surface

| Var | Mode | Required | Notes |
|---|---|---|---|
| `ROUTER_STRATEGY` | both | no — defaults `planner` | The one sanctioned in-code default, so existing installs are untouched. Unknown values hard-fail. |
| `ROUTER_LAMBDA` | cluster | **yes, no default** | §6 says not to pick a permanent λ before evaluation, so requiring it forces an explicit choice. Ships commented in `.env.example` with the sweep `0 / 0.02 / 0.05 / 0.10 / 0.20 / 0.40`. Validate finite and ≥ 0 — `requireEnvNumber` accepts `Infinity` and negatives today. Echo the parsed value in the status line so a sweep run can't be mislabeled. |
| `ROUTER_EMBEDDING_PROVIDER` / `_MODEL` | cluster | yes | Provider is a name resolved via `resolveProviderConnection`; model is cross-checked against the artifact. |
| `ROUTER_CLUSTER_MAP_PATH` / `ROUTER_MODEL_PROFILES_PATH` | cluster | yes | |
| `ROUTER_EMBEDDING_TIMEOUT_MS` | cluster | no — 30000 | Matches `planTier`; configurable for cold local models. |
| `ROUTER_EMBEDDING_WARMUP` | cluster | no — `true` | |
| `ROUTER_LOG_SCORE_TABLE` | cluster | no — `true` | |
| `ROUTER_LOG_EMBEDDING` | cluster | no — `false` | 768 floats/row. |
| `TIER_n_CAPABILITY` | planner | only when N ≠ 4 | New, optional. |
| `PLANNER_*` | planner | yes, **in planner mode only** | Currently unconditional. |

### Logging (`core/logger.ts`)

Add a required `routerStrategy` plus optional `sessionId`, `agentTurns`, `toolCalls`, `manualModelOverride`, `routingConfigDigest`, and a `cluster?: {...}` sub-object: artifact ids, embedding model/dimensions/endpoint host, `clusterId`/`clusterLabel`/`clusterDistance`/`runnerUp*`, `lambda`, the selected model's `predictedError`/`errorSource`/`normalisedCost`/`staticPricePer1M`/`routingScore`, `eligibleModels`, `excluded[]`, the full per-candidate `scores[]`, `embeddingLatencyMs`, `scoringLatencyMs`, `reusedFrozenDecision`, `rerouteReason`, `skippedReason`. Everything optional so planner-mode rows stay schema-compatible with today's.

Log the score table by default: without it a row records *what* was chosen but not *why*, and the λ sweep can't be analysed without re-running every session. `excluded[]` is what tells you whether a savings figure came from routing or from budget exhaustion.

### Testing

Almost all of it is pure and needs no network. `tests/embedding.test.ts` (preprocessing boundaries, base64 rejection, zero-norm, dimension guard); `tests/cluster-artifacts.test.ts` — **one** valid fixture pair in `tests/fixtures/` (k=3, dim=4), each invalid case a `structuredClone` + one targeted mutation, so ~20 checks cost one fixture pair and cannot drift out of sync; `tests/cluster-router.test.ts` (nearest-centroid with an exact tie, `normaliseCosts` zero-range, λ=0 ⇒ argmin error, large λ ⇒ cheapest, score tie ⇒ lower tier, `NaN` excluded, empty eligible set, and the property that scaling all prices by 1000 changes nothing).

For fetch: export `buildEmbeddingRequest` and assert URL/headers/body with zero mocking — the same precedent as `envPrefix`/`resolveProviderConnection` being exported for tests. `embedQuery` itself uses `vi.stubGlobal('fetch', ...)` (vitest built-in, no new dependency) plus `vi.useFakeTimers()` for the abort path; every failure case must return `{ok:false}` and never throw, so the keep-current-model path holds. Stub new cluster env vars per test rather than relying on a local `.env`.

---

## Phase 5 — Docs and the pipeline spec

- `docs/superpowers/specs/2026-08-04-cluster-routing-design.md`, following the existing spec conventions (`## Context` → mechanism sections → `## Verification` → `## Explicitly out of scope`, recording rejected alternatives with reasons). This is the deliverable that lets the artifact pipeline be built separately: the frozen embedding model id, the preprocessing rule, the corpus/dedup requirements from §4.1, the K-means config and K-selection criteria from §4.2, the calibration protocol and stored fields from §5, and the exact JSON schemas.
- `.env.example`: new `Strategy`/`Cluster routing` banner sections in the existing style; a commented `TIER_5_*` block plus `TIER_n_CAPABILITY`; contiguity and cheapest-first rules in the Tiers preamble.
- `README.md`: "How it works" gains the strategy split; fix the budget-defaults table at [README.md:75-78](../../../README.md#L75-L78), which says 20k/30k/40k/80k while [budget-manager.ts:13-16](../../../core/budget-manager.ts#L13-L16) says 10k/20k/35k/60k.
- `HOW_TO_RUN.md`: a section on running an embedding server and pointing at artifacts.
- `core/README.md`: currently lists `keywords.ts`, `inspector.ts`, `router.ts` — none of which exist — and omits `planner.ts` and `budget-manager.ts`. Rewrite for the real module set.

---

## Pre-existing issues found

Not caused by this work; each needs a call.

1. **`setupError` is never checked in `before_agent_start`.** With a bad `.env` the factory's `catch` leaves `providerConnections` empty, `session_start` returns early, but `before_agent_start` still calls `planTier(undefined)` and throws on every prompt. Cluster mode makes artifact-load failure the common path, so **fix this in Phase 4** (one guard line).
2. **`retryEntry.attempts` double-increments.** [routing-layer.ts:455](../../../.pi/extensions/routing-layer.ts#L455) adds 1, then `recordRunOutcome()` at [line 180](../../../.pi/extensions/routing-layer.ts#L180) adds 1 again when `lastBashFailed` — so `RETRY_ESCALATION_THRESHOLD = 2` is reachable after a *single* failed run. Cluster mode's retry-floor filter depends on `attempts` being meaningful. **Recommend fixing in its own commit before Phase 3**, since it is a behaviour change to a mechanism the spec says to preserve.
3. **`getModelPricing` returns `{0, 0}` for an unknown model id** ([models.ts:87](../../../core/models.ts#L87)). Harmless today; in cluster mode a drifted id would become *free* and win on cost unconditionally. Mitigated by computing `static_price` from the `Candidate` record only, but the function should signal a miss so the logger can flag it.
4. **`logger.ts` rewrites the entire array on every append** and has no `try/catch`, so bytes written grow quadratically and a full disk throws out of `agent_end`. Score tables make this worse. Recommend JSONL + `appendFileSync` before any evaluation run — **deferred**, not in this plan.
5. **Duplicate model ids across tiers.** Impossible to hit meaningfully at N=4 but likely at larger N (same weights via two providers, or one model at two reasoning efforts). Would pass duplicate ids into one `registerProvider` call and make `getModelPricing` return the wrong tier's price. Phase 1 hard-errors on a duplicate `(provider, id)`; allowing duplicates later needs a tier-keyed `priceForTier`.
6. **`MAX_BUDGET_TIER_n=0` means "unlimited", not "never use"** ([budget-manager.ts:94](../../../core/budget-manager.ts#L94)) — the opposite of what an operator disabling a tier would expect. Document it.
7. **`turnState` is a single module-level slot.** A queued/steering message arriving mid-stream would clobber the in-flight run's state and lose its log row. Verify against the Pi runtime; the frozen session route reduces but does not remove the exposure.

---

## Verification

1. `npm run typecheck` (now including `tests/`), `npm test`, `npm run lint`.
2. **N=4 regression gate.** With the unmodified 4-tier `.env` and `ROUTER_STRATEGY` unset, diff `buildPlannerPrompt('x', {referencedCode:''})` and the `mapProfileToTier` × 12-profile matrix against `git stash`-ed HEAD. Byte-identical, or Phase 1 isn't done. Also confirm `getTierBudget(1..4)` returns 10k/20k/35k/60k.
3. **Planner mode end-to-end unchanged.** `npm run router`, submit a read-only question and a single-file edit, confirm T1/T2 selection, the status line, and a `routing.log.json` entry matching today's schema.
4. **N≠4 ladder.** Add a `TIER_5_*` block plus `TIER_1..5_CAPABILITY` to `.env`, restart, confirm the planner menu shows 5 rungs and escalation ceilings at T5. Then remove `TIER_4_MODEL` and confirm startup fails naming `TIER_4_MODEL`.
5. **Cluster mode with fixtures.** Point the artifact paths at the k=3/dim=4 test fixtures, run a local embedding server (`ollama pull` a code embedding model, or TEI), set `ROUTER_LAMBDA=0.05`, and submit a prompt. Confirm: no planner HTTP call is made (check with a bogus `PLANNER_MODEL` — it must not be read), the status line shows the cluster and score, and the log row carries `clusterId`, the score table, and `routingConfigDigest`. Submit a second prompt and confirm `reusedFrozenDecision: true` with `embeddingLatencyMs: null`.
6. **λ sweep sanity.** Same prompt at `ROUTER_LAMBDA=0` and `=0.40`; the former must pick the lowest-error model, the latter the cheapest. Confirm the logged `scores[]` explains both.
7. **Failure paths.** Point `ROUTER_EMBEDDING_MODEL` at a name the artifact doesn't declare → startup error naming both. Remove a candidate's profile entry → startup error naming the model and the calibration step. Stop the embedding server mid-session → per-request notify and the run proceeds on the current model.
8. **Budget interaction.** Set `MAX_BUDGET_TIER_n` low enough to exhaust the selected tier, confirm the log's `excluded[]` shows `budget-exhausted` and selection moved to the second-best *by score*, not by cost adjacency.
