# Router UI: interactive routing decisions

A small local web page over the runtime's `decide` step. Type a prompt, set the cost weight λ, and
see which cluster the prompt falls into and which model the router would pick. Every candidate's
score is shown next to the winner's.

It's for exploring and demoing routing behaviour. It does **not** call any LLM, so using it costs
nothing.

## Starting it

```bash
uv run router runtime ui                 # http://127.0.0.1:8765
uv run router runtime ui --port 9000     # if 8765 is taken
uv run router runtime ui --model gpt-5.6-luna --model gpt-6.1-sol   # a custom candidate set
```

Wait for `router UI ready at …`. The first start takes about 15 s because it loads the embedding
model. After that, each new prompt takes about 0.1 s. Changing λ on a prompt you've already
routed is near-instant, because its embedding is cached.

The server listens on `127.0.0.1` only. Stop it with Ctrl+C. If the port is busy, the command
says so and exits; pass a different `--port`.

## Candidates

By default the UI offers only the four GPT models. They are the ones calibrated in full on the
same 240-task set, so their numbers compare directly:

| Model | Tier | Global pass rate | Price per 1M tokens (in / out) |
|---|---|---|---|
| gpt-6.1-sol | high (flagship) | 70.5% | $2.00 / $10.00 |
| gpt-5.4-mini | mid | 59.7% | $0.75 / $4.50 |
| gpt-5.6-luna | low | 71.5% | $0.20 / $1.20 |
| gpt-5-nano | lowest | 43.1% | $0.05 / $0.40 (retires Dec 11 2026) |

The **tier** is a price band, not a quality rank. It comes from the same min-max cost scale the router
uses: normalised cost ≥ 0.5 is *high*, ≥ 0.2 *mid*, above 0 *low*, and the cheapest candidate is
*lowest*. With a custom `--model` set the tiers are recomputed for that set.

Cost is scaled across whichever candidates are offered. With these four, nano counts as 0 and sol
as 1. So scores in the UI won't match `router runtime decide`, which uses the whole roster in
`config/models.yaml`.

## Reading the screen

**score = predicted error + λ × normalised cost.** The lowest score wins.

- **λ = 0** picks the most accurate model for the cluster, ignoring price.
- **Higher λ** favours cheaper models. The buttons are the calibration sweep values
  (0, 0.02, 0.05, 0.1, 0.2, 0.4). The slider goes to 1, and the number box accepts any value
  of 0 or more.

The left column holds the inputs: an **example prompt** menu (the test prompts below), the prompt
box, the λ controls, and a list of **recent prompts** you can click to re-route.

The middle column shows the results. Each model keeps the same colour throughout.

The right-hand **Candidate models** panel lists every candidate, dearest first, with its tier badge,
price, overall pass rate, context window and pinned reasoning level. After you route a prompt it also
shows each model's pass count in that prompt's cluster, and marks the selected one. Click a model to
highlight its line in the chart; click again to clear it. On narrower screens the panel moves under
the inputs.

1. **Summary cards.**
   - *Selected model*: the winner, its score and predicted error, and how it did on this
     cluster's calibration tasks (for example `8/10 passed`).
   - *Prompt lands in*: the nearest cluster, its size and distance, and the runner-up cluster
     with the gap between the two. A gap under 0.05 is flagged **borderline**: rewording the
     prompt could move it to the other cluster.
   - *Cluster makeup*: which datasets the cluster's corpus tasks come from.
2. **Score vs λ chart.** One line per model, with each model's score plotted across λ from 0 to 1.
   - The shaded bands show which model wins over each range of λ, and the vertical rule marks
     the current λ.
   - Hover anywhere on the chart to see every model's score at that λ. Click to set λ to that
     point.
   - Switches that happen past λ = 1, such as nano at about 2–7, are listed under the chart.
3. **All candidates at this λ.** For each model: score, predicted error, a bar with the cluster
   calibration pass count, price, and scaled cost. The selected model is marked.

### Shareable links

Query parameters prefill the page and route straight away:

- `http://127.0.0.1:8765/?example=2&lambda=0.1` loads test prompt 2 below with λ = 0.1.
- `http://127.0.0.1:8765/?prompt=Fix%20the%20flaky%20test&lambda=0` routes any prompt you give it.

## Test prompts

Each prompt below was routed against the current artifacts (profiles as of 2026-11-03). The
cluster and winner ranges are what the UI showed. If the artifacts are rebuilt or recalibrated,
the results will change.

### gpt-6.1-sol wins at low λ

Move λ from 0 to 0.4 and the winner changes from **sol → luna**. Above about 2 it changes to
nano.

**1. A bigcodebench-style function.** Cluster 7.

```
Write a function task_func(csv_path) that reads a CSV with pandas, drops rows with missing values, and returns a matplotlib Axes with a histogram of the 'age' column.
```

sol wins for λ < 0.05, luna up to 2.77, then nano. Calibration results in this cluster: sol 7/9,
luna 7/10.

**2. A Go concurrency fix.** Cluster 10. This has the widest sol range of the ten.

```
Fix the Go HTTP handler so that a context cancellation from the client stops the downstream gRPC call instead of leaking the goroutine. Add a test using httptest.
```

sol wins for λ < 0.14, luna up to 1.96, then nano. Calibration results: sol 8/10, luna 6/10,
mini 4/9, nano 5/10.

**3. A React refactor.** Cluster 18. Borderline: the gap to the runner-up cluster is 0.042.

```
In our TypeScript React app, the useEffect in UserList re-fetches on every render because the options object is recreated. Refactor so it only refetches when filters change.
```

sol wins for λ < 0.07, then luna. Calibration results: sol 9/10, luna 8/10.

### gpt-5.6-luna wins at every normal λ

**4. A pandas question.** Cluster 6, mostly DS-1000 tasks.

```
I have a pandas DataFrame with columns user_id, date and amount. How do I compute a 7-day rolling sum of amount per user, keeping the original row order?
```

luna wins by a wide margin. Calibration results: luna 7/9, sol 4/9, mini 3/9, nano 0/9.

**5. A numpy question.** Cluster 15.

```
Using numpy, given a 2D array a of shape (n, m), return the indices of the k largest values in each row, sorted descending.
```

luna and sol both passed 9/10 in this cluster, so luna wins on price.

**6. A bug report in SWE-smith style.** Cluster 11, a hard cluster.

```
Bug: `Config.load()` silently ignores the `env_prefix` argument when the YAML file also defines `prefix`. Expected env vars like APP_DEBUG to override the file. Steps to reproduce: create config.yaml with prefix: X, set APP_DEBUG=1, call Config.load('config.yaml', env_prefix='APP_'). Actual: debug stays False.
```

luna, sol and mini all passed 4/8. With accuracy tied, the cheapest of the three wins.

**7. A traceback fix.** Cluster 1.

```
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int' raised from Parser._advance() when the input string ends with a trailing comma. Please fix the tokenizer so trailing commas are accepted.
```

**8. A Rust panic.** Also cluster 1.

```
The Rust crate panics with 'index out of bounds' in Lexer::peek when parsing an empty file. Return Ok(empty token stream) instead and add a unit test.
```

This lands in the same cluster as #7. The clusters group by type of task (fix a parser crash) more
than by language.

**9. A SQLAlchemy regression.** Cluster 19.

```
SQLAlchemy: session.merge() on a detached object with a lazy-loaded relationship raises DetachedInstanceError in 2.0 but worked in 1.4. Make merge load the relationship eagerly.
```

**10. A Java utility.** Cluster 12. Borderline: the gap to the runner-up cluster is 0.030.

```
Write a Java method that parses ISO-8601 durations like PT1H30M into total seconds, throwing IllegalArgumentException on invalid input.
```

luna, sol and mini all passed 7/9.

### More prompts where the flagship (gpt-6.1-sol) wins

No cluster keeps sol ahead past λ ≈ 0.15. sol is far dearer, so beyond that its accuracy edge stops
paying for itself. Prompts #2 and #11–16 below land in the three clusters where sol holds on longest,
so they pick sol at every sweep value from 0 to 0.1. #17–19 pick sol only at low λ.

| # | Prompt (start) | Cluster | sol wins for | Then |
|---|---|---|---|---|
| 11 | python-docx: paragraph.style returns None… | 3 | λ < 0.15 | luna to 0.37, then **nano** |
| 12 | In the Word document export, tables nested inside table cells… | 3 | λ < 0.15 | luna to 0.37, then **nano** |
| 13 | Starlette: request.form() hangs when the multipart body… | 10 | λ < 0.14 | luna to 1.96 |
| 14 | Our Flask-style router returns 404 instead of 405… | 10 | λ < 0.14 | luna to 1.96 |
| 15 | oauthlib: the token endpoint accepts a refresh_token grant… | 10 | λ < 0.14 | luna to 1.96 |
| 16 | dask.dataframe.read_parquet ignores the `columns` argument… | 14 | λ < 0.14 | luna to 2.30 |
| 17 | Go: the worker pool deadlocks when Stop() is called… | 18 | λ < 0.07 | luna to 3.17 |
| 18 | textdistance: jaro_winkler returns a value greater than 1.0… | 5 | λ < 0.07 | luna to 1.32 |
| 19 | mock_s3: list_objects_v2 returns keys in the wrong order… | 0 | λ < 0.04 | luna to 2.48 |

The full text of each prompt is in the example menu. Cluster 3 is mostly python-docx and
python-pptx tasks. sol passed 8/9 there, but nano passed 7/9, so from λ ≈ 0.37 the cheapest model
wins. It's the only cluster where all three tiers win somewhere inside the slider's 0–1 range.
Cluster 10 is web/HTTP frameworks (oauthlib, Starlette, bottle), and cluster 14 is mostly SWE-Gym
data-library issues.

### Things to try

- Route **#11** and drag λ from 0 to 0.5. The winner goes high tier → low → lowest (sol → luna →
  nano) inside the slider's range.
- Route **#2** and click λ = 0, 0.1, 0.2 and 0.4 in turn. This shows the change from sol to luna
  most clearly.
- Type **λ = 3** into the number box. Every prompt above then goes to **gpt-5-nano**. That only
  happens when price is weighted far above accuracy.
- Route **#7** and **#8** one after the other. They land in the same cluster even though the
  languages differ.
- Reword **#3** or **#10** slightly. Their cluster matches are borderline, so a small change can
  move them to a different cluster and a different winner.

## Caveats

- **Small samples per cluster.** Each cluster's pass rate comes from only 7–10 calibration tasks,
  so a single cluster's pass rate is uncertain by about ±30 points. A narrow sol-versus-luna win in
  one cluster is mostly noise. Large gaps, such as luna or sol against nano, are real.
- **Different calibration runs.** luna and nano were calibrated on 2026-09-08. That run's task
  file is no longer on disk, so it can't be proven that they used the same 240 tasks as sol and
  mini. The settings were identical (seed 42, 10 tasks per cluster, the same 5 sources).
- **Calibration tasks, not real usage.** The predictions come from benchmark coding tasks. Prompts
  unlike those tasks, such as chat or prose, still get assigned to the nearest cluster, but the
  prediction means less for them.

## Code

- [`ui.py`](ui.py): the HTTP server, the embedding cache, and the per-λ winner calculation
  (`lambda_segments`).
- [`ui.html`](ui.html): the single page. It has no external scripts or fonts.
- [`cli.py`](cli.py): the `router runtime ui` command.
- Tests: [`tests/runtime/test_ui.py`](../../../tests/runtime/test_ui.py).
