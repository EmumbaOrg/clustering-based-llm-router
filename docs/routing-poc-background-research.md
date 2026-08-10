# **Model Router for Coding Agents – PoC Implementation Specification**

## **1\. Objective**

Build a model-routing extension for the Pi coding-agent harness. In a previous version of this PoC, we already built a model-routing extension in Pi. Reuse the current routing extension structure and replace only its decision logic with the routing mechanism outlined below.

## **2\. Routing** **Mechanism**


In summary, for each new coding task, the router will:

1. Identify the semantic cluster that best matches the task.  
2. Estimate the expected error and execution cost of each candidate model.  
3. Select the model with the best configured accuracy–cost trade-off.  
4. Keep that model for the full agent run unless an escalation / budget constraint condition is triggered.

### **High-level architecture**



## **3\. Query representation**

Use the initial user prompt as the routing input.

Freeze one embedding model and version for the full PoC.

A reasonable local default is `jina-embeddings-v2-base-code`. 

Store the embedding model ID in a Readme file or similar. Changing the embedding model requires rebuilding the cluster map and model profiles.

## **4\. Initial Clustering**

### **4.1 Clustering corpus**

The clustering corpus does not require ground-truth answers. Only the task prompt is used.

The corpus should primarily contain repository-level coding-agent tasks. Standalone programming problems should be included in a smaller proportion so that the map also covers code generation, API use and data-science tasks.

### **Recommended corpus**

| Source | Recommended use |
| ----- | ----- |
| **SWE-smith** | Sample approximately 20,000 tasks. SWE-smith provides more than 50,000 software-engineering tasks from 128 repositories. Cap its contribution so synthetic Python tasks do not dominate the map. |
| **SWE-Gym** | Include all 2,438 real repository-level Python tasks. Each includes a natural-language task, executable environment and tests. |
| **Multi-SWE-RL** | Include its 4,723 multilingual issue-resolution instances to improve language and ecosystem coverage. |
| **BigCodeBench-Instruct** | Include its 1,140 instruction-based tasks to represent library use, function calls and greenfield implementation. |
| **DS-1000** | Include its 1,000 realistic data-science programming tasks across seven Python libraries. |

This produces an initial corpus of approximately 29,000 prompts.

Deduplicate prompts and prevent any single repository, dataset or programming language from dominating the corpus.

Public benchmark data is only a proxy for real Pi usage. Anonymised internal Pi task prompts should gradually be added once enough representative sessions are available.

### **4.2 K-means configuration**

Embed prompts and then run K-means clustering on the embeddings. Start with this configuration for K-means clustering:

```
Embedding normalisation: L2
Distance: Euclidean on normalised vectors
Candidate K values: 16, 24 and 32
Default K: 24
Random initialisations: at least 10
```

Select K using:

1. cluster stability across random seeds;  
2. reasonable semantic coherence;  
3. sufficient calibration coverage;  
4. absence of extremely small clusters.

## **5\. Model calibration**

### **5.1 Calibration dataset**

A labelled subset of the clustering corpus can be used for model calibration. You first need to generate ground truth answers or use a scoring criterion that allows marking an answer as correct or incorrect. Since we are picking tasks from benchmarks, these typically come with executable tests or in some cases ground truth answers.

Make sure to:

* select the subset after clustering;  
* ensure adequate representation of every cluster;  
* keep the final evaluation set completely separate;

Target **800 prompts** for the first full calibration. With an 800-prompt calibration set, aim for at least 20 calibration prompts per cluster. Reduce K or merge clusters when this cannot be achieved. Aim for the following distribution:

* approximately 60% repository-level Python tasks;  
* approximately 25% multilingual repository tasks;  
* approximately 15% standalone implementation and library-use tasks.

Use constrained, cluster-stratified sampling rather than a purely random sample.

## **5.2. Build model cluster profiles**

For each candidate model and prompt:

1. Start from the same clean environment.  
2. Run the task through the same Pi harness.  
3. Use identical tools, system prompts, turn limits and execution budgets.  
4. Run the task’s test suite.  
5. Record success or failure.  
6. For future analysis, record the complete agent cost and execution metadata.

For every candidate model and cluster, store:

```
number_of_tasks
number_succeeded
number_failed
raw_error_rate
smoothed_error_rate
mean_session_cost
median_session_cost
mean_input_tokens
mean_output_tokens
mean_agent_turns
mean_tool_calls
```

**A note on cost definition:**

Do not use the model’s published price per million tokens. Use the **observed total Pi session cost** for the complete task based on total input and output tokens across a full session.

## **6\. Runtime routing** 

When a user query arrives:

1. Generate its embedding using the same embedding model used to build the cluster map.  
2. Assign it to the nearest K-means centroid.  
3. Let the assigned cluster be `c`.  
4. Calculate a routing score for every remaining candidate model `m`.

Use the model’s calibrated error rate for the assigned cluster:

```
predicted_error[m] = cluster_error[m, c]
```

This is the observed failure rate of model `m` on calibration tasks assigned to cluster `c`.

Use a fixed cost value derived from each model’s published API pricing:

```
static_price[m] =
input price per 1M tokens
+ output price per 1M tokens
```

This value is a deterministic cost index. It is not intended to predict the actual cost of the current Pi session. Cached-input pricing, expected token counts, predicted agent turns, and historical session costs are excluded from the PoC routing decision.

Normalise the static price across the current eligible model set:

```
normalised_cost[m] =
(static_price[m] - cheapest_static_price)
/
(most_expensive_static_price - cheapest_static_price)
```

This gives the cheapest eligible model a cost score of `0` and the most expensive eligible model a cost score of `1`.

Calculate:

```
routing_score[m] =
predicted_error[m]
+ lambda × normalised_cost[m]
```

Select the eligible model with the lowest routing score.

The parameter `lambda` controls the accuracy–cost trade-off:

* `0`: prioritise predicted accuracy only.  
* A small value: modest preference for cheaper models.  
* A larger value: stronger preference for cheaper models.

Do not select a permanent value of `lambda` before evaluation. Sweep several values and compare the resulting task accuracy and realised Pi session cost.

Suggested initial sweep:

```
0
0.02
0.05
0.10
0.20
0.40
```

***Actual session cost should still be calculated after execution using the observed token and cache usage. It is used for router evaluation and savings reporting, but not as an input to the routing decision.***

Preserve the escalation and budget-constraint mechanisms already implemented in the previous PoC version.

## **7\. Evaluation plan**

Use a held-out portion of the benchmark that was not used for clustering or calibration. Label it similar to how you labeled the dataset for model calibration.

Recommended evaluation sources:

* **SWE-bench Verified:** 500 expert-verified repository tasks.  
* **SWE-bench Multilingual:** 300 tasks across 42 repositories and nine languages.  
* **SWE-bench Live:** More recent, lower-contamination issue-resolution tasks. The May 2026 multilingual release contained 743 tasks across six languages and 381 repositories.

Start with a balanced 200–300 task evaluation subset.

### **Primary metrics**

* Task resolution rate.  
* Total and average session cost.  
* Cost reduction relative to always-strong.  
* Resolution-rate difference relative to always-strong.  
* Model selection distribution.

Compare the router against:

1. Always using the strongest model.  
2. Always using the cheapest model.  
3. Per-task oracle, reported only as an upper bound.

Report results across the full `lambda` sweep rather than presenting only the best setting.

## **8\. Acceptance criteria**

The PoC is complete when:

* a new Pi task is routed before the first LLM call;  
* all routing decisions are reproducible from versioned artifacts;  
* every candidate model has cluster-level quality and cost profiles;  
* a new model can be onboarded by running only the calibration suite;  
* adding a model does not require rebuilding K-means;  
* router accuracy and cost are measured on a held-out benchmark;  
* results include an accuracy–cost curve and always-strong baseline;  
* the router adds negligible latency compared with an agent session.

## **9\. Future enhancements**

Enhancements should be driven by observed failure patterns. Some possibilities:

1. Highest priority: soft weighting over the nearest clusters, similar to Avengers-Pro’s use of several nearby clusters or UniRoute’s LearnedMap of soft cluster weights.  
2. Query profiling: queries can be scored for reasoning, debugging, code generation, error handling and other capabilities required. This can be done through labeling a training set of queries with LLM-generated attributes which are later distilled into a small classifier, similar to HyDRA and UniRoute’s attribute representation.  
3. Including repository metadata and lightweight codebase features in the routing decision.  
4. Experiment with building model profiles from public benchmarks instead of requiring calibration, like HyDRA.  
5. Online learning/updates to model cluster profiles from validated internal Pi sessions.  
6. Mid-session re-routing after every user or tool message.
