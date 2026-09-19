# Normalized memory is valid; selective retention still fails

All **15 fresh fits** completed after correcting recurrent probability normalization. The numerical checks and independent saved-output audit passed. The scientific result remains **FAIL: 7/13 predeclared checks passed**. Selective memory reaches **72.89%** unseen-service macro accuracy, versus **72.58%** for simpler scalar memory and **65.76%** for literal copying. Its advantage over scalar remains below the required half percentage point, and it loses on seen services.

This replaces the compromised scalar comparison in the [historical study](dialogue-copy-results.md). It is development evidence for a supplied-service, supplied-slot, finite-candidate task, not an untouched confirmation or an established new architecture. Official test contents remain untouched.

![All fifteen corrected fits and both deterministic references](../output/dialogue-copy-v2/report-01/comparison.png)

## What was corrected

The [failed internal replay](dialogue-evidence-qualification-results.md) found that the old scalar update amplified probability-mass drift even though final softmax outputs were valid. V2 normalizes the categorical state before it enters features and after each real update. Padding remains unchanged; gradients stay attached. No new learned parameter, observation, loss, hyperparameter, probability floor or mass clipping was added.

The [published protocol](dialogue-copy-v2-protocol.md) retained all five methods, three seeds, training data, lexical features, epoch orders, 20 epochs and final-only evaluation. Every fit started from tensors matching its original initialization digest. These are new fits, not repaired old checkpoints. The same 13 scientific criteria apply. All source files were frozen in [commit 8f93782](https://github.com/kw2828/OpenJev/commit/8f93782) before training.

## Results

Means include every seed. Three-stratum macro accuracy gives equal weight to unmentioned retention, assigned retention and changed states. The development panel has 29,236 seen-service and 33,093 unseen-service questions, including 241 and 203 revisions. It contains no clear examples.

| Model | Seen macro | Unseen macro | Seen revisions | Unseen revisions |
|---|---:|---:|---:|---:|
| Readout with literal-history features | 73.19% | 65.77% | 57.26% | 62.56% |
| Scalar candidate memory | 79.12% | 72.58% | 55.05% | 72.58% |
| **Selective candidate memory, primary** | **78.01%** | **72.89%** | **56.15%** | **73.56%** |
| Selective, lexical/history features removed | 71.49% | 51.78% | 40.25% | 39.41% |
| Candidate GRU | 79.78% | 68.67% | 56.43% | 70.61% |
| Literal mention and carry | 54.75% | 65.76% | 49.79% | 79.80% |
| Always NOT_MENTIONED | 33.33% | 33.33% | 0.00% | 0.00% |

Scalar memory exceeds readout by **6.81 points** on unseen macro accuracy. Readout retains deterministic literal history, so it is not completely stateless. Removing the selective model's eight lexical/history observations lowers unseen macro accuracy by **21.11 points**; the bundled ablation does not isolate which observation causes the gain.

The six failed requirements are unchanged:

- Selective-minus-scalar macro is **-1.12 points seen** and **+0.31 points unseen**, missing the required +0.5 on both panels.
- Seen NLL is worse: **0.55345 versus 0.52294**.
- Selective loses all three paired seen macro comparisons.
- Selective trails the GRU's seen macro accuracy.
- Unseen revision accuracy is **6.24 points below literal carry**, outside the allowed one-point deficit.

Selective wins two of three unseen macro pairs and has slightly better unseen NLL, **0.74290 versus 0.75139**. Those passes do not overturn the fixed failure. All unrounded criteria, proper scores and seed values are in the [generated report](../output/dialogue-copy-v2/report-01/report.md) and [summary](../output/dialogue-copy-v2/report-01/summary.json).

Compared with V1, every family's seen/unseen mean macro changes by less than **0.015 percentage points**; revision accuracies are unchanged. This is a descriptive fresh-fit comparison. The correction restores valid state arithmetic but does not produce the missing selective-retention advantage. It does not retrospectively validate the old internal states.

## Remaining errors

A separate [saved-output diagnostic](../output/dialogue-copy-v2/error-diagnostic-01/result-01/README.md) preserves the historical subgroup definitions and includes every fit. It is outside the continuation rule.

| Unseen subgroup | Support per fit | Scalar | Selective | Literal |
|---|---:|---:|---:|---:|
| Assigned boolean, all targets TRUE | 524 | 5.22% | 6.62% | 0.00% |
| Assigned FALSE | 0 | undefined | undefined | undefined |
| DONTCARE | 179 | 0.00% | 0.00% | 0.00% |

Seen assigned-TRUE accuracy is 87.29% for scalar and 87.12% for selective. On unseen revisions, selective makes a mean 53.67 errors: 43 choose the previous scored gold value and 10.67 choose another wrong value. These endpoint patterns cannot identify the internal cause, and the absence of FALSE targets prevents a general polarity claim. The three seeds reuse the same examples.

The [source-reviewed observation control](dialogue-observation-design.md) remains a useful next hypothesis: compare current independent pooled embeddings against joint schema/candidate/dialogue encoding, each under matched readout and normalized scalar heads. Target TRUE and DONTCARE interpretation under service shift, without claiming a demonstrated negation deficit. Freeze that comparison and its full encoding-cost accounting before any new fitting. No such encoding or training ran as part of this study.

## Numerical validity, cost and provenance

Training and evaluation checked **27,556,800** and **1,760,790** real supplied-question updates. With executed dummy question positions included, each of the incoming-state, feature-prior and outgoing-state checks covered **56,202,005** positions. The largest mass error was **3.123e-7**, below **2e-6**. All **33,721,203** released-mass checks passed, with zero overshoots above one. These engineering checks establish recorded numerical validity, not predictive effectiveness.

The whole run took **1,486.05 seconds** on an Apple M5 Max with four CPU threads, within the fixed 3,600-second cap. It performed **19,200 optimizer updates**, **15,522,300 supervised presentations** and produced **934,935 saved development predictions**. All 15 fits completed on their first attempt. There were no new encoder or external model API calls. Shared prior MiniLM encoding cost 15.26 seconds and lexical preparation 2.27 seconds. Instrumented training/evaluation time is not serving latency; it cannot be interpreted as a V1 speed comparison.

Non-GRU heads register 99,458 parameters and candidate GRU 103,411. The [report](../output/dialogue-copy-v2/report-01/report.md) retains per-family time and all per-fit work counts. The 89 V2 model/runner/reporter tests passed before freezing; 15 independent-auditor and 21 diagnostic tests passed before their saved-output executions. Source-only peer reviews found no material blocker.

An [independent NumPy audit](../output/dialogue-copy-v2/independent-audit-01/result-01/completed.json) recomputed every metric and all 13 decisions, authenticated all **64 execution files** and **20 frozen sources**, and reconstructed recorded batch-check coverage. It agreed with the reporter. It did not replay training or deserialize weights; full configuration validation remains the authenticated reporter's responsibility. The error diagnostic separately reuses hash-pinned historical subgroup arithmetic, so it is not an independent implementation of those definitions.

[Public execution metadata](../output/dialogue-copy-v2/execution-01/manifest.json) includes exact copies of all receipts and training batch journals: 33 of the 64 original files. Weights, individual predictions and reference predictions remain local, with their hashes published. [Reproduction instructions](dialogue-copy-v2-reproduction.md) describe the required authenticated local inputs, not a tested clean-install recipe.

| Artifact | SHA-256 |
|---|---|
| Frozen plan | `9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905` |
| Execution completion | `768376e24824523b9a5c26923c14206e567fda4f3578ed1babdf7cbdfb1d89f4` |
| Report receipt | `fa4e47979b9fab18c40f009423ed1aeb0603991aa4944b29cfa8ffb5e6002a2a` |
| Independent audit completion | `0cea53e6bbd75748f3c3bfa16ed8e578bdbe100c8beec8696db55a71d7ab9a0e` |
| Error diagnostic receipt | `83e19988960418bd286e9f0854ccfa487b8bd7f6dad87ca5947b7df9945b981e` |

The source is [Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78), licensed CC BY-SA 4.0. Its human-paraphrased simulated outlines, supplied schemas/candidates, repeated development exposure, small revision subgroups and frozen transformer's unknown pretraining overlap limit the claim. Unseen service is not necessarily unseen domain. This experiment establishes neither RL, calibrated probabilities, connectome learning, a learned world model nor ICLR readiness.
