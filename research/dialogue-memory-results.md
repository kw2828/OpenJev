# Recurrent text memory: a useful signal, but a simpler rule still wins under shift

**The proposed surprise-adaptive memory failed its fixed continuation rule: 8/11 checks passed.** It improved unseen-service macro accuracy from ordinary Kalman's **52.16% to 53.98%**, but a literal-match-and-carry rule reached **65.76%** there. This is not a new-architecture breakthrough or an ICLR-ready result. The fixed recipe is closed without retuning it on these development outcomes.

![All seven neural models, all three seeds, and both deterministic memory references](../output/dialogue-memory-v1/visualization-01/comparison.png)

## What we trained

Twenty-one small decision models, seven families across three seeds, trained for 12 epochs on the same **2,017 dialogues and 51,741 categorical state labels**. All final models were evaluated on **2,363 official development dialogues and 62,329 queries**. No official test dialogue contents were accessed. Training and final evaluations took **1,036.55 seconds**, with 16,128 optimizer updates, 13,038,732 supervised query presentations, and no external model API calls. There were no failed or replaced fits.

The task uses the [Schema-Guided Dialogue dataset](https://arxiv.org/abs/1909.05855), whose simulated dialogue outlines were paraphrased by people. A caller supplies the service, slot and candidate set. The model infers the current categorical value from public dialogue so far. This is conditional state tracking, not the official full dialogue-state task or service routing. The streams are short: a median of ten USER turns and a development maximum of 29.

All methods share a frozen MiniLM Transformer encoder. The global memory models update only from the preceding SYSTEM utterance and current USER utterance, without gold state, frame metadata, or query identity. Questions and candidates enter at read time. The stronger learned carry control can condition on each supplied question while replaying its prefix, using its own predicted state rather than a gold previous value.

The candidate mechanism adds key-local covariance inflation when a learned latent residual is surprising. It borrows dense covariance tracking from [Kalman Delta Networks](https://arxiv.org/abs/2609.07816), with an adaptive-filter extension. Neither the Kalman update nor surprise-dependent adaptation is claimed as a new invention. This experiment trains supervised memory heads, not RL policies, connectomes or action-conditioned world models.

## Complete comparison

Macro accuracy gives equal weight to three strata: an unmentioned value remaining unmentioned, an assigned value being retained, and a value changing. Neural rows are equal means over all three final fits. They reuse the same development questions; three seeds are not three independent datasets. References run once on those same questions.

| Method | Seen macro accuracy | Unseen macro accuracy | Seen revision accuracy | Unseen revision accuracy |
|---|---:|---:|---:|---:|
| Current turn | 57.52% | 47.91% | 38.04% | 38.42% |
| GRU | 57.53% | 46.32% | 32.92% | 38.26% |
| Full attention | 59.33% | 46.93% | 31.40% | 36.45% |
| Gated delta | 71.82% | 52.44% | 44.12% | 40.89% |
| Kalman | 71.30% | 52.16% | 37.90% | 42.04% |
| Innovation Kalman, fixed proposal | 71.65% | 53.98% | 38.17% | 39.57% |
| Learned carry | 59.44% | 48.10% | 36.79% | 34.65% |
| Literal mention and carry | 54.75% | 65.76% | 49.79% | 79.80% |
| Always NOT_MENTIONED | 33.33% | 33.33% | 0.00% | 0.00% |

Seen/unseen means membership in the official training service schema, not entirely novel domains. There are **241 seen and 203 unseen revision queries per fit**, with no development clears. Revision accuracy excludes first assignments and retained states. The literal reference chooses the longest uniquely matched bounded ontology string in USER text and otherwise retains its prior choice. It has no negation, yes/no, or DONTCARE parser and uses no annotation spans. Even this limited rule beats every neural family on unseen macro accuracy and on revision accuracy in both panels.

The proposal passes all five unseen-service checks. Its **1.83 percentage-point** macro gain over ordinary Kalman appears in all three paired seeds, and mean NLL improves from **1.1652 to 1.0700**. But its unseen revision accuracy falls from **42.04% to 39.57%**. The aggregate improvement does not support a revision-specific benefit.

Three seen-service requirements fail: the macro gain over Kalman is **0.36 points**, below the required one point; NLL worsens from **0.7229 to 0.7279**; and macro accuracy is below gated delta by **0.17 points**. Those failures close the proposal even though it beats the tested GRU, attention and learned carry heads on macro accuracy. The literal rule was a reported reference rather than a fixed gate comparator; its stronger unseen result is an additional practical limitation, not a retroactively changed threshold.

Micro accuracy tells another useful part of the story: across all development queries, the literal rule scores **72.66%**, compared with **68.30%** for the proposal. Empty-state persistence is common, which is why the protocol also requires the three-stratum view. Full micro accuracy, NLL, Brier scores, transition bins, every seed and every fixed check are in the [saved-output report](../output/dialogue-memory-v1/report-01/report.md) and [machine-readable summary](../output/dialogue-memory-v1/report-01/summary.json).

## Computation and limits

Each matrix model has 66,675 trainable parameters downstream of the frozen encoder. The GRU has 155,523 and learned carry 166,434 registered parameters; some parameters in other arms are unused by their selected update, with final gradient counts recorded per fit. This is a common-data/common-epoch comparison, not parameter- or compute-matched training.

Summed across three fits, gated delta trained in **113.18 seconds**, ordinary Kalman in **140.75**, and the proposal in **143.45**. Their final evaluations took **6.85**, **6.96** and **7.51 seconds**, respectively. These are local batch timings including preparation of batches and prediction storage, not serving latency. Carry replays prefixes per question and is not an optimized incremental dictionary implementation.

At float32, the matrix-state payload is 1,024 bytes per dialogue for delta and 2,048 for either Kalman variant. Full attention's retained tensor payload is 129 bytes per stored turn, so recurrence is not automatically smaller on these short streams. These counts exclude the encoder, parameters, temporary tensors, autograd and Python objects. No peak-memory or end-to-end speed advantage is established.

The adaptive inflation activated on **11.21%, 52.06% and 47.64%** of development USER updates across the three fits. These source-bound diagnostics are learned-feature quantities, not calibrated semantic confidence. They do not explain causally which prediction changes helped. MiniLM's pretraining overlap was not audited; duplicate checks only excluded exact normalized whole-dialogue overlap, finding none. The results are descriptive development evidence, without a significance test or confidence-interval claim.

## What follows

The useful finding is that matrix memories outperform the tested conventional neural heads in this small-data setup. The central obstacle is stronger: simple exact-value retention transfers better than every learned model to unseen services. A future experiment should test whether language-mediated update detection adds value to an explicit state ledger, especially on paraphrases, negation and genuine corrections, while retaining the literal baseline. That requires a new frozen protocol and fresh confirmation evidence. It is a hypothesis for the next study, not a result from this one.

## Reproduce and inspect

- [Frozen protocol](dialogue-memory-protocol.md), [exact training plan](../output/dialogue-memory-v1/execution-01/plan.json), and [reproduction commands](dialogue-memory-reproduction.md).
- [Source review](sgd-state-source-review.md), [preparation and preserved recoveries](dialogue-memory-preparation.md), and [preparation manifest](../output/dialogue-memory-v1/preparation-manifest.json). Dataset: CC BY-SA 4.0; encoder: Apache 2.0.
- [Execution completion and all 21 fit receipts](../output/dialogue-memory-v1/execution-01/completed.json), [execution metadata manifest](../output/dialogue-memory-v1/execution-01/manifest.json), and [saved-output audit](../output/dialogue-memory-v1/report-01/receipt.json).
- [Independent arithmetic cross-check](../output/dialogue-memory-v1/independent-audit-01.json), implemented separately without importing the reporter's metric functions.
- [Revision comparison including literal memory](../output/dialogue-memory-v1/visualization-01/revisions.png), [visualization receipt](../output/dialogue-memory-v1/visualization-01/receipt.json), and [original six-panel plot](../output/dialogue-memory-v1/report-01/accuracy.png).
- [Model implementation](../src/openjev/research/dialogue_fast_memory.py), [carry control](../src/openjev/research/dialogue_carry.py), [runner](../scripts/study_dialogue_memory.py), and [independent reporter](../scripts/report_dialogue_memory.py).

The 142 synthetic preflight tests and Ruff passed before fitting. The reporter authenticated all 21 weight/prediction hashes, exact cohort membership and labels, paired matrix initialization, and complete training counts, then recomputed the scores without neural inference. This verifies saved-output arithmetic and provenance; it does not independently replay learned predictions. Raw corpora, features, individual predictions and weights remain local. The full fetch had a preserved seven-file TLS failure and recovery; one encoder preparation failed and a separate attempt succeeded. Neither was a scored-fit retry.

A separate saved-output audit checked all **1,308,909 prediction rows**, authenticated 66 execution files and 12 frozen source files, and independently recomputed the metrics and all 11 continuation decisions. Metric discrepancies were at most **2.22e-16** and every pass/fail decision matched. Neither reporting pass reran the neural models.

Training plan SHA-256: `8c62bd5853299f29fc232b044b69f24d1d0bf4fe9e1a860954a1515d1a617a38`. Execution completion: `d3b59edc9000237fe5e9cc34cf91d9d416bd2a3561b60044aeda2a02610cbc44`. Saved-output audit receipt: `4b661bffe85cd63d5b496a2232d3c6a15137637e063162a01d37d307a89a8b99`.
