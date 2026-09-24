# Same GRU, conditional cost labels: DEV_FAIL

**This is a training-label comparison on a new fixed-horizon task.** Six models were fitted from scratch, and the independent saved-output audit completed under the original process closures. The result does not change earlier study outcomes or establish a new architecture.

![All paired fits and prespecified five-percent margins](otto-conditional-label-results/benchmark.png)

Both arms use the same 32 hypothetical teacher-cost labels per TRAIN prefix. `sampled` visits every label three times across 96 epochs; `mean32` uses their precomputed mean. Paired models have the same initialization, case order, optimizer recipe and number of updates. All eight blind transitions execute, but only the horizon-eight cost prediction is trained. Lambda3 is the TRAIN regime; lambda4 is a held-out sensing-regime shift.

| Setting | Prefixes | Sampled regret | Mean32 regret | Gain | Gain minus 5% | Approx. 95% margin interval | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| lambda3 | 79 | 0.250508 | 0.238248 | 0.012259 | -0.000266 | [-0.035844, 0.037920] | fail |
| lambda4 | 94 | 0.242702 | 0.257626 | -0.014924 | -0.027059 | [-0.046485, -0.009706] | fail |

Regret is `Q[chosen] - min(Q)`, using the lowest-index minimum predicted action. Found histories contribute zero and remain in every denominator. Draws are averaged within each prefix, then prefixes equally. The primary result averages three policies as separate decisions, not an ensemble of their costs.

| Setting | At least 64 prefixes | Positive sampled regret | Positive lower 5% margin | All 3 paired gains positive |
|---|---|---|---|---|
| lambda3 | True | True | False | False |
| lambda4 | True | True | False | False |

Both settings must pass every condition. Intervals use 2,000 paired hierarchical bootstrap replicates per setting and are conditional on the fixed TRAIN bank and fitted policies. They do not include training variability or claim rigorous coverage. The 32/128 draws and three fit seeds are not independent cases.

## What the comparison controls

For each prefix, let `z_j` be centered teacher costs divided by 64, including found zeros, and `z_bar = mean_j(z_j)`. At a fixed prediction `f`, the finite-bank identity is

```text
mean_j ||f - z_j||^2 = ||f - z_bar||^2 + mean_j ||z_j - z_bar||^2.
```

The last term does not depend on the prediction, so full-bank squared loss and mean-target loss have the same gradient and finite-bank optimum. The intervention here changes per-update gradient noise under matched optimizer work. It does not match target contributions per update, and a 32-draw mean is still an estimate rather than an exact conditional expectation. See the [target/loss implementation](../src/openjev/research/otto_conditional_label.py), [fabricated identity checks](../tests/test_otto_conditional_label.py) and [frozen protocol](otto-conditional-label-protocol.md).

The common scale is derived only from all sampled TRAIN contrasts, not from the smaller mean-target variance. Both arms use the unchanged 28-state GRU: 8,299 total parameters, 8,096 potentially updated parameters and 203 frozen outcome/auxiliary-head parameters. Their forward compute is still charged. Only the public nine-row prefix, lengths and committed action sequence enter the model; beliefs, hidden sources, draws and teacher costs are not model inputs.

## Every paired fit and secondary loss

| Setting | Fit seed | Sampled regret | Mean32 regret | Paired gain | Sampled target MSE: sampled / mean32 | Mean-target MSE: sampled / mean32 |
|---|---|---:|---:|---:|---|---|
| lambda3 | 343000001 | 0.235395 | 0.233097 | 0.002297 | 0.00004857 / 0.00004713 | 0.00002292 / 0.00002148 |
| lambda3 | 343000002 | 0.255191 | 0.214628 | 0.040563 | 0.00005102 / 0.00005258 | 0.00002537 / 0.00002693 |
| lambda3 | 343000003 | 0.260938 | 0.267020 | -0.006082 | 0.00005438 / 0.00005517 | 0.00002874 / 0.00002952 |
| lambda4 | 343000001 | 0.257927 | 0.261163 | -0.003236 | 0.00005786 / 0.00005643 | 0.00003529 / 0.00003387 |
| lambda4 | 343000002 | 0.247372 | 0.243969 | 0.003403 | 0.00005174 / 0.00005542 | 0.00002917 / 0.00003285 |
| lambda4 | 343000003 | 0.222808 | 0.267748 | -0.044940 | 0.00005909 / 0.00006558 | 0.00003653 / 0.00004302 |

MSE is in unstandardized centered `Q/64` units against the independent DEV bank. Within-bank variance and all case-level scores remain in the JSON. Lower MSE alone does not imply lower decision regret.

## Support and charged computation

Retained **502/768** fresh observed prefixes; **266** found-during-prefix exclusions were logged without replacement. TRAIN retained **329** lambda3 prefixes with 32 draws each. No native environment continuation was executed after the observed prefix.

| DEV setting | Prefixes | All-found banks retained | Surviving / allocated draws | Minimum surviving draws |
|---|---:|---:|---|---:|
| lambda3 | 79 | 0 | 9395/10112 | 71 |
| lambda4 | 94 | 0 | 11373/12032 | 71 |

| Original phase | Worker seconds |
|---|---:|
| collection | 1200.875 |
| fit | 15.972 |
| audit | 69.086 |

| Arm | Fit seed | Fit seconds | DEV inference seconds | Optimizer updates | Conceptual target contributions |
|---|---|---:|---:|---:|---:|
| sampled | 343000001 | 2.776 | 0.091 | 1056 | 31584 |
| mean32 | 343000001 | 2.107 | 0.085 | 1056 | 1010688 |
| mean32 | 343000002 | 2.126 | 0.081 | 1056 | 1010688 |
| sampled | 343000002 | 2.196 | 0.085 | 1056 | 31584 |
| sampled | 343000003 | 2.197 | 0.084 | 1056 | 31584 |
| mean32 | 343000003 | 2.377 | 0.088 | 1056 | 1010688 |

Total new worker time: **1285.933s**. Shared bank construction used **9744** TRAIN and **20768** DEV teacher endpoint annotations, plus **5226** actual prefix steps. The mean target was aggregated once from **10528** draws in **0.000098s**. Its repeated conceptual contributions are not repeated teacher calls or repeated reductions. Fit and inference times are nested within phase time; do not add them again. Qualification, historical teacher training and publication are outside this total. Matched updates do not establish equal wall time or a speedup.

## Interpretation limits

This is supervised fixed-path teacher-cost imitation. It differs from the earlier multi-horizon, survival-normalized objectives and leaves their gates unchanged. A pass supports this label-aggregation intervention under the registered recipe and scenario shift; a failure means it did not meet the preset continuation rule. Neither result establishes calibration, reinforcement learning, an architectural advance, autonomous return or benefit/no harm on normal-observation tasks. Those were not tested.

The audit replays saved sampling laws, targets, schedules, metrics and bootstrap, and checks checkpoint schemas/hashes. Teacher values and optimizer updates are not independently re-executed. Paired initial equality remains a producer attestation supported by frozen source and fabricated invariants. All audit limitations are retained in the JSON.

[Complete scores and costs](otto-conditional-label-results/summary.json) | [Protocol](otto-conditional-label-protocol.md) | [Original closure](../output/otto-conditional-label-v1/closure-01.json)
