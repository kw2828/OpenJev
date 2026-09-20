# Typed decisions: lower loss, worse decisions

All **12 fits completed**. Separating answer-type probability from candidate competition reduced the primary prediction loss by **26.92%**, with improvement at every seed. It nevertheless **failed the fixed continuation rule, 5/9 checks passed**: rare-category recall was insufficient and false predictions increased. Changed-state accuracy fell from **50.52% to 47.40%** in the primary comparison. The original flat head with stratum weighting reached **70.88%** on the same rows.

This closes the tested normalization recipe. It supplies neither a new architecture result nor evidence that adding recurrence will fix the remaining errors. Every model receives the correct previous value; this is a conditional observation experiment, not an autonomous dialogue tracker.

![All twelve typed/support fits, primary loss, rare recall, error rates and observed cost](../output/dialogue-typed-v1/figure-01/typed-observation.png)

## What changed

The [frozen protocol](dialogue-typed-protocol.md) compared flat and typed output normalization, each with ordinary stratum weighting and rare-category weighting, at seeds 6101, 6102 and 6103. All four arms use the same frozen MiniLM features, candidate-conditioned attention, public candidate-type flags and privileged previous value. Each has 173,506 registered parameters. Initial weights and training orders match within each seed.

Both heads compute candidate scores and three branch scores: NOT_MENTIONED, DONTCARE and concrete values. Flat normalization lets candidate scores and candidate counts influence branch mass. Typed normalization separates branch mass from competition within a branch. It adds no semantic evidence, and equal registered parameter counts do not imply equal effective capacity.

Training used **29,211 rows**, twenty epochs and **2,300 updates per fit**, for **27,600 total updates**. Whole dialogues containing any of six prospectively selected services were excluded from fitting. The primary panel contains **7,819 rows**, including **578 changes** and **7,241 retained values**. All rows came from official TRAIN and were used in earlier experiments, so this is a development comparison, not untouched confirmation. Official DEV was not inferred on in this run; official test remains untouched.

## Primary comparison

The fixed contrast is typed-balanced versus flat-balanced. NLL is the mean across services of changed-state negative log likelihood, then averaged across the three fits. Lower is better.

| Seed | Flat balanced NLL | Typed balanced NLL | Difference |
|---|---:|---:|---:|
| 6101 | 2.617557 | 2.194627 | -0.422930 |
| 6102 | 2.570637 | 1.680112 | -0.890525 |
| 6103 | 2.409477 | 1.677834 | -0.731643 |
| Mean | 2.532557 | 1.850857 | -0.681699 |

The loss reduction exceeds the required 5%, and all three seed checks pass. The other criteria prevent this probability improvement from being mistaken for a useful decision improvement:

| Decision measure | Flat balanced | Typed balanced | Difference | Fixed requirement | Result |
|---|---:|---:|---:|---|---|
| TRUE-change recall | 4.60% | 9.20% | +4.60 pp | At least +5 pp | Fail |
| DONTCARE-change recall | 6.67% | 0.00% | -6.67 pp | At least +5 pp | Fail |
| TRUE false-positive rate | 1.06% | 1.80% | +0.74 pp | At most +0.5 pp | Fail |
| DONTCARE false-positive rate | 0.25% | 0.36% | +0.12 pp | At most +0.5 pp | Pass |
| Retained-state error | 2.18% | 3.52% | +1.34 pp | At most +0.5 pp | Fail |

Rates average the three final fits on identical rows. TRUE false positives use **2,039 non-TRUE rows where TRUE is an available candidate**; DONTCARE uses **7,804 non-DONTCARE rows**. Unsupported candidates never enter these denominators.

TRUE recall has only **29 changed rows**. Correct counts in seed order are **3, 0, 1** for flat-balanced and **1, 7, 0** for typed-balanced. DONTCARE has only **five changed rows in one schema query**: flat-balanced gets **0, 0, 1** right; typed-balanced gets **0, 0, 0**. These repeated fits are not independent new examples. The primary panel contains no FALSE changes or clears.

## The simpler control matters

Each number below averages all three final fits. Accuracy and Brier weight rows equally; the loss column uses the primary equal-service weighting. Brier is the multiclass sum of squared probability errors, with lower values better.

| Arm | Changed NLL | Changed accuracy | Changed Brier | Retained error | TRUE recall | DONTCARE recall |
|---|---:|---:|---:|---:|---:|---:|
| Flat, stratum weights | 2.239130 | **70.88%** | **0.500418** | 4.09% | 5.75% | 0.00% |
| Flat, rare-category weights | 2.532557 | 50.52% | 0.815889 | **2.18%** | 4.60% | 6.67% |
| Typed, stratum weights | **1.836914** | 54.33% | 0.695090 | 3.55% | 5.75% | 0.00% |
| Typed, rare-category weights | 1.850857 | 47.40% | 0.774189 | 3.52% | 9.20% | 0.00% |

Rare-category weighting does not produce a general improvement. With the flat head, it reduces retained errors but lowers changed-state accuracy by **20.36 percentage points**, worsens changed NLL and Brier, and slightly lowers TRUE recall. With the typed head, it lowers changed accuracy by **6.92 points** and worsens changed NLL and Brier; retained error is nearly unchanged. Neither weighting contrast replaces the fixed primary.

The typed-balanced head also regresses on all primary-panel rows: accuracy falls **94.32% to 92.85%**, row-mean NLL rises **0.211985 to 0.223812**, and Brier rises **0.092811 to 0.111646**, compared with flat-balanced. Lower changed-state NLL alone is not calibration evidence or a deployable improvement.

The [complete report](../output/dialogue-typed-v1/analysis-02/report.md) lists every fit. The [full summary](../output/dialogue-typed-v1/analysis-02/summary.json) includes all 144 groups per fit, row/service/dialogue means, per-service changes, deterministic references and factorial contrasts. Empty groups remain undefined.

## Execution and verification

The whole training run took **551.82 seconds (9.20 minutes)**, including shared loading, all fits, evaluation, saving and payload hashing. Process-lifetime peak RSS was **1.79 GiB**; the saved run occupies **105.30 MiB**. Each fit takes 44.83-46.11 seconds. These are observed CPU training costs, not inference latency or a speedup benchmark. An unrelated background VM was active and is disclosed in the [launch record](../output/dialogue-typed-v1/launch-01/started.json).

Feature preparation is inherited, not free: the earlier pooled, lexical and token stages recorded 15.26, 2.27 and 17.88 seconds, respectively; conditional metadata preparation took 3.89 seconds. These are separate recorded stages, not a new end-to-end timing. This experiment's 81-check preflight took 1.99 seconds; the corrected saved reader took 5.38 seconds and the independent numerical check took 1.66 seconds. Source review and engineering time are not included in those process measurements.

The original report reader failed **before prediction scoring** because it expected only two split lists, while the saved metadata contains six. Its [failure](../output/dialogue-typed-v1/analysis-01/failed.json) is preserved. A separately published [correction](../output/dialogue-typed-v1/analysis-correction-01/README.md) validates every list and passed 35 synthetic checks. All metric and continuation functions are unchanged; all 25 frozen sources, training outputs, seeds and thresholds remain unchanged. No model was retrained. The corrected reader was published in commit fbdcedf before scoring.

An independent saved-output checker reproduced **108 scoped metric cells, 60 decision-count cells and all nine continuation checks**, with no discrepancies. Its [receipt](../output/dialogue-typed-v1/independent-audit-01/result-01/receipt.json) authenticates 118 inputs. It did not independently recompute Brier, every detailed subgroup or measured execution times, and did not load model weights. The full reporter checks those saved artifacts within its declared scope.

All **56 run files**, including all twelve final weights, distributions, update logs and row orders, are [published byte for byte](../output/dialogue-typed-v1/training-01/). Completion SHA256: 65bd57084dbc7f7b99b782e140bae2e18f25b9e237e89a0132cead5fbfe2a095. Analysis SHA256: 3969bcb7803d3f625a90a9e165a775bb9108f8301b9f225d198eed2e9e84a81c.

## Next question

Neither the output factorization nor extra rare-category weighting solves the decision errors. The next bounded probe should decompose the saved loss into the correct branch's loss and the correct value's conditional loss within that branch. Use all twelve fits and unchanged rows, with paired correct-to-wrong and wrong-to-correct decision counts. This can explain whether lower NLL comes from assigning less extreme probabilities to wrong branch choices or from improved value discrimination. It requires no new inference or fitting and cannot establish a causal representation change between separately trained models.

If branch selection remains the bottleneck, a later blinded context-sufficiency check can compare the current SYSTEM/USER pair with its preceding public context, preserving the original labels and staying inside exposed training material. Recurrence becomes motivated only where history supplies missing evidence; otherwise strengthen the semantic observation baseline.

This is a proposed diagnostic, not an executed result, a changed continuation rule or an ICLR novelty claim.
