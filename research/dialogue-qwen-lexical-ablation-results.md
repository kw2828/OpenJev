# Removing lexical flags improves update accuracy but fails the retention test

The single complete lexical-input ablation **fails its development continuation
rule: 5 of 16 conditions pass, and neither context arm passes**. Removing the
lexical flags and their explanatory text improves changed-state accuracy from
**83.74% to 85.64%** for `current` and **82.35% to 84.60%** for `history4`.
It does not deliver the required retention improvement. Current-context retained
errors rise from **1,260 to 1,437 of 7,241**; history-context errors rise from
**1,569 to 1,570**. Overall NLL worsens for both arms under both weightings.

This rejects the specified cleaner-input promotion rule. It does not establish
that every lexical feature helps, that removing flags never helps individual
services, or that a recurrent architecture would fix the remaining errors.

![Original and no-flags Qwen results under row and equal-service weighting](../output/dialogue-qwen-lexical-ablation-v1/figure-01/comparison.png)

## Fixed intervention and scope

The [prospective protocol](dialogue-qwen-lexical-ablation-protocol.md) preserves
the frozen `mlx-community/Qwen3-4B-Instruct-2507-4bit` model at revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, tokenizer, candidate readout,
temperature, label-order tie rule and full-prompt batch method. It removes only
the ten candidate lexical flags, their two TASK explanatory sentences and the
question's flag-order legend. All other prompt characters, candidate mappings,
row identities, grouping and order are preserved. Evaluator labels are copied
byte-for-byte, without decoding them during preparation or inference.

Each arm completes the same **7,819 rows**: **578 changes**, **4,032 unmentioned
retentions** and **3,209 assigned retentions**. `current` sees the preceding
SYSTEM/current USER pair; `history4` adds up to three earlier public exchanges.
Both receive the **correct previous value**, schema and candidates. The original
lexical bundle also contained a full-prefix literal register; removing it removes
that information as well as lexical hints and explanatory text. This bundled
intervention cannot identify the effect of an individual flag or instruction.

These are repeatedly exposed **official TRAIN** rows from six services held out
of the earlier local fitting split. No new official DEV or TEST evaluation,
training, generated answers, calibration, threshold fitting or model selection
occurred. Unknown model pretraining overlap remains a limitation. This is a
paired observation comparison with gold previous state, not autonomous memory,
RL training or an architecture result. The [original results](dialogue-qwen-observation-results.md)
and their failed decisions remain unchanged.

## All four original and new results

Each entry is **row weighting / equal-service weighting**. Accuracy and error
are percentages; NLL is in nats; Brier is summed squared categorical error.
Equal-service means give each supported service the same weight within the
stratum. There is one frozen-model execution per condition, not three fitted
seeds or independent evaluation populations.

| Input and context | Changed accuracy | Retained error | Overall accuracy | Overall NLL | Overall Brier |
|---|---:|---:|---:|---:|---:|
| Original current | 83.7370 / 72.9665 | 17.4009 / 16.8298 | 82.6832 / 82.8765 | 1.5871 / 1.4383 | 0.3095 / 0.3022 |
| No-flags current | 85.6401 / 77.5376 | 19.8453 / 19.2376 | 80.5602 / 81.0071 | 1.8403 / 1.6251 | 0.3575 / 0.3483 |
| Original history4 | 82.3529 / 70.5144 | 21.6683 / 19.2217 | 78.6290 / 80.3768 | 2.3085 / 2.0541 | 0.4010 / 0.3681 |
| No-flags history4 | 84.6021 / 77.8851 | 21.6821 / 19.0124 | 78.7825 / 80.8294 | 2.5104 / 2.1328 | 0.4025 / 0.3611 |

Changed-state correct counts rise **484 to 495** for current and **476 to 489**
for history4. Current loses **166 correct decisions overall**. History4 gains
**12 overall**, but its pooled retained error and Brier slightly worsen. Their
equal-service directions reverse: history4 retained error improves by **0.2093
percentage points**, and Brier improves by **0.00704**. The retention reduction
is still well below the required two percentage points.

The complete [saved summary](../output/dialogue-qwen-lexical-ablation-v1/report-01/summary.json)
also reports equal-dialogue scores, branch/value errors, all service and type
partitions, and the unchanged historical context. Historical trained controls
are not matched for model size, pretraining or compute. Original literal-register
controls were unavailable to the no-flags actor and are descriptive context only.

## All sixteen fixed conditions

Every difference is **no-flags minus original**. Rate differences below are
percentage points, while NLL and Brier differences retain their score units.
Decisions use the unrounded values and inclusive boundaries. Both weightings
must pass every condition for both arms.

| Condition | Requirement | Current difference | Current | History4 difference | History4 |
|---|---:|---:|---|---:|---|
| Retained error, row | ≤ -2 pp | +2.4444 pp | FAIL | +0.0138 pp | FAIL |
| Retained error, equal-service | ≤ -2 pp | +2.4078 pp | FAIL | -0.2093 pp | FAIL |
| Changed accuracy, row | ≥ -1 pp | +1.9031 pp | PASS | +2.2491 pp | PASS |
| Changed accuracy, equal-service | ≥ -1 pp | +4.5710 pp | PASS | +7.3707 pp | PASS |
| Overall NLL, row | ≤ 0 | +0.253230 | FAIL | +0.201891 | FAIL |
| Overall NLL, equal-service | ≤ 0 | +0.186821 | FAIL | +0.078677 | FAIL |
| Overall Brier, row | ≤ 0 | +0.048028 | FAIL | +0.001526 | FAIL |
| Overall Brier, equal-service | ≤ 0 | +0.046078 | FAIL | -0.007040 | PASS |

Current passes **2/8** and history4 **3/8**. The joint result is **FAIL, 5/16
passed**. Improved change accuracy does not override the failed retention or
proper-score requirements. See the [machine-generated report](../output/dialogue-qwen-lexical-ablation-v1/report-01/report.md)
for the original fractional differences.

## Repairs, harms and the retention cancellation

A repair changes an originally wrong decision to correct; a harm changes an
originally correct decision to wrong. Counts include every paired row, not a
selected set of examples. The five displayed strata overlap: changed and
retained partition all rows; the two retention subtypes partition retained rows.

| Context | Stratum | Rows | Repairs | Harms | Net correct change |
|---|---|---:|---:|---:|---:|
| current | All | 7,819 | 332 | 498 | -166 |
| current | Changed | 578 | 17 | 6 | +11 |
| current | Retained | 7,241 | 315 | 492 | -177 |
| current | Unmentioned retention | 4,032 | 96 | 312 | -216 |
| current | Assigned retention | 3,209 | 219 | 180 | +39 |
| history4 | All | 7,819 | 350 | 338 | +12 |
| history4 | Changed | 578 | 22 | 9 | +13 |
| history4 | Retained | 7,241 | 328 | 329 | -1 |
| history4 | Unmentioned retention | 4,032 | 99 | 242 | -143 |
| history4 | Assigned retention | 3,209 | 229 | 87 | +142 |

Current unmentioned-retention errors rise **461 to 677**, while assigned-retention
errors fall **799 to 760**. History4 unmentioned errors rise **780 to 923**,
while assigned errors fall **789 to 647**. Thus its nearly unchanged aggregate
retention score hides substantial, opposing effects. Removing the bundle helps
some existing assignments while creating more false updates from NOT_MENTIONED.

## Every represented service

The table shows percentage-point changes, again no-flags minus original.
Positive changed accuracy is favorable; positive retained error is unfavorable.
The summary contains the full original/new metrics and paired counts for all six
services, with no service selected as an alternative winner.

| Service | Changed / retained rows | Current changed accuracy | Current retained error | History4 changed accuracy | History4 retained error |
|---|---:|---:|---:|---:|---:|
| Events_1 | 172 / 1,302 | +1.1628 | +23.2719 | +1.1628 | +10.9831 |
| Homes_1 | 117 / 2,379 | +1.7094 | -0.1261 | +3.4188 | -1.2190 |
| Hotels_1 | 109 / 2,123 | +2.7523 | -4.7574 | +6.4220 | -2.9675 |
| Music_2 | 111 / 352 | +1.8018 | -2.5568 | 0.0000 | 0.0000 |
| Services_1 | 10 / 617 | +20.0000 | -4.3760 | +40.0000 | -8.2658 |
| Services_3 | 59 / 468 | 0.0000 | +2.9915 | -6.7797 | +0.2137 |

Events_1 retention deteriorates sharply in both arms. The large Services_1
change-rate gains represent only **two and four additional correct changes out
of ten**, illustrating why equal-service and pooled-row results need both to be
shown. These are descriptive service differences, not independent replications
or evidence of statistical significance.

## Rare targets and false positives

Each cell below is **original → no-flags**; fractions retain the full denominator.
False-positive denominators include only non-target rows whose candidate set
supports the relevant type.

| Measure | Current | History4 |
|---|---:|---:|
| TRUE recall, all 580 targets | 32/580 → 132/580 | 23/580 → 146/580 |
| TRUE recall, 29 changed targets | 2/29 → 5/29 | 3/29 → 9/29 |
| TRUE false positives | 15/2,039 → 66/2,039 | 92/2,039 → 178/2,039 |
| DONTCARE recall, all 15 targets | 5/15 → 6/15 | 0/15 → 0/15 |
| DONTCARE recall, five changed targets | 0/5 → 0/5 | 0/5 → 0/5 |
| DONTCARE false positives | 482/7,804 → 700/7,804 | 421/7,804 → 393/7,804 |

TRUE recall improves, but remains low and is accompanied by more supported
TRUE false positives. Neither condition correctly identifies any of the five
DONTCARE changes. There are **no FALSE targets or clears**; these absent outcomes
cannot support a general polarity or state-deletion claim. The earlier
[blinded review](dialogue-qwen-retention-results.md) also identified ambiguity
between persistent preference and selected option in a few error-conditioned
cases. This ablation keeps every official target unchanged and does not turn
those reviews into label corrections or population accuracy estimates.

## Paid work, verification and limits

The new preparation, pilot and full inference each completed once. Whole-phase
times include their authentication, loading, serialization and closing hashes.
The pilot's repeated requests are paid work, not quality replicates.

| Phase | Seconds | Recorded work |
|---|---:|---|
| New preparation | 18.2558 | Local retokenization of 7,444 requests; labels copied opaque; zero model calls |
| New pilot | 13.5761 | 28 model calls, 56 decisions; quality outputs discarded |
| New full inference | 2,479.6357 | 7,444 model calls, all 15,638 decisions, zero generated tokens or optimizer updates |
| Preparation + pilot + full inference | 2,511.4676 | Sum of those three separate phases |
| Saved-output report | 7.7417 | Zero model/tokenizer calls or checkpoint deserializations |
| Independent result audit | 3.4398 | Zero model/tokenizer calls or checkpoint deserializations |

The 4,171-second pilot projection admitted the single run under the fixed
7,200-second full-run cap. Peak full-run process RSS was **3,491,479,552 bytes
(3.252 GiB)**. The three-phase sum above excludes preflights, separate input
auditing, reporting and figure generation; it is not a total project cost.

Padded token slots fall from **13,771,479 to 9,681,995**, a **29.70%** reduction:
current **6,491,568 to 4,446,826**, history4 **7,279,911 to 5,235,169**. The prior
full run took **3,229.3961 seconds**, versus **2,479.6357** here. These sequential
runs are descriptive cost records, not an interleaved speed benchmark or a
shared-prefix speedup claim.

The main reporter verifies both complete artifact chains and replays the exact
public-input subtraction. The separate [paired-input audit](../output/dialogue-qwen-lexical-ablation-v1/preparation-audit-01/receipt.json)
checks preparation identity without model or tokenizer execution. The
[independent result audit](../output/dialogue-qwen-lexical-ablation-v1/result-audit-01/receipt.json)
agrees on **1,455 scalar checks, 84 primary metric cells and all 16 conditions**.
It independently reconstructs candidate choices, supported probabilities and
row/equal-service primary metrics using the pinned independent original auditor.
Rare/subtype, equal-dialogue and paired-secondary metrics, costs, exact prompt
subtraction, tokenization, full-vocabulary mass and actual inference inherit the
authenticated main report or execution witnesses; they are not all independently
replayed by that arithmetic audit. No extra inference was performed to write or
audit these results.

| Artifact | SHA-256 |
|---|---|
| [Frozen plan](../output/dialogue-qwen-lexical-ablation-v1/preparation-01/plan.json) | `89b90d4a7decddf35e2dfbfacd53ce15a61e455cb32be9a719092839367caa36` |
| [Full completion](../output/dialogue-qwen-lexical-ablation-v1/run-01/completed.json) | `165552f6e7ca391145d6beade603829e2330e7d66511670d567af122109f9045` |
| [Summary](../output/dialogue-qwen-lexical-ablation-v1/report-01/summary.json) | `3668414c4785d1d35bedb186b5a560550edb5612b97e8c8b669aae08934d583b` |
| [Report receipt](../output/dialogue-qwen-lexical-ablation-v1/report-01/receipt.json) | `37a947e727099b05df00c64458e1e9218cc4de02931845eaee15e1e03d84fb0d` |
| [Independent audit receipt](../output/dialogue-qwen-lexical-ablation-v1/result-audit-01/receipt.json) | `f0e8d66984fb0cb68d9190803d74fa2a547d1d8f4480f87a7cd52023d8edbec6` |

The fixed no-flags representation is not promoted by this result. Neither the
mixed behavioral effects nor shorter prompts establish a learned-memory,
connectome, calibration or architectural advantage.
