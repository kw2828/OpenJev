# Calibrating probabilities recovers card gameplay

**Completed: 3,584 fresh games. All five associative-memory families finish every game after calibration, but the overall continuation rule fails, 11/12.** The GRU gets slightly worse and breaches the preset no-regression threshold. The original architecture result remains FAIL, 1/6.

![All memory families: gameplay and probability quality](../output/card-calibration-v1/visualization-01/calibration.png)

[![First fixed deck under original, calibrated and hard probabilities](../output/card-calibration-v1/visualization-01/fixed-public-replay.gif)](../output/card-calibration-v1/visualization-01/fixed-public-replay.gif)

The replay is the first scheduled fresh deck with Kalman pair 0, chosen before fitting or evaluation. All three policies finish this game. Baseline return is 0.5962, temperature 0.6923, and hard 0.7115. The frames contain only actual public observations; this is not a selected best game.

[Complete report](../output/card-calibration-v1/report-01/report.md) · [Every fit and score](../output/card-calibration-v1/report-01/summary.json) · [Independent audit](../output/card-calibration-v1/independent-review-01/completed.json) · [Full evidence release](https://github.com/kw2828/OpenJev/releases/tag/research-card-calibration-v1)

## Gameplay with the same weights

| Memory | Baseline return | Temperature return | Hard return | Baseline finished | Temperature / hard finished |
| --- | ---: | ---: | ---: | ---: | ---: |
| Delta | 0.3360 | 0.6571 | 0.6597 | 67/192 | 192/192, 192/192 |
| Gated delta | 0.3823 | 0.6557 | 0.6582 | 82/192 | 192/192, 192/192 |
| Diagonal Kalman | 0.3928 | 0.6531 | 0.6535 | 89/192 | 192/192, 192/192 |
| Local innovation | 0.3909 | 0.6508 | 0.6544 | 86/192 | 192/192, 192/192 |
| Gain-matched diffuse | 0.3841 | 0.6497 | 0.6513 | 83/192 | 192/192, 192/192 |
| GRU512 | -0.9754 | -0.9871 | -0.9306 | 0/192 | 0/192, 0/192 |

All 15 associative-memory fits finish 64/64 decks with temperature and with hard decisions: **960/960 games for each condition**, compared with 407/960 under baseline. Including the three GRU fits, completion rises from **35.3% to 83.3%**. There are 64 shared deck draws, not thousands of independent layouts.

The exact public-memory reference and last-32-reveals reference both finish 64/64. Their returns are **0.6992 and 0.6797**, respectively, above every calibrated learned family. Equal completion does not imply equal efficiency. Both references run once per deck; we do not artificially replicate them across the three probability conditions.

The proposed local-innovation memory remains below ordinary delta and diagonal Kalman in return. These results do not establish its superiority, a connectome advantage, or an improved recurrent world-model architecture.

## Probability quality on the same histories

On all 1,152 new baseline trajectories, score each transformation against identical public labels and saved raw predictions. No alternative trajectory or fresh model call is used for this comparison. Scores average queried cards within a decision, decisions within an episode, episodes within a fit, and all 18 fits equally.

| Beliefs | NLL, lower is better | Brier, lower is better | Top-rank accuracy |
| --- | ---: | ---: | ---: |
| Baseline | 0.696981 | 0.221553 | 84.73% |
| Temperature | 0.444019 | 0.162069 | 84.73% |
| Hard | Infinite | 0.305380 | 84.73% |

Temperature lowers NLL by **36.29%** and Brier by **26.85%**. Top-rank accuracy stays identical, as positive temperature scaling preserves rank order. Hard decisions incur infinite NLL on 63,915 wrong position-time queries. We preserve those errors without smoothing or averaging them away. These repeated query counts are not independent observations.

## Why the overall rule fails

The temperature-minus-baseline mean native gain is **+0.228098** across all 18 fits, above the required +0.03. All three paired-fit group averages improve, and both probability-quality requirements pass. But GRU mean return falls from **-0.975361 to -0.987079**, a loss of **0.01171875**, greater than the allowed 0.01. Every condition was required. We retain the failed result without dropping the GRU, relaxing the threshold or substituting the hard control.

The stronger families reveal a useful diagnosis: changing confidence alone was sufficient to recover completion with their fixed memories. That supports confidence as a bottleneck in this controller. It does not make temperature scaling a new architecture. The method is established; see [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).

## Reproduction and limits

The [prospective study](card-calibration-study.md), [protocol](../evidence/card-calibration-v1/protocol.json), literal seeds and source snapshots were published in commit [`c464120`](https://github.com/kw2828/OpenJev/commit/c464120cbd4acfde1f3a21fdf36db3eec324aa30) before scalar fitting or fresh evaluation. All 18 final neural checkpoints are inherited unchanged.

For each checkpoint, one scalar inverse temperature was fitted to its 64 previous C-controller histories. Those earlier test histories are explicitly reused as **calibration training data** for this separate study. Their original outcomes remain unchanged. The new 64 deck seeds are disjoint from prior train, development, test and engineering streams. No test label selected a temperature or checkpoint.

Calibration took **32.52 seconds** with 1,224 bounded objective/derivative passes and no model or native calls. Evaluation took **211.68 seconds**, covering 336,784 native actions and 325,994 learned prediction/write pairs. Both ran once without retry. Timings include instrumented processing and serialization on a shared host; different trajectory lengths and probability transformations do not establish an intrinsic speedup.

The saved-output report audited scalar objectives, public transitions, probability transformations, native choices and all criteria. A separately implemented NumPy audit independently verified native-row arithmetic, public deck reconstruction, fresh-prefix scores and all 12 checks. Its validation passed while the scientific rule failed. The implementation passed 121 focused synthetic tests before execution; those tests are engineering checks, not empirical evidence.

The release contains all new scalar fits, 3,584 trajectories, reports, figures, audit and prospective source snapshots. Full lineage also requires the [original training release](https://github.com/kw2828/OpenJev/releases/tag/research-card-memory-pilot-v1) for checkpoints and the [controller release](https://github.com/kw2828/OpenJev/releases/tag/research-card-controllers-v1) for calibration-training histories.

This static card task now has a completion ceiling under ordinary methods. Further architecture work needs a demonstrated state-estimation or memory problem that a cheap table, finite observation window, or classical filter does not explain. This experiment alone does not justify a new architecture claim or ICLR readiness.

A [candidate qualification note](../output/card-calibration-v1/next-direction.md) examines action-conditioned noisy-state prediction with short-history and classical-filter controls. It is not a launched study or evidence of a new architectural need.
