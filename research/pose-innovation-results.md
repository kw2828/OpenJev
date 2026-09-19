# Ordered prediction errors did not earn a recurrent updater

**The fixed recipe failed its training-parent screen: 223/368 checks passed.**
The recurrent correction head improves the frozen GRU's position/rotation
RMSE by 4.49%/4.13%. A simple linear model of summary features improves them
by 9.28%/7.60% and beats the recurrent head on both endpoints in all nine
paired fold/seed comparisons. Shuffling only past errors changes the recurrent
head's pooled errors by less than 1%.

[Protocol](pose-innovation-protocol.md) · [Audited numerical results](../output/pose-innovation-v1/report-01/summary.json) · [Audit receipt](../output/pose-innovation-v1/report-01/receipt.json) · [Related work](pose-innovation-related-work.md)

![All nine methods, nine paired fold/seed results and the failed training-parent screen](../output/pose-innovation-v1/visualization-01/physical-errors.png)

## The question and comparison

Earlier experiments already tested residual correction, learned ridge
adaptation, and past-error blending. This experiment asks a narrower question:
does the temporal organization of explicit prediction errors add enough value
to justify learning a recurrent update rule?

Each frozen backbone sees 32 observed poses and forecasts 25 future poses.
During context assimilation, we record every one-step prediction before
feeding its successor into the model. The correction head sees public
observation/action features and those 31 completed errors. It produces a
separate root-frame correction at each future lead. Corrections do not feed
back into the backbone, so this is a signal diagnostic rather than a new
dynamics model.

We compare zero correction, a shrunk past-error bias, two linear ridge
models, a summary MLP, and four GRU conditions. The GRU conditions use ordered
history, whole-token shuffled history, no error channels, or shuffled error
channels with public history kept in order. The last control distinguishes
error timing/alignment from ordinary public-history order. It does not
separate error sequence order from error/action pairing.

## All results

RMSE pools squared errors before taking the square root. Values below are
held-out correction-test results. Each method has 864 window predictions
across three seeds, three folds and twelve distinct test parents.

| Method | Position RMSE, m | Rotation RMSE, rad |
|---|---:|---:|
| Frozen GRU | 0.009593 | 0.015142 |
| Shrinkage bias | 0.017354 | 0.016818 |
| Summary ridge | **0.008703** | **0.013991** |
| Ordered ridge | 0.008921 | 0.014542 |
| Summary MLP | 0.008965 | 0.014606 |
| Ordered GRU, primary | 0.009162 | 0.014517 |
| Whole-token shuffle | 0.009254 | 0.014920 |
| No-error GRU | 0.009269 | 0.014898 |
| Error-only shuffle | 0.009194 | 0.014656 |

The primary has **5.28% more position error and 3.76% more rotation error**
than summary ridge. Summary ridge wins all nine paired comparisons on each
endpoint, and 12/12 position and 11/12 rotation parent comparisons after
pooling seeds. These are consistent descriptive differences, not independent
significance tests.

Against error-only shuffling, the primary improves position by **0.34%** and
rotation by **0.95%**. It wins only 7/9 and 5/9 paired comparisons respectively.
Against the no-error GRU, gains are **1.15%/2.56%**. This leaves some possible
value in explicit errors, but does not establish a useful temporal mechanism
under the tested settings.

Training errors do not rescue the hypothesis. Ordered ridge has lower training
error than summary ridge, but worse held-out error on both endpoints. The
primary's training RMSE is 0.008065 m / 0.011148 rad, versus summary ridge's
0.007307 m / 0.011352 rad. The [complete report](../output/pose-innovation-v1/report-01/README.md)
retains training and test metrics for every method. This fixed small-head,
40-epoch result does not prove that every possible error-history architecture
must fail.

## Exclusion, costs and verification

For each fold, both correction-training and correction-test parents were
excluded from that backbone's training and fold normalization. Six excluded
parents train the head; four evaluate it. The original prepared training
archive has 24 overlapping windows per parent. Each test parent is excluded
from its own prediction pipeline, not from all other backbones. This is
held-out development within previously used training data, not a fresh
external benchmark. The dev, plain and zigzag panels were not evaluated.

All **45 neural fits, 9,000 optimizer updates and 81 evaluation rows** completed
once. Initial/final weights, losses, batch orders and every method's predictions
were retained. The nine fresh backbone cache comparisons are bitwise equal to
their previous forecasts. No backbone weights or native environments changed.

Execution took **18.006 seconds** on an Apple M5 Max using one CPU thread;
the independent saved-output audit took **1.531 seconds**. Cached correction
and application median latency was **0.210 ms** for the primary and
**0.090 ms** for the summary MLP. These exclude the backbone and preprocessing
and must not be presented as full forecast or robot-control latency. Each
neural method has 108 timed samples after warmups. No overall speed advantage
is claimed.

- [102-test preflight](../output/pose-innovation-v1/preflight-01/receipt.json).
- [Frozen protocol and source snapshots](../output/pose-innovation-v1/experiment-01/protocol.json).
- [Execution log](../output/pose-innovation-v1/experiment-01/execution.log) and [380-file completion seal](../output/pose-innovation-v1/experiment-01/run-01/completed.json).
- [Saved physical errors](../output/pose-innovation-v1/report-01/window-errors.npz) and [chart receipt](../output/pose-innovation-v1/visualization-01/receipt.json).

The independent audit reconstructs signed errors, token arithmetic, root-frame
corrections, physical metrics, linear-control normal equations and all 368
screening comparisons. It checks shared initial weights, source identities,
parent exclusions and complete artifacts. Neural learning and pre-assimilation
prediction provenance remain supported by source and tests, rather than an
independent neural rerun. Raw third-party inputs and full forecast caches stay
local under the unresolved upstream data license.

## Decision

Close this direct error-history recipe without reopening the exposed external
panels. Summary ridge is the strongest control in this screen, not a claimed
new architecture. A next mechanism must distinguish itself from these simple
controls and the earlier failed correction studies. Forecast improvements
would still need a matched control experiment with an explicit sensing/update
schedule; the [related-work note](pose-innovation-related-work.md) explains why.
No connectome advantage, closed-loop robot gain or ICLR-ready result is established.
