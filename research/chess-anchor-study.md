# Input-anchored recurrence in OpenJev chess

The previous model became increasingly confident while move quality fell at untrained recurrent depths. This pilot asks whether changing the recurrent skip input helps, and whether simply training at several depths explains any gain.

## Controlled design

Let `x` be the encoded board and `h0 = x`. The same two convolutions compute `F(h)` in both architectures:

- Residual: `next = ReLU(h + F(h))`.
- Input-anchored: `next = ReLU(x + F(h))`.

The change introduces no parameters or additional recurrent calls. It changes both the identity skip and the direct encoder-gradient paths; a gain would not by itself identify memory preservation or prove stable dynamics.

| Arm | Skip input | Training depths |
|---|---|---|
| residual_fixed | Previous hidden state | 4 |
| residual_mixed | Previous hidden state | Exactly balanced 2, 4, 6 |
| anchor_fixed | Original encoded board | 4 |
| anchor_mixed | Original encoded board | Exactly balanced 2, 4, 6 |

All arms start from identical fresh weights within each of seeds 17, 29 and 43. The unchanged encoder, candidate scorer and value head contain 33,185 parameters. Checkpoints also retain the unused 10,541-parameter auxiliary decoder for identical initialization and state-dictionary structure, making 43,726 stored parameters.

Each fit sees the same 32,768 original training positions for six epochs: 1,536 updates at batch size 128. Both depth regimes use exactly 6,144 recurrent iterations per fit. The mixed schedule shuffles 512 updates at each depth independently of the common minibatch ordering. The loss is legal-move cross entropy plus 0.5 bounded-value MSE. No auxiliary prediction, mate fine-tuning, gating, search or new optimizer is added.

## Evaluation

The protocol generates 2,048 fresh ordinary and 2,048 fresh shifted positions from fixed new game seeds. It excludes all prior v1/spatial inputs, recorded behavior and auxiliary successor targets, previously selected mate puzzles, the old public puzzle panel and all recorded arena positions, including color-mirrored equivalents. Source hashes and exact quotas are frozen before generation. The same generators and shift were studied previously, so these are fresh prospectively frozen positions, not broad independent confirmation.

Every final checkpoint is evaluated at depths 2, 4, 8 and 16. All fitting finishes before any fresh neural evaluation is exposed. Three starting-board warmups and 64 selected single-position CPU decisions measure latency for each configuration; warmup costs are separate. Wall time is measured rather than assumed equal from operation counts.

A fixed 128-position subset per split receives stronger Stockfish grading for the two mixed arms at depths 4 and 8, all three seeds. Each position has one unrestricted reference and one call per distinct chosen move, at 20,000 nodes. The ceiling is 3,328 calls and 66.56 million requested nodes. Signed finite-search score differences are retained.

Uncertainty intervals resample complete evaluation games and remain conditional on the three fitted seeds. They are pointwise, not simultaneous confidence across all comparisons.

## Continuation rules

The predetermined primary comparison is `anchor_mixed@4` versus `residual_mixed@4`. It must improve mean teacher agreement by at least two percentage points on both panels, leave no paired seed worse by over one point, and lower mean bounded stronger-engine score loss on both panels.

The separate additional-computation gate requires `anchor_mixed@8` to improve agreement over its own depth-four result by at least one point on both panels, beat `residual_mixed@8`, and avoid increased engine-score loss against either comparator. Depth 16 is a reported stress test, not an alternative winner selected afterward.

The other two arms and the architecture-by-training interaction are always reported. Passing these engineering rules would justify larger training and game comparisons; it would not establish Elo or a publishable novelty result.

## Relationship to prior work

[End-to-end Algorithm Synthesis with Recurrent Networks](https://arxiv.org/html/2202.05826v3) already studies recall and extra recurrent computation, including chess. Its recall architecture concatenates input features and its progressive objective uses a detached prefix. Our small skip-source ablation and balanced random-depth training do not reproduce those mechanisms exactly, and neither is claimed novel on its own.

Status: pre-run validation. The frozen plan and all results will accompany execution.
