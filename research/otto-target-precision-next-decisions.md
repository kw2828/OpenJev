# Decisions after the target-precision comparison

22 September 2026. **Proposed next experiment, not frozen or admitted.** This interpretation uses the completed worker's saved JSON. The [independent audit](../output/otto-target-precision-v1/audit-01/receipt.json) completed with agreement on all 33 conditions, including the 7/33 failure (receipt SHA-256 `549eb00f8a04904f22bb9ff351c35c4e5f965b88a6f1944d0186886ac693baa7`; original session 55141, terminal 687e4a). No labels, predictions, fitting or searches were generated for this note. The [frozen protocol](otto-target-precision-protocol.md) and its result remain unchanged.

The worker reports **FAIL: 7/33 conditions passed**, comprising 3/3 analytic positive controls, **0/18 R64 competence checks**, and 4/12 relative checks. All analytic searches succeeded. The precision intervention helped one setting and hurt two; it did not produce competent control.

| Known sensing length | R16 weighted success | R64 weighted success | R16 capped moves | R64 capped moves |
| --- | ---: | ---: | ---: | ---: |
| 3 | 21.43% | 19.78% | 1,721.91 | 1,841.77 |
| 4 | 12.79% | 17.89% | 1,911.09 | 1,809.07 |
| 5 | 15.11% | 13.06% | 1,863.42 | 1,903.74 |

These are initial-hit-mixture-weighted means across all three fits, with censoring at 2,188 moves. The 504 episodes share 72 environmental cases. Positive paired block gains occurred in only 2/8, 3/8 and 1/8 blocks. This small paired comparison is not a significance test. [Saved summary](../output/otto-target-precision-v1/run-01/summary.json)

## What the result distinguishes

Appending 48 replicas bought 96,912 additional continuations and 2,708,844 moves, all found, for **1,028.31 seconds**. It was insufficient to rescue this fixed learner. That does not establish that target noise is absent: R16 is nested inside R64, finite-sample rankings remain uncertain, and these costs describe one forced action followed by the analytic teacher, not repeated decisions by the learned policy. Do not automatically double the replicas again. [Collection](../output/otto-target-precision-v1/run-01/collection.json)

The R64 centered target RMS is 1.13376, versus the preserved R16 scale of 1.91311. Its lower training loss therefore cannot be read as better action learning across arms. On that common scale, approximate zero-output R64 target energy is **0.351205**. Final online epoch losses retain **79.1-82.2%** of that energy. They are pre-update minibatch losses, not a final checkpoint's full-cohort error or action-ranking assessment. [Target metadata](../output/otto-target-precision-v1/run-01/training-data.json)

| Fit seed | Mean loss, epochs 1-20 | Mean loss, epochs 61-80 | Epoch 80 | Reduction, epoch 60 to 80 |
| --- | ---: | ---: | ---: | ---: |
| 20101 | 0.342510 | 0.289522 | 0.283690 | 4.37% |
| 20102 | 0.345251 | 0.294517 | 0.288864 | 4.20% |
| 20103 | 0.337361 | 0.283137 | 0.277881 | 4.34% |

All seeds were still improving. The last twenty epochs' fitted slopes are about -0.00061 to -0.00065 loss per epoch, slower than -0.00147 to -0.00171 early. This is evidence against declaring a flat plateau, not a forecast of eventual fit. Representation, conflicting finite-sample targets, the regression objective and state coverage remain competing explanations. Earlier scalar capacity results used different supervision and cannot settle this action-cost problem. [Saved curves](../output/otto-target-precision-v1/run-01/fits.jsonl)

## One bounded next test: optimization allocation

Compare **80 versus 320 epochs** on the identical R64 cache. Keep all 558 anchors, features, masks, episode weights, original global scale, ordinary 32/16 Tanh head, eight D4 views, loss, Adam settings and clipping unchanged. Use three fresh paired initializations with identical first eighty epoch orders and updates. Freeze the longer schedule in advance, save fixed final checkpoints, and retain every fit. This changes update allocation alone: 400 versus 1,600 updates per fit, **6,000 total**. The present six fits cost 7.92 seconds; this motivates a modest extension, not an unqualified runtime guarantee or 800-epoch escalation.

Record final full-cohort loss and eligible-action regret against the saved R64 mean costs for every row and fit. These expose whether improved regression reaches the decision ordering; they are TRAIN diagnostics, not true teacher regret or an efficacy gate. No confidence filtering, replacement labels, learning-rate search or best-epoch selection.

Use the same **504 fresh paired full-horizon episodes**, three kernels and analytic control, with newly reserved seeds and complete costs. Preserve the 33-condition structure: analytic success at least 95%; every longer-trained fit at least 95% success and at most 105% of analytic moves; family success no lower than the short control, at least 5% fewer moves, positive gains in at least six blocks, and controller cost at most 105%. Report the short control's absolute outcomes too. Do not lower competence thresholds or expand autonomous allocation to obtain a pass. Separately freeze a feasible runtime cap no larger than the current study's; charge the extra fitting.

If training fit improves but competence fails, reject longer optimization as sufficient and investigate coverage or the decision objective. If fitting barely improves, examine representational restrictions and inconsistent targets before proposing another backbone. Neither outcome rules out memory or establishes architecture novelty. There is no automatic additional epoch or replica round.
