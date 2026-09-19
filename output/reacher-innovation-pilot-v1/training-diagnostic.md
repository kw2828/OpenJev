# Post hoc training diagnostic: innovation pilot

The residual-moment objective was active in all 12 fits. Every one of the 1,920 recorded updates had a finite, positive variance-head gradient; none of the variance-head or backbone gradients was clipped. All variance-head parameters changed, their final Adam counters were 160, and the reported residual-moment score fell from the first to the eighth epoch in every fit. This excludes an entirely omitted or inactive auxiliary objective. It does not establish convergence, explain the failed gate causally, or show that longer training would help.

The original continuation result remains **21/29, failed**. This diagnostic adds no criterion and makes no native-control or superiority claim.

## Training progress

These are equal averages across all three paired fits and all 20 equal-sized minibatches within each epoch. They describe training passes with changing parameters, not fixed-checkpoint evaluations. Lower residual-moment score is better under the recorded objective.

| Variant | Moment score, epoch 1 -> 8 | Mean predicted second moment, epoch 1 -> 8 | Observed angle MSE, epoch 1 -> 8 | Variance-head gradient norm, epoch 1 -> 8 |
|---|---:|---:|---:|---:|
| constant | 1.758836 -> -0.727420 | 1.922862 -> 0.773355 | 0.436453 -> 0.037868 | 0.051346 -> 0.126531 |
| age | 1.758766 -> -0.727740 | 1.922820 -> 0.773296 | 0.436419 -> 0.037870 | 0.051350 -> 0.126545 |
| raw | 1.755664 -> -0.743230 | 1.920821 -> 0.770048 | 0.435066 -> 0.037999 | 0.051655 -> 0.127267 |
| normalized | 1.756676 -> -0.736088 | 1.921497 -> 0.772159 | 0.435488 -> 0.037973 | 0.051560 -> 0.126981 |

For normalized gating, predicted scale decreases substantially, while squared prediction error decreases faster. The epoch-8 training ratio is approximately 20.3, consistent with a residual-size overprediction question rather than a zero-gradient or clipping explanation. This is a marginal ratio across training targets; it is not conditional calibration, uncertainty coverage, or evidence of a particular optimization cause. The separate saved-development-prediction analysis should be used for checkpoint-level scale ratios.

The variance head contains 260 parameters. All 260 changed in each fit; its relative L2 change ranges from 2.273 to 2.429 of its initial L2 norm across the 12 fits. This is a parameter-update description, not a measure of predictive quality. The final first and second Adam moments are finite, and all parameter-group counters agree with the 160-update fit receipts.

The JSON retains every fit, every epoch, all recorded loss components and mask counts, both gradient-group statistics and clipping frequencies, all named parameter-group changes, and final Adam-moment summaries. No fit or epoch was selected as a winner.

All fit wall times sum to 91.566706 seconds. Recorded update intervals sum to 90.724633 seconds and are nested within those fit times; they omit some logging and final artifact work. These are descriptive shared-host times, not an isolated speed comparison.

## Authentication and limits

- Plan SHA256: `561eb5a73f30ce81453941a6ade73cf15f0332af26cfb6ef49a7a17a35eff3de`.
- Completed execution SHA256: `1a511ebecad5cc6e363719555b97536671f1111948ae0aa0c7070a21ff7a9e69`.
- Independent audit summary SHA256: `350fe70a3b6c02f38e5d11595ecaaee4d22f836d3aabf60a5f7f39d119f44575`.
- Diagnostic JSON SHA256: `b081d83c90a376f704db55dd071ac9e643167223064f50e3f5e66b68f90829db`.
- All 87 read files were checked against the external roots or their authenticated member maps, then rehashed at exit. Initial, final, and checkpoint canonical tensor identities were checked against fit receipts.
- Coverage: 12 fits, eight epochs each, 160 updates each, 1,920 updates total. Three paired initializations and all four gates are retained.
- Work: saved weights, final optimizer state, and logs only. Zero forward passes, backward passes, optimizer steps, RNG draws, native calls, or accesses to the original corpus/development prediction tensors.
- Recorded gradient and optimizer evidence was authenticated, not independently regenerated. A changing mean predictor changes the residual target, so improved moment loss does not certify a well-calibrated head or adequate optimization.
- The scientific gate and all frozen sources are unchanged. This is explicitly post hoc and remains separate from the prespecified results.
