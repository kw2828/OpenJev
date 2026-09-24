# Frozen-readout diagnostic stopped before evaluation

**Three of nine readout solves pass the numerical certificate. The study stops before creating its fresh development data.** This run supplies no new decision-performance comparison and does not advance a model.

![All nine reported numerical gaps against the fixed threshold](finite-convex-readout-results/certificate-gaps.svg)

[All solve records and timing](finite-convex-readout-results/report.md) ·
[Frozen protocol](finite-convex-readout-protocol.md) ·
[Complete preserved evidence](https://github.com/kw2828/OpenJev/releases/tag/finite-convex-readout-v1)

## What ran

The diagnostic loads all nine final checkpoints from the [factorized recurrent study](finite-factorized-dynamics-results.md). It freezes their recurrent dynamics and solves only the action-cost readout on the original training histories and H1/H2 targets. Each parent family retains all three seeds. No new training labels, model-weight updates or seed selection occur.

All nine SLSQP calls return success and reduce their training objective. The registered requirement additionally checks a direct-residual numerical Frank-Wolfe gap at most 1e-8. Six reported gaps are between 1.1242e-7 and 4.6953e-7, so those solves fail. Optimizer success and the required certificate are different conditions.

| Parent family | Passing certificates | Failing seeds |
| --- | ---: | --- |
| Factorized | 1/3 | 426261002, 426261003 |
| Initially matched free | 2/3 | 426261003 |
| Dense free | 0/3 | 426261001, 426261002, 426261003 |

The first factorized solve reports zero gap. It is retained with the others, not promoted separately. A small training-loss reduction is not evidence of better held-out decisions.

## Terminal state and evidence

Qualification passes **73 tests** and lint. The original qualification supervisor closes successfully in 3.017 seconds. The scientific supervisor closes with exit code 1 after 1.754 seconds, without a timeout. All nine solve records, nine TRAIN-state files, original TRAIN bytes, configuration and failure record remain intact: 22 producer files.

Counters record nine checkpoint loads, nine solves, 144 TRAIN forwards and **zero fresh DEV generations, evaluation forwards, checkpoint writes, model-weight optimizer updates, teacher calls or external-model calls**. The nine model-state hashes match before and after their readout solve. The gate prevents the planned fresh pool from being created. There is no success barrier, prediction file, completed summary or scientific audit.

The publication check authenticates source hashes, upstream provenance, original process receipts, saved-file hashes and the JSON joins. It does **not** decode arrays, rerun models, or independently recompute the reported numerical certificates. The figure shows producer-reported gaps. Its 1e-8 line is the original criterion, not a revised threshold.

Registration SHA256: `129af3b025548eda6e475cb2d69bd5a7f3bfb325b085a56dff3ee71af798bab4`.

## What this tells us

The solver recipe did not reliably reach the required accuracy. We cannot yet conclude whether readout fitting explains the earlier recurrent failures. Nothing here changes the previous failed all-seed criteria or supports a connectome, calibration, native-transfer or novel-architecture claim.

The closed-simplex readout also permits boundary values that a finite softmax approaches only in a limit. Even a completed comparison would need to disclose that model-class difference. [Next: qualify certificate-driven numerical termination](finite-convex-readout-next.md).
