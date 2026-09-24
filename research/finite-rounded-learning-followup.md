# Next question: separate transition structure from training exposure

The [closed equal-time comparison](finite-rounded-learning-results.md) fails
its continuation rule: 14/21 conditions pass. Rounded mean H4/H8 regret is
11.09% / 17.88% higher than original free, using original free as the denominator.
This outcome stays failed. The original controls and all seeds remain in the
report; the large average gains against matched free do not rescue it.

One concrete difference remains: under the same training-time allowance,
rounded accepted 2,583-2,866 joint updates, versus 3,207-3,523 for original
free. Prefix counts also differ. Parameterization, gradients and training
exposure all changed. Better observed KL concerns filtering with intervening
observations, whereas the regret criterion concerns blind forecasting. That
contrast alone does not diagnose a prediction-to-decision alignment defect.

The next falsifiable question is whether rounded transport improves blind
decisions at the same number of training updates. Keep all three arms,
initial-function checks, public histories, H1/H2 objectives and optimizer
resets. Freeze one common prefix-update count and one common joint-update
count before fresh scientific data or seeds are used. Pair the batch order
and charge complete runtime, checkpoint work and normalization. Retain every
fit and publish failures. All final checkpoints must precede development
generation. Use separate source qualification, registration and a saved-output
audit; do not resume or extend any fit in the closed comparison.

Retain the absolute SHORT/BLIND/OBSERVED criteria and paired H4/H8 comparisons
against both free controls. A positive scientific result would require
consistent paired improvements and at least 10% improvement in each horizon
mean against both controls. This would answer an equal-update question only.
It would not pass the failed equal-time rule or establish compute efficiency.
Actual timing remains part of the result, even if rounded takes longer.

This note does not set an execution count, register a run or claim a gain.
Choose the exposure using a separate engineering workload, then freeze the
new protocol before evaluation. A success would still need untouched
replication and a world without the built-in doubly stochastic prior before
any transfer claim. Neither this synthetic learner nor the supplied
[decision-model architecture posts](jev-architecture-source-review.md)
establish a new connectome architecture, calibrated probabilities or an ICLR
contribution.
