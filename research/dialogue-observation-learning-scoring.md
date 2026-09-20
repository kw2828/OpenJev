# Observation learning: prospective scoring contract

This specifies scoring before any of the twelve scientific fits. It supplements
the [prepared factorial](dialogue-observation-learning-preparation.md), without
changing its cohorts, training recipe or seven practical conditions. Execution
allocation and the implementation are still pending; this document alone does
not authorize an unbounded run or establish a result.

## Outputs and denominators

Save final-epoch predictions for every one of the 62,329 DEV endpoints in the
prepared source order. Store the model's normalized float32 log probabilities,
with twelve candidate columns and negative infinity only in padded columns.
Preserve each endpoint's dialogue, time, question and supported candidate IDs.
Do not apply a temperature, probability floor, second softmax or renormalization.
Choose the first supported maximum in the supplied candidate order.

For NLL, promote the saved log probability of the target to float64 and negate
it directly. For multiclass Brier, exponentiate the promoted log probabilities,
sum squared error over all candidates, then average across endpoints. This
avoids manufacturing infinite NLL by first rounding very small probabilities
to float32 zero. Validate finite supported log probabilities, exact padded
support and total probability mass within the unchanged 2e-6 tolerance.
Any nonfinite supported log probability is a technical failure. Preserve its
artifact and failure receipt; do not drop the endpoint, substitute a floor,
or issue a passing result from the remaining fits.

Report all, seen-service and unseen-service panels. The three-stratum macro
accuracy is the unweighted mean of accuracy on unmentioned retention, assigned
retention, and changed endpoints. Changed pools first assignment, revision and
clear; it does not average those three bins. Assigned-retention error is
incorrect predictions divided by all assigned-retention endpoints in its panel.
NLL and Brier guards use endpoint-micro means within the panel. Average each
arm's metrics equally across its three seeds. Accuracy comparisons and practical
margins use exact count ratios; probabilistic comparisons use float64 without
an epsilon allowance. Empty required support leaves a condition unevaluable
and therefore unable to pass.

## Seven primary conditions

Compare trainable_numbers against frozen_numbers, on final fits only:

1. Mean unseen three-stratum macro accuracy improves by at least 0.01.
2. Unseen macro accuracy improves strictly in at least two paired seeds.
3. Mean unseen endpoint-micro NLL does not increase.
4. Mean unseen endpoint-micro Brier does not increase.
5. Mean seen three-stratum macro accuracy decreases by at most 0.01.
6. Mean seen assigned-retention error increases by at most 0.005.
7. Mean unseen assigned-retention error increases by at most 0.005.

All seven must hold. Complete valid execution of all twelve fits is a separate
mandatory technical prerequisite, not an eighth scientific condition. Do not
select a seed, epoch, checkpoint, arm, threshold or subset after seeing results.

## Mandatory descriptive results

Show every arm and seed, both trainable-minus-frozen contrasts, both
numbers-minus-original contrasts, and the paired difference of those encoder
contrasts as the factorial interaction. Keep the sign convention explicit:
positive accuracy is better, while positive NLL, Brier or retention error is
worse. The interaction measures departure from additive effects in this
experiment; it is not a novel-mechanism attribution or significance test.

Report all five transition bins and each service with support counts. Show
TRUE, FALSE and DONTCARE target support and accuracy, plus predicted support,
true positives and false positives with their denominators. The false-positive
denominator is all non-target endpoints whose supplied candidate set supports
the value. Keep absent groups as zero support with undefined rates. The
prepared catalog has no Boolean flag: designate a question Boolean only when
the set of its ordinary candidate values, stripped and casefolded, is exactly
`{"true", "false"}`. Classify those candidates by exact value, and DONTCARE by
its reserved ID. This evaluator rule is independent of number normalization.
Report both original
and number-normalized deterministic literal-register references as accuracy
controls only; these rules supply no probabilistic forecasts.

Include complete training, evaluation and qualification compute separately.
Official DEV is exposed development data and official TEST remains sealed.
Passing these practical margins would support a stronger observation baseline
and untouched confirmation. It would not establish calibration, connectome or
world-model advantage, or publication readiness.
