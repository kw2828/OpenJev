# Next hypothesis: control the largest action-error contrast

**Prospective design only. No new training or evaluation is registered here.**
The [saved prediction diagnostic](finite-decision-error-results.md) leaves the
parent continuation failure unchanged. Its purpose is to choose a useful next
question, not turn the failed 14/15 rule into a pass.

The losing seed's H8 policies disagree on nine of 486 cases. One disagreement
in the predeclared true-margin-at-least-0.1 group contributes +0.00047412 to
the mean regret difference, while the other margin groups together favor the
candidate. The underlying case has regret 0.23042372. Therefore, focusing only
on tiny true gaps would miss the error that reverses this comparison. Costs
are already centered by construction; subtracting an action-independent offset
provides no new mechanism.

## Test an optimization explanation before attributing gains to structure

For centered four-action error e = predicted costs minus true costs, consider
the per-case, per-step loss

\[
L_{\mathrm{range}} = \frac{(\max_a e_a-\min_a e_a)^2}{4}.
\]

Decision regret is bounded above by the error range. For a centered vector of
four errors, ordinary mean squared error satisfies

\[
\operatorname{MSE}(e) \leq L_{\mathrm{range}} \leq 2\operatorname{MSE}(e).
\]

These inequalities motivate a scale control. They do not guarantee better
optimization, finite-sample performance or extrapolation to H8.

A separate comparison could cross both random-head architectures, rounded and
initially matched free, with three blind-cost losses: ordinary MSE, twice MSE,
and error-range loss. Keep the observed-cost, survival, event and prefix losses
unchanged, together with H1/H2 supervision, batch order and update counts. The
twice-MSE arm tests a simple weight explanation; it does not match gradients.
No coefficient search or retraining of completed checkpoints is proposed.

Before execution, qualify the four-action inequalities and gradients on
fabricated witnesses, freeze every choice and register fresh data and seeds.
Use independent training/evaluation cohorts for each paired block so a single
shared dataset cannot be mistaken for replication across populations. Preserve
all six arms, computation costs and original/noisier observations. Define the
comparison and prediction-quality requirements before any new evaluation.
The completed diagnostic cases remain development evidence and cannot serve
as the new confirmation set.

If changing the loss helps both architectures, describe that as an optimization
result. Any architecture claim still needs stronger controls, a less favorable
transition family, a second environment and evidence of novelty. The broad
research goal remains open.

## Prior work and reasons to keep claims narrow

[Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005) already connects
prediction training to downstream decisions. [Liu and Grigas (2021)](https://proceedings.neurips.cc/paper/2021/hash/b943325cc7b7422d2871b345bf9b067f-Abstract.html)
analyze calibration bounds for SPO+ under stated assumptions. An error-range
control is not an established new architecture or a calibrated-probability
method merely because it bounds regret.

The repository's [earlier decision-focused study](otto-action-focused-results.md)
and [protected-readout study](otto-protected-readout-results.md) were negative.
This proposal keeps that evidence visible and isolates a different loss
geometry with an explicit scale control. It remains a hypothesis.
