# Next decision after the innovation pilot

**Prioritize an explicit persistent-dynamics adaptation task. Do not spend another full fit merely correcting residual scale.** Keep the completed pilot negative at 21/29. Its scale mismatch is worth documenting, but marginal overprediction does not establish that scale correction would improve the gate or control.

The next question should be: can a small, causally updated context identify persistent dynamics changes and improve decisions beyond ordinary recurrence and conventional state/parameter estimation?

## What the saved evidence establishes

All 1,920 variance-head updates had finite, positive gradients and no clipping; all 260 variance parameters changed in every fit. For normalized gating, epoch-mean predicted moment decreased from 1.92150 to 0.77216 while angle MSE decreased from 0.43549 to 0.03797. These changing-model training averages demonstrate an active objective, not convergence or conditional calibration.

Gate parameter changes follow the source masks: one of three coefficients changes for constant gating, two for age gating, and all three for raw/normalized gating. All 64 innovation-projection parameters change in every fit. Every transition parameter changes, including its 3,072 input coefficients connecting the 16-dimensional context to the three GRU gates. Normalized gate L2 changes are 0.10556, 0.13526, and 0.13274 across the three pairs.

This proves parameter updates along the permitted pathways. It does **not** establish context activation magnitude, gate saturation, effective write size, context contribution to predictions, or counterfactual benefit. Those activation/intervention quantities were not saved. All four arms already have persistent fast state and slow context; constant gating is not a no-context baseline. Shared improvements can therefore come from the fast GRU. Rescaling the innovation statistic also changes a learned gate input, so a marginal scale error alone does not prove that input is useless.

## If pursuing scale correction

Allow at most a bounded fixed-backbone follow-up, not another architecture campaign. The strongest simple statistical control is a per-coordinate, measurement-age-conditioned empirical squared-residual table, fitted only on training residuals with predeclared shrinkage toward a global mean. Include global scalar rescaling and the original head. Recomputing residuals or predictions still costs inference; this is not free.

A falsifiable continuation rule would require the learned scaling approach to outperform that table, age gating, and raw gating on newly sealed episodes: at least 3% lower post-reacquisition error in both gap panels, all three pairs nonworse, retaining the existing angle/reward guards. Better moment score or a ratio nearer one alone is insufficient. A context-disabled/frozen-gate intervention could diagnose utilization first, but is an exposed, potentially off-distribution ablation and cannot rescue the old gate. Stop this path if simple scaling explains the result.

## Preferred single experiment

Vary **one hidden actuator-gain parameter**, persistent within an episode, with one predeclared unannounced change. Keep action noise and sensing schedules fixed across paired methods. Use an identical brief probing sequence so gain is identifiable separately from velocity/noise; split gain values, change times, and seeds before fitting. True gain/native velocity remain audit-only. Do not simultaneously vary friction, mass, topology, and loss.

Compare ordinary GRU, explicit two-observation/history state, constant-update slow context, and innovation-gated context at matched training data and measured compute. The strongest classical control here is nominal-physics joint position/velocity/gain estimation with an augmented-state UKF, followed by the same planner; a known-gain observer is an oracle reference, not a deployable competitor. Joint state/parameter filtering is established prior art, including [dual EKF methods](https://onlinelibrary.wiley.com/doi/pdf/10.1002/0471221546.ch5).

Before launching, freeze a proposed continuation rule: at least 5% lower post-change native cost than every deployable comparator at the declared compute budget, all three pairs nonworse, and at most 2% regression without a change. Report recovery time and parameter estimation separately. A win would support this specific adaptation mechanism, not biological novelty. A failure ends this mechanism's campaign rather than triggering successive scale/loss tweaks.

## Provenance

Read-only proposal; no fitting, inference, native calls, or RNG. Training diagnostic SHA256: `b081d83c90a376f704db55dd071ac9e643167223064f50e3f5e66b68f90829db`. Architecture SHA256: `1735cf7445908be08aa3e9dd75fe69133621745826b3c130a059beb883ada80a`, checked against plan `561eb5a73f30ce81453941a6ade73cf15f0332af26cfb6ef49a7a17a35eff3de`. All proposed thresholds are prospective recommendations, not an authorized or frozen protocol.
