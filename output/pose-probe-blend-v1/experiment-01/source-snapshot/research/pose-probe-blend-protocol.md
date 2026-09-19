# Can completed past errors select a useful continuous blend?

The previous goal turn made progress: a saved-output capacity calculation
ruled out hard routing of the current private forecasts and showed room for
continuous mixtures. It did not establish that useful coefficients can be
estimated from available observations. This experiment tests that missing step.

## Fixed mechanism and controls

Reuse the original three pairs of fast recency-and-robustness predictors and
body-frame GRUs. Do not train new weights or modify any frozen parent source.
Keep the original full 32-pose context, 31 completed action blocks and 25-step
future forecast. Full expert trajectories remain private and do not receive
blended-prediction feedback.

At the current root, pose index 31, construct one retrospective probe:

1. The fast predictor observes poses 0..26 and action blocks 0..25. Its support
   consists of transitions indexed 1..25, with ages 24..0, half-life 5, three
   Huber reweighting iterations, delta 1.5 and ridge precision 1. All features
   and priors come from the existing trained predictor.
2. The GRU assimilates actual poses and completed actions through index 26.
   Clone that state for its probe. Preserve the original CV1 motion convention.
3. Roll both experts for five steps using the recorded applied actions 26..30.
   Only after producing those predictions, compare them with observed poses
   27..31. Continue the separate actual GRU branch through index 31.
4. Choose position and rotation coefficients separately, then hold them fixed
   for the full 25-step forecast from index 31. The full fast predictor uses
   the original 30-support fit. The GRU uses its actual state at index 31.

This computation uses only information available at the current root. The
probe's action blocks were not necessarily known at historical origin 26, so
the probe is a retrospective conditional forecast, not a claim of an online
forecast issued at that earlier instant. Both full forecasts condition on
recorded applied future torques, exactly as the previous studies did.

Run six variants in the declared order: `fast`, `slow`, `half`, `probe_half`,
`probe_inverse`, `probe_fit`. The primary is `probe_fit`.

- `fast`, `slow` and `half` reproduce the original expert and equal-mixture
  predictions, with fresh timing.
- `probe_half` computes the same fitted coefficients as the primary, including
  the rotation grid, then discards them and applies exact half weights. Its
  predictions must match `half`; it isolates the use of the selected weights.
- `probe_inverse` uses the opposite expert's mean squared error divided by
  their sum, independently for each endpoint. Common scaling prevents overflow.
  It does not pay for the unused least-squares fit or rotation grid.
- `probe_fit` minimizes probe position squared error using clipped analytic
  least squares. Rotation chooses the smallest-loss coefficient from the fixed
  ascending grid 0, 1/16, ..., 1. Each grid entry uses the actual float32
  production rotation blend and a float64 squared-geodesic error metric.
  Exact score ties choose the first grid index. This is a discrete grid best,
  not a certified continuous rotation optimum.

For position let D be the mean squared separation between expert probe
positions. If D is at most max(float64.tiny, eps(input dtype)² times the larger
expert position MSE), use the authenticated original training-only
`full_constant` position coefficient. The inverse-error rule uses the same
training-only fallback for an endpoint when both errors are zero. Never use
coefficients from the target-aware capacity diagnosis. Position coefficients
are computed in float64, then cast to the expert dtype before deployment.

## Evaluation and unchanged continuation rule

Use the same two exposed archives, seeds 1101, 1202 and 1303, all 160 windows
and all ten parents per archive. Keep the preceding 33 canonical configurations.
Fresh `fast`, `slow` and `half` are aliases of `decay_huber3`, `gru` and `half`;
check replay and do not count them twice. Three additional variants give 36
canonical configurations, 35 controls and 192 distinct rows including the
six references that are not fitted separately per seed.

The primary must improve pooled RMSE by at least 10% against every positive
control on both endpoints and both archives. It must also be nonworse on every
paired seed, nonworse on at least eight of ten parents, strictly better for all
ten leave-one-parent-out aggregates, and no slower than 1.5 times the freshly
measured GRU median. All 17 groups and 2,101 underlying comparisons remain
conjunctive. Average squared errors before taking square roots. Do not treat
overlapping checks as independent statistical tests or change the criteria.

Run one fixed execution with 36 batch forecasts, three warmups and twenty
timed single-window forecasts per row. Use one CPU thread and deterministic
algorithms. Warmup cases are indices 0..2; timed cases are 3..22. The 864 total
forecast calls include every variant's own probe, with no cross-arm cache.
Time the full callable path, including validation, probe calculation, selection,
expert rollouts, blending and returned diagnostics. Exclude loading, packaging
saved probe arrays, metric reporting and artifact I/O. Preserve all timings.

## Evidence and stop rules

Before model calls, freeze 36 source files, parent hashes, all expert weights,
training-only fallbacks, data identities, variants, grid and numerical tolerances.
Require the externally recorded protocol digest before and after execution.
Do not retry, tune thresholds, refine the grid, drop cases or replace seeds.
On failure, retain every returned batch artifact, completed row and active-row
timing record; do not resume the same execution.
Demote any premature completion record if a later execution step fails. Total
execution time includes the closing payload hash scan, but excludes writing
the final completion record and its console message.

Save 94 successful execution files: start/completion records, two complete
input archives, 36 forecast arrays, 18 probe arrays and 36 row records. Probe
arrays retain the actual grid rotation matrices where used, allowing an
independent audit to verify the production arithmetic and exact grid selection
without resolving a different numerical optimization problem.
Probe arrays are saved after the callable returns. No pre-scoring persistence
callback is used; causal ordering is bound to source and synthetic perturbation
tests, not independently established by a before-scoring artifact seal.

The saved-output audit reconstructs past-only errors, coefficient calculations,
all grid losses and output blends, validates original input and replay bindings,
recomputes physical errors and all continuation checks, and accounts for the
full forecast work. It does not repeat neural inference. Probe neural outputs
and timed-call outputs remain source-bound rather than independently regenerated.

Past-error forecast combination is established, including
[Bates and Granger (1969)](https://www.tandfonline.com/doi/abs/10.1057/jors.1969.103).
Here the five-step probe spans
100 ms while deployment spans 500 ms and uses a larger support set. Transfer
of the coefficient is a hypothesis, not an assumption. These are repeatedly
exposed simulated-robot development archives, not fresh confirmation, closed-loop
robot performance, calibrated confidence, connectome superiority or proof of a
new architecture. A passing development screen would still need those further
experiments. A failed screen keeps the main research objective open.
