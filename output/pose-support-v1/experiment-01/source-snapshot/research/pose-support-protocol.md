# Recency and robust support: fixed-checkpoint development screen

September 19, 2026. Prospective protocol for `pose-support-v1`.
The [saved-output diagnosis](pose-adaptation-failure-diagnosis.md) found that
large early context changes dominate the previous model's excess position
error, although later windows often benefit from adaptation. This screen asks
whether old support, large residual influence, or merely prior shrinkage
explains that failure. It does not change any earlier experiment or its gate.

## Frozen models and data

Use the three final meta checkpoints from pose-adaptation-v1, seeds 1101,
1202 and 1303. No new training, feature learning, prior changes or calibration
is performed. Retain all 160 windows from each already exposed plain and zigzag
archive, ten parent trajectories per archive. Source-start windows stay in
every primary metric. The original prepared data, normalization, geometry and
25-step forecast remain unchanged.

The model sees 32 observed poses and 31 completed action blocks. Its 30 support
rows are transitions t=1..30, with the same body-frame changes of world motion
as before. Forecasts start from pose31 and actual motion30 to31. They receive
only future actions31..55, not future poses. Fitted weights remain fixed during
the 25-step rollout. The unchanged model maps physical state plus40 ordered
torques to13 features and uses a13-by6 global prior. See the
[original protocol](pose-adaptation-protocol.md) for physical units and limits.

## Prespecified support mechanisms

All weighted fits retain unit ridge precision centered at the original W0.
For each output j, solve in float64 and cast final weights to float32:

```
W_j = W0_j + solve(Phi.T diag(q_j) Phi + I,
                  Phi.T diag(q_j) (Y_j - Phi W0_j))
```

The ten configurations are:

- **prior:** disable fitting; use the identical meta checkpoint's prior.
- **full:** unchanged unit-weight30-row context fit.
- **recent5:** weight only the last five supports by one, the rest zero.
- **decay5:** all30 supports weighted by2^(-age/5), age29..0.
- **huber3:** exactly three iteratively reweighted solves, starting from W0.
  At each step q_ij=min(1,1.5/abs(Y_ij-Phi_i W_j)).
- **decay_huber3:** multiply the exponential recency weight by that Huber weight
  at each of the three steps. This is the primary candidate, fixed beforehand.
- **recent5_mass, decay5_mass, huber3_mass, decay_huber3_mass:** fully compute
  the corresponding method, then replace its final support weights by their
  per-output average over30 rows and solve once more around W0. These preserve
  total evidence mass but remove assignment to particular transitions.

The Huber threshold is1.5 in existing normalized target units, not an estimated
noise standard deviation. Three IRLS steps are an approximate robust solve,
not an exact converged robust estimator. Weighting enters both sides of the
normal equation. Mass controls include all work used to derive their weights
in measured latency; they are not supplied free statistics from another arm.

Record weight mass, effective support size (sum q)^2/sum(q^2), the fraction
of weights below one, and posterior-correction norm for every window/output.
Report both actual and pre-uniformization summaries for mass controls. Prior
uses zero support and the explicit zero-summary convention. Diagnostic JSON
formatting is outside timing; deriving all weights and fitting/forecasting is
inside it. No diagnostic uses future targets.

Forgetting and bounded influence are established estimation tools, including
[Kovacevic et al., robust weighted least squares](https://pmc.ncbi.nlm.nih.gov/articles/PMC4962952/)
and [Bruce et al., variable-rate forgetting](https://arxiv.org/abs/2003.02737).
This screen does not claim a new architecture, biological mechanism or
calibrated uncertainty. Large early changes may be real dynamics, not bad data.

## Comparisons, execution and unchanged standards

Freshly evaluate the previous static, static-with-adaptation, public-feature and
GRU checkpoints for every seed. Keep hold, CV1, CV16, LS16, body16 and the
previously fitted torque ridge as references. There are14 neural configurations
times3 seeds plus6 references per panel:96 rows total. The controls' forecast
arrays, and prior/full, must reproduce their previous saved outputs within the
auditor's declared arithmetic tolerance: float32 rtol=1e-6 and atol=2e-7;
ridge float64 rtol=1e-12 and atol=1e-12. Report byte equality and maximum
absolute differences separately. No reference is retrained.

Use deterministic float32 CPU prediction with one thread, retaining float64
for the old ridge reference and context solves. Each row receives three warmups
and20 measured complete-window calls. Each neural family has120 timing samples;
each reference40. Include context computation and all25 predicted steps,
excluding loading, normalization, metric reduction and service overhead.
Module initialization is overwritten by bound weights; no optimizer is run.

Before execution, bind sources, tests, prior protocol/completion/checkpoints,
prepared data and this protocol by hashes. Create one exclusive run directory.
There are no retries, parameter sweeps, replacement seeds, window exclusions,
selected checkpoints or post hoc changes to the primary method. Save all
predictions and diagnostics before independent saved-output auditing.

The continuation rule stays demanding. For each of the two physical endpoints
on each panel, primary decay_huber3 must beat **every other19 configurations**:

1. Pooled RMSE at least10% lower.
2. All three paired-fit MSE comparisons nonworse.
3. At least8/10 parent MSEs nonworse.
4. Every leave-one-parent-out MSE strictly lower.

Median full-window candidate latency must be at most1.5 times the freshly timed
GRU. These are17 grouped conjunctions retaining1,141 underlying comparisons
(76 aggregate,228 paired,76 parent-count,760 leave-one-parent-out,1 latency).
All groups must pass to justify further scaling. Passing would still require
fresh confirmation and a separate novelty case. A failure remains failed.

Always report all-window physical position/rotation errors. Source-start versus
later-window comparisons are descriptive localization, not an exclusion rule.
The matched evidence-mass controls prevent crediting recency or robustness for
simply trusting the prior more. If recent5 or a uniform shrinkage control
explains an improvement, report it as such rather than an architecture gain.

The experiment remains offline simulated-robot prediction conditioned on future
recorded applied torques. Data chronology here does not establish causal system
identification or closed-loop control. Raw/prepared arrays, targets and forecasts
remain local because upstream data licensing is not explicit. Publish our source,
numeric errors, aggregate support diagnostics and execution receipts.
