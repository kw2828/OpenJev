# Next hypothesis after the action-latent pilot

Prospective proposal only. This note admits no collection, fit, retest or earlier
TEST/confirmation data. The completed pilot remains **DEV_FAIL: 2/18 cells**.
Neither architectural novelty nor autonomous-control effectiveness is established.

## What the completed pilot supports

The [collection summary](../output/otto-action-latent-v1/collection-01/summary.json)
retains 124 of 192 TRAIN cases, 30 of 48 lambda3 DEV cases and 38 of 48 lambda4
DEV cases. The 96 cases found during the eight-transition prefix were excluded
without replacement. Thus the estimand is prediction for survivors of that
analytic prefix, not all starting cases. TRAIN clears the registered minimum of
120 by four cases. The three initialization seeds share all evaluation cases;
they are not three independent datasets.

The [saved independent audit](../output/otto-action-latent-v1/audit-01/audit.json)
agrees with the producer and records 2 passing cells out of 18. Both passing
cells are seed323000002, lambda3, against direct_horizon and ridge. The audit's
internal technical_complete=false requires its original supervisor closure; it
is not a scientific-gate failure marker. The completed original process closures
are separate from the failed scientific continuation rule.

Below, positive percentages favor action_recurrent. Each entry is the **mean of
the three paired relative reductions**, 100 times mean((control-candidate)/control),
using the saved case-weighted metrics. These are descriptive summaries, not
significance estimates. Ridge was fitted once and reused across comparisons.

| Evaluation | Regime | Control | Log-score reduction | Decision-gap reduction |
| --- | --- | --- | ---: | ---: |
| Gap, horizons5-8 | lambda3 | action_blind | -2.66% | +9.03% |
| Gap, horizons5-8 | lambda3 | direct_horizon | +3.80% | -0.12% |
| Gap, horizons5-8 | lambda3 | ridge | +53.18% | -6.62% |
| Gap, horizons5-8 | lambda4 | action_blind | -1.73% | -4.26% |
| Gap, horizons5-8 | lambda4 | direct_horizon | +2.28% | -16.57% |
| Gap, horizons5-8 | lambda4 | ridge | +63.25% | -14.25% |
| Normal, horizons1-8 | lambda3 | action_blind | -2.47% | +15.03% |
| Normal, horizons1-8 | lambda3 | direct_horizon | +3.99% | +12.76% |
| Normal, horizons1-8 | lambda3 | ridge | +47.15% | +10.62% |
| Normal, horizons1-8 | lambda4 | action_blind | -3.45% | -2.71% |
| Normal, horizons1-8 | lambda4 | direct_horizon | +4.65% | -1.56% |
| Normal, horizons1-8 | lambda4 | ridge | +65.14% | -0.68% |

The candidate has worse mean log score than action_blind in every row pairing.
Its lambda4 long-gap decision error is worse than direct_horizon for all three
seeds. A better log score than ridge is insufficient: in that same lambda4
long-gap group, mean Brier is 0.46027 for the candidate versus 0.40821 for ridge,
which favors ridge. Ridge's different categorical objective and near-zero
probabilities can make log score and Brier rank it differently. This does not
establish a calibration result. Also, the lambda3 long-gap decision comparison
against direct has -0.12% mean paired improvement, although a ratio of aggregate
means would give +2.06%; these two summaries must not be interchanged.

## One specific limitation to correct in a new study

The [normal-rollout implementation](../src/openjev/research/otto_action_latent_model.py)
forecasts before receiving each observation. After observed found, it freezes
the pre-assimilation latent and repeatedly applies its learned readout. It does
not make the already-known terminal suffix deterministic. The ridge reference
also freezes its descriptor. This follows the registered protocol but spends
prediction loss on an event already known to be absorbing.

For a separate study, all methods should retain the forecast at the first found
event, then output probability one for found on subsequent **observed-terminal**
horizons. Gap forecasts must never receive the realized found flag. Report
preterminal/first-event and already-known terminal suffixes separately. This
common semantic correction cannot establish a recurrent-model advantage and
cannot explain away this pilot's gap failures. Do not rewrite its saved scores.

## Recommended mechanism: distill a compact predictive belief state

Before expanding the architecture, establish the known-model predictive
reference, then test whether its soft distributions improve the existing small
action-conditioned state. The hypothesis is that 124 sampled four-step blocks
give too little distributional supervision for F and G to learn useful action
effects. This is a hypothesis, not an explanation established by the failure.

OTTO has a static source, known movement and a public observation likelihood.
Let p(x) be the normalized source belief after the observed prefix, q_h the
planned position at horizon h, and V_h the set of unique positions visited by
the committed block through h. For a gap without intervening observations:

```
P(found by h) = sum_{x in V_h} p(x)
P(odor=k, alive at h) = sum_{x not in V_h} p(x) L_k(x, q_h), k=0..3
```

These are five unconditional marginal probabilities, not probabilities obtained
by rolling forward one imagined odor sequence. Duplicate visits contribute
source mass once. For normal forecasts, use the latest observed public belief
before the next action; apply the likelihood update only after its observation.
An observed terminal state has a deterministic absorbing future.

Qualify this reference first on small enumerated source grids, repeated visits,
normalization, zero-support branches and terminal transitions. In particular,
the current public filter can retain subnormalized mass below its numerical
floor. Do not silently renormalize that representation and call it an exact
reference: define and validate the probabilistic filter separately, disclose
any divergence, and fail undefined cases according to the new protocol. The
reference uses the **full belief and known likelihood**, information beyond the
learner's compressed31-feature input. It is a privileged expected-proper-score
ceiling under those assumptions, not a fair learned baseline or a guarantee of
the best realized score on every finite sample.

The proposed single training change is to replace sampled categorical labels
with these exact TRAIN predictive distributions in the outcome cross entropy.
Keep the31-feature learner input,28-dimensional F/G, cost and auxiliary losses,
normalization, optimizer and dataset budget fixed. No full belief, future odor
or true source location enters the learned model at inference. This makes the
latent state learn a compact predictor of Bayesian belief dynamics without
adding a larger recurrent architecture. The full-belief teacher is a training
resource whose annotation and computation costs must be charged.

Register four arms: action_recurrent with soft targets, the same model with
sampled targets, and action_blind and direct_horizon with the same soft targets.
The first contrast tests distributional supervision; the latter two test whether
actions and recurrence add value once supervision is matched. Preserve the
teacher-cost decision metric: the categorical oracle does not supply an exact
expectation of the nonlinear future teacher policy. Do not imply it does.

Use newly reserved originating seeds with the same precommitted action-block
causality, short TRAIN gaps and longer DEV gaps plus the lambda shift. Freeze
retention floors and total starting-case budget before collection; do not replace
prefix terminations or raise the sample count after observing the new results.
Report survival/support denominators, normal and gap proper scores, decision
error, teacher-relative predictive regret and actual total computation. A
predeclared action-permutation diagnostic can test whether learned action
sensitivity agrees with the known-model reference, without selecting successful
cases after evaluation.

Advance only if the soft-target recurrent model improves on its sampled-target
twin and both matched soft-target controls on the predeclared long-gap metrics,
survives the lambda shift, and avoids normal-observation regression. The precise
thresholds and budget need a separate registration. If the exact predictor has
little useful action sensitivity on these cases, or distillation still fails to
beat action_blind, stop promoting this task as evidence for action recurrence.
This proposal makes no novelty, robotics transfer, calibration or paper claim.

## Evidence identity and scope of this note

Only existing source and saved JSON were read; no arrays or checkpoints were
decoded and no model, solver, simulator or teacher was called. No registered
source, output or acceptance rule was changed.

* collection-01/summary.json SHA256
  `9daf1220e5091330ee6cd9d9eb7e6a6db1dccefd758f85c8641762f3b2d3ffb1`
* fit-01/summary.json SHA256
  `3f943ce6f9db23ff46232885fd584d919b15e5fe2af2e8a4bb42769beab8fb27`
* audit-01/audit.json SHA256
  `943942f88cfa985c18fadcdf52743e596fc6f2482f6d0b8ec5bf2fb52bb6a65e`

The existing [protocol](otto-action-latent-protocol.md) remains authoritative for
the completed failure. This document is an unregistered proposal, not permission
to reuse exposed DEV as fresh evidence or reopen any earlier TEST panel.
