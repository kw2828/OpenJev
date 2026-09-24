# Independent-cohort action-error loss comparison

This is a new loss-control hypothesis motivated by the completed
[decision-error diagnostic](finite-decision-error-results.md). It preserves
the previous 14/15 continuation failure. It does not claim a new architecture,
calibrated probabilities, biological learning, or improvement before evaluation.
The [primitive qualification](finite-action-range-qualification-protocol.md)
passed all 46 fabricated tests in its original clean 6.706187041-second native
process. That result alone does not admit scientific training.

## Fixed comparison

Cross the unchanged 352-parameter rounded and initially matched free recurrent
models with three blind-cost losses: MSE, twice MSE, and the mean of the squared
four-action error range divided by four. The six arms are `rounded_mse`,
`rounded_double`, `rounded_range`, `free_mse`, `free_double`, and `free_range`.
Use the qualified random-head factory. All observed-cost, survival, event and
public-prefix losses remain unchanged. Extra scalar loss work is recorded
separately from the qualified shared model computation. These counts are not
FLOPs, and equal update counts do not imply equal compute or elapsed time.

Use five independent data namespaces 437260924 through 437260928, paired in
order with fit seeds 437261001 through 437261005. Each cohort has 512 TRAIN
attempts, 512 BASE attempts, and 512 SHIFT attempts. Within a cohort, all six
arms share data, random head, and ordered minibatches. The three loss variants
of an architecture share exact initial and prefix-boundary model and Adam
states. The initial transition functions match across architectures, not their
raw coordinates or gradients. Rotate the six-arm execution order by the
cohort index. There is one fit per arm and cohort, with no replacements.

Use 1,024 full-prefix updates and 3,072 joint updates, batch size 64, Adam
learning rate 0.003, gradient clipping 5, the original prefix prior 0.001 and
no joint prior. Reset Adam at the same prefix/joint boundary as the qualified
trainer. Supervise only horizons 1 and 2. All 30 final checkpoints must exist
before any BASE or SHIFT generation. Evaluate those same checkpoints at
horizons 1, 2, 4 and 8 without adaptation. BASE uses observation noise 0.12;
SHIFT uses 0.30. Transitions, hazard and true costs stay fixed. This is one
synthetic world family, not an independent second environment.

## Prospective continuation rule

The fixed candidate is `rounded_range`. The primary controls are `rounded_mse`
and `rounded_double`. There are 54 top-level conditions, all required:

1. For each control, cohort, regime and horizon in {4, 8}, candidate mean blind
   decision regret must be strictly lower: 40 paired conditions.
2. For each control, regime and horizon in {4, 8}, the control's equal-cohort
   mean regret must be positive, and the candidate's equal-cohort mean must
   be at least 10% lower: eight conditions. Do not weight a cohort more because
   it retained more eligible cases.
3. The candidate must pass each of the inherited short-horizon, blind
   extrapolation and observed-filtering criteria independently in BASE and
   SHIFT: six groups. Check every constituent condition for every cohort.

The inherited criteria require at least 256 eligible TRAIN cases and 64 eligible
evaluation cases in each cohort. At H1/H2 and H4/H8 respectively, blind-cost
MSE and regret must each be no more than half their positive uniform-belief
reference values. The short-horizon criterion also requires observed KL at
most 0.1 at H1/H2. Observed-filtering extrapolation requires observed KL at most
0.1 at H4/H8. Blind extrapolation additionally requires H8 survival MAE at most
0.05. References and support are checked within each cohort before aggregation.

No mean improvement rescues a paired or absolute failure. Report all six arms,
all cohorts, both regimes, all metrics, complete fitting time and loss-work
counts. Comparisons against free models and loss effects in free models are
descriptive controls. This is a prespecified point-estimate continuation rule,
not a significance test, a search for the best-performing arm or a claim of
architecture superiority. A pass would justify further validation of this loss
on a less favorable transition family and second environment. A failure closes
this attempt; retain it without changing the rule, seeds or budget.

## Engineering admission and execution

Freeze the 152 primitive/inherited sources plus nine integration files, for
161 files, with exact runtime and source snapshots. The engineering phase
runs fixed Ruff and fabricated runner/auditor/admission tests, then a TRAIN-only
feasibility probe and one archived full-run smoke. No engineering effectiveness
value chooses a loss, seed, recipe or scientific acceptance threshold.

The feasibility probe uses only namespace 948201 and seed 948301, 512 TRAIN
attempts, batch 64, 32 prefix and 64 joint updates per arm, with a 30-second
per-fit cap and 240-second internal native cap. `dev_attempts=8` is unused
configuration; no DEV generation or prediction metrics are allowed. For each
arm project the two measured stage times to 1,024/3,072 updates, multiply their
sum by two, and add the measured non-stage fitting time. Every projection must
be at most 90 seconds. Keep all six outputs and original saved training audits.
This projection is feasibility evidence, not a runtime guarantee or speed result.

The archived smoke uses one separate cohort 948001/948101, 64 TRAIN and 32
attempts per evaluation regime, batch 16, two prefix updates, three joint
updates, and a 30-second fit cap. It must complete all six fits, both regimes
and an independent saved-output audit. Its scientific continuation conditions
need not pass because it is deliberately too small to test efficacy.

The native engineering, science and audit caps are 600, 1,800 and 600 seconds.
Science also has a 120-second cap per fit. Use one numerical thread, sampled
4 GiB worker RSS, 1 GiB output per phase and at most 2,048 output entries.
The output budget accounts for 30 full training traces and two evaluation
regimes. The wrapper samples its own RSS; the exposure child checks its own.
This is not a continuously enforced aggregate process-tree memory limit.

Require engineering success and original clean process closure before creating
the scientific registration. Bind the full original qualification, feasibility
and smoke outputs. The science process writes all checkpoints, optimizer states,
traces, predictions and work counters before a separately supervised audit.
The audit reconstructs data and metrics from saved arrays and independently
joins all boundaries and counts. It executes no learned model, optimizer,
simulator or fresh data generator. Intermediate logged training losses are
execution attestations supported by the qualified loss code; the audit does
not replay their gradients or optimization trajectory.

Every phase has exclusive paths, immutable registration and a source snapshot.
Preserve partial files, failure receipts and native terminal status on failure.
Do not retry, extend budgets, replace seeds or edit frozen sources. Require
original process-group closure and unchanged producer files before publication.

## Claim boundary and related work

[Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005) already motivates
training predictors for downstream decisions. The action-error bound is an
optimization hypothesis, not an established novel model. Neither this study
nor the decision API establishes TypeSafe's private architecture or reproduces
its unpublished RLCD recipe. Decision correctness, latency and empirical
probability calibration require separate measurements.
