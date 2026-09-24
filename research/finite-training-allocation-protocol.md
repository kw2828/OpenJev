# Equal eligibility time for recurrent learning allocation

Prospective study **finite-training-allocation-v1**. The previous
[pretraining comparison](finite-expected-count-learning-results.md) improved
gradient-pretrained averages but spent more time training. No arm passed its
long-horizon criterion. This experiment tests whether the learning allocation
still helps when every arm gets the same overall eligibility window.

## Model, controls and inputs

Use the unchanged eight-state factorized recurrent model: 352 float64
parameters, known uniform reset prior and privileged initial cost readout.
No true latent state, oracle posterior or world transition/emission/hazard
parameter enters the learner. All three arms share complete original parameters
for each seed and the same fresh public histories.

| Arm | First stage, until elapsed 10 s | Second stage, until elapsed 40 s |
| --- | --- | --- |
| joint_continuous | Joint objective, all parameters | Continue Adam and next accepted minibatch |
| joint_restart | Joint objective, all parameters | Fresh Adam, next accepted minibatch |
| prefix_then_joint | Full-batch prefix likelihood, dynamics only | Fresh Adam, joint minibatch zero |

The primary method comparison is prefix_then_joint versus joint_restart.
Joint_continuous is the practical uninterrupted alternative. Stage-one update
counts can differ because each fit has its own measured clock; the two joint
controls are not an exact isolated intervention on Adam moments unless their
boundary states and cursors happen to match. Report their actual counts.

Scientific namespace **430260924**, TRAIN split 0 with **512 attempted
histories**, DEV split 1 with **128 attempts**, seeds **430261001,
430261002,430261003**. Preserve the existing PCG64 case generator and public
reset plus at most eight action/event pairs, including first-found terminations.
Use only eligible prefixes for H1/H2 endpoint supervision, but every valid
public-prefix event for likelihood. Forecast actions are committed before
observations. All nine final checkpoints must precede DEV generation.

## Fixed learning objectives and schedule

For joint learning keep the existing reduction: blind and observed cost MSE,
half the sum of blind and observed survival MSE, observed-event soft
cross-entropy, and public-prefix NLL. With attempted N, eligible S, valid-event
E and batch b, the minibatch loss is
`(N/b) * (sum_eligible endpoint_loss/S + sum_valid prefix_NLL/E)`.
Use batch size **64**, Adam **0.003** with default betas/epsilon, and global
gradient clipping at **5**. Preserve explicit zero gradients on endpoint-empty
batches. The pretraining prior is not included in this objective.

Prefix pretraining updates the 320 dynamics parameters only. Minimize negative
total public-prefix log likelihood minus **0.001** times
`sum(log(T)) + sum(log(O)) + sum(log(h)) + sum(log(1-h))`, divided by the fixed
valid-event count. Use the same Adam settings. No cost-head update is allowed.

Resolve each joint minibatch from its accepted-update cursor: epoch and batch
offset come from division by `ceil(N/64)` and the full epoch permutation is
`PCG64(SeedSequence([fit_seed,epoch,818])).permutation(N)`. Cache immutable
permutations if useful. No mutable sampling RNG advances on rejected work.
The two joint arms continue their accepted cursor across the stage boundary;
the prefix arm starts joint training at zero. Rotate arm order across seeds.

## One clock and complete costs

Start a single monotonic eligibility clock before per-fit model construction,
validation, copies, initial snapshots and optimizer setup. Stage one ends at
elapsed **10 seconds**; stage two ends at elapsed **40 seconds from the same
start**. Shared dataset generation and tensor conversion happen before fitting
and are accounted separately in producer time.

An atomic update snapshots parameters, optimizer moments/steps and the accepted
joint cursor. Accept it only if the update, validation, diagnostics and state
hashes finish by the stage deadline. A late attempt restores all three, records
its attempted and retained hashes, and ends that stage. Discarded computation,
rollback and stage-boundary checkpoint work reduce the remaining global
allowance; no fresh forty-second window or make-up time is available. Keep
the last accepted state, not a best-objective state. Require at least one
accepted update per stage. Reaching **100,000 attempts across the fit** before
the active deadline is a failure. Exceptions preserve failure records; no retries,
replacement seeds, extended caps or relaxed criteria.

Save parameter NPZ and lossless optimizer-state JSON at initial, stage-one
boundary and final states. The boundary checkpoint precedes any restart. Keep
every attempted update's batch, loss, work, timing, model/optimizer hashes and
cursor. Include late atomic work, restoration, logging, construction and
checkpoints in measured fit time. Final summaries outside the eligibility
window are measured separately and still charged. The clock is `perf_counter`;
its suspend behavior is platform dependent. The outer process cap uses the
suspend-inclusive native clock. Report actual elapsed time and overshoot;
equal eligibility windows are not exact FLOP or wall-time equality.

## Outcomes and advance rule

Retain the original H1/H2/H4/H8 metrics, known-dynamics uniform-state reference
and cyclic history-shuffle control. Require at least 256 eligible TRAIN cases
and 64 eligible DEV cases. For every fit seed:

- SHORT: H1/H2 blind cost MSE and regret at most half the positive uniform
  reference, and observed-event KL at most 0.1 nats.
- BLIND: H4/H8 blind cost MSE and regret at most half the positive uniform
  reference, and H8 blind-survival MAE at most 0.05.
- OBSERVED: H4/H8 observed-event KL at most 0.1 nats.

Advance the prefix recipe only if all three original criteria pass and,
against **each** control, all six paired H4/H8 regret differences are
nonpositive, mean regret falls at least **10% at both horizons**, and mean
complete fit time is no more than **5% greater**. No mean rescues a failed
seed-level original condition. These comparisons are descriptive; passing
would motivate an untouched replication and scenario shift, not establish
significance, latent identification, a new architecture or an ICLR result.
The percentage-reduction conditions require a positive control mean; zero
versus zero is reported as tied optimal performance, not an improvement.

This comparison keeps H1/H2 training. It does not reopen the earlier failed
readout replication or its closed longer-horizon training proposal.

## Engineering and independent audit

Engineering uses namespace **939001**, fit seed **939101**, eight TRAIN/DEV
attempts, batch size **3**, first-stage deadline **2 seconds** and overall
deadline **4 seconds**, plus fabricated callback state and scripted clocks
for controller tests. Its actual saved outputs pass through the independent
auditor's explicit immutable `engineering-939001` profile. Original scientific
gate thresholds are unchanged, so small-support engineering criteria can fail
while technical agreement passes. The scientific worker always uses the
default science profile. No scientific arrays or checkpoints are used during
engineering. Pin the complete local import chain, protocol, dependency files,
runtime and all selected tests before the first qualification invocation.

Original supervised caps: **300 seconds qualification, 1200 seconds fit and
evaluation, 600 seconds independent audit**, with 4 GiB worker RSS and
512 MiB output. Use exclusive original paths. Any failed qualification is
retained with its original source snapshot before a new engineering version.

The saved-output audit independently reconstructs analytic targets, metrics,
all original criteria and the comparative advance rule. It checks all boundary
parameter and optimizer hashes, prefix cost-head preservation, cursor/batch
schedules, accepted/rejected work and the shared deadline. It runs no model,
optimizer or environment. Intermediate execution, counters and historical
timings remain source-qualified attestations. Publish every fit and the
original process closures regardless of outcome.
