# Expected-count prefix learning before joint recurrent training

Prospective study **finite-expected-count-learning-v1**. The [readout diagnostic](finite-gap-readout-study-results.md)
did not resolve decision failures after all nine frozen heads met the numerical
criterion. This study changes how the same recurrent model is initialized by
public-history learning. It does not reopen the failed readout replication or
its longer-horizon training proposal.

## Question, model and controls

Does explicit expected-count learning improve decisions after accounting for
its additional training, compared with gradient pretraining and no pretraining?
All three arms use the unchanged eight-state factorized model, **352 trainable
float64 parameters**, uniform reset prior and privileged world-aligned initial
cost head. Transition, emission and hazard initial values remain independent
of the world's true values. The action-conditioned T, shared observation law O
and next-state hazard h have the same meanings as in the
[factorized model protocol](finite-factorized-dynamics-protocol.md).

| Arm | Public-prefix stage | Joint decision stage |
| --- | --- | --- |
| joint_only | None | Original 480-epoch recipe |
| gradient_prefix | Full-batch Adam on regularized prefix likelihood | Same 480 epochs |
| em_prefix | Forward-backward MAP expected-count updates | Same 480 epochs |

Pair complete original parameters and cost-head bytes by seed. Prefix stages
update only the 320 dynamics parameters. The common joint phase learns all
352 parameters, starting with a fresh Adam optimizer in every arm. Do not align
latent states to the true world after pretraining. Save original and post-prefix
parameter arrays for every arm, including the unchanged control.

Both prefix learners maximize total log likelihood plus **0.001** times
`sum(log(T)) + sum(log(O)) + sum(log(h)) + sum(log(1-h))`.
The gradient loss is its negative divided by the fixed valid-event count, with
Adam learning rate **0.003**, default betas/epsilon and gradient clipping at **5**.
EM adds the explicit count 0.001 to each category/hazard outcome and normalizes.
This is MAP, not plain MLE. The unchanged cost head receives no prefix gradients.
The [numerical qualification](finite-expected-count-qualification-protocol.md)
checks the inference against independent exact hidden-path sums, checks the
shared objective and verifies finite-logit conversions without clipping.

## Time allocation and complete accounting

Each gradient/EM prefix stage gets **10 seconds of update eligibility** on the
same CPU, with one thread. Timing starts before validation, initial diagnostics,
public-token conversion and optimizer construction. It includes parameter
copies, hashing, inference, gradient updates and before/after objective checks.
The gradient optimizer persists throughout that stage.

Start a new complete update only while elapsed time is below 10 seconds. Accept
it only if all its work finishes at or before 10 seconds. If it finishes late,
restore the previous accepted dynamics byte-for-byte, retain the attempted
result in the trace and stop. No update after that late attempt is allowed.
Charge its computation and rollback. A secondary 100,000-update cap reached
before the time deadline is an explicit failure, not an alternative success.
Require at least one accepted update per active prefix stage.

This matches the eligibility allocation, **not exact FLOPs or total wall time**.
Record actual elapsed time, atomic overshoot, separate final-summary costs,
accepted and attempted updates, gradient passes, expected-count passes,
diagnostics, public-event exposures, copies, hashes and rollback work. The outer
phase supervisor is suspend-inclusive; the eligibility clock is monotonic
`perf_counter`, whose suspend behavior is platform-dependent. Machine contention
can change update counts. Report complete costs and do not label equal update
counts or equal nominal seconds as identical computation.

Record original and post-prefix checkpoints before joint training. The joint
initial hash must match the retained post-prefix state, including any rollback.
No best-objective checkpoint selection, restart, replacement seed or extended
allocation is allowed. All checkpoints and failures stay in the result.

## Fresh data and common training

Scientific namespace **429260924**, TRAIN split 0 with **512 attempts**, DEV
split 1 with **128 attempts**. Fit seeds **429261001, 429261002, 429261003** give
nine fits. Each case uses the unchanged `PCG64(SeedSequence([namespace,split,index]))`
collector. Preserve reset and up to eight public action/event pairs, including
first-found terminal histories. Every valid prefix event contributes likelihood;
only surviving prefixes receive H1/H2 endpoint training targets. No old arrays,
checkpoints, hidden states, oracle posteriors or true world parameters enter
a learned arm. Exact world information is used only for labels and reference
validation. Forecast actions are committed before their observations.

Use the original joint objective: blind cost MSE, observed cost MSE, half the
sum of blind/observed survival MSE, observed-event soft cross-entropy, and prefix
event NLL with coefficient 1. Keep all original component reductions. For a
batch of b attempted histories, fixed attempted N, eligible S and event E:

`joint loss = (N/b) * (sum_eligible endpoint_loss/S + sum_valid event_NLL/E)`.

The pretraining prior is **not** added to the common joint loss. Train for **480
epochs**, batch size **64**, Adam at **0.003**, global gradient clipping **5**.
Pair the same attempt permutations from `PCG64(SeedSequence([fit_seed,epoch,818]))`.
Retain explicit zero gradients and the original behavior for no-endpoint batches.
Rotate arm execution order by seed. All nine final checkpoints must be durable
before DEV generation, with the original exact-reference checks before fitting
and after DEV generation. No fitting or checkpoint selection uses DEV.

## Evaluation and continuation

Evaluate every arm and seed at **H1/H2/H4/H8**, with the unchanged known-dynamics
uniform-state reference and cyclic public-history shuffle. Public-prefix NLL
includes all attempted DEV histories, including found events. Means and paired
differences are descriptive; there is no ensemble or significance claim.

Retain all existing criteria without adjustment. Require at least **256 eligible
TRAIN** and **64 eligible DEV** cases and positive reference MSE/regret. For every
seed:

- SHORT: H1/H2 blind cost MSE and regret at most half the uniform reference,
  observed-event KL at most 0.1 nats.
- BLIND: H4/H8 blind cost MSE and regret at most half the uniform reference,
  H8 blind-survival MAE at most 0.05.
- OBSERVED: H4/H8 observed-event KL at most 0.1 nats.

No favorable average rescues a failed condition. A prefix-learner improvement
must be assessed against both controls with all added work charged. Passing
these small-world criteria would establish a useful learning recipe here, not
latent-state identification, a connectome advantage or an ICLR novelty claim.
Scenario shift and a second environment would require separate protocols.

## Qualification and independent audit

Engineering uses namespace **938001**, seed **938101**, eight TRAIN/DEV attempts,
one joint epoch and one-second prefix eligibility allocations. The separate
clock/controller tests use fabricated histories and scripted timestamps at
seed937101. No scientific data is generated in engineering.

Pin this protocol, the complete local import chain, dependencies, runtime,
fixed settings and the successful numerical prerequisite. Qualify lint and all
selected tests before registering science. Original supervisor caps are **300
seconds integration**, **1800 seconds fit/evaluation**, and **600 seconds audit**,
with 4 GiB worker RSS and 512 MiB phase output. The deadline is checked on every
callback; resource checks occur initially, finally and every 256 callbacks.
Use exclusive output paths and one attempt per registered scientific phase.

After the original producer closes, an independent audit reads saved arrays,
reconstructs analytic targets and all metrics, checks original/final prefix
checkpoint hashes and head preservation, and verifies cutoff/rollback traces
and common-training work. It decodes 18 prefix-stage checkpoints without
replaying a model, optimizer or generator. Intermediate parameter updates and
timings remain source-qualified producer attestations; saved hashes cannot
independently prove their historical execution. Preserve failed engineering
attempts separately; any repair requires a new source version and registration.
