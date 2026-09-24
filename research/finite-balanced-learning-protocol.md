# Balanced transitions under a fixed training allowance

Prospective study **finite-balanced-learning-v1**. The saved
[transport diagnostic](finite-transport-diagnostic-results.md) found an
association between inaccurate fits and transition collapse. This comparison
tests a specific constraint, without claiming that association is causal or
that doubly stochastic matrices necessarily preserve information.

## Models and controls

All arms use the existing eight-state recurrent filter with learned transition,
emission, hazard and cost fields, a known uniform reset prior and privileged,
world-aligned initial cost readout. Only public histories and supervised H1/H2
targets enter learning. No true hidden states, reference posteriors or true
transition, emission or hazard parameters enter the learner.

| Arm | Initial transition | Transition during learning |
| --- | --- | --- |
| original_free | Original column softmax | Column softmax |
| matched_free | Same effective T as balanced | Column softmax |
| balanced | Qualified log-domain normalization | 64 fixed row/column sweeps |

Original_free and balanced share identical initial raw transition logits.
Matched_free receives log(initial balanced T), checked against a column-softmax
round trip to 1e-12. All three share exact initial emission, hazard and cost
parameters. The initialization match is independently reconstructed from raw
saved checkpoints. No parent-trained checkpoint is reused.

Balance only T, not the survival operator `S = diag(1-h) T`. The finite
64-sweep autograd graph is the trained function. Every construction requires
finite strictly positive entries below one and maximum row/column mass residual
at most **1e-12**. This tolerance is a numerical acceptance check, not a
convergence guarantee. Failure stops the original attempt; no extra sweeps,
relaxed tolerance, replacement seed or fallback is permitted.

All models store **352 float64 parameters** (320 dynamics plus 32 cost) and
zero buffers. This is not a capacity match: the ideal transition spaces have
224 free dimensions for four column-stochastic matrices versus 196 for four
doubly stochastic matrices, apart from finite numerical tolerance. The true
world uses permutation-plus-uniform transitions, so balanced has a correct
structural prior. Results cannot establish general recurrent-world-model or
biological-learning superiority.

## Fresh data and fixed learning

Scientific namespace **431260924**; TRAIN split 0, **512 attempted histories**;
DEV split 1, **128 attempts**; fit seeds **431261001,431261002,431261003**.
Preserve the existing PCG64 public reset and up-to-eight action/event history
generator, including found terminations. Require at least 256 eligible TRAIN
and 64 eligible DEV histories. Use every valid prefix event for likelihood;
only eligible histories receive H1/H2 endpoint supervision. All nine final
checkpoints must exist before DEV generation. Rotate arm order across seeds.

Every arm follows `prefix_then_joint`: prefix-only dynamics fitting until
elapsed **10 seconds**, then fresh joint Adam from minibatch zero until elapsed
**40 seconds from the same start**. Adam learning rate **0.003**, default
betas/epsilon, global gradient clipping **5**, batch size **64**.

Prefix loss is negative public-history log likelihood minus **0.001** times
`sum(log(T)) + sum(log(O)) + sum(log(h)) + sum(log(1-h))`, divided by the fixed
valid-event count. For balanced, log(T) is the same finite normalization used
by its likelihood. The cost head receives no gradient or update in this stage.

Joint loss keeps blind/observed cost MSE, half the sum of blind/observed
survival MSE, observed-event soft cross-entropy and public-prefix NLL. For
attempted N, eligible S, valid events E and minibatch b, use
`(N/b) * (sum_eligible endpoint_loss/S + sum_valid prefix_NLL/E)`.
There is no prior in joint training. Endpoint-empty batches retain explicit
zero gradients. Accepted joint cursor determines epoch and offset using
`ceil(N/64)` batches and `PCG64(SeedSequence([fit_seed,epoch,818]))` permutations.
Rejected updates do not advance the cursor. No held-out model selection occurs.

## Measured costs and retained states

Start the eligibility clock before model construction, matching, validation,
initial snapshots and optimizer setup. Include normalization and its backward
graph, copies, atomic model/optimizer snapshots, diagnostics and hashes. Accept
an update only if all atomic work finishes by the active deadline. A late
attempt restores model, optimizer and cursor and ends that stage. Rollback,
boundary files and restarts consume the remaining shared allowance. No make-up
time is available. Keep the last accepted state and require at least one
accepted update per stage. A 100,000-attempt guard remains binding.

Save parameter NPZ and lossless Adam JSON at initial, stage-one boundary before
restart, and final states. Retain every attempted update's work, diagnostics,
loss, clocks, hashes and acceptance. Count discarded work. Report complete fit
wrapper time through durable allocation trace and final-summary time; fit-row
publication, shared generation, conversion and evaluation remain in outer
producer time. Eligibility uses perf_counter, whose suspend behavior is
platform dependent; outer caps use the native suspend-inclusive clock. Equal
eligibility is not exact equality of FLOPs, accepted updates or actual wall
time. Forward work counts do not measure backward FLOPs or replay history.

## Metrics and continuation rule

Retain the existing H1/H2/H4/H8 metrics, known-dynamics uniform-state reference
and cyclic shuffled-history control. For balanced, every seed must satisfy:

- SHORT: H1/H2 blind cost MSE and regret at most half the positive uniform
  reference; observed-event KL at most 0.1 nats.
- BLIND: H4/H8 blind cost MSE and regret at most half the positive uniform
  reference; H8 blind-survival MAE at most 0.05.
- OBSERVED: H4/H8 observed-event KL at most 0.1 nats.

Against **each** free control, all six paired H4/H8 regret differences must be
nonpositive, mean regret must improve at least **10% at both horizons**, and
mean complete fit time must be no more than **5% greater**. A positive control
mean is required for a percentage improvement. Zero versus zero is a tie.
Only the conjunction earns **BALANCED_ADVANCE**. No average rescues a failed
seed. Passing motivates untouched replication and a scenario shift; it does
not establish significance, novelty, latent identification or an ICLR result.
This study does not reopen the closed H4-supervised follow-up.

## Qualification and independent audit

Engineering only: namespace **940001**, seed **940101**, eight TRAIN and eight
DEV attempts, batch **3**, deadlines **2 seconds / 4 seconds**. One three-fit
integration run goes through the independent auditor's fixed
`engineering-940001` profile. It uses no scientific cases or checkpoints.
Small-support scientific gates are expected to fail even if technical checks
pass. Fabricated cases exercise probability semantics, hidden-path sums,
gradients, initialization, rollback, counters and failure guards.

Freeze the complete source/import chain, selected tests, dependency files,
runtime and this protocol in exclusive registrations and source snapshots
before any qualification execution. Admit the previously closed numerical
primitive qualification. Qualification, fit/evaluation and audit have original
supervised caps of **300 / 1200 / 600 seconds**, respectively; worker RSS **4 GiB**,
phase output **512 MiB**. Disable ambient pytest plugins, conftest discovery,
bytecode and thread pools beyond one thread. Qualification must close on the
same sources with its exact two successful lint/test commands before fitting.
Retain a failed engineering attempt and its original sources before revisions.
Scientific failure is terminal for this registration.

Audit saved payloads only: reconstruct targets, all metrics, the continuation
rule, initial effective transitions, parameter/optimizer boundary hashes,
public-prefix head preservation, schedules, exposures and source-derived work
counts. Decode 45 NPZ files (27 checkpoints) and 27 optimizer JSONs in science.
Do not run learners, optimizers or the data generator in the audit. Historical
execution, intermediate hashes/timing and normalization checks after initial
states remain source-qualified attestations. Authenticate original sources,
commands, payloads and process closures. Publish every fit and failures.
