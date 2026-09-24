# Equal-update recurrent learning with qualified computation reuse

This fresh experiment follows the [measured integrated throughput gain](finite-joint-reuse-throughput-results.md).
It asks the still-unanswered equal-update question from the
[earlier stopped protocol](finite-update-learning-protocol.md), using the
qualified shared-computation implementation for all three models. The stopped
study and the unsuccessful equal-time comparison remain closed. No saved fit
is extended or repeated, and no previous outcome changes status.

## Question and fixed models

Does the rounded recurrent transition improve blind four/eight-step decisions
after equal prefix and joint training exposure? Compare `rounded` against both
`original_free` and `matched_free`. Retain all 352 stored parameters, four
normalization sweeps, slack 1e-8 and row/column residual tolerance 1e-12.
Original free and rounded share raw initial transition logits; matched free
shares rounded's effective initial transition within 1e-12. Emissions, hazards
and privileged starting readouts remain paired. All model and objective
sources stay unchanged.

Use the qualified `run_finite_joint_reuse_training.train` with explicit
`implementation='reuse'` for every arm. Reuse only per-call probability
construction and endpoint history filtering. Preserve attached gradients,
separate reset arithmetic, pre-observation predictions, global reductions
and all original loss coefficients. Record actual joint work in its five
disjoint routes; do not count shared operations twice or call logical
rollout counts backward FLOPs.

## Fixed exposure and costs

Every scientific fit must complete exactly 1,024 prefix updates and then
3,072 joint updates. Prefix learning updates dynamics only with full public
history likelihood and the existing 0.001 log prior. Save prefix Adam and
model at the boundary, create fresh joint Adam, and reset the joint batch
cursor to zero. Keep H1/H2 endpoint supervision plus globally normalized
prefix likelihood without the prior in joint training. Batch size 64,
learning rate 0.003 and gradient clip 5 remain unchanged. Pair ordered
minibatches across arms using each fit seed and epoch.

The same 120-second controller cap is a failure limit, not an eligibility
window. Charge validation, construction, all snapshots/checks, optimizer
work, boundaries and final summary. Preserve partial traces and rollbacks
on any exception; do not retry, replace a seed or proceed to DEV after a
failed fit. Full fit time includes durable allocation serialization; final
fit-row writes are charged to the outer producer. Report controller, full
fit, generation, prediction, audit and native phase costs with their scopes.
Equal updates do not imply equal FLOPs, wall time or gradient geometry.

## Engineering admission

Freeze source, protocol, runtime, configuration, commands and snapshots before
one qualification. Authenticate the complete original throughput result and
closed native phases, including all 92 frozen sources and the preserved first
qualification failure. The throughput result motivates this implementation;
it does not automatically admit this scientific exposure.

Run lint, selected qualification tests and one fresh exposure probe in that
order. Smoke fixtures use namespace 944001, seed 944101, 8 TRAIN/8 DEV attempts,
batch size 3, 3 prefix/4 joint updates and a 30-second controller cap. The
exposure probe uses namespace 944201, seed 944301, 512 TRAIN/8 DEV attempts,
batch size 64, 32 prefix/64 joint updates and the same 30-second cap. Run all
three arms once and audit their saved outputs. Do not select models using
engineering effectiveness metrics.

For each arm, project runtime with the unchanged conservative rule:
`2 * (prefix_stage_seconds * 1024/32 + joint_stage_seconds * 3072/64)`
plus measured non-stage full-fit time. Every projected value must be strictly
below 90 seconds. This estimates feasibility, not guaranteed runtime or a
new speed result. A failure stops this registration; counts, thresholds,
caps and source files cannot be adjusted in place.

## Fresh scientific data and evaluation

Only after the original qualification closes successfully, create a scientific
registration with identical sources and settings. Use namespace 434260924,
fit seeds 434261001, 434261002 and 434261003, 512 TRAIN attempts and 128 DEV
attempts. Keep every attempted history, including early termination, and
require at least 256 eligible TRAIN and 64 eligible DEV cases. These identities
must not be used by qualification or inspected before the scientific run.

Rotate arm order by seed. Complete all nine final checkpoints before generating
DEV. Train on H1/H2 targets and evaluate H1/H2/H4/H8. Retain the known-dynamics
uniform-state reference, exact target witness, cyclic history shuffle, blind
predictions and observed-filtering predictions. Learned models receive public
histories and supervised targets; oracle states are confined to target and
reference checks. No scenario-shift data are generated in this experiment.

## Unchanged advancement rule

`UPDATE_MATCHED_ADVANCE` retains all 19 conditions:

- Rounded passes the original all-seed SHORT, BLIND and OBSERVED criteria.
- Against each free control, every paired H4/H8 regret difference is nonpositive.
- Against each free control, mean H4 and H8 regret are each at least 10% lower,
  with positive control means.

Any failed condition means no advancement. Preserve every arm, seed and
criterion, including ties and negative results. Report mean full-fit times
and ratios against both controls as descriptive costs; there is no runtime
condition in this effectiveness rule. Do not substitute the easier control,
average away a losing seed, or change a criterion after results are available.

## Supervision, independent audit and publication

Use separate native-supervised phases: qualification 300 seconds, producer
1,200 seconds and audit 600 seconds. Require original process closure before
downstream admission. Use one numerical thread, disable ambient pytest plugins
and bytecode writes, and retain the existing sampled 4 GiB RSS and 512 MiB phase
output bounds without claiming continuous resource enforcement. Registrations
and output paths are exclusive. Any timeout, incomplete cleanup, numerical
failure or failed audit ends the original attempt and preserves its evidence.

The auditor reads saved arrays and metadata only. Independently reconstruct
targets, metrics, original criteria, actual probability constructions, every
accepted batch/exposure, disjoint work and all 27 model plus 27 optimizer
boundaries. It must not replay learning or call a world generator. Intermediate
losses and timers remain source-qualified execution records. Retain source
snapshots, closures, full traces, every checkpoint and all prediction outputs.

Publish the complete result with a chart and all evidence, including a failure
if that is the outcome. A local pass would justify a separate untouched
replication and scenario-shift test, followed by a second environment. It
would not establish biological wiring, a new architecture, calibration, RLCD,
statistical significance or ICLR readiness. The small synthetic world, known
doubly stochastic prior and privileged starting readouts remain limitations.
