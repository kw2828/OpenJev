# Fresh recurrent replication and observation shift

The [local equal-update comparison](finite-reuse-learning-results.md) passed
19/19 predeclared conditions, with all paired seed regrets favorable. Two weak
fits in each control dominate its large mean differences. This new study asks
whether the same recipe repeats across five fresh seeds and transfers to noisier
observations. It does not reopen or extend any prior fit or failed study.

## Models and training held fixed

Train `original_free`, `matched_free` and `rounded` once per seed, using the
unchanged 352-parameter constructors, objectives, initialization rules and
qualified `implementation='reuse'` training helper. Every fit must complete
exactly 1,024 prefix updates then 3,072 joint updates. Retain batch size 64,
learning rate 0.003, gradient clip 5, the 0.001 prefix-stage prior and fresh joint
Adam state and batch cursor at the stage boundary. Pair ordered minibatches by
fit seed across all arms. Rotate arm order across seeds.

The 120-second per-fit controller cap is a failure limit. Charge setup,
validation, updates, optimizer state, checkpoints and final summary as before;
report durable trace serialization in complete fit time. A failure preserves
partial work and ends the run. No retry, replacement seed, extra optimization,
threshold change or evaluation-driven selection is allowed.

## Fresh data and fixed observation shift

Use namespace **435260924**, fit seeds **435261001 through 435261005**, and
512 TRAIN attempts from split 0 at epsilon 0.12. Train on H1/H2 endpoint labels
and every valid prefix event, including prefixes that terminate early. Retain
all attempts and require at least 256 eligible TRAIN cases through the unchanged
absolute criteria. No old data or checkpoints enter training.

Finish all 15 final model checkpoints and optimizer checkpoints before either
evaluation cohort is generated. Evaluate each unchanged model on the same
512 BASE attempts from split 1 at epsilon 0.12 and the same 512 SHIFT attempts
from split 2 at epsilon 0.30. Both use H1/H2/H4/H8 metrics. Use the frozen
generator and its per-case PCG64 streams. Record every attempted prefix, terminal
event and retained endpoint. The existing support floor of 64 retained cases
applies separately to each evaluation regime; no replacements are drawn.

SHIFT changes the correct-odor probability from 0.88 to 0.70 and each other
odor probability from 0.04 to 0.10. Transitions, hazards, cost semantics and
action distribution remain unchanged. The true transition remains doubly
stochastic. BASE and SHIFT have independent case streams, not paired noise
perturbations of the same case. Within a regime, all models see identical data.

There are no SHIFT updates, calibration, input noise-level hints, oracle
prefixes for learned models or model selection. Learned parameter hashes must
remain unchanged across both evaluations. Preserve cyclic shuffled-history
controls, the uniform-state known-dynamics reference and every prediction.

## Explicit exact references

The earlier exact-reference constructor hardcodes epsilon 0.12. Use a new
explicitly epsilon-bound, zero-parameter reference with owned world buffers.
Its state hash includes epsilon and its metadata identifies the actual world.
Use a 0.12 reference for TRAIN and BASE. Construct a separate 0.30 reference
only after the final-checkpoint barrier for SHIFT. Never mutate a global world,
the old reference constructor or learned models to change regimes.

Before numerical interpretation, check every exact target against its reference.
Qualify 0.12 parity with the old reference and 0.30 full observed probabilities,
conditioning and absorption against an independent rational/scalar calculation.
Blind-only agreement is insufficient because marginal transport is unchanged
by this observation shift. The saved-output auditor independently reconstructs
both endpoint targets and all-attempt prefix probabilities at the correct epsilon.

## Prospective continuation rule

Keep the local study's numerical thresholds and comparison logic unchanged.
Apply them separately to BASE and SHIFT with five binding seeds. Each regime
has **27 conditions**: three original rounded absolute criteria (SHORT, BLIND,
OBSERVED); 20 nonpositive paired H4/H8 regret differences against the two free
controls; and four mean regret reductions of at least 10%, with positive control
means. Both regimes must pass, for **54/54 overall conditions**.

An aggregate mean cannot rescue a losing seed, failed support or absolute
criterion. Neither regime can rescue the other. Report both individual results
and their overall conjunction, every model/seed/horizon, raw regret and paired
differences. Fit-time ratios and likelihood diagnostics remain descriptive,
not alternative continuation gates. Equal updates do not imply equal time,
FLOPs, gradient geometry or effective degrees of freedom.

## Qualification and process boundaries

Freeze source, runtime, this protocol, commands and full source snapshots before
one engineering qualification. Bind the exact published local-pass registration,
all original process closures and publication receipt, preserving its evidence
and the earlier failures. Qualify the new reference and regime integration with
the existing numerical/model/controller checks. Disable ambient pytest plugins,
bytecode writes and numerical multithreading.

Smoke data uses namespace 945001, seed 945101, 8 TRAIN and 8 attempts per DEV
regime, batch 3, three prefix and four joint updates, and cap 30 seconds. One
fresh exposure probe uses namespace 945201, seed 945301, 512 TRAIN and 8 attempts
per DEV regime, batch 64, 32 prefix and 64 joint updates, and cap 30 seconds.
Both run all three arms and audit saved results. They must not inspect namespace
435260924 or the five scientific fit seeds. Their effectiveness is not a model
selection signal.

Retain the same feasibility rule for every probe arm:
`2 * (prefix_stage_seconds * 1024/32 + joint_stage_seconds * 3072/64)`
plus non-stage complete-fit time must be strictly below 90 seconds. It is only
a conservative admission estimate; no counts or caps may be adjusted afterward.

Only after original qualification succeeds and closes, freeze the scientific
registration with identical source and settings. Native supervised phase caps
are qualification 300 seconds, fitting/evaluation 1,200 seconds and independent
audit 600 seconds. Keep sampled 4 GiB RSS and 512 MiB phase output bounds and
exclusive paths. A timeout, numerical failure, incomplete cleanup or failed
audit preserves the original attempt and stops it. These bounds are not a
claim of continuous resource enforcement.

The independent auditor reads saved arrays and metadata only. It reconstructs
targets, metrics, paired minibatches, accepted update exposure, disjoint shared
work and all 45 model plus 45 optimizer boundary files. It does not replay
learning or call a generator. Preserve original traces, all predictions,
checkpoints, source snapshots, receipts and closures in the published evidence.

## Interpretation and next decision

Publish either outcome, with every seed and both regimes visible. A pass would
support replication and transfer to this observation-noise change. Five fitted
seeds sharing one training cohort are not five independent population studies.
Known transition structure and privileged starting readouts remain substantial
limitations. No statistical significance, biological-wiring advantage,
calibrated text probabilities or architectural novelty is established here.

Before a broader architectural claim, the next registered test must challenge
the favorable transition prior and privileged initialization. A second
environment remains required. Any future change follows a separate protocol;
this run cannot tune itself on BASE or SHIFT outcomes.
