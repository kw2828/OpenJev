# Does recurrent performance depend on a task-derived starting head?

This is a separate diagnostic after the [five-seed replication](finite-reuse-replication-results.md)
failed its continuation rule with 45/54 conditions passing. Its large mean improvements did not hold
for every fit, and noisier observations broke probability accuracy. This study
does not reopen that result or claim to resolve its noise-transfer failure.

## Intervention and matched controls

Fit three arms for each of five fresh seeds:

1. `rounded_anchor`: the unchanged rounded transport and task-derived learned-head
   initialization from the earlier study.
2. `rounded_random`: the identical rounded transport, with task-independent
   initial logits for the learned head.
3. `matched_free_random`: the initially function-matched column-stochastic
   transport with exactly the same random head as `rounded_random`.

All arms retain 352 stored float64 parameters and the existing linear readout
of surviving mass. For both random arms, replace only `cost_logits` with a
4-by-8 array from
`numpy.random.Generator(PCG64(SeedSequence([fit_seed, 436, 1]))).standard_normal((4, 8))`.
The distribution, unit scale and stream are fixed before qualification. Do not
center, rescale, permute, select or tune these draws. They are independent of
task cost templates and do not advance the global NumPy or Torch RNG. Supervised
cost labels and the readout parameterization remain task information.

The existing constructor can transiently create its old head before replacement;
that head must not survive in the random arms' parameters, optimizer, buffers
or effective model metadata. Report this explicitly and charge all construction,
random draws, copies and metadata to complete fitting time. The anchor is a
diagnostic reference, not a fallback candidate. Column shuffling alone would
retain the known cost-template structure and is not this intervention. Generic
random logits also change optimization scale, so a result concerns the complete
initialization intervention rather than latent-label alignment alone.

Match initial emission and hazard parameters across all arms. Match rounded
raw transition parameters between anchor and random arms, and match their
transition probabilities to the free control within the unchanged 1e-12 tolerance.
The prefix objective excludes the head: rounded anchor and random arms must
have byte-identical dynamics and prefix-Adam state at both initial and prefix
boundaries. Every head must remain byte-identical to its own initial head through
the prefix stage. Check these properties before DEV exists and independently
again from saved boundaries. A mismatch is an integrity failure, not a result.

## Fixed training and fresh data

Use namespace **436260924**, seeds **436261001 through 436261005**, 512 TRAIN
attempts from split 0 and 512 BASE attempts from split 1. Both use epsilon 0.12.
The world, public prefix generation, hazards, actions and cost targets are unchanged.
Retain every attempt, including early found events. Require at least 256 eligible
TRAIN cases and 64 eligible BASE cases through the unchanged absolute criteria.
There are no replacements or old cases/checkpoints.

Each fit completes exactly **1,024 prefix updates and 3,072 joint updates**,
using the qualified per-call computation reuse. Keep batch size 64, learning
rate 0.003, clipping at 5, prefix prior 0.001, no joint prior, fresh joint Adam
and joint batch cursor zero. Pair same-seed ordered minibatches across arms.
Rotate the three-arm execution order by seed index. Do not change the objective,
parameter ordering, optimizer or gradient arithmetic.

The 120-second per-fit cap is a failure boundary, not a variable training budget.
Construction, validation, updates, model/optimizer saves and final summary are
charged; complete fit time also includes durable allocation serialization.
Save all 45 model and 45 optimizer initial, prefix-boundary and final artifacts.
All 15 final checkpoints must exist before BASE generation. Evaluate H1/H2/H4/H8
on unchanged models, preserving shuffled-history and uniform-state references.
The explicit epsilon-0.12 zero-parameter reference validates every target;
learned models receive no oracle state. BASE is diagnostic development data,
not a final held-out TEST claim.

## Prospective decision rule

The candidate is `rounded_random`; its only primary control is
`matched_free_random`. Apply **15 binding conditions**:

- All three unchanged absolute criteria: SHORT, BLIND and OBSERVED.
- All ten H4/H8 candidate-minus-control regret differences, across the five
  fit seeds, must be nonpositive.
- Mean regret must improve by at least 10% at each horizon, with positive
  control means.

Every condition must pass. Report absolute results for every arm, raw regret,
all paired differences and complete costs. Comparisons to `rounded_anchor` are
descriptive and cannot rescue a failure. No pooled mean or successful seed can
rescue another seed. Timing is descriptive and is not an alternative gate.

If SHORT fails, stop continuation and report that this initialization did not
establish basic supervised learning under the fixed recipe. Do not interpret
that alone as a long-horizon transport defect. If absolute criteria pass but
the matched contrast fails, superiority without the task-derived starting head
remains unsupported. A pass supports only this local diagnostic and does not
repair the earlier noise-shift failure. It can motivate a separately registered
nonuniform-transition study and then a second environment. No such study is
admitted by this registration, and no extra updates or initialization search
are permitted in this experiment.

## Qualification, audit and evidence

Bind this protocol, source and complete runtime before one engineering phase.
Preserve and authenticate the preceding failed replication, including its original
closures and publication. Snapshot every frozen source. Disable ambient pytest
plugins, bytecode writes and numerical multithreading.

The smoke profile uses namespace 946001, seed 946101, 8 TRAIN and 8 BASE attempts,
batch 3, three prefix and four joint updates, cap 30 seconds. The single exposure
probe uses namespace 946201, seed 946301, 512 TRAIN and 8 BASE attempts, batch 64,
32 prefix and 64 joint updates, cap 30 seconds. Both run all three arms and audit
their saved outputs. Neither can inspect scientific namespace 436260924 or its
fit seeds; engineering effectiveness cannot select a model or initialization.

For each exposure-probe arm, require
`2 * (prefix_stage_seconds * 1024/32 + joint_stage_seconds * 3072/64)`
plus non-stage complete-fit time to be strictly below 90 seconds. This unchanged
feasibility estimate does not guarantee runtime or justify changed update counts.

Only original successful qualification and clean process closure permit the
scientific registration. Native phase caps are qualification 300 seconds,
fitting/evaluation 1,200 seconds and independent audit 600 seconds. Keep sampled
4 GiB RSS and 512 MiB phase-output bounds and exclusive paths. Preserve partial
work and stop on failure, timeout, cleanup failure or source change. No retry,
replacement seed, tolerance relaxation or evaluation-driven change is allowed.
Sampled checks are not continuous resource enforcement.

The independent auditor authenticates metadata first, reconstructs targets and
metrics from saved arrays, checks all checkpoint/Adam boundaries and ordered
updates, and accounts for actual shared forward work. It invokes no model,
optimizer or world generator. Reconstructing the 32-number initial random head
per fit seed is explicitly counted as initializer reconstruction, separate from
those zero-call statements. Publish either outcome with every fit, complete
source snapshots, predictions, original process closures and evidence receipts.

Five seeds share one training cohort. Equal parameters and updates do not mean
equal compute, optimization geometry or effective transition dimensions. The
uniform reset and doubly stochastic transition assumptions still favor this
world. This study establishes neither architectural novelty, biological benefit,
calibrated text decisions, noise robustness nor ICLR readiness.
