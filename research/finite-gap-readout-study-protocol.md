# Frozen-dynamics gap-certified readout study v1

Prospective protocol for **finite-gap-readout-study-v1**, following the
[separate numerical prerequisite](finite-convex-readout-next.md). This is a
TRAIN-only cost-head refit of nine completed models, not nine new randomized
fits or a continuation of their dynamics training. The parent factorized study
and earlier replication remain closed with their original failed criteria.
The stopped SLSQP study also remains closed, with its nine reported solves,
three passes, six certificate failures and zero DEV generation preserved.
Its reserved namespace 427260924 is not reused. This successor is a separately
registered attempt, not a repaired run or a retrospective tolerance change.
No favorable seed or mean can replace any prior verdict.

## Frozen parents and data boundary

Retain every parent model: factorized, matched_free and dense_free, each with
fit seeds **426261001, 426261002 and 426261003**. Authenticate the original
factorized study registration, qualified source closure, successful original
producer/audit process closures, complete saved payload hashes and all nine
checkpoint identities before any numerical decode or model construction.
The parent's scientific failure is accepted evidence, not an admission failure
to bypass. Also authenticate the completed synthetic solver qualification and
the stopped predecessor's original failed process, source and complete file
inventories. Their arrays are opaque historical evidence, not new solver
inputs. No prior DEV array or saved predecessor latent-state array may be
decoded for this study. Extract TRAIN states anew from the authenticated
original checkpoints and original TRAIN only.

Reuse the original **426260924** TRAIN population, generated from 512 attempts
at base epsilon **0.12** with H1/H2 forecast labels. Its 471 eligible surviving
prefixes supply the endpoint population; its 41 terminal prefixes do not gain
invented endpoint labels. Reuse is explicit. Do not regenerate, replace,
resample or relabel TRAIN. Public prefixes preserve reset and up to eight
action/event pairs, including first found and padding. Only original public
prefixes and permitted forecast observations enter each frozen model.

New DEV uses namespace **428260924**, split 1, **128 attempted prefixes** and
H8 forecasts from the unchanged finite world and all-attempt collector. Each
attempt uses `PCG64(SeedSequence([namespace, split_id, attempt_index]))`.
Generate it only after all nine TRAIN solves and the durable solve barrier.
Retain all attempts without replacements; endpoint metrics use survivors.
This is fresh development after inspecting prior outcomes, not untouched final
confirmation or a scenario shift.

Engineering uses namespace **936001** and actual model seed **936101**.
Fabricated audit rules may use labels **936101-936103** without introducing
additional scientific fits and **934001** as a synthetic prior-lineage
namespace. That label does not permit reading a previous engineering dataset.
No scientific checkpoint or TRAIN array is used as an engineering fixture.

No hidden state, true posterior, belief array or true operator is a learner
input. The parent models already embody task-derived factorization or free
dynamics, a known uniform reset prior, and world-aligned initial heads. Those
privileges remain disclosed. An exact target-consistency reference alone may
receive the true boundary posterior and operators.

## Nine frozen-state problems and eighteen evaluation views

Load each parent checkpoint strictly into its original model class and keep
every parameter unchanged. Use `no_grad`, no optimizer and before/after full
state hashes; retain the original `requires_grad` flags because the inherited
model guards require them. Extract float64 blind and observed **prior** states at H1/H2
from the original TRAIN public histories. Blind states are unnormalized
surviving probability mass. Observed forecasts precede the current event;
only earlier allowed events may be assimilated. After known found, the state
and cost are zero. Do not insert an oracle posterior or future observation.

For each checkpoint retain two cost readouts: its unchanged original head and
one newly solved matrix. Dynamics, reset/emission law, state transitions,
event predictions and survival predictions are common to both views. The
solved matrix is applied directly to state; do not convert boundary values to
infinite logits or change any parent parameter. Hash parent state before and
after extraction and evaluation. There are nine solves and eighteen paired
original/solved model views, not eighteen separately trained dynamics models.

Preserve nine TRAIN state files, one per checkpoint. For DEV, replay each
original and solved view independently through the same frozen model. Save
ten state arrays per checkpoint: original/solved blind, observed and shuffled
prior states with shape `[N,H,8]`, plus original/solved prefix and shuffled
prefix states with shape `[N,8]`. Require bitwise equality for each of these
five state pairs and all common event/survival predictions. A parity failure
is a technical failure, not an empirical effect of the readout intervention.

Write the readout as

`C[:,s] = 0.25 - P[:,s]`, with `P[:,s] >= 0` and `sum_a P[a,s] = 1`.

There are 32 stored coefficients and 24 simplex degrees of freedom. Bounds on
C are [-0.75, 0.25], its action columns sum to zero, and no intercept is added.
For a zero absorbed state, C produces zero cost. Linear expectation commutes
with this readout; do not evaluate a nonlinear head on a mean state.

With S eligible TRAIN cases, two forecast horizons and four actions, minimize

`F(P) = sum_blind (C z - y)^2 / (S * 2 * 4)
      + sum_observed (C z - y)^2 / (S * 2 * 4)`.

Preserve all original action/horizon entries, including absorbing zero suffixes,
and both unit-weight cost terms. This is the full-population cost portion of
the original global-denominator objective. With dynamics frozen, survival
MSE, observed event cross-entropy and all-attempt prefix NLL are constants.
There is no reweighting by survival, conditional sample count or batch size,
and no H4/H8 target enters a solve.

The quadratic is convex in P. Rank deficiency may make the optimizer
nonunique; no unique latent basis or uniquely identified readout is claimed.
The closed simplex permits exact vertices that finite softmax logits only
approach. Any benefit combines head refitting with this closure difference;
it cannot be attributed solely to an optimizer change or greater information
in the frozen state.

## Qualified algorithm and independent numerical certificate

The completed, separately registered synthetic qualification retained all 18
fixtures and both solvers: **18/18** candidate solves and **10/18** unchanged
SLSQP solves met their independent scalar checks. Original phase time was
**3.042223125 seconds**, including setup, lint, 73 fabricated tests, all 36
solves, checks and cleanup. Registration SHA256 is
`3513fd540b1a01af6568a4268a8ea7e0a93680ad39ef35ddb206d998cbf255eb`.
Its 12 source pins and 58 payloads are retained. This is bounded engineering
evidence on a predetermined synthetic suite, not proof of reliability on the
nine scientific problems or a learning result. Zero-support and tiny-mass
fixtures can qualify at initialization. The new study requires its own
source-bound integration qualification before scientific execution.

Use CPU float64 and the qualified `finite_gap_readout.solve`, starting from
the checkpoint's unchanged original probability head. Runtime pins include
NumPy **2.5.3**, PyTorch **2.14.0** and SciPy **1.18.1** for the retained
reference/testing dependency, plus exact interpreter and source identities.
The candidate does not call SciPy. There is no SLSQP success requirement,
`ftol`, maximum objective-call setting or one-time export repair in this study.
No solver search, alternate method or DEV-based termination is permitted.

The fixed algorithm is projected FISTA with a monotonicity restart. Set
`L = 2 * max_i sum_j abs(Gram[i,j])` for the original sum-of-two-MSE quadratic,
and use step `1/L`. Each gradient step is projected onto the eight independent
four-action simplexes. If the proposed quadratic exceeds the current value by
more than **1e-15**, discard that proposal, reset momentum and perform one
ordinary projected-gradient step from the current iterate. If that step also
exceeds the numerical tolerance, return failure. Charge both trials. This
predeclared within-solve restart is part of the algorithm; it is not a retry
of a failed solve. No line search, backtracking or parameter retuning occurs.

Maximum iterations are **20,000**. Assess the direct residual certificate at
iteration zero, every **10** accepted iterations and the final retained
iterate, without a duplicate assessment at the same iteration. For zero
curvature, retain the initial point and require its direct certificate. A
small update, constant iterate or exhausted budget never substitutes for
certification. Preserve the complete assessment history and termination
reason and every completed failed solve result. Rejected trial work is counted;
this does not claim that every discarded numerical iterate is serialized.

Require primal simplex error at most **1e-12**, signed Frank-Wolfe gap between
**-1e-12 and 1e-8**, and direct objective at most the original head objective
plus **1e-12**. Keep the signed computed gap instead of clipping a negative
inconsistency. For gradient G at feasible P,

`gap(P) = sum_s (sum_a G[a,s] * P[a,s] - min_a G[a,s])`.

The exact-arithmetic gap bounds `F(P) - F(P*)` for this differentiable convex
objective. Float64 assessment is a numerical certificate, not an interval
proof. A separate saved-output auditor recomputes the final loss, full
gradient and gap directly from the saved original TRAIN rows, not only cached
Gram products or the producer's certificate. Preserve both original and
solved heads. Returned raw and final probabilities are identical copies of
the retained feasible iterate; no projection or export repair follows the
final certificate. A failure in feasibility, nonincrease, certificate,
numerical arithmetic or budget prevents admission. All nine precommitted
solves must succeed before any fresh DEV generation. Preserve unsuccessful
results; do not select passing parents or silently retry.

## Saved artifacts and explicit work accounting

The producer retains **38 payloads before summary.json**. Eleven common files
are config.json, train.npz, solves.jsonl, solve-barrier.json, base.npz,
base-prefix.npz, base-oracle.npz, oracle-base.npz, oracle-base-check.json,
prediction-times.jsonl and predictions-base.npz. Each of nine parents adds
one states-train file, one solve JSON record and one states-base file. Preserve
the full solver result in both its individual record and the complete journal,
including raw/final matrices, gradient, signed gap and original objective.

Producer numerical decoding is exactly **one original TRAIN file and nine
original checkpoints**. Construct nine original model instances, strictly load
their saved states, and construct one separate zero-parameter exact reference.
Constructor initialization is overwritten by the authenticated checkpoint; it
is not a new trained fit. Export nine original probability heads with nine
head softmaxes. Do not write new neural checkpoints. Model optimizer updates,
external model calls, native calls and teacher calls are all zero.

At batch size 64, original S=471 requires eight batches per parent and route:
**144 TRAIN forwards** over blind/observed routes, 144 explicit readout
validation products and **16,956 validation state rows**. Build nine quadratic problems and perform
nine candidate solves. Solver work is variable and reported for every parent:
iterations I, accepted iterations, monotonicity restarts R, Gram-gradient calls,
quadratic-value calls, update projections, direct certificate passes K and
budget callbacks. For a successful solve, accepted iterations equal I,
gradients and projections equal I+R, and quadratic-value calls Q equal
`1+I+R` if I>0 or zero if the initial certificate passes. Budget callbacks
are `2+K+Q+I+2*(I+R)`. K is the exact saved assessment-history length, covering
zero, each tenth iteration and a distinct final iterate when needed. A
successful fixed-20,000-iteration production solve terminates at zero or a
multiple of ten; qualification also tests off-schedule final assessment.
There is no fixed three-pass or final-projection count. These operation counts
are not exhaustive FLOPs; construction and raw-row certificates have different
costs from cached Gram arithmetic.

For N eligible fresh DEV cases, let K=ceil(N/64). Eighteen head views each
perform blind, observed and shuffled routes, giving **54*K frozen forwards**,
the same number of explicit head products, **432*N mapped state rows** and
**18*N evaluated case views**. With 65-128 eligible cases this is 108 forwards.
The exact target reference adds K blind and K observed forwards separately.
Keep each of the three families' eight forward-work routes: TRAIN blind and
observed, plus both heads' blind, observed and shuffled DEV routes. Structural
counters are not exhaustive FLOPs.

Solved views still execute the original model's cost head during their frozen
forward, then apply the exported solved matrix to saved priors. Original views
also calculate an explicit map for validation. Report this extra work rather
than claiming that either timing represents optimized single-decision latency.
The original-head map must agree with the inherited forward within 1e-12.

The independent audit decodes **24 numerical files**: TRAIN; fresh DEV data,
all-attempt prefix data, exact boundary posterior, exact-reference predictions
and learned predictions; and eighteen TRAIN/DEV latent-state files. It decodes
no checkpoint and makes zero learned-model, generator, solver, optimizer,
native or teacher calls. Original checkpoints, old DEV artifacts and source
closures may be hashed as opaque bytes without numerical decoding. All public
target and metric reconstruction is local saved-record arithmetic.

## Evaluation, audit and interpretation

Evaluate every original/solved pair on the same fresh eligible DEV histories
at H1, H2, H4 and H8. Preserve all **72 endpoint rows**: three parent families,
three seeds, two heads and four horizons. Retain each pair's cost MSE and
decision regret, common observed event KL, survival errors, support,
parameter/storage metadata and measured costs. Retain **36 paired contrasts**
across the nine checkpoints and all four horizons. Of these, six H4/H8
contrasts per family give **18 longer-horizon paired contrasts**.

Apply the original three absolute criteria separately to each family/head
group across all three parent fit seeds. Require at least **256 eligible TRAIN**
and **64 eligible DEV** cases and positive required uniform-reference costs.
SHORT requires H1/H2 blind cost MSE and regret at most half the uniform
reference and observed KL at most 0.1 nats. BLIND requires the same H4/H8 cost
thresholds and H8 survival MAE at most 0.05. OBSERVED requires H4/H8 event KL
at most 0.1. Preserve their original 24, 21 and 8 conditions; no relative gain
or mean rescues an all-seed failure. Observed filtering uses intervening
evidence and is distinct from forecasting without observations.

Use raw argmin for every learned original/solved head. Only the uniform-state
known-dynamics reference uses the prior lowest-index 1e-12 tie rule. Report
all seeds and signed paired differences without retrospective significance
tests, selected-seed averages or ensemble claims. Recompute common event and
survival outputs and require bitwise equality; unchanged non-cost metrics
are controls, not newly learned improvements. All-attempt prefix likelihood
is not rerun. Its functional invariance follows from unchanged dynamics and
the absence of a cost-head path, not a new numeric prefix-NLL recheck.

Close the original producer before an independent saved-output audit. It
must reconstruct targets, public-event chronology, both readouts' costs,
scalar metrics, all criteria, and the nine TRAIN objective/gradient/gap
certificates without running a solver or learned model. Saved latent states
must carry source/checkpoint/route identity; their correspondence to actual
model execution remains source-qualified attestation. Byte hashes alone do
not prove historical causal execution. Never reuse producer metrics or its
reported certificate as the independent numerical oracle.

Freeze a complete transitive source and dependency closure, exact artifact
roster, declared decode/construction/forward/readout/solver counters and
original-process joins before engineering qualification. Qualification uses
fabricated data only, including rank-deficient and ill-conditioned states,
boundary optima, chronology, absorption, head parity and rejection-before-DEV.
No scientific checkpoint or TRAIN array is an engineering fixture.

Use the original suspend-inclusive supervisor with **300 seconds
qualification, 600 seconds extraction/solve/evaluation, 600 seconds audit**,
**4 GiB worker RSS**, **512 MiB phase output**, one CPU thread and no accelerator,
native environment, external model or paid inference. Record checkpoint load,
model construction, frozen-state extraction, solver, certificate/update-projection,
validation and evaluation costs separately inside whole-phase timing. Do not
add nested timings twice or describe cached-head arithmetic as end-to-end
inference speed.

Preserve failed engineering attempts and source snapshots. Any repair needs
a new exclusive engineering registration. Science has one attempt, without
retry, extra iterations, cap extension, replacement cases or another head
after a failure. Predeclared momentum restarts inside a live solve are counted
algorithm steps, not permission to rerun a closed failure. Retain all nine original heads, all solved heads and all
failures. No old source or registered parent artifact may change.

Improved TRAIN costs without fresh decision improvement leave the failure
unresolved. Consistent improvement would support a readout limitation under
this particular constrained family, including the boundary extension. A
remaining failure does not prove that latent information is absent, nonlinear
heads cannot help, or optimization has converged. No outcome by itself
establishes architecture novelty, calibration, connectome learning, RL,
scenario-shift robustness or transfer to Doom, chess or robotics. The earlier
OTTO direct-readout study concerns a different model/task and retains its own
verdict. Any follow-up still needs a separate frozen protocol.
