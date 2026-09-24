# Frozen-dynamics convex readout diagnostic v1

Prospective protocol for **finite-convex-readout-v1**, implementing the
[readout diagnostic proposal](finite-factorized-dynamics-next.md). This is a
TRAIN-only cost-head refit of nine completed models, not nine new randomized
fits or a continuation of their dynamics training. The parent factorized study
and earlier replication remain closed with their original failed criteria.
No favorable seed or mean can replace those verdicts.

## Frozen parents and data boundary

Retain every parent model: factorized, matched_free and dense_free, each with
fit seeds **426261001, 426261002 and 426261003**. Authenticate the original
factorized study registration, qualified source closure, successful original
producer/audit process closures, complete saved payload hashes and all nine
checkpoint identities before any numerical decode or model construction.
The parent's scientific failure is accepted evidence, not an admission failure
to bypass. No prior DEV array may be decoded for this diagnostic.

Reuse the original **426260924** TRAIN population, generated from 512 attempts
at base epsilon **0.12** with H1/H2 forecast labels. Its 471 eligible surviving
prefixes supply the endpoint population; its 41 terminal prefixes do not gain
invented endpoint labels. Reuse is explicit. Do not regenerate, replace,
resample or relabel TRAIN. Public prefixes preserve reset and up to eight
action/event pairs, including first found and padding. Only original public
prefixes and permitted forecast observations enter each frozen model.

New DEV uses namespace **427260924**, split 1, **128 attempted prefixes** and
H8 forecasts from the unchanged finite world and all-attempt collector. Each
attempt uses `PCG64(SeedSequence([namespace, split_id, attempt_index]))`.
Generate it only after all nine TRAIN solves and the durable solve barrier.
Retain all attempts without replacements; endpoint metrics use survivors.
This is fresh development after inspecting prior outcomes, not untouched final
confirmation or a scenario shift.

Engineering uses namespace **935001** and actual model seed **935101**.
Fabricated audit rules may use labels **935101-935103** without introducing
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

## Solver and independent numerical certificate

Use CPU float64 and a single fixed constrained solver, starting from the
checkpoint's original probability head. Runtime pins include SciPy **1.18.1**,
NumPy **2.5.3** and PyTorch **2.14.0**, plus exact interpreter and source
identities. No hyperparameter search, restart, alternate solver or DEV-based
termination is permitted. Preserve raw solver output, termination metadata,
iteration/function/gradient counts, elapsed time and final exported readout.

Use SLSQP with analytic objective/gradient, maximum **2,000 iterations**,
at most **10,000 objective calls**, and `ftol=1e-12`. Require SciPy's successful
termination status. Raw simplex violation must be at most **1e-10**. Apply
one declared deterministic simplex projection for export and require its
maximum entry change to be at most **1e-10**. Projection is part of the
declared algorithm, not an unrecorded repair or a remedy for a materially
infeasible solve. Preserve both raw and exported matrices and diagnostics.

Independently evaluate final primal error at most **1e-12** and Frank-Wolfe gap
at most **1e-8** on the original TRAIN rows, not only cached solver products.
Retain the signed computed gap; values below **-1e-12** fail the numerical
consistency check rather than being silently clipped to zero.
For gradient G at a feasible P, the gap is

`gap(P) = sum_s (sum_a G[a,s] * P[a,s] - min_a G[a,s])`.

For this differentiable convex objective, the exact-arithmetic gap bounds
`F(P) - F(P*)`. State the float64 tolerances and rounding limitations rather
than claiming an exact proof. Require final objective at most the original
head's objective plus **1e-12**, and check the final certificate after
projection. A failed status, nonfinite quantity,
feasibility failure, certificate failure or exhausted budget fails the solve;
do not export it as a best readout or silently restart. No fresh DEV generation
is admitted unless all nine solves satisfy the frozen technical contract.

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
validation products and **16,956 validation state rows**. Nine completed
solver calls each retain three direct objective/gradient passes and one
simplex projection, totaling **27** such direct passes and **nine** projections.
Record actual SLSQP objective, gradient and iteration-callback counts; each
combined objective/gradient evaluation increments both counters. SciPy's
reported counters are retained separately from callback work.

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
model construction, frozen-state extraction, solver, certificate/projection,
validation and evaluation costs separately inside whole-phase timing. Do not
add nested timings twice or describe cached-head arithmetic as end-to-end
inference speed.

Preserve failed engineering attempts and source snapshots. Any repair needs
a new exclusive engineering registration. Science has one attempt, without
retry, extra iterations, cap extension, replacement cases or another head
after a failure. Retain all nine original heads, all solved heads and all
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
