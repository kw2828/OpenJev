# Factorized recurrent dynamics diagnostic v1

Prospective protocol for **finite-factorized-dynamics-v1**, implementing only
the [structural-stability proposal](finite-cost-readout-replication-next.md).
The [readout replication](finite-cost-readout-replication-results.md) failed
its continuation rule despite all six relative regret gains. That study stays
closed, and its proposed H4-supervision follow-up remains unadmitted. This
new comparison tests a task-informed dynamics factorization while retaining
the learned readout, current H2 loss and fixed optimization schedule.

## Task structure, fresh data and information boundary

Use the unchanged eight-state finite world with four actions, four ordinary
observations and absorbing found, at base sensing epsilon **0.12**. Scientific
namespace **426260924** uses TRAIN split 0 with **512 attempts** and H2
forecasts, then DEV split 1 with **128 attempts** and H8 forecasts. Attempts
use `PCG64(SeedSequence([namespace, split_id, attempt_index]))`. New engineering
uses namespace **934001**, actual model/fit seed **934101**, and fabricated
audit labels **934101-934103** for three-seed rules. No engineering fixture
uses a scientific seed.

Reuse the frozen all-attempt collector. Preserve reset plus up to eight public
action/event pairs in nine 31-wide float32 rows, including the first found row,
then zero padding with lengths, event masks, endpoint eligibility and survivor
mapping. All attempts contribute prefix likelihood; only prefixes surviving
all eight actions receive endpoint forecasts. Commit the forecast action block
before its observations. No replacement cases, old arrays, old checkpoints,
warm starts or selected successful seeds are allowed.

The [world source](../src/openjev/research/finite_observation_world.py) specifies
action-conditioned transitions, a hazard dependent on action and next state,
and ordinary emissions dependent only on next state. Reset uses the same
ordinary emission law without a transition or hazard. This conditional
structure is supplied task knowledge, not discovered by training. Its actual
transition, emission and hazard values must never initialize a learner.

Every learned arm receives identical public histories and target labels, with
no hidden state, oracle posterior, belief array or true dynamics input. Uniform
reset prior, eight-coordinate state size and the world-aligned initial cost
head are shared privileges. Targets use analytic finite-world laws up to
float64 roundoff, not a claimed bit-exact equivalence to finite sampler bins.

## Three arms, one shared learned readout

Let s be current state, n next state, a action and o an ordinary observation.
For the factorized arm, learn a transition T, emission O and hazard h:

`B[a,o,n,s] = O[o,n] * (1 - h[a,n]) * T[a,n,s]`

`found[a,s] = sum_n h[a,n] * T[a,n,s]`.

T has shape **4 by 8 by 8**, normalized over n for each (a,s); O has shape
**4 by 8**, normalized over o for each n; h has shape **4 by 8**, strictly
between zero and one. Therefore each column of B together with its found
probability sums to one. The blind survival operator is `A = sum_o B`.
Share O between reset and subsequent observations in this arm.

| Arm | Dynamics and reset | Trainable parameters |
| --- | --- | ---: |
| factorized | Learned T, O and h, with shared reset/emission O | 352 |
| matched_free | Unrestricted branch/reset model initialized to the factorized arm's actual B, found and O | 1,120 |
| dense_free | Unrestricted branch/reset model with the existing independent dense initialization | 1,120 |

The factorized model stores 256 transition, 32 emission, 32 hazard and 32
cost logits. Each unrestricted model stores 1,056 branch, 32 reset and 32
cost logits. All parameters/operators/states are float64 on CPU. Parameter
storage is **2,816 bytes** for factorized and **8,960 bytes** for each free
model; report actual buffers separately. Optimizer state, gradients,
activations and operator products are additional. Fewer parameters alone do
not establish lower measured cost or a speedup.

Every arm uses the unchanged bounded readout
`C_L[:,s] = 0.25 - softmax_action(L[:,s])`, initialized with
`L[:,s] = log(0.9 * onehot(g(s)) + 0.1/4)`. All initial heads match the fixed
softened function: preferred cost -0.675, other costs 0.225. The 32 head logits
have 24 identifiable degrees of freedom. The readout is linear in state,
without bias: `C_L @ E[b] = E[C_L @ b]`. It maps absorbed zero state to zero.
Finite logits cannot exactly attain the true cost vertices in real arithmetic;
float64 subtraction may round a valid cost to a boundary. Preserve closed-bound
validation with 1e-12 tolerance and no clipping; reject nonfinite values and
underflow rather than silently repairing them.

The true positive dynamics lie in the factorized family at finite logits.
This does not make the bounded learned cost head exactly represent the true
vertices, and it does not imply identifiable latent coordinates or easy
optimization. Both model classes retain the same readout limitation.

## Initializer and matched-function control

Use local float64 CPU normal draws and fixed domain-separated streams:

- Transition logits: `0.05 * Z_T`, shape 4 by 8 by 8, seed
  `fit_seed XOR 0x9E3779B9`; softmax over next states.
- Emission logits: `0.05 * Z_O`, shape 4 by 8, seed
  `fit_seed XOR 0x85EBCA6B`; softmax over ordinary observations.
- Hazard logits: `-log(32) + 0.05 * Z_h`, shape 4 by 8, seed
  `fit_seed XOR 0xC2B2AE35`; sigmoid elementwise.

The nominal hazard 1/33 is inherited from the free model's flat 33-outcome
initializer, not copied from the world's true hazard. These seeded domains
are not a mathematical independence claim. Dense transitions remain allowed
to contract history; no sticky or retentive initialization is added.

Initialize matched_free from the factorized model's actual materialized
operators: branch logits
`log(concat(B.reshape(4,32,8), found[:,None,:]))` and reset logits `log(O)`.
The flattened order is ordinary observation then next state, followed by
found, matching the existing free model. This matches initial B/found/O and
the complete predictive function within 1e-12. The free parameters then evolve
without the factorization or reset/emission-sharing constraint.

Dense_free preserves the existing 0.05-normal branch logits, shape 4 by 33 by 8,
and reset logits, shape 4 by 8, with the existing corresponding XOR domains.
All three learned heads initialize identically. Preserve parameter group
hashes and actual initial operator/reset/readout snapshots. Initial function
matching does **not** require raw gradients, clipped gradients or updates to
match across different parameterizations. All parameters share global clip 5,
so gradient geometry is part of the declared intervention.

## Unchanged prediction chronology and objective

Prefix NLL has coefficient **1** in every arm and includes every valid event.
Predict the four-way reset law from the uniform prior before action/hazard,
score the reset observation, then condition. Each later action predicts the
five-event law before assimilating its event. Normalize ordinary posteriors;
score found once and end the prefix. Padding contributes zero. Positive
observed events with zero predicted probability fail explicitly.

Eligible endpoint cases use the unchanged H1/H2 four-component objective:
blind centered-cost MSE, observed centered-cost MSE, half the sum of blind and
observed survival MSE, and observed five-event soft cross-entropy. Preserve
the existing reductions, unit weights and cost scale 1. Endpoint denotes the
post-prefix forecast objective, not final-horizon-only supervision. Blind
state is unnormalized surviving mass. Observed forecasts precede the current
event; after known found, cost/survival are zero and the event law is found with
probability one. Conditional observed survival is not blind survival mass.

Freeze attempted TRAIN count `N = 512`, eligible TRAIN count `S` and total
valid TRAIN events `E` before fitting. For actual batch size b, retain

`(N / b) * (sum_eligible endpoint_loss / S + sum_valid event_NLL / E)`.

Use global S/E and actual b, including partial engineering batches. Do not use
random batch denominators or averages of per-sequence NLL means. No-endpoint
batches retain their prefix events and are not resampled. Explicit zero
gradients for parameters without a current loss path preserve the existing
Adam recipe; stored moments may still move those parameters. Successive
without-replacement batches are not independent optimizer draws.

## Fixed schedule, barrier and measured work

Fit seeds **426261001, 426261002 and 426261003** give **nine fits**. Every fit
uses **480 epochs**, batch size **64**, Adam at **0.003** with default
betas/epsilon and global norm clipping at **5**. Pair attempt permutations with
`PCG64(SeedSequence([fit_seed, epoch, 818]))`; rotate the three-arm execution
order by seed index. No early stopping, schedule, warm start, restart,
checkpoint selection or best-seed selection is allowed.

The fixed schedule totals **34,560 optimizer updates**, **2,211,840 attempt
exposures**, `9 * 480 * S` eligible endpoint exposures and `9 * 480 * E`
prefix-event exposures. All nine final checkpoints and a durable nine-fit
barrier precede any DEV generation or decoding.

Before the first learned fit, the zero-parameter exact/exact reference must
match all five eligible TRAIN target fields within 1e-12; repeat after DEV
generation. Only that target-consistency reference receives true boundary
posteriors and operators. Retain the history-ignorant, known-dynamics
uniform-state reference; it is not an optimal history-ignorant policy or a
learned control.

Retain initial and final readout and operator/reset snapshots for all nine
fits, with owned arrays, exact dimensions and canonical hashes. Engineering
qualifies initial factorized/matched_free B/found/O, readout and full-function
agreement within 1e-12 before scientific training. Each scientific fit records
its actual initial snapshots before its own updates; the independent audit
later compares the saved paired snapshots. There is no additional runtime
pairwise gate or uncounted export before fitting. Parameter-byte hashes need
not match. Record all 18 readout
snapshots and their 18 learned-head softmax evaluations separately from
forward work, plus 18 dynamics snapshots. Model initialization and matching
conversion have separate `construction_seconds` and construction-work records;
they are outside each fit's training duration but inside the original phase.
Operator/reset construction for snapshots and snapshot validation are included
in training duration. Report both timing scopes and their actual counters.

Report per-arm, per-route construction and propagation work across training
prefix/blind/observed, evaluation prefix/blind/observed/shuffled routes. Include
transition/emission softmax, hazard sigmoid, factor products and found sums
where executed, along with shared head costs, state propagation, zero-endpoint
batches, storage and measured training/inference durations. Counters are not
exhaustive FLOPs. Equal data, epochs or optimizer updates do not imply matched
computation. Preserve all nine fit durations and original phase costs without
adding nested times twice.

## Unchanged absolute criteria and complete reporting

Evaluate every arm and seed at H1, H2, H4 and H8. Require at least **256
eligible TRAIN** and **64 eligible DEV** cases, separate from attempted counts,
and strictly positive required uniform-reference cost MSE and regret. Apply
the original criteria separately to all three arms; every applicable condition
must hold for every seed.

| Criterion | Conditions required for every seed |
| --- | --- |
| SHORT_HORIZON_LEARNING | At each of H1/H2, blind cost MSE and regret are at most half the uniform reference; observed event-law KL is at most 0.1 nats. |
| BLIND_EXTRAPOLATION | At each of H4/H8, blind cost MSE and regret are at most half the uniform reference; H8 blind survival MAE is at most 0.05. |
| OBSERVED_FILTERING_EXTRAPOLATION | At each of H4/H8, observed event-law KL is at most 0.1 nats. |

These retain 24, 21 and 8 conditions per arm including support/reference checks.
If factorized fails any criterion, it cannot advance as a stable candidate
based on favorable means, isolated seeds or likelihood improvements. Passing
all three does not prove superiority over controls or automatically admit
another study. There is **no additional relative-promotion gate** in this
diagnostic and no alteration to the earlier replication's rule.

Regret uses exact expected cost of the selected action minus the minimum.
Models use raw argmin; only the uniform reference uses the lowest-index tie
within 1e-12. Retain every **36** arm/seed/horizon row and all **12** paired
H4/H8 contrasts of factorized against matched_free and dense_free, including
regret, blind/observed cost MSE and observed KL. Report prefix event-mean NLL,
support, storage, timings and complete conditions. Arithmetic means are over
separately fitted policies, not ensembles or confidence intervals. Signed
contrasts are descriptive, without a retrospective significance criterion.

Preserve offset-1 cyclic shuffling of complete public prefix and length within
eligible DEV cases, with forecast actions/targets unchanged. It is a diagnostic,
not a selection or significance test. Observed filtering receives intervening
evidence and must remain distinct from blind-gap performance.

## Qualification, closure and interpretation

Freeze protocol, complete transitive source closure, runtime, paths,
configuration and successful engineering evidence before scientific generation.
Only new adapters/wrappers may be added; previous pinned sources remain
unchanged. Qualify fabricated and declared engineering cases only. Verify the
factorized law's normalization, reset without hazard, terminal absorption,
prediction-before-conditioning, linear cost expectation and strict guards.
Check full factorized/matched_free initial prefix states, event probabilities,
blind/observed costs and survival within 1e-12, model ownership/gradients,
global loss denominators, oracle-input boundaries and the nine-fit DEV barrier.
Do not substitute gradient or optimizer-update parity for function parity.

Use the inherited original suspend-inclusive supervisor: **300 seconds
qualification, 1800 seconds fit/evaluation, 600 seconds audit**, **4 GiB
worker RSS**, **512 MiB phase output**, one CPU thread and no accelerator,
native environment, external model or paid inference. Preserve every failed
engineering attempt and source snapshot; repairs require a new engineering
registration and exclusive paths. Science has **one attempt**, with no retry,
cap extension, replacement cases or outcome-dependent extra fitting.

Close the original producer before independent saved-record audit. Authenticate
original process joins, unchanged sources and payload hashes before numerical
reconstruction. Audit targets, public-event likelihoods, scalar metrics,
schedules, snapshot arithmetic and declared work without executing learned
models or replaying optimization. Actual historical causal execution remains
source-qualified attestation, not proved by saved hashes alone. Keep technical
completion separate from scientific pass/fail and retain all final checkpoints.

An improvement over initially matched_free concerns the bundled structural
constraint, parameterization and reduced capacity. Sharing reset and later
emissions is part of that bundle and needs a separate ablation to isolate.
If both initially matched arms pass while dense_free fails, initialization
remains a plausible explanation. Dense transitions can still erase history;
failure does not prove convergence, nonlearnability or an initializer-only
cause. No outcome identifies true latent coordinates.

Factorized hidden-state filtering is established modeling. The explicit
conditional structure comes from this favorable synthetic source, while all
arms retain privileged readout initialization. This three-seed base-only
diagnostic makes no architecture-novelty, biological/connectome, calibrated
text, RL, robustness or native-transfer claim. Earlier outcomes and the failed
replication's stop rule remain unchanged.
