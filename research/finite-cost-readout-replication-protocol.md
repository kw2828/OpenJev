# Learned cost-readout replication v1

Prospective protocol for **finite-cost-readout-replication-v1**. This is only
the first step in the [committed follow-up proposal](finite-cost-readout-next.md).
The [completed study](finite-cost-readout-results.md) found improved decisions
with a learned readout, but two of three fits still failed the eight-step
blind cost thresholds. This experiment repeats the function-matched
learned-versus-softened comparison on new data and fit seeds. It does not
repeat the exact-C anchor, train on longer horizons, or reopen prior studies.

## Fresh population and information boundary

Use the unchanged eight-state finite world, four actions, four ordinary
observations and absorbing found, at base sensing epsilon **0.12**. Scientific
namespace **425260924** uses TRAIN split 0 with **512 attempts** and H2
forecasts, then DEV split 1 with **128 attempts** and H8 forecasts. Each attempt
uses `PCG64(SeedSequence([namespace, split_id, attempt_index]))`. Engineering
uses namespace **933001** and actual runner model/fit seed **933101**. Fabricated
audit fixtures may use identifiers **933101-933103** for three-seed rules,
without additional learned fits. The unchanged inherited readout-component
tests retain fabricated model seed **932101**; they generate no scientific
cases. Engineering never uses scientific seeds.

Reuse the qualified all-attempt collector without changing its numerical
behavior. Preserve the reset odor and up to eight public action/event pairs
in nine 31-wide float32 rows. Include the first found row, then zero padding,
with lengths, valid-event masks, endpoint eligibility and survivor-row mapping.
All attempts belong to the prefix population. Only prefixes surviving all
eight actions receive endpoint forecasts. No replacements, survivor-only
likelihood sample, old data arrays, old checkpoints or warm starts are allowed.
Commit each forecast action block before its observations.

Both arms receive the same public inputs and labels. No hidden state, belief
array, oracle boundary posterior or true transition/observation operators
enter either learned model. The uniform reset prior and world-aligned initial
readout are privileged task knowledge shared by the arms. Targets use analytic
finite-world laws up to float64 roundoff; this does not claim bit-exact identity
between analytic laws and the finite categorical sampler.

## Two unchanged arms

Reuse the frozen cost-readout model and prefix-likelihood adapter. Let
`C[:,s] = 0.25 - onehot(g(s))`, with the same known preferred action g for each
latent coordinate, and preserve **delta = 0.1**.

| Arm | Readout | Trainable parameters | Parameter bytes | Buffer bytes |
| --- | --- | ---: | ---: | ---: |
| fixed_softened | Fixed `0.9 * C` | 1,088 | 8,704 | 256 |
| learned_readout | `0.25 - softmax(L[:,s])` over actions | 1,120 | 8,960 | 0 |

Initialize `L[:,s] = log(0.9 * onehot(g(s)) + 0.1/4)`. The initial learned
matrix matches the fixed softened matrix within 1e-12: preferred cost -0.675,
other costs 0.225. The sign is **0.25 minus probability**. The head adds 32
stored logits and 24 identifiable degrees of freedom, not 32 independently
identifiable probabilities. Both arms use 8,960 parameter-plus-buffer bytes;
optimizer state, gradients, activations and intermediate products are extra.

The readout is linear in the eight-coordinate state, with no additive bias.
It therefore commutes with state averaging, including unnormalized surviving
mass, and maps the absorbed zero state to zero costs. Softmax constructs the
matrix from parameters; it is not applied after averaging states. Columns sum
to zero. Preserve the existing numeric guards: finite-logit costs are interior
to [-0.75, 0.25] in real arithmetic, subtraction may round to a boundary, and
closed-bound validation allows 1e-12 without clipping. Nonfinite values or
softmax underflow remain explicit failures.

Pair the same dense 0.05-normal branch logits, shape 4 by 33 by 8, using local
seed `fit_seed XOR 0x9E3779B9`, and reset logits, shape 4 by 8, using
`fit_seed XOR 0x85EBCA6B`. Preserve cross-arm initial group hashes. These are
domain-separated seeded streams, not a statistical independence claim.
Readout initialization is deterministic. No initializer, delta, architecture,
head parameterization or loss-weight search is allowed.

## Unchanged chronology and objective

Every attempt contributes prefix likelihood with coefficient **1**. Predict
the four-way reset odor distribution from the uniform prior before any
action or hazard, score the observed odor, then condition. For each subsequent
action, predict the full five-event law before assimilating that event.
Ordinary branches normalize; found is scored once and terminates the prefix.
Padded rows contribute zero. A positive event assigned zero probability is an
explicit failure, not a clipped likelihood.

Eligible endpoint cases retain the four-component H1/H2 objective: blind
centered-cost MSE, observed centered-cost MSE, half the sum of blind and observed
survival MSE, and observed five-event soft cross-entropy. Preserve the existing
reductions, unit weights and cost scale 1. Endpoint denotes the post-prefix
forecast objective, not final-horizon-only supervision. Blind state is
unnormalized surviving mass. Observed predictions precede the current event;
after known found, costs/survival are zero and the event law is found with
probability one.

Freeze attempted TRAIN count `N = 512`, eligible TRAIN count `S` and total
valid TRAIN events `E` before fitting. For actual attempt-batch size `b`, use

`(N / b) * (sum_eligible endpoint_loss / S + sum_valid event_NLL / E)`.

This preserves a global eligible-case mean plus a global event mean. Do not
substitute batch-specific denominators or average per-sequence NLL means.
Use actual b in partial engineering batches. A batch with no eligible
endpoints contributes zero endpoint loss without resampling, but still uses
its prefix events. Keep explicit zero gradients for parameters with no current
loss path; Adam's stored moments can still move them. Without-replacement
batches do not imply independent optimizer updates.

## Fixed training schedule and DEV barrier

Fit seeds **425261001, 425261002 and 425261003** produce **six fits**. Train
every fit for **480 epochs**, batch size **64**, Adam learning rate **0.003**
with default betas/epsilon, and global gradient norm clipping at **5**. Pair
attempt orders with `PCG64(SeedSequence([fit_seed, epoch, 818]))`; rotate the
two-arm execution order by seed index. No early stopping, schedule, restart,
best-seed selection or checkpoint selection is allowed.

The fixed schedule has **23,040 optimizer updates**, **1,474,560 attempt
exposures**, `6 * 480 * S` eligible endpoint exposures and `6 * 480 * E`
prefix-event exposures. Every final checkpoint and the durable six-fit barrier
must precede any DEV generation or decoding.

Before the first learned fit, the zero-parameter exact/exact reference must
match all five eligible TRAIN target fields within 1e-12; repeat this check
after DEV generation. Only that reference receives true boundary posteriors
and operators. It is a target-consistency reference, not a fitted exact-C
control. Retain the known-dynamics uniform-state reference, which discards
prefix information and is not an optimal history-ignorant policy.

Report all fit durations and counters, ordinary/shuffled inference, prefix
NLL evaluation, recurrence/head work and all zero-endpoint batches. Preserve
**12** initial/final readout-matrix snapshots, including **six** learned-head
snapshot softmax evaluations, separately from forward-work counters and inside
fit timing. Fixed forwards and prefix likelihood have no head softmax; each
learned forecast readout includes its unchanged matrix-construction cost.
Counters do not represent exhaustive FLOPs. Matched data, epochs and updates
do not establish matched measured compute, optimizer storage or a speedup.

## Absolute criteria and replication continuation

Evaluate both arms and every seed at H1, H2, H4 and H8. Preserve minimum
support of **256 eligible TRAIN** and **64 eligible DEV** cases, separate from
attempt counts, and strictly positive required uniform-reference MSE/regret.
Apply all three unchanged criteria separately to each arm. Every applicable
condition must hold in all three seeds; arithmetic means cannot rescue failure.

| Criterion | Conditions required for every seed |
| --- | --- |
| SHORT_HORIZON_LEARNING | At each of H1/H2, blind cost MSE and regret are at most half the uniform reference; observed event-law KL is at most 0.1 nats. |
| BLIND_EXTRAPOLATION | At each of H4/H8, blind cost MSE and regret are at most half the uniform reference; H8 blind survival MAE is at most 0.05. |
| OBSERVED_FILTERING_EXTRAPOLATION | At each of H4/H8, observed event-law KL is at most 0.1 nats. |

Including support/reference conditions, these retain 24, 21 and 8 checks
respectively per arm. Regret is exact expected cost of the selected action
minus the minimum. Model predictions use raw argmin; only the uniform reference
uses the lowest-index tie within 1e-12 of its minimum.

The separate **replication continuation** is the conjunction of exactly eight
checks: learned SHORT_HORIZON_LEARNING passes, learned
OBSERVED_FILTERING_EXTRAPOLATION passes, and learned blind regret is **strictly
lower** than its paired fixed-softened control for each of three seeds at
each of H4 and H8. The six signed differences must each be less than zero;
ties do not pass. BLIND_EXTRAPOLATION remains a separate unchanged result.
A continuation pass does not turn a blind-criterion failure into success.

Report all **24** arm/seed/horizon rows, all six paired H4/H8 contrasts in
regret, blind/observed cost MSE and observed KL, prefix NLL over every valid
DEV event, complete condition results and measured costs. Arithmetic means
and signed comparisons are not ensembles, confidence intervals or statistical
significance. Keep the offset-1 cyclic shuffle of complete public prefix and
length within eligible DEV cases, leaving forecast actions/targets unchanged;
it is descriptive, not a selection or significance test. Observed filtering
receives intervening evidence and is not a blind-gap result.

Continuation failure ends this registered study without replacement seeds,
retuning or promotion of a subset. Success supports only a separate prospective
registration of the proposed longer-supervision comparison. Neither outcome
automatically executes or admits that later study.

## Qualification, original closure and interpretation

Freeze this protocol, complete transitive source closure, runtime, configuration,
paths and successful engineering evidence before any scientific generation.
New wrappers may reuse immutable qualified functions; do not modify previous
pinned sources. Qualify fabricated or declared engineering cases only. Verify
initial full learned/softened forward and raw B/E-gradient parity within
1e-12, fixed/trainable ownership, linear expectation, absorbed zero, terminal
masks, global loss scaling, oracle-input boundaries and the six-checkpoint
barrier. Raw-gradient equivalence is before clipping: the learned head joins
the global clip norm, so clipped B/E gradients and updates need not match.

Use the inherited original suspend-inclusive supervisor with **300 seconds
qualification, 1800 seconds fit/evaluation and 600 seconds audit**, **4 GiB
worker RSS** and **512 MiB output per phase**. Use one CPU thread, no
accelerator, native environment, external model or paid inference calls.
Preserve every failed qualification attempt and its pinned source snapshot;
engineering repairs require a new registration and exclusive paths. Science
has **one attempt**, with no retry, cap extension or replacement cases.

Close the original producer before the independent saved-record audit.
Authenticate original process joins, source identity and opaque payload hashes
before reconstructing targets, likelihoods, metrics and schedules. The audit
does not execute learned models or replay optimization; historical ordering
and work remain source-qualified producer attestations. Preserve technical
failures distinctly from completed scientific failures and retain all evidence.

This narrower replication omits the fitted exact-C anchor. A gain over the
softened control can include undoing softening; it does not identify latent
alignment, convergence or state recovery. The extra head also changes
parameter count and clipping/optimization geometry. Both initial readouts
remain privileged and aligned to the known world. Prior studies stay closed
and their conclusions are not recomputed from new seeds.

[Value equivalence](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)
is established context for decision-useful models, not a new algorithm or
guarantee implemented here. This favorable synthetic family, three fit seeds
and single base DEV population establish no architectural novelty, calibrated
text capability, connectome benefit, RL advance or native-task transfer.
