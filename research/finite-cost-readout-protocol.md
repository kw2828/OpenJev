# Learned cost-readout diagnostic v1

Prospective protocol for **finite-cost-readout-v1**, not an executed result.
The [closed prefix-loss study](finite-prefix-learning-results.md) improved
observation prediction but failed all three criteria; observed cost error
worsened. This study tests whether training the cost readout alongside the
same filter improves decisions under the fixed recipe. The true world is
already representable with the existing exact fixed C. The hypothesis concerns
optimization and representation alignment, not an expressivity impossibility.
Previous studies remain closed.

## Fresh data and information boundary

Use the unchanged eight-state finite world, four actions, four ordinary
observations and absorbing found, with base sensing epsilon = 0.12. Scientific
namespace **424260924** uses TRAIN split 0, **512 attempts** and H2 forecasts,
then fresh DEV split 1, **128 attempts** and H8 forecasts. Each attempt uses
`PCG64(SeedSequence([namespace, split_id, attempt_index]))`. Engineering uses
namespace **932001** and actual model/fit seed **932101**. Fabricated audit
fixtures may use identifiers **932101-932103** for three-seed rules, without
additional learned fits. No engineering call uses scientific seeds.

Reuse the qualified all-attempt collector: retain the initial ordinary odor
and up to eight public action/event pairs in nine 31-wide float32 rows.
Include the first found row, then zero padding, with explicit lengths, valid
event masks, endpoint eligibility and survivor-row mapping. All attempts
remain in the prefix population. Only prefixes surviving all eight actions
receive endpoint forecasts. No replacements, survivor-only likelihood sample,
old arrays or old checkpoints are allowed. Commit each forecast action block
before its observations.

All arms receive identical public inputs and labels. No hidden state, belief
array, oracle boundary posterior or true operators enter a learned model.
Uniform reset prior and the world-aligned initial cost readout are privileged
task knowledge shared by all arms. Targets are analytic finite-world values
up to float64 roundoff, not a bit-exact identity between rational laws and
the finite categorical sampler.

## One readout intervention with an initialization control

All arms use the same eight-state shared prefix/forecast filter. Let `g(s)`
be the known cost-preferred action for latent coordinate s. The existing cost
column is `C[:,s] = 0.25 - onehot(g(s))`. Fix **delta = 0.1** before generation.

| Arm | Cost readout | Trainable parameters |
| --- | --- | ---: |
| fixed_exact | Exact C, fixed | 1,088 |
| fixed_softened | `C_delta = 0.9 * C`, fixed | 1,088 |
| learned_readout | `C_L[:,s] = 0.25 - softmax(L[:,s])`, trainable | 1,120 |

Softmax runs over the four actions separately for each latent coordinate.
Initialize `P_delta[:,s] = 0.9 * onehot(g(s)) + 0.1/4` and
`L[:,s] = log(P_delta[:,s])`. This initializes the learned head to the softened
head up to float64 roundoff: preferred cost -0.675 and other costs 0.225.
The sign is **0.25 minus probability**. Finite logits give costs strictly
between -0.75 and 0.25 in real arithmetic and cannot attain exact-C boundary
vertices. Float64 subtraction can round to a boundary; numeric cost validation
uses the closed bounds with 1e-12 tolerance, without clipping. Nonfinite values
or softmax underflow remain explicit failures.

Only learned-versus-fixed-softened isolates readout trainability at the same
initial function. Gains against that control alone may simply undo softening;
they do not establish latent-coordinate alignment. Fixed exact C is the
privileged anchor and must remain in all results. Report both contrasts.
No delta, head parameterization or loss-weight sweep is permitted.

The readout remains linear in state with no additive bias: `cost = C_L @ b`.
It preserves `C_L @ E[b] = E[C_L @ b]`, including unnormalized surviving mass,
and maps absorbed zero state to zero cost. Softmax constructs C_L from model
parameters; it is not applied to the state or after averaging states.
Columns sum to zero. The new head has 32 stored logits but only 24 identifiable
degrees of freedom because adding a constant to an action column's logits
leaves its probabilities unchanged. Report parameter counts separately from
these degrees of freedom and from any latent-state identifiability claim.

Use unchanged dense 0.05-normal initialization for shared branch logits
4 by 33 by 8, with local seed `fit_seed XOR 0x9E3779B9`, and reset logits
4 by 8 with seed `fit_seed XOR 0x85EBCA6B`. Pair B/E weights and initial group
hashes across all three arms. These are domain-separated seeded streams, not
a statistical independence claim. The learned head initialization is the
fixed deterministic expression above, not another random stream. There is
no warm start, retentive initializer, factorized transition family or new RL
algorithm.

## Prediction chronology and common objective

Every arm uses prefix NLL coefficient **1**, including found-terminated
attempts. Reset predicts four ordinary odors from the uniform prior before
any action or hazard, scores the observed odor, then conditions. Each later
action predicts the full five-event law before assimilating that event.
Ordinary branches normalize their posterior. Found is scored once and ends
the prefix; padded events contribute zero. Positive observed events assigned
zero probability fail explicitly rather than being clipped.

The endpoint objective remains the same four components on eligible cases:
blind centered-cost MSE, observed centered-cost MSE, half the sum of blind and
observed survival MSE, and observed five-event soft cross-entropy. Retain the
same H1/H2 reductions, unit weights and cost scale 1. Endpoint means the
existing post-prefix forecast objective, not final-horizon-only supervision.
Blind state is unnormalized surviving mass. Observed forecasts precede the
current label; after known found, costs/survival are zero and the event law
puts probability one on found.

Freeze `N = 512`, eligible TRAIN count `S`, and total valid TRAIN events `E`
before fitting. For an attempt batch of actual size `b`, use

`(N / b) * (sum_eligible endpoint_loss / S + sum_valid event_NLL / E)`.

This targets one global eligible-case mean plus one global event mean. Do not
replace S/E with random batch denominators or average per-sequence NLL means.
Use actual b for partial engineering batches. Without-replacement batches do
not imply independent successive optimizer updates. A batch with no endpoints
contributes zero endpoint loss, is not resampled, and still trains on its
prefix events. Keep explicit zero gradients for parameters without a current
loss path, including the learned head on such batches. Adam's stored moments
can still move these parameters; zero current loss is not an unchanged-state
claim.

## Training, barrier and measured work

Fit seeds **424261001, 424261002, 424261003** give **nine fits**. Every fit uses
480 epochs, batch size 64, Adam at 0.003 with default betas/epsilon and
gradient norm clipping at 5. Pair attempt orders using
`PCG64(SeedSequence([fit_seed, epoch, 818]))`, rotating arm execution by seed
index. No early stopping, learning-rate schedule, restart, best-seed or
checkpoint selection is allowed.

The fixed schedule has **34,560 optimizer updates**, **2,211,840 attempt
exposures**, `9 * 480 * S` eligible endpoint exposures and `9 * 480 * E`
prefix-event exposures. Every final checkpoint and a durable nine-fit barrier
must precede any DEV generation or decoding.

Before the first learned fit, the zero-parameter exact/exact reference must
match all five eligible TRAIN target fields within 1e-12; repeat this check
after DEV generation. Only that reference receives true boundary posteriors
and operators. Retain the known-dynamics uniform-state reference, which
discards prefix information and is not an optimal history-ignorant policy.

Report every fit's storage, duration, counters and final state. The learned
head adds 256 parameter bytes to the fixed arms' 8,704 parameter bytes, plus
optimizer and gradient storage; report actual buffers separately. Shared
filter states, operators and costs remain float64, with float32 public input
tokens. Fixed arms have a 256-byte C buffer; the learned arm constructs its
readout from parameters and has no cost buffer. Include head softmax/readout
work as well as prefix/forecast work,
ordinary and shuffled inference, original phase costs and all zero-endpoint
batches. Counters are not exhaustive FLOPs. Equal data, epochs and optimizer
updates are not equal parameters, operations or measured compute.

## Unchanged criteria and complete reporting

Evaluate every arm and seed at H1, H2, H4 and H8. Require at least **256
eligible TRAIN** and **64 eligible DEV** cases, distinct from attempted
counts. Required uniform-reference cost MSE and regret must be strictly
positive. Apply each criterion separately to each arm, requiring all three
seeds to pass every applicable condition. Means cannot rescue failed cells.

| Criterion | Conditions required for all three seeds |
| --- | --- |
| SHORT_HORIZON_LEARNING | At each of H1/H2, blind cost MSE and regret are at most half the uniform reference; observed event-law KL is at most 0.1 nats. |
| BLIND_EXTRAPOLATION | At each of H4/H8, blind cost MSE and regret are at most half the uniform reference; H8 blind survival MAE is at most 0.05. |
| OBSERVED_FILTERING_EXTRAPOLATION | At each of H4/H8, observed event-law KL is at most 0.1 nats. |

Regret uses exact expected cost of the selected action minus the minimum.
Learned predictions use raw argmin; only the uniform reference uses the
lowest-index tie within 1e-12 of its minimum. Report every condition and all
36 arm/seed/horizon rows. Show both learned-minus-softened and
learned-minus-exact paired H4/H8 differences in blind regret, blind and
observed cost MSE, and observed KL, together with prefix NLL and actual costs.
These comparisons and arithmetic means are descriptive, not ensembles,
confidence intervals or an extra retrospective superiority gate.

Prefix NLL is reported over all DEV attempts and all valid events with both
denominators. It cannot rescue failed criteria. Keep the offset-1 cyclic
shuffle of complete public prefix and length within eligible DEV cases,
leaving forecast actions/targets unchanged; it is not a significance or
selection test. Observed filtering has intervening observations and must not
be presented as a blind-gap result.

## Interpretation and original closure

Fixed exact C plus the true positive B/E is already in the model class. The
learned head can change optimization and moment alignment, but does not fix
a demonstrated expressivity impossibility. A gain could merely compensate
for initial softening; unchanged observation quality and improvements over
the exact-C anchor are necessary context for a stronger interpretation.
Failure does not establish convergence or nonlearnability. No outcome proves
recovery of true latent coordinates. The exact cost basis has four decision
signatures across eight states and rank at most three; a learned centered
four-action readout also has rank at most three.

[Value equivalence](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)
is established context for decision-useful models, not an algorithm or
guarantee implemented by this experiment. This favorable synthetic-family
diagnostic claims no calibration, connectome benefit, architectural novelty,
native transfer, text capability or robotics result.

Freeze protocol, complete source closure, runtime, configuration, paths and
successful engineering evidence before generating scientific data. Engineering
must check full initial learned/softened forward parity and raw shared B/E
gradient parity within 1e-12, before clipping. The common global clip threshold
also includes learned-head gradients, so clipped B/E gradients and updates
need not match when clipping activates. This coupling is part of the declared
optimizer recipe, not evidence of different raw initial functions. Check
fixed versus trainable ownership, head bounds/centering,
linear expectation and absorbed-zero identities, all-attempt masks, source
and oracle-input boundaries, global loss scaling and the nine-checkpoint
DEV barrier. Qualify only fabricated or declared engineering-namespace cases.

Use the inherited original suspend-inclusive supervisor with **300 seconds
qualification, 1800 seconds fit/evaluation, 600 seconds audit**, 4 GiB worker
RSS and 512 MiB output per phase. Use one CPU thread with no accelerator,
native environment, external model or paid inference calls. Preserve failed
qualification attempts and source snapshots; repairs need a fresh engineering
registration and exclusive paths. Science has one attempt with no retry,
cap extension or replacement cases.

Close the original producer before independent saved-output audit. Authenticate
original process joins, unchanged source pins and all payload bytes before
reconstructing public-history targets, likelihoods, metrics and schedules.
The auditor does not replay training or models; historical ordering and work
remain source-qualified producer attestations. No result automatically admits
another study.
