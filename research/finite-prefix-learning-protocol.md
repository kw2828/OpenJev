# Public prefix likelihood diagnostic v1

Prospective protocol for **finite-prefix-learning-v1**, not an executed result.
The [closed shared-filter study](finite-shared-filter-results.md) failed all
three criteria. This study tests one supervision change on the same model:
whether predicting public prefix observations before assimilating them helps
held-out decisions and filtering. Previous results remain closed.

## Data and information boundary

Use the unchanged eight-state finite world, four actions, four ordinary
observations and absorbing found, with base sensing epsilon = 0.12. Scientific
namespace **423260924** uses TRAIN split 0, 512 attempts and H2 forecasts, and
fresh DEV split 1, 128 attempts and H8 forecasts. Per-attempt sampling uses
`PCG64(SeedSequence([namespace, split_id, attempt_index]))`. Engineering uses
namespace **931001** and model/fit seed **931101**, never the scientific namespace.
Fabricated audit fixtures use seed identifiers **931101-931103** to test
three-seed rules; these identifiers do not trigger additional learned fits.

A new collector retains every attempted prefix: the initial ordinary odor
and up to eight public action/event pairs in nine 31-wide float32 rows. Include
the first found row and zero-pad the suffix. Record lengths, valid-event masks,
endpoint eligibility and the survivor-row mapping. No replacements or
survivor-only likelihood dataset are allowed. Only attempts surviving all
eight prefix actions receive endpoint forecast targets; their draw order and
targets agree with the frozen generator. Commit forecast actions before
their observations. No old scientific arrays enter this study.

Both learned arms receive identical public data, with no hidden state, belief
array, oracle boundary posterior or true operators as learner inputs. The
exact fixed centered cost readout C in the world's state basis and the uniform
reset prior remain disclosed privileged structure. Analytic targets are exact
finite-world quantities up to float64 roundoff, not a claim that the rational
law and finite categorical sampler are bit exact.

## One intervention, paired models

| Arm | Forecast objective | Added prefix objective |
| --- | --- | --- |
| endpoint_only | Existing four-component H1/H2 objective | Coefficient 0 |
| endpoint_plus_prefix | Identical objective | Prefix event NLL, coefficient 1 |

Each arm is the unchanged eight-state `shared_filter`, with **1,088 trainable
float64 parameters**, float64 carried state and fixed C. There is no extra
head, architecture, initializer search, warm start or posterior loss. Use
dense 0.05-scaled normal logits: reset logits 4 by 8 with local seed
`fit_seed XOR 0x85EBCA6B`, and shared prefix/forecast branch logits 4 by 33 by 8
with seed `fit_seed XOR 0x9E3779B9`. Pair complete initial states across arms
and retain their hashes. These are domain-separated seeded streams, not a
statistical independence claim.

Reset predicts four ordinary odors from the uniform prior before any action
or found hazard: `p(o0) = sum_s E0[o0,s] / 8`. Score the observed odor, then
normalize its posterior. For each later public action, predict the complete
five-event law from the current normalized state **before** conditioning on
that event. Ordinary events normalize the selected branch. Found is scored
once and sets state to absorbing zero; post-found padding contributes no NLL.
A zero assigned probability for a valid event is a failure, not clipped.

The existing forecast objective per eligible case is blind centered-cost
MSE, observed centered-cost MSE, half the sum of blind and observed survival
MSE, and observed five-event soft cross-entropy, with the same reductions and
unit component weights. Costs use scale 1. “Endpoint” means this existing
post-prefix H1/H2 objective, not final-horizon-only supervision. Blind forecasts
propagate unnormalized surviving mass; observed forecasts predict before the
current label, then condition. Known found suffixes have zero costs/survival
and event probability one on found.

## Fixed loss normalization and optimization

Before fitting, freeze `N = 512`, the number `S` of eligible TRAIN attempts,
and total valid prefix events `E`. For a uniformly ordered attempt batch of
actual size `b`, use

`(N / b) * (sum_eligible endpoint_loss / S + lambda * sum_valid event_NLL / E)`.

Here lambda is 0 or 1 by arm. Prefix NLL is a global event mean, not a mean of
per-sequence means. Endpoint normalization is over eligible cases, not all
attempts. Use actual `b`, including partial batches in engineering. Freeze the
denominators over the complete TRAIN attempt pool; do not substitute random
within-batch survivor or event counts. This inclusion scaling targets the
declared dataset objective without shrinking it by the dataset size. It does
not assert that successive without-replacement optimizer steps are independent.

An all-ineligible batch has zero endpoint loss and is never resampled. In the
endpoint-only arm, assign explicit zero gradients and still take the Adam
step. Stored optimizer moments may move parameters on that step; report a
zero-endpoint batch, not an unchanged model. The other arm still receives
its valid prefix-event loss. Prefix NLL need not be computed for training the
zero-coefficient arm; report the resulting work difference.

Fit seeds **423261001, 423261002, 423261003** give six fits. Every fit uses
480 epochs, batch size 64, Adam at 0.003 with default betas/epsilon and
gradient norm clipping at 5. Pair attempt orders through
`PCG64(SeedSequence([fit_seed, epoch, 818]))`; rotate execution order by seed
index. No early stopping, schedule, restart, best seed or checkpoint selection
is permitted. The fixed schedule has **23,040 updates** and **1,474,560 attempt
exposures**. Eligible endpoint exposures total `6 * 480 * S`; added prefix
event exposures total `3 * 480 * E`.

Before the first learned fit, check the untrained exact/exact reference's
five target fields on eligible TRAIN cases within 1e-12. Only that zero-parameter
reference receives an exact boundary posterior and exact operators. Save all
six final checkpoints and a durable completion barrier before any DEV
generation or decoding. Repeat the exact reference check after DEV generation.
Retain the known-dynamics uniform-state reference, which discards prefix
information and is not an optimal history-ignorant policy.

Report all event/attempt/eligible counts, zero-endpoint batches, optimizer
steps, parameter storage, per-fit and inference times, original phase times,
and source-defined structural work by arm and route. These counters are not
exhaustive FLOPs. Equal attempts, epochs and updates do not mean equal
supervision, arithmetic or compute.

## Unchanged prespecified criteria

Evaluate both arms and every fit seed at H1, H2, H4 and H8. Require at least
**256 eligible TRAIN** and **64 eligible DEV** cases, distinct from total
attempt counts. Required uniform-reference cost MSE and regret must be
strictly positive. Every condition must pass for all three seeds; arithmetic
means cannot rescue a failed cell.

| Criterion | Conditions required for all three seeds |
| --- | --- |
| SHORT_HORIZON_LEARNING | At each of H1/H2, blind cost MSE and regret are at most half the uniform reference; observed event-law KL is at most 0.1 nats. |
| BLIND_EXTRAPOLATION | At each of H4/H8, blind cost MSE and regret are at most half the uniform reference; H8 blind survival MAE is at most 0.05. |
| OBSERVED_FILTERING_EXTRAPOLATION | At each of H4/H8, observed event-law KL is at most 0.1 nats. |

Decision regret is exact expected cost of the chosen action minus its minimum.
Learned predictions use raw argmin. Only the uniform reference uses the
lowest-index tie within 1e-12 of its minimum. Preserve every failed cell and
paired arm difference. Means and paired differences are descriptive, without
ensemble, confidence-interval or architecture-superiority claims.

Report prefix NLL across **all DEV attempts**, normalized by all valid DEV
events, with both denominators. This is descriptive and cannot rescue failed
criteria. Keep the offset-1 cyclic shuffle of the complete public prefix and
length within eligible DEV cases, leaving forecast actions/targets unchanged.
It is neither a selection rule nor a significance test. Observed filtering
uses intervening observations and is not a blind-gap result.

## Interpretation, qualification and closure

Any pass establishes only a bounded supervision result with a known fixed cost
basis. Its rank is at most three and it supplies four decision signatures
across eight states; successful decisions do not prove unique latent-state
recovery. Failure does not establish convergence or nonlearnability. The
generating family favors these categorical filter models; no calibration,
connectome, native-environment, robotics or novelty claim follows.

Predictive-state filtering and structured recurrent inference are established
in [PSIM](https://proceedings.mlr.press/v48/sun16.html),
[PSRNNs](https://proceedings.neurips.cc/paper/2017/file/2bb0502c80b7432eee4c5847a5fd077b-Paper.pdf)
and [Recurrent Kalman Networks](https://proceedings.mlr.press/v97/becker19a.html).
This study neither implements their complete algorithms nor inherits their
guarantees; the intervention is observable prefix-likelihood supervision.

Freeze protocol, complete source closure, runtime, configuration, paths and a
successful engineering receipt before scientific generation. Fabricated tests
must cover survivor equivalence, found-inclusive masks, prediction-before-label
chronology, global normalization with partial/all-ineligible batches, paired
initialization and six-checkpoint DEV barrier. Use the original inherited
suspend-inclusive supervisor: **300 seconds qualification, 1800 seconds
fit/evaluation and 600 seconds audit**, 4 GiB worker RSS and 512 MiB output per
phase. Use one CPU thread, no accelerator, native environment or external
model/API calls.

Preserve failed engineering attempts and their source snapshots; repairs need
fresh engineering registration and exclusive paths. Science has one attempt,
with no retry, cap extension or replacement cases. Close the original producer
before independent saved-output audit. Authenticate process joins, source pins
and payload bytes, then independently reconstruct observations, analytic
targets, NLL and metrics without model/optimizer replay. Historical fit/order
and work counters remain producer attestations checked against the schedule.
No result automatically admits another study.
