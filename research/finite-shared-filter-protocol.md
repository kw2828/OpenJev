# Shared filtering and forecasting diagnostic v1

Prospective protocol for **finite-shared-filter-v1**, not an executed result.
The [completed factor diagnostic](finite-factor-learning-results.md) passed
with an exact boundary posterior and learned operators, while both cells that
learned the prefix state failed their criteria. This study tests whether using
one learned filtering mechanism for history and forecasting helps under the
fixed recipe. Previous results remain closed.

## Data and information boundary

Use the unchanged eight-state finite world, four actions, four ordinary
observations and absorbing found, with the base sensing law epsilon = 0.12.
The scientific data namespace is **422260924**. TRAIN uses split 0 with 512
attempted cases and two forecast steps; fresh DEV uses split 1 with 128
attempted cases and eight forecast steps. Each attempt uses
`PCG64(SeedSequence([namespace, split_id, attempt_index]))`.

Each case has the initial observation plus eight public action/observation
pairs, represented by the same nine 31-wide float32 rows. Prefixes ending in
found are excluded without replacement. Require at least 256 retained TRAIN
and 64 retained DEV cases for a criterion to pass. Commit the forecast action
block before its observations. No old arrays, replacement seeds or
DEV-dependent selection are allowed. Engineering uses namespace **930001**
and fit seed **930101**, never the scientific namespace.

All learned arms receive public history and forecast actions, with no oracle
boundary posterior, hidden state or true operators. They retain the exact,
fixed centered cost readout C in the world's state basis. The uniform reset
prior is disclosed task knowledge; the filter arms use it explicitly, whereas
the GRU-prefix arm learns its history mapping from the public tokens. This
privileged structure limits any claim of generality.

Targets are analytic finite-world quantities up to float64 roundoff. This is
not a claim that the rational law and finite categorical sampler are bit exact.

## Mechanism and controls

| Arm | Prefix computation | Forecast computation | Trainable parameters |
| --- | --- | --- | ---: |
| shared_filter | Learned reset emission and learned branch operators | The same branch operators | 1,088 |
| untied_filter | Learned reset emission and separate prefix branch operators | Learned forecast branch operators | 2,144 |
| gru_prefix | Existing GRU prefix and eight-state probability projection | Learned forecast branch operators | 6,412 |

`gru_prefix` is the existing `FactorModel('learned_learned')`, not the earlier
ordinary GRU with separate learned prediction heads. All arms carry eight
nonnegative probability-mass coordinates and use the same C. Filter operators,
reset probabilities and carried states use float64. The GRU prefix uses the
existing float32 GRU and float64 projection. Equal state width does not imply
equal parameters, storage, arithmetic or optimization difficulty.

For each state, normalize learned reset logits E0 over the four ordinary
observations. The reset observation occurs before transition or found hazard:

`b_reset[s] = E0[initial_observation,s] / sum_s E0[initial_observation,s]`.

This is Bayes normalization with the fixed uniform prior. For each of the
eight later prefix pairs, update `b = B[action,observation] @ b`, then normalize
by its sum. Prefix evidence normalizers are not added to the objective.
Joint column normalization covers four nonterminal destination branches and
one found branch, so `A[action] = sum_observation B[action,observation]` is the
blind operator. The untied arm has its own normalized prefix B.

Blind forecasts propagate unnormalized surviving mass. Observed forecasts
emit costs, survival and the five-event law before assimilating the current
observation; nonterminal updates normalize the selected branch. Found is
absorbing: subsequent cost and survival are zero, and the event law assigns
probability one to found. Costs are a linear readout of propagated state, not
a nonlinear readout of its mean. A zero predicted probability on positive
target support is a numerical failure, not silently clipped.

Use dense 0.05-scaled normal logits. Forecast logits have shape 4 by 33 by 8
and a local generator seeded with `fit_seed XOR 0x9E3779B9`; pair them across
all arms. Reset logits have shape 4 by 8 and seed `fit_seed XOR 0x85EBCA6B`;
pair them across the filter arms. Initialize untied prefix logits by a detached
copy of its forecast logits, with independent parameter storage. The GRU
prefix retains initialization from the fit seed. These are domain-separated
seeded streams, not a statistical independence claim.

Engineering must verify initial shared/untied predictions agree within 1e-12,
copy independence and the intended gradient paths. Scientific records retain
paired initial forecast and reset group hashes. Sharing means both prefix and
forecast gradients update the same B; it must not detach prefix state or skip
prefix backpropagation. No retentive initializer, warm start or initializer
selection is allowed.

Retain one untrained exact/exact implementation reference. Only that reference
receives an exact boundary posterior and exact operators. Check all five
target fields within 1e-12 on TRAIN **before the first learned fit**, and on
DEV after its generation. It has zero trained parameters and no optimizer.
Also retain the known-dynamics uniform-state reference, which discards prefix
information and is not an optimal history-ignorant policy.

## Training and measured work

Fit seeds **422261001, 422261002, 422261003** give nine learned fits. Every fit
uses 480 epochs, batch size 64, Adam at 0.003 with default betas/epsilon and
gradient norm clipping at 5. Pair case orders through
`PCG64(SeedSequence([fit_seed, epoch, 818]))`. Rotate arm execution by seed
index. No early stopping, schedule, restart, best seed or checkpoint selection
is permitted.

Keep the four-component objective unchanged: blind centered-cost MSE, observed
centered-cost MSE, half the sum of blind and observed survival MSE, and observed
five-event soft cross-entropy, each with weight one. Costs use scale 1. Add no
latent-state, posterior or prefix-likelihood loss. Exact soft cross-entropy
equals target entropy, so a positive objective alone does not show error.

All nine final checkpoints and their durable barrier precede any DEV
generation or decoding. With N retained TRAIN cases, the schedule has
`9 * 480 * ceil(N / 64)` optimizer updates and `9 * 480 * N` case exposures.
Report parameter counts, precision, storage, actual fit/inference/phase time
and source-defined structural work by arm and route. Include prefix and
forecast work for training, ordinary inference and shuffled inference, with
the exact-control work separately identified. Structural counters are not
exhaustive FLOPs. Equal epochs and examples do not establish equal compute.

## Prespecified criteria

Evaluate every arm and seed at H1, H2, H4 and H8. Apply each of the three
criteria separately to **all three arms**. Every applicable condition must
hold for every fit seed; means cannot rescue a failed cell. Both retained-case
minima apply. Required reference cost MSE and regret must be strictly positive.

| Criterion | Conditions required for all three seeds |
| --- | --- |
| SHORT_HORIZON_LEARNING | At each of H1/H2, blind cost MSE and regret are at most half the uniform reference; observed event-law KL is at most 0.1 nats. |
| BLIND_EXTRAPOLATION | At each of H4/H8, blind cost MSE and regret are at most half the uniform reference; H8 blind survival MAE is at most 0.05. |
| OBSERVED_FILTERING_EXTRAPOLATION | At each of H4/H8, observed event-law KL is at most 0.1 nats. |

Decision regret uses the exact expected cost of the selected action minus the
minimum exact expected cost. Learned predictions use raw argmin. Only the
uniform reference uses the lowest-index tie within 1e-12 of its minimum.
Observed filtering receives intervening observations and is not a blind-gap
result. Report all conditions, metrics, seeds and failures. Report paired
arm differences and arithmetic means descriptively, without an ensemble or
confidence-interval claim. There is no prespecified architecture-win gate.

Keep the descriptive cyclic prefix-shuffle check at fixed offset 1 within
DEV. Move the whole public prefix and its length together, leaving forecast
actions and targets unchanged. This is not a significance or selection test.

## Interpretation and closure

Shared versus untied filtering tests the sharing constraint more directly;
the GRU comparison also changes the encoder family and measured work. A
shared-only pass would support this constraint under the recipe, not establish
a universal architectural advantage. If both filters pass, that does not
isolate sharing. Failure does not establish nonlearnability or convergence.
The fixed C has rank at most three and four decision signatures across eight
states; successful costs do not prove unique latent-state recovery.

The model uses established filtering structure. Relevant primary context is
[spectral HMM learning](https://arxiv.org/abs/0811.4413),
[Predictive State Inference Machines](https://proceedings.mlr.press/v48/sun16.html)
and [value equivalence](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html).
Their algorithms or guarantees are not claimed here. No novelty, calibration,
connectome, robotics or native-environment result follows from this diagnostic.

Freeze the protocol, complete sources, runtime, configuration, paths and
successful engineering receipt before scientific generation. Use the inherited
original suspend-inclusive supervisor with caps of 300 seconds for
qualification, 1800 seconds for fit/evaluation and 600 seconds for audit.
Worker peak RSS is limited to 4 GiB and each phase output to 512 MiB. Use one
CPU thread, with no accelerator, native environment, external model or paid
inference calls.

Preserve failed qualification attempts and their source snapshots. Mechanical
repairs require a new engineering registration and exclusive paths. Science
has one attempt with no retry, cap extension or replacement cases. Close the
original producer before independent saved-output audit; authenticate original
process joins, unchanged source pins and payload bytes. Independently
reconstruct targets and metrics without model/optimizer replay. Historical
training order, model execution and source-defined operation counts remain
producer attestations, checked against their declared schedule. No result
automatically admits another study.
