# Finite factor learning diagnostic v1

Prospective protocol, not an executed result or an architecture claim. The
[previous finite-world study](finite-observation-learning-results.md) remains
closed with both registered criteria failed. This diagnostic separates initial
history encoding from subsequent operator learning. It changes state width,
readout knowledge, factorization and training budget together, so comparisons
with the previous 48-epoch study cannot isolate a budget or architecture effect.

## World, cases and information

Use the unchanged, source-pinned eight-state finite world and public-token
generator. There are four actions, four ordinary observations and absorbing
found. The primary study uses only the base sensing law, epsilon = 0.12. It does
not supply or evaluate an unannounced sensor shift.

Each attempted case observes an initial symbol and eight random prefix actions
and outcomes. A found prefix is excluded without replacement. The public input
is nine 31-wide float32 rows containing action/observation indicators and the
reset marker; unused coordinates are zero. The full forecast action block is
committed before its observations. Forecasts precede the current observation.
Blind state is unnormalized surviving mass. Observed nonterminal updates
normalize the selected branch, and found is absorbing: later cost and survival
targets are zero, while the event-law target places probability one on found.

The scientific data namespace is **421260924**. TRAIN uses split 0, 512 attempted
cases and horizons 1/2; fresh base DEV uses split 1, 128 attempted cases and
horizons 1 through 8. Each attempted case uses
`PCG64(SeedSequence([namespace, split_id, attempt_index]))`. Minimum retained
support is 256 TRAIN and 64 DEV cases. No old study arrays, replacement cases
or DEV-dependent selection are allowed. Qualification uses separate data
namespace **929001** and fit seed **929101**, never the scientific namespace.

All targets are analytic finite-world quantities up to float64 roundoff,
conditional on the sampled contexts. This does not claim bit-exact agreement
between a rational law and NumPy's finite categorical sampler.

## Four factor cells and descriptive controls

All factor cells have eight nonnegative probability-mass coordinates and the same frozen,
exact centered cost readout C in the world's state basis. The blind operator
is the sum of the four nonterminal observation branches. Operators and carried
state use float64; a learned public-prefix encoder uses the inherited float32
GRU followed by its float64 probability projection.

| Cell | State at the forecast boundary | Operators afterward | Role |
| --- | --- | --- | --- |
| exact_exact | Exact prefix posterior | Exact base-world operators | Untrained numerical reference |
| learned_exact | Learned from public prefix | Exact base-world operators | Isolate prefix learning |
| exact_learned | Exact prefix posterior | Learned operators | Isolate operator learning |
| learned_learned | Learned from public prefix | Learned operators | Joint learning |

An exact-prefix cell receives the oracle posterior **only at the forecast
boundary**. It must subsequently propagate and condition using its assigned
operators. No future oracle posterior is injected, and no latent-state or
posterior reconstruction loss is added. Exact states saved separately for an
audit cannot enter any other learner input or loss.

The two learned-operator cells use the same dense initialization: 0.05-scaled
Gaussian logits followed by the declared column normalization.
There is no retentive template, true-operator warm start or initializer search.
Initialization uses domain-separated seeded streams: the encoder uses the fit
seed, and the local operator generator uses `fit_seed XOR 0x9E3779B9`. This is
reproducible pairing, not a claim of statistical independence. For each fit
seed, learned-prefix cells start with identical encoder weights and
learned-operator cells start with identical operator weights.

Retain the ordinary GRU as a descriptive learned control, using the same public
inputs, data, loss and update schedule. Its learned readout, capacity and
parameterization differ from the factor cells; it is not a matched factor.
Also retain the privileged uniform-state reference: start from 1/8 after the
prefix and propagate the exact blind operator. This reference discards prefix
information and is not an optimal history-ignorant policy, because retaining a
prefix conditions its state distribution.

Exact/exact must reproduce blind costs and survival, observed costs and
survival, and event probabilities within a maximum absolute float64 error of
1e-12. Failure is a technical failure, not evidence about learning. It receives
no optimizer or fitted checkpoint. There is one such untrained reference,
not three independently trained replicas.

## Fixed training and evaluation

Fit seeds are **421261001, 421261002, 421261003**. The three learned factor
cells and GRU yield **12 learned fits**. Every fit receives 480 epochs, batch 64,
Adam with learning rate 0.003 and its default betas/epsilon, and gradient norm
clip 5. There is no schedule, early stopping, best epoch, best seed, continuation
or checkpoint selection. Case permutations use
`PCG64(SeedSequence([fit_seed, epoch, 818]))` and are paired across learned cells.
Rotate learned-arm execution order by seed index, fixed in source.

The objective remains the sum of four equally weighted components:

1. Mean squared blind centered-cost error.
2. Mean squared observed centered-cost error.
3. Half the sum of blind and observed survival MSE.
4. Five-event soft cross-entropy for the observed prediction.

Costs use scale 1. No DEV normalization or additional oracle loss is allowed.
Soft cross-entropy at an exact prediction equals target entropy, so a positive
total objective is not itself prediction error or evidence of failed learning.
Report the components and forecast errors separately where recorded; do not
claim convergence merely from completing 480 epochs.

All 12 final checkpoints and their durable barrier must precede any DEV
generation or decoding. Evaluate the final checkpoints at horizons 1, 2, 4, 8.
Report every learned arm and seed, plus both references. Report actual
parameters, trainable parameters, storage, updates, cases and measured time;
equal updates do not imply equal compute. With N retained TRAIN cases, the
learned fits complete `12 * 480 * ceil(N / 64)` optimizer updates and
`12 * 480 * N` case exposures.

## Three separate criteria

Decision regret is exact expected cost of the chosen decision minus the
minimum exact expected cost. Learned predictions use raw argmin. Only the
uniform-state reference uses the frozen lowest-index tie rule within 1e-12 of
its minimum. Means summarize separately evaluated policies, not an ensemble.

Evaluate each criterion separately for each of the three learned factor cells.
Every fit seed must pass each applicable threshold; means cannot rescue a
failed seed. Both support minima apply. Required reference MSE and regret must
be strictly positive; otherwise the relative criterion does not pass.

| Criterion | Required conditions for all three seeds |
| --- | --- |
| SHORT_HORIZON_LEARNING | At each of H1/H2, blind cost MSE and regret are at most half the corresponding uniform reference; observed event-law KL is at most 0.1 nats. |
| BLIND_EXTRAPOLATION | At each of H4/H8, blind cost MSE and regret are at most half the corresponding uniform reference; H8 blind survival MAE is at most 0.05. |
| OBSERVED_FILTERING_EXTRAPOLATION | At each of H4/H8, observed event-law KL is at most 0.1 nats. |

The last criterion consumes intervening observations and must not be described
as a blind-gap result. Report all per-horizon costs, regrets, survival errors
and KL, including failed or mixed cells. A zero prediction where the target
has positive probability is an explicit numerical failure, never clipped.
The GRU remains descriptive and cannot be promoted into a prespecified factor
candidate after seeing results. There is no architectural superiority gate.

Retain a descriptive prefix-shuffle check with fixed cyclic offset 1 within
DEV. Move the complete public prefix and its length together; an exact-prefix
cell also receives the matching shifted oracle boundary state. Keep the
original forecast actions and targets. This check is not a significance test
or a continuation criterion.

## Interpretation and limits

If both isolated learned components pass a criterion but their joint cell
fails it, that supports a coupled-learning explanation under this recipe. If
exact-prefix/learned-operator fails, prefix encoding alone cannot explain that
failure. If learned-prefix/exact-operator fails, the encoder/training path is
insufficient under this budget even with exact dynamics. A short-horizon pass
followed by extrapolation failure distinguishes fitted-horizon performance
from longer-horizon behavior; it does not prove a unique cause.

Neither fixed C nor eight coordinates establishes latent identifiability. The
four centered costs have rank at most three, and predictive equivalences may
remain. Score held-out forecasts and decisions, not purported recovery of a
unique hidden-state basis. Privileged exact-posterior or exact-operator
success is not a deployable-policy or architecture result. Failure after this
larger fixed budget still does not prove nonlearnability or convergence.

Related established ideas inform this diagnostic:
[spectral HMM learning](https://arxiv.org/abs/0811.4413) requires appropriate rank/separation conditions;
its single-event full-column-rank condition cannot simply be assumed for five
outcomes and eight states.
[Predictive State Inference Machines](https://proceedings.mlr.press/v48/sun16.html) emphasize learning filtering on
the learner's recurrent states; their algorithm and guarantees are not claimed
here.
[Value equivalence](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)
distinguishes agreement on selected predictive functions from complete model
identity. This diagnostic claims none of these mechanisms as novel.

## Admission, budgets and closure

Freeze protocol, source hashes, runtime, configuration, exact outputs and a
successful engineering receipt before scientific data generation. Use the
inherited original suspend-inclusive supervisor: qualification 300 s, combined
fit/evaluation 1800 s and independent saved-output audit 600 s. Worker peak RSS
is capped at 4 GiB and each phase output at 512 MiB. Use single CPU threads, no
accelerator, native environment, paid inference or external model calls.

Preserve a failed qualification and its source snapshot. A mechanical repair
requires a fresh engineering registration and exclusive attempt paths; it
does not silently reuse failed evidence. The scientific registration permits
one attempt, with no retry, replacement seeds or cap extension. Preserve any
partial output and close failures honestly.

Close the original producer before the independent saved-output audit.
Authenticate original supervisor joins, source pins and all payload bytes.
The audit independently reconstructs finite-world targets and metrics without
model or optimizer replay. Disclose that historical training order and
checkpoint execution remain authenticated producer attestations. No result
automatically authorizes a native task or supports a robotics, Doom, chess,
connectome, calibration, RL or novelty claim.
