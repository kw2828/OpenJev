# Supervising the forecast used to correct memory

This prospective experiment follows the closed [cross-query comparison](otto-cross-query-forecast-results.md),
which failed its continuation rule. Its ordinary error-fed GRU beat the explicit
correction candidate on mean action-score gap in both settings. That result stays
closed. Source inspection found that the forecast made immediately before each
later query supplies the correction error but receives no direct target loss.
This experiment tests that specific missing supervision. It does not assume the
omission caused the earlier failure or establish a new learning algorithm.

## Two architectures, two objectives

Use the unchanged `innovation` and `innovation_gru` models, with 5,978 and 5,996
parameters respectively. Both consume the same 31 public features and actual
teacher scores only at absolute steps 0, 4, 8 and so on. Both retain caller-owned
hidden memory across queries and chunks. At a later query, they first process
the current features with the old actual score anchor and compute a forecast.
Only then does the current observed score enter the correction. The explicit
candidate adds `tanh(B e)` to hidden state; the ordinary control processes the
error and new anchor in a second GRU update. Both return the actual score at
query rows. Forecasts between queries are offsets from the last actual anchor,
not recursively accumulated predicted scores. The scale stays 64.

The new wrapper exposes that already-computed prequery forecast with its gradient
intact. It adds no parameter, recurrent update or readout call. Its ordinary
predictions and final carry must match the original implementation at identical
weights. The first query has no prequery forecast or auxiliary target. Padding
and ended lanes cannot advance state or enter the auxiliary objective. Exposed
prior predictions cannot depend on the current query score or future scores.

| Reporting cell | Architecture | Objective |
|---|---|---|
| `innovation_aux` | Explicit correction | Original loss plus prequery loss |
| `innovation_mse` | Explicit correction | Original loss |
| `innovation_gru_mse` | Ordinary error-fed GRU | Original loss |
| `innovation_gru_aux` | Ordinary error-fed GRU | Original loss plus prequery loss |

The candidate is `innovation_aux`. The two MSE cells control for changed data and
fit seeds; the GRU auxiliary cell tests whether the supervision helps an ordinary
recurrent baseline as much or more. This is supervised forecast training on fixed
controller trajectories, not an autonomous policy experiment, RL, Bayesian
inference, a biological connectome or a calibrated uncertainty model.

## Fresh data and unchanged sampling

Collect 54 TRAIN and 36 VALID complete paths, using analytic, always-neural and
period-four-held-Q controllers for each originating case. Each sensing length
3 and 4 has nine TRAIN cases and six VALID cases. Preserve the 2,188-step horizon,
initial-hit stratification, rotating controller order, paired source and indexed
observation streams, public filter and original eight-view TensorFlow teacher.
Keep every path through discovery or horizon, including its final update. Three
controller paths from one originating case are not three independent cases.

TRAIN seeds are 271000001-271000009 and 272000001-272000009. VALID seeds are
273000001-273000006 and 274000001-274000006. Fit seeds are 275000001, 275000002,
275000003. TRAIN selection seeds are 276000001-276000054 in global TRAIN episode
order. Reserve all 87 only after a saved scoped collision review, before collection.

For episode length T, let W=ceil(T/4), k=min(8,W), and M=T-W. Use the unchanged
PCG64 sampler to draw k disjoint period-four windows uniformly without replacement,
including short or query-only tails. TRAIN annotations are the union of these
windows and all actual query rows. Reuse deployment returns; reconstruct missing
scores only after the completed path, verifying every public posterior witness
and final update. Annotation cannot change actions or the deployed cache. VALID
keeps a complete teacher-score census. No new teacher call is needed specifically
for prequery supervision: these later-query scores already enter both models.

## Fixed objectives and fitting

The original loss centers predictions and targets over legal actions, divides by
64, and averages their squared difference over legal actions. Each selected
nonquery row has weight W/(k*54*M), or zero if M=0. Do not renormalize realized
sample weights. Unselected nonquery labels cannot enter either training objective.

For prequery supervision, let K=ceil(T/4)-1. At each actual query at absolute
step 4 or later, center the exposed prior prediction and observed target over
all four coordinates, divide by 64 and take mean squared difference. Use all four
coordinates because the existing correction centers all four error coordinates,
including actions that are currently illegal. Give each such row weight 1/(54*K).
If K=0 all auxiliary weights are zero. All 54 episodes stay in the denominator.
The auxiliary coefficient is fixed at one; there is no tuning or rescaling from
realized target statistics. The combined objective is the sum of these two losses.
The MSE branch must not numerically inspect auxiliary targets or weights.

Each of 12 fits uses 80 epochs, CPU float32, Adam at 0.003, zero weight decay,
gradient norm clipping at 5 and six complete episodes per batch. A local generator
permutes all 54 episodes each epoch, identically across cells at each fit seed.
There are nine updates per epoch, 720 per fit and 8,640 total. Process chronological
chunks of 32 with parameters fixed across a batch. Accumulate chunk gradients,
detach state at chunk boundaries, then clip once and update once after the batch.
Multiply each weighted batch sum by 54 divided by its actual episode count.
Zero-target chunks can use no-grad; positive auxiliary weight makes an auxiliary
chunk a gradient-bearing chunk even when its nonquery weight is zero. Record
actual forward, backward and optimizer work and wall time separately. Equal
epochs and optimizer counts do not imply equal training compute. Zero-target
batches still receive a zero-gradient Adam update for every parameter.

Save exact projected inputs, separate targets/weights, episode orders, costs,
checkpoints and final TRAIN rescoring. Complete all 12 final checkpoints before
decoding VALID. Save ordinary predictions and the prequery predictions/masks so
an independent audit can reconstruct both metrics without calling a model. No
checkpoint selection, early stopping, optimizer change or fit after VALID exposure.

## Fixed continuation rule

Preserve the existing action-score metric: primary nonquery rows start at absolute
step 5; steps 1-3 appear only in full-trajectory results. Every episode within a
reported setting or age retains equal denominator share; unsupported episodes
contribute zero and are listed. Predictions select the first legal index whose
float32 score difference from its minimum is strictly less than float32(1e-10).
Agreement accepts any teacher near-minimum action, with first-argmin agreement
reported separately. Raw gap is the chosen action's teacher score minus its best
legal score. Report all fits and cases, plus support for every aggregate.

The new calibration metric is centered all-four prequery MSE in squared raw score units,
episode-balanced over actual later queries. Preserve zero-support episodes and
report their count; do not report that quantity as conditional accuracy. The
auxiliary training loss has the additional scale factor 1/64^2.

Advance only if all **55 conditions** pass:

1. One technical condition: complete collection and fits, finite evidence, original
   supervisors and independent saved-output audit close.
2. Six support conditions: at least four distinct originating VALID cases contribute
   primary rows for each age 1, 2 and 3 in each setting.
3. Twelve hold conditions: each candidate fit in each setting has agreement at least
   hold and raw gap at most 80% of hold.
4. Eighteen age conditions: each candidate fit/setting/age has gap no larger than hold.
5. Twelve learned-control conditions: candidate three-fit mean agreement is at least
   each of the three controls in each setting; mean gap is strictly smaller and at
   most 90% of each control. The strict comparison prevents zero gaps from counting
   as architectural improvement.
6. Four full-path conditions: candidate mean agreement and raw gap do not regress
   against `innovation_mse` in either setting.
7. Two prequery conditions: candidate mean prequery MSE is at most 80% of
   `innovation_mse` separately in each setting.

This is a development screen, not a significance test or paper novelty threshold.
A pass permits a separately frozen fresh autonomous and scenario-shift comparison
with total computation charged and fuller reference baselines. If the ordinary
error-fed GRU still wins, retire the explicit B-correction candidate rather than
attribute the gain to a novel architecture. Report negative results without selecting
a favorable fit, setting, age or subset.

## Admission and execution boundaries

Before any fresh collection, hold all source, this protocol, fabricated engineering
tests and a new synthetic capacity qualification. Capacity uses six fabricated
episodes of length 2,188, chunks of 32, period-four queries, eight selected nonquery
windows per episode and all later queries as auxiliary targets. Measure a complete
chronological batch, gradient accumulation, clipping and one optimizer update for
each of the four reporting cells. No recorded path or teacher is used.

Freeze admission before measurement as
`1.5 * 3 * 720 * sum(four full-batch seconds) + 120 <= 8100` seconds.
The 50% headroom and fixed 120 seconds cover expected later rescoring, evaluation
and closure inside a 10,800-second training allocation. It is a planning screen,
not a guarantee. Synthetic capacity allocation is 120 seconds, CPU1, 4 GiB RSS,
128 MiB output. A failure preserves the attempt; only separately qualified
engineering may be revised before scientific collection, without shrinking the
scientific comparison or speculatively collecting the cohort.

Collection allocation: 7,200 seconds, CPU1, 4 GiB RSS, 2 GiB output. Teacher-call
cap stays 138,708: 10,278 analytic TRAIN, 39,384 neural TRAIN, 10,278 held TRAIN,
78,768 VALID. Bounds are 90 native resets, 196,920 moves, 54 public replay resets,
118,152 replay updates and 144 teacher-policy bindings. Setup, annotation, I/O,
evaluation and cleanup count. Training allocation is 10,800 seconds, CPU1, 4 GiB
RSS, 2 GiB output, covering all 12 fits and scoring. Saved-output audit allocation
is 120 seconds, CPU1, 2 GiB RSS, 128 MiB output, with no new teacher, environment,
model or optimizer calls.

Each phase binds its original inputs and supervisor, uses an exclusive output
directory and preserves partial files and failure receipts. No scientific retry,
resumption, replacement seed, time extension or fitting on incomplete collection.
Any engineering failure discovered before collection remains recorded with its
source revision and new qualification receipt.

## Related work and claim limits

[Recurrent Kalman Networks](https://proceedings.mlr.press/v97/becker19a.html) and
[Gated Delta Networks](https://arxiv.org/abs/2412.06464) establish substantial prior
art for observation-driven latent corrections and error-based recurrent memory.
[Policy distillation](https://arxiv.org/abs/1511.06295) supplies prior art for score
supervision. These small GRUs do not replicate those architectures. The narrow
question is whether directly supervising the forecast used in a correction closes
an observed gap to an ordinary recurrent control, and at what added training cost.
