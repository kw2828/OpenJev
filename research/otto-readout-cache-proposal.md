# Frozen-feature residual training: unregistered engineering proposal

This is a possible implementation improvement, not an admitted experiment or a
measured speedup. It does not change the registered readout/state ablation or the
immediate compute comparison. The immediate comparison retains the existing
scheduled forward and training kernel.

## What can be cached

For `action_residual_only`, the six frozen GRU/base-readout tensors determine all
hidden states and uncorrected forecasts from public observations and actual P4
teacher queries. The 116 trainable residual parameters affect only nonquery
actions and later-query shadow forecasts. They do not enter query innovation,
the recurrent carry, or raw anchors. Frozen recurrent features and base forecasts
can therefore be reused while fitting that head.

This does not extend to `both_readouts`: updating the base readout changes later
query innovation, which changes subsequent hidden states even when GRU weights
are frozen. Full-joint features also change after each update.

Source: [scheduled forward](../src/openjev/research/otto_scheduled_predictor.py),
[parameter masks](../src/openjev/research/otto_readout_ablation_model.py), and
[unchanged training recipe](../src/openjev/research/otto_readout_ablation_training.py).

## Exact parity is more demanding than mathematical equivalence

The current trainer shuffles 54 episodes independently every epoch, then takes
ordered batches of six. The scheduled kernel groups nonquery sequences by
remaining length, and uses different active-lane groups at episode tails.
Changing batch membership, gradient context, tensor strides, or Linear call
shapes can change float32 rounding. A once-per-episode cache computed at batch
size one is mathematically equivalent but cannot be called bitwise equivalent
to current batch-six training without evidence for the admitted runtime.

A cache keyed by each complete ordered six-episode batch could preserve existing
geometry, but would require a frozen forward for every distinct batch. With 360
shuffled batches per 40-epoch fit, reuse is not guaranteed. Building that cache
solely for one fit may add overhead without removing recurrent work.

A cheap strict-parity candidate for a separate study is to fix nine ordered
groups of six per seed and shuffle only group order each epoch, in both arms.
Cache those nine frozen residual-arm forwards once. This changes the historical
minibatch recipe and must be registered explicitly; it is not a transparent
optimization of the current shuffled-episode experiment.

## Proposed narrow interface and invariants

An independent `build_batch_cache(model, data, indices)` would accept only the
residual-only model and one ordered complete TRAIN batch. It would bind the six
frozen tensors, original source/runtime identity, data identity, P4 schedule,
chunk boundaries, and indices. Each 32-step cache packet would retain base
predictions, causal pre-assimilation priors, and the exact nonquery and shadow
residual input blocks, including shapes, strides, lane maps, and call order.
Copy owned float32 data, preserve parameter flags and CPU RNG, and retain no
autograd graph. Do not cache corrected predictions that depend on the head.

A `cached_batch_update(model, optimizer, cache, data, indices)` would replay only
the residual head and prediction composition, then use the existing
`weighted_loss` and ordinary Adam. Preserve all of the following:

- `64 * Linear(hidden)` before all-four residual centering, then addition to the
  base forecast. Do not algebraically move the scale or center or precompute a
  differently rounded regression target.
- Nonquery legal-centered MSE and all-four prequery MSE in /64 units, each with
  coefficient one. Keep separate eligible-row denominators per episode and the
  complete episode-count / actual-batch-size multiplier. Zero-support episodes
  stay in the denominator.
- Original residual call grouping, ordering, output stack/index operations, and
  chronological chunk-wise backward accumulation before one whole-batch update.
- Exactly the two residual tensors in Adam, learning rate .003, default moments,
  gradient completion, clipping at 5, and one update even for a zero-support
  batch. Frozen weights must remain byte-identical.
- Actual query scores copied unchanged, positive-zero inactive priors/padding,
  no first-query shadow loss, and no target values entering cached features.

Fabricated qualification should compare original and cached predictions, each
loss term, accumulated gradients, clipping norm, parameters, Adam moments and
step counters after one and several updates. Include unequal tails, chunk
31/32/33 boundaries, zero-support paths, masked NaN/Inf poison, nonzero initial
heads, and changed labels after cache construction. Label changes may change
loss/gradients but cannot change the frozen cached features. Reject stale
weights, mismatched batch geometry, shared mutable buffers, and nonfinite cache
values. Test ordinary torch gradient mode and exact input strides; do not assume
`no_grad` or contiguous copies preserve the native call result. If bitwise
parity fails, disclose and separately qualify the numerical variant rather than
loosening the original study's checks.

## Compute comparison

Charge cache construction, validation, allocation, storage, loading and copying
to the residual arm, along with all head updates and ordinary logging. Report
cache-build cost, update cost, peak memory, and total training time separately.
Do not report only steady-state cached batches. Count actual recurrent work as
cache construction, not as if it happened on each subsequent update.

Equal update counts test an implementation speedup for a fixed recipe. Equal
total training time tests whether that saved work produces better utility when
spent on extra updates. These are different comparisons. A future study should
fix the cache recipe, budgets, checkpoints, evaluation paths, and continuation
rule before results. Fresh evaluation paths must be independent of the current
exposed DEV cohort. Existing negative results remain closed and unchanged.
