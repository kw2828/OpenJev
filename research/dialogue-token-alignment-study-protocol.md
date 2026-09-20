# Conditional token alignment comparison

Prospective scientific protocol. Execution requires a completed, admitted
[capacity measurement](dialogue-token-alignment-capacity-protocol.md) and a
separately published source/input/order freeze. This document does not by
itself authorize resuming a failed capacity screen or reducing its workload.

## Question and controls

Does bidirectional token alignment improve actual branch decisions beyond
both an input/parameter-matched token-mean comparison and the existing
candidate-attention scorer? This is an adaptation of established
align/compare/aggregate attention, not a new architecture claim. No recurrence,
biological wiring, Bayesian update, calibration guarantee, or new RL algorithm
is part of this experiment.

Train three arms from scratch at seeds 6201, 6202 and 6203: `flat_stratum`,
`token_mean`, `token_aligned`. Rotate arm execution order by seed. Copy all
initial tensors between the two new arms; share the explicitly named common
query/candidate projections, feature layer, head and gate with the baseline.
Retain the baseline's own attention and turn projection. Report both total
registered parameters and the shared subset, including inactive entropy
columns and common softmax shifts. Do not call all three arms equal capacity.

The new arms receive the same exact frozen schema tokens, dialogue tokens,
lexical features, public candidate types and privileged correct previous value.
Their only intervention is alignment versus the opposite sequence's
prior-weighted mean. Both use shared projection, comparison and aggregation
networks and the same flat candidate-plus-branch-gate normalization. All
candidates remain present, independent of the current label. No joint
candidate/dialogue Transformer encoding is added.

## Data and training

Reuse the original typed study's fixed original-TRAIN split: 29,211 fitting
rows and 13,599 evaluation rows. Primary evaluation contains 7,819 rows from
six held-out services, of which 578 change and 7,241 retain the previous value.
These are historically exposed development examples, not untouched evidence.
Official DEV inference and TEST access are excluded.

Use all frozen orders from the admitted capacity freeze, 20 epochs, effective
batch 256, row microbatch 32, CPU float32, deterministic operations, four
intra-op threads and one inter-op thread. AdamW uses learning rate .001 and
weight decay .0001. Divide microbatch weighted-loss sums by the full effective
batch size, including its final short batch, then accumulate gradients. Clip
norm at 1 and update once per effective batch. Use ordinary three-stratum
weights from fitting counts 17,666 / 9,246 / 2,299; no rare-type balancing.

There are 2,300 optimizer updates per fit and 20,700 overall. Every arm uses
the same orders and all observations. Save final weights and all evaluation
log probabilities after epoch 20. No validation-based checkpoint selection,
early stopping for quality, interim quality scoring, alternate seeds, resumed
fits, or omitted failed arms. Save every update's rows, loss, work counts and
normalization witness. A failure stops the entire study and preserves partial
weights, predictions, counters and the original error.

The whole nine-fit attempt has a 3,600-second wall ceiling, 6-GiB process peak
RSS limit and 512-MiB output cap. Include input authentication, gathering,
padding, fitting, evaluation, serialization and hashing. Attribute previous
frozen-encoder preparation and the new schema cache separately. These are
local development timings under recorded process load, not a Rust/Python or
hardware speed comparison.

## One continuation rule

Score only after all nine fits complete and their saved artifacts pass
authentication. Compare alignment separately with each control. On the same
578 primary changed rows, require:

- Mean accuracy across three seeds improves by at least 2 percentage points.
- Mean wrong-selected-branch rate falls by at least 2 percentage points.
- Neither accuracy nor wrong-selected-branch rate worsens in any paired seed.
- Mean retained-state error increases by at most .5 percentage points.
- Mean supported TRUE false-positive rate increases by at most .5 points.
- Mean supported DONTCARE false-positive rate increases by at most .5 points.

All conditions must hold against both controls. Compare exact counts and
rational thresholds where possible. Branch refers to the selected candidate's
NONE, DONTCARE or concrete branch, not the argmax of summed branch mass.
Retained-state error uses the 7,241 primary retained rows. Each supported
false-positive denominator uses all primary rows whose supplied candidate set
contains that type and whose current target is not that type.

Report all fits, per-seed and equal-service metrics, raw candidate/branch/value
NLL, Brier score, actual error partitions, and paired repairs/new mistakes.
Report TRUE and DONTCARE changed recall descriptively: support is only 29 and
five rows respectively, with no FALSE changes or clears. Repeated seeds do
not create new independent examples. Avoid calibration claims from lower NLL
or Brier alone. Retain the previous-value and literal-register references.

Success supports further development of this observation component only. If
token mean performs similarly, the results do not establish a benefit specific
to alignment. If the rule fails, preserve that result without relaxing its
thresholds or attributing failure to missing memory. A novel recurrent-model
claim needs a separate memory-dependent task and stronger untouched evidence.
