# Common action-cost regression objective

The [new component](../src/openjev/research/otto_cost_regression.py) passes twelve independently authored [fabricated tests](../tests/test_otto_cost_regression.py). It prepares two target families and one common loss for the proposed [teacher-cost learning comparison](otto-action-cost-followup.md). No empirical targets were decoded, no optimizer ran, and no policy was fitted in this qualification. One test constructs an unfitted ordinary head to check its gradient path.

For each state, center costs over its eligible actions. The analytic arm divides the centered heuristic scores by `0.25 * max(eligible_range, 1e-8)`. This recovers the centered negative logits underlying the historical analytic preference distribution. The continuation arm centers the supplied mean capped continuation costs directly, retaining their relative action gaps.

Each episode receives equal total training weight. Each target family then uses one weighted RMS scale computed over the entire supplied TRAIN cohort, with a numerical floor of `1e-8`. The continuation arm has no per-state range normalization. A tiny action gap consequently remains small relative to gaps at other states. Original float64 centered targets are retained alongside the float32 training casts; casting can still round very small differences.

Both arms use the same eligible-action mean squared error. Predicted and target costs are centered separately, blocked actions contribute no error or gradient, and losses are averaged across the qualified head's eight D4 training views before the caller applies episode weights. The component adds no optimizer, clipping, entropy loss, hard winner labels or inference ensemble. It supports the existing ordinary `dense_augmented` head.

The tests cover analytic preference equivalence, preservation of tiny cost gaps, exact weighted scaling, episode duplication invariance, tied and single-action rows, immutable ownership, invalid inputs, all eight transformed views, a separate gradient oracle, blocked gradients and invariance to additive prediction offsets. [Original test and lint receipt](../output/otto-cost-regression-component-v1/engineering-01/receipt.json).

This is a supervision comparison under a new shared regression objective. It is not a replication of the old cross-entropy training recipe, proof that continuation estimates are precise, or evidence of policy improvement. Sampling allocation, training and fresh autonomous evaluation remain separate work.
