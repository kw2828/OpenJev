# Next execution candidate: project before dense scattering

Follow-up: this proposal was implemented and [measured](dialogue-token-projection-results.md).
It passes 10/16 speed checks, including all four real-mask cells, but fails
the full qualification. The original prospective design follows.

Status: source-based proposal only. No implementation, speed measurement or
training admission. The [completed mask-packing qualification](dialogue-token-packing-results.md)
failed, and its sources and results remain unchanged.

The packed implementation removes most padded attention scores, then writes
pooled evidence back into a dense 384-wide tensor. The original recurrent
head immediately projects that evidence to 64 dimensions. A distinct next
candidate would perform this projection within each packed group and scatter
the 64-wide result instead.

## Preserve the model

For each supported candidate and real turn, retain the exact operation order:

1. Compute attention and its weighted sum of raw 384-dimensional tokens.
2. Apply the original L2 normalization with epsilon `1e-12` in 384 dimensions.
3. Apply the existing `turn_projection`, including its bias, then `tanh`.
4. Scatter the resulting 64-dimensional value into the dense head input.
5. Run the original feature and recurrent transition equations in turn order.

The original head maps zero evidence to `tanh(turn_projection.bias)`. Skipped
cells must receive that value, not zero. Preserve gradient connections,
including zero versus absent gradients for all-padded input. The current
frozen head always projects its evidence internally, so this requires a new
explicit adapter accepting preprojected evidence inside the existing monitor.
It must not apply the projection twice or bypass feature/state checks.

Projecting raw tokens before pooling and normalizing in 64 dimensions is
not equivalent. Neither is moving `tanh` through the weighted sum. Both would
change the model and need a different scientific comparison.

## Concrete cost hypothesis

The preselected geometry contains 88,320 dense candidate positions and
13,555 supported candidate/turn positions, including dummy NONE support.
The current dense evidence payload is 33,914,880 float32 elements, or
135,659,520 bytes. A 64-wide dense result would contain 5,652,480 elements,
or 22,609,920 bytes: **one sixth of that intermediate payload**.

Turn projection would process the 13,555 packed positions, plus the bias
fill for skipped positions, instead of 88,320 dense positions. The input
assembler, 384-dimensional group pooling and sequential head remain. These
are algebraic work/payload estimates, not peak-memory or runtime forecasts.
Gathering, scattering and backward overhead can still erase the savings.

## Qualification before any training

If implemented, freeze a new protocol and source closure before measurement.
Keep all sixteen failed/passing workload cells, their exact shapes, seeds,
geometry and speed thresholds. Compare directly with the original model,
including complete update cost and all new operations. Reuse the matched
initialization and optimizer reset recipe; retain every numerical and
probability-normalization check.

Tests must cover both heads and attention modes, sparse masks, nonzero
projection bias, all padding, absent gradients, and full input/parameter
gradient agreement. A new bounded cost run must fail closed on any numerical
or speed failure. Passing it would permit a separately declared representative
cost check, not an automatic restart of the incomplete twelve-fit study.

This is a possible implementation improvement for the same model. It supplies
no evidence of a novel architecture, biological learning or better decisions.
