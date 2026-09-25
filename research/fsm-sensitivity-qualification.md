# Recurrent sensitivity penalty: arithmetic qualified

The conditional [sensitivity proposal](fsm-sensitivity-experiment-draft.md) now
has a separate implementation with **43 fabricated tests plus lint passing**.
This is a training-loss component for the existing tanh feedback model, not a
new architecture or a measured performance result. No training study, development
evaluation or reserved-data access was launched.

The hypothesis remains specific: penalizing sensitivity added beyond a frozen
linear model might improve forecasts more than ordinary shrinkage or penalizing
all sensitivity. The [prior-art review](fsm-sensitivity-prior-art.md) explains why
the broader linear-plus-residual and sensitivity ideas are established.

## What the implementation does

The [new helper](../src/openjev/research/fsm_sensitivity.py) propagates two kinds
of values through the complete recurrent prediction: the ordinary predicted
outputs, and how those outputs change when the observed output history changes.
The derivative includes every lag shift and later residual evaluation. Past and
future inputs remain unperturbed. A separate frozen-linear derivative provides
the comparison at each requested horizon.

The supplied sampler draws unit Rademacher directions from an explicit caller
generator. Gain is the squared directional output norm divided by `3h`, for each
window, direction and horizon. The relative penalty uses each direction's own
linear gain. The max-envelope control takes the largest linear gain across the
directions of that same window and horizon.

There are six unweighted penalty terms matching the draft: none, parameter L2,
per-step residual magnitude, squared total sensitivity, max-envelope excess and
direction-specific excess. The arm-selective API computes only the requested
term. Unregularized and L2 calls need no trajectory; residual magnitude needs
only the neural corrections; total sensitivity needs no baseline derivative.
A primal-only rollout lets the simple controls avoid derivative work. The bulk
API is explicitly for diagnostics and is not a fair training path for those
simple controls.

## What the tests establish

- Exact predicted outputs and final states match the unchanged native model.
  The primal-only path also preserves parameter, input and initial-state gradients.
- An independently constructed linear state matrix reproduces the baseline
  derivative, including order-32, horizon-128 requests and input-lag shifts.
- Nonzero nonlinear derivatives match independent differentiation of the native
  rollout and centered differences of the starting history.
- An active penalty's gradients match independent autodiff and centered parameter
  differences for all 29 parameters of a small nonlinear fixture. This includes
  the output bias, whose derivative effect appears through later recurrence.
- At the exact-zero output-head initialization, both excess penalties and every
  parameter gradient are exactly zero at horizons 8, 32 and 128. Total sensitivity
  deliberately has no such zero-penalty invariant.
- Hand calculations check normalization and per-window direction reductions.
  Additional fixtures check causal prefixes, caller ownership, random streams,
  nonfinite failures, and isolation from unused penalties.

The [original qualification receipt](../output/fsm-sensitivity-engineering-v1/arithmetic-qualification-01/receipt.json)
retains commands, logs and source snapshots. The [source review and closure](../output/fsm-sensitivity-engineering-v1/qualification-closure.json)
bind the unchanged model source and runtime. Test duration is not a training or
inference benchmark.

## One-update learning path

The [learning-step primitive](../src/openjev/research/fsm_sensitivity_learning.py)
now applies the selected penalty to the actual supervised objective, propagates
its gradient through the recurrence, clips with the native norm policy and takes
one ordinary Adam step. It accepts caller-supplied context, future inputs, targets
and paired directions. It does not load data, schedule a study or select a
checkpoint. Simple arms use the primal path; all three sensitivity arms use the
same learned and frozen-linear tangent propagation.

Its [16 fabricated tests](../tests/test_fsm_sensitivity_learning.py) pass with lint:

- H128 unregularized updates match the native model bitwise, including existing
  Adam moments, step counters and frozen coefficients.
- Both excess penalties preserve the first zero-head update bitwise. Active L2,
  relative and max-envelope penalties contribute the independently assembled
  gradients and produce the expected clipped Adam update.
- Branch witnesses verify reported work and the absence of tangent work for
  simple controls. Targets change the loss and update without changing the
  preceding prediction or mutating caller inputs.
- Invalid gradients, overflowing native norms and mismatched optimizer ownership
  fail before an optimizer update. Returned diagnostics contain scalars rather
  than graphs or a persistent trajectory cache.

The [qualification receipt](../output/fsm-sensitivity-engineering-v1/learning-qualification-01/receipt.json)
and [source review](../output/fsm-sensitivity-engineering-v1/learning-qualification-closure.json)
retain this evidence separately from the 43 arithmetic tests. These are short
updates on fabricated tensors, not the proposed 18 empirical fits. Full training
schedules, recovery-free failure retention, paired directions across arms,
runtime qualification and independent result checks still need a study runner.

## Conditions for a useful result

The [conventional-reference comparison](fsm-author-nllfr-factorial-protocol.md)
still has to close before deciding whether to run the proposed six-arm study.
That study needs its own registration, matched initial weights, batches and
directions, complete training-cost accounting and the declared continuation rule.
No existing checkpoint or scientific criterion changed during this preparation.

Two sampled directions do not bound worst-case sensitivity. A smaller penalty
does not establish better forecasts, preserved input response, stability or
architectural novelty. The prospective fixed-coefficient screen also cannot
establish superiority over tuned regularizers. The implementation is ready for
further qualification if the stronger-reference result justifies that experiment.
