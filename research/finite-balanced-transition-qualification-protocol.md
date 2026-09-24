# Qualify a balanced transition primitive on fabricated inputs

Prospective numerical qualification **finite-balanced-transition-qualification-v1**.
This implements the first engineering step in [the balanced transport proposal](balanced-transport-next.md).
It trains no model, opens no saved checkpoints and creates no empirical cases.
Existing experiments and source pins remain unchanged.

## Fixed computation

Input is one CPU float64 tensor with shape `[action=4,next=8,current=8]`.
Starting from logits, execute **64 complete sweeps** of row logsumexp
normalization over current state, then column logsumexp normalization over
next state. Exponentiate after the final column normalization. Return both
the normalized log probabilities and probabilities with their autograd graph.

Require finite input and output, strict `0<T<1`, and maximum absolute row and
column sum residuals at most **1e-12**. Failure stops the call. No clipping,
extra sweeps, temperature, hidden retries, alternate model or custom backward
is allowed. The gradient is of this finite computation, not an exact
infinite-iteration Sinkhorn map. The API accepts one to 64 fixed sweeps and
a tighter positive tolerance for explicit failure and validation witnesses;
this qualification does not select a training hyperparameter.

Balance T only. No hazard-weighted operator, emission, cost readout, history
filter or optimizer is implemented by this primitive. A uniform T remains
possible and destroys all zero-sum state distinctions.

## Independent fabricated witnesses

Tests cover closed-form uniform and rank-one inputs, positive
permutation-plus-uniform matrices, asymmetric positive matrices, additive
logit offsets, row/column permutation equivariance, strict residuals, owned
outputs, unchanged inputs and preserved autograd. Check invalid shapes,
dtypes, devices, nonfinite values, underflow and a deliberately insufficient
iteration cap. Check callback failure propagation and exact structural work.

At uniform logits, compare gradients with the analytic double-centering
Jacobian. Independently compare directional gradients with central finite
differences of the actual finite algorithm on a fabricated asymmetric input.
Tolerances reflect float64 roundoff or truncation error; success thresholds
are fixed in the source snapshot before the first numerical execution.

## Bounded execution and evidence

Freeze this protocol, the primitive, its tests, qualification worker, package
initializers, project configuration, lock file and native supervisor/clock
before running lint and only the selected test file. Store a complete source
snapshot, runtime versions and single-thread environment in an exclusive
registration. Disable pytest plugin autoload, conftest loading and cache writes;
require empty environment overrides for pytest options and plugins.

One original native supervisor allows **90 seconds**, including worker setup,
lint, test-process startup, all checks and process-group cleanup. Phase log
files must total at most **8 MiB** when checked after each command. The worker
reports self/child peak RSS at closure; no continuous RSS bound is claimed.
Preserve all failures, original logs and terminal receipts. No timeout retry
or scientific run follows automatically from a pass.

After the original process closes, authenticate its source and runtime joins,
launch identity, complete logs, status and cleanup. Publish the test count and
all evidence. This pass would establish numerical primitive behavior on the
specified witnesses only. It would not establish model integration, convergence
on arbitrary positive matrices, learning reliability, improved gameplay or a
novel recurrent architecture.

The next integration step must use the same balanced probabilities in both
filtering and the prefix log prior. The existing prefix bridge hardcodes
column-softmax log probabilities, so it cannot be reused unchanged for this
candidate. Preserve the distinction between prefix-only training's log prior
and joint minibatch training, which currently has no such prior.
