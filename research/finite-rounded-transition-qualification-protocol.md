# Fixed rounding for balanced transition probabilities

Prospective fabricated-input qualification **finite-rounded-transition-qualification-v1**.
The preceding fixed64 balancing integration stopped at its original tolerance
before any scientific fit or DEV generation. That attempt remains closed.
This proposal defines a different forward function; it neither repairs the
old checkpoints nor reruns the old training attempt.

## Fixed function

Input: finite CPU float64 logits with shape (4,8,8), indexed
`[action,destination,origin]`. Execute exactly **four** row-then-column
log-domain normalization sweeps. Exponentiate. With fixed **delta=1e-8**,
contract every row by `(1-delta) / max(row_sum,1-delta)`, then every column
by `(1-delta) / max(column_sum,1-delta)`, using the updated matrix for the
column sums. Let the resulting matrix be X. Set row deficit `r=1-X1` and column
deficit `c=1-X^T1`. Return `T=X + outer(r/sum(r),c)` for each action, dividing before the
outer product to reduce underflow risk.

In exact arithmetic the contractions leave positive row and column deficits,
which have the same total mass. The rank-one term fills both deficits. The
slack keeps its denominator away from zero and adds a small explicit smoothing
bias. This adapts the row/column scaling and residual correction in
[Altschuler, Weed and Rigollet, Algorithm 2](https://arxiv.org/abs/1705.09634).
The four sweeps and positive slack are our declared adaptation, not a claim
to reproduce their exact algorithm or inherit its full theorem. It is not generally the
same matrix as the infinite Sinkhorn limit. Differentiation follows the four
finite sweeps, contractions, max branches and rank-one term. No derivative of
an infinite fixed point is claimed, and kink points require explicit handling
in tests rather than a smoothness claim.

Return log of the **final rounded T**, so a subsequent probability prior could
share the actual forward probabilities. Do not expose pre-rounding log values
as final log probabilities. Use no tensor cache or numerical fallback.

Acceptance: finite strict `0<T<1`, finite final log T, maximum row and column
mass residual at most **1e-12**. All intermediate deficit totals must be
positive. Finite input logits do not guarantee arbitrary floating-point
operations succeed; explicit guards reject nonfinite intermediates and failed
bounds. Keep the tolerance fixed; no added sweep, post-failure rescaling or
retry. Count all sweeps, exponentials, contractions, sums, rank-one work, logs
and external checks with retained partial counters on failure.

## Qualification scope

Tests use fabricated inputs only: uniform matrices, positive permutation
mixtures, asymmetric logits, strongly imbalanced inputs, large common offsets,
state relabeling and owned outputs. Independently reconstruct the four-sweep
and rounding arithmetic in NumPy. Use analytic or central finite-difference
checks away from max kinks, verify autograd and returned final logs, and test
invalid inputs, overflow, explicit stopping and work accounting.

No scientific arrays, trained checkpoints, data generator, optimizer, model
fit or task evaluation is permitted. Passing licenses only subsequent model
integration qualification. It does not establish task performance, calibration,
memory preservation, connectome efficacy or a novel model.

## Original execution boundary

Freeze kernel, tests, qualifier, protocol, supervisor/clock, package init files,
pyproject and lockfile in an exclusive registration and full source snapshot
before the first numerical invocation. Run the exact registered Ruff and
pytest commands, with no ambient pytest plugins or conftest, no bytecode,
and single-thread numerical libraries. Native suspend-inclusive cap **90 seconds**,
logs **8 MiB**. Peak RSS is recorded, not a continuously enforced limit.
Require the original process exit, clean reaping, source hashes and log hashes.
Retain any failure and its snapshot. Do not change the frozen function or
interpret a failed attempt as qualification.
