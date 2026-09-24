# Rounded transition primitive qualifies

**36 tests and lint pass.** The original supervised qualification closed
successfully in **1.895500541 seconds**, including startup and cleanup.
All ten registered source files remained unchanged. This is a numerical
qualification on fabricated inputs, with **zero training or task-evaluation
calls**. It does not establish better learning or model performance.

## What changed

The [previous integration attempt](finite-balanced-learning-stop-results.md)
failed when fixed 64-sweep balancing exceeded its 1e-12 residual tolerance.
That attempt remains closed. The new function uses four fixed log-domain
sweeps, contracts rows and columns to at most `1-1e-8`, and fills the remaining
mass deficits with a rank-one correction. It keeps the same final **1e-12**
row/column tolerance and rejects failures without additional iterations.

This adapts established [optimal-transport rounding by Altschuler, Weed and
Rigollet](https://arxiv.org/abs/1705.09634), not a new architecture. Their
[reference implementation](https://github.com/JasonAltschuler/OptimalTransportNIPS17/blob/master/algorithms/round_transpoly.m)
uses row/column contractions and residual correction. Our fixed sweep count
and positive slack define a separate finite function; their full theoretical
claims are not automatically claims about this implementation.

The correction changes the output. In exact arithmetic, an already balanced
matrix X becomes `(1-1e-8) X + 1e-8 U`, where U is uniform. Less balanced
inputs can change more. The implementation records pre-round residuals,
deficit totals, correction mass and maximum change. It returns the log of the
**final** probabilities for later use in a matching probability prior.

## What was checked

- Uniform and positive permutation-mixture closed forms, including the
  declared smoothing of an already balanced matrix.
- Independent NumPy reconstruction, final row/column sums, strict positivity
  and correction diagnostics.
- Analytic uniform-input derivatives and directional finite differences
  away from branch changes, including gradients through final log probabilities.
- State/action relabeling, representable offsets, selected extreme inputs,
  input preservation and separate output storage.
- Fixed work counters, external stops, invalid inputs, overflow rejection
  and rejection of numerical tuning overrides.

Four sweeps replace 64 in the declared function, but this qualification is
**not a latency comparison**. The new correction has its own cost, and no
backward-FLOP or training-efficiency gain was measured. The function remains
piecewise differentiable and still permits completely uniform mixing.

The [next step is integration](finite-rounded-learning-next.md), followed by a
separately registered learning comparison with original and matched-initialization
controls. There is no new scientific performance result, biological-wiring
advantage or ICLR-level contribution established here.

[Protocol](finite-rounded-transition-qualification-protocol.md) ·
[Summary](finite-rounded-transition-qualification-results/summary.json) ·
[Original sources, logs and process closure](finite-rounded-transition-qualification-results/evidence.tar.gz) ·
[Publication receipt](finite-rounded-transition-qualification-results/receipt.json).
