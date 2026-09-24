# A direct-solve readout control

This is unregistered future work. It does not change the completed
[calibrated-cost experiment](otto-readout-compute-protocol.md), and no solver,
new cache, empirical fit or effectiveness result is implemented here.

The residual-only head never feeds the GRU, query innovation or raw anchor.
Conditional on a fixed cache of hidden states and base forecasts, its supervised
loss is a convex quadratic in real arithmetic. A direct solve would test whether
ordinary Adam leaves this cheap baseline underoptimized before attributing a
recurrent model's advantage to representation learning.

## Objective

Let `z = [hidden; 1]` have 29 entries, `Theta = [W b]` be the 4-by-29 residual
head, and `C = I - 11'/4`. The code applies `64 * Linear(hidden)` before
centering and adds this to the frozen base forecast `base`. Its normalized
prediction is `base / 64 + C Theta z`.

For legal-action indicator `l` with `m = sum(l)`, define
`P = diag(l) - l l' / m` and `y = (teacher - base) / 64`.
Since `P C = P`, the complete TRAIN objective is

$$
J(\Theta)=\frac1{54}\sum_e\left[
\frac1{n_e}\sum_{r\in N_e}\frac{\|P_r(\Theta z_r-y_r)\|^2}{m_r}
+\frac1{k_e}\sum_{r\in Q_e^+}\frac{\|C(\Theta z_r-y_r)\|^2}{4}
\right].
$$

`N_e` contains nonquery rows; `Q_e^+` contains later queries before assimilation.
Their counts are `n_e` and `k_e`. A term with no support contributes zero, but its
episode remains in the denominator 54. The current six-episode batch objective
is the average of those six episode terms. Average, rather than sum, the nine
batch objectives when matching a full-data ridge penalty.

Stack weighted linear blocks `sqrt(w) * (z' tensor P)` and targets
`sqrt(w) * P y`, using `C` for the prequery blocks. Adding `1 a'` to the head
does not change either loss or contrasts: 29 of the 116 coordinates are a
common-action gauge. Use a fixed orthonormal basis `U` of the three-dimensional
action-contrast space and fit `Theta = Theta_parent + U B`. This leaves **87
increment coordinates**. Feature collinearity and missing legal contrasts can
reduce rank further. Single-legal-action nonqueries provide no information.

## Proposed comparison

Compare the unchanged parent, the existing Adam continuation, minimum-norm OLS
in the contrast coordinates, and one positive-ridge solve of
`J + lambda * ||B||^2`. Include increment biases in the penalty to obtain a
unique ridge solution and preserve parent parameters in unsupported directions.
Use QR/SVD with a fixed rank tolerance; do not silently invert a singular normal
matrix. Fix the ridge strength prospectively, or choose it using TRAIN-only
folds that keep all three collectors from each environment case together.

Declare cache geometry and preserve causal query visibility. Charge extraction,
solving, export and ordinary inference. Retain all parent seeds, numerical
ranks, condition diagnostics and failed solves. Export the float32 head into
the normal predictor and score it on separately registered fresh development
cases. Do not select solver choices on the current fresh panel after seeing it.

This is a conventional linear-regression control, not a new learning algorithm.
An OLS optimum bounds its explicitly cached quadratic, not action-cost gap,
autonomous reward or the literal float32 Adam program. Grouped GRU calls,
parameter flags, tensor geometry, centering and float32 addition can change
rounding. Qualification must compare cache/export predictions and scalar losses
with fixed tolerances, preserve all frozen tensors, and disclose discrepancies.
It cannot claim bitwise equivalence without establishing it.

Implementation sources: [scheduled predictor](../src/openjev/research/otto_scheduled_predictor.py),
[masked model](../src/openjev/research/otto_readout_ablation_model.py),
[training loss](../src/openjev/research/otto_query_memory_data.py).
Related engineering constraint: [feature-cache proposal](otto-readout-cache-proposal.md).
