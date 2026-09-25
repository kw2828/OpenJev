# Prospective sensitivity regularization experiment

**Unregistered and unrun.** This is a conditional development proposal, not
authorization to train or open reserved data. The running conventional-reference
check must close and be independently audited first. If it removes the residual
model's advantage, report that result before deciding whether this extra study
is worthwhile. No current NL-LFR outcomes informed this draft.

The [published residual comparison](fsm-residual-results.md) supports a useful
feedback effect on exposed DEV, but does not establish robustness or stability.
Test one narrower hypothesis: **penalizing added finite-horizon sensitivity to
observed-output history improves feedback forecasts more than ordinary
regularization, without suppressing useful input-driven dynamics.**

## Mechanism and closest control

Keep the existing float64 order-32 VARX backbone and width-24 tanh-feedback
residual unchanged in form. Its state contains 96 standardized output lags and
96 input lags. Let `Y_theta[1:h](s,u)` be its complete recursive forecast; let
`Y_0` be the frozen native VARX forecast from the identical state and inputs.
For a seeded unit Rademacher vector `v` supported only on the 96 output lags:

```text
S_theta(h,v) = || D_s Y_theta[1:h](s,u) v ||² / (3h)
R_excess = mean_{h in {8,32,128}, batch, v} max(0, S_theta(h,v)-S_0(h,v))²
R_max = mean_{h, batch, v} max(0, S_theta(h,v)-max_{v'} S_0(h,v'))²
loss = H128 forecast MSE + lambda * regularizer
```

Propagate the tangent through the entire corrected recurrence, including lag
shifts and later residual evaluations. Differentiate the penalty with respect
to all residual weights; no detached states or local-head-only Jacobian.
Two directions per scheduled window use stream `200000+model_seed` and are
shared across arms. The max envelope uses the larger of those same two baseline
gains at each window and horizon. Fabricated qualification must verify that,
with the exact-zero output head, both hinge penalties and their gradients with
respect to every trainable parameter are exactly zero. A mean envelope would
not satisfy this invariant when the two baseline gains differ.
Context inputs and supplied future inputs remain unperturbed; no future output enters
the tangent or initialization. This measures sampled output gain, not a global
Jacobian norm or a stability certificate, and does not guarantee preserved
input response. Lag-shift rows also make strict one-step Euclidean contraction
an inappropriate target here; see the [bounded literature check](fsm-sensitivity-prior-art.md).

The minimum informative comparison is six arms:

| Arm | Added training penalty |
|---|---|
| Unregularized | None |
| Ordinary L2 | Mean squared residual parameters, including biases |
| Residual magnitude | Mean squared residual output along the same rollout |
| Total sensitivity | Mean `S_theta(h,v)²`, with the identical tangent budget |
| Max-envelope sensitivity | `R_max`, with the same baseline computations and tangent budget |
| Proposed relative sensitivity | `R_excess` above |

Total sensitivity provides a no-allowance control. The max envelope provides
a shared threshold without penalizing the starting backbone, but is more
permissive than the direction-specific envelope. A fixed-strength win therefore
cannot isolate directional alignment or establish the mechanism. L2 and residual
magnitude test generic shrinkage. Broad sensitivity penalties
and stable hybrid residual models are established: see
[Frank et al.](https://doi.org/10.1080/00207179.2026.2679228),
[recurrent equilibrium networks](https://arxiv.org/abs/2104.05942), and
[stable recurrent models](https://arxiv.org/abs/1805.10369).
[Guided residual search](https://arxiv.org/abs/2602.22964) already addresses
recursion mismatch. This is learning-rule research on an unchanged architecture;
architectural novelty is not established.

## Matched budget and information

The minimum **engineering screen** proposes **18 fresh fits**: six arms,
seeds 9301/9302/9303, one recipe each.
All use the same 4,779 trainable scalars, frozen coefficients/normalizer,
zero output-head initialization, per-seed initial tensors and saved FIT batches.
Use 2,048 Adam updates, batch 16, H128 MSE, native gradient clipping at 1,
learning rate 3e-4. Propose `lambda=1e-3` for every normalized penalty and zero
for the unregularized arm. These are fixed coefficients, not claims of matched
penalty strength or optimally tuned regularizers. No DEV coefficient, rate,
checkpoint or seed selection. This screen cannot establish superiority over
well-tuned ordinary regularization.

A **mechanism study** would require a separate registration with the same
predeclared strength grid for all five penalties, for example
`{1e-4,1e-3,1e-2}`: 45 penalized fits plus three unregularized fits. Each arm
would select one strength using the same exposed-DEV mean over all three seeds,
with identical update and search budgets; no per-seed selection. Equal grids
still do not guarantee equally effective penalty strengths or bracketed optima.
This is a possible later design, not an expansion or authorization of this screen.

Reuse only the twelve existing FIT records for gradients and twelve exposed
DEV records for final evaluation, with periods separate and the original
C100/99-input/H128 alignment. Neither 300mV nor official test is admitted.
Retain failed attempts without repair or replacement. Propose 600 seconds per
fit and 12,600 seconds overall, frozen after fabricated throughput qualification.

Equal update counts do not equal compute. Report objective/tangent/backward
work, full training duration, peak workspace and failures. At inference the
penalty disappears: retain the same complete-request accounting and 44,592
numeric bytes per stream; do not deploy or hide a tangent cache. Re-time all
arms with paired/interleaved order before making latency claims.

## Falsifier and interpretation

Freeze a screen criterion before fitting: all 18 fits complete; relative
sensitivity improves equal-record/equal-seed RMSE by at least 5% over the single
strongest other arm, every paired seed improves, neither amplitude worsens,
and no record mean worsens by more than 2%. It must also reduce sampled excess
sensitivity on independently seeded diagnostic directions. Retain penalty
magnitudes and hinge-active fractions so an inactive penalty remains visible.
No clean-forecast gain, or no advantage over either sensitivity control, rejects
this recipe at this budget; passing only motivates the matched-strength study.

All six arms have the **same nonlearned lag initializer**. Verify bitwise equal
conditioned states and replay each forecast from that common state. Improvement
must persist on clean contexts, with positive mean improvement on the late 64
steps against that same reference. Separately use one fresh unit direction per
request, seed 300000, and paired history perturbations `+/-0.01*v` in standardized
units. This is total history-vector norm 0.01, or per-coordinate magnitude
`0.01/sqrt(96)`, not a 1% perturbation of every observation. Future inputs and
targets remain unchanged. An effect confined to
perturbed histories supports initialization-error robustness, not better clean
dynamics. Exposed-DEV success would still need separately registered untouched
confirmation; it would prove neither stability nor physical identification.
