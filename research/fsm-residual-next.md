# Proposed next checks for the FSM residual model

**PARTIALLY EXECUTED.** The longer-VARX and folded-affine checks below were
subsequently [registered](fsm-linear-controls-protocol.md) and
[passed their development rule](fsm-linear-controls-results.md), with independent
audit agreement. The subsequent [FIT-only author BLA28 adaptation](fsm-author-bla-results.md)
also completed, with 4/4 continuation checks and independent audit agreement.
The nonlinear NL-LFR adaptation now has a [frozen protocol](fsm-author-nllfr-protocol.md)
and [registration](fsm-author-nllfr-registration.json), after 162 fabricated tests
and a full-size synthetic runtime qualification. It has no audited performance
result yet. The 300 mV shift remains unregistered and unexecuted.
No reserved measurements have been read under
this proposal. The remaining steps below are prospective; exposed-DEV selection
remains development.
See the [frozen protocol](fsm-residual-protocol.md) and
[prior-art limits](fsm-residual-prior-art.md).

## 1. Strengthen the references before opening the reserve

- **Fold affine feedback into native VARX coefficients.** Use each already
  selected seed's unchanged weights. Qualify forecast equivalence before timing
  complete C100/H128 requests, including normalization, state setup and output
  conversion. Report all seeds and actual retained storage. The unfused PyTorch
  control is not its minimum deployable cost; this requires no new fitting.
- **Test longer linear memory on the existing FIT/DEV split.** A proposed small
  fixed grid is orders 32, 64 and 96 with penalties 1e-6, 1e-3 and 0.1, using
  the same FIT normalizer, within-period fitting and current-input alignment.
  All orders fit inside the supplied C100 context. Retain failures without pole
  repair, and select one recipe on exposed DEV before any reserve is opened.
  This asks whether the nonlinear gain survives stronger linear memory.
- **Qualify the authors' NL-LFR procedure using FIT data only.**
  BLA28 is now fitted and evaluated under the causal contract; the nonlinear
  reference is now registered, with performance and independent audit pending.
  Do not use supplied weights fitted with all three amplitudes.
  Freeze any adaptation, initialization and optimization budget in advance;
  record unsupported dependencies or incomplete reproduction honestly.
  The [authors' pinned repository](https://github.com/merijnfloren/fsm-benchmark-data/tree/539a12fef384b086a8562b500498b2fa3899ef70)
  defines the original methods and CC BY 4.0 data attribution.

Keep the selected residual checkpoints unchanged during these checks. If a
stronger conventional reference removes the advantage, report that outcome
before deciding whether any reserved-data comparison is worth conducting.

## 2. Register one untouched 300 mV scenario-shift evaluation

Only after the reference comparison is closed, freeze the four DEV-selected
residual architecture/rate groups with all three seeds (12 checkpoints), plus
every control, normalization, metric and stop rule.
No new training, calibration, seed selection or fallback is allowed after access.
Admit only the explicitly registered `u_300mV_train` / `y_300mV_train` members;
their archive names do not authorize using them for training in this experiment.
Keep every official `_test` member closed. Qualify the new access guard on
fabricated inputs before opening any 300 mV measurement.

Propose the complete six-realization, two-period 300 mV roster, preserving each
period as a separate record, C100/H128 and the existing fixed start schedule.
Freeze that metadata contract first and stop on a mismatch. Supply only observed
context and applied inputs through the predicted step; no future output labels.
Report all records, channels, seeds and failures, with the original FIT scales
and separate native-unit errors. Distinguish realization groups previously used
for lower-amplitude FIT and DEV; repeated periods and matched excitations are
correlated, not independent evidence of six new environments.

A proposed quality rule is at least 5% lower equal-record mean error than the
single strongest control fixed on exposed DEV, with no record more than 2%
worse and all declared forecasts finite. Report every other frozen control too.
Register the exact cost requirement separately using the newly qualified native
linear reference. Failure closes this proposed claim; no reserve-driven retuning
or substitute checkpoint. Passing would support amplitude-shift robustness on
this system, not general environmental transfer or closed-loop control.

## 3. Separate method transfer from novelty research

A second measured system needs a new primary-source data contract, causal input
interface, independent records, strong conventional controls and a frozen budget.
Retraining the method there is method transfer, not transfer of these weights.
Do not call additional exposed FSM analysis a second environment.

Novelty is a separate question. The current linear-plus-residual construction is
established. Only after useful prediction survives these controls should a new
mechanism, such as sensitivity-aware residual regularization, receive its own
matched ablation against ordinary regularization and existing hybrid methods.
Neither biological inspiration nor a development pass supplies that evidence.

The [four-paper sensitivity review](fsm-sensitivity-prior-art.md) identifies
the closest established mechanisms. A separate [six-arm experiment draft](fsm-sensitivity-experiment-draft.md)
tests a penalty on sensitivity added beyond the frozen linear model against
ordinary shrinkage and matched sensitivity controls. It is unregistered and
unrun; deciding whether to pursue it follows closure and independent audit of
the current nonlinear reference. It does not alter the frozen comparisons above.
