# Proposed gain-initialization × clipping-norm experiment

**Unregistered and unrun.** This is an optional numerical-attribution study, not permission to restart a failed fit, revise a closed result, or access new evaluation data. No model, array, backward pass or test was run to prepare this note. Only frozen sources, the seven saved probe JSON records and the closed audit's scalar summary were inspected.

**Deprioritized attribution study, not the next automatic 24-fit run.** The position-observer study and [independent audit](../output/robot-position-observer-audit-v1/audit.json) are closed: evidence audit PASS with agreement; scientific outcome `POSITION_OBSERVER_DEVELOPMENT_FAIL`, three of five criteria passed. All six fits completed 4,096 updates each, but the strongest-control improvement and per-file harm requirements failed. Numerical attribution may still be informative, but it requires a separate decision to spend that budget. Improved training completion has not established useful quality gains, and this proposal does not reopen the failed quality gate. No new measurements are authorized.

## Question and four cells

On exactly the same frozen transition and paired batches, does `[I;0]` help because it avoids an adverse initial correction, because native clipping overflows, or both?

| Cell | Initial 12×6 gain | Clipping norm |
|---|---|---|
| IN | `[I;I]` | Current native float32 operation |
| IS | `[I;I]` | Scaled float64 reduction below |
| PN | `[I;0]` | Current native float32 operation |
| PS | `[I;0]` | Scaled float64 reduction below |

All four learn the same unrestricted 72 gain entries with the same frozen 590-parameter cell. The lower six latent coordinates are not verified physical velocities. Keep the exact thirty prefix corrections, C32, H128, future measured-torque inputs, normalization, loss and causal observation boundary from the [position protocol](robot-position-observer-protocol.md).

The saved probes already distinguish finite gradient entries from an infinite native aggregate. They also show large pre-clipping forward-loss and prefix-state differences between the initial gains. They do not show whether safer clipping lets the old gain learn a competitive predictor; that is the missing counterfactual. Three first batches and one archived failure checkpoint are not an estimate of failure frequency.

## Exact proposed safe operation

After the unchanged float32 backward pass, require that the sole present trainable gradient is CPU float32 `gain.grad[12,6]`, with no frozen-cell gradient. Preserve its raw values before either clipping path. Missing or malformed gradients are fatal implementation errors. Nonfinite entries stop the fit as a retained numerical failure; do not replace, truncate or repair them.

For the safe cell, under `no_grad`, flatten the existing 72 entries and convert a detached copy to float64. Let `m=max(abs(g64))`. Set `n64=0` if `m=0`; otherwise compute `n64=m*sqrt(sum((g64/m)^2))`, with the sum and square root in float64. Require a finite result. Compute `c64=min(1,1/(n64+1e-6))`, cast this scalar once to float32, and multiply the original float32 gradient in place by that scalar. Adam consumes that gradient with its original float32 parameters and moment arrays. No float64 forward pass, backward pass or optimizer is introduced.

The installed [PyTorch clipping source](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/torch/nn/utils/clip_grad.py) uses the same threshold and `1e-6` denominator term. Native cells must call the original `clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)` and keep the original post-call finite-norm failure guard. Pin its runtime/source before execution. The safe intervention changes reduction and coefficient arithmetic, so ordinary finite steps need not be bitwise identical. It is not solely an exception handler at the first overflow.

Use identical detached diagnostics in all cells: prefix maxima, pre-clipping entry count/finiteness, float64 norm, the norm actually used, clipping coefficient and update outcome. Compute the alternate norm only on a detached copy; never invoke a second in-place clip. Charge diagnostic and safe-reduction workspace and elapsed time. Save every original attempt and the final raw pre-clipping arrays, as in the [current trainer](../scripts/robot_position_observer_study.py).

Before registration, fabricated qualification must cover zero gradients, a 3-4-5 norm, finite gradients whose float32 squared norm overflows, nonfinite-entry rejection, unchanged gradient direction and frozen weights, coefficient rounding and a single Adam update. Qualification must not choose thresholds or numerics using measured outcomes.

## Matched execution and evidence

Execute all four cells **fresh from initialization**, rather than splice in current or previous native fits. This gives identical instrumentation, process rules and complete fit timing. Historical native runs remain linked evidence, not cells in the new factorial table. Fresh execution does not make their already exposed data independent.

Proposed fixed budget: three inherited backbone seeds × both rates `.001/.003` × four cells = **24 scheduled fits**, each 4,096 original paired batches of 16, Adam `.9/.999`, epsilon `1e-8`, clipping threshold 1.0. That is at most 98,304 completed optimizer updates. Reuse the exact authenticated batch arrays and backbone checkpoints; restart each fit with empty Adam state. No continuation from archived failed weights, early stopping, adaptive rates or extra seeds. Rotate the four-cell order deterministically within seed/rate blocks before registration, with one CPU thread and sequential fits.

Retain the original 1,800-second per-fit cap; propose a 14,400-second whole-campaign cap plus 60-second external allowance. A global budget failure preserves the partial roster and prevents a complete factorial conclusion; it does not authorize a larger cap or restart. Close all 24 attempts and their checkpoints before evaluation access. Retain early numerical failures and timeouts without substituting later checkpoints or imputing quality scores.

Reuse the seven processed FIT and four already exposed evaluation recordings only. Keep all 22 windows, six joints and H64/H128 scores. Freeze the established control recipes, including both fixed gains, last-two, local/temporal heads, historical recurrent controls and both causal ridges. Retain all control scores under their original audit; retime complete requests on the current host if making cost comparisons. Never recalculate normalization, refit controls or decode raw MAT/official TEST data.

## Endpoints and interpretation

Primary attribution endpoint: the complete 3-seed × 2-rate table of completed updates, first failure stage and numerical classification for every cell. Report completion counts and all paired contrasts. Distinguish nonfinite forward/loss, nonfinite individual gradient entries, finite entries with nonfinite native aggregate, nonfinite Adam/parameters, and deadline failures.

Quality endpoints: all per-seed/file H128 RMSEs, four-file equal means, H64/joint errors, complete fit time, complete-request latency and persistent numeric bytes. Report within-rate initialization contrasts and within-initialization norm contrasts. A difference-in-differences is descriptive and defined only for complete matched quartets; report its denominator and all excluded failures. Do not use complete cases to erase a cell's failed fits or claim population-level significance from three previously used backbones.

For a separate recipe report, the original two DEV files alone may select one pooled rate per cell, lower rate on exact ties, retaining every option. This does not replace the full paired-rate attribution table. Neither the former-confirmation files nor the best four-way mean may select a continuation candidate.

- IS completing pairs where IN stops at a native aggregate supports an arithmetic limitation under the specified initialization. It does not establish useful predictions.
- PS outperforming IS under the same safe arithmetic supports an initialization effect on this exposed workload. Similar quality would weaken a claim that position-only initialization is necessary for quality.
- Interaction or persistent safe-cell failure means neither factor alone explains the outcome. Even a completed trajectory is not an observer-stability or calibration theorem.

No numerical-attribution finding is an advance criterion. If a continuation decision is desired, designate PS before registration and require the same five development criteria against the frozen complete control set, including fixed `[I;0]`: complete forecasts/costs, 5% gain versus the strongest control, per-file 2% harm bound, ≤1.5× last-two full-request latency, and error/latency/storage nondominance. Also disclose any complete factorial cell that dominates PS; do not switch candidates after results. A failure closes this continuation without fallback. A pass permits only a separately registered untouched-environment method-transfer test, as outlined in the [transfer plan](robot-observer-transfer-plan.md), not architectural novelty or robot-control claims.
