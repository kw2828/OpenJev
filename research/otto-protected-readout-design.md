# Preserve recurrent forecasts while adapting decisions

Status: prospective engineering design, not a frozen scientific protocol or a result. No new empirical data or training has been admitted for this comparison.

The [previous fixed SPO+ recipe](otto-action-focused-results.md) lowered its TRAIN decision surrogate but worsened the later teacher-score gap in every paired comparison. Both score-MSE components rose. That observation motivates a causal comparison of where adaptation is allowed; it does not demonstrate damaged memory, conflicting gradients or insufficient recurrent capacity.

## One comparison

Use the ordinary shared-output GRU as the backbone. Train three fresh AUX models, then fork each final checkpoint into four adaptation branches:

| Recurrent predictor | MSE objective | MSE plus SPO+ objective |
| --- | --- | --- |
| Frozen | Frozen-MSE control | Protected-SPO candidate |
| Trainable | Joint-MSE control | Joint-SPO control |

All branches receive the same zero-initialized linear action readout from the existing hidden state. Its centered residual is added to the original nonquery action scores. It never feeds the recurrent correction or prior prediction. Genuine query actions and information access remain unchanged.

Frozen branches lock every original parameter and detach their recurrent features. Joint branches permit gradients into the original predictor. Keep the bare pretrained backbone as a reference. Frozen-MSE controls the extra readout; joint-MSE controls extra training exposure; joint-SPO controls the gradient route for the decision objective. A GRU-only screen avoids conflating an adaptation mechanism with a new architecture.

## Prospective training conventions

Use 80 AUX pretraining epochs and 40 adaptation epochs: three shared pretrains and twelve adaptation fits. Match fresh data, chronological batches, sampled targets, importance weights, initialization at each fork, and stage-two optimizer settings across branches. Retain the established scale of 64 and unit SPO+ coefficient without searching the exposed evaluation outcomes. MSE applies to the deployed nonquery action scores; prior-query MSE applies to the unchanged base prior. The frozen prior term remains recorded but cannot train frozen parameters.

The final protocol must reserve fresh collection, fitting and selection seeds and specify exact cohort sizes, optimizer resets, schedules, row masks, detach conventions, loss arithmetic, budgets and source pins before collection. No old VALID trajectory may select a branch, coefficient or stopping time. Save final checkpoints only and close all branches before untouched evaluation is decoded. Previous failed and incomplete studies remain closed.

## Required evidence

The intended continuation rule requires Protected-SPO to lower mean later teacher-score gap in both settings against Frozen-MSE, Joint-MSE and Joint-SPO, preserve full and initial action agreement, and avoid paired-seed regression against Frozen-MSE. Also report every branch against its bare pretrained reference. Exact effect thresholds and scope denominators must be frozen before execution, not chosen after results.

Before real training, fabricated tests must establish identical initial predictions, legal query and padding behavior, chronological chunk equivalence, no residual feedback, and gradient separation. During training, bitwise parameter comparisons must show that frozen backbones remain unchanged; saved predictions must verify unchanged base forecasts on the same paths. Record all branch costs: matching data and optimizer updates does not equal matching computation. Use the qualified detached supervisor for fresh phases and preserve failures without retries.

Frozen features, residual adaptation and [SPO+](https://arxiv.org/abs/1710.08005) are established techniques. This experiment can support an adaptation finding if it succeeds. It cannot by itself establish a novel architecture, biological learning, autonomous performance or a paper-level result. A retained gain still needs fresh confirmation, scenario shift, total-compute comparisons and a second environment before testing additional connectome or associative-memory structure.
