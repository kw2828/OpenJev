# Preserve recurrent forecasts while adapting decisions

Status: completed and independently audited. All 90 fresh trajectories and 15 final fits are retained; all original collection, training and audit supervisors closed successfully. The protected-readout rule fails **15/29**. The [results](otto-protected-readout-results.md) retain all controls, paired seeds and costs. The [collection plan](../output/otto-protected-readout-v1/collection-plan-01.json) and [training plan](../output/otto-protected-readout-v1/training-plan-01.json) bind the original data, seeds, protocol, runtime and source hashes. The prospective design below is preserved; this result authorizes no conditional confirmation.

The [previous fixed SPO+ recipe](otto-action-focused-results.md) lowered its TRAIN decision surrogate but worsened the later teacher-score gap in every paired comparison. Both score-MSE components rose. That observation motivates a causal comparison of where adaptation is allowed; it does not demonstrate damaged memory, conflicting gradients or insufficient recurrent capacity.

## One comparison

Use the ordinary shared-output GRU as the backbone. Train three fresh AUX models, then fork each final checkpoint into four adaptation branches:

| Recurrent predictor | MSE objective | MSE plus SPO+ objective |
| --- | --- | --- |
| Frozen | Frozen-MSE control | Protected-SPO candidate |
| Trainable | Joint-MSE control | Joint-SPO control |

All branches receive the same zero-initialized linear action readout from the existing hidden state. Its centered residual is added to the original nonquery action scores. It never feeds the recurrent correction or prior prediction. Genuine query actions and information access remain unchanged.

The [original engineering adapter](../src/openjev/research/otto_protected_readout.py) adds 116 parameters to the 5,996-parameter GRU. Its raw-score residual remains unchanged. The [training adapter](../src/openjev/research/otto_protected_training_model.py) supplies a factor of 64 before centering, matching the backbone's normalized readout units. Both adaptation modes use this same head and learning rate. Twenty fabricated output-unit and strict-copy checks passed. This parameterization was fixed before collecting new data; it was not selected from VALID outcomes.

Frozen branches lock every original parameter and detach their recurrent features. Joint branches permit gradients into the original predictor. Keep the bare pretrained backbone as a reference. Frozen-MSE controls the extra readout; joint-MSE controls extra training exposure; joint-SPO controls the gradient route for the decision objective. A GRU-only screen avoids conflating an adaptation mechanism with a new architecture.

## Prospective training conventions

Use 80 AUX pretraining epochs and 40 adaptation epochs: three shared pretrains and twelve adaptation fits. Match fresh data, chronological batches, sampled targets, importance weights, initialization at each fork, and stage-two optimizer settings across branches. Retain the established scale of 64 and unit SPO+ coefficient without searching the exposed evaluation outcomes. MSE applies to the deployed nonquery action scores; prior-query MSE applies to the unchanged base prior. The frozen prior term remains recorded but cannot train frozen parameters.

The final protocol must reserve fresh collection, fitting and selection seeds and specify exact cohort sizes, optimizer resets, schedules, row masks, detach conventions, loss arithmetic, budgets and source pins before collection. No old VALID trajectory may select a branch, coefficient or stopping time. Save final checkpoints only and close all branches before untouched evaluation is decoded. Previous failed and incomplete studies remain closed.

The [protocol](otto-protected-readout-protocol.md) specifies 54 fresh TRAIN paths, 36 fresh VALID paths, 15 final models including references and one 29-condition gate. Its initial implementation-stage status is superseded by the published, hash-bound plans. The training plan authenticates the collection's successful original supervisor closure before any training array is decoded. [Integrated qualification](../output/otto-protected-readout-v1/integrated-engineering-01/qualification-01/receipt.json) and [source reviews](../output/otto-protected-readout-v1/study-engineering-01/source-review-01.json) precede empirical work.

## Required evidence

The protocol requires Protected-SPO to lower mean later teacher-score gap by at least 10% in both settings against each of Frozen-MSE, Joint-MSE, Joint-SPO and the bare pretrained reference. It must also preserve full and initial action agreement against the strongest control mean and avoid paired-seed regression against Frozen-MSE. Support and technical-completion checks complete the 29-condition rule. These thresholds and scope denominators must be frozen before execution, not chosen after results.

Before real training, fabricated tests must establish identical initial predictions, legal query and padding behavior, chronological chunk equivalence, no residual feedback, and gradient separation. During training, bitwise parameter comparisons must show that frozen backbones remain unchanged; saved predictions must verify unchanged base forecasts on the same paths. Record all branch costs: matching data and optimizer updates does not equal matching computation. Use the qualified detached supervisor for fresh phases and preserve failures without retries.

Exact numerical comparisons must use matching parameter-gradient flags and execution contexts. A four-forward fabricated diagnostic found small float32 differences when freezing the original GRU itself; each adapter matched the original with equivalent flags bitwise. This is an engineering observation, not an empirical model result or an identified backend-kernel cause. Frozen branches must be compared before and after adaptation under the same frozen configuration. Frozen prior-only chunks may have no differentiable loss, so the trainer must not call backward unconditionally; the protocol must specify how zero-gradient chunks and optimizer updates are counted.

All **33 fabricated engineering tests** and Ruff passed, with independent source review. Checks include initial equality against the matched original, unchanged recurrent calls, gradient separation, hidden-state causality, mixed episode lengths, chunk behavior, ignored padding and capture cleanup. Two earlier failed qualification attempts are preserved alongside the numerical diagnosis; no empirical experiment was retried. [Qualification receipt](../output/otto-protected-readout-v1/engineering-01/qualification-03/receipt.json).

Frozen features, residual adaptation and [SPO+](https://arxiv.org/abs/1710.08005) are established techniques. This experiment can support an adaptation finding if it succeeds. It cannot by itself establish a novel architecture, biological learning, autonomous performance or a paper-level result. A retained gain still needs fresh confirmation, scenario shift, total-compute comparisons and a second environment before testing additional connectome or associative-memory structure.

The [September 23 novelty audit](protected-memory-literature-audit.md) adds direct prior work on frozen world models, fast plasticity and connectome controls, and identifies a subsequent temporal-credit question. It leaves this study and its acceptance rule unchanged.
