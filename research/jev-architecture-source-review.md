# Decision-only models: what the public sources establish

Source review, September 23, 2026. This checks the supplied social-media post
against primary sources. No upstream model was run or timing reproduced. This
note does not change a registered experiment or its rule.

## The architecture recipe is a hypothesis

[TypeSafe's announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
describes parallel numerical outputs and RLCD training. It does not publish the
complete architecture or training algorithm.

[Archer Hume's article](https://archerhume.com/posts/jevs-architecture-unmasked/)
explicitly distinguishes observations from inference. The 255-option limit is
enforced by the API; it does not establish a 255-slot classifier. A slot head,
pointer-style readout and reserved vocabulary rows remain relevant alternatives.
The article also finds interactions between options, which an independent
candidate scorer with fixed-temperature softmax cannot explain by normalization
alone. Its sparse-backbone proposal remains speculative.

Removing text generation is useful engineering. It does not establish that a
particular language head was removed, that the probabilities are calibrated, or
that the proprietary system has been reproduced.

## Public implementations are different baselines

**SemIf**, reviewed at
[`1f2dea3`](https://github.com/TheoLeeCJ/SemIf/tree/1f2dea3e25379f9dfc98cb83c324f00ab5deda37):
its [direct scorer](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/src/semif_phase1/direct.py)
retains the causal-LM vocabulary head and selects label-token logits for 2-16
options. It does not replace that head with 255 output slots. Its
[shared path](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/src/semif_phase1/shared.py)
reuses state computation; CUDA batches suffix branches, while PyTorch/MPS
evaluates copied-cache branches serially. The
[calibration benchmark](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/benchmarks/calibrate.py)
fits a positive temperature using labeled log loss and group-disjoint
out-of-fold evaluation. This is offline calibration of a frozen model, not RL
or a blanket calibration guarantee for its API.

**Jevlike**, reviewed at
[`94f5fd1`](https://github.com/vinnylarouge/jevlike/tree/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452):
its [text model](https://github.com/vinnylarouge/jevlike/blob/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/jevlike/model.py)
uses option queries attending to context, without an LM vocabulary head or
option-to-option attention. Menus are variable, without a 255-option cap.
[Text training](https://github.com/vinnylarouge/jevlike/blob/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/jevlike/train.py)
supports CPU and supervised cross-entropy;
[evaluation](https://github.com/vinnylarouge/jevlike/blob/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/jevlike/eval.py)
measures calibration error without fitting a calibrator. Its separate game
policies use learned button embeddings. This is an independent implementation,
not recovered TypeSafe code.

## Calibration does not require RL

[Log loss and Brier loss](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
are proper scoring rules that support supervised probability learning.
[Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) is an
established post-training calibration baseline. Neither guarantees calibration
under arbitrary distribution shift. A softmax output of 0.9 alone does not mean
90% observed correctness, and RL optimized for task reward need not produce
honest probabilities.

[Conformal prediction](https://arxiv.org/html/2107.07511v6) instead constructs
prediction sets with marginal coverage under its assumptions. That does not
make each candidate probability calibrated or guarantee each individual
decision. Correlated control trajectories and changing policies require an
appropriate sequential treatment; ordinary exchangeable-sample guarantees
cannot simply be asserted for them.

The API's [confidence statistic](https://docs.typesafe.ai/confidence) describes
distribution concentration. It is distinct from candidate probabilities and
does not replace an empirical calibration measurement.

## What to borrow for OpenJev

Our [local scorer](../src/openjev/decisions.py) already reads unique candidate
label probabilities without generating an explanation. It retains the
vocabulary projection and explicitly labels its probabilities uncalibrated.
The separate [shared-prefix experiment](shared-prefix-results.md) measures
cache reuse; it supplies no calibration or learning claim.

The useful additional baseline is a variable-candidate attention readout. A
future comparison could keep the readout architecture fixed while varying current-state
input versus recurrent memory, with a separate option-interaction control.
Evaluate unfamiliar candidate sets and paraphrases, not just a fixed label
catalog. This is a proposed comparison, not an admitted follow-up or new result.

For probability claims, reserve separate calibration and test data and compare
raw probabilities, temperature scaling and supervised proper-score training
before adding RL. Report accuracy, log loss, Brier score, reliability plots and
decision utility with total compute. Teacher agreement and softmax-transformed
costs are not substitutes for observed correctness or outcome calibration.

The [protected-readout comparison](otto-protected-readout-results.md) and
[action-latent pilot](otto-action-latent-results.md) both failed their continuation
rules. The new [belief-distillation study](otto-belief-distillation-protocol.md)
tests full conditional outcome probabilities as supervised TRAIN targets for a
compact recurrent predictor. Its teacher uses a checked Bayesian filter under a
known sensor law; the student cannot inspect the full belief during inference.
A same-seed sampled-label twin, action-blind model and direct predictor separate
supervision and recurrence effects. This is not a reproduction of TypeSafe RLCD,
a calibration guarantee or evidence for connectome wiring.
