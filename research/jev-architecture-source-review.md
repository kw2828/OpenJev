# Decision-only models: what the public sources establish

Source review, September 24, 2026. This checks the supplied social-media posts
against primary sources. No upstream model was run or timing reproduced. This
note does not change a registered experiment or its rule.

## The architecture recipe is a hypothesis

[TypeSafe's announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
describes parallel numerical outputs and RLCD training. It does not publish the
complete architecture or training algorithm.

The public [Choice limit](https://docs.typesafe.ai/primitives/choice) is 255
options per question. [Score](https://docs.typesafe.ai/primitives/score) returns
probabilities over 2-10 ordered levels and their probability-weighted mean.
Neither interface establishes the screenshot's proposed fixed classifier or
sigmoid score head.

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

**SemIf**, formerly OpenJev, reviewed at
[`23cf1f3`](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/23cf1f39fc9534fe81437200959b6dfc7106e45a):
its [direct scorer](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/src/semif_phase1/direct.py)
retains the causal-LM vocabulary head and selects label-token logits for 2-16
options. It does not replace that head with 255 output slots. The project's
[reported timing](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/README.md#speed)
is 1.023 seconds for direct readout versus 5.332 seconds for compact generated
JSON, a 5.21x ratio, using the same frozen Qwen3.5-4B, state and 21 binary
questions on an RTX 3090. Choices agree on 18/21 questions. These are the
publisher's systems measurements, not a reproduced or quality-matched Jev
comparison.

Its [calibration benchmark](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/docs/CALIBRATION.md)
fits a positive temperature using labeled log loss and group-disjoint five-fold
evaluation. It reports WANLI expected calibration error falling from 0.208 to
0.069 with unchanged accuracy; improvements on two other workloads have
overlapping uncertainty intervals. This is offline calibration of a frozen
model, not RL or a guarantee for arbitrary API requests.

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

Its [reported roughly 100x timing comparison](https://github.com/vinnylarouge/jevlike/blob/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452/README.md)
uses eight options against a small decoder forced to emit 400 tokens. That
denominator does not establish a 100x advantage over a minimal, equally accurate
decision baseline or the proprietary Jev service.

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

Our **kw2828/OpenJev** is a separate project from TypeSafe's Jev, SemIf and
jevlike. Our [local scorer](../src/openjev/decisions.py) already reads unique candidate
label probabilities without generating an explanation. It retains the
vocabulary projection and explicitly labels its probabilities uncalibrated.
The separate [shared-prefix experiment](shared-prefix-results.md) measures
cache reuse; it supplies no calibration or learning claim.

A bounded text baseline would freeze one Qwen checkpoint and its candidate
prompts, then compare raw option probabilities with one positive temperature
fitted on a separate, group-disjoint calibration split. Predeclare the workload,
candidate permutations, untouched test groups, decision costs and latency
measurement. Report accuracy, log loss, Brier score, reliability plots and
decision utility, charging tokenization, state encoding and scoring. Temperature
scaling preserves argmax; any utility gain must come from a declared
probability-dependent decision rule. This is a proposal, not an admitted study.
Teacher agreement and softmax-transformed costs are not outcome calibration.

Our recurrent forecasting and control experiments are separate from TypeSafe
RLCD and from the text scorer. They test learning mechanisms under explicit
protocols; they do not establish calibrated text probabilities or biological
wiring benefits. Current results and continuation rules are in the
[experiment archive](experiment-index.md). The [prefix-likelihood protocol](finite-prefix-learning-protocol.md)
tests one supervised prediction loss on an unchanged recurrent filter, not a
recovered proprietary training algorithm.
