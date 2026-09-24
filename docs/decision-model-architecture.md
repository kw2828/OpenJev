# What the public decision-model implementations establish

A decision API can encode context once and score supplied options without
generating an explanation. This is useful for OpenJev, but speed, decision
quality and probability calibration are separate measurements.

The widely shared three-step diagram is a possible implementation, not a
verified specification of TypeSafe Jev. Archer Hume's
[investigation](https://archerhume.com/posts/jevs-architecture-unmasked/)
explicitly leaves both fixed-slot and pointer-style readouts possible. An API
limit of 255 options does not identify the neural head. TypeSafe describes
[RLCD's objective](https://docs.typesafe.ai/introduction/machine-learning-primer)
but does not provide a reproducible training algorithm on that page.

[Jevlike](https://github.com/vinnylarouge/jevlike) implements option queries that
attend to context, followed by shared scoring and softmax. It identifies itself
as an independent starter, not a reproduction. Its reported 100x comparison
forces a small decoder to generate 400 tokens; that is not an equal-quality
comparison against a direct one-token classifier. The separate
[zhihz/OpenJev project](https://github.com/zhihz/openjev) also describes itself
as an independent research preview. It is not this repository.

The [Qwen-2.5-1B-RLCD model card](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
describes a Qwen2.5-1.5B inference engine with shared KV cache, candidate-logit
slicing and token-tree continuation. Its name alone does not establish that
RL training occurred. The card's normalized softmax formula also does not
establish empirical calibration; that requires predictions and held-out labels.

Softmax normalizes scores; it does not establish calibration. RL is not a
prerequisite for calibration either: supervised proper losses and held-out
post-processing are relevant baselines. Guo et al.'s
[calibration study](https://arxiv.org/abs/1706.04599) found temperature scaling
effective on many of its tested datasets. That result does not guarantee
performance on a new workflow or under distribution shift.

For this project, a useful comparison would retain identical candidate IDs,
test shuffled option order and changed candidate sets, measure log loss and
Brier score alongside correctness, and report latency at matched task quality.
A recurrent model should additionally justify its memory and transition
predictions against simpler controls. Neither replacing the output head nor
adding a calibration loss establishes architectural novelty.

The current [action-error study protocol](../research/finite-action-range-study-protocol.md)
tests a narrower loss hypothesis in a synthetic recurrent world. It does not
implement RLCD or test text-probability calibration. Treat its result separately
from the local Qwen decision API and the browser demo.
