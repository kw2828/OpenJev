# Next hypothesis: learn the effect of an action

Proposal following the closed belief-distillation pilot. This is a research
direction, not an execution registration or evidence of a new architecture.
The completed study remains DEV_FAIL 7/18. No failed cell is removed or retuned.

## What the new evidence changes

The [audited comparison](otto-belief-distillation-results.md) gives a strong
within-study supervision result: recurrent probability-target training lowers
mean long-gap sampled log loss 31.94% / 33.74% versus its identical sampled-target
twin, using ratios of three-fit means. Mean decision gaps fall 5.86% / 14.91%, but several paired seeds miss the
registered threshold. Those averages do not overturn the failed gate.

There is a more specific failure to investigate. Under the sensing shift,
long-horizon action-effect error E is 0.019487 for the soft recurrent model,
versus 0.011549 for the direct predictor and 0.017444 for predicting no action
effect at all. Thus its good marginal forecasts do not establish accurate
counterfactual dynamics. In the base setting E=0.024194 versus signal S=0.032133;
the recurrent model captures some effect there. These are supplementary
observations on exposed DEV, not fresh confirmation or significance tests.

The expanded dataset and common terminal shortcut also differ from the previous
pilot. Cross-study score changes cannot isolate memory, sample size or training
signal. The clear attribution is the matched soft-target versus sampled-target
comparison within the new study.

## Immediate mechanism to test

Use paired proposed-action blocks from the same observed prefix. Fit both
ordinary outcome probabilities and their signed change when actions change.
The current collector already computes factual and opposite-block probability
targets; the current learner only trains on the factual block. A successor can
ask whether explicitly teaching action effects improves shifted dynamics.

The essential controls would be:

1. Same recurrent model trained on factual and opposite blocks with ordinary
   soft cross entropy. This controls for the extra branches and target data.
2. Identical data, architecture and updates, with a fixed signed-effect loss
   in addition to that proper-score objective. Any weight must be fixed using
   TRAIN-only reasoning or a separate calibration split before fresh DEV.
3. The direct predictor trained on the same paired blocks, plus an action-blind
   reference reporting its exact zero-effect prediction.

Hold the prefix representation, branch queries, fit seeds, optimizer updates,
parameter budget and inference accounting fixed. Retain the current cost and
auxiliary objectives equally. Do not attribute an improvement from merely
adding paired data to the effect loss. Compare long-horizon effect error,
forecast loss and teacher-action gap on new originating cases, including a
predeclared shift. Require useful decisions as well as an effect improvement.
No action-effect ratio is defined for zero-signal cases; keep those cases in
absolute-error reporting.

Before any run: specify exact data reuse, new seeds, supervision weight, limits,
source closure and continuation rule, then qualify on fabricated cases. Existing
DEV is exposed and cannot become a fresh test. The existing TRAIN split may be
reused only with explicit provenance and equal access for all arms; account for
its original teacher cost. Autonomous control requires a later closed-loop
experiment with complete costs, not a reinterpretation of fixed-path imitation.

## Literature that gives this a stronger foundation

[Value-Directed Compression of POMDPs, Poupart and Boutilier (2002)](https://papers.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html)
asks which compressed belief information preserves decisions. It motivates
teaching a compact state to retain action-value distinctions. Its linear
compression conditions do not automatically apply to our nonlinear teacher.

[Predictive State Inference Machines, Sun et al. (2016)](https://proceedings.mlr.press/v48/sun16.html)
learn filtering directly through predictive states. This motivates representing
memory by explicit future questions and training on states reached by the
student's own updates. Action-conditioned question sets would be an adaptation
we need to test, with matched teacher-state versus student-state training.

[The Value Equivalence Principle, Grimm et al. (2020)](https://arxiv.org/abs/2011.03506)
defines model equivalence relative to policies and value-function updates.
It supports the broader aim of preserving information useful for decisions
instead of reconstructing every state detail. Our teacher-imitation costs are
not Bellman values, so the current model has no proven value-equivalence claim.

[Deep Variational RL for POMDPs, Igl et al. (2018)](https://proceedings.mlr.press/v80/igl18a.html)
learns latent environment inference jointly with a policy. It provides a later
particle-state baseline if we can show that the compact deterministic state
loses separate hypotheses. Compare weighted versus equal-weight particles and a
matched-compute deterministic state; charge all propagation and update work.
The present action-effect diagnostic alone does not prove hypothesis collapse.

These ingredients are established work. A possible contribution would be a
specific compact update mechanism that preserves useful action effects through
missing observations, with convincing shifted and closed-loop comparisons.
Calling a GRU biological, adding connectome wiring or changing the paper title
would not supply that evidence.
