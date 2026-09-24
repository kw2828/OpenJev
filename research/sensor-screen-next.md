# Require evidence that history matters

Prospective constraint after the rejected TRAIN-only sensor screen, 24 September
2026. No new experiment is registered by this note.

The [sensor screen](sensor-screen-results.md) rejects benzene calibration as the
next learned-memory benchmark: both static headroom conditions fail, and no
adaptive model reaches the repeated 10% improvement requirement. June gains are
real within TRAIN, but do not override the rule. Later labels stay unopened.
No learned sketch, RL selector or recurrent model was trained in this screen.

The architecture goal remains active. The next task should make hidden dynamics
and decision consequences observable in a fair experiment. A candidate direction
is control with unknown actuator lag or hysteresis, where a current observation
alone may not identify the state that determines the next action. This is a
research direction to source and qualify, not a new result or a selected dataset.
Do not retrofit this task onto the evaluated benzene labels.

Before training a new core:

1. Select a distinct dataset or environment with an explicit public observation,
   action and feedback interface. Establish from task semantics why earlier
   events can matter; do not hide a useful current sensor merely to create a win.
2. On an allowed development split, compare current-observation prediction,
   short fixed history, an appropriate conventional state estimator, and a small
   GRU at comparable inputs and resources. Freeze the screen before seeing its
   outcomes. Existing OpenJev robot studies already show that weak memory-reset
   controls can mislead; use the stronger two-observation baseline too where it
   is appropriate.
3. Only advance a learned retention mechanism if the task has reproducible
   decision headroom beyond those controls. Count model weights, normalization,
   pending feedback inputs, state, computation and scratch memory separately.
   Delay cannot be implemented through an uncharged historical-input lookup.
4. Freeze fresh evaluation trajectories and a scenario shift before selecting
   an architecture. Untouched evaluation data, a second environment and an
   actual matched-resource advantage remain prerequisites for stronger claims.

A possible retention mechanism appends the newly available labeled feature row
to a low-rank sketch, then learns which singular direction to discard using
future decision costs. That is not inherently new:
[Learning-Augmented Frequent Directions, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6de668dab370194fa304a08be5aacd85-Paper-Conference.pdf)
already predicts important directions in streaming matrices;
[Frequent Directions](https://arxiv.org/abs/1501.01711) is the analytic control;
[ALPaCA](https://arxiv.org/abs/1807.08912) learns features and Bayesian priors for
online regression. Any contribution must specify what the mechanism improves
and demonstrate it beyond these conventional and learned controls. A learned
compression policy is not automatically an exact or calibrated posterior.

Keep the [successful supplied-law baseline](measurement-results.md) and the
[failed RL retention comparison](retention-results.md) as separate evidence.
This screen neither invalidates the former nor rehabilitates the latter.
