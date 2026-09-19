# Prediction-error memory updates: prototype

**Status:** the model, loss and synthetic integration checks are complete. The
[twelve-fit development pilot](../../research/reacher-innovation-pilot.md) has
now finished: 21/29 checks passed, and the normalized-error gate failed its
continuation rule. Controller integration and comparative speed remain
unmeasured. The separate two-observation study is unchanged.

The question is whether prediction error helps decide when to update slow
memory after sensing returns, beyond knowing the elapsed observation gap.
The [prior-art screen](../reacher-two-observation-control-v1/error-gated-mechanism-screen.md)
identifies close filtering, multiple-timescale and selective-memory precedents.
This combination alone is not an established architectural contribution.

## One model, four information pathways

The [model](../../src/openjev/research/reacher_innovation_context.py) has a fast
GRU state and a smaller slow context. Each issued action advances the fast
state, conditioned on context, and predicts the next public cosine/sine
features, four residual second moments and reward. When a real visible packet
arrives, a GRU corrects the fast state. A gated residual projection updates
slow context. Missing observations and action-only imagination hold context
fixed. Startup encodes its observation without inventing a preceding error.

| Variant | Inputs to the slow-update gate |
| --- | --- |
| Constant | Learned bias only |
| Age | Elapsed time since the preceding valid observation |
| Raw | Elapsed time and log(1 + squared prediction error) |
| Normalized | Elapsed time and log(1 + error squared divided by predicted residual second moments) |

All variants instantiate the same modules and parameter shapes, compute every
statistic, and differ only in the gate-input masks. Gate parameters remain
trainable. Integer public-history clocks supply elapsed time: six missing
packets mean seven action intervals between the surrounding valid packets;
ten mean eleven. A returning packet's own age is zero and cannot supply that
interval. Candidate states remain private.

## What the uncertainty objective means

The [loss](../../src/openjev/research/reacher_innovation_loss.py) keeps the
existing one-step and open-loop mean/reward MSE terms. It adds, at an explicitly
supplied weight, a visible-target residual moment score:

```text
0.5 * mean_over_visible_targets(sum_j(log(v_j) + stopgrad((y_j - mu_j)^2) / v_j))
```

Both predicted mean and residual moments come from the preceding action
transition, before target assimilation. The final packet can supervise that
prediction but causes no extra action. The variance head sees detached fast
features; raw and normalized scalar gate statistics are detached too. The
residual vector used to change context remains differentiable. This separates
auxiliary variance-head learning from direct changes to the mean/reward
backbone, and prevents a direct gradient incentive to inflate variance to
manipulate the gate.

With a biased mean, this estimates conditional residual second moments,
including prediction bias. Bounded outputs constrain the optimum. The loss
reports near-bound fractions and predicted scale. It does not establish joint
Gaussian density, coordinate independence, calibrated confidence or a
chi-square threshold on the constrained cosine/sine representation. Default
widths, step size and variance bounds are provisional engineering choices.

## Validation and next comparison

The 37 [model tests](../../tests/test_reacher_innovation_context.py) check
causality, gap clocks, initial parameter identity, gradient paths, slow-state
holding, branch isolation and completed leaf-module work. The 20
[loss tests](../../tests/test_reacher_innovation_loss.py) include analytical
score/gradient checks, comparison with the existing objective across all four
gates, missing targets, terminal handling and three tiny optimizer steps per
variant. An independent review checked both components and the integration.
[Validation receipt](validation.json).

These checks use synthetic data and small states. The model is not registered
in an existing trainer or controller, and no checkpoint is being substituted
into a frozen experiment. Leaf-module counters omit validation, copies,
analytic reward arithmetic and backward/optimizer work; they are not total
compute or wall-time measurements.

The next experiment must hold public information, initialization pairing,
training data, objective, search and evaluation cases fixed across gate arms.
A cheap strong control is an age-conditioned empirical error-scale lookup,
estimated from training-development residuals only, used in the same normalized
gate while retaining and paying the unused learned variance head. It tests
whether learned uncertainty improves on elapsed-time scaling. Conventional
learned filters and single-/multiple-rate recurrence remain necessary before
a broader architecture claim. The existing iid actuator disturbance supports
state-estimation research; it does not establish adaptation to persistent
hidden dynamics or a connectome advantage.
