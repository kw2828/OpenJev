# Next: qualify expected-count learning of hidden dynamics

**Proposal only. No new scientific run is registered or executed here.**
The [completed readout diagnostic](finite-gap-readout-study-results.md) leaves
the short-horizon and blind-forecast failures unresolved after all nine linear
heads satisfy the numerical criterion. Another tolerance change or a selected
successful seed would not answer the remaining learning question.

Test whether an explicit latent-state inference step gives the same recurrent
model a more useful starting point than gradient-based likelihood training.
Use forward-backward expected counts to fit the existing action-conditioned
transition, observation and hazard model, followed by the same joint H1/H2
training recipe. This changes the learning procedure, not the state size or
information supplied to the policy.

This is an application of established
[expectation-maximization](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x).
EM can also settle at poor local solutions. It supplies neither a novel
architecture nor a biological-learning claim. The question is whether this
ordinary baseline resolves a failure before adding a more elaborate mechanism.

## First qualify the arithmetic

On small fabricated sequences, enumerate hidden paths independently and check
forward likelihoods, smoothed state/transition counts, terminal-event counts,
and a complete update. Cover reset observations, variable lengths, first-found
absorption, padding, action-conditioned transitions, and the shared observation
law. Define zero-count and boundary handling before running. Never clip an
invalid likelihood silently or claim a monotonicity guarantee for an update
that does not maximize the declared expected log likelihood.

Smoothing may use later observations inside TRAIN to estimate parameters.
At evaluation, the carried state must still use only the public prefix already
seen. No true hidden state, oracle posterior, world parameters, DEV observation
or forecast outcome enters training inference. Check causal predictions against
an independent forward filter.

## A comparison that separates the explanations

Use the same eight-state factorized model and paired initial operators for:

1. The existing joint-training recipe from its original initialization.
2. Prefix-likelihood gradient pretraining, then the same joint training.
3. Prefix-likelihood expected-count pretraining, then the same joint training.

The second arm separates a pretraining benefit from a benefit specific to the
inference/update rule. Give the two pretraining arms the same data and objective.
Freeze a measured-work budget and accounting convention before scientific use;
equal epoch counts alone would not match EM and gradient computation. Charge
pretraining, smoothing, all joint updates, and final scoring. Report both
performance and cost against the unpretrained control.

Retain the same readout initializer in all arms, disclose its privileged
world-aligned structure, and learn it in the common joint phase. Do not align
latent coordinates using true states after pretraining. Likelihood can improve
while decision quality worsens, so do not select checkpoints by likelihood or
substitute likelihood for decision regret.

## Conditions before a scientific run

Register fresh data namespaces, all seeds, exact update rules, numerical and
compute caps, source/runtime hashes, and the successful fabricated qualification.
Retain the existing H1/H2 training labels and the all-seed short, blind and
observed criteria. Save all final checkpoints before fresh DEV is created.
Preserve failed attempts without replacement seeds or budget extensions.

Advance only if the decision criteria pass and the result is useful after
charging the extra training. A positive result would first establish a learning
recipe in this small synthetic world. Scenario shift and a second environment
would need separate protocols. Connectome constraints, online plasticity,
JEPA objectives and RL remain later hypotheses, each requiring a comparison
that isolates its contribution.

The social-media decision-head recipe is a separate serving/calibration topic;
see the [primary-source review](jev-architecture-source-review.md). Temperature
scaling preserves the chosen action and cannot repair these ranking failures.
