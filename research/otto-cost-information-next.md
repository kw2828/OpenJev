# Next: separate conditional labels from recurrent state design

**Prospective proposal, written while the diagnostic is running.** This note assumes no headroom result. A follow-up is conditional on the [current prespecified criterion](otto-cost-information-protocol.md) resolving headroom and needs a separate protocol freezing sources, data, budgets and acceptance criteria before execution. The current study and earlier failed studies retain their original outcomes and scope.

## First isolate target noise

Let `I` contain the full observed public history and a committed action path. Let `X = g(I)` be the student's input representation, and `Y` the fixed-horizon teacher-cost vector, with zero cost on absorbing-found histories and a legal set determined by `I`. Define `mu(I) = E[Y | I]`. For square-integrable targets and any prediction `f(X)`, the exact identity is

```text
E[||f(X) - Y||^2]
  = E[||f(X) - mu(I)||^2] + E[tr Cov(Y | I)].
```

Thus exact conditional-mean labels preserve the population squared-error optimum while removing conditional label variation. The identity also holds with the same nonnegative, `I`-measurable weight multiplying every term, provided expectations exist. Dropping found cases, changing survival weights, or using a different horizon average can change the objective. Finite Monte Carlo labels remain noisy; biased approximations need separate checks. Lower squared error does not guarantee lower decision regret.

Integrating tractable uncertainty is established variance reduction, exemplified by [Rao-Blackwellised Particle Filtering](https://arxiv.org/abs/1301.3853). Here it would be conditional-target construction, not that particle-filter algorithm. Reference headroom does not establish what a finite learned recurrent state can attain. Teacher evaluation must average costs of individual legacy-filter histories under the strict reference law, not evaluate the teacher on an averaged belief.

## Source-level correction to the registered information comparison

The frozen protocol described the reference as having richer information than the student's 31-feature prefix. Source review during collection, before prediction or outcome analysis, found that this overstates the difference in observed information. The protocol and all thresholds remain unchanged; this paragraph corrects its interpretation.

The [feature builder](../src/openjev/research/otto_query_gate.py) preserves odor categories in columns `7:11`, prior actions in `11:15`, grid positions in `0:2` and sensing regime in column `6`. The [collector](../scripts/collect_otto_cost_information.py) retains all nine rows from reset, and the [model](../src/openjev/research/otto_action_latent_model.py) assimilates every row. Thus the retained prefix preserves the observed history needed to reconstruct both beliefs given the fixed public initialization and laws. It is not merely a single vector of belief statistics.

The reference has explicit sampling-law knowledge, full filtering computation and teacher evaluations, not additional realized observations. Information can be lost inside the learned 28-dimensional state, and finite-data learning or computational approximation can fail. This diagnostic does not demonstrate an information deficit at the input interface. Its sampled horizon-8 reference also remains an approximate decision rule, not an exact information floor.

## Then test one recurrent mechanism

The proposed mechanism is a small recurrent bank of decision-relevant predictive moments. Its observed update assimilates an odor; its blind update preserves the corresponding probability-weighted cost predictions. For a fixed action sequence `u`, let `v_h(H,u)` predict the unconditional terminal teacher-cost vector. The target consistency is

```text
v_h(H,u) = sum_y p(y | H,u[0]) * v_(h-1)(H,u[0],y,u[1:]).
```

Found branches contribute zero. Apply consistency to predicted cost moments, not a nonlinear head evaluated on an averaged latent state. Four current action costs generally do not form a representation closed under observation updates. A specified additional predictive basis and its measured closure error would be the architectural question.

[Value Prediction Networks](https://arxiv.org/abs/1707.03497) already learn action-conditioned abstract dynamics for reward/value prediction. [DeepMDP](https://proceedings.mlr.press/v97/gelada19a.html) already couples reward prediction with next-latent-state distribution learning. Its MDP and smoothness assumptions do not automatically cover our compressed histories. Neither conditional labels nor a decision-focused recurrent model alone establishes novelty.

## Four matched arms

1. Current GRU with sampled teacher-cost labels.
2. Identical GRU with conditional expected-cost labels.
3. Structured predictive-moment recurrence with those same conditional labels.
4. The same structured model with the consistency constraint removed.

Match input information, data, forecast objectives, optimizer recipe and fit seeds. Predeclare parameter and compute comparisons, charge all teacher-label construction and branch evaluations, and distinguish matched training examples from matched total cost. Arm 2 isolates denoising or distillation; arms 3 and 4 test the proposed structure. Use fresh evaluation paths and retain both long-gap and normal-observation performance under the scenario shift. Any architectural claim would require improvements beyond the identical-GRU conditional-label control, not merely better averaged targets.
