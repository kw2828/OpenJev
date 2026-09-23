# Recurrent score forecasts: prior art and the actual question

Recurrent prediction followed by observation correction is established prior
art. The [fixed OpenJev pilot](otto-score-forecast-protocol.md) asks whether one
small implementation predicts a planner's choices better between queries. It
does not claim a new general prediction-correction mechanism.

| Primary source | Established idea | Boundary for this pilot |
|---|---|---|
| [Recurrent Kalman Networks, Becker et al., ICML 2019](https://proceedings.mlr.press/v97/becker19a.html) | Learn a latent representation with state prediction, uncertainty and observation correction. | The score GRU has no Kalman update or uncertainty representation. Replacing an anchor with a planner output is exact relative to that planner, not to an environment's true value function. |
| [PredNet, Lotter et al., ICLR 2017](https://arxiv.org/abs/1605.08104) | Use prediction errors to update recurrent representations, motivated by biological predictive coding. | A predicted score increment is not an observed error. The pilot does not feed hidden skipped-step labels into its memory or implement biological wiring. |
| [Successor Features, Barreto et al., NeurIPS 2017](https://arxiv.org/abs/1606.05312) | Factor value into expected discounted feature accumulation and reward weights. | Four remembered planner costs lack that decomposition and Bellman constraint. Reward-transfer guarantees do not automatically apply to changing sensing dynamics. |
| [Value Prediction Networks, Oh et al., NeurIPS 2017](https://arxiv.org/abs/1707.03497) | Learn abstract transitions that predict quantities useful for planning. | Predicting scores on a collected path does not itself demonstrate action-conditioned planning or useful autonomous rollouts. |

The residual GRU feeds its previous predicted score vector back into the next
update. The direct GRU instead repeatedly offsets the fixed query anchor. Both
reset hidden state at every four-step window. A finite-history MLP tests whether
the same observations can explain any gain without recurrent computation.

This comparison can identify a useful model structure on these targets. It
cannot establish memory correction across query cycles, successful autonomous
control, a connectome benefit or publication-level novelty.

## A later, separate hypothesis

At a real query, retain hidden state and update it from the discrepancy between
the new teacher scores and the prior forecast. Constrain this added correction
to vanish at zero discrepancy. Compare it with an ordinary persistent GRU that
receives exactly the same information, under matched parameter, query and
training budgets. Skipped annotations remain inaccessible to both models.

A new protocol would have to fix independent cases and require lower
teacher-score gap without worse action agreement, especially after corrections.
An ordinary GRU matching the result would reject the proposed structural
advantage. Later autonomous evaluation must also count correction and training
costs. This is a falsifiable extension motivated by existing work, not an
implemented result or an authorization to retune the current pilot.
