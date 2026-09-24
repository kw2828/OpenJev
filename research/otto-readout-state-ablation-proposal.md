# Next question: readout learning or recurrent-state adaptation?

**Proposal only, not registered or executed.** The
[closed Bayesian comparison](otto-residual-reanalysis-results.md) favors ordinary
joint auxiliary training over the residual-memory candidate. It does not explain
which part of joint training matters.

In the original implementation, pretraining updates all 6,112 slow-model
parameters for 80 epochs. Joint auxiliary training copies that same-seed
checkpoint and continues the GRU and both readouts for 40 epochs. Trace-delta
training instead freezes the slow predictor and updates only a 224-parameter
projection. The branch data, objective, optimizer settings and schedules match.

One parameter-unfreezing ablation can separate three explanations:

| Continuation arm | Trainable parameters | Question |
| --- | ---: | --- |
| Action-residual readout only | 116 | Does fitting a better readout of the same recurrent state explain the gain? |
| Both readouts, frozen GRU weights | 232 | Is adapting the base prediction and resulting innovation sufficient? |
| Full joint auxiliary model | 6,112 | Do recurrent weight updates provide a further advantage? |

The middle arm is essential: the base readout changes the prediction error fed
back into the GRU. Frozen GRU weights therefore do not guarantee unchanged
hidden trajectories. The residual-only arm can preserve those trajectories
because its readout does not feed the innovation path.

Start every arm from the same three original 80-epoch parent checkpoints. Keep
the 54 TRAIN paths, period-four observations, legal-centered nonquery MSE plus
equally weighted all-four prequery MSE, /64 score units, Adam 0.003, clip 5,
six-episode batches, 32-step carry detachment and fresh branch optimizer states.
Match the 40-epoch, 360-update continuation and restarted seed permutations.
Record actual computation rather than treating equal updates as equal cost.

Matching full joint training with residual-only learning would support a fixed
state/readout explanation. Matching it only with both readouts would show that
recurrent weight adaptation is unnecessary under this recipe. A reproducible
full-joint advantage would support a role for recurrent parameter updates,
without by itself identifying a world-model or biological mechanism.

Before execution, qualify the parameter masks, optimizer updates and hidden-state
invariants, then register data access, budgets, comparisons and continuation
rules. This proposal specifies no admission, new seeds or success threshold.
Both failed screens remain closed. Their reserved confirmation cohorts and the
earlier TEST remain unused; any claim of generalization needs a separately
registered fresh cohort.
