# Action-conditioned prediction through observation gaps

Unregistered next architecture hypothesis. No collection, fitting, performance
claim or admission of earlier TEST/confirmation data follows from this proposal.

## The capability that is missing

The current GRU already receives the previous action, current odor, public belief
summaries and query-error feedback. Adding those inputs is not a new mechanism.
Its objective predicts current teacher scores and prequery scores. It does not
train a transition that can advance under a proposed action without the next
observation. The fast score memory adjusts only three independent contrasts
among four actions and cannot change the slow recurrent state.

The completed readout and memory failures motivate testing this distinct
capability. They do not show that recurrent state is unhelpful. The direct-solve
baseline first checks whether further gains can be explained by optimizing a
fixed representation more effectively.

## Proposed mechanism

Separate action prediction from observation assimilation:

`prior_next = F(posterior, action)`

`posterior_next = G(prior_next, observation_next)`

Start with a small state comparable to the existing 28-dimensional GRU. Use a
fixed public observation/belief representation initially, an action-conditioned
latent rollout objective, a categorical next-odor prediction loss and the
existing decision objective. Stochastic odor outcomes require a distribution,
not a deterministic target pretending the future is known. Future teacher scores
may supply training labels but never rollout inputs.

## Experiment that distinguishes the mechanism

Collect complete legal action blocks chosen from a shared observed prefix
**before** their future odor observations. During each block, withhold odor,
refreshed belief summaries and refreshed analytic scores from the predictor.
Ordinary collector actions can encode the intervening observations and cannot
silently substitute for these precommitted blocks or counterfactual branches.

Train on gaps of at most four steps and evaluate eight-step gaps on fresh
originating cases. Keep all blocks from one originating case in the same split.
Retain normal-observation evaluation to detect regressions.

Compare the action-conditioned recurrent predictor with a parameter-budgeted
direct horizon predictor receiving the same prefix and precommitted action
sequence, an action-blind recurrent control and the strongest solved-readout
baseline. Match data and target information; charge collection, feature
construction, fitting and rollout computation. A separate observation-reset
control can test whether carried state causes any gain, if included before
registration rather than added after a favorable result.

Require improved proper odor-prediction loss and downstream decision gap through
longer gaps without worsening normal-observation behavior. Thresholds, resource
budgets, seed allocations and all controls must be registered before collection.
Prediction gains alone do not establish autonomous planning or control gains.

## Prior art and possible contribution

[PlaNet](https://arxiv.org/abs/1811.04551) already combines recurrent latent
dynamics and multistep objectives under partial observability.
[TD-MPC2](https://arxiv.org/abs/2310.16828) learns action-conditioned latent
transitions for control. [V-JEPA 2](https://arxiv.org/abs/2506.09985) supplies a
relevant action-conditioned prediction example with a pretrained visual encoder.
Borrowing these mechanisms is established engineering. A paper would need a
specific improvement and convincing comparisons, potentially a reliable way to
decide when a new observation should correct or replace a stale latent forecast.
Neither novelty nor effectiveness is established here.
