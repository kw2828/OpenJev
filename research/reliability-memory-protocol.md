# Learning causal observation reliability

This is a new experiment after the action-range loss comparison failed 17/54
conditions. That result and every earlier failure remain unchanged. The new
question is whether a small recurrent reliability module improves decisions
beyond a conventional switching filter and an otherwise identical memory-reset
control. It does not establish novelty merely by adding a recurrent gate.

## Models and information

Freeze all five `rounded_mse` final backbones from the completed independent-
cohort study, in their original cohort order. Select no winning checkpoint.
Export their learned transition, emission, hazard and cost matrices once. Every
arm in a cohort receives the same frozen fields. Backbones retain 352 stored
parameters; adaptation adds parameters and state rather than replacing them.

The supplied structural family is O(q) = (1-q)O + q/4, with bank values
q = {0, 2/7, 4/7}. For the ideal original emission these correspond to noise
0.12, 0.30 and 0.48. This family is prior task knowledge supplied to all arms.
It does not reveal the active noise level or a future change. Compare:

1. `unchanged`: original emission, no new parameters.
2. `global`: one learned, episode-independent q in (0, 4/7).
3. `static_bank`: exact joint state over three reliability modes and eight
   hidden states, uniform mode prior and no mode transitions.
4. `markov_bank`: joint state with a learned initial/reset prior and learned
   constant probability of resetting mode to that prior. This is exact finite
   filtering for its assumed transition law, not an approximate weak control.
5. `recurrent_bank`: the same joint state and learned prior, with a four-unit
   GRU selecting the next mode-reset probability from past evidence.
6. `reset_bank`: identical parameters and initialization to `recurrent_bank`,
   but the GRU hidden state is zeroed before every evidence update. Its Bayesian
   joint state still persists, isolating the extra GRU memory.

The GRU arms store 164 added parameters, but the reset control's 48 recurrent
matrix entries always multiply zero. This matches stored size, not effective
parameter count. Their persistent numerical state has 28 entries; the Bayesian
banks have 24 and the single-emission controls have eight, excluding flags and
temporary work.

At each action, reliability mixing preserves the latent-state marginal. The
gate uses the previous GRU state. Predict the five-event law before observing
the current event, condition the joint state once, then update the GRU from
the three mode likelihoods, three posterior mode probabilities and surprise.
Log features may be bounded as documented by the implementation; probabilities
are never repaired or clipped for likelihood scoring. Found is absorbing.
Reset has a uniform latent prior and no transition or hazard.

## Data and fitting

Use five fresh namespaces 438260924 through 438260928, paired with model seeds
438261001 through 438261005 and the five frozen backbones in order. Each cohort
has 512 TRAIN episodes of 32 actions plus one reset event. Cycle equally through
static noise 0.12, static 0.48, and one unannounced change in either direction.
Change location is uniformly sampled from event indices 8 through 24. Actions
and all eight-step fork action sequences are drawn before observations.
Retain every episode, including early found events; do not replace seeds.

Fit only on public action/event histories using observed-event negative log
likelihood. The four trainable arms receive identical batches: 256 Adam updates,
batch 64, learning rate 0.01, gradient clipping 5, CPU float64 and one numerical
thread. Each update uses a fresh batch from deterministic shuffled TRAIN epochs.
Use the final update without early stopping or model selection. Rotate fitting
order by cohort index. Save initial/final model states, final Adam state,
minibatches, every update loss and full fit time. No gradient enters the frozen
backbone. All 20 trained final checkpoints precede all evaluation generation.

Evaluate each cohort on four fresh 512-episode strata: BASE noise 0.12, SHIFT
noise 0.30, SWITCH with one 0.12/0.30 change in alternating directions at a
random index 8..24, and STRESS noise 0.60. The last lies beyond the bank family
and is a descriptive stress test, not part of model selection. No evaluation
episode updates parameters. Hidden noise, switch indices, case IDs and cost/state
targets never enter a model call. Precommitted fork actions are public controls,
even though the generator stores them alongside evaluation targets.

## What is measured

Score prequential log loss on every actual event, including first found but not
padding. Score decisions immediately after each assimilated nonterminal event,
and four/eight-step blind forks from those states. A fork receives no subsequent
events; observation noise cannot change its already-fixed blind dynamics.

The generator's reference posterior conditions on the private true noise path.
It supplies exact conditional expected action costs for evaluation only. Regret
against its best action is an explicitly privileged reference, not an attainable
public-information Bayes floor. Paired differences compare the same cost targets.
Event KL against that reference includes the value of its extra information and
must not be called empirical calibration. Training uses sampled public events,
not those targets or latent-state labels.

The primary regret is the per-episode mean of H4/H8 fork regret over alive
boundaries with event index at least 8, then the mean over supported episodes,
then the equal-cohort mean. Report support and all excluded late-support cases.
Keep H4, H8, immediate decision regret, full-event log loss, all cohorts and all
four strata separately. Report SWITCH delays 1, 2, 4 and 8 after the actual
change descriptively. Report inference time including blind forks and the gate,
training time, trainable parameter counts and recurrent-state storage. Runtime
is one machine's complete measured calls, not FLOPs or equal-compute training.
Histories use random public actions. This tests estimation and counterfactual
decision scoring, not autonomous gameplay, robot control or a learned policy.

## Prospective continuation rule

The fixed candidate is `recurrent_bank`; primary controls are `markov_bank` and
`reset_bank`. All 13 conditions are required:

- For each control and each of SHIFT and SWITCH, positive mean control regret
  and at least 10% lower candidate equal-cohort primary regret: four conditions.
- For each of those four comparisons, strictly lower primary regret in at least
  four of five paired cohorts: four conditions.
- On BASE, candidate equal-cohort primary regret no more than 5% above each
  control, with an absolute tolerance of 1e-6: two conditions.
- Apply the same BASE preservation bound against the unchanged backbone:
  one condition. Improvement over newly introduced filters cannot hide harm
  to the original model in its training regime.
- Candidate mean inference time across all cohort/stratum calls no more than
  twice `markov_bank`: one condition.
- Candidate BASE mean event log loss no more than 0.01 nats above `markov_bank`:
  one condition.

Require at least 256 late-supported evaluation episodes per cohort and stratum,
all finite predictions, correct probability mass, unchanged backbones and an
independent saved-output audit. A pass is pilot evidence for further validation,
not statistical significance, a novel architecture claim or robotics transfer.
A failure closes this recipe without retuning on these evaluation episodes.

## Execution and provenance

Develop numerical tests on fabricated fixtures. Then freeze the protocol,
source closure, environment versions and exact parent checkpoint hashes in an
exclusive registration. Complete the frozen qualification before scientific
training: tests, lint and a TRAIN-only feasibility probe on a separate namespace.
The probe uses 64 episodes and eight updates per trainable arm. Twice its measured
per-fit time scaled to 256 updates must be below 120 seconds. It provides no
effectiveness-based selection. Five times the sum of the four projections must
also be below 1,200 seconds, leaving time for data and evaluation within the
complete cap. Preserve the probe with the final evidence.

Native caps: qualification 180 seconds, scientific run 1,800 seconds, saved-output
audit 300 seconds. Each scientific fit also checks a 180-second limit. Native
supervision must close cleanly before publication. Preserve partial outputs and
failure receipts; no restart, replacement, extended budget or frozen-source edit.
The independent audit reconstructs metrics and rules from saved targets and
predictions without running models, generators or optimization.

## Related work and scope

[Recurrent Kalman Networks](https://proceedings.mlr.press/v97/becker19a.html)
already integrate uncertainty with learned recurrent filtering.
[Switching Recurrent Kalman Networks](https://arxiv.org/abs/2111.08291) combine
switching latent dynamics and recurrent inference. Finite mode filtering is
also established. Our proposed extra benefit is specifically learned history-
dependent reliability updates over strong exact-filter and memory-reset controls.
This experiment must establish that benefit before broader architecture work.
