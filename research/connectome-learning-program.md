# Learning mechanisms for the next OpenJev experiments

Updated September 18, 2026. The completed
[adaptive-search study](reacher-adaptive-search.md) gives this roadmap a stronger
control baseline. With unchanged fitted models, residual-family CEM256 lowered
mean episode cost by **25.48% under ordinary sensing and 25.26% under shifted
sensing** versus RS256, at the same candidate-scoring budget. The study passed
all eight primary search checks; CEM256 passed all 15 control-competence checks.
This supports conventional adaptive planning. It does not establish useful
recurrent memory, biological topology, or a JEPA learning contribution.

The earlier [corrected Reacher study](reacher-reward-residual-control.md) remains
a failed continuation test under its original planner. The new search result
preserves that history and does not rerun its memory-reset interventions.

The proposed mechanism is **action-conditioned prediction of future recurrent
state, followed by a controlled test of biological wiring and temporal
hierarchy**. Hold the now-qualified CEM256 planner fixed for the next learning
comparison. Raw and latent objective components are implemented, but no model
has been trained with either auxiliary objective.

## What we are borrowing

| Source | Mechanism to test | Boundary |
|---|---|---|
| [I-JEPA](https://arxiv.org/abs/2301.08243) and [official code](https://github.com/facebookresearch/ijepa) | Asymmetric prediction of stop-gradient targets from an EMA teacher | The original task predicts image regions, not action-dependent dynamics; its backbone is a transformer. |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) and [official code](https://github.com/facebookresearch/vjepa2) | Action-conditioned latent prediction and planning after representation learning | Its robot trajectories include actions and proprioception. Goal-latent planning is not reward/value learning or online weight adaptation. |
| [VICReg](https://arxiv.org/abs/2105.04906) and [official code](https://github.com/facebookresearch/vicreg) | Measure feature collapse; test variance/covariance regularization separately | Noncollapsed features need not preserve useful control information. Avoid treating changing physical states as invariant. |
| [TD-MPC2](https://arxiv.org/abs/2310.16828) and [official code](https://github.com/nicklashansen/tdmpc2) | Pair latent prediction with reward, value and policy objectives | Its online data collection differs from our fixed dataset. Adding all of these at once obscures the mechanism. |
| [S4WM](https://arxiv.org/abs/2307.02064) and [S5 implementation](https://github.com/lindermanlab/S5) | Conventional state-space sequence backbones as non-attention controls | The world-model study does not imply every SSM improves control or rollout speed. A sequence library is not a ready-made robot world model. |
| [iCEM](https://arxiv.org/abs/2008.06389) | Adaptive proposal distributions with explicit scoring budgets | The first OpenJev comparison omits colored noise and elite reuse; those would require later ablations. |

The [broader robotics reading](connectome-robotics-next-experiments.md) adds
MTS3/HiP-RSSM context separation, DALI adaptation, uncertainty-aware control,
continuous-time cells and Flybody transfer. These are established ingredients.
Combining their names does not establish novelty.

The separate [Qwen RLCD reference review](parallel-constrained-decoding-reference.md)
identifies shared-prefix caching as a useful text-inference comparison. That
repository supplies an inference engine around standard Qwen weights, not a
new learning algorithm for this research program.

## A small JEPA-style learning experiment

The [auxiliary component](../src/openjev/research/reacher_latent_consistency.py)
is an engineering preview for the existing deterministic GRU/residual-GRU state
interface. It does not implement video pretraining, V-JEPA 2, a stochastic RSSM
teacher or a biological model. Existing frozen losses and checkpoints stay
unchanged.

The preview passes 41 synthetic checks, plus 67 existing world-model and
reward-residual checks. Tests include action alignment, gradient isolation,
poisoned missing features, terminal boundaries and the seven-step span needed
to bridge six missing observations. Independent review found no blocking issue.
These engineering checks do not establish control performance. No model has
been trained with this auxiliary objective.

Start from a student's recurrent state after assimilating a public observation.
Advance it using the recorded commands without intermediate observations. A
small head predicts the teacher's future assimilated recurrent state. The EMA
teacher processes the recorded public history with no gradients. Future
observations are training targets only; they never enter a student prediction
before their simulated arrival. Missing angular features must be zeroed even
when a training array happens to contain hidden values.

Test fixed horizons of 1, 3 and 7 steps. Both roots and targets require observed
public measurements: bridging six consecutive missing measurements needs a
seven-step target, not six. A ten-step blackout needs an eleven-step span;
longer gaps remain a separate generalization test. Keep the current native reward and
observation losses as anchors. Stop at episode boundaries and use explicit
validity masks. Report latent variance and effective rank. Add variance and
covariance regularization only as a separate, named treatment. The teacher and
auxiliary head add training computation; omit them at deployment when the
controller does not need them.

The next comparison keeps CEM256, the GRU architecture and the residual reward
formulation fixed:

1. Unchanged native observation/reward objective.
2. The same objective plus matched-horizon raw endpoint prediction.
3. The same objective plus EMA-teacher latent consistency.

Use fresh fitting seeds with paired student initializations, identical training
data and minibatch orders, and matched optimizer-update counts. Match auxiliary
horizons and masks. Evaluate on fresh paired native seeds under ordinary and
longer-gap sensing, with the same CEM256 budget in every arm. Include a
prespecified memory-reset intervention to test whether control depends on
recurrent history. The completed search cohort is development evidence, not a
fresh confirmation set for choosing the learning objective.

Report all training and evaluation computation, including teacher passes and
updates, student transitions, auxiliary and shared-decoder work, diagnostics,
planner overhead, native interactions, memory and wall time.
A compute-matched raw-prediction comparator is necessary before claiming that
latent targets are more efficient. More training computation is not free.

The [raw endpoint component](../src/openjev/research/reacher_raw_endpoint.py)
now supplies the engineering control for the same horizons, observed endpoints,
terminal boundaries and horizon weighting. It adds the same hidden-to-hidden
linear predictor as the latent auxiliary, then uses the student's existing
observation decoder to predict cosine/sine targets. At width 64, each auxiliary
adds 4,160 trainable predictor parameters. The raw decoder receives additional
gradients; raw and latent targets differ in dimension and scale. Matching this
parameter count does not match gradient strength or total training computation.

Its 50 synthetic checks pass, alongside 108 existing latent/world-model/residual
checks. Both components leave the original anchor loss unchanged. Simply
increasing the original rollout horizon would also change reward supervision
and terminal weighting, so it is not the matched comparator. No objective
ablation has been trained yet.

A proposed continuation rule is at least 5% lower native episode cost than both
controls on fresh ordinary and longer-gap cases, no paired fit worse, and a
consistent deterioration when useful recurrent memory is reset. The exact
cohort, seeds, budgets, masks, loss weights and thresholds require a new frozen
protocol before training. Lower latent loss alone cannot pass.

Reset hidden state before assimilating the last visible packet, preserving
current measured information. Pair initial conditions, sensing and exogenous
noise in closed loop, while allowing actions and later observations to diverge.
A reset loss establishes sensitivity to prior hidden state; reset-induced
distribution shift remains an alternative explanation. A separately trained
current-observation baseline is required for a stronger memory-advantage claim.
The [unfrozen objective draft](reacher-objective-ablation-draft.md) specifies the
next engineering and experiment requirements; it contains no scored result.

Checkpoint metadata must also bind the auxiliary horizons, teacher EMA momentum,
regularization weights and model settings. These Python configuration values
are not contained in a tensor-only state dictionary. Count CPU transfers and
diagnostic computation as well as teacher/student forward passes.

## Make biology the tested variable

If useful memory is demonstrated, cross the useful objective with biological
and multiple role/degree-preserving rewired graphs using the same sparse
recurrent cell. Keep node count, signed edge counts, input/output roles,
activation, optimizer and temporal dynamics matched. Preserve role-to-role
mixing rather than accidentally damaging an interface only in the control.

Then cross single-rate versus fast/slow state updates. Include timescale
shuffles within matched neuron roles. Cell-type labels require actual annotation
provenance; otherwise they are learned groups. Compare dense GRU, two-rate GRU
and an independently implemented S5 model under both objectives. A biological
mask imposed on S5 changes the S5 mechanism and would be another treatment.

The topology claim concerns an interaction: does the new objective or temporal
organization help biological wiring more than it helps matched rewires? Test
multiple graph draws and fitting seeds, and repeat on a second embodied task.
A gain common to all graphs belongs to the learning objective or hierarchy.

Use interventions chosen on development data: erase slow context, reset fast
state, shuffle action history and compare targeted lesions with size/degree/
activity-matched random lesions. An interpretable probe is useful only if its
state intervention changes actual control. Small white actuator noise is not
persistent hidden physics; qualify a task with identifiable hidden actuator
changes before interpreting slow context as system identification.

## Biological learning requires more than biological edges

[Differentiable plasticity](https://arxiv.org/html/1804.02464v3), equations 1-3,
separates a fixed synaptic weight from an episode-specific Hebbian trace and
learns the plasticity coefficients through backpropagation. The trace resets
between episodes; base weights and plasticity coefficients persist. Its maze
and memory results motivate a conventional plastic-recurrence comparator, not
evidence that a fly connectome will improve robot control.

[Backpropamine](https://arxiv.org/html/2002.10585v1) extends this idea with a
learned signal that modulates plastic updates. This is useful prior art for a
fast adaptation mechanism. Outer training still uses gradients; a locally
expressible deployment update does not make all training biologically local.
The [authors' code](https://github.com/uber-research/backpropamine) carries a
non-commercial license for its core experiments. Borrow the published mechanism
with attribution; do not silently import that code into the MIT project.

[E-prop](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/) combines synaptic
eligibility traces with learning signals to approximate recurrent gradient
training. It is relevant if training memory or temporal credit assignment is
the measured bottleneck. It is a separate learning-rule experiment, not another
name for a Hebbian trace or a drop-in exact replacement for backpropagation.

A later bounded contrast should cross fixed versus plastic recurrence with
biological versus matched rewired graphs. Give every arm the same deployed
information. Begin with an episode-local plastic state, an explicit reset and
a bounded update; preserve the graph's sign and role constraints if those are
part of the hypothesis. Compare ungated plasticity, learned modulation and a
conventional recurrent/context-adapter baseline at matched total state and cost.

At deployment, update from real observations and issued actions only. If the
update uses prediction errors, wait for the corresponding observable outcome;
blackouts do not provide hidden labels. Planner branches must receive private
copies of both neural state and plastic state. Imagined success must not update
the actual agent's memory or contaminate another candidate's evaluation.

Measure adaptation after a persistent hidden dynamics change, recovery time,
ordinary control, forgetting, trace saturation and full decision cost. Reset
only plastic state as a causal diagnostic. A positive plasticity effect shared
by biological and random graphs supports plasticity, not connectome-specific
learning. This stage is proposed, not implemented or scored.

## Improve the experiment loop without learning the test set

The existing source-bound runners already freeze within-run plans and preserve
completion/failure artifacts. The cross-experiment improvement loop should use
these boundaries:

1. Read completed audited development results and name one measured failure.
2. Register a bounded mechanism change, closest conventional baseline, total
   budget, expected failure signature and continuation rule.
3. Test engineering invariants on synthetic fixtures, then freeze code, data,
   graph/checkpoint provenance, runtime and every random role before scoring.
4. Run every declared seed and comparator. Preserve invalid, failed and
   scientifically negative attempts; count all work in cumulative cost.
5. Recompute scores from saved outputs. Native replay, source integrity,
   scientific efficacy and a novel-architecture claim are distinct checks.
6. Promote only on the declared utility-versus-compute criterion. Publish
   diagnostics and all model fits, not just the winner. A failure changes the
   next separately declared question, not the finished experiment's threshold.
7. Treat cohorts used to choose the next idea as development data. Reserve an
   untouched confirmation cohort after the mechanism is selected, and then a
   second environment. Repeated fresh development tests alone do not remove
   adaptive model-selection bias.

This is a research improvement loop, not an established self-improving learning
algorithm. Model weights change only in explicitly declared training stages;
reporting never silently refits them. No unbounded mutation or automatic
promotion policy has been added.

RL follows after a competent predictor/planner comparison: a separate online
interaction stage can test a learned policy prior or continuation value against
model-free and fixed-model controls. Charge all new interactions and training;
never describe results from a fixed corpus as online-RL sample efficiency.
