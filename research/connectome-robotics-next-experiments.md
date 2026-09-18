# Connectome robotics: mechanisms worth testing next

Primary-paper reading follow-up, September 18, 2026. This extends the
[earlier reading map](connectome-world-model-robotics-reading.md). Paper results
are source-reported, not reproduced here. Proposed experiments below are not
frozen protocols or new performance results.

The strongest direction is **a recurrent belief model that separates fast
motion from slow hidden physics, then tests whether biological connectivity
improves adaptation at the same computational cost**. A connectome plus a world
model is already covered by related work. A useful contribution needs a specific
mechanism, an interaction against rewired controls, and real control improvement.

Our immediate prerequisite is the [Reacher evaluation correction](reacher-rng-independence-correction.md).
The six completed residual/free-head fits can be evaluated unchanged under a
new stream contract. Their interrupted evaluation supplies no efficacy result.
The separate [30-fit chess mapping study](chess-connectome-mapping-study.md)
failed its continuation rule; it does not support biological superiority.

## Additional papers that change the design

### Separate fast state from slow context

[HiP-RSSM: Hidden Parameter Recurrent State Space Models](https://arxiv.org/html/2206.14697v3),
Shaj et al., 2022/2023, sections 2 and appendix D. A Gaussian context posterior
conditions fast state filtering. At deployment, a context set of previous
interactions updates the task belief for the next window. The hidden parameter
is locally fixed; it is not a universal change-point detector. The paper tests
system identification and prediction on robots with changing dynamics, not a
closed-loop control advantage. Borrow the explicit context/state separation and
compare posterior means with uncertainty-aware conditioning. Do not label an
ordinary GRU hidden state a Bayesian posterior.

[Multi Time Scale World Models (MTS3)](https://arxiv.org/html/2310.18534v3),
Shaj et al., 2023, sections 3-4 and 6. Slow context beliefs and fast state beliefs
evolve on different clocks. Training masks both individual observations and
whole windows; the task includes masked velocities. The experiments establish
prediction quality, including probabilistic estimates, rather than policy
return. This is direct prior art for a two-timescale world model. Compare our
candidate against a conventional two-rate recurrent model and MTS3-style
masking, not only a single untuned GRU. Extra internal iterations must not
advance physical time.

[DALI: Dynamics-Aligned Latent Imagination](https://arxiv.org/html/2508.20294v3),
January 2026 revision, sections 4 and 6. A compact context encoder learns from
short histories through forward-dynamics prediction and conditions a Dreamer
world model and policy. Shallow versus deep context injection is an explicit
ablation. This is closer prior art than treating a context head as new. The
formal setup keeps context fixed within each episode; abrupt physical changes
need their own evidence. Include domain-randomized Dreamer and a true-context
privileged reference, with access clearly distinguished.

### Make model uncertainty useful to control

[Uncertainty-Aware Robotic World Model (RWM-U)](https://arxiv.org/html/2504.16680),
Li et al., 2025, sections 4-5 and appendix A. It combines autoregressive
prediction, a shared recurrent feature extractor and bootstrap observation
heads. MOPO-PPO subtracts an uncertainty penalty from imagined reward. The
paper reports hardware deployment and an important tradeoff: too little
penalty exploits model errors, while too much can produce near-inaction.
Its implementation is available in the [authors' repository](https://github.com/leggedrobotics/robotic_world_model).
For OpenJev, compare no penalty, ensemble disagreement and calibrated residual
penalties with identical action proposals. Shared-core disagreement can miss
errors common to every head, so include a small independently trained ensemble
at matched total cost. Predictive variance is not automatically calibrated.

[DuSt-MPC: Dual Online Stein Variational Inference](https://arxiv.org/html/2103.12890),
2021, section IV. Particles represent uncertainty in both physical parameters
and control sequences. Only real transitions update the dynamics belief;
imagined desirable outcomes must not become evidence about the environment.
The approach uses a parametric simulator and includes a real vehicle test.
Compare planning with a posterior mean against planning across parameter
samples. Keep a sampled persistent parameter fixed along each rollout and
charge all extra transitions. A same-cost planner with more action candidates
tests whether any gain comes from belief treatment or simply more computation.

[DoublyAware](https://arxiv.org/html/2506.12095v1), 2025, sections III-B/C,
combines TD-MPC-style planning with a Group Relative Policy Constraint and
candidate quantile filtering. This is related work for the earlier GRPO and
conformal ideas, but it is not interchangeable with GRPO. Equations 4-7 derive
the threshold from the current candidates' normalized predicted returns.
Our assessment: that operation alone does not calibrate error against realized
outcomes. Include an ordinary top-quantile ranking control before assigning
gains to conformal uncertainty.

[Self-adapting Robotic Agents through Online Continual RL](https://arxiv.org/html/2603.04029),
Domberg and Schildbach, 2026, section III. Rolling observation and reward
residuals trigger DreamerV3 finetuning at a three-standard-deviation threshold.
Pre-change transitions are excluded from the new replay buffer. Tests include
simulated actuator damage and a real model vehicle. Thus residual-triggered
adaptation is established prior art. A narrower experiment would compare
updating only a small context adapter against full-model finetuning, frozen
inference, periodic updates and matched random-trigger updates. Charge data
collection and updates to the cost budget; forgetting and false triggers are
outcomes, not just implementation details.

### Match the structure to the body

[Closed-form Continuous-time Neural Models (CfC)](https://arxiv.org/html/2106.13898v2),
Hasani et al., 2022, methods and Table 7. Learned interpolation provides an
elapsed-time-dependent recurrent update without numerical ODE integration.
Irregular Walker2D results concern dynamics prediction, not closed-loop return.
The [official implementation](https://github.com/raminmh/CfC) includes several
gating variants. Test CfC against a GRU receiving the same elapsed time and
missingness inputs if sampling-rate transfer is the demonstrated problem.
Hold topology fixed first. This is an established cell family, not a newly
invented biological memory mechanism.

[Liquid-Graph Time-Constant Networks](https://arxiv.org/html/2404.13982v2),
2024, equations 8-10 and experiments. Graph filtering and state-dependent
damping support recurrent dynamics with stated contraction conditions.
Imitation plus DAgger is evaluated on simulated flocking and communication
shifts. The graph represents agents, not neurons. A signed directed fly graph
does not automatically satisfy assumptions used with normalized Laplacians.
If rollout instability is measured, compare ordinary versus controlled-gain
updates across biological and rewired graphs. Check memory loss as well as
stability: stronger contraction can erase useful context. An official code
release was not verified.

[Can a Compact Neuronal Circuit Policy Be Re-purposed?](https://arxiv.org/pdf/1809.04423),
2018, methods and Table II. An 11-neuron, 28-synapse worm circuit with learned
parameters and linear interfaces tackles simple control tasks. It includes
random-circuit comparisons, but their node/edge matching is weaker than signed
degree and role matching. [Legacy code](https://github.com/mlech26l/ordinary_neural_circuits)
is available. A compact circuit motif is a useful low-cost comparator to a
large fly graph. Match neuron dynamics, optimizer and interface capacity;
separate biological topology from sparse recurrence and decoder competence.

[SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour](https://arxiv.org/html/2606.19928),
Lan et al., 2026, section III and IV-B. Reflection equivariance constrains the
recurrent world model as well as actor/critic components. Mirrored-terrain and
hardware experiments motivate morphology-aware structure. The paper compares
hard equivariance, its removal, and a symmetry loss; it explicitly omits a
pure augmentation baseline. Include that baseline in our comparison. A
bilateral connectome candidate must compete with a generic symmetric network
and symmetry-preserving rewires. Natural left/right anatomy alone is not a
guarantee that the learned model is equivariant. First verify the environment's
actual state/action symmetry, including asymmetric faults.

[Whole-body physics simulation of fruit fly locomotion (Flybody)](https://www.nature.com/articles/s41586-025-09029-4),
2025, supplies a 102-degree-of-freedom MuJoCo body and pretrained locomotion
controllers. The reported controllers are feedforward MLPs trained with DMPO,
not connectome models; reported training uses roughly a billion walking steps
and a hundred million flight steps. The
[official code release](https://github.com/TuragaLab/flybody) supplies the simulator;
[the accompanying data record](https://doi.org/10.25378/janelia.25309105) supplies
imitation data and trained controllers. Reuse one fixed low-level controller under each high-level
decision model. This offers an embodied transfer test without conflating a
new memory mechanism with the cost of learning locomotion from scratch.

### Evaluate feedback and calibration, not only imagined error

[Do Better Imagined Rollouts Mean Better Robot Control?](https://arxiv.org/html/2609.02811v1),
2026, sections 3-5. In one simulated mobile-robot study, measurement-free
rollout error selects a different observer from the closed-loop optimum in
18 of 24 conditions, versus 5 of 24 for replay position error. The tested
models are analytic estimators and recurrent residual observers, not general
video world models. Extended blackout training also adds trajectories, so it
does not isolate masking alone. The useful lesson is to vary rollout horizon
and measurement-update frequency separately, then measure native control.
We should report action-ranking regret and recovery after observations return,
alongside prediction error.

[Adaptive Conformal Prediction for Motion Planning](https://arxiv.org/html/2212.00278),
Dixit et al., 2022/2023, sections 3-4. Delayed multistep outcomes update
horizon-specific uncertainty regions used by MPC. This supplies a concrete
causal calibration baseline. Its average coverage statement does not mean
every state, candidate action or full episode receives the same guarantee.
For our predictor, retain forecasts with their issue time and update only after
the required observation actually arrives. A sensor blackout does not provide
a hidden label for free.

[Who Moved My Distribution?](https://arxiv.org/html/2511.11567),
2025, sections 3-4. Tightening an MPC constraint changes interaction
trajectories, which can change the distribution used for calibration. The
proposed iterative recalibration has a convergence result under Lipschitz and
contraction assumptions, with exchangeable interaction trajectories per round.
This is a specific warning for our deployment test: residuals from exploratory
data need not calibrate plans selected by a different controller. Report
coverage on executed actions separately from offline or counterfactual
candidate coverage. Do not import the theorem without its assumptions.

[CoCoNav](https://arxiv.org/html/2608.07751), 2026, section IV. It combines
delayed horizon-specific conformal PI calibration with soft MPC and separate
verification of nominal and contingency plans. Its clearance theorem concerns
the executed first action and adds terms for missing tracks and uncertified
fallbacks. It is not a blanket guarantee of safe episodes. Borrow the explicit
fallback accounting: report how often the learned plan is used, when fallback
fails, and the cost of extra verification. A zero-command fallback in our
Reacher task is a reference controller, not a proven safe action.

## A staged experiment for OpenJev

### 1. Establish useful control with the existing fits

Finish a prospectively frozen, evaluation-only recovery of every final fit.
Keep the original seventeen criteria and compare all six models. If learned
planning remains weak, use the [matched-budget search comparison](reacher-planning-prior-work.md)
before adding a new recurrent architecture: one batch of 256 proposals versus
four CEM iterations of 64, charging every score and update. This is a planning
diagnostic, not architecture novelty. The frozen v1 experiment remains stopped.

### 2. Qualify a task that needs persistent physics estimation

Specify a new diagnostic with hidden, persistent actuator changes and missing
observations. White actuator noise alone is not a persistent hidden parameter
that longer memory can predict. Before architecture training, use a separate
qualification split to check excitation and identifiability, a fitted
system-identification estimator, short history and a tuned GRU. Stop if short
history saturates the benefit, as in the earlier Pendulum qualification.

The candidate has a fast observation filter and a slow context state that
conditions action-dependent dynamics. Compare a single-rate GRU, a two-rate
GRU, a probabilistic context model, and then biological versus rewired sparse
cores. Initially hold the planner, objective and data fixed. Do not add online
RL, conformal selection and topology in the same first contrast.

### 3. Test the biological mechanism

Cross topology with temporal organization: biological, independently sampled
signed-degree-preserving rewires and conventional sparse/block recurrence,
each with single-rate versus fast/slow dynamics. Include a control that
shuffles the assignment of timescales while retaining their distribution.
Preserve sensory/motor interfaces and node roles across rewires, including
role-to-role mixing. Add a timescale shuffle within matched roles alongside
the unrestricted shuffle, so disrupting the interface hierarchy cannot be
mistaken for evidence about biological edges.
If cell annotations are unavailable, call these learned groups, not biological
cell types. Report parameters, state size, transition evaluations and wall time
separately; none substitutes for the others.

Predefine the key interaction as the fast/slow improvement for biology minus
the same improvement for rewires. Test independent graph draws and fit seeds.
Measure integrated excess control cost after a hidden change, recovery time,
failure rate, ordinary return and decision cost on untouched episodes. A gain
shared by every graph supports a temporal-model improvement, not a biological
advantage.

Use state interventions to explain a successful result: erase only slow context,
reset only fast state, shuffle action history, and compare lesions with random
subsets matched for size, degree and activity. Select intervention sites on
training/calibration data. An effect on a diagnostic probe alone does not show
that the state supports control, and arbitrary patches may be off-manifold.

### 4. Add a decision layer, then test embodiment

After competent control, compare a learned plan, an ensemble-penalized plan and
a causally calibrated fallback selector. Keep a sealed calibration split and
report deployed-policy coverage, intervention rate, return and full cost.
Additional recurrent work, the gate and every fallback must be charged.
Neither confidence nor extra computation may fix missing information.

For transfer, use an independently specified embodied task. A practical fly
demo is steering above the same frozen low-level gait controller for every
method. A sensory/motor graph requires its own provenance and interfaces;
the descending-only chess graph cannot be relabeled as that circuit. Robot
symmetry and a generic structured controller are necessary competitors.

## What would justify a paper

A possible research question is: **Does biological connectivity make a
two-timescale belief model adapt more efficiently under sensing loss and
hidden physical changes?** Existing papers already cover its individual
ingredients. This reading does not establish novelty, an ICLR-ready result,
or superiority to the conventional alternatives.

Continue only after competent control, a measured topology interaction at
matched cost, and transfer to a second task. Preserve unsuccessful settings
and report any simpler explanation, such as search quality, generic hierarchy,
symmetry or uncertainty handling.
