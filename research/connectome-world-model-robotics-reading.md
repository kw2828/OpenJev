# Connectomes, recurrent world models and robotics: what to test next

Literature checked September 18, 2026. This is a research recommendation, not a
new performance result or a frozen training protocol. Paper results below are
the authors' reports; we have not reproduced them.

Follow-up: the [completed Pendulum qualification](robotics-pendulum-qualification.md)
passed all six checks. Short-history control approaches the known-state mean,
so we will not expand that clean diagnostic into a connectome architecture
study. The report includes per-gain failures and the weak switch challenge.

**Recommended direction: test whether biological wiring and elapsed-time-aware
latent dynamics improve a persistent belief model under hidden dynamics and missing
sensors.** Start with inexpensive simulated control, then transfer the same
mechanism to an embodied task. The possible contribution is a measured
interaction between topology, dynamics and adaptation, supported by controlled
interventions. Adding JEPA, recurrence or a connectome is already established.

## What this changes for OpenJev

The [completed chess comparison](../docs/chess-connectome.md) failed its
biological-topology criterion. The separate
[spatial-mapping study](chess-connectome-mapping-study.md) tests an interface
hypothesis; it does not test persistent memory across decisions.

The [completed recurrent prediction pilot](../docs/recurrent-world-model-study.md)
also failed its continuation criteria. Resetting memory did not change its
recorded outcomes, and policies mostly failed to acquire the cue. This makes
information availability and memory use prerequisites for the next comparison.

Local BLACKOUT prototypes already contain learned node descriptors, retention
gates, action-conditioned imagination and readout interventions. The existing
planning proposal already calls for latent prediction, MPC and targeted
repair. These are not new ideas from this review. The useful additions are:

- distinguish persistent observation filtering from imagined future rollout;
- separate latent integration time constants from topology and generic gating;
- test hidden-parameter adaptation against system identification and strong
  history-based controls;
- use a second, independently specified robotics task before claiming transfer.

## Papers that change the experiment

The reading covered methods and evaluation sections where noted. Availability
means an author or official project link was checked, not that the software
was installed or reproduced. Version dates matter, especially for FlyGM.

### Connectome mechanisms and embodiment

| Paper | Mechanism and evidence | Consequence for our experiment |
|---|---|---|
| [FlyGM](https://arxiv.org/html/2602.17997v3), Jin et al., 2026. **v3: June 14**. Methods, §4.1, Appendices G/H. | Fixed connectome propagation, learned cell descriptors, sensory/motor interfaces, expert imitation followed by PPO. Appendix G adds unweighted biology and descriptor ablations after full IL+PPO. Its unweighted biological graph has lower reported angle error than the unit-weight rewire in §4.1/Table 1. Standard GNN comparisons use different widths/depths under GPU-memory constraints. | The latest version addresses part of the earlier weight confound. Compare topology with identical dynamics, interfaces and normalization. Our existing learned gates/descriptors are related ingredients, not novelty. [Project](https://lnsgroup.cc/research/FlyGM/) and [older project](https://sites.google.com/view/flygm) advertise forthcoming/review code; review-code contents were not verified. |
| [Connectome-constrained networks predict neural activity across the fly visual system](https://www.nature.com/articles/s41586-024-07939-3), Lappalainen et al., 2024. Methods and official implementation. | Leaky dynamics, cell-type sharing and retinotopic organization trained for optic flow, assessed against physiological responses. This concerns neural prediction rather than broad control superiority. | A visual-motion front end has a closer sensory match than an arbitrary board-to-neuron map. Match preprocessing and task objectives across biological, rewired and conventional retinotopic models. [Official flyvis code/models](https://github.com/TuragaLab/flyvis). |
| [Connectome-constrained modeling identifies neurons and synapses that sustain spontaneous activity in Drosophila](https://www.biorxiv.org/content/10.64898/2026.08.21.745055v1.full), Li et al., **posted August 25, 2026**. Full methods. | Fits synaptic magnitudes and per-neuron time constants with fixed topology/polarity and an anatomical readout. Evaluates spontaneous calcium dynamics, not robot control. Training has no separate validation segment; processed data, weights and figure code are promised upon acceptance. | Test time constants as a separate factor. Learned time constants are not measured biological constants. Do not transfer neural-prediction or model-lesion claims directly to robotics. |
| [The digital sphinx: Can a worm brain control a fly body?](https://faculty.washington.edu/tuthill/docs/TheSphinx_2026.pdf), 2026, bioRxiv DOI **10.64898/2026.03.20.713233**, posted March 24. Full paper. | Fixed worm circuitry and a random sensory projection feed a trained motor decoder that produces plausible simulated fly walking. A mismatched nervous system can support the behavior. The suggestion that random recurrence might suffice is not a reported matched random-network result. | Keep decoder-capacity, random/frozen-core and no-message controls. Plausible movement alone cannot identify a biological mechanism. [Code](https://github.com/Brunton-Lab/DigitalSphinx2026). |
| [NeuroMechFly v2](https://www.nature.com/articles/s41592-024-02497-y), 2024. Methods and [paper PDF](https://gizemozd.github.io/assets/pdf/2024_neuromechflyv2.pdf). | Embodied simulation combines gait generation, sensory corrections and higher-level navigation. Includes terrain, path integration and multimodal behavior; it is a framework with several controllers. | A practical later demo: OpenJev selects steering/skills above one fixed gait controller. Compare decision modules in the same hierarchy. [FlyGym](https://github.com/NeLy-EPFL/flygym/) and [paper experiments](https://github.com/NeLy-EPFL/nmf2-paper) are public. |

### Memory, hidden physics and strong controls

| Paper | Mechanism and evidence | Consequence for our experiment |
|---|---|---|
| [Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://arxiv.org/html/2110.05038v3), Ni et al., 2021/ICML 2022. Methods. | Separate recurrent actor/critic networks with off-policy RL are strong across partial-observation tasks. Configuration was selected per benchmark. | Budget tuning for GRU and temporal-CNN baselines. An untuned weak GRU is insufficient. [Official code](https://github.com/twni2016/pomdp-baselines); some environments have old dependencies. |
| [VariBAD](https://arxiv.org/html/1910.08348), Zintgraf et al., 2019/ICLR 2020. Methods and experiment details. | A recurrent encoder infers a distribution over hidden task embeddings for a belief-conditioned policy. Published MuJoCo experiments use reward reconstruction only, including randomized-dynamics Walker. Task parameters are fixed within a task. | Useful Bayesian baseline, but posterior conditioning and physical-change detection require separate tests. Compare posterior mean versus distribution, and ordinary recurrence. [Official code](https://github.com/lmzintgraf/varibad). |
| [RMA: Rapid Motor Adaptation](https://arxiv.org/html/2107.04034v1), Kumar et al., 2021. Methods and evaluation. | A privileged policy is paired with a history-based adapter trained to estimate its environment latent. The adapter is a temporal CNN, not an RNN. Includes real quadruped tests. | Freeze the policy and targets when comparing adapters. Privileged training information must be shared or reported separately. The [official project](https://ashish-kmr.github.io/rma-legged-robots/) links a [modified derivative implementation](https://github.com/antonilo/rl_locomotion), not a verified exact original release. |
| [Recurrent Kalman Networks](https://arxiv.org/pdf/1905.07357), Becker et al., 2019. Methods and robot-prediction experiments. | Structured latent uncertainty and factorized Kalman-style updates handle noisy/missing observations; action-conditioned robot prediction is included. These are prediction/estimation results, not closed-loop RL. | Compare uncertainty and filtering against an explicit probabilistic estimator, not only a deterministic GRU. [Original code](https://github.com/LCAS/RKN), [lab PyTorch version](https://github.com/ALRhub/rkn_share); original robot data availability unverified. |
| [PlaNet](https://arxiv.org/html/1811.04551v5), 2018/ICML 2019; [DreamerV3](https://arxiv.org/html/2301.04104v2), 2023/2024, and [2025 Nature version](https://www.nature.com/articles/s41586-025-08744-2). Methods. | Recurrent state-space models combine deterministic memory, stochastic latents and action-conditioned dynamics. PlaNet plans; Dreamer trains behavior through imagined trajectories. The later Dreamer implementation uses an eight-block GRU. | Include an RSSM and structured block-GRU control. A small component swap is not a complete Dreamer reproduction. [Dreamer code](https://github.com/danijar/dreamerv3); [PlaNet code](https://github.com/google-research/planet) is archived. |

### Compact predictive models and planning

| Paper | Mechanism and evidence | Consequence for our experiment |
|---|---|---|
| [TD-MPC2](https://arxiv.org/html/2310.16828), Hansen et al., 2023/ICLR 2024. §3. | Joint latent, reward and value learning supports short-horizon planning with a terminal value. Evaluated across 104 continuous-control tasks. | A strong control-oriented comparator. The ordinary state encoder does not itself supply a persistent belief filter under hidden dynamics. Keep reward access and planner budgets explicit. [Official release](https://tdmpc2.com). |
| [DINO-WM](https://arxiv.org/html/2411.04983v2), Zhou et al., 2024/2025. Methods and baseline settings. | Frozen spatial DINOv2 features, action-conditioned prediction and CEM planning from offline trajectories. Its Dreamer/TD-MPC comparisons adapt them to reward-free offline data; those are not their canonical online-RL settings. | Reuse frozen features only when pixels become necessary. Match pretraining access. [Official code](https://github.com/gaoyuezhou/dino_wm). |
| [V-JEPA 2](https://arxiv.org/html/2506.09985v1), Assran et al., 2025. §3 and robot limitations. | Frozen video encoder plus action-conditioned prediction, teacher forcing and short autoregressive rollout loss. Robot post-training uses under 62 hours after internet-scale video pretraining. Planning depends on camera placement. | Borrow the rollout-loss comparison, not the claim that the whole model learns from 62 hours. Do not treat its large predictor as the cheapest starting point. [Official code](https://github.com/facebookresearch/vjepa2). |
| [LeWorldModel](https://arxiv.org/html/2603.19312v3), Maes et al., 2026. **v3: June 3**. Methods, controls and limitations. | Approximately 15M parameters; joint pixel encoder/dynamics training with prediction and SIGReg losses, then latent CEM planning. Reports a few hours on one GPU. Short horizons and low-diversity data remain limitations. | A useful compact visual baseline. SIGReg is an established option, not our proposed invention or a guarantee of useful control. [Official code](https://github.com/lucas-maes/le-wm). |
| [Sensorimotor World Models](https://arxiv.org/html/2606.20104), Ivashkov et al., 2026. Methods and planning comparison. | Forward latent prediction plus inverse action prediction; controlled examples and visual planning compare against the same backbone with SIGReg. | Action recovery may help representation learning, but hidden disturbances relevant to reward may be uncontrollable. Preserve observable-consequence/cost tests rather than discarding such variables by design. |
| [Toward Physically Grounded JEPA World Models](https://arxiv.org/html/2609.03565), Liu et al., **September 3, 2026**. Full short paper. | Adds inverse dynamics and training-only physical-state alignment. Three seeds; baseline means are taken from LeWorldModel. Code/configurations promised upon acceptance. | This is direct prior art for adding physical targets. Full simulator state is additional supervision; do not quietly give it to only our method. Latent geometry/probe scores cannot replace control outcomes. |
| [One Future, Every Robot](https://arxiv.org/html/2607.28443), Gazzaev et al., 2026. **v3: August 6**. Methods and limitations. | Local GRUs exchange bounded messages and predict a shared latent future. Includes simulator control under communication faults; counterfactual branch labels are used in training. It does not show improved selected-plan regret in the separate value test. | Close prior art for graph recurrence plus JEPA. Swarm communication is a later direction, not a new generic combination. Hardware transfer remains open; a code release was not verified. |

Two additional checks constrain interpretation. [Topological Sensitivity in
Connectome-Constrained Neural Networks](https://arxiv.org/abs/2604.04033) motivates
shared initialization and degree-preserving nulls, as already recorded in our
earlier review. [Multistep Inverse Is Not All You Need](https://arxiv.org/abs/2403.11940)
gives counterexamples to a particular inverse-only representation method and
adds forward prediction under explicit assumptions. The latter was checked at
abstract/publication level here; it is not a theorem against every inverse
model or a guarantee for our proposed POMDP.

## Ranked experiments

### 1. Qualify a persistent belief model under hidden dynamics

Begin with [Gymnasium Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/):
mask angular velocity and add an explicitly documented hidden actuator-gain
wrapper. Keep the native task alongside the modified one. Start with stationary
hidden gain; define abrupt changes and observation gaps as separate shifts.
This is a diagnostic environment, not a standard Pendulum leaderboard result
or sufficient evidence for an ICLR paper.

First test whether available histories identify the relevant hidden state.
Compare current observation, frame stacking/finite-difference velocity,
temporal CNN, tuned GRU and a simple fitted system-identification filter. Use
common exploratory trajectories with enough torque variation. A supplied-
physics controller is a privileged reference, not a learned competitor.

Distinct gains must produce distinguishable observed trajectories under those
actions; zero torque and clipping can remove the signal. Audit rewards and
auxiliary fields for leakage of masked velocity/gain. If short-history
estimation already saturates control, stop expanding this diagnostic instead
of increasing difficulty after seeing scored results.

The candidate should have two explicit operations: assimilate each real
observation and previous action into persistent state; then copy that state
and roll candidate actions forward without future observations. Reset only at
episode boundaries. Prediction quality must beat copying/persistence, and
planning must improve realized return at the same search budget. Reset and
action-history interventions test use of memory rather than mere capacity.
Report observation-filtering error separately from open-loop rollout error.

Train predictors first on one frozen transition dataset and use common
observation interfaces and planner candidates. This isolates model learning;
an online RL comparison requires a separate interaction-budget protocol.
Track whether planned actions leave the training data's coverage: a planner
can exploit prediction errors even when average held-out prediction improves.
Measure return, multistep prediction, change-recovery cost, training cost and
full decision latency. Learned models must earn an advantage over the strongest
simple control before expanding to a large topology experiment.

### 2. Separate biological wiring from memory timescales

After qualification, cross biological and several independently sampled
degree-preserving rewires with fixed leak, learned global time constant and
learned per-node time constants. A prospective update is
`alpha_i(dt) = 1 - exp(-dt / tau_i)` with bounded positive `tau_i`.
It represents actual elapsed time; extra internal computation must not advance
the simulated physical clock. The equation is an established leaky-dynamics
parameterization, not a novel algorithm or an identified physical parameter.
At fixed `dt`, constant retention is equivalent to such a time constant:
`r_i = exp(-dt / tau_i)`. The new hypothesis concerns transfer across elapsed
times or constrained sharing. Give every model the same time information and
include a `dt`-conditioned retention-gate control.

Also compare the existing learned retention gate. If reliable cell-type
annotations are available, add shared type-level constants and a shuffled-type
control; otherwise call them learned node constants, not anatomical cell types.
Preserve the same signed degree, role mixing, initial weight/gain distribution,
interfaces and update schedule within each topology contrast. Include generic
sparse and block-GRU controls. Parameter count, hidden-state size and measured
compute are separate matching axes; report residual differences and cost
frontiers rather than asserting that all can be matched exactly.
Check activity scale and recurrent stability; matching degrees alone does not
match weighted strength or mixing dynamics.

The key question is whether the gain from dynamics changes is greater for
biology than for rewires, and survives varied sampling intervals and missing
observations. Predefine the interaction as the biological learned-minus-fixed
effect minus the mean rewired learned-minus-fixed effect on the primary metric.
Independent training seeds and graph draws are distinct replication units;
episode counts do not substitute for either. A significant biological effect
and nonsignificant rewire effect alone do not establish an interaction.
An equal gain everywhere supports a dynamics improvement. It
does not support a biological-wiring advantage. Strong controls beating the
candidate end that architecture claim in this setting.

### 3. Transfer to embodied control, then test targeted repair

Use [DeepMind Control Reacher](https://github.com/google-deepmind/dm_control) as
an independently specified second task. Only after that, test OpenJev steering
decisions above a frozen FlyGym gait controller. Every architecture gets the
same low-level controller, sensory preprocessing and small decoder. Changing
from our descending-only chess graph to sensory/motor circuitry requires new
graph provenance and a new experiment; do not relabel existing nodes.

For mechanistic work, select candidate memory subsets using training or
calibration data. Patch states between paired trajectories with controlled
physical differences; compare random patches matched for size, degree and
activity, and check off-manifold effects. These are offline diagnostics;
patched states from a paired trajectory may be unavailable during deployment.
State patching tests a model's causal dependence, not a unique explanation of
its learning advantage. Online parameter
adaptation is a different experiment: compare no adaptation, global updates,
random subsets and gradient-selected subsets with equal data and measured
cost, including circuit selection. Test the same selection rule on rewired
models and include an ordinary observation-update control. Deployed repair
must use only information available at that moment and improve actual control
after a new fault, not only a probe.

## Continuation and implementation boundary

Freeze data splits, tuning allocations, metrics, independent training seeds,
graph draws and a numerical minimum useful effect before scored fits. Use
fresh confirmation episodes and training seeds after development. A topology
claim requires beating matched rewires and the strongest conventional control;
a planning claim requires useful imagined dynamics; a Bayesian claim requires
an uncertainty ablation. Positive diagnostics alone satisfy none of these.

The review itself changes no frozen chess criteria. Subsequent work completed
the [Pendulum qualification](robotics-pendulum-qualification.md), which found
that a short history largely sufficed, and the [nine-fit Reacher pilot](reacher-world-model-pilot.md).
The latter failed its continuation rule: GRU and RSSM world models did not
improve over zero commands, while supplied-physics planning did. It provides
no biological-wiring result. The immediate next test separates a known motor
penalty from learned task reward and gives both matched GRU arms a longer
fixed training budget. Useful learned control must precede the topology test.
Visual pretraining, whole-fly training and a large combined architecture remain
later steps.
