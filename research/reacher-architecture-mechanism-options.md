# Architecture mechanisms after a trained history control

**Current status:** the [fixed-geometry comparison](reacher-geometry-memory.md)
now establishes a persistent-policy advantage over the trained current-packet
and single-observation controls. The supplied public-kinematic controller still
performs better. The [two-observation learned control](../output/reacher-two-observation-control-v1/README.md)
is implemented and has passed a complete tiny engineering rehearsal, including
all 42 control rows and independent audit. Its scientific comparison has not run. The original mechanism review below remains a proposal, with no
biological or architectural efficacy claim.

Decision note, September 18, 2026. **Proposed, unfrozen and unrun.** This review
uses published papers, official source code and the existing
[learning roadmap](connectome-learning-program.md) and
[robotics reading](connectome-robotics-next-experiments.md). It does not inspect
the running objective study or establish a new architectural contribution.

**First finish the trained current-packet GRU and MLP controls. Then test a
short-history predictor before adding biological structure.** In this task,
history may mainly estimate unobserved velocity. A fixed window of public
packets, issued commands, missingness and ages is a stronger comparator than a
reset intervention alone. Explicitly distinguish a window of physical steps
from the last two valid measurements plus their elapsed time and intervening
commands; they retain different information during a blackout. Give each
predictor the same action-conditioned rollout interface and fixed CEM256.

The existing [HistoryWorldModel](../src/openjev/research/reacher_world_models.py)
uses a 12-slot packet/action window but retains its own angle forecasts through
missing observations. Those forecasts can carry older information: finite input
width does not imply bounded effective history. A genuinely bounded comparator
must rebuild its real-decision input only from recent public packets/actions,
discarding prior forecasts before each decision; candidate rollouts may then
generate their own temporary predictions. Also include a small public-only
kinematic control that estimates angular velocity from the last two valid
measurements and their actual elapsed time, using wrapped angle differences.
It has explicit memory and angular-aliasing limits, but can test whether a
simple physical estimate explains a learned representation's apparent benefit.

The strongest interpretable mechanism candidate is **separated fast state and
slow context**, conditional on qualifying a task that actually needs slow
context. Current independent white actuator noise does not supply persistent
hidden physics. None of the mechanisms below is new by itself.

| Mechanism | Concrete addition beyond a fixed mask | Closest necessary comparator | Evidence that would reject its intended explanation |
|---|---|---|---|
| Fast motion plus slow context | A separately updated context state conditions action transitions. | Dense two-rate GRU, equal-state single-rate GRU, short-history predictor. | Same gain from ordinary history; slow-state erasure has no effect; no benefit under identifiable persistent changes. |
| Episode-local plastic trace | A bounded activity trace changes effective transition weights during real experience. | Fixed adapter, ungated trace, context-gated adapter with comparable dynamic-state size. | Gain survives trace erasure, or arises entirely from extra state/parameters. |
| Stable state-space recurrence | Explicit decaying/oscillating recurrent modes replace the transition GRU. | GRU and two-rate GRU with the same public information and measured cost. | Longer-gap prediction improves but native control and action ranking do not. |

## 1. Fast state and slow context: preferred conditional mechanism

[Clockwork RNN](https://arxiv.org/abs/1402.3511) already separates update rates.
[HiP-RSSM](https://arxiv.org/abs/2206.14697) separates a hidden task parameter
from fast state; [MTS3](https://arxiv.org/html/2310.18534v3) learns interacting
fast/slow beliefs and action-conditioned predictions, including missing
observations. [DALI](https://arxiv.org/abs/2508.20294) conditions world-model
imagination on learned dynamics context. A slow context head therefore needs
these conventional explanations ruled out before a biological interpretation.

Start with a deterministic fast GRU and a small slow state. Update context
only from completed public interactions, using a declared clock or bounded
leak; retain the clock and observation accumulator in checkpoint/state
metadata. Fast transitions consume every issued action. Initially hold the
inferred context fixed throughout each imagined candidate: forecasts do not
provide new evidence about real physics. Slow-state dimension, update interval
and any accumulator storage count toward the deployed-state budget.

Qualify a separate longer task with a hidden actuator gain or damping parameter
that persists long enough to identify, followed by a prespecified change.
Keep the original task as a competence check. Verify identifiability with an
ordinary history estimator and a clearly privileged true-parameter reference.
Otherwise weak excitation or a too-short episode can make every context model
unidentifiable. Compare one-rate versus two-rate updates, fast-only versus
slow-only reset, and a conventional context adapter. Report recovery time after
the change, forgetting and native cost, alongside prediction error.

Biological motivation is narrower than proof: the
[flyvis paper](https://www.nature.com/articles/s41586-024-07939-3) couples measured
connectivity to mechanistic neural dynamics, and its
[official dynamics code](https://github.com/TuragaLab/flyvis/blob/main/flyvis/network/dynamics.py)
includes neuronal time constants. This does not provide measured time constants
for our chosen motor graph. Label unmeasured assignments as learned or
engineered, not biological annotations.

**One falsifiable topology interaction:** let gain be the reduction in native
cost from single-rate to separated-rate dynamics. Test whether
`gain(biological graph) - mean(gain(matched rewires)) > 0` on untouched cases,
across multiple graph draws and fit seeds. Keep signed degree, neuron roles,
role-to-role mixing, interfaces, dynamics and objective matched. Add timescale
shuffles within comparable neuron roles, preserving the rate multiset. If every
graph benefits equally, the finding concerns temporal organization. If the
biological assignment performs like the shuffles, anatomy-specific timescale
placement is unsupported. A topology claim then needs confirmation in a second
embodied task, with a multiplicity/continuation rule frozen beforehand.

## 2. Plastic state: an adaptation comparator, not online RL by default

[Differentiable plasticity](https://arxiv.org/html/1804.02464v3) already combines
fixed weights with an episode-specific Hebbian trace;
[Backpropamine](https://arxiv.org/abs/2002.10585) learns modulation of that trace.
A bounded OpenJev candidate could start with a low-rank transition adapter
`W_eff = W + U H V^T`, where a small matrix `H` holds decaying projected
pre/post-activity products. Outer training learns the fixed parameters; only
`H` changes during an episode. This is a proposed adapter, not a reproduction
of either paper or a biological circuit.

Compare zero trace, ungated trace, learned modulation and a context-gated
adapter at comparable total state and cost. Reset only `H` as an intervention.
Update once after a real observable outcome; any prediction-error gate must
use an outcome that has actually arrived. Freeze adaptation during missing
observations and imagined branches in the first comparison. Reset traces at
episode boundaries; planner candidates get private state. Measure saturation,
adaptation delay and ordinary-task degradation. Extra adaptation computation
and state copies must be paid.

A later edge-local plasticity experiment can preserve a biological mask, but
the low-rank adapter generally does not. Do not present it as topology
preserving. Dense traces cost quadratic state; sparse traces still cost one
dynamic value per plastic edge. Equal neuron count does not match memory.

## 3. State-space recurrence: the strong nonbiological challenger

[S4WM](https://arxiv.org/abs/2307.02064) already studies state-space world-model
backbones, including an S5 instantiation.
[S5](https://arxiv.org/abs/2208.04933) supplies a multi-input state-space layer.
Use a small causal S5-style action transition with the same observation
correction, residual reward formulation and decoder budget as the GRU.
Explicitly constrain the continuous-time eigenvalues' real parts when testing
stable decay; the official implementation makes eigenvalue clipping optional.

Compare learned multiple decay rates with a common-rate control and GRU.
Count complex state as two real values. Check recurrent/sequence equivalence,
chunk boundaries, action alignment and per-case state cloning. A physical
transition advances time once; observation assimilation must not add another
physical tick. No bidirectional sequence processing is allowed. Missing angular
features remain sanitized, and imagined packets never masquerade as real
measurements. Parallel training speed does not establish sequential CEM speed;
measure both. A diagonal stable core also does not prove stability of the
complete nonlinear decoder/controller.

## Implementation and reuse constraints

The `initial / assimilate / advance` interface must preserve any additional
state through branch cloning, batch repeats, resets and checkpoints. Test
candidate independence, poisoned missing inputs and terminal boundaries before
new fits. Hold the qualified objective and CEM fixed. Report training cost,
deployed state, planning work and native interactions; equal parameters do not
mean equal compute.

Official sources checked for this note:

- [MTS3](https://github.com/ALRhub/MTS3) has no root license file or GitHub license
  declaration visible at review; its
  [requirements](https://github.com/ALRhub/MTS3/blob/master/requirements.txt)
  pin an older PyTorch stack. Use the published equations for an independent
  implementation; do not assume permission to vendor the repository.
- [Backpropamine's license](https://github.com/uber-research/backpropamine/blob/master/LICENSE)
  is non-commercial for its core experiments. Do not copy that code into the
  MIT project under an unrestricted license. An attributed independent
  implementation of the paper mechanism is the proposed route.
- [S5](https://github.com/lindermanlab/S5/blob/main/LICENSE) and
  [flyvis](https://github.com/TuragaLab/flyvis/blob/main/license) publish MIT
  licenses. Retain notices when reusing source and verify data/checkpoint terms
  separately. S5's official layer is JAX/Flax, so a PyTorch port needs arithmetic
  parity tests and actual deployment timing. Pin code revisions before reuse.

No new model, dataset, protocol, fit or score was produced for this note.
