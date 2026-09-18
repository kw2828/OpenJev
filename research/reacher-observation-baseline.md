# A trained control for the memory claim

**Engineering component only, September 18, 2026. No research fitting or evaluation.**
The [running objective comparison](reacher-objective-ablation.md) includes a
history-reset intervention. A reset penalty may reflect either useful history
or an unfamiliar hidden state. These new components enable a later comparison
with a model trained to make every real decision from its current packet.

The preferred `CurrentObservationGRUWorldModel` inherits the existing residual
GRU's constructor and action transitions unchanged. Before every real packet,
including a missing measurement, it creates a zero initial state and assimilates
only that packet. Parameter names, order, count and seeded initialization match
the original model. Training applies this rule too; it is not a reset imposed
only after fitting.

During planning, repeated imagined action transitions may retain temporary
state. A new real packet discards that state completely. The distinction is
between remembering past observations and simulating a proposed future, not
between one-step and multistep planning.

An optional `FeedForwardObservationWorldModel` provides a conventional packet
and action MLP. Its only imagined state is its predicted packet. It also
discards that prediction at every real assimilation. Both models use the same
public inputs, total native reward targets and known actuator-cost residual.

| Component | Parameters | Dense affine MACs per assimilation | Per imagined advance |
|---|---:|---:|---:|
| Current-observation GRU, width 64 | 36,805 | 13,824 | 22,080 |
| Feedforward model, width 107 | 36,599 | 0 | 36,166 |

These operation counts exclude activations, GRU gate arithmetic, validation,
copies, reward arithmetic, backward passes and optimizer work. The MLP has
0.56% fewer parameters but more dense arithmetic per imagined transition.
Neither the counts nor matching parameters establishes a speed comparison.

## What is verified

[34 synthetic tests](../tests/test_reacher_observation_baseline.py) cover exact
GRU initialization and transition parity, erased history and gradient paths,
preserved current information, ignored missing-angle placeholders, independent
planner branches, unchanged sequence-loss targets/gradients and operation counts
checked against actual layer shapes. The
[receipt](../evidence/reacher-observation-baseline-engineering-v1/receipt.json)
binds the component and tests. The 42 files in the running study remain unchanged.

The current packet omits velocity and, during blackouts, angles. It is not a
fully observed physical state. Beating this comparator alone would therefore
not establish a new memory architecture. A genuinely bounded public-history
model and a last-two-measurements velocity estimate are stronger follow-ups;
the [architecture note](reacher-architecture-mechanism-options.md) explains them.

## Before a scientific comparison

Use a separate frozen protocol after the current study is audited. Bind the
actual constructor and assimilation rule, paired initial tensors, training
data/order, objective, update count, planner and fresh evaluation cases.
The existing frozen trainer hardcodes the original GRU class. Compatible weight
names do not authorize substituting either new class into that trainer or its
checkpoint restoration. These components are not part of the running study.

Report every fit, native control and all training/deployment costs. Keep
observed history and temporary imagined state distinct in checkpoint metadata,
branch cloning, rollout accounting and any future biological comparator.
# Strict three-packet comparator added

The separate [bounded-history component](../src/openjev/research/reacher_bounded_history.py)
rebuilds the same 36,805-parameter GRU from only the latest three real public
packets and two issued commands, with explicit startup masks. Imagined rollouts
never enter its raw history. Real assimilation accepts episode start or exactly
one selected-action advance from the previous root; it rejects a multi-step
candidate terminal state.

[63 synthetic checks](../evidence/reacher-bounded-history-engineering-v1/receipt.json)
passed across this component and the current-observation controls. Matching
final windows with different earlier histories yields identical outputs and
zero gradients to discarded inputs. All frozen objective-study sources remain
unchanged. No comparator has been fitted or scored.

This is a three-physical-packet window, not the last three valid measurements.
It loses all observed angles during a sufficiently long blackout. It should
therefore be accompanied by the public-only last-two-valid-measurements
kinematic control described in the architecture note. Rebuilding costs three
observation updates and two full transitions per real assimilation, in addition
to ordinary imagined planning; equal parameters do not match computation.
Class-bound deployment checkpoints are implemented, but they are not training
resume checkpoints or permission to substitute models into a frozen run.

## Anchor-only training component

The separate [memory trainer](../src/openjev/research/reacher_memory_training.py)
now supports persistent GRU, current-packet GRU, strict three-packet GRU and
current-packet MLP through explicit classes. The three GRUs share an identical
initial tensor payload; the MLP has its own named initialization. All arms can
consume the same saved minibatch permutations and unchanged sequence loss,
Adam configuration and gradient clipping.

Full checkpoints bind the actual class, public training data, initialization,
orders, optimizer state, source/runtime metadata and exact next batch. Restore
requires external expected bindings and rejects failed partial trainers. This
does not authorize resuming any frozen experiment.

[114 synthetic checks](../evidence/reacher-memory-training-engineering-v1/receipt.json)
passed, including 51 trainer checks and the 63 model checks above. Two-update
parity and uninterrupted-versus-restored training were checked for all four
classes, including an epoch boundary and partial minibatch. These are
engineering checks, not control results. A scientific protocol, deployment
runner, independent audit and cost checks remain to be completed before a new
comparison is launched.
